c
c
c
c
c
      implicit real *8 (a-h,o-z)
      real *8 epsvals(5),deltas(20),pps(20,5),totinfo(10,20,5)
      real *8 rpars(100)
      integer nsrcs(20)
      integer dim
c	  
      call prini(6,13)

      pi = 4*atan(1.0d0)
      
      epsvals(1) = 1d-3
      epsvals(2) = 1d-6
      epsvals(3) = 1d-9
      epsvals(4) = 1d-12
c
      do i=1,10
         deltas(i)=10.0d0**(-i+2)
      enddo  

      do i=1,8
         nsrcs(i)=i*10**6
      enddo
c     nd - number of different densities
      nd=1
c     dim - dimension of the problem
      dim=2
c     ikernel - 0: Yukawa kernel; 1: Laplace kernel; 2: square root Laplace kernel
      ikernel = 1
c     rlambda - the parameter in the Yukawa kernel or the exponent of the
c     power function kernel
      rlambda=6.0d0
      rpars(1)=rlambda
      
c     iperiod -> 0: free space; 1: periodic in all dimensions
      iperiod=0
c     ntarg: number of extra targets
      ntarg=0
c     nsrc: number of source points

c     whether to have monopole sources -> 1: yes; 0: no
      ifcharge=0
c     whether to have dipole sources -> 1: yes; 0: no
      ifdipole=1
c     evaluation flag for sources -> 1: pot; 2: pot+grad; 3: pot+grad+hess
      ifpgh=1
c     evaluation flag for targets -> 1: pot; 2: pot+grad; 3: pot+grad+hess
      ifpghtarg=1
c
c     
c
c     test all parameters
c
      iw = 70
      iw2 = 80

      ifuniform=1

      if (ikernel.eq.0.and.dim.eq.2.and.ifuniform.eq.1) then
         open(iw, file='y2dptsu.txt', position='append')
      elseif (ikernel.eq.0.and.dim.eq.2.and.ifuniform.eq.0) then
         open(iw, file='y2dptsa.txt', position='append')
      elseif (ikernel.eq.0.and.dim.eq.3.and.ifuniform.eq.1) then
         open(iw, file='y3dptsu.txt', position='append')
      elseif (ikernel.eq.0.and.dim.eq.3.and.ifuniform.eq.0) then
         open(iw, file='y3dptsa.txt', position='append')
      elseif (ikernel.eq.1.and.dim.eq.2.and.ifuniform.eq.1) then
         open(iw, file='l2dptsu.txt', position='append')
      elseif (ikernel.eq.1.and.dim.eq.2.and.ifuniform.eq.0) then
         open(iw, file='l2dptsa.txt', position='append')
      elseif (ikernel.eq.1.and.dim.eq.3.and.ifuniform.eq.1) then
         open(iw, file='l3dptsu.txt', position='append')
      elseif (ikernel.eq.1.and.dim.eq.3.and.ifuniform.eq.0) then
         open(iw, file='l3dptsa.txt', position='append')
      elseif (ikernel.eq.2.and.dim.eq.2.and.ifuniform.eq.1) then
         open(iw, file='sl2dptsu.txt', position='append')
      elseif (ikernel.eq.2.and.dim.eq.2.and.ifuniform.eq.0) then
         open(iw, file='sl2dptsa.txt', position='append')
      elseif (ikernel.eq.2.and.dim.eq.3.and.ifuniform.eq.1) then
         open(iw, file='sl3dptsu.txt', position='append')
      elseif (ikernel.eq.2.and.dim.eq.3.and.ifuniform.eq.0) then
         open(iw, file='sl3dptsa.txt', position='append')
      endif

cccc  open(iw2, file='timing.txt', position='append')
      
      do i=2,2
         eps = epsvals(i)
cccc         do j=4,4
         do j=1,1
            nsrc=nsrcs(j)
cccc            nsrc=3
            nsrc=1*10**5
            ntarg=nsrc
            if (ifpghtarg.eq.0) ntarg=1
c
            call testpdmk(nd,dim,eps,ikernel,rpars,
     1          iperiod,nsrc,ntarg,ifcharge,
     1          ifdipole,ifpgh,ifpghtarg,ifuniform,totinfo(1,j,i),rerr)
 1400       FORMAT(6(2X,E11.5))
            write(iw,1400) (totinfo(k,j,i),k=1,6)
c$$$ 4800       format(2x,D8.2,1x,'&',2x,D8.2,1x,'\\')
c$$$            write(iw,4800) eps,  rerr
c     
         enddo  
      enddo

c$$$      do j=1,8
c$$$         write(iw2,*) pps(j,1), pps(j,2), pps(j,3), pps(j,4)
c$$$      enddo
      
      close(iw)
cccc      close(iw2)

      end
c
c
c
c
      subroutine testpdmk(nd,dim,eps,ikernel,rpars,iperiod,nsrc,ntarg,
     1    ifcharge,ifdipole,ifpgh,ifpghtarg,ifuniform,totinfo,rerr)
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 rpars(*)
      real *8, allocatable :: sources(:,:),targ(:,:)
      real *8, allocatable :: sim(:,:)
      real *8, allocatable :: rnormal(:,:)
      real *8, allocatable :: charges(:,:),dipstr(:,:)
      real *8, allocatable :: charge1(:,:)
      real *8, allocatable :: pot(:,:),grad(:,:,:),hess(:,:,:)
      real *8, allocatable :: pottarg(:,:),gradtarg(:,:,:),
     1    hesstarg(:,:,:)
      real *8, allocatable :: potex(:,:),gradex(:,:,:),hessex(:,:,:)
      real *8, allocatable :: pottargex(:,:),gradtargex(:,:,:),
     1                             hesstargex(:,:,:)

      real *8 shifts(dim),thresh
      real *8 xs(100),timeinfo(20),totinfo(20)
      complex *16 ima
      data ima/(0.0d0,1.0d0)/

      call prini(6,13)

      done = 1
      pi = atan(done)*4

cccc      call prinf(' nsrc = *',nsrc,1)
c
      allocate(sources(dim,nsrc),charges(nd,nsrc),dipstr(nd,nsrc))
      allocate(rnormal(dim,nsrc))
      allocate(targ(dim,ntarg))
      nhess=dim*(dim+1)/2
      allocate(pot(nd,nsrc),grad(nd,dim,nsrc),hess(nd,nhess,nsrc))
      allocate(pottarg(nd,ntarg),gradtarg(nd,dim,ntarg))
      allocate(hesstarg(nd,nhess,ntarg))

      rin = 0.45d0
      rwig = 0.12d0
      rwig = 0
      nwig = 6
      
      do i=1,nsrc
         if (ifuniform.eq.0) then
c        nonuniform source distribution
            theta = hkrand(0)*pi
            rr=rin+rwig*cos(nwig*theta)
            ct=cos(theta)
            st=sin(theta)

            phi = hkrand(0)*2*pi
            cp=cos(phi)
            sp=sin(phi)

            if (dim.eq.3) then
               sources(1,i) = rr*st*cp+0.5d0
               sources(2,i) = rr*st*sp+0.5d0
               sources(3,i) = rr*ct+0.5d0
            elseif (dim.eq.2) then
               sources(1,i) = rr*cp+0.5d0
               sources(2,i) = rr*sp+0.5d0
            elseif (dim.eq.1) then
               sources(1,i) = (cos(i*pi/(nsrc+1))+1)/2
            endif
         else
c        uniform source distribution
            do k=1,dim
               sources(k,i) = hkrand(0)
            enddo
         endif

         do k=1,dim
            rnormal(k,i) = hkrand(0)-0.5d0
         enddo
         do ind = 1,nd
            charges(ind,i) = hkrand(0)-0.5d0 
            dipstr(ind,i) = hkrand(0)-0.5d0
         enddo
      enddo

      do ind=1,nd
         dipstr(ind,1) = 0.0d0 
         dipstr(ind,2) = 0.0d0 
         charges(ind,1) = 0.0d0 
         charges(ind,2) = 0.0d0 
         charges(ind,3) = 1.0d0
      enddo

cccc      itype=0
cccc      norder=16
cccc      call chebexps(itype,norder,xs,u,v,ws)
      do k=1,dim
         sources(k,1) = 0.0d0
      enddo

      do k=1,dim
         sources(k,2) = 1.0d0
      enddo

      do k=1,dim
         sources(k,3) = 0.05d0
cccc         sources(k,3) = (xs(1)+1)/2
      enddo
 
      do i=1,ntarg
         do k=1,dim
            targ(k,i) = hkrand(0)
         enddo
      enddo

      ntest=1000
      nts = min(ntest,nsrc)
      ntt = min(ntest,ntarg)

cccc      call prinf('ntt=*',ntt,1)
cccc      call prin2('targ=*',targ,dim*ntt)
      
      allocate(potex(nd,nts),gradex(nd,dim,nts),hessex(nd,nhess,nts))
      allocate(pottargex(nd,ntt),gradtargex(nd,dim,ntt))
      allocate(hesstargex(nd,nhess,ntt))
c
      call dzero(potex,nts*nd)
      call dzero(gradex,dim*nts*nd)
      call dzero(hessex,nhess*nts*nd)
      
      call dzero(pottargex,ntt*nd)
      call dzero(gradtargex,dim*ntt*nd)
      call dzero(hesstargex,nhess*ntt*nd)
      
c      call prinf('ifcharge is *',ifcharge,1)
c      call prinf('ifdipole is *',ifdipole,1)
c      call prinf('ifpgh is *',ifpgh,1)
c      call prinf('ifpghtarg is *',ifpghtarg,1)

      
      call cpu_time(t1)
C$    t1 = omp_get_wtime()      
      call pdmk(nd,dim,eps,ikernel,rpars,
     1    iperiod,nsrc,sources,
     2    ifcharge,charges,ifdipole,rnormal,dipstr,
     2    ifpgh,pot,grad,hess,ntarg,targ,
     3    ifpghtarg,pottarg,gradtarg,hesstarg,timeinfo)
      call cpu_time(t2)
C$    t2=omp_get_wtime()

      pps=(nsrc*ifpgh+ntarg*ifpghtarg+0.0d0)/(t2-t1)
      call prin2('time in pdmk=*',t2-t1,1)
      totinfo(2)=nsrc*1.0d0
      call prin2('points per sec=*',pps,1)

      nprint=min(20,nts)
cccc      nprint=nts
      call prin2('pot=*',pot,nd*nprint)
cccc      call prin2('gradtarg=*',gradtarg,nd*dim*ntt)
cccc      call prin2('hesstarg=*',hesstarg,nd*6*ntt)

      thresh=1.0d-16
      call cpu_time(t1)
C$    t1 = omp_get_wtime()      
      call kernel_direct(nd,dim,ikernel,rpars,
     1    thresh,1,nsrc,sources,
     1    ifcharge,charges,ifdipole,rnormal,dipstr,
     2    1,nts,sources,ifpgh,potex,gradex,hessex)
      call prin2('potex=*',potex,nd*nprint)
      call cpu_time(t2)
C$    t2 = omp_get_wtime()      

      call kernel_direct(nd,dim,ikernel,rpars,
     1    thresh,1,nsrc,sources,
     1    ifcharge,charges,ifdipole,rnormal,dipstr,
     2    1,ntt,targ,ifpghtarg,pottargex,gradtargex,hesstargex)
      if (ifpghtarg.gt.0) then
         nprint=min(20,ntt)
         call prin2('pottarg=*',pottarg,nd*nprint)
         call prin2('pottargex=*',pottargex,nd*nprint)
      endif
      
      if (ifpgh .gt. 0)
     1    call derr(potex,pot,nts*nd,errps,pnorm,errpa)
      if (ifpgh .gt. 1)
     1    call derr(gradex,grad,dim*nts*nd,errgs,gnorm,errga)
      if (ifpgh .gt. 2)
     1    call derr(hessex,hess,nhess*nts*nd,errhs,hnorm,errha)

      if (ifpghtarg .gt. 0)
     1    call derr(pottargex,pottarg,ntt*nd,errpt,tmp1,tmp2)
      if (ifpghtarg .gt. 1)
     1    call derr(gradtargex,gradtarg,dim*ntt*nd,errgt,tmp1,tmp2)
      if (ifpghtarg .gt. 2) 
     1    call derr(hesstargex,hesstarg,nhess*ntt*nd,errht,
     2    tmp1,tmp2)

      call errprint(errps,errgs,errhs,errpt,errgt,errht,ifpgh,ifpghtarg)

      if (iperiod.eq.1) then
         rerr=errpt/pnorm
         if (rerr.gt.eps*10) print *, pnorm,errpa,gnorm,errga
      else
         rerr=errps
      endif
      totinfo(1)=rerr
      
      return
      end
c
c
c
c
      subroutine dzero(vec,n)
      implicit real *8 (a-h,o-z)
      real *8 vec(*)

      do i=1,n
         vec(i) = 0
      enddo

      return
      end
c
c
c
c
      subroutine derr(vec1,vec2,n,relerr,rnorm1,abserr)
      implicit real *8 (a-h,o-z)
      real *8 vec1(*),vec2(*)

      ra = 0
      erra = 0
      do i=1,n
         ra = ra + vec1(i)**2
         erra = erra + (vec1(i)-vec2(i))**2
      enddo

      rnorm1=sqrt(ra)
      abserr=sqrt(erra)
      relerr=abserr/rnorm1

      return
      end
c
c
c
c
      subroutine errprint(errps,errgs,errhs,errpt,errgt,errht,
     1    ifpgh,ifpghtarg)
      implicit real *8 (a-h,o-z)
 1100 format(3(2x,e11.5))


      write(6,*) 'ifpgh is ', ifpgh
      write(6,*) 'ifpghtarg is ', ifpghtarg
      if (ifpgh .gt. 0) write(6,*) 'error in sources'
      if (ifpgh .gt. 0) call prin2('pot err =*', errps,1)
      if (ifpgh .gt. 1) call prin2('grad err=*', errgs,1)
      if (ifpgh .gt. 2) call prin2('hess err=*', errhs,1)
      write(6,*) 
      if (ifpghtarg .gt. 0) write(6,* ) 'error in targets'
      if (ifpghtarg .gt. 0) call prin2('pot err =*', errpt,1)
      if (ifpghtarg .gt. 1) call prin2('grad err=*', errgt,1)
      if (ifpghtarg .gt. 2) call prin2('hess err=*', errht,1)
      write(6,*)
      write(6,*)'==================='

      return
      end
c
c
c
c
      subroutine print_tree2d_matlab(ndim,itree,ltree,nboxes,centers,
     1   boxsize,nlevels,iptr,ns,src,nt,targ,fname1,fname2,fname3)
c
c        this subroutine writes the tree info to a file
c
c        input arguments:
c          itree - integer (ltree)
c             packed array containing tree info
c          ltree - integer
c            length of itree
c          nboxes - integer
c             number of boxes
c          centers - real *8 (2,nboxes)
c             xy coordinates of box centers in tree hierarchy
c          boxsize - real *8 (0:nlevels)
c             size of box at various levels
c          nlevels - integer
c             number of levels
c          iptr - integer(12)
c            pointer to various arrays inside itree
c          ns - integer
c            number of sources
c          src - real *8 (2,ns)
c            xy coorindates of source locations
c          nt - integer
c            number of targets
c          targ - real *8 (2,nt)
c            xy cooridnates of target locations
c          fname1 - character *
c            file name to which tree info is to be written
c          fname1 - character *
c            file name to which source points are to be written
c          fname3 - character *
c            file name to which target points are to be written
c 
c          output
c          files with name fname1, fname2, fname3,
c            which contains the tree info, source points, target points
c            file can be plotted using the matlab script
c            tree_plot.m

      implicit real *8 (a-h,o-z)
      integer itree(ltree),ltree,nboxes,nlevels,iptr(12),ns,nt
      real *8 centers(ndim,nboxes),boxsize(0:nlevels)
      real *8 x(5),y(5),src(ndim,ns),targ(ndim,nt)
      character (len=*) fname1,fname2,fname3

      open(unit=33,file=trim(fname1))
      nleafbox = 0
      
      do i=1,nboxes
        if(itree(iptr(4)+i-1).eq.0) nleafbox = nleafbox+1
      enddo

 1111 format(10(2x,e11.5))      

      do ibox=1,nboxes
         if(itree(iptr(4)+ibox-1).eq.0) then
           ilev = itree(iptr(2)+ibox-1)
           bs = boxsize(ilev)
           x1 = centers(1,ibox) - bs/2
           x2 = centers(1,ibox) + bs/2

           if (ndim.eq.2) then
              y1 = centers(2,ibox) - bs/2
              y2 = centers(2,ibox) + bs/2
           endif

           if (ndim.eq.2) then
              write(33,1111) x1,x2,x2,x1,x1,y1,y1,y2,y2,y1
           else
              write(33,1111) x1,x2
           endif
         endif
      enddo
      close(33)

 2222 format(2(2x,e11.5))

      open(unit=33,file=trim(fname2))
      if (ns .gt. 0) then
         do i=1,ns
            write(33,2222) src(1,i),src(2,i)
         enddo
      endif
      
      close(33)
      open(unit=33,file=trim(fname3))
      if (nt .gt. 0) then
         do i=1,nt
            write(33,2222) targ(1,i),targ(2,i)
         enddo
      endif

      close(33)

      return
      end
c
c
c
c

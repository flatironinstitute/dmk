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
      nsrcs(1) =   100 000
      nsrcs(2) =   250 000
      nsrcs(3) =   500 000
      nsrcs(4) = 1 000 000
      nsrcs(5) = 2 000 000
      nsrcs(6) = 4 000 000

c     nd - number of different densities
      nd=1
c     dim - dimension of the problem
      dim=2
c     ikernel - 0: Yukawa kernel; 1: Laplace kernel;
c               2: square root Laplace kernel; 3: Stokeslet
      ikernel = 3
      
c     iperiod -> 0: free space; 1: periodic in all dimensions
      iperiod=0
c     ntarg: number of extra targets
cccc      ntarg=3
c     nsrc: number of source points

c     whether to have Stokeslet sources -> 1: yes; 0: no
      ifstoklet=1
c     whether to have stresslet sources -> 1: yes; 0: no
      ifstrslet=0
c     evaluation flag for sources -> 1: pot; 2: pot+grad; 3: pot+grad+hess
      ifppreg=1
c     evaluation flag for targets -> 1: pot; 2: pot+grad; 3: pot+grad+hess
      ifppregtarg=1
c
c     
c
c     test all parameters
c
      iw = 70
      iw2 = 80

      ifuniform=0

      if (dim.eq.2.and.ifuniform.eq.1.and.ifstoklet.eq.1) then
         open(iw, file='stok2dunif.txt', position='append')
      elseif (dim.eq.2.and.ifuniform.eq.0.and.ifstoklet.eq.1) then
         open(iw, file='stok2dadap.txt', position='append')
      elseif (dim.eq.2.and.ifuniform.eq.1.and.ifstrslet.eq.1) then
         open(iw, file='strs2dunif.txt', position='append')
      elseif (dim.eq.2.and.ifuniform.eq.0.and.ifstrslet.eq.1) then
         open(iw, file='strs2dadap.txt', position='append')
      elseif (dim.eq.3.and.ifuniform.eq.1.and.ifstoklet.eq.1) then
         open(iw, file='stok3dunif.txt', position='append')
      elseif (dim.eq.3.and.ifuniform.eq.0.and.ifstoklet.eq.1) then
         open(iw, file='stok3dadap.txt', position='append')
      elseif (dim.eq.3.and.ifuniform.eq.1.and.ifstrslet.eq.1) then
         open(iw, file='strs3dunif.txt', position='append')
      elseif (dim.eq.3.and.ifuniform.eq.0.and.ifstrslet.eq.1) then
         open(iw, file='strs3dadap.txt', position='append')
      endif

cccc  open(iw2, file='timing.txt', position='append')
      
      do i=2,2
         eps = epsvals(i)
cccc         do j=4,4
         do j=1,1
            nsrc=nsrcs(j)
cccc  for tree plotting
cccc            nsrc=125
cccc            nsrc=3
            ntarg=nsrc
            if (ifppregtarg.eq.0) ntarg=1
c
            call teststokesdmk(nd,dim,eps,ikernel,
     1          iperiod,nsrc,ntarg,ifstoklet,
     2          ifstrslet,ifppreg,ifppregtarg,ifuniform,
     3          totinfo(1,j,i),rerr)
 1400       FORMAT(4(2X,E9.3))
            write(iw,1400) (totinfo(k,j,i),k=1,4)
cccc  format(2x,D8.2,1x,'&',2x,D8.2,1x,'\\')
cccc  write(iw,4800) eps,  rerr
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
      subroutine teststokesdmk(nd,dim,eps,ikernel,iperiod,
     1    nsrc,ntarg,ifstoklet,ifstrslet,ifppreg,ifppregtarg,
     2    ifuniform,totinfo,rerr)
      implicit real *8 (a-h,o-z)
      integer dim
      real *8, allocatable :: sources(:,:),targ(:,:)
      real *8, allocatable :: sim(:,:)
      real *8, allocatable :: stoklet(:,:,:),strslet(:,:,:)
      real *8, allocatable :: strsvec(:,:,:)
      real *8, allocatable :: pot(:,:,:),pre(:,:),grad(:,:,:,:)
      real *8, allocatable :: pottarg(:,:,:),pretarg(:,:),
     1    gradtarg(:,:,:,:)
      real *8, allocatable :: potex(:,:,:),preex(:,:),gradex(:,:,:,:)
      real *8, allocatable :: pottargex(:,:,:),pretargex(:,:),
     1    gradtargex(:,:,:,:)

      real *8 shifts(dim),thresh,cen0(3),bs0,cavg(3)
      real *8 xs(100),timeinfo(20),totinfo(20)
      complex *16 ima
      data ima/(0.0d0,1.0d0)/

      call prini(6,13)

      done = 1
      pi = atan(done)*4

cccc      call prinf(' nsrc = *',nsrc,1)
c
      allocate(sources(dim,nsrc),stoklet(nd,dim,nsrc))
      allocate(strslet(nd,dim,nsrc),strsvec(nd,dim,nsrc))
      allocate(targ(dim,ntarg))

      allocate(pot(nd,dim,nsrc),pre(nd,nsrc),grad(nd,dim,dim,nsrc))
      allocate(pottarg(nd,dim,ntarg),gradtarg(nd,dim,dim,ntarg))
      allocate(pretarg(nd,ntarg))

      rin = 0.45d0
      rwig = 0.12d0
      rwig = 0
      nwig = 6
      
      h=8.0d1/nsrc
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
cccc  for tree plotting
cccc            do k=1,dim
cccc               sources(k,i) = exp(-h*i)
cccc            enddo
         endif

         do ind = 1,nd
         do k=1,dim
            stoklet(ind,k,i) = hkrand(0)-0.5d0 
            strslet(ind,k,i) = hkrand(0)-0.5d0
            strsvec(ind,k,i) = hkrand(0)-0.5d0
         enddo
         enddo
      enddo

      do ind=1,nd
      do k=1,dim   
         stoklet(ind,k,1) = 0.0d0 
         stoklet(ind,k,2) = 0.0d0 
         stoklet(ind,k,3) = 1.0d0

         strslet(ind,k,1) = 0.0d0 
         strslet(ind,k,2) = 0.0d0 
         strslet(ind,k,3) = 1.0d0
      enddo
      enddo


      if (iperiod.eq.1) then
         do k=1,dim
            cavg(k)=0
            do i=1,nsrc
               cavg(k)=cavg(k)+stoklet(1,k,i)
            enddo
            cavg(k) = cavg(k)/nsrc
         enddo

         do k=1,dim
            do i=1,nsrc
               stoklet(1,k,i) = stoklet(1,k,i)-cavg(k)
            enddo
         enddo
      endif
      
      do k=1,dim
         sources(k,1) = 0.0d0
      enddo

      do k=1,dim
         sources(k,2) = 1.1d0
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

      if (iperiod.eq.1) then
         bs0=1.3d0
         do k=1,dim
            cen0(k)=bs0/2
         enddo

         nx=1
         h=bs0/(nx+1)

         if (dim.eq.1) then
            targ(1,1) = -bs0/2+cen0(1)
            ntthalf=1
            targ(1,2) = bs0/2+cen0(1)
            ntt=2
         elseif (dim.eq.2) then
            ii=0
            do kk=1,dim
            do i=1,nx
               ii=ii+1
               do k=1,dim
                  if (k.eq.kk) then
                     targ(k,ii)=-bs0/2+i*h+cen0(k)
                  else
                     targ(k,ii)=-bs0/2+cen0(k)
                  endif
               enddo
            enddo
            enddo
            ntthalf=ii
            do kk=1,dim
               do i=1,nx
                  ii=ii+1
                  do k=1,dim
                     if (k.eq.kk) then
                        targ(k,ii)=-bs0/2+i*h+cen0(k)
                     else
                        targ(k,ii)=bs0/2+cen0(k)
                     endif
                  enddo
               enddo
            enddo
            ntt=ii
         elseif (dim.eq.3) then
            ii=0
            do i=1,nx
            do j=1,nx      
               ii=ii+1
               targ(1,ii)=-bs0/2+j*h+cen0(1)
               targ(2,ii)=-bs0/2+i*h+cen0(2)
               targ(3,ii)=-bs0/2+cen0(3)
            enddo
            enddo
            do i=1,nx
            do j=1,nx      
               ii=ii+1
               targ(1,ii)=-bs0/2+j*h+cen0(1)
               targ(2,ii)=-bs0/2+cen0(2)
               targ(3,ii)=-bs0/2+i*h+cen0(3)
            enddo
            enddo
            do i=1,nx
            do j=1,nx      
               ii=ii+1
               targ(1,ii)=-bs0/2+cen0(1)
               targ(2,ii)=-bs0/2+j*h+cen0(2)
               targ(3,ii)=-bs0/2+i*h+cen0(3)
            enddo
            enddo
            ntthalf=ii
            do i=1,nx
            do j=1,nx      
               ii=ii+1
               targ(1,ii)=-bs0/2+j*h+cen0(1)
               targ(2,ii)=-bs0/2+i*h+cen0(2)
               targ(3,ii)= bs0/2+cen0(3)
            enddo
            enddo
            do i=1,nx
            do j=1,nx      
               ii=ii+1
               targ(1,ii)=-bs0/2+j*h+cen0(1)
               targ(2,ii)= bs0/2+cen0(2)
               targ(3,ii)=-bs0/2+i*h+cen0(3)
            enddo
            enddo
            do i=1,nx
            do j=1,nx      
               ii=ii+1
               targ(1,ii)= bs0/2+cen0(1)
               targ(2,ii)=-bs0/2+j*h+cen0(2)
               targ(3,ii)=-bs0/2+i*h+cen0(3)
            enddo
            enddo
            ntt=ii
         endif
      endif
      
      ntest=1000
      nts = min(ntest,nsrc)
      if (iperiod.eq.0) ntt = min(ntest,ntarg)

cccc      call prinf('ntt=*',ntt,1)
cccc      call prin2('targ=*',targ,dim*ntt)
      
      allocate(potex(nd,dim,nts),preex(nd,nts),gradex(nd,dim,dim,nts))
      allocate(pottargex(nd,dim,ntt),gradtargex(nd,dim,dim,ntt))
      allocate(pretargex(nd,ntt))
c
      call dzero(potex,nts*nd*dim)
      call dzero(gradex,dim*nts*nd*dim)
      call dzero(preex,nts*nd)
      
      call dzero(pottargex,ntt*nd*dim)
      call dzero(gradtargex,dim*ntt*nd*dim)
      call dzero(pretargex,nhess*ntt*nd*dim)
      
      call cpu_time(t1)
C$    t1 = omp_get_wtime()      
      call stokesdmk(nd,dim,eps,iperiod,bs0,cen0,
     1    nsrc,sources,
     2    ifstoklet,stoklet,ifstrslet,strslet,strsvec,
     2    ifppreg,pot,pre,grad,ntarg,targ,
     3    ifppregtarg,pottarg,pretarg,gradtarg,timeinfo)
      call cpu_time(t2)
C$    t2=omp_get_wtime()

      pps=(nsrc*ifppreg+ntarg*ifppregtarg+0.0d0)/(t2-t1)
      call prin2('time in pdmk=*',t2-t1,1)
      totinfo(1) = nsrc*ifppreg+ntarg*ifppregtarg+0.0d0
c     time on building the tree and sorting the points
cccc      totinfo(2) = timeinfo(1)+timeinfo(2)
c     time on the plane-wave part
cccc      totinfo(3) = timeinfo(4)+timeinfo(5)+timeinfo(6)
c     time on direct interactions
cccc      totinfo(4) = timeinfo(7)
c     total time
      totinfo(2) = t2-t1
      totinfo(3) = pps
      call prin2('points per sec=*',pps,1)

      nprint=min(20,nsrc)
cccc      nprint=nts
      call prin2('pot=*',pot,nd*dim*nprint)
cccc      call prin2('gradtarg=*',gradtarg,nd*dim*ntt)
cccc      call prin2('hesstarg=*',hesstarg,nd*6*ntt)

      thresh=1.0d-16
      call cpu_time(t1)
C$      t1 = omp_get_wtime()      
      call stokes_kernel_direct(nd,dim,ikernel,
     1    thresh,1,nsrc,sources,
     1    ifstoklet,stoklet,ifstrslet,strslet,strsvec,
     2    1,nts,sources,ifppreg,potex,preex,gradex)
      call prin2('potex=*',potex,nd*nprint*dim)
      call cpu_time(t2)
C$       t2 = omp_get_wtime()      
cccc      print *, 'direct eval time = ', t2-t1

      call stokes_kernel_direct(nd,dim,ikernel,
     1    thresh,1,nsrc,sources,
     1    ifstoklet,stoklet,ifstrslet,strslet,strsvec,
     2    1,ntt,targ,ifppregtarg,pottargex,pretargex,gradtargex)
      if (ifppregtarg.gt.0) then
         if (iperiod.eq.0) then
            call prin2('pottarg=*',pottarg,nd*dim*nprint)
         else
            call prin2('pottarg=*',pottarg,nd*dim*ntt)
         endif
         call prin2('pottargex=*',pottargex,nd*dim*nprint)
      endif
      
      if (ifppreg .gt. 0)
     1    call derr(potex,pot,nts*nd*dim,errps,pnorm,errpa)
      if (ifppreg .gt. 1)
     1    call derr(preex,pre,nts*nd,errhs,hnorm,errha)
      if (ifppreg .gt. 2)
     1    call derr(gradex,grad,dim*nts*nd*dim,errgs,gnorm,errga)

      if (iperiod.eq.0) then
         if (ifppregtarg .gt. 0)
     1       call derr(pottargex,pottarg,ntt*nd*dim,errpt,tmp1,tmp2)
         if (ifppregtarg .gt. 1) 
     1       call derr(pretargex,pretarg,ntt*nd,errht,
     2       tmp1,tmp2)
         if (ifppregtarg .gt. 2)
     1       call derr(gradtargex,gradtarg,dim*ntt*nd*dim,
     2       errgt,tmp1,tmp2)
      else
         if (ifppregtarg .gt. 0) call derr(pottarg,
     1       pottarg(1,1,ntthalf+1),ntthalf*nd*dim,errpt,tmp1,tmp2)
      endif
         
      call errprint(errps,errgs,errhs,errpt,errgt,errht,
     1    ifppreg,ifppregtarg)

      
      totinfo(4)=errps
      
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
         dd = vec1(i) - vec2(i)
         erra = erra + dd**2
cccc         print *, dd
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
     1    ifppreg,ifppregtarg)
      implicit real *8 (a-h,o-z)
 1100 format(3(2x,e11.5))


      write(6,*) 'ifppreg is ', ifppreg
      write(6,*) 'ifppregtarg is ', ifppregtarg
      if (ifppreg .gt. 0) write(6,*) 'error in sources'
      if (ifppreg .gt. 0) call prin2('pot err =*', errps,1)
      if (ifppreg .gt. 1) call prin2('pressure err=*', errgs,1)
      if (ifppreg .gt. 2) call prin2('grad err=*', errhs,1)
      write(6,*) 
      if (ifppregtarg .gt. 0) write(6,* ) 'error in targets'
      if (ifppregtarg .gt. 0) call prin2('pot err =*', errpt,1)
      if (ifppregtarg .gt. 1) call prin2('pressure err=*', errgt,1)
      if (ifppregtarg .gt. 2) call prin2('grad err=*', errht,1)
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

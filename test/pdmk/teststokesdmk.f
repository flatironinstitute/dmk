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
      dim=3
c     ikernel - 0: Yukawa kernel; 1: Laplace kernel;
c               2: square root Laplace kernel; 3: Stokeslet
      ikernel = 3
      
c     iperiod -> 0: free space; 1: periodic in all dimensions
      iperiod=0
c     ntarg: number of extra targets
      ntarg=3
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

      do ikernel=3,3
      do dim=3,3
      if (ikernel.eq.3.and.dim.eq.2.and.ifuniform.eq.1) then
         open(iw, file='st2dptsu.txt', position='append')
      elseif (ikernel.eq.3.and.dim.eq.2.and.ifuniform.eq.0) then
         open(iw, file='st2dptsa.txt', position='append')
      elseif (ikernel.eq.3.and.dim.eq.3.and.ifuniform.eq.1) then
         open(iw, file='st3dptsu.txt', position='append')
      elseif (ikernel.eq.3.and.dim.eq.3.and.ifuniform.eq.0) then
         open(iw, file='st3dptsa.txt', position='append')
      endif

cccc  open(iw2, file='timing.txt', position='append')
      
      do i=2,2
         eps = epsvals(i)
cccc         do j=4,4
         do j=1,1
            nsrc=nsrcs(j)
            nsrc=1*10**5
            nsrc=3
            ntarg=nsrc
            if (ifppregtarg.eq.0) ntarg=1
c
            call teststokesdmk(nd,dim,eps,ikernel,
     1          iperiod,nsrc,ntarg,ifstoklet,
     2          ifstress,ifppreg,ifppregtarg,ifuniform,
     3          totinfo(1,j,i),rerr)
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
      enddo
      enddo
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

      real *8 shifts(dim),thresh
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
      enddo
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
      call stokesdmk(nd,dim,eps,
     1    iperiod,nsrc,sources,
     2    ifstoklet,stoklet,ifstrslet,strslet,strsvec,
     2    ifppreg,pot,pre,grad,ntarg,targ,
     3    ifppregtarg,pottarg,pretarg,gradtarg,timeinfo)
      call cpu_time(t2)
C$    t2=omp_get_wtime()

      pps=(nsrc*ifppreg+ntarg*ifppregtarg+0.0d0)/(t2-t1)
      call prin2('time in pdmk=*',t2-t1,1)
      totinfo(2)=nsrc*1.0d0
c     time on building the tree and sorting the points
      totinfo(3)=timeinfo(1)+timeinfo(2)
c     time on the plane-wave part
      totinfo(4)=timeinfo(4)+timeinfo(5)+timeinfo(6)
c     time on direct interactions
      totinfo(5)=timeinfo(7)
c     total time
      totinfo(6)=t2-t1
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
         call prin2('pottarg=*',pottarg,nd*dim*nprint)
         call prin2('pottargex=*',pottargex,nd*dim*nprint)
      endif
      
      if (ifppreg .gt. 0)
     1    call derr(potex,pot,nts*nd*dim,errps,pnorm,errpa)
      if (ifppreg .gt. 1)
     1    call derr(preex,pre,nts*nd,errhs,hnorm,errha)
      if (ifppreg .gt. 2)
     1    call derr(gradex,grad,dim*nts*nd*dim,errgs,gnorm,errga)

      if (ifppregtarg .gt. 0)
     1    call derr(pottargex,pottarg,ntt*nd*dim,errpt,tmp1,tmp2)
      if (ifppregtarg .gt. 1) 
     1    call derr(pretargex,pretarg,ntt*nd,errht,
     2    tmp1,tmp2)
      if (ifppregtarg .gt. 2)
     1    call derr(gradtargex,gradtarg,dim*ntt*nd*dim,errgt,tmp1,tmp2)

      call errprint(errps,errgs,errhs,errpt,errgt,errht,
     1    ifppreg,ifppregtarg)

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

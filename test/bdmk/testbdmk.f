      implicit real *8 (a-h,o-z)
      real *8 epsvals(5),deltas(20),pps(20,5),rsig2(10),rsig3(10)
      real *8 rsig0(10)
      integer ntot(20)
c	  
      call prini(6,13)

      pi = 4*atan(1.0d0)

      epsvals(1) = 1d-3
      epsvals(2) = 1d-6
      epsvals(3) = 1d-9
      epsvals(4) = 1d-11

      rsig0(1) = 60.0d0
      rsig0(2) = 96.0d0
      rsig0(3) = 110.0d0
      rsig0(4) = 180.0d0

      rsig2(1) = 1.0d-5
      rsig2(2) = 1.0d-5/3
      rsig2(3) = 1.0d-5/9
      rsig2(4) = 1.0d-5/27

      rsig3(1) = 4.0d-3
      rsig3(2) = 1.0d-3
      rsig3(3) = 1.0d-4
      rsig3(4) = 1.0d-5
      
c     ipoly=0: Legendre; 1: Chebyshev
      ipoly=0
c     ikernel - 0: Yukawa kernel; 1: Laplace kernel; 2: square root Laplace kernel
      ikernel = 1
c     beta - the parameter in the Yukawa kernel or the exponent of the
c     power function kernel
      beta=0.1d-5
      beta=0.1d1
c     polynomial expansion order in each dimension
      norder=16
c     nd - number of different densities
      nd=1
c     ndim - dimension of the underlying space
      ndim=3
c     ifpgh = 1: potential; 2: pot+grad; 3: pot+grad+hess
      ifpgh=1
c     ifpghtarg: flag for arbitrary targets
      ifpghtarg=0
c     ntarg: number of extra targets
      ntarg=1 000 000
c     
c
c     test all parameters
c
c      do ikernel=0,2
      do ikernel=1,1
c      do ndim=2,2
      do ndim=3,3
      
      iw = 80
      if (ikernel.eq.0.and.ndim.eq.2) then
         open(iw, file='y2dbox.txt', position='append')
      elseif (ikernel.eq.0.and.ndim.eq.3) then
         open(iw, file='y3dbox.txt', position='append')
      elseif (ikernel.eq.1.and.ndim.eq.2) then
         open(iw, file='l2dbox.txt', position='append')
      elseif (ikernel.eq.1.and.ndim.eq.3) then
         open(iw, file='l3dbox.txt', position='append')
      elseif (ikernel.eq.2.and.ndim.eq.2) then
         open(iw, file='sl2dbox.txt', position='append')
      elseif (ikernel.eq.2.and.ndim.eq.3) then
         open(iw, file='sl3dbox.txt', position='append')
      endif

c      write(iw,*) 'ipoly=',ipoly,' iperiod=',iperiod,' norder =',norder
c      write(iw,*) 'ndim=',ndim,' ifpgh=',ifpgh
      
c      do i=1,4
      do i=2,2
         eps = epsvals(i)
cccc         do j=1,1
         do j=3,3
            if (ndim.eq.2) then
               rsig=rsig2(j)
            elseif (ndim.eq.3) then
               rsig=rsig3(j)
            endif
            if (ikernel.le.1 .and. ndim.eq.2) then
               rsig=rsig0(j)
            endif
            
            call testbdmk(ikernel,beta,eps,rsig,nd,ndim,norder,ipoly,
     1           ifpgh,ifpghtarg,rerr,pps(1,1),ntot(1))

            write(iw,*) rerr, pps(1,1), ntot(1)
         enddo
      enddo
      
      
      close(iw)

      enddo
      enddo
      
      end
c
c
c
      subroutine testbdmk(ikernel,beta,eps,rsig,nd,ndim,norder,ipoly,
     1    ifpgh,pfpghtarg,errp,pps,ntot)
      implicit real *8 (a-h,o-z)
      real *8 dpars(1000)
      integer iptr(8),ltree,ipars(100)
      integer ifpgh,ifpghtarg
      integer, allocatable :: itree(:)
      real *8, allocatable :: fvals(:,:,:),centers(:,:),boxsize(:)
      real *8, allocatable :: fltrue(:,:,:),flvals(:,:,:)
      real *8, allocatable :: fl2true(:,:,:),fl2vals(:,:,:)

      real *8, allocatable :: xref(:,:),wts(:)
      real *8 rintl(0:200)
c
      real *8 timeinfo(100)
      complex *16 zpars(10)

      real *8, allocatable :: pot(:,:,:), potex(:,:,:)
      real *8, allocatable :: grad(:,:,:,:), gradex(:,:,:,:)
      real *8, allocatable :: hess(:,:,:,:), hessex(:,:,:,:)

      real *8, allocatable :: coefs(:,:,:)
      real *8, allocatable :: coefsg(:,:,:,:)
      real *8, allocatable :: adiff(:,:)
      
      real *8, allocatable :: targs(:,:)

      real *8, allocatable :: pote(:,:)
      real *8, allocatable :: grade(:,:,:)
      real *8, allocatable :: hesse(:,:,:)

      real *8, allocatable :: potexe(:,:)
      real *8, allocatable :: gradexe(:,:,:)
      real *8, allocatable :: hessexe(:,:,:)

      integer ntot
      complex *16 ima,zz,ztmp,zk

      real *8 xs(100),ws(100),vmat(2000)
      real *8 vpmat(2000),vppmat(2000)
      real *8 ainte(2000),endinter(1000),work(10000)
      real *8 polin(100),polout(100)

      real *8, allocatable :: umat(:,:),umat_nd(:,:,:)
      real *8 alpha,beta,src(ndim,1)
      character *12 fname1
      character *8 fname2
      character *9 fname3
      real *8, allocatable :: targ(:)

      character *1 type
      data ima/(0.0d0,1.0d0)/
      real *8 omp_get_wtime
      
      external rhsfun,uexact

      done = 1
      pi = atan(done)*4

      ifnewtree=0
      
c      delta=1.0d-3
      
      ipars(1) = ndim
      ipars(2) = ikernel
      
      delta=1.0d0
c     for exact solution
      dpars(201)=beta

      iperiod=0
      ipars(5)=iperiod
      
c     p in L^p norm - 0: L^infty norm; 1: L^1 norm; 2: L^2 norm
      iptype = 2
cccc      if (iperiod.eq.1) iptype = 2
cccc      eta = 1.0d0
      eta = 0.0d0
      
c     number of gaussians in the rhs function
      ng = 2
      if (ikernel.eq.2 .and. ndim.eq.2) ng=40
      ipars(3) = ng

c     number of points per box
      npbox = norder**ndim

      type = 'f'

      ipars(10) = max(ifpgh,ifpghtarg)
      
cccc  call prini_off()

c
c     initialize function parameters
c
      boxlen = 1.18d0
cccc      boxlen = 1.0d0
c     gaussian variance of the input data
c      rsig = 1.0d0/4000.0d0
c      rsig = 0.00025d0
c      if (ndim.eq.2) then
c         rsig = 1.0d-6
c         rsig = 1.0d-5
c         rsig = 2.0d-6
c         rsig = 1.0d-5
c      elseif (ndim.eq.3) then
c         rsig = 1.0d-7
c         rsig = 1.0d-6
c         rsig = 1.0d-5
c         rsig = 1.0d-4
c         rsig = 1.0d-3
c         rsig = 4.0d-3
c      endif

      
      zk = ipars(2)*7.05d0
      zk = 30.0d0
c     needs to be make sure that the tree_build is capable of catching
c     sharply peaked Gaussian input data!
c      if (rsig.le.1e-5) zk = ipars(2)*27.05d0

      delta=1.0d0
c     proper normalization of the input data
      rsign=(rsig*delta)**(ndim/2.0d0)
      
c     first gaussian
c     centers
      dpars(1) = 0.1d0
      dpars(2) = 0.02d0
      dpars(3) = 0.04d0
c     variance
      dpars(4) = rsig
c     strength
      dpars(5) = 1/pi/rsign
c      dpars(5) = 1.0d0

c     second gaussian
      dpars(6) = 0.03d0
      dpars(7) = -0.1d0
      dpars(8) = 0.05d0

      dpars(9) = rsig/2
      dpars(10) = -0.5d0/pi/rsign
c      dpars(10) = -0.5d0
c      dpars(10) = 2.0d0

      
c     third gaussian
      dpars(11) = 0.180
      dpars(12) = -0.1d0
      dpars(13) = 0.03d0

      dpars(14) = rsig
      dpars(15) = -1.0d0/pi/rsign
      
c     fourth gaussian
      dpars(16) = -0.09d0
      dpars(17) = 0.3d0
      dpars(18) = 0.17d0

      dpars(19) = rsig
      dpars(20) = 2.0d0/pi/rsign

c     fifth gaussian
      dpars(21) = -0.3d0
      dpars(22) = -0.05d0
      dpars(23) = -0.17d0

      dpars(24) = rsig
      dpars(25) = -3.0d0/pi/rsign

      if (ikernel.eq.2) then
         r0=0.15d0
         do i=1,ng
            istart=(i-1)*5
            theta=2*pi/ng*i
            dpars(istart+1)=r0*cos(theta)
            dpars(istart+2)=r0*sin(theta)
            dpars(istart+3)=0.0d0
            dpars(istart+4)=rsig
            dpars(istart+5)=1.0d0/pi/rsign*(hkrand(0)-0.5d0)
         enddo
      endif
      
      if (ikernel.le.1) then
         if (ndim.eq.2) then
c            dpars(1) = 70.0d0
c            dpars(1) = 90.0d0
c     dpars(1) = 200.0d0
            dpars(1) = rsig
            dpars(2) = 0.25d0
         endif
      endif
      
      ntarg = 20
      nhess = ndim*(ndim+1)/2
      allocate(targs(ndim,ntarg),pote(nd,ntarg))
      allocate(grade(nd,ndim,ntarg),hesse(nd,nhess,ntarg))

      do i=1,ntarg
         do j=1,ndim
            targs(j,i) = (hkrand(0)-0.5d0)*boxlen
         enddo
      enddo

      epstree=eps/10
      epstree=eps*500
cccc      epstree=eps*20
cccc      epstree=1.0d-12
      call cpu_time(t1)
C$    t1 = omp_get_wtime()
      call vol_tree_mem(ndim,ipoly,iperiod,epstree,zk,boxlen,
     1    norder,iptype,eta,rhsfun,nd,dpars,zpars,ipars,ifnewtree,
     2    nboxes,nlevels,ltree,rintl)
      
      allocate(fvals(nd,npbox,nboxes),centers(ndim,nboxes))
      allocate(boxsize(0:nlevels),itree(ltree))

      call vol_tree_build(ndim,ipoly,iperiod,epstree,zk,boxlen,
     1    norder,iptype,eta,rhsfun,nd,dpars,zpars,ipars,rintl,
     2    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,fvals)
      call cpu_time(t2)
C$    t2 = omp_get_wtime()      

      call prinf('nboxes=*',nboxes,1)
      call prinf('nlevels=*',nlevels,1)
      call prinf('laddr=*',itree,2*(nlevels+1))

c     compute the number of leaf boxes
      nlfbox = 0
      do ilevel=1,nlevels
        do ibox=itree(2*ilevel+1),itree(2*ilevel+2)
          if(itree(iptr(4)+ibox-1).eq.0) then
            nlfbox = nlfbox+1
          endif
        enddo
      enddo
      call prinf('nlfbox=*',nlfbox,1)

      call prin2('time taken to build tree=*',t2-t1,1)
      call prin2('speed in points per sec=*',
     1   (nboxes*npbox+0.0d0)/(t2-t1),1)

c     allocate memory and initialization

      allocate(fltrue(nd,npbox,nboxes))
      allocate(fl2true(nd,npbox,nboxes))
      allocate(flvals(nd,npbox,nboxes))
      allocate(fl2vals(nd,npbox,nboxes))

      allocate(pot(nd,npbox,nboxes))
      allocate(grad(nd,ndim,npbox,nboxes))
      allocate(hess(nd,nhess,npbox,nboxes))

      do i=1,nboxes
      do j=1,npbox
      do ind=1,nd
         pot(ind,j,i) = 0
         if (ifpgh.ge.2) then
            do k=1,ndim
               grad(ind,k,j,i) = 0
            enddo
         endif
         if (ifpgh.ge.3) then
            do k=1,nhess
               hess(ind,k,j,i) = 0
            enddo
         endif
      enddo
      enddo
      enddo

      allocate(potex(nd,npbox,nboxes))
      allocate(gradex(nd,ndim,npbox,nboxes))
      allocate(hessex(nd,nhess,npbox,nboxes))

c     compute exact solutions on tensor grid
      allocate(xref(ndim,npbox),wts(npbox))
      itype = 0
      call polytens_exps_nd(ndim,ipoly,itype,norder,type,xref,
     1    utmp,1,vtmp,1,wts)

      call prin2('Computing the analytic solution*',potex,0)

      allocate(targ(ndim))
      do ilevel=0,nlevels
        bs = boxsize(ilevel)/2.0d0
        do ibox=itree(2*ilevel+1),itree(2*ilevel+2)
          if(itree(iptr(4)+ibox-1).eq.0) then
             do j=1,npbox
               do k=1,ndim
                 targ(k)=centers(k,ibox) + xref(k,j)*bs
               enddo

               call uexact(nd,targ,dpars,zpars,ipars,
     1             ifpgh,potex(1,j,ibox),
     2             gradex(1,1,j,ibox),hessex(1,1,j,ibox))
             enddo
          endif
        enddo
      enddo
      
      call cpu_time(t1) 
C$    t1 = omp_get_wtime()      

      call bdmk(nd,ndim,eps,ikernel,beta,ipoly,norder,npbox,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,fvals,
     2    ifpgh,pot,grad,hess,ntarg,targs,
     3    ifpghtarg,pote,grade,hesse,timeinfo)

      call cpu_time(t2) 
C$    t2 = omp_get_wtime()
      
      call prin2('time taken in bdmk=*',t2-t1,1)
      ntot=npbox*nlfbox*ifpgh+ntarg*ifpghtarg
      pps=(ntot+0.0d0)/(t2-t1)
      print *, 'ntotal=',ntot, npbox,nlfbox,ntarg
      print *, 'pps=', pps
      call prinf('ntotal=*',ntot,1)
      call prin2('speed in pps=*',pps,1)
      
      if (ifpgh.ge.1) then
         call treedata_derror(nd,nlevels,itree,iptr,
     1       npbox,potex,pot,abserrp,rnormp,nleaf)
c     example 1a
         if (rnormp.lt.1d-6) then
            errp = abserrp
         else
            errp = abserrp/rnormp
         endif
         call prin2('pot l2 norm=*',rnormp,1)
         call prin2('absolute pot l2 error=*',abserrp,1)
         call prin2('relative pot l2 error=*',errp,1)
      endif

      if (ifpgh.ge.2) then
         call treedata_derror(nd*ndim,nlevels,itree,iptr,
     1       npbox,gradex,grad,abserrg,rnormg,nleaf)
         errg = abserrg/rnormg
         call prin2('grad l2 norm=*',rnormg,1)
         call prin2('absolute grad l2 error=*',abserrg,1)
         call prin2('relative grad l2 error=*',errg,1)
      endif

      if (ifpgh.ge.3) then
         call treedata_derror(nd*nhess,nlevels,itree,iptr,
     1       npbox,hessex,hess,abserrh,rnormh,nleaf)
         errh = abserrh/rnormh
         call prin2('hess l2 norm=*',rnormh,1)
         call prin2('absolute hess l2 error=*',abserrh,1)
         call prin2('relative hess l2 error=*',errh,1)
      endif

c     compute exact solutions on arbitrary targets      
      if (ifpghtarg.ge.1) then
         allocate(potexe(nd,ntarg))
         allocate(gradexe(nd,ndim,ntarg))
         allocate(hessexe(nd,nhess,ntarg))
         ntest=ntarg
         do j=1,ntest
            call uexact(nd,targs(1,j),dpars,zpars,ipars,
     1          ifpghtarg,potexe(1,j),gradexe(1,1,j),hessexe(1,1,j))
         enddo
c
c     compute relative error
         call derr(potexe,pote,nd*ntest,errpe)
         call prin2('relative pottarg l2 error=*',errpe,1)
         call prin2('pote=*',pote,ntest)
         call prin2('potexe=*',potexe,ntest)
      endif
      
      return
      end
c
c
c
      subroutine rhsfun(nd,xyz,dpars,zpars,ipars,f)
c     right-hand-side function
c     for free space problem:  consisting of several gaussians, their
c       centers are given in dpars(1:3), their 
c       variances in dpars(4), and their strength in dpars(5)
c
      implicit real *8 (a-h,o-z)
      integer nd,ndim,ipars(*)
      complex *16 zpars(*)
      real *8 dpars(*),f(nd),xyz(*),f2(nd),pi,grad(100),hess(100)
      data pi/3.1415926535 8979323846 2643383279 5028841971 693993751d0/
c
c     
      ndim=ipars(1)
      ikernel=ipars(2)
c     number of Gaussians
      ng=ipars(3)

      if (ikernel.eq.2) then
         call fgaussn(nd,xyz,dpars,zpars,ipars,f)
         return
      endif
      
      if (ndim.eq.3) then
         do ind=1,nd
            f(ind)=0
            do i=1,ng
               idp = (i-1)*5
               rr=0
               do k=1,ndim
                  rr = rr + ( xyz(k) - dpars(idp+k) )**2  
               enddo
               sigma = dpars(idp+4)
               f(ind) = f(ind)+dpars(idp+5)*exp(-rr/sigma)
     1              *(-2*ndim+4*rr/sigma)/sigma
            enddo
         enddo
      elseif (ndim.eq.2) then
         alpha=dpars(1)
         r0=dpars(2)
         r02=r0*r0
         r2=0
         do k=1,ndim
            r2=r2+xyz(k)**2
         enddo
         r=sqrt(r2)
         ralpha=(r/r0)**alpha
         rbeta=ralpha/r2
         expr=exp(-ralpha)
         do ind=1,nd
            f(ind)=(-ndim-alpha+2+alpha*ralpha)*alpha*expr*rbeta
c            f(ind)=(ralpha-1)*alpha*alpha*expr*rbeta
         enddo
      endif

      if (ikernel.eq.0) then
         ifpgh=1
         call uexact(nd,xyz,dpars,zpars,ipars,ifpgh,f2,grad,hess)
         if (ndim.eq.3) then
            dfac=-4*pi
         elseif (ndim.eq.2) then
            if (ikernel.eq.0) then
               dfac=-2*pi
            elseif (ikernel.eq.1) then
               dfac=2*pi
            endif
         endif

         
         beta=dpars(201)
         beta2=beta*beta
         
         do ind=1,nd
            f(ind)=f(ind)-beta2*f2(ind)/dfac
         enddo
      endif
c     
      return
      end
c
c
c
c
      subroutine rhslap(nd,xyz,dpars,zpars,ipars,f,f2)
c     compute the laplacian of the right-hand-side function
c
      implicit real *8 (a-h,o-z)
      integer nd,ndim,ipars(*)
      complex *16 zpars(*)
      real *8 dpars(*),f(nd),xyz(*),f2(nd)
c
c     
c     number of Gaussians
      ng=ipars(2)
      ndim=ipars(1)
      
      do ind=1,nd
         f(ind)=0
         f2(ind)=0
         do i=1,ng
            idp = (i-1)*5
            r2=0
            do k=1,ndim
               r2 = r2 + ( xyz(k) - dpars(idp+k) )**2  
            enddo
            r4 = r2*r2
            r6 = r4*r2
            s = dpars(idp+4)
            s2 = s*s
            s3 = s2*s
            s4 = s2*s2
            s6 = s4*s2
            f(ind) = f(ind)+dpars(idp+5)*exp(-r2/s)
     1          *4*(4*r4+15*s2-20*s*r2)/s4
            f2(ind) = f2(ind)+dpars(idp+5)*exp(-r2/s)
     1          *8*(8*r6-84*s*r4+210*s2*r2-105*s3)/s6
         enddo
      enddo
c     
      return
      end
c
c
c
c
      subroutine uexact(nd,targ,dpars,zpars,ipars,
     1    ifpgh,pot,grad,hess)
c     exact solution for the given rhs function
c     ifpgh>1 only supported for dim=2 and ikernel=0,1
      implicit real*8 (a-h,o-z)
      real*8 targ(*),pot(nd),grad(nd,*),hess(nd,*)
      real*8 dpars(*),rdiff(10)
      complex *16 zpars(*)
      integer ipars(*)
      real *8 pi
      data pi/3.1415926535 8979323846 2643383279 5028841971 693993751d0/

      ndim=ipars(1)
      ikernel=ipars(2)
      ng=ipars(3)

      nhess=ndim*(ndim+1)/2
      
      if (ikernel.eq.2) then
c     use truncated trapezoidal rule to calculate the "analytic" solution
c     the algorithm actually works for any alpha \in (0,2], but we have
c     only implemented the SOG approximation for 1/r in 2d and 1/r^2 in 3d
c     for now.
         if (ndim.eq.3) then
            alpha=2.0d0
            gval=1.0d0
         elseif (ndim.eq.2) then
            alpha=1.0d0
            gval= sqrt(pi)
         endif

         eps=1.0d-16
         h = 4*pi/log(1.0d0/eps)
c     intel mkl library special function vdTGamma
c     gval = gamma(alpha/2) for general alpha
cccc         call vdTGamma(1,alpha/2,gval)
         
         do ind=1,nd
            pot(ind)=0
            if (ifpgh.ge.2) then
               do j=1,ndim
                  grad(ind,j)=0
               enddo
            endif

            if (ifpgh.ge.3) then
               do j=1,nhess
                  hess(ind,j)=0
               enddo
            endif
            
            do i=1,ng
               idp = (i-1)*5
               r2=0
               do k=1,ndim
                  rdiff(k) = targ(k) - dpars(idp+k)
                  r2 = r2 + rdiff(k)*rdiff(k)
               enddo
               sigma = dpars(idp+4)
               tupper = 2.0d0/(ndim-alpha)*
     1             (log(1.0d0/eps)+ndim*log(1.0d0/sigma)/2)
               tlower = 2.0d0/alpha*log(eps)
               mmin=floor(tlower/h)
               mmax=ceiling(tupper/h)

               do k=mmin,mmax
                  t=k*h
                  
                  d1=sqrt(pi/(exp(t)+1.0d0/sigma))**ndim
                  d2=exp(-r2/(exp(-t)+sigma)+alpha*t/2)
                  dd=d1*d2*h*dpars(idp+5)/gval
                  pot(ind)=pot(ind)+dd

                  if (ifpgh.ge.2) then
                     sc = -2/(exp(-t)+sigma)
                     do j=1,ndim
                        grad(ind,j)=grad(ind,j)+dd*rdiff(j)*sc
                     enddo
                  endif

                  if (ifpgh.ge.3) then
                     sc = -2/(exp(-t)+sigma)
                     sc2=sc*sc
                     if (ndim.eq.2) then
                        hess(ind,1)=hess(ind,1)+dd*((rdiff(1)*sc)**2+sc)
                        hess(ind,2)=hess(ind,2)+dd*sc2*rdiff(1)*rdiff(2)
                        hess(ind,3)=hess(ind,3)+dd*((rdiff(2)*sc)**2+sc)
                     endif
                     if (ndim.eq.3) then
                        hess(ind,1)=hess(ind,1)+dd*((rdiff(1)*sc)**2+sc)
                        hess(ind,2)=hess(ind,2)+dd*((rdiff(2)*sc)**2+sc)
                        hess(ind,3)=hess(ind,3)+dd*((rdiff(3)*sc)**2+sc)

                        hess(ind,4)=hess(ind,4)+dd*sc2*rdiff(1)*rdiff(2)
                        hess(ind,5)=hess(ind,5)+dd*sc2*rdiff(1)*rdiff(3)
                        hess(ind,6)=hess(ind,6)+dd*sc2*rdiff(2)*rdiff(3)
                     endif
                  endif
               enddo
            enddo
         enddo
         
         return
      endif

      
      if (ndim.eq.3) then
         dfac=-4*pi
c         if (ikernel.eq.0) then
c            dfac=-2*pi
c         elseif (ikernel.eq.1) then
c            dfac=2*pi
c         endif
         call fgaussn2(nd,targ,dpars,zpars,ipars,
     1       ifpgh,pot,grad,hess)
         do ind=1,nd
            pot(ind)=pot(ind)*dfac
         enddo
         do ind=1,nd
            do k=1,ndim
               grad(ind,k)=grad(ind,k)*dfac
            enddo
         enddo
         do ind=1,nd
            do k=1,nhess
               hess(ind,k)=hess(ind,k)*dfac
            enddo
         enddo
      elseif (ndim.eq.2) then
         if (ikernel.eq.0) then
            dfac=-2*pi
         elseif (ikernel.eq.1) then
            dfac=2*pi
         endif

         alpha=dpars(1)
         r0=dpars(2)
         r2=0
         do k=1,ndim
            r2=r2+targ(k)**2
         enddo
         r=sqrt(r2)
         ralpha=(r/r0)**alpha
         expr=exp(-ralpha)
         do ind=1,nd
            pot(ind)=expr*dfac
         enddo
         if (ifpgh .ge.2) then
            do ind=1,nd
               tmp = -alpha*expr*dfac*(r/r0)**(alpha-2)/r0**2
               do k=1,ndim
                  grad(ind,k)=tmp*targ(k)
               enddo
            enddo
         endif
         if (ifpgh.ge.3) then
            do ind=1,nd
               t0 = expr*dfac
               t1 =  alpha**2*(r/r0)**(2*alpha-4)/r0**4
               t2 = -alpha*(alpha-2)*(r/r0)**(alpha-4)/r0**4
               t3 = -alpha*(r/r0)**(alpha-2)/r0**2

               hess(ind,1) = t0*(t1+t2)*targ(1)**2 + t0*t3
               hess(ind,2) = t0*(t1+t2)*targ(1)*targ(2)
               hess(ind,3) = t0*(t1+t2)*targ(2)**2 + t0*t3
            enddo
         endif
      endif
      
c     
      
      return
      end
c
c
c
c     
      subroutine fgaussn(nd,xyz,dpars,zpars,ipars,f)
c     right-hand-side function
c       consisting of several gaussians, their
c       centers are given in dpars(1:3), their 
c       variances in dpars(4), and their strength in dpars(5)
c
      implicit real *8 (a-h,o-z)
      integer nd,ndim,ipars(*)
      complex *16 zpars(*)
      real *8 dpars(*),xyz(*),f(nd)
      real *8 pi,dfac
      pi=4.0d0*atan(1.0d0)
c     number of Gaussians
      ndim=ipars(1)
      ng=ipars(3)

      do ind=1,nd
         f(ind)=0
         do i=1,ng
            idp = (i-1)*5
            r2=0
            do k=1,ndim
               r2 = r2 + ( xyz(k) - dpars(idp+k) )**2  
            enddo
            sigma = dpars(idp+4)
            f(ind) = f(ind)+dpars(idp+5)*exp(-r2/sigma)
         enddo
      enddo

      return
      end
c
c
c
c     
      subroutine fgaussn2(nd,xyz,dpars,zpars,ipars,
     1    ifpgh,pot,grad,hess)
c       consisting of several gaussians, their
c       centers are given in dpars(1:3), their 
c       variances in dpars(4), and their strength in dpars(5)
c
      implicit real *8 (a-h,o-z)
      integer nd,ndim,ipars(*)
      complex *16 zpars(*)
      real *8 pot(nd),grad(nd,*),hess(nd,*)
      real *8 dpars(*),xyz(*),rdiff(10)
      real *8 pi,dfac
      pi=4.0d0*atan(1.0d0)
c     number of Gaussians
      ndim=ipars(1)
      ng=ipars(3)
      nhess=ndim*(ndim+1)/2

      do ind=1,nd
         pot(ind)=0
         if (ifpgh.ge.2) then
            do j=1,ndim
               grad(ind,j)=0
            enddo
         endif

         if (ifpgh.ge.3) then
            do j=1,nhess
               hess(ind,j)=0
            enddo
         endif

         do i=1,ng
            idp = (i-1)*5
            r2=0
            do k=1,ndim
               rdiff(k)=xyz(k) - dpars(idp+k)
               r2 = r2 + rdiff(k)*rdiff(k)
            enddo
            sigma = dpars(idp+4)
            dd = dpars(idp+5)*exp(-r2/sigma)

            pot(ind) = pot(ind)+dd

            if (ifpgh.ge.2) then
               sc = -2/sigma
               do j=1,ndim
                  grad(ind,j)=grad(ind,j)+dd*rdiff(j)*sc
               enddo
            endif

            if (ifpgh.ge.3) then
               sc = -2/sigma
               sc2=sc*sc
               if (ndim.eq.2) then
                  hess(ind,1)=hess(ind,1)+dd*((rdiff(1)*sc)**2+sc)
                  hess(ind,2)=hess(ind,2)+dd*sc2*rdiff(1)*rdiff(2)
                  hess(ind,3)=hess(ind,3)+dd*((rdiff(2)*sc)**2+sc)
               endif
               if (ndim.eq.3) then
                  hess(ind,1)=hess(ind,1)+dd*((rdiff(1)*sc)**2+sc)
                  hess(ind,2)=hess(ind,2)+dd*((rdiff(2)*sc)**2+sc)
                  hess(ind,3)=hess(ind,3)+dd*((rdiff(3)*sc)**2+sc)
                  
                  hess(ind,4)=hess(ind,4)+dd*sc2*rdiff(1)*rdiff(2)
                  hess(ind,5)=hess(ind,5)+dd*sc2*rdiff(1)*rdiff(3)
                  hess(ind,6)=hess(ind,6)+dd*sc2*rdiff(2)*rdiff(3)
               endif
            endif
         enddo
      enddo

      return
      end
c
c
c
c
      subroutine derr(vec1,vec2,n,erra)
      implicit real *8 (a-h,o-z)
      real *8 vec1(*),vec2(*)

      ra = 0
      erra = 0
      
      do i=1,n
         ra = ra + vec1(i)**2
c         if (ra .lt. abs(vec1(i))) ra=abs(vec1(i))
c         if (erra.lt.abs(vec1(i)-vec2(i))) then
c            erra=abs(vec1(i)-vec2(i))
c         endif
         erra = erra + (vec1(i)-vec2(i))**2
      enddo

      print *, 'erra=',erra,'ra=',ra
c      erra=erra/ra

      if (sqrt(ra)/n .lt. 1d-10) then
         call prin2('vector norm =*', sqrt(ra)/n,1)
         call prin2('switch to absolute error*',a,0)
         erra = sqrt(erra)/n
      else
         erra = sqrt(erra/ra)
      endif
ccc      

      return
      end
c----------------------------------












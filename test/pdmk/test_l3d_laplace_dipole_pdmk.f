c
c     Small validation for the 3D Laplace dipole path in pdmk4.
c
c     The test compares target potentials from pdmk against the direct
c     evaluator in pdmk4.f for three cases:
c
c       1. charges only,
c       2. dipoles only,
c       3. charges and dipoles together.
c
c     The source and target sets are separated randomized point clouds,
c     so no singular self term is present.  This is intended as a cheap
c     compile/run test for DLP support, not a performance test.
c
      implicit real *8 (a-h,o-z)
      integer nd,dim,ikernel,iperiod,ns,nt
      integer ifpgh,ifpghtarg,ifcharge,ifdipole,nhess
      real *8 eps,rpars(10),err0,err1,err2,tinfo(20)
      real *8, allocatable :: sources(:,:),targ(:,:)
      real *8, allocatable :: charge(:,:),dipstr(:,:),rnormal(:,:)
      real *8, allocatable :: pot(:,:),grad(:,:,:),hess(:,:,:)
      real *8, allocatable :: pottarg(:,:),gradtarg(:,:,:)
      real *8, allocatable :: hesstarg(:,:,:)
      real *8, allocatable :: pottargex(:,:),gradtargex(:,:,:)
      real *8, allocatable :: hesstargex(:,:,:)
c
      call prini(6,13)
c
      nd=1
      dim=3
      ikernel=1
      iperiod=0
      ns=320
      nt=280
      eps=1.0d-12
      ifpgh=0
      ifpghtarg=1
      nhess=dim*(dim+1)/2
      rpars(1)=0.0d0
c
      allocate(sources(dim,ns),targ(dim,nt))
      allocate(charge(nd,ns),dipstr(nd,ns),rnormal(dim,ns))
      allocate(pot(nd,ns),grad(nd,dim,ns),hess(nd,nhess,ns))
      allocate(pottarg(nd,nt),gradtarg(nd,dim,nt))
      allocate(hesstarg(nd,nhess,nt))
      allocate(pottargex(nd,nt),gradtargex(nd,dim,nt))
      allocate(hesstargex(nd,nhess,nt))
c
      call fill_l3d_test_data(nd,dim,ns,nt,sources,targ,
     1     charge,dipstr,rnormal)
c
      ifcharge=1
      ifdipole=0
      call run_l3d_dipole_case(nd,dim,eps,ikernel,rpars,iperiod,
     1     ns,sources,ifcharge,charge,ifdipole,rnormal,dipstr,
     2     ifpgh,pot,grad,hess,nt,targ,ifpghtarg,pottarg,gradtarg,
     3     hesstarg,pottargex,gradtargex,hesstargex,tinfo,err0)
c
      ifcharge=0
      ifdipole=1
      call run_l3d_dipole_case(nd,dim,eps,ikernel,rpars,iperiod,
     1     ns,sources,ifcharge,charge,ifdipole,rnormal,dipstr,
     2     ifpgh,pot,grad,hess,nt,targ,ifpghtarg,pottarg,gradtarg,
     3     hesstarg,pottargex,gradtargex,hesstargex,tinfo,err1)
c
      ifcharge=1
      ifdipole=1
      call run_l3d_dipole_case(nd,dim,eps,ikernel,rpars,iperiod,
     1     ns,sources,ifcharge,charge,ifdipole,rnormal,dipstr,
     2     ifpgh,pot,grad,hess,nt,targ,ifpghtarg,pottarg,gradtarg,
     3     hesstarg,pottargex,gradtargex,hesstargex,tinfo,err2)
c
      call prin2('charge only relative error=*',err0,1)
      call prin2('dipole only relative error=*',err1,1)
      call prin2('charge plus dipole relative error=*',err2,1)
c
      if(err0.gt.5.0d-12) then
         write(6,*) 'FAILED: charge-only error too large'
         stop 1
      endif
      if(err1.gt.5.0d-12) then
         write(6,*) 'FAILED: dipole-only error too large'
         stop 1
      endif
      if(err2.gt.5.0d-12) then
         write(6,*) 'FAILED: charge+dipole error too large'
         stop 1
      endif
c
      write(6,*) 'test_l3d_laplace_dipole_pdmk passed'
c
      deallocate(sources,targ,charge,dipstr,rnormal)
      deallocate(pot,grad,hess,pottarg,gradtarg,hesstarg)
      deallocate(pottargex,gradtargex,hesstargex)
c
      end
c
c
c
      subroutine fill_l3d_test_data(nd,dim,ns,nt,sources,targ,
     1     charge,dipstr,rnormal)
      implicit real *8 (a-h,o-z)
      integer nd,dim,ns,nt
      real *8 sources(dim,ns),targ(dim,nt)
      real *8 charge(nd,ns),dipstr(nd,ns),rnormal(dim,ns)
c
      do i=1,ns
         sources(1,i)=0.80d0*hkrand(0)+0.10d0
         sources(2,i)=0.80d0*hkrand(0)+0.10d0
         sources(3,i)=0.80d0*hkrand(0)+0.10d0
         rnormal(1,i)=hkrand(0)-0.50d0
         rnormal(2,i)=hkrand(0)-0.50d0
         rnormal(3,i)=hkrand(0)-0.50d0
         rnorm=sqrt(rnormal(1,i)**2+rnormal(2,i)**2+
     1      rnormal(3,i)**2)
         rnormal(1,i)=rnormal(1,i)/rnorm
         rnormal(2,i)=rnormal(2,i)/rnorm
         rnormal(3,i)=rnormal(3,i)/rnorm
         do id=1,nd
            charge(id,i)=hkrand(0)-0.50d0
            dipstr(id,i)=hkrand(0)-0.50d0
         enddo
      enddo
c
      do i=1,nt
         targ(1,i)=1.20d0+0.35d0*hkrand(0)
         targ(2,i)=0.15d0+0.70d0*hkrand(0)
         targ(3,i)=0.15d0+0.70d0*hkrand(0)
      enddo
c
      return
      end
c
c
c
      subroutine run_l3d_dipole_case(nd,dim,eps,ikernel,rpars,
     1     iperiod,ns,sources,ifcharge,charge,ifdipole,rnormal,
     2     dipstr,ifpgh,pot,grad,hess,nt,targ,ifpghtarg,pottarg,
     3     gradtarg,hesstarg,pottargex,gradtargex,hesstargex,
     4     tinfo,relerr)
      implicit real *8 (a-h,o-z)
      integer nd,dim,ikernel,iperiod,ns,nt
      integer ifcharge,ifdipole,ifpgh,ifpghtarg,nhess
      real *8 eps,rpars(*),sources(dim,ns),targ(dim,nt)
      real *8 charge(nd,ns),dipstr(nd,ns),rnormal(dim,ns)
      real *8 pot(nd,*),grad(nd,dim,*),hess(nd,*)
      real *8 pottarg(nd,nt),gradtarg(nd,dim,nt)
      real *8 hesstarg(nd,*)
      real *8 pottargex(nd,nt),gradtargex(nd,dim,nt)
      real *8 hesstargex(nd,*),tinfo(*),relerr
c
      nhess=dim*(dim+1)/2
      call dzero_l3d(pot,nd*ns)
      call dzero_l3d(grad,nd*dim*ns)
      call dzero_l3d(hess,nd*nhess*ns)
      call dzero_l3d(pottarg,nd*nt)
      call dzero_l3d(gradtarg,nd*dim*nt)
      call dzero_l3d(hesstarg,nd*nhess*nt)
      call dzero_l3d(pottargex,nd*nt)
      call dzero_l3d(gradtargex,nd*dim*nt)
      call dzero_l3d(hesstargex,nd*nhess*nt)
c
      call pdmk(nd,dim,eps,ikernel,rpars,iperiod,ns,sources,
     1     ifcharge,charge,ifdipole,rnormal,dipstr,ifpgh,pot,
     2     grad,hess,nt,targ,ifpghtarg,pottarg,gradtarg,hesstarg,
     3     tinfo)
c
      thresh=1.0d-16
      call kernel_direct(nd,dim,ikernel,rpars,thresh,1,ns,
     1     sources,ifcharge,charge,ifdipole,rnormal,dipstr,1,nt,
     2     targ,ifpghtarg,pottargex,gradtargex,hesstargex)
c
      call derr_l3d(pottargex,pottarg,nd*nt,relerr,rnorm,abserr)
c
      return
      end
c
c
c
      subroutine dzero_l3d(x,n)
      implicit real *8 (a-h,o-z)
      real *8 x(*)
c
      do i=1,n
         x(i)=0.0d0
      enddo
c
      return
      end
c
c
c
      subroutine derr_l3d(x,y,n,relerr,rnorm,abserr)
      implicit real *8 (a-h,o-z)
      real *8 x(*),y(*)
c
      rnorm=0.0d0
      abserr=0.0d0
      do i=1,n
         rnorm=rnorm+x(i)**2
         abserr=abserr+(x(i)-y(i))**2
      enddo
      rnorm=sqrt(rnorm)
      abserr=sqrt(abserr)
      relerr=abserr/rnorm
c
      return
      end

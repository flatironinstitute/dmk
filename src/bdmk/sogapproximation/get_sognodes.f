c*********************************************************************
C
C     get sum-of-Gaussian approximation nodes and weights for the following
C     kernels:
C     ikernel=0: the Yukawa kernel: e^{-beta r}/r for ndim=3;
c                K_0(beta r) for ndim=2
c     ikernel=1: the Laplace kernel: 1/r for ndim=3; log(r) for ndim=2 
C     ikernel=2: the kernel of the square root Laplacian: 1/r^2 for ndim=3;
C                1/r for ndim=2
C
C*********************************************************************
      subroutine get_sognodes(ndim,ikernel,eps,nlevels,norder,beta,
     1    r0,n,ws,ts)
      implicit real *8 (a-h,o-z)
      real *8 eps
      real *8 ws(*),ts(*)

      if (ikernel.eq.0) then
         rmin=1.0d0/(2**nlevels)/norder**2
cccc         rmin=1.0d0/2**16
         rmax=sqrt(3.0d0)
         r0=rmin
         if (ndim.eq.3) then
            call y3dsognodes(beta,rmin,rmax,eps,n,ws,ts)
         elseif (ndim.eq.2) then
            call y2dsognodes(beta,rmin,rmax,eps,n,ws,ts)
         endif

      elseif (ikernel.eq.1) then
         if (ndim.eq.3) then
            call l3dsognodes(eps,n,ws,ts)
         elseif (ndim.eq.2) then
            call l2dsognodes(eps,n,ws,ts)
         endif
         r0=1.0d0/2**16
         
      elseif (ikernel.eq.2) then
         if (ndim.eq.3) then
            call sl3dsognodes(eps,n,ws,ts)
         elseif (ndim.eq.2) then
            call l3dsognodes(eps,n,ws,ts)
         endif
         r0=1.0d0/2**16
         
      endif
c
      return
      end

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
      subroutine get_sognodes(ndim,ikernel,eps,bsize,nlevels,norder,
     1    beta,r0,n,ws,ts)
      implicit real *8 (a-h,o-z)
      real *8 eps
      real *8 ws(*),ts(*)
      complex *16 z, h0, h1, ima
      data ima/(0.0d0,1.0d0)/,pi/0.31415926535897932D+01/
      
      bsize2=bsize*bsize
      
      if (ikernel.eq.0) then
         rmin=bsize/(2**nlevels)/norder**2
cccc         rmin=bsize/2**16
         rmax=sqrt(3.0d0)*bsize
         r0=rmin
c         beta0=beta*bsize
         if (ndim.eq.3) then
            call y3dsognodes(beta,rmin,rmax,eps,n,ws,ts)
c            do i=1,n
c               ws(i)=ws(i)/bsize
c            enddo
         elseif (ndim.eq.2) then
            call y2dsognodes(beta,rmin,rmax,eps,n,ws,ts)
         endif
      elseif (ikernel.eq.1) then
         if (ndim.eq.3) then
            call l3dsognodes(eps,n,ws,ts)
            do i=1,n
               ws(i)=ws(i)/bsize
            enddo
         elseif (ndim.eq.2) then
            call l2dsognodes(eps,n,ws,ts)
         endif
         r0=bsize/2**16
         
      elseif (ikernel.eq.2) then
         if (ndim.eq.3) then
            call sl3dsognodes(eps,n,ws,ts)
            do i=1,n
               ws(i)=ws(i)/bsize2
            enddo
            
         elseif (ndim.eq.2) then
            call l3dsognodes(eps,n,ws,ts)
            do i=1,n
               ws(i)=ws(i)/bsize
            enddo
         endif
         r0=bsize/2**16
         
      endif

      if (ikernel.gt.0) then
         do i=1,n
            ts(i)=ts(i)*bsize2
         enddo
      endif

      
c     testing the accuracy of the SOG approximation
cccc      do k=1,100
cccc         x=r0+(bsize-r0)*(k-1.0d0)/(100-1)
cccc         f1=0
cccc         do i=1,n
cccc            f1=f1+ws(i)*exp(-x*x/ts(i))
cccc         enddo
cccc         if (ikernel.eq.0 .and. ndim.eq.3) then
cccc            f2=exp(-beta*x)/x
cccc         endif
cccc         if (ikernel.eq.0 .and. ndim.eq.2) then
cccc            ifexpon=1
cccc            z=ima*beta*x
cccc            call hank103(z,h0,h1,ifexpon)
cccc            f2 = dble(0.5d0*pi*ima*h0)
cccc         endif
cccc
cccc         if (ikernel.eq.1 .and. ndim.eq.3) then
cccc            f2=1/x
cccc         endif
cccc         if (ikernel.eq.1 .and. ndim.eq.2) then
cccc            f2=log(x)
cccc            do i=1,n
cccc               f2=f2+ws(i)*exp(-1.0d0/ts(i))
cccc            enddo
cccc         endif
cccc         if (ikernel.eq.2 .and. ndim.eq.3) then
cccc            f2=1/x**2
cccc         endif
cccc         if (ikernel.eq.2 .and. ndim.eq.2) then
cccc            f2=1/x
cccc         endif
cccc
cccc         print *, x, f1, f2, abs((f1-f2)/f2)
cccc      enddo
cccc      pause
c
      return
      end

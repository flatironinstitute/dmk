c-----------------------------------------------------------------------------
c     This file contains subroutines determining the proper number terms
c     for the plane-wave expansions used in the DMK.
c
c
c-----------------------------------------------------------------------------
c
C
      subroutine get_PSWF_difference_kernel_pwterms(ikernel,rpars,
     1    beta,ndim,bsize,eps,hpw,npw,ws)
c      
c     Determine the number of terms in the plane wave expansions 
c     for boxes with side length = bsize for the PSWF truncated kernel
c
c     INPUT:
c     bsize: side length of the box
c     eps: desired precision
c    
c     OUTPUT:
c     npw: number of terms in the plane wave expansion
c     hpw : stepsize in the Fourier space
c     ws : weight for the tensor product trapezoidal rule
c
c     
      implicit none

      real *8 rpars(*),rlambda,beta
      real *8 delta,bsize,eps,hpw,ws,pi,b1,hpw0

      integer ikernel,npw,ndigits,ndim,npw2
c
      pi = 4.0d0*atan(1.0d0)
      ndigits=nint(log10(1.0d0/eps)-0.1)

      if (ikernel.eq.2.and.ndim.eq.3) then
         if (ndigits.le.3) then
            npw=13
            hpw = pi*0.662d0/bsize
         elseif (ndigits.le.6) then
            npw=27
            hpw = pi*0.667d0/bsize
         elseif (ndigits.le.9) then
            npw=39
            hpw = pi*0.6625d0/bsize
         elseif (ndigits.le.12) then
            npw=55
            hpw = pi*0.667d0/bsize
         endif
      else
         if (ndigits.le.3) then
            npw=13
            hpw = pi*0.662d0/bsize
         elseif (ndigits.le.6) then
            npw=25
            hpw = pi*0.6686d0/bsize
         elseif (ndigits.le.9) then
            npw=39
            hpw = pi*0.6625d0/bsize
         elseif (ndigits.le.12) then
            npw=53
            hpw = pi*0.6677d0/bsize
         endif
      endif

      ws = hpw**ndim/pi**(ndim-1)/2
      
      return
      end 
c
c
c
c
      subroutine get_PSWF_truncated_kernel_pwterms(ikernel,rpars,
     1    beta,ndim,bsize,eps,hpw,npw,ws,rl)
c      
c     Determine the number of terms in the plane wave expansions 
c     for boxes with side length = bsize for the PSWF truncated kernel
c
c     INPUT:
c     bsize: side length of the box
c     eps: desired precision
c    
c     OUTPUT:
c     npw: number of terms in the plane wave expansion
c     hpw : stepsize in the Fourier space
c     ws : weight for the tensor product trapezoidal rule
c     rl : the radius of the truncation ball
c
c     
      implicit none
      integer ikernel
      integer npw,ndigits,ndim,npw2

      real *8 rpars(*),rlambda,beta
      real *8 delta,bsize,eps,hpw,ws,rl,pi,b1,hpw0
c
      pi = 4.0d0*atan(1.0d0)
      ndigits=nint(log10(1.0d0/eps)-0.1)

      if (ndigits.le.3) then
         npw=13
c         hpw = pi*0.36d0/bsize
         hpw = pi*0.34d0/bsize
      elseif (ndigits.le.6) then
         npw=25
c         hpw = pi*0.362d0/bsize
         hpw = pi*0.357d0/bsize
      elseif (ndigits.le.9) then
         npw=39
c         hpw = pi*0.363d0/bsize
         hpw = pi*0.357d0/bsize
      elseif (ndigits.le.12) then
         npw=53
c         hpw = pi*0.358d0/bsize
         hpw = pi*0.338d0/bsize
      endif

c      if (ikernel.eq.0) then
c         rlambda=rpars(1)
c         npw2=npw/2
c         hpw0=sqrt((beta/bsize)**2-rlambda**2)/npw2
c         print *, hpw, hpw0
c         if (hpw0.lt.hpw) hpw=(hpw0+hpw)/2
c      endif
      
      ws = hpw**ndim/pi**(ndim-1)/2
c      rl = bsize*(sqrt(ndim*1.0d0)+1)
      rl = bsize*sqrt(ndim*1.0d0)*2

      return
      end 
c
c
c
c

      

c*********************************************************************
C
C     get sum-of-Gaussian approximation nodes and weights for 
c     the 2D Yukawa kernel with parameter beta on the interval [rmin,rmax]
c     to the prescribed precision eps
C
C     algorithm: truncated trapezoidal rule 
C
C
C*********************************************************************
      subroutine y2dsognodes(beta,rmin,rmax,eps,n,ws,ts)
      implicit real *8 (a-h,o-z)
      real *8 eps
      real *8 ws(*),ts(*)
      real *8 pi

      pi = 4*atan(1.0d0)
      
      h = 0.72d0*2*pi/log(1.0d0/eps)

      tlower = -log(2.0d0/beta*sqrt(log(1.0d0/eps)))
      tupper = log(rmax/rmin*sqrt(log(1.0d0/eps)))

      mmin=floor(tlower/h)
      mmax=ceiling(tupper/h)

      n=0
      do i=mmin,mmax
         n=n+1
         ts(n)=exp(-2*h*i)
         ws(n)=exp(-beta**2/4*ts(n))*h
      enddo
c
      return
      end

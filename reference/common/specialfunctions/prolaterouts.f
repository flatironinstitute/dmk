c-----------------------------------------------------------------------------
c     This file contains additional subroutines for the prolates that are used in
c     the MFM.
c
c
c-----------------------------------------------------------------------------
c
c
c
      subroutine prolate_intvals(beta,wprolate,c0,c1,c2,c4)
c     calculates two integrals of the canonical prolate function. By "canonical" we mean
c     that the prolate function is supported on [-1,1].
c
c     input:
c     beta - the parameter in the \psi_0^\beta function
c      
c     output:
c     c0 = \int_0^1 \psi_0^\beta(x)dx
c     c1 = \int_0^1 x \psi_0^\beta(x)dx
c     c2 = \int_0^1 x^2 \psi_0^\beta(x)dx
c     c4 = \int_0^1 x^4 \psi_0^\beta(x)dx
c      
      implicit none
      real *8 xs(1000),ws(1000),fvals(1000)
      integer itype,npts,i
      real *8 beta,c0,c1,c2,c4,u,v,w2
      real *8 wprolate(*),psi0,derpsi0
      
      npts=200
      itype=1
      call legeexps(itype,npts,xs,u,v,ws)

c     scale the nodes and weights to [0,1]
      do i=1,npts
         xs(i)=(xs(i)+1)/2
         ws(i)=ws(i)/2
      enddo
      
      w2=1.0d0
      do i=1,npts
         call prol0eva(xs(i),wprolate,fvals(i),derpsi0)         
      enddo
      
      c0=0
      c1=0
      c2=0
      c4=0
      do i=1,npts
         c0=c0+ws(i)*fvals(i)
         c2=c2+ws(i)*fvals(i)*xs(i)*xs(i)
         c4=c4+ws(i)*fvals(i)*xs(i)**4
         c1=c1+ws(i)*fvals(i)*xs(i)
      enddo
      c2=c2/c0
      
      return
      end
c
c
c
c
      subroutine prolate_fourier_transform(nk,rk,beta,wprolate,fhat)
c     calculates the Fourier transform of the ES function for a given set of frequency values
c     input:
c     nk - number of frequencies
c     rk - frequency values
c     beta - the parameter in the ES function
c     
c     output:
c     fhat = \int_{-1}^1 e^{-i k x} \psi_0^\beta(x) dx, the interval is [-1,1]
c     
c     Since the ES function is even, we have
c     fhat = 2\int_0^1 cos(kx) ES(x) dx.
c     Thus, fhat is actually real
c
      implicit none
      integer nk
      real *8 rk(nk),beta,u,v,w2,derpsi0
      real *8 xs(1000),ws(1000),fvals(1000)
      real *8 fhat(nk),wprolate(*)
      integer npts, itype, i, j
      
      npts=400
      itype=1
      call legeexps(itype,npts,xs,u,v,ws)

c     scale the nodes and weights to [0,1]
      do i=1,npts
         xs(i)=(xs(i)+1)/2
         ws(i)=ws(i)/2
      enddo
      
      do i=1,npts
         call prol0eva(xs(i),wprolate,fvals(i),derpsi0)         
      enddo
      
      do i=1,nk
         fhat(i)=0
         do j=1,npts
            fhat(i)=fhat(i)+ws(j)*fvals(j)*cos(rk(i)*xs(j))
         enddo
         fhat(i)=fhat(i)*2
      enddo
      
      return
      end
c
c
c
c
      subroutine prolate_localkernel(x,fval)
      implicit none
c     computes the ES local kernel 
c
c           1-\int_0^x exp(-beta(1-\sqrt(1-u^2)))du /c0,
c
c     where c0 is the value of the integral on [0,1].
c     
c     This subroutine has about 8 digits of accuracy (in relative l2 norm)
c     for only one value of beta, i.e., beta=1.249142003432574e+01, which is
c     for 6 digits of accuracy
c     calculation. To be more precise, the rational approximation is only valid
c     for the aforementioned value of beta!
c
c     input:
c     beta - not used
c
c     output:
c     fval - the value of the ES local kernel
c
      real *8 a(7),b(6),beta,fval,x,num,den
      integer i
      data a/2.513540716069032d-1,-1.783570920416137d0,
     1    5.292235895245520d0,-8.382628118685370d0,7.461664313878896d0,
     2    -3.534673594991488d0,6.956183729024348d-1/
      data b/-2.352996023078147d0,3.779441281768189d0,
     1    -3.677615294816419d0,3.066300645062891d0,-1.510642863443010d0,
     2    6.956183798638753d-1/

      num=a(1)*x+a(2)
      do i=3,7
         num=num*x+a(i)
      enddo
      den=x+b(1)
      do i=2,6
         den=den*x+b(i)
      enddo

      fval=num/den

      return
      end
c
c
c
c

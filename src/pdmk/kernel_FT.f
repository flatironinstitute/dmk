c
c     Compute the radial Fourier transform 
c     of the windowed and difference kernels
c     of various kernels in two and three dimensions. The kernel splitting 
c     is done with the prolate spheroidal wave functions.
c
c      
c      
c
      subroutine get_windowed_kernel_Fourier_transform(ikernel,
     1    rpars,dim,beta,
     2    bsize,rl,npw,hpw,ws,wprolate,nfourier,fhat)
      implicit none
      integer dim,npw,nfourier,ikernel,i
      real *8 rpars(*),beta,bsize,rl,hpw,ws
      real *8 wprolate(*)
      real *8 xi(0:nfourier)
      real *8 fhat(0:nfourier)

      do i=0,nfourier
         xi(i)=sqrt(1.0d0*i)*hpw
      enddo
      
      if (ikernel.eq.0) then
         call yukawa_windowed_kernel_Fourier_transform(dim,rpars,
     1    beta,bsize,rl,npw,hpw,ws,wprolate,nfourier,fhat)
      elseif (ikernel .eq.1) then
         if (dim.eq.2) then
            call log_windowed_kernel_Fourier_transform(dim,beta,
     1          bsize,rl,npw,hpw,ws,wprolate,nfourier,fhat)
         elseif (dim.eq.3) then
            call l3d_windowed_kernel_Fourier_transform(dim,beta,
     1          bsize,rl,npw,hpw,ws,wprolate,nfourier,fhat)
         endif
      elseif (ikernel.eq.2) then
         if (dim.eq.2) then
            call sl2d_windowed_kernel_Fourier_transform(dim,beta,
     1          bsize,rl,npw,hpw,ws,wprolate,nfourier,fhat)
         elseif (dim.eq.3) then
            call sl3d_windowed_kernel_Fourier_transform(dim,beta,
     1          bsize,rl,npw,hpw,ws,wprolate,nfourier,fhat)
         endif
      elseif (ikernel.eq.3) then
         call stokes_windowed_kernel_Fourier_transform(dim,beta,
     1          bsize,rl,xi,wprolate,nfourier+1,fhat)
         do i=0,nfourier
            fhat(i)=fhat(i)*ws
         enddo
      endif

      return
      end
c      
c      
c      
c      
      subroutine get_difference_kernel_Fourier_transform(ikernel,
     1    rpars,dim,beta,
     2    bsizesmall,bsizebig,npw,hpw,ws,wprolate,nfourier,fhat)
      implicit none
      integer dim,npw,nfourier,ikernel,i
      real *8 rpars(*),beta,bsizesmall,bsizebig,hpw,ws
      real *8 wprolate(*)
      real *8 xi(0:nfourier)
      real *8 fhat(0:nfourier)

      do i=0,nfourier
         xi(i)=sqrt(1.0d0*i)*hpw
      enddo
      
      if (ikernel.eq.0) then
         call yukawa_difference_kernel_Fourier_transform(dim,rpars,
     1    beta,bsizesmall,bsizebig,npw,hpw,ws,wprolate,nfourier,fhat)
      elseif (ikernel .eq.1) then
         if (dim.eq.2) then
            call log_difference_kernel_Fourier_transform(dim,beta,
     1          bsizesmall,bsizebig,npw,hpw,ws,wprolate,nfourier,fhat)
         elseif (dim.eq.3) then
            call l3d_difference_kernel_Fourier_transform(dim,beta,
     1          bsizesmall,bsizebig,npw,hpw,ws,wprolate,nfourier,fhat)
         endif
      elseif (ikernel.eq.2) then
         if (dim.eq.2) then
            call sl2d_difference_kernel_Fourier_transform(dim,beta,
     1          bsizesmall,bsizebig,npw,hpw,ws,wprolate,nfourier,fhat)
         elseif (dim.eq.3) then
            call sl3d_difference_kernel_Fourier_transform(dim,beta,
     1          bsizesmall,bsizebig,npw,hpw,ws,wprolate,nfourier,fhat)
         endif
      elseif (ikernel.eq.3) then
         call stokes_difference_kernel_Fourier_transform(dim,beta,
     1       bsizesmall,bsizebig,xi,wprolate,nfourier+1,fhat)
         do i=0,nfourier
            fhat(i)=fhat(i)*ws
         enddo
      endif

      return
      end
c      
c      
c      
c      
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      
c      
c     1/r in two dimensions
c      
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      
      subroutine sl2d_difference_kernel_Fourier_transform(dim,beta,
     1    bsizesmall,bsizebig,npw,hpw,ws,wprolate,nfourier,fhat)
c
c     compute the Fourier transform of the difference kernel
c     of the 1/r kernel in two dimensions
c
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 beta,bsizesmall,bsizebig,hpw,ws
      real *8 ts(-npw/2:(npw-1)/2)

      real *8 wprolate(*)

      real *8 xs(1000),whts(1000),coefs(1000)
      real *8 x1(1000),w1(1000),fv1(1000)
      real *8 x2(1000),w2(1000),fv2(1000)
      real *8 xval,fval,psi0,derpsi0
      real *8 fhat(0:nfourier)
      complex *16 z,h0,h1

      call prolate_intvals(beta,wprolate,c0,c1,g0d2,c4)

      iw=wprolate(1)
      nterms=wprolate(5)
      call legeinte(wprolate(iw),nterms,coefs)      
      coefs(1)=0.0d0

      itype = 1
      nquad = 100
      call legeexps(itype,nquad,xs,u,v,whts)

      do i=1,nquad
         x2(i)=(xs(i)+1)/2*bsizebig
         w2(i)=whts(i)/2*bsizebig
      enddo

      do i=1,nquad
         xval=(xs(i)+1)/2
         call legeexev(xval,fval,coefs,nterms)
         xval2=xval*2
         if (xval2.lt.1) then
            call legeexev(xval2,fval2,coefs,nterms)
         else
cccc            call legeexev(1.0d0,fval2,coefs,nterms)
            fval2=c0
         endif
         fv1(i)=fval
         fv2(i)=fval2
      enddo

      ifexpon=1
      do i=0,nfourier
         rk=sqrt(1.0d0*i)*hpw
         rk2=rk*rk
         fhat(i)=0
         do j=1,nquad
            z=rk*x2(j)
            if (abs(z).lt.1d-15) then
               dj0=1.0d0
            else
               call hank103(z,h0,h1,ifexpon)
               dj0=dble(h0)
            endif
            fhat(i)=fhat(i)+dj0*(fv2(j)-fv1(j))*w2(j)/c0
         enddo
         fhat(i)=fhat(i)*ws
      enddo

      return
      end
c
c
c
c
      subroutine sl2d_windowed_kernel_Fourier_transform(dim,beta,
     1    bsize,rl,npw,hpw,ws,wprolate,nfourier,fhat)
c
c     compute the Fourier transform of the windowed kernel
c     of the 1/r kernel in two dimensions
c
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 beta,bsizesmall,bsizebig,hpw,ws

      real *8 wprolate(*)

      real *8 xs(1000),whts(1000),coefs(1000),fvals(1000)
      real *8 xval,fval,psi0,derpsi0
      real *8 fhat(0:nfourier)
      complex *16 z,h0,h1

      call prolate_intvals(beta,wprolate,c0,c1,g0d2,c4)

      iw=wprolate(1)
      nterms=wprolate(5)
      call legeinte(wprolate(iw),nterms,coefs)      
      coefs(1)=0.0d0

      itype = 1
      nquad = 200
      call legeexps(itype,nquad,xs,u,v,whts)
      do i=1,nquad
         xs(i)=(xs(i)+1)/4*bsize*rl*3
         whts(i)=whts(i)/4*bsize*rl*3
      enddo

      do i=1,nquad
         xval0=xs(i)/bsize
         if (abs(xval0).lt.1.0d0) then
            call legeexev(xval0,fval0,coefs,nterms)
         elseif (xval0.ge.1.0d0) then
            fval0=c0
         elseif (xval0.le.-1.0d0) then
            fval0=-c0
         endif

         xval1=(rl+xs(i))/bsize
         if (abs(xval1).lt.1.0d0) then
            call legeexev(xval1,fval1,coefs,nterms)
            fvals(i)=fval0
         elseif (xval1.ge.1.0d0) then
            fval1=c0
         elseif (xval1.le.-1.0d0) then
            fval1=-c0
         endif
         
         xval2=(rl-xs(i))/bsize
         if (abs(xval2).lt.1.0d0) then
            call legeexev(xval2,fval2,coefs,nterms)
         elseif (xval2.ge.1.0d0) then
            fval2=c0
         elseif (xval2.le.-1.0d0) then
            fval2=-c0
         endif

         fvals(i)=fval0-0.5d0*fval1+0.5d0*fval2
      enddo

      ifexpon=1
      do i=0,nfourier
         rk=sqrt(i*1.0d0)*hpw
         fhat(i)=0
         do j=1,nquad
            z=rk*xs(j)
            if (i.eq.0) then
               dj0=1.0d0
            else
               call hank103(z,h0,h1,ifexpon)
               dj0=dble(h0)
            endif
            fhat(i)=fhat(i)+dj0*fvals(j)*whts(j)/c0
         enddo
         fhat(i)=fhat(i)*ws
      enddo
         
      return
      end
c
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      
c      
c     1/r in three dimensions
c      
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      
      subroutine l3d_difference_kernel_Fourier_transform(dim,beta,
     1    bsizesmall,bsizebig,npw,hpw,ws,wprolate,nfourier,fhat)
c
c     compute the Fourier transform of the difference kernel
c     of the 1/r kernel in three dimensions
c
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 beta,bsizesmall,bsizebig,hpw,ws
      real *8 wprolate(*)

      real *8 xs(1000),whts(1000),fvals(1000)
      real *8 x1(1000),x2(1000)
      
      real *8 xval,fval,psi0,derpsi0
      real *8 fhat(0:nfourier)

      call prolate_intvals(beta,wprolate,c0,c1,g0d2,c4)

      itype = 1
      nquad = 100
      call legeexps(itype,nquad,xs,u,v,whts)

      do i=1,nquad
         x1(i)=(xs(i)+1)/2*bsizesmall         
         x2(i)=(xs(i)+1)/2*bsizebig
      enddo

      do i=1,nquad
         xval=(xs(i)+1)/2
         call prol0eva(xval,wprolate,fval,derpsi0)
         fvals(i)=fval*whts(i)/2/c0
      enddo

      bsizesmall2=bsizesmall*bsizesmall
      bsizebig2=bsizebig*bsizebig

      do i=0,nfourier
         rk=sqrt(1.0d0*i)*hpw
         rk2=rk*rk
         fhat(i)=0
         do j=1,nquad
            fhat(i)=fhat(i)+(cos(rk*x1(j))-cos(rk*x2(j)))*fvals(j)
         enddo
c     for symmetric trapezoidal rule
         if (i .gt. 0) then
            fhat(i)=fhat(i)*ws/rk2
         else
            fhat(i)=ws*g0d2*(bsizebig2-bsizesmall2)/2
         endif
      enddo

      return
      end
c
c
c
c
      subroutine l3d_windowed_kernel_Fourier_transform(dim,beta,
     1    bsize,rl,npw,hpw,ws,wprolate,nfourier,fhat)
c
c     compute the Fourier transform of the windowed kernel
c     of the 1/r kernel in three dimensions
c
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 beta,bsizesmall,bsizebig,hpw,ws

      real *8 wprolate(*)

      real *8 xs(1000),whts(1000),fvals(1000)
      real *8 xval,fval,psi0,derpsi0
      real *8 fhat(0:nfourier)

      call prolate_intvals(beta,wprolate,c0,tmp,g0d2,c4)
      c1 = c0*bsize


      itype = 1
      nquad = 100
      call legeexps(itype,nquad,xs,u,v,whts)
      do i=1,nquad
         xs(i)=(xs(i)+1)/2*bsize
         whts(i)=whts(i)/2*bsize
      enddo

      do i=1,nquad
         xval=xs(i)/bsize
         call prol0eva(xval,wprolate,fvals(i),derpsi0)
         fvals(i)=fvals(i)*whts(i)/c1
      enddo

      do i=0,nfourier
         rk=sqrt(i*1.0d0)*hpw
         fhat(i)=0
         do j=1,nquad
            fhat(i)=fhat(i)+cos(rk*xs(j))*fvals(j)
         enddo
         if (i .gt. 0) then
            fhat(i)=fhat(i)*ws*2*(sin(rk*rl/2)/rk)**2
         else
            fhat(i)=fhat(i)*ws*rl**2/2
         endif
      enddo
      
      return
      end
c
c
c
c
c
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      
c      
c     1/r^2 in three dimensions
c      
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      
      subroutine sl3d_difference_kernel_Fourier_transform(dim,beta,
     1    bsizesmall,bsizebig,npw,hpw,ws,wprolate,nfourier,fhat)
c
c     compute the Fourier transform of the difference kernel
c     of the 1/r kernel in three dimensions
c
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 beta,bsizesmall,bsizebig,hpw,ws
      real *8 wprolate(*)

      real *8 coefs0(0:1000),coefs(1000)
      real *8 xs(1000),whts(1000),fvals(1000)
      real *8 x1(1000),x2(1000),w2(1000),fv1(1000),fv2(1000)
      
      real *8 xval,fval,psi0,derpsi0
      real *8 fhat(0:nfourier)

      iw=wprolate(1)
      nterms=wprolate(5)

c     calculate Legendre expansion coefficients of x\psi_0^c(x)
      do i=0,nterms
         coefs0(i)=0
      enddo
      
      do i=nterms,2,-1
         coefs0(i)=coefs0(i)+wprolate(iw+i-1)*i/(2*i-1.0d0)
         coefs0(i-2)=coefs0(i-2)+wprolate(iw+i-1)*(i-1)/(2*i-1.0d0)
      enddo
      coefs0(1)=coefs0(1)+wprolate(iw)
      
c     calculate Legendre expansion coefficients of \int_0^x t\psi_0^c(t)dt
      call legeinte(coefs0,nterms,coefs)      
      nterms=nterms+1
      call legeexev(0.0d0,fval,coefs,nterms)
      coefs(1)=coefs(1)-fval
      done=1.0d0
      call legeexev(done,fval,coefs,nterms)
      c0=fval

      
      itype = 1
      nquad = 100
      call legeexps(itype,nquad,xs,u,v,whts)
      
      do i=1,nquad
         x1(i)=(xs(i)+1)/2*bsizesmall         
         x2(i)=(xs(i)+1)/2*bsizebig
         w2(i)=whts(i)/2*bsizebig
      enddo

      do i=1,nquad
         xval=(xs(i)+1)/2
         call legeexev(xval,fval,coefs,nterms)
         xval2=xval*2
         if (xval2.lt.1) then
            call legeexev(xval2,fval2,coefs,nterms)
         else
cccc            call legeexev(1.0d0,fval2,coefs,nterms)
            fval2=c0
         endif
         fv1(i)=fval
         fv2(i)=fval2
      enddo

      do i=0,nfourier
         rk=sqrt(1.0d0*i)*hpw
         rk2=rk*rk
         fhat(i)=0
         if (i.gt.0) then
            do j=1,nquad
               fhat(i)=fhat(i)+(fv2(j)-fv1(j))*w2(j)
     1             *sin(rk*x2(j))/rk/x2(j)
            enddo
         else
            do j=1,nquad
               fhat(i)=fhat(i)+(fv2(j)-fv1(j))*w2(j)
            enddo
         endif

         fhat(i)=fhat(i)*ws/c0
      enddo

      return
      end
c
c
c
c
      subroutine sl3d_windowed_kernel_Fourier_transform(dim,beta,
     1    bsize,rl,npw,hpw,ws,wprolate,nfourier,fhat)
c
c     compute the Fourier transform of the windowed kernel
c     of the 1/r^2 kernel in three dimensions
c
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 beta,bsizesmall,bsizebig,hpw,ws

      real *8 wprolate(*)

      real *8 coefs0(0:1000),coefs1(1000),coefs2(1000),fvals(1000)
      real *8 xs(1000),whts(1000),x1(1000),w1(1000),x2(1000),w2(1000)
      real *8 xval,fval,psi0,derpsi0,fv1(1000),fv2(1000)
      real *8 fhat(0:nfourier)

      iw=wprolate(1)
      nterms=wprolate(5)

c     calculate Legendre expansion coefficients of x\psi_0^c(x)
      do i=0,nterms
         coefs0(i)=0
      enddo
      
      do i=nterms,2,-1
         coefs0(i)=coefs0(i)+wprolate(iw+i-1)*i/(2*i-1.0d0)
         coefs0(i-2)=coefs0(i-2)+wprolate(iw+i-1)*(i-1)/(2*i-1.0d0)
      enddo
      coefs0(1)=coefs0(1)+wprolate(iw)
      
c     calculate Legendre expansion coefficients of \int_0^x t\psi_0^c(t)dt
      call legeinte(coefs0,nterms,coefs1)      
      call legeexev(0.0d0,fval,coefs1,nterms+1)
      coefs1(1)=coefs1(1)-fval
      done=1.0d0
      call legeexev(done,fval,coefs1,nterms+1)
      c1=fval
      
      
c     calculate Legendre expansion coefficients of \int_0^x \psi_0^c(t)dt
      call legeinte(wprolate(iw),nterms,coefs2)      
      call legeexev(done,fval,coefs2,nterms)
      c0=fval
      call legeexev(-done,fval,coefs2,nterms)

      dl=rl/2
      
      itype = 1
      nquad = 200
      call legeexps(itype,nquad,xs,u,v,whts)
      do i=1,nquad
         xs(i)=(xs(i)+1)/2*dl*3
         whts(i)=whts(i)/2*dl*3
      enddo

      do i=1,nquad
         xval1=xs(i)/bsize
         if (xval1.lt.1.0d0) then
            call legeexev(xval1,fval1,coefs1,nterms)
         elseif (xval1.ge.1.0d0) then
            fval1=c1
         endif

c        window function
         xval0=(xs(i)-dl-bsize)/bsize
         if (abs(xval0).lt.1.0d0) then
            call legeexev(xval0,fval0,coefs2,nterms)
         elseif (xval0.ge.1.0d0) then
            fval0=c0
         elseif (xval0.le.-1.0d0) then
            fval0=0
         endif
         
         fvals(i)=fval1/c1*(1-fval0/c0)
      enddo

      do i=0,nfourier
         rk=sqrt(i*1.0d0)*hpw
         fhat(i)=0
         do j=1,nquad
            if (i.eq.0) then
               fhat(i)=fhat(i)+fvals(j)*whts(j)
            else
               fhat(i)=fhat(i)+fvals(j)*whts(j)
     1             *sin(rk*xs(j))/(rk*xs(j))
            endif
         enddo
         fhat(i)=fhat(i)*ws
      enddo

      return
      end
c
c
c
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      
c      
c     log(r) in two dimensions
c      
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      
      subroutine log_difference_kernel_Fourier_transform(dim,beta,
     1    bsizesmall,bsizebig,npw,hpw,ws,wprolate,nfourier,fhat)
c
c     compute the Fourier transform of the difference kernel
c     of the log(r) kernel in two dimensions
c
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 beta,bsizesmall,bsizebig,hpw,ws
      real *8 wprolate(*)

      real *8 xval,fval,psi0,derpsi0
      real *8 fhat(0:nfourier)

      call prolate_intvals(beta,wprolate,c0,c1,g0d2,c4)

      zero=0.0d0
      call prol0eva(zero,wprolate,psi0,derpsi0)
      
      bsizesmall2=bsizesmall*bsizesmall
      bsizebig2=bsizebig*bsizebig

      do i=0,nfourier
         if (i.eq.0) then
            fhat(i)=ws*g0d2*(bsizesmall2-bsizebig2)/2
         else
            rk=sqrt(1.0d0*i)*hpw
            rk2=rk*rk

            xval=rk*bsizesmall/beta
            if (xval.lt.1) then
               call prol0eva(xval,wprolate,fval1,derpsi0)
            else
               fval1=0.0d0
            endif
         
            xval=rk*bsizebig/beta
            if (xval.le.1) then
               call prol0eva(xval,wprolate,fval2,derpsi0)
            else
               fval2=0.0d0
            endif
            fhat(i)=(fval2-fval1)/psi0/rk2
            fhat(i)=ws*fhat(i)
         endif
      enddo

      return
      end
c
c
c
c
      subroutine log_windowed_kernel_Fourier_transform(dim,beta,
     1    bsize,rl,npw,hpw,ws,wprolate,nfourier,fhat)
c
c     compute the Fourier transform of the windowed kernel
c     of the log kernel in two dimensions
c
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 beta,bsizesmall,bsizebig,hpw,ws

      real *8 wprolate(*)

      real *8 xs(1000),whts(1000),fvals(1000)
      real *8 xval,fval,psi0,derpsi0
      real *8 fhat(0:nfourier)
      complex *16 z,h0,h1
      
      zero=0.0d0
      call prol0eva(zero,wprolate,psi0,derpsi0)
      
      ifexpon=1
      dfac=rl*log(rl)
      do i=0,nfourier
         if (i.eq.0) then
            fhat(i)=ws*(-0.25d0*rl*rl+0.5d0*dfac*rl)
         else
            rk=sqrt(i*1.0d0)*hpw
            rk2=rk*rk
            
            xval=rk*bsize/beta
            if (xval.lt.1) then
               call prol0eva(xval,wprolate,fval,derpsi0)
            else
               fval=0.0d0
            endif
            z=rl*rk
            call hank103(z,h0,h1,ifexpon)
            dj0=dble(h0)
            dj1=dble(h1)

            tker=-(1-dj0)/rk2+dfac*dj1/rk
            
            fhat(i)=ws*tker*fval/psi0
         endif
      enddo

      return
      end
c
c
c
c
c
c
      subroutine log_windowed_kernel(dim,beta,
     1    bsize,rl,wprolate,rval,fval)
c
c     compute the value of the windowed kernel
c     of the log kernel in two dimensions
c
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 beta,bsize

      real *8 wprolate(*)

      real *8 xs(1000),whts(1000),fvals(1000)
      real *8 xval,fval,psi0,derpsi0
      complex *16 z,h0,h1

      zero=0.0d0
      call prol0eva(zero,wprolate,psi0,derpsi0)

      itype = 1
      nquad = 100
      call legeexps(itype,nquad,xs,u,v,whts)
      do i=1,nquad
         xs(i)=(xs(i)+1)/2*beta/bsize
         whts(i)=whts(i)/2*beta/bsize
      enddo

      ifexpon=1
      dfac=rl*log(rl)
      fval=0.0d0
      do i=1,nquad
         xval=xs(i)*bsize/beta
         call prol0eva(xval,wprolate,fval0,derpsi0)

         z=rl*xs(i)
         call hank103(z,h0,h1,ifexpon)
         dj0=dble(h0)
         dj1=dble(h1)

         tker=-(1-dj0)/xs(i)/xs(i)+dfac*dj1/xs(i)
            
         fhat=tker*fval0/psi0
         if (rval.gt.0) then
            z=rval*xs(i)
            call hank103(z,h0,h1,ifexpon)
            dj0=dble(h0)
         else
            dj0=1.0d0
         endif

         fval=fval+fhat*dj0*whts(i)*xs(i)
      enddo

      return
      end
c
c
c
c
c
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      
c      
c     Yukawa in two and three dimensions (K_0(lambda r) and exp(-lambda r)/r)
c      
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      
      subroutine yukawa_difference_kernel_Fourier_transform(dim,rpars,
     1    beta,bsizesmall,bsizebig,npw,hpw,ws,wprolate,nfourier,fhat)
c
c     compute the Fourier transform of the difference kernel
c     of the Yukawa kernel in two and three dimensions
c
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 beta,bsizesmall,bsizebig,hpw,ws
      real *8 rpars(*),wprolate(*)

      real *8 xval,fval,psi0,derpsi0
      real *8 fhat(0:nfourier)

      call prolate_intvals(beta,wprolate,c0,c1,g0d2,c4)
      
      rlambda=rpars(1)
      rlambda2=rlambda*rlambda
      
      zero=0.0d0
      call prol0eva(zero,wprolate,psi0,derpsi0)

      one=1.0d0
      call prol0eva(one,wprolate,fone,derpsi0)
      
      bsizesmall2=bsizesmall*bsizesmall
      bsizebig2=bsizebig*bsizebig

      do i=0,nfourier
         rk=sqrt(1.0d0*i)*hpw
         xi2=rk*rk+rlambda2
         xi=sqrt(xi2)
            
         xval=xi*bsizesmall/beta
         if (xval.le.1) then
            call prol0eva(xval,wprolate,fval1,derpsi0)
         else
            fval1=0
         endif
         
         xval=xi*bsizebig/beta
         if (xval.le.1) then
            call prol0eva(xval,wprolate,fval2,derpsi0)
         else
            fval2=0
         endif
         
         fhat(i)=ws*(fval1-fval2)/psi0/xi2
      enddo

c     the following lines compute fhat(0) accurately when there is
c     a low-frequency breakdown
      if (rlambda*bsizebig/beta .lt. 1.0d-4) then
         tmp=ws*g0d2*(bsizebig2-bsizesmall2)/2
     1       +ws*(bsizesmall**4-bsizebig**4)*rlambda2*c4/c0/24
c         print *, rlambda*bsizebig/beta
c         print *, fhat(0),tmp,fhat(0)-tmp
c         pause
         fhat(0)=tmp
      endif
      
      return
      end
c
c
c
c
      subroutine yukawa_windowed_kernel_Fourier_transform(dim,rpars,
     1    beta,bsize,rl,npw,hpw,ws,wprolate,nfourier,fhat)
c
c     compute the Fourier transform of the windowed kernel
c     of the Yukawa kernel in two and three dimensions
c
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 beta,bsizesmall,bsizebig,hpw,ws

      real *8 rpars(*),wprolate(*)

      real *8 xs(1000),whts(1000),fvals(1000)
      real *8 xval,fval,psi0,derpsi0
      real *8 fhat(0:nfourier)

      real *8 besk0,besk1,besj0,besj1
      external besk0,besk1,besj0,besj1
      
      rlambda=rpars(1)
      rlambda2=rlambda*rlambda
c     determine whether one needs to smooth out the 1/(k^2+lambda^2) factor
c     at the origin
      dtmp = rlambda*bsize/beta
      ifnearcorrection=0
      if (dtmp.lt.1.0d-2) then
         ifnearcorrection=1
      endif
c     needed in the calculation of kernel-smoothing when
c     there is low-frequency breakdown
      if (ifnearcorrection.eq.1) then
         if (dim.eq.2) then
            dk0=besk0(rl*rlambda)
            dk1=besk1(rl*rlambda)
         endif
         if (dim.eq.3) then
            delam=exp(-rl*rlambda)
         endif
      endif
      
      zero=0.0d0
      call prol0eva(zero,wprolate,psi0,derpsi0)

      one=1.0d0
      call prol0eva(one,wprolate,fone,derpsi0)

      do i=0,nfourier
         rk=sqrt(i*1.0d0)*hpw
         xi2=rk*rk+rlambda2
         xi=sqrt(xi2)
         
         xval=xi*bsize/beta
         if (xval.le.1) then
            call prol0eva(xval,wprolate,fval,derpsi0)
         else
            fval=0
         endif
         
         fhat(i)=ws*fval/psi0/xi2

c     deal with low-frequency breakdown
c     use truncated kernel trick to smooth the kernel
c     when the kernel is singular or nearly singular!!
         if (ifnearcorrection.eq.1) then
            if (dim.eq.2) then
               xsc=rl*rk
               sker=-rl*rlambda*besj0(xsc)*dk1
     1             +1+xsc*besj1(xsc)*dk0
            elseif (dim.eq.3) then
               xsc=rl*rk
               sker=1-delam*(cos(xsc)+rlambda/rk*sin(xsc))
            endif               
            fhat(i)=fhat(i)*sker
         endif         
      enddo

      return
      end
c
c
c
c
c
c
      subroutine yukawa_windowed_kernel_value_at_zero(dim,rpars,beta,
     1    bsize,rl,wprolate,fval)
c
c     compute the value of the windowed kernel at the origin
c     of the Yukawa kernel in two and three dimensions
c
c     Output: T_l(0) for self interactions
c      
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 beta,bsize

      real *8 rpars(*),wprolate(*)

      real *8 xs(1000),whts(1000),fvals(1000)
      real *8 xval,fval,psi0,derpsi0
      real *8 besk0,besk1,besj0,besj1
      external besk0,besk1,besj0,besj1

      twooverpi=2.0d0/(4.0d0*atan(1.0d0))
      
      rlambda=rpars(1)
      rlambda2=rlambda*rlambda
c     determine whether one needs to smooth out the 1/(k^2+lambda^2) factor
c     at the origin
      dtmp = rlambda*bsize/beta
      ifnearcorrection=0
      if (dtmp.lt.1.0d-2) then
         ifnearcorrection=1
      endif
      if (ifnearcorrection.eq.1) then
         if (dim.eq.2) then
            dk0=besk0(rl*rlambda)
            dk1=besk1(rl*rlambda)
         endif
         if (dim.eq.3) then
            delam=exp(-rl*rlambda)
         endif
      endif
      
      zero=0.0d0
      call prol0eva(zero,wprolate,psi0,derpsi0)
      one=1.0d0
      call prol0eva(one,wprolate,fone,derpsi0)
      
      itype = 1
      nquad = 100
      call legeexps(itype,nquad,xs,u,v,whts)
      do i=1,nquad
         xs(i)=(xs(i)+1)/2*beta/bsize
         whts(i)=whts(i)/2*beta/bsize
      enddo

      fval=0.0d0
      do i=1,nquad
         xi2=xs(i)*xs(i)+rlambda2
         xval=sqrt(xi2)*bsize/beta

         if (xval.le.1.0d0) then
            call prol0eva(xval,wprolate,fval0,derpsi0)
         else
            fval0=0
         endif
         fhat=fval0/psi0/xi2

         if (ifnearcorrection.eq.1) then
            if (dim.eq.2) then
               xsc=rl*xs(i)
               sker=-rl*rlambda*besj0(xsc)*dk1
     1             +1+xsc*besj1(xsc)*dk0
            elseif (dim.eq.3) then
               xsc=rl*xs(i)
               sker=1-delam*(cos(xsc)+rlambda/xs(i)*sin(xsc))
            endif               
            fhat=fhat*sker
         endif

         
         if (dim.eq.2) then
            fval=fval+fhat*whts(i)*xs(i)
         elseif (dim.eq.3) then
            fval=fval+fhat*whts(i)*xs(i)*xs(i)*twooverpi
         endif
      enddo

      return
      end
c
c
c
c
c
c
      subroutine yukawa_residual_kernel_coefs(eps,dim,rpars,beta,
     1    bsize,rl,wprolate,ncoefs1,coefs1)
c
c     compute the Chebyshev expansion coefficients of 
c     the Yukawa residual kernel in two and three dimensions
c
c     Output:
c       ncoefs1 - order of Chebyshev expansions for points in list1
c       coefs1  - Chebyshev expansion coefficients for points in list1
c      
      implicit real *8 (a-h,o-z)
      integer dim

      real *8 beta,bsize

      real *8 rpars(*),wprolate(*)
      real *8 coefs1(*)
      
      real *8 xs(1000),whts(1000),fvals1(1000)
      real *8 r1(1000),w1(1000),fhat(1000)
      real *8, allocatable :: ftker1(:,:),u1(:,:)
      real *8, allocatable :: v1(:,:)
      
      real *8 xval,fval,psi0,derpsi0

      complex *16 z,h0,h1

      real *8 besk0,besk1,besj0,besj1
      external besk0,besk1,besj0,besj1
      
      twooverpi=2.0d0/(4.0d0*atan(1.0d0))
      
      rlambda=rpars(1)
      rlambda2=rlambda*rlambda

c     determine whether one needs to smooth out the 1/(k^2+lambda^2) factor
c     at the origin
      dtmp = rlambda*bsize/beta
      ifnearcorrection=0
      if (dtmp.lt.1.0d-2) then
         ifnearcorrection=1
      endif
      if (ifnearcorrection.eq.1) then
         if (dim.eq.2) then
            dk0=besk0(rl*rlambda)
            dk1=besk1(rl*rlambda)
         endif
         if (dim.eq.3) then
            delam=exp(-rl*rlambda)
         endif
      endif
      
      zero=0.0d0
      call prol0eva(zero,wprolate,psi0,derpsi0)
      one=1.0d0
      call prol0eva(one,wprolate,fone,derpsi0)

      itype = 1
      nquad = 100
      call legeexps(itype,nquad,xs,u,v,whts)
      do i=1,nquad
         xs(i)=(xs(i)+1)/2*beta/bsize
         whts(i)=whts(i)/2*beta/bsize
      enddo

      do i=1,nquad
         xi2=xs(i)*xs(i)+rlambda2
         xval=sqrt(xi2)*bsize/beta

         if (xval.le.1.0d0) then
            call prol0eva(xval,wprolate,fval0,derpsi0)
         else
            fval0=0
         endif
         fhat(i)=fval0/psi0/xi2
         
         if (dim.eq.2) then
            fhat(i)=fhat(i)*whts(i)*xs(i)
         elseif (dim.eq.3) then
            fhat(i)=fhat(i)*whts(i)*xs(i)*xs(i)*twooverpi
         endif

         if (ifnearcorrection.eq.1) then
            if (dim.eq.2) then
               xsc=rl*xs(i)
               sker=-rl*rlambda*besj0(xsc)*dk1
     1             +1+xsc*besj1(xsc)*dk0
            elseif (dim.eq.3) then
               xsc=rl*xs(i)
               sker=1-delam*(cos(xsc)+rlambda/xs(i)*sin(xsc))
            endif               
            fhat(i)=fhat(i)*sker
         endif         
      enddo

      itype = 2
      nr1 = 100
      allocate(ftker1(nquad,nr1))
      allocate(u1(nr1,nr1),v1(nr1,nr1))

      call chebexps(itype,nr1,r1,u1,v1,w1)
      
c     Chebyshev nodes on [0,bsize] for list1
      do i=1,nr1
         r1(i)=(r1(i)+1)/2*bsize
      enddo

      if (dim.eq.2) then
         ifexpon=1
         do i=1,nr1
         do j=1,nquad
            z=r1(i)*xs(j)
            call hank103(z,h0,h1,ifexpon)
            dj0=dble(h0)
            ftker1(j,i)=dj0
         enddo
         enddo
      elseif (dim.eq.3) then
         do i=1,nr1
         do j=1,nquad
            dd=r1(i)*xs(j)
            ftker1(j,i)=sin(dd)/dd
         enddo
         enddo
      endif

      do i=1,nr1
         fvals1(i)=0
         do j=1,nquad
            fvals1(i)=fvals1(i)-ftker1(j,i)*fhat(j)
         enddo
      enddo

c     calculate the Chebyshev expansion coefficients
      do i=1,nr1
         coefs1(i)=0
         do j=1,nr1
            coefs1(i)=coefs1(i)+u1(i,j)*fvals1(j)
         enddo
      enddo

c     simplified adaptation of the truncation rule used in Chebfun
c     See https://arxiv.org/pdf/1512.01803.pdf about the details
c     on how to chop a Chebyshev series
      coefsmax=abs(coefs1(1))
      do i=2,nr1
         if (abs(coefs1(i)) .gt. coefsmax) then
            coefsmax=abs(coefs1(i))
         endif
      enddo

      
      ncoefs1=1
      releps = eps*coefsmax
      do i=1,nr1-2
         if ((abs(coefs1(i)).lt. releps) .and.
     1       (abs(coefs1(i+1)).lt. releps) .and.
     2       (abs(coefs1(i+2)).lt. releps)) then
            ncoefs1=i
            exit
         endif
      enddo

      return
      end
c
c
c
c

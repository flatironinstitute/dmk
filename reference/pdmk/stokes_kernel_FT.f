c
c     Compute the radial Fourier transform 
c     of the windowed and difference kernels
c     of various kernels in two and three dimensions. The kernel splitting 
c     is done with the prolate spheroidal wave functions.
c
c      
c      
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      
c      
c     Stokeslet in two and three dimensions
c      
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      
      subroutine stokes_difference_kernel_Fourier_transform(dim,
     1    beta,bsizesmall,bsizebig,rk,wprolate,nfourier,fhat)
c
c     compute the Fourier transform of the difference kernel
c     of the Stokeslet in two and three dimensions
c
      implicit none
      integer dim,i,nfourier
      real *8 beta,bsizesmall,bsizebig,hpw,ws
      real *8 wprolate(*)

      real *8 xval,psi0,derpsi0
      real *8 xi
      real *8 c0,c1,g0d2,c4,zero,fval1,fval2,dd
      real *8 rk(*),fhat(*)

      call prolate_intvals(beta,wprolate,c0,c1,g0d2,c4)
      
      zero=0.0d0
      call prol0eva(zero,wprolate,psi0,derpsi0)

      dd = 0.5d0*g0d2
      
      do i=1,nfourier
         xi=rk(i)
         if (xi .lt. 1d-10) then
            fhat(i)=(6*g0d2**2-c4/c0)*(bsizesmall**4-bsizebig**4)/24
         else
            xval=xi*bsizesmall/beta
            if (xval.le.1) then
               call prol0eva(xval,wprolate,fval1,derpsi0)
               fval1 = fval1*(1+dd*(xi*bsizesmall)**2)
            else
               fval1=0
            endif
         
            xval=xi*bsizebig/beta
            if (xval.le.1) then
               call prol0eva(xval,wprolate,fval2,derpsi0)
               fval2 = fval2*(1+dd*(xi*bsizebig)**2)
            else
               fval2=0
            endif
         
            fhat(i)=(fval2-fval1)/psi0/xi**4
         endif
      enddo

      return
      end
c
c
c
c
      subroutine stokes_windowed_kernel_Fourier_transform(dim,
     1    beta,bsize,rl,rk,wprolate,nfourier,fhat)
c
c     compute the Fourier transform of the windowed kernel
c     of the Stokeslet in two and three dimensions
c
      implicit none
      integer dim,nfourier
      real *8 beta,bsize,rl

      real *8 wprolate(*)
      real *8 rk(*),fhat(*)


      integer i
      real *8 xs(1000),whts(1000),fvals(1000)
      real *8 xval,fval,psi0,derpsi0,xi,zero
      real *8 c0,c1,c4,g0d2,x,x2,x4,x6,x8
      real *8 dj0,dj1,dfac,tker,df

      real *8 besj0,besj1
      external besj0,besj1

      call prolate_intvals(beta,wprolate,c0,c1,g0d2,c4)
      
      zero=0.0d0
      call prol0eva(zero,wprolate,psi0,derpsi0)

      do i=1,nfourier
         xi=rk(i)
         
         xval=xi*bsize/beta
         if (xval.le.1) then
            call prol0eva(xval,wprolate,fval,derpsi0)
            fval = fval*(1+(xi*bsize)**2*g0d2/2)
         else
            fval=0
         endif
         
         fhat(i)=fval/psi0


c     truncated Fourier kernel of the original biharmonic kernel 1/k^4
         if (dim.eq.2) then
            x=rl*xi
            if (x.gt.2.0d-1) then
               dj0=besj0(x)
               dj1=besj1(x)
               dfac=log(rl)
               tker=( (dj0-1)-(dfac-1.5d0)*x**3*dj1/4 
     1             + (dfac-0.5d0)*x*dj1 - (dfac-1)*x**2*dj0/2 )/xi**4
            else
               df=log(rl)
               x2=x*x
               x4=x2*x2
               x6=x2*x4
               x8=x2*x6
               tker=-df/16 + 7.0d0/64 + (df/96 - 5.0d0/288)*x2
     1             +(13.0d0/16384 - df/2048)*x4
     2             + (df/92160 - 1.0d0/57600)*x6
     3             +(19.0d0/84934695 - df/7077888)*x8
               tker=tker*rl**4
            endif
         elseif (dim.eq.3) then
            x=rl*xi
            if (x.gt.2.0d-1) then
               tker=((2-x**2)*cos(x)+2*x*sin(x)-2)/xi**4/2
            else
               x2=x*x
               x4=x2*x2
               x6=x2*x4
               x8=x2*x6
               tker=(1.0d0/8 - x2/72 + x4/1920 - x6/100800 + x8/8709120)
               tker=tker*rl**4
            endif
         endif               
         fhat(i)=fhat(i)*tker
      enddo

      return
      end
c
c
c
c
c
c
      subroutine stokes_windowed_kernel_value_at_zero(dim,beta,
     1    bsize,rl,wprolate,fval)
c
c     compute the value of the windowed kernel at the origin
c     of the Stokeslet in two and three dimensions
c
c     Output: fval = W_l(0) for self interactions
c      
      implicit none
      integer dim
      real *8 beta,bsize,rl

      real *8 wprolate(*)

      integer itype,nquad,i
      
      real *8 xs(1000),whts(1000),fhat(1000)
      real *8 xval,fval,twooverpi,u,v

      twooverpi=2.0d0/(4.0d0*atan(1.0d0))
      
      itype = 1
      nquad = 100
      call legeexps(itype,nquad,xs,u,v,whts)
      do i=1,nquad
         xs(i)=(xs(i)+1)/2*beta/bsize
         whts(i)=whts(i)/2*beta/bsize
      enddo

      call stokes_windowed_kernel_Fourier_transform(dim,
     1    beta,bsize,rl,xs,wprolate,nquad,fhat)

      fval=0.0d0
      do i=1,nquad
         if (dim.eq.2) then
            fval=fval-fhat(i)*whts(i)*xs(i)**3
         elseif (dim.eq.3) then
            fval=fval-fhat(i)*whts(i)*xs(i)**4
         endif
      enddo

      if (dim.eq.2) fval=0.5d0*fval
      if (dim.eq.3) fval=fval*twooverpi*2.0d0/3
      
      return
      end
c
c
c
c
      subroutine stokes_residual_kernel_coefs(eps,dim,beta,
     1    bsize,rl,wprolate,
     2    n1_diag,coefs1_diag,
     3    n2_diag,coefs2_diag,
     4    n1_offd,coefs1_offd,
     5    n2_offd,coefs2_offd)
c
c     compute the Chebyshev expansion coefficients of 
c     the Stokes residual kernel in two and three dimensions
c
c     Output:
c     n1_diag      - order of Chebyshev expansions for diagonal
c                    interactions for points in list1
c     coefs1_diag  - Chebyshev expansion coefficients for
c                    diagonal interactions for points in list1
c     n2_diag      - order of Chebyshev expansions for diagonal
c                    interactions for points in list2
c     coefs2_diag  - Chebyshev expansion coefficients for
c                    diagonal interactions for points in list2
c      
c     n1_offd      - order of Chebyshev expansions for offdiagonal
c                    interactions for points in list1
c     coefs1_offd  - Chebyshev expansion coefficients for
c                    offdiagonal interactions for points in list1
c     n2_offd      - order of Chebyshev expansions for offdiagonal
c                    interactions for points in list2
c     coefs2_offd  - Chebyshev expansion coefficients for
c                    offdiagonal interactions for points in list2
c      
c      
c      
c      
      implicit none
      integer dim,itype,nquad,i,nr,listtype

      real *8 beta,bsize,twooverpi,u,v,ws,shift,dlen,rl,eps

      real *8 wprolate(*)
      real *8 coefs1_diag(*),coefs2_diag(*)
      real *8 coefs1_offd(*),coefs2_offd(*)

      integer n1_diag,n2_diag,n1_offd,n2_offd
      
      real *8 rks(1000),whts(1000)

      real *8 r1(1000),r2(1000),fhat(1000)
      
      twooverpi=2.0d0/(4.0d0*atan(1.0d0))
      
      itype = 1
      nquad = 200
      call legeexps(itype,nquad,rks,u,v,whts)
      do i=1,nquad
         rks(i)=(rks(i)+1)/2*beta/bsize
         whts(i)=whts(i)/2*beta/bsize
      enddo

      call stokes_windowed_kernel_Fourier_transform(dim,
     1    beta,bsize,rl,rks,wprolate,nquad,fhat)

      do i=1,nquad
         if (dim.eq.2) then
            fhat(i)=fhat(i)*whts(i)*rks(i)
         elseif (dim.eq.3) then
            fhat(i)=fhat(i)*whts(i)*rks(i)*rks(i)*twooverpi
         endif
      enddo

      itype = 0
      nr = 100
      call chebexps(itype,nr,r1,u,v,ws)
      call chebexps(itype,nr,r2,u,v,ws)
      
c     Chebyshev nodes on [0,bsize] for list1
      do i=1,nr
         if (dim.eq.3) then
            r1(i)=(r1(i)+1)/2*bsize
         elseif (dim.eq.2) then
            r1(i)=(r1(i)+1)/2*bsize*bsize
         endif
      enddo

      listtype = 1
      call stokes_residual_kernel_coefs0(dim,nquad,rks,fhat,
     1    nr,r1,coefs1_diag,coefs1_offd,listtype)

      call find_chebyshev_expansion_length(eps,nr,coefs1_diag,n1_diag)
      call find_chebyshev_expansion_length(eps,nr,coefs1_offd,n1_offd)
      
c     repeat the whole procedure for points in list2, where the
c     whole local kernel is smooth.
c     Chebyshev nodes on [(bsize/2)^2,bsize^2] for list2
c     note that the independent variable is r^2 instead of r here!!!
      shift=bsize**2/4
      dlen=bsize**2*3.0d0/4
      do i=1,nr
         r2(i)=(r2(i)+1)/2*dlen+shift
      enddo

      listtype = 2
      call stokes_residual_kernel_coefs0(dim,nquad,rks,fhat,
     1    nr,r2,coefs2_diag,coefs2_offd,listtype)


      call find_chebyshev_expansion_length(eps,nr,coefs2_diag,n2_diag)
      call find_chebyshev_expansion_length(eps,nr,coefs2_offd,n2_offd)
      
      
      return
      end
c
c
c
c
      subroutine stokes_residual_kernel_coefs0(dim,nk,dks,fhat,
     1    nr,rvals,coefs_diag,coefs_offd,listtype)
      
      implicit none
      integer dim,nk,nr,listtype
      real *8 dks(nk),whts(nk),fhat(nk)
      real *8 rvals(nr),coefs_diag(nr),coefs_offd(nr)

      integer i,j,itype
      real *8 fvals_diag(nr),fvals_offd(nr)
      real *8 df(1000),d2f(1000)
      real *8 x,r,dd,dj0,dj1
      real *8, allocatable :: drft(:,:)
      real *8, allocatable :: d2rft(:,:)
      real *8, allocatable :: u(:,:),v(:,:)
      real *8, allocatable :: xs(:),ws(:)

      real *8 besj0,besj1
      external besj0,besj1
      
c     first derivative of the radial fourier transform kernel
      allocate(drft(nk,nr))
c     second derivative of the radial fourier transform kernel
      allocate(d2rft(nk,nr))

      if (dim.eq.2) then
         do i=1,nr
         do j=1,nk
            if (listtype.eq.1 .and. dim.eq.3) then   
               r=rvals(i)
            elseif (listtype.eq.2 .or. dim.eq.2) then
               r=sqrt(rvals(i))
            endif
            dd=r*dks(j)
            dj0=besj0(dd)
            dj1=besj1(dd)

            drft(j,i)= -dks(j)*dj1
            d2rft(j,i)= -dks(j)**2*(dj0-dj1/dd)
         enddo
         enddo
      elseif (dim.eq.3) then
         do i=1,nr
         do j=1,nk
            if (listtype.eq.1 .and. dim.eq.3) then   
               r=rvals(i)
            elseif (listtype.eq.2 .or. dim.eq.2) then
               r=sqrt(rvals(i))
            endif
            dd=r*dks(j)
            drft(j,i)= (dd*cos(dd)-sin(dd))*dks(j)/dd**2
            d2rft(j,i)= ((2-dd**2)*sin(dd)-2*dd*cos(dd))
     1          *dks(j)**2/dd**3
         enddo
         enddo
      endif

      do i=1,nr
         df(i)=0
         do j=1,nk
            df(i)=df(i)+drft(j,i)*fhat(j)
         enddo
      enddo

      do i=1,nr
         d2f(i)=0
         do j=1,nk
            d2f(i)=d2f(i)+d2rft(j,i)*fhat(j)
         enddo
      enddo

      do i=1,nr
         if (listtype.eq.1 .and. dim.eq.3) then   
            r=rvals(i)
         elseif (listtype.eq.2 .or. dim.eq.2) then
            r=sqrt(rvals(i))
         endif

         if (dim.eq.2) then
            fvals_diag(i)= d2f(i)
            fvals_offd(i)= -d2f(i)+df(i)/r
         endif

         if (dim.eq.3) then
            fvals_diag(i)=  r*d2f(i)+df(i)
            fvals_offd(i)= -r*d2f(i)+df(i)
         endif
      enddo

c     points in list2. add the original kernel and divide by r or r^3
c     the whole expression is smooth
      if (listtype.eq.2) then
         if (dim.eq.2) then
            do i=1,nr
               fvals_diag(i)= -0.25d0*log(rvals(i))-fvals_diag(i)
               fvals_offd(i)= (0.5d0-fvals_offd(i))/rvals(i)
            enddo
         endif

         if (dim.eq.3) then
            do i=1,nr
               r=sqrt(rvals(i))
               fvals_diag(i)= (0.5d0-fvals_diag(i))/r
               fvals_offd(i)= (0.5d0-fvals_offd(i))/r**3
            enddo
         endif
      endif
      
      itype = 2
      allocate(u(nr,nr),v(nr,nr))
      allocate(xs(nr),ws(nr))

      call chebexps(itype,nr,xs,u,v,ws)

c     calculate the Chebyshev expansion coefficients
      do i=1,nr
         coefs_diag(i)=0
         do j=1,nr
            coefs_diag(i)=coefs_diag(i)+u(i,j)*fvals_diag(j)
         enddo
      enddo

      do i=1,nr
         coefs_offd(i)=0
         do j=1,nr
            coefs_offd(i)=coefs_offd(i)+u(i,j)*fvals_offd(j)
         enddo
      enddo
      
      return
      end
c
c
c
c
      subroutine find_chebyshev_expansion_length(eps,ncoefs,coefs,ntrue)
      implicit real *8 (a-h,o-z)
      real *8 coefs(*)

c     simplified adaptation of the truncation rule used in Chebfun
c     See https://arxiv.org/pdf/1512.01803.pdf about the details
c     on how to chop a Chebyshev series
      coefsmax=abs(coefs(1))
      do i=2,ncoefs
         if (abs(coefs(i)) .gt. coefsmax) then
            coefsmax=abs(coefs(i))
         endif
      enddo

      
      ntrue=-1
      releps = eps*coefsmax
      do i=1,ncoefs-2
         if ((abs(coefs(i)).lt. releps) .and.
     1       (abs(coefs(i+1)).lt. releps) .and.
     2       (abs(coefs(i+2)).lt. releps)) then
            ntrue=i
            exit
         endif
      enddo

      if (ntrue .lt. 0) ntrue=ncoefs
      
      return
      end
c
c
c
c

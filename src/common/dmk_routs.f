c
c
c*********************************************************************
c
c     construct a set of 1D transformation matrices for 
c     parent-to-children interpolation
c     and children-to-parent anterpolation
c     
c*********************************************************************
      subroutine dmk_get_coefs_translation_matrices(ndim,ipoly,
     1    norder,isgn,p2ctransmat,c2ptransmat)
c     This subroutine returns to the user a list of matrices needed to
c     carry out interpolation from the parent box to its 2**ndim children.
c
c     input:
c     ndim - integer
c            dimension of the underlying space
c     ipoly - integer
c            0: Legendre polynomials
c            1: Chebyshev polynomials
c     norder - integer
c           order of expansions for input function value array
c     isgn - integer (ndim,2**ndim)
c           contains the signs of center coordinates of child boxes
c           when the parent box is centered at the origin
c
c     output:
c     p2ctransmat - double precision (norder,norder,ndim,2**ndim)
c             p2ctransmat(1,1,1,j) contains ndim matrices where each of them contains
c             the transformation matrix converting the coefficients of the parent box
c             to the child box
c     c2ptransmat - double precision (norder,norder,ndim,2**ndim)
c             c2ptransmat(1,1,1,j) contains ndim matrices where each of them contains
c             the transformation matrix converting the coefficients of the child box
c             to the parent box
c
      implicit none
      integer norder,mc,itype,i,j,k,ndim,norder2,ipoly
      real *8 dp,dm
      real *8 p2ctransmat(norder,norder,ndim,2**ndim) 
      real *8 c2ptransmat(norder,norder,ndim,2**ndim)                
      real *8 xs(norder),ws(norder)
      real *8 umat(norder,norder),vmat(norder,norder)
      real *8 xp(norder),xm(norder)
      real *8 polyvp(norder,norder),polyvm(norder,norder)
      real *8 transp(norder,norder),transm(norder,norder)
      real *8 c2ptransp(norder,norder),c2ptransm(norder,norder)
      
      integer isgn(ndim,2**ndim)

      mc=2**ndim
      norder2=norder*norder

      itype=2
      if (ipoly.eq.0) then
         call legeexps(itype,norder,xs,umat,vmat,ws)
      elseif (ipoly.eq.1) then
         call chebexps(itype,norder,xs,umat,vmat,ws)
      endif

      do i=1,norder
         xm(i) = xs(i)/2-0.5d0
         xp(i) = xs(i)/2+0.5d0
      enddo

      do i=1,norder
         if (ipoly.eq.0) then
            call legepols(xp(i),norder-1,polyvp(1,i))
            call legepols(xm(i),norder-1,polyvm(1,i))
         elseif (ipoly.eq.1) then
            call chebpols(xp(i),norder-1,polyvp(1,i))
            call chebpols(xm(i),norder-1,polyvm(1,i))
         endif
      enddo

      do j=1,norder
         do i=1,norder
            dp=0
            dm=0
            do k=1,norder
               dp=dp+umat(i,k)*polyvp(j,k)
               dm=dm+umat(i,k)*polyvm(j,k)
            enddo
            transp(i,j)=dp
            transm(i,j)=dm
         enddo
      enddo

      do j=1,mc
         do k=1,ndim
            if (isgn(k,j).lt.0) then
               call dcopy_f77(norder2,transm,1,p2ctransmat(1,1,k,j),1)
            else
               call dcopy_f77(norder2,transp,1,p2ctransmat(1,1,k,j),1)
            endif
         enddo
      enddo

      do j=1,norder
         do i=1,norder
            c2ptransp(i,j)=transp(j,i)
         enddo
      enddo

      do j=1,norder
         do i=1,norder
            c2ptransm(i,j)=transm(j,i)
         enddo
      enddo

      do j=1,mc
      do k=1,ndim
         if (isgn(k,j).lt.0) then
            call dcopy_f77(norder2,c2ptransm,1,c2ptransmat(1,1,k,j),1)
         else
            call dcopy_f77(norder2,c2ptransp,1,c2ptransmat(1,1,k,j),1)
         endif
      enddo
      enddo

      
      return
      end
c      
c            
c
c
c*********************************************************************
c
c     1D transformation matrix converting polynomial expansion coefficients
c     to plane-wave expansion coefficients
c     
c*********************************************************************
      subroutine dmk_mk_coefs_pw_conversion_tables(ipoly,norder,npw,
     1    ts,xs,hpw,boxdim,tab_coefs2pw,tab_pw2coefs)
c     generates a table converting polynomial expansion coefficients
c     to plane wave expansion coefficients
c      
c     INPUT:
c     ipoly - integer
c            0: Legendre polynomials
c            1: Chebyshev polynomials
c     norder   number of Legendre nodes
c     npw      number of plane waves
c     ts       nodes of plane wave quadrature
c     xs       Chebyshev nodes
c     boxdim   box dimension at current level
c
c     OUTPUT:
c     tab_coefs2pw   converting proxy charge polynomial expansion coefficients
c                    to outgoing plane wave expansion coefficients
c     tab_pw2coefs   converting incoming plane wave expansion coeffcients
c                    to proxy potential polynomial expansion coefficients
c----------------------------------------------------------------------c
      implicit none
      integer ipoly,norder,npw,i,j,k,itype
      real *8 ts(npw),xs(norder),hpw,boxdim,dsq,x
      complex *16 tab_coefs2pw(npw,norder)
      complex *16 tab_pw2coefs(npw,norder)
      complex *16 eye,qqx,qq1,z
      complex *16 tmp1(npw,norder)
      real *8 xq(100),wq(100),umat(norder,norder),vmat(norder,norder)
c
      eye = dcmplx(0.0d0,1.0d0)
c
      dsq = boxdim/2
C
      do i=1,norder
         x = xs(i)*dsq
         do j=1,npw
            tmp1(j,i) = exp(eye*ts(j)*x)
         enddo
      enddo

      itype = 2
      if (ipoly.eq.0) then
         call legeexps(itype,norder,xq,umat,vmat,wq)
      elseif (ipoly.eq.1) then
         call chebexps(itype,norder,xq,umat,vmat,wq)
      endif
      
c     plane wave to targets, e^{ik t}
      do i=1,norder
         do j=1,npw
            z=0
            do k=1,norder
               z=z+tmp1(j,k)*umat(i,k)
            enddo
            tab_pw2coefs(j,i)=z
         enddo
      enddo

c     sources to plane wave, e^{-ik s}
      do i=1,norder
         do j=1,npw
            tab_coefs2pw(j,i)=conjg(tab_pw2coefs(j,i))
         enddo
      enddo
      
      return
      end subroutine
c      
c            
c      
c
      subroutine dmk_mk_coefs_pw_tables_pgh(ipoly,norder,npw,
     1    ts,xs,hpw,boxdim,tab_coefs2pw,tab_pw2pot,
     2    tab_pw2potx,tab_pw2potxx)
c     generates a table converting polynomial expansion coefficients
c     to plane wave expansion coefficients and vice versa
c      
c     INPUT:
c     ipoly - integer
c            0: Legendre polynomials
c            1: Chebyshev polynomials
c     norder   number of Legendre nodes
c     npw      number of plane waves
c     ts       nodes of plane wave quadrature
c     xs       Chebyshev nodes
c     boxdim   box dimension at current level
c
c     OUTPUT:
c     tab_coefs2pw   converting proxy charge polynomial expansion coefficients
c                    to outgoing plane wave expansion coefficients
c     tab_pw2pot     converting incoming plane wave expansion coeffcients
c                    to proxy potential polynomial expansion coefficients
c     tab_pw2potx    converting incoming plane wave expansion coeffcients
c                    to proxy gradient polynomial expansion coefficients
c     tab_pw2potxx   converting incoming plane wave expansion coeffcients
c                    to proxy hessian polynomial expansion coefficients
c----------------------------------------------------------------------c
      implicit none
      integer ipoly,norder,npw,i,j,k,itype
      real *8 ts(npw),xs(norder),hpw,boxdim,dsq,x
      complex *16 tab_coefs2pw(npw,norder)
      complex *16 tab_pw2pot(npw,norder)
      complex *16 tab_pw2potx(npw,norder)
      complex *16 tab_pw2potxx(npw,norder)
      complex *16 eye,qqx,qq1,z,z1,z2
      complex *16 tmp1(npw,norder)
      real *8 xq(100),wq(100),umat(norder,norder),vmat(norder,norder)
c
      eye = dcmplx(0.0d0,1.0d0)
c
      dsq = boxdim/2
C
      do i=1,norder
         x = xs(i)*dsq
         do j=1,npw
            tmp1(j,i) = exp(eye*ts(j)*x)
         enddo
      enddo

      itype = 2
      if (ipoly.eq.0) then
         call legeexps(itype,norder,xq,umat,vmat,wq)
      elseif (ipoly.eq.1) then
         call chebexps(itype,norder,xq,umat,vmat,wq)
      endif
      
c     plane wave to targets, e^{ik t}
      do i=1,norder
         do j=1,npw
            z=0
            z1=0
            z2=0
            do k=1,norder
               z  = z +tmp1(j,k)*umat(i,k)
               z1 = z1+eye*ts(j)*tmp1(j,k)*umat(i,k)
               z2 = z2-ts(j)*ts(j)*tmp1(j,k)*umat(i,k)
            enddo
            tab_pw2pot(j,i)=z
            tab_pw2potx(j,i)=z1
            tab_pw2potxx(j,i)=z2
         enddo
      enddo

c     sources to plane wave, e^{-ik s}
      do i=1,norder
         do j=1,npw
            tab_coefs2pw(j,i)=conjg(tab_pw2pot(j,i))
         enddo
      enddo
      
      return
      end subroutine
c      
c            
c      
c
c*********************************************************************
c
c     construct diagonal translation matrix converting outgoing plane-wave
c     expansion to incoming plane-wave expansion
c     
c*********************************************************************
      subroutine mk_pw_translation_matrices(ndim,xmin,npw,ts,nmax,
     1              wshift)
C
C     This subroutine precomputes all translation matrices for PW
C     expansions for mp to loc at the cutoff level.
c      
C     INPUT
C     ndim    = dimension of underlying space
c     xmin    = scaled (by 1/sqrt(delta) boxsize at the cutoff level
C     npw     = number of terms in 1d plane wave expansion
C     ts      = 1d pw expansion nodes
C     nmax    = number of different translation lengths in the whole scheme 
c      
C     OUTPUT:
C
C     wshift  = table of translation matrices for PW shift 
C
      implicit real *8 (a-h,o-z)
      real *8 ts(npw)
      
      complex *16 wshift(((npw+1)/2)*npw**(ndim-1),(2*nmax+1)**ndim)

      if (ndim.eq.1) then
         call  mk_pw_translation_matrices_1d(xmin,npw,ts,nmax,
     1       wshift)
      elseif (ndim.eq.2) then
         call  mk_pw_translation_matrices_2d(xmin,npw,ts,nmax,
     1       wshift)
      elseif (ndim.eq.3) then
         call  mk_pw_translation_matrices_3d(xmin,npw,ts,nmax,
     1       wshift)
      endif
      
      return
      end
c
c
c     
c
      subroutine mk_pw_translation_matrices_1d(xmin,npw,ts,nmax,
     1              wshift)
      implicit real *8 (a-h,o-z)
      real *8 ts(npw)
      
      complex *16 wshift(((npw+1)/2),(2*nmax+1))
      
      complex *16,allocatable:: ww(:,:)

      complex *16 eye,ztmp
C
      eye = dcmplx(0,1)
      
      allocate(ww(npw,-nmax:nmax))


      do j1=1,npw
         ztmp = exp(eye*ts(j1)*xmin)
         ww(j1,0)=1
         do k1=1,nmax
            ww(j1,k1) = ztmp
            ztmp = ztmp*ztmp
            ww(j1,-k1) = dconjg(ww(j1,k1))
         enddo
      enddo

      k=0
      do k1=-nmax,nmax
         k=k+1
         j=0   
         do j1=1,((npw+1)/2)
            j=j+1
            wshift(j,k) = ww(j1,k1)
         enddo
      enddo
c
      return
      end
c
c
c     
c
      subroutine mk_pw_translation_matrices_2d(xmin,npw,ts,nmax,
     1              wshift)
      implicit real *8 (a-h,o-z)
      real *8 ts(npw)
      
      complex *16 wshift(npw*((npw+1)/2),(2*nmax+1)**2)
      
      complex *16,allocatable:: ww(:,:)

      complex *16 eye,ztmp
C
      eye = dcmplx(0,1)
      
      allocate(ww(npw,-nmax:nmax))


      do j1=1,npw
         ztmp = exp(eye*ts(j1)*xmin)
         ww(j1,0)=1
         do k1=1,nmax
            ww(j1,k1) = ztmp
            ztmp = ztmp*ztmp
            ww(j1,-k1) = dconjg(ww(j1,k1))
         enddo
      enddo

      k=0
      do k1=-nmax,nmax
      do k2=-nmax,nmax
         k=k+1
         j=0   
         do j1=1,((npw+1)/2)
         do j2=1,npw
            j=j+1
            wshift(j,k) = ww(j2,k2)*ww(j1,k1)
         enddo
         enddo
      enddo
      enddo
c
      return
      end
c
c
c     
c
      subroutine mk_pw_translation_matrices_3d(xmin,npw,ts,nmax,
     1              wshift)
      implicit real *8 (a-h,o-z)
      real *8 ts(npw)
      
      complex *16 wshift(npw*npw*((npw+1)/2),(2*nmax+1)**3)
      
      complex *16,allocatable:: ww(:,:)

      complex *16 eye,ztmp
C
      eye = dcmplx(0,1)
      
      allocate(ww(npw,-nmax:nmax))


      do j1=1,npw
         ztmp = exp(eye*ts(j1)*xmin)
         ww(j1,0)=1
         do k1=1,nmax
            ww(j1,k1) = ztmp
            ztmp = ztmp*ztmp
            ww(j1,-k1) = dconjg(ww(j1,k1))
         enddo
      enddo

      k=0
      do k1=-nmax,nmax
      do k2=-nmax,nmax
      do k3=-nmax,nmax
         k=k+1
         j=0   
         do j1=1,((npw+1)/2)
         do j2=1,npw
         do j3=1,npw
            j=j+1
            wshift(j,k) = ww(j3,k3)*ww(j2,k2)*ww(j1,k1)
         enddo
         enddo
         enddo
      enddo
      enddo
      enddo
c
      return
      end
c
c
c
c
C*********************************************************************C      
c
c     tensor product proxy charges -> outgoing plane wave expansions 
c     
c
C*********************************************************************C      
      subroutine dmk_proxycharge2pw(ndim,nd,n,coefs,npw,
     1    tab_coefs2pw,pwexp)
c     This routine computes the plane wave expansion from
c     polynomial expansion coefficients.
c     calls zgemm to compute the matrix-matrix product
c
c     INPUT:
c     nd       vector length (for multiple RHS)
c     n        dimension of coeff array
c     coefs    Chebyshev polynomial expansion coefficients
c
c     npw      number of plane waves
c                 NOTE 3D convention is pwexp(npw,npw,((npw+1)/2))
c     tab_coefs2pw  precomputed table of 1D conversion factors
c
c     OUTPUT:
c     pwexp    plane wave expansion
c----------------------------------------------------------------------c
      implicit none
      integer nd,n,npw,ndim
      real *8 coefs(n**ndim,nd)
      complex *16 tab_coefs2pw(npw,n)
      complex *16 pwexp(npw**(ndim-1)*((npw+1)/2),nd)

      if (ndim.eq.2) then
         call dmk_proxycharge2pw_2d(nd,n,coefs,npw,
     1    tab_coefs2pw,pwexp)
      elseif (ndim.eq.3) then
         call dmk_proxycharge2pw_3d(nd,n,coefs,npw,
     1       tab_coefs2pw,pwexp)
      endif
      
      return
      end subroutine
c
c
c
C*********************************************************************C
      subroutine dmk_proxycharge2pw_2d(nd,n,coefs,npw,
     1    tab_coefs2pw,pwexp)
C*********************************************************************C
      implicit none
      integer nd,n,npw,ind,m1,m2,m3,k3,npw2
      real *8 coefs(n,n,nd)
      complex *16 tab_coefs2pw(npw,n)
      complex *16 pwexp(npw,(npw+1)/2,nd)

      complex *16 ff(n,(npw+1)/2)
      complex *16 cd
      complex *16 alpha,beta
c
      complex *16 zcoefs(n,n)

      npw2 = (npw+1)/2
      
      alpha=1.0d0
      beta=0.0d0
      do ind = 1,nd
         do m2=1,n
         do m1=1,n
            zcoefs(m1,m2)=coefs(m1,m2,ind)
         enddo
         enddo
      
c        transform in y
         call zgemm('n','t',n,npw2,n,alpha,zcoefs,n,
     1       tab_coefs2pw,npw,beta,ff,n)
c
c        transform in x
         call zgemm('n','n',npw,npw2,n,alpha,
     1       tab_coefs2pw,npw,ff,n,
     2       beta,pwexp(1,1,ind),npw)
c     
      enddo
      
      return
      end subroutine
c
c
c
C*********************************************************************C
      subroutine dmk_proxycharge2pw_3d(nd,n,coefs,npw,
     1    tab_coefs2pw,pwexp)
C*********************************************************************C
      implicit none
      integer nd,n,npw,ind,m1,m2,m3,k3,npw2
      real *8 coefs(n,n,n,nd)
      complex *16 tab_coefs2pw(npw,n)
      complex *16 pwexp(npw,npw,(npw+1)/2,nd)

      complex *16 ff(n,n,(npw+1)/2)
      complex *16 fft(n,(npw+1)/2,n)
      complex *16 ff2(npw,(npw+1)/2,n)
      complex *16 cd
      complex *16 alpha,beta
c
      complex *16 zcoefs(n,n,n)

      npw2=(npw+1)/2
      
      alpha=1.0d0
      beta=0.0d0
      do ind = 1,nd
         do m3=1,n
         do m2=1,n
         do m1=1,n
            zcoefs(m1,m2,m3)=coefs(m1,m2,m3,ind)
         enddo
         enddo
         enddo
c        transform in z
         call zgemm('n','t',n*n,npw2,n,alpha,zcoefs,n*n,
     1       tab_coefs2pw,npw,beta,ff,n*n)
c
         do m1 = 1,n
         do k3 = 1,npw2
         do m2 = 1,n
            fft(m2,k3,m1) = ff(m1,m2,k3)
         enddo
         enddo
         enddo
c        transform in y
         call zgemm('n','n',npw,npw2*n,n,alpha,
     1       tab_coefs2pw,npw,fft,n,
     2       beta,ff2,npw)
c        transform in x
         call zgemm('n','t',npw,npw*npw2,n,alpha,
     1       tab_coefs2pw,npw,ff2,npw*npw2,
     2       beta,pwexp(1,1,1,ind),npw)
c     
      enddo
      
      return
      end subroutine
c
c
c
C*********************************************************************
c
c     incoming plane-wave expansions -> proxy potentials
C
C*********************************************************************
      subroutine dmk_pw2proxypot(ndim,nd,n,npw,pwexp,
     1    tab_pw2coefs,coefs)
c     This routine computes the potential on a tensor grid from 
c     the plane wave expansion coefficients (implicitly about the box center).
c
c     uses BLAS zgemm!
c
c     INPUT:
c     ndim     dimension of the physical space
c     nd       vector length (for multiple RHS)
c     n        number of Legendre nodes
c     npw      number of plane waves
c     pwexp    plane wave expansion
c                 NOTE 3D convention is pwexp(npw,npw,((npw+1)/2))
c     tab_pw2coefs  precomputed table of 1D conversion 
c
c     OUTPUT:
c     pot      potential values on the tensor grid
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      complex *16 pwexp(npw**(ndim-1)*((npw+1)/2),nd)
      complex *16 tab_pw2coefs(npw,n)
      real *8 coefs(n**ndim,nd)

      if (ndim.eq.2) then
         call dmk_pw2proxypot_2d(nd,n,npw,pwexp,tab_pw2coefs,coefs)
      elseif (ndim.eq.3) then
         call dmk_pw2proxypot_3d(nd,n,npw,pwexp,tab_pw2coefs,coefs)
      endif
      
      return
      end
c
c
c
c
C*********************************************************************C
      subroutine dmk_pw2proxypot_2d(nd,n,npw,pwexp,tab_pw2coefs,coefs)
C*********************************************************************C
      implicit none
      integer npw,nd,n,npw2,k1,k2,m2,ind
      complex *16 pwexp(npw,(npw+1)/2,nd)
      complex *16 tab_pw2coefs(npw,n)
      real *8 coefs(n,n,nd)

      complex *16 ff(n,(npw+1)/2)
      complex *16 zcoefs(n,n)
      complex *16 cd
      complex *16 alpha,beta
c
      npw2=npw/2
      alpha=1.0d0
      beta=0.0d0
      
      do ind = 1,nd
c        transform in x
         call zgemm('t','n',n,(npw+1)/2,npw,alpha,
     1       tab_pw2coefs,npw,pwexp(1,1,ind),npw,beta,ff,n)

c
         do m2 = 1,(npw+1)/2
         do k1 = 1,n
            if (m2.gt.npw2) then
               ff(k1,m2) = ff(k1,m2)/2
            endif
         enddo
         enddo
      
c        transform in y
         call zgemm('n','n',n,n,(npw+1)/2,alpha,
     1       ff,n,tab_pw2coefs,npw,beta,zcoefs,n)
c
         do k2 = 1,n
         do k1 = 1,n
            coefs(k1,k2,ind)=coefs(k1,k2,ind)
     1          +dreal(zcoefs(k1,k2))*2
         enddo
         enddo
c
      enddo
      
      return
      end subroutine
c
c
c
c
C*********************************************************************C
      subroutine dmk_pw2proxypot_3d(nd,n,npw,pwexp,tab_pw2coefs,coefs)
C*********************************************************************C
      implicit none
      integer npw,nd,n,npw2,k1,m2,m3,k2,k3,ind
      complex *16 pwexp(npw,npw,(npw+1)/2,nd)
      complex *16 tab_pw2coefs(npw,n)
      real *8 coefs(n,n,n,nd)

      complex *16 ff(n,npw,(npw+1)/2)
      complex *16 fft(npw,(npw+1)/2,n)
      complex *16 ff2t(n,(npw+1)/2,n)
      complex *16 ff2(n,n,(npw+1)/2)
      complex *16 zcoefs(n,n,n)
      complex *16 cd
      complex *16 alpha,beta
c
      npw2=npw/2
      alpha=1.0d0
      beta=0.0d0
      
      do ind = 1,nd
c        transform in x
         call zgemm('t','n',n,npw*((npw+1)/2),npw,alpha,
     1       tab_pw2coefs,npw,pwexp(1,1,1,ind),npw,beta,ff,n)

c
         do k1 = 1,n
         do m3 = 1,(npw+1)/2
         do m2 = 1,npw
            fft(m2,m3,k1)=ff(k1,m2,m3)
         enddo
         enddo
         enddo
      
c        transform in y
         call zgemm('t','n',n,((npw+1)/2)*n,npw,alpha,
     1       tab_pw2coefs,npw,fft,npw,beta,ff2t,n)
         
         do m3 = 1,(npw+1)/2
         do k2 = 1,n
         do k1 = 1,n
            ff2(k1,k2,m3) = ff2t(k2,m3,k1)
            if (m3.gt.npw2) then
               ff2(k1,k2,m3) = ff2t(k2,m3,k1)/2
            endif
         enddo
         enddo
         enddo
      
c        transform in z
         call zgemm('n','n',n*n,n,(npw+1)/2,alpha,
     1       ff2,n*n,tab_pw2coefs,npw,beta,zcoefs,n*n)
c
         do k3 = 1,n
         do k2 = 1,n
         do k1 = 1,n
            coefs(k1,k2,k3,ind)=coefs(k1,k2,k3,ind)
     1          +dreal(zcoefs(k1,k2,k3))*2
         enddo
         enddo
         enddo
c
      enddo
      
      return
      end subroutine
c
c
c
c
C*********************************************************************
c
c     incoming plane-wave expansions -> proxy potentials,
c                                       gradients, and hessians
C
C*********************************************************************
      subroutine dmk_pw2proxypgh(ndim,nd,n,npw,pwexp,tab_pw2pot,
     1           tab_pw2der,tab_pw2dxx,ifpgh,pot,grad,hess)
C*********************************************************************C
c     This routine computes the polynomial expansion coefficients of the
c     potential, gradient, and hessian on a tensor 
c     grid from the plane wave expansion coefficients (implicitly about the box center).
c
c     INPUT:
c     ndim     dimension of underlying space
c     nd       vector length (for multiple RHS)
c     n        number of Legendre nodes
c     npw      number of plane wave expansion along each dimension
c     pwexp    plane wave expansion
c                 NOTE 3D convention is pwexp(npw,npw,((npw+1)/2))
c     tab_pw2pot  precomputed table of 1D conversion for potential evaluation
c     tab_pw2der  precomputed table of 1D conversion for gradient evaluation
c     tab_pw2dxx  precomputed table of 1D conversion for hessian evaluation
c
c     ifpgh: 1 -> pot only
c            2 -> pot+grad
c            3 -> pot+grad+hessian
c
c
c     OUTPUT:
c     pot      potential values on the tensor grid
c     grad     gradient values on the tensor grid
c     hess     hessian values on the tensor grid
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      complex *16 pwexp(((npw+1)/2)*npw**(ndim-1),nd)
      complex *16 tab_pw2pot(npw,n)
      complex *16 tab_pw2der(npw,n)
      complex *16 tab_pw2dxx(npw,n)
      real *8 pot(n**ndim,nd)
      real *8 grad(n**ndim,nd,ndim)
      real *8 hess(n**ndim,nd,ndim*(ndim+1)/2)
c
      if (ifpgh.eq.1) then
        if (ndim.eq.1) then
cccc  not implemented
        elseif (ndim.eq.2) then
          call dmk_pw2proxypot_2d(nd,n,npw,pwexp,tab_pw2pot,pot)
        elseif (ndim.eq.3) then
          call dmk_pw2proxypot_3d(nd,n,npw,pwexp,tab_pw2pot,pot)
        endif
      elseif (ifpgh.eq.2) then
        if (ndim.eq.1) then
          call dmk_pw2proxypg_1d(nd,n,npw,pwexp,tab_pw2pot,tab_pw2der,
     1         pot,grad)
        elseif (ndim.eq.2) then
          call dmk_pw2proxypg_2d(nd,n,npw,pwexp,tab_pw2pot,tab_pw2der,
     1         pot,grad)
        elseif (ndim.eq.3) then
          call dmk_pw2proxypg_3d(nd,n,npw,pwexp,tab_pw2pot,tab_pw2der,
     1         pot,grad)
        endif
      elseif (ifpgh.eq.3) then
        if (ndim.eq.1) then
          call dmk_pw2proxypgh_1d(nd,n,npw,pwexp,tab_pw2pot,tab_pw2der,
     1         tab_pw2dxx,pot,grad,hess)
        elseif (ndim.eq.2) then
          call dmk_pw2proxypgh_2d(nd,n,npw,pwexp,tab_pw2pot,tab_pw2der,
     1         tab_pw2dxx,pot,grad,hess)
        elseif (ndim.eq.3) then
          call dmk_pw2proxypgh_3d(nd,n,npw,pwexp,tab_pw2pot,tab_pw2der,
     1         tab_pw2dxx,pot,grad,hess)
        endif
      endif
      
      return
      end subroutine
c
c
c
c      
C*********************************************************************
c
c     incoming plane-wave expansions -> proxy potentials and
c                                       gradients
C
c*********************************************************************      
      subroutine dmk_pw2proxypg_1d(nd,n,npw,pwexp,tab_pw2pot,
     1           tab_pw2der,pot,grad)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      real *8 pot(n,nd)
      real *8 grad(n,nd)
      complex *16 tab_pw2pot(npw,n)
      complex *16 tab_pw2der(npw,n)
      complex *16 pwexp(((npw+1)/2),nd),cd,cdx
c
      npw2=npw/2
      do ind = 1,nd
         do k1 = 1,n
            cd=0.0d0
            cdx=0.0d0
            do m1 = 1,npw2
               cd   = cd   + tab_pw2pot(m1,k1)*pwexp(m1,ind)
               cdx  = cdx  + tab_pw2der(m1,k1)*pwexp(m1,ind)
            enddo
            m1=((npw+1)/2)
            if (m1.gt.npw2) then
               cd   = cd   + tab_pw2pot(m1,k1)*pwexp(m1,ind)/2
               cdx  = cdx  + tab_pw2der(m1,k1)*pwexp(m1,ind)/2
            endif
            pot(k1,ind)  = pot(k1,ind)  + dreal(cd)*2
            grad(k1,ind) = grad(k1,ind) + dreal(cdx)*2
         enddo
      enddo
c
      return
      end subroutine
c
c
c
c
c
C*********************************************************************C
      subroutine dmk_pw2proxypg_2d(nd,n,npw,pwexp,tab_pw2pot,
     1           tab_pw2der,pot,grad)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      real *8 pot(n,n,nd)
      real *8 grad(n,n,nd,2)
      complex *16 tab_pw2pot(npw,n)
      complex *16 tab_pw2der(npw,n)
      complex *16 ff(((npw+1)/2),n)
      complex *16 ffx(((npw+1)/2),n)
      complex *16 pwexp(npw,((npw+1)/2),nd),cd,cdx,cdy
c
      npw2=npw/2
      
      do ind = 1,nd
         do m2 = 1,((npw+1)/2)
         do k1 = 1,n
            cd=0.0d0
            cdx=0.0d0
            do m1 = 1,npw
               cd   = cd   + tab_pw2pot(m1,k1)*pwexp(m1,m2,ind)
               cdx  = cdx  + tab_pw2der(m1,k1)*pwexp(m1,m2,ind)
            enddo
            ff(m2,k1) = cd
            ffx(m2,k1) = cdx
         enddo
         enddo
c
         do k2 = 1,n
         do k1 = 1,n
            cd = 0.0d0
            cdx = 0.0d0
            cdy = 0.0d0
            do m2 = 1,npw2
               cd   = cd   + tab_pw2pot(m2,k2)*ff(m2,k1)
               cdy  = cdy  + tab_pw2der(m2,k2)*ff(m2,k1)

               cdx  = cdx  + tab_pw2pot(m2,k2)*ffx(m2,k1)
            enddo
            m2=((npw+1)/2)
            if (m2.gt.npw2) then
               cd   = cd   + tab_pw2pot(m2,k2)*ff(m2,k1)/2
               cdy  = cdy  + tab_pw2der(m2,k2)*ff(m2,k1)/2
               cdx  = cdx  + tab_pw2pot(m2,k2)*ffx(m2,k1)/2
            endif
            pot(k1,k2,ind)=pot(k1,k2,ind)+dreal(cd)*2

            grad(k1,k2,ind,1)=grad(k1,k2,ind,1)+dreal(cdx)*2
            grad(k1,k2,ind,2)=grad(k1,k2,ind,2)+dreal(cdy)*2
         enddo
         enddo
c
      enddo
c
      return
      end subroutine
c
c
c
c
c
C*********************************************************************C
      subroutine dmk_pw2proxypg_3d(nd,n,npw,pwexp,tab_pw2pot,
     1           tab_pw2der,pot,grad)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      complex *16 pwexp(npw,npw,((npw+1)/2),nd)
      complex *16 tab_pw2pot(npw,n)
      complex *16 tab_pw2der(npw,n)

      real *8 pot(n,n,n,nd)
      real *8 grad(n,n,n,nd,3)

      complex *16 ff(n,npw,((npw+1)/2))
      complex *16 ffx(n,npw,((npw+1)/2))
      
      complex *16 ff2(n,n,((npw+1)/2))
      complex *16 ff2x(n,n,((npw+1)/2))
      complex *16 ff2y(n,n,((npw+1)/2))
      
      complex *16 cd,cdx,cdy,cdz
c
      npw2=npw/2
      
      do ind = 1,nd
c       transformation in x
        do m3 = 1,((npw+1)/2)
        do m2 = 1,npw
        do k1 = 1,n
           cd=0.0d0
           cdx=0.0d0
           do m1 = 1,npw
              cd =   cd   + tab_pw2pot(m1,k1)*pwexp(m1,m2,m3,ind)
              cdx  = cdx  + tab_pw2der(m1,k1)*pwexp(m1,m2,m3,ind)
           enddo
           ff(k1,m2,m3) = cd
           ffx(k1,m2,m3) = cdx
        enddo
        enddo
        enddo
c       transformation in y
        do m3 = 1,((npw+1)/2)
        do k2 = 1,n
        do k1 = 1,n
           cd = 0.0d0
           cdx = 0.0d0
           cdy = 0.0d0
           do m2 = 1,npw
              cd   = cd   + tab_pw2pot(m2,k2)*ff(k1,m2,m3)
              cdy  = cdy  + tab_pw2der(m2,k2)*ff(k1,m2,m3)
              cdx  = cdx  + tab_pw2pot(m2,k2)*ffx(k1,m2,m3)
           enddo
           ff2(k1,k2,m3) = cd
           ff2x(k1,k2,m3) = cdx
           ff2y(k1,k2,m3) = cdy
        enddo
        enddo
        enddo
c       transformation in z
        do k3 = 1,n
        do k2 = 1,n
        do k1 = 1,n
           cd = 0.0d0
           cdx = 0.0d0
           cdy = 0.0d0
           cdz = 0.0d0
           do m3 = 1,npw2
              cd   = cd  +  tab_pw2pot(m3,k3)*ff2(k1,k2,m3)
              cdz  = cdz +  tab_pw2der(m3,k3)*ff2(k1,k2,m3)
              cdx  = cdx  + tab_pw2pot(m3,k3)*ff2x(k1,k2,m3)
              cdy  = cdy  + tab_pw2pot(m3,k3)*ff2y(k1,k2,m3)
           enddo
           m3=((npw+1)/2)
           if (m3.gt.npw2) then
              cd   = cd  +  tab_pw2pot(m3,k3)*ff2(k1,k2,m3)/2
              cdz  = cdz +  tab_pw2der(m3,k3)*ff2(k1,k2,m3)/2
              cdx  = cdx  + tab_pw2pot(m3,k3)*ff2x(k1,k2,m3)/2
              cdy  = cdy  + tab_pw2pot(m3,k3)*ff2y(k1,k2,m3)/2
           endif
           pot(k1,k2,k3,ind)=pot(k1,k2,k3,ind)+dreal(cd)*2
           
           grad(k1,k2,k3,ind,1)=grad(k1,k2,k3,ind,1)+dreal(cdx)*2
           grad(k1,k2,k3,ind,2)=grad(k1,k2,k3,ind,2)+dreal(cdy)*2
           grad(k1,k2,k3,ind,3)=grad(k1,k2,k3,ind,3)+dreal(cdz)*2
        enddo
        enddo
        enddo
c
      enddo
      
      return
      end subroutine
c
c
C
C*********************************************************************
c
c     incoming plane-wave expansions -> proxy potentials,
c                                       gradients, and hessians
C
c*********************************************************************      
      subroutine dmk_pw2proxypgh_1d(nd,n,npw,pwexp,tab_pw2pot,
     1           tab_pw2der,tab_pw2dxx,pot,grad,hess)
C*********************************************************************C
c     This routine computes the potential and gradient on a tensor grid from 
c     the plane wave expansion coefficients (implicitly about the box center).
c
c     INPUT:
c     nd       vector length (for multiple RHS)
c     n        number of nodes
c     npw      number of plane waves
c     pwexp    plane wave expansion
c     tab_pw2pot    precomputed table of 1D conversion 
c     tab_pw2der  precomputed table of deriv of 1D conversion 
c     tab_pw2dxx  precomputed table of second deriv of 1D conversion 
c
c     OUTPUT:
c     pot      potential values on the tensor grid
c     grad     grad values on the tensor grid
c     hess     hess values on the tensor grid
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      real *8 pot(n,nd)
      real *8 grad(n,nd)
      real *8 hess(n,nd)
      complex *16 tab_pw2pot(npw,n)
      complex *16 tab_pw2der(npw,n)
      complex *16 tab_pw2dxx(npw,n)
      complex *16 pwexp(((npw+1)/2),nd),cd,cdx,cdxx
c
      npw2=npw/2
      do ind = 1,nd
         do k1 = 1,n
            cd=0.0d0
            cdx=0.0d0
            cdxx=0.0d0
            do m1 = 1,npw2
               cd   = cd   + tab_pw2pot(m1,k1)*pwexp(m1,ind)
               cdx  = cdx  + tab_pw2der(m1,k1)*pwexp(m1,ind)
               cdxx = cdxx + tab_pw2dxx(m1,k1)*pwexp(m1,ind)
            enddo
            m1=((npw+1)/2)
            if (m1.gt.npw2) then
               cd   = cd   + tab_pw2pot(m1,k1)*pwexp(m1,ind)/2
               cdx  = cdx  + tab_pw2der(m1,k1)*pwexp(m1,ind)/2
               cdxx = cdxx + tab_pw2dxx(m1,k1)*pwexp(m1,ind)/2
            endif
            pot(k1,ind)  = pot(k1,ind)  + dreal(cd)*2
            grad(k1,ind) = grad(k1,ind) + dreal(cdx)*2
            hess(k1,ind) = hess(k1,ind) + dreal(cdxx)*2
         enddo
      enddo
c
      return
      end subroutine
c
c
c
c
c
C*********************************************************************C
      subroutine dmk_pw2proxypgh_2d(nd,n,npw,pwexp,tab_pw2pot,
     1           tab_pw2der,tab_pw2dxx,pot,grad,hess)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      real *8 pot(n,n,nd)
      real *8 grad(n,n,nd,2)
      real *8 hess(n,n,nd,3)
      complex *16 tab_pw2pot(npw,n)
      complex *16 tab_pw2der(npw,n)
      complex *16 tab_pw2dxx(npw,n)
      complex *16 ff(((npw+1)/2),n)
      complex *16 ffx(((npw+1)/2),n)
      complex *16 ffxx(((npw+1)/2),n)
      complex *16 pwexp(npw,((npw+1)/2),nd),cd,cdx,cdy
      complex *16 cdxx,cdxy,cdyy
c
      npw2=npw/2
      
      do ind = 1,nd
         do m2 = 1,((npw+1)/2)
         do k1 = 1,n
            cd=0.0d0
            cdx=0.0d0
            cdxx=0.0d0
            do m1 = 1,npw
               cd   = cd   + tab_pw2pot(m1,k1)*pwexp(m1,m2,ind)
               cdx  = cdx  + tab_pw2der(m1,k1)*pwexp(m1,m2,ind)
               cdxx = cdxx + tab_pw2dxx(m1,k1)*pwexp(m1,m2,ind)
            enddo
            ff(m2,k1) = cd
            ffx(m2,k1) = cdx
            ffxx(m2,k1) = cdxx
         enddo
         enddo
c
         do k2 = 1,n
         do k1 = 1,n
            cd = 0.0d0
            cdx = 0.0d0
            cdy = 0.0d0
            cdxx = 0.0d0
            cdxy = 0.0d0
            cdyy = 0.0d0
            do m2 = 1,npw2
               cd   = cd   + tab_pw2pot(m2,k2)*ff(m2,k1)
               cdy  = cdy  + tab_pw2der(m2,k2)*ff(m2,k1)
               cdyy = cdyy + tab_pw2dxx(m2,k2)*ff(m2,k1)

               cdx  = cdx  + tab_pw2pot(m2,k2)*ffx(m2,k1)
               cdxy = cdxy + tab_pw2der(m2,k2)*ffx(m2,k1)

               cdxx = cdxx + tab_pw2pot(m2,k2)*ffxx(m2,k1)
            enddo
            m2=((npw+1)/2)
            if (m2.gt.npw2) then
               cd   = cd   + tab_pw2pot(m2,k2)*ff(m2,k1)/2
               cdy  = cdy  + tab_pw2der(m2,k2)*ff(m2,k1)/2
               cdyy = cdyy + tab_pw2dxx(m2,k2)*ff(m2,k1)/2

               cdx  = cdx  + tab_pw2pot(m2,k2)*ffx(m2,k1)/2
               cdxy = cdxy + tab_pw2der(m2,k2)*ffx(m2,k1)/2

               cdxx = cdxx + tab_pw2pot(m2,k2)*ffxx(m2,k1)/2
            endif
            pot(k1,k2,ind)=pot(k1,k2,ind)+dreal(cd)*2

            grad(k1,k2,ind,1)=grad(k1,k2,ind,1)+dreal(cdx)*2
            grad(k1,k2,ind,2)=grad(k1,k2,ind,2)+dreal(cdy)*2

            hess(k1,k2,ind,1)=hess(k1,k2,ind,1)+dreal(cdxx)*2
            hess(k1,k2,ind,2)=hess(k1,k2,ind,2)+dreal(cdxy)*2
            hess(k1,k2,ind,3)=hess(k1,k2,ind,3)+dreal(cdyy)*2
         enddo
         enddo
c
      enddo
c
      return
      end subroutine
c
c
c
c
c
C*********************************************************************C
      subroutine dmk_pw2proxypgh_3d(nd,n,npw,pwexp,tab_pw2pot,
     1           tab_pw2der,tab_pw2dxx,pot,grad,hess)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      complex *16 pwexp(npw,npw,((npw+1)/2),nd)
      complex *16 tab_pw2pot(npw,n)
      complex *16 tab_pw2der(npw,n)
      complex *16 tab_pw2dxx(npw,n)
      real *8 pot(n,n,n,nd)
      real *8 grad(n,n,n,nd,3)
      real *8 hess(n,n,n,nd,6)

      complex *16 ff(n,npw,((npw+1)/2))
      complex *16 ffx(n,npw,((npw+1)/2))
      complex *16 ffxx(n,npw,((npw+1)/2))
      
      complex *16 ff2(n,n,((npw+1)/2))
      complex *16 ff2x(n,n,((npw+1)/2))
      complex *16 ff2xy(n,n,((npw+1)/2))
      complex *16 ff2xx(n,n,((npw+1)/2))
      complex *16 ff2y(n,n,((npw+1)/2))
      complex *16 ff2yy(n,n,((npw+1)/2))
      
      complex *16 cd,cdx,cdy,cdz,cdxx,cdyy,cdzz,cdxy,cdxz,cdyz
c
      npw2=npw/2
      
      do ind = 1,nd
c       transformation in x
        do m3 = 1,((npw+1)/2)
        do m2 = 1,npw
        do k1 = 1,n
           cd=0.0d0
           cdx=0.0d0
           cdxx=0.0d0
           do m1 = 1,npw
              cd =   cd   + tab_pw2pot(m1,k1)*pwexp(m1,m2,m3,ind)
              cdx  = cdx  + tab_pw2der(m1,k1)*pwexp(m1,m2,m3,ind)
              cdxx = cdxx + tab_pw2dxx(m1,k1)*pwexp(m1,m2,m3,ind)
           enddo
           ff(k1,m2,m3) = cd
           ffx(k1,m2,m3) = cdx
           ffxx(k1,m2,m3) = cdxx
        enddo
        enddo
        enddo
c       transformation in y
        do m3 = 1,((npw+1)/2)
        do k2 = 1,n
        do k1 = 1,n
           cd = 0.0d0
           cdx = 0.0d0
           cdy = 0.0d0
           cdxx = 0.0d0
           cdxy = 0.0d0
           cdyy = 0.0d0
           do m2 = 1,npw
              cd   = cd   + tab_pw2pot(m2,k2)*ff(k1,m2,m3)
              cdy  = cdy  + tab_pw2der(m2,k2)*ff(k1,m2,m3)
              cdyy = cdyy + tab_pw2dxx(m2,k2)*ff(k1,m2,m3)

              cdx  = cdx  + tab_pw2pot(m2,k2)*ffx(k1,m2,m3)
              cdxy = cdxy + tab_pw2der(m2,k2)*ffx(k1,m2,m3)

              cdxx = cdxx  + tab_pw2pot(m2,k2)*ffxx(k1,m2,m3)
           enddo
           ff2(k1,k2,m3) = cd
           ff2x(k1,k2,m3) = cdx
           ff2xx(k1,k2,m3) = cdxx
           ff2y(k1,k2,m3) = cdy
           ff2yy(k1,k2,m3) = cdyy
           ff2xy(k1,k2,m3) = cdxy
        enddo
        enddo
        enddo
c       transformation in z
        do k3 = 1,n
        do k2 = 1,n
        do k1 = 1,n
           cd = 0.0d0
           cdx = 0.0d0
           cdy = 0.0d0
           cdz = 0.0d0
           cdxx = 0.0d0
           cdyy = 0.0d0
           cdzz = 0.0d0
           cdxy = 0.0d0
           cdxz = 0.0d0
           cdyz = 0.0d0
           do m3 = 1,npw2
              cd   = cd  +  tab_pw2pot(m3,k3)*ff2(k1,k2,m3)
              cdz  = cdz +  tab_pw2der(m3,k3)*ff2(k1,k2,m3)
              cdzz = cdzz + tab_pw2dxx(m3,k3)*ff2(k1,k2,m3)

              cdx  = cdx  + tab_pw2pot(m3,k3)*ff2x(k1,k2,m3)
              cdxz = cdxz + tab_pw2der(m3,k3)*ff2x(k1,k2,m3)

              cdy  = cdy  + tab_pw2pot(m3,k3)*ff2y(k1,k2,m3)
              cdyz = cdyz + tab_pw2der(m3,k3)*ff2y(k1,k2,m3)

              cdxx = cdxx + tab_pw2pot(m3,k3)*ff2xx(k1,k2,m3)
              cdxy = cdxy + tab_pw2pot(m3,k3)*ff2xy(k1,k2,m3)
              cdyy = cdyy + tab_pw2pot(m3,k3)*ff2yy(k1,k2,m3)
           enddo
           m3=((npw+1)/2)
           if (m3.gt.npw2) then
              cd   = cd  +  tab_pw2pot(m3,k3)*ff2(k1,k2,m3)/2
              cdz  = cdz +  tab_pw2der(m3,k3)*ff2(k1,k2,m3)/2
              cdzz = cdzz + tab_pw2dxx(m3,k3)*ff2(k1,k2,m3)/2

              cdx  = cdx  + tab_pw2pot(m3,k3)*ff2x(k1,k2,m3)/2
              cdxz = cdxz + tab_pw2der(m3,k3)*ff2x(k1,k2,m3)/2

              cdy  = cdy  + tab_pw2pot(m3,k3)*ff2y(k1,k2,m3)/2
              cdyz = cdyz + tab_pw2der(m3,k3)*ff2y(k1,k2,m3)/2

              cdxx = cdxx + tab_pw2pot(m3,k3)*ff2xx(k1,k2,m3)/2
              cdxy = cdxy + tab_pw2pot(m3,k3)*ff2xy(k1,k2,m3)/2
              cdyy = cdyy + tab_pw2pot(m3,k3)*ff2yy(k1,k2,m3)/2
           endif
           pot(k1,k2,k3,ind)=pot(k1,k2,k3,ind)+dreal(cd)*2
           
           grad(k1,k2,k3,ind,1)=grad(k1,k2,k3,ind,1)+dreal(cdx)*2
           grad(k1,k2,k3,ind,2)=grad(k1,k2,k3,ind,2)+dreal(cdy)*2
           grad(k1,k2,k3,ind,3)=grad(k1,k2,k3,ind,3)+dreal(cdz)*2

           hess(k1,k2,k3,ind,1)=hess(k1,k2,k3,ind,1)+dreal(cdxx)*2
           hess(k1,k2,k3,ind,2)=hess(k1,k2,k3,ind,2)+dreal(cdyy)*2
           hess(k1,k2,k3,ind,3)=hess(k1,k2,k3,ind,3)+dreal(cdzz)*2

           hess(k1,k2,k3,ind,4)=hess(k1,k2,k3,ind,4)+dreal(cdxy)*2
           hess(k1,k2,k3,ind,5)=hess(k1,k2,k3,ind,5)+dreal(cdxz)*2
           hess(k1,k2,k3,ind,6)=hess(k1,k2,k3,ind,6)+dreal(cdyz)*2
        enddo
        enddo
        enddo
c
      enddo
      
      return
      end subroutine
c
c
C
c
C
c************************************************************************
C
C     outgoing plane-wave expansion -> incoming plane-wave expansion
C 
C
C************************************************************************
      subroutine dmk_shiftpw(nd,nexp,pwexp1,
     1              pwexp2,wshift)
C
C     This subroutine converts the PW expansion (pwexp1) about
C     the center (CENT1) into an PW expansion (pwexp2) about 
C     (CENT2) using precomputed translation matrix wshift.
C
C     INPUT
C
c     nd      = vector length (for vector input)
C     delta   = Gaussian variance
C     nn      = number of terms in PW expansion
C     pwexp1  = original expansion 
C     wshift  = precomputed PW exp translation matrix 
C
C     OUTPUT:
C
C     pwexp2 = shifted expansion 
C
      implicit none
      integer nd,j,ind,nexp
      complex *16 pwexp1(nexp,nd)
      complex *16 pwexp2(nexp,nd)
      complex *16 wshift(nexp)
c      
      do ind=1,nd
      do j=1,nexp
         pwexp2(j,ind) = pwexp2(j,ind) + pwexp1(j,ind)*wshift(j)
      enddo
      enddo
c
      return
      end
c
C
c
      subroutine dmk_find_pwshift_ind(ndim,iperiod,tcenter,scenter,
     1   bs0,xmin,nmax,ind)
c     returns an index used in dmk_shiftpw
c
c     input:
c     ndim - dimension of the underlying space
c     iperiod - 0: free space; 1: doubly periodic
c     tcenter - target box center
c     scenter - source box center
c     bs0 - root box size
c     xmin - box size at npwlevel
c     nmax - (2*nmax+1) is the number of possible translations along
c            each direction
c     output
c     ind - index to determine which precomputed pwshift matrix 
c           should be used in plane-wave translation at the cut-off
c           level
c
      implicit real *8 (a-h,o-z)
      real *8 tcenter(ndim),scenter(ndim)
      integer jxyz(ndim),i

      do i=1,ndim
         jx= nint((tcenter(i) - scenter(i))/xmin)
         
         if (iperiod .eq. 1) then
            jxp1=nint((tcenter(i) - scenter(i) - bs0)/xmin)
            jxm1=nint((tcenter(i) - scenter(i) + bs0)/xmin)
            if (abs(jx).gt.abs(jxp1)) jx=jxp1
            if (abs(jx).gt.abs(jxm1)) jx=jxm1
         endif

         jxyz(i)=jx+nmax
      enddo
      
      n=2*nmax+1
      ind=1
      inc=1
      do i=1,ndim
         ind=ind+jxyz(i)*inc
         inc=inc*n
      enddo
      
      return
      end subroutine
c
c
c
c
      subroutine find_pwshift_indices_periodic_level1(ndim,tcenter,
     1   scenter,bs0,xmin,nmax,inds,nitotal)
c     returns an index used in dmk_shiftpw
c
c     input:
c     ndim - dimension of the underlying space
c     iperiod - 0: free space; 1: doubly periodic
c     tcenter - target box center
c     scenter - source box center
c     bs0 - root box size
c     xmin - box size at npwlevel
c     nmax - (2*nmax+1) is the number of possible translations along
c            each direction
c     output
c     ind - index to determine which precomputed pwshift matrix 
c           should be used in plane-wave translation at the cut-off
c           level
c
      implicit real *8 (a-h,o-z)
      real *8 tcenter(ndim),scenter(ndim)
      integer jxyz(ndim,10),i,nimage(ndim),nitotal,inds(*),dts(10)

      do i=1,ndim
         nimage(i)=1
      enddo

      do i=1,ndim
         dts(i) = nint((tcenter(i) - scenter(i))/xmin)
         if (abs(dts(i)).gt.0) nimage(i)=2
      enddo

      nitotal=1
      do i=1,ndim
         nitotal=nitotal*nimage(i)
      enddo

      ii=1
      do k=1,ndim
         jxyz(k,ii)=dts(k)+nmax
      enddo
      do i=1,ndim
         do j=1,nimage(i)
            if (j.eq.2) then
               do kk=ii+1,2*ii
                  do k=1,ndim
                     if (k.eq.i) then
                        jxyz(k,kk)=-dts(k)+nmax
                     else
                        jxyz(k,kk)=jxyz(k,kk-ii)
                     endif                     
                  enddo
               enddo
               ii=2*ii
            endif
         enddo
      enddo

      n=2*nmax+1
      do j=1,nitotal
         inds(j)=1
         inc=1
         do i=1,ndim
            inds(j)=inds(j)+jxyz(i,j)*inc
            inc=inc*n
         enddo
      enddo
      
      return
      end subroutine
c
c
c
c
C*********************************************************************
c
c     multiply the Fourier transform of density with the Fourier 
c     transform of the kernel
C
C*********************************************************************
      subroutine dmk_multiply_kernelFT(nd,nexp,pwexp,
     1              weights)
C     INPUT
C
c     nd      = vector length (for vector input)
C     delta   = Gaussian variance
C     nn      = number of terms in PW expansion
C     pwexp   = Fourier transform of the density
C     weights = Fourier transform of the kernel
C
C     OUTPUT:
C
C     pwexp = Fourier transform of the potential
C
      implicit none
      integer nd,j,ind,nexp
      complex *16 pwexp(nexp,nd)
      real *8 weights(nexp)
      
      do ind=1,nd
      do j=1,nexp
         pwexp(j,ind) = pwexp(j,ind)*weights(j)
      enddo
      enddo
c
      return
      end
c
c
c
c
C*********************************************************************
c
c     multiply the Fourier transform of density with the Fourier 
c     transform of the kernel: charge+dipole -> potential
C
C*********************************************************************
      subroutine dmk_multiply_kernelFT_cd2p(nd,ndim,ifcharge,ifdipole,
     1    nexp,pwexp,radialft,rk)

c     The Fourier transform of the windowed and difference kernels
c
C     INPUT
C
c     nd      = vector length (for vector input)
C     delta   = Gaussian variance
C     nexp      = number of terms in PW expansion
C     pwexp   = Fourier transform of the density
C     radialft = radial part of the Fourier transform of the kernel
C     rk      = (k_x,k_y,k_z), i.e., xyz coordinates of the Fourier nodes
c      
C     OUTPUT:
C
C     pwexp = Fourier transform of the potential
C
      implicit none
      integer nd,ndim,ifcharge,ifdipole,nexp
      integer i,j,n,ind,id
      complex *16 pwexp(nexp,nd,*)
      real *8 radialft(nexp),rk(ndim,nexp),dd,di

      complex *16 ima
      data ima/(0.0d0,1.0d0)/
      complex *16, allocatable :: pwexp1(:,:)
      
      allocate(pwexp1(nexp,nd))
      
      do ind=1,nd
         do n=1,nexp
            pwexp1(n,ind)=0.0d0
         enddo
      enddo

      id=0
      
      if (ifcharge.eq.1) then
         id=id+1
         do ind=1,nd
            do n=1,nexp
               pwexp1(n,ind)=pwexp(n,ind,id)
            enddo
         enddo
      endif
      
      if (ifdipole.eq.1) then
         do ind=1,nd
            do n=1,nexp
               do j=1,ndim
                  pwexp1(n,ind)=pwexp1(n,ind)
     1                -pwexp(n,ind,j+id)*ima*rk(j,n)
               enddo
            enddo
         enddo
      endif

      do ind=1,nd
         do n=1,nexp
            pwexp(n,ind,1)=pwexp1(n,ind)*radialft(n)
         enddo
      enddo
      
               
      return
      end
c
c
c
c
C*********************************************************************
c
c     multiply the Fourier transform of density with the Fourier 
c     transform of the kernel for the Stokeslet
C
C*********************************************************************
      subroutine stokesdmk_multiply_windowed_kernelFT(ndim,nd,
     1    nexp,pwexp,radialft,rk,rksq,npw,cval)

c     The Fourier transform of the windowed and difference kernels
c     of the Stokeslet has the same structure as that of the original
c     Stokeslet. That is,

c     F(k)_{ij} = f(|k|) (-|k|^2 \delta_{ij} + k_i k_j),
c     where f(|k|) is the radial part.
      
C     INPUT
C
c     nd      = vector length (for vector input)
C     delta   = Gaussian variance
C     nn      = number of terms in PW expansion
C     pwexp   = Fourier transform of the density
C     radialft = radial part of the Fourier transform of the kernel
C     rk      = (k_x,k_y,k_z), i.e., xyz coordinates of the Fourier nodes
c     rksq    = |k|^2
      
C     OUTPUT:
C
C     pwexp = Fourier transform of the potential
C
      implicit none
      integer nd,i,j,n,ind,nexp,ndim,n0,npw,npw2
      complex *16 pwexp(nexp,nd,ndim)
      real *8 radialft(nexp),rk(ndim,nexp),rksq(nexp),dd,di
      real *8 cval
      complex *16, allocatable :: pwexp1(:),cvec(:,:)
      
      allocate(pwexp1(nexp))
      allocate(cvec(nd,ndim))
      
      npw2 = npw/2
      if (ndim.eq.3) then
         n0 = npw2*npw**(ndim-1)+npw2+1+npw*npw2
      elseif (ndim.eq.2) then
         n0 = npw2+1+npw*npw2
      endif
cccc      print *, 'n0=',n0

      do j=1,ndim
         do ind=1,nd
            cvec(ind,j) = pwexp(n0,ind,j)
         enddo
      enddo
      
      do ind=1,nd
         do n=1,nexp
            pwexp1(n)=0.0d0
            do j=1,ndim
               pwexp1(n)=pwexp1(n)+pwexp(n,ind,j)*rk(j,n)
            enddo
         enddo

         do n=1,nexp
            dd=rksq(n)*radialft(n)
            do i=1,ndim
               di=rk(i,n)*radialft(n)
               pwexp(n,ind,i)=pwexp1(n)*di-pwexp(n,ind,i)*dd
            enddo
         enddo
      enddo

      do i=1,ndim
      do ind=1,nd
         pwexp(n0,ind,i) = pwexp(n0,ind,i) + cval*cvec(ind,i)
      enddo
      enddo
      
      return
      end
c
c
c
c
      subroutine stokesdmk_multiply_kernelFT(ndim,nd,nexp,pwexp,
     1    radialft,rk,rksq)

c     The Fourier transform of the windowed and difference kernels
c     of the Stokeslet has the same structure as that of the original
c     Stokeslet. That is,

c     F(k)_{ij} = f(|k|) (-|k|^2 \delta_{ij} + k_i k_j),
c     where f(|k|) is the radial part.
      
C     INPUT
C
c     nd      = vector length (for vector input)
C     delta   = Gaussian variance
C     nn      = number of terms in PW expansion
C     pwexp   = Fourier transform of the density
C     radialft = radial part of the Fourier transform of the kernel
C     rk      = (k_x,k_y,k_z), i.e., xyz coordinates of the Fourier nodes
c     rksq    = |k|^2
      
C     OUTPUT:
C
C     pwexp = Fourier transform of the potential
C
      implicit none
      integer nd,i,j,n,ind,nexp,ndim
      complex *16 pwexp(nexp,nd,ndim)
      real *8 radialft(nexp),rk(ndim,nexp),rksq(nexp),dd,di
      
      complex *16, allocatable :: pwexp1(:)
      
      allocate(pwexp1(nexp))
      
      do ind=1,nd
         do n=1,nexp
            pwexp1(n)=0.0d0
            do j=1,ndim
               pwexp1(n)=pwexp1(n)+pwexp(n,ind,j)*rk(j,n)
            enddo
         enddo

         do n=1,nexp
            dd=rksq(n)*radialft(n)
            do i=1,ndim
               di=rk(i,n)*radialft(n)
               pwexp(n,ind,i)=pwexp1(n)*di-pwexp(n,ind,i)*dd
            enddo
         enddo
      enddo
         
      return
      end
c
c
c
c
      subroutine stressdmk_multiply_kernelFT(ndim,nd,nexp,pwexp,
     1    radialft,rk,rksq,uhat)
C     INPUT
C
c     nd      = vector length (for vector input)
C     delta   = Gaussian variance
C     nn      = number of terms in PW expansion
C     pwexp   = Fourier transform of the density
C     radialft = radial part of the Fourier transform of the kernel
C     rk      = (k_x,k_y,k_z), i.e., xyz coordinates of the Fourier nodes
c     rksq    = |k|^2
      
C     OUTPUT:
C
C     uhat = Fourier transform of the potential
C
      implicit none
      integer nd,i,j,n,ind,nexp,ndim
      complex *16 pwexp(nexp,ndim,ndim,nd)
      complex *16 uhat(nexp,ndim,nd)
      real *8 radialft(nexp),rk(ndim,nexp),rksq(nexp),dd,di
      
      complex *16, allocatable :: double_product(:),fhat_diag(:)
      complex *16, allocatable :: products(:,:)
      complex *16 ima,ztmp,zz
      data ima/(0.0d0,1.0d0)/
      
      allocate(double_product(nexp))
      allocate(products(ndim,nexp))
      allocate(fhat_diag(nexp))
      
      do ind=1,nd
         do n=1,nexp
            double_product(n)=0.0d0
            do j=1,ndim
            do i=1,ndim
               double_product(n)=double_product(n)+pwexp(n,i,j,ind)
     1             *rk(i,n)*rk(j,n)
            enddo
            enddo
         enddo

         do n=1,nexp
            fhat_diag(n)=0.0d0
            do i=1,ndim
               fhat_diag(n) = fhat_diag(n)+pwexp(n,i,i,ind)
            enddo
         enddo

         do n=1,nexp
            do i=1,ndim
               products(i,n)=0.0d0
               do j=1,ndim
                  products(i,n) = products(i,n)+pwexp(n,i,j,ind)*rk(j,n)
     1                +pwexp(n,j,i,ind)*rk(j,n)
               enddo
            enddo
         enddo
         
         do n=1,nexp
            ztmp = -ima*radialft(n)
            zz = rksq(n)*fhat_diag(n)-2*double_product(n) 
            do i=1,ndim
               uhat(n,i,ind)=ztmp*( rk(i,n)*zz+rksq(n)*products(i,n) )
            enddo
         enddo
      enddo
         
      return
      end
c
c
c
c
c*********************************************************************
c
c     copy plane-wave expansions
c     
c*********************************************************************
      subroutine dmk_copy_pwexp(nd,nexp,pwexp1,pwexp2)
      implicit none
      integer nd,nn,j,ind,nexp
      complex *16 pwexp1(nexp,nd)
      complex *16 pwexp2(nexp,nd)

C
      do ind=1,nd
      do j=1,nexp
         pwexp2(j,ind) = pwexp1(j,ind)
      enddo
      enddo
c
      return
      end
c
c
c
C
c*********************************************************************
c
c     initialize plane-wave expansions to zero
c     
c*********************************************************************
      subroutine dmk_pwzero(nd,nexp,pwexp)
      implicit none
      integer nd,nexp,n,ind
      complex *16 pwexp(nexp,nd)
c
      do ind=1,nd
      do n=1,nexp
         pwexp(n,ind)=0.0d0
      enddo
      enddo
      
      return
      end
C
C
c
c

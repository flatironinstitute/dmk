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
c     tab_coefs2pw   table 
c     tab_pw2coefs   table 
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
      integer nd,n,npw,ind,m1,m2,m3,k3
      real *8 coefs(n,n,nd)
      complex *16 tab_coefs2pw(npw,n)
      complex *16 pwexp(npw,(npw+1)/2,nd)

      complex *16 ff(n,(npw+1)/2)
      complex *16 cd
      complex *16 alpha,beta
c
      complex *16 zcoefs(n,n)

      alpha=1.0d0
      beta=0.0d0
      do ind = 1,nd
         do m2=1,n
         do m1=1,n
            zcoefs(m1,m2)=coefs(m1,m2,ind)
         enddo
         enddo
      
c        transform in y
         call zgemm('n','t',n,(npw+1)/2,n,alpha,zcoefs,n,
     1       tab_coefs2pw,npw,beta,ff,n)
c
c        transform in x
         call zgemm('n','n',npw,(npw+1)/2,n,alpha,
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
      integer nd,n,npw,ind,m1,m2,m3,k3
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
         call zgemm('n','t',n*n,(npw+1)/2,n,alpha,zcoefs,n*n,
     1       tab_coefs2pw,npw,beta,ff,n*n)
c
         do m1 = 1,n
         do k3 = 1,(npw+1)/2
         do m2 = 1,n
            fft(m2,k3,m1) = ff(m1,m2,k3)
         enddo
         enddo
         enddo
c        transform in y
         call zgemm('n','n',npw,((npw+1)/2)*n,n,alpha,
     1       tab_coefs2pw,npw,fft,n,
     2       beta,ff2,npw)
c        transform in x
         call zgemm('n','t',npw,npw*((npw+1)/2),n,alpha,
     1       tab_coefs2pw,npw,ff2,npw*((npw+1)/2),
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
         call zgemm('t','n',n,n*(npw/2+1),npw,alpha,
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

      
c            
c      
c
c
c
c      
c*********************************************************************
C
C     charges at arbitrary locations -> tensor product proxy charges
C
C*********************************************************************
      subroutine pdmk_charge2proxycharge(ndim,nd,norder,
     1    ns,sources,charge,cen,sc,coefs)
c     use "anterpolation" to assign equivalent charges on the tensor product grid.
c     actually, it returns the polynomial expansion coefficients when those equivalent
c     charges are regarded as polynomial values.
c
c     input:
c     nd - number of coefficient vectors
c     norder - polynomial expansion order
c     ns - number of sources
c     sources - coordinates of sources
c     cen - coordinates of the center of the box
c     sc - scaling factor that scales the points in the box to the standard interval [-1,1]
c
c     output:
c     coefs - polynomial expansion coefficients for given charges
c
      implicit none
      integer nd,ns,norder,ndim
      real *8 charge(nd,ns)
      real *8 coefs(norder**ndim,nd)
      real *8 sources(ndim,ns),cen(ndim)
      real *8 sc

      if (ndim.eq.2) then
         call pdmk_charge2proxycharge_2d(nd,norder,
     1       ns,sources,charge,cen,sc,coefs)
      elseif (ndim.eq.3) then
         call pdmk_charge2proxycharge_3d(nd,norder,
     1       ns,sources,charge,cen,sc,coefs)
      endif

      return
      end
c
c      
c
c      
      subroutine pdmk_charge2proxycharge_2d(nd,norder,
     1    ns,sources,charge,cen,sc,coefs)
      implicit none
      integer nd,ns,norder,i,j,k,ind,m
      real *8 charge(nd,ns)
      real *8 coefs(norder,norder,nd)
      real *8 sources(2,ns),cen(2)
      real *8 sc,dd,x,y,z,alpha,beta

      real *8, allocatable :: px(:,:),py(:,:)
      real *8, allocatable :: dy(:,:)

      allocate(dy(ns,norder))
      
      allocate(px(norder,ns))
      allocate(py(norder,ns))

      do i=1,ns
         x=(sources(1,i)-cen(1))*sc
         call chebpols(x,norder-1,px(1,i))
      enddo

      do i=1,ns
         y=(sources(2,i)-cen(2))*sc
         call chebpols(y,norder-1,py(1,i))
      enddo

      alpha=1.0d0
      beta=1.0d0
      
      do ind=1,nd
         do k=1,norder
         do m=1,ns
            dy(m,k)=charge(ind,m)*py(k,m)
         enddo
         enddo

         call dgemm('n','n',norder,norder,ns,alpha,
     1       px,norder,dy,ns,
     2       beta,coefs(1,1,ind),norder)
      enddo
      
      return
      end
c
c
c
c
c
      subroutine pdmk_charge2proxycharge_3d(nd,norder,
     1    ns,sources,charge,cen,sc,coefs)
      implicit none
      integer nd,ns,norder,i,j,k,ind,m
      real *8 charge(nd,ns)
      real *8 coefs(norder,norder,norder,nd)
      real *8 sources(3,ns),cen(3)
      real *8 sc,dd,x,y,z,alpha,beta

      real *8, allocatable :: px(:,:),py(:,:),pz(:,:)
      real *8, allocatable :: dz(:,:),dyz(:,:,:)

      allocate(dz(ns,norder),dyz(ns,norder,norder))
      
      allocate(px(norder,ns))
      allocate(py(norder,ns))
      allocate(pz(norder,ns))

      do i=1,ns
         x=(sources(1,i)-cen(1))*sc
         call chebpols(x,norder-1,px(1,i))
      enddo

      do i=1,ns
         y=(sources(2,i)-cen(2))*sc
         call chebpols(y,norder-1,py(1,i))
      enddo

      do i=1,ns
         z=(sources(3,i)-cen(3))*sc
         call chebpols(z,norder-1,pz(1,i))
      enddo

      alpha=1.0d0
      beta=1.0d0
      
      do ind=1,nd
         do k=1,norder
         do m=1,ns
            dz(m,k)=charge(ind,m)*pz(k,m)
         enddo
         enddo

         do k=1,norder
         do j=1,norder
         do m=1,ns
            dyz(m,j,k)=py(j,m)*dz(m,k)
         enddo
         enddo
         enddo

         call dgemm('n','n',norder,norder*norder,ns,alpha,
     1       px,norder,dyz,ns,
     2       beta,coefs(1,1,1,ind),norder)
      enddo
      
      return
      end
c
c
c
c
c*********************************************************************
C
C     compute the full tensor product Fourier transform of the kernel
c     from its 1D radial Fourier transform
C
C*********************************************************************
      subroutine mk_tensor_product_Fourier_transform(dim,
     1    npw,nfourier,fhat,nexp,pswfft)
C
C     This subroutine precomputes the tensor product Fourier transform  
C     in dimension dim via its Fourier transform along the radial direction
C
C     INPUT
C
c     dim      = dimension of the underlying space
c     npw      = number of Fourier modes in each dimension
C     nexp     = number of terms in the full plane-wave expansion
c     nfourier = number of Fourier modes along the radial direction
c     fhat     = Fourier transform of the difference kernel along 
c                the radial direction
c      
C     OUTPUT:
C
C     pswfft - tensor product Fourier transform of the difference kernel 
c
      implicit none
      integer dim,nexp,nfourier
      integer j,j1,j2,j3,k2,npw,npw2,j1p,j2p

      real *8 pswfft(nexp),fhat(0:nfourier)


      npw2=npw/2

      j=0
      if (dim.eq.1) then
         do j1=-npw2,0
            j=j+1
            k2 = j1*j1
            pswfft(j) = fhat(k2)
         enddo
      elseif (dim.eq.2) then
         do j2=-npw2,0
         do j1=-npw2,(npw-1)/2
            j=j+1
c           for symmetric trapezoidal rule - npw odd
            k2 = j1*j1+j2*j2
            pswfft(j) = fhat(k2)
         enddo
         enddo
      elseif (dim.eq.3) then
c        for symmetric trapezoidal rule - npw odd
         do j3=-npw2,0
         do j2=-npw2,(npw-1)/2
         do j1=-npw2,(npw-1)/2
            j=j+1
c           for symmetric trapezoidal rule - npw odd
            k2 = j1*j1+j2*j2+j3*j3
            pswfft(j) = fhat(k2)
         enddo
         enddo
         enddo
      endif

      return
      end
c
c
c
C
      subroutine pdmk_coefsp_zero(nd,nexp,coefsp)
C     Initialize the polynomial expansion to zero.
      implicit none
      integer nd,ind,nexp,j
      real *8 coefsp(nexp,nd)

C
      do ind=1,nd
         do j=1,nexp
            coefsp(j,ind) = 0
         enddo
      enddo
c
      return
      end
c
C

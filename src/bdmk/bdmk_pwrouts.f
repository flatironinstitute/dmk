c      
c
c
c     
C*********************************************************************
c
c     density -> proxy charges
C
C*********************************************************************
      subroutine bdmk_density2proxycharge(ndim,nd,nin,fin,nout,fout,
     1    umat0,sc)
c
c     this subroutine transform the input data given on a tensor product grid
c     to the output data of different size.
c
c     input:
c     nd - number of input data
c     nin - order of the expansion in each dimension of the input data
c     fin - input data
c     nout - order of the expansion in each dimension of the output data
c     umat0 - 1D transformation matrix along each direction
c     sc - scaling factor
c      
c      
c     output:
c     fout - the transformed data, see the above explanation.
c
      implicit none
      integer nd,nin,nout,ndim
      real *8 alpha,beta,sc
      real *8 fin(nd,nin**ndim)
      real *8 fout(nout**ndim,nd)
      real *8 umat0(nout,nin)
      
      real *8 fin0(nin**ndim)
      real *8 umat(nout,nin)
      real *8 umat_nd(nout,nin,ndim)
      integer ind,j1,j2,i,np,ifadd

      do j2=1,nin
         do j1=1,nout
            umat(j1,j2)=umat0(j1,j2)*sc
         enddo
      enddo

      np=nout*nin
      do i=1,ndim
         call dcopy_f77(np,umat,1,umat_nd(1,1,i),1)
      enddo
      
      ifadd=0
      do ind=1,nd
         do j1=1,nin**ndim
            fin0(j1)=fin(ind,j1)
         enddo
         call tens_prod_trans(ndim,nin,fin0,nout,
     1       fout(1,ind),umat_nd,ifadd)
      enddo
      
      return
      end subroutine
c
c
c
c      
C*********************************************************************
c
c      proxy potential -> potential on the input density tensor grid
C
C*********************************************************************
      subroutine bdmk_proxypot2pot(ndim,nd,nin,fin,nout,fout,umat)
c
c     this subroutine transform the input data given on a tensor product grid
c     to the output data of different size.
c
c     input:
c     nd - number of input data
c     nin - order of the expansion in each dimension of the input data
c     fin - potential on proxy tensor product grid
c     nout - order of the expansion in each dimension of the output data
c     umat - 1D transformation matrix along each direction
c      
c     output:
c     fout - potential on the original tensor product grid
c
c      implicit real *8 (a-h,o-z)
      implicit none
      integer nin,nout,nd,ndim,ind,j,np,i,ifadd
      real *8 fin(nin**ndim,nd)
      real *8 fout(nd,nout**ndim)
      
      real *8 umat(nout,nin)
      real *8 umat_nd(nout,nin,ndim)

      real *8 fout0(nout**ndim)

      np=nout*nin
      do i=1,ndim
         call dcopy_f77(np,umat,1,umat_nd(1,1,i),1)
      enddo

      ifadd=0
      do ind=1,nd
         call tens_prod_trans(ndim,nin,fin(1,ind),nout,fout0,
     1       umat_nd,ifadd)
         do j=1,nout**ndim
            fout(ind,j)=fout(ind,j)+fout0(j)
         enddo
      enddo
      
      return
      end subroutine
c
c
c
c*********************************************************************
C
C     Construct Fourier transform of weighted Gaussians
C
C*********************************************************************
      subroutine mk_kernel_Fourier_transform(dim,ndelta,deltas,
     1    dwhts,npw,ws,ts,nexp,wpwexp)
C
C     This subroutine precomputes nufft weights for "form mp" and 
C     "eval loc" stages
C
C     INPUT
C
c     dim     = dimension of the underlying space
c     ndelta  = number of Gaussians
c     deltas  = values of delta in these Gaussians
c     dwhts   = weights of each Gaussian
C     npw     = number of terms in plane wave exp
c     ws,ts   = weights and nodes of the 1d plane-wave expansion
C     nexp    = number of terms in the full plane-wave expansion
c      
C     OUTPUT:
C
C     wpwexp  - Fourier transform of weighted Gaussians
c
      implicit real *8 (a-h,o-z)
      integer dim
      real *8 deltas(ndelta),dwhts(ndelta)
      real *8 ws(npw),ts(npw),ww(npw,ndelta)
      
      real *8 wpwexp(nexp)

      do k=1,ndelta
         delta=deltas(k)
         do i=1,npw
            ww(i,k)=ws(i)*exp(-ts(i)**2*delta/4)*sqrt(delta)
         enddo
      enddo
      
      j=0
      if (dim.eq.1) then
         do j1=1,((npw+1)/2)
            j=j+1
            wpwexp(j) = 0
            do k=1,ndelta
               wpwexp(j) = wpwexp(j) + dwhts(k)*ww(j1,k)
            enddo
         enddo
      elseif (dim.eq.2) then
         do j2=1,((npw+1)/2)
         do j1=1,npw
            j=j+1
            wpwexp(j) = 0
            do k=1,ndelta
               wpwexp(j) = wpwexp(j) + dwhts(k)*ww(j1,k)*ww(j2,k)
            enddo
         enddo
         enddo
      elseif (dim.eq.3) then
         do j3=1,((npw+1)/2)
         do j2=1,npw
         do j1=1,npw
            j=j+1
            wpwexp(j) = 0
            do k=1,ndelta
               wpwexp(j) = wpwexp(j) + dwhts(k)
     1             *ww(j1,k)*ww(j2,k)*ww(j3,k)
            enddo
         enddo
         enddo
         enddo
      endif
c

      return
      end
c
c
c
c

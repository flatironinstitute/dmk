c 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c     this is the end of the debugging code and the beginning
c     of the tensor product routines
c 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c 
c 
c     This file contains a set of subroutines for the handling
c     of tensor product in n dimensions.
c     Following is a brief description of these
c     subroutines.
c
C*********************************************************************
c
c      real*8 -> real *8 tensor product transform
C
C*********************************************************************
c
c
c 
C*********************************************************************
      subroutine tens_prod_trans(ndim,nin,fin,nout,fout,umat,ifadd)
C*********************************************************************
c
c     this subroutine transform the input data given on a tensor product grid
c     to the output data of different size.
c
c     input:
c     ndim - dimension of the tensor
c     nin - order of the expansion in each dimension of the input tensor
c     fin - input tensor
c     nout - order of the expansion in each dimension of the output tensor
c     umat - 1D transformation matrices along each direction
c     ifadd - whether the original output tensor is added
c      
c     output:
c     fout - output tensor
c
      implicit none
      integer nin,nout,ndim,ifadd
      real *8 fin(nin**ndim)
      real *8 fout(nout**ndim)

      real *8 umat(nout,nin,ndim)

      if (ndim.eq.1) then
         call tens_prod_trans_1d(nin,fin,nout,fout,umat,ifadd)
      elseif (ndim.eq.2) then
         call tens_prod_trans_2d(nin,fin,nout,fout,umat,ifadd)
      elseif (ndim.eq.3) then
         call tens_prod_trans_3d(nin,fin,nout,fout,umat,ifadd)
      endif
      
      return
      end subroutine
c
c
c
      subroutine tens_prod_trans_1d(nin,fin,nout,fout,umat,ifadd)
      implicit none
      integer nin,nout,i,j,ifadd
      real *8 fin(nin)
      real *8 fout(nout),fout0(nout)
      real *8 umat(nout,nin)
      real *8 dd
      
      do j=1,nout
         dd=0
         do i=1,nin
            dd=dd+umat(j,i)*fin(i)
         enddo
         fout0(j)=dd
      enddo

      if (ifadd .eq. 1) then
         do j=1,nout
            fout(j)=fout(j)+fout0(j)
         enddo
      else
         do j=1,nout
            fout(j)=fout0(j)
         enddo
      endif
      
      return
      end
c
c
c
c
      subroutine tens_prod_trans_2d(nin,fin,nout,fout,umat,ifadd)
      implicit none
      integer nin,nout,ifadd
      real *8 alpha,beta
      real *8 fin(nin,nin)
      real *8 fout(nout,nout)

      real *8 umat(nout,nin,2)

      real *8 ff(nin,nout)

      alpha=1.0d0
      beta=0.0d0

c     transform in y
      call dgemm('n','t',nin,nout,nin,alpha,
     1    fin,nin,umat(1,1,2),nout,beta,ff,nin)

      beta=ifadd*1.0d0
c     transform in x
      call dgemm('n','n',nout,nout,nin,alpha,
     1    umat,nout,ff,nin,beta,fout,nout)
      
      return
      end
c
c
c
c
c
c
      subroutine tens_prod_trans_3d(nin,fin,nout,fout,umat,ifadd)
      implicit none
      integer nin,nout,j1,j2,k3,ifadd
      real *8 alpha,beta
      real *8 fin(nin,nin,nin)
      real *8 fout(nout,nout,nout)

      real *8 umat(nout,nin,3)

      real *8 ff(nin,nin,nout)
      real *8 fft(nin,nout,nin)
      real *8 ff2(nout,nout,nin)
      

      alpha=1.0d0
      beta=0.0d0

c     transform in z
      call dgemm('n','t',nin*nin,nout,nin,alpha,
     1    fin,nin*nin,umat(1,1,3),nout,beta,ff,nin*nin)

      do j1=1,nin
      do k3=1,nout
      do j2=1,nin
         fft(j2,k3,j1)=ff(j1,j2,k3)
      enddo
      enddo
      enddo

c     transform in y
      call dgemm('n','n',nout,nout*nin,nin,alpha,
     1    umat(1,1,2),nout,fft,nin,beta,ff2,nout)

c     transform in x
      beta = ifadd*1.0d0
      call dgemm('n','t',nout,nout*nout,nin,alpha,
     1    umat,nout,ff2,nout*nout,beta,fout,nout)

      return
      end
c      
c
c
c
      subroutine tens_prod_trans_add(ndim,nvec,nin,fin,nout,fout,
     1    umat)
c     ifadd = 1
c     additional input:
c     nvec - number of input tensors
c    
      implicit none
      integer ndim,nvec,nin,nout,iv,j,ifadd
      real *8 fin(nin**ndim,nvec)
      real *8 fout(nout**ndim,nvec)
      real *8 umat(nout,nin,ndim)

      ifadd=1
      do iv=1,nvec
         call tens_prod_trans(ndim,nin,fin(1,iv),nout,fout(1,iv),
     1       umat,ifadd)
      enddo
      
      return
      end
c
c
c
c
      subroutine tens_prod_trans_vec(ndim,nvec,nin,fin,nout,fout,
     1    umat)
c     ifadd=0
c     additional input:
c     nvec - number of input tensors
c
      implicit none
      integer ndim,nvec,nin,nout,iv,j,ifadd
      real *8 fin(nvec,nin**ndim)
      real *8 fout(nvec,nout**ndim)
      real *8 umat(nout,nin,ndim)

      real *8 fin0(nin**ndim)
      real *8 fout0(nout**ndim)
      
      ifadd=0
      do iv=1,nvec
         do j=1,nin**ndim
            fin0(j)=fin(iv,j)
         enddo
         call tens_prod_trans(ndim,nin,fin0,nout,fout0,
     1       umat,ifadd)
         do j=1,nout**ndim
            fout(iv,j)=fout0(j)
         enddo
      enddo
      
      return
      end
c
c
c
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     evaluate gradient at tensor grid given potential coefficients
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      subroutine ortho_evalg_nd(ndim,nd,norder,coefs,sc,
     1    grad,vmat,vpmat)
c
c     this subroutine evaluates the gradient at the tensor grid
c
c     input:
c     nd - number of input data
c     norder - order of polynomial expansion
c     coefs - orthogonal polynomial expansion coefficients for the potential
c     sc - scaling factor for the derivative
c     vmat - matrix converting coefficients into function values
c     vpmat - matrix converting coefficients into values of the first derivative
c      
c     output:
c     grad - value of the gradient
c
      implicit real *8 (a-h,o-z)
      real *8 coefs(nd,norder**ndim)
      real *8 grad(nd,ndim,norder**ndim)

      real *8 vmat(norder,norder)
      real *8 vpmat(norder,norder)

      if (ndim.eq.1) then
         call ortho_evalg_1d(nd,norder,coefs,sc,grad,vmat,vpmat)
      elseif (ndim.eq.2) then
         call ortho_evalg_2d(nd,norder,coefs,sc,grad,vmat,vpmat)
      elseif (ndim.eq.3) then
         call ortho_evalg_3d(nd,norder,coefs,sc,grad,vmat,vpmat)
      endif
      
      return
      end
c
c
c
c
C*********************************************************************C
      subroutine ortho_evalg_1d(nd,n,fcoefs,sc,grad,
     1    vmat,vpmat)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      real *8 fcoefs(nd,n),grad(nd,n)
      real *8 vmat(n,n)
      real *8 vpmat(n,n)
c
      
      do ind = 1,nd
c        transform in x
         do k1=1,n
            cdx=0
            do j1=1,n
               cdx=cdx+vpmat(k1,j1)*fcoefs(ind,j1)
            enddo
            grad(ind,k1)=cdx*sc
         enddo
      enddo
      
      return
      end subroutine
c
c
C
c
C*********************************************************************C
      subroutine ortho_evalg_2d(nd,n,fcoefs,sc,grad,
     1    vmat,vpmat)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      real *8 fcoefs(nd,n,n),pot(nd,n,n),grad(nd,2,n,n)
      real *8 vmat(n,n)
      real *8 vpmat(n,n)
      real *8 ff(n,n)
      real *8 ffx(n,n)
c
      do ind = 1,nd
c        transform in x
         do j2=1,n
         do k1=1,n
            cd=0
            cdx=0
            do j1=1,n
               cd=cd+vmat(k1,j1)*fcoefs(ind,j1,j2)
               cdx=cdx+vpmat(k1,j1)*fcoefs(ind,j1,j2)
            enddo
            ff(k1,j2)=cd
            ffx(k1,j2)=cdx
         enddo
         enddo
c        transfrom in y
         do k2=1,n
         do k1=1,n
c            cd=0
            cdx = 0.0d0
            cdy = 0.0d0
            do j2=1,n
               cdy=cdy+vpmat(k2,j2)*ff(k1,j2)
               cdx=cdx+vmat(k2,j2)*ffx(k1,j2)
            enddo
            grad(ind,1,k1,k2)=cdx*sc
            grad(ind,2,k1,k2)=cdy*sc
         enddo
         enddo
c     end of the ind loop
      enddo
      
      return
      end subroutine
c
c
C
c
C*********************************************************************C
      subroutine ortho_evalg_3d(nd,n,fcoefs,sc,grad,
     1    vmat,vpmat)
C*********************************************************************C
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      real *8 fcoefs(nd,n,n,n)
      real *8 grad(nd,3,n,n,n)
      real *8 vmat(n,n)
      real *8 vpmat(n,n)

      real *8 ff(n,n,n),ff2(n,n,n)
      real *8 ffx(n,n,n),ff2x(n,n,n)
      real *8 ff2y(n,n,n)
c
      
      do ind = 1,nd
c        transform in x
         do j3=1,n
         do j2=1,n
         do k1=1,n
            cd=0
            cdx=0.0d0
            do j1=1,n
               cd=cd+vmat(k1,j1)*fcoefs(ind,j1,j2,j3)
               cdx=cdx+vpmat(k1,j1)*fcoefs(ind,j1,j2,j3)
            enddo
            ff(k1,j2,j3)=cd
            ffx(k1,j2,j3)=cdx
         enddo
         enddo
         enddo

c        transform in y
         do j3=1,n
         do k2=1,n
         do k1=1,n
            cd=0
            cdx = 0.0d0
            cdy = 0.0d0
            do j2=1,n
               cd=cd+vmat(k2,j2)*ff(k1,j2,j3)
               cdy=cdy+vpmat(k2,j2)*ff(k1,j2,j3)
               
               cdx=cdx+vmat(k2,j2)*ffx(k1,j2,j3)
            enddo
            ff2(k1,k2,j3)=cd
            ff2x(k1,k2,j3)=cdx
            ff2y(k1,k2,j3)=cdy
         enddo
         enddo
         enddo

c        transform in z
         do k3=1,n
         do k2=1,n
         do k1=1,n
            cdx = 0.0d0
            cdy = 0.0d0
            cdz = 0.0d0
            do j3=1,n
               cdz=cdz+vpmat(k3,j3)*ff2(k1,k2,j3)
               
               cdx=cdx+vmat(k3,j3)*ff2x(k1,k2,j3)
               cdy=cdy+vmat(k3,j3)*ff2y(k1,k2,j3)
            enddo
            grad(ind,1,k1,k2,k3)=cdx*sc
            grad(ind,2,k1,k2,k3)=cdy*sc
            grad(ind,3,k1,k2,k3)=cdz*sc
         enddo
         enddo
         enddo
c     end of the ind loop
      enddo
      
      return
      end subroutine
c
c
c
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     evaluate gradient and hessian at tensor grid given potential coefficients
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine ortho_evalgh_nd(ndim,nd,norder,coefs,sc,grad,
     1    hess,vmat,vpmat,vppmat)
c
c     this subroutine evaluates the hessian at the tensor grid
c
c     input:
c     nd - number of input data
c     norder - order of polynomial expansion
c     coefs - orthogonal polynomial expansion coefficients for the potential
c     sc - scaling factor for the derivative
c     vmat - matrix converting coefficients into function values
c     vpmat - matrix converting coefficients into values of the first derivative
c     vppmat - matrix converting coefficients into values of the second derivative
c
c     output:
c     grad - value of the gradient
c     hess - value of the hessian
c
      implicit real *8 (a-h,o-z)
      real *8 coefs(nd,norder**ndim)
      real *8 grad(nd,ndim,norder**ndim)
      real *8 hess(nd,ndim*(ndim+1)/2,norder**ndim)

      real *8 vmat(norder,norder)
      real *8 vpmat(norder,norder)
      real *8 vppmat(norder,norder)

      if (ndim.eq.1) then
         call ortho_evalgh_1d(nd,norder,coefs,sc,grad,hess,vmat,
     1       vpmat,vppmat)
      elseif (ndim.eq.2) then
         call ortho_evalgh_2d(nd,norder,coefs,sc,grad,hess,vmat,
     1       vpmat,vppmat)
      elseif (ndim.eq.3) then
         call ortho_evalgh_3d(nd,norder,coefs,sc,grad,hess,vmat,
     1       vpmat,vppmat)
      endif
      
      return
      end
c
c
c
c
C*********************************************************************C
      subroutine ortho_evalgh_1d(nd,n,fcoefs,sc,grad,hess,
     1    vmat,vpmat,vppmat)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      real *8 fcoefs(nd,n),grad(nd,n),hess(nd,n)
      real *8 vmat(n,n)
      real *8 vpmat(n,n)
      real *8 vppmat(n,n)
c
      sc2=sc*sc
      
      do ind = 1,nd
c        transform in x
         do k1=1,n
c            cd=0
            cdx=0
            cdxx=0
            do j1=1,n
c               cd=cd+vmat(k1,j1)*fcoefs(ind,j1)
               cdx=cdx+vpmat(k1,j1)*fcoefs(ind,j1)
               cdxx=cdxx+vppmat(k1,j1)*fcoefs(ind,j1)
            enddo
            grad(ind,k1)=cdx*sc
            hess(ind,k1)=cdxx*sc2
         enddo
      enddo
      
      return
      end subroutine
c
c
C
c
      subroutine ortho_evalgh_2d(nd,n,fcoefs,sc,grad,hess,
     1    vmat,vpmat,vppmat)
      implicit real *8 (a-h,o-z)
      real *8 fcoefs(nd,n,n),grad(nd,2,n,n),hess(nd,3,n,n)
      real *8 vmat(n,n)
      real *8 vpmat(n,n)
      real *8 vppmat(n,n)

      real *8 ff(n,n)
      real *8 ffx(n,n)
      real *8 ffxx(n,n)
c
      sc2=sc*sc
      
      do ind = 1,nd
c        transform in x
         do j2=1,n
         do k1=1,n
            cd=0
            cdx=0
            cdxx=0
            do j1=1,n
               cd=cd+vmat(k1,j1)*fcoefs(ind,j1,j2)
               cdx=cdx+vpmat(k1,j1)*fcoefs(ind,j1,j2)
               cdxx=cdxx+vppmat(k1,j1)*fcoefs(ind,j1,j2)
            enddo
            ff(k1,j2)=cd
            ffx(k1,j2)=cdx
            ffxx(k1,j2)=cdxx
         enddo
         enddo
c        transfrom in y
         do k2=1,n
         do k1=1,n
c            cd = 0.0d0
            cdx = 0.0d0
            cdy = 0.0d0
            cdxx = 0.0d0
            cdxy = 0.0d0
            cdyy = 0.0d0
            do j2=1,n
c               cd=cd+vmat(k2,j2)*ff(k1,j2)
               cdy=cdy+vpmat(k2,j2)*ff(k1,j2)
               cdyy=cdyy+vppmat(k2,j2)*ff(k1,j2)

               cdx=cdx+vmat(k2,j2)*ffx(k1,j2)
               cdxy=cdxy+vpmat(k2,j2)*ffx(k1,j2)

               cdxx=cdxx+vmat(k2,j2)*ffxx(k1,j2)
            enddo
            grad(ind,1,k1,k2)=cdx*sc
            grad(ind,2,k1,k2)=cdy*sc

            hess(ind,1,k1,k2)=cdxx*sc2
            hess(ind,2,k1,k2)=cdxy*sc2
            hess(ind,3,k1,k2)=cdyy*sc2
         enddo
         enddo
c     end of the ind loop
      enddo
      
      return
      end subroutine
c
c
C
c
      subroutine ortho_evalgh_3d(nd,n,fcoefs,sc,grad,hess,
     1    vmat,vpmat,vppmat)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      real *8 fcoefs(nd,n,n,n)
c      real *8 pot(nd,n,n,n)
      real *8 grad(nd,3,n,n,n)
      real *8 hess(nd,6,n,n,n)

      real *8 vmat(n,n)
      real *8 vpmat(n,n)
      real *8 vppmat(n,n)

      real *8 ff(n,n,n)
      real *8 ffx(n,n,n)
      real *8 ffxx(n,n,n)
      
      real *8 ff2(n,n,n)
      real *8 ff2x(n,n,n)
      real *8 ff2xy(n,n,n)
      real *8 ff2xx(n,n,n)
      
      real *8 ff2y(n,n,n)
      real *8 ff2yy(n,n,n)
c
      sc2=sc*sc
      
      do ind = 1,nd
c        transform in x
         do j3=1,n
         do j2=1,n
         do k1=1,n
            cd=0
            cdx=0.0d0
            cdxx=0.0d0
            do j1=1,n
               cd=cd+vmat(k1,j1)*fcoefs(ind,j1,j2,j3)
               cdx=cdx+vpmat(k1,j1)*fcoefs(ind,j1,j2,j3)
               cdxx=cdxx+vppmat(k1,j1)*fcoefs(ind,j1,j2,j3)
            enddo
            ff(k1,j2,j3)=cd
            ffx(k1,j2,j3)=cdx
            ffxx(k1,j2,j3)=cdxx
         enddo
         enddo
         enddo

c        transform in y
         do j3=1,n
         do k2=1,n
         do k1=1,n
            cd=0
            cdx = 0.0d0
            cdy = 0.0d0
            cdxx = 0.0d0
            cdxy = 0.0d0
            cdyy = 0.0d0
            do j2=1,n
               cd   = cd   +   vmat(k2,j2)*ff(k1,j2,j3)
               cdy  = cdy  +  vpmat(k2,j2)*ff(k1,j2,j3)
               cdyy = cdyy + vppmat(k2,j2)*ff(k1,j2,j3)

               cdx  = cdx  +   vmat(k2,j2)*ffx(k1,j2,j3)
               cdxy = cdxy +  vpmat(k2,j2)*ffx(k1,j2,j3)
               
               cdxx = cdxx +   vmat(k2,j2)*ffxx(k1,j2,j3)
            enddo
            ff2(k1,k2,j3)=cd
            ff2x(k1,k2,j3)=cdx
            ff2xx(k1,k2,j3)=cdxx
            ff2y(k1,k2,j3)=cdy
            ff2yy(k1,k2,j3)=cdyy
            ff2xy(k1,k2,j3)=cdxy
         enddo
         enddo
         enddo

c        transform in z
         do k3=1,n
         do k2=1,n
         do k1=1,n
c            cd=0
            cdx = 0.0d0
            cdy = 0.0d0
            cdz = 0.0d0
            cdxx = 0.0d0
            cdyy = 0.0d0
            cdzz = 0.0d0
            cdxy = 0.0d0
            cdxz = 0.0d0
            cdyz = 0.0d0
            do j3=1,n
c               cd   = cd   +   vmat(k3,j3)*ff2(k1,k2,j3)
               cdz  = cdz  +  vpmat(k3,j3)*ff2(k1,k2,j3)
               cdzz = cdzz + vppmat(k3,j3)*ff2(k1,k2,j3)

               cdx  = cdx  +  vmat(k3,j3)*ff2x(k1,k2,j3)
               cdxz = cdxz + vpmat(k3,j3)*ff2x(k1,k2,j3)

               cdy  = cdy  +  vmat(k3,j3)*ff2y(k1,k2,j3)
               cdyz = cdyz + vpmat(k3,j3)*ff2y(k1,k2,j3)
               
               cdxx = cdxx + vmat(k3,j3)*ff2xx(k1,k2,j3)
               cdxy = cdxy + vmat(k3,j3)*ff2xy(k1,k2,j3)
               cdyy = cdyy + vmat(k3,j3)*ff2yy(k1,k2,j3)
            enddo
c            pot(ind,k1,k2,k3)=pot(ind,k1,k2,k3)+cd
            grad(ind,1,k1,k2,k3)=cdx*sc
            grad(ind,2,k1,k2,k3)=cdy*sc
            grad(ind,3,k1,k2,k3)=cdz*sc

            hess(ind,1,k1,k2,k3)=cdxx*sc2
            hess(ind,2,k1,k2,k3)=cdyy*sc2
            hess(ind,3,k1,k2,k3)=cdzz*sc2

            hess(ind,4,k1,k2,k3)=cdxy*sc2
            hess(ind,5,k1,k2,k3)=cdxz*sc2
            hess(ind,6,k1,k2,k3)=cdyz*sc2
         enddo
         enddo
         enddo
c     end of the ind loop
      enddo
      
      return
      end subroutine
c
c      
c      
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     evaluate the laplacian at tensor grid given potential coefficients
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine ortho_eval_laplacian_nd(ndim,nd,norder,coefs,sc,rlap,
     1    vmat,vppmat)
c
c     this subroutine evaluates the laplacian at the tensor grid
c
c     input:
c     ndim - dimension of the underlying space
c     nd - number of input data
c     norder - order of polynomial expansion
c     coefs - orthogonal polynomial expansion coefficients for the potential
c     sc - scaling factor for the derivative
c     vmat - matrix converting coefficients into function values
c     vppmat - matrix converting coefficients into values of the second derivative
c
c     output:
c     rlap - value of the laplacian on the tensor grid
c
      implicit real *8 (a-h,o-z)
      real *8 coefs(nd,norder**ndim)
      real *8 rlap(nd,norder**ndim)

      real *8 vmat(norder,norder)
      real *8 vppmat(norder,norder)

      if (ndim.eq.1) then
         call ortho_eval_laplacian_1d(nd,norder,coefs,sc,rlap,
     1    vmat,vppmat)
      elseif (ndim.eq.2) then
         call ortho_eval_laplacian_2d(nd,norder,coefs,sc,rlap,
     1    vmat,vppmat)
      elseif (ndim.eq.3) then
         call ortho_eval_laplacian_3d_blas(nd,norder,coefs,sc,rlap,
     1    vmat,vppmat)
      endif

      return
      end
c
c
c
c
C*********************************************************************C
      subroutine ortho_eval_laplacian_1d(nd,n,fcoefs,sc,rlap,
     1    vmat,vppmat)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      real *8 fcoefs(nd,n),rlap(nd,n)
      real *8 vmat(n,n)
      real *8 vppmat(n,n)
c
      sc2=sc*sc
      
      do ind = 1,nd
c        transform in x
         do k1=1,n
            cdxx=0
            do j1=1,n
               cdxx=cdxx+vppmat(k1,j1)*fcoefs(ind,j1)
            enddo
            rlap(ind,k1)=cdxx*sc2
         enddo
      enddo
      
      return
      end subroutine
c
c
C
c
      subroutine ortho_eval_laplacian_2d(nd,n,fcoefs,sc,rlap,
     1    vmat,vppmat)
      implicit real *8 (a-h,o-z)
      real *8 fcoefs(nd,n,n),rlap(nd,n,n)
      real *8 vmat(n,n)
      real *8 vppmat(n,n)

      real *8 ff(n,n)
      real *8 ffxx(n,n)
c
      sc2=sc*sc
      
      do ind = 1,nd
c        transform in x
         do j2=1,n
         do k1=1,n
            cd=0
            cdxx=0
            do j1=1,n
               cd=cd+vmat(k1,j1)*fcoefs(ind,j1,j2)
               cdxx=cdxx+vppmat(k1,j1)*fcoefs(ind,j1,j2)
            enddo
            ff(k1,j2)=cd
            ffxx(k1,j2)=cdxx
         enddo
         enddo
c        transfrom in y
         do k2=1,n
         do k1=1,n
            cdxx = 0.0d0
            cdyy = 0.0d0
            do j2=1,n
               cdxx=cdxx+vmat(k2,j2)*ffxx(k1,j2)
               cdyy=cdyy+vppmat(k2,j2)*ff(k1,j2)
            enddo
            rlap(ind,k1,k2)=(cdxx+cdyy)*sc2
         enddo
         enddo
c     end of the ind loop
      enddo
      
      return
      end subroutine
c
c
C
c
      subroutine ortho_eval_laplacian_3d(nd,n,fcoefs,sc,rlap,
     1    vmat,vppmat)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      real *8 fcoefs(nd,n,n,n)
      real *8 rlap(nd,n,n,n)

      real *8 vmat(n,n)
      real *8 vppmat(n,n)

      real *8 ff(n,n,n)
      real *8 ffxx(n,n,n)
      
      real *8 ff2(n,n,n)
      real *8 ff2xx(n,n,n)
      real *8 ff2yy(n,n,n)
c
      sc2=sc*sc
      
      do ind = 1,nd
c        transform in x
         do j3=1,n
         do j2=1,n
         do k1=1,n
            cd=0
            cdxx=0.0d0
            do j1=1,n
               cd=cd+vmat(k1,j1)*fcoefs(ind,j1,j2,j3)
               cdxx=cdxx+vppmat(k1,j1)*fcoefs(ind,j1,j2,j3)
            enddo
            ff(k1,j2,j3)=cd
            ffxx(k1,j2,j3)=cdxx
         enddo
         enddo
         enddo

c        transform in y
         do j3=1,n
         do k2=1,n
         do k1=1,n
            cd=0
            cdxx = 0.0d0
            cdyy = 0.0d0
            do j2=1,n
               cd   = cd   +   vmat(k2,j2)*ff(k1,j2,j3)
               cdyy = cdyy + vppmat(k2,j2)*ff(k1,j2,j3)
               cdxx = cdxx +   vmat(k2,j2)*ffxx(k1,j2,j3)
            enddo
            ff2(k1,k2,j3)=cd
            ff2xx(k1,k2,j3)=cdxx
            ff2yy(k1,k2,j3)=cdyy
         enddo
         enddo
         enddo

c        transform in z
         do k3=1,n
         do k2=1,n
         do k1=1,n
            cdxx = 0.0d0
            cdyy = 0.0d0
            cdzz = 0.0d0
            do j3=1,n
               cdxx = cdxx + vmat(k3,j3)*ff2xx(k1,k2,j3)
               cdyy = cdyy + vmat(k3,j3)*ff2yy(k1,k2,j3)
               cdzz = cdzz + vppmat(k3,j3)*ff2(k1,k2,j3)
            enddo
            rlap(ind,k1,k2,k3)=(cdxx+cdyy+cdzz)*sc2
         enddo
         enddo
         enddo
c     end of the ind loop
      enddo
      
      return
      end subroutine
c
c      
c      
      subroutine ortho_eval_laplacian_3d_blas(nd,n,fcoefs,sc,rlap,
     1    vmat,vppmat)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      real *8 fcoefs(nd,n,n,n)
      real *8 rlap(nd,n,n,n)

      real *8 vmat(n,n)
      real *8 vppmat(n,n)

      real *8 fv(n,n,n)

      real *8 ff(n,n,n)
      real *8 ffzz(n,n,n)

      real *8 fft(n,n,n)
      real *8 ffzzt(n,n,n)
      
      real *8 ff2(n,n,n)
      real *8 ff2zz(n,n,n)
      real *8 ff2yy(n,n,n)

      real *8 ff3xx(n,n,n)
      real *8 ff3yy(n,n,n)
      real *8 ff3zz(n,n,n)
c
      sc2=sc*sc

      alpha=1.0d0
      beta=0.0d0
      
      do ind = 1,nd
         do j3=1,n
         do j2=1,n
         do j1=1,n
            fv(j1,j2,j3)=fcoefs(ind,j1,j2,j3)
         enddo
         enddo
         enddo
         
c        transform in z
         call dgemm('n','t',n*n,n,n,alpha,
     1       fv,n*n,vmat,n,beta,ff,n*n)
         call dgemm('n','t',n*n,n,n,alpha,
     1       fv,n*n,vppmat,n,beta,ffzz,n*n)
         
         do j1=1,n
         do k3=1,n
         do j2=1,n
            fft(j2,k3,j1)=ff(j1,j2,k3)
         enddo
         enddo
         enddo

         do j1=1,n
         do k3=1,n
         do j2=1,n
            ffzzt(j2,k3,j1)=ffzz(j1,j2,k3)
         enddo
         enddo
         enddo

c        transform in y
         call dgemm('n','n',n,n*n,n,alpha,
     1       vmat,n,fft,n,beta,ff2,n)
         call dgemm('n','n',n,n*n,n,alpha,
     1       vmat,n,ffzzt,n,beta,ff2zz,n)
         call dgemm('n','n',n,n*n,n,alpha,
     1       vppmat,n,fft,n,beta,ff2yy,n)

c        transform in x
         call dgemm('n','t',n,n*n,n,alpha,
     1       vmat,n,ff2zz,n*n,beta,ff3zz,n)
         call dgemm('n','t',n,n*n,n,alpha,
     1       vmat,n,ff2yy,n*n,beta,ff3yy,n)
         call dgemm('n','t',n,n*n,n,alpha,
     1       vppmat,n,ff2,n*n,beta,ff3xx,n)
         
         do k3=1,n
         do k2=1,n
         do k1=1,n
            rlap(ind,k1,k2,k3)=(ff3xx(k1,k2,k3)
     1          +ff3yy(k1,k2,k3)+ff3zz(k1,k2,k3))*sc2
         enddo
         enddo
         enddo
c     end of the ind loop
      enddo
      
      return
      end subroutine
c
c      
c      
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     make 1D tables for computing function values and its derivatives
c     from its expansion coefficients
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine ortho_eval_tables(ipoly,n,umat,vmat,vpmat,vppmat)
c
c     returns 1d transformation tables 
c
c     input:
c     ipoly - ipoly=0 Legendre polynomials
c                   1 Chebyshev polynomials
c     n - order of polynomial expansion
c      
c     output:
c     umat - nxn matrix converting function values to expansion coefficients 
c     vmat - nxn matrix converting expansion coefficients to function values
c     vpmat - nxn matrix converting expansion coefficients to first derivatives
c     vppmat - nxn matrix converting expansion coefficients to second derivatives
c
      implicit real *8 (a-h,o-z)
      real *8 umat(n,n)
      real *8 vmat(n,n)
      real *8 vpmat(n,n)
      real *8 vppmat(n,n)
      real *8 xs(n),ws(n)

      itype=2
      if (ipoly.eq.0) then
         call legeexps2(itype,n,xs,umat,vmat,ws,vpmat,vppmat)
      elseif (ipoly.eq.1) then
         call chebexps2(itype,n,xs,umat,vmat,ws,vpmat,vppmat)
      endif

      return
      end
c
c
c
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     evaluate potential at arbitrary target 
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine ortho_evalt_nd(ndim,nd,ipoly,norder,coefs,targ,pot)
c
c     this subroutine evaluates the orthogonal polynomial expansion at the given target point
c
c     input:
c     nd - number of input data
c     ipoly - itype=0 Legendre polynomials
c                   1 Chebyshev polynomials
c     norder - order of polynomial expansion
c     coefs - orthogonal polynomial expansion coefficients
c     targ - xy coordinates of the target point
c      
c     output:
c     pot - value of the function at the target point
c
      implicit real *8 (a-h,o-z)
      real *8 coefs(nd,norder**ndim)
      real *8 targ(ndim),pot(nd)

      if (ndim.eq.1) then
         call ortho_evalt_1d(nd,ipoly,norder,coefs,targ,pot)
      elseif (ndim.eq.2) then
         call ortho_evalt_2d(nd,ipoly,norder,coefs,targ,pot)
      elseif (ndim.eq.3) then
         call ortho_evalt_3d(nd,ipoly,norder,coefs,targ,pot)
      endif
      
      return
      end
c
c
c
      subroutine ortho_evalt_1d(nd,ipoly,norder,coefs,
     1    targ,pot)
c
c     this subroutine evaluates the potential at the given target point
c
c     input:
c     nd - number of input data
c     ipoly - ipoly=0 Legendre polynomials
c                   1 Chebyshev polynomials
c     norder - order of polynomial expansion
c     coefs - orthogonal polynomial expansion coefficients for the potential
c     targ - xy coordinates of the target point
c      
c     output:
c     pot - value of the potential
c
      implicit real *8 (a-h,o-z)
      real *8 coefs(nd,norder)
      real *8 targ(1),pot(nd)

      real *8 px(norder)
      
      x=targ(1)
      
      if (ipoly .eq. 0) then
         call legepols(x,norder-1,px)
      elseif (ipoly .eq. 1) then
         call chebpols(x,norder-1,px)
      endif
c     
      do ind=1,nd
         pp=0
         do k=1,norder
            pp=pp+coefs(ind,k)*px(k)
         enddo
         pot(ind)=pot(ind)+pp
      enddo

      return
      end
c
c
c
c
      subroutine ortho_evalt_2d(nd,ipoly,norder,coefs,
     1    targ,pot)
      implicit real *8 (a-h,o-z)
      real *8 coefs(nd,norder,norder)
      real *8 targ(2),pot(nd)

      real *8 px(norder),py(norder)
      
      real *8, allocatable :: tmp(:,:)

      allocate(tmp(nd,norder))
      
      x=targ(1)
      y=targ(2)
      
      if (ipoly .eq. 0) then
         call legepols(x,norder-1,px)
         call legepols(y,norder-1,py)
      elseif (ipoly .eq. 1) then
         call chebpols(x,norder-1,px)
         call chebpols(y,norder-1,py)
      endif
c     
      do j=1,norder
         do ind=1,nd
            pp=0
            do k=1,norder
               pp=pp+coefs(ind,k,j)*px(k)
            enddo
            tmp(ind,j)=pp
         enddo
      enddo
c
      do ind=1,nd
         pp=0
         do j=1,norder
            pp=pp+tmp(ind,j)*py(j)
         enddo
         pot(ind)=pot(ind)+pp
      enddo

      return
      end
c
c
c
c
      subroutine ortho_evalt_3d(nd,ipoly,norder,coefs,
     1    targ,pot)
      implicit real *8 (a-h,o-z)
      real *8 coefs(nd,norder,norder,norder)
      real *8 targ(3),pot(nd)

      real *8 px(norder),py(norder),pz(norder)
      
      real *8, allocatable :: tmp(:,:,:),tmp2(:,:)

      allocate(tmp(nd,norder,norder))
      allocate(tmp2(nd,norder))
      
      x=targ(1)
      y=targ(2)
      z=targ(3)
      
      if (ipoly .eq. 0) then
         call legepols(x,norder-1,px)
         call legepols(y,norder-1,py)
         call legepols(z,norder-1,pz)
      elseif (ipoly .eq. 1) then
         call chebpols(x,norder-1,px)
         call chebpols(y,norder-1,py)
         call chebpols(z,norder-1,pz)
      endif
c     
      do i=1,norder
      do j=1,norder
         do ind=1,nd
            pp=0
            do k=1,norder
               pp=pp+coefs(ind,k,j,i)*px(k)
            enddo
            tmp(ind,j,i)=pp
         enddo 
      enddo
      enddo
c
      do i=1,norder
      do ind=1,nd
         pp=0
         do j=1,norder
            pp=pp+tmp(ind,j,i)*py(j)
         enddo
         tmp2(ind,i)=pp
      enddo
      enddo
c
      do ind=1,nd
         pp=0
         do j=1,norder
            pp=pp+tmp2(ind,j)*pz(j)
         enddo
         pot(ind)=pot(ind)+pp
      enddo
      
      return
      end
c
c
c
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     evaluate tensor product polynomial expansion at 
c     given target points
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine pdmk_ortho_evalt_nd(ndim,nd,norder,coefs,
     1    nt,targ,cen,sc,pot)
c     evaluate potential at targets given its tensor product polynomial expansion
c     coefficients
c
c     input:
c     nd - number of coefficient vectors
c     norder - polynomial expansion order
c     coefs - polynomial expansion coefficients
c     nt - number of targets
c     targ - coordinates of targets
c     cen - coordinates of the center of the box
c     sc - scaling factor that scales the points in the box to the standard interval [-1,1]
c
c     output:
c     pot - potential at targets
c
      implicit real *8 (a-h,o-z)
      real *8 coefs(norder**ndim,nd)
      real *8 targ(ndim,nt),pot(nd,nt),cen(ndim)

      if (ndim.eq.2) then
         call pdmk_ortho_evalt_2d(nd,norder,coefs,
     1       nt,targ,cen,sc,pot)
      elseif (ndim.eq.3) then
         call pdmk_ortho_evalt_3d(nd,norder,coefs,
     1       nt,targ,cen,sc,pot)
      endif

      return
      end
c
c
c
c
      subroutine pdmk_ortho_evalt_2d(nd,norder,coefs,
     1    nt,targ,cen,sc,pot)
c     evaluate potential at targets given its tensor product polynomial expansion
c     coefficients
c
c     input:
c     nd - number of coefficient vectors
c     norder - polynomial expansion order
c     coefs - polynomial expansion coefficients
c     nt - number of targets
c     targ - coordinates of targets
c     cen - coordinates of the center of the box
c     sc - scaling factor that scales the points in the box to the standard interval [-1,1]
c
c     output:
c     pot - potential at targets
c
      implicit real *8 (a-h,o-z)
      real *8 coefs(norder,norder,nd)
      real *8 targ(2,nt),pot(nd,nt),cen(2)

      real *8, allocatable :: tmp(:,:)
      real *8, allocatable :: px(:,:),py(:,:)

      allocate(px(norder,nt))
      allocate(py(norder,nt))

      allocate(tmp(norder,nt))

      do i=1,nt
         x=(targ(1,i)-cen(1))*sc
         call chebpols(x,norder-1,px(1,i))
      enddo

      do i=1,nt
         y=(targ(2,i)-cen(2))*sc
         call chebpols(y,norder-1,py(1,i))
      enddo

      alpha=1.0d0
      beta=0.0d0
c
c      incx=1
c      incy=1
      do ind=1,nd
c        transform in y
         call dgemm('n','n',norder,nt,norder,alpha,
     1       coefs(1,1,ind),norder,py,norder,beta,
     2       tmp,norder)
        
c
         do k=1,nt
            pp=0
            do i=1,norder
               pp=pp+tmp(i,k)*px(i,k)
            enddo
            pot(ind,k)=pot(ind,k)+pp
         enddo
      enddo
      
      return
      end
c
c
c
c
c
      subroutine pdmk_ortho_evalt_3d(nd,norder,coefs,
     1    nt,targ,cen,sc,pot)
c     evaluate potential at targets given its tensor product polynomial expansion
c     coefficients
c
c     input:
c     nd - number of coefficient vectors
c     norder - polynomial expansion order
c     coefs - polynomial expansion coefficients
c     nt - number of targets
c     targ - coordinates of targets
c     cen - coordinates of the center of the box
c     sc - scaling factor that scales the points in the box to the standard interval [-1,1]
c
c     output:
c     pot - potential at targets
c
      implicit real *8 (a-h,o-z)
      real *8 coefs(norder,norder,norder,nd)
      real *8 targ(3,nt),pot(nd,nt),cen(3)

      real *8, allocatable :: tmp(:,:,:),tmp2(:,:)
      real *8, allocatable :: px(:,:),py(:,:),pz(:,:)

      allocate(px(norder,nt))
      allocate(py(norder,nt))
      allocate(pz(norder,nt))

      allocate(tmp(norder,norder,nt))
      allocate(tmp2(norder,nt))

      do i=1,nt
         x=(targ(1,i)-cen(1))*sc
         call chebpols(x,norder-1,px(1,i))
      enddo

      do i=1,nt
         y=(targ(2,i)-cen(2))*sc
         call chebpols(y,norder-1,py(1,i))
      enddo

      do i=1,nt
         z=(targ(3,i)-cen(3))*sc
         call chebpols(z,norder-1,pz(1,i))
      enddo

      alpha=1.0d0
      beta=0.0d0
c
c      incx=1
c      incy=1
      do ind=1,nd
c        transform in z
         call dgemm('n','n',norder*norder,nt,norder,alpha,
     1       coefs(1,1,1,ind),norder*norder,pz,norder,beta,
     2       tmp,norder*norder)
        
c
         do k=1,nt
            do i=1,norder
               pp=0
               do j=1,norder
                  pp=pp+tmp(j,i,k)*px(j,k)
               enddo
               tmp2(i,k)=pp
            enddo
c            call dgemv('t',norder,norder,alpha,tmp(1,1,k),
c     1          norder,px(1,k),incx,beta,tmp2(1,k),incy)
         enddo

         do k=1,nt
            pp=0
            do i=1,norder
               pp=pp+tmp2(i,k)*py(i,k)
            enddo
            pot(ind,k)=pot(ind,k)+pp
         enddo
      enddo
      
      return
      end
c
c
c
c

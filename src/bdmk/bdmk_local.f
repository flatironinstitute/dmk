
c     This file contains routines for direct interactions in the box fgt
c     in n dimensions. The direct interactions are evaluated using tensor
c     product structure with precomputed 1d tables. Here we have taken
c     advantage of the fact that these tables contain many zeros when the
c     Gaussian variance is small.
c
c
c
C**************************************************************************
c
C     direction interactions by precomputed 1D tables
c     fast version - use sparse patterns of local tables
C
c******************************************************************************
      subroutine bdmk_tens_prod_to_potloc(ndim,nd,n,ws,fvals,pot,
     1    ntab,tab_loc,ind_loc,ixyz)
C*********************************************************************C
c     This routine computes the volume Gauss transform in general n dimensions
c     over a single box source distribution given as function values on a
c     tensor product grid.
c      
c     The target points have a fixed location w.r.t. source box
c     and the integrals of Gaussians times Legendre polynomials at 
c     those points is assumed to have been precomputed and stored 
c     in arrays tab_loc.
c     Thus, the specific geometric relation of the source and target
c     boxes are IMPLICITLY contained in these arrays.
c     There are many such relations in higher dimensions, but only a few one-dimensional
c     tables are needed corresponding to the range of possible shifts
c     of the box center in any single dimension.
c
c
c     Case 1: same level
c          _____ _____ ____  
c         |     |     |    | 
c         |     |     |    | 
c         |_____|_____|____| 
c         |     |     |    | 
c         |     |  T  |    |    target points in T
c         |_____|_____|____|    source box has offset in x and y.  
c         |     |     |    |    Because of separation of variables,
c         |     |     |    |    we can use 1D tables for desired 
c         |_____|_____|____|    offsets in x or y in range (-1,0,1). 
c
c     Case 2: different levels
c          _____ _____ ____  
c         |     |     |    | 
c         |     |     |    | 
c         |_____|_____|____| 
c         |     |A |  |    | 
c         |     |--|--| B  |   for target points in small box A, of 
c         |_____|__|__|____|   dimension D, adjacent large boxes can be   
c         |     |     |    |   offset by one of -3D/2,-D/2,D/2,3D/2
c         |     |     |    |   in either x, or y.
c         |_____|_____|____|   
c                              For target points in large box B, of
c                              dimension D, adjacent small boxes can be
c                              offset by one of -3D/4,-D/4,D/4,3D/4
c                              in either x, or y.
c
c     INPUT:
c     ndim        dimension of the underlying space
c     nd          vector length (for multiple RHS)
c     n           number of nodes along each dimension
c     ws          weight of the Gaussian
c     fvals       function values at tensor grid
c     tab_loc     precomputed tables of 1D integrals
c     ind_loc     precomputed nonzero pattern of tables
c                 This is used to speed up the matrix-vector multiplication,
c                 especially when the Gaussian is sharply peaked, i.e., small delta.
c
c     ixyz        pointers to local table, specify which local table should
c                 be used along each direction
c     OUTPUT:
c     pot         output on tensor product grid
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      real *8 fvals(nd,n**ndim),pot(nd,n*ndim)
      real *8 tab_loc(n,n,-ntab:ntab)
      integer ind_loc(2,n+1,-ntab:ntab)
      integer ixyz(ndim)

      if (ndim.eq.1) then
         call tens_prod_to_potloc_1d(nd,n,ws,fvals,pot,ntab,
     1    tab_loc,ind_loc,ixyz)
      elseif (ndim.eq.2) then
         call tens_prod_to_potloc_2d(nd,n,ws,fvals,pot,ntab,
     1    tab_loc,ind_loc,ixyz)
      elseif (ndim.eq.3) then
         call tens_prod_to_potloc_3d(nd,n,ws,fvals,pot,ntab,
     1    tab_loc,ind_loc,ixyz)
      endif
      
      return
      end subroutine
c
c
C
c
C
C
C*********************************************************************C
      subroutine tens_prod_to_potloc_1d(nd,n,ws,fvals,pot,
     1    ntab,tab_loc,ind_loc,ixyz)
C*********************************************************************C
c     This routine computes the 1D volume Gauss transform over a 
c     single box source distribution given as function values on
c     a tensor product grid.
c
c     INPUT:
c     nd          vector length (for multiple RHS)
c     n             number of nodes along each dimension
c     n           dimension of coeff array
c     fvals       function values at tensor grid
c     tab_loc     precomputed tables of 1D integrals
c     ind_loc     precomputed nonzero pattern of tables
c     ixy         pointers to local table, specify which local table should
c                 be used
c     OUTPUT:
c     pot         output on tensor product grid
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      real *8 fvals(nd,n),pot(nd,n)
      real *8 tab_loc(n,n,-ntab:ntab)
      integer ind_loc(2,n+1,-ntab:ntab)
      integer ixyz(1)
c
      ix = ixyz(1)
      nx = ind_loc(2,n+1,ix)-ind_loc(1,n+1,ix)+1

      if (nx.eq.0) return
      
      do ind = 1,nd
c        transform in x
         do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
            cd=0
            do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
               cd=cd+tab_loc(j1,k1,ix)*fvals(ind,j1)
            enddo
            pot(ind,k1)=pot(ind,k1)+cd*ws
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
      subroutine tens_prod_to_potloc_2d(nd,n,ws,fvals,pot,
     1    ntab,tab_loc,ind_loc,ixy)
C*********************************************************************C
c     This routine computes the 2D volume Gauss transform over a 
c     single box source distribution given as function values on
c     a tensor product grid.
c
c     INPUT:
c     nd          vector length (for multiple RHS)
c     n             number of nodes along each dimension
c     n           dimension of coeff array
c     fvals       function values at tensor grid
c     tab_loc     precomputed tables of 1D integrals
c     ind_loc     precomputed nonzero pattern of tables
c     ixy         pointers to local table, specify which local table should
c                 be used
c     OUTPUT:
c     pot         output on tensor product grid
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      real *8 fvals(nd,n,n),pot(nd,n,n)
      real *8 tab_loc(n,n,-ntab:ntab)
      integer ind_loc(2,n+1,-ntab:ntab)
      integer ixy(2)
      real *8 ff(n,n)
c
      ix = ixy(1)
      iy = ixy(2)
      
      nx = ind_loc(2,n+1,ix)-ind_loc(1,n+1,ix)+1
      ny = ind_loc(2,n+1,iy)-ind_loc(1,n+1,iy)+1

      if (nx.eq.0 .or. ny.eq.0) return

      if (nx+ny .ge. n/2) then
         call tens_prod_to_potloc_blas_2d(nd,n,ws,fvals,pot,
     1       ntab,tab_loc,ind_loc,ixy)
         return
      endif
      
      if (nx.le.ny) then
      
      do ind = 1,nd
c        transform in x
         do j2=1,n
            do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
               cd=0
               do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
                  cd=cd+tab_loc(j1,k1,ix)*fvals(ind,j1,j2)
               enddo
               ff(k1,j2)=cd
            enddo
         enddo
c        transfrom in y
         do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
            do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
               cd=0
               do j2=ind_loc(1,k2,iy),ind_loc(2,k2,iy)
                  cd=cd+tab_loc(j2,k2,iy)*ff(k1,j2)
               enddo
               pot(ind,k1,k2)=pot(ind,k1,k2)+cd*ws
            enddo
         enddo
c     end of the ind loop
      enddo
      else
      do ind = 1,nd
c        transform in y
         do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
            do j1=1,n
               cd=0
               do j2=ind_loc(1,k2,iy),ind_loc(2,k2,iy)
                  cd=cd+tab_loc(j2,k2,iy)*fvals(ind,j1,j2)
               enddo
               ff(j1,k2)=cd
            enddo
         enddo
c        transfrom in x
         do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
            do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
               cd=0
               do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
                  cd=cd+tab_loc(j1,k1,ix)*ff(j1,k2)
               enddo
               pot(ind,k1,k2)=pot(ind,k1,k2)+cd*ws
            enddo
         enddo
c     end of the ind loop
      enddo
      endif
      
      return
      end subroutine
c
c
C
c
C*********************************************************************C
      subroutine tens_prod_to_potloc_3d(nd,n,ws,fvals,pot,
     1    ntab,tab_loc,ind_loc,ixyz)
C*********************************************************************C
c     This routine computes 3D volume Gauss transform over a 
c     single box source distribution given as function values
c     on a tensor product grid.
c
c     INPUT:
c     nd          vector length (for multiple RHS)
c     n             number of nodes along each dimension
c     n           dimension of coeff array
c     fvals       function values at tensor grid
c     tab_loc     precomputed tables of 1D integrals
c     ind_loc     nonzero patterns of tab_loc
c     ixyz        pointers to local table, specify which local table should
c                 be used
c
c     OUTPUT:
c     pot         output on tensor product grid
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      real *8 fvals(nd,n,n,n)
      real *8 pot(nd,n,n,n)
      real *8 tab_loc(n,n,-ntab:ntab)
      integer ind_loc(2,n+1,-ntab:ntab)
      integer ixyz(3)

      real *8 ff(n,n,n),ff2(n,n,n)
c

      ix=ixyz(1)
      iy=ixyz(2)
      iz=ixyz(3)
      
      nx = ind_loc(2,n+1,ix)-ind_loc(1,n+1,ix)+1
      ny = ind_loc(2,n+1,iy)-ind_loc(1,n+1,iy)+1
      nz = ind_loc(2,n+1,iz)-ind_loc(1,n+1,iz)+1

      if (nx.eq.0 .or. ny.eq.0 .or. nz.eq.0) return

      if (nx+ny+nz .ge. n) then
         call tens_prod_to_potloc_blas_3d(nd,n,ws,fvals,pot,
     1       ntab,tab_loc,ind_loc,ixyz)
         return
      endif
      
      if (nx.le.ny .and. ny.le.nz) then
      do ind = 1,nd
c        transform in x
         do j3=1,n
            do j2=1,n
               do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
                  cd=0
                  do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
                     cd=cd+tab_loc(j1,k1,ix)*fvals(ind,j1,j2,j3)
                  enddo
                  ff(k1,j2,j3)=cd
               enddo
            enddo
         enddo

c        transform in y
         do j3=1,n
            do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
               do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
                  cd=0
                  do j2=ind_loc(1,k2,iy),ind_loc(2,k2,iy)
                     cd=cd+tab_loc(j2,k2,iy)*ff(k1,j2,j3)
                  enddo
                  ff2(k1,k2,j3)=cd
               enddo
            enddo
         enddo

c        transform in z
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
            do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
               do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
                  cd=0
                  do j3=ind_loc(1,k3,iz),ind_loc(2,k3,iz)
                     cd=cd+tab_loc(j3,k3,iz)*ff2(k1,k2,j3)
                  enddo
                  pot(ind,k1,k2,k3)=pot(ind,k1,k2,k3)+cd*ws
               enddo
            enddo
         enddo
c     end of the ind loop
      enddo
      elseif (nx .le. nz .and. nz.lt.ny) then
      do ind = 1,nd
c        transform in x
         do j3=1,n
            do j2=1,n
               do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
                  cd=0
                  do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
                     cd=cd+tab_loc(j1,k1,ix)*fvals(ind,j1,j2,j3)
                  enddo
                  ff(k1,j2,j3)=cd
               enddo
            enddo
         enddo

c        transform in z
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
            do j2=1,n
               do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
                  cd=0
                  do j3=ind_loc(1,k3,iz),ind_loc(2,k3,iz)
                     cd=cd+tab_loc(j3,k3,iz)*ff(k1,j2,j3)
                  enddo
                  ff2(k1,j2,k3)=cd
               enddo
            enddo
         enddo

c        transform in y
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
            do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
               do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
                  cd=0
                  do j2=ind_loc(1,k2,iy),ind_loc(2,k2,iy)
                     cd=cd+tab_loc(j2,k2,iy)*ff2(k1,j2,k3)
                  enddo
                  pot(ind,k1,k2,k3)=pot(ind,k1,k2,k3)+cd*ws
               enddo
            enddo
         enddo
c     end of the ind loop
      enddo
      elseif (ny .lt. nx .and. nx.le.nz) then
      do ind = 1,nd
c        transform in y
         do j3=1,n
            do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
               do j1=1,n
                  cd=0
                  do j2=ind_loc(1,k2,iy),ind_loc(2,k2,iy)
                     cd=cd+tab_loc(j2,k2,iy)*fvals(ind,j1,j2,j3)
                  enddo
                  ff(j1,k2,j3)=cd
               enddo
            enddo
         enddo

c        transform in x
         do j3=1,n
            do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
               do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
                  cd=0
                  do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
                     cd=cd+tab_loc(j1,k1,ix)*ff(j1,k2,j3)
                  enddo
                  ff2(k1,k2,j3)=cd
               enddo
            enddo
         enddo

c        transform in z
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
            do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
               do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
                  cd=0
                  do j3=ind_loc(1,k3,iz),ind_loc(2,k3,iz)
                     cd=cd+tab_loc(j3,k3,iz)*ff2(k1,k2,j3)
                  enddo
                  pot(ind,k1,k2,k3)=pot(ind,k1,k2,k3)+cd*ws
               enddo
            enddo
         enddo
c     end of the ind loop
      enddo
      elseif (ny .le. nz .and. nz.lt.nx) then
      do ind = 1,nd
c        transform in y
         do j3=1,n
            do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
               do j1=1,n
                  cd=0
                  do j2=ind_loc(1,k2,iy),ind_loc(2,k2,iy)
                     cd=cd+tab_loc(j2,k2,iy)*fvals(ind,j1,j2,j3)
                  enddo
                  ff(j1,k2,j3)=cd
               enddo
            enddo
         enddo

c        transform in z
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
            do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
               do j1=1,n
                  cd=0
                  do j3=ind_loc(1,k3,iz),ind_loc(2,k3,iz)
                     cd=cd+tab_loc(j3,k3,iz)*ff(j1,k2,j3)
                  enddo
                  ff2(j1,k2,k3)=cd
               enddo
            enddo
         enddo

c        transform in x
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
            do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
               do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
                  cd=0
                  do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
                     cd=cd+tab_loc(j1,k1,ix)*ff2(j1,k2,k3)
                  enddo
                  pot(ind,k1,k2,k3)=pot(ind,k1,k2,k3)+cd*ws
               enddo
            enddo
         enddo
c     end of the ind loop
      enddo      
      elseif (nz .lt. nx .and. nx.le.ny) then
      do ind = 1,nd
c        transform in z
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
            do j2=1,n
               do j1=1,n
                  cd=0
                  do j3=ind_loc(1,k3,iz),ind_loc(2,k3,iz)
                     cd=cd+tab_loc(j3,k3,iz)*fvals(ind,j1,j2,j3)
                  enddo
                  ff(j1,j2,k3)=cd
               enddo
            enddo
         enddo

c        transform in x
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
            do j2=1,n
               do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
                  cd=0
                  do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
                     cd=cd+tab_loc(j1,k1,ix)*ff(j1,j2,k3)
                  enddo
                  ff2(k1,j2,k3)=cd
               enddo
            enddo
         enddo

c        transform in y
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
            do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
               do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
                  cd=0
                  do j2=ind_loc(1,k2,iy),ind_loc(2,k2,iy)
                     cd=cd+tab_loc(j2,k2,iy)*ff2(k1,j2,k3)
                  enddo
                  pot(ind,k1,k2,k3)=pot(ind,k1,k2,k3)+cd*ws
               enddo
            enddo
         enddo
c     end of the ind loop
      enddo
      elseif (nz .lt. ny .and. ny.lt.nx) then
      do ind = 1,nd
c        transform in z
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
            do j2=1,n
               do j1=1,n
                  cd=0
                  do j3=ind_loc(1,k3,iz),ind_loc(2,k3,iz)
                     cd=cd+tab_loc(j3,k3,iz)*fvals(ind,j1,j2,j3)
                  enddo
                  ff(j1,j2,k3)=cd
               enddo
            enddo
         enddo

c        transform in y
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
            do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
               do j1=1,n
                  cd=0
                  do j2=ind_loc(1,k2,iy),ind_loc(2,k2,iy)
                     cd=cd+tab_loc(j2,k2,iy)*ff(j1,j2,k3)
                  enddo
                  ff2(j1,k2,k3)=cd
               enddo
            enddo
         enddo

c        transform in x
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
            do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
               do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
                  cd=0
                  do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
                     cd=cd+tab_loc(j1,k1,ix)*ff2(j1,k2,k3)
                  enddo
                  pot(ind,k1,k2,k3)=pot(ind,k1,k2,k3)+cd*ws
               enddo
            enddo
         enddo
c     end of the ind loop
      enddo      
      endif

      
      return
      end subroutine
c
C
c
C
c
C
c
C
      subroutine tens_prod_to_potloc_blas_2d(nd,n,ws,fvals,pot,
     1    ntab,tab_loc,ind_loc,ixy)
C*********************************************************************C
c     This routine computes 3D volume Gauss transform over a 
c     single box source distribution given as function values
c     on a tensor product grid.
c
c     INPUT:
c     nd            vector length (for multiple RHS)
c     n             number of nodes along each dimension
c     fvals         function values at tensor grid
c     tab_loc       precomputed tables of 1D integrals
c     tabx_loc      precomputed tables for first derivatives
c     tabxx_loc     precomputed tables for second derivatives
c     ind_loc       precomputed nonzero pattern of tables
c     ixy           pointers to local table, specify which local table should
c                   be used
c     OUTPUT:
c     pot           potential values on tensor product grid
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      real *8 fvals(nd,n,n)
      real *8 pot(nd,n,n)

      real *8 tab_loc(n,n,-ntab:ntab)
      integer ind_loc(2,n+1,-ntab:ntab)
      integer ixy(2)

      real *8 fv(n,n)
      real *8 ff(n,n)
      real *8 dpot(n,n)
c
      ix=ixy(1)
      iy=ixy(2)

      alpha=1.0d0
      beta=0.0d0
      
      do ind = 1,nd
         do j2=1,n
         do j1=1,n
            fv(j1,j2)=fvals(ind,j1,j2)
         enddo
         enddo
      
c        transform in y
         call dgemm('n','n',n,n,n,alpha,
     1       fv,n,tab_loc(1,1,iy),n,beta,ff,n)

c        transform in x
         call dgemm('t','n',n,n,n,alpha,
     1       tab_loc(1,1,ix),n,ff,n,beta,dpot,n)
         
         do k2=1,n
         do k1=1,n
            pot(ind,k1,k2)=pot(ind,k1,k2)+ws*dpot(k1,k2)
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
C
C
c
C
      subroutine tens_prod_to_potloc_blas_3d(nd,n,ws,fvals,pot,
     1    ntab,tab_loc,ind_loc,ixyz)
C*********************************************************************C
c     This routine computes 3D volume Gauss transform over a 
c     single box source distribution given as function values
c     on a tensor product grid.
c
c     INPUT:
c     nd            vector length (for multiple RHS)
c     n             number of nodes along each dimension
c     fvals         function values at tensor grid
c     tab_loc       precomputed tables of 1D integrals
c     tabx_loc      precomputed tables for first derivatives
c     tabxx_loc     precomputed tables for second derivatives
c     ind_loc       precomputed nonzero pattern of tables
c     ixy           pointers to local table, specify which local table should
c                   be used
c     OUTPUT:
c     pot           potential values on tensor product grid
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      real *8 fvals(nd,n,n,n)
      real *8 pot(nd,n,n,n)

      real *8 tab_loc(n,n,-ntab:ntab)
      integer ind_loc(2,n+1,-ntab:ntab)
      integer ixyz(3)

      real *8 fv(n,n,n)
      real *8 ff(n,n,n)
      real *8 fft(n,n,n)
      real *8 ff2(n,n,n)
      real *8 dpot(n,n,n)
c
      ix=ixyz(1)
      iy=ixyz(2)
      iz=ixyz(3)

      alpha=1.0d0
      beta=0.0d0
      
      do ind = 1,nd
         do j3=1,n
         do j2=1,n
         do j1=1,n
            fv(j1,j2,j3)=fvals(ind,j1,j2,j3)
         enddo
         enddo
         enddo
      
c        transform in z
         call dgemm('n','n',n*n,n,n,alpha,
     1       fv,n*n,tab_loc(1,1,iz),n,beta,ff,n*n)

         do j1=1,n
         do k3=1,n
         do j2=1,n
            fft(j2,k3,j1)=ff(j1,j2,k3)
         enddo
         enddo
         enddo
c        transform in y
         call dgemm('t','n',n,n*n,n,alpha,
     1       tab_loc(1,1,iy),n,fft,n,beta,ff2,n)

c        transform in x
         call dgemm('t','t',n,n*n,n,alpha,
     1       tab_loc(1,1,ix),n,ff2,n*n,beta,dpot,n)
         
         do k3=1,n
         do k2=1,n
         do k1=1,n
            pot(ind,k1,k2,k3)=pot(ind,k1,k2,k3)+ws*dpot(k1,k2,k3)
         enddo
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
C
      subroutine bdmk_find_loctab_ind(ndim,iperiod,tcenter,scenter,
     1    sboxsize,bs0,mrefinelev,ixyz)
c     returns an index arrary used in bdmk_tens_prod_to_potloc
c
c     input:
c     ndim - dimension of the underlying space
c     iperiod - 0: free space; 1: doubly periodic
c     tcenter - target box center
c     scenter - source box center
c     sboxsize - source box size
c     bs0 - root box size
c     mrefinelev - maximum refinement level
c
c     output
c     ixyz - an index array determining which local table 
c            should be used along each dimension
c
      implicit real *8 (a-h,o-z)
      real *8 tcenter(ndim),scenter(ndim)
      integer ixyz(ndim),i

      bs=sboxsize/2**(2+mrefinelev)

      do i=1,ndim
         dx = nint((tcenter(i)-scenter(i))/bs)
         if (iperiod .eq. 1) then
            dxp1=dx-bs0/bs
            dxm1=dx+bs0/bs
            if (abs(dx).gt.abs(dxp1)) dx=dxp1
            if (abs(dx).gt.abs(dxm1)) dx=dxm1
         endif
         ixyz(i)=dx
      enddo
      
      
      return
      end subroutine
c
c
c
c
C**************************************************************************
c
C     direction interactions by precomputed 1D tables
C
C*********************************************************************C
      subroutine bdmk_tens_prod_to_pghloc(ndim,nd,n,ws,fvals,
     1    ifpgh,pot,grad,hess,ntab,
     2    tab_loc,tabx_loc,tabxx_loc,ind_loc,ixyz)
C*********************************************************************C
c     nd            vector length (for multiple RHS)
c     n             number of nodes along each dimension
c     ws            weight of the Gaussian in the SOG approximation of the kernel
c     fvals         function values at tensor grid
c     tab_loc       precomputed tables of 1D integrals
c     tabx_loc      precomputed tables for first derivatives
c     tabxx_loc     precomputed tables for second derivatives
c     ind_loc       precomputed nonzero pattern of tables
c     ixyz          pointers to local table, specify which local table should
c                   be used
c     OUTPUT:
c     pot           potential values on tensor product grid
c     grad          gradient values on tensor product grid
c     hess          hessian values on tensor product grid
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      real *8 fvals(nd,n**ndim)
      real *8 pot(nd,n**ndim)
      real *8 grad(nd,ndim,n**ndim)
      real *8 hess(nd,ndim*(ndim+1)/2,n**ndim)

      real *8 tab_loc(n,n,-ntab:ntab)
      real *8 tabx_loc(n,n,-ntab:ntab)
      real *8 tabxx_loc(n,n,-ntab:ntab)
      integer ind_loc(2,n+1,-ntab:ntab)
      integer ixyz(ndim)

      if (ifpgh.eq.1) goto 1100
      if (ifpgh.eq.2) goto 2200
      if (ifpgh.eq.3) goto 3300

 1100 continue
      call bdmk_tens_prod_to_potloc(ndim,nd,n,ws,fvals,pot,
     1    ntab,tab_loc,ind_loc,ixyz)      
      return
      
 2200 continue
      if (ndim.eq.1) then
         call tens_prod_to_pgloc_1d(nd,n,ws,fvals,pot,grad,
     1    ntab,tab_loc,tabx_loc,ind_loc,ixyz)
      elseif (ndim.eq.2) then
         call tens_prod_to_pgloc_2d(nd,n,ws,fvals,pot,grad,
     1    ntab,tab_loc,tabx_loc,ind_loc,ixyz)
      elseif (ndim.eq.3) then
         call tens_prod_to_pgloc_3d(nd,n,ws,fvals,pot,grad,
     1    ntab,tab_loc,tabx_loc,ind_loc,ixyz)
      endif
      return

 3300 continue
      if (ndim.eq.1) then
         call tens_prod_to_pghloc_1d(nd,n,ws,fvals,pot,grad,hess,
     1    ntab,tab_loc,tabx_loc,tabxx_loc,ind_loc,ixyz)
      elseif (ndim.eq.2) then
         call tens_prod_to_pghloc_2d(nd,n,ws,fvals,pot,grad,hess,
     1    ntab,tab_loc,tabx_loc,tabxx_loc,ind_loc,ixyz)
      elseif (ndim.eq.3) then
         call tens_prod_to_pghloc_3d(nd,n,ws,fvals,pot,grad,hess,
     1    ntab,tab_loc,tabx_loc,tabxx_loc,ind_loc,ixyz)
      endif
         
      return
      end subroutine
c
c
C
c
C
C
C**************************************************************************
c
C     direction interactions by precomputed 1D tables
C     potential+gradient
C
C*********************************************************************C
      subroutine tens_prod_to_pgloc_1d(nd,n,ws,fvals,pot,grad,
     1    ntab,tab_loc,tabx_loc,ind_loc,ixyz)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      real *8 fvals(nd,n),pot(nd,n),grad(nd,n)
      real *8 tab_loc(n,n,-ntab:ntab)
      real *8 tabx_loc(n,n,-ntab:ntab)
      integer ind_loc(2,n+1,-ntab:ntab)
      integer ixyz(1)
c
      ix = ixyz(1)
      nx = ind_loc(2,n+1,ix)-ind_loc(1,n+1,ix)+1

      if (nx.eq.0) return
      
      do ind = 1,nd
c        transform in x
         do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
            cd=0
            cdx=0
            do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
               cd=cd+tab_loc(j1,k1,ix)*fvals(ind,j1)
               cdx=cdx+tabx_loc(j1,k1,ix)*fvals(ind,j1)
            enddo
            pot(ind,k1)=pot(ind,k1)+cd*ws
            grad(ind,k1)=grad(ind,k1)+cdx*ws
         enddo
      enddo
      
      return
      end subroutine
c
c
C
c
      subroutine tens_prod_to_pgloc_2d(nd,n,ws,fvals,pot,grad,
     1    ntab,tab_loc,tabx_loc,ind_loc,ixy)
C*********************************************************************C
      implicit real *8 (a-h,o-z)
      real *8 fvals(nd,n,n),pot(nd,n,n),grad(nd,2,n,n)
      real *8 tab_loc(n,n,-ntab:ntab)
      real *8 tabx_loc(n,n,-ntab:ntab)
      integer ind_loc(2,n+1,-ntab:ntab)
      integer ixy(2)
      real *8 ff(n,n)
      real *8 ffx(n,n)
c
      ix = ixy(1)
      iy = ixy(2)
      
      nx = ind_loc(2,n+1,ix)-ind_loc(1,n+1,ix)+1
      ny = ind_loc(2,n+1,iy)-ind_loc(1,n+1,iy)+1

      if (nx.eq.0 .or. ny.eq.0) return
      
      do ind = 1,nd
c        transform in x
         do j2=1,n
         do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
            cd=0
            cdx=0
            do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
               cd=cd+tab_loc(j1,k1,ix)*fvals(ind,j1,j2)
               cdx=cdx+tabx_loc(j1,k1,ix)*fvals(ind,j1,j2)
            enddo
            ff(k1,j2)=cd
            ffx(k1,j2)=cdx
         enddo
         enddo
c        transfrom in y
         do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
         do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
            cd = 0.0d0
            cdx = 0.0d0
            cdy = 0.0d0
            do j2=ind_loc(1,k2,iy),ind_loc(2,k2,iy)
               cd=cd+tab_loc(j2,k2,iy)*ff(k1,j2)
               cdy=cdy+tabx_loc(j2,k2,iy)*ff(k1,j2)
               cdx=cdx+tab_loc(j2,k2,iy)*ffx(k1,j2)
            enddo
            pot(ind,k1,k2)=pot(ind,k1,k2)+cd*ws
            grad(ind,1,k1,k2)=grad(ind,1,k1,k2)+cdx*ws
            grad(ind,2,k1,k2)=grad(ind,2,k1,k2)+cdy*ws
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
      subroutine tens_prod_to_pgloc_3d(nd,n,ws,fvals,pot,grad,
     1    ntab,tab_loc,tabx_loc,ind_loc,ixyz)
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      real *8 fvals(nd,n,n,n)
      real *8 pot(nd,n,n,n)
      real *8 grad(nd,3,n,n,n)

      real *8 tab_loc(n,n,-ntab:ntab)
      real *8 tabx_loc(n,n,-ntab:ntab)
      integer ind_loc(2,n+1,-ntab:ntab)
      integer ixyz(3)

      real *8 ff(n,n,n)
      real *8 ffx(n,n,n)
      
      real *8 ff2(n,n,n)
      real *8 ff2x(n,n,n)
      
      real *8 ff2y(n,n,n)
c

      ix=ixyz(1)
      iy=ixyz(2)
      iz=ixyz(3)
      
      nx = ind_loc(2,n+1,ix)-ind_loc(1,n+1,ix)+1
      ny = ind_loc(2,n+1,iy)-ind_loc(1,n+1,iy)+1
      nz = ind_loc(2,n+1,iz)-ind_loc(1,n+1,iz)+1

      if (nx.eq.0 .or. ny.eq.0 .or. nz.eq.0) return
      
      do ind = 1,nd
c        transform in x
         do j3=1,n
         do j2=1,n
         do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
c         do k1=1,n
            cd=0
            cdx=0.0d0
            do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
c            do j1=1,n
               cd=cd+tab_loc(j1,k1,ix)*fvals(ind,j1,j2,j3)
               cdx=cdx+tabx_loc(j1,k1,ix)*fvals(ind,j1,j2,j3)
            enddo
            ff(k1,j2,j3)=cd
            ffx(k1,j2,j3)=cdx
         enddo
         enddo
         enddo

c        transform in y
         do j3=1,n
         do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
         do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
c         do k2=1,n
c         do k1=1,n
            cd=0
            cdx = 0.0d0
            cdy = 0.0d0
            do j2=ind_loc(1,k2,iy),ind_loc(2,k2,iy)
c            do j2=1,n
               cd   = cd   +   tab_loc(j2,k2,iy)*ff(k1,j2,j3)
               cdy  = cdy  +  tabx_loc(j2,k2,iy)*ff(k1,j2,j3)
               cdx  = cdx  +   tab_loc(j2,k2,iy)*ffx(k1,j2,j3)
            enddo
            ff2(k1,k2,j3)=cd
            ff2x(k1,k2,j3)=cdx
            ff2y(k1,k2,j3)=cdy
         enddo
         enddo
         enddo

c        transform in z
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
         do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
         do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
c         do k3=1,n
c         do k2=1,n
c         do k1=1,n
            cd=0
            cdx = 0.0d0
            cdy = 0.0d0
            cdz = 0.0d0
            do j3=ind_loc(1,k3,iz),ind_loc(2,k3,iz)
c            do j3=1,n
               cd   = cd   +   tab_loc(j3,k3,iz)*ff2(k1,k2,j3)
               cdz  = cdz  +  tabx_loc(j3,k3,iz)*ff2(k1,k2,j3)
               cdx  = cdx  +  tab_loc(j3,k3,iz)*ff2x(k1,k2,j3)
               cdy  = cdy  +  tab_loc(j3,k3,iz)*ff2y(k1,k2,j3)
            enddo
            pot(ind,k1,k2,k3)=pot(ind,k1,k2,k3)+cd*ws

            grad(ind,1,k1,k2,k3)=grad(ind,1,k1,k2,k3)+cdx*ws
            grad(ind,2,k1,k2,k3)=grad(ind,2,k1,k2,k3)+cdy*ws
            grad(ind,3,k1,k2,k3)=grad(ind,3,k1,k2,k3)+cdz*ws
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
C**************************************************************************
c
C     direction interactions by precomputed 1D tables
C     potential+gradient+hessian
C
C*********************************************************************C
      subroutine tens_prod_to_pghloc_1d(nd,n,ws,fvals,pot,grad,hess,
     1    ntab,tab_loc,tabx_loc,tabxx_loc,ind_loc,ixyz)
C*********************************************************************C
c     This routine computes the 1D volume Gauss transform over a 
c     single box source distribution given as function values on
c     a tensor product grid.
c
c     INPUT:
c     nd            vector length (for multiple RHS)
c     n             number of nodes along each dimension
c     fvals         function values at tensor grid
c     tab_loc       precomputed tables of 1D integrals
c     tabx_loc      precomputed tables for first derivatives
c     tabxx_loc     precomputed tables for second derivatives
c     ind_loc       precomputed nonzero pattern of tables
c     ixy           pointers to local table, specify which local table should
c                   be used
c     OUTPUT:
c     pot          potential values on tensor product grid
c     grad         gradient values on tensor product grid
c     hess         hessian values on tensor product grid
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      real *8 fvals(nd,n),pot(nd,n),grad(nd,n),hess(nd,n)
      real *8 tab_loc(n,n,-ntab:ntab)
      real *8 tabx_loc(n,n,-ntab:ntab)
      real *8 tabxx_loc(n,n,-ntab:ntab)
      integer ind_loc(2,n+1,-ntab:ntab)
      integer ixyz(1)
c
      ix = ixyz(1)
      nx = ind_loc(2,n+1,ix)-ind_loc(1,n+1,ix)+1

      if (nx.eq.0) return
      
      do ind = 1,nd
c        transform in x
         do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
            cd=0
            cdx=0
            cdxx=0
            do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
               cd=cd+tab_loc(j1,k1,ix)*fvals(ind,j1)
               cdx=cdx+tabx_loc(j1,k1,ix)*fvals(ind,j1)
               cdxx=cdxx+tabxx_loc(j1,k1,ix)*fvals(ind,j1)
            enddo
            pot(ind,k1)=pot(ind,k1)+cd*ws
            grad(ind,k1)=grad(ind,k1)+cdx*ws
            hess(ind,k1)=hess(ind,k1)+cdxx*ws
         enddo
      enddo
      
      return
      end subroutine
c
c
C
c
      subroutine tens_prod_to_pghloc_2d(nd,n,ws,fvals,pot,grad,hess,
     1    ntab,tab_loc,tabx_loc,tabxx_loc,ind_loc,ixy)
C*********************************************************************C
c     This routine computes the 2D volume Gauss transform over a 
c     single box source distribution given as function values on
c     a tensor product grid.
c
c     INPUT:
c     nd            vector length (for multiple RHS)
c     n             number of nodes along each dimension
c     fvals         function values at tensor grid
c     tab_loc       precomputed tables of 1D integrals
c     tabx_loc      precomputed tables for first derivatives
c     tabxx_loc     precomputed tables for second derivatives
c     ind_loc       precomputed nonzero pattern of tables
c     ixy           pointers to local table, specify which local table should
c                   be used
c     OUTPUT:
c     pot           potential values on tensor product grid
c     grad          gradient values on tensor product grid
c     hess          hessian values on tensor product grid
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      real *8 fvals(nd,n,n),pot(nd,n,n),grad(nd,2,n,n),hess(nd,3,n,n)
      real *8 tab_loc(n,n,-ntab:ntab)
      real *8 tabx_loc(n,n,-ntab:ntab)
      real *8 tabxx_loc(n,n,-ntab:ntab)
      integer ind_loc(2,n+1,-ntab:ntab)
      integer ixy(2)
      real *8 ff(n,n)
      real *8 ffx(n,n)
      real *8 ffxx(n,n)
c
      ix = ixy(1)
      iy = ixy(2)
      
      nx = ind_loc(2,n+1,ix)-ind_loc(1,n+1,ix)+1
      ny = ind_loc(2,n+1,iy)-ind_loc(1,n+1,iy)+1

      if (nx.eq.0 .or. ny.eq.0) return
      
      do ind = 1,nd
c        transform in x
         do j2=1,n
         do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
            cd=0
            cdx=0
            cdxx=0
            do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
               cd=cd+tab_loc(j1,k1,ix)*fvals(ind,j1,j2)
               cdx=cdx+tabx_loc(j1,k1,ix)*fvals(ind,j1,j2)
               cdxx=cdxx+tabxx_loc(j1,k1,ix)*fvals(ind,j1,j2)
            enddo
            ff(k1,j2)=cd
            ffx(k1,j2)=cdx
            ffxx(k1,j2)=cdxx
         enddo
         enddo
c        transfrom in y
         do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
         do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
            cd = 0.0d0
            cdx = 0.0d0
            cdy = 0.0d0
            cdxx = 0.0d0
            cdxy = 0.0d0
            cdyy = 0.0d0
            do j2=ind_loc(1,k2,iy),ind_loc(2,k2,iy)
               cd=cd+tab_loc(j2,k2,iy)*ff(k1,j2)
               cdy=cdy+tabx_loc(j2,k2,iy)*ff(k1,j2)
               cdyy=cdyy+tabxx_loc(j2,k2,iy)*ff(k1,j2)

               cdx=cdx+tab_loc(j2,k2,iy)*ffx(k1,j2)
               cdxy=cdxy+tabx_loc(j2,k2,iy)*ffx(k1,j2)

               cdxx=cdxx+tab_loc(j2,k2,iy)*ffxx(k1,j2)
            enddo
            pot(ind,k1,k2)=pot(ind,k1,k2)+cd*ws
            grad(ind,1,k1,k2)=grad(ind,1,k1,k2)+cdx*ws
            grad(ind,2,k1,k2)=grad(ind,2,k1,k2)+cdy*ws

            hess(ind,1,k1,k2)=hess(ind,1,k1,k2)+cdxx*ws
            hess(ind,2,k1,k2)=hess(ind,2,k1,k2)+cdxy*ws
            hess(ind,3,k1,k2)=hess(ind,3,k1,k2)+cdyy*ws
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
      subroutine tens_prod_to_pghloc_3d(nd,n,ws,fvals,pot,grad,hess,
     1    ntab,tab_loc,tabx_loc,tabxx_loc,ind_loc,ixyz)
c----------------------------------------------------------------------c
      implicit real *8 (a-h,o-z)
      real *8 fvals(nd,n,n,n)
      real *8 pot(nd,n,n,n)
      real *8 grad(nd,3,n,n,n)
      real *8 hess(nd,6,n,n,n)

      real *8 tab_loc(n,n,-ntab:ntab)
      real *8 tabx_loc(n,n,-ntab:ntab)
      real *8 tabxx_loc(n,n,-ntab:ntab)
      integer ind_loc(2,n+1,-ntab:ntab)
      integer ixyz(3)

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

      ix=ixyz(1)
      iy=ixyz(2)
      iz=ixyz(3)
      
      nx = ind_loc(2,n+1,ix)-ind_loc(1,n+1,ix)+1
      ny = ind_loc(2,n+1,iy)-ind_loc(1,n+1,iy)+1
      nz = ind_loc(2,n+1,iz)-ind_loc(1,n+1,iz)+1

      if (nx.eq.0 .or. ny.eq.0 .or. nz.eq.0) return
      
      do ind = 1,nd
c        transform in x
         do j3=1,n
         do j2=1,n
         do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
c         do k1=1,n
            cd=0
            cdx=0.0d0
            cdxx=0.0d0
            do j1=ind_loc(1,k1,ix),ind_loc(2,k1,ix)
c            do j1=1,n
               cd=cd+tab_loc(j1,k1,ix)*fvals(ind,j1,j2,j3)
               cdx=cdx+tabx_loc(j1,k1,ix)*fvals(ind,j1,j2,j3)
               cdxx=cdxx+tabxx_loc(j1,k1,ix)*fvals(ind,j1,j2,j3)
            enddo
            ff(k1,j2,j3)=cd
            ffx(k1,j2,j3)=cdx
            ffxx(k1,j2,j3)=cdxx
         enddo
         enddo
         enddo

c        transform in y
         do j3=1,n
         do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
         do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
c         do k2=1,n
c         do k1=1,n
            cd=0
            cdx = 0.0d0
            cdy = 0.0d0
            cdxx = 0.0d0
            cdxy = 0.0d0
            cdyy = 0.0d0
            do j2=ind_loc(1,k2,iy),ind_loc(2,k2,iy)
c            do j2=1,n
               cd   = cd   +   tab_loc(j2,k2,iy)*ff(k1,j2,j3)
               cdy  = cdy  +  tabx_loc(j2,k2,iy)*ff(k1,j2,j3)
               cdyy = cdyy + tabxx_loc(j2,k2,iy)*ff(k1,j2,j3)

               cdx  = cdx  +   tab_loc(j2,k2,iy)*ffx(k1,j2,j3)
               cdxy = cdxy +  tabx_loc(j2,k2,iy)*ffx(k1,j2,j3)
               
               cdxx = cdxx +   tab_loc(j2,k2,iy)*ffxx(k1,j2,j3)
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
         do k3=ind_loc(1,n+1,iz),ind_loc(2,n+1,iz)
         do k2=ind_loc(1,n+1,iy),ind_loc(2,n+1,iy)
         do k1=ind_loc(1,n+1,ix),ind_loc(2,n+1,ix)
c         do k3=1,n
c         do k2=1,n
c         do k1=1,n
            cd=0
            cdx = 0.0d0
            cdy = 0.0d0
            cdz = 0.0d0
            cdxx = 0.0d0
            cdyy = 0.0d0
            cdzz = 0.0d0
            cdxy = 0.0d0
            cdxz = 0.0d0
            cdyz = 0.0d0
            do j3=ind_loc(1,k3,iz),ind_loc(2,k3,iz)
c            do j3=1,n
               cd   = cd   +   tab_loc(j3,k3,iz)*ff2(k1,k2,j3)
               cdz  = cdz  +  tabx_loc(j3,k3,iz)*ff2(k1,k2,j3)
               cdzz = cdzz + tabxx_loc(j3,k3,iz)*ff2(k1,k2,j3)

               cdx  = cdx  +  tab_loc(j3,k3,iz)*ff2x(k1,k2,j3)
               cdxz = cdxz + tabx_loc(j3,k3,iz)*ff2x(k1,k2,j3)

               cdy  = cdy  +  tab_loc(j3,k3,iz)*ff2y(k1,k2,j3)
               cdyz = cdyz + tabx_loc(j3,k3,iz)*ff2y(k1,k2,j3)
               
               cdxx = cdxx + tab_loc(j3,k3,iz)*ff2xx(k1,k2,j3)
               cdxy = cdxy + tab_loc(j3,k3,iz)*ff2xy(k1,k2,j3)
               cdyy = cdyy + tab_loc(j3,k3,iz)*ff2yy(k1,k2,j3)
            enddo
            pot(ind,k1,k2,k3)=pot(ind,k1,k2,k3)+cd*ws
            grad(ind,1,k1,k2,k3)=grad(ind,1,k1,k2,k3)+cdx*ws
            grad(ind,2,k1,k2,k3)=grad(ind,2,k1,k2,k3)+cdy*ws
            grad(ind,3,k1,k2,k3)=grad(ind,3,k1,k2,k3)+cdz*ws

            hess(ind,1,k1,k2,k3)=hess(ind,1,k1,k2,k3)+cdxx*ws
            hess(ind,2,k1,k2,k3)=hess(ind,2,k1,k2,k3)+cdyy*ws
            hess(ind,3,k1,k2,k3)=hess(ind,3,k1,k2,k3)+cdzz*ws

            hess(ind,4,k1,k2,k3)=hess(ind,4,k1,k2,k3)+cdxy*ws
            hess(ind,5,k1,k2,k3)=hess(ind,5,k1,k2,k3)+cdxz*ws
            hess(ind,6,k1,k2,k3)=hess(ind,6,k1,k2,k3)+cdyz*ws
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
      

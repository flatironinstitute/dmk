c 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c     this is the end of the debugging code and the beginning
c     of the legendre expansion routines
c 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c 
c 
c     This file contains a set of subroutines for the handling
c     of a function defined on an adaptive tree. 
c     Following is a brief description of these
c     subroutines.
c
      subroutine treedata_trans_nd(ndim,nd,nlevels,itree,iptr,
     1    boxsize,norder,fin,fout,umat)
c
c     This code converts an input tree data given on a tensor product grid on each leaf box
c     to the output tree data on each leaf box. 
c     Depending on the 1d transformation matrices, it could be:  
c     (1) converting function values to the coefficients of orthogonal polynomial expansion
c     coefficients;
c     (2) if umax converts coefficients to function values, then it's the inverse map of (1).
c     (3) converting fcoefs to the coefficients of its derivatives.
c 
c     It's user's responsibility to ensure that umat are the correct 1D
c     transformation matrices.
c 
c     input:
c     ndim - dimension of the underlying space
c     nd - integer,   number of functions
c     nlevels - integer
c              number of levels
c     itree - integer(ltree)
c            array containing the tree structure
c     iptr - integer(8)
c            pointer to various parts of the tree structure
c           iptr(1) - laddr
c           iptr(2) - ilevel
c           iptr(3) - iparent
c           iptr(4) - nchild
c           iptr(5) - ichild
c           iptr(6) - ncoll
c           iptr(7) - coll
c           iptr(8) - ltree
c     norder - integer
c           order of expansions for input coefficients array
c     fin - double (nd,norder**2,nboxes)
c            input data on the tree
c     umat - 1D transformation matrices along each direction
c      
c     output:
c     fout - double precision (nd,norder**ndim,nboxes)
c            output data given on each leaf box
c
      implicit real *8 (a-h,o-z)
      integer nd
      integer nlevels
      integer itree(*),iptr(8)
      real *8 boxsize(0:nlevels)
      real *8 fin(nd,norder**ndim,*)
      real *8 fout(nd,norder**ndim,*)

      real *8 umat(norder,norder)
      real *8 umat_nd(norder,norder,ndim)

      norder2=norder*norder
      do i=1,ndim
         call dcopy_f77(norder2,umat,1,umat_nd(1,1,i),1)
      enddo

      do ilev = 0,nlevels
        sc = 2.0d0/boxsize(ilev)
C$OMP PARALLEL DO DEFAULT (SHARED)
C$OMP$PRIVATE(ibox,nchild,j,ind)
C$OMP$SCHEDULE(DYNAMIC)
        do ibox = itree(2*ilev+1),itree(2*ilev+2)
          nchild = itree(iptr(4) + ibox-1)
          if(nchild.eq.0) then
             call tens_prod_trans_vec(ndim,nd,norder,
     1           fin(1,1,ibox),norder,fout(1,1,ibox),umat_nd)
          endif
        enddo
C$OMP END PARALLEL DO
      enddo


      end
c
c
c
c
      subroutine treedata_eval_laplacian_nd(ndim,nd,ipoly,nlevels,
     1    itree,iptr,boxsize,norder,coefs,rlap)
c
c     This code evaluates the laplacian at the tensor grid on an adaptive tree
c     given the orthogonal polynomial expansion coefficients at each leaf box.
c 
c     input:
c     ndim - dimension of the underlying space
c     nd - integer,   number of functions
c     ipoly - 0: Legendre polynomials
c                 1: Chebyshev polynomials
c     nlevels - integer
c            number of levels
c     itree - integer(ltree)
c            array containing the tree structure
c     iptr - integer(8)
c            pointer to various parts of the tree structure
c           iptr(1) - laddr
c           iptr(2) - ilevel
c           iptr(3) - iparent
c           iptr(4) - nchild
c           iptr(5) - ichild
c           iptr(6) - ncoll
c           iptr(7) - coll
c           iptr(8) - ltree
c     norder - integer
c           order of expansions for input coefficients array
c     coefs - double (nd,norder**2,nboxes)
c           expansion coefficients on quad tree
c
c     output:
c     laplacian - double precision (nd,norder**ndim,nboxes)
c            gradient values on tensor grid on each leaf box
c
      implicit real *8 (a-h,o-z)
      integer nd
      integer nlevels
      integer itree(*),iptr(8)
      real *8 boxsize(0:nlevels)
      real *8 coefs(nd,norder**ndim,*)
      real *8 rlap(nd,norder**ndim,*)

      real *8 umat(norder,norder)
      real *8 vmat(norder,norder)
      real *8 vpmat(norder,norder)
      real *8 vppmat(norder,norder)

      call ortho_eval_tables(ipoly,norder,umat,vmat,vpmat,vppmat)
      
      do ilev = 0,nlevels
        sc = 2.0d0/boxsize(ilev)
C$OMP PARALLEL DO DEFAULT (SHARED)
C$OMP$PRIVATE(ibox,nchild)
C$OMP$SCHEDULE(DYNAMIC)
        do ibox = itree(2*ilev+1),itree(2*ilev+2)
          nchild = itree(iptr(4) + ibox-1)
          if(nchild.eq.0) then
             call ortho_eval_laplacian_nd(ndim,nd,norder,
     1           coefs(1,1,ibox),sc,rlap(1,1,ibox),vmat,vppmat)             
          endif
        enddo
C$OMP END PARALLEL DO
      enddo

      return
      end
c
c
c
c
      subroutine treedata_eval_pot_nd_asym(ndim,nd,delta,ipoly,nasym,
     1    nlevels,itree,iptr,boxsize,norder,fvals,pot)
c
c     This code evaluates the potential value at the tensor grid on an adaptive tree
c     using asympotitic expansions 
c 
c     input:
c     ndim - dimension of the underlying space
c     nd - integer,   number of functions
c     ipoly - 0: Legendre polynomials
c                 1: Chebyshev polynomials
c     nasym - order of asympotitic expansion
c     nlevels - integer
c            number of levels
c     itree - integer(ltree)
c            array containing the tree structure
c     iptr - integer(8)
c            pointer to various parts of the tree structure
c           iptr(1) - laddr
c           iptr(2) - ilevel
c           iptr(3) - iparent
c           iptr(4) - nchild
c           iptr(5) - ichild
c           iptr(6) - ncoll
c           iptr(7) - coll
c           iptr(8) - ltree
c     norder - integer
c           order of expansions for input coefficients array
c     fvals - double (nd,norder**ndim,nboxes)
c           function values on the tree
c
c     output:
c     pot - double precision (nd,norder**ndim,nboxes)
c           potential values on the tree
c
      implicit real *8 (a-h,o-z)
      integer nd
      integer nlevels
      integer itree(*),iptr(8)
      real *8 boxsize(0:nlevels)
      real *8 fvals(nd,norder**ndim,*)
      real *8 pot(nd,norder**ndim,*)

      real *8 umat(norder,norder)
      real *8 vmat(norder,norder)
      real *8 vpmat(norder,norder)
      real *8 vppmat(norder,norder)
      real *8 umat_nd(norder,norder,ndim)

      real *8, allocatable :: fcoefs(:,:)
      real *8, allocatable :: flvals(:,:)
      real *8, allocatable :: flcoefs(:,:)
      real *8, allocatable :: fl2vals(:,:)

      call ortho_eval_tables(ipoly,norder,umat,vmat,vpmat,vppmat)
      norder2=norder*norder
      do i=1,ndim
         call dcopy_f77(norder2,umat,1,umat_nd(1,1,i),1)
      enddo
      
      npbox=norder**ndim
      allocate(fcoefs(nd,npbox))
      allocate(flvals(nd,npbox))
      allocate(flcoefs(nd,npbox))
      allocate(fl2vals(nd,npbox))
      
      sqrtpi = sqrt(4*atan(1.0d0))
      d0 = sqrtpi*sqrt(delta)
      d2 = delta/4
      d4 = delta*delta/32

      c0 = d0**ndim
      c2 = c0*d2
      c4 = c0*d4
      do ilev = 0,nlevels
        sc=2.0d0/boxsize(ilev)
        do ibox = itree(2*ilev+1),itree(2*ilev+2)
          nchild = itree(iptr(4) + ibox-1)
          if(nchild.eq.0) then
             if (nasym.eq.2) then
                call tens_prod_trans_vec(ndim,nd,norder,
     1              fvals(1,1,ibox),norder,fcoefs,umat_nd)
                call ortho_eval_laplacian_nd(ndim,nd,norder,fcoefs,
     1              sc,flvals,vmat,vppmat)
                do j=1,npbox
                do ind=1,nd
                   pot(ind,j,ibox)=c0*fvals(ind,j,ibox)
     1                 +c2*flvals(ind,j)
                enddo
                enddo
             elseif (nasym.eq.3) then
                call tens_prod_trans_vec(ndim,nd,norder,
     1              fvals(1,1,ibox),norder,fcoefs,umat_nd)
                call ortho_eval_laplacian_nd(ndim,nd,norder,fcoefs,
     1              sc,flvals,vmat,vppmat)
                call tens_prod_trans_vec(ndim,nd,norder,flvals,
     1              norder,flcoefs,umat_nd)
                call ortho_eval_laplacian_nd(ndim,nd,norder,flcoefs,
     1              sc,fl2vals,vmat,vppmat)
                do j=1,npbox
                do ind=1,nd
                   pot(ind,j,ibox)=c0*fvals(ind,j,ibox)
     1                 +c2*flvals(ind,j)+c4*fl2vals(ind,j)
                enddo
                enddo
             endif
          endif
        enddo
      enddo
      return
      end
c
c
c
c
      subroutine treedata_eval_pot_nd_asym_fast(ndim,nd,ndelta,deltas,
     1    dwhts,idelta,ipoly,nasym,nlevels,itree,iptr,boxsize,norder,
     2    fvals,flvals,fl2vals,pot)
cccc     2    fvals,flvals,fl2vals,fl3vals,pot)
c
c     This code evaluates the potential value at the tensor grid on an adaptive tree
c     using asympotitic expansions with precompted Laplacian and biLaplacian
c 
c     input:
c     ndim - dimension of the underlying space
c     nd - integer,   number of functions
c     ndelta - integer, total number of deltas
c     deltas - real *8(ndelta), values of delta
c     dwhts - weights of each delta
c     idelta - integer (0:nlevels), number of deltas that can be handled via asymptotic expansion
c     ipoly - 0: Legendre polynomials
c                 1: Chebyshev polynomials
c     nasym - order of asympotitic expansion
c     nlevels - integer
c            number of levels
c     itree - integer(ltree)
c            array containing the tree structure
c     iptr - integer(8)
c            pointer to various parts of the tree structure
c           iptr(1) - laddr
c           iptr(2) - ilevel
c           iptr(3) - iparent
c           iptr(4) - nchild
c           iptr(5) - ichild
c           iptr(6) - ncoll
c           iptr(7) - coll
c           iptr(8) - ltree
c     norder - integer
c           order of expansions for input coefficients array
c     fvals - double (nd,norder**ndim,nboxes)
c           function values on the tree
c     flvals - double (nd,norder**ndim,nboxes)
c           Laplacian of the function
c     fl2vals - double (nd,norder**ndim,nboxes)
c           BiLaplacian of the function
c
c     output:
c     pot - double precision (nd,norder**ndim,nboxes)
c           potential values on the tree
c
      implicit real *8 (a-h,o-z)
      integer nd
      integer nlevels
      integer itree(*),iptr(8),idelta(0:nlevels)
      real *8 deltas(ndelta),dwhts(ndelta)
      real *8 boxsize(0:nlevels)
      real *8 fvals(nd,norder**ndim,*)
      real *8 flvals(nd,norder**ndim,*)
      real *8 fl2vals(nd,norder**ndim,*)
cccc      real *8 fl3vals(nd,norder**ndim,*)
      real *8 pot(nd,norder**ndim,*)

      real *8 c0(0:nlevels),c2(0:nlevels),c4(0:nlevels)
c      real *8 c6(0:nlevels)
      
      npbox=norder**ndim
      
      sqrtpi = sqrt(4*atan(1.0d0))

      do ilev=0,nlevels
         c0(ilev) = 0
         c2(ilev) = 0
         c4(ilev) = 0
c         c6(ilev) = 0
      enddo

      do ilev=0,nlevels
         do i=1,idelta(ilev)
            delta=deltas(ndelta-i+1)

            d0 = sqrtpi*sqrt(delta)
            d0 = d0**ndim
         
            d2 = d0*delta/4
            d4 = d0*delta**2/32
c            d6 = d0*delta**3/384
            
            c0(ilev) = c0(ilev) + d0*dwhts(ndelta-i+1)
            c2(ilev) = c2(ilev) + d2*dwhts(ndelta-i+1)
            c4(ilev) = c4(ilev) + d4*dwhts(ndelta-i+1)
c            c6(ilev) = c6(ilev) + d6*dwhts(ndelta-i+1)
         enddo
      enddo

      itype=0
      do ilev = 0,nlevels
        if (idelta(ilev).gt.0) then
           do ibox = itree(2*ilev+1),itree(2*ilev+2)
              nchild = itree(iptr(4) + ibox-1)
              if(nchild.eq.0) then
                 if (nasym.eq.2) then
                    do j=1,npbox
                    do ind=1,nd
                       pot(ind,j,ibox)=pot(ind,j,ibox)
     1                     +c0(ilev)*fvals(ind,j,ibox)
     2                     +c2(ilev)*flvals(ind,j,ibox)
                    enddo
                    enddo
                 elseif (nasym.eq.3) then
                    do j=1,npbox
                    do ind=1,nd
                       pot(ind,j,ibox)=pot(ind,j,ibox)
     1                     +c0(ilev)*fvals(ind,j,ibox)
     2                     +c2(ilev)*flvals(ind,j,ibox)
     3                     +c4(ilev)*fl2vals(ind,j,ibox)
                    enddo
                    enddo
c                 elseif (nasym.eq.4) then
c                    do j=1,npbox
c                       do ind=1,nd
c                          pot(ind,j,ibox)=pot(ind,j,ibox)
c     1                        +c0(ilev)*fvals(ind,j,ibox)
c     2                        +c2(ilev)*flvals(ind,j,ibox)
c     3                        +c4(ilev)*fl2vals(ind,j,ibox)
c     4                        +c6(ilev)*fl3vals(ind,j,ibox)
c                       enddo
c                    enddo
                 endif
              endif
           enddo
        endif
      enddo
      
      return
      end
c
c
c
c
      subroutine treedata_evalt_nd(ndim,nd,ipoly,norder,nboxes,nlevels,
     1    ltree,itree,iptr,tcenters,boxsize,fcoefs,nt,targ,fvals)
c
c     This code evaluates function values at nt target points where the function is given 
c     as the orthogonal polynomial expansion coefficients on each leaf box
c 
c     input:
c     nd - integer,   number of functions
c     ipoly - 0: Legendre polynomials
c                 1: Chebyshev polynomials
c     norder - integer
c           order of expansions for input coefficients array
c     nboxes - integer
c            number of boxes
c     nlevels - integer
c            number of levels
c     ltree - integer
c            length of array containing the tree structure
c     itree - integer(ltree)
c            array containing the tree structure
c     iptr - integer(8)
c            pointer to various parts of the tree structure
c           iptr(1) - laddr
c           iptr(2) - ilevel
c           iptr(3) - iparent
c           iptr(4) - nchild
c           iptr(5) - ichild
c           iptr(6) - ncoll
c           iptr(7) - coll
c           iptr(8) - ltree
c     tcenters - double (ndim,nboxes) coordinates of box centers
c     boxsize - double(nboxes) box size of each box
c      
c      
c     fcoefs - double (nd,norder*norder,nboxes)
c                  expansion coefficients on each leaf box
c     nt - number of target points
c     targ - (ndim,nt) coordinates of targets
c      
c      
c     output:
c     fvals - double precision (nd,nt) function values
c            
      implicit real *8 (a-h,o-z)
      integer nd
      integer nboxes,nlevels
      integer iptr(8),ltree
      integer itree(ltree)
      real *8 tcenters(ndim,nboxes),boxsize(0:nlevels)
      
      real *8 fcoefs(nd,norder**ndim,nboxes)
      real *8 targ(ndim,nt),fvals(nd,nt)
      real *8 xyz(ndim),cen(ndim)
      
      integer, allocatable :: itarg(:),itargse(:,:)
      real *8, allocatable :: targsort(:,:)
      real *8, allocatable :: potsort(:,:)

      allocate(itarg(nt),itargse(2,nboxes))
      call pts_tree_sort(ndim,nt,targ,itree,ltree,nboxes,nlevels,iptr,
     1    tcenters,itarg,itargse)
      allocate(targsort(ndim,nt),potsort(nd,nt))
      do i=1,nt
         do ind=1,nd
            potsort(ind,i)=0
         enddo
      enddo
c      
c     reorder targets
c
      call dreorderf(ndim,nt,targ,targsort,itarg)
c
c     
c     
      do ilev = 0,nlevels
         sc = 2.0d0/boxsize(ilev)
C$OMP PARALLEL DO DEFAULT (SHARED)
C$OMP$PRIVATE(ibox,nchild,istart,iend,npts,j,i,cen,xyz)
C$OMP$SCHEDULE(DYNAMIC)
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
cccc            call prinf('ibox=*',ibox,1)
            nchild = itree(iptr(4)+ibox-1)
            if (nchild.eq.0) then
               istart = itargse(1,ibox) 
               iend = itargse(2,ibox) 
               npts = iend-istart+1
cccc               print *, ibox, istart,iend,npts
               if (npts.gt.0) then
                  do j=1,ndim
                     cen(j)=tcenters(j,ibox)
                  enddo
                  do i=istart,iend
                     do j=1,ndim
                        xyz(j) = (targsort(j,i)-cen(j))*sc
                     enddo
                     call ortho_evalt_nd(ndim,nd,ipoly,norder,
     1                   fcoefs(1,1,ibox),xyz,potsort(1,i))
                  enddo
               endif
            endif
        enddo
C$OMP END PARALLEL DO
      enddo
c
c     resort the output arrays in input order
c
      call dreorderi(nd,nt,potsort,fvals,itarg)

      return
      end
c
c
c
c
      subroutine treedata_derror(nd,nlevels,itree,iptr,
     1    npbox,fex,fcomp,abserr,rnorm,nleaf)
c
c     computes the absolute l2 error of two tree data fcomp
c     fex, the l2 norm of fex, and the total number of leaf boxes
c 
c     input:
c
c     nd - integer,   number of functions
c     nlevels - integer
c            number of levels
c     itree - integer(ltree)
c            array containing the tree structure
c     iptr - integer(8)
c            pointer to various parts of the tree structure
c           iptr(1) - laddr
c           iptr(2) - ilevel
c           iptr(3) - iparent
c           iptr(4) - nchild
c           iptr(5) - ichild
c           iptr(6) - ncoll
c           iptr(7) - coll
c           iptr(8) - ltree
c     npbox - integer
c           number of points in each leaf box
c     fex - double (nd,npbox,nboxes)
c           values of the reference tree data on each leaf box
c     fcomp - double (nd,npbox,nboxes)
c           values of the numerical tree data on each leaf box
c     output:
c     abserr - absolute l2 error
c     rnorm - l2 norm of fex
c     nleaf  - total number of leaf boxes
c      
      implicit real *8 (a-h,o-z)
      integer nd
      integer nlevels
      integer itree(*),iptr(*)
      real *8 fex(nd,npbox,*)
      real *8 fcomp(nd,npbox,*)

      abserr=0
      rnorm=0
      nleaf=0

      do ilev = 0,nlevels
C$OMP PARALLEL DO DEFAULT (SHARED) 
C$OMP$PRIVATE(ibox,nchild,j,ind) reduction( + : rnorm,abserr,nleaf)
C$OMP$SCHEDULE(DYNAMIC)
        do ibox = itree(2*ilev+1),itree(2*ilev+2)
          nchild = itree(iptr(4) + ibox-1)
          if(nchild.eq.0) then
             nleaf=nleaf+1
             do j=1,npbox
                do ind=1,nd
                   rnorm=rnorm+fex(ind,j,ibox)**2
c                   if (rnorm.lt.abs(fex(ind,j,ibox))) then
c                      rnorm=abs(fex(ind,j,ibox))
c                   endif
                   abserr=abserr+(fex(ind,j,ibox)-fcomp(ind,j,ibox))**2
c                   if (abserr.lt.
c     1                 abs(fex(ind,j,ibox)-fcomp(ind,j,ibox))) then
c                      abserr = abs(fex(ind,j,ibox)-fcomp(ind,j,ibox))
c                   endif 
                enddo
             enddo
          endif
        enddo
C$OMP END PARALLEL DO
      enddo

      abserr=sqrt(abserr)
      rnorm=sqrt(rnorm)

      return
      end
c
c
c
c
      
      subroutine treedata_lpnorm(ndim,iptype,ipoly,nd,nlevels,itree,
     1    iptr,boxsize,norder,npbox,fvals,rnorm,nleaf)
c
c     computes the lp norm of the tree date fvals
c 
c     input:
c
c     nd - integer,   number of functions
c     ndim - integer,   dimension of the space
c     iptype - integer, norm type 0: maximum norm, 1: 1 norm, 2: 2 norm
c     nlevels - integer
c            number of levels
c     itree - integer(ltree)
c            array containing the tree structure
c     iptr - integer(8)
c            pointer to various parts of the tree structure
c           iptr(1) - laddr
c           iptr(2) - ilevel
c           iptr(3) - iparent
c           iptr(4) - nchild
c           iptr(5) - ichild
c           iptr(6) - ncoll
c           iptr(7) - coll
c           iptr(8) - ltree
c     npbox - integer
c           number of points in each leaf box
c     fvals - double (nd,npbox,nboxes)
c           values of the tree data on each leaf box
c
c     output:
c     rnorm - lp norm of fex
c     nleaf  - total number of leaf boxes
c      
      implicit real *8 (a-h,o-z)
      integer nd
      integer nlevels
      integer itree(*),iptr(8)
      real *8 boxsize(0:nlevels)
      real *8 fvals(nd,npbox,*)

      real *8, allocatable :: wts(:),xs(:,:)

      allocate(xs(ndim,npbox),wts(npbox))
      
      itype = 1
      call polytens_exps_nd(ndim,ipoly,itype,norder,'f',xs,
     1    utmp,1,vtmp,1,wts)
      
      rnorm=0
      nleaf=0

      if (iptype.eq.0) then
         do ilev = 0,nlevels
C$OMP PARALLEL DO DEFAULT (SHARED)
C$OMP$PRIVATE(ibox,nchild) reduction(+:nleaf)
C$OMP$reduction(max:rnorm)
C$OMP$SCHEDULE(DYNAMIC)
            do ibox = itree(2*ilev+1),itree(2*ilev+2)
               nchild = itree(iptr(4) + ibox-1)
               if(nchild.eq.0) then
                  nleaf=nleaf+1
                  rtmp=maxval(fvals(1:nd,1:npbox,ibox))
                  if (rtmp.gt.rnorm) rnorm=rtmp
               endif
            enddo
C$OMP END PARALLEL DO
         enddo

      else
         do ilev = 0,nlevels
            sc = (boxsize(ilev)/2)**ndim
C$OMP PARALLEL DO DEFAULT (SHARED)
C$OMP$PRIVATE(ibox,nchild,j,ind) reduction(+:rnorm,nleaf)
C$OMP$SCHEDULE(DYNAMIC)
            do ibox = itree(2*ilev+1),itree(2*ilev+2)
               nchild = itree(iptr(4) + ibox-1)
               if(nchild.eq.0) then
                  nleaf=nleaf+1
                  do i=1,npbox
                     do ind=1,nd
                        rnorm=rnorm
     1                      +abs(fvals(ind,i,ibox))**iptype*wts(i)*sc
                     enddo
                  enddo
               endif
            enddo
C$OMP END PARALLEL DO
         enddo
         rnorm=rnorm**(1.0d0/iptype)
      endif
      
      return
      end
c
c
c
c

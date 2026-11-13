ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     $Date$
c     $Revision$
c
c     created on 03/18/2024 by Shidong Jiang
c     last modified on 03/18/2024 by Shidong Jiang
c
c
c**********************************************************************
c      
c     We take the following conventions for the Stokes kernels
c
c     For a source y and target x, let r_i = x_i-y_i
c     and let r = sqrt(sum_i r_i^2)
c
c     The Stokeslet G_{ij} (without the 1/2pi scaling) in 2D is
c
c     G_{ij}(x,y) = (r_i r_j)/(2r^2) - delta_{ij}log(r)/2
c
c     The Stokeslet G_{ij} (without the 1/4pi scaling) in 3D is
c
c     G_{ij}(x,y) = (r_i r_j)/(2r^3) + delta_{ij}/(2r)
c
c
c     Stokes DMK in R^{2,3}: evaluate all pairwise particle
c     interactions (ignoring self-interactions) and
c     interactions with targs.
c      
c     This routine computes sums of the form
c
c       u(x) = sum_m G_{ij}(x,y^{(m)}) sigma^{(m)}_j
c
c     where sigma^{(m)} is the Stokeslet charge, 
c     (note that each of these is a ndim vector per source point y^{(m)}).
c     For x a source point, the self-interaction in the sum is omitted. 
c
c
c     modified from stokesdmk.f. Difference: here we don't do refine once
c     to try to acclerate local interactions. Instead, it relies on SIMD
c     mask operation to take advantage of the compactness of the residual
c     kernel. However, SIMD mask operation is only available for AVX512
c     machine for now.
      
      subroutine stokesdmk(nd,dim,eps,
     1    iperiod,ns,sources,
     2    ifstoklet, stoklet, ifstrslet, strslet, strsvec,
     3    ifppreg, pot, pre, grad, nt, targ, 
     4    ifppregtarg, pottarg, pretarg, gradtarg, tottimeinfo)
c----------------------------------------------
c   INPUT PARAMETERS:
c   nd            : number of DMKs (same source and target locations, 
c                   different charge, dipole strengths)
c   dim           : dimension of the space
c   eps           : precision requested
c   iperiod       : 0: free space; 1: periodic - not implemented yet      
c      
c   ns            : number of sources
c   sources(dim,ns) : source locations
c   source  in: double precision (dim,nsource)
c               source(k,j) is the kth component of the jth
c               source locations
c
c   ifstoklet  in: integer  
c               Stokeslet charge computation flag
c               ifstoklet = 1   =>  include Stokeslet contribution
c                                   otherwise do not
c 
c   stoklet in: double precision (nd,dim,nsource) 
c               Stokeslet charge strengths (sigma vectors above)
c
c   ifstrslet in: integer
c               stresslet computation flag
c               ifstrslet = 1   =>  include standard stresslet
c                                   (type I)
c
c   strslet  in: double precision (nd,dim,nsource) 
c               stresslet strengths (mu vectors above)
c
c   strsvec  in: double precision (nd,dim,nsource)   
c               stresslet orientations (nu vectors above)
c
c   ifppreg  in: integer      
c               flag for evaluating potential, gradient, and pressure
c               at the sources
c               ifppreg = 1, only potential
c               ifppreg = 2, potential and pressure
c         GRADIENT NOT IMPLEMENTED
c               ifppreg = 3, potential, pressure, and gradient 
c      
c   nt      in: integer  
c              number of targs 
c
c   targ    in: double precision (dim,nt)
c             targ(k,j) is the kth component of the jth
c             targ location
c      
c   ifppregtarg in: integer
c                flag for evaluating potential, gradient, and pressure
c                at the targets
c                ifppregtarg = 1, only potential
c                ifppregtarg = 2, potential and pressure
c                ifppregtarg = 3, potential, pressure, and gradient
c
c-----------------------------------------------------------------------
c
c   OUTPUT parameters:
c
c   pot   out: double precision(nd,dim,nsource) 
c           velocity at the source locations
c      
c   pre   out: double precision(nd,nsource)
c           pressure at the source locations
c      
c         GRADIENT NOT IMPLEMENTED
c   grad   out: double precision(nd,dim,dim,nsource) 
c              gradient of velocity at the source locations
c              grad(l,i,j,k) is the ith component of the
c              gradient of the jth component of the velocity
c              for the lth density at the kth source location
c     
c   pottarg   out: double precision(nd,dim,ntarg) 
c               velocity at the targets
c      
c   pretarg   out: double precision(nd,ntarg)
c               pressure at the targets
c      
c   gradtarg   out: double precision(nd,dim,dim,ntarg) 
c               gradient of velocity at the targets
c               gradtarg(l,i,j,k) is the ith component of the
c               gradient of the jth component of the velocity
c               for the lth density at the kth target
c   tottimeinfo out: double precision array
c               breakup of timing results of various steps in the algorithm
c
c     ONLY potential evaluation for stokeslet is implemented!
c      
      implicit none
c
cc      calling sequence variables
c 
      integer nd,dim,ifstoklet, ifstrslet
      real *8 eps,rpars(10)
      integer ns,nt,iperiod

      integer ifppreg, ifppregtarg
      real *8 sources(dim,ns),targ(dim,nt)
      real *8 stoklet(nd,dim,*), strslet(nd,dim,*)
      real *8 strsvec(nd,dim,*)

      real *8 pot(nd,dim,*),grad(nd,dim,dim,*),pre(nd,*)
      real *8 pottarg(nd,dim,*),gradtarg(nd,dim,dim,*),pretarg(nd,*)

c
cc      Tree variables
c
      integer, allocatable :: itree(:)
      integer iptr(8)
      integer nlmin,nlmax,ifunif
      real *8, allocatable :: centers(:,:),boxsize(:)
      integer idivflag,nlevels,nboxes,ndiv,ndiv0
      integer ltree

c
cc     sorted arrays
c
      integer, allocatable :: isrc(:),isrcse(:,:)
      integer, allocatable :: itarg(:),itargse(:,:)

      integer, allocatable :: nboxsrcpts(:),nboxtargpts(:)
      integer, allocatable :: ifleafbox(:)
      integer, allocatable :: ifpwexpform(:)
      integer, allocatable :: ifpwexpeval(:)
      integer, allocatable :: iftensprodeval(:)
      integer, allocatable :: iftensprodform(:)

      real *8, allocatable :: sourcesort(:,:),csourcesort(:,:)
      real *8, allocatable :: targsort(:,:),ctargsort(:,:)
      real *8, allocatable :: stokletsort(:,:,:)
      real *8, allocatable :: strsletsort(:,:,:)
      real *8, allocatable :: strsvecsort(:,:,:)
      real *8, allocatable :: potsort(:,:,:)
      real *8, allocatable :: pottargsort(:,:,:)
      real *8, allocatable :: presort(:,:)
      real *8, allocatable :: pretargsort(:,:)
      real *8, allocatable :: gradsort(:,:,:,:)
      real *8, allocatable :: gradtargsort(:,:,:,:)

      integer norder,npbox,npwmax
      real *8, allocatable :: coefsp(:),rmlexp(:)
      integer *8, allocatable :: lpaddr(:,:)
      integer, allocatable :: isgn(:,:)
      real *8, allocatable :: p2ctransmat(:,:,:,:)
      real *8, allocatable :: c2ptransmat(:,:,:,:)
c     Fourier transforms of the windowed and difference kernels
      real *8, allocatable :: dkernelft(:,:),hpw(:),ws(:),rl(:)

c     residual kernel Chebysheve expansion cofficients
c     for points in list1 and list2
c     list1 - only the smooth part of the residual kernel
c     list2 - the whole residual kernel for points away from the origin
      integer n1_diag, n2_diag, n1_offd, n2_offd
      real *8, allocatable :: coefs1_diag(:),coefs2_diag(:)
      real *8, allocatable :: coefs1_offd(:),coefs2_offd(:)
      integer *8 lcoefsptot,lmptotmax
      
c
cc      temporary variables
c
      integer npwlevel,nleafbox,ntot,ns2tp,nlevstart,ndigits
      integer, allocatable :: npw(:),nfourier(:)
      
      integer nfouriermax,ncoefsmax,ndtot
      
      integer i,ilev,j,jlev,lmptmp,id,k,nhess
      integer ifprint,ier,ikernel
      integer ibox,istart,iend,jstart,jend,ifplot,ind,npts,jbox,nchild
      integer mc,dad,ipoly
      integer lenw,keep,ltot,iw,nterms
      
      real *8 beta,bsize,c0,c1,c2,c4,scale,rl0,g0d2
      real *8 pswfeps,bsizesmall,bsizebig
      
      real *8 omp_get_wtime,pps,sc,pi,d
      real *8 time1,time2,ttotal,dt,dttree,dtsort
      real *8 timeinfo(20),tottimeinfo(20)
      real *8 wprolate(5000),rlam20,rkhi,psi0,derpsi0,zero
      real *8 dlogtk0,fval,st2dwk0

      ifprint=1
      ndigits=nint(log10(1.0d0/eps)-0.1)

      call prolc180(eps/(2*(ndigits+1))**2,beta)
      if(ifprint.ge.1) then
         call prin2('prolate parameter value=*',beta,1)
      endif

      lenw=10 000
      call prol0ini(ier,beta,wprolate,rlam20,rkhi,lenw,keep,ltot)
      nterms=wprolate(5)
      if(ifprint.ge.1) then
         call prinf('after prol0ini, ier=*',ier,1)
      endif
      iw=wprolate(1)
      
      pi = 4.0d0*atan(1.0d0)
      mc = 2**dim
      
      do i=1,10
         tottimeinfo(i)=0
      enddo
C
c     set criterion for box subdivision
c
      call lndiv_fast(dim,eps,ns,nt,ifstoklet,ifstrslet,ifppreg,
     1    ifppregtarg,ndiv,idivflag)

c
cc      set tree flags
c 
      nlmax = 51
      nlevels = 0
      nboxes = 0
      ltree = 0
      nlmin = 0
      ifunif = 0

      call cpu_time(time1)
C$    time1=omp_get_wtime()
c     find the memory requirements for the tree
      call pts_tree_mem(dim,sources,ns,targ,nt,idivflag,
     1    ndiv,nlmin,nlmax,ifunif,iperiod,
     2    nlevels,nboxes,ltree)
c 
      if (ifprint.eq.1) call prinf('nlevels=*',nlevels,1)
c     memory allocation for the tree
      allocate(itree(ltree))
      allocate(boxsize(0:nlevels))
      allocate(centers(dim,nboxes))
c
c     build the actual tree
c
      call pts_tree_build(dim,sources,ns,targ,nt,
     1    idivflag,ndiv,nlmin,nlmax,ifunif,iperiod,nlevels,nboxes,
     2    ltree,itree,iptr,centers,boxsize)
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      dttree = time2-time1
      tottimeinfo(1)=dttree
      
      if( ifprint .eq. 1 ) then
         call prin2('time in tree build=*',dttree,1)
         call prin2('boxsize(0)=*',boxsize(0),1)
         pps=(ns+nt+0.0d0)/dttree
         call prin2('points per sec=*',pps,1)
      endif

      allocate(isrc(ns),isrcse(2,nboxes))
      allocate(itarg(nt),itargse(2,nboxes))

      call cpu_time(time1)
C$    time1=omp_get_wtime()
c     sort source points to the tree
      call pts_tree_sort(dim,ns,sources,itree,ltree,nboxes,nlevels,
     1    iptr,centers,isrc,isrcse)
cccc      call prinf('isrcse=*',isrcse,20)

c     sort target points to the tree
      call pts_tree_sort(dim,nt,targ,itree,ltree,nboxes,nlevels,iptr,
     1   centers,itarg,itargse)
cccc      call prinf('itargse=*',itargse,nboxes)
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      dtsort = time2-time1
      tottimeinfo(2)=dtsort
      
      if( ifprint .eq. 1 ) then
         call prin2('time in pts_tree_sort=*',dtsort,1)
         pps=(ns+nt+0.0d0)/dtsort
         call prin2('points per sec=*',pps,1)
      endif

c     allocate memory for sorted quantities
      allocate(sourcesort(dim,ns))
      allocate(targsort(dim,nt))

      if(ifstoklet.eq.1.and.ifstrslet.eq.0) then
        allocate(stokletsort(nd,dim,ns))
        allocate(strsletsort(nd,dim,1))
        allocate(strsvecsort(nd,dim,1))
      endif

      if(ifppreg.eq.1) then
        allocate(potsort(nd,dim,ns),gradsort(nd,dim,dim,1),
     1     presort(nd,1))
      endif
c      
      if(ifppregtarg.eq.1) then
        allocate(pottargsort(nd,dim,nt),gradtargsort(nd,dim,dim,1),
     1     pretargsort(nd,1))
      endif
c
c     initialize potentials
c
      if(ifppreg.ge.1) then
         do i=1,ns
            do k=1,dim
               do id=1,nd
                  potsort(id,k,i) = 0
               enddo
            enddo
         enddo
      endif

      if(ifppregtarg.ge.1) then
         do i=1,nt
            do k=1,dim
               do id=1,nd
                  pottargsort(id,k,i) = 0
               enddo
            enddo
         enddo
      endif

c
c
c     reorder sources
c
      ndtot = nd*dim
      call dreorderf(dim,ns,sources,sourcesort,isrc)
      if(ifstoklet.eq.1) 
     1    call dreorderf(ndtot,ns,stoklet,stokletsort,isrc)
c
cc     reorder targets
c
      call dreorderf(dim,nt,targ,targsort,itarg)


      
c
c     DMK main steps
c
      ttotal=0

      call cpu_time(time1)
C$    time1=omp_get_wtime()

c     nboxsrcpts contains the number of source points in each box
      allocate(nboxsrcpts(nboxes))
      do ilev=0,nlevels
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
            istart = isrcse(1,ibox)
            iend = isrcse(2,ibox)
            nboxsrcpts(ibox) = iend-istart+1
         enddo
      enddo
      
c     nboxtargpts contains the number of source points in each box
      allocate(nboxtargpts(nboxes))
      do ilev=0,nlevels
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
            istart = itargse(1,ibox)
            iend = itargse(2,ibox)
            nboxtargpts(ibox) = iend-istart+1
         enddo
      enddo

c     ifleafbox: a box is regarded as a leaf box if it contains <= ndiv
c     source points, all its children due to level restriction are not
c     leaf boxes
      allocate(ifleafbox(nboxes))
      do i=1,nboxes
         ifleafbox(i)=0
      enddo

      if (nboxsrcpts(1).le.ndiv) ifleafbox(1)=1
      
      do ilev=1,nlevels
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
            if (itree(iptr(4)+ibox-1).eq.0) then
               ifleafbox(ibox)=1
            endif
         enddo
      enddo

      nleafbox=0
      do ibox=1,nboxes
         if (ifleafbox(ibox).eq.1) nleafbox=nleafbox+1
      enddo
      if (ifprint.eq.1) call prinf('nleafbox=*',nleafbox,1)
      
c     transpose sources and targets to be used in the fast kernel evaluation
c     written in C++
      allocate(ctargsort(nt,dim))
      allocate(csourcesort(ns,dim))

      do i=1,dim
         do j=1,nt
            ctargsort(j,i)=targsort(i,j)
         enddo
      enddo

      do i=1,dim
         do j=1,ns
            csourcesort(j,i)=sourcesort(i,j)
         enddo
      enddo

c     check whether we need to form and/or evaluate planewave expansions 
c     for boxes
      allocate(ifpwexpform(nboxes))
      allocate(ifpwexpeval(nboxes))
      allocate(iftensprodeval(nboxes))
      allocate(iftensprodform(nboxes))
      call pdmk_find_all_pwexp_boxes2(dim,nboxes,
     1    nlevels,ltree,itree,iptr,
     2    ndiv,nboxsrcpts,nboxtargpts,ifleafbox,
     3    ifpwexpform,ifpwexpeval,iftensprodeval)
c
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      ttotal = ttotal+time2-time1
      tottimeinfo(3)=tottimeinfo(3)+time2-time1
      
      call prin2('time on precomputation steps=*',time2-time1,1)
      
      call cpu_time(time1)
C$    time1=omp_get_wtime()

c     calculate and allocate maximum memory for planewave expansions
c     needed for one level
      if (ndigits.le.3) then
cccc         Laplace kernel
cccc         npwmax=13
cccc         norder=9
         npwmax=27
         norder=16
      elseif (ndigits.le.6) then
cccc         npwmax=25
cccc         norder=18
         npwmax=39
         norder=26
      elseif (ndigits.le.9) then
cccc         npwmax=39
cccc         norder=28
         npwmax=53
         norder=36
      elseif (ndigits.le.12) then
cccc         npwmax=53
cccc         norder=38
         npwmax=69
         norder=46
      endif
      
      call pdmk_mpmaxalloc(ndtot,dim,npwmax,itree,
     1    nlevels,ifpwexpform,ifpwexpeval,lmptotmax)
      if(ifprint .eq. 1)
     1  call prinf_long('memory for planewave expansions=*',lmptotmax,1)
      
      allocate(rmlexp(lmptotmax),stat=ier)
      if(ier.ne.0) then
         print *, "Cannot allocate workspace for plane wave expansions"
         print *, "lmptot=", lmptotmax
         ier = 4
         return
      endif

c     calculate and allocate memory for tensor product grid for all levels
      npbox = norder**dim
      
      allocate(lpaddr(2,nboxes))
      call pdmk_coefspalloc_stokes(ndtot,dim,itree,lpaddr,
     1    nlevels,ifleafbox,ifpwexpform,ifpwexpeval,lcoefsptot,norder)
      
      if(ifprint .eq. 1)
     1    call prinf_long('memory for tens_prod grid=*',lcoefsptot,1)
      allocate(coefsp(lcoefsptot),stat=ier)
      if(ier.ne.0) then
         print *, "Cannot allocate workspace for proxy charge/potential"
         print *, "lcoefsptot=", lcoefsptot
         ier = 4
         return
      endif

c     initialization of the work array
      do ilev = 0,nlevels-1
      do ibox=itree(2*ilev+1),itree(2*ilev+2)
         if (ifpwexpform(ibox).eq.1) then
            call pdmk_coefsp_zero(ndtot,npbox,coefsp(lpaddr(1,ibox)))
         endif
      enddo
      enddo
      do ilev = 0,nlevels-1
      do ibox=itree(2*ilev+1),itree(2*ilev+2)
         if (ifpwexpeval(ibox).eq.1) then
            call pdmk_coefsp_zero(ndtot,npbox,coefsp(lpaddr(2,ibox)))
         endif
      enddo
      enddo

      call cpu_time(time2)
C$    time2=omp_get_wtime()
      ttotal = ttotal+time2-time1
      call prin2('time in malloc and initialization=*',time2-time1,1)
      
      allocate(isgn(dim,mc))
      call get_child_box_sign(dim,isgn)

c     c2ptransmat - 1d translation matrices used in the upward pass of forming
c     proxy charges from child boxes to the parent box
c     p2ctransmat - 1d translation matrices used in the downward pass of splitting
c     proxy potential of the parent box to its child boxes
c     They are transpose of each other by symmetry
      allocate(p2ctransmat(norder,norder,dim,mc))
      allocate(c2ptransmat(norder,norder,dim,mc))
c     we use Chebyshev tensor product grid since 1. it's slightly cheaper
c     to evaluate Chebyshev polynomials; 2. Chebyshev polynomials are slightly
c     better than Legendre polynomials if we only need to do interpolation, i.e.,
c     there is no need to do integration.
c
c     Maybe prolates are better for bandlimited functions?
      ipoly=1
      call dmk_get_coefs_translation_matrices(dim,ipoly,
     1    norder,isgn,p2ctransmat,c2ptransmat)

      do ibox=1,nboxes
         iftensprodform(ibox)=0
      enddo





      ikernel = 3
      
c     compute Fourier transform of the truncated kernel at level -1 and
c     Fourier transforms of the difference kernels at levels 0, ..., nlevels-1
c     along radial directions!
      
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      nfouriermax = dim*(npwmax/2)**2
      allocate(dkernelft(0:nfouriermax,-1:nlevels))
      allocate(npw(-1:nlevels))
      allocate(nfourier(-1:nlevels))
      allocate(hpw(-1:nlevels))
      allocate(ws(-1:nlevels))
      allocate(rl(-1:nlevels))

c     windowed kernel at level -1
      bsize = boxsize(0)
      
      call get_PSWF_windowed_kernel_pwterms(ikernel,rpars,
     1    beta,dim,bsize,eps,hpw(-1),npw(-1),ws(-1),rl(-1))
      nfourier(-1)=dim*(npw(-1)/2)**2
      call get_windowed_kernel_Fourier_transform(ikernel,
     1    rpars,dim,beta,
     1    bsize,rl(-1),npw(-1),hpw(-1),ws(-1),wprolate,
     3    nfourier(-1),dkernelft(0,-1))

      rl(0)=rl(-1)
      do ilev=1,nlevels
         rl(ilev)=rl(ilev-1)/2
      enddo
c     difference kernels at levels 0, ..., nlevels-1
      do ilev=0,nlevels
         bsize = boxsize(ilev)
         bsizebig = bsize
         bsizesmall = bsize/2

         call get_PSWF_difference_kernel_pwterms(ikernel,rpars,
     1       beta,dim,bsize,eps,hpw(ilev),npw(ilev),ws(ilev))

         nfourier(ilev)=dim*(npw(ilev)/2)**2

         if (ilev.eq.0) then
            call get_difference_kernel_Fourier_transform(ikernel,
     1          rpars,dim,beta,bsizesmall,bsizebig,
     2          npw(ilev),hpw(ilev),ws(ilev),
     3          wprolate,nfourier(ilev),dkernelft(0,ilev))
         endif

         if (ilev.gt.0) then
            if (ikernel.eq.0) then
c           Yukawa kernel in 2 and 3 dimensions, no scale invariance
               call get_difference_kernel_Fourier_transform(ikernel,
     1             rpars,dim,beta,
     2             bsizesmall,bsizebig,npw(ilev),hpw(ilev),ws(ilev),
     3             wprolate,nfourier(ilev),dkernelft(0,ilev))
            elseif (ikernel.eq.2.and.dim.eq.3) then
c           1/r^2 kernel in 3d
               do i=0,nfourier(0)
                  dkernelft(i,ilev)=dkernelft(i,ilev-1)*4
               enddo
            elseif (ikernel.eq.1.and.dim.eq.2) then
c           log(r) kernel in 2d
               do i=0,nfourier(0)
                  dkernelft(i,ilev)=dkernelft(i,ilev-1)
               enddo
            elseif ((ikernel.eq.1.and.dim.eq.3) .or.
     1              (ikernel.eq.2.and.dim.eq.2)) then
c           1/r kernel in 2d and 3d
               do i=0,nfourier(0)
                  dkernelft(i,ilev)=dkernelft(i,ilev-1)*2
               enddo
            elseif ((ikernel.eq.3).and.(dim.eq.3)) then
c           3d stokeslet
               do i=0,nfourier(0)
                  dkernelft(i,ilev)=dkernelft(i,ilev-1)/2
               enddo
            elseif ((ikernel.eq.3).and.(dim.eq.2)) then
c           2d stokeslet
               do i=0,nfourier(0)
                  dkernelft(i,ilev)=dkernelft(i,ilev-1)/4
               enddo
            endif
         endif
      enddo

c     residual kernels at all levels for the Stokeslet

      if (ikernel .eq. 3) then
         ncoefsmax=200
         allocate(coefs1_diag(ncoefsmax))
         allocate(coefs2_diag(ncoefsmax))
         allocate(coefs1_offd(ncoefsmax))
         allocate(coefs2_offd(ncoefsmax))

         bsize = 1.0d0
         rl0 = rl(0)/boxsize(0)
         call stokes_residual_kernel_coefs(eps,dim,beta,
     1       bsize,rl0,wprolate,n1_diag,coefs1_diag,
     2       n2_diag,coefs2_diag,n1_offd,coefs1_offd,
     3       n2_offd,coefs2_offd)

cccc output for Matlab to produce monomial coefficients for SIMD fast kernel evaluation         
cccc 1111    format(E23.16, 50(1x,E23.16))
cccc         print *, 'n1_diag=', n1_diag, 'n1_offd=', n1_offd
cccc         print *, 'n2_diag=', n2_diag, 'n2_offd=', n2_offd
cccc         
cccc         write(*,'(a)', advance='no') 'chebc1d = ['
cccc         write(*,1111,advance='no') (coefs1_diag(j), j=1,n1_diag)
cccc         write(*,*) '];' 
cccc
cccc         write(*,'(a)', advance='no') 'chebc1o = ['
cccc         write(*,1111,advance='no') (coefs1_offd(j), j=1,n1_offd)
cccc         write(*,*) '];'
cccc
cccc         write(*,'(a)', advance='no') 'chebc2d = ['
cccc         write(*,1111,advance='no') (coefs2_diag(j), j=1,n2_diag) 
cccc         write(*,*) '];'
cccc
cccc         write(*,'(a)', advance='no') 'chebc2o = ['
cccc         write(*,1111,advance='no') (coefs2_offd(j), j=1,n2_offd)
cccc         write(*,*) '];'
cccc         pause
      endif
      
      
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      tottimeinfo(3)=tottimeinfo(3)+time2-time1
      
      call prin2('time on computing Fourier transforms=*',time2-time1,1)
      if (ifprint.eq.1) call prinf('npw=*',npw,nlevels)






      
c
c     upward pass for calculating equivalent charges
c     
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      ntot=0
      ns2tp=0
      nlevstart = max(nlevels-1,0)
      do ilev=nlevstart,nlevstart
         sc=2.0d0/boxsize(ilev)
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
            istart = isrcse(1,ibox)
            iend = isrcse(2,ibox)
            npts = iend-istart+1

c           Check if current box needs to form pw exp         
            if (ifpwexpform(ibox).eq.1) then
c              form equivalent charges directly form sources
               call pdmk_charge2proxycharge(dim,ndtot,norder,
     1             npts,sourcesort(1,istart),stokletsort(1,1,istart),
     2             centers(1,ibox),sc,coefsp(lpaddr(1,ibox)))
               iftensprodform(ibox)=1
               ntot=ntot+npts
               ns2tp=ns2tp+1
            endif
         enddo
      enddo

      do ilev=nlevstart-1,0,-1
         sc=2.0d0/boxsize(ilev)
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
c           Check if current box needs to form pw exp         
            if (ifpwexpform(ibox).eq.1) then
c           find equivalent charges of the parent box by merging 
c           equivalent charges of child boxes
               nchild = itree(iptr(4)+ibox-1)
               do j=1,nchild
                  jbox = itree(iptr(5)+mc*(ibox-1)+j-1)
                  if (iftensprodform(jbox).eq.1) then
c                    translate equivalent charges from child to parent
                     call tens_prod_trans_add(dim,ndtot,norder,
     1                   coefsp(lpaddr(1,jbox)),norder,
     2                   coefsp(lpaddr(1,ibox)),
     3                   c2ptransmat(1,1,1,j))
                  else
                     jstart = isrcse(1,jbox) 
                     jend = isrcse(2,jbox)
                     npts = jend-jstart+1
                     if (npts.gt.0) then
c                       form equivalent charges directly form sources
                        call pdmk_charge2proxycharge(dim,ndtot,
     1                      norder,npts,sourcesort(1,jstart),
     2                      stokletsort(1,1,jstart),centers(1,ibox),
     3                      sc,coefsp(lpaddr(1,ibox)))
                        ntot=ntot+npts
                        ns2tp=ns2tp+1
                     endif
                  endif
               enddo
               iftensprodform(ibox)=1
            endif
         enddo
      enddo
      
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      dt=time2-time1
      ttotal = ttotal+dt
      tottimeinfo(4) = tottimeinfo(4)+dt

      if (ifprint.eq.1) then
         call prinf('number of boxes calling charge2tensprod=*',ns2tp,1)
         call prinf('number of source points involved=*',ntot,1)
         call prinf('total number of source points=*',ns,1)
         call prin2('time in forming equivalent charges=*',dt,1)
      endif




      call cpu_time(time1)
C$    time1=omp_get_wtime()

c     main loop over levels
      do ilev=-1,nlevels
         npwlevel = ilev
         call stokesdmkmain(nd,dim,eps,ikernel,iperiod,
     1       ifstoklet,stokletsort,ifstrslet,strsletsort,strsvecsort,
     2       ns,sourcesort,csourcesort,
     3       nt,targsort,ctargsort,
     4       nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     5       npwlevel,ndiv,nboxsrcpts,nboxtargpts,ifleafbox,
     6       norder,lpaddr,coefsp,rmlexp,ifpwexpform,ifpwexpeval,
     7       iftensprodeval,p2ctransmat,beta,wprolate,
     8       npw(ilev),hpw(ilev),nfourier(ilev),dkernelft(0,ilev),
     9       n1_diag,coefs1_diag,n2_diag,coefs2_diag,
     *       n1_offd,coefs1_offd,n2_offd,coefs2_offd,
     1       isrcse,itargse,ifppreg,potsort,presort,gradsort,
     2       ifppregtarg,pottargsort,pretargsort,gradtargsort,timeinfo)       
         do i=1,5
            tottimeinfo(i+2)=tottimeinfo(i+2)+timeinfo(i)
         enddo
      enddo






      
c     finally, needs to subtract the self-interaction from the planewave sweeping
      zero=0.0d0
      call prol0eva(zero,wprolate,psi0,derpsi0)
      call prolate_intvals(beta,wprolate,c0,c1,g0d2,c4)

      if (ikernel.eq.3.and.dim.eq.2) then
         bsize=boxsize(0)
         rl0=bsize*sqrt(dim*1.0d0)*2
         
         call stokes_windowed_kernel_value_at_zero(dim,beta,
     1          bsize,rl0,wprolate,st2dwk0)
      endif

      do ilev=0,nlevels
         bsize = boxsize(ilev)
         if (ilev .eq. 0) bsize=bsize*0.5d0
c        sc is the value of the windowed kernel at the origin
         if (ikernel.eq.3 .and. dim.eq.2) then
            jlev = ilev
            if (ilev .eq. 0) jlev = 1
            sc=st2dwk0+0.5d0*jlev*log(2.0d0)
         elseif (ikernel.eq.3 .and. dim.eq.3) then
            sc = (2.0d0/3)*(1+0.5d0*g0d2**2*beta**2)*psi0/(c0*bsize)
         elseif (ikernel.eq.2.and.dim.eq.3) then
c        1/r^2 kernel in 3d
            sc=psi0/(2*c1*bsize*bsize)
         elseif (ikernel.eq.1.and.dim.eq.2) then
c        log(r) kernel in 2d
            sc=dlogtk0-ilev*log(2.0d0)
         elseif ((ikernel.eq.1.and.dim.eq.3) .or.
     1           (ikernel.eq.2.and.dim.eq.2)) then
c        1/r kernel in 2d and 3d
            sc=psi0/(c0*bsize)
         endif
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
            istart = isrcse(1,ibox)
            iend = isrcse(2,ibox)
c           subtract the self-interaction
            if (ifleafbox(ibox).eq.1) then
               do i=istart,iend
                  do ind=1,nd
                  do k=1,dim
                     potsort(ind,k,i)=potsort(ind,k,i)
     1                   -sc*stokletsort(ind,k,i)
                  enddo
                  enddo
               enddo
            endif
         enddo
      enddo
         
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      ttotal = ttotal+time2-time1




      
      ifprint=1
      if (ifprint.eq.1) then
         call prinf('============================================*',i,0)      
         call prinf('laddr=*',itree,2*(nlevels+1))
         call prinf('nlevels=*',nlevels,1)
         call prinf('nboxes=*',nboxes,1)
         call prinf('nleafbox=*',nleafbox,1)
         call prin2('time in tree build=*',dttree,1)
         pps=(ns+nt+0.0d0)/dttree
         call prin2('points per sec in tree build=*',pps,1)
         call prin2('time in pts_tree_sort=*',dtsort,1)
         pps=(ns+nt+0.0d0)/dtsort
         call prin2('points per sec in tree sort=*',pps,1)
         call prinf('=== STEP 1 (build tree) =========*',i,0)         
         call prinf('=== STEP 2 (sort points) ========*',i,0)         
         call prinf('=== STEP 3 (precomputation) =====*',i,0)         
         call prinf('=== STEP 4 (form mp) ============*',i,0)         
         call prinf('=== STEP 5 (mp to loc) ==========*',i,0)         
         call prinf('=== STEP 6 (eval loc) ===========*',i,0)         
         call prinf('=== STEP 7 (direct) =============*',i,0)         
         call prin2('total time info=*', tottimeinfo,7)
         call prin2('time in stokesdmk main=*',ttotal,1)
         pps=(ns*ifppreg+nt*ifppregtarg+0.0d0)/ttotal
         call prin2('points per sec=*',pps,1)
         call prinf('============================================*',i,0)
      endif







      
c
c     resort the output arrays in input order
c
      if(ifppreg.eq.1) then
        call dreorderi(ndtot,ns,potsort,pot,isrc)
      endif

      if(ifppregtarg.eq.1) then
        call dreorderi(ndtot,nt,pottargsort,pottarg,itarg)
      endif

      return
      end
c
c
c
c
c
      subroutine stokesdmkmain(nd,dim,eps,ikernel,iperiod,
     1    ifstoklet,stokletsort,ifstrslet,strsletsort,strsvecsort,
     2    nsource,sourcesort,csourcesort,
     3    ntarget,targetsort,ctargetsort,
     4    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     5    npwlevel,ndiv,nboxsrcpts,nboxtargpts,ifleafbox,
     6    norder,lpaddr,coefsp,rmlexp,ifpwexpform,ifpwexpeval,
     7    iftensprodeval,p2ctransmat,beta,wprolate,
     8    npw,hpw,nfourier,fhat,
     9    n1_diag,coefs1_diag,n2_diag,coefs2_diag,
     *    n1_offd,coefs1_offd,n2_offd,coefs2_offd,
     1    isrcse,itargse,ifppreg,pot,pre,grad,
     2    ifppregtarg,pottarg,pretarg,gradtarg,timeinfo)
c
      implicit none
      integer nd,dim,ikernel,iperiod,nsource,ntarget

      integer ndiv,nlevels,npwlevel,ncutoff
      integer ifstoklet,ifstrslet
      integer ifppreg,ifppregtarg,ndigits
      integer n1_diag,n2_diag,n1_offd,n2_offd
      
      real *8 eps,rpars(10)
      real *8 coefs1_diag(*),coefs2_diag(*)
      real *8 coefs1_offd(*),coefs2_offd(*)

      real *8 sourcesort(dim,nsource)
      real *8 csourcesort(nsource,dim)

      real *8 stokletsort(nd,dim,*)
      real *8 strsletsort(nd,dim,*)
      real *8 strsvecsort(nd,dim,*)

      real *8 targetsort(dim,ntarget)
      real *8 ctargetsort(ntarget,dim)

      real *8 pot(nd,dim,*)
      real *8 pre(nd,*)
      real *8 grad(nd,dim,dim,*)

      real *8 pottarg(nd,dim,*)
      real *8 pretarg(nd,*)
      real *8 gradtarg(nd,dim,dim,*)

      real *8 coefsp(*),rmlexp(*)
      real *8 p2ctransmat(norder,norder,dim,2**dim)
      real *8 wprolate(*)
c
      real *8 timeinfo(*),time1,time2
      real *8 centers(dim,*)
      real *8 boxsize(0:nlevels)
c
      integer iptr(8)
      integer ltree
      integer itree(ltree)
      integer nboxes
      integer isrcse(2,nboxes),itargse(2,nboxes)
      integer nboxsrcpts(nboxes),nboxtargpts(nboxes)
      integer ifleafbox(nboxes)
      integer norder
      integer *8 lpaddr(2,nboxes)
c
c     local variables
      integer, allocatable :: nlist1(:), list1(:,:)
      integer, allocatable :: nlistpw(:), listpw(:,:)
c
      integer *8, allocatable :: iaddr(:,:)
      integer *8 lmptot,i8

      integer nfourier,ndtot
      integer ifprint,itype,dad,nchild,ncoll,nnbors
      integer ibox,jbox,ind, npw,npw2, istart,iend,jstart,jend
      integer istartt,iendt,jstartt,jendt,istarts,iends,jends,jstarts
      integer nmax,nhess,mc,nexp,n1,n2,ns,nb,ier,ilev,jlev
      integer isep,i,j,k,iperiod0,ifself,ipoly0
      integer mnlistpw,npts,nptssrc,nptstarg
      integer mnbors,mnlist1,mnlist2
      integer ifpwexpform(nboxes),ifpwexpeval(nboxes)
      integer iftensprodeval(nboxes)
      
      real *8, allocatable :: ts(:),rk(:,:),rksq(:)
      real *8, allocatable :: pswfft(:)
      real *8 hpw,fhat(0:nfourier)
      
      real *8 bs0,bsize,d,sc
      real *8 rsc,cen,rscnear,cennear,bsizeinv,d2min,d2max,thresh2

      complex *16, allocatable :: wpwshift(:,:)

      real *8 omp_get_wtime
      real *8 beta
c
      complex *16, allocatable :: tab_coefs2pw(:,:)
      complex *16, allocatable :: tab_pw2coefs(:,:)

      real *8 xq(100),wts,umat,vmat
      
            
      do i=1,10
         timeinfo(i)=0
      enddo

      itype = 0
      call chebexps(itype,norder,xq,umat,vmat,wts)
c      
c     ifprint is an internal information printing flag. 
c     Suppressed if ifprint=0.
c     Prints timing breakdown and other things if ifprint=1.
c
      ifprint=1

      
      bs0 = boxsize(0)
      mc = 2**dim
      mnbors=3**dim
      ndtot = nd*dim
      
      ncutoff=max(npwlevel,0)
      bsize = boxsize(ncutoff)
      
      if (ifprint.ge.1) then
         call prin2('============================================*',d,0)      
         call prinf('npwlevel =*',npwlevel,1)
      endif
      
c     get planewave nodes
      allocate(ts(-npw/2:(npw-1)/2))
      allocate(rk(dim,npw**dim),rksq(npw**dim))

      
c     trapezoidal rule - npw odd
      do i=-npw/2,(npw-1)/2
         ts(i)=i*hpw
      enddo
      
c     tables converting tensor product polynomial expansion coefficients of 
c     the charges to planewave expansion coefficients - on the source side
      allocate(tab_coefs2pw(npw,norder))
c     tables converting planewave expansions to tensor product polynomial
c     expansion coefficients of the potentials - on the target side
      allocate(tab_pw2coefs(npw,norder))

      ipoly0=1
      call dmk_mk_coefs_pw_conversion_tables(ipoly0,norder,npw,
     1    ts,xq,hpw,bsize,tab_coefs2pw,tab_pw2coefs)
         
c     
c     compute list info
c
      call dmk_compute_mnlistpw(dim,nboxes,nlevels,ltree,itree,
     1    iptr,centers,boxsize,mnlistpw)

      allocate(nlistpw(nboxes),listpw(mnlistpw,nboxes))
c     listpw contains source boxes in the pw interaction
      call pdmk_compute_listpw(dim,ncutoff,nboxes,nlevels,
     1    ltree,itree,iptr,centers,boxsize,itree(iptr(1)),
     3    ifpwexpform,mnlistpw,nlistpw,listpw)      
c
c
c     compute list info for direct interactions
c
c     list1 contains boxes that are neighbors of the given box
c
      mnlist1 = 3**dim
      allocate(list1(mnlist1,nboxes),nlist1(nboxes))
c     modified list1 for direct evaluation
c     list1 of a childless source box ibox at ilev<=npwlevel
c     contains all childless target boxes that are neighbors of ibox
c     at or above npwlevel
      iperiod0=0
      call pdmk_compute_modified_list1(dim,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,iperiod0,
     2    ifleafbox,ncutoff,
     3    mnlist1,nlist1,list1)
c
c     direct evaluation if the cutoff level is >= the maximum level 
      if (npwlevel .ge. nlevels .and. npwlevel.gt.0) goto 1800

      
c
c     Multipole and local planewave expansions will be held in workspace
c     in locations pointed to by array iaddr(2,nboxes).
      allocate(iaddr(2,nboxes))
c     calculate memory needed for multipole and local planewave expansions
      call pdmk_mpalloc_stokes(ndtot,dim,itree,iaddr,
     1    nlevels,ncutoff,ifpwexpform,ifpwexpeval,lmptot,npw)
      if(ifprint .eq. 1) call prinf_long('lmptot is *',lmptot,1)

         

c     number of plane-wave modes
      nexp = npw**(dim-1)*((npw+1)/2)
      if(ifprint .eq. 1) call prinf_long('nexp is *',nexp,1)
      
      call meshnd(dim,ts,npw,rk)
      call meshndsq(dim,nexp,rk,rksq)
      
c     initialization of the work array 
      do ilev = ncutoff,ncutoff
      do ibox=itree(2*ilev+1),itree(2*ilev+2)
c        only these boxes need initialization
c        we use initialization only necessary since the initialization
c        somehow is expensive in Fortran? 
c        need better memory management or switch to a better language
         if (ifpwexpeval(ibox).eq.1 .and. ifpwexpform(ibox).eq.0) then
            call dmk_pwzero(ndtot,nexp,rmlexp(iaddr(2,ibox)))
         endif
      enddo
      enddo
c     precomputation time
      call cpu_time(time1)
C$    time1=omp_get_wtime()
      
c      
c     compute translation matrices for PW expansions
c     translation only at the cutoff level
      nmax = 1

      allocate(wpwshift(nexp,(2*nmax+1)**dim),stat=ier)
      call mk_pw_translation_matrices(dim,bsize,npw,ts,nmax,
     1    wpwshift)

      allocate(pswfft(nexp),stat=ier)

      call mk_tensor_product_Fourier_transform(dim,
     1    npw,nfourier,fhat,nexp,pswfft)

      call cpu_time(time2)
C$    time2=omp_get_wtime()
      timeinfo(1)=time2-time1





      
c
c
      if(ifprint .ge. 1) 
     $   call prinf('=== STEP 1 (form mp) ====*',i,0)
      call cpu_time(time1)
C$    time1=omp_get_wtime()

c     
c       ... step 1, form multipole pw expansions at the cutoff level
c       
      nb=0
      do 1100 ilev = ncutoff,ncutoff
C$OMP PARALLEL DO DEFAULT (SHARED)
C$OMP$PRIVATE(ibox)
C$OMP$SCHEDULE(DYNAMIC)
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
c           Check if current box needs to form pw exp         
            if(ifpwexpform(ibox).eq.1) then
               nb=nb+1
c              form the pw expansion
               call dmk_proxycharge2pw(dim,ndtot,norder,
     1             coefsp(lpaddr(1,ibox)),npw,tab_coefs2pw,
     3             rmlexp(iaddr(1,ibox)))

               call stokesdmk_multiply_kernelFT(dim,nd,nexp,
     1              rmlexp(iaddr(1,ibox)),pswfft,rk,rksq)

c              copy the multipole PW exp into local PW exp
c              for self interaction
               call dmk_copy_pwexp(ndtot,nexp,rmlexp(iaddr(1,ibox)),
     1             rmlexp(iaddr(2,ibox)))
            endif
         enddo
C$OMP END PARALLEL DO 
c     end of ilev do loop
 1100 continue

      call cpu_time(time2)
C$    time2=omp_get_wtime()
      timeinfo(2)=time2-time1

      if(ifprint.ge.1)
     $     call prinf('number of boxes in form mp=*',nb,1)





      

      
      if(ifprint.ge.1)
     $    call prinf('=== Step 2 (mp to loc) ===*',i,0)
c      ... step 2, convert multipole pw expansions into local
c       pw expansions

      call cpu_time(time1)
C$    time1=omp_get_wtime()
      
      do 1300 ilev = ncutoff,ncutoff
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,jbox,j,ind)
C$OMP$SCHEDULE(DYNAMIC)
         do 1250 ibox = itree(2*ilev+1),itree(2*ilev+2)
            npts=nboxsrcpts(ibox)+nboxtargpts(ibox)
            if (npts.eq.0) goto 1250 
c           ibox is the target box
c           shift PW expansions
            
            do j=1,nlistpw(ibox)
               jbox=listpw(j,ibox)

c              jbox is the source box
               call dmk_find_pwshift_ind(dim,iperiod0,
     1             centers(1,ibox),centers(1,jbox),
     2             bs0,bsize,nmax,ind)
               call dmk_shiftpw(ndtot,nexp,rmlexp(iaddr(1,jbox)),
     1             rmlexp(iaddr(2,ibox)),wpwshift(1,ind))
            enddo
 1250    continue
C$OMP END PARALLEL DO        
 1300 continue
c
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      timeinfo(3) = time2-time1



      
      
      

      if(ifprint.ge.1)
     $    call prinf('=== step 3 (eval loc) ===*',i,0)

c     ... step 3, evaluate all local pw expansions
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      nb=0
      do 1500 ilev = ncutoff,ncutoff
         sc = 2.0d0/boxsize(ilev)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,istartt,iendt,istarts,iends,nptssrc,nptstarg)
C$OMP$PRIVATE(i,j,k)
C$OMP$SCHEDULE(DYNAMIC)
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
            npts=nboxsrcpts(ibox)+nboxtargpts(ibox)
            if (ifpwexpeval(ibox).eq.1 .and. npts.gt.0) then
               nb=nb+1
               istartt = itargse(1,ibox) 
               iendt = itargse(2,ibox)
               nptstarg = iendt-istartt + 1
              
               istarts = isrcse(1,ibox)
               iends = isrcse(2,ibox)
               nptssrc = iends-istarts+1

               call dmk_pw2proxypot(dim,ndtot,norder,npw,
     1            rmlexp(iaddr(2,ibox)),tab_pw2coefs,
     3            coefsp(lpaddr(2,ibox)))
               if (npwlevel.eq.-1) goto 3000

               if (iftensprodeval(ibox).eq.0) goto 1400

               if (ifppregtarg.gt.0 .and. nptstarg.gt.0) then
                  call pdmk_ortho_evalt_nd(dim,ndtot,norder,
     1                coefsp(lpaddr(2,ibox)),nptstarg,
     2                targetsort(1,istartt),centers(1,ibox),sc,
     3                pottarg(1,1,istartt))
               endif
               
               if (ifppreg.gt.0 .and. nptssrc.gt.0) then
                  call pdmk_ortho_evalt_nd(dim,ndtot,norder,
     1                coefsp(lpaddr(2,ibox)),nptssrc,
     2                sourcesort(1,istarts),centers(1,ibox),sc,
     3                pot(1,1,istarts))
               endif

 1400          continue
               nchild = itree(iptr(4)+ibox-1)
               do j=1,nchild
                  jbox = itree(iptr(5) + (ibox-1)*mc+j-1)
                  if (iftensprodeval(jbox).eq.1 .and.
     1                ifpwexpeval(jbox).eq.0) then
c                    evaluate tensor product polynomial approximation at targets
                     jstartt = itargse(1,jbox) 
                     jendt = itargse(2,jbox)
                     nptstarg = jendt-jstartt + 1
                     
                     jstarts = isrcse(1,jbox)
                     jends = isrcse(2,jbox)
                     nptssrc = jends-jstarts+1
                     if (ifppregtarg.gt.0 .and. nptstarg.gt.0) then
                        call pdmk_ortho_evalt_nd(dim,ndtot,norder,
     1                      coefsp(lpaddr(2,ibox)),nptstarg,
     2                      targetsort(1,jstartt),centers(1,ibox),sc,
     3                      pottarg(1,1,jstartt))
                     endif
               
                     if (ifppreg.gt.0 .and. nptssrc.gt.0) then
                        call pdmk_ortho_evalt_nd(dim,ndtot,norder,
     1                      coefsp(lpaddr(2,ibox)),nptssrc,
     2                      sourcesort(1,jstarts),centers(1,ibox),sc,
     3                      pot(1,1,jstarts))
                     endif
                     
                  elseif (ifpwexpeval(jbox).eq.1) then
c                    translate tensor product polynomial from parent to child
                     call tens_prod_trans_add(dim,ndtot,norder,
     1                   coefsp(lpaddr(2,ibox)),norder,
     2                   coefsp(lpaddr(2,jbox)),
     3                   p2ctransmat(1,1,1,j))
                  endif
               enddo
            endif
         enddo
C$OMP END PARALLEL DO        
 1500 continue
cccc      deallocate(rmlexp)
      call cpu_time(time2)
C$    time2 = omp_get_wtime()      
      timeinfo(4) = time2 - time1
      if(ifprint.ge.1)
     $     call prinf('number of boxes in local eval=*',nb,1)



      
 1800 continue
      if(ifprint .ge. 1)
     $     call prinf('=== STEP 4 (direct) =====*',i,0)
c
cc
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      bsize = boxsize(ncutoff)
c     residual kernel starts from level 1
      if (ncutoff.eq.0) bsize=bsize/2
c     kernel truncated at bsize, i.e., K(x,y)=0 for |x-y|^2 > d2max
      d2max = bsize**2
c     minimal value of r^2 for list 2
      d2min = d2max/4
c     minimal value of r^2 to ignore self-interaction
c     compatible with the FMM3d, but should really rescale using
c     boxsize(0)!
      thresh2 = 1.0d-30
      
      bsizeinv = 1.0d0/bsize

c     used in the kernel approximatin for boxes in list1
c     in 3d, we use r as the independent variable in polynomial approximation
      rsc = bsizeinv*2
      cen = -bsize/2
c     in 2d, we use r^2 as the independent variable in polynomial approximation
      if (dim.eq.2) then
         rsc=bsizeinv*bsizeinv*2
         cen=-1.0d0
      endif

c     used in the kernel approximation for boxes in list2
c     polynomial approximation using r^2 as the independent variable for boxes
c     in list 2
      cennear = -5.0d0/3
      rscnear = bsizeinv**2*8.0d0/3
      

      ndigits=nint(log10(1.0d0/eps)-0.1)

      nb=0
      do 2000 jlev = ncutoff,ncutoff
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,jbox,istart,iend,jstartt,jendt,jstarts,jends)
C$OMP$PRIVATE(ns,n1,nptssrc,nptstarg,npts)
C$OMP$SCHEDULE(DYNAMIC)  
         do jbox = itree(2*jlev+1),itree(2*jlev+2)
c           jbox is the source box here            
            jstart = isrcse(1,jbox)
            jend = isrcse(2,jbox)
            npts = jend-jstart+1

            n1 = nlist1(jbox)
            
            if (npts.gt.0 .and. n1.gt.0) then
               nb=nb+1
            endif
            
            if (npts.gt.0 .and. n1.gt.0) then
               ifself=1
               do i=1,n1
cccc              ibox is the target box
                  ibox = list1(i,jbox)

                  istarts = isrcse(1,ibox)
                  iends = isrcse(2,ibox)
                  nptssrc = iends-istarts + 1

                  istartt = itargse(1,ibox)
                  iendt = itargse(2,ibox)
                  nptstarg = iendt-istartt+1
                  
                  if (nptstarg.gt.0.and.ifppregtarg.gt.0) then
                     call stokesdmk_direct_c(nd,dim,ikernel,
     1                   ndigits,rsc,cen,bsizeinv,ifself,
     2                   n1_diag,coefs1_diag,n1_offd,coefs1_offd,
     3                   thresh2,d2max,jstart,jend,sourcesort,
     4                   ifstoklet,stokletsort,
     5                   ifstrslet,strsletsort,strsvecsort,
     6                   istartt,iendt,ntarget,ctargetsort,
     7                   ifppregtarg,pottarg,pretarg,gradtarg)
                  endif
                     
                  if (nptssrc.gt.0.and.ifppreg.gt.0) then
                     call stokesdmk_direct_c(nd,dim,ikernel,
     1                   ndigits,rsc,cen,bsizeinv,ifself,
     2                   n1_diag,coefs1_diag,n1_offd,coefs1_offd,
     3                   thresh2,d2max,jstart,jend,sourcesort,
     4                   ifstoklet,stokletsort,
     5                   ifstrslet,strsletsort,strsvecsort,
     6                   istarts,iends,nsource,csourcesort,
     7                   ifppreg,pot,pre,grad)
                  endif
               enddo
            endif
         enddo
C$OMP END PARALLEL DO         
 2000 continue
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      
      if(ifprint.ge.1)
     $     call prinf('number of direct interaction boxes =*',nb,1)
      timeinfo(5) = time2-time1



      
      
 3000 continue
      if(ifprint.ge.1) call prin2('timeinfo=*',timeinfo,5)
      d = 0
      do i = 1,5
         d = d + timeinfo(i)
      enddo

      if(ifprint.ge.1) call prin2('sum(timeinfo)=*',d,1)
      
      return
      end
c
c
c
c
c
c
c
c
c------------------------------------------------------------------     
      subroutine stokesdmk_direct_c(nd,dim,ikernel,
     1    ndigits,rsc,cen,bsizeinv,ifself,
     2    n_diag,coefs_diag,n_offd,coefs_offd,
     3    thresh2,d2max,istart,iend,source,
     4    ifstoklet,stoklet,
     5    ifstrslet,strslet,strsvec,
     6    jstart,jend,ntarget,ctarg,
     7    ifppreg,pot,pre,grad)
c--------------------------------------------------------------------
c     This subroutine adds the contribution due to sources
c     istart to iend in the source array to the fields at targets
c     jstart to jend in the target array.
c
      implicit none
c
      integer nd,nsource,ntarget,ndigits
      integer dim,iperiod,ikernel,n_diag,n_offd
      integer istart,iend,jstart,jend,ns,ntarg
      integer ifstoklet,ifstrslet
      integer ifppreg,ifself
      integer i,j,k,iffast
c
      real *8 eps,coefs_diag(*),coefs_offd(*)
      real *8 source(dim,*)
      real *8 stoklet(nd,dim,*)
      real *8 strslet(nd,dim,*)
      real *8 strsvec(nd,dim,*)
      real *8 bsizeinv,thresh2,d2max,rsc,cen,rlambda
      real *8 ctarg(ntarget,dim),ztarg
      real *8 pot(nd,dim,*)
      real *8 pre(nd,*)
      real *8 grad(nd,dim,dim,*)
c
        
      ns = iend - istart + 1
      ntarg = jend-jstart+1

      iffast = 1

      if (iffast.eq.1) goto 1200

      if ((ifstoklet.eq.1).and.(ifstrslet.eq.0)) then
         if((ifppreg.eq.1) .and. (ifself.eq.1)) then
            if (ikernel.eq.3 .and. dim.eq.2) then
               call st2d_local_kernel_directcp(nd,dim,rsc,cen,bsizeinv,
     1             d2max,source(1,istart),ns,stoklet(1,1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),
     3             ntarg,n_diag,coefs_diag,n_offd,coefs_offd,
     4             pot(1,1,jstart))
            endif

            if (ikernel.eq.3 .and. dim.eq.3) then
               call st3d_local_kernel_directcp(nd,dim,rsc,cen,bsizeinv,
     1             d2max,source(1,istart),ns,stoklet(1,1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     3             ntarg,n_diag,coefs_diag,n_offd,coefs_offd,
     4                pot(1,1,jstart))
            endif
         endif
      endif

      return
 1200 continue

      if (dim.eq.3) then
         call stokes_local_kernel_directcp_fast(nd,dim,ndigits,
     1       rsc,cen,bsizeinv,thresh2,d2max,source(1,istart),ns,
     2       stoklet(1,1,istart),
     3       ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     4       ntarg,pot(1,1,jstart))
      elseif (dim.eq.2) then
         call stokes_local_kernel_directcp_fast(nd,dim,ndigits,
     1       rsc,cen,bsizeinv,thresh2,d2max,source(1,istart),ns,
     2       stoklet(1,1,istart),
     3       ctarg(jstart,1),ctarg(jstart,2),ztarg,
     4       ntarg,pot(1,1,jstart))
      endif
      
      return
      end
c
c
c
c
c
c
c
c
c------------------------------------------------------------------     
      subroutine stokesdmk_direct_near_c(nd,dim,ikernel,
     1    ndigits,rsc,cen,bsizeinv,ifself,
     2    n_diag,coefs_diag,n_offd,coefs_offd,
     3    d2min,d2max,istart,iend,source,
     4    ifstoklet,stoklet,
     5    ifstrslet,strslet,strsvec,
     6    jstart,jend,ntarget,ctarg,
     7    ifppreg,pot,pre,grad)
c--------------------------------------------------------------------
c     This subroutine adds the contribution due to sources
c     istart to iend in the source array to the fields at targets
c     jstart to jend in the target array.
c
      implicit none
c
      integer nd,nsource,ntarget,ndigits
      integer dim,iperiod,ikernel,n_diag,n_offd
      integer istart,iend,jstart,jend,ns,ntarg
      integer ifstoklet,ifstrslet
      integer ifppreg,ifself
      integer i,j,k,iffast
c
      real *8 eps,coefs_diag(*),coefs_offd(*)
      real *8 source(dim,*)
      real *8 cen,d2min,d2max,rsc,bsizeinv
      real *8 stoklet(nd,dim,*),strslet(nd,dim,*),strsvec(nd,dim,*)
      real *8 ctarg(ntarget,dim),ztarg
      real *8 pot(nd,dim,*)
      real *8 pre(nd,*)
      real *8 grad(nd,dim,dim,*)
c
        
      ns = iend - istart + 1
      ntarg = jend-jstart+1

      iffast = 1

      if (iffast.eq.1) goto 1200

      if ((ifstoklet.eq.1).and.(ifstrslet.eq.0)) then
         if(ifppreg.eq.1) then
            if (ikernel.eq.3.and.dim.eq.2) then
               call st2d_near_kernel_directcp(nd,dim,rsc,cen,bsizeinv,
     1             d2max,source(1,istart),ns,stoklet(1,1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),
     3             ntarg,n_diag,coefs_diag,n_offd,coefs_offd,
     4             pot(1,1,jstart))
            endif

            if (ikernel.eq.3.and.dim.eq.3) then
               call st3d_near_kernel_directcp(nd,dim,rsc,cen,bsizeinv,
     1             d2max,source(1,istart),ns,stoklet(1,1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     3             ntarg,n_diag,coefs_diag,n_offd,coefs_offd,
     4             pot(1,1,jstart))
            endif
         endif
      endif

      return
 1200 continue

      if (dim.eq.3) then
         call stokes_near_kernel_directcp_fast(nd,dim,ndigits,
     1       rsc,cen,bsizeinv,d2min,d2max,source(1,istart),ns,
     2       stoklet(1,1,istart),
     3       ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     4       ntarg,pot(1,1,jstart))
      elseif (dim.eq.2) then
         call stokes_near_kernel_directcp_fast(nd,dim,ndigits,
     1       rsc,cen,bsizeinv,d2min,d2max,source(1,istart),ns,
     2       stoklet(1,1,istart),
     3       ctarg(jstart,1),ctarg(jstart,2),ztarg,
     4       ntarg,pot(1,1,jstart))
      endif
      
      return
      end
c
c
c
c
c
c
c
c------------------------------------------------------------------     
      subroutine stokes_kernel_direct(nd,dim,ikernel,
     1    thresh,istart,iend,source,
     1    ifstoklet,stoklet,ifstrslet,strslet,strsvec,
     2    jstart,jend,targ,ifppreg,pot,pre,grad)
      implicit none
c
      integer nd
      integer dim,ikernel
      integer istart,iend,jstart,jend,ns,ntarg
      integer ifstoklet,ifstrslet
      integer ifppreg
      integer i,j,k
c
      real *8 source(dim,*)
      real *8 stoklet(nd,dim,*),strslet(nd,dim,*),strsvec(nd,dim,*)
      real *8 targ(dim,*),thresh
      real *8 pot(nd,dim,*)
      real *8 pre(nd,*)
      real *8 grad(nd,dim,dim,*)
c
        
      ns = iend - istart + 1
      ntarg = jend-jstart+1

      if(ifstoklet.eq.1.and.ifstrslet.eq.0) then
         if(ifppreg.eq.1) then
            if (ikernel.eq.3.and.dim.eq.2) then
               call st2ddirectcp(dim,nd,source(1,istart),
     1             stoklet(1,1,istart),ns,targ(1,jstart),ntarg,
     2             pot(1,1,jstart),thresh)
            endif

            if (ikernel.eq.3.and.dim.eq.3) then
               call st3ddirectcp(dim,nd,source(1,istart),
     1             stoklet(1,1,istart),ns,targ(1,jstart),ntarg,
     2             pot(1,1,jstart),thresh)
            endif
         endif

      endif
c
      return
      end
c
c
c
c
c
c
c------------------------------------------------------------------    
      subroutine pdmk_mpmaxalloc(nd,dim,npw,laddr,
     1    nlevels,ifpwexpform,ifpwexpeval,lmptotmax)
c     This subroutine determines the size of the array
c     to be allocated for multipole/local expansions
c
c     Input arguments
c     nd          in: integer
c                 number of expansions
c
c     dim         in: integer
c                 dimension of the space
c
c     laddr       in: Integer(2,0:nlevels)
c                 indexing array providing access to boxes at each
c                 level
c
c     nlevels     in: Integer
c                 Total numner of levels
c     
c     npwlevel    in: Integer
c                 cutoff level where the plane wave expansion is valid
c
c     npw         in: Integer
c                 Number of terms in the plane wave expansion
c
c
c------------------------------------------------------------------
c     Output arguments
c     lmptotmax      out: Integer
c                 Maximal length of expansions array required
c------------------------------------------------------------------

      implicit none
      integer dim
      integer nlevels,npwlevel,npw,nd,npw2
      integer laddr(2,0:nlevels), ifpwexpform(*),ifpwexpeval(*)
      integer *8 lmptot(0:nlevels)
      integer *8 lmptotmax,istart,nn,itmp,itmp2
      integer ibox,i
c
      npw2 = (npw+1)/2
      nn = npw**(dim-1)*npw2
c     the factor 2 is the (complex *16)/(real *8) ratio
      nn = nn*2*nd

      do i = 0,nlevels
         istart = 1
         itmp=0
         do ibox = laddr(1,i),laddr(2,i)
c          calculate memory for the multipole PW expansions
           if (ifpwexpform(ibox).eq.1) then
              itmp = itmp+1
           endif
         enddo
c
         do ibox = laddr(1,i),laddr(2,i)
c          calculate memory for the local PW expansions
           if (ifpwexpeval(ibox).eq.1) then
              itmp = itmp+1
           endif
         enddo
         istart = istart + itmp*nn
         lmptot(i) = istart
      enddo

      lmptotmax = lmptot(0)
      do i=1,nlevels
         if (lmptot(i).gt.lmptotmax) lmptotmax=lmptot(i)
      enddo
      
      return
      end
c
c
c
c
c
c
c------------------------------------------------------------------    
      subroutine pdmk_mpalloc_stokes(nd,dim,laddr,iaddr,
     1    nlevels,npwlevel,ifpwexpform,ifpwexpeval,lmptot,npw)
c     This subroutine determines the size of the array
c     to be allocated for multipole/local expansions
c
c     Input arguments
c     nd          in: integer
c                 number of expansions
c
c     dim         in: integer
c                 dimension of the space
c
c     laddr       in: Integer(2,0:nlevels)
c                 indexing array providing access to boxes at each
c                 level
c
c     nlevels     in: Integer
c                 Total numner of levels
c     
c     npwlevel    in: Integer
c                 cutoff level where the plane wave expansion is valid
c
c     npw         in: Integer
c                 Number of terms in the plane wave expansion
c
c
c------------------------------------------------------------------
c     Output arguments
c     iaddr: (2,nboxes): pointer in rmlexp where multipole
c                      and local expansions for each
c                      box is stored.
c                      iaddr(1,ibox) is the
c                      starting index in rmlexp for the 
c                      multipole PW expansion of ibox;
c                      iaddr(2,ibox) is the
c                      starting index in rmlexp
c                      for the local PW expansion of ibox.
c     lmptot      out: Integer
c                 Total length of expansions array required
c------------------------------------------------------------------

      implicit none
      integer dim
      integer nlevels,npwlevel,npw,nd,npw2
      integer laddr(2,0:nlevels), ifpwexpform(*),ifpwexpeval(*)
      integer *8 iaddr(2,*)
      integer *8 lmptot,istart,nn,itmp,itmp2
      integer ibox,i,nlevstart,istarts,iends,npts
c
      istart = 1
      if (npwlevel .gt. nlevels) then
         lmptot=0
         return
      endif
      
      nlevstart = 0
      if (npwlevel .ge. 0) nlevstart = npwlevel

      npw2 = (npw+1)/2
      nn = npw**(dim-1)*npw2
c     the factor 2 is the (complex *16)/(real *8) ratio
      nn = nn*2*nd

      itmp=0
      do i = nlevstart,nlevstart
         do ibox = laddr(1,i),laddr(2,i)
c          Allocate memory for the multipole PW expansions
           if (ifpwexpform(ibox).eq.1) then
              iaddr(1,ibox) = istart + itmp*nn
              itmp = itmp+1
           endif
         enddo
         istart = istart + itmp*nn
      enddo
c
      itmp2=0
      do i = nlevstart,nlevstart
         do ibox = laddr(1,i),laddr(2,i)
c          Allocate memory for the local PW expansions
           if (ifpwexpeval(ibox).eq.1) then
              iaddr(2,ibox) = istart + itmp2*nn
              itmp2 = itmp2+1
           endif
         enddo
         istart = istart + itmp2*nn
      enddo

      lmptot = istart

      return
      end
c
c
c
c
c------------------------------------------------------------------    
      subroutine pdmk_coefspalloc_stokes(nd,dim,laddr,iaddr,
     1    nlevels,ifleafbox,ifpwexpform,ifpwexpeval,lmptot,norder)
c     This subroutine determines the size of the array
c     to be allocated for multipole/local expansions
c
c     Input arguments
c     nd          in: integer
c                 number of expansions
c
c     dim         in: integer
c                 dimension of the space
c
c     laddr       in: Integer(2,0:nlevels)
c                 indexing array providing access to boxes at each
c                 level
c
c     nlevels     in: Integer
c                 Total numner of levels
c     
c     npwlevel    in: Integer
c                 cutoff level where the plane wave expansion is valid
c
c     norder      in: Integer
c                 Number of terms in the polynomial approximation
c
c
c------------------------------------------------------------------
c     Output arguments
c     iaddr: (2,nboxes): pointer in rmlexp where multipole
c                      and local expansions for each
c                      box is stored.
c                      iaddr(1,ibox) is the
c                      starting index in rmlexp for the 
c                      multipole PW expansion of ibox;
c                      iaddr(2,ibox) is the
c                      starting index in rmlexp
c                      for the local PW expansion of ibox.
c     lmptot      out: Integer
c                 Total length of expansions array required
c------------------------------------------------------------------

      implicit none
      integer dim
      integer nlevels,npw,nd,norder
      integer *8 iaddr(2,*)
      integer laddr(2,0:nlevels), ifpwexpform(*),ifpwexpeval(*)
      integer ifleafbox(*)
      integer *8 lmptot,istart,nn,itmp,itmp2
      integer ibox,i,istarts,iends,npts
c
      istart = 1
      
      nn = norder**dim
      nn = nn*nd

      itmp=0
      do i = 0,nlevels
         do ibox = laddr(1,i),laddr(2,i)
c     Allocate memory for the multipole PW expansion         
c
           if (ifpwexpform(ibox).eq.1) then
              iaddr(1,ibox) = istart + itmp*nn
              itmp = itmp+1
           endif
         enddo
         istart = istart + itmp*nn
      enddo
c
      itmp2=0
      do i = 0,nlevels
         do ibox = laddr(1,i),laddr(2,i)
c     Allocate memory for the local PW expansion         
c
           if (ifpwexpeval(ibox).eq.1) then
              iaddr(2,ibox) = istart + itmp2*nn
              itmp2 = itmp2+1
           endif
         enddo
         istart = istart + itmp2*nn
      enddo

      lmptot = istart

      return
      end
c
c
c
c

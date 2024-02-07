c     
c    $Date$
c    $Revision$

      subroutine boxfgt_md(nd,ndim,ndelta,deltas,dwhts,
     1    npwlevel,eps,ipoly,norder,npbox,nboxes,nlevels,ltree,
     2    itree,iptr,centers,boxsize,fvals,flvals,fl2vals,
     3    porder,ncbox,proxycharge,proxypotential,
     4    p2ctransmat,ifpgh,pot,grad,hess,timeinfo)
c     This code computes the Gauss transform for a collection of functions
c     defined on a tensor product grid of each leaf node in an adaptive tree
c 
c     Difference between this code and boxfgt.f: boxfgt.f takes only one delta
c     as the input, while this code deals with several deltas corresponding to 
c     the same cutoff level! This is an internal routine used in the kernel-independent
c     boxcode!
c
c     pot(x) = \sum_{k=1}^{ndelta} dwhts(k) \int_B G(x,y;deltas(k)) f(y) dy,
c     where B is a cube in R^ndim
c
c 
c     input
c     nd - integer
c            number of right hand sides
c     ndim - integer
c            dimension of the underlying space
c     ndelta - integer
c            number of different Gaussian variances 
c     deltas - double precision
c            value of Gaussian variances, sorted in descending order! 
c     dwhts - double precision
c            weights of different Gaussian kernels
c     npwlevel - the cutoff level for ALL input deltas!
c     eps - double precision
c            precision requested
c     ipoly - integer
c            0: Legendre polynomials
c            1: Chebyshev polynomials
c     norder - integer
c           order of expansions for input function value array
c     npbox - integer
c           number of points per box where potential is to be dumped = (norder**ndim)
c     fvals - double precision (nd,npbox,nboxes)
c           function values tabulated on a tensor grid in each leaf node
c     flvals - double precision (nd,npbox,nboxes)
c           Laplacian of the function values tabulated on a tensor grid in each leaf node
c     fl2vals - double precision (nd,npbox,nboxes)
c           BiLaplacian of the function values tabulated on a tensor grid in each leaf node
c     ifpgh   : flag for computing pot/grad/hess
c                   ifpgh = 1, only potential is computed
c                   ifpgh = 2, potential and gradient are computed
c                   ifpgh = 3, potential, gradient, and hessian 
c                   are computed
c
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
c     centers - double precision (ndim,nboxes)
c           xyz coordintes of boxes in the tree structure
c     boxsize - double precision (0:nlevels)
c           size of boxes at each of the levels
c
c     output:
c     pot - double precision (nd,npbox,nboxes)
c            volume potential on the tree structure (note that 
c            the potential is non-zero only in the leaf boxes of the new tree
c     grad - double precision (nd,ndim,npbox,nboxes)
c            gradient of the volume potential on the tree structure 
c     hess - double precision (nd,ndim*(ndim+1)/2,npbox,nboxes)
c            hessian of the volume potential on the tree structure 
c            in 2d, the order is xx, xy, yy 
c            in 3d, the order is xx, yy, zz, xy, xz, yz
c
cccc      implicit real *8 (a-h,o-z)
      implicit none
      real *8 eps
      integer nboxes,nlevels,ntarg,ifpgh
      integer iptr(8),ltree
      integer itree(ltree),norder,npbox,porder,ncbox
      integer nd,ndim,ipoly
      integer ifproxypoteval(nboxes)
      real *8 deltas(ndelta),dwhts(ndelta)
      real *8 fvals(nd,npbox,nboxes)
      real *8 flvals(nd,npbox,nboxes)
      real *8 fl2vals(nd,npbox,nboxes)
cccc      real *8 fl3vals(nd,npbox,nboxes)
      real *8 proxycharge(ncbox,nd,nboxes)
      real *8 proxypotential(ncbox,nd,nboxes)
      real *8 p2ctransmat(porder,porder,ndim,2**ndim)
      

      real *8 pot(nd,npbox,nboxes)
      real *8 grad(nd,ndim,npbox,*)
      real *8 hess(nd,ndim*(ndim+1)/2,npbox,*)

      real *8 centers(ndim,nboxes)
      real *8 boxsize(0:nlevels)
      real *8 timeinfo(*)
c
cc     additional fgt variables
c
      integer *8, allocatable :: iaddr(:,:)
c
cc      temporary variables
c
      integer npwlevel
      integer i,ilev,lmptmp,idim
      integer ndelta
      integer npw,nadd,ifprint,ier
      real *8 dcutoff
      real *8 omp_get_wtime
      real *8 time1,time2,pi,done,pmax,bs0,bsize,pweps

      ifprint = 1

      if(ifprint.eq.1) call prinf(' npwlevel =*',npwlevel,1)
      if(ifprint.eq.1) call prinf(' nlevels =*',nlevels,1)
c
c
c     compute the length of plane wave expansion
      npw=0
      if (npwlevel.le.nlevels) then
         call bdmk_pwterms(eps,npwlevel,npw)
      endif
      if (ifprint.eq.1) call prinf(' npw =*',npw,1)
c       
c     Multipole and local expansions will be held in workspace
c     in locations pointed to by array iaddr(2,nboxes).
c
c     iiaddr is pointer to iaddr array, itself contained in workspace.
c
c       ... allocate iaddr and temporary arrays
c
      allocate(iaddr(2,nboxes))
c
c     iaddr is pointer for workspace need by various expansions.
c
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      call bfgt_md_main(nd,ndim,ndelta,deltas,dwhts,npwlevel,
     1    eps,ipoly,norder,npbox,nboxes,nlevels,ltree,itree,iptr,
     2    centers,boxsize,fvals,flvals,fl2vals,porder,ncbox,
     3    proxycharge,proxypotential,p2ctransmat,
     4    iaddr,npw,ifpgh,pot,grad,hess,timeinfo)
      
      call cpu_time(time2)
C$        time2=omp_get_wtime()
      if( ifprint .eq. 1 ) call prin2('time in box fgt main=*',
     1   time2-time1,1)

      return
      end
c
c
c
c
      subroutine bfgt_md_main(nd,ndim,ndelta,deltas,dwhts,npwlevel,
     1    eps,ipoly,norder,npbox,nboxes,nlevels,ltree,itree,iptr,
     2    centers,boxsize,fvals,flvals,fl2vals,porder,ncbox,
     3    proxycharge,proxypotential,p2ctransmat,
     4    iaddr,npw,ifpgh,pot,grad,hess,timeinfo)
c
c
c     This code compute the Gauss transform for a collection of functions
c     defined on a tensor product grid of each leaf node in an adaptive tree
c 
c     input
c     nd - integer
c          number of right hand sides
c     ndim - integer
c            dimension of the underlying space
c     ndelta - integer
c            number of different Gaussian variances 
c     deltas - double precision
c            value of Gaussian variances, sorted in descending order! 
c     dwhts - double precision
c            weights of different Gaussian kernels
c     npwlevel - the cutoff level for ALL input deltas!
c
c     eps - double precision
c            precision requested
c     ipoly - integer
c            0: Legendre polynomials
c            1: Chebyshev polynomials
c     norder - integer
c           order of expansions for input function value array
c     npbox - integer
c            number of points per box where potential is to be dumped = (norder**ndim)
c     fvals - double precision (nd,npbox,nboxes)
c           function values tabulated on a tensor grid in each leaf node
c     flvals - double precision (nd,npbox,nboxes)
c           Laplacian of the function values tabulated on a tensor grid in each leaf node
c     fl2vals - double precision (nd,npbox,nboxes)
c           BiLaplacian of the function values tabulated on a tensor grid in each leaf node
c     iaddr - (2,nboxes): pointer in rmlexp where multipole
c                      and local expansions for each
c                      box is stored
c                      iaddr(1,ibox) is the
c                      starting index in rmlexp for the 
c                      multipole PW expansion of ibox
c                      and iaddr(2,ibox) is the
c                      starting index in rmlexp
c                      for the local PW expansion of ibox
c     rmlexp - double precision, stores multipole and local PW expansions
c                  for each box below the cutoff level npwlevel
c
c     npw  - integer,  length of planewave expansions
c      
c     ifpgh   : flag for computing pot/grad/hess
c                   ifpgh = 1, only potential is computed
c                   ifpgh = 2, potential and gradient are computed
c                   ifpgh = 3, potential, gradient, and hessian 
c                   are computed
c
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
c     centers - double precision (ndim,nboxes)
c           xyz coordintes of boxes in the tree structure
c     boxsize - double precision (0:nlevels)
c           size of boxes at each of the levels
c
c     output:
c     pot - double precision (nd,npbox,nboxes)
c            volume potential on the tree structure (note that 
c            the potential is non-zero only in the leaf boxes of the new tree
c     grad - double precision (nd,ndim,npbox,nboxes)
c            gradient of the volume potential on the tree structure 
c     hess - double precision (nd,ndim*(ndim+1)/2,npbox,nboxes)
c            hessian of the volume potential on the tree structure 
c            in 2d, the order is xx, xy, yy 
c            in 3d, the order is xx, yy, zz, xy, xz, yz
c
c      implicit real *8 (a-h,o-z)
      implicit none
      integer nd,ndim,ndelta,norder
      real *8 deltas(ndelta),dwhts(ndelta),eps
      integer nboxes,nlevels,ntarg,porder,ncbox
      integer iptr(8),ltree
      integer itree(ltree),npbox
      real *8 fvals(nd,npbox,nboxes)
      real *8 flvals(nd,npbox,nboxes)
      real *8 fl2vals(nd,npbox,nboxes)
cccc      real *8 fl3vals(nd,npbox,nboxes)

      real *8 proxycharge(ncbox,nd,nboxes)
      real *8 proxypotential(ncbox,nd,nboxes)
      real *8 p2ctransmat(porder,porder,ndim,2**ndim)

      real *8 pot(nd,npbox,nboxes)
      real *8 grad(nd,ndim,npbox,*)
      real *8 hess(nd,ndim*(ndim+1)/2,npbox,*)

      real *8 boxsize(0:nlevels),centers(ndim,nboxes)
      integer *8 iaddr(2,nboxes)
      real *8, allocatable :: rmlexp(:)
      real *8 pmax
      real *8 timeinfo(*)

      integer *8 lmptot,i8
      
c
cc        temporary variables
c
c     direction interaction list
      integer, allocatable :: nlist1(:),list1(:,:)
c     plane wave interaction list
      integer, allocatable :: nlistpw(:), listpw(:,:)
c     box flag array
      integer, allocatable :: ifpwexp(:)

      integer ndirect,itype
      integer ixyz(ndim)
c     
      integer idelta(0:nlevels)

      real *8, allocatable :: xq(:),wts(:),umat(:,:),vmat(:,:)
      real *8, allocatable :: xq1(:),wts1(:),umat1(:,:),vmat1(:,:)
      real *8, allocatable :: umat_nd(:,:,:)

      real *8 ws(100),ts(100),ws0(100)

c     plane wave m2l translation operators
      complex *16, allocatable :: wpwshift(:,:)

      
c     planewave expansion weights
      real *8, allocatable :: wpwexp(:)
      
c     for asymptotic calculations
      real *8, allocatable :: fcoefs(:,:,:)
      real *8, allocatable :: gvals(:,:,:,:)
      real *8, allocatable :: hvals(:,:,:,:)

c     1d direct evaluation tables
      real *8, allocatable :: tab_loc(:,:,:,:,:)
      real *8, allocatable :: tabx_loc(:,:,:,:,:)
      real *8, allocatable :: tabxx_loc(:,:,:,:,:)
      integer, allocatable :: ind_loc(:,:,:,:,:)

      complex *16, allocatable :: tab_coefs2pw(:,:)
      complex *16, allocatable :: tab_pw2coefs(:,:)
      
      integer, allocatable :: isgn(:,:)
      
      complex *16 ima/(0,1)/
      integer nthd,ithd
      integer omp_get_max_threads,omp_get_thread_num
      integer ifprint,mc,mnbors,nhess,ncutoff,npwlevel,i,ilevel,nleafbox
      integer npw,ipoly,norder2,ibox,jbox,ilev,klev,k,nloctab,nloctab2,j
      integer ind,iind,ngrandchild,ier,mnlistpw,nexp,nmax,nl1,jlev,id
      integer kdelta,iperiod,isep,mnlist1,mrefinelev,nnodes
      integer ifpgh,nlevend,idad,nasym,nchild,ipoly0
      real *8 done,pi,bs0,bsize,time1,time2,hpw,delta,xmin
      real *8 d,bs,asymerr
      
      ifprint = 1

      mc=2**ndim
      allocate(isgn(ndim,mc))
      call get_child_box_sign(ndim,isgn)

      done = 1
      pi = atan(done)*4

      bs0 = boxsize(0)
      mc = 2**ndim
      mnbors=3**ndim
      nhess=ndim*(ndim+1)/2

      ncutoff=max(npwlevel,0)
      bsize = boxsize(ncutoff)
      
c
c
      if(ifprint .ge. 1)
     $     call prinf('=== STEP 0 (precomputation) =====*',i,0)
      call cpu_time(time1)
C$    time1=omp_get_wtime()
c      
c     get planewave nodes and weights
      call get_pwnodes_md(eps,nlevels,npwlevel,npw,ws,ts)
      do i=1,npw
         ws0(i)=1.0d0
      enddo
      
      if(ifprint.eq.1) call prinf('npw=*',npw,1)
      allocate(xq(norder),umat(norder,norder),
     1    vmat(norder,norder),wts(norder))
      allocate(xq1(porder),umat1(porder,porder),
     1    vmat1(porder,porder),wts1(porder))
     
      itype = 2
      if (ipoly.eq.0) call legeexps(itype,norder,xq,umat,vmat,wts)
      if (ipoly.eq.1) call chebexps(itype,norder,xq,umat,vmat,wts)
      call legeexps(itype,porder,xq1,umat1,vmat1,wts1)

c     tables converting tensor product polynomial expansion coefficients of 
c     the charges to planewave expansion coefficients - on the source side
      allocate(tab_coefs2pw(npw,porder))
c     tables converting planewave expansions to tensor product polynomial
c     expansion coefficients of the potentials - on the target side
      allocate(tab_pw2coefs(npw,porder))

      ipoly0=0
      call dmk_mk_coefs_pw_conversion_tables(ipoly0,porder,npw,
     1    ts,xq1,hpw,bsize,tab_coefs2pw,tab_pw2coefs)
         
c      
      allocate(umat_nd(norder,norder,ndim))
      norder2=norder*norder
      do i=1,ndim
         call dcopy_f77(norder2,umat,1,umat_nd(1,1,i),1)
      enddo

      nlevend=nlevels
      if (npwlevel.lt.nlevels) nlevend=npwlevel

c     compute the number of leaf boxes
      nleafbox = 0
      do ilevel=1,nlevels
        do ibox=itree(2*ilevel+1),itree(2*ilevel+2)
          if(itree(iptr(4)+ibox-1).eq.0) nleafbox = nleafbox+1
        enddo
      enddo
      if(ifprint.eq.1) call prinf('nleafbox=*',nleafbox,1)
      
c     estimate number of direct evaluation boxes
      ndirect=0
      do ilev = 0,nlevend
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
            nchild = itree(iptr(4)+ibox-1)
c           Check if the current box is a leaf box            
            if(nchild.eq.0) then
               ndirect = ndirect+1
            endif
         enddo
      enddo
      if(ifprint.eq.1) call 
     1    prinf('number of direct evaluation source boxes=*',ndirect,1)

      do i=1,20
         timeinfo(i)=0
      enddo

c     find the number of deltas that can be handled by asymptotic expansions at each level
      nasym=3
      do ilev=0,nlevels
         idelta(ilev)=0
      enddo

      do ilev=0,nlevels
         do k=ndelta,1,-1
            delta=deltas(k)
c           empirical error formula for asymptotic expansions
            asymerr=(delta*4/boxsize(ilev)**2)**nasym
            if (asymerr .le. eps) then
               idelta(ilev)=idelta(ilev)+1
            endif
         enddo
      enddo
      call prinf('ndelta=*',ndelta,1)
      call prinf('idelta=*',idelta,nlevels+1)
c
c
c       compute list info
c
c     check whether we need to create and evaluate planewave expansions 
c     for boxes
      allocate(ifpwexp(nboxes))
      iperiod=0
      call bdmk_find_pwexp_boxes(ndim,npwlevel,nboxes,
     1    nlevels,ltree,itree,iptr,iperiod,ifpwexp)

      isep = 1
      call compute_mnlist1(ndim,nboxes,nlevels,itree(iptr(1)),centers,
     1    boxsize,itree(iptr(3)),itree(iptr(4)),itree(iptr(5)),
     2    isep,itree(iptr(6)),itree(iptr(7)),iperiod,mnlist1)
      
      allocate(list1(mnlist1,nboxes),nlist1(nboxes))

c     modified list1 for direct evaluation
c     list1 of a childless source box ibox at ilev<=npwlevel
c     contains all childless target boxes that are neighbors of ibox
c     at or above npwlevel
      call bdmk_compute_modified_list1(ndim,npwlevel,ifpwexp,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,iperiod,
     2    mnlist1,nlist1,list1)

c     compute the tables converting Legendre polynomial expansion to potential
c     values, used in direct evaluation
c
c     no tree refinement, the tree is the usual level-restricted tree
      mrefinelev=0
      nloctab=2**(mrefinelev+1)*(mrefinelev+3)

      nloctab2=2*nloctab+1
      allocate(  tab_loc(norder,norder,nloctab2,ndelta,0:nlevels))
      allocate( tabx_loc(norder,norder,nloctab2,ndelta,0:nlevels))
      allocate(tabxx_loc(norder,norder,nloctab2,ndelta,0:nlevels))
      allocate(ind_loc(2,norder+1,nloctab2,ndelta,0:nlevels))
      nnodes=50
      do ilev = 0,nlevels
      do id=1,ndelta
         delta=deltas(id)
         call mk_loctab_all(eps,ipoly,norder,nnodes,delta,
     1       boxsize(ilev),mrefinelev,nloctab,tab_loc(1,1,1,id,ilev),
     2       tabx_loc(1,1,1,id,ilev),tabxx_loc(1,1,1,id,ilev),
     3       ind_loc(1,1,1,id,ilev))
      enddo
      enddo

c     direct evaluation if the cutoff level is >= the maximum level 
      if (npwlevel .ge. nlevels) goto 1800
cccc      if (npwlevel .ge. nlevels) goto 3000


c     compute the planewave expansion weights for all deltas
      nexp=(npw+1)/2
      do i=1,ndim-1
         nexp = nexp*npw
      enddo

      nthd = 1
C$    nthd = omp_get_max_threads()

      allocate(wpwexp(nexp))
      call mk_kernel_Fourier_transform(ndim,ndelta,deltas,
     1    dwhts,npw,ws,ts,nexp,wpwexp)

c     diagonal multipole to local plane wave translation matrices
      nmax = 1
      allocate(wpwshift(nexp,(2*nmax+1)**ndim))
c     xmin is used in shiftpw subroutines to
c     determine the right translation matrices
      xmin  = boxsize(ncutoff)
      call mk_pw_translation_matrices(ndim,xmin,npw,ts,nmax,
     1    wpwshift)
c
c     compute list info
c
      call dmk_compute_mnlistpw(ndim,nboxes,nlevels,ltree,itree,
     1    iptr,centers,boxsize,mnlistpw)
      allocate(nlistpw(nboxes),listpw(mnlistpw,nboxes))
c     listpw contains source boxes in the pw interaction
      call bdmk_compute_listpw(ndim,npwlevel,nboxes,nlevels,
     1    ltree,itree,iptr,centers,boxsize,itree(iptr(1)),
     3    mnlistpw,nlistpw,listpw)
      
c
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      timeinfo(1) = timeinfo(1) + time2-time1
      
      if (ifprint.eq.1) 
     1    call prin2('fgt precomputation time =*', time2-time1,1)
      
      if (ifprint.eq.1)
     1    call prinf('laddr=*',itree(iptr(1)),2*(nlevels+1))


c
c     allocate memory need by multipole, local expansions at all levels
c
      call bdmk_mpalloc_md(nd,ndim,ifpwexp,itree,iaddr,
     1    nboxes,nlevels,itree,iptr,npwlevel,lmptot,npw)
      
      if(ifprint .eq. 1) call prinf_long(' lmptot is *',lmptot,1)
      
      call cpu_time(time1)
C$      time1=omp_get_wtime()
      allocate(rmlexp(lmptot),stat=ier)
      if(ier.ne.0) then
         print *, "Cannot allocate workspace for plane wave expansions"
         print *, "lmptot=", lmptot
         ier = 4
         return
      endif
      
      call cpu_time(time2)
C$        time2=omp_get_wtime()
      if( ifprint .eq. 1 ) call prin2('time in allocating rmlexp=*',
     1   time2-time1,1)
c



      



      
c
c        step 1: convert function values to planewave expansions
c
    
      if(ifprint.eq.1) 
     1   call prinf("=== STEP 1 (values -> mp pwexps) ===*",i,0)
      
      call cpu_time(time1)
C$    time1=omp_get_wtime()

c     form planewave expansions at the cutoff level
      do 1100 ilev =  ncutoff,ncutoff
C$OMP PARALLEL DO DEFAULT (SHARED)
C$OMP$PRIVATE(ibox,nchild)
C$OMP$SCHEDULE(DYNAMIC)
        do ibox=itree(2*ilev+1),itree(2*ilev+2)
          if (ifpwexp(ibox).eq.1) then
c              form the pw expansion
            call dmk_proxycharge2pw(ndim,nd,porder,
     1          proxycharge(1,1,ibox),npw,tab_coefs2pw,
     3          rmlexp(iaddr(1,ibox)))
c     copy the multipole PW exp into local PW exp
c     for self interaction
            call dmk_copy_pwexp(nd,nexp,rmlexp(iaddr(1,ibox)),
     1          rmlexp(iaddr(2,ibox)))
          endif
        enddo
C$OMP END PARALLEL DO
 1100 continue

      call cpu_time(time2)
C$       time2 = omp_get_wtime()
      timeinfo(2) = timeinfo(2) + time2-time1









      
      if(ifprint.ge.1)
     1    call prinf('=== STEP 2 (mp to loc) ===*',i,0)
c      ... step 2, convert multipole expansions into local
c       expansions

      call cpu_time(time1)
C$    time1=omp_get_wtime()
      
      do 1300 ilev = ncutoff,ncutoff
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,jbox,j,ind)
C$OMP$SCHEDULE(DYNAMIC)
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
            if (ifpwexp(ibox).eq.1) then
c              shift PW expansions
               do j=1,nlistpw(ibox)
                  jbox=listpw(j,ibox)
                  call dmk_find_pwshift_ind(ndim,iperiod,
     1                centers(1,ibox),centers(1,jbox),bs0,xmin,nmax,ind)
                  call dmk_shiftpw(nd,nexp,rmlexp(iaddr(1,jbox)),
     1                rmlexp(iaddr(2,ibox)),wpwshift(1,ind))
               enddo
c     multiply the Fourier transform of the kernel
               call dmk_multiply_kernelFT(nd,nexp,
     1             rmlexp(iaddr(2,ibox)),wpwexp)
            endif
        enddo
C$OMP END PARALLEL DO        
 1300 continue
c      
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      timeinfo(3) = timeinfo(3) + time2-time1










      
      if(ifprint.ge.1)
     1    call prinf('=== STEP 3 (eval loc pwexps) ===*',i,0)

c     ... step 3, convert local plane wave expansions to proxy potential
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      do 1500 ilev = ncutoff,ncutoff
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,nchild)
C$OMP$SCHEDULE(DYNAMIC)
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
            nchild = itree(iptr(4)+ibox-1)
            if (ifpwexp(ibox).eq.1) then
               call dmk_pw2proxypot(ndim,nd,porder,npw,
     1             rmlexp(iaddr(2,ibox)),tab_pw2coefs,
     3             proxypotential(1,1,ibox))
               if ((nchild.gt.0).and.(npwlevel.ge.0)) then
                  do j=1,nchild
                     jbox = itree(iptr(5) + (ibox-1)*mc+j-1)
c                    translate tensor product polynomial from parent to child
                     call tens_prod_trans_add(ndim,nd,porder,
     1                   proxypotential(1,1,ibox),porder,
     2                   proxypotential(1,1,jbox),
     3                   p2ctransmat(1,1,1,j))
                  enddo
               endif   
            endif
         enddo
C$OMP END PARALLEL DO        
 1500 continue

      
      deallocate(rmlexp)
      
      call cpu_time(time2)
C$    time2 = omp_get_wtime()      
      timeinfo(4) = timeinfo(4) + time2 - time1




      




      
 1800 continue

      if(ifprint .ge. 1)
     1     call prinf('=== STEP 4 (direct interactions) =====*',i,0)
c
cc
      call cpu_time(time1)
C$    time1=omp_get_wtime()  
      do 2000 ilev = 0,nlevend
         kdelta=ndelta-idelta(ilev)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,jbox,j,nl1,ixyz,jlev,bs,id)
C$OMP$SCHEDULE(DYNAMIC)  
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
c        ibox is the target box            
           nl1 = nlist1(ibox)
           do j=1,nl1
cccc       jbox is the source box
             jbox = list1(j,ibox)
             jlev = itree(iptr(2)+jbox-1)
             bs = boxsize(jlev)
             call bdmk_find_loctab_ind(ndim,iperiod,
     1           centers(1,ibox),centers(1,jbox),bs,bs0,mrefinelev,ixyz)
cccc  call prinf('ixyz=*',ixyz,ndim)
             do id=1,kdelta
                call bdmk_tens_prod_to_potloc(ndim,nd,norder,
     1              dwhts(id),fvals(1,1,jbox),pot(1,1,ibox),
     2              nloctab,tab_loc(1,1,1,id,ilev),
     3              ind_loc(1,1,1,id,ilev),ixyz)
             enddo
           enddo
         enddo
C$OMP END PARALLEL DO         
 2000 continue
c
      call cpu_time(time2)
C$    time2=omp_get_wtime()  
      timeinfo(5) = timeinfo(5) + time2-time1







      
 3000 continue
      if(ifprint .ge. 1)
     $ call prinf('=== STEP 5 (direct asymptotic interaction) ===*',i,0)
      call cpu_time(time1)
C$    time1=omp_get_wtime()

c     evaluate potential
      call treedata_eval_pot_nd_asym_fast(ndim,nd,ndelta,deltas,
     1    dwhts,idelta,ipoly,nasym,nlevels,itree,iptr,boxsize,norder,
     2    fvals,flvals,fl2vals,pot)
c     2    fvals,flvals,fl2vals,fl3vals,pot)

      call cpu_time(time2)
C$    time2=omp_get_wtime()  
      timeinfo(6) = timeinfo(6) + time2-time1


 4000 continue
      
      if(ifprint.eq.1) then 
         call prin2('timeinfo=*',timeinfo,6)
         d= 0
         do i = 1,6
            d = d + timeinfo(i)
         enddo
         call prin2('time on tensor grid=*',d,1)
         call prin2('tensor grid speed in pps=*',
     1       (nleafbox*npbox*nd+0.0d0)/d,1)
      endif
      
      return
      end
c
c
c
c
c
c------------------------------------------------------------------    
      subroutine bdmk_mpalloc_md(nd,ndim,ifpwexp,laddr,iaddr,
     1    nboxes,nlevels,itree,iptr,npwlevel,lmptot,npw)
c     This subroutine determines the size of the array
c     to be allocated for multipole/local expansions
c
c     Note: no memory is allocated for leaf nodes below the cutoff level!
c     This will save the memory by a factor of up to 8 in many cases.
c
c
c     Input arguments
c     nd          in: integer
c                 number of expansions
c
c     ndim        in: integer
c                 dimension of the underlying space
c
c     laddr       in: Integer(2,0:nlevels)
c                 indexing array providing access to boxes at each
c                 level
c
c     nlevels     in: Integer
c                 Total numner of levels
c     
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
c     npwlevel    in: Integer
c                 cutoff level where the plane wave expansion is
c                 valid at or below ilev = npwlevel
c
c     npw         in: Integer
c                 Number of terms in the plane wave expansion
c
c
c------------------------------------------------------------------
c     Output arguments
c     iaddr: (2,nboxes): pointer in rmlexp where multipole
c                      and local expansions for each
c                      box is stored
c                      iaddr(1,ibox) is the
c                      starting index in rmlexp for the 
c                      multipole PW expansion of ibox
c                      and iaddr(2,ibox) is the
c                      starting index in rmlexp
c                      for the local PW expansion of ibox
c     lmptot      out: Integer
c                 Total length of expansions array required
c------------------------------------------------------------------

      implicit none
      integer nd,ndim,nlevels,npwlevel,npw,nboxes
      integer laddr(2,0:nlevels),itree(*),iptr(*),ifpwexp(*)
      integer *8 lmptot
      integer ibox,i,ncutoff,jbox,nchild
      integer *8 iaddr(2,*),istart,nn,itmp
c
      istart = 1
      if (npwlevel .ge. nlevels) then
         lmptot=0
         return
      endif
      
      ncutoff = 0
      if (npwlevel .ge. 0) ncutoff = npwlevel

      nn = (npw+1)/2
      do i=1,ndim-1
         nn = nn*npw
      enddo
      nn = nn*2*nd

      do i = ncutoff,ncutoff
         itmp = 0
         do ibox = laddr(1,i),laddr(2,i)
            if (ifpwexp(ibox).eq.1) then
c              Allocate memory for the multipole PW expansion         
               iaddr(1,ibox) = istart + itmp*2*nn
c              Allocate memory for the local PW expansion at the cutoff level         
               iaddr(2,ibox) = istart + itmp*2*nn + nn
               itmp = itmp+1
            endif
         enddo
         istart = istart + itmp*2*nn
      enddo
c
      lmptot = istart

      return
      end
c----------------------------------------------------------------     
c
c
c
c

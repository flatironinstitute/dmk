ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     $Date$
c     $Revision$
c
c     last modified on 02/04/2024
c
c
c
c
c
c
      
      subroutine pdmk(nd,dim,eps,ikernel,rpars,
     1    iperiod,ns,sources,
     2    ifcharge,charge,ifdipole,rnormal,dipstr,
     3    ifpgh,pot,grad,hess,nt,targ,
     4    ifpghtarg,pottarg,gradtarg,hesstarg,tottimeinfo)
c----------------------------------------------
c   INPUT PARAMETERS:
c   nd            : number of MFMs (same source and target locations, 
c                   different charge, dipole strengths)
c   dim           : dimension of the space
c   eps           : precision requested
c   ikernel       : 0: Yukawa kernel; 1: Laplace kernel; 2: square root Laplace kernel
c   rpars         : parameter array used in the kernel evaluation  
c   iperiod       : 0: free space; 1: periodic - not implemented yet      
c      
c   ns            : number of sources
c   sources(dim,ns) : source locations
c   ifcharge      : flag for including charge interactions
c                   charge interactions included if ifcharge =1
c                   not included otherwise
c   charge(nd,ns) : charge strengths
c   ifdipole      : flag for including dipole interactions
c                   dipole interactions included if ifcharge =1
c                   not included otherwise
c   rnormal(dim,ns) : dipole directions
c   dipstr(nd,ns) : dipole strengths
c   iper          : flag for periodic implmentations. Currently unused
c   ifpgh         : flag for computing pot/grad/hess
c                   ifpgh = 1, only potential is computed
c                   ifpgh = 2, potential and gradient are computed
c                   ifpgh = 3, potential, gradient, and hessian 
c                   are computed
c   nt            : number of targets
c   targ(dim,nt)    : target locations
c   ifpghtarg     : flag for computing pottarg/gradtarg/hesstarg
c                   ifpghtarg = 1, only potential is computed at targets
c                   ifpghtarg = 2, potential and gradient are 
c                   computed at targets
c                   ifpghtarg = 3, potential, gradient, and hessian are 
c                   computed at targets
c
c   OUTPUT PARAMETERS
c   pot(nd,*)       : potential at the source locations
c   grad(nd,dim,*)    : gradients at the source locations
c   hess(nd,dim*(dim+1)/2,*)    : hessian at the source locations
c   pottarg(nd,*)   : potential at the target locations
c   gradtarg(nd,dim,*): gradient at the target locations
c   hesstarg(nd,dim*(dim+1)/2,*): hessian at the target locations
c
c   Note: hessians are in the order xx, yy, zz, xy, xz, yz for 3D
c     and xx, xy, yy for 2D
c      
      implicit none
c
cc      calling sequence variables
c 
      integer nd,dim,ikernel
      real *8 eps,rpars(*)
      integer ns,nt,iperiod
      integer ifcharge,ifdipole
      integer ifpgh,ifpghtarg
      real *8 sources(dim,ns),targ(dim,nt)
      real *8 rnormal(dim,ns)
      real *8 charge(nd,*),dipstr(nd,*)

      real *8 pot(nd,*),grad(nd,dim,*),hess(nd,dim*(dim+1)/2,*)
      real *8 pottarg(nd,*),gradtarg(nd,dim,*)
      real *8 hesstarg(nd,dim*(dim+1)/2,*)

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
      real *8, allocatable :: rnormalsort(:,:)
      real *8, allocatable :: targsort(:,:),ctargsort(:,:)
      real *8, allocatable :: chargesort(:,:),dipstrsort(:,:)
      real *8, allocatable :: potsort(:,:),gradsort(:,:,:),
     1                             hesssort(:,:,:)
      real *8, allocatable :: pottargsort(:,:),gradtargsort(:,:,:),
     1                              hesstargsort(:,:,:)

      integer norder,npbox,npwmax
      real *8, allocatable :: coefsp(:),rmlexp(:)
      integer *8, allocatable :: lpaddr(:,:)
      integer, allocatable :: isgn(:,:),ncoefs1(:),ncoefs2(:)
      real *8, allocatable :: p2ctransmat(:,:,:,:)
      real *8, allocatable :: c2ptransmat(:,:,:,:)
c     Fourier transforms of the truncated and difference kernels
      real *8, allocatable :: dkernelft(:,:),hpw(:),ws(:),rl(:)
      real *8, allocatable :: coefs1(:,:),coefs2(:,:)
      integer *8 lcoefsptot,lmptotmax
      
c
cc      temporary variables
c
      integer npwlevel,nleafbox,ntot,ns2tp,nlevstart,ndigits
      integer, allocatable :: npw(:),nfourier(:)
      
      integer nfouriermax,ncoefsmax
      
      integer i,ilev,j,jlev,lmptmp,id,k,nhess
      integer ifprint,ier
      integer ibox,istart,iend,jstart,jend,ifplot,ind,npts,jbox,nchild
      integer mc,dad,ipoly,i80,granddad
      integer lenw,keep,ltot,iw,nterms
      
      real *8 beta,bsize,c0,c1,c2,c4,scale
      real *8 pswfeps,bsizesmall,bsizebig
      
      real *8 omp_get_wtime,pps,sc,pi,d
      real *8 time1,time2,ttotal,dt,dttree,dtsort
      real *8 timeinfo(20),tottimeinfo(20)
      real *8 wprolate(5000),rlam20,rkhi,psi0,derpsi0,zero
      real *8 dlogtk0,fval
cccc  data beta/1.454264269259397d1/

      ifprint=1
      ndigits=nint(log10(1.0d0/eps)-0.1)

      if (eps.ge.1.0d-3) then
         scale=8
      elseif (eps.ge.1d-6) then
         scale=20
      elseif (eps.ge.1d-9) then
         scale=25
      elseif (eps.ge.1d-12) then
         scale=25
      endif
      
      call prolc180(eps*scale,beta)
      if(ifprint.ge.1) then
         call prin2('prolate parameter value=*',beta,1)
      endif
      print *, 'beta=',beta
cccc      call provcget(ier,eps*22,beta)
cccc      call prin2('prolate parameter value=*',beta,1)
      lenw=10 000
      call prol0ini(ier,beta,wprolate,rlam20,rkhi,lenw,keep,ltot)
      nterms=wprolate(5)
      if(ifprint.ge.1) then
         call prinf('after prol0ini, ier=*',ier,1)
      endif
cccc      print *, 'nterms=',nterms
      iw=wprolate(1)
cccc      print "(*(g22.15,','))", (wprolate(iw+i-1), i=1,nterms)
cccc      print *, 'keep=',keep, 'ltot=',ltot
      call prolate_intvals(beta,wprolate,c0,c1,c2,c4)
cccc      print *, 'c0=',c0,'c2=',c2
      
      pi = 4.0d0*atan(1.0d0)
      mc=2**dim
      
      do i=1,10
         tottimeinfo(i)=0
      enddo
C
c     set criterion for box subdivision
c
      if (ikernel.eq.0) then
         call lndiv(dim,eps,ns,nt,ifcharge,ifdipole,ifpgh,
     1       ifpghtarg,ndiv,idivflag)
      else
         call lndiv_fast(dim,eps,ns,nt,ifcharge,ifdipole,ifpgh,
     1       ifpghtarg,ndiv,idivflag)
      endif

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
      call pts_tree_refine_once_mem(dim,sources,ns,targ,nt,idivflag,
cccc      call pts_tree_mem(dim,sources,ns,targ,nt,idivflag,
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
      call pts_tree_refine_once_build(dim,sources,ns,targ,nt,
cccc      call pts_tree_build(dim,sources,ns,targ,nt,
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

      if(ifcharge.eq.1.and.ifdipole.eq.0) then
        allocate(chargesort(nd,ns),dipstrsort(nd,1))
        allocate(rnormalsort(dim,1))
      endif
      if(ifcharge.eq.0.and.ifdipole.eq.1) then
        allocate(chargesort(nd,1),dipstrsort(nd,ns))
        allocate(rnormalsort(dim,ns))
      endif
      if(ifcharge.eq.1.and.ifdipole.eq.1) then
        allocate(chargesort(nd,ns),dipstrsort(nd,ns))
        allocate(rnormalsort(dim,ns))
      endif

      nhess = dim*(dim+1)/2
      if(ifpgh.eq.1) then
         allocate(potsort(nd,ns),gradsort(nd,dim,1),
     1    hesssort(nd,nhess,1))
      else if(ifpgh.eq.2) then
         allocate(potsort(nd,ns),gradsort(nd,dim,ns),
     1       hesssort(nd,nhess,1))
      else if(ifpgh.eq.3) then
         allocate(potsort(nd,ns),gradsort(nd,dim,ns),
     1       hesssort(nd,nhess,ns))
      else
         allocate(potsort(nd,1),gradsort(nd,dim,1),
     1       hesssort(nd,nhess,1))
      endif
c      
      if(ifpghtarg.eq.1) then
        allocate(pottargsort(nd,nt),gradtargsort(nd,dim,1),
     1     hesstargsort(nd,nhess,1))
      else if(ifpghtarg.eq.2) then
        allocate(pottargsort(nd,nt),gradtargsort(nd,dim,nt),
     1      hesstargsort(nd,nhess,1))
      else if(ifpghtarg.eq.3) then
        allocate(pottargsort(nd,nt),gradtargsort(nd,dim,nt),
     1     hesstargsort(nd,nhess,nt))
      else
        allocate(pottargsort(nd,1),gradtargsort(nd,dim,1),
     1     hesstargsort(nd,nhess,1))
      endif
c
c     initialize potentials,hessians,gradients
c
      if(ifpgh.eq.1) then
        do i=1,ns
          do id=1,nd
            potsort(id,i) = 0
          enddo
        enddo
      endif

      if(ifpgh.eq.2) then
        do i=1,ns
          do id=1,nd
             potsort(id,i) = 0
             do k=1,dim
                gradsort(id,k,i) = 0
             enddo
          enddo
        enddo
      endif

      if(ifpgh.eq.3) then
        do i=1,ns
          do id=1,nd
            potsort(id,i) = 0
            do k=1,dim
               gradsort(id,k,i) = 0
            enddo
            do k=1,nhess
               hesssort(id,k,i) = 0
            enddo
          enddo
        enddo
      endif


      if(ifpghtarg.eq.1) then
        do i=1,nt
          do id=1,nd
            pottargsort(id,i) = 0
          enddo
        enddo
      endif

      if(ifpghtarg.eq.2) then
        do i=1,nt
          do id=1,nd
            pottargsort(id,i) = 0
            do k=1,dim
               gradtargsort(id,k,i) = 0
            enddo
          enddo
        enddo
      endif

      if(ifpghtarg.eq.3) then
        do i=1,nt
          do id=1,nd
            pottargsort(id,i) = 0
            do k=1,dim
               gradtargsort(id,k,i) = 0
            enddo
            do k=1,nhess
               hesstargsort(id,k,i) = 0
            enddo
          enddo
        enddo
      endif

      
c
c
c     reorder sources
c
      call dreorderf(dim,ns,sources,sourcesort,isrc)
      if(ifcharge.eq.1) 
     1    call dreorderf(nd,ns,charge,chargesort,isrc)
      if(ifdipole.eq.1) then
         call dreorderf(nd,ns,dipstr,dipstrsort,isrc)
         call dreorderf(dim,ns,rnormal,rnormalsort,isrc)
      endif
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
      
      do ilev=0,nlevels
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
            dad =  itree(iptr(3)+ibox-1)
            if (nboxsrcpts(ibox) .le. ndiv .and.
     1          nboxsrcpts(dad).gt.ndiv .and.
     2          nboxsrcpts(ibox) .gt. 0) then
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
      call pdmk_find_all_pwexp_boxes(dim,nboxes,
     1    nlevels,ltree,itree,iptr,
     2    ndiv,nboxsrcpts,nboxtargpts,
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
      if (ikernel.eq.2.and.dim.eq.3) then
         if (ndigits.le.3) then
            npwmax=13
            norder=9
         elseif (ndigits.le.6) then
            npwmax=27
            norder=18
         elseif (ndigits.le.9) then
            npwmax=39
            norder=28
         elseif (ndigits.le.12) then
            npwmax=55
            norder=38
         endif
      else
         if (ndigits.le.3) then
            npwmax=13
            norder=9
         elseif (ndigits.le.6) then
            npwmax=25
            norder=18
         elseif (ndigits.le.9) then
            npwmax=39
            norder=28
         elseif (ndigits.le.12) then
            npwmax=53
            norder=38
         endif
      endif
      
      call pdmk_mpalloc_mem(nd,dim,npwmax,itree,
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
      call pdmk_coefspalloc(nd,dim,itree,lpaddr,
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
               call pdmk_coefsp_zero(nd,npbox,coefsp(lpaddr(1,ibox)))
            endif
         enddo
      enddo
      do ilev = 0,nlevels-1
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
            if (ifpwexpeval(ibox).eq.1) then
               call pdmk_coefsp_zero(nd,npbox,coefsp(lpaddr(2,ibox)))
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





      
c     compute Fourier transform of the truncated kernel at level -1 and
c     Fourier transforms of the difference kernels at levels 0, ..., nlevels-1
c     along radial directions!
      
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      nfouriermax = dim*(npwmax/2)**2
      allocate(dkernelft(0:nfouriermax,-1:nlevels-1))
      allocate(npw(-1:nlevels-1))
      allocate(nfourier(-1:nlevels-1))
      allocate(hpw(-1:nlevels-1))
      allocate(ws(-1:nlevels-1))
      allocate(rl(-1:nlevels-1))

c     truncated kernel at level -1
      bsize = boxsize(0)
      call get_PSWF_truncated_kernel_pwterms(ikernel,rpars,
     1    beta,dim,bsize,eps,hpw(-1),npw(-1),ws(-1),rl(-1))
      nfourier(-1)=dim*(npw(-1)/2)**2
      call get_truncated_kernel_Fourier_transform(ikernel,
     1    rpars,dim,beta,
     1    bsize,rl(-1),npw(-1),hpw(-1),ws(-1),wprolate,
     3    nfourier(-1),dkernelft(0,-1))

      rl(0)=rl(-1)
      do ilev=1,nlevels
         rl(ilev)=rl(ilev-1)/2
      enddo
c     difference kernels at levels 0, ..., nlevels-1
      do ilev=0,nlevels-1
         bsize = boxsize(ilev)
         bsizebig = bsize
         bsizesmall = bsize/2

         call get_PSWF_difference_kernel_pwterms(ikernel,rpars,
     1       beta,dim,bsize,eps,hpw(ilev),npw(ilev),ws(ilev))

         nfourier(ilev)=dim*(npw(ilev)/2)**2

         if (ilev.eq.0) then
            call get_difference_kernel_Fourier_transform(ikernel,
     1          rpars,dim,beta,
     2          bsizesmall,bsizebig,npw(ilev),hpw(ilev),ws(ilev),
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
            endif
         endif
      enddo

c     local kernels at all levels for the Yukawa kernel
      
      ncoefsmax=200
      allocate(coefs1(ncoefsmax,-1:nlevels))
      allocate(ncoefs1(-1:nlevels))
      allocate(coefs2(ncoefsmax,-1:nlevels))
      allocate(ncoefs2(-1:nlevels))
      if (ikernel.gt.0) goto 1200

      do ilev=0,nlevels
         bsize=boxsize(ilev)
         call yukawa_local_kernel_coefs(eps,dim,rpars,beta,
     1       bsize,rl(ilev),wprolate,ncoefs1(ilev),coefs1(1,ilev),
     2       ncoefs2(ilev),coefs2(1,ilev))
c         call prin2('coefs1=*',coefs1(1,ilev),ncoefs1(ilev))
c         call prin2('coefs2=*',coefs2(1,ilev),ncoefs2(ilev))
      enddo
      call prinf('ncoefs1=*',ncoefs1(0),nlevels+1)
      call prinf('ncoefs2=*',ncoefs2(0),nlevels+1)
cccc      pause
 1200 continue
      
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
      nlevstart = max(nlevels-2,0)
      do ilev=nlevstart,nlevstart
         sc=2.0d0/boxsize(ilev)
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
            istart = isrcse(1,ibox)
            iend = isrcse(2,ibox)
            npts = iend-istart+1

c           Check if current box needs to form pw exp         
            if (ifpwexpform(ibox).eq.1) then
c              form equivalent charges directly form sources
               call pdmk_charge2proxycharge(dim,nd,norder,
     1             npts,sourcesort(1,istart),chargesort(1,istart),
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
                     call tens_prod_trans_add(dim,nd,norder,
     1                   coefsp(lpaddr(1,jbox)),norder,
     2                   coefsp(lpaddr(1,ibox)),
     3                   c2ptransmat(1,1,1,j))
                  else
                     jstart = isrcse(1,jbox) 
                     jend = isrcse(2,jbox)
                     npts = jend-jstart+1
                     if (npts.gt.0) then
c                       form equivalent charges directly form sources
                        call pdmk_charge2proxycharge(dim,nd,
     1                      norder,npts,sourcesort(1,jstart),
     2                      chargesort(1,jstart),centers(1,ibox),
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
      do ilev=-1,nlevels-1
         npwlevel = ilev
         call pdmkmain(nd,dim,eps,ikernel,rpars,iperiod,
     1       ifcharge,chargesort,ifdipole,rnormalsort,dipstrsort,
     2       ns,sourcesort,csourcesort,
     3       nt,targsort,ctargsort,
     4       nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     5       npwlevel,ndiv,nboxsrcpts,nboxtargpts,ifleafbox,
     6       norder,lpaddr,coefsp,rmlexp,ifpwexpform,ifpwexpeval,
     7       iftensprodeval,p2ctransmat,beta,wprolate,
     8       npw(ilev),hpw(ilev),nfourier(ilev),dkernelft(0,ilev),
     9       ncoefs1(ilev),coefs1(1,ilev),ncoefs2(ilev),coefs2(1,ilev),
     9       isrcse,itargse,ifpgh,potsort,gradsort,hesssort,
     *       ifpghtarg,pottargsort,gradtargsort,hesstargsort,timeinfo)       
         do i=1,5
            tottimeinfo(i+2)=tottimeinfo(i+2)+timeinfo(i)
         enddo
      enddo






      
c     finally, needs to subtract the self-interaction from the planewave sweeping
      zero=0.0d0
      call prol0eva(zero,wprolate,psi0,derpsi0)
cccc      print *, 'psi_0(0)=',psi0
      if (ikernel.eq.1.and.dim.eq.2) then
         bsize=boxsize(0)
         rl=bsize*sqrt(dim*1.0d0)*2
         
         call log_truncated_kernel(dim,beta,
     1       bsize,rl,wprolate,zero,dlogtk0)
      endif

      do ilev=0,nlevels
         bsize = boxsize(ilev)
cccc         if (ilev .eq. 0) bsize=bsize*0.5d0
c        sc is the value of the truncated kernel at the origin
         if (ikernel.eq.0) then
            call yukawa_truncated_kernel_value_at_zero(dim,rpars,beta,
     1          bsize,rl(ilev),wprolate,fval)
            sc=fval
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
                     potsort(ind,i)=potsort(ind,i)-sc*chargesort(ind,i)
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
         call prin2('time in pdmk main=*',ttotal,1)
         pps=(ns*ifpgh+nt*ifpghtarg+0.0d0)/ttotal
         call prin2('points per sec=*',pps,1)
         call prinf('============================================*',i,0)
      endif







      
c
c     resort the output arrays in input order
c
      if(ifpgh.eq.1) then
        call dreorderi(nd,ns,potsort,pot,isrc)
      endif

      if(ifpgh.eq.2) then
        call dreorderi(nd,ns,potsort,pot,isrc)
        call dreorderiv(nd,dim,ns,gradsort,grad,isrc)
      endif

      if(ifpgh.eq.3) then
        call dreorderi(nd,ns,potsort,pot,isrc)
        call dreorderiv(nd,dim,ns,gradsort,grad,isrc)
        call dreorderiv(nd,nhess,ns,hesssort,hess,isrc)
      endif

      if(ifpghtarg.eq.1) then
        call dreorderi(nd,nt,pottargsort,pottarg,itarg)
      endif

      if(ifpghtarg.eq.2) then
        call dreorderi(nd,nt,pottargsort,pottarg,itarg)
        call dreorderiv(nd,dim,nt,gradtargsort,gradtarg,itarg)
      endif

      if(ifpghtarg.eq.3) then
        call dreorderi(nd,nt,pottargsort,pottarg,itarg)
        call dreorderiv(nd,dim,nt,gradtargsort,gradtarg,itarg)
        call dreorderiv(nd,nhess,nt,hesstargsort,hesstarg,itarg)
      endif

      return
      end
c
c
c
c
c
      subroutine pdmkmain(nd,dim,eps,ikernel,rpars,iperiod,
     1    ifcharge,chargesort,ifdipole,rnormalsort,dipstrsort,
     2    nsource,sourcesort,csourcesort,
     3    ntarget,targetsort,ctargetsort,
     4    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     5    npwlevel,ndiv,nboxsrcpts,nboxtargpts,ifleafbox,
     6    norder,lpaddr,coefsp,rmlexp,ifpwexpform,ifpwexpeval,
     7    iftensprodeval,p2ctransmat,beta,wprolate,
     8    npw,hpw,nfourier,fhat,ncoefs1,coefs1,ncoefs2,coefs2,
     9    isrcse,itargse,ifpgh,pot,grad,hess,
     *    ifpghtarg,pottarg,gradtarg,hesstarg,timeinfo)
c
c
c   the FGT in R^dim: evaluate all pairwise particle
c   interactions
c   and interactions with targets using PW expansions.
c
c   \phi(x_i) = \sum_{j\ne i} charge_j 1/|x_i-x_j|
c
c   All the source/target/expansion center related quantities
c   are assumed to be tree-sorted
c
c-----------------------------------------------------------------------
c   INPUT PARAMETERS:
c
c   nd:   number of charge densities
c
c   dim:  dimension of the space
c
c   eps:  FGT precision requested
c
c   iperiod in: integer
c             flag for periodic implementation
c
c   ifcharge:  charge computation flag
c              ifcharge = 1   =>  include charge contribution
c                                     otherwise do not
c   chargesort: complex *16 (nsource): charge strengths
c
c   ifdipole:  dipole computation flag
c              ifdipole = 1   =>  include dipole contribution
c                                     otherwise do not
c   dipstrsort: complex *16 (nsource): dipole strengths
c   nsource:     integer:  number of sources
c   sourcesort: real *8 (dim,ns):  source locations
c
c   ntarget: integer:  number of targets
c   targetsort: real *8 (dim,ntarget):  target locations
c
c   itree    in: integer (ltree)
c             This array contains all the information
c             about the tree
c             Refer to pts_tree_nd.f
c
c   ltree    in: integer 
c            length of tree
c
c    iptr in: integer(8)
c             iptr is a collection of pointers 
c             which points to where different elements 
c             of the tree are stored in the itree array
c
c     nlevels in: integer
c             number of levels in the tree
c
c     
c     npwlevel in: integer
c             cutoff level at which the plane-wave expansion is valid
c
c     
c     nboxes  in: integer
c             number of boxes in the tree
c
c     boxsize in: real*8 (0:nlevels)
c             boxsize(i) is the size of the box from end to end
c             at level i
c
c     centers in: real *8(dim,nboxes)
c                 array containing the centers of all the boxes
c
c     isrcse in: integer(2,nboxes)
c               starting and ending location of sources in ibox
c                in sorted list of sources
c
c     itargse in: integer(2,nboxes)
c               starting and ending location of targets in ibox
c                in sorted list of sources
c
c     pmax    in:  cutoff limit in the planewave expansion
c     npw     in:  length of planewave expansions
c
c     ifpgh  in: integer
c             flag for evaluating potential/gradients/hessians 
c             at sources.
c             ifpgh = 1, only potentials will be evaluated
c             ifpgh = 2, potentials/gradients will be evaluated
c             ifpgh = 3, potentials/gradients/hessians will be evaluated
c
c     ifpghtarg  in: integer
c             flag for evaluating potential/gradients/hessians 
c             at targets.
c             ifpghtarg = 1, only potentials will be evaluated
c             ifpghtarg = 2, potentials/gradients will be evaluated
c             ifpghtarg = 3, potentials/gradients/hessians will be evaluated
c
c   OUTPUT
c
c   pot: potential at the source locations
c   grad: gradient at the source locations
c   hess: gradient at the source locations
c  
c   pottarg: potential at the target locations
c   gradtarg: gradient at the target locations
c   hesstarg: gradient at the target locations
c------------------------------------------------------------------
c      implicit real *8 (a-h,o-z)
      implicit none
      integer nd,dim,ikernel,iperiod,nsource,ntarget

      integer ndiv,nlevels,npwlevel,ncutoff
      integer ifcharge,ifdipole,ncoefs1,ncoefs2
      integer ifpgh,ifpghtarg,ndigits

      real *8 eps,d2max,rpars(*),coefs1(*),coefs2(*)
      real *8 sourcesort(dim,nsource)
      real *8 csourcesort(nsource,dim)
      real *8 rnormalsort(dim,*)
      real *8 chargesort(nd,*)
      real *8 dipstrsort(nd,*)
      real *8 targetsort(dim,ntarget)
      real *8 ctargetsort(ntarget,dim)
      real *8 pot(nd,*)
      real *8 grad(nd,dim,*)
      real *8 hess(nd,dim*(dim+1)/2,*)
      real *8 pottarg(nd,*)
      real *8 gradtarg(nd,dim,*)
      real *8 hesstarg(nd,dim*(dim+1)/2,*)
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
c     temp variables
      integer, allocatable :: nlist1(:), list1(:,:)
      integer, allocatable :: nlist2(:), list2(:,:)
      integer, allocatable :: nlistpw(:), listpw(:,:)
c
      integer *8, allocatable :: iaddr(:,:)
      integer *8 lmptot,i8

      integer nfourier
      integer ifprint,itype,dad,nchild,ncoll,nnbors
      integer ibox,jbox,ind, npw,npw2, istart,iend,jstart,jend
      integer istartt,iendt,jstartt,jendt,istarts,iends,jends,jstarts
      integer nmax,nhess,mc,nexp,n1,n2,ns,nb,ier,ilev,jlev
      integer isep,i,j,k,iperiod0,ifself,ipoly0
      integer mnlistpw,npts,nptssrc,nptstarg
      integer mnbors,mnlist1,mnlist2
      integer ifpwexpform(nboxes),ifpwexpeval(nboxes)
      integer iftensprodeval(nboxes)
      
      real *8, allocatable :: ts(:)
      real *8, allocatable :: pswfft(:)
      real *8 hpw,fhat(0:nfourier)
      
      real *8 bs0,bsize,d,sc
      real *8 rsc,cen,rscnear,cennear,bsizeinv

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
      ifprint=0

      
      bs0 = boxsize(0)
      mc = 2**dim
      mnbors=3**dim
      nhess=dim*(dim+1)/2
      
      ncutoff=max(npwlevel,0)
      bsize = boxsize(ncutoff)
      
      if (ifprint.ge.1) then
         call prin2('============================================*',d,0)      
         call prinf('npwlevel =*',npwlevel,1)
      endif
      
c     get planewave nodes
      allocate(ts(-npw/2:(npw-1)/2))
      do i=-npw/2,(npw-1)/2
c        symmetric trapezoidal rule - npw odd
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
c     list2 contains boxes that are not neighbors of the given box
c     but still withing the range of the kernel
c
c     use different fast kernel evaluators for these two lists to gain
c     more efficiency
c     
c     boxes are refined once to reduce the total number of points
c     that require at least the squared distance calculation
c
c     further refinement does not lead to reduction in computational time
c     due to cache misses in SIMD vectorization.
      isep=dim
      call compute_mnlists(dim,nboxes,nlevels,itree(iptr(1)),
     1  centers,boxsize,itree(iptr(3)),itree(iptr(4)),
     2  itree(iptr(5)),isep,itree(iptr(6)),
     3  itree(iptr(7)),iperiod,mnlist1,mnlist2)

      allocate(list1(mnlist1,nboxes),nlist1(nboxes))
      allocate(list2(mnlist2,nboxes),nlist2(nboxes))
c     modified list1 for direct evaluation
c     list1 of a childless source box ibox at ilev<=npwlevel
c     contains all childless target boxes that are neighbors of ibox
c     at or above npwlevel
      iperiod0=0
      call pdmk_compute_modified_lists(dim,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,iperiod0,
     2    ifleafbox,ncutoff,isep,
     3    mnlist1,nlist1,list1,
     4    mnlist2,nlist2,list2)
c
c     direct evaluation if the cutoff level is >= the maximum level 
      if (npwlevel .ge. nlevels .and. npwlevel.gt.0) goto 1800

      
c
c     Multipole and local planewave expansions will be held in workspace
c     in locations pointed to by array iaddr(2,nboxes).
      allocate(iaddr(2,nboxes))
c     calculate memory needed for multipole and local planewave expansions
      call pdmk_mpalloc(nd,dim,itree,iaddr,
     1    nlevels,ncutoff,ifpwexpform,ifpwexpeval,lmptot,npw)
      if(ifprint .eq. 1) call prinf_long('lmptot is *',lmptot,1)

         

c     number of plane-wave modes
      nexp = npw**(dim-1)*(npw/2+1)
c     initialization of the work array 
      do ilev = ncutoff,ncutoff
      do ibox=itree(2*ilev+1),itree(2*ilev+2)
c        only these boxes need initialization
c        we use initialization only necessary since the initialization
c        somehow is expensive in Fortran? 
c        need better memory management or switch to a better language
         if (ifpwexpeval(ibox).eq.1 .and. ifpwexpform(ibox).eq.0) then
            call dmk_pwzero(nd,nexp,rmlexp(iaddr(2,ibox)))
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
         sc=2.0d0/boxsize(ilev)
C$OMP PARALLEL DO DEFAULT (SHARED)
C$OMP$PRIVATE(ibox)
C$OMP$SCHEDULE(DYNAMIC)
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
c           Check if current box needs to form pw exp         
            if(ifpwexpform(ibox).eq.1) then
               nb=nb+1
c              form the pw expansion
               call dmk_proxycharge2pw(dim,nd,norder,
     1             coefsp(lpaddr(1,ibox)),npw,tab_coefs2pw,
     3             rmlexp(iaddr(1,ibox)))

               call dmk_multiply_kernelFT(nd,nexp,
     1              rmlexp(iaddr(1,ibox)),pswfft)
            
c              copy the multipole PW exp into local PW exp
c              for self interaction
               call dmk_copy_pwexp(nd,nexp,rmlexp(iaddr(1,ibox)),
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
               call dmk_shiftpw(nd,nexp,rmlexp(iaddr(1,jbox)),
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

c     ... step 5, evaluate all local pw expansions
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

               call dmk_pw2proxypot(dim,nd,norder,npw,
     1            rmlexp(iaddr(2,ibox)),tab_pw2coefs,
     3            coefsp(lpaddr(2,ibox)))
               if (npwlevel.eq.-1) goto 3000

               if (iftensprodeval(ibox).eq.0) goto 1400

               if (ifpghtarg.gt.0 .and. nptstarg.gt.0) then
                  call pdmk_ortho_evalt_nd(dim,nd,norder,
     1                coefsp(lpaddr(2,ibox)),nptstarg,
     2                targetsort(1,istartt),centers(1,ibox),sc,
     3                pottarg(1,istartt))
               endif
               
               if (ifpgh.gt.0 .and. nptssrc.gt.0) then
                  call pdmk_ortho_evalt_nd(dim,nd,norder,
     1                coefsp(lpaddr(2,ibox)),nptssrc,
     2                sourcesort(1,istarts),centers(1,ibox),sc,
     3                pot(1,istarts))
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
                     if (ifpghtarg.gt.0 .and. nptstarg.gt.0) then
                        call pdmk_ortho_evalt_nd(dim,nd,norder,
     1                      coefsp(lpaddr(2,ibox)),nptstarg,
     2                      targetsort(1,jstartt),centers(1,ibox),sc,
     3                      pottarg(1,jstartt))
                     endif
               
                     if (ifpgh.gt.0 .and. nptssrc.gt.0) then
                        call pdmk_ortho_evalt_nd(dim,nd,norder,
     1                      coefsp(lpaddr(2,ibox)),nptssrc,
     2                      sourcesort(1,jstarts),centers(1,ibox),sc,
     3                      pot(1,jstarts))
                     endif
                     
                  elseif (ifpwexpeval(jbox).eq.1) then
c                    translate tensor product polynomial from parent to child
                     call tens_prod_trans_add(dim,nd,norder,
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
cccc      if (ncutoff.eq.0) bsize=bsize/2
c     kernel truncated at bsize, i.e., K(x,y)=0 for |x-y|^2 > d2max
      d2max = bsize**2
c     used in the kernel approximation for boxes in list2
      bsizeinv = 1.0d0/bsize
c     for polynomial approximation
      cennear = -5.0d0/3
      rscnear = bsizeinv**2*8.0d0/3
c     for rational approximation
cccc      rscnear = bsizeinv**2
      
c     used in the kernel approximatin for boxes in list1
      rsc = bsizeinv*2
      cen = -bsize/2
      if ((ikernel.eq.2.and.dim.eq.3).or.
     1    (ikernel.eq.1.and.dim.eq.2)) then
         rsc=bsizeinv*bsizeinv*2
         cen=-1.0d0
      endif

      if (ikernel.eq.0) then
         rsc=bsizeinv*2
         cen=-1.0d0
      endif
      
      ndigits=nint(log10(1.0d0/eps)-0.1)

      nb=0
      if (ncutoff+1 .gt. nlevels) goto 3000
      do 2000 jlev = ncutoff,ncutoff+1
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
            n2 = nlist2(jbox)
            if (npts.gt.0 .and. n1+n2.gt.0) then
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
                  
                  if (nptstarg.gt.0.and.ifpghtarg.gt.0) then
                     call pdmk_direct_c(nd,dim,ikernel,rpars,
     1                   ndigits,rsc,cen,ifself,ncoefs1,coefs1,
     1                   d2max,jstart,jend,sourcesort,
     1                   ifcharge,chargesort,
     2                   ifdipole,rnormalsort,dipstrsort,
     3                   istartt,iendt,ntarget,ctargetsort,
     4                   ifpghtarg,pottarg,gradtarg,hesstarg)
                  endif
                     
                  if (nptssrc.gt.0.and.ifpgh.gt.0) then
                     call pdmk_direct_c(nd,dim,ikernel,rpars,
     1                   ndigits,rsc,cen,ifself,ncoefs1,coefs1,
     1                   d2max,jstart,jend,sourcesort,
     1                   ifcharge,chargesort,
     2                   ifdipole,rnormalsort,dipstrsort,
     3                   istarts,iends,nsource,csourcesort,
     4                   ifpgh,pot,grad,hess)
                  endif
               enddo
            endif

            if (npts.gt.0 .and. n2.gt.0) then
               ifself=0
               do i=1,n2
cccc              ibox is the target box
                  ibox = list2(i,jbox)

                  istarts = isrcse(1,ibox)
                  iends = isrcse(2,ibox)
                  nptssrc = iends-istarts + 1

                  istartt = itargse(1,ibox)
                  iendt = itargse(2,ibox)
                  nptstarg = iendt-istartt+1

                  if (nptstarg.gt.0.and.ifpghtarg.gt.0) then
                     call pdmk_direct_near_c(nd,dim,ikernel,rpars,
     1                   ndigits,rscnear,cennear,bsizeinv,ifself,
     2                   ncoefs2,coefs2,d2max,jstart,jend,sourcesort,
     3                   ifcharge,chargesort,
     4                   ifdipole,rnormalsort,dipstrsort,
     5                   istartt,iendt,ntarget,ctargetsort,
     6                   ifpghtarg,pottarg,gradtarg,hesstarg)
                  endif
                     
                  if (nptssrc.gt.0.and.ifpgh.gt.0) then
                     call pdmk_direct_near_c(nd,dim,ikernel,rpars,
     1                   ndigits,rscnear,cennear,bsizeinv,ifself,
     2                   ncoefs2,coefs2,d2max,jstart,jend,sourcesort,
     3                   ifcharge,chargesort,
     4                   ifdipole,rnormalsort,dipstrsort,
     5                   istarts,iends,nsource,csourcesort,
     6                   ifpgh,pot,grad,hess)
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
      subroutine pdmk_direct_c(nd,dim,ikernel,rpars,ndigits,
     1    rsc,cen,ifself,ncoefs,coefs,d2max,
     $    istart,iend,source,ifcharge,charge,
     2    ifdipole,rnormal,dipstr,
     $    jstart,jend,ntarget,ctarg,ifpgh,pot,grad,hess)
c--------------------------------------------------------------------
c     This subroutine adds the contribution due to sources
c     istart to iend in the source array to the fields at targets
c     jstart to jend in the target array.
c
c     INPUT arguments
c-------------------------------------------------------------------
c     nd           in: integer
c                  number of charge densities
c     dim          in: integer
c                  dimension of the space
c
c     bsize        in: box size of the source box
c
c     dmax         in: maximum distance squared at which the Gaussian kernel is regarded as 0
c
c     iperiod      in: 0: free space; 1: periodic
c
c     shifts       in: real *8 (dim) the source center shifts when iperiod=1
c
c     istart       in:Integer
c                  Starting index in source array whose expansions
c                  we wish to add
c
c     iend         in:Integer
c                  Last index in source array whose expansions
c                  we wish to add
c
c     jstart       in: Integer
c                  First index in target array at which we
c                  wish to update the potential and gradients
c 
c     jend         in:Integer
c                  Last index in target array at which we wish
c                  to update the potential and gradients
c
c     source       in: real *8(dim,ns)
c                  Source locations
c
c     ifcharge     in: Integer
c                  flag for including expansions due to charges
c                  The expansion due to charges will be included
c                  if ifcharge == 1
c
c     charge       in: complex *16
c                  Charge at the source locations
c
c     ifdipole     in: Integer
c                 flag for including expansions due to dipoles
c                 The expansion due to dipoles will be included
c                 if ifdipole == 1
c
c     rnormal        in: real *8(dim,ns)
c                 dipole directions at the source locations
c     dipstr        in: real *8(nd,ns)
c                 dipole strengths at the source locations
c
c     targ        in: real *8(dim,nt)
c                 target locations
c
c     ifpgh        in: Integer
c                  Flag for computing the potential/gradient/hessian.
c                  ifpgh = 1, only potential is computed
c                  ifpgh = 2, potential/gradient are computed
c                  ifpgh = 3, potential/gradient/hessian are computed
c
c------------------------------------------------------------
c     OUTPUT
c
c     pot          potential incremented at targets
c     grad         gradients incremented at targets
c     hess         Hessians  incremented at targets
c-------------------------------------------------------               
      implicit none
c
      integer nd,nsource,ntarget,ndigits
      integer dim,iperiod,ikernel,ncoefs
      integer istart,iend,jstart,jend,ns,ntarg
      integer ifcharge,ifdipole
      integer ifpgh,ifself
      integer i,j,k
c
      real *8 eps,rpars(*),coefs(*)
      real *8 source(dim,*)
      real *8 rnormal(dim,*)
      real *8 bsizeinv,d2max,rsc,cen,rlambda
      real *8 charge(nd,*),dipstr(nd,*)
      real *8 ctarg(ntarget,dim),ztarg
      real *8 pot(nd,*)
      real *8 grad(nd,dim,*)
      real *8 hess(nd,dim*(dim+1)/2,*)
c
        
      ns = iend - istart + 1
      ntarg = jend-jstart+1

      if ((ifcharge.eq.1).and.(ifdipole.eq.0)) then
         if((ifpgh.eq.1) .and. (ifself.eq.1)) then
            if (ikernel.eq.0.and.dim.eq.2) then
               rlambda=rpars(1)
               call y2d_local_kernel_directcp(nd,dim,rlambda,rsc,
     1             cen,d2max,source(1,istart),ns,charge(1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ntarg,
     3             ncoefs,coefs,pot(1,jstart))
            elseif (ikernel.eq.0.and.dim.eq.3) then
               rlambda=rpars(1)
               call y3d_local_kernel_directcp(nd,dim,rlambda,rsc,
     1             cen,d2max,source(1,istart),ns,charge(1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     3             ntarg,ncoefs,coefs,pot(1,jstart))
c               call y3ddirectcp_fast(nd,ndigits,rlambda,
c     1             d2max,source(1,istart),ns,charge(1,istart),
c     2             ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
c     3             ntarg,pot(1,jstart))
            elseif (ikernel.eq.1.and.dim.eq.3) then
               call l3d_local_kernel_directcp(nd,dim,ndigits,rsc,
     1             cen,d2max,source(1,istart),ns,charge(1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     3             ntarg,pot(1,jstart))
            elseif (ikernel.eq.1.and.dim.eq.2) then
               call log_local_kernel_directcp(nd,dim,ndigits,rsc,
     1             cen,d2max,source(1,istart),ns,charge(1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ztarg,
     3             ntarg,pot(1,jstart))
            elseif (ikernel.eq.2.and.dim.eq.2) then
               call l3d_local_kernel_directcp(nd,dim,ndigits,rsc,
     1             cen,d2max,source(1,istart),ns,charge(1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ztarg,
     3             ntarg,pot(1,jstart))
            elseif (ikernel.eq.2.and.dim.eq.3) then
               call sl3d_local_kernel_directcp(nd,dim,ndigits,rsc,
     1             cen,d2max,source(1,istart),ns,charge(1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     3             ntarg,pot(1,jstart))
            endif
         endif
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
      subroutine pdmk_direct_near_c(nd,dim,ikernel,rpars,
     1    ndigits,rsc,cen,bsizeinv,ifself,
     $    ncoefs,coefs,d2max,istart,iend,source,ifcharge,charge,
     2    ifdipole,rnormal,dipstr,
     $    jstart,jend,ntarget,ctarg,ifpgh,pot,grad,hess)
c--------------------------------------------------------------------
c     This subroutine adds the contribution due to sources
c     istart to iend in the source array to the fields at targets
c     jstart to jend in the target array.
c
c     INPUT arguments
c-------------------------------------------------------------------
c     nd           in: integer
c                  number of charge densities
c     dim          in: integer
c                  dimension of the space
c
c     bsize        in: box size of the source box
c
c     dmax         in: maximum distance squared at which the Gaussian kernel is regarded as 0
c
c     iperiod      in: 0: free space; 1: periodic
c
c     shifts       in: real *8 (dim) the source center shifts when iperiod=1
c
c     istart       in:Integer
c                  Starting index in source array whose expansions
c                  we wish to add
c
c     iend         in:Integer
c                  Last index in source array whose expansions
c                  we wish to add
c
c     jstart       in: Integer
c                  First index in target array at which we
c                  wish to update the potential and gradients
c 
c     jend         in:Integer
c                  Last index in target array at which we wish
c                  to update the potential and gradients
c
c     source       in: real *8(dim,ns)
c                  Source locations
c
c     ifcharge     in: Integer
c                  flag for including expansions due to charges
c                  The expansion due to charges will be included
c                  if ifcharge == 1
c
c     charge       in: complex *16
c                  Charge at the source locations
c
c     ifdipole     in: Integer
c                 flag for including expansions due to dipoles
c                 The expansion due to dipoles will be included
c                 if ifdipole == 1
c
c     rnormal        in: real *8(dim,ns)
c                 dipole directions at the source locations
c     dipstr        in: real *8(nd,ns)
c                 dipole strengths at the source locations
c
c     targ        in: real *8(dim,nt)
c                 target locations
c
c     ifpgh        in: Integer
c                  Flag for computing the potential/gradient/hessian.
c                  ifpgh = 1, only potential is computed
c                  ifpgh = 2, potential/gradient are computed
c                  ifpgh = 3, potential/gradient/hessian are computed
c
c------------------------------------------------------------
c     OUTPUT
c
c     pot          potential incremented at targets
c     grad         gradients incremented at targets
c     hess         Hessians  incremented at targets
c-------------------------------------------------------               
      implicit none
c
      integer nd,nsource,ntarget,ndigits
      integer dim,iperiod,ikernel,ncoefs
      integer istart,iend,jstart,jend,ns,ntarg
      integer ifcharge,ifdipole
      integer ifpgh,ifself
      integer i,j,k
c
      real *8 eps,rpars(*),coefs(*)
      real *8 source(dim,*)
      real *8 rnormal(dim,*)
      real *8 cen,d2max,rsc,bsizeinv,rlambda
      real *8 charge(nd,*),dipstr(nd,*)
      real *8 ctarg(ntarget,dim),ztarg
      real *8 pot(nd,*)
      real *8 grad(nd,dim,*)
      real *8 hess(nd,dim*(dim+1)/2,*)
c
        
      ns = iend - istart + 1
      ntarg = jend-jstart+1

      if ((ifcharge.eq.1).and.(ifdipole.eq.0)) then
         if(ifpgh.eq.1) then
            if (ikernel.eq.0.and.dim.eq.2) then
               rlambda=rpars(1)
               call y2d_near_kernel_directcp(nd,dim,rlambda,rsc,cen,
     1             d2max,source(1,istart),ns,charge(1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ntarg,
     3             ncoefs,coefs,pot(1,jstart))
            elseif (ikernel.eq.0.and.dim.eq.3) then
               rlambda=rpars(1)
               call y3d_near_kernel_directcp(nd,dim,rlambda,rsc,cen,
     1             d2max,source(1,istart),ns,charge(1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     3             ntarg,ncoefs,coefs,pot(1,jstart))
            elseif (ikernel.eq.1.and.dim.eq.3) then
               call l3d_near_kernel_directcp(nd,dim,ndigits,rsc,cen,
     1             bsizeinv,d2max,source(1,istart),ns,charge(1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     3             ntarg,pot(1,jstart))
            elseif (ikernel.eq.1.and.dim.eq.2) then
               call log_near_kernel_directcp(nd,dim,ndigits,rsc,cen,
     1             bsizeinv,d2max,source(1,istart),ns,charge(1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ztarg,
     3             ntarg,pot(1,jstart))
            elseif (ikernel.eq.2.and.dim.eq.2) then
               call l3d_near_kernel_directcp(nd,dim,ndigits,rsc,cen,
     1             bsizeinv,d2max,source(1,istart),ns,charge(1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ztarg,
     3             ntarg,pot(1,jstart))
            elseif (ikernel.eq.2.and.dim.eq.3) then
               call sl3d_near_kernel_directcp(nd,dim,ndigits,rsc,cen,
     1             bsizeinv,d2max,source(1,istart),ns,charge(1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     3             ntarg,pot(1,jstart))
            endif
         endif
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
      subroutine kernel_direct(nd,dim,ikernel,rpars,
     1    thresh,istart,iend,source,
     1    ifcharge,charge,ifdipole,rnormal,dipstr,
     2    jstart,jend,targ,ifpgh,pot,grad,hess)
      
c--------------------------------------------------------------------
c     This subroutine adds the contribution due to sources
c     istart to iend in the source array to the fields at targets
c     jstart to jend in the target array.
c
c     INPUT arguments
c-------------------------------------------------------------------
c     nd           in: integer
c                  number of charge densities
c     dim          in: integer
c                  dimension of the space
c
c     dmax         in: maximum distance squared at which the Gaussian kernel is regarded as 0
c
c     istart       in:Integer
c                  Starting index in source array whose expansions
c                  we wish to add
c
c     iend         in:Integer
c                  Last index in source array whose expansions
c                  we wish to add
c
c     jstart       in: Integer
c                  First index in target array at which we
c                  wish to update the potential and gradients
c 
c     jend         in:Integer
c                  Last index in target array at which we wish
c                  to update the potential and gradients
c
c     source       in: real *8(dim,ns)
c                  Source locations
c
c     ifcharge     in: Integer
c                  flag for including expansions due to charges
c                  The expansion due to charges will be included
c                  if ifcharge == 1
c
c     charge       in: complex *16
c                  Charge at the source locations
c
c     ifdipole     in: Integer
c                 flag for including expansions due to dipoles
c                 The expansion due to dipoles will be included
c                 if ifdipole == 1
c
c     rnormal        in: real *8(dim,ns)
c                 dipole directions at the source locations
c     dipstr        in: real *8(nd,ns)
c                 dipole strengths at the source locations
c
c     targ        in: real *8(dim,nt)
c                 target locations
c
c     ifpgh        in: Integer
c                  Flag for computing the potential/gradient/hessian.
c                  ifpgh = 1, only potential is computed
c                  ifpgh = 2, potential/gradient are computed
c                  ifpgh = 3, potential/gradient/hessian are computed
c
c------------------------------------------------------------
c     OUTPUT
c
c     pot          potential incremented at targets
c     grad         gradients incremented at targets
c     hess         Hessians  incremented at targets
c-------------------------------------------------------               
      implicit none
c
      integer nd
      integer dim,ikernel
      integer istart,iend,jstart,jend,ns,ntarg
      integer ifcharge,ifdipole
      integer ifpgh
      integer i,j,k
c
      real *8 rpars(*)
      real *8 source(dim,*)
      real *8 rnormal(dim,*)
      real *8 charge(nd,*),dipstr(nd,*)
      real *8 targ(dim,*),thresh,rlambda
      real *8 pot(nd,*)
      real *8 grad(nd,dim,*)
      real *8 hess(nd,dim*(dim+1)/2,*)
c
        
      ns = iend - istart + 1
      ntarg = jend-jstart+1

      if(ifcharge.eq.1.and.ifdipole.eq.0) then
         if(ifpgh.eq.1) then
            if (ikernel.eq.0.and.dim.eq.2) then
               rlambda=rpars(1)
               call y2ddirectcp(dim,nd,rlambda,source(1,istart),
     1             charge(1,istart),ns,targ(1,jstart),ntarg,
     2             pot(1,jstart),thresh)
            elseif (ikernel.eq.0.and.dim.eq.3) then
               rlambda=rpars(1)
               call y3ddirectcp(dim,nd,rlambda,source(1,istart),
     1             charge(1,istart),ns,targ(1,jstart),ntarg,
     2             pot(1,jstart),thresh)
            elseif ((ikernel.eq.1.and.dim.eq.3).or.
     1          (ikernel.eq.2.and.dim.eq.2)) then
               call l3ddirectcp(dim,nd,source(1,istart),
     1             charge(1,istart),ns,targ(1,jstart),ntarg,
     2             pot(1,jstart),thresh)
            elseif (ikernel.eq.2.and.dim.eq.3) then
               call sl3ddirectcp(dim,nd,source(1,istart),
     1             charge(1,istart),ns,targ(1,jstart),ntarg,
     2             pot(1,jstart),thresh)
            elseif (ikernel.eq.1.and.dim.eq.2) then
               call logdirectcp(dim,nd,source(1,istart),
     1             charge(1,istart),ns,targ(1,jstart),ntarg,
     2             pot(1,jstart),thresh)
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
      subroutine pdmk_mpalloc_mem(nd,dim,npw,laddr,
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
      integer nlevels,npwlevel,npw,nd
      integer laddr(2,0:nlevels), ifpwexpform(*),ifpwexpeval(*)
      integer *8 lmptot(0:nlevels)
      integer *8 lmptotmax,istart,nn,itmp,itmp2
      integer ibox,i
c

      nn = npw**(dim-1)*(npw/2+1)
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
c
c
c------------------------------------------------------------------    
      subroutine pdmk_mpalloc(nd,dim,laddr,iaddr,
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
      integer nlevels,npwlevel,npw,nd
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

      nn = npw**(dim-1)*(npw/2+1)
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
c
c
c
c
c------------------------------------------------------------------    
      subroutine pdmk_coefspalloc(nd,dim,laddr,iaddr,
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

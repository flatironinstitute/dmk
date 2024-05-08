ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     $Date$
c     $Revision$
c
c     created on 04/02/2024 by Shidong Jiang
c
c     based on pdmk2.f. changes: change pdmkmain to fmmmain style,
c     mostly reoganization of the code within this file.
c
c     added dipole to potential on 04/25/2024 by Shidong Jiang
      
      subroutine pdmk(nd,dim,eps,ikernel,rpars,
     1    iperiod,ns,sources,
     2    ifcharge,charge,ifdipole,rnormal,dipstr,
     3    ifpgh,pot,grad,hess,nt,targ,
     4    ifpghtarg,pottarg,gradtarg,hesstarg,tottimeinfo)
c----------------------------------------------
c   INPUT PARAMETERS:
c   nd            : number of densities (same source and target locations, 
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
cc     sorted arrays
c
      integer, allocatable :: isrc(:),isrcse(:,:)
      integer, allocatable :: itarg(:),itargse(:,:)

      real *8, allocatable :: sourcesort(:,:),csourcesort(:,:)
      real *8, allocatable :: rnormalsort(:,:)
      real *8, allocatable :: targsort(:,:),ctargsort(:,:)
      real *8, allocatable :: chargesort(:,:),dipstrsort(:,:)
      real *8, allocatable :: potsort(:,:),gradsort(:,:,:),
     1                             hesssort(:,:,:)
      real *8, allocatable :: pottargsort(:,:),gradtargsort(:,:,:),
     1                              hesstargsort(:,:,:)

      integer nhess
c
cc      temporary variables
c
c
cc      Tree variables
c
      integer, allocatable :: itree(:)
      integer iptr(8)
      real *8, allocatable :: centers(:,:),boxsize(:)

      integer nlmin,nlmax,ifunif
      integer idivflag,nlevels,nboxes,ndiv
      integer ltree
      
      integer i,j,k,id
      integer ifprint
      
      real *8 omp_get_wtime,pps
      real *8 time1,time2,tmain,dt,dttree,dtsort,ttotal
      real *8 timeinfo(20),tottimeinfo(20)

      ifprint=1
      do i=1,10
         tottimeinfo(i)=0
      enddo
C
c     set criterion for box subdivision
c
      if (ikernel.eq.0.or.ifdipole.eq.1) then
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

c      
c     transpose sources and targets, to be used in the fast kernel evaluation
c     written in C++
      allocate(ctargsort(nt,dim))
      allocate(csourcesort(ns,dim))

      
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
c     reorder targets
c
      call dreorderf(dim,nt,targ,targsort,itarg)

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

      
      
      call cpu_time(time1)
C$    time1=omp_get_wtime()

c     main subroutine on sorted quantities
      call pdmkmain(nd,dim,eps,ikernel,rpars,iperiod,
     1    ifcharge,chargesort,ifdipole,rnormalsort,dipstrsort,
     2    ns,sourcesort,csourcesort,nt,targsort,ctargsort,
     3    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     4    isrcse,itargse,ifpgh,potsort,gradsort,hesssort,
     5    ifpghtarg,pottargsort,gradtargsort,hesstargsort,timeinfo)

      call cpu_time(time2)
C$    time2=omp_get_wtime()
      tmain = time2-time1
      
      do i=1,6
         tottimeinfo(i+2)=tottimeinfo(i+2)+timeinfo(i)
      enddo


      ttotal = 0
      do i=1,8
         ttotal = ttotal + tottimeinfo(i)
      enddo
      
      
      ifprint=1
      if (ifprint.eq.1) then
         call prinf('============================================*',i,0)      
         call prinf('laddr=*',itree,2*(nlevels+1))
         call prinf('nlevels=*',nlevels,1)
         call prinf('nboxes=*',nboxes,1)
         call prin2('time in tree build=*',dttree,1)
         pps=(ns+nt+0.0d0)/dttree
         call prin2('points per sec in tree build=*',pps,1)
         call prin2('time in pts_tree_sort=*',dtsort,1)
         pps=(ns+nt+0.0d0)/dtsort
         call prin2('points per sec in tree sort=*',pps,1)
         call prinf('=== STEP 1 (build tree) ===================*',i,0)         
         call prinf('=== STEP 2 (sort points) ==================*',i,0)         
         call prinf('=== STEP 3 (precomputation) ===============*',i,0)         
         call prinf('=== STEP 4 (form proxy charge) ============*',i,0)         
         call prinf('=== STEP 5 (form mp pwexp) ================*',i,0)         
         call prinf('=== STEP 6 (mp to loc) ====================*',i,0)         
         call prinf('=== STEP 7 (eval loc pwexp) ===============*',i,0)         
         call prinf('=== STEP 8 (direct interactions) ==========*',i,0)         
         call prin2('total time info=*', tottimeinfo,8)
         call prin2('time in pdmk main=*',tmain,1)
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
     2    nsource,sourcesort,csourcesort,ntarget,targetsort,ctargetsort,
     3    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     4    isrcse,itargse,ifpgh,pot,grad,hess,
     5    ifpghtarg,pottarg,gradtarg,hesstarg,timeinfo)
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
      implicit none
      integer nd,dim,ikernel,iperiod,nsource,ntarget
      integer ifcharge,ifdipole,ifpgh,ifpghtarg

      real *8 eps,rpars(*)

      real *8 sourcesort(dim,nsource)
      real *8 csourcesort(nsource,dim)
      real *8 targetsort(dim,ntarget)
      real *8 ctargetsort(ntarget,dim)

      real *8 chargesort(nd,*)
      real *8 rnormalsort(dim,*)
      real *8 dipstrsort(nd,*)

      real *8 centers(dim,*)
      real *8 boxsize(0:nlevels)
c
      integer iptr(8)
      integer nboxes,nlevels,ltree
      integer itree(ltree)
      integer isrcse(2,nboxes),itargse(2,nboxes)

      real *8 pot(nd,*)
      real *8 grad(nd,dim,*)
      real *8 hess(nd,dim*(dim+1)/2,*)

      real *8 pottarg(nd,*)
      real *8 gradtarg(nd,dim,*)
      real *8 hesstarg(nd,dim*(dim+1)/2,*)
c
c     local variables
c
      integer ndigits
      integer npw,norder,npbox
      integer, allocatable :: isgn(:,:)

c     number of sources and targets in each box
      integer, allocatable :: nboxsrcpts(:),nboxtargpts(:)

c     flags for each box
      integer, allocatable :: ifpwexpform(:)
      integer, allocatable :: ifpwexpeval(:)
      integer, allocatable :: iftensprodeval(:)
      integer, allocatable :: iftensprodform(:)

c     list of boxes for plane-wave interactions
      integer, allocatable :: nlistpw(:), listpw(:,:)
c     list of boxes for direct interactions
      integer, allocatable :: nlist1(:), list1(:,:)

c     proxy charge child to parent transformation matrix 
      real *8, allocatable :: c2ptransmat(:,:,:,:)
c     proxy charge to outgoing plane-wave transformation matrix
      complex *16, allocatable :: tab_coefs2pw(:,:,:)
c     outoging plane-wave to incoming plane-wave shift matrices
      complex *16, allocatable :: wpwshift(:,:,:)
c     incoming plane-wave to proxy potential transformation matrix
      complex *16, allocatable :: tab_pw2coefs(:,:,:)
c     proxy potential parent to children transformation matrix 
      real *8, allocatable :: p2ctransmat(:,:,:,:)
      
      integer *8 lcoefsptot,lmptot

c     memory space for all densities
      real *8, allocatable :: densitysort(:,:),dipvecsort(:,:,:)
      
c     memory space for proxy charges and potential 
      real *8, allocatable :: coefsp(:)
c     pointer for coefsp
      integer *8, allocatable :: lpaddr(:,:)
c     memory space for outgoing and incoming plane-wave expansions
      real *8, allocatable :: rmlexp(:)
c     pointer for rmlexp
      integer *8, allocatable :: iaddr(:,:)

c     plane-wave nodes
      real *8, allocatable :: ts(:,:),rk(:,:,:)
c     Fourier transforms of the windowed and difference kernels
      integer nfourier
      real *8, allocatable :: pswfft(:,:),fhat(:,:)
      real *8, allocatable :: dkernelft(:,:),hpw(:),ws(:),rl(:)

c     residual kernel poly expansion coefficients
      integer ncoefsmax
      integer, allocatable :: ncoefs1(:),ncoefs2(:)
      real *8, allocatable :: coefs1(:,:),coefs2(:,:)

      real *8 timeinfo(*),time1,time2

      integer ndform,ndeval,npw0
      integer ifprint,itype,nchild,ncoll,nnbors
      integer mnlistpw,npts,nptssrc,nptstarg
      integer mnbors,mnlist1,nmax,nhess,mc,nexp,n1,nb
      integer ntot,ns2tp,nlevstart
      integer ibox,ichild,jbox,ind,istart,iend,jstart,jend
      integer istartt,iendt,jstartt,jendt,istarts,iends,jends,jstarts
      integer ilev
      integer i,j,k,iperiod0,ifself,ipoly,id
      
      real *8 bs0,bsize,bsizebig,bsizesmall,d,sc
      real *8 rsc,cen,bsizeinv

      real *8 omp_get_wtime

      real *8 xq(100),wts,umat,vmat
      integer lenw,keep,ltot,ier
      real *8 beta,scale,wprolate(5000),rlam20,rkhi,psi0,derpsi0,zero
      real *8 d2max,d2min,dlogtk0,fval
      real *8 c0,c1,c2,c4
            
      do i=1,10
         timeinfo(i)=0
      enddo
c
c     preprocessing and precomputation
c      
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      ndform=0
      if (ifcharge.eq.1) ndform=ndform+nd
      if (ifdipole.eq.1) ndform=ndform+nd*dim
      allocate(densitysort(ndform,nsource))
      if (ifdipole.eq.1) allocate(dipvecsort(nd,dim,nsource))
      do i=1,nsource
         id=0
         if (ifcharge.eq.1) then
            do ind=1,nd
               id=id+1
               densitysort(id,i)=chargesort(ind,i)
            enddo
         endif
         if (ifdipole.eq.1) then
            do k=1,dim
               do ind=1,nd
                  id=id+1
                  densitysort(id,i)=dipstrsort(ind,i)*rnormalsort(k,i)
                  dipvecsort(ind,k,i)=densitysort(id,i)
               enddo
            enddo
         endif
      enddo

      
      ndeval=nd

c      
c     ifprint is an internal information printing flag. 
c     Suppressed if ifprint=0.
c     Prints timing breakdown and other things if ifprint=1.
c
      ifprint=1

      
      bs0 = boxsize(0)
      mc = 2**dim
      mnbors=3**dim
      nhess=dim*(dim+1)/2


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

      if (ifdipole.eq.1) then
         scale=1.0d0
      endif
      
      call prolc180(eps*scale,beta)
      if(ifprint.ge.1) then
         call prin2('prolate parameter value=*',beta,1)
      endif
      lenw=10 000
      call prol0ini(ier,beta,wprolate,rlam20,rkhi,lenw,keep,ltot)
      if(ifprint.ge.1) then
         call prinf('after prol0ini, ier=*',ier,1)
      endif
      call prolate_intvals(beta,wprolate,c0,c1,c2,c4)

      
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

c     check whether we need to form and/or evaluate planewave expansions 
c     for boxes
      allocate(ifpwexpform(nboxes))
      allocate(ifpwexpeval(nboxes))
      allocate(iftensprodeval(nboxes))
      allocate(iftensprodform(nboxes))
      call pdmk_find_all_pwexp_boxes3(dim,nboxes,
     1    nlevels,ltree,itree,iptr,nboxsrcpts,nboxtargpts,
     2    ifpwexpform,ifpwexpeval,iftensprodeval)
c
      
c     calculate and allocate maximum memory for planewave expansions
c     needed for one level
      if (ikernel.eq.2.and.dim.eq.3) then
         if (ndigits.le.3) then
            npw=13
            norder=9
         elseif (ndigits.le.6) then
            npw=27
            norder=18
         elseif (ndigits.le.9) then
            npw=39
            norder=28
         elseif (ndigits.le.12) then
            npw=55
            norder=38
         endif
      else
         if (ndigits.le.3) then
            npw=13
            norder=9
         elseif (ndigits.le.6) then
            npw=25
            norder=18
         elseif (ndigits.le.9) then
            npw=39
            norder=28
         elseif (ndigits.le.12) then
            npw=53
            norder=38
         endif
      endif

      if (ifdipole.eq.1) then
         if (ndigits.le.3) then
            npw=21
            norder=12
         elseif (ndigits.le.6) then
            npw=35
            norder=22
         elseif (ndigits.le.9) then
            npw=47
            norder=32
         elseif (ndigits.le.12) then
            npw=61
            norder=42
         endif
      endif
      
      itype = 0
      call chebexps(itype,norder,xq,umat,vmat,wts)
c
c     Multipole and local planewave expansions will be held in workspace
c     in locations pointed to by array iaddr(2,nboxes).
      allocate(iaddr(2,nboxes))
c     calculate memory needed for multipole and local planewave expansions
      call pdmk_mpalloc(ndform,ndeval,dim,npw,nlevels,itree,
     1    ifpwexpform,ifpwexpeval,iaddr,lmptot)
      if(ifprint .eq. 1)
     1  call prinf_long('memory for planewave expansions=*',lmptot,1)
      if(ifprint .eq. 1) call prinf_long('lmptot is *',lmptot,1)
      allocate(rmlexp(lmptot),stat=ier)
      if(ier.ne.0) then
         print *, "Cannot allocate workspace for plane wave expansions"
         print *, "lmptot=", lmptot
         ier = 4
         return
      endif

c     number of plane-wave modes
      nexp = npw**(dim-1)*((npw+1)/2)
c     initialization of the work array 
      do ilev = 0,nlevels
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
            if (ifpwexpeval(ibox).eq.1) 
     1          call dmk_pwzero(nd,nexp,rmlexp(iaddr(2,ibox)))
         enddo
      enddo
      
c     calculate and allocate memory for tensor product grid for all levels
      npbox = norder**dim
      
      allocate(lpaddr(2,nboxes))
      call pdmk_coefspalloc(ndform,ndeval,dim,norder,nlevels,itree,
     1    ifpwexpform,ifpwexpeval,lpaddr,lcoefsptot)
      
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
      do ilev = 0,nlevels
      do ibox=itree(2*ilev+1),itree(2*ilev+2)
         if (ifpwexpform(ibox).eq.1) then
            call pdmk_coefsp_zero(ndform,npbox,coefsp(lpaddr(1,ibox)))
         endif
      enddo
      enddo
      do ilev = 0,nlevels
      do ibox=itree(2*ilev+1),itree(2*ilev+2)
         if (ifpwexpeval(ibox).eq.1) then
            call pdmk_coefsp_zero(ndeval,npbox,coefsp(lpaddr(2,ibox)))
         endif
      enddo
      enddo


      
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





      
c     compute Fourier transform of the windowed kernel at level -1 and
c     Fourier transforms of the difference kernels at levels 0, ..., nlevels-1
c     along radial directions!

      nfourier = dim*(npw/2)**2
      allocate(dkernelft(0:nfourier,-1:nlevels))
      allocate(hpw(-1:nlevels))
      allocate(ws(-1:nlevels))
      allocate(rl(-1:nlevels))

c     windowed kernel at level 0
      bsize = boxsize(0)
      call get_PSWF_windowed_kernel_pwterms(ikernel,rpars,
     1    beta,dim,bsize,eps,hpw(-1),npw0,ws(-1),rl(-1))
      call get_windowed_kernel_Fourier_transform(ikernel,
     1    rpars,dim,beta,
     1    bsize,rl(-1),npw,hpw(-1),ws(-1),wprolate,
     3    nfourier,dkernelft(0,-1))

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
     1       beta,dim,bsize,eps,hpw(ilev),npw0,ws(ilev))

         if (ilev.eq.0) then
            call get_difference_kernel_Fourier_transform(ikernel,
     1          rpars,dim,beta,bsizesmall,bsizebig,
     2          npw,hpw(ilev),ws(ilev),wprolate,
     3          nfourier,dkernelft(0,ilev))
         endif

         if (ilev.gt.0) then
            if (ikernel.eq.0) then
c           Yukawa kernel in 2 and 3 dimensions, no scale invariance
               call get_difference_kernel_Fourier_transform(ikernel,
     1             rpars,dim,beta,bsizesmall,bsizebig,
     2             npw,hpw(ilev),ws(ilev),
     3             wprolate,nfourier,dkernelft(0,ilev))
            elseif (ikernel.eq.2.and.dim.eq.3) then
c           1/r^2 kernel in 3d
               do i=0,nfourier
                  dkernelft(i,ilev)=dkernelft(i,ilev-1)*4
               enddo
            elseif (ikernel.eq.1.and.dim.eq.2) then
c           log(r) kernel in 2d
               do i=0,nfourier
                  dkernelft(i,ilev)=dkernelft(i,ilev-1)
               enddo
            elseif ((ikernel.eq.1.and.dim.eq.3) .or.
     1              (ikernel.eq.2.and.dim.eq.2)) then
c           1/r kernel in 2d and 3d
               do i=0,nfourier
                  dkernelft(i,ilev)=dkernelft(i,ilev-1)*2
               enddo
            endif
         endif
      enddo

c     residual kernels at all levels for the Yukawa kernel
      
      ncoefsmax=200
      allocate(coefs1(ncoefsmax,0:nlevels))
      allocate(ncoefs1(0:nlevels))
      allocate(coefs2(ncoefsmax,0:nlevels))
      allocate(ncoefs2(0:nlevels))

      if (ikernel.eq.0) then
         do ilev=0,nlevels
            bsize=boxsize(ilev)
            if (ilev .eq. 0) bsize=bsize*0.5d0
            call yukawa_residual_kernel_coefs(eps,dim,rpars,beta,
     1          bsize,rl(ilev),wprolate,ncoefs1(ilev),coefs1(1,ilev),
     2          ncoefs2(ilev),coefs2(1,ilev))
         enddo
      elseif (ikernel.eq.1.and.dim.eq.2.and.ifdipole.eq.1) then
         do ilev=0,nlevels
            bsize=boxsize(ilev)
            if (ilev .eq. 0) bsize=bsize*0.5d0
            call log_residual_kernel_coefs(eps,dim,beta,
     1          bsize,rl(ilev),wprolate,ncoefs1(ilev),coefs1(1,ilev))
         enddo
      endif

      
      
      if (ifprint.ge.1) then
         call prin2('============================================*',d,0)      
      endif
      
c     get planewave nodes
      allocate(ts(-npw/2:(npw-1)/2,-1:nlevels))
      allocate(rk(dim,npw**dim,-1:nlevels))
      do ilev = -1,nlevels
         do i=-npw/2,(npw-1)/2
c        symmetric trapezoidal rule - npw odd
            ts(i,ilev)=i*hpw(ilev)
         enddo
         call meshnd(dim,ts(-npw/2,ilev),npw,rk(1,1,ilev))
      enddo

c     tables converting tensor product polynomial expansion coefficients of 
c     the charges to planewave expansion coefficients - on the source side
      allocate(tab_coefs2pw(npw,norder,-1:nlevels))
c     tables converting planewave expansions to tensor product polynomial
c     expansion coefficients of the potentials - on the target side
      allocate(tab_pw2coefs(npw,norder,-1:nlevels))
c      
c     compute translation matrices for PW expansions
c     translation only at the cutoff level
      nmax = 1

      allocate(wpwshift(nexp,(2*nmax+1)**dim,-1:nlevels),stat=ier)
      allocate(pswfft(nexp,-1:nlevels),stat=ier)

      ipoly=1
      do ilev = -1,nlevels
         if (ilev .eq. -1) then
            bsize = boxsize(0)
         else
            bsize = boxsize(ilev)
         endif
         
         call dmk_mk_coefs_pw_conversion_tables(ipoly,norder,npw,
     1       ts(-npw/2,ilev),xq,hpw(ilev),bsize,
     2       tab_coefs2pw(1,1,ilev),tab_pw2coefs(1,1,ilev))
         call mk_pw_translation_matrices(dim,bsize,npw,
     1       ts(-npw/2,ilev),nmax,wpwshift(1,1,ilev))
         call mk_tensor_product_Fourier_transform(dim,
     1       npw,nfourier,dkernelft(0,ilev),nexp,pswfft(1,ilev))
      enddo
c     
c     compute list info for plane-wave sweeping
c
      mnlistpw = 3**dim
      allocate(nlistpw(nboxes),listpw(mnlistpw,nboxes))
c     listpw contains source boxes in the pw interaction
      call pdmk_compute_all_listpw(dim,nboxes,nlevels,
     1    ltree,itree,iptr,centers,boxsize,itree(iptr(1)),
     3    ifpwexpform,mnlistpw,nlistpw,listpw)      
c
c
c     compute list info for direct interactions
c
c     list1 contains boxes that are neighbors of the given box
      mnlist1 = 3**dim
      allocate(list1(mnlist1,nboxes),nlist1(nboxes))
c     modified list1 for direct evaluation
c     list1 of a childless source box ibox at ilev<=npwlevel
c     contains all childless target boxes that are neighbors of ibox
c     at or above npwlevel
      iperiod0=0
      call pdmk_compute_all_modified_list1(dim,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,iperiod0,
     3    mnlist1,nlist1,list1)
c

      call cpu_time(time2)
C$    time2=omp_get_wtime()
      timeinfo(1)=time2-time1
      if (ifprint .eq. 1) 
     1    call prin2('time on precomputation=*',timeinfo(1),1)






      

c
c     DMK main steps
c


c
c     Step 1: upward pass for calculating equivalent charges
c     
      if(ifprint .ge. 1) 
     $   call prinf('=== STEP 1 (form proxy charge) ====*',i,0)
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
               call pdmk_charge2proxycharge(dim,ndform,norder,
     1             npts,sourcesort(1,istart),densitysort(1,istart),
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
                     call tens_prod_trans_add(dim,ndform,norder,
     1                   coefsp(lpaddr(1,jbox)),norder,
     2                   coefsp(lpaddr(1,ibox)),
     3                   c2ptransmat(1,1,1,j))
                  else
                     jstart = isrcse(1,jbox) 
                     jend = isrcse(2,jbox)
                     npts = jend-jstart+1
                     if (npts.gt.0) then
c                       form equivalent charges directly form sources
                        call pdmk_charge2proxycharge(dim,ndform,
     1                      norder,npts,sourcesort(1,jstart),
     2                      densitysort(1,jstart),centers(1,ibox),
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
      timeinfo(2) = time2-time1

      if (ifprint.eq.1) then
         call prinf('number of boxes calling charge2tensprod=*',ns2tp,1)
         call prinf('number of source points involved=*',ntot,1)
         call prinf('total number of source points=*',nsource,1)
         call prin2('time on forming equivalent charge=*',timeinfo(2),1)
      endif





      
c
c
c     Everything else is in the downward pass, but step 2 and step 3 
c     can be done in either direction since they are carried out
c     within each level of the tree.
c
c      
      if(ifprint .ge. 1) 
     $   call prinf('=== STEP 2 (form mp pwexp) ====*',i,0)
      call cpu_time(time1)
C$    time1=omp_get_wtime()

c     
c    first, deal with the windowed kernel at the root level
c
      ibox = 1
      call dmk_proxycharge2pw(dim,ndform,norder,
     1    coefsp(lpaddr(1,ibox)),npw,tab_coefs2pw(1,1,-1),
     3    rmlexp(iaddr(1,ibox)))

      call dmk_multiply_kernelFT_cd2p(nd,dim,ifcharge,ifdipole,
     1    nexp,rmlexp(iaddr(1,ibox)),pswfft(1,-1),rk(1,1,-1))
            
c     copy the multipole PW exp into local PW exp
c     for self interaction
      call dmk_copy_pwexp(ndeval,nexp,rmlexp(iaddr(1,ibox)),
     1    rmlexp(iaddr(2,ibox)))
      
      call dmk_pw2proxypot(dim,ndeval,norder,npw,
     1    rmlexp(iaddr(2,ibox)),tab_pw2coefs(1,1,-1),
     3    coefsp(lpaddr(2,ibox)))

c     
c     now the difference kernels at all levels
c      
      nb=0
      do 1100 ilev = 0,nlevels
C$OMP PARALLEL DO DEFAULT (SHARED)
C$OMP$PRIVATE(ibox)
C$OMP$SCHEDULE(DYNAMIC)
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
c           Check if current box needs to form pw exp         
            if(ifpwexpform(ibox).eq.1) then
               nb=nb+1
c              form the pw expansion
               call dmk_proxycharge2pw(dim,ndform,norder,
     1             coefsp(lpaddr(1,ibox)),npw,tab_coefs2pw(1,1,ilev),
     3             rmlexp(iaddr(1,ibox)))

               call dmk_multiply_kernelFT_cd2p(nd,dim,ifcharge,ifdipole,
     1             nexp,rmlexp(iaddr(1,ibox)),
     2             pswfft(1,ilev),rk(1,1,ilev))
            
c              copy the multipole PW exp into local PW exp
c              for self interaction
               call dmk_copy_pwexp(ndeval,nexp,rmlexp(iaddr(1,ibox)),
     1             rmlexp(iaddr(2,ibox)))
            endif
         enddo
C$OMP END PARALLEL DO 
c     end of ilev do loop
 1100 continue

      call cpu_time(time2)
C$    time2=omp_get_wtime()
      timeinfo(3)=time2-time1

      if(ifprint.ge.1)
     $     call prinf('number of boxes in form mp=*',nb,1)





      

      
      if(ifprint.ge.1)
     $    call prinf('=== Step 3 (mp to loc) ===*',i,0)
c      ... step 3, convert multipole pw expansions into local
c       pw expansions

      call cpu_time(time1)
C$    time1=omp_get_wtime()
      
      do 1300 ilev = 0,nlevels
         bsize = boxsize(ilev)
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
               call dmk_shiftpw(ndeval,nexp,rmlexp(iaddr(1,jbox)),
     1             rmlexp(iaddr(2,ibox)),wpwshift(1,ind,ilev))
            enddo
 1250    continue
C$OMP END PARALLEL DO        
 1300 continue
c
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      timeinfo(4) = time2-time1



      
      
      

      if(ifprint.ge.1)
     $    call prinf('=== step 4 (eval loc pwexp) ===*',i,0)

c     ... step 4, evaluate all local pw expansions
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      nb=0
      do 1500 ilev = 0,nlevels
         sc = 2.0d0/boxsize(ilev)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,istartt,iendt,istarts,iends,nptssrc,nptstarg)
C$OMP$PRIVATE(i,j,k)
C$OMP$SCHEDULE(DYNAMIC)
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
            npts=nboxsrcpts(ibox)+nboxtargpts(ibox)
            if (ifpwexpeval(ibox).eq.1 .and. npts.gt.0) then
               nb=nb+1
               call dmk_pw2proxypot(dim,ndeval,norder,npw,
     1            rmlexp(iaddr(2,ibox)),tab_pw2coefs(1,1,ilev),
     2            coefsp(lpaddr(2,ibox)))

               if (iftensprodeval(ibox).eq.0) goto 1400

               istarts = isrcse(1,ibox)
               iends = isrcse(2,ibox)
               nptssrc = iends-istarts+1
               if (ifpgh.gt.0 .and. nptssrc.gt.0) then
                  call pdmk_ortho_evalt_nd(dim,ndeval,norder,
     1                coefsp(lpaddr(2,ibox)),nptssrc,
     2                sourcesort(1,istarts),centers(1,ibox),sc,
     3                pot(1,istarts))
               endif

               istartt = itargse(1,ibox) 
               iendt = itargse(2,ibox)
               nptstarg = iendt-istartt + 1
               if (ifpghtarg.gt.0 .and. nptstarg.gt.0) then
                  call pdmk_ortho_evalt_nd(dim,ndeval,norder,
     1                coefsp(lpaddr(2,ibox)),nptstarg,
     2                targetsort(1,istartt),centers(1,ibox),sc,
     3                pottarg(1,istartt))
               endif
               

 1400          continue
               nchild = itree(iptr(4)+ibox-1)
               do j=1,nchild
                  jbox = itree(iptr(5) + (ibox-1)*mc+j-1)
                  if (iftensprodeval(jbox).eq.1 .and.
     1                ifpwexpeval(jbox).eq.0) then
c                    evaluate tensor product polynomial approximation at sources
                     jstarts = isrcse(1,jbox)
                     jends = isrcse(2,jbox)
                     nptssrc = jends-jstarts+1
                     if (ifpgh.gt.0 .and. nptssrc.gt.0) then
                        call pdmk_ortho_evalt_nd(dim,ndeval,norder,
     1                      coefsp(lpaddr(2,ibox)),nptssrc,
     2                      sourcesort(1,jstarts),centers(1,ibox),sc,
     3                      pot(1,jstarts))
                     endif

c                    evaluate tensor product polynomial approximation at targets
                     jstartt = itargse(1,jbox) 
                     jendt = itargse(2,jbox)
                     nptstarg = jendt-jstartt + 1
                     if (ifpghtarg.gt.0 .and. nptstarg.gt.0) then
                        call pdmk_ortho_evalt_nd(dim,ndeval,norder,
     1                      coefsp(lpaddr(2,ibox)),nptstarg,
     2                      targetsort(1,jstartt),centers(1,ibox),sc,
     3                      pottarg(1,jstartt))
                     endif
                     
                  elseif (ifpwexpeval(jbox).eq.1) then
c                    translate tensor product polynomial from parent to child
                     call tens_prod_trans_add(dim,ndeval,norder,
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
      timeinfo(5) = time2 - time1
      if(ifprint.ge.1)
     $     call prinf('number of boxes in local eval=*',nb,1)





      
      
      
 1800 continue
      if(ifprint .ge. 1)
     $     call prinf('=== STEP 5 (direct interactions) =====*',i,0)
c
cc
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      ndigits=nint(log10(1.0d0/eps)-0.1)

      nb=0
      do 2000 ilev = 0,nlevels
         bsize = boxsize(ilev)
         if (ilev.eq.0) bsize=bsize/2
c     kernel truncated at bsize, i.e., K(x,y)=0 for |x-y|^2 > d2max
         d2max = bsize**2
         bsizeinv = 1.0d0/bsize
      
c     used in the kernel approximatin for boxes in list1
         rsc = bsizeinv*2
         cen = -bsize/2
         if ((ikernel.eq.2.and.dim.eq.3).or.
     1       (ikernel.eq.1.and.dim.eq.2)) then
            rsc=bsizeinv*bsizeinv*2
            cen=-1.0d0
         endif

         if (ikernel.eq.0) then
            rsc=bsizeinv*2
            cen=-1.0d0
         endif
         
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,jbox,istart,iend,jstartt,jendt,jstarts,jends)
C$OMP$PRIVATE(ns,n1,nptssrc,nptstarg,npts)
C$OMP$SCHEDULE(DYNAMIC)  
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
c           ibox is the source box here            
            istart = isrcse(1,ibox)
            iend = isrcse(2,ibox)
            npts = iend-istart+1

            n1 = nlist1(ibox)
            if (npts.gt.0 .and. n1.gt.0) then
               nb=nb+1
            endif

            if (npts.gt.0 .and. n1.gt.0) then
               ifself=1
               do i=1,n1
cccc              jbox is the target box
                  jbox = list1(i,ibox)

c                 eval at sources
                  jstarts = isrcse(1,jbox)
                  jends = isrcse(2,jbox)
                  nptssrc = jends-jstarts + 1
                  if (nptssrc.gt.0.and.ifpgh.gt.0) then
                     call pdmk_direct_c(nd,dim,ikernel,rpars,
     1                   ndigits,rsc,cen,ifself,
     2                   ncoefs1(ilev),coefs1(1,ilev),
     3                   d2max,istart,iend,sourcesort,
     4                   ifcharge,chargesort,
     5                   ifdipole,dipvecsort,
     6                   jstarts,jends,nsource,csourcesort,
     7                   ifpgh,pot,grad,hess)
                  endif

c                 eval at targets
                  jstartt = itargse(1,jbox)
                  jendt = itargse(2,jbox)
                  nptstarg = jendt-jstartt+1
                  if (nptstarg.gt.0.and.ifpghtarg.gt.0) then
                     call pdmk_direct_c(nd,dim,ikernel,rpars,
     1                   ndigits,rsc,cen,ifself,
     2                   ncoefs1(ilev),coefs1(1,ilev),
     3                   d2max,istart,iend,sourcesort,
     4                   ifcharge,chargesort,
     5                   ifdipole,dipvecsort,
     6                   jstartt,jendt,ntarget,ctargetsort,
     7                   ifpghtarg,pottarg,gradtarg,hesstarg)
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
      timeinfo(6) = time2-time1




      

      
      
      
 3000 continue
      if (ifcharge.eq.0) goto 4000
c     finally, needs to subtract the self-interaction from the planewave sweeping
      zero=0.0d0
      call prol0eva(zero,wprolate,psi0,derpsi0)
cccc      print *, 'psi_0(0)=',psi0
      if (ikernel.eq.1.and.dim.eq.2) then
         bsize=boxsize(0)
         rl=bsize*sqrt(dim*1.0d0)*2
         
         call log_windowed_kernel(dim,beta,
     1       bsize,rl,wprolate,zero,dlogtk0)
      endif

      do ilev=0,nlevels
         bsize = boxsize(ilev)
         if (ilev .eq. 0) bsize=bsize*0.5d0
c        sc is the value of the windowed kernel at the origin
         if (ikernel.eq.0) then
            call yukawa_windowed_kernel_value_at_zero(dim,rpars,beta,
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
c     subtract the self-interaction
            ichild = itree(iptr(4)+ibox-1)
            if (ichild.eq.0) then
               do i=istart,iend
                  do ind=1,nd
                     pot(ind,i)=pot(ind,i)-sc*chargesort(ind,i)
                  enddo
               enddo
            endif
         enddo
      enddo
         
      call cpu_time(time2)
C$    time2=omp_get_wtime()









 4000 continue
      
      if(ifprint.ge.1) call prin2('timeinfo=*',timeinfo,6)
      d = 0
      do i = 1,6
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
     2    ifdipole,dipvec,
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
      real *8 bsizeinv,d2max,rsc,cen,rlambda
      real *8 charge(nd,*),dipvec(nd,dim,*)
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

      if ((ifcharge.eq.0).and.(ifdipole.eq.1)) then
         if((ifpgh.eq.1) .and. (ifself.eq.1)) then
            if (ikernel.eq.1.and.dim.eq.2) then
               call log_local_kernel_directdp_fast(nd,dim,ndigits,rsc,
     1             cen,d2max,source(1,istart),ns,dipvec(1,1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),
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

      if((ifcharge.eq.1).and.(ifdipole.eq.0)) then
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

      if((ifcharge.eq.0).and.(ifdipole.eq.1)) then
         if(ifpgh.eq.1) then      
            if (ikernel.eq.1.and.dim.eq.2) then
               call logdirectdp(dim,nd,source(1,istart),ns,
     1             dipstr(1,istart),rnormal(1,istart),
     2             targ(1,jstart),ntarg,
     3             pot(1,jstart),thresh)
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
c------------------------------------------------------------------    
      subroutine pdmk_mpalloc(ndform,ndeval,dim,npw,nlevels,laddr,
     1    ifpwexpform,ifpwexpeval,iaddr,lmptot)
c     This subroutine determines the size of the array
c     to be allocated for multipole/local expansions
c
c     Input arguments
c     ndform      in: integer
c                 number of outgoing expansions
c
c     ndeval      in: integer
c                 number of incoming expansions
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
      integer nlevels,npw,ndform,ndeval
      integer laddr(2,0:nlevels),ifpwexpform(*),ifpwexpeval(*)
      integer *8 iaddr(2,*),ilmptot(0:nlevels)
      integer *8 lmptot,istart,nn,nn1,nn2,itmp,itmp2
      integer ibox,i,istarts,iends,npts
c
      nn = npw**(dim-1)*((npw+1)/2)
c     the factor 2 is the (complex *16)/(real *8) ratio
      nn1 = nn*2*ndform
      nn2 = nn*2*ndeval

c     assign memory pointers
      istart = 1
      itmp=0
      do i = 0,nlevels
         do ibox = laddr(1,i),laddr(2,i)
c          Allocate memory for the multipole PW expansions
           if (ifpwexpform(ibox).eq.1) then
              iaddr(1,ibox) = istart + itmp*nn1
              itmp = itmp+1
           endif
         enddo
         istart = istart + itmp*nn1
      enddo
c
      itmp2=0
      do i = 0,nlevels
         do ibox = laddr(1,i),laddr(2,i)
c          Allocate memory for the local PW expansions
           if (ifpwexpeval(ibox).eq.1) then
              iaddr(2,ibox) = istart + itmp2*nn2
              itmp2 = itmp2+1
           endif
         enddo
         istart = istart + itmp2*nn2
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
      subroutine pdmk_coefspalloc(ndform,ndeval,dim,norder,nlevels,
     1    laddr,ifpwexpform,ifpwexpeval,iaddr,lmptot)
c     This subroutine determines the size of the array
c     to be allocated for multipole/local expansions
c
c     Input arguments
c     ndform      in: integer
c                 number of outgoing expansions
c
c     ndeval      in: integer
c                 number of incoming expansions
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
      integer nlevels,npw,ndform,ndeval,norder
      integer *8 iaddr(2,*)
      integer laddr(2,0:nlevels), ifpwexpform(*),ifpwexpeval(*)
      integer *8 lmptot,istart,nn,itmp,itmp2,nn1,nn2
      integer ibox,i,istarts,iends,npts
c
      istart = 1
      
      nn = norder**dim
      nn1 = nn*ndform
      nn2 = nn*ndeval      

      itmp=0
      do i = 0,nlevels
         do ibox = laddr(1,i),laddr(2,i)
c     Allocate memory for the multipole PW expansion         
c
           if (ifpwexpform(ibox).eq.1) then
              iaddr(1,ibox) = istart + itmp*nn1
              itmp = itmp+1
           endif
         enddo
         istart = istart + itmp*nn1
      enddo
c
      itmp2=0
      do i = 0,nlevels
         do ibox = laddr(1,i),laddr(2,i)
c     Allocate memory for the local PW expansion         
c
           if (ifpwexpeval(ibox).eq.1) then
              iaddr(2,ibox) = istart + itmp2*nn2
              itmp2 = itmp2+1
           endif
         enddo
         istart = istart + itmp2*nn2
      enddo

      lmptot = istart

      return
      end
c
c
c
c

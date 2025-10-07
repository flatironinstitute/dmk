ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     $Date$
c     $Revision$
c
c     created on 06/01/2025 by Shidong Jiang
c     
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
c     modified from stokesdmk6.f.
c     Difference:
c     add periodic boundary conditions
c      
c     Note: added bs0 and cen0 to the input parameters!!!
c
c
      
      subroutine stokesdmk(nd,dim,eps,
     1    iperiod,rbsize,rbcenter,ns,sources,
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
c     used only for PBCs
      real *8 rbsize, rbcenter(dim)
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
      integer idivflag,nlevels,nboxes,ndiv
      integer ltree

c
cc     sorted arrays
c
      integer, allocatable :: isrc(:),isrcse(:,:)
      integer, allocatable :: itarg(:),itargse(:,:)

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

      integer i,j,id,k,nhess,ndtot
      integer ifprint,ier,ikernel
      integer nsplot,ntplot,ifplot
      
      real *8 omp_get_wtime,pps
      real *8 time1,time2,ttotal,dt,dttree,dtsort,dtmain
      real *8 timeinfo(20),tottimeinfo(20)
      character *9 fname1
      character *8 fname2
      character *9 fname3

      ifprint=1
      
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
      if (iperiod .eq. 1) nlmin = 1
      ifunif = 0

      call cpu_time(time1)
C$    time1=omp_get_wtime()
c     find the memory requirements for the tree
      call pts_tree_mem(dim,sources,ns,targ,nt,idivflag,
     1    ndiv,nlmin,nlmax,ifunif,iperiod,rbsize,rbcenter,
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
     1    idivflag,ndiv,nlmin,nlmax,ifunif,iperiod,rbsize,rbcenter,
     2    nlevels,nboxes,ltree,itree,iptr,centers,boxsize)
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      dttree = time2-time1
      tottimeinfo(1)=dttree


      ifplot=0
      if (ifplot.eq.1) then
         fname1 = 'tree.data'
         fname2 = 'src.data'
         fname3 = 'targ.data'
      
         nsplot=0
         ntplot=0
c        given ns and nt
c        generate random sources and targets 
c        and sort into the tree?
         call print_tree2d_matlab(dim,itree,ltree,nboxes,centers,
     1       boxsize,nlevels,iptr,nsplot,sources,
     2       ntplot,targ,fname1,fname2,fname3)
      endif
      
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
      if(ifstoklet.eq.1.and.ifstrslet.eq.0) then
        allocate(stokletsort(nd,dim,ns))
        allocate(strsletsort(nd,dim,1))
        allocate(strsvecsort(nd,dim,1))
      endif

      if(ifstoklet.eq.0.and.ifstrslet.eq.1) then
        allocate(stokletsort(nd,dim,1))
        allocate(strsletsort(nd,dim,ns))
        allocate(strsvecsort(nd,dim,ns))
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
c     reorder sources
c
      ndtot = nd*dim
      call dreorderf(dim,ns,sources,sourcesort,isrc)
      if(ifstoklet.eq.1) 
     1    call dreorderf(ndtot,ns,stoklet,stokletsort,isrc)
      if(ifstrslet.eq.1) then
         call dreorderf(ndtot,ns,strslet,strsletsort,isrc)
         call dreorderf(ndtot,ns,strsvec,strsvecsort,isrc)
      endif
c
c     reorder targets
c
      call dreorderf(dim,nt,targ,targsort,itarg)
c
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

      call cpu_time(time1)
C$    time1=omp_get_wtime()

      
c     main subroutine on sorted quantities
      ikernel = 3
      call stokesdmkmain(nd,dim,eps,ikernel,iperiod,
     1       ifstoklet,stokletsort,ifstrslet,strsletsort,strsvecsort,
     2       ns,sourcesort,csourcesort,nt,targsort,ctargsort,
     3       nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     4       isrcse,itargse,ifppreg,potsort,presort,gradsort,
     5       ifppregtarg,pottargsort,pretargsort,gradtargsort,timeinfo)       

      call cpu_time(time2)
C$    time2=omp_get_wtime()
      dtmain = time2-time1

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
         call prinf('=== STEP 7 (proxy pot to pot) =============*',i,0)         
         call prinf('=== STEP 8 (direct interactions) ==========*',i,0)         
         call prin2('total time info=*', tottimeinfo,8)
         call prin2('time in stokesdmk main=*',dtmain,1)
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
     2    nsource,sourcesort,csourcesort,ntarget,targetsort,ctargetsort,
     3    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     4    isrcse,itargse,ifppreg,pot,pre,grad,
     5    ifppregtarg,pottarg,pretarg,gradtarg,timeinfo)
c
      implicit none
      integer nd,dim,ikernel,iperiod,nsource,ntarget

      integer ndiv,nlevels,npwlevel,ncutoff
      integer ifstoklet,ifstrslet
      integer ifppreg,ifppregtarg,ndigits
      
      real *8 eps,rpars(10)

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
c
c     local variables
c
      integer, allocatable :: isgn(:,:)

c     number of sources and targets in each box
      integer, allocatable :: nboxsrcpts(:),nboxtargpts(:),nboxpts(:)

c     flags for each box
      integer, allocatable :: ifpwexp(:)
      integer, allocatable :: iftensprodform(:)      
      integer, allocatable :: iftensprodeval(:)      
c     list of boxes for plane-wave interactions
      integer, allocatable :: nlistpw(:), listpw(:,:)
c     list of boxes for direct interactions
      integer, allocatable :: nlist1(:), list1(:,:)

c     proxy charge child to parent transformation matrix 
      real *8, allocatable :: c2ptransmat(:,:,:,:)
c     proxy charge to outgoing plane-wave transformation matrix
      complex *16, allocatable :: tab_coefs2pw(:,:,:)
      complex *16, allocatable :: tab_coefs2pw_win(:,:)
c     outoging plane-wave to incoming plane-wave shift matrices
      complex *16, allocatable :: wpwshift(:,:,:)
      complex *16, allocatable :: pwexp_win(:,:),localpwexp(:,:)
c     incoming plane-wave to proxy potential transformation matrix
      complex *16, allocatable :: tab_pw2coefs(:,:,:)
      complex *16, allocatable :: tab_pw2coefs_win(:,:)
c     proxy potential parent to children transformation matrix 
      real *8, allocatable :: p2ctransmat(:,:,:,:)

c     memory space for all densities
      real *8, allocatable :: densitysort(:,:)

c     memory space for proxy charges and proxy potentials 
      real *8, allocatable :: coefsp(:)
c     pointer for coefsp
      integer *8, allocatable :: lpaddr(:,:)
c     memory space for outgoing plane-wave expansions
      real *8, allocatable :: rmlexp(:)
c     pointer for rmlexp
      integer *8, allocatable :: iaddr(:)

c     plane-wave nodes
      real *8, allocatable :: ts(:,:),rk(:,:,:),rksq(:,:)
      real *8, allocatable :: ts_win(:),rk_win(:,:),rksq_win(:)

c     Fourier transforms of the windowed and difference kernels
      real *8, allocatable :: pswfft(:,:),fhat(:,:)
      complex *16, allocatable :: uhat_win(:,:,:),uhat(:,:,:)
      real *8, allocatable :: dkernelft(:,:),hpw(:),ws(:),rl(:)
      real *8, allocatable :: pswfft_win(:),fhat_win(:),dkernelft_win(:)

c     chebyshev expansion coeficients for the residual kernel
      real *8, allocatable :: coefs_diag(:),coefs_offd(:)
      
      integer *8 lcoefsptot,lmptot
      integer norder,npbox,ncoefsmax
      integer ndform,ndeval
      integer ipoly
      
      integer nfourier,nfourier_win
      integer n_diag,n_offd
      
      integer ifprint,itype,dad,nchild,ncoll,nnbors,iftpform,iftpeval
      integer ibox,jbox,istart,iend,jstart,jend,ilev,jlev,nlevstart
      integer istartt,iendt,jstartt,jendt,istarts,iends,jends,jstarts
      integer nmax,nhess,mc,n1,n2,nb,ier
      integer isep,i,j,k,id,ind,inds(10),nitotal
      integer mnlistpw,npts,nptssrc,nptstarg,nptsj
      integer mnbors,mnlist1,mnlist2

      integer ntot,ns2tp
      real *8 beta,hf,hf_win
      real *8 wprolate(5000),rlam20,rkhi

      integer nf,npw,nexp
      integer nf_win,npw_win,nexp_win
      real *8 hpw_win,ws_win,rl_win

      integer lenw,keep,ltot,nterms
      
      real *8 bs0,bsize,bsizebig,bsizesmall,rl0,sc,sci
      real *8 rsc,cen,rscnear,cennear,bsizeinv,d2min,d2max,thresh2

      real *8 zero,psi0,derpsi0,c0,c1,c2,c4,g0d2,cval
      real *8, allocatable :: cvec(:,:)
      
      real *8 d,st2dwk0
      real *8 omp_get_wtime
c
      real *8 xq(100),wts,umat,vmat
      real *8 pi

c     for PBCs, sourceimages and shifts
      integer ifshift
      real *8 cshift(10)

      
      
      call cpu_time(time1)
C$    time1=omp_get_wtime()
      
      pi = 4.0d0*atan(1.0d0)
            
      do i=1,10
         timeinfo(i)=0
      enddo

c      
c     ifprint is an internal information printing flag. 
c     Suppressed if ifprint=0.
c     Prints timing breakdown and other things if ifprint=1.
c
      ifprint=1

      ndform=0
      if (ifstoklet.eq.1) ndform=ndform+nd*dim
      if (ifstrslet.eq.1) ndform=ndform+nd*dim*dim
      allocate(densitysort(ndform,nsource))
      do i=1,nsource
         id=0
         if (ifstoklet.eq.1) then
            do ind=1,nd
            do k=1,dim
               id=id+1
               densitysort(id,i)=stokletsort(ind,k,i)
            enddo
            enddo
         endif

         if (ifstrslet.eq.1) then
            do ind=1,nd
            do k=1,dim
            do j=1,dim   
               id=id+1
               densitysort(id,i)=strsletsort(ind,k,i)
     1             *strsvecsort(ind,j,i)
            enddo
            enddo
            enddo
         endif
      enddo

      ndeval = nd*dim
      
      bs0    = boxsize(0)
      mc     = 2**dim
      mnbors = 3**dim

c     determine the parameter value for the PSWF
c     from Ludvig's matlab code
c      if (ifstoklet.eq.1) then
c         beta = pi/3.0d0*ceiling(3.0d0/pi*(1.11d0-log10(eps)) / 0.41d0)
c      endif
c     if (ifstrslet.eq.1)
c     use the same value for stokeslet and stresslet
      beta = pi/3.0d0*ceiling(3.0d0/pi*(0.69d0-log10(eps)) / 0.39d0)

c     number of Fourier modes along each dimension
      nf = ceiling(3*beta/pi*(1-eps))
c     Fourier spacing for the difference kernel
      hf = 2.0d0*beta/nf
      nf = nf-1
c
c     polynomial order for proxy charge/potential
c     From Ludvig's matlab code
cccc      if (ifstoklet.eq.1) norder = ceiling(1.43d0*beta - 2.76d0)
cccc      if (ifstrslet.eq.1) norder = ceiling(1.43d0*beta - 3.26d0)
c     use the same value for stokeslet and stresslet
      norder = ceiling(1.43d0*beta - 3.26d0)
      call prinf('norder=*',norder,1)

      if (iperiod.eq.0) then
c        Windowed kernel: Fourier spacing
         hf_win = 1.0d0
c        number of Fourier modes along each dimension
         nf_win = ceiling(beta)-1
      elseif (iperiod.eq.1) then
c        we still use ..._win to denote periodic quantities!!!
c        hf_win should be equal to 2*pi/boxsize(0), after
c        normalization, this is simply 2*pi.
         hf_win = 2*pi
c        periodic kernel at level 1, not level 0!!!
c        This is why there is 2 in nf_win
         nf_win = ceiling(2*beta/hf_win)
      endif
      
c     Total number of Fourier modes along each dimension
      npw_win = 2*nf_win + 1
c     total number of plane-wave modes
      nexp_win = npw_win**(dim-1)*((npw_win+1)/2)

c     Total number of Fourier modes for the difference kernel along each dimension
      npw  = 2*nf + 1
c     total number of plane-wave modes
      nexp = npw**(dim-1)*((npw+1)/2)

      print *, beta, npw_win, npw

      if(ifprint.ge.1) then
         call prin2('prolate parameter value=*',beta,1)
         call prin2('Fourier spacing=*',hf,1)
      endif

c     precompute PSWF Legendre polynomial expansion coefficients
      lenw=10 000
      call prol0ini(ier,beta,wprolate,rlam20,rkhi,lenw,keep,ltot)
      nterms=wprolate(5)
      if(ifprint.ge.1) then
         call prinf('after prol0ini, ier=*',ier,1)
      endif

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

      allocate(nboxpts(nboxes))
      do ilev=0,nlevels
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
            nboxpts(ibox) = nboxsrcpts(ibox) + nboxtargpts(ibox)
         enddo
      enddo

c     check whether we need to form and/or evaluate planewave expansions 
c     for boxes
      allocate(ifpwexp(nboxes))
      call bdmk_find_all_pwexp_boxes(dim,nboxes,
     1    nlevels,ltree,itree,iptr,iperiod,ifpwexp)
      
      itype = 0
      call chebexps(itype,norder,xq,umat,vmat,wts)
c
c     Multipole and local planewave expansions will be held in workspace
c     in locations pointed to by array iaddr(nboxes).
      allocate(iaddr(nboxes))
c     calculate memory needed for multipole and local planewave expansions
      call pdmk_mpalloc(ndform,dim,npw,nlevels,itree,
     1    ifpwexp,iaddr,lmptot)
      if(ifprint .eq. 1)
     1  call prinf_long('memory for planewave expansions=*',lmptot,1)
cccc      if(ifprint .eq. 1) call prinf_long('lmptot is *',lmptot,1)
      allocate(rmlexp(lmptot),stat=ier)
      if(ier.ne.0) then
         print *, "Cannot allocate workspace for plane wave expansions"
         print *, "lmptot=", lmptot
         ier = 4
         return
      endif
c     initialization of the work array 
      do ilev = 0,nlevels
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
            if (ifpwexp(ibox).eq.1) 
     1          call dmk_pwzero(ndform,nexp,rmlexp(iaddr(ibox)))
         enddo
      enddo

c     calculate and allocate memory for tensor product grid for all levels
      allocate(iftensprodform(nboxes))
      allocate(iftensprodeval(nboxes))
      do ibox=1,nboxes
         iftensprodform(ibox)=0
      enddo

      do ibox=1,nboxes
         iftensprodeval(ibox)=0
      enddo

      do ilev = 0,nlevels
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
            if (ifpwexp(ibox).eq.1 .and. nboxpts(ibox).gt.0) then
c     determine whether we need to evaluate proxypotential at ibox
               iftpeval = 1
               nchild = itree(iptr(4)+ibox-1)
               do j=1,nchild
                  jbox = itree(iptr(5) + (ibox-1)*mc+j-1)
                  if (ifpwexp(jbox).eq.1) then
                     iftpeval = 0
                  endif
               enddo
               iftensprodeval(ibox) = iftpeval
               if (iftpeval .eq. 0) then
                  do j=1,nchild
                     jbox = itree(iptr(5) + (ibox-1)*mc+j-1)
                     if (nboxpts(jbox).gt.0) then
                        if (ifpwexp(jbox).eq.0) then
                           iftensprodeval(jbox) = 1
                        endif
                     endif
                  enddo
               endif
            endif
         enddo
      enddo
      
      npbox = norder**dim
      
      allocate(lpaddr(2,nboxes))
      call pdmk_coefspalloc(ndform,ndeval,dim,norder,nlevels,itree,
     1    ifpwexp,iftensprodeval,lpaddr,lcoefsptot)
      
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
         if (ifpwexp(ibox).eq.1) then
            call pdmk_coefsp_zero(ndform,npbox,coefsp(lpaddr(1,ibox)))
         endif
      enddo
      enddo
      do ilev = 0,nlevels
      do ibox=itree(2*ilev+1),itree(2*ilev+2)
         if ((ifpwexp(ibox).eq.1) .or.
     1       (iftensprodeval(ibox).eq.1)) then
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

      
c      
c     compute Fourier transform of the windowed kernel and
c     Fourier transforms of the difference kernels at levels 0, ..., nlevels-1
c     along radial directions!

c     windowed kernel at level 0 or periodized kernel at level 2
      bsize = boxsize(0)
      nfourier_win = dim*(npw_win/2)**2
      allocate(dkernelft_win(0:nfourier_win))

      hpw_win = hf_win/bsize
      ws_win  = hpw_win**dim/pi**(dim-1)/2

      
      rl_win = bsize*(sqrt(dim*1.0d0)+1)

      nfourier = dim*(npw/2)**2
      allocate(dkernelft(0:nfourier,0:nlevels))
      allocate(hpw(0:nlevels))
      allocate(ws(0:nlevels))
      allocate(rl(0:nlevels))

c     windowed kernel
      bsize = boxsize(0)

      if (iperiod.eq.0) then
         call get_windowed_kernel_Fourier_transform(ikernel,
     1       rpars,dim,beta,
     2       bsize,rl_win,npw_win,hpw_win,ws_win,wprolate,
     3       nfourier_win,dkernelft_win)
      endif
      
      if (iperiod.eq.1) then
         bsize = boxsize(0)/2
         call get_periodic_kernel_Fourier_transform(ikernel,
     1       rpars,dim,beta,
     2       bsize,npw_win,hpw_win,ws_win,wprolate,
     3       nfourier_win,dkernelft_win)
      endif
      
      rl(0)=rl_win
      do ilev=1,nlevels
         rl(ilev)=rl(ilev-1)/2
      enddo
c     difference kernels at levels 0, ..., nlevels-1
      do ilev=0,nlevels
         bsize = boxsize(ilev)
         bsizebig = bsize
         bsizesmall = bsize/2

         hpw(ilev) = hf/bsize
         ws(ilev)  = hpw(ilev)**dim/pi**(dim-1)/2

         if (ilev.eq.0) then
            call get_difference_kernel_Fourier_transform(ikernel,
     1          rpars,dim,beta,bsizesmall,bsizebig,
     2          npw,hpw(ilev),ws(ilev),
     3          wprolate,nfourier,dkernelft(0,ilev))
         endif

         if (ilev.gt.0) then
            if ((dim.eq.2)) then
c           2d stokeslet
               do i=0,nfourier
                  dkernelft(i,ilev)=dkernelft(i,ilev-1)/4
               enddo
            elseif ((dim.eq.3)) then
c           3d stokeslet
               do i=0,nfourier
                  dkernelft(i,ilev)=dkernelft(i,ilev-1)/2
               enddo
            endif
         endif
      enddo

c     residual kernels at all levels for the Stokeslet

      ncoefsmax=200
      allocate(coefs_diag(ncoefsmax))
      allocate(coefs_offd(ncoefsmax))

      bsize = 1.0d0
      rl0 = rl(0)/boxsize(0)
      if (ifstoklet .eq. 1) then
         call stokes_residual_kernel_coefs3(eps,dim,beta,bsize,
     1       rl0,wprolate,n_diag,coefs_diag,n_offd,coefs_offd)
      endif
      if (ifstrslet .eq. 1) then
         call stresslet_reskernel_coefs(eps,dim,beta,bsize,
     1       rl0,wprolate,n_diag,coefs_diag,n_offd,coefs_offd)
      endif
      
cccc output for Matlab to produce monomial coefficients for SIMD fast kernel evaluation         
cccc 1111 format(E23.16, 50(1x,E23.16))
cccc      print *, 'n_diag=', n_diag, 'n_offd=', n_offd
cccc      
cccc      write(*,'(a)', advance='no') 'chebcd = ['
cccc      write(*,1111,advance='no') (coefs_diag(j), j=1,n_diag)
cccc      write(*,*) '];' 
cccc
cccc      write(*,'(a)', advance='no') 'chebco = ['
cccc      write(*,1111,advance='no') (coefs_offd(j), j=1,n_offd)
cccc      write(*,*) '];'
cccc
cccc      pause
      
c     get planewave nodes
      allocate(ts_win(-npw_win/2:(npw_win-1)/2))
      allocate(rk_win(dim,npw_win**dim))
      allocate(rksq_win(npw_win**dim))

      do i=-npw_win/2,(npw_win-1)/2
c     symmetric trapezoidal rule - npw odd
         ts_win(i)=i*hpw_win
      enddo
      call meshnd(dim,ts_win,npw_win,rk_win)
      call meshndsq(dim,nexp_win,rk_win,rksq_win)

      allocate(ts(-npw/2:(npw-1)/2,0:nlevels))
      allocate(rk(dim,npw**dim,0:nlevels))
      allocate(rksq(npw**dim,0:nlevels))
      do ilev = 0,nlevels
         do i=-npw/2,(npw-1)/2
c        symmetric trapezoidal rule - npw odd
            ts(i,ilev)=i*hpw(ilev)
         enddo
         call meshnd(dim,ts(-npw/2,ilev),npw,rk(1,1,ilev))
         call meshndsq(dim,nexp,rk(1,1,ilev),rksq(1,ilev))
      enddo
     
c     tables converting tensor product polynomial expansion coefficients of 
c     the charges to planewave expansion coefficients - on the source side
      allocate(tab_coefs2pw_win(npw_win,norder))
      allocate(tab_coefs2pw(npw,norder,0:nlevels))
c     tables converting planewave expansions to tensor product polynomial
c     expansion coefficients of the potentials - on the target side
      allocate(tab_pw2coefs_win(npw_win,norder))
      allocate(tab_pw2coefs(npw,norder,0:nlevels))
c      
c     compute translation matrices for PW expansions
c     translation only at the cutoff level
      nmax = 1
      ipoly=1

      allocate(pswfft_win(nexp_win),stat=ier)

      bsize = boxsize(0)
      call dmk_mk_coefs_pw_conversion_tables(ipoly,norder,npw_win,
     1    ts_win,xq,hpw_win,bsize,
     2    tab_coefs2pw_win,tab_pw2coefs_win)
      call mk_tensor_product_Fourier_transform(dim,
     1    npw_win,nfourier_win,dkernelft_win,nexp_win,pswfft_win)
      
      allocate(wpwshift(nexp,(2*nmax+1)**dim,0:nlevels),stat=ier)
      allocate(pswfft(nexp,0:nlevels),stat=ier)
      do ilev = 0,nlevels
         bsize = boxsize(ilev)
         
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
      call bdmk_compute_all_listpw(dim,nboxes,nlevels,
     1    ltree,itree,iptr,centers,boxsize,itree(iptr(1)),
     3    ifpwexp,mnlistpw,nlistpw,listpw)
c
c
c     compute list info for direct interactions
c
c     list1 contains boxes that are neighbors of the given box
      isep = 1
      call compute_mnlist1(dim,nboxes,nlevels,itree(iptr(1)),
     1    centers,boxsize,itree(iptr(3)),itree(iptr(4)),
     2    itree(iptr(5)),isep,itree(iptr(6)),itree(iptr(7)),
     3    iperiod,mnlist1)

      allocate(list1(mnlist1,nboxes),nlist1(nboxes))
      call bdmk_compute_all_modified_list1(dim,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,iperiod,
     3    mnlist1,nlist1,list1)

      nlevstart = 0
      if (iperiod.eq.1) nlevstart = 1
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

      do ilev=nlevels,0,-1
         sc=2.0d0/boxsize(ilev)
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
            istart = isrcse(1,ibox)
            npts = nboxsrcpts(ibox)
            nchild = itree(iptr(4)+ibox-1)
            if (npts.gt.0 .and. ifpwexp(ibox).eq.1) then
               iftpform = 0
               do j=1,nchild
                  jbox = itree(iptr(5)+mc*(ibox-1)+j-1)
                  if (iftensprodform(jbox).eq.1) then
                     iftpform=1
                     exit
                  endif
               enddo
               if (iftpform .eq. 1) then
                  do j=1,nchild
                     jbox = itree(iptr(5)+mc*(ibox-1)+j-1)
                     nptsj = nboxsrcpts(jbox)
                     if (iftensprodform(jbox).eq.1) then
c                       translate equivalent charges from child to parent
                        call tens_prod_trans_add(dim,ndform,norder,
     1                      coefsp(lpaddr(1,jbox)),norder,
     2                      coefsp(lpaddr(1,ibox)),
     3                      c2ptransmat(1,1,1,j))
                     elseif (nptsj .gt. 0) then
                        jstart = isrcse(1,jbox)
c                       form equivalent charges directly form sources
                        call pdmk_charge2proxycharge(dim,ndform,
     1                      norder,nptsj,sourcesort(1,jstart),
     2                      densitysort(1,jstart),centers(1,ibox),
     3                      sc,coefsp(lpaddr(1,ibox)))
                        ntot=ntot+nptsj
                        ns2tp=ns2tp+1
                     endif
                  enddo
               else
                  call pdmk_charge2proxycharge(dim,ndform,norder,
     1                npts,sourcesort(1,istart),densitysort(1,istart),
     2                centers(1,ibox),sc,coefsp(lpaddr(1,ibox)))
                  ntot=ntot+npts
                  ns2tp=ns2tp+1
               endif
               iftensprodform(ibox) = 1
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
c     Everything else is in the downward pass, but step 2
c     can be done in either direction since it is carried out
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
      if (dim.eq.2) then
         cval = 0.5d0*(1-log(rl_win))
      endif
      
      if (dim.eq.3) then
         cval = 1.0d0/rl_win
      endif

      allocate(pwexp_win(nexp_win,ndform))
      allocate(localpwexp(nexp,ndeval))
      allocate(uhat_win(nexp_win,dim,nd))
      allocate(uhat(nexp,dim,nd))
      
      ibox = 1
      call dmk_proxycharge2pw(dim,ndform,norder,
     1    coefsp(lpaddr(1,ibox)),npw_win,tab_coefs2pw_win,
     2    pwexp_win)

      if (ifstoklet .eq. 1) then
         if (iperiod.eq.0) then
            call stokesdmk_multiply_windowed_kernelFT(dim,nd,
     1          nexp_win,pwexp_win,pswfft_win,rk_win,rksq_win,
     2          npw_win,cval)
         else
            call stokesdmk_multiply_kernelFT(dim,nd,
     1          nexp_win,pwexp_win,pswfft_win,rk_win,rksq_win)
         endif
         call dmk_pw2proxypot(dim,ndeval,norder,npw_win,
     1       pwexp_win,tab_pw2coefs_win,
     2       coefsp(lpaddr(2,ibox)))
      endif

      if (ifstrslet .eq. 1) then
         call stressdmk_multiply_kernelFT(dim,nd,nexp_win,
     1       pwexp_win,pswfft_win,rk_win,rksq_win,uhat_win)
         call dmk_pw2proxypot(dim,ndeval,norder,npw_win,
     1       uhat_win,tab_pw2coefs_win,
     2       coefsp(lpaddr(2,ibox)))
      endif
      
c     
c     now the difference kernels at all levels
c      
      nb=0
      do 1100 ilev = nlevstart,nlevels
C$OMP PARALLEL DO DEFAULT (SHARED)
C$OMP$PRIVATE(ibox)
C$OMP$SCHEDULE(DYNAMIC)
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
c           Check if current box needs to form pw exp         
            if(ifpwexp(ibox).eq.1 .and. nboxsrcpts(ibox).gt.0) then
               nb=nb+1
c              form the pw expansion
               call dmk_proxycharge2pw(dim,ndform,norder,
     1             coefsp(lpaddr(1,ibox)),npw,tab_coefs2pw(1,1,ilev),
     2             rmlexp(iaddr(ibox)))
               if (ifstoklet.eq.1) then
                  call stokesdmk_multiply_kernelFT(dim,nd,
     1                nexp,rmlexp(iaddr(ibox)),
     2                pswfft(1,ilev),rk(1,1,ilev),rksq(1,ilev))
               endif
               if (ifstrslet.eq.1) then
                  call stressdmk_multiply_kernelFT(dim,nd,
     1                nexp,rmlexp(iaddr(ibox)),
     2                pswfft(1,ilev),rk(1,1,ilev),rksq(1,ilev),uhat)
                  call dmk_copy_pwexp(ndeval,nexp,uhat,
     1                rmlexp(iaddr(ibox)))
               endif
               
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
c       proxy potentials

      call cpu_time(time1)
C$    time1=omp_get_wtime()
      
      do 1300 ilev = 0,nlevels
         bsize = boxsize(ilev)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,jbox,j,ind)
C$OMP$SCHEDULE(DYNAMIC)
         do 1250 ibox = itree(2*ilev+1),itree(2*ilev+2)
c                ibox is the target box
            if (ifpwexp(ibox).eq.1 .and. nboxpts(ibox).gt.0) then
c              copy the multipole PW exp into local PW exp
c              for self interaction
               call dmk_copy_pwexp(ndeval,nexp,rmlexp(iaddr(ibox)),
     1             localpwexp)
c              translate PW expansions from neighboring source boxes
c              to the target box
               do j=1,nlistpw(ibox)
                  jbox=listpw(j,ibox)
c                 jbox is the source box
                  if ( (iperiod.eq.0) .or.
     1                ((iperiod.eq.1) .and. (ilev.gt.1)) ) then 
                     call dmk_find_pwshift_ind(dim,iperiod,
     1                   centers(1,ibox),centers(1,jbox),
     2                   bs0,bsize,nmax,ind)
                     call dmk_shiftpw(ndeval,nexp,rmlexp(iaddr(jbox)),
     1                   localpwexp,wpwshift(1,ind,ilev))
                  else
                     call find_pwshift_indices_periodic_level1(dim,
     1                   centers(1,ibox),centers(1,jbox),
     2                   bs0,bsize,nmax,inds,nitotal)

                     do i=1,nitotal
                        call dmk_shiftpw(ndeval,nexp,
     1                      rmlexp(iaddr(jbox)),localpwexp,
     2                      wpwshift(1,inds(i),ilev))
                     enddo
                  endif
               enddo
c              convert plane wave expansion into proxy potential
               call dmk_pw2proxypot(dim,ndeval,norder,npw,
     1             localpwexp,tab_pw2coefs(1,1,ilev),
     2             coefsp(lpaddr(2,ibox)))
c              translate proxy potential from parent to child
               nchild = itree(iptr(4)+ibox-1)
               if (iftensprodeval(ibox) .eq. 0) then
                  do j=1,nchild
                     jbox = itree(iptr(5) + (ibox-1)*mc+j-1)
                     if (nboxpts(jbox).gt.0) then
                        call tens_prod_trans_add(dim,ndeval,norder,
     1                      coefsp(lpaddr(2,ibox)),norder,
     2                      coefsp(lpaddr(2,jbox)),
     3                      p2ctransmat(1,1,1,j))
                     endif
                  enddo
               endif
            endif
 1250    continue
C$OMP END PARALLEL DO        
 1300 continue
c
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      timeinfo(4) = time2-time1



      
      
      


      if(ifprint .ge. 1)
     $    call prinf('=== STEP 4 (proxy potential -> pot) ===*',i,0)
c     evaluate the total contribution of the long-range part via proxypotential
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      do ilev=0,nlevels
         sc = 2.0d0/boxsize(ilev)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,nchild,istarts,iends,nptssrc)
C$OMP$PRIVATE(istartt,iendt,nptstarg)
C$OMP$SCHEDULE(DYNAMIC)  
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
            nchild = itree(iptr(4)+ibox-1)
            if (iftensprodeval(ibox).eq.1) then
               istarts = isrcse(1,ibox)
               iends = isrcse(2,ibox)
               nptssrc = iends-istarts+1
               if (ifppreg.gt.0 .and. nptssrc.gt.0) then
                  call pdmk_ortho_evalt_nd(dim,ndeval,norder,
     1                coefsp(lpaddr(2,ibox)),nptssrc,
     2                sourcesort(1,istarts),centers(1,ibox),sc,
     3                pot(1,1,istarts))
               endif

               istartt = itargse(1,ibox) 
               iendt = itargse(2,ibox)
               nptstarg = iendt-istartt + 1
               if (ifppregtarg.gt.0 .and. nptstarg.gt.0) then
                  call pdmk_ortho_evalt_nd(dim,ndeval,norder,
     1                coefsp(lpaddr(2,ibox)),nptstarg,
     2                targetsort(1,istartt),centers(1,ibox),sc,
     3                pottarg(1,1,istartt))
               endif
            endif
         enddo
C$OMP END PARALLEL DO
      enddo
      call cpu_time(time2)
C$    time2 = omp_get_wtime()      
      timeinfo(5) = time2 - time1
      
 1800 continue
      if(ifprint .ge. 1)
     $     call prinf('=== STEP 5 (direct interactions) =====*',i,0)
c
cc
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      do i=1,dim
         cshift(i) = 0.0d0
      enddo
      
      ndigits=nint(log10(1.0d0/eps)-0.1)
c     minimal value of r^2 to ignore self-interaction
c     compatible with the FMM3d, but should really rescale using
c     boxsize(0)!
      thresh2 = 1.0d-30

      nb=0
      do 2000 ilev = nlevstart,nlevels
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,jbox,istarts,iends,istartt,iendt,jstart,jend)
C$OMP$PRIVATE(n1,nptssrc,nptstarg,npts,j)
C$OMP$SCHEDULE(DYNAMIC)  
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
c           ibox is the target box here            
            istarts = isrcse(1,ibox)
            iends = isrcse(2,ibox)
            nptssrc = iends-istarts+1

            istartt = itargse(1,ibox)
            iendt = itargse(2,ibox)
            nptstarg = iendt-istartt+1

            npts = nptssrc+nptstarg
            
            n1 = nlist1(ibox)
            
            if (npts.gt.0 .and. n1.gt.0) then
               nb=nb+1
            endif

            if (npts.gt.0 .and. n1.gt.0) then
               do j=1,n1
cccc              jbox is the source box
                  jbox = list1(j,ibox)
                  
                  jstart = isrcse(1,jbox)
                  jend = isrcse(2,jbox)

                  jlev = itree(iptr(2)+jbox-1)
                  bsize = boxsize(jlev)

c     now find the interaction range of the residual kernel
                  if (ifpwexp(jbox).eq.1 .and. jbox.eq.ibox) then
c                 when ifpwexp(jbox)=1, self interaction at its own
c                 level is taken care of by plane-wave expansion
                     bsize = bsize/2
                  elseif (jlev.lt.ilev) then
c                 when the source box is bigger than the target box, 
c                 residual interaction starts from the target box level
                     bsize = boxsize(ilev)
                  endif 

c                 kernel truncated at bsize, i.e., K(x,y)=0 for |x-y|^2 > d2max
                  d2max = bsize**2
                  bsizeinv = 1.0d0/bsize
      
c                 used in the kernel approximatin for boxes in list1
                  rsc = bsizeinv*2
                  cen = -bsize/2
                  if (dim.eq.2) then
                     rsc=bsizeinv*bsizeinv*2
                     cen=-1.0d0
                  endif

                  if (iperiod.eq.1) then
                     call pdmk_find_image_shift(dim,centers(1,ibox),
     1                   centers(1,jbox),bs0,cshift)
                  endif

c                 eval at sources
                  if (nptssrc.gt.0.and.ifppreg.gt.0) then
                     call stokesdmk_direct_c(nd,dim,ikernel,cshift,
     1                   ndigits,rsc,cen,bsizeinv,
     2                   n_diag,coefs_diag,n_offd,coefs_offd,
     3                   thresh2,d2max,jstart,jend,sourcesort,
     4                   ifstoklet,stokletsort,
     5                   ifstrslet,strsletsort,strsvecsort,
     6                   istarts,iends,nsource,csourcesort,
     7                   ifppreg,pot,pre,grad)
                  endif

c                 eval at targets
                  if (nptstarg.gt.0.and.ifppregtarg.gt.0) then
                     call stokesdmk_direct_c(nd,dim,ikernel,cshift,
     1                   ndigits,rsc,cen,bsizeinv,
     2                   n_diag,coefs_diag,n_offd,coefs_offd,
     3                   thresh2,d2max,jstart,jend,sourcesort,
     4                   ifstoklet,stokletsort,
     5                   ifstrslet,strsletsort,strsvecsort,
     6                   istartt,iendt,ntarget,ctargetsort,
     7                   ifppregtarg,pottarg,pretarg,gradtarg)
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

      if (ifstoklet.eq.0) goto 4000
c     
c     finally, subtract the self-interaction from the planewave sweeping
      zero=0.0d0
      call prol0eva(zero,wprolate,psi0,derpsi0)
      call prolate_intvals(beta,wprolate,c0,c1,g0d2,c4)

c      if (dim.eq.2) then
         bsize=boxsize(0)
         rl0=rl(0)
         call stokes_windowed_kernel_value_at_zero3(dim,beta,
     1          bsize,rl0,wprolate,st2dwk0)
c      endif

      do ilev=0,nlevels
         bsize = boxsize(ilev)
cccc         if (ilev .eq. 0) bsize=bsize*0.5d0
c        sc is the value of the windowed kernel at the origin
         if (dim.eq.2) then
            jlev = ilev
c            if (ilev .eq. 0) jlev = 1
            sc = st2dwk0+0.5d0*jlev*log(2.0d0)
         elseif (dim.eq.3) then
c     for split kernel \psi(x)-1/2 x \psi'(x)
            sc = psi0/(c0*bsize)
            print *, 'true sc=',sc
            sc = st2dwk0/bsize*boxsize(0)
            print *, 'sc=',sc
c     for split kernel \psi(1+cx^2)
cccc            sc = (2.0d0/3)*(1+0.5d0*g0d2**2*beta**2)*psi0/(c0*bsize)
         endif
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
            istart = isrcse(1,ibox)
            iend = isrcse(2,ibox)
            if (ifpwexp(ibox).eq.1) then
               if (dim.eq.3) sci = sc*2
               if (dim.eq.2) sci = st2dwk0+0.5d0*(jlev+1)*log(2.0d0)
            else
               sci = sc
            endif
c           subtract the self-interaction
            nchild = itree(iptr(4)+ibox-1)
            if (nchild.eq.0) then
               do i=istart,iend
                  do k=1,dim
                  do ind=1,nd
                     pot(ind,k,i)=pot(ind,k,i)
     1                   -sci*stokletsort(ind,k,i)
                  enddo
                  enddo
               enddo
            endif
         enddo
      enddo






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
c------------------------------------------------------------------     
      subroutine stokesdmk_direct_c(nd,dim,ikernel,cshift,
     1    ndigits,rsc,cen,bsizeinv,
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
      integer ifppreg
      integer i,j,k,iffast
c
      real *8 eps,coefs_diag(*),coefs_offd(*)
      real *8 cshift(dim)
      real *8 source(dim,*)
      real *8 stoklet(nd,dim,*)
      real *8 strslet(nd,dim,*)
      real *8 strsvec(nd,dim,*)
      real *8 bsizeinv,thresh2,d2max,rsc,cen,rlambda
      real *8 ctarg(ntarget,dim),ztarg
      real *8 pot(nd,dim,*)
      real *8 pre(nd,*)
      real *8 grad(nd,dim,dim,*)

      real *8, allocatable :: sim(:,:)
c
        
      ns = iend - istart + 1
      ntarg = jend-jstart+1

      allocate(sim(dim,ns))
      do i=1,ns
         do j=1,dim
            sim(j,i)=source(j,istart+i-1)+cshift(j)
         enddo
      enddo

      iffast = 1

      if (iffast.eq.1) goto 1200

      if ((ifstoklet.eq.1).and.(ifstrslet.eq.0)) then
         if((ifppreg.eq.1)) then
            if (ikernel.eq.3 .and. dim.eq.2) then
               call st2d_local_kernel_directcp(nd,dim,rsc,cen,bsizeinv,
     1             d2max,sim,ns,stoklet(1,1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),
     3             ntarg,n_diag,coefs_diag,n_offd,coefs_offd,
     4             pot(1,1,jstart))
            endif

            if (ikernel.eq.3 .and. dim.eq.3) then
               call st3d_local_kernel_directcp(nd,dim,rsc,cen,bsizeinv,
     1             d2max,sim,ns,stoklet(1,1,istart),
     2             ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     3             ntarg,n_diag,coefs_diag,n_offd,coefs_offd,
     4             pot(1,1,jstart))
            endif
         endif
      endif

      if ((ifstoklet.eq.0).and.(ifstrslet.eq.1)) then
         if((ifppreg.eq.1)) then
            if (ikernel.eq.3 .and. dim.eq.2) then
               call st2d_local_kernel_directdp(nd,dim,rsc,cen,bsizeinv,
     1             d2max,sim,ns,
     2             strslet(1,1,istart),strsvec(1,1,istart),
     3             ctarg(jstart,1),ctarg(jstart,2),
     4             ntarg,n_diag,coefs_diag,n_offd,coefs_offd,
     5             pot(1,1,jstart))
            endif

            if (ikernel.eq.3 .and. dim.eq.3) then
               call st3d_local_kernel_directdp(nd,dim,rsc,cen,bsizeinv,
     1             d2max,sim,ns,
     2             strslet(1,1,istart),strsvec(1,1,istart),
     3             ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     4             ntarg,n_diag,coefs_diag,n_offd,coefs_offd,
     5             pot(1,1,jstart))
            endif
         endif
      endif

      return
 1200 continue

      if ((ifstoklet.eq.1).and.(ifstrslet.eq.0)) then
         if (dim.eq.3) then
            call stokes_local_kernel_directcp_fast(nd,dim,ndigits,
     1          rsc,cen,bsizeinv,thresh2,d2max,sim,ns,
     2          stoklet(1,1,istart),
     3          ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     4          ntarg,pot(1,1,jstart))
         elseif (dim.eq.2) then
            call stokes_local_kernel_directcp_fast(nd,dim,ndigits,
     1          rsc,cen,bsizeinv,thresh2,d2max,sim,ns,
     2          stoklet(1,1,istart),
     3          ctarg(jstart,1),ctarg(jstart,2),ztarg,
     4          ntarg,pot(1,1,jstart))
         endif
      endif
      
      if ((ifstoklet.eq.0).and.(ifstrslet.eq.1)) then
         if (dim.eq.3) then
            call stokes_local_kernel_directdp_fast(nd,dim,ndigits,
     1          rsc,cen,bsizeinv,thresh2,d2max,sim,ns,
     2          strslet(1,1,istart),strsvec(1,1,istart),
     3          ctarg(jstart,1),ctarg(jstart,2),ctarg(jstart,3),
     4          ntarg,pot(1,1,jstart))
         elseif (dim.eq.2) then
            call stokes_local_kernel_directdp_fast(nd,dim,ndigits,
     1          rsc,cen,bsizeinv,thresh2,d2max,sim,ns,
     2          strslet(1,1,istart),strsvec(1,1,istart),
     3          ctarg(jstart,1),ctarg(jstart,2),ztarg,
     4          ntarg,pot(1,1,jstart))
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

      if(ifstoklet.eq.0.and.ifstrslet.eq.1) then
         if(ifppreg.eq.1) then
            if (ikernel.eq.3.and.dim.eq.2) then
               call st2ddirectdp(nd,source(1,istart),
     1             strslet(1,1,istart),strsvec(1,1,istart),
     2             ns,targ(1,jstart),ntarg,
     3             pot(1,1,jstart),thresh)
            endif

            if (ikernel.eq.3.and.dim.eq.3) then
               call st3ddirectdp(nd,source(1,istart),
     1             strslet(1,1,istart),strsvec(1,1,istart),
     2             ns,targ(1,jstart),ntarg,
     3             pot(1,1,jstart),thresh)
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
      subroutine pdmk_mpalloc(ndform,dim,npw,nlevels,laddr,
     1    ifpwexp,iaddr,lmptot)
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
c     iaddr: (nboxes): pointer in rmlexp where multipole
c                      and local expansions for each
c                      box is stored.
c                      iaddr(ibox) is the
c                      starting index in rmlexp for the 
c                      multipole PW expansion of ibox.
c     lmptot      out: Integer
c                 Total length of expansions array required
c------------------------------------------------------------------

      implicit none
      integer dim
      integer nlevels,npw,ndform
      integer laddr(2,0:nlevels),ifpwexp(*)
      integer *8 iaddr(*)
      integer *8 lmptot,istart,nn,nn1,nn2,itmp,itmp2
      integer ibox,i,istarts,iends,npts
c
      nn = npw**(dim-1)*((npw+1)/2)
c     the factor 2 is the (complex *16)/(real *8) ratio
      nn1 = nn*2*ndform

c     assign memory pointers
      istart = 1
      itmp=0
      do i = 0,nlevels
         do ibox = laddr(1,i),laddr(2,i)
c          Allocate memory for the multipole PW expansions
           if (ifpwexp(ibox).eq.1) then
              iaddr(ibox) = istart + itmp*nn1
              itmp = itmp+1
           endif
         enddo
      enddo
      istart = istart + itmp*nn1
c
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
c------------------------------------------------------------------    
      subroutine pdmk_coefspalloc(ndform,ndeval,dim,norder,nlevels,
     1    laddr,ifpwexp,iftensprodeval,iaddr,lmptot)
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
      integer laddr(2,0:nlevels), ifpwexp(*), iftensprodeval(*)
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
c     Allocate memory for proxy charges
c
           if (ifpwexp(ibox).gt.0) then
              iaddr(1,ibox) = istart + itmp*nn1
              itmp = itmp+1
           endif
         enddo
      enddo
      istart = istart + itmp*nn1
c
      itmp2=0
      do i = 0,nlevels
         do ibox = laddr(1,i),laddr(2,i)
c     Allocate memory for proxy potentials
c
            if ((ifpwexp(ibox).eq.1) .or.
     1          (iftensprodeval(ibox).eq.1)) then
              iaddr(2,ibox) = istart + itmp2*nn2
              itmp2 = itmp2+1
           endif
         enddo
      enddo
      istart = istart + itmp2*nn2

      lmptot = istart

      return
      end
c
c
c
c
      subroutine pdmk_find_image_shift(ndim,tcenter,scenter,
     1    bs0,shift)
c     returns the center shift of the image cell for the PBCs
c
c     input:
c     ndim - dimension of the underlying space
c     tcenter - target box center
c     scenter - source box center
c     bs0 - root box size
c
c     output
c     shift - source center shift
c     new source center = scenter + shift 
c
c
      implicit real *8 (a-h,o-z)
      real *8 tcenter(ndim),scenter(ndim),shift(ndim)


      do i=1,ndim
         dx = tcenter(i)-scenter(i)
         shift(i)=0
         dxp1=dx-bs0
         dxm1=dx+bs0
         if (abs(dx).gt.abs(dxp1)) then
            dx=dxp1
            shift(i)=bs0
         endif
         if (abs(dx).gt.abs(dxm1)) shift(i)=-bs0
      enddo
      
      
      return
      end subroutine

c     Modified based on bdmk3.f on 06/18/2024
c     added gradient and hessian calculation
c     Last modified by Shidong Jiang on 09/25/2024
c      
      subroutine bdmk(nd,ndim,eps,ikernel,beta,ipoly,norder,
     1    npbox,nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     2    fvals,ifpgh,pot,grad,hess,ntarg,targs,
     3    ifpghtarg,pote,grade,hesse,tottimeinfo)
c     
c
c     This code computes the volume potential on a box for densities
c     defined on a tensor product grid of each leaf node in an adaptive tree.
c
c     input
c     nd - integer
c          number of right hand sides
c     ndim - integer
c           dimension of the underlying space
c     eps - double precision
c           precision requested
c     ikernel - integer
c            0: the Yukawa kernel; 1: the Laplace kernel; 2: the square-root Laplace kernel. 
c     beta - double precision
c            either the parameter in the Yukawa kernel or the exponent of the power
c            function kernel
c     ipoly - integer
c            0: Legendre polynomials
c            1: Chebyshev polynomials
c     norder - integer
c           order of expansions for input function value array
c     npbox - integer
c           number of points per box where potential is to be dumped = (norder**ndim)
c     fvals - double precision (nd,npbox,nboxes)
c           function values tabulated on a tensor grid in each leaf node
c     ifpgh   : flag for computing pot/grad/hess
c                   ifpgh = 1, only potential is computed
c                   ifpgh = 2, potential and gradient are computed
c                   ifpgh = 3, potential, gradient, and hessian 
c                   are computed
c     ifpghtarg: flag for computing pottarg/gradtarg/hesstarg
c                    ifpghtarg = 1, only potential is computed at targets
c                    ifpghtarg = 2, potential and gradient are 
c                    computed at targets
c                    ifpghtarg = 3, potential, gradient, and hessian are 
c                    computed at targets
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
c     pote - double precision (nd,ntarg)
c            volume potential at targets
c     grade - double precision (nd,ndim,ntarg)
c            gradient of the volume potential at targets
c     hesse - double precision (nd,ndim*(ndim+1)/2,ntarg)
c            hessian of the volume potential at targets
c
      implicit none
      real *8 eps,beta
      integer nd,ndim,nboxes,nlevels,ntarg
      integer ikernel,iperiod,ifpgh,ifpghtarg
      integer iptr(8),ltree
      integer itree(ltree),norder,npbox
      real *8 targs(ndim,ntarg)
      real *8 fvals(nd,npbox,nboxes)

      real *8 pot(nd,npbox,nboxes)
      real *8 grad(nd,ndim,npbox,*)
      real *8 hess(nd,ndim*(ndim+1)/2,npbox,*)

      real *8 pote(nd,ntarg)
      real *8 grade(nd,ndim,*)
      real *8 hesse(nd,ndim*(ndim+1)/2,*)

      real *8 centers(ndim,nboxes)
      real *8 boxsize(0:nlevels)
      real *8 tottimeinfo(*)
      real *8 timeinfo(20,-100:20)


      real *8 umat(norder,norder)
      real *8 vmat(norder,norder)
      real *8 vpmat(norder,norder)
      real *8 vppmat(norder,norder)
      real *8 umat_nd(norder,norder,ndim)

      real *8, allocatable :: fcoefs(:,:,:)
      real *8, allocatable :: flcoefs(:,:,:)
      real *8, allocatable :: coefsp(:,:,:)
      real *8, allocatable :: flvals(:,:,:)
      real *8, allocatable :: fl2vals(:,:,:)
c     gradient and the Laplacian of the gradient of the density
      real *8, allocatable :: gcoefs(:,:,:,:)
      real *8, allocatable :: gvals(:,:,:,:)
      real *8, allocatable :: glvals(:,:,:,:)
c     hessian and the Laplacian of the hessian of the density
      real *8, allocatable :: hcoefs(:,:,:,:)
      real *8, allocatable :: hvals(:,:,:,:)
      real *8, allocatable :: hlvals(:,:,:,:)


      real *8 wdeltas(200),deltas(200),pval(200)
      real *8 ts0(200),ws0(200),xs(200),whts(200),whtsp(200)
      integer npwlevels(200)
      integer ipwaddr(2,-100:nlevels+1),ipw(-100:nlevels+1)
      integer porder,ncbox,mc,ipoly0
      integer nhess,mnbors,npw,npw0,nexp,nexp0,npwlevel,npwlevel0
      integer nasym,nlevend
      integer isep,mnlist1,mrefinelev
      
      real *8, allocatable :: proxycharge(:,:,:)
      real *8, allocatable :: proxypotential(:,:,:)
      real *8, allocatable :: proxygradient(:,:,:,:)
      real *8, allocatable :: proxyhessian(:,:,:,:)
      real *8, allocatable :: den2pcmat(:,:),potevalmat(:,:)
      integer, allocatable :: isgn(:,:)
      real *8, allocatable :: umatp(:,:),umatp_nd(:,:,:)
      real *8, allocatable :: vmatp(:,:),vtmp(:,:),vtmp2(:,:)
      real *8, allocatable :: umat0(:,:),vmat0(:,:)
      real *8, allocatable :: ts(:,:),ws(:,:)
      real *8, allocatable :: p2ctransmat(:,:,:,:)
      real *8, allocatable :: c2ptransmat(:,:,:,:)
      real *8, allocatable :: whtsnd(:),fint(:)


c     direction interaction list
      integer, allocatable :: nlist1(:),list1(:,:)
c     plane wave interaction list
      integer, allocatable :: nlistpw(:), listpw(:,:)
c     box flag array
      integer, allocatable :: ifpwexp(:)

      integer ixyz(ndim)
      integer idelta(0:nlevels)
      
c     1d direct evaluation tables
      real *8, allocatable :: tab_loc(:,:,:,:,:)
      real *8, allocatable :: tabx_loc(:,:,:,:,:)
      real *8, allocatable :: tabxx_loc(:,:,:,:,:)
      integer, allocatable :: ind_loc(:,:,:,:,:)
      
      integer *8, allocatable :: iaddr(:,:)
      integer *8 lmptot,lmptot0
      real *8, allocatable :: rmlexp(:),rmlexp0(:)

      complex *16, allocatable :: tab_coefs2pw0(:,:)
      complex *16, allocatable :: tab_pw2pot0(:,:)
      complex *16, allocatable :: tab_pw2potx0(:,:)
      complex *16, allocatable :: tab_pw2potxx0(:,:)
      
      complex *16, allocatable :: tab_coefs2pw(:,:,:)
      complex *16, allocatable :: tab_pw2pot(:,:,:)
      complex *16, allocatable :: tab_pw2potx(:,:,:)
      complex *16, allocatable :: tab_pw2potxx(:,:,:)
      
c     planewave expansion weights
      real *8, allocatable :: wpwexp0(:),wpwexp(:,:)
c     outoging plane-wave to incoming plane-wave shift matrices
      complex *16, allocatable :: wpwshift(:,:,:)

      integer i,j,k,ilev,jlev,klev,ibox,jbox,gbox,hbox,ind,n,m
      integer ndg,ndh
      integer istart,iend,jstart,jend
      integer ndeltas,ntaylor,norder2
      integer porder2,ngs,itype,ipoly,nchild
      integer nloctab,nloctab2,nnodes,nmax,mnlistpw,nl1
      integer id
      integer ifprint,ier,ifexpon,nlevstart
      
      real *8 fv(0:10),c(0:10)
      real *8 dlogr0,r2,r4,r6,dk1,dk0
      real *8 pi,r0,de,br,br2,br3,br4,br5,br6,br7,br8,br9
      real *8 b2,b4,b6,ebr,delta
      real *8 sc,dd,vinttrue,beta2,vintcomp,derr,dwnorm
      real *8 c0,eulergamma
      real *8 time1,time2

      real *8 bs0,hpw,bs,bsize,xmin,asymerr
      
      complex *16 zk,ima,ztmp,zh0,zh1
      data ima/(0.0d0,1.0d0)/
      data pi/3.1415926535 8979323846 2643383279 5028841971 693993751d0/
      data eulergamma/0.5772156649015328606065120900824024310421593d0/
      real *8 omp_get_wtime
      
      ifprint=1
      iperiod=0
      
      do i=1,20
         tottimeinfo(i)=0
      enddo

      if(ifprint .ge. 1)
     $    call prinf('=== STEP 1 (Precomputation) ===*',i,0)
      call cpu_time(time1)
C$    time1=omp_get_wtime()

c     r0 is the cutoff length, i.e., for r>r0, 1/r is well apprxoimated by
c     sum of Gaussians. Thus, one only needs to compute the correction for
c     r\in [0,r0]
      call get_sognodes(ndim,ikernel,eps,boxsize(0),nlevels,norder,beta,
     1    r0,ndeltas,wdeltas,deltas)

cccc      if(ifprint .ge. 1) then
cccc         call prin2('r0=*',r0,1)
cccc         call prin2('deltas=*',deltas,ndeltas)
cccc         call prin2('wdeltas=*',wdeltas,ndeltas)
cccc         call prin2('boxsize=*',boxsize,nlevels+1)
cccc      endif
      
      nlevstart=-100
      do ilev=nlevstart,nlevels
         ipw(ilev)=0
      enddo
      
      do i=1,ndeltas
         call find_npwlevel(eps,nlevels,boxsize,deltas(i),npwlevels(i))
         ipw(npwlevels(i)) = ipw(npwlevels(i))+1
      enddo

      istart=1
      ipwaddr(1,nlevstart)=1
      ipwaddr(2,nlevstart)=ipw(nlevstart)
      
      do ilev=nlevstart+1,nlevels
         istart=ipwaddr(2,ilev-1)+1
         ipwaddr(1,ilev)=istart
         ipwaddr(2,ilev)=istart+ipw(ilev)-1
      enddo
cccc      call prinf('npwlevels=*',npwlevels,ndeltas)
cccc      call prinf('ipw=*',ipw,nlevels-nlevstart+2)
cccc      call prinf('ipwaddr=*',ipwaddr,2*(nlevels-nlevstart+2))
      
c     polynomial order for proxy charges and potentials
      if (eps.ge.0.8d-3) then
         porder=16
      elseif (eps.ge.0.8d-4) then
         porder=22
      elseif (eps.ge.0.8d-5) then
         porder=26
      elseif (eps.ge.0.8d-6) then
         porder=30
      elseif (eps.ge.0.8d-7) then
         porder=36
      elseif (eps.ge.0.8d-8) then
         porder=42
      elseif (eps.ge.0.8d-9) then
         porder=46
      elseif (eps.ge.0.8d-10) then
         porder=50
      elseif (eps.ge.0.8d-11) then
         porder=56
      elseif (eps.ge.0.8d-12) then
         porder=62
      endif
      
      ncbox=porder**ndim
      nhess=ndim*(ndim+1)/2
      ndg = nd*ndim
      ndh = nd*nhess
      
      allocate(proxycharge(ncbox,nd,nboxes),stat=ier)
      if(ier.ne.0) then
         print *, "Cannot allocate workspace for proxy charges"
         print *, "length=", ncbox*nd*nboxes
         ier = 4
         return
      endif
      
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,j,ind)
      do ibox=1,nboxes
         do ind=1,nd
            do j=1,ncbox
               proxycharge(j,ind,ibox)=0
            enddo
         enddo
      enddo
C$OMP END PARALLEL DO         

      allocate(proxypotential(ncbox,nd,nboxes),stat=ier)
      if(ier.ne.0) then
         print *, "Cannot allocate workspace for proxy potential"
         print *, "length=", ncbox*nd*nboxes
         ier = 4
         return
      endif

C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,j,ind)
      do ibox=1,nboxes
         do ind=1,nd
            do j=1,ncbox
               proxypotential(j,ind,ibox)=0
            enddo
         enddo
      enddo
C$OMP END PARALLEL DO         

      if (ifpgh.ge.2) then
         allocate(proxygradient(ncbox,nd,ndim,nboxes),stat=ier)
         if(ier.ne.0) then
            print *, "Cannot allocate workspace for proxy gradient"
            print *, "length=", ncbox*nd*nboxes*ndim
            ier = 4
            return
         endif
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,j,k,ind)
         do ibox=1,nboxes
            do k=1,ndim
               do ind=1,nd
                  do j=1,ncbox
                     proxygradient(j,ind,k,ibox)=0
                  enddo
               enddo
            enddo
         enddo
C$OMP END PARALLEL DO         
      else
         allocate(proxygradient(ncbox,nd,ndim,1),stat=ier)
      endif

      if (ifpgh.ge.3) then
         allocate(proxyhessian(ncbox,nd,nhess,nboxes),stat=ier)
         if(ier.ne.0) then
            print *, "Cannot allocate workspace for proxy hessian"
            print *, "length=", ncbox*nd*nboxes*nhess
            ier = 4
            return
         endif

C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,j,k,ind)
         do ibox=1,nboxes
            do k=1,nhess
               do ind=1,nd
                  do j=1,ncbox
                     proxyhessian(j,ind,k,ibox)=0
                  enddo
               enddo
            enddo
         enddo
C$OMP END PARALLEL DO
      else
         allocate(proxyhessian(ncbox,nd,nhess,1),stat=ier)
      endif
      
      bs0 = boxsize(0)
      mc = 2**ndim
      mnbors=3**ndim
c     always use Legendre polynomials for proxy charges since we need to do integration
      ipoly0=0

      allocate(isgn(ndim,mc))
      call get_child_box_sign(ndim,isgn)
      
      allocate(p2ctransmat(porder,porder,ndim,mc))
      allocate(c2ptransmat(porder,porder,ndim,mc))
      call dmk_get_coefs_translation_matrices(ndim,ipoly0,
     1    porder,isgn,p2ctransmat,c2ptransmat)

c     compute 1D density-to-proxy-charge transformation matrix
      allocate(den2pcmat(porder,norder))

      allocate(umatp(porder,porder))
      allocate(umatp_nd(porder,porder,ndim))
      allocate(vmatp(porder,porder))

      allocate(vtmp(norder,porder))

      itype=2
      if (ipoly0.eq.0) then
         call legeexps(itype,porder,xs,umatp,vmatp,whtsp)
      elseif (ipoly0.eq.1) then
         call chebexps(itype,porder,xs,umatp,vmatp,whtsp)
      endif

      porder2=porder*porder
      do i=1,ndim
         do j=1,porder
            do k=1,porder
               umatp_nd(k,j,i)=umatp(j,k)
            enddo
         enddo
      enddo
      
      if (ipoly.eq.0) then
         do j=1,porder
            call legepols(xs(j),norder-1,vtmp(1,j))
         enddo
      elseif (ipoly.eq.1) then
         do j=1,porder
            call chebpols(xs(j),norder-1,vtmp(1,j))
         enddo
      endif 

      do i=1,norder
         do j=1,porder
            vtmp(i,j)=vtmp(i,j)*whtsp(j)
         enddo
      enddo
      
      do i=1,norder
         do j=1,porder
            dd=0
            do k=1,porder
               dd=dd+vtmp(i,k)*vmatp(k,j)
            enddo
            den2pcmat(j,i)=dd
         enddo
      enddo

c     compute 1D potential evaluation matrix, used in proxypotential to pot eval routine
      allocate(potevalmat(norder,porder))
      itype=2
      if (ipoly.eq.0) then
         call legeexps(itype,norder,xs,umat,vmat,whts)
      elseif (ipoly.eq.1) then
         call chebexps(itype,norder,xs,umat,vmat,whts)
      endif

c     needed for the logarithmic kernel log(r)
      if (ndim.eq.2 .and. ikernel.le.1) then
         allocate(whtsnd(norder*norder))
         k=0
         do j=1,norder
            do i=1,norder
               k=k+1
               whtsnd(k)=whts(i)*whts(j)
            enddo
         enddo
      endif
      
      do j=1,porder
         do i=1,norder
            call legepols(xs(i),porder-1,pval)
            potevalmat(i,j)=pval(j)
         enddo
      enddo

      allocate(umat0(porder,porder),vmat0(porder,porder))
      itype = 2
      call legeexps(itype,porder,xs,umat0,vmat0,whts)
      
c     check whether we need to create and evaluate planewave expansions 
c     for boxes
      allocate(ifpwexp(nboxes))
      call bdmk_find_all_pwexp_boxes(ndim,nboxes,
     1    nlevels,ltree,itree,iptr,iperiod,ifpwexp)


c     compute the tables converting Legendre polynomial expansion to potential
c     values, used in direct evaluation
c
c     no tree refinement, the tree is the usual level-restricted tree
      mrefinelev=0
      nloctab=2**(mrefinelev+1)*(mrefinelev+3)

cccc      if (ifprint.eq.1) call prinf('nloctab=*',nloctab,1)
      nloctab2=2*nloctab+1
      allocate(  tab_loc(norder,norder,nloctab2,ndeltas,0:nlevels))
      allocate( tabx_loc(norder,norder,nloctab2,ndeltas,0:nlevels))
      allocate(tabxx_loc(norder,norder,nloctab2,ndeltas,0:nlevels))
      allocate(ind_loc(2,norder+1,nloctab2,ndeltas,0:nlevels))
      nnodes=50
      do ilev = 0,nlevels
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(id,delta)
      do id=1,ndeltas
         delta=deltas(id)
         call mk_loctab_all(eps,ipoly,norder,nnodes,delta,
     1       boxsize(ilev),mrefinelev,nloctab,tab_loc(1,1,1,id,ilev),
     2       tabx_loc(1,1,1,id,ilev),tabxx_loc(1,1,1,id,ilev),
     3       ind_loc(1,1,1,id,ilev))
      enddo
C$OMP END PARALLEL DO         
      enddo

      npwlevel=0
      call bdmk_pwterms(eps,npwlevel,npw)
      allocate(tab_coefs2pw(npw,porder,0:nlevels))
      allocate(tab_pw2pot(npw,porder,0:nlevels))
      allocate(tab_pw2potx(npw,porder,0:nlevels))
      allocate(tab_pw2potxx(npw,porder,0:nlevels))
      nexp=(npw+1)/2
      do i=1,ndim-1
         nexp = nexp*npw
      enddo
      allocate(ts(npw,0:nlevels),ws(npw,0:nlevels))
      allocate(wpwexp(nexp,0:nlevels))

c     diagonal multipole to local plane wave translation matrices
      nmax = 1
      allocate(wpwshift(nexp,(2*nmax+1)**ndim,0:nlevels))

C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ilev,istart,iend,ngs,bsize,hpw,xmin)
      do ilev=0,nlevels
         istart=ipwaddr(1,ilev)
         iend=ipwaddr(2,ilev)
         ngs=iend-istart+1
         bsize=boxsize(ilev)
         if (ngs.gt.0) then
c     get planewave nodes and weights
            call get_pwnodes_md(eps,nlevels,ilev,npw,
     1          ws(1,ilev),ts(1,ilev),bs0)

            call dmk_mk_coefs_pw_tables_pgh(ipoly0,porder,npw,
     1          ts(1,ilev),xs,hpw,bsize,
     2          tab_coefs2pw(1,1,ilev),tab_pw2pot(1,1,ilev),
     3          tab_pw2potx(1,1,ilev),tab_pw2potxx(1,1,ilev))
            
            call mk_kernel_Fourier_transform(ndim,ngs,deltas(istart),
     1          wdeltas(istart),npw,ws(1,ilev),ts(1,ilev),
     2          nexp,wpwexp(1,ilev))

            xmin = boxsize(ilev)
            call mk_pw_translation_matrices(ndim,xmin,npw,ts(1,ilev),
     1          nmax,wpwshift(1,1,ilev))
         endif
      enddo
C$OMP END PARALLEL DO         

c     Multipole and local planewave expansions will be held in workspace
c     in locations pointed to by array iaddr(2,nboxes).
      allocate(iaddr(2,nboxes))
c     calculate memory needed for multipole and local planewave expansions
      call bdmk_mpalloc(nd,ndim,npw,nlevels,itree,
     1    ifpwexp,iaddr,lmptot)
      if(ifprint .eq. 1)
     1  call prin2('memory for planewave expansions=*',lmptot*1.0d0,1)
cccc      if(ifprint .eq. 1) call prin2('lmptot is *',lmptot*1.0d0,1)
      allocate(rmlexp(lmptot),stat=ier)
      if(ier.ne.0) then
         print *, "Cannot allocate workspace for plane wave expansions"
         print *, "lmptot=", lmptot
         ier = 4
         return
      endif

c
c     compute list info for plane-wave sweeping
c
      mnlistpw = 3**ndim
      allocate(nlistpw(nboxes),listpw(mnlistpw,nboxes))
c     listpw contains source boxes in the pw interaction
      call bdmk_compute_all_listpw(ndim,nboxes,nlevels,
     1    ltree,itree,iptr,centers,boxsize,itree(iptr(1)),
     3    ifpwexp,mnlistpw,nlistpw,listpw)      
c
c
c     compute list info for direct interactions
c
c     list1 contains boxes that are neighbors of the given box
      isep = 1
      call compute_mnlist1(ndim,nboxes,nlevels,itree(iptr(1)),
     1    centers,boxsize,itree(iptr(3)),itree(iptr(4)),
     2    itree(iptr(5)),isep,itree(iptr(6)),itree(iptr(7)),
     3    iperiod,mnlist1)

      allocate(list1(mnlist1,nboxes),nlist1(nboxes))
      call bdmk_compute_all_modified_list1(ndim,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,iperiod,
     3    mnlist1,nlist1,list1)
c

c     find the number of deltas that can be handled by asymptotic expansions at each level
      nasym=3
      do ilev=0,nlevels
         idelta(ilev)=0
      enddo

      do ilev=0,nlevels
         do k=ndeltas,1,-1
            delta=deltas(k)
c           empirical error formula for asymptotic expansions
            asymerr=(delta*4/boxsize(ilev)**2)**nasym
            if (asymerr .le. eps) then
               idelta(ilev)=idelta(ilev)+1
            endif
         enddo
      enddo
      call prinf('ndeltas=*',ndeltas,1)
      call prinf('idelta=*',idelta,nlevels+1)
      


      
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      tottimeinfo(1)=time2-time1








      
      if(ifprint .ge. 1)
     $    call prinf('=== STEP 2 (local Taylor expansions) ===*',i,0)
      call cpu_time(time1)
C$    time1=omp_get_wtime()
      
c     for the log kernel, there is a nonzero constant mode
      if (ndim.eq.2 .and. ikernel.eq.1) then
         c0=0
         do i=1,ndeltas
            c0=c0-wdeltas(i)*exp(-1.0d0/deltas(i))
         enddo
         call prin2('c0=*',c0,1)
      endif
      

      ntaylor=2
c     contributions from the original kernel, which obviously depends on
c     the specific kernel
      if (ndim.eq.3 .and. ikernel.eq.0) then
c        c(k) = int_0^r0 exp(-beta*r)/r * r^(2k)*r^2 dr
         br=beta*r0
         br2=br*br
         br3=br2*br
         br4=br3*br
         br5=br4*br
         br6=br5*br
         br7=br6*br
         br8=br7*br
         br9=br8*br
         
         b2=beta*beta
         b4=b2*b2
         b6=b2*b4
         ebr=exp(-br)
         if (br.gt.1d-3) then
            c(0)=1/b2-ebr*(br+1)/b2
            c(1)=6/b4-ebr*(br3+3*br2+6*br+6)/b4
            c(2)=120/b6-ebr*(br5+5*br4+20*br3+60*br2+120*br+120)/b6
         else
            c(0)=(br2/2-br3/3+br4/8-br5/30+br6/144)/b2
            c(1)=(br4/4-br5/5+br6/12-br7/42)/b4
            c(2)=(br6/6-br7/7+br8/16-br9/54)/b6
         endif
      elseif (ndim.eq.2 .and. ikernel.eq.0) then
c        c(k) = int_0^r0 K_0(beta,r) r r^(2k) dr
         zk = ima*beta
         ztmp = zk*r0
         ifexpon=1
         call hank103(ztmp,zh0,zh1,ifexpon)
c        K_0(beta*r0)
         dk0 = dble(0.5d0*pi*ima*zh0)
c        K_1(beta*r0)
         dk1 = dble(-0.5d0*pi*zh1)
         
         br=beta*r0
         br2=br*br
         br3=br2*br
         br4=br3*br
         br5=br4*br
         br6=br5*br
         
         b2=beta*beta
         b4=b2*b2
         b6=b2*b4

         dd = eulergamma+log(br/2)
         if (br.le.1.0d-3) then
c-x^2*(eulergamma/2-log(2)/2+log(x)/2- 1/4)-x^4*(eulergamma/16 - log(2)/16 + log(x)/16 - 5/64)
             
            c(0)=-(br2*(dd-0.5d0)/2+br4*(dd-5.0d0/4)/16)/b2
         else
            c(0)=(1-br*dk1)/b2
         endif


         if (br.le.1.0d-3) then
            c(1)=-((dd-0.25d0)*br4/4 + (dd-7.0d0/6)*br6/24)/b4
         else
            c(1)=(4-2*br2*dk0-br*(br2+4)*dk1)/b4            
         endif


      elseif (ndim.eq.3 .and. ikernel.eq.1) then
c        c(k) = int_0^r0 r^(-1) r^2 r^(2k) dr
         do k=0,ntaylor
            de=2*k+2
            c(k)=r0**de/de
         enddo
      elseif (ndim.eq.2 .and. ikernel.eq.1) then
c        c(k) = int_0^r0 log(r) r r^(2k) dr
         dlogr0=log(r0)
         r2=r0*r0
         r4=r2*r2
         r6=r2*r4
c         c(0) = r2*(dlogr0-0.5d0)/2-c0*r2/2
c         c(1) = r4*(dlogr0-0.25d0)/4-c0*r4/4
c         c(2) = r6*(dlogr0-1.0d0/6)/6-c0*r6/6
         c(0) = r2*(dlogr0-0.5d0-c0)/2
         c(1) = r4*(dlogr0-0.25d0-c0)/4
         c(2) = r6*(dlogr0-1.0d0/6-c0)/6
      elseif (ikernel.eq.2) then
c        c(k) = int_0^r0 r^(-2) r^(2k) r^2dr for 3D
c        c(k) = int_0^r0 r^(-1) r^(2k) rdr for 2D
c        happans to be the same value
         do k=0,ntaylor
            de=2*k+1
            c(k)=r0**de/de
         enddo
      endif
      call prin2('c=*',c,ntaylor+1)
c
c     now subtract the contribution from each Gaussian
c
      do i=1,ndeltas
         delta=deltas(i)
         call faddeevan(ndim,ntaylor,r0,delta,fv)
         
         do k=0,ntaylor
            c(k)=c(k)-wdeltas(i)*fv(k)
         enddo
      enddo

c     finally, multiplied them by proper constants
c     these constants depends only on the dimension of the underlying space
      if (ndim.eq.2) then
c        2 pi is the perimeter of the unit circle
         c(0)=c(0)*2*pi
c        2=2!, and 1/2 comes from the fact, say, x^2 has 1/2 contribution of r^2
         c(1)=c(1)*2*pi/2/2
cccc         c(1)=c(1)*2*pi/2
c        24=4!, and 3/8 comes from the fact, say, x^4 has 3/8 contribution of r^4
c         c(2)=c(2)*2*pi/24*3/8
         c(2)=0
c        7!=5040
         c(3)=c(3)*2*pi/720*5/16
      elseif (ndim.eq.3) then
c        4 pi is the surface area of the unit sphere
         c(0)=c(0)*4*pi
c        2=2!, and 1/3 comes from the fact, say, x^2 has 1/3 contribution of r^2
         c(1)=c(1)*4*pi/2/3
c        24=4!, and 1/5 comes from the fact, say, z^4 has 1/5 contribution of r^4
         c(2)=c(2)*4*pi/24/5
c        7!=5040
         c(3)=c(3)*4*pi/5040
      endif

      call prin2('after substracting gaussian contri, c=*',c,ntaylor+1)
      
      call ortho_eval_tables(ipoly,norder,umat,vmat,vpmat,vppmat)
      norder2=norder*norder
      do i=1,ndim
         call dcopy_f77(norder2,umat,1,umat_nd(1,1,i),1)
      enddo

      
      
      allocate(fcoefs(nd,npbox,nboxes))
      allocate(flvals(nd,npbox,nboxes))
      allocate(flcoefs(nd,npbox,nboxes))
      allocate(fl2vals(nd,npbox,nboxes))
c      allocate(fl3vals(nd,npbox,nboxes))

      if (ifpgh.ge.2) then
         allocate(gcoefs(nd,ndim,npbox,nboxes))
         allocate(gvals(nd,ndim,npbox,nboxes))
         allocate(glvals(nd,ndim,npbox,nboxes))
      endif

      if (ifpgh.ge.3) then
         allocate(hcoefs(nd,nhess,npbox,nboxes))
         allocate(hvals(nd,nhess,npbox,nboxes))
         allocate(hlvals(nd,nhess,npbox,nboxes))
      endif

      do ilev = 0,nlevels
        sc=2.0d0/boxsize(ilev)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,j,k,ind,nchild)
C$OMP$SCHEDULE(DYNAMIC)  
        do ibox = itree(2*ilev+1),itree(2*ilev+2)
          nchild = itree(iptr(4) + ibox-1)
          if(nchild.eq.0) then
c            compute the Laplacian of f
             call tens_prod_trans_vec(ndim,nd,norder,
     1           fvals(1,1,ibox),norder,fcoefs(1,1,ibox),umat_nd)
             call ortho_eval_laplacian_nd(ndim,nd,norder,
     1           fcoefs(1,1,ibox),sc,flvals(1,1,ibox),vmat,vppmat)
c            compute the BiLaplacian of f 
             call tens_prod_trans_vec(ndim,nd,norder,
     1           flvals(1,1,ibox),norder,flcoefs(1,1,ibox),umat_nd)
             call ortho_eval_laplacian_nd(ndim,nd,norder,
     1           flcoefs(1,1,ibox),sc,fl2vals(1,1,ibox),vmat,vppmat)

             if (ifpgh.eq.2) then
c            compute gradient of f
                call ortho_evalg_nd(ndim,nd,norder,fcoefs(1,1,ibox),sc,
     1              gvals(1,1,1,ibox),vmat,vpmat)
c            compute the Laplacian of gradient of f 
                call tens_prod_trans_vec(ndim,ndg,norder,
     1              gvals(1,1,1,ibox),norder,gcoefs(1,1,1,ibox),umat_nd)
                call ortho_eval_laplacian_nd(ndim,ndg,norder,
     1              gcoefs(1,1,1,ibox),sc,glvals(1,1,1,ibox),
     2              vmat,vppmat)
             endif

             if (ifpgh.eq.3) then
                call ortho_evalgh_nd(ndim,nd,norder,fcoefs(1,1,ibox),sc,
     1              gvals(1,1,1,ibox),
     1              hvals(1,1,1,ibox),vmat,vpmat,vppmat)
c            compute the Laplacian of gradient of f
                call tens_prod_trans_vec(ndim,ndg,norder,
     1              gvals(1,1,1,ibox),norder,gcoefs(1,1,1,ibox),umat_nd)
                call ortho_eval_laplacian_nd(ndim,ndg,norder,
     1              gcoefs(1,1,1,ibox),sc,glvals(1,1,1,ibox),
     2              vmat,vppmat)
c            compute the Laplacian of hessian of f
                call tens_prod_trans_vec(ndim,ndh,norder,
     1              hvals(1,1,1,ibox),norder,hcoefs(1,1,1,ibox),umat_nd)
                call ortho_eval_laplacian_nd(ndim,ndh,norder,
     1              hcoefs(1,1,1,ibox),sc,hlvals(1,1,1,ibox),
     2              vmat,vppmat)
             endif
                
c     compute the TriLaplacian of f
c     Note: this is unreliable since the condition number of spectral differentiation
c     is (norder^2)^6=norder^12.
c             call ortho_trans_nd(ndim,nd,norder,
c     1           fl2vals(1,1,ibox),fcoefs,umat_nd)
c             call ortho_eval_laplacian_nd(ndim,nd,norder,fcoefs,
c     1           sc,fl3vals(1,1,ibox),vmat,vppmat)
             do j=1,npbox
             do ind=1,nd
                pot(ind,j,ibox)=c(0)*fvals(ind,j,ibox)
     1              +c(1)*flvals(ind,j,ibox)+c(2)*fl2vals(ind,j,ibox)
cccc  2                 +c(3)*fl3vals(ind,j,ibox)
              enddo
              enddo

              if (ifpgh.ge.2) then
                 do j=1,npbox
                 do k=1,ndim
                 do ind=1,nd
                    grad(ind,k,j,ibox)=c(0)*gvals(ind,k,j,ibox)
     1                  +c(1)*glvals(ind,k,j,ibox)
                 enddo
                 enddo
                 enddo
              endif

              if (ifpgh.eq.3) then
                 do j=1,npbox
                 do k=1,nhess
                 do ind=1,nd
                    hess(ind,k,j,ibox)=c(0)*hvals(ind,k,j,ibox)
     1                  +c(1)*hlvals(ind,k,j,ibox)
                 enddo
                 enddo
                 enddo
              endif
          endif
        enddo
C$OMP END PARALLEL DO         
      enddo

      call cpu_time(time2)
C$    time2=omp_get_wtime()  
      tottimeinfo(2)=time2-time1






      


      if(ifprint .ge. 1)
     $    call prinf('=== STEP 3 (proxy charge evaluation) ===*',i,0)
c
c     upward pass for calculating proxy charges
c
      call cpu_time(time1)
C$    time1=omp_get_wtime()
      
      do ilev=nlevels,0,-1
         sc=boxsize(ilev)/2.0d0
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,j,jbox,nchild)
C$OMP$SCHEDULE(DYNAMIC)  
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
            nchild = itree(iptr(4)+ibox-1)
            if (nchild.eq.0) then
               call bdmk_density2proxycharge(ndim,nd,
     1             norder,fcoefs(1,1,ibox),porder,
     2             proxycharge(1,1,ibox),den2pcmat,sc)
            else
c     translate proxy charges from child to parent
               do j=1,nchild
                  jbox = itree(iptr(5)+mc*(ibox-1)+j-1)
                  call tens_prod_trans_add(ndim,nd,porder,
     1                proxycharge(1,1,jbox),porder,
     2                proxycharge(1,1,ibox),
     3                c2ptransmat(1,1,1,j))
               enddo
            endif
         enddo
C$OMP END PARALLEL DO         
      enddo
      call cpu_time(time2)
C$    time2=omp_get_wtime()  
      tottimeinfo(3)=time2-time1




      
c     Downward pass
c
c        step 4: convert function values to planewave expansions
c
    
      if(ifprint.eq.1) 
     1   call prinf("=== STEP 4 (values -> mp pwexps) ===*",i,0)
      
      call cpu_time(time1)
C$    time1=omp_get_wtime()

c     first, deal with all fat Gaussians on the root box
      ibox=1
      do ilev=nlevstart,-1
         istart=ipwaddr(1,ilev)
         iend=ipwaddr(2,ilev)
         ngs=iend-istart+1
         
         if (ngs .gt. 0) then
            npwlevel0=ilev
c         get planewave nodes and weights
            call bdmk_pwterms(eps,npwlevel0,npw0)
            call get_pwnodes_md(eps,nlevels,npwlevel0,npw0,ws0,ts0,bs0)

            allocate(tab_coefs2pw0(npw0,porder))
            allocate(tab_pw2pot0(npw0,porder))
            allocate(tab_pw2potx0(npw0,porder))
            allocate(tab_pw2potxx0(npw0,porder))
            call dmk_mk_coefs_pw_tables_pgh(ipoly0,porder,npw0,
     1          ts0,xs,hpw,bs0,tab_coefs2pw0,tab_pw2pot0,
     2          tab_pw2potx0,tab_pw2potxx0)
            nexp0=(npw0+1)/2
            do i=1,ndim-1
               nexp0 = nexp0*npw0
            enddo

            allocate(wpwexp0(nexp0))
            call mk_kernel_Fourier_transform(ndim,ngs,deltas(istart),
     1          wdeltas(istart),npw0,ws0,ts0,nexp0,wpwexp0)
      
            lmptot0=nexp0*nd*2
            allocate(rmlexp0(lmptot0),stat=ier)
            call dmk_proxycharge2pw(ndim,nd,porder,
     1          proxycharge(1,1,ibox),npw0,tab_coefs2pw0,rmlexp0)
            call dmk_multiply_kernelFT(nd,nexp0,rmlexp0,wpwexp0)
            call dmk_pw2proxypgh(ndim,nd,porder,npw0,
     1          rmlexp0,tab_pw2pot0,tab_pw2potx0,tab_pw2potxx0,
     2          ifpgh,proxypotential(1,1,ibox),
     3          proxygradient(1,1,1,ibox),proxyhessian(1,1,1,ibox))
            deallocate(tab_coefs2pw0,wpwexp0,rmlexp0)
            deallocate(tab_pw2pot0,tab_pw2potx0,tab_pw2potxx0)
         endif
      enddo

      if(ifprint.eq.1) 
     1   call prinf("done with fat Gaussians on the root box*",i,0)      

c     form planewave expansions 
      do 1100 ilev =  0,nlevels
C$OMP PARALLEL DO DEFAULT (SHARED)
C$OMP$PRIVATE(ibox)
C$OMP$SCHEDULE(DYNAMIC)
        do ibox=itree(2*ilev+1),itree(2*ilev+2)
          if (ifpwexp(ibox).eq.1) then
c           form the pw expansion
            call dmk_proxycharge2pw(ndim,nd,porder,
     1          proxycharge(1,1,ibox),npw,tab_coefs2pw(1,1,ilev),
     2          rmlexp(iaddr(1,ibox)))
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
      tottimeinfo(4) = time2-time1








      
      if(ifprint.ge.1)
     1    call prinf('=== STEP 5 (mp to loc) ===*',i,0)
c      ... step 5, convert multipole expansions into local
c       expansions

      call cpu_time(time1)
C$    time1=omp_get_wtime()
      
      do 1300 ilev = 0,nlevels
         xmin = boxsize(ilev)
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
     1                rmlexp(iaddr(2,ibox)),wpwshift(1,ind,ilev))
               enddo
c     multiply the Fourier transform of the kernel
               call dmk_multiply_kernelFT(nd,nexp,
     1             rmlexp(iaddr(2,ibox)),wpwexp(1,ilev))
            endif
        enddo
C$OMP END PARALLEL DO        
 1300 continue
c      
      call cpu_time(time2)
C$    time2=omp_get_wtime()
      tottimeinfo(5) = time2-time1





      



      if(ifprint.ge.1)
     1    call prinf('=== STEP 6 (eval loc pwexps) ===*',i,0)

c     ... step 6, convert local plane wave expansions to proxy potential
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      do 1500 ilev = 0,nlevels
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,nchild,j,jbox,gbox,hbox)
C$OMP$SCHEDULE(DYNAMIC)
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
            nchild = itree(iptr(4)+ibox-1)
            if (ifpwexp(ibox).eq.1) then
               if (ifpgh.ge.2) then
                  gbox=ibox
               else
                  gbox=1
               endif
               if (ifpgh.ge.3) then
                  hbox=ibox
               else
                  hbox=1
               endif
               call dmk_pw2proxypgh(ndim,nd,porder,npw,
     1             rmlexp(iaddr(2,ibox)),tab_pw2pot(1,1,ilev),
     2             tab_pw2potx(1,1,ilev),tab_pw2potxx(1,1,ilev),
     3             ifpgh,proxypotential(1,1,ibox),
     4             proxygradient(1,1,1,gbox),proxyhessian(1,1,1,hbox))
               if (nchild.gt.0) then
                  do j=1,nchild
                     jbox = itree(iptr(5) + (ibox-1)*mc+j-1)
c                    translate tensor product polynomial from parent to child
                     call tens_prod_trans_add(ndim,nd,porder,
     1                   proxypotential(1,1,ibox),porder,
     2                   proxypotential(1,1,jbox),
     3                   p2ctransmat(1,1,1,j))
                     if (ifpgh.ge.2) then
                        call tens_prod_trans_add(ndim,ndg,porder,
     1                      proxygradient(1,1,1,ibox),porder,
     2                      proxygradient(1,1,1,jbox),
     3                      p2ctransmat(1,1,1,j))
                     endif
                     if (ifpgh.ge.3) then
                        call tens_prod_trans_add(ndim,ndh,porder,
     1                      proxyhessian(1,1,1,ibox),porder,
     2                      proxyhessian(1,1,1,jbox),
     3                      p2ctransmat(1,1,1,j))
                     endif
                        
                  enddo
               endif   
            endif
         enddo
C$OMP END PARALLEL DO        
 1500 continue

      call cpu_time(time2)
C$    time2 = omp_get_wtime()      
      tottimeinfo(6) = time2 - time1








      if(ifprint .ge. 1)
     1     call prinf('=== STEP 7 (direct interactions) =====*',i,0)
c
cc
      call cpu_time(time1)
C$    time1=omp_get_wtime()
      do ilev=0,nlevels
         iend=ndeltas-idelta(ilev)
         
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,nl1,jbox,jlev,bs,ixyz,jstart,id,gbox,hbox)
C$OMP$SCHEDULE(DYNAMIC)  
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
c           ibox is the target box            

            nl1 = nlist1(ibox)
            do j=1,nl1
cccc           jbox is the source box
               jbox = list1(j,ibox)
               jlev = itree(iptr(2)+jbox-1)
               bs = boxsize(jlev)
               call bdmk_find_loctab_ind(ndim,iperiod,
     1             centers(1,ibox),centers(1,jbox),bs,bs0,
     2             mrefinelev,ixyz)
               
               if (ifpwexp(jbox).eq.1 .and. jbox.eq.ibox) then
c     when ifpwexp(jbox)=1, self interaction at its own
c     level is taken care of by plane-wave expansion
                  jstart=ipwaddr(1,jlev+1)
               elseif (jlev.lt.ilev) then
c     when the source box is bigger than the target box, interaction
c     starts from the target box level
                  jstart=ipwaddr(1,ilev)
               else
                  jstart=ipwaddr(1,jlev)
               endif

c     the starting index jstart depends on the source box
c     the ending index iend always depends on the target box, since
c     all remaining Gaussians will be taken care of by the asymptotic
c     expansions
               do id=jstart,iend
                  if (ifpgh.ge.2) then
                     gbox=ibox
                  else
                     gbox=1
                  endif
                  if (ifpgh.ge.3) then
                     hbox=ibox
                  else
                     hbox=1
                  endif

                  call bdmk_tens_prod_to_pghloc(ndim,nd,norder,
     1                wdeltas(id),fvals(1,1,jbox),ifpgh,pot(1,1,ibox),
     2                grad(1,1,1,gbox),hess(1,1,1,hbox),
     3                nloctab,tab_loc(1,1,1,id,ilev),
     4                tabx_loc(1,1,1,id,ilev),tabxx_loc(1,1,1,id,ilev),
     5                ind_loc(1,1,1,id,ilev),ixyz)
               enddo
            enddo
         enddo
C$OMP END PARALLEL DO
      enddo
c
      call cpu_time(time2)
C$    time2=omp_get_wtime()  
      tottimeinfo(7) = time2-time1

      





      
      




      
      if(ifprint .ge. 1)
     $ call prinf('=== STEP 8 (direct asymptotic interaction) ===*',i,0)
      call cpu_time(time1)
C$    time1=omp_get_wtime()

      if (ifpgh.eq.1) then
c     evaluate potential
         call treedata_eval_pot_nd_asym_fast(ndim,nd,ndeltas,deltas,
     1       wdeltas,idelta,ipoly,nasym,nlevels,itree,iptr,boxsize,
     2       norder,fvals,flvals,fl2vals,pot)
      endif

      if (ifpgh.eq.2) then
c     evaluate potential and gradient
         call treedata_eval_pg_nd_asym_fast(ndim,nd,ndeltas,deltas,
     1       wdeltas,idelta,ipoly,nasym,nlevels,itree,iptr,boxsize,
     2       norder,fvals,flvals,fl2vals,gvals,glvals,pot,grad)
      endif
      
      if (ifpgh.eq.3) then
c     evaluate potential, gradient, and hessian
         call treedata_eval_pgh_nd_asym_fast(ndim,nd,ndeltas,deltas,
     1       wdeltas,idelta,ipoly,nasym,nlevels,itree,iptr,boxsize,
     2       norder,fvals,flvals,fl2vals,gvals,glvals,hvals,hlvals,
     3       pot,grad,hess)
      endif
      
      call cpu_time(time2)
C$    time2=omp_get_wtime()  
      tottimeinfo(8) = time2-time1




      

      
      
      if(ifprint .ge. 1)
     $    call prinf('=== STEP 9 (proxy potential -> pot) ===*',i,0)
c     evaluate the total contribution of the plane wave part via proxypotential
      call cpu_time(time1)
C$    time1=omp_get_wtime()
      
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,nchild)
C$OMP$SCHEDULE(DYNAMIC)  
      do ibox=1,nboxes
         nchild = itree(iptr(4)+ibox-1)
         if (nchild.eq.0) then
            call bdmk_proxypot2pot(ndim,nd,porder,
     1          proxypotential(1,1,ibox),
     2          norder,pot(1,1,ibox),potevalmat)
            if (ifpgh.ge.2) then
               call bdmk_proxypot2pot(ndim,ndg,porder,
     1             proxygradient(1,1,1,ibox),
     2             norder,grad(1,1,1,ibox),potevalmat)
            endif
            if (ifpgh.ge.3) then
               call bdmk_proxypot2pot(ndim,ndh,porder,
     1             proxyhessian(1,1,1,ibox),
     2             norder,hess(1,1,1,ibox),potevalmat)
            endif
         endif
      enddo
C$OMP END PARALLEL DO         

c     add the contribution from the constant term for the logarithmic kernel
      if (ndim .eq.2 .and. ikernel.le.1) then
         allocate(fint(nd))
         do ind=1,nd
            fint(ind)=0.0d0
         enddo
         
         do ilev=0,nlevels
            sc=boxsize(ilev)/2.0d0
            sc=sc**ndim
            do ibox=itree(2*ilev+1),itree(2*ilev+2)
               nchild = itree(iptr(4)+ibox-1)
               if (nchild.eq.0) then
                  do ind=1,nd
                     do i=1,npbox
                        fint(ind)=fint(ind)+sc*whtsnd(i)
     1                      *fvals(ind,i,ibox)
                     enddo
                  enddo
               endif
            enddo
         enddo

         call prin2('the integral of the rhs=*',fint,nd)


         if (ikernel.eq.1) then
            do ind=1,nd
               fint(ind)=fint(ind)*c0
            enddo
            do ibox=1,nboxes
               nchild = itree(iptr(4)+ibox-1)
               if (nchild.eq.0) then
                  do i=1,npbox
                     do ind=1,nd
                        pot(ind,i,ibox)=pot(ind,i,ibox)+fint(ind)
                     enddo
                  enddo
               endif
            enddo
         endif
      endif
      
      call cpu_time(time2)
C$    time2=omp_get_wtime()  
      tottimeinfo(9)=time2-time1








      
      
c     evaluate potential at extra targets
      if (ifpghtarg.ge.1) then
         call bdmk_potevaltarg(nd,ndim,ipoly,norder,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     2    pot,ntarg,targs,
     3    pote)
      endif

c     evaluate gradient at extra targets
      if (ifpghtarg.ge.2) then
         call bdmk_potevaltarg(ndg,ndim,ipoly,norder,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     2    grad,ntarg,targs,
     3    grade)
      endif

c     evaluate hessian at extra targets
      if (ifpghtarg.ge.3) then
         call bdmk_potevaltarg(ndh,ndim,ipoly,norder,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     2    hess,ntarg,targs,
     3    hesse)
      endif

      if (ifprint .ge. 1) then
cccc         call prinf('=== STEP 1 (precomputation) ===*',i,0)
cccc         call prinf('=== STEP 2 (local Taylor expansion) ===*',i,0)
cccc         call prinf('=== STEP 3 (proxy charge evaluation) ===*',i,0)
cccc         call prinf('=== STEP 4 (proxy charge -> plane wave) ===*',i,0)
cccc         call prinf('=== STEP 5 (plane wave mp to loc) ===*',i,0)
cccc         call prinf('=== STEP 6 (loc pw -> proxy potential) ===*',i,0)
cccc         call prinf('=== STEP 7 (direct table interaction) ===*',i,0)
cccc       call prinf('=== STEP 8 (direct asymptotic interaction) ===*',i,0)
cccc         call prinf('=== STEP 9 (proxy potential -> pot) ===*',i,0)
         call prin2('total timeinfo=*',tottimeinfo,9)
      endif
      
      return
      end
C
C
C
c  
      subroutine faddeevan(ndim,ntaylor,a,delta,fout)
c     returns fout = \int_0^a exp(-t^2/delta) t^ndt for n=1,3,5 in 2D,
c     or fout = \int_0^a exp(-t^2/delta) t^ndt for n=2,4,6 in 3D.
c     Both correspond to Taylor expansions of order 0, 2, 4.
      implicit real *8 (a-h,o-z)
      real *8 fout(0:ntaylor)

      if (ndim.eq.2) then
         call faddeevan_2d(ntaylor,a,delta,fout)
      elseif (ndim.eq.3) then
         call faddeevan_3d(ntaylor,a,delta,fout)
      endif
      
      return
      end subroutine
C
C
c  
      subroutine faddeevan_2d(ntaylor,a,delta,fout)
c     returns fout = \int_0^a exp(-t^2/delta) t^ndt for n=1,3,5,
c     which corresponds to Taylor expansion of order 0, 2, 4 in 2D.
      implicit real *8 (a-h,o-z)
      real *8 fac(0:100),d(0:20)
      real *8 c(0:10),fout(0:ntaylor)
      data pi/3.1415926535 8979323846 2643383279 5028841971 693993751d0/

      fac(0)=1.0d0
      do i=1,20
         fac(i)=fac(i-1)*i
      enddo
      
      x=a/sqrt(delta)
      x2=x*x

      c(0)=delta
      do i=1,ntaylor
         c(i)=c(i-1)*delta
      enddo
      
      if (x.lt.0.8d0) then
c      if (x.lt.0.0d0) then
         d(0)=x**2
         do i=1,20
            d(i)=d(i-1)*x2
         enddo

         do k=0,ntaylor
            fout(k)=0
            sign=1
            do i=0,20
               fout(k)=fout(k)+sign*d(i)*x2**k
     1             /fac(i)/(2*i+2+2*k)
               sign=-sign
            enddo
            fout(k)=fout(k)*c(k)
         enddo
      else
         expx2=exp(-x2)
         x4=x2*x2
         x6=x4*x2
         
         fout(0)=c(0)*(1.0d0-expx2)/2         
         fout(1)=c(1)*(1-expx2*(x2+1))/2
         if (ntaylor .ge. 2 ) then
            fout(2)=c(2)*(1-0.5d0*expx2*(x4+2*x2+2))
         endif
         if (ntaylor .ge. 3) then
            fout(3)=c(3)*(3-expx2*
     1          (6+6*x2+3*x4+x6)/2)
         endif
      endif
      
      return
      end
c
c
c
c
c
      subroutine faddeevan_3d(ntaylor,a,delta,fout)
c     returns fout = \int_0^a exp(-t^2/delta) t^ndt for n=2,4,6,
c     which corresponds to Taylor expansion of order 0, 2, 4 in 3D.
      implicit real *8 (a-h,o-z)
      real *8 fac(0:100),d(0:20)
      real *8 c(0:10),fout(0:ntaylor)
      data pi/3.1415926535 8979323846 2643383279 5028841971 693993751d0/

      fac(0)=1.0d0
      do i=1,20
         fac(i)=fac(i-1)*i
      enddo
      
      x=a/sqrt(delta)
      x2=x*x

      c(0)=delta**1.5d0
      do i=1,ntaylor
         c(i)=c(i-1)*delta
      enddo
      
      if (x.lt.1.0d0) then
         d(0)=x**3
         do i=1,20
            d(i)=d(i-1)*x2
         enddo

         do k=0,ntaylor
            fout(k)=0
            sign=1
            do i=0,20
               fout(k)=fout(k)+sign*d(i)*x2**k
     1             /fac(i)/(2*i+3+2*k)
               sign=-sign
            enddo
            fout(k)=fout(k)*c(k)
         enddo
      else
         sqpi=sqrt(pi)
         erfx=erf(x)
         expx2=exp(-x2)
         x4=x2*x2
         x6=x4*x2
         
         fout(0)=c(0)*(erfx*sqpi/2-x*expx2)/2         
         fout(1)=c(1)*(3*sqpi*erfx/8.0d0
     1       -x*expx2*(x2/2+0.75d0))
         if (ntaylor .ge. 2 ) then
            fout(2)=c(2)*(15.0d0*sqpi*erfx/16
     1          -x*expx2*(15.0d0/8+5*x2/4+x4/2))
         endif
         if (ntaylor .ge. 3) then
            fout(3)=c(3)*(105.0d0*sqpi*erfx/32
     1          -x*expx2*(105.0d0/16+35*x2/8+7*x4/4+x6/2))
         endif
      endif
      
      return
      end
c
c
c
c
c
      subroutine find_npwlevel(eps,nlevels,boxsize,delta,npwlevel)
      implicit real *8 (a-h,o-z)
      real *8 boxsize(0:nlevels)
      real *8, allocatable :: boxsize0(:)

      nlevstart=-100
c     cutoff length      
      dcutoff = sqrt(delta*log(1.0d0/eps))

      allocate(boxsize0(nlevstart:nlevels))
      
      do ilev=0,nlevels
         boxsize0(ilev)=boxsize(ilev)
      enddo
      
      do ilev=-1,nlevstart,-1
         boxsize0(ilev)=boxsize0(ilev+1)*2
      enddo

cccc      call prin2(' dcutoff=*',dcutoff,1)
cccc      call prin2(' boxsize0=*',boxsize0(-10),nlevels+11)
      
c     find the cutoff level
      npwlevel = nlevels+1
      do i=nlevels,nlevstart,-1
         if (boxsize0(i).ge. dcutoff) then
            npwlevel=i
            exit
         endif
      enddo
c
      if (boxsize(nlevels).gt.dcutoff) npwlevel=nlevels
      if (boxsize0(nlevstart) .lt. dcutoff) then
         print *, 'warning: npwlevel<-100 no implemented!'
         npwlevel=nlevstart
         pause
      endif

      return
      end
c      
c
c
c
      subroutine bdmk_potevaltarg(nd,ndim,ipoly,norder,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     2    pot,ntarg,targs,
     3    pottarg)
c     
c
c     This code computes the volume potential on arbitrary targets given
c     the potential on a tensor product grid of each leaf node in an adaptive tree.
c
c     input
c     nd - integer
c          number of right hand sides
c     ndim - integer
c           dimension of the underlying space
c     ipoly - integer
c            0: Legendre polynomials
c            1: Chebyshev polynomials
c     norder - integer
c           order of expansions for input function value array
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
c     pot - double precision (nd,npbox,nboxes)
c            volume potential on the tree structure (note that 
c           the potential is non-zero only in the leaf boxes of the new tree
c     ntarg - number of targets
c     targs - double precision (ndim,ntarg)
c            coordinates of target points
c
c     output:
c     pottarg - double precision (nd,ntarg)
c            volume potential at targets
c
      implicit none
      integer nd,ndim,ipoly
      integer nboxes,nlevels,ntarg
      integer iptr(8),ltree
      integer itree(ltree),norder,npbox
      real *8 targs(ndim,ntarg)

      real *8 pot(nd,norder**ndim,nboxes)

      real *8 pottarg(nd,ntarg)

      real *8 centers(ndim,nboxes)
      real *8 boxsize(0:nlevels)

c     local variables
      integer norder2,i
      real *8 umat(norder,norder)
      real *8 vmat(norder,norder)
      real *8 vpmat(norder,norder)
      real *8 vppmat(norder,norder)
      real *8 umat_nd(norder,norder,ndim)
      
      real *8, allocatable :: coefsp(:,:,:)

      
      call ortho_eval_tables(ipoly,norder,umat,vmat,vpmat,vppmat)
      norder2=norder*norder
      do i=1,ndim
         call dcopy_f77(norder2,umat,1,umat_nd(1,1,i),1)
      enddo
      
      allocate(coefsp(nd,norder**ndim,nboxes))
      call treedata_trans_nd(ndim,nd,
     1    nlevels,itree,iptr,boxsize,
     2    norder,pot,coefsp,umat_nd)
      call treedata_evalt_nd(ndim,nd,ipoly,norder,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,
     2    coefsp,ntarg,targs,pottarg)

      return
      end
c
c
c
c
c
c------------------------------------------------------------------    
      subroutine bdmk_mpalloc(nd,dim,npw,nlevels,laddr,
     1    ifpwexp,iaddr,lmptot)
c     This subroutine determines the size of the array
c     to be allocated for multipole/local expansions
c
c     Input arguments
c     nd          in: integer
c                 number of outgoing/incoming expansions
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
      integer nlevels,npw,nd
      integer laddr(2,0:nlevels),ifpwexp(*)
      integer *8 iaddr(2,*)
      integer *8 lmptot,istart,nn,nn1,nn2,itmp,itmp2
      integer ibox,i,istarts,iends,npts
c
      nn = npw**(dim-1)*((npw+1)/2)
c     the factor 2 is the (complex *16)/(real *8) ratio
      nn = nn*2*nd

c     assign memory pointers
      istart = 1
      itmp=0
      do i = 0,nlevels
         do ibox = laddr(1,i),laddr(2,i)
c          Allocate memory for the multipole PW expansions
           if (ifpwexp(ibox).eq.1) then
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
c          Allocate memory for the local PW expansions
           if (ifpwexp(ibox).eq.1) then
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

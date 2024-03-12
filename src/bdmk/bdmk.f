c     Genuine O(N) algorithm
c     Last modified by Shidong Jiang on 02/05/2024
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
c            0: the Yukawa kernel; 1: the power kernel; 2: the log kernel in 2D. 
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
      integer nd,ndim
      integer ikernel,nboxes,nlevels,ntarg,ifpgh,ifpghtarg
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
      real *8 timeinfo(20,-10:20)


      real *8 umat(norder,norder)
      real *8 vmat(norder,norder)
      real *8 vpmat(norder,norder)
      real *8 vppmat(norder,norder)
      real *8 umat_nd(norder,norder,ndim)

      real *8, allocatable :: fcoefs(:,:,:)
      real *8, allocatable :: flcoefs(:,:,:)
      real *8, allocatable :: coefsp(:,:,:)
cccc      real *8 flvals(nd,npbox,nboxes)
cccc      real *8 fl2vals(nd,npbox,nboxes)
      real *8, allocatable :: flvals(:,:,:)
      real *8, allocatable :: fl2vals(:,:,:)
cccc      real *8, allocatable :: fl3vals(:,:,:)
      real *8, allocatable :: potgs(:,:,:)

      real *8 ws(200),deltas(200),pval(200),xs(200),whts(200),whtsp(200)
      integer npwlevels(200),ipwaddr(2,-10:nlevels+1),ipw(-10:nlevels+1)
      integer porder,ncbox,mc,ipoly0
      
      real *8, allocatable :: proxycharge(:,:,:)
      real *8, allocatable :: proxypotential(:,:,:)
      real *8, allocatable :: den2pcmat(:,:),potevalmat(:,:)
      integer, allocatable :: isgn(:,:)
      real *8, allocatable :: umatp(:,:),umatp_nd(:,:,:)
      real *8, allocatable :: vmatp(:,:),vtmp(:,:),vtmp2(:,:)
      real *8, allocatable :: p2ctransmat(:,:,:,:)
      real *8, allocatable :: c2ptransmat(:,:,:,:)
      real *8, allocatable :: whtsnd(:),fint(:)

      integer ilev,ng,i,istart,ntaylor,norder2,j,ibox,ind,n,m
      integer k,porder2,iend,ngs,npwlevel,jbox,itype,ipoly,nchild
      integer ifprint,ier,ifexpon
      real *8 fv(0:10),c(0:10)
      real *8 dlogr0,r2,r4,r6,dk1,dk0
      real *8 pi,r0,de,br,br2,br3,br4,br5,br6,br7,br8,br9
      real *8 b2,b4,b6,ebr,delta
      real *8 sc,dd,vinttrue,beta2,vintcomp,derr,dwnorm
      real *8 c0,eulergamma
      real *8 time1,time2

      complex *16 zk,ima,ztmp,zh0,zh1
      data ima/(0.0d0,1.0d0)/
      data pi/3.1415926535 8979323846 2643383279 5028841971 693993751d0/
      data eulergamma/0.5772156649015328606065120900824024310421593d0/
      
      ifprint=1
      
c     r0 is the cutoff length, i.e., for r>r0, 1/r is well apprxoimated by
c     sum of Gaussians. Thus, one only needs to compute the correction for
c     r\in [0,r0]
      call get_sognodes(ndim,ikernel,eps,nlevels,norder,beta,
     1    r0,ng,ws,deltas)
c     for the log kernel, there is a nonzero constant mode
      if (ndim.eq.2 .and. ikernel.eq.1) then
         c0=0
         do i=1,ng
            c0=c0-ws(i)*exp(-1.0d0/deltas(i))
         enddo
         call prin2('c0=*',c0,1)
      endif
      
      call prin2('r0=*',r0,1)
      call prin2('deltas=*',deltas,ng)
      call prin2('ws=*',ws,ng)

      do ilev=-10,nlevels+1
         ipw(ilev)=0
      enddo
      
      do i=1,ng
         call find_npwlevel(eps,nlevels,boxsize,deltas(i),npwlevels(i))
         ipw(npwlevels(i)) = ipw(npwlevels(i))+1
      enddo

      istart=1
      ipwaddr(1,-10)=1
      ipwaddr(2,-10)=0

      do ilev=-9,nlevels+1
         istart=ipwaddr(2,ilev-1)+1
         ipwaddr(1,ilev)=istart
         ipwaddr(2,ilev)=istart+ipw(ilev)-1
      enddo
      call prinf('npwlevels=*',npwlevels,ng)
      call prinf('ipw=*',ipw,nlevels+12)
      call prinf('ipwaddr=*',ipwaddr,2*(nlevels+12))



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
         
         b2=beta*beta
         b4=b2*b2
         b6=b2*b4


         c(0)=(1-br*dk1)/b2

         if (br.le.1.0d-3) then
            c(1)=-(eulergamma+log(br/2)-0.25d0)*br4/4/b4
         else
            c(1)=(4-2*br2*dk0-br*(br2+4)*dk1)/b4            
         endif
c         c(2)=120/b6-ebr*(br5+5*br4+20*br3+60*br2+120*br+120)/b6
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
         c(0) = r2*(dlogr0-0.5d0)/2-c0*r2/2
         c(1) = r4*(dlogr0-0.25d0)/4-c0*r4/4
         c(2) = r6*(dlogr0-1.0d0/6)/6-c0*r6/6
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
      do i=1,ng
         delta=deltas(i)
         call faddeevan(ndim,ntaylor,r0,delta,fv)
         
         do k=0,ntaylor
            c(k)=c(k)-ws(i)*fv(k)
         enddo
      enddo

c     finally, multiplied them by proper constants
c     these constants depends only on the dimension of the underlying space
      if (ndim.eq.2) then
c        2 pi is the perimeter of the unit circle
         c(0)=c(0)*2*pi
c        2=2!, and 1/2 comes from the fact, say, x^2 has 1/2 contribution of r^2
         c(1)=c(1)*2*pi/2/2
c        24=4!, and 3/8 comes from the fact, say, x^4 has 3/8 contribution of r^4
         c(2)=c(2)*2*pi/24*3/8
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

      call prin2('c=*',c,ntaylor+1)
      
      call ortho_eval_tables(ipoly,norder,umat,vmat,vpmat,vppmat)
      norder2=norder*norder
      do i=1,ndim
         call dcopy_f77(norder2,umat,1,umat_nd(1,1,i),1)
      enddo

      do i=1,20
         tottimeinfo(i)=0
      enddo
      
      call cpu_time(time1)
C$    time1=omp_get_wtime()
      
      allocate(fcoefs(nd,npbox,nboxes))
      allocate(flcoefs(nd,npbox,nboxes))
      allocate(flvals(nd,npbox,nboxes))
      allocate(fl2vals(nd,npbox,nboxes))
c      allocate(fl3vals(nd,npbox,nboxes))

      do ilev = 0,nlevels
        sc=2.0d0/boxsize(ilev)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,j,ind,nchild)
C$OMP$SCHEDULE(DYNAMIC)  
        do ibox = itree(2*ilev+1),itree(2*ilev+2)
          nchild = itree(iptr(4) + ibox-1)
          if(nchild.eq.0) then
cccc             if (1.eq.1) then
c     compute the Laplacian of f
             call tens_prod_trans_vec(ndim,nd,norder,
     1           fvals(1,1,ibox),norder,fcoefs(1,1,ibox),umat_nd)
ccc             call prin2('fcoefs=*',fcoefs(1,1,ibox),npbox)
ccc             pause
             
             call ortho_eval_laplacian_nd(ndim,nd,norder,
     1           fcoefs(1,1,ibox),sc,flvals(1,1,ibox),vmat,vppmat)
c            compute the BiLaplacian of f
             call tens_prod_trans_vec(ndim,nd,norder,
     1           flvals(1,1,ibox),norder,flcoefs(1,1,ibox),umat_nd)
             call ortho_eval_laplacian_nd(ndim,nd,norder,
     1           flcoefs(1,1,ibox),sc,fl2vals(1,1,ibox),vmat,vppmat)
c     compute the TriLaplacian of f
c     Note: this is unreliable since the condition number of spectral differentiation
c     is (norder^2)^6=norder^12.
c             call ortho_trans_nd(ndim,nd,norder,
c     1           fl2vals(1,1,ibox),fcoefs,umat_nd)
c             call ortho_eval_laplacian_nd(ndim,nd,norder,fcoefs,
c     1           sc,fl3vals(1,1,ibox),vmat,vppmat)
cccc             endif
             do j=1,npbox
             do ind=1,nd
                pot(ind,j,ibox)=c(0)*fvals(ind,j,ibox)
     1              +c(1)*flvals(ind,j,ibox)+c(2)*fl2vals(ind,j,ibox)
cccc  2                 +c(3)*fl3vals(ind,j,ibox)
             enddo
             enddo
          endif
        enddo
C$OMP END PARALLEL DO         
      enddo

      call cpu_time(time2)
C$    time2=omp_get_wtime()  
      tottimeinfo(1)=time2-time1

c     compute proxy charges for all boxes
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
cccc      porder=norder
      
      ncbox=porder**ndim
      allocate(proxycharge(ncbox,nd,nboxes),stat=ier)
      if(ier.ne.0) then
         print *, "Cannot allocate workspace for proxy charges"
         print *, "length=", ncbox*nd*nboxes
         ier = 4
         return
      endif
      
      allocate(proxypotential(ncbox,nd,nboxes),stat=ier)
      if(ier.ne.0) then
         print *, "Cannot allocate workspace for proxy potential"
         print *, "length=", ncbox*nd*nboxes
         ier = 4
         return
      endif

      do ibox=1,nboxes
         do ind=1,nd
            do j=1,ncbox
               proxycharge(j,ind,ibox)=0
            enddo
         enddo
      enddo

      do ibox=1,nboxes
         do ind=1,nd
            do j=1,ncbox
               proxypotential(j,ind,ibox)=0
            enddo
         enddo
      enddo
      
      mc=2**ndim
      allocate(isgn(ndim,mc))
      call get_child_box_sign(ndim,isgn)
      
      allocate(p2ctransmat(porder,porder,ndim,mc))
      allocate(c2ptransmat(porder,porder,ndim,mc))
c     always use Legendre polynomials for proxy charges since we need to do integration
      ipoly0=0
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
      if (ndim.eq.2 .and. ikernel.eq.1) then
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

c
c     upward pass for calculating proxy charges
c
      call cpu_time(time1)
C$    time1=omp_get_wtime()
      do ilev=nlevels,0,-1
         sc=boxsize(ilev)/2.0d0
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
      enddo
      call cpu_time(time2)
C$    time2=omp_get_wtime()  
      tottimeinfo(2)=time2-time1

c     downward pass
c     now call box FGT with many deltas
      
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ilev,istart,iend,ngs,npwlevel,i)
C$OMP$SCHEDULE(DYNAMIC)  
      do ilev=-10,nlevels+1
         istart=ipwaddr(1,ilev)
         iend=ipwaddr(2,ilev)
         ngs=iend-istart+1

         if (ngs .gt. 0) then
            npwlevel=ilev
            call boxfgt_md(nd,ndim,ngs,deltas(istart),ws(istart),
     1          npwlevel,eps,ipoly,norder,npbox,nboxes,nlevels,ltree,
     2          itree,iptr,centers,boxsize,
     3          fvals,flvals,fl2vals,porder,ncbox,
     4          proxycharge,proxypotential,p2ctransmat,
     5          ifpgh,pot,grad,hess,timeinfo(1,ilev))
            do i=1,10
               tottimeinfo(i+2)=tottimeinfo(i+2)+timeinfo(i,ilev)
            enddo
         endif
      enddo
C$OMP END PARALLEL DO         

c     evaluate the total contribution of the plane wave part via proxypotential
      call cpu_time(time1)
C$    time1=omp_get_wtime()
      do ibox=1,nboxes
         nchild = itree(iptr(4)+ibox-1)
         if (nchild.eq.0) then
            call bdmk_proxypot2pot(ndim,nd,porder,
     1          proxypotential(1,1,ibox),
     2          norder,pot(1,1,ibox),potevalmat)            
         endif
      enddo

c     add the contribution from the constant term for the logarithmic kernel
      if (ndim .eq.2 .and. ikernel.eq.1) then
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

         do ind=1,nd
            fint(ind)=fint(ind)*c0
         enddo

         call prin2('fint=*',fint,nd)
         
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

cccc  call prin2('pote=*',pote,240)
      if (ifprint .ge. 1) then
         call prinf('=== STEP 1 (local Taylor expansion) ===*',i,0)
         call prinf('=== STEP 2 (proxy charge evaluation) ===*',i,0)
         call prinf('=== STEP 3 (precomputation) ===*',i,0)
         call prinf('=== STEP 4 (proxy charge -> plane wave) ===*',i,0)
         call prinf('=== STEP 5 (plane wave mp to loc) ===*',i,0)
         call prinf('=== STEP 6 (loc pw -> proxy potential) ===*',i,0)
         call prinf('=== STEP 7 (direct table interaction) ===*',i,0)
       call prinf('=== STEP 8 (direct asymptotic interaction) ===*',i,0)
         call prinf('=== STEP 9 (proxy potential -> pot) ===*',i,0)
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
      real *8 boxsize0(-10:nlevels)

c     cutoff length      
      dcutoff = sqrt(delta*log(1.0d0/eps))
cccc      call prin2(' dcutoff=*',dcutoff,1)

      do ilev=0,nlevels
         boxsize0(ilev)=boxsize(ilev)
      enddo
      
      do ilev=-1,-10,-1
         boxsize0(ilev)=boxsize0(ilev+1)*2
      enddo
      
c     find the cutoff level
      npwlevel = nlevels+1
      do i=nlevels,-10,-1
         if (boxsize0(i).ge. dcutoff) then
            npwlevel=i
            exit
         endif
      enddo
c
      if (boxsize(nlevels).gt.dcutoff) npwlevel=nlevels+1
c      if (boxsize(0) .le. dcutoff) npwlevel=0

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

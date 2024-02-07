c
c     pts_tree in ndim=1,2,3 dimensions
c     modifed from pts_tree3d.f by Shidong Jiang 04/01/2022
c     last modified by Shidong Jiang on 02/07/2024
c
c   generate level restricted tree based on resolving points
c    (Currently only supports, sorting on sources, sorting on
c      targets, or sorting on sources and targets)
c
c   There is additional functionality to further subdivide
c    an existing tree based on resolving a function, 
c      this functionality tree will be added in later
c
c   This code has the following user callable routines
c
c      pts_tree_mem -> returns memory requirements for creating
c         a tree based on max number of sources/targets
c         in a box (tree length
c         number of boxes, number of levels)
c      pts_tree_build -> Make the actual tree, returns centers of boxes,
c        colleague info, pts sorted on leaf boxes
c
c      iptr(1) - laddr
c      iptr(2) - ilevel
c      iptr(3) - iparent
c      iptr(4) - nchild
c      iptr(5) - ichild
c      iptr(6) - ncoll
c      iptr(7) - coll
c      iptr(8) - ltree
c   
c 


      subroutine pts_tree_mem(ndim,src,ns,targ,nt,idivflag,
     1    ndiv,nlmin,nlmax,ifunif,iper,
     2    nlevels,nboxes,ltree)
c
c
c
c----------------------------------------
c  get memory requirements for the tree
c
c
c  input parameters:
c    - ndim: dimension of the underlying space
c    - src: real *8 (ndim,ns)
c        source locations
c    - targ: real *8 (ndim,nt) 
c        target locations
c    - idivflag: integer
c        subdivision criterion
c          * divflag = 0 -> subdivide on sources only
c          * idivflag = 1 -> subdivide on targets only
c          * idivflag = 2 -> subdivide on max(sources+targets)
c    - ndiv: integer
c        subdivide if relevant number of particles
c        per box is greater than ndiv
c    - nlmin: integer
c        minimum number of levels of uniform refinement.
c        Note that empty boxes are not pruned along the way
c    - nlmax: integer
c        max number of levels
c    - ifunif: integer
c        flag for creating uniform pruned tree
c        Tree is uniform if ifunif=1 (Currently pruned part
c        under construction)
c    - iper: integer
c        flag for periodic implementations. 
c    - bs0 : real *8
c        side length of the bounding box
c    - cen0(ndim) : center of the bounding box
c        
c  output parameters
c    - nlevels: integer
c        number of levels
c    - nboxes: integer
c        number of boxes
c    - ltree: integer
c        length of tree
c----------------------------------
c
      implicit none
      integer ndim,nlevels,nboxes,idivflag
      integer ltree
      integer nbmax,nbtot
      integer ns,nt,ndiv
      integer nlmin,iper,ifunif
      double precision src(ndim,ns),targ(ndim,nt),cen0(ndim),bs0

      integer, allocatable :: laddr(:,:),ilevel(:),iparent(:),nchild(:)
      integer, allocatable :: ichild(:,:)
      double precision, allocatable :: centers(:,:)
      integer, allocatable :: nbors(:,:),nnbors(:)

      integer, allocatable :: isrc(:),itarg(:),isrcse(:,:),itargse(:,:)

      integer, allocatable :: ilevel2(:),iparent2(:),nchild2(:),
     1    ichild2(:,:),isrcse2(:,:),itargse2(:,:)
      double precision, allocatable :: centers2(:,:)

      integer nlmax
      integer i,j,mc,mnbors

      double precision, allocatable :: boxsize(:)
      integer, allocatable :: irefinebox(:)

      integer nbloc,nbctr,nbadd,irefine,ilev,ifirstbox,ilastbox
      integer ibox,nn,nss,ntt

      nbmax = 100 000
      mc = 2**ndim
      mnbors = 3**ndim
      
      allocate(boxsize(0:nlmax))

      
      allocate(laddr(2,0:nlmax),ilevel(nbmax),iparent(nbmax))
      allocate(nchild(nbmax),ichild(mc,nbmax))

      allocate(centers(ndim,nbmax),isrcse(2,nbmax),itargse(2,nbmax))
      allocate(isrc(ns),itarg(nt))

c
c     step 1: find enclosing box
c
      call pts_tree_boxsize0(ndim,iper,src,ns,targ,nt,
     1    bs0,cen0)
      
      boxsize(0) = bs0

      do i=1,ndim
         centers(i,1) = cen0(i)
      enddo
      
c
c      set tree info for level 0
c
      laddr(1,0) = 1
      laddr(2,0) = 1
      ilevel(1) = 0
      iparent(1) = -1
      nchild(1) = 0
      do i=1,mc
        ichild(i,1) = -1
      enddo

      isrcse(1,1) = 1
      isrcse(2,1) = ns
      
      itargse(1,1) = 1
      itargse(2,1) = nt

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,ns
        isrc(i) = i
      enddo
C$OMP END PARALLEL DO      

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,nt
        itarg(i) = i
      enddo
C$OMP END PARALLEL DO      


      nbctr = 1

      do ilev=0,nlmax-1
        irefine = 0

        ifirstbox = laddr(1,ilev) 
        ilastbox = laddr(2,ilev)

        nbloc = ilastbox-ifirstbox+1
        allocate(irefinebox(nbloc))
c
c          determine which boxes need to be refined
c


        if(ilev.ge.nlmin) then
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,ibox,nss,ntt,nn)
          do i=1,nbloc
            irefinebox(i) = 0
            ibox = ifirstbox + i-1
            nss = isrcse(2,ibox)-isrcse(1,ibox)+1
            ntt = itargse(2,ibox)-itargse(1,ibox)+1
            if(idivflag.eq.0) nn = nss
            if(idivflag.eq.1) nn = ntt
            if(idivflag.eq.2) nn = max(ntt,nss)

            if(nn.gt.ndiv) irefinebox(i) = 1
          enddo
C$OMP END PARALLEL DO        

          irefine = maxval(irefinebox(1:nbloc))
          if(ifunif.eq.1) then
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)         
            do i=1,nbloc
              irefinebox(i) = irefine
            enddo
C$OMP END PARALLEL DO            
          endif
        else
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
          do i=1,nbloc
            irefinebox(i) = 1
          enddo
C$OMP END PARALLEL DO         
          irefine = 1
        endif

c
c
c          figure out if current set of boxes is sufficient
c

        nbadd = 0 
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:nbadd)        
        do i=1,nbloc
          if(irefinebox(i).eq.1) nbadd = nbadd+mc
        enddo
C$OMP END PARALLEL DO        

        nbtot = nbctr+nbadd

c
c         if current memory is not sufficient reallocate
c
        if(nbtot.gt.nbmax) then
          print *, "Reallocating"
          allocate(centers2(ndim,nbmax),ilevel2(nbmax),iparent2(nbmax))
          allocate(nchild2(nbmax),ichild2(mc,nbmax),isrcse2(2,nbmax))
          allocate(itargse2(2,nbmax))

          call tree_copy(ndim,nbctr,centers,ilevel,iparent,nchild,
     1            ichild,centers2,ilevel2,iparent2,
     2            nchild2,ichild2)

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)     
          do i=1,nbctr
            isrcse2(1,i) = isrcse(1,i)
            isrcse2(2,i) = isrcse(2,i)
            itargse2(1,i) = itargse(1,i)
            itargse2(2,i) = itargse(2,i)
          enddo
C$OMP END PARALLEL DO          


          deallocate(centers,ilevel,iparent,nchild,ichild,
     1        isrcse,itargse)

          nbmax = nbtot
          allocate(centers(ndim,nbmax),ilevel(nbmax),iparent(nbmax))
          allocate(nchild(nbmax),ichild(mc,nbmax),isrcse(2,nbmax))
          allocate(itargse(2,nbmax))


          call tree_copy(ndim,nbctr,centers2,ilevel2,iparent2,
     1            nchild2,ichild2,centers,ilevel,iparent,nchild,ichild)

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
          do i=1,nbctr
            isrcse(1,i) = isrcse2(1,i)
            isrcse(2,i) = isrcse2(2,i)
            itargse(1,i) = itargse2(1,i)
            itargse(2,i) = itargse2(2,i)
          enddo
C$OMP END PARALLEL DO          

          deallocate(centers2,ilevel2,iparent2,nchild2,ichild2,
     1       isrcse2,itargse2)
        endif


        if(irefine.eq.1) then
          boxsize(ilev+1) = boxsize(ilev)/2
          laddr(1,ilev+1) = nbctr+1
          call tree_refine_boxes(ndim,irefinebox,nbmax,
     1       ifirstbox,nbloc,centers,boxsize(ilev+1),nbctr,ilev+1,
     2       ilevel,iparent,nchild,ichild)

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,ibox)
C$OMP$SCHEDULE(DYNAMIC)  
          do i=1,nbloc
            ibox = ifirstbox+i-1
            if(irefinebox(i).eq.1) then
              call sort_pts_to_children(ndim,ibox,nbmax,centers,ichild,
     1            src,ns,isrc,isrcse)
              call sort_pts_to_children(ndim,ibox,nbmax,centers,ichild,
     1            targ,nt,itarg,itargse)
            endif
          enddo
C$OMP END PARALLEL DO          

          laddr(2,ilev+1) = nbctr
        else
          exit
        endif

        deallocate(irefinebox)
      enddo

      nboxes = nbctr
      nlevels = ilev

      if(nlevels.ge.2.and.ifunif.ne.1) then

        nbtot = 2*mc*nboxes
        if(nbtot.gt.nbmax) then
          allocate(centers2(ndim,nbmax),ilevel2(nbmax),iparent2(nbmax))
          allocate(nchild2(nbmax),ichild2(mc,nbmax),isrcse2(2,nbmax))
          allocate(itargse2(2,nbmax))
          call tree_copy(ndim,nbctr,centers,ilevel,iparent,nchild,
     1            ichild,centers2,ilevel2,iparent2,
     2            nchild2,ichild2)

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)     
          do i=1,nbctr
            isrcse2(1,i) = isrcse(1,i)
            isrcse2(2,i) = isrcse(2,i)
            itargse2(1,i) = itargse(1,i)
            itargse2(2,i) = itargse(2,i)
          enddo
C$OMP END PARALLEL DO          

          deallocate(centers,ilevel,iparent,nchild,ichild,
     1        isrcse,itargse)

          nbmax = nbtot
          allocate(centers(ndim,nbmax),ilevel(nbmax),iparent(nbmax))
          allocate(nchild(nbmax),ichild(mc,nbmax),isrcse(2,nbmax))
          allocate(itargse(2,nbmax))


          call tree_copy(ndim,nbctr,centers2,ilevel2,iparent2,
     1            nchild2,ichild2,centers,ilevel,iparent,nchild,ichild)

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
          do i=1,nbctr
            isrcse(1,i) = isrcse2(1,i)
            isrcse(2,i) = isrcse2(2,i)
            itargse(1,i) = itargse2(1,i)
            itargse(2,i) = itargse2(2,i)
          enddo
C$OMP END PARALLEL DO          

          deallocate(centers2,ilevel2,iparent2,nchild2,ichild2,
     1       isrcse2,itargse2)

        endif

        allocate(nnbors(nbmax))
        allocate(nbors(mnbors,nbmax))
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
        do i=1,nboxes
          nnbors(i) = 0
          do j=1,mnbors
            nbors(j,i) = -1
          enddo
        enddo
C$OMP END PARALLEL DO        

        call computecoll(ndim,nlevels,nboxes,laddr,boxsize,centers,
     1        iparent,nchild,ichild,iper,nnbors,nbors)

        if(nlevels.ge.2.and.ifunif.ne.1) then
          call pts_tree_fix_lr(ndim,centers,nlevels,nboxes,boxsize,
     1         nbmax,nlmax,iper,laddr,ilevel,iparent,nchild,ichild,
     2         nnbors,nbors)
        endif

      endif

      ltree = (4+mc+mnbors)*nboxes + 2*(nlevels+1) 

      return
      end
c
c
c
c
c

      subroutine pts_tree_build(ndim,src,ns,targ,nt,idivflag,ndiv,
     1    nlmin,nlmax,ifunif,iper,nlevels,nboxes,
     2    ltree,itree,iptr,centers,
     3    boxsize)
c
c
c
c----------------------------------------
c  build tree
c
c
c input parameters:
c    - ndim: dimension of the underlying space
c    - src: real *8 (ndim,ns)
c        source locations
c    - targ: real *8 (ndim,nt) 
c        target locations
c    - idivflag: integer
c        subdivision criterion
c          * divflag = 0 -> subdivide on sources only
c          * idivflag = 1 -> subdivide on targets only
c          * idivflag = 2 -> subdivide on max(sources+targets)
c    - ndiv: integer
c        subdivide if relevant number of particles
c        per box is greater than ndiv
c    - nlmin: integer
c        minimum number of levels of uniform refinement.
c        Note that empty boxes are not pruned along the way
c    - nlmax: integer
c        max number of levels
c    - ifunif: integer
c        flag for creating uniform pruned tree
c        Tree is uniform if ifunif=1 (Currently pruned part
c        under construction)
c    - iper: integer
c        flag for periodic implementations. Currently unused.
c        Feature under construction
c    - nlevels: integer
c        number of levels
c    - nboxes: integer
c        number of boxes
c    - ltree: integer
c     - bs0 : real
c        side length of the bounding box
c     - cen0(3) : center of the bounding box
c
c  output:
c    - itree: integer(ltree)
c        tree info
c    - iptr: integer(8)
c        * iptr(1) - laddr
c        * iptr(2) - ilevel
c        * iptr(3) - iparent
c        * iptr(4) - nchild
c        * iptr(5) - ichild
c        * iptr(6) - ncoll
c        * iptr(7) - coll
c        * iptr(8) - ltree
c    - centers: double precision (dim,nboxes)
c        coordinates of box centers in the oct tree
c    - boxsize: double precision (0:nlevels)
c        size of box at each of the levels
c

      implicit none
      integer ndim,nlevels,nboxes,ns,nt,idivflag,ndiv
      integer iptr(8),ltree
      integer itree(ltree),iper
      integer ifunif,nlmin,nlmax
      double precision centers(ndim,nboxes),src(ndim,ns),targ(ndim,nt)
      real *8 bs0,cen0(ndim)
      integer, allocatable :: irefinebox(:)
      double precision boxsize(0:nlevels)
      integer, allocatable :: isrc(:),itarg(:),isrcse(:,:),itargse(:,:)

      integer i,ilev,irefine
      integer ifirstbox,ilastbox,nbctr,nbloc

      integer j,nboxes0
      integer ibox,nn,nss,ntt,mc,mnbors

      mc=2**ndim
      mnbors=3**ndim
c
      iptr(1) = 1
      iptr(2) = 2*(nlevels+1)+1
      iptr(3) = iptr(2) + nboxes
      iptr(4) = iptr(3) + nboxes
      iptr(5) = iptr(4) + nboxes
      iptr(6) = iptr(5) + mc*nboxes
      iptr(7) = iptr(6) + nboxes
      iptr(8) = iptr(7) + mnbors*nboxes
c
c     step 1: find enclosing box
c
      call pts_tree_boxsize0(ndim,iper,src,ns,targ,nt,
     1    bs0,cen0)

      boxsize(0) = bs0

      do i=1,ndim
         centers(i,1) = cen0(i)
      enddo

      allocate(isrc(ns),itarg(nt),isrcse(2,nboxes),itargse(2,nboxes))

c
c      set tree info for level 0
c
      itree(1) = 1
      itree(2) = 1
      itree(iptr(2)) = 0
      itree(iptr(3)) = -1
      itree(iptr(4)) = 0
      do i=1,mc
        itree(iptr(5)+i-1) = -1
      enddo

      isrcse(1,1) = 1
      isrcse(2,1) = ns
      itargse(1,1) = 1
      itargse(2,1) = nt

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,ns
        isrc(i) = i
      enddo
C$OMP END PARALLEL DO      

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,nt
        itarg(i) = i
      enddo
C$OMP END PARALLEL DO      


c
c       Reset nlevels, nboxes
c
      nbctr = 1

      do ilev=0,nlevels-1
        irefine = 0

        ifirstbox = itree(2*ilev+1) 
        ilastbox = itree(2*ilev+2)

        nbloc = ilastbox-ifirstbox+1
        allocate(irefinebox(nbloc))

c
c          determine which boxes need to be refined
c
c       
        if(ilev.ge.nlmin) then
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,ibox,nss,ntt,nn)
          do i=1,nbloc
            irefinebox(i) = 0
            ibox = ifirstbox + i-1
            nss = isrcse(2,ibox)-isrcse(1,ibox)+1
            ntt = itargse(2,ibox)-itargse(1,ibox)+1
          
            if(idivflag.eq.0) nn = nss
            if(idivflag.eq.1) nn = ntt
            if(idivflag.eq.2) nn = max(ntt,nss)

            if(nn.gt.ndiv) irefinebox(i) = 1
          enddo
C$OMP END PARALLEL DO        
          irefine = maxval(irefinebox(1:nbloc))

          if(ifunif.eq.1) then
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
            do i=1,nbloc
              irefinebox(i) = irefine
            enddo
C$OMP END PARALLEL DO 
          endif
        else
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
          do i=1,nbloc
            irefinebox(i) = 1
          enddo
C$OMP END PARALLEL DO 
          irefine = 1
        endif
        

        if(irefine.eq.1) then
          boxsize(ilev+1) = boxsize(ilev)/2
          itree(2*ilev+3) = nbctr+1

          call tree_refine_boxes(ndim,irefinebox,nboxes,
     1       ifirstbox,nbloc,centers,boxsize(ilev+1),nbctr,ilev+1,
     2       itree(iptr(2)),itree(iptr(3)),itree(iptr(4)),
     3       itree(iptr(5)))

c
c     re sort points in refined boxes
c
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,ibox)
C$OMP$SCHEDULE(DYNAMIC)
          do i=1,nbloc
            ibox = ifirstbox+i-1
            if(irefinebox(i).eq.1) then
              call sort_pts_to_children(ndim,ibox,nboxes,centers,
     1          itree(iptr(5)),src,ns,isrc,isrcse)
              call sort_pts_to_children(ndim,ibox,nboxes,centers,
     1          itree(iptr(5)),targ,nt,itarg,itargse)
            endif
          enddo
C$OMP END PARALLEL DO          

          
          itree(2*ilev+4) = nbctr
        else
          exit
        endif

        deallocate(irefinebox)
      enddo

      nboxes0 = nbctr
      nlevels = ilev


C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
      do i=1,nboxes0
        itree(iptr(6)+i-1) = 0
        do j=1,mnbors
          itree(iptr(7)+mnbors*(i-1)+j-1) = -1
        enddo
      enddo
C$OMP END PARALLEL DO      


      call computecoll(ndim,nlevels,nboxes0,itree(iptr(1)),boxsize,
     1    centers,itree(iptr(3)),itree(iptr(4)),itree(iptr(5)),iper,
     2    itree(iptr(6)),itree(iptr(7)))

      if(nlevels.ge.2.and.ifunif.ne.1) then
         call pts_tree_fix_lr(ndim,centers,nlevels,
     1       nboxes0,boxsize,nboxes,nlevels,iper,itree(iptr(1)),
     2       itree(iptr(2)),itree(iptr(3)),itree(iptr(4)),
     3       itree(iptr(5)),itree(iptr(6)),itree(iptr(7)))

      endif

      return
      end
c
c
c     sort points to child boxes. works for arbitrary dimensions
c     algorithm: bin sort
c     
      subroutine sort_pts_to_children(ndim,ibox,nboxes,centers,
     1   ichild,src,ns,isrc,isrcse)
      implicit real *8 (a-h,o-z)
      integer nboxes
      double precision centers(ndim,nboxes),src(ndim,ns)
      integer ns, isrc(ns),isrcse(2,nboxes)
      integer ichild(2**ndim,nboxes)
      integer iss, nsrc(2**ndim),ip(2**ndim+1)
      integer, allocatable :: isrcbox(:),isrctmp(:)

      mc=2**ndim
      if (ns.eq.0) then
         do i=1,mc
            jbox = ichild(i,ibox)
            isrcse(1,jbox) = 1
            isrcse(2,jbox) = 0
         enddo
         return
      endif
      
      npts = isrcse(2,ibox)-isrcse(1,ibox)+1
      allocate(isrcbox(npts))
      allocate(isrctmp(npts))

      do i=1,mc
         nsrc(i)=0
      enddo

      istart=isrcse(1,ibox)
      do iss=isrcse(1,ibox),isrcse(2,ibox)
         call find_childbox_ind(ndim,src(1,isrc(iss)),centers(1,ibox),k)
         isrcbox(iss-istart+1)=k
         nsrc(k)=nsrc(k)+1
      enddo

      ip(1)=0
      call cumsum(mc,nsrc,ip(2))
      do i=1,mc
         jbox = ichild(i,ibox)
         isrcse(1,jbox) = istart+ip(i)
         isrcse(2,jbox) = istart+ip(i+1)-1
      enddo

      do iss=isrcse(1,ibox),isrcse(2,ibox)
         ic=isrcbox(iss-istart+1)
         ip(ic)=ip(ic)+1
         isrctmp(ip(ic))=isrc(iss)
      enddo

      do iss=isrcse(1,ibox),isrcse(2,ibox)
         isrc(iss)=isrctmp(iss-istart+1)
      enddo
c
c

      return
      end
c       
c
c       
c
      subroutine find_childbox_ind(ndim,src,cen,k)
C
C     This subroutine returns the child box index of a given point
c     works for arbitrary dimension
c
C     INPUT
C     ndim - dimension of the underlysing space
c     src - coordinates of the point
C     cen - box center
C
C     OUTPUT:
C
C     k - child box index
C
      implicit real *8 (a-h,o-z)
      real *8 src(ndim),cen(ndim)
      integer k, i

C
      k=1
      inc=1
      do i=1,ndim
         dx = src(i)-cen(i)
         if (dx.ge.0) k=k+inc
         inc=inc*2
      enddo
c
      return
      end
c
C
c
c-------------------------------------------------------------      
      subroutine pts_tree_fix_lr(ndim,centers,nlevels,nboxes,
     1       boxsize,nbmax,nlmax,iper,laddr,ilevel,iparent,nchild,
     2       ichild,nnbors,nbors)
c
c     convert an adaptive tree into a level restricted tree
c     works for arbitrary dimension
c
      implicit none
      integer ndim,nlevels,nboxes,nlmax
      integer nbmax,iper
      double precision centers(ndim,nbmax),boxsize(0:nlmax)
      integer laddr(2,0:nlmax),ilevel(nbmax),iparent(nbmax)
      integer nchild(nbmax),ichild(2**ndim,nbmax),nnbors(nbmax)
      integer nbors(3**ndim,nbmax)
      integer laddrtail(2,0:nlmax)
      integer, allocatable :: iflag(:)

      integer i,j,k,ibox,jbox,kbox,ilev,idad,igranddad
      integer nbloc,ict,mc,mnbors,ifnbor
      double precision dis,distest

      mc = 2**ndim
      mnbors = 3**ndim
      
      allocate(iflag(nbmax))

c     Initialize flag array
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,nboxes
         iflag(i) = 0
      enddo
C$OMP END PARALLEL DO     



c     Flag boxes that violate level restriction by "1"
c     Violatioin refers to any box that is directly touching
c     a box that is more than one level finer
c
c     Method:
c     1) Carry out upward pass. For each box B, look at
c     the colleagues of B's grandparent
c     2) See if any of those colleagues are childless and in
c     contact with B.
c
c     Note that we only need to get up to level two, as
c     we will not find a violation at level 0 and level 1
c
c     For such boxes, we set iflag(i) = 1
c
      do ilev=nlevels,2,-1
c        This is the distance to test if two boxes separated
c        by two levels are touching
         distest = 1.05d0*(boxsize(ilev-1) + boxsize(ilev-2))/2.0d0
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ibox,idad,igranddad,i,jbox)         
C$OMP$ PRIVATE(ict,dis,k)
         do ibox = laddr(1,ilev),laddr(2,ilev) 
            idad = iparent(ibox)
            igranddad = iparent(idad)
            
c           Loop over colleagues of granddad            
            do i=1,nnbors(igranddad)
               jbox = nbors(i,igranddad)
c              Check if the colleague of grandad
c              is a leaf node. This automatically
c              eliminates the granddad
               if(nchild(jbox).eq.0.and.iflag(jbox).eq.0) then
                   ict = 0
                   do k=1,ndim
                      dis = centers(k,jbox) - centers(k,idad)
                      if(abs(dis).le.distest) ict = ict + 1
                   enddo
                   if(ict.eq.ndim) then
                      iflag(jbox) = 1
                   endif
               endif
c              End of checking criteria for the colleague of
c              granddad
            enddo
c           End of looping over colleagues of
c           granddad
         enddo
c        End of looping over boxes at ilev         
C$OMP END PARALLEL DO
      enddo
c     End of looping over levels and flagging boxes


c     Find all boxes that need to be given a flag+
c     A flag+ box will be denoted by setting iflag(box) = 2
c     This refers to any box that is not already flagged and
c     is bigger than and is contacting a flagged box
c     or another box that has already been given a flag +.
c     It is found by performing an upward pass and looking
c     at the flagged box's parents colleagues and a flag+
c     box's parents colleagues and seeing if they are
c     childless and present the case where a bigger box 
c     is contacting a flagged or flag+ box.

      do ilev = nlevels,1,-1
c        This is the distance to test if two boxes separated
c        by one level are touching
         distest = 1.05d0*(boxsize(ilev) + boxsize(ilev-1))/2.0d0
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ibox,idad,i,jbox,dis)
C$OMP$PRIVATE(k,ict)
         do ibox = laddr(1,ilev),laddr(2,ilev)
            if(iflag(ibox).eq.1.or.iflag(ibox).eq.2) then
               idad = iparent(ibox)
c              Loop over dad's colleagues               
               do i=1,nnbors(idad)
                  jbox = nbors(i,idad)
c                 Check if the colleague of dad
c                 is a leaf node. This automatically
c                 eliminates the dad
                  if(nchild(jbox).eq.0.and.iflag(jbox).eq.0) then
                     ict = 0
                     do k=1,ndim
                        dis = centers(k,jbox) - centers(k,ibox)
                        if(abs(dis).le.distest) ict = ict + 1
                     enddo
                     if(ict.eq.ndim) then
                        iflag(jbox) = 2
                     endif
                  endif
c                 End of checking criteria for the colleague of
c                dad
               enddo
c              End of looping over dad's colleagues               
            endif
c           End of checking if current box is relevant for
c           flagging flag+ boxes
         enddo
c        End of looping over boxes at ilev        
C$OMP END PARALLEL DO 
      enddo
c     End of looping over levels

c     Subdivide all flag and flag+ boxes. Flag all the children
c     of flagged boxes as flag++. Flag++ boxes are denoted
c     by setting iflag(box) = 3. The flag++ boxes need 
c     to be checked later to see which of them need further
c     refinement. While creating new boxes, we will
c     need to update all the tree structures as well.
c     Note that all the flagged boxes live between
c     levels 1 and nlevels - 2. We process the boxes via a
c     downward pass. We first determine the number of boxes
c     that are going to be subdivided at each level and 
c     everything else accordingly
      do ilev = 0,nlevels
         laddrtail(1,ilev) = 0
         laddrtail(2,ilev) = -1
      enddo

 
      do ilev = 1,nlevels-2
c        First subdivide all the flag and flag+
c        boxes with boxno nboxes+1, nboxes+ 2
c        and so on. In the second step, we reorganize
c        all the structures again to bring it back
c        in the standard format

         laddrtail(1,ilev+1) = nboxes+1


         nbloc = laddr(2,ilev)-laddr(1,ilev)+1
         call tree_refine_boxes_flag(ndim,iflag,nbmax,laddr(1,ilev),
     1    nbloc,centers,boxsize(ilev+1),nboxes,ilev,ilevel,iparent,
     2    nchild,ichild)


         laddrtail(2,ilev+1) = nboxes
      enddo
c     Reorganize the tree to get it back in the standard format

      call pts_tree_reorg(ndim,nboxes,centers,nlevels,laddr,
     1          laddrtail,ilevel,iparent,nchild,ichild,
     2          iflag)

c     Compute colleague information again      

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
      do i=1,nboxes
         nnbors(i) = 0
         do j=1,mnbors
            nbors(j,i) = -1
         enddo
      enddo
C$OMP END PARALLEL DO      
      call computecoll(ndim,nlevels,nboxes,laddr, boxsize,
     1                   centers,iparent,nchild,
     2                   ichild,iper,nnbors,nbors)

c     Processing of flag and flag+ boxes is done
c     Start processing flag++ boxes. We will use a similar
c     strategy as before. We keep checking the flag++
c     boxes that require subdivision if they still
c     violate the level restriction criterion, create
c     the new boxes, append them to the end of the list to begin
c     with and in the end reorganize the tree structure.
c     We shall accomplish this via a downward pass
c     as new boxes that get added in the downward pass
c     will also be processed simultaneously.
c     We shall additionally also need to keep on updating
c     the colleague information as we proceed in the 
c     downward pass

c     Reset the flags array to remove all the flag and flag+
c     cases. This is to ensure reusability of the subdivide
c     _flag routine to handle the flag++ case

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ibox)
      do ibox=1,nboxes
         if(iflag(ibox).ne.3) iflag(ibox) = 0
      enddo
C$OMP END PARALLEL DO      
 
      do ilev = 0,nlevels
         laddrtail(1,ilev) = 0
         laddrtail(2,ilev) = -1
      enddo


      do ilev = 2,nlevels-2

c     Step 1: Determine which of the flag++ boxes need
c     further division. In the even a flag++ box needs
c     further subdivision then flag the box with iflag(box) = 1
c     This will again ensure that the subdivide_flag routine
c     will take care of handling the flag++ case
         call updateflags(ndim,ilev,nboxes,nlevels,laddr,nchild,
     1       ichild,nnbors,nbors,centers,boxsize,iflag)

         call updateflags(ndim,ilev,nboxes,nlevels,laddrtail,nchild,
     1       ichild,nnbors,nbors,centers,boxsize,iflag)
        
c      Step 2: Subdivide all the boxes that need subdivision
c      in the laddr set and the laddrtail set as well
         laddrtail(1,ilev+1) = nboxes + 1

         nbloc = laddr(2,ilev)-laddr(1,ilev)+1
         call tree_refine_boxes_flag(ndim,iflag,nbmax,laddr(1,ilev),
     1    nbloc,centers,boxsize(ilev+1),nboxes,ilev,ilevel,iparent,
     2    nchild,ichild)

         nbloc = laddrtail(2,ilev)-laddrtail(1,ilev)+1
         call tree_refine_boxes_flag(ndim,iflag,nbmax,laddrtail(1,ilev),
     1    nbloc,centers,boxsize(ilev+1),nboxes,ilev,ilevel,iparent,
     2    nchild,ichild)

         laddrtail(2,ilev+1) = nboxes         
c      Step 3: Update the colleague information for the newly
c      created boxes

C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,i,idad,jbox,j,kbox,k,ifnbor)
          do ibox = laddrtail(1,ilev+1),laddrtail(2,ilev+1)
            nnbors(ibox) = 0
c           Find the parent of the current box         
            idad = iparent(ibox)
c           Loop over the neighbors of the parent box
c           to find out colleagues
            do i=1,nnbors(idad)
                jbox = nbors(i,idad)
                do j=1,mc
c               ichild(j,jbox) is one of the children of the
c               neighbors of the parent of the current
c               box
                   kbox = ichild(j,jbox)
                   if(kbox.gt.0) then
c     Check if kbox is a nearest neighbor or in list 2
                      ifnbor=1
                      do k=1,ndim
                         if((abs(centers(k,kbox)-centers(k,ibox)).gt.
     1                       1.05*boxsize(ilev+1))) then
                            ifnbor=0
                            exit
                         endif
                      enddo
                      if (ifnbor.eq.1) then
                         nnbors(ibox) = nnbors(ibox)+1
                         nbors(nnbors(ibox),ibox) = kbox
                      endif
                   endif
                enddo
            enddo
c           End of computing colleagues of box i
         enddo
C$OMP END PARALLEL DO         
      enddo

c     Reorganize tree once again and we are all done      
      call pts_tree_reorg(ndim,nboxes,centers,nlevels,laddr,
     1          laddrtail,ilevel,iparent,nchild,ichild,
     2          iflag)

c     Compute colleague information again      

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
      do i=1,nboxes
         nnbors(i) = 0
         do j=1,mnbors
            nbors(j,i) = -1
         enddo
      enddo
C$OMP END PARALLEL DO    

      call computecoll(ndim,nlevels,nboxes,laddr, boxsize,
     1                   centers,iparent,nchild,
     2                   ichild,iper,nnbors,nbors)
      

      return
      end
      

c-------------------------------------------------------------      

      subroutine pts_tree_reorg(ndim,nboxes,centers,nlevels,laddr,
     1     laddrtail,ilevel,iparent,nchild,ichild,iflag)

c    This subroutine reorganizes the current data in all the tree
c    arrays to rearrange them in the standard format.
c    The boxes on input are assumed to be arranged in the following
c    format
c    boxes on level i are the boxes from laddr(1,i) to 
c    laddr(2,i) and also from laddrtail(1,i) to laddrtail(2,i)
c
c    At the end of the sorting, the boxes on level i
c    are arranged from laddr(1,i) to laddr(2,i)  
c
c    INPUT/OUTPUT arguments
c    ndim           in: integer
c                   dimension of the space
c
c    nboxes         in: integer
c                   number of boxes
c
c    centers        in/out: double precision(3,nboxes)
c                   x and y coordinates of the center of boxes
c
c    nlevels        in: integer
c                   Number of levels in the tree
c
c    laddr          in/out: integer(2,0:nlevels)
c                   boxes at level i are numbered between
c                   laddr(1,i) to laddr(2,i)
c
c    laddrtail      in: integer(2,0:nlevels)
c                   new boxes to be added to the tree
c                   structure are numbered from
c                   laddrtail(1,i) to laddrtail(2,i)
c
c    ilevel      in/out: integer(nboxes)
c                ilevel(i) is the level of box i
c
c    iparent     in/out: integer(nboxes)
c                 iparent(i) is the parent of box i
c
c    nchild      in/out: integer(nboxes)
c                nchild(i) is the number of children 
c                of box i
c
c    ichild       in/out: integer(4,nboxes)
c                 ichild(j,i) is the jth child of box i
c
c    iflag        in/out: integer(nboxes)
c                 iflag(i) is a flag for box i required to generate
c                 level restricted tree from adaptive tree

      implicit none
c     Calling sequence variables and temporary variables
      integer ndim,nboxes,nlevels
      double precision centers(ndim,nboxes)
      integer laddr(2,0:nlevels), tladdr(2,0:nlevels)
      integer laddrtail(2,0:nlevels)
      integer ilevel(nboxes)
      integer iparent(nboxes)
      integer nchild(nboxes)
      integer ichild(2**ndim,nboxes)
      integer iflag(nboxes)
      
      integer, allocatable :: tilevel(:),tiparent(:),tnchild(:)
      integer, allocatable :: tichild(:,:),tiflag(:)
      integer, allocatable :: iboxtocurbox(:),ilevptr(:),ilevptr2(:)

      double precision, allocatable :: tcenters(:,:)



c     Temporary variables
      integer i,k,mc
      integer ibox,ilev, curbox,nblev

      mc=2**ndim

      allocate(tilevel(nboxes),tiparent(nboxes),tnchild(nboxes))
      allocate(tichild(mc,nboxes),tiflag(nboxes),iboxtocurbox(nboxes))
      allocate(tcenters(ndim,nboxes))

      do ilev = 0,nlevels
         tladdr(1,ilev) = laddr(1,ilev)
         tladdr(2,ilev) = laddr(2,ilev)
      enddo
      call tree_copy(ndim,nboxes,centers,ilevel,iparent,nchild,
     1            ichild,tcenters,tilevel,tiparent,
     2            tnchild,tichild)

      do ibox=1,nboxes
         tiflag(ibox) = iflag(ibox)
      enddo
     
c     Rearrange old arrays now

      do ilev = 0,0
         do ibox = laddr(1,ilev),laddr(2,ilev)
           iboxtocurbox(ibox) = ibox
         enddo
      enddo

      allocate(ilevptr(nlevels+1),ilevptr2(nlevels))

      ilevptr(2) = laddr(1,1)

      do ilev=1,nlevels
        nblev = laddr(2,ilev)-laddr(1,ilev)+1
        ilevptr2(ilev) = ilevptr(ilev) + nblev
        nblev = laddrtail(2,ilev)-laddrtail(1,ilev)+1
        ilevptr(ilev+1) = ilevptr2(ilev) + nblev
      enddo

      curbox = laddr(1,1)
      do ilev=1,nlevels
         laddr(1,ilev) = curbox
         do ibox = tladdr(1,ilev),tladdr(2,ilev)
            ilevel(curbox) = tilevel(ibox)
            nchild(curbox) = tnchild(ibox)
            do k=1,ndim
               centers(k,curbox) = tcenters(k,ibox)
            enddo
            iflag(curbox) = tiflag(ibox)
            iboxtocurbox(ibox) = curbox

            curbox = curbox + 1
         enddo
         do ibox = laddrtail(1,ilev),laddrtail(2,ilev)
            ilevel(curbox) = tilevel(ibox)
            do k=1,ndim
               centers(k,curbox) = tcenters(k,ibox)
            enddo
            nchild(curbox) = tnchild(ibox)
            iflag(curbox) = tiflag(ibox)
            iboxtocurbox(ibox) = curbox

            curbox = curbox + 1
         enddo
         laddr(2,ilev) = curbox-1
      enddo

c     Handle the parent children part of the tree 
c     using the mapping iboxtocurbox

      do ibox=1,nboxes
         if(tiparent(ibox).eq.-1) iparent(iboxtocurbox(ibox)) = -1
         if(tiparent(ibox).gt.0) 
     1    iparent(iboxtocurbox(ibox)) = iboxtocurbox(tiparent(ibox))
         do i=1,mc
            if(tichild(i,ibox).eq.-1) ichild(i,iboxtocurbox(ibox)) = -1
            if(tichild(i,ibox).gt.0) 
     1      ichild(i,iboxtocurbox(ibox)) = iboxtocurbox(tichild(i,ibox))
         enddo
      enddo

      return
      end
c
c
c
c
c
c
      subroutine pts_tree_sort(ndim,n,xys,itree,ltree,nboxes,nlevels,
     1    iptr,centers,ixy,ixyse)
c     sort points to the tree, works for arbitrary dimension
      implicit double precision (a-h,o-z)
      integer iptr(8),ltree
      integer n,nboxes,nlevels,itree(ltree)
      integer ixy(n),ixyse(2,nboxes)
      double precision xys(ndim,n),centers(ndim,nboxes)

      do i=1,n
        ixy(i) = i
      enddo

      ixyse(1,1) = 1
      ixyse(2,1) = n

      do ilev = 0,nlevels-1
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox)
C$OMP$SCHEDULE(DYNAMIC)  
        do ibox=itree(2*ilev+1),itree(2*ilev+2)
          if(itree(iptr(4)+ibox-1).gt.0) then
            call sort_pts_to_children(ndim,ibox,nboxes,centers,
     1          itree(iptr(5)),xys,n,ixy,ixyse)
          endif
        enddo
C$OMP END PARALLEL DO         
      enddo

      return
      end

c
c
c--------------------------------------------------------------
c
      subroutine subdividebox_new(ndim,pos,npts,center,boxsize,
     1           isorted,iboxfl,subcenters)
      implicit none
      double precision pos(ndim,npts)
      double precision center(ndim)
      double precision subcenters(ndim,2**ndim)
      double precision boxsize,bsh
      integer ndim,npts
      integer isorted(*)
      integer iboxfl(2,2**ndim)
      double precision centerstmp(ndim,2**ndim+1)
      integer iboxfltmp(2,2**ndim+1),ichild(2**ndim,2**ndim+1)
      integer nboxes
      integer i,ibox,j,mc,k
      integer isgn(ndim,2**ndim)
      
      call get_child_box_sign(ndim,isgn)
      
      bsh = boxsize/2.0d0

      mc = 2**ndim
      nboxes = mc+1

      iboxfltmp(1,1) = 1
      iboxfltmp(2,1) = npts

      do i=1,ndim
         centerstmp(i,1) = center(i)
      enddo

      do i=1,mc
        ichild(i,1) = i+1
      enddo

      do i=2,mc+1
        do j=1,mc
          ichild(j,i) = -1
        enddo
      enddo

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,npts
        isorted(i) = i
      enddo
C$OMP END PARALLEL DO      

      do i=1,mc
         do k=1,ndim
            subcenters(k,i) = center(k)+isgn(k,i)*bsh
         enddo
         do k=1,ndim
            centerstmp(k,i+1) = subcenters(k,i)
         enddo
      enddo
      ibox = 1
      call sort_pts_to_children(ndim,ibox,nboxes,centerstmp,ichild,
     1    pos,npts,isorted,iboxfltmp)
      do i=1,mc
        iboxfl(1,i) = iboxfltmp(1,i+1)
        iboxfl(2,i) = iboxfltmp(2,i+1)
      enddo

      return
      end
c--------------------------------------------------------------------     
c
c
c
c
      subroutine pts_tree_boxsize0(ndim,iperiod,
     1    src,ns,targ,nt,bs0,cen0)
c      
c     given a collection of sources and targs, this subroutine returns to the user 
c     the cutoff lev, the side length and the center of the bounding box.
c      
c     
c     input parameters:
c     ns            : number of sources
c     src(3,ns)     : source locations
c     nt            : number of targets
c     targ(3,nt)    : target locationsc     
c     iperiod       : 1: periodic; 0: free-space 
c      
c     output parameters:
c
c     parameters: input when iperiod=1, output when iperiod=0
c     bs0 : the side length of the bounding box
c     cen0(3) : the center of the bounding box
c     
c      
      implicit real *8 (a-h,o-z)
      integer ns,nt
      real *8 src(ndim,ns),targ(ndim,nt),bs0,cen0(ndim)
      real *8 xyzmin(ndim),xyzmax(ndim)

      if (iperiod.eq.1) goto 1200
      do i=1,ndim
         xyzmin(i) = src(i,1)
         xyzmax(i) = src(i,1)
      enddo
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,k)
C$OMP$REDUCTION(min:xyzmin)
C$OMP$REDUCTION(max:xyzmax)
      do i=1,ns
         do k=1,ndim
            if(src(k,i).lt.xyzmin(k)) xyzmin(k) = src(k,i)
            if(src(k,i).gt.xyzmax(k)) xyzmax(k) = src(k,i)
         enddo
      enddo
C$OMP END PARALLEL DO      

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,k)
C$OMP$REDUCTION(min:xyzmin)
C$OMP$REDUCTION(max:xyzmax)
      do i=1,nt
         do k=1,ndim
            if(targ(k,i).lt.xyzmin(k)) xyzmin(k) = targ(k,i)
            if(targ(k,i).gt.xyzmax(k)) xyzmax(k) = targ(k,i)
         enddo
      enddo
C$OMP END PARALLEL DO      

      bs0 = (xyzmax(1) - xyzmin(1))
      do k=2,ndim
         sizek = xyzmax(k)-xyzmin(k)
         if(sizek.gt.bs0) bs0 = sizek
      enddo

      do i=1,ndim
         cen0(i) = (xyzmin(i)+xyzmax(i))/2
      enddo

 1200 continue
      
      return
      end
c      
c
c      
      subroutine get_child_box_sign(ndim,isgn)
c     This subroutine computes the signs of all child boxes
c     The convention is as follows.
c      
c     1d - 1:m 2:p
c      
c     2d - 1:mm 
c          2:pm
c          3:mp
c          4:pp
c      
c     3d - 1:mmm
c          2:pmm
c          3:mpm
c          4:ppm
c          5:mmp
c          6:pmp
c          7:mpp
c          8:ppp
c
c     input:
c     ndim - dimension of the underlying space
c
c     output:
c     isgn - integer(ndim,2**ndim)
c            the signs of the center coordinates of each child box,
c            when the center of the parent box is at the origin
c
      implicit real *8 (a-h,o-z)
      integer isgn(ndim,2**ndim)

      mc = 2**ndim
      do j=1,ndim
         isgn(j,1)=-1
      enddo

      do j=1,ndim
         do i=1,mc,2**(j-1)
            if (i.gt.1) isgn(j,i)=-isgn(j,i-2**(j-1))
            do k=1,2**(j-1)-1
               isgn(j,i+k)=isgn(j,i)
            enddo
         enddo
      enddo

      return
      end
c     
c      
c      
c      
c
C
c
c-------------------------------------------------------------      
      subroutine pts_tree_refine_once(ndim,centers,nlevels,nboxes,
     1       boxsize,nbmax,nlmax,iper,laddr,ilevel,iparent,nchild,
     2       ichild,nnbors,nbors,isrcse,itargse)
c
c     refine all leaf boxes once 
c
      implicit none
      integer ndim,nlevels,nboxes,nlmax
      integer nbmax,iper
      double precision centers(ndim,nbmax),boxsize(0:nlmax)
      integer laddr(2,0:nlmax),ilevel(nbmax),iparent(nbmax)
      integer nchild(nbmax),ichild(2**ndim,nbmax),nnbors(nbmax)
      integer nbors(3**ndim,nbmax)
      integer isrcse(2,nbmax),itargse(2,nbmax)
      integer laddrtail(2,0:nlmax)
      integer, allocatable :: iflag(:)

      integer i,j,k,ibox,jbox,kbox,ilev,idad,igranddad
      integer nbloc,ict,mc,mnbors,ifnbor,npts
      double precision dis,distest

      boxsize(nlevels+1)=boxsize(nlevels)/2
      
      mc = 2**ndim
      mnbors = 3**ndim
      
      allocate(iflag(nbmax))

c     Initialize flag array
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,nboxes
         iflag(i) = 0
      enddo
C$OMP END PARALLEL DO     

c     Flag leaf boxes by "1"
      do ilev=0,nlevels
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ibox)         
         do ibox = laddr(1,ilev),laddr(2,ilev)
cccc            npts=isrcse(2,ibox)-isrcse(1,ibox)+1
cccc     1          +itargse(2,ibox)-itargse(1,ibox)+1
            if (nchild(ibox).eq.0) iflag(ibox)=2
cccc            if (nchild(ibox).eq.0 .and. npts.gt.0) iflag(ibox)=2
         enddo
c        End of looping over boxes at ilev         
C$OMP END PARALLEL DO
      enddo
c     End of looping over levels and flagging boxes

c     Subdivide all flag boxes.
      do ilev = 0,nlevels
         laddrtail(1,ilev) = 0
         laddrtail(2,ilev) = -1
      enddo
      laddr(1,nlevels+1)=nboxes+1
      laddr(2,nlevels+1)=nboxes
 
      do ilev = 0,nlevels
         laddrtail(1,ilev+1) = nboxes+1

         nbloc = laddr(2,ilev)-laddr(1,ilev)+1
         call tree_refine_boxes_flag(ndim,iflag,nbmax,laddr(1,ilev),
     1    nbloc,centers,boxsize(ilev+1),nboxes,ilev,ilevel,iparent,
     2    nchild,ichild)

         laddrtail(2,ilev+1) = nboxes
      enddo
      
c     Reorganize the tree to get it back in the standard format
      nlevels=nlevels+1
      call pts_tree_reorg(ndim,nboxes,centers,nlevels,laddr,
     1          laddrtail,ilevel,iparent,nchild,ichild,
     2          iflag)

c     Compute colleague information again      

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
      do i=1,nboxes
         nnbors(i) = 0
         do j=1,mnbors
            nbors(j,i) = -1
         enddo
      enddo
C$OMP END PARALLEL DO      
      call computecoll(ndim,nlevels,nboxes,laddr, boxsize,
     1                   centers,iparent,nchild,
     2                   ichild,iper,nnbors,nbors)

      return
      end
      



      subroutine pts_tree_refine_once_mem(ndim,src,ns,targ,nt,idivflag,
     1    ndiv,nlmin,nlmax,ifunif,iper,
     2    nlevels,nboxes,ltree)
c
c
c
c----------------------------------------
c  get memory requirements for the tree
c
c
c  input parameters:
c    - ndim: dimension of the underlying space
c    - src: real *8 (ndim,ns)
c        source locations
c    - targ: real *8 (ndim,nt) 
c        target locations
c    - idivflag: integer
c        subdivision criterion
c          * divflag = 0 -> subdivide on sources only
c          * idivflag = 1 -> subdivide on targets only
c          * idivflag = 2 -> subdivide on max(sources+targets)
c    - ndiv: integer
c        subdivide if relevant number of particles
c        per box is greater than ndiv
c    - nlmin: integer
c        minimum number of levels of uniform refinement.
c        Note that empty boxes are not pruned along the way
c    - nlmax: integer
c        max number of levels
c    - ifunif: integer
c        flag for creating uniform pruned tree
c        Tree is uniform if ifunif=1 (Currently pruned part
c        under construction)
c    - iper: integer
c        flag for periodic implementations. 
c    - bs0 : real *8
c        side length of the bounding box
c    - cen0(ndim) : center of the bounding box
c        
c  output parameters
c    - nlevels: integer
c        number of levels
c    - nboxes: integer
c        number of boxes
c    - ltree: integer
c        length of tree
c----------------------------------
c
      implicit none
      integer ndim,nlevels,nboxes,idivflag
      integer ltree
      integer nbmax,nbtot
      integer ns,nt,ndiv
      integer nlmin,iper,ifunif
      double precision src(ndim,ns),targ(ndim,nt),cen0(ndim),bs0

      integer, allocatable :: laddr(:,:),ilevel(:),iparent(:),nchild(:)
      integer, allocatable :: ichild(:,:)
      double precision, allocatable :: centers(:,:)
      integer, allocatable :: nbors(:,:),nnbors(:)

      integer, allocatable :: isrc(:),itarg(:),isrcse(:,:),itargse(:,:)

      integer, allocatable :: ilevel2(:),iparent2(:),nchild2(:),
     1    ichild2(:,:),isrcse2(:,:),itargse2(:,:)
      double precision, allocatable :: centers2(:,:)

      integer nlmax
      integer i,j,mc,mnbors

      double precision, allocatable :: boxsize(:)
      integer, allocatable :: irefinebox(:)

      integer nbloc,nbctr,nbadd,irefine,ilev,ifirstbox,ilastbox
      integer ibox,nn,nss,ntt

      nbmax = 100 000
      mc = 2**ndim
      mnbors = 3**ndim
      
      allocate(boxsize(0:nlmax))

      
      allocate(laddr(2,0:nlmax),ilevel(nbmax),iparent(nbmax))
      allocate(nchild(nbmax),ichild(mc,nbmax))

      allocate(centers(ndim,nbmax),isrcse(2,nbmax),itargse(2,nbmax))
      allocate(isrc(ns),itarg(nt))

c
c     step 1: find enclosing box
c
      call pts_tree_boxsize0(ndim,iper,src,ns,targ,nt,
     1    bs0,cen0)
      
      boxsize(0) = bs0

      do i=1,ndim
         centers(i,1) = cen0(i)
      enddo
      
c
c      set tree info for level 0
c
      laddr(1,0) = 1
      laddr(2,0) = 1
      ilevel(1) = 0
      iparent(1) = -1
      nchild(1) = 0
      do i=1,mc
        ichild(i,1) = -1
      enddo

      isrcse(1,1) = 1
      isrcse(2,1) = ns
      
      itargse(1,1) = 1
      itargse(2,1) = nt

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,ns
        isrc(i) = i
      enddo
C$OMP END PARALLEL DO      

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,nt
        itarg(i) = i
      enddo
C$OMP END PARALLEL DO      


      nbctr = 1

      do ilev=0,nlmax-1
        irefine = 0

        ifirstbox = laddr(1,ilev) 
        ilastbox = laddr(2,ilev)

        nbloc = ilastbox-ifirstbox+1
        allocate(irefinebox(nbloc))
c
c          determine which boxes need to be refined
c


        if(ilev.ge.nlmin) then
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,ibox,nss,ntt,nn)
          do i=1,nbloc
            irefinebox(i) = 0
            ibox = ifirstbox + i-1
            nss = isrcse(2,ibox)-isrcse(1,ibox)+1
            ntt = itargse(2,ibox)-itargse(1,ibox)+1
            if(idivflag.eq.0) nn = nss
            if(idivflag.eq.1) nn = ntt
            if(idivflag.eq.2) nn = max(ntt,nss)

            if(nn.gt.ndiv) irefinebox(i) = 1
          enddo
C$OMP END PARALLEL DO        

          irefine = maxval(irefinebox(1:nbloc))
          if(ifunif.eq.1) then
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)         
            do i=1,nbloc
              irefinebox(i) = irefine
            enddo
C$OMP END PARALLEL DO            
          endif
        else
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
          do i=1,nbloc
            irefinebox(i) = 1
          enddo
C$OMP END PARALLEL DO         
          irefine = 1
        endif

c
c
c          figure out if current set of boxes is sufficient
c

        nbadd = 0 
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i) REDUCTION(+:nbadd)        
        do i=1,nbloc
          if(irefinebox(i).eq.1) nbadd = nbadd+mc
        enddo
C$OMP END PARALLEL DO        

        nbtot = nbctr+nbadd

c
c         if current memory is not sufficient reallocate
c
        if(nbtot.gt.nbmax) then
          print *, "Reallocating"
          allocate(centers2(ndim,nbmax),ilevel2(nbmax),iparent2(nbmax))
          allocate(nchild2(nbmax),ichild2(mc,nbmax),isrcse2(2,nbmax))
          allocate(itargse2(2,nbmax))

          call tree_copy(ndim,nbctr,centers,ilevel,iparent,nchild,
     1            ichild,centers2,ilevel2,iparent2,
     2            nchild2,ichild2)

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)     
          do i=1,nbctr
            isrcse2(1,i) = isrcse(1,i)
            isrcse2(2,i) = isrcse(2,i)
            itargse2(1,i) = itargse(1,i)
            itargse2(2,i) = itargse(2,i)
          enddo
C$OMP END PARALLEL DO          


          deallocate(centers,ilevel,iparent,nchild,ichild,
     1        isrcse,itargse)

          nbmax = nbtot
          allocate(centers(ndim,nbmax),ilevel(nbmax),iparent(nbmax))
          allocate(nchild(nbmax),ichild(mc,nbmax),isrcse(2,nbmax))
          allocate(itargse(2,nbmax))


          call tree_copy(ndim,nbctr,centers2,ilevel2,iparent2,
     1            nchild2,ichild2,centers,ilevel,iparent,nchild,ichild)

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
          do i=1,nbctr
            isrcse(1,i) = isrcse2(1,i)
            isrcse(2,i) = isrcse2(2,i)
            itargse(1,i) = itargse2(1,i)
            itargse(2,i) = itargse2(2,i)
          enddo
C$OMP END PARALLEL DO          

          deallocate(centers2,ilevel2,iparent2,nchild2,ichild2,
     1       isrcse2,itargse2)
        endif


        if(irefine.eq.1) then
          boxsize(ilev+1) = boxsize(ilev)/2
          laddr(1,ilev+1) = nbctr+1
          call tree_refine_boxes(ndim,irefinebox,nbmax,
     1       ifirstbox,nbloc,centers,boxsize(ilev+1),nbctr,ilev+1,
     2       ilevel,iparent,nchild,ichild)

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,ibox)
C$OMP$SCHEDULE(DYNAMIC)  
          do i=1,nbloc
            ibox = ifirstbox+i-1
            if(irefinebox(i).eq.1) then
              call sort_pts_to_children(ndim,ibox,nbmax,centers,ichild,
     1            src,ns,isrc,isrcse)
              call sort_pts_to_children(ndim,ibox,nbmax,centers,ichild,
     1            targ,nt,itarg,itargse)
            endif
          enddo
C$OMP END PARALLEL DO          

          laddr(2,ilev+1) = nbctr
        else
          exit
        endif

        deallocate(irefinebox)
      enddo

      nboxes = nbctr
      nlevels = ilev

      if(nlevels.ge.0.and.ifunif.ne.1) then

cccc        nbtot = 2*mc*nboxes
        nbtot = 3*mc*nboxes
        if(nbtot.gt.nbmax) then
          allocate(centers2(ndim,nbmax),ilevel2(nbmax),iparent2(nbmax))
          allocate(nchild2(nbmax),ichild2(mc,nbmax),isrcse2(2,nbmax))
          allocate(itargse2(2,nbmax))
          call tree_copy(ndim,nbctr,centers,ilevel,iparent,nchild,
     1            ichild,centers2,ilevel2,iparent2,
     2            nchild2,ichild2)

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)     
          do i=1,nbctr
            isrcse2(1,i) = isrcse(1,i)
            isrcse2(2,i) = isrcse(2,i)
            itargse2(1,i) = itargse(1,i)
            itargse2(2,i) = itargse(2,i)
          enddo
C$OMP END PARALLEL DO          

          deallocate(centers,ilevel,iparent,nchild,ichild,
     1        isrcse,itargse)

          nbmax = nbtot
          allocate(centers(ndim,nbmax),ilevel(nbmax),iparent(nbmax))
          allocate(nchild(nbmax),ichild(mc,nbmax),isrcse(2,nbmax))
          allocate(itargse(2,nbmax))


          call tree_copy(ndim,nbctr,centers2,ilevel2,iparent2,
     1            nchild2,ichild2,centers,ilevel,iparent,nchild,ichild)

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
          do i=1,nbctr
            isrcse(1,i) = isrcse2(1,i)
            isrcse(2,i) = isrcse2(2,i)
            itargse(1,i) = itargse2(1,i)
            itargse(2,i) = itargse2(2,i)
          enddo
C$OMP END PARALLEL DO          

          deallocate(centers2,ilevel2,iparent2,nchild2,ichild2,
     1       isrcse2,itargse2)

        endif

        allocate(nnbors(nbmax))
        allocate(nbors(mnbors,nbmax))
        do i=1,nboxes
          nnbors(i) = 0
          do j=1,mnbors
            nbors(j,i) = -1
          enddo
        enddo

        call computecoll(ndim,nlevels,nboxes,laddr,boxsize,centers,
     1        iparent,nchild,ichild,iper,nnbors,nbors)
        print *, nlevels, nboxes

        if(nlevels.ge.2.and.ifunif.ne.1) then
          call pts_tree_fix_lr(ndim,centers,nlevels,nboxes,
     1         boxsize,nbmax,nlmax,iper,laddr,ilevel,iparent,nchild,
     2         ichild,nnbors,nbors)
        endif
        print *, nlevels, nboxes

        call pts_tree_refine_once(ndim,centers,nlevels,nboxes,
     1       boxsize,nbmax,nlmax,iper,laddr,ilevel,iparent,nchild,
     2       ichild,nnbors,nbors,isrcse,itargse)
        print *, nlevels, nboxes
        
      endif

      ltree = (4+mc+mnbors)*nboxes + 2*(nlevels+1) 

      return
      end
c
c
c
c
c

      subroutine pts_tree_refine_once_build(ndim,src,ns,targ,nt,
     1    idivflag,ndiv,nlmin,nlmax,ifunif,iper,nlevels,nboxes,
     2    ltree,itree,iptr,centers,
     3    boxsize)
c
c
c
c----------------------------------------
c  build tree
c
c
c input parameters:
c    - ndim: dimension of the underlying space
c    - src: real *8 (ndim,ns)
c        source locations
c    - targ: real *8 (ndim,nt) 
c        target locations
c    - idivflag: integer
c        subdivision criterion
c          * divflag = 0 -> subdivide on sources only
c          * idivflag = 1 -> subdivide on targets only
c          * idivflag = 2 -> subdivide on max(sources+targets)
c    - ndiv: integer
c        subdivide if relevant number of particles
c        per box is greater than ndiv
c    - nlmin: integer
c        minimum number of levels of uniform refinement.
c        Note that empty boxes are not pruned along the way
c    - nlmax: integer
c        max number of levels
c    - ifunif: integer
c        flag for creating uniform pruned tree
c        Tree is uniform if ifunif=1 (Currently pruned part
c        under construction)
c    - iper: integer
c        flag for periodic implementations. Currently unused.
c        Feature under construction
c    - nlevels: integer
c        number of levels
c    - nboxes: integer
c        number of boxes
c    - ltree: integer
c     - bs0 : real
c        side length of the bounding box
c     - cen0(3) : center of the bounding box
c
c  output:
c    - itree: integer(ltree)
c        tree info
c    - iptr: integer(8)
c        * iptr(1) - laddr
c        * iptr(2) - ilevel
c        * iptr(3) - iparent
c        * iptr(4) - nchild
c        * iptr(5) - ichild
c        * iptr(6) - ncoll
c        * iptr(7) - coll
c        * iptr(8) - ltree
c    - centers: double precision (dim,nboxes)
c        coordinates of box centers in the oct tree
c    - boxsize: double precision (0:nlevels)
c        size of box at each of the levels
c

      implicit none
      integer ndim,nlevels,nboxes,ns,nt,idivflag,ndiv
      integer iptr(8),ltree
      integer itree(ltree),iper
      integer ifunif,nlmin,nlmax
      double precision centers(ndim,nboxes),src(ndim,ns),targ(ndim,nt)
      real *8 bs0,cen0(ndim)
      integer, allocatable :: irefinebox(:)
      double precision boxsize(0:nlevels)
      integer, allocatable :: isrc(:),itarg(:),isrcse(:,:),itargse(:,:)

      integer i,ilev,irefine
      integer ifirstbox,ilastbox,nbctr,nbloc

      integer j,nboxes0
      integer ibox,nn,nss,ntt,mc,mnbors

      mc=2**ndim
      mnbors=3**ndim
c
      iptr(1) = 1
      iptr(2) = 2*(nlevels+1)+1
      iptr(3) = iptr(2) + nboxes
      iptr(4) = iptr(3) + nboxes
      iptr(5) = iptr(4) + nboxes
      iptr(6) = iptr(5) + mc*nboxes
      iptr(7) = iptr(6) + nboxes
      iptr(8) = iptr(7) + mnbors*nboxes
c
c     step 1: find enclosing box
c
      call pts_tree_boxsize0(ndim,iper,src,ns,targ,nt,
     1    bs0,cen0)

      boxsize(0) = bs0

      do i=1,ndim
         centers(i,1) = cen0(i)
      enddo

      allocate(isrc(ns),itarg(nt),isrcse(2,nboxes),itargse(2,nboxes))

c
c      set tree info for level 0
c
      itree(1) = 1
      itree(2) = 1
      itree(iptr(2)) = 0
      itree(iptr(3)) = -1
      itree(iptr(4)) = 0
      do i=1,mc
        itree(iptr(5)+i-1) = -1
      enddo

      isrcse(1,1) = 1
      isrcse(2,1) = ns
      itargse(1,1) = 1
      itargse(2,1) = nt

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,ns
        isrc(i) = i
      enddo
C$OMP END PARALLEL DO      

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,nt
        itarg(i) = i
      enddo
C$OMP END PARALLEL DO      


c
c       Reset nlevels, nboxes
c
      nbctr = 1

      do ilev=0,nlevels-1
        irefine = 0

        ifirstbox = itree(2*ilev+1) 
        ilastbox = itree(2*ilev+2)

        nbloc = ilastbox-ifirstbox+1
        allocate(irefinebox(nbloc))

c
c          determine which boxes need to be refined
c
c       
        if(ilev.ge.nlmin) then
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,ibox,nss,ntt,nn)
          do i=1,nbloc
            irefinebox(i) = 0
            ibox = ifirstbox + i-1
            nss = isrcse(2,ibox)-isrcse(1,ibox)+1
            ntt = itargse(2,ibox)-itargse(1,ibox)+1
          
            if(idivflag.eq.0) nn = nss
            if(idivflag.eq.1) nn = ntt
            if(idivflag.eq.2) nn = max(ntt,nss)

            if(nn.gt.ndiv) irefinebox(i) = 1
          enddo
C$OMP END PARALLEL DO        
          irefine = maxval(irefinebox(1:nbloc))

          if(ifunif.eq.1) then
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
            do i=1,nbloc
              irefinebox(i) = irefine
            enddo
C$OMP END PARALLEL DO 
          endif
        else
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
          do i=1,nbloc
            irefinebox(i) = 1
          enddo
C$OMP END PARALLEL DO 
          irefine = 1
        endif
        

        if(irefine.eq.1) then
          boxsize(ilev+1) = boxsize(ilev)/2
          itree(2*ilev+3) = nbctr+1

          call tree_refine_boxes(ndim,irefinebox,nboxes,
     1       ifirstbox,nbloc,centers,boxsize(ilev+1),nbctr,ilev+1,
     2       itree(iptr(2)),itree(iptr(3)),itree(iptr(4)),
     3       itree(iptr(5)))

c
c     re sort points in refined boxes
c
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,ibox)
C$OMP$SCHEDULE(DYNAMIC)
          do i=1,nbloc
            ibox = ifirstbox+i-1
            if(irefinebox(i).eq.1) then
              call sort_pts_to_children(ndim,ibox,nboxes,centers,
     1          itree(iptr(5)),src,ns,isrc,isrcse)
              call sort_pts_to_children(ndim,ibox,nboxes,centers,
     1          itree(iptr(5)),targ,nt,itarg,itargse)
            endif
          enddo
C$OMP END PARALLEL DO          

          
          itree(2*ilev+4) = nbctr
        else
          exit
        endif

        deallocate(irefinebox)
      enddo

      nboxes0 = nbctr
      nlevels = ilev

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
      do i=1,nboxes0
        itree(iptr(6)+i-1) = 0
        do j=1,mnbors
          itree(iptr(7)+mnbors*(i-1)+j-1) = -1
        enddo
      enddo
C$OMP END PARALLEL DO      

      call computecoll(ndim,nlevels,nboxes0,itree(iptr(1)),boxsize,
     1    centers,itree(iptr(3)),itree(iptr(4)),itree(iptr(5)),iper,
     2    itree(iptr(6)),itree(iptr(7)))
        print *, nlevels, nboxes0

      if(nlevels.ge.2.and.ifunif.ne.1) then
         call pts_tree_fix_lr(ndim,centers,nlevels,
     1       nboxes0,boxsize,nboxes,nlevels,iper,itree(iptr(1)),
     2       itree(iptr(2)),itree(iptr(3)),itree(iptr(4)),
     3       itree(iptr(5)),itree(iptr(6)),itree(iptr(7)))
        print *, nlevels, nboxes0

      endif

      call pts_tree_refine_once(ndim,centers,nlevels,
     1    nboxes0,boxsize,nboxes,nlmax,iper,itree(iptr(1)),
     2    itree(iptr(2)),itree(iptr(3)),itree(iptr(4)),
     3    itree(iptr(5)),itree(iptr(6)),itree(iptr(7)),
     4    isrcse,itargse)
      print *, nlevels, nboxes0

      
      return
      end
c

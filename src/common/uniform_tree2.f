c
c     uniform tree in n dimensions
c     created by Shidong Jiang on 03/19/2024
c
c   This code has the following user callable routines
c
c      uniform_tree_mem -> returns memory requirements for creating
c         a tree based on the number of levels
c         in a box (tree length
c         number of boxes, number of levels)
c      uniform_tree_build -> Make the actual tree, returns centers of boxes,
c        colleague info
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


      subroutine uniform_tree_mem(ndim,nlevels,nboxes,ltree)
c
c
c
c----------------------------------------
c  get memory requirements for the tree
c
c
c  input parameters:
c    - ndim: dimension of the underlying space
c    - nlevels: integer
c        number of levels
c        
c  output parameters
c    - nboxes: integer
c        number of boxes
c    - ltree: integer
c        length of tree
c----------------------------------
c
      implicit none
      integer ndim,nlevels,nboxes
      integer ltree


      integer mc,mnbors

      mc = 2**ndim
      mnbors = 3**ndim

      nboxes = (mc**(nlevels+1)-1)/(mc-1)
      ltree = (4+mc+mnbors)*nboxes + 2*(nlevels+1)

      return
      end
c
c
c
c
c
      subroutine uniform_tree_build(ndim,bs0,cen0,iperiod,nlevels,
     1    nboxes,ltree,itree,iptr,centers,boxsize)
c
c
c
c----------------------------------------
c  build tree
c
c
c input parameters:
c    - ndim: dimension of the underlying space
c     - bs0 : real
c        side length of the bounding box
c     - cen0(ndim) : center of the bounding box
c    - nlevels: integer
c        number of levels
c    - nboxes: integer
c        number of boxes
c    - ltree: integer
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
      integer ndim,nlevels,nboxes
      integer iptr(8),ltree
      integer itree(ltree),iperiod
      double precision centers(ndim,nboxes)
      real *8 bs0,cen0(ndim)
      integer, allocatable :: irefinebox(:)
      double precision boxsize(0:nlevels)

      integer i,ilev,irefine
      integer ifirstbox,ilastbox,nbctr,nbloc,nleaf

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

      boxsize(0) = bs0

      do i=1,ndim
         centers(i,1) = cen0(i)
      enddo
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

      nleaf = mc**nlevels
      allocate(irefinebox(nleaf))
      do i=1,nleaf
         irefinebox(i)=1
      enddo
      
c
c       Reset nlevels, nboxes
c
      nbctr = 1

      do ilev=0,nlevels-1
        ifirstbox = itree(2*ilev+1) 
        ilastbox = itree(2*ilev+2)

        nbloc = ilastbox-ifirstbox+1

        boxsize(ilev+1) = boxsize(ilev)/2
        itree(2*ilev+3) = nbctr+1

        call tree_refine_boxes(ndim,irefinebox,nboxes,
     1      ifirstbox,nbloc,centers,boxsize(ilev+1),nbctr,ilev+1,
     2      itree(iptr(2)),itree(iptr(3)),itree(iptr(4)),
     3      itree(iptr(5)))
          
        itree(2*ilev+4) = nbctr
      enddo

      do i=1,nboxes
        itree(iptr(6)+i-1) = 0
        do j=1,mnbors
          itree(iptr(7)+mnbors*(i-1)+j-1) = -1
        enddo
      enddo

      call computecoll(ndim,nlevels,nboxes,itree(iptr(1)),boxsize,
     1    centers,itree(iptr(3)),itree(iptr(4)),itree(iptr(5)),iperiod,
     2    itree(iptr(6)),itree(iptr(7)))

      return
      end
c      
c
c
c
c----------------------------------------------------------------
c      
      subroutine uniform_dmk_find_all_pwexp_boxes(ndim,nboxes,
     1    nlevels,ltree,itree,iptr,
     2    nboxsrcpts,nboxtargpts,
     3    ifpwexpform,ifpwexpeval,iftensprodeval)
c
c
c     Determine whether a box needs plane wave expansions
c     in the point dmk for uniform tree.
c
c     At the cutoff level, i.e., npwlevel, a box needs
c     plane wave expansion if it's nonempty box and it has a colleague
c     with more than ndiv source points
c                  
c     Thus, a leaf box at the cutoff level may or may not 
c     have plane wave expansion.
c     But if it has plane wave expansion, then the self interaction
c     is handled by the plane wave expansion instead of 
c     direct evaluation. 
c      
c      
c     INPUT arguments
c     ndim        in: integer
c                 dimension of the space
c
c     npwlevel    in: integer
c                 Cutoff level
c
c     nboxes      in: integer
c                 Total number of boxes
c
c     nlevels     in: integer
c                 Number of levels
c
c     itree       in: integer(ltree)
c                   array containing tree info - see start of file
c                   for documentation
c     ltree       in: integer
c                   length of itree array
c 
c     iptr        in: integer(8)
c                   pointer for various arrays in itree
c
c--------------------------------------------------------------
c     OUTPUT arguments:
c     ifpwexp     out: integer(nboxes)
c                 ifpwexp(ibox)=1, ibox needs plane wave expansion
c                 ifpwexp(ibox)=0, ibox does not need pwexp
c      
c---------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer nlevels,npwlevel,nboxes,ndim
      integer iptr(8),ltree
      integer nboxsrcpts(nboxes),nboxtargpts(nboxes)
      integer itree(ltree)
      integer ifpwexpform(nboxes),ifpwexpeval(nboxes)
      integer iftensprodeval(nboxes)

      mnbors=3**ndim
      mc=2**ndim
      
      do i=1,nboxes
         ifpwexpform(i)=0
      enddo
      
      do i=1,nboxes
         ifpwexpeval(i)=0
      enddo

      do i=1,nboxes
         iftensprodeval(i)=0
      enddo

      ifpwexpform(1)=1
      ifpwexpeval(1)=1
      
      do ilev=0,nlevels
         do ibox=itree(2*ilev+1),itree(2*ilev+2)
            if (nboxsrcpts(ibox).gt.0) ifpwexpform(ibox)=1
         enddo
      enddo

      do ilev=0,nlevels
         do ibox=itree(2*ilev+1),itree(2*ilev+2)         
            ncoll = itree(iptr(6)+ibox-1)
            npts=nboxsrcpts(ibox)+nboxtargpts(ibox)
            do j=1,ncoll
               jbox = itree(iptr(7) + (ibox-1)*mnbors+j-1)
c               if (ifpwexpform(jbox).eq.1 .and. npts.gt.0) then
               if (ifpwexpform(jbox).eq.1) then
                  ifpwexpeval(ibox)=1
                  goto 1000
               endif
            enddo
               
 1000       continue
         enddo
      enddo

      do ilev=0,nlevels
         do ibox=itree(2*ilev+1),itree(2*ilev+2)         
            if (ifpwexpeval(ibox).eq.1) then
               nchild = itree(iptr(4)+ibox-1)

               iftpeval=1
               do j=1,nchild
                  jbox = itree(iptr(5) + (ibox-1)*mc+j-1)
               
                  if (ifpwexpeval(jbox).eq.1) then
                     iftpeval=0
                     goto 2000
                  endif
               enddo

               if (iftpeval.eq.1) then
                  iftensprodeval(ibox)=1
               endif

 2000          continue

               if (iftensprodeval(ibox).eq.0) then
                  do j=1,nchild
                     jbox = itree(iptr(5) + (ibox-1)*mc+j-1)
               
                     if (ifpwexpeval(jbox).eq.0) then
                        iftensprodeval(jbox)=1
                     endif
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
c      
c
c
c
      subroutine compute_uniform_tree_leaf_box_id(ndim,nlevels,nboxes,
     1    ltree,itree,iptr,centers,boxsize,ibinid,iboxid)
c
c     construct a map to find the leaf box id on a uniform tree
c     given the integral coordinates of its center in R^d
c----------------------------------------
c     input parameters:
c    - ndim: dimension of the underlying space
c    - nlevels: integer
c        number of levels
c    - nboxes: integer
c        number of boxes
c    - ltree: integer
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
c     output parameters:
c     ibinid: the leaf box id given the integral coordinates
c     of its center in the lattice
c     iboxid: label the leaf boxes in natural order 1, 2, 3, etc.
c
      implicit none
      integer ndim,nlevels,nboxes
      integer iptr(8),ltree
      integer itree(ltree),iper
      double precision centers(ndim,nboxes)
      double precision boxsize(0:nlevels)
      real *8 bs0,bsinv,cen0(ndim),ref(ndim)

      integer ibinid(*),iboxid(*)
      
      integer i,ilev,ind,id,ixyz(ndim)

      integer ibox,mc,mnbors,ml(0:ndim),ml0

      mc=2**ndim
      mnbors=3**ndim

      ml0 = 2**nlevels
      ml(0)=1
      do i=1,ndim
         ml(i) = ml(i-1)*ml0
      enddo
      
      bs0 = boxsize(0)
      do i=1,ndim
         cen0(i) = centers(i,1)
      enddo

      do i=1,ndim
         ref(i) = cen0(i)-0.5d0*bs0
      enddo

      bsinv = 1.0d0/boxsize(nlevels)
      do i=1,ndim
         ref(i) = -ref(i)*bsinv
      enddo

      ilev = nlevels
      id = 0
      do ibox=itree(2*ilev+1),itree(2*ilev+2)
         do i=1,ndim
            ixyz(i) = centers(i,ibox)*bsinv+ref(i)
         enddo
         ind = 1
         do i=1,ndim
            ind = ind+ixyz(i)*ml(i-1)
         enddo
         ibinid(ind) = ibox

         id=id+1
         iboxid(ibox) = id
      enddo

      return
      end
c
c
c
c
      subroutine sort_pts_to_uniform_tree(ndim,
     1    nlevels,nboxes,ltree,itree,iptr,centers,boxsize,
     2    ibinid,iboxid,prelist,ns,sources,isrc,isrcinv,isrcse)
c
c     bin sort points to a uniform tree in R^d
c----------------------------------------
c     input parameters:
c    - ndim: dimension of the underlying space
c    - nlevels: integer
c        number of levels
c    - nboxes: integer
c        number of boxes
c    - ltree: integer
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
c    - prelist: list of boxes in the order of preorder traversal
c      
c     output parameters:
c     isrc : 
c     isrcse : 
c
      implicit none
      integer ndim,nlevels,nboxes,ns
      integer iptr(8),ltree
      integer itree(ltree),iper
      double precision centers(ndim,nboxes)
      double precision boxsize(0:nlevels)
      real *8 sources(ndim,ns)
      real *8 bs0,cen0(ndim),ref(ndim)

      integer ibinid(*),iboxid(*),prelist(*)
      integer isrc(ns),isrcinv(ns),isrcse(2,nboxes)
      
      integer i,ilev,ind,ibin,j

      integer ibox,jbox,nchild
      integer mc,mnbors,boxid,ml(0:ndim),ixyz(ndim)
      integer, allocatable :: nsrc(:),isrcbox(:),ip(:)

      integer id,ml0,mlm1,istart,iend,bin,jstart,jend
      real *8 bsinv
      
      do ibox=1,nboxes
         isrcse(1,ibox)=1
         isrcse(2,ibox)=0
      enddo
      
c     number of sources in each box
      allocate(nsrc(nboxes))
c     box id for each source
      allocate(isrcbox(ns))

      allocate(ip(nboxes+1))
      
      do i=1,nboxes
         nsrc(i)=0
      enddo
      
      mc=2**ndim
      mnbors=3**ndim
      ml0 = 2**nlevels
      ml(0)=1
      do i=1,ndim
         ml(i) = ml(i-1)*ml0
      enddo
      
      mlm1 = ml0-1
      
      bs0 = boxsize(0)
      do i=1,ndim
         cen0(i) = centers(i,1)
      enddo

      do i=1,ndim
         ref(i) = cen0(i)-0.5d0*bs0
      enddo

      bsinv = 1.0d0/boxsize(nlevels)
      do i=1,ndim
         ref(i) = -ref(i)*bsinv
      enddo

c     first, find which leaf box each source point lies in
c     and the number of source points in each leaf box
      do i=1,ns
         do j=1,ndim
            ixyz(j) = sources(j,i)*bsinv+ref(j)
c     commented out for efficiency. One can alway avoid the boundary points
c     by enlarging the root box slightly.
            ixyz(j) = max(ixyz(j),0)
            ixyz(j) = min(ixyz(j),mlm1)
         enddo
         ind = 1
         do j=1,ndim
            ind = ind + ixyz(j)*ml(j-1)
         enddo
         ibox = ibinid(ind)
         isrcbox(i) = ibox
         nsrc(ibox) = nsrc(ibox) + 1
      enddo

c     second, post-order traversal on the boxes and put source points
c     to leaf boxes, then to nonleaf boxes
      ind=1
      ip(1)=0
      do i=nboxes,1,-1
         ibox=prelist(i)
         nchild=itree(iptr(4)+ibox-1)
         if (nchild.eq.0) then
            ind=ind+1
            ip(ind)=ip(ind-1)+nsrc(ibox)
            isrcse(1,ibox) = ip(ind-1)+1
            isrcse(2,ibox) = ip(ind)
         else
            j=1
            jbox=itree(iptr(5)+mc*(ibox-1)+j-1)
            istart=isrcse(1,jbox)
            iend=isrcse(2,jbox)
            do j=2,nchild
               jbox=itree(iptr(5)+mc*(ibox-1)+j-1)
               jstart=isrcse(1,jbox)
               jend=isrcse(2,jbox)
               if (jstart.lt.istart) istart=jstart
               if (jend.gt.iend) iend=jend
            enddo
            isrcse(1,ibox)=istart
            isrcse(2,ibox)=iend
         endif
      enddo

      do i=1,ns
         ibox=isrcbox(i)
         bin = iboxid(ibox)
         ip(bin)=ip(bin)+1
         isrc(ip(bin))=i
         isrcinv(i)=ip(bin)
      enddo

      return
      end
c
c
c
c
      subroutine preordertraversal(ndim,nlevels,nboxes,
     1    ltree,itree,iptr,centers,boxsize,prelist)
c
c     construct a map to find the leaf box id on a uniform tree
c     given the integral coordinates of its center in R^d
c----------------------------------------
c     input parameters:
c    - ndim: dimension of the underlying space
c    - nlevels: integer
c        number of levels
c    - nboxes: integer
c        number of boxes
c    - ltree: integer
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
c     output parameters:
c     prelist: list of all boxes in the tree in the order of 
c     preorder traversal
c
      implicit none
      integer ndim,nlevels,nboxes
      integer iptr(8),ltree
      integer itree(ltree),iper
      double precision centers(ndim,nboxes)
      double precision boxsize(0:nlevels)
      real *8 bs0,bsinv,cen0(ndim),ref(ndim)

      integer prelist(nboxes)
      integer, allocatable :: boxstack(:)
      integer i,ilev,ind,id,ixyz(ndim)

      integer ibox,mc,mnbors,ns,jbox,nchild

      allocate(boxstack(nboxes))
      
      mc=2**ndim
      mnbors=3**ndim

c     push the root box to the stack
      boxstack(1)=1
      ns = 1

      ind = 0
      do while (ns.gt.0)
         ind = ind+1
         prelist(ind) = boxstack(ns)
         ibox = boxstack(ns)
         ns = ns-1
         nchild = itree(iptr(4)+ibox-1)
         if (nchild.gt.0) then
            do i=1,nchild
               jbox = itree(iptr(5)+(ibox-1)*mc+i-1)
               ns = ns+1
               boxstack(ns) = jbox
            enddo
         endif
      enddo

      return
      end
            

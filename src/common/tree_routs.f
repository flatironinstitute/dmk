c
c
c     common routines for generating and processing
c     a level restricted quad tree in ndim=1,2,3 dimensions
c   
c     last modified by Shidong Jiang on 02/06/2024
c
c
      subroutine tree_refine_boxes(ndim,irefinebox,nboxes,
     1  ifirstbox,nbloc,centers,bs,nbctr,nlctr,
     2  ilevel,iparent,nchild,ichild)
      implicit none
      integer ndim,nboxes,nbloc,nbctr,nlctr
      real *8 centers(ndim,nboxes),bs
      integer ilevel(nboxes),iparent(nboxes)
      integer ichild(2**ndim,nboxes),nchild(nboxes)
      integer irefinebox(nbloc)
      integer ifirstbox
      integer, allocatable :: isum(:)

      integer i,ibox,j,l,jbox,nbl,k,mc
      real *8 bsh
      integer isgn(ndim,2**ndim)

      call get_child_box_sign(ndim,isgn)

      mc=2**ndim
      bsh = bs/2

      allocate(isum(nbloc))
      if(nbloc.gt.0) call cumsum(nbloc,irefinebox,isum)
      
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,ibox,nbl,j,jbox)
      do i = 1,nbloc
        ibox = ifirstbox + i-1
        if(irefinebox(i).eq.1) then
          nbl = nbctr + (isum(i)-1)*mc
          
          nchild(ibox) = mc
          do j=1,mc
            jbox = nbl+j
            do k=1,ndim
               centers(k,jbox) = centers(k,ibox)+isgn(k,j)*bsh
            enddo
            iparent(jbox) = ibox
            nchild(jbox) = 0
            do l=1,mc
              ichild(l,jbox) = -1
            enddo
            ichild(j,ibox) = jbox
            ilevel(jbox) = nlctr 
          enddo
        endif
      enddo
C$OMP END PARALLEL DO      

      if(nbloc.gt.0) nbctr = nbctr + isum(nbloc)*mc


      return
      end
c
c
c
c
c
c
       subroutine tree_copy(ndim,nb,centers,ilevel,iparent,nchild,
     1              ichild,centers2,ilevel2,iparent2,nchild2,ichild2)

       implicit none
       integer ndim,nb
       real *8 centers(ndim,nb),centers2(ndim,nb)
       integer ilevel(nb),ilevel2(nb)
       integer iparent(nb),iparent2(nb)
       integer nchild(nb),nchild2(nb)
       integer ichild(2**ndim,nb),ichild2(2**ndim,nb)

       integer i,j,mc

       mc=2**ndim

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
       do i=1,nb
         do j=1,ndim
            centers2(j,i) = centers(j,i)
         enddo
         ilevel2(i) = ilevel(i)
         iparent2(i) = iparent(i)
         nchild2(i) = nchild(i)
         do j=1,mc
            ichild2(j,i) = ichild(j,i)
         enddo
       enddo
C$OMP END PARALLEL DO       
       

       return
       end
c
c
c
c
c
      subroutine computecoll(ndim,nlevels,nboxes,laddr,boxsize,
     1    centers,iparent,nchild,ichild,iperiod,
     2    nnbors,nbors)

c     This subroutine computes the colleagues for an adaptive
c     pruned tree. box j is a colleague of box i, if they share a
c     vertex or an edge and the two boxes are at the same
c     level in the tree
c
c     INPUT arguments
c     ndim        in: integer
c                 dimension of the space
c
c     nlevels     in: integer
c                 Number of levels
c
c     nboxes      in: integer
c                 Total number of boxes
c
c     laddr       in: integer(2,0:nlevels)
c                 indexing array providing access to boxes at
c                 each level. 
c                 the first box on level i is laddr(1,i)
c                 the last box on level i is laddr(2,i)
c
c     boxsize     in: double precision(0:nlevels)
c                 Array of boxsizes
c 
c     centers     in: double precision(2,nboxes)
c                 array of centers of boxes
c   
c     iparent     in: integer(nboxes)
c                 iparent(i) is the box number of the parent of
c                 box i
c
c     nchild      in: integer(nboxes)
c                 nchild(i) is the number of children of box i
c
c     ichild      in: integer(4,nboxes)
c                 ichild(j,i) is the box id of the jth child of
c                 box i
c
c     iperiod     in: integer
c                 0: free space; 1: doubly periodic
c
c----------------------------------------------------------------
c     OUTPUT
c     nnbors      out: integer(nboxes)
c                 nnbors(i) is the number of colleague boxes of
c                 box i
c
c     nbors       out: integer(3**ndim,nboxes)
c                 nbors(j,i) is the box id of the jth colleague
c                 box of box i
c---------------------------------------------------------------
      implicit none
      integer ndim,nlevels,nboxes
      integer iperiod
      integer laddr(2,0:nlevels)
      double precision boxsize(0:nlevels)
      double precision centers(ndim,nboxes)
      integer iparent(nboxes), nchild(nboxes), ichild(2**ndim,nboxes)
      integer nnbors(nboxes)
      integer nbors(3**ndim,nboxes)

c     Temp variables
      integer ilev,ibox,jbox,kbox,dad
      integer i,j,ifirstbox,ilastbox,mc,mnbors,k,ifnbor
      real *8 bs0,dis,dp1

      bs0=boxsize(0)
      
      mc= 2**ndim
      mnbors=3**ndim
      
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
      do i=1,nboxes
         nnbors(i) = 0
         do j=1,mnbors
            nbors(j,i) = -1
         enddo
      enddo
C$OMP END PARALLEL DO     
      
c     Setting parameters for level = 0
      nnbors(1) = 1
      nbors(1,1) = 1
      do ilev = 1,nlevels
c        Find the first and the last box at level ilev      
         ifirstbox = laddr(1,ilev)
         ilastbox = laddr(2,ilev)
c        Loop over all boxes to evaluate neighbors, list1 and updating
c        hunglists of targets

C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,dad,i,jbox,j,kbox,ifnbor,k,dp1,dis)
         do ibox = ifirstbox,ilastbox
c           Find the parent of the current box         
            dad = iparent(ibox)
c           Loop over the neighbors of the parent box
c           to find out list 1 and list 2
            do i=1,nnbors(dad)
                jbox = nbors(i,dad)
                do j=1,mc
c               ichild(j,jbox) is one of the children of the
c               neighbors of the parent of the current
c               box
                   kbox = ichild(j,jbox)
                   if(kbox.gt.0) then
c     Check if kbox is a nearest neighbor or in list 2
                      ifnbor=1
                      do k=1,ndim
                         dis=abs(centers(k,kbox)-centers(k,ibox))
                         if (iperiod.eq.1) then
                            dp1=bs0-dis
                            if (dp1.lt.dis) dis=dp1
                         endif
                         if (dis.gt.1.05*boxsize(ilev)) then
                            ifnbor=0
                            exit
                         endif
                      enddo
                         
                      if(ifnbor.eq.1) then
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

      return
      end
c
c
c
c
c--------------------------------------------------------------------      
      subroutine updateflags(ndim,curlev,nboxes,nlevels,laddr,
     1    nchild,ichild,nnbors,nbors,centers,boxsize,iflag)

c      This subroutine is to check the boxes flagged as flag++
c      and determine which of the boxes need refinement. The flag
c      of the box which need refinement is updated to iflag(box)=1
c      and that of the boxes which do not need refinement is
c      updated to iflag(box) = 0
c
c      INPUT arguments
c      curlev         in: integer
c                     the level for which boxes need to be processed
c
c      nboxes         in: integer
c                     total number of boxes
c
c      nlevels        in: integer
c                     total number of levels
c
c      laddr          in: integer(2,0:nlevels)
c                     boxes from laddr(1,ilev) to laddr(2,ilev)
c                     are at level ilev
c
c      nchild         in: integer(nboxes)
c                     nchild(ibox) is the number of children
c                     of box ibox
c
c      ichild         in: integer(4,nboxes)
c                     ichild(j,ibox) is the box id of the jth
c                     child of box ibox
c
c      nnbors         in: integer(nboxes)
c                     nnbors(ibox) is the number of colleagues
c                     of box ibox
c
c      nbors          in: integer(9,nboxes)
c                     nbors(j,ibox) is the jth colleague of box
c                     ibox
c
c      centers        in: double precision(2,nboxes)
c                     x and y coordinates of the box centers
c
c      boxsize        in: double precision(0:nlevels)
c                     boxsize(i) is the size of the box at level i
c
c      iflag          in/out: integer(nboxes)
c                     iflag(ibox)=3 if it is flag++. iflag(ibox) =1
c                     or 0 at the end of routine depending on
c                     whether box needs to be subdivided or not
c
      implicit none
c     Calling sequence variables
      integer ndim,curlev, nboxes, nlevels
      integer laddr(2,0:nlevels),nchild(nboxes),ichild(2**ndim,nboxes)
      integer nnbors(nboxes), nbors(3**ndim,nboxes)
      integer iflag(nboxes)
      double precision centers(ndim,nboxes),boxsize(0:nlevels)

c     Temporary variables
      integer i,j,k,ibox,jbox,kbox,ict, mc
      double precision distest,dis

      mc=2**ndim
      distest = 1.05d0*(boxsize(curlev) + boxsize(curlev+1))/2.0d0
c     Loop over all boxes at the current level     

C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ibox,i,jbox,j,kbox,dis,ict)
      do ibox = laddr(1,curlev),laddr(2,curlev)
         if(iflag(ibox).eq.3) then
            iflag(ibox) = 0
c           Loop over colleagues of the current box      
            do i=1,nnbors(ibox)
c              Loop over colleagues of flag++ box        
               jbox = nbors(i,ibox)
              
c              Loop over the children of the colleague box
c              Note we do not need to exclude self from
c              the list of colleagues as a self box which
c              is flag++ does not have any children 
c              and will not enter the next loop
               do j=1,mc
                  kbox = ichild(j,jbox)
                  if(kbox.gt.0) then
                     if(nchild(kbox).gt.0) then
                        ict = 0
                        do k=1,ndim
                           dis = centers(k,kbox) - centers(k,ibox) 
                           if(abs(dis).le.distest) ict = ict + 1
                        enddo
                        if(ict.eq.ndim) then
                           iflag(ibox) = 1
                           goto 1111
                        endif
                     endif
                  endif
c                 End of looping over the children of the child
c                 of the colleague box
               enddo
c              End of looping over the children of the colleague box       
            enddo
c           End of looping over colleagues            
 1111       continue        
         endif
c        End of testing if the current box needs to checked for         
      enddo
c     End of looping over boxes at the current level      
C$OMP END PARALLEL DO      

      return
      end
c
c
c
c
c
c

      subroutine tree_refine_boxes_flag(ndim,iflag,nboxes,
     1  ifirstbox,nbloc,centers,bs,nbctr,nlctr,
     2  ilevel,iparent,nchild,ichild)
      implicit none
      integer ndim,nboxes,nbloc,nbctr,nlctr
      real *8 centers(ndim,nboxes),bs,bsh
      integer ilevel(nboxes),iparent(nboxes)
      integer ichild(2**ndim,nboxes),nchild(nboxes)
      integer iflag(nboxes)
      integer ifirstbox
      integer, allocatable :: isum(:),itmp(:)

      integer i,ibox,j,l,jbox,nbl,k
      integer mc
      integer isgn(ndim,2**ndim)

      call get_child_box_sign(ndim,isgn)
      
      bsh = bs/2
      mc = 2**ndim

      allocate(isum(nbloc),itmp(nbloc))
      do i=1,nbloc
        ibox = ifirstbox+i-1
        itmp(i) = 0
        if(iflag(ibox).gt.0) itmp(i) = 1
      enddo
      if(nbloc.gt.0) call cumsum(nbloc,itmp,isum)
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,ibox,nbl,j,jbox)
      do i = 1,nbloc
        ibox = ifirstbox + i-1
        if(iflag(ibox).gt.0) then
          nbl = nbctr + (isum(i)-1)*mc
          
          nchild(ibox) = mc
          do j=1,mc
            jbox = nbl+j
            do k=1,ndim
               centers(k,jbox) = centers(k,ibox)+isgn(k,j)*bsh
            enddo
            iparent(jbox) = ibox
            nchild(jbox) = 0
            do l=1,mc
              ichild(l,jbox) = -1
            enddo
            ichild(j,ibox) = jbox
            ilevel(jbox) = nlctr+1 
            if(iflag(ibox).eq.1) iflag(jbox) = 3
            if(iflag(ibox).eq.2) iflag(jbox) = 0
          enddo
        endif
      enddo
C$OMP END PARALLEL DO      

      if(nbloc.gt.0) nbctr = nbctr + isum(nbloc)*mc

      return
      end
c
c
c
c
c
c
      subroutine compute_mnlist1(ndim,nboxes,nlevels,laddr,
     1    centers,boxsize,iparent,nchild,
     2    ichild,isep,nnbors,nbors,iper,mnlist1)
c     Compute max number of boxes in list1
      implicit none
      integer ndim,nlevels,nboxes
      integer iper
      integer laddr(2,0:nlevels)
      double precision boxsize(0:nlevels)
      double precision centers(ndim,nboxes)
      integer iparent(nboxes),nchild(nboxes),ichild(2**ndim,nboxes)
      integer mnbors,isep
      integer nnbors(nboxes),nbors(3**ndim,nboxes)
      integer mnlist1
      integer nlist1(nboxes)

c     Temp variables
      integer ilev,ibox,jbox,kbox,i,j,k
      integer firstbox,lastbox,dad,mc,ifnbor,iflist1
      double precision dis,distest,bs0,dp1

      bs0=boxsize(0)

      mnbors=3**ndim
      mc=2**ndim
      
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,nboxes
         nlist1(i) = 0
      enddo
C$OMP END PARALLEL DO
      
      nlist1(1) = 1

      do ilev = 1,nlevels
         firstbox = laddr(1,ilev)
         lastbox = laddr(2,ilev)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,dad,i,jbox,j,kbox,distest,dis,iflist1,ifnbor)
         do ibox = firstbox,lastbox
            dad = iparent(ibox)
c           Compute list1 and list3 of ibox if it is childless
            if(nchild(ibox).eq.0) then
               do i=1,nnbors(ibox)
                  jbox = nbors(i,ibox)

c
cc                     check for list1 at the same level
c
                  if(nchild(jbox).eq.0) then
                     nlist1(ibox) = nlist1(ibox) + 1
                  endif
c
cc                     check for list1 and list3 at one ilev+1
                  if(nchild(jbox).gt.0) then
                    distest = 1.05d0*(boxsize(ilev)+boxsize(ilev+1))/
     1                   2.0d0*isep
                    do j=1,mc
                      kbox = ichild(j,jbox)
                      if(kbox.gt.0) then
                        ifnbor=1
                        do k=1,ndim
                          dis = dabs(centers(k,kbox)-centers(k,ibox))
                          if (iper .eq. 1) then
                             dp1 = bs0-dis
                             if (dp1.lt.dis) dis=dp1
                          endif
                           
                          if (dis.gt.distest) then
                            ifnbor=0
                            exit
                          endif
                        enddo
                        if(ifnbor.eq.1) then
                           nlist1(ibox) = nlist1(ibox)+1
                        endif
                      endif
                    enddo
                  endif
               enddo
c
cc               compute list1 and list4 for boxes at level ilev-1 
               do i=1,nnbors(dad)
                   jbox = nbors(i,dad)
                   if(nchild(jbox).eq.0) then
                      distest = 1.05d0*(boxsize(ilev)+boxsize(ilev-1))/
     1                    2.0d0*isep
                      iflist1=1
                      do k=1,ndim
                         dis = dabs(centers(k,jbox)-centers(k,ibox))
                         if (iper .eq. 1) then
                            dp1 = bs0-dis
                            if (dp1.lt.dis) dis=dp1
                         endif
                         
                         if (dis.ge.distest) then
                            iflist1=0
                            exit
                         endif
                      enddo
                      if(iflist1.eq.1) then
                         nlist1(ibox) = nlist1(ibox)+1
                      endif
                   endif
               enddo
            endif
         enddo
C$OMP END PARALLEL DO         
      enddo

      mnlist1 = 0
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i) 
C$OMP$REDUCTION(max:mnlist1)      
      do i=1,nboxes
         if(nlist1(i).gt.mnlist1) mnlist1 = nlist1(i)
      enddo
C$OMP END PARALLEL DO      

      return
      end
c      
c      
c      
c      
      subroutine flag_box_refine(ndim,nlevels,nboxes,laddr,boxsize,
     1                   centers,iparent,nchild,
     2                   ichild,isep,nnbors,mnbors,nbors,iper,
     3                   ifrefine,nrefine)
c     flag boxes that may require refinement for the box FGT
      implicit none
      integer ndim,nlevels,nboxes
      integer iper
      integer laddr(2,0:nlevels)
      double precision boxsize(0:nlevels)
      double precision centers(ndim,nboxes)
      integer iparent(nboxes),nchild(nboxes),ichild(2**ndim,nboxes)
      integer mnbors
      integer nnbors(nboxes),nbors(mnbors,nboxes)
      integer isep
      integer ifrefine(nboxes),nrefine

c     Temp variables
      integer ilev,ibox,jbox,kbox,i,j,k,mc
      integer firstbox,lastbox,iflist1
      double precision dis,distest

      mc=2**ndim
      
C$OMP PARALLEL DO DEFAULT(SHARED)
      do i=1,nboxes
         ifrefine(i)=0
      enddo
C$OMP END PARALLEL DO      

      nrefine=0
      do ilev = 1,nlevels-1
         firstbox = laddr(1,ilev)
         lastbox = laddr(2,ilev)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,i,jbox,j,kbox,dis,distest,k,iflist1)
         do 1100 ibox = firstbox,lastbox
c           Compute list1 and list3 of ibox if it is childless
            if(nchild(ibox).eq.0) then
               do i=1,nnbors(ibox)
                  jbox = nbors(i,ibox)
                  if(nchild(jbox).ne.0) then
c     
cc                     boxes in list1 and list3 at level ilev+1
c
                     distest = 1.05d0*(boxsize(ilev)+boxsize(ilev+1))/
     1                   2.0d0*isep
                     do j=1,mc
                        kbox = ichild(j,jbox)
                        if(kbox.gt.0) then
                          iflist1=1
                          do k=1,ndim
                            dis = dabs(centers(k,kbox)-centers(k,ibox))
                            if (dis.ge.distest) then
                               iflist1=0
                               exit
                            endif
                          enddo
                          if(iflist1.eq.1) then
                             ifrefine(ibox)=1
                             nrefine=nrefine+1
                             goto 1100
                          endif
                       endif
                     enddo
                  endif
               enddo
            endif
 1100    continue
C$OMP END PARALLEL DO         
      enddo

      return
      end
c      
c      
c      
c      
c----------------------------------------------------------------
c      
      subroutine bdmk_compute_modified_list1(ndim,npwlevel,ifpwexp,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,iperiod,
     2    mnlist1,nlist1,list1)
c
c
c     This subroutine computes the modified list1 of a given tree
c     structure for the box DMK code
c
c     Note: new definition of list 1 - for ilev <= npwlevel,             
c     list1 of a leaf box contains all childless neighbors at or
c     above npwlevel. ifpwexp(ibox)=1, then ibox is excluded from 
c     list1 since then the self interaction is handled via
c     plane wave expansions.
c                  
c     Assume that the tree is level-restricted. Then the
c     new list1 of source ibox contains all target boxes 
c     that require the evaluation of direct interactions with ibox 
c     in the box FGT.  
c      
c      
c     haven't tested iper=1 case.
c      
c      
c     INPUT arguments
c     nlevels     in: integer
c                 Number of levels
c
c     npwlevel    in: integer
c                 Cutoff level
c     ifpwexp     in: integer(nboxes)
c                 1: requires pwexp; 0: does not require pwexp
c      
c     nboxes      in: integer
c                 Total number of boxes
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
c     centers     in: real *8(2,nboxes)
c                 xy coordinates of centers of boxes
c   
c     boxsize     in: real *8(0:nlevels)
c                 Array of boxsizes
c   
c     iperiod     in: integer
c                 flag for periodic implementations. Currently not used.
c                 Feature under construction
c 
c     mnlist1     in: integer
c                 max number of boxes in list 1 of a box
c
c--------------------------------------------------------------
c     OUTPUT arguments:
c     nlist1      out: integer(nboxes)
c                 nlist1(i) is the number of boxes in list 1 
c                 of box i
c
c     list1       out: integer(mnlist1,nboxes)
c                 list1(j,i) is the box id of the jth box in 
c                 list1 of box i.
c                  
c      
c---------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer nlevels,npwlevel,nboxes,ndim
      integer iptr(8),ltree
      integer iperiod
      integer itree(ltree),ifpwexp(nboxes)
      real *8 boxsize(0:nlevels)
      real *8 centers(ndim,nboxes)
      integer mnlist1
      integer nlist1(nboxes), list1(mnlist1,nboxes)

c     Temp variables
      integer ilev,ibox,jbox,kbox,dad,mnbors,mc
      integer i,j,ifirstbox,ilastbox,iflist1,k
      real *8 distest,dis,bs0,dp1

      bs0=boxsize(0)

      mnbors=3**ndim
      mc=2**ndim
      
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,nboxes
        nlist1(i) = 0
      enddo
C$OMP END PARALLEL DO      


c     Setting parameters for level = 0
      if(itree(iptr(4)).eq.0) then
         nlist1(1) = 1 
         list1(1,1) = 1
      else
         nlist1(1) = 0
      endif

      nlevend=nlevels
      if (npwlevel.lt.nlevels) nlevend=npwlevel
      do ilev = 1,nlevend
         ifirstbox = itree(2*ilev+1)
         ilastbox = itree(2*ilev+2)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,dad,i,jbox,j,kbox,dis,distest,k)
C$OMP$PRIVATE(iflist1,dp1)
         do ibox = ifirstbox,ilastbox
            dad = itree(iptr(3)+ibox-1)
c           Compute list1 of ibox if it is childless
            if(itree(iptr(4)+ibox-1).eq.0) then
               do i=1,itree(iptr(6)+ibox-1)
                  jbox = itree(iptr(7)+mnbors*(ibox-1)+i-1)
c
cc                boxes in list 1 at the same level
c

                  if (itree(iptr(4)+jbox-1).eq.0) then
                     if ((ifpwexp(ibox).ne.1) .or. (jbox.ne.ibox)) then
                        nlist1(ibox) = nlist1(ibox) + 1
                        list1(nlist1(ibox),ibox) = jbox
                     endif
                  else
c
cc                     boxes in list1 at level ilev+1
c
                     if (ilev.lt.npwlevel) then
                        distest = 1.05d0*(boxsize(ilev)+boxsize(ilev+1))
     1                      /2.0d0
                        do j=1,mc
                           kbox = itree(iptr(5)+mc*(jbox-1)+j-1)
                           if(itree(iptr(4)+kbox-1).eq.0) then
                              iflist1=1
                              do k=1,ndim
                                 dis = dabs(centers(k,kbox)
     1                               -centers(k,ibox))
                                 if (iperiod .eq. 1) then
                                    dp1 = bs0-dis
                                    if (dp1.lt.dis) dis=dp1
                                 endif
                                 
                                 if (dis.ge.distest) then
                                    iflist1=0
                                    exit
                                 endif
                              enddo
                              if(iflist1.eq.1) then
                                 nlist1(ibox) = nlist1(ibox)+1
                                 list1(nlist1(ibox),ibox) = kbox
                              endif
                           endif
                        enddo
                     endif
                  endif
               enddo
c
cc               compute list1 at level ilev-1 
               do i=1,itree(iptr(6)+dad-1)
                  jbox = itree(iptr(7)+mnbors*(dad-1)+i-1)
                  if(itree(iptr(4)+jbox-1).eq.0) then
                     distest = 1.05d0*(boxsize(ilev)+boxsize(ilev-1))/
     1                   2.0d0
                     iflist1=1
                     do k=1,ndim
                        dis = dabs(centers(k,jbox)-centers(k,ibox))
                        if (iperiod .eq. 1) then
                           dp1 = bs0-dis
                           if (dp1.lt.dis) dis=dp1
                        endif
                        if (dis.ge.distest) then
                           iflist1=0
                           exit
                        endif
                     enddo
                     if(iflist1.eq.1) then
                        nlist1(ibox) = nlist1(ibox)+1
                        list1(nlist1(ibox),ibox) = jbox
                     endif
                  endif
               enddo
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
c      
c----------------------------------------------------------------
c      
c      
      subroutine pdmk_compute_modified_list1(ndim,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,iperiod,
     2    ifleafbox,npwlevel,
     3    mnlist1,nlist1,list1)
c
c
c     This subroutine computes the modified list1 of a given tree
c     structure for the point DMK code.
c
c     Note: new definition of list 1 - for ilev <= npwlevel,             
c     list1 of a leaf box contains all childless neighbors at or
c     above npwlevel. ifpwexp(ibox)=1, then ibox is excluded from 
c     list1 since then the self interaction is handled via
c     plane wave expansions.
c                  
c     Assume that the tree is level-restricted. Then the
c     new list1 of source ibox contains all target boxes 
c     that require the evaluation of direct interactions with ibox 
c     in the point DMK.  
c      
c      
c     INPUT arguments
c     ndim        in: integer
c                 dimension of the space
c
c     nlevels     in: integer
c                 Number of levels
c
c     npwlevel    in: integer
c                 Cutoff level
c     ifpwexp     in: integer(nboxes)
c                 1: requires pwexp; 0: does not require pwexp
c      
c     nboxes      in: integer
c                 Total number of boxes
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
c     centers     in: real *8(ndim,nboxes)
c                 xy coordinates of centers of boxes
c   
c     boxsize     in: real *8(0:nlevels)
c                 Array of boxsizes
c   
c     iperiod     in: integer
c                 0: free space; 1: periodic
c 
c     mnlist1     in: integer
c                 max number of boxes in list 1 of a box
c
c--------------------------------------------------------------
c     OUTPUT arguments:
c     nlist1      out: integer(nboxes)
c                 nlist1(i) is the number of boxes in list 1 
c                 of box i
c
c     list1       out: integer(mnlist1,nboxes)
c                 list1(j,i) is the box id of the jth box in 
c                 list1 of box i.
c                  
c      
c---------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer nlevels,npwlevel,nboxes,ndim
      integer iptr(8),ltree
      integer iperiod,itree(ltree)
      integer ifleafbox(nboxes)
      real *8 boxsize(0:nlevels)
      real *8 centers(ndim,nboxes)
      integer mnlist1
      integer nlist1(nboxes), list1(mnlist1,nboxes)

c     Temp variables
      integer ilev,ibox,jbox,dad,mnbors,mc
      integer i,ifirstbox,ilastbox,iflist1,k
      real *8 distest,dis,bs0,dp1

      bs0=boxsize(0)

      mnbors=3**ndim
      mc=2**ndim
      
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,nboxes
        nlist1(i) = 0
      enddo
C$OMP END PARALLEL DO      


c     Setting parameters for level = 0
      if(itree(iptr(4)).eq.0) then
         nlist1(1) = 1 
         list1(1,1) = 1
         return
      else
         nlist1(1) = 0
      endif

      nlevend=nlevels
      if (npwlevel.lt.nlevels) nlevend=npwlevel
      do ilev = npwlevel,npwlevel
         ifirstbox = itree(2*ilev+1)
         ilastbox = itree(2*ilev+2)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,dad,i,jbox,j,kbox,dis,distest,k)
C$OMP$PRIVATE(iflist1,dp1)
         do ibox = ifirstbox,ilastbox
c           ibox is the source box
            dad = itree(iptr(3)+ibox-1)
c     Compute list1 of ibox if it has never been used as a direct
c     interaction source box
            if (ifleafbox(ibox).eq.1) then
               do i=1,itree(iptr(6)+ibox-1)
                  jbox = itree(iptr(7)+mnbors*(ibox-1)+i-1)
c              target boxes in list 1 at the same level
                  nlist1(ibox) = nlist1(ibox) + 1
                  list1(nlist1(ibox),ibox) = jbox
               enddo

               
c     c        compute list1 at level ilev-1
               do i=1,itree(iptr(6)+dad-1)
                  jbox = itree(iptr(7)+mnbors*(dad-1)+i-1)
                  if(ifleafbox(jbox).eq.1) then
                     distest = 1.05d0*(boxsize(ilev)+boxsize(ilev-1))/
     1                   2.0d0
                     iflist1=1
                     do k=1,ndim
                        dis = dabs(centers(k,jbox)-centers(k,ibox))
                        if (iperiod .eq. 1) then
                           dp1 = bs0-dis
                           if (dp1.lt.dis) dis=dp1
                        endif
                        if (dis.ge.distest) then
                           iflist1=0
                           exit
                        endif
                     enddo
                     if(iflist1.eq.1) then
                        nlist1(ibox) = nlist1(ibox)+1
                        list1(nlist1(ibox),ibox) = jbox
                     endif
                  endif
               enddo
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
c      
c      
c----------------------------------------------------------------
c      
c      
      subroutine pdmk_compute_all_modified_list1(ndim,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,iperiod,
     2    mnlist1,nlist1,list1)
c
c
c     This subroutine computes the modified list1 of a given tree
c     structure for the point DMK code.
c
c     Note: new definition of list 1 - for ilev <= npwlevel,             
c     list1 of a leaf box contains all childless neighbors at or
c     above npwlevel. ifpwexp(ibox)=1, then ibox is excluded from 
c     list1 since then the self interaction is handled via
c     plane wave expansions.
c                  
c     Assume that the tree is level-restricted. Then the
c     new list1 of source ibox contains all target boxes 
c     that require the evaluation of direct interactions with ibox 
c     in the point DMK.  
c      
c      
c     INPUT arguments
c     ndim        in: integer
c                 dimension of the space
c
c     nlevels     in: integer
c                 Number of levels
c
c     ifpwexp     in: integer(nboxes)
c                 1: requires pwexp; 0: does not require pwexp
c      
c     nboxes      in: integer
c                 Total number of boxes
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
c     centers     in: real *8(ndim,nboxes)
c                 xy coordinates of centers of boxes
c   
c     boxsize     in: real *8(0:nlevels)
c                 Array of boxsizes
c   
c     iperiod     in: integer
c                 0: free space; 1: periodic
c 
c     mnlist1     in: integer
c                 max number of boxes in list 1 of a box
c
c--------------------------------------------------------------
c     OUTPUT arguments:
c     nlist1      out: integer(nboxes)
c                 nlist1(i) is the number of boxes in list 1 
c                 of box i
c
c     list1       out: integer(mnlist1,nboxes)
c                 list1(j,i) is the box id of the jth box in 
c                 list1 of box i.
c                  
c      
c---------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer nlevels,nboxes,ndim
      integer iptr(8),ltree
      integer iperiod,itree(ltree)
      real *8 boxsize(0:nlevels)
      real *8 centers(ndim,nboxes)
      integer mnlist1
      integer nlist1(nboxes), list1(mnlist1,nboxes)

c     Temp variables
      integer ilev,ibox,jbox,dad,mnbors,mc,ichild,jchild
      integer i,ifirstbox,ilastbox,iflist1,k
      real *8 distest,dis,bs0,dp1

      bs0=boxsize(0)

      mnbors=3**ndim
      mc=2**ndim
      
C$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,nboxes
        nlist1(i) = 0
      enddo
C$OMP END PARALLEL DO      


c     Setting parameters for level = 0
      if(itree(iptr(4)).eq.0) then
         nlist1(1) = 1 
         list1(1,1) = 1
         return
      else
         nlist1(1) = 0
      endif

      do ilev = 0,nlevels
         ifirstbox = itree(2*ilev+1)
         ilastbox = itree(2*ilev+2)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,dad,i,jbox,j,kbox,dis,distest,k)
C$OMP$PRIVATE(iflist1,dp1)
         do ibox = ifirstbox,ilastbox
c           ibox is the source box
            dad = itree(iptr(3)+ibox-1)
            ichild = itree(iptr(4)+ibox-1)
c     Compute list1 of ibox if it has never been used as a direct
c     interaction source box
            if (ichild.eq.0) then
               do i=1,itree(iptr(6)+ibox-1)
                  jbox = itree(iptr(7)+mnbors*(ibox-1)+i-1)
c              target boxes in list 1 at the same level
                  nlist1(ibox) = nlist1(ibox) + 1
                  list1(nlist1(ibox),ibox) = jbox
               enddo

               
c     c        compute list1 at level ilev-1
               do i=1,itree(iptr(6)+dad-1)
                  jbox = itree(iptr(7)+mnbors*(dad-1)+i-1)
                  jchild = itree(iptr(4)+jbox-1)
                  if(jchild.eq.0) then
                     distest = 1.05d0*(boxsize(ilev)+boxsize(ilev-1))/
     1                   2.0d0
                     iflist1=1
                     do k=1,ndim
                        dis = dabs(centers(k,jbox)-centers(k,ibox))
                        if (iperiod .eq. 1) then
                           dp1 = bs0-dis
                           if (dp1.lt.dis) dis=dp1
                        endif
                        if (dis.ge.distest) then
                           iflist1=0
                           exit
                        endif
                     enddo
                     if(iflist1.eq.1) then
                        nlist1(ibox) = nlist1(ibox)+1
                        list1(nlist1(ibox),ibox) = jbox
                     endif
                  endif
               enddo
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
c----------------------------------------------------------------
c      
      subroutine bdmk_find_pwexp_boxes(ndim,npwlevel,nboxes,
     1    nlevels,ltree,itree,iptr,iper,ifpwexp)
c
c
c     Determine whether a box needs plane wave expansions
c     in the box fgt.
c
c     1. all boxes below npwlevel need plane wave expansions
c     2. at the cutoff level, i.e., npwlevel, a box needs
c     plane wave expansion if one of its colleagues is a nonleaf 
c     box. 
c                  
c     Thus, a leaf box at the cutoff level may or may not 
c     have plane wave expansion.
c     But if it has plane wave expansion, then the self interaction
c     is handled by the plane wave expansion instead of 
c     direct evaluation. 
c      
c      
c     INPUT arguments
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
c     iper        in: integer
c                 0: free space; 1: periodic
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
      integer iper
      integer itree(ltree)
      integer ifpwexp(nboxes)

      mnbors=3**ndim
      
      do i=1,nboxes
         ifpwexp(i)=0
      enddo

      if (npwlevel .le. 0) then
         ifpwexp(1)=1
      endif

      ncutoff = npwlevel
      if (npwlevel .le. 0) ncutoff=0
      
      if (npwlevel .le. nlevels) then
c     At the cutoff level, a box will have plane wave expansion if 1. it's a nonleaf
c     box; or 2. it's a leaf box but has a nonleaf colleague.
c     Note that we also use plane wave expansion to compute the self interaction of 
c     this box as well
         do ilev=ncutoff,ncutoff
            do 1000 ibox=itree(2*ilev+1),itree(2*ilev+2)
               nchild = itree(iptr(4)+ibox-1)
               if (nchild .eq. 0) then
                  ncoll = itree(iptr(6)+ibox-1)
                  do j=1,ncoll
                     jbox = itree(iptr(7) + (ibox-1)*mnbors+j-1)
                     nchild = itree(iptr(4)+jbox-1)
c                     jlev = itree(iptr(2)+jbox-1)
c                     if (nchild .gt. 0 .and. jlev.eq.ilev) then
                     if (nchild .gt. 0) then
                        ifpwexp(ibox)=1
                        exit
                     endif
                  enddo
               else
                  ifpwexp(ibox)=1
               endif
 1000       continue
         enddo
      endif      

      return
      end
c      
c
c
c
c----------------------------------------------------------------
c      
      subroutine pdmk_find_pwexp_boxes(ndim,npwlevel,nboxes,
     1    nlevels,ltree,itree,iptr,
     2    ndiv,nboxsrcpts,nboxtargpts,
     3    ifpwexpform,ifpwexpeval)
c
c
c     Determine whether a box needs plane wave expansions
c     in the point fgt.
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

      mnbors=3**ndim
      
      do i=1,nboxes
         ifpwexpform(i)=0
      enddo
      
      do i=1,nboxes
         ifpwexpeval(i)=0
      enddo

      if (npwlevel .le. 0) then
         ifpwexpform(1)=1
         ifpwexpeval(1)=1
         return
      endif
      
      if (npwlevel .ge. 0 .and. npwlevel .le. nlevels) then
c     At the cutoff level, a box will have plane wave expansion if 1. it's a nonleaf
c     box; or 2. it's a leaf box but has a nonleaf colleague.
c     Note that we also use plane wave expansion to compute the self interaction of 
c     this box as well
         do ilev=npwlevel,npwlevel
            do ibox=itree(2*ilev+1),itree(2*ilev+2)
               if (nboxsrcpts(ibox).gt.ndiv) ifpwexpform(ibox)=1
            enddo
         enddo

         do ilev=npwlevel,npwlevel
            do ibox=itree(2*ilev+1),itree(2*ilev+2)         
               ncoll = itree(iptr(6)+ibox-1)
               do j=1,ncoll
                  jbox = itree(iptr(7) + (ibox-1)*mnbors+j-1)
                  
                  if (ifpwexpform(jbox).eq.1) then
                     ifpwexpeval(ibox)=1
                     goto 1000
                  endif
               enddo
               
 1000          continue
            enddo
         enddo
      endif      

      return
      end
c      
c

      subroutine dmk_compute_mnlistpw(ndim,nboxes,nlevels,ltree,itree,
     1   iptr,centers,boxsize,mnlistpw)
c
c     determine maximum number of elements in listpw
c
c     NOTE: we use max values
c
      implicit real *8 (a-h,o-z)
      integer iptr(8),ltree
      integer nlevels,nboxes,itree(ltree)
      real *8 centers(ndim,nboxes),boxsize(0:nlevels)
      integer mnlistpw

      mnlistpw = 3**ndim-1

      return
      end
c
c
c
c
c
      subroutine pdmk_compute_listpw(ndim,npwlevel,nboxes,nlevels,
     1    ltree,itree,iptr,centers,boxsize,laddr,ifpwexpform,
     2    mnlistpw,nlistpw,listpw)
c     this subroutine returns lists of pw interaction boxes for point DMK
c
c     input parameters:
c
c     nlevels = number of levels
c     npwlevel   = the cutoff level
c     nboxes = total number of boxes in the tree
c     itree - laddr: tree stuff
c
c     mnlistpw = maximum number of PW interaction boxes at the cutoff level
c      
c     output parameters:
c
c     nlistpw (nboxes) = contains the number of boxes for the PW interaction
c                            for each box
c     listpw (mnlistpw,nboxes) = contains the box ID for the PW interaction
c                            for each box
c      
      implicit real *8 (a-h,o-z)
      integer nlevels,nboxes
      integer iptr(8),ltree,ifpwexpform(nboxes)
      integer itree(ltree),laddr(2,0:nlevels)
      real *8 centers(ndim,nboxes),boxsize(0:nlevels)
      integer mnlistpw
      integer nlistpw(nboxes), listpw(mnlistpw,nboxes)

      integer ibox

      mnbors=3**ndim
      
      do ibox=1,nboxes
         nlistpw(ibox)=0
      enddo

      if (npwlevel.gt.nlevels) return
      
c     make sure ncutoff is >= 0
      ncutoff=max(npwlevel,0)
      do ilev=ncutoff,ncutoff
        do ibox = laddr(1,ilev),laddr(2,ilev)
           ncoll = itree(iptr(6)+ibox-1)
           do i=1,ncoll
              jbox = itree(iptr(7) + (ibox-1)*mnbors+i-1)
c              if (jbox.ne.ibox .and. ((ifpwexpform(ibox).eq.1) 
c     1            .or.(ifpwexpform(jbox).eq.1))) then
              if (jbox.ne.ibox .and. (ifpwexpform(ibox).eq.1)) then 
c             The pw list of ibox at the cutoff level contains its
c             nonleaf colleagues jbox at the cutoff level.
c             If both of them are leaf boxes, their
c             interaction will be computed directly. Self interaction
c             is always directly copied from mp exp to the loc exp and
c             thus is not in the pw list.
c                    nlistpw(ibox)=nlistpw(ibox)+1
c                    listpw(nlistpw(ibox),ibox) = jbox

c     jbox is the target box, ibox is the source box
c     switched for openmp efficiency reasons
                 nlistpw(jbox)=nlistpw(jbox)+1
                 listpw(nlistpw(jbox),jbox) = ibox
              endif
           enddo
        enddo
c     end of ilev do loop
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
c
      subroutine pdmk_compute_all_listpw(ndim,nboxes,nlevels,
     1    ltree,itree,iptr,centers,boxsize,laddr,ifpwexpform,
     2    mnlistpw,nlistpw,listpw)
c     this subroutine returns lists of pw interaction boxes for point DMK
c
c     input parameters:
c
c     nlevels = number of levels
c     nboxes = total number of boxes in the tree
c     itree - laddr: tree stuff
c
c     mnlistpw = maximum number of PW interaction boxes at the cutoff level
c      
c     output parameters:
c
c     nlistpw (nboxes) = contains the number of boxes for the PW interaction
c                            for each box
c     listpw (mnlistpw,nboxes) = contains the box ID for the PW interaction
c                            for each box
c      
      implicit real *8 (a-h,o-z)
      integer nlevels,nboxes
      integer iptr(8),ltree,ifpwexpform(nboxes)
      integer itree(ltree),laddr(2,0:nlevels)
      real *8 centers(ndim,nboxes),boxsize(0:nlevels)
      integer mnlistpw
      integer nlistpw(nboxes), listpw(mnlistpw,nboxes)

      integer ibox

      mnbors=3**ndim
      
      do ibox=1,nboxes
         nlistpw(ibox)=0
      enddo

      do ilev=0,nlevels
        do ibox = laddr(1,ilev),laddr(2,ilev)
           ncoll = itree(iptr(6)+ibox-1)
           do i=1,ncoll
              jbox = itree(iptr(7) + (ibox-1)*mnbors+i-1)
c              if (jbox.ne.ibox .and. ((ifpwexpform(ibox).eq.1) 
c     1            .or.(ifpwexpform(jbox).eq.1))) then
              if (jbox.ne.ibox .and. (ifpwexpform(ibox).eq.1)) then 
c             The pw list of ibox at the cutoff level contains its
c             nonleaf colleagues jbox at the cutoff level.
c             If both of them are leaf boxes, their
c             interaction will be computed directly. Self interaction
c             is always directly copied from mp exp to the loc exp and
c             thus is not in the pw list.
c                    nlistpw(ibox)=nlistpw(ibox)+1
c                    listpw(nlistpw(ibox),ibox) = jbox

c     jbox is the target box, ibox is the source box
c     switched for openmp efficiency reasons
                 nlistpw(jbox)=nlistpw(jbox)+1
                 listpw(nlistpw(jbox),jbox) = ibox
              endif
           enddo
        enddo
c     end of ilev do loop
      enddo

      return
      end
c
c
c
c
c
c
      subroutine bdmk_compute_listpw(ndim,npwlevel,nboxes,nlevels,
     1    ltree,itree,iptr,centers,boxsize,laddr,
     2    mnlistpw,nlistpw,listpw)
c     this subroutine returns lists of pw interaction boxes for the box FGT
c
c     input parameters:
c
c     nlevels = number of levels
c     npwlevel   = the cutoff level
c     nboxes = total number of boxes in the tree
c     itree - laddr: tree stuff
c
c     mnlistpw = maximum number of a particular type of PW expansions
c      
c     output parameters:
c
c     nlistpw (nboxes) = contains the number of boxes for the PW interaction
c                            for each box
c     listpw (mnlistpw,nboxes) = contains the box ID for the PW interaction
c                            for each box
c      
      implicit real *8 (a-h,o-z)
      integer nlevels,nboxes
      integer iptr(8),ltree
      integer itree(ltree),laddr(2,0:nlevels)
      real *8 centers(ndim,nboxes),boxsize(0:nlevels)
      integer mnlistpw
      integer nlistpw(nboxes), listpw(mnlistpw,nboxes)

      integer ibox

      mnbors=3**ndim
      
      do ibox=1,nboxes
         nlistpw(ibox)=0
      enddo

      if (npwlevel.gt.nlevels) return
      
c     make sure ncutoff is >= 0
      ncutoff=max(npwlevel,0)
      do ilev=ncutoff,ncutoff
        do ibox = laddr(1,ilev),laddr(2,ilev)
           ncoll = itree(iptr(6)+ibox-1)
           do i=1,ncoll
              jbox = itree(iptr(7) + (ibox-1)*mnbors+i-1)
c              jlev = itree(iptr(2)+jbox-1)
c              if (jbox.ne.ibox .and. ilev.eq.jlev .and. 
              if (jbox.ne.ibox .and. 
     1            (itree(iptr(4)+jbox-1).gt.0
     2            .or. itree(iptr(4)+ibox-1).gt.0)) then
c             The pw list of ibox at the cutoff level contains its
c             colleague jbox at the cutoff level if either of them
c             has children. If both of them are leaf boxes, their
c             interaction will be computed directly. Self interaction
c             is always directly copied from mp exp to the loc exp and
c             thus is not in the pw list.
                 nlistpw(ibox)=nlistpw(ibox)+1
                 listpw(nlistpw(ibox),ibox) = jbox
              endif
           enddo
        enddo
c     end of ilev do loop
      enddo

      return
      end
c
c
c
c
      subroutine compute_mnlists(ndim,nboxes,nlevels,laddr,
     1    centers,boxsize,iparent,nchild,
     2    ichild,isep,nnbors,nbors,iper,mnlist1,mnlist2)
c     Compute max number of boxes in list1 and list2
c     
c     Both lists contains boxes in the direct interaction part of the DMK.
c     The direct interaction is calculated by refining all leaf boxes one more time,
c     in order to reduce the number of points that require the distance calculation.
c     
c     Everything being equal, this would reduce the number of direct interaction
c     points from 3^3*s to (2.5)^3*s. A reduction from 6^3 to 5^3, almost a factor of 2.
c      
c     The main purpose of doing this is that if one wants to write a SIMD vectorized
c     fast direct interaction routine, then one will need to reduce the number of direct
c     interaction points when the kernel is truncated. For truncated kernels, there is always
c     at least one "if" statement inside the SIMD function. And after the "if" statement,
c     the kernel will always be evaluated regardless whether its value is actually needed 
c     or not. Thus, if one evaluates the direct interaction at the genuine leaf nodes,
c     the reduction from 27 to 4pi/3 due to the finite range of the trucated kernel is gone
c     from the SIMD vectorization. This fact alone would make the SIMD code almost useless since
c     27/(4pi/3) is about 7, and SIMD can gain at most a factor of 8 speedup.
c      
c     
c      
c     list 1 contains boxes that are closer to the given source box.
c     the parameter isep determines which boxes belong to list 1, and which boxes
c     belong to list 2.
c
c     Suppose that d2 = \|C_S-C_T\|^2, where C_S is the center of the given source box,
c     C_T is the center of the target box, and \| . \| is the distance between them.
c     Then C_T belongs to list 1 if d2 <= isep * boxsize(source box)^2. And it belongs
c     to list 2 otherwise.
c
c     When the distribution is uniform, there are 5^ndim boxes altogether in these 
c     two lists.
c
c     list 1 will use SIMD vectorization to speed up direct interactions. With proper
c     implementation of both SIMD code and sequential code and proper division of these
c     two lists, SIMD code will gain a 
c     factor of 8 speedup since points in boxes in list 1 are almost always within
c     the domain of influence of the kernel. While points in boxes in list 2 are almost
c     always outside the domain of influence of the kernel, and thus kernel evaluation
c     is not needed. In this case, a sequential code should be faster than the SIMD code.
c     Hence the division of these two lists.
c
c     Note, fast kernel evaluation only works for isep=3 for now!
c     
      implicit none
      integer ndim,nlevels,nboxes
      integer iper
      integer laddr(2,0:nlevels)
      double precision boxsize(0:nlevels)
      double precision centers(ndim,nboxes)
      integer iparent(nboxes),nchild(nboxes),ichild(2**ndim,nboxes)
      integer mnbors,isep
      integer nnbors(nboxes),nbors(3**ndim,nboxes)
      integer mnlist1,mnlist2

c     Temp variables
      integer ilev,ibox,jbox,kbox,i,j,k
      integer firstbox,lastbox,dad,mc,ifnbor,iflist1
      double precision dis,distest,bs0,dp1

      if (ndim.eq.3) then
         if (isep.eq.0) mnlist1=1
         if (isep.eq.3) mnlist1=27 
         if (isep.eq.4) mnlist1=33
         if (isep.eq.5) mnlist1=57
         if (isep.eq.6) mnlist1=81
c     use large values of isep so that all boxes are in list1
         if (isep.ge.20) mnlist1=125
            
         mnlist2=5**3-mnlist1
      elseif (ndim.eq.2) then
         if (isep.eq.0) mnlist1=1
         if (isep.eq.2) mnlist1=9
         if (isep.ge.20) mnlist1=25
         
         mnlist2=5**2-mnlist1
      endif

      return
      end
c      
c      
c      
c      
c      
c----------------------------------------------------------------
c      
c      
      subroutine pdmk_compute_modified_lists(ndim,
     1    nboxes,nlevels,ltree,itree,iptr,centers,boxsize,iperiod,
     2    ifleafbox,npwlevel,isep,
     3    mnlist1,nlist1,list1,
     3    mnlist2,nlist2,list2)
c
c
c     This subroutine computes list 1 and list 2 in the above subroutine.
c
c      
c      
c     INPUT arguments
c     ndim        in: integer
c                 dimension of the space
c
c     nlevels     in: integer
c                 Number of levels
c
c     npwlevel    in: integer
c                 Cutoff level
c     ifpwexp     in: integer(nboxes)
c                 1: requires pwexp; 0: does not require pwexp
c      
c     nboxes      in: integer
c                 Total number of boxes
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
c     centers     in: real *8(ndim,nboxes)
c                 xy coordinates of centers of boxes
c   
c     boxsize     in: real *8(0:nlevels)
c                 Array of boxsizes
c   
c     iperiod     in: integer
c                 0: free space; 1: periodic
c 
c     mnlist1     in: integer
c                 max number of boxes in list 1 of a box
c
c--------------------------------------------------------------
c     OUTPUT arguments:
c     nlist1      out: integer(nboxes)
c                 nlist1(i) is the number of boxes in list 1 
c                 of box i
c
c     list1       out: integer(mnlist1,nboxes)
c                 list1(j,i) is the box id of the jth box in 
c                 list1 of box i.
c                  
c      
c---------------------------------------------------------------
      implicit none
      integer nlevels,npwlevel,nboxes,ndim
      integer iptr(8),ltree,isep
      integer iperiod,itree(ltree)
      integer ifleafbox(nboxes)
      real *8 boxsize(0:nlevels)
      real *8 centers(ndim,nboxes)
      integer mnlist1,mnlist2
      integer nlist1(nboxes), list1(mnlist1,nboxes)
      integer nlist2(nboxes), list2(mnlist2,nboxes)

c     Temp variables
      integer ilev,ibox,jbox,dad,mnbors,mc,jboxchild
      integer i,ifirstbox,ilastbox,iflist1,iflist2,k,j,jchild
      real *8 distest,dis,bs0,dp1,distest2,d2
      
      bs0=boxsize(0)

      mnbors=3**ndim
      mc=2**ndim
      
      if (npwlevel+1.gt.nlevels) return

      do ilev = npwlevel,npwlevel+1
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
            nlist1(ibox) = 0
            nlist2(ibox) = 0
         enddo
      enddo

c     Setting parameters for level = 0
      if(itree(iptr(4)).eq.0) then
         nlist1(1) = 1 
         list1(1,1) = 1
         return
      endif


c     .... Step 1 deal with self interactions and leaf boxes at the cutoff level
c      
c     1. Self interactions are always evaluated at the coarse level
c     to reduce cache misses in SIMD fast kernel evaluation code
c      
c     2. Leaf boxes at the coarse level are also evaluated at the coarse level
      do ilev = npwlevel,npwlevel
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
            if (ifleafbox(ibox).eq.1) then
               do i=1,itree(iptr(6)+ibox-1)
                  jbox = itree(iptr(7)+mnbors*(ibox-1)+i-1)
                  jchild = itree(iptr(4)+jbox-1)
                  if ((jbox.eq.ibox) .or. (jchild.eq.0)) then
                     nlist1(ibox) = nlist1(ibox) + 1
                     list1(nlist1(ibox),ibox) = jbox
                  endif
               enddo
            endif
         enddo
      enddo

c     By the design of the algorithm, there is no need to check
c     boxes at the finer level.
c     Since every leaf box is refined once, there are no boxes
c     at level ilev-1 in the colleague list. That is, the neighbors of a box
c     at the cutoff level are exactly its colleagues. Thus, there is no need
c     to check boxes above the cutoff level either.
c
      do ilev = npwlevel+1,npwlevel+1
c        same level distance test parameter
         distest = 1.05d0*boxsize(ilev)*2
c        squared distance separating list1 and list2 boxes
         distest2 = 1.05d0*boxsize(ilev)**2*isep
         
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,dad,i,jbox,j,kbox,dis,distest,k)
         do ibox = itree(2*ilev+1),itree(2*ilev+2)
c           ibox is the source box
            dad = itree(iptr(3)+ibox-1)
c     Compute list1 of ibox if it contains less than ndiv sources
            if (ifleafbox(dad).eq.1) then
               do i=1,itree(iptr(6)+dad-1)
                  jbox = itree(iptr(7)+mnbors*(dad-1)+i-1)
                  jchild = itree(iptr(4)+jbox-1)
                  if ((jbox.ne.dad) .and. (jchild.gt.0)) then
c                    boxes that are children of the parent box's neighbors
                     do 2000 j=1,jchild
                        jboxchild=itree(iptr(5)+mc*(jbox-1)+j-1)
                        d2 = 0
                        do k=1,ndim
                           dis = dabs(centers(k,jboxchild)
     1                         -centers(k,ibox))
                           if (dis.ge.distest) then
                              goto 2000
                           endif
                           d2 = d2+dis*dis
                        enddo

                        if (d2.le. distest2) then
                           nlist1(ibox) = nlist1(ibox) + 1
                           list1(nlist1(ibox),ibox) = jboxchild
                        else
                           nlist2(ibox) = nlist2(ibox) + 1
                           list2(nlist2(ibox),ibox) = jboxchild
                        endif
 2000                continue
                  endif
               enddo
            endif
         enddo
C$OMP END PARALLEL DO         
      enddo
      
      return
      end
c      
c
c
cc      
c      
c
c
c
c----------------------------------------------------------------
c      
      subroutine pdmk_find_all_pwexp_boxes(ndim,nboxes,
     1    nlevels,ltree,itree,iptr,
     2    ndiv,nboxsrcpts,nboxtargpts,
     3    ifpwexpform,ifpwexpeval,iftensprodeval)
c
c
c     Determine whether a box needs plane wave expansions
c     in the point dmk.
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
            if (nboxsrcpts(ibox).gt.ndiv) ifpwexpform(ibox)=1
         enddo
      enddo

      do ilev=0,nlevels
         do ibox=itree(2*ilev+1),itree(2*ilev+2)         
            ncoll = itree(iptr(6)+ibox-1)
            do j=1,ncoll
               jbox = itree(iptr(7) + (ibox-1)*mnbors+j-1)
               npts=nboxsrcpts(jbox)+nboxtargpts(jbox)
               if (ifpwexpform(jbox).eq.1 .and. npts.gt.0) then
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
c----------------------------------------------------------------
c      
      subroutine pdmk_find_all_pwexp_boxes2(ndim,nboxes,
     1    nlevels,ltree,itree,iptr,
     2    ndiv,nboxsrcpts,nboxtargpts,ifleafbox,
     3    ifpwexpform,ifpwexpeval,iftensprodeval)
c
c
c     Determine whether a box needs plane wave expansions
c     in the point fgt.
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
      integer itree(ltree),ifleafbox(nboxes)
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
            if (ifleafbox(ibox).eq.0) ifpwexpform(ibox)=1
         enddo
      enddo

      do ilev=0,nlevels
         do ibox=itree(2*ilev+1),itree(2*ilev+2)         
            ncoll = itree(iptr(6)+ibox-1)
            do j=1,ncoll
               jbox = itree(iptr(7) + (ibox-1)*mnbors+j-1)
               npts=nboxsrcpts(jbox)+nboxtargpts(jbox)
               if (ifpwexpform(jbox).eq.1 .and. npts.gt.0) then
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
c      
c
c
c
c----------------------------------------------------------------
c      
      subroutine pdmk_find_all_pwexp_boxes3(ndim,nboxes,
     1    nlevels,ltree,itree,iptr,nboxsrcpts,nboxtargpts,
     2    ifpwexpform,ifpwexpeval,iftensprodeval)
c
c
c     Determine whether a box needs plane wave expansions
c     in the point DMK.
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
      integer nlevels,npwlevel,nboxes,ndim,nchild
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
            nchild = itree(iptr(4)+ibox-1)
            if (nchild.gt.0) ifpwexpform(ibox)=1
         enddo
      enddo

      do ilev=0,nlevels
         do ibox=itree(2*ilev+1),itree(2*ilev+2)         
            ncoll = itree(iptr(6)+ibox-1)
            do j=1,ncoll
               jbox = itree(iptr(7) + (ibox-1)*mnbors+j-1)
               npts=nboxsrcpts(jbox)+nboxtargpts(jbox)
               if (ifpwexpform(jbox).eq.1 .and. npts.gt.0) then
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

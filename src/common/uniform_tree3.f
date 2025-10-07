c
c     uniform tree in n dimensions
c     created by Shidong Jiang on 02/07/2025
c
c   This code has the following user callable routines
c
c     sort points to bins of arbitrary size, removing 2**(-nlevels) restriction
c     on the size of bins.
c 
      subroutine compute_uniform_tree_info(ndim,cen0,bs0,
     1    ml0,nboxes,binsize,centers,ibinid,iboxid,ncollid)
c
c     compute useful information about bins on a uniform tree
c----------------------------------------
c     input parameters:
c    - ndim: dimension of the underlying space
c    - cen0(ndim): center of the root box
c    - bs0: size of the root box
c    - ml0: integer
c          number of bins in each dimension
c    - nboxes: integer
c             number of boxes = ml0**ndim
c    - binsize: double precision 
c              size of the binsc

c     output parameters:
c     centers: double precision (dim,nboxes)
c              coordinates of bin centers
c     ibinid: the leaf box id given the integral coordinates
c             of its center in the lattice
c     iboxid: label the bins in natural order 1, 2, 3, etc.
c     ncollid: list of colleagues for each bin
      
      implicit none
      integer ndim,nboxes
      double precision centers(ndim,nboxes)
      double precision rcutoff,binsize
      real *8 bs0,bsinv,cen0(ndim),ref(ndim)

      integer ibinid(*),iboxid(*)
      
      integer i,j,k,n,ind,id,ixyz(ndim),kxyz(ndim)

      integer ibox,jbox,mnbors,ml(0:ndim),ml0
      integer ncollid(3**ndim,nboxes)
      
      mnbors=3**ndim

      ml(0)=1
      do i=1,ndim
         ml(i) = ml(i-1)*ml0
      enddo
      
      do i=1,ndim
         ref(i) = cen0(i)-0.5d0*bs0
      enddo

      bsinv = 1.0d0/binsize
      do i=1,ndim
         ref(i) = -ref(i)*bsinv
      enddo

      ibox=0
      do i=1,ml0
      do j=1,ml0
      do k=1,ml0
         ibox=ibox+1
         centers(1,ibox)=cen0(1)-0.5d0*bs0+(k-0.5d0)*binsize
         centers(2,ibox)=cen0(2)-0.5d0*bs0+(j-0.5d0)*binsize
         centers(3,ibox)=cen0(3)-0.5d0*bs0+(i-0.5d0)*binsize
      enddo
      enddo
      enddo
      
      id = 0
      do ibox=1,nboxes
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

      do ibox=1,nboxes
         do i=1,ndim
            ixyz(i) = centers(i,ibox)*bsinv+ref(i)
         enddo

         id=0
         do i=-1,1
         do j=-1,1
         do k=-1,1
            kxyz(1)=ixyz(1)+k
            kxyz(2)=ixyz(2)+j
            kxyz(3)=ixyz(3)+i
            if (kxyz(1).lt.0) kxyz(1)=kxyz(1)+ml0
            if (kxyz(1).ge.ml0) kxyz(1)=kxyz(1)-ml0
            if (kxyz(2).lt.0) kxyz(2)=kxyz(2)+ml0
            if (kxyz(2).ge.ml0) kxyz(2)=kxyz(2)-ml0
            if (kxyz(3).lt.0) kxyz(3)=kxyz(3)+ml0
            if (kxyz(3).ge.ml0) kxyz(3)=kxyz(3)-ml0
            ind=1
            do n=1,ndim
               ind = ind+kxyz(n)*ml(n-1)
            enddo
            jbox = ibinid(ind)
            id=id+1
            ncollid(id,ibox)=jbox
         enddo
         enddo
         enddo
      enddo
      
      return
      end
c
c
c
c
      subroutine compute_uniform_tree_info2(ndim,cen0,bs0,
     1    ml0,nboxes,binsize,centers,ibinid,iboxid,ncollid)
c
c     compute useful information about bins on a uniform tree
c----------------------------------------
c     input parameters:
c    - ndim: dimension of the underlying space
c    - cen0(ndim): center of the root box
c    - bs0: size of the root box
c    - ml0: integer
c          number of bins in each dimension
c    - nboxes: integer
c             number of boxes = ml0**ndim
c    - binsize: double precision 
c              size of the binsc

c     output parameters:
c     centers: double precision (dim,nboxes)
c              coordinates of bin centers
c     ibinid: the leaf box id given the integral coordinates
c             of its center in the lattice
c     iboxid: label the bins in natural order 1, 2, 3, etc.
c     ncollid: list of colleagues for each bin
      
      implicit none
      integer ndim,nboxes
      double precision centers(ndim,nboxes)
      double precision rcutoff,binsize
      real *8 bs0,bsinv,cen0(ndim),ref(ndim)

      integer ibinid(*),iboxid(*)
      
      integer i,j,k,m,n,ind,id,ixyz(ndim),kxyz(ndim)

      integer ibox,jbox,mnbors,ml(0:ndim),ml0
      integer ncollid(5**ndim,nboxes)

      m=2
      mnbors=(2*m+1)**ndim

      ml(0)=1
      do i=1,ndim
         ml(i) = ml(i-1)*ml0
      enddo
      
      do i=1,ndim
         ref(i) = cen0(i)-0.5d0*bs0
      enddo

      bsinv = 1.0d0/binsize
      do i=1,ndim
         ref(i) = -ref(i)*bsinv
      enddo

      ibox=0
      do i=1,ml0
      do j=1,ml0
      do k=1,ml0
         ibox=ibox+1
         centers(1,ibox)=cen0(1)-0.5d0*bs0+(k-0.5d0)*binsize
         centers(2,ibox)=cen0(2)-0.5d0*bs0+(j-0.5d0)*binsize
         centers(3,ibox)=cen0(3)-0.5d0*bs0+(i-0.5d0)*binsize
      enddo
      enddo
      enddo
      
      id = 0
      do ibox=1,nboxes
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

      do ibox=1,nboxes
         do i=1,ndim
            ixyz(i) = centers(i,ibox)*bsinv+ref(i)
         enddo

         id=0
         do i=-m,m
         do j=-m,m
         do k=-m,m
            kxyz(1)=ixyz(1)+k
            kxyz(2)=ixyz(2)+j
            kxyz(3)=ixyz(3)+i
            if (kxyz(1).lt.0) kxyz(1)=kxyz(1)+ml0
            if (kxyz(1).ge.ml0) kxyz(1)=kxyz(1)-ml0
            if (kxyz(2).lt.0) kxyz(2)=kxyz(2)+ml0
            if (kxyz(2).ge.ml0) kxyz(2)=kxyz(2)-ml0
            if (kxyz(3).lt.0) kxyz(3)=kxyz(3)+ml0
            if (kxyz(3).ge.ml0) kxyz(3)=kxyz(3)-ml0
            ind=1
            do n=1,ndim
               ind = ind+kxyz(n)*ml(n-1)
            enddo
            jbox = ibinid(ind)
            id=id+1
            ncollid(id,ibox)=jbox
         enddo
         enddo
         enddo
      enddo
      
      return
      end
c
c
c
c
      subroutine sort_pts_to_bins(ndim,cen0,bs0,ml0,nboxes,
     1    binsize,centers,ibinid,iboxid,
     2    ns,sources,isrc,isrcinv,isrcse)
c
c     bin sort points to the leaf boxes on a uniform tree in R^d
c----------------------------------------
c     input parameters:
c    - ndim: dimension of the underlying space
c    - nboxes: integer
c        number of boxes
c    - centers: double precision (dim,nboxes)
c        coordinates of box centers in the oct tree
c    - binsize: double precision
c        size of the bin
c
c     output parameters:
c     isrc : 
c     isrcse : 
c
      implicit none
      integer ndim,nboxes,ns,ml0
      double precision centers(ndim,nboxes)
      double precision binsize
      real *8 sources(ndim,ns)
      real *8 bs0,cen0(ndim),ref(ndim)

      integer ibinid(*),iboxid(*)
      integer isrc(ns),isrcinv(ns),isrcse(2,nboxes)
      
      integer i,ilev,ind,ibin,j

      integer ibox,jbox,boxid,ml(0:ndim),ixyz(ndim)
      integer, allocatable :: nsrc(:),isrcbox(:),ip(:)

      integer id,mlm1,istart,bin
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
      
      ml(0)=1
      do i=1,ndim
         ml(i) = ml(i-1)*ml0
      enddo
      
      mlm1 = ml0-1
      
      do i=1,ndim
         ref(i) = cen0(i)-0.5d0*bs0
      enddo

      bsinv = 1.0d0/binsize
      do i=1,ndim
         ref(i) = -ref(i)*bsinv
      enddo

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

      ind=1
      istart=1
      ip(1)=0
      do ibox=1,nboxes
         ind=ind+1
         ip(ind)=ip(ind-1)+nsrc(ibox)
         isrcse(1,ibox) = istart+ip(ind-1)
         isrcse(2,ibox) = istart+ip(ind)-1
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

            

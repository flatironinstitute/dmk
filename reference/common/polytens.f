c
c
c
c     this file contains routines for working with tensor
c     product gauss-legendre nodes and polynomials in 2 and
c     3 dimensions; also provides some related spectral
c     differentiation operators
c
c--------------------------------------------------------------
c
c     CONVENTIONS
c
c     ndeg refers to polynomial degree
c      
c     n refers to order of approximation and number of nodes
c     used in each direction (n = ndeg+1)
c
c     type character specifies the set of tensor
c     polynomials to use. current options
c
c     type = 'F', full degree polynomials (T_i(x)T_j(y)T_k(z) where
c                 each of i, j, and k goes from 0 to ndeg)
c     type = 'T', total degree polynomials (T_i(x)T_j(y)T_k(z) where
c                 sum of i,j,k>=0 is less than or equal to ndeg)
c
c     TODO: implement "Euclidean degree"
c
c     a tensor grid of points is traversed with x on the inner
c        loop, y on the loop above that, and z above that.
c        e.g. for a 2x2x2 grid, we have the order:
c            (x1,y1,z1), (x2,y1,z1), (x1,y2,z1), (x2,y2,z1)
c     (x1,y1,z2), (x2,y1,z2), (x1,y2,z2), (x2,y2,z2)
c      
c     tensor product polynomials are numbered analagously
c        e.g. for full degree at ndeg = 3, we would have the order
c        1, T_1(x), T_2(x), T_1(y), T_1(x)T_1(y), T_2(x)T_1(y),
c        T_2(y), T_1(x)T_2(y), T_2(x)T_2(y), T_1(z), T_1(x)T_1(z),
c        T_2(x) T_1(z), T_1(y)T_1(z), T_1(x)T_1(y)T_1(z), T_2(x)
c        T_1(y)T_1(z), T_2(y) T_1(z), T_1(x)T_2(y) T_1(z),
c        T_2(x) T_2(y) T_1(z), T_2(z), T_1(x)T_2(z), T_2(x) T_2(z),
c        T_1(y)T_2(z), T_1(x)T_1(y)T_2(z), T_2(x) T_1(y) T_2(z),
c        T_2(y) T_2(z), T_1(x)T_2(y) T_2(z), T_2(x) T_2(y) T_2(z)
c 
c        e.g. for total degree, we would have the order
c        1, T_1(x), T_2(x), T_1(y), T_1(x)T_1(y), T_2(y),
c        T_1(z), T_1(x)T_1(z), T_1(y)T_1(z), T_2(z)
c
c-------------------------------------------------------------     
c
c     ROUTINES
c      
c     legetens_npol_*d - total number of polynomials up
c                        to a given degree of specified
c                        type in 2 or 3 dimensions
c     legetens_exps_*d - get nodes, weights, and, optionally,
c                        projection and evaluation matrices      
c                        in 2 or 3 dimensions

      subroutine polytens_npol_2d(ndeg,type,npol)
c
c     return the number of polynomials of a given type
c     up to degree ndeg
c      
      integer n, npol, ndeg
      character type

      n = ndeg + 1
      
      if (type .eq. 'f' .or. type .eq. 'F') then
         npol = n**2
      else if (type .eq. 't' .or. type .eq. 'T') then
         npol = n*(n+1)/2
      endif

      return
      end

c
c
c
c
      subroutine polytens_npol_3d(ndeg,type,npol)
c
c     return the number of polynomials of a given type
c     up to degree ndeg
c      
      integer n, npol, ndeg
      character type

      n = ndeg + 1
      
      if (type .eq. 'f' .or. type .eq. 'F') then
         npol = n**3
      else if (type .eq. 't' .or. type .eq. 'T') then
         npol = n*(n+1)*(n+2)/6
      endif

      return
      end
c
c
c
c


      subroutine polytens_pow2ind_3d(ndeg,type,ip2ind)
      integer ndeg, ip2ind(ndeg+1,ndeg+1,ndeg+1)
      character type

      integer i, j, k, ipol

      do i = 1,ndeg+1
         do j = 1,ndeg+1
            do k = 1,ndeg+1
               ip2ind(k,j,i) = -1
            enddo
         enddo
      enddo

      if (type .eq. 'f' .or. type .eq. 'F') then
         ipol = 0
         do i = 1,ndeg+1
            do j = 1,ndeg+1
               do k = 1,ndeg+1
                  ipol = ipol+1
                  ip2ind(k,j,i) = ipol
               enddo
            enddo
         enddo
      else if (type .eq. 't' .or. type .eq. 'T') then
         ipol = 0
         do i = 1,ndeg+1
            do j = 1,ndeg+1+1-i
               do k = 1,ndeg+1+2-i-j
                  ipol = ipol+1
                  ip2ind(k,j,i) = ipol
               enddo
            enddo
         enddo
      endif

      return
      end
c
c
c
c
      subroutine polytens_ind2pow(ndim,ndeg,type,iind2p)
      integer ndeg, iind2p(ndim,*)
      character type

      integer i,ndim

      if (ndim.eq.1) then
         do i=1,ndeg+1
            iind2p(1,i)=i-1
         enddo
      elseif (ndim.eq.2) then
         call polytens_ind2pow_2d(ndeg,type,iind2p)
      elseif (ndim.eq.3) then
         call polytens_ind2pow_3d(ndeg,type,iind2p)
      endif
      
      return
      end
c
c
c
c
      subroutine polytens_ind2pow_2d(ndeg,type,iind2p)
      integer ndeg, iind2p(2,*)
      character type

      integer i, j, ipol


      if (type .eq. 'f' .or. type .eq. 'F') then
         ipol = 0
         do i = 1,ndeg+1
            do j = 1,ndeg+1
               ipol = ipol+1
               iind2p(1,ipol) = j-1
               iind2p(2,ipol) = i-1                
            enddo
         enddo
      else if (type .eq. 't' .or. type .eq. 'T') then
         ipol = 0
         do i = 1,ndeg+1
            do j = 1,ndeg+1+1-i
               ipol = ipol+1
               iind2p(1,ipol) = j-1
               iind2p(2,ipol) = i-1
            enddo
         enddo
      endif

      return
      end
c
c
c
c
      subroutine polytens_ind2pow_3d(ndeg,type,iind2p)
      integer ndeg, iind2p(3,*)
      character type

      integer i, j, k, ipol


      if (type .eq. 'f' .or. type .eq. 'F') then
         ipol = 0
         do i = 1,ndeg+1
            do j = 1,ndeg+1
               do k = 1,ndeg+1
                  ipol = ipol+1
                  iind2p(1,ipol) = k-1
                  iind2p(2,ipol) = j-1
                  iind2p(3,ipol) = i-1                
               enddo
            enddo
         enddo
      else if (type .eq. 't' .or. type .eq. 'T') then
         ipol = 0
         do i = 1,ndeg+1
            do j = 1,ndeg+1+1-i
               do k = 1,ndeg+1+2-i-j
                  ipol = ipol+1
                  iind2p(1,ipol) = k-1
                  iind2p(2,ipol) = j-1
                  iind2p(3,ipol) = i-1
               enddo
            enddo
         enddo
      endif

      return
      end
c
c
c
c
      subroutine polytens_pow2ind_2d(ndeg,type,ip2ind)
      integer ndeg, ip2ind(ndeg+1,ndeg+1)
      character type

      integer i, j, ipol

      do i = 1,ndeg+1
         do j = 1,ndeg+1
            ip2ind(j,i) = -1
         enddo
      enddo
      
      if (type .eq. 'f' .or. type .eq. 'F') then
         ipol = 0
         do i = 1,ndeg+1
            do j = 1,ndeg+1
               ipol = ipol+1
               ip2ind(j,i) = ipol
            enddo
         enddo
      else if (type .eq. 't' .or. type .eq. 'T') then
         ipol = 0
         do i = 1,ndeg+1
            do j = 1,ndeg+1+1-i
               ipol = ipol+1
               ip2ind(j,i) = ipol
            enddo
         enddo
      endif

      return
      end
c
c
c
c
c
      subroutine polytens_exps_nd(ndim,ipoly,itype,n,type,x,
     1    u,ldu,v,ldv,w)
c                 input parameters:
c
c  itype - the type of the calculation to be performed
c          itype=0 means that only the gaussian nodes are 
c                  to be constructed. 
c          itype=1 means that only the nodes and the weights 
c                  are to be constructed
c          itype=2 means that the nodes, the weights, and
c                  the matrices u, v are to be constructed
c          itype=3 only construct u
c          itype=4 only construct v
      
      implicit none
      integer ndim,ipoly, itype, n, ldu, ldv
      character type
      real *8 x(ndim,*),w(*)
      real *8 u(ldu,*), v(ldv,*)

      if (ndim.eq.1) then
         if (ipoly.eq.0) then
            call legeexps(itype,n,x,u,v,w)
         elseif (ipoly.eq.1) then
            call chebexps(itype,n,x,u,v,w)
         endif
      elseif (ndim.eq.2) then
         call polytens_exps_2d(ipoly,itype,n,type,x,u,ldu,v,ldv,w)
      elseif (ndim.eq.3) then
         call polytens_exps_3d(ipoly,itype,n,type,x,u,ldu,v,ldv,w)
      endif
      
      return
      end
c
c
      subroutine polytens_exps_2d(ipoly,itype,n,type,x,u,ldu,v,ldv,w)
c                 input parameters:
c
c  itype - the type of the calculation to be performed
c          itype=0 means that only the gaussian nodes are 
c                  to be constructed. 
c          itype=1 means that only the nodes and the weights 
c                  are to be constructed
c          itype=2 means that the nodes, the weights, and
c                  the matrices u, v are to be constructed
c          itype=3 only construct u
c          itype=4 only construct v
      
      implicit none
      integer ipoly, itype, n, ldu, ldv
      character type
      real *8 x(2,*),w(*)
      real *8 u(ldu,*), v(ldv,*)
      real *8 x1d(n), w1d(n), u1d(n,n), v1d(n,n)
      integer i,j,ipt,itype1d,io, jo,ipol
      
      itype1d = 0
      if (itype .ge. 1) then
         itype1d = 1
      endif
      if (itype .ge. 2) then
         itype1d = 2
      endif
      
      if (ipoly.eq.0) call legeexps(itype1d,n,x1d,u1d,v1d,w1d)
      if (ipoly.eq.1) call chebexps(itype1d,n,x1d,u1d,v1d,w1d)

      ipt = 0
      do i=1,n
         do j=1,n
            ipt = ipt + 1
            x(1,ipt) = x1d(j)
            x(2,ipt) = x1d(i)               
        enddo
      enddo

      if (itype .ge. 1) then
         ipt = 0
         do i=1,n
            do j=1,n
               ipt = ipt + 1
               w(ipt) = w1d(i)*w1d(j)
            enddo
         enddo
      endif


      if (itype .eq. 2 .or. itype .eq. 3) then
c     construct u from 1d u
         if (type .eq. 'f' .or. type .eq. 'F') then         
            ipt = 0
            do io = 1,n
            do jo = 1,n
               ipt = ipt + 1
               ipol = 0
               do i=1,n
               do j=1,n
                  ipol = ipol + 1
                  u(ipol,ipt) =  u1d(i,io)*u1d(j,jo)
               enddo
               enddo
            enddo
            enddo

         else if (type .eq. 't' .or. type .eq. 'T') then

            ipt = 0
            do io = 1,n
            do jo = 1,n
               ipt = ipt + 1
               ipol = 0
               do i=1,n
               do j=1,n+1-i
                  ipol = ipol + 1
                  u(ipol,ipt) = u1d(i,io)*u1d(j,jo)
               enddo
               enddo
            enddo
            enddo
         endif
      endif
      
      if (itype .eq. 2 .or. itype .eq. 4) then
c     construct v from 1d v
         if (type .eq. 'f' .or. type .eq. 'F') then         
            ipol = 0
            do io = 1,n
            do jo = 1,n
               ipol = ipol + 1
               ipt = 0
               do i=1,n
               do j=1,n
                  ipt = ipt + 1
                  v(ipt,ipol) =  v1d(i,io)*v1d(j,jo)
               enddo
               enddo
            enddo
            enddo

         else if (type .eq. 't' .or. type .eq. 'T') then

            ipol = 0
            do io = 1,n
            do jo = 1,n+1-io
               ipol = ipol + 1
               ipt = 0
               do i=1,n
               do j=1,n
                  ipt = ipt + 1
                  v(ipt,ipol) = v1d(i,io)*v1d(j,jo)
               enddo
               enddo
            enddo
            enddo
         endif
      endif         

      return
      end
c
c
      subroutine polytens_exps_3d(ipoly,itype,n,type,x,u,ldu,v,ldv,w)
c                 input parameters:
c
c  itype - the type of the calculation to be performed
c          itype=0 means that only the gaussian nodes are 
c                  to be constructed. 
c          itype=1 means that only the nodes and the weights 
c                  are to be constructed
c          itype=2 means that the nodes, the weights, and
c                  the matrices u, v are to be constructed
c          itype=3 only construct x,w, and u
c          itype=4 only construct x,w, and v
      
      implicit none
      integer ipoly, itype, n, ldu, ldv
      character type
      real *8 x(3,*),w(*)
      real *8 u(ldu,*), v(ldv,*)
      real *8 x1d(n), w1d(n), u1d(n,n), v1d(n,n)
      integer i,j,ipt,itype1d, k, io, jo, ko, ipol
      
      itype1d = 0
      if (itype .ge. 1) then
         itype1d = 1
      endif
      if (itype .ge. 2) then
         itype1d = 2
      endif
      
      if (ipoly.eq.0) call legeexps(itype1d,n,x1d,u1d,v1d,w1d)
      if (ipoly.eq.1) call chebexps(itype1d,n,x1d,u1d,v1d,w1d)

      ipt = 0
      do i=1,n
         do j=1,n
            do k = 1,n
               ipt = ipt + 1
               x(1,ipt) = x1d(k)
               x(2,ipt) = x1d(j)               
               x(3,ipt) = x1d(i)
            enddo
        enddo
      enddo

      if (itype .ge. 1) then
         ipt = 0
         do i=1,n
            do j=1,n
               do k = 1,n
                  ipt = ipt + 1
                  w(ipt) = w1d(i)*w1d(j)*w1d(k) 
               enddo
            enddo
         enddo
      endif


      if (itype .eq. 2 .or. itype .eq. 3) then
c     construct u from 1d u
         if (type .eq. 'f' .or. type .eq. 'F') then         
            ipt = 0
            do io = 1,n
            do jo = 1,n
            do ko = 1,n
               ipt = ipt + 1
               ipol = 0
               do i=1,n
               do j=1,n
               do k = 1,n
                  ipol = ipol + 1
                  u(ipol,ipt) =  u1d(i,io)*u1d(j,jo)*u1d(k,ko)
               enddo
               enddo
               enddo
            enddo
            enddo
            enddo

         else if (type .eq. 't' .or. type .eq. 'T') then

            ipt = 0
            do io = 1,n
            do jo = 1,n
            do ko = 1,n
               ipt = ipt + 1
               ipol = 0
               do i=1,n
               do j=1,n+1-i
               do k = 1,n+2-i-j
                  ipol = ipol + 1
                  u(ipol,ipt) = u1d(i,io)*u1d(j,jo)*u1d(k,ko)
               enddo
               enddo
               enddo
            enddo
            enddo
            enddo
         endif
      endif
      
      if (itype .eq. 2 .or. itype .eq. 4) then
c     construct v from 1d v
         if (type .eq. 'f' .or. type .eq. 'F') then         
            ipol = 0
            do io = 1,n
            do jo = 1,n
            do ko = 1,n
               ipol = ipol + 1
               ipt = 0
               do i=1,n
               do j=1,n
               do k=1,n
                  ipt = ipt + 1
                  v(ipt,ipol) =  v1d(i,io)*v1d(j,jo)*v1d(k,ko)
               enddo
               enddo
               enddo
            enddo
            enddo
            enddo

         else if (type .eq. 't' .or. type .eq. 'T') then

            ipol = 0
            do io = 1,n
            do jo = 1,n+1-io
            do ko = 1,n+2-io-jo
               ipol = ipol + 1
               ipt = 0
               do i=1,n
               do j=1,n
               do k=1,n
                  ipt = ipt + 1
                  v(ipt,ipol) = v1d(i,io)*v1d(j,jo)*v1d(k,ko)
               enddo
               enddo
               enddo
            enddo
            enddo
            enddo
         endif
      endif         

      return
      end
c
c
c
c
      subroutine tens_prod_get_rmask(ndim,iptype,norder,npols,
     1    rmask,rsum)
c
c     get the tail indices for estimating the function error
c
c
c
      implicit real *8 (a-h,o-z)
      integer iind2p(ndim,npols)
      real *8 rmask(npols)

      call polytens_ind2pow(ndim,norder-1,'f',iind2p)

      n=max(iptype,1)
      morder=norder
c     when ndim=1, use the last two coefficients
      if (ndim.eq.1) morder=norder-1
      
      rsum = 0
      do i=1,npols
        rmask(i) = 0.0d0
        i1=0
        do k=1,ndim
cccc           i1=i1+iind2p(k,i)
           i1=i1+(iind2p(k,i)+1)**n
        enddo
cccc        if(i1.eq.norder-1) then
        if(i1 .ge. morder**n) then
          rmask(i) = 1.0d0
          rsum = rsum + 1
        endif
      enddo

      if(iptype.eq.2) rsum = sqrt(rsum)
      if(iptype.eq.0) rsum = 1

      return
      end
c
c
c
c
      

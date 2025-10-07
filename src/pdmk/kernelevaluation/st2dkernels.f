c     This file contains the direct evaluation kernels for 2D Stokeslet
c     and stresslet
c
c     st2ddirectcp: direct calculation of potential for a collection
c                     of Stokeslet sources to a collection of targets
c
c     st2ddirectdp: direct calculation of potential for a collection
c                     of stresslet sources to a collection of targets
c
c     The Stokeslet G_{ij} (without the 1/2pi scaling) is
c
c     G_{ij}(x,y) = (r_i r_j)/(2r^2) - delta_{ij}log(r)/(2)
c
c     The (Type I) stresslet, T_{ijk}, and its associated pressure
c     tensor, PI_{jk}, (without the 1/2pi scaling) are
c     
c     T_{ijk}(x,y) = -2 r_i r_j r_k/ r^4
c
C***********************************************************************
      subroutine st2ddirectcp(ndim,nd,sources,stoklet,ns,targ,nt,
     1            pot,thresh)
c**********************************************************************
c
c     This subroutine evaluates the potential due to a collection
c     of sources and adds to existing
c     quantities.
c
c       pot(x) = pot(x) + sum_m G_{ij}(x,y^{(m)}) sigma^{(m)}_j
c                        j
c                 
c     where sigma^m is the Stokeslet charge. For x a source point,
c     the self-interaction in the sum is omitted.
c
c      If |r| < thresh 
c          then the subroutine does not update the potential
c          (recommended value = |boxsize(0)|*machine precision
c           for boxsize(0) is the size of the computational domain) 
c
c
c-----------------------------------------------------------------------
c     INPUT:
c
c     nd     :    number of charge densities
c     sources:    source locations
C     charge :    charge strengths
C     ns     :    number of sources
c     targ   :    target locations
c     ntarg  :    number of targets
c     thresh :    threshold for updating potential,
c                 potential at target won't be updated if
c                 |t - s| <= thresh, where t is the target
c                 location and, and s is the source location 
c                 
c-----------------------------------------------------------------------
c     OUTPUT:
c     
c     pot    :    updated potential at targ 
c
c-----------------------------------------------------------------------
      implicit none
cf2py intent(in) nd,sources,charge,ns,ztarg,nt,thresh
cf2py intent(out) pot
c
cc      calling sequence variables
c  
      integer ns,nt,nd,ndim
      real *8 sources(ndim,ns),targ(ndim,nt)
      real *8 stoklet(nd,ndim,ns),pot(nd,ndim,nt)
      real *8 thresh
      
c
cc     temporary variables
c
      real *8 diff(ndim),threshsq,r2,pl,dd,d2
      integer i,j,idim,k

      threshsq = thresh**2
      do i=1,nt
         do j=1,ns
            do k=1,ndim
               diff(k)=targ(k,i)-sources(k,j)
            enddo
            r2=diff(1)**2
            do k=2,ndim
               r2=r2+diff(k)**2
            enddo
            
            if(r2.lt.threshsq) goto 1000

            dd = -log(r2)*0.25d0
            d2 = 0.5d0/r2
            
            do idim = 1,nd
               do k=1,ndim
                  pot(idim,k,i) = pot(idim,k,i) +
     1                stoklet(idim,k,j)*dd
               enddo

               pl = diff(1)*stoklet(idim,1,j)
               do k=2,ndim
                  pl = pl + diff(k)*stoklet(idim,k,j)
               enddo
               pl = pl*d2

               do k=1,ndim
                  pot(idim,k,i) = pot(idim,k,i) + diff(k)*pl
               enddo
            enddo
 1000       continue
         enddo
      enddo


      return
      end
c
c
c
c
      subroutine st2ddirectdp(nd,sources,
     1     strslet, strsvec, ns,targ,nt,pot,thresh)
c
c     This subroutine evaluates the potential and gradient due
c     to a collection of stresslet sources and adds
c     to existing quantities (see definitions at top of file).
c
c       pot(x) = pot(x) + sum_m G_{ij}(x,y^{(m)}) sigma^{(m)}_j
c
c     where mu^{(m)} is the
c     stresslet charge, and nu^{(m)} is the stresslet orientation
c     (note that each of these is a 2 vector per source point y^{(m)}).
c     For x a source point, the self-interaction in the sum is omitted. 
c
c
c-----------------------------------------------------------------------
      
c     INPUT:
c     
c     nd in: integer
c        number of densities
c     
c     nsource in: integer  
c        number of sources
c
c     source  in: double precision (2,nsource)
c        source(k,j) is the kth component of the jth
c        source location
c
c     strslet  in: double precision (nd,2,nsource) 
c        stresslet strengths (mu vectors above)
c
c     strsvec  in: double precision (nd,2,nsource)   
c        stresslet orientations (nu vectors above)
c
c     ntarg   in: integer  
c        number of targs 
c
c     targ    in: double precision (2,ntarg)
c        targ(k,j) is the kth component of the jth
c        targ location
c     
c     thresh in: double precision
c        threshold for updating potential,
c        potential at target won't be updated if
c        |t - s| <= thresh, where t is the target
c        location and, and s is the source location 
c
c-----------------------------------------------------------------------
c
c   OUTPUT:
c
c     pot out: double precision(nd,2,ntarg) 
c        velocity at the targets
c      
c------------------------------------------------------------------
      implicit none
cf2py intent(in) nd,sources,ns
cf2py intent(in) targ,nt,thresh
cf2py intent(out) pot

      integer nd, ns, nt
      real *8 sources(2,ns),targ(2,nt)
      real *8 strslet(nd,2,ns),strsvec(nd,2,ns)
      real *8 pot(nd,2,nt)
      real *8 thresh
      
c     local

      real *8 zdiff(2), d1, d2, d3
      real *8 pl, pl2,pv, dmu(2), dnu(2), temp, r, r2, r3, r4, r6
      real *8 dmunu,rtmp      
      real *8 threshsq

      integer i, j, idim, l

      threshsq = thresh**2

c     stresslet contribution
      
      do i = 1,nt
         do j = 1,ns
            zdiff(1) = targ(1,i)-sources(1,j)
            zdiff(2) = targ(2,i)-sources(2,j)

            r2 = zdiff(1)**2 + zdiff(2)**2 
            if (r2 .lt. threshsq) goto 10

            rtmp = zdiff(1)**2 - zdiff(2)**2
            r4 = r2*r2
            
            do idim = 1,nd
               pl = (zdiff(1)*strslet(idim,1,j) +
     1              zdiff(2)*strslet(idim,2,j))

               pl2 = zdiff(1)*strsvec(idim,1,j) + 
     1               zdiff(2)*strsvec(idim,2,j)
               
               pot(idim,1,i) = pot(idim,1,i) - 2*zdiff(1)*pl*pl2/r4
               pot(idim,2,i) = pot(idim,2,i) - 2*zdiff(2)*pl*pl2/r4
            enddo
 10         continue
         enddo
      enddo
      
      return
      end


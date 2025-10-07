c     This file contains the direct evaluation kernels for 3D Stokeslet
c     and Stresslet
c
c      st3ddirectcp: direct calculation of potential for a collection
c                     of stokeslet sources to a collection of targets
c
c      st3ddirectdp: direct calculation of potential for a collection
c                     of stresslet sources to a collection of targets
c
c     The Stokeslet G_{ij} (without the 1/4pi scaling) is
c
c     G_{ij}(x,y) = (r_i r_j)/(2r^3) + delta_{ij}/(2r)
c
c     The (Type I) stresslet, T_{ijk}, and its associated pressure
c     tensor, PI_{jk}, (without the 1/4pi scaling) are
c     
c     T_{ijk}(x,y) = -3 r_i r_j r_k/ r^5
c
C***********************************************************************
      subroutine st3ddirectcp(ndim,nd,sources,stoklet,ns,ztarg,nt,
     1            pot,thresh)
c**********************************************************************
c
c     INPUT:
c
c     nd     :    number of charge densities
c     sources:    source locations
C     ns     :    number of sources
c     ztarg  :    target locations
c     ntarg  :    number of targets
c     thresh :    threshold for updating potential,
c                 potential at target won't be updated if
c                 |t - s| <= thresh, where t is the target
c                 location and, and s is the source location 
c                 
c-----------------------------------------------------------------------
c     OUTPUT:
c     
c     pot    :    updated potential at ztarg 
c
c-----------------------------------------------------------------------
      implicit none
cf2py intent(in) nd,sources,ns,ztarg,nt,thresh
cf2py intent(out) pot
c
cc      calling sequence variables
c  
      integer ns,nt,nd,ndim
      real *8 sources(ndim,ns),ztarg(ndim,nt)
      real *8 stoklet(nd,ndim,ns),pot(nd,ndim,nt)
      real *8 thresh
      
c
cc     temporary variables
c
      real *8 zdiff(ndim),threshsq,r,r2,r3,pl
      integer i,j,idim,k

      threshsq = thresh**2
      do i=1,nt
         do j=1,ns
            do k=1,ndim
               zdiff(k)=ztarg(k,i)-sources(k,j)
            enddo
            r2=zdiff(1)**2
            do k=2,ndim
               r2=r2+zdiff(k)**2
            enddo
            if(r2.lt.threshsq) goto 1000
            r = sqrt(r2)
            r3 = r*r2
            
            do idim = 1,nd
               do k=1,ndim
                  pot(idim,k,i) = pot(idim,k,i) +
     1                stoklet(idim,k,j)/(2*r)
               enddo

               pl = zdiff(1)*stoklet(idim,1,j)
               do k=2,ndim
                  pl = pl + zdiff(k)*stoklet(idim,k,j)
               enddo
               pl = pl/(r3*2)

               do k=1,ndim
                  pot(idim,k,i) = pot(idim,k,i) + zdiff(k)*pl
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
      subroutine st3ddirectdp(nd,sources,
     1     strslet,strsvec,ns,targ,nt,pot,thresh)
cf2py  intent(in) nd
cf2py  intent(in) sources
cf2py  intent(in) strslet,strsvec
cf2py  intent(in) ns
cf2py  intent(in) targ
cf2py  intent(in) nt
cf2py  intent(in) thresh
cf2py  intent(out) pot
c
c     This subroutine evaluates the potential due
c     to a collection of stresslet sources and adds
c     to existing quantities (see definitions at top of file).
c
c       pot(x) = pot(x) + 
c                + sum_m T_{ijk}(x,y^{(m)}) mu^{(m)}_j nu^{(m)}_k
c
c     where mu^{(m)} is the
c     stresslet charge, and nu^{(m)} is the stresslet orientation
c     (note that each of these is a 3 vector per source point y^{(m)}).
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
c     source  in: double precision (3,nsource)
c        source(k,j) is the kth component of the jth
c        source location
c
c     strslet  in: double precision (nd,3,nsource) 
c        stresslet strengths (mu vectors above)
c
c     strsvec  in: double precision (nd,3,nsource)   
c        stresslet orientations (nu vectors above)
c      
c     ntarg   in: integer  
c        number of targs 
c
c     targ    in: double precision (3,ntarg)
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
c     pot out: double precision(nd,3,ntarg) 
c        velocity at the targets
c      
c------------------------------------------------------------------
      implicit none

      integer nd, ns, nt
      real *8 sources(3,ns),targ(3,nt),strslet(nd,3,ns)
      real *8 strsvec(nd,3,ns)
      real *8 pot(nd,3,nt)
      real *8 thresh
      
c     local

      real *8 zdiff(3), d1, d2, d3
      real *8 pl, pv, dmu(3), dnu(3), temp, r, r2, r3, r5
      real *8 dmunu      
      real *8 threshsq

      integer i, j, idim, l

      threshsq = thresh**2


c     type I stresslet
      
      do i = 1,nt
         do j = 1,ns
            zdiff(1) = targ(1,i)-sources(1,j)
            zdiff(2) = targ(2,i)-sources(2,j)
            zdiff(3) = targ(3,i)-sources(3,j)

            r2 = zdiff(1)**2 + zdiff(2)**2 + zdiff(3)**2
            if (r2 .lt. threshsq) goto 20

            r = sqrt(r2)
            r3 = r*r2
            r5 = r3*r2
            
            do idim = 1,nd

               dmu(1) = strslet(idim,1,j)
               dmu(2) = strslet(idim,2,j)
               dmu(3) = strslet(idim,3,j)

               dnu(1) = strsvec(idim,1,j)
               dnu(2) = strsvec(idim,2,j)
               dnu(3) = strsvec(idim,3,j)

               pl = zdiff(1)*dmu(1) + zdiff(2)*dmu(2) + zdiff(3)*dmu(3)
               pv = zdiff(1)*dnu(1) + zdiff(2)*dnu(2) + zdiff(3)*dnu(3)

               temp = -3.0d0*pl*pv/r5
               
               pot(idim,1,i) = pot(idim,1,i) + zdiff(1)*temp
               pot(idim,2,i) = pot(idim,2,i) + zdiff(2)*temp
               pot(idim,3,i) = pot(idim,3,i) + zdiff(3)*temp
            enddo
            
 20         continue
         enddo
      enddo
      
      return
      end
      

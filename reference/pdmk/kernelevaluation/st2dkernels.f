c      This file contains the direct evaluation kernels for 2D Stokeslet
c
c      st2ddirectcp: direct calculation of potential for a collection
c                     of charge sources to a collection of targets
c
c     The Stokeslet G_{ij} (without the 1/2pi scaling) is
c
c     G_{ij}(x,y) = (r_i r_j)/(2r^2) - delta_{ij}log(r)/(2)
c
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

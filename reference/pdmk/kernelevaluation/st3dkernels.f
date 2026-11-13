c      This file contains the direct evaluation kernels for 3D Stokeslet
c
c      st3ddirectcp: direct calculation of potential for a collection
c                     of charge sources to a collection of targets
c
c     The Stokeslet G_{ij} (without the 1/4pi scaling) is
c
c     G_{ij}(x,y) = (r_i r_j)/(2r^3) + delta_{ij}/(2r)
c
c
C***********************************************************************
      subroutine st3ddirectcp(ndim,nd,sources,stoklet,ns,ztarg,nt,
     1            pot,thresh)
c**********************************************************************
c
c     This subroutine evaluates the potential due to a collection
c     of sources and adds to existing
c     quantities.
c
c     pot(x) = pot(x) + sum  q_{j} /|x-x_{j}|^2 
c                        j
c                 
c      where q_{j} is the charge strength
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
cf2py intent(in) nd,sources,charge,ns,ztarg,nt,thresh
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

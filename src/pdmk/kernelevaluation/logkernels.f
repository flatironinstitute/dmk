c      This file contains the direct evaluation kernels for the 2D log kernel
c
c      l2ddirectcp: direct calculation of potential for a collection
c                     of charge sources to a collection of targets
c 
c
c
c
C***********************************************************************
      subroutine logdirectcp(ndim,nd,sources,charge,ns,ztarg,nt,
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
      real *8 charge(nd,ns),pot(nd,nt)
      real *8 thresh
      
c
cc     temporary variables
c
      real *8 zdiff(ndim),dd,d,ztmp,threshsq
      integer i,j,idim,k


      threshsq = thresh**2
      do i=1,nt
         do j=1,ns
            do k=1,ndim
               zdiff(k)=ztarg(k,i)-sources(k,j)
            enddo
c          zdiff(1) = ztarg(1,i)-sources(1,j)
c          zdiff(2) = ztarg(2,i)-sources(2,j)
c          zdiff(3) = ztarg(3,i)-sources(3,j)
            dd=zdiff(1)**2
            do k=2,ndim
               dd=dd+zdiff(k)**2
            enddo
c          dd = zdiff(1)**2 + zdiff(2)**2 + zdiff(3)**2
            if(dd.lt.threshsq) goto 1000

            ztmp = 0.5d0*log(dd)
            do idim=1,nd
               pot(idim,i) = pot(idim,i) + charge(idim,j)*ztmp
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

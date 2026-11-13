c     This file contains the direct evaluation routines for the 3D 
c     Yukawa kernel exp(-kr)/r
c      
c     y3ddirectcp: direct calculation of potential for a collection
c                     of charge sources to a collection of targets
c 
c
c
c
c***********************************************************************
c
c     charge to potential 
c
c**********************************************************************
      subroutine y3ddirectcp_fast(nd,ndigits,rlambda,
     $    d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot)
      implicit none
c**********************************************************************
c
c     This subroutine evaluates the potential due to a collection
c     of sources and adds to existing
c     quantities.
c
c     pot(x) = pot(x) + sum  q_{j} exp(-lambda |x-x_j|)/|x-x_j|
c                        j
c                 
c      where q_{j} is the charge strength
c      If |r| < thresh 
c          then the subroutine does not update the potential
c          (recommended value = |boxsize(0)|*machine precision
c     for boxsize(0) is the size of the computational domain)
c      
      integer ns,nd,ntarg
      integer ndigits
      real *8 rlambda,d2max,eps
      real *8 sources(3,ns),xtarg(*),ytarg(*),ztarg(*)
      real *8 threshsq
      real *8 pot(nd,ntarg)
      real *8 charge(nd,ns)
c
      threshsq = 1.0d-30

      call y3ddirectcp_cpp(nd,ndigits,rlambda,
     1    d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot,threshsq)
      
      return
      end
c
c
c

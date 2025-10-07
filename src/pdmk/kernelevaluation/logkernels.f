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
c
c
c
c
c**********************************************************************
      subroutine logdirectdp(ndim,nd,sources,ns,dipstr,dipvec,
     $           targ,nt,pot,thresh)
cf2py  intent(in) nd
cf2py  intent(in) ns,sources,dipstr,dipvec,targ,nt,thresh
cf2py  intent(out) pot
      implicit none
c**********************************************************************
c
c     This subroutine INCREMENTS the potentials POT
c     at the target points TARGET, due to a vector of
c     charges at SOURCE(2,ns). 
c     We use the unscaled version of log
c     response: i.e., log|z|
c     
c     pot(ii,i)  = \sum_j dipstr(ii,j)* dipstr(ii,:,j) \cdot
c                               \nabla_src \log |targ(:,i)-sources(:,j)|
c
c     The potential is not computed if |r| < thresh
c     (Recommended value for threshold in an FMM is 
c     R*eps, where R is the size of the computation
c     domain and eps is machine precision
c     
c---------------------------------------------------------------------
c     INPUT:
c
c     sources(2,ns) :   location of the sources
c     ns            :   number of sources
c     dipstr(nd,ns) :   dipole strengths
c     dipvec(nd,2,ns) :   dipole orientations
c     targ(2,nt)    :   location of the targets
c     thresh        :   threshold for computing potential
c---------------------------------------------------------------------
c     OUTPUT:
c
c     pot(nd,nt)   (real *8)      : potential is incremented
c---------------------------------------------------------------------
      integer i,ns,ii,nd,j,nt,ndim,k
      real *8 sources(ndim,ns),targ(ndim,nt),rr,r
      real *8 thresh,thresh2,p1,p2
      real *8 pot(nd,nt)
      real *8 dipstr(nd,ns)
      real *8 dipvec(ndim,ns)
      real *8 diff(ndim),dd,dp(ndim)
c
      thresh2 = thresh*thresh
      
      do j = 1,nt
         do i = 1,ns
            do k=1,ndim
               diff(k)=targ(k,j)-sources(k,i)
            enddo
            dd=diff(1)**2
            do k=2,ndim
               dd=dd+diff(k)**2
            enddo
            if(dd.le.thresh2) goto 1000

            do k=1,ndim
               dp(k)=-diff(k)/dd
            enddo

            rr=0
            do k=1,ndim
               rr=rr+dipvec(k,i)*dp(k)
            enddo

            do ii=1,nd
               pot(ii,j) = pot(ii,j) + dipstr(ii,i)*rr
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
c
c

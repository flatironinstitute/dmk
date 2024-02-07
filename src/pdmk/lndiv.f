c     returns the maximum number of points in each leaf box 
c     purely empirical and not extensively tested!
c
c
c
c
c
c
c

      subroutine lndiv(ndim,eps,ns,nt,ifcharge,ifdipole,ifpgh,
     1    ifpghtarg,ndiv,idivflag)
      implicit none
      real *8 eps
      integer ns,nt,ifcharge,ifdipole,ifpgh,ifpghtarg,ndiv
      integer ndim,idivflag

      if (ndim.eq.2) then
         call lndiv2d(eps,ns,nt,ifcharge,ifdipole,ifpgh,
     1       ifpghtarg,ndiv,idivflag)
      elseif (ndim.eq.3) then
         call lndiv3d(eps,ns,nt,ifcharge,ifdipole,ifpgh,
     1       ifpghtarg,ndiv,idivflag)
      endif

      return
      end subroutine
c
c
c
c
c
c
c

      subroutine lndiv2d(eps,ns,nt,ifcharge,ifdipole,ifpgh,
     1   ifpghtarg,ndiv,idivflag)
c
c
c       this subroutine estimates ndiv and idivflag 
c       based on geometry parameters
c       
c
      implicit none
      real *8 eps
      integer ns,nt,ifcharge,ifdipole,ifpgh,ifpghtarg,ndiv
      integer idivflag

      idivflag = 0


       if(eps.ge.0.5d-0) then
          ndiv = 10
       else if(eps.ge.0.5d-1) then
          ndiv = 20
       else if(eps.ge.0.5d-2) then
          ndiv = 20
       else if(eps.ge.0.5d-3) then
          ndiv = 30
       else if(eps.ge.0.5d-6) then
          ndiv = 30
       else if(eps.ge.0.5d-9) then
          ndiv = 45
       else if(eps.ge.0.5d-12) then
          ndiv = 45
       else if(eps.ge.0.5d-15) then
          ndiv = 45
       else
          ndiv = ns+nt
       endif


      return
      end
c
c
c
c
      subroutine lndiv3d(eps,ns,nt,ifcharge,ifdipole,ifpgh,
     1   ifpghtarg,ndiv,idivflag)
c
c
c       this subroutine estimates ndiv and idivflag 
c       based on geometry parameters
c       
c
      implicit none
      real *8 eps
      integer ns,nt,ifcharge,ifdipole,ifpgh,ifpghtarg,ndiv
      integer idivflag

      idivflag = 0


       if(eps.ge.0.5d-0) then
          ndiv = 40
       else if(eps.ge.0.5d-1) then
          ndiv = 40
       else if(eps.ge.0.5d-2) then
          ndiv = 40
       else if(eps.ge.0.5d-3) then
          ndiv = 40
       else if(eps.ge.0.5d-6) then
          ndiv = 80
       else if(eps.ge.0.5d-9) then
          ndiv = 300
       else if(eps.ge.0.5d-12) then
          ndiv = 600
       else if(eps.ge.0.5d-15) then
          ndiv = 600
       else
          ndiv = ns+nt
       endif


      return
      end
      

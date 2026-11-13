c
c
c     this file takes the following conventions
c
c     a tensor grid of points is traversed with x on the inner
c        loop, y on the loop above that, and z above that.
c        e.g. for a 2x2x2 grid, we have the order:
c            (x1,y1,z1), (x2,y1,z1), (x1,y2,z1), (x2,y2,z1)
c     (x1,y1,z2), (x2,y1,z2), (x1,y2,z2), (x2,y2,z2)
c      
c     mesh2d - like matlab's mesh generation function in 3d
c     split2d - separate components of an array of points in 3d
c     lens2d - compute the magnitude of each point in an array of
c                3d points
c     dists2d - compute pairwise distances between the 3d points
c     in two arrays
c      
      subroutine mesh2d(x,nx,y,ny,xy)
      implicit real *8 (a-h,o-z)
      dimension x(*), y(*), xy(2,*)

      ind = 0
      do iy = 1,ny
         do ix = 1,nx
            ind = ind+1
            xy(1,ind) = x(ix)
            xy(2,ind) = y(iy)
         enddo
      enddo

      return
      end

      subroutine split2d(xy,n,x,y)
      implicit real *8 (a-h,o-z)
      dimension x(*), y(*), xy(2,*)

      do i = 1,n
         x(i) = xy(1,i)
         y(i) = xy(2,i)
      enddo

      return
      end

      subroutine lens2d(xy,n,dlens)
      implicit real*8 (a-h,o-z)
      dimension xy(2,*), dlens(*)

      do i = 1,n
         dlens(i) = 0.0d0
         do j = 1,2
            dlens(i) = dlens(i) + xy(j,i)**2
         enddo
         dlens(i) = sqrt(dlens(i))
      enddo

      return
      end

      subroutine dists2d(a,n,b,m,dists)
      implicit real*8 (a-h,o-z)
      dimension a(2,*), b(2,*), dists(n,m)

      do i = 1,m
         do j = 1,n
            dists(j,i) = 0
            do k = 1,2
               dists(j,i) = dists(j,i) + (a(k,j)-b(k,i))**2
            enddo
            dists(j,i) = sqrt(dists(j,i))
         enddo
      enddo

      return
      end


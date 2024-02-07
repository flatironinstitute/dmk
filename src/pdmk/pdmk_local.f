c     This file contains nine direction evaluation subroutines
c     for the DMK framework in two and three dimensions.
c
c     Note: all output variables are incremented. So proper initialization are needed, but
c     the subroutines can be called many times to calculate the output variable due to all
c     sources.
c
c     The union of input and output arguments are as follows.
c
c     Input parameters:
c     nd: number of input vectors (charge, rnormal, dipstr) and output vectors (pot, grad, hess)
c     dim: dimension of the underlying space
c     bssrc: box size of the source box
c     d2max: the squared distance outside which the local kernel is regarded as 0
c     ns: number of sources
c     sources: (dim,ns) source coordinates
c     ntarg: number of targets
c     targ: (dim,ntarg) target coordinates
c     charge: (nd,ns) charge strengths
c     rnormal: (dim,ns) dipole orientation vectors
c     dipstr: (nd,ns) dipole strengths
c
c     Output parameters:
c     pot: (nd,ntarg) incremented potential at targets
c     grad: (nd,dim,ntarg) incremented gradient at targets
c     hess: (nd,dim*(dim+1)/2,ntarg) incremented hessian at targets
c***********************************************************************
c
c     charge to potential for the local PSWF kernel of 1/r
c
c**********************************************************************
      subroutine l3d_local_kernel_directcp(nd,dim,ndigits,rscale,
     $    center,d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot)
      implicit none
c**********************************************************************
      integer i,ns,ii,nd,itarg,k,ntarg
      integer dim,ndigits
      real *8 bssrcinv,d2max,eps
      real *8 sources(dim,ns),xtarg(*),ytarg(*),ztarg(*)
      real *8 dr(10)
      real *8 rtmp,fval,rscale,rr,rr2,threshsq,center
      real *8 pot(nd,ntarg)
      real *8 charge(nd,ns)
c
      threshsq = 1.0d-30

      call l3d_local_kernel_directcp_cpp(nd,dim,ndigits,rscale,center,
     1    d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot,threshsq)
      
      return
      end
c     
c
c
c**********************************************************************
      subroutine l3d_near_kernel_directcp(nd,dim,ndigits,rscale,center,
     $    bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot)
      implicit none
c**********************************************************************
      integer i,ns,ii,nd,itarg,k,ntarg
      integer dim,ndigits
      real *8 bsizeinv,d2max,eps
      real *8 sources(dim,ns),xtarg(*),ytarg(*),ztarg(*)
      real *8 dr(10)
      real *8 rtmp,fval,rscale,rr,rr2,threshsq,center
      real *8 pot(nd,ntarg)
      real *8 charge(nd,ns)
c

      call l3d_near_kernel_directcp_cpp(nd,dim,ndigits,rscale,center,
     1    bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot)
      
      return
      end
c     
c
c
c
c***********************************************************************
c
c     charge to potential for the local PSWF kernel of 1/r^2
c
c**********************************************************************
      subroutine sl3d_local_kernel_directcp(nd,dim,ndigits,rscale,
     $    center,d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot)
      implicit none
c**********************************************************************
      integer i,ns,ii,nd,itarg,k,ntarg
      integer dim,ndigits
      real *8 bssrcinv,d2max,eps
      real *8 sources(dim,ns),xtarg(*),ytarg(*),ztarg(*)
      real *8 dr(10)
      real *8 rtmp,fval,rscale,rr,rr2,threshsq,center
      real *8 pot(nd,ntarg)
      real *8 charge(nd,ns)
c
      threshsq = 1.0d-30

      call sl3d_local_kernel_directcp_cpp(nd,dim,ndigits,rscale,center,
     1    d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot,threshsq)
      
      return
      end
c     
c
c
c**********************************************************************
      subroutine sl3d_near_kernel_directcp(nd,dim,ndigits,rscale,center,
     $    bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot)
      implicit none
c**********************************************************************
      integer i,ns,ii,nd,itarg,k,ntarg
      integer dim,ndigits
      real *8 bsizeinv,d2max,eps
      real *8 sources(dim,ns),xtarg(*),ytarg(*),ztarg(*)
      real *8 dr(10)
      real *8 rtmp,fval,rscale,rr,rr2,threshsq,center
      real *8 pot(nd,ntarg)
      real *8 charge(nd,ns)
c

      call sl3d_near_kernel_directcp_cpp(nd,dim,ndigits,rscale,center,
     1    bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot)
      
      return
      end
c     
c
c
c
c***********************************************************************
c
c     charge to potential for the local PSWF kernel of log(r)
c
c**********************************************************************
      subroutine log_local_kernel_directcp(nd,dim,ndigits,rscale,
     $    center,d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot)
      implicit none
c**********************************************************************
      integer i,ns,ii,nd,itarg,k,ntarg
      integer dim,ndigits
      real *8 bssrcinv,d2max,eps
      real *8 sources(dim,ns),xtarg(*),ytarg(*),ztarg(*)
      real *8 dr(10)
      real *8 rtmp,fval,rscale,rr,rr2,threshsq,center
      real *8 pot(nd,ntarg)
      real *8 charge(nd,ns)
c
      threshsq = 1.0d-30

      call log_local_kernel_directcp_cpp(nd,dim,ndigits,rscale,center,
     1    d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot,threshsq)
      
      return
      end
c     
c
c
c**********************************************************************
      subroutine log_near_kernel_directcp(nd,dim,ndigits,rscale,center,
     $    bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot)
      implicit none
c**********************************************************************
      integer i,ns,ii,nd,itarg,k,ntarg
      integer dim,ndigits
      real *8 bsizeinv,d2max,eps
      real *8 sources(dim,ns),xtarg(*),ytarg(*),ztarg(*)
      real *8 dr(10)
      real *8 rtmp,fval,rscale,rr,rr2,threshsq,center
      real *8 pot(nd,ntarg)
      real *8 charge(nd,ns)
c

      call log_near_kernel_directcp_cpp(nd,dim,ndigits,rscale,center,
     1    bsizeinv,d2max,sources,ns,charge,xtarg,ytarg,ztarg,ntarg,pot)
      
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
c***********************************************************************
c
c     charge to potential for the 2d Yukawa local kernel
c     The 2d Yukawa kernel is K_0(rlambda r)
c
c**********************************************************************
      subroutine y2d_local_kernel_directcp(nd,dim,rlambda,
     $    rscale,center,d2max,sources,ns,charge,
     2    xtarg,ytarg,ntarg,ncoefs,coefs,pot)
      implicit none
c**********************************************************************
      integer i,j,ns,ind,nd,itarg,k,ntarg
      integer dim,ndigits,ncoefs

      real *8 coefs(*),rlambda
      real *8 bsize,d2max,eps
      real *8 sources(dim,ns),xtarg(*),ytarg(*)
      real *8 dr(10),xval,dd,dkval
      real *8 rtmp,fval,rscale,r,rr2,threshsq,center
      real *8 pot(nd,ntarg)
      real *8 charge(nd,ns)
      real *8 besk0
      external besk0
c
      threshsq = 1.0d-30

      do i=1,ntarg
         do j=1,ns
            dr(1) = xtarg(i)-sources(1,j)
            dr(2) = ytarg(i)-sources(2,j)

            dd = dr(1)*dr(1)+dr(2)*dr(2)

            if ((dd.lt.threshsq).or.(dd.gt.d2max)) goto 1000

            r = sqrt(dd)
            xval=r*rscale+center

c           smooth part of the local kernel
            call chebexev(xval,fval,coefs,ncoefs)
c           value of the original kernel
            dkval=besk0(rlambda*r)
            do ind=1,nd
               pot(ind,i)=pot(ind,i)+charge(ind,j)*(dkval+fval)
            enddo
 1000       continue
         enddo
      enddo
            
      return
      end
c     
c
c
c**********************************************************************
      subroutine y2d_near_kernel_directcp(nd,dim,rlambda,
     $    rscale,center,d2max,sources,ns,charge,
     2    xtarg,ytarg,ntarg,ncoefs,coefs,pot)
      implicit none
c**********************************************************************
      integer i,j,ind,ns,ii,nd,itarg,k,ntarg
      integer dim,ndigits,ncoefs
      real *8 rlambda,coefs(*),bsize,d2max,eps
      real *8 sources(dim,ns),xtarg(*),ytarg(*)
      real *8 dr(10),dd,xval
      real *8 rtmp,fval,rscale,rr,rr2,threshsq,center
      real *8 pot(nd,ntarg)
      real *8 charge(nd,ns)
c
      do i=1,ntarg
         do j=1,ns
            dr(1) = xtarg(i)-sources(1,j)
            dr(2) = ytarg(i)-sources(2,j)
            dd = dr(1)*dr(1)+dr(2)*dr(2)

            if (dd.gt.d2max) goto 1000
c           normalize r^2 to [-1,1]
            xval=dd*rscale+center
c           value of the local kernel
            call chebexev(xval,fval,coefs,ncoefs)

            do ind=1,nd
               pot(ind,i)=pot(ind,i)+charge(ind,j)*fval
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
c***********************************************************************
c
c     charge to potential for the 3d Yukawa local kernel
c     The 3d Yukawa kernel is exp(-lambda r)/r
c
c**********************************************************************
      subroutine y3d_local_kernel_directcp(nd,dim,rlambda,
     $    rscale,center,d2max,sources,ns,charge,
     2    xtarg,ytarg,ztarg,ntarg,ncoefs,coefs,pot)
      implicit none
c**********************************************************************
      integer i,j,ns,ind,nd,itarg,k,ntarg
      integer dim,ndigits,ncoefs
      real *8 coefs(*),rlambda
      real *8 bsize,d2max,eps
      real *8 sources(dim,ns),xtarg(*),ytarg(*),ztarg(*)
      real *8 dr(10),xval,dd,dkval
      real *8 rtmp,fval,rscale,r,rr2,threshsq,center
      real *8 pot(nd,ntarg)
      real *8 charge(nd,ns)
c
      threshsq = 1.0d-30
      
      do i=1,ntarg
         do j=1,ns
            dr(1) = xtarg(i)-sources(1,j)
            dr(2) = ytarg(i)-sources(2,j)
            dr(3) = ztarg(i)-sources(3,j)
            dd = dr(1)*dr(1)+dr(2)*dr(2)+dr(3)*dr(3)

            if ((dd.lt.threshsq).or.(dd.gt.d2max)) goto 1000

            r = sqrt(dd)
            xval=r*rscale+center
c           smooth part of the local kernel
            call chebexev(xval,fval,coefs,ncoefs)
c           value of the original kernel
            dkval=exp(-rlambda*r)/r
            do ind=1,nd
               pot(ind,i)=pot(ind,i)+charge(ind,j)*(dkval+fval)
cccc               pot(ind,i)=pot(ind,i)+charge(ind,j)*fval
            enddo
 1000       continue
         enddo
      enddo
            
      return
      end
c     
c
c
c**********************************************************************
      subroutine y3d_near_kernel_directcp(nd,dim,rlambda,
     $    rscale,center,d2max,sources,ns,charge,
     2    xtarg,ytarg,ztarg,ntarg,ncoefs,coefs,pot)
      implicit none
c**********************************************************************
      integer i,j,ind,ns,ii,nd,itarg,k,ntarg
      integer dim,ndigits,ncoefs
      real *8 rlambda,coefs(*),bsize,d2max,eps
      real *8 sources(dim,ns),xtarg(*),ytarg(*),ztarg(*)
      real *8 dr(10),dd,xval
      real *8 rtmp,fval,rscale,rr,rr2,threshsq,center
      real *8 pot(nd,ntarg)
      real *8 charge(nd,ns)
c
      do i=1,ntarg
         do j=1,ns
            dr(1) = xtarg(i)-sources(1,j)
            dr(2) = ytarg(i)-sources(2,j)
            dr(3) = ztarg(i)-sources(3,j)
            dd = dr(1)*dr(1)+dr(2)*dr(2)+dr(3)*dr(3)

            if (dd.gt.d2max) goto 1000
c           normalize r^2 to [-1,1]
            xval=dd*rscale+center
c           value of the local kernel
            call chebexev(xval,fval,coefs,ncoefs)

            do ind=1,nd
               pot(ind,i)=pot(ind,i)+charge(ind,j)*fval
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

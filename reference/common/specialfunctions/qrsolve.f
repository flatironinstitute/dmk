cccc        implicit real *8 (a-h,o-z)
cccc        dimension a(10 000 000),b(10 000 000),a2(10 000 000)
ccccc 
cccc        call prini(6,13)
ccccC 
ccccC       SET ALL PARAMETERS
ccccC 
cccc         PRINT *, 'ENTER n'
cccc         READ *,n
cccc         CALL PRINF('n=*',n,1 )
ccccc 
cccc        call testit(a,n,b,a2)
cccc  
ccccc 
cccc        stop
cccc        end
ccccc 
ccccc 
ccccc 
ccccc 
ccccc 
cccc        subroutine testit(a,n,b,a2)
cccc        implicit real *8 (a-h,o-z)
cccc        save
cccc        real *8 a(n,n),b(n,n),y2(10 000),w(10 000),
cccc     1      x(10 000),y(10 000),x2(10 000),a2(1),diffs(10 000),
cccc     2      xs(10 000),xs7(10 000),y7(10 000)
ccccc 
ccccc 
ccccc        create the test matrix
ccccc 
cccc        do 1400 i=1,n
cccc        do 1200 j=1,n
ccccc 
cccc        a(j,i)=i**2-j
cccc        a(j,i)=0
cccc 1200 continue
cccc  
cccc        a(i,i)=a(i,i)+i**3
cccc        a(i,i)=i+1
cccc 1400 continue
cccc  
cccc        call arrmove(a,a2,n*n*2)
cccc  
cccc  
cccc  
cccc        call prin2('test matrix as created *',a,n*n)
ccccc 
ccccc       construct its QR decomposition
ccccc 
cccc  
cccc        t1=clotatim()
cccc  
cccc        call qrdecom(a,n,b,rcond)
cccc  
cccc  
cccc        t2=clotatim()
cccc  
cccc        call prin2('cpu time for qrdecom=*',t2-t1,1)
cccc        call prin2('after qrdecom, rcond=*',rcond,1)
cccc
cccc
cccc  
ccccc 
ccccc       construct the test vector
ccccc 
cccc        done=1
ccccc 
cccc        do 2200 i=1,n
ccccc 
cccc        x(i)=done/(i**2+1)
cccc 2200 continue
cccc  
cccc        call qrmatvec(a,n,x,y)
cccc  
cccc        call prin2('x as created*',x,n)
cccc        call prin2('y as created*',y,n)
cccc  
cccc        call arrmove(y,y2,n)
cccc  
cccc  
cccc        call qrsolve(n,b,y,x2)
cccc  
cccc  
cccc        call prin2('after qrsolve, x2=*',x2,n)
cccc  
cccc        do 2400 i=1,n
ccccc 
cccc        diffs(i)=x(i)-x2(i)
cccc 2400 continue
cccc  
cccc        call prin2('and x2-x=*',diffs,n)
ccccc 
ccccc       now, solve the linear system using qrsolv
ccccc 
ccccc 
ccccc       test the solver for the adjoint system
ccccc 
ccccc 
ccccc        . . . construct a test vector
ccccc 
cccc        do 3200 i=1,n
ccccc 
cccc        xs7(i)=i
cccc 3200 continue
cccc  
cccc        call prin2('xs7 as created*',xs7,n)
ccccc 
ccccc        apply the putative (a^*)^{-1} to xs7
ccccc 
cccc        call qrsolve_adj(n,b,xs7,x)
cccc  
cccc        call prin2('after qrsolve_adj, x=*',x,n)
ccccc 
ccccc 
ccccc        apply A^* to x
ccccc 
cccc        call qrmatavec(a,n,x,y7)
cccc  
cccc        call prin2('and y7=*',y7,n)
ccccc 
ccccc        estimate the condotion number of the matrix a
ccccc 
cccc        call qrcond(n,a,b,rcond,w)
ccccc 
cccc        call prin2('after qrcond, rcond=*',rcond,1)
cccc  
cccc  
cccc        stop
cccc        end
cccc  
cccc  
ccccc 
ccccc 
ccccc 
ccccc 
ccccc 
cccc        subroutine arrmove(x,y,n)
cccc        implicit real *8 (a-h,o-z)
ccccc 
cccc        save
cccc        dimension x(1),y(1)
ccccc 
cccc        do 1200 i=1,n
ccccc 
cccc        y(i)=x(i)
cccc 1200 continue
ccccc 
cccc        return
cccc        end



c 
c 
c 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c        This is the end of the debugging code and the beginning of the
c        linear solver proper
c 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c 
c 
c        This file contains six user-callable subroutines: qrsolv,
c        qrdecom, qrsolve, qrsolve_adj, qrcond. Following is a brief
c        description of these subroutines.
c 
c 
c   qrsolv - uses a version of QR-decomposition to solve the user-supplied
c       system of linear algebraic equations, with the matrix a and
c       right-hand side rhs. Both the matrix a and the right-hand side rhs
c       are destroyed in the process. This is a primitive, short, and
c       reasonably efficient subroutine. IMPORTANT NOTE: THIS IS A
c       PRIMITIVE ROUTINE IN THAT IT SOLVES THE SYSTEM WITH A SINGLE
c       RIGHT-HAND SIDE, AND WILL PERFORM THE QR DECOMPOSITION AGAIN AND
c       AGAIN FOR EACH NEW RIGHT-HAND SIDE. IT SHOULD NOT BE USED IN THIS
c       REGIME! Instead, the subroutines qrdecom, qrsolve (see) should be
c       used! On the other hand, for a single right-hand side, this routine
c       is about twice more efficient than the subroutine qrdecom.
c 
c   qrdecom - subroutine constructs a QR-decomposition of the real
c       user-supplied matrix a. It is expected that this decomposition
c       will be used by the subroutine qrsolve (see) for the solution
c       of linear systems with the matrix a; this subroutine has no
c       known uses as a stand-alone device.
c 
c   qrsolve  - uses a version of QR-decomposition to solve the
c       user-supplied system of linear algebraic equations
c 
c              A(x)=y.                                                (1)
c 
c       Hopefully, the QR-decomposition has been prepared by a
c       prior call to the subroutine qrdecom (see); this subroutine
c       has no known uses as a stand-alone device.
c 
c   qrsolve_c  - uses a version of QR-decomposition to solve the
c       user-supplied system of linear algebraic equations
c 
c              A(x)=y                                                 (2)
c 
c       with a REAL *8 MATRIX AND A COMPLEX *8 RIGHT-HAND SIDE.
C       Hopefully, the QR-decomposition has been prepared by a
c       prior call to the subroutine qrdecom (see); this subroutine
c       has no known uses as a stand-alone device.
c 
c   qrsolve_adj - a companion to the subroutine qrsolve, using the
c       same QR-decomposition to solve the user-supplied system of
c       linear algebraic equations
c 
c              A^* (x)=y.                                             (3)
c 
c       The QR-decomposition is prepared by a prior call to the
c       subroutine qrdecom (see); this subroutine has no known
c       uses as a stand-alone device.
c 
c   qrcond - the QR-decomposition obtained via a preceding call
c       to the subroutine qrdecom to estimate thye condition number
c       of the matrix a. This estimate is enormously more accurate
c       than that produced by the original call to qrdecom.
c 
c 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c 
c 
c 
        subroutine qrsolv(a,n,rhs,rcond)
        implicit real *8 (a-h,o-z)
        save
        real *8 a(n,n),u(2,2),aa(2),rhs(n)
c 
c  This subroutine uses a version of QR-decomposition to solve
c       the user-supplied system of linear algebraic equations, with the
c       matrix a and right-hand side rhs. Both the matrix a and the right
c       -hand side rhs are destroyed in the process.
c 
c       IMPORTANT NOTE:
C        THIS IS A PRIMITIVE ROUTINE IN THAT IT SOLVES THE SYSTEM
C        WITH A SINGLE RIGHT-HAND SIDE, AND WILL PERFORM THE
C        QR DECOMPOSITION AGAIN AND AGAIN FOR EACH NEW RIGHT-HAND
C        SIDE. IT SHOULD NOT BE USED IN THIS REGIME! Instead, the
c        subroutines qrdecom, qrsolve (see) should be used
c 
c                    Input parameters:
c 
c  n - the dimensionality of the system being solved
c  b - the array containing the QR-decomposition of the matrix of the
c       system (hopefully, it has been produced by a prior call to the
c       subroutine qrdecom)
c  y - the right-hand side of the system to be solved; not damaged by this
c       subroutine in any way
c 
c                    Output parameters:
c 
c  x - the solution of the system
c  rcond - a fairly crude estimate of the condition number of a
c 
c 
c        . . . transpose the input matrix a
c 
        size22=0
        do 1400 i=1,n
        do 1200 j=1,i
c 
        d=a(j,i)
        a(j,i)=a(i,j)
        a(i,j)=d
c 
        size22=size22+a(j,i)**2
        size22=size22+a(i,j)**2
 1200 continue
 1400 continue
c 
c       eliminate the upper right triangle
c 
        do 2000 i=1,n-1
c 
        do 1600 j=n,i+1,-1
c 
        aa(1)=a(i,j-1)
        aa(2)=a(i,j)
        call qrrotfnd2(aa,u,size22)
c 
        call qrrotate(u,a(1,j-1),a(1,j),n,i)
c 
        d1=u(1,1)*rhs(j-1)+u(1,2)*rhs(j)
        d2=u(2,1)*rhs(j-1)+u(2,2)*rhs(j)
        rhs(j-1)=d1
        rhs(j)=d2
c 
 1600 continue
 2000 continue
c 
c       estimate the condition number
c 
        dmax=-1
        dmin=abs(a(1,1))
        do 2200 i=1,n
c 
        if(dmax .lt. abs(a(i,i)) ) dmax=abs(a(i,i))
        if(dmin .gt. abs(a(i,i)) ) dmin=abs(a(i,i))
 2200 continue
c 
        if(dmin .eq. 0) then
            rcond=-1
            return
        endif
c 
        rcond=dmax/dmin
c 
        call qrtrin(a,rhs,n)
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine qrsolve(n,b,y,x)
        implicit real *8 (a-h,o-z)
        save
        real *8 b(1),x(1),y(1)
c 
c       This subroutine uses a version of QR-decomposition to solve
c       the user-supplied system of linear algebraic equations. The
c       QR-decomposition is prepared by a prior call to the subroutine
c       qrdecom (see); this subroutine has no known uses as a stand-alone
c       device.
c 
c                    Input parameters:
c 
c  n - the dimensionality of the system being solved
c  b - the array containing the QR-decomposition of the matrix of the
c       system (hopefully, it has been produced by a prior call to the
c       subroutine qrdecom)
c  y - the right-hand side of the system to be solved; not damaged by this
c       subroutine in any way
c 
c                    Output parameters:
c 
c  x - the solution of the system
c 
c 
c        . . . construct the memory map
c 
        ia=1
        la=n*n+2
c 
        ib=ia+la
        lb=n*n+2
c 
        iz=ib+lb
        lz=n+2
c 
c       . . . apply the inverse of the factored matrix to the
c             right-hand side
c 
        call qrtrinv(b(ia),y,b(iz),n)
        call qrmatvec(b(ib),n,b(iz),x)
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine qrsolve_adj(n,b,y,x)
        implicit real *8 (a-h,o-z)
        save
        real *8 b(1),x(1),y(1)
c 
c       This subroutine uses a version of QR-decomposition to solve
c       the user-supplied system of linear algebraic equations
c 
c                   A^* (x)=y.                                         (1)
c 
c       Please note that A^* in (1) above denotes the adjoint of
c       the  matrix A. Also, please note that the QR-decomposition
c       is prepared by a prior call to the subroutine qrdecom (see);
c       this subroutine has no known uses as a stand-alone device.
c       Also, please note that this subroutine is the companion to
c       the subroutine qrsolve (see), solving the system of linear
c       equations
c 
c                   A(x)=y.                                            (2)
c 
c                    Input parameters:
c 
c  n - the dimensionality of the system being solved
c  b - the array containing the QR-decomposition of the matrix of the
c       system (hopefully, it has been produced by a prior call to the
c       subroutine qrdecom)
c  y - the right-hand side of the system to be solved; not damaged by this
c       subroutine in any way
c 
c                    Output parameters:
c 
c  x - the solution of the system
c 
c 
c        . . . construct the memory map
c 
        ia=1
        la=n*n+2
c 
        ib=ia+la
        lb=n*n+2
c 
c        apply the inverse of a to y
c 
        call qrmatavec(b(ib),n,y,x)
c 
        call qrtrin(b(ia),x,n)
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine qrdecom(a,n,b,rcond)
        implicit real *8 (a-h,o-z)
        save
        real *8 a(1),b(1)
c 
c       This subroutine constructs a QR-decomposition of the real
c       user-supplied matrix a. It is expected that this decomposition
c       will be used by the subroutine qrsolve (see) for the solution
c       of linear systems with the matrix a; this subroutine has no
c       known uses as a stand-alone device.
c 
c                    Input parameters:
c 
c  a - the matrix to be QR-decomposed; not damaged by the subroutine
c       in any way
c  n - the dimensionality of the matrix a
c 
c                    Output parameters:
c 
c  b - the array containing the QR-decomposition of a; must be
c       2*n**2+n+10 real *8 locations long
c  rcond - a fairly crude estimate of the condition number of a
c 
c        . . . allocate memory
c 
        ia=1
        la=n*n+2
c 
        ib=ia+la
        lb=n*n+2
c 
        do 1200 i=1,n*n
c 
        b(i)=a(i)
 1200 continue
c 
c       construct the QR-decomposition
c 
        call qrelim(b,n,b(ib),rcond)
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine qrcond(n,a,b,rcond,w)
        implicit real *8 (a-h,o-z)
        save
        real *8 a(n,n),b(n,n),w(1)
c 
c       This subroutine uses a version of QR-decomposition
c       (obtained via a preceding call to the subroutine qrdecom)
c       to estimate thye condition number of the matrix a
c 
c                      Input parameters:
c 
c  n - the dimensionality of the problem
c  a - the matrix whose condition number; must be the same as that
c       from which the subroutine qrdecom has produced the paramater
c       b in the calling sequence of this subroutine
c  b - the factorization of the matrix a (hopefully) produced via
c       a preceding call to the subroutine qrdecom (see)
c 
c                      Output parameters:
c 
c  rcond - a (fairly good) estimate of the condition number of the
c       matrix a
c 
c                      Work arrays:
c 
c  w - must be at least 2*n+2 real *8 locations in length
c 
c 
c        . . . initialize the array to be used for forward iteration
c 
        call qrsolve_rand(n,w)
        call qrsolve_rand(n,w)
c 
c       estimate the maximum singular value of the matrix a
c 
        numit=10
        do 1800 i=1,numit
c 
        iy=n+1
        call qrmatvec(a,n,w,w(iy))
        call qrmatavec(a,n,w(iy),w)
c 
        d=0
        do 1200 j=1,n
c 
        d=d+w(j)**2
 1200 continue
c 
        d=sqrt(d)
c 
        do 1400 j=1,n
c 
        w(j)=w(j)/d
 1400 continue
c 
 1800 continue
c 
 2000 continue
c 
        rlammax=sqrt(d)
c 
c       estimate the minimum singular value of the matrix a
c 
        call qrsolve_rand(n,w)
c 
        do 2800 i=1,numit
c 
        iy=n+1
c 
        call qrsolve(n,b,w,w(iy))
        call qrsolve_adj(n,b,w(iy),w)
c 
        d=0
        do 2200 j=1,n
c 
        d=d+w(j)**2
 2200 continue
c 
        d=sqrt(d)
        do 2400 j=1,n
c 
        w(j)=w(j)/d
 2400 continue
c 
 2800 continue
c 
        rlammin=1/sqrt(d)
        rcond=rlammax/rlammin
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine qrtrinv(a,y,x,n)
        implicit real *8 (a-h,o-z)
        save
        real *8 a(n,n),x(1),y(1)
c 
c       start the process
c 
        x(1)=y(1)/a(1,1)
        do 1400 i=2,n
c 
        d=0
        do 1200 j=1,i-1
c 
        d=d+a(i,j)*x(j)
 1200 continue
c 
        x(i)=(y(i)-d)/a(i,i)
 1400 continue
        return
        end
c 
c 
c 
c 
c 
        subroutine qrtrin(a,y,n)
        implicit real *8 (a-h,o-z)
        save
        real *8 a(n,n),y(1)
c 
c       apply the inverse of the trizngular matrix a to y
c 
        y(n)=y(n)/a(n,n)
        do 1400 i=n-1,1,-1
c 
        d=0
        do 1200 j=n,i+1,-1
c 
        d=d+a(j,i)*y(j)
 1200 continue
c 
        y(i)=(y(i)-d)/a(i,i)
 1400 continue
        return
        end
c 
c 
c 
c 
c 
        subroutine qrelim(a,n,b,rcond)
        implicit real *8 (a-h,o-z)
        save
        real *8 a(n,n),u(2,2),aa(2),b(n,n)
c 
c       construct the unity matrix
c 
        size22=0
        do 1400 i=1,n
        do 1200 j=1,n
c 
        b(j,i)=0
        size22=size22+a(j,i)**2
 1200 continue
c 
        b(i,i)=1
 1400 continue
c 
c       eliminate the upper right triangle
c 
        do 2000 i=1,n-1
c 
        do 1600 j=n,i+1,-1
c 
        aa(1)=a(i,j-1)
        aa(2)=a(i,j)
        call qrrotfnd2(aa,u,size22)
c 
        call qrrotate(u,a(1,j-1),a(1,j),n,i)
c 
        ii=j-i
        call qrrotate(u,b(1,j-1),b(1,j),n,ii)
 1600 continue
 2000 continue
c 
c       estimate the condition number
c 
        dmax=-1
        dmin=abs(a(1,1))
        do 2200 i=1,n
c 
        if(dmax .lt. abs(a(i,i)) ) dmax=abs(a(i,i))
        if(dmin .gt. abs(a(i,i)) ) dmin=abs(a(i,i))
 2200 continue
c 
        if(dmin .eq. 0) then
            rcond=-1
            return
        endif
c 
        rcond=dmax/dmin
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine qrrotate(a,x,y,n,n0)
        implicit real *8 (a-h,o-z)
        save
        real *8 a(2,2),x(1),y(1)
c 
        do 1200 i=n0,n
c 
        d1=a(1,1)*x(i)+a(1,2)*y(i)
        d2=a(2,1)*x(i)+a(2,2)*y(i)
c 
        x(i)=d1
        y(i)=d2
 1200 continue
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine qrrotfnd2(a,u,size22)
        implicit real *8 (a-h,o-z)
        save
        real *8 a(2),u(2,2)
c 
        u22=-a(1)
        u12=a(2)
c 
        d=u22**2+u12**2
c 
        if(d .lt. size22*1.0d-66) then
c 
            u(2,2)=1
            u(1,2)=0
            u(1,1)=1
            u(2,1)=0
            return
        endif
c 
        d=sqrt(d)
c 
        u(2,2)=u22/d
        u(1,2)=u12/d
        u(1,1)=-u(2,2)
        u(2,1)=u(1,2)
        return
        end
c 
c 
c 
c 
c 
        subroutine qrmatvec(a,n,x,y)
        implicit real *8 (a-h,o-z)
        save
        real *8 a(n,n),x(1),y(1),cd
c 
c        apply the matrix a to the vector x obtaining y
c 
        do 1400 i=1,n
        cd=0
        do 1200 j=1,n
        cd=cd+a(i,j)*x(j)
 1200 continue
        y(i)=cd
 1400 continue
        return
        end
c 
c 
c 
c 
c 
        subroutine qrmatavec(a,n,x,y)
        implicit real *8 (a-h,o-z)
        save
        real *8 a(n,n),x(1),y(1),cd
c 
c        apply the matrix a to the vector x obtaining y
c 
        do 1400 i=1,n
        cd=0
        do 1200 j=1,n
cccc        cd=cd+a(i,j)*x(j)
        cd=cd+a(j,i)*x(j)
 1200 continue
        y(i)=cd
 1400 continue
        return
        end
  
  
c 
c 
c 
c 
c 
        subroutine qrsolve_rand(n,y)
        implicit real *8 (a-h,o-z)
        save
        dimension y(1)
        data ifcalled/0/
c 
c       generate pseudo-random numbers
c 
        if(ifcalled .ne. 0) goto 1100
c 
        done=1
        pi=atan(done)*4
        add=sqrt(7*done)
        x=1.1010101010101
        ifcalled=1
 1100 continue
c 
        do 1200 i=1,n
c 
        phi=x*100000*pi+add
        j=phi
        phi=(phi-j)
        x=phi
c 
        y(i)=x
 1200 continue
c 
        return
c 
c 
c 
c 
        entry qrsolve_rand_reset(x7)
c 
        done=1
        pi=atan(done)*4
        add=sqrt(7*done)
        x=x7
        ifcalled=1
        return
        end
c 
c 
c 
c 
c 
        subroutine qrsolve_c(n,b,y,x)
        implicit real *8 (a-h,o-z)
        save
        real *8 b(1)
        complex *16 x(1),y(1)
c 
c       This subroutine uses a version of QR-decomposition to solve
c       the user-supplied system of linear algebraic equations with
c       REAL *8 MATRIX AND COMPLEX *16 RIGHT-HAND SIDE. The 
c       QR-decomposition is prepared by a prior call to the subroutine
c       qrdecom (see); this subroutine has no known uses as a stand-alone
c       device.
c 
c                    Input parameters:
c 
c  n - the dimensionality of the system being solved
c  b - the array containing the QR-decomposition of the matrix of the
c       system (hopefully, it has been produced by a prior call to the
c       subroutine qrdecom)
c  y - the right-hand side of the system to be solved; not damaged by this
c       subroutine in any way
c 
c                    Output parameters:
c 
c  x - the solution of the system
c 
c 
c        . . . construct the memory map
c 
        ia=1
        la=n*n+2
c 
        ib=ia+la
        lb=n*n+2
c 
        iz=ib+lb
        lz=n+2
c 
c       . . . apply the inverse of the factored matrix to the
c             right-hand side
c 
        call qrtrinv_c(b(ia),y,b(iz),n)
        call qrmatvec_c(b(ib),n,b(iz),x)
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine qrmatvec_c(a,n,x,y)
        implicit real *8 (a-h,o-z)
        save
        real *8 a(n,n)
        complex *16 x(1),y(1),cd
c 
c        apply the matrix a to the vector x obtaining y
c 
        do 1400 i=1,n
        cd=0
        do 1200 j=1,n
        cd=cd+a(i,j)*x(j)
 1200 continue
        y(i)=cd
 1400 continue
        return
        end
c 
c 
c 
c 
c 
        subroutine qrtrinv_c(a,y,x,n)
        implicit real *8 (a-h,o-z)
        save
        real *8 a(n,n)
        complex *16 x(1),y(1),d
c 
c       start the process
c 
        x(1)=y(1)/a(1,1)
        do 1400 i=2,n
c 
        d=0
        do 1200 j=1,i-1
c 
        d=d+a(i,j)*x(j)
 1200 continue
c 
        x(i)=(y(i)-d)/a(i,i)
 1400 continue
        return
        end  

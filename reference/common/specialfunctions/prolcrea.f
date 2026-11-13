cccc        implicit real *8 (a-h,o-z)
cccc        real *8 ts(10 000),whts(10 000)
ccccc 
cccc        real *8 w(1000 000),khi(10 000),
cccc     1      rlams2(10 000),fs(10 000),ders(10 000),
cccc     2      fs2(10000),ders2(10000)
ccccc 
cccc        real *8 errs(10 000),errsder(10 000),
cccc     1      fs3(10 000),errs3(10 000),rats(10000)
ccccc
cccc        complex *16 rlams(10 000)     
ccccc 
cccc        call prini(6,13)
ccccC 
ccccC       SET ALL PARAMETERS
ccccC 
cccc        PRINT *, 'ENTER c'
cccc        READ *,c
cccc        CALL PRIN2('c=*',c,1 )
ccccC 
cccc        PRINT *, 'ENTER k'
cccc        READ *,k
cccc        CALL PRINf('k=*',k,1 )
ccccc 
ccccc        construct the discretization of the interval [-1,1]
ccccc 
cccc        itype=1
cccc        nn=1000
cccc        call legeexps(itype,nn,ts,u,v,whts)
ccccc 
ccccc       construct prolate functions using the new version
ccccc       of the subroutine
ccccc 
cccc        lenw=1000 000
cccc        call prolcrea(ier,c,w,lenw,nvects,nhigh,
cccc     1    khi,rlams,rlams2,keep,lused)
cccc  
cccc        call prin2('after prolcrea khi are*',khi,nvects)
cccc        call prin2('after prolcrea, rlams2=*',rlams2,nvects)
cccc        call prin2('after prolcrea, rlams=*',rlams,nvects*2)
ccccc 
cccc        call prinf('ier=*',ier,1)
cccc        call prinf('nhigh=*',nhigh,1)
cccc        call prinf('keep=*',keep,1)
cccc        call prinf('lused=*',lused,1)
cccc        call prinf('nvects=*',nvects,1)
cccc
cccc
cccc        stop
cccc
cccc 1800 format(2x,i3,2x,E22.16)
cccc        write(6,1800) (i-1,khi(i),i=1,nvects)
cccc        write(13,1800) (i-1,khi(i),i=1,nvects)
cccc
cccc        stop
cccc
cccc
cccc
ccccc
cccc        do 2100 i=1,nvects
ccccc
cccc        fs2(i)=rlams2(i)-1
cccc
cccc        ts(i)=i
cccc
cccc        if(rlams2(i) .lt. 1.0d-10) rlams2(i)=0
cccc
cccc 2100 continue
ccccc
cccc        call prin2('and fs2=*',fs2,nvects)
cccc
cccc
cccc        iw=21
cccc        call quagraph(iw,ts,rlams2,nvects,2,'rlams2*')
cccc
cccc
cccc
cccc        stop
cccc
cccc
cccc
cccc  
cccc
cccc
ccccc 
ccccc       use the obtained data to evaluate the prolate function at a
ccccc       bunch of nodes
ccccc 
ccccc 
cccc        do 2200 i=1,nn
ccccc 
cccc        call proleva(k,ts(i),w,fs(i),ders(i) )
ccccc 
cccc 2200 continue
ccccc 
cccc        iw=21
cccccccc        call lotagraph(iw,ts,fs,nn,'the prolate wave function*')
cccc
cccc        call quagraph(iw,ts,fs,nn,3,'the prolate wave function*')
cccc
cccc
cccc         call prin2('rlams2(k)=*',rlams2(k),1)
cccc
cccc         call prin2('rlams(k)=*',rlams(k),2)
cccc         call prin2('rlams(k/2)=*',rlams(k/2),2)
cccc         call prin2('rlams(k)/rlams(k/2)=*',rlams(k)/rlams(k/2),2)
cccc
cccc
cccc        stop
ccccc 
ccccc       evaluate the prolate function using the faster version of
ccccc       proleva (known as proleval), and compare the results to those
ccccc       obrained with proleva
ccccc 
cccc        do 2400 i=1,nn
ccccc 
cccc        call proleval(k,ts(i),w,fs2(i),ders2(i) )
cccc  
cccc        errs(i)=fs2(i)-fs(i)
cccc        errsder(i)=ders2(i)-ders(i)
cccc  
cccc        rats(i)=fs2(i)/fs(i)
cccc 2400 continue
ccccc 
cccc        call prin2('now, fs2=*',fs2,nn)
cccc        call prin2('and fs=*',fs,nn)
cccc  
cccc  
cccc        call prin2('and errs=*',errs,nn)
cccccccc        call prin2('and rats=*',rats,nn)
cccc  
cccc  
cccc  
cccc  
ccccccccc        stop
cccc  
cccc  
cccc        call prin2('and errsder=*',errsder,nn)
cccc  
cccc  
ccccc 
ccccc       evaluate the prolate function using the faster version of
ccccc       proleva that does not calculat derivatives (known as prolevv),
ccccc       and compare the results to those obrained with proleva
ccccc 
cccc        do 2600 i=1,nn
ccccc 
cccc        call prolevv(k,ts(i),w,fs3(i) )
cccc  
cccc        errs3(i)=fs3(i)-fs(i)
cccc 2600 continue
ccccc 
cccc        call prin2('and fs3 as calculated via prolevv are*',
cccc     1      fs3,nn)
cccc        call prin2('and errors in fs3 are*',errs3,nn)
cccc  
cccc  
cccc        eps=0.0001
cccc        eps=1.0e-15
cccc        do 2800 i=1,nvects
cccc        if(rlams2(i) .gt. eps**2) num=i-1
cccc 2800 continue
cccc  
cccc  
cccc  
cccc        call prinf('and num=*',num,1)
cccc        call prin2('and rlams2(k)=*',rlams2(k),1)
cccc        call prin2('and rlams2(k)-1=*',rlams2(k)-1,1)
cccc
cccc        zero=0
cccc        call prolevv(k,zero,w,fzero)
cccc
cccc        call prin2('and fzero=*',fzero,1)
cccc
cccc        stop
cccc        end
c 
c 
  
c 
c 
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c 
c      this is the end of the debugging code and the beginning of
c      the code for the actual evaluation of the prolate spheroidal
c      wave functions
c 
c 
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c 
c      This file contains a collection of subroutines for the
c      handling (evaluation, etc) of prolate spheroidal wave functions.
c      There are 9 subroutines in it that are user callable. Following
c      is a brief summary of such subroutines.
c 
c 
c   prolcrea - initializes the evaluation of prolate functions; has
c      to be invoked prior to calls to any other subroutines here,
c      except prol0ini, prol0eva, provcget, proicget (see below)
c 
c   proleval - evaluates the user-specified prolate function at the
c      user-specifeid point on the interval [-1,1]; also returnes
c      the derivative of the prolatefunction
c 
c   prolevv - evaluates the user-specified prolate function at the
c      user-specifeid point on the interval [-1,1]; it is different
c      from proleval in that it does NOT compute the derivative, and
c      is about twice faster than proleval
c 
c   proleva - evaluates the user-specified prolate function at the
c      user-specifeid point on the interval [-1,1]; also returnes
c      the derivative of the prolatefunction. It is an earlier version
c      of proleval (see above); it is simpler than proleval, and about
c      three times slower.
c 
c   prolunpk - returns to the user the Legendre expansion
c        of the k-th prolate spheroidal wave function \psi_k,
c        valid for x on the interval [-1,1]
c 
c 
c   prol0ini - precomputes the array w to be used by the
c       subroutine prol0eva (see) to evaluate the function \psi^c_0
c       (prolate function number zero) and its derivative at arbitrary
c       points in R^1. PLEASE NOTE THAT THIS SUBROUTINE IS INDEPENDENT
C       OF PROLCREA, BEING A MUCH SIMPLIFIED AND ACCELERATED VERSION
C       OF THE LATTER. THE OUTPUT OF THIS SUBROUTINE IS USED
C       BY THE SUBROUTINE PROL0EVA BELOW.
c 
c   prol0eva - evaluates the function \psi^c_0 and its
c        derivative at the user-specified point x \in R^1. The
c        subroutine uses the array w that must have been precomputed
c        by a preceding call to the subroutine prol0ini (see above)
c 
c   provcget - for a user-specifeid small number eps, this subroutine
c       finds the value of the parameter c such that
c 
c       \psi_0^c(1)) = eps.                                   (1)
c 
c       In other words, the subroutine finds such c that the value of
c       the prolate function \psi_0^c at 1 is equal to eps.
c 
c   proicget - for a user-specified small number eps, this subroutine
c       finds the value of the parameter c such that
c 
c       \int_{-\infty}^{+\infty} (\psi_0^c(x))^2 dx -1 =eps^2.   (1)
c 
c       In other words, the subroutine finds such c that the L^2 norm
c       of the prolate function \psi_0^c on \R^2 \[-1,1] is equal to eps.
c 
c   prolnofc - finds an integer nout such that
c 
c             rlams2_c(nout) \sim 10^{-logeps},                    (1)
c 
c        with logeps, c two user-specified parameters. The intended
c        use of this subroutine is for finding "the dimensionality"
c        of the space of band-limited functions on the interval [-1,1],
c        with the band-limit c and precision 10^{-logeps/2}.
c 
c   prolcofn - finds a real c such that
c 
c             rlams2_c(n) \sim 10^{-logeps},                    (1)
c 
c        with logeps, n two user-specified parameters. The intended
c        use of this subroutine is for finding "the dimensionality"
c        of the space of band-limited functions on the interval [-1,1],
c        with the band-limit c and precision 10^{-logeps/2}; all of its
c        uses so far have been as an inverse of the subroutine prolnofc
c        (see).
c 
c   protoleg - converts a prolate expansion of a real function
c        on the interval [-1,1] into a Legendre expansion. It uses
c        as input the array w, constructed by a prior call to the
c        subroutine prolcrea (see). This subroutine has no use as a
c        stand-alone device.
c 
c   cprotole - converts a prolate expansion of a complex function on
c        the interval [-1,1] into a Legendre expansion. It uses as
c        input the array w, constructed by a prior call to the
c        subroutine prolcrea (see). This subroutine has no use as a
c        stand-alone device.
c 
c   prolexps - constructs the matrix u, converting the coefficients of
c       a prolate expansion on the interval [-1,1] into the values of
c       the said expansion at prolate nodes, and the matrix v, converting
c       the values of a prolate expansion at the prolate nodes on the
c       interval [-1,1] into the coefficients of the said expansion. It
c       also produces the prolate nodes and corresponding weights
c   prolc18 - given the user-specified integer 0 < ndigits < 19,
c       this subroutine returns the value c such that
c 
c              \psi_0^c (1)=0.1^{ndigits};
c 
c       this is one of two subroutines (the other being prolc180)
c       that are supposed to replace the subroutine provcget (see)
c   prolc180 - given the user-specified real 1.0E-18 < eps 0.1,
c       this subroutine returns the value c such that
c 
c              \psi_0^c (1) \sim eps;
c       this is one of two subroutines (the other being prolc180)
c       that are supposed to replace the subroutine provcget (see)
c 
c 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c 
        subroutine prolcrea(ier,c,w,lenw,nvects,nhigh,
     1    khi,rlams,rlams2,keep,lused)
        implicit real *8 (a-h,o-z)
        save
        real *8 khi(1),rlams2(1),w(1)
        complex *16 rlams(1),ima
        dimension ns(20),nvectss(20)
c 
        data ima/(0.0d0,1.0d0)/
        data nvectss/25,34,43,50,58,65,72,80,87,94,100,107,114,121,
     1       128,134,141,148,154,161/
        data ns/48,64,80,92,106,120,130,144,156,168,
     1       178,190,202,214,224,236,248,258,268,280/
c 
c        this subroutine constructs the eigenfunctions of the
c        operator
c 
c         F(\phi) (x) = \int_{-1}^1  \phi (t) * e^{i * c *x * t) dt    (1)
c 
c        corresponding to all eigenvalues of (1) whose absolute values
c        are greater than 1.0d-15 (or so). It also produces the eigenvalues
c        of (1) (all of them whose absolute values are greater than
c        1.0d-15), in decreasing order. In addition, it produces all
c        eigenvalues of the operator
c 
c 
c 
c         G(\phi) (x) = (1 / \pi) *  \int_{-1}^1  \phi (t) *
c                                                                      (2)
c         sin(c*(x-t))/(x-t) dt,
c 
c         also in decreasing order. The actual evaluation of the
c         eigenfunctions of (1) (which are also the eigenfunctions
c         of (2)) is performed by the subroutine prolev0 (see);
c         prolev0 uses Legendre expansions of these eigenfunctions
c         created by this subroutine and stored in array w, in a form
c         not readily accessible by the user.
c 
c         The subroutine uses the fact that the eigenfunctions of the
c         operator (1) are also the eigenfunctions of the Sturm-Liouville
c         problem
c 
c      D               D
c      -  ( (1-x**2) * -  \phi (x) ) + (khi-c**2*x**2) * \phi(x) = 0.    (3)
c      Dx              Dx
c 
c        Note that the equation (3) defines the prolate speheroidal
c        wave functions. The coefficients khi are the so-called
c        separation coefficients; these are also calculated by this
c        subroutine and returned to the user. The only other
c        user-accesible subroutine in this collection are prolunpk
c        and proleva (see both below). Proleva evaluates at a point
c        x \in [-1,1] the user-specified prolate function and its
c        derivative. Prolunpk returns to the user the coefficients
c        of the Legendre expansion of the user-specified prolate
c        function.
c 
c                          input parameters:
c 
c  c - the coefficient (real) in the formula (1)
c  lenw - the amount of storage space provided in the array lw; must
c       be sufficiently large; if it is too small, the error code ier
c       is set to 4, and the execution of the subroutine is terminated.
c       Note that this parameter, normally, does not need to be very large.
c 
c                          output parameters:
c 
c  ier - error return code.
c         ier=0 means successful execution.
c         ier=4 and ier=8 means that the amount of storage space provided in
c                 the array store is insufficient (see input
c                 parameter lenw above). This is a fatal error.
c         ier=1024 means that the subroutine prolql1 (a standard
c                 prehistoric subroutine from eispack) has failed
c                 to find the spectrum of a certain tridiagonal matrix;
c                 this has never happened yet, and probably means that memory
c                 has been mismanaged by the user. This is a fatal error.
c  w - the storage area where all the information is stored to be
c       used by the subroutine prolev0 to evaluate the prolate spheroidal
c       wave functions and their derivatives
c  nvects - the number of solutions of the equation (1) whose Legendre
c       expansions have been computed
c  nhigh - the highest order of the Legendre expansion used to express
c       any of the nvects prolate functions.
c  khi - the coefficients in (2) (nvects of them) at which the equation
c       (3) has a non-trivial solution (the "separation coefficients"
c       for the prolate spheroidal wave function).
c  rlams - the eigenvalues (complex) of the operator (1)
c  rlams2 - the eigenvalues of the operator (2)
c  keep - the number of elements of the array w that has to be
c       unchanged between the call to this subroutine and the subsequent
c       calls to the subroutine prolev0 (see)
c 
c  lused - the number of elements of array w that have been used by
c       this subroutine (it is somewhat greater than keep)
c 
c 
c   IMPORTANT (SOMETIMES) NOTE: THE FIRST 20 ELEMENTS OF ARRAY W ARE
C       USED BY THE SUBROUTINE TO STORE CERTAIN CONTROL TABLES; THE
C       ELEMENTS 101 THROUGH KEEP ARE USED FOR OTHER STORAGE NEEDS.
C       THE ELEMENTS 21 THROUGH 100 ARE NOT USED, AND CAN BE UTILIZED
C       BY THE USER TO TRANSMIT INFORMATION TO OTHER SUBROUTINES USING
C       OTHER PARTS OF W.
c 
c 
c        . . . find out how many essentially non-zero eigenvalues
c              correspond to the user-specified c, and how many
c              Legendre polynomials to choose in the expansions of
c              the eigenvectors
c 
         eps=1.0d-15
         n=c*3
         n=n/2
         nvects=c
c 
         i=c/10
         if(i .le. 19) n=ns(i+1)
         if(i .le. 19) nvects=nvectss(i+1)
  
         n=n+30
         nvects=nvects+30
  
cccc         call prinf('n as determined in prolcrea is*',n,1)
cccc         call prinf('nvects as determined in prolcrea is*',nvects,1)
c 
c        allocate memory for the subroutine prolvect that
c        obtains coefficients of Legendre expansions of prolate
c        spheroidal wave functions
c 
        istart=101
c 
        ias=istart
        las=n/2+6
c 
        ibs=ias+las
        lbs=n/2+6
c 
        ics=ibs+lbs
        lcs=n/2+6
c 
        ixk=ics+lcs
        lxk=n/2+6
c 
        iu=ixk+lxk
        lu=n/2+6
c 
        iv=iu+lu
        lv=n/2+6
c 
        iw=iv+lv
        lw=n/2+6
c 
        iladdr=iw+lw
        lladdr=nvects*4
c 
        istore=iladdr+lladdr
c 
        lleft=lenw-istore
c 
        if(lleft .gt. 200) goto 1400
        ier=8
        return
 1400 continue
c 
c       obtain the coefficients of Legendre expansions
c       for the first nvects prolate spheroidal wave functions
c 
      call prolvect(ier,n,c,w(ias),w(ibs),w(ics),
     1     w(ixk),khi,nvects,w(istore),w(iladdr),nstore,eps,
     2     w(iu),w(iv),w(iw),lleft,nhigh,iscale,lneed)
c 
      lused=istore+lneed
c 
c 
c        perform garbage collection
c 
        iladdr2=istart
        lladdr=nvects*4
c 
        call prolarrm(w(iladdr),w(iladdr2),lladdr)
c 
        iladdr=iladdr2
c 
        istore2=iladdr+lladdr
        lstore=nstore+4
c 
        call prolarrm(w(istore),w(istore2),lstore)
c 
        istore=istore2
c 
        iwork=istore+lstore
        lwork=nhigh*2+10
  
        keep=iwork+lwork
  
        call prinf('in prolcrea, keep =*',keep,1)
c 
c       store varous types of integer data in the beginning of array w
c 
        w(1)=iladdr+0.1
        w(2)=istore+0.1
        w(3)=istore+iscale-1+0.1
        w(4)=nhigh+0.1
        w(5)=iwork+0.1
c 
c        allocate memory for the subroutine prollam0 that will
c        find the spectrum of the integral operator
c 
        nn=n+10
        ipexp=keep+1
        lpexp=nn+4
c 
        ix=ipexp+lpexp
        lx=nn+2
c 
        iwhts=ix+lx
        lwhts=nn+2
c 
c        calculate the largest eigenvalue of the integral operator
c 
c         F(\phi) (x) = \int_{-1}^1  \phi (t) * e^{i * c *x * t) dt
c 
        call prollam0(nn,c,w(istore),w(iladdr),
     1      w(ipexp),w(ix),w(iwhts),nhigh,rlam0,w)
  
  
        call prin2('after prollam0, rlam0=*',rlam0,1)
  
c 
c       find the absolute values of first nvects eigenvalues
c       of the integral operator
c 
c         F(\phi) (x) = \int_{-1}^1  \phi (t) * e^{i * c *x * t) dt
c 
c 
        irats77=keep+10
        lrats77=nvects+5
c 
        icoefsk=irats77+lrats77
        lcoefsk=nn
c 
        icoefskp1=icoefsk+lcoefsk
        lcoefskp1=nn
c 
        iderk=icoefskp1+lcoefskp1
        lderk=nn
c 
        iderkp1=iderk+lderk
        lderkp1=nn
c 
        iscales=iderkp1+lderkp1
        lscales=nhigh+10
c 
        ltot=iscales+lscales
        if(ltot .gt. lused) lused=ltot
        if(lused .lt. lenw) goto 3200
        ier=4
        return
 3200 continue
c 
        call prolrat(c,w(istore),w(iladdr),nvects,rlam0,
     1      rlams2,nhigh,w(irats77),
     2      w(icoefsk),w(icoefskp1),w(iderk),w(iderkp1),w(iscales) )
c 
c       find the first nvects eigenvalues of the integral operator
c 
c         F(\phi) (x) = \int_{-1}^1  \phi (t) * e^{i * c *x * t) dt
c 
        do 3400 i=1,nvects
c 
        rlams(i)=ima**(i-1) * rlams2(i)
 3400 continue
c 
c       calculate the largest nvects eigenvalues of the integral operator
c 
c 
c         G(\phi) (x) = \int_{-1}^1  \phi (t) *
c 
c         sin(c*(x-t))/(x-t) dt
c 
        done=1
        pi=atan(done)*4
c 
        do 3600 i=1,nvects
        d = rlams2(i) **2
        rlams2(i)=d/pi/2 *c
 3600 continue
  
c 
c       create and store in array w the scaling data to be used by
c       the subroutine proleval for the (relatively)
c       rapid evaluation of prolate functions and their derivatives
c 
        ninit=nhigh+6
c 
        ipjc1=keep+6
        lpjc1=ninit+6
  
        lpjc1=lpjc1*2
c 
c 
        ipjc2=ipjc1+lpjc1
        lpjc2=ninit+6
  
        lpjc2=lpjc2*2
  
c 
        w(11)=ninit+0.1
        w(12)=ipjc1 +0.1
        w(13)=ipjc2 +0.1
c 
        d=0
        nn2=10
        call legeFDE2(d,dd,ddd,w(istore),Nn2,
     1    w(ipjc1),w(ipjc2),ninit)
c 
c       create and store in array w the scaling data to be used by
c       the subroutine prolevv for the (relatively)
c       rapid evaluation of prolate functions
c 
        ninit26=ninit/2+6
c 
        icpnm1=ipjc2+lpjc2
        lcpnm1=ninit26
c 
        icpnp1=icpnm1+lcpnm1
        lcpnp1=ninit26
c 
        icxpnp1=icpnp1+lcpnp1
        lcxpnp1=ninit26
c 
        icpnm1o=icxpnp1+lcxpnp1
        lcpnm1o=ninit26
c 
        icpnp1o=icpnm1o+lcpnm1o
        lcpnp1o=ninit26
c 
        icxpnp1o=icpnp1o+lcpnp1o
        lcxpnp1o=ninit26
c 
        w(14)=icpnm1 +0.1
        w(15)=icpnp1 +0.1
        w(16)=icxpnp1 +0.1
c 
        w(17)=icpnm1o +0.1
        w(18)=icpnp1o +0.1
        w(19)=icxpnp1o +0.1
c 
        call legeevev(d,nn2,w(istore),dd,ninit,
     1      w(icpnm1),w(icpnp1),w(icxpnp1) )
  
        call legeodev(d,nn2,w(istore),dd,ninit,
     1      w(icpnm1o),w(icpnp1o),w(icxpnp1o) )
c 
        keep=icxpnp1o+lcxpnp1o
c 
        nvects=nvects-1
c 
        if(lused .lt. keep) lused=keep
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prolevv(k,x,w,val)
        implicit real *8 (a-h,o-z)
        save
        dimension w(1)
c 
c 
c        this subroutine evaluates the k-th prolate spheroidal
c        wave function \psi_k (to be referred to simply as prolate
c        function) at the point x. Note that the subroutine is stable
c        for x on the interval [-1,1]. It works in the small neighborhood
c        of the integral [-1,1], but its accuracy deteriorates rapidly
c        as |x| becomes grater than 1. The precision with which the \psi_k
c        is calculated is abiout 15 digits.
c 
c 
c  NOTE: THIS SUBROUTINE IS IDENTICAL TO PROLEVAL (see), EXCEPT it does
c        not calculate the derivative of the prolate function, while
c        proleval does. In addition, this subroutine is about twice
c        faster than proleval.
c 
c                  input parameters:
c 
c  k - the order of the prolate function to be evaluated. Note that k
c        must be between zero and nvects, where nvects has been returned
c        by a prior call to the subroutine prolcrea (see).
c  x - the point where the prolate function is to be evaluated
c  w - the array that has been created by a prior call to the subroutine
c        prolcrea (see)
c 
c                  output parameters:
c 
c  val - the value of the k-th prolate function at the point x
c 
c        . . . interpret the first few elements of array w
c              containing its map
c 
        iladdr=w(1)
        istore=w(2)
        iscale=w(3)
        nhigh=w(4)
        iwork=w(5)
c 
        icpnm1=w(14)
        icpnp1=w(15)
        icxpnp1=w(16)
c 
        icpnm1o=w(17)
        icpnp1o=w(18)
        icxpnp1o=w(19)
c 
        ninit0=0
c 
        call prolevv0(k,x,w(istore),w(iladdr),val,w(iwork),
     1      w(iscale),ninit0,w(icpnm1),w(icpnp1),w(icxpnp1),
     2      w(icpnm1o),w(icpnp1o),w(icxpnp1o) )
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine proleval(k,x,w,val,der)
        implicit real *8 (a-h,o-z)
        save
        dimension w(1)
c 
c        this subroutine evaluates the k-th prolate spheroidal
c        wave function \psi_k (to be referred to simply as prolate
c        function) at the point x; it also calculates the derivative
c        \psi_k at the point x. Note that the subroutine is stable
c        for x on the interval [-1,1]. It works in the small neighborhood
c        of the integral [-1,1], but its accuracy deteriorates rapidly
c        as |x| becomes grater than 1. The precision with which the \psi_k
c        is calculated is abiout 15 digits.
c 
c  NOTE: THIS SUBROUTINE REPLACES THE SUBROUTINE PROLEVA (SEE),
C        THE CALLING SEQUENCES OF THESE SUBROUTINES ARE IDENTICAL,
C        BUT THIS ONE IS ABOUT THREE TIMES FASTER THAT PROLEVA;
C        PROLEVA IS LEFT IN EXISTENCE FOR PURPOSES OF COMPATIBILITY,
C        AND ALSO FOR TESDTING AND DEBUGGING PURPOSES, SHOULD THIS
C        ONE HAVE A BUG. - 11.1.99
c 
c                  input parameters:
c 
c  k - the order of the prolate function to be evaluated. Note that k
c        must be between zero and nvects, where nvects has been returned
c        by a prior call to the subroutine prolcrea (see).
c  x - the point where the prolate function is to be evaluated
c  w - the array that has been created by a prior call to the subroutine
c        prolcrea (see)
c 
c                  output parameters:
c 
c  val - the value of the k-th prolate function at the point x
c  der - derivative of the k-th prolate function at the point x
c 
c 
c        . . . interpret the first few elements of array w
c              containing its map
c 
        iladdr=w(1)
        istore=w(2)
        iscale=w(3)
        nhigh=w(4)
        iwork=w(5)
c 
        ipjc1=w(12)
        ipjc2=w(13)
c 
        ninit0=0
c 
c       evaluate the k-th prolate function and its derivative at
c       the point x
c 
        call proleva0(k,x,w(istore),w(iladdr),val,der,w(iwork),
     1      nhigh,w(iscale),w(ipjc1),w(ipjc2),ninit0)
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine proleva(k,x,w,val,der)
        implicit real *8 (a-h,o-z)
        save
        dimension w(1)
c 
c        this subroutine evaluates the k-th prolate spheroidal
c        wave function \psi_k (to be referred to simply as prolate
c        function) at the point x; it also calculates the derivative
c        \psi_k at the point x. Note that the subroutine is stable
c        for x on the interval [-1,1]. It works in the small neighborhood
c        of the integral [-1,1], but its accuracy deteriorates rapidly
c        as |x| becomes grater than 1. The precision with which the \psi_k
c        is calculated is abiout 15 digits.
c 
c 
c  NOTE: THIS SUBROUTINE HAS BEEN REPLACED BY THE SUBROUTINE PROLEVAL
C        (SEE), THE CALLING SEQUENCES OF THESE SUBROUTINES ARE IDENTICAL,
C        BUT PROLEVAL IS ABOUT THREE TIMES FASTER THAT PROLEVA;
C        PROLEVA IS LEFT IN EXISTENCE FOR PURPOSES OF COMPATIBILITY,
C        AND ALSO FOR TESDTING AND DEBUGGING PURPOSES, SHOULD PROLEVAL
C        HAVE A BUG. - 11.1.99
c 
c                  input parameters:
c 
c  k - the order of the prolate function to be evaluated. Note that k
c        must be between zero and nvects, where nvects has been returned
c        by a prior call to the subroutine prolcrea (see).
c  x - the point where the prolate function is to be evaluated
c  w - the array that has been created by a prior call to the subroutine
c        prolcrea (see)
c 
c                  output parameters:
c 
c  val - the value of the k-th prolate function at the point x
c  der - derivative of the k-th prolate function at the point x
c 
c 
c        . . . interpret the first few elements of array w containing its map
c 
        iladdr=w(1)
        istore=w(2)
        nhigh=w(4)
        iscale=w(3)
        iwork=w(5)
c 
c       evaluate the k-th prolate function and its derivative at
c       the point x
c 
        call prolev0(k,x,w(istore),w(iladdr),val,der,w(iwork),
     1      nhigh,w(iscale) )
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prolunpk(k,w,coefs,n)
        implicit real *8 (a-h,o-z)
        save
        dimension w(1),coefs(1)
c 
c        this subroutine returns to the user the Legendre expansion
c        of the k-th prolate spheroidal wave function \psi_k (to be
c        referred to simply as prolate function). Note that the
c        expansion is valid for x on the interval [-1,1]. It works in
c        the small neighborhood  of the integral [-1,1], but its accuracy
c        deteriorates rapidly  as |x| becomes grater than 1. Also, note
c        that the expansion (with the n terms returned) is accurate to
c        about 15 digits.
c 
c        IMPORTANT NOTE:
c 
c   The coefficients of the expansion returned by this subroutine
c        correspond to a somewhat unusual normalization of the
c        Legendre polynomials. Specifically, they are scaled so that
c        the norm of each polynomial on the interval [-1,1] is equal
c        to 1.
c 
c                  input parameters:
c 
c  k - the order of the prolate function the coefficients of whose
c        legendre series are to be evaluated. Note that k must be between
c        zero and nvects, where nvects has been returned by a prior call
c        to the subroutine prolcrea (see).
c  w - the array that has been created by a prior call to the subroutine
c        prolcrea (see)
c 
c                  output parameters:
c 
c  coefs - the first n coefficients in the Legendre expansion of the k-th
c         prolate function.
c  n - the number of terms returned in the array coefs
c 
c        . . . interpret the first few wlwmwnts of array w containing its map
c 
        iladdr=w(1)
        istore=w(2)
        iscale=w(3)
        nhigh=w(4)
        iwork=w(5)
c 
c       retrieve from array w the Legendre coefficients of the
c       k-th prolate function
c 
        call prolunp0(k,coefs,w(istore),w(iladdr),n,w(iwork))
c 
c       scale the array of Legendre coefficients to make
c       it properly normalized
c 
        do 1400 i=1,n
        coefs(i)=coefs(i)*w(iscale+i-1)
 1400 continue
        return
        end
c 
c 
c 
c 
c 
        subroutine prol0eva(x,w,psi0,derpsi0)
        implicit real *8 (a-h,o-z)
        save
        dimension w(1)
c 
c        This subroutine evaluates the function \psi^c_0 and its
c        derivative at the user-specified point x \in R^1. The
c        subroutine uses the array w that must have been precomputed
c        by a preceding call to the subroutine prol0ini (see).
c 
c                         Input parameters:
c 
c  x - the point at which \psi^c_0 and its derivative are to be
c        calculated
c  w - the array with varous types of information stored there
c        by a preceding call to the subroutine prol0ini (see).
c 
c                         Output parameters:
c 
c     psi0 - the value \psi^c_0(x)
c     derpsi0 - the value (\psi^c_0)'(x)
c 
c 
c        . . . evaluate the prolate function \psi^c_0 (x) in
c              the case when x \in [-1,1]
c 
        iw=w(1)
        its=w(2)
        iwhts=w(3)
        ifs=w(4)
c 
        nterms=w(5)
        ngauss=w(6)
        rlam=w(7)
        c=w(8)
        thresh=w(9)
c 
        if(abs(x) .gt. 1) goto 3000
c 
        call legeFDER(X,psi0,derpsi0,w(iw),Nterms-2)
c 
        return
c 
 3000 continue
c 
        if(c .lt. thresh-1.0d-10) goto 3200
        psi0=0
        derpsi0=0
        return
c 
 3200 continue
c 
c        evaluate the prolate function \psi^c_0 (x) in
c        the case when x is outside [-1,1]
c 
        call prosinin(c,w(its),w(iwhts),w(ifs),x,ngauss,
     1      psi0,derpsi0)
c 
        psi0=psi0/rlam
        derpsi0=derpsi0/rlam
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prol0ini(ier,c,w,rlam20,rkhi,lenw,keep,ltot)
        implicit real *8 (a-h,o-z)
        save
        dimension w(1)
c 
c       this subroutine precomputes the array w to be used by the
c       subroutine prol0eva (see) to evaluate the function \psi^c_0
c       and its derivative at arbitrary points in R^1. Here, \psi^c_0
c       is the first eigenvector of the integral opertor
c 
c         G(\phi) (x) = \int_{-1}^1  \phi (t) *
c         sin(c*(x-t))/(x-t) dt                                      (1)
c 
c       Please notre that this subroutine has no function as a
c       stand-alone device.
c 
c                         Input parameters:
c 
c  c - the parameter "c" in \psi^c_0
c  lenw - the length of the usr-supplied array w, in real *8 locations;
c       used by the subroutine to bomb if the length is insufficient.
c 
c                         Output parameters:
c 
c  ier - error return code.
c          ier=0 means successful execution
c          ier=512 means that the length of the user-provided array w
c                   is insufficient for the user-specified parameter c.
c                   the length should be increased
c          ier=1024 means that the length of the user-provided array w
c                   is insufficient for the user-specified parameter c.
c                   however, the subroutine discoverde this fact at a
c                   strange point. Something is fishy.
c          ier=2048 means serious trouble (a bug in the code?)
c  w - the array containing all sorts of data to be used by the
c       subroutine prol0eva. Its first keep elements should not be changed
c       between the call to this subroutine and the subsequent calls tp
c       prol0eva.
c  rlam20 - the eigenvalue of the operator (1) corresponding to the
c       eigenfunction  \psi^c_0
c  keep - the number of elements of the array w that  should not be changed
c       between the call to this subroutine and the subsequent calls tp
c       prol0eva.
c  ltot - the maximum number of elements of the array w that were used by
c       the subroutine at any one time (generally, it is much bigger than
c       keep).
c 
c 
        ier=0
        thresh=45
        iw=11
        w(1)=iw+0.1
        w(9)=thresh
c 
c        create the data to be used in the evaluation of the
c        function \psi^c_0(x) for x \in [-1,1]
c 
        call prolps0i(ier,c,w(iw),lenw,nterms,ltot,rkhi)
c 
        if(ier  .ne. 0) return
c 
c       if c > thresh, do not prepare data for the evaluation of
c       psi^c_0 outside the interval [-1,1], since to the
c       double precision, it is zero anyway
c 
        if(c .lt. thresh) goto 1100
c 
        w(8)=c
        w(5)=nterms+0.1
        keep=nterms+3
        return
c 
 1100 continue
c 
c        create the data to be used in the evaluation of the
c        function \psi^c_0(x) for x outside the interval [-1,1]
c 
c        . . . construct the Gaussian nodes
c 
        ngauss=nterms*2
c 
        lw=nterms+2
c 
        its=iw+lw
        lts=ngauss+2
c 
        iwhts=its+lts
        lwhts=ngauss+2
c 
        ifs=iwhts+lwhts
        lfs=ngauss+2
c 
        keep=ifs+lfs
        if(keep .gt. ltot) ltot=keep
        if(keep .lt. lenw) goto 1200
c 
        ier=1024
        return
 1200 continue
c 
        w(2)=its+0.1
        w(3)=iwhts+0.1
        w(4)=ifs+0.1
c 
        itype=1
        call legeexps(itype,ngauss,w(its),u,v,w(iwhts) )
c 
c        . . . evaluate the prolate function at the Gaussian nodes
c 
        do 1400 i=1,ngauss
c 
        call legeexev(w(its+i-1),w(ifs+i-1),w(iw),Nterms-1)
 1400 continue
c 
c       calculate the eigenvalue corresponding to \psi^c_0
c 
        rlam=0
        x0=0
        call legeexev(x0,f0,w(iw),Nterms-1)
        call prosinin(c,w(its),w(iwhts),w(ifs),x0,ngauss,rlam,der)
c 
        rlam=rlam/f0
        rlam20=rlam
c 
        w(5)=nterms+0.1
        w(6)=ngauss+0.1
        w(7)=rlam
        w(8)=c
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine provcget(ier,eps,c)
        implicit real *8 (a-h,o-z)
c 
        save
        dimension coefs(70),coefs2(80)
c 
        data u1/0.14161773748182886E+00/,
     1      v1/-.12302984923784413E+01/,
     2      epslogmin/0.16261980771158072E+01/,
     3      epslogmax/0.15748722808572009E+02/
  
        data u2/0.13660050775677802E+01/,
     1      v2/-.12725545209468967E+01/,
     2      epslogmin2/0.19952672608815595E+00/,
     3      epslogmax2/0.16636501271234361E+01/
c 
        data coefs/
     1      0.23240703176515906E+02,0.16904482622672959E+02,
     2      -.19943834681469926E+00,0.76771929236285979E-01,
     3      -.32446982527893721E-01,0.14485227741385082E-01,
     4      -.67141923806662698E-02,0.32020322292731768E-02,
     5      -.15627638681443994E-02,0.77786509452320461E-03,
     6      -.39390867448902229E-03,0.20253110524719580E-03,
     7      -.10552372871053500E-03,0.55599140745361291E-04,
     8      -.29555888108961769E-04,0.15811920151959714E-04,
     9      -.84907493435835983E-05,0.45644993837931693E-05,
     *      -.24505383710468714E-05,0.13110229064258095E-05,
     1      -.69768648130260381E-06,0.36882311103566368E-06,
     2      -.19350162228376685E-06,0.10070656151626846E-06,
     3      -.51990664174913910E-07,0.26634953918211929E-07,
     4      -.13551195695024258E-07,0.68542404919930472E-08,
     5      -.34516530576288569E-08,0.17330464542926014E-08,
     6      -.86907566058780340E-09,0.43608239914074199E-09,
     7      -.21905279675060003E-09,0.11059496679906502E-09,
     8      -.55853723348490209E-10,0.28483844554965322E-10,
     9      -.14538990838137619E-10,0.74012315351642199E-11,
     *      -.39283928245288447E-11,0.18864335174004793E-11,
     1      -.11155674041520673E-11,0.47775566940002720E-12,
     2      -.29211492167985606E-12,0.17230547755173475E-12,
     3      -.12197421160994312E-13,0.12267089161264047E-12,
     4      0.67157888944552291E-13,0.84720802808910688E-13,
     5      0.40993077262944658E-13,0.11744586749595478E-13,
     6      -.33372963863612195E-13,-.67284865509883850E-13,
     7      -.91742797992529643E-13,-.95084014075016570E-13,
     8      -.77311084398161870E-13,-.39795206745387068E-13,
     9      0.84431916490306523E-14,0.55919683552803538E-13,
     *      0.90193594055583296E-13,0.10282432072964640E-12,
     1      0.90805223964984440E-13,0.58453428174968428E-13,
     2      0.13598334424233368E-13,-.31754108115451064E-13,
     3      -.70096951858356577E-13,-.90588390002626053E-13,
     4      -.95711951587030412E-13,-.76130335634293996E-13,
     5      -.48676454520208286E-13,-.14439749625328486E-14/
c 
        data coefs2/
     1      0.39433092526651749E+01,0.23420420512832133E+01,
     2      -.28080656955998076E+00,0.14592289365482385E+00,
     3      -.77328180316848850E-01,0.41500341088233284E-01,
     4      -.22885682289927782E-01,0.13036293447684436E-01,
     5      -.76435534166024648E-02,0.45857602342354682E-02,
     6      -.28009938858272514E-02,0.17353882177967897E-02,
     7      -.10876573244909715E-02,0.68819482011690215E-03,
     8      -.43890467840939901E-03,0.28179611358677326E-03,
     9      -.18196398710529291E-03,0.11808264619004208E-03,
     *      -.76960052206710900E-04,0.50350113337321222E-04,
     1      -.33052645842753734E-04,0.21763418641024278E-04,
     2      -.14369149962141869E-04,0.95105373637853779E-05,
     3      -.63088684818037559E-05,0.41935855902373428E-05,
     4      -.27927558263865910E-05,0.18630673262574445E-05,
     5      -.12448427854622971E-05,0.83298945586063779E-06,
     6      -.55815871369216285E-06,0.37447948653645623E-06,
     7      -.25154373009809009E-06,0.16915338952198106E-06,
     8      -.11386739730379532E-06,0.76725911520092721E-07,
     9      -.51746813948663900E-07,0.34930205955912955E-07,
     *      -.23597915375938931E-07,0.15954448330461734E-07,
     1      -.10794631681617861E-07,0.73086193816526132E-08,
     2      -.49516412416265365E-08,0.33568756203775055E-08,
     3      -.22770917557551091E-08,0.15455137667098250E-08,
     4      -.10495448087568028E-08,0.71310634756728894E-09,
     5      -.48475575204033648E-09,0.32968391899489364E-09,
     6      -.22432105201320930E-09,0.15269740669987466E-09,
     7      -.10398603565555017E-09,0.70842346296701998E-10,
     8      -.48281265542576707E-10,0.32917418624789703E-10,
     9      -.22450617528450786E-10,0.15317239915497241E-10,
     *      -.10453869001638040E-10,0.71369524132720939E-11,
     1      -.48739706281473864E-11,0.33295286344173240E-11,
     2      -.22751388973558053E-11,0.15550848958109820E-11,
     3      -.10632051253304832E-11,0.72709701167453122E-12,
     4      -.49736470312776501E-12,0.34029714866587067E-12,
     5      -.23288010127097064E-12,0.15939676052441259E-12,
     6      -.10911018472825273E-12,0.74682589863883010E-13,
     7      -.51096760207702941E-13,0.34919667476333076E-13,
     8      -.23799321754342901E-13,0.16120518903955726E-13,
     9      -.10768704647953097E-13,0.69671722535962199E-14,
     *      -.41636386458366930E-14,0.19495166654582977E-14/
c 
c       for a user-specifeid small number eps, this subroutine
c       finds the value of the parameter c such that
c 
c       \psi_0^c(1)) = eps**2                                   (1)
c 
c       In other words, the subroutine finds such c that
c       the value of the prolate function \psi_0^c at 1 is
c       equal to eps. Please note that this
c       subroutine can handle eps in the interval
c       [10**(-0.2), 10**(-15.7)] or so. For eps outside the
c       interval [10**(-0.2), 10**(-15.7)], ier is set to 4 and
c       the execution is teminated.
c 
c                       Input parameters:
c 
c  eps - the parameter in (1) above
c 
c                       Output paremeters:
c 
c  ier - error return code.
c    ier=0 means successful execution
c    ier=4 means that eps is outside permitted bounds
c  c - the parameter c solving the equation (1) above
c 
c         . . . decide which of the two intervals eps lives on
c 
        ier=0
        epslog=-log10(eps)
c 
        if( (epslog .gt. epslogmin2) .and. (epslog .lt. epslogmax))
     1      goto 1200
c 
        ier=4
        return
 1200 continue
c 
        if(epslog .lt. 1.66) goto 2000
c 
        xx=u1*epslog+v1
c 
        n=69
        call proexleg(xX,VAL,coefs,N)
c 
        c=val
        return
c 
 2000 continue
c 
        xx=u2*epslog+v2
c 
        n=79
        call proexleg(xX,VAL,coefs2,N)
c 
        c=val
c 
        return
        end
  
c 
c 
c 
c 
c 
        subroutine proicget(ier,eps,c)
        implicit real *8 (a-h,o-z)
        save
        dimension coefs(60),coefs2(40)
c 
        data u1/0.13751134246371496D+00/,
     1      v1/-.12755872687011086D+01/,
     2      epslogmin/0.20041057251246578D+01/,
     3      epslogmax/0.16548360505618412D+02/
  
        data u2/0.10091366131216004E+01/,
     1      v2/-.10640723511491342E+01/,
     2      epslogmin2/0.63492247051602654E-01/,
     3      epslogmax2/0.20453844646110512E+01/
c 
        data coefs/
     1      0.23082915855041622D+02,0.16967610377993772D+02,
     2      -.69269590777798827D-01,0.26381134031217221D-01,
     3      -.10918142102591140D-01,0.47398534660042369D-02,
     4      -.21263846453383401D-02,0.97836680381712732D-03,
     5      -.45971162838038489D-03,0.21997808745226123D-03,
     6      -.10695422137594323D-03,0.52713604424290993D-04,
     7      -.26262709247538202D-04,0.13181284035340756D-04,
     8      -.66374578280030189D-05,0.33377424510309894D-05,
     9      -.16676577391630224D-05,0.82340474918376963D-06,
     *      -.39944592633267686D-06,0.18917078158277347D-06,
     1      -.86792083316138139D-07,0.38188060216835932D-07,
     2      -.15866630658297623D-07,0.60541884132447431D-08,
     3      -.19914722515697689D-08,0.45326952806282617D-09,
     4      0.43913898010425224D-10,-.15040725630266032D-09,
     5      0.13311560700146793D-09,-.91214245076553512D-10,
     6      0.55412017733269359D-10,-.31265497987947531D-10,
     7      0.16753440667924376D-10,-.86325619958975813D-11,
     8      0.43141754534611603D-11,-.21032254422732301D-11,
     9      0.10045355142630821D-11,-.47292937008856985D-12,
     *      0.21864063354062021D-12,-.10104800108992540D-12,
     1      0.45580525951455670D-13,-.20616358117361743D-13,
     2      0.95878518231484388D-14,-.36331252673790827D-14,
     3      0.23614530266717513D-14,-.32549927705917392D-15,
     4      0.61276308003922899D-15,-.12382222170004186D-15,
     5      -.18539293294737018D-15,-.46803783292227017D-15,
     6      -.53573369639974040D-15,-.51483765879308662D-15,
     7      -.35746638886987143D-15,-.11548467924650958D-15,
     8      0.15938928200564729D-15,0.39995007940403710D-15,
     9      0.55045155623332236D-15,0.56288980467989356D-15,
     *      0.45296330984682223D-15,0.18930774188805079D-15/
c 
        data coefs2/
     1      0.36102884231580403E+01,0.25541616952821197E+01,
     2      -.76193022799009296E-01,0.65492461342406742E-02,
     3      0.11440314091155057E-01,-.95911600378030585E-02,
     4      0.44684185075945580E-02,-.12939812815980280E-02,
     5      0.99942848785446136E-04,0.13841559559481339E-03,
     6      -.98539160567081169E-04,0.37924868810789218E-04,
     7      -.76495079098134755E-05,-.12134238085480661E-05,
     8      0.18857564702823714E-05,-.93423586299956027E-06,
     9      0.26596113215837060E-06,-.14726547462738301E-07,
     *      -.31948624246389338E-07,0.21106203332173448E-07,
     1      -.75450582902171172E-08,0.12040082825153839E-08,
     2      0.45130931849474553E-09,-.45072735321061583E-09,
     3      0.19524673202609229E-09,-.45026042500158324E-10,
     4      -.37964930827281598E-11,0.92170944933950731E-11,
     5      -.48213521622938181E-11,0.13943209533772695E-11,
     6      -.65595751430587704E-13,-.17985789596395242E-12,
     7      0.11603557861146880E-12,-.39943061769248924E-13,
     8      0.51919187963196781E-14,0.32780164067522772E-14,
     9      -.27718736480326069E-14,0.11140561302715374E-14,
     *      -.19458609357189449E-15,-.14671841796562289E-15/
c 
c       for a user-specifeid small number eps, this subroutine
c       finds the value of the parameter c such that
c 
c       \int_{-\infty}^{+\infty} (\psi_0^c(x))^2 dx -1 =eps^2.   (1)
c 
c       In other words, the subroutine finds such c that
c       the L^2 norm of the prolate function \psi_0^c
c       on \R^2 \[-1,1] is equal to eps. Please note that this
c       subroutine can handle eps in the interval
c       [10**(-0.634), 10**(-16.5]) or so. For eps outside the
c       interval [10**(-0.634), 10**(-16.5)], ier is set to 4 and
c       the execution is teminated.
c 
c                       Input parameters:
c 
c  eps - the parameter in (1) above
c 
c                       Output paremeters:
c 
c  ier - error return code.
c    ier=0 means successful execution
c    ier=4 means that eps is outside permitted bounds
c  c - the parameter c solving the equation (1) above
c 
c 
c         . . . decide which of the two intervals eps lives on
c 
        ier=0
        epslog=-log10(eps)
c 
        if( (epslog .gt. epslogmin2) .and. (epslog .lt. epslogmax))
     1      goto 1200
c 
        ier=4
        return
 1200 continue
c 
c       eps < 0.01 - act accordingly
c 
        if(epslog .lt. 2) goto 2000
c 
        xx=u1*epslog+v1
c 
        n=59
        call proexleg(xX,VAL,coefs,N)
c 
        c=val
        return
c 
 2000 continue
c 
        xx=u2*epslog+v2
c 
        n=39
        call proexleg(xX,VAL,coefs2,N)
        c=val
c 
        return
        end
c 
c 
c 
c 
c 
        SUBROUTINE proexleg(X,VAL,PEXP,N)
        IMPLICIT REAL *8 (A-H,O-Z)
        save
        REAL *8 PEXP(1)
C 
C     evaluate the Legendre expansion of order n  with
c     coefficients pexp at the point x
c 
        pjm2=1
        pjm1=x
        val=pexp(1)*pjm2+pexp(2)*pjm1
c 
        DO 600 J = 2,N
c 
        pj= ( (2*j-1)*x*pjm1-(j-1)*pjm2 ) / j
        val=val+pexp(j+1)*pj
c 
        pjm2=pjm1
        pjm1=pj
  
 600   CONTINUE
c 
       RETURN
       END
c 
c 
c 
c 
c 
        subroutine prolvect(ier,n,c,as,bs,cs,
     1     xk,rlamouts,nvects,store,laddr,istore,eps,
     2     u,v,w,lw,nhigh,iscale,lneed)
        implicit real *8 (a-h,o-z)
        save
        dimension as(1),bs(1),cs(1),u(1),v(1),w(1),xk(1),rlamouts(1),
     1      store(1),laddr(4,1)
c 
c        this subroutine evaluates the Legendre coefficients
c        of the first nvects solutions of the equation
c 
c      D               D
c      -  ( (1-x**2) * -  \phi (x) ) + (rlam-c**2*x**2) * \phi(x) = 0,    (1)
c      Dx              Dx
c 
c        and the corresponding coefficients rlam. Note that the equation (1)
c        defines the prolate speheroidal wave functions; the coefficients
c        rlam are the so-called separation coefficients. The coefficients
c        are stored in array store, and can be used to evaluate the
c        prolate spheroidfal wave functions and theirr derivatives. The
c        recommended way to do so is by calling the subroutine prolev0 (see).
c 
c                          input parameters:
c 
c  n - the maximum permitted number of legendre coefficients in the
c       expansion of any wave function to be computed. Must be
c       sufficiently large (to be supplied by the user!!)
c  c - the coefficient (real) in the formula (1)
c  nvects - the number of solutions of the equation (1) whose Legendre
c       expansions are to be computed
c  eps - the accuracy to which the calculations are to be performed
c  lw - the amount of storage space provided in the array lw; must
c       be sufficiently large; if it is too small, the error code ier
c       is set to 4, and the execution of the subroutine is terminated.
c       Note that this parameter, normally, does not need to be very large.
c 
c                          output parameters:
c 
c  ier - error return code.
c         ier=0 means successful execution.
c         ier=4 means that the amount of storage space provided in
c                 the array store is insufficient (see input
c                 parameter lw above). This is a fatal error.
c         ier=1024 means that the subroutine prolql1 (a standard
c                 prehistoric subroutine from eispack) has failed
c                 to find the spectrum of a certain tridiagonal matrix;
c                 this has never happened yet, and probably means that memory
c                 has been mismanaged by the user. This is a fatal error.
c  rlamouts - the first nvects values of the coefficient rlam in (1) for
c       which (1) has a non-trivial soulution.
c  store - the array containing the coefficients of the Legendre
c          expansions of the first nvects non-sero solutions of (1).
c          Note that these solutions are normalized (the L^2 norm of
c          each is equal to 1). Normally, it is to be used by the
c          subroutine prolev0 (see). Also, note that the first istore
c          elements of the array store should not be altered between the
c          call to this subroutine and the subsequent calls to prolev0.
c  laddr - the map of the array store; to be used by the subroutine
c          prolev0 (see). it will be 4*nvects integer *4 elements long.
c          EXPLANATION of the meaning of entries in the array laddr:
c      laddr(1,i) =i
c      laddr(2,i) - the location in array store of the first coefficient
c          of the Legendre expansion for the i-th eigenfunction
c      laddr(3,i) - the number of coefficients of the Legendre expansion
c          of the i-the eigenfunction whose absolute values are greater
c          than eps; thus, the elements of the array store with numbers
c          laddr(2,i) through laddr(2,i)+laddr(3,i)-1 contain the
c          coefficients of the Legendre expansion of the i-th
c          eigenfunction.
c      laddr(4,1) - the number of the first coefficient in the Legendre
c          series of the i-th eigenfunction; this entry exists because
c          the first several coefficients in the Legendre series of the
c          higher order eigenfunctions tend to be negligibly small.
c  istore - the total number of elements in array store where something
c          has been stored
c  nhigh - the highest order of a legendre expansion actually used
c          for any eigenfunction
c  lneed - the amount of space in array store that have been used (note
c          that this is different from istore that has been produced
c          as output and should not be altered between the
c          call to this subroutine and the subsequent calls to prolev0.
c 
c                             work arrays:
c 
c  as,bs,cs,um,v,w - must be at least n/2+6 each
c 
c        . . . construct the tridiagonal matrix whose eigenvalues
c              are the "separation coefficients" for the prolate spheroidal
c              wave functions
c 
c         . . . for the even-numbered separation coefficients
c 
        ier=0
        delta=1.0d-8
        ifsymm=1
        numit=4
        rlam=0
        ifodd=-1
        call prolmatr(as,bs,cs,n,c,rlam,ifsymm,ifodd)
c 
c       find the spectrum of the tridiagonal matrix
c 
        call PROLQL1(N/2,bs,as,IERR)
        if(ierr .ne. 0) ier=1024
        if(ier .ne. 0) return
c 
        do 1040 i=1,n/2
        u(n/2-i+1)=-bs(i)
 1040 continue
c 
c         . . . for the even-numbered separation coefficients
c 
        delta=1.0d-8
        ifsymm=1
        numit=4
        rlam=0
        ifodd=1
        call prolmatr(as,bs,cs,n,c,rlam,ifsymm,ifodd)
c 
c       find the spectrum of the tridiagonal matrix
c 
        call PROLQL1(N/2,bs,as,IERR)
        if(ierr .ne. 0) ier=1024
        if(ier .ne. 0) return
c 
        do 1080 i=1,n/2
        v(n/2-i+1)=-bs(i)
 1080 continue
c 
        j=0
        do 1100 i=1,n/2
        j=j+1
        rlamouts(j)=u(i)
        j=j+1
        rlamouts(j)=v(i)
c 
        if(j .eq. nvects) goto 1150
 1100 continue
 1150 continue
c 
cccc        call prin2('and rlamouts=*',rlamouts,n/2)
c 
c       use the inverse power method to get the eigenvectors
c 
        istore=1
        i=0
        ifodd=1
        do 4000 i7=1,nvects
c 
        i=i+1
        ifodd=-ifodd
        if(i .gt. nvects) return
c 
        do 1200 j=1,n
        xk(j)=1
 1200 continue
c 
        done=1
c 
c        construct the tridiagonal matrix
c 
        rlam=rlamouts(i)+delta
        ifsymm=1
        call prolmatr(as,bs,cs,n,c,rlam,ifsymm,ifodd)
c 
c        construct the l-u decomposition of the unsymmtrized tridiagonal
c        matrix
c 
         call prolfact(bs,cs,as,n/2,u,v,w)
c 
c        repeatedly apply the inverse of the unsymmtrized diagonal matrix
c        to the seed vector, in the hope of getting
  
        do 3000 ijk=1,numit
c 
        call prolsolv(u,v,w,n/2,xk)
c 
        d=0
        do 2400 j=1,n/2
        d=d+xk(j)**2
c 
 2400 continue
c 
        d=sqrt(d)
        do 2500 j=1,n/2
        xk(j)=xk(j)/d
 2500 continue
c 
        err=0
        do 2600 j=1,n/2
        err=err+(as(j)-xk(j))**2
c 
        as(j)=xk(j)
 2600 continue
        err=sqrt(err)
 3000 continue
c 
        lneed=istore+n/2+10
c 
        if(lneed .lt. lw) goto 3400
        ier=4
          call prinf('bombing from prolvect, ier=*',ier,1)
        return
 3400 continue
c 
c        store the non-zero part of the vector xk in the array store
c 
        call prolpack(i,xk,store,laddr,istore,n/2,eps)
 4000 continue
c 
c        find the highest order of the Legendre expansion for any
c        eigenfunction
c 
        nhigh=0
c 
        do 4200 i=1,nvects
        ihigh=laddr(3,i)+laddr(4,i)
        if(ihigh .gt. nhigh) nhigh=ihigh
 4200 continue
        nhigh=nhigh*2+2
c 
c        construct and put in arrays store the scaling coefficients
c        to be used by the subroutine prolev0
c 
        iscale=istore+1
        lneed=iscale+nhigh+2
        if(lneed .lt. lw) goto 4400
         call prinf('bombing from prolvect, ier=*',ier,1)
        ier=4
        return
 4400 continue
c 
        half=1
        half=half/2
        do 4600 i=1,nhigh
        store(iscale+i-1)=sqrt(i-half)
 4600 continue
c 
        istore=lneed
        istore=istore+2*nhigh+8
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prollam0(n,c,store,laddr,
     1      pexp,x,whts,nhigh,rlam,w)
        implicit real *8 (a-h,o-z)
        save
        dimension store(1),laddr(4,1),x(1),whts(1),pexp(1),w(1)
c 
c        starting with the first eigenvector of the integral
c        operator
c 
c         F(\phi) (x) = \int_{-1}^1  \phi (t) * e^{i * c *x * t) dt,     (1)
c 
c        this subroutine constructs the corresponding eigenvalue
c        of the operator (1). This is done in the dumbest possible
c        manner; specifically, the subroutine discretizes the interval
c        [-1,1] into Gaussian nodes, and evaluates the integral (1)
c        for a single point. This point is chosen among the Gaussian
c        nodes on the interval [-1,1]. The subroutine chooses the node
c        where the eigenvector is the biggest (the obvious choice to
c        minimize the round-off error).
c 
c                          input parameters:
c 
c  n - the number of nodes in the Gaussian discretization of the
c       interval [-1,1] to be used to evaluate the integral (1)
c  c - the coefficient (real) in the formula (1)
c  store - the array containing the coefficients of the Legendre
c          expansions of the first nvects non-sero solutions of (1).
c          This array must have been produced by a prior call to
c          the subroutine prolvect (see)
c  laddr - the map of the array store; this array must have been
c          produced by a prior call to the subroutine prolvect (see).
c 
c                          output parameters:
c 
c  rlam - the first eigenvalue of the operator (1)
  
c                          work arrays:
c 
c  x,whts - must be at least n+2 real *8 elements each
c  pexp - must be at least nhigh real *8 elements eash; the parameter
c         nhigh must have been produced by a prior call to the
c         subroutine prolvect (see)
c 
c        . . . construct the Gaussian nodes and weights on
c              the interval [-1,1]
c 
        itype=1
        call legeexps(itype,n,x,u,v,whts)
c 
c        Construct the first eigenvector at the gaussian nodes,
c        and apply the integral operator to the thing at one point
c 
c        . . . construct the eigenfunction
c 
        iscale=w(3)
        imax=n/2
        rat2=0
c 
        do 1600 j=1,n
c 
        call prolev0(i0,x(j),store,laddr,val,der,pexp,nhigh,
     1      w(iscale) )
c 
        cd=cos(c*x(imax)*x(j) )
        rat2=rat2+cd*whts(j)*val
c 
        if(j .eq. imax) valmax=val
 1600 continue
        rlam=rat2/valmax
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prolevv0(k,x,w,laddr,val,pexp,coefs,
     1      ninit,coepnm1,coepnp1,coexpnp1,
     2      coepnm1o,coepnp1o,coexpnp1o)
        implicit real *8 (a-h,o-z)
        save
        dimension laddr(4,1),w(1),pexp(1),coefs(1),
     1      coepnm1(1),coepnp1(1),coexpnp1(1),
     2      coepnm1o(1),coepnp1o(1),coexpnp1o(1)
c 
c       . . . expand the chunk of the array w corresponding to
c             the functions order k, so that it could be fed into
c             the subroutine legefder
c 
        k1=k+1
        i=k/2
        i00=k-i*2
c 
        j0=laddr(2,k1)-1
        nn=laddr(3,k1)+laddr(4,k1)
c 
        do 1200 i=1,nn+10
        pexp(i)=0
 1200 continue
c 
        i0=(laddr(4,k1)-1)*2
        i0=i0+i00
c 
c       evaluate the value of the k-th function in the case of even k
c 
        jj1=k/2
        jj1=k-jj1*2
        if(jj1 .eq. 1) goto 2200
c 
        j=0
        i02=i0/2
        i0p1=i0+1
c 
        do 1400 i=1,nn+2
        j=j+1
        pexp(i+i02)=w(j+j0) * coefs((i-1)*2+i0p1)
 1400 continue
c 
        nn2=nn*2-2
c 
c       if this is the first call, make sure that the computation
c       of odd Legendre expansions is also initialized
c 
        if(ninit .ne. 0)
cccc     1      call legeodev(x,nn2-2,pexp,val,ninit,
     1      call legeodev(x,nn2,pexp,val,ninit,
     2      coepnm1o,coepnp1o,coexpnp1o)
  
cccc        call legeevev(x,nn2-2,pexp,val,ninit,
        call legeevev(x,nn2,pexp,val,ninit,
     1      coepnm1,coepnp1,coexpnp1)
c 
        return
c 
 2200 continue
c 
c       evaluate the value of the k-th function in the case of even k
c 
        j=0
        i02=i0/2
        i0m1=i0-1
c 
        do 2400 i=1,nn+2
        j=j+1
        pexp(i+i02)=w(j+j0) * coefs(i*2+i0m1)
 2400 continue
c 
        nn2=nn*2-2
c 
c       if this is the first call, make sure that the computation
c       of even Legendre expansions is also initialized
c 
        if(ninit .ne. 0)
     1      call legeevev(x,nn2,pexp,val,ninit,
     2      coepnm1o,coepnp1o,coexpnp1o)
c 
        call legeodev(x,nn2,pexp,val,ninit,
     1      coepnm1o,coepnp1o,coexpnp1o)
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prolmatr(as,bs,cs,n,c,rlam,ifsymm,ifodd)
        implicit real *8 (a-h,o-z)
        save
        dimension as(1),bs(1),cs(1)
c 
c       construct the tridiagonal matrix corresponding
c       to odd-numbered P_k
c 
        if(ifodd .le. 0) goto 1300
        k=0
        done=1
        half=done/2
        do 1200 k0=1,n+2,2
c 
        k=k+1
c 
        call prolcoef(rlam,k0,c,alpha0,beta0,gamma0,
     1      alpha,beta,gamma)
c 
        as(k)=alpha
        bs(k)=beta
        cs(k)=gamma
c 
c        remembering that the norm of P_n is not equal to 1,
c        rescale the matrix to make it symmetric
  
        if(ifsymm .eq. 0)  goto 1200
c 
        if(k0 .gt. 1)
     1    as(k)=as(k)/sqrt(k0-2+half)*sqrt(k0+half)
c 
        cs(k)=cs(k)*sqrt(k0+half)/sqrt(k0+half+2)
c 
 1200 continue
c 
        return
 1300 continue
c 
c       construct the tridiagonal matrix corresponding
c       to even-numbered P_k
c 
        k=0
        done=1
        half=done/2
        do 1400 k0=0,n+2,2
c 
        k=k+1
c 
        call prolcoef(rlam,k0,c,alpha0,beta0,gamma0,
     1      alpha,beta,gamma)
c 
        as(k)=alpha
        bs(k)=beta
        cs(k)=gamma
c 
c        remembering that the norm of P_n is not equal to 1,
c        rescale the matrix to make it symmetric
c 
        if(ifsymm .eq. 0) goto 1400
c 
        if(k0 .ne. 0)
     1    as(k)=as(k)/sqrt(k0-2+half)*sqrt(k0+half)
  
c 
        cs(k)=cs(k)*sqrt(k0+half)/sqrt(k0+half+2)
c 
 1400 continue
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prolfact(a,b,c,n,u,v,w)
        implicit real *8 (a-h,o-z)
        save
        dimension a(1),b(1),c(1),u(1),v(1),w(1),rhs(1)
c 
c        eliminate down
c 
        do 1200 i=1,n-1
        d=c(i+1)/a(i)
        a(i+1)=a(i+1)-b(i)*d
        u(i)=d
 1200 continue
c 
c        eliminate up
c 
        do 1400 i=n,2,-1
        d=b(i-1)/a(i)
        v(i)=d
 1400 continue
c 
c       scale the diagonal
c 
        done=1
        do 1600 i=1,n
        w(i)=done/a(i)
 1600 continue
c 
        return
c 
c 
c 
c 
        entry prolsolv(u,v,w,n,rhs)
c 
c        eliminate down
c 
        do 2400 i=1,n-1
        rhs(i+1)=rhs(i+1)-u(i)*rhs(i)
 2400 continue
c 
c        eliminate up
c 
        do 2600 i=n,2,-1
        rhs(i-1)=rhs(i-1)-rhs(i)*v(i)
 2600 continue
c 
c       scale
c 
        do 2800 i=1,n
        rhs(i)=rhs(i)*w(i)
 2800 continue
        return
        end
c 
c 
c 
c 
c 
        subroutine prolcoef(rlam,k,c,alpha0,beta0,gamma0,
     1      alpha,beta,gamma)
        implicit real *8 (a-h,o-z)
c 
c        this subroutine evaluates the Legendre coefficients
c        alpha0, beta0, gamma0, alpha, beta, gamma of two functions:
c 
c 
c           (1-x**2)   P_k (x) )  =
c                                                                         (1)
c           alpha0 * P_{k-2} + beta0 * P_{k} + gamma0 * P_{k+2},
c 
c        and
c 
c           D               D
c           -  ( (1-x**2) * -  P_k (x) ) + (rlam-c**2*x**2) * P_k(x) =
c           Dx              Dx
c                                                                         (2)
c           alpha * P_{k-2} + beta * P_{k} + gamma * P_{k+2}.
c 
c 
c                          input parameters:
c 
c  rlam - the coefficient (real) in the formula (2)
c  k - the index in the formulae (1), (2)
c  c - the coefficient (real) in the formula (2)
c 
c                          output parameters:
c 
c  alpha0, beta0, gamma0 - coefficients in the expansion (1)
c  alpha, beta, gamma - coefficients in the expansion (2)
c 
c 
        save
        d=k*(k-1)
        d=d/(2*k+1)/(2*k-1)
        uk=d
c 
        d=(k+1)**2
        d=d/(2*k+3)
        d2=k**2
        d2=d2/(2*k-1)
        vk=(d+d2)/(2*k+1)
c 
        d=(k+1)*(k+2)
        d=d/(2*k+1)/(2*k+3)
        wk=d
c 
        alpha=-c**2*uk
        beta=rlam-k*(k+1)-c**2*vk
        gamma=-c**2*wk
c 
        alpha0=uk
        beta0=vk
        gamma0=wk
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prolev0(k,x,w,laddr,val,der,pexp,nhigh,coefs)
        implicit real *8 (a-h,o-z)
        save
        dimension laddr(4,1),w(1),pexp(1),coefs(1)
c 
c        this subroutine calculates the values at the point x of the
c        k-th prolate spheroidal function and its derivative. The
c        subroutine uses the arrays w and laddr that are produced
c        by the subroutine prolvect (see). Note that the array w in
c        this subroutine is known under the name store in prolvect.
c 
c                         input parameters:
c 
c  k - the order of the prolate function to be evaluated. note that the
c       permitted values are 0, 1, ..., nvects-1 (see the subroutine
c       prolvect for the definition of the parameter nvects).
c  x - the point where the prolate function and its derivative are to
c       be evaluated; should be in the interval [-1,1]
c  w - produced by the subroutine prolvect (see) under the name store
c  laddr - produced by the subroutine prolvect under the same name
c  nhigh - the highest order of the legendre expansion for any of
c       the eigenfunctions (as produced by the subroutine prolvect)
c 
c                           output parameters:
c 
c  val - the value of the k-th prolate spheroidal wave function at
c       the point x.
c  der - the value of the derivative of the k-th prolate spheroidal
c       wave function at the point x.
c 
c                           work arrays:
c 
c  pexp - must be nhigh real *8 elements long, where lenmax is produced
c       by a preceding call to the subroutine prolvect (see).
c 
c 
c       . . . expand the chunk of the array w corresponding to
c             the functions order k, so that it could be fed into
c             the subroutine legefder
c 
        k1=k+1
        i=k/2
        i00=k-i*2
c 
        j0=laddr(2,k1)-1
        nn=laddr(3,k1)+laddr(4,k1)
c 
cccc        call prinf('in prolev0, laddr(k1)=*',laddr(1,k1),3)
c 
        do 1200 i=1,nn*2+10
        pexp(i)=0
 1200 continue
c 
        i0=(laddr(4,k1)-1)*2
        i0=i0+i00
cccc        call prinf('i0=*',i0,1)
c 
        j=0
        do 1400 i=1,nn*2+2,2
        j=j+1
        pexp(i+i0)=w(j+j0)
 1400 continue
c 
c        scale the coefficients properly
c 
        half=1
        half=half/2
        do 1600 i=1,nn*2+4
        i0=i-1
cccc        pexp(i)=pexp(i)*sqrt(i0+half)
c 
        pexp(i)=pexp(i)*coefs(i)
 1600 continue
c 
        nn2=nn*2-2
c 
c       evaluate the value of the k-th function and its derivative
c 
        call legeFDER(X,VAL,der,PEXP,Nn2-1)
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prolpack(k,xk,store,laddr,istore,n,eps)
        implicit real *8 (a-h,o-z)
        save
        dimension store(1),laddr(4,1),xk(1)
c 
c       find the first element of the vector of coefficients
c       that is non-zero
c 
        do 1200 i=1,n
        i1=i
        if(abs(xk(i)) .lt. eps) goto 1200
        goto 1400
 1200 continue
 1400 continue
c 
c       find the last coefficient of the vector xk that is non-zero
c 
        do 1600 i=n,i1,-1
        i2=i
        if(abs(xk(i)) .lt. eps) goto 1600
        goto 1800
 1600 continue
 1800 continue
c 
c        store in array store the non-zero elements of this vector
c 
        nn=i2-i1+1
        do 2000 i=1,nn
        store(istore+i-1)=xk(i1+i-1)
 2000 continue
c 
c       enter the appropriate information in the array laddr
c 
        laddr(1,k)=k
        laddr(2,k)=istore
        laddr(3,k)=nn
        laddr(4,k)=i1
        istore=istore+nn
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine proleva0(k,x,w,laddr,val,der,pexp,nhigh,coefs,
     1      pjcoefs1,pjcoefs2,ninit)
        implicit real *8 (a-h,o-z)
        save
        dimension laddr(4,1),w(1),pexp(1),coefs(1),
     1      pjcoefs1(1),pjcoefs2(1)
c 
c       . . . expand the chunk of the array w corresponding to
c             the functions order k, so that it could be fed into
c             the subroutine legefder
c 
        k1=k+1
        i=k/2
        i00=k-i*2
c 
        j0=laddr(2,k1)-1
        nn=laddr(3,k1)+laddr(4,k1)
c 
        do 1200 i=1,nn*2+10
        pexp(i)=0
 1200 continue
c 
        i0=(laddr(4,k1)-1)*2
        i0=i0+i00
c 
        j=0
        do 1400 i=1,nn*2+2,2
        j=j+1
        pexp(i+i0)=w(j+j0)
 1400 continue
c 
c        scale the coefficients properly
c 
        do 1600 i=1,nn*2+4
        i0=i-1
c 
        pexp(i)=pexp(i)*coefs(i)
 1600 continue
c 
        nn2=nn*2-2
c 
c       evaluate the value of the k-th function and its derivative
c 
        call legeFDE2(X,VAL,der,PEXP,Nn2-1,
     1    pjcoefs1,pjcoefs2,ninit)
  
        return
        end
c 
c 
c 
c 
c 
        subroutine prolunp0(k,coefs,store,laddr,n,w)
        implicit real *8 (a-h,o-z)
        save
        dimension coefs(1),store(1),laddr(4,1),w(1)
c 
c       retrieve the legendre coefficients corresponding to the
c       k-th eigenfunction in their compressed (odd-even) form
c 
        call prolret(k,w,store,laddr,n)
c 
c        . . . unpack
c 
        ii=k/2
        i0=k-ii*2
c 
        do 1200 i=1,n*2+2
        coefs(i)=0
 1200 continue
c 
        do 1400 i=1,n
        j=(i-1)*2+i0+1
        coefs(j)=w(i)
 1400 continue
c 
        n=n*2
        return
        end
c 
c 
c 
c 
c 
        subroutine prolret(kk,coefs,store,laddr,nn)
        implicit real *8 (a-h,o-z)
        save
        dimension store(1),laddr(4,1),coefs(1)
c 
c       retrieve from array store the coefficients of the Legendre
c       expansion of the k-th prolate spherical wave function
c 
        k=kk+1
        i0=laddr(2,k)
        j0=laddr(4,k)
        nj=laddr(3,k)
        nn=j0+nj+1
c 
        do 1200 i=1,nn
        coefs(i)=0
 1200 continue
c 
        do 1400 i=1,nj
        coefs(j0+i-1)=store(i0+i-1)
 1400 continue
        return
        end
c 
c 
c 
c 
c 
        subroutine prolrat(c,store,laddr,nvects,rlam0,
     1      rlams2,nhigh,rats,coefsk,coefskp1,derk,derkp1,scales)
        implicit real *8 (a-h,o-z)
        save
        dimension store(1),laddr(3,1),rlams2(1),coefsk(1),
     1      coefskp1(1),derk(1),derkp1(1),rats(1),scales(1)
c 
c        this subroutine evaluates the absolute values of the first
c        nvects eigenvalues of the operator
c 
c         F(\phi) (x) = \int_{-1}^1  \phi (t) * e^{i * c *x * t) dt   (1)
c 
c        Note: this subroutine uses parameters store, laddr,nhigh
c        that must have been produced by the subroutine prolvect (see),
c        and the parameter rlam0 that must have been produced by the
c        subroutine prollam0. This subroutine has no uses as a stand
c        alone device.
c 
c                         input parameters:
c 
c  c - the coefficient (real) in the formula (1)
c  nvects - the number of solutions of the equation (1) whose Legendre
c       expansions are to be computed
c  store - produced by the subroutine prolvect (see).
c  laddr - produced by the subroutine prolvect
c  rlams0 - the largest eigenvalue of the operator (1) (note that
c        it is real, as well as all other even-numbered eigenvalues)
c  nhigh - the highest order of the legendre expansion for any of
c       the eigenfunctions (as produced by the subroutine prolvect)
c 
c                           output parameters:
c 
c  rlams2 - the absolute values of the first nvects eigenvalues
c       of the operator (1)
c 
c                           work arrays:
c 
c  rats - must be at least nvects+1 real *8 elements long
c  coefsk,coefskp1,derk,derkp1 - must be at least nhigh+2 real *8
c       elements each
c 
c 
c        . . . one eigenfunction after another, construct the ratios
c              of consecutive pairs of eigenvectors
c 
        done=1
        do 1100 i=1,nhigh
        scales(i)=sqrt(2*i-done)
 1100 continue
c 
        do 2000 k=1,nvects-1
c 
c        retrieve from array store the Legendre expansions for the
c        i-th and i+1-st eigenfunctions
c 
        kk=k-1
        call prolunp0(kk,coefsk,store,laddr,nnk,derk)
c 
        call prolunp0(k,coefskp1,store,laddr,nnkp1,derk)
c 
c 
c        rescale both expansion so that the standard differentiation
c        routine would work
c 
        do 1200 i=1,nnkp1+4
c 
        coefsk(i)=coefsk(i)*scales(i)
        coefskp1(i)=coefskp1(i)*scales(i)
 1200 continue
c 
c       differentiate both expansions
c 
        call legediff(coefsk,nnk,derk)
        call legediff(coefskp1,nnkp1,derkp1)
c 
c        rescale back the whole bunch
c 
        do 1300 i=1,nnkp1
c 
        ddd=done/scales(i)
c 
        coefsk(i)=coefsk(i)*ddd
        coefskp1(i)=coefskp1(i)*ddd
c 
        derk(i)=derk(i)*ddd
        derkp1(i)=derkp1(i)*ddd
 1300 continue
c 
c        construct the cross-integrals
c 
        nn=nnk
        if(nnkp1 .gt. nn) nn=nnkp1
        nn=nn-1
c 
        d1=0
        d2=0
        do 1400 i=1,nn-1
c 
        d1=d1+coefsk(i)*derkp1(i)
        d2=d2+coefskp1(i)*derk(i)
 1400 continue
c 
        rats(k)=d2/d1
        rlams2(k)=d1
 2000 continue
c 
c        using the newly obtained ratios and previously obtained
c        largest eigenvalue of the integral operator, construct the
c        first nvects values of the integral operator
c 
        rlams2(1)=rlam0
        do 2200 i=1,nvects-1
        rlams2(i+1)=rlams2(i)*sqrt(-rats(i))
 2200 continue
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prosinin(c,ts,whts,fs,x,n,rint,derrint)
        implicit real *8 (a-h,o-z)
        save
        dimension ts(1),whts(1),fs(1)
c 
c        use the integral formula to evaluate the prolate function
c 
        rint=0
        derrint=0
        do 1600 i=1,n
c 
        rint=rint+whts(i)*fs(i)*sin(c*(x-ts(i)))/(x-ts(i))
c 
        derrint=derrint+whts(i)*fs(i)/(x-ts(i))**2 *
     1      ( c*(x-ts(i))*cos(c*(x-ts(i))) - sin(c*(x-ts(i))) )
 1600 continue
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prolps0i(ier,c,w,lenw,nterms,ltot,rkhi)
        implicit real *8 (a-h,o-z)
c 
        save
        dimension ns(20),w(1)
c 
        data ns/48,64,80,92,106,120,130,144,156,168,
     1       178,190,202,214,224,236,248,258,268,280/
c 
c       This subroutine constructs the Legendre expansion of the
c       prolate function \psi^c_0, for the user-specified c. The
c       expansion is returned in the first nterms positions in the
c       array w; more (specifically, ltot) elements of w are  used
c       by the subroutine during the calculation.
c 
c                      Input parameters:
c 
c  c - the parameter "c" in \psi^c_0
c  lenw - the length of the user-supplied array w (in real *8 words);
c       please note that lenw = 7*(3/2*c+300) +100 is always more than
c       sufficient
c 
c                      Output parameters:
c 
c  w - the coefficients of the Legendre expansion of the function
c       \psi_0^c (nterms of them)
c  nterms - the number of coefficients of the  Legendre expansion of
c       the function \psi_0^c returned in the array w
c  ltot - the maximum number of elements of the array w used by
c       the subroutine at any time;
c 
c 
c        . . . Find how many Legendre polynomials to choose in
c              the expansions of the eigenvectors
c 
         eps=1.0d-16
         n=c*3
         n=n/2
c 
         i=c/10
         if(i .le. 19) n=ns(i+1)
c 
c       allocate the memory for the subroutine prolfun0
c 
        ier=0
        ixk=1
        lxk=n+2
c 
        ias=ixk+lxk
        las=n+2
c 
        ibs=ias+las
        lbs=n+2
c 
        ics=ibs+lbs
        lcs=n+2
c 
        iu=ics+lcs
        lu=n+2
c 
        iv=iu+lu
        lv=n+2
c 
        iw=iv+lv
        lw=n+2
c 
        ltot=iw+lw
c 
        if(ltot .lt. lenw) goto 1100
        ier=512
        return
 1100 continue
c 
c       construct the coefficients of the Legendre expansion
c       of \psi_0
c 
        call prolfun0(ier,n,c,w(ias),w(ibs),w(ics),w(ixk),
     1      w(iu),w(iv),w(iw),eps,nterms,rkhi)
  
        if(ier .ne. 0) return
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prolfun0(ier,n,c,as,bs,cs,xk,u,v,w,eps,nterms,
     1      rkhi)
        implicit real *8 (a-h,o-z)
        save
        dimension as(1),bs(1),cs(1),u(1),v(1),w(1),xk(1)
c 
c        for the even-numbered separation coefficients, construct the
c        tridiagonal matrix whose eigenvalues are the said "separation
c        coefficients" for the prolate spheroidal wave functions
c 
        ier=0
        delta=1.0d-8
        ifsymm=1
        numit=4
        rlam=0
        ifodd=-1
        call prolmatr(as,bs,cs,n,c,rlam,ifsymm,ifodd)
c 
c       find the spectrum of the tridiagonal matrix
c 
        call PROLQL1(N/2,bs,as,IERR)
        if(ierr .ne. 0) ier=2048
        if(ier .ne. 0) return
c 
        rkhi=-bs(n/2)
        rlam=-bs(n/2) + delta
  
c       use the inverse power method to get the eigenvectors
c 
        do 1200 j=1,n
        xk(j)=1
 1200 continue
c 
        done=1
c 
c        construct the tridiagonal matrix
c 
        ifsymm=1
        call prolmatr(as,bs,cs,n,c,rlam,ifsymm,ifodd)
c 
c        construct the l-u decomposition of the unsymmtrized tridiagonal
c        matrix
c 
         call prolfact(bs,cs,as,n/2,u,v,w)
c 
c        repeatedly apply the inverse of the unsymmtrized diagonal matrix
c        to the seed vector, in the hope of getting
  
        do 3000 ijk=1,numit
c 
        call prolsolv(u,v,w,n/2,xk)
c 
        d=0
        do 2400 j=1,n/2
        d=d+xk(j)**2
c 
 2400 continue
c 
        d=sqrt(d)
        do 2500 j=1,n/2
        xk(j)=xk(j)/d
 2500 continue
c 
        err=0
        do 2600 j=1,n/2
        err=err+(as(j)-xk(j))**2
c 
        as(j)=xk(j)
 2600 continue
        err=sqrt(err)
c 
 3000 continue
c 
c       determine the number of elements in the vector xk
c       that are greater than eps
c 
        half=1
        half=half/2
        do 3200 i=1,n/2
        if(abs(xk(i)) .gt. eps) nterms=i
c 
        xk(i)=xk(i)*sqrt((i-1)*2+half)
c 
        cs(i)=xk(i)
c 
 3200 continue
c 
c        repack them things so that they would contain both
c        even and odd powers (the coefficients at the latter
c        are zeroes, obviously)
c 
        j=0
        do 3400 i=1,nterms+1
        j=j+1
        xk(j)=cs(i)
        j=j+1
        xk(j)=0
 3400 continue
c 
        nterms=nterms*2
c 
        return
        end
c 
c 
c 
c 
c 
      SUBROUTINE PROLQL1(N,D,E,IERR)
C 
        save
      INTEGER I,J,L,M,N,II,MML,IERR
      real *8  D(N),E(N)
      real *8 B,C,F,G,P,R,S,TST1,TST2
C 
C     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE IMTQL1,
C     NUM. MATH. 12, 377-383(1968) BY MARTIN AND WILKINSON,
C     AS MODIFIED IN NUM. MATH. 15, 450(1970) BY DUBRULLE.
C     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 241-248(1971).
c 
c      Very minor changes have been introduced by V. Rokhlin, on
c      5.22.96
c 
C     THIS SUBROUTINE FINDS THE EIGENVALUES OF A SYMMETRIC
C     TRIDIAGONAL MATRIX BY THE IMPLICIT QL METHOD.
C 
C     ON INPUT
C 
C        N IS THE ORDER OF THE MATRIX.
C 
C        D CONTAINS THE DIAGONAL ELEMENTS OF THE INPUT MATRIX.
C 
C        E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE INPUT MATRIX
C          IN ITS LAST N-1 POSITIONS.  E(1) IS ARBITRARY.
C 
C      ON OUTPUT
C 
C        D CONTAINS THE EIGENVALUES IN ASCENDING ORDER.  IF AN
C          ERROR EXIT IS MADE, THE EIGENVALUES ARE CORRECT AND
C          ORDERED FOR INDICES 1,2,...IERR-1, BUT MAY NOT BE
C          THE SMALLEST EIGENVALUES.
C 
C        E HAS BEEN DESTROYED.
C 
C        IERR IS SET TO
C          ZERO       FOR NORMAL RETURN,
C          J          IF THE J-TH EIGENVALUE HAS NOT BEEN
C                     DETERMINED AFTER 30 ITERATIONS.
C 
C     CALLS PYTHAG FOR  SQRT(A*A + B*B) .
C 
C     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW,
C     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY
C 
C     THIS VERSION DATED AUGUST 1983.
C 
C     ------------------------------------------------------------------
C 
      IERR = 0
      IF (N .EQ. 1) GO TO 1001
C 
      DO 100 I = 2, N
  100 E(I-1) = E(I)
C 
      E(N) = 0.0D0
C 
      DO 290 L = 1, N
         J = 0
C     .......... LOOK FOR SMALL SUB-DIAGONAL ELEMENT ..........
  105    DO 110 M = L, N
            IF (M .EQ. N) GO TO 120
            TST1 = ABS(D(M)) + ABS(D(M+1))
            TST2 = TST1 + ABS(E(M))
            IF (TST2 .EQ. TST1) GO TO 120
  110    CONTINUE
C 
  120    P = D(L)
         IF (M .EQ. L) GO TO 215
         IF (J .EQ. 30) GO TO 1000
         J = J + 1
C     .......... FORM SHIFT ..........
         G = (D(L+1) - P) / (2.0D0 * E(L))
cccc         R = PYTHAG(G,1.0D0)
         r=sqrt(g**2+1)
         G = D(M) - P + E(L) / (G + DSIGN(R,G))
         S = 1.0D0
         C = 1.0D0
         P = 0.0D0
         MML = M - L
C     .......... FOR I=M-1 STEP -1 UNTIL L DO -- ..........
         DO 200 II = 1, MML
            I = M - II
            F = S * E(I)
            B = C * E(I)
cccc            R = PYTHAG(F,G)
            r=sqrt(f**2+g**2)
            E(I+1) = R
            IF (R .EQ. 0.0D0) GO TO 210
            S = F / R
            C = G / R
            G = D(I+1) - P
            R = (D(I) - G) * S + 2.0D0 * C * B
            P = S * R
            D(I+1) = G + P
            G = C * R - B
  200    CONTINUE
C 
         D(L) = D(L) - P
         E(L) = G
         E(M) = 0.0D0
         GO TO 105
C     .......... RECOVER FROM UNDERFLOW ..........
  210    D(I+1) = D(I+1) - P
         E(M) = 0.0D0
         GO TO 105
C     .......... ORDER EIGENVALUES ..........
  215    IF (L .EQ. 1) GO TO 250
C     .......... FOR I=L STEP -1 UNTIL 2 DO -- ..........
         DO 230 II = 2, L
            I = L + 2 - II
            IF (P .GE. D(I-1)) GO TO 270
            D(I) = D(I-1)
  230    CONTINUE
C 
  250    I = 1
  270    D(I) = P
  290 CONTINUE
C 
      GO TO 1001
C     .......... SET ERROR -- NO CONVERGENCE TO AN
C                EIGENVALUE AFTER 30 ITERATIONS ..........
 1000 IERR = L
 1001 RETURN
      END
c 
c 
c 
c 
c 
        subroutine prolarrm(x,y,n)
        implicit real *8 (a-h,o-z)
        save
        dimension x(1),y(1)
c 
        do 1200 i=1,n
        y(i)=x(i)
 1200 continue
        return
        end
  
  
  
c 
c 
c 
c 
c 
        subroutine prolnofc(logeps,c,nout)
        implicit real *8 (a-h,o-z)
c 
c        This subroutine finds an integer nout such that
c 
c             rlams2_c(nout) \sim 10^{-logeps},                    (1)
c 
c        with logeps, c two user-specified parameters. The intended
c        use of this subroutine is for finding "the dimensionality"
c        of the space of band-limited functions on the interval [-1,1],
c        with the band-limit c and precision 10^{-logeps/2}.
c    IMPORTANT NOTE: PLEASE NOTE THAT IN (1), WE HAVE RLAMS2(NOUT),
C    AS OPPOSED TO RLAMS(NOUT). THUS, THE PEREMITTED VALUES OF
C    LOGEPS ARE BETWEEN 1 AND 36, AND 10^{-logeps}, IS THE SQUARE OF
C    PRECISION, NOT THE PRECISION ITSELF. Also, note that the subroutine
c        tends to overestimate nout; for small c, the overestimation
c        is significant; when both c and logeps are small, the
c        overestimation becomes dramatic.
c 
c                        Input parameters:
c 
c  logeps - the exponent in (1); permitted values: 1,2,...,36
c  c - the band-limit; should be greater than 10 or so.
c 
c                        Output parameters:
c 
c  nout - the smallest integer satisfying (1). ATTENTION: this subroutine
c        will work only if nout < 2000. DANGER!!!!!!!!
c 
c        . . . for this c, and log10 of accuracy (actually, 2*log10
c              of accuracy), find the index n
c 
        save
        n1=1
        n2=2000
c 
        call prolcofn(logeps,n1,c1)
        call prolcofn(logeps,n2,c2)
c 
        do 1600 i=1,20
c 
        n3=(n1+n2)/2
        call prolcofn(logeps,n3,c3)
c 
        if(c3 .ge. c) c2=c3
        if(c3 .ge. c) n2=n3
c 
        if(c3 .lt. c) c1=c3
        if(c3 .lt. c) n1=n3
 1600 continue
c 
        nout=n3+1
c 
        return
        end
  
  
c 
c 
c 
c 
        subroutine prolcofn(logeps,n7,cout)
        implicit real *8 (a-h,o-z)
c 
c        This subroutine finds a real c such that
c 
c             rlams2_c(n7) \sim 10^{-logeps},                    (1)
c 
c        with logeps, n7 two user-specified parameters. The intended
c        use of this subroutine is for finding "the dimensionality"
c        of the space of band-limited functions on the interval [-1,1],
c        with the band-limit c and precision 10^{-logeps/2}; all of its
c        uses so far have been as an inverse of the subroutine prolnofc
c        (see).
c    IMPORTANT NOTE: PLEASE NOTE THAT IN (1), WE HAVE RLAMS2(NOUT),
C    AS OPPOSED TO RLAMS(NOUT). THUS, THE PERMITTED VALUES OF
C    LOGEPS ARE BETWEEN 1 AND 36, AND 10^{-logeps}, IS THE SQUARE OF
C    PRECISION, NOT THE PRECISION ITSELF.
c 
c                        Input parameters:
c 
c  logeps - the exponent in (1); permitted values: 1,2,...,36
c  n7 - the parameter in (1); should be greater than 10.
c 
c                        Output parameters:
c 
c  cout - the band-limit satisfying (1) above
c 
        save
        dimension coefs1(28),coefs2(28),coefs3(28),coefs4(28),
     1      coefs5(28),coefs6(28),coefs7(28),coefs8(28),
     2      coefs9(28),coefs10(28),coefs11(28),coefs12(28),
     3      coefs13(28),coefs14(28),coefs15(28),coefs16(28),
     4      coefs17(28),coefs18(28),coefs19(28),coefs20(28),
     5      coefs21(28),coefs22(28),coefs23(28),coefs24(28),
     6      coefs25(28),coefs26(28),coefs27(28),coefs28(28),
     7      coefs29(28),coefs30(28),coefs31(28),coefs32(28),
     8      coefs33(28),coefs34(28),coefs35(28),coefs36(28)
c 
c    In this run, epslog= -.10000E+01
c    the starting interval is [10,20]
c 
        data coefs1/
     1      0.156287E+01,-.312850E+01,0.568021E+01,-.120211E+02,
     2      0.156685E+01,-.337314E+01,0.107863E+02,-.480865E+02,
     3      0.156882E+01,-.361703E+01,0.209804E+02,-.192335E+03,
     4      0.156981E+01,-.386005E+01,0.412976E+02,-.767473E+03,
     5      0.157030E+01,-.410283E+01,0.819040E+02,-.306735E+04,
     6      0.157055E+01,-.434539E+01,0.163043E+03,-.122591E+05,
     7      0.157067E+01,-.458788E+01,0.325291E+03,-.490228E+05/
c 
c    In this run, epslog= -.20000E+01
c    the starting interval is [10,20]
c 
        data coefs2/
     1      0.155413E+01,-.533444E+01,0.126530E+02,-.256127E+02,
     2      0.156251E+01,-.584991E+01,0.234108E+02,-.101590E+03,
     3      0.156667E+01,-.636243E+01,0.448304E+02,-.404643E+03,
     4      0.156873E+01,-.687203E+01,0.874304E+02,-.161046E+04,
     5      0.156977E+01,-.738048E+01,0.172467E+03,-.642660E+04,
     6      0.157028E+01,-.788811E+01,0.342275E+03,-.256627E+05,
     7      0.157054E+01,-.839543E+01,0.681708E+03,-.102574E+06/
c 
c    In this run, epslog= -.30000E+01
c    the starting interval is [10,20]
c 
        data coefs3/
     1      0.154565E+01,-.722106E+01,0.200977E+02,-.395438E+02,
     2      0.155830E+01,-.799928E+01,0.363409E+02,-.154273E+03,
     3      0.156458E+01,-.877276E+01,0.686645E+02,-.611563E+03,
     4      0.156769E+01,-.954059E+01,0.132846E+03,-.242813E+04,
     5      0.156925E+01,-.103058E+02,0.260824E+03,-.967601E+04,
     6      0.157002E+01,-.110694E+02,0.516225E+03,-.386076E+05,
     7      0.157041E+01,-.118321E+02,0.102659E+04,-.154248E+06/
c 
c    In this run, epslog= -.40000E+01
c    the starting interval is [10,20]
c 
        data coefs4/
     1      0.153720E+01,-.895970E+01,0.282222E+02,-.547875E+02,
     2      0.155409E+01,-.999914E+01,0.499247E+02,-.208124E+03,
     3      0.156249E+01,-.110342E+02,0.931771E+02,-.819997E+03,
     4      0.156666E+01,-.120604E+02,0.178953E+03,-.324765E+04,
     5      0.156873E+01,-.130822E+02,0.349824E+03,-.129243E+05,
     6      0.156976E+01,-.141010E+02,0.690626E+03,-.515292E+05,
     7      0.157028E+01,-.151186E+02,0.137143E+04,-.205787E+06/
c 
c    In this run, epslog= -.50000E+01
c    the starting interval is [10,20]
c 
        data coefs5/
     1      0.152878E+01,-.105992E+02,0.370676E+02,-.719802E+02,
     2      0.154986E+01,-.118975E+02,0.641871E+02,-.263671E+03,
     3      0.156040E+01,-.131951E+02,0.118409E+03,-.103073E+04,
     4      0.156562E+01,-.144807E+02,0.225855E+03,-.407152E+04,
     5      0.156821E+01,-.157594E+02,0.439701E+03,-.161815E+05,
     6      0.156951E+01,-.170339E+02,0.865980E+03,-.644680E+05,
     7      0.157015E+01,-.183062E+02,0.171727E+04,-.257355E+06/
c 
c    In this run, epslog= -.60000E+01
c    the starting interval is [10,20]
c 
        data coefs6/
     1      0.152041E+01,-.121629E+02,0.466486E+02,-.916887E+02,
     2      0.154563E+01,-.137171E+02,0.791291E+02,-.321374E+03,
     3      0.155830E+01,-.152778E+02,0.144348E+03,-.124402E+04,
     4      0.156458E+01,-.168236E+02,0.273536E+03,-.490000E+04,
     5      0.156769E+01,-.183599E+02,0.530451E+03,-.194485E+05,
     6      0.156925E+01,-.198902E+02,0.104231E+04,-.774277E+05,
     7      0.157002E+01,-.214176E+02,0.206420E+04,-.308966E+06/
c 
c    In this run, epslog= -.70000E+01
c    the starting interval is [10,20]
c 
        data coefs7/
     1      0.151206E+01,-.136641E+02,0.569578E+02,-.114370E+03,
     2      0.154139E+01,-.154720E+02,0.947540E+02,-.381729E+03,
     3      0.155620E+01,-.172959E+02,0.170978E+03,-.146015E+04,
     4      0.156353E+01,-.191027E+02,0.321976E+03,-.573317E+04,
     5      0.156718E+01,-.208972E+02,0.622049E+03,-.227251E+05,
     6      0.156899E+01,-.226837E+02,0.121959E+04,-.904083E+05,
     7      0.156989E+01,-.244663E+02,0.241218E+04,-.360621E+06/
c 
c    In this run, epslog= -.80000E+01
c    the starting interval is [10,20]
c 
        data coefs8/
     1      0.150370E+01,-.151101E+02,0.679607E+02,-.140326E+03,
     2      0.153715E+01,-.171721E+02,0.111068E+03,-.445260E+03,
     3      0.155409E+01,-.192589E+02,0.198292E+03,-.167943E+04,
     4      0.156249E+01,-.213274E+02,0.371156E+03,-.657114E+04,
     5      0.156666E+01,-.233806E+02,0.714475E+03,-.260113E+05,
     6      0.156873E+01,-.254236E+02,0.139779E+04,-.103410E+06,
     7      0.156976E+01,-.274615E+02,0.276120E+04,-.412321E+06/
c 
c    In this run, epslog= -.90000E+01
c    the starting interval is [10,20]
c 
        data coefs9/
     1      0.149525E+01,-.165043E+02,0.795941E+02,-.169679E+03,
     2      0.153291E+01,-.188245E+02,0.128077E+03,-.512496E+03,
     3      0.155198E+01,-.211737E+02,0.226281E+03,-.190222E+04,
     4      0.156144E+01,-.235045E+02,0.421062E+03,-.741403E+04,
     5      0.156614E+01,-.258169E+02,0.807709E+03,-.293071E+05,
     6      0.156847E+01,-.281167E+02,0.157691E+04,-.116431E+06,
     7      0.156964E+01,-.304101E+02,0.311123E+04,-.464065E+06/
c 
c    In this run, epslog= -.10000E+02
c    the starting interval is [10,20]
c 
        data coefs10/
     1      0.148663E+01,-.178471E+02,0.917680E+02,-.202376E+03,
     2      0.152868E+01,-.204350E+02,0.145785E+03,-.583957E+03,
     3      0.154986E+01,-.230459E+02,0.254944E+03,-.212892E+04,
     4      0.156040E+01,-.256394E+02,0.471683E+03,-.826203E+04,
     5      0.156562E+01,-.282115E+02,0.901738E+03,-.326123E+05,
     6      0.156821E+01,-.307684E+02,0.175691E+04,-.129473E+06,
     7      0.156951E+01,-.333175E+02,0.346226E+04,-.515852E+06/
c 
c    In this run, epslog= -.11000E+02
c    the starting interval is [10,20]
c 
        data coefs11/
     1      0.147770E+01,-.191366E+02,0.104371E+03,-.238203E+03,
     2      0.152445E+01,-.220078E+02,0.164195E+03,-.660128E+03,
     3      0.154775E+01,-.248796E+02,0.284279E+03,-.235994E+04,
     4      0.155935E+01,-.277363E+02,0.523008E+03,-.911534E+04,
     5      0.156510E+01,-.305685E+02,0.996546E+03,-.359271E+05,
     6      0.156795E+01,-.333828E+02,0.193779E+04,-.142535E+06,
     7      0.156938E+01,-.361877E+02,0.381427E+04,-.567682E+06/
c 
c    In this run, epslog= -.12000E+02
c    the starting interval is [10,20]
c 
        data coefs12/
     1      0.146837E+01,-.203698E+02,0.117276E+03,-.276815E+03,
     2      0.152022E+01,-.235465E+02,0.183304E+03,-.741451E+03,
     3      0.154563E+01,-.266785E+02,0.314285E+03,-.259574E+04,
     4      0.155830E+01,-.297985E+02,0.575029E+03,-.997420E+04,
     5      0.156458E+01,-.328913E+02,0.109212E+04,-.392514E+05,
     6      0.156769E+01,-.359633E+02,0.211953E+04,-.155616E+06,
     7      0.156925E+01,-.390242E+02,0.416725E+04,-.619555E+06/
c 
c    In this run, epslog= -.13000E+02
c    the starting interval is [10,20]
c 
        data coefs13/
     1      0.145851E+01,-.215427E+02,0.130349E+03,-.317772E+03,
     2      0.151599E+01,-.250536E+02,0.203103E+03,-.828308E+03,
     3      0.154351E+01,-.284454E+02,0.344964E+03,-.283679E+04,
     4      0.155725E+01,-.318291E+02,0.627740E+03,-.108389E+05,
     5      0.156406E+01,-.351828E+02,0.118846E+04,-.425854E+05,
     6      0.156744E+01,-.385128E+02,0.230212E+04,-.168717E+06,
     7      0.156912E+01,-.418298E+02,0.452117E+04,-.671470E+06/
c 
c    In this run, epslog= -.14000E+02
c    the starting interval is [10,20]
c 
        data coefs14/
     1      0.144801E+01,-.226509E+02,0.143451E+03,-.360570E+03,
     2      0.151176E+01,-.265312E+02,0.223578E+03,-.921011E+03,
     3      0.154138E+01,-.301830E+02,0.376317E+03,-.308358E+04,
     4      0.155619E+01,-.338303E+02,0.681135E+03,-.117096E+05,
     5      0.156353E+01,-.374454E+02,0.128554E+04,-.459290E+05,
     6      0.156718E+01,-.410336E+02,0.248555E+04,-.181838E+06,
     7      0.156899E+01,-.446069E+02,0.487603E+04,-.723427E+06/
c 
c    In this run, epslog= -.15000E+02
c    the starting interval is [10,20]
c 
        data coefs15/
     1      0.143676E+01,-.236904E+02,0.156448E+03,-.404671E+03,
     2      0.150752E+01,-.279808E+02,0.244709E+03,-.101980E+04,
     3      0.153926E+01,-.318934E+02,0.408347E+03,-.333659E+04,
     4      0.155514E+01,-.358043E+02,0.735211E+03,-.125868E+05,
     5      0.156301E+01,-.396811E+02,0.138336E+04,-.492824E+05,
     6      0.156692E+01,-.435279E+02,0.266980E+04,-.194978E+06,
     7      0.156886E+01,-.473576E+02,0.523182E+04,-.775425E+06/
c 
c    In this run, epslog= -.16000E+02
c    the starting interval is [10,20]
c 
        data coefs16/
     1      0.142468E+01,-.246574E+02,0.169215E+03,-.449532E+03,
     2      0.150326E+01,-.294034E+02,0.266468E+03,-.112482E+04,
     3      0.153714E+01,-.335784E+02,0.441056E+03,-.359634E+04,
     4      0.155409E+01,-.377529E+02,0.789962E+03,-.134708E+05,
     5      0.156249E+01,-.418917E+02,0.148191E+04,-.526458E+05,
     6      0.156666E+01,-.459972E+02,0.285487E+04,-.208138E+06,
     7      0.156873E+01,-.500836E+02,0.558853E+04,-.827465E+06/
c 
c    In this run, epslog= -.17000E+02
c    the starting interval is [10,20]
c 
        data coefs17/
     1      0.141171E+01,-.255487E+02,0.181634E+03,-.494625E+03,
     2      0.149898E+01,-.307997E+02,0.288823E+03,-.123616E+04,
     3      0.153501E+01,-.352397E+02,0.474446E+03,-.386330E+04,
     4      0.155303E+01,-.396776E+02,0.845387E+03,-.143618E+05,
     5      0.156197E+01,-.440788E+02,0.158119E+04,-.560192E+05,
     6      0.156640E+01,-.484433E+02,0.304075E+04,-.221317E+06,
     7      0.156860E+01,-.527864E+02,0.594615E+04,-.879545E+06/
c 
c    In this run, epslog= -.18000E+02
c    the starting interval is [10,20]
c 
        data coefs18/
     1      0.139779E+01,-.263619E+02,0.193602E+03,-.539448E+03,
     2      0.149466E+01,-.321699E+02,0.311735E+03,-.135380E+04,
     3      0.153289E+01,-.368787E+02,0.508519E+03,-.413799E+04,
     4      0.155198E+01,-.415798E+02,0.901482E+03,-.152603E+05,
     5      0.156144E+01,-.462437E+02,0.168119E+04,-.594028E+05,
     6      0.156614E+01,-.508675E+02,0.322743E+04,-.234515E+06,
     7      0.156847E+01,-.554675E+02,0.630466E+04,-.931667E+06/
c 
c    In this run, epslog= -.19000E+02
c    the starting interval is [10,20]
c 
        data coefs19/
     1      0.138290E+01,-.270952E+02,0.205028E+03,-.583541E+03,
     2      0.149029E+01,-.335141E+02,0.335157E+03,-.147768E+04,
     3      0.153076E+01,-.384966E+02,0.543275E+03,-.442087E+04,
     4      0.155092E+01,-.434608E+02,0.958247E+03,-.161667E+05,
     5      0.156092E+01,-.483876E+02,0.178190E+04,-.627968E+05,
     6      0.156588E+01,-.532709E+02,0.341490E+04,-.247732E+06,
     7      0.156834E+01,-.581279E+02,0.666406E+04,-.983829E+06/
c 
c    In this run, epslog= -.20000E+02
c    the starting interval is [10,20]
c 
        data coefs20/
     1      0.136701E+01,-.277478E+02,0.215838E+03,-.626489E+03,
     2      0.148586E+01,-.348320E+02,0.359041E+03,-.160762E+04,
     3      0.152864E+01,-.400947E+02,0.578717E+03,-.471241E+04,
     4      0.154986E+01,-.453217E+02,0.101568E+04,-.170813E+05,
     5      0.156040E+01,-.505116E+02,0.188332E+04,-.662014E+05,
     6      0.156562E+01,-.556546E+02,0.360316E+04,-.260969E+06,
     7      0.156821E+01,-.607689E+02,0.702435E+04,-.103603E+07/
c 
c    In this run, epslog= -.21000E+02
c    the starting interval is [10,20]
c 
        data coefs21/
     1      0.135013E+01,-.283193E+02,0.225968E+03,-.667927E+03,
     2      0.148136E+01,-.361232E+02,0.383332E+03,-.174341E+04,
     3      0.152652E+01,-.416739E+02,0.614844E+03,-.501308E+04,
     4      0.154880E+01,-.471635E+02,0.107378E+04,-.180046E+05,
     5      0.155987E+01,-.526167E+02,0.198544E+04,-.696168E+05,
     6      0.156536E+01,-.580197E+02,0.379219E+04,-.274225E+06,
     7      0.156808E+01,-.633913E+02,0.738551E+04,-.108827E+07/
c 
c    In this run, epslog= -.22000E+02
c    the starting interval is [10,20]
c 
        data coefs22/
     1      0.133228E+01,-.288103E+02,0.235370E+03,-.707541E+03,
     2      0.147677E+01,-.373871E+02,0.407973E+03,-.188476E+04,
     3      0.152439E+01,-.432352E+02,0.651654E+03,-.532329E+04,
     4      0.154774E+01,-.489871E+02,0.113255E+04,-.189370E+05,
     5      0.155935E+01,-.547038E+02,0.208826E+04,-.730432E+05,
     6      0.156510E+01,-.603670E+02,0.398200E+04,-.287500E+06,
     7      0.156795E+01,-.659961E+02,0.774753E+04,-.114056E+07/
c 
c    In this run, epslog= -.23000E+02
c    the starting interval is [10,20]
c 
        data coefs23/
     1      0.131348E+01,-.292218E+02,0.244006E+03,-.745067E+03,
     2      0.147207E+01,-.386228E+02,0.432903E+03,-.203134E+04,
     3      0.152227E+01,-.447792E+02,0.689144E+03,-.564345E+04,
     4      0.154668E+01,-.507933E+02,0.119198E+04,-.198790E+05,
     5      0.155882E+01,-.567737E+02,0.219178E+04,-.764809E+05,
     6      0.156484E+01,-.626973E+02,0.417258E+04,-.300795E+06,
     7      0.156782E+01,-.685841E+02,0.811042E+04,-.119288E+07/
c 
c    In this run, epslog= -.24000E+02
c    the starting interval is [10,20]
c 
        data coefs24/
     1      0.129377E+01,-.295556E+02,0.251852E+03,-.780289E+03,
     2      0.146727E+01,-.398296E+02,0.458059E+03,-.218277E+04,
     3      0.152015E+01,-.463066E+02,0.727312E+03,-.597394E+04,
     4      0.154562E+01,-.525829E+02,0.125209E+04,-.208309E+05,
     5      0.155830E+01,-.588271E+02,0.229598E+04,-.799301E+05,
     6      0.156458E+01,-.650114E+02,0.436392E+04,-.314108E+06,
     7      0.156769E+01,-.711559E+02,0.847416E+04,-.124524E+07/
c 
c    In this run, epslog= -.25000E+02
c    the starting interval is [10,20]
c 
        data coefs25/
     1      0.127320E+01,-.298136E+02,0.258891E+03,-.813036E+03,
     2      0.146233E+01,-.410066E+02,0.483379E+03,-.233862E+04,
     3      0.151802E+01,-.478182E+02,0.766153E+03,-.631511E+04,
     4      0.154456E+01,-.543566E+02,0.131286E+04,-.217933E+05,
     5      0.155777E+01,-.608646E+02,0.240088E+04,-.833911E+05,
     6      0.156432E+01,-.673098E+02,0.455602E+04,-.327442E+06,
     7      0.156757E+01,-.737123E+02,0.883875E+04,-.129764E+07/
c 
c    In this run, epslog= -.26000E+02
c    the starting interval is [10,20]
c 
        data coefs26/
     1      0.125183E+01,-.299985E+02,0.265119E+03,-.843179E+03,
     2      0.145726E+01,-.421528E+02,0.508797E+03,-.249845E+04,
     3      0.151590E+01,-.493144E+02,0.805659E+03,-.666727E+04,
     4      0.154350E+01,-.561150E+02,0.137429E+04,-.227667E+05,
     5      0.155725E+01,-.628871E+02,0.250646E+04,-.868641E+05,
     6      0.156406E+01,-.695933E+02,0.474887E+04,-.340795E+06,
     7      0.156744E+01,-.762538E+02,0.920418E+04,-.135009E+07/
c 
c    In this run, epslog= -.27000E+02
c    the starting interval is [10,20]
c 
        data coefs27/
     1      0.122970E+01,-.301130E+02,0.270537E+03,-.870629E+03,
     2      0.145203E+01,-.432673E+02,0.534250E+03,-.266178E+04,
     3      0.151377E+01,-.507958E+02,0.845823E+03,-.703071E+04,
     4      0.154244E+01,-.578588E+02,0.143640E+04,-.237514E+05,
     5      0.155672E+01,-.648949E+02,0.261273E+04,-.903494E+05,
     6      0.156379E+01,-.718623E+02,0.494247E+04,-.354167E+06,
     7      0.156731E+01,-.787810E+02,0.957045E+04,-.140257E+07/
c 
c    In this run, epslog= -.28000E+02
c    the starting interval is [10,20]
c 
        data coefs28/
     1      0.120689E+01,-.301603E+02,0.275155E+03,-.895329E+03,
     2      0.144663E+01,-.443491E+02,0.559673E+03,-.282813E+04,
     3      0.151164E+01,-.522626E+02,0.886636E+03,-.740566E+04,
     4      0.154137E+01,-.595884E+02,0.149918E+04,-.247480E+05,
     5      0.155620E+01,-.668886E+02,0.271967E+04,-.938474E+05,
     6      0.156353E+01,-.741175E+02,0.513681E+04,-.367559E+06,
     7      0.156718E+01,-.812945E+02,0.993755E+04,-.145509E+07/
c 
c    In this run, epslog= -.29000E+02
c    the starting interval is [10,20]
c 
        data coefs29/
     1      0.118346E+01,-.301437E+02,0.278987E+03,-.917254E+03,
     2      0.144105E+01,-.453972E+02,0.585005E+03,-.299699E+04,
     3      0.150951E+01,-.537153E+02,0.928087E+03,-.779234E+04,
     4      0.154031E+01,-.613044E+02,0.156263E+04,-.257569E+05,
     5      0.155567E+01,-.688687E+02,0.282729E+04,-.973583E+05,
     6      0.156327E+01,-.763592E+02,0.533189E+04,-.380971E+06,
     7      0.156705E+01,-.837947E+02,0.103055E+05,-.150765E+07/
c 
c    In this run, epslog= -.30000E+02
c    the starting interval is [10,20]
c 
        data coefs30/
     1      0.115948E+01,-.300666E+02,0.282054E+03,-.936408E+03,
     2      0.143529E+01,-.464109E+02,0.610185E+03,-.316787E+04,
     3      0.150737E+01,-.551541E+02,0.970164E+03,-.819092E+04,
     4      0.153925E+01,-.630073E+02,0.162674E+04,-.267787E+05,
     5      0.155514E+01,-.708358E+02,0.293559E+04,-.100882E+06,
     6      0.156301E+01,-.785881E+02,0.552770E+04,-.394403E+06,
     7      0.156692E+01,-.862822E+02,0.106742E+05,-.156024E+07/
c 
c    In this run, epslog= -.31000E+02
c    the starting interval is [10,20]
c 
        data coefs31/
     1      0.113501E+01,-.299327E+02,0.284380E+03,-.952815E+03,
     2      0.142933E+01,-.473892E+02,0.635155E+03,-.334025E+04,
     3      0.150523E+01,-.565793E+02,0.101285E+04,-.860154E+04,
     4      0.153819E+01,-.646975E+02,0.169154E+04,-.278137E+05,
     5      0.155462E+01,-.727901E+02,0.304457E+04,-.104420E+06,
     6      0.156275E+01,-.808044E+02,0.572425E+04,-.407855E+06,
     7      0.156679E+01,-.887572E+02,0.110438E+05,-.161288E+07/
c 
c    In this run, epslog= -.32000E+02
c    the starting interval is [10,20]
c 
        data coefs32/
     1      0.111013E+01,-.297456E+02,0.285994E+03,-.966524E+03,
     2      0.142316E+01,-.483315E+02,0.659856E+03,-.351365E+04,
     3      0.150308E+01,-.579911E+02,0.105614E+04,-.902428E+04,
     4      0.153712E+01,-.663754E+02,0.175700E+04,-.288626E+05,
     5      0.155409E+01,-.747322E+02,0.315422E+04,-.107972E+06,
     6      0.156249E+01,-.830085E+02,0.592152E+04,-.421327E+06,
     7      0.156666E+01,-.912202E+02,0.114142E+05,-.166556E+07/
c 
c    In this run, epslog= -.33000E+02
c    the starting interval is [10,20]
c 
        data coefs33/
     1      0.108491E+01,-.295089E+02,0.286926E+03,-.977599E+03,
     2      0.141678E+01,-.492370E+02,0.684237E+03,-.368758E+04,
     3      0.150093E+01,-.593896E+02,0.110000E+04,-.945921E+04,
     4      0.153606E+01,-.680415E+02,0.182314E+04,-.299257E+05,
     5      0.155356E+01,-.766623E+02,0.326454E+04,-.111538E+06,
     6      0.156223E+01,-.852010E+02,0.611952E+04,-.434820E+06,
     7      0.156653E+01,-.936716E+02,0.117854E+05,-.171827E+07/
c 
c    In this run, epslog= -.34000E+02
c    the starting interval is [10,20]
c 
        data coefs34/
     1      0.105941E+01,-.292263E+02,0.287209E+03,-.986119E+03,
     2      0.141018E+01,-.501051E+02,0.708247E+03,-.386155E+04,
     3      0.149876E+01,-.607749E+02,0.114443E+04,-.990635E+04,
     4      0.153500E+01,-.696961E+02,0.188995E+04,-.310036E+05,
     5      0.155303E+01,-.785809E+02,0.337553E+04,-.115119E+06,
     6      0.156197E+01,-.873820E+02,0.631824E+04,-.448333E+06,
     7      0.156640E+01,-.961118E+02,0.121574E+05,-.177103E+07/
c 
c    In this run, epslog= -.35000E+02
c    the starting interval is [10,20]
c 
        data coefs35/
     1      0.103369E+01,-.289014E+02,0.286878E+03,-.992177E+03,
     2      0.140336E+01,-.509353E+02,0.731837E+03,-.403511E+04,
     3      0.149659E+01,-.621471E+02,0.118939E+04,-.103657E+05,
     4      0.153393E+01,-.713395E+02,0.195744E+04,-.320967E+05,
     5      0.155250E+01,-.804883E+02,0.348719E+04,-.118714E+06,
     6      0.156171E+01,-.895519E+02,0.651768E+04,-.461866E+06,
     7      0.156627E+01,-.985410E+02,0.125301E+05,-.182382E+07/
c 
c    In this run, epslog= -.36000E+02
c    the starting interval is [10,20]
c 
        data coefs36/
     1      0.100783E+01,-.285379E+02,0.285969E+03,-.995874E+03,
     2      0.139630E+01,-.517272E+02,0.754962E+03,-.420782E+04,
     3      0.149441E+01,-.635062E+02,0.123488E+04,-.108371E+05,
     4      0.153287E+01,-.729720E+02,0.202561E+04,-.332054E+05,
     5      0.155198E+01,-.823848E+02,0.359951E+04,-.122325E+06,
     6      0.156144E+01,-.917111E+02,0.671783E+04,-.475420E+06,
     7      0.156614E+01,-.100960E+03,0.129037E+05,-.187665E+06/
c 
c           for the user-supplied n, construct the c for which
c           rlams2(n((c) = 10^logeps
c 
        n=n7+1
c 
        if(logeps .eq. 1) call prolcsev(n,coefs1,cout)
        if(logeps .eq. 2) call prolcsev(n,coefs2,cout)
        if(logeps .eq. 3) call prolcsev(n,coefs3,cout)
        if(logeps .eq. 4) call prolcsev(n,coefs4,cout)
        if(logeps .eq. 5) call prolcsev(n,coefs5,cout)
        if(logeps .eq. 6) call prolcsev(n,coefs6,cout)
        if(logeps .eq. 7) call prolcsev(n,coefs7,cout)
        if(logeps .eq. 8) call prolcsev(n,coefs8,cout)
        if(logeps .eq. 9) call prolcsev(n,coefs9,cout)
        if(logeps .eq. 10) call prolcsev(n,coefs10,cout)
c 
        if(logeps .eq. 11) call prolcsev(n,coefs11,cout)
        if(logeps .eq. 12) call prolcsev(n,coefs12,cout)
        if(logeps .eq. 13) call prolcsev(n,coefs13,cout)
        if(logeps .eq. 14) call prolcsev(n,coefs14,cout)
        if(logeps .eq. 15) call prolcsev(n,coefs15,cout)
        if(logeps .eq. 16) call prolcsev(n,coefs16,cout)
        if(logeps .eq. 17) call prolcsev(n,coefs17,cout)
        if(logeps .eq. 18) call prolcsev(n,coefs18,cout)
        if(logeps .eq. 19) call prolcsev(n,coefs19,cout)
        if(logeps .eq. 20) call prolcsev(n,coefs20,cout)
c 
        if(logeps .eq. 21) call prolcsev(n,coefs21,cout)
        if(logeps .eq. 22) call prolcsev(n,coefs22,cout)
        if(logeps .eq. 23) call prolcsev(n,coefs23,cout)
        if(logeps .eq. 24) call prolcsev(n,coefs24,cout)
        if(logeps .eq. 25) call prolcsev(n,coefs25,cout)
        if(logeps .eq. 26) call prolcsev(n,coefs26,cout)
        if(logeps .eq. 27) call prolcsev(n,coefs27,cout)
        if(logeps .eq. 28) call prolcsev(n,coefs28,cout)
        if(logeps .eq. 29) call prolcsev(n,coefs29,cout)
        if(logeps .eq. 30) call prolcsev(n,coefs30,cout)
c 
        if(logeps .eq. 31) call prolcsev(n,coefs31,cout)
        if(logeps .eq. 32) call prolcsev(n,coefs32,cout)
        if(logeps .eq. 33) call prolcsev(n,coefs33,cout)
        if(logeps .eq. 34) call prolcsev(n,coefs34,cout)
        if(logeps .eq. 35) call prolcsev(n,coefs35,cout)
        if(logeps .eq. 36) call prolcsev(n,coefs36,cout)
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prolcsev(n,coefs,cout)
        implicit real *8 (a-h,o-z)
c 
        save
        dimension coefs(4,7)
c 
c       find the location in array coefs of the coefficients
c       corresponding to the user-supplied n
c 
        n1=5
        n2=10
        ii=0
c 
        do 1200 i=1,7
c 
        n1=n2
        n2=n2*2
  
        if( (n2 .ge. n) .and. (n1 .le. n) ) ii=i
 1200 continue
c 
cccc        call prinf('in prolcsev, ii=*',ii,1)
  
        i=ii
        if(ii .eq. 0) i=7
        d=n
        cout=coefs(1,i)*d+coefs(2,i)+coefs(3,i)/d+
     1      coefs(4,i)/d**2
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine protoleg(coefprol,nprol,w,coeflege,nlege)
        implicit real *8 (a-h,o-z)
        save
        dimension w(1),coeflege(1),coefprol(1),coefs(1000)
c 
c        this subroutine converts a prolate expansion of a function
c        on the interval [-1,1] into a Legendre expansion. It uses as
c        input the array w, constructed by a prior call to the
c        subroutine prolcrea (see). This subroutine has no use as a
c        stand-alone device.
c 
c                    Input parameters:
c 
c  coefprol - the coefficients of the prolate expansion to be
c        converted into the Legenedre expansion
c  nprol - the number of coefficients in the prolate expansion (note that
c        this is THE NUMBER OF coefficients. The highest order of the
c        prolate function to be added up is nprol-1
c  w - the array created by the preceding call to the subroutine prolcrea
c 
c                    Output parameters:
c 
c  coefslege - coefficients of the Legendre expansion ahose values on
c        the interval [-1,1] are close (within 1.0d-15 or so) to the
c        values of the user-supplied prolate expansion
c  nlege - the number of coefficients in the expansion coefslege that
c        are greater than 1.0d-15
c 
c 
c       . . . extract from array w various parameters describing
c             the situation
c 
        nhigh=w(4)
        iwork=w(5)
c 
c       construct the Legendre coefficients of the obtained
c       pattern and of the source that generated it
c 
        do 4200 i=1,nhigh
        coeflege(i)=0
 4200 continue
c 
        half=0.5d0
        do 4600 i=1,nprol
c 
        call prolunpk(i-1,w,coefs,ndum)
c 
        do 4400 j=1,ndum
c 
        coeflege(j)=coeflege(j)+coefprol(i)*coefs(j)
c 
 4400 continue
 4600 continue
c 
c        if the need be, truncate the arrays pattcoe, srccoe
c 
        eps=1.0d-15
        nlege=0
c 
        do 4800 i=1,nhigh
c 
        if(abs(coeflege(i)) .gt. eps) nlege=i
 4800 continue
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine cprotole(coefprol,nprol,w,coeflege,nlege)
        implicit real *8 (a-h,o-z)
        save
        dimension w(1),coefs(1000)
        complex *16 coeflege(1),coefprol(1)
c 
c        this subroutine converts a prolate expansion of a function
c        on the interval [-1,1] into a Legendre expansion. It uses as
c        input the array w, constructed by a prior call to the
c        subroutine prolcrea (see). This subroutine has no use as a
c        stand-alone device.
c 
c                    Input parameters:
c 
c  coefprol - the coefficients of the prolate expansion to be
c        converted into the Legenedre expansion
c  nprol - the number of coefficients in the prolate expansion (note that
c        this is THE NUMBER OF coefficients. The highest order of the
c        prolate function to be added up is nprol-1
c  w - the array created by the preceding call to the subroutine prolcrea
c 
c                    Output parameters:
c 
c  coefslege - coefficients of the Legendre expansion ahose values on
c        the interval [-1,1] are close (within 1.0d-15 or so) to the
c        values of the user-supplied prolate expansion
c  nlege - the number of coefficients in the expansion coefslege that
c        are greater than 1.0d-15
c 
c 
c       . . . extract from array w various parameters describing
c             the situation
c 
        nhigh=w(4)
        iwork=w(5)
c 
c       construct the Legendre coefficients of the obtained
c       pattern and of the source that generated it
c 
        do 4200 i=1,nhigh
        coeflege(i)=0
 4200 continue
c 
        half=0.5d0
        do 4600 i=1,nprol
c 
        call prolunpk(i-1,w,coefs,ndum)
c 
        do 4400 j=1,ndum
c 
        coeflege(j)=coeflege(j)+coefprol(i)*coefs(j)
c 
 4400 continue
 4600 continue
c 
c        if the need be, truncate the arrays pattcoe, srccoe
c 
        eps=1.0d-15
        nlege=0
c 
        do 4800 i=1,nhigh
c 
        if(abs(coeflege(i)) .gt. eps) nlege=i
 4800 continue
c 
        return
        end
c 
c 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c       This is the end of the testing code and the beginning of the
c       prolate expansion code proper.
c 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c 
        subroutine prolexps(ier,c,eps,ifrefine,n,xs,ws,u,v,
     1      w,lenw,keep,lused)
        implicit real *8 (a-h,o-z)
        save
        dimension xs(1),ws(1),u(1),v(1),w(1)
c 
c       This subroutine constructs the matrix u, converting the
c       coefficients of a prolate expansion on the interval [-1,1]
c       into the values of the said expansion at prolate nodes, and
c       the matrix v, converting the values of a prolate expansion
c       at the prolate nodes on the interval [-1,1] into the
c       coefficients of the said expansion. It
c       also produces the prolate nodes and corresponding weights
c       on the interval [-1,1]. In addition, the array w generated by
c       this subroutine can be used by the subroutine proleva and
c       its relatives for the evaluation of prolate functions, their
c       derivatives, etc.
c 
c                       Input parameters:
c 
c  c - the band-limit to which the prolate functions are related
c  eps - thye precision to which the generated prolate nodes and
c       weights will correspond
c  ifrefine - the parameter telling the subroutine which type of
c       prolate nodes and weights to generate:
c    ifrefine=0 will cause the nodes based on the division theorem
c       to be generated (i.e. the resulting noides will be th eroots
c       of the appropriately chosen prolate function)
c    ifrefine=1 will cause the optimal prolate nodes to be generated.
c       PLEASE NOTE THAT THE TWO SETS OF NODES ARE VERY CLOSE TO EACH
c       OTHER. AS LONG AS THEY ARE USED STRICTLY FOR INTERPOLATION,
c       THERE IS NO SERIOUS ADVANTAGE FOR EITHER SET. HOWEVER, WHEN
c       USED AS QUADRATURE NODES, THE OPTIMAL NODES GAIN A COUPLE
c       OF DIGITS ON THE ONES PRODUCED WITH IFREFINE=0; on the other
c       hand, ifinit=1 causes a much greater expenditure of CPU time
c       than ifinit=0.
c  lenw - the amount of storage space provided in the array lw; must
c       be sufficiently large; if it is too small, the error code ier
c       is set to 4, and the execution of the subroutine is terminated.
c 
c                          output parameters:
c 
c  ier - error return code.
c         ier=0 means successful execution.
c         ier=4 means that the amount of storage space provided in
c                 the array store is insufficient (see input
c                 parameter lenw above). This is a fatal error.
c  xs - the prolate nodes
c  ws - the corresponding quadrature weights
c  u - the matrix converting the n coefficients of a prolate expansion
c       on the interval [-1,1] into the values of the said expansion
c       at prolate nodes.
c  v - the matrix converting the values of a prolate expansion
c       at prolate nodes into  the n coefficients of the said prolate
c       expansion. Needless to say, u,v are inverses of each other.
c  w - the storage area where information is stored that used by the
c       subroutine proleva and its relatives for the evaluation of
c       prolate functions, their derivatives, etc.
c 
c  keep - the number of elements in the array w that have to be
c       unchanged between the call to this subroutine and the subsequent
c       calls (if any) to the subroutine proleva and its relatives (see)
c  lused - the number of elements of array w that have been used by
c       this subroutine (it is somewhat greater than keep)
c 
        c2=c*2
c 
c       allocate memory for the construction of interpolation nodes
c 
        ier=0
c 
        irlams3=1
        lrlams3=(200+c*2)*2
c 
        irlams32=irlams3+lrlams3
        lrlams32=200+c*2
c 
        iw=irlams32+lrlams32
        lenw2=lenw-iw
c 
c       construct prolate interpolation nodes and weights for this c
c 
        eps22=eps**2
        ifeven=1
        call proquadr(jer,c2,eps22,ifrefine,ifeven,npts,xs,ws,
     1      w(irlams3),nvects,w(irlams32),nvects2,w(iw),lenw2,lused)
c 
        if(jer .ne. 0) ier=16
        if(ier .ne. 0) return
c 
c      construct the matrix converting the coefficients of a prolate
c      expansion into its values at the prolate nodes, and its inverse
c 
        call prolexp0(jer,c,w,lenw,u,v,npts,xs,ws,keep,lused2)
c 
        if(lused .lt. lused2) lused=lused2
c 
        if(jer .ne. 0) ier=8
        n=npts
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prolexp0(ier,c,w,lenw,amatr,bmatr,npts,xs,ws,
     1      keep,lused)
        implicit real *8 (a-h,o-z)
c 
        save
        dimension xs(1),ws(1),amatr(npts,1),bmatr(npts,1),w(1)
c 
c       allocate memory for the subroutine prolcrea
c 
        ikhi=1
        lkhi=200+c*2
c 
        irlams=ikhi+lkhi
        lrlams=c*4+400
c 
        irlams2=irlams+lrlams
        lrlams2=c*2+200
c 
        iw=irlams2+lrlams2
        lenw2=lenw-iw
c 
c       construct the prolate functions
c 
        call prolcrea(ier,c,w(iw),lenw2,nvects,nhigh,
     1    w(ikhi),w(irlams),w(irlams2),keep,lused)
c 
        lused=lused+iw
c 
cccc        call prin2('after prolcrea w(ikhi) are*',w(ikhi),nvects)
cccc        call prin2('after prolcrea, w(irlams2)=*',w(irlams2),nvects)
cccc        call prin2('after prolcrea, w(irlams)=*',w(irlams),nvects*2)
c 
        call prinf('after prolcrea, ier=*',ier,1)
cccc        call prinf('nhigh=*',nhigh,1)
cccc        call prinf('keep=*',keep,1)
cccc        call prinf('lused=*',lused,1)
cccc        call prinf('nvects=*',nvects,1)
c 
        if(ier .ne. 0) return
c 
c       perform garbage collection
c 
        call prolarrm(w(iw),w,keep)
c 
c       construct the matrix of values of prolate functions at the
c       nodes of the prolate discretization
c 
        do 1100 i=1,npts
        do 1050 j=1,npts
c 
        call prolevv(j-1,xs(i),w,val)
c 
        amatr(i,j)=val
        bmatr(i,j)=val
 1050 continue
 1100 continue
c 
c       invert the matrix of values of prolate functions at the
c       nodes of the prolate discretization
c 
        iwork=keep+2
        ltot=keep+npts*(npts+1)+2
c 
        if(ltot .lt. lenw) goto 1150
c 
        ier=4
        return
 1150 continue
c 
        call orthom(bmatr,npts,w(iwork),cond)
c 
        call prin2('in prolexp0 after orthom, cond=*',cond,1)
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prolc18(ndigits,c)
        implicit real *8 (a-h,o-z)
        save
        real *8 cs(18)
        data cs/
     1   0.426214940813475E+01,0.698617652394456E+01,
     2   0.953915223960300E+01,0.120241941287835E+02,
     3   0.144712250599186E+02,0.168936898731331E+02,
     4   0.192988865939722E+02,0.216912469241496E+02,
     5   0.240736739904003E+02,0.264481773450600E+02,
     6   0.288162077394038E+02,0.311788478769652E+02,
     7   0.335369278460025E+02,0.358910984171198E+02,
     8   0.382418794595321E+02,0.405896929947373E+02,
     9   0.429348864032829E+02,0.452777491155961E+02/
c 
c       given the user-specified integer 0 < ndigits < 19,
c       this subroutine returns the value c such that
c 
c              \psi_0^c (1)=0.1^{ndigits}
c 
         c=cs(ndigits)
c 
        return
        end
  
c 
c 
c 
c 
c 
        subroutine prolc180(eps,c)
        implicit real *8 (a-h,o-z)
        save
        real *8 cs(180)
        data cs/
     1  .43368E-16,.10048E+01,.17298E+01,.22271E+01,.26382E+01,
     2  .30035E+01,.33409E+01,.36598E+01,.39658E+01,.42621E+01,
     3  .45513E+01,.48347E+01,.51136E+01,.53887E+01,.56606E+01,
     4  .59299E+01,.61968E+01,.64616E+01,.67247E+01,.69862E+01,
     5  .72462E+01,.75049E+01,.77625E+01,.80189E+01,.82744E+01,
     6  .85289E+01,.87826E+01,.90355E+01,.92877E+01,.95392E+01,
     7  .97900E+01,.10040E+02,.10290E+02,.10539E+02,.10788E+02,
     8  .11036E+02,.11284E+02,.11531E+02,.11778E+02,.12024E+02,
     9  .12270E+02,.12516E+02,.12762E+02,.13007E+02,.13251E+02,
     *  .13496E+02,.13740E+02,.13984E+02,.14228E+02,.14471E+02,
     1  .14714E+02,.14957E+02,.15200E+02,.15443E+02,.15685E+02,
     2  .15927E+02,.16169E+02,.16411E+02,.16652E+02,.16894E+02,
     3  .17135E+02,.17376E+02,.17617E+02,.17858E+02,.18098E+02,
     4  .18339E+02,.18579E+02,.18819E+02,.19059E+02,.19299E+02,
     5  .19539E+02,.19778E+02,.20018E+02,.20257E+02,.20496E+02,
     6  .20736E+02,.20975E+02,.21214E+02,.21452E+02,.21691E+02,
     7  .21930E+02,.22168E+02,.22407E+02,.22645E+02,.22884E+02,
     8  .23122E+02,.23360E+02,.23598E+02,.23836E+02,.24074E+02,
     9  .24311E+02,.24549E+02,.24787E+02,.25024E+02,.25262E+02,
     *  .25499E+02,.25737E+02,.25974E+02,.26211E+02,.26448E+02,
     1  .26685E+02,.26922E+02,.27159E+02,.27396E+02,.27633E+02,
     2  .27870E+02,.28106E+02,.28343E+02,.28580E+02,.28816E+02,
     3  .29053E+02,.29289E+02,.29526E+02,.29762E+02,.29998E+02,
     4  .30234E+02,.30471E+02,.30707E+02,.30943E+02,.31179E+02,
     5  .31415E+02,.31651E+02,.31887E+02,.32123E+02,.32358E+02,
     6  .32594E+02,.32830E+02,.33066E+02,.33301E+02,.33537E+02,
     7  .33773E+02,.34008E+02,.34244E+02,.34479E+02,.34714E+02,
     8  .34950E+02,.35185E+02,.35421E+02,.35656E+02,.35891E+02,
     9  .36126E+02,.36362E+02,.36597E+02,.36832E+02,.37067E+02,
     *  .37302E+02,.37537E+02,.37772E+02,.38007E+02,.38242E+02,
     1  .38477E+02,.38712E+02,.38947E+02,.39181E+02,.39416E+02,
     2  .39651E+02,.39886E+02,.40120E+02,.40355E+02,.40590E+02,
     3  .40824E+02,.41059E+02,.41294E+02,.41528E+02,.41763E+02,
     4  .41997E+02,.42232E+02,.42466E+02,.42700E+02,.42935E+02,
     5  .43169E+02,.43404E+02,.43638E+02,.43872E+02,.44107E+02,
     6  .44341E+02,.44575E+02,.44809E+02,.45044E+02,.45278E+02/
c 
c       given the user-specified real 1.0E-18 < eps 0.1,
c       this subroutine returns the value c such that
c 
c              \psi_0^c (1) \sim eps
c 
        d=-log10(eps)
        i=d*10+0.1
        c=cs(i)
c 
        return
        end

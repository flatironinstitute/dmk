cccc        implicit real *8 (a-h,o-z)
cccc        dimension xs(10 000),ws(10 000),rlams3(10 000),diffs(10 000),
cccc     1      w(40 000 000),rlams32(10 000)
ccccc 
cccc        call prini(6,13)
ccccC 
ccccC       SET ALL PARAMETERS
ccccC 
cccc        PRINT *, 'ENTER c'
cccc        READ *,c
cccc        CALL PRIN2('c=*',c,1 )
ccccC 
ccccC 
cccc        PRINT *, 'ENTER -log_10(eps)'
cccc        READ *,d
cccc  
cccc        eps=10.0**(-d)
cccc        CALL PRIN2('eps=*',eps,1 )
cccc  
cccc  
ccccc       find the eigenvectors
ccccc 
ccccc 
cccc        lenw=40 000 000
cccc        lenw=1 000 000
cccc        ifrefine=1
cccc        ifeven=1
cccc  
cccc        time1=clotatim()
cccc  
cccc  
cccc        call proquadr(ier,c,eps,ifrefine,ifeven,npts,xs,ws,
cccc     1      rlams3,nvects,rlams32,nvects2,w,lenw,lused)
cccc  
cccc        time2=clotatim()
cccc  
cccc  
cccc        call prin2('and time for proquadr is*',time2-time1,1)
cccc  
cccccccc        stop
cccc  
cccc        call prinf('and after proquadr, npts=*',npts,1)
cccc        call prinf('and after proquadr, lused=*',lused,1)
cccc        call prinf('and after proquadr, ier=*',ier,1)
cccc        call prin2('and xs=*',xs,npts)
cccc        call prin2('and ws=*',ws,npts)
cccc  
cccc        if(ier .ne. 0) stop
ccccc 
cccc        call prin2('after proquadr, rlams32 =*',rlams32,nvects2)
cccc  
cccc  
cccc        do 1200 i=1,npts-1
ccccc 
cccc        diffs(i)=xs(i+1)-xs(i)
cccc 1200 continue
cccc  
cccc        call prin2('and distances between nodes are*',diffs,npts-1)
ccccc 
ccccc       test the obtained quadrature
ccccc 
cccc        ntest=1000
cccc        cc=c
cccc        call quatest(cc,ntest,xs,ws,npts,errmax)
ccccc 
ccccc       store the obtained quadrature on disk
ccccc 
cccc        iw=17
cccc  
cccc        call quadwrt(iw,xs,ws,npts,c,eps,rlams3(2*npts+1),errmax)
cccc  
cccc
cccc        rat=diffs(npts/2)/diffs(1)
cccc
cccc        call prin2('and rat=*',rat,1)  
cccc
cccc        done=1
cccc        call proleva(npts*2,done,w,val,der)
cccccccc        call prolevv(1,done,w,val,der)
cccc
cccc        call prin2('and val=*',val,1)
cccc        call prin2('and der=*',der,1)
cccc
cccc
cccc        call prin2('and xs(1)+1=*',xs(1)+1,1)
cccc
cccc  
cccc        stop
cccc        end
ccccc 
ccccc 
ccccc 
ccccc 
ccccc 
cccc        subroutine quadwrt(iw,roots,whts,npts,c,eps,rlam,errmax)
cccc        implicit real *8 (a-h,o-z)
cccc        save
cccc        dimension roots(1),whts(1)
ccccc 
ccccc       store on disk the nodes of the quadrature
ccccc 
cccc        numwrt=(npts+1)/2
cccc  
cccc        call prinf('in quadwrt, numwrt=*',numwrt,1)
cccc        call prinf('in quadwrt, npts=*',npts,1)
cccc  
cccc        nlines=(numwrt+1)/2
ccccc 
cccc  
cccc        call prinf('in quadwrt, nlines=*',nlines,1)
cccc  
cccc  
cccc 1050 format(10x,/,'c     ','In this run, c=',e22.16,', ',
cccc     1    'eps=',e22.16,',',/,'c     ','npts=',i6,
cccc     2    ' and expected error of quadrature is ',e11.5,/,
cccc     3    'c      and maximum error of quadrature is ',e11.5)
cccc  
cccc 1060 format(10x,/,'c     ','Please note that only the first ',
cccc     1  'half of the nodes ',/,
cccc     2    'c     (those that are \leq 0) are output',/,10x)
cccc  
cccc        write(iw,1050) c,eps,npts,rlam,errmax
cccc        write(iw,1060)
cccc  
cccc  
cccc 1100 format('c        Nodes are',/,10x)
cccc        write(iw,1100)
cccc  
cccc        do 1400 i=1,nlines
ccccc 
cccc        ii=i
cccc        j=i/10
cccc        j=i-j*10
cccc        if(j .eq. 0) j=11
cccc  
cccc        if(i*2 .le. numwrt)
cccc     1      write(iw,1200) j,roots(i*2-1),roots(i*2)
ccccc 
cccc        if(i*2 .gt. numwrt)
cccc     1      write(iw,1200) j,roots(i*2-1)
ccccc 
cccc 1200 format(5x,i1,2x,e22.16,',',e22.16,',')
cccc 1400 continue
ccccc 
cccc  
cccc       call prin2('whts=*',whts,npts)
cccc  
cccc 2100 format(10x,/,'c        Weights are',/,10x)
cccc        write(iw,2100)
cccc  
cccc        do 2400 i=1,nlines
ccccc 
cccc        ii=i
cccc        j=i/10
cccc        j=i-j*10
cccc        if(j .eq. 0) j=11
cccc  
cccc        if(i*2 .le. numwrt)
cccc     1       write(iw,2200) j,whts(i*2-1),whts(i*2)
ccccc 
cccc        if(i*2 .gt. numwrt)
cccc     1      write(iw,2200) j,whts(i*2-1)
ccccc 
cccc 2200 format(5x,i1,2x,e22.16,',',e22.16,',')
cccc 2400 continue
ccccc 
cccc        return
cccc        end
ccccc 
ccccc 
ccccc 
ccccc 
ccccc 
cccc        subroutine quatest(c,ntest,xs,ws,npts,errmax)
cccc        implicit real *8 (a-h,o-z)
cccc        save
cccc        complex *16 ima,cints(10 000),cints2(10 000)
cccc        dimension cs(10 000),xs(1),ws(1),diffs(10 000)
ccccc 
cccc        data ima/(0.0d0,1.0d0)/
ccccc 
ccccc       discretize the space of exponetials
ccccc 
cccc        hcs=c/ntest
cccc        do 1200 i=1,ntest
ccccc 
cccc        cs(i)=i*hcs
cccc 1200 continue
ccccc 
cccccccccc        call prin2('cs=*',cs,ntest)
ccccc 
ccccc       one exponential after another, evaluate their integrals
ccccc       on the interval [-1,1] analytically
ccccc 
cccc        do 1600 i=1,ntest
ccccc 
cccc        cints(i)=1/(ima*cs(i))*(exp(ima*cs(i))-exp(-ima*cs(i)))
cccc 1600 continue
ccccc 
cccccccc        call prin2('integrals evaluated analytically*',
cccccccc     1      cints,ntest*2)
ccccc 
ccccc       evaluate the integrals numerically
ccccc 
cccc        errmax=0
cccc        do 2400 i=1,ntest
ccccc 
cccc        cd=0
cccc        do 2200 j=1,npts
cccc        cd=cd+ws(j)*exp(ima*cs(i)*xs(j))
cccc 2200 continue
ccccc 
cccc        cints2(i)=cd
cccc        diffs(i)=abs(cints2(i)-cints(i))
cccc  
cccc        if(errmax .lt. diffs(i)) errmax=diffs(i)
cccc 2400 continue
ccccc 
cccccccc        call prin2('and integrals evaluated numerically*',
cccccccc     1      cints2,ntest*2)
ccccc 
cccc  
cccccccc        call prin2('and absolute values of differences are*',
cccccccc     1         diffs,ntest)
cccc  
cccc        call prin2('and errmax=*',errmax,1)
cccc  
cccc        return
cccc        end
c 
c 
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c       This is the end of the debugging and output code and the
c       beginning of the quadrature code proper
c 
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c 
c 
        subroutine proquadr(ier,c,eps,ifrefine,ifeven,npts,
     1      xs,ws,rlams3,nvects,rlams32,nvects2,w,lenw,lused)
        implicit real *8 (a-h,o-z)
        save
        real *8 w(1),khi(10 000),rlams2(10 000),rlams3(1),
     1      rints(10 000),xs(1),ws(1),rlams32(1),errs(10 000),
     2      roots(10 000)
c 
        complex *16 rlams(10 000)
c 
c       This subroutine constructs quadratures for band-limited functions
c       on the interval [-1,1]. Specifically, it constructs an npts-point
c       quadrature integrating exactly the first npts*2 prolate functions
c       corresponding to the user-specified band-limit c; npts is chosen
c       to be the smallest integer such that the prolate eigenvalue
c       number npts*2+1 is less than eps; this guarantees (more or less)
c       that any function of the form
c 
c       exp(i \cdot u \cdot t)                                            (1)
c 
c       with |u|< c
c 
c       is integrated on the interval [-1,1] with precision better than
c       eps.
c 
c       Please note that the subroutine gobbles up order n^3 operations,
c       and the proportionality coefficient is large; the largest number
c       of nodes actually obtained with this subroutine is about 500.
c       Thus, there is no serious memory management. The user supplies
c       the array w to be used to store all arrays that are of the order
c       npts^2 in size. The spaces for all arrays that are of the order
c       npts are set to 10 000 - rough and ready.
c 
c 
c                             Input parameters:
c 
c  c - the band-limit
c  eps - the precision of the quadrature to be constructed
c  ifrefine - an integer aparmeter telling the subroutine which of the
c       two procedures it should use. Setting ifrefine=0 will cause the
c       subroutine to use the scheme based on the division (Euclid)
c       algoritrhm for the prolate functions, and loses about three
c       digits compared to the user-specified eps; in other words, setting
c       eps=1.0d-10 will produce a roughly 7-digit quadrature. Setting
c       ifrefine=1 will cause the subroutine to use the full-dress
c       optimization scheme, and produce the full requested accuracy;
c       plese note that this parameter does not affect the selection of the
c       number of nodes. THE REASON FOR THE EXISTENCE OF THIS parameter
c       is the fact that setting ifrefine=0 results in a dtamatically
c       faster execution, and that when used for interpolation (as opposed
c       to quadrature), the resulting nodes are equally good.
c  ifeven - an integer parameter telling the subroutine whether to insist
c       on the number of nodes being even or odd.
c    ifeven=-1 will cause the subroutine to choose an odd number of nodes
c    ifeven=1 will cause the subroutine to choose an even number of nodes
c    ifeevn=0 will cause the subroutine to choose the number of nodes
c       that produces the requested precision, without regard to the
c       evenness or oddness of the resulting number of nodes
c  lenw - the length of the work array w supplied to the subroutine
c 
c 
c                             Output parameters:
c 
c  ier - error return code
c           ier=0 means successful conclusion
c           ier=16 means that the memory in array w turned out to
c                 be insufficient during first call to the subroutine
c                 prolcrea evaluating the prolate functions. Normally,
c                 it means that the length of the array w (as specified
c                 by the parameter lenw) is grossly insufficient.
c           ier=8 means that the subroutine ran out of memory at a later
c                 stage (most probably, during the second call to prolcrea)
c 
c           ier=512 means that the Newton process used to find
c                 the nodes and weights failed to converge after
c                 20 iterations. This has never happened, and probably
c                 means serious programming trouble
c           ier=1024 means bad programming trouble
c  npts - the number of nodes in the constructed quadrature
c  xs - the npts nodes
c  ws - the npts weights corresponding to the nodes xs
c  rlams3 - the absolute values of the prolate eigenvalues corresponding
c       to the user-specified parameter c
c  nvects - the number of eigenvalues stored in the array rlams3
c  rlams32 - the absolute values of the prolate eigenvalues corresponding
c       to the user-specified parameter c/2; these are expected to be
c       useful if and when the nodes are used for the interpolation
c       (as opposed to integration)
c  nvects2 - the number of eigenvalues stored in the array rlams32
c  lused - the amount of storage in array w (in real *8 words) actually
c       used by the subroutine
c 
        ier=0
c 
c       construct the prolate functions
c 
        call prolcrea(jer,c,w,lenw,nvects,nhigh,
     1    khi,rlams,rlams2,keep,lused1)
c 
        if(jer .eq. 0) goto 2200
c 
        call prinf('after prolcrea, jer=*',jer,1)
        ier=16
        return
 2200 continue
c 
        iw2=keep+10
        lenw2=lenw-iw2-2
c 
c       select npts for this c
c 
        do 2400 i=1,nvects
        rlams3(i)=abs(rlams(i))
c 
        if(rlams3(i)/rlams3(1) .gt. eps) npts=i
 2400 continue
c 
        npts=(npts+1)/2
  
        call prinf('initial npts is*',npts,1)
  
        j=npts/2
        j=npts-j*2
  
        if( (ifeven .eq. -1) .and. (j .eq. 0)) npts=npts+1
        if( (ifeven .eq. 1) .and. (j .eq. 1)) npts=npts+1
  
        call prinf(' npts after adjustment is*',npts,1)
c 
        call prinf('and npts as selected=*',npts,1)
c 
c       evaluate the integrals of the requisite prolate functions
c 
        do 2900 i=1,npts*2
c 
        call prolunpk(i-1,w,rlams2,n)
c 
        rints(i)=rlams2(1)*2
 2900 continue
c 
c        construct the initial nodes for the Newton
c 
        c2=c/2
        ifinit=1
        call proqrts(jer,c2,npts,eps,ifinit,w(iw2),lenw2,
     1      lused2,keep2,xs,errs,khi,rlams,rlams2,nvects2)
  
        call prin2('after proqrts, errors are*',errs,npts)
cccc        call prin2('after proqrts, xs=*',xs,npts)
  
        do 2940 i=1,nvects2
        rlams32(i)=abs(rlams(i))
c 
 2940 continue
  
        do 2950 i=1,npts
        errs(i)=roots(i)-xs(i)
 2950 continue
c 
        if(jer .eq. 0) goto 3000
c 
        ier=1024
        if(jer .eq. 16) ier=8
        return
 3000 continue
c 
cccc        call prin2('while errors are*',rlams2,npts)
  
        lused=lused1+lused2
c 
c       construct the initial weights for the Newton
c 
        call prinf('iw2=*',iw2,1)
  
        ibmatr=iw2+keep2+2
        lbmatr=npts**2*4+4
        lused3=ibmatr+npts**2*4+4
        if(lused3 .gt. lused) lused=lused3
        if(lused .lt. lenw) goto 3200
        ier=8
        return
 3200 continue
c 
        nsys=npts/2
        i=npts-nsys*2
        if(i .ne. 0) nsys=nsys+1
c 
        call proqwht1(npts,nsys,xs,ws,w(ibmatr),w(iw2),
     1      rlams,rlams2)
c 
        if(ifrefine .eq. 0) return
c 
c       conduct the Newton iterations
c 
        iww=ibmatr+lbmatr
        lww=lenw-iww
        call prinf('lenw=*',lenw,1)
        call prinf('lww=*',lww,1)
  
cccc        stop
c 
        call proqnewt(jer,w,npts,rints,w(ibmatr),xs,ws,
     1          niter,rlams,rlams2,nhigh,w(iww),lww)
c 
        call prinf('after proqnewt, niter=*',niter,1)
c 
        if(jer .ne. 0) ier=512
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine proqwht1(npts,n,xs,ws,bmatr,w,rhs,coefs77)
        implicit real *8 (a-h,o-z)
        save
        dimension bmatr(n,n),xs(1),w(1),rhs(1),coefs77(1),ws(1)
c 
c       evaluate the integrals of the requisite prolate functions
c 
        do 2900 i=1,n
c 
        jj=(i-1)*2
c 
        call prolunpk(jj,w,coefs77,nnn)
c 
        rhs(i)=coefs77(1)
 2900 continue
c 
c       find the weights corresponding to the nodes xs
c 
        ier=0
c 
        do 3400 i=1,n
        do 3200 j=1,n
c 
        jj=(j-1)*2
        call prolevv(jj,xs(i),w,val,der)
c 
        bmatr(j,i)=val
c 
 3200 continue
c 
 3400 continue
c 
        call qrsolv(bmatr,n,rhs,rcond)
c 
        do 3500 i=1,n
c 
        ws(i)=rhs(i)
 3500 continue
c 
        i=npts/2
        i=npts-i*2
        if(i .ne. 0) ws(n)=ws(n)*2
c 
c       reflect the obtained weights around the point x=0 in
c       order to produce those of them corresponding to the
c       positive rootrs
c 
        do 3600 i=1,n
        ws(npts-i+1)=ws(i)
 3600 continue
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine proqnewt(ier,w,npts,rints,bmatr,xs,ws,
     1      niter,rhs,sol,nhigh,ww,lww)
        implicit real *8 (a-h,o-z)
        save
        dimension bmatr(1),w(1),rints(1),rhs(1),ww(1),
     1      ws(1),xs(1),sol(1)
c 
c 
c       conduct the Newton iterations
c 
        ier=0
        numit=20
        ifout=0
        do 3000 k=1,numit
c 
        niter=k
c 
c       construct the linear system to be solved
c 
        call proqmatr(jer,w,npts,xs,ws,bmatr,rints,rhs,nhigh,
     1      ww,lww)
c 
        if(jer .ne. 0) then
            ier=512
            return
        endif
c 
c       solve the linear system
c 
        call qrsolv(bmatr,npts*2,rhs,rcond)
  
        do 1800 i=1,npts*2
c 
        sol(i)=rhs(i)
 1800 continue
c 
        eps=1.0d-10
        diff=0
        do 2400 i=1,npts
c 
        xs(i)=xs(i)+sol(i+npts)
        ws(i)=ws(i)+sol(i)
c 
        diff=diff+sol(i)**2+sol(npts+i)**2
c 
 2400 continue
c 
        diff=sqrt(diff)
c 
        if(ifout .eq. 1) goto 3200
        if(diff .lt. eps) ifout=1
c 
 3000 continue
c 
        ier=16
 3200 continue
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine proqmatr(ier,w,npts,xs,ws,amatr,rints,rhs,nhigh,
     1      ww,lww)
        implicit real *8 (a-h,o-z)
c 
        save
        dimension amatr(npts*2,npts*2),w(1),xs(1),ws(1),
     1      rhs(1),rints(1),ww(1),funs(10 000),ders(10 000)
c 
c 
c       construct the matrix of the jacobian
c 
        ifunpack=1
c 
c 
        nnn=npts/2
        if(nnn*2 .ne. npts) nnn=nnn+1
cccc        do 1400 i=1,npts/2+1
        do 1400 i=1,nnn
c 
  
  
cccc        do 1400 i=1,npts/2
c 
        call prolderfast(ier,xs(i),w,ifunpack,nhigh,
     1      funs,ders,npts*2,ww,lww,keep)
c 
        if(ier .ne. 0) return
c 
        do 1200 j=1,npts*2
c 
        amatr(j,i)=funs(j)
        amatr(j,npts+i)=ders(j)*ws(i)
c 
 1200 continue
 1400 continue
c 
        do 1800 i=1,npts/2
        d=1
        do 1600 j=1,npts*2
c 
        amatr(j,npts-i+1)=amatr(j,i)* d
        amatr(j,2*npts-i+1)=-amatr(j,npts+i)* d
c 
        d=-d
 1600 continue
 1800 continue
c 
c       construct the right-hand side
c 
        do 2400 i=1,npts*2
        d=0
        do 2200 j=1,npts
c 
        d=d+amatr(i,j)*ws(j)
c 
 2200 continue
        rhs(i)=rints(i)-d
 2400 continue
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine prolderfast(ier,x,w,ifunpack,nhigh,
     1      funs,ders,nvects,ww,lww,keep)
        implicit real *8 (a-h,o-z)
        save
        dimension w(1),funs(1),ders(1),ww(1)
c 
c 
c        This subroutine evaluates at the point x \in [-1,1] the
c        first nvects Prolate Spheroidal wave functions, together
c        with their derivatives.
c 
c 
c 
c       . . . allocate memory
c 
        ier=0
c 
        icoefs=1
        lcoefs=nhigh*nvects+10
c 
        in0s=icoefs+lcoefs
        ln0s=nvects+10
c 
        ins=in0s+ln0s
        lns=nvects+10
c 
        ipols=ins+lns
        lpols=nhigh+10
c 
        ipolders=ipols+lpols
        lpolders=nhigh+10
c 
        keep=ipolders+lpolders
c 
        if(keep .le. lww) goto 2200
c 
        call prinf('bombing from prolderfast, keep > lww, keep=*',
     1      keep,1)
c 
        ier=32
        return
 2200 continue
c 
        call prolderfas(x,w,ifunpack,nhigh,
     1      funs,ders,nvects,
     2      ww(icoefs),ww(in0s),ww(ins),ww(ipols),ww(ipolders) )
c 
        return
        end
  
c 
c 
c 
c 
c 
        subroutine prolderfas(x,w,ifunpack,nhigh,
     1      funs,ders,nvects,coefs,n0s,ns,pols,polders)
        implicit real *8 (a-h,o-z)
        save
        dimension w(1),coefs(nhigh,1),funs(1),ders(1),ns(1),
     1      pols(1),polders(1),n0s(1)
c 
c        if the user so requested, unpack the Legendre expansions
c        of all nvects prolate functions
c 
        if(ifunpack .eq. 0) goto 1300
        do 1200 i=1,nvects
c 
        call prolunpk(i-1,w,coefs(1,i),ns(i) )
c 
        do 1100 j=1,ns(i)
c 
        n0s(i)=j
        if(abs(coefs(j,i)) .gt. 1.0d-30) goto 1150
c 
 1100 continue
 1150 continue
c 
 1200 continue
c 
 1300 continue
c 
c       evaluate the Legendre polynomials at the point x
c 
        call legepolders(X,pols,polders,Nhigh)
c 
c       evaluate all Prolate functions
c 
        do 1600 i=1,nvects
c 
        nn0=n0s(i)
        nn=ns(i)
        dd=0
        d=0
        do 1400 j=nn0,nn,2
c 
        d=d+pols(j)*coefs(j,i)
        dd=dd+polders(j)*coefs(j,i)
 1400 continue
c 
        funs(i)=d
        ders(i)=dd
 1600 continue
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine proqrts(ier,c,n,eps,ifinit,w,lenw,lused,keep,
     1      roots,errs,khi,rlams,rlams2,nvects)
        implicit real *8 (a-h,o-z)
c 
        save
        real *8 khi(1),w(1),rlams2(1),roots(1),errs(1)
        complex *16 rlams(1)
c 
c 
c        This subroutine finds the roots of the user-specified
c        prolate function. The user specifies the bandlimit and
c        the sequence number of the function whose roots are to
c        be found. The sequence number of the function can be
c        specified in one of two ways:
c 
c   a. The user can specify the sequence number n .leq. 1
c   b. The user can set n=0, and the subroutine will find determine
c        n as the smallest integer such that
c 
c        abs(rlams(n))< eps,                                         (1)
c 
c        with eps the user-specified threshold
c 
c        The subroutine can operate in one of two regimes, determined
c        by the user-specified parameter ifinit. Setting ifinit=1
c        will cause the subroutine to initialize the prolate function
c        computer; ifinit has to be set to 1 each time the subroutine
c        is called with a new value of the parameter c (c is the
c        bandlimit). After that, roots ofd additional prolate functions
c        with the same c (specified by different values of n or eps)
c        are best found by calling the subroutine with ifinit=0.
c        Please note that failure to set ifinit=0 when t is called for
c        will cause a dramatic increase in the CPU time requirements of
c        the subroutine.
c 
c                          input parameters:
c 
c  c - the coefficient (real) in the formula (1)
c  n - the sequence number of the prolate function whose roots are
c        to be found (it has n roots). Please note that setting n=0
c        will convert n into an output parameter, to be determined
c        via (1) above and the user-specified eps.
c  eps - the tolerance in (1) above to be used to determine the sequence
c        number n of the prolate function whose roots are to be determined.
c        This parameter is ignored if the user-specified value of n is
c        greater than 0
c  lenw - the amount of storage space provided in the array lw; must
c       be sufficiently large; if it is too small, the error code ier
c       is set to 4 or to 8, and the execution of the subroutine is
c       terminated. Note that this parameter, normally, does not need to
c       be very large.
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
c  n - the sequence number of the prolate function whose roots have been
c        found (it has n roots). Please note that thid is an output
c        parameter only if on input it had been set to zero. Otherwise, it
c        is entirely an input parameter.
c  w - the storage area where all the information is stored to be
c       by subsequent calls to this subroutine with ifinit=0. PLEASE NOTE
c       THAT THIS ARRAY CAN BE used by the subroutine proleva and its
c       relatives (proleval, prolevv, etc.) to evaluate the prolate
c       spheroidal wave functions and their derivatives
c  lused - the number of elements of array w that have been used by
c       this subroutine (it is somewhat greater than keep)
c  keep - the number of elements of the array w that has to be
c       unchanged between the call to this subroutine and the subsequent
c       calls to the subroutine prolev0 (see)
c  roots - the roots of the n-th prolate function (n of them things)
c  errs - the precision to which the n-th prolate function is equal to
c       zero at each of the alleged roots
c  khi - the coefficients in (2) (nvects of them) at which the equation
c       (3) has a non-trivial solution (the "separation coefficients"
c       for the prolate spheroidal wave function).
c  rlams - the eigenvalues (complex) of the operator (1)
c  rlams2 - the eigenvalues of the operator (2)
c 
c 
c       . . . if the initialization is not necessary - bypass it
c 
        if(ifinit .eq. 0) goto 1200
c 
c       construct prolate functions
c 
        call prolcrea(ier,c,w,lenw,nvects,nhigh,
     1    khi,rlams,rlams2,keep,lused)
c 
        call prinf('ier=*',ier,1)
        call prinf('nhigh=*',nhigh,1)
        call prinf('keep=*',keep,1)
        call prinf('lused=*',lused,1)
        call prinf('nvects=*',nvects,1)
c 
 1200 continue
c 
c       is the user-specified n is equal to zero, set it to the
c       sequence number of the prolate function whose corresponding
c       eigenvalue is the first one less than eps
c 
        if(n .gt. 0) goto 1600
c 
        do 1400 i=1,nvects
c 
        d=abs(rlams(i))
        if(d .gt. eps) goto 1400
        n=i
        goto 1600
 1400 continue
 1600 continue
c 
c       find the roots of the user-specified prolate function
c       if its sequence number is even
c 
        done=1
        pi=atan(done)*4
        m=20
c 
        i=n/2
        ifodd=n-i*2
c 
        if (ifodd .eq. 1) goto 3000
c 
        a=0
        b=pi/2
        ts0=a
c 
        call oneroot(a,b,m,khi(n+1),ts0,tsm,c)
        call newt3stp(n,tsm,w,roots(1),errs(1))
c 
        do 2400 i=1,n/2-1
c 
        a=b
        b=b+pi
        ts0=tsm
c 
        call oneroot(a,b,m,khi(n+1),ts0,tsm,c)
        call newt3stp(n,tsm,w,roots(i+1),errs(i+1))
 2400 continue
c 
c       reverse the obtained roots and reflect them around x=0
c       to obtain the second n/2 roots
c 
        do 2600 i=1,n/4
        d=roots(n/2-i+1)
        roots(n/2-i+1)=roots(i)
        roots(i)=d
c 
        d=errs(n/2-i+1)
        errs(n/2-i+1)=errs(i)
        errs(i)=d
 2600 continue
c 
        do 2800 i=1,n/2
        roots(n/2+i)=-roots(n/2-i+1)
        errs(n/2+i)=errs(n/2-i+1)
 2800 continue
c 
 3000 continue
c 
        if(ifodd .eq. 0) return
c 
        tsm=0
c 
c       find the roots of the user-specified prolate function
c       if its sequence number is odd
c 
        b=pi/2
c 
        do 3400 i=1,n/2
c 
        a=b
        b=b+pi
        ts0=tsm
c 
        roots(n/2+1)=0
        errs(n/2+1)=0
c 
        call oneroot(a,b,m,khi(n+1),ts0,tsm,c)
        call newt3stp(n,tsm,w,roots(i),errs(i))
 3400 continue
c 
c       reverse the obtained roots and reflect them around x=0
c       to obtain the second n/2 roots
c 
        do 3600 i=1,n/4
        d=roots(n/2-i+1)
        roots(n/2-i+1)=roots(i)
        roots(i)=d
c 
        d=errs(n/2-i+1)
        errs(n/2-i+1)=errs(i)
        errs(i)=d
 3600 continue
c 
        do 3800 i=1,n/2
        roots(n/2+i+1)=-roots(n/2-i+1)
        errs(n/2+i+1)=errs(n/2-i+1)
 3800 continue
c 
        return
        end
c 
c 
c 
c 
c 
        subroutine oneroot(a,b,m,rkhi,ts0,tsm,c)
        implicit real *8 (a-h,o-z)
        save
        dimension ts(100),thetas(100)
c 
        ts(1)=ts0
        thetas(1)=a
c 
        h=(b-a)/(m-1)
  
        do 1400 i=1,m
c 
        thetas(i+1)=thetas(i)+h
c 
        p=1-ts(i)**2
        pder=-2*ts(i)
c 
        q=rkhi-c**2*ts(i)**2
        qder=-2*c**2*ts(i)
c 
        if(i .ne. 1) goto 1200
  
        denom=sqrt(q/p)+(qder*p+pder*q)/p/q/4*sin(2*thetas(i))
c 
        der=-1/denom
  
 1200 continue
  
        ts(i+1)=ts(i)+h*der
c 
        p=1-ts(i+1)**2
        pder=-2*ts(i+1)
c 
        q=rkhi-c**2*ts(i+1)**2
        qder=-2*c**2*ts(i+1)
        denom2=sqrt(q/p)+(qder*p+pder*q)/p/q/4*sin(2*thetas(i+1))
c 
        der2=-1/denom2
c 
        ts(i+1)=ts(i)+h/2*(der+der2)
        der=der2
 1400 continue
c 
        tsm=ts(m)
  
        return
        end
c 
c 
c 
c 
c 
        subroutine newt3stp(n,tsm,w,root,err)
        implicit real *8 (a-h,o-z)
        save
        dimension w(1)
c 
        call proleval(n,tsm,w,d,der )
        tsm=tsm-d/der
        call proleval(n,tsm,w,d,der )
        tsm=tsm-d/der
        call proleval(n,tsm,w,d,der )
        tsm=tsm-d/der
        call proleval(n,tsm,w,d,der )
        tsm=tsm-d/der
c 
        root=tsm
        err=abs(d/der)
        return
        end

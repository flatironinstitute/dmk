# Laplace 3D monopole & dipole free-space direct summation kernel module.
# pure python and numba-jit versions (making module keeps jits around; yay).
# Barnett 9/12/18

import numpy as np
from numpy import array,zeros,ones,eye,empty,random
from numpy.linalg import norm,cond
import numba
import numexpr as ne
from time import perf_counter as tic

def lap3dcharge_native(y,q,x,ifgrad=False):
    """evaluate potentials & fields of 3D Laplace charges at non-self targets,
    via naive direct sum. Slow native-python reference implementation.

    Evaluate at the nt (non-self) targets x,

                1   ns-1 
    pot(x_i) = ---  sum  q_j / r_ij                 for i=0,..,nt
               4pi  j=0

    If ifgrad true, also evaluate (where k is the vector component 0,1,2),

                    -1  ns-1
    (grad(x_i))_k = --- sum  q_j (R_ij)_k / r_ij^3
                    4pi j=0

    Here R_ij := x_i - y_j is src-targ displacement, where x_i, y_j in R^3.
    r_ij := |R_ij|, where |.| is the 2-norm in R^3.
    No targets coincident with any source are allowed, ie, no self-interaction.

    Inputs:
    y : ns*3 source locations
    q : ns   source charges
    x : nt*3 target locations
    ifgrad : whether to compute gradients

    Outputs:
    if ifgrad==False:
    pot : ns potential values at targets
    if ifgrad==True:
    tuple of:
    pot : ns potential values at targets
    grad : ns*3 vectors of grad pot (ie negative of E field) at targets

    Barnett 9/11/18.
    """
    y = np.atleast_2d(y)     # handle ns=1 case: make 1x3 not 3-vecs
    x = np.atleast_2d(x)
    ns = y.shape[0]
    nt = x.shape[0]
    prefac = 1.0/(4.0*np.pi)
    pot = zeros(nt)
    if ifgrad:
        grad = zeros([nt,3])
    for j in range(ns):       # loop over sources, ie vectorize over targs...
        R = x - y[j]                   # nt*3
        r2 = np.sum(R**2,axis=1)       # squared dists
        r = np.sqrt(r2)                # dists
        pot += (prefac * q[j]) / r     # contrib from this src
        if ifgrad:
            grad += R * ((-prefac*q[j])/(r*r2))[:,None]
    if ifgrad:
        return pot,grad
    else:
        return pot

def lap3ddipole_native(y,d,x,ifgrad=False):
    """evaluate potentials & fields of 3D Laplace dipoles at non-self targets,
    via naive direct sum, slow native-python reference implementation.

    Evaluate at the nt (non-self) targets x,

                1   ns-1 
    pot(x_i) = ---  sum R_ij.d_j / r_ij^3                 for i=0,..,nt
               4pi  j=0

    If ifgrad true, also evaluate (where k is the vector component 0,1,2),

                     1  ns-1
    (grad(x_i))_k = --- sum [ d_k - 3 (R_ij.d_j) (R_ij)_k / r_ij^2] / r_ij^3
                    4pi j=0

    Here R_ij := x_i - y_j is src-targ displacement vec, where x_i, y_j in R^3.
    r_ij := |R_ij|, where |.| is the 2-norm in R^3.
    No targets coincident with any source are allowed, ie, no self-interaction.
    
    Formulae are the source-directional derivative of the Poisson fundamental
    solution (charge, as in previous routine):  u = 1/(4 pi r).

    Inputs:
    y : ns*3 source locations
    d : ns*3 vector dipole source strengths
    x : nt*3 target locations
    ifgrad : whether to compute gradients

    Outputs:
    if ifgrad==False:
    pot : ns potential values at targets
    if ifgrad==True:
    tuple of:
    pot : ns potential values at targets
    grad : ns*3 vectors of grad pot (ie negative of E field) at targets

    Barnett 9/6/18. flipped from 3*n input ordering for simplicity, but
    this made factor 3 slower unless now use order='F' for inputs.
    """
    y = np.atleast_2d(y)     # handle ns=1 case: make 1x3 not 3-vecs
    d = np.atleast_2d(d)
    x = np.atleast_2d(x)
    ns = y.shape[0]
    nt = x.shape[0]
    prefac = 1.0/(4.0*np.pi)
    pot = zeros(nt)
    if ifgrad:
        grad = zeros([nt,3])
    for j in range(ns):       # loop over sources, ie vectorize over targs...
        R = x - y[j]                   # nt*3
        r2 = np.sum(R**2,axis=1)       # squared dists
        r = np.sqrt(r2)                # dists
        ir3 = 1 / (r*r2)
        srcdip = prefac * d[j]         # 3-el vec
        ddotR = R.dot(srcdip)          # dot = MV prod
        pot += ddotR * ir3             # contrib from this src
        if ifgrad:
            grad += (np.atleast_2d(srcdip) - (3*ddotR/r2)[:,None]*R ) * ir3[:,None]
    if ifgrad:
        return pot,grad
    else:
        return pot

#@numba.njit('void(f8[:,:],f8[:],f8[:,:],f8[:],f8[:,:],b1)',parallel=True,fastmath=True)   # explicit signature, makes it cache? but can't do optional args?
# Also note cache=True fails w/ parallel=True in numba 0.39
@numba.njit(parallel=True,fastmath=True)   # recompiles every run, slow
def lap3dcharge_numba(y,q,x,pot,grad,add=False):
    """evaluate pot & grad of 3D Laplace charges, non-self, naive sum,
    numba jit. Writes into pot and grad.
    See lap3dcharge_native.
    Optional input: add - if True, add to what's in pot,grad; False overwrite.
    pot,grad passed in since njit fails with internal pot=zeros(nt)
    """
    y = np.atleast_2d(y)     # handle ns=1 case: make 1x3 not 3-vecs
    x = np.atleast_2d(x)
    grad = np.atleast_2d(grad)
    ns = y.shape[0]
    nt = x.shape[0]
    assert(pot.shape==(nt,))
    assert(grad.shape==(nt,3))
    prefac = 1.0/(4.0*np.pi)
    for i in numba.prange(nt):    # loop over targs
        if not add:
            pot[i] = grad[i,0] = grad[i,1] = grad[i,2] = 0.0
        for j in range(ns):
            R0 = x[i,0]-y[j,0]
            R1 = x[i,1]-y[j,1]
            R2 = x[i,2]-y[j,2]
            r2 = R0**2+R1**2+R2**2
            r = np.sqrt(r2)
            pqj = prefac*q[j]
            pot[i] += pqj / r
            pqjir3 = pqj / (r*r2)
            grad[i,0] -= R0 * pqjir3
            grad[i,1] -= R1 * pqjir3
            grad[i,2] -= R2 * pqjir3

@numba.njit(parallel=True,fastmath=True)
def lap3ddipole_numba(y,d,x,pot,grad,add=False):
    """evaluate pot & grad of 3D Laplace dipoles, non-self, naive sum,
    numba jit. Writes into pot and grad.
    See lap3ddipole_native.
    Optional input: add - if True, add to what's in pot,grad; False overwrite.
    """
    y = np.atleast_2d(y)     # handle ns=1 case: make 1x3 not 3-vecs
    d = np.atleast_2d(d)
    x = np.atleast_2d(x)
    grad = np.atleast_2d(grad)
    ns = y.shape[0]
    nt = x.shape[0]
    assert(pot.shape==(nt,))
    assert(grad.shape==(nt,3))
    prefac = 1.0/(4.0*np.pi)
    for i in numba.prange(nt):    # loop over targs
        if not add:
            pot[i] = grad[i,0] = grad[i,1] = grad[i,2] = 0.0
        for j in range(ns):
            R0 = x[i,0]-y[j,0]
            R1 = x[i,1]-y[j,1]
            R2 = x[i,2]-y[j,2]
            r2 = R0**2+R1**2+R2**2
            r = np.sqrt(r2)
            ir2 = 1.0/r2
            pir3 = prefac/(r*r2)             # includes prefactor
            ddotR = R0*d[j,0]+R1*d[j,1]+R2*d[j,2]
            pot[i] += ddotR * pir3
            grad[i,0] += (d[j,0] - 3*ddotR*R0*ir2) * pir3
            grad[i,1] += (d[j,1] - 3*ddotR*R1*ir2) * pir3
            grad[i,2] += (d[j,2] - 3*ddotR*R2*ir2) * pir3

@numba.njit(parallel=True,fastmath=True)
def lap3dchargemat_numba(y,x,e,A,An):
    """Fill dense matrix for pot & direc-grad of 3D Laplace charges, non-self.
    numba jit.
    Inputs:
    y - ns*3 source locs
    x - nt*3 target locs
    e - nt*3 target normals (ought to be unit)
    Outputs: (must be preallocated)
    A - nt*ns matrix mapping source charges to target pots
    An - nt*ns matrix mapping source charges to target normal-grads
    See lap3ddipole_native for math definitions.
    """
    y = np.atleast_2d(y)     # handle ns=1 case: make 1x3 not 3-vecs
    x = np.atleast_2d(x)
    e = np.atleast_2d(e)
    ns = y.shape[0]
    nt = x.shape[0]
    assert(A.shape==(nt,ns))
    assert(An.shape==(nt,ns))
    prefac = 1.0/(4.0*np.pi)
    for i in numba.prange(nt):    # outer loop over targs
        for j in range(ns):
            R0 = x[i,0]-y[j,0]
            R1 = x[i,1]-y[j,1]
            R2 = x[i,2]-y[j,2]
            r2 = R0**2+R1**2+R2**2
            r = np.sqrt(r2)
            edotR = R0*e[i,0]+R1*e[i,1]+R2*e[i,2]
            A[i,j] = prefac / r
            An[i,j] = -prefac * edotR / (r2*r)

def lap3dchargemat_ne(y,x,e,A,An):
    """Fill dense matrix for pot & direc-grad of 3D Laplace charges, non-self.
    numexpr.
    Inputs:
    y - ns*3 source locs
    x - nt*3 target locs
    e - nt*3 target normals (ought to be unit)
    Outputs: (must be preallocated, both filled for now)
    A - nt*ns matrix mapping source charges to target pots
    An - nt*ns matrix mapping source charges to target normal-grads
    See lap3ddipole_native for math definitions.
    """
    y = np.atleast_2d(y)     # handle ns=1 case: make 1x3 not 3-vecs
    x = np.atleast_2d(x)
    e = np.atleast_2d(e)
    ns = y.shape[0]
    nt = x.shape[0]
    assert(A.shape==(nt,ns))
    assert(An.shape==(nt,ns))
    x0 = x[:,0][:,None]
    x1 = x[:,1][:,None]
    x2 = x[:,2][:,None]
    n0 = e[:,0][:,None]
    n1 = e[:,1][:,None]
    n2 = e[:,2][:,None]
    y0 = y[:,0]
    y1 = y[:,1]
    y2 = y[:,2]
    d0 = ne.evaluate('x0 - y0')  # outer, displ mats x-y
    d1 = ne.evaluate('x1 - y1')
    d2 = ne.evaluate('x2 - y2')
    ir = ne.evaluate('1/sqrt(d0**2 + d1**2 + d2**2)')
    prefac = 1.0/(4.0*np.pi)
    ne.evaluate('prefac*ir',out=A)    # writes into A, not creates new symbol
    ne.evaluate('-prefac*(d0*n0 + d1*n1 + d2*n2)*(ir*ir*ir)',out=An)

@numba.njit(parallel=True,fastmath=True)
def lap3ddipolemat_numba(y,d,x,e,A,An):
    """Fill dense matrix for pot & direc-grad of 3D Laplace dipoles, non-self.
    numba jit.
    Inputs:
    y - ns*3 source locs
    d - ns*3 src dipole directions (ought to be unit)
    x - nt*3 target locs
    e - nt*3 target normals (ought to be unit)
    Outputs: (must be preallocated)
    A - nt*ns matrix mapping source dipole strengths to target pots
    An - nt*ns matrix mapping source dipole strengths to target normal-grads
    See lap3ddipole_native for math definitions.
    """
    y = np.atleast_2d(y)     # handle ns=1 case: make 1x3 not 3-vecs
    d = np.atleast_2d(d)
    x = np.atleast_2d(x)
    e = np.atleast_2d(e)
    ns = y.shape[0]
    nt = x.shape[0]
    assert(A.shape==(nt,ns))
    assert(An.shape==(nt,ns))
    prefac = 1.0/(4.0*np.pi)
    for i in numba.prange(nt):    # outer loop over targs
        for j in range(ns):
            R0 = x[i,0]-y[j,0]
            R1 = x[i,1]-y[j,1]
            R2 = x[i,2]-y[j,2]
            r2 = R0**2+R1**2+R2**2
            r = np.sqrt(r2)
            pir3 = prefac/(r*r2)             # includes prefactor
            ddotR = R0*d[j,0]+R1*d[j,1]+R2*d[j,2]
            ddote = d[j,0]*e[i,0]+d[j,1]*e[i,1]+d[j,2]*e[i,2]
            edotR = R0*e[i,0]+R1*e[i,1]+R2*e[i,2]
            A[i,j] = ddotR * pir3
            An[i,j] = (ddote - 3*ddotR*edotR/r2) * pir3
            
def lap3ddipolemat_ne(y,d,x,e,A,An):
    """Fill dense matrix for pot & direc-grad of 3D Laplace dipoles, non-self.
    nunexpr.
    Inputs:
    y - ns*3 source locs
    d - ns*3 src dipole directions (ought to be unit)
    x - nt*3 target locs
    e - nt*3 target normals (ought to be unit)
    Outputs: (must be preallocated)
    A - nt*ns matrix mapping source dipole strengths to target pots
    An - nt*ns matrix mapping source dipole strengths to target normal-grads
    See lap3ddipole_native for math definitions.
    """
    y = np.atleast_2d(y)     # handle ns=1 case: make 1x3 not 3-vecs
    d = np.atleast_2d(d)
    x = np.atleast_2d(x)
    e = np.atleast_2d(e)
    ns = y.shape[0]
    nt = x.shape[0]
    assert(A.shape==(nt,ns))
    assert(An.shape==(nt,ns))
    x0 = x[:,0][:,None]       # make col vecs for numexpr
    x1 = x[:,1][:,None]
    x2 = x[:,2][:,None]
    n0 = e[:,0][:,None]
    n1 = e[:,1][:,None]
    n2 = e[:,2][:,None]
    y0 = y[:,0]
    y1 = y[:,1]
    y2 = y[:,2]
    d0 = d[:,0]
    d1 = d[:,1]
    d2 = d[:,2]
    R0 = ne.evaluate('x0 - y0')  # outer, displ mats x-y
    R1 = ne.evaluate('x1 - y1')
    R2 = ne.evaluate('x2 - y2')
    # this is all slow I assume because of passing over RAM too many times...
    ir = ne.evaluate('1/sqrt(R0**2 + R1**2 + R2**2)')
    prefac = 1.0/(4.0*np.pi)
    ddotR = ne.evaluate('R0*d0 + R1*d1 + R2*d2')
    pir3 = ne.evaluate('prefac*(ir*ir*ir)')
    ne.evaluate('ddotR*pir3',out=A)
    ne.evaluate('((d0*n0 + d1*n1 + d2*n2) - 3*(ddotR*(n0*R0+n1*R1+n2*R2))*(ir*ir))*pir3',out=An)

@numba.njit(parallel=True,fastmath=True)   # recompiles every run, slow
def lap3dchargeself_numba(y,q,pot,grad,add=False):
    """evaluate pot & grad of 3D Laplace charges, self (j!=i), naive sum,
    numba jit. Writes into pot and grad.
    See lap3dcharge_native.
    Optional input: add - if True, add to what's in pot,grad; False overwrite.
    pot,grad passed in since njit fails with internal pot=zeros(nt)
    """
    if y.ndim==1:          # n=1, no self-int, no need for atleast_2d
        return
    n = y.shape[0]
    assert(pot.shape==(n,))
    assert(grad.shape==(n,3))
    prefac = 1.0/(4.0*np.pi)
    for i in numba.prange(n):    # loop over targs
        if not add:
            pot[i] = grad[i,0] = grad[i,1] = grad[i,2] = 0.0
        for j in range(n):
            if j!=i:       # same speed as splitting to explicit j<i, j>i cases
                R0 = y[i,0]-y[j,0]
                R1 = y[i,1]-y[j,1]
                R2 = y[i,2]-y[j,2]
                r2 = R0**2+R1**2+R2**2
                r = np.sqrt(r2)
                pqj = prefac*q[j]
                pot[i] += pqj / r
                pqjir3 = pqj / (r*r2)
                grad[i,0] -= R0 * pqjir3
                grad[i,1] -= R1 * pqjir3
                grad[i,2] -= R2 * pqjir3

@numba.njit(parallel=True,fastmath=True)
def lap3ddipoleself_numba(y,d,pot,grad,add=False):
    """evaluate pot & grad of 3D Laplace dipoles, self (j!=i), naive sum,
    numba jit.

    Inputs which are written into:
    pot  float(n) potential at n sources
    grad float(n,3) gradient (negative of E field) at n sources
    Optional input: add - if True, add to what's in pot,grad; False overwrite.

    Definition of pot and grad are as in lap3ddipole_native, omitting j=i term.
 
    Issues: * why is this code 1/2 the speed of lap3ddipole_numba ? (no, it's
    not the i j!=i conditional...).
    """
    if y.ndim==1:          # n=1, no self-int, no need for atleast_2d
        return
    n = y.shape[0]
    assert(pot.shape==(n,))
    assert(grad.shape==(n,3))
    prefac = 1.0/(4.0*np.pi)
    for i in numba.prange(n):    # loop over targs
        if not add:
            pot[i] = grad[i,0] = grad[i,1] = grad[i,2] = 0.0
        for j in range(n):
            if j!=i:       # same speed as splitting to explicit j<i, j>i cases
                R0 = y[i,0]-y[j,0]
                R1 = y[i,1]-y[j,1]
                R2 = y[i,2]-y[j,2]
                r2 = R0**2+R1**2+R2**2
                r = np.sqrt(r2)
                ir2 = 1.0/r2
                pir3 = prefac/(r*r2)             # includes prefactor
                ddotR = R0*d[j,0]+R1*d[j,1]+R2*d[j,2]
                pot[i] += ddotR * pir3
                grad[i,0] += (d[j,0] - 3*ddotR*R0*ir2) * pir3
                grad[i,1] += (d[j,1] - 3*ddotR*R1*ir2) * pir3
                grad[i,2] += (d[j,2] - 3*ddotR*R2*ir2) * pir3

def test_lap3dcharge():
    """ test gradient of pot in lap3dcharge, eval speeds of slow & jit & self.
    Barnett 9/11/18
    """
    x = array([0,1,-1])  # choose a targ
    y = array([1,2,.4]); q = array([2])   # 1 src
    u = 0*q; g = 0*y
    lap3dcharge_numba(y,q,x,u,g) # compile & check jit can be called w/ nt=ns=1
    u = lap3dcharge_native(y,q,x)   # check native can be called w/ nt=ns=1
    # check grad... inline funcs needed (NB 2nd tuple in gradf)
    f = lambda x: lap3dcharge_native(y,q,x.T,ifgrad=False)
    gradf = lambda x: lap3dcharge_native(y,q,x.T,ifgrad=True)[1].ravel()
    print('test_lap3dcharge: grad check (native): ', checkgrad(x,f,gradf))
    # perf tests...
    ns = 2000                    # sources
    nt = 1000                    # targs (check rect case)
    y = random.rand(ns,3)     # sources in [0,1]^3
    q = random.randn(ns)      # charges
    x = random.rand(nt,3)     # targs
    #y=np.asfortranarray(y); x=np.asfortranarray(x); q=np.asfortranarray(q)
    t0=tic()
    u,g = lap3dcharge_native(y,q,x,ifgrad=True)    # native python
    t=tic()-t0
    print("native: %d src-targ pairs in %.3g s: %.3g Gpair/s" % (ns*nt,t,ns*nt/t/1e9))
    u2 = zeros(nt)    # numba version writes outputs to arguments
    g2 = zeros([nt,3])
    lap3dcharge_numba(y,q,x,u2,g2)
    t0=tic()
    lap3dcharge_numba(y,q,x,u2,g2)
    t =tic()-t0
    print("numba:  %d src-targ pairs in %.3g s: %.3g Gpair/s" % (ns*nt,t,ns*nt/t/1e9))
    print("pot err numba vs native:  %.3g"%(np.max(np.abs(u-u2))))
    print("grad err numba vs native: %.3g"%(np.max(np.abs(g-g2))))

def test_lap3ddipole():
    """ test gradient of pot in lap3ddipole, eval speeds of slow & jit & self.
    Barnett 9/5/18
    """
    x = array([0,1,-1])  # choose a targ
    y = array([1,2,.4]); d = array([2,1,3])   # 1 src and dipole strength
    u = array([0]); g = 0*d
    lap3ddipole_numba(y,d,x,u,g)  # compile & check jit can be called w/ nt=ns=1
    u = lap3ddipole_native(y,d,x)   # check native can be called w/ nt=ns=1
    # check grad... inline funcs needed (NB 2nd tuple in gradf)
    f = lambda x: lap3ddipole_native(y,d,x.T,ifgrad=False)
    gradf = lambda x: lap3ddipole_native(y,d,x.T,ifgrad=True)[1].ravel()
    print('test_lap3ddipole: grad check (native): ', checkgrad(x,f,gradf))

    # perf tests...
    ns = 1000                    # sources
    nt = 2000                    # targs (check rect case)
    y = random.rand(ns,3)     # sources in [0,1]^3
    d = random.randn(ns,3)    # strength vectors
    x = random.rand(nt,3)     # targs
    # try swap storage order: (2x speed up for native code, strangely)...
    #y=np.asfortranarray(y); x=np.asfortranarray(x); d=np.asfortranarray(d)
    t0=tic()
    u,g = lap3ddipole_native(y,d,x,ifgrad=True)    # native python
    t =tic()-t0
    print("native: %d src-targ pairs in %.3g s: %.3g Gpair/s" % (ns*nt,t,ns*nt/t/1e9))
    u2 = zeros(nt)    # numba version writes outputs to arguments
    g2 = zeros([nt,3])
    lap3ddipole_numba(y,d,x,u2,g2)   # warm up
    t0=tic()
    lap3ddipole_numba(y,d,x,u2,g2)
    t =tic()-t0
    print("numba:  %d src-targ pairs in %.3g s: %.3g Gpair/s" % (ns*nt,t,ns*nt/t/1e9))
    print("pot err numba vs native:  %.3g"%(np.max(np.abs(u-u2))))
    print("grad err numba vs native: %.3g"%(np.max(np.abs(g-g2))))

    n = 2000                        # sources for self-eval j!=i test
    y = random.rand(n,3)              # in [0,1]^3
    d = random.randn(n,3)
    pot = 0*y[:,0];  grad = 0*y   # allocate output arrays
    lap3ddipoleself_numba(y,d,pot,grad)   # compile to warm-up, 0.3 s!
    t0=tic()
    lap3ddipoleself_numba(y,d,pot,grad)
    t=tic()-t0
    print("numba self: %d src-targ pairs in %.3g s: %.3g Gpair/s" % (n*n,t,n*n/t/1e9))

def test_lap3dmats():
    """test the matrix fillers match the native evaluator answers.
    Also tests timing. Conclusion: numba can fill around 1e9 els/sec but
    numexpr is 4x slower, at least on i7.
    """
    ns = 5000                    # sources
    y = random.rand(ns,3)     # sources in [0,1]^3
    d = random.randn(ns,3)    # strength vectors (ought to be unit len)
    q = random.randn(ns)      # charges
    nt = 10000                    # targs (check rect case)
    x = random.rand(nt,3)     # targs
    e = random.randn(nt,3)    # targ normals (ought to be unit len)
    u = zeros(nt)             # true pot and grad outputs
    g = zeros([nt,3])
    for meth in range(2):
        print("meth=numba:" if meth==0 else "meth=numexpr:")
        # charge (monopole)...
        lap3dcharge_numba(y,q,x,u,g)
        A = zeros([nt,ns]); An = zeros([nt,ns]);  # alloc mats
        t0=tic()
        lap3dchargemat_numba(y,x,e,A,An) if meth==0 else lap3dchargemat_ne(y,x,e,A,An)
        t = tic()-t0
        print("chg mats fill:  two %d*%d mats in %.3g s: %.3g Gels/s" % (nt,ns,t,2*ns*nt/t/1e9))
        t0 = tic()
        ufrommat = A @ q[:,None]
        t = tic()-t0
        print("matvec: %.3g s: %.3g Gops/s" % (t,ns*nt/t/1e9))
        print('chg mat pot err nrm = ', norm(u[:,None] - ufrommat))  # u make col vec!
        gfrommat = An @ q[:,None]
        gdote = np.sum(g*e,axis=1)[:,None]   # e-direc derivs
        print('chg mat n-grad err nrm = ', norm(gdote - gfrommat))
        # dipole...
        lap3ddipole_numba(y,d,x,u,g)
        A = zeros([nt,ns]); An = zeros([nt,ns]);  # alloc mats
        t0=tic()
        lap3ddipolemat_numba(y,d,x,e,A,An) if meth==0 else lap3ddipolemat_ne(y,d,x,e,A,An)
        t = tic()-t0
        print("dip mats fill:  two %d*%d mats in %.3g s: %.3g Gels/s" % (nt,ns,t,2*ns*nt/t/1e9))
        ufrommat = A @ np.ones([ns,1])   # pot from unit dipstrs
        print('dip mat pot err nrm = ', norm(u[:,None] - ufrommat))
        gfrommat = An @ np.ones([ns,1])   # grad from unit dipstrs
        gdote = np.sum(g*e,axis=1)[:,None]   # e-direc derivs
        print('dip mat n-grad err nrm = ', norm(gdote - gfrommat),'\n')

def checkgrad(x,f,Df):
    """
    Check a gradient of any func f:R3->R is correct via finite differencing.

    x is 3-element array.
    f(x) should accept 3*n matrix of n locations, 
      and return f (n values)
    Df(x) should accept 3-element location, and return
      gradient of f (3-element vector)
    Returns: worst-case error found. Should be of order 1e-10.

    Usage: see test_checkgrad()

    Barnett 9/5/18. Note this uses 3*n coord array, not n*3.
    """
    e = 1e-5        # roughly emach^(1/3)
    xx = x[:,None] + e*np.hstack((eye(3),-eye(3)))   # six test pts
    ff = f(xx)
    Dfest = (ff[0:3]-ff[3:6])/(2*e)        # centered diff approx
    return max(np.abs(Df(x) - Dfest))
    
def test_checkgrad():
    """ Tester of the checkgrad util
    """
    v = random.randn(3)            # vec in R3
    f = lambda x: v.dot(x)            # inline func, linear
    gradf = lambda x: v               # inline func, Df indep of x
    x = array([1,2,3])
    #print(f(x),f(np.hstack([x[:,None],x[:,None]])),gradf(x))  # test f, gradf!
    x = random.randn(3)
    print('test checkgrad: ', checkgrad(x,f,gradf))

def warmup():
    """Warm-up (compile) the numba jits for little (n>1 needed) cases
    """
    print('lap3dkernels: wait for numba jit compiles...')
    n=10; y = random.rand(n,3); q=random.rand(n)
    d = 1*y; x=-1*y; g = 0*x; u=0*x[:,0]
    t0=tic()
    lap3dcharge_numba(y,q,x,u,g)
    lap3dcharge_numba(y,q,x,u,g,add=True)  # apparently a separate compile?
    lap3ddipole_numba(y,d,x,u,g)
    lap3ddipole_numba(y,d,x,u,g,add=True)
    lap3ddipoleself_numba(y,d,u,g)
    A = zeros([n,n]); An=0*A;
    lap3dchargemat_numba(y,x,d,A,An)
    lap3ddipolemat_numba(y,d,x,d,A,An)
    print('all jit compiles %.3g s'%(tic()-t0))
    
def test():
    """All self-tests for lap3dkernels
    """
    test_checkgrad()
    test_lap3dcharge()
    test_lap3ddipole()
    test_lap3dmats()

def show_slice():
    """early code to plot potential on a slice in 2D and 3D
    """
    ns = 300                        # sources
    y = random.rand(ns,3)-0.5    # in [-1/2,1/2]^3
    d = random.randn(ns,3)
    z0=0.3; gmax=1.0                 # slice z, extent
    x,xx,yy = slicepts(z0=z0,a=gmax)
    nt = x.shape[0]
    u = zeros(nt)    # numba version writes outputs to arguments
    g = zeros([nt,3])
    lap3ddipole_numba(y,d,x,u,g)

    sc = 10   # colorscale
    
    # 2d image
    fig,ax = pl.subplots()
    #pl.figure()   # new fig
    im = ax.imshow(u.reshape(xx.shape),cmap='jet',vmin=-sc,vmax=sc,extent=(-gmax,gmax,-gmax,gmax))
    ax.set_xlabel('x'); ax.set_ylabel('y')
    #fig.colorbar(im)
    myplotutils.myshow(ax,im,fig)
    #myplotutils.goodcolorbar(im)   # acts on the image
    #pl.show()   # only needed if pl.ioff(), equiv of drawnow
    # can then close with pl.close(1)  etc

    # do 3d slice plot
    fig = pl.figure()   # new fig
    ax = fig.gca(projection='3d',aspect='equal')
    pts = ax.scatter(xs=y[:,0],ys=y[:,1],zs=y[:,2],s=1)
    usc = (u.reshape(xx.shape)+sc)/(2*sc)   # scale to [0,1] for cmap
    # replace by x coords?
    slice = ax.plot_surface(xx,yy,0*xx+z0,facecolors=pl.cm.jet(usc))
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

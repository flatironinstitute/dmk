import numpy as np
from prolate0 import Prolate0
import sys
sys.path.append('./perilap3d')
from perilap3d import lap3d3p
import finufft


def get_test1_input():
    n = 2
    r_src = np.array([
        [0.4, 0.4, 0.4],
        [0.1, 0.4, 0.4]
    ]) - 0.5
    charges = np.array([0.5, -0.5])
    return n, r_src, charges

def get_input():
    n = 10  # number of source points
    r_src = np.array([
        [0.131538, 0.686773, 0.98255],
        [0.45865,  0.930436, 0.753356],
        [0.218959, 0.526929, 0.0726859],
        [0.678865, 0.653919, 0.884707],
        [0.934693, 0.701191, 0.436411],
        [0.519416, 0.762198, 0.477732],
        [0.0345721, 0.0474645, 0.274907],
        [0.5297,   0.328234, 0.166507],
        [0.00769819, 0.75641, 0.897656],
        [0.0668422, 0.365339, 0.0605643]
    ]) - 0.5  # shape (10, 3)

    charges = np.array([0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.1, -0.1])

    return n, r_src, charges

def min_image_distance(r_i, r_j, L):
    delta = r_i - r_j
    delta = delta - L * np.round(delta / L)  # wrap to [-L/2, L/2]
    return np.linalg.norm(delta)

def min_image_vector(r_i, r_j, L):
    delta = r_i - r_j
    delta = delta - L * np.round(delta / L)  # wrap to [-L/2, L/2]
    return delta  # return vector, not norm

def short_range(distance, r_c, pswf, c0):
    if distance > r_c:
        return 0.0
    return (1 - pswf.integral(0.0, distance / r_c) / c0) / (4 * np.pi * distance) 

def pswf_hat(k, pswf, lambda0, c, c0):
    return lambda0 * pswf(k / c) 

def S_hat(k_vec, lambda_0, pswf, c, r_c, c0):
    k_mag = np.linalg.norm(k_vec)
    #s = lambda_0 * pswf(k_mag * r_c / c) / (2 * k_mag**2) / c0
    s = pswf_hat(k_mag * r_c, pswf, lambda_0, c, c0) / (2 * k_mag**2) / c0
    return s

def long_range(i, r_i, r_src, charges, L, n, n_f, r_c, pswf, lambda0, c, c0):
    long_range_potential = 0.0
    k_idx = np.arange(-n_f//2, n_f//2)  # -n_f/2, ..., n_f/2 - 1
    for kx in k_idx:
        for ky in k_idx:
            for kz in k_idx:
                if kx == 0 and ky == 0 and kz == 0:
                    continue
                k = np.array([kx, ky, kz])

                interior_sum = 0.0
                for j in range(n):
                    interior_sum += np.exp(1j * 2 * np.pi * np.dot(r_src[j], k) / L) * charges[j]

                s = S_hat(2 * np.pi * k / L, lambda0, pswf, c, r_c, c0)
                long_range_potential += np.exp(-1j * 2 * np.pi * np.dot(r_i, k) / L) * interior_sum * s / (L**3)

    return (long_range_potential).real

import finufft

def long_range_nufft(r_src, charges, L, n_f, r_c, pswf, lambda0, c, c0):
    #get positions from [-L/2, L/2] to [-pi, pi] interval for NUFFT
    x = 2 * np.pi * r_src[:, 0] / L
    y = 2 * np.pi * r_src[:, 1] / L
    z = 2 * np.pi * r_src[:, 2] / L

    # step 1: NUFFT type 1 - nonuniform to uniform
    F = finufft.nufft3d1(x, y, z, charges.astype(complex), n_modes=(n_f, n_f, n_f))
    #F is a 3D complex array of shape (n_f, n_f, n_f) where each element F[kx, ky, kz] corresponds to the Fourier transform at the mode (kx, ky, kz) on the uniform grid in Fourier space.

    # step 2: diagonal scaling by S_hat / L^3
    k_idx = np.arange(-n_f//2, n_f//2)
    kx, ky, kz = np.meshgrid(k_idx, k_idx, k_idx, indexing='ij')
    k_vecs = 2 * np.pi * np.stack([kx, ky, kz], axis=-1) / L  # shape (n_f, n_f, n_f, 3)

    s = np.zeros((n_f, n_f, n_f))
    for idx in np.ndindex(n_f, n_f, n_f):
        k_vec = k_vecs[idx]
        if np.any(k_vec != 0):
            s[idx] = S_hat(k_vec, lambda0, pswf, c, r_c, c0)

    F_scaled = F * s / L**3

    # step 3: NUFFT type 2 - uniform to nonuniform
    pot = finufft.nufft3d2(x, y, z, F_scaled)

    return pot.real

def phi(x, h, P, pswf):
    # x is a 3D displacement vector
    return pswf(2*x[0]/(P*h)) * pswf(2*x[1]/(P*h)) * pswf(2*x[2]/(P*h))

def precompute_phi_hat_1d(k_idx, n_f, r_c, pswf, L, P, h, lambda0, c, c0):
    phi_hat_1d = np.zeros(n_f)
    for i, k in enumerate(k_idx):
        k_vec = 2 * np.pi * k / L
        argument = k_vec * (P * h) / 2
        phi_hat_1d[i] = (P * h / 2) * pswf_hat(argument, pswf, lambda0, c, c0)
    return phi_hat_1d

def precompute_scaling_coefficients(n_f, r_c, pswf, lambda0, c, L, P, h, c0):
    #k_idx = np.arange(-n_f//2, n_f//2)
    k_idx = np.fft.fftfreq(n_f, d=1.0/n_f).astype(int)  # FFT order directly
    print("k_idx:", k_idx)
    kx, ky, kz = np.meshgrid(k_idx, k_idx, k_idx, indexing='ij')
    k_vecs = np.stack([kx, ky, kz], axis=-1)  # (n_f, n_f, n_f, 3)

    phi_hat_1d = precompute_phi_hat_1d(k_idx, n_f, r_c, pswf, L, P, h, lambda0, c, c0) 
    print("phi_hat_1d min/max:", phi_hat_1d.min(), phi_hat_1d.max())
    print("phi_hat_1d:", phi_hat_1d) 

    S = np.zeros((n_f, n_f, n_f))
    p = np.zeros((n_f, n_f, n_f))
    for idx in np.ndindex(n_f, n_f, n_f):
        k_vec = k_vecs[idx]
        if np.any(k_vec != 0):
            s = S_hat(2 * np.pi * k_vec / L, lambda0, pswf, c, r_c, c0)
            phi_hat = phi_hat_1d[idx[0]] * phi_hat_1d[idx[1]] * phi_hat_1d[idx[2]]
            p[idx] = s / (L**3 * phi_hat**2 * n_f**3) # not sure why we have to divide by n_f^3 here, but it seems to be necessary to match the direct sum results. 
            
            #equilavence check
            #k_vec_mag = np.linalg.norm(k_vec)
            #numerator = pswf_hat(r_c * k_vec_mag * 2 * np.pi / L, pswf, lambda0, c, c0) / c0
            #denominator = 2 * (2 * np.pi)**2 * L * (phi_hat**2) * (k_vec_mag**2)
            #p[idx] = numerator / denominator
    return p

def stencil_offsets(P):
    if P <= 0:
        raise ValueError("P must be positive")
    if P % 2 == 1:
        m = P // 2
        return np.arange(-m, m + 1)
    return np.arange(-P // 2, P // 2)

def spreading(r_src, charges, n_f, L, pswf, lambda0, c, P):
    h = L / n_f
    b = np.zeros((n_f, n_f, n_f), dtype=complex)
    
    for j in range(len(charges)):
        # nearest grid point in each dimension
        l_center = np.round(r_src[j] / h).astype(int)
        
        # P neighbors in each dimension
        offsets = stencil_offsets(P)  # shape (P,)
        lx = (l_center[0] + offsets) % n_f
        ly = (l_center[1] + offsets) % n_f
        lz = (l_center[2] + offsets) % n_f
        
        # evaluate phi at each of the P^3 neighbors
        for ix, ell_x in enumerate(lx):
            for iy, ell_y in enumerate(ly):
                for iz, ell_z in enumerate(lz):
                    displacement = min_image_vector(r_src[j], h * np.array([ell_x, ell_y, ell_z]), L)
                    phi_val = phi(displacement, h, P, pswf)
                    b[ell_x, ell_y, ell_z] += charges[j] * phi_val
    
    return b

def interpolation(c, r_src, n_f, L, pswf, P):
    h = L / n_f
    pot = np.zeros(len(r_src))
    for i in range(len(r_src)):
        # evaluate phi at each of the P^3 neighbors
        l_center = np.round(r_src[i] / h).astype(int)
        
        # P neighbors in each dimension
        offsets = stencil_offsets(P)  # shape (P,)
        lx = (l_center[0] + offsets) % n_f
        ly = (l_center[1] + offsets) % n_f
        lz = (l_center[2] + offsets) % n_f
        for ix, ell_x in enumerate(lx):
            for iy, ell_y in enumerate(ly):
                for iz, ell_z in enumerate(lz):
                    displacement = min_image_vector(r_src[i], h * np.array([ell_x, ell_y, ell_z]), L)
                    pot[i] += c[ell_x, ell_y, ell_z] * phi(displacement, h, P, pswf).real

    return pot

def long_range_fast(r_src, charges, L, n_f, r_c, pswf, lambda0, c, c0, P):
    b = spreading(r_src, charges, n_f, L, pswf, lambda0, c, P=5)
    print("b sum:", b.sum(), "b max:", np.abs(b).max())
    
    b_hat = np.fft.fftn(b)
    print("b_hat max:", np.abs(b_hat).max())
    
    p = precompute_scaling_coefficients(n_f, r_c, pswf, lambda0, c, L, P, L/n_f, c0)
    print("p max:", np.abs(p).max(), "p min:", np.abs(p).min())
    
    b_hat_scaled = b_hat * p
    print("b_hat_scaled max:", np.abs(b_hat_scaled).max())
    
    grid = np.fft.ifftn(b_hat_scaled)
    print("grid max:", np.abs(grid).max())
    
    long_range_pot = interpolation(grid, r_src, n_f, L, pswf, P=5)
    print("long_range_pot:", long_range_pot)
    return long_range_pot

def init_PSWF(eps=1e-6, L=1.0, r_c=0.2):
    pswf = Prolate0.from_eps(eps)
    
    # normalize so that ∫_0^1 χ(x)dx = 1 (paper's convention)
    c0 = pswf.integral(0.0, 1.0)
    pswf = pswf.normalize()  # now pswf(0) = 1 

    c, mu = pswf.c, pswf.eigenvalue
    lambda0 = np.sqrt(2 * np.pi * mu / c)
    n_f = int(np.ceil(c * L / (np.pi * r_c)))
    
    return pswf, lambda0, n_f, r_c, c, c0, L

def main():
    n, r_src, charges = get_test1_input()
    pswf, lambda0, n_f, r_c, c, c0, L = init_PSWF(1e-6, 1.0, 0.2)
    h = L / n_f #grid spacing
    #P = r_c * 2 / h #the number of grid points per dimension each particle contributes to
    P = 5

    print(h, P, n_f, lambda0, c, L)
    print(f"pswf(0) = {pswf(0.0)}")
    print(f"pswf(1) = {pswf(1.0)}")
    print(f"integral(0,1) = {pswf.integral(0.0, 1.0)}")
    print(f"integral(0, 0.05) = {pswf.integral(0.0, 0.05)}")

    potential_long_range_fast = long_range_fast(r_src, charges, L, n_f, r_c, pswf, lambda0, c, c0, P)

    box = np.array([[1,0,0],[0,1,0],[0,0,1]])
    p = lap3d3p(box)
    p.precomp(tol=1e-6)

    pot, grad = p.eval(r_src, None, None, charges)
    pot = pot - 0.5 * sum(pot)
    print("pot =", pot)
    
    potential_long_range_nufft = long_range_nufft(r_src, charges, L, n_f, r_c, pswf, lambda0, c, c0)
    

    for i in range(n):
        r_i = r_src[i]
        potential_short_range = 0.0
        potential_long_range = 0.0
        for j in range(n):
            if i == j:
                continue
            r_j = r_src[j]
            distance = min_image_distance(r_i, r_j, L)
            
            potential_short_range += charges[j] * short_range(distance, r_c, pswf, c0)
        
        potential_long_range = long_range(i, r_i, r_src, charges, L, n=n, n_f=n_f, r_c=r_c, pswf=pswf, lambda0=lambda0, c=c, c0=c0)            
        potential_self_interaction = charges[i] * pswf(0) / (r_c * 4 * np.pi * c0)

        potential = potential_short_range + potential_long_range - potential_self_interaction
        print(f"Potential short-range at point {i}: {potential_short_range}")
        print(f"Potential long-range at point {i}: {potential_long_range}")
        print(f"Potential long-range (NUFFT) at point {i}: {potential_long_range_nufft[i]}")
        print(f"Potential long-range (fast) at point {i}: {potential_long_range_fast[i]}")
        print(f"Potential analytic self-interaction at point {i}: {potential_self_interaction}")
        print(f"Potential at point {i}: {potential}")
        print(f"Potential error at point {i}: {potential - pot[i]}")
        print()

if __name__ == "__main__":
    main()

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

def get_test2_input():
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

class ESPParams:
    def __init__(self, L, r_c, P, pswf, n):
        self.L = L
        self.r_c = r_c
        self.P = P #P = r_c * 2 / h #the number of grid points per dimension each particle contributes to
        self.n_f = int(np.ceil(pswf.c * L / (np.pi * r_c)))
        self.h = L / self.n_f
        self.lambda0 = pswf.lambda0
        self.c = pswf.c
        self.c0 = pswf.c0
        self.n = n #number of charge points

class PSWFKernel:
    def __init__(self, eps):
        self.pswf = Prolate0.from_eps(eps)
        self.pswf = self.pswf.normalize()  # now pswf(0) = 1 
        self.c, mu = self.pswf.c, self.pswf.eigenvalue
        self.lambda0 = np.sqrt(2 * np.pi * mu / self.c)
        self.c0 = self.pswf.integral(0.0, 1.0)
    
    def __call__(self, x):
        return self.pswf(x)
    
    def integral(self, a, b):
        return self.pswf.integral(a, b)
    
    def pswf_hat(self, k):
        """pswf_hat: Fourier transform of the PSWF"""
        return self.lambda0 * self.pswf(k / self.c) 

def min_image_distance(r_i, r_j, L):
    delta = r_i - r_j
    delta = delta - L * np.round(delta / L)  # wrap to [-L/2, L/2]
    return np.linalg.norm(delta)

def min_image_vector(r_i, r_j, L):
    delta = r_i - r_j
    delta = delta - L * np.round(delta / L)  # wrap to [-L/2, L/2]
    return delta  # return vector, not norm

def build_cell_list(r_src, params):
    n_cells = int(np.floor(params.L / params.r_c))
    cell_size = params.L / n_cells
    
    # each cell is a list of particle indices
    cells = {}
    for j in range(params.n):
        # shift from [-L/2, L/2] to [0, L] first
        cell_idx = tuple(
            (np.floor((r_src[j] + params.L/2) / cell_size).astype(int)) % n_cells
        )
        if cell_idx not in cells:
            cells[cell_idx] = []
        cells[cell_idx].append(j)
    
    return cells, n_cells

def build_neighbor_list(r_src, params):
    cells, n_cells = build_cell_list(r_src, params)
    neighbors = [[] for _ in range(params.n)]
    
    for i in range(params.n):
        cell_size = params.L / n_cells
        cell_idx = (np.floor((r_src[i] + params.L/2) / cell_size).astype(int)) % n_cells
        
        # check all 27 neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_cell = tuple((cell_idx + np.array([dx, dy, dz])) % n_cells)
                    for j in cells.get(neighbor_cell, []):
                        if i == j:
                            continue
                        if min_image_distance(r_src[i], r_src[j], params.L) <= params.r_c:
                            neighbors[i].append(j)
    
    return neighbors


def short_range(r_src, charges,pswf, params, neighbors):
    potential_short_range = np.zeros(params.n)
    for i in range(params.n):
        r_i = r_src[i]
        for j in neighbors[i]:
            if i == j:
                continue
            r_j = r_src[j]
            distance = min_image_distance(r_i, r_j, params.L)
            x = (1 - pswf.integral(0.0, distance / params.r_c) / params.c0) / (4 * np.pi * distance)
            potential_short_range[i] += charges[j] * x
    return potential_short_range 

def S_hat(pswf, params, k_vec):
    k_mag = np.linalg.norm(k_vec)
    #s = lambda_0 * pswf(k_mag * r_c / c) / (2 * k_mag**2) / c0
    s = pswf.pswf_hat(k_mag * params.r_c) / (2 * k_mag**2) / params.c0
    return s

def long_range_slow(r_src, charges, pswf, params):
    potential_long_range = np.zeros(params.n)

    k_idx = np.arange(-params.n_f//2, params.n_f//2)  # -n_f/2, ..., n_f/2 - 1

    # Precompute S_hat for all k in the grid
    s = np.zeros((params.n_f, params.n_f, params.n_f))
    for kx in k_idx:
        for ky in k_idx:
            for kz in k_idx:
                if kx == 0 and ky == 0 and kz == 0:
                    continue
                k = np.array([kx, ky, kz])
                s[kx, ky, kz] = S_hat(pswf, params, 2 * np.pi * k / params.L)

    for i in range(params.n):
        r_i = r_src[i]
        for kx in k_idx:
            for ky in k_idx:
                for kz in k_idx:
                    if kx == 0 and ky == 0 and kz == 0:
                        continue
                    k = np.array([kx, ky, kz])

                    interior_sum = 0.0
                    for j in range(params.n):
                        interior_sum += np.exp(1j * 2 * np.pi * np.dot(r_src[j], k) / params.L) * charges[j]

                    potential_long_range[i] += (np.exp(-1j * 2 * np.pi * np.dot(r_i, k) / params.L) * interior_sum * s[kx, ky, kz] / (params.L**3)).real

    return potential_long_range

def long_range_nufft(r_src, charges, pswf, params):
    #get positions from [-L/2, L/2] to [-pi, pi] interval for NUFFT
    x = 2 * np.pi * r_src[:, 0] / params.L
    y = 2 * np.pi * r_src[:, 1] / params.L
    z = 2 * np.pi * r_src[:, 2] / params.L

    # step 1: NUFFT type 1 - nonuniform to uniform
    F = finufft.nufft3d1(x, y, z, charges.astype(complex), n_modes=(params.n_f, params.n_f, params.n_f))
    #F is a 3D complex array of shape (n_f, n_f, n_f) where each element F[kx, ky, kz] corresponds to the Fourier transform at the mode (kx, ky, kz) on the uniform grid in Fourier space.

    # step 2: diagonal scaling by S_hat / L^3
    k_idx = np.arange(-params.n_f//2, params.n_f//2)
    kx, ky, kz = np.meshgrid(k_idx, k_idx, k_idx, indexing='ij')
    k_vecs = 2 * np.pi * np.stack([kx, ky, kz], axis=-1) / params.L  # shape (n_f, n_f, n_f, 3)

    s = np.zeros((params.n_f, params.n_f, params.n_f))
    for idx in np.ndindex(params.n_f, params.n_f, params.n_f):
        k_vec = k_vecs[idx]
        if np.any(k_vec != 0):
            s[idx] = S_hat(pswf, params, k_vec)

    F_scaled = F * s / params.L**3

    # step 3: NUFFT type 2 - uniform to nonuniform
    pot = finufft.nufft3d2(x, y, z, F_scaled)

    return pot.real

def phi(x, pswf, params):
    # x is a 3D displacement vector
    Ph = params.P * params.h
    return pswf(2*x[0]/Ph) * pswf(2*x[1]/Ph) * pswf(2*x[2]/Ph)

def precompute_phi_hat_1d(k_idx, pswf, params):
    phi_hat_1d = np.zeros(params.n_f)
    for i, k in enumerate(k_idx):
        k_vec = 2 * np.pi * k / params.L
        argument = k_vec * (params.P * params.h) / 2
        phi_hat_1d[i] = (params.P * params.h / 2) * pswf.pswf_hat(argument)
    return phi_hat_1d

def precompute_scaling_coefficients(pswf, params):
    #k_idx = np.arange(-n_f//2, n_f//2)
    k_idx = np.fft.fftfreq(params.n_f, d=1.0/params.n_f).astype(int)  # FFT order directly
    kx, ky, kz = np.meshgrid(k_idx, k_idx, k_idx, indexing='ij')
    k_vecs = np.stack([kx, ky, kz], axis=-1)  # (n_f, n_f, n_f, 3)

    phi_hat_1d = precompute_phi_hat_1d(k_idx, pswf, params) 

    S = np.zeros((params.n_f, params.n_f, params.n_f))
    p = np.zeros((params.n_f, params.n_f, params.n_f))
    for idx in np.ndindex(params.n_f, params.n_f, params.n_f):
        k_vec = k_vecs[idx]
        if np.any(k_vec != 0):
            s = S_hat(pswf, params, 2 * np.pi * k_vec / params.L)
            phi_hat = phi_hat_1d[idx[0]] * phi_hat_1d[idx[1]] * phi_hat_1d[idx[2]]
            p[idx] = s / (params.L**3 * phi_hat**2 * params.n_f**3) # not sure why we have to divide by n_f^3 here, but it seems to be necessary to match the direct sum results. 
            
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

def spreading(r_src, charges, pswf, params):
    b = np.zeros((params.n_f, params.n_f, params.n_f), dtype=complex)
    offsets = stencil_offsets(params.P)  # shape (P,)
    for j in range(len(charges)):
        # nearest grid point in each dimension
        l_center = np.round(r_src[j] / params.h).astype(int)
        
        # P neighbors in each dimension
        lx = (l_center[0] + offsets) % params.n_f
        ly = (l_center[1] + offsets) % params.n_f
        lz = (l_center[2] + offsets) % params.n_f
        
        # evaluate phi at each of the P^3 neighbors
        for ix, ell_x in enumerate(lx):
            for iy, ell_y in enumerate(ly):
                for iz, ell_z in enumerate(lz):
                    displacement = min_image_vector(r_src[j], params.h * np.array([ell_x, ell_y, ell_z]), params.L)
                    phi_val = phi(displacement, pswf, params)
                    b[ell_x, ell_y, ell_z] += charges[j] * phi_val
    
    return b

def interpolation(c, r_src, pswf, params):
    pot = np.zeros(len(r_src))
    offsets = stencil_offsets(params.P)  # shape (P,)
    for i in range(len(r_src)):
        # evaluate phi at each of the P^3 neighbors
        l_center = np.round(r_src[i] / params.h).astype(int)
        
        # P neighbors in each dimension
        lx = (l_center[0] + offsets) % params.n_f
        ly = (l_center[1] + offsets) % params.n_f
        lz = (l_center[2] + offsets) % params.n_f
        for ix, ell_x in enumerate(lx):
            for iy, ell_y in enumerate(ly):
                for iz, ell_z in enumerate(lz):
                    displacement = min_image_vector(r_src[i], params.h * np.array([ell_x, ell_y, ell_z]), params.L)
                    pot[i] += c[ell_x, ell_y, ell_z] * phi(displacement, pswf, params).real

    return pot

def long_range_fast(r_src, charges, pswf, params):
    b = spreading(r_src, charges, pswf, params)
    b_hat = np.fft.fftn(b)
    p = precompute_scaling_coefficients(pswf, params)
    b_hat_scaled = b_hat * p
    grid = np.fft.ifftn(b_hat_scaled)
    long_range_pot = interpolation(grid, r_src, pswf, params)
    return long_range_pot

def self_interaction(r_src, charges, pswf, params):
    self_interaction = np.zeros(params.n)
    for i in range(params.n):
        r_i = r_src[i]
        self_interaction[i] = charges[i] * pswf(0) / (params.r_c * 4 * np.pi * params.c0)
    return self_interaction

def reference_potential(r_src, charges):
    box = np.array([[1,0,0],[0,1,0],[0,0,1]])
    p = lap3d3p(box)
    p.precomp(tol=1e-6)

    pot, grad = p.eval(r_src, None, None, charges)
    pot = pot - 0.5 * sum(pot)
    return pot


def main():
    n, r_src, charges = get_test2_input()
    pswf = PSWFKernel(1e-6)
    params = ESPParams(L=1.0, r_c=0.2, P=7, pswf=pswf, n=charges.size)

    print(f"c           = {pswf.c:.6f}")
    print(f"lambda0     = {pswf.lambda0:.6f}")
    print(f"c0          = {pswf.c0:.6f}")
    print(f"pswf(0)     = {pswf(0.0):.6f}")
    print(f"pswf(1)     = {pswf(1.0):.6e}")
    print(f"pswf_hat(0) = {pswf.pswf_hat(0.0):.6f}")
    print("mu - eigenvalue", pswf.pswf.eigenvalue)

    potential_reference = reference_potential(r_src, charges)
    potential_long_range_nufft = long_range_nufft(r_src, charges, pswf, params)
    potential_long_range_fast = long_range_fast(r_src, charges, pswf, params)
    potential_long_range_slow = long_range_slow(r_src, charges, pswf, params)
    potential_self_interaction = self_interaction(r_src, charges, pswf, params)

    neighbors = build_neighbor_list(r_src, params)
    potential_short_range = short_range(r_src, charges, pswf, params, neighbors)
    total_potential = potential_short_range + potential_long_range_slow - potential_self_interaction


    for i in range(params.n):
        print(f"Reference total potential at point {i} (perilap3d): {potential_reference[i]}")
        print(f"Potential short-range at point {i}: {potential_short_range[i]}")
        print(f"Potential long-range (slow) at point {i}: {potential_long_range_slow[i]}")
        print(f"Potential long-range (NUFFT) at point {i}: {potential_long_range_nufft[i]}")
        print(f"Potential long-range (fast) at point {i}: {potential_long_range_fast[i]}")
        print(f"Potential analytic self-interaction at point {i}: {potential_self_interaction[i]}")
        print(f"Potential at point {i}: {total_potential[i]}")
        print(f"Potential error: long-range (slow) - long range (fast) at point {i}: {potential_long_range_slow[i] - potential_long_range_fast[i]}")
        print(f"Potential error at point {i}: {total_potential[i] - potential_reference[i]}")
        print()

if __name__ == "__main__":
    main()

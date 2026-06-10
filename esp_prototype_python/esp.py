import numpy as np
from prolate0 import Prolate0
import sys
sys.path.append('./perilap3d')
from perilap3d import lap3d3p


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

def short_range(distance, r_c, pswf):
    if distance > r_c:
        return 0.0
    return (1 - pswf.integral(0.0, distance / r_c)) / (4 * np.pi * distance) 

def S_hat(k_vec, lambda_0, pswf, c, r_c):
    k_mag = np.linalg.norm(k_vec)
    s = lambda_0 * pswf(k_mag * r_c / c) / (2 * k_mag**2)
    return s

def long_range(i, r_i, r_src, charges, L, n, n_f, r_c, pswf, lambda0, c):
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

                s = S_hat(2 * np.pi * k / L, lambda0, pswf, c, r_c)
                long_range_potential += np.exp(-1j * 2 * np.pi * np.dot(r_i, k) / L) * interior_sum * s / (L**3)

    return (long_range_potential).real



def init_PSWF(eps=1e-6, L=1.0, r_c=0.2):
    pswf = Prolate0.from_eps(eps)
    
    # normalize so that ∫_0^1 χ(x)dx = 1 (paper's convention)
    c0 = pswf.integral(0.0, 1.0)
    pswf = pswf.normalize()  # now pswf(0) = 1 and ∫_0^1 pswf(x) dx = 1

    c, mu = pswf.c, pswf.eigenvalue
    lambda0 = np.sqrt(2 * np.pi * mu / c)
    n_f = int(np.ceil(c * L / (np.pi * r_c)))
    
    return pswf, lambda0, n_f, r_c, c, L

def main():
    n, r_src, charges = get_input()
    pswf, lambda0, n_f, r_c, c, L = init_PSWF(1e-6, 1.0, 0.2)

    print(n_f, lambda0, c, L)
    print(f"pswf(0) = {pswf(0.0)}")
    print(f"pswf(1) = {pswf(1.0)}")
    print(f"integral(0,1) = {pswf.integral(0.0, 1.0)}")
    print(f"integral(0, 0.05) = {pswf.integral(0.0, 0.05)}")

    box = np.array([[1,0,0],[0,1,0],[0,0,1]])
    p = lap3d3p(box)
    p.precomp(tol=1e-6)

    pot, grad = p.eval(r_src, None, None, charges)
    pot = pot - 0.5 * sum(pot)
    print("pot =", pot)
    
    for i in range(n):
        r_i = r_src[i]
        potential_short_range = 0.0
        potential_long_range = 0.0
        for j in range(n):
            if i == j:
                continue
            r_j = r_src[j]
            distance = min_image_distance(r_i, r_j, L)
            
            potential_short_range += charges[j] * short_range(distance, r_c, pswf)
        
        potential_long_range = long_range(i, r_i, r_src, charges, L, n=n, n_f=n_f, r_c=r_c, pswf=pswf, lambda0=lambda0, c=c)            
        potential_self_interaction = charges[i] * pswf(0) / (r_c * 4 * np.pi)

        potential = potential_short_range + potential_long_range - potential_self_interaction
        print(f"Potential short-range at point {i}: {potential_short_range}")
        print(f"Potential long-range at point {i}: {potential_long_range}")
        print(f"Potential analytic self-interaction at point {i}: {potential_self_interaction}")
        print(f"Potential at point {i}: {potential}")
        print(f"Potential error at point {i}: {potential - pot[i]}")
        print()

if __name__ == "__main__":
    main()

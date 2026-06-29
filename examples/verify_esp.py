#!/usr/bin/env python3
"""Helper called by benchmark_esp -V: reads binary particle data, runs perilap3d, prints potentials."""
import sys
import numpy as np

perilap3d_dir = sys.argv[2]
sys.path.insert(0, perilap3d_dir)
from perilap3d import lap3d3p

datafile = sys.argv[1]
with open(datafile, "rb") as f:
    n = np.frombuffer(f.read(4), dtype=np.int32)[0]
    r_src   = np.frombuffer(f.read(n * 3 * 8), dtype=np.float64).reshape(n, 3)
    charges = np.frombuffer(f.read(n * 8),     dtype=np.float64)

print(f"n={n}, r_src.shape={r_src.shape}, charges.shape={charges.shape}", file=sys.stderr)
print(f"charges sum={charges.sum():.6g}", file=sys.stderr)

box = np.eye(3)
p = lap3d3p(box)
p.precomp(tol=1e-6)
print("precomp done", file=sys.stderr)

pot, _ = p.eval(r_src, None, None, charges)
print(f"pot.shape={pot.shape}, pot.dtype={pot.dtype}", file=sys.stderr)

pot = pot.real - np.mean(pot.real)

for v in pot:
    print(repr(float(v)))

#!/usr/bin/env python3
"""
Exploratory visualization of measure_error_esp output.

For each (kernel, dim) it plots achieved accuracy (-log10 of the relative L2) against
the requested digits, for both potential and force, picking the best r_c at each digit
(faint dots show every r_c). The dotted y=x line is "you get exactly what you ask for";
force sitting below it is the missing derivative compensation. A linear fit of the
best-r_c force curve is overlaid, and the effective-digit offset (how many extra digits
to request so the force L2 meets the target) is printed per kernel/dim.

Usage:
    python scripts/analyze_esp_error.py results.csv
    python scripts/analyze_esp_error.py results.csv --save out.png
"""
import argparse
import csv
import math
from collections import defaultdict

import numpy as np


def load(path):
    data = []
    with open(path) as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    for r in csv.DictReader(lines):
        try:
            data.append(
                dict(
                    kernel=r["kernel"],
                    dim=int(r["dim"]),
                    digits=int(r["digits"]),
                    r_c=float(r["r_c"]),
                    c=float(r["c"]),
                    n_f=int(r["n_f"]),
                    pot_l2=float(r["pot_l2"]),
                    force_l2=float(r["force_l2"]),
                    time=float(r["time"]),
                )
            )
        except ValueError:
            continue  # FAILED rows
    return data


def achieved(l2):
    return -math.log10(l2) if l2 > 0 else float("nan")


def best_per_digit(rows, metric):
    """For each requested-digit level, the (digit, achieved, r_c) with the smallest error."""
    by_dig = defaultdict(list)
    for r in rows:
        by_dig[r["digits"]].append(r)
    out = []
    for d in sorted(by_dig):
        best = min(by_dig[d], key=lambda r: r[metric])
        out.append((d, achieved(best[metric]), best["r_c"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--save", help="write figure to this path instead of showing it")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('TKAgg')

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(args.csv)
    groups = sorted({(r["kernel"], r["dim"]) for r in data})
    if not groups:
        raise SystemExit("no usable rows in " + args.csv)

    ncol = min(3, len(groups))
    nrow = math.ceil(len(groups) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.2 * nrow), squeeze=False)

    print(f"{'kernel':>13} {'dim':>3} {'metric':>6} {'offset(min/mean/max)':>22} {'fit: achieved~a*d+b':>22}")
    for i, (kernel, dim) in enumerate(groups):
        ax = axes[i // ncol][i % ncol]
        rows = [r for r in data if r["kernel"] == kernel and r["dim"] == dim]

        # faint scatter of every r_c
        for r in rows:
            ax.plot(r["digits"], achieved(r["pot_l2"]), ".", color="tab:blue", alpha=0.2, ms=5)
            ax.plot(r["digits"], achieved(r["force_l2"]), ".", color="tab:red", alpha=0.2, ms=5)

        for metric, color, label in (("pot_l2", "tab:blue", "potential"), ("force_l2", "tab:red", "force")):
            pts = best_per_digit(rows, metric)
            dd = np.array([p[0] for p in pts], float)
            aa = np.array([p[1] for p in pts], float)
            ok = np.isfinite(aa)
            dd, aa = dd[ok], aa[ok]
            if len(dd) == 0:
                continue
            ax.plot(dd, aa, "-o", color=color, label=f"{label} (best r_c)")

            offs = dd - aa  # extra digits to request to reach the achieved level
            a, b = np.polyfit(dd, aa, 1) if len(dd) > 1 else (float("nan"), float("nan"))
            print(
                f"{kernel:>13} {dim:>3} {label[:6]:>6} "
                f"{offs.min():6.2f}/{offs.mean():6.2f}/{offs.max():6.2f}   "
                f"a={a:5.2f} b={b:6.2f}"
            )
            if metric == "force_l2" and math.isfinite(a):
                ax.plot(dd, a * dd + b, "--", color=color, alpha=0.7, lw=1)

        lim = [min(r["digits"] for r in rows), max(r["digits"] for r in rows)]
        ax.plot(lim, lim, ":", color="gray", label="y = x")
        ax.set_title(f"{kernel} {dim}D")
        ax.set_xlabel("requested digits")
        ax.set_ylabel("achieved digits (-log10 L2)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(len(groups), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=130)
        print("wrote " + args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()

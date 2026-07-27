#!/usr/bin/env python3
"""
Exploratory visualization of measure_error_esp output, and calibration of the
requested->effective tolerance curves.

For each (kernel, dim) it plots achieved accuracy (-log10 relative L2) against the
requested digits, for potential and force. The key output is a conservative LOWER
ENVELOPE per metric: a line achieved >= a*digits + b that lies beneath *every* point
(all r_c), so it is a guaranteed floor. Inverting it gives the minimal tolerance to
request to reach a target accuracy -- i.e. the fewest digits for the most guaranteed
digits. The force envelope is what feeds esp_grad_eps() in include/dmk/esp.hpp; the
potential envelope should sit near y=x ("what you ask is what you get").

To CALIBRATE (map requested -> achieved rawly), run the sweep with the derivative
bump disabled so requested digits == internal resolution:
    DMK_ESP_NO_GRAD_BUMP=1 ./examples/measure_error_esp -k all -d 0 > raw.csv
    python scripts/analyze_esp_error.py raw.csv
Run WITHOUT the env var to validate that the baked-in bump makes force meet target.

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
                    pot_l2=float(r["pot_l2"]),
                    force_l2=float(r["force_l2"]),
                )
            )
        except ValueError:
            continue  # FAILED rows
    return data


def achieved(l2):
    return -math.log10(l2) if l2 > 0 else float("nan")


def lower_envelope(dd, aa):
    """Line a*x+b lying beneath all (dd, aa) points: least-squares slope, then intercept
    lowered until it touches the lowest point. Guarantees achieved >= a*requested + b."""
    if len(dd) < 2:
        return float("nan"), float("nan")
    a, _ = np.polyfit(dd, aa, 1)
    b = float(np.min(aa - a * dd))
    return float(a), b


def best_per_digit(rows, metric):
    by_dig = defaultdict(list)
    for r in rows:
        by_dig[r["digits"]].append(r)
    out = []
    for d in sorted(by_dig):
        best = min(by_dig[d], key=lambda r: r[metric])
        out.append((d, achieved(best[metric])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--save", help="write figure to this path instead of showing it")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg" if args.save else "TkAgg")
    import matplotlib.pyplot as plt

    data = load(args.csv)
    groups = sorted({(r["kernel"], r["dim"]) for r in data})
    if not groups:
        raise SystemExit("no usable rows in " + args.csv)

    ncol = min(3, len(groups))
    nrow = math.ceil(len(groups) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.2 * nrow), squeeze=False)

    # target accuracies to report the minimal request for
    targets = [3, 6, 9]
    hdr = f"{'kernel':>13} {'dim':>3} {'metric':>6}  {'lower env achieved>=a*d+b':>26}  request for target " + "/".join(
        str(t) for t in targets
    )
    print(hdr)
    for i, (kernel, dim) in enumerate(groups):
        ax = axes[i // ncol][i % ncol]
        rows = [r for r in data if r["kernel"] == kernel and r["dim"] == dim]

        for metric, color, label in (("pot_l2", "tab:blue", "pot"), ("force_l2", "tab:red", "force")):
            # every (r_c, digit) point -- the envelope must sit beneath all of them
            dd = np.array([r["digits"] for r in rows], float)
            aa = np.array([achieved(r[metric]) for r in rows], float)
            ok = np.isfinite(aa)
            dd, aa = dd[ok], aa[ok]
            if len(dd) == 0:
                continue
            ax.plot(dd, aa, ".", color=color, alpha=0.25, ms=5)
            # best-r_c trace (the achievable), for context
            bp = best_per_digit(rows, metric)
            ax.plot([p[0] for p in bp], [p[1] for p in bp], "-", color=color, alpha=0.5, lw=1, label=f"{label} best r_c")

            a, b = lower_envelope(dd, aa)
            xs = np.array([dd.min(), dd.max()])
            ax.plot(xs, a * xs + b, "--", color=color, lw=2, label=f"{label} lower env")

            # minimal requested digits to guarantee each target: a*d+b >= t  =>  d >= (t-b)/a
            reqs = [f"{int(math.ceil((t - b) / a))}" if a > 0 else "inf" for t in targets]
            print(f"{kernel:>13} {dim:>3} {label:>6}  a={a:5.2f} b={b:6.2f}{'':>10}  " + "/".join(reqs))

        lim = [min(r["digits"] for r in rows), max(r["digits"] for r in rows)]
        ax.plot(lim, lim, ":", color="gray", label="y = x")
        ax.set_title(f"{kernel} {dim}D")
        ax.set_xlabel("requested digits")
        ax.set_ylabel("achieved digits (-log10 L2)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

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

#!/usr/bin/env python3
"""Per-particle relative error distributions from benchmark_kernels --err-dump files.

The layout follows Fig. 4 of the jz-fmm paper (Stuecker & Foldal, arXiv 2609.09307): one panel
per quantity, x = log10 of the per-particle relative error |u - u_ref| / |u_ref|, y = dn/dlog10(eps),
one histogram per run. A dump with gradient columns gives a Force and a Potential panel, any other
dump one panel over all its components (e.g. Stokes velocity).

    plot_error_dist.py -o fig.png [--ordinal] [--title T] label=path [label=path ...]

--ordinal colours the runs as an ordered sweep (light to dark, e.g. loosening to tightening eps);
otherwise they take categorical colours in the order given. Quantiles are printed to stdout. Exact
hits (zero error, common for float potentials) have no log10: they are counted in the printout and
left out of the histogram, whose area is the fraction of particles with a nonzero error.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDINAL = ["#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#104281", "#0d366b"]


def ordinal_colors(n):
    idx = np.linspace(0, len(ORDINAL) - 1, n).round().astype(int) if n > 1 else [3]
    return [ORDINAL[i] for i in idx]


def load(path):
    d = np.loadtxt(path, comments="#", ndmin=2)
    if d.shape[1] == 4:
        return {"Force": (d[:, 2], d[:, 3]), "Potential": (d[:, 0], d[:, 1])}
    return {"Value": (d[:, 0], d[:, 1])}


def rel_err(diff, ref):
    keep = ref > 0
    return diff[keep] / ref[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--ordinal", action="store_true")
    ap.add_argument("--title", default="")
    ap.add_argument("--panel", default="", help="name for the panel of a dump without gradient columns")
    ap.add_argument("runs", nargs="+", metavar="label=path")
    args = ap.parse_args()

    runs = [(lab, load(p)) for lab, p in (r.rsplit("=", 1) for r in args.runs)]
    panels = list(runs[0][1].keys())
    colors = ordinal_colors(len(runs)) if args.ordinal else CATEGORICAL[: len(runs)]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
    width = max(6.4, 5.2 * len(panels))
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 3.9), squeeze=False, facecolor=SURFACE)
    for ax, panel in zip(axes[0], panels):
        errs = [rel_err(*r[panel]) for _, r in runs]
        logs = [np.log10(e[e > 0]) for e in errs]
        lo = np.floor(min(np.percentile(v, 0.01) for v in logs) * 10) / 10
        hi = np.ceil(max(v.max() for v in logs) * 10) / 10 + 0.1
        bins = np.arange(lo, hi + 0.05, 0.1)
        print(f"{panel}: per-particle relative error quantiles")
        print(f"  {'run':24s} {'n':>7s} {'exact':>6s} {'p50':>9s} {'p90':>9s} {'p99':>9s} {'p99.9':>9s} {'max':>9s}")
        for (label, _), e, v, c in zip(runs, errs, logs, colors):
            h, edges = np.histogram(v, bins=bins)
            dens = h / (len(e) * np.diff(edges))
            ax.stairs(dens, edges, fill=True, color=c, alpha=0.14, linewidth=0)
            ax.stairs(dens, edges, color=c, linewidth=2, label=label)
            if len(runs) <= 4:  # past four the legend alone names them
                peak = np.argmax(dens)
                ax.annotate(label, ((edges[peak] + edges[peak + 1]) / 2, dens[peak]), xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK2)
            q = np.percentile(e, [50, 90, 99, 99.9])
            print(f"  {label:24s} {len(e):7d} {np.sum(e == 0):6d} " + " ".join(f"{x:9.2e}" for x in q)
                  + f" {e.max():9.2e}")
        ax.set_facecolor(SURFACE)
        ax.set_title(args.panel if panel == "Value" and args.panel else panel, color=INK, fontsize=11)
        ax.set_xlabel("log10(relative error)")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(bottom=0)
    axes[0][0].set_ylabel("dn / dlog10 ε")
    axes[0][-1].legend(frameon=False, fontsize=8, labelcolor=INK2)
    if args.title:
        fig.suptitle(args.title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()

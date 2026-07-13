#!/usr/bin/env python3
"""
Visualise ESP benchmark_esp profiler output.

Matches the layout of scripts/performance_analysis.py:
    total in the top column, sub-timers each on their own row,
    medians over runs, IQR outlier removal, t_min + gap bars.

Usage:
    python scripts/performance_analysis_esp.py res.dat
    python scripts/performance_analysis_esp.py --baseline base.dat --prune prune.dat
"""
import re
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')


def remove_outliers_iqr(data, k=1.5):
    """Return a boolean mask keeping values within [Q1-k*IQR, Q3+k*IQR]."""
    import numpy as np
    arr = np.array(data)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (arr >= lo) & (arr <= hi)


@dataclass
class TimerSeries:
    """Time series for one profiler timer across runs."""
    path: str  # e.g. "short_range" or "short_range/build_cell_list"
    t_avg: List[float] = field(default_factory=list)
    t_max: List[float] = field(default_factory=list)
    t_min: List[float] = field(default_factory=list)
    custom1: List[float] = field(default_factory=list)
    custom2: List[float] = field(default_factory=list)
    f_avg: List[float] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.t_avg)

    def median_t_max(self) -> float:
        return float(sorted(self.t_max)[len(self.t_max) // 2]) if self.t_max else 0.0

    def median_t_min(self) -> float:
        return float(sorted(self.t_min)[len(self.t_min) // 2]) if self.t_min else 0.0

    def median_t_avg(self) -> float:
        return float(sorted(self.t_avg)[len(self.t_avg) // 2]) if self.t_avg else 0.0

    def std_gap(self) -> float:
        import statistics
        gaps = [mx - mn for mx, mn in zip(self.t_max, self.t_min)]
        return statistics.stdev(gaps) if len(gaps) > 1 else 0.0


def parse_esp_benchmark(path: str) -> Tuple[Dict[str, str], List["PhaseBlock"]]:
    """Return (metadata dict, list of PhaseBlock) from a benchmark_esp output file."""
    metadata: Dict[str, str] = {}
    phases: List[PhaseBlock] = []
    _pending_header = False

    with open(path) as f:
        for line in f:
            line_stripped = line.rstrip('\n')
            if not line_stripped:
                continue

            if line_stripped.startswith('#'):
                # Config comments e.g. "# key: value"
                m = re.match(r'^#\s*(\S+):\s*(.*)$', line_stripped)
                if m:
                    metadata[m.group(1)] = m.group(2).strip()

                # Phase declaration
                m2 = re.match(r'^#\s*phase:\s*(\S+)', line_stripped)
                if m2:
                    _pending_header = False
                    phases.append(PhaseBlock(name=m2.group(1)))
                continue

            # Non-comment line
            if not phases:
                continue
            phase = phases[-1]

            if not _pending_header:
                _pending_header = True
                phase.cols = next(csv.reader([line_stripped]))
                continue

            # Data row
            cells = next(csv.reader([line_stripped]))
            if not cells:
                continue
            row = {h: v for h, v in zip(phase.cols, cells)}
            phase.rows.append(row)

    return metadata, phases


@dataclass
class PhaseBlock:
    name: str
    cols: List[str] = field(default_factory=list)
    rows: List[Dict[str, str]] = field(default_factory=list)
    timers: Dict[str, TimerSeries] = field(default_factory=dict)

    def col_values(self, key: str):
        vals = []
        for r in self.rows:
            try:
                vals.append(float(r.get(key, 0)))
            except (ValueError, TypeError):
                vals.append(0.0)
        return vals

    def build_timers(self):
        """Aggregate row data into TimerSeries objects, keyed by timer path."""
        # first pass: collect raw series
        raw: Dict[str, Dict[str, List[float]]] = {}
        for row in self.rows:
            for col, val in row.items():
                if '|' not in col:
                    continue
                path, field = col.rsplit('|', 1)
                try:
                    fv = float(val)
                except (ValueError, TypeError):
                    continue
                d = raw.setdefault(path, {})
                d.setdefault(field, []).append(fv)

        # second pass: outlier removal on t_max (same mask applied to all fields)
        self.timers = {}
        for path, field_map in raw.items():
            t_max = field_map.get('t_max', field_map.get('t_avg', []))
            if len(t_max) < 3:
                mask = [True] * len(t_max)
            else:
                mask = remove_outliers_iqr(t_max).tolist()

            ts = TimerSeries(path=path)
            for field in ('t_avg', 't_max', 't_min', 'custom1', 'custom2', 'f_avg'):
                all_vals = field_map.get(field, [])
                filt = [v for v, m in zip(all_vals, mask) if m]
                setattr(ts, field, filt)
            self.timers[path] = ts


def hierarchical_sort(timers: List[str]) -> List[str]:
    """Group children under their parent, depth-first. Preserve first-appearance order."""
    roots = [t for t in timers if '/' not in t]
    children = {}
    for t in timers:
        if '/' in t:
            parent = t.rsplit('/', 1)[0]
            children.setdefault(parent, []).append(t)

    def walk(path):
        result = [path]
        for child in children.get(path, []):
            result.extend(walk(child))
        return result

    result = []
    for r in roots:
        result.extend(walk(r))
    # Add any remaining timers (e.g. orphaned child whose parent isn't present)
    seen = set(result)
    for t in timers:
        if t not in seen:
            result.append(t)
    return result


def build_stage_records(timers: Dict[str, TimerSeries], total_time_median: float) -> List[Dict]:
    """Build records mirroring performance_analysis.py's data DataFrame."""
    # Preserve insertion order (first appearance in columns) instead of sorting.
    stages = hierarchical_sort(list(timers.keys()))
    all_paths = set(stages)
    leaves = {p for p in all_paths
              if not any(o.startswith(p + '/') for o in all_paths if o != p)}

    records = []
    for stage in stages:
        ts = timers[stage]
        t_max_med = ts.median_t_max()
        t_min_med = ts.median_t_min()
        gap_series = [mx - mn for mx, mn in zip(ts.t_max, ts.t_min)]
        gap_series = sorted(gap_series)
        gap_med = float(gap_series[len(gap_series) // 2]) if gap_series else 0.0
        gap_std = ts.std_gap()

        records.append({
            'path': stage,
            'is_leaf': stage in leaves,
            't_max_median': t_max_med,
            't_min_median': t_min_med,
            'gap_median': gap_med,
            'gap_std': gap_std,
            'imbalance_pct': (gap_med / t_max_med * 100) if t_max_med > 0 else 0.0,
            'pct_of_total': (t_max_med / total_time_median * 100) if total_time_median > 0 else 0.0,
            'gap_pct_of_total': (gap_med / total_time_median * 100) if total_time_median > 0 else 0.0,
        })
    return records


def plot_profile_bars(records: List[Dict], total_time_median: float, ax, title: str = ""):
    """Draw horizontal bar chart in the style of performance_analysis.py."""
    import pandas as pd
    data = pd.DataFrame(records).iloc[::-1].reset_index(drop=True)

    ax.barh(data['path'], data['t_min_median'], label='t_min (fastest rank)', color='steelblue')
    ax.barh(data['path'], data['gap_median'], left=data['t_min_median'],
            xerr=data['gap_std'], label='gap +/- 1 sigma', color='salmon',
            ecolor='black', capsize=3)
    ax.axvline(total_time_median, color='black', linestyle='--',
               label=f'total = {total_time_median:.4f}s')

    for i, row in data.iterrows():
        bar_end = row['t_max_median']
        err = row['gap_std']
        x_anchor = bar_end + err + total_time_median * 0.005

        pct_total = row['pct_of_total']
        imbal_pct = row['imbalance_pct']

        label = f"{pct_total:.1f}%"
        if imbal_pct > 0.5:
            label += f"  ({imbal_pct:.0f}% imbal)"

        ax.text(x_anchor, i, label, va='center', ha='left', fontsize=8,
                color='black' if pct_total >= 5 else 'gray')

    for tick, (_, row) in zip(ax.get_yticklabels(), data.iterrows()):
        if row['is_leaf']:
            tick.set_fontweight('bold')

    x_max = data['t_max_median'].max() + data['gap_std'].max()
    ax.set_xlim(right=x_max * 1.35)
    ax.margins(y=0.01)
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.legend()


def print_phase_timings(phase: PhaseBlock, total_time_median: float, indent: str = "  "):
    """Print key:timing pairs for every timer in DFS order."""
    records = build_stage_records(phase.timers, total_time_median)
    for r in records:
        depth = r['path'].count('/')
        prefix = indent * depth
        t = r['t_max_median']
        pct = r['pct_of_total']
        print(f"{prefix}{r['path']}: {t:.6f}s ({pct:.1f}%)")


def plot_phase(phase: PhaseBlock, ax, title: str = ""):
    """Build timer data and draw the profile bar chart for one phase."""
    phase.build_timers()

    total_time_vals = phase.col_values('total_time')
    if total_time_vals:
        total_time_vals = sorted(total_time_vals)
        total_time_median = float(total_time_vals[len(total_time_vals) // 2])
    else:
        total_time_median = 0.0

    records = build_stage_records(phase.timers, total_time_median)

    if not records:
        ax.text(0.5, 0.5, "no profiler data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    plot_profile_bars(records, total_time_median, ax, title)


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Visualise benchmark_esp profiler output")
    parser.add_argument("files", nargs="*", default=["res.dat"], help="benchmark_esp output file(s)")
    args = parser.parse_args()

    files = args.files
    if len(files) == 1:
        # Single-file mode
        meta, phases = parse_esp_benchmark(files[0])
        print(f"Parsed {files[0]}:")
        for k, v in meta.items():
            print(f"  {k}: {v}")

        eval_phases = [p for p in phases if p.name.startswith('eval_')]
        n_plots = len([p for p in eval_phases if p.rows])
        if n_plots == 0:
            print("No eval phases found with data.")
            return

        fig, axes = plt.subplots(n_plots, 1, figsize=(16, max(7, n_plots * 4)))
        axes = [axes] if n_plots == 1 else list(axes)

        for idx, ph in enumerate(eval_phases):
            if not ph.rows:
                continue
            ph.build_timers()
            total_time_vals = ph.col_values('total_time')
            total_time_median = float(sorted(total_time_vals)[len(total_time_vals) // 2]) if total_time_vals else 0.0
            print(f"\n=== {ph.name} ===")
            print_phase_timings(ph, total_time_median)
            title = f"{ph.name}  (n_runs={len(ph.rows)}, prec={meta.get('prec', '?')}, " \
                    f"n={meta.get('n_src', '?')})"
            plot_phase(ph, axes[idx], title)

        fig.set_constrained_layout(True)
        plt.show()
        return

    # Comparison mode: stack bars from all files on one plot per shared phase
    parsed = []
    for path in files:
        meta, phases = parse_esp_benchmark(path)
        label = os.path.splitext(os.path.basename(path))[0]
        parsed.append((label, phases))

    # Find shared phase names across all files (preserve first-file order)
    seen_phases = []
    phase_names = set()
    for _, phases in parsed:
        for p in phases:
            if p.name.startswith('eval_') and p.name not in phase_names:
                seen_phases.append(p.name)
                phase_names.add(p.name)
    shared = [p for p in seen_phases]  # insert order from first file
    if not shared:
        print("No shared eval phases found.")
        return

    fig, axes = plt.subplots(1, len(shared), figsize=(9 * len(shared), max(7, 3 * 3)),
                             constrained_layout=True)
    axes = [axes] if len(shared) == 1 else list(axes)

    colors = ['steelblue', 'salmon', 'seagreen', 'goldenrod', 'mediumpurple',
              'coral', 'cadetblue', 'orchid', 'olivedrab']

    for idx, phase_name in enumerate(shared):
        print(f"\n=== {phase_name} ===")
        for label, phases in parsed:
            ph = next((p for p in phases if p.name == phase_name), None)
            if not ph or not ph.rows:
                continue
            ph.build_timers()
            total_time_vals = ph.col_values('total_time')
            total_time_median = float(sorted(total_time_vals)[len(total_time_vals) // 2]) if total_time_vals else 0.0
            print(f"  [{label}]:")
            print_phase_timings(ph, total_time_median, indent="    ")

        ax = axes[idx]

        # Collect timer names in first-appearance order (first file's order wins).
        ordered_timers = []
        seen_timers = set()
        file_data = []
        for label, phases in parsed:
            ph = next((p for p in phases if p.name == phase_name), None)
            if not ph or not ph.rows:
                continue
            ph.build_timers()
            for t in ph.timers.keys():
                if t not in seen_timers:
                    ordered_timers.append(t)
                    seen_timers.add(t)
            file_data.append((label, ph))

        timers = hierarchical_sort(ordered_timers)
        n = len(timers)
        m = len(file_data)
        if n == 0:
            ax.text(0.5, 0.5, "no profiler data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(phase_name)
            continue

        w = 0.8 / m
        for j, (label, ph) in enumerate(file_data):
            offset = j * w - 0.4 + w / 2
            med = [ph.timers[t].median_t_max() if t in ph.timers else 0 for t in timers]
            ax.barh([i + offset for i in range(n)], med, height=w, label=label,
                    color=colors[j % len(colors)])

        ax.set_yticks(range(n))
        ax.set_yticklabels(timers)
        ax.invert_yaxis()
        ax.set_xlabel("median t_max (s)")
        ax.set_title(phase_name)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    plt.show()


if __name__ == '__main__':
    main()

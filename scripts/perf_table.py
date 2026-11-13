#!/usr/bin/env python3
"""Collate benchmark_kernels CSVs (bench_sweep.sh output or ad-hoc runs) into markdown tables.

Each CSV becomes one record: its '# key: value' header plus a statistic (default median) of the
build and eval times, dropping the first timed run (JIT compilation, first touch) when there is
more than one. Throughputs are points per second in MP/s; 'total' is one build plus one eval
(dmk: tree build, esp: plan creation, direct: no build).

Keys usable in --rows/--cols/--split/--where: label solver path kernel precision eps dist n ranks
threads grad pin host gpu_model cpu_model, plus any raw header key. Direct runs have no tolerance and
are repeated under every tolerance in the table.

Values (--value): eval build total (MP/s), eval_s build_s total_s (seconds), l2 max_rel (errors).

Examples:
  # The slide table: one throughput column per machine, in the order the directories are given
  perf_table.py bench_results/rtx6000 bench_results/h100 bench_results/genoa-mpi --where solver=dmk

  # Crossover against direct sum: N down the side, one column per solver, one table per kernel
  perf_table.py bench_results/rtx6000 --rows n --cols solver --split kernel,precision,eps --where dist=Uniform

  # Same, several machines and one esp cutoff scale (--where skips records without the key)
  perf_table.py bench_results/rtx6000 bench_results/h100 --rows n --cols label,solver \
      --split kernel,precision,eps --where dist=Uniform --where esp_rc_spec=scale=2

  # Eval-only vs build+eval throughput side by side
  perf_table.py bench_results/rtx6000 --value eval,total --rename LobedNSphere=SphereShell
"""
import argparse
import math
import pathlib
import sys

import pandas as pd

DIST = {0: 'Uniform', 1: 'LobedNSphere', 2: 'NCubePartialFacet', 3: 'GradedVolume'}
SOLVER = {'dmk_time': 'dmk', 'direct_time': 'direct', 'total_time': 'esp'}
BUILD = ('build_time', 'plan_time')
VALUES = {'eval': 'MP/s', 'build': 'build MP/s', 'total': 'total MP/s', 'eval_s': 'eval s',
          'build_s': 'build s', 'total_s': 'total s', 'l2': 'rel L2', 'max_rel': 'max rel'}
NAMES = {'kernel': 'Kernel', 'precision': 'Precision', 'eps': 'Tolerance', 'dist': 'Distribution',
         'n': 'Points', 'solver': 'Solver', 'label': 'Label', 'path': 'Path', 'ranks': 'Ranks',
         'threads': 'Threads', 'grad': 'Grad'}


def read_csv(path):
    """Header metadata and one DataFrame per table, keyed by the table's *_time column."""
    meta, tables, key = {}, {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith('#'):
            k, sep, v = line[1:].partition(':')
            k = k.strip()
            if sep and k and ' ' not in k:
                meta.setdefault(k, v.strip())
        elif line[:1].isalpha():
            cols = line.split(',')
            key = next(c for c in cols if c.endswith('_time'))
            tables[key] = (cols, [])
        elif line and key:
            tables[key][1].append(line.split(','))
    return meta, {k: pd.DataFrame(rows, columns=cols).apply(pd.to_numeric, errors='coerce')
                  for k, (cols, rows) in tables.items()}


def record(path, stat, skip):
    meta, tables = read_csv(path)
    ev = next((k for k in tables if k in SOLVER), None)
    if ev is None or tables[ev].empty:
        print(f'warning: no eval timings in {path}', file=sys.stderr)
        return None

    def reduce(df, col):
        s = df[col].dropna()
        return getattr(s.iloc[skip:] if len(s) > skip else s, stat)()

    solver = SOLVER[ev]
    n = int(float(meta['n_src'])) + (int(float(meta.get('n_trg', 0))) if solver != 'esp' else 0)
    eval_s = reduce(tables[ev], ev)
    bk = next((k for k in BUILD if k in tables), None)
    build_s = reduce(tables[bk], bk) if bk else (0.0 if solver == 'direct' else math.nan)
    err = tables[ev]
    l2 = next((err[c].median() for c in ('src_l2_rel_err', 'l2_rel_err') if c in err), math.nan)
    max_rel = err['src_max_rel_err'].median() if 'src_max_rel_err' in err else math.nan
    rec = dict(meta)
    rec.update(
        file=str(path), label=meta.get('label', path.parent.name), solver=solver, kernel=meta['kernel'],
        precision={'float': 'FP32', 'double': 'FP64'}[meta.get('precision', meta.get('prec'))],
        eps=math.nan if solver == 'direct' else float(meta['eps']),
        dist=DIST.get(int(meta.get('dist', 0)), meta.get('dist')), n=n,
        path='gpu' if meta.get('eval_path') in ('1', 'gpu') else 'cpu',
        ranks=int(meta.get('mpi_ranks', 1)),
        threads=int(meta.get('omp_threads_per_rank', meta.get('omp_threads', 0))),
        grad=meta.get('with_grad', meta.get('bench_grad')) in ('1', 'true'),
        pin={'1': True, '0': False}.get(meta.get('pin_host')),
        eval_s=eval_s, build_s=build_s, total_s=eval_s + build_s,
        eval=n / eval_s / 1e6, build=n / build_s / 1e6 if build_s else math.nan,
        total=n / (eval_s + build_s) / 1e6, l2=l2, max_rel=max_rel)
    return rec


def sci(x):
    if isinstance(x, str) or math.isnan(x):
        return '—' if not isinstance(x, str) else x
    m, e = f'{x:.6e}'.split('e')
    return f"{m.rstrip('0').rstrip('.')}E{int(e)}"


def fmt_value(v, value):
    if math.isnan(v):
        return '—'
    if value in ('l2', 'max_rel'):
        return f'{v:.1e}'
    return f'{v:.0f}' if v >= 1000 else f'{v:.3g}'


def fmt_key(v, key):
    if key in ('eps', 'n'):
        return sci(v)
    return '—' if isinstance(v, float) and math.isnan(v) else str(v)


def matches(col, vals):
    try:
        return col.astype(float).isin([float(v) for v in vals])
    except ValueError:
        return col.astype(str).isin(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', type=pathlib.Path, help='CSV files or directories (searched recursively)')
    ap.add_argument('--rows', default='kernel,precision,eps,dist,n')
    ap.add_argument('--cols', default='label')
    ap.add_argument('--split', default='', help='keys to make one table per value of')
    ap.add_argument('--value', default='eval', help='comma list of: ' + ' '.join(VALUES))
    ap.add_argument('--where', action='append', default=[], help='KEY=V1[,V2...]; repeatable')
    ap.add_argument('--stat', default='median', choices=('median', 'min', 'mean'))
    ap.add_argument('--skip', type=int, default=1, help='leading timed runs to drop (default 1)')
    ap.add_argument('--rename', action='append', default=[], help='OLD=NEW for any printed name; repeatable')
    ap.add_argument('--format', default='md', choices=('md', 'csv'))
    ap.add_argument('--flat', action='store_true', help='print every record as CSV instead of tables')
    args = ap.parse_args()

    files = [f for p in args.paths for f in (sorted(p.rglob('*.csv')) if p.is_dir() else [p])]
    df = pd.DataFrame([r for f in files if (r := record(f, args.stat, args.skip))])
    if df.empty:
        sys.exit('no records')
    split = [k for k in args.split.split(',') if k]
    rows = [k for k in args.rows.split(',') if k]
    cols = [k for k in args.cols.split(',') if k]
    values = args.value.split(',')
    keys = split + rows + cols
    where = [w.split('=', 1) for w in args.where]

    if not args.flat and 'eps' in keys + [k for k, _ in where]:
        eps = df.loc[df.solver != 'direct', 'eps'].dropna().unique()
        if len(eps):
            d = df[df.solver == 'direct']
            df = pd.concat([df[df.solver != 'direct']] + [d.assign(eps=e) for e in eps])
    # Records without the key (e.g. esp_rc_spec on dmk runs) are not filtered by it.
    for k, v in where:
        df = df[df[k].isna() | matches(df[k], v.split(','))]
    if df.empty:
        sys.exit('no records match --where (dist and precision are names, e.g. dist=Uniform precision=FP32)')

    if args.flat:
        first = ['label', 'solver', 'path', 'kernel', 'precision', 'eps', 'dist', 'n', 'ranks', 'threads', 'grad', 'pin']
        df[first + list(VALUES) + ['file']].to_csv(sys.stdout, index=False)
        return

    # Strings keep the order they first appear in, so table columns follow the order of the paths.
    for k in keys:
        if not pd.api.types.is_numeric_dtype(df[k]):
            df[k] = pd.Categorical(df[k], categories=list(dict.fromkeys(df[k].dropna())))

    g = df.groupby(keys, dropna=False, observed=True)
    dup = g.size()
    if (dup > 1).any():
        print(f'warning: {int((dup > 1).sum())} cell(s) collate more than one run (median taken); '
              'add a distinguishing key to --rows/--cols/--split or filter with --where', file=sys.stderr)
    agg = g[values].median()

    ren = dict(r.split('=', 1) for r in args.rename)
    name = lambda s: ren.get(str(s), str(s))
    sep = ',' if args.format == 'csv' else ' | '

    groups = agg.groupby(level=split, dropna=False, observed=True) if split else [((), agg)]
    for gkey, sub in groups:
        t = sub.droplevel(split) if split else sub
        if cols:
            t = t.unstack(cols).dropna(axis=1, how='all')
            t.columns = t.columns.reorder_levels(list(range(1, t.columns.nlevels)) + [0])
            t = t.sort_index(axis=1, level=list(range(t.columns.nlevels - 1)), sort_remaining=False)
        colkeys = [c if isinstance(c, tuple) else (c,) for c in t.columns]
        header = [name(NAMES.get(k, k)) for k in rows]
        header += [' '.join([name(fmt_key(x, k)) for x, k in zip(c[:-1], cols)] + [name(VALUES[c[-1]])])
                   if cols else name(VALUES[c[0]]) for c in colkeys]

        if split:
            gkey = gkey if isinstance(gkey, tuple) else (gkey,)
            title = ', '.join(f'{name(NAMES.get(k, k))}={name(fmt_key(v, k))}' for k, v in zip(split, gkey))
            print(f'# {title}' if args.format == 'csv' else f'### {title}\n')
        if args.format == 'md':
            print('| ' + sep.join(header) + ' |')
            print('|' + '|'.join([':-:'] * len(header)) + '|')
        else:
            print(sep.join(header))
        for idx, row in t.iterrows():
            idx = idx if isinstance(idx, tuple) else (idx,)
            cells = [name(fmt_key(v, k)) for v, k in zip(idx, rows)]
            cells += [fmt_value(row[c], c[-1] if cols else c) for c in t.columns]
            print(('| ' + sep.join(cells) + ' |') if args.format == 'md' else sep.join(cells))
        print()


if __name__ == '__main__':
    main()

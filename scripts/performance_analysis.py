import pandas as pd
import yaml
import pprint
import sys
from dataclasses import dataclass, fields
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')


@dataclass
class RunConfig:
    mpi_ranks: int = 1
    omp_threads_per_rank: int = 1
    n_src: int = 0
    precision: str = 'float'
    uniform_dist: bool = False
    eps: float = 1e-5
    n_per_leaf_dmk: int = 100
    n_runs: int = 50
    pvfmm_enabled: bool = False
    n_per_leaf_pvfmm: int = 500
    m_pvfmm: int = 6
    direct_enabled: bool = False
    n_direct: int = -1


def read_csv_with_yaml_header(path):
    """Return (metadata_dict, DataFrame) from a CSV with '# ---' YAML header."""
    yaml_lines = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                yaml_lines.append(line.lstrip('#').strip())
            else:
                break
    meta = yaml.safe_load('\n'.join(yaml_lines)) if yaml_lines else {}
    cfg = RunConfig(**{k: v for k, v in meta.items()
                       if k in {f.name for f in fields(RunConfig)}})
    df = pd.read_csv(path, comment='#')
    return cfg, df


if len(sys.argv) == 2:
    csv_file = sys.argv[1]
else:
    csv_file = 'res.csv'

cfg, df = read_csv_with_yaml_header(csv_file)
pprint.pprint(cfg)


def remove_outliers_iqr(df, columns=None, k=1.5):
    if columns is None:
        columns = df.select_dtypes(include='number').columns
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        mask &= df[col].between(q1 - k * iqr, q3 + k * iqr)
    return df[mask]


df = remove_outliers_iqr(df, columns=['dmk_time'])

# Handle custom profiler keys: these are nanosecond timings that need to be
# injected as synthetic t_min/t_max columns (they don't have per-rank breakdowns,
# so t_min == t_max).
custom_key_map = {
    'pdmk_tree_eval/downward_pass/expansion_propagation_and_eval|custom1':
        'pdmk_tree_eval/downward_pass/expansion_propagation_and_eval/form_outgoing_expansions',
    'pdmk_tree_eval/downward_pass/expansion_propagation_and_eval|custom2':
        'pdmk_tree_eval/downward_pass/expansion_propagation_and_eval/form_eval_expansions',
}
for raw_col, nice_path in custom_key_map.items():
    if raw_col in df.columns:
        scaled = df[raw_col] / 1e9
        df[f'{nice_path}|t_min'] = scaled
        df[f'{nice_path}|t_max'] = scaled

df_tree = df[df.columns[df.columns.str.contains(r'\|(t_min|t_max)$', regex=True)]]
total_time_median = df['dmk_time'].median()

print(df.describe())

cols = df_tree.columns.tolist()
stages = list(dict.fromkeys(c.rsplit('|', 1)[0] for c in cols))

# Synthetic children from custom keys ended up at the end of the list.
# Move each one to right after its parent, preserving insertion order.
inserted = {}
for nice_path in custom_key_map.values():
    if nice_path in stages:
        parent = nice_path.rsplit('/', 1)[0]
        stages.remove(nice_path)
        parent_idx = stages.index(parent)
        offset = inserted.get(parent, 0)
        stages.insert(parent_idx + 1 + offset, nice_path)
        inserted[parent] = offset + 1

# Identify leaves
all_paths = set(stages)
leaves = set(p for p in all_paths
             if not any(o.startswith(p + '/') for o in all_paths if o != p))

records = []
for stage in stages:
    t_max_med = df_tree[f'{stage}|t_max'].median()
    t_min_med = df_tree[f'{stage}|t_min'].median()
    gap_series = df_tree[f'{stage}|t_max'] - df_tree[f'{stage}|t_min']
    gap_med = gap_series.median()
    gap_std = gap_series.std()
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

data = pd.DataFrame(records).iloc[::-1].reset_index(drop=True)

# Print timing summary to terminal
print(f"\n{'Stage':<80s} {'t_min (s)':>10s} {'t_max (s)':>10s}")
print('-' * 102)
for _, row in data.iloc[::-1].iterrows():
    print(f"{row['path']:<80s} {row['t_min_median']:10.5f} {row['t_max_median']:10.5f}")
print(f"{'TOTAL':<80s} {total_time_median:10.5f}")
print()

fig, ax = plt.subplots(figsize=(12, max(6, len(data) * 0.4)))

ax.barh(data['path'], data['t_min_median'],
        label='t_min (fastest rank)', color='steelblue')
ax.barh(data['path'], data['gap_median'], left=data['t_min_median'],
        xerr=data['gap_std'], label='gap +/- 1 sigma', color='salmon',
        ecolor='black', capsize=3)
ax.axvline(total_time_median, color='black', linestyle='--',
           label=f'total = {total_time_median:.4f}s')

# Annotate each bar with % of total and imbalance %
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

# Add some right margin for the annotations
x_max = data['t_max_median'].max() + data['gap_std'].max()
ax.set_xlim(right=x_max * 1.35)

ax.margins(y=0.01)
fig.subplots_adjust(left=0.01, right=0.98)
ax.set_xlabel('Time (s)')
ax.set_title('Load Imbalance (median over runs)')
ax.legend()
fig.tight_layout()
plt.show()

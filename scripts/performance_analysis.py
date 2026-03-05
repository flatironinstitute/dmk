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


df = df[10:]
df_tree = df[df.columns[df.columns.str.contains(r'(t_min|t_max)$')]]
total_time_median = df['dmk_time'].median()
print(df.describe())

cols = df_tree.columns.tolist()
stages = list(dict.fromkeys(c.rsplit('|', 1)[0] for c in cols))

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

fig, ax = plt.subplots(figsize=(12, max(6, len(data) * 0.4)))
ax.barh(data['path'], data['t_min_median'],
        label='t_min (fastest rank)', color='steelblue')
ax.barh(data['path'], data['gap_median'], left=data['t_min_median'],
        xerr=data['gap_std'], label='gap ± 1σ', color='salmon',
        ecolor='black', capsize=3)
ax.axvline(total_time_median, color='black', linestyle='--',
           label=f'total = {total_time_median:.4f}s')

for tick, (_, row) in zip(ax.get_yticklabels(), data.iterrows()):
    if row['is_leaf']:
        tick.set_fontweight('bold')

ax.margins(y=0.01)
fig.subplots_adjust(left=0.01, right=0.98)
ax.set_xlabel('Time (s)')
ax.set_title('Load Imbalance (median over runs)')
ax.legend()
fig.tight_layout()
plt.show()

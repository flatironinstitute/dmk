import pandas as pd
import yaml
import pprint
import sys
from dataclasses import dataclass, fields

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
print(df.describe())

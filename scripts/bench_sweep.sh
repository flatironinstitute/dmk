#!/usr/bin/env bash
# Sweep benchmark_kernels over a grid of configurations inside an existing allocation (salloc or
# sbatch), or directly on a workstation. Writes one CSV per configuration; a configuration whose CSV
# already exists is skipped, so resubmitting a timed-out job resumes the sweep. Collate the results
# with scripts/perf_table.py.
set -uo pipefail

usage() {
    cat <<'EOF'
Usage: bench_sweep.sh --label NAME [options]

  --label NAME        Machine/config name; becomes a table column (required)
  --solver LIST       dmk,esp,direct (default dmk)
  --path c|g          CPU or GPU eval path (default c)
  --kernel LIST       e.g. stokeslet,stresslet,laplace_dipole (default laplace)
  --prec LIST         f,d (default f)
  --eps LIST          Tolerances; not used by direct (default 1e-5)
  --dist LIST         Point distributions, see benchmark_kernels -h; esp is uniform only (default 0)
  --n LIST            Number of points, e.g. 1e4,1e5,1e6 (default 1e6); run largest first
  --runs N            Timed runs for dmk (default 10)
  --esp-runs N        Timed runs for esp, plan creation included (default 3)
  --direct-runs N     Timed runs for direct (default 3)
  --esp-rc-scale LIST esp cutoff r_c = S / cbrt(N), capped at 1/3 (constant neighbours per point)
  --esp-rc R|auto     Fixed esp cutoff, or auto for the benchmark's accuracy search over 8 cutoffs
                      at full N on the CPU (slow at large N). esp needs this or --esp-rc-scale.
  --ranks N           MPI ranks (default $SLURM_NTASKS or 1); esp is skipped when > 1
  --threads N         OpenMP threads per rank (default $SLURM_CPUS_PER_TASK or nproc/ranks)
  --extra "ARGS"      Extra benchmark_kernels arguments for every run, e.g. "-g -n 280 -D 1000"
  --launcher "CMD"    Launch prefix (default: srun inside a Slurm job, mpirun when ranks > 1 outside one)
  --no-pin            GPU runs use pageable host buffers (default: --pin, page-locked; files get _nopin)
  --tune-cache PATH   GPU autotune cache (default $DMK_JIT_AUTOTUNE_CACHE, else
                      $HOME/.cache/dmk/jit_autotune.jsonl so every node shares it)
  --bin PATH          Benchmark binary (default <repo>/build/examples/benchmark_kernels)
  --out DIR           Output directory (default ./bench_results/<label>)
  --dry-run           Print the commands only

dmk runs with --bench-build and esp with --bench-plan, so every CSV carries build and eval times.

The GPU autotuner runs on the first encounter of each (GPU, kernel, precision, tolerance) and caches
the winner regardless of N, so the first configuration per key is slow and the N it runs at decides
the tuning for all N. That is why N runs largest first. The tuned run is the first timed run, which
perf_table.py drops.

Examples:
  salloc -p gpu --gpus=1 -c 16 ...
  scripts/bench_sweep.sh --label rtx6000 --path g --solver dmk,esp,direct --esp-rc-scale 2 \
      --kernel stokeslet,stresslet --n 1e3,1e4,1e5,1e6

  sbatch -N1 --ntasks=12 -c 8 --wrap "scripts/bench_sweep.sh --label genoa-mpi --kernel stokeslet --n 1e6"

Multiple GPUs need one GPU per rank, e.g. allocate with --gpus-per-task=1.
EOF
}

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
label="" solvers=dmk path=c kernels=laplace precs=f eps_list=1e-5 dists=0 ns=1e6
runs=10 esp_runs=3 direct_runs=3 esp_rc="" esp_rc_scale=""
ranks=${SLURM_NTASKS:-1} threads=${SLURM_CPUS_PER_TASK:-} extra="" launcher=""
tune_cache=${DMK_JIT_AUTOTUNE_CACHE:-$HOME/.cache/dmk/jit_autotune.jsonl}
bin=$repo/build/examples/benchmark_kernels out="" dry=0 pin=1

while [[ $# -gt 0 ]]; do
    case $1 in
    --label) label=$2; shift 2 ;;
    --solver) solvers=$2; shift 2 ;;
    --path) path=$2; shift 2 ;;
    --kernel) kernels=$2; shift 2 ;;
    --prec) precs=$2; shift 2 ;;
    --eps) eps_list=$2; shift 2 ;;
    --dist) dists=$2; shift 2 ;;
    --n) ns=$2; shift 2 ;;
    --runs) runs=$2; shift 2 ;;
    --esp-runs) esp_runs=$2; shift 2 ;;
    --direct-runs) direct_runs=$2; shift 2 ;;
    --esp-rc) esp_rc=$2; shift 2 ;;
    --esp-rc-scale) esp_rc_scale=$2; shift 2 ;;
    --ranks) ranks=$2; shift 2 ;;
    --threads) threads=$2; shift 2 ;;
    --extra) extra=$2; shift 2 ;;
    --launcher) launcher=$2; shift 2 ;;
    --no-pin) pin=0; shift ;;
    --tune-cache) tune_cache=$2; shift 2 ;;
    --bin) bin=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --dry-run) dry=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

die() { echo "$*" >&2; exit 1; }
[[ -n $label ]] || die "--label is required"
[[ -x $bin ]] || die "benchmark binary not found: $bin"
if [[ ,$solvers, == *,esp,* ]]; then
    [[ -n $esp_rc || -n $esp_rc_scale ]] || die "esp needs --esp-rc-scale S, --esp-rc R or --esp-rc auto"
    [[ -z $esp_rc || -z $esp_rc_scale ]] || die "--esp-rc and --esp-rc-scale are exclusive"
fi
threads=${threads:-$(($(nproc) / ranks))}
out=${out:-bench_results/$label}
mkdir -p "$out"

export OMP_NUM_THREADS=$threads DMK_JIT_AUTOTUNE_CACHE=$tune_cache
launch_cmd() { # $1 = ranks
    if [[ -n $launcher ]]; then
        echo "$launcher"
    elif [[ -n ${SLURM_JOB_ID:-} ]]; then
        echo "srun -n $1 -c $threads --cpu-bind=cores"
    elif (($1 > 1)); then
        echo "mpirun -n $1 --bind-to none"
    fi
}
if [[ -n ${SLURM_JOB_ID:-} && -z $launcher ]]; then
    export OMP_PLACES=${OMP_PLACES:-cores} OMP_PROC_BIND=${OMP_PROC_BIND:-close}
fi

cpu_model=$(lscpu | sed -n 's/^Model name: *//p' | head -1)
gpu_model=$([[ $path == g ]] && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
dmk_git=$(git -C "$repo" describe --always --dirty 2>/dev/null)
xtag=$([[ -n $extra ]] && printf '_x%s' "$(printf %s "$extra" | tr -cs 'A-Za-z0-9.=' '_')")

IFS=, read -ra S <<<"$solvers"
IFS=, read -ra K <<<"$kernels"
IFS=, read -ra P <<<"$precs"
IFS=, read -ra E <<<"$eps_list"
IFS=, read -ra D <<<"$dists"
IFS=, read -ra RCS <<<"${esp_rc_scale:-$esp_rc}"
mapfile -t NS < <(tr , '\n' <<<"$ns" | sort -g -r)

# One configuration: $1 solver, $2 kernel, $3 prec, $4 eps, $5 dist, $6 N, $7 esp cutoff spec.
run_one() {
    local solver=$1 kernel=$2 prec=$3 eps=$4 dist=$5 n=$6 rcv=$7 rc="" tag=""
    local args=(--solver "$solver" -N "$n" -k "$kernel" -t "$prec" -p "$path")
    case $solver in
    dmk) args+=(-e "$eps" -u "$dist" -r "$runs" --bench-build) ;;
    direct) args+=(-u "$dist" -r "$direct_runs") ;;
    esp)
        args+=(-e "$eps" -r "$esp_runs" --bench-plan)
        [[ $path == g ]] && args+=(--skip-cpu-baseline)
        if [[ -n $esp_rc_scale ]]; then
            rc=$(awk -v s="$rcv" -v n="$n" 'BEGIN { r = s / exp(log(n) / 3); printf "%.4g", (r > 1/3 ? 1/3 : r) }')
            tag=_rcs$rcv
        elif [[ $rcv != auto ]]; then
            rc=$rcv tag=_rc$rcv
        fi
        [[ -n $rc ]] && args+=(-c "$rc")
        ;;
    esac
    if [[ $path == g ]]; then
        ((pin)) && args+=(--pin) || tag+=_nopin
    fi
    # shellcheck disable=SC2206
    args+=($extra)

    local name="${solver}_${kernel}_${prec}_eps${eps}_dist${dist}_N${n}_${path}_np${ranks}x${threads}${tag}${xtag}"
    local f=$out/$name.csv
    # shellcheck disable=SC2207
    local cmd=($(launch_cmd "$ranks") "$bin" "${args[@]}")
    if [[ -e $f ]]; then
        echo "exists, skipping: $f"
        return
    fi
    echo "$(date +%T) $name"
    if ((dry)); then
        echo "  ${cmd[*]}"
        return
    fi
    {
        echo "# label: $label"
        echo "# host: $(hostname -s)"
        echo "# cpu_model: $cpu_model"
        echo "# gpu_model: ${gpu_model:-none}"
        echo "# slurm_job: ${SLURM_JOB_ID:-none}"
        echo "# dmk_git: $dmk_git"
        echo "# date: $(date -Is)"
        echo "# tune_cache: $tune_cache"
        [[ $solver == esp ]] && echo "# esp_rc_spec: ${esp_rc_scale:+scale=}$rcv"
        echo "# cmd: ${cmd[*]}"
    } >"$f.part"
    local t0=$SECONDS
    if "${cmd[@]}" >>"$f.part" 2>"$out/$name.err"; then
        mv "$f.part" "$f"
        rm -f "$out/$name.failed"
        echo "  done in $((SECONDS - t0)) s"
    else
        mv "$f.part" "$out/$name.failed"
        failed+=("$name")
        echo "  FAILED after $((SECONDS - t0)) s, see $out/$name.err"
    fi
}

failed=()
for n in "${NS[@]}"; do
    for solver in "${S[@]}"; do
        if [[ $solver == esp ]] && ((ranks > 1)); then
            echo "skipping esp: single rank only (ranks=$ranks)"
            continue
        fi
        # direct has no tolerance, esp generates its own uniform points, and only esp has a cutoff.
        eps_loop=("${E[@]}") dist_loop=("${D[@]}") rc_loop=(-)
        [[ $solver == direct ]] && eps_loop=(-)
        [[ $solver == esp ]] && dist_loop=(0) rc_loop=("${RCS[@]}")
        for kernel in "${K[@]}"; do
            for prec in "${P[@]}"; do
                for eps in "${eps_loop[@]}"; do
                    for dist in "${dist_loop[@]}"; do
                        for rcv in "${rc_loop[@]}"; do
                            run_one "$solver" "$kernel" "$prec" "$eps" "$dist" "$n" "$rcv"
                        done
                    done
                done
            done
        done
    done
done

if ((${#failed[@]})); then
    echo "${#failed[@]} failed configuration(s):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi

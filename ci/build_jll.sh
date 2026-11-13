#!/bin/bash
# Builds DMK_jll tarballs with ci/build_tarballs.jl. Everything goes under $DMK_JLL_DIR, which should be on local
# disk: the Yggdrasil checkout, BinaryBuilder's build/ and products/, and (through the first depot) compiler shards,
# CUDA SDKs and the --deploy=local JLL. ~/.julia stays on the depot path for the packages and registries already
# installed there.
#   DMK_JLL_DIR=<dir> ci/build_jll.sh                                  the default single CPU platform below
#   DMK_JLL_DIR=<dir> ci/build_jll.sh --debug <flags...> <t1,t2,...>   build_tarballs.jl flags and triplets; --debug
#                                                                      opens a shell on failure; no triplets builds
#                                                                      every platform
set -euo pipefail

: "${DMK_JLL_DIR:?set DMK_JLL_DIR to a local-disk directory for the depot, Yggdrasil checkout and build products}"
dmk=$(cd "$(dirname "$0")/.." && pwd)
work=$(mkdir -p "$DMK_JLL_DIR" && cd "$DMK_JLL_DIR" && pwd)
export YGGDRASIL_DIR=$work/yggdrasil
export JULIA_DEPOT_PATH=$work/depot:$HOME/.julia:

if [ ! -d "$YGGDRASIL_DIR" ]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/JuliaPackaging/Yggdrasil.git "$YGGDRASIL_DIR"
    git -C "$YGGDRASIL_DIR" sparse-checkout set platforms .ci/bb1_project
fi
julia --project="$YGGDRASIL_DIR" -e 'using Pkg; Pkg.instantiate()'

if [ $# -eq 0 ]; then
    set -- --verbose --deploy=local x86_64-linux-gnu-cxx11-march+avx2-cuda+none-mpi+mpich
fi
flags=()
triplets=()
for arg in "$@"; do
    if [[ $arg == --* ]]; then
        flags+=("$arg")
    else
        IFS=, read -ra triplets <<< "$arg"
    fi
done

run() {
    # BinaryBuilder fixes its --deploy=local directory when it is precompiled, so repoint it at this depot.
    julia --project="$YGGDRASIL_DIR" \
        -e 'import BinaryBuilder, Pkg; BinaryBuilder.devdir[] = Pkg.devdir(); include(popfirst!(ARGS))' \
        "$dmk/ci/build_tarballs.jl" "$@"
}

# BinaryBuilder writes build/ and products/ into the working directory.
cd "$work"
# A per-platform recipe hands build_tarballs every triplet in ARGS, so Yggdrasil CI runs one triplet at a time.
if [ ${#triplets[@]} -eq 0 ]; then
    run "${flags[@]}"
fi
for t in "${triplets[@]}"; do
    run "${flags[@]}" "$t"
done

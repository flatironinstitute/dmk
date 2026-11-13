#!/bin/bash
# Writes bindings/DMK_jll for the tarballs that ci/build_jll.sh left in $DMK_JLL_DIR/products, with artifact URLs
# under <prefix>/<version>+<build>/:
#   DMK_JLL_DIR=<dir> ci/publish_jll.sh https://users.flatironinstitute.org/~rblackwell/<dir>
# The build number is bindings/DMK_jll's, plus one when any tarball it lists has changed. Then copy the tarballs
# (not DMK-logs) to the directory that serves the URL it prints.
set -euo pipefail

: "${DMK_JLL_DIR:?set DMK_JLL_DIR to the directory ci/build_jll.sh used}"
dmk=$(cd "$(dirname "$0")/.." && pwd)
work=$(cd "$DMK_JLL_DIR" && pwd)
export YGGDRASIL_DIR=$work/yggdrasil
export JULIA_DEPOT_PATH=$work/depot:$HOME/.julia:

exec julia --project="$YGGDRASIL_DIR" "$dmk/ci/publish_jll.jl" "$1" "$work/products"

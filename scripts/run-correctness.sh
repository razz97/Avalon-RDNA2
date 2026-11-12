#!/usr/bin/env bash
# Synthetic correctness of the RDNA2 sparse prefill attention on the real gfx1030 GPU. Usage: run-correctness.sh [--quick] [--bench]
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
V="${ROCM_VENV:-$HOME/.local/share/lifeos-inference/avalon-q8-prep-e09e775/rocm-7.14.1-venv}"
R="$(ls -d "$V"/lib/python3*/site-packages/_rocm_sdk_devel)"
B="${BUILD_DIR:-$SRC/build-hip-gfx1030-qsa}"
export LD_LIBRARY_PATH="$R/lib:$R/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
exec "$B/bin/test-qsa-rdna2" "$@"

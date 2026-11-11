#!/usr/bin/env bash
# Isolated HIP build for gfx1030 of the RDNA2 QSA prefill branch. Usage: build-gfx1030.sh [cmake target ...]
# ROCM_VENV: ROCm SDK venv (default: the Avalon q8-prep one). CMAKE: cmake binary. BUILD_DIR: default <src>/build-hip-gfx1030-qsa
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
V="${ROCM_VENV:-$HOME/.local/share/lifeos-inference/avalon-q8-prep-e09e775/rocm-7.14.1-venv}"
CMAKE="${CMAKE:-$HOME/.local/share/lifeos-inference/results/qwen38-grammar-experiment/tools-venv/bin/cmake}"
R="$(ls -d "$V"/lib/python3*/site-packages/_rocm_sdk_devel)"
B="${BUILD_DIR:-$SRC/build-hip-gfx1030-qsa}"
export ROCM_PATH="$R" HIP_PATH="$R" LD_LIBRARY_PATH="$R/lib:$R/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
"$CMAKE" -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 \
  -DGGML_CUDA_FA=ON -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON -DGGML_HIP_NO_VMM=ON \
  -DCMAKE_C_COMPILER="$V/bin/amdclang" -DCMAKE_CXX_COMPILER="$V/bin/amdclang++" \
  -DCMAKE_HIP_COMPILER="$R/lib/llvm/bin/clang++" -DCMAKE_PREFIX_PATH="$R"
TARGETS=()
for t in "$@"; do TARGETS+=(--target "$t"); done
nice -n 15 "$CMAKE" --build "$B" -j "${JOBS:-8}" "${TARGETS[@]}"

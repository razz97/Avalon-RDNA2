#!/usr/bin/env bash
# A/B real-model prefill. Usage: run-ab-bench.sh A|B PROMPT_TOKENS BATCH UBATCH [extra oneshot-ab.py args]
# A = baseline (feature off), B = LLAMA_QSA_RDNA2_PREFILL=1 (min kv 32768). Same model/args otherwise. Does not touch services.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
ARM="$1"; TOK="$2"; BATCH="$3"; UB="$4"; shift 4
V="${ROCM_VENV:-$HOME/.local/share/lifeos-inference/avalon-q8-prep-e09e775/rocm-7.14.1-venv}"
export AB_BIN_DIR="${BUILD_DIR:-$SRC/build-main}/bin"
common='"GGML_SCHED_BORROW_COMPUTE":"1","LLAMA_QSA_TILES":"0","LLAMA_QSA_TILES_ALL":"0"'
if [ "$ARM" = A ]; then export AB_ENV="{$common,\"LLAMA_QSA_RDNA2_PREFILL\":\"0\"}"
else export AB_ENV="{$common,\"LLAMA_QSA_RDNA2_PREFILL\":\"1\",\"LLAMA_QSA_RDNA2_MIN_KV\":\"32768\"}"; fi
OUT="$SRC/results/runs/${ARM}-${TOK}-b${BATCH}-ub${UB}-$(date +%H%M%S)"
exec python3 "$SRC/scripts/oneshot-ab.py" --prompt-tokens "$TOK" --batch "$BATCH" --ubatch "$UB" --predict 128 --out "$OUT" "$@"

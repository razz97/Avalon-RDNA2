#!/usr/bin/env bash
# R3 matrix: (no rocprof wrap: kernel-trace made it ~20x slower; dispatch proof = r3-dispatch.sh) A short, B short (MIN_KV=1024), A long, B long (MIN_KV=32768). One process per arm/class.
set -u; cd "$(dirname "$0")/.."; source scripts/r3-env.sh
V="${ROCM_VENV:-$HOME/.local/share/lifeos-inference/avalon-q8-prep-e09e775/rocm-7.14.1-venv}"
O=results-r3/runs; mkdir -p $O
wrap() { echo "$V/bin/rocprofv3 --kernel-trace --stats -d $PWD/$O/$1/prof -o k --output-format csv --"; }
scripts/r3-dispatch.sh > results-r3/dispatch.log 2>&1
scripts/run-correctness.sh > results-r3/s0-correctness.log 2>&1; grep -E 'RESULT|FAIL' results-r3/s0-correctness.log > results-r3/s0-result.txt
python3 scripts/r3-run.py --arm A --spec results-r3/prompts/short.spec --out $O/A-short --tag A-short > $O/A-short.out 2>&1
python3 scripts/r3-run.py --arm B --min-kv 1024 --spec results-r3/prompts/short.spec --out $O/B-short --tag B-short > $O/B-short.out 2>&1
python3 scripts/r3-run.py --arm A --spec results-r3/prompts/long.spec --out $O/A-long --tag A-long > $O/A-long.out 2>&1
python3 scripts/r3-run.py --arm B --min-kv 32768 --spec results-r3/prompts/long.spec --out $O/B-long --tag B-long > $O/B-long.out 2>&1
echo done > $O/done

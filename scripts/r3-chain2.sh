#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."; source scripts/r3-env.sh
V="${ROCM_VENV:-$HOME/.local/share/lifeos-inference/avalon-q8-prep-e09e775/rocm-7.14.1-venv}"
O=results-r3/runs2; mkdir -p $O
wrap() { echo "$V/bin/rocprofv3 --kernel-trace --stats -d $PWD/$O/$1/prof -o k --output-format csv --"; }
python3 scripts/r3-run.py --arm A --spec results-r3/prompts/short.spec --out $O/A-short --tag A-rep2-short > $O/A-short.out 2>&1
python3 scripts/r3-run.py --arm B --min-kv 1024 --spec results-r3/prompts/short.spec --out $O/B-short --tag B-rep2-short > $O/B-short.out 2>&1
python3 scripts/r3-run.py --arm A --spec results-r3/prompts/long.spec --out $O/A-long --tag A-rep2-long > $O/A-long.out 2>&1
python3 scripts/r3-run.py --arm B --min-kv 32768 --spec results-r3/prompts/long.spec --out $O/B-long --tag B-rep2-long > $O/B-long.out 2>&1
echo done > $O/done

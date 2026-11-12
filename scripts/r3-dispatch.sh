#!/usr/bin/env bash
# R3: prove that arm B executes k_fattn_qsa_rdna2 at MIN_KV=1024 (and arm A does not) on one 2048-token record, under rocprofv3 kernel-trace.
set -u; cd "$(dirname "$0")/.."; source scripts/r3-env.sh
V="${ROCM_VENV:-$HOME/.local/share/lifeos-inference/avalon-q8-prep-e09e775/rocm-7.14.1-venv}"; O=results-r3/dispatch; mkdir -p $O
for arm in B A; do
  python3 scripts/r3-run.py --arm $arm --min-kv 1024 --spec results-r3/dispatch.spec --out $O/$arm --tag dispatch-$arm \
    --wrap "$V/bin/rocprofv3 --kernel-trace --stats -d $PWD/$O/$arm/prof -o k --output-format csv --" > $O/$arm.out 2>&1
  echo "$arm: $(grep -h -c k_fattn_qsa_rdna2 $(find $O/$arm/prof -name '*kernel_stats.csv' | head -1) 2>/dev/null)"
done

#!/usr/bin/env bash
# R2 chain 4: re-validate the P2 binary (QSA_R2_WAVES_PER_EU=8): B only, A numbers come from chain 1 (same A binary path).
cd "$(dirname "$0")/.."
L=results/logs/r2p2; mkdir -p $L
date '+%Y-%m-%d %H:%M:%S' > $L/kernel-since
P() { name=$1; shift; python3 scripts/logit-probe.py --out results/probes/$name.json "$@" > $L/$name.out 2>&1 < /dev/null || echo "FAILED probe $name" >> $L/fail; }
H() { arm=$1; tok=$2; shift 2; scripts/run-ab-bench.sh $arm $tok 8192 4096 "$@" > $L/ab-$arm-$tok-$(date +%H%M%S).log 2>&1 < /dev/null || echo "FAILED ab $arm $tok" >> $L/fail; }
for i in 1 2 3; do H B 40960; done
for i in 1 2 3; do H B 65536; done
for i in 1 2 3; do H B 126972; done
H B 204800
H B 250000
for i in 1 2 3; do P B-p2-free-$i --arm B; done
P B-p2-forcedA1 --arm B --force results/probes/A-free-1.json
echo done > $L/done

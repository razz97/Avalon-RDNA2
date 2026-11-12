#!/usr/bin/env bash
# R2 chain 3: noise floor at 127K (second free run per arm). Waits for chain 2.
cd "$(dirname "$0")/.."
L=results/logs/r2
while [ ! -f $L/done2 ]; do sleep 20; done
P() { name=$1; shift; python3 scripts/logit-probe.py --out results/probes/$name.json "$@" > $L/$name.out 2>&1 < /dev/null || echo "FAILED probe $name" >> $L/fail; }
P A-free-127k-2 --arm A --prompt-tokens 126972
P B-free-127k-2 --arm B --prompt-tokens 126972
echo done > $L/done3

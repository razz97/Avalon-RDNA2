#!/usr/bin/env bash
# R2 validation chain (serial, background). Every step is its own fresh server; failures are logged, not fatal.
cd "$(dirname "$0")/.."
L=results/logs/r2; mkdir -p $L results/probes
date -Is > $L/start; journalctl -k --no-pager -o short-iso --since now > /dev/null 2>&1; echo "$(date '+%Y-%m-%d %H:%M:%S')" > $L/kernel-since
P() { name=$1; shift; python3 scripts/logit-probe.py --out results/probes/$name.json "$@" > $L/$name.out 2>&1 < /dev/null || echo "FAILED probe $name" >> $L/fail; }
H() { arm=$1; tok=$2; shift 2; scripts/run-ab-bench.sh $arm $tok 8192 4096 "$@" > $L/ab-$arm-$tok-$(date +%H%M%S).log 2>&1 < /dev/null || echo "FAILED ab $arm $tok" >> $L/fail; }
# 1. no-MTP free runs, 40K: determinism A/A, B/B and A/B (A-free-1 already exists)
for i in 2 3 4 5; do P A-free-$i --arm A; done
for i in 1 2 3 4 5; do P B-free-$i --arm B; done
# 2. teacher forcing on A's tokens (same prefix) + FP32-arithmetic kernel + unpatched base binary
P B-forcedA1 --arm B --force results/probes/A-free-1.json
P B-fp32-forcedA1 --arm B --force results/probes/A-free-1.json --env '{"LLAMA_QSA_RDNA2_FP32":"1"}'
P B-fp32-free-1 --arm B --env '{"LLAMA_QSA_RDNA2_FP32":"1"}'
P U-free-1 --arm A --bin-dir "$HOME/.local/share/lifeos-inference/avalon-q8-prep-e09e775/build-hip-trace-o3-gfx1030/bin"
# 3. 127K probes
P A-free-127k --arm A --prompt-tokens 126972
P B-free-127k --arm B --prompt-tokens 126972
P B-forcedA-127k --arm B --prompt-tokens 126972 --force results/probes/A-free-127k.json
# 4. MTP on, top-5 logprobs of the generated tokens (harness), 40K x3 per arm
export AB_PAYLOAD='{"n_probs":5}'
for i in 1 2 3; do H A 40960; H B 40960; done
unset AB_PAYLOAD
# 5. paired repetitions 65K and 127K (to reach >=3 per arm)
for i in 1 2; do H A 65536; H B 65536; done
for i in 1 2; do H A 126972; H B 126972; done
# 6. long contexts
H B 204800; H A 204800; H B 250000; H A 250000
echo done > $L/done

#!/usr/bin/env bash
# R2 chain 2: removes the hosted-MoE prefill ring (GGML_CUDA_MOE_RING_PREFILL=0) to isolate the baseline run-to-run
# non-determinism from the attention kernel. Waits for chain 1.
cd "$(dirname "$0")/.."
L=results/logs/r2
while [ ! -f $L/done ]; do sleep 20; done
P() { name=$1; shift; python3 scripts/logit-probe.py --out results/probes/$name.json "$@" > $L/$name.out 2>&1 < /dev/null || echo "FAILED probe $name" >> $L/fail; }
R0='{"GGML_CUDA_MOE_RING_PREFILL":"0"}'
for i in 1 2 3; do P A-ring0-$i --arm A --env "$R0"; done
for i in 1 2 3; do P B-ring0-$i --arm B --env "$R0"; done
for i in 1 2; do P B-fp32-ring0-$i --arm B --env '{"GGML_CUDA_MOE_RING_PREFILL":"0","LLAMA_QSA_RDNA2_FP32":"1"}'; done
echo done > $L/done2

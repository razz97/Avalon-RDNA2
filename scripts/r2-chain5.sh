#!/usr/bin/env bash
# R2 chain 5: probes against the PATCHED binary (chains 1-3 probes accidentally used the unpatched binary; they are kept in
# results/probes-baseline-binary as baseline run-to-run noise evidence only).
cd "$(dirname "$0")/.."
L=results/logs/r2q; mkdir -p $L results/probes
BIN=$PWD/build-main/bin
date '+%Y-%m-%d %H:%M:%S' > $L/kernel-since
P() { name=$1; shift; python3 scripts/logit-probe.py --bin-dir $BIN --out results/probes/$name.json "$@" > $L/$name.out 2>&1 < /dev/null || echo "FAILED probe $name" >> $L/fail; }
for i in 1 2 3; do P q-A-free-$i --arm A; done
for i in 1 2 3 4 5; do P q-B-free-$i --arm B; done
P q-B-forcedA1 --arm B --force results/probes/q-A-free-1.json
P q-B-fp32-forcedA1 --arm B --force results/probes/q-A-free-1.json --env '{"LLAMA_QSA_RDNA2_FP32":"1"}'
P q-B-fp32-free-1 --arm B --env '{"LLAMA_QSA_RDNA2_FP32":"1"}'
for i in 1 2; do P q-A-free-127k-$i --arm A --prompt-tokens 126972; P q-B-free-127k-$i --arm B --prompt-tokens 126972; done
P q-B-forcedA-127k --arm B --prompt-tokens 126972 --force results/probes/q-A-free-127k-1.json
R=$HOME/.local/share/lifeos-inference/avalon-q8-prep-e09e775/rocm-7.14.1-venv
rm -rf results/profile-server
PROBE_WRAP="$R/bin/rocprofv3 --kernel-trace --stats -d $PWD/results/profile-server -o srv --output-format csv --" P q-PROF-B-65536 --arm B --prompt-tokens 65536 --steps 1
P q-SCHED-B-40960 --arm B --prompt-tokens 40960 --steps 1 --env '{"GGML_SCHED_DEBUG":"2","LLAMA_ARG_LOG_VERBOSITY":"4"}'
echo done > $L/done

# Avalon RDNA2 QSA sparse prefill, R2

Patch over e09e775 (patches/), sources (source/), tests, scripts, results, STATUS.md (truthful), CORRECTNESS.md, PERFORMANCE.md, RISKS.md, CHANGES_R2.md, SHA256SUMS.
Build: scripts/build-gfx1030.sh; tests: scripts/run-correctness.sh [--bench]; A/B: scripts/run-ab-bench.sh ARM TOKENS BATCH UBATCH; probes: scripts/logit-probe.py (use --bin-dir!); chains: scripts/r2-chain*.sh; analysis: scripts/analyze-r2q.py, scripts/analyze-r2.py.
Env: LLAMA_QSA_RDNA2_PREFILL (default 0), LLAMA_QSA_RDNA2_MIN_KV, LLAMA_QSA_RDNA2_FP32 (validation reference only).
No deploy, no production restart. Models and MoE ranking file untouched.

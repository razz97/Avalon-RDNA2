# R2 changes vs review R1
- P0 greedy: probes with logprob/margins (scripts/logit-probe.py), forced prefix, A/A, B/B, A/B repeats, MTP on/off, 127K, prompt SHA256 + prompt-plan.json per run, unpatched-binary noise floor, ring-off check. See CORRECTNESS.md.
- P0 LDS: dead slots no longer read LDS (fattn-qsa-rdna2.cu).
- P1 FP32 reference variant LLAMA_QSA_RDNA2_FP32=1 (Q FP32, K FP32 dequant) to test FP16 hypothesis.
- P1 backend probe cached per shape set; pointer alignment asserts; skipped tests fail; n_sel partial test; absolute test gate scaled by V magnitude.
- P1 200K/250K runs; 3 repetitions at 65K/127K; journalctl -k check; rocprofv3 kernel trace of the real server.
- P2 one iteration: waves_per_eu(8) (240 -> 128 VGPR), -30% kernel time, +5..7% end-to-end prefill.
- Not done: sanitizer (none available), SCHED_DEBUG dump, MoE hit/miss/PCIe, request-chain stress, forced allocation failure.

# STATUS R2: EXPERIMENTAL (NOT COMPLETE+VERIFIED)

Base: e09e775fe540cf8526b62b0f98b924c0ed7eb414. Flag `LLAMA_QSA_RDNA2_PREFILL=0` by default. Nothing deployed, no production service touched.

## What is demonstrated (RX 6800 XT, gfx1030, real model, ub=4096, MTP on)
- Throughput (prefill mean tok/s, A = flag off, B = flag on, B = optimised P2 kernel): 40K 800 -> 910; 65K 658 -> 917; 127K 455 -> 910; 204K 326 -> 887; 250K 279 -> 862. Decode unchanged (A 47-57, B 47-56 tok/s). Compute buffer 3177 -> 1184 MiB.
- rocprofv3 trace of the real server (65K prefill): `k_fattn_qsa_rdna2` ran 108 times on the GPU (no CPU route; mode 9 is refused by CPU). This replaces the SCHED_DEBUG dump, which I could not obtain (see RISKS).
- No amdgpu/ring/page-fault messages in `journalctl -k` during the R2 chains (ranges in results/logs/r2*/kernel-since).
- 19/19 synthetic cases pass (default and FP32 reference mode), 0 skipped; skipped cases now fail the matrix.

## What is NOT demonstrated (hard gates still open)
1. Greedy equivalence. The divergence is NOT attributable to a kernel bug, but A/B equivalence of the generated text is not proven either: see CORRECTNESS.md. At 127K the A/B top-5 logprob deltas (1.2-1.6 nats, 4 pairs) are larger than the A/A and B/B pairs I measured (0.0 and 0.72, n=2 each). Needs an end-to-end quality gate (perplexity / long-context retrieval), not run.
2. Bit identity flag-off vs unpatched binary: impossible to test because the baseline itself is not deterministic (A/A pairs differ 0.3-0.7 nats).
3. compute-sanitizer / HIP sanitizer: none available in the installed ROCm venv (not run). The uninitialised LDS read was fixed by inspection + tests only.
4. SCHED_DEBUG node dump: not captured (verbosity setting did not emit it).
5. VRAM margin is still ~345 MiB minimum free (unchanged); ubatch 8192 not authorised by this work.
6. MoE hit/miss and post-prefill PCIe not measured; only the log counters (evictions, in-place experts) are compared.
7. Alternation prefill/decode with MTP was exercised only through the A/B harness (prefill + 128 decode tokens per run, HIP graphs on). No stress chaining of requests, no forced allocation-failure test.

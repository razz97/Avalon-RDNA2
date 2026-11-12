# CORRECTNESS R2

## Important correction of my own method
The first R2 probe campaign (chains 1-3) ran the UNPATCHED binary for both arms because `logit-probe.py` was missing `--bin-dir`. Those probes are kept in `results/probes-baseline-binary/` and are valid ONLY as baseline run-to-run noise (45 pairs, unpatched binary, 40K, no MTP). The patched-binary probes are `results/probes/q-*` (chain 5, P2 binary `build-hip-gfx1030-qsa`, argv recorded in every JSON). The A/B harness runs (MTP on) always used the patched binary. Prompt SHA256 of all 40K runs: 70ad59c7c7c79d7f8b7bf87fbe81cdbac45c3a031ccf23b9cc7a1c44b116ba52 (127K prompt: equal between arms, hash in each probe JSON; per-run prompt-plan.json included).

## Findings (patched binary, 40K, no MTP, temperature 0, top-5 logprobs, steps 0..32)
- Position 23 is a near tie (tokens 279 vs 9626, margin 0.11-0.25 nats). Positions 24 and 26 also have margins 0.03-0.12.
- A/A (flag off, same binary): 3 runs, 2 distinct sequences (token 23 = 279, 9626, 279); pairwise max |dlogprob| 0.39-0.59.
- B/B: 5 runs, 3 distinct sequences (first diff at 24 or 26 in some pairs); pairwise 0.0-0.61.
- A/B: 15 pairs, 0.43-0.69; first token difference at 23/24/26 or none.
- Unpatched binary alone (10 runs, 45 pairs): 2 distinct sequences, flips at token 23, pairwise up to 0.76. So the non-determinism exists without any QSA code; it is not caused by the hosted-MoE ring (disabled with GGML_CUDA_MOE_RING_PREFILL=0 it persists: A-ring0 flips 279/9626). Its root cause was not identified.
- Teacher forcing with A's prefix: B has no argmax flips (max dlogprob 0.535); the FP32 reference variant (Q FP32, K dequantised in FP32) DOES flip at k=23 and k=26 (margins 0.11 and 0.025) and has the same delta magnitude (0.48). So the FP16 Q/half2 arithmetic is not the explanation (hypothesis P1 tested and rejected as the cause of the flips).
- MTP on (harness, patched binary, 40K, 128 tokens): token 23 = 9626 in 6/6 A runs and in 5/6 B runs (one B run chose 279); without MTP A chooses 279. Batch composition of the verification step moves the tie in both arms; per-run margins are in the events-0.jsonl of the run dirs (completion_probabilities).
- 127K, no MTP: A/A identical (delta 0.0, n=2), B/B delta 0.72 (n=2), A/B 1.22-1.58 (4 pairs); all sequences identical over 33 tokens, no argmax flips with forced prefix (max dlogprob 1.58). This is the one result that may indicate a real systematic difference; sample too small to say.

Conclusion: 40K divergences are explained by near-tie margins plus pre-existing non-determinism of the whole engine; the 127K logprob deltas are unexplained and keep the status EXPERIMENTAL.

## Kernel tests
`tests/test-qsa-rdna2.cpp`: 19/19 (CPU FP32 oracle; mode 9 vs dense), incl. many dead slots, n_sel partial (2300), all-masked, big K values; absolute gate scaled with V magnitude. Skipped counts as failure. NRMSE ~5e-4 is NOT claimed as equivalence.
LDS: dead slots no longer read sK/sV (writes 0.0 to the reduction slot directly).
Runtime alignment asserts for K/V/Q/mask/idx/dst pointers added.

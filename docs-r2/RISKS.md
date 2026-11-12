# RISKS / OPEN ITEMS
- 127K A/B logprob deltas exceed measured noise (small n). Needs a quality gate (PPL / retrieval) before any production use.
- Baseline engine is non-deterministic at temperature 0 (cause unknown). Any A/B text comparison is inherently noisy.
- Spec said no scratch: kernel now spills 68 B/thread to private scratch after the P2 change.
- VRAM headroom ~345 MiB; unchanged. Do not raise ubatch automatically.
- No sanitizer; LDS fix verified by inspection and tests only.
- SCHED_DEBUG node dump not obtained; evidence of GPU placement is the rocprofv3 kernel trace plus the CPU refusal of mode 9.
- Probe campaign mistake (wrong binary) detected and redone; the P2 binary is the only one probed. R1-kernel results exist only in the harness A/B data.
- A leftover defunct llama-server from a rocprofv3 run blocked one probe start (logit-probe aborts on competing processes); harmless.

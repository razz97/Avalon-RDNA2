# BASELINE (2026-11-12) — upstream 66182c6a1 + RDNA2 QSA prefill

Reference for the next diagnostic/optimisations. Base: `main` = merge of `upstream/main` 66182c6a1 into the integrated fork (HEAD at measurement: 1ef68d78e).
Status of the QSA prefill kernel is still EXPERIMENTAL (docs-r3/STATUS.md); this baseline is a speed reference, not a quality validation.

## Chosen baseline configuration: QSA ON, ub 4096
```
llama-server -m Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf -c 262144 -b 8192 -ub 4096 -fa on -ctk q8_0 -ctv q4_0 \
  --parallel 1 -t 6 --jinja --host 127.0.0.1 -md MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0
env: LLAMA_QSA_RDNA2_PREFILL=1 (MIN_KV default 20480) GGML_SCHED_BORROW_COMPUTE=1 GPU_MAX_HW_QUEUES=2 HSA_ENABLE_SDMA=1
```
Harness: `scripts/oneshot-ab.py --prompt-tokens 100000 --batch 8192 --ubatch 4096 --predict 1024` (fresh prompt, no cache, ignore_eos, temperature 0, one sample per arm).
Hardware: RX 6800 XT gfx1030 16 GiB, ROCm 7.14.1 (venv SDK), i7-8700K, 64 GiB RAM, nothing else on the GPU.

## Results (100K fresh prompt, 1024 generated tokens, MTP on)
| config | prefill tok/s | prefill s | decode tok/s | MTP acceptance | min free VRAM |
|---|---|---|---|---|---|
| A QSA off, ub4096 | 521.5 | 191.8 | 61.2 | 72.2 % (604/836) | 344 MiB |
| **B QSA on, ub4096 (baseline)** | **928.7** | 107.7 | **63.8** | 74.3 % (611/822) | 337 MiB |
| B QSA on, ub8192 (not chosen) | 991.4 | 100.9 | 62.2 | 74.2 % (611/823) | 336 MiB |
| A QSA off, ub8192 | fails: cudaMalloc OOM (687 MiB compute buffer) | | | | |

MoE after prefill (evicted entries before rewarm / in-place expert reads): A ub4096 405 / 13.7 %; B ub4096 190 / 15.1 %; B ub8192 625 / 14.2 %.
Decode differences between arms are within single-sample noise. Raw logs: `~/work/avalon-diag/baseline/` (not in the repo).

# Avalon-RDNA2 (private) — maintenance

Status of the RDNA2 QSA sparse prefill: **EXPERIMENTAL** (docs-r3/STATUS.md). `LLAMA_QSA_RDNA2_PREFILL=0` is the default; it must stay that way.

## Layout
- `origin`   = https://github.com/razz97/Avalon-RDNA2 (PRIVATE).
- `upstream` = https://github.com/freriadoc/Avalon (public, never push there).
- `main` = pristine mirror of `upstream/main` (never commit experimental work here; fast-forward only).
- `feat/rdna2-qsa-prefill` = our work (base e09e775fe540cf8526b62b0f98b924c0ed7eb414, ~10 commits + docs). Not merged into main.
- Large artefacts (results/, results-r3/, models, builds) are NOT in the repo; the R2/R3 ZIPs hold the evidence.

## Update from upstream (keeps our branch isolated)
```
git fetch upstream --tags
git switch main && git merge --ff-only upstream/main && git push origin main
# try the new base on a throwaway branch first:
git switch -c try/rebase-$(date +%F) feat/rdna2-qsa-prefill
git rebase upstream/main          # conflicts expected only in src/models/qwen4exp.cpp, ggml/src/ggml.c, ggml-cuda/fattn.cu, tests/CMakeLists.txt
# build (scripts/build-gfx1030.sh) + scripts/run-correctness.sh (19/19) before adopting
git branch -m feat/rdna2-qsa-prefill feat/rdna2-qsa-prefill-old-e09e775 && git branch -m try/rebase-$(date +%F) feat/rdna2-qsa-prefill
git push origin feat/rdna2-qsa-prefill-old-e09e775 && git push --force-with-lease origin feat/rdna2-qsa-prefill
```
Alternative with fewer rewrites: `git merge upstream/main` into a throwaway branch. Our changes are additive (new files fattn-qsa-rdna2.cu/.cuh, tests, tools/qsa-nll, scripts/r2-*, r3-*) plus small hooks in the files listed above, so the diff against any upstream is `git diff upstream/main...feat/rdna2-qsa-prefill -- ggml src tests tools`.
After any rebase re-run quality checks (docs-r3/QUALITY_FAST.md method) before changing status.

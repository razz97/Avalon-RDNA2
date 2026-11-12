# CORRECTNESS (R3)
- **S0/S5**: `scripts/run-correctness.sh`: **19/19 PASS, 0 omitidos**, incluyendo KV sintético hasta 262144 (principio/medio/final, último KV válido, celdas inválidas). Log: `results-r3/s0-correctness.log`, `s0-result.txt`. Pruebas completas de R2 en `docs-r2/CORRECTNESS.md` (reutilizadas, no repetidas).
- **Despacho**: `results-r3/dispatch*` (rocprofv3 kernel-trace sobre 2048 tokens, MIN_KV=1024): B ejecuta `k_fattn_qsa_rdna2` (12 llamadas), A 0 (confirma que `PREFILL=0` usa la ruta antigua). Ver nota de límites: no se trazó cada ejecución de calidad (rocprof ralentiza ~20×).
- **NLL**: sin NaN/Inf; ver QUALITY_FAST.md.
- **Smoke MTP**: ver MTP_DIAGNOSIS.md; sin incidencias en `journalctl -k` (amdgpu/reset/hang/OOM/page fault: ninguna) en `smoke-*/journalctl-k.txt`.
- Sanitizers: no ejecutados (pendiente desde R2).

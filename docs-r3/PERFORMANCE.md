# PERFORMANCE (R3)
R3 no repite curvas de rendimiento. Los números de R2 (A→B prefill 40K 800→910, 65K 658→919, 127K 455→911, 204K 326→888, 250K 279→863 tok/s; compute 3177→1184 MiB; margen VRAM ≈345 MiB) están en `docs-r2/PERFORMANCE.md` y `results/runs-p2/` (ZIP R2 SHA256 ffc5794897cb1d046695c99d326b9d9dad1ba3e32e54670fe5149e1a18c62c57).
Datos incidentales de R3 (con evaluador de logits, no comparables a throughput): registro 127K en `runs/*-long/engine.log` (“record S4 … done in … s”); mínimo VRAM libre observado: A 342–343 MiB, B 332–337 MiB (`vram_min_free_mib` en meta.json).
**Lección operativa**: ejecutados en un scope de Hermes (MemoryMax 4 GiB) los prefills son ~40× más lentos por thrash de la caché de páginas del modelo mmap; usar `scripts/r3-launch.sh` (systemd-run --user).

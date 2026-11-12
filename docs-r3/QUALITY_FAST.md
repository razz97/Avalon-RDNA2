# QUALITY_FAST (R3)

Método: `llama-qsa-nll` (tools/qsa-nll), teacher-forced con IDs idénticos, NLL = logsumexp(logits) − logits[target] sobre el **vocabulario completo** en doble precisión, logits solo en posiciones elegidas (32 repartidas ~25/50/75 % y final + 32 de continuación fija = 64 por prompt; 63–64 en JSON). Mismo binario P2 parcheado en A y B, MTP OFF, ctx 262144, -b 8192 -ub 4096, KV Q8_0/Q4_0, `GGML_SCHED_BORROW_COMPUTE=1`, `LLAMA_QSA_TILES=0`. A: `LLAMA_QSA_RDNA2_PREFILL=0`; B: `=1` con `MIN_KV=1024` (8K/16K, solo diagnóstico) o 32768 (40K/127K). Off-by-one verificado con `sanity` (results-r3/sanity*). Los JSON incluyen binario, hashes, argv, flags, token-ID hash y seed.

| Prompt | Tipo / contexto | Targets | NLL_A | NLL_B | ΔNLL medio (B−A) ± se | máx \|Δ\| | targets \|Δ\|>0,5 | Gate G1 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| S1a-code-8K | código (R2 long_task truncado) 8K | 64 | 14.652 | 14.724 | +0.072 ± 0.275 | 10.3 | 35 | **ALARMA (>0.02)** |
| S1b-retrieval-8K | recuperación 8K | 64 | 1.049 | 1.056 | +0.007 ± 0.052 | 2.8 | 2 | PASS |
| S2-structured-16K | estructurado 16K | 64 | 0.681 | 0.631 | -0.050 ± 0.037 | 2.3 | 2 | PASS |
| S3a-code-40K_R2_prompt | código (prompt R2) 40K | 64 | 12.130 | 11.632 | -0.498 ± 0.283 | 12.9 | 21 | PASS |
| S3b-retrieval-40K | recuperación 40K | 64 | 1.268 | 1.282 | +0.014 ± 0.044 | 2.4 | 3 | PASS |
| S4-127K_R2_prompt | prompt R2 127K (126972) | 64 | 6.826 | 6.462 | -0.365 ± 0.204 | 6.8 | 34 | PASS |

Todos los pares se ejecutaron **dos veces en procesos nuevos** (runs/ y runs2/): A1≡A2 y B1≡B2 (ruido entre procesos ≈ 0, máx 0,1 en un caso a 127K). Las diferencias A↔B son por tanto **reales y deterministas**, no ruido de ejecución.

## Lectura honesta
- No se observa **sesgo sistemático**: el signo de ΔNLL varía entre prompts (−0,50 … +0,07) y los errores estándar son de 0,04–0,28. Pero con 64 targets/prompt **no se demuestra equivalencia de calidad**.
- Hay targets con |Δ|>0,5 en todos los prompts (incl. recuperación, rms 0,3–0,4 nats con NLL≈1). Los prompts de código tienen NLL 12–14 nats (tokens casi aleatorios para el modelo), muy sensibles.
- **Yardstick** (`results-r3/yard`, A con ubatch 2048 vs A con 4096, mismo conjunto): rms 1,96 / 0,36 / 0,23 y 39 / 3 / 2 targets >0,5 en S1a/S1b/S2, frente a A→B 2,18 / 0,41 / 0,30 y 35 / 2 / 2. Es decir B perturba la numérica en la misma magnitud que un cambio de ubatch. Es una **referencia útil, no una justificación suficiente** para descartar los outliers: no sustituye una prueba de no inferioridad.
- **S1a (+0,072) supera el umbral de alarma +0,02** y se registra como **riesgo residual**; su se es 0,275 (≈0,26 σ) y el yardstick da −0,398 en la dirección opuesta, pero el gate G1 literal no se cumple. S1a no mide recuperación de contexto (texto casi aleatorio para el modelo).
- S3a/S4 mejoran en media (−0,50/−0,37) pero con outliers de hasta 12,9/6,8 nats: tampoco se interpreta como mejora.
- Discrepancia histórica de R2 a 127K: R3 usa el prompt/hash de R2 con NLL por posición; el resultado A/B es determinista y reproducible (A1=A2, B1=B2), de modo que la diferencia de R2 era una diferencia numérica real y no ruido, pero su relevancia para calidad sigue sin acotarse más allá de lo anterior.
- Análisis completo: `results-r3/r3-noise.json` (subconjunto “predecible” NLL<8 incluido), `results-r3/r3-compare.json`.

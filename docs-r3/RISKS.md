# RISKS
1. **G1 literal no cumplido en S1a (+0,072 > +0,02)**; outliers |Δ|>0,5 en todos los prompts. Diferencias deterministas y de magnitud similar a cambiar ubatch, pero no probada la no inferioridad (64 targets/prompt, 6 prompts, sin IC).
2. 250K real **no revalidado en R3** (R2 sí); sin calidad medida >127K.
3. Smoke MTP con prompts de relleno no reproducible (también en A); causa no aislada; estado heredado entre peticiones no descartado de forma concluyente fuera del prompt natural.
4. Sin sanitizers (compute-sanitizer/ASan no disponibles); solo 19/19 tests + trazas.
5. Margen de VRAM mínimo ≈332–337 MiB (umbral G3 300 MiB): estrecho; no subir ubatch ni reservas MoE.
6. Despacho verificado por separado (rocprof 2048 tokens), no en cada run de calidad.
7. Con el flag de B activo, incluso prompts cortos (<64 tokens) cambian levemente la numérica (reservas/plan de cómputo): el flag no es neutro fuera de la ruta nueva.
Rollback: `LLAMA_QSA_RDNA2_PREFILL=0` (defecto) o revertir el parche (`git apply -R`, comprobado).

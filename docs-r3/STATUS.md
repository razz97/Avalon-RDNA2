# STATUS

**EXPERIMENTAL**

Motivo (sin rebajar criterios): el gate G1 literal falla en S1a (+0,072 nat/token > +0,02) y los outliers |Δ|>0,5 se reproducen de forma determinista sin una explicación que descarte degradación (el yardstick de ubatch es solo referencia); la discrepancia MTP quedó **no atribuible a B** (ocurre con A) pero su causa no está aislada.
Cumplido: G0 (19/19, despacho B probado, A usa ruta antigua), G3 (flag OFF por defecto, rollback comprobado, margen ≥300 MiB), no hay NaN, crash, ni errores de kernel; no hay regresión reproducible atribuible a B.
Si el revisor acepta S1a y los outliers como riesgo residual, el paquete puede reevaluarse como `PASS_LIMITED / CANARY_CANDIDATE`. No desplegado, no reiniciado nada, producción intacta. `LLAMA_QSA_RDNA2_PREFILL=0` por defecto.

#pragma once

#include "common.cuh"

// Native sparse prefill attention of QSA for RDNA2 (gfx1030): GGML_OP_FLASH_ATTN_EXT whose mask is a
// GGML_QSA_SELECT_RDNA2_PREFILL compact mask F16 [n_sel, n_t] and whose src[5] is the I32 selection [n_sel, n_t].
// K (Q8_0) and V (Q4_0) are read in place at the selected cells: no mask [n_kv, n_t], no gather, no F16 copy.

// true if dst (with that mask) is one this kernel computes on the device
bool ggml_cuda_flash_attn_ext_qsa_rdna2_supported(int device, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_qsa_rdna2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

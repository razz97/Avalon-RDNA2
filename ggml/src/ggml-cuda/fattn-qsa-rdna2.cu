#include "fattn-qsa-rdna2.cuh"
#include <atomic>

// One CTA = one query x one group of 4 query heads of a KV head, 256 threads (8 waves of 32), 32 selected slots per step,
// online softmax in FP32 in registers/LDS (no split-K, no global scratch).
//
// Wave w owns the quant block w (dims 32w..32w+31) of every K and V row:
//  - scores: lane = slot of the step; the wave dequantizes block w of that slot's K row out of LDS and dots it with the
//    4 heads' Q slice held in registers (v_dot2_f32_f16, FP32 accumulation); the 8 partial sums are added through LDS
//  - softmax: waves 0..3 = one head each, lane = slot (wave reductions)
//  - P*V: thread d accumulates dim d of the 4 heads (FP32), dequantizing V (Q4_0) out of LDS
// Only the live slots of a step (mask != -inf and 0 <= cell < n_kv) are ever loaded: a dead slot's cell is never read.

#define QSA_R2_BN   32
#define QSA_R2_NT   256
#ifndef QSA_R2_WAVES_PER_EU
#define QSA_R2_WAVES_PER_EU 8   // P2: cap VGPRs at 128 so two 8-wave CTAs share a CU (was 240 VGPR = 1 CTA/CU)
#endif
#define QSA_R2_HG   4                       // query heads per CTA
#define QSA_R2_KW   68                      // words of one K row of one head: 8 Q8_0 blocks of 34 bytes
#define QSA_R2_KS   69                      // its LDS stride in words (odd: the 32 lanes read 32 banks)
#define QSA_R2_VW   36                      // words of one V row of one head: 8 Q4_0 blocks of 18 bytes

static_assert(sizeof(block_q8_0) == 34 && sizeof(block_q4_0) == 18, "unexpected quant block size");
static_assert(WARP_SIZE == 32, "the kernel is written for wave32");

template <typename TQ>
static __device__ __forceinline__ void qsa_r2_load_q(half2 * q, const char * src) {
    // 32 consecutive elements (16 half2)
    if constexpr (std::is_same<TQ, float>::value) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 v = *(const float4 *) (src + 16*i);
            q[2*i + 0] = __floats2half2_rn(v.x, v.y);
            q[2*i + 1] = __floats2half2_rn(v.z, v.w);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint4 v = *(const uint4 *) (src + 16*i);
            q[4*i + 0] = *(const half2 *) &v.x;
            q[4*i + 1] = *(const half2 *) &v.y;
            q[4*i + 2] = *(const half2 *) &v.z;
            q[4*i + 3] = *(const half2 *) &v.w;
        }
    }
}

template <typename TQ, bool HP>
__launch_bounds__(QSA_R2_NT) __attribute__((amdgpu_waves_per_eu(QSA_R2_WAVES_PER_EU)))
static __global__ void k_fattn_qsa_rdna2(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const half * __restrict__ mask, const int32_t * __restrict__ idx, char * __restrict__ dst,
        const int n_sel, const int64_t n_kv, const float scale, const int gqa,
        const int64_t q_nb1, const int64_t q_nb2, const int64_t k_nb1, const int64_t k_nb2,
        const int64_t v_nb1, const int64_t v_nb2, const int64_t m_s1, const int64_t i_s1,
        const int64_t d_nb1, const int64_t d_nb2) {
#if defined(RDNA2)
    __shared__ uint32_t sK[QSA_R2_BN*QSA_R2_KS];
    __shared__ uint32_t sV[QSA_R2_BN*QSA_R2_VW];
    __shared__ float    sRed[8][QSA_R2_HG][QSA_R2_BN];
    __shared__ float    sP[QSA_R2_BN][QSA_R2_HG];
    __shared__ int      sC[2][QSA_R2_BN];
    __shared__ float    sBias[2][QSA_R2_BN];
    __shared__ uint32_t sLive[2];
    __shared__ float    sAlpha[QSA_R2_HG];
    __shared__ float    sL[QSA_R2_HG];

    const int t    = threadIdx.x;
    const int w    = t / WARP_SIZE;
    const int lane = t % WARP_SIZE;
    const int64_t q = blockIdx.x;                 // the query
    const int h0   = blockIdx.y * QSA_R2_HG;      // first of its 4 query heads
    const int kvh  = h0 / gqa;

    // HP (validation reference only, LLAMA_QSA_RDNA2_FP32=1): Q kept in FP32 and K dequantized + accumulated in FP32
    half2 qreg[HP ? 1 : QSA_R2_HG][HP ? 1 : 16];
    float qf[HP ? QSA_R2_HG : 1][HP ? 32 : 1];
#pragma unroll
    for (int hh = 0; hh < QSA_R2_HG; ++hh) {
        const char * qsrc = Q + q*q_nb1 + (int64_t) (h0 + hh)*q_nb2 + (int64_t) (32*w)*sizeof(TQ);
        if constexpr (HP) {
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                if constexpr (std::is_same<TQ, float>::value) { qf[hh][i] = ((const float *) qsrc)[i]; }
                else                                          { qf[hh][i] = __half2float(((const half *) qsrc)[i]); }
            }
        } else {
            qsa_r2_load_q<TQ>(qreg[hh], qsrc);
        }
    }

    const char * k_head = K + (int64_t) kvh*k_nb2;
    const char * v_head = V + (int64_t) kvh*v_nb2;
    const half    * m_row = mask + q*m_s1;
    const int32_t * i_row = idx  + q*i_s1;

    float acc[QSA_R2_HG] = {0.0f, 0.0f, 0.0f, 0.0f};
    float run_m = -INFINITY;      // waves 0..3 (head = wave): the running max and sum, identical in all its lanes
    float run_l = 0.0f;

    int par = 0;
    for (int j0 = 0; j0 < n_sel; j0 += QSA_R2_BN, par ^= 1) {
        // the slots of the step: the cell, the additive mask, whether the slot names a visible cell
        if (t < QSA_R2_BN) {
            const int j = j0 + t;
            int c = 0;
            float b = 0.0f;
            bool live = false;
            if (j < n_sel) {
                b = __half2float(m_row[j]);
                c = i_row[j];
                live = b > -INFINITY && (unsigned) c < (unsigned) n_kv;   // NaN: dead
            }
            sC[par][t]    = c;
            sBias[par][t] = b;
            const uint32_t bal = (uint32_t) __ballot(live);
            if (t == 0) {
                sLive[par] = bal;
            }
        }
        __syncthreads();
        const uint32_t live = sLive[par];
        if (live == 0) {
            continue;
        }

        // K and V rows of the live slots, raw, into LDS (words: the rows are 4-byte aligned, checked by the launcher)
#pragma unroll
        for (int r = 0; r < (QSA_R2_BN*QSA_R2_KW + QSA_R2_NT - 1)/QSA_R2_NT; ++r) {
            const int e = t + r*QSA_R2_NT;
            if (e < QSA_R2_BN*QSA_R2_KW) {
                const int s = e / QSA_R2_KW;
                const int x = e - s*QSA_R2_KW;
                if ((live >> s) & 1) {
                    sK[s*QSA_R2_KS + x] = *(const uint32_t *) (k_head + (int64_t) sC[par][s]*k_nb1 + 4*x);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < (QSA_R2_BN*QSA_R2_VW + QSA_R2_NT - 1)/QSA_R2_NT; ++r) {
            const int e = t + r*QSA_R2_NT;
            if (e < QSA_R2_BN*QSA_R2_VW) {
                const int s = e / QSA_R2_VW;
                const int x = e - s*QSA_R2_VW;
                if ((live >> s) & 1) {
                    sV[s*QSA_R2_VW + x] = *(const uint32_t *) (v_head + (int64_t) sC[par][s]*v_nb1 + 4*x);
                }
            }
        }
        __syncthreads();

        // partial scores of the quant block w: lane = slot. Dead slots never touch sK (never written this step):
        // their partial is exactly 0 and the softmax masks them anyway.
        {
            float s[QSA_R2_HG] = {0.0f, 0.0f, 0.0f, 0.0f};
            if ((live >> lane) & 1) {
                const uint16_t * kb = (const uint16_t *) ((const char *) sK + (lane*QSA_R2_KS)*4 + w*sizeof(block_q8_0));
                if constexpr (HP) {
                    const float d = __half2float(__ushort_as_half(kb[0]));
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        const uint32_t u = kb[1 + k];
                        const float k0 = d*(float) (int8_t) (u & 0xff);
                        const float k1 = d*(float) (int8_t) (u >> 8);
#pragma unroll
                        for (int hh = 0; hh < QSA_R2_HG; ++hh) {
                            s[hh] = fmaf(k0, qf[hh][2*k], fmaf(k1, qf[hh][2*k + 1], s[hh]));
                        }
                    }
                } else {
                    const half2 dd = __half2half2(__ushort_as_half(kb[0]));
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        const uint32_t u = kb[1 + k];
                        const half2 kv = __hmul2(dd, __halves2half2(__int2half_rn((int8_t) (u & 0xff)), __int2half_rn((int8_t) (u >> 8))));
#pragma unroll
                        for (int hh = 0; hh < QSA_R2_HG; ++hh) {
                            ggml_cuda_mad(s[hh], kv, qreg[hh][k]);
                        }
                    }
                }
            }
#pragma unroll
            for (int hh = 0; hh < QSA_R2_HG; ++hh) {
                sRed[w][hh][lane] = s[hh];
            }
        }
        __syncthreads();

        // online softmax of the step: wave hh = head hh, lane = slot
        if (t < QSA_R2_HG*WARP_SIZE) {
            const int hh = w;
            float x = 0.0f;
#pragma unroll
            for (int ww = 0; ww < 8; ++ww) {
                x += sRed[ww][hh][lane];
            }
            const bool lv = (live >> lane) & 1;
            const float logit = lv ? x*scale + sBias[par][lane] : -INFINITY;
            const float m_new = fmaxf(run_m, warp_reduce_max(logit));
            const float alpha = run_m == -INFINITY ? 0.0f : expf(run_m - m_new);
            const float p     = lv ? expf(logit - m_new) : 0.0f;
            run_l = run_l*alpha + warp_reduce_sum(p);
            run_m = m_new;
            sP[lane][hh] = p;
            if (lane == 0) {
                sAlpha[hh] = alpha;
            }
        }
        __syncthreads();

        // P*V: thread t = dim t
        {
#pragma unroll
            for (int hh = 0; hh < QSA_R2_HG; ++hh) {
                acc[hh] *= sAlpha[hh];
            }
            const int e = t & 31;
            uint32_t lm = live;
            while (lm) {
                const int s = __ffs(lm) - 1;
                lm &= lm - 1;
                const char * vb = (const char *) sV + s*(QSA_R2_VW*4) + w*sizeof(block_q4_0);
                const float d  = __half2float(__ushort_as_half(*(const uint16_t *) vb));
                const int   qb = (uint8_t) vb[2 + (e & 15)];
                const float v  = d*(float) ((e < 16 ? (qb & 0xF) : (qb >> 4)) - 8);
                const float4 p = *(const float4 *) &sP[s][0];
                acc[0] += p.x*v;
                acc[1] += p.y*v;
                acc[2] += p.z*v;
                acc[3] += p.w*v;
            }
        }
        // the next step writes sK/sV after its first barrier, sRed after its second, sP after its third: every wave has left
        // this step's reads by then
    }

    if (t < QSA_R2_HG*WARP_SIZE && lane == 0) {
        sL[w] = run_l;
    }
    __syncthreads();
#pragma unroll
    for (int hh = 0; hh < QSA_R2_HG; ++hh) {
        const float l = sL[hh];
        *(float *) (dst + q*d_nb2 + (int64_t) (h0 + hh)*d_nb1 + (int64_t) t*sizeof(float)) = l > 0.0f ? acc[hh]/l : 0.0f;
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, idx, dst, n_sel, n_kv, scale, gqa, q_nb1, q_nb2, k_nb1, k_nb2, v_nb1, v_nb2, m_s1, i_s1, d_nb1, d_nb2);
    NO_DEVICE_CODE;
#endif // defined(RDNA2)
}

bool ggml_cuda_flash_attn_ext_qsa_rdna2_supported(int device, const ggml_tensor * dst) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED_VARS(device, dst);
    return false;
#else
    if (ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_RDNA2) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sink = dst->src[4];
    const ggml_tensor * idx  = dst->src[5];
    if (!Q || !K || !V || !mask || !idx || sink) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (mask->op != GGML_OP_QSA_SELECT || ggml_get_op_params_i32(mask, 1) != GGML_QSA_SELECT_RDNA2_PREFILL || idx != mask->src[0]) {
        return false;
    }
    if ((Q->type != GGML_TYPE_F32 && Q->type != GGML_TYPE_F16) || K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q4_0 ||
        mask->type != GGML_TYPE_F16 || idx->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 ||
        K->ne[2] != V->ne[2] || K->ne[1] != V->ne[1] || K->ne[2] <= 0 || Q->ne[2] % K->ne[2] != 0 || (Q->ne[2] / K->ne[2]) % QSA_R2_HG != 0) {
        return false;
    }
    if (!ggml_is_contiguous(mask) || !ggml_is_contiguous(idx) || mask->ne[0] != idx->ne[0] || mask->ne[1] != idx->ne[1] ||
        mask->ne[1] != Q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1 || idx->ne[2] != 1 || idx->ne[3] != 1) {
        return false;
    }
    // the rows are read as 4-byte words, Q as 16-byte vectors
    const size_t qs = ggml_type_size(Q->type);
    if (K->nb[1] % 4 || K->nb[2] % 4 || V->nb[1] % 4 || V->nb[2] % 4 || K->nb[0] != sizeof(block_q8_0) || V->nb[0] != sizeof(block_q4_0) ||
        Q->nb[0] != qs || Q->nb[1] % 16 || Q->nb[2] % 16 || dst->nb[0] != sizeof(float) || dst->nb[1] % 4 || dst->nb[2] % 4) {
        return false;
    }
    return true;
#endif
}

template <typename TQ, bool HP>
static void qsa_r2_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * idx  = dst->src[5];
    // the shape/stride conditions are checked by _supported(); the pointers only exist now: the kernel reads K/V rows as
    // 4-byte words, Q as 16-byte vectors, mask as halves, indices as ints
    GGML_ASSERT(((uintptr_t) K->data & 3) == 0 && ((uintptr_t) V->data & 3) == 0 && ((uintptr_t) Q->data & 15) == 0 &&
                ((uintptr_t) mask->data & 1) == 0 && ((uintptr_t) idx->data & 3) == 0 && ((uintptr_t) dst->data & 3) == 0);
    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));
    const int gqa = (int) (Q->ne[2] / K->ne[2]);
    const dim3 grid((unsigned) Q->ne[1], (unsigned) (Q->ne[2] / QSA_R2_HG), 1);
    k_fattn_qsa_rdna2<TQ, HP><<<grid, QSA_R2_NT, 0, ctx.stream()>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data,
        (const half *) mask->data, (const int32_t *) idx->data, (char *) dst->data,
        (int) idx->ne[0], K->ne[1], scale, gqa,
        Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2],
        mask->nb[1]/(int64_t) sizeof(half), idx->nb[1]/(int64_t) sizeof(int32_t), dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_attn_ext_qsa_rdna2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_qsa_rdna2_supported(ctx.device, dst));
    // validation switch: FP32 reference arithmetic (slower), read once
    static const bool hp = getenv("LLAMA_QSA_RDNA2_FP32") != nullptr && atoi(getenv("LLAMA_QSA_RDNA2_FP32")) != 0;
    // opt-in dispatch evidence (AVALON_RDNA2_DISPATCH_LOG=1): host-side launcher entries, not GPU completion
    static const bool log_dispatch = getenv("AVALON_RDNA2_DISPATCH_LOG") && atoi(getenv("AVALON_RDNA2_DISPATCH_LOG")) != 0;
    if (log_dispatch) {
        static std::atomic<long> calls{0};
        const long c = ++calls;
        if (c <= 2 || c % 32 == 0) {
            GGML_LOG_INFO("QSA_DISPATCH rdna2 sparse flash attention n_q=%lld n_kv=%lld n_sel=%lld calls=%ld\n",
                (long long) dst->src[0]->ne[1], (long long) dst->src[1]->ne[1], (long long) dst->src[3]->ne[0], c);
        }
    }
    if (dst->src[0]->type == GGML_TYPE_F32) {
        if (hp) { qsa_r2_launch<float, true>(ctx, dst); } else { qsa_r2_launch<float, false>(ctx, dst); }
    } else {
        if (hp) { qsa_r2_launch<half, true>(ctx, dst); } else { qsa_r2_launch<half, false>(ctx, dst); }
    }
}

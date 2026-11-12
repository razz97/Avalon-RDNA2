// Correctness (and, with --bench, microbenchmark) of the RDNA2 sparse prefill attention of QSA:
// GGML_OP_FLASH_ATTN_EXT with a GGML_QSA_SELECT_RDNA2_PREFILL compact mask, K Q8_0 / V Q4_0 read in place (gfx1030 HIP).
//
// Oracles:
//   1. an independent CPU FP32 reference written here (dequantize K/V with the ggml type traits, mask/scale/softmax literally)
//   2. the dense GPU path: GGML_QSA_SELECT_DENSE mask [n_kv, n_t] + flash attention over F16 copies of K/V
//   3. the mask itself: mode 9 against mode 1 (MASK) on the GPU, bitwise, and against the CPU reference
// The graph is run several times (the CUDA/HIP graph is captured then replayed); every run must match.
//
//   test-qsa-rdna2 [--quick] [--bench] [--cases name,name] [--seed N]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

static const int D = 256, N_HEAD = 24, N_KVH = 2;

struct tcase {
    std::string name;
    int64_t n_kv, n_t, n_sel;
    int     kpool, n_top;       // n_sel = n_top*kpool + tail
    bool    kv_perm   = true;   // K/V as the cache views them ([D, n_kvh, n_kv] permuted), else [D, n_kv, n_kvh]
    bool    q_perm    = true;   // Q as the model builds it ([D, n_head, n_t] permuted)
    ggml_type q_type  = GGML_TYPE_F32;
    bool    invalid_idx = false; // slots naming c < 0, c == n_kv, c > n_kv
    bool    dead_pools  = false; // pools with a -inf score
    bool    holes       = false; // cells with position -1 (other sequence)
    bool    empty       = false; // every slot dead
    bool    all_masked  = false; // every query before every cell
    bool    edge_cells  = false; // cells 0 and n_kv-1 present
    float   kscale = 1.0f, vscale = 1.0f; // magnitude of K, V
    bool    dense_ref   = true;
};

static void geometry(int64_t n_sel, int & kpool, int & n_top) {
    if (n_sel >= 2304)      { kpool = 128; n_top = (int) ((n_sel - 256) / 128); }
    else if (n_sel >= 512)  { kpool = 64;  n_top = (int) ((n_sel - 128) / 64); }
    else                    { kpool = 64;  n_top = (int) ((n_sel - 64) / 64); }
}

static tcase mk(const char * name, int64_t n_kv, int64_t n_t, int64_t n_sel) {
    tcase c; c.name = name; c.n_kv = n_kv; c.n_t = n_t; c.n_sel = n_sel; geometry(n_sel, c.kpool, c.n_top); return c;
}

struct stats { double nrmse = 0, max_abs = 0, p99 = 0; int64_t n_nan = 0; };

static stats compare(const std::vector<float> & a, const std::vector<float> & ref) {
    stats s; double e2 = 0, r2 = 0; std::vector<float> errs; errs.reserve(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(ref[i])) { s.n_nan++; continue; }
        const double d = (double) a[i] - ref[i];
        e2 += d*d; r2 += (double) ref[i]*ref[i];
        s.max_abs = std::max(s.max_abs, std::fabs(d));
        errs.push_back((float) std::fabs(d));
    }
    s.nrmse = r2 > 0 ? std::sqrt(e2/r2) : std::sqrt(e2);
    if (!errs.empty()) { std::sort(errs.begin(), errs.end()); s.p99 = errs[(size_t) ((errs.size() - 1)*0.99)]; }
    return s;
}

static ggml_backend_t g_gpu = nullptr;

static std::vector<float> to_f32(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

struct inputs {
    std::vector<float>   q;          // [t][h][d]
    std::vector<uint8_t> k, v;       // quantized rows, row r = cell*N_KVH + kvh (perm) or kvh*n_kv + cell
    std::vector<int32_t> sel;        // [t][n_sel]
    std::vector<float>   top_score;  // [t][n_top]
    std::vector<int32_t> cell_pos, tok_pos;
};

static inputs make_inputs(const tcase & c, std::mt19937_64 & rng) {
    inputs in;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    in.q.resize((size_t) c.n_t*N_HEAD*D);
    for (auto & x : in.q) { x = nd(rng); }

    const ggml_type_traits * tk = ggml_get_type_traits(GGML_TYPE_Q8_0);
    const ggml_type_traits * tv = ggml_get_type_traits(GGML_TYPE_Q4_0);
    const size_t rk = ggml_row_size(GGML_TYPE_Q8_0, D), rv = ggml_row_size(GGML_TYPE_Q4_0, D);
    in.k.resize((size_t) c.n_kv*N_KVH*rk);
    in.v.resize((size_t) c.n_kv*N_KVH*rv);
    std::vector<float> row(D);
    for (int64_t r = 0; r < c.n_kv*N_KVH; r++) {
        // per-row magnitude variety: the scales of the blocks cover small, large and negative values
        const float m = c.kscale*(0.25f + 2.0f*ud(rng));
        for (auto & x : row) { x = nd(rng)*m; }
        tk->from_float_ref(row.data(), in.k.data() + r*rk, D);
        const float mv = c.vscale*(0.25f + 2.0f*ud(rng));
        for (auto & x : row) { x = nd(rng)*mv; }
        tv->from_float_ref(row.data(), in.v.data() + r*rv, D);
    }

    in.cell_pos.resize(c.n_kv);
    for (int64_t i = 0; i < c.n_kv; i++) { in.cell_pos[i] = (c.holes && ud(rng) < 0.2f) ? -1 : (int32_t) i; }
    in.tok_pos.resize(c.n_t);
    for (int64_t t = 0; t < c.n_t; t++) {
        // a causal cut inside the cells (queries early, in the middle or at the end of the cache)
        in.tok_pos[t] = c.all_masked ? -1 : (int32_t) (c.n_kv/4 + (int64_t) (ud(rng)*(float) (3*c.n_kv/4)));
        if (t % 5 == 0 && !c.all_masked) { in.tok_pos[t] = (int32_t) (c.n_kv - 1); }
    }

    in.sel.assign((size_t) c.n_t*c.n_sel, 0);
    in.top_score.resize((size_t) c.n_t*c.n_top);
    for (int64_t t = 0; t < c.n_t; t++) {
        int32_t * s = &in.sel[t*c.n_sel];
        std::unordered_set<int32_t> used;
        std::uniform_int_distribution<int64_t> ui(0, c.n_kv - 1);
        for (int64_t j = 0; j < c.n_sel; j++) {
            int32_t cell;
            do { cell = (int32_t) ui(rng); } while (!used.insert(cell).second && (int64_t) used.size() < c.n_kv);
            s[j] = cell;
        }
        if (c.edge_cells) {
            // cells 0 and n_kv-1 at the first and last slots, and nowhere else (the live cells are distinct)
            s[0] = 0; s[c.n_sel - 1] = (int32_t) (c.n_kv - 1);
            for (int64_t j = 1; j < c.n_sel - 1; j++) {
                if (s[j] == 0 || s[j] == (int32_t) (c.n_kv - 1)) {
                    int32_t cell; do { cell = (int32_t) ui(rng); } while (cell == 0 || cell == (int32_t) (c.n_kv - 1) || !used.insert(cell).second);
                    s[j] = cell;
                }
            }
        }
        if (c.invalid_idx) {
            for (int64_t j = 3; j < c.n_sel; j += 37) {
                const int w = (int) (j % 3);
                s[j] = w == 0 ? -1 : w == 1 ? (int32_t) c.n_kv : (int32_t) (c.n_kv + 12345);
            }
        }
        for (int p = 0; p < c.n_top; p++) {
            const bool dead = c.empty || (c.dead_pools && ud(rng) < 0.3f);
            in.top_score[t*c.n_top + p] = dead ? -INFINITY : (p % 7 == 0 ? 0.0f : nd(rng));
        }
        if (c.empty) { for (int64_t j = c.n_top*c.kpool; j < c.n_sel; j++) { s[j] = -1; } }
    }
    return in;
}

// the reference mask value (F32: 0 or -inf)
static float ref_mask(const tcase & c, const inputs & in, int64_t t, int64_t j) {
    const int32_t cell = in.sel[t*c.n_sel + j];
    const bool ok = cell >= 0 && cell < c.n_kv;
    const bool live = ok && (j >= (int64_t) c.n_top*c.kpool || in.top_score[t*c.n_top + j/c.kpool] != -INFINITY);
    if (!live) { return -INFINITY; }
    const int32_t pc = in.cell_pos[cell];
    return pc >= 0 && pc <= in.tok_pos[t] ? 0.0f : -INFINITY;
}

static std::vector<float> cpu_reference(const tcase & c, const inputs & in, float scale) {
    const ggml_type_traits * tk = ggml_get_type_traits(GGML_TYPE_Q8_0);
    const ggml_type_traits * tv = ggml_get_type_traits(GGML_TYPE_Q4_0);
    const size_t rk = ggml_row_size(GGML_TYPE_Q8_0, D), rv = ggml_row_size(GGML_TYPE_Q4_0, D);
    const auto row_of = [&](int64_t cell, int kvh) { return c.kv_perm ? cell*N_KVH + kvh : (int64_t) kvh*c.n_kv + cell; };
    std::vector<float> out((size_t) c.n_t*N_HEAD*D, 0.0f);
    std::vector<float> kd(c.n_sel*D), vd(c.n_sel*D), z(c.n_sel);
    std::vector<char> alive(c.n_sel);
    for (int64_t t = 0; t < c.n_t; t++) {
        for (int kvh = 0; kvh < N_KVH; kvh++) {
            for (int64_t j = 0; j < c.n_sel; j++) {
                alive[j] = ref_mask(c, in, t, j) != -INFINITY;
                if (alive[j]) {
                    const int32_t cell = in.sel[t*c.n_sel + j];
                    tk->to_float(in.k.data() + row_of(cell, kvh)*rk, &kd[j*D], D);
                    tv->to_float(in.v.data() + row_of(cell, kvh)*rv, &vd[j*D], D);
                }
            }
            for (int hh = 0; hh < N_HEAD/N_KVH; hh++) {
                const int h = kvh*(N_HEAD/N_KVH) + hh;
                const float * q = &in.q[((size_t) t*N_HEAD + h)*D];
                double m = -INFINITY;
                for (int64_t j = 0; j < c.n_sel; j++) {
                    if (!alive[j]) { continue; }
                    double s = 0; for (int d = 0; d < D; d++) { s += (double) q[d]*kd[j*D + d]; }
                    z[j] = (float) (s*scale);
                    m = std::max(m, (double) z[j]);
                }
                double l = 0; std::vector<double> acc(D, 0.0);
                for (int64_t j = 0; j < c.n_sel; j++) {
                    if (!alive[j]) { continue; }
                    const double p = std::exp((double) z[j] - m);
                    l += p; for (int d = 0; d < D; d++) { acc[d] += p*vd[j*D + d]; }
                }
                float * o = &out[((size_t) t*N_HEAD + h)*D];
                for (int d = 0; d < D; d++) { o[d] = l > 0 ? (float) (acc[d]/l) : 0.0f; }
            }
        }
    }
    return out;
}

struct built {
    ggml_context_ptr ctx; ggml_cgraph * gf = nullptr; ggml_backend_buffer_ptr buf;
    ggml_tensor *q, *k, *v, *sel, *score, *cpos, *tpos, *mask9, *mask1, *out, *dense_mask, *out_dense;
};

static void upload(const tcase & c, const inputs & in, built & b) {
    // Q as the model builds it: the tensor ([D, n_head, n_t]) permuted into [D, n_t, n_head]: row (t, h) at t*n_head + h
    if (c.q_type == GGML_TYPE_F32) {
        if (c.q_perm) { ggml_backend_tensor_set(b.q->src[0] ? b.q->src[0] : b.q, in.q.data(), 0, in.q.size()*4); }
        else {
            std::vector<float> tmp(in.q.size());
            for (int64_t t = 0; t < c.n_t; t++) for (int h = 0; h < N_HEAD; h++)
                memcpy(&tmp[((size_t) h*c.n_t + t)*D], &in.q[((size_t) t*N_HEAD + h)*D], D*4);
            ggml_backend_tensor_set(b.q, tmp.data(), 0, tmp.size()*4);
        }
    } else {
        std::vector<ggml_fp16_t> tmp(in.q.size());
        const bool perm = c.q_perm;
        for (int64_t t = 0; t < c.n_t; t++) for (int h = 0; h < N_HEAD; h++)
            ggml_fp32_to_fp16_row(&in.q[((size_t) t*N_HEAD + h)*D], &tmp[perm ? ((size_t) t*N_HEAD + h)*D : ((size_t) h*c.n_t + t)*D], D);
        ggml_backend_tensor_set(perm ? b.q->src[0] : b.q, tmp.data(), 0, tmp.size()*2);
    }
    ggml_tensor * kr = c.kv_perm ? b.k->src[0] : b.k;
    ggml_tensor * vr = c.kv_perm ? b.v->src[0] : b.v;
    ggml_backend_tensor_set(kr, in.k.data(), 0, in.k.size());
    ggml_backend_tensor_set(vr, in.v.data(), 0, in.v.size());
    ggml_backend_tensor_set(b.sel, in.sel.data(), 0, in.sel.size()*4);
    ggml_backend_tensor_set(b.score, in.top_score.data(), 0, in.top_score.size()*4);
    ggml_backend_tensor_set(b.cpos, in.cell_pos.data(), 0, in.cell_pos.size()*4);
    ggml_backend_tensor_set(b.tpos, in.tok_pos.data(), 0, in.tok_pos.size()*4);
}

// K/V/Q views as qwen4exp + build_attn_mha hand them to the flash attention
static built build(const tcase & c, float scale, bool with_dense) {
    built b;
    ggml_init_params ip = { 256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    b.ctx.reset(ggml_init(ip));
    ggml_context * x = b.ctx.get();
    if (c.q_perm) {
        ggml_tensor * qr = ggml_new_tensor_3d(x, c.q_type, D, N_HEAD, c.n_t);
        b.q = ggml_permute(x, qr, 0, 2, 1, 3);
    } else {
        b.q = ggml_new_tensor_4d(x, c.q_type, D, c.n_t, N_HEAD, 1);
    }
    if (c.kv_perm) {
        b.k = ggml_permute(x, ggml_new_tensor_3d(x, GGML_TYPE_Q8_0, D, N_KVH, c.n_kv), 0, 2, 1, 3);
        b.v = ggml_permute(x, ggml_new_tensor_3d(x, GGML_TYPE_Q4_0, D, N_KVH, c.n_kv), 0, 2, 1, 3);
    } else {
        b.k = ggml_new_tensor_4d(x, GGML_TYPE_Q8_0, D, c.n_kv, N_KVH, 1);
        b.v = ggml_new_tensor_4d(x, GGML_TYPE_Q4_0, D, c.n_kv, N_KVH, 1);
    }
    b.sel   = ggml_new_tensor_2d(x, GGML_TYPE_I32, c.n_sel, c.n_t);
    b.score = ggml_new_tensor_2d(x, GGML_TYPE_F32, c.n_top, c.n_t);
    b.cpos  = ggml_new_tensor_1d(x, GGML_TYPE_I32, c.n_kv);
    b.tpos  = ggml_new_tensor_1d(x, GGML_TYPE_I32, c.n_t);
    b.mask9 = ggml_qsa_select(x, b.sel, b.score, b.cpos, b.tpos, c.kpool, GGML_QSA_SELECT_RDNA2_PREFILL);
    b.mask1 = ggml_qsa_select(x, b.sel, b.score, b.cpos, b.tpos, c.kpool, GGML_QSA_SELECT_MASK);
    b.out   = ggml_flash_attn_ext(x, b.q, b.k, b.v, b.mask9, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_n_kv_max(b.out, (int32_t) c.n_sel);
    ggml_prec_set_acc(b.out, GGML_PREC_F32);
    b.gf = ggml_new_graph(x);
    ggml_build_forward_expand(b.gf, b.sel->op == GGML_OP_NONE ? b.mask1 : b.mask1);
    ggml_build_forward_expand(b.gf, b.mask9);
    ggml_build_forward_expand(b.gf, b.out);
    b.dense_mask = b.out_dense = nullptr;
    if (with_dense) {
        b.dense_mask = ggml_qsa_select(x, b.sel, b.score, b.cpos, b.tpos, c.kpool, GGML_QSA_SELECT_DENSE);
        // as build_attn_qsa: the cache's K/V converted to F16 in their own layout, then viewed per head
        ggml_tensor * k16 = c.kv_perm ? ggml_permute(x, ggml_cast(x, b.k->src[0], GGML_TYPE_F16), 0, 2, 1, 3) : ggml_cast(x, b.k, GGML_TYPE_F16);
        ggml_tensor * v16 = c.kv_perm ? ggml_permute(x, ggml_cast(x, b.v->src[0], GGML_TYPE_F16), 0, 2, 1, 3) : ggml_cast(x, b.v, GGML_TYPE_F16);
        ggml_tensor * dm = ggml_reshape_4d(x, b.dense_mask, c.n_kv, c.n_t, 1, 1);
        b.out_dense = ggml_flash_attn_ext(x, b.q, k16, v16, dm, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(b.out_dense, GGML_PREC_F32);
        ggml_build_forward_expand(b.gf, b.out_dense);
    }
    b.buf.reset(ggml_backend_alloc_ctx_tensors(x, g_gpu));
    return b;
}

static int g_skipped = 0;   // cases not run (allocation failure): never counted as PASS

static bool run_case(const tcase & c, uint64_t seed, bool bench) {
    std::mt19937_64 rng(seed);
    const float scale = 1.0f/16.0f;
    printf("%-34s n_kv=%-7" PRId64 " n_t=%-5" PRId64 " n_sel=%-5" PRId64 " ", c.name.c_str(), c.n_kv, c.n_t, c.n_sel);
    fflush(stdout);
    const inputs in = make_inputs(c, rng);
    built b = build(c, scale, c.dense_ref);
    if (!b.buf) { printf("SKIP (out of memory allocating the case)\n"); g_skipped++; return true; }
    upload(c, in, b);
    bool ok = true;
    std::string report;

    // the graph several times: the first runs are eager, the later ones are replays of the captured graph
    std::vector<float> first;
    for (int r = 0; r < 5; r++) {
        if (ggml_backend_graph_compute(g_gpu, b.gf) != GGML_STATUS_SUCCESS) { printf("FAIL compute\n"); return false; }
        std::vector<float> o = to_f32(b.out);
        if (r == 0) { first = o; }
        else if (o != first) { ok = false; report += " [run " + std::to_string(r) + " differs from run 0]"; }
    }
    // the mask: mode 9 against mode 1 (bitwise) and against the CPU reference
    std::vector<ggml_fp16_t> m9(ggml_nelements(b.mask9)), m1(ggml_nelements(b.mask1));
    ggml_backend_tensor_get(b.mask9, m9.data(), 0, m9.size()*2);
    ggml_backend_tensor_get(b.mask1, m1.data(), 0, m1.size()*2);
    int64_t mask_diff = 0, mask_ref_diff = 0;
    for (size_t i = 0; i < m9.size(); i++) {
        if (m9[i] != m1[i]) { mask_diff++; }
        const float rv = ref_mask(c, in, (int64_t) (i / c.n_sel), (int64_t) (i % c.n_sel));
        if (ggml_fp16_to_fp32(m9[i]) != rv) { mask_ref_diff++; }
    }
    if (mask_diff || mask_ref_diff) { ok = false; report += " [mask: " + std::to_string(mask_diff) + " vs mode 1, " + std::to_string(mask_ref_diff) + " vs reference]"; }
    // live cells distinct (the contract of select)
    int64_t dups = 0;
    for (int64_t t = 0; t < c.n_t; t++) {
        std::unordered_set<int32_t> seen;
        for (int64_t j = 0; j < c.n_sel; j++) {
            if (in.sel[t*c.n_sel + j] >= 0 && in.sel[t*c.n_sel + j] < c.n_kv &&
                (j >= (int64_t) c.n_top*c.kpool || in.top_score[t*c.n_top + j/c.kpool] != -INFINITY)) {
                if (!seen.insert(in.sel[t*c.n_sel + j]).second) { dups++; }
            }
        }
    }
    if (dups) { ok = false; report += " [" + std::to_string(dups) + " duplicate live cells]"; }

    const std::vector<float> ref = cpu_reference(c, in, scale);
    // the output of the flash attention: [D, n_head, n_t] contiguous, element ((t*n_head + h)*D + d), as `ref`
    const stats s = compare(first, ref);
    const bool pass = s.n_nan == 0 && s.nrmse <= 3e-3 && s.max_abs <= 0.05*std::max(1.0f, c.vscale);   // absolute gate scaled by the V magnitude of the case
    ok = ok && pass;
    char buf[256];
    snprintf(buf, sizeof(buf), "cpu: nrmse=%.2e max=%.2e p99=%.2e nan=%" PRId64, s.nrmse, s.max_abs, s.p99, s.n_nan);
    report = std::string(buf) + report;
    if (c.dense_ref) {
        // the dense path converts K/V to F16 and accumulates in F16 tiles: it is itself ~1e-3..5e-3 from the FP32 reference
        // (shown as "dense vs cpu"), so it is a sanity bound (<1e-2), not the strict gate: that is the CPU FP32 oracle
        const stats d = compare(first, to_f32(b.out_dense));
        const bool dp = d.n_nan == 0 && d.nrmse <= 1e-2;
        ok = ok && dp;
        const stats dc = compare(to_f32(b.out_dense), ref);
        snprintf(buf, sizeof(buf), " | dense-gpu: nrmse=%.2e max=%.2e p99=%.2e (dense vs cpu: %.2e)", d.nrmse, d.max_abs, d.p99, dc.nrmse);
        report += buf;
    }
    bool zero_ok = true;
    if (c.empty || c.all_masked) {
        for (float x : first) { if (x != 0.0f) { zero_ok = false; } }
        ok = ok && zero_ok;
        report += zero_ok ? " | empty -> exactly 0" : " | empty NOT 0";
    }
    printf("%s  %s\n", ok ? "PASS" : "FAIL", report.c_str());
    (void) bench;
    return ok;
}

static double time_graph(ggml_cgraph * gf, int iters) {
    for (int i = 0; i < 3; i++) { ggml_backend_graph_compute(g_gpu, gf); }
    ggml_backend_synchronize(g_gpu);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) { ggml_backend_graph_compute(g_gpu, gf); }
    ggml_backend_synchronize(g_gpu);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()/iters;
}

// n_t queries by the sparse kernel, against the dense path of the same inputs by slices as the model runs it (a slice
// of queries whose dense mask is under 512 MiB)
static void bench_case(int64_t n_kv, int64_t n_t, int64_t n_sel, int iters, std::mt19937_64 & rng) {
    tcase c = mk("bench", n_kv, n_t, n_sel);
    c.dense_ref = false;
    const inputs in = make_inputs(c, rng);
    built b = build(c, 1.0f/16.0f, false);
    if (!b.buf) { printf("bench n_kv=%" PRId64 " n_t=%" PRId64 ": SKIP (memory)\n", n_kv, n_t); return; }
    upload(c, in, b);
    // the sparse op alone (the select nodes run too: one mask kernel, negligible next to the attention)
    ggml_cgraph * g1 = ggml_new_graph(b.ctx.get());
    ggml_build_forward_expand(g1, b.out);
    const double ms_sparse = time_graph(g1, iters);

    const int64_t n_slice = std::max<int64_t>(16, std::min<int64_t>(n_t, (512ll << 20)/(n_kv*2)) & ~15ll);
    tcase cd = mk("dense", n_kv, n_slice, n_sel);
    cd.dense_ref = true;
    std::mt19937_64 rng2(1);
    const inputs ind = make_inputs(cd, rng2);
    built bd = build(cd, 1.0f/16.0f, true);
    double ms_dense = -1;
    if (bd.buf) {
        upload(cd, ind, bd);
        ggml_cgraph * g2 = ggml_new_graph(bd.ctx.get());
        ggml_build_forward_expand(g2, bd.out_dense);
        const double ms_slice = time_graph(g2, std::max(1, iters/2)); // includes the K/V F16 cast of the cache: the model does it once
        ms_dense = ms_slice*(double) n_t/(double) n_slice;
    }
    const double kvbytes = (double) n_t*n_sel*N_KVH*(272 + 144);
    printf("bench n_kv=%-7" PRId64 " n_t=%-5" PRId64 " n_sel=%-5" PRId64 " sparse=%8.2f ms (%.0f GB/s of K/V cells read at least once per query)  dense-slices(%" PRId64 "/slice, scaled, incl. F16 cast)=%s%.2f ms\n",
           n_kv, n_t, n_sel, ms_sparse, kvbytes/ms_sparse/1e6, n_slice, ms_dense < 0 ? "OOM " : "", ms_dense < 0 ? 0.0 : ms_dense);
}

int main(int argc, char ** argv) {
    bool quick = false, bench = false; uint64_t seed = 1234; std::string only;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--quick")) quick = true;
        else if (!strcmp(argv[i], "--bench")) bench = true;
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--cases") && i + 1 < argc) only = argv[++i];
    }
    ggml_backend_load_all();
    g_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!g_gpu) { printf("no GPU backend: NOT VERIFIED\n"); return 2; }
    printf("backend: %s\n", ggml_backend_name(g_gpu));
    {   // dispatch: the CPU backend must refuse the compact mask, the GPU must take it
        tcase c = mk("dispatch", 4096, 64, 512);
        built b = build(c, 1.0f, false);
        ggml_backend_dev_t gdev = ggml_backend_get_device(g_gpu);
        ggml_backend_dev_t cdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        const bool gpu_ok = ggml_backend_dev_supports_op(gdev, b.out);
        const bool cpu_no = !ggml_backend_dev_supports_op(cdev, b.out);
        printf("dispatch: GPU supports mode 9 FLASH_ATTN_EXT: %s; CPU refuses it: %s\n", gpu_ok ? "yes" : "NO", cpu_no ? "yes" : "NO");
        if (!gpu_ok || !cpu_no) { printf("RESULT: FAIL (dispatch)\n"); return 1; }
    }

    std::vector<tcase> cases;
    { auto c = mk("base-64q-8k", 8192, 64, 2304); cases.push_back(c); }
    { auto c = mk("q255-1024kv-256sel", 1024, 255, 256); cases.push_back(c); }
    { auto c = mk("q256-kv32767-sel512", 32767, 256, 512); cases.push_back(c); }
    { auto c = mk("q256-kv32768-sel2304", 32768, 256, 2304); c.edge_cells = true; cases.push_back(c); }
    { auto c = mk("sel2560-kv65536", 65536, 128, 2560); cases.push_back(c); }
    { auto c = mk("invalid-indices", 32768, 96, 2304); c.invalid_idx = true; c.edge_cells = true; cases.push_back(c); }
    { auto c = mk("dead-pools", 32768, 96, 2304); c.dead_pools = true; cases.push_back(c); }
    { auto c = mk("holes-other-seq", 32768, 96, 2304); c.holes = true; c.dead_pools = true; c.invalid_idx = true; cases.push_back(c); }
    { auto c = mk("empty-selection", 32768, 64, 2304); c.empty = true; c.dense_ref = false; cases.push_back(c); }
    { auto c = mk("all-masked-by-causality", 32768, 64, 2304); c.all_masked = true; c.dense_ref = false; cases.push_back(c); }  // a fully masked row is NaN in the dense path
    { auto c = mk("kv-not-permuted", 32768, 64, 2304); c.kv_perm = false; c.q_perm = false; cases.push_back(c); }
    { auto c = mk("q-f16", 32768, 64, 2304); c.q_type = GGML_TYPE_F16; c.dense_ref = false; cases.push_back(c); }
    { auto c = mk("nsel-partial-2300", 32768, 64, 2300); cases.push_back(c); }   // n_sel not a multiple of the 32-slot step
    { auto c = mk("big-k-values", 32768, 64, 2304); c.kscale = 3.0f; c.vscale = 4.0f; cases.push_back(c); }
    { auto c = mk("small-values", 32768, 64, 2304); c.kscale = 1e-3f; c.vscale = 1e-3f; cases.push_back(c); }
    if (!quick) {
        { auto c = mk("q4096-kv65536", 65536, 4096, 2304); c.dense_ref = false; cases.push_back(c); }
        { auto c = mk("kv131072", 131072, 256, 2304); c.dense_ref = false; cases.push_back(c); }
        { auto c = mk("kv262144", 262144, 128, 2304); c.dense_ref = false; c.edge_cells = true; cases.push_back(c); }
        { auto c = mk("q8192-kv65536", 65536, 8192, 2304); c.dense_ref = false; cases.push_back(c); }
    }

    int fail = 0, total = 0;
    for (const auto & c : cases) {
        if (!only.empty() && only.find(c.name) == std::string::npos) { continue; }
        total++;
        if (!run_case(c, seed + (uint64_t) total, bench)) { fail++; }
    }
    if (bench) {
        std::mt19937_64 rng(7);
        const int64_t kvs[] = { 32768, 65536, 131072, 262144 };
        const int64_t nts[] = { 4096, 8192 };
        for (int64_t n_kv : kvs) for (int64_t n_t : nts) { bench_case(n_kv, n_t, 2304, 5, rng); }
    }
    // a skipped case (allocation failure) is neither a pass nor silently ignored: it fails the matrix
    printf("RESULT: %d/%d cases %s (skipped: %d)\n", total - fail - g_skipped, total, (fail || g_skipped) ? "FAILED" : "PASS", g_skipped);
    ggml_backend_free(g_gpu);
    return (fail || g_skipped) ? 1 : 0;
}

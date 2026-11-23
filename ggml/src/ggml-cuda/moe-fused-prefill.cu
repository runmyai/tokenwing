#include "moe-fused-prefill.cuh"
#include "mma.cuh"
#include "mmid.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <atomic>
#include <cstdlib>

static std::atomic<bool> g_mmq_moe_opt_on{false};
static std::atomic<bool> g_moe_fused_prefill_on{false};

bool ggml_cuda_mmq_moe_opt_enabled() {
    static const int env = []() {
        const char * s = std::getenv("GGML_CUDA_MMQ_MOE_OPT");
        return s ? std::atoi(s) : -1;
    }();
    return env >= 0 ? (env != 0) : g_mmq_moe_opt_on.load(std::memory_order_relaxed);
}

void ggml_cuda_set_mmq_moe_opt(bool on) {
    g_mmq_moe_opt_on.store(on, std::memory_order_relaxed);
}

bool ggml_cuda_moe_fused_prefill_enabled() {
    static const int env = []() {
        const char * s = std::getenv("GGML_CUDA_MOE_FUSED");
        return s ? std::atoi(s) : -1;
    }();
    return env >= 0 ? (env != 0) : g_moe_fused_prefill_on.load(std::memory_order_relaxed);
}

void ggml_cuda_set_moe_fused_prefill(bool on) {
    g_moe_fused_prefill_on.store(on, std::memory_order_relaxed);
}

struct alignas(8) moe_q8_blk {
    float   d;
    int16_t sum_lo;
    int16_t sum_hi;
    int32_t qs[8];
};
static_assert(sizeof(moe_q8_blk) == 40, "unexpected moe_q8_blk size");

struct moe_work_tile {
    int32_t expert;
    int32_t pair0;
    int32_t count;
    int32_t pad;
};

static __global__ void moe_quantize_x_kernel(
        const float * __restrict__ x,
        moe_q8_blk *  __restrict__ out,
        int                        k_blks,
        int                        nt,
        int64_t                    s_tok) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x & 31;
    const int total   = k_blks * nt;
    if (warp_id >= total) {
        return;
    }
    const int blk = warp_id % k_blks;
    const int tok = warp_id / k_blks;
    const float v = x[(int64_t) tok * s_tok + blk * 32 + lane];

    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off, 32));
    }
    const float d     = amax / 127.0f;
    const float inv_d = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int   q     = max(-127, min(127, __float2int_rn(v * inv_d)));

    int half_sum = q;
    #pragma unroll
    for (int off = 8; off >= 1; off >>= 1) {
        half_sum += __shfl_xor_sync(0xFFFFFFFFu, half_sum, off, 32);
    }
    const int sum_lo = __shfl_sync(0xFFFFFFFFu, half_sum, 0,  32);
    const int sum_hi = __shfl_sync(0xFFFFFFFFu, half_sum, 16, 32);

    const int q0 = __shfl_sync(0xFFFFFFFFu, q, (lane & 7) * 4 + 0, 32);
    const int q1 = __shfl_sync(0xFFFFFFFFu, q, (lane & 7) * 4 + 1, 32);
    const int q2 = __shfl_sync(0xFFFFFFFFu, q, (lane & 7) * 4 + 2, 32);
    const int q3 = __shfl_sync(0xFFFFFFFFu, q, (lane & 7) * 4 + 3, 32);
    const uint32_t packed = ((uint32_t)(uint8_t)q0)       |
                            ((uint32_t)(uint8_t)q1 << 8)  |
                            ((uint32_t)(uint8_t)q2 << 16) |
                            ((uint32_t)(uint8_t)q3 << 24);

    moe_q8_blk & dst = out[(int64_t) tok * k_blks + blk];
    if (lane < 8) {
        dst.qs[lane] = (int32_t) packed;
    }
    if (lane == 0) {
        dst.d      = d;
        dst.sum_lo = (int16_t) sum_lo;
        dst.sum_hi = (int16_t) sum_hi;
    }
}

static __global__ void moe_plan_dual_tiles_kernel(
        const int32_t * __restrict__ expert_bounds,
        moe_work_tile * __restrict__ dense_tiles,
        moe_work_tile * __restrict__ sparse_tiles,
        int32_t *       __restrict__ tile_counts,
        int                          active_experts) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int32_t nd = 0;
        int32_t ns = 0;
        for (int e = 0; e < active_experts; ++e) {
            int32_t p0  = expert_bounds[e];
            int32_t rem = expert_bounds[e + 1] - p0;
            while (rem >= 17) {
                const int32_t c = rem >= 32 ? 32 : rem;
                dense_tiles[nd++] = { e, p0, c, 0 };
                p0  += c;
                rem -= c;
            }
            while (rem > 0) {
                const int32_t c = rem >= 8 ? 8 : rem;
                sparse_tiles[ns++] = { e, p0, c, 0 };
                p0  += c;
                rem -= c;
            }
        }
        tile_counts[0] = nd;
        tile_counts[1] = ns;
    }
}

__device__ __constant__ int8_t k_moe_iq4nl_lut[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

__device__ __constant__ int8_t k_moe_mxfp4_lut[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

static __device__ __forceinline__ int pack4_s8(int b0, int b1, int b2, int b3) {
    return (int)(((uint32_t)(uint8_t)b0)       |
                 ((uint32_t)(uint8_t)b1 << 8)  |
                 ((uint32_t)(uint8_t)b2 << 16) |
                 ((uint32_t)(uint8_t)b3 << 24));
}

template <ggml_type wtype>
static __device__ __forceinline__ void moe_unpack_w_blk32(
        const char * __restrict__ row_ptr,
        int                       kb32,
        int                       t,
        int &                     a_lo,
        int &                     a_hi,
        float &                   w_scale,
        float &                   w_zp_lo,
        float &                   w_zp_hi) {
    if constexpr (wtype == GGML_TYPE_Q8_0) {
        const block_q8_0 * b = (const block_q8_0 *) row_ptr + kb32;
        w_scale = __half2float(b->d);
        w_zp_lo = 0.0f;
        w_zp_hi = 0.0f;
        a_lo    = get_int_b2(b->qs, t);
        a_hi    = get_int_b2(b->qs, 4 + t);
    } else if constexpr (wtype == GGML_TYPE_Q4_0) {
        const block_q4_0 * b = (const block_q4_0 *) row_ptr + kb32;
        const float d        = __half2float(b->d);
        const uint32_t p     = (uint32_t) get_int_b2(b->qs, t);
        w_scale = d;
        w_zp_lo = 8.0f * d;
        w_zp_hi = 8.0f * d;
        a_lo    = (int)(p & 0x0F0F0F0Fu);
        a_hi    = (int)((p >> 4) & 0x0F0F0F0Fu);
    } else if constexpr (wtype == GGML_TYPE_Q2_0) {
        const block_q2_0 * b = (const block_q2_0 *) row_ptr + (kb32 >> 1);
        const int          h = (kb32 & 1) * 8;
        const float        d = __half2float(b->d);
        const uint8_t     b0 = b->qs[h + t];
        const uint8_t     b1 = b->qs[h + 4 + t];
        w_scale = d;
        w_zp_lo = d;
        w_zp_hi = d;
        a_lo    = pack4_s8(b0 & 3, (b0 >> 2) & 3, (b0 >> 4) & 3, (b0 >> 6) & 3);
        a_hi    = pack4_s8(b1 & 3, (b1 >> 2) & 3, (b1 >> 4) & 3, (b1 >> 6) & 3);
    } else if constexpr (wtype == GGML_TYPE_Q3_K) {
        const block_q3_K * b   = (const block_q3_K *) row_ptr + (kb32 >> 3);
        const int sub32        = kb32 & 7;
        const int half128      = sub32 >> 2;
        const int l            = sub32 & 3;
        const float d          = __half2float(b->d);
        const uint32_t ql0     = ((uint32_t) get_int_b2(b->qs, half128 * 8 + t)     >> (2 * l)) & 0x03030303u;
        const uint32_t ql1     = ((uint32_t) get_int_b2(b->qs, half128 * 8 + 4 + t) >> (2 * l)) & 0x03030303u;
        const uint32_t qh0     = (((uint32_t) get_int_b2(b->hmask, t)     >> sub32) & 0x01010101u) << 2;
        const uint32_t qh1     = (((uint32_t) get_int_b2(b->hmask, 4 + t) >> sub32) & 0x01010101u) << 2;
        const uint32_t u0      = ql0 | qh0;
        const uint32_t u1      = ql1 | qh1;

        const int ksc      = sub32 >> 1;
        const uint32_t scl = ((uint32_t) get_int_b2(b->scales, ksc & 1) >> (4 * (ksc >> 1))) & 0x0F0F0F0Fu;
        const uint32_t sch = (((uint32_t) get_int_b2(b->scales, 2) >> (2 * ksc)) << 4) & 0x30303030u;
        const uint32_t scraw = scl | sch;
        const int sh       = 16 * (sub32 & 1);
        const int s0       = (int)((scraw >> sh) & 0x3Fu) - 32;
        const int s1       = (int)((scraw >> (sh + 8)) & 0x3Fu) - 32;

        // Exact two's-complement shift: (u - 3) in [-3..+4], (u - 3)*s in [-128..+124] fits int8_t without overflow.
        // Since (u - 4)*s = (u - 3)*s - s, the -s term folds cleanly into sum_lo and sum_hi.
        a_lo = pack4_s8(((int)(u0 & 7u) - 3) * s0,
                        ((int)((u0 >> 8) & 7u) - 3) * s0,
                        ((int)((u0 >> 16) & 7u) - 3) * s0,
                        ((int)((u0 >> 24) & 7u) - 3) * s0);
        a_hi = pack4_s8(((int)(u1 & 7u) - 3) * s1,
                        ((int)((u1 >> 8) & 7u) - 3) * s1,
                        ((int)((u1 >> 16) & 7u) - 3) * s1,
                        ((int)((u1 >> 24) & 7u) - 3) * s1);
        w_scale = d;
        w_zp_lo = d * (float) s0;
        w_zp_hi = d * (float) s1;
    } else if constexpr (wtype == GGML_TYPE_Q4_K) {
        const block_q4_K * b = (const block_q4_K *) row_ptr + (kb32 >> 3);
        const int is         = kb32 & 7;
        uint8_t sc, m;
        if (is < 4) {
            sc = b->scales[is] & 63;
            m  = b->scales[is + 4] & 63;
        } else {
            sc = (b->scales[is + 4] & 0x0F) | ((b->scales[is - 4] >> 6) << 4);
            m  = (b->scales[is + 4] >> 4)   | ((b->scales[is]     >> 6) << 4);
        }
        const float d        = __low2float(b->dm)  * (float) sc;
        const float dm       = __high2float(b->dm) * (float) m;
        const int q_off      = (is >> 1) * 8;
        const int q_sh       = (is & 1) * 4;
        a_lo    = (int)(((uint32_t) get_int_b2(b->qs, q_off + t)     >> q_sh) & 0x0F0F0F0Fu);
        a_hi    = (int)(((uint32_t) get_int_b2(b->qs, q_off + 4 + t) >> q_sh) & 0x0F0F0F0Fu);
        w_scale = d;
        w_zp_lo = dm;
        w_zp_hi = dm;
    } else if constexpr (wtype == GGML_TYPE_Q5_K) {
        const block_q5_K * b = (const block_q5_K *) row_ptr + (kb32 >> 3);
        const int is         = kb32 & 7;
        uint8_t sc, m;
        if (is < 4) {
            sc = b->scales[is] & 63;
            m  = b->scales[is + 4] & 63;
        } else {
            sc = (b->scales[is + 4] & 0x0F) | ((b->scales[is - 4] >> 6) << 4);
            m  = (b->scales[is + 4] >> 4)   | ((b->scales[is]     >> 6) << 4);
        }
        const float d        = __low2float(b->dm)  * (float) sc;
        const float dm       = __high2float(b->dm) * (float) m;
        const int q_off      = (is >> 1) * 8;
        const int q_sh       = (is & 1) * 4;
        const uint32_t ql0   = ((uint32_t) get_int_b2(b->qs, q_off + t)     >> q_sh) & 0x0F0F0F0Fu;
        const uint32_t ql1   = ((uint32_t) get_int_b2(b->qs, q_off + 4 + t) >> q_sh) & 0x0F0F0F0Fu;
        const uint32_t qh0   = (((uint32_t) get_int_b2(b->qh, t)     >> is) & 0x01010101u) << 4;
        const uint32_t qh1   = (((uint32_t) get_int_b2(b->qh, 4 + t) >> is) & 0x01010101u) << 4;
        a_lo    = (int)(ql0 | qh0);
        a_hi    = (int)(ql1 | qh1);
        w_scale = d;
        w_zp_lo = dm;
        w_zp_hi = dm;
    } else if constexpr (wtype == GGML_TYPE_IQ4_NL) {
        const block_iq4_nl * b = (const block_iq4_nl *) row_ptr + kb32;
        const uint32_t p       = (uint32_t) get_int_b2(b->qs, t);
        w_scale = __half2float(b->d);
        w_zp_lo = 0.0f;
        w_zp_hi = 0.0f;
        a_lo    = pack4_s8(k_moe_iq4nl_lut[p & 15u], k_moe_iq4nl_lut[(p >> 8) & 15u],
                           k_moe_iq4nl_lut[(p >> 16) & 15u], k_moe_iq4nl_lut[(p >> 24) & 15u]);
        a_hi    = pack4_s8(k_moe_iq4nl_lut[(p >> 4) & 15u], k_moe_iq4nl_lut[(p >> 12) & 15u],
                           k_moe_iq4nl_lut[(p >> 20) & 15u], k_moe_iq4nl_lut[(p >> 28) & 15u]);
    } else if constexpr (wtype == GGML_TYPE_MXFP4) {
        const block_mxfp4 * b = (const block_mxfp4 *) row_ptr + kb32;
        const uint32_t p      = (uint32_t) get_int_b1(b->qs, t);
        w_scale = ggml_cuda_e8m0_to_fp32(b->e) * 0.5f;
        w_zp_lo = 0.0f;
        w_zp_hi = 0.0f;
        a_lo    = pack4_s8(k_moe_mxfp4_lut[p & 15u], k_moe_mxfp4_lut[(p >> 8) & 15u],
                           k_moe_mxfp4_lut[(p >> 16) & 15u], k_moe_mxfp4_lut[(p >> 24) & 15u]);
        a_hi    = pack4_s8(k_moe_mxfp4_lut[(p >> 4) & 15u], k_moe_mxfp4_lut[(p >> 12) & 15u],
                           k_moe_mxfp4_lut[(p >> 20) & 15u], k_moe_mxfp4_lut[(p >> 28) & 15u]);
    }
}

static __device__ __forceinline__ void moe_mma_16x8x32_f32(
        const int               a_lo_0,
        const int               a_hi_0,
        const int               a_lo_1,
        const int               a_hi_1,
        const moe_q8_blk *      smem_b8,
        const int               g,
        const int               t,
        float &                 dot_r0_c0,
        float &                 dot_r0_c1,
        float &                 dot_r1_c0,
        float &                 dot_r1_c1) {
#if defined(TURING_MMA_AVAILABLE)
    ggml_cuda_mma::tile<16, 8, int> A;
    ggml_cuda_mma::tile<8,  8, int> B;
    ggml_cuda_mma::tile<16, 8, int> D;
    A.x[0] = a_lo_0;
    A.x[1] = a_lo_1;
    A.x[2] = a_hi_0;
    A.x[3] = a_hi_1;
    B.x[0] = smem_b8[g].qs[t];
    B.x[1] = smem_b8[g].qs[4 + t];
    D.x[0] = 0x4B400000;
    D.x[1] = 0x4B400000;
    D.x[2] = 0x4B400000;
    D.x[3] = 0x4B400000;
    ggml_cuda_mma::mma(D, A, B);
    dot_r0_c0 = __int_as_float(D.x[0]) - 12582912.0f;
    dot_r0_c1 = __int_as_float(D.x[1]) - 12582912.0f;
    dot_r1_c0 = __int_as_float(D.x[2]) - 12582912.0f;
    dot_r1_c1 = __int_as_float(D.x[3]) - 12582912.0f;
#else
    int s00 = 0, s01 = 0, s10 = 0, s11 = 0;
    const int c0 = 2 * t;
    const int c1 = 2 * t + 1;
    #pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
        const int src_lane = g * 4 + ki;
        const int alo0 = __shfl_sync(0xFFFFFFFFu, a_lo_0, src_lane, 32);
        const int ahi0 = __shfl_sync(0xFFFFFFFFu, a_hi_0, src_lane, 32);
        const int alo1 = __shfl_sync(0xFFFFFFFFu, a_lo_1, src_lane, 32);
        const int ahi1 = __shfl_sync(0xFFFFFFFFu, a_hi_1, src_lane, 32);
        s00 = ggml_cuda_dp4a(alo0, smem_b8[c0].qs[ki],     s00);
        s00 = ggml_cuda_dp4a(ahi0, smem_b8[c0].qs[4 + ki], s00);
        s01 = ggml_cuda_dp4a(alo0, smem_b8[c1].qs[ki],     s01);
        s01 = ggml_cuda_dp4a(ahi0, smem_b8[c1].qs[4 + ki], s01);
        s10 = ggml_cuda_dp4a(alo1, smem_b8[c0].qs[ki],     s10);
        s10 = ggml_cuda_dp4a(ahi1, smem_b8[c0].qs[4 + ki], s10);
        s11 = ggml_cuda_dp4a(alo1, smem_b8[c1].qs[ki],     s11);
        s11 = ggml_cuda_dp4a(ahi1, smem_b8[c1].qs[4 + ki], s11);
    }
    dot_r0_c0 = (float) s00;
    dot_r0_c1 = (float) s01;
    dot_r1_c0 = (float) s10;
    dot_r1_c1 = (float) s11;
#endif
}

template <ggml_type wtype, int N_COLS>
static __global__ void __launch_bounds__(128) moe_gate_up_swiglu_q8_kernel(
        const char *          __restrict__ w_gate,
        const char *          __restrict__ w_up,
        const moe_q8_blk *    __restrict__ x_q8,
        const int32_t *       __restrict__ ids_src1,
        const moe_work_tile * __restrict__ tiles,
        const int32_t *       __restrict__ tile_counts,
        int                                count_idx,
        moe_q8_blk *          __restrict__ h_q8,
        int                                k_blks,
        int                                n_ff,
        int                                n_ff_blks,
        int                                sis1,
        size_t                             nb_row_gu,
        size_t                             nb_exp_gu,
        ggml_glu_op                        glu_op,
        float                              glu_limit) {
    constexpr int SUB_TILES = N_COLS / 8;
    __shared__ union {
        moe_q8_blk b[N_COLS];
        float      h[N_COLS][32];
    } smem;

    const int n_tiles      = tile_counts[count_idx];
    const int total_blocks = n_tiles * n_ff_blks;
    const int tid_flat     = threadIdx.y * 32 + threadIdx.x;
    const int lane         = threadIdx.x;
    const int warp_id      = threadIdx.y;
    const int g            = lane >> 2;
    const int t            = lane & 3;

    for (int bid = blockIdx.x; bid < total_blocks; bid += gridDim.x) {
        const int tile_idx = bid / n_ff_blks;
        const int m_blk    = bid % n_ff_blks;
        const moe_work_tile tile = tiles[tile_idx];

        const int f_raw  = m_blk * 32 + warp_id * 8 + g;
        const int f_safe = min(f_raw, n_ff - 1);
        const char * gate_row = w_gate + (size_t) tile.expert * nb_exp_gu + (size_t) f_safe * nb_row_gu;
        const char * up_row   = w_up   + (size_t) tile.expert * nb_exp_gu + (size_t) f_safe * nb_row_gu;

        float acc_g[SUB_TILES][2] = {};
        float acc_u[SUB_TILES][2] = {};

        for (int kb32 = 0; kb32 < k_blks; ++kb32) {
            for (int idx = tid_flat * 2; idx < N_COLS * 10; idx += 256) {
                const int c = idx / 10;
                const int w = idx % 10;
                int2 val = { 0, 0 };
                if (c < tile.count) {
                    const int tok = ids_src1[tile.pair0 + c] / sis1;
                    val = ((const int2 *) &x_q8[(int64_t) tok * k_blks + kb32])[w >> 1];
                }
                ((int2 *) &smem.b[c])[w >> 1] = val;
            }
            __syncthreads();

            int alo_g, ahi_g, alo_u, ahi_u;
            float sc_g, zplo_g, zphi_g, sc_u, zplo_u, zphi_u;
            moe_unpack_w_blk32<wtype>(gate_row, kb32, t, alo_g, ahi_g, sc_g, zplo_g, zphi_g);
            moe_unpack_w_blk32<wtype>(up_row,   kb32, t, alo_u, ahi_u, sc_u, zplo_u, zphi_u);

            #pragma unroll
            for (int sub = 0; sub < SUB_TILES; ++sub) {
                const moe_q8_blk * b8 = &smem.b[sub * 8];
                float dg0, dg1, du0, du1;
                moe_mma_16x8x32_f32(alo_g, ahi_g, alo_u, ahi_u, b8, g, t, dg0, dg1, du0, du1);

                const moe_q8_blk & m0 = b8[2 * t + 0];
                const moe_q8_blk & m1 = b8[2 * t + 1];
                acc_g[sub][0] += m0.d * (sc_g * dg0 - zplo_g * (float) m0.sum_lo - zphi_g * (float) m0.sum_hi);
                acc_g[sub][1] += m1.d * (sc_g * dg1 - zplo_g * (float) m1.sum_lo - zphi_g * (float) m1.sum_hi);
                acc_u[sub][0] += m0.d * (sc_u * du0 - zplo_u * (float) m0.sum_lo - zphi_u * (float) m0.sum_hi);
                acc_u[sub][1] += m1.d * (sc_u * du1 - zplo_u * (float) m1.sum_lo - zphi_u * (float) m1.sum_hi);
            }
            __syncthreads();
        }

        const int feat_local = warp_id * 8 + g;
        #pragma unroll
        for (int sub = 0; sub < SUB_TILES; ++sub) {
            const int c0 = sub * 8 + 2 * t + 0;
            const int c1 = sub * 8 + 2 * t + 1;
            smem.h[c0][feat_local] = (f_raw < n_ff && c0 < tile.count)
                ? ggml_cuda_op_glu_single(glu_op, acc_g[sub][0], acc_u[sub][0], glu_limit) : 0.0f;
            smem.h[c1][feat_local] = (f_raw < n_ff && c1 < tile.count)
                ? ggml_cuda_op_glu_single(glu_op, acc_g[sub][1], acc_u[sub][1], glu_limit) : 0.0f;
        }
        __syncthreads();

        for (int c = warp_id; c < tile.count; c += 4) {
            const float v = smem.h[c][lane];
            float amax = fabsf(v);
            #pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off, 32));
            }
            const float d     = amax / 127.0f;
            const float inv_d = amax > 0.0f ? 127.0f / amax : 0.0f;
            const int   q     = max(-127, min(127, __float2int_rn(v * inv_d)));

            int half_sum = q;
            #pragma unroll
            for (int off = 8; off >= 1; off >>= 1) {
                half_sum += __shfl_xor_sync(0xFFFFFFFFu, half_sum, off, 32);
            }
            const int sum_lo = __shfl_sync(0xFFFFFFFFu, half_sum, 0,  32);
            const int sum_hi = __shfl_sync(0xFFFFFFFFu, half_sum, 16, 32);

            const int q0 = __shfl_sync(0xFFFFFFFFu, q, (lane & 7) * 4 + 0, 32);
            const int q1 = __shfl_sync(0xFFFFFFFFu, q, (lane & 7) * 4 + 1, 32);
            const int q2 = __shfl_sync(0xFFFFFFFFu, q, (lane & 7) * 4 + 2, 32);
            const int q3 = __shfl_sync(0xFFFFFFFFu, q, (lane & 7) * 4 + 3, 32);

            moe_q8_blk & out_blk = h_q8[(int64_t)(tile.pair0 + c) * n_ff_blks + m_blk];
            if (lane < 8) {
                out_blk.qs[lane] = pack4_s8(q0, q1, q2, q3);
            }
            if (lane == 0) {
                out_blk.d      = d;
                out_blk.sum_lo = (int16_t) sum_lo;
                out_blk.sum_hi = (int16_t) sum_hi;
            }
        }
        __syncthreads();
    }
}

template <ggml_type wtype, int N_COLS>
static __global__ void __launch_bounds__(128) moe_down_kernel(
        const char *          __restrict__ w_down,
        const moe_q8_blk *    __restrict__ h_q8,
        const int32_t *       __restrict__ ids_dst,
        const moe_work_tile * __restrict__ tiles,
        const int32_t *       __restrict__ tile_counts,
        int                                count_idx,
        float *               __restrict__ dst,
        int                                k_blks,
        int                                n_out,
        int                                n_out_tiles64,
        int64_t                            s_dst,
        size_t                             nb_row_dn,
        size_t                             nb_exp_dn) {
    constexpr int SUB_TILES = N_COLS / 8;
    __shared__ moe_q8_blk smem_b[N_COLS];

    const int n_tiles      = tile_counts[count_idx];
    const int total_blocks = n_tiles * n_out_tiles64;
    const int tid_flat     = threadIdx.y * 32 + threadIdx.x;
    const int lane         = threadIdx.x;
    const int warp_id      = threadIdx.y;
    const int g            = lane >> 2;
    const int t            = lane & 3;

    for (int bid = blockIdx.x; bid < total_blocks; bid += gridDim.x) {
        const int tile_idx = bid / n_out_tiles64;
        const int m_tile   = bid % n_out_tiles64;
        const moe_work_tile tile = tiles[tile_idx];

        const int r0_raw  = m_tile * 64 + warp_id * 16 + g;
        const int r1_raw  = r0_raw + 8;
        const int r0_safe = min(r0_raw, n_out - 1);
        const int r1_safe = min(r1_raw, n_out - 1);
        const char * dn_r0 = w_down + (size_t) tile.expert * nb_exp_dn + (size_t) r0_safe * nb_row_dn;
        const char * dn_r1 = w_down + (size_t) tile.expert * nb_exp_dn + (size_t) r1_safe * nb_row_dn;

        float acc0[SUB_TILES][2] = {};
        float acc1[SUB_TILES][2] = {};

        for (int kb32 = 0; kb32 < k_blks; ++kb32) {
            for (int idx = tid_flat * 2; idx < N_COLS * 10; idx += 256) {
                const int c = idx / 10;
                const int w = idx % 10;
                int2 val = { 0, 0 };
                if (c < tile.count) {
                    val = ((const int2 *) &h_q8[(int64_t)(tile.pair0 + c) * k_blks + kb32])[w >> 1];
                }
                ((int2 *) &smem_b[c])[w >> 1] = val;
            }
            __syncthreads();

            int alo0, ahi0, alo1, ahi1;
            float sc0, zplo0, zphi0, sc1, zplo1, zphi1;
            moe_unpack_w_blk32<wtype>(dn_r0, kb32, t, alo0, ahi0, sc0, zplo0, zphi0);
            moe_unpack_w_blk32<wtype>(dn_r1, kb32, t, alo1, ahi1, sc1, zplo1, zphi1);

            #pragma unroll
            for (int sub = 0; sub < SUB_TILES; ++sub) {
                const moe_q8_blk * b8 = &smem_b[sub * 8];
                float d00, d01, d10, d11;
                moe_mma_16x8x32_f32(alo0, ahi0, alo1, ahi1, b8, g, t, d00, d01, d10, d11);

                const moe_q8_blk & m0 = b8[2 * t + 0];
                const moe_q8_blk & m1 = b8[2 * t + 1];
                acc0[sub][0] += m0.d * (sc0 * d00 - zplo0 * (float) m0.sum_lo - zphi0 * (float) m0.sum_hi);
                acc0[sub][1] += m1.d * (sc0 * d01 - zplo0 * (float) m1.sum_lo - zphi0 * (float) m1.sum_hi);
                acc1[sub][0] += m0.d * (sc1 * d10 - zplo1 * (float) m0.sum_lo - zphi1 * (float) m0.sum_hi);
                acc1[sub][1] += m1.d * (sc1 * d11 - zplo1 * (float) m1.sum_lo - zphi1 * (float) m1.sum_hi);
            }
            __syncthreads();
        }

        #pragma unroll
        for (int sub = 0; sub < SUB_TILES; ++sub) {
            const int c0 = sub * 8 + 2 * t + 0;
            const int c1 = sub * 8 + 2 * t + 1;
            if (c0 < tile.count) {
                const int64_t col_off = (int64_t) ids_dst[tile.pair0 + c0] * s_dst;
                if (r0_raw < n_out) { dst[col_off + r0_raw] = acc0[sub][0]; }
                if (r1_raw < n_out) { dst[col_off + r1_raw] = acc1[sub][0]; }
            }
            if (c1 < tile.count) {
                const int64_t col_off = (int64_t) ids_dst[tile.pair0 + c1] * s_dst;
                if (r0_raw < n_out) { dst[col_off + r0_raw] = acc0[sub][1]; }
                if (r1_raw < n_out) { dst[col_off + r1_raw] = acc1[sub][1]; }
            }
        }
    }
}

static bool moe_wtype_supported(ggml_type t) {
    return t == GGML_TYPE_Q3_K   || t == GGML_TYPE_Q4_0   || t == GGML_TYPE_Q2_0 ||
           t == GGML_TYPE_Q8_0   || t == GGML_TYPE_Q4_K   || t == GGML_TYPE_Q5_K ||
           t == GGML_TYPE_IQ4_NL || t == GGML_TYPE_MXFP4;
}

template <ggml_type wtype>
static void launch_gu_typed(
        int grid_d, int grid_s, cudaStream_t stream,
        const char * wg, const char * wu, const moe_q8_blk * xq, const int32_t * ids_src1,
        const moe_work_tile * dt, const moe_work_tile * st, const int32_t * tc, moe_q8_blk * hq,
        int k_blks, int n_ff, int n_ff_blks, int sis1, size_t nbr, size_t nbe, ggml_glu_op op, float lim) {
    const dim3 blk(32, 4, 1);
    if (grid_d > 0) {
        moe_gate_up_swiglu_q8_kernel<wtype, 32><<<grid_d, blk, 0, stream>>>(
            wg, wu, xq, ids_src1, dt, tc, 0, hq, k_blks, n_ff, n_ff_blks, sis1, nbr, nbe, op, lim);
    }
    if (grid_s > 0) {
        moe_gate_up_swiglu_q8_kernel<wtype, 8><<<grid_s, blk, 0, stream>>>(
            wg, wu, xq, ids_src1, st, tc, 1, hq, k_blks, n_ff, n_ff_blks, sis1, nbr, nbe, op, lim);
    }
}

template <ggml_type wtype>
static void launch_dn_typed(
        int grid_d, int grid_s, cudaStream_t stream,
        const char * wd, const moe_q8_blk * hq, const int32_t * ids_dst,
        const moe_work_tile * dt, const moe_work_tile * st, const int32_t * tc, float * dst,
        int k_blks, int n_out, int n_out_t64, int64_t s_dst, size_t nbr, size_t nbe) {
    const dim3 blk(32, 4, 1);
    if (grid_d > 0) {
        moe_down_kernel<wtype, 32><<<grid_d, blk, 0, stream>>>(
            wd, hq, ids_dst, dt, tc, 0, dst, k_blks, n_out, n_out_t64, s_dst, nbr, nbe);
    }
    if (grid_s > 0) {
        moe_down_kernel<wtype, 8><<<grid_s, blk, 0, stream>>>(
            wd, hq, ids_dst, st, tc, 1, dst, k_blks, n_out, n_out_t64, s_dst, nbr, nbe);
    }
}

bool ggml_cuda_moe_fused_prefill_supported(
        const ggml_tensor * up,
        const ggml_tensor * gate,
        const ggml_tensor * glu,
        const ggml_tensor * down,
        int                 device) {
    if (!ggml_cuda_moe_fused_prefill_enabled()) {
        return false;
    }
    if (device < 0 || device >= ggml_cuda_info().device_count || ggml_cuda_info().devices[device].warp_size != 32) {
        return false;
    }
    if (!up || !gate || !glu || !down) {
        return false;
    }
    if (up->op != GGML_OP_MUL_MAT_ID || gate->op != GGML_OP_MUL_MAT_ID ||
        glu->op != GGML_OP_GLU       || down->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    const ggml_tensor * wu  = up->src[0];
    const ggml_tensor * wg  = gate->src[0];
    const ggml_tensor * wd  = down->src[0];
    const ggml_tensor * x   = up->src[1];
    const ggml_tensor * ids = up->src[2];

    if (gate->src[1] != x || gate->src[2] != ids || down->src[1] != glu || down->src[2] != ids) {
        return false;
    }
    if (glu->src[0] != gate || glu->src[1] != up || ggml_get_op_params_i32(glu, 1) != 0) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || down->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (x->ne[1] != 1 || x->ne[3] != 1 || x->ne[2] < 8 || (x->ne[0] % 32) != 0 || (glu->ne[0] % 32) != 0) {
        return false;
    }
    if (x->nb[0] != sizeof(float) || down->nb[0] != sizeof(float) || ids->nb[0] != sizeof(int32_t)) {
        return false;
    }
    if ((x->nb[2] % sizeof(float)) != 0 || (down->nb[1] % sizeof(float)) != 0 || (down->nb[2] % down->nb[1]) != 0) {
        return false;
    }
    if (!moe_wtype_supported(wu->type) || wu->type != wg->type || !moe_wtype_supported(wd->type)) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    if (blackwell_mma_available(cc) && (wu->type == GGML_TYPE_MXFP4 || wd->type == GGML_TYPE_MXFP4)) {
        return false;   // Blackwell native FP4 MMQ beats INT8 LUT unpack
    }
    if (ampere_mma_available(cc) && x->ne[2] > 64) {
        return false;   // Ampere+ cp.async pipelined MMQ + zero-tail quant skip handles large prefill batches
    }
    if (!ggml_are_same_shape(wu, wg) || !ggml_are_same_stride(wu, wg) || wu->ne[2] != wd->ne[2] || wu->ne[3] != 1 || wd->ne[3] != 1) {
        return false;
    }
    if (ggml_get_op_params_i32(up, 1) != GGML_HINT_NONE ||
        ggml_get_op_params_i32(gate, 1) != GGML_HINT_NONE ||
        ggml_get_op_params_i32(down, 1) != GGML_HINT_NONE) {
        return false;
    }
    const ggml_glu_op op = ggml_get_glu_op(glu);
    if (op != GGML_GLU_OP_SWIGLU && op != GGML_GLU_OP_GEGLU &&
        op != GGML_GLU_OP_SWIGLU_OAI && op != GGML_GLU_OP_SWIGLU_CLAMP) {
        return false;
    }
    if (op == GGML_GLU_OP_SWIGLU_OAI && (ggml_get_op_params_f32(glu, 2) != 1.702f || ggml_get_op_params_f32(glu, 3) != 7.0f)) {
        return false;
    }
    return true;
}

void ggml_cuda_moe_fused_prefill(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         up,
        const ggml_tensor *         gate,
        const ggml_tensor *         glu,
        ggml_tensor *               down,
        void (*mark_after_gu)(void *),
        void *                      mark_ud) {
    cudaStream_t stream = ctx.stream();
    const int dev = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[dev].nsm;

    const ggml_tensor * wu  = up->src[0];
    const ggml_tensor * wg  = gate->src[0];
    const ggml_tensor * wd  = down->src[0];
    const ggml_tensor * x   = up->src[1];
    const ggml_tensor * ids = up->src[2];

    const int K_in      = (int) x->ne[0];
    const int nt        = (int) x->ne[2];
    const int k_used    = (int) ids->ne[0];
    const int n_experts = (int) wu->ne[2];
    const int n_ff      = (int) wu->ne[1];
    const int n_out     = (int) wd->ne[1];
    const int n_pairs   = nt * k_used;
    const int k_blks    = K_in / 32;
    const int n_ff_blks = n_ff / 32;
    const int n_out_t64 = (n_out + 63) / 64;

    const int moe_nz   = ggml_cuda_moe_zero_tail_of(down);
    const int act_exps = (moe_nz > 0 && moe_nz < n_experts) ? (n_experts - moe_nz) : n_experts;

    ggml_cuda_pool_alloc<moe_q8_blk>    x_q8(ctx.pool(), (size_t) nt * k_blks);
    ggml_cuda_pool_alloc<moe_q8_blk>    h_q8(ctx.pool(), (size_t) n_pairs * n_ff_blks);
    ggml_cuda_pool_alloc<int32_t>       ids_src1(ctx.pool(), n_pairs);
    ggml_cuda_pool_alloc<int32_t>       ids_dst(ctx.pool(), n_pairs + 32);
    ggml_cuda_pool_alloc<int32_t>       expert_bounds(ctx.pool(), n_experts + 1);
    ggml_cuda_pool_alloc<moe_work_tile> dense_tiles(ctx.pool(), (n_pairs + 16) / 17 + 1);
    ggml_cuda_pool_alloc<moe_work_tile> sparse_tiles(ctx.pool(), n_pairs + 1);
    ggml_cuda_pool_alloc<int32_t>       tile_counts(ctx.pool(), 2);

    const int q_warps = nt * k_blks;
    moe_quantize_x_kernel<<<(q_warps + 7) / 8, 256, 0, stream>>>(
        (const float *) x->data, x_q8.get(), k_blks, nt, (int64_t)(x->nb[2] / sizeof(float)));

    const int si1  = (int)(ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int)(x->nb[2] / x->nb[1]);
    ggml_cuda_launch_mm_ids_helper(
        (const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, nt, k_used, /*ne11=*/1, si1, sis1, /*write_inverse=*/false, stream);

    const int64_t s_dst = (int64_t)(down->nb[1] / sizeof(float));
    if (act_exps < n_experts) {
        ggml_cuda_moe_zero_tail_apply(
            expert_bounds.get(), ids_dst.get(), (float *) down->data,
            n_experts, moe_nz, n_out, s_dst, n_pairs, stream);
    }

    moe_plan_dual_tiles_kernel<<<1, 32, 0, stream>>>(
        expert_bounds.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), act_exps);

    const int max_d_tiles = (n_pairs + 16) / 17;
    const int max_s_tiles = std::min(n_pairs, 2 * act_exps);
    const int grid_gu_d   = std::min(max_d_tiles * n_ff_blks, nsm * 8);
    const int grid_gu_s   = std::min(max_s_tiles * n_ff_blks, nsm * 8);
    const int grid_dn_d   = std::min(max_d_tiles * n_out_t64, nsm * 8);
    const int grid_dn_s   = std::min(max_s_tiles * n_out_t64, nsm * 8);

    const ggml_glu_op glu_op    = ggml_get_glu_op(glu);
    const float       glu_limit = ggml_get_op_params_f32(glu, 3);

    switch (wu->type) {
        case GGML_TYPE_Q3_K:   launch_gu_typed<GGML_TYPE_Q3_K>  (grid_gu_d, grid_gu_s, stream, (const char *)wg->data, (const char *)wu->data, x_q8.get(), ids_src1.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), h_q8.get(), k_blks, n_ff, n_ff_blks, sis1, wu->nb[1], wu->nb[2], glu_op, glu_limit); break;
        case GGML_TYPE_Q4_0:   launch_gu_typed<GGML_TYPE_Q4_0>  (grid_gu_d, grid_gu_s, stream, (const char *)wg->data, (const char *)wu->data, x_q8.get(), ids_src1.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), h_q8.get(), k_blks, n_ff, n_ff_blks, sis1, wu->nb[1], wu->nb[2], glu_op, glu_limit); break;
        case GGML_TYPE_Q2_0:   launch_gu_typed<GGML_TYPE_Q2_0>  (grid_gu_d, grid_gu_s, stream, (const char *)wg->data, (const char *)wu->data, x_q8.get(), ids_src1.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), h_q8.get(), k_blks, n_ff, n_ff_blks, sis1, wu->nb[1], wu->nb[2], glu_op, glu_limit); break;
        case GGML_TYPE_Q8_0:   launch_gu_typed<GGML_TYPE_Q8_0>  (grid_gu_d, grid_gu_s, stream, (const char *)wg->data, (const char *)wu->data, x_q8.get(), ids_src1.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), h_q8.get(), k_blks, n_ff, n_ff_blks, sis1, wu->nb[1], wu->nb[2], glu_op, glu_limit); break;
        case GGML_TYPE_Q4_K:   launch_gu_typed<GGML_TYPE_Q4_K>  (grid_gu_d, grid_gu_s, stream, (const char *)wg->data, (const char *)wu->data, x_q8.get(), ids_src1.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), h_q8.get(), k_blks, n_ff, n_ff_blks, sis1, wu->nb[1], wu->nb[2], glu_op, glu_limit); break;
        case GGML_TYPE_Q5_K:   launch_gu_typed<GGML_TYPE_Q5_K>  (grid_gu_d, grid_gu_s, stream, (const char *)wg->data, (const char *)wu->data, x_q8.get(), ids_src1.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), h_q8.get(), k_blks, n_ff, n_ff_blks, sis1, wu->nb[1], wu->nb[2], glu_op, glu_limit); break;
        case GGML_TYPE_IQ4_NL: launch_gu_typed<GGML_TYPE_IQ4_NL>(grid_gu_d, grid_gu_s, stream, (const char *)wg->data, (const char *)wu->data, x_q8.get(), ids_src1.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), h_q8.get(), k_blks, n_ff, n_ff_blks, sis1, wu->nb[1], wu->nb[2], glu_op, glu_limit); break;
        case GGML_TYPE_MXFP4:  launch_gu_typed<GGML_TYPE_MXFP4> (grid_gu_d, grid_gu_s, stream, (const char *)wg->data, (const char *)wu->data, x_q8.get(), ids_src1.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), h_q8.get(), k_blks, n_ff, n_ff_blks, sis1, wu->nb[1], wu->nb[2], glu_op, glu_limit); break;
        default: GGML_ABORT("unsupported gate/up quant type in moe_fused_prefill");
    }

    if (mark_after_gu) {
        mark_after_gu(mark_ud);
    }

    switch (wd->type) {
        case GGML_TYPE_Q3_K:   launch_dn_typed<GGML_TYPE_Q3_K>  (grid_dn_d, grid_dn_s, stream, (const char *)wd->data, h_q8.get(), ids_dst.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), (float *)down->data, n_ff_blks, n_out, n_out_t64, s_dst, wd->nb[1], wd->nb[2]); break;
        case GGML_TYPE_Q4_0:   launch_dn_typed<GGML_TYPE_Q4_0>  (grid_dn_d, grid_dn_s, stream, (const char *)wd->data, h_q8.get(), ids_dst.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), (float *)down->data, n_ff_blks, n_out, n_out_t64, s_dst, wd->nb[1], wd->nb[2]); break;
        case GGML_TYPE_Q2_0:   launch_dn_typed<GGML_TYPE_Q2_0>  (grid_dn_d, grid_dn_s, stream, (const char *)wd->data, h_q8.get(), ids_dst.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), (float *)down->data, n_ff_blks, n_out, n_out_t64, s_dst, wd->nb[1], wd->nb[2]); break;
        case GGML_TYPE_Q8_0:   launch_dn_typed<GGML_TYPE_Q8_0>  (grid_dn_d, grid_dn_s, stream, (const char *)wd->data, h_q8.get(), ids_dst.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), (float *)down->data, n_ff_blks, n_out, n_out_t64, s_dst, wd->nb[1], wd->nb[2]); break;
        case GGML_TYPE_Q4_K:   launch_dn_typed<GGML_TYPE_Q4_K>  (grid_dn_d, grid_dn_s, stream, (const char *)wd->data, h_q8.get(), ids_dst.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), (float *)down->data, n_ff_blks, n_out, n_out_t64, s_dst, wd->nb[1], wd->nb[2]); break;
        case GGML_TYPE_Q5_K:   launch_dn_typed<GGML_TYPE_Q5_K>  (grid_dn_d, grid_dn_s, stream, (const char *)wd->data, h_q8.get(), ids_dst.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), (float *)down->data, n_ff_blks, n_out, n_out_t64, s_dst, wd->nb[1], wd->nb[2]); break;
        case GGML_TYPE_IQ4_NL: launch_dn_typed<GGML_TYPE_IQ4_NL>(grid_dn_d, grid_dn_s, stream, (const char *)wd->data, h_q8.get(), ids_dst.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), (float *)down->data, n_ff_blks, n_out, n_out_t64, s_dst, wd->nb[1], wd->nb[2]); break;
        case GGML_TYPE_MXFP4:  launch_dn_typed<GGML_TYPE_MXFP4> (grid_dn_d, grid_dn_s, stream, (const char *)wd->data, h_q8.get(), ids_dst.get(), dense_tiles.get(), sparse_tiles.get(), tile_counts.get(), (float *)down->data, n_ff_blks, n_out, n_out_t64, s_dst, wd->nb[1], wd->nb[2]); break;
        default: GGML_ABORT("unsupported down quant type in moe_fused_prefill");
    }
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void moe_weighted_sum_f32_kernel(
        const float * __restrict__ experts,
        const float * __restrict__ weights,
        float *       __restrict__ dst,
        int                        n_embd,
        int                        k_used,
        int                        nt,
        int64_t                    se1,
        int64_t                    se2,
        int64_t                    sw1,
        int64_t                    sw2,
        int64_t                    sd1) {
    const int64_t idx   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t) n_embd * nt;
    if (idx >= total) {
        return;
    }
    const int i   = (int)(idx % n_embd);
    const int tok = (int)(idx / n_embd);

    const float * exp_base = experts + (int64_t) tok * se2 + i;
    const float * w_base   = weights + (int64_t) tok * sw2;

    float sum = 0.0f;
    for (int j = 0; j < k_used; ++j) {
        sum += exp_base[(int64_t) j * se1] * w_base[(int64_t) j * sw1];
    }
    dst[(int64_t) tok * sd1 + i] = sum;
}

bool ggml_cuda_moe_weighted_sum_supported(
        const ggml_cgraph * cgraph,
        int                 node_idx,
        int *               out_n_nodes) {
    if (!ggml_cuda_moe_fused_prefill_enabled() || !cgraph || node_idx < 0 || node_idx >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * mul = cgraph->nodes[node_idx];
    if (!mul || mul->op != GGML_OP_MUL) {
        return false;
    }
    const ggml_tensor * experts = mul->src[0];
    const ggml_tensor * weights = mul->src[1];
    if (!experts || !weights || experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        return false;
    }
    if (experts->ne[3] != 1 || weights->ne[0] != 1 || weights->ne[1] != experts->ne[1] ||
        weights->ne[2] != experts->ne[2] || weights->ne[3] != 1) {
        return false;
    }
    if (experts->nb[0] != sizeof(float) || (experts->nb[1] % sizeof(float)) != 0 || (experts->nb[2] % sizeof(float)) != 0 ||
        (weights->nb[1] % sizeof(float)) != 0 || (weights->nb[2] % sizeof(float)) != 0) {
        return false;
    }

    const int k_max = (int) experts->ne[1];
    if (k_max < 2 || k_max > 64) {
        return false;
    }
    // Count how many consecutive VIEW(mul, slice i) nodes follow mul
    int k = 0;
    while (k < k_max && node_idx + 1 + k < cgraph->n_nodes) {
        const ggml_tensor * v = cgraph->nodes[node_idx + 1 + k];
        if (!v || v->op != GGML_OP_VIEW || v->src[0] != mul) {
            break;
        }
        if (v->ne[0] != experts->ne[0] || v->ne[1] != experts->ne[2] || v->ne[2] != 1 || v->ne[3] != 1) {
            return false;
        }
        if (v->nb[1] != mul->nb[2] || (size_t) ggml_get_op_params_i32(v, 0) != (size_t) k * mul->nb[1]) {
            return false;
        }
        const size_t h_v = ggml_hash_find(&cgraph->visited_hash_set, v);
        if (h_v == GGML_HASHSET_FULL || cgraph->use_counts[h_v] != 1) {
            return false;
        }
        ++k;
    }
    if (k < 2 || node_idx + 2 * k - 1 >= cgraph->n_nodes) {
        return false;
    }
    const size_t h_mul = ggml_hash_find(&cgraph->visited_hash_set, mul);
    if (h_mul == GGML_HASHSET_FULL || cgraph->use_counts[h_mul] != k) {
        return false;
    }

    const ggml_tensor * prev = cgraph->nodes[node_idx + 1];
    for (int j = 1; j < k; ++j) {
        const ggml_tensor * add = cgraph->nodes[node_idx + k + j];
        const ggml_tensor * vj  = cgraph->nodes[node_idx + 1 + j];
        if (!add || add->op != GGML_OP_ADD || add->src[0] != prev || add->src[1] != vj || add->type != GGML_TYPE_F32) {
            return false;
        }
        if (j + 1 < k) {
            const size_t h_add = ggml_hash_find(&cgraph->visited_hash_set, add);
            if (h_add == GGML_HASHSET_FULL || cgraph->use_counts[h_add] != 1) {
                return false;
            }
        }
        prev = add;
    }
    if (prev->nb[0] != sizeof(float) || (prev->nb[1] % sizeof(float)) != 0) {
        return false;
    }
    if (out_n_nodes) {
        *out_n_nodes = 2 * k;
    }
    return true;
}

void ggml_cuda_moe_weighted_sum(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         experts,
        const ggml_tensor *         weights,
        ggml_tensor *               dst,
        int                         n_expert_used) {
    const int     n_embd = (int) dst->ne[0];
    const int     nt     = (int) dst->ne[1];
    const int64_t total  = (int64_t) n_embd * nt;
    const int     blk    = 256;
    const int     grid   = (int)((total + blk - 1) / blk);

    moe_weighted_sum_f32_kernel<<<grid, blk, 0, ctx.stream()>>>(
        (const float *) experts->data,
        (const float *) weights->data,
        (float *)       dst->data,
        n_embd,
        n_expert_used,
        nt,
        (int64_t)(experts->nb[1] / sizeof(float)),
        (int64_t)(experts->nb[2] / sizeof(float)),
        (int64_t)(weights->nb[1] / sizeof(float)),
        (int64_t)(weights->nb[2] / sizeof(float)),
        (int64_t)(dst->nb[1]     / sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
}

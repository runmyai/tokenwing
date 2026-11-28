#pragma once
#include "wv_common.cuh"

namespace wv {

static constexpr int      INK2_HIDDEN          = 2304;
static constexpr int      INK2_INTER           = 1152;
static constexpr int      INK2_ATTN_DIM        = 3072; // 24 * 128
static constexpr int      INK2_KV_DIM          = 1024; // 8 * 128
static constexpr int      INK2_REL_DIM         = 384;  // 24 * 16
static constexpr int      INK2_SWA_WIN         = 1024;
static constexpr int      INK2_SWA_RING        = 1040; // >= 1024 + 16: zero-copy K=8 speculative safety
static constexpr int      INK2_NUM_LAYERS_MAX  = 32;
static constexpr int      INK2_NUM_VISITS_MAX  = 64;
static constexpr int      INK2_NUM_EXPERTS_MAX = 128;
static constexpr int      INK2_TOP_K_MAX       = 8;
static constexpr int      INK2_MTP_LAYERS_MAX  = 4;
static constexpr uint32_t INK2_COLD_SLAB_BYTES = 2239488u;
static constexpr uint32_t INK2_OFF_GU_PK       = 0u;
static constexpr uint32_t INK2_OFF_GU_SC       = 1327104u;
static constexpr uint32_t INK2_OFF_DN_PK       = 1492992u;
static constexpr uint32_t INK2_OFF_DN_SC       = 2156544u;

static __constant__ uint4    c_ink_lut4_u128[8]   = WV_DEF_LUT4_INIT;
static __constant__ uint32_t c_ink_vq8_uv[2][512] = WV_DEF_VQ8_INIT;

struct WvInkSwz8Linear {
    const uint2*  pk = nullptr; // [N/8, K/32, 32] uint2 (int32[2])
    const __half* sc = nullptr; // [N/8, K/32, 8] fp16 (scale * 127.0)
};

__device__ __forceinline__ int32_t wv_ink_unpack_wv6_4b38_vec4(
    const uint8_t* row_bytes, int b128, int lane, float& out_ws) {
    const uint8_t* bp = row_bytes + static_cast<size_t>(b128) * 70;
    uint16_t d_sup = *reinterpret_cast<const uint16_t*>(bp);
    uint32_t m     = wv_ld_u32_align2(bp + 2);
    const uint8_t* qs_nib = bp + 6;
    int p    = lane >> 3;
    int pos0 = 4 * lane;
    uint32_t u6 = (m >> (6 * p)) & 0x3Fu;
    out_ws = wv_fp16_to_f32(d_sup) * (WV_U6_NORM * static_cast<float>(u6));
    uint4 lut16 = c_ink_lut4_u128[(m >> 24) & 7u];
    uint32_t nib4 = (pos0 < 64)
        ? (wv_ld_u32_align2(qs_nib + pos0) & 0x0F0F0F0Fu)
        : ((wv_ld_u32_align2(qs_nib + pos0 - 64) >> 4) & 0x0F0F0F0Fu);
    return wv_prmt16_lut4(lut16, nib4);
}

// Gather WV6_4B38 [vocab_full, 18*70=1260] -> FWHT-128 -> BF16 -> RMSNorm(embed_norm) -> BF16 [M, 2304]
// Grid <<<M, dim3(32, 6)>>>
__global__ void wv_ink_embed_gather_norm_kernel(
    const int32_t* __restrict__ tok_ids,
    const uint8_t* __restrict__ emb_wire,
    const float* __restrict__ embed_norm,
    __nv_bfloat16* __restrict__ h_out,
    int M,
    int vocab_full) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m >= M) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x;
    int wid  = threadIdx.y; // 0..5
    int tok  = tok_ids[m];
    if (tok < 0 || tok >= vocab_full) tok = 0;
    const uint8_t* row_p = emb_wire + static_cast<size_t>(tok) * (18 * 70);
    __nv_bfloat16* dst_m = h_out + static_cast<size_t>(m) * INK2_HIDDEN;

    __shared__ float s_wsum[6];
    float reg_e[3][4];
    float p_sq = 0.0f;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int b128 = wid + it * 6;
        float ws = 0.0f;
        int32_t w4 = wv_ink_unpack_wv6_4b38_vec4(row_p, b128, lane, ws);
        float f0 = static_cast<float>(static_cast<int8_t>(w4 & 0xFF)) * ws;
        float f1 = static_cast<float>(static_cast<int8_t>((w4 >> 8) & 0xFF)) * ws;
        float f2 = static_cast<float>(static_cast<int8_t>((w4 >> 16) & 0xFF)) * ws;
        float f3 = static_cast<float>(static_cast<int8_t>((w4 >> 24) & 0xFF)) * ws;
        wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
        f0 = __bfloat162float(__float2bfloat16(f0));
        f1 = __bfloat162float(__float2bfloat16(f1));
        f2 = __bfloat162float(__float2bfloat16(f2));
        f3 = __bfloat162float(__float2bfloat16(f3));
        reg_e[it][0] = f0; reg_e[it][1] = f1; reg_e[it][2] = f2; reg_e[it][3] = f3;
        p_sq += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
    }
    p_sq = wv_warp_sum_f32(p_sq);
    if (lane == 0) s_wsum[wid] = p_sq;
    __syncthreads();
    float inv_rms = rsqrtf((s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3] + s_wsum[4] + s_wsum[5]) * (1.0f / 2304.0f) + 1e-6f);
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int b128 = wid + it * 6;
        int base = b128 * 128 + lane * 4;
        float4 nw = reinterpret_cast<const float4*>(embed_norm + base)[0];
        dst_m[base + 0] = __float2bfloat16(reg_e[it][0] * inv_rms * nw.x);
        dst_m[base + 1] = __float2bfloat16(reg_e[it][1] * inv_rms * nw.y);
        dst_m[base + 2] = __float2bfloat16(reg_e[it][2] * inv_rms * nw.z);
        dst_m[base + 3] = __float2bfloat16(reg_e[it][3] * inv_rms * nw.w);
    }
    wv_pdl_trigger();
}

// Scatter post-merger BF16 vision embeddings [N_vis, 2304] into cur_h [M, 2304] after embed_gather_norm:
__global__ void wv_ink_scatter_vision_embeds_kernel(
    const int32_t* __restrict__ vis_idx,
    const __nv_bfloat16* __restrict__ vis_emb,
    __nv_bfloat16* __restrict__ h_out,
    int M, int N_vis) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m < M) {
        int vi = vis_idx[m];
        if (vi >= 0 && vi < N_vis) {
            const uint4* src = reinterpret_cast<const uint4*>(vis_emb + static_cast<size_t>(vi) * INK2_HIDDEN);
            uint4*       dst = reinterpret_cast<uint4*>(h_out   + static_cast<size_t>(m)  * INK2_HIDDEN);
            for (int q = threadIdx.x; q < 288; q += blockDim.x) {
                dst[q] = src[q];
            }
        }
    }
    wv_pdl_trigger();
}

// Generic BF16 [M, K] (K % 128 == 0) -> FWHT-128 + Q8_32: Grid <<<(total_blks128 + 3) / 4, dim3(32, 4)>>>
__global__ void wv_ink_bf16_fwht128_q8_kernel(
    const __nv_bfloat16* __restrict__ x_bf16,
    int8_t* __restrict__ X_q8,
    float* __restrict__ X_sc,
    int total_blks128) {
    wv_pdl_wait();
    int blk_id = blockIdx.x * blockDim.y + threadIdx.y;
    int lane   = threadIdx.x;
    if (blk_id >= total_blks128) { wv_pdl_trigger(); return; }
    size_t off = static_cast<size_t>(blk_id) * 128 + lane * 4;
    float f0 = __bfloat162float(x_bf16[off + 0]);
    float f1 = __bfloat162float(x_bf16[off + 1]);
    float f2 = __bfloat162float(x_bf16[off + 2]);
    float f3 = __bfloat162float(x_bf16[off + 3]);
    wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
    wv_quant_q8_32_warp(f0, f1, f2, f3, lane, X_q8 + static_cast<size_t>(blk_id) * 128, X_sc + blk_id * 4);
    wv_pdl_trigger();
}

// Fused FP32 [M, 2304] -> RMSNorm(w_norm) -> BF16 y_bf16 + FWHT-128 + Q8_32 (Y_q8, Y_sc):
// Grid <<<M, dim3(32, 6)>>>
__global__ void wv_ink_f32_rmsnorm_bf16_fwht_q8_kernel(
    const float* __restrict__ x_f32,
    const float* __restrict__ w_norm,
    __nv_bfloat16* __restrict__ y_bf16,
    int8_t* __restrict__ Y_q8,
    float* __restrict__ Y_sc,
    int M) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m >= M) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x, wid = threadIdx.y;
    const float* xm = x_f32 + static_cast<size_t>(m) * INK2_HIDDEN;
    __shared__ float s_wsum[6];
    float4 reg_x[3];
    float p_sq = 0.0f;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int base = (wid + it * 6) * 128 + lane * 4;
        float4 xv = reinterpret_cast<const float4*>(xm + base)[0];
        reg_x[it] = xv;
        p_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    p_sq = wv_warp_sum_f32(p_sq);
    if (lane == 0) s_wsum[wid] = p_sq;
    __syncthreads();
    float inv_rms = rsqrtf((s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3] + s_wsum[4] + s_wsum[5]) * (1.0f / 2304.0f) + 1e-6f);

    __nv_bfloat16* yb = (y_bf16 != nullptr) ? (y_bf16 + static_cast<size_t>(m) * INK2_HIDDEN) : nullptr;
    int8_t* q8_m      = (Y_q8 != nullptr)   ? (Y_q8 + static_cast<size_t>(m) * INK2_HIDDEN)   : nullptr;
    float*  sc_m      = (Y_sc != nullptr)   ? (Y_sc + static_cast<size_t>(m) * 72)            : nullptr;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int b128 = wid + it * 6;
        int base = b128 * 128 + lane * 4;
        float4 xv = reg_x[it];
        float4 nw = __ldg(reinterpret_cast<const float4*>(w_norm + base));
        __nv_bfloat16 b0 = __float2bfloat16(xv.x * inv_rms * nw.x);
        __nv_bfloat16 b1 = __float2bfloat16(xv.y * inv_rms * nw.y);
        __nv_bfloat16 b2 = __float2bfloat16(xv.z * inv_rms * nw.z);
        __nv_bfloat16 b3 = __float2bfloat16(xv.w * inv_rms * nw.w);
        if (yb != nullptr) {
            yb[base + 0] = b0; yb[base + 1] = b1; yb[base + 2] = b2; yb[base + 3] = b3;
        }
        if (q8_m != nullptr && sc_m != nullptr) {
            float f0 = __bfloat162float(b0), f1 = __bfloat162float(b1);
            float f2 = __bfloat162float(b2), f3 = __bfloat162float(b3);
            wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
            wv_quant_q8_32_warp(f0, f1, f2, f3, lane, q8_m + b128 * 128, sc_m + b128 * 4);
        }
    }
    wv_pdl_trigger();
}

// Warp-coalesced Tensor Core mma.m16n8k32 SWZ8_G32 Linear (K % 128 == 0, N % 8 == 0):
// M_REGIME: 2 -> CTA_M = 8 (4..8), 0 -> CTA_M = 16 (9..48), 1 -> CTA_M = 64 (>48)
template<int M_REGIME, bool OUT_BF16, bool SOFTCAP30, bool STREAM_W = false>
__global__ __launch_bounds__(128, (M_REGIME == 1) ? 2 : 6)
void wv_ink_swz8_mma_linear_kernel(
    const uint2* __restrict__ w_pk,   // [N/8, K/32, 32] uint2
    const __half* __restrict__ w_sc,  // [N/8, K/32, 8] fp16
    const float* __restrict__ bias,   // [N] or nullptr
    const int8_t* __restrict__ X_q8,  // [M, K]
    const float* __restrict__ X_sc,   // [M, K/32]
    void* __restrict__ dst,
    int K, int N, int M) {
    constexpr int CTA_M     = (M_REGIME == 2) ? 8 : ((M_REGIME == 0) ? 16 : 64);
    constexpr int SM_STRIDE = 68; // 256 bytes = 64 uint32 + 4 bank-conflict pad
    constexpr int N_TILES16 = (M_REGIME == 2) ? 1 : (CTA_M / 16);

    __shared__ uint32_t s_X[CTA_M][SM_STRIDE];
    __shared__ float    s_Xsc[CTA_M][8];

    wv_pdl_wait();
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int lane_n  = lane >> 2;
    int c_k     = lane & 3;
    int r0      = lane >> 2;
    int r1      = r0 + 8;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;
    int n_tile8 = blockIdx.x * 4 + warp_id;
    int out_n0  = n_tile8 * 8 + col_n0;
    int m_base  = blockIdx.y * CTA_M;
    int cur_m   = min(CTA_M, M - m_base);
    if (cur_m <= 0) { wv_pdl_trigger(); return; }

    int n_kc128   = K >> 7;
    int n_kc32    = K >> 5;
    bool active_n = (n_tile8 * 8 < N);
    const uint2*  pk_base = active_n ? (w_pk + static_cast<size_t>(n_tile8) * n_kc32 * 32 + lane) : nullptr;
    const __half* sc_base = active_n ? (w_sc + static_cast<size_t>(n_tile8) * n_kc32 * 8 + lane_n) : nullptr;

    float acc[N_TILES16][4];
    #pragma unroll
    for (int t = 0; t < N_TILES16; ++t) {
        acc[t][0] = 0.f; acc[t][1] = 0.f; acc[t][2] = 0.f; acc[t][3] = 0.f;
    }
    constexpr float kInv127 = 1.0f / 127.0f;

    for (int kc256 = 0; kc256 < n_kc128; kc256 += 2) {
        int has_c1 = (kc256 + 1 < n_kc128) ? 1 : 0;
        constexpr int LD_ITERS = (CTA_M * 16) / 128;
        #pragma unroll
        for (int u = 0; u < LD_ITERS; ++u) {
            int idx  = u * 128 + threadIdx.x;
            int ld_m = idx >> 4;
            int ld_q = idx & 15; // 0..15 -> 16 * 16B = 256B
            bool ok_k = (ld_q < 8) || (has_c1 != 0);
            uint4 xv = (ld_m < cur_m && ok_k)
                ? reinterpret_cast<const uint4*>(X_q8 + static_cast<size_t>(m_base + ld_m) * K + kc256 * 128)[ld_q]
                : make_uint4(0, 0, 0, 0);
            int c4 = ld_q * 4;
            s_X[ld_m][c4 + 0] = xv.x; s_X[ld_m][c4 + 1] = xv.y;
            s_X[ld_m][c4 + 2] = xv.z; s_X[ld_m][c4 + 3] = xv.w;
        }
        if (threadIdx.x < CTA_M * 2) {
            int ld_m  = threadIdx.x >> 1;
            int ld_c  = threadIdx.x & 1;
            bool ok_c = (ld_c == 0) || (has_c1 != 0);
            float4 xs = (ld_m < cur_m && ok_c)
                ? reinterpret_cast<const float4*>(X_sc + static_cast<size_t>(m_base + ld_m) * n_kc32 + kc256 * 4 + ld_c * 4)[0]
                : make_float4(0.f, 0.f, 0.f, 0.f);
            reinterpret_cast<float4*>(s_Xsc[ld_m])[ld_c] = xs;
        }
        __syncthreads();

        if (active_n) {
            for (int c = 0; c <= has_c1; ++c) {
                int kc128 = kc256 + c;
                #pragma unroll
                for (int sub = 0; sub < 4; ++sub) {
                    int kc32  = kc128 * 4 + sub;
                    int s_off = (c * 4 + sub) * 8 + c_k;
                    uint2 pkv = STREAM_W ? wv_ld_stream_u64(pk_base + kc32 * 32) : __ldg(pk_base + kc32 * 32);
                    float ws_self = __half2float(__ldg(sc_base + kc32 * 8)) * kInv127;
                    float ws0 = __shfl_sync(0xffffffffu, ws_self, col_n0 * 4);
                    float ws1 = __shfl_sync(0xffffffffu, ws_self, col_n1 * 4);
                    #pragma unroll
                    for (int t = 0; t < N_TILES16; ++t) {
                        if (cur_m > t * 16) {
                            int tr0 = t * 16 + r0;
                            float xs0 = s_Xsc[tr0][c * 4 + sub];
                            uint32_t a0 = s_X[tr0][s_off];
                            uint32_t a2 = s_X[tr0][s_off + 4];
                            uint32_t a1 = 0u, a3 = 0u;
                            float xs1 = 0.0f;
                            if constexpr (M_REGIME != 2) {
                                int tr1 = t * 16 + r1;
                                xs1 = s_Xsc[tr1][c * 4 + sub];
                                a1 = s_X[tr1][s_off];
                                a3 = s_X[tr1][s_off + 4];
                            }
                            int d0, d1, d2, d3;
                            wv_mma_m16n8k32_s8(d0, d1, d2, d3, a0, a1, a2, a3, pkv.x, pkv.y);
                            acc[t][0] += static_cast<float>(d0) * (ws0 * xs0);
                            acc[t][1] += static_cast<float>(d1) * (ws1 * xs0);
                            if constexpr (M_REGIME != 2) {
                                acc[t][2] += static_cast<float>(d2) * (ws0 * xs1);
                                acc[t][3] += static_cast<float>(d3) * (ws1 * xs1);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (active_n) {
        float b0_bias = (bias != nullptr && out_n0 < N)     ? __ldg(bias + out_n0)     : 0.0f;
        float b1_bias = (bias != nullptr && out_n0 + 1 < N) ? __ldg(bias + out_n0 + 1) : 0.0f;
        auto st_fn = [&](int m_row, int col, float v) {
            if (col >= N) return;
            if constexpr (SOFTCAP30) {
                float bf_v = __bfloat162float(__float2bfloat16(v));
                v = 30.0f * tanhf(bf_v * (1.0f / 30.0f));
            }
            size_t ix = static_cast<size_t>(m_row) * N + col;
            if constexpr (OUT_BF16) {
                reinterpret_cast<__nv_bfloat16*>(dst)[ix] = __float2bfloat16(v);
            } else {
                reinterpret_cast<float*>(dst)[ix] = v;
            }
        };
        #pragma unroll
        for (int t = 0; t < N_TILES16; ++t) {
            int m0 = m_base + t * 16 + r0;
            if (m0 < M) { st_fn(m0, out_n0, acc[t][0] + b0_bias); st_fn(m0, out_n0 + 1, acc[t][1] + b1_bias); }
            if constexpr (M_REGIME != 2) {
                int m1 = m_base + t * 16 + r1;
                if (m1 < M) { st_fn(m1, out_n0, acc[t][2] + b0_bias); st_fn(m1, out_n0 + 1, acc[t][3] + b1_bias); }
            }
        }
    }
    wv_pdl_trigger();
}

// Coalesced 256B-sector __dp4a Warp-GEMV for M = 1..3 (0 wasted MMA rows, 0 inner-loop barriers)
template<int M_CONST, bool OUT_BF16, bool SOFTCAP30, bool STREAM_W = false>
__global__ __launch_bounds__(128, 6)
void wv_ink_swz8_gemv_m8_kernel(
    const uint2* __restrict__ w_pk,   // [N/8, K/32, 32] uint2 (lane = 4 * col_n + c_k)
    const __half* __restrict__ w_sc,  // [N/8, K/32, 8] fp16 (scale * 127.0)
    const float* __restrict__ bias,   // [N] or nullptr
    const int8_t* __restrict__ X_q8,  // [M_CONST, K]
    const float* __restrict__ X_sc,   // [M_CONST, K/32]
    void* __restrict__ dst,
    int K, int N) {
    wv_pdl_wait();
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int col_n   = lane >> 2;
    int c_k     = lane & 3;
    int n_tile8 = blockIdx.x * 4 + warp_id;
    int out_col = n_tile8 * 8 + col_n;
    if (n_tile8 * 8 >= N) { wv_pdl_trigger(); return; }

    int n_kc32 = K >> 5;
    const uint2*  pk_base = w_pk + static_cast<size_t>(n_tile8) * n_kc32 * 32 + lane;
    const __half* sc_base = w_sc + static_cast<size_t>(n_tile8) * n_kc32 * 8 + col_n;

    float acc[M_CONST];
    #pragma unroll
    for (int m = 0; m < M_CONST; ++m) acc[m] = 0.0f;
    constexpr float kInv127 = 1.0f / 127.0f;

    for (int kc32 = 0; kc32 < n_kc32; ++kc32) {
        uint2 pkv = STREAM_W ? wv_ld_stream_u64(pk_base + kc32 * 32) : __ldg(pk_base + kc32 * 32);
        float ws  = __half2float(__ldg(sc_base + kc32 * 8)) * kInv127;
        #pragma unroll
        for (int m = 0; m < M_CONST; ++m) {
            const int32_t* xp = reinterpret_cast<const int32_t*>(X_q8 + static_cast<size_t>(m) * K + kc32 * 32);
            int32_t d = __dp4a(xp[c_k],     static_cast<int32_t>(pkv.x),
                        __dp4a(xp[4 + c_k], static_cast<int32_t>(pkv.y), 0));
            d += __shfl_xor_sync(0xffffffffu, d, 2);
            d += __shfl_xor_sync(0xffffffffu, d, 1);
            acc[m] += static_cast<float>(d) * (ws * X_sc[static_cast<size_t>(m) * n_kc32 + kc32]);
        }
    }

    if (c_k == 0 && out_col < N) {
        float b_val = (bias != nullptr) ? __ldg(bias + out_col) : 0.0f;
        #pragma unroll
        for (int m = 0; m < M_CONST; ++m) {
            float v = acc[m] + b_val;
            if constexpr (SOFTCAP30) {
                float bf_v = __bfloat162float(__float2bfloat16(v));
                v = 30.0f * tanhf(bf_v * (1.0f / 30.0f));
            }
            size_t ix = static_cast<size_t>(m) * N + out_col;
            if constexpr (OUT_BF16) {
                reinterpret_cast<__nv_bfloat16*>(dst)[ix] = __float2bfloat16(v);
            } else {
                reinterpret_cast<float*>(dst)[ix] = v;
            }
        }
    }
    wv_pdl_trigger();
}

// 2-Warp (64-thread, 16 output cols/CTA -> 2x SM parallelism for narrow-N!) __dp4a Warp-GEMV for M = 1..3:
template<int M_CONST, bool OUT_BF16, bool SOFTCAP30>
__global__ __launch_bounds__(64, 8)
void wv_ink_swz8_gemv_m8_2w_kernel(
    const uint2* __restrict__ w_pk,
    const __half* __restrict__ w_sc,
    const float* __restrict__ bias,
    const int8_t* __restrict__ X_q8,
    const float* __restrict__ X_sc,
    void* __restrict__ dst,
    int K, int N) {
    wv_pdl_wait();
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int col_n   = lane >> 2;
    int c_k     = lane & 3;
    int n_tile8 = blockIdx.x * 2 + warp_id;
    int out_col = n_tile8 * 8 + col_n;
    if (n_tile8 * 8 >= N) { wv_pdl_trigger(); return; }

    int n_kc32 = K >> 5;
    const uint2*  pk_base = w_pk + static_cast<size_t>(n_tile8) * n_kc32 * 32 + lane;
    const __half* sc_base = w_sc + static_cast<size_t>(n_tile8) * n_kc32 * 8 + col_n;

    float acc[M_CONST];
    #pragma unroll
    for (int m = 0; m < M_CONST; ++m) acc[m] = 0.0f;
    constexpr float kInv127 = 1.0f / 127.0f;

    for (int kc32 = 0; kc32 < n_kc32; ++kc32) {
        uint2 pkv = __ldg(pk_base + kc32 * 32);
        float ws  = __half2float(__ldg(sc_base + kc32 * 8)) * kInv127;
        #pragma unroll
        for (int m = 0; m < M_CONST; ++m) {
            const int32_t* xp = reinterpret_cast<const int32_t*>(X_q8 + static_cast<size_t>(m) * K + kc32 * 32);
            int32_t d = __dp4a(xp[c_k],     static_cast<int32_t>(pkv.x),
                        __dp4a(xp[4 + c_k], static_cast<int32_t>(pkv.y), 0));
            d += __shfl_xor_sync(0xffffffffu, d, 2);
            d += __shfl_xor_sync(0xffffffffu, d, 1);
            acc[m] += static_cast<float>(d) * (ws * X_sc[static_cast<size_t>(m) * n_kc32 + kc32]);
        }
    }

    if (c_k == 0 && out_col < N) {
        float b_val = (bias != nullptr) ? __ldg(bias + out_col) : 0.0f;
        #pragma unroll
        for (int m = 0; m < M_CONST; ++m) {
            float v = acc[m] + b_val;
            if constexpr (SOFTCAP30) {
                float bf_v = __bfloat162float(__float2bfloat16(v));
                v = 30.0f * tanhf(bf_v * (1.0f / 30.0f));
            }
            size_t ix = static_cast<size_t>(m) * N + out_col;
            if constexpr (OUT_BF16) {
                reinterpret_cast<__nv_bfloat16*>(dst)[ix] = __float2bfloat16(v);
            } else {
                reinterpret_cast<float*>(dst)[ix] = v;
            }
        }
    }
    wv_pdl_trigger();
}

// 2-Warp (64-thread, 16 cols/CTA) + kc512 mma.m16n8k32 for M = 4..8:
template<bool OUT_BF16, bool SOFTCAP30, bool STREAM_W = false>
__global__ __launch_bounds__(64, 8)
void wv_ink_swz8_mma_linear_m8_2w_kernel(
    const uint2* __restrict__ w_pk,
    const __half* __restrict__ w_sc,
    const float* __restrict__ bias,
    const int8_t* __restrict__ X_q8,
    const float* __restrict__ X_sc,
    void* __restrict__ dst,
    int K, int N, int M) {
    __shared__ uint32_t s_X[8][132];

    wv_pdl_wait();
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int lane_n  = lane >> 2;
    int c_k     = lane & 3;
    int r0      = lane >> 2;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;
    int n_tile8 = blockIdx.x * 2 + warp_id;
    int out_n0  = n_tile8 * 8 + col_n0;
    if (M <= 0) { wv_pdl_trigger(); return; }

    int n_kc128   = K >> 7;
    int n_kc32    = K >> 5;
    bool active_n = (n_tile8 * 8 < N);
    const uint2*  pk_base = active_n ? (w_pk + static_cast<size_t>(n_tile8) * n_kc32 * 32 + lane) : nullptr;
    const __half* sc_base = active_n ? (w_sc + static_cast<size_t>(n_tile8) * n_kc32 * 8 + lane_n) : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f;
    constexpr float kInv127 = 1.0f / 127.0f;

    for (int kc512 = 0; kc512 < n_kc128; kc512 += 4) {
        int rem_c = min(4, n_kc128 - kc512);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int idx   = u * 64 + threadIdx.x;
            int ld_m  = idx >> 5;
            int ld_q  = idx & 31;
            bool ok_k = (ld_q < rem_c * 8);
            uint4 xv  = (ld_m < M && ok_k)
                ? reinterpret_cast<const uint4*>(X_q8 + static_cast<size_t>(ld_m) * K + static_cast<size_t>(kc512) * 128)[ld_q]
                : make_uint4(0, 0, 0, 0);
            int c4 = ld_q * 4;
            s_X[ld_m][c4 + 0] = xv.x; s_X[ld_m][c4 + 1] = xv.y;
            s_X[ld_m][c4 + 2] = xv.z; s_X[ld_m][c4 + 3] = xv.w;
        }
        __syncthreads();

        if (active_n) {
            for (int c = 0; c < rem_c; ++c) {
                int kc128 = kc512 + c;
                #pragma unroll
                for (int sub = 0; sub < 4; ++sub) {
                    int kc32  = kc128 * 4 + sub;
                    int s_off = (c * 4 + sub) * 8 + c_k;
                    uint2 pkv = STREAM_W ? wv_ld_stream_u64(pk_base + kc32 * 32) : __ldg(pk_base + kc32 * 32);
                    float ws_self = __half2float(__ldg(sc_base + kc32 * 8)) * kInv127;
                    float ws0 = __shfl_sync(0xffffffffu, ws_self, col_n0 * 4);
                    float ws1 = __shfl_sync(0xffffffffu, ws_self, col_n1 * 4);
                    float xs0 = (r0 < M) ? X_sc[static_cast<size_t>(r0) * n_kc32 + kc32] : 0.0f;
                    int d0, d1, d2, d3;
                    wv_mma_m16n8k32_s8(d0, d1, d2, d3, s_X[r0][s_off], 0u, s_X[r0][s_off + 4], 0u, pkv.x, pkv.y);
                    acc0 += static_cast<float>(d0) * (ws0 * xs0);
                    acc1 += static_cast<float>(d1) * (ws1 * xs0);
                }
            }
        }
        __syncthreads();
    }

    if (active_n && r0 < M) {
        float b0_bias = (bias != nullptr && out_n0 < N)     ? __ldg(bias + out_n0)     : 0.0f;
        float b1_bias = (bias != nullptr && out_n0 + 1 < N) ? __ldg(bias + out_n0 + 1) : 0.0f;
        auto st_fn = [&](int col, float v) {
            if (col >= N) return;
            if constexpr (SOFTCAP30) {
                float bf_v = __bfloat162float(__float2bfloat16(v));
                v = 30.0f * tanhf(bf_v * (1.0f / 30.0f));
            }
            size_t ix = static_cast<size_t>(r0) * N + col;
            if constexpr (OUT_BF16) {
                reinterpret_cast<__nv_bfloat16*>(dst)[ix] = __float2bfloat16(v);
            } else {
                reinterpret_cast<float*>(dst)[ix] = v;
            }
        };
        st_fn(out_n0,     acc0 + b0_bias);
        st_fn(out_n0 + 1, acc1 + b1_bias);
    }
    wv_pdl_trigger();
}

// 1-Warp (32-thread, 8 output cols/CTA -> full 188-SM coverage on N <= 4608, e.g. wo N=2304 -> 288 CTAs!) for M == 1:
template<bool OUT_BF16, bool SOFTCAP30>
__global__ __launch_bounds__(32, 12)
void wv_ink_swz8_gemv_m1_1w_kernel(
    const uint2* __restrict__ w_pk,
    const __half* __restrict__ w_sc,
    const float* __restrict__ bias,
    const int8_t* __restrict__ X_q8,
    const float* __restrict__ X_sc,
    void* __restrict__ dst,
    int K, int N) {
    wv_pdl_wait();
    int lane    = threadIdx.x & 31;
    int col_n   = lane >> 2;
    int c_k     = lane & 3;
    int n_tile8 = blockIdx.x;
    int out_col = n_tile8 * 8 + col_n;
    if (n_tile8 * 8 >= N) { wv_pdl_trigger(); return; }

    int n_kc32 = K >> 5;
    const uint2*  pk_base = w_pk + static_cast<size_t>(n_tile8) * n_kc32 * 32 + lane;
    const __half* sc_base = w_sc + static_cast<size_t>(n_tile8) * n_kc32 * 8 + col_n;
    const int32_t* xp_b   = reinterpret_cast<const int32_t*>(X_q8);

    float acc = 0.0f;
    constexpr float kInv127 = 1.0f / 127.0f;
    auto step_kc = [&](int kc32) {
        uint2 pkv = __ldg(pk_base + kc32 * 32);
        float ws  = __half2float(__ldg(sc_base + kc32 * 8)) * kInv127;
        const int32_t* xp = xp_b + kc32 * 8;
        int32_t d = __dp4a(__ldg(xp + c_k),     static_cast<int32_t>(pkv.x),
                    __dp4a(__ldg(xp + 4 + c_k), static_cast<int32_t>(pkv.y), 0));
        d += __shfl_xor_sync(0xffffffffu, d, 2);
        d += __shfl_xor_sync(0xffffffffu, d, 1);
        acc += static_cast<float>(d) * (ws * __ldg(X_sc + kc32));
    };
    if (n_kc32 == 96) {
        #pragma unroll 8
        for (int kc32 = 0; kc32 < 96; ++kc32) step_kc(kc32);
    } else if (n_kc32 == 72) {
        #pragma unroll 8
        for (int kc32 = 0; kc32 < 72; ++kc32) step_kc(kc32);
    } else {
        #pragma unroll 4
        for (int kc32 = 0; kc32 < n_kc32; ++kc32) step_kc(kc32);
    }

    if (c_k == 0 && out_col < N) {
        float b_val = (bias != nullptr) ? __ldg(bias + out_col) : 0.0f;
        float v = acc + b_val;
        if constexpr (SOFTCAP30) {
            float bf_v = __bfloat162float(__float2bfloat16(v));
            v = 30.0f * tanhf(bf_v * (1.0f / 30.0f));
        }
        if constexpr (OUT_BF16) {
            reinterpret_cast<__nv_bfloat16*>(dst)[out_col] = __float2bfloat16(v);
        } else {
            reinterpret_cast<float*>(dst)[out_col] = v;
        }
    }
    wv_pdl_trigger();
}

template<bool OUT_BF16, bool SOFTCAP30 = false>
static inline void launch_wv_ink_swz8_linear(
    const WvInkSwz8Linear& W, const float* bias,
    const int8_t* X_q8, const float* X_sc, void* dst,
    int K, int N, int M, cudaStream_t stream, bool enable_pdl = false, bool stream_w = false) {
    if (M <= 0) return;
    int gx = (N + 31) / 32;
    if (N <= 9216) {
        if (M == 1 && N <= 4608 && !stream_w) {
            wv_launch_pdl(enable_pdl, dim3((N + 7) / 8), dim3(32), 0, stream,
                wv_ink_swz8_gemv_m1_1w_kernel<OUT_BF16, SOFTCAP30>, W.pk, W.sc, bias, X_q8, X_sc, dst, K, N);
            return;
        }
        int gx2 = (N + 15) / 16;
        if (M <= 3) {
            #define WV_INK_LAUNCH_GEMV2W_M(MC) \
                case MC: wv_launch_pdl(enable_pdl, dim3(gx2), dim3(64), 0, stream, \
                    wv_ink_swz8_gemv_m8_2w_kernel<MC, OUT_BF16, SOFTCAP30>, W.pk, W.sc, bias, X_q8, X_sc, dst, K, N); break
            switch (M) {
                WV_INK_LAUNCH_GEMV2W_M(1); WV_INK_LAUNCH_GEMV2W_M(2); WV_INK_LAUNCH_GEMV2W_M(3);
            }
            #undef WV_INK_LAUNCH_GEMV2W_M
            return;
        } else if (M <= 8) {
            wv_launch_pdl(enable_pdl, dim3(gx2, 1), dim3(64), 0, stream,
                wv_ink_swz8_mma_linear_m8_2w_kernel<OUT_BF16, SOFTCAP30>, W.pk, W.sc, bias, X_q8, X_sc, dst, K, N, M);
            return;
        }
    } else {
        if (M == 1) {
            if (stream_w) {
                wv_launch_pdl(enable_pdl, dim3(gx), dim3(128), 0, stream,
                    wv_ink_swz8_gemv_m8_kernel<1, OUT_BF16, SOFTCAP30, true>, W.pk, W.sc, bias, X_q8, X_sc, dst, K, N);
            } else {
                wv_launch_pdl(enable_pdl, dim3(gx), dim3(128), 0, stream,
                    wv_ink_swz8_gemv_m8_kernel<1, OUT_BF16, SOFTCAP30, false>, W.pk, W.sc, bias, X_q8, X_sc, dst, K, N);
            }
            return;
        } else if (M <= 8) {
            if (stream_w) {
                wv_launch_pdl(enable_pdl, dim3(gx, 1), dim3(128), 0, stream,
                    wv_ink_swz8_mma_linear_kernel<2, OUT_BF16, SOFTCAP30, true>, W.pk, W.sc, bias, X_q8, X_sc, dst, K, N, M);
            } else {
                wv_launch_pdl(enable_pdl, dim3(gx, 1), dim3(128), 0, stream,
                    wv_ink_swz8_mma_linear_kernel<2, OUT_BF16, SOFTCAP30, false>, W.pk, W.sc, bias, X_q8, X_sc, dst, K, N, M);
            }
            return;
        }
    }
    if (M <= 48) {
        int gy = (M + 15) / 16;
        wv_launch_pdl(enable_pdl, dim3(gx, gy), dim3(128), 0, stream,
            wv_ink_swz8_mma_linear_kernel<0, OUT_BF16, SOFTCAP30, false>, W.pk, W.sc, bias, X_q8, X_sc, dst, K, N, M);
    } else {
        int gy = (M + 63) / 64;
        wv_launch_pdl(enable_pdl, dim3(gx, gy), dim3(128), 0, stream,
            wv_ink_swz8_mma_linear_kernel<1, OUT_BF16, SOFTCAP30, false>, W.pk, W.sc, bias, X_q8, X_sc, dst, K, N, M);
    }
}

// Fused wr [384, 2304] (48 blocks -> BF16) + w_sink [24, 2304] (3 blocks -> FP32) for M > 8:
// Grid <<<dim3(51, M), dim3(32, 8)>>>
__global__ void wv_ink_bf16_fused_wr_sink_kernel(
    const __nv_bfloat16* __restrict__ qin_bf16, // [M, 2304]
    const __nv_bfloat16* __restrict__ wr_bf16,  // [384, 2304]
    const __nv_bfloat16* __restrict__ ws_bf16,  // [24, 2304]
    __nv_bfloat16* __restrict__ r_out_bf16,     // [M, 384]
    float* __restrict__ sinks_out_f32,          // [M, 24]
    int M) {
    wv_pdl_wait();
    int bx   = blockIdx.x; // 0..50
    int m    = blockIdx.y;
    if (m >= M) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x;
    bool is_wr = (bx < 48);
    int n      = (is_wr ? bx : (bx - 48)) * 8 + threadIdx.y;
    const __nv_bfloat16* xm = qin_bf16 + static_cast<size_t>(m) * INK2_HIDDEN;
    const __nv_bfloat16* wn = (is_wr ? wr_bf16 : ws_bf16) + static_cast<size_t>(n) * INK2_HIDDEN;
    float acc = 0.0f;
    for (int k = lane * 4; k < INK2_HIDDEN; k += 128) {
        uint2 xv  = *reinterpret_cast<const uint2*>(xm + k);
        uint2 wv2 = *reinterpret_cast<const uint2*>(wn + k);
        const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
        const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv2);
        float2 x0 = __bfloat1622float2(xb[0]), x1 = __bfloat1622float2(xb[1]);
        float2 w0 = __bfloat1622float2(wb[0]), w1 = __bfloat1622float2(wb[1]);
        acc += x0.x * w0.x + x0.y * w0.y + x1.x * w1.x + x1.y * w1.y;
    }
    acc = wv_warp_sum_f32(acc);
    if (lane == 0) {
        if (is_wr) {
            r_out_bf16[static_cast<size_t>(m) * INK2_REL_DIM + n] = __float2bfloat16(acc);
        } else {
            sinks_out_f32[static_cast<size_t>(m) * 24 + n] = acc;
        }
    }
    wv_pdl_trigger();
}

// Fused 6-projection Attention Input (wq + wk + wv + wg + wr + w_sink) for M = 1..3 (358 CTAs x 128 -> 2x SM occupancy!):
template<int M_CONST>
__global__ __launch_bounds__(128, 6)
void wv_ink_swz8_fused_qkvg_wr_sink_dp4a_kernel(
    WvInkSwz8Linear wq, WvInkSwz8Linear wk, WvInkSwz8Linear w_v, WvInkSwz8Linear wg,
    const __nv_bfloat16* __restrict__ wr_bf16,
    const __nv_bfloat16* __restrict__ ws_bf16,
    const int8_t* __restrict__ q8_q, const float* __restrict__ sc_q,
    const int8_t* __restrict__ q8_k, const float* __restrict__ sc_k,
    const int8_t* __restrict__ q8_v, const float* __restrict__ sc_v,
    const __nv_bfloat16* __restrict__ qin_bf16,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    __nv_bfloat16* __restrict__ g_out,
    __nv_bfloat16* __restrict__ r_out,
    float* __restrict__ sinks_out) {
    wv_pdl_wait();
    int cta     = blockIdx.x; // 0..357
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;

    if (cta >= 256) {
        int bx     = cta - 256; // 0..101 (96*4=384 wr + 6*4=24 w_sink)
        bool is_wr = (bx < 96);
        int n      = (is_wr ? bx : (bx - 96)) * 4 + warp_id;
        const __nv_bfloat16* wn = (is_wr ? wr_bf16 : ws_bf16) + static_cast<size_t>(n) * INK2_HIDDEN;
        float acc[M_CONST];
        #pragma unroll
        for (int m = 0; m < M_CONST; ++m) acc[m] = 0.0f;
        #pragma unroll 6
        for (int k = lane * 4; k < INK2_HIDDEN; k += 128) {
            uint2 wv2 = __ldg(reinterpret_cast<const uint2*>(wn + k));
            const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv2);
            float2 w0 = __bfloat1622float2(wb[0]), w1 = __bfloat1622float2(wb[1]);
            #pragma unroll
            for (int m = 0; m < M_CONST; ++m) {
                uint2 xv = __ldg(reinterpret_cast<const uint2*>(qin_bf16 + static_cast<size_t>(m) * INK2_HIDDEN + k));
                const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
                float2 x0 = __bfloat1622float2(xb[0]), x1 = __bfloat1622float2(xb[1]);
                acc[m] += x0.x * w0.x + x0.y * w0.y + x1.x * w1.x + x1.y * w1.y;
            }
        }
        #pragma unroll
        for (int m = 0; m < M_CONST; ++m) acc[m] = wv_warp_sum_f32(acc[m]);
        if (lane == 0) {
            #pragma unroll
            for (int m = 0; m < M_CONST; ++m) {
                if (is_wr) r_out[static_cast<size_t>(m) * INK2_REL_DIM + n] = __float2bfloat16(acc[m]);
                else       sinks_out[static_cast<size_t>(m) * 24 + n]       = acc[m];
            }
        }
        wv_pdl_trigger();
        return;
    }

    int col_n   = lane >> 2;
    int c_k     = lane & 3;
    const uint2*  w_pk;
    const __half* w_sc;
    const int8_t* X_q8;
    const float*  X_sc;
    __nv_bfloat16* dst;
    int n_tile8, N;
    if (cta < 96) {
        w_pk = wq.pk; w_sc = wq.sc; X_q8 = q8_q; X_sc = sc_q; dst = q_out;
        n_tile8 = cta * 4 + warp_id; N = INK2_ATTN_DIM;
    } else if (cta < 128) {
        w_pk = wk.pk; w_sc = wk.sc; X_q8 = q8_k; X_sc = sc_k; dst = k_out;
        n_tile8 = (cta - 96) * 4 + warp_id; N = INK2_KV_DIM;
    } else if (cta < 160) {
        w_pk = w_v.pk; w_sc = w_v.sc; X_q8 = q8_v; X_sc = sc_v; dst = v_out;
        n_tile8 = (cta - 128) * 4 + warp_id; N = INK2_KV_DIM;
    } else {
        w_pk = wg.pk; w_sc = wg.sc; X_q8 = q8_q; X_sc = sc_q; dst = g_out;
        n_tile8 = (cta - 160) * 4 + warp_id; N = INK2_ATTN_DIM;
    }

    constexpr int n_kc32  = INK2_HIDDEN >> 5; // 72
    constexpr float kInv127 = 1.0f / 127.0f;
    const uint2*  pk_base = w_pk + static_cast<size_t>(n_tile8) * n_kc32 * 32 + lane;
    const __half* sc_base = w_sc + static_cast<size_t>(n_tile8) * n_kc32 * 8 + col_n;

    float acc[M_CONST];
    #pragma unroll
    for (int m = 0; m < M_CONST; ++m) acc[m] = 0.0f;
    #pragma unroll 4
    for (int kc32 = 0; kc32 < n_kc32; ++kc32) {
        uint2 pkv = __ldg(pk_base + kc32 * 32);
        float ws  = __half2float(__ldg(sc_base + kc32 * 8)) * kInv127;
        #pragma unroll
        for (int m = 0; m < M_CONST; ++m) {
            const int32_t* xp = reinterpret_cast<const int32_t*>(X_q8 + static_cast<size_t>(m) * INK2_HIDDEN + kc32 * 32);
            int32_t d = __dp4a(__ldg(xp + c_k),     static_cast<int32_t>(pkv.x),
                        __dp4a(__ldg(xp + 4 + c_k), static_cast<int32_t>(pkv.y), 0));
            d += __shfl_xor_sync(0xffffffffu, d, 2);
            d += __shfl_xor_sync(0xffffffffu, d, 1);
            acc[m] += static_cast<float>(d) * (ws * __ldg(X_sc + static_cast<size_t>(m) * n_kc32 + kc32));
        }
    }
    if (c_k == 0) {
        int out_col = n_tile8 * 8 + col_n;
        #pragma unroll
        for (int m = 0; m < M_CONST; ++m) {
            dst[static_cast<size_t>(m) * N + out_col] = __float2bfloat16(acc[m]);
        }
    }
    wv_pdl_trigger();
}

// Fused 6-projection Attention Input via Tensor Core CTA_M = 8 mma.m16n8k32 (M = 4..8, 358 CTAs x 128):
__global__ __launch_bounds__(128, 6)
void wv_ink_swz8_fused_qkvg_wr_sink_mma_m8_kernel(
    WvInkSwz8Linear wq, WvInkSwz8Linear wk, WvInkSwz8Linear w_v, WvInkSwz8Linear wg,
    const __nv_bfloat16* __restrict__ wr_bf16,
    const __nv_bfloat16* __restrict__ ws_bf16,
    const int8_t* __restrict__ q8_q, const float* __restrict__ sc_q,
    const int8_t* __restrict__ q8_k, const float* __restrict__ sc_k,
    const int8_t* __restrict__ q8_v, const float* __restrict__ sc_v,
    const __nv_bfloat16* __restrict__ qin_bf16,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    __nv_bfloat16* __restrict__ g_out,
    __nv_bfloat16* __restrict__ r_out,
    float* __restrict__ sinks_out,
    int M) {
    __shared__ uint32_t s_X[8][68];
    wv_pdl_wait();
    int cta     = blockIdx.x; // 0..357
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;

    if (cta >= 256) {
        int bx     = cta - 256; // 0..101 (96*4=384 wr + 6*4=24 w_sink)
        bool is_wr = (bx < 96);
        int n      = (is_wr ? bx : (bx - 96)) * 4 + warp_id;
        const __nv_bfloat16* wn = (is_wr ? wr_bf16 : ws_bf16) + static_cast<size_t>(n) * INK2_HIDDEN;
        float acc[8];
        #pragma unroll
        for (int m = 0; m < 8; ++m) acc[m] = 0.0f;
        for (int k = lane * 4; k < INK2_HIDDEN; k += 128) {
            uint2 wv2 = __ldg(reinterpret_cast<const uint2*>(wn + k));
            const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv2);
            float2 w0 = __bfloat1622float2(wb[0]), w1 = __bfloat1622float2(wb[1]);
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                if (m < M) {
                    uint2 xv = *reinterpret_cast<const uint2*>(qin_bf16 + static_cast<size_t>(m) * INK2_HIDDEN + k);
                    const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
                    float2 x0 = __bfloat1622float2(xb[0]), x1 = __bfloat1622float2(xb[1]);
                    acc[m] += x0.x * w0.x + x0.y * w0.y + x1.x * w1.x + x1.y * w1.y;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < M) acc[m] = wv_warp_sum_f32(acc[m]);
        }
        if (lane == 0) {
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                if (m < M) {
                    if (is_wr) r_out[static_cast<size_t>(m) * INK2_REL_DIM + n] = __float2bfloat16(acc[m]);
                    else       sinks_out[static_cast<size_t>(m) * 24 + n]       = acc[m];
                }
            }
        }
        wv_pdl_trigger();
        return;
    }

    const uint2*  w_pk;
    const __half* w_sc;
    const int8_t* X_q8;
    const float*  X_sc;
    __nv_bfloat16* dst;
    int n_tile8, N;
    if (cta < 96) {
        w_pk = wq.pk; w_sc = wq.sc; X_q8 = q8_q; X_sc = sc_q; dst = q_out;
        n_tile8 = cta * 4 + warp_id; N = INK2_ATTN_DIM;
    } else if (cta < 128) {
        w_pk = wk.pk; w_sc = wk.sc; X_q8 = q8_k; X_sc = sc_k; dst = k_out;
        n_tile8 = (cta - 96) * 4 + warp_id; N = INK2_KV_DIM;
    } else if (cta < 160) {
        w_pk = w_v.pk; w_sc = w_v.sc; X_q8 = q8_v; X_sc = sc_v; dst = v_out;
        n_tile8 = (cta - 128) * 4 + warp_id; N = INK2_KV_DIM;
    } else {
        w_pk = wg.pk; w_sc = wg.sc; X_q8 = q8_q; X_sc = sc_q; dst = g_out;
        n_tile8 = (cta - 160) * 4 + warp_id; N = INK2_ATTN_DIM;
    }

    int lane_n  = lane >> 2;
    int c_k     = lane & 3;
    int r0      = lane >> 2;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;
    int out_n0  = n_tile8 * 8 + col_n0;
    constexpr int n_kc32    = INK2_HIDDEN >> 5; // 72
    constexpr float kInv127 = 1.0f / 127.0f;
    const uint2*  pk_base   = w_pk + static_cast<size_t>(n_tile8) * n_kc32 * 32 + lane;
    const __half* sc_base   = w_sc + static_cast<size_t>(n_tile8) * n_kc32 * 8 + lane_n;

    float acc0 = 0.0f, acc1 = 0.0f;
    int ld_m = threadIdx.x >> 4; // 0..7
    int ld_q = threadIdx.x & 15; // 0..15

    for (int kc256 = 0; kc256 < 18; kc256 += 2) {
        uint4 xv = (ld_m < M)
            ? reinterpret_cast<const uint4*>(X_q8 + static_cast<size_t>(ld_m) * INK2_HIDDEN + kc256 * 128)[ld_q]
            : make_uint4(0, 0, 0, 0);
        int c4 = ld_q * 4;
        s_X[ld_m][c4 + 0] = xv.x; s_X[ld_m][c4 + 1] = xv.y;
        s_X[ld_m][c4 + 2] = xv.z; s_X[ld_m][c4 + 3] = xv.w;
        __syncthreads();

        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            int kc128 = kc256 + c;
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int kc32  = kc128 * 4 + sub;
                int s_off = (c * 4 + sub) * 8 + c_k;
                uint2 pkv = __ldg(pk_base + kc32 * 32);
                float ws_self = __half2float(__ldg(sc_base + kc32 * 8)) * kInv127;
                float ws0 = __shfl_sync(0xffffffffu, ws_self, col_n0 * 4);
                float ws1 = __shfl_sync(0xffffffffu, ws_self, col_n1 * 4);
                float xs0 = (r0 < M) ? X_sc[static_cast<size_t>(r0) * n_kc32 + kc32] : 0.0f;
                int d0, d1, d2, d3;
                wv_mma_m16n8k32_s8(d0, d1, d2, d3, s_X[r0][s_off], 0u, s_X[r0][s_off + 4], 0u, pkv.x, pkv.y);
                acc0 += static_cast<float>(d0) * (ws0 * xs0);
                acc1 += static_cast<float>(d1) * (ws1 * xs0);
            }
        }
        __syncthreads();
    }
    if (r0 < M) {
        size_t row_off = static_cast<size_t>(r0) * N;
        dst[row_off + out_n0]     = __float2bfloat16(acc0);
        dst[row_off + out_n0 + 1] = __float2bfloat16(acc1);
    }
    wv_pdl_trigger();
}

// 8-Token Register-Tiled wr [384, 2304] + w_sink [24, 2304] (8x fewer DRAM weight reads for M > 8 prefill!):
__global__ void wv_ink_bf16_fused_wr_sink_tiled8_kernel(
    const __nv_bfloat16* __restrict__ qin_bf16,
    const __nv_bfloat16* __restrict__ wr_bf16,
    const __nv_bfloat16* __restrict__ ws_bf16,
    __nv_bfloat16* __restrict__ r_out_bf16,
    float* __restrict__ sinks_out_f32,
    int M) {
    wv_pdl_wait();
    int bx     = blockIdx.x;
    int m_base = blockIdx.y * 8;
    int cur_m  = min(8, M - m_base);
    if (cur_m <= 0) { wv_pdl_trigger(); return; }
    int lane   = threadIdx.x;
    bool is_wr = (bx < 48);
    int n      = (is_wr ? bx : (bx - 48)) * 8 + threadIdx.y;
    const __nv_bfloat16* wn = (is_wr ? wr_bf16 : ws_bf16) + static_cast<size_t>(n) * INK2_HIDDEN;
    float acc[8];
    #pragma unroll
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;

    for (int k = lane * 4; k < INK2_HIDDEN; k += 128) {
        uint2 wv2 = __ldg(reinterpret_cast<const uint2*>(wn + k));
        const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv2);
        float2 w0 = __bfloat1622float2(wb[0]), w1 = __bfloat1622float2(wb[1]);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < cur_m) {
                uint2 xv = *reinterpret_cast<const uint2*>(qin_bf16 + static_cast<size_t>(m_base + t) * INK2_HIDDEN + k);
                const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
                float2 x0 = __bfloat1622float2(xb[0]), x1 = __bfloat1622float2(xb[1]);
                acc[t] += x0.x * w0.x + x0.y * w0.y + x1.x * w1.x + x1.y * w1.y;
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t < cur_m) acc[t] = wv_warp_sum_f32(acc[t]);
    }
    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < cur_m) {
                int m = m_base + t;
                if (is_wr) r_out_bf16[static_cast<size_t>(m) * INK2_REL_DIM + n] = __float2bfloat16(acc[t]);
                else       sinks_out_f32[static_cast<size_t>(m) * 24 + n]        = acc[t];
            }
        }
    }
    wv_pdl_trigger();
}

static inline void launch_wv_ink_swz8_fused_qkvg_wr_sink(
    const WvInkSwz8Linear& wq, const WvInkSwz8Linear& wk, const WvInkSwz8Linear& w_v, const WvInkSwz8Linear& wg,
    const __nv_bfloat16* wr_bf16, const __nv_bfloat16* ws_bf16,
    const int8_t* q8_q, const float* sc_q,
    const int8_t* q8_k, const float* sc_k,
    const int8_t* q8_v, const float* sc_v,
    const __nv_bfloat16* qin_bf16,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out, __nv_bfloat16* v_out, __nv_bfloat16* g_out,
    __nv_bfloat16* r_out, float* sinks_out,
    int M, cudaStream_t stream, bool enable_pdl = false) {
    if (M <= 3) {
        #define WV_INK_LAUNCH_QKVG_WR_M(MC) \
            case MC: wv_launch_pdl(enable_pdl, dim3(358), dim3(128), 0, stream, \
                wv_ink_swz8_fused_qkvg_wr_sink_dp4a_kernel<MC>, \
                wq, wk, w_v, wg, wr_bf16, ws_bf16, q8_q, sc_q, q8_k, sc_k, q8_v, sc_v, \
                qin_bf16, q_out, k_out, v_out, g_out, r_out, sinks_out); break
        switch (M) {
            WV_INK_LAUNCH_QKVG_WR_M(1); WV_INK_LAUNCH_QKVG_WR_M(2); WV_INK_LAUNCH_QKVG_WR_M(3);
        }
        #undef WV_INK_LAUNCH_QKVG_WR_M
    } else if (M <= 8) {
        wv_launch_pdl(enable_pdl, dim3(358), dim3(128), 0, stream,
            wv_ink_swz8_fused_qkvg_wr_sink_mma_m8_kernel,
            wq, wk, w_v, wg, wr_bf16, ws_bf16, q8_q, sc_q, q8_k, sc_k, q8_v, sc_v,
            qin_bf16, q_out, k_out, v_out, g_out, r_out, sinks_out, M);
    } else {
        launch_wv_ink_swz8_linear<true>(wq,  nullptr, q8_q, sc_q, q_out, INK2_HIDDEN, INK2_ATTN_DIM, M, stream, enable_pdl);
        launch_wv_ink_swz8_linear<true>(wk,  nullptr, q8_k, sc_k, k_out, INK2_HIDDEN, INK2_KV_DIM,   M, stream, enable_pdl);
        launch_wv_ink_swz8_linear<true>(w_v, nullptr, q8_v, sc_v, v_out, INK2_HIDDEN, INK2_KV_DIM,   M, stream, enable_pdl);
        launch_wv_ink_swz8_linear<true>(wg,  nullptr, q8_q, sc_q, g_out, INK2_HIDDEN, INK2_ATTN_DIM, M, stream, enable_pdl);
        wv_launch_pdl(enable_pdl, dim3(51, (M + 7) / 8), dim3(32, 8), 0, stream,
            wv_ink_bf16_fused_wr_sink_tiled8_kernel, qin_bf16, wr_bf16, ws_bf16, r_out, sinks_out, M);
    }
}

// Multi-CTA greedy argmax over [:valid_vocab] (256 CTAs for full 188-SM Blackwell saturation!):
// Stage 1: Grid <<<dim3(256, M), 256>>> -> partials [M, 256] (v, idx)
__global__ void wv_ink_argmax_vocab_part_kernel(
    const float* __restrict__ logits, // [M, stride_v]
    float* __restrict__ part_v,       // [M, 256]
    int32_t* __restrict__ part_i,     // [M, 256]
    int stride_v,
    int valid_vocab) {
    wv_pdl_wait();
    int blk    = blockIdx.x; // 0..255
    size_t m   = static_cast<size_t>(blockIdx.y);
    int tid    = threadIdx.x;
    const float* row = logits + m * static_cast<size_t>(stride_v);
    float best_v = -1e30f;
    int   best_i = 0;
    for (int i = blk * 256 + tid; i < valid_vocab; i += 256 * 256) {
        float v = row[i];
        if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
    }
    int lane = tid & 31, wid = tid >> 5;
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best_v, mask);
        int   oi = __shfl_xor_sync(0xffffffffu, best_i, mask);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    __shared__ float s_v[8];
    __shared__ int   s_i[8];
    if (lane == 0) { s_v[wid] = best_v; s_i[wid] = best_i; }
    __syncthreads();
    if (tid == 0) {
        float fv = s_v[0]; int fi = s_i[0];
        for (int w = 1; w < 8; ++w) {
            if (s_v[w] > fv || (s_v[w] == fv && s_i[w] < fi)) { fv = s_v[w]; fi = s_i[w]; }
        }
        part_v[m * 256 + blk] = fv;
        part_i[m * 256 + blk] = fi;
    }
    wv_pdl_trigger();
}

// Fused Streaming (L2::evict_first) SWZ8 M=1 Greedy Head + Exact SOFTCAP30 + 256-CTA Argmax Stage 1:
// Grid <<<dim3(256, B_act), 128>>> -> partials [B_act, 256] (0.0-ULP bitwise parity, preserves 96 MB L2!)
__global__ __launch_bounds__(128, 4)
void wv_ink_swz8_gemv_m1_greedy_part_kernel(
    const uint2* __restrict__ w_pk,
    const __half* __restrict__ w_sc,
    const float* __restrict__ bias,
    const int8_t* __restrict__ X_q8,
    const float* __restrict__ X_sc,
    float* __restrict__ part_v,
    int32_t* __restrict__ part_i,
    int valid_vocab) {
    __shared__ int32_t s_xq[INK2_HIDDEN / 4];
    __shared__ float   s_xs[INK2_HIDDEN / 32];
    __shared__ float   s_v[4];
    __shared__ int     s_i[4];

    wv_pdl_wait();
    int blk  = blockIdx.x;    // 0..255
    size_t b = static_cast<size_t>(blockIdx.y);
    int tid  = threadIdx.x;   // 0..127
    int wid  = tid >> 5;      // 0..3
    int lane = tid & 31;      // 0..31
    int col_n = lane >> 2;    // 0..7
    int c_k   = lane & 3;     // 0..3

    const int32_t* xp_b = reinterpret_cast<const int32_t*>(X_q8 + b * INK2_HIDDEN);
    const float*   xs_b = X_sc + b * (INK2_HIDDEN / 32);
    for (int q = tid; q < (INK2_HIDDEN / 4); q += 128) s_xq[q] = xp_b[q];
    if (tid < (INK2_HIDDEN / 32)) s_xs[tid] = xs_b[tid];
    __syncthreads();

    constexpr int n_kc32  = INK2_HIDDEN >> 5; // 72
    constexpr float kInv127 = 1.0f / 127.0f;
    float best_v = -1e30f;
    int   best_i = 0;
    int n_tiles8 = (valid_vocab + 7) >> 3;

    for (int n_tile8 = blk * 4 + wid; n_tile8 < n_tiles8; n_tile8 += 256 * 4) {
        const uint2*  pk_base = w_pk + static_cast<size_t>(n_tile8) * n_kc32 * 32 + lane;
        const __half* sc_base = w_sc + static_cast<size_t>(n_tile8) * n_kc32 * 8 + col_n;
        float acc = 0.0f;
        #pragma unroll 8
        for (int kc32 = 0; kc32 < n_kc32; ++kc32) {
            uint2 pkv = wv_ld_stream_u64(pk_base + kc32 * 32);
            float ws  = __half2float(__ldg(sc_base + kc32 * 8)) * kInv127;
            int32_t d = __dp4a(s_xq[kc32 * 8 + c_k],     static_cast<int32_t>(pkv.x),
                        __dp4a(s_xq[kc32 * 8 + 4 + c_k], static_cast<int32_t>(pkv.y), 0));
            d += __shfl_xor_sync(0xffffffffu, d, 2);
            d += __shfl_xor_sync(0xffffffffu, d, 1);
            acc += static_cast<float>(d) * (ws * s_xs[kc32]);
        }
        int out_col = n_tile8 * 8 + col_n;
        if (c_k == 0 && out_col < valid_vocab) {
            float b_val = (bias != nullptr) ? __ldg(bias + out_col) : 0.0f;
            float bf_v  = __bfloat162float(__float2bfloat16(acc + b_val));
            float v     = 30.0f * tanhf(bf_v * (1.0f / 30.0f));
            if (v > best_v || (v == best_v && out_col < best_i)) {
                best_v = v;
                best_i = out_col;
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best_v, mask);
        int   oi = __shfl_xor_sync(0xffffffffu, best_i, mask);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    if (lane == 0) { s_v[wid] = best_v; s_i[wid] = best_i; }
    __syncthreads();
    if (tid == 0) {
        float fv = s_v[0]; int fi = s_i[0];
        for (int w = 1; w < 4; ++w) {
            if (s_v[w] > fv || (s_v[w] == fv && s_i[w] < fi)) { fv = s_v[w]; fi = s_i[w]; }
        }
        part_v[b * 256 + blk] = fv;
        part_i[b * 256 + blk] = fi;
    }
    wv_pdl_trigger();
}

// Stage 2: Grid <<<M, 256>>> -> exact global tie-broken out_ids[m]
__global__ void wv_ink_argmax_vocab_final_kernel(
    const float* __restrict__ part_v,
    const int32_t* __restrict__ part_i,
    int32_t* __restrict__ out_ids) {
    wv_pdl_wait();
    size_t m = static_cast<size_t>(blockIdx.x);
    int tid  = threadIdx.x; // 0..255
    int lane = tid & 31, wid = tid >> 5;
    float bv = part_v[m * 256 + tid];
    int   bi = part_i[m * 256 + tid];
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, bv, mask);
        int   oi = __shfl_xor_sync(0xffffffffu, bi, mask);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float s_v[8];
    __shared__ int   s_i[8];
    if (lane == 0) { s_v[wid] = bv; s_i[wid] = bi; }
    __syncthreads();
    if (tid == 0) {
        float fv = s_v[0]; int fi = s_i[0];
        for (int w = 1; w < 8; ++w) {
            if (s_v[w] > fv || (s_v[w] == fv && s_i[w] < fi)) { fv = s_v[w]; fi = s_i[w]; }
        }
        out_ids[m] = fi;
    }
    wv_pdl_trigger();
}

static inline void launch_wv_ink_argmax_vocab(
    const float* logits, float* ws_part_v, int32_t* ws_part_i, int32_t* out_ids,
    int M, int stride_v, int valid_vocab, bool enable_pdl, cudaStream_t stream) {
    wv_launch_pdl(enable_pdl, dim3(256, M), dim3(256), 0, stream,
        wv_ink_argmax_vocab_part_kernel, logits, ws_part_v, ws_part_i, stride_v, valid_vocab);
    wv_launch_pdl(enable_pdl, dim3(M), dim3(256), 0, stream,
        wv_ink_argmax_vocab_final_kernel, ws_part_v, ws_part_i, out_ids);
}

struct WvInkSampleCfg {
    float    temp     = 0.6f;
    float    top_p    = 0.95f;
    int32_t  eos_id   = 200006;
    int32_t  pad0     = 0;
    uint64_t seed     = 42ULL;
    uint64_t rng_step = 0ULL;
};

__device__ __forceinline__ float wv_ink_rng_uniform01(uint64_t seed, uint64_t step, uint32_t sub) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL * (step + 1ULL) + 0xBF58476D1CE4E5B9ULL * static_cast<uint64_t>(sub + 1u);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    uint32_t u24 = static_cast<uint32_t>(z >> 40);
    return (static_cast<float>(u24) + 0.5f) * (1.0f / 16777216.0f);
}

struct WvInkTopPCutoff {
    float m_max;
    float inv_temp;
    float thresh_logit;
    int   tie_max_idx;
    int   argmax_idx;
    float inv_kept_Z;
};

__device__ __forceinline__ bool wv_ink_topp_is_kept(float l_i, int i, const WvInkTopPCutoff& c) {
    return (i == c.argmax_idx) || (l_i > c.thresh_logit) || (l_i == c.thresh_logit && i <= c.tie_max_idx);
}

// Exact, bitwise-deterministic 512-thread (16 warps) Top-p nucleus solver over valid_vocab:
template<typename LogitFn>
__device__ inline WvInkTopPCutoff wv_ink_compute_topp_cutoff_cta512(
    LogitFn logit_fn, int valid_vocab, float temp, float top_p) {
    int tid  = threadIdx.x; // 0..511
    int lane = tid & 31, wid = tid >> 5;

    __shared__ float s_wv[16];
    __shared__ int   s_wi[16];
    __shared__ unsigned long long s_bin_q32[256];
    __shared__ int   s_bin_cnt[256];
    __shared__ int   s_cand_n;
    __shared__ float s_cand_v[512], s_sort_v[512];
    __shared__ int   s_cand_i[512], s_sort_i[512];
    __shared__ WvInkTopPCutoff s_out;

    float best_v = -1e30f;
    int   best_i = 0;
    for (int i = tid; i < valid_vocab; i += 512) {
        float v = logit_fn(i);
        if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
    }
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best_v, mask);
        int   oi = __shfl_xor_sync(0xffffffffu, best_i, mask);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    if (lane == 0) { s_wv[wid] = best_v; s_wi[wid] = best_i; }
    __syncthreads();
    if (tid == 0) {
        float fv = s_wv[0]; int fi = s_wi[0];
        for (int w = 1; w < 16; ++w) {
            if (s_wv[w] > fv || (s_wv[w] == fv && s_wi[w] < fi)) { fv = s_wv[w]; fi = s_wi[w]; }
        }
        s_out.m_max      = fv;
        s_out.argmax_idx = fi;
        s_out.inv_temp   = 1.0f / fmaxf(temp, 1e-5f);
    }
    __syncthreads();

    float m_max    = s_out.m_max;
    float inv_temp = s_out.inv_temp;
    float p_sum    = 0.0f;
    for (int i = tid; i < valid_vocab; i += 512) {
        p_sum += expf((logit_fn(i) - m_max) * inv_temp);
    }
    p_sum = wv_warp_sum_f32(p_sum);
    if (lane == 0) s_wv[wid] = p_sum;
    __syncthreads();
    float Z_all = 0.0f;
    #pragma unroll
    for (int w = 0; w < 16; ++w) Z_all += s_wv[w];
    float inv_Z = 1.0f / fmaxf(Z_all, 1e-20f);

    if (top_p <= 0.0f || top_p >= 0.9999f) {
        if (tid == 0) {
            s_out.thresh_logit = -1e30f;
            s_out.tie_max_idx  = valid_vocab;
            s_out.inv_kept_Z   = inv_Z;
        }
        __syncthreads();
        return s_out;
    }
    if (inv_Z >= top_p) {
        if (tid == 0) {
            s_out.thresh_logit = 1e30f;
            s_out.tie_max_idx  = -1;
            s_out.inv_kept_Z   = 1.0f;
        }
        __syncthreads();
        return s_out;
    }

    if (tid < 256) { s_bin_q32[tid] = 0ULL; s_bin_cnt[tid] = 0; }
    if (tid == 0) s_cand_n = 0;
    __syncthreads();

    constexpr double kQ32 = 4294967296.0;
    for (int i = tid; i < valid_vocab; i += 512) {
        float d = (m_max - logit_fn(i)) * inv_temp;
        if (d < 20.0f) {
            int b = max(0, static_cast<int>(d * (255.0f / 20.0f)));
            if (b < 255) {
                float pr = expf(-d) * inv_Z;
                atomicAdd(&s_bin_q32[b], static_cast<unsigned long long>(static_cast<double>(pr) * kQ32 + 0.5));
                atomicAdd(&s_bin_cnt[b], 1);
            }
        }
    }
    __syncthreads();
    __shared__ int   s_b_crit;
    __shared__ float s_mass_pre;
    if (tid == 0) {
        unsigned long long acc_q = 0ULL;
        unsigned long long tgt_q = static_cast<unsigned long long>(static_cast<double>(top_p) * kQ32);
        int bc = 255;
        for (int b = 0; b < 255; ++b) {
            if (acc_q + s_bin_q32[b] >= tgt_q) { bc = b; break; }
            acc_q += s_bin_q32[b];
        }
        s_b_crit   = bc;
        s_mass_pre = static_cast<float>(static_cast<double>(acc_q) * (1.0 / kQ32));
    }
    __syncthreads();

    int b_crit = s_b_crit;
    if (b_crit < 255 && s_bin_cnt[b_crit] <= 512) {
        for (int i = tid; i < valid_vocab; i += 512) {
            float lv = logit_fn(i);
            float d  = (m_max - lv) * inv_temp;
            int b    = min(255, max(0, static_cast<int>(d * (255.0f / 20.0f))));
            if (b == b_crit) {
                int pos = atomicAdd(&s_cand_n, 1);
                if (pos < 512) { s_cand_v[pos] = lv; s_cand_i[pos] = i; }
            }
        }
        __syncthreads();
        int cnt = min(512, s_cand_n);
        if (tid < cnt) {
            float mv = s_cand_v[tid];
            int   mi = s_cand_i[tid];
            int rank = 0;
            for (int j = 0; j < cnt; ++j) {
                float ov = s_cand_v[j]; int oi = s_cand_i[j];
                if (ov > mv || (ov == mv && oi < mi)) rank++;
            }
            s_sort_v[rank] = mv;
            s_sort_i[rank] = mi;
        }
        __syncthreads();
        if (tid == 0) {
            float cur = s_mass_pre;
            float tl  = -1e30f;
            int   ti  = valid_vocab;
            for (int r = 0; r < cnt; ++r) {
                tl = s_sort_v[r];
                ti = s_sort_i[r];
                cur += expf((tl - m_max) * inv_temp) * inv_Z;
                if (cur >= top_p) break;
            }
            s_out.thresh_logit = tl;
            s_out.tie_max_idx  = ti;
        }
        __syncthreads();
    } else {
        float d_lo = static_cast<float>(b_crit) * (20.0f / 255.0f);
        float d_hi = (b_crit == 255) ? 60.0f : (static_cast<float>(b_crit + 1) * (20.0f / 255.0f));
        float l_hi = m_max - d_lo / inv_temp;
        float l_lo = m_max - d_hi / inv_temp;
        for (int it = 0; it < 12; ++it) {
            float l_mid = 0.5f * (l_lo + l_hi);
            float ps = 0.0f;
            for (int i = tid; i < valid_vocab; i += 512) {
                float lv = logit_fn(i);
                if (lv >= l_mid) ps += expf((lv - m_max) * inv_temp);
            }
            ps = wv_warp_sum_f32(ps);
            if (lane == 0) s_wv[wid] = ps;
            __syncthreads();
            float tot = 0.0f;
            for (int w = 0; w < 16; ++w) tot += s_wv[w];
            if (tot * inv_Z >= top_p) l_lo = l_mid; else l_hi = l_mid;
            __syncthreads();
        }
        if (tid == 0) { s_out.thresh_logit = l_lo; s_out.tie_max_idx = valid_vocab; }
        __syncthreads();
    }

    float kept_sum = 0.0f;
    for (int i = tid; i < valid_vocab; i += 512) {
        float lv = logit_fn(i);
        if (wv_ink_topp_is_kept(lv, i, s_out)) kept_sum += expf((lv - m_max) * inv_temp);
    }
    kept_sum = wv_warp_sum_f32(kept_sum);
    if (lane == 0) s_wv[wid] = kept_sum;
    __syncthreads();
    if (tid == 0) {
        float kz = 0.0f;
        for (int w = 0; w < 16; ++w) kz += s_wv[w];
        s_out.inv_kept_Z = 1.0f / fmaxf(kz, 1e-20f);
    }
    __syncthreads();
    return s_out;
}

// Deterministic 512-thread parallel prefix-sum CDF inversion over w_fn(i) >= 0 (u01 in (0, 1)):
template<typename WeightFn>
__device__ inline int wv_ink_sample_cdf_cta512(
    WeightFn w_fn, int valid_vocab, float W_tot, float u01, int fallback_idx) {
    int tid  = threadIdx.x;
    int lane = tid & 31, wid = tid >> 5;
    __shared__ float s_wsum[16];
    __shared__ float s_wexcl[16];
    __shared__ int   s_win_i[16];
    __shared__ int   s_final_tok;

    float target = u01 * W_tot;
    float cum    = 0.0f;
    int   my_nz  = -1;
    if (tid == 0) s_final_tok = fallback_idx;
    __syncthreads();

    for (int base = 0; base < valid_vocab; base += 512) {
        int i    = base + tid;
        float wi = (i < valid_vocab) ? w_fn(i) : 0.0f;
        if (wi > 0.0f) my_nz = i;
        float pf = wi;
        #pragma unroll
        for (int off = 1; off <= 16; off <<= 1) {
            float n = __shfl_up_sync(0xffffffffu, pf, off);
            if (lane >= off) pf += n;
        }
        if (lane == 31) s_wsum[wid] = pf;
        __syncthreads();
        if (wid == 0) {
            float ws = (lane < 16) ? s_wsum[lane] : 0.0f;
            float wp = ws;
            #pragma unroll
            for (int off = 1; off <= 8; off <<= 1) {
                float n = __shfl_up_sync(0xffffffffu, wp, off);
                if (lane >= off) wp += n;
            }
            if (lane < 16) s_wexcl[lane] = wp - ws;
        }
        __syncthreads();
        float chunk_tot = s_wexcl[15] + s_wsum[15];
        if (cum + chunk_tot > target) {
            float my_incl = cum + s_wexcl[wid] + pf;
            int cand = (wi > 0.0f && my_incl > target) ? i : 0x7fffffff;
            #pragma unroll
            for (int mask = 16; mask >= 1; mask >>= 1) {
                cand = min(cand, __shfl_xor_sync(0xffffffffu, cand, mask));
            }
            if (lane == 0) s_win_i[wid] = cand;
            __syncthreads();
            if (tid == 0) {
                int best = 0x7fffffff;
                for (int w = 0; w < 16; ++w) best = min(best, s_win_i[w]);
                if (best < valid_vocab) s_final_tok = best;
            }
            __syncthreads();
            int out_tok = s_final_tok;
            __syncthreads();
            return out_tok;
        }
        cum += chunk_tot;
        __syncthreads();
    }
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) my_nz = max(my_nz, __shfl_xor_sync(0xffffffffu, my_nz, mask));
    if (lane == 0) s_win_i[wid] = my_nz;
    __syncthreads();
    if (tid == 0) {
        int last_nz = -1;
        for (int w = 0; w < 16; ++w) last_nz = max(last_nz, s_win_i[w]);
        if (last_nz >= 0) s_final_tok = last_nz;
    }
    __syncthreads();
    int out_tok = s_final_tok;
    __syncthreads();
    return out_tok;
}

// Batched M=1 (temp, top_p) Nucleus Sampler (temp <= 0 -> exact greedy argmax): Grid <<<M, 512>>>
__global__ void wv_ink_sample_topp_vocab_kernel(
    const float* __restrict__ logits,     // [M, stride_v]
    int32_t* __restrict__ out_ids,        // [M]
    const WvInkSampleCfg* __restrict__ cfg,
    int stride_v,
    int valid_vocab) {
    wv_pdl_wait();
    int m = blockIdx.x;
    const float* row = logits + static_cast<size_t>(m) * stride_v;
    float temp  = cfg->temp;
    float top_p = cfg->top_p;
    auto lfn = [&](int i) -> float { return row[i]; };
    WvInkTopPCutoff cut = wv_ink_compute_topp_cutoff_cta512(lfn, valid_vocab, fmaxf(temp, 1e-5f), top_p);
    if (temp <= 0.0f) {
        if (threadIdx.x == 0) out_ids[m] = cut.argmax_idx;
        wv_pdl_trigger();
        return;
    }
    float u01 = wv_ink_rng_uniform01(cfg->seed, cfg->rng_step, static_cast<uint32_t>(m));
    auto wfn = [&](int i) -> float {
        float lv = row[i];
        return wv_ink_topp_is_kept(lv, i, cut) ? (expf((lv - cut.m_max) * cut.inv_temp) * cut.inv_kept_Z) : 0.0f;
    };
    int tok = wv_ink_sample_cdf_cta512(wfn, valid_vocab, 1.0f, u01, cut.argmax_idx);
    if (threadIdx.x == 0) out_ids[m] = tok;
    wv_pdl_trigger();
}

// Converts logits [M, stride_v] IN-PLACE into normalized Top-p probabilities: Grid <<<M, 512>>>
__global__ void wv_ink_logits_to_topp_probs_kernel(
    float* __restrict__ logits_io,
    const WvInkSampleCfg* __restrict__ cfg,
    int stride_v,
    int valid_vocab) {
    wv_pdl_wait();
    int m   = blockIdx.x;
    int tid = threadIdx.x;
    float* row = logits_io + static_cast<size_t>(m) * stride_v;
    auto lfn = [&](int i) -> float { return row[i]; };
    WvInkTopPCutoff cut = wv_ink_compute_topp_cutoff_cta512(lfn, valid_vocab, cfg->temp, cfg->top_p);
    __syncthreads();
    for (int i = tid; i < stride_v; i += 512) {
        if (i < valid_vocab) {
            float lv = row[i];
            row[i] = wv_ink_topp_is_kept(lv, i, cut) ? (expf((lv - cut.m_max) * cut.inv_temp) * cut.inv_kept_Z) : 0.0f;
        } else {
            row[i] = 0.0f;
        }
    }
    wv_pdl_trigger();
}

// Greedy MTP-8 (temp <= 0) GPU verifier: writes step_out = [n_acc, tok_0 .. tok_{n_acc-1}] & updates draft_blk8[0]
__global__ void wv_ink_mtp8_greedy_verify_kernel(
    int32_t* __restrict__ draft_blk8,
    const int32_t* __restrict__ target8,
    const WvInkSampleCfg* __restrict__ cfg,
    int32_t* __restrict__ step_out) {
    wv_pdl_wait();
    if (threadIdx.x == 0) {
        int eos_id = cfg->eos_id;
        int n_acc  = 0;
        for (int k = 0; k < 8; ++k) {
            int32_t t = target8[k];
            step_out[1 + k] = t;
            n_acc = k + 1;
            if (t == eos_id) break;
            if (k < 7 && draft_blk8[1 + k] != t) break;
        }
        step_out[0]   = n_acc;
        draft_blk8[0] = step_out[n_acc];
    }
    wv_pdl_trigger();
}

// Lossless Stochastic MTP-8 (temp > 0) Speculative Rejection Sampler: Grid <<<1, 512>>>
__global__ void wv_ink_mtp8_spec_verify_kernel(
    int32_t* __restrict__ draft_blk8,       // [8] ([anchor, d1..d7]) -> draft_blk8[0] updated to new anchor
    const float* __restrict__ p_probs,      // [8, stride_v] target Top-p probs
    const float* __restrict__ q_probs,      // [7, stride_v] draft Top-p probs
    const WvInkSampleCfg* __restrict__ cfg,
    int32_t* __restrict__ step_out,         // [9]: [n_acc, acc_0..acc_7]
    int stride_v,
    int valid_vocab) {
    wv_pdl_wait();
    int tid  = threadIdx.x;
    int lane = tid & 31, wid = tid >> 5;
    __shared__ float s_wv[16];
    __shared__ int   s_accept;
    __shared__ int   s_done;
    __shared__ int   s_n_acc;

    uint64_t seed = cfg->seed;
    uint64_t step = cfg->rng_step;
    int eos_id    = cfg->eos_id;
    if (tid == 0) { s_done = 0; s_n_acc = 0; }
    __syncthreads();

    for (int k = 0; k < 7; ++k) {
        const float* pk = p_probs + static_cast<size_t>(k) * stride_v;
        const float* qk = q_probs + static_cast<size_t>(k) * stride_v;
        if (tid == 0) {
            int x     = draft_blk8[1 + k];
            float px  = (x >= 0 && x < valid_vocab) ? pk[x] : 0.0f;
            float qx  = (x >= 0 && x < valid_vocab) ? qk[x] : 0.0f;
            float u   = wv_ink_rng_uniform01(seed, step, 100u + static_cast<uint32_t>(k));
            bool ok   = (qx > 0.0f) && (u < fminf(1.0f, px / qx));
            s_accept  = ok ? 1 : 0;
            if (ok) {
                step_out[1 + k] = x;
                s_n_acc = k + 1;
                if (x == eos_id) s_done = 1;
            }
        }
        __syncthreads();
        int accept_k = s_accept;
        int done_k   = s_done;
        __syncthreads();
        if (accept_k) {
            if (done_k) break;
            continue;
        }
        float r_sum = 0.0f;
        for (int i = tid; i < valid_vocab; i += 512) r_sum += fmaxf(0.0f, pk[i] - qk[i]);
        r_sum = wv_warp_sum_f32(r_sum);
        if (lane == 0) s_wv[wid] = r_sum;
        __syncthreads();
        float Z_res = 0.0f;
        for (int w = 0; w < 16; ++w) Z_res += s_wv[w];
        float u_res = wv_ink_rng_uniform01(seed, step, 200u + static_cast<uint32_t>(k));
        int tok_res;
        if (Z_res > 1e-12f) {
            tok_res = wv_ink_sample_cdf_cta512([&](int i) { return fmaxf(0.0f, pk[i] - qk[i]); }, valid_vocab, Z_res, u_res, 0);
        } else {
            tok_res = wv_ink_sample_cdf_cta512([&](int i) { return pk[i]; }, valid_vocab, 1.0f, u_res, 0);
        }
        if (tid == 0) {
            step_out[1 + k] = tok_res;
            s_n_acc = k + 1;
            s_done  = 1;
        }
        __syncthreads();
        break;
    }

    if (!s_done) {
        const float* p7 = p_probs + static_cast<size_t>(7) * stride_v;
        float u_bon = wv_ink_rng_uniform01(seed, step, 300u);
        int tok_bon = wv_ink_sample_cdf_cta512([&](int i) { return p7[i]; }, valid_vocab, 1.0f, u_bon, 0);
        if (tid == 0) {
            step_out[8] = tok_bon;
            s_n_acc = 8;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int n = s_n_acc;
        step_out[0]   = n;
        draft_blk8[0] = step_out[n];
    }
    wv_pdl_trigger();
}

// Parameterized S_ver (2..8, e.g. 4 for MTP-4 or 8 for MTP-8) Greedy GPU Verifier:
__global__ void wv_ink_mtpk_greedy_verify_kernel(
    int32_t* __restrict__ draft_blk8,
    const int32_t* __restrict__ target_k,
    const WvInkSampleCfg* __restrict__ cfg,
    int32_t* __restrict__ step_out,
    int S_ver) {
    wv_pdl_wait();
    if (threadIdx.x == 0) {
        int eos_id = cfg->eos_id;
        int n_acc  = 0;
        for (int k = 0; k < S_ver; ++k) {
            int32_t t = target_k[k];
            step_out[1 + k] = t;
            n_acc = k + 1;
            if (t == eos_id) break;
            if (k < (S_ver - 1) && draft_blk8[1 + k] != t) break;
        }
        step_out[0]   = n_acc;
        draft_blk8[0] = step_out[n_acc];
    }
    wv_pdl_trigger();
}

// Parameterized S_ver (2..8) Stochastic Speculative Rejection Sampler (0.0-ULP lossless target distribution):
__global__ void wv_ink_mtpk_spec_verify_kernel(
    int32_t* __restrict__ draft_blk8,
    const float* __restrict__ p_probs,
    const float* __restrict__ q_probs,
    const WvInkSampleCfg* __restrict__ cfg,
    int32_t* __restrict__ step_out,
    int stride_v,
    int valid_vocab,
    int S_ver) {
    wv_pdl_wait();
    int tid  = threadIdx.x;
    int lane = tid & 31, wid = tid >> 5;
    __shared__ float s_wv[16];
    __shared__ int   s_accept;
    __shared__ int   s_done;
    __shared__ int   s_n_acc;

    uint64_t seed = cfg->seed;
    uint64_t step = cfg->rng_step;
    int eos_id    = cfg->eos_id;
    int K_draft   = S_ver - 1;
    if (tid == 0) { s_done = 0; s_n_acc = 0; }
    __syncthreads();

    for (int k = 0; k < K_draft; ++k) {
        const float* pk = p_probs + static_cast<size_t>(k) * stride_v;
        const float* qk = q_probs + static_cast<size_t>(k) * stride_v;
        if (tid == 0) {
            int x    = draft_blk8[1 + k];
            float px = (x >= 0 && x < valid_vocab) ? pk[x] : 0.0f;
            float qx = (x >= 0 && x < valid_vocab) ? qk[x] : 0.0f;
            float u  = wv_ink_rng_uniform01(seed, step, 100u + static_cast<uint32_t>(k));
            bool ok  = (qx > 0.0f) && (u < fminf(1.0f, px / qx));
            s_accept = ok ? 1 : 0;
            if (ok) {
                step_out[1 + k] = x;
                s_n_acc = k + 1;
                if (x == eos_id) s_done = 1;
            }
        }
        __syncthreads();
        int accept_k = s_accept;
        int done_k   = s_done;
        __syncthreads();
        if (accept_k) {
            if (done_k) break;
            continue;
        }
        float r_sum = 0.0f;
        for (int i = tid; i < valid_vocab; i += 512) r_sum += fmaxf(0.0f, pk[i] - qk[i]);
        r_sum = wv_warp_sum_f32(r_sum);
        if (lane == 0) s_wv[wid] = r_sum;
        __syncthreads();
        float Z_res = 0.0f;
        for (int w = 0; w < 16; ++w) Z_res += s_wv[w];
        float u_res = wv_ink_rng_uniform01(seed, step, 200u + static_cast<uint32_t>(k));
        int tok_res;
        if (Z_res > 1e-12f) {
            tok_res = wv_ink_sample_cdf_cta512([&](int i) { return fmaxf(0.0f, pk[i] - qk[i]); }, valid_vocab, Z_res, u_res, 0);
        } else {
            tok_res = wv_ink_sample_cdf_cta512([&](int i) { return pk[i]; }, valid_vocab, 1.0f, u_res, 0);
        }
        if (tid == 0) {
            step_out[1 + k] = tok_res;
            s_n_acc = k + 1;
            s_done  = 1;
        }
        __syncthreads();
        break;
    }

    if (!s_done) {
        const float* plast = p_probs + static_cast<size_t>(K_draft) * stride_v;
        float u_bon = wv_ink_rng_uniform01(seed, step, 300u);
        int tok_bon = wv_ink_sample_cdf_cta512([&](int i) { return plast[i]; }, valid_vocab, 1.0f, u_bon, 0);
        if (tid == 0) {
            step_out[S_ver] = tok_bon;
            s_n_acc = S_ver;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int n = s_n_acc;
        step_out[0]   = n;
        draft_blk8[0] = step_out[n];
    }
    wv_pdl_trigger();
}

// Multi-step burst recorder for B=1 decode_m1 (4x fewer host cudaStreamSynchronize stalls!):
__global__ void wv_ink_m1_record_burst_step_kernel(
    const int32_t* __restrict__ ws_tok_in,
    int32_t* __restrict__ d_slot_active,
    const WvInkSampleCfg* __restrict__ cfg,
    int32_t* __restrict__ d_burst_out,
    int burst_idx) {
    wv_pdl_wait();
    if (threadIdx.x == 0) {
        int32_t tok = ws_tok_in[0];
        d_burst_out[burst_idx] = tok;
        if (tok == cfg->eos_id) {
            d_slot_active[0] = 0;
        }
    }
    wv_pdl_trigger();
}

} // namespace wv

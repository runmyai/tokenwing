#pragma once
#include "wv_inkling_types.cuh"

namespace wv {

// Block Attention-Residual (`n_blk <= 6`, `Q <= 4`): Grid <<<M, dim3(32, 6)>>>
// `partial_is_bf16 == 1` reads `partial_ptr` as `__nv_bfloat16*`; else `float*`.
__global__ void wv_ink_block_attn_res_kernel(
    const __nv_bfloat16* __restrict__ blocks_bf16, // [M, max_blk, 2304]
    const void* __restrict__ partial_ptr,          // [M, 2304]
    const float* __restrict__ query_f32,           // [Q, 2304]
    float* __restrict__ out_f32,                   // [M, Q, 2304]
    int n_blk,
    int max_blk,
    int Q,
    int partial_is_bf16,
    int M) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m >= M) { wv_pdl_trigger(); return; }
    int lane  = threadIdx.x;
    int wid   = threadIdx.y; // 0..5
    int n_src = n_blk + 1;   // <= 6

    __shared__ float s_wsum[6][6];
    __shared__ float s_dot[4][6][6];
    __shared__ float s_prob[4][6];

    float vec[6][3][4];
    #pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < n_src) {
            float sq = 0.0f;
            #pragma unroll
            for (int it = 0; it < 3; ++it) {
                int base = (wid + it * 6) * 128 + lane * 4;
                float x0, x1, x2, x3;
                if (s < n_blk) {
                    const __nv_bfloat16* bp = blocks_bf16 + (static_cast<size_t>(m) * max_blk + s) * INK2_HIDDEN + base;
                    x0 = __bfloat162float(bp[0]); x1 = __bfloat162float(bp[1]);
                    x2 = __bfloat162float(bp[2]); x3 = __bfloat162float(bp[3]);
                } else if (partial_is_bf16) {
                    const __nv_bfloat16* pp = static_cast<const __nv_bfloat16*>(partial_ptr) + static_cast<size_t>(m) * INK2_HIDDEN + base;
                    x0 = __bfloat162float(pp[0]); x1 = __bfloat162float(pp[1]);
                    x2 = __bfloat162float(pp[2]); x3 = __bfloat162float(pp[3]);
                } else {
                    const float4 pv = reinterpret_cast<const float4*>(static_cast<const float*>(partial_ptr) + static_cast<size_t>(m) * INK2_HIDDEN + base)[0];
                    x0 = pv.x; x1 = pv.y; x2 = pv.z; x3 = pv.w;
                }
                vec[s][it][0] = x0; vec[s][it][1] = x1; vec[s][it][2] = x2; vec[s][it][3] = x3;
                sq += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
            }
            sq = wv_warp_sum_f32(sq);
            if (lane == 0) s_wsum[s][wid] = sq;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < n_src) {
            float tot = s_wsum[s][0] + s_wsum[s][1] + s_wsum[s][2] + s_wsum[s][3] + s_wsum[s][4] + s_wsum[s][5];
            float inv = rsqrtf(tot * (1.0f / 2304.0f) + 1e-6f);
            #pragma unroll
            for (int it = 0; it < 3; ++it) {
                vec[s][it][0] *= inv; vec[s][it][1] *= inv; vec[s][it][2] *= inv; vec[s][it][3] *= inv;
            }
        }
    }

    constexpr float kInvSqrtD = 1.0f / 48.0f; // 2304^-0.5 == 1/48 exact!
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        if (q < Q) {
            const float* qp = query_f32 + static_cast<size_t>(q) * INK2_HIDDEN;
            float qv[3][4];
            #pragma unroll
            for (int it = 0; it < 3; ++it) {
                int base = (wid + it * 6) * 128 + lane * 4;
                float4 ql = reinterpret_cast<const float4*>(qp + base)[0];
                qv[it][0] = ql.x * kInvSqrtD; qv[it][1] = ql.y * kInvSqrtD;
                qv[it][2] = ql.z * kInvSqrtD; qv[it][3] = ql.w * kInvSqrtD;
            }
            #pragma unroll
            for (int s = 0; s < 6; ++s) {
                if (s < n_src) {
                    float d = 0.0f;
                    #pragma unroll
                    for (int it = 0; it < 3; ++it) {
                        d += qv[it][0] * vec[s][it][0] + qv[it][1] * vec[s][it][1]
                           + qv[it][2] * vec[s][it][2] + qv[it][3] * vec[s][it][3];
                    }
                    d = wv_warp_sum_f32(d);
                    if (lane == 0) s_dot[q][s][wid] = d;
                }
            }
        }
    }
    __syncthreads();

    if (wid == 0 && lane == 0) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (q < Q) {
                float mx = -1e30f;
                float lg[6];
                #pragma unroll
                for (int s = 0; s < 6; ++s) {
                    if (s < n_src) {
                        lg[s] = s_dot[q][s][0] + s_dot[q][s][1] + s_dot[q][s][2] + s_dot[q][s][3] + s_dot[q][s][4] + s_dot[q][s][5];
                        if (lg[s] > mx) mx = lg[s];
                    }
                }
                float sm = 0.0f;
                #pragma unroll
                for (int s = 0; s < 6; ++s) {
                    if (s < n_src) {
                        lg[s] = expf(lg[s] - mx);
                        sm += lg[s];
                    }
                }
                float inv_s = 1.0f / fmaxf(sm, 1e-20f);
                #pragma unroll
                for (int s = 0; s < 6; ++s) {
                    if (s < n_src) s_prob[q][s] = lg[s] * inv_s;
                }
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        if (q < Q) {
            float* dst = out_f32 + (static_cast<size_t>(q) * M + m) * INK2_HIDDEN;
            #pragma unroll
            for (int it = 0; it < 3; ++it) {
                float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
                #pragma unroll
                for (int s = 0; s < 6; ++s) {
                    if (s < n_src) {
                        float p = s_prob[q][s];
                        o0 += p * vec[s][it][0]; o1 += p * vec[s][it][1];
                        o2 += p * vec[s][it][2]; o3 += p * vec[s][it][3];
                    }
                }
                int base = (wid + it * 6) * 128 + lane * 4;
                reinterpret_cast<float4*>(dst + base)[0] = make_float4(o0, o1, o2, o3);
            }
        }
    }
    wv_pdl_trigger();
}

// Fused BF16 [M, 2304] -> RMSNorm(w_norm) -> BF16 `y_bf16` + FWHT-128 + Q8_32 (`v_idx == 0` skips `bf16_to_f32`!):
// Grid `<<<M, dim3(32, 6)>>>`
__global__ void wv_ink_bf16_rmsnorm_bf16_fwht_q8_kernel(
    const __nv_bfloat16* __restrict__ x_bf16,
    const float* __restrict__ w_norm,
    __nv_bfloat16* __restrict__ y_bf16,
    int8_t* __restrict__ Y_q8,
    float* __restrict__ Y_sc,
    int M) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m >= M) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x, wid = threadIdx.y;
    const __nv_bfloat16* xm = x_bf16 + static_cast<size_t>(m) * INK2_HIDDEN;
    __shared__ float s_wsum[6];
    float reg_x[3][4];
    float p_sq = 0.0f;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int base = (wid + it * 6) * 128 + lane * 4;
        float x0 = __bfloat162float(xm[base + 0]), x1 = __bfloat162float(xm[base + 1]);
        float x2 = __bfloat162float(xm[base + 2]), x3 = __bfloat162float(xm[base + 3]);
        reg_x[it][0] = x0; reg_x[it][1] = x1; reg_x[it][2] = x2; reg_x[it][3] = x3;
        p_sq += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
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
        float4 nw = __ldg(reinterpret_cast<const float4*>(w_norm + base));
        __nv_bfloat16 b0 = __float2bfloat16(reg_x[it][0] * inv_rms * nw.x);
        __nv_bfloat16 b1 = __float2bfloat16(reg_x[it][1] * inv_rms * nw.y);
        __nv_bfloat16 b2 = __float2bfloat16(reg_x[it][2] * inv_rms * nw.z);
        __nv_bfloat16 b3 = __float2bfloat16(reg_x[it][3] * inv_rms * nw.w);
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

// Fused `BlockAttnRes(Q=1)` + `RMSNorm(w_norm)` + `BF16` + `FWHT-128 + Q8_32` (`0.0-ULP` FP32->BF16 rounding preserved!):
// Grid `<<<M, dim3(32, 6)>>>`
__global__ void wv_ink_block_attn_res_fused_norm_q8_kernel(
    const __nv_bfloat16* __restrict__ blocks_bf16, // [M, max_blk, 2304]
    const void* __restrict__ partial_ptr,          // [M, 2304]
    const float* __restrict__ query_f32,           // [2304]
    const float* __restrict__ w_norm,              // [2304]
    __nv_bfloat16* __restrict__ y_bf16,            // [M, 2304] or nullptr
    int8_t* __restrict__ Y_q8,                     // [M, 2304] or nullptr
    float* __restrict__ Y_sc,                      // [M, 72] or nullptr
    int n_blk, int max_blk, int partial_is_bf16, int M) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m >= M) { wv_pdl_trigger(); return; }
    int lane  = threadIdx.x;
    int wid   = threadIdx.y; // 0..5
    int n_src = n_blk + 1;   // <= 6

    __shared__ float s_wsum[6][6];
    __shared__ float s_dot[6][6];
    __shared__ float s_prob[6];

    float vec[6][3][4];
    #pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < n_src) {
            float sq = 0.0f;
            #pragma unroll
            for (int it = 0; it < 3; ++it) {
                int base = (wid + it * 6) * 128 + lane * 4;
                float x0, x1, x2, x3;
                if (s < n_blk) {
                    const __nv_bfloat16* bp = blocks_bf16 + (static_cast<size_t>(m) * max_blk + s) * INK2_HIDDEN + base;
                    x0 = __bfloat162float(bp[0]); x1 = __bfloat162float(bp[1]);
                    x2 = __bfloat162float(bp[2]); x3 = __bfloat162float(bp[3]);
                } else if (partial_is_bf16) {
                    const __nv_bfloat16* pp = static_cast<const __nv_bfloat16*>(partial_ptr) + static_cast<size_t>(m) * INK2_HIDDEN + base;
                    x0 = __bfloat162float(pp[0]); x1 = __bfloat162float(pp[1]);
                    x2 = __bfloat162float(pp[2]); x3 = __bfloat162float(pp[3]);
                } else {
                    const float4 pv = reinterpret_cast<const float4*>(static_cast<const float*>(partial_ptr) + static_cast<size_t>(m) * INK2_HIDDEN + base)[0];
                    x0 = pv.x; x1 = pv.y; x2 = pv.z; x3 = pv.w;
                }
                vec[s][it][0] = x0; vec[s][it][1] = x1; vec[s][it][2] = x2; vec[s][it][3] = x3;
                sq += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
            }
            sq = wv_warp_sum_f32(sq);
            if (lane == 0) s_wsum[s][wid] = sq;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < n_src) {
            float tot = s_wsum[s][0] + s_wsum[s][1] + s_wsum[s][2] + s_wsum[s][3] + s_wsum[s][4] + s_wsum[s][5];
            float inv = rsqrtf(tot * (1.0f / 2304.0f) + 1e-6f);
            #pragma unroll
            for (int it = 0; it < 3; ++it) {
                vec[s][it][0] *= inv; vec[s][it][1] *= inv; vec[s][it][2] *= inv; vec[s][it][3] *= inv;
            }
        }
    }

    constexpr float kInvSqrtD = 1.0f / 48.0f;
    float qv[3][4];
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int base = (wid + it * 6) * 128 + lane * 4;
        float4 ql = reinterpret_cast<const float4*>(query_f32 + base)[0];
        qv[it][0] = ql.x * kInvSqrtD; qv[it][1] = ql.y * kInvSqrtD;
        qv[it][2] = ql.z * kInvSqrtD; qv[it][3] = ql.w * kInvSqrtD;
    }
    #pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < n_src) {
            float d = 0.0f;
            #pragma unroll
            for (int it = 0; it < 3; ++it) {
                d += qv[it][0] * vec[s][it][0] + qv[it][1] * vec[s][it][1]
                   + qv[it][2] * vec[s][it][2] + qv[it][3] * vec[s][it][3];
            }
            d = wv_warp_sum_f32(d);
            if (lane == 0) s_dot[s][wid] = d;
        }
    }
    __syncthreads();

    if (wid == 0 && lane == 0) {
        float mx = -1e30f;
        float lg[6];
        #pragma unroll
        for (int s = 0; s < 6; ++s) {
            if (s < n_src) {
                lg[s] = s_dot[s][0] + s_dot[s][1] + s_dot[s][2] + s_dot[s][3] + s_dot[s][4] + s_dot[s][5];
                if (lg[s] > mx) mx = lg[s];
            }
        }
        float sm = 0.0f;
        #pragma unroll
        for (int s = 0; s < 6; ++s) {
            if (s < n_src) { lg[s] = expf(lg[s] - mx); sm += lg[s]; }
        }
        float inv_s = 1.0f / fmaxf(sm, 1e-20f);
        #pragma unroll
        for (int s = 0; s < 6; ++s) {
            if (s < n_src) s_prob[s] = lg[s] * inv_s;
        }
    }
    __syncthreads();

    float reg_o[3][4];
    float p_sq = 0.0f;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        #pragma unroll
        for (int s = 0; s < 6; ++s) {
            if (s < n_src) {
                float p = s_prob[s];
                o0 += p * vec[s][it][0]; o1 += p * vec[s][it][1];
                o2 += p * vec[s][it][2]; o3 += p * vec[s][it][3];
            }
        }
        reg_o[it][0] = o0; reg_o[it][1] = o1; reg_o[it][2] = o2; reg_o[it][3] = o3;
        p_sq += o0 * o0 + o1 * o1 + o2 * o2 + o3 * o3;
    }
    p_sq = wv_warp_sum_f32(p_sq);
    if (lane == 0) s_wsum[0][wid] = p_sq;
    __syncthreads();
    float inv_rms = rsqrtf((s_wsum[0][0] + s_wsum[0][1] + s_wsum[0][2] + s_wsum[0][3] + s_wsum[0][4] + s_wsum[0][5]) * (1.0f / 2304.0f) + 1e-6f);

    __nv_bfloat16* yb = (y_bf16 != nullptr) ? (y_bf16 + static_cast<size_t>(m) * INK2_HIDDEN) : nullptr;
    int8_t* q8_m      = (Y_q8 != nullptr)   ? (Y_q8 + static_cast<size_t>(m) * INK2_HIDDEN)   : nullptr;
    float*  sc_m      = (Y_sc != nullptr)   ? (Y_sc + static_cast<size_t>(m) * 72)            : nullptr;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int b128 = wid + it * 6;
        int base = b128 * 128 + lane * 4;
        float4 nw = __ldg(reinterpret_cast<const float4*>(w_norm + base));
        __nv_bfloat16 b0 = __float2bfloat16(reg_o[it][0] * inv_rms * nw.x);
        __nv_bfloat16 b1 = __float2bfloat16(reg_o[it][1] * inv_rms * nw.y);
        __nv_bfloat16 b2 = __float2bfloat16(reg_o[it][2] * inv_rms * nw.z);
        __nv_bfloat16 b3 = __float2bfloat16(reg_o[it][3] * inv_rms * nw.w);
        if (yb != nullptr) { yb[base + 0] = b0; yb[base + 1] = b1; yb[base + 2] = b2; yb[base + 3] = b3; }
        if (q8_m != nullptr && sc_m != nullptr) {
            float f0 = __bfloat162float(b0), f1 = __bfloat162float(b1);
            float f2 = __bfloat162float(b2), f3 = __bfloat162float(b3);
            wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
            wv_quant_q8_32_warp(f0, f1, f2, f3, lane, q8_m + b128 * 128, sc_m + b128 * 4);
        }
    }
    wv_pdl_trigger();
}

struct WvInkNormPtrs3 {
    const float*   w[3];
    __nv_bfloat16* y[3];
    int8_t*        q8[3];
    float*         sc[3];
};

// 3-Way Batched Q/K/V Input RMSNorm + BF16 + FWHT-128 + Q8_32 (`0B` stack frame via scalar ternary selection!):
// Grid `<<<dim3(M, 3), dim3(32, 6)>>>`
__global__ void wv_ink_f32_rmsnorm_bf16_fwht_q8_3way_kernel(
    const float* __restrict__ x_f32,
    WvInkNormPtrs3 ptrs,
    int M) {
    wv_pdl_wait();
    int m  = blockIdx.x;
    int sl = blockIdx.y;
    if (m >= M || sl >= 3) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x, wid = threadIdx.y;
    const float* xm     = x_f32 + (static_cast<size_t>(sl) * M + m) * INK2_HIDDEN;
    const float* w_norm = (sl == 0) ? ptrs.w[0] : ((sl == 1) ? ptrs.w[1] : ptrs.w[2]);
    __nv_bfloat16* y_sl = (sl == 0) ? ptrs.y[0] : ((sl == 1) ? ptrs.y[1] : ptrs.y[2]);
    int8_t*       q8_sl = (sl == 0) ? ptrs.q8[0] : ((sl == 1) ? ptrs.q8[1] : ptrs.q8[2]);
    float*        sc_sl = (sl == 0) ? ptrs.sc[0] : ((sl == 1) ? ptrs.sc[1] : ptrs.sc[2]);
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

    __nv_bfloat16* yb = (y_sl != nullptr) ? (y_sl + static_cast<size_t>(m) * INK2_HIDDEN) : nullptr;
    int8_t* q8_m      = q8_sl + static_cast<size_t>(m) * INK2_HIDDEN;
    float*  sc_m      = sc_sl + static_cast<size_t>(m) * 72;
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
        if (yb != nullptr) { yb[base + 0] = b0; yb[base + 1] = b1; yb[base + 2] = b2; yb[base + 3] = b3; }
        float f0 = __bfloat162float(b0), f1 = __bfloat162float(b1);
        float f2 = __bfloat162float(b2), f3 = __bfloat162float(b3);
        wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
        wv_quant_q8_32_warp(f0, f1, f2, f3, lane, q8_m + b128 * 128, sc_m + b128 * 4);
    }
    wv_pdl_trigger();
}

struct WvInkCtrlPtrs3 {
    const __nv_bfloat16* p[3];
};

// Fused 2-Launch LoopMessageGate (`n_slots` = 3 for `q,k,v` or 1 for `mlp`, `0B` stack frame):
// Stage A (`<<<dim3(8, M * n_slots), dim3(32, 8)>>>`): normalizes `(src, msg)` into `9 KB` `s_cat[4608]` & computes `h64` in 1 pass!
__global__ void wv_ink_loop_gate_down_fused_kernel(
    const float* __restrict__ src_in,
    const float* __restrict__ messages_f32,
    WvInkCtrlPtrs3 w_down,
    float* __restrict__ h64_out,
    int first_msg_slot,
    int shared_src,
    int M) {
    wv_pdl_wait();
    int sm   = blockIdx.y; // 0 .. n_slots * M - 1
    int sl   = sm / M;
    int m    = sm - sl * M;
    int lane = threadIdx.x;
    int wid  = threadIdx.y;
    int tid  = wid * 32 + lane;
    int r    = blockIdx.x * 8 + wid; // 0..63

    __shared__ __align__(16) __nv_bfloat16 s_cat[4608];
    __shared__ float s_dsum[8], s_msum[8];

    const float* sp = src_in + (static_cast<size_t>(shared_src ? 0 : sl) * M + m) * INK2_HIDDEN;
    const float* mp = messages_f32 + (static_cast<size_t>(first_msg_slot + sl) * M + m) * INK2_HIDDEN;
    float d_sq = 0.0f, m_sq = 0.0f;
    for (int i = tid; i < INK2_HIDDEN; i += 256) {
        float dv = sp[i], mv = mp[i];
        d_sq += dv * dv;
        m_sq += mv * mv;
    }
    d_sq = wv_warp_sum_f32(d_sq);
    m_sq = wv_warp_sum_f32(m_sq);
    if (lane == 0) { s_dsum[wid] = d_sq; s_msum[wid] = m_sq; }
    __syncthreads();

    float inv_d = rsqrtf((s_dsum[0]+s_dsum[1]+s_dsum[2]+s_dsum[3]+s_dsum[4]+s_dsum[5]+s_dsum[6]+s_dsum[7]) * (1.0f / 2304.0f) + 1e-6f);
    float inv_m = rsqrtf((s_msum[0]+s_msum[1]+s_msum[2]+s_msum[3]+s_msum[4]+s_msum[5]+s_msum[6]+s_msum[7]) * (1.0f / 2304.0f) + 1e-6f);
    for (int i = tid; i < INK2_HIDDEN; i += 256) {
        s_cat[i]               = __float2bfloat16(sp[i] * inv_d);
        s_cat[INK2_HIDDEN + i] = __float2bfloat16(mp[i] * inv_m);
    }
    __syncthreads();

    const __nv_bfloat16* w_sl = (sl == 0) ? w_down.p[0] : ((sl == 1) ? w_down.p[1] : w_down.p[2]);
    const __nv_bfloat16* wr   = w_sl + static_cast<size_t>(r) * 4608;
    float acc = 0.0f;
    #pragma unroll 6
    for (int k = lane * 4; k < 4608; k += 128) {
        uint2 wv_tmp = __ldg(reinterpret_cast<const uint2*>(wr + k));
        uint2 cv = *reinterpret_cast<const uint2*>(s_cat + k);
        const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv_tmp);
        const __nv_bfloat162* cb = reinterpret_cast<const __nv_bfloat162*>(&cv);
        float2 w0 = __bfloat1622float2(wb[0]), w1 = __bfloat1622float2(wb[1]);
        float2 c0 = __bfloat1622float2(cb[0]), c1 = __bfloat1622float2(cb[1]);
        acc += w0.x * c0.x + w0.y * c0.y + w1.x * c1.x + w1.y * c1.y;
    }
    acc = wv_warp_sum_f32(acc);
    if (lane == 0) h64_out[static_cast<size_t>(sm) * 64 + r] = acc;
    wv_pdl_trigger();
}

// Stage B (`<<<dim3(144, M, n_slots), 256>>>`, `0B` stack frame): 16-lane coalesced 128B row loads for `w_forget` & `w_gain` (`[2304, 64]`)!
__global__ void wv_ink_loop_gate_apply_kernel(
    const float* __restrict__ src_in,
    float* __restrict__ dst_out,
    const float* __restrict__ messages_f32,
    const float* __restrict__ h64_in,
    WvInkCtrlPtrs3 w_forget,
    WvInkCtrlPtrs3 w_gain,
    int first_msg_slot,
    int shared_src,
    int M) {
    wv_pdl_wait();
    int m   = blockIdx.y;
    int sl  = blockIdx.z;
    if (m >= M) { wv_pdl_trigger(); return; }
    int tid = threadIdx.x;
    int sm  = sl * M + m;

    __shared__ float s_h64[64];
    if (tid < 64) s_h64[tid] = h64_in[static_cast<size_t>(sm) * 64 + tid];
    __syncthreads();

    int grp      = tid >> 4;  // 0..15 (16 channels per CTA; 144 * 16 == 2304!)
    int sub_lane = tid & 15;  // 0..15 (16 * 4 bf16 = 64 bf16 = 128B coalesced!)
    int i        = blockIdx.x * 16 + grp;
    int k        = sub_lane * 4;

    const __nv_bfloat16* wf_sl = (sl == 0) ? w_forget.p[0] : ((sl == 1) ? w_forget.p[1] : w_forget.p[2]);
    const __nv_bfloat16* wg_sl = (sl == 0) ? w_gain.p[0]   : ((sl == 1) ? w_gain.p[1]   : w_gain.p[2]);
    uint2 fv = __ldg(reinterpret_cast<const uint2*>(wf_sl + static_cast<size_t>(i) * 64 + k));
    uint2 gv = __ldg(reinterpret_cast<const uint2*>(wg_sl + static_cast<size_t>(i) * 64 + k));
    float4 hv = *reinterpret_cast<const float4*>(s_h64 + k);
    const __nv_bfloat162* fb = reinterpret_cast<const __nv_bfloat162*>(&fv);
    const __nv_bfloat162* gb = reinterpret_cast<const __nv_bfloat162*>(&gv);
    float2 f0 = __bfloat1622float2(fb[0]), f1 = __bfloat1622float2(fb[1]);
    float2 g0 = __bfloat1622float2(gb[0]), g1 = __bfloat1622float2(gb[1]);

    float f_acc = f0.x * hv.x + f0.y * hv.y + f1.x * hv.z + f1.y * hv.w;
    float g_acc = g0.x * hv.x + g0.y * hv.y + g1.x * hv.z + g1.y * hv.w;
    #pragma unroll
    for (int mask = 8; mask >= 1; mask >>= 1) {
        f_acc += __shfl_xor_sync(0xffffffffu, f_acc, mask);
        g_acc += __shfl_xor_sync(0xffffffffu, g_acc, mask);
    }
    if (sub_lane == 0) {
        const float* sp = src_in + (static_cast<size_t>(shared_src ? 0 : sl) * M + m) * INK2_HIDDEN;
        const float* mp = messages_f32 + (static_cast<size_t>(first_msg_slot + sl) * M + m) * INK2_HIDDEN;
        float*       dp = dst_out + static_cast<size_t>(sm) * INK2_HIDDEN;
        dp[i] = (1.0f - tanhf(f_acc)) * sp[i] + tanhf(g_acc) * mp[i];
    }
    wv_pdl_trigger();
}

// 16-Token Weight-Reusing LoopMessageGate Stage B (`8x` fewer DRAM weight reads in MTP-8, `16x` fewer in Prefill!):
// Grid `<<<dim3(144, (M + 15) / 16, n_slots), 256>>>`
__global__ void wv_ink_loop_gate_apply_tiled_kernel(
    const float* __restrict__ src_in,
    float* __restrict__ dst_out,
    const float* __restrict__ messages_f32,
    const float* __restrict__ h64_in,
    WvInkCtrlPtrs3 w_forget,
    WvInkCtrlPtrs3 w_gain,
    int first_msg_slot,
    int shared_src,
    int M) {
    wv_pdl_wait();
    int m_base = blockIdx.y * 16;
    int cur_m  = min(16, M - m_base);
    int sl     = blockIdx.z;
    if (cur_m <= 0) { wv_pdl_trigger(); return; }
    int tid    = threadIdx.x;

    __shared__ float s_h64[16][64];
    for (int idx = tid; idx < cur_m * 64; idx += 256) {
        int tm = idx >> 6;
        int c  = idx & 63;
        s_h64[tm][c] = h64_in[static_cast<size_t>(sl * M + m_base + tm) * 64 + c];
    }
    __syncthreads();

    int grp      = tid >> 4;  // 0..15
    int sub_lane = tid & 15;  // 0..15
    int i        = blockIdx.x * 16 + grp;
    int k        = sub_lane * 4;

    const __nv_bfloat16* wf_sl = (sl == 0) ? w_forget.p[0] : ((sl == 1) ? w_forget.p[1] : w_forget.p[2]);
    const __nv_bfloat16* wg_sl = (sl == 0) ? w_gain.p[0]   : ((sl == 1) ? w_gain.p[1]   : w_gain.p[2]);
    uint2 fv = __ldg(reinterpret_cast<const uint2*>(wf_sl + static_cast<size_t>(i) * 64 + k));
    uint2 gv = __ldg(reinterpret_cast<const uint2*>(wg_sl + static_cast<size_t>(i) * 64 + k));
    const __nv_bfloat162* fb = reinterpret_cast<const __nv_bfloat162*>(&fv);
    const __nv_bfloat162* gb = reinterpret_cast<const __nv_bfloat162*>(&gv);
    float2 f0 = __bfloat1622float2(fb[0]), f1 = __bfloat1622float2(fb[1]);
    float2 g0 = __bfloat1622float2(gb[0]), g1 = __bfloat1622float2(gb[1]);

    for (int tm = 0; tm < cur_m; ++tm) {
        float4 hv = *reinterpret_cast<const float4*>(s_h64[tm] + k);
        float f_acc = f0.x * hv.x + f0.y * hv.y + f1.x * hv.z + f1.y * hv.w;
        float g_acc = g0.x * hv.x + g0.y * hv.y + g1.x * hv.z + g1.y * hv.w;
        #pragma unroll
        for (int mask = 8; mask >= 1; mask >>= 1) {
            f_acc += __shfl_xor_sync(0xffffffffu, f_acc, mask);
            g_acc += __shfl_xor_sync(0xffffffffu, g_acc, mask);
        }
        if (sub_lane == 0) {
            int m = m_base + tm;
            int sm = sl * M + m;
            const float* sp = src_in + (static_cast<size_t>(shared_src ? 0 : sl) * M + m) * INK2_HIDDEN;
            const float* mp = messages_f32 + (static_cast<size_t>(first_msg_slot + sl) * M + m) * INK2_HIDDEN;
            float*       dp = dst_out + static_cast<size_t>(sm) * INK2_HIDDEN;
            dp[i] = (1.0f - tanhf(f_acc)) * sp[i] + tanhf(g_acc) * mp[i];
        }
    }
    wv_pdl_trigger();
}

static inline void launch_wv_ink_loop_msg_gate(
    const float* src_in, float* dst_out, const float* messages_f32,
    float* ws_h64,
    WvInkCtrlPtrs3 w_down, WvInkCtrlPtrs3 w_forget, WvInkCtrlPtrs3 w_gain,
    int first_msg_slot, int n_slots, int shared_src, int M, cudaStream_t stream, bool enable_pdl = false) {
    wv_launch_pdl(enable_pdl, dim3(8, M * n_slots), dim3(32, 8), 0, stream,
        wv_ink_loop_gate_down_fused_kernel, src_in, messages_f32, w_down, ws_h64, first_msg_slot, shared_src, M);
    if (M <= 1) {
        wv_launch_pdl(enable_pdl, dim3(144, M, n_slots), dim3(256), 0, stream,
            wv_ink_loop_gate_apply_kernel, src_in, dst_out, messages_f32, ws_h64, w_forget, w_gain, first_msg_slot, shared_src, M);
    } else {
        wv_launch_pdl(enable_pdl, dim3(144, (M + 15) / 16, n_slots), dim3(256), 0, stream,
            wv_ink_loop_gate_apply_tiled_kernel, src_in, dst_out, messages_f32, ws_h64, w_forget, w_gain, first_msg_slot, shared_src, M);
    }
}


// Fast warp-per-row BF16 small linear `y [M, N] = x_bf16 [M, K] @ W_bf16 [N, K]^T`:
// Grid <<<(N + 7) / 8, M>>>, Block dim3(32, 8)
template<bool OUT_BF16>
__global__ void wv_ink_bf16_small_linear_kernel(
    const __nv_bfloat16* __restrict__ x_bf16, // [M, K]
    const __nv_bfloat16* __restrict__ w_bf16, // [N, K]
    void* __restrict__ y_out,                 // [M, N]
    int K, int N, int M) {
    wv_pdl_wait();
    int n = blockIdx.x * 8 + threadIdx.y;
    int m = blockIdx.y;
    if (n >= N || m >= M) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x;
    const __nv_bfloat16* xm = x_bf16 + static_cast<size_t>(m) * K;
    const __nv_bfloat16* wn = w_bf16 + static_cast<size_t>(n) * K;
    float acc = 0.0f;
    for (int k = lane * 4; k < K; k += 128) {
        uint2 xv = *reinterpret_cast<const uint2*>(xm + k);
        uint2 wv_tmp = *reinterpret_cast<const uint2*>(wn + k);
        const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
        const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv_tmp);
        float2 x0 = __bfloat1622float2(xb[0]), x1 = __bfloat1622float2(xb[1]);
        float2 w0 = __bfloat1622float2(wb[0]), w1 = __bfloat1622float2(wb[1]);
        acc += x0.x * w0.x + x0.y * w0.y + x1.x * w1.x + x1.y * w1.y;
    }
    acc = wv_warp_sum_f32(acc);
    if (lane == 0) {
        size_t ix = static_cast<size_t>(m) * N + n;
        if constexpr (OUT_BF16) {
            static_cast<__nv_bfloat16*>(y_out)[ix] = __float2bfloat16(acc);
        } else {
            static_cast<float*>(y_out)[ix] = acc;
        }
    }
    wv_pdl_trigger();
}

// Static ShortConvolution `(conv1d_k4(x) + x) / sqrt(5)` over `[B, S, C]`:
// `w` has shape `[C, 4]` (`w[c, 3]` = lag 0 current token, `w[c, 0]` = lag 3!).
// `conv_state` has shape `[B, 3, C]` (`state[b, 2, c]` = lag 1, `state[b, 0, c]` = lag 3!).
// Grid <<<(C + 255) / 256, B>>>, Block 256
__global__ void wv_ink_static_sconv_kernel(
    const __nv_bfloat16* __restrict__ x_in,   // [B, S, C]
    const float* __restrict__ w_k4,           // [C, 4]
    float* __restrict__ conv_state,           // [b_max, 3, C]
    float* __restrict__ cand_save,            // [b_max, 8, C] if commit_now == 0 else nullptr
    const int32_t* __restrict__ d_slot_map,   // [B] or nullptr
    __nv_bfloat16* __restrict__ y_out,        // [B, S, C]
    int B, int S, int C,
    int commit_now) {
    wv_pdl_wait();
    int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int b = blockIdx.y;
    if (c >= C || b >= B || S <= 0) { wv_pdl_trigger(); return; }
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;

    float4 w4 = reinterpret_cast<const float4*>(w_k4 + static_cast<size_t>(c) * 4)[0];
    float* st_b = conv_state + static_cast<size_t>(phys_b) * 3 * C;
    float h0 = st_b[0 * C + c];
    float h1 = st_b[1 * C + c];
    float h2 = st_b[2 * C + c];
    constexpr float kInvSqrt5 = 0.44721359549995793928f;

    const __nv_bfloat16* xb = x_in  + static_cast<size_t>(b) * S * C;
    __nv_bfloat16*       yb = y_out + static_cast<size_t>(b) * S * C;
    float*               cb = (cand_save != nullptr) ? (cand_save + static_cast<size_t>(phys_b) * 8 * C) : nullptr;

    for (int s = 0; s < S; ++s) {
        float x_cur = __bfloat162float(xb[static_cast<size_t>(s) * C + c]);
        if (cb != nullptr && s < 8) {
            cb[s * C + c] = x_cur;
        }
        float conv = w4.x * h0 + w4.y * h1 + w4.z * h2 + w4.w * x_cur;
        __nv_bfloat16 ybf = __float2bfloat16((conv + x_cur) * kInvSqrt5);
        yb[static_cast<size_t>(s) * C + c] = ybf;
        h0 = h1; h1 = h2; h2 = x_cur;
    }
    if (commit_now) {
        st_b[0 * C + c] = h0;
        st_b[1 * C + c] = h1;
        st_b[2 * C + c] = h2;
    }
    wv_pdl_trigger();
}

// Dynamic ShortConvolution `(dyn_conv_k4(x) + x) / 3.0` over `[B, S, 1024]` (`8` heads x `128`):
// Grid <<<4, B>>>, Block 256 (`C = 1024`)
__global__ void wv_ink_dynamic_sconv_kernel(
    const __nv_bfloat16* __restrict__ x_in,   // [B, S, 1024]
    const float* __restrict__ dw_f32,         // [B, S, 32]
    const float* __restrict__ static_w,       // [4, 1024]
    float* __restrict__ conv_state,           // [b_max, 3, 1024]
    float* __restrict__ cand_save,            // [b_max, 8, 1024] if commit_now == 0 else nullptr
    const int32_t* __restrict__ d_slot_map,   // [B] or nullptr
    __nv_bfloat16* __restrict__ y_out,        // [B, S, 1024]
    int B, int S,
    int commit_now) {
    wv_pdl_wait();
    int c = blockIdx.x * 256 + threadIdx.x; // 0..1023
    int b = blockIdx.y;
    if (c >= 1024 || b >= B || S <= 0) { wv_pdl_trigger(); return; }
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    int h = c >> 7; // 0..7

    float sw0 = static_w[0 * 1024 + c];
    float sw1 = static_w[1 * 1024 + c];
    float sw2 = static_w[2 * 1024 + c];
    float sw3 = static_w[3 * 1024 + c];

    float* st_b = conv_state + static_cast<size_t>(phys_b) * 3 * 1024;
    float lag3 = st_b[0 * 1024 + c];
    float lag2 = st_b[1 * 1024 + c];
    float lag1 = st_b[2 * 1024 + c];

    const __nv_bfloat16* xb = x_in  + static_cast<size_t>(b) * S * 1024;
    __nv_bfloat16*       yb = y_out + static_cast<size_t>(b) * S * 1024;
    float*               cb = (cand_save != nullptr) ? (cand_save + static_cast<size_t>(phys_b) * 8 * 1024) : nullptr;

    for (int s = 0; s < S; ++s) {
        float x_cur = __bfloat162float(xb[static_cast<size_t>(s) * 1024 + c]);
        if (cb != nullptr && s < 8) {
            cb[s * 1024 + c] = x_cur;
        }
        const float4 dw = reinterpret_cast<const float4*>(dw_f32 + (static_cast<size_t>(b * S + s) * 8 + h) * 4)[0];
        float y = (sw0 + dw.x) * x_cur + (sw1 + dw.y) * lag1 + (sw2 + dw.z) * lag2 + (sw3 + dw.w) * lag3;
        __nv_bfloat16 ybf = __float2bfloat16((y + x_cur) * (1.0f / 3.0f));
        yb[static_cast<size_t>(s) * 1024 + c] = ybf;
        lag3 = lag2; lag2 = lag1; lag1 = x_cur;
    }
    if (commit_now) {
        st_b[0 * 1024 + c] = lag3;
        st_b[1 * 1024 + c] = lag2;
        st_b[2 * 1024 + c] = lag1;
    }
    wv_pdl_trigger();
}

// Fuses ALL `48 * 4 = 192` SConv candidate commits into ONE kernel launch (`dim3(26, 48), 256`):
// `bx` in `[0,4)`: `sc_k` (`C=1024`), `[4,8)`: `sc_v` (`C=1024`), `[8,17)`: `sc_a` (`C=2304`), `[17,26)`: `sc_m` (`C=2304`)!
__global__ void wv_ink_commit_all_sconv_cand_kernel(
    float* __restrict__ sc_k_all,    const float* __restrict__ cand_k_all,
    float* __restrict__ sc_v_all,    const float* __restrict__ cand_v_all,
    float* __restrict__ sc_a_all,    const float* __restrict__ cand_a_all,
    float* __restrict__ sc_m_all,    const float* __restrict__ cand_m_all,
    const int32_t* __restrict__ d_n_acc,
    int b_max,
    int n_visits) {
    wv_pdl_wait();
    int n_acc = *d_n_acc;
    int v_idx = blockIdx.y;
    if (n_acc <= 0 || v_idx >= n_visits) { wv_pdl_trigger(); return; }
    int bx    = blockIdx.x; // 0..25
    float*       st_p;
    const float* cd_p;
    int c, C;
    if (bx < 4) {
        C = 1024; c = bx * 256 + threadIdx.x;
        st_p = sc_k_all   + static_cast<size_t>(v_idx) * b_max * 3 * 1024;
        cd_p = cand_k_all + static_cast<size_t>(v_idx) * b_max * 8 * 1024;
    } else if (bx < 8) {
        C = 1024; c = (bx - 4) * 256 + threadIdx.x;
        st_p = sc_v_all   + static_cast<size_t>(v_idx) * b_max * 3 * 1024;
        cd_p = cand_v_all + static_cast<size_t>(v_idx) * b_max * 8 * 1024;
    } else if (bx < 17) {
        C = INK2_HIDDEN; c = (bx - 8) * 256 + threadIdx.x;
        st_p = sc_a_all   + static_cast<size_t>(v_idx) * b_max * 3 * INK2_HIDDEN;
        cd_p = cand_a_all + static_cast<size_t>(v_idx) * b_max * 8 * INK2_HIDDEN;
    } else {
        C = INK2_HIDDEN; c = (bx - 17) * 256 + threadIdx.x;
        st_p = sc_m_all   + static_cast<size_t>(v_idx) * b_max * 3 * INK2_HIDDEN;
        cd_p = cand_m_all + static_cast<size_t>(v_idx) * b_max * 8 * INK2_HIDDEN;
    }
    if (c >= C) { wv_pdl_trigger(); return; }
    float h0 = st_p[0 * C + c], h1 = st_p[1 * C + c], h2 = st_p[2 * C + c];
    for (int s = 0; s < n_acc; ++s) {
        float x = cd_p[s * C + c];
        h0 = h1; h1 = h2; h2 = x;
    }
    st_p[0 * C + c] = h0; st_p[1 * C + c] = h1; st_p[2 * C + c] = h2;
    wv_pdl_trigger();
}

// Fused K+V Static ShortConvolution (`is_local && !(S == 1 && commit_sconv)`): Grid `<<<dim3(8, B_act), 256>>>`
__global__ void wv_ink_kv_static_sconv_kernel(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const float* __restrict__ k_w4,
    const float* __restrict__ v_w4,
    float* __restrict__ k_state,
    float* __restrict__ v_state,
    float* __restrict__ k_cand,
    float* __restrict__ v_cand,
    const int32_t* __restrict__ d_slot_map,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int B, int S, int commit_now) {
    wv_pdl_wait();
    bool is_v = (blockIdx.x >= 4);
    int c = (is_v ? (blockIdx.x - 4) : blockIdx.x) * 256 + threadIdx.x;
    int b = blockIdx.y;
    if (c >= 1024 || b >= B || S <= 0) { wv_pdl_trigger(); return; }
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    float4 w4 = reinterpret_cast<const float4*>((is_v ? v_w4 : k_w4) + static_cast<size_t>(c) * 4)[0];
    float* st_b = (is_v ? v_state : k_state) + static_cast<size_t>(phys_b) * 3 * 1024;
    float h0 = st_b[0 * 1024 + c], h1 = st_b[1 * 1024 + c], h2 = st_b[2 * 1024 + c];
    constexpr float kInvSqrt5 = 0.44721359549995793928f;
    const __nv_bfloat16* xb = (is_v ? v_in : k_in) + static_cast<size_t>(b) * S * 1024;
    __nv_bfloat16*       yb = (is_v ? v_out : k_out) + static_cast<size_t>(b) * S * 1024;
    float* cand_p = is_v ? v_cand : k_cand;
    float* cb = (cand_p != nullptr) ? (cand_p + static_cast<size_t>(phys_b) * 8 * 1024) : nullptr;
    for (int s = 0; s < S; ++s) {
        float x_cur = __bfloat162float(xb[static_cast<size_t>(s) * 1024 + c]);
        if (cb != nullptr && s < 8) cb[s * 1024 + c] = x_cur;
        float conv = w4.x * h0 + w4.y * h1 + w4.z * h2 + w4.w * x_cur;
        __nv_bfloat16 ybf = __float2bfloat16((conv + x_cur) * kInvSqrt5);
        yb[static_cast<size_t>(s) * 1024 + c] = ybf;
        h0 = h1; h1 = h2; h2 = x_cur;
    }
    if (commit_now) {
        st_b[0 * 1024 + c] = h0; st_b[1 * 1024 + c] = h1; st_b[2 * 1024 + c] = h2;
    }
    wv_pdl_trigger();
}

// Fused K+V Dynamic SConv Weight Projection (`k_wproj + v_wproj`): Grid `<<<dim3(8, M), dim3(32, 8)>>>` -> `dw_out [M, 64]`
__global__ void wv_ink_kv_wproj_kernel(
    const __nv_bfloat16* __restrict__ kin_bf16,
    const __nv_bfloat16* __restrict__ vin_bf16,
    const __nv_bfloat16* __restrict__ k_wproj,
    const __nv_bfloat16* __restrict__ v_wproj,
    float* __restrict__ dw_out, // [M, 64] (`0..31` K, `32..63` V)
    int M) {
    wv_pdl_wait();
    bool is_v = (blockIdx.x >= 4);
    int n     = (is_v ? (blockIdx.x - 4) : blockIdx.x) * 8 + threadIdx.y; // 0..31
    int m     = blockIdx.y;
    if (m >= M) { wv_pdl_trigger(); return; }
    int lane  = threadIdx.x;
    const __nv_bfloat16* xm = (is_v ? vin_bf16 : kin_bf16) + static_cast<size_t>(m) * INK2_HIDDEN;
    const __nv_bfloat16* wn = (is_v ? v_wproj  : k_wproj)  + static_cast<size_t>(n) * INK2_HIDDEN;
    float acc = 0.0f;
    for (int k = lane * 4; k < INK2_HIDDEN; k += 128) {
        uint2 xv = *reinterpret_cast<const uint2*>(xm + k);
        uint2 wv_tmp = *reinterpret_cast<const uint2*>(wn + k);
        const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
        const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv_tmp);
        float2 x0 = __bfloat1622float2(xb[0]), x1 = __bfloat1622float2(xb[1]);
        float2 w0 = __bfloat1622float2(wb[0]), w1 = __bfloat1622float2(wb[1]);
        acc += x0.x * w0.x + x0.y * w0.y + x1.x * w1.x + x1.y * w1.y;
    }
    acc = wv_warp_sum_f32(acc);
    if (lane == 0) dw_out[static_cast<size_t>(m) * 64 + (is_v ? 32 : 0) + n] = acc;
    wv_pdl_trigger();
}

// Fused K+V Dynamic ShortConvolution (`!is_local`): Grid `<<<dim3(8, B_act), 256>>>`
__global__ void wv_ink_kv_dynamic_sconv_kernel(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const float* __restrict__ dw_f32, // [M, 64]
    const float* __restrict__ k_static_w,
    const float* __restrict__ v_static_w,
    float* __restrict__ k_state,
    float* __restrict__ v_state,
    float* __restrict__ k_cand,
    float* __restrict__ v_cand,
    const int32_t* __restrict__ d_slot_map,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int B, int S, int commit_now) {
    wv_pdl_wait();
    bool is_v = (blockIdx.x >= 4);
    int c = (is_v ? (blockIdx.x - 4) : blockIdx.x) * 256 + threadIdx.x;
    int b = blockIdx.y;
    if (c >= 1024 || b >= B || S <= 0) { wv_pdl_trigger(); return; }
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    int h = c >> 7;
    const float* static_w = is_v ? v_static_w : k_static_w;
    float sw0 = static_w[0 * 1024 + c], sw1 = static_w[1 * 1024 + c];
    float sw2 = static_w[2 * 1024 + c], sw3 = static_w[3 * 1024 + c];
    float* st_b = (is_v ? v_state : k_state) + static_cast<size_t>(phys_b) * 3 * 1024;
    float lag3 = st_b[0 * 1024 + c], lag2 = st_b[1 * 1024 + c], lag1 = st_b[2 * 1024 + c];
    const __nv_bfloat16* xb = (is_v ? v_in : k_in) + static_cast<size_t>(b) * S * 1024;
    __nv_bfloat16*       yb = (is_v ? v_out : k_out) + static_cast<size_t>(b) * S * 1024;
    float* cand_p = is_v ? v_cand : k_cand;
    float* cb = (cand_p != nullptr) ? (cand_p + static_cast<size_t>(phys_b) * 8 * 1024) : nullptr;
    for (int s = 0; s < S; ++s) {
        float x_cur = __bfloat162float(xb[static_cast<size_t>(s) * 1024 + c]);
        if (cb != nullptr && s < 8) cb[s * 1024 + c] = x_cur;
        const float4 dw = reinterpret_cast<const float4*>(dw_f32 + static_cast<size_t>(b * S + s) * 64 + (is_v ? 32 : 0) + h * 4)[0];
        float y = (sw0 + dw.x) * x_cur + (sw1 + dw.y) * lag1 + (sw2 + dw.z) * lag2 + (sw3 + dw.w) * lag3;
        __nv_bfloat16 ybf = __float2bfloat16((y + x_cur) * (1.0f / 3.0f));
        yb[static_cast<size_t>(s) * 1024 + c] = ybf;
        lag3 = lag2; lag2 = lag1; lag1 = x_cur;
    }
    if (commit_now) {
        st_b[0 * 1024 + c] = lag3; st_b[1 * 1024 + c] = lag2; st_b[2 * 1024 + c] = lag1;
    }
    wv_pdl_trigger();
}


__device__ __forceinline__ uint32_t wv_ink_pack_q8_head128_warp(
    float x0, float x1, float x2, float x3, float& out_sc) {
    float amax = wv_warp_max_f32(fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3))));
    out_sc = amax * (1.0f / 127.0f);
    float inv = (amax > 1e-30f) ? (127.0f / amax) : 0.0f;
    auto q1 = [&](float v) -> uint32_t {
        int qi = __float2int_rn(v * inv);
        qi = max(-127, min(127, qi));
        return static_cast<uint32_t>(static_cast<int8_t>(qi)) & 0xFFu;
    };
    return q1(x0) | (q1(x1) << 8) | (q1(x2) << 16) | (q1(x3) << 24);
}

__device__ __forceinline__ void wv_ink_unpack_s8x4_f32(
    uint32_t pk, float& y0, float& y1, float& y2, float& y3) {
    y0 = static_cast<float>(static_cast<int8_t>(pk & 0xFFu));
    y1 = static_cast<float>(static_cast<int8_t>((pk >> 8) & 0xFFu));
    y2 = static_cast<float>(static_cast<int8_t>((pk >> 16) & 0xFFu));
    y3 = static_cast<float>(static_cast<int8_t>((pk >> 24) & 0xFFu));
}

// Applies Q/K/Rel L2-clips, V RMSNorm, optional Global log-length scaling, and writes K/V into visit cache:
// Grid <<<M, dim3(32, 8)>>> (`M = B * S`)
__global__ void wv_ink_prep_qkv_rel_kvcache_kernel(
    const __nv_bfloat16* __restrict__ q_raw,      // [B, S, 3072]
    const __nv_bfloat16* __restrict__ k_sconv,    // [B, S, 1024]
    const __nv_bfloat16* __restrict__ v_sconv,    // [B, S, 1024]
    const __nv_bfloat16* __restrict__ r_raw,      // [B, S, 384]
    const float* __restrict__ q_clip_scale,       // [3072]
    const float* __restrict__ r_clip_scale,       // [384]
    __nv_bfloat16* __restrict__ q_out,            // [B, S, 24, 128]
    float* __restrict__ r_out_f32,                // [B, S, 24, 16]
    void* __restrict__ k_cache,                   // [B_max, kv_cap, 8, 128] bf16 or int8
    void* __restrict__ v_cache,                   // [B_max, kv_cap, 8, 128] bf16 or int8
    float* __restrict__ k_cache_sc,               // [B_max, kv_cap, 8] or nullptr
    float* __restrict__ v_cache_sc,               // [B_max, kv_cap, 8] or nullptr
    const int32_t* __restrict__ d_slot_pos,       // [B_max]
    const int32_t* __restrict__ d_slot_map,       // [B] or nullptr
    int S, int kv_cap, int is_local, int kv_is_q8, int M) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m >= M) { wv_pdl_trigger(); return; }
    int b      = m / S;
    int s      = m - b * S;
    int lane   = threadIdx.x; // 0..31
    int wid    = threadIdx.y; // 0..7
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    int q_pos  = d_slot_pos[phys_b] + s;
    float tau  = is_local ? 1.0f : (1.0f + 0.1f * logf(fmaxf(1.0f, static_cast<float>(q_pos + 1) * (1.0f / 8192.0f))));
    constexpr float kClipQK = 22.627416997969522f;

    // 1. Q heads:
    for (int sub = 0; sub < 3; ++sub) {
        int hq = wid * 3 + sub;
        int off = hq * 128 + lane * 4;
        const __nv_bfloat16* qp = q_raw + static_cast<size_t>(m) * 3072 + off;
        float4 sc = reinterpret_cast<const float4*>(q_clip_scale + off)[0];
        float f0 = __bfloat162float(qp[0]) * sc.x;
        float f1 = __bfloat162float(qp[1]) * sc.y;
        float f2 = __bfloat162float(qp[2]) * sc.z;
        float f3 = __bfloat162float(qp[3]) * sc.w;
        float nrm = sqrtf(wv_warp_sum_f32(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3));
        float cf  = fminf(1.0f, kClipQK / (nrm + 1e-6f));
        __nv_bfloat16 b0 = __float2bfloat16(f0 * cf), b1 = __float2bfloat16(f1 * cf);
        __nv_bfloat16 b2 = __float2bfloat16(f2 * cf), b3 = __float2bfloat16(f3 * cf);
        if (!is_local) {
            b0 = __float2bfloat16(__bfloat162float(b0) * tau);
            b1 = __float2bfloat16(__bfloat162float(b1) * tau);
            b2 = __float2bfloat16(__bfloat162float(b2) * tau);
            b3 = __float2bfloat16(__bfloat162float(b3) * tau);
        }
        __nv_bfloat16* qd = q_out + static_cast<size_t>(m) * 3072 + off;
        qd[0] = b0; qd[1] = b1; qd[2] = b2; qd[3] = b3;
    }

    // 2. Compute Rel-pos into registers (written after PDL trigger):
    float r_fin[8];
    if (wid < 3) {
        for (int sub = 0; sub < 8; ++sub) {
            int hq = wid * 8 + sub;
            int d  = lane & 15;
            float rv = (lane < 16) ? (__bfloat162float(r_raw[static_cast<size_t>(m) * 384 + hq * 16 + d]) * r_clip_scale[hq * 16 + d]) : 0.0f;
            float sq = rv * rv;
            #pragma unroll
            for (int mask = 8; mask >= 1; mask >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, mask);
            float rc = fminf(1.0f, 8.0f / (sqrtf(sq) + 1e-6f));
            __nv_bfloat16 rb = __float2bfloat16(rv * rc);
            if (!is_local) rb = __float2bfloat16(__bfloat162float(rb) * tau);
            r_fin[sub] = __bfloat162float(rb);
        }
    }

    // 3. K & V heads + PDL trigger before final cache & rel stores:
    int kv_slot = is_local ? (q_pos % kv_cap) : q_pos;
    bool wr_kv  = (kv_slot >= 0 && kv_slot < kv_cap);
    float k0 = 0.f, k1 = 0.f, k2 = 0.f, k3 = 0.f, kc = 0.f;
    float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f, vr = 0.f;
    int off_kv = wid * 128 + lane * 4;
    if (wr_kv) {
        const __nv_bfloat16* kp = k_sconv + static_cast<size_t>(m) * 1024 + off_kv;
        const __nv_bfloat16* vp = v_sconv + static_cast<size_t>(m) * 1024 + off_kv;
        k0 = __bfloat162float(kp[0]); k1 = __bfloat162float(kp[1]); k2 = __bfloat162float(kp[2]); k3 = __bfloat162float(kp[3]);
        float kn = sqrtf(wv_warp_sum_f32(k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3));
        kc = fminf(1.0f, kClipQK / (kn + 1e-6f));
        v0 = __bfloat162float(vp[0]); v1 = __bfloat162float(vp[1]); v2 = __bfloat162float(vp[2]); v3 = __bfloat162float(vp[3]);
        vr = rsqrtf(wv_warp_sum_f32(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3) * (1.0f / 128.0f) + 1e-6f);
    }
    if (wr_kv) {
        if (!is_local && kv_is_q8 != 0) {
            float k_sc = 0.0f, v_sc = 0.0f;
            uint32_t k_pk = wv_ink_pack_q8_head128_warp(k0 * kc, k1 * kc, k2 * kc, k3 * kc, k_sc);
            uint32_t v_pk = wv_ink_pack_q8_head128_warp(v0 * vr, v1 * vr, v2 * vr, v3 * vr, v_sc);
            size_t tok_h = (static_cast<size_t>(phys_b) * kv_cap + static_cast<size_t>(kv_slot)) * 8 + wid;
            static_cast<uint32_t*>(k_cache)[tok_h * 32 + lane] = k_pk;
            static_cast<uint32_t*>(v_cache)[tok_h * 32 + lane] = v_pk;
            if (lane == 0) {
                k_cache_sc[tok_h] = k_sc;
                v_cache_sc[tok_h] = v_sc;
            }
        } else {
            __nv_bfloat16* k_bf = static_cast<__nv_bfloat16*>(k_cache);
            __nv_bfloat16* v_bf = static_cast<__nv_bfloat16*>(v_cache);
            size_t dst_off = (static_cast<size_t>(phys_b) * kv_cap + static_cast<size_t>(kv_slot)) * 1024 + off_kv;
            k_bf[dst_off + 0] = __float2bfloat16(k0 * kc); k_bf[dst_off + 1] = __float2bfloat16(k1 * kc);
            k_bf[dst_off + 2] = __float2bfloat16(k2 * kc); k_bf[dst_off + 3] = __float2bfloat16(k3 * kc);
            v_bf[dst_off + 0] = __float2bfloat16(v0 * vr); v_bf[dst_off + 1] = __float2bfloat16(v1 * vr);
            v_bf[dst_off + 2] = __float2bfloat16(v2 * vr); v_bf[dst_off + 3] = __float2bfloat16(v3 * vr);
        }
    }
    if (wid < 3 && lane < 16) {
        for (int sub = 0; sub < 8; ++sub) {
            r_out_f32[(static_cast<size_t>(m) * 24 + wid * 8 + sub) * 16 + lane] = r_fin[sub];
        }
    }
    wv_pdl_trigger();
}

// Fused SWA `S == 1 && commit_sconv == 1` K/V Static SConv + Q/K/Rel Prep + KV Cache Write: Grid `<<<B_act, dim3(32, 8)>>>`
__global__ void wv_ink_prep_qkv_rel_kvcache_fused_swa_s1_kernel(
    const __nv_bfloat16* __restrict__ q_raw,
    const __nv_bfloat16* __restrict__ k_raw,
    const __nv_bfloat16* __restrict__ v_raw,
    const __nv_bfloat16* __restrict__ r_raw,
    const float* __restrict__ k_sconv_w,
    const float* __restrict__ v_sconv_w,
    float* __restrict__ k_state,
    float* __restrict__ v_state,
    const float* __restrict__ q_clip_scale,
    const float* __restrict__ r_clip_scale,
    __nv_bfloat16* __restrict__ q_out,
    float* __restrict__ r_out_f32,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ d_slot_pos,
    const int32_t* __restrict__ d_slot_map,
    int kv_cap, int B) {
    wv_pdl_wait();
    int b = blockIdx.x;
    if (b >= B) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x, wid = threadIdx.y;
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    int q_pos  = d_slot_pos[phys_b];
    constexpr float kClipQK   = 22.627416997969522f;
    constexpr float kInvSqrt5 = 0.44721359549995793928f;

    for (int sub = 0; sub < 3; ++sub) {
        int hq = wid * 3 + sub, off = hq * 128 + lane * 4;
        const __nv_bfloat16* qp = q_raw + static_cast<size_t>(b) * 3072 + off;
        float4 sc = reinterpret_cast<const float4*>(q_clip_scale + off)[0];
        float f0 = __bfloat162float(qp[0]) * sc.x, f1 = __bfloat162float(qp[1]) * sc.y;
        float f2 = __bfloat162float(qp[2]) * sc.z, f3 = __bfloat162float(qp[3]) * sc.w;
        float cf = fminf(1.0f, kClipQK / (sqrtf(wv_warp_sum_f32(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3)) + 1e-6f));
        __nv_bfloat16* qd = q_out + static_cast<size_t>(b) * 3072 + off;
        qd[0] = __float2bfloat16(f0 * cf); qd[1] = __float2bfloat16(f1 * cf);
        qd[2] = __float2bfloat16(f2 * cf); qd[3] = __float2bfloat16(f3 * cf);
    }

    float r_fin[8];
    if (wid < 3) {
        for (int sub = 0; sub < 8; ++sub) {
            int hq = wid * 8 + sub, d = lane & 15;
            float rv = (lane < 16) ? (__bfloat162float(r_raw[static_cast<size_t>(b) * 384 + hq * 16 + d]) * r_clip_scale[hq * 16 + d]) : 0.0f;
            float sq = rv * rv;
            #pragma unroll
            for (int mask = 8; mask >= 1; mask >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, mask);
            float rc = fminf(1.0f, 8.0f / (sqrtf(sq) + 1e-6f));
            r_fin[sub] = __bfloat162float(__float2bfloat16(rv * rc));
        }
    }

    int off_kv = wid * 128 + lane * 4;
    float* k_st = k_state + static_cast<size_t>(phys_b) * 3 * 1024;
    float* v_st = v_state + static_cast<size_t>(phys_b) * 3 * 1024;
    float kx[4], kh1[4], kh2[4], k_s[4], vx[4], vh1[4], vh2[4], v_s[4];
    const __nv_bfloat16* kp = k_raw + static_cast<size_t>(b) * 1024 + off_kv;
    const __nv_bfloat16* vp = v_raw + static_cast<size_t>(b) * 1024 + off_kv;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int c = off_kv + r;
        float4 kw4 = reinterpret_cast<const float4*>(k_sconv_w)[c];
        float4 vw4 = reinterpret_cast<const float4*>(v_sconv_w)[c];
        kx[r] = __bfloat162float(kp[r]); kh1[r] = k_st[1024 + c]; kh2[r] = k_st[2048 + c];
        vx[r] = __bfloat162float(vp[r]); vh1[r] = v_st[1024 + c]; vh2[r] = v_st[2048 + c];
        float k_cv = kw4.x * k_st[c] + kw4.y * kh1[r] + kw4.z * kh2[r] + kw4.w * kx[r];
        float v_cv = vw4.x * v_st[c] + vw4.y * vh1[r] + vw4.z * vh2[r] + vw4.w * vx[r];
        k_s[r] = __bfloat162float(__float2bfloat16((k_cv + kx[r]) * kInvSqrt5));
        v_s[r] = __bfloat162float(__float2bfloat16((v_cv + vx[r]) * kInvSqrt5));
    }
    float kc = fminf(1.0f, kClipQK / (sqrtf(wv_warp_sum_f32(k_s[0]*k_s[0] + k_s[1]*k_s[1] + k_s[2]*k_s[2] + k_s[3]*k_s[3])) + 1e-6f));
    float vr = rsqrtf(wv_warp_sum_f32(v_s[0]*v_s[0] + v_s[1]*v_s[1] + v_s[2]*v_s[2] + v_s[3]*v_s[3]) * (1.0f / 128.0f) + 1e-6f);
    int kv_slot = q_pos % kv_cap;

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int c = off_kv + r;
        k_st[c] = kh1[r]; k_st[1024 + c] = kh2[r]; k_st[2048 + c] = kx[r];
        v_st[c] = vh1[r]; v_st[1024 + c] = vh2[r]; v_st[2048 + c] = vx[r];
    }
    if (kv_slot >= 0 && kv_slot < kv_cap) {
        size_t dst_off = (static_cast<size_t>(phys_b) * kv_cap + kv_slot) * 1024 + off_kv;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            k_cache[dst_off + r] = __float2bfloat16(k_s[r] * kc);
            v_cache[dst_off + r] = __float2bfloat16(v_s[r] * vr);
        }
    }
    if (wid < 3 && lane < 16) {
        for (int sub = 0; sub < 8; ++sub) {
            r_out_f32[(static_cast<size_t>(b) * 24 + wid * 8 + sub) * 16 + lane] = r_fin[sub];
        }
    }
    wv_pdl_trigger();
}

// Fused SWA `S <= 8` (`commit_now == 0` OR `1`) K/V Static SConv + Q/K/Rel Prep + KV Cache Write (`-40` launches/step in MTP, `8x` SMs!):
// Grid `<<<dim3(8, B_act), dim3(32, 4)>>>` (`wid == 0` handles K/V head `hkv`; `wid == 1..3` handle Q/Rel heads `3*hkv + 0..2` in parallel!)
__global__ void wv_ink_prep_qkv_rel_kvcache_fused_swa_sm_kernel(
    const __nv_bfloat16* __restrict__ q_raw,
    const __nv_bfloat16* __restrict__ k_raw,
    const __nv_bfloat16* __restrict__ v_raw,
    const __nv_bfloat16* __restrict__ r_raw,
    const float* __restrict__ k_sconv_w,
    const float* __restrict__ v_sconv_w,
    float* __restrict__ k_state,
    float* __restrict__ v_state,
    float* __restrict__ k_cand,
    float* __restrict__ v_cand,
    const float* __restrict__ q_clip_scale,
    const float* __restrict__ r_clip_scale,
    __nv_bfloat16* __restrict__ q_out,
    float* __restrict__ r_out_f32,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const int32_t* __restrict__ d_slot_pos,
    const int32_t* __restrict__ d_slot_map,
    int kv_cap, int B, int S, int commit_now) {
    wv_pdl_wait();
    int hkv = blockIdx.x; // 0..7
    int b   = blockIdx.y;
    if (b >= B || S <= 0) { wv_pdl_trigger(); return; }
    int lane   = threadIdx.x;
    int wid    = threadIdx.y; // 0..3
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    int q_pos0 = d_slot_pos[phys_b];
    constexpr float kClipQK   = 22.627416997969522f;
    constexpr float kInvSqrt5 = 0.44721359549995793928f;

    if (wid == 0) {
        int off_kv  = hkv * 128 + lane * 4;
        float* k_st = k_state + static_cast<size_t>(phys_b) * 3 * 1024;
        float* v_st = v_state + static_cast<size_t>(phys_b) * 3 * 1024;
        float* cb_k = (k_cand != nullptr) ? (k_cand + static_cast<size_t>(phys_b) * 8 * 1024) : nullptr;
        float* cb_v = (v_cand != nullptr) ? (v_cand + static_cast<size_t>(phys_b) * 8 * 1024) : nullptr;

        float kh0[4], kh1[4], kh2[4], vh0[4], vh1[4], vh2[4];
        float4 kw4[4], vw4[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int c = off_kv + r;
            kh0[r] = k_st[c]; kh1[r] = k_st[1024 + c]; kh2[r] = k_st[2048 + c];
            vh0[r] = v_st[c]; vh1[r] = v_st[1024 + c]; vh2[r] = v_st[2048 + c];
            kw4[r] = reinterpret_cast<const float4*>(k_sconv_w)[c];
            vw4[r] = reinterpret_cast<const float4*>(v_sconv_w)[c];
        }

        for (int s = 0; s < S; ++s) {
            size_t m    = static_cast<size_t>(b) * S + s;
            int q_pos   = q_pos0 + s;
            float k_s[4], v_s[4];
            const __nv_bfloat16* kp = k_raw + m * 1024 + off_kv;
            const __nv_bfloat16* vp = v_raw + m * 1024 + off_kv;
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                int c = off_kv + r;
                float kx = __bfloat162float(kp[r]);
                float vx = __bfloat162float(vp[r]);
                if (cb_k != nullptr && s < 8) cb_k[static_cast<size_t>(s) * 1024 + c] = kx;
                if (cb_v != nullptr && s < 8) cb_v[static_cast<size_t>(s) * 1024 + c] = vx;
                float k_cv = kw4[r].x * kh0[r] + kw4[r].y * kh1[r] + kw4[r].z * kh2[r] + kw4[r].w * kx;
                float v_cv = vw4[r].x * vh0[r] + vw4[r].y * vh1[r] + vw4[r].z * vh2[r] + vw4[r].w * vx;
                k_s[r] = __bfloat162float(__float2bfloat16((k_cv + kx) * kInvSqrt5));
                v_s[r] = __bfloat162float(__float2bfloat16((v_cv + vx) * kInvSqrt5));
                kh0[r] = kh1[r]; kh1[r] = kh2[r]; kh2[r] = kx;
                vh0[r] = vh1[r]; vh1[r] = vh2[r]; vh2[r] = vx;
            }
            float kc = fminf(1.0f, kClipQK / (sqrtf(wv_warp_sum_f32(k_s[0]*k_s[0] + k_s[1]*k_s[1] + k_s[2]*k_s[2] + k_s[3]*k_s[3])) + 1e-6f));
            float vr = rsqrtf(wv_warp_sum_f32(v_s[0]*v_s[0] + v_s[1]*v_s[1] + v_s[2]*v_s[2] + v_s[3]*v_s[3]) * (1.0f / 128.0f) + 1e-6f);
            int kv_slot = q_pos % kv_cap;
            if (kv_slot >= 0 && kv_slot < kv_cap) {
                size_t dst_off = (static_cast<size_t>(phys_b) * kv_cap + kv_slot) * 1024 + off_kv;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    k_cache[dst_off + r] = __float2bfloat16(k_s[r] * kc);
                    v_cache[dst_off + r] = __float2bfloat16(v_s[r] * vr);
                }
            }
        }

        if (commit_now) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                int c = off_kv + r;
                k_st[c] = kh0[r]; k_st[1024 + c] = kh1[r]; k_st[2048 + c] = kh2[r];
                v_st[c] = vh0[r]; v_st[1024 + c] = vh1[r]; v_st[2048 + c] = vh2[r];
            }
        }
    } else {
        int hq     = hkv * 3 + (wid - 1);
        int off_q  = hq * 128 + lane * 4;
        int d      = lane & 15;
        float4 qsc = reinterpret_cast<const float4*>(q_clip_scale + off_q)[0];
        float  rsc = (lane < 16) ? r_clip_scale[hq * 16 + d] : 0.0f;

        for (int s = 0; s < S; ++s) {
            size_t m = static_cast<size_t>(b) * S + s;
            const __nv_bfloat16* qp = q_raw + m * 3072 + off_q;
            float f0 = __bfloat162float(qp[0]) * qsc.x;
            float f1 = __bfloat162float(qp[1]) * qsc.y;
            float f2 = __bfloat162float(qp[2]) * qsc.z;
            float f3 = __bfloat162float(qp[3]) * qsc.w;
            float cf = fminf(1.0f, kClipQK / (sqrtf(wv_warp_sum_f32(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3)) + 1e-6f));
            __nv_bfloat16* qd = q_out + m * 3072 + off_q;
            qd[0] = __float2bfloat16(f0 * cf); qd[1] = __float2bfloat16(f1 * cf);
            qd[2] = __float2bfloat16(f2 * cf); qd[3] = __float2bfloat16(f3 * cf);

            float rv = (lane < 16) ? (__bfloat162float(r_raw[m * 384 + hq * 16 + d]) * rsc) : 0.0f;
            float sq = rv * rv;
            #pragma unroll
            for (int mask = 8; mask >= 1; mask >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, mask);
            float rc = fminf(1.0f, 8.0f / (sqrtf(sq) + 1e-6f));
            if (lane < 16) {
                r_out_f32[(m * 24 + hq) * 16 + lane] = __bfloat162float(__float2bfloat16(rv * rc));
            }
        }
    }
    wv_pdl_trigger();
}

// Fused Attn Static ShortConvolution + Block Boundary / Residual Update (`0.0-ULP` BF16 rounding preserved!):
// Grid `<<<dim3(9, B_act), 256>>>`
__global__ void wv_ink_attn_sconv_res_update_kernel(
    const __nv_bfloat16* __restrict__ wo_in,
    const __nv_bfloat16* __restrict__ block_res_bf16,
    const float* __restrict__ w_k4,
    float* __restrict__ conv_state,
    float* __restrict__ cand_save,
    const int32_t* __restrict__ d_slot_map,
    __nv_bfloat16* __restrict__ blocks_bf16,
    float* __restrict__ h_mid_f32,
    int B, int S, int n_blk_before, int max_blk, int is_boundary, int commit_now) {
    wv_pdl_wait();
    int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int b = blockIdx.y;
    if (c >= INK2_HIDDEN || b >= B || S <= 0) { wv_pdl_trigger(); return; }
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    float4 w4  = reinterpret_cast<const float4*>(w_k4)[c];
    float* st_b = conv_state + static_cast<size_t>(phys_b) * 3 * INK2_HIDDEN;
    float h0 = st_b[0 * INK2_HIDDEN + c], h1 = st_b[1 * INK2_HIDDEN + c], h2 = st_b[2 * INK2_HIDDEN + c];
    float* cb = (cand_save != nullptr) ? (cand_save + static_cast<size_t>(phys_b) * 8 * INK2_HIDDEN) : nullptr;
    constexpr float kInvSqrt5 = 0.44721359549995793928f;

    for (int s = 0; s < S; ++s) {
        size_t idx  = static_cast<size_t>(b * S + s) * INK2_HIDDEN + c;
        float x_cur = __bfloat162float(wo_in[idx]);
        if (cb != nullptr && s < 8) cb[static_cast<size_t>(s) * INK2_HIDDEN + c] = x_cur;
        float conv = w4.x * h0 + w4.y * h1 + w4.z * h2 + w4.w * x_cur;
        float a    = __bfloat162float(__float2bfloat16((conv + x_cur) * kInvSqrt5));
        __nv_bfloat16 br = block_res_bf16[idx];
        if (is_boundary) {
            int m = b * S + s;
            blocks_bf16[(static_cast<size_t>(m) * max_blk + n_blk_before) * INK2_HIDDEN + c] = br;
            h_mid_f32[idx] = a;
        } else {
            h_mid_f32[idx] = __bfloat162float(br) + a;
        }
        h0 = h1; h1 = h2; h2 = x_cur;
    }
    if (commit_now) {
        st_b[0 * INK2_HIDDEN + c] = h0; st_b[1 * INK2_HIDDEN + c] = h1; st_b[2 * INK2_HIDDEN + c] = h2;
    }
    wv_pdl_trigger();
}

// Split-KV Online Flash-GQA w/ Relative Position Profile (`rel_proj_t [1024, 16]`)!
// Grid `dim3(M * 24, n_splits)`, Block `dim3(32, 4)` (`4` warps per `(m, hq)`)
__global__ void wv_ink_flash_gqa_rel_sink_split_kernel(
    const __nv_bfloat16* __restrict__ q_all,      // [B, S, 24, 128]
    const float* __restrict__ r_all_f32,          // [B, S, 24, 16]
    const float* __restrict__ rel_proj_t,         // [1024, 16] transposed!
    const __nv_bfloat16* __restrict__ k_cache,    // [B_max, kv_cap, 8, 128]
    const __nv_bfloat16* __restrict__ v_cache,    // [B_max, kv_cap, 8, 128]
    const int32_t* __restrict__ d_slot_pos,       // [B_max]
    const int32_t* __restrict__ d_slot_map,       // [B] or nullptr
    float* __restrict__ m_part,                   // [M, 24, n_splits]
    float* __restrict__ l_part,                   // [M, 24, n_splits]
    float* __restrict__ o_part,                   // [M, 24, n_splits, 128]
    int S, int kv_cap, int is_local, int n_splits) {
    wv_pdl_wait();
    int mh     = blockIdx.x;
    int sp     = blockIdx.y;
    int m      = mh / 24;
    int hq     = mh - m * 24;
    int hkv    = hq / 3;
    int b      = m / S;
    int s      = m - b * S;
    int lane   = threadIdx.x; // 0..31
    int wid    = threadIdx.y; // 0..3

    __shared__ float s_r16[16];
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    int q_pos  = d_slot_pos[phys_b] + s;
    int kv_beg = is_local ? max(0, q_pos - (INK2_SWA_WIN - 1)) : 0;
    int kv_tot = (q_pos + 1) - kv_beg;
    int chunk  = (kv_tot + n_splits - 1) / n_splits;
    int p0     = kv_beg + sp * chunk;
    int p1     = min(q_pos + 1, p0 + chunk);

    if (wid == 0 && lane < 16) {
        s_r16[lane] = r_all_f32[static_cast<size_t>(mh) * 16 + lane];
    }
    __syncthreads();

    const __nv_bfloat16* qp = q_all + static_cast<size_t>(mh) * 128 + lane * 4;
    constexpr float kScale = 0.08838834764831845f; // 128^-0.5
    float q0 = __bfloat162float(qp[0]) * kScale, q1 = __bfloat162float(qp[1]) * kScale;
    float q2 = __bfloat162float(qp[2]) * kScale, q3 = __bfloat162float(qp[3]) * kScale;
    float4 r0 = reinterpret_cast<const float4*>(s_r16 + 0)[0];
    float4 r1 = reinterpret_cast<const float4*>(s_r16 + 4)[0];
    float4 r2 = reinterpret_cast<const float4*>(s_r16 + 8)[0];
    float4 r3 = reinterpret_cast<const float4*>(s_r16 + 12)[0];

    float m_max = -1e30f, l_sum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    const __nv_bfloat16* kb_base = k_cache + static_cast<size_t>(phys_b) * kv_cap * 1024 + hkv * 128 + lane * 4;
    const __nv_bfloat16* vb_base = v_cache + static_cast<size_t>(phys_b) * kv_cap * 1024 + hkv * 128 + lane * 4;

    for (int kp = p0 + wid; kp < p1; kp += 4) {
        int slot = is_local ? (kp % kv_cap) : kp;
        size_t s_off = static_cast<size_t>(slot) * 1024;
        const __nv_bfloat16* kt = kb_base + s_off;
        float qk = q0 * __bfloat162float(kt[0]) + q1 * __bfloat162float(kt[1])
                 + q2 * __bfloat162float(kt[2]) + q3 * __bfloat162float(kt[3]);
        qk = wv_warp_sum_f32(qk);
        int dist = q_pos - kp;
        if (dist >= 0 && dist < 1024) {
            const float4* rp = reinterpret_cast<const float4*>(rel_proj_t + dist * 16);
            float4 p_0 = rp[0], p_1 = rp[1], p_2 = rp[2], p_3 = rp[3];
            qk += r0.x*p_0.x + r0.y*p_0.y + r0.z*p_0.z + r0.w*p_0.w
                + r1.x*p_1.x + r1.y*p_1.y + r1.z*p_1.z + r1.w*p_1.w
                + r2.x*p_2.x + r2.y*p_2.y + r2.z*p_2.z + r2.w*p_2.w
                + r3.x*p_3.x + r3.y*p_3.y + r3.z*p_3.z + r3.w*p_3.w;
        }
        float m_new = fmaxf(m_max, qk);
        float alpha = (m_max > -1e29f) ? expf(m_max - m_new) : 0.0f;
        float wt    = expf(qk - m_new);
        l_sum = l_sum * alpha + wt;
        m_max = m_new;
        const __nv_bfloat16* vt = vb_base + s_off;
        o0 = o0 * alpha + wt * __bfloat162float(vt[0]);
        o1 = o1 * alpha + wt * __bfloat162float(vt[1]);
        o2 = o2 * alpha + wt * __bfloat162float(vt[2]);
        o3 = o3 * alpha + wt * __bfloat162float(vt[3]);
    }

    __shared__ float s_m[4], s_l[4], s_o[4][128];
    if (lane == 0) { s_m[wid] = m_max; s_l[wid] = l_sum; }
    s_o[wid][lane * 4 + 0] = o0; s_o[wid][lane * 4 + 1] = o1;
    s_o[wid][lane * 4 + 2] = o2; s_o[wid][lane * 4 + 3] = o3;
    __syncthreads();

    if (wid == 0) {
        float fm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
        float fl = 0.0f, fo0 = 0.0f, fo1 = 0.0f, fo2 = 0.0f, fo3 = 0.0f;
        if (fm > -1e29f) {
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (s_l[w] > 0.0f) {
                    float a = expf(s_m[w] - fm);
                    fl  += s_l[w] * a;
                    fo0 += s_o[w][lane * 4 + 0] * a; fo1 += s_o[w][lane * 4 + 1] * a;
                    fo2 += s_o[w][lane * 4 + 2] * a; fo3 += s_o[w][lane * 4 + 3] * a;
                }
            }
        }
        size_t idx_sp = static_cast<size_t>(mh) * n_splits + sp;
        if (lane == 0) { m_part[idx_sp] = fm; l_part[idx_sp] = fl; }
        float* od = o_part + idx_sp * 128 + lane * 4;
        od[0] = fo0; od[1] = fo1; od[2] = fo2; od[3] = fo3;
    }
    wv_pdl_trigger();
}


// Combines `n_splits` Flash-GQA partials + Sink (`exp(sink - m_tot)`), applies `sigmoid(gate_attn)`,
// and fuses `FWHT-128 + Q8_32` (`128` == `head_dim`, so each Q head IS a 128-element FWHT block!):
// Grid `<<<(M * 24 + 3) / 4, dim3(32, 4)>>>`
__global__ void wv_ink_flash_gqa_combine_gate_fwht_q8_kernel(
    const float* __restrict__ m_part,             // [M, 24, n_splits]
    const float* __restrict__ l_part,             // [M, 24, n_splits]
    const float* __restrict__ o_part,             // [M, 24, n_splits, 128]
    const float* __restrict__ sinks_f32,          // [M, 24]
    const __nv_bfloat16* __restrict__ gate_attn,  // [M, 3072]
    int8_t* __restrict__ O_q8,                    // [M, 3072]
    float* __restrict__ O_sc,                     // [M, 96]
    int total_mh,
    int n_splits) {
    wv_pdl_wait();
    int mh = blockIdx.x * blockDim.y + threadIdx.y;
    if (mh >= total_mh) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x;
    float sink = sinks_f32[mh];
    float m_tot = sink;
    size_t sp_base = static_cast<size_t>(mh) * n_splits;
    for (int s = 0; s < n_splits; ++s) {
        if (l_part[sp_base + s] > 0.0f) {
            m_tot = fmaxf(m_tot, m_part[sp_base + s]);
        }
    }
    float denom = expf(sink - m_tot);
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        float ls = l_part[sp_base + s];
        if (ls > 0.0f) {
            float a = expf(m_part[sp_base + s] - m_tot);
            denom += ls * a;
            const float* op = o_part + (sp_base + s) * 128 + lane * 4;
            o0 += op[0] * a; o1 += op[1] * a; o2 += op[2] * a; o3 += op[3] * a;
        }
    }
    float inv_d = 1.0f / fmaxf(denom, 1e-20f);
    // Match `attn_out.to(bf16)` before `sigmoid(gate_attn)` multiplication:
    float a0 = __bfloat162float(__float2bfloat16(o0 * inv_d));
    float a1 = __bfloat162float(__float2bfloat16(o1 * inv_d));
    float a2 = __bfloat162float(__float2bfloat16(o2 * inv_d));
    float a3 = __bfloat162float(__float2bfloat16(o3 * inv_d));

    const __nv_bfloat16* gp = gate_attn + static_cast<size_t>(mh) * 128 + lane * 4;
    auto sig_bf = [](float a, __nv_bfloat16 gb) -> float {
        float g  = __bfloat162float(gb);
        float sg = 1.0f / (1.0f + expf(-g));
        return __bfloat162float(__float2bfloat16(a * sg));
    };
    float f0 = sig_bf(a0, gp[0]), f1 = sig_bf(a1, gp[1]);
    float f2 = sig_bf(a2, gp[2]), f3 = sig_bf(a3, gp[3]);
    wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
    wv_quant_q8_32_warp(f0, f1, f2, f3, lane, O_q8 + static_cast<size_t>(mh) * 128, O_sc + mh * 4);
    wv_pdl_trigger();
}

__global__ void wv_ink_bf16_to_f32_flat_kernel(
    const __nv_bfloat16* __restrict__ src, float* __restrict__ dst, int total_elems) {
    wv_pdl_wait();
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= total_elems) { wv_pdl_trigger(); return; }
    float v = __bfloat162float(src[idx]);
    dst[idx] = v;
    wv_pdl_trigger();
}

__global__ void wv_ink_stash_tap_kernel(
    const __nv_bfloat16* __restrict__ h_src, // [M, 2304]
    __nv_bfloat16* __restrict__ taps_dst,    // [M, 4, 2304]
    int tap_slot, int M) {
    wv_pdl_wait();
    int idx = blockIdx.x * 256 + threadIdx.x;
    int tot = M * INK2_HIDDEN;
    if (idx >= tot) { wv_pdl_trigger(); return; }
    int m = idx / INK2_HIDDEN, d = idx - m * INK2_HIDDEN;
    __nv_bfloat16 v = h_src[idx];
    taps_dst[(static_cast<size_t>(m) * 4 + tap_slot) * INK2_HIDDEN + d] = v;
    wv_pdl_trigger();
}

__global__ void wv_ink_copy_blk_slot_kernel(
    const __nv_bfloat16* __restrict__ src_blks, int src_max_blk, int src_slot,
    __nv_bfloat16* __restrict__ dst_blks, int dst_max_blk, int dst_slot, int M) {
    wv_pdl_wait();
    int idx = blockIdx.x * 256 + threadIdx.x;
    int tot = M * INK2_HIDDEN;
    if (idx >= tot) { wv_pdl_trigger(); return; }
    int m = idx / INK2_HIDDEN, d = idx - m * INK2_HIDDEN;
    __nv_bfloat16 v = src_blks[(static_cast<size_t>(m) * src_max_blk + src_slot) * INK2_HIDDEN + d];
    dst_blks[(static_cast<size_t>(m) * dst_max_blk + dst_slot) * INK2_HIDDEN + d] = v;
    wv_pdl_trigger();
}

__global__ void wv_ink_save_prelude_kernel(
    const __nv_bfloat16* __restrict__ cur_h,
    const __nv_bfloat16* __restrict__ blks,
    __nv_bfloat16* __restrict__ pre_h,
    __nv_bfloat16* __restrict__ pre_blk,
    int M, int max_blk) {
    wv_pdl_wait();
    int idx = blockIdx.x * 256 + threadIdx.x;
    int tot = M * INK2_HIDDEN;
    if (idx >= tot) { wv_pdl_trigger(); return; }
    int m = idx / INK2_HIDDEN, d = idx - m * INK2_HIDDEN;
    __nv_bfloat16 ch = cur_h[idx];
    __nv_bfloat16 b0 = blks[(static_cast<size_t>(m) * max_blk + 0) * INK2_HIDDEN + d];
    pre_h[idx] = ch;
    pre_blk[(static_cast<size_t>(m) * 2 + 0) * INK2_HIDDEN + d] = b0;
    wv_pdl_trigger();
}

__global__ void wv_ink_loop_pass_reset_kernel(
    __nv_bfloat16* __restrict__ cur_h,
    __nv_bfloat16* __restrict__ prev_h,
    const __nv_bfloat16* __restrict__ pre_h,
    __nv_bfloat16* __restrict__ blks,
    __nv_bfloat16* __restrict__ prv_blk,
    const __nv_bfloat16* __restrict__ pre_blk,
    int M, int max_blk, int prv_src_slot) {
    wv_pdl_wait();
    int idx = blockIdx.x * 256 + threadIdx.x;
    int tot = M * INK2_HIDDEN;
    if (idx >= tot) { wv_pdl_trigger(); return; }
    int m = idx / INK2_HIDDEN, d = idx - m * INK2_HIDDEN;
    __nv_bfloat16 b2 = blks[(static_cast<size_t>(m) * max_blk + prv_src_slot) * INK2_HIDDEN + d];
    __nv_bfloat16 ch = cur_h[idx];
    __nv_bfloat16 ph = pre_h[idx];
    __nv_bfloat16 pb = pre_blk[(static_cast<size_t>(m) * 2 + 0) * INK2_HIDDEN + d];
    prv_blk[(static_cast<size_t>(m) * 2 + 0) * INK2_HIDDEN + d] = b2;
    prev_h[idx] = ch;
    cur_h[idx]  = ph;
    blks[(static_cast<size_t>(m) * max_blk + 0) * INK2_HIDDEN + d] = pb;
    wv_pdl_trigger();
}

// GQA-3 Grouped Split-KV Online Flash-GQA (`3x` fewer `k_cache`, `v_cache`, & `rel_proj_t` loads, BF16 `0.0-ULP` + Global INT8 `__dp4a`):
// Grid `dim3(M * 8, n_splits)`, Block `dim3(32, 4)` (`1` CTA computes all `3` Q heads sharing `hkv = 0..7`!)
__global__ void wv_ink_flash_gqa3_rel_sink_split_kernel(
    const __nv_bfloat16* __restrict__ q_all,      // [B, S, 24, 128]
    const float* __restrict__ r_all_f32,          // [B, S, 24, 16]
    const float* __restrict__ rel_proj_t,         // [1024, 16]
    const void* __restrict__ k_cache,             // [B_max, kv_cap, 8, 128] bf16 or int8
    const void* __restrict__ v_cache,             // [B_max, kv_cap, 8, 128] bf16 or int8
    const float* __restrict__ k_cache_sc,         // [B_max, kv_cap, 8] or nullptr
    const float* __restrict__ v_cache_sc,         // [B_max, kv_cap, 8] or nullptr
    const int32_t* __restrict__ d_slot_pos,       // [B_max]
    const int32_t* __restrict__ d_slot_map,       // [B] or nullptr
    float* __restrict__ m_part,                   // [M, 24, n_splits]
    float* __restrict__ l_part,                   // [M, 24, n_splits]
    float* __restrict__ o_part,                   // [M, 24, n_splits, 128]
    int S, int kv_cap, int is_local, int kv_is_q8, int n_splits) {
    wv_pdl_wait();
    int mhkv = blockIdx.x; // 0 .. M * 8 - 1
    int sp   = blockIdx.y;
    int m    = mhkv >> 3;
    int hkv  = mhkv & 7;
    int hq0  = hkv * 3;
    int b    = m / S;
    int s    = m - b * S;
    int lane = threadIdx.x; // 0..31
    int wid  = threadIdx.y; // 0..3

    __shared__ float s_r16[3][16];
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    int q_pos  = d_slot_pos[phys_b] + s;
    int kv_beg = is_local ? max(0, q_pos - (INK2_SWA_WIN - 1)) : 0;
    int kv_tot = (q_pos + 1) - kv_beg;
    int chunk  = (kv_tot + n_splits - 1) / n_splits;
    int p0     = kv_beg + sp * chunk;
    int p1     = min(q_pos + 1, p0 + chunk);

    if (wid < 3 && lane < 16) {
        s_r16[wid][lane] = r_all_f32[static_cast<size_t>(m * 24 + hq0 + wid) * 16 + lane];
    }
    __syncthreads();

    constexpr float kScale = 0.08838834764831845f; // 128^-0.5
    float qv[3][4];
    float m_max[3], l_sum[3], o_acc[3][4];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        const __nv_bfloat16* qp = q_all + static_cast<size_t>(m * 24 + hq0 + g) * 128 + lane * 4;
        qv[g][0] = __bfloat162float(qp[0]) * kScale;
        qv[g][1] = __bfloat162float(qp[1]) * kScale;
        qv[g][2] = __bfloat162float(qp[2]) * kScale;
        qv[g][3] = __bfloat162float(qp[3]) * kScale;
        m_max[g] = -1e30f;
        l_sum[g] = 0.0f;
        o_acc[g][0] = 0.0f; o_acc[g][1] = 0.0f; o_acc[g][2] = 0.0f; o_acc[g][3] = 0.0f;
    }

    if (!is_local && kv_is_q8 != 0) {
        uint32_t q_pk[3];
        float q_sc[3];
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            q_pk[g] = wv_ink_pack_q8_head128_warp(qv[g][0], qv[g][1], qv[g][2], qv[g][3], q_sc[g]);
        }
        const uint32_t* kb_i8 = static_cast<const uint32_t*>(k_cache) + (static_cast<size_t>(phys_b) * kv_cap * 8 + hkv) * 32 + lane;
        const uint32_t* vb_i8 = static_cast<const uint32_t*>(v_cache) + (static_cast<size_t>(phys_b) * kv_cap * 8 + hkv) * 32 + lane;
        const float* ks_base  = k_cache_sc + static_cast<size_t>(phys_b) * kv_cap * 8 + hkv;
        const float* vs_base  = v_cache_sc + static_cast<size_t>(phys_b) * kv_cap * 8 + hkv;

        for (int kp = p0 + wid; kp < p1; kp += 4) {
            size_t tok_off = static_cast<size_t>(kp) * 256;
            size_t sc_off  = static_cast<size_t>(kp) * 8;
            uint32_t k_pk  = kb_i8[tok_off];
            uint32_t v_pk  = vb_i8[tok_off];
            float k_sc     = ks_base[sc_off];
            float v_sc     = vs_base[sc_off];
            float vt0, vt1, vt2, vt3;
            wv_ink_unpack_s8x4_f32(v_pk, vt0, vt1, vt2, vt3);

            int dist = q_pos - kp;
            float4 p_0 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 p_1 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 p_2 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 p_3 = make_float4(0.f, 0.f, 0.f, 0.f);
            bool has_rel = (dist >= 0 && dist < 1024);
            if (has_rel) {
                const float4* rp = reinterpret_cast<const float4*>(rel_proj_t + static_cast<size_t>(dist) * 16);
                p_0 = __ldg(rp + 0); p_1 = __ldg(rp + 1); p_2 = __ldg(rp + 2); p_3 = __ldg(rp + 3);
            }

            #pragma unroll
            for (int g = 0; g < 3; ++g) {
                int32_t dp = __dp4a(static_cast<int32_t>(q_pk[g]), static_cast<int32_t>(k_pk), 0);
                float qk = wv_warp_sum_f32(static_cast<float>(dp)) * (q_sc[g] * k_sc);
                if (has_rel) {
                    float4 r0 = reinterpret_cast<const float4*>(s_r16[g] + 0)[0];
                    float4 r1 = reinterpret_cast<const float4*>(s_r16[g] + 4)[0];
                    float4 r2 = reinterpret_cast<const float4*>(s_r16[g] + 8)[0];
                    float4 r3 = reinterpret_cast<const float4*>(s_r16[g] + 12)[0];
                    qk += r0.x*p_0.x + r0.y*p_0.y + r0.z*p_0.z + r0.w*p_0.w
                        + r1.x*p_1.x + r1.y*p_1.y + r1.z*p_1.z + r1.w*p_1.w
                        + r2.x*p_2.x + r2.y*p_2.y + r2.z*p_2.z + r2.w*p_2.w
                        + r3.x*p_3.x + r3.y*p_3.y + r3.z*p_3.z + r3.w*p_3.w;
                }
                float m_new = fmaxf(m_max[g], qk);
                float alpha = (m_max[g] > -1e29f) ? expf(m_max[g] - m_new) : 0.0f;
                float wt    = expf(qk - m_new);
                float wt_v  = wt * v_sc;
                l_sum[g]    = l_sum[g] * alpha + wt;
                m_max[g]    = m_new;
                o_acc[g][0] = o_acc[g][0] * alpha + wt_v * vt0;
                o_acc[g][1] = o_acc[g][1] * alpha + wt_v * vt1;
                o_acc[g][2] = o_acc[g][2] * alpha + wt_v * vt2;
                o_acc[g][3] = o_acc[g][3] * alpha + wt_v * vt3;
            }
        }
    } else {
        const __nv_bfloat16* kb_base = static_cast<const __nv_bfloat16*>(k_cache) + static_cast<size_t>(phys_b) * kv_cap * 1024 + hkv * 128 + lane * 4;
        const __nv_bfloat16* vb_base = static_cast<const __nv_bfloat16*>(v_cache) + static_cast<size_t>(phys_b) * kv_cap * 1024 + hkv * 128 + lane * 4;

        for (int kp = p0 + wid; kp < p1; kp += 4) {
            int slot     = is_local ? (kp % kv_cap) : kp;
            size_t s_off = static_cast<size_t>(slot) * 1024;
            const __nv_bfloat16* kt = kb_base + s_off;
            float kt0 = __bfloat162float(kt[0]), kt1 = __bfloat162float(kt[1]);
            float kt2 = __bfloat162float(kt[2]), kt3 = __bfloat162float(kt[3]);
            const __nv_bfloat16* vt = vb_base + s_off;
            float vt0 = __bfloat162float(vt[0]), vt1 = __bfloat162float(vt[1]);
            float vt2 = __bfloat162float(vt[2]), vt3 = __bfloat162float(vt[3]);

            int dist = q_pos - kp;
            float4 p_0 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 p_1 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 p_2 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 p_3 = make_float4(0.f, 0.f, 0.f, 0.f);
            bool has_rel = (dist >= 0 && dist < 1024);
            if (has_rel) {
                const float4* rp = reinterpret_cast<const float4*>(rel_proj_t + static_cast<size_t>(dist) * 16);
                p_0 = __ldg(rp + 0); p_1 = __ldg(rp + 1); p_2 = __ldg(rp + 2); p_3 = __ldg(rp + 3);
            }

            #pragma unroll
            for (int g = 0; g < 3; ++g) {
                float qk = qv[g][0] * kt0 + qv[g][1] * kt1 + qv[g][2] * kt2 + qv[g][3] * kt3;
                qk = wv_warp_sum_f32(qk);
                if (has_rel) {
                    float4 r0 = reinterpret_cast<const float4*>(s_r16[g] + 0)[0];
                    float4 r1 = reinterpret_cast<const float4*>(s_r16[g] + 4)[0];
                    float4 r2 = reinterpret_cast<const float4*>(s_r16[g] + 8)[0];
                    float4 r3 = reinterpret_cast<const float4*>(s_r16[g] + 12)[0];
                    qk += r0.x*p_0.x + r0.y*p_0.y + r0.z*p_0.z + r0.w*p_0.w
                        + r1.x*p_1.x + r1.y*p_1.y + r1.z*p_1.z + r1.w*p_1.w
                        + r2.x*p_2.x + r2.y*p_2.y + r2.z*p_2.z + r2.w*p_2.w
                        + r3.x*p_3.x + r3.y*p_3.y + r3.z*p_3.z + r3.w*p_3.w;
                }
                float m_new = fmaxf(m_max[g], qk);
                float alpha = (m_max[g] > -1e29f) ? expf(m_max[g] - m_new) : 0.0f;
                float wt    = expf(qk - m_new);
                l_sum[g]    = l_sum[g] * alpha + wt;
                m_max[g]    = m_new;
                o_acc[g][0] = o_acc[g][0] * alpha + wt * vt0;
                o_acc[g][1] = o_acc[g][1] * alpha + wt * vt1;
                o_acc[g][2] = o_acc[g][2] * alpha + wt * vt2;
                o_acc[g][3] = o_acc[g][3] * alpha + wt * vt3;
            }
        }
    }

    __shared__ float s_m[3][4], s_l[3][4], s_o[3][4][128];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) { s_m[g][wid] = m_max[g]; s_l[g][wid] = l_sum[g]; }
        s_o[g][wid][lane * 4 + 0] = o_acc[g][0]; s_o[g][wid][lane * 4 + 1] = o_acc[g][1];
        s_o[g][wid][lane * 4 + 2] = o_acc[g][2]; s_o[g][wid][lane * 4 + 3] = o_acc[g][3];
    }
    __syncthreads();

    if (wid < 3) {
        int g = wid;
        float fm = fmaxf(fmaxf(s_m[g][0], s_m[g][1]), fmaxf(s_m[g][2], s_m[g][3]));
        float fl = 0.0f, fo0 = 0.0f, fo1 = 0.0f, fo2 = 0.0f, fo3 = 0.0f;
        if (fm > -1e29f) {
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (s_l[g][w] > 0.0f) {
                    float a = expf(s_m[g][w] - fm);
                    fl  += s_l[g][w] * a;
                    fo0 += s_o[g][w][lane * 4 + 0] * a; fo1 += s_o[g][w][lane * 4 + 1] * a;
                    fo2 += s_o[g][w][lane * 4 + 2] * a; fo3 += s_o[g][w][lane * 4 + 3] * a;
                }
            }
        }
        int mh = m * 24 + hq0 + g;
        size_t idx_sp = static_cast<size_t>(mh) * n_splits + sp;
        if (lane == 0) { m_part[idx_sp] = fm; l_part[idx_sp] = fl; }
        float* od = o_part + idx_sp * 128 + lane * 4;
        od[0] = fo0; od[1] = fo1; od[2] = fo2; od[3] = fo3;
    }
    wv_pdl_trigger();
}

// S-Shared Long-Context Global Split-KV Flash-GQA (`B == 1, 2 <= S <= NW`):
// All `NW` warps cooperatively stage `NW` KV positions from DRAM into SMEM ONCE per tile,
// warp `wid = s` evaluates all 3 GQA heads of token `s` (`0` spills, `0B` stack), CUDA-Graph-safe!
// Grid `dim3(8, n_splits)`, Block `dim3(32, NW)`
template<int NW, bool KV_Q8>
__global__ void wv_ink_flash_gqa3_s_shared_split_kernel(
    const __nv_bfloat16* __restrict__ q_all,      // [1, S, 24, 128]
    const float* __restrict__ r_all_f32,          // [1, S, 24, 16]
    const float* __restrict__ rel_proj_t,         // [1024, 16]
    const void* __restrict__ k_cache,             // [B_max, kv_cap, 8, 128] bf16 or int8
    const void* __restrict__ v_cache,             // [B_max, kv_cap, 8, 128] bf16 or int8
    const float* __restrict__ k_cache_sc,         // [B_max, kv_cap, 8] or nullptr
    const float* __restrict__ v_cache_sc,         // [B_max, kv_cap, 8] or nullptr
    const int32_t* __restrict__ d_slot_pos,       // [B_max]
    const int32_t* __restrict__ d_slot_map,       // [1] or nullptr
    float* __restrict__ m_part,                   // [S, 24, n_splits]
    float* __restrict__ l_part,                   // [S, 24, n_splits]
    float* __restrict__ o_part,                   // [S, 24, n_splits, 128]
    int S, int kv_cap, int n_splits) {
    wv_pdl_wait();
    int hkv    = blockIdx.x; // 0..7
    int sp     = blockIdx.y; // 0..n_splits-1
    int hq0    = hkv * 3;
    int lane   = threadIdx.x; // 0..31
    int wid    = threadIdx.y; // 0..NW-1
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[0] : 0;
    int q_pos0 = d_slot_pos[phys_b];
    int kv_tot = q_pos0 + S;
    int chunk  = (kv_tot + n_splits - 1) / n_splits;
    int p0     = sp * chunk;
    int p1     = min(kv_tot, p0 + chunk);

    if (p0 >= p1) {
        if (wid < S) {
            #pragma unroll
            for (int g = 0; g < 3; ++g) {
                int mh = wid * 24 + hq0 + g;
                size_t idx_sp = static_cast<size_t>(mh) * n_splits + sp;
                if (lane == 0) { m_part[idx_sp] = -1e30f; l_part[idx_sp] = 0.0f; }
                float* od = o_part + idx_sp * 128 + lane * 4;
                od[0] = 0.0f; od[1] = 0.0f; od[2] = 0.0f; od[3] = 0.0f;
            }
        }
        wv_pdl_trigger();
        return;
    }

    __shared__ float s_r16[NW][3][16];
    __shared__ union {
        struct { uint32_t k[NW][32]; uint32_t v[NW][32]; float ks[NW]; float vs[NW]; } q8;
        struct { uint32_t kl[NW][32], kh[NW][32]; uint32_t vl[NW][32], vh[NW][32]; } bf;
    } s_kv;

    bool active_s = (wid < S);
    int  q_pos    = q_pos0 + wid;
    if (active_s && lane < 16) {
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            s_r16[wid][g][lane] = r_all_f32[static_cast<size_t>(wid * 24 + hq0 + g) * 16 + lane];
        }
    }

    constexpr float kScale = 0.08838834764831845f;
    float    qv[3][4];
    uint32_t q_pk[3] = {0u, 0u, 0u};
    float    q_sc[3] = {0.f, 0.f, 0.f};
    float    m_max[3], l_sum[3], o_acc[3][4];
    if (active_s) {
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            const __nv_bfloat16* qp = q_all + static_cast<size_t>(wid * 24 + hq0 + g) * 128 + lane * 4;
            qv[g][0] = __bfloat162float(qp[0]) * kScale;
            qv[g][1] = __bfloat162float(qp[1]) * kScale;
            qv[g][2] = __bfloat162float(qp[2]) * kScale;
            qv[g][3] = __bfloat162float(qp[3]) * kScale;
            if constexpr (KV_Q8) {
                q_pk[g] = wv_ink_pack_q8_head128_warp(qv[g][0], qv[g][1], qv[g][2], qv[g][3], q_sc[g]);
            }
            m_max[g] = -1e30f; l_sum[g] = 0.0f;
            o_acc[g][0] = 0.0f; o_acc[g][1] = 0.0f; o_acc[g][2] = 0.0f; o_acc[g][3] = 0.0f;
        }
    }

    const uint32_t* kb_i8      = KV_Q8 ? (static_cast<const uint32_t*>(k_cache) + (static_cast<size_t>(phys_b) * kv_cap * 8 + hkv) * 32 + lane) : nullptr;
    const uint32_t* vb_i8      = KV_Q8 ? (static_cast<const uint32_t*>(v_cache) + (static_cast<size_t>(phys_b) * kv_cap * 8 + hkv) * 32 + lane) : nullptr;
    const float*    ks_base    = KV_Q8 ? (k_cache_sc + static_cast<size_t>(phys_b) * kv_cap * 8 + hkv) : nullptr;
    const float*    vs_base    = KV_Q8 ? (v_cache_sc + static_cast<size_t>(phys_b) * kv_cap * 8 + hkv) : nullptr;
    const __nv_bfloat16* kb_bf = (!KV_Q8) ? (static_cast<const __nv_bfloat16*>(k_cache) + static_cast<size_t>(phys_b) * kv_cap * 1024 + hkv * 128 + lane * 4) : nullptr;
    const __nv_bfloat16* vb_bf = (!KV_Q8) ? (static_cast<const __nv_bfloat16*>(v_cache) + static_cast<size_t>(phys_b) * kv_cap * 1024 + hkv * 128 + lane * 4) : nullptr;

    for (int kp_base = p0; kp_base < p1; kp_base += NW) {
        int tile_cnt = min(NW, p1 - kp_base);
        if (wid < tile_cnt) {
            int kp = kp_base + wid;
            if constexpr (KV_Q8) {
                size_t tok_off = static_cast<size_t>(kp) * 256;
                s_kv.q8.k[wid][lane] = wv_ld_stream_u32(kb_i8 + tok_off);
                s_kv.q8.v[wid][lane] = wv_ld_stream_u32(vb_i8 + tok_off);
                if (lane == 0) {
                    size_t sc_off = static_cast<size_t>(kp) * 8;
                    s_kv.q8.ks[wid] = ks_base[sc_off];
                    s_kv.q8.vs[wid] = vs_base[sc_off];
                }
            } else {
                size_t s_off = static_cast<size_t>(kp) * 1024;
                uint2 kl2 = *reinterpret_cast<const uint2*>(kb_bf + s_off);
                uint2 vl2 = *reinterpret_cast<const uint2*>(vb_bf + s_off);
                s_kv.bf.kl[wid][lane] = kl2.x; s_kv.bf.kh[wid][lane] = kl2.y;
                s_kv.bf.vl[wid][lane] = vl2.x; s_kv.bf.vh[wid][lane] = vl2.y;
            }
        }
        __syncthreads();

        if (active_s) {
            bool all_far = (kp_base + tile_cnt <= q_pos0 - 1023);
            if (all_far) {
                for (int t = 0; t < tile_cnt; ++t) {
                    if constexpr (KV_Q8) {
                        uint32_t k_pk = s_kv.q8.k[t][lane];
                        uint32_t v_pk = s_kv.q8.v[t][lane];
                        float k_sc    = s_kv.q8.ks[t];
                        float v_sc    = s_kv.q8.vs[t];
                        float vt0, vt1, vt2, vt3;
                        wv_ink_unpack_s8x4_f32(v_pk, vt0, vt1, vt2, vt3);
                        #pragma unroll
                        for (int g = 0; g < 3; ++g) {
                            int32_t dp  = __dp4a(static_cast<int32_t>(q_pk[g]), static_cast<int32_t>(k_pk), 0);
                            float qk    = wv_warp_sum_f32(static_cast<float>(dp)) * (q_sc[g] * k_sc);
                            float m_new = fmaxf(m_max[g], qk);
                            float alpha = (m_max[g] > -1e29f) ? expf(m_max[g] - m_new) : 0.0f;
                            float wt    = expf(qk - m_new);
                            float wt_v  = wt * v_sc;
                            l_sum[g]    = l_sum[g] * alpha + wt;
                            m_max[g]    = m_new;
                            o_acc[g][0] = o_acc[g][0] * alpha + wt_v * vt0;
                            o_acc[g][1] = o_acc[g][1] * alpha + wt_v * vt1;
                            o_acc[g][2] = o_acc[g][2] * alpha + wt_v * vt2;
                            o_acc[g][3] = o_acc[g][3] * alpha + wt_v * vt3;
                        }
                    } else {
                        uint32_t kl_w = s_kv.bf.kl[t][lane], kh_w = s_kv.bf.kh[t][lane];
                        uint32_t vl_w = s_kv.bf.vl[t][lane], vh_w = s_kv.bf.vh[t][lane];
                        float2 kt01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kl_w));
                        float2 kt23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kh_w));
                        float2 vt01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vl_w));
                        float2 vt23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vh_w));
                        #pragma unroll
                        for (int g = 0; g < 3; ++g) {
                            float qk = qv[g][0] * kt01.x + qv[g][1] * kt01.y + qv[g][2] * kt23.x + qv[g][3] * kt23.y;
                            qk = wv_warp_sum_f32(qk);
                            float m_new = fmaxf(m_max[g], qk);
                            float alpha = (m_max[g] > -1e29f) ? expf(m_max[g] - m_new) : 0.0f;
                            float wt    = expf(qk - m_new);
                            l_sum[g]    = l_sum[g] * alpha + wt;
                            m_max[g]    = m_new;
                            o_acc[g][0] = o_acc[g][0] * alpha + wt * vt01.x;
                            o_acc[g][1] = o_acc[g][1] * alpha + wt * vt01.y;
                            o_acc[g][2] = o_acc[g][2] * alpha + wt * vt23.x;
                            o_acc[g][3] = o_acc[g][3] * alpha + wt * vt23.y;
                        }
                    }
                }
            } else {
                for (int t = 0; t < tile_cnt; ++t) {
                    int kp = kp_base + t;
                    if (kp > q_pos) break;
                    int dist = q_pos - kp;
                    float4 p_0 = make_float4(0.f, 0.f, 0.f, 0.f);
                    float4 p_1 = make_float4(0.f, 0.f, 0.f, 0.f);
                    float4 p_2 = make_float4(0.f, 0.f, 0.f, 0.f);
                    float4 p_3 = make_float4(0.f, 0.f, 0.f, 0.f);
                    bool has_rel = (dist >= 0 && dist < 1024);
                    if (has_rel) {
                        const float4* rp = reinterpret_cast<const float4*>(rel_proj_t + static_cast<size_t>(dist) * 16);
                        p_0 = __ldg(rp + 0); p_1 = __ldg(rp + 1); p_2 = __ldg(rp + 2); p_3 = __ldg(rp + 3);
                    }
                    if constexpr (KV_Q8) {
                        uint32_t k_pk = s_kv.q8.k[t][lane];
                        uint32_t v_pk = s_kv.q8.v[t][lane];
                        float k_sc    = s_kv.q8.ks[t];
                        float v_sc    = s_kv.q8.vs[t];
                        float vt0, vt1, vt2, vt3;
                        wv_ink_unpack_s8x4_f32(v_pk, vt0, vt1, vt2, vt3);
                        #pragma unroll
                        for (int g = 0; g < 3; ++g) {
                            int32_t dp = __dp4a(static_cast<int32_t>(q_pk[g]), static_cast<int32_t>(k_pk), 0);
                            float qk   = wv_warp_sum_f32(static_cast<float>(dp)) * (q_sc[g] * k_sc);
                            if (has_rel) {
                                float4 r0 = reinterpret_cast<const float4*>(s_r16[wid][g] + 0)[0];
                                float4 r1 = reinterpret_cast<const float4*>(s_r16[wid][g] + 4)[0];
                                float4 r2 = reinterpret_cast<const float4*>(s_r16[wid][g] + 8)[0];
                                float4 r3 = reinterpret_cast<const float4*>(s_r16[wid][g] + 12)[0];
                                qk += r0.x*p_0.x + r0.y*p_0.y + r0.z*p_0.z + r0.w*p_0.w
                                    + r1.x*p_1.x + r1.y*p_1.y + r1.z*p_1.z + r1.w*p_1.w
                                    + r2.x*p_2.x + r2.y*p_2.y + r2.z*p_2.z + r2.w*p_2.w
                                    + r3.x*p_3.x + r3.y*p_3.y + r3.z*p_3.z + r3.w*p_3.w;
                            }
                            float m_new = fmaxf(m_max[g], qk);
                            float alpha = (m_max[g] > -1e29f) ? expf(m_max[g] - m_new) : 0.0f;
                            float wt    = expf(qk - m_new);
                            float wt_v  = wt * v_sc;
                            l_sum[g]    = l_sum[g] * alpha + wt;
                            m_max[g]    = m_new;
                            o_acc[g][0] = o_acc[g][0] * alpha + wt_v * vt0;
                            o_acc[g][1] = o_acc[g][1] * alpha + wt_v * vt1;
                            o_acc[g][2] = o_acc[g][2] * alpha + wt_v * vt2;
                            o_acc[g][3] = o_acc[g][3] * alpha + wt_v * vt3;
                        }
                    } else {
                        uint32_t kl_w = s_kv.bf.kl[t][lane], kh_w = s_kv.bf.kh[t][lane];
                        uint32_t vl_w = s_kv.bf.vl[t][lane], vh_w = s_kv.bf.vh[t][lane];
                        float2 kt01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kl_w));
                        float2 kt23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kh_w));
                        float2 vt01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vl_w));
                        float2 vt23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vh_w));
                        #pragma unroll
                        for (int g = 0; g < 3; ++g) {
                            float qk = qv[g][0] * kt01.x + qv[g][1] * kt01.y + qv[g][2] * kt23.x + qv[g][3] * kt23.y;
                            qk = wv_warp_sum_f32(qk);
                            if (has_rel) {
                                float4 r0 = reinterpret_cast<const float4*>(s_r16[wid][g] + 0)[0];
                                float4 r1 = reinterpret_cast<const float4*>(s_r16[wid][g] + 4)[0];
                                float4 r2 = reinterpret_cast<const float4*>(s_r16[wid][g] + 8)[0];
                                float4 r3 = reinterpret_cast<const float4*>(s_r16[wid][g] + 12)[0];
                                qk += r0.x*p_0.x + r0.y*p_0.y + r0.z*p_0.z + r0.w*p_0.w
                                    + r1.x*p_1.x + r1.y*p_1.y + r1.z*p_1.z + r1.w*p_1.w
                                    + r2.x*p_2.x + r2.y*p_2.y + r2.z*p_2.z + r2.w*p_2.w
                                    + r3.x*p_3.x + r3.y*p_3.y + r3.z*p_3.z + r3.w*p_3.w;
                            }
                            float m_new = fmaxf(m_max[g], qk);
                            float alpha = (m_max[g] > -1e29f) ? expf(m_max[g] - m_new) : 0.0f;
                            float wt    = expf(qk - m_new);
                            l_sum[g]    = l_sum[g] * alpha + wt;
                            m_max[g]    = m_new;
                            o_acc[g][0] = o_acc[g][0] * alpha + wt * vt01.x;
                            o_acc[g][1] = o_acc[g][1] * alpha + wt * vt01.y;
                            o_acc[g][2] = o_acc[g][2] * alpha + wt * vt23.x;
                            o_acc[g][3] = o_acc[g][3] * alpha + wt * vt23.y;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (active_s) {
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            int mh = wid * 24 + hq0 + g;
            size_t idx_sp = static_cast<size_t>(mh) * n_splits + sp;
            if (lane == 0) { m_part[idx_sp] = m_max[g]; l_part[idx_sp] = l_sum[g]; }
            float* od = o_part + idx_sp * 128 + lane * 4;
            od[0] = o_acc[g][0]; od[1] = o_acc[g][1]; od[2] = o_acc[g][2]; od[3] = o_acc[g][3];
        }
    }
    wv_pdl_trigger();
}

// Single-Launch Fused GQA-3 FlashAttention + Sink + Gate + FWHT-128 + Q8_32 (`n_splits == 1`, ZERO `ws_opart` DRAM traffic!):
// Grid `dim3(M * 8)`, Block `dim3(32, 4)` (used in prefill `M >= 16`!)
__global__ void wv_ink_flash_gqa3_nosplit_gate_fwht_q8_kernel(
    const __nv_bfloat16* __restrict__ q_all,      // [B, S, 24, 128]
    const float* __restrict__ r_all_f32,          // [B, S, 24, 16]
    const float* __restrict__ rel_proj_t,         // [1024, 16]
    const __nv_bfloat16* __restrict__ k_cache,    // [B_max, kv_cap, 8, 128]
    const __nv_bfloat16* __restrict__ v_cache,    // [B_max, kv_cap, 8, 128]
    const int32_t* __restrict__ d_slot_pos,       // [B_max]
    const int32_t* __restrict__ d_slot_map,       // [B] or nullptr
    const float* __restrict__ sinks_f32,          // [M, 24]
    const __nv_bfloat16* __restrict__ gate_attn,  // [M, 3072]
    int8_t* __restrict__ O_q8,                    // [M, 3072]
    float* __restrict__ O_sc,                     // [M, 96]
    int S, int kv_cap, int is_local) {
    wv_pdl_wait();
    int mhkv = blockIdx.x;
    int m    = mhkv >> 3;
    int hkv  = mhkv & 7;
    int hq0  = hkv * 3;
    int b    = m / S;
    int s    = m - b * S;
    int lane = threadIdx.x;
    int wid  = threadIdx.y;

    __shared__ float s_r16[3][16];
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    int q_pos  = d_slot_pos[phys_b] + s;
    int kv_beg = is_local ? max(0, q_pos - (INK2_SWA_WIN - 1)) : 0;
    int p1     = q_pos + 1;

    if (wid < 3 && lane < 16) {
        s_r16[wid][lane] = r_all_f32[static_cast<size_t>(m * 24 + hq0 + wid) * 16 + lane];
    }
    __syncthreads();

    constexpr float kScale = 0.08838834764831845f;
    float qv[3][4];
    float m_max[3], l_sum[3], o_acc[3][4];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        const __nv_bfloat16* qp = q_all + static_cast<size_t>(m * 24 + hq0 + g) * 128 + lane * 4;
        qv[g][0] = __bfloat162float(qp[0]) * kScale;
        qv[g][1] = __bfloat162float(qp[1]) * kScale;
        qv[g][2] = __bfloat162float(qp[2]) * kScale;
        qv[g][3] = __bfloat162float(qp[3]) * kScale;
        m_max[g] = -1e30f;
        l_sum[g] = 0.0f;
        o_acc[g][0] = 0.0f; o_acc[g][1] = 0.0f; o_acc[g][2] = 0.0f; o_acc[g][3] = 0.0f;
    }

    const __nv_bfloat16* kb_base = k_cache + static_cast<size_t>(phys_b) * kv_cap * 1024 + hkv * 128 + lane * 4;
    const __nv_bfloat16* vb_base = v_cache + static_cast<size_t>(phys_b) * kv_cap * 1024 + hkv * 128 + lane * 4;

    for (int kp = kv_beg + wid; kp < p1; kp += 4) {
        int slot     = is_local ? (kp % kv_cap) : kp;
        size_t s_off = static_cast<size_t>(slot) * 1024;
        const __nv_bfloat16* kt = kb_base + s_off;
        float kt0 = __bfloat162float(kt[0]), kt1 = __bfloat162float(kt[1]);
        float kt2 = __bfloat162float(kt[2]), kt3 = __bfloat162float(kt[3]);
        const __nv_bfloat16* vt = vb_base + s_off;
        float vt0 = __bfloat162float(vt[0]), vt1 = __bfloat162float(vt[1]);
        float vt2 = __bfloat162float(vt[2]), vt3 = __bfloat162float(vt[3]);

        int dist = q_pos - kp;
        float4 p_0 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 p_1 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 p_2 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 p_3 = make_float4(0.f, 0.f, 0.f, 0.f);
        bool has_rel = (dist >= 0 && dist < 1024);
        if (has_rel) {
            const float4* rp = reinterpret_cast<const float4*>(rel_proj_t + static_cast<size_t>(dist) * 16);
            p_0 = __ldg(rp + 0); p_1 = __ldg(rp + 1); p_2 = __ldg(rp + 2); p_3 = __ldg(rp + 3);
        }

        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            float qk = qv[g][0] * kt0 + qv[g][1] * kt1 + qv[g][2] * kt2 + qv[g][3] * kt3;
            qk = wv_warp_sum_f32(qk);
            if (has_rel) {
                float4 r0 = reinterpret_cast<const float4*>(s_r16[g] + 0)[0];
                float4 r1 = reinterpret_cast<const float4*>(s_r16[g] + 4)[0];
                float4 r2 = reinterpret_cast<const float4*>(s_r16[g] + 8)[0];
                float4 r3 = reinterpret_cast<const float4*>(s_r16[g] + 12)[0];
                qk += r0.x*p_0.x + r0.y*p_0.y + r0.z*p_0.z + r0.w*p_0.w
                    + r1.x*p_1.x + r1.y*p_1.y + r1.z*p_1.z + r1.w*p_1.w
                    + r2.x*p_2.x + r2.y*p_2.y + r2.z*p_2.z + r2.w*p_2.w
                    + r3.x*p_3.x + r3.y*p_3.y + r3.z*p_3.z + r3.w*p_3.w;
            }
            float m_new = fmaxf(m_max[g], qk);
            float alpha = (m_max[g] > -1e29f) ? expf(m_max[g] - m_new) : 0.0f;
            float wt    = expf(qk - m_new);
            l_sum[g]    = l_sum[g] * alpha + wt;
            m_max[g]    = m_new;
            o_acc[g][0] = o_acc[g][0] * alpha + wt * vt0;
            o_acc[g][1] = o_acc[g][1] * alpha + wt * vt1;
            o_acc[g][2] = o_acc[g][2] * alpha + wt * vt2;
            o_acc[g][3] = o_acc[g][3] * alpha + wt * vt3;
        }
    }

    __shared__ float s_m[3][4], s_l[3][4], s_o[3][4][128];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        if (lane == 0) { s_m[g][wid] = m_max[g]; s_l[g][wid] = l_sum[g]; }
        s_o[g][wid][lane * 4 + 0] = o_acc[g][0]; s_o[g][wid][lane * 4 + 1] = o_acc[g][1];
        s_o[g][wid][lane * 4 + 2] = o_acc[g][2]; s_o[g][wid][lane * 4 + 3] = o_acc[g][3];
    }
    __syncthreads();

    if (wid < 3) {
        int g  = wid;
        int mh = m * 24 + hq0 + g;
        float fm = fmaxf(fmaxf(s_m[g][0], s_m[g][1]), fmaxf(s_m[g][2], s_m[g][3]));
        float fl = 0.0f, fo0 = 0.0f, fo1 = 0.0f, fo2 = 0.0f, fo3 = 0.0f;
        if (fm > -1e29f) {
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (s_l[g][w] > 0.0f) {
                    float a = expf(s_m[g][w] - fm);
                    fl  += s_l[g][w] * a;
                    fo0 += s_o[g][w][lane * 4 + 0] * a; fo1 += s_o[g][w][lane * 4 + 1] * a;
                    fo2 += s_o[g][w][lane * 4 + 2] * a; fo3 += s_o[g][w][lane * 4 + 3] * a;
                }
            }
        }
        float sink  = sinks_f32[mh];
        float m_tot = (fl > 0.0f) ? fmaxf(sink, fm) : sink;
        float denom = expf(sink - m_tot);
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        if (fl > 0.0f) {
            float a = expf(fm - m_tot);
            denom += fl * a;
            o0 = fo0 * a; o1 = fo1 * a; o2 = fo2 * a; o3 = fo3 * a;
        }
        float inv_d = 1.0f / fmaxf(denom, 1e-20f);
        float a0 = __bfloat162float(__float2bfloat16(o0 * inv_d));
        float a1 = __bfloat162float(__float2bfloat16(o1 * inv_d));
        float a2 = __bfloat162float(__float2bfloat16(o2 * inv_d));
        float a3 = __bfloat162float(__float2bfloat16(o3 * inv_d));
        const __nv_bfloat16* gp = gate_attn + static_cast<size_t>(mh) * 128 + lane * 4;
        auto sig_bf = [](float a, __nv_bfloat16 gb) -> float {
            float gv = __bfloat162float(gb);
            return __bfloat162float(__float2bfloat16(a * (1.0f / (1.0f + expf(-gv)))));
        };
        float f0 = sig_bf(a0, gp[0]), f1 = sig_bf(a1, gp[1]);
        float f2 = sig_bf(a2, gp[2]), f3 = sig_bf(a3, gp[3]);
        wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
        wv_quant_q8_32_warp(f0, f1, f2, f3, lane, O_q8 + static_cast<size_t>(mh) * 128, O_sc + mh * 4);
    }
    wv_pdl_trigger();
}

} // namespace wv

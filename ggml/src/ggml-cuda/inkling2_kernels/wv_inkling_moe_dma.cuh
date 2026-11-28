#pragma once
#include "wv_inkling_attn.cuh"
#include <algorithm>
#include <cstring>
#include <vector>

namespace wv {

static constexpr int INK2_MAX_STAGE_SLABS = 48;

__device__ __forceinline__ void wv_ink_unpack_vq12_meta(
    const uint16_t* mb20_base, int kc128, int lane_n, float ws_out[4]) {
    wv_unpack_vq12_meta_mb20(mb20_base, kc128, lane_n, ws_out);
}

__device__ __forceinline__ void wv_ink_unpack_vq12_sub4(
    uint32_t w0, uint32_t w1, uint32_t w2, const uint32_t* s_cb4096,
    uint32_t b0[4], uint32_t b1[4]) {
    wv_unpack_vq12_sub4_tab(w0, w1, w2, s_cb4096, b0, b1);
}

__device__ __forceinline__ void wv_ink_unpack_vq12_sub4_ldg(
    uint32_t w0, uint32_t w1, uint32_t w2, const uint32_t* cb4096_gmem,
    uint32_t b0[4], uint32_t b1[4]) {
    wv_unpack_vq12_sub4_ldg_tab(w0, w1, w2, cb4096_gmem, b0, b1);
}

__device__ __forceinline__ void wv_ink_unpack_vq8_sb6_meta(
    const uint16_t* vq_meta_base, int kc128, float ws_out[4]) {
    int kc64_0 = kc128 * 2;
    int sup64  = (kc64_0 / 6) * 6;
    uint16_t d_u16 = static_cast<uint16_t>(
          ((__ldg(vq_meta_base + (sup64 + 0) * 8) >> 12) & 0xFu)
        | (((__ldg(vq_meta_base + (sup64 + 1) * 8) >> 12) & 0xFu) << 4)
        | (((__ldg(vq_meta_base + (sup64 + 2) * 8) >> 12) & 0xFu) << 8)
        | (((__ldg(vq_meta_base + (sup64 + 3) * 8) >> 12) & 0xFu) << 12));
    float ds   = __half2float(__ushort_as_half(d_u16)) * (1.0f / 63.0f);
    uint16_t m0 = __ldg(vq_meta_base + kc64_0 * 8);
    uint16_t m1 = __ldg(vq_meta_base + (kc64_0 + 1) * 8);
    ws_out[0] = __half2float(__float2half(ds * static_cast<float>(m0 & 0x3Fu))) * (1.0f / 127.0f);
    ws_out[1] = __half2float(__float2half(ds * static_cast<float>((m0 >> 6) & 0x3Fu))) * (1.0f / 127.0f);
    ws_out[2] = __half2float(__float2half(ds * static_cast<float>(m1 & 0x3Fu))) * (1.0f / 127.0f);
    ws_out[3] = __half2float(__float2half(ds * static_cast<float>((m1 >> 6) & 0x3Fu))) * (1.0f / 127.0f);
}

__device__ __forceinline__ void wv_ink_unpack_vq8_h(
    uint32_t pkv, int lane, const uint32_t* s_vq8_uv,
    uint32_t& b0_s0, uint32_t& b1_s0, uint32_t& b0_s1, uint32_t& b1_s1) {
    wv_unpack_vq8_h_smem(pkv, lane, s_vq8_uv, b0_s0, b1_s0, b0_s1, b1_s1);
}

__device__ __forceinline__ void wv_ink_unpack_vq8_h_ldg(
    uint32_t pkv, int lane, const uint32_t* vq8_uv_gmem,
    uint32_t& b0_s0, uint32_t& b1_s0, uint32_t& b0_s1, uint32_t& b1_s1) {
    wv_unpack_vq8_h_ldg_tab(pkv, lane, vq8_uv_gmem, b0_s0, b1_s0, b0_s1, b1_s1);
}

// Multi-CTA Coalesced Sigmoid + Bias Top-K Router (n_experts <= 128, top_k <= 8):
// Stage A (<<<dim3((n_experts + 7) / 8, M), dim3(32, 8)>>>)
__global__ void wv_ink_router_gemv_kernel(
    const __nv_bfloat16* __restrict__ x_bf16, // [M, 2304]
    const float* __restrict__ router_w,       // [n_experts, 2304]
    float* __restrict__ router_sc,            // [M, n_experts]
    int n_experts, int M) {
    wv_pdl_wait();
    int e    = blockIdx.x * 8 + threadIdx.y;
    int m    = blockIdx.y;
    if (e >= n_experts || m >= M) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x;

    const __nv_bfloat16* xm = x_bf16 + static_cast<size_t>(m) * INK2_HIDDEN;
    const float4* wr = reinterpret_cast<const float4*>(router_w + static_cast<size_t>(e) * INK2_HIDDEN);
    float acc = 0.0f;
    for (int k4 = lane; k4 < (INK2_HIDDEN / 4); k4 += 32) {
        float4 wv4 = wr[k4];
        uint2  xu  = reinterpret_cast<const uint2*>(xm)[k4];
        const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xu);
        float2 x0 = __bfloat1622float2(xb[0]), x1 = __bfloat1622float2(xb[1]);
        acc += wv4.x * x0.x + wv4.y * x0.y + wv4.z * x1.x + wv4.w * x1.y;
    }
    acc = wv_warp_sum_f32(acc);
    if (lane == 0) {
        router_sc[static_cast<size_t>(m) * n_experts + e] = 1.0f / (1.0f + expf(-acc));
    }
    wv_pdl_trigger();
}

// Stage B (<<<M, 32>>>): Single warp selects top-K by sig + router_b & scales by top_k / sum
__global__ void wv_ink_router_select_topk_kernel(
    const float* __restrict__ router_sc,      // [M, n_experts]
    const float* __restrict__ router_b,       // [n_experts]
    int32_t* __restrict__ top_eid,            // [M, top_k]
    float* __restrict__ top_w,                // [M, top_k]
    int n_experts, int top_k, int M) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m >= M) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x; // 0..31
    const float* sc_m = router_sc + static_cast<size_t>(m) * n_experts;

    float lv[4], ls[4];
    int   li[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int ei = lane * 4 + j;
        if (ei < n_experts) {
            float s = sc_m[ei];
            ls[j] = s;
            lv[j] = s + router_b[ei];
        } else {
            ls[j] = 0.0f;
            lv[j] = -1e30f;
        }
        li[j] = ei;
    }
    int   my_e  = 0;
    float my_s  = 0.0f;
    float w_sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        float bv = -1e30f, bs = 0.0f;
        int   bi = n_experts;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (lv[j] > bv || (lv[j] == bv && li[j] < bi)) { bv = lv[j]; bi = li[j]; bs = ls[j]; }
        }
        #pragma unroll
        for (int mask = 16; mask >= 1; mask >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, mask);
            int   oi = __shfl_xor_sync(0xffffffffu, bi, mask);
            float os = __shfl_xor_sync(0xffffffffu, bs, mask);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bs = os; }
        }
        if (lane == k) {
            my_e = bi;
            my_s = bs;
        }
        w_sum += bs;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (li[j] == bi) lv[j] = -1e30f;
        }
    }
    if (lane < top_k) {
        float norm = static_cast<float>(top_k) / (w_sum + 1e-20f);
        top_eid[m * top_k + lane] = my_e;
        top_w[m * top_k + lane]   = my_s * norm;
    }
    wv_pdl_trigger();
}

// 8-Token Register-Tiled Router GEMV (grid <<<dim3((n_experts + 1)/2, (M + 7)/8), dim3(32, 2)>>>):
__global__ void wv_ink_router_gemv_tiled_kernel(
    const __nv_bfloat16* __restrict__ x_bf16, // [M, 2304]
    const float* __restrict__ router_w,       // [n_experts, 2304]
    float* __restrict__ router_sc,            // [M, n_experts]
    int n_experts, int M) {
    wv_pdl_wait();
    int e      = blockIdx.x * 2 + threadIdx.y;
    int m_base = blockIdx.y * 8;
    int cur_m  = min(8, M - m_base);
    if (e >= n_experts || cur_m <= 0) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x;

    const float4* wr = reinterpret_cast<const float4*>(router_w + static_cast<size_t>(e) * INK2_HIDDEN);
    float acc[8];
    #pragma unroll
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;

    for (int k4 = lane; k4 < (INK2_HIDDEN / 4); k4 += 32) {
        float4 wv = __ldg(wr + k4);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < cur_m) {
                const __nv_bfloat16* xm = x_bf16 + static_cast<size_t>(m_base + t) * INK2_HIDDEN;
                uint2 xu = reinterpret_cast<const uint2*>(xm)[k4];
                const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xu);
                float2 x0 = __bfloat1622float2(xb[0]), x1 = __bfloat1622float2(xb[1]);
                acc[t] += wv.x * x0.x + wv.y * x0.y + wv.z * x1.x + wv.w * x1.y;
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
                router_sc[static_cast<size_t>(m_base + t) * n_experts + e] = 1.0f / (1.0f + expf(-acc[t]));
            }
        }
    }
    wv_pdl_trigger();
}

// Dedicated 128-CTA M == 1 Router GEMV (128 CTAs across 128 distinct SMs, 0 M-branch overhead, 0.0-ULP identical!):
__global__ __launch_bounds__(32, 16)
void wv_ink_router_gemv_m1_kernel(
    const __nv_bfloat16* __restrict__ x_bf16, // [2304]
    const float* __restrict__ router_w,       // [n_experts, 2304]
    float* __restrict__ router_sc,            // [n_experts]
    int n_experts) {
    wv_pdl_wait();
    int e = blockIdx.x;
    if (e >= n_experts) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x;
    const float4* wr = reinterpret_cast<const float4*>(router_w + static_cast<size_t>(e) * INK2_HIDDEN);
    const uint2*  xr = reinterpret_cast<const uint2*>(x_bf16);
    float acc = 0.0f;
    #pragma unroll 9
    for (int k4 = lane; k4 < (INK2_HIDDEN / 4); k4 += 32) {
        float4 wv = __ldg(wr + k4);
        uint2  xu = __ldg(xr + k4);
        const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xu);
        float2 x0 = __bfloat1622float2(xb[0]);
        float2 x1 = __bfloat1622float2(xb[1]);
        acc += wv.x * x0.x + wv.y * x0.y + wv.z * x1.x + wv.w * x1.y;
    }
    acc = wv_warp_sum_f32(acc);
    if (lane == 0) {
        router_sc[e] = 1.0f / (1.0f + expf(-acc));
    }
    wv_pdl_trigger();
}

static inline void launch_wv_ink_router_topk(
    const __nv_bfloat16* x_bf16, const float* router_w, const float* router_b,
    float* ws_router_sc, int32_t* top_eid, float* top_w,
    int n_experts, int top_k, int M, cudaStream_t stream, bool enable_pdl = false, int num_sms = 64) {
    if (M == 1) {
        if (num_sms < 100) {
            int gx = (n_experts + 7) / 8;
            wv_launch_pdl(enable_pdl, dim3(gx, 1), dim3(32, 8), 0, stream,
                wv_ink_router_gemv_kernel, x_bf16, router_w, ws_router_sc, n_experts, 1);
        } else {
            wv_launch_pdl(enable_pdl, dim3(n_experts, 1), dim3(32), 0, stream,
                wv_ink_router_gemv_m1_kernel, x_bf16, router_w, ws_router_sc, n_experts);
        }
    } else {
        int gx2 = (n_experts + 1) / 2;
        wv_launch_pdl(enable_pdl, dim3(gx2, (M + 7) / 8), dim3(32, 2), 0, stream,
            wv_ink_router_gemv_tiled_kernel, x_bf16, router_w, ws_router_sc, n_experts, M);
    }
    wv_launch_pdl(enable_pdl, dim3(M), dim3(32), 0, stream,
        wv_ink_router_select_topk_kernel, ws_router_sc, router_b, top_eid, top_w, n_experts, top_k, M);
}

// Groups M * top_k routed pairs by expert & resolves u_slab_ptr on GPU (all_vram OR tier_slab_ptrs):
// Grid <<<1, 128>>>
__global__ void wv_ink_build_expert_groups_kernel(
    const int32_t* __restrict__ top_eid,          // [M * top_k]
    const int32_t* __restrict__ tmap,             // [n_experts] (>=0 hot_idx, <0 ~cold_idx)
    const uint8_t* __restrict__ all_vram_slabs,   // [E_c, 2239488] if all_vram else nullptr
    const uint64_t* __restrict__ cold_slab_ptrs,  // [128] resolved ptrs if tiered else nullptr
    int32_t* __restrict__ d_num_tiles16,
    int32_t* __restrict__ tile_hot_idx,
    uint64_t* __restrict__ tile_slab_ptr,
    int32_t* __restrict__ tile_beg,
    int32_t* __restrict__ tile_len,
    int32_t* __restrict__ pair_tok,
    int32_t* __restrict__ pair_slot,
    int n_experts, int top_k, int M) {
    wv_pdl_wait();
    int e   = threadIdx.x; // 0..127
    int tot = M * top_k;
    __shared__ int s_cnt[128];
    __shared__ int s_off[128];
    __shared__ int s_tcnt[128];
    __shared__ int s_toff[128];
    __shared__ int s_tot_tiles;

    int cnt = 0;
    if (e < n_experts) {
        for (int p = 0; p < tot; ++p) {
            if (top_eid[p] == e) cnt++;
        }
    }
    s_cnt[e]  = cnt;
    s_tcnt[e] = (cnt + 15) >> 4;
    __syncthreads();

    if (e == 0) {
        int po = 0, to = 0;
        for (int i = 0; i < n_experts; ++i) {
            s_off[i]  = po; po += s_cnt[i];
            s_toff[i] = to; to += s_tcnt[i];
        }
        s_tot_tiles = to;
    }
    __syncthreads();
    if (e == 0) *d_num_tiles16 = s_tot_tiles;

    if (e < n_experts && cnt > 0) {
        int wr_p = s_off[e];
        for (int p = 0; p < tot; ++p) {
            if (top_eid[p] == e) {
                pair_tok[wr_p]  = (top_k == 8) ? (p >> 3) : (p / top_k);
                pair_slot[wr_p] = p;
                wr_p++;
            }
        }
        int tm = tmap[e];
        int h_idx = (tm >= 0) ? tm : -1;
        uint64_t c_ptr = 0ULL;
        if (tm < 0) {
            int ci = ~tm;
            c_ptr = (all_vram_slabs != nullptr)
                ? reinterpret_cast<uint64_t>(all_vram_slabs + static_cast<size_t>(ci) * INK2_COLD_SLAB_BYTES)
                : cold_slab_ptrs[ci];
        }
        int n_t = s_tcnt[e];
        int tb  = s_toff[e];
        for (int t = 0; t < n_t; ++t) {
            tile_hot_idx[tb + t]  = h_idx;
            tile_slab_ptr[tb + t] = c_ptr;
            tile_beg[tb + t]      = s_off[e] + t * 16;
            tile_len[tb + t]      = min(16, cnt - t * 16);
        }
    }
    wv_pdl_trigger();
}

// Fused Router Top-K Selection + Expert Grouping (all_vram and 2 <= M <= 32): Grid <<<1, 1024>>>
__global__ void wv_ink_router_select_and_group_allvram_kernel(
    const float* __restrict__ router_sc,      // [M, n_experts]
    const float* __restrict__ router_b,       // [n_experts]
    const int32_t* __restrict__ tmap,         // [n_experts]
    const uint8_t* __restrict__ all_vram_slabs,
    int32_t* __restrict__ top_eid,            // [M, top_k]
    float* __restrict__ top_w,                // [M, top_k]
    int32_t* __restrict__ d_num_tiles16,
    int32_t* __restrict__ tile_hot_idx,
    uint64_t* __restrict__ tile_slab_ptr,
    int32_t* __restrict__ tile_beg,
    int32_t* __restrict__ tile_len,
    int32_t* __restrict__ pair_tok,
    int32_t* __restrict__ pair_slot,
    int n_experts, int top_k, int M) {
    wv_pdl_wait();
    int tid  = threadIdx.x;
    int wid  = tid >> 5;
    int lane = tid & 31;
    __shared__ int32_t s_top_eid[256];
    __shared__ int s_cnt[128], s_off[128], s_tcnt[128], s_toff[128], s_tot_tiles;

    if (wid < M) {
        const float* sc_m = router_sc + static_cast<size_t>(wid) * n_experts;
        float lv[4], ls[4];
        int   li[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int ei = lane * 4 + j;
            if (ei < n_experts) {
                float s = sc_m[ei];
                ls[j] = s;
                lv[j] = s + router_b[ei];
            } else {
                ls[j] = 0.0f;
                lv[j] = -1e30f;
            }
            li[j] = ei;
        }
        int   my_e  = 0;
        float my_s  = 0.0f;
        float w_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            float bv = -1e30f, bs = 0.0f;
            int   bi = n_experts;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (lv[j] > bv || (lv[j] == bv && li[j] < bi)) { bv = lv[j]; bi = li[j]; bs = ls[j]; }
            }
            #pragma unroll
            for (int mask = 16; mask >= 1; mask >>= 1) {
                float ov = __shfl_xor_sync(0xffffffffu, bv, mask);
                int   oi = __shfl_xor_sync(0xffffffffu, bi, mask);
                float os = __shfl_xor_sync(0xffffffffu, bs, mask);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bs = os; }
            }
            if (lane == k) {
                my_e = bi;
                my_s = bs;
            }
            w_sum += bs;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (li[j] == bi) lv[j] = -1e30f;
            }
        }
        if (lane < top_k) {
            float norm = static_cast<float>(top_k) / (w_sum + 1e-20f);
            int32_t eid = my_e;
            s_top_eid[wid * top_k + lane] = eid;
            top_eid[wid * top_k + lane]   = eid;
            top_w[wid * top_k + lane]     = my_s * norm;
        }
    }
    __syncthreads();

    int tot = M * top_k;
    int cnt = 0;
    if (tid < n_experts) {
        for (int p = 0; p < tot; ++p) {
            if (s_top_eid[p] == tid) cnt++;
        }
        s_cnt[tid]  = cnt;
        s_tcnt[tid] = (cnt + 15) >> 4;
    }
    __syncthreads();

    if (tid == 0) {
        int po = 0, to = 0;
        for (int i = 0; i < n_experts; ++i) {
            s_off[i]  = po; po += s_cnt[i];
            s_toff[i] = to; to += s_tcnt[i];
        }
        s_tot_tiles = to;
    }
    __syncthreads();

    if (tid == 0) *d_num_tiles16 = s_tot_tiles;
    if (tid < n_experts && cnt > 0) {
        int wr_p = s_off[tid];
        for (int p = 0; p < tot; ++p) {
            if (s_top_eid[p] == tid) {
                pair_tok[wr_p]  = (top_k == 8) ? (p >> 3) : (p / top_k);
                pair_slot[wr_p] = p;
                wr_p++;
            }
        }
        int tm = tmap[tid];
        int h_idx = (tm >= 0) ? tm : -1;
        uint64_t c_ptr = (tm < 0)
            ? reinterpret_cast<uint64_t>(all_vram_slabs + static_cast<size_t>(~tm) * INK2_COLD_SLAB_BYTES)
            : 0ULL;
        int n_t = s_tcnt[tid];
        int tb  = s_toff[tid];
        for (int t = 0; t < n_t; ++t) {
            tile_hot_idx[tb + t]  = h_idx;
            tile_slab_ptr[tb + t] = c_ptr;
            tile_beg[tb + t]      = s_off[tid] + t * 16;
            tile_len[tb + t]      = min(16, cnt - t * 16);
        }
    }
    wv_pdl_trigger();
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  #define WV_INK_MOE_LB(B_SM90, B_PRE90) __launch_bounds__(128, B_SM90)
#else
  #define WV_INK_MOE_LB(B_SM90, B_PRE90) __launch_bounds__(128, B_PRE90)
#endif

// Lazy 4 KB SMEM-VQ8 Tensor Core mma.m16n8k32 MoE Gate+Up + Exact Softcapped SwiGLU (4.0f, 25.0f):
// Grid <<<72, grid_y>>>, Block 128
__global__ WV_INK_MOE_LB(6, 5)
void wv_ink_moe_gate_up_swiglu_kernel(
    const int8_t* __restrict__ X_q8,              // [M, 2304]
    const float* __restrict__ X_sc,               // [M, 72]
    const uint32_t* __restrict__ cb4096_gmem,     // [4096]
    const uint32_t* __restrict__ vq8_uv_gmem,     // [1024]
    const uint32_t* __restrict__ hot_gu_pk3,      // [E_h, 288, 18, 3, 32]
    const uint16_t* __restrict__ hot_gu_sc,       // [E_h, 288, 18, 20]
    const int32_t* __restrict__ d_num_tiles16,
    const int32_t* __restrict__ tile_hot_idx,
    const uint64_t* __restrict__ tile_slab_ptr,
    const int32_t* __restrict__ tile_beg,
    const int32_t* __restrict__ tile_len,
    const int32_t* __restrict__ pair_tok,
    const int32_t* __restrict__ pair_slot,
    __nv_bfloat16* __restrict__ H_inter_bf16) {   // [M * top_k, 1152]
    __shared__ uint32_t s_X[16][68];
    __shared__ float    s_up[2][16][8];
    __shared__ uint32_t s_vq8_uv[1024];

    wv_pdl_wait();
    int num_t = *d_num_tiles16;
    if (num_t <= 0) { wv_pdl_trigger(); return; }

    int warp_id   = threadIdx.x >> 5; // 0,2 = gate; 1,3 = up
    int lane      = threadIdx.x & 31;
    int lane_n    = lane >> 2;
    int c_k       = lane & 3;
    int r0        = lane >> 2, r1 = r0 + 8;
    int col_n0    = c_k * 2, col_n1 = col_n0 + 1;
    int tile8     = blockIdx.x * 4 + warp_id; // 0..287
    bool vq8_ready = false;

    for (int ti = blockIdx.y; ti < num_t; ti += gridDim.y) {
        int h_idx = tile_hot_idx[ti];
        const uint8_t* slab = reinterpret_cast<const uint8_t*>(tile_slab_ptr[ti]);
        int beg   = tile_beg[ti];
        int len   = tile_len[ti];

        if (h_idx < 0 && !vq8_ready) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                int ix = u * 128 + threadIdx.x;
                s_vq8_uv[ix] = __ldg(vq8_uv_gmem + ix);
            }
            vq8_ready = true;
        }

        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        const uint32_t* h_pk_b = (h_idx >= 0) ? (hot_gu_pk3 + (static_cast<size_t>(h_idx) * 288 + tile8) * (18 * 96) + lane) : nullptr;
        const uint16_t* h_sc_b = (h_idx >= 0) ? (hot_gu_sc  + (static_cast<size_t>(h_idx) * 288 + tile8) * (18 * 20)) : nullptr;
        const uint32_t* c_pk_b = (h_idx < 0)  ? (reinterpret_cast<const uint32_t*>(slab + INK2_OFF_GU_PK) + static_cast<size_t>(tile8) * (36 * 32) + lane) : nullptr;
        const uint16_t* c_sc_b = (h_idx < 0)  ? (reinterpret_cast<const uint16_t*>(slab + INK2_OFF_GU_SC) + static_cast<size_t>(tile8) * (36 * 8) + lane_n) : nullptr;

        for (int kc256 = 0; kc256 < 18; kc256 += 2) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                int idx  = u * 128 + threadIdx.x;
                int ld_m = idx >> 4;
                int ld_q = idx & 15;
                uint4 xv = (ld_m < len)
                    ? reinterpret_cast<const uint4*>(X_q8 + static_cast<size_t>(pair_tok[beg + ld_m]) * INK2_HIDDEN + kc256 * 128)[ld_q]
                    : make_uint4(0, 0, 0, 0);
                int c4 = ld_q * 4;
                s_X[ld_m][c4 + 0] = xv.x; s_X[ld_m][c4 + 1] = xv.y;
                s_X[ld_m][c4 + 2] = xv.z; s_X[ld_m][c4 + 3] = xv.w;
            }
            __syncthreads();

            int m_tok0 = (r0 < len) ? pair_tok[beg + r0] : 0;
            int m_tok1 = (r1 < len) ? pair_tok[beg + r1] : 0;
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                int kc128 = kc256 + c;
                uint32_t b0[4], b1[4];
                float ws[4];
                if (h_idx >= 0) {
                    const uint32_t* pk = h_pk_b + kc128 * 96;
                    wv_ink_unpack_vq12_meta(h_sc_b, kc128, lane_n, ws);
                    wv_ink_unpack_vq12_sub4_ldg(__ldg(pk), __ldg(pk + 32), __ldg(pk + 64), cb4096_gmem, b0, b1);
                } else {
                    wv_ink_unpack_vq8_sb6_meta(c_sc_b, kc128, ws);
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        uint32_t pkv = __ldg(c_pk_b + (kc128 * 2 + h) * 32);
                        wv_ink_unpack_vq8_h(pkv, lane, s_vq8_uv, b0[2 * h], b1[2 * h], b0[2 * h + 1], b1[2 * h + 1]);
                    }
                }
                #pragma unroll
                for (int sub = 0; sub < 4; ++sub) {
                    int kc32  = kc128 * 4 + sub;
                    int s_off = (c * 4 + sub) * 8 + c_k;
                    float ws0 = __shfl_sync(0xffffffffu, ws[sub], col_n0 * 4);
                    float ws1 = __shfl_sync(0xffffffffu, ws[sub], col_n1 * 4);
                    float xs0 = (r0 < len) ? X_sc[static_cast<size_t>(m_tok0) * 72 + kc32] : 0.0f;
                    float xs1 = (r1 < len) ? X_sc[static_cast<size_t>(m_tok1) * 72 + kc32] : 0.0f;
                    int d0, d1, d2, d3;
                    wv_mma_m16n8k32_s8(d0, d1, d2, d3,
                        s_X[r0][s_off], s_X[r1][s_off], s_X[r0][s_off + 4], s_X[r1][s_off + 4],
                        b0[sub], b1[sub]);
                    acc0 += static_cast<float>(d0) * (ws0 * xs0); acc1 += static_cast<float>(d1) * (ws1 * xs0);
                    acc2 += static_cast<float>(d2) * (ws0 * xs1); acc3 += static_cast<float>(d3) * (ws1 * xs1);
                }
            }
            __syncthreads();
        }

        if ((warp_id & 1) != 0) {
            int p_id = warp_id >> 1;
            s_up[p_id][r0][col_n0] = acc0; s_up[p_id][r0][col_n1] = acc1;
            s_up[p_id][r1][col_n0] = acc2; s_up[p_id][r1][col_n1] = acc3;
        }
        __syncthreads();

        if ((warp_id & 1) == 0) {
            int p_id   = warp_id >> 1;
            int inter8 = blockIdx.x * 2 + p_id; // 0..143 (144 * 8 = 1152)
            auto swig = [](float g, float u) -> __nv_bfloat16 {
                float gl = 4.0f * tanhf(g * 0.25f);
                float ul = 25.0f * tanhf(u * 0.04f);
                return __float2bfloat16(gl * (1.0f / (1.0f + expf(-g))) * ul);
            };
            if (r0 < len) {
                __nv_bfloat16* dst = H_inter_bf16 + static_cast<size_t>(pair_slot[beg + r0]) * INK2_INTER + inter8 * 8;
                dst[col_n0] = swig(acc0, s_up[p_id][r0][col_n0]);
                dst[col_n1] = swig(acc1, s_up[p_id][r0][col_n1]);
            }
            if (r1 < len) {
                __nv_bfloat16* dst = H_inter_bf16 + static_cast<size_t>(pair_slot[beg + r1]) * INK2_INTER + inter8 * 8;
                dst[col_n0] = swig(acc2, s_up[p_id][r1][col_n0]);
                dst[col_n1] = swig(acc3, s_up[p_id][r1][col_n1]);
            }
        }
        __syncthreads();
    }
    wv_pdl_trigger();
}

// Fast 4 KB SMEM-VQ8 __dp4a MoE Gate+Up + SwiGLU for M == 1: Grid <<<dim3(72, top_k), 128>>>
__global__ __launch_bounds__(128, 6)
void wv_ink_moe_gate_up_swiglu_dp4a_m1_kernel(
    const int8_t* __restrict__ X_q8,              // [2304]
    const float* __restrict__ X_sc,               // [72]
    const uint32_t* __restrict__ cb4096_gmem,     // [4096]
    const uint32_t* __restrict__ vq8_uv_gmem,     // [1024]
    const uint32_t* __restrict__ hot_gu_pk3,      // [E_h, 288, 18, 3, 32]
    const uint16_t* __restrict__ hot_gu_sc,       // [E_h, 288, 18, 20]
    const int32_t* __restrict__ top_eid,          // [top_k]
    const int32_t* __restrict__ tmap,             // [n_experts]
    const uint8_t* __restrict__ all_vram_slabs,   // [E_c, 2239488] if all_vram else nullptr
    const uint64_t* __restrict__ cold_slab_ptrs,  // [128] resolved ptrs if tiered else nullptr
    __nv_bfloat16* __restrict__ H_inter_bf16) {   // [top_k, 1152]
    __shared__ uint32_t s_vq8_uv[1024];
    __shared__ int32_t  s_xq[576];
    __shared__ float    s_xs[72];
    __shared__ float    s_up_m1[2][8];
    wv_pdl_wait();
    int k_slot  = blockIdx.y; // 0 .. top_k - 1
    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int lane_n  = lane >> 2;
    int c_k     = lane & 3;
    int tile8   = blockIdx.x * 4 + warp_id; // 0..287

    const int32_t* xp_g = reinterpret_cast<const int32_t*>(X_q8);
    for (int q = tid; q < 576; q += 128) s_xq[q] = __ldg(xp_g + q);
    if (tid < 72) s_xs[tid] = __ldg(X_sc + tid);

    int e     = top_eid[k_slot];
    int tm    = tmap[e];
    int h_idx = (tm >= 0) ? tm : -1;
    const uint8_t* slab = nullptr;
    if (h_idx < 0) {
        int ci = ~tm;
        slab = (all_vram_slabs != nullptr)
            ? (all_vram_slabs + static_cast<size_t>(ci) * INK2_COLD_SLAB_BYTES)
            : reinterpret_cast<const uint8_t*>(cold_slab_ptrs[ci]);
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            int ix = u * 128 + tid;
            s_vq8_uv[ix] = __ldg(vq8_uv_gmem + ix);
        }
    }
    __syncthreads();

    float acc = 0.0f;
    if (h_idx >= 0) {
        const uint32_t* h_pk_b = hot_gu_pk3 + (static_cast<size_t>(h_idx) * 288 + tile8) * (18 * 96) + lane;
        const uint16_t* h_sc_b = hot_gu_sc  + (static_cast<size_t>(h_idx) * 288 + tile8) * (18 * 20);
        #pragma unroll 2
        for (int kc128 = 0; kc128 < 18; ++kc128) {
            uint32_t b0[4], b1[4];
            float ws[4];
            const uint32_t* pk = h_pk_b + kc128 * 96;
            wv_ink_unpack_vq12_meta(h_sc_b, kc128, lane_n, ws);
            wv_ink_unpack_vq12_sub4_ldg(__ldg(pk), __ldg(pk + 32), __ldg(pk + 64), cb4096_gmem, b0, b1);
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int kc32 = kc128 * 4 + sub;
                const int32_t* xp = s_xq + kc32 * 8;
                int32_t d = __dp4a(xp[c_k], static_cast<int32_t>(b0[sub]), __dp4a(xp[4 + c_k], static_cast<int32_t>(b1[sub]), 0));
                d += __shfl_xor_sync(0xffffffffu, d, 2);
                d += __shfl_xor_sync(0xffffffffu, d, 1);
                acc += static_cast<float>(d) * (ws[sub] * s_xs[kc32]);
            }
        }
    } else {
        const uint32_t* c_pk_b = reinterpret_cast<const uint32_t*>(slab + INK2_OFF_GU_PK) + static_cast<size_t>(tile8) * (36 * 32) + lane;
        const uint16_t* c_sc_b = reinterpret_cast<const uint16_t*>(slab + INK2_OFF_GU_SC) + static_cast<size_t>(tile8) * (36 * 8) + lane_n;
        #pragma unroll 2
        for (int kc128 = 0; kc128 < 18; ++kc128) {
            uint32_t b0[4], b1[4];
            float ws[4];
            wv_ink_unpack_vq8_sb6_meta(c_sc_b, kc128, ws);
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t pkv = __ldg(c_pk_b + (kc128 * 2 + h) * 32);
                wv_ink_unpack_vq8_h(pkv, lane, s_vq8_uv, b0[2 * h], b1[2 * h], b0[2 * h + 1], b1[2 * h + 1]);
            }
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int kc32 = kc128 * 4 + sub;
                const int32_t* xp = s_xq + kc32 * 8;
                int32_t d = __dp4a(xp[c_k], static_cast<int32_t>(b0[sub]), __dp4a(xp[4 + c_k], static_cast<int32_t>(b1[sub]), 0));
                d += __shfl_xor_sync(0xffffffffu, d, 2);
                d += __shfl_xor_sync(0xffffffffu, d, 1);
                acc += static_cast<float>(d) * (ws[sub] * s_xs[kc32]);
            }
        }
    }

    if ((warp_id & 1) != 0 && c_k == 0) {
        s_up_m1[warp_id >> 1][lane_n] = acc;
    }
    __syncthreads();
    if ((warp_id & 1) == 0 && c_k == 0) {
        int p_id   = warp_id >> 1;
        int inter8 = blockIdx.x * 2 + p_id;
        float u  = s_up_m1[p_id][lane_n];
        float gl = 4.0f * tanhf(acc * 0.25f);
        float ul = 25.0f * tanhf(u * 0.04f);
        H_inter_bf16[static_cast<size_t>(k_slot) * INK2_INTER + inter8 * 8 + lane_n] =
            __float2bfloat16(gl * (1.0f / (1.0f + expf(-acc))) * ul);
    }
    wv_pdl_trigger();
}

// Specialized CTA_M = 8 Tensor Core mma.m16n8k32 + kc768 + Lazy 4 KB SMEM-VQ8 MoE Gate+Up + SwiGLU (2 <= M <= 8):
__global__ WV_INK_MOE_LB(5, 4)
void wv_ink_moe_gate_up_swiglu_mma_m8_kernel(
    const int8_t* __restrict__ X_q8,              // [M, 2304]
    const float* __restrict__ X_sc,               // [M, 72]
    const uint32_t* __restrict__ cb4096_gmem,     // [4096]
    const uint32_t* __restrict__ vq8_uv_gmem,     // [1024]
    const uint32_t* __restrict__ hot_gu_pk3,      // [E_h, 288, 18, 3, 32]
    const uint16_t* __restrict__ hot_gu_sc,       // [E_h, 288, 18, 20]
    const int32_t* __restrict__ d_num_tiles16,
    const int32_t* __restrict__ tile_hot_idx,
    const uint64_t* __restrict__ tile_slab_ptr,
    const int32_t* __restrict__ tile_beg,
    const int32_t* __restrict__ tile_len,
    const int32_t* __restrict__ pair_tok,
    const int32_t* __restrict__ pair_slot,
    __nv_bfloat16* __restrict__ H_inter_bf16) {   // [M * top_k, 1152]
    __shared__ uint32_t s_X[8][196];
    __shared__ float    s_up[2][8][8];
    __shared__ uint32_t s_vq8_uv[1024];

    wv_pdl_wait();
    int num_t = *d_num_tiles16;
    if (num_t <= 0) { wv_pdl_trigger(); return; }

    int warp_id   = threadIdx.x >> 5; // 0,2 = gate; 1,3 = up
    int lane      = threadIdx.x & 31;
    int lane_n    = lane >> 2;
    int c_k       = lane & 3;
    int r0        = lane >> 2;
    int col_n0    = c_k * 2, col_n1 = col_n0 + 1;
    int tile8     = blockIdx.x * 4 + warp_id; // 0..287
    int ld_m      = threadIdx.x >> 4;         // 0..7
    int ld_q0     = threadIdx.x & 15;         // 0..15
    bool vq8_ready = false;

    for (int ti = blockIdx.y; ti < num_t; ti += gridDim.y) {
        int h_idx = tile_hot_idx[ti];
        const uint8_t* slab = reinterpret_cast<const uint8_t*>(tile_slab_ptr[ti]);
        int beg   = tile_beg[ti];
        int len   = min(8, tile_len[ti]);

        if (h_idx < 0 && !vq8_ready) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                int ix = u * 128 + threadIdx.x;
                s_vq8_uv[ix] = __ldg(vq8_uv_gmem + ix);
            }
            vq8_ready = true;
        }

        float acc0 = 0.f, acc1 = 0.f;
        const uint32_t* h_pk_b = (h_idx >= 0) ? (hot_gu_pk3 + (static_cast<size_t>(h_idx) * 288 + tile8) * (18 * 96) + lane) : nullptr;
        const uint16_t* h_sc_b = (h_idx >= 0) ? (hot_gu_sc  + (static_cast<size_t>(h_idx) * 288 + tile8) * (18 * 20)) : nullptr;
        const uint32_t* c_pk_b = (h_idx < 0)  ? (reinterpret_cast<const uint32_t*>(slab + INK2_OFF_GU_PK) + static_cast<size_t>(tile8) * (36 * 32) + lane) : nullptr;
        const uint16_t* c_sc_b = (h_idx < 0)  ? (reinterpret_cast<const uint16_t*>(slab + INK2_OFF_GU_SC) + static_cast<size_t>(tile8) * (36 * 8) + lane_n) : nullptr;

        int ld_tok = (ld_m < len) ? pair_tok[beg + ld_m] : 0;
        int m_tok0 = (r0 < len)   ? pair_tok[beg + r0]   : 0;

        for (int kc_base = 0; kc_base < 18; kc_base += 6) {
            const uint4* xp_row = (ld_m < len)
                ? reinterpret_cast<const uint4*>(X_q8 + static_cast<size_t>(ld_tok) * INK2_HIDDEN + kc_base * 128)
                : nullptr;
            #pragma unroll
            for (int u = 0; u < 3; ++u) {
                int ld_q = u * 16 + ld_q0; // 0..47 (48 * 16B = 768B)
                uint4 xv = (xp_row != nullptr) ? xp_row[ld_q] : make_uint4(0, 0, 0, 0);
                int c4   = ld_q * 4;
                s_X[ld_m][c4 + 0] = xv.x; s_X[ld_m][c4 + 1] = xv.y;
                s_X[ld_m][c4 + 2] = xv.z; s_X[ld_m][c4 + 3] = xv.w;
            }
            __syncthreads();

            for (int c = 0; c < 6; ++c) {
                int kc128 = kc_base + c;
                uint32_t b0[4], b1[4];
                float ws[4];
                if (h_idx >= 0) {
                    const uint32_t* pk = h_pk_b + kc128 * 96;
                    wv_ink_unpack_vq12_meta(h_sc_b, kc128, lane_n, ws);
                    wv_ink_unpack_vq12_sub4_ldg(__ldg(pk), __ldg(pk + 32), __ldg(pk + 64), cb4096_gmem, b0, b1);
                } else {
                    wv_ink_unpack_vq8_sb6_meta(c_sc_b, kc128, ws);
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        uint32_t pkv = __ldg(c_pk_b + (kc128 * 2 + h) * 32);
                        wv_ink_unpack_vq8_h(pkv, lane, s_vq8_uv, b0[2 * h], b1[2 * h], b0[2 * h + 1], b1[2 * h + 1]);
                    }
                }
                #pragma unroll
                for (int sub = 0; sub < 4; ++sub) {
                    int kc32  = kc128 * 4 + sub;
                    int s_off = (c * 4 + sub) * 8 + c_k;
                    float ws0 = __shfl_sync(0xffffffffu, ws[sub], col_n0 * 4);
                    float ws1 = __shfl_sync(0xffffffffu, ws[sub], col_n1 * 4);
                    float xs0 = (r0 < len) ? X_sc[static_cast<size_t>(m_tok0) * 72 + kc32] : 0.0f;
                    int d0, d1, d2, d3;
                    wv_mma_m16n8k32_s8(d0, d1, d2, d3, s_X[r0][s_off], 0u, s_X[r0][s_off + 4], 0u, b0[sub], b1[sub]);
                    acc0 += static_cast<float>(d0) * (ws0 * xs0);
                    acc1 += static_cast<float>(d1) * (ws1 * xs0);
                }
            }
            __syncthreads();
        }

        if ((warp_id & 1) != 0) {
            int p_id = warp_id >> 1;
            s_up[p_id][r0][col_n0] = acc0;
            s_up[p_id][r0][col_n1] = acc1;
        }
        __syncthreads();

        if ((warp_id & 1) == 0 && r0 < len) {
            int p_id   = warp_id >> 1;
            int inter8 = blockIdx.x * 2 + p_id;
            auto swig = [](float g, float u) -> __nv_bfloat16 {
                float gl = 4.0f * tanhf(g * 0.25f);
                float ul = 25.0f * tanhf(u * 0.04f);
                return __float2bfloat16(gl * (1.0f / (1.0f + expf(-g))) * ul);
            };
            __nv_bfloat16* dst = H_inter_bf16 + static_cast<size_t>(pair_slot[beg + r0]) * INK2_INTER + inter8 * 8;
            dst[col_n0] = swig(acc0, s_up[p_id][r0][col_n0]);
            dst[col_n1] = swig(acc1, s_up[p_id][r0][col_n1]);
        }
        __syncthreads();
    }
    wv_pdl_trigger();
}

// Lazy 4 KB SMEM-VQ8 Tensor Core mma.m16n8k32 MoE Down (K = 1152 = 9 * 128, N = 2304 = 288 * 8):
// Grid <<<72, grid_y>>>, Block 128
__global__ WV_INK_MOE_LB(6, 4)
void wv_ink_moe_down_kernel(
    const int8_t* __restrict__ H_q8,              // [M * top_k, 1152]
    const float* __restrict__ H_sc,               // [M * top_k, 36]
    const uint32_t* __restrict__ cb4096_gmem,     // [4096]
    const uint32_t* __restrict__ vq8_uv_gmem,     // [1024]
    const uint32_t* __restrict__ hot_dn_pk3,      // [E_h, 288, 9, 3, 32]
    const uint16_t* __restrict__ hot_dn_sc,       // [E_h, 288, 9, 20]
    const int32_t* __restrict__ d_num_tiles16,
    const int32_t* __restrict__ tile_hot_idx,
    const uint64_t* __restrict__ tile_slab_ptr,
    const int32_t* __restrict__ tile_beg,
    const int32_t* __restrict__ tile_len,
    const int32_t* __restrict__ pair_slot,
    __nv_bfloat16* __restrict__ slot_out_bf16) {  // [M * top_k, 2304]
    __shared__ uint32_t s_H[16][36];
    __shared__ uint32_t s_vq8_uv[1024];

    wv_pdl_wait();
    int num_t = *d_num_tiles16;
    if (num_t <= 0) { wv_pdl_trigger(); return; }

    int warp_id   = threadIdx.x >> 5;
    int lane      = threadIdx.x & 31;
    int lane_n    = lane >> 2;
    int c_k       = lane & 3;
    int r0        = lane >> 2, r1 = r0 + 8;
    int col_n0    = c_k * 2, col_n1 = col_n0 + 1;
    int tile8     = blockIdx.x * 4 + warp_id; // 0..287
    bool vq8_ready = false;

    for (int ti = blockIdx.y; ti < num_t; ti += gridDim.y) {
        int h_idx = tile_hot_idx[ti];
        const uint8_t* slab = reinterpret_cast<const uint8_t*>(tile_slab_ptr[ti]);
        int beg   = tile_beg[ti];
        int len   = tile_len[ti];

        if (h_idx < 0 && !vq8_ready) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                int ix = u * 128 + threadIdx.x;
                s_vq8_uv[ix] = __ldg(vq8_uv_gmem + ix);
            }
            vq8_ready = true;
        }

        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        const uint32_t* h_pk_b = (h_idx >= 0) ? (hot_dn_pk3 + (static_cast<size_t>(h_idx) * 288 + tile8) * (9 * 96) + lane) : nullptr;
        const uint16_t* h_sc_b = (h_idx >= 0) ? (hot_dn_sc  + (static_cast<size_t>(h_idx) * 288 + tile8) * (9 * 20)) : nullptr;
        const uint32_t* c_pk_b = (h_idx < 0)  ? (reinterpret_cast<const uint32_t*>(slab + INK2_OFF_DN_PK) + static_cast<size_t>(tile8) * (18 * 32) + lane) : nullptr;
        const uint16_t* c_sc_b = (h_idx < 0)  ? (reinterpret_cast<const uint16_t*>(slab + INK2_OFF_DN_SC) + static_cast<size_t>(tile8) * (18 * 8) + lane_n) : nullptr;

        int sl0 = (r0 < len) ? pair_slot[beg + r0] : 0;
        int sl1 = (r1 < len) ? pair_slot[beg + r1] : 0;
        for (int kc128 = 0; kc128 < 9; ++kc128) {
            int ld_m = threadIdx.x >> 3; // 0..15
            int ld_q = threadIdx.x & 7;  // 0..7 -> 128B
            uint4 hv = (ld_m < len)
                ? reinterpret_cast<const uint4*>(H_q8 + static_cast<size_t>(pair_slot[beg + ld_m]) * INK2_INTER + kc128 * 128)[ld_q]
                : make_uint4(0, 0, 0, 0);
            int c4 = ld_q * 4;
            s_H[ld_m][c4 + 0] = hv.x; s_H[ld_m][c4 + 1] = hv.y;
            s_H[ld_m][c4 + 2] = hv.z; s_H[ld_m][c4 + 3] = hv.w;
            __syncthreads();

            uint32_t b0[4], b1[4];
            float ws[4];
            if (h_idx >= 0) {
                const uint32_t* pk = h_pk_b + kc128 * 96;
                wv_ink_unpack_vq12_meta(h_sc_b, kc128, lane_n, ws);
                wv_ink_unpack_vq12_sub4_ldg(__ldg(pk), __ldg(pk + 32), __ldg(pk + 64), cb4096_gmem, b0, b1);
            } else {
                wv_ink_unpack_vq8_sb6_meta(c_sc_b, kc128, ws);
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t pkv = __ldg(c_pk_b + (kc128 * 2 + h) * 32);
                    wv_ink_unpack_vq8_h(pkv, lane, s_vq8_uv, b0[2 * h], b1[2 * h], b0[2 * h + 1], b1[2 * h + 1]);
                }
            }
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int kc32  = kc128 * 4 + sub;
                int s_off = sub * 8 + c_k;
                float ws0 = __shfl_sync(0xffffffffu, ws[sub], col_n0 * 4);
                float ws1 = __shfl_sync(0xffffffffu, ws[sub], col_n1 * 4);
                float xs0 = (r0 < len) ? H_sc[static_cast<size_t>(sl0) * 36 + kc32] : 0.0f;
                float xs1 = (r1 < len) ? H_sc[static_cast<size_t>(sl1) * 36 + kc32] : 0.0f;
                int d0, d1, d2, d3;
                wv_mma_m16n8k32_s8(d0, d1, d2, d3,
                    s_H[r0][s_off], s_H[r1][s_off], s_H[r0][s_off + 4], s_H[r1][s_off + 4],
                    b0[sub], b1[sub]);
                acc0 += static_cast<float>(d0) * (ws0 * xs0); acc1 += static_cast<float>(d1) * (ws1 * xs0);
                acc2 += static_cast<float>(d2) * (ws0 * xs1); acc3 += static_cast<float>(d3) * (ws1 * xs1);
            }
            __syncthreads();
        }

        if (r0 < len) {
            __nv_bfloat16* dst = slot_out_bf16 + static_cast<size_t>(sl0) * INK2_HIDDEN + tile8 * 8;
            dst[col_n0] = __float2bfloat16(acc0); dst[col_n1] = __float2bfloat16(acc1);
        }
        if (r1 < len) {
            __nv_bfloat16* dst = slot_out_bf16 + static_cast<size_t>(sl1) * INK2_HIDDEN + tile8 * 8;
            dst[col_n0] = __float2bfloat16(acc2); dst[col_n1] = __float2bfloat16(acc3);
        }
    }
    wv_pdl_trigger();
}

// Fast 4 KB SMEM-VQ8 __dp4a MoE Down for M == 1: Grid <<<dim3(72, top_k), 128>>>
__global__ __launch_bounds__(128, 6)
void wv_ink_moe_down_dp4a_m1_kernel(
    const int8_t* __restrict__ H_q8,              // [top_k, 1152]
    const float* __restrict__ H_sc,               // [top_k, 36]
    const uint32_t* __restrict__ cb4096_gmem,     // [4096]
    const uint32_t* __restrict__ vq8_uv_gmem,     // [1024]
    const uint32_t* __restrict__ hot_dn_pk3,      // [E_h, 288, 9, 3, 32]
    const uint16_t* __restrict__ hot_dn_sc,       // [E_h, 288, 9, 20]
    const int32_t* __restrict__ top_eid,          // [top_k]
    const int32_t* __restrict__ tmap,             // [n_experts]
    const uint8_t* __restrict__ all_vram_slabs,   // [E_c, 2239488] if all_vram else nullptr
    const uint64_t* __restrict__ cold_slab_ptrs,  // [128] resolved ptrs if tiered else nullptr
    __nv_bfloat16* __restrict__ slot_out_bf16) {  // [top_k, 2304]
    __shared__ uint32_t s_vq8_uv[1024];
    __shared__ int32_t  s_hq[288];
    __shared__ float    s_hs[36];
    wv_pdl_wait();
    int k_slot  = blockIdx.y; // 0 .. top_k - 1
    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int lane_n  = lane >> 2;
    int c_k     = lane & 3;
    int tile8   = blockIdx.x * 4 + warp_id; // 0..287

    const int32_t* hq_g = reinterpret_cast<const int32_t*>(H_q8 + static_cast<size_t>(k_slot) * INK2_INTER);
    const float*   hs_g = H_sc + static_cast<size_t>(k_slot) * 36;
    for (int q = tid; q < 288; q += 128) s_hq[q] = __ldg(hq_g + q);
    if (tid < 36) s_hs[tid] = __ldg(hs_g + tid);

    int e     = top_eid[k_slot];
    int tm    = tmap[e];
    int h_idx = (tm >= 0) ? tm : -1;
    const uint8_t* slab = nullptr;
    if (h_idx < 0) {
        int ci = ~tm;
        slab = (all_vram_slabs != nullptr)
            ? (all_vram_slabs + static_cast<size_t>(ci) * INK2_COLD_SLAB_BYTES)
            : reinterpret_cast<const uint8_t*>(cold_slab_ptrs[ci]);
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            int ix = u * 128 + tid;
            s_vq8_uv[ix] = __ldg(vq8_uv_gmem + ix);
        }
    }
    __syncthreads();

    float acc = 0.0f;
    if (h_idx >= 0) {
        const uint32_t* h_pk_b = hot_dn_pk3 + (static_cast<size_t>(h_idx) * 288 + tile8) * (9 * 96) + lane;
        const uint16_t* h_sc_b = hot_dn_sc  + (static_cast<size_t>(h_idx) * 288 + tile8) * (9 * 20);
        #pragma unroll 3
        for (int kc128 = 0; kc128 < 9; ++kc128) {
            uint32_t b0[4], b1[4];
            float ws[4];
            const uint32_t* pk = h_pk_b + kc128 * 96;
            wv_ink_unpack_vq12_meta(h_sc_b, kc128, lane_n, ws);
            wv_ink_unpack_vq12_sub4_ldg(__ldg(pk), __ldg(pk + 32), __ldg(pk + 64), cb4096_gmem, b0, b1);
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int kc32 = kc128 * 4 + sub;
                const int32_t* xp = s_hq + kc32 * 8;
                int32_t d = __dp4a(xp[c_k], static_cast<int32_t>(b0[sub]), __dp4a(xp[4 + c_k], static_cast<int32_t>(b1[sub]), 0));
                d += __shfl_xor_sync(0xffffffffu, d, 2);
                d += __shfl_xor_sync(0xffffffffu, d, 1);
                acc += static_cast<float>(d) * (ws[sub] * s_hs[kc32]);
            }
        }
    } else {
        const uint32_t* c_pk_b = reinterpret_cast<const uint32_t*>(slab + INK2_OFF_DN_PK) + static_cast<size_t>(tile8) * (18 * 32) + lane;
        const uint16_t* c_sc_b = reinterpret_cast<const uint16_t*>(slab + INK2_OFF_DN_SC) + static_cast<size_t>(tile8) * (18 * 8) + lane_n;
        #pragma unroll 3
        for (int kc128 = 0; kc128 < 9; ++kc128) {
            uint32_t b0[4], b1[4];
            float ws[4];
            wv_ink_unpack_vq8_sb6_meta(c_sc_b, kc128, ws);
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t pkv = __ldg(c_pk_b + (kc128 * 2 + h) * 32);
                wv_ink_unpack_vq8_h(pkv, lane, s_vq8_uv, b0[2 * h], b1[2 * h], b0[2 * h + 1], b1[2 * h + 1]);
            }
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int kc32 = kc128 * 4 + sub;
                const int32_t* xp = s_hq + kc32 * 8;
                int32_t d = __dp4a(xp[c_k], static_cast<int32_t>(b0[sub]), __dp4a(xp[4 + c_k], static_cast<int32_t>(b1[sub]), 0));
                d += __shfl_xor_sync(0xffffffffu, d, 2);
                d += __shfl_xor_sync(0xffffffffu, d, 1);
                acc += static_cast<float>(d) * (ws[sub] * s_hs[kc32]);
            }
        }
    }
    if (c_k == 0) {
        slot_out_bf16[static_cast<size_t>(k_slot) * INK2_HIDDEN + tile8 * 8 + lane_n] = __float2bfloat16(acc);
    }
    wv_pdl_trigger();
}

// Specialized CTA_M = 8 Tensor Core mma.m16n8k32 + Full-K=1152 SMEM + Lazy 4 KB SMEM-VQ8 MoE Down (2 <= M <= 8):
__global__ WV_INK_MOE_LB(5, 3)
void wv_ink_moe_down_mma_m8_kernel(
    const int8_t* __restrict__ H_q8,              // [M * top_k, 1152]
    const float* __restrict__ H_sc,               // [M * top_k, 36]
    const uint32_t* __restrict__ cb4096_gmem,     // [4096]
    const uint32_t* __restrict__ vq8_uv_gmem,     // [1024]
    const uint32_t* __restrict__ hot_dn_pk3,      // [E_h, 288, 9, 3, 32]
    const uint16_t* __restrict__ hot_dn_sc,       // [E_h, 288, 9, 20]
    const int32_t* __restrict__ d_num_tiles16,
    const int32_t* __restrict__ tile_hot_idx,
    const uint64_t* __restrict__ tile_slab_ptr,
    const int32_t* __restrict__ tile_beg,
    const int32_t* __restrict__ tile_len,
    const int32_t* __restrict__ pair_slot,
    __nv_bfloat16* __restrict__ slot_out_bf16) {  // [M * top_k, 2304]
    __shared__ uint32_t s_H[8][292];
    __shared__ uint32_t s_vq8_uv[1024];

    wv_pdl_wait();
    int num_t = *d_num_tiles16;
    if (num_t <= 0) { wv_pdl_trigger(); return; }

    int warp_id   = threadIdx.x >> 5;
    int lane      = threadIdx.x & 31;
    int lane_n    = lane >> 2;
    int c_k       = lane & 3;
    int r0        = lane >> 2;
    int col_n0    = c_k * 2, col_n1 = col_n0 + 1;
    int tile8     = blockIdx.x * 4 + warp_id; // 0..287
    int ld_m      = threadIdx.x >> 4;         // 0..7
    int ld_q0     = threadIdx.x & 15;         // 0..15
    bool vq8_ready = false;

    for (int ti = blockIdx.y; ti < num_t; ti += gridDim.y) {
        int h_idx = tile_hot_idx[ti];
        const uint8_t* slab = reinterpret_cast<const uint8_t*>(tile_slab_ptr[ti]);
        int beg   = tile_beg[ti];
        int len   = min(8, tile_len[ti]);

        if (h_idx < 0 && !vq8_ready) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                int ix = u * 128 + threadIdx.x;
                s_vq8_uv[ix] = __ldg(vq8_uv_gmem + ix);
            }
            vq8_ready = true;
        }

        int ld_sl = (ld_m < len) ? pair_slot[beg + ld_m] : 0;
        const uint4* hp_row = (ld_m < len)
            ? reinterpret_cast<const uint4*>(H_q8 + static_cast<size_t>(ld_sl) * INK2_INTER)
            : nullptr;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int ld_q = u * 16 + ld_q0; // 0..63 (1024B)
            uint4 hv = (hp_row != nullptr) ? hp_row[ld_q] : make_uint4(0, 0, 0, 0);
            int c4   = ld_q * 4;
            s_H[ld_m][c4 + 0] = hv.x; s_H[ld_m][c4 + 1] = hv.y;
            s_H[ld_m][c4 + 2] = hv.z; s_H[ld_m][c4 + 3] = hv.w;
        }
        if (ld_q0 < 8) {
            int ld_q = 64 + ld_q0; // 64..71 (+128B = 1152B)
            uint4 hv = (hp_row != nullptr) ? hp_row[ld_q] : make_uint4(0, 0, 0, 0);
            int c4   = ld_q * 4;
            s_H[ld_m][c4 + 0] = hv.x; s_H[ld_m][c4 + 1] = hv.y;
            s_H[ld_m][c4 + 2] = hv.z; s_H[ld_m][c4 + 3] = hv.w;
        }
        __syncthreads();

        float acc0 = 0.f, acc1 = 0.f;
        const uint32_t* h_pk_b = (h_idx >= 0) ? (hot_dn_pk3 + (static_cast<size_t>(h_idx) * 288 + tile8) * (9 * 96) + lane) : nullptr;
        const uint16_t* h_sc_b = (h_idx >= 0) ? (hot_dn_sc  + (static_cast<size_t>(h_idx) * 288 + tile8) * (9 * 20)) : nullptr;
        const uint32_t* c_pk_b = (h_idx < 0)  ? (reinterpret_cast<const uint32_t*>(slab + INK2_OFF_DN_PK) + static_cast<size_t>(tile8) * (18 * 32) + lane) : nullptr;
        const uint16_t* c_sc_b = (h_idx < 0)  ? (reinterpret_cast<const uint16_t*>(slab + INK2_OFF_DN_SC) + static_cast<size_t>(tile8) * (18 * 8) + lane_n) : nullptr;

        int sl0 = (r0 < len) ? pair_slot[beg + r0] : 0;
        for (int kc128 = 0; kc128 < 9; ++kc128) {
            uint32_t b0[4], b1[4];
            float ws[4];
            if (h_idx >= 0) {
                const uint32_t* pk = h_pk_b + kc128 * 96;
                wv_ink_unpack_vq12_meta(h_sc_b, kc128, lane_n, ws);
                wv_ink_unpack_vq12_sub4_ldg(__ldg(pk), __ldg(pk + 32), __ldg(pk + 64), cb4096_gmem, b0, b1);
            } else {
                wv_ink_unpack_vq8_sb6_meta(c_sc_b, kc128, ws);
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t pkv = __ldg(c_pk_b + (kc128 * 2 + h) * 32);
                    wv_ink_unpack_vq8_h(pkv, lane, s_vq8_uv, b0[2 * h], b1[2 * h], b0[2 * h + 1], b1[2 * h + 1]);
                }
            }
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int kc32  = kc128 * 4 + sub;
                int s_off = kc32 * 8 + c_k;
                float ws0 = __shfl_sync(0xffffffffu, ws[sub], col_n0 * 4);
                float ws1 = __shfl_sync(0xffffffffu, ws[sub], col_n1 * 4);
                float xs0 = (r0 < len) ? H_sc[static_cast<size_t>(sl0) * 36 + kc32] : 0.0f;
                int d0, d1, d2, d3;
                wv_mma_m16n8k32_s8(d0, d1, d2, d3, s_H[r0][s_off], 0u, s_H[r0][s_off + 4], 0u, b0[sub], b1[sub]);
                acc0 += static_cast<float>(d0) * (ws0 * xs0);
                acc1 += static_cast<float>(d1) * (ws1 * xs0);
            }
        }

        if (r0 < len) {
            __nv_bfloat16* dst = slot_out_bf16 + static_cast<size_t>(sl0) * INK2_HIDDEN + tile8 * 8;
            dst[col_n0] = __float2bfloat16(acc0);
            dst[col_n1] = __float2bfloat16(acc1);
        }
        __syncthreads();
    }
    wv_pdl_trigger();
}

// Deterministic k = 0..top_k-1 FP32 weighted sum: Grid <<<M, dim3(32, 6)>>>
__global__ void wv_ink_moe_combine_top8_kernel(
    const __nv_bfloat16* __restrict__ slot_out_bf16, // [M * top_k, 2304]
    const float* __restrict__ top_w,                 // [M, top_k]
    __nv_bfloat16* __restrict__ mlp_out_bf16,        // [M, 2304]
    int top_k, int M) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m >= M) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x, wid = threadIdx.y;
    float w[INK2_TOP_K_MAX];
    #pragma unroll
    for (int k = 0; k < INK2_TOP_K_MAX; ++k) {
        w[k] = (k < top_k) ? top_w[m * top_k + k] : 0.0f;
    }

    __nv_bfloat16* dst = mlp_out_bf16 + static_cast<size_t>(m) * INK2_HIDDEN;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int base = (wid + it * 6) * 128 + lane * 4;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        #pragma unroll
        for (int k = 0; k < INK2_TOP_K_MAX; ++k) {
            if (k < top_k) {
                const __nv_bfloat16* sp = slot_out_bf16 + static_cast<size_t>(m * top_k + k) * INK2_HIDDEN + base;
                float wk = w[k];
                s0 += wk * __bfloat162float(sp[0]);
                s1 += wk * __bfloat162float(sp[1]);
                s2 += wk * __bfloat162float(sp[2]);
                s3 += wk * __bfloat162float(sp[3]);
            }
        }
        dst[base + 0] = __float2bfloat16(s0); dst[base + 1] = __float2bfloat16(s1);
        dst[base + 2] = __float2bfloat16(s2); dst[base + 3] = __float2bfloat16(s3);
    }
    wv_pdl_trigger();
}

__global__ void wv_ink_attn_res_update_kernel(
    const __nv_bfloat16* __restrict__ block_res_bf16,
    const __nv_bfloat16* __restrict__ attn_sconv_bf16,
    __nv_bfloat16* __restrict__ blocks_bf16,
    float* __restrict__ h_mid_f32,
    int n_blk_before, int max_blk, int is_boundary, int M) {
    wv_pdl_wait();
    int idx = blockIdx.x * 256 + threadIdx.x;
    int tot = M * INK2_HIDDEN;
    if (idx >= tot) { wv_pdl_trigger(); return; }
    float a = __bfloat162float(attn_sconv_bf16[idx]);
    if (is_boundary) {
        int m = idx / INK2_HIDDEN;
        int d = idx - m * INK2_HIDDEN;
        __nv_bfloat16 br = block_res_bf16[idx];
        blocks_bf16[(static_cast<size_t>(m) * max_blk + n_blk_before) * INK2_HIDDEN + d] = br;
        h_mid_f32[idx] = a;
    } else {
        float hm = __bfloat162float(block_res_bf16[idx]) + a;
        h_mid_f32[idx] = hm;
    }
    wv_pdl_trigger();
}

__global__ void wv_ink_mlp_res_add_bf16_kernel(
    const float* __restrict__ residual_f32,
    const __nv_bfloat16* __restrict__ mlp_sconv_bf16,
    __nv_bfloat16* __restrict__ h_out_bf16,
    int total_elems) {
    wv_pdl_wait();
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= total_elems) { wv_pdl_trigger(); return; }
    float v = residual_f32[idx] + __bfloat162float(mlp_sconv_bf16[idx]);
    h_out_bf16[idx] = __float2bfloat16(v);
    wv_pdl_trigger();
}

// Fused MoE Top-K Combine + MLP Static ShortConvolution + Residual Add: Grid <<<dim3(9, B_act), 256>>>
__global__ void wv_ink_moe_combine_sconv_res_add_kernel(
    const __nv_bfloat16* __restrict__ slot_out_bf16,
    const float* __restrict__ top_w,
    const float* __restrict__ mlp_sconv_w,
    float* __restrict__ conv_state,
    float* __restrict__ cand_save,
    const int32_t* __restrict__ d_slot_map,
    const float* __restrict__ h_mid_f32,
    __nv_bfloat16* __restrict__ h_out_bf16,
    int top_k, int B, int S, int commit_now) {
    wv_pdl_wait();
    int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int b = blockIdx.y;
    if (c >= INK2_HIDDEN || b >= B || S <= 0) { wv_pdl_trigger(); return; }
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;

    float4 w4 = reinterpret_cast<const float4*>(mlp_sconv_w)[c];
    float* st_b = conv_state + static_cast<size_t>(phys_b) * 3 * INK2_HIDDEN;
    float h0 = st_b[0 * INK2_HIDDEN + c];
    float h1 = st_b[1 * INK2_HIDDEN + c];
    float h2 = st_b[2 * INK2_HIDDEN + c];
    float* cb = (cand_save != nullptr) ? (cand_save + static_cast<size_t>(phys_b) * 8 * INK2_HIDDEN) : nullptr;
    constexpr float kInvSqrt5 = 0.44721359549995793928f;

    for (int s = 0; s < S; ++s) {
        int m = b * S + s;
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            sum += top_w[m * top_k + k] * __bfloat162float(slot_out_bf16[static_cast<size_t>(m * top_k + k) * INK2_HIDDEN + c]);
        }
        float x_cur = __bfloat162float(__float2bfloat16(sum));
        if (cb != nullptr && s < 8) {
            cb[static_cast<size_t>(s) * INK2_HIDDEN + c] = x_cur;
        }
        float conv  = w4.x * h0 + w4.y * h1 + w4.z * h2 + w4.w * x_cur;
        float y_sc  = __bfloat162float(__float2bfloat16((conv + x_cur) * kInvSqrt5));
        float hm    = h_mid_f32[static_cast<size_t>(m) * INK2_HIDDEN + c];
        h_out_bf16[static_cast<size_t>(m) * INK2_HIDDEN + c] = __float2bfloat16(hm + y_sc);
        h0 = h1; h1 = h2; h2 = x_cur;
    }
    if (commit_now) {
        st_b[0 * INK2_HIDDEN + c] = h0;
        st_b[1 * INK2_HIDDEN + c] = h1;
        st_b[2 * INK2_HIDDEN + c] = h2;
    }
    wv_pdl_trigger();
}

struct WvInkColdCandItem {
    double score;
    int    li;
    int    ci;
};

// Loop-aware VRAM Cache + Ping-Pong PCIe DMA Manager (pure C++ / CUDA, zero PyTorch deps)
class WvInkVramCacheDmaManager {
public:
    int      dev_id          = 0;
    int      num_layers      = INK2_NUM_LAYERS_MAX;
    int      num_experts     = INK2_NUM_EXPERTS_MAX;
    int      loop_beg        = 6;
    int      loop_end        = 18;
    bool     all_vram        = true;
    bool     has_cold_dma    = false;
    int      num_cache_slots = 0;
    size_t   cache_bytes     = 0;

    uint8_t*             d_all_vram_slabs[INK2_NUM_LAYERS_MAX] = {};
    WvPinnedBuffer       host_slabs[INK2_NUM_LAYERS_MAX];
    std::vector<uint8_t*> gpu_allocs;

    uint8_t*  d_cache_slabs    = nullptr;
    uint8_t*  d_stage_slabs[2] = {nullptr, nullptr};
    uint64_t* d_resolved_ptrs  = nullptr;
    WvPinnedBuffer h_top_eid_pinned;
    WvPinnedBuffer h_resolved_pinned;

    std::vector<int>     n_cold_per_layer;
    std::vector<int32_t> h_tmap[INK2_NUM_LAYERS_MAX];
    std::vector<float>   cold_base_score[INK2_NUM_LAYERS_MAX];
    std::vector<int32_t> vram_slot_of[INK2_NUM_LAYERS_MAX];
    std::vector<int>     slot_owner_layer;
    std::vector<int>     slot_owner_ci;
    std::vector<double>  slot_priority;
    std::vector<uint64_t> slot_last_step;

    int              staged_layer[2] = {-1, -1};
    std::vector<int> staged_ci[2];
    std::vector<int> last_visit_cold[INK2_NUM_LAYERS_MAX];

    cudaStream_t dma_stream = nullptr;
    cudaEvent_t  dma_ev[2]  = {nullptr, nullptr};
    cudaEvent_t  main_ev    = nullptr;

    uint64_t step_tick         = 0;
    uint64_t dma_hits          = 0;
    uint64_t dma_prefetch_hits = 0;
    uint64_t dma_misses        = 0;
    uint64_t h2d_bytes         = 0;

    ~WvInkVramCacheDmaManager() { release(); }

    int32_t alloc_tracked_u8(size_t nbytes, uint8_t** out_ptr) {
        if (nbytes == 0) {
            *out_ptr = nullptr;
            return 0;
        }
        DeviceGuard guard(dev_id);
        uint8_t* p = nullptr;
        WV_CUDA_CHECK(cudaMalloc(&p, nbytes));
        gpu_allocs.push_back(p);
        *out_ptr = p;
        return 0;
    }

    void release() {
        DeviceGuard guard(dev_id);
        for (uint8_t* p : gpu_allocs) {
            if (p != nullptr) cudaFree(p);
        }
        gpu_allocs.clear();
        for (int l = 0; l < INK2_NUM_LAYERS_MAX; ++l) {
            d_all_vram_slabs[l] = nullptr;
            host_slabs[l].release();
            h_tmap[l].clear();
            cold_base_score[l].clear();
            vram_slot_of[l].clear();
            last_visit_cold[l].clear();
        }
        n_cold_per_layer.clear();
        slot_owner_layer.clear();
        slot_owner_ci.clear();
        slot_priority.clear();
        slot_last_step.clear();
        staged_ci[0].clear();
        staged_ci[1].clear();
        staged_layer[0] = staged_layer[1] = -1;
        d_cache_slabs   = nullptr;
        d_stage_slabs[0] = d_stage_slabs[1] = nullptr;
        d_resolved_ptrs = nullptr;
        for (int b = 0; b < 2; ++b) {
            if (dma_ev[b] != nullptr) { cudaEventDestroy(dma_ev[b]); dma_ev[b] = nullptr; }
        }
        if (main_ev != nullptr) { cudaEventDestroy(main_ev); main_ev = nullptr; }
        if (dma_stream != nullptr) { cudaStreamDestroy(dma_stream); dma_stream = nullptr; }
        h_top_eid_pinned.release();
        h_resolved_pinned.release();
        num_cache_slots = 0;
        cache_bytes     = 0;
        has_cold_dma    = false;
    }

    void reset_stats() {
        step_tick = 0;
        dma_hits = 0;
        dma_prefetch_hits = 0;
        dma_misses = 0;
        h2d_bytes = 0;
    }
};

inline int32_t wv_ink_init_dma_manager(
    WvInkVramCacheDmaManager& mgr,
    int dev_id,
    int n_layers,
    int n_experts,
    int top_k,
    int loop_beg,
    int loop_end,
    bool use_all_vram,
    size_t vram_cache_budget_bytes,
    const std::vector<const uint8_t*>& raw_slab_ptrs,
    const std::vector<int>& n_cold_vec,
    const std::vector<const int32_t*>& tmap_ptrs,
    const std::vector<const float*>& cold_omega_ptrs,
    int max_tokens_step) {
    mgr.release();
    if (n_layers <= 0 || n_layers > INK2_NUM_LAYERS_MAX || n_experts <= 0 || n_experts > INK2_NUM_EXPERTS_MAX) {
        return set_error("Invalid Inkling DMA layer or expert count");
    }
    DeviceGuard guard(dev_id);
    mgr.dev_id           = dev_id;
    mgr.num_layers       = n_layers;
    mgr.num_experts      = n_experts;
    mgr.loop_beg         = loop_beg;
    mgr.loop_end         = loop_end;
    mgr.all_vram         = use_all_vram;
    mgr.n_cold_per_layer = n_cold_vec;

    int total_cold = 0;
    for (int l = 0; l < n_layers; ++l) {
        mgr.h_tmap[l].assign(tmap_ptrs[l], tmap_ptrs[l] + n_experts);
        int nc = n_cold_vec[l];
        total_cold += nc;
        if (nc > 0 && cold_omega_ptrs[l] != nullptr) {
            mgr.cold_base_score[l].assign(cold_omega_ptrs[l], cold_omega_ptrs[l] + nc);
        } else {
            mgr.cold_base_score[l].assign(nc, 1.0f);
        }
        mgr.vram_slot_of[l].assign(nc, -1);
    }

    if (use_all_vram || total_cold == 0) {
        for (int l = 0; l < n_layers; ++l) {
            size_t nb = static_cast<size_t>(n_cold_vec[l]) * INK2_COLD_SLAB_BYTES;
            if (nb > 0 && raw_slab_ptrs[l] != nullptr) {
                if (mgr.alloc_tracked_u8(nb, &mgr.d_all_vram_slabs[l]) != 0) { mgr.release(); return -1; }
                WV_CUDA_CHECK(cudaMemcpy(mgr.d_all_vram_slabs[l], raw_slab_ptrs[l], nb, cudaMemcpyHostToDevice));
            }
        }
        return 0;
    }

    mgr.has_cold_dma = true;
    int lo_p = 0, hi_p = 0;
    cudaDeviceGetStreamPriorityRange(&lo_p, &hi_p);
    WV_CUDA_CHECK(cudaStreamCreateWithPriority(&mgr.dma_stream, cudaStreamNonBlocking, hi_p));
    for (int b = 0; b < 2; ++b) {
        WV_CUDA_CHECK(cudaEventCreateWithFlags(&mgr.dma_ev[b], cudaEventDisableTiming));
        if (mgr.alloc_tracked_u8(static_cast<size_t>(INK2_MAX_STAGE_SLABS) * INK2_COLD_SLAB_BYTES, &mgr.d_stage_slabs[b]) != 0) {
            mgr.release(); return -1;
        }
        mgr.staged_ci[b].clear();
        mgr.staged_layer[b] = -1;
    }
    WV_CUDA_CHECK(cudaEventCreateWithFlags(&mgr.main_ev, cudaEventDisableTiming));
    uint8_t* p_res = nullptr;
    if (mgr.alloc_tracked_u8(128 * sizeof(uint64_t), &p_res) != 0) { mgr.release(); return -1; }
    mgr.d_resolved_ptrs = reinterpret_cast<uint64_t*>(p_res);
    if (!mgr.h_top_eid_pinned.allocate(static_cast<size_t>(max_tokens_step) * top_k * sizeof(int32_t), true) ||
        !mgr.h_resolved_pinned.allocate(128 * sizeof(uint64_t), true)) {
        mgr.release();
        return -1;
    }

    std::vector<WvInkColdCandItem> all_cands;
    all_cands.reserve(total_cold);
    for (int l = 0; l < n_layers; ++l) {
        int nc    = n_cold_vec[l];
        size_t nb = static_cast<size_t>(nc) * INK2_COLD_SLAB_BYTES;
        if (nb > 0) {
            if (!mgr.host_slabs[l].allocate(nb, true)) { mgr.release(); return -1; }
            std::memcpy(mgr.host_slabs[l].ptr, raw_slab_ptrs[l], nb);
        }
        double loop_wt = (l >= loop_beg && l < loop_end) ? 3.0 : 1.0;
        for (int ci = 0; ci < nc; ++ci) {
            all_cands.push_back({static_cast<double>(mgr.cold_base_score[l][ci]) * loop_wt, l, ci});
        }
    }
    std::sort(all_cands.begin(), all_cands.end(), [](const WvInkColdCandItem& a, const WvInkColdCandItem& b) {
        if (a.score != b.score) return a.score > b.score;
        return (a.li != b.li) ? (a.li < b.li) : (a.ci < b.ci);
    });

    int budget_slots = static_cast<int>(vram_cache_budget_bytes / INK2_COLD_SLAB_BYTES);
    int min_floor    = (budget_slots > 0) ? 1 : std::min(64, total_cold);
    int max_slots    = std::min(total_cold, std::max(min_floor, budget_slots));
    mgr.num_cache_slots = max_slots;
    mgr.cache_bytes     = static_cast<size_t>(max_slots) * INK2_COLD_SLAB_BYTES;
    if (mgr.alloc_tracked_u8(mgr.cache_bytes, &mgr.d_cache_slabs) != 0) { mgr.release(); return -1; }

    int n_dyn    = (max_slots >= 6) ? std::min(96, max_slots / 6) : ((max_slots >= 2) ? 1 : 0);
    int n_static = max_slots - n_dyn;
    mgr.slot_owner_layer.assign(max_slots, -1);
    mgr.slot_owner_ci.assign(max_slots, -1);
    mgr.slot_priority.assign(max_slots, 0.0);
    mgr.slot_last_step.assign(max_slots, 0ULL);

    for (int s = 0; s < n_static; ++s) {
        int li = all_cands[s].li, ci = all_cands[s].ci;
        uint8_t* dst = mgr.d_cache_slabs + static_cast<size_t>(s) * INK2_COLD_SLAB_BYTES;
        const uint8_t* src = mgr.host_slabs[li].ptr + static_cast<size_t>(ci) * INK2_COLD_SLAB_BYTES;
        WV_CUDA_CHECK(cudaMemcpy(dst, src, INK2_COLD_SLAB_BYTES, cudaMemcpyHostToDevice));
        mgr.vram_slot_of[li][ci] = s;
        mgr.slot_owner_layer[s]  = li;
        mgr.slot_owner_ci[s]     = ci;
        mgr.slot_priority[s]     = 1e30;
    }
    return 0;
}

// Resolves active cold experts for visit_idx (li), promoting loop-core pass-0 misses into dynamic VRAM ring
// and overlapping visit_idx + 1 prefetch on dma_stream:
inline int32_t wv_ink_resolve_and_prefetch_visit_dma(
    WvInkVramCacheDmaManager& mgr,
    int visit_idx, int li, int next_li,
    const int32_t* d_top_eid, int top_k, int M,
    cudaEvent_t stage_free_ev[2],
    cudaStream_t main_stream) {
    if (!mgr.has_cold_dma) return 0;
    DeviceGuard guard(mgr.dev_id);
    int buf_id = visit_idx & 1;
    size_t n_eids = static_cast<size_t>(M) * top_k;
    WV_CUDA_CHECK(cudaMemcpyAsync(mgr.h_top_eid_pinned.ptr, d_top_eid, n_eids * sizeof(int32_t), cudaMemcpyDeviceToHost, main_stream));
    WV_CUDA_CHECK(cudaStreamSynchronize(main_stream));

    const int32_t* eids = reinterpret_cast<const int32_t*>(mgr.h_top_eid_pinned.ptr);
    uint64_t* res_ptrs  = reinterpret_cast<uint64_t*>(mgr.h_resolved_pinned.ptr);
    std::memset(res_ptrs, 0, 128 * sizeof(uint64_t));

    uint32_t seen_ci[4] = {0u, 0u, 0u, 0u};
    int n_dyn = (mgr.num_cache_slots >= 6)
        ? std::min(96, mgr.num_cache_slots / 6)
        : ((mgr.num_cache_slots >= 2) ? 1 : 0);
    int dyn_base     = mgr.num_cache_slots - n_dyn;
    int stage_used   = (mgr.staged_layer[buf_id] == li) ? static_cast<int>(mgr.staged_ci[buf_id].size()) : 0;
    bool waited_stage = false;
    bool issued_dma   = false;

    for (size_t i = 0; i < n_eids; ++i) {
        int e = eids[i];
        int tm = mgr.h_tmap[li][e];
        if (tm >= 0) continue; // Hot VQ-4096 already in VRAM
        int ci = ~tm;
        if (ci < 0 || ci >= 128) continue;
        uint32_t bit = 1u << (ci & 31);
        if (seen_ci[ci >> 5] & bit) continue;
        seen_ci[ci >> 5] |= bit;

        int slot = mgr.vram_slot_of[li][ci];
        if (slot >= 0) {
            res_ptrs[ci] = reinterpret_cast<uint64_t>(mgr.d_cache_slabs + static_cast<size_t>(slot) * INK2_COLD_SLAB_BYTES);
            mgr.dma_hits++;
            continue;
        }
        int pref_s = -1;
        if (mgr.staged_layer[buf_id] == li) {
            for (size_t k = 0; k < mgr.staged_ci[buf_id].size(); ++k) {
                if (mgr.staged_ci[buf_id][k] == ci) { pref_s = static_cast<int>(k); break; }
            }
        }
        if (pref_s >= 0) {
            if (!issued_dma) {
                WV_CUDA_CHECK(cudaStreamWaitEvent(main_stream, mgr.dma_ev[buf_id], 0));
            }
            res_ptrs[ci] = reinterpret_cast<uint64_t>(mgr.d_stage_slabs[buf_id] + static_cast<size_t>(pref_s) * INK2_COLD_SLAB_BYTES);
            mgr.dma_hits++;
            mgr.dma_prefetch_hits++;
            continue;
        }

        mgr.dma_misses++;
        mgr.h2d_bytes += INK2_COLD_SLAB_BYTES;
        const uint8_t* src = mgr.host_slabs[li].ptr + static_cast<size_t>(ci) * INK2_COLD_SLAB_BYTES;
        uint8_t* dst = nullptr;
        if (li >= mgr.loop_beg && li < mgr.loop_end && n_dyn > 0) {
            int ring_s = dyn_base + static_cast<int>(mgr.step_tick++ % static_cast<uint64_t>(n_dyn));
            int old_l = mgr.slot_owner_layer[ring_s], old_c = mgr.slot_owner_ci[ring_s];
            if (old_l != li) {
                if (old_l >= 0 && old_c >= 0) mgr.vram_slot_of[old_l][old_c] = -1;
                mgr.vram_slot_of[li][ci]     = ring_s;
                mgr.slot_owner_layer[ring_s] = li;
                mgr.slot_owner_ci[ring_s]    = ci;
                dst = mgr.d_cache_slabs + static_cast<size_t>(ring_s) * INK2_COLD_SLAB_BYTES;
            }
        }
        if (dst == nullptr && stage_used < INK2_MAX_STAGE_SLABS) {
            if (!waited_stage) {
                WV_CUDA_CHECK(cudaStreamWaitEvent(mgr.dma_stream, stage_free_ev[buf_id], 0));
                waited_stage = true;
            }
            int st_s = stage_used++;
            dst = mgr.d_stage_slabs[buf_id] + static_cast<size_t>(st_s) * INK2_COLD_SLAB_BYTES;
        }
        if (dst == nullptr) {
            for (int tries = 0; tries < n_dyn; ++tries) {
                int ring_s = dyn_base + static_cast<int>(mgr.step_tick++ % static_cast<uint64_t>(n_dyn));
                int old_l = mgr.slot_owner_layer[ring_s], old_c = mgr.slot_owner_ci[ring_s];
                if (old_l != li) {
                    if (old_l >= 0 && old_c >= 0) mgr.vram_slot_of[old_l][old_c] = -1;
                    mgr.vram_slot_of[li][ci]     = ring_s;
                    mgr.slot_owner_layer[ring_s] = li;
                    mgr.slot_owner_ci[ring_s]    = ci;
                    dst = mgr.d_cache_slabs + static_cast<size_t>(ring_s) * INK2_COLD_SLAB_BYTES;
                    break;
                }
            }
            if (dst == nullptr) {
                return set_error("Inkling DMA stage_used + dynamic ring exhausted");
            }
        }
        WV_CUDA_CHECK(cudaMemcpyAsync(dst, src, INK2_COLD_SLAB_BYTES, cudaMemcpyHostToDevice, mgr.dma_stream));
        res_ptrs[ci] = reinterpret_cast<uint64_t>(dst);
        issued_dma   = true;
    }

    WV_CUDA_CHECK(cudaMemcpyAsync(mgr.d_resolved_ptrs, res_ptrs, 128 * sizeof(uint64_t), cudaMemcpyHostToDevice, mgr.dma_stream));
    WV_CUDA_CHECK(cudaEventRecord(mgr.dma_ev[buf_id], mgr.dma_stream));
    WV_CUDA_CHECK(cudaStreamWaitEvent(main_stream, mgr.dma_ev[buf_id], 0));
    mgr.staged_layer[buf_id] = -1;
    mgr.staged_ci[buf_id].clear();

    if (next_li >= 0 && !mgr.last_visit_cold[next_li].empty()) {
        int next_buf = (visit_idx + 1) & 1;
        WV_CUDA_CHECK(cudaStreamWaitEvent(mgr.dma_stream, stage_free_ev[next_buf], 0));
        mgr.staged_layer[next_buf] = next_li;
        mgr.staged_ci[next_buf].clear();
        for (int ci : mgr.last_visit_cold[next_li]) {
            if (mgr.vram_slot_of[next_li][ci] < 0 && static_cast<int>(mgr.staged_ci[next_buf].size()) < 2) {
                int s_idx = static_cast<int>(mgr.staged_ci[next_buf].size());
                uint8_t* d_p = mgr.d_stage_slabs[next_buf] + static_cast<size_t>(s_idx) * INK2_COLD_SLAB_BYTES;
                const uint8_t* s_p = mgr.host_slabs[next_li].ptr + static_cast<size_t>(ci) * INK2_COLD_SLAB_BYTES;
                WV_CUDA_CHECK(cudaMemcpyAsync(d_p, s_p, INK2_COLD_SLAB_BYTES, cudaMemcpyHostToDevice, mgr.dma_stream));
                mgr.staged_ci[next_buf].push_back(ci);
                mgr.h2d_bytes += INK2_COLD_SLAB_BYTES;
            }
        }
        WV_CUDA_CHECK(cudaEventRecord(mgr.dma_ev[next_buf], mgr.dma_stream));
    }
    mgr.last_visit_cold[li].clear();
    for (int ci = 0; ci < mgr.n_cold_per_layer[li]; ++ci) {
        if ((seen_ci[ci >> 5] & (1u << (ci & 31))) != 0u) {
            mgr.last_visit_cold[li].push_back(ci);
        }
    }
    return 0;
}

} // namespace wv

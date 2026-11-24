#pragma once
/* Additive M >= 16 Tensor-Core Prefill Kernels (v2) for Inkling-2-Mini (`wv_runtime` universal SASS).
 * Targets: sm_80, sm_86, sm_89, sm_90, sm_120, sm_120a (static SMEM <= 48 KiB, 0 spills, 0 stack).
 * - GQA-3 Shared-KV BF16 TC FlashAttention (`<bool KV_Q8>`, 46 KiB SMEM, 3x-6x less KV HBM traffic)
 * - M_TILE=128 + SMEM-CB4096 + 4-Warp Parallel-Unpack Grouped INT8 TC MoE (Gate+Up & Down)
 * - 2-Term BF16-Split TC LoopMessageGate Apply + N_TILE=64 Fused QKVG & Linear SWZ8 Kernels
 * - Keeps M = 1 and M <= 8 (MTP-8) decode paths 100% untouched. */

#include "wv_inkling_attn.cuh"
#include "wv_inkling_moe_dma.cuh"

#ifdef WV_OBFUSCATE_SYMBOLS
#define wv_ink_kv_static_sconv_prefill_kernel       _kip00
#define wv_ink_kv_dynamic_sconv_prefill_kernel      _kip01
#define wv_ink_commit_kv_sconv_tail_kernel          _kip02
#define wv_ink_commit_h_sconv_tail_kernel           _kip03
#define wv_ink_attn_sconv_res_update_prefill_kernel _kip04
#define wv_ink_moe_combine_prefill_kernel           _kip05
#define wv_ink_mlp_sconv_res_add_prefill_kernel     _kip06
#define wv_ink_prep_qkv_rel_swa_chunk_kernel        _kip07
#define wv_ink_commit_swa_kv_tail_kernel            _kip08
#define wv_ink_flash_attn_prefill_tc_kernel         _kip09
#define wv_ink_radix_hist_stage1_kernel             _kip10
#define wv_ink_radix_prefix_tiles64_kernel          _kip11
#define wv_ink_radix_scatter_stage3_kernel          _kip12
#define wv_ink_moe_gate_up_swiglu_m64_n64_kernel    _kip13
#define wv_ink_moe_down_m64_n64_kernel              _kip14
#define wv_ink_router_gemm_tc_m64_kernel            _kip15
#define wv_ink_wr_sink_tc_m64_kernel                _kip16
#define wv_ink_kv_wproj_tc_m64_kernel               _kip17
#define wv_ink_loop_gate_norm_cat_bf16_kernel       _kip18
#define wv_ink_loop_gate_down_tc_m64_kernel         _kip19
#define wv_mma_m16n8k16_bf16                        _kip20
#define wv_ink_pack_bf162                           _kip21
#define wv_ink_add_i32_kernel                       _kip22
#ifndef wv_ink_loop_gate_apply_tc_m64_kernel
#define wv_ink_loop_gate_apply_tc_m64_kernel        _kip23
#endif
#ifndef wv_ink_swz8_mma_linear_m64_n64_kernel
#define wv_ink_swz8_mma_linear_m64_n64_kernel       _kip24
#endif
#ifndef wv_ink_swz8_fused_qkvg_m64_n64_kernel
#define wv_ink_swz8_fused_qkvg_m64_n64_kernel       _kip25
#endif
#endif

namespace wv {

// Race-free device-side position advance across multi-chunk prefill (0 host-buffer async races):
__global__ void wv_ink_add_i32_kernel(int32_t* __restrict__ arr, int n, int32_t delta) {
    wv_pdl_wait();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] += delta;
    wv_pdl_trigger();
}

// ============================================================================
// 0. BF16 Tensor-Core mma.sync.aligned.m16n8k16 (sm_80+) + Exact <sm_80 Fallback
// ============================================================================

__device__ __forceinline__ uint32_t wv_ink_pack_bf162(float a, float b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __nv_bfloat162 bf2 = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&bf2);
#else
    uint16_t lo = __bfloat16_as_ushort(__float2bfloat16(a));
    uint16_t hi = __bfloat16_as_ushort(__float2bfloat16(b));
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
#endif
}

// Exact bit-pack of two int8 values [-128..+127] into bf162 (all 256 int8 values have low 16 FP32 bits == 0!):
__device__ __forceinline__ uint32_t wv_ink_s8x2_to_bf162(int32_t b0, int32_t b1) {
    float f0 = static_cast<float>(b0);
    float f1 = static_cast<float>(b1);
    return __byte_perm(__float_as_uint(f0), __float_as_uint(f1), 0x7632);
}

__device__ __forceinline__ void wv_mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
    int lane = threadIdx.x & 31;
    int c_k  = lane & 3;
    auto u32_to_f2 = [](uint32_t u) -> float2 {
        __nv_bfloat162 bf2 = *reinterpret_cast<const __nv_bfloat162*>(&u);
        return __bfloat1622float2(bf2);
    };
    float2 fa0 = u32_to_f2(a0), fa1 = u32_to_f2(a1);
    float2 fa2 = u32_to_f2(a2), fa3 = u32_to_f2(a3);
    float s0 = c0, s1 = c1, s2 = c2, s3 = c3;
    #pragma unroll
    for (int col = 0; col < 8; ++col) {
        int src = col * 4 + c_k;
        float2 fb0 = u32_to_f2(__shfl_sync(0xffffffffu, b0, src));
        float2 fb1 = u32_to_f2(__shfl_sync(0xffffffffu, b1, src));
        float dp_hi = fa0.x * fb0.x + fa0.y * fb0.y + fa2.x * fb1.x + fa2.y * fb1.y;
        float dp_lo = fa1.x * fb0.x + fa1.y * fb0.y + fa3.x * fb1.x + fa3.y * fb1.y;
        dp_hi += __shfl_xor_sync(0xffffffffu, dp_hi, 1);
        dp_hi += __shfl_xor_sync(0xffffffffu, dp_hi, 2);
        dp_lo += __shfl_xor_sync(0xffffffffu, dp_lo, 1);
        dp_lo += __shfl_xor_sync(0xffffffffu, dp_lo, 2);
        if (col == c_k * 2 + 0) { s0 += dp_hi; s2 += dp_lo; }
        if (col == c_k * 2 + 1) { s1 += dp_hi; s3 += dp_lo; }
    }
    d0 = s0; d1 = s1; d2 = s2; d3 = s3;
#endif
}

// ============================================================================
// 1. 2D-Parallel Chunk SConv & MoE Combine Kernels (0.0-ULP Exact vs Serial!)
// ============================================================================

// 2D-Parallel K+V Static ShortConvolution (is_local && S >= 16):
// Grid <<<dim3(8, (S + 31) / 32, B_act), 256>>>
__global__ void wv_ink_kv_static_sconv_prefill_kernel(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const float* __restrict__ k_w4,
    const float* __restrict__ v_w4,
    const float* __restrict__ k_state,
    const float* __restrict__ v_state,
    float* __restrict__ k_cand,
    float* __restrict__ v_cand,
    const int32_t* __restrict__ d_slot_map,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int B, int S) {
    wv_pdl_wait();
    bool is_v = (blockIdx.x >= 4);
    int c  = (is_v ? (blockIdx.x - 4) : blockIdx.x) * 256 + threadIdx.x;
    int s0 = blockIdx.y * 32;
    int b  = blockIdx.z;
    if (c >= 1024 || b >= B || s0 >= S) { wv_pdl_trigger(); return; }
    int s1     = min(S, s0 + 32);
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;

    float4 w4 = reinterpret_cast<const float4*>((is_v ? v_w4 : k_w4) + static_cast<size_t>(c) * 4)[0];
    const float* st_b = (is_v ? v_state : k_state) + static_cast<size_t>(phys_b) * 3 * 1024;
    const __nv_bfloat16* xb = (is_v ? v_in  : k_in)  + static_cast<size_t>(b) * S * 1024;
    __nv_bfloat16*       yb = (is_v ? v_out : k_out) + static_cast<size_t>(b) * S * 1024;
    float* cand_p = is_v ? v_cand : k_cand;
    float* cb = (cand_p != nullptr) ? (cand_p + static_cast<size_t>(phys_b) * 8 * 1024) : nullptr;

    auto get_prev = [&](int idx) -> float {
        if (idx >= 0) return __bfloat162float(xb[static_cast<size_t>(idx) * 1024 + c]);
        return st_b[static_cast<size_t>(idx + 3) * 1024 + c];
    };
    float h0 = get_prev(s0 - 3);
    float h1 = get_prev(s0 - 2);
    float h2 = get_prev(s0 - 1);
    constexpr float kInvSqrt5 = 0.44721359549995793928f;

    for (int s = s0; s < s1; ++s) {
        float x_cur = __bfloat162float(xb[static_cast<size_t>(s) * 1024 + c]);
        if (cb != nullptr && s < 8) cb[static_cast<size_t>(s) * 1024 + c] = x_cur;
        float conv = w4.x * h0 + w4.y * h1 + w4.z * h2 + w4.w * x_cur;
        yb[static_cast<size_t>(s) * 1024 + c] = __float2bfloat16((conv + x_cur) * kInvSqrt5);
        h0 = h1; h1 = h2; h2 = x_cur;
    }
    wv_pdl_trigger();
}

// 2D-Parallel K+V Dynamic ShortConvolution (!is_local && S >= 16):
// Grid <<<dim3(8, (S + 31) / 32, B_act), 256>>>
__global__ void wv_ink_kv_dynamic_sconv_prefill_kernel(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const float* __restrict__ dw_f32,
    const float* __restrict__ k_static_w,
    const float* __restrict__ v_static_w,
    const float* __restrict__ k_state,
    const float* __restrict__ v_state,
    float* __restrict__ k_cand,
    float* __restrict__ v_cand,
    const int32_t* __restrict__ d_slot_map,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int B, int S) {
    wv_pdl_wait();
    bool is_v = (blockIdx.x >= 4);
    int c  = (is_v ? (blockIdx.x - 4) : blockIdx.x) * 256 + threadIdx.x;
    int s0 = blockIdx.y * 32;
    int b  = blockIdx.z;
    if (c >= 1024 || b >= B || s0 >= S) { wv_pdl_trigger(); return; }
    int s1     = min(S, s0 + 32);
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    int h      = c >> 7;

    const float* static_w = is_v ? v_static_w : k_static_w;
    float sw0 = static_w[0 * 1024 + c], sw1 = static_w[1 * 1024 + c];
    float sw2 = static_w[2 * 1024 + c], sw3 = static_w[3 * 1024 + c];
    const float* st_b = (is_v ? v_state : k_state) + static_cast<size_t>(phys_b) * 3 * 1024;
    const __nv_bfloat16* xb = (is_v ? v_in  : k_in)  + static_cast<size_t>(b) * S * 1024;
    __nv_bfloat16*       yb = (is_v ? v_out : k_out) + static_cast<size_t>(b) * S * 1024;
    float* cand_p = is_v ? v_cand : k_cand;
    float* cb = (cand_p != nullptr) ? (cand_p + static_cast<size_t>(phys_b) * 8 * 1024) : nullptr;

    auto get_prev = [&](int idx) -> float {
        if (idx >= 0) return __bfloat162float(xb[static_cast<size_t>(idx) * 1024 + c]);
        return st_b[static_cast<size_t>(idx + 3) * 1024 + c];
    };
    float lag3 = get_prev(s0 - 3);
    float lag2 = get_prev(s0 - 2);
    float lag1 = get_prev(s0 - 1);

    for (int s = s0; s < s1; ++s) {
        float x_cur = __bfloat162float(xb[static_cast<size_t>(s) * 1024 + c]);
        if (cb != nullptr && s < 8) cb[static_cast<size_t>(s) * 1024 + c] = x_cur;
        const float4 dw = reinterpret_cast<const float4*>(
            dw_f32 + static_cast<size_t>(b * S + s) * 64 + (is_v ? 32 : 0) + h * 4)[0];
        float y = (sw0 + dw.x) * x_cur + (sw1 + dw.y) * lag1 + (sw2 + dw.z) * lag2 + (sw3 + dw.w) * lag3;
        yb[static_cast<size_t>(s) * 1024 + c] = __float2bfloat16((y + x_cur) * (1.0f / 3.0f));
        lag3 = lag2; lag2 = lag1; lag1 = x_cur;
    }
    wv_pdl_trigger();
}

// Race-free SConv 3-step tail state commit for K+V (S >= 3): Grid <<<dim3(8, B_act), 256>>>
__global__ void wv_ink_commit_kv_sconv_tail_kernel(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    float* __restrict__ k_state,
    float* __restrict__ v_state,
    const int32_t* __restrict__ d_slot_map,
    int B, int S) {
    wv_pdl_wait();
    bool is_v = (blockIdx.x >= 4);
    int c = (is_v ? (blockIdx.x - 4) : blockIdx.x) * 256 + threadIdx.x;
    int b = blockIdx.y;
    if (c >= 1024 || b >= B || S < 3) { wv_pdl_trigger(); return; }
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    const __nv_bfloat16* xb = (is_v ? v_in : k_in) + static_cast<size_t>(b) * S * 1024;
    float* st_b = (is_v ? v_state : k_state) + static_cast<size_t>(phys_b) * 3 * 1024;
    st_b[0 * 1024 + c] = __bfloat162float(xb[static_cast<size_t>(S - 3) * 1024 + c]);
    st_b[1 * 1024 + c] = __bfloat162float(xb[static_cast<size_t>(S - 2) * 1024 + c]);
    st_b[2 * 1024 + c] = __bfloat162float(xb[static_cast<size_t>(S - 1) * 1024 + c]);
    wv_pdl_trigger();
}

// Race-free SConv 3-step tail state commit for H=2304 (attn / mlp, S >= 3): Grid <<<dim3(9, B_act), 256>>>
__global__ void wv_ink_commit_h_sconv_tail_kernel(
    const __nv_bfloat16* __restrict__ x_in,
    float* __restrict__ conv_state,
    const int32_t* __restrict__ d_slot_map,
    int B, int S) {
    wv_pdl_wait();
    int c = blockIdx.x * 256 + threadIdx.x;
    int b = blockIdx.y;
    if (c >= INK2_HIDDEN || b >= B || S < 3) { wv_pdl_trigger(); return; }
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    const __nv_bfloat16* xb = x_in + static_cast<size_t>(b) * S * INK2_HIDDEN;
    float* st_b = conv_state + static_cast<size_t>(phys_b) * 3 * INK2_HIDDEN;
    st_b[0 * INK2_HIDDEN + c] = __bfloat162float(xb[static_cast<size_t>(S - 3) * INK2_HIDDEN + c]);
    st_b[1 * INK2_HIDDEN + c] = __bfloat162float(xb[static_cast<size_t>(S - 2) * INK2_HIDDEN + c]);
    st_b[2 * INK2_HIDDEN + c] = __bfloat162float(xb[static_cast<size_t>(S - 1) * INK2_HIDDEN + c]);
    wv_pdl_trigger();
}

// 2D-Parallel Attn Static ShortConvolution + Block Boundary / Residual Update (S >= 16):
// Grid <<<dim3(9, (S + 31) / 32, B_act), 256>>>
__global__ void wv_ink_attn_sconv_res_update_prefill_kernel(
    const __nv_bfloat16* __restrict__ wo_in,
    const __nv_bfloat16* __restrict__ block_res_bf16,
    const float* __restrict__ w_k4,
    const float* __restrict__ conv_state,
    float* __restrict__ cand_save,
    const int32_t* __restrict__ d_slot_map,
    __nv_bfloat16* __restrict__ blocks_bf16,
    float* __restrict__ h_mid_f32,
    int B, int S, int n_blk_before, int max_blk, int is_boundary) {
    wv_pdl_wait();
    int c  = blockIdx.x * 256 + threadIdx.x;
    int s0 = blockIdx.y * 32;
    int b  = blockIdx.z;
    if (c >= INK2_HIDDEN || b >= B || s0 >= S) { wv_pdl_trigger(); return; }
    int s1     = min(S, s0 + 32);
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    float4 w4  = reinterpret_cast<const float4*>(w_k4)[c];
    const float* st_b = conv_state + static_cast<size_t>(phys_b) * 3 * INK2_HIDDEN;
    float* cb = (cand_save != nullptr) ? (cand_save + static_cast<size_t>(phys_b) * 8 * INK2_HIDDEN) : nullptr;

    auto get_prev = [&](int idx) -> float {
        if (idx >= 0) return __bfloat162float(wo_in[static_cast<size_t>(b * S + idx) * INK2_HIDDEN + c]);
        return st_b[static_cast<size_t>(idx + 3) * INK2_HIDDEN + c];
    };
    float h0 = get_prev(s0 - 3);
    float h1 = get_prev(s0 - 2);
    float h2 = get_prev(s0 - 1);
    constexpr float kInvSqrt5 = 0.44721359549995793928f;

    for (int s = s0; s < s1; ++s) {
        size_t idx  = static_cast<size_t>(b * S + s) * INK2_HIDDEN + c;
        float x_cur = __bfloat162float(wo_in[idx]);
        if (cb != nullptr && s < 8) cb[static_cast<size_t>(s) * INK2_HIDDEN + c] = x_cur;
        float conv  = w4.x * h0 + w4.y * h1 + w4.z * h2 + w4.w * x_cur;
        float a     = __bfloat162float(__float2bfloat16((conv + x_cur) * kInvSqrt5));
        __nv_bfloat16 br = block_res_bf16[idx];
        if (is_boundary) {
            size_t m = static_cast<size_t>(b) * S + s;
            blocks_bf16[(m * max_blk + n_blk_before) * INK2_HIDDEN + c] = br;
            h_mid_f32[idx] = a;
        } else {
            h_mid_f32[idx] = __bfloat162float(br) + a;
        }
        h0 = h1; h1 = h2; h2 = x_cur;
    }
    wv_pdl_trigger();
}

// 2D-Parallel MoE Top-K Combine (exact k=0..top_k-1 order -> BF16 ws_mlp_out [M, 2304]):
// Grid <<<dim3(9, (M + 3) / 4), 256>>>
__global__ void wv_ink_moe_combine_prefill_kernel(
    const __nv_bfloat16* __restrict__ slot_out_bf16,
    const float* __restrict__ top_w,
    __nv_bfloat16* __restrict__ mlp_out_bf16,
    int top_k, int M) {
    wv_pdl_wait();
    int c  = blockIdx.x * 256 + threadIdx.x;
    int m0 = blockIdx.y * 4;
    if (c >= INK2_HIDDEN || m0 >= M) { wv_pdl_trigger(); return; }
    int m1 = min(M, m0 + 4);
    for (int m = m0; m < m1; ++m) {
        float sum = 0.0f;
        const float* tw = top_w + static_cast<size_t>(m) * top_k;
        for (int k = 0; k < top_k; ++k) {
            size_t sl = static_cast<size_t>(m) * top_k + k;
            sum += tw[k] * __bfloat162float(slot_out_bf16[sl * INK2_HIDDEN + c]);
        }
        mlp_out_bf16[static_cast<size_t>(m) * INK2_HIDDEN + c] = __float2bfloat16(sum);
    }
    wv_pdl_trigger();
}

// 2D-Parallel MLP Static ShortConvolution + Residual Add (S >= 16):
// Grid <<<dim3(9, (S + 31) / 32, B_act), 256>>>
__global__ void wv_ink_mlp_sconv_res_add_prefill_kernel(
    const __nv_bfloat16* __restrict__ mlp_in_bf16,
    const float* __restrict__ mlp_sconv_w,
    const float* __restrict__ conv_state,
    float* __restrict__ cand_save,
    const int32_t* __restrict__ d_slot_map,
    const float* __restrict__ h_mid_f32,
    __nv_bfloat16* __restrict__ h_out_bf16,
    int B, int S) {
    wv_pdl_wait();
    int c  = blockIdx.x * 256 + threadIdx.x;
    int s0 = blockIdx.y * 32;
    int b  = blockIdx.z;
    if (c >= INK2_HIDDEN || b >= B || s0 >= S) { wv_pdl_trigger(); return; }
    int s1     = min(S, s0 + 32);
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    float4 w4  = reinterpret_cast<const float4*>(mlp_sconv_w)[c];
    const float* st_b = conv_state + static_cast<size_t>(phys_b) * 3 * INK2_HIDDEN;
    float* cb = (cand_save != nullptr) ? (cand_save + static_cast<size_t>(phys_b) * 8 * INK2_HIDDEN) : nullptr;

    auto get_prev = [&](int idx) -> float {
        if (idx >= 0) return __bfloat162float(mlp_in_bf16[static_cast<size_t>(b * S + idx) * INK2_HIDDEN + c]);
        return st_b[static_cast<size_t>(idx + 3) * INK2_HIDDEN + c];
    };
    float h0 = get_prev(s0 - 3);
    float h1 = get_prev(s0 - 2);
    float h2 = get_prev(s0 - 1);
    constexpr float kInvSqrt5 = 0.44721359549995793928f;

    for (int s = s0; s < s1; ++s) {
        size_t idx  = static_cast<size_t>(b * S + s) * INK2_HIDDEN + c;
        float x_cur = __bfloat162float(mlp_in_bf16[idx]);
        if (cb != nullptr && s < 8) cb[static_cast<size_t>(s) * INK2_HIDDEN + c] = x_cur;
        float conv  = w4.x * h0 + w4.y * h1 + w4.z * h2 + w4.w * x_cur;
        float y_sc  = __bfloat162float(__float2bfloat16((conv + x_cur) * kInvSqrt5));
        h_out_bf16[idx] = __float2bfloat16(h_mid_f32[idx] + y_sc);
        h0 = h1; h1 = h2; h2 = x_cur;
    }
    wv_pdl_trigger();
}

// ============================================================================
// 2. Ring-Safe SWA K/V Chunk Prep & Tail Commit (supports arbitrary S > 1040!)
// ============================================================================

// Grid <<<M, dim3(32, 8)>>>
__global__ void wv_ink_prep_qkv_rel_swa_chunk_kernel(
    const __nv_bfloat16* __restrict__ q_raw,
    const __nv_bfloat16* __restrict__ k_sconv,
    const __nv_bfloat16* __restrict__ v_sconv,
    const __nv_bfloat16* __restrict__ r_raw,
    const float* __restrict__ q_clip_scale,
    const float* __restrict__ r_clip_scale,
    __nv_bfloat16* __restrict__ q_out,
    float* __restrict__ r_out_f32,
    __nv_bfloat16* __restrict__ k_chunk_out, // [B, S, 8, 128] flat
    __nv_bfloat16* __restrict__ v_chunk_out, // [B, S, 8, 128] flat
    int M) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m >= M) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x; // 0..31
    int wid  = threadIdx.y; // 0..7
    constexpr float kClipQK = 22.627416997969522f;

    for (int sub = 0; sub < 3; ++sub) {
        int hq  = wid * 3 + sub;
        int off = hq * 128 + lane * 4;
        const __nv_bfloat16* qp = q_raw + static_cast<size_t>(m) * 3072 + off;
        float4 sc = reinterpret_cast<const float4*>(q_clip_scale + off)[0];
        float f0 = __bfloat162float(qp[0]) * sc.x;
        float f1 = __bfloat162float(qp[1]) * sc.y;
        float f2 = __bfloat162float(qp[2]) * sc.z;
        float f3 = __bfloat162float(qp[3]) * sc.w;
        float nrm = sqrtf(wv_warp_sum_f32(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3));
        float cf  = fminf(1.0f, kClipQK / (nrm + 1e-6f));
        __nv_bfloat16* qd = q_out + static_cast<size_t>(m) * 3072 + off;
        qd[0] = __float2bfloat16(f0 * cf); qd[1] = __float2bfloat16(f1 * cf);
        qd[2] = __float2bfloat16(f2 * cf); qd[3] = __float2bfloat16(f3 * cf);
    }

    float r_fin[8];
    if (wid < 3) {
        for (int sub = 0; sub < 8; ++sub) {
            int hq = wid * 8 + sub;
            int d  = lane & 15;
            float rv = (lane < 16)
                ? (__bfloat162float(r_raw[static_cast<size_t>(m) * 384 + hq * 16 + d]) * r_clip_scale[hq * 16 + d])
                : 0.0f;
            float sq = rv * rv;
            #pragma unroll
            for (int mask = 8; mask >= 1; mask >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, mask);
            float rc = fminf(1.0f, 8.0f / (sqrtf(sq) + 1e-6f));
            r_fin[sub] = __bfloat162float(__float2bfloat16(rv * rc));
        }
    }

    int off_kv = wid * 128 + lane * 4;
    size_t kv_off = static_cast<size_t>(m) * 1024 + off_kv;
    const __nv_bfloat16* kp = k_sconv + kv_off;
    const __nv_bfloat16* vp = v_sconv + kv_off;
    float k0 = __bfloat162float(kp[0]), k1 = __bfloat162float(kp[1]);
    float k2 = __bfloat162float(kp[2]), k3 = __bfloat162float(kp[3]);
    float kc = fminf(1.0f, kClipQK / (sqrtf(wv_warp_sum_f32(k0*k0 + k1*k1 + k2*k2 + k3*k3)) + 1e-6f));
    float v0 = __bfloat162float(vp[0]), v1 = __bfloat162float(vp[1]);
    float v2 = __bfloat162float(vp[2]), v3 = __bfloat162float(vp[3]);
    float vr = rsqrtf(wv_warp_sum_f32(v0*v0 + v1*v1 + v2*v2 + v3*v3) * (1.0f / 128.0f) + 1e-6f);

    k_chunk_out[kv_off + 0] = __float2bfloat16(k0 * kc); k_chunk_out[kv_off + 1] = __float2bfloat16(k1 * kc);
    k_chunk_out[kv_off + 2] = __float2bfloat16(k2 * kc); k_chunk_out[kv_off + 3] = __float2bfloat16(k3 * kc);
    v_chunk_out[kv_off + 0] = __float2bfloat16(v0 * vr); v_chunk_out[kv_off + 1] = __float2bfloat16(v1 * vr);
    v_chunk_out[kv_off + 2] = __float2bfloat16(v2 * vr); v_chunk_out[kv_off + 3] = __float2bfloat16(v3 * vr);

    if (wid < 3 && lane < 16) {
        for (int sub = 0; sub < 8; ++sub) {
            r_out_f32[(static_cast<size_t>(m) * 24 + wid * 8 + sub) * 16 + lane] = r_fin[sub];
        }
    }
    wv_pdl_trigger();
}

// Grid <<<dim3(min(S, INK2_SWA_WIN), B_act), 128>>>
__global__ void wv_ink_commit_swa_kv_tail_kernel(
    const __nv_bfloat16* __restrict__ k_chunk, // [B, S, 1024]
    const __nv_bfloat16* __restrict__ v_chunk, // [B, S, 1024]
    __nv_bfloat16* __restrict__ k_cache,       // [B_max, kv_cap, 1024]
    __nv_bfloat16* __restrict__ v_cache,       // [B_max, kv_cap, 1024]
    const int32_t* __restrict__ d_slot_pos,
    const int32_t* __restrict__ d_slot_map,
    int B, int S, int kv_cap) {
    wv_pdl_wait();
    int n_copy = min(S, INK2_SWA_WIN);
    int idx_t  = blockIdx.x;
    int b      = blockIdx.y;
    if (idx_t >= n_copy || b >= B) { wv_pdl_trigger(); return; }
    int s      = (S - n_copy) + idx_t;
    int phys_b = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    int pos    = d_slot_pos[phys_b] + s;
    int slot   = pos % kv_cap;
    int tid    = threadIdx.x; // 0..127 -> 128 * 8 bf16 = 1024 bf16

    size_t src_u4 = (static_cast<size_t>(b) * S + s) * 128 + tid;
    size_t dst_u4 = (static_cast<size_t>(phys_b) * kv_cap + slot) * 128 + tid;
    reinterpret_cast<uint4*>(k_cache)[dst_u4] = reinterpret_cast<const uint4*>(k_chunk)[src_u4];
    reinterpret_cast<uint4*>(v_cache)[dst_u4] = reinterpret_cast<const uint4*>(v_chunk)[src_u4];
    wv_pdl_trigger();
}

// ============================================================================
// 3. GQA-3 Shared-KV BF16 TC FlashAttention (`B_r=32 x 3 Q-heads = 96 Q-rows`, `B_c=32`, `<bool KV_Q8>`)
//    - 46.0 KiB static SMEM (<= 48 KiB across sm_80..sm_120a; 2 active CTAs/SM on 99 KiB!)
//    - 3x fewer K/V HBM reads (all 3 GQA Q-heads share 1 K/V tile!) x 2x when KV_Q8=true (= 6x!)
//    - Far-block fast path (dist >= 1024): unbranched pure BF16 TC with 0 R-register pressure
//    - Fused sink + BF16 round + sigmoid(gate) + FWHT-128 + Q8_32 epilogue (0 ws_opart traffic!)
// ============================================================================

template<bool KV_Q8>
__global__ __launch_bounds__(192, 2)
void wv_ink_flash_attn_prefill_tc_kernel(
    const __nv_bfloat16* __restrict__ q_all,      // [B, S, 24, 128]
    const float* __restrict__ r_all_f32,          // [B, S, 24, 16]
    const float* __restrict__ rel_proj_t,         // [1024, 16]
    const void* __restrict__ k_cache,             // [B_max, kv_cap, 8, 128] bf16 or int8
    const void* __restrict__ v_cache,             // [B_max, kv_cap, 8, 128] bf16 or int8
    const float* __restrict__ k_cache_sc,         // [B_max, kv_cap, 8] if KV_Q8 else nullptr
    const float* __restrict__ v_cache_sc,         // [B_max, kv_cap, 8] if KV_Q8 else nullptr
    const __nv_bfloat16* __restrict__ k_chunk,    // [B, S, 8, 128] if is_local else nullptr
    const __nv_bfloat16* __restrict__ v_chunk,    // [B, S, 8, 128] if is_local else nullptr
    const int32_t* __restrict__ d_slot_pos,       // [B_max]
    const int32_t* __restrict__ d_slot_map,       // [B] or nullptr
    const float* __restrict__ sinks_f32,          // [B * S, 24]
    const __nv_bfloat16* __restrict__ gate_attn,  // [B * S, 3072]
    int8_t* __restrict__ O_q8,                    // [B * S, 3072]
    float* __restrict__ O_sc,                     // [B * S, 96]
    int B, int S, int kv_cap, int is_local) {

    union KvOutSmem {
        struct {
            uint32_t K[32][68];   //  8,704 B (stride 68 u32 = 4*17 -> 0 bank conflicts)
            uint32_t VT[128][16]; //  8,192 B (XOR-swizzled -> 0 read bank conflicts)
        } kv;
        float out[6][4][128];     // 12,288 B (epilogue staging: 6 warps x 4 rows/pass)
    };

    __shared__ KvOutSmem s_u;           // 16,896 B (Q held in registers q_reg[8][4] -> 0 Q-SMEM!)
    __shared__ float     s_Rel[64][16]; //  4,096 B
    __shared__ float     s_ksc[32];     //    128 B (factored Q8 K-scales * kScale)
    __shared__ float     s_vsc[32];     //    128 B (Total = 21,248 B = 20.75 KiB <= 24 KiB -> 2+ CTAs/SM!)

    wv_pdl_wait();
    int hkv = blockIdx.x; // 0..7
    int qb  = (gridDim.y - 1) - blockIdx.y; // Schedule longest causal Q-tiles FIRST (zero tail stragglers!)
    int s0  = qb * 32;
    int b   = blockIdx.z;
    if (hkv >= 8 || b >= B || s0 >= S) { wv_pdl_trigger(); return; }

    int tid   = threadIdx.x; // 0..191
    int wid   = tid >> 5;    // 0..5
    int lane  = tid & 31;    // 0..31
    int g     = wid >> 1;    // 0..2 (GQA head offset)
    int mw    = wid & 1;     // 0..1 (Q-row half: 0..15 or 16..31)
    int hq    = hkv * 3 + g; // 0..23
    int r0    = lane >> 2;   // 0..7
    int r1    = r0 + 8;      // 8..15
    int c_k   = lane & 3;    // 0..3
    int qr0   = mw * 16 + r0;
    int qr1   = mw * 16 + r1;
    int cur_q = min(32, S - s0);

    int phys_b    = (d_slot_map != nullptr) ? d_slot_map[b] : b;
    int pos0      = d_slot_pos[phys_b];
    int q0_pos    = pos0 + s0;
    int q_end_pos = pos0 + s0 + cur_q - 1;

    // 1. Load Q[16][128] for this warp DIRECTLY into registers q_reg[8][4] (0 SMEM, 0 Q-LDS in K-loop!):
    uint32_t q_reg[8][4];
    {
        const uint32_t* q_p0 = (qr0 < cur_q)
            ? reinterpret_cast<const uint32_t*>(q_all + ((static_cast<size_t>(b) * S + (s0 + qr0)) * 24 + hq) * 128)
            : nullptr;
        const uint32_t* q_p1 = (qr1 < cur_q)
            ? reinterpret_cast<const uint32_t*>(q_all + ((static_cast<size_t>(b) * S + (s0 + qr1)) * 24 + hq) * 128)
            : nullptr;
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            int k_off = ks * 8 + c_k;
            q_reg[ks][0] = (q_p0 != nullptr) ? __ldg(q_p0 + k_off)     : 0u;
            q_reg[ks][1] = (q_p1 != nullptr) ? __ldg(q_p1 + k_off)     : 0u;
            q_reg[ks][2] = (q_p0 != nullptr) ? __ldg(q_p0 + k_off + 4) : 0u;
            q_reg[ks][3] = (q_p1 != nullptr) ? __ldg(q_p1 + k_off + 4) : 0u;
        }
    }

    // 2. Online softmax state & O_acc[16][4]:
    float m_max0 = -1e30f, l_sum0 = 0.0f;
    float m_max1 = -1e30f, l_sum1 = 0.0f;
    float o_acc[16][4];
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        o_acc[dt][0] = 0.0f; o_acc[dt][1] = 0.0f;
        o_acc[dt][2] = 0.0f; o_acc[dt][3] = 0.0f;
    }

    int kv_beg = is_local ? max(0, q0_pos - (INK2_SWA_WIN - 1)) : 0;
    int kv_end = q_end_pos + 1;
    size_t kv_b_off  = static_cast<size_t>(phys_b) * kv_cap;
    size_t ch_b_off  = static_cast<size_t>(b) * S;

    constexpr float kScale = 0.08838834764831845f; // 128^-0.5
    int qpos_r0 = q0_pos + qr0;
    int qpos_r1 = q0_pos + qr1;

    // 3. Main K/V tile loop (B_c = 32):
    for (int k0 = kv_beg; k0 < kv_end; k0 += 32) {
        int cur_k = min(32, kv_end - k0);

        if constexpr (KV_Q8) {
            if (tid < 32) {
                int kp = k0 + tid;
                size_t sc_ix = (kv_b_off + kp) * 8 + hkv;
                s_ksc[tid] = (tid < cur_k) ? (__ldg(k_cache_sc + sc_ix) * kScale) : 0.0f;
                s_vsc[tid] = (tid < cur_k) ? __ldg(v_cache_sc + sc_ix) : 0.0f;
            }
        }

        // 3a. Load K[32][128] into s_u.kv.K[32][68] (512 chunks / 192 threads -> 3 iters):
        #pragma unroll
        for (int u = 0; u < 3; ++u) {
            int idx = u * 192 + tid;
            if (idx < 512) {
                int kr   = idx >> 4; // 0..31
                int colq = idx & 15; // 0..15 -> 8 elems
                uint4 kv = make_uint4(0u, 0u, 0u, 0u);
                if (kr < cur_k) {
                    int kp = k0 + kr;
                    if constexpr (KV_Q8) {
                        const uint32_t* kp_i8 = static_cast<const uint32_t*>(k_cache) + ((kv_b_off + kp) * 8 + hkv) * 32;
                        uint2 pk2 = __ldg(reinterpret_cast<const uint2*>(kp_i8) + colq);
                        kv.x = wv_ink_s8x2_to_bf162(static_cast<int8_t>(pk2.x & 0xFFu), static_cast<int8_t>((pk2.x >> 8) & 0xFFu));
                        kv.y = wv_ink_s8x2_to_bf162(static_cast<int8_t>((pk2.x >> 16) & 0xFFu), static_cast<int8_t>((pk2.x >> 24) & 0xFFu));
                        kv.z = wv_ink_s8x2_to_bf162(static_cast<int8_t>(pk2.y & 0xFFu), static_cast<int8_t>((pk2.y >> 8) & 0xFFu));
                        kv.w = wv_ink_s8x2_to_bf162(static_cast<int8_t>((pk2.y >> 16) & 0xFFu), static_cast<int8_t>((pk2.y >> 24) & 0xFFu));
                    } else {
                        const __nv_bfloat16* src;
                        if (is_local && k_chunk != nullptr && kp >= pos0) {
                            src = k_chunk + (ch_b_off + (kp - pos0)) * 1024 + hkv * 128;
                        } else {
                            int slot = is_local ? (kp % INK2_SWA_RING) : kp;
                            src = static_cast<const __nv_bfloat16*>(k_cache) + (kv_b_off + slot) * 1024 + hkv * 128;
                        }
                        kv = __ldg(reinterpret_cast<const uint4*>(src) + colq);
                    }
                }
                int c4 = colq * 4;
                s_u.kv.K[kr][c4 + 0] = kv.x; s_u.kv.K[kr][c4 + 1] = kv.y;
                s_u.kv.K[kr][c4 + 2] = kv.z; s_u.kv.K[kr][c4 + 3] = kv.w;
            }
        }

        // 3b. Load V[32][128] transposed + XOR-swizzled into s_u.kv.VT[128][16] (256 pairs / 192 threads -> 2 iters):
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            int idx = u * 192 + tid;
            if (idx < 256) {
                int kp_pair = idx >> 4; // 0..15
                int colq    = idx & 15; // 0..15 -> 8 d-elements
                int kr_e    = kp_pair * 2;
                int kr_o    = kr_e + 1;
                int d0      = colq * 8;
                auto st_vt = [&](int d, uint32_t val) {
                    int swz = kp_pair ^ ((((d >> 1) ^ (d >> 3)) & 3) << 2);
                    s_u.kv.VT[d][swz] = val;
                };
                if constexpr (KV_Q8) {
                    uint2 pke = (kr_e < cur_k)
                        ? __ldg(reinterpret_cast<const uint2*>(static_cast<const uint32_t*>(v_cache) + ((kv_b_off + (k0 + kr_e)) * 8 + hkv) * 32) + colq)
                        : make_uint2(0u, 0u);
                    uint2 pko = (kr_o < cur_k)
                        ? __ldg(reinterpret_cast<const uint2*>(static_cast<const uint32_t*>(v_cache) + ((kv_b_off + (k0 + kr_o)) * 8 + hkv) * 32) + colq)
                        : make_uint2(0u, 0u);
                    st_vt(d0 + 0, wv_ink_s8x2_to_bf162(static_cast<int8_t>(pke.x & 0xFFu),         static_cast<int8_t>(pko.x & 0xFFu)));
                    st_vt(d0 + 1, wv_ink_s8x2_to_bf162(static_cast<int8_t>((pke.x >> 8) & 0xFFu),  static_cast<int8_t>((pko.x >> 8) & 0xFFu)));
                    st_vt(d0 + 2, wv_ink_s8x2_to_bf162(static_cast<int8_t>((pke.x >> 16) & 0xFFu), static_cast<int8_t>((pko.x >> 16) & 0xFFu)));
                    st_vt(d0 + 3, wv_ink_s8x2_to_bf162(static_cast<int8_t>((pke.x >> 24) & 0xFFu), static_cast<int8_t>((pko.x >> 24) & 0xFFu)));
                    st_vt(d0 + 4, wv_ink_s8x2_to_bf162(static_cast<int8_t>(pke.y & 0xFFu),         static_cast<int8_t>(pko.y & 0xFFu)));
                    st_vt(d0 + 5, wv_ink_s8x2_to_bf162(static_cast<int8_t>((pke.y >> 8) & 0xFFu),  static_cast<int8_t>((pko.y >> 8) & 0xFFu)));
                    st_vt(d0 + 6, wv_ink_s8x2_to_bf162(static_cast<int8_t>((pke.y >> 16) & 0xFFu), static_cast<int8_t>((pko.y >> 16) & 0xFFu)));
                    st_vt(d0 + 7, wv_ink_s8x2_to_bf162(static_cast<int8_t>((pke.y >> 24) & 0xFFu), static_cast<int8_t>((pko.y >> 24) & 0xFFu)));
                } else {
                    uint4 ve = make_uint4(0u, 0u, 0u, 0u);
                    uint4 vo = make_uint4(0u, 0u, 0u, 0u);
                    if (kr_e < cur_k) {
                        int kp = k0 + kr_e;
                        const __nv_bfloat16* src = (is_local && v_chunk != nullptr && kp >= pos0)
                            ? (v_chunk + (ch_b_off + (kp - pos0)) * 1024 + hkv * 128)
                            : (static_cast<const __nv_bfloat16*>(v_cache) + (kv_b_off + (is_local ? (kp % INK2_SWA_RING) : kp)) * 1024 + hkv * 128);
                        ve = __ldg(reinterpret_cast<const uint4*>(src) + colq);
                    }
                    if (kr_o < cur_k) {
                        int kp = k0 + kr_o;
                        const __nv_bfloat16* src = (is_local && v_chunk != nullptr && kp >= pos0)
                            ? (v_chunk + (ch_b_off + (kp - pos0)) * 1024 + hkv * 128)
                            : (static_cast<const __nv_bfloat16*>(v_cache) + (kv_b_off + (is_local ? (kp % INK2_SWA_RING) : kp)) * 1024 + hkv * 128);
                        vo = __ldg(reinterpret_cast<const uint4*>(src) + colq);
                    }
                    st_vt(d0 + 0, __byte_perm(ve.x, vo.x, 0x5410)); st_vt(d0 + 1, __byte_perm(ve.x, vo.x, 0x7632));
                    st_vt(d0 + 2, __byte_perm(ve.y, vo.y, 0x5410)); st_vt(d0 + 3, __byte_perm(ve.y, vo.y, 0x7632));
                    st_vt(d0 + 4, __byte_perm(ve.z, vo.z, 0x5410)); st_vt(d0 + 5, __byte_perm(ve.z, vo.z, 0x7632));
                    st_vt(d0 + 6, __byte_perm(ve.w, vo.w, 0x5410)); st_vt(d0 + 7, __byte_perm(ve.w, vo.w, 0x7632));
                }
            }
        }

        // 3c. Far-block vs near-1024 window check (CTA-uniform!):
        int d_base  = q0_pos - (k0 + 31);
        bool is_far = (!is_local) && (d_base >= 1024) && (cur_k == 32);
        if (!is_far) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                int idx = u * 192 + tid;
                if (idx < 256) {
                    int ri = idx >> 2;
                    int c4 = idx & 3;
                    int d  = d_base + ri;
                    float4 val = (d >= 0 && d < 1024)
                        ? __ldg(reinterpret_cast<const float4*>(rel_proj_t + static_cast<size_t>(d) * 16) + c4)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
                    reinterpret_cast<float4*>(s_Rel[ri])[c4] = val;
                }
            }
        }
        __syncthreads();

        // 3d. S[16][32] = Q_reg[16][128] @ K[32][128]^T on BF16 Tensor Cores (0 Q-SMEM reads!):
        float S_tile[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            S_tile[nt][0] = 0.0f; S_tile[nt][1] = 0.0f;
            S_tile[nt][2] = 0.0f; S_tile[nt][3] = 0.0f;
        }
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            int k_off = ks * 8 + c_k;
            uint32_t qa0 = q_reg[ks][0];
            uint32_t qa1 = q_reg[ks][1];
            uint32_t qa2 = q_reg[ks][2];
            uint32_t qa3 = q_reg[ks][3];
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                int kn = nt * 8 + r0;
                uint32_t kb0 = s_u.kv.K[kn][k_off];
                uint32_t kb1 = s_u.kv.K[kn][k_off + 4];
                wv_mma_m16n8k16_bf16(
                    S_tile[nt][0], S_tile[nt][1], S_tile[nt][2], S_tile[nt][3],
                    qa0, qa1, qa2, qa3, kb0, kb1,
                    S_tile[nt][0], S_tile[nt][1], S_tile[nt][2], S_tile[nt][3]);
            }
        }

        // 3e. Scale (kScale or factored s_ksc) + optional near-window rel_proj_t bias & causal/SWA mask:
        if (is_far) {
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                int kc0 = nt * 8 + c_k * 2;
                float sc0 = KV_Q8 ? s_ksc[kc0]     : kScale;
                float sc1 = KV_Q8 ? s_ksc[kc0 + 1] : kScale;
                S_tile[nt][0] *= sc0; S_tile[nt][1] *= sc1;
                S_tile[nt][2] *= sc0; S_tile[nt][3] *= sc1;
            }
        } else {
            auto apply_row_bias = [&](int qr, int qpos, int idx_a, int idx_b) {
                const float4* rp = (qr < cur_q)
                    ? reinterpret_cast<const float4*>(r_all_f32 + ((ch_b_off + (s0 + qr)) * 24 + hq) * 16)
                    : nullptr;
                auto one_elem = [&](float s_val, int kr) -> float {
                    if (kr >= cur_k) return -1e30f;
                    int dist = qpos - (k0 + kr);
                    if (dist < 0 || (is_local && dist >= INK2_SWA_WIN)) return -1e30f;
                    float v = s_val * (KV_Q8 ? s_ksc[kr] : kScale);
                    if (dist < 1024 && rp != nullptr) {
                        int ri = dist - d_base;
                        #pragma unroll 1
                        for (int j = 0; j < 4; ++j) {
                            float4 rr = __ldg(rp + j);
                            float4 pp = reinterpret_cast<const float4*>(s_Rel[ri])[j];
                            v += rr.x * pp.x + rr.y * pp.y + rr.z * pp.z + rr.w * pp.w;
                        }
                    }
                    return v;
                };
                #pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    int kc0 = nt * 8 + c_k * 2;
                    S_tile[nt][idx_a] = one_elem(S_tile[nt][idx_a], kc0);
                    S_tile[nt][idx_b] = one_elem(S_tile[nt][idx_b], kc0 + 1);
                }
            };
            apply_row_bias(qr0, qpos_r0, 0, 1);
            apply_row_bias(qr1, qpos_r1, 2, 3);
        }

        // 3f. Fast MUFU.EX2 Online Softmax (4-lane warp reduction):
        float tmax0 = -1e30f, tmax1 = -1e30f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            tmax0 = fmaxf(tmax0, fmaxf(S_tile[nt][0], S_tile[nt][1]));
            tmax1 = fmaxf(tmax1, fmaxf(S_tile[nt][2], S_tile[nt][3]));
        }
        tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 1));
        tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 2));
        tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 1));
        tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 2));

        float m_new0 = fmaxf(m_max0, tmax0);
        float m_new1 = fmaxf(m_max1, tmax1);
        float alpha0 = (tmax0 > -1e29f) ? ((m_max0 > -1e29f) ? __expf(m_max0 - m_new0) : 0.0f) : 1.0f;
        float alpha1 = (tmax1 > -1e29f) ? ((m_max1 > -1e29f) ? __expf(m_max1 - m_new1) : 0.0f) : 1.0f;

        if (alpha0 != 1.0f || alpha1 != 1.0f) {
            #pragma unroll
            for (int dt = 0; dt < 16; ++dt) {
                o_acc[dt][0] *= alpha0; o_acc[dt][1] *= alpha0;
                o_acc[dt][2] *= alpha1; o_acc[dt][3] *= alpha1;
            }
        }

        uint32_t P_bf[4][2];
        float tsum0 = 0.0f, tsum1 = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            float p0, p1, p2, p3;
            if (is_far) {
                p0 = __expf(S_tile[nt][0] - m_new0);
                p1 = __expf(S_tile[nt][1] - m_new0);
                p2 = __expf(S_tile[nt][2] - m_new1);
                p3 = __expf(S_tile[nt][3] - m_new1);
            } else {
                p0 = (tmax0 > -1e29f && S_tile[nt][0] > -1e29f) ? __expf(S_tile[nt][0] - m_new0) : 0.0f;
                p1 = (tmax0 > -1e29f && S_tile[nt][1] > -1e29f) ? __expf(S_tile[nt][1] - m_new0) : 0.0f;
                p2 = (tmax1 > -1e29f && S_tile[nt][2] > -1e29f) ? __expf(S_tile[nt][2] - m_new1) : 0.0f;
                p3 = (tmax1 > -1e29f && S_tile[nt][3] > -1e29f) ? __expf(S_tile[nt][3] - m_new1) : 0.0f;
            }
            tsum0 += p0 + p1;
            tsum1 += p2 + p3;
            if constexpr (KV_Q8) {
                int kc0 = nt * 8 + c_k * 2;
                float vs0 = s_vsc[kc0], vs1 = s_vsc[kc0 + 1];
                P_bf[nt][0] = wv_ink_pack_bf162(p0 * vs0, p1 * vs1);
                P_bf[nt][1] = wv_ink_pack_bf162(p2 * vs0, p3 * vs1);
            } else {
                P_bf[nt][0] = wv_ink_pack_bf162(p0, p1);
                P_bf[nt][1] = wv_ink_pack_bf162(p2, p3);
            }
        }
        tsum0 += __shfl_xor_sync(0xffffffffu, tsum0, 1);
        tsum0 += __shfl_xor_sync(0xffffffffu, tsum0, 2);
        tsum1 += __shfl_xor_sync(0xffffffffu, tsum1, 1);
        tsum1 += __shfl_xor_sync(0xffffffffu, tsum1, 2);

        l_sum0 = l_sum0 * alpha0 + tsum0;
        l_sum1 = l_sum1 * alpha1 + tsum1;
        m_max0 = m_new0;
        m_max1 = m_new1;

        // 3g. O[16][128] += P[16][32] @ V[32][128] on BF16 Tensor Cores (0 bank conflicts!):
        #pragma unroll
        for (int k_sub = 0; k_sub < 2; ++k_sub) {
            uint32_t pa0 = P_bf[k_sub * 2 + 0][0];
            uint32_t pa1 = P_bf[k_sub * 2 + 0][1];
            uint32_t pa2 = P_bf[k_sub * 2 + 1][0];
            uint32_t pa3 = P_bf[k_sub * 2 + 1][1];
            int kp0 = k_sub * 8 + c_k;
            int kp1 = kp0 + 4;
            #pragma unroll
            for (int dt = 0; dt < 16; ++dt) {
                int d = dt * 8 + r0;
                int swz_mask = (((d >> 1) ^ (d >> 3)) & 3) << 2;
                uint32_t vb0 = s_u.kv.VT[d][kp0 ^ swz_mask];
                uint32_t vb1 = s_u.kv.VT[d][kp1 ^ swz_mask];
                wv_mma_m16n8k16_bf16(
                    o_acc[dt][0], o_acc[dt][1], o_acc[dt][2], o_acc[dt][3],
                    pa0, pa1, pa2, pa3, vb0, vb1,
                    o_acc[dt][0], o_acc[dt][1], o_acc[dt][2], o_acc[dt][3]);
            }
        }
        __syncthreads();
    }

    // 4. Fused Epilogue: Sink + BF16 round + sigmoid(gate_attn) + FWHT-128 + Q8_32!
    //    Reuses s_u.out[6][4][128] (4 passes of 4 rows/warp, ZERO DRAM intermediate!).
    auto calc_o_scale = [&](int qr, float fm, float fl) -> float {
        if (qr >= cur_q) return 0.0f;
        size_t mh = (static_cast<size_t>(b) * S + (s0 + qr)) * 24 + hq;
        float sink  = sinks_f32[mh];
        float m_tot = (fl > 0.0f) ? fmaxf(sink, fm) : sink;
        float denom = expf(sink - m_tot);
        if (fl > 0.0f) {
            float a = expf(fm - m_tot);
            denom += fl * a;
            return a / fmaxf(denom, 1e-20f);
        }
        return 0.0f;
    };
    float sc0 = calc_o_scale(qr0, m_max0, l_sum0);
    float sc1 = calc_o_scale(qr1, m_max1, l_sum1);

    auto emit_4rows = [&](int row_off) {
        __syncwarp();
        for (int rr = 0; rr < 4; ++rr) {
            int qr = mw * 16 + row_off + rr;
            if (qr < cur_q) {
                size_t mh = (static_cast<size_t>(b) * S + (s0 + qr)) * 24 + hq;
                const float* row_o = s_u.out[wid][rr] + lane * 4;
                float a0 = __bfloat162float(__float2bfloat16(row_o[0]));
                float a1 = __bfloat162float(__float2bfloat16(row_o[1]));
                float a2 = __bfloat162float(__float2bfloat16(row_o[2]));
                float a3 = __bfloat162float(__float2bfloat16(row_o[3]));
                const __nv_bfloat16* gp = gate_attn + mh * 128 + lane * 4;
                auto sig_bf = [](float a, __nv_bfloat16 gb) -> float {
                    float gv = __bfloat162float(gb);
                    return __bfloat162float(__float2bfloat16(a * (1.0f / (1.0f + expf(-gv)))));
                };
                float f0 = sig_bf(a0, gp[0]), f1 = sig_bf(a1, gp[1]);
                float f2 = sig_bf(a2, gp[2]), f3 = sig_bf(a3, gp[3]);
                wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
                wv_quant_q8_32_warp(f0, f1, f2, f3, lane, O_q8 + mh * 128, O_sc + mh * 4);
            }
        }
    };

    #pragma unroll 1
    for (int pass = 0; pass < 4; ++pass) {
        if (pass > 0) __syncwarp();
        bool hi_r = ((pass & 1) != 0);
        bool hi_h = ((pass & 2) != 0);
        if ((r0 >= 4) == hi_r) {
            int dst_r = hi_r ? (r0 - 4) : r0;
            float sc  = hi_h ? sc1 : sc0;
            #pragma unroll
            for (int dt = 0; dt < 16; ++dt) {
                s_u.out[wid][dst_r][dt * 8 + c_k * 2 + 0] = (hi_h ? o_acc[dt][2] : o_acc[dt][0]) * sc;
                s_u.out[wid][dst_r][dt * 8 + c_k * 2 + 1] = (hi_h ? o_acc[dt][3] : o_acc[dt][1]) * sc;
            }
        }
        emit_4rows(pass * 4);
    }

    wv_pdl_trigger();
}

// ============================================================================
// 4. Zero-Sync Multi-CTA GPU Radix-Histogram Grouping + M_TILE=128 Parallel-Unpack TC MoE
// ============================================================================

// Stage 1: Per-block 128-expert histogram (Grid <<<n_blks, 256>>>)
__global__ void wv_ink_radix_hist_stage1_kernel(
    const int32_t* __restrict__ top_eid, // [tot]
    int32_t* __restrict__ blk_cnt,       // [n_blks, 128]
    int tot, int n_experts) {
    wv_pdl_wait();
    __shared__ int32_t s_cnt[128];
    int tid = threadIdx.x;
    if (tid < 128) s_cnt[tid] = 0;
    __syncthreads();

    int p = blockIdx.x * 256 + tid;
    if (p < tot) {
        int e = top_eid[p];
        if (e >= 0 && e < n_experts) atomicAdd(&s_cnt[e], 1);
    }
    __syncthreads();

    if (tid < 128) {
        blk_cnt[static_cast<size_t>(blockIdx.x) * 128 + tid] = s_cnt[tid];
    }
    wv_pdl_trigger();
}

// Stage 2: Exclusive prefix sums + M_TILE=64 tile descriptor emission (Grid <<<1, 128>>>)
__global__ void wv_ink_radix_prefix_tiles64_kernel(
    const int32_t* __restrict__ blk_cnt, // [n_blks, 128]
    int32_t* __restrict__ blk_off,       // [n_blks, 128]
    const int32_t* __restrict__ tmap,
    const uint8_t* __restrict__ all_vram_slabs,
    const uint64_t* __restrict__ cold_slab_ptrs,
    int32_t* __restrict__ d_num_tiles64,
    int32_t* __restrict__ tile_hot_idx,
    uint64_t* __restrict__ tile_slab_ptr,
    int32_t* __restrict__ tile_beg,
    int32_t* __restrict__ tile_len,
    int n_blks, int n_experts) {
    wv_pdl_wait();
    int e = threadIdx.x; // 0..127
    __shared__ int32_t s_cnt[128];
    __shared__ int32_t s_off[128];
    __shared__ int32_t s_tcnt[128];
    __shared__ int32_t s_toff[128];
    __shared__ int32_t s_tot_tiles;

    int sum = 0;
    if (e < n_experts) {
        for (int b = 0; b < n_blks; ++b) {
            int c = blk_cnt[static_cast<size_t>(b) * 128 + e];
            blk_off[static_cast<size_t>(b) * 128 + e] = sum;
            sum += c;
        }
    }
    s_cnt[e]  = sum;
    s_tcnt[e] = (sum + 63) >> 6;
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

    if (e == 0) *d_num_tiles64 = s_tot_tiles;
    if (e < n_experts && sum > 0) {
        int base_e = s_off[e];
        for (int b = 0; b < n_blks; ++b) {
            blk_off[static_cast<size_t>(b) * 128 + e] += base_e;
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
            tile_beg[tb + t]      = base_e + t * 64;
            tile_len[tb + t]      = min(64, sum - t * 64);
        }
    }
    wv_pdl_trigger();
}

// Stage 3: Multi-CTA parallel pair scatter (Grid <<<n_blks, 256>>>)
__global__ void wv_ink_radix_scatter_stage3_kernel(
    const int32_t* __restrict__ top_eid,
    const int32_t* __restrict__ blk_off,
    int32_t* __restrict__ pair_tok,
    int32_t* __restrict__ pair_slot,
    int tot, int top_k, int n_experts) {
    wv_pdl_wait();
    __shared__ int32_t s_wr[128];
    int tid = threadIdx.x;
    if (tid < 128) s_wr[tid] = blk_off[static_cast<size_t>(blockIdx.x) * 128 + tid];
    __syncthreads();

    int p = blockIdx.x * 256 + tid;
    if (p < tot) {
        int e = top_eid[p];
        if (e >= 0 && e < n_experts) {
            int pos = atomicAdd(&s_wr[e], 1);
            pair_tok[pos]  = (top_k == 8) ? (p >> 3) : (p / top_k);
            pair_slot[pos] = p;
        }
    }
    wv_pdl_trigger();
}

// High-Throughput M_TILE=64, N_TILE=64, K_TILE=256 Direct-Grid Grouped MoE Gate+Up + Exact Softcapped SwiGLU:
// Grid <<<dim3(36, max_tiles64), 256>>> (4x less X_q8 traffic vs N_TILE=16, 19.25 KiB SMEM, 2+ CTAs/SM, 0 idle warps!)
__global__ __launch_bounds__(256, 2)
void wv_ink_moe_gate_up_swiglu_m64_n64_kernel(
    const int8_t* __restrict__ X_q8,              // [M, 2304]
    const float* __restrict__ X_sc,               // [M, 72]
    const uint32_t* __restrict__ cb4096_gmem,     // [4096] (16 KB L1-resident via __ldg!)
    const uint32_t* __restrict__ vq8_uv_gmem,     // [1024] (4 KB L1-resident via __ldg!)
    const uint32_t* __restrict__ hot_gu_pk3,      // [E_h, 288, 18, 3, 32]
    const uint16_t* __restrict__ hot_gu_sc,       // [E_h, 288, 18, 20]
    const int32_t* __restrict__ d_num_tiles64,
    const int32_t* __restrict__ tile_hot_idx,
    const uint64_t* __restrict__ tile_slab_ptr,
    const int32_t* __restrict__ tile_beg,
    const int32_t* __restrict__ tile_len,
    const int32_t* __restrict__ pair_tok,
    const int32_t* __restrict__ pair_slot,
    __nv_bfloat16* __restrict__ H_inter_bf16) {   // [M * top_k, 1152]

    union XUpSmem64 {
        uint32_t X[64][68];      // 17,408 B (256 B = 64 u32 + 4 pad -> 0 bank conflicts!)
        float    up[4][64][8];   //  8,192 B (4 gate/up warp pairs x 64 rows x 8 cols)
    };
    __shared__ XUpSmem64 s_u;                        // 17,408 B
    __shared__ float     s_Xsc[64][8];               //  2,048 B
    __shared__ int32_t   s_tok[64];                  //    256 B
    __shared__ __align__(16) uint32_t s_vq8_uv[1024];//  4,096 B (Total = 23,808 B = 23.25 KiB <= 24 KiB -> 2+ CTAs/SM!)

    wv_pdl_wait();
    int ti = blockIdx.y;
    if (ti >= *d_num_tiles64) { wv_pdl_trigger(); return; }

    int tid     = threadIdx.x; // 0..255
    int warp_id = tid >> 5;    // 0..7 (0,2,4,6 = gate; 1,3,5,7 = up)
    int lane    = tid & 31;
    int lane_n  = lane >> 2;
    int c_k     = lane & 3;
    int r0      = lane >> 2, r1 = r0 + 8;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;
    int tile8   = blockIdx.x * 8 + warp_id; // 0..287 (36 * 8 = 288)

    int h_idx = tile_hot_idx[ti];
    const uint8_t* slab = reinterpret_cast<const uint8_t*>(tile_slab_ptr[ti]);
    int beg = tile_beg[ti];
    int len = tile_len[ti];

    if (h_idx < 0) {
        reinterpret_cast<uint4*>(s_vq8_uv)[tid] = __ldg(reinterpret_cast<const uint4*>(vq8_uv_gmem) + tid);
    }
    if (tid < 64) {
        s_tok[tid] = (tid < len) ? pair_tok[beg + tid] : 0;
    }
    __syncthreads();

    float acc[4][4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        acc[t][0] = 0.f; acc[t][1] = 0.f; acc[t][2] = 0.f; acc[t][3] = 0.f;
    }

    const uint32_t* h_pk_b = (h_idx >= 0) ? (hot_gu_pk3 + (static_cast<size_t>(h_idx) * 288 + tile8) * (18 * 96) + lane) : nullptr;
    const uint16_t* h_sc_b = (h_idx >= 0) ? (hot_gu_sc  + (static_cast<size_t>(h_idx) * 288 + tile8) * (18 * 20)) : nullptr;
    const uint32_t* c_pk_b = (h_idx < 0)  ? (reinterpret_cast<const uint32_t*>(slab + INK2_OFF_GU_PK) + static_cast<size_t>(tile8) * (36 * 32) + lane) : nullptr;
    const uint16_t* c_sc_b = (h_idx < 0)  ? (reinterpret_cast<const uint16_t*>(slab + INK2_OFF_GU_SC) + static_cast<size_t>(tile8) * (36 * 8) + lane_n) : nullptr;

    for (int kc256 = 0; kc256 < 18; kc256 += 2) {
        // Cooperative load s_u.X[64][64] (1024 uint4 / 256 threads = 4 iters) + s_Xsc[64][8]:
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int idx  = u * 256 + tid;
            int ld_m = idx >> 4;
            int ld_q = idx & 15;
            uint4 xv = (ld_m < len)
                ? reinterpret_cast<const uint4*>(X_q8 + static_cast<size_t>(s_tok[ld_m]) * INK2_HIDDEN + kc256 * 128)[ld_q]
                : make_uint4(0u, 0u, 0u, 0u);
            int c4 = ld_q * 4;
            s_u.X[ld_m][c4 + 0] = xv.x; s_u.X[ld_m][c4 + 1] = xv.y;
            s_u.X[ld_m][c4 + 2] = xv.z; s_u.X[ld_m][c4 + 3] = xv.w;
        }
        if (tid < 128) {
            int ld_m = tid >> 1;
            int ld_c = tid & 1;
            float4 xs = (ld_m < len)
                ? reinterpret_cast<const float4*>(X_sc + static_cast<size_t>(s_tok[ld_m]) * 72 + kc256 * 4 + ld_c * 4)[0]
                : make_float4(0.f, 0.f, 0.f, 0.f);
            reinterpret_cast<float4*>(s_Xsc[ld_m])[ld_c] = xs;
        }
        __syncthreads();

        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            int kc128 = kc256 + c;
            if (h_idx >= 0) {
                const uint32_t* pk = h_pk_b + kc128 * 96;
                uint32_t w0 = __ldg(pk);
                uint32_t w1 = __ldg(pk + 32);
                uint32_t w2 = __ldg(pk + 64);
                float ws[4];
                wv_ink_unpack_vq12_meta(h_sc_b, kc128, lane_n, ws);
                #pragma unroll
                for (int sub = 0; sub < 4; ++sub) {
                    uint32_t i0, i1;
                    if (sub == 0)      { i0 = w0 & 0xFFFu;                     i1 = (w0 >> 12) & 0xFFFu; }
                    else if (sub == 1) { i0 = (w0 >> 24) | ((w1 & 0xFu) << 8); i1 = (w1 >> 4) & 0xFFFu; }
                    else if (sub == 2) { i0 = (w1 >> 16) & 0xFFFu;             i1 = (w1 >> 28) | ((w2 & 0xFFu) << 4); }
                    else               { i0 = (w2 >> 8) & 0xFFFu;              i1 = w2 >> 20; }
                    uint32_t b0 = __ldg(cb4096_gmem + i0);
                    uint32_t b1 = __ldg(cb4096_gmem + i1);
                    int s_off   = (c * 4 + sub) * 8 + c_k;
                    float ws0   = __shfl_sync(0xffffffffu, ws[sub], col_n0 * 4);
                    float ws1   = __shfl_sync(0xffffffffu, ws[sub], col_n1 * 4);
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        if (len > t * 16) {
                            int tr0 = t * 16 + r0;
                            int tr1 = t * 16 + r1;
                            float xs0 = s_Xsc[tr0][c * 4 + sub];
                            float xs1 = s_Xsc[tr1][c * 4 + sub];
                            int d0, d1, d2, d3;
                            wv_mma_m16n8k32_s8(d0, d1, d2, d3,
                                s_u.X[tr0][s_off], s_u.X[tr1][s_off],
                                s_u.X[tr0][s_off + 4], s_u.X[tr1][s_off + 4],
                                b0, b1);
                            acc[t][0] += static_cast<float>(d0) * (ws0 * xs0);
                            acc[t][1] += static_cast<float>(d1) * (ws1 * xs0);
                            acc[t][2] += static_cast<float>(d2) * (ws0 * xs1);
                            acc[t][3] += static_cast<float>(d3) * (ws1 * xs1);
                        }
                    }
                }
            } else {
                float ws[4];
                wv_ink_unpack_vq8_sb6_meta(c_sc_b, kc128, ws);
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t pkv = __ldg(c_pk_b + (kc128 * 2 + h) * 32);
                    uint32_t tb0[2], tb1[2];
                    wv_ink_unpack_vq8_h(pkv, lane, s_vq8_uv, tb0[0], tb1[0], tb0[1], tb1[1]);
                    #pragma unroll
                    for (int hs = 0; hs < 2; ++hs) {
                        int sub   = h * 2 + hs;
                        int s_off = (c * 4 + sub) * 8 + c_k;
                        float ws0 = __shfl_sync(0xffffffffu, ws[sub], col_n0 * 4);
                        float ws1 = __shfl_sync(0xffffffffu, ws[sub], col_n1 * 4);
                        #pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            if (len > t * 16) {
                                int tr0 = t * 16 + r0;
                                int tr1 = t * 16 + r1;
                                float xs0 = s_Xsc[tr0][c * 4 + sub];
                                float xs1 = s_Xsc[tr1][c * 4 + sub];
                                int d0, d1, d2, d3;
                                wv_mma_m16n8k32_s8(d0, d1, d2, d3,
                                    s_u.X[tr0][s_off], s_u.X[tr1][s_off],
                                    s_u.X[tr0][s_off + 4], s_u.X[tr1][s_off + 4],
                                    tb0[hs], tb1[hs]);
                                acc[t][0] += static_cast<float>(d0) * (ws0 * xs0);
                                acc[t][1] += static_cast<float>(d1) * (ws1 * xs0);
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

    int pair_id = warp_id >> 1; // 0..3
    if ((warp_id & 1) != 0) {
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int tr0 = t * 16 + r0;
            int tr1 = t * 16 + r1;
            s_u.up[pair_id][tr0][col_n0] = acc[t][0]; s_u.up[pair_id][tr0][col_n1] = acc[t][1];
            s_u.up[pair_id][tr1][col_n0] = acc[t][2]; s_u.up[pair_id][tr1][col_n1] = acc[t][3];
        }
    }
    __syncthreads();

    if ((warp_id & 1) == 0) {
        int inter8 = blockIdx.x * 4 + pair_id; // 0..143
        auto swig = [](float g, float u) -> __nv_bfloat16 {
            float gl = 4.0f * tanhf(g * 0.25f);
            float ul = 25.0f * tanhf(u * 0.04f);
            return __float2bfloat16(gl * (1.0f / (1.0f + __expf(-g))) * ul);
        };
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int tr0 = t * 16 + r0;
            int tr1 = t * 16 + r1;
            if (tr0 < len) {
                __nv_bfloat16* dst = H_inter_bf16 + static_cast<size_t>(pair_slot[beg + tr0]) * INK2_INTER + inter8 * 8;
                dst[col_n0] = swig(acc[t][0], s_u.up[pair_id][tr0][col_n0]);
                dst[col_n1] = swig(acc[t][1], s_u.up[pair_id][tr0][col_n1]);
            }
            if (tr1 < len) {
                __nv_bfloat16* dst = H_inter_bf16 + static_cast<size_t>(pair_slot[beg + tr1]) * INK2_INTER + inter8 * 8;
                dst[col_n0] = swig(acc[t][2], s_u.up[pair_id][tr1][col_n0]);
                dst[col_n1] = swig(acc[t][3], s_u.up[pair_id][tr1][col_n1]);
            }
        }
    }
    wv_pdl_trigger();
}

// High-Throughput M_TILE=64, N_TILE=64, K_TILE=256 Direct-Grid Grouped MoE Down Projection:
// Grid <<<dim3(36, max_tiles64), 256>>> (19.25 KiB SMEM, 2+ CTAs/SM, 4x less H_q8 traffic!)
__global__ __launch_bounds__(256, 2)
void wv_ink_moe_down_m64_n64_kernel(
    const int8_t* __restrict__ H_q8,              // [M * top_k, 1152]
    const float* __restrict__ H_sc,               // [M * top_k, 36]
    const uint32_t* __restrict__ cb4096_gmem,     // [4096]
    const uint32_t* __restrict__ vq8_uv_gmem,     // [1024]
    const uint32_t* __restrict__ hot_dn_pk3,      // [E_h, 288, 9, 3, 32]
    const uint16_t* __restrict__ hot_dn_sc,       // [E_h, 288, 9, 20]
    const int32_t* __restrict__ d_num_tiles64,
    const int32_t* __restrict__ tile_hot_idx,
    const uint64_t* __restrict__ tile_slab_ptr,
    const int32_t* __restrict__ tile_beg,
    const int32_t* __restrict__ tile_len,
    const int32_t* __restrict__ pair_slot,
    __nv_bfloat16* __restrict__ slot_out_bf16) {  // [M * top_k, 2304]

    __shared__ uint32_t s_H[64][68];                 // 17,408 B
    __shared__ float    s_Hsc[64][8];                //  2,048 B
    __shared__ int32_t  s_sl[64];                    //    256 B
    __shared__ __align__(16) uint32_t s_vq8_uv[1024];//  4,096 B (Total = 23,808 B = 23.25 KiB <= 24 KiB!)

    wv_pdl_wait();
    int ti = blockIdx.y;
    if (ti >= *d_num_tiles64) { wv_pdl_trigger(); return; }

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int lane_n  = lane >> 2;
    int c_k     = lane & 3;
    int r0      = lane >> 2, r1 = r0 + 8;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;
    int tile8   = blockIdx.x * 8 + warp_id; // 0..287

    int h_idx = tile_hot_idx[ti];
    const uint8_t* slab = reinterpret_cast<const uint8_t*>(tile_slab_ptr[ti]);
    int beg = tile_beg[ti];
    int len = tile_len[ti];

    if (h_idx < 0) {
        reinterpret_cast<uint4*>(s_vq8_uv)[tid] = __ldg(reinterpret_cast<const uint4*>(vq8_uv_gmem) + tid);
    }
    if (tid < 64) {
        s_sl[tid] = (tid < len) ? pair_slot[beg + tid] : 0;
    }
    __syncthreads();

    float acc[4][4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        acc[t][0] = 0.f; acc[t][1] = 0.f; acc[t][2] = 0.f; acc[t][3] = 0.f;
    }

    const uint32_t* h_pk_b = (h_idx >= 0) ? (hot_dn_pk3 + (static_cast<size_t>(h_idx) * 288 + tile8) * (9 * 96) + lane) : nullptr;
    const uint16_t* h_sc_b = (h_idx >= 0) ? (hot_dn_sc  + (static_cast<size_t>(h_idx) * 288 + tile8) * (9 * 20)) : nullptr;
    const uint32_t* c_pk_b = (h_idx < 0)  ? (reinterpret_cast<const uint32_t*>(slab + INK2_OFF_DN_PK) + static_cast<size_t>(tile8) * (18 * 32) + lane) : nullptr;
    const uint16_t* c_sc_b = (h_idx < 0)  ? (reinterpret_cast<const uint16_t*>(slab + INK2_OFF_DN_SC) + static_cast<size_t>(tile8) * (18 * 8) + lane_n) : nullptr;

    for (int kc256 = 0; kc256 < 9; kc256 += 2) {
        int has_c1 = (kc256 + 1 < 9) ? 1 : 0;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int idx  = u * 256 + tid;
            int ld_m = idx >> 4;
            int ld_q = idx & 15;
            bool ok_k = (ld_q < 8) || (has_c1 != 0);
            uint4 hv = (ld_m < len && ok_k)
                ? reinterpret_cast<const uint4*>(H_q8 + static_cast<size_t>(s_sl[ld_m]) * INK2_INTER + kc256 * 128)[ld_q]
                : make_uint4(0u, 0u, 0u, 0u);
            int c4 = ld_q * 4;
            s_H[ld_m][c4 + 0] = hv.x; s_H[ld_m][c4 + 1] = hv.y;
            s_H[ld_m][c4 + 2] = hv.z; s_H[ld_m][c4 + 3] = hv.w;
        }
        if (tid < 128) {
            int ld_m  = tid >> 1;
            int ld_c  = tid & 1;
            bool ok_c = (ld_c == 0) || (has_c1 != 0);
            float4 hs = (ld_m < len && ok_c)
                ? reinterpret_cast<const float4*>(H_sc + static_cast<size_t>(s_sl[ld_m]) * 36 + kc256 * 4 + ld_c * 4)[0]
                : make_float4(0.f, 0.f, 0.f, 0.f);
            reinterpret_cast<float4*>(s_Hsc[ld_m])[ld_c] = hs;
        }
        __syncthreads();

        for (int c = 0; c <= has_c1; ++c) {
            int kc128 = kc256 + c;
            if (h_idx >= 0) {
                const uint32_t* pk = h_pk_b + kc128 * 96;
                uint32_t w0 = __ldg(pk);
                uint32_t w1 = __ldg(pk + 32);
                uint32_t w2 = __ldg(pk + 64);
                float ws[4];
                wv_ink_unpack_vq12_meta(h_sc_b, kc128, lane_n, ws);
                #pragma unroll
                for (int sub = 0; sub < 4; ++sub) {
                    uint32_t i0, i1;
                    if (sub == 0)      { i0 = w0 & 0xFFFu;                     i1 = (w0 >> 12) & 0xFFFu; }
                    else if (sub == 1) { i0 = (w0 >> 24) | ((w1 & 0xFu) << 8); i1 = (w1 >> 4) & 0xFFFu; }
                    else if (sub == 2) { i0 = (w1 >> 16) & 0xFFFu;             i1 = (w1 >> 28) | ((w2 & 0xFFu) << 4); }
                    else               { i0 = (w2 >> 8) & 0xFFFu;              i1 = w2 >> 20; }
                    uint32_t b0 = __ldg(cb4096_gmem + i0);
                    uint32_t b1 = __ldg(cb4096_gmem + i1);
                    int s_off   = (c * 4 + sub) * 8 + c_k;
                    float ws0   = __shfl_sync(0xffffffffu, ws[sub], col_n0 * 4);
                    float ws1   = __shfl_sync(0xffffffffu, ws[sub], col_n1 * 4);
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        if (len > t * 16) {
                            int tr0 = t * 16 + r0;
                            int tr1 = t * 16 + r1;
                            float xs0 = s_Hsc[tr0][c * 4 + sub];
                            float xs1 = s_Hsc[tr1][c * 4 + sub];
                            int d0, d1, d2, d3;
                            wv_mma_m16n8k32_s8(d0, d1, d2, d3,
                                s_H[tr0][s_off], s_H[tr1][s_off],
                                s_H[tr0][s_off + 4], s_H[tr1][s_off + 4],
                                b0, b1);
                            acc[t][0] += static_cast<float>(d0) * (ws0 * xs0);
                            acc[t][1] += static_cast<float>(d1) * (ws1 * xs0);
                            acc[t][2] += static_cast<float>(d2) * (ws0 * xs1);
                            acc[t][3] += static_cast<float>(d3) * (ws1 * xs1);
                        }
                    }
                }
            } else {
                float ws[4];
                wv_ink_unpack_vq8_sb6_meta(c_sc_b, kc128, ws);
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t pkv = __ldg(c_pk_b + (kc128 * 2 + h) * 32);
                    uint32_t tb0[2], tb1[2];
                    wv_ink_unpack_vq8_h(pkv, lane, s_vq8_uv, tb0[0], tb1[0], tb0[1], tb1[1]);
                    #pragma unroll
                    for (int hs = 0; hs < 2; ++hs) {
                        int sub   = h * 2 + hs;
                        int s_off = (c * 4 + sub) * 8 + c_k;
                        float ws0 = __shfl_sync(0xffffffffu, ws[sub], col_n0 * 4);
                        float ws1 = __shfl_sync(0xffffffffu, ws[sub], col_n1 * 4);
                        #pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            if (len > t * 16) {
                                int tr0 = t * 16 + r0;
                                int tr1 = t * 16 + r1;
                                float xs0 = s_Hsc[tr0][c * 4 + sub];
                                float xs1 = s_Hsc[tr1][c * 4 + sub];
                                int d0, d1, d2, d3;
                                wv_mma_m16n8k32_s8(d0, d1, d2, d3,
                                    s_H[tr0][s_off], s_H[tr1][s_off],
                                    s_H[tr0][s_off + 4], s_H[tr1][s_off + 4],
                                    tb0[hs], tb1[hs]);
                                acc[t][0] += static_cast<float>(d0) * (ws0 * xs0);
                                acc[t][1] += static_cast<float>(d1) * (ws1 * xs0);
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

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int tr0 = t * 16 + r0;
        int tr1 = t * 16 + r1;
        if (tr0 < len) {
            __nv_bfloat16* dst = slot_out_bf16 + static_cast<size_t>(s_sl[tr0]) * INK2_HIDDEN + tile8 * 8;
            dst[col_n0] = __float2bfloat16(acc[t][0]);
            dst[col_n1] = __float2bfloat16(acc[t][1]);
        }
        if (tr1 < len) {
            __nv_bfloat16* dst = slot_out_bf16 + static_cast<size_t>(s_sl[tr1]) * INK2_HIDDEN + tile8 * 8;
            dst[col_n0] = __float2bfloat16(acc[t][2]);
            dst[col_n1] = __float2bfloat16(acc[t][3]);
        }
    }
    wv_pdl_trigger();
}

// ============================================================================
// 5. M_TILE = 64 BF16 TC Aux GEMMs (Router, wr+w_sink, kv_wproj, LoopGate Down & Apply)
// ============================================================================

// 2-Term BF16-Split TC Router GEMM (M > 32): Grid <<<dim3((n_experts + 31) / 32, (M + 63) / 64), 128>>>
__global__ __launch_bounds__(128, 4)
void wv_ink_router_gemm_tc_m64_kernel(
    const __nv_bfloat16* __restrict__ x_bf16, // [M, 2304]
    const float* __restrict__ router_w,       // [n_experts, 2304]
    float* __restrict__ router_sc,            // [M, n_experts]
    int n_experts, int M) {
    __shared__ uint32_t s_X[64][20];
    __shared__ uint32_t s_Whi[32][20];
    __shared__ uint32_t s_Wlo[32][20];

    wv_pdl_wait();
    int e_base = blockIdx.x * 32;
    int m_base = blockIdx.y * 64;
    int cur_m  = min(64, M - m_base);
    if (cur_m <= 0 || e_base >= n_experts) { wv_pdl_trigger(); return; }

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int r0      = lane >> 2, r1 = r0 + 8;
    int c_k     = lane & 3;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;
    int n_base  = e_base + warp_id * 8;

    float acc[4][4];
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        acc[mt][0] = 0.f; acc[mt][1] = 0.f; acc[mt][2] = 0.f; acc[mt][3] = 0.f;
    }

    for (int k32 = 0; k32 < INK2_HIDDEN; k32 += 32) {
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            int idx  = u * 128 + tid;
            int rm   = idx >> 2;
            int cq   = idx & 3;
            uint4 xv = (rm < cur_m)
                ? reinterpret_cast<const uint4*>(x_bf16 + static_cast<size_t>(m_base + rm) * INK2_HIDDEN + k32)[cq]
                : make_uint4(0u, 0u, 0u, 0u);
            int c4 = cq * 4;
            s_X[rm][c4 + 0] = xv.x; s_X[rm][c4 + 1] = xv.y;
            s_X[rm][c4 + 2] = xv.z; s_X[rm][c4 + 3] = xv.w;
        }
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            int idx = u * 128 + tid;
            int re  = idx >> 3;
            int cq  = idx & 7;
            int eg  = e_base + re;
            float4 wf = (eg < n_experts)
                ? __ldg(reinterpret_cast<const float4*>(router_w + static_cast<size_t>(eg) * INK2_HIDDEN + k32) + cq)
                : make_float4(0.f, 0.f, 0.f, 0.f);
            auto split2 = [](float a, float b, uint32_t& hi_u, uint32_t& lo_u) {
                __nv_bfloat16 ha = __float2bfloat16(a), hb = __float2bfloat16(b);
                __nv_bfloat16 la = __float2bfloat16(a - __bfloat162float(ha));
                __nv_bfloat16 lb = __float2bfloat16(b - __bfloat162float(hb));
                hi_u = static_cast<uint32_t>(__bfloat16_as_ushort(ha)) | (static_cast<uint32_t>(__bfloat16_as_ushort(hb)) << 16);
                lo_u = static_cast<uint32_t>(__bfloat16_as_ushort(la)) | (static_cast<uint32_t>(__bfloat16_as_ushort(lb)) << 16);
            };
            uint32_t h0, l0, h1, l1;
            split2(wf.x, wf.y, h0, l0);
            split2(wf.z, wf.w, h1, l1);
            s_Whi[re][cq * 2 + 0] = h0; s_Whi[re][cq * 2 + 1] = h1;
            s_Wlo[re][cq * 2 + 0] = l0; s_Wlo[re][cq * 2 + 1] = l1;
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            int off = ks * 8 + c_k;
            int wn  = warp_id * 8 + r0;
            uint32_t bhi0 = s_Whi[wn][off], bhi1 = s_Whi[wn][off + 4];
            uint32_t blo0 = s_Wlo[wn][off], blo1 = s_Wlo[wn][off + 4];
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                if (cur_m > mt * 16) {
                    int tr0 = mt * 16 + r0;
                    int tr1 = mt * 16 + r1;
                    uint32_t a0 = s_X[tr0][off],     a1 = s_X[tr1][off];
                    uint32_t a2 = s_X[tr0][off + 4], a3 = s_X[tr1][off + 4];
                    wv_mma_m16n8k16_bf16(
                        acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3],
                        a0, a1, a2, a3, bhi0, bhi1,
                        acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3]);
                    wv_mma_m16n8k16_bf16(
                        acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3],
                        a0, a1, a2, a3, blo0, blo1,
                        acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3]);
                }
            }
        }
        __syncthreads();
    }

    auto st_sig = [&](int m_row, int e_col, float v) {
        if (m_row < M && e_col < n_experts) {
            router_sc[static_cast<size_t>(m_row) * n_experts + e_col] = 1.0f / (1.0f + expf(-v));
        }
    };
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        int m0 = m_base + mt * 16 + r0;
        int m1 = m_base + mt * 16 + r1;
        st_sig(m0, n_base + col_n0, acc[mt][0]); st_sig(m0, n_base + col_n1, acc[mt][1]);
        st_sig(m1, n_base + col_n0, acc[mt][2]); st_sig(m1, n_base + col_n1, acc[mt][3]);
    }
    wv_pdl_trigger();
}

// Fused wr+w_sink BF16 TC GEMM (M > 32): Grid <<<dim3(13, (M + 63) / 64), 128>>>
__global__ __launch_bounds__(128, 6)
void wv_ink_wr_sink_tc_m64_kernel(
    const __nv_bfloat16* __restrict__ qin_bf16, // [M, 2304]
    const __nv_bfloat16* __restrict__ wr_bf16,  // [384, 2304]
    const __nv_bfloat16* __restrict__ ws_bf16,  // [24, 2304]
    __nv_bfloat16* __restrict__ r_out_bf16,     // [M, 384]
    float* __restrict__ sinks_out_f32,          // [M, 24]
    int M) {
    __shared__ uint32_t s_X[64][20];
    __shared__ uint32_t s_W[32][20];

    wv_pdl_wait();
    int bx     = blockIdx.x;
    int m_base = blockIdx.y * 64;
    int cur_m  = min(64, M - m_base);
    if (cur_m <= 0) { wv_pdl_trigger(); return; }

    bool is_wr = (bx < 12);
    int n_lim  = is_wr ? INK2_REL_DIM : 24;
    int n_tile = is_wr ? (bx * 32) : 0;
    const __nv_bfloat16* w_src = is_wr ? wr_bf16 : ws_bf16;

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int r0      = lane >> 2, r1 = r0 + 8;
    int c_k     = lane & 3;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;

    float acc[4][4];
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        acc[mt][0] = 0.f; acc[mt][1] = 0.f; acc[mt][2] = 0.f; acc[mt][3] = 0.f;
    }

    for (int k32 = 0; k32 < INK2_HIDDEN; k32 += 32) {
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            int idx  = u * 128 + tid;
            int rm   = idx >> 2;
            int cq   = idx & 3;
            uint4 xv = (rm < cur_m)
                ? reinterpret_cast<const uint4*>(qin_bf16 + static_cast<size_t>(m_base + rm) * INK2_HIDDEN + k32)[cq]
                : make_uint4(0u, 0u, 0u, 0u);
            int c4 = cq * 4;
            s_X[rm][c4 + 0] = xv.x; s_X[rm][c4 + 1] = xv.y;
            s_X[rm][c4 + 2] = xv.z; s_X[rm][c4 + 3] = xv.w;
        }
        {
            int rn = tid >> 2;
            int cq = tid & 3;
            int ng = n_tile + rn;
            uint4 wv = (ng < n_lim)
                ? __ldg(reinterpret_cast<const uint4*>(w_src + static_cast<size_t>(ng) * INK2_HIDDEN + k32) + cq)
                : make_uint4(0u, 0u, 0u, 0u);
            int c4 = cq * 4;
            s_W[rn][c4 + 0] = wv.x; s_W[rn][c4 + 1] = wv.y;
            s_W[rn][c4 + 2] = wv.z; s_W[rn][c4 + 3] = wv.w;
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            int off = ks * 8 + c_k;
            int wn  = warp_id * 8 + r0;
            uint32_t b0 = s_W[wn][off], b1 = s_W[wn][off + 4];
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                if (cur_m > mt * 16) {
                    int tr0 = mt * 16 + r0;
                    int tr1 = mt * 16 + r1;
                    wv_mma_m16n8k16_bf16(
                        acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3],
                        s_X[tr0][off], s_X[tr1][off], s_X[tr0][off + 4], s_X[tr1][off + 4],
                        b0, b1,
                        acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3]);
                }
            }
        }
        __syncthreads();
    }

    int n_base = n_tile + warp_id * 8;
    auto st_val = [&](int m_row, int n_col, float v) {
        if (m_row >= M || n_col >= n_lim) return;
        if (is_wr) {
            r_out_bf16[static_cast<size_t>(m_row) * INK2_REL_DIM + n_col] = __float2bfloat16(v);
        } else {
            sinks_out_f32[static_cast<size_t>(m_row) * 24 + n_col] = v;
        }
    };
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        int m0 = m_base + mt * 16 + r0;
        int m1 = m_base + mt * 16 + r1;
        st_val(m0, n_base + col_n0, acc[mt][0]); st_val(m0, n_base + col_n1, acc[mt][1]);
        st_val(m1, n_base + col_n0, acc[mt][2]); st_val(m1, n_base + col_n1, acc[mt][3]);
    }
    wv_pdl_trigger();
}

// Fused K+V Dynamic SConv Weight Projection on BF16 TC (M > 32): Grid <<<dim3(2, (M + 63) / 64), 128>>>
__global__ __launch_bounds__(128, 6)
void wv_ink_kv_wproj_tc_m64_kernel(
    const __nv_bfloat16* __restrict__ kin_bf16,
    const __nv_bfloat16* __restrict__ vin_bf16,
    const __nv_bfloat16* __restrict__ k_wproj,
    const __nv_bfloat16* __restrict__ v_wproj,
    float* __restrict__ dw_out,
    int M) {
    __shared__ uint32_t s_X[64][20];
    __shared__ uint32_t s_W[32][20];

    wv_pdl_wait();
    bool is_v  = (blockIdx.x != 0);
    int m_base = blockIdx.y * 64;
    int cur_m  = min(64, M - m_base);
    if (cur_m <= 0) { wv_pdl_trigger(); return; }

    const __nv_bfloat16* x_src = is_v ? vin_bf16 : kin_bf16;
    const __nv_bfloat16* w_src = is_v ? v_wproj  : k_wproj;

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int r0      = lane >> 2, r1 = r0 + 8;
    int c_k     = lane & 3;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;

    float acc[4][4];
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        acc[mt][0] = 0.f; acc[mt][1] = 0.f; acc[mt][2] = 0.f; acc[mt][3] = 0.f;
    }

    for (int k32 = 0; k32 < INK2_HIDDEN; k32 += 32) {
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            int idx  = u * 128 + tid;
            int rm   = idx >> 2;
            int cq   = idx & 3;
            uint4 xv = (rm < cur_m)
                ? reinterpret_cast<const uint4*>(x_src + static_cast<size_t>(m_base + rm) * INK2_HIDDEN + k32)[cq]
                : make_uint4(0u, 0u, 0u, 0u);
            int c4 = cq * 4;
            s_X[rm][c4 + 0] = xv.x; s_X[rm][c4 + 1] = xv.y;
            s_X[rm][c4 + 2] = xv.z; s_X[rm][c4 + 3] = xv.w;
        }
        {
            int rn = tid >> 2;
            int cq = tid & 3;
            uint4 wv = __ldg(reinterpret_cast<const uint4*>(w_src + static_cast<size_t>(rn) * INK2_HIDDEN + k32) + cq);
            int c4 = cq * 4;
            s_W[rn][c4 + 0] = wv.x; s_W[rn][c4 + 1] = wv.y;
            s_W[rn][c4 + 2] = wv.z; s_W[rn][c4 + 3] = wv.w;
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            int off = ks * 8 + c_k;
            int wn  = warp_id * 8 + r0;
            uint32_t b0 = s_W[wn][off], b1 = s_W[wn][off + 4];
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                if (cur_m > mt * 16) {
                    int tr0 = mt * 16 + r0;
                    int tr1 = mt * 16 + r1;
                    wv_mma_m16n8k16_bf16(
                        acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3],
                        s_X[tr0][off], s_X[tr1][off], s_X[tr0][off + 4], s_X[tr1][off + 4],
                        b0, b1,
                        acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3]);
                }
            }
        }
        __syncthreads();
    }

    int out_base = (is_v ? 32 : 0) + warp_id * 8;
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        int m0 = m_base + mt * 16 + r0;
        int m1 = m_base + mt * 16 + r1;
        if (m0 < M) {
            dw_out[static_cast<size_t>(m0) * 64 + out_base + col_n0] = acc[mt][0];
            dw_out[static_cast<size_t>(m0) * 64 + out_base + col_n1] = acc[mt][1];
        }
        if (m1 < M) {
            dw_out[static_cast<size_t>(m1) * 64 + out_base + col_n0] = acc[mt][2];
            dw_out[static_cast<size_t>(m1) * 64 + out_base + col_n1] = acc[mt][3];
        }
    }
    wv_pdl_trigger();
}

// 2-Stage LoopMessageGate Stage A1: Grid <<<dim3(M, n_slots), dim3(32, 6)>>>
__global__ void wv_ink_loop_gate_norm_cat_bf16_kernel(
    const float* __restrict__ src_in,
    const float* __restrict__ messages_f32,
    __nv_bfloat16* __restrict__ cat_out_bf16,
    int first_msg_slot,
    int shared_src,
    int M) {
    wv_pdl_wait();
    int m  = blockIdx.x;
    int sl = blockIdx.y;
    if (m >= M) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x;
    int wid  = threadIdx.y;

    __shared__ float s_dsum[6], s_msum[6];
    const float* sp = src_in + (static_cast<size_t>(shared_src ? 0 : sl) * M + m) * INK2_HIDDEN;
    const float* mp = messages_f32 + (static_cast<size_t>(first_msg_slot + sl) * M + m) * INK2_HIDDEN;

    float4 rv_s[3], rv_m[3];
    float d_sq = 0.0f, m_sq = 0.0f;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int base = (wid + it * 6) * 128 + lane * 4;
        float4 sv = reinterpret_cast<const float4*>(sp + base)[0];
        float4 mv = reinterpret_cast<const float4*>(mp + base)[0];
        rv_s[it] = sv; rv_m[it] = mv;
        d_sq += sv.x * sv.x + sv.y * sv.y + sv.z * sv.z + sv.w * sv.w;
        m_sq += mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w;
    }
    d_sq = wv_warp_sum_f32(d_sq);
    m_sq = wv_warp_sum_f32(m_sq);
    if (lane == 0) { s_dsum[wid] = d_sq; s_msum[wid] = m_sq; }
    __syncthreads();

    float inv_d = rsqrtf((s_dsum[0]+s_dsum[1]+s_dsum[2]+s_dsum[3]+s_dsum[4]+s_dsum[5]) * (1.0f / 2304.0f) + 1e-6f);
    float inv_m = rsqrtf((s_msum[0]+s_msum[1]+s_msum[2]+s_msum[3]+s_msum[4]+s_msum[5]) * (1.0f / 2304.0f) + 1e-6f);

    __nv_bfloat16* dst = cat_out_bf16 + static_cast<size_t>(sl * M + m) * 4608;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int base = (wid + it * 6) * 128 + lane * 4;
        float4 sv = rv_s[it], mv = rv_m[it];
        dst[base + 0] = __float2bfloat16(sv.x * inv_d); dst[base + 1] = __float2bfloat16(sv.y * inv_d);
        dst[base + 2] = __float2bfloat16(sv.z * inv_d); dst[base + 3] = __float2bfloat16(sv.w * inv_d);
        dst[INK2_HIDDEN + base + 0] = __float2bfloat16(mv.x * inv_m); dst[INK2_HIDDEN + base + 1] = __float2bfloat16(mv.y * inv_m);
        dst[INK2_HIDDEN + base + 2] = __float2bfloat16(mv.z * inv_m); dst[INK2_HIDDEN + base + 3] = __float2bfloat16(mv.w * inv_m);
    }
    wv_pdl_trigger();
}

// 2-Stage LoopMessageGate Stage A2: Grid <<<dim3(2, (M + 63) / 64, n_slots), 128>>>
__global__ __launch_bounds__(128, 6)
void wv_ink_loop_gate_down_tc_m64_kernel(
    const __nv_bfloat16* __restrict__ cat_bf16,
    WvInkCtrlPtrs3 w_down,
    float* __restrict__ h64_out,
    int M) {
    __shared__ uint32_t s_X[64][20];
    __shared__ uint32_t s_W[32][20];

    wv_pdl_wait();
    int n_tile = blockIdx.x * 32;
    int m_base = blockIdx.y * 64;
    int sl     = blockIdx.z;
    int cur_m  = min(64, M - m_base);
    if (cur_m <= 0) { wv_pdl_trigger(); return; }

    const __nv_bfloat16* x_sl = cat_bf16 + static_cast<size_t>(sl) * M * 4608;
    const __nv_bfloat16* w_sl = (sl == 0) ? w_down.p[0] : ((sl == 1) ? w_down.p[1] : w_down.p[2]);

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int r0      = lane >> 2, r1 = r0 + 8;
    int c_k     = lane & 3;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;

    float acc[4][4];
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        acc[mt][0] = 0.f; acc[mt][1] = 0.f; acc[mt][2] = 0.f; acc[mt][3] = 0.f;
    }

    for (int k32 = 0; k32 < 4608; k32 += 32) {
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            int idx  = u * 128 + tid;
            int rm   = idx >> 2;
            int cq   = idx & 3;
            uint4 xv = (rm < cur_m)
                ? reinterpret_cast<const uint4*>(x_sl + static_cast<size_t>(m_base + rm) * 4608 + k32)[cq]
                : make_uint4(0u, 0u, 0u, 0u);
            int c4 = cq * 4;
            s_X[rm][c4 + 0] = xv.x; s_X[rm][c4 + 1] = xv.y;
            s_X[rm][c4 + 2] = xv.z; s_X[rm][c4 + 3] = xv.w;
        }
        {
            int rn = tid >> 2;
            int cq = tid & 3;
            uint4 wv = __ldg(reinterpret_cast<const uint4*>(w_sl + static_cast<size_t>(n_tile + rn) * 4608 + k32) + cq);
            int c4 = cq * 4;
            s_W[rn][c4 + 0] = wv.x; s_W[rn][c4 + 1] = wv.y;
            s_W[rn][c4 + 2] = wv.z; s_W[rn][c4 + 3] = wv.w;
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            int off = ks * 8 + c_k;
            int wn  = warp_id * 8 + r0;
            uint32_t b0 = s_W[wn][off], b1 = s_W[wn][off + 4];
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                if (cur_m > mt * 16) {
                    int tr0 = mt * 16 + r0;
                    int tr1 = mt * 16 + r1;
                    wv_mma_m16n8k16_bf16(
                        acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3],
                        s_X[tr0][off], s_X[tr1][off], s_X[tr0][off + 4], s_X[tr1][off + 4],
                        b0, b1,
                        acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3]);
                }
            }
        }
        __syncthreads();
    }

    int n_base = n_tile + warp_id * 8;
    float* dst_sl = h64_out + static_cast<size_t>(sl) * M * 64;
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        int m0 = m_base + mt * 16 + r0;
        int m1 = m_base + mt * 16 + r1;
        if (m0 < M) {
            dst_sl[static_cast<size_t>(m0) * 64 + n_base + col_n0] = acc[mt][0];
            dst_sl[static_cast<size_t>(m0) * 64 + n_base + col_n1] = acc[mt][1];
        }
        if (m1 < M) {
            dst_sl[static_cast<size_t>(m1) * 64 + n_base + col_n0] = acc[mt][2];
            dst_sl[static_cast<size_t>(m1) * 64 + n_base + col_n1] = acc[mt][3];
        }
    }
    wv_pdl_trigger();
}

// 2-Term BF16-Split Tensor-Core LoopMessageGate Stage B Apply (replaces scalar 16-row tiled apply!):
// Computes f_acc = h64 @ w_forget^T and g_acc = h64 @ w_gain^T in ONE SMEM pass (K=64 -> 1 barrier!)
// Grid <<<dim3(72, (M + 63) / 64, n_slots), 128>>>, Static SMEM = 27,648 B (27 KiB <= 48 KiB)
__global__ __launch_bounds__(128, 1)
void wv_ink_loop_gate_apply_tc_m64_kernel(
    const float* __restrict__ src_in,
    float* __restrict__ dst_out,
    const float* __restrict__ messages_f32,
    const float* __restrict__ h64_in,
    WvInkCtrlPtrs3 w_forget,
    WvInkCtrlPtrs3 w_gain,
    int first_msg_slot,
    int shared_src,
    int M) {
    __shared__ uint32_t s_Hhi[64][36]; // 9,216 B (64 bf16 = 32 u32 + 4 pad)
    __shared__ uint32_t s_Hlo[64][36]; // 9,216 B
    __shared__ uint32_t s_Wf[32][36];  // 4,608 B
    __shared__ uint32_t s_Wg[32][36];  // 4,608 B (Total = 27,648 B)

    wv_pdl_wait();
    int n_tile = blockIdx.x * 32; // 0..2272 (72 * 32 = 2304)
    int m_base = blockIdx.y * 64;
    int sl     = blockIdx.z;
    int cur_m  = min(64, M - m_base);
    if (cur_m <= 0) { wv_pdl_trigger(); return; }

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int r0      = lane >> 2, r1 = r0 + 8;
    int c_k     = lane & 3;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;

    // 1. Load & 2-term split h64_in[cur_m][64] (1024 float4 / 128 threads = 8 iters):
    const float* h_sl = h64_in + static_cast<size_t>(sl) * M * 64;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        int idx = u * 128 + tid;
        int rm  = idx >> 4; // 0..63
        int cq  = idx & 15; // 0..15 -> 4 floats = 2 u32 bf162
        float4 hv = (rm < cur_m)
            ? reinterpret_cast<const float4*>(h_sl + static_cast<size_t>(m_base + rm) * 64)[cq]
            : make_float4(0.f, 0.f, 0.f, 0.f);
        auto split2 = [](float a, float b, uint32_t& hi_u, uint32_t& lo_u) {
            __nv_bfloat16 ha = __float2bfloat16(a), hb = __float2bfloat16(b);
            __nv_bfloat16 la = __float2bfloat16(a - __bfloat162float(ha));
            __nv_bfloat16 lb = __float2bfloat16(b - __bfloat162float(hb));
            hi_u = static_cast<uint32_t>(__bfloat16_as_ushort(ha)) | (static_cast<uint32_t>(__bfloat16_as_ushort(hb)) << 16);
            lo_u = static_cast<uint32_t>(__bfloat16_as_ushort(la)) | (static_cast<uint32_t>(__bfloat16_as_ushort(lb)) << 16);
        };
        uint32_t h0, l0, h1, l1;
        split2(hv.x, hv.y, h0, l0);
        split2(hv.z, hv.w, h1, l1);
        s_Hhi[rm][cq * 2 + 0] = h0; s_Hhi[rm][cq * 2 + 1] = h1;
        s_Hlo[rm][cq * 2 + 0] = l0; s_Hlo[rm][cq * 2 + 1] = l1;
    }

    // 2. Load w_forget[32][64] and w_gain[32][64] BF16 (256 uint4 each / 128 threads = 2 iters):
    const __nv_bfloat16* wf_sl = (sl == 0) ? w_forget.p[0] : ((sl == 1) ? w_forget.p[1] : w_forget.p[2]);
    const __nv_bfloat16* wg_sl = (sl == 0) ? w_gain.p[0]   : ((sl == 1) ? w_gain.p[1]   : w_gain.p[2]);
    #pragma unroll
    for (int u = 0; u < 2; ++u) {
        int idx = u * 128 + tid;
        int rn  = idx >> 3; // 0..31
        int cq  = idx & 7;  // 0..7 -> 8 bf16 = 4 u32
        uint4 fv = __ldg(reinterpret_cast<const uint4*>(wf_sl + static_cast<size_t>(n_tile + rn) * 64) + cq);
        uint4 gv = __ldg(reinterpret_cast<const uint4*>(wg_sl + static_cast<size_t>(n_tile + rn) * 64) + cq);
        int c4 = cq * 4;
        s_Wf[rn][c4 + 0] = fv.x; s_Wf[rn][c4 + 1] = fv.y; s_Wf[rn][c4 + 2] = fv.z; s_Wf[rn][c4 + 3] = fv.w;
        s_Wg[rn][c4 + 0] = gv.x; s_Wg[rn][c4 + 1] = gv.y; s_Wg[rn][c4 + 2] = gv.z; s_Wg[rn][c4 + 3] = gv.w;
    }
    __syncthreads();

    float f_acc[4][4], g_acc[4][4];
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        f_acc[mt][0] = 0.f; f_acc[mt][1] = 0.f; f_acc[mt][2] = 0.f; f_acc[mt][3] = 0.f;
        g_acc[mt][0] = 0.f; g_acc[mt][1] = 0.f; g_acc[mt][2] = 0.f; g_acc[mt][3] = 0.f;
    }

    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        int off = ks * 8 + c_k;
        int wn  = warp_id * 8 + r0;
        uint32_t bf0 = s_Wf[wn][off], bf1 = s_Wf[wn][off + 4];
        uint32_t bg0 = s_Wg[wn][off], bg1 = s_Wg[wn][off + 4];
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            if (cur_m > mt * 16) {
                int tr0 = mt * 16 + r0;
                int tr1 = mt * 16 + r1;
                uint32_t ah0 = s_Hhi[tr0][off], ah1 = s_Hhi[tr1][off], ah2 = s_Hhi[tr0][off + 4], ah3 = s_Hhi[tr1][off + 4];
                uint32_t al0 = s_Hlo[tr0][off], al1 = s_Hlo[tr1][off], al2 = s_Hlo[tr0][off + 4], al3 = s_Hlo[tr1][off + 4];
                wv_mma_m16n8k16_bf16(f_acc[mt][0], f_acc[mt][1], f_acc[mt][2], f_acc[mt][3], ah0, ah1, ah2, ah3, bf0, bf1, f_acc[mt][0], f_acc[mt][1], f_acc[mt][2], f_acc[mt][3]);
                wv_mma_m16n8k16_bf16(f_acc[mt][0], f_acc[mt][1], f_acc[mt][2], f_acc[mt][3], al0, al1, al2, al3, bf0, bf1, f_acc[mt][0], f_acc[mt][1], f_acc[mt][2], f_acc[mt][3]);
                wv_mma_m16n8k16_bf16(g_acc[mt][0], g_acc[mt][1], g_acc[mt][2], g_acc[mt][3], ah0, ah1, ah2, ah3, bg0, bg1, g_acc[mt][0], g_acc[mt][1], g_acc[mt][2], g_acc[mt][3]);
                wv_mma_m16n8k16_bf16(g_acc[mt][0], g_acc[mt][1], g_acc[mt][2], g_acc[mt][3], al0, al1, al2, al3, bg0, bg1, g_acc[mt][0], g_acc[mt][1], g_acc[mt][2], g_acc[mt][3]);
            }
        }
    }

    int n0 = n_tile + warp_id * 8 + col_n0;
    int n1 = n_tile + warp_id * 8 + col_n1;
    auto apply_st = [&](int m_row, int col, float fv, float gv) {
        if (m_row >= M) return;
        size_t s_off = (static_cast<size_t>(shared_src ? 0 : sl) * M + m_row) * INK2_HIDDEN + col;
        size_t m_off = (static_cast<size_t>(first_msg_slot + sl) * M + m_row) * INK2_HIDDEN + col;
        size_t d_off = static_cast<size_t>(sl * M + m_row) * INK2_HIDDEN + col;
        dst_out[d_off] = (1.0f - tanhf(fv)) * src_in[s_off] + tanhf(gv) * messages_f32[m_off];
    };
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        int m0 = m_base + mt * 16 + r0;
        int m1 = m_base + mt * 16 + r1;
        apply_st(m0, n0, f_acc[mt][0], g_acc[mt][0]); apply_st(m0, n1, f_acc[mt][1], g_acc[mt][1]);
        apply_st(m1, n0, f_acc[mt][2], g_acc[mt][2]); apply_st(m1, n1, f_acc[mt][3], g_acc[mt][3]);
    }
    wv_pdl_trigger();
}

// ============================================================================
// 6. N_TILE = 64 SWZ8_G32 INT8 TC Linear & Fused QKVG (2x less X_q8 traffic!)
// ============================================================================

// High-throughput CTA_M=64, N_TILE=64 SWZ8_G32 Linear (256 threads, 19.5 KiB SMEM, 0.0-ULP exact!)
// Grid <<<dim3((N + 63) / 64, (M + 63) / 64), 256>>>
template<bool OUT_BF16, bool SOFTCAP30>
__global__ __launch_bounds__(256, 2)
void wv_ink_swz8_mma_linear_m64_n64_kernel(
    const uint2* __restrict__ w_pk,   // [N/8, K/32, 32] uint2
    const __half* __restrict__ w_sc,  // [N/8, K/32, 8] fp16
    const float* __restrict__ bias,   // [N] or nullptr
    const int8_t* __restrict__ X_q8,  // [M, K]
    const float* __restrict__ X_sc,   // [M, K/32]
    void* __restrict__ dst,
    int K, int N, int M) {
    __shared__ uint32_t s_X[64][68]; // 17,408 B
    __shared__ float    s_Xsc[64][8];//  2,048 B (Total = 19,456 B <= 48 KiB, 2+ CTAs/SM!)

    wv_pdl_wait();
    int tid     = threadIdx.x; // 0..255
    int warp_id = tid >> 5;    // 0..7
    int lane    = tid & 31;
    int lane_n  = lane >> 2;
    int c_k     = lane & 3;
    int r0      = lane >> 2, r1 = r0 + 8;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;
    int n_tile8 = blockIdx.x * 8 + warp_id;
    int out_n0  = n_tile8 * 8 + col_n0;
    int m_base  = blockIdx.y * 64;
    int cur_m   = min(64, M - m_base);
    if (cur_m <= 0) { wv_pdl_trigger(); return; }

    int n_kc128   = K >> 7;
    int n_kc32    = K >> 5;
    bool active_n = (n_tile8 * 8 < N);
    const uint2*  pk_base = active_n ? (w_pk + static_cast<size_t>(n_tile8) * n_kc32 * 32 + lane) : nullptr;
    const __half* sc_base = active_n ? (w_sc + static_cast<size_t>(n_tile8) * n_kc32 * 8 + lane_n) : nullptr;

    float acc[4][4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        acc[t][0] = 0.f; acc[t][1] = 0.f; acc[t][2] = 0.f; acc[t][3] = 0.f;
    }
    constexpr float kInv127 = 1.0f / 127.0f;

    for (int kc256 = 0; kc256 < n_kc128; kc256 += 2) {
        int has_c1 = (kc256 + 1 < n_kc128) ? 1 : 0;
        // 64 * 16 = 1024 uint4 / 256 threads = 4 iters/thread (2x faster staging than 128-thread kernel!)
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int idx  = u * 256 + tid;
            int ld_m = idx >> 4;
            int ld_q = idx & 15;
            bool ok_k = (ld_q < 8) || (has_c1 != 0);
            uint4 xv = (ld_m < cur_m && ok_k)
                ? reinterpret_cast<const uint4*>(X_q8 + static_cast<size_t>(m_base + ld_m) * K + kc256 * 128)[ld_q]
                : make_uint4(0u, 0u, 0u, 0u);
            int c4 = ld_q * 4;
            s_X[ld_m][c4 + 0] = xv.x; s_X[ld_m][c4 + 1] = xv.y;
            s_X[ld_m][c4 + 2] = xv.z; s_X[ld_m][c4 + 3] = xv.w;
        }
        if (tid < 128) {
            int ld_m  = tid >> 1;
            int ld_c  = tid & 1;
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
                    uint2 pkv = __ldg(pk_base + kc32 * 32);
                    float ws_self = __half2float(__ldg(sc_base + kc32 * 8)) * kInv127;
                    float ws0 = __shfl_sync(0xffffffffu, ws_self, col_n0 * 4);
                    float ws1 = __shfl_sync(0xffffffffu, ws_self, col_n1 * 4);
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        if (cur_m > t * 16) {
                            int tr0 = t * 16 + r0;
                            int tr1 = t * 16 + r1;
                            float xs0 = s_Xsc[tr0][c * 4 + sub];
                            float xs1 = s_Xsc[tr1][c * 4 + sub];
                            int d0, d1, d2, d3;
                            wv_mma_m16n8k32_s8(d0, d1, d2, d3,
                                s_X[tr0][s_off], s_X[tr1][s_off],
                                s_X[tr0][s_off + 4], s_X[tr1][s_off + 4],
                                pkv.x, pkv.y);
                            acc[t][0] += static_cast<float>(d0) * (ws0 * xs0);
                            acc[t][1] += static_cast<float>(d1) * (ws1 * xs0);
                            acc[t][2] += static_cast<float>(d2) * (ws0 * xs1);
                            acc[t][3] += static_cast<float>(d3) * (ws1 * xs1);
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
        for (int t = 0; t < 4; ++t) {
            int m0 = m_base + t * 16 + r0;
            int m1 = m_base + t * 16 + r1;
            if (m0 < M) { st_fn(m0, out_n0, acc[t][0] + b0_bias); st_fn(m0, out_n0 + 1, acc[t][1] + b1_bias); }
            if (m1 < M) { st_fn(m1, out_n0, acc[t][2] + b0_bias); st_fn(m1, out_n0 + 1, acc[t][3] + b1_bias); }
        }
    }
    wv_pdl_trigger();
}

// Single-Launch Fused Q[3072] + K[1024] + V[1024] + G[3072] N_TILE=64 SWZ8_G32 (all 48 visits!):
// Grid <<<dim3(128, (M + 63) / 64), 256>>> (bx: 0..47 Q, 48..63 K, 64..79 V, 80..127 G)
__global__ __launch_bounds__(256, 2)
void wv_ink_swz8_fused_qkvg_m64_n64_kernel(
    WvInkSwz8Linear wq, WvInkSwz8Linear wk, WvInkSwz8Linear wv, WvInkSwz8Linear wg,
    const int8_t* __restrict__ q8_q, const float* __restrict__ sc_q,
    const int8_t* __restrict__ q8_k, const float* __restrict__ sc_k,
    const int8_t* __restrict__ q8_v, const float* __restrict__ sc_v,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    __nv_bfloat16* __restrict__ g_out,
    int M) {
    __shared__ uint32_t s_X[64][68];
    __shared__ float    s_Xsc[64][8];

    wv_pdl_wait();
    int bx     = blockIdx.x; // 0..127
    int m_base = blockIdx.y * 64;
    int cur_m  = min(64, M - m_base);
    if (cur_m <= 0) { wv_pdl_trigger(); return; }

    int tid     = threadIdx.x;
    int warp_id = tid >> 5;
    int lane    = tid & 31;
    int lane_n  = lane >> 2;
    int c_k     = lane & 3;
    int r0      = lane >> 2, r1 = r0 + 8;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;

    const uint2*   w_pk;
    const __half*  w_sc;
    const int8_t*  X_q8;
    const float*   X_sc;
    __nv_bfloat16* dst;
    int n_tile8, N;
    if (bx < 48) {
        w_pk = wq.pk; w_sc = wq.sc; X_q8 = q8_q; X_sc = sc_q; dst = q_out; n_tile8 = bx * 8 + warp_id; N = INK2_ATTN_DIM;
    } else if (bx < 64) {
        w_pk = wk.pk; w_sc = wk.sc; X_q8 = q8_k; X_sc = sc_k; dst = k_out; n_tile8 = (bx - 48) * 8 + warp_id; N = INK2_KV_DIM;
    } else if (bx < 80) {
        w_pk = wv.pk; w_sc = wv.sc; X_q8 = q8_v; X_sc = sc_v; dst = v_out; n_tile8 = (bx - 64) * 8 + warp_id; N = INK2_KV_DIM;
    } else {
        w_pk = wg.pk; w_sc = wg.sc; X_q8 = q8_q; X_sc = sc_q; dst = g_out; n_tile8 = (bx - 80) * 8 + warp_id; N = INK2_ATTN_DIM;
    }

    int out_n0 = n_tile8 * 8 + col_n0;
    constexpr int n_kc128   = INK2_HIDDEN >> 7; // 18
    constexpr int n_kc32    = INK2_HIDDEN >> 5; // 72
    constexpr float kInv127 = 1.0f / 127.0f;
    const uint2*  pk_base   = w_pk + static_cast<size_t>(n_tile8) * n_kc32 * 32 + lane;
    const __half* sc_base   = w_sc + static_cast<size_t>(n_tile8) * n_kc32 * 8 + lane_n;

    float acc[4][4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        acc[t][0] = 0.f; acc[t][1] = 0.f; acc[t][2] = 0.f; acc[t][3] = 0.f;
    }

    for (int kc256 = 0; kc256 < n_kc128; kc256 += 2) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int idx  = u * 256 + tid;
            int ld_m = idx >> 4;
            int ld_q = idx & 15;
            uint4 xv = (ld_m < cur_m)
                ? reinterpret_cast<const uint4*>(X_q8 + static_cast<size_t>(m_base + ld_m) * INK2_HIDDEN + kc256 * 128)[ld_q]
                : make_uint4(0u, 0u, 0u, 0u);
            int c4 = ld_q * 4;
            s_X[ld_m][c4 + 0] = xv.x; s_X[ld_m][c4 + 1] = xv.y;
            s_X[ld_m][c4 + 2] = xv.z; s_X[ld_m][c4 + 3] = xv.w;
        }
        if (tid < 128) {
            int ld_m = tid >> 1;
            int ld_c = tid & 1;
            float4 xs = (ld_m < cur_m)
                ? reinterpret_cast<const float4*>(X_sc + static_cast<size_t>(m_base + ld_m) * n_kc32 + kc256 * 4 + ld_c * 4)[0]
                : make_float4(0.f, 0.f, 0.f, 0.f);
            reinterpret_cast<float4*>(s_Xsc[ld_m])[ld_c] = xs;
        }
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
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (cur_m > t * 16) {
                        int tr0 = t * 16 + r0;
                        int tr1 = t * 16 + r1;
                        float xs0 = s_Xsc[tr0][c * 4 + sub];
                        float xs1 = s_Xsc[tr1][c * 4 + sub];
                        int d0, d1, d2, d3;
                        wv_mma_m16n8k32_s8(d0, d1, d2, d3,
                            s_X[tr0][s_off], s_X[tr1][s_off],
                            s_X[tr0][s_off + 4], s_X[tr1][s_off + 4],
                            pkv.x, pkv.y);
                        acc[t][0] += static_cast<float>(d0) * (ws0 * xs0);
                        acc[t][1] += static_cast<float>(d1) * (ws1 * xs0);
                        acc[t][2] += static_cast<float>(d2) * (ws0 * xs1);
                        acc[t][3] += static_cast<float>(d3) * (ws1 * xs1);
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int m0 = m_base + t * 16 + r0;
        int m1 = m_base + t * 16 + r1;
        if (m0 < M) {
            size_t row = static_cast<size_t>(m0) * N;
            dst[row + out_n0]     = __float2bfloat16(acc[t][0]);
            dst[row + out_n0 + 1] = __float2bfloat16(acc[t][1]);
        }
        if (m1 < M) {
            size_t row = static_cast<size_t>(m1) * N;
            dst[row + out_n0]     = __float2bfloat16(acc[t][2]);
            dst[row + out_n0 + 1] = __float2bfloat16(acc[t][3]);
        }
    }
    wv_pdl_trigger();
}

} // namespace wv

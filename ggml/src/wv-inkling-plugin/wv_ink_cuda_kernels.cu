#include "wv_ink_plugin_internal.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <mutex>
#include <vector>

#ifdef WV_OBFUSCATE_SYMBOLS
#define wv_ink_plugin_fwht128_q8_kernel         s_twp_00
#define wv_ink_plugin_ext3_gemv_kernel          s_twp_01
#define wv_ink_plugin_ext3_multi_gemv_kernel    s_twp_02
#define wv_ink_plugin_ext3_tc_m64_n64_kernel    s_twp_03
#define wv_ink_plugin_ext012_mm_kernel          s_twp_04
#define wv_ink_plugin_ext012_tc_m64_n64_kernel  s_twp_05
#define wv_ink_plugin_mmid_zero_and_hist_kernel s_twp_06
#define wv_ink_plugin_mmid_plan_kernel          s_twp_07
#define wv_ink_plugin_mmid_scatter_kernel       s_twp_08
#define wv_ink_plugin_mmid_smallm_kernel        s_twp_09
#define wv_ink_plugin_mmid_grouped8_kernel      s_twp_10
#define wv_ink_plugin_mmid_tc_m64_n64_kernel    s_twp_11
#endif

namespace twp_detail {

struct WvInkPluginDevState {
    bool      tables_ready = false;
    uint32_t* d_cb4096     = nullptr; // [4096] (16 KiB)
    uint32_t* d_vq8_uv     = nullptr; // [1024] (4 KiB: [2][512])
    uint4*    d_lut4       = nullptr; // [8]    (128 B)
    uint8_t*  d_aux_blob   = nullptr; // QWV2 aux blob on GPU
    size_t    aux_bytes    = 0;

    int8_t*   d_yq8        = nullptr;
    float*    d_ysc        = nullptr;
    size_t    y_cap_elems  = 0;

    int32_t*  d_pair_e     = nullptr; // [pair_cap] (exp_cur scratch)
    int32_t*  d_pair_j     = nullptr; // [pair_cap]
    int32_t*  d_pair_t     = nullptr; // [pair_cap]
    int32_t*  d_exp_cnt    = nullptr; // [exp_cap]
    int32_t*  d_exp_off    = nullptr; // [exp_cap + 1]
    int32_t*  d_active_e   = nullptr; // [exp_cap]
    int32_t*  d_meta_cnt   = nullptr; // [2]: [0] = num_active, [1] = num_tiles64
    int32_t*  d_tile_exp   = nullptr; // [pair_cap]
    int32_t*  d_tile_beg   = nullptr; // [pair_cap]
    int32_t*  d_tile_len   = nullptr; // [pair_cap]
    size_t    pair_cap     = 0;
    size_t    exp_cap      = 0;
};

static std::mutex          g_cuda_mtx;
static WvInkPluginDevState g_devs[16];
static std::vector<uint8_t> g_host_aux_mirror;
static wv_ink_host_tables  g_host_tab_mirror;
static bool                g_have_mirror = false;
static uint64_t            g_aux_epoch   = 1;
static uint64_t            g_dev_epoch[16] = {0};

static int32_t sync_dev_tables_locked(int dev_id) {
    if (dev_id < 0 || dev_id >= 16) {
        wv_ink_set_error("Invalid CUDA device_id");
        return -1;
    }
    WvInkPluginDevState & st = g_devs[dev_id];
    if (st.tables_ready && g_dev_epoch[dev_id] == g_aux_epoch) {
        return 0;
    }
    cudaError_t err = cudaSetDevice(dev_id);
    if (err != cudaSuccess) {
        wv_ink_set_error(cudaGetErrorString(err));
        return -1;
    }
    if (!st.d_cb4096) {
        if (cudaMalloc(&st.d_cb4096, 16384) != cudaSuccess) return -1;
    }
    if (!st.d_vq8_uv) {
        if (cudaMalloc(&st.d_vq8_uv, 4096) != cudaSuccess) return -1;
    }
    if (!st.d_lut4) {
        if (cudaMalloc(&st.d_lut4, 128) != cudaSuccess) return -1;
    }
    if (g_have_mirror) {
        cudaMemcpy(st.d_cb4096, g_host_tab_mirror.cb4096, 16384, cudaMemcpyHostToDevice);
        cudaMemcpy(st.d_vq8_uv, g_host_tab_mirror.vq8_uv, 4096,  cudaMemcpyHostToDevice);
        cudaMemcpy(st.d_lut4,   g_host_tab_mirror.lut4,   128,   cudaMemcpyHostToDevice);
    }
    if (st.d_aux_blob) {
        cudaFree(st.d_aux_blob);
        st.d_aux_blob = nullptr;
        st.aux_bytes  = 0;
    }
    if (!g_host_aux_mirror.empty()) {
        if (cudaMalloc(&st.d_aux_blob, g_host_aux_mirror.size()) != cudaSuccess) return -1;
        cudaMemcpy(st.d_aux_blob, g_host_aux_mirror.data(), g_host_aux_mirror.size(), cudaMemcpyHostToDevice);
        st.aux_bytes = g_host_aux_mirror.size();
    }
    st.tables_ready = true;
    g_dev_epoch[dev_id] = g_aux_epoch;
    return 0;
}

static int32_t ensure_act_workspace_locked(int dev_id, size_t total_elems) {
    WvInkPluginDevState & st = g_devs[dev_id];
    if (total_elems <= st.y_cap_elems && st.d_yq8 && st.d_ysc) {
        return 0;
    }
    cudaSetDevice(dev_id);
    if (st.d_yq8) { cudaFree(st.d_yq8); st.d_yq8 = nullptr; }
    if (st.d_ysc) { cudaFree(st.d_ysc); st.d_ysc = nullptr; }
    const size_t cap = std::max<size_t>(total_elems, 256ull * 2304ull);
    if (cudaMalloc(&st.d_yq8, cap * sizeof(int8_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&st.d_ysc, (cap / 32) * sizeof(float)) != cudaSuccess) return -1;
    st.y_cap_elems = cap;
    return 0;
}

static int32_t ensure_pair_workspace_locked(int dev_id, size_t n_pairs, size_t n_experts) {
    WvInkPluginDevState & st = g_devs[dev_id];
    if (n_pairs <= st.pair_cap && n_experts <= st.exp_cap &&
        st.d_pair_e && st.d_exp_cnt && st.d_exp_off &&
        st.d_active_e && st.d_meta_cnt && st.d_tile_exp && st.d_tile_beg && st.d_tile_len) {
        return 0;
    }
    cudaSetDevice(dev_id);
    if (st.d_pair_e)   { cudaFree(st.d_pair_e);   st.d_pair_e   = nullptr; }
    if (st.d_pair_j)   { cudaFree(st.d_pair_j);   st.d_pair_j   = nullptr; }
    if (st.d_pair_t)   { cudaFree(st.d_pair_t);   st.d_pair_t   = nullptr; }
    if (st.d_exp_cnt)  { cudaFree(st.d_exp_cnt);  st.d_exp_cnt  = nullptr; }
    if (st.d_exp_off)  { cudaFree(st.d_exp_off);  st.d_exp_off  = nullptr; }
    if (st.d_active_e) { cudaFree(st.d_active_e); st.d_active_e = nullptr; }
    if (st.d_meta_cnt) { cudaFree(st.d_meta_cnt); st.d_meta_cnt = nullptr; }
    if (st.d_tile_exp) { cudaFree(st.d_tile_exp); st.d_tile_exp = nullptr; }
    if (st.d_tile_beg) { cudaFree(st.d_tile_beg); st.d_tile_beg = nullptr; }
    if (st.d_tile_len) { cudaFree(st.d_tile_len); st.d_tile_len = nullptr; }
    const size_t ecap = std::max<size_t>(std::max<size_t>(n_experts, st.exp_cap), 4096ull);
    const size_t cap  = std::max<size_t>(std::max<size_t>(n_pairs, st.pair_cap), std::max<size_t>(ecap, 4096ull * 8ull));
    if (cudaMalloc(&st.d_pair_e,   cap * sizeof(int32_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&st.d_pair_j,   cap * sizeof(int32_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&st.d_pair_t,   cap * sizeof(int32_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&st.d_exp_cnt,  ecap * sizeof(int32_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&st.d_exp_off,  (ecap + 1) * sizeof(int32_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&st.d_active_e, ecap * sizeof(int32_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&st.d_meta_cnt, 2 * sizeof(int32_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&st.d_tile_exp, cap * sizeof(int32_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&st.d_tile_beg, cap * sizeof(int32_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&st.d_tile_len, cap * sizeof(int32_t)) != cudaSuccess) return -1;
    st.pair_cap = cap;
    st.exp_cap  = ecap;
    return 0;
}

__device__ __forceinline__ uint16_t dev_ld_u16(const void * p) {
    const uint8_t * b = static_cast<const uint8_t *>(p);
    return (uint16_t)((uint32_t)b[0] | ((uint32_t)b[1] << 8));
}

__device__ __forceinline__ uint32_t dev_ld_u32(const void * p) {
    const uint8_t * b = static_cast<const uint8_t *>(p);
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

// Fast 2-byte-aligned 32-bit load for `wv_ink_block_swz8` (`34 B` stride -> `qs + 4*j` is always 2B-aligned when row is 2B-aligned)
__device__ __forceinline__ uint32_t dev_ld_u32_swz8(const void * p) {
    if (((uintptr_t)p & 1u) == 0u) {
        const uint16_t * h = static_cast<const uint16_t *>(p);
        return (uint32_t)__ldg(h) | ((uint32_t)__ldg(h + 1) << 16);
    }
    return dev_ld_u32(p);
}

__device__ __forceinline__ uint64_t dev_ld_u64(const void * p) {
    return (uint64_t)dev_ld_u32(p) | ((uint64_t)dev_ld_u32(static_cast<const uint8_t *>(p) + 4) << 32);
}

__device__ __forceinline__ void dev_mma_m16n8k32_s8(
    int & d0, int & d1, int & d2, int & d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(0), "r"(0), "r"(0), "r"(0));
#else
    const int lane = threadIdx.x & 31;
    const int c_k  = lane & 3;
    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    #pragma unroll
    for (int col = 0; col < 8; ++col) {
        const int src_base = col * 4;
        const int32_t b0_c = (int32_t)__shfl_sync(0xffffffffu, b0, src_base + c_k);
        const int32_t b1_c = (int32_t)__shfl_sync(0xffffffffu, b1, src_base + c_k);
        int32_t dp_hi = __dp4a((int32_t)a2, b1_c, __dp4a((int32_t)a0, b0_c, 0));
        int32_t dp_lo = __dp4a((int32_t)a3, b1_c, __dp4a((int32_t)a1, b0_c, 0));
        dp_hi += __shfl_xor_sync(0xffffffffu, dp_hi, 1);
        dp_hi += __shfl_xor_sync(0xffffffffu, dp_hi, 2);
        dp_lo += __shfl_xor_sync(0xffffffffu, dp_lo, 1);
        dp_lo += __shfl_xor_sync(0xffffffffu, dp_lo, 2);
        if (col == c_k * 2 + 0) { s0 = dp_hi; s2 = dp_lo; }
        if (col == c_k * 2 + 1) { s1 = dp_hi; s3 = dp_lo; }
    }
    d0 = s0; d1 = s1; d2 = s2; d3 = s3;
#endif
}

// -----------------------------------------------------------------------------
// 1. Strided F32 -> FWHT-128 + Q8_32 Activation Quantizer (0 SMEM, Warp Shuffle)
// -----------------------------------------------------------------------------
__global__ void wv_ink_plugin_fwht128_q8_kernel(
    const uint8_t * __restrict__ vy_bytes,
    int8_t * __restrict__ Y_q8,
    float * __restrict__ Y_sc,
    int K,
    int ne11,
    int ne12,
    size_t nb11,
    size_t nb12,
    size_t nb13,
    int total_vecs) {
    const int n_b128 = K >> 7;
    const int blk_id = blockIdx.x * (int)blockDim.y + (int)threadIdx.y;
    const int total_blks = total_vecs * n_b128;
    if (blk_id >= total_blks) return;

    const int lane   = threadIdx.x; // 0..31
    const int vec_id = blk_id / n_b128;
    const int b128   = blk_id - vec_id * n_b128;

    const int i11 = vec_id % ne11;
    const int t12 = vec_id / ne11;
    const int i12 = t12 % ne12;
    const int i13 = t12 / ne12;

    const float * y_row = reinterpret_cast<const float *>(
        vy_bytes + (size_t)i11 * nb11 + (size_t)i12 * nb12 + (size_t)i13 * nb13);
    const float4 v4 = reinterpret_cast<const float4 *>(y_row + (size_t)b128 * 128)[lane];
    float f0 = v4.x, f1 = v4.y, f2 = v4.z, f3 = v4.w;

    {
        const float a0 = f0 + f1, a1 = f0 - f1, a2 = f2 + f3, a3 = f2 - f3;
        f0 = a0 + a2; f1 = a1 + a3; f2 = a0 - a2; f3 = a1 - a3;
    }
    #pragma unroll
    for (int mask = 1; mask <= 16; mask <<= 1) {
        const float p0 = __shfl_xor_sync(0xffffffffu, f0, mask);
        const float p1 = __shfl_xor_sync(0xffffffffu, f1, mask);
        const float p2 = __shfl_xor_sync(0xffffffffu, f2, mask);
        const float p3 = __shfl_xor_sync(0xffffffffu, f3, mask);
        const bool  hi = (lane & mask) != 0;
        f0 = hi ? (p0 - f0) : (f0 + p0);
        f1 = hi ? (p1 - f1) : (f1 + p1);
        f2 = hi ? (p2 - f2) : (f2 + p2);
        f3 = hi ? (p3 - f3) : (f3 + p3);
    }
    f0 *= WV_INK_INV_SQRT128; f1 *= WV_INK_INV_SQRT128;
    f2 *= WV_INK_INV_SQRT128; f3 *= WV_INK_INV_SQRT128;

    float amax = fmaxf(fmaxf(fabsf(f0), fabsf(f1)), fmaxf(fabsf(f2), fabsf(f3)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));

    const float sc     = amax * WV_INK_INV_127;
    const float inv_sc = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
    if ((lane & 7) == 0) {
        Y_sc[(size_t)blk_id * 4 + (lane >> 3)] = sc;
    }
    char4 q4;
    q4.x = (int8_t)__float2int_rn(f0 * inv_sc);
    q4.y = (int8_t)__float2int_rn(f1 * inv_sc);
    q4.z = (int8_t)__float2int_rn(f2 * inv_sc);
    q4.w = (int8_t)__float2int_rn(f3 * inv_sc);
    reinterpret_cast<char4 *>(Y_q8 + (size_t)blk_id * 128)[lane] = q4;
}

// -----------------------------------------------------------------------------
// 2. Shared Device Unpack Helpers (`EXT0`, `EXT1`, `EXT2`, `EXT3`)
// -----------------------------------------------------------------------------

// Reads 1 byte at logical offset `idx` of a Hot VQ12 row (`tag == 0xF`) stored in EXT0 + aux_blob
__device__ __forceinline__ uint8_t hot_vq12_byte(
    const uint8_t * ext0_row, int in_slot_cap, const uint8_t * aux_tail_ptr, int idx) {
    if (idx < (WV_INK_HOT_TAG_OFF - 4)) {
        return __ldg(ext0_row + 4 + idx);
    }
    if (idx < in_slot_cap) {
        return __ldg(ext0_row + 6 + idx);
    }
    return aux_tail_ptr ? __ldg(aux_tail_ptr + (idx - in_slot_cap)) : 0u;
}

__device__ __forceinline__ uint32_t unpack_hot_vq12_idx12_and_ws(
    const uint8_t * ext0_row,
    int b128,
    int lane,
    int K,
    const uint8_t * d_aux_blob,
    size_t aux_bytes,
    float & out_ws) {
    const int row_ext0    = (K >> 6) * 18;
    const int in_slot_cap = row_ext0 - 6;
    const uint32_t aux_off = dev_ld_u32(ext0_row);
    const uint8_t * tail_p = (d_aux_blob && (size_t)aux_off < aux_bytes) ? (d_aux_blob + aux_off) : nullptr;

    const int base = b128 * 53;
    const uint16_t d_sup = (uint16_t)((uint32_t)hot_vq12_byte(ext0_row, in_slot_cap, tail_p, base + 0)
                         | ((uint32_t)hot_vq12_byte(ext0_row, in_slot_cap, tail_p, base + 1) << 8));
    const uint32_t m24   = (uint32_t)hot_vq12_byte(ext0_row, in_slot_cap, tail_p, base + 2)
                         | ((uint32_t)hot_vq12_byte(ext0_row, in_slot_cap, tail_p, base + 3) << 8)
                         | ((uint32_t)hot_vq12_byte(ext0_row, in_slot_cap, tail_p, base + 4) << 16);
    const int sub = lane >> 3;
    const uint32_t u6 = (m24 >> (6 * sub)) & 0x3Fu;
    out_ws = wv_ink_round_fp16_ws(wv_ink_fp16_to_f32(d_sup) * WV_INK_INV_63, u6);

    const int p   = lane >> 1;
    const int off = base + 5 + 3 * p;
    const uint32_t b0 = hot_vq12_byte(ext0_row, in_slot_cap, tail_p, off + 0);
    const uint32_t b1 = hot_vq12_byte(ext0_row, in_slot_cap, tail_p, off + 1);
    const uint32_t b2 = hot_vq12_byte(ext0_row, in_slot_cap, tail_p, off + 2);
    return ((lane & 1) == 0)
        ? (b0 | ((b1 & 0x0Fu) << 8))
        : ((b1 >> 4) | (b2 << 4));
}

__device__ __forceinline__ int32_t unpack_hot_vq12_from_ext0_lane(
    const uint8_t * ext0_row,
    int b128,
    int lane,
    int K,
    const uint8_t * d_aux_blob,
    size_t aux_bytes,
    const uint32_t * s_cb4096,
    float & out_ws) {
    const uint32_t idx12 = unpack_hot_vq12_idx12_and_ws(ext0_row, b128, lane, K, d_aux_blob, aux_bytes, out_ws);
    return (int32_t)s_cb4096[idx12];
}

__device__ __forceinline__ int32_t unpack_hot_vq12_from_ext0_ldg(
    const uint8_t * ext0_row,
    int b128,
    int lane,
    int K,
    const uint8_t * d_aux_blob,
    size_t aux_bytes,
    const uint32_t * d_cb4096,
    float & out_ws) {
    const uint32_t idx12 = unpack_hot_vq12_idx12_and_ws(ext0_row, b128, lane, K, d_aux_blob, aux_bytes, out_ws);
    return (int32_t)__ldg(d_cb4096 + idx12);
}

__device__ __forceinline__ uint32_t unpack_vq12_direct_idx12_and_ws(
    const uint8_t * row_bytes,
    int b128,
    int lane,
    float & out_ws) {
    const uint8_t * bp   = row_bytes + (size_t)b128 * 53;
    const uint16_t d_sup = dev_ld_u16(bp);
    const uint32_t m24   = (uint32_t)bp[2] | ((uint32_t)bp[3] << 8) | ((uint32_t)bp[4] << 16);
    const int sub = lane >> 3;
    const uint32_t u6 = (m24 >> (6 * sub)) & 0x3Fu;
    out_ws = wv_ink_round_fp16_ws(wv_ink_fp16_to_f32(d_sup) * WV_INK_INV_63, u6);

    const int p = lane >> 1;
    const uint32_t b0 = bp[5 + 3 * p + 0];
    const uint32_t b1 = bp[5 + 3 * p + 1];
    const uint32_t b2 = bp[5 + 3 * p + 2];
    return ((lane & 1) == 0)
        ? (b0 | ((b1 & 0x0Fu) << 8))
        : ((b1 >> 4) | (b2 << 4));
}

__device__ __forceinline__ int32_t unpack_vq12_direct_lane(
    const uint8_t * row_bytes,
    int b128,
    int lane,
    const uint32_t * s_cb4096,
    float & out_ws) {
    const uint32_t idx12 = unpack_vq12_direct_idx12_and_ws(row_bytes, b128, lane, out_ws);
    return (int32_t)s_cb4096[idx12];
}

__device__ __forceinline__ int32_t unpack_vq12_direct_ldg(
    const uint8_t * row_bytes,
    int b128,
    int lane,
    const uint32_t * d_cb4096,
    float & out_ws) {
    const uint32_t idx12 = unpack_vq12_direct_idx12_and_ws(row_bytes, b128, lane, out_ws);
    return (int32_t)__ldg(d_cb4096 + idx12);
}

__device__ __forceinline__ int32_t unpack_vq8_sb6_lane(
    const uint8_t * row_bytes,
    int b128,
    int lane,
    const uint32_t s_vq8_uv[2][512],
    float & out_ws) {
    const int blk64  = 2 * b128 + (lane >> 4);
    const int c384   = blk64 / 6;
    const int sub_s  = blk64 - c384 * 6; // 0..5
    const int g_in_s = (lane >> 3) & 1;  // 0..1
    const int g12    = 2 * sub_s + g_in_s;
    const int j8     = (lane & 15) >> 1; // 0..7
    const int is_v   = lane & 1;

    const uint8_t * cp   = row_bytes + (size_t)c384 * 108;
    const uint16_t tag_w = dev_ld_u16(cp + WV_INK_HOT_TAG_OFF);
    uint16_t d_u16 = 0;
    uint32_t u6    = 0;
    uint16_t idx16 = 0;

    if ((tag_w >> 12) == 0x4u) {
        d_u16 = dev_ld_u16(cp);
        if (g12 == 0) {
            u6 = tag_w & 0x3Fu;
        } else if (g12 == 1) {
            u6 = (tag_w >> 6) & 0x3Fu;
        } else {
            const uint64_t sc10 = dev_ld_u64(cp + 2);
            u6 = (uint32_t)((sc10 >> (6 * (g12 - 2))) & 0x3Full);
        }
        const int idx48_pos = 8 * sub_s + j8;
        idx16 = (idx48_pos < 31)
            ? dev_ld_u16(cp + 10 + 2 * idx48_pos)
            : dev_ld_u16(cp + 74 + 2 * (idx48_pos - 31));
    } else {
        const auto * b = reinterpret_cast<const wv_ink_block_vq8 *>(cp);
        d_u16 = (uint16_t)(((b[0].sc_meta >> 12) & 0xFu)
              | (((b[1].sc_meta >> 12) & 0xFu) << 4)
              | (((b[2].sc_meta >> 12) & 0xFu) << 8)
              | (((b[3].sc_meta >> 12) & 0xFu) << 12));
        const uint16_t sm = b[sub_s].sc_meta;
        u6    = g_in_s ? ((sm >> 6) & 0x3Fu) : (sm & 0x3Fu);
        idx16 = dev_ld_u16(b[sub_s].qs + 2 * j8);
    }

    out_ws = wv_ink_round_fp16_ws(wv_ink_fp16_to_f32(d_u16) * WV_INK_INV_63, u6);
    const uint32_t c_hi = ((uint32_t)idx16 >> 14) << 7;
    const uint32_t sub7 = is_v ? ((uint32_t)idx16 & 0x7Fu) : (((uint32_t)idx16 >> 7) & 0x7Fu);
    return (int32_t)s_vq8_uv[is_v][c_hi | sub7];
}

__device__ __forceinline__ int32_t prmt16_lut4_dev(uint4 lut16, uint32_t nib4) {
    const uint32_t lo3  = nib4 & 0x07070707u;
    const uint32_t s16  = (lo3 & 0x7u) | ((lo3 >> 4) & 0x70u) | ((lo3 >> 8) & 0x700u) | ((lo3 >> 12) & 0x7000u);
    const uint32_t v_lo = __byte_perm(lut16.x, lut16.y, s16);
    const uint32_t v_hi = __byte_perm(lut16.z, lut16.w, s16);
    const uint32_t b3   = (nib4 & 0x08080808u) >> 3;
    const uint32_t mask = (b3 << 8) - b3;
    return (int32_t)((v_lo & ~mask) | (v_hi & mask));
}

__device__ __forceinline__ int32_t unpack_4b38_lane(
    const uint8_t * row_bytes,
    int b128,
    int lane,
    const uint4 * d_lut4,
    float & out_ws) {
    const uint8_t * bp   = row_bytes + (size_t)b128 * 70;
    const uint16_t d_sup = dev_ld_u16(bp);
    const uint32_t m     = dev_ld_u32(bp + 2);
    const int sub        = lane >> 3;
    const uint32_t u6    = (m >> (6 * sub)) & 0x3Fu;
    out_ws = wv_ink_fp16_to_f32(d_sup) * (WV_INK_U6_NORM * (float)u6);

    const uint4 lut16 = d_lut4[(m >> 24) & 7u];
    const int pos0    = 4 * lane;
    const uint8_t * q = bp + 6;
    const uint32_t nib4 = (pos0 < 64)
        ? (dev_ld_u32(q + pos0) & 0x0F0F0F0Fu)
        : ((dev_ld_u32(q + pos0 - 64) >> 4) & 0x0F0F0F0Fu);
    return prmt16_lut4_dev(lut16, nib4);
}

// -----------------------------------------------------------------------------
// 3. `EXT3` (`wv6_swz8_g32`) Fast GEMV (`M == 1`), Multi-GEMV (`2 <= M < 16`), & INT8 TC GEMM (`M >= 16`)
// -----------------------------------------------------------------------------

// 4 warps = 4 rows/CTA, 0 SMEM, exact group-32 int32 dp4a accumulation
__global__ __launch_bounds__(128)
void wv_ink_plugin_ext3_gemv_kernel(
    const uint8_t * __restrict__ vx,
    const int8_t * __restrict__ Y_q8,
    const float * __restrict__ Y_sc,
    uint8_t * __restrict__ dst_bytes,
    int K,
    int N,
    int ne11, int ne12, int ne13,
    int r2, int r3,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb1,  size_t nb2,  size_t nb3) {
    const int lane   = threadIdx.x & 31;
    const int wid    = threadIdx.x >> 5; // 0..3
    const int row    = blockIdx.x * 4 + wid;
    const int vec_id = blockIdx.y;
    if (row >= N) return;

    const int i11 = vec_id % ne11;
    const int t12 = vec_id / ne11;
    const int i12 = t12 % ne12;
    const int i13 = t12 / ne12;
    const int i02 = i12 / r2;
    const int i03 = i13 / r3;

    const auto * w_row = reinterpret_cast<const wv_ink_block_swz8 *>(
        vx + (size_t)row * nb01 + (size_t)i02 * nb02 + (size_t)i03 * nb03);
    const int32_t * y_i32 = reinterpret_cast<const int32_t *>(Y_q8 + (size_t)vec_id * K);
    const float *   y_sc  = Y_sc + (size_t)vec_id * (K >> 5);

    const int sub   = lane >> 3; // 0..3
    const int j4    = lane & 7;  // 0..7
    const int nb128 = K >> 7;
    float total_sum = 0.0f;

    for (int b = 0; b < nb128; ++b) {
        const int gid = 4 * b + sub;
        const auto & bk = w_row[gid];
        const int32_t w4 = (int32_t)dev_ld_u32_swz8(bk.qs + 4 * j4);
        const int32_t y4 = y_i32[b * 32 + lane];
        int32_t dp = __dp4a(w4, y4, 0);
        dp += __shfl_xor_sync(0xffffffffu, dp, 4);
        dp += __shfl_xor_sync(0xffffffffu, dp, 2);
        dp += __shfl_xor_sync(0xffffffffu, dp, 1);

        const float ws = wv_ink_fp16_to_f32(bk.d127) * WV_INK_INV_127;
        float part = (float)dp * (ws * y_sc[gid]);
        part += __shfl_xor_sync(0xffffffffu, part, 16);
        part += __shfl_xor_sync(0xffffffffu, part, 8);
        total_sum += part;
    }

    if (lane == 0) {
        float * out_p = reinterpret_cast<float *>(
            dst_bytes + (size_t)i11 * nb1 + (size_t)i12 * nb2 + (size_t)i13 * nb3);
        out_p[row] = total_sum;
    }
}

// Multi-token (`2 <= total_vecs < 16`, e.g. MTP-8 `M=8` verification!) single-read EXT3 GEMV (`0` SMEM, `0.0-ULP` identical to gemv)
template<int MAX_M>
__global__ __launch_bounds__(128, 4)
void wv_ink_plugin_ext3_multi_gemv_kernel(
    const uint8_t * __restrict__ vx,
    const int8_t * __restrict__ Y_q8,
    const float * __restrict__ Y_sc,
    uint8_t * __restrict__ dst_bytes,
    int K,
    int N,
    int total_vecs,
    int ne11, int ne12,
    size_t nb01,
    size_t nb1, size_t nb2, size_t nb3) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5; // 0..3
    const int row  = blockIdx.x * 4 + wid;
    if (row >= N) return;

    const auto * w_row = reinterpret_cast<const wv_ink_block_swz8 *>(vx + (size_t)row * nb01);
    const int sub   = lane >> 3; // 0..3
    const int j4    = lane & 7;  // 0..7
    const int nb128 = K >> 7;
    const int n_g32 = K >> 5;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (int b = 0; b < nb128; ++b) {
        const int gid   = 4 * b + sub;
        const auto & bk = w_row[gid];
        const int32_t w4 = (int32_t)dev_ld_u32_swz8(bk.qs + 4 * j4);
        const float   ws = wv_ink_fp16_to_f32(bk.d127) * WV_INK_INV_127;

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < total_vecs) {
                const int32_t y4 = reinterpret_cast<const int32_t *>(Y_q8 + (size_t)m * K)[b * 32 + lane];
                int32_t dp = __dp4a(w4, y4, 0);
                dp += __shfl_xor_sync(0xffffffffu, dp, 4);
                dp += __shfl_xor_sync(0xffffffffu, dp, 2);
                dp += __shfl_xor_sync(0xffffffffu, dp, 1);
                float part = (float)dp * (ws * Y_sc[(size_t)m * n_g32 + gid]);
                part += __shfl_xor_sync(0xffffffffu, part, 16);
                part += __shfl_xor_sync(0xffffffffu, part, 8);
                acc[m] += part;
            }
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < total_vecs) {
                const int i11 = m % ne11;
                const int t12 = m / ne11;
                const int i12 = t12 % ne12;
                const int i13 = t12 / ne12;
                float * out_p = reinterpret_cast<float *>(
                    dst_bytes + (size_t)i11 * nb1 + (size_t)i12 * nb2 + (size_t)i13 * nb3);
                out_p[row] = acc[m];
            }
        }
    }
}

// CTA_M = 64, CTA_N = 64, K_TILE = 64 INT8 Tensor Core `mma.m16n8k32` GEMM (`9.5 KiB` static SMEM <= 48 KiB!)
// Dispatched whenever `total_vecs >= 16 && ne02 == 1 && ne03 == 1`
__global__ __launch_bounds__(128, 4)
void wv_ink_plugin_ext3_tc_m64_n64_kernel(
    const uint8_t * __restrict__ vx,
    const int8_t * __restrict__ Y_q8,
    const float * __restrict__ Y_sc,
    uint8_t * __restrict__ dst_bytes,
    int K,
    int N,
    int total_vecs,
    int ne11, int ne12,
    size_t nb01,
    size_t nb1, size_t nb2, size_t nb3) {
    __shared__ uint32_t s_X[64][17]; // 64 x 64 int8 padded by 1 word (4,352 B)
    __shared__ uint32_t s_W[64][17]; // 64 x 64 int8 padded by 1 word (4,352 B)
    __shared__ float    s_Xsc[64][2];
    __shared__ float    s_Wsc[64][2];

    const int tid    = threadIdx.x; // 0..127
    const int wid    = tid >> 5;    // 0..3
    const int lane   = tid & 31;
    const int warp_m = wid >> 1;    // 0..1
    const int warp_n = wid & 1;     // 0..1

    const int m_base = blockIdx.y * 64;
    const int n_base = blockIdx.x * 64;

    float acc[2][4][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            acc[mt][nt][0] = 0.f; acc[mt][nt][1] = 0.f;
            acc[mt][nt][2] = 0.f; acc[mt][nt][3] = 0.f;
        }
    }

    const int r_grp   = lane >> 2; // 0..7
    const int c_k     = lane & 3;  // 0..3
    const int k_tiles = K / 64;

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int g0 = 2 * kt;
        // Cooperative load of s_X [64, 64] + s_Xsc [64, 2] (128 threads: 2 threads per row, 32 bytes each)
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int idx = pass * 128 + tid; // 0..255 -> 64 rows x 4 (16-byte chunks)
            const int lm  = idx >> 2;         // 0..63
            const int q16 = idx & 3;          // 0..3
            const int gm  = m_base + lm;
            if (gm < total_vecs) {
                const uint4 xv = reinterpret_cast<const uint4 *>(Y_q8 + (size_t)gm * K + (size_t)kt * 64)[q16];
                s_X[lm][4 * q16 + 0] = xv.x;
                s_X[lm][4 * q16 + 1] = xv.y;
                s_X[lm][4 * q16 + 2] = xv.z;
                s_X[lm][4 * q16 + 3] = xv.w;
                if (q16 < 2) {
                    s_Xsc[lm][q16] = Y_sc[(size_t)gm * (K >> 5) + g0 + q16];
                }
            } else {
                s_X[lm][4 * q16 + 0] = 0u;
                s_X[lm][4 * q16 + 1] = 0u;
                s_X[lm][4 * q16 + 2] = 0u;
                s_X[lm][4 * q16 + 3] = 0u;
                if (q16 < 2) s_Xsc[lm][q16] = 0.f;
            }
        }

        // Cooperative load of s_W [64, 64] + s_Wsc [64, 2] (128 threads: tid/2 = ln in 0..63, tid&1 = sub in 0..1)
        {
            const int ln  = tid >> 1; // 0..63
            const int sub = tid & 1;  // 0..1 (32 weights = 1 wv_ink_block_swz8)
            const int gn  = n_base + ln;
            if (gn < N) {
                const auto * wr = reinterpret_cast<const wv_ink_block_swz8 *>(vx + (size_t)gn * nb01);
                const auto & bk = wr[g0 + sub];
                s_Wsc[ln][sub] = wv_ink_fp16_to_f32(bk.d127) * WV_INK_INV_127;
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    s_W[ln][8 * sub + w] = dev_ld_u32_swz8(bk.qs + 4 * w);
                }
            } else {
                s_Wsc[ln][sub] = 0.f;
                #pragma unroll
                for (int w = 0; w < 8; ++w) s_W[ln][8 * sub + w] = 0u;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            const int k_w0 = 8 * sub + c_k;
            const int k_w1 = k_w0 + 4;
            uint32_t a_frag[2][4];
            float    x_sc[2][2];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int m_r0 = warp_m * 32 + mt * 16 + r_grp;
                const int m_r1 = m_r0 + 8;
                a_frag[mt][0] = s_X[m_r0][k_w0];
                a_frag[mt][1] = s_X[m_r1][k_w0];
                a_frag[mt][2] = s_X[m_r0][k_w1];
                a_frag[mt][3] = s_X[m_r1][k_w1];
                x_sc[mt][0]   = s_Xsc[m_r0][sub];
                x_sc[mt][1]   = s_Xsc[m_r1][sub];
            }
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int n_col_lane = warp_n * 32 + nt * 8 + r_grp;
                const uint32_t b0 = s_W[n_col_lane][k_w0];
                const uint32_t b1 = s_W[n_col_lane][k_w1];
                const int c_n0 = warp_n * 32 + nt * 8 + 2 * c_k;
                const float ws0 = s_Wsc[c_n0 + 0][sub];
                const float ws1 = s_Wsc[c_n0 + 1][sub];
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    int d0, d1, d2, d3;
                    dev_mma_m16n8k32_s8(
                        d0, d1, d2, d3,
                        a_frag[mt][0], a_frag[mt][1], a_frag[mt][2], a_frag[mt][3],
                        b0, b1);
                    acc[mt][nt][0] += (float)d0 * (x_sc[mt][0] * ws0);
                    acc[mt][nt][1] += (float)d1 * (x_sc[mt][0] * ws1);
                    acc[mt][nt][2] += (float)d2 * (x_sc[mt][1] * ws0);
                    acc[mt][nt][3] += (float)d3 * (x_sc[mt][1] * ws1);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const int gm0 = m_base + warp_m * 32 + mt * 16 + r_grp;
        const int gm1 = gm0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int gn0 = n_base + warp_n * 32 + nt * 8 + 2 * c_k;
            if (gm0 < total_vecs && gn0 < N) {
                const int i11 = gm0 % ne11;
                const int t12 = gm0 / ne11;
                const int i12 = t12 % ne12;
                const int i13 = t12 / ne12;
                float * out0 = reinterpret_cast<float *>(
                    dst_bytes + (size_t)i11 * nb1 + (size_t)i12 * nb2 + (size_t)i13 * nb3);
                out0[gn0] = acc[mt][nt][0];
                if (gn0 + 1 < N) out0[gn0 + 1] = acc[mt][nt][1];
            }
            if (gm1 < total_vecs && gn0 < N) {
                const int i11 = gm1 % ne11;
                const int t12 = gm1 / ne11;
                const int i12 = t12 % ne12;
                const int i13 = t12 / ne12;
                float * out1 = reinterpret_cast<float *>(
                    dst_bytes + (size_t)i11 * nb1 + (size_t)i12 * nb2 + (size_t)i13 * nb3);
                out1[gn0] = acc[mt][nt][2];
                if (gn0 + 1 < N) out1[gn0 + 1] = acc[mt][nt][3];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 4. Generic `mul_mat_cuda` for `EXT0`, `EXT1`, `EXT2`: Small-M GEMV + INT8 TC GEMM (`M >= 16`)
// -----------------------------------------------------------------------------
template<int EXT_SLOT>
__global__ __launch_bounds__(128)
void wv_ink_plugin_ext012_mm_kernel(
    const uint8_t * __restrict__ vx,
    const int8_t * __restrict__ Y_q8,
    const float * __restrict__ Y_sc,
    uint8_t * __restrict__ dst_bytes,
    const uint32_t * __restrict__ d_cb4096,
    const uint32_t * __restrict__ d_vq8_uv,
    const uint4 * __restrict__ d_lut4,
    const uint8_t * __restrict__ d_aux_blob,
    size_t aux_bytes,
    int K,
    int N,
    int ne11, int ne12,
    int r2, int r3,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb1,  size_t nb2,  size_t nb3) {
    __shared__ uint32_t s_cb4096[4096];
    __shared__ uint32_t s_vq8_uv[2][512];

    const int tid = threadIdx.x;
    if constexpr (EXT_SLOT == 0 || EXT_SLOT == 1) {
        for (int i = tid; i < 4096; i += 128) s_cb4096[i] = d_cb4096[i];
    }
    if constexpr (EXT_SLOT == 0) {
        for (int i = tid; i < 1024; i += 128) reinterpret_cast<uint32_t *>(s_vq8_uv)[i] = d_vq8_uv[i];
    }
    __syncthreads();

    const int lane   = tid & 31;
    const int wid    = tid >> 5;
    const int row    = blockIdx.x * 4 + wid;
    const int vec_id = blockIdx.y;
    if (row >= N) return;

    const int i11 = vec_id % ne11;
    const int t12 = vec_id / ne11;
    const int i12 = t12 % ne12;
    const int i13 = t12 / ne12;
    const int i02 = i12 / r2;
    const int i03 = i13 / r3;

    const uint8_t * row_p = vx + (size_t)row * nb01 + (size_t)i02 * nb02 + (size_t)i03 * nb03;
    const int32_t * y_i32 = reinterpret_cast<const int32_t *>(Y_q8 + (size_t)vec_id * K);
    const float *   y_sc  = Y_sc + (size_t)vec_id * (K >> 5);
    const int sub   = lane >> 3;
    const int nb128 = K >> 7;
    const bool is_hot = (EXT_SLOT == 0) && ((dev_ld_u16(row_p + WV_INK_HOT_TAG_OFF) >> 12) == 0xFu);

    float total_sum = 0.0f;
    for (int b = 0; b < nb128; ++b) {
        float ws = 0.0f;
        int32_t w4 = 0;
        if constexpr (EXT_SLOT == 0) {
            w4 = is_hot
                ? unpack_hot_vq12_from_ext0_lane(row_p, b, lane, K, d_aux_blob, aux_bytes, s_cb4096, ws)
                : unpack_vq8_sb6_lane(row_p, b, lane, s_vq8_uv, ws);
        } else if constexpr (EXT_SLOT == 1) {
            w4 = unpack_vq12_direct_lane(row_p, b, lane, s_cb4096, ws);
        } else {
            w4 = unpack_4b38_lane(row_p, b, lane, d_lut4, ws);
        }
        const int32_t y4 = y_i32[b * 32 + lane];
        int32_t dp = __dp4a(w4, y4, 0);
        dp += __shfl_xor_sync(0xffffffffu, dp, 4);
        dp += __shfl_xor_sync(0xffffffffu, dp, 2);
        dp += __shfl_xor_sync(0xffffffffu, dp, 1);
        float part = (float)dp * (ws * y_sc[4 * b + sub]);
        part += __shfl_xor_sync(0xffffffffu, part, 16);
        part += __shfl_xor_sync(0xffffffffu, part, 8);
        total_sum += part;
    }

    if (lane == 0) {
        float * out_p = reinterpret_cast<float *>(
            dst_bytes + (size_t)i11 * nb1 + (size_t)i12 * nb2 + (size_t)i13 * nb3);
        out_p[row] = total_sum;
    }
}

// CTA_M = 64, CTA_N = 64, K_TILE = 128 INT8 Tensor Core `mma.m16n8k32` GEMM for `EXT0`, `EXT1`, `EXT2`
// Static SMEM = `23,040 B` (`22.50 KiB <= 24 KiB`, `2+ CTAs/SM` on all `sm_80..sm_120a`, `0` spills!)
template<int EXT_SLOT>
__global__ __launch_bounds__(256, 2)
void wv_ink_plugin_ext012_tc_m64_n64_kernel(
    const uint8_t * __restrict__ vx,
    const int8_t * __restrict__ Y_q8,
    const float * __restrict__ Y_sc,
    uint8_t * __restrict__ dst_bytes,
    const uint32_t * __restrict__ d_cb4096,
    const uint32_t * __restrict__ d_vq8_uv,
    const uint4 * __restrict__ d_lut4,
    const uint8_t * __restrict__ d_aux_blob,
    size_t aux_bytes,
    int K,
    int N,
    int total_vecs,
    int ne11, int ne12,
    size_t nb01,
    size_t nb1, size_t nb2, size_t nb3) {
    __shared__ uint32_t s_X[64][33];      // 8,448 B
    __shared__ uint32_t s_W[64][33];      // 8,448 B
    __shared__ float    s_Xsc[64][4];     // 1,024 B
    __shared__ float    s_Wsc[64][4];     // 1,024 B
    __shared__ uint32_t s_vq8_uv[2][512]; // 4,096 B

    const int tid    = threadIdx.x; // 0..255
    const int wid    = tid >> 5;    // 0..7
    const int lane   = tid & 31;    // 0..31
    const int warp_m = wid >> 2;    // 0..1 (32 M-rows/warp)
    const int warp_n = wid & 3;     // 0..3 (16 N-cols/warp)

    if constexpr (EXT_SLOT == 0) {
        #pragma unroll
        for (int i = tid; i < 1024; i += 256) {
            reinterpret_cast<uint32_t *>(s_vq8_uv)[i] = d_vq8_uv[i];
        }
        __syncthreads();
    }

    const int m_base = blockIdx.y * 64;
    const int n_base = blockIdx.x * 64;

    uint32_t is_hot_mask = 0u;
    if constexpr (EXT_SLOT == 0) {
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int gn = n_base + wid * 8 + r;
            if (gn < N) {
                const uint8_t * row_p = vx + (size_t)gn * nb01;
                if ((dev_ld_u16(row_p + WV_INK_HOT_TAG_OFF) >> 12) == 0xFu) {
                    is_hot_mask |= (1u << r);
                }
            }
        }
    }

    float acc[2][2][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            acc[mt][nt][0] = 0.f; acc[mt][nt][1] = 0.f;
            acc[mt][nt][2] = 0.f; acc[mt][nt][3] = 0.f;
        }
    }

    const int r_grp = lane >> 2; // 0..7
    const int c_k   = lane & 3;  // 0..3
    const int nb128 = K >> 7;
    const int n_g32 = K >> 5;

    for (int b = 0; b < nb128; ++b) {
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int idx = pass * 256 + tid; // 0..511 -> 64 rows x 8 uint4s
            const int lm  = idx >> 3;         // 0..63
            const int q16 = idx & 7;          // 0..7
            const int gm  = m_base + lm;
            if (gm < total_vecs) {
                const uint4 xv = reinterpret_cast<const uint4 *>(Y_q8 + (size_t)gm * K + (size_t)b * 128)[q16];
                s_X[lm][4 * q16 + 0] = xv.x;
                s_X[lm][4 * q16 + 1] = xv.y;
                s_X[lm][4 * q16 + 2] = xv.z;
                s_X[lm][4 * q16 + 3] = xv.w;
                if (q16 < 4) {
                    s_Xsc[lm][q16] = Y_sc[(size_t)gm * n_g32 + 4 * b + q16];
                }
            } else {
                s_X[lm][4 * q16 + 0] = 0u;
                s_X[lm][4 * q16 + 1] = 0u;
                s_X[lm][4 * q16 + 2] = 0u;
                s_X[lm][4 * q16 + 3] = 0u;
                if (q16 < 4) s_Xsc[lm][q16] = 0.f;
            }
        }

        #pragma unroll 2
        for (int r = 0; r < 8; ++r) {
            const int ln = wid * 8 + r;
            const int gn = n_base + ln;
            if (gn < N) {
                const uint8_t * row_p = vx + (size_t)gn * nb01;
                float ws = 0.0f;
                int32_t w4 = 0;
                if constexpr (EXT_SLOT == 0) {
                    w4 = ((is_hot_mask >> r) & 1u)
                        ? unpack_hot_vq12_from_ext0_ldg(row_p, b, lane, K, d_aux_blob, aux_bytes, d_cb4096, ws)
                        : unpack_vq8_sb6_lane(row_p, b, lane, s_vq8_uv, ws);
                } else if constexpr (EXT_SLOT == 1) {
                    w4 = unpack_vq12_direct_ldg(row_p, b, lane, d_cb4096, ws);
                } else {
                    w4 = unpack_4b38_lane(row_p, b, lane, d_lut4, ws);
                }
                s_W[ln][lane] = (uint32_t)w4;
                if ((lane & 7) == 0) s_Wsc[ln][lane >> 3] = ws;
            } else {
                s_W[ln][lane] = 0u;
                if ((lane & 7) == 0) s_Wsc[ln][lane >> 3] = 0.f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const int k_w0 = 8 * sub + c_k;
            const int k_w1 = k_w0 + 4;
            uint32_t a_frag[2][4];
            float    x_sc[2][2];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int m_r0 = warp_m * 32 + mt * 16 + r_grp;
                const int m_r1 = m_r0 + 8;
                a_frag[mt][0] = s_X[m_r0][k_w0];
                a_frag[mt][1] = s_X[m_r1][k_w0];
                a_frag[mt][2] = s_X[m_r0][k_w1];
                a_frag[mt][3] = s_X[m_r1][k_w1];
                x_sc[mt][0]   = s_Xsc[m_r0][sub];
                x_sc[mt][1]   = s_Xsc[m_r1][sub];
            }
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int n_col_lane = warp_n * 16 + nt * 8 + r_grp;
                const uint32_t b0 = s_W[n_col_lane][k_w0];
                const uint32_t b1 = s_W[n_col_lane][k_w1];
                const int c_n0 = warp_n * 16 + nt * 8 + 2 * c_k;
                const float ws0 = s_Wsc[c_n0 + 0][sub];
                const float ws1 = s_Wsc[c_n0 + 1][sub];
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    int d0, d1, d2, d3;
                    dev_mma_m16n8k32_s8(
                        d0, d1, d2, d3,
                        a_frag[mt][0], a_frag[mt][1], a_frag[mt][2], a_frag[mt][3],
                        b0, b1);
                    acc[mt][nt][0] += (float)d0 * (x_sc[mt][0] * ws0);
                    acc[mt][nt][1] += (float)d1 * (x_sc[mt][0] * ws1);
                    acc[mt][nt][2] += (float)d2 * (x_sc[mt][1] * ws0);
                    acc[mt][nt][3] += (float)d3 * (x_sc[mt][1] * ws1);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const int gm0 = m_base + warp_m * 32 + mt * 16 + r_grp;
        const int gm1 = gm0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int gn0 = n_base + warp_n * 16 + nt * 8 + 2 * c_k;
            if (gm0 < total_vecs && gn0 < N) {
                const int i11 = gm0 % ne11;
                const int t12 = gm0 / ne11;
                const int i12 = t12 % ne12;
                const int i13 = t12 / ne12;
                float * out0 = reinterpret_cast<float *>(
                    dst_bytes + (size_t)i11 * nb1 + (size_t)i12 * nb2 + (size_t)i13 * nb3);
                out0[gn0] = acc[mt][nt][0];
                if (gn0 + 1 < N) out0[gn0 + 1] = acc[mt][nt][1];
            }
            if (gm1 < total_vecs && gn0 < N) {
                const int i11 = gm1 % ne11;
                const int t12 = gm1 / ne11;
                const int i12 = t12 % ne12;
                const int i13 = t12 / ne12;
                float * out1 = reinterpret_cast<float *>(
                    dst_bytes + (size_t)i11 * nb1 + (size_t)i12 * nb2 + (size_t)i13 * nb3);
                out1[gn0] = acc[mt][nt][2];
                if (gn0 + 1 < N) out1[gn0 + 1] = acc[mt][nt][3];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 5. `mul_mat_id_cuda` (`EXT0` & `EXT1`): Decode (`M < 4`), Compact Grouped8 (`4 <= M < 16`), & INT8 TC (`M >= 16`)!
// -----------------------------------------------------------------------------

__global__ void wv_ink_plugin_mmid_zero_and_hist_kernel(
    const uint8_t * __restrict__ ids_bytes,
    uint8_t * __restrict__ dst_bytes,
    int32_t * __restrict__ exp_cnt,
    int N,
    int n_expert,
    int n_expert_used,
    int ne12,
    size_t ids_nb0,
    size_t ids_nb1,
    size_t nb1,
    size_t nb2,
    int32_t first_zero) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = n_expert_used * ne12;
    if (tid >= total_pairs) return;
    const int j = tid % n_expert_used;
    const int t = tid / n_expert_used;
    const int32_t e = *reinterpret_cast<const int32_t *>(ids_bytes + (size_t)j * ids_nb0 + (size_t)t * ids_nb1);
    if (e < 0 || e >= n_expert || (first_zero >= 0 && e >= first_zero)) {
        float * out_p = reinterpret_cast<float *>(dst_bytes + (size_t)j * nb1 + (size_t)t * nb2);
        for (int r = 0; r < N; ++r) out_p[r] = 0.0f;
        return;
    }
    atomicAdd(&exp_cnt[e], 1);
}

// Parallel 256-thread prefix scan + active-expert compaction + 64-token tile builder (`0` host sync, arbitrary `n_expert <= 65536`)
__global__ __launch_bounds__(256)
void wv_ink_plugin_mmid_plan_kernel(
    const int32_t * __restrict__ exp_cnt,
    int32_t * __restrict__ exp_off,
    int32_t * __restrict__ exp_cur,
    int32_t * __restrict__ active_e,
    int32_t * __restrict__ tile_exp,
    int32_t * __restrict__ tile_beg,
    int32_t * __restrict__ tile_len,
    int32_t * __restrict__ meta_cnt,
    int n_expert,
    int build_tiles64) {
    __shared__ int32_t s_p[256];
    __shared__ int32_t s_a[256];
    __shared__ int32_t s_t[256];
    __shared__ int32_t s_base_p;
    __shared__ int32_t s_base_a;
    __shared__ int32_t s_base_t;

    const int tid = threadIdx.x;
    if (tid == 0) {
        s_base_p = 0;
        s_base_a = 0;
        s_base_t = 0;
    }
    __syncthreads();

    for (int base = 0; base < n_expert; base += 256) {
        const int e = base + tid;
        const int32_t c  = (e < n_expert) ? exp_cnt[e] : 0;
        const int32_t a  = (c > 0) ? 1 : 0;
        const int32_t nt = (build_tiles64 && c > 0) ? ((c + 63) >> 6) : 0;

        s_p[tid] = c;
        s_a[tid] = a;
        s_t[tid] = nt;
        __syncthreads();

        #pragma unroll
        for (int offset = 1; offset < 256; offset <<= 1) {
            int32_t vp = 0, va = 0, vt = 0;
            if (tid >= offset) {
                vp = s_p[tid - offset];
                va = s_a[tid - offset];
                vt = s_t[tid - offset];
            }
            __syncthreads();
            if (tid >= offset) {
                s_p[tid] += vp;
                s_a[tid] += va;
                s_t[tid] += vt;
            }
            __syncthreads();
        }

        const int32_t off_p = s_base_p + ((tid > 0) ? s_p[tid - 1] : 0);
        const int32_t off_a = s_base_a + ((tid > 0) ? s_a[tid - 1] : 0);
        const int32_t off_t = s_base_t + ((tid > 0) ? s_t[tid - 1] : 0);

        if (e < n_expert) {
            exp_off[e] = off_p;
            exp_cur[e] = off_p;
            if (a) {
                active_e[off_a] = e;
            }
            for (int k = 0; k < nt; ++k) {
                tile_exp[off_t + k] = e;
                tile_beg[off_t + k] = off_p + k * 64;
                const int32_t rem = c - k * 64;
                tile_len[off_t + k] = (rem < 64) ? rem : 64;
            }
        }

        const int32_t chunk_p = s_p[255];
        const int32_t chunk_a = s_a[255];
        const int32_t chunk_t = s_t[255];
        __syncthreads();
        if (tid == 0) {
            s_base_p += chunk_p;
            s_base_a += chunk_a;
            s_base_t += chunk_t;
        }
        __syncthreads();
    }

    if (tid == 0) {
        exp_off[n_expert] = s_base_p;
        meta_cnt[0]       = s_base_a;
        meta_cnt[1]       = s_base_t;
    }
}

__global__ void wv_ink_plugin_mmid_scatter_kernel(
    const uint8_t * __restrict__ ids_bytes,
    int32_t * __restrict__ exp_cur,
    int32_t * __restrict__ pair_j,
    int32_t * __restrict__ pair_t,
    int n_expert,
    int n_expert_used,
    int ne12,
    size_t ids_nb0,
    size_t ids_nb1,
    int32_t first_zero) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = n_expert_used * ne12;
    if (tid >= total_pairs) return;
    const int j = tid % n_expert_used;
    const int t = tid / n_expert_used;
    const int32_t e = *reinterpret_cast<const int32_t *>(ids_bytes + (size_t)j * ids_nb0 + (size_t)t * ids_nb1);
    if (e < 0 || e >= n_expert || (first_zero >= 0 && e >= first_zero)) return;
    const int pos = atomicAdd(&exp_cur[e], 1);
    pair_j[pos] = j;
    pair_t[pos] = t;
}

// Small-M (`ne12 < 4`) direct `mul_mat_id_cuda` (`8` warps = `8` rows/CTA, `20 KiB` SMEM)
template<int EXT_SLOT>
__global__ __launch_bounds__(256, 2)
void wv_ink_plugin_mmid_smallm_kernel(
    const uint8_t * __restrict__ vx,
    const int8_t * __restrict__ Y_q8,
    const float * __restrict__ Y_sc,
    const uint8_t * __restrict__ ids_bytes,
    uint8_t * __restrict__ dst_bytes,
    const uint32_t * __restrict__ d_cb4096,
    const uint32_t * __restrict__ d_vq8_uv,
    const uint8_t * __restrict__ d_aux_blob,
    size_t aux_bytes,
    int K,
    int N,
    int n_expert,
    int ne11,
    int n_expert_used,
    size_t nb01, size_t nb02,
    size_t ids_nb0, size_t ids_nb1,
    size_t nb1, size_t nb2,
    int32_t first_zero) {
    __shared__ uint32_t s_cb4096[4096];
    __shared__ uint32_t s_vq8_uv[2][512];

    const int pair_id = blockIdx.y;
    const int j       = pair_id % n_expert_used;
    const int t       = pair_id / n_expert_used;
    const int32_t e   = *reinterpret_cast<const int32_t *>(ids_bytes + (size_t)j * ids_nb0 + (size_t)t * ids_nb1);
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int wid     = tid >> 5; // 0..7
    const int row     = blockIdx.x * 8 + wid;

    if (e < 0 || e >= n_expert || (first_zero >= 0 && e >= first_zero)) {
        if (row < N && lane == 0) {
            float * out_p = reinterpret_cast<float *>(dst_bytes + (size_t)j * nb1 + (size_t)t * nb2);
            out_p[row] = 0.0f;
        }
        return;
    }

    for (int i = tid; i < 4096; i += 256) s_cb4096[i] = d_cb4096[i];
    if constexpr (EXT_SLOT == 0) {
        for (int i = tid; i < 1024; i += 256) reinterpret_cast<uint32_t *>(s_vq8_uv)[i] = d_vq8_uv[i];
    }
    __syncthreads();

    if (row >= N) return;

    const int y_col = (ne11 == 1) ? 0 : j;
    const int vec_id = t * ne11 + y_col;
    const uint8_t * row_p = vx + (size_t)e * nb02 + (size_t)row * nb01;
    const int32_t * y_i32 = reinterpret_cast<const int32_t *>(Y_q8 + (size_t)vec_id * K);
    const float *   y_sc  = Y_sc + (size_t)vec_id * (K >> 5);
    const int sub   = lane >> 3;
    const int nb128 = K >> 7;
    const bool is_hot = (EXT_SLOT == 0) && ((dev_ld_u16(row_p + WV_INK_HOT_TAG_OFF) >> 12) == 0xFu);

    float total_sum = 0.0f;
    for (int b = 0; b < nb128; ++b) {
        float ws = 0.0f;
        int32_t w4 = 0;
        if constexpr (EXT_SLOT == 0) {
            w4 = is_hot
                ? unpack_hot_vq12_from_ext0_lane(row_p, b, lane, K, d_aux_blob, aux_bytes, s_cb4096, ws)
                : unpack_vq8_sb6_lane(row_p, b, lane, s_vq8_uv, ws);
        } else {
            w4 = unpack_vq12_direct_lane(row_p, b, lane, s_cb4096, ws);
        }
        const int32_t y4 = y_i32[b * 32 + lane];
        int32_t dp = __dp4a(w4, y4, 0);
        dp += __shfl_xor_sync(0xffffffffu, dp, 4);
        dp += __shfl_xor_sync(0xffffffffu, dp, 2);
        dp += __shfl_xor_sync(0xffffffffu, dp, 1);
        float part = (float)dp * (ws * y_sc[4 * b + sub]);
        part += __shfl_xor_sync(0xffffffffu, part, 16);
        part += __shfl_xor_sync(0xffffffffu, part, 8);
        total_sum += part;
    }

    if (lane == 0) {
        float * out_p = reinterpret_cast<float *>(dst_bytes + (size_t)j * nb1 + (size_t)t * nb2);
        out_p[row] = total_sum;
    }
}

// Compact Active-Expert Grouped8 (`4 <= ne12 < 16`, e.g. MTP-8!): launches ONLY over active experts (`meta_cnt[0]`)!
template<int EXT_SLOT>
__global__ __launch_bounds__(256, 2)
void wv_ink_plugin_mmid_grouped8_kernel(
    const uint8_t * __restrict__ vx,
    const int8_t * __restrict__ Y_q8,
    const float * __restrict__ Y_sc,
    const int32_t * __restrict__ active_e,
    const int32_t * __restrict__ meta_cnt,
    const int32_t * __restrict__ exp_off,
    const int32_t * __restrict__ pair_j,
    const int32_t * __restrict__ pair_t,
    uint8_t * __restrict__ dst_bytes,
    const uint32_t * __restrict__ d_cb4096,
    const uint32_t * __restrict__ d_vq8_uv,
    const uint8_t * __restrict__ d_aux_blob,
    size_t aux_bytes,
    int K,
    int N,
    int ne11,
    size_t nb01, size_t nb02,
    size_t nb1,  size_t nb2) {
    __shared__ uint32_t s_cb4096[4096];
    __shared__ uint32_t s_vq8_uv[2][512];

    const int num_active = meta_cnt[0];
    if ((int)blockIdx.y >= num_active) return;

    const int tid = threadIdx.x;
    for (int i = tid; i < 4096; i += 256) s_cb4096[i] = d_cb4096[i];
    if constexpr (EXT_SLOT == 0) {
        for (int i = tid; i < 1024; i += 256) reinterpret_cast<uint32_t *>(s_vq8_uv)[i] = d_vq8_uv[i];
    }
    __syncthreads();

    const int lane   = tid & 31;
    const int wid    = tid >> 5; // 0..7
    const int row    = blockIdx.x * 8 + wid;
    if (row >= N) return;

    const int sub   = lane >> 3;
    const int nb128 = K >> 7;

    for (int ai = (int)blockIdx.y; ai < num_active; ai += (int)gridDim.y) {
        const int e  = active_e[ai];
        const int p0 = exp_off[e];
        const int p1 = exp_off[e + 1];
        const uint8_t * row_p = vx + (size_t)e * nb02 + (size_t)row * nb01;
        const bool is_hot = (EXT_SLOT == 0) && ((dev_ld_u16(row_p + WV_INK_HOT_TAG_OFF) >> 12) == 0xFu);

        for (int pb = p0; pb < p1; pb += 8) {
            const int cnt = min(8, p1 - pb);
            int vec_ids[8];
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                if (m < cnt) {
                    const int j = pair_j[pb + m];
                    const int t = pair_t[pb + m];
                    vec_ids[m] = t * ne11 + ((ne11 == 1) ? 0 : j);
                } else {
                    vec_ids[m] = 0;
                }
            }

            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int b = 0; b < nb128; ++b) {
                float ws = 0.0f;
                int32_t w4 = 0;
                if constexpr (EXT_SLOT == 0) {
                    w4 = is_hot
                        ? unpack_hot_vq12_from_ext0_lane(row_p, b, lane, K, d_aux_blob, aux_bytes, s_cb4096, ws)
                        : unpack_vq8_sb6_lane(row_p, b, lane, s_vq8_uv, ws);
                } else {
                    w4 = unpack_vq12_direct_lane(row_p, b, lane, s_cb4096, ws);
                }

                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    if (m < cnt) {
                        const int vid = vec_ids[m];
                        const int32_t y4 = reinterpret_cast<const int32_t *>(Y_q8 + (size_t)vid * K)[b * 32 + lane];
                        int32_t dp = __dp4a(w4, y4, 0);
                        dp += __shfl_xor_sync(0xffffffffu, dp, 4);
                        dp += __shfl_xor_sync(0xffffffffu, dp, 2);
                        dp += __shfl_xor_sync(0xffffffffu, dp, 1);
                        const float ysc = Y_sc[(size_t)vid * (K >> 5) + 4 * b + sub];
                        float part = (float)dp * (ws * ysc);
                        part += __shfl_xor_sync(0xffffffffu, part, 16);
                        part += __shfl_xor_sync(0xffffffffu, part, 8);
                        acc[m] += part;
                    }
                }
            }

            if (lane == 0) {
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    if (m < cnt) {
                        const int j = pair_j[pb + m];
                        const int t = pair_t[pb + m];
                        float * out_p = reinterpret_cast<float *>(dst_bytes + (size_t)j * nb1 + (size_t)t * nb2);
                        out_p[row] = acc[m];
                    }
                }
            }
        }
    }
}

// Grouped INT8 Tensor Core `mma.m16n8k32` `mul_mat_id_cuda` (`ne12 >= 16`)
// CTA_M = 64 (routed pairs for expert `e`), CTA_N = 64 (expert rows), K_TILE = 128 (`256` threads = `8` warps)
// Static SMEM = `23,808 B` (`23.25 KiB <= 24 KiB`, `2+ CTAs/SM` on all `sm_80..sm_120a`, `0` spills, `0` host sync!)
template<int EXT_SLOT>
__global__ __launch_bounds__(256, 2)
void wv_ink_plugin_mmid_tc_m64_n64_kernel(
    const uint8_t * __restrict__ vx,
    const int8_t * __restrict__ Y_q8,
    const float * __restrict__ Y_sc,
    const int32_t * __restrict__ tile_exp,
    const int32_t * __restrict__ tile_beg,
    const int32_t * __restrict__ tile_len,
    const int32_t * __restrict__ meta_cnt,
    const int32_t * __restrict__ pair_j,
    const int32_t * __restrict__ pair_t,
    uint8_t * __restrict__ dst_bytes,
    const uint32_t * __restrict__ d_cb4096,
    const uint32_t * __restrict__ d_vq8_uv,
    const uint8_t * __restrict__ d_aux_blob,
    size_t aux_bytes,
    int K,
    int N,
    int ne11,
    size_t nb01, size_t nb02,
    size_t nb1,  size_t nb2) {
    __shared__ uint32_t s_X[64][33];      // 8,448 B
    __shared__ uint32_t s_W[64][33];      // 8,448 B
    __shared__ float    s_Xsc[64][4];     // 1,024 B
    __shared__ float    s_Wsc[64][4];     // 1,024 B
    __shared__ uint32_t s_vq8_uv[2][512]; // 4,096 B
    __shared__ int32_t  s_vid[64];        //   256 B
    __shared__ int32_t  s_pair_j[64];     //   256 B
    __shared__ int32_t  s_pair_t[64];     //   256 B

    const int num_tiles = meta_cnt[1];
    if ((int)blockIdx.y >= num_tiles) return;

    const int tid    = threadIdx.x; // 0..255
    const int wid    = tid >> 5;    // 0..7
    const int lane   = tid & 31;    // 0..31
    const int warp_m = wid >> 2;    // 0..1 (32 M-rows/warp)
    const int warp_n = wid & 3;     // 0..3 (16 N-cols/warp)
    const int n_base = blockIdx.x * 64;

    if constexpr (EXT_SLOT == 0) {
        #pragma unroll
        for (int i = tid; i < 1024; i += 256) {
            reinterpret_cast<uint32_t *>(s_vq8_uv)[i] = d_vq8_uv[i];
        }
    }

    const int r_grp = lane >> 2; // 0..7
    const int c_k   = lane & 3;  // 0..3
    const int nb128 = K >> 7;
    const int n_g32 = K >> 5;

    for (int tile_id = (int)blockIdx.y; tile_id < num_tiles; tile_id += (int)gridDim.y) {
        const int e     = tile_exp[tile_id];
        const int p_beg = tile_beg[tile_id];
        const int t_cnt = tile_len[tile_id];

        if (tid < 64) {
            if (tid < t_cnt) {
                const int j = pair_j[p_beg + tid];
                const int t = pair_t[p_beg + tid];
                s_pair_j[tid] = j;
                s_pair_t[tid] = t;
                s_vid[tid]    = t * ne11 + ((ne11 == 1) ? 0 : j);
            } else {
                s_pair_j[tid] = 0;
                s_pair_t[tid] = 0;
                s_vid[tid]    = -1;
            }
        }
        __syncthreads();

        uint32_t is_hot_mask = 0u;
        if constexpr (EXT_SLOT == 0) {
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int gn = n_base + wid * 8 + r;
                if (gn < N) {
                    const uint8_t * row_p = vx + (size_t)e * nb02 + (size_t)gn * nb01;
                    if ((dev_ld_u16(row_p + WV_INK_HOT_TAG_OFF) >> 12) == 0xFu) {
                        is_hot_mask |= (1u << r);
                    }
                }
            }
        }

        float acc[2][2][4];
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                acc[mt][nt][0] = 0.f; acc[mt][nt][1] = 0.f;
                acc[mt][nt][2] = 0.f; acc[mt][nt][3] = 0.f;
            }
        }

        for (int b = 0; b < nb128; ++b) {
            #pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int idx = pass * 256 + tid; // 0..511 -> 64 rows x 8 uint4s
                const int lm  = idx >> 3;         // 0..63
                const int q16 = idx & 7;          // 0..7
                const int vid = s_vid[lm];
                if (vid >= 0) {
                    const uint4 xv = reinterpret_cast<const uint4 *>(Y_q8 + (size_t)vid * K + (size_t)b * 128)[q16];
                    s_X[lm][4 * q16 + 0] = xv.x;
                    s_X[lm][4 * q16 + 1] = xv.y;
                    s_X[lm][4 * q16 + 2] = xv.z;
                    s_X[lm][4 * q16 + 3] = xv.w;
                    if (q16 < 4) {
                        s_Xsc[lm][q16] = Y_sc[(size_t)vid * n_g32 + 4 * b + q16];
                    }
                } else {
                    s_X[lm][4 * q16 + 0] = 0u;
                    s_X[lm][4 * q16 + 1] = 0u;
                    s_X[lm][4 * q16 + 2] = 0u;
                    s_X[lm][4 * q16 + 3] = 0u;
                    if (q16 < 4) s_Xsc[lm][q16] = 0.f;
                }
            }

            #pragma unroll 2
            for (int r = 0; r < 8; ++r) {
                const int ln = wid * 8 + r;
                const int gn = n_base + ln;
                if (gn < N) {
                    const uint8_t * row_p = vx + (size_t)e * nb02 + (size_t)gn * nb01;
                    float ws = 0.0f;
                    int32_t w4 = 0;
                    if constexpr (EXT_SLOT == 0) {
                        w4 = ((is_hot_mask >> r) & 1u)
                            ? unpack_hot_vq12_from_ext0_ldg(row_p, b, lane, K, d_aux_blob, aux_bytes, d_cb4096, ws)
                            : unpack_vq8_sb6_lane(row_p, b, lane, s_vq8_uv, ws);
                    } else {
                        w4 = unpack_vq12_direct_ldg(row_p, b, lane, d_cb4096, ws);
                    }
                    s_W[ln][lane] = (uint32_t)w4;
                    if ((lane & 7) == 0) s_Wsc[ln][lane >> 3] = ws;
                } else {
                    s_W[ln][lane] = 0u;
                    if ((lane & 7) == 0) s_Wsc[ln][lane >> 3] = 0.f;
                }
            }
            __syncthreads();

            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                const int k_w0 = 8 * sub + c_k;
                const int k_w1 = k_w0 + 4;
                uint32_t a_frag[2][4];
                float    x_sc[2][2];
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const int m_r0 = warp_m * 32 + mt * 16 + r_grp;
                    const int m_r1 = m_r0 + 8;
                    a_frag[mt][0] = s_X[m_r0][k_w0];
                    a_frag[mt][1] = s_X[m_r1][k_w0];
                    a_frag[mt][2] = s_X[m_r0][k_w1];
                    a_frag[mt][3] = s_X[m_r1][k_w1];
                    x_sc[mt][0]   = s_Xsc[m_r0][sub];
                    x_sc[mt][1]   = s_Xsc[m_r1][sub];
                }
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int n_col_lane = warp_n * 16 + nt * 8 + r_grp;
                    const uint32_t b0 = s_W[n_col_lane][k_w0];
                    const uint32_t b1 = s_W[n_col_lane][k_w1];
                    const int c_n0 = warp_n * 16 + nt * 8 + 2 * c_k;
                    const float ws0 = s_Wsc[c_n0 + 0][sub];
                    const float ws1 = s_Wsc[c_n0 + 1][sub];
                    #pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        int d0, d1, d2, d3;
                        dev_mma_m16n8k32_s8(
                            d0, d1, d2, d3,
                            a_frag[mt][0], a_frag[mt][1], a_frag[mt][2], a_frag[mt][3],
                            b0, b1);
                        acc[mt][nt][0] += (float)d0 * (x_sc[mt][0] * ws0);
                        acc[mt][nt][1] += (float)d1 * (x_sc[mt][0] * ws1);
                        acc[mt][nt][2] += (float)d2 * (x_sc[mt][1] * ws0);
                        acc[mt][nt][3] += (float)d3 * (x_sc[mt][1] * ws1);
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int lm0 = warp_m * 32 + mt * 16 + r_grp;
            const int lm1 = lm0 + 8;
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int gn0 = n_base + warp_n * 16 + nt * 8 + 2 * c_k;
                if (lm0 < t_cnt && gn0 < N) {
                    const int j = s_pair_j[lm0];
                    const int t = s_pair_t[lm0];
                    float * out0 = reinterpret_cast<float *>(dst_bytes + (size_t)j * nb1 + (size_t)t * nb2);
                    out0[gn0] = acc[mt][nt][0];
                    if (gn0 + 1 < N) out0[gn0 + 1] = acc[mt][nt][1];
                }
                if (lm1 < t_cnt && gn0 < N) {
                    const int j = s_pair_j[lm1];
                    const int t = s_pair_t[lm1];
                    float * out1 = reinterpret_cast<float *>(dst_bytes + (size_t)j * nb1 + (size_t)t * nb2);
                    out1[gn0] = acc[mt][nt][2];
                    if (gn0 + 1 < N) out1[gn0 + 1] = acc[mt][nt][3];
                }
            }
        }
        __syncthreads();
    }
}

} // namespace twp_detail

using namespace twp_detail;

int32_t wv_ink_cuda_load_aux(const uint8_t * aux_data, size_t aux_size, const wv_ink_host_tables * htab) {
    std::lock_guard<std::mutex> lk(g_cuda_mtx);
    if (htab) {
        g_host_tab_mirror = *htab;
        g_have_mirror = true;
    }
    if (aux_data && aux_size > 0) {
        g_host_aux_mirror.assign(aux_data, aux_data + aux_size);
    } else {
        g_host_aux_mirror.clear();
    }
    g_aux_epoch++;
    return 0;
}

void wv_ink_cuda_free_aux(void) {
    std::lock_guard<std::mutex> lk(g_cuda_mtx);
    for (int d = 0; d < 16; ++d) {
        WvInkPluginDevState & st = g_devs[d];
        if (!st.tables_ready && !st.d_yq8 && !st.d_pair_e) continue;
        cudaSetDevice(d);
        if (st.d_cb4096)   cudaFree(st.d_cb4096);
        if (st.d_vq8_uv)   cudaFree(st.d_vq8_uv);
        if (st.d_lut4)     cudaFree(st.d_lut4);
        if (st.d_aux_blob) cudaFree(st.d_aux_blob);
        if (st.d_yq8)      cudaFree(st.d_yq8);
        if (st.d_ysc)      cudaFree(st.d_ysc);
        if (st.d_pair_e)   cudaFree(st.d_pair_e);
        if (st.d_pair_j)   cudaFree(st.d_pair_j);
        if (st.d_pair_t)   cudaFree(st.d_pair_t);
        if (st.d_exp_cnt)  cudaFree(st.d_exp_cnt);
        if (st.d_exp_off)  cudaFree(st.d_exp_off);
        if (st.d_active_e) cudaFree(st.d_active_e);
        if (st.d_meta_cnt) cudaFree(st.d_meta_cnt);
        if (st.d_tile_exp) cudaFree(st.d_tile_exp);
        if (st.d_tile_beg) cudaFree(st.d_tile_beg);
        if (st.d_tile_len) cudaFree(st.d_tile_len);
        st = WvInkPluginDevState{};
        g_dev_epoch[d] = 0;
    }
    g_host_aux_mirror.clear();
    g_have_mirror = false;
}

int32_t wv_ink_mul_mat_cuda(
        int32_t ggml_type_id,
        const void * vx, const float * vy, float * dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        size_t  nb01, size_t  nb02, size_t  nb03,
        int64_t ne11, int64_t ne12, int64_t ne13,
        size_t  nb11, size_t  nb12, size_t  nb13,
        size_t  nb1,  size_t  nb2,  size_t  nb3,
        int32_t device_id, void * cuda_stream) {
    if ((ne00 & 127) != 0 || ne02 <= 0 || ne03 <= 0) {
        wv_ink_set_error("wv_ink_mul_mat_cuda: ne00 must be a multiple of 128");
        return -1;
    }
    const int ext_slot = ggml_type_id - GGML_QUANT_PLUGIN_TYPE_FIRST;
    if (ext_slot < 0 || ext_slot > 3) {
        wv_ink_set_error("wv_ink_mul_mat_cuda: unsupported ggml_type_id");
        return -1;
    }

    const int64_t total_vecs64 = ne11 * ne12 * ne13;
    if (total_vecs64 <= 0 || ne01 <= 0) return 0;

    int8_t * d_yq8 = nullptr;
    float *  d_ysc = nullptr;
    WvInkPluginDevState snap;
    {
        std::lock_guard<std::mutex> lk(g_cuda_mtx);
        if (sync_dev_tables_locked(device_id) != 0) return -1;
        if (ensure_act_workspace_locked(device_id, (size_t)total_vecs64 * (size_t)ne00) != 0) return -1;
        snap  = g_devs[device_id];
        d_yq8 = snap.d_yq8;
        d_ysc = snap.d_ysc;
    }

    cudaSetDevice(device_id);
    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
    const int K = (int)ne00;
    const int N = (int)ne01;
    const int total_vecs = (int)total_vecs64;
    const int total_blks = total_vecs * (K >> 7);

    wv_ink_plugin_fwht128_q8_kernel<<<(total_blks + 3) / 4, dim3(32, 4), 0, stream>>>(
        reinterpret_cast<const uint8_t *>(vy),
        d_yq8, d_ysc, K, (int)ne11, (int)ne12, nb11, nb12, nb13, total_vecs);

    const int r2 = (int)(ne12 / ne02);
    const int r3 = (int)(ne13 / ne03);
    const bool single_w_slice = (ne02 == 1 && ne03 == 1);

    if (ext_slot == 3) {
        if (total_vecs >= 16 && single_w_slice) {
            dim3 grid((N + 63) / 64, (total_vecs + 63) / 64);
            wv_ink_plugin_ext3_tc_m64_n64_kernel<<<grid, 128, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(vx), d_yq8, d_ysc,
                reinterpret_cast<uint8_t *>(dst),
                K, N, total_vecs, (int)ne11, (int)ne12,
                nb01, nb1, nb2, nb3);
        } else if (total_vecs >= 2 && total_vecs <= 8 && single_w_slice) {
            dim3 grid((N + 3) / 4, 1);
            wv_ink_plugin_ext3_multi_gemv_kernel<8><<<grid, 128, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(vx), d_yq8, d_ysc,
                reinterpret_cast<uint8_t *>(dst),
                K, N, total_vecs, (int)ne11, (int)ne12,
                nb01, nb1, nb2, nb3);
        } else if (total_vecs >= 9 && total_vecs < 16 && single_w_slice) {
            dim3 grid((N + 3) / 4, 1);
            wv_ink_plugin_ext3_multi_gemv_kernel<15><<<grid, 128, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(vx), d_yq8, d_ysc,
                reinterpret_cast<uint8_t *>(dst),
                K, N, total_vecs, (int)ne11, (int)ne12,
                nb01, nb1, nb2, nb3);
        } else {
            dim3 grid((N + 3) / 4, total_vecs);
            wv_ink_plugin_ext3_gemv_kernel<<<grid, 128, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(vx), d_yq8, d_ysc,
                reinterpret_cast<uint8_t *>(dst),
                K, N, (int)ne11, (int)ne12, (int)ne13, r2, r3,
                nb01, nb02, nb03, nb1, nb2, nb3);
        }
    } else if (ext_slot == 0) {
        if (total_vecs >= 16 && single_w_slice) {
            dim3 grid((N + 63) / 64, (total_vecs + 63) / 64);
            wv_ink_plugin_ext012_tc_m64_n64_kernel<0><<<grid, 256, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(vx), d_yq8, d_ysc,
                reinterpret_cast<uint8_t *>(dst),
                snap.d_cb4096, snap.d_vq8_uv, snap.d_lut4, snap.d_aux_blob, snap.aux_bytes,
                K, N, total_vecs, (int)ne11, (int)ne12, nb01, nb1, nb2, nb3);
        } else {
            dim3 grid((N + 3) / 4, total_vecs);
            wv_ink_plugin_ext012_mm_kernel<0><<<grid, 128, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(vx), d_yq8, d_ysc,
                reinterpret_cast<uint8_t *>(dst),
                snap.d_cb4096, snap.d_vq8_uv, snap.d_lut4, snap.d_aux_blob, snap.aux_bytes,
                K, N, (int)ne11, (int)ne12, r2, r3, nb01, nb02, nb03, nb1, nb2, nb3);
        }
    } else if (ext_slot == 1) {
        if (total_vecs >= 16 && single_w_slice) {
            dim3 grid((N + 63) / 64, (total_vecs + 63) / 64);
            wv_ink_plugin_ext012_tc_m64_n64_kernel<1><<<grid, 256, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(vx), d_yq8, d_ysc,
                reinterpret_cast<uint8_t *>(dst),
                snap.d_cb4096, snap.d_vq8_uv, snap.d_lut4, snap.d_aux_blob, snap.aux_bytes,
                K, N, total_vecs, (int)ne11, (int)ne12, nb01, nb1, nb2, nb3);
        } else {
            dim3 grid((N + 3) / 4, total_vecs);
            wv_ink_plugin_ext012_mm_kernel<1><<<grid, 128, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(vx), d_yq8, d_ysc,
                reinterpret_cast<uint8_t *>(dst),
                snap.d_cb4096, snap.d_vq8_uv, snap.d_lut4, snap.d_aux_blob, snap.aux_bytes,
                K, N, (int)ne11, (int)ne12, r2, r3, nb01, nb02, nb03, nb1, nb2, nb3);
        }
    } else {
        if (total_vecs >= 16 && single_w_slice) {
            dim3 grid((N + 63) / 64, (total_vecs + 63) / 64);
            wv_ink_plugin_ext012_tc_m64_n64_kernel<2><<<grid, 256, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(vx), d_yq8, d_ysc,
                reinterpret_cast<uint8_t *>(dst),
                snap.d_cb4096, snap.d_vq8_uv, snap.d_lut4, snap.d_aux_blob, snap.aux_bytes,
                K, N, total_vecs, (int)ne11, (int)ne12, nb01, nb1, nb2, nb3);
        } else {
            dim3 grid((N + 3) / 4, total_vecs);
            wv_ink_plugin_ext012_mm_kernel<2><<<grid, 128, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(vx), d_yq8, d_ysc,
                reinterpret_cast<uint8_t *>(dst),
                snap.d_cb4096, snap.d_vq8_uv, snap.d_lut4, snap.d_aux_blob, snap.aux_bytes,
                K, N, (int)ne11, (int)ne12, r2, r3, nb01, nb02, nb03, nb1, nb2, nb3);
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        wv_ink_set_error(cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int32_t wv_ink_mul_mat_id_cuda(
        int32_t ggml_type_id,
        const void * vx, const float * vy, const int32_t * ids, float * dst,
        int64_t ne00, int64_t ne01, int64_t ne02,
        size_t  nb01, size_t  nb02,
        int64_t ne11, int64_t ne12,
        size_t  nb11, size_t  nb12,
        int64_t n_expert_used,
        size_t  ids_nb0, size_t ids_nb1,
        size_t  nb1,     size_t nb2,
        int32_t first_zero,
        int32_t device_id, void * cuda_stream) {
    if ((ne00 & 127) != 0 || ne02 <= 0 || ne02 > 65536) {
        wv_ink_set_error("wv_ink_mul_mat_id_cuda: invalid ne00 or ne02");
        return -1;
    }
    const int ext_slot = ggml_type_id - GGML_QUANT_PLUGIN_TYPE_FIRST;
    if (ext_slot != 0 && ext_slot != 1) {
        wv_ink_set_error("wv_ink_mul_mat_id_cuda: expected EXT0 or EXT1");
        return -1;
    }
    const int64_t total_vecs64 = ne11 * ne12;
    const int64_t total_pairs  = n_expert_used * ne12;
    if (total_vecs64 <= 0 || total_pairs <= 0 || ne01 <= 0) return 0;

    WvInkPluginDevState snap;
    {
        std::lock_guard<std::mutex> lk(g_cuda_mtx);
        if (sync_dev_tables_locked(device_id) != 0) return -1;
        if (ensure_act_workspace_locked(device_id, (size_t)total_vecs64 * (size_t)ne00) != 0) return -1;
        if (ne12 >= 4 && ensure_pair_workspace_locked(device_id, (size_t)total_pairs, (size_t)ne02) != 0) return -1;
        snap = g_devs[device_id];
    }

    cudaSetDevice(device_id);
    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
    const int K = (int)ne00;
    const int N = (int)ne01;
    const int total_vecs = (int)total_vecs64;
    const int total_blks = total_vecs * (K >> 7);

    wv_ink_plugin_fwht128_q8_kernel<<<(total_blks + 3) / 4, dim3(32, 4), 0, stream>>>(
        reinterpret_cast<const uint8_t *>(vy),
        snap.d_yq8, snap.d_ysc, K, (int)ne11, (int)ne12, nb11, nb12, 0, total_vecs);

    if (ne12 < 4) {
        dim3 grid((N + 7) / 8, (int)total_pairs);
        if (ext_slot == 0) {
            wv_ink_plugin_mmid_smallm_kernel<0><<<grid, 256, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(vx), snap.d_yq8, snap.d_ysc,
                reinterpret_cast<const uint8_t *>(ids), reinterpret_cast<uint8_t *>(dst),
                snap.d_cb4096, snap.d_vq8_uv, snap.d_aux_blob, snap.aux_bytes,
                K, N, (int)ne02, (int)ne11, (int)n_expert_used,
                nb01, nb02, ids_nb0, ids_nb1, nb1, nb2, first_zero);
        } else {
            wv_ink_plugin_mmid_smallm_kernel<1><<<grid, 256, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(vx), snap.d_yq8, snap.d_ysc,
                reinterpret_cast<const uint8_t *>(ids), reinterpret_cast<uint8_t *>(dst),
                snap.d_cb4096, snap.d_vq8_uv, snap.d_aux_blob, snap.aux_bytes,
                K, N, (int)ne02, (int)ne11, (int)n_expert_used,
                nb01, nb02, ids_nb0, ids_nb1, nb1, nb2, first_zero);
        }
    } else {
        const int active_lim = (first_zero >= 0 && first_zero < (int)ne02) ? first_zero : (int)ne02;
        if (active_lim > 0) {
            cudaMemsetAsync(snap.d_exp_cnt, 0, (size_t)active_lim * sizeof(int32_t), stream);
        }
        const int p_blocks = ((int)total_pairs + 255) / 256;
        wv_ink_plugin_mmid_zero_and_hist_kernel<<<p_blocks, 256, 0, stream>>>(
            reinterpret_cast<const uint8_t *>(ids), reinterpret_cast<uint8_t *>(dst),
            snap.d_exp_cnt, N, (int)ne02, (int)n_expert_used, (int)ne12,
            ids_nb0, ids_nb1, nb1, nb2, first_zero);

        if (active_lim > 0) {
            const bool use_tc = (ne12 >= 16);
            wv_ink_plugin_mmid_plan_kernel<<<1, 256, 0, stream>>>(
                snap.d_exp_cnt, snap.d_exp_off, snap.d_pair_e,
                snap.d_active_e, snap.d_tile_exp, snap.d_tile_beg, snap.d_tile_len,
                snap.d_meta_cnt, active_lim, use_tc ? 1 : 0);
            wv_ink_plugin_mmid_scatter_kernel<<<p_blocks, 256, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(ids), snap.d_pair_e,
                snap.d_pair_j, snap.d_pair_t,
                (int)ne02, (int)n_expert_used, (int)ne12, ids_nb0, ids_nb1, first_zero);

            const int max_active = (active_lim < (int)total_pairs) ? active_lim : (int)total_pairs;
            if (!use_tc) {
                dim3 grid((N + 7) / 8, std::min(max_active, 65535));
                if (ext_slot == 0) {
                    wv_ink_plugin_mmid_grouped8_kernel<0><<<grid, 256, 0, stream>>>(
                        reinterpret_cast<const uint8_t *>(vx), snap.d_yq8, snap.d_ysc,
                        snap.d_active_e, snap.d_meta_cnt,
                        snap.d_exp_off, snap.d_pair_j, snap.d_pair_t, reinterpret_cast<uint8_t *>(dst),
                        snap.d_cb4096, snap.d_vq8_uv, snap.d_aux_blob, snap.aux_bytes,
                        K, N, (int)ne11, nb01, nb02, nb1, nb2);
                } else {
                    wv_ink_plugin_mmid_grouped8_kernel<1><<<grid, 256, 0, stream>>>(
                        reinterpret_cast<const uint8_t *>(vx), snap.d_yq8, snap.d_ysc,
                        snap.d_active_e, snap.d_meta_cnt,
                        snap.d_exp_off, snap.d_pair_j, snap.d_pair_t, reinterpret_cast<uint8_t *>(dst),
                        snap.d_cb4096, snap.d_vq8_uv, snap.d_aux_blob, snap.aux_bytes,
                        K, N, (int)ne11, nb01, nb02, nb1, nb2);
                }
            } else {
                const int64_t max_tiles64 = (int64_t)max_active + ((total_pairs - (int64_t)max_active) / 64);
                const int grid_y = (int)std::min<int64_t>(std::max<int64_t>(1, max_tiles64), 65535);
                dim3 grid((N + 63) / 64, grid_y);
                if (ext_slot == 0) {
                    wv_ink_plugin_mmid_tc_m64_n64_kernel<0><<<grid, 256, 0, stream>>>(
                        reinterpret_cast<const uint8_t *>(vx), snap.d_yq8, snap.d_ysc,
                        snap.d_tile_exp, snap.d_tile_beg, snap.d_tile_len, snap.d_meta_cnt,
                        snap.d_pair_j, snap.d_pair_t, reinterpret_cast<uint8_t *>(dst),
                        snap.d_cb4096, snap.d_vq8_uv, snap.d_aux_blob, snap.aux_bytes,
                        K, N, (int)ne11, nb01, nb02, nb1, nb2);
                } else {
                    wv_ink_plugin_mmid_tc_m64_n64_kernel<1><<<grid, 256, 0, stream>>>(
                        reinterpret_cast<const uint8_t *>(vx), snap.d_yq8, snap.d_ysc,
                        snap.d_tile_exp, snap.d_tile_beg, snap.d_tile_len, snap.d_meta_cnt,
                        snap.d_pair_j, snap.d_pair_t, reinterpret_cast<uint8_t *>(dst),
                        snap.d_cb4096, snap.d_vq8_uv, snap.d_aux_blob, snap.aux_bytes,
                        K, N, (int)ne11, nb01, nb02, nb1, nb2);
                }
            }
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        wv_ink_set_error(cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

#pragma once
/* Per-TU constant/device codebook banks (`static` for isolated 64 KB __constant__ banks per .cu TU
 * under -rdc=false) + Tier-1 micro-primitive includes & multi-TU table sync macros. */

#include "wv_micro_ops.cuh"
#include "wv_default_codebooks.cuh"

namespace wv {

static __constant__ uint32_t c_wv_vq8_uv[2][512] = WV_DEF_VQ8_INIT; /* 4 KB 4-Coset Interstitial VQ8 */
static __constant__ uint32_t c_wv_vq4_u32[256]   = WV_DEF_VQ4_INIT; /* 1 KB 1-Byte VQ4_256 */
static __constant__ int      c_wv_2b25_is_vq4    = 0;
static __constant__ uint2    c_wv_lut3_u64[8]    = WV_DEF_LUT3_INIT;
static __constant__ uint4    c_wv_lut4_u128[8]   = WV_DEF_LUT4_INIT;
static __constant__ uint4    c_wv_lut16[16]      = WV_DEF_LUT16_INIT;
static __constant__ int8_t   c_wv_lloyd4_i8[16]  = {
    -127, -98, -77, -60, -45, -31, -18, -6, 6, 18, 31, 45, 60, 77, 98, 127
};
static __device__ uint32_t   d_wv_vq12_cb4096[4096]; /* 16 KB Unconstrained 4D VQ-4096 */

__device__ __forceinline__ void wv_load_vq8_smem(TableVQ8& s_vq8, int tid, int block_threads = 128) {
    if (c_wv_2b25_is_vq4 != 0) {
        for (int idx = tid; idx < 256; idx += block_threads) {
            uint32_t v = c_wv_vq4_u32[idx];
            s_vq8.uv[0][idx][0] = v;
            s_vq8.uv[0][idx][1] = v;
        }
        return;
    }
    for (int idx = tid; idx < 1024; idx += block_threads) {
        uint32_t v = reinterpret_cast<const uint32_t*>(c_wv_vq8_uv)[idx];
        reinterpret_cast<uint2*>(s_vq8.uv)[idx] = make_uint2(v, v);
    }
}

__device__ __forceinline__ void wv_load_vq8_compact_smem(TableVQ8Compact& s_vq8, int tid, int block_threads = 128) {
    if (c_wv_2b25_is_vq4 != 0) {
        for (int idx = tid; idx < 256; idx += block_threads) {
            s_vq8.uv[0][idx] = c_wv_vq4_u32[idx];
        }
        return;
    }
    for (int idx = tid; idx < 256; idx += block_threads) {
        reinterpret_cast<uint4*>(s_vq8.uv)[idx] = reinterpret_cast<const uint4*>(c_wv_vq8_uv)[idx];
    }
}

__device__ __forceinline__ void wv_load_vq12_smem(TableVQ12& s_vq12, int tid, int block_threads = 128) {
    for (int idx = tid; idx < 1024; idx += block_threads) {
        reinterpret_cast<uint4*>(s_vq12.cb)[idx] = reinterpret_cast<const uint4*>(d_wv_vq12_cb4096)[idx];
    }
}

__device__ __forceinline__ void wv_unpack_swz_vq8_blk64(
    uint32_t pkg, int c_k, int rep, const TableVQ8& s_vq8,
    uint32_t& b0_s0, uint32_t& b1_s0, uint32_t& b0_s1, uint32_t& b1_s1) {
    int odd = c_k & 1;
    uint32_t mate = __shfl_xor_sync(0xffffffffu, pkg, 1);
    uint32_t p_s0 = odd ? mate : pkg;
    uint32_t p_s1 = odd ? pkg  : mate;
    uint32_t lo0 = p_s0 & 0xFFFFu, hi0 = p_s0 >> 16;
    uint32_t lo1 = p_s1 & 0xFFFFu, hi1 = p_s1 >> 16;
    if (c_wv_2b25_is_vq4 != 0) {
        uint32_t s_lo0 = odd ? ((lo0 >> 8) & 0xFFu) : (lo0 & 0xFFu);
        uint32_t s_hi0 = odd ? ((hi0 >> 8) & 0xFFu) : (hi0 & 0xFFu);
        uint32_t s_lo1 = odd ? ((lo1 >> 8) & 0xFFu) : (lo1 & 0xFFu);
        uint32_t s_hi1 = odd ? ((hi1 >> 8) & 0xFFu) : (hi1 & 0xFFu);
        b0_s0 = s_vq8.uv[0][s_lo0][rep]; b1_s0 = s_vq8.uv[0][s_hi0][rep];
        b0_s1 = s_vq8.uv[0][s_lo1][rep]; b1_s1 = s_vq8.uv[0][s_hi1][rep];
        return;
    }
    uint32_t s_lo0 = odd ? (((lo0 >> 7) & 0x180u) | (lo0 & 0x7Fu)) : (lo0 >> 7);
    uint32_t s_hi0 = odd ? (((hi0 >> 7) & 0x180u) | (hi0 & 0x7Fu)) : (hi0 >> 7);
    uint32_t s_lo1 = odd ? (((lo1 >> 7) & 0x180u) | (lo1 & 0x7Fu)) : (lo1 >> 7);
    uint32_t s_hi1 = odd ? (((hi1 >> 7) & 0x180u) | (hi1 & 0x7Fu)) : (hi1 >> 7);
    b0_s0 = s_vq8.uv[odd][s_lo0][rep]; b1_s0 = s_vq8.uv[odd][s_hi0][rep];
    b0_s1 = s_vq8.uv[odd][s_lo1][rep]; b1_s1 = s_vq8.uv[odd][s_hi1][rep];
}

__device__ __forceinline__ void wv_unpack_swz_vq8_compact_blk64(
    uint32_t pkg, int lane, const TableVQ8Compact& s_vq8,
    uint32_t& b0_s0, uint32_t& b1_s0, uint32_t& b0_s1, uint32_t& b1_s1) {
    int odd = lane & 1;
    uint32_t mate = __shfl_xor_sync(0xffffffffu, pkg, 1);
    uint32_t p_s0 = odd ? mate : pkg;
    uint32_t p_s1 = odd ? pkg  : mate;
    uint32_t lo0 = p_s0 & 0xFFFFu, hi0 = p_s0 >> 16;
    uint32_t lo1 = p_s1 & 0xFFFFu, hi1 = p_s1 >> 16;
    if (c_wv_2b25_is_vq4 != 0) {
        uint32_t s_lo0 = odd ? ((lo0 >> 8) & 0xFFu) : (lo0 & 0xFFu);
        uint32_t s_hi0 = odd ? ((hi0 >> 8) & 0xFFu) : (hi0 & 0xFFu);
        uint32_t s_lo1 = odd ? ((lo1 >> 8) & 0xFFu) : (lo1 & 0xFFu);
        uint32_t s_hi1 = odd ? ((hi1 >> 8) & 0xFFu) : (hi1 & 0xFFu);
        b0_s0 = s_vq8.uv[0][s_lo0]; b1_s0 = s_vq8.uv[0][s_hi0];
        b0_s1 = s_vq8.uv[0][s_lo1]; b1_s1 = s_vq8.uv[0][s_hi1];
        return;
    }
    uint32_t s_lo0 = odd ? (((lo0 >> 7) & 0x180u) | (lo0 & 0x7Fu)) : (lo0 >> 7);
    uint32_t s_hi0 = odd ? (((hi0 >> 7) & 0x180u) | (hi0 & 0x7Fu)) : (hi0 >> 7);
    uint32_t s_lo1 = odd ? (((lo1 >> 7) & 0x180u) | (lo1 & 0x7Fu)) : (lo1 >> 7);
    uint32_t s_hi1 = odd ? (((hi1 >> 7) & 0x180u) | (hi1 & 0x7Fu)) : (hi1 >> 7);
    b0_s0 = s_vq8.uv[odd][s_lo0]; b1_s0 = s_vq8.uv[odd][s_hi0];
    b0_s1 = s_vq8.uv[odd][s_lo1]; b1_s1 = s_vq8.uv[odd][s_hi1];
}

__device__ __forceinline__ void wv_unpack_vq8_meta_generic(
    const __half* vq_sc_base, int kc128, float ws_out[4]) {
    const __half* p = vq_sc_base + kc128 * 4 * 8;
    ws_out[0] = __half2float(__ldg(p + 0 * 8)) * (1.0f / 127.0f);
    ws_out[1] = __half2float(__ldg(p + 1 * 8)) * (1.0f / 127.0f);
    ws_out[2] = __half2float(__ldg(p + 2 * 8)) * (1.0f / 127.0f);
    ws_out[3] = __half2float(__ldg(p + 3 * 8)) * (1.0f / 127.0f);
}

template<bool ALWAYS_NV>
__device__ __forceinline__ void wv_nvfp4_or_mixed_scales(
    bool is_nv_row, uint16_t sc_u16, int exp2_bias, int col_n0, int col_n1,
    float& wlo0, float& wlo1, float& whi0, float& whi1) {
    constexpr float kInv127 = 1.0f / 127.0f;
    float wlo_self, whi_self;
    if (ALWAYS_NV || is_nv_row) {
        wlo_self = wv_decode_nvfp4_ue4m3(static_cast<uint8_t>(sc_u16 & 0xFFu), exp2_bias);
        whi_self = wv_decode_nvfp4_ue4m3(static_cast<uint8_t>(sc_u16 >> 8), exp2_bias);
    } else {
        float ws = __half2float(__ushort_as_half(sc_u16)) * kInv127;
        wlo_self = ws;
        whi_self = ws;
    }
    wlo0 = __shfl_sync(0xffffffffu, wlo_self, col_n0 * 4);
    wlo1 = __shfl_sync(0xffffffffu, wlo_self, col_n1 * 4);
    whi0 = __shfl_sync(0xffffffffu, whi_self, col_n0 * 4);
    whi1 = __shfl_sync(0xffffffffu, whi_self, col_n1 * 4);
}

template<int W_MODE, bool ONE_ROW>
__device__ __forceinline__ void wv_mma_sub_acc(
    uint32_t ax0_lo, uint32_t ax1_lo, uint32_t ax0_hi, uint32_t ax1_hi,
    uint32_t b0, uint32_t b1,
    uint16_t sc_u16, uint8_t bk_id, bool mixed_has_nv,
    int col_n0, int col_n1, float xs0, float xs1,
    float& acc0, float& acc1, float& acc2, float& acc3) {
    if ((W_MODE == 4) || (W_MODE == 0 && mixed_has_nv)) {
        float wlo0, wlo1, whi0, whi1;
        wv_nvfp4_or_mixed_scales<(W_MODE == 4)>(
            (bk_id & 0x80u) != 0u, sc_u16, static_cast<int>(bk_id) - 73,
            col_n0, col_n1, wlo0, wlo1, whi0, whi1);
        int dl0, dl1, dl2, dl3, dh0, dh1, dh2, dh3;
        if constexpr (ONE_ROW) {
            wv_mma_m16n8k32_s8(dl0, dl1, dl2, dl3, ax0_lo, 0u, 0u, 0u, b0, 0u);
            wv_mma_m16n8k32_s8(dh0, dh1, dh2, dh3, 0u, 0u, ax0_hi, 0u, 0u, b1);
            acc0 += (static_cast<float>(dl0) * wlo0 + static_cast<float>(dh0) * whi0) * xs0;
            acc1 += (static_cast<float>(dl1) * wlo1 + static_cast<float>(dh1) * whi1) * xs0;
        } else {
            wv_mma_m16n8k32_s8(dl0, dl1, dl2, dl3, ax0_lo, ax1_lo, 0u, 0u, b0, 0u);
            wv_mma_m16n8k32_s8(dh0, dh1, dh2, dh3, 0u, 0u, ax0_hi, ax1_hi, 0u, b1);
            acc0 += (static_cast<float>(dl0) * wlo0 + static_cast<float>(dh0) * whi0) * xs0;
            acc1 += (static_cast<float>(dl1) * wlo1 + static_cast<float>(dh1) * whi1) * xs0;
            acc2 += (static_cast<float>(dl2) * wlo0 + static_cast<float>(dh2) * whi0) * xs1;
            acc3 += (static_cast<float>(dl3) * wlo1 + static_cast<float>(dh3) * whi1) * xs1;
        }
    } else {
        float ws_self = __half2float(__ushort_as_half(sc_u16)) * (1.0f / 127.0f);
        float ws0 = __shfl_sync(0xffffffffu, ws_self, col_n0 * 4);
        float ws1 = __shfl_sync(0xffffffffu, ws_self, col_n1 * 4);
        int d0, d1, d2, d3;
        if constexpr (ONE_ROW) {
            wv_mma_m16n8k32_s8(d0, d1, d2, d3, ax0_lo, 0u, ax0_hi, 0u, b0, b1);
            acc0 += static_cast<float>(d0) * (ws0 * xs0);
            acc1 += static_cast<float>(d1) * (ws1 * xs0);
        } else {
            wv_mma_m16n8k32_s8(d0, d1, d2, d3, ax0_lo, ax1_lo, ax0_hi, ax1_hi, b0, b1);
            acc0 += static_cast<float>(d0) * (ws0 * xs0);
            acc1 += static_cast<float>(d1) * (ws1 * xs0);
            acc2 += static_cast<float>(d2) * (ws0 * xs1);
            acc3 += static_cast<float>(d3) * (ws1 * xs1);
        }
    }
}

template<int TIER>
__device__ __forceinline__ int32_t wv_unpack4_i8(
    const uint8_t* row_bytes, int b128, int k_sub, int K, float& out_w_scale, float ws2_f32_row = 1.0f) {
    if constexpr (TIER == 0) {
        int n_blk64 = K >> 6;
        int blk64   = 2 * b128 + (k_sub >> 4);
        int sb_base = ((n_blk64 % 5) == 0) ? ((blk64 / 5) * 5) : (blk64 & ~3);
        const auto* blocks = reinterpret_cast<const wv6_block_2b25*>(row_bytes);
        int i0 = min(sb_base + 0, n_blk64 - 1), i1 = min(sb_base + 1, n_blk64 - 1);
        int i2 = min(sb_base + 2, n_blk64 - 1), i3 = min(sb_base + 3, n_blk64 - 1);
        uint16_t d_u16 = static_cast<uint16_t>(
              ((blocks[i0].sc_meta >> 12) & 0xFu)
            | (((blocks[i1].sc_meta >> 12) & 0xFu) << 4)
            | (((blocks[i2].sc_meta >> 12) & 0xFu) << 8)
            | (((blocks[i3].sc_meta >> 12) & 0xFu) << 12));
        const auto& bk = blocks[blk64];
        int sub16 = k_sub & 15;
        uint32_t u6 = (sub16 < 8) ? (bk.sc_meta & 0x3Fu) : ((bk.sc_meta >> 6) & 0x3Fu);
        float sc127 = __half2float(__float2half(wv_fp16_to_f32(d_u16) * (static_cast<float>(u6) / 63.0f)));
        out_w_scale = sc127 * (1.0f / 127.0f);
        if (c_wv_2b25_is_vq4 != 0) {
            return static_cast<int32_t>(c_wv_vq4_u32[bk.qs[sub16]]);
        }
        const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(bk.qs);
        uint32_t idx16 = static_cast<uint32_t>(qs16[sub16 >> 1]);
        uint32_t c = (idx16 >> 14) & 3u;
        uint32_t i = (idx16 >> 7) & 127u;
        uint32_t j = idx16 & 127u;
        return ((sub16 & 1) == 0)
            ? static_cast<int32_t>(c_wv_vq8_uv[0][(c << 7) | i])
            : static_cast<int32_t>(c_wv_vq8_uv[1][(c << 7) | j]);
    } else if constexpr (TIER == 1) {
        const uint8_t* bp = row_bytes + static_cast<size_t>(b128) * 54;
        uint16_t d_sup = *reinterpret_cast<const uint16_t*>(bp);
        uint32_t m = wv_ld_u32_align2(bp + 2);
        const uint8_t* qs_lo2 = bp + 6;
        const uint8_t* qs_hi1 = bp + 38;
        int p  = k_sub >> 3;
        int j4 = (k_sub & 7) << 2;
        uint32_t u6 = (m >> (6 * p)) & 0x3Fu;
        float sc127 = __half2float(__float2half(wv_fp16_to_f32(d_sup) * (static_cast<float>(u6) / 63.0f)));
        out_w_scale = sc127 * (1.0f / 127.0f);
        uint2 lut8 = c_wv_lut3_u64[(m >> 24) & 7u];
        uint32_t sel = 0;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int j   = j4 + r;
            int pos = 32 * p + j;
            uint32_t lo2 = (qs_lo2[j] >> (2 * p)) & 3u;
            uint32_t hi1 = ((qs_hi1[pos & 15] >> (pos >> 4)) & 1u) << 2;
            sel |= (lo2 | hi1) << (4 * r);
        }
        return static_cast<int32_t>(__byte_perm(lut8.x, lut8.y, sel));
    } else if constexpr (TIER == 3) {
        const uint8_t* bp = row_bytes + static_cast<size_t>(b128) * 53;
        uint16_t d_sup = static_cast<uint16_t>(bp[0]) | (static_cast<uint16_t>(bp[1]) << 8);
        uint32_t u24 = static_cast<uint32_t>(bp[2]) | (static_cast<uint32_t>(bp[3]) << 8) | (static_cast<uint32_t>(bp[4]) << 16);
        int p = k_sub >> 3;
        uint32_t u6 = (u24 >> (6 * p)) & 0x3Fu;
        float sc127 = __half2float(__float2half(wv_fp16_to_f32(d_sup) * (static_cast<float>(u6) / 63.0f)));
        out_w_scale = sc127 * (1.0f / 127.0f);
        const uint8_t* qp = bp + 5 + (k_sub >> 1) * 3;
        uint32_t p24 = static_cast<uint32_t>(qp[0]) | (static_cast<uint32_t>(qp[1]) << 8) | (static_cast<uint32_t>(qp[2]) << 16);
        uint32_t idx12 = ((k_sub & 1) == 0) ? (p24 & 0xFFFu) : ((p24 >> 12) & 0xFFFu);
        return static_cast<int32_t>(d_wv_vq12_cb4096[idx12]);
    } else if constexpr (TIER == 4) {
        const uint8_t* bp = row_bytes + static_cast<size_t>(b128) * 72;
        uint8_t u8 = bp[k_sub >> 2];
        int exp2_ws2 = max(-64, min(63, static_cast<int>(lrintf(log2f(fmaxf(ws2_f32_row, 1e-30f))))));
        out_w_scale = wv_decode_nvfp4_ue4m3(u8, exp2_ws2 + 119);
        uint16_t raw16 = static_cast<uint16_t>(bp[8 + 2 * k_sub]) | (static_cast<uint16_t>(bp[9 + 2 * k_sub]) << 8);
        uint32_t sel = static_cast<uint32_t>(raw16) & 0x7777u;
        uint32_t lo  = __byte_perm(0x03020100u, 0x0C080604u, sel);
        uint32_t hi  = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, sel);
        uint32_t m   = __byte_perm(0u, 0xFFFFFFFFu, (static_cast<uint32_t>(raw16) & 0x8888u) >> 1);
        return static_cast<int32_t>((hi & m) | (lo & ~m));
    } else if constexpr (TIER == 8) {
        int p  = k_sub >> 3;
        int j4 = (k_sub & 7) << 2;
        const uint8_t* gp = row_bytes + static_cast<size_t>(b128 * 4 + p) * 34;
        out_w_scale = wv_fp16_to_f32(*reinterpret_cast<const uint16_t*>(gp)) * (1.0f / 127.0f);
        return static_cast<int32_t>(wv_ld_u32_align2(gp + 2 + j4));
    } else {
        return wv_unpack_wv6_4b38_row_vec4(row_bytes, b128, k_sub, c_wv_lut4_u128, out_w_scale);
    }
}

inline cudaError_t wv_sync_this_tu_tables(
    const void* vq8_src, const void* lut3_src, const void* lut4_src,
    const void* lut16_src, const void* vq12_src) {
    cudaError_t err = cudaMemcpyToSymbol(c_wv_vq8_uv, vq8_src, 4096);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(c_wv_lut3_u64, lut3_src, 64);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(c_wv_lut4_u128, lut4_src, 128);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyToSymbol(c_wv_lut16, lut16_src, 256);
    if (err != cudaSuccess) return err;
    if (vq12_src != nullptr) {
        err = cudaMemcpyToSymbol(d_wv_vq12_cb4096, vq12_src, 16384);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

inline cudaError_t wv_sync_this_tu_vq4(int enable_vq4, const void* vq4_src) {
    cudaError_t err = cudaMemcpyToSymbol(c_wv_vq4_u32, vq4_src, 1024);
    if (err != cudaSuccess) return err;
    int flag = (enable_vq4 != 0) ? 1 : 0;
    return cudaMemcpyToSymbol(c_wv_2b25_is_vq4, &flag, sizeof(flag));
}

} // namespace wv

#define WV_DECLARE_TU_SYNC(fn_tables, fn_vq4) \
    cudaError_t fn_tables(const void*, const void*, const void*, const void*, const void*); \
    cudaError_t fn_vq4(int, const void*);

#define WV_DEFINE_TU_SYNC(fn_tables, fn_vq4) \
    cudaError_t fn_tables(const void* vq8, const void* l3, const void* l4, const void* l16, const void* vq12) { \
        return ::wv::wv_sync_this_tu_tables(vq8, l3, l4, l16, vq12); \
    } \
    cudaError_t fn_vq4(int en, const void* vq4) { \
        return ::wv::wv_sync_this_tu_vq4(en, vq4); \
    }

#pragma once
/* Tier-1 Shared @device Micro-Primitives (100% inline leaf functions, zero __global__ kernels,
 * zero mutable globals). Shared across all 5 translation units of wv_runtime_core. */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace wv {

static constexpr float WV_INV_SQRT128 = 0.08838834764831845f;
static constexpr float WV_U6_NORM     = 1.0f / (63.0f * 127.0f);

#pragma pack(push, 1)
struct wv6_block_2b25 {
    uint16_t sc_meta;
    uint8_t  qs[16];
};
struct wv6_block_3b38 {
    uint16_t d_super;
    uint32_t sc_u6_lut;
    uint8_t  qs_lo2[32];
    uint8_t  qs_hi1[16];
};
struct wv6_block_4b38 {
    uint16_t d_super;
    uint32_t sc_u6_lut;
    uint8_t  qs_nib[64];
};
struct wv6_block_vq12 {
    uint16_t d_super;
    uint8_t  sc_u6_3b[3];
    uint8_t  qs_idx12[48];
};

struct wv6_block_nvfp4 {
    uint8_t  bp[8];
    uint8_t  qs[64];
};
#pragma pack(pop)

static_assert(sizeof(wv6_block_2b25) == 18, "wv6_block_2b25 size");
static_assert(sizeof(wv6_block_3b38) == 54, "wv6_block_3b38 size");
static_assert(sizeof(wv6_block_4b38) == 70, "wv6_block_4b38 size");
static_assert(sizeof(wv6_block_vq12) == 53, "wv6_block_vq12 size");
static_assert(sizeof(wv6_block_nvfp4) == 72, "wv6_block_nvfp4 size");

__device__ __forceinline__ float wv_decode_nvfp4_ue4m3(uint8_t u8, int exp2_bias) {
    return (u8 >= 8u)
        ? __uint_as_float(static_cast<uint32_t>((static_cast<int>(u8) << 20) + (exp2_bias << 23)))
        : 0.0f;
}

__device__ __forceinline__ uint4 wv_nvfp4_lut_2x_e2m1() {
    return make_uint4(0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu);
}

struct TableVQ8      { uint32_t uv[2][512][2]; }; /* 8 KB SMEM (dual-bank copy) */
struct TableVQ8Compact { uint32_t uv[2][512]; };  /* 4 KB SMEM (compact CTA-uniform VQ8) */
struct TableVQ12     { uint32_t cb[4096]; };       /* 16 KB SMEM (4D VQ-4096) */

__device__ __forceinline__ uint32_t wv_ld_u32_align2(const void* p) {
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(p);
    return static_cast<uint32_t>(p16[0]) | (static_cast<uint32_t>(p16[1]) << 16);
}

__device__ __forceinline__ float wv_fp16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ float wv_warp_sum_f32(float v) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    }
    return v;
}

__device__ __forceinline__ float wv_warp_max_f32(float v) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, mask));
    }
    return v;
}

__device__ __forceinline__ void wv_warp_argmax_f32(float& val, int32_t& idx) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float   ov = __shfl_xor_sync(0xffffffffu, val, mask);
        int32_t oi = __shfl_xor_sync(0xffffffffu, idx, mask);
        if (ov > val || (ov == val && oi < idx)) {
            val = ov;
            idx = oi;
        }
    }
}

__device__ __forceinline__ void wv_pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void wv_pdl_trigger() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

template<typename KernelFn, typename... Args>
inline void wv_launch_pdl(
    bool enable_pdl, dim3 grid, dim3 block, size_t smem_bytes, cudaStream_t stream,
    KernelFn kernel, Args... args) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
    if (enable_pdl) {
        cudaLaunchConfig_t cfg {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem_bytes;
        cfg.stream = stream;
        cudaLaunchAttribute attr {};
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, kernel, args...);
        return;
    }
#else
    (void)enable_pdl;
#endif
    kernel<<<grid, block, smem_bytes, stream>>>(args...);
}

__device__ __forceinline__ uint32_t wv_ld_stream_u32(const uint32_t* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    uint64_t pol;
    uint32_t r;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "l"(pol));
    return r;
#else
    return __ldg(p);
#endif
}

__device__ __forceinline__ uint2 wv_ld_stream_u64(const uint2* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    uint64_t pol;
    uint2 r;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;"
        : "=r"(r.x), "=r"(r.y) : "l"(p), "l"(pol));
    return r;
#else
    return __ldg(p);
#endif
}

__device__ __forceinline__ uint2 wv_ld_cg_u64(const uint2* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    uint2 r;
    asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
#else
    return *p;
#endif
}

__device__ __forceinline__ void wv_bitonic_sort32_desc(float& val, int& idx, int lane) {
    #pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {
        for (int j = k >> 1; j >= 1; j >>= 1) {
            float o_val = __shfl_xor_sync(0xffffffffu, val, j);
            int   o_idx = __shfl_xor_sync(0xffffffffu, idx, j);
            bool dir_desc = ((lane & k) == 0);
            bool i_am_hi  = ((lane & j) != 0);
            bool want_max = (dir_desc != i_am_hi);
            bool take_other = want_max
                ? (o_val > val || (o_val == val && o_idx < idx))
                : (o_val < val || (o_val == val && o_idx > idx));
            if (take_other) { val = o_val; idx = o_idx; }
        }
    }
}

__device__ __forceinline__ void wv_bitonic_merge32_desc(float& val, int& idx, int lane) {
    #pragma unroll
    for (int j = 16; j >= 1; j >>= 1) {
        float o_val = __shfl_xor_sync(0xffffffffu, val, j);
        int   o_idx = __shfl_xor_sync(0xffffffffu, idx, j);
        bool i_am_hi = ((lane & j) != 0);
        bool take_other = !i_am_hi
            ? (o_val > val || (o_val == val && o_idx < idx))
            : (o_val < val || (o_val == val && o_idx > idx));
        if (take_other) { val = o_val; idx = o_idx; }
    }
}

__device__ __forceinline__ void wv_fwht128_warp_inplace(float& f0, float& f1, float& f2, float& f3, int lane) {
    {
        float a0 = f0 + f1, a1 = f0 - f1, a2 = f2 + f3, a3 = f2 - f3;
        f0 = a0 + a2; f1 = a1 + a3; f2 = a0 - a2; f3 = a1 - a3;
    }
    #pragma unroll
    for (int mask = 1; mask <= 16; mask <<= 1) {
        float p0 = __shfl_xor_sync(0xffffffffu, f0, mask);
        float p1 = __shfl_xor_sync(0xffffffffu, f1, mask);
        float p2 = __shfl_xor_sync(0xffffffffu, f2, mask);
        float p3 = __shfl_xor_sync(0xffffffffu, f3, mask);
        bool hi = (lane & mask) != 0;
        f0 = hi ? (p0 - f0) : (f0 + p0);
        f1 = hi ? (p1 - f1) : (f1 + p1);
        f2 = hi ? (p2 - f2) : (f2 + p2);
        f3 = hi ? (p3 - f3) : (f3 + p3);
    }
    f0 *= WV_INV_SQRT128; f1 *= WV_INV_SQRT128;
    f2 *= WV_INV_SQRT128; f3 *= WV_INV_SQRT128;
}

__device__ __forceinline__ void wv_quant_q8_32_warp(
    float f0, float f1, float f2, float f3, int lane,
    int8_t* __restrict__ q8_blk128, float* __restrict__ sc_blk4) {
    float amax = fmaxf(fmaxf(fabsf(f0), fabsf(f1)), fmaxf(fabsf(f2), fabsf(f3)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    float sc     = amax * (1.0f / 127.0f);
    float inv_sc = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
    if ((lane & 7) == 0) sc_blk4[lane >> 3] = sc;
    char4 q4;
    q4.x = static_cast<int8_t>(__float2int_rn(f0 * inv_sc));
    q4.y = static_cast<int8_t>(__float2int_rn(f1 * inv_sc));
    q4.z = static_cast<int8_t>(__float2int_rn(f2 * inv_sc));
    q4.w = static_cast<int8_t>(__float2int_rn(f3 * inv_sc));
    reinterpret_cast<char4*>(q8_blk128)[lane] = q4;
}

__device__ __forceinline__ void wv_mma_m16n8k32_s8(
    int& d0, int& d1, int& d2, int& d3,
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
    int lane = threadIdx.x & 31;
    int c_k  = lane & 3;
    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    #pragma unroll
    for (int col = 0; col < 8; ++col) {
        int src_base = col * 4;
        int32_t b0_c = static_cast<int32_t>(__shfl_sync(0xffffffffu, b0, src_base + c_k));
        int32_t b1_c = static_cast<int32_t>(__shfl_sync(0xffffffffu, b1, src_base + c_k));
        int32_t dp_hi = __dp4a(static_cast<int32_t>(a2), b1_c, __dp4a(static_cast<int32_t>(a0), b0_c, 0));
        int32_t dp_lo = __dp4a(static_cast<int32_t>(a3), b1_c, __dp4a(static_cast<int32_t>(a1), b0_c, 0));
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

__device__ __forceinline__ int32_t wv_prmt16_lut4(uint4 lut16, uint32_t nib4) {
    uint32_t lo3  = nib4 & 0x07070707u;
    uint32_t s16  = (lo3 & 0x7u) | ((lo3 >> 4) & 0x70u) | ((lo3 >> 8) & 0x700u) | ((lo3 >> 12) & 0x7000u);
    uint32_t v_lo = __byte_perm(lut16.x, lut16.y, s16);
    uint32_t v_hi = __byte_perm(lut16.z, lut16.w, s16);
    uint32_t b3   = (nib4 & 0x08080808u) >> 3;
    uint32_t mask = (b3 << 8) - b3;
    return static_cast<int32_t>((v_lo & ~mask) | (v_hi & mask));
}

__device__ __forceinline__ void wv_unpack_swz4_b01(uint32_t pk, uint4 lut, uint32_t& b0, uint32_t& b1) {
    uint32_t pk_lo = pk & 0xFFFFu;
    uint32_t pk_hi = pk >> 16;
    uint32_t sel0  = pk_lo & 0x7777u;
    uint32_t lo0   = __byte_perm(lut.x, lut.y, sel0);
    uint32_t hi0   = __byte_perm(lut.z, lut.w, sel0);
    uint32_t m0    = __byte_perm(0u, 0xFFFFFFFFu, (pk_lo & 0x8888u) >> 1);
    b0 = (hi0 & m0) | (lo0 & ~m0);

    uint32_t sel1  = pk_hi & 0x7777u;
    uint32_t lo1   = __byte_perm(lut.x, lut.y, sel1);
    uint32_t hi1   = __byte_perm(lut.z, lut.w, sel1);
    uint32_t m1    = __byte_perm(0u, 0xFFFFFFFFu, (pk_hi & 0x8888u) >> 1);
    b1 = (hi1 & m1) | (lo1 & ~m1);
}

__device__ __forceinline__ void wv_unpack_swz3_sub4_micro(
    uint32_t w0, uint32_t w1, uint32_t w2, uint2 lut8, uint32_t b0[4], uint32_t b1[4]) {
    uint32_t s6 = ((w0 & 0x8888u) >> 3) | (((w0 >> 16) & 0x8888u) >> 2) | ((w1 & 0x8888u) >> 1);
    uint32_t s7 = (((w1 >> 16) & 0x8888u) >> 3) | ((w2 & 0x8888u) >> 2) | (((w2 >> 16) & 0x8888u) >> 1);
    b0[0] = __byte_perm(lut8.x, lut8.y, w0 & 0x7777u);
    b1[0] = __byte_perm(lut8.x, lut8.y, (w0 >> 16) & 0x7777u);
    b0[1] = __byte_perm(lut8.x, lut8.y, w1 & 0x7777u);
    b1[1] = __byte_perm(lut8.x, lut8.y, (w1 >> 16) & 0x7777u);
    b0[2] = __byte_perm(lut8.x, lut8.y, w2 & 0x7777u);
    b1[2] = __byte_perm(lut8.x, lut8.y, (w2 >> 16) & 0x7777u);
    b0[3] = __byte_perm(lut8.x, lut8.y, s6);
    b1[3] = __byte_perm(lut8.x, lut8.y, s7);
}

__device__ __forceinline__ void wv_unpack_vq12_sub4_tab(
    uint32_t w0, uint32_t w1, uint32_t w2, const uint32_t* s_cb4096,
    uint32_t b0[4], uint32_t b1[4]) {
    b0[0] = s_cb4096[w0 & 0xFFFu];
    b1[0] = s_cb4096[(w0 >> 12) & 0xFFFu];
    b0[1] = s_cb4096[(w0 >> 24) | ((w1 & 0xFu) << 8)];
    b1[1] = s_cb4096[(w1 >> 4) & 0xFFFu];
    b0[2] = s_cb4096[(w1 >> 16) & 0xFFFu];
    b1[2] = s_cb4096[(w1 >> 28) | ((w2 & 0xFFu) << 4)];
    b0[3] = s_cb4096[(w2 >> 8) & 0xFFFu];
    b1[3] = s_cb4096[w2 >> 20];
}

__device__ __forceinline__ void wv_unpack_vq12_sub4_ldg_tab(
    uint32_t w0, uint32_t w1, uint32_t w2, const uint32_t* cb4096_gmem,
    uint32_t b0[4], uint32_t b1[4]) {
    b0[0] = __ldg(cb4096_gmem + (w0 & 0xFFFu));
    b1[0] = __ldg(cb4096_gmem + ((w0 >> 12) & 0xFFFu));
    b0[1] = __ldg(cb4096_gmem + ((w0 >> 24) | ((w1 & 0xFu) << 8)));
    b1[1] = __ldg(cb4096_gmem + ((w1 >> 4) & 0xFFFu));
    b0[2] = __ldg(cb4096_gmem + ((w1 >> 16) & 0xFFFu));
    b1[2] = __ldg(cb4096_gmem + ((w1 >> 28) | ((w2 & 0xFFu) << 4)));
    b0[3] = __ldg(cb4096_gmem + ((w2 >> 8) & 0xFFFu));
    b1[3] = __ldg(cb4096_gmem + (w2 >> 20));
}

__device__ __forceinline__ void wv_unpack_vq12_meta_mb20(
    const uint16_t* mb20_base, int kc128, int lane_n, float ws_out[4]) {
    constexpr float kInv127 = 1.0f / 127.0f;
    const uint16_t* p = mb20_base + kc128 * 20;
    uint16_t w0  = __ldg(p + lane_n);
    uint16_t w1  = __ldg(p + 8 + lane_n);
    uint32_t hi8 = static_cast<uint32_t>(__ldg(reinterpret_cast<const uint8_t*>(p + 16) + lane_n));
    float ds = wv_fp16_to_f32(w0) * (1.0f / 63.0f);
    ws_out[0] = __half2float(__float2half(ds * static_cast<float>(w1 & 0x3Fu))) * kInv127;
    ws_out[1] = __half2float(__float2half(ds * static_cast<float>((w1 >> 6) & 0x3Fu))) * kInv127;
    ws_out[2] = __half2float(__float2half(ds * static_cast<float>(((w1 >> 12) & 0xFu) | ((hi8 & 0x3u) << 4)))) * kInv127;
    ws_out[3] = __half2float(__float2half(ds * static_cast<float>((hi8 >> 2) & 0x3Fu))) * kInv127;
}

__device__ __forceinline__ void wv_unpack_vq8_h_smem(
    uint32_t pkv, int lane, const uint32_t* s_vq8_uv,
    uint32_t& b0_s0, uint32_t& b1_s0, uint32_t& b0_s1, uint32_t& b1_s1) {
    uint32_t pkv_p = __shfl_xor_sync(0xffffffffu, pkv, 1);
    int is_odd = lane & 1;
    uint32_t pk_s0 = is_odd ? pkv_p : pkv;
    uint32_t pk_s1 = is_odd ? pkv   : pkv_p;
    int sh7 = is_odd ? 0 : 7;
    const uint32_t* tab = s_vq8_uv + (is_odd << 9);
    uint32_t u0 = pk_s0 & 0xFFFFu, u1 = pk_s0 >> 16;
    uint32_t u2 = pk_s1 & 0xFFFFu, u3 = pk_s1 >> 16;
    b0_s0 = tab[((u0 >> 14) << 7) | ((u0 >> sh7) & 0x7Fu)];
    b1_s0 = tab[((u1 >> 14) << 7) | ((u1 >> sh7) & 0x7Fu)];
    b0_s1 = tab[((u2 >> 14) << 7) | ((u2 >> sh7) & 0x7Fu)];
    b1_s1 = tab[((u3 >> 14) << 7) | ((u3 >> sh7) & 0x7Fu)];
}

__device__ __forceinline__ void wv_unpack_vq8_h_ldg_tab(
    uint32_t pkv, int lane, const uint32_t* vq8_uv_gmem,
    uint32_t& b0_s0, uint32_t& b1_s0, uint32_t& b0_s1, uint32_t& b1_s1) {
    uint32_t pkv_p = __shfl_xor_sync(0xffffffffu, pkv, 1);
    int is_odd = lane & 1;
    uint32_t pk_s0 = is_odd ? pkv_p : pkv;
    uint32_t pk_s1 = is_odd ? pkv   : pkv_p;
    int sh7 = is_odd ? 0 : 7;
    const uint32_t* tab = vq8_uv_gmem + (is_odd << 9);
    uint32_t u0 = pk_s0 & 0xFFFFu, u1 = pk_s0 >> 16;
    uint32_t u2 = pk_s1 & 0xFFFFu, u3 = pk_s1 >> 16;
    b0_s0 = __ldg(tab + (((u0 >> 14) << 7) | ((u0 >> sh7) & 0x7Fu)));
    b1_s0 = __ldg(tab + (((u1 >> 14) << 7) | ((u1 >> sh7) & 0x7Fu)));
    b0_s1 = __ldg(tab + (((u2 >> 14) << 7) | ((u2 >> sh7) & 0x7Fu)));
    b1_s1 = __ldg(tab + (((u3 >> 14) << 7) | ((u3 >> sh7) & 0x7Fu)));
}

__device__ __forceinline__ int32_t wv_unpack_wv6_4b38_row_vec4(
    const uint8_t* row_bytes, int b128, int k_sub, const uint4 lut4_tab[8], float& out_ws) {
    const uint8_t* bp = row_bytes + static_cast<size_t>(b128) * 70;
    uint16_t d_sup = *reinterpret_cast<const uint16_t*>(bp);
    uint32_t m     = wv_ld_u32_align2(bp + 2);
    const uint8_t* qs_nib = bp + 6;
    int p    = k_sub >> 3;
    int pos0 = 4 * k_sub;
    uint32_t u6 = (m >> (6 * p)) & 0x3Fu;
    float sc127 = __half2float(__float2half(wv_fp16_to_f32(d_sup) * (static_cast<float>(u6) / 63.0f)));
    out_ws = sc127 * (1.0f / 127.0f);
    uint4 lut16 = lut4_tab[(m >> 24) & 7u];
    uint32_t nib4 = (pos0 < 64)
        ? (wv_ld_u32_align2(qs_nib + pos0) & 0x0F0F0F0Fu)
        : ((wv_ld_u32_align2(qs_nib + pos0 - 64) >> 4) & 0x0F0F0F0Fu);
    return wv_prmt16_lut4(lut16, nib4);
}

} // namespace wv

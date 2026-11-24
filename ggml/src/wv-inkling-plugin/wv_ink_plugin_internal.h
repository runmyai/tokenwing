#pragma once

#include "ggml-quant-plugin.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define WV_INK_GGML_TYPE_F32  0
#define WV_INK_GGML_TYPE_EXT0 (GGML_QUANT_PLUGIN_TYPE_FIRST + 0) // 112: wv6_vq8_sb6   (blck=64,  18 B = 2.2500 bpw; supports 0x0 Plain, 0x4 Repacked, 0xF Hot VQ12)
#define WV_INK_GGML_TYPE_EXT1 (GGML_QUANT_PLUGIN_TYPE_FIRST + 1) // 113: wv6_vq12_3b31 (blck=128, 53 B = 3.3125 bpw)
#define WV_INK_GGML_TYPE_EXT2 (GGML_QUANT_PLUGIN_TYPE_FIRST + 2) // 114: wv6_4b38      (blck=128, 70 B = 4.3750 bpw)
#define WV_INK_GGML_TYPE_EXT3 (GGML_QUANT_PLUGIN_TYPE_FIRST + 3) // 115: wv6_swz8_g32  (blck=32,  34 B = 8.5000 bpw)

#define WV_QWV2_AUX_MAGIC   0x32565751u // "QWV2" little-endian
#define WV_QWV2_AUX_VERSION 2u
#define WV_INK_HOT_TAG_OFF  72          // Byte offset of 16-bit tag word inside every 384-weight (108B) chunk / row

static constexpr float WV_INK_INV_SQRT128 = 0.08838834764831845f;
static constexpr float WV_INK_U6_NORM     = 1.0f / (63.0f * 127.0f); // 1 / 8001.0f
static constexpr float WV_INK_INV_63      = 1.0f / 63.0f;
static constexpr float WV_INK_INV_127     = 1.0f / 127.0f;

#pragma pack(push, 1)

// EXT0 Plain (`tag == 0x0` at chunk byte +72): 64 weights in 18 bytes (2.250 bpw), super-block = 6 blocks (384 weights = 108 B)
struct wv_ink_block_vq8 {
    uint16_t sc_meta;
    uint8_t  qs[16];
};

// EXT0 Repacked (`tag == 0x4` at chunk byte +72): 384 weights (6 blocks) in 108 bytes (lossless bijection with Plain)
//   d_u16:         contiguous FP16 super-scale
//   sc10_packed:   10 x 6-bit unsigned scales u6[2..11] packed in bits [59:0]
//   idx_lo[31]:    first 31 uint16_t 4-Coset VQ8 indices (j = 0..30)
//   tag_and_sc01:  0x4000 | (u6[0] & 0x3F) | ((u6[1] & 0x3F) << 6)  [at exact offset +72!]
//   idx_hi[17]:    remaining 17 uint16_t 4-Coset VQ8 indices (j = 31..47)
struct wv_ink_vq8_repacked_sb6 {
    uint16_t d_u16;
    uint64_t sc10_packed;
    uint16_t idx_lo[31];
    uint16_t tag_and_sc01;
    uint16_t idx_hi[17];
};

// EXT1 (`wv6_vq12_3b31`): 128 weights in 53 bytes (3.3125 bpw)
//   d_super:       FP16 super-scale
//   sc_u6_3b[3]:   4 x 6-bit unsigned scales packed in 24 bits little-endian
//   qs_idx12[48]:  32 x 12-bit cb4096 centroid indices (16 pairs of 3 bytes: b0=ev[7:0], b1=ev[11:8]|(od[3:0]<<4), b2=od[11:4])
struct wv_ink_block_vq12 {
    uint16_t d_super;
    uint8_t  sc_u6_3b[3];
    uint8_t  qs_idx12[48];
};

// EXT2 (`wv6_4b38`): 128 weights in 70 bytes (4.375 bpw)
struct wv_ink_block_4b38 {
    uint16_t d_super;
    uint32_t sc_u6_lut;
    uint8_t  qs_nib[64];
};

// EXT3 (`wv6_swz8_g32`): 32 FWHT-128 rotated int8 weights in 34 bytes (8.500 bpw)
struct wv_ink_block_swz8 {
    uint16_t d127;
    int8_t   qs[32];
};

// QWV2 quant_plugin_aux header (64 bytes)
struct wv_qwv2_aux_header {
    uint32_t magic;       // WV_QWV2_AUX_MAGIC ("QWV2")
    uint32_t version;     // 2
    uint32_t n_layers;    // 24
    uint32_t n_experts;   // 128
    uint64_t total_bytes;
    uint64_t cb4096_off;  // 16,384 B: int8[4096][4]
    uint64_t vq8_uv_off;  // 4,096 B:  int8[2][512][4]
    uint64_t lut4_off;    // 128 B:    int8[8][16]
    uint64_t tmap_off;    // 12,288 B: int32[24][128]
    uint64_t tails_off;   // 4B-aligned hot VQ12 row tails
};

#pragma pack(pop)

static_assert(sizeof(wv_ink_block_vq8) == 18, "wv_ink_block_vq8 must be 18 bytes");
static_assert(sizeof(wv_ink_vq8_repacked_sb6) == 108, "wv_ink_vq8_repacked_sb6 must be 108 bytes");
static_assert(offsetof(wv_ink_vq8_repacked_sb6, tag_and_sc01) == WV_INK_HOT_TAG_OFF, "tag_and_sc01 must sit at +72");
static_assert(sizeof(wv_ink_block_vq12) == 53, "wv_ink_block_vq12 must be 53 bytes");
static_assert(sizeof(wv_ink_block_4b38) == 70, "wv_ink_block_4b38 must be 70 bytes");
static_assert(sizeof(wv_ink_block_swz8) == 34, "wv_ink_block_swz8 must be 34 bytes");
static_assert(sizeof(wv_qwv2_aux_header) == 64, "wv_qwv2_aux_header must be 64 bytes");

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#define WV_INK_HD __host__ __device__ __forceinline__
#else
#define WV_INK_HD static inline
#endif

WV_INK_HD float wv_ink_fp16_to_f32(uint16_t h) {
#if defined(__CUDA_ARCH__)
    return __half2float(__ushort_as_half(h));
#else
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 14;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
#endif
}

// Exact IEEE-754 round-to-nearest-even FP32 -> FP16 -> FP32 matching CUDA `__half2float(__float2half(x))`!
WV_INK_HD uint16_t wv_ink_f32_to_fp16_bits(float x) {
#if defined(__CUDA_ARCH__)
    return __half_as_ushort(__float2half(x));
#else
    uint32_t f;
    std::memcpy(&f, &x, sizeof(uint32_t));
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t)((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = f & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        const int shift = 14 - exp;
        const uint32_t half_bit = 1u << (shift - 1);
        const uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t m16 = mant >> shift;
        if (rem > half_bit || (rem == half_bit && (m16 & 1u))) m16++;
        return (uint16_t)(sign | m16);
    } else if (exp == 0xFF - 127 + 15) {
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    } else {
        const uint32_t rem = mant & 0x1FFFu;
        uint32_t m16 = mant >> 13;
        if (rem > 0x1000u || (rem == 0x1000u && (m16 & 1u))) {
            m16++;
            if (m16 == 0x0400u) { m16 = 0; exp++; }
        }
        if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
        return (uint16_t)(sign | ((uint32_t)exp << 10) | m16);
    }
#endif
}

WV_INK_HD float wv_ink_round_fp16_ws(float ds, uint32_t u6) {
#if defined(__CUDA_ARCH__)
    return __half2float(__float2half(ds * (float)u6)) * WV_INK_INV_127;
#else
    return wv_ink_fp16_to_f32(wv_ink_f32_to_fp16_bits(ds * (float)u6)) * WV_INK_INV_127;
#endif
}

static inline void wv_ink_fwht128_inplace_scalar(float * v) {
    for (int h = 1; h < 128; h <<= 1) {
        for (int i = 0; i < 128; i += (h << 1)) {
            for (int j = i; j < i + h; ++j) {
                const float a = v[j];
                const float b = v[j + h];
                v[j]     = a + b;
                v[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; ++i) {
        v[i] *= WV_INK_INV_SQRT128;
    }
}

// 64B-aligned TLS activation cache slot (`FWHT-128 + Q8_32` + 16-element int16 sums for AVX2 unsigned VPMADDUBSW bias)
struct alignas(64) wv_ink_act_q8_slot {
    const void * src_ptr = nullptr;
    int          n       = 0;
    uint64_t     sig0    = 0;
    uint64_t     sig1    = 0;
    alignas(64) int8_t  q8[16384];
    alignas(64) float   sc[16384 / 32];
    alignas(64) int16_t qsum16[16384 / 16];
};

struct wv_ink_host_tables {
    int8_t         cb4096[4096][4];
    int8_t         vq8_uv[2][512][4];
    int8_t         lut4[8][16];
    const uint8_t* aux_base = nullptr;
    size_t         aux_size = 0;
};

const wv_ink_host_tables * wv_ink_get_host_tables(void);
void wv_ink_set_error(const char * msg);

void wv_ink_cpu_init_dispatch(void);
const wv_ink_act_q8_slot * wv_ink_prepare_act_q8_tls(const float * y, int n);

void wv_ink_to_float_ext0(const void * x, float * y, int64_t k);
void wv_ink_to_float_ext1(const void * x, float * y, int64_t k);
void wv_ink_to_float_ext2(const void * x, float * y, int64_t k);
void wv_ink_to_float_ext3(const void * x, float * y, int64_t k);

ggml_quant_plugin_vec_dot_t    wv_ink_get_vec_dot_ext0(void);
ggml_quant_plugin_vec_dot_x4_t wv_ink_get_vec_dot_x4_ext0(void);
ggml_quant_plugin_vec_dot_t    wv_ink_get_vec_dot_ext1(void);
ggml_quant_plugin_vec_dot_x4_t wv_ink_get_vec_dot_x4_ext1(void);
ggml_quant_plugin_vec_dot_t    wv_ink_get_vec_dot_ext2(void);
ggml_quant_plugin_vec_dot_x4_t wv_ink_get_vec_dot_x4_ext2(void);
ggml_quant_plugin_vec_dot_t    wv_ink_get_vec_dot_ext3(void);
ggml_quant_plugin_vec_dot_x4_t wv_ink_get_vec_dot_x4_ext3(void);

int32_t wv_ink_repack_expert_slot_cpu(
    int32_t ggml_type_id, void * dst_slot, const void * src_plain, int64_t ne0, int64_t ne1);

int32_t wv_ink_vec_dot_x4_m_cpu(
    int32_t ext_idx, int n, int n_rows, int m_cnt, float * const * dst_ptrs,
    const void * vx, size_t bx, const float * const * vy_ptrs, int pf);

#ifdef WV_INK_PLUGIN_ENABLE_CUDA
int32_t wv_ink_cuda_load_aux(const uint8_t * aux_data, size_t aux_size, const wv_ink_host_tables * htab);
void    wv_ink_cuda_free_aux(void);

int32_t wv_ink_mul_mat_cuda(
        int32_t ggml_type_id,
        const void * vx, const float * vy, float * dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        size_t  nb01, size_t  nb02, size_t  nb03,
        int64_t ne11, int64_t ne12, int64_t ne13,
        size_t  nb11, size_t  nb12, size_t  nb13,
        size_t  nb1,  size_t  nb2,  size_t  nb3,
        int32_t device_id, void * cuda_stream);

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
        int32_t device_id, void * cuda_stream);
#endif

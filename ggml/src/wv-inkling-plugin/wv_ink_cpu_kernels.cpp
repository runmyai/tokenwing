#include "wv_ink_plugin_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(__x86_64__)) && !defined(_M_ARM64EC)
#include <immintrin.h>
#include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
#include <cpuid.h>
#include <immintrin.h>
#endif

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64))
#define WV_TARGET_AVX2       __attribute__((target("avx2,fma,f16c")))
#define WV_TARGET_AVX512VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
#define WV_NOINLINE_AVX512   __attribute__((noinline))
#define WV_HAVE_X86_SIMD     1
#define WV_HAVE_X86_AVX512   1
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(__x86_64__)) && !defined(_M_ARM64EC)
#define WV_TARGET_AVX2
#define WV_TARGET_AVX512VNNI
#define WV_NOINLINE_AVX512   __declspec(noinline)
#define WV_HAVE_X86_SIMD     1
#define WV_HAVE_X86_AVX512   1
#else
#define WV_TARGET_AVX2
#define WV_TARGET_AVX512VNNI
#define WV_NOINLINE_AVX512
#define WV_HAVE_X86_SIMD     0
#define WV_HAVE_X86_AVX512   0
#endif

namespace {

static int g_isa_level = 0; // 0 = Scalar, 1 = AVX2, 2 = AVX-512 VNNI

static inline uint16_t rd_u16(const void * p) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

static inline uint32_t rd_u32(const void * p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

static inline uint64_t rd_u64(const void * p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline void compute_act_sig(const float * y, int n, uint64_t & s0, uint64_t & s1) {
    const uint8_t * p = reinterpret_cast<const uint8_t *>(y);
    const int nu = n >> 1;
    const int step = std::max(1, nu >> 4);
    uint64_t a = 0x9e3779b97f4a7c15ULL ^ (uint64_t)n;
    uint64_t b = reinterpret_cast<uintptr_t>(y) ^ 0xbf58476d1ce4e5b9ULL;
    for (int i = 0; i < nu; i += step) {
        const uint64_t v = rd_u64(p + (size_t)i * 8);
        a = (a ^ v) * 0x9e3779b97f4a7c15ULL;
        b = (b ^ (v >> 17)) * 0xbf58476d1ce4e5b9ULL;
        a ^= (b >> 29);
    }
    a ^= rd_u64(p + (size_t)(nu - 1) * 8);
    b ^= rd_u64(p + (size_t)(nu - 2) * 8);
    s0 = mix64(a);
    s1 = mix64(b);
}

static constexpr int kTlsRing = 64;
struct wv_ink_tls_ring {
    wv_ink_act_q8_slot slots[kTlsRing];
    uint32_t           hand = 0;
};
static thread_local wv_ink_tls_ring g_tls_act;

#if WV_HAVE_X86_SIMD
WV_TARGET_AVX2 static void fwht128_avx2(const float * src, float * dst) {
    __m256 v[16];
    for (int i = 0; i < 16; ++i) v[i] = _mm256_loadu_ps(src + 8 * i);
    const __m256 sign_h1 = _mm256_setr_ps(1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f);
    const __m256 sign_h2 = _mm256_setr_ps(1.f, 1.f, -1.f, -1.f, 1.f, 1.f, -1.f, -1.f);
    const __m256 sign_h4 = _mm256_setr_ps(1.f, 1.f, 1.f, 1.f, -1.f, -1.f, -1.f, -1.f);
    for (int i = 0; i < 16; ++i) {
        __m256 p1 = _mm256_permute_ps(v[i], 0xB1);
        v[i] = _mm256_fmadd_ps(v[i], sign_h1, p1);
        __m256 p2 = _mm256_permute_ps(v[i], 0x4E);
        v[i] = _mm256_fmadd_ps(v[i], sign_h2, p2);
        __m256 p4 = _mm256_permute2f128_ps(v[i], v[i], 0x01);
        v[i] = _mm256_fmadd_ps(v[i], sign_h4, p4);
    }
    for (int step = 1; step < 16; step <<= 1) {
        for (int base = 0; base < 16; base += (step << 1)) {
            for (int j = base; j < base + step; ++j) {
                __m256 a = v[j];
                __m256 b = v[j + step];
                v[j]        = _mm256_add_ps(a, b);
                v[j + step] = _mm256_sub_ps(a, b);
            }
        }
    }
    const __m256 scale = _mm256_set1_ps(WV_INK_INV_SQRT128);
    for (int i = 0; i < 16; ++i) {
        _mm256_storeu_ps(dst + 8 * i, _mm256_mul_ps(v[i], scale));
    }
}

WV_TARGET_AVX2 static inline int32_t dot32_i8_avx2(const int8_t * w32, const int8_t * y32, const int16_t * ysum2) {
    (void)ysum2;
    const __m256i w_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w32)));
    const __m256i y_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(y32)));
    const __m256i w_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w32 + 16)));
    const __m256i y_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(y32 + 16)));
    const __m256i p32  = _mm256_add_epi32(_mm256_madd_epi16(w_lo, y_lo), _mm256_madd_epi16(w_hi, y_hi));
    const __m128i lo   = _mm256_castsi256_si128(p32);
    const __m128i hi   = _mm256_extracti128_si256(p32, 1);
    __m128i s4         = _mm_add_epi32(lo, hi);
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(1, 0, 3, 2)));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s4);
}

WV_TARGET_AVX2 static inline void quantize_rot32_avx2(
    const float * rg,
    float & out_sc,
    int8_t * qdst,
    int16_t * qsum2) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 v0 = _mm256_load_ps(rg + 0);
    const __m256 v1 = _mm256_load_ps(rg + 8);
    const __m256 v2 = _mm256_load_ps(rg + 16);
    const __m256 v3 = _mm256_load_ps(rg + 24);

    const __m256 m01 = _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask));
    const __m256 m23 = _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask));
    const __m256 m8  = _mm256_max_ps(m01, m23);
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m8), _mm256_extractf128_ps(m8, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    const float amax = _mm_cvtss_f32(m4);

    out_sc = amax * WV_INK_INV_127;
    const float id = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
    const __m256 vid = _mm256_set1_ps(id);

    const __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vid));
    const __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vid));
    const __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vid));
    const __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vid));

    const __m128i s16_0 = _mm_packs_epi32(_mm256_castsi256_si128(i0), _mm256_extracti128_si256(i0, 1));
    const __m128i s16_1 = _mm_packs_epi32(_mm256_castsi256_si128(i1), _mm256_extracti128_si256(i1, 1));
    const __m128i s16_2 = _mm_packs_epi32(_mm256_castsi256_si128(i2), _mm256_extracti128_si256(i2, 1));
    const __m128i s16_3 = _mm_packs_epi32(_mm256_castsi256_si128(i3), _mm256_extracti128_si256(i3, 1));

    _mm_storeu_si128(reinterpret_cast<__m128i *>(qdst + 0),  _mm_packs_epi16(s16_0, s16_1));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(qdst + 16), _mm_packs_epi16(s16_2, s16_3));

    const __m128i ones16 = _mm_set1_epi16(1);
    __m128i h0 = _mm_madd_epi16(_mm_add_epi16(s16_0, s16_1), ones16);
    __m128i h1 = _mm_madd_epi16(_mm_add_epi16(s16_2, s16_3), ones16);
    h0 = _mm_add_epi32(h0, _mm_shuffle_epi32(h0, _MM_SHUFFLE(1, 0, 3, 2)));
    h0 = _mm_add_epi32(h0, _mm_shuffle_epi32(h0, _MM_SHUFFLE(2, 3, 0, 1)));
    h1 = _mm_add_epi32(h1, _mm_shuffle_epi32(h1, _MM_SHUFFLE(1, 0, 3, 2)));
    h1 = _mm_add_epi32(h1, _mm_shuffle_epi32(h1, _MM_SHUFFLE(2, 3, 0, 1)));
    qsum2[0] = (int16_t)_mm_cvtsi128_si32(h0);
    qsum2[1] = (int16_t)_mm_cvtsi128_si32(h1);
}

WV_TARGET_AVX2 static inline void round_fp16_ws4_avx2(float ds, const uint32_t u6_4[4], float ws4_out[4]) {
    const __m128i iu = _mm_setr_epi32((int)u6_4[0], (int)u6_4[1], (int)u6_4[2], (int)u6_4[3]);
    const __m128  fv = _mm_mul_ps(_mm_cvtepi32_ps(iu), _mm_set1_ps(ds));
    const __m128i h4 = _mm_cvtps_ph(fv, _MM_FROUND_TO_NEAREST_INT);
    const __m128  rv = _mm_mul_ps(_mm_cvtph_ps(h4), _mm_set1_ps(WV_INK_INV_127));
    _mm_storeu_ps(ws4_out, rv);
}

WV_TARGET_AVX2 static inline void round_fp16_ws12_avx2(float ds, const int32_t u6_12[12], float ws12_out[12]) {
    const __m256  vds   = _mm256_set1_ps(ds);
    const __m256  vinv  = _mm256_set1_ps(WV_INK_INV_127);
    const __m256i iu_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(u6_12 + 0));
    const __m128i h8    = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtepi32_ps(iu_lo), vds), _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_ps(ws12_out + 0, _mm256_mul_ps(_mm256_cvtph_ps(h8), vinv));

    const __m128i iu_hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(u6_12 + 8));
    const __m128i h4    = _mm_cvtps_ph(_mm_mul_ps(_mm_cvtepi32_ps(iu_hi), _mm_set1_ps(ds)), _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_ps(ws12_out + 8, _mm_mul_ps(_mm_cvtph_ps(h4), _mm_set1_ps(WV_INK_INV_127)));
}

#if WV_HAVE_X86_AVX512
WV_TARGET_AVX512VNNI WV_NOINLINE_AVX512 static int32_t dot32_i8_vnni(const int8_t * w32, const int8_t * y32, const int16_t * ysum2) {
    const __m256i bias = _mm256_set1_epi8((char)0x80);
    const __m256i wv   = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w32));
    const __m256i yv   = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y32));
    const __m256i wu   = _mm256_xor_si256(wv, bias);
    const __m256i acc  = _mm256_dpbusd_epi32(_mm256_setzero_si256(), wu, yv);
    const __m128i lo   = _mm256_castsi256_si128(acc);
    const __m128i hi   = _mm256_extracti128_si256(acc, 1);
    __m128i s4         = _mm_add_epi32(lo, hi);
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(1, 0, 3, 2)));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(2, 3, 0, 1)));
    const int32_t raw  = _mm_cvtsi128_si32(s4);
    const int32_t ysum = (int32_t)ysum2[0] + (int32_t)ysum2[1];
    return raw - 128 * ysum;
}

// Accumulates `n_groups` (4 or 12) of 32-element int8 dot products for 4 weight rows x up to 16 tokens
// in ZMM registers (`4x4` VNNI register-tiled `vpdpbusd` with zero horizontal reductions, 0.0-ULP exact!).
WV_TARGET_AVX512VNNI WV_NOINLINE_AVX512 static void accum_groups_4xM_vnni(
    const int8_t w4[4][384],
    const float  ws4[4][12],
    int          n_groups,
    int          gid_base,
    int          n_quads,
    const wv_ink_act_q8_slot * const acts_pad[16],
    __m512       z_acc[4]) {
    const __m256i bias80  = _mm256_set1_epi8((char)0x80);
    const __m512i ws_perm = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);

    for (int g = 0; g < n_groups; ++g) {
        const int gid = gid_base + g;
        const __m256i wu0 = _mm256_xor_si256(_mm256_load_si256(reinterpret_cast<const __m256i *>(w4[0] + 32 * g)), bias80);
        const __m256i wu1 = _mm256_xor_si256(_mm256_load_si256(reinterpret_cast<const __m256i *>(w4[1] + 32 * g)), bias80);
        const __m256i wu2 = _mm256_xor_si256(_mm256_load_si256(reinterpret_cast<const __m256i *>(w4[2] + 32 * g)), bias80);
        const __m256i wu3 = _mm256_xor_si256(_mm256_load_si256(reinterpret_cast<const __m256i *>(w4[3] + 32 * g)), bias80);

        __m512i z_lo = _mm512_castsi128_si512(_mm256_castsi256_si128(wu0));
        z_lo = _mm512_inserti32x4(z_lo, _mm256_castsi256_si128(wu1), 1);
        z_lo = _mm512_inserti32x4(z_lo, _mm256_castsi256_si128(wu2), 2);
        z_lo = _mm512_inserti32x4(z_lo, _mm256_castsi256_si128(wu3), 3);

        __m512i z_hi = _mm512_castsi128_si512(_mm256_extracti128_si256(wu0, 1));
        z_hi = _mm512_inserti32x4(z_hi, _mm256_extracti128_si256(wu1, 1), 1);
        z_hi = _mm512_inserti32x4(z_hi, _mm256_extracti128_si256(wu2, 1), 2);
        z_hi = _mm512_inserti32x4(z_hi, _mm256_extracti128_si256(wu3, 1), 3);

        const __m512i zw0 = _mm512_shuffle_epi32(z_lo, (_MM_PERM_ENUM)0x00);
        const __m512i zw1 = _mm512_shuffle_epi32(z_lo, (_MM_PERM_ENUM)0x55);
        const __m512i zw2 = _mm512_shuffle_epi32(z_lo, (_MM_PERM_ENUM)0xAA);
        const __m512i zw3 = _mm512_shuffle_epi32(z_lo, (_MM_PERM_ENUM)0xFF);
        const __m512i zw4 = _mm512_shuffle_epi32(z_hi, (_MM_PERM_ENUM)0x00);
        const __m512i zw5 = _mm512_shuffle_epi32(z_hi, (_MM_PERM_ENUM)0x55);
        const __m512i zw6 = _mm512_shuffle_epi32(z_hi, (_MM_PERM_ENUM)0xAA);
        const __m512i zw7 = _mm512_shuffle_epi32(z_hi, (_MM_PERM_ENUM)0xFF);

        const __m128  v_ws4 = _mm_setr_ps(ws4[0][g], ws4[1][g], ws4[2][g], ws4[3][g]);
        const __m512  z_ws  = _mm512_permutexvar_ps(ws_perm, _mm512_castps128_ps512(v_ws4));

        for (int mq = 0; mq < n_quads; ++mq) {
            const auto * a0 = acts_pad[4 * mq + 0];
            const auto * a1 = acts_pad[4 * mq + 1];
            const auto * a2 = acts_pad[4 * mq + 2];
            const auto * a3 = acts_pad[4 * mq + 3];

            const __m256i ym0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(a0->q8 + 32 * gid));
            const __m256i ym1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(a1->q8 + 32 * gid));
            const __m256i ym2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(a2->q8 + 32 * gid));
            const __m256i ym3 = _mm256_load_si256(reinterpret_cast<const __m256i *>(a3->q8 + 32 * gid));

            const __m256i t0  = _mm256_unpacklo_epi32(ym0, ym1);
            const __m256i t1  = _mm256_unpackhi_epi32(ym0, ym1);
            const __m256i t2  = _mm256_unpacklo_epi32(ym2, ym3);
            const __m256i t3  = _mm256_unpackhi_epi32(ym2, ym3);
            const __m256i u04 = _mm256_unpacklo_epi64(t0, t2);
            const __m256i u15 = _mm256_unpackhi_epi64(t0, t2);
            const __m256i u26 = _mm256_unpacklo_epi64(t1, t3);
            const __m256i u37 = _mm256_unpackhi_epi64(t1, t3);

            __m512i acc_i = _mm512_setzero_si512();
            acc_i = _mm512_dpbusd_epi32(acc_i, zw0, _mm512_broadcast_i32x4(_mm256_castsi256_si128(u04)));
            acc_i = _mm512_dpbusd_epi32(acc_i, zw1, _mm512_broadcast_i32x4(_mm256_castsi256_si128(u15)));
            acc_i = _mm512_dpbusd_epi32(acc_i, zw2, _mm512_broadcast_i32x4(_mm256_castsi256_si128(u26)));
            acc_i = _mm512_dpbusd_epi32(acc_i, zw3, _mm512_broadcast_i32x4(_mm256_castsi256_si128(u37)));
            acc_i = _mm512_dpbusd_epi32(acc_i, zw4, _mm512_broadcast_i32x4(_mm256_extracti128_si256(u04, 1)));
            acc_i = _mm512_dpbusd_epi32(acc_i, zw5, _mm512_broadcast_i32x4(_mm256_extracti128_si256(u15, 1)));
            acc_i = _mm512_dpbusd_epi32(acc_i, zw6, _mm512_broadcast_i32x4(_mm256_extracti128_si256(u26, 1)));
            acc_i = _mm512_dpbusd_epi32(acc_i, zw7, _mm512_broadcast_i32x4(_mm256_extracti128_si256(u37, 1)));

            const int32_t b0 = ((int32_t)a0->qsum16[2 * gid + 0] + (int32_t)a0->qsum16[2 * gid + 1]) << 7;
            const int32_t b1 = ((int32_t)a1->qsum16[2 * gid + 0] + (int32_t)a1->qsum16[2 * gid + 1]) << 7;
            const int32_t b2 = ((int32_t)a2->qsum16[2 * gid + 0] + (int32_t)a2->qsum16[2 * gid + 1]) << 7;
            const int32_t b3 = ((int32_t)a3->qsum16[2 * gid + 0] + (int32_t)a3->qsum16[2 * gid + 1]) << 7;
            const __m512i z_bias = _mm512_broadcast_i32x4(_mm_setr_epi32(b0, b1, b2, b3));

            const __m512 z_dpf = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc_i, z_bias));
            const __m512 z_ysc = _mm512_broadcast_f32x4(_mm_setr_ps(a0->sc[gid], a1->sc[gid], a2->sc[gid], a3->sc[gid]));
            const __m512 z_sc  = _mm512_mul_ps(z_ws, z_ysc);
            z_acc[mq] = _mm512_fmadd_ps(z_dpf, z_sc, z_acc[mq]);
        }
    }
}

WV_TARGET_AVX512VNNI WV_NOINLINE_AVX512 static void store_4xM_vnni(
    const __m512 z_acc[4],
    int m_cnt,
    float * const * dst_ptrs,
    int row_off) {
    const int n_quads = (m_cnt + 3) >> 2;
    for (int mq = 0; mq < n_quads; ++mq) {
        __m128 r0 = _mm512_castps512_ps128(z_acc[mq]);
        __m128 r1 = _mm512_extractf32x4_ps(z_acc[mq], 1);
        __m128 r2 = _mm512_extractf32x4_ps(z_acc[mq], 2);
        __m128 r3 = _mm512_extractf32x4_ps(z_acc[mq], 3);
        _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

        const int m_base = 4 * mq;
        if (m_base + 0 < m_cnt) _mm_storeu_ps(dst_ptrs[m_base + 0] + row_off, r0);
        if (m_base + 1 < m_cnt) _mm_storeu_ps(dst_ptrs[m_base + 1] + row_off, r1);
        if (m_base + 2 < m_cnt) _mm_storeu_ps(dst_ptrs[m_base + 2] + row_off, r2);
        if (m_base + 3 < m_cnt) _mm_storeu_ps(dst_ptrs[m_base + 3] + row_off, r3);
    }
}
#endif
#endif

static inline int32_t dot32_i8_scalar(const int8_t * w32, const int8_t * y32) {
    int32_t s = 0;
    for (int i = 0; i < 32; ++i) s += (int32_t)w32[i] * (int32_t)y32[i];
    return s;
}

// Reconstructs a contiguous 128-weight VQ12 block (or full row) from an EXT0 Hot VQ12 row (`tag == 0xF`)
static inline void materialize_hot_vq12_row(const uint8_t * ext0_row, int K, const wv_ink_host_tables * htab, uint8_t * out_vq12_wire) {
    const int nb = K >> 7;
    const int row_ext0 = (K >> 6) * 18;
    const int in_slot_cap = row_ext0 - 6;
    const int raw_tail_len = nb * 53 - in_slot_cap;
    const uint32_t aux_off = rd_u32(ext0_row);

    std::memcpy(out_vq12_wire, ext0_row + 4, WV_INK_HOT_TAG_OFF - 4);
    std::memcpy(out_vq12_wire + (WV_INK_HOT_TAG_OFF - 4), ext0_row + WV_INK_HOT_TAG_OFF + 2, (size_t)(in_slot_cap - (WV_INK_HOT_TAG_OFF - 4)));
    if (htab->aux_base && (size_t)aux_off + (size_t)raw_tail_len <= htab->aux_size) {
        std::memcpy(out_vq12_wire + in_slot_cap, htab->aux_base + aux_off, (size_t)raw_tail_len);
    } else {
        std::memset(out_vq12_wire + in_slot_cap, 0, (size_t)raw_tail_len);
    }
}

// Unpacks 128 int8 weights + 4 FP32 group scales from a 53-byte `wv_ink_block_vq12`
static inline void unpack_vq12_blk128(
    const wv_ink_block_vq12 & bk,
    const int8_t cb4096[4096][4],
    int8_t * w128_out,
    float ws4_out[4]) {
    const float ds = wv_ink_fp16_to_f32(bk.d_super) * WV_INK_INV_63;
    const uint32_t m24 = (uint32_t)bk.sc_u6_3b[0] | ((uint32_t)bk.sc_u6_3b[1] << 8) | ((uint32_t)bk.sc_u6_3b[2] << 16);
#if WV_HAVE_X86_SIMD
    if (g_isa_level >= 1) {
        const uint32_t u6_4[4] = { m24 & 0x3Fu, (m24 >> 6) & 0x3Fu, (m24 >> 12) & 0x3Fu, (m24 >> 18) & 0x3Fu };
        round_fp16_ws4_avx2(ds, u6_4, ws4_out);
    } else
#endif
    {
        ws4_out[0] = wv_ink_round_fp16_ws(ds, m24 & 0x3Fu);
        ws4_out[1] = wv_ink_round_fp16_ws(ds, (m24 >> 6) & 0x3Fu);
        ws4_out[2] = wv_ink_round_fp16_ws(ds, (m24 >> 12) & 0x3Fu);
        ws4_out[3] = wv_ink_round_fp16_ws(ds, (m24 >> 18) & 0x3Fu);
    }

    uint32_t * dst32 = reinterpret_cast<uint32_t *>(w128_out);
    const uint32_t * cb32 = reinterpret_cast<const uint32_t *>(cb4096);
    for (int p = 0; p < 16; ++p) {
        const uint8_t b0 = bk.qs_idx12[3 * p + 0];
        const uint8_t b1 = bk.qs_idx12[3 * p + 1];
        const uint8_t b2 = bk.qs_idx12[3 * p + 2];
        const uint32_t i_ev = (uint32_t)b0 | (((uint32_t)b1 & 0x0Fu) << 8);
        const uint32_t i_od = ((uint32_t)b1 >> 4) | ((uint32_t)b2 << 4);
        dst32[2 * p + 0] = cb32[i_ev];
        dst32[2 * p + 1] = cb32[i_od];
    }
}

// Unpacks 384 int8 weights + 12 FP32 group scales from a 108-byte Plain (`0x0`) or Repacked (`0x4`) VQ8_SB6 super-chunk
static inline void unpack_vq8_sb6_chunk384(
    const uint8_t * chunk108,
    const int8_t vq8_uv[2][512][4],
    int8_t * w384_out,
    float ws12_out[12]) {
    const uint16_t tag_w = rd_u16(chunk108 + WV_INK_HOT_TAG_OFF);
    const uint32_t tag   = tag_w >> 12;
    const uint32_t * tab_u = reinterpret_cast<const uint32_t *>(vq8_uv[0]);
    const uint32_t * tab_v = reinterpret_cast<const uint32_t *>(vq8_uv[1]);
    uint32_t * dst32 = reinterpret_cast<uint32_t *>(w384_out);

    if (tag == 0x4u) {
        const auto & rc = *reinterpret_cast<const wv_ink_vq8_repacked_sb6 *>(chunk108);
        const float ds  = wv_ink_fp16_to_f32(rc.d_u16) * WV_INK_INV_63;
#if WV_HAVE_X86_SIMD
        if (g_isa_level >= 1) {
            int32_t u6_12[12];
            u6_12[0] = (int32_t)(rc.tag_and_sc01 & 0x3Fu);
            u6_12[1] = (int32_t)((rc.tag_and_sc01 >> 6) & 0x3Fu);
            uint64_t s10 = rc.sc10_packed;
            for (int g = 2; g < 12; ++g) {
                u6_12[g] = (int32_t)(s10 & 0x3Full);
                s10 >>= 6;
            }
            round_fp16_ws12_avx2(ds, u6_12, ws12_out);
        } else
#endif
        {
            ws12_out[0] = wv_ink_round_fp16_ws(ds, rc.tag_and_sc01 & 0x3Fu);
            ws12_out[1] = wv_ink_round_fp16_ws(ds, (rc.tag_and_sc01 >> 6) & 0x3Fu);
            uint64_t s10 = rc.sc10_packed;
            for (int g = 2; g < 12; ++g) {
                ws12_out[g] = wv_ink_round_fp16_ws(ds, (uint32_t)(s10 & 0x3Full));
                s10 >>= 6;
            }
        }
        for (int j = 0; j < 31; ++j) {
            const uint32_t u = rc.idx_lo[j];
            const uint32_t c_hi = (u >> 14) << 7;
            dst32[2 * j + 0] = tab_u[c_hi | ((u >> 7) & 0x7Fu)];
            dst32[2 * j + 1] = tab_v[c_hi | (u & 0x7Fu)];
        }
        for (int j = 0; j < 17; ++j) {
            const uint32_t u = rc.idx_hi[j];
            const uint32_t c_hi = (u >> 14) << 7;
            dst32[2 * (31 + j) + 0] = tab_u[c_hi | ((u >> 7) & 0x7Fu)];
            dst32[2 * (31 + j) + 1] = tab_v[c_hi | (u & 0x7Fu)];
        }
    } else {
        const auto * b = reinterpret_cast<const wv_ink_block_vq8 *>(chunk108);
        const uint16_t d_u16 = (uint16_t)(((b[0].sc_meta >> 12) & 0xFu)
                             | (((b[1].sc_meta >> 12) & 0xFu) << 4)
                             | (((b[2].sc_meta >> 12) & 0xFu) << 8)
                             | (((b[3].sc_meta >> 12) & 0xFu) << 12));
        const float ds = wv_ink_fp16_to_f32(d_u16) * WV_INK_INV_63;
#if WV_HAVE_X86_SIMD
        if (g_isa_level >= 1) {
            int32_t u6_12[12];
            for (int s = 0; s < 6; ++s) {
                u6_12[2 * s + 0] = (int32_t)(b[s].sc_meta & 0x3Fu);
                u6_12[2 * s + 1] = (int32_t)((b[s].sc_meta >> 6) & 0x3Fu);
                const uint16_t * q16 = reinterpret_cast<const uint16_t *>(b[s].qs);
                for (int j = 0; j < 8; ++j) {
                    const uint32_t u = q16[j];
                    const uint32_t c_hi = (u >> 14) << 7;
                    dst32[16 * s + 2 * j + 0] = tab_u[c_hi | ((u >> 7) & 0x7Fu)];
                    dst32[16 * s + 2 * j + 1] = tab_v[c_hi | (u & 0x7Fu)];
                }
            }
            round_fp16_ws12_avx2(ds, u6_12, ws12_out);
        } else
#endif
        {
            for (int s = 0; s < 6; ++s) {
                ws12_out[2 * s + 0] = wv_ink_round_fp16_ws(ds, b[s].sc_meta & 0x3Fu);
                ws12_out[2 * s + 1] = wv_ink_round_fp16_ws(ds, (b[s].sc_meta >> 6) & 0x3Fu);
                const uint16_t * q16 = reinterpret_cast<const uint16_t *>(b[s].qs);
                for (int j = 0; j < 8; ++j) {
                    const uint32_t u = q16[j];
                    const uint32_t c_hi = (u >> 14) << 7;
                    dst32[16 * s + 2 * j + 0] = tab_u[c_hi | ((u >> 7) & 0x7Fu)];
                    dst32[16 * s + 2 * j + 1] = tab_v[c_hi | (u & 0x7Fu)];
                }
            }
        }
    }
}

static inline float dot_vq12_row(const wv_ink_block_vq12 * bks, int nb, const wv_ink_act_q8_slot * act, const wv_ink_host_tables * htab, int isa) {
    alignas(64) int8_t w128[128];
    float ws[4];
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        unpack_vq12_blk128(bks[b], htab->cb4096, w128, ws);
        for (int g = 0; g < 4; ++g) {
            const int gid = 4 * b + g;
            int32_t dp = 0;
#if WV_HAVE_X86_SIMD
#if WV_HAVE_X86_AVX512
            if (isa >= 2) {
                dp = dot32_i8_vnni(w128 + 32 * g, act->q8 + 32 * gid, act->qsum16 + 2 * gid);
            } else
#endif
            if (isa >= 1) {
                dp = dot32_i8_avx2(w128 + 32 * g, act->q8 + 32 * gid, act->qsum16 + 2 * gid);
            } else
#endif
            {
                (void)isa;
                dp = dot32_i8_scalar(w128 + 32 * g, act->q8 + 32 * gid);
            }
            const float sc = ws[g] * act->sc[gid];
            sum = std::fmaf((float)dp, sc, sum);
        }
    }
    return sum;
}

static inline float dot_ext0_single_row(const uint8_t * row_p, int n, const wv_ink_act_q8_slot * act, const wv_ink_host_tables * htab, int isa) {
    const uint16_t tag_w = rd_u16(row_p + WV_INK_HOT_TAG_OFF);
    if ((tag_w >> 12) == 0xFu) {
        alignas(64) uint8_t vq12_wire[64 * 53]; // up to K=8192
        materialize_hot_vq12_row(row_p, n, htab, vq12_wire);
        return dot_vq12_row(reinterpret_cast<const wv_ink_block_vq12 *>(vq12_wire), n >> 7, act, htab, isa);
    }
    alignas(64) int8_t w384[384];
    float ws[12];
    float sum = 0.0f;
    const int n_chunks = n / 384;
    for (int c = 0; c < n_chunks; ++c) {
        unpack_vq8_sb6_chunk384(row_p + (size_t)c * 108, htab->vq8_uv, w384, ws);
        for (int g = 0; g < 12; ++g) {
            const int gid = 12 * c + g;
            int32_t dp = 0;
#if WV_HAVE_X86_SIMD
#if WV_HAVE_X86_AVX512
            if (isa >= 2) {
                dp = dot32_i8_vnni(w384 + 32 * g, act->q8 + 32 * gid, act->qsum16 + 2 * gid);
            } else
#endif
            if (isa >= 1) {
                dp = dot32_i8_avx2(w384 + 32 * g, act->q8 + 32 * gid, act->qsum16 + 2 * gid);
            } else
#endif
            {
                dp = dot32_i8_scalar(w384 + 32 * g, act->q8 + 32 * gid);
            }
            const float sc = ws[g] * act->sc[gid];
            sum = std::fmaf((float)dp, sc, sum);
        }
    }
    return sum;
}

// Fast 4-row `EXT0` (`wv6_vq8_sb6`) Repacked/Plain SIMD kernel sharing 1 activation broadcast across 4 rows!
static void vec_dot_ext0_impl(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    const auto * act  = wv_ink_prepare_act_q8_tls(static_cast<const float *>(vy), n);
    const auto * htab = wv_ink_get_host_tables();
    *s = dot_ext0_single_row(static_cast<const uint8_t *>(vx), n, act, htab, g_isa_level);
}

static void vec_dot_x4_ext0_impl(int n, float * s, const void * vx, size_t bx, const void * vy, int width, int pf) {
    (void)width;
    const auto * act  = wv_ink_prepare_act_q8_tls(static_cast<const float *>(vy), n);
    const auto * htab = wv_ink_get_host_tables();
    const uint8_t * r0 = static_cast<const uint8_t *>(vx);
    const uint8_t * r1 = r0 + bx;
    const uint8_t * r2 = r1 + bx;
    const uint8_t * r3 = r2 + bx;

#if WV_HAVE_X86_SIMD
    if (pf != 0) {
        const size_t pf_off = (pf < 0) ? (4 * bx) : (size_t)pf * 108;
        _mm_prefetch(reinterpret_cast<const char *>(r0 + pf_off), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(r1 + pf_off), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(r2 + pf_off), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(r3 + pf_off), _MM_HINT_T0);
    }
#else
    (void)pf;
#endif

    // Check if any of the 4 rows is a Hot VQ12 row (`0xF`):
    const uint16_t t0 = rd_u16(r0 + WV_INK_HOT_TAG_OFF) >> 12;
    const uint16_t t1 = rd_u16(r1 + WV_INK_HOT_TAG_OFF) >> 12;
    const uint16_t t2 = rd_u16(r2 + WV_INK_HOT_TAG_OFF) >> 12;
    const uint16_t t3 = rd_u16(r3 + WV_INK_HOT_TAG_OFF) >> 12;
    if (t0 == 0xFu || t1 == 0xFu || t2 == 0xFu || t3 == 0xFu) {
        s[0] = dot_ext0_single_row(r0, n, act, htab, g_isa_level);
        s[1] = dot_ext0_single_row(r1, n, act, htab, g_isa_level);
        s[2] = dot_ext0_single_row(r2, n, act, htab, g_isa_level);
        s[3] = dot_ext0_single_row(r3, n, act, htab, g_isa_level);
        return;
    }

    alignas(64) int8_t w0[384], w1[384], w2[384], w3[384];
    float ws0[12], ws1[12], ws2[12], ws3[12];
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int n_chunks = n / 384;
    const int isa = g_isa_level;

    for (int c = 0; c < n_chunks; ++c) {
        const size_t off = (size_t)c * 108;
        unpack_vq8_sb6_chunk384(r0 + off, htab->vq8_uv, w0, ws0);
        unpack_vq8_sb6_chunk384(r1 + off, htab->vq8_uv, w1, ws1);
        unpack_vq8_sb6_chunk384(r2 + off, htab->vq8_uv, w2, ws2);
        unpack_vq8_sb6_chunk384(r3 + off, htab->vq8_uv, w3, ws3);

        for (int g = 0; g < 12; ++g) {
            const int gid = 12 * c + g;
            const float ysc = act->sc[gid];
            const int8_t * yq = act->q8 + 32 * gid;
            int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if WV_HAVE_X86_SIMD
            const int16_t * ys = act->qsum16 + 2 * gid;
#if WV_HAVE_X86_AVX512
            if (isa >= 2) {
                d0 = dot32_i8_vnni(w0 + 32 * g, yq, ys);
                d1 = dot32_i8_vnni(w1 + 32 * g, yq, ys);
                d2 = dot32_i8_vnni(w2 + 32 * g, yq, ys);
                d3 = dot32_i8_vnni(w3 + 32 * g, yq, ys);
            } else
#endif
            if (isa >= 1) {
                d0 = dot32_i8_avx2(w0 + 32 * g, yq, ys);
                d1 = dot32_i8_avx2(w1 + 32 * g, yq, ys);
                d2 = dot32_i8_avx2(w2 + 32 * g, yq, ys);
                d3 = dot32_i8_avx2(w3 + 32 * g, yq, ys);
            } else
#endif
            {
                (void)isa;
                d0 = dot32_i8_scalar(w0 + 32 * g, yq);
                d1 = dot32_i8_scalar(w1 + 32 * g, yq);
                d2 = dot32_i8_scalar(w2 + 32 * g, yq);
                d3 = dot32_i8_scalar(w3 + 32 * g, yq);
            }
            acc0 = std::fmaf((float)d0, ws0[g] * ysc, acc0);
            acc1 = std::fmaf((float)d1, ws1[g] * ysc, acc1);
            acc2 = std::fmaf((float)d2, ws2[g] * ysc, acc2);
            acc3 = std::fmaf((float)d3, ws3[g] * ysc, acc3);
        }
    }
    s[0] = acc0; s[1] = acc1; s[2] = acc2; s[3] = acc3;
}

static void vec_dot_ext1_impl(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    const auto * act  = wv_ink_prepare_act_q8_tls(static_cast<const float *>(vy), n);
    const auto * htab = wv_ink_get_host_tables();
    *s = dot_vq12_row(static_cast<const wv_ink_block_vq12 *>(vx), n >> 7, act, htab, g_isa_level);
}

static void vec_dot_x4_ext1_impl(int n, float * s, const void * vx, size_t bx, const void * vy, int width, int pf) {
    (void)width; (void)pf;
    const auto * act  = wv_ink_prepare_act_q8_tls(static_cast<const float *>(vy), n);
    const auto * htab = wv_ink_get_host_tables();
    const uint8_t * base = static_cast<const uint8_t *>(vx);
    for (int r = 0; r < 4; ++r) {
        s[r] = dot_vq12_row(reinterpret_cast<const wv_ink_block_vq12 *>(base + r * bx), n >> 7, act, htab, g_isa_level);
    }
}

static inline void unpack_4b38_blk128(const wv_ink_block_4b38 & bk, const int8_t lut4[8][16], int8_t * w128_out, float ws4_out[4]) {
    const float ds = wv_ink_fp16_to_f32(bk.d_super);
    const uint32_t m = bk.sc_u6_lut;
    const int8_t * bank = lut4[(m >> 24) & 7u];
    for (int p = 0; p < 4; ++p) {
        ws4_out[p] = ds * (WV_INK_U6_NORM * (float)((m >> (6 * p)) & 0x3Fu));
    }
    for (int j = 0; j < 64; ++j) {
        const uint8_t b = bk.qs_nib[j];
        w128_out[j]      = bank[b & 0x0Fu];
        w128_out[j + 64] = bank[b >> 4];
    }
}

static void vec_dot_ext2_impl(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    const auto * act  = wv_ink_prepare_act_q8_tls(static_cast<const float *>(vy), n);
    const auto * htab = wv_ink_get_host_tables();
    const auto * bks  = static_cast<const wv_ink_block_4b38 *>(vx);
    alignas(64) int8_t w128[128];
    float ws[4];
    float sum = 0.0f;
    const int nb = n >> 7;
    for (int b = 0; b < nb; ++b) {
        unpack_4b38_blk128(bks[b], htab->lut4, w128, ws);
        for (int g = 0; g < 4; ++g) {
            const int gid = 4 * b + g;
            const int32_t dp = dot32_i8_scalar(w128 + 32 * g, act->q8 + 32 * gid);
            const float sc = ws[g] * act->sc[gid];
            sum = std::fmaf((float)dp, sc, sum);
        }
    }
    *s = sum;
}

static void vec_dot_x4_ext2_impl(int n, float * s, const void * vx, size_t bx, const void * vy, int width, int pf) {
    (void)width; (void)pf;
    const uint8_t * base = static_cast<const uint8_t *>(vx);
    for (int r = 0; r < 4; ++r) {
        vec_dot_ext2_impl(n, s + r, 0, base + r * bx, 0, vy, 0, 1);
    }
}

static inline float dot_swz8_row(const wv_ink_block_swz8 * bks, int k32, const wv_ink_act_q8_slot * act, int isa) {
    float sum = 0.0f;
    for (int g = 0; g < k32; ++g) {
        const float ws = wv_ink_fp16_to_f32(bks[g].d127) * WV_INK_INV_127;
        int32_t dp = 0;
#if WV_HAVE_X86_SIMD
#if WV_HAVE_X86_AVX512
        if (isa >= 2) {
            dp = dot32_i8_vnni(bks[g].qs, act->q8 + 32 * g, act->qsum16 + 2 * g);
        } else
#endif
        if (isa >= 1) {
            dp = dot32_i8_avx2(bks[g].qs, act->q8 + 32 * g, act->qsum16 + 2 * g);
        } else
#endif
        {
            (void)isa;
            dp = dot32_i8_scalar(bks[g].qs, act->q8 + 32 * g);
        }
        const float sc = ws * act->sc[g];
        sum = std::fmaf((float)dp, sc, sum);
    }
    return sum;
}

static void vec_dot_ext3_impl(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    const auto * act = wv_ink_prepare_act_q8_tls(static_cast<const float *>(vy), n);
    *s = dot_swz8_row(static_cast<const wv_ink_block_swz8 *>(vx), n >> 5, act, g_isa_level);
}

static void vec_dot_x4_ext3_impl(int n, float * s, const void * vx, size_t bx, const void * vy, int width, int pf) {
    (void)width; (void)pf;
    const auto * act = wv_ink_prepare_act_q8_tls(static_cast<const float *>(vy), n);
    const int k32 = n >> 5;
    const int isa = g_isa_level;
    const uint8_t * base = static_cast<const uint8_t *>(vx);
    const auto * b0 = reinterpret_cast<const wv_ink_block_swz8 *>(base + 0 * bx);
    const auto * b1 = reinterpret_cast<const wv_ink_block_swz8 *>(base + 1 * bx);
    const auto * b2 = reinterpret_cast<const wv_ink_block_swz8 *>(base + 2 * bx);
    const auto * b3 = reinterpret_cast<const wv_ink_block_swz8 *>(base + 3 * bx);

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int g = 0; g < k32; ++g) {
        const float ysc = act->sc[g];
        const int8_t * yq = act->q8 + 32 * g;
        int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if WV_HAVE_X86_SIMD
        const int16_t * ys = act->qsum16 + 2 * g;
#if WV_HAVE_X86_AVX512
        if (isa >= 2) {
            d0 = dot32_i8_vnni(b0[g].qs, yq, ys);
            d1 = dot32_i8_vnni(b1[g].qs, yq, ys);
            d2 = dot32_i8_vnni(b2[g].qs, yq, ys);
            d3 = dot32_i8_vnni(b3[g].qs, yq, ys);
        } else
#endif
        if (isa >= 1) {
            d0 = dot32_i8_avx2(b0[g].qs, yq, ys);
            d1 = dot32_i8_avx2(b1[g].qs, yq, ys);
            d2 = dot32_i8_avx2(b2[g].qs, yq, ys);
            d3 = dot32_i8_avx2(b3[g].qs, yq, ys);
        } else
#endif
        {
            (void)isa;
            d0 = dot32_i8_scalar(b0[g].qs, yq);
            d1 = dot32_i8_scalar(b1[g].qs, yq);
            d2 = dot32_i8_scalar(b2[g].qs, yq);
            d3 = dot32_i8_scalar(b3[g].qs, yq);
        }
        a0 = std::fmaf((float)d0, (wv_ink_fp16_to_f32(b0[g].d127) * WV_INK_INV_127) * ysc, a0);
        a1 = std::fmaf((float)d1, (wv_ink_fp16_to_f32(b1[g].d127) * WV_INK_INV_127) * ysc, a1);
        a2 = std::fmaf((float)d2, (wv_ink_fp16_to_f32(b2[g].d127) * WV_INK_INV_127) * ysc, a2);
        a3 = std::fmaf((float)d3, (wv_ink_fp16_to_f32(b3[g].d127) * WV_INK_INV_127) * ysc, a3);
    }
    s[0] = a0; s[1] = a1; s[2] = a2; s[3] = a3;
}

} // namespace

void wv_ink_cpu_init_dispatch(void) {
    int max_hw_isa = 0;
#if WV_HAVE_X86_SIMD
#if defined(_MSC_VER)
    int info1[4] = {0};
    int info7[4] = {0};
    __cpuid(info1, 1);
    __cpuidex(info7, 7, 0);
    const bool has_osxsave  = (info1[2] & (1 << 27)) != 0;
    const uint64_t xcr0     = has_osxsave ? _xgetbv(0) : 0ull;
    const bool os_avx_ok    = (xcr0 & 0x6ull) == 0x6ull;
    const bool os_avx512_ok = (xcr0 & 0xE6ull) == 0xE6ull;
    const bool has_avx2     = os_avx_ok && ((info7[1] & (1 << 5)) != 0) && ((info1[2] & (1 << 12)) != 0) && ((info1[2] & (1 << 29)) != 0);
    const bool has_avx512f  = os_avx512_ok && ((info7[1] & (1 << 16)) != 0);
    const bool has_avx512bw = os_avx512_ok && ((info7[1] & (1 << 30)) != 0);
    const bool has_avx512vl = os_avx512_ok && ((info7[1] & (1u << 31)) != 0);
    const bool has_vnni     = os_avx512_ok && ((info7[2] & (1 << 11)) != 0);
    if (has_avx2) max_hw_isa = 1;
    if (has_avx2 && has_avx512f && has_avx512bw && has_avx512vl && has_vnni) max_hw_isa = 2;
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) {
        max_hw_isa = 1;
    }
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni") &&
        __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) {
        max_hw_isa = 2;
    }
#endif
#endif
    g_isa_level = max_hw_isa;
    const char * force_isa = std::getenv("WV_INK_CPU_ISA");
    if (force_isa && force_isa[0]) {
        g_isa_level = std::clamp(std::atoi(force_isa), 0, max_hw_isa);
    }
}

const wv_ink_act_q8_slot * wv_ink_prepare_act_q8_tls(const float * y, int n) {
    uint64_t s0 = 0, s1 = 0;
    compute_act_sig(y, n, s0, s1);
    const uint32_t cur_h = g_tls_act.hand;
    for (int i = 0; i < kTlsRing; ++i) {
        const wv_ink_act_q8_slot & sl = g_tls_act.slots[(cur_h - 1u - (uint32_t)i) & (kTlsRing - 1)];
        if (sl.src_ptr == y && sl.n == n && sl.sig0 == s0 && sl.sig1 == s1) {
            return &sl;
        }
    }
    wv_ink_act_q8_slot & dst = g_tls_act.slots[g_tls_act.hand++ & (kTlsRing - 1)];
    dst.src_ptr = y;
    dst.n       = n;
    dst.sig0    = s0;
    dst.sig1    = s1;

    alignas(64) float rot_buf[128];
    const int nb  = n >> 7;
    const int isa = g_isa_level;
    for (int b = 0; b < nb; ++b) {
#if WV_HAVE_X86_SIMD
        if (isa >= 1) {
            fwht128_avx2(y + 128 * b, rot_buf);
            for (int g = 0; g < 4; ++g) {
                const int gid = 4 * b + g;
                quantize_rot32_avx2(rot_buf + 32 * g, dst.sc[gid], dst.q8 + 32 * gid, dst.qsum16 + 2 * gid);
            }
            continue;
        }
#else
        (void)isa;
#endif
        std::memcpy(rot_buf, y + 128 * b, 128 * sizeof(float));
        wv_ink_fwht128_inplace_scalar(rot_buf);
        for (int g = 0; g < 4; ++g) {
            const float * rg = rot_buf + 32 * g;
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) {
                float av = std::fabs(rg[j]);
                if (av > amax) amax = av;
            }
            const float d  = amax * WV_INK_INV_127;
            const float id = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
            const int gid  = 4 * b + g;
            dst.sc[gid]    = d;
            int8_t * qdst  = dst.q8 + 32 * gid;
            int sum0 = 0, sum1 = 0;
            for (int j = 0; j < 16; ++j) {
                const int qv0 = (int)std::nearbyintf(rg[j] * id);
                const int qv1 = (int)std::nearbyintf(rg[j + 16] * id);
                qdst[j]      = (int8_t)qv0;
                qdst[j + 16] = (int8_t)qv1;
                sum0 += qv0;
                sum1 += qv1;
            }
            dst.qsum16[2 * gid + 0] = (int16_t)sum0;
            dst.qsum16[2 * gid + 1] = (int16_t)sum1;
        }
    }
    return &dst;
}

// `to_float` dequantizers (convert rotated weights back to un-rotated spatial domain via self-inverse FWHT-128!)
void wv_ink_to_float_ext0(const void * x, float * y, int64_t k) {
    const auto * htab = wv_ink_get_host_tables();
    const uint8_t * p = static_cast<const uint8_t *>(x);
    const int n = (int)k;
    const uint16_t tag_w = rd_u16(p + WV_INK_HOT_TAG_OFF);
    if ((tag_w >> 12) == 0xFu) {
        alignas(64) uint8_t vq12_wire[64 * 53];
        materialize_hot_vq12_row(p, n, htab, vq12_wire);
        wv_ink_to_float_ext1(vq12_wire, y, k);
        return;
    }
    alignas(64) int8_t w384[384];
    float ws[12];
    const int n_chunks = n / 384;
    for (int c = 0; c < n_chunks; ++c) {
        unpack_vq8_sb6_chunk384(p + (size_t)c * 108, htab->vq8_uv, w384, ws);
        float * yc = y + 384 * c;
        for (int g = 0; g < 12; ++g) {
            for (int j = 0; j < 32; ++j) {
                yc[32 * g + j] = (float)w384[32 * g + j] * ws[g];
            }
        }
        for (int b = 0; b < 3; ++b) {
            wv_ink_fwht128_inplace_scalar(yc + 128 * b);
        }
    }
}

void wv_ink_to_float_ext1(const void * x, float * y, int64_t k) {
    const auto * htab = wv_ink_get_host_tables();
    const auto * bks  = static_cast<const wv_ink_block_vq12 *>(x);
    alignas(64) int8_t w128[128];
    float ws[4];
    const int nb = (int)(k >> 7);
    for (int b = 0; b < nb; ++b) {
        unpack_vq12_blk128(bks[b], htab->cb4096, w128, ws);
        float * yb = y + 128 * b;
        for (int g = 0; g < 4; ++g) {
            for (int j = 0; j < 32; ++j) {
                yb[32 * g + j] = (float)w128[32 * g + j] * ws[g];
            }
        }
        wv_ink_fwht128_inplace_scalar(yb);
    }
}

void wv_ink_to_float_ext2(const void * x, float * y, int64_t k) {
    const auto * htab = wv_ink_get_host_tables();
    const auto * bks  = static_cast<const wv_ink_block_4b38 *>(x);
    alignas(64) int8_t w128[128];
    float ws[4];
    const int nb = (int)(k >> 7);
    for (int b = 0; b < nb; ++b) {
        unpack_4b38_blk128(bks[b], htab->lut4, w128, ws);
        float * yb = y + 128 * b;
        for (int g = 0; g < 4; ++g) {
            for (int j = 0; j < 32; ++j) {
                yb[32 * g + j] = (float)w128[32 * g + j] * ws[g];
            }
        }
        wv_ink_fwht128_inplace_scalar(yb);
        // Exact Inkling-2 embed_gather_norm BF16 intermediate round (`wv_inkling_types.cuh:81-84`)
        for (int j = 0; j < 128; ++j) {
            uint32_t bits;
            std::memcpy(&bits, &yb[j], sizeof(bits));
            bits = (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
            std::memcpy(&yb[j], &bits, sizeof(bits));
        }
    }
}

void wv_ink_to_float_ext3(const void * x, float * y, int64_t k) {
    const auto * bks = static_cast<const wv_ink_block_swz8 *>(x);
    const int nb = (int)(k >> 7);
    for (int b = 0; b < nb; ++b) {
        float * yb = y + 128 * b;
        for (int g = 0; g < 4; ++g) {
            const auto & bk = bks[4 * b + g];
            const float ws  = wv_ink_fp16_to_f32(bk.d127) * WV_INK_INV_127;
            for (int j = 0; j < 32; ++j) {
                yb[32 * g + j] = (float)bk.qs[j] * ws;
            }
        }
        wv_ink_fwht128_inplace_scalar(yb);
    }
}

int32_t wv_ink_repack_expert_slot_cpu(
    int32_t ggml_type_id, void * dst_slot, const void * src_plain, int64_t ne0, int64_t ne1) {
    if (dst_slot == nullptr && src_plain == nullptr) {
        if (ne0 == -1 && ne1 == 0) {
            return (ggml_type_id == WV_INK_GGML_TYPE_EXT0) ? 1 : 0; // CUDA reads repacked EXT0!
        }
        if (ggml_type_id == WV_INK_GGML_TYPE_EXT0 && ne0 > 0 && (ne0 % 384 == 0) && ne1 > 0) {
            return 0;
        }
        return -1;
    }
    if (ggml_type_id != WV_INK_GGML_TYPE_EXT0 || ne0 <= 0 || (ne0 % 384 != 0) || ne1 <= 0) {
        return -1;
    }

    const size_t row_bytes = (size_t)(ne0 / 64) * 18;
    const int n_chunks = (int)(ne0 / 384);
    for (int64_t r = 0; r < ne1; ++r) {
        const uint8_t * src_r = static_cast<const uint8_t *>(src_plain) + (size_t)r * row_bytes;
        uint8_t *       dst_r = static_cast<uint8_t *>(dst_slot) + (size_t)r * row_bytes;
        const uint16_t  tag_w = rd_u16(src_r + WV_INK_HOT_TAG_OFF);
        const uint32_t  tag   = tag_w >> 12;
        if (tag == 0xFu || tag == 0x4u) {
            if (dst_r != src_r) {
                std::memcpy(dst_r, src_r, row_bytes);
            }
            continue;
        }
        for (int c = 0; c < n_chunks; ++c) {
            wv_ink_block_vq8 b[6];
            std::memcpy(b, src_r + (size_t)c * 108, 108);
            wv_ink_vq8_repacked_sb6 out;
            out.d_u16 = (uint16_t)(((b[0].sc_meta >> 12) & 0xFu)
                      | (((b[1].sc_meta >> 12) & 0xFu) << 4)
                      | (((b[2].sc_meta >> 12) & 0xFu) << 8)
                      | (((b[3].sc_meta >> 12) & 0xFu) << 12));
            uint32_t u6[12];
            uint16_t idx48[48];
            for (int s = 0; s < 6; ++s) {
                u6[2 * s + 0] = b[s].sc_meta & 0x3Fu;
                u6[2 * s + 1] = (b[s].sc_meta >> 6) & 0x3Fu;
                std::memcpy(idx48 + 8 * s, b[s].qs, 16);
            }
            uint64_t sc10 = 0;
            for (int g = 11; g >= 2; --g) {
                sc10 = (sc10 << 6) | (uint64_t)(u6[g] & 0x3Fu);
            }
            out.sc10_packed  = sc10;
            out.tag_and_sc01 = (uint16_t)(0x4000u | (u6[0] & 0x3Fu) | ((u6[1] & 0x3Fu) << 6));
            std::memcpy(out.idx_lo, idx48, 31 * sizeof(uint16_t));
            std::memcpy(out.idx_hi, idx48 + 31, 17 * sizeof(uint16_t));
            std::memcpy(dst_r + (size_t)c * 108, &out, 108);
        }
    }
    return 0;
}

#if WV_HAVE_X86_SIMD && WV_HAVE_X86_AVX512
WV_TARGET_AVX512VNNI WV_NOINLINE_AVX512 static void run_x4_m_vnni_driver(
    int32_t ext_idx,
    int n,
    int m_cnt,
    float * const * dst_ptrs,
    int row_off,
    const uint8_t * const rr[4],
    const wv_ink_act_q8_slot * const acts_pad[16],
    const wv_ink_host_tables * htab) {
    const int n_quads = (m_cnt + 3) >> 2;
    __m512 z_acc[4] = {
        _mm512_setzero_ps(),
        _mm512_setzero_ps(),
        _mm512_setzero_ps(),
        _mm512_setzero_ps(),
    };
    alignas(64) int8_t w4[4][384];
    float ws4[4][12];

    if (ext_idx == 0) {
        bool is_hot[4] = { false, false, false, false };
        alignas(64) uint8_t vq12_wire[4][64 * 53];
        for (int r = 0; r < 4; ++r) {
            is_hot[r] = ((rd_u16(rr[r] + WV_INK_HOT_TAG_OFF) >> 12) == 0xFu);
            if (is_hot[r]) {
                materialize_hot_vq12_row(rr[r], n, htab, vq12_wire[r]);
            }
        }
        const int n_chunks = n / 384;
        for (int c = 0; c < n_chunks; ++c) {
            for (int r = 0; r < 4; ++r) {
                if (is_hot[r]) {
                    const auto * bks = reinterpret_cast<const wv_ink_block_vq12 *>(vq12_wire[r]) + 3 * c;
                    unpack_vq12_blk128(bks[0], htab->cb4096, w4[r] +   0, ws4[r] + 0);
                    unpack_vq12_blk128(bks[1], htab->cb4096, w4[r] + 128, ws4[r] + 4);
                    unpack_vq12_blk128(bks[2], htab->cb4096, w4[r] + 256, ws4[r] + 8);
                } else {
                    unpack_vq8_sb6_chunk384(rr[r] + (size_t)c * 108, htab->vq8_uv, w4[r], ws4[r]);
                }
            }
            accum_groups_4xM_vnni(w4, ws4, 12, 12 * c, n_quads, acts_pad, z_acc);
        }
    } else if (ext_idx == 1) {
        const int nb = n >> 7;
        for (int b = 0; b < nb; ++b) {
            for (int r = 0; r < 4; ++r) {
                const auto * bks = reinterpret_cast<const wv_ink_block_vq12 *>(rr[r]);
                unpack_vq12_blk128(bks[b], htab->cb4096, w4[r], ws4[r]);
            }
            accum_groups_4xM_vnni(w4, ws4, 4, 4 * b, n_quads, acts_pad, z_acc);
        }
    } else {
        const int k32 = n >> 5;
        for (int g0 = 0; g0 < k32; g0 += 12) {
            const int g_cnt = std::min(12, k32 - g0);
            for (int r = 0; r < 4; ++r) {
                const auto * bks = reinterpret_cast<const wv_ink_block_swz8 *>(rr[r]) + g0;
                for (int g = 0; g < g_cnt; ++g) {
                    ws4[r][g] = wv_ink_fp16_to_f32(bks[g].d127) * WV_INK_INV_127;
                    std::memcpy(w4[r] + 32 * g, bks[g].qs, 32);
                }
            }
            accum_groups_4xM_vnni(w4, ws4, g_cnt, g0, n_quads, acts_pad, z_acc);
        }
    }

    store_4xM_vnni(z_acc, m_cnt, dst_ptrs, row_off);
}
#endif

int32_t wv_ink_vec_dot_x4_m_cpu(
    int32_t ext_idx,
    int n,
    int n_rows,
    int m_cnt,
    float * const * dst_ptrs,
    const void * vx,
    size_t bx,
    const float * const * vy_ptrs,
    int pf) {
    if (m_cnt <= 0 || m_cnt > 16 || n_rows < 4 || (n_rows & 3) != 0 || !dst_ptrs || !vx || !vy_ptrs) {
        return -1;
    }
    if (ext_idx == 0) {
        if (n <= 0 || (n % 384 != 0) || n > 8192) return -1;
    } else if (ext_idx == 1) {
        if (n <= 0 || ((n & 127) != 0) || n > 8192) return -1;
    } else if (ext_idx == 3) {
        if (n <= 0 || ((n & 31) != 0) || n > 16384) return -1;
    } else {
        return -1;
    }

    const wv_ink_act_q8_slot * acts_pad[16];
    for (int m = 0; m < m_cnt; ++m) {
        acts_pad[m] = wv_ink_prepare_act_q8_tls(vy_ptrs[m], n);
    }
    for (int m = m_cnt; m < 16; ++m) {
        acts_pad[m] = acts_pad[0];
    }

    const uint8_t * base = static_cast<const uint8_t *>(vx);
    const auto * htab = wv_ink_get_host_tables();
    const int isa = g_isa_level;

    for (int r4 = 0; r4 < n_rows; r4 += 4) {
        const uint8_t * r0 = base + (size_t)r4 * bx;
        const uint8_t * rr[4] = { r0, r0 + bx, r0 + 2 * bx, r0 + 3 * bx };

#if WV_HAVE_X86_SIMD
        if (pf != 0 && ext_idx == 0) {
            const size_t pf_off = (pf < 0) ? (4 * bx) : (size_t)pf * 108;
            _mm_prefetch(reinterpret_cast<const char *>(rr[0] + pf_off), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(rr[1] + pf_off), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(rr[2] + pf_off), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(rr[3] + pf_off), _MM_HINT_T0);
        }
#if WV_HAVE_X86_AVX512
        if (isa >= 2) {
            run_x4_m_vnni_driver(ext_idx, n, m_cnt, dst_ptrs, r4, rr, acts_pad, htab);
            continue;
        }
#endif
#else
        (void)pf;
#endif

        // AVX2 (`isa == 1`) & Scalar (`isa == 0`) tiled path: unpacks 4 rows ONCE across all `m_cnt` tokens!
        float acc[16][4] = {};
        alignas(64) int8_t w4[4][384];
        float ws4[4][12];

        auto accum_groups_avx2_or_scalar = [&](int n_groups, int gid_base) {
            for (int g = 0; g < n_groups; ++g) {
                const int gid = gid_base + g;
                const float ws0 = ws4[0][g], ws1 = ws4[1][g], ws2 = ws4[2][g], ws3 = ws4[3][g];
                const int8_t * wp0 = w4[0] + 32 * g;
                const int8_t * wp1 = w4[1] + 32 * g;
                const int8_t * wp2 = w4[2] + 32 * g;
                const int8_t * wp3 = w4[3] + 32 * g;
                for (int m = 0; m < m_cnt; ++m) {
                    const auto * act = acts_pad[m];
                    const float ysc  = act->sc[gid];
                    const int8_t * yq = act->q8 + 32 * gid;
                    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if WV_HAVE_X86_SIMD
                    if (isa >= 1) {
                        const int16_t * ys = act->qsum16 + 2 * gid;
                        d0 = dot32_i8_avx2(wp0, yq, ys);
                        d1 = dot32_i8_avx2(wp1, yq, ys);
                        d2 = dot32_i8_avx2(wp2, yq, ys);
                        d3 = dot32_i8_avx2(wp3, yq, ys);
                    } else
#endif
                    {
                        (void)isa;
                        d0 = dot32_i8_scalar(wp0, yq);
                        d1 = dot32_i8_scalar(wp1, yq);
                        d2 = dot32_i8_scalar(wp2, yq);
                        d3 = dot32_i8_scalar(wp3, yq);
                    }
                    const float sc0 = ws0 * ysc, sc1 = ws1 * ysc, sc2 = ws2 * ysc, sc3 = ws3 * ysc;
                    acc[m][0] = std::fmaf((float)d0, sc0, acc[m][0]);
                    acc[m][1] = std::fmaf((float)d1, sc1, acc[m][1]);
                    acc[m][2] = std::fmaf((float)d2, sc2, acc[m][2]);
                    acc[m][3] = std::fmaf((float)d3, sc3, acc[m][3]);
                }
            }
        };

        if (ext_idx == 0) {
            bool is_hot[4] = { false, false, false, false };
            alignas(64) uint8_t vq12_wire[4][64 * 53];
            for (int r = 0; r < 4; ++r) {
                is_hot[r] = ((rd_u16(rr[r] + WV_INK_HOT_TAG_OFF) >> 12) == 0xFu);
                if (is_hot[r]) {
                    materialize_hot_vq12_row(rr[r], n, htab, vq12_wire[r]);
                }
            }
            const int n_chunks = n / 384;
            for (int c = 0; c < n_chunks; ++c) {
                for (int r = 0; r < 4; ++r) {
                    if (is_hot[r]) {
                        const auto * bks = reinterpret_cast<const wv_ink_block_vq12 *>(vq12_wire[r]) + 3 * c;
                        unpack_vq12_blk128(bks[0], htab->cb4096, w4[r] +   0, ws4[r] + 0);
                        unpack_vq12_blk128(bks[1], htab->cb4096, w4[r] + 128, ws4[r] + 4);
                        unpack_vq12_blk128(bks[2], htab->cb4096, w4[r] + 256, ws4[r] + 8);
                    } else {
                        unpack_vq8_sb6_chunk384(rr[r] + (size_t)c * 108, htab->vq8_uv, w4[r], ws4[r]);
                    }
                }
                accum_groups_avx2_or_scalar(12, 12 * c);
            }
        } else if (ext_idx == 1) {
            const int nb = n >> 7;
            for (int b = 0; b < nb; ++b) {
                for (int r = 0; r < 4; ++r) {
                    const auto * bks = reinterpret_cast<const wv_ink_block_vq12 *>(rr[r]);
                    unpack_vq12_blk128(bks[b], htab->cb4096, w4[r], ws4[r]);
                }
                accum_groups_avx2_or_scalar(4, 4 * b);
            }
        } else {
            const int k32 = n >> 5;
            for (int g0 = 0; g0 < k32; g0 += 12) {
                const int g_cnt = std::min(12, k32 - g0);
                for (int r = 0; r < 4; ++r) {
                    const auto * bks = reinterpret_cast<const wv_ink_block_swz8 *>(rr[r]) + g0;
                    for (int g = 0; g < g_cnt; ++g) {
                        ws4[r][g] = wv_ink_fp16_to_f32(bks[g].d127) * WV_INK_INV_127;
                        std::memcpy(w4[r] + 32 * g, bks[g].qs, 32);
                    }
                }
                accum_groups_avx2_or_scalar(g_cnt, g0);
            }
        }

        for (int m = 0; m < m_cnt; ++m) {
            std::memcpy(dst_ptrs[m] + r4, acc[m], 4 * sizeof(float));
        }
    }
    return 0;
}

ggml_quant_plugin_vec_dot_t    wv_ink_get_vec_dot_ext0(void)    { return vec_dot_ext0_impl; }
ggml_quant_plugin_vec_dot_x4_t wv_ink_get_vec_dot_x4_ext0(void) { return vec_dot_x4_ext0_impl; }
ggml_quant_plugin_vec_dot_t    wv_ink_get_vec_dot_ext1(void)    { return vec_dot_ext1_impl; }
ggml_quant_plugin_vec_dot_x4_t wv_ink_get_vec_dot_x4_ext1(void) { return vec_dot_x4_ext1_impl; }
ggml_quant_plugin_vec_dot_t    wv_ink_get_vec_dot_ext2(void)    { return vec_dot_ext2_impl; }
ggml_quant_plugin_vec_dot_x4_t wv_ink_get_vec_dot_x4_ext2(void) { return vec_dot_x4_ext2_impl; }
ggml_quant_plugin_vec_dot_t    wv_ink_get_vec_dot_ext3(void)    { return vec_dot_ext3_impl; }
ggml_quant_plugin_vec_dot_x4_t wv_ink_get_vec_dot_x4_ext3(void) { return vec_dot_x4_ext3_impl; }

// ik_llama.cpp's expert matmul (MIT, Copyright (C) 2024-2025 Iwan Kawrakow) for our MUL_MAT_ID at prompt sizes.
// MulMat::mul_mat_NxM, the type tables and iqk_mul_mat_moe below are ik's iqk_mul_mat.cpp cut down to the formats Flint's experts use; the
// kernel files next to this one are ik's, with its private types kept inside this port (iqk_private.h). Off unless ggml_cpu_set_iqk(N) is
// called; see iqk_port.h for what it computes and when it steps aside.
#include "iqk_config.h"
#include "iqk_port.h"

#if defined IQK_IMPLEMENT

#include <algorithm>
#include <array>
#include <cstring>
#include <new>
#include <vector>

#include "ggml-impl.h"
#include "ggml-quants.h"
#include "iqk_mul_mat.h"
#include "iqk_common.h"
#include "iqk_gemm_kquants.h"
#include "iqk_gemm_iquants.h"
#include "iqk_gemm_legacy_quants.h"
#include "iqk_utils.h"

namespace {

struct MulMat {
    std::array<mul_mat_t, IQK_MAX_NY> funcs = {};
    mul_mat_t func16 = nullptr;
    inline void mul_mat_NxM(int n, const void * vx, size_t bx, DataInfo& info, int nrc_x, int nrc_y) {
#ifdef __aarch64__
        constexpr int k_x_step = 64; //8192; // Tiling does not seem to help on my M2 Max (but difference to tiling is small)
#else
        constexpr int k_x_step = 64; // This works best on my Ryzen-7950X (but differences to other tile size are small)
#endif
        if (func16 && nrc_y >= 16) {
            int n_step = (nrc_y - info.cur_y)/16;
            for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                auto this_info = info;
                this_info.s += ix;
                int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                for (int iy = 0; iy < n_step; ++iy) {
                    func16(n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                    this_info.cur_y += 16;
                }
            }
            info.cur_y += 16 * n_step;
            if (info.cur_y == nrc_y) return;
        }
        int ny = funcs.size();
        while (!funcs[ny-1] && ny > 0) --ny;
        int n_left = nrc_y - info.cur_y;
        int n_step = n_left/ny;
        if (n_step > 0) {
            if (n_step*ny != n_left) {
                ++n_step;
                int ny1 = n_left/n_step;
                int ny2 = ny1 + 1;
                int my1 = n_step*ny2 - n_left;
                int my2 = n_step - my1;
                for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                    auto this_info = info;
                    this_info.s += ix;
                    int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                    for (int iy = 0; iy < my1; ++iy) {
                        funcs[ny1-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny1;
                    }
                    for (int iy = 0; iy < my2; ++iy) {
                        funcs[ny2-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny2;
                    }
                }
                info.cur_y += n_left;
            }
            else {
                for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                    auto this_info = info;
                    this_info.s += ix;
                    int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                    for (int iy = 0; iy < n_step; ++iy) {
                        funcs[ny-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny;
                    }
                }
                info.cur_y += ny * n_step;
            }
        }
        n_left = nrc_y - info.cur_y;
        if (n_left > 0) {
            funcs[n_left-1](n, vx, bx, info, nrc_x);
        }
    }
    static bool prepare(int typeA, int typeB, int ne00, MulMat & mm) {
        switch (typeA) {
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_IQ4_XS:
            case GGML_TYPE_Q8_K_R8:
            case GGML_TYPE_Q8_K_R16:
                return iqk_set_kernels_kquants(ne00, typeA, typeB, mm.funcs, mm.func16);
            case GGML_TYPE_IQ3_S:
                return iqk_set_kernels_iquants(ne00, typeA, typeB, mm.funcs, mm.func16);
            case GGML_TYPE_Q2_0:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q8_1:
            case GGML_TYPE_Q8_0_R8:
            case GGML_TYPE_Q8_0_R16:
                return iqk_set_kernels_legacy_quants(ne00, typeA, typeB, mm.funcs, mm.func16);
            default:
                return false;
        }
    }

    // ik's choice: which 8-bit interleaved form the weights are unpacked to for this many activation rows (the same type = no unpacking)
    static int is_dequant_better(int type, int nrc_y) {
#ifdef HAVE_FANCY_SIMD
        const int q8_k_type = GGML_TYPE_Q8_K_R16, q8_0_type = GGML_TYPE_Q8_0_R16, k_min_nrc_y = 16;
#else
        const int q8_k_type = GGML_TYPE_Q8_K_R8, q8_0_type = GGML_TYPE_Q8_0_R8, k_min_nrc_y = 32;
#endif
        switch (type) {
            case GGML_TYPE_Q3_K:
#ifdef HAVE_FANCY_SIMD
                return nrc_y >= k_min_nrc_y ? q8_k_type : type;
#else
                return type;
#endif
            case GGML_TYPE_IQ4_XS: return nrc_y >= k_min_nrc_y ? q8_k_type : type;
            case GGML_TYPE_IQ3_S:  return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:   return nrc_y >= 32 ? (int) GGML_TYPE_Q8_1 : type;
            case GGML_TYPE_Q2_0:
            case GGML_TYPE_Q4_0:   return nrc_y >= k_min_nrc_y ? q8_0_type : type;
            case GGML_TYPE_Q8_0:   return nrc_y >= 32 ? q8_0_type : type;
            default:               return type;
        }
    }

    static int num_rows(int type) {
        switch (type) {
#ifdef HAVE_FANCY_SIMD
            case GGML_TYPE_Q8_K_R8:
                return 8;
            case GGML_TYPE_Q8_0_R8:
            case GGML_TYPE_Q8_1:
            case GGML_TYPE_Q8_0_R16:
            case GGML_TYPE_Q8_K_R16:
                return 16;
#else
            case GGML_TYPE_Q8_0_R8:
            case GGML_TYPE_Q8_1:
            case GGML_TYPE_Q8_K_R8:
                return 8;
            case GGML_TYPE_Q8_0_R16:
            case GGML_TYPE_Q8_K_R16:
                return 16;
#endif
            default:
                return 1;
        }
    }

    // bytes of one row of n values of an unpacked form (ik's type table)
    static size_t row_size(int type, int n) {
        switch (type) {
            case GGML_TYPE_Q8_K_R8:  return (size_t) n / QK_K * sizeof(block_q8_k_r8) / 8;
            case GGML_TYPE_Q8_K_R16: return (size_t) n / QK_K * sizeof(block_q8_k_r16) / 16;
            case GGML_TYPE_Q8_0_R8:  return (size_t) n / QK8_0 * sizeof(block_q8_0);
            case GGML_TYPE_Q8_0_R16: return (size_t) n / QK8_0 * sizeof(block_q8_0_r16) / 16;
            default:                 return ggml_row_size(ggml_type(type), n);
        }
    }
};

// ik's activation quantizer (iqk_quantize.cpp: iqk_quantize_row_q8_K_T)
template <int q8_type>
void iqk_quantize_row_q8_K_T(const float * x, void * vy, int64_t k) {
    const int nb = k / QK_K;
    const int nb32 = k / 32;
    const int nbtail = nb32 - 8*nb;
    //printf("Quantizing row with k = %ld, nb = %d, nbtail = %d\n", k, nb, nbtail);
    block_q8_K_ik * y = (block_q8_K_ik *)vy;
#ifdef __AVX2__
    const __m256 signBit = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    auto quantize_block_32 = [&perm] (__m256 mul, const float * x, int8_t * q8, int16_t & sum1, int16_t & sum2) {
        __m256 v0 = _mm256_mul_ps(mul, _mm256_loadu_ps(x +  0));
        __m256 v1 = _mm256_mul_ps(mul, _mm256_loadu_ps(x +  8));
        __m256 v2 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 16));
        __m256 v3 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 24));
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);
        sum1 = hsum_i32_8(_mm256_add_epi32(i0, i1));
        sum2 = hsum_i32_8(_mm256_add_epi32(i2, i3));
        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);
        _mm256_storeu_si256((__m256i *)q8, i0);
    };
    for (int i = 0; i < nb; i++) {
        const float * xb = x + i*QK_K;
        __m256 maxAbs = _mm256_setzero_ps();
        const float * xx = xb;
        for (int ib = 0; ib < QK_K/8; ++ib) {
            const __m256 v = _mm256_loadu_ps(xx); xx += 8;
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps(signBit, v));
        }
        const float maxScalar = hmax_f32_8(maxAbs);
        const float d = maxScalar / 127.f;
        y[i].d = d;
        const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);
        int8_t * q8 = y[i].qs;
        int block_sum_i32 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int16_t s1, s2;
            quantize_block_32(mul, xb + 32*ib, q8, s1, s2);
            int s = s1 + s2;
            if constexpr (q8_type == 1) {
                auto bs = (float *)y[i].bsums;
                bs[ib] = d*s;
            } else {
                y[i].bsums[2*ib+0] = s1;
                y[i].bsums[2*ib+1] = s2;
            }
            block_sum_i32 += s;
            q8 += 32;
        }
        y[i].sum = d*block_sum_i32;
        //if constexpr (q8_type == 2) {
        //    auto bs = (float *)y[i].bsums;
        //    float sum = 0;
        //    for (int ib = 0; ib < QK_K/32; ++ib) sum += bs[ib];
        //    bs[0] = sum;
        //}
    }
    if (nbtail > 0) {
        const float * xb = x + nb*QK_K;
        __m256 maxAbs = _mm256_setzero_ps();
        const float * xx = xb;
        for (int ib = 0; ib < 4*nbtail; ++ib) {
            const __m256 v = _mm256_loadu_ps(xx); xx += 8;
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps(signBit, v));
        }
        const float maxScalar = hmax_f32_8(maxAbs);
        const float d = maxScalar / 127.f;
        y[nb].d = d;
        const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);
        int8_t * q8 = y[nb].qs;
        auto bsums = (int16_t *)(y[nb].qs + 32*nbtail);
        int block_sum_i32 = 0;
        for (int ib = 0; ib < nbtail; ++ib) {
            int16_t s1, s2;
            quantize_block_32(mul, xb + 32*ib, q8, s1, s2);
            bsums[2*ib+0] = s1;
            bsums[2*ib+1] = s2;
            block_sum_i32 += s1 + s2;
            q8 += 32;
        }
        y[nb].sum = d*block_sum_i32;
    }
#else
    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        float d = 1/iscale;
        if constexpr (q8_type == 1) {
            auto bs = (float *)y[i].bsums;
            int tot = 0;
            for (int j = 0; j < QK_K/32; ++j) {
                int sum = 0;
                for (int ii = 0; ii < 32; ++ii) {
                    sum += y[i].qs[j*32 + ii];
                }
                bs[j] = d*sum;
                tot += sum;
            }
            y[i].sum = d*tot;
        } else {
            int tot = 0;
            for (int j = 0; j < QK_K/16; ++j) {
                int sum = 0;
                for (int ii = 0; ii < 16; ++ii) {
                    sum += y[i].qs[j*16 + ii];
                }
                y[i].bsums[j] = sum;
                tot += sum;
            }
            y[i].sum = d*tot;
        }
        y[i].d = d;
        x += QK_K;
    }
#endif
}

template <class T>
struct CacheLineAllocator {
    using value_type = T;
    static constexpr std::align_val_t align{GGML_Q8_K_R16_ALIGN};
    CacheLineAllocator() = default;
    template <class U> CacheLineAllocator(const CacheLineAllocator<U> &) noexcept {}
    T *  allocate  (size_t n)               { return static_cast<T *>(::operator new(n*sizeof(T), align)); }
    void deallocate(T * p, size_t) noexcept { ::operator delete(p, align); }
    template <class U> bool operator==(const CacheLineAllocator<U> &) const noexcept { return true; }
    template <class U> bool operator!=(const CacheLineAllocator<U> &) const noexcept { return false; }
};

using work_buffer_t = std::vector<char, CacheLineAllocator<char>>;

static work_buffer_t & thread_local_work_buffer() {
    thread_local work_buffer_t f;
    return f;
}

static bool convert_repack(int typeA, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    switch (typeA) {
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ4_XS:
            return iqk_convert_kquants_q8X_r8(typeA, n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ3_S:
            return iqk_convert_iquants_q80_r8(typeA, n, vx, bx, vy, nrc_x);
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return iqk_convert_legacy_quants_q8_r8(typeA, n, vx, bx, vy, nrc_x);
        default:
            return false;
    }
}

} // namespace

extern "C" {

int iqk_port_available(void) { return 1; }

// which activation layout ik's kernels want: 1 = ik's Q8_K (our Q8_K rows converted), 2 = Q8_2_X4 (quantized from the floats)
int iqk_port_b_kind(int type) { return type == GGML_TYPE_Q3_K || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ3_S ? 1 : 2; }

size_t iqk_port_b_row_size(int type, long n) {
    return iqk_port_b_kind(type) == 1 ? iqk_port_q8k_row_size(n) : (size_t) n / QK8_2 * sizeof(block_q8_2);
}

// ik's quantize_row_q8_2_x4 (iqk_quantize.cpp, the AVX2 branch): one activation row, 4 blocks of 32 grouped; scales as bf16, the block's sum of quants in the s field
void iqk_port_q8_2_x4_row(const float * x, void * vy, long k) {
    const int nb  = (int) (k / QK8_2);
    const int nb4 = 4*(nb/4);
    block_q8_2 *    y  = (block_q8_2 *) vy;
    block_q8_2_x4 * y4 = (block_q8_2_x4 *) vy;
    for (int i = 0; i < nb; i++) {
        const int i4 = i/4, ir = i%4;
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;

        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float max_scalar = _mm_cvtss_f32(max4);

        float d = max_scalar / 127.f;
        auto t = GGML_FP32_TO_BF16(d);
        d = ggml_bf16_to_fp32(t);
        if (i < nb4) {
            y4[i4].d[ir] = t.bits;
        } else {
            y[i].d = t.bits;
        }
        const float id = d > 0 ? 1/d : 0.f;
        const __m256 mul = _mm256_set1_ps(id);

        v0 = _mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_ROUND_NEAREST);

        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);

        const int isum = hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
        if (i < nb4) {
            ((int16_t *) y4[i4].d)[ir + 4] = (int16_t) isum;
        } else {
            *(int16_t *) &y[i].s = (int16_t) isum;
        }

        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);

        if (i < nb4) {
            _mm256_storeu_si256((__m256i *) y4[i4].qs + ir, i0);
        } else {
            _mm256_storeu_si256((__m256i *) y[i].qs, i0);
        }
    }
}

int iqk_port_node_ok(int type, long ne00, long ne01) {
    switch (type) {
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
            return ne00 % QK_K == 0 && ne01 % 16 == 0;   // whole blocks, weight rows the 16-row kernels can tile
        case GGML_TYPE_Q2_0:
            return ne00 % QK2_0 == 0 && ne01 % 16 == 0;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return ne00 % QK8_0 == 0 && ne01 % 16 == 0;
        default:
            return 0;   // q6_K: ik loses at 2048 rows; any other format: the plain path
    }
}

size_t iqk_port_q8k_row_size(long n) { return (size_t) n / QK_K * sizeof(block_q8_K_ik); }

// blocks [first_block, end_block) of one activation row, quantized by ik's quantizer: into ik's layout (ik_row, or NULL) and ours (ours_row, or NULL)
void iqk_port_q8k_quant_range(const float * x, void * ours_row, void * ik_row, long first_block, long end_block) {
    block_q8_K * y = (block_q8_K *) ours_row;
    block_q8_K_ik * z = (block_q8_K_ik *) ik_row;
    for (long b = first_block; b < end_block; b++) {
        block_q8_K_ik t;
        iqk_quantize_row_q8_K_T<0>(x + b*QK_K, &t, QK_K);
        if (z) {
            z[b] = t;
        }
        if (y) {
            y[b].d = t.d;
            memcpy(y[b].qs, t.qs, sizeof(y[b].qs));
            memcpy(y[b].bsums, t.bsums, sizeof(y[b].bsums));
        }
    }
}

// can the kernels take an expert with Ny rows (the same test iqk_port_mmid makes)
int iqk_port_can(int typeA, long Ny, long ne00) {
#ifndef HAVE_FANCY_SIMD
    if (typeA == GGML_TYPE_Q3_K && Ny > 32) {
        return 0;   // without AVX-512 VNNI, ggml's tiled path overtakes Q3_K above 32 rows
    }
#endif
    MulMat mm;
    const int typeB = iqk_port_b_kind(typeA) == 1 ? (int) GGML_TYPE_Q8_K : GGML_TYPE_Q8_2_X4;
    const int dequant_type = MulMat::is_dequant_better(typeA, (int) Ny);
    return dequant_type != typeA ? (MulMat::prepare(dequant_type, typeB, (int) ne00, mm) || MulMat::prepare(typeA, typeB, (int) ne00, mm)) : MulMat::prepare(typeA, typeB, (int) ne00, mm);
}

// ik's iqk_mul_mat_moe: rows of the weights split by thread, the rows routed to this expert all done by each thread for its weight rows.
// Nx weight rows, Ny = the rows routed to this expert. No shared state between threads, no barrier
int iqk_port_mmid(long Nx, long Ny, long ne00, int ne11, int typeA, const void * A, long strideA,
                  const void * B, long strideB, float * C, long nb1, long nb2,
                  const void * vrow_mapping, int ith, int nth) {
    const mmid_row_mapping * row_mapping = (const mmid_row_mapping *) vrow_mapping;
    const int typeB = iqk_port_b_kind(typeA) == 1 ? (int) GGML_TYPE_Q8_K : GGML_TYPE_Q8_2_X4;

    MulMat mm;

    const int dequant_type = MulMat::is_dequant_better(typeA, (int) Ny);

    if (dequant_type != typeA && Nx % MulMat::num_rows(dequant_type) == 0 && MulMat::prepare(dequant_type, typeB, (int) ne00, mm)) {
        constexpr int k_x_step = 32;

        auto num_rows = MulMat::num_rows(dequant_type);
        auto nrc_x = (Nx/num_rows + nth - 1)/nth;
        auto first_x = ith*nrc_x;
        if (first_x + nrc_x > Nx/num_rows) nrc_x = Nx/num_rows - first_x;
        first_x *= num_rows;
        nrc_x   *= num_rows;

        size_t row_size_qx = MulMat::row_size(dequant_type, (int) ne00);
        size_t row_size_qy = strideB;

        DataInfo info{C + first_x, (const char *) B, (size_t) nb1/sizeof(float), row_size_qy, 0, ne11, row_mapping, (size_t) nb2/sizeof(float)};

        auto & f = thread_local_work_buffer();

        for (int ix = 0; ix < nrc_x; ix += k_x_step) {
            auto this_info = info;
            this_info.s += ix;
            int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : (int) (nrc_x - ix);
            if (f.size() < row_size_qx*this_nrc_x) f.resize(row_size_qx*this_nrc_x);
            if (!convert_repack(typeA, (int) ne00, (const char *) A + (first_x + ix)*strideA, strideA, f.data(), this_nrc_x)) {
                GGML_ABORT("iqk port: the weights could not be unpacked");
            }
            mm.mul_mat_NxM((int) ne00, f.data(), row_size_qx, this_info, this_nrc_x, (int) Ny);
        }
        return 1;
    }

    if (!MulMat::prepare(typeA, typeB, (int) ne00, mm)) {
        return 0;
    }
    auto nrc_x = (Nx + nth - 1)/nth;
    auto first_x = ith*nrc_x;
    if (first_x + nrc_x > Nx) nrc_x = Nx - first_x;
    DataInfo info{C + first_x, (const char *) B, (size_t) nb1/sizeof(float), (size_t) strideB, 0, ne11, row_mapping, (size_t) nb2/sizeof(float)};
    mm.mul_mat_NxM((int) ne00, (const char *) A + strideA*first_x, strideA, info, (int) nrc_x, (int) Ny);
    return 1;
}

} // extern "C"

#else // !IQK_IMPLEMENT

extern "C" {
int iqk_port_available(void) { return 0; }
int iqk_port_node_ok(int, long, long) { return 0; }
size_t iqk_port_q8k_row_size(long) { return 0; }
void iqk_port_q8k_quant_range(const float *, void *, void *, long, long) {}
int iqk_port_b_kind(int) { return 0; }
int iqk_port_can(int, long, long) { return 0; }
size_t iqk_port_b_row_size(int, long) { return 0; }
void iqk_port_q8_2_x4_row(const float *, void *, long) {}
int iqk_port_mmid(long, long, long, int, int, const void *, long, const void *, long, float *, long, long, const void *, int, int) { return 0; }
}

#endif

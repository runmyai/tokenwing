#include "qsa-sparse-attn.cuh"
#include "mma.cuh"
#include "vecdotq.cuh"

#include <atomic>
#include <cstdlib>
#include <cstring>

static std::atomic<bool> g_qsa_fused_on{false};
static std::atomic<bool> g_hc_fused_on{false};
static std::atomic<bool> g_ple_fast_on{false};

static bool qsa_warp32_ok() {
    const int dev = ggml_cuda_get_device();
    return dev >= 0 && dev < ggml_cuda_info().device_count && ggml_cuda_info().devices[dev].warp_size == 32;
}

bool ggml_cuda_qsa_fused_enabled() {
    static const int env = []() {
        const char * s = std::getenv("GGML_CUDA_QSA_FUSED");
        return s ? std::atoi(s) : -1;
    }();
    const bool on = env >= 0 ? (env != 0) : g_qsa_fused_on.load(std::memory_order_relaxed);
    return on && qsa_warp32_ok();
}

void ggml_cuda_set_qsa_fused(bool on) {
    g_qsa_fused_on.store(on, std::memory_order_relaxed);
}

bool ggml_cuda_hc_fused_enabled() {
    static const int env = []() {
        const char * s = std::getenv("GGML_CUDA_HC_FUSED");
        return s ? std::atoi(s) : -1;
    }();
    const bool on = env >= 0 ? (env != 0) : g_hc_fused_on.load(std::memory_order_relaxed);
    return on && qsa_warp32_ok();
}

void ggml_cuda_set_hc_fused(bool on) {
    g_hc_fused_on.store(on, std::memory_order_relaxed);
}

bool ggml_cuda_ple_fast_enabled() {
    static const int env = []() {
        const char * s = std::getenv("GGML_CUDA_PLE_FAST");
        return s ? std::atoi(s) : -1;
    }();
    return env >= 0 ? (env != 0) : g_ple_fast_on.load(std::memory_order_relaxed);
}

void ggml_cuda_set_ple_fast(bool on) {
    g_ple_fast_on.store(on, std::memory_order_relaxed);
}

static __global__ void qsa_build_sorted_cells_kernel(
        const int32_t * __restrict__ top_k,
        const float *   __restrict__ score,
        const int32_t * __restrict__ pool_idxs,
        const int32_t * __restrict__ tail_idxs,
        int32_t *       __restrict__ dst_cells,
        int32_t                      n_top_pool,
        int32_t                      n_pool,
        int32_t                      kpool,
        int32_t                      n_tail,
        int32_t                      n_sel,
        int32_t                      n_kv,
        int64_t                      stk1,
        int64_t                      ssc1,
        int64_t                      spi1,
        int64_t                      sti1,
        int64_t                      sd1) {
    __shared__ int32_t s_pools[256];
    __shared__ int32_t s_write_pos;

    const int tok = blockIdx.x;
    const int tid = threadIdx.x;

    int sort_n = 1;
    while (sort_n < n_top_pool && sort_n < 256) {
        sort_n <<= 1;
    }

    if (n_top_pool <= 256) {
        for (int i = tid; i < sort_n; i += blockDim.x) {
            int32_t p = INT32_MAX;
            if (i < n_top_pool) {
                const int32_t cand = top_k[(int64_t) tok * stk1 + i];
                if (cand >= 0 && cand < n_pool) {
                    const float sc = score[(int64_t) tok * ssc1 + cand];
                    if (sc > -INFINITY) {
                        p = cand;
                    }
                }
            }
            s_pools[i] = p;
        }
        __syncthreads();

        for (int k = 2; k <= sort_n; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int i = tid; i < sort_n; i += blockDim.x) {
                    const int ixj = i ^ j;
                    if (ixj > i) {
                        const bool asc = ((i & k) == 0);
                        const int32_t a = s_pools[i];
                        const int32_t b = s_pools[ixj];
                        if ((asc && a > b) || (!asc && a < b)) {
                            s_pools[i]   = b;
                            s_pools[ixj] = a;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    if (tid == 0) {
        int32_t out_n = 0;
        int32_t * out_row = dst_cells + (int64_t) tok * sd1;
        const int lim = n_top_pool <= 256 ? n_top_pool : 0;
        auto push_unique_cell = [&](int32_t cell) {
            if (cell < 0 || cell >= n_kv || out_n >= n_sel) {
                return;
            }
            for (int k = 0; k < out_n; ++k) {
                if (out_row[k] == cell) {
                    return;
                }
            }
            out_row[out_n++] = cell;
        };
        if (lim > 0) {
            for (int i = 0; i < lim; ++i) {
                const int32_t p = s_pools[i];
                if (p == INT32_MAX) {
                    break;
                }
                if (i > 0 && p == s_pools[i - 1]) {
                    continue;
                }
                const int32_t * prow = pool_idxs + (int64_t) p * spi1;
                for (int c = 0; c < kpool; ++c) {
                    const int32_t cell = prow[c];
                    if (cell >= 0 && cell < n_kv && out_n < n_sel) {
                        out_row[out_n++] = cell;
                    }
                }
            }
        } else {
            for (int i = 0; i < n_top_pool; ++i) {
                const int32_t p = top_k[(int64_t) tok * stk1 + i];
                if (p < 0 || p >= n_pool) {
                    continue;
                }
                const float sc = score[(int64_t) tok * ssc1 + p];
                if (!(sc > -INFINITY)) {
                    continue;
                }
                const int32_t * prow = pool_idxs + (int64_t) p * spi1;
                for (int c = 0; c < kpool; ++c) {
                    push_unique_cell(prow[c]);
                }
            }
        }
        for (int c = 0; c < n_tail; ++c) {
            push_unique_cell(tail_idxs[(int64_t) tok * sti1 + c]);
        }
        s_write_pos = out_n;
    }
    __syncthreads();

    int32_t * out_row = dst_cells + (int64_t) tok * sd1;
    for (int i = s_write_pos + tid; i < n_sel; i += blockDim.x) {
        out_row[i] = -1;
    }
}

void ggml_cuda_qsa_build_cells(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         top_k,
        const ggml_tensor *         score,
        const ggml_tensor *         pool_idxs,
        const ggml_tensor *         tail_idxs,
        int32_t                     n_kv,
        ggml_tensor *               dst_cells) {
    const int32_t n_top_pool = (int32_t) top_k->ne[0];
    const int32_t nt         = (int32_t) top_k->ne[1];
    const int32_t n_pool     = (int32_t) score->ne[0];
    const int32_t kpool      = (int32_t) pool_idxs->ne[0];
    const int32_t n_tail     = (int32_t) tail_idxs->ne[0];
    const int32_t n_sel      = (int32_t) dst_cells->ne[0];

    qsa_build_sorted_cells_kernel<<<nt, 128, 0, ctx.stream()>>>(
        (const int32_t *) top_k->data,
        (const float *)   score->data,
        (const int32_t *) pool_idxs->data,
        (const int32_t *) tail_idxs->data,
        (int32_t *)       dst_cells->data,
        n_top_pool, n_pool, kpool, n_tail, n_sel, n_kv,
        (int64_t)(top_k->nb[1]     / sizeof(int32_t)),
        (int64_t)(score->nb[1]     / sizeof(float)),
        (int64_t)(pool_idxs->nb[1] / sizeof(int32_t)),
        (int64_t)(tail_idxs->nb[1] > 0 && tail_idxs->ne[1] > 1 ? tail_idxs->nb[1] / sizeof(int32_t) : 0),
        (int64_t)(dst_cells->nb[1] / sizeof(int32_t)));
    CUDA_CHECK(cudaGetLastError());
}

static __device__ __forceinline__ half2 qsa_q8x2_to_half2(uint16_t u16, half2 d2) {
    const uint32_t spread = ((uint32_t)(u16 & 0x00FFu)) | (((uint32_t)(u16 & 0xFF00u)) << 8);
    const uint32_t biased = (spread ^ 0x00800080u) | 0x64006400u;
    half2 h2;
    memcpy(&h2, &biased, sizeof(h2));
    const half  zp1 = __float2half(1152.0f);
    const half2 zp2 = __halves2half2(zp1, zp1);
    return __hmul2(__hsub2(h2, zp2), d2);
}

template <ggml_type kv_type>
static __device__ __forceinline__ void qsa_load_kv_row_to_half(
        const char * __restrict__ src_row,
        half *       __restrict__ dst_half,
        int                       D,
        int                       lane) {
    if constexpr (kv_type == GGML_TYPE_F16) {
        const half2 * s2 = (const half2 *) src_row;
        half2 *       d2 = (half2 *)       dst_half;
        for (int i = lane; i < D / 2; i += 32) {
            d2[i] = s2[i];
        }
    } else if constexpr (kv_type == GGML_TYPE_BF16) {
        const uint16_t * sb = (const uint16_t *) src_row;
        for (int i = lane; i < D; i += 32) {
            const float f = __uint_as_float(((uint32_t) sb[i]) << 16);
            dst_half[i]   = __float2half(f);
        }
    } else if constexpr (kv_type == GGML_TYPE_Q8_0) {
        const block_q8_0 * blks = (const block_q8_0 *) src_row;
        half2 * d2 = (half2 *) dst_half;
        for (int i = lane; i < D / 2; i += 32) {
            const int b_idx = i >> 4;
            const int sub   = i & 15;
            const half dh   = blks[b_idx].d;
            uint16_t u16;
            memcpy(&u16, &blks[b_idx].qs[sub * 2], sizeof(u16));
            d2[i] = qsa_q8x2_to_half2(u16, __halves2half2(dh, dh));
        }
    } else if constexpr (kv_type == GGML_TYPE_Q4_0) {
        const block_q4_0 * blks = (const block_q4_0 *) src_row;
        for (int i = lane; i < D / 2; i += 32) {
            const int b_idx  = i >> 3;
            const int j      = i & 7;
            const float d    = __half2float(blks[b_idx].d);
            const uint8_t b0 = blks[b_idx].qs[2 * j + 0];
            const uint8_t b1 = blks[b_idx].qs[2 * j + 1];
            dst_half[b_idx * 32 + 2 * j + 0]      = __float2half(((int)(b0 & 0x0Fu) - 8) * d);
            dst_half[b_idx * 32 + 2 * j + 1]      = __float2half(((int)(b1 & 0x0Fu) - 8) * d);
            dst_half[b_idx * 32 + 16 + 2 * j + 0] = __float2half(((int)(b0 >> 4)    - 8) * d);
            dst_half[b_idx * 32 + 16 + 2 * j + 1] = __float2half(((int)(b1 >> 4)    - 8) * d);
        }
    }
}

template <ggml_type k_type, ggml_type v_type, int HEAD_DIM>
static __global__ void __launch_bounds__(128) qsa_gqa_transposed_attn_kernel(
        const float *   __restrict__ q,
        const char *    __restrict__ k,
        const char *    __restrict__ v,
        const int32_t * __restrict__ sel_cells,
        float *         __restrict__ dst,
        int                          n_head_q,
        int                          n_head_kv,
        int                          n_sel,
        int                          nt_per_stream,
        float                        scale,
        size_t                       nbq1,
        size_t                       nbq2,
        size_t                       nbq3,
        size_t                       nbk1,
        size_t                       nbk2,
        size_t                       nbk3,
        size_t                       nbv1,
        size_t                       nbv2,
        size_t                       nbv3,
        int64_t                      ssel1,
        size_t                       nbd1,
        size_t                       nbd2,
        size_t                       nbd3) {
    constexpr int D_PAD   = HEAD_DIM + 8;
    constexpr int D_PER_T = HEAD_DIM / 8;

    __shared__ half  smem_q[16][D_PAD];
    __shared__ half  smem_kv[8][D_PAD];
    __shared__ float smem_part[4][16][8];
    __shared__ float smem_s[16][8];

    const int tok_global = blockIdx.x;
    const int h_kv       = blockIdx.y;
    const int gqa_grp    = blockIdx.z;
    const int stream_id  = nt_per_stream > 0 ? (tok_global / nt_per_stream) : 0;
    const int tok_local  = nt_per_stream > 0 ? (tok_global % nt_per_stream) : tok_global;

    const int gqa_ratio = n_head_q / n_head_kv;
    const int hq_base   = h_kv * gqa_ratio + gqa_grp * 16;
    const int hq_valid  = min(16, max(0, gqa_ratio - gqa_grp * 16));

    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid     = warp_id * 32 + lane;
    const int g       = lane >> 2;
    const int t       = lane & 3;

    // In GGML_OP_FLASH_ATTN_EXT: q is [D, nt_local, n_head_q, n_stream], k/v are [D, n_kv, n_head_kv, n_stream]
    for (int h = warp_id; h < 16; h += 4) {
        const float * q_row = (h < hq_valid)
            ? (const float *)((const char *) q + (size_t) tok_local * nbq1 + (size_t)(hq_base + h) * nbq2 + (size_t) stream_id * nbq3)
            : nullptr;
        for (int d = lane; d < HEAD_DIM; d += 32) {
            smem_q[h][d] = q_row ? __float2half(q_row[d]) : __float2half(0.0f);
        }
    }
    __syncthreads();

    const int my_h    = tid >> 3;      // 0..15
    const int my_d0   = (tid & 7) * D_PER_T;
    float     m_i     = -INFINITY;
    float     l_i     = 0.0f;
    float     o_reg[D_PER_T] = {};

    const int32_t * cells_row = sel_cells + (int64_t) tok_global * ssel1;
    const char * k_base = k + (size_t) h_kv * nbk2 + (size_t) stream_id * nbk3;
    const char * v_base = v + (size_t) h_kv * nbv2 + (size_t) stream_id * nbv3;

    for (int s0 = 0; s0 < n_sel; s0 += 8) {
        int32_t c_idx[8];
        int valid_cols = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int32_t ci = (s0 + j < n_sel) ? cells_row[s0 + j] : -1;
            c_idx[j] = ci;
            if (ci >= 0) {
                ++valid_cols;
            }
        }
        if (valid_cols == 0) {
            break;
        }

        // Decode 8 selected K rows across 4 warps (2 rows per warp)
        for (int j = warp_id; j < 8; j += 4) {
            if (c_idx[j] >= 0) {
                qsa_load_kv_row_to_half<k_type>(k_base + (size_t) c_idx[j] * nbk1, smem_kv[j], HEAD_DIM, lane);
            } else {
                for (int d = lane; d < HEAD_DIM; d += 32) {
                    smem_kv[j][d] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // Split HEAD_DIM across 4 warps on Tensor Cores
        constexpr int D_WARP = HEAD_DIM / 4;
        const int d_start    = warp_id * D_WARP;
#if defined(TURING_MMA_AVAILABLE)
        ggml_cuda_mma::tile<16, 8, float> D_s;
        for (int kd = 0; kd < D_WARP; kd += 16) {
            ggml_cuda_mma::tile<16, 8, half2> A_q;
            ggml_cuda_mma::tile<8,  8, half2> B_k;
            const half2 * q_g0 = (const half2 *) &smem_q[g    ][d_start + kd];
            const half2 * q_g8 = (const half2 *) &smem_q[g + 8][d_start + kd];
            const half2 * k_g0 = (const half2 *) &smem_kv[g   ][d_start + kd];
            A_q.x[0] = q_g0[t];
            A_q.x[1] = q_g8[t];
            A_q.x[2] = q_g0[4 + t];
            A_q.x[3] = q_g8[4 + t];
            B_k.x[0] = k_g0[t];
            B_k.x[1] = k_g0[4 + t];
            ggml_cuda_mma::mma(D_s, A_q, B_k);
        }
        smem_part[warp_id][g    ][2 * t + 0] = D_s.x[0];
        smem_part[warp_id][g    ][2 * t + 1] = D_s.x[1];
        smem_part[warp_id][g + 8][2 * t + 0] = D_s.x[2];
        smem_part[warp_id][g + 8][2 * t + 1] = D_s.x[3];
#else
        const int r0 = g, r1 = g + 8, c0 = 2 * t, c1 = 2 * t + 1;
        float s00 = 0.0f, s01 = 0.0f, s10 = 0.0f, s11 = 0.0f;
        for (int kd = d_start; kd < d_start + D_WARP; ++kd) {
            const float q0 = __half2float(smem_q[r0][kd]);
            const float q1 = __half2float(smem_q[r1][kd]);
            const float k0 = __half2float(smem_kv[c0][kd]);
            const float k1 = __half2float(smem_kv[c1][kd]);
            s00 += q0 * k0; s01 += q0 * k1; s10 += q1 * k0; s11 += q1 * k1;
        }
        smem_part[warp_id][r0][c0] = s00; smem_part[warp_id][r0][c1] = s01;
        smem_part[warp_id][r1][c0] = s10; smem_part[warp_id][r1][c1] = s11;
#endif
        __syncthreads();

        // Reduce 4 warps' partial dot products (128 elements = 1 per thread!) + decode 8 V rows
        {
            const int hr = tid >> 3;
            const int jc = tid & 7;
            const float raw = smem_part[0][hr][jc] + smem_part[1][hr][jc] + smem_part[2][hr][jc] + smem_part[3][hr][jc];
            smem_s[hr][jc]  = (c_idx[jc] >= 0) ? (raw * scale) : -INFINITY;
        }
        for (int j = warp_id; j < 8; j += 4) {
            if (c_idx[j] >= 0) {
                qsa_load_kv_row_to_half<v_type>(v_base + (size_t) c_idx[j] * nbv1, smem_kv[j], HEAD_DIM, lane);
            }
        }
        __syncthreads();

        if (my_h < hq_valid) {
            float block_max = m_i;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                block_max = fmaxf(block_max, smem_s[my_h][j]);
            }
            const float alpha = isfinite(m_i) ? expf(m_i - block_max) : 0.0f;
            #pragma unroll
            for (int d = 0; d < D_PER_T; ++d) {
                o_reg[d] *= alpha;
            }
            float p_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float s_val = smem_s[my_h][j];
                if (isfinite(s_val)) {
                    const float p = expf(s_val - block_max);
                    p_sum += p;
                    const half * vj = smem_kv[j] + my_d0;
                    #pragma unroll
                    for (int d = 0; d < D_PER_T; ++d) {
                        o_reg[d] += p * __half2float(vj[d]);
                    }
                }
            }
            l_i = l_i * alpha + p_sum;
            m_i = block_max;
        }
        __syncthreads();
    }

    if (my_h < hq_valid) {
        const float inv_l = l_i > 0.0f ? (1.0f / l_i) : 0.0f;
        // Note: flash_attn_ext output layout in ggml is [D, n_head_q, nt_local, n_stream]
        float * out_row = (float *)((char *) dst + (size_t)(hq_base + my_h) * nbd1 + (size_t) tok_local * nbd2 + (size_t) stream_id * nbd3);
        #pragma unroll
        for (int d = 0; d < D_PER_T; ++d) {
            out_row[my_d0 + d] = o_reg[d] * inv_l;
        }
    }
}

template <ggml_type kv_type, int HEAD_DIM>
static void launch_qsa_dim(
        cudaStream_t stream, const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v,
        const ggml_tensor * sel_cells, ggml_tensor * dst, float scale) {
    const int nt_local  = (int) q->ne[1];
    const int n_head_q  = (int) q->ne[2];
    const int n_stream  = (int) q->ne[3];
    const int n_head_kv = (int) k->ne[2];
    const int n_sel     = (int) sel_cells->ne[0];
    const int nt_total  = nt_local * n_stream;
    const int gqa_ratio = n_head_q / n_head_kv;
    const int gqa_grps  = (gqa_ratio + 15) / 16;

    const dim3 grid(nt_total, n_head_kv, gqa_grps);
    const dim3 blk(32, 4, 1);
    qsa_gqa_transposed_attn_kernel<kv_type, kv_type, HEAD_DIM><<<grid, blk, 0, stream>>>(
        (const float *) q->data, (const char *) k->data, (const char *) v->data,
        (const int32_t *) sel_cells->data, (float *) dst->data,
        n_head_q, n_head_kv, n_sel, nt_local, scale,
        q->nb[1], q->nb[2], q->nb[3],
        k->nb[1], k->nb[2], k->nb[3],
        v->nb[1], v->nb[2], v->nb[3],
        (int64_t)(sel_cells->nb[1] / sizeof(int32_t)),
        dst->nb[1], dst->nb[2], dst->nb[3]);
}

bool ggml_cuda_qsa_sparse_attn_supported(
        const ggml_tensor * q,
        const ggml_tensor * k,
        const ggml_tensor * v,
        const ggml_tensor * sel_cells) {
    if (!q || !k || !v || !sel_cells) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || sel_cells->type != GGML_TYPE_I32 || k->type != v->type) {
        return false;
    }
    if (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16 &&
        k->type != GGML_TYPE_Q8_0 && k->type != GGML_TYPE_Q4_0) {
        return false;
    }
    const int64_t D = q->ne[0];
    if ((D != 64 && D != 128 && D != 256) || k->ne[0] != D || v->ne[0] != D) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || sel_cells->nb[0] != sizeof(int32_t) || (q->ne[2] % k->ne[2]) != 0) {
        return false;
    }
    if (sel_cells->ne[1] != q->ne[1] * q->ne[3]) {
        return false;
    }
    return true;
}

void ggml_cuda_qsa_sparse_attn(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         q,
        const ggml_tensor *         k,
        const ggml_tensor *         v,
        const ggml_tensor *         sel_cells,
        ggml_tensor *               dst,
        float                       scale) {
    cudaStream_t stream = ctx.stream();
    const int D = (int) q->ne[0];
    switch (k->type) {
        case GGML_TYPE_F16:
            if      (D ==  64) { launch_qsa_dim<GGML_TYPE_F16,  64>(stream, q, k, v, sel_cells, dst, scale); }
            else if (D == 128) { launch_qsa_dim<GGML_TYPE_F16, 128>(stream, q, k, v, sel_cells, dst, scale); }
            else               { launch_qsa_dim<GGML_TYPE_F16, 256>(stream, q, k, v, sel_cells, dst, scale); }
            break;
        case GGML_TYPE_BF16:
            if      (D ==  64) { launch_qsa_dim<GGML_TYPE_BF16,  64>(stream, q, k, v, sel_cells, dst, scale); }
            else if (D == 128) { launch_qsa_dim<GGML_TYPE_BF16, 128>(stream, q, k, v, sel_cells, dst, scale); }
            else               { launch_qsa_dim<GGML_TYPE_BF16, 256>(stream, q, k, v, sel_cells, dst, scale); }
            break;
        case GGML_TYPE_Q8_0:
            if      (D ==  64) { launch_qsa_dim<GGML_TYPE_Q8_0,  64>(stream, q, k, v, sel_cells, dst, scale); }
            else if (D == 128) { launch_qsa_dim<GGML_TYPE_Q8_0, 128>(stream, q, k, v, sel_cells, dst, scale); }
            else               { launch_qsa_dim<GGML_TYPE_Q8_0, 256>(stream, q, k, v, sel_cells, dst, scale); }
            break;
        case GGML_TYPE_Q4_0:
            if      (D ==  64) { launch_qsa_dim<GGML_TYPE_Q4_0,  64>(stream, q, k, v, sel_cells, dst, scale); }
            else if (D == 128) { launch_qsa_dim<GGML_TYPE_Q4_0, 128>(stream, q, k, v, sel_cells, dst, scale); }
            else               { launch_qsa_dim<GGML_TYPE_Q4_0, 256>(stream, q, k, v, sel_cells, dst, scale); }
            break;
        default:
            GGML_ABORT("unsupported kv type in qsa_sparse_attn");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <ggml_type wtype>
static __global__ void hc_up_pre_fused_kernel(
        const float * __restrict__ xn,
        const char *  __restrict__ w_up,
        const float * __restrict__ lo,
        float *       __restrict__ dst,
        int                        n_embd,
        int                        hc,
        int                        hc_lr,
        float                      scale,
        int64_t                    sx1,
        int64_t                    sx2,
        size_t                     nbw1,
        int64_t                    slo1,
        int64_t                    sd1) {
    __shared__ float s_lo[64];

    const int tok = blockIdx.y;
    for (int r = threadIdx.x; r < hc_lr; r += blockDim.x) {
        s_lo[r] = lo[(int64_t) tok * slo1 + r];
    }
    __syncthreads();

    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i0 >= n_embd) {
        return;
    }

    float acc_mix = 0.0f;
    for (int ih = 0; ih < hc; ++ih) {
        const int row_idx   = ih * n_embd + i0;
        const char * w_row  = w_up + (size_t) row_idx * nbw1;
        float gate_val      = 0.0f;
        if constexpr (wtype == GGML_TYPE_F32) {
            const float * wf = (const float *) w_row;
            for (int r = 0; r < hc_lr; ++r) { gate_val += wf[r] * s_lo[r]; }
        } else if constexpr (wtype == GGML_TYPE_F16) {
            const half * wh = (const half *) w_row;
            for (int r = 0; r < hc_lr; ++r) { gate_val += __half2float(wh[r]) * s_lo[r]; }
        } else if constexpr (wtype == GGML_TYPE_BF16) {
            const uint16_t * wb = (const uint16_t *) w_row;
            for (int r = 0; r < hc_lr; ++r) { gate_val += __uint_as_float(((uint32_t) wb[r]) << 16) * s_lo[r]; }
        }
        const float sig = 1.0f / (1.0f + expf(-gate_val));
        const float xv  = xn[(int64_t) tok * sx2 + (int64_t) ih * sx1 + i0];
        acc_mix += xv * sig;
    }
    dst[(int64_t) tok * sd1 + i0] = acc_mix * scale;
}

bool ggml_cuda_hc_up_pre_supported(
        const ggml_tensor * up_mm,
        const ggml_tensor * hc_pre) {
    if (!ggml_cuda_hc_fused_enabled() || !up_mm || !hc_pre) {
        return false;
    }
    if (up_mm->op != GGML_OP_MUL_MAT || hc_pre->op != GGML_OP_DSV4_HC_PRE) {
        return false;
    }
    if (ggml_get_op_params_i32(hc_pre, 1) == 0) {
        return false;
    }
    const ggml_tensor * w_up    = up_mm->src[0];
    const ggml_tensor * lo      = up_mm->src[1];
    const ggml_tensor * xn      = hc_pre->src[0];
    const ggml_tensor * gate_v  = hc_pre->src[1];
    if (!w_up || !lo || !xn || !gate_v) {
        return false;
    }
    if (gate_v != up_mm && !(gate_v->op == GGML_OP_RESHAPE && gate_v->src[0] == up_mm)) {
        return false;
    }
    if (xn->type != GGML_TYPE_F32 || lo->type != GGML_TYPE_F32 || hc_pre->type != GGML_TYPE_F32) {
        return false;
    }
    if (w_up->type != GGML_TYPE_F32 && w_up->type != GGML_TYPE_F16 && w_up->type != GGML_TYPE_BF16) {
        return false;
    }
    const int64_t hc_lr = w_up->ne[0];
    if (hc_lr <= 0 || hc_lr > 64 || w_up->ne[2] != 1 || w_up->ne[3] != 1 || lo->ne[2] != 1 || lo->ne[3] != 1) {
        return false;
    }
    if (w_up->nb[0] != ggml_type_size(w_up->type) || lo->nb[0] != sizeof(float) ||
        xn->nb[0] != sizeof(float) || hc_pre->nb[0] != sizeof(float)) {
        return false;
    }
    if ((xn->nb[1] % sizeof(float)) != 0 || (xn->nb[2] % sizeof(float)) != 0 ||
        (lo->nb[1] % sizeof(float)) != 0 || (hc_pre->nb[1] % sizeof(float)) != 0) {
        return false;
    }
    if (xn->ne[2] > 64) {
        return false;   // Large prefill batches use Tensor-Core cuBLAS GEMM + DSV4_HC_PRE
    }
    return w_up->ne[1] == xn->ne[0] * xn->ne[1] && lo->ne[1] == xn->ne[2];
}

void ggml_cuda_hc_up_pre_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         up_mm,
        ggml_tensor *               hc_pre) {
    const ggml_tensor * w_up = up_mm->src[0];
    const ggml_tensor * lo   = up_mm->src[1];
    const ggml_tensor * xn   = hc_pre->src[0];

    const int   n_embd = (int) xn->ne[0];
    const int   hc     = (int) xn->ne[1];
    const int   nt     = (int) xn->ne[2];
    const int   hc_lr  = (int) w_up->ne[0];
    const float scale  = ggml_get_op_params_f32(hc_pre, 0);

    const dim3 blk(256, 1, 1);
    const dim3 grid((n_embd + 255) / 256, nt, 1);
    const int64_t sx1  = (int64_t)(xn->nb[1] / sizeof(float));
    const int64_t sx2  = (int64_t)(xn->nb[2] / sizeof(float));
    const int64_t slo1 = (int64_t)(lo->nb[1] / sizeof(float));
    const int64_t sd1  = (int64_t)(hc_pre->nb[1] / sizeof(float));

    switch (w_up->type) {
        case GGML_TYPE_F32:
            hc_up_pre_fused_kernel<GGML_TYPE_F32><<<grid, blk, 0, ctx.stream()>>>(
                (const float *) xn->data, (const char *) w_up->data, (const float *) lo->data, (float *) hc_pre->data,
                n_embd, hc, hc_lr, scale, sx1, sx2, w_up->nb[1], slo1, sd1);
            break;
        case GGML_TYPE_F16:
            hc_up_pre_fused_kernel<GGML_TYPE_F16><<<grid, blk, 0, ctx.stream()>>>(
                (const float *) xn->data, (const char *) w_up->data, (const float *) lo->data, (float *) hc_pre->data,
                n_embd, hc, hc_lr, scale, sx1, sx2, w_up->nb[1], slo1, sd1);
            break;
        case GGML_TYPE_BF16:
            hc_up_pre_fused_kernel<GGML_TYPE_BF16><<<grid, blk, 0, ctx.stream()>>>(
                (const float *) xn->data, (const char *) w_up->data, (const float *) lo->data, (float *) hc_pre->data,
                n_embd, hc, hc_lr, scale, sx1, sx2, w_up->nb[1], slo1, sd1);
            break;
        default:
            GGML_ABORT("unsupported w_up type in hc_up_pre_fused");
    }
    CUDA_CHECK(cudaGetLastError());
}

__device__ __constant__ int8_t k_ple_iq4nl_lut[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static __global__ void ple_gather_dequant_kernel(
        const char *    __restrict__ uniq_raw,
        ggml_type                    ple_type,
        const int32_t * __restrict__ map_idx,
        float *         __restrict__ dst,
        int                          head_dim,
        int                          n_rows_out,
        size_t                       row_bytes) {
    const int64_t idx   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t) head_dim * n_rows_out;
    if (idx >= total) {
        return;
    }
    const int d   = (int)(idx % head_dim);
    const int r   = (int)(idx / head_dim);
    const int u   = map_idx[r];
    const char * src = uniq_raw + (size_t) u * row_bytes;

    float val = 0.0f;
    if (ple_type == GGML_TYPE_IQ4_NL) {
        const block_iq4_nl * b = (const block_iq4_nl *) src + (d >> 5);
        const int j            = d & 31;
        const uint8_t byte     = b->qs[j & 15];
        const uint8_t nib      = (j < 16) ? (byte & 0x0Fu) : (byte >> 4);
        val = __half2float(b->d) * (float) k_ple_iq4nl_lut[nib];
    } else if (ple_type == GGML_TYPE_Q8_0) {
        const block_q8_0 * b = (const block_q8_0 *) src + (d >> 5);
        val = __half2float(b->d) * (float) b->qs[d & 31];
    } else if (ple_type == GGML_TYPE_Q4_0) {
        const block_q4_0 * b = (const block_q4_0 *) src + (d >> 5);
        const int j          = d & 31;
        const uint8_t byte   = b->qs[j & 15];
        const int nib        = (j < 16) ? (byte & 0x0Fu) : (byte >> 4);
        val = __half2float(b->d) * (float)(nib - 8);
    } else if (ple_type == GGML_TYPE_F16) {
        val = __half2float(((const half *) src)[d]);
    } else if (ple_type == GGML_TYPE_BF16) {
        val = __uint_as_float(((uint32_t)((const uint16_t *) src)[d]) << 16);
    } else if (ple_type == GGML_TYPE_F32) {
        val = ((const float *) src)[d];
    }
    dst[idx] = val;
}

void ggml_cuda_ple_gather_dequant(
        ggml_backend_cuda_context & ctx,
        const void *                uniq_raw_d,
        ggml_type                   ple_type,
        const int32_t *             map_idx_d,
        float *                     dst_d,
        int                         head_dim,
        int                         n_rows_out) {
    const size_t  row_bytes = ggml_row_size(ple_type, head_dim);
    const int64_t total     = (int64_t) head_dim * n_rows_out;
    ple_gather_dequant_kernel<<<(int)((total + 255) / 256), 256, 0, ctx.stream()>>>(
        (const char *) uniq_raw_d, ple_type, map_idx_d, dst_d, head_dim, n_rows_out, row_bytes);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void ple_dilated_conv_silu_kernel(
        const float * __restrict__ padded,
        const char *  __restrict__ conv_w,
        ggml_type                  w_type,
        float *       __restrict__ dst,
        int                        hc_dim,
        int                        n_seq_tokens,
        int                        n_seqs,
        int                        kern,
        int                        dil,
        int                        hist,
        size_t                     nbp1,
        size_t                     nbp2,
        size_t                     nbw0,
        size_t                     nbw1,
        int64_t                    sd1) {
    const int64_t idx   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t) hc_dim * n_seq_tokens * n_seqs;
    if (idx >= total) {
        return;
    }
    const int c     = (int)(idx % hc_dim);
    const int t_all = (int)(idx / hc_dim);
    const int t_seq = t_all % n_seq_tokens;
    const int seq   = t_all / n_seq_tokens;

    const float * p_chan = (const float *)((const char *) padded + (size_t) c * nbp1 + (size_t) seq * nbp2);
    const char *  w_chan = conv_w + (size_t) c * nbw1;

    float acc = 0.0f;
    for (int k = 0; k < kern; ++k) {
        const int pos = hist - (kern - 1 - k) * dil + t_seq;
        float wk = 0.0f;
        const char * wp = w_chan + (size_t) k * nbw0;
        if      (w_type == GGML_TYPE_F32)  { wk = *(const float *) wp; }
        else if (w_type == GGML_TYPE_F16)  { wk = __half2float(*(const half *) wp); }
        else if (w_type == GGML_TYPE_BF16) { wk = __uint_as_float(((uint32_t)(*(const uint16_t *) wp)) << 16); }
        acc += wk * p_chan[pos];
    }
    dst[(int64_t) t_all * sd1 + c] = acc / (1.0f + expf(-acc));
}

bool ggml_cuda_ple_conv_silu_supported(
        const ggml_cgraph *  cgraph,
        int                  node_idx,
        const ggml_tensor ** out_padded,
        const ggml_tensor ** out_conv_w,
        int *                out_kern,
        int *                out_dil,
        int *                out_span) {
    if (!ggml_cuda_ple_fast_enabled() || !cgraph || node_idx < 0 || node_idx + 6 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * c0 = cgraph->nodes[node_idx];
    if (!c0 || c0->op != GGML_OP_CONT || !c0->src[0] || c0->src[0]->op != GGML_OP_TRANSPOSE) {
        return false;
    }
    const ggml_tensor * t0 = c0->src[0];
    const ggml_tensor * v0 = t0->src[0];
    if (!v0 || v0->op != GGML_OP_VIEW || !v0->src[0] || v0->op_params[0] != 0) {
        return false;
    }
    const ggml_tensor * padded = v0->src[0];
    if (padded->type != GGML_TYPE_F32 || padded->nb[0] != sizeof(float) || v0->ne[3] != 1 || padded->ne[3] != 1) {
        return false;
    }
    const int64_t n_seq_tokens = v0->ne[0];
    const int64_t hc_dim       = v0->ne[1];
    const int64_t n_seqs       = v0->ne[2];
    const int64_t hist         = padded->ne[0] - n_seq_tokens;
    if (n_seq_tokens <= 0 || hc_dim <= 0 || n_seqs <= 0 || hist <= 0 ||
        padded->ne[1] != hc_dim || padded->ne[2] != n_seqs) {
        return false;
    }

    int p = node_idx;
    const ggml_tensor * conv_w = nullptr;
    int kern = 0;
    int dil  = 0;
    const ggml_tensor * acc = nullptr;

    for (int k = 0; k < (kern == 0 ? 1 : kern); ++k) {
        const ggml_tensor * vp = nullptr;
        const ggml_tensor * tp = nullptr;
        const ggml_tensor * cp = nullptr;
        if (k == 0) {
            vp = v0;
            tp = t0;
            cp = cgraph->nodes[p++];
        } else {
            if (p + 5 >= cgraph->n_nodes) { return false; }
            vp = cgraph->nodes[p++];
            tp = cgraph->nodes[p++];
            cp = cgraph->nodes[p++];
        }
        if (p + 3 >= cgraph->n_nodes) { return false; }
        const ggml_tensor * vw = cgraph->nodes[p++];
        const ggml_tensor * cw = cgraph->nodes[p++];
        const ggml_tensor * rw = cgraph->nodes[p++];
        if (!vp || !tp || !cp || !vw || !cw || !rw) { return false; }
        if (vp->op != GGML_OP_VIEW      || vp->src[0] != padded) { return false; }
        if (tp->op != GGML_OP_TRANSPOSE || tp->src[0] != vp)     { return false; }
        if (cp->op != GGML_OP_CONT      || cp->src[0] != tp)     { return false; }
        if (vw->op != GGML_OP_VIEW      || !vw->src[0])          { return false; }
        if (cw->op != GGML_OP_CONT      || cw->src[0] != vw)     { return false; }
        if (rw->op != GGML_OP_RESHAPE   || rw->src[0] != cw)     { return false; }

        if (k == 0) {
            conv_w = vw->src[0];
            if (conv_w->type != GGML_TYPE_F32 && conv_w->type != GGML_TYPE_F16 && conv_w->type != GGML_TYPE_BF16) {
                return false;
            }
            kern = (int) conv_w->ne[0];
            if (kern < 2 || kern > 16 || conv_w->ne[1] != hc_dim || conv_w->ne[2] != 1 || conv_w->ne[3] != 1) {
                return false;
            }
            if ((hist % (kern - 1)) != 0) {
                return false;
            }
            dil = (int)(hist / (kern - 1));
            if (dil < 1) { return false; }
        } else if (vw->src[0] != conv_w) {
            return false;
        }

        const int64_t start = hist - (int64_t)(kern - 1 - k) * dil;
        if ((size_t)(uint32_t) vp->op_params[0] != (size_t) start * sizeof(float) ||
            (size_t)(uint32_t) vw->op_params[0] != (size_t) k * conv_w->nb[0]) {
            return false;
        }
        if (vp->ne[0] != n_seq_tokens || vp->ne[1] != hc_dim || vp->ne[2] != n_seqs ||
            vp->nb[1] != padded->nb[1] || vp->nb[2] != padded->nb[2]) {
            return false;
        }

        const ggml_tensor * wk = rw;
        if (conv_w->type != GGML_TYPE_F32) {
            if (p >= cgraph->n_nodes) { return false; }
            const ggml_tensor * fw = cgraph->nodes[p++];
            if (!fw || fw->op != GGML_OP_CPY || fw->src[0] != rw || fw->type != GGML_TYPE_F32) {
                return false;
            }
            wk = fw;
        }
        if (p >= cgraph->n_nodes) { return false; }
        const ggml_tensor * mul = cgraph->nodes[p++];
        if (!mul || mul->op != GGML_OP_MUL || mul->src[0] != cp || mul->src[1] != wk || mul->type != GGML_TYPE_F32) {
            return false;
        }
        if (k == 0) {
            acc = mul;
        } else {
            if (p >= cgraph->n_nodes) { return false; }
            const ggml_tensor * add = cgraph->nodes[p++];
            if (!add || add->op != GGML_OP_ADD || add->src[0] != acc || add->src[1] != mul || add->type != GGML_TYPE_F32) {
                return false;
            }
            acc = add;
        }
    }

    if (p >= cgraph->n_nodes) { return false; }
    const ggml_tensor * silu = cgraph->nodes[p++];
    if (!silu || silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || silu->src[0] != acc) {
        return false;
    }
    if (p < cgraph->n_nodes) {
        const ggml_tensor * next_c = cgraph->nodes[p];
        const size_t h_s = ggml_hash_find(&cgraph->visited_hash_set, silu);
        if (next_c && next_c->op == GGML_OP_CONT && next_c->src[0] == silu &&
            h_s != GGML_HASHSET_FULL && cgraph->use_counts[h_s] == 1) {
            ++p;
        }
    }

    const int out_idx = p - 1;
    for (int m = node_idx; m < out_idx; ++m) {
        const size_t h = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[m]);
        if (h == GGML_HASHSET_FULL || cgraph->use_counts[h] != 1) {
            return false;
        }
    }
    const ggml_tensor * dst = cgraph->nodes[out_idx];
    if (!dst || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || dst->ne[0] != hc_dim) {
        return false;
    }

    if (out_padded) { *out_padded = padded; }
    if (out_conv_w) { *out_conv_w = conv_w; }
    if (out_kern)   { *out_kern   = kern; }
    if (out_dil)    { *out_dil    = dil; }
    if (out_span)   { *out_span   = p - node_idx; }
    return true;
}

void ggml_cuda_ple_dilated_conv_silu(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         padded,
        const ggml_tensor *         conv_w,
        ggml_tensor *               dst,
        int                         kern,
        int                         dil) {
    const int hc_dim       = (int) padded->ne[1];
    const int n_seqs       = (int) padded->ne[2];
    const int hist         = (kern - 1) * dil;
    const int n_seq_tokens = (int) padded->ne[0] - hist;
    const int64_t total    = (int64_t) hc_dim * n_seq_tokens * n_seqs;

    ple_dilated_conv_silu_kernel<<<(int)((total + 255) / 256), 256, 0, ctx.stream()>>>(
        (const float *) padded->data, (const char *) conv_w->data, conv_w->type, (float *) dst->data,
        hc_dim, n_seq_tokens, n_seqs, kern, dil, hist,
        padded->nb[1], padded->nb[2], conv_w->nb[0], conv_w->nb[1],
        (int64_t)(dst->nb[1] / sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
}

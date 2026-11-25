#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"
#include "moe-fused-prefill.cuh"

#include <cstdint>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const ggml_prec prec_src1) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            // src1 at Q4 uses the native FP4 instructions, which are Blackwell-only
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q8>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// overrides the src1 precision requested by the graph, "auto" keeps the requested one
static ggml_prec ggml_cuda_mmq_get_prec_env() {
    const char * env_c = getenv("GGML_CUDA_MMQ_PREC");
    if (env_c == nullptr) {
        return GGML_PREC_UNDEFINED;
    }
    std::string env_cpp = env_c;
    for (char & c : env_cpp) {
        c = std::tolower(c);
    }
    if (env_cpp == "q4") {
        return GGML_PREC_Q4;
    }
    if (env_cpp == "q8") {
        return GGML_PREC_Q8;
    }
    if (env_cpp != "auto") {
        GGML_LOG_WARN("%s: Unknown value for GGML_CUDA_MMQ_PREC: '%s'. Available: 'q4', 'q8', 'auto'.\n", __func__, env_cpp.c_str());
    }
    return GGML_PREC_UNDEFINED;
}

// src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they
// default to the native W4A4 instructions on Blackwell.
static ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_tensor * src0, const ggml_tensor * dst, const int cc) {
    static const ggml_prec prec_env = ggml_cuda_mmq_get_prec_env();

    ggml_prec prec = prec_env;
    if (prec == GGML_PREC_UNDEFINED) {
        prec = (ggml_prec) ggml_get_op_params_i32(dst, 3);
    }

    // Q4 only for the FP4 types on Blackwell
    GGML_ASSERT(prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4);
    const bool can_use_q4 = (src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc);
    if (prec == GGML_PREC_Q8 || !can_use_q4) {
        return GGML_PREC_Q8;
    }
    return GGML_PREC_Q4;
}

// Tile width in ne11 (dense) / ne12 (MoE) direction with the fewest tiles that fits into shared memory, 0 if there is none.
static int ggml_cuda_mmq_get_J_best(
        const ggml_type type, const int64_t ncols, const bool fallback, const int cc, const size_t smpbo, const ggml_prec prec_src1,
        int * nthreads_best = nullptr) {
    int J_best        = 0;
    int ntiles_J_best = INT_MAX;

    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }

        if (mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }

        const int ntiles_x = (ncols + config.J - 1) / config.J;

        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            if (nthreads_best) {
                *nthreads_best = config.nthreads;
            }
            ntiles_J_best = ntiles_x;
        }
    }
    return J_best;
}

static __global__ void build_mmid_work_tiles_kernel(
        const int32_t * __restrict__ expert_bounds,
        int2 * __restrict__ work_tiles,
        int * __restrict__ n_work_tiles,
        const int ne02_active,
        const int J,
        const int max_tiles) {
    const int zt = blockIdx.x * blockDim.x + threadIdx.x;
    if (zt >= ne02_active) {
        return;
    }
    const int col_low  = expert_bounds[zt + 0];
    const int col_high = expert_bounds[zt + 1];
    const int col_diff = col_high - col_low;
    if (col_diff <= 0) {
        return;
    }
    const int n_jt = (col_diff + J - 1) / J;
    const int base = atomicAdd(n_work_tiles, n_jt);
    for (int jt = 0; jt < n_jt; ++jt) {
        if (base + jt < max_tiles) {
            work_tiles[base + jt] = make_int2(zt, jt);
        }
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    // Fused gate/up/GLU: src0 is the up weight, dst the GLU output.
    const ggml_tensor * gate = fusion ? fusion->gate : nullptr;
    ggml_cuda_mm_fusion_args_device fusion_device = {};
    if (fusion) {
        GGML_ASSERT(gate && !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
        GGML_ASSERT(gate->type == src0->type && ggml_are_same_shape(gate, src0) && ggml_are_same_stride(gate, src0));
        fusion_device.gate      = gate->data;
        fusion_device.glu_op    = fusion->glu_op;
        fusion_device.glu_limit = fusion->glu_limit;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const int    id    = ggml_cuda_get_device();
    const int    cc    = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 (or the fused gate weight) is a temporary compute buffer, clear any potential padding.
    for (const ggml_tensor * t : { src0, gate }) {
        if (t && !(t->flags & GGML_TENSOR_FLAG_PREZERO_PAD) &&
                ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            const size_t size_data  = ggml_nbytes(t);
            const size_t size_alloc = ggml_backend_buffer_get_alloc_size(t->buffer, t);
            if (size_alloc > size_data) {
                GGML_ASSERT(ggml_is_contiguously_allocated(t));
                GGML_ASSERT(!t->view_src);
                CUDA_CHECK(cudaMemsetAsync((char *) t->data + size_data, 0, size_alloc - size_data, stream));
            }
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ggml_cuda_mmq_needs_fallback(ne01);

    // With fusion dst is the GLU output, not a MUL_MAT; the fused types always quantize src1 to Q8_1.
    // TODO support for Q4 precision if both tensors allow it
    const ggml_prec prec_src1 = gate ? GGML_PREC_Q8 : ggml_cuda_mmq_get_prec_src1(src0, dst, cc);

    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    const int moe_nz = ids ? ggml_cuda_moe_zero_tail_of(fusion && fusion->zero_tail_node ? fusion->zero_tail_node : dst) : 0;

    int J_best        = 0;
    int nthreads_best = 0;
    {
        int64_t ncols_opt = ne11;
        if (ids) {
            const int64_t n_expert_used = ids->ne[0];
            ncols_opt = ne12;

            // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
            if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
                ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
            } else if (ggml_cuda_mmq_moe_opt_enabled() && cc >= GGML_CUDA_CC_AMPERE && ne02 >= 64) {
                const int64_t ne02_est = (moe_nz > 0 && moe_nz < ne02) ? (ne02 - moe_nz) : ne02;
                const int64_t avg_tok  = (ne12 * n_expert_used + ne02_est - 1) / ne02_est;
                ncols_opt = std::min<int64_t>(ne12, std::max<int64_t>(16, (avg_tok * 5 + 3) / 4));
            }
        }

        J_best = ggml_cuda_mmq_get_J_best(src0->type, ncols_opt, fallback, cc, smpbo, prec_src1, &nthreads_best);
    }
    GGML_ASSERT(J_best > 0);

    // A tile of size J can read in at most J - 1 extra columns.
    // For simplicity, round up the padding of a full tile to a multiple of the number of bytes that nthreads can load in parallel.
    const size_t src1_load_chunk_size = nthreads_best * sizeof(int);
    const size_t src1_q8_1_padding = ((J_best * sizeof(block_q8_1_mmq) + src1_load_chunk_size - 1) / src1_load_chunk_size)
        * src1_load_chunk_size;

    const auto mmid_pop_s1 = [&]() {
        if (ctx.mmid_share.scale_buf != nullptr) {
            ctx.pool().free(ctx.mmid_share.scale_buf, ctx.mmid_share.scale_cap);
            ctx.mmid_share.scale_buf = nullptr;
            ctx.mmid_share.scale_cap = 0;
        }
        if (ctx.mmid_share.q8_buf != nullptr) {
            ctx.pool().free(ctx.mmid_share.q8_buf, ctx.mmid_share.q8_cap);
            ctx.mmid_share.q8_buf = nullptr;
            ctx.mmid_share.q8_cap = 0;
        }
        ctx.mmid_share.src1 = nullptr;
    };
    const auto mmid_pop_all = [&]() {
        mmid_pop_s1();
        if (ctx.mmid_share.expert_bounds != nullptr) {
            ctx.pool().free(ctx.mmid_share.expert_bounds, ctx.mmid_share.bounds_cap);
            ctx.mmid_share.expert_bounds = nullptr;
            ctx.mmid_share.bounds_cap    = 0;
        }
        if (ctx.mmid_share.ids_dst != nullptr) {
            ctx.pool().free(ctx.mmid_share.ids_dst, ctx.mmid_share.ids_dst_cap);
            ctx.mmid_share.ids_dst     = nullptr;
            ctx.mmid_share.ids_dst_cap = 0;
        }
        ctx.mmid_share.ids = nullptr;
    };

    if (!ids) {
        const int64_t s11_in = src1->nb[1] / ts_src1;
        const int64_t s12_in = src1->nb[2] / ts_src1;
        const int64_t s13_in = src1->nb[3] / ts_src1;
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block + src1_q8_1_padding;
        const int dense_q8_fmt = 2000 + (int) mmq_get_q8_1_ds_layout(src0->type);
        const bool can_share_dense = ctx.mmid_share.active && ggml_cuda_mmq_moe_opt_enabled() && !use_native_fp4 &&
                                     ne10 > 0 && ne11 > 0 && ne12 > 0 && ne13 > 0 &&
                                     (int64_t) (int) s11_in == s11_in && (int64_t) (int) ne13 == ne13 &&
                                     ctx.q8_1_share.is_graph_input(src1);
        const char * s1_lo = (const char *) src1->data;
        const size_t s1_span = can_share_dense
            ? (size_t) (ne13 - 1) * src1->nb[3] + (size_t) (ne12 - 1) * src1->nb[2] + (size_t) (ne11 - 1) * src1->nb[1] + (size_t) ne10 * ts_src1
            : 0;
        const bool hit_dense = can_share_dense &&
                               ctx.mmid_share.ids == src1 && ctx.mmid_share.lo == s1_lo &&
                               ctx.mmid_share.src1 == src1 && ctx.mmid_share.s1_lo == s1_lo &&
                               ctx.mmid_share.ne10 == ne10 && ctx.mmid_share.ne10_padded == ne10_padded &&
                               ctx.mmid_share.n_expert_used == ne11 && ctx.mmid_share.ne12 == ne12 &&
                               ctx.mmid_share.moe_nz == (int) ne13 &&
                               ctx.mmid_share.si1 == (int) s11_in && ctx.mmid_share.s12 == s12_in &&
                               ctx.mmid_share.ne02 == s13_in &&
                               ctx.mmid_share.q8_fmt == dense_q8_fmt &&
                               ctx.mmid_share.q8_pad >= src1_q8_1_padding &&
                               ctx.mmid_share.q8_buf != nullptr;

        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        char * q8_ptr = nullptr;

        if (hit_dense) {
            q8_ptr = ctx.mmid_share.q8_buf;
        } else {
            mmid_pop_all();
            if (can_share_dense) {
                ctx.mmid_share.q8_buf = (char *) ctx.pool().alloc(nbytes_src1_q8_1, &ctx.mmid_share.q8_cap);
                q8_ptr = ctx.mmid_share.q8_buf;
            } else {
                src1_q8_1.alloc(nbytes_src1_q8_1);
                q8_ptr = src1_q8_1.get();
            }
            if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
                src1_scale.alloc(ne13*ne12*ne11);
            }

            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, q8_ptr, src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11_in, s12_in, s13_in, ne10_padded,
                                        ne11, ne12, ne13, stream);
            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, q8_ptr, src0->type, ne10, s11_in, s12_in, s13_in, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());

            if (can_share_dense) {
                ctx.mmid_share.ids           = src1;
                ctx.mmid_share.lo            = s1_lo;
                ctx.mmid_share.hi            = s1_lo + s1_span;
                ctx.mmid_share.src1          = src1;
                ctx.mmid_share.s1_lo         = s1_lo;
                ctx.mmid_share.s1_hi         = s1_lo + s1_span;
                ctx.mmid_share.ne02          = s13_in;
                ctx.mmid_share.ne12          = ne12;
                ctx.mmid_share.n_expert_used = ne11;
                ctx.mmid_share.ne10          = ne10;
                ctx.mmid_share.ne10_padded   = ne10_padded;
                ctx.mmid_share.s12           = s12_in;
                ctx.mmid_share.si1           = (int) s11_in;
                ctx.mmid_share.moe_nz        = (int) ne13;
                ctx.mmid_share.pad_j         = J_best;
                ctx.mmid_share.q8_fmt        = dense_q8_fmt;
                ctx.mmid_share.q8_pad        = src1_q8_1_padding;
            }
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) q8_ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, J_best, fusion_device};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
    const int si1  = ids->nb[1] / ggml_element_size(ids);
    const int sis1 = nb12 / nb11;

    const bool can_share_ids = ctx.mmid_share.active && ggml_cuda_mmq_moe_opt_enabled() && ctx.q8_1_share.is_graph_input(ids);
    const bool can_share_s1  = can_share_ids && dedup_bcast && ctx.q8_1_share.is_graph_input(src1);
    const char * ids_lo      = (const char *) ids->data;
    const size_t ids_span    = ne12 > 0 ? (size_t)(ne12 - 1) * ids->nb[1] + (size_t) n_expert_used * sizeof(int32_t) : 0;
    const char * s1_lo       = (const char *) src1->data;
    const size_t s1_span     = ne12 > 0 ? (size_t)(ne12 - 1) * src1->nb[2] + (size_t) ne10 * ts_src1 : 0;
    const int64_t s12_in     = src1->nb[2] / ts_src1;
    const int q8_fmt         = use_native_fp4 ? (1000 + (int) src0->type) : (int) mmq_get_q8_1_ds_layout(src0->type);
    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block + src1_q8_1_padding;
    const bool need_scale    = (src0->type == GGML_TYPE_NVFP4 && use_native_fp4);

    const bool ids_match = can_share_ids &&
                           ctx.mmid_share.ids == ids && ctx.mmid_share.lo == ids_lo &&
                           ctx.mmid_share.ne02 == ne02 && ctx.mmid_share.ne12 == ne12 &&
                           ctx.mmid_share.n_expert_used == n_expert_used &&
                           ctx.mmid_share.si1 == si1 && ctx.mmid_share.moe_nz == moe_nz &&
                           ctx.mmid_share.pad_j >= J_best;

    const bool hit_q8 = can_share_s1 && ids_match &&
                        ctx.mmid_share.src1 == src1 && ctx.mmid_share.s1_lo == s1_lo &&
                        ctx.mmid_share.ne10 == ne10 && ctx.mmid_share.ne10_padded == ne10_padded &&
                        ctx.mmid_share.s12 == s12_in && ctx.mmid_share.q8_fmt == q8_fmt &&
                        ctx.mmid_share.q8_pad >= src1_q8_1_padding;

    const bool hit_down = ids_match && !dedup_bcast &&
                          sis1 == (int) n_expert_used && ne11 == n_expert_used;

    {
        ggml_cuda_pool_alloc<int32_t> ids_src1_pool(ctx.pool());
        ggml_cuda_pool_alloc<int32_t> ids_dst_pool(ctx.pool());
        ggml_cuda_pool_alloc<int32_t> expert_bounds_pool(ctx.pool());
        ggml_cuda_pool_alloc<char>    src1_q8_1_pool(ctx.pool());
        ggml_cuda_pool_alloc<float>   src1_scale_pool(ctx.pool());

        int32_t * ids_src1_ptr = nullptr;
        int32_t * ids_dst_ptr  = nullptr;
        int32_t * bounds_ptr   = nullptr;
        char *    q8_ptr       = nullptr;
        float *   scale_ptr    = nullptr;

        if (hit_q8 || hit_down) {
            if (hit_down) {
                mmid_pop_s1();
            }
            ids_dst_ptr  = ctx.mmid_share.ids_dst;
            ids_src1_ptr = ctx.mmid_share.ids_dst;
            bounds_ptr   = ctx.mmid_share.expert_bounds;
            if (moe_nz > 0 && moe_nz < ne02 && !(fusion && fusion->skip_zero_tail_rows)) {
                ggml_cuda_moe_zero_tail_rows_cached(bounds_ptr, ids_dst_ptr, dst_d, (int) ne02, moe_nz, (int) ne0, s1, (int) ne_get_rows, stream);
                CUDA_CHECK(cudaGetLastError());
            }
        } else if (can_share_ids) {
            mmid_pop_all();
            const int pad_j = std::max(J_best, 128);
            const size_t want_dst = (size_t) ne_get_rows + (size_t) pad_j;
            const size_t want_bnd = (size_t) ne02 + 2;
            ctx.mmid_share.ids_dst       = (int32_t *) ctx.pool().alloc(want_dst * sizeof(int32_t), &ctx.mmid_share.ids_dst_cap);
            ctx.mmid_share.expert_bounds = (int32_t *) ctx.pool().alloc(want_bnd * sizeof(int32_t), &ctx.mmid_share.bounds_cap);
            if (can_share_s1 && fusion == nullptr) {
                ctx.mmid_share.q8_buf = (char *) ctx.pool().alloc(nbytes_src1_q8_1, &ctx.mmid_share.q8_cap);
                if (need_scale) {
                    ctx.mmid_share.scale_buf = (float *) ctx.pool().alloc((size_t)(ne12 * n_expert_used) * sizeof(float), &ctx.mmid_share.scale_cap);
                }
            }
            ids_src1_pool.alloc(ne_get_rows);
            ids_src1_ptr = ids_src1_pool.get();
            ids_dst_ptr  = ctx.mmid_share.ids_dst;
            bounds_ptr   = ctx.mmid_share.expert_bounds;

            ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1_ptr, ids_dst_ptr, bounds_ptr,
                ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
            CUDA_CHECK(cudaGetLastError());
            if (moe_nz > 0 && moe_nz < ne02) {
                ggml_cuda_moe_zero_tail_apply(bounds_ptr, ids_dst_ptr, dst_d, (int) ne02, moe_nz, (int) ne0, s1, (int) ne_get_rows,
                        stream, /*skip_zero_rows =*/ fusion && fusion->skip_zero_tail_rows, /*save_orig_end =*/ true);
                CUDA_CHECK(cudaGetLastError());
            }

            ctx.mmid_share.ids           = ids;
            ctx.mmid_share.lo            = ids_lo;
            ctx.mmid_share.hi            = ids_lo + ids_span;
            ctx.mmid_share.src1          = nullptr;
            ctx.mmid_share.ne02          = ne02;
            ctx.mmid_share.ne12          = ne12;
            ctx.mmid_share.n_expert_used = n_expert_used;
            ctx.mmid_share.si1           = si1;
            ctx.mmid_share.moe_nz        = moe_nz;
            ctx.mmid_share.pad_j         = pad_j;
        } else {
            mmid_pop_all();
            ids_src1_pool.alloc(ne_get_rows);
            ids_dst_pool.alloc(ne_get_rows + J_best - 1);
            expert_bounds_pool.alloc(ne02 + 1);
            ids_src1_ptr = ids_src1_pool.get();
            ids_dst_ptr  = ids_dst_pool.get();
            bounds_ptr   = expert_bounds_pool.get();

            ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1_ptr, ids_dst_ptr, bounds_ptr,
                ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
            CUDA_CHECK(cudaGetLastError());
            if (moe_nz > 0 && moe_nz < ne02) {
                ggml_cuda_moe_zero_tail_apply(bounds_ptr, ids_dst_ptr, dst_d, (int) ne02, moe_nz, (int) ne0, s1, (int) ne_get_rows,
                        stream);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        const int32_t * active_lim_ptr = (ggml_cuda_mmq_moe_opt_enabled() && moe_nz > 0 && moe_nz < ne02)
            ? &bounds_ptr[ne02 - moe_nz] : nullptr;

        if (hit_q8) {
            q8_ptr    = ctx.mmid_share.q8_buf;
            scale_ptr = need_scale ? ctx.mmid_share.scale_buf : nullptr;
        } else {
            if (can_share_s1 && fusion == nullptr) {
                q8_ptr    = ctx.mmid_share.q8_buf;
                scale_ptr = need_scale ? ctx.mmid_share.scale_buf : nullptr;
            } else {
                src1_q8_1_pool.alloc(nbytes_src1_q8_1);
                q8_ptr = src1_q8_1_pool.get();
                if (need_scale) {
                    src1_scale_pool.alloc(ne12 * n_expert_used);
                    scale_ptr = src1_scale_pool.ptr;
                }
            }

            const int64_t ne11_flat = ne12*n_expert_used;
            const int64_t ne12_flat = 1;
            const int64_t ne13_flat = 1;
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;

            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                if (dedup_bcast) {
                    quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1_ptr, q8_ptr, scale_ptr, src0->type, use_aligned_float8, ne10,
                                            /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream, active_lim_ptr, J_best);
                } else {
                    quantize_mmq_fp4_cuda(src1_d, ids_src1_ptr, q8_ptr, scale_ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                            ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream, active_lim_ptr, J_best);
                }
            } else if (dedup_bcast) {
                quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1_ptr, q8_ptr, src0->type, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream, active_lim_ptr, J_best);
            } else {
                quantize_mmq_q8_1_cuda(src1_d, ids_src1_ptr, q8_ptr, src0->type, ne10, s11, s12, s13,
                                       ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream, active_lim_ptr, J_best);
            }
            CUDA_CHECK(cudaGetLastError());

            if (can_share_s1 && fusion == nullptr) {
                ctx.mmid_share.src1        = src1;
                ctx.mmid_share.s1_lo       = s1_lo;
                ctx.mmid_share.s1_hi       = s1_lo + s1_span;
                ctx.mmid_share.ne10        = ne10;
                ctx.mmid_share.ne10_padded = ne10_padded;
                ctx.mmid_share.s12         = s12_in;
                ctx.mmid_share.q8_fmt      = q8_fmt;
                ctx.mmid_share.q8_pad      = src1_q8_1_padding;
            }
        }

        static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
        const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                             ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        ggml_cuda_pool_alloc<int2> work_tiles_pool(ctx.pool());
        ggml_cuda_pool_alloc<int>  n_work_pool(ctx.pool());
        const int64_t ntx_virtual = (ne12 + J_best - 1) / J_best;
        int max_work_tiles = 0;
        if (ggml_cuda_mmq_moe_opt_enabled() && ne12 >= 32 && ne13 == 1 && ne02 * ntx_virtual >= 512) {
            const int ne02_active = (moe_nz > 0 && moe_nz < ne02) ? (int)(ne02 - moe_nz) : (int) ne02;
            const int64_t cap64   = std::min<int64_t>(ne02_active, ne_get_rows) + (ne_get_rows + J_best - 1) / J_best + 16;
            max_work_tiles        = (int) std::min<int64_t>(cap64, INT32_MAX / 2);
            work_tiles_pool.alloc((size_t) max_work_tiles);
            n_work_pool.alloc(1);
            CUDA_CHECK(cudaMemsetAsync(n_work_pool.get(), 0, sizeof(int), stream));
            const int nblk = (ne02_active + 255) / 256;
            if (nblk > 0) {
                build_mmid_work_tiles_kernel<<<nblk, 256, 0, stream>>>(
                    bounds_ptr, work_tiles_pool.get(), n_work_pool.get(), ne02_active, J_best, max_work_tiles);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
        const mmq_args args = {
            src0_d, src0->type, (const int *) q8_ptr, ids_dst_ptr, bounds_ptr, dst_d,
            scale_ptr,
            ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
            ne02, ne02, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne12, J_best, fusion_device,
            max_work_tiles > 0 ? work_tiles_pool.get() : nullptr,
            max_work_tiles > 0 ? n_work_pool.get()     : nullptr,
            max_work_tiles};

        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
    }

    if (hit_q8) {
        mmid_pop_s1();
    } else if (hit_down) {
        mmid_pop_all();
    }
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    // MUSA: the MMQ kernels compute wrong values on PH1 (MTT S5000).
    if (cc == GGML_CUDA_CC_PH1) {
        return false;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

bool ggml_cuda_mmq_fusion_is_efficient(const ggml_type type, const int64_t nrows_x, const int64_t ncols, const int64_t nchannels, const int device) {
    const int    cc    = ggml_cuda_info().devices[device].cc;
    const int    nsm   = ggml_cuda_info().devices[device].nsm;
    const size_t smpbo = ggml_cuda_info().devices[device].smpbo;

    // Same tile width as ggml_cuda_mul_mat_q; the fused kernel has no fallback, quantizes src1 to Q8_1 and writes I/2 output rows per tile.
    const bool fallback = false;
    const int  J_best   = ggml_cuda_mmq_get_J_best(type, ncols, fallback, cc, smpbo, GGML_PREC_Q8);
    if (J_best == 0) {
        return false;
    }
    const int64_t nrows_tile = ggml_cuda_mmq_get_config(type, J_best, fallback, cc, GGML_PREC_Q8).I / 2;
    const int64_t ntiles     = (ncols + J_best - 1)/J_best * ((nrows_x + nrows_tile - 1)/nrows_tile) * nchannels;
    // Same condition as launch_mul_mat_q uses to run the stream-k kernel with one block per tile.
    const int64_t tiles_nwaves             = (ntiles + nsm - 1) / nsm;
    const int64_t tiles_efficiency_percent = 100 * ntiles / (nsm*tiles_nwaves);
    return tiles_efficiency_percent >= 90 || (tiles_nwaves >= 4 && tiles_efficiency_percent >= 75);
}

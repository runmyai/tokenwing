#include "inkling2-fusions.cuh"
#include "inkling2_kernels/inkling2_core_bridge.h"
#include "ggml-backend-impl.h"

#include <cstring>

namespace {
struct ink2_custom_op_params_prefix {
    ggml_custom_op_t fun;
    int              n_tasks;
    void *           userdata;
};
} // namespace

const struct ggml_cuda_inkling2_api_t * ggml_cuda_inkling2_api(void) {
    return ink2_core_get_api();
}

bool ggml_cuda_inkling2_supports_op(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CUSTOM) {
        return false;
    }
    if (static_cast<uint32_t>(op->op_params[GGML_CUDA_INK2_OP_MAGIC_SLOT]) != GGML_CUDA_INK2_OP_MAGIC) {
        return false;
    }
    const int32_t kind = op->op_params[GGML_CUDA_INK2_OP_MAGIC_SLOT + 1];
    if (kind == GGML_CUDA_INK2_OP_ATTN_REL_SINK) {
        ink2_custom_op_params_prefix cop {};
        std::memcpy(&cop, op->op_params, sizeof(cop));
        const auto * sw = static_cast<const ggml_tensor * const *>(cop.userdata);
        if (sw == nullptr) {
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            if (sw[i] == nullptr || sw[i]->buffer == nullptr) {
                return false;
            }
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(sw[i]->buffer);
            ggml_backend_dev_t         dev  = buft ? ggml_backend_buft_get_device(buft) : nullptr;
            if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                return false;
            }
        }
    }
    return true;
}

void ggml_cuda_inkling2_compute_forward(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_inkling2_supports_op(dst));

    ink2_custom_op_params_prefix cop {};
    std::memcpy(&cop, dst->op_params, sizeof(cop));

    const int32_t kind      = dst->op_params[GGML_CUDA_INK2_OP_MAGIC_SLOT + 1];
    cudaStream_t  stream    = ctx.stream();
    int32_t       rc        = 0;

    switch (kind) {
        case GGML_CUDA_INK2_OP_BLOCK_ATTN_RES: {
            const int32_t n_blk          = dst->op_params[10];
            const int32_t n_q            = dst->op_params[11];
            const int32_t round_cur_bf16 = dst->op_params[12];
            const int32_t m_tokens       = static_cast<int32_t>(dst->src[0]->ne[1]);
            const float * d_blks[5]      = { nullptr, nullptr, nullptr, nullptr, nullptr };
            for (int32_t i = 0; i < n_blk && i < 5; ++i) {
                GGML_ASSERT(dst->src[2 + i] != nullptr && ggml_is_contiguous(dst->src[2 + i]));
                d_blks[i] = static_cast<const float *>(dst->src[2 + i]->data);
            }
            rc = ink2_core_op_block_attn_res(
                static_cast<const float *>(dst->src[0]->data),
                static_cast<const float *>(dst->src[1]->data),
                d_blks,
                static_cast<float *>(dst->data),
                n_blk, n_q, round_cur_bf16, m_tokens, stream);
            break;
        }
        case GGML_CUDA_INK2_OP_LOOP_MSG_GATE: {
            const int32_t m_tokens = static_cast<int32_t>(dst->ne[1]);
            rc = ink2_core_op_loop_msg_gate(
                static_cast<const float *>(dst->src[0]->data),
                static_cast<const float *>(dst->src[1]->data),
                dst->src[2]->data,
                static_cast<const float *>(dst->src[3]->data),
                static_cast<const float *>(dst->src[4]->data),
                static_cast<float *>(dst->data),
                m_tokens, stream);
            break;
        }
        case GGML_CUDA_INK2_OP_SCONV_STATIC: {
            const int32_t channels = static_cast<int32_t>(dst->ne[0]);
            const int32_t s_tokens = static_cast<int32_t>(dst->ne[1]);
            const int32_t b_seqs   = static_cast<int32_t>(dst->ne[2]);
            rc = ink2_core_op_sconv_static(
                static_cast<const float *>(dst->src[0]->data),
                static_cast<const float *>(dst->src[1]->data),
                static_cast<const float *>(dst->src[2]->data),
                static_cast<float *>(dst->data),
                channels, s_tokens, b_seqs, stream);
            break;
        }
        case GGML_CUDA_INK2_OP_SCONV_DYNAMIC: {
            const int32_t s_tokens = static_cast<int32_t>(dst->ne[1]);
            const int32_t b_seqs   = static_cast<int32_t>(dst->ne[2]);
            rc = ink2_core_op_sconv_dynamic(
                static_cast<const float *>(dst->src[0]->data),
                static_cast<const float *>(dst->src[1]->data),
                static_cast<const float *>(dst->src[2]->data),
                static_cast<const float *>(dst->src[3]->data),
                static_cast<float *>(dst->data),
                s_tokens, b_seqs, stream);
            break;
        }
        case GGML_CUDA_INK2_OP_PREP_KV: {
            const int32_t is_v     = dst->op_params[10];
            const int32_t m_tokens = static_cast<int32_t>(ggml_nelements(dst) / 1024);
            rc = ink2_core_op_prep_kv(
                static_cast<const float *>(dst->src[0]->data),
                static_cast<float *>(dst->data),
                is_v, m_tokens, stream);
            break;
        }
        case GGML_CUDA_INK2_OP_ATTN_REL_SINK: {
            const int32_t is_local = dst->op_params[10];
            const int32_t m_tokens = static_cast<int32_t>(dst->ne[1]);
            const ggml_tensor * tk = dst->src[1];
            const ggml_tensor * tv = dst->src[2];
            const auto * sw        = static_cast<const ggml_tensor * const *>(cop.userdata);
            rc = ink2_core_op_attn_rel_sink(
                static_cast<const float *>(dst->src[0]->data),
                tk->data,
                tv->data,
                static_cast<int32_t>(tk->type),
                static_cast<int32_t>(tk->ne[2]),
                static_cast<int32_t>(tk->ne[3]),
                tk->nb[1], tk->nb[2], tk->nb[3],
                tv->nb[1], tv->nb[2], tv->nb[3],
                static_cast<const float *>(dst->src[3]->data),
                static_cast<const float *>(sw[0]->data),
                static_cast<const float *>(sw[1]->data),
                static_cast<const float *>(sw[2]->data),
                static_cast<const float *>(dst->src[4]->data),
                static_cast<const float *>(dst->src[5]->data),
                static_cast<const int32_t *>(dst->src[6]->data),
                static_cast<float *>(dst->data),
                is_local, m_tokens, stream);
            break;
        }
        case GGML_CUDA_INK2_OP_SOFTCAP_MASK_HEAD: {
            const int32_t vocab_unpad = dst->op_params[10];
            float softcap = 30.0f;
            std::memcpy(&softcap, &dst->op_params[11], sizeof(float));
            const int32_t in_stride = static_cast<int32_t>(dst->src[0]->ne[0]);
            const int32_t out_vocab = static_cast<int32_t>(dst->ne[0]);
            const int32_t n_rows    = static_cast<int32_t>(dst->ne[1]);
            rc = ink2_core_op_softcap_mask_head(
                static_cast<const float *>(dst->src[0]->data),
                static_cast<float *>(dst->data),
                in_stride, out_vocab, vocab_unpad, softcap, n_rows, stream);
            break;
        }
        case GGML_CUDA_INK2_OP_MTP_STASH_TAP: {
            const int32_t tap_slot = dst->op_params[10];
            const int32_t m_tokens = static_cast<int32_t>(dst->ne[1]);
            rc = ink2_core_op_mtp_stash_tap(
                cop.userdata,
                static_cast<const float *>(dst->src[0]->data),
                static_cast<float *>(dst->data),
                tap_slot, m_tokens, stream);
            break;
        }
        case GGML_CUDA_INK2_OP_FUSED_TRUNK: {
            const int32_t b_act           = dst->op_params[10];
            const int32_t s_tokens        = dst->op_params[11];
            const int32_t single_seq_id   = dst->op_params[12];
            const int32_t has_tokens_src0 = dst->op_params[13];
            const int32_t opt_mask        = dst->op_params[14];
            const int32_t n_outputs       = dst->op_params[15];

            int next_src = 2;
            const int32_t * d_seq_ids = (opt_mask & 1) ? static_cast<const int32_t *>(dst->src[next_src++]->data) : nullptr;
            const int32_t * d_out_ids = nullptr;
            if (opt_mask & 2) {
                const ggml_tensor * t_oid = dst->src[next_src++];
                d_out_ids = (n_outputs > 0 && t_oid != nullptr) ? static_cast<const int32_t *>(t_oid->data) : nullptr;
            }

            rc = ink2_core_op_fused_trunk(
                cop.userdata,
                dst->src[0]->data,
                has_tokens_src0,
                static_cast<const int32_t *>(dst->src[1]->data),
                d_seq_ids,
                single_seq_id,
                d_out_ids,
                b_act,
                s_tokens,
                n_outputs,
                static_cast<float *>(dst->data),
                stream);
            break;
        }
        default:
            GGML_ABORT("Unknown Inkling-2 CUDA custom op kind=%d", kind);
    }

    if (rc != 0) {
        const char * err = ink2_core_get_api()->last_error(cop.userdata);
        GGML_ABORT("Inkling-2 CUDA op (kind=%d) failed: %s", kind, err ? err : "unknown error");
    }
    CUDA_CHECK(cudaGetLastError());
}

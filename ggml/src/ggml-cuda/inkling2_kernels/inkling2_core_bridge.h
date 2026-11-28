#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GGML_CUDA_INK2_OP_MAGIC
#define GGML_CUDA_INK2_OP_MAGIC      0x494E4B32u /* "INK2" */
#define GGML_CUDA_INK2_OP_MAGIC_SLOT 8

enum ggml_cuda_ink2_op_kind {
    GGML_CUDA_INK2_OP_BLOCK_ATTN_RES     = 1,
    GGML_CUDA_INK2_OP_LOOP_MSG_GATE      = 2,
    GGML_CUDA_INK2_OP_SCONV_STATIC       = 3,
    GGML_CUDA_INK2_OP_SCONV_DYNAMIC      = 4,
    GGML_CUDA_INK2_OP_ATTN_REL_SINK      = 5,
    GGML_CUDA_INK2_OP_SOFTCAP_MASK_HEAD  = 6,
    GGML_CUDA_INK2_OP_FUSED_TRUNK        = 7,
    GGML_CUDA_INK2_OP_MTP_STASH_TAP      = 8,
    GGML_CUDA_INK2_OP_PREP_KV            = 9,
};

struct ggml_ink2_layer_ptrs {
    int32_t is_local;
    int32_t is_loop;
    const void * wq_ext3;
    const void * wk_ext3;
    const void * wv_ext3;
    const void * wg_ext3;
    const void * wo_ext3;
    const void * wr_bf16;
    const void * w_sink_bf16;
    const float * q_clip_f32;
    const float * r_clip_f32;
    const float * rel_proj_t_f32;
    const float * attn_sconv_f32;
    const float * ffn_sconv_f32;
    const float * ffn_norm_f32;
    const float * router_w_f32;
    const float * router_b_f32;
    const float * attn_norm_f32;
    const float * k_sconv_f32;
    const float * v_sconv_f32;
    const float * q_in_norm_f32;
    const float * k_in_norm_f32;
    const float * v_in_norm_f32;
    const float * k_static_w_f32;
    const void *  k_wproj_bf16;
    const float * v_static_w_f32;
    const void *  v_wproj_bf16;
    const void *  ctrl_down_bf16[4];
    const float * ctrl_forget_f32[4];
    const float * ctrl_gain_f32[4];
    const void *  gate_exps_ext0;
    const void *  up_exps_ext0;
    const void *  down_exps_ext0;
};

struct ggml_ink2_mtp_layer_ptrs {
    int32_t is_local;
    const float * dflare_norm_f32;
    const float * in_norm_f32;
    const float * post_norm_f32;
    const float * q_norm_f32;
    const float * k_norm_f32;
    const float * k_norm_ctx_f32;
    const void *  wq_ext3;
    const void *  wk_ext3;
    const void *  wv_ext3;
    const void *  wk_ctx_ext3;
    const void *  wv_ctx_ext3;
    const void *  wo_ext3;
    const void *  ffn_gate_ext3;
    const void *  ffn_up_ext3;
    const void *  ffn_down_ext3;
};

struct ggml_ink2_model_desc {
    const void * model_key;
    int32_t device_id;
    int32_t b_max;
    int32_t max_seq;
    int32_t n_phys_layers;
    int32_t loop_beg;
    int32_t loop_end;
    int32_t n_loop_passes;
    int32_t block_size;
    int32_t n_experts;
    int32_t top_k;
    int32_t vocab_unpad;
    int32_t vocab_pad;
    int32_t vocab_full;
    int32_t eos_id;
    int32_t n_mtp_layers;
    int32_t tap_visits[4];
    int32_t unpack_full_vram_moe;

    const void *  tok_embd_ext2;
    const float * tok_norm_f32;
    const float * output_norm_f32;
    const void *  output_ext3;
    const float * output_res_query_f32;
    const float * loop_msg_queries_f32;
    const float * mlp_res_queries_f32;
    const float * attn_res_queries_f32[24];

    const void *  aux_blob_host;
    size_t        aux_blob_size;

    struct ggml_ink2_layer_ptrs layers[24];

    const float * mtp_fuse_alpha_f32;
    const void *  mtp_mask_embed_bf16;
    const float * mtp_norm_f32;
    const float * mtp_mh_dflare_f32;
    const float * mtp_rnn_w_f32;
    const float * mtp_rnn_b_f32;
    const void *  mtp_fold_ext3;
    const float * mtp_fold_b_f32;
    const void *  mtp_mh_w1_q8_0;
    int32_t       mtp_mh_w1_rows;
    struct ggml_ink2_mtp_layer_ptrs mtp_layers[3];
};

struct ggml_ink2_mtp_stats {
    int64_t spec_steps;
    int64_t drafted_tokens;
    int64_t accepted_tokens;
    double  mean_accepted_len;
    int64_t s4_steps;
    int64_t s8_steps;
};

struct ggml_cuda_inkling2_api_t {
    size_t struct_size;

    void * (*ensure_handle)(const struct ggml_ink2_model_desc * desc);
    void   (*destroy_by_model)(const void * model_key);
    int32_t (*reset_cache)(void * handle);
    int32_t (*seq_rm)(void * handle, int32_t seq_id, int32_t p0, int32_t p1);
    int32_t (*set_mtp_policy)(void * handle, int32_t policy);

    int32_t (*prefill)(
        void * handle,
        const int32_t * ids_b_s_host,
        int32_t b_act,
        int32_t s_tot,
        int32_t enable_mtp_commit,
        float temp,
        float top_p,
        uint64_t seed,
        int32_t * out_next_toks_host,
        const void * vis_bf16_or_f32,
        int32_t vis_is_f32,
        int32_t n_vis,
        int32_t vis_is_device,
        int32_t image_token_id);

    int32_t (*decode_m1)(
        void * handle,
        const int32_t * first_toks_host,
        int32_t b_act,
        int32_t max_new_tokens,
        int32_t eos_id,
        float temp,
        float top_p,
        uint64_t seed,
        int32_t * out_ids_b_t_host,
        int32_t * out_steps_done);

    int32_t (*decode_mtp8)(
        void * handle,
        int32_t first_tok,
        int32_t max_new_tokens,
        int32_t eos_id,
        float temp,
        float top_p,
        uint64_t seed,
        int32_t * out_tokens_host,
        int32_t * out_len,
        struct ggml_ink2_mtp_stats * out_stats);

    int32_t (*prefill_slot)(
        void * handle,
        int32_t slot_b,
        const int32_t * ids_s_host,
        int32_t s_tot,
        float temp,
        float top_p,
        uint64_t seed,
        int32_t * out_tok0);

    int32_t (*step_continuous_m1)(
        void * handle,
        const int32_t * cur_toks_host,
        int32_t b_span,
        int32_t upload_cur_toks,
        float temp,
        float top_p,
        uint64_t seed,
        int32_t * out_toks_host);

    int32_t (*mtp_draft_block)(
        void * handle,
        int32_t pos0,
        int32_t anchor_tok,
        int32_t k_draft,
        int32_t * out_draft_toks_host);

    void (*notify_ubatch)(
        void * handle,
        int32_t b_act,
        int32_t s_tokens,
        const int32_t * host_seq_ids,
        const int32_t * host_pos0_per_seq);

    int32_t (*forward_logits_f32)(
        void * handle,
        const int32_t * ids_s_host,
        int32_t s_tot,
        float * out_logits_s_v_host);

    const char * (*last_error)(void * handle);
};
#endif /* GGML_CUDA_INK2_OP_MAGIC */

const struct ggml_cuda_inkling2_api_t * ink2_core_get_api(void);

int32_t ink2_core_op_block_attn_res(
    const float * d_cur_h,
    const float * d_queries,
    const float * const d_blks[5],
    float * d_out,
    int32_t n_blk,
    int32_t n_q,
    int32_t round_cur_bf16,
    int32_t m_tokens,
    cudaStream_t stream);

int32_t ink2_core_op_loop_msg_gate(
    const float * d_src,
    const float * d_msg,
    const void *  d_ctrl_down_bf16,
    const float * d_ctrl_forget_f32,
    const float * d_ctrl_gain_f32,
    float * d_out,
    int32_t m_tokens,
    cudaStream_t stream);

int32_t ink2_core_op_sconv_static(
    const float * d_x_in,
    const float * d_state_3_c_b,
    const float * d_w4_c,
    float * d_out,
    int32_t channels,
    int32_t s_tokens,
    int32_t b_seqs,
    cudaStream_t stream);

int32_t ink2_core_op_sconv_dynamic(
    const float * d_x_in,
    const float * d_state_3_c_b,
    const float * d_static_w_c_4,
    const float * d_dw_32_s_b,
    float * d_out,
    int32_t s_tokens,
    int32_t b_seqs,
    cudaStream_t stream);

int32_t ink2_core_op_prep_kv(
    const float * d_in,
    float * d_out,
    int32_t is_v,
    int32_t m_tokens,
    cudaStream_t stream);

int32_t ink2_core_op_attn_rel_sink(
    const float * d_q_raw,
    const void *  d_k_cache,
    const void *  d_v_cache,
    int32_t       kv_ggml_type,
    int32_t       n_kv,
    int32_t       n_stream,
    size_t        nb1_k,
    size_t        nb2_k,
    size_t        nb3_k,
    size_t        nb1_v,
    size_t        nb2_v,
    size_t        nb3_v,
    const float * d_r_raw,
    const float * d_q_clip,
    const float * d_r_clip,
    const float * d_rel_proj_t,
    const float * d_sinks,
    const float * d_gate_raw,
    const int32_t * d_pos_pack,
    float * d_out,
    int32_t is_local,
    int32_t m_tokens,
    cudaStream_t stream);

int32_t ink2_core_op_softcap_mask_head(
    const float * d_logits_in,
    float * d_logits_out,
    int32_t in_stride_vocab,
    int32_t out_vocab_full,
    int32_t vocab_unpad,
    float   softcap,
    int32_t n_rows,
    cudaStream_t stream);

int32_t ink2_core_op_mtp_stash_tap(
    void * handle,
    const float * d_src_f32,
    float * d_dst_f32,
    int32_t tap_slot,
    int32_t m_tokens,
    cudaStream_t stream);

int32_t ink2_core_op_fused_trunk(
    void * handle,
    const void * d_src0_tok_or_embd,
    int32_t has_tokens_src0,
    const int32_t * d_pos,
    const int32_t * d_seq_ids,
    int32_t single_seq_id,
    const int32_t * d_out_ids,
    int32_t b_act,
    int32_t s_tokens,
    int32_t n_outputs,
    float * d_logits_out,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

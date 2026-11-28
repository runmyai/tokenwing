#include "models.h"
#include "../llama-context.h"
#include "../llama-memory-hybrid-iswa.h"
#include "../llama-moe-stream.h"
#include "ggml-inkling2-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace inkling2_model_detail {

using get_ink2_cuda_api_fn = const struct ggml_cuda_inkling2_api_t * (*)(void);

static inline float bf16_round_f32(float x) {
    ggml_bf16_t b = ggml_fp32_to_bf16(x);
    return ggml_bf16_to_fp32(b);
}

static const struct ggml_cuda_inkling2_api_t * find_cuda_api(const llama_model & model) {
    for (const auto & ld : model.devices) {
        if (!ld.dev || ggml_backend_dev_type(ld.dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ld.dev);
        if (!reg) {
            continue;
        }
        auto fn = reinterpret_cast<get_ink2_cuda_api_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_cuda_inkling2_api"));
        if (fn) {
            const auto * api = fn();
            if (api && api->struct_size == sizeof(ggml_cuda_inkling2_api_t)) {
                return api;
            }
        }
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg) {
            auto fn = reinterpret_cast<get_ink2_cuda_api_fn>(
                ggml_backend_reg_get_proc_address(reg, "ggml_cuda_inkling2_api"));
            if (fn) {
                const auto * api = fn();
                if (api && api->struct_size == sizeof(ggml_cuda_inkling2_api_t)) {
                    return api;
                }
            }
        }
    }
    return nullptr;
}

static bool tensor_on_gpu(const ggml_tensor * t, int32_t * out_cuda_dev_id = nullptr) {
    if (!t || !t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
    if (!buft) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    if (out_cuda_dev_id) {
        const char * nm = ggml_backend_dev_name(dev);
        if (nm && strncmp(nm, "CUDA", 4) == 0 && nm[4] >= '0' && nm[4] <= '9') {
            *out_cuda_dev_id = atoi(nm + 4);
        } else {
            *out_cuda_dev_id = 0;
        }
    }
    return true;
}

static int visit_to_phys_layer(int v, int loop_beg, int loop_end, int n_passes) {
    if (v < loop_beg) {
        return v;
    }
    const int loop_len = loop_end - loop_beg;
    const int loop_tot = n_passes * loop_len;
    if (loop_len > 0 && v < loop_beg + loop_tot) {
        return loop_beg + ((v - loop_beg) % loop_len);
    }
    return loop_end + (v - (loop_beg + loop_tot));
}

static int phys_layer_to_first_visit(int phys_li, int loop_beg, int loop_end, int n_passes) {
    if (phys_li < loop_end) {
        return phys_li;
    }
    const int loop_len = loop_end - loop_beg;
    return loop_beg + n_passes * loop_len + (phys_li - loop_end);
}

static bool model_all_on_single_gpu(const llama_model & model, int32_t * out_dev_id) {
    int32_t dev0 = -1;
    auto chk = [&](const ggml_tensor * t) -> bool {
        if (!t) return true;
        int32_t d = -1;
        if (!tensor_on_gpu(t, &d)) return false;
        if (dev0 < 0) dev0 = d;
        return d == dev0;
    };
    if (!model.tok_embd || !model.tok_embd->data ||
        !chk(model.tok_norm) || !chk(model.output_norm) || !chk(model.output)) {
        return false;
    }
    const int n_phys   = (int) model.hparams.ink2_phys_block_count;
    const int loop_beg = (int) model.hparams.ink2_loop_start;
    const int loop_end = (int) model.hparams.ink2_loop_end;
    const int n_passes = (int) model.hparams.ink2_loop_passes;
    for (int li = 0; li < n_phys; ++li) {
        const int v0 = phys_layer_to_first_visit(li, loop_beg, loop_end, n_passes);
        const auto & L = model.layers[v0];
        if (!chk(L.wq) || !chk(L.wk) || !chk(L.wv) || !chk(L.wqkv_gate) || !chk(L.wo) ||
            !chk(L.ffn_gate_inp) || !chk(L.ffn_gate_exps) || !chk(L.ffn_up_exps) || !chk(L.ffn_down_exps)) {
            return false;
        }
    }
    if (out_dev_id) *out_dev_id = (dev0 >= 0) ? dev0 : 0;
    return dev0 >= 0;
}

static bool model_mtp_on_gpu(const llama_model & model, int32_t * out_dev_id) {
    if (model.hparams.ink2_mtp_layer_count == 0 || model.ink2_mtp_layers.empty()) {
        return false;
    }
    int32_t dev0 = -1;
    if (!tensor_on_gpu(model.output, &dev0)) {
        for (const auto & ld : model.devices) {
            if (ld.dev && ggml_backend_dev_type(ld.dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                const char * nm = ggml_backend_dev_name(ld.dev);
                dev0 = (nm && strncmp(nm, "CUDA", 4) == 0 && nm[4] >= '0' && nm[4] <= '9') ? atoi(nm + 4) : 0;
                break;
            }
        }
    }
    if (dev0 < 0) {
        return false;
    }
    auto has_data = [](const ggml_tensor * t) -> bool {
        return t != nullptr && t->data != nullptr;
    };
    if (!has_data(model.tok_embd) || !has_data(model.ink2_mtp_mh_w1) ||
        !has_data(model.tok_norm) || !has_data(model.output_norm) || !has_data(model.output) ||
        !has_data(model.ink2_mtp_fuse_alpha) || !has_data(model.ink2_mtp_mask_embed) || !has_data(model.ink2_mtp_norm) ||
        !has_data(model.ink2_mtp_mh_dflare) || !has_data(model.ink2_mtp_rnn_w) || !has_data(model.ink2_mtp_rnn_b) ||
        !has_data(model.ink2_mtp_fold_w) || !has_data(model.ink2_mtp_fold_b)) {
        return false;
    }
    for (const auto & ml : model.ink2_mtp_layers) {
        if (!has_data(ml.dflare_norm) || !has_data(ml.in_norm) || !has_data(ml.post_norm) ||
            !has_data(ml.q_norm) || !has_data(ml.k_norm) || !has_data(ml.k_norm_ctx) ||
            !has_data(ml.wq) || !has_data(ml.wk) || !has_data(ml.wv) || !has_data(ml.wk_ctx) || !has_data(ml.wv_ctx) ||
            !has_data(ml.wo) || !has_data(ml.ffn_gate) || !has_data(ml.ffn_up) || !has_data(ml.ffn_down)) {
            return false;
        }
    }
    if (out_dev_id) *out_dev_id = dev0;
    return true;
}

static void * ensure_cuda_handle(
        const llama_model & model,
        int32_t b_max,
        int32_t max_seq,
        bool want_full_vram_moe) {
    const auto * api = find_cuda_api(model);
    if (!api || model.ink2_quant_aux_blob.empty()) {
        return nullptr;
    }
    int32_t dev_id = 0;
    if (want_full_vram_moe) {
        if (!model_all_on_single_gpu(model, &dev_id)) {
            return nullptr;
        }
    } else {
        if (!model_mtp_on_gpu(model, &dev_id)) {
            return nullptr;
        }
    }

    const auto & hp = model.hparams;
    ggml_ink2_model_desc d {};
    d.model_key            = &model;
    d.device_id            = dev_id;
    d.b_max                = std::max<int32_t>(1, b_max);
    d.max_seq              = std::max<int32_t>(1040, max_seq);
    d.n_phys_layers        = (int32_t) hp.ink2_phys_block_count;
    d.loop_beg             = (int32_t) hp.ink2_loop_start;
    d.loop_end             = (int32_t) hp.ink2_loop_end;
    d.n_loop_passes        = (int32_t) hp.ink2_loop_passes;
    d.block_size           = (int32_t) hp.attn_res_block_size;
    d.n_experts            = (int32_t) hp.n_expert;
    d.top_k                = (int32_t) hp.n_expert_used();
    d.vocab_unpad          = (int32_t) hp.ink2_unpadded_vocab_size;
    d.vocab_pad            = (int32_t) hp.ink2_padded_head_vocab_size;
    d.vocab_full           = (int32_t) model.vocab.n_tokens();
    d.eos_id               = (int32_t) model.vocab.token_eos();
    d.n_mtp_layers         = model_mtp_on_gpu(model, nullptr) ? (int32_t) hp.ink2_mtp_layer_count : 0;
    for (int i = 0; i < 4; ++i) {
        d.tap_visits[i] = hp.ink2_mtp_tap_visits[i];
    }
    d.unpack_full_vram_moe = want_full_vram_moe ? 1 : 0;

    d.tok_embd_ext2        = model.tok_embd ? model.tok_embd->data : nullptr;
    d.tok_norm_f32         = model.tok_norm ? static_cast<const float *>(model.tok_norm->data) : nullptr;
    d.output_norm_f32      = model.output_norm ? static_cast<const float *>(model.output_norm->data) : nullptr;
    d.output_ext3          = model.output ? model.output->data : nullptr;
    d.output_res_query_f32 = model.ink2_output_res_query ? static_cast<const float *>(model.ink2_output_res_query->data) : nullptr;
    d.loop_msg_queries_f32 = model.ink2_loop_msg_queries ? static_cast<const float *>(model.ink2_loop_msg_queries->data) : nullptr;
    d.mlp_res_queries_f32  = model.ink2_mlp_res_queries  ? static_cast<const float *>(model.ink2_mlp_res_queries->data)  : nullptr;

    d.aux_blob_host        = model.ink2_quant_aux_blob.data();
    d.aux_blob_size        = model.ink2_quant_aux_blob.size();

    for (int li = 0; li < d.n_phys_layers && li < 24; ++li) {
        const int v0      = phys_layer_to_first_visit(li, d.loop_beg, d.loop_end, d.n_loop_passes);
        const auto & L    = model.layers[v0];
        auto & dst_l      = d.layers[li];
        const bool is_loc = hp.is_swa(v0);
        const bool is_lp  = (li >= d.loop_beg && li < d.loop_end);

        d.attn_res_queries_f32[li] = L.ink2_attn_res_query ? static_cast<const float *>(L.ink2_attn_res_query->data) : nullptr;
        dst_l.is_local = is_loc ? 1 : 0;
        dst_l.is_loop  = is_lp  ? 1 : 0;
        if (want_full_vram_moe) {
            dst_l.wq_ext3        = L.wq ? L.wq->data : nullptr;
            dst_l.wk_ext3        = L.wk ? L.wk->data : nullptr;
            dst_l.wv_ext3        = L.wv ? L.wv->data : nullptr;
            dst_l.wg_ext3        = L.wqkv_gate ? L.wqkv_gate->data : nullptr;
            dst_l.wo_ext3        = L.wo ? L.wo->data : nullptr;
            dst_l.wr_bf16        = L.ink2_attn_rel ? L.ink2_attn_rel->data : nullptr;
            dst_l.w_sink_bf16    = L.ink2_attn_sink ? L.ink2_attn_sink->data : nullptr;
            dst_l.q_clip_f32     = L.ink2_attn_q_clip ? static_cast<const float *>(L.ink2_attn_q_clip->data) : nullptr;
            dst_l.r_clip_f32     = L.ink2_attn_r_clip ? static_cast<const float *>(L.ink2_attn_r_clip->data) : nullptr;
            dst_l.rel_proj_t_f32 = L.ink2_attn_rel_proj ? static_cast<const float *>(L.ink2_attn_rel_proj->data) : nullptr;
            dst_l.attn_sconv_f32 = L.ink2_attn_sconv ? static_cast<const float *>(L.ink2_attn_sconv->data) : nullptr;
            dst_l.ffn_sconv_f32  = L.ink2_ffn_sconv ? static_cast<const float *>(L.ink2_ffn_sconv->data) : nullptr;
            dst_l.ffn_norm_f32   = L.ffn_norm ? static_cast<const float *>(L.ffn_norm->data) : nullptr;
            dst_l.router_w_f32   = L.ffn_gate_inp ? static_cast<const float *>(L.ffn_gate_inp->data) : nullptr;
            dst_l.router_b_f32   = L.ffn_exp_probs_b ? static_cast<const float *>(L.ffn_exp_probs_b->data) : nullptr;
            dst_l.attn_norm_f32  = L.attn_norm ? static_cast<const float *>(L.attn_norm->data) : nullptr;
            dst_l.k_sconv_f32    = L.ink2_attn_k_sconv ? static_cast<const float *>(L.ink2_attn_k_sconv->data) : nullptr;
            dst_l.v_sconv_f32    = L.ink2_attn_v_sconv ? static_cast<const float *>(L.ink2_attn_v_sconv->data) : nullptr;
            dst_l.q_in_norm_f32  = L.ink2_attn_q_in_norm ? static_cast<const float *>(L.ink2_attn_q_in_norm->data) : nullptr;
            dst_l.k_in_norm_f32  = L.ink2_attn_k_in_norm ? static_cast<const float *>(L.ink2_attn_k_in_norm->data) : nullptr;
            dst_l.v_in_norm_f32  = L.ink2_attn_v_in_norm ? static_cast<const float *>(L.ink2_attn_v_in_norm->data) : nullptr;
            dst_l.k_static_w_f32 = L.ink2_attn_k_static_w ? static_cast<const float *>(L.ink2_attn_k_static_w->data) : nullptr;
            dst_l.k_wproj_bf16   = L.ink2_attn_k_wproj ? L.ink2_attn_k_wproj->data : nullptr;
            dst_l.v_static_w_f32 = L.ink2_attn_v_static_w ? static_cast<const float *>(L.ink2_attn_v_static_w->data) : nullptr;
            dst_l.v_wproj_bf16   = L.ink2_attn_v_wproj ? L.ink2_attn_v_wproj->data : nullptr;
            for (int c = 0; c < 4; ++c) {
                dst_l.ctrl_down_bf16[c]  = L.ink2_ctrl_down[c] ? L.ink2_ctrl_down[c]->data : nullptr;
                dst_l.ctrl_forget_f32[c] = L.ink2_ctrl_forget[c] ? static_cast<const float *>(L.ink2_ctrl_forget[c]->data) : nullptr;
                dst_l.ctrl_gain_f32[c]   = L.ink2_ctrl_gain[c] ? static_cast<const float *>(L.ink2_ctrl_gain[c]->data) : nullptr;
            }
            dst_l.gate_exps_ext0 = L.ffn_gate_exps ? L.ffn_gate_exps->data : nullptr;
            dst_l.up_exps_ext0   = L.ffn_up_exps ? L.ffn_up_exps->data : nullptr;
            dst_l.down_exps_ext0 = L.ffn_down_exps ? L.ffn_down_exps->data : nullptr;
        }
    }

    if (d.n_mtp_layers > 0) {
        d.mtp_fuse_alpha_f32 = static_cast<const float *>(model.ink2_mtp_fuse_alpha->data);
        d.mtp_mask_embed_bf16 = model.ink2_mtp_mask_embed->data;
        d.mtp_norm_f32       = static_cast<const float *>(model.ink2_mtp_norm->data);
        d.mtp_mh_dflare_f32  = static_cast<const float *>(model.ink2_mtp_mh_dflare->data);
        d.mtp_rnn_w_f32      = static_cast<const float *>(model.ink2_mtp_rnn_w->data);
        d.mtp_rnn_b_f32      = static_cast<const float *>(model.ink2_mtp_rnn_b->data);
        d.mtp_fold_ext3      = model.ink2_mtp_fold_w->data;
        d.mtp_fold_b_f32     = static_cast<const float *>(model.ink2_mtp_fold_b->data);
        d.mtp_mh_w1_q8_0     = model.ink2_mtp_mh_w1->data;
        d.mtp_mh_w1_rows     = (int32_t) model.ink2_mtp_mh_w1->ne[1];
        for (int mi = 0; mi < d.n_mtp_layers && mi < 3; ++mi) {
            const auto & sm = model.ink2_mtp_layers[mi];
            auto & dm       = d.mtp_layers[mi];
            dm.is_local        = (mi != 1) ? 1 : 0;
            dm.dflare_norm_f32 = static_cast<const float *>(sm.dflare_norm->data);
            dm.in_norm_f32     = static_cast<const float *>(sm.in_norm->data);
            dm.post_norm_f32   = static_cast<const float *>(sm.post_norm->data);
            dm.q_norm_f32      = static_cast<const float *>(sm.q_norm->data);
            dm.k_norm_f32      = static_cast<const float *>(sm.k_norm->data);
            dm.k_norm_ctx_f32  = static_cast<const float *>(sm.k_norm_ctx->data);
            dm.wq_ext3         = sm.wq->data;
            dm.wk_ext3         = sm.wk->data;
            dm.wv_ext3         = sm.wv->data;
            dm.wk_ctx_ext3     = sm.wk_ctx->data;
            dm.wv_ctx_ext3     = sm.wv_ctx->data;
            dm.wo_ext3         = sm.wo->data;
            dm.ffn_gate_ext3   = sm.ffn_gate->data;
            dm.ffn_up_ext3     = sm.ffn_up->data;
            dm.ffn_down_ext3   = sm.ffn_down->data;
        }
    }

    return api->ensure_handle(&d);
}

// ============================================================================
// Graph Inputs (`FUSED_TRUNK` ubatch notifier + `ATTN_REL_SINK` KV pos_pack)
// ============================================================================

class llm_graph_input_ink2_ftrunk : public llm_graph_input_i {
public:
    llm_graph_input_ink2_ftrunk(
            const ggml_cuda_inkling2_api_t * api,
            void * cuda_handle,
            ggml_tensor * t_seq_ids,
            const llama_memory_context_i * mctx_init,
            bool has_tokens_src0) :
        api(api), cuda_handle(cuda_handle), t_seq_ids(t_seq_ids), cur_mctx(mctx_init), has_tokens_src0(has_tokens_src0) {}

    void set_input(const llama_ubatch * ubatch) override {
        if (!ubatch || ubatch->n_tokens == 0) return;
        if (cur_mctx) {
            const auto * hyb = static_cast<const llama_memory_hybrid_iswa_context *>(cur_mctx);
            if (hyb && hyb->get_recr()) {
                const uint32_t n_rs = hyb->get_recr()->get_n_rs();
                for (uint32_t i = 0; i < n_rs; ++i) {
                    (void) hyb->get_recr()->s_copy((int) i);
                }
            }
        }
        const int32_t b_act = (int32_t) std::max(1u, ubatch->n_seqs);
        const int32_t s_tok = (int32_t) std::max(1u, ubatch->n_seq_tokens);
        host_seq_ids.resize(b_act);
        host_pos0.resize(b_act);
        for (int32_t b = 0; b < b_act; ++b) {
            const uint32_t row0 = (uint32_t) b * (uint32_t) s_tok;
            host_seq_ids[b] = (ubatch->seq_id && row0 < ubatch->n_tokens) ? ubatch->seq_id[row0][0] : b;
            host_pos0[b]    = (ubatch->pos    && row0 < ubatch->n_tokens) ? ubatch->pos[row0]       : 0;
        }
        if (api && api->notify_ubatch && cuda_handle) {
            api->notify_ubatch(cuda_handle, b_act, s_tok, host_seq_ids.data(), host_pos0.data());
        }
        if (t_seq_ids) {
            ggml_backend_tensor_set(t_seq_ids, host_seq_ids.data(), 0, b_act * sizeof(int32_t));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        const bool want_tok = (params.ubatch.token != nullptr && !params.ubatch.embd);
        if (want_tok != this->has_tokens_src0) {
            return false;
        }
        this->cur_mctx = params.mctx;
        return true;
    }

private:
    const ggml_cuda_inkling2_api_t * api      = nullptr;
    void *                           cuda_handle = nullptr;
    ggml_tensor *                    t_seq_ids   = nullptr;
    const llama_memory_context_i *   cur_mctx    = nullptr;
    bool                             has_tokens_src0 = true;
    std::vector<int32_t> host_seq_ids;
    std::vector<int32_t> host_pos0;
};

class llm_graph_input_ink2_kpos : public llm_graph_input_i {
public:
    llm_graph_input_ink2_kpos(
            const llama_kv_cache_iswa_context * mctx_iswa,
            const ggml_cuda_inkling2_api_t * api,
            void * mtp_handle,
            ggml_tensor * t_pos_base,
            ggml_tensor * t_pos_swa,
            int32_t n_kv_base,
            int32_t ns_base,
            int32_t n_kv_swa,
            int32_t ns_swa) :
        mctx_iswa(mctx_iswa), api(api), mtp_handle(mtp_handle),
        t_pos_base(t_pos_base), t_pos_swa(t_pos_swa),
        n_kv_base(n_kv_base), ns_base(ns_base),
        n_kv_swa(n_kv_swa), ns_swa(ns_swa) {}

    static void fill_pack(
            const llama_kv_cache_context * kctx,
            const llama_ubatch * ubatch,
            int32_t n_kv,
            int32_t ns,
            std::vector<int32_t> & buf,
            ggml_tensor * dst) {
        if (!kctx || !dst || !ubatch) return;
        const int32_t M = (int32_t) ubatch->n_tokens;
        buf.assign((size_t)(2 * M + n_kv * ns), -1);
        const auto * kv     = kctx->get_kv();
        const uint32_t s0   = kctx->get_s0();
        const uint32_t n_st = kv->get_n_stream();
        for (int32_t m = 0; m < M; ++m) {
            buf[m] = ubatch->pos[m];
            const llama_seq_id sid = ubatch->seq_id[m][0];
            buf[M + m] = (n_st > 1) ? (int32_t)(kv->get_stream(sid) - s0) : 0;
        }
        const auto & v_cells = kctx->get_active_cells();
        const llama_seq_id uni_sid = ubatch->seq_id[0][0];
        for (int32_t st = 0; st < ns; ++st) {
            const uint32_t g_st = s0 + (uint32_t) st;
            if (g_st >= v_cells.size()) continue;
            const auto & cells = v_cells[g_st];
            int32_t * out_k = buf.data() + 2 * M + (size_t) st * n_kv;
            const uint32_t lim = std::min<uint32_t>((uint32_t) n_kv, cells.size());
            for (uint32_t j = 0; j < lim; ++j) {
                if (!cells.is_empty(j) && (n_st > 1 || ubatch->n_seqs_unq <= 1 || cells.seq_has(j, uni_sid))) {
                    out_k[j] = cells.pos_get(j);
                }
            }
        }
        ggml_backend_tensor_set(dst, buf.data(), 0, buf.size() * sizeof(int32_t));
    }

    void set_input(const llama_ubatch * ubatch) override {
        if (!ubatch || ubatch->n_tokens == 0) return;
        if (api && api->notify_ubatch && mtp_handle) {
            const int32_t b_act = (int32_t) std::max(1u, ubatch->n_seqs);
            const int32_t s_tok = (int32_t) std::max(1u, ubatch->n_seq_tokens);
            int32_t sid0 = ubatch->seq_id ? ubatch->seq_id[0][0] : 0;
            int32_t pos0 = ubatch->pos    ? ubatch->pos[0]       : 0;
            api->notify_ubatch(mtp_handle, b_act, s_tok, &sid0, &pos0);
        }
        if (mctx_iswa) {
            fill_pack(mctx_iswa->get_base(), ubatch, n_kv_base, ns_base, buf_base, t_pos_base);
            fill_pack(mctx_iswa->get_swa(),  ubatch, n_kv_swa,  ns_swa,  buf_swa,  t_pos_swa);
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx_hyb = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);
        if (!mctx_hyb || !mctx_hyb->get_attn()) return false;
        this->mctx_iswa = mctx_hyb->get_attn();
        return (int32_t) mctx_iswa->get_base()->get_n_kv() == n_kv_base &&
               (int32_t) mctx_iswa->get_base()->get_ns()   == ns_base   &&
               (int32_t) mctx_iswa->get_swa()->get_n_kv()  == n_kv_swa  &&
               (int32_t) mctx_iswa->get_swa()->get_ns()    == ns_swa;
    }

private:
    const llama_kv_cache_iswa_context * mctx_iswa = nullptr;
    const ggml_cuda_inkling2_api_t *    api       = nullptr;
    void *                              mtp_handle = nullptr;
    ggml_tensor * t_pos_base = nullptr;
    ggml_tensor * t_pos_swa  = nullptr;
    int32_t n_kv_base = 0, ns_base = 1;
    int32_t n_kv_swa  = 0, ns_swa  = 1;
    std::vector<int32_t> buf_base;
    std::vector<int32_t> buf_swa;
};

// ============================================================================
// CPU Reference Callbacks (`-ngl 0` Bit-Consistent Path)
// ============================================================================

static void cpu_op_noop_abort(ggml_tensor *, int, int, void *) {
    GGML_ABORT("GGML_CUDA_INK2_OP_FUSED_TRUNK requires CUDA backend");
}

static void cpu_op_block_attn_res(ggml_tensor * dst, int ith, int nth, void *) {
    const int32_t n_blk          = dst->op_params[10];
    const int32_t Q              = dst->op_params[11];
    const int32_t round_cur_bf16 = dst->op_params[12];
    const int32_t M              = (int32_t) dst->src[0]->ne[1];
    const int32_t D              = 2304;
    const int32_t n_src          = n_blk + 1;

    const float * cur_h   = static_cast<const float *>(dst->src[0]->data);
    const float * queries = static_cast<const float *>(dst->src[1]->data);
    float *       out     = static_cast<float *>(dst->data);

    std::vector<float> normed((size_t) n_src * D);
    for (int32_t m = ith; m < M; m += nth) {
        for (int32_t s = 0; s < n_src; ++s) {
            const float * sp = (s < n_blk)
                ? (static_cast<const float *>(dst->src[2 + s]->data) + (size_t) m * D)
                : (cur_h + (size_t) m * D);
            const bool do_bf16 = (s < n_blk) || (round_cur_bf16 != 0);
            float * vp = normed.data() + (size_t) s * D;
            float sq = 0.0f;
            for (int32_t d = 0; d < D; ++d) {
                float v = do_bf16 ? bf16_round_f32(sp[d]) : sp[d];
                vp[d] = v;
                sq += v * v;
            }
            const float inv = 1.0f / sqrtf(sq * (1.0f / 2304.0f) + 1e-6f);
            for (int32_t d = 0; d < D; ++d) vp[d] *= inv;
        }

        for (int32_t q = 0; q < Q; ++q) {
            const float * qp = queries + (size_t) q * D;
            float lg[6];
            float mx = -1e30f;
            for (int32_t s = 0; s < n_src; ++s) {
                const float * vp = normed.data() + (size_t) s * D;
                float dot = 0.0f;
                for (int32_t d = 0; d < D; ++d) dot += (qp[d] * (1.0f / 48.0f)) * vp[d];
                lg[s] = dot;
                if (dot > mx) mx = dot;
            }
            float sm = 0.0f;
            for (int32_t s = 0; s < n_src; ++s) {
                lg[s] = expf(lg[s] - mx);
                sm += lg[s];
            }
            const float inv_s = 1.0f / fmaxf(sm, 1e-20f);
            float * dp = out + ((size_t) q * M + m) * D;
            for (int32_t d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int32_t s = 0; s < n_src; ++s) {
                    acc += (lg[s] * inv_s) * normed[(size_t) s * D + d];
                }
                dp[d] = acc;
            }
        }
    }
}

static void cpu_op_loop_msg_gate(ggml_tensor * dst, int ith, int nth, void *) {
    const int32_t M = (int32_t) dst->ne[1];
    const int32_t D = 2304;
    const float *       src_in = static_cast<const float *>(dst->src[0]->data);
    const float *       msg_in = static_cast<const float *>(dst->src[1]->data);
    const ggml_bf16_t * c_down = static_cast<const ggml_bf16_t *>(dst->src[2]->data);
    const float *       c_forg = static_cast<const float *>(dst->src[3]->data);
    const float *       c_gain = static_cast<const float *>(dst->src[4]->data);
    float *             out    = static_cast<float *>(dst->data);

    float cat[4608];
    float h64[64];
    for (int32_t m = ith; m < M; m += nth) {
        const float * sp = src_in + (size_t) m * D;
        const float * mp = msg_in + (size_t) m * D;
        float d_sq = 0.0f, m_sq = 0.0f;
        for (int32_t i = 0; i < D; ++i) {
            d_sq += sp[i] * sp[i];
            m_sq += mp[i] * mp[i];
        }
        const float inv_d = 1.0f / sqrtf(d_sq * (1.0f / 2304.0f) + 1e-6f);
        const float inv_m = 1.0f / sqrtf(m_sq * (1.0f / 2304.0f) + 1e-6f);
        for (int32_t i = 0; i < D; ++i) {
            cat[i]     = bf16_round_f32(sp[i] * inv_d);
            cat[D + i] = bf16_round_f32(mp[i] * inv_m);
        }
        for (int32_t r = 0; r < 64; ++r) {
            const ggml_bf16_t * wr = c_down + (size_t) r * 4608;
            float acc = 0.0f;
            for (int32_t k = 0; k < 4608; ++k) acc += ggml_bf16_to_fp32(wr[k]) * cat[k];
            h64[r] = acc;
        }
        float * dp = out + (size_t) m * D;
        for (int32_t ch = 0; ch < D; ++ch) {
            const float * fw = c_forg + (size_t) ch * 64;
            const float * gw = c_gain + (size_t) ch * 64;
            float f_acc = 0.0f, g_acc = 0.0f;
            for (int32_t k = 0; k < 64; ++k) {
                f_acc += bf16_round_f32(fw[k]) * h64[k];
                g_acc += bf16_round_f32(gw[k]) * h64[k];
            }
            dp[ch] = (1.0f - tanhf(f_acc)) * sp[ch] + tanhf(g_acc) * mp[ch];
        }
    }
}

static void cpu_op_sconv_static(ggml_tensor * dst, int ith, int nth, void *) {
    const int32_t C = (int32_t) dst->ne[0];
    const int32_t S = (int32_t) dst->ne[1];
    const int32_t B = (int32_t) dst->ne[2];
    const float * x_in = static_cast<const float *>(dst->src[0]->data);
    const float * st   = static_cast<const float *>(dst->src[1]->data);
    const float * w4   = static_cast<const float *>(dst->src[2]->data);
    float *       out  = static_cast<float *>(dst->data);
    constexpr float kInvSqrt5 = 0.44721359549995793928f;

    const int32_t tot_bc = B * C;
    for (int32_t bc = ith; bc < tot_bc; bc += nth) {
        const int32_t b = bc / C;
        const int32_t c = bc - b * C;
        const float * wc = w4 + (size_t) c * 4;
        const float * sb = st + ((size_t) b * C + c) * 3;
        float h0 = sb[0], h1 = sb[1], h2 = sb[2];
        for (int32_t s = 0; s < S; ++s) {
            const size_t idx = ((size_t) b * S + s) * C + c;
            const float x_cur = bf16_round_f32(x_in[idx]);
            const float conv  = wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * x_cur;
            out[idx] = bf16_round_f32((conv + x_cur) * kInvSqrt5);
            h0 = h1; h1 = h2; h2 = x_cur;
        }
    }
}

static void cpu_op_sconv_dynamic(ggml_tensor * dst, int ith, int nth, void *) {
    const int32_t C = 1024;
    const int32_t S = (int32_t) dst->ne[1];
    const int32_t B = (int32_t) dst->ne[2];
    const float * x_in     = static_cast<const float *>(dst->src[0]->data);
    const float * st       = static_cast<const float *>(dst->src[1]->data);
    const float * static_w = static_cast<const float *>(dst->src[2]->data);
    const float * dw_all   = static_cast<const float *>(dst->src[3]->data);
    float *       out      = static_cast<float *>(dst->data);

    const int32_t tot_bc = B * C;
    for (int32_t bc = ith; bc < tot_bc; bc += nth) {
        const int32_t b = bc / C;
        const int32_t c = bc - b * C;
        const int32_t h = c >> 7;
        const float sw0 = static_w[0 * C + c], sw1 = static_w[1 * C + c];
        const float sw2 = static_w[2 * C + c], sw3 = static_w[3 * C + c];
        const float * sb = st + ((size_t) b * C + c) * 3;
        float lag3 = sb[0], lag2 = sb[1], lag1 = sb[2];
        for (int32_t s = 0; s < S; ++s) {
            const size_t idx  = ((size_t) b * S + s) * C + c;
            const float x_cur = bf16_round_f32(x_in[idx]);
            const float * dw  = dw_all + ((size_t) b * S + s) * 32 + h * 4;
            const float y     = (sw0 + dw[0]) * x_cur + (sw1 + dw[1]) * lag1 + (sw2 + dw[2]) * lag2 + (sw3 + dw[3]) * lag3;
            out[idx] = bf16_round_f32((y + x_cur) * (1.0f / 3.0f));
            lag3 = lag2; lag2 = lag1; lag1 = x_cur;
        }
    }
}

static void cpu_op_prep_kv(ggml_tensor * dst, int ith, int nth, void *) {
    const int32_t is_v = dst->op_params[10];
    const int32_t M    = (int32_t) (ggml_nelements(dst) / 1024);
    const float * x_in = static_cast<const float *>(dst->src[0]->data);
    float *       out  = static_cast<float *>(dst->data);
    constexpr float kClipQK = 22.627416997969522f;

    const int32_t tot_mh = M * 8;
    for (int32_t mh = ith; mh < tot_mh; mh += nth) {
        const float * xp = x_in + (size_t) mh * 128;
        float *       yp = out  + (size_t) mh * 128;
        float tmp[128];
        float sq = 0.0f;
        for (int d = 0; d < 128; ++d) {
            tmp[d] = bf16_round_f32(xp[d]);
            sq += tmp[d] * tmp[d];
        }
        const float sc = (is_v != 0)
            ? (1.0f / sqrtf(sq * (1.0f / 128.0f) + 1e-6f))
            : fminf(1.0f, kClipQK / (sqrtf(sq) + 1e-6f));
        for (int d = 0; d < 128; ++d) {
            yp[d] = bf16_round_f32(tmp[d] * sc);
        }
    }
}

static float cpu_read_kv_elem(const uint8_t * base, ggml_type tp, int d) {
    if (tp == GGML_TYPE_BF16) {
        return ggml_bf16_to_fp32(reinterpret_cast<const ggml_bf16_t *>(base)[d]);
    }
    if (tp == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(base)[d]);
    }
    return reinterpret_cast<const float *>(base)[d];
}

static void cpu_op_attn_rel_sink(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const int32_t is_local = dst->op_params[10];
    const int32_t M        = (int32_t) dst->ne[1];
    const ggml_tensor * tk = dst->src[1];
    const ggml_tensor * tv = dst->src[2];
    const int32_t n_kv     = (int32_t) tk->ne[2];
    const int32_t ns       = (int32_t) tk->ne[3];
    const ggml_type kvt    = tk->type;
    const auto * sw        = static_cast<const ggml_tensor * const *>(userdata);

    const float *   q_raw    = static_cast<const float *>(dst->src[0]->data);
    const uint8_t * k_cache  = static_cast<const uint8_t *>(tk->data);
    const uint8_t * v_cache  = static_cast<const uint8_t *>(tv->data);
    const float *   r_raw    = static_cast<const float *>(dst->src[3]->data);
    const float *   q_clip   = static_cast<const float *>(sw[0]->data);
    const float *   r_clip   = static_cast<const float *>(sw[1]->data);
    const float *   rel_proj = static_cast<const float *>(sw[2]->data);
    const float *   sinks    = static_cast<const float *>(dst->src[4]->data);
    const float *   gate_raw = static_cast<const float *>(dst->src[5]->data);
    const int32_t * pos_pack = static_cast<const int32_t *>(dst->src[6]->data);
    float *         out      = static_cast<float *>(dst->data);

    constexpr float kClipQK = 22.627416997969522f;
    constexpr float kScale  = 0.08838834764831845f;

    const int32_t tot_mh = M * 24;
    for (int32_t mh = ith; mh < tot_mh; mh += nth) {
        const int32_t m   = mh / 24;
        const int32_t hq  = mh - m * 24;
        const int32_t hkv = hq / 3;
        const int32_t q_pos = pos_pack[m];
        const int32_t q_st  = std::max(0, std::min(ns - 1, pos_pack[M + m]));
        const int32_t * k_pos_st = pos_pack + 2 * M + (size_t) q_st * n_kv;
        const float tau = is_local ? 1.0f : (1.0f + 0.1f * logf(fmaxf(1.0f, (float)(q_pos + 1) * (1.0f / 8192.0f))));

        float qv[128], rv[16], o_acc[128];
        float q_sq = 0.0f;
        const int off_q = hq * 128;
        for (int d = 0; d < 128; ++d) {
            qv[d] = bf16_round_f32(q_raw[(size_t) m * 3072 + off_q + d]) * q_clip[off_q + d];
            q_sq += qv[d] * qv[d];
        }
        const float q_cf = fminf(1.0f, kClipQK / (sqrtf(q_sq) + 1e-6f));
        for (int d = 0; d < 128; ++d) {
            qv[d] = bf16_round_f32(qv[d] * q_cf);
            if (!is_local) qv[d] = bf16_round_f32(qv[d] * tau);
            qv[d] *= kScale;
            o_acc[d] = 0.0f;
        }

        float r_sq = 0.0f;
        for (int d = 0; d < 16; ++d) {
            rv[d] = bf16_round_f32(r_raw[(size_t) m * 384 + hq * 16 + d]) * r_clip[hq * 16 + d];
            r_sq += rv[d] * rv[d];
        }
        const float r_cf = fminf(1.0f, 8.0f / (sqrtf(r_sq) + 1e-6f));
        for (int d = 0; d < 16; ++d) {
            rv[d] = bf16_round_f32(rv[d] * r_cf);
            if (!is_local) rv[d] = bf16_round_f32(rv[d] * tau);
        }

        const uint8_t * k_st_base = k_cache + (size_t) q_st * tk->nb[3] + (size_t) hkv * tk->nb[1];
        const uint8_t * v_st_base = v_cache + (size_t) q_st * tv->nb[3] + (size_t) hkv * tv->nb[1];
        float m_max = -1e30f, l_sum = 0.0f;

        for (int32_t kv_i = 0; kv_i < n_kv; ++kv_i) {
            const int32_t kp   = k_pos_st[kv_i];
            const int32_t dist = q_pos - kp;
            if (kp < 0 || dist < 0 || (is_local && dist >= 1024)) continue;

            const uint8_t * kp_bytes = k_st_base + (size_t) kv_i * tk->nb[2];
            float qk = 0.0f;
            for (int d = 0; d < 128; ++d) {
                qk += qv[d] * cpu_read_kv_elem(kp_bytes, kvt, d);
            }
            if (dist < 1024) {
                const float * rp = rel_proj + (size_t) dist * 16;
                for (int d = 0; d < 16; ++d) qk += rv[d] * rp[d];
            }
            const float m_new = fmaxf(m_max, qk);
            const float alpha = (m_max > -1e29f) ? expf(m_max - m_new) : 0.0f;
            const float wt    = expf(qk - m_new);
            l_sum = l_sum * alpha + wt;
            m_max = m_new;
            const uint8_t * vp_bytes = v_st_base + (size_t) kv_i * tv->nb[2];
            for (int d = 0; d < 128; ++d) {
                o_acc[d] = o_acc[d] * alpha + wt * cpu_read_kv_elem(vp_bytes, kvt, d);
            }
        }

        const float sink  = sinks[(size_t) m * 24 + hq];
        const float m_tot = (l_sum > 0.0f) ? fmaxf(sink, m_max) : sink;
        float denom = expf(sink - m_tot);
        float a_scale = 0.0f;
        if (l_sum > 0.0f) {
            a_scale = expf(m_max - m_tot);
            denom += l_sum * a_scale;
        }
        const float inv_d = 1.0f / fmaxf(denom, 1e-20f);
        float * dp = out + (size_t) m * 3072 + off_q;
        const float * gp = gate_raw + (size_t) m * 3072 + off_q;
        for (int d = 0; d < 128; ++d) {
            const float a  = bf16_round_f32(o_acc[d] * a_scale * inv_d);
            const float g  = bf16_round_f32(gp[d]);
            const float sg = 1.0f / (1.0f + expf(-g));
            dp[d] = bf16_round_f32(a * sg);
        }
    }
}

static void cpu_op_softcap_mask_head(ggml_tensor * dst, int ith, int nth, void *) {
    const int32_t vocab_unpad = dst->op_params[10];
    float softcap = 30.0f;
    std::memcpy(&softcap, &dst->op_params[11], sizeof(float));
    const int32_t in_stride = (int32_t) dst->src[0]->ne[0];
    const int32_t out_vocab = (int32_t) dst->ne[0];
    const int32_t n_rows    = (int32_t) dst->ne[1];
    const float * src       = static_cast<const float *>(dst->src[0]->data);
    float *       out       = static_cast<float *>(dst->data);

    for (int32_t r = ith; r < n_rows; r += nth) {
        const float * sp = src + (size_t) r * in_stride;
        float *       dp = out + (size_t) r * out_vocab;
        const int32_t lim = std::min(vocab_unpad, std::min(in_stride, out_vocab));
        for (int32_t v = 0; v < lim; ++v) {
            dp[v] = (softcap > 0.0f) ? (softcap * tanhf(sp[v] / softcap)) : sp[v];
        }
        for (int32_t v = lim; v < out_vocab; ++v) {
            dp[v] = -INFINITY;
        }
    }
}

static void cpu_op_mtp_stash_tap(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) return;
    const size_t nb = ggml_nbytes(dst);
    if (dst->data != dst->src[0]->data) {
        std::memcpy(dst->data, dst->src[0]->data, nb);
    }
}

static void tag_ink2_op(ggml_tensor * t, ggml_cuda_ink2_op_kind kind) {
    t->op_params[GGML_CUDA_INK2_OP_MAGIC_SLOT]     = (int32_t) GGML_CUDA_INK2_OP_MAGIC;
    t->op_params[GGML_CUDA_INK2_OP_MAGIC_SLOT + 1] = (int32_t) kind;
}

} // namespace inkling2_model_detail

// ============================================================================
// Public `LLAMA_API` Hooks (`common/speculative.cpp` + `llama-inkling2-bench`)
// ============================================================================

bool llama_model_has_inkling2_mtp(const struct llama_model * model) {
    return model != nullptr &&
           model->arch == LLM_ARCH_INKLING2 &&
           model->hparams.ink2_mtp_layer_count > 0 &&
           !model->ink2_mtp_layers.empty();
}

void * llama_inkling2_get_cuda_handle(
        const struct llama_model * model,
        int32_t                    b_max,
        int32_t                    max_seq,
        bool                       want_full_vram_moe) {
    if (!model || model->arch != LLM_ARCH_INKLING2) {
        return nullptr;
    }
    return inkling2_model_detail::ensure_cuda_handle(*model, b_max, max_seq, want_full_vram_moe);
}

int32_t llama_inkling2_mtp_draft(
        struct llama_context * lctx,
        int32_t                pos0,
        llama_token            anchor_tok,
        int32_t                k_draft,
        llama_token          * out_draft_toks) {
    if (!lctx || !out_draft_toks || k_draft <= 0) {
        return 0;
    }
    const llama_model * model = llama_get_model(lctx);
    if (!llama_model_has_inkling2_mtp(model)) {
        return 0;
    }
    const auto * api = inkling2_model_detail::find_cuda_api(*model);
    if (!api || !api->mtp_draft_block) {
        return 0;
    }
    const bool has_mstream = (model->moe_stream() != nullptr);
    const bool phase_a_env = getenv("INKLING2_PHASE_A_PLUGIN_ONLY") && atoi(getenv("INKLING2_PHASE_A_PLUGIN_ONLY")) != 0;
    const bool full_moe    = !has_mstream && !phase_a_env && inkling2_model_detail::model_all_on_single_gpu(*model, nullptr);
    void * handle = inkling2_model_detail::ensure_cuda_handle(
        *model,
        (int32_t) std::max(1u, llama_n_seq_max(lctx)),
        (int32_t) std::max(1040u, llama_n_ctx(lctx)),
        full_moe);
    if (!handle) {
        return 0;
    }
    int32_t rc = api->mtp_draft_block(handle, pos0, (int32_t) anchor_tok, k_draft, reinterpret_cast<int32_t *>(out_draft_toks));
    return (rc > 0) ? rc : 0;
}

// ============================================================================
// `llama_model_inkling2` Implementation
// ============================================================================

llama_model_inkling2::~llama_model_inkling2() {
    const auto * api = inkling2_model_detail::find_cuda_api(*this);
    if (api && api->destroy_by_model) {
        api->destroy_by_model(this);
    }
}

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    hparams.f_norm_rms_eps = 1e-6f;
    if (!ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false) && ml.metadata != nullptr) {
        const int64_t kv_id = gguf_find_key(ml.metadata, "inkling2.attention.layer_norm_rms_eps");
        if (kv_id >= 0) {
            hparams.f_norm_rms_eps = gguf_get_val_f32(ml.metadata, kv_id);
        }
    }
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,              hparams.n_expert_shared, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,               hparams.expert_gating_func);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,             hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,              hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,              hparams.attn_res_block_size, false);
    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,             hparams.situ_beta, false);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA,      hparams.situ_linear_beta, false);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,          hparams.f_final_logit_softcapping, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa, false);

    ml.get_key(LLM_KV_INK2_UNPADDED_VOCAB_SIZE,    hparams.ink2_unpadded_vocab_size);
    ml.get_key(LLM_KV_INK2_PADDED_HEAD_VOCAB_SIZE, hparams.ink2_padded_head_vocab_size);
    ml.get_key(LLM_KV_INK2_IMAGE_TOKEN_ID,         hparams.ink2_image_token_id, false);
    ml.get_key(LLM_KV_INK2_PHYS_BLOCK_COUNT,       hparams.ink2_phys_block_count);
    ml.get_key(LLM_KV_INK2_LOOP_START,             hparams.ink2_loop_start);
    ml.get_key(LLM_KV_INK2_LOOP_END,               hparams.ink2_loop_end);
    ml.get_key(LLM_KV_INK2_LOOP_PASSES,            hparams.ink2_loop_passes);
    ml.get_key(LLM_KV_INK2_LOOP_GATE_RANK,         hparams.ink2_loop_gate_rank, false);
    ml.get_key(LLM_KV_INK2_D_REL,                  hparams.ink2_d_rel, false);
    ml.get_key(LLM_KV_INK2_REL_EXTENT,             hparams.ink2_rel_extent, false);
    ml.get_key(LLM_KV_INK2_QK_L2_CLIP,             hparams.ink2_qk_l2_clip, false);
    ml.get_key(LLM_KV_INK2_REL_L2_CLIP,            hparams.ink2_rel_l2_clip, false);
    ml.get_key(LLM_KV_INK2_LOG_SCALING_FLOOR,      hparams.ink2_log_scaling_floor, false);
    ml.get_key(LLM_KV_INK2_LOG_SCALING_ALPHA,      hparams.ink2_log_scaling_alpha, false);
    ml.get_key(LLM_KV_INK2_SCONV_KERNEL,           hparams.ink2_sconv_kernel);
    ml.get_key(LLM_KV_INK2_MTP_LAYER_COUNT,        hparams.ink2_mtp_layer_count, false);
    ml.get_key(LLM_KV_INK2_MTP_FF_LENGTH,          hparams.ink2_mtp_ff_length, false);
    ml.get_key(LLM_KV_INK2_MTP_BLOCK_SIZE,         hparams.ink2_mtp_block_size, false);
    ml.get_key(LLM_KV_INK2_MTP_MARKOV_RANK,        hparams.ink2_mtp_markov_rank, false);
    ml.get_key(LLM_KV_INK2_MTP_ROPE_THETA,         hparams.ink2_mtp_rope_theta, false);
    ml.get_key_or_arr(LLM_KV_INK2_MTP_TAP_VISITS,  hparams.ink2_mtp_tap_visits, 4, false);

    if (hparams.n_swa == 0) {
        hparams.n_swa = 1024;
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    const int loop_beg = (int) hparams.ink2_loop_start;
    const int loop_end = (int) hparams.ink2_loop_end;
    const int n_passes = (int) hparams.ink2_loop_passes;
    const int tot_v    = loop_beg + n_passes * (loop_end - loop_beg) + ((int) hparams.ink2_phys_block_count - loop_end);
    GGML_ASSERT((uint32_t) tot_v == hparams.n_layer_all);

    for (uint32_t v = 0; v < hparams.n_layer_all; ++v) {
        const int phys_li = inkling2_model_detail::visit_to_phys_layer((int) v, loop_beg, loop_end, n_passes);
        const bool is_global = (phys_li == 5 || phys_li == 11 || phys_li == 17 || phys_li == 23);
        hparams.is_swa_impl[v]  = !is_global;
        hparams.is_recr_impl[v] = true;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_phys   = hparams.ink2_phys_block_count;
    const int64_t loop_beg = hparams.ink2_loop_start;
    const int64_t loop_end = hparams.ink2_loop_end;
    const int64_t n_passes = hparams.ink2_loop_passes;
    const int64_t loop_len = loop_end - loop_beg;
    const int64_t d_q      = n_embd_head_k * n_head;
    const int64_t d_kv     = n_embd_k_gqa;
    const int64_t d_rel    = (int64_t) hparams.ink2_d_rel * n_head;
    const int64_t k_sc     = hparams.ink2_sconv_kernel;
    const int64_t g_rank   = hparams.ink2_loop_gate_rank;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    { n_embd, n_vocab }, 0);
    tok_norm    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), { n_embd }, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,     "weight"),    { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,          "weight"), { n_embd, n_vocab }, 0);

    ink2_output_res_query = create_tensor(tn(LLM_TENSOR_INK2_OUTPUT_RES_QUERY, "weight"), { n_embd }, 0);
    ink2_loop_msg_queries = create_tensor(tn(LLM_TENSOR_INK2_LOOP_MSG_QUERIES, "weight"), { n_embd, 4, loop_len }, 0);
    ink2_mlp_res_queries  = create_tensor(tn(LLM_TENSOR_INK2_MLP_RES_QUERIES,  "weight"), { n_embd, n_phys }, 0);

    for (int li = 0; li < (int) n_phys; ++li) {
        const int v0 = inkling2_model_detail::phys_layer_to_first_visit(li, (int) loop_beg, (int) loop_end, (int) n_passes);
        auto & layer = layers[v0];
        const bool is_local = (li != 5 && li != 11 && li != 17 && li != 23);
        const bool is_loop  = (li >= loop_beg && li < loop_end);

        if (li >= 1) {
            if (is_local) {
                layer.ink2_attn_res_query = create_tensor(tn(LLM_TENSOR_INK2_ATTN_RES_QUERY, "weight", li), { n_embd }, 0);
            } else {
                layer.ink2_attn_res_query = create_tensor(tn(LLM_TENSOR_INK2_ATTN_RES_QUERY, "weight", li), { n_embd, 3 }, 0);
            }
        }

        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", li), { n_embd, d_q }, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", li), { n_embd, d_kv }, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", li), { n_embd, d_kv }, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", li), { n_embd, d_q }, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", li), { d_q, n_embd }, 0);

        layer.ink2_attn_rel      = create_tensor(tn(LLM_TENSOR_INK2_ATTN_REL,      "weight", li), { n_embd, d_rel }, 0);
        layer.ink2_attn_sink     = create_tensor(tn(LLM_TENSOR_INK2_ATTN_SINK,     "weight", li), { n_embd, n_head }, 0);
        layer.ink2_attn_q_clip   = create_tensor(tn(LLM_TENSOR_INK2_ATTN_Q_CLIP,   "weight", li), { n_embd_head_k, n_head }, TENSOR_ALLOW_RESHAPE);
        layer.ink2_attn_r_clip   = create_tensor(tn(LLM_TENSOR_INK2_ATTN_R_CLIP,   "weight", li), { (int64_t) hparams.ink2_d_rel, n_head }, TENSOR_ALLOW_RESHAPE);
        layer.ink2_attn_rel_proj = create_tensor(tn(LLM_TENSOR_INK2_ATTN_REL_PROJ, "weight", li), { (int64_t) hparams.ink2_d_rel, (int64_t) hparams.ink2_rel_extent }, 0);
        layer.ink2_attn_sconv    = create_tensor(tn(LLM_TENSOR_INK2_ATTN_SCONV,    "weight", li), { k_sc, n_embd }, 0);
        layer.ink2_ffn_sconv     = create_tensor(tn(LLM_TENSOR_INK2_FFN_SCONV,     "weight", li), { k_sc, n_embd }, 0);

        layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", li), { n_embd }, 0);
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", li), { n_embd, n_expert }, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   li), { n_expert }, 0);
        layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,   "weight", li), { n_embd, hparams.n_ff_exp(), n_expert }, 0);
        layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,     "weight", li), { n_embd, hparams.n_ff_exp(), n_expert }, 0);
        layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", li), { hparams.n_ff_exp(), n_embd, n_expert }, 0);

        if (is_local) {
            layer.attn_norm         = create_tensor(tn(LLM_TENSOR_ATTN_NORM,        "weight", li), { n_embd }, 0);
            layer.ink2_attn_k_sconv = create_tensor(tn(LLM_TENSOR_INK2_ATTN_K_SCONV,"weight", li), { k_sc, d_kv }, 0);
            layer.ink2_attn_v_sconv = create_tensor(tn(LLM_TENSOR_INK2_ATTN_V_SCONV,"weight", li), { k_sc, d_kv }, 0);
        } else {
            layer.ink2_attn_q_in_norm  = create_tensor(tn(LLM_TENSOR_INK2_ATTN_Q_IN_NORM,  "weight", li), { n_embd }, 0);
            layer.ink2_attn_k_in_norm  = create_tensor(tn(LLM_TENSOR_INK2_ATTN_K_IN_NORM,  "weight", li), { n_embd }, 0);
            layer.ink2_attn_v_in_norm  = create_tensor(tn(LLM_TENSOR_INK2_ATTN_V_IN_NORM,  "weight", li), { n_embd }, 0);
            layer.ink2_attn_k_static_w = create_tensor(tn(LLM_TENSOR_INK2_ATTN_K_STATIC_W, "weight", li), { d_kv, k_sc }, 0);
            layer.ink2_attn_k_wproj    = create_tensor(tn(LLM_TENSOR_INK2_ATTN_K_WPROJ,    "weight", li), { n_embd, n_head_kv * k_sc }, 0);
            layer.ink2_attn_v_static_w = create_tensor(tn(LLM_TENSOR_INK2_ATTN_V_STATIC_W, "weight", li), { d_kv, k_sc }, 0);
            layer.ink2_attn_v_wproj    = create_tensor(tn(LLM_TENSOR_INK2_ATTN_V_WPROJ,    "weight", li), { n_embd, n_head_kv * k_sc }, 0);
        }

        if (is_loop) {
            const llm_tensor t_dn[4] = {
                LLM_TENSOR_INK2_CTRL_ATTN_DOWN, LLM_TENSOR_INK2_CTRL_K_DOWN,
                LLM_TENSOR_INK2_CTRL_V_DOWN,    LLM_TENSOR_INK2_CTRL_MLP_DOWN
            };
            const llm_tensor t_fg[4] = {
                LLM_TENSOR_INK2_CTRL_ATTN_FORGET, LLM_TENSOR_INK2_CTRL_K_FORGET,
                LLM_TENSOR_INK2_CTRL_V_FORGET,    LLM_TENSOR_INK2_CTRL_MLP_FORGET
            };
            const llm_tensor t_gn[4] = {
                LLM_TENSOR_INK2_CTRL_ATTN_GAIN, LLM_TENSOR_INK2_CTRL_K_GAIN,
                LLM_TENSOR_INK2_CTRL_V_GAIN,    LLM_TENSOR_INK2_CTRL_MLP_GAIN
            };
            for (int c = 0; c < 4; ++c) {
                layer.ink2_ctrl_down[c]   = create_tensor(tn(t_dn[c], "weight", li), { 2 * n_embd, g_rank }, 0);
                layer.ink2_ctrl_forget[c] = create_tensor(tn(t_fg[c], "weight", li), { g_rank, n_embd }, 0);
                layer.ink2_ctrl_gain[c]   = create_tensor(tn(t_gn[c], "weight", li), { g_rank, n_embd }, 0);
            }
        }
    }

    // Alias loop passes p = 1, 2 (`visits 18..41`) to `layers[6..17]`:
    for (int p = 1; p < (int) n_passes; ++p) {
        for (int j = 0; j < (int) loop_len; ++j) {
            layers[(size_t)(loop_beg + p * loop_len + j)] = layers[(size_t)(loop_beg + j)];
        }
    }

    // Embedded MTP-8 (`DSpark`) tensors:
    if (hparams.ink2_mtp_layer_count > 0 && ml.get_weight(tn(LLM_TENSOR_INK2_MTP_FUSE_ALPHA, "weight").str().c_str()) != nullptr) {
        const int64_t v_pad  = hparams.ink2_padded_head_vocab_size;
        const int64_t m_rank = hparams.ink2_mtp_markov_rank;
        const int64_t m_ff   = hparams.ink2_mtp_ff_length;
        const auto * w_w1    = ml.get_weight(tn(LLM_TENSOR_INK2_MTP_MH_W1, "weight").str().c_str());
        const int64_t w1_rows = w_w1 ? w_w1->tensor->ne[1] : n_vocab;

        ink2_mtp_fuse_alpha = create_tensor(tn(LLM_TENSOR_INK2_MTP_FUSE_ALPHA,     "weight"), { 4, 4 },       TENSOR_NOT_REQUIRED);
        ink2_mtp_mask_embed = create_tensor(tn(LLM_TENSOR_INK2_MTP_MASK_EMBED,     "weight"), { n_embd },     TENSOR_NOT_REQUIRED);
        ink2_mtp_norm       = create_tensor(tn(LLM_TENSOR_INK2_MTP_NORM,           "weight"), { n_embd },     TENSOR_NOT_REQUIRED);
        ink2_mtp_mh_dflare  = create_tensor(tn(LLM_TENSOR_INK2_MTP_MH_DFLARE_NORM, "weight"), { n_embd },     TENSOR_NOT_REQUIRED);
        ink2_mtp_rnn_w      = create_tensor(tn(LLM_TENSOR_INK2_MTP_RNN,            "weight"), { 5120, 768 },  TENSOR_NOT_REQUIRED);
        ink2_mtp_rnn_b      = create_tensor(tn(LLM_TENSOR_INK2_MTP_RNN,            "bias"),   { 768 },        TENSOR_NOT_REQUIRED);
        ink2_mtp_fold_w     = create_tensor(tn(LLM_TENSOR_INK2_MTP_FOLD,           "weight"), { m_rank, v_pad }, TENSOR_NOT_REQUIRED);
        ink2_mtp_fold_b     = create_tensor(tn(LLM_TENSOR_INK2_MTP_FOLD,           "bias"),   { v_pad },      TENSOR_NOT_REQUIRED);
        ink2_mtp_mh_w1      = create_tensor(tn(LLM_TENSOR_INK2_MTP_MH_W1,          "weight"), { m_rank, w1_rows }, TENSOR_NOT_REQUIRED);

        ink2_mtp_layers.resize(hparams.ink2_mtp_layer_count);
        for (int mi = 0; mi < (int) hparams.ink2_mtp_layer_count; ++mi) {
            auto & m = ink2_mtp_layers[mi];
            auto norm_dim = [&](llm_tensor lt, int64_t fallback) -> int64_t {
                const auto * wt = ml.get_weight(tn(lt, "weight", mi).str().c_str());
                return wt ? wt->tensor->ne[0] : fallback;
            };
            m.dflare_norm = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_DFLARE_NORM, "weight", mi), { n_embd }, TENSOR_NOT_REQUIRED);
            m.in_norm     = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_IN_NORM,     "weight", mi), { n_embd }, TENSOR_NOT_REQUIRED);
            m.post_norm   = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_POST_NORM,   "weight", mi), { n_embd }, TENSOR_NOT_REQUIRED);
            m.q_norm      = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_Q_NORM,      "weight", mi), { norm_dim(LLM_TENSOR_INK2_MTP_BLK_Q_NORM, n_embd_head_k) }, TENSOR_NOT_REQUIRED);
            m.k_norm      = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_K_NORM,      "weight", mi), { norm_dim(LLM_TENSOR_INK2_MTP_BLK_K_NORM, n_embd_head_k) }, TENSOR_NOT_REQUIRED);
            m.k_norm_ctx  = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_K_NORM_CTX,  "weight", mi), { norm_dim(LLM_TENSOR_INK2_MTP_BLK_K_NORM_CTX, n_embd_head_k) }, TENSOR_NOT_REQUIRED);
            m.wq          = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_ATTN_Q,      "weight", mi), { n_embd, d_q },  TENSOR_NOT_REQUIRED);
            m.wk          = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_ATTN_K,      "weight", mi), { n_embd, d_kv }, TENSOR_NOT_REQUIRED);
            m.wv          = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_ATTN_V,      "weight", mi), { n_embd, d_kv }, TENSOR_NOT_REQUIRED);
            m.wk_ctx      = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_ATTN_K_CTX,  "weight", mi), { n_embd, d_kv }, TENSOR_NOT_REQUIRED);
            m.wv_ctx      = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_ATTN_V_CTX,  "weight", mi), { n_embd, d_kv }, TENSOR_NOT_REQUIRED);
            m.wo          = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_ATTN_OUT,    "weight", mi), { d_q, n_embd },  TENSOR_NOT_REQUIRED);
            m.ffn_gate    = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_FFN_GATE,    "weight", mi), { n_embd, m_ff }, TENSOR_NOT_REQUIRED);
            m.ffn_up      = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_FFN_UP,      "weight", mi), { n_embd, m_ff }, TENSOR_NOT_REQUIRED);
            m.ffn_down    = create_tensor(tn(LLM_TENSOR_INK2_MTP_BLK_FFN_DOWN,    "weight", mi), { m_ff, n_embd }, TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_inkling2::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params), model(model) {

    // Layer-major Phase 2 expert wave graph (`--moe-stream-layer-major 1`):
    if (params.lm_phase == 2) {
        const int phys_il = params.lm_il0;
        const int v0 = inkling2_model_detail::phys_layer_to_first_visit(
            phys_il, (int) hparams.ink2_loop_start, (int) hparams.ink2_loop_end, (int) hparams.ink2_loop_passes);
        const auto & layer = model.layers[v0];
        build_moe_ffn(params.lm_x,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SITU,
            hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            phys_il);
        return;
    }

    const int64_t n_vocab  = model.vocab.n_tokens();
    const int64_t n_seqs   = std::max<int64_t>(1, ubatch.n_seqs);
    const int64_t n_seq_t  = std::max<int64_t>(1, ubatch.n_seq_tokens);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(n_tokens == n_seqs * n_seq_t);

    const bool phase_a_only = getenv("INKLING2_PHASE_A_PLUGIN_ONLY") && atoi(getenv("INKLING2_PHASE_A_PLUGIN_ONLY")) != 0;
    const auto * cuda_api   = inkling2_model_detail::find_cuda_api(model);

    // ========================================================================
    // Phase B Fast Path (`-ngl 99` Full-VRAM Fused 48-Visit CUDA Trunk + Head)
    // ========================================================================
    if (!phase_a_only && mstream == nullptr && params.lm_phase == 0 &&
        params.lm_il0 == 0 && params.lm_il1 < 0 && !cparams.embeddings && cuda_api != nullptr) {
        void * cuda_h = inkling2_model_detail::ensure_cuda_handle(
            model, (int32_t) std::max(1u, cparams.n_seq_max), (int32_t) std::max(1040u, cparams.n_ctx), true);
        if (cuda_h != nullptr) {
            ggml_tensor * inpL        = build_inp_embd(model.tok_embd, 1.0f, model.tok_norm);
            ggml_tensor * inp_pos     = build_inp_pos();
            ggml_tensor * inp_out_ids = build_inp_out_ids();

            const bool use_tok_src0 = (res->t_inp_tokens != nullptr && ubatch.token != nullptr && !ubatch.embd);
            const int32_t single_sid = (n_seqs == 1) ? (ubatch.seq_id ? ubatch.seq_id[0][0] : 0) : -1;
            ggml_tensor * t_seq_ids = nullptr;
            if (single_sid < 0) {
                t_seq_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_seqs);
                ggml_set_input(t_seq_ids);
            }
            res->add_input(std::make_unique<inkling2_model_detail::llm_graph_input_ink2_ftrunk>(
                cuda_api, cuda_h, t_seq_ids, mctx, use_tok_src0));

            const int32_t actual_n_outputs = inp_out_ids ? (int32_t) inp_out_ids->ne[0] : (int32_t) n_tokens;
            const int64_t tensor_rows      = std::max<int64_t>(1, actual_n_outputs);

            ggml_tensor * args[4] = { nullptr, nullptr, nullptr, nullptr };
            int n_args   = 0;
            int opt_mask = 0;
            args[n_args++] = use_tok_src0 ? res->t_inp_tokens : inpL;
            args[n_args++] = inp_pos;
            if (t_seq_ids) {
                args[n_args++] = t_seq_ids;
                opt_mask |= 1;
            }
            if (inp_out_ids) {
                args[n_args++] = inp_out_ids;
                opt_mask |= 2;
            }

            ggml_tensor * cur = ggml_custom_4d(
                ctx0, GGML_TYPE_F32, n_vocab, tensor_rows, 1, 1,
                args, n_args, inkling2_model_detail::cpu_op_noop_abort, 1, cuda_h);
            inkling2_model_detail::tag_ink2_op(cur, GGML_CUDA_INK2_OP_FUSED_TRUNK);
            cur->op_params[10] = (int32_t) n_seqs;
            cur->op_params[11] = (int32_t) n_seq_t;
            cur->op_params[12] = single_sid;
            cur->op_params[13] = use_tok_src0 ? 1 : 0;
            cur->op_params[14] = opt_mask;
            cur->op_params[15] = actual_n_outputs;
            ggml_set_name(cur, "ink2_fused_trunk_logits");

            res->t_logits = cur;
            ggml_build_forward_expand(gf, cur);
            return;
        }
    }

    // ========================================================================
    // Modular 48-Visit Graph (`--moe-stream` SSD Tiering, `Phase A`, `-ngl 0`)
    // ========================================================================
    ggml_tensor * cur_h = build_inp_embd(model.tok_embd, 1.0f, model.tok_norm);
    cb(cur_h, "model.embed_tokens", -1);
    ggml_build_forward_expand(gf, cur_h);

    auto * inp_hybrid      = build_inp_mem_hybrid_iswa();
    auto * inp_recr        = inp_hybrid->get_recr();
    auto * inp_attn        = inp_hybrid->get_attn();
    const auto * mctx_hyb  = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);
    const auto * mctx_recr = mctx_hyb->get_recr();
    const auto * mctx_iswa = mctx_hyb->get_attn();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Optional VRAM MTP-8 handle (`~0.255 GiB`) for `--spec-type draft-mtp` during `--moe-stream` / `Phase A`:
    void * mtp_cuda_h = (cuda_api != nullptr)
        ? inkling2_model_detail::ensure_cuda_handle(
            model, (int32_t) std::max(1u, cparams.n_seq_max), (int32_t) std::max(1040u, cparams.n_ctx), false)
        : nullptr;

    // Build KV position packs (`base` = Global 4-layer visits, `swa` = Local SWA-1024 visits):
    const int32_t n_kv_base = (int32_t) mctx_iswa->get_base()->get_n_kv();
    const int32_t ns_base   = (int32_t) mctx_iswa->get_base()->get_ns();
    const int32_t n_kv_swa  = (int32_t) mctx_iswa->get_swa()->get_n_kv();
    const int32_t ns_swa    = (int32_t) mctx_iswa->get_swa()->get_ns();

    ggml_tensor * t_pos_base = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 2 * n_tokens + (int64_t) n_kv_base * ns_base);
    ggml_tensor * t_pos_swa  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 2 * n_tokens + (int64_t) n_kv_swa  * ns_swa);
    ggml_set_input(t_pos_base);
    ggml_set_input(t_pos_swa);
    res->add_input(std::make_unique<inkling2_model_detail::llm_graph_input_ink2_kpos>(
        mctx_iswa, cuda_api, mtp_cuda_h, t_pos_base, t_pos_swa, n_kv_base, ns_base, n_kv_swa, ns_swa));

    auto emit_block_attn_res = [&](ggml_tensor * h_in, ggml_tensor * q_w,
                                   ggml_tensor * const * blk_arr, int n_b, int Q, int round_cur) -> ggml_tensor * {
        ggml_tensor * args[7] = { ggml_cont(ctx0, h_in), q_w, nullptr, nullptr, nullptr, nullptr, nullptr };
        for (int i = 0; i < n_b && i < 5; ++i) {
            args[2 + i] = ggml_cont(ctx0, blk_arr[i]);
        }
        ggml_tensor * out = ggml_custom_4d(
            ctx0, GGML_TYPE_F32, n_embd, n_tokens, Q, 1,
            args, 2 + n_b, inkling2_model_detail::cpu_op_block_attn_res, GGML_N_TASKS_MAX, nullptr);
        inkling2_model_detail::tag_ink2_op(out, GGML_CUDA_INK2_OP_BLOCK_ATTN_RES);
        out->op_params[10] = n_b;
        out->op_params[11] = Q;
        out->op_params[12] = round_cur;
        return out;
    };

    auto emit_loop_msg_gate = [&](ggml_tensor * sp, ggml_tensor * mp,
                                  ggml_tensor * c_dn, ggml_tensor * c_fg, ggml_tensor * c_gn) -> ggml_tensor * {
        ggml_tensor * args[5] = { ggml_cont(ctx0, sp), ggml_cont(ctx0, mp), c_dn, c_fg, c_gn };
        ggml_tensor * out = ggml_custom_4d(
            ctx0, GGML_TYPE_F32, n_embd, n_tokens, 1, 1,
            args, 5, inkling2_model_detail::cpu_op_loop_msg_gate, GGML_N_TASKS_MAX, nullptr);
        inkling2_model_detail::tag_ink2_op(out, GGML_CUDA_INK2_OP_LOOP_MSG_GATE);
        return out;
    };

    auto emit_sconv_static = [&](ggml_tensor * x_2d, ggml_tensor * st_3_c_b, ggml_tensor * w4, int64_t C) -> ggml_tensor * {
        ggml_tensor * x_3d  = ggml_reshape_3d(ctx0, ggml_cont(ctx0, x_2d), C, n_seq_t, n_seqs);
        ggml_tensor * st_c  = ggml_cont(ctx0, st_3_c_b);
        ggml_tensor * args[3] = { x_3d, st_c, w4 };
        ggml_tensor * y_3d  = ggml_custom_4d(
            ctx0, GGML_TYPE_F32, C, n_seq_t, n_seqs, 1,
            args, 3, inkling2_model_detail::cpu_op_sconv_static, GGML_N_TASKS_MAX, nullptr);
        inkling2_model_detail::tag_ink2_op(y_3d, GGML_CUDA_INK2_OP_SCONV_STATIC);
        return ggml_reshape_2d(ctx0, y_3d, C, n_tokens);
    };

    auto emit_sconv_dynamic = [&](ggml_tensor * x_2d, ggml_tensor * st_3_c_b,
                                  ggml_tensor * static_w, ggml_tensor * dw_2d) -> ggml_tensor * {
        const int64_t C    = 1024;
        ggml_tensor * x_3d  = ggml_reshape_3d(ctx0, ggml_cont(ctx0, x_2d), C, n_seq_t, n_seqs);
        ggml_tensor * st_c  = ggml_cont(ctx0, st_3_c_b);
        ggml_tensor * dw_3d = ggml_reshape_3d(ctx0, ggml_cont(ctx0, dw_2d), 32, n_seq_t, n_seqs);
        ggml_tensor * args[4] = { x_3d, st_c, static_w, dw_3d };
        ggml_tensor * y_3d  = ggml_custom_4d(
            ctx0, GGML_TYPE_F32, C, n_seq_t, n_seqs, 1,
            args, 4, inkling2_model_detail::cpu_op_sconv_dynamic, GGML_N_TASKS_MAX, nullptr);
        inkling2_model_detail::tag_ink2_op(y_3d, GGML_CUDA_INK2_OP_SCONV_DYNAMIC);
        return ggml_reshape_2d(ctx0, y_3d, C, n_tokens);
    };

    auto emit_prep_kv = [&](ggml_tensor * kv_2d, int is_v) -> ggml_tensor * {
        ggml_tensor * args[1] = { ggml_cont(ctx0, kv_2d) };
        ggml_tensor * out = ggml_custom_4d(
            ctx0, GGML_TYPE_F32, 128, 8, n_tokens, 1,
            args, 1, inkling2_model_detail::cpu_op_prep_kv, GGML_N_TASKS_MAX, nullptr);
        inkling2_model_detail::tag_ink2_op(out, GGML_CUDA_INK2_OP_PREP_KV);
        out->op_params[10] = is_v;
        return out;
    };

    auto maybe_stash_mtp_tap = [&](ggml_tensor * h, int tap_slot) -> ggml_tensor * {
        if (!mtp_cuda_h) return h;
        ggml_tensor * args[1] = { ggml_cont(ctx0, h) };
        ggml_tensor * out = ggml_custom_4d(
            ctx0, GGML_TYPE_F32, n_embd, n_tokens, 1, 1,
            args, 1, inkling2_model_detail::cpu_op_mtp_stash_tap, 1, mtp_cuda_h);
        inkling2_model_detail::tag_ink2_op(out, GGML_CUDA_INK2_OP_MTP_STASH_TAP);
        out->op_params[10] = tap_slot;
        ggml_build_forward_expand(gf, out);
        return out;
    };

    const int loop_beg   = (int) hparams.ink2_loop_start;
    const int loop_end   = (int) hparams.ink2_loop_end;
    const int n_passes   = (int) hparams.ink2_loop_passes;
    const int loop_len   = loop_end - loop_beg;
    const int block_sz   = (int) hparams.attn_res_block_size;
    const int64_t d_conv = (int64_t) hparams.ink2_sconv_kernel - 1; // 3
    const int64_t c_tot  = 6656; // 1024 + 1024 + 2304 + 2304
    const uint32_t rs_hd = mctx_recr->get_head();
    const uint32_t rs_sz = mctx_recr->get_size();
    const int64_t k_snap = cparams.n_rs_seq > 0 ? (int64_t) cparams.n_rs_seq + 1 : 1;
    const int64_t n_wr   = std::min<int64_t>(n_seq_t, k_snap);

    ggml_tensor * blks[6]  = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    int           n_blk    = 0;

    auto run_visit_graph = [&](int v_idx, int phys_li, int pass_idx, int next_phys_li, ggml_tensor * msgs_4) {
        for (int t = 0; t < 3; ++t) {
            if (hparams.ink2_mtp_tap_visits[t] == v_idx) {
                cur_h = maybe_stash_mtp_tap(cur_h, t);
            }
        }
        res->t_layer_inp[v_idx] = cur_h;

        const auto & L      = model.layers[v_idx];
        const bool is_local = hparams.is_swa(v_idx);
        const bool is_bound = (block_sz > 0 && (phys_li % block_sz == 0) && (n_blk < 5));

        // 1. Unpack this visit's 4 ShortConvolution states `[3, 6656, n_seqs]`:
        ggml_tensor * conv_state = mctx_recr->get_r_l(v_idx);
        ggml_tensor * conv_rs    = build_rs(inp_recr, conv_state, hparams.n_embd_r(), (int32_t) n_seqs);
        ggml_tensor * conv_3d    = ggml_reshape_3d(ctx0, conv_rs, d_conv, c_tot, n_seqs);
        ggml_tensor * st_k       = ggml_view_3d(ctx0, conv_3d, d_conv, 1024,   n_seqs, conv_3d->nb[1], conv_3d->nb[2], 0 * conv_3d->nb[1]);
        ggml_tensor * st_v       = ggml_view_3d(ctx0, conv_3d, d_conv, 1024,   n_seqs, conv_3d->nb[1], conv_3d->nb[2], 1024 * conv_3d->nb[1]);
        ggml_tensor * st_a       = ggml_view_3d(ctx0, conv_3d, d_conv, n_embd, n_seqs, conv_3d->nb[1], conv_3d->nb[2], 2048 * conv_3d->nb[1]);
        ggml_tensor * st_m       = ggml_view_3d(ctx0, conv_3d, d_conv, n_embd, n_seqs, conv_3d->nb[1], conv_3d->nb[2], (2048 + n_embd) * conv_3d->nb[1]);

        auto msg_slot = [&](int sl) -> ggml_tensor * {
            return ggml_view_2d(ctx0, msgs_4, n_embd, n_tokens, msgs_4->nb[1], (size_t) sl * msgs_4->nb[2]);
        };

        // 2. Pre-Attention BlockAttentionResidual + Optional LoopMessageGate + Input RMSNorm:
        ggml_tensor * qin = nullptr;
        ggml_tensor * kin = nullptr;
        ggml_tensor * vin = nullptr;
        if (is_local) {
            ggml_tensor * ares = L.ink2_attn_res_query
                ? ggml_reshape_2d(ctx0, emit_block_attn_res(cur_h, L.ink2_attn_res_query, blks, n_blk, 1, 1), n_embd, n_tokens)
                : cur_h;
            if (msgs_4 != nullptr) {
                ggml_tensor * q_g = emit_loop_msg_gate(ares, msg_slot(0), L.ink2_ctrl_down[0], L.ink2_ctrl_forget[0], L.ink2_ctrl_gain[0]);
                ggml_tensor * k_g = emit_loop_msg_gate(ares, msg_slot(1), L.ink2_ctrl_down[1], L.ink2_ctrl_forget[1], L.ink2_ctrl_gain[1]);
                ggml_tensor * v_g = emit_loop_msg_gate(ares, msg_slot(2), L.ink2_ctrl_down[2], L.ink2_ctrl_forget[2], L.ink2_ctrl_gain[2]);
                qin = build_norm(q_g, L.attn_norm, nullptr, LLM_NORM_RMS, v_idx);
                kin = build_norm(k_g, L.attn_norm, nullptr, LLM_NORM_RMS, v_idx);
                vin = build_norm(v_g, L.attn_norm, nullptr, LLM_NORM_RMS, v_idx);
            } else {
                qin = build_norm(ares, L.attn_norm, nullptr, LLM_NORM_RMS, v_idx);
                kin = qin;
                vin = qin;
            }
        } else {
            ggml_tensor * ares3 = emit_block_attn_res(cur_h, L.ink2_attn_res_query, blks, n_blk, 3, 1);
            ggml_tensor * aq = ggml_view_2d(ctx0, ares3, n_embd, n_tokens, ares3->nb[1], 0 * ares3->nb[2]);
            ggml_tensor * ak = ggml_view_2d(ctx0, ares3, n_embd, n_tokens, ares3->nb[1], 1 * ares3->nb[2]);
            ggml_tensor * av = ggml_view_2d(ctx0, ares3, n_embd, n_tokens, ares3->nb[1], 2 * ares3->nb[2]);
            if (msgs_4 != nullptr) {
                aq = emit_loop_msg_gate(aq, msg_slot(0), L.ink2_ctrl_down[0], L.ink2_ctrl_forget[0], L.ink2_ctrl_gain[0]);
                ak = emit_loop_msg_gate(ak, msg_slot(1), L.ink2_ctrl_down[1], L.ink2_ctrl_forget[1], L.ink2_ctrl_gain[1]);
                av = emit_loop_msg_gate(av, msg_slot(2), L.ink2_ctrl_down[2], L.ink2_ctrl_forget[2], L.ink2_ctrl_gain[2]);
            }
            qin = build_norm(aq, L.ink2_attn_q_in_norm, nullptr, LLM_NORM_RMS, v_idx);
            kin = build_norm(ak, L.ink2_attn_k_in_norm, nullptr, LLM_NORM_RMS, v_idx);
            vin = build_norm(av, L.ink2_attn_v_in_norm, nullptr, LLM_NORM_RMS, v_idx);
        }

        // 3. Q / K / V / Gate / Rel / Sink Projections (`EXT3` plugin + `BF16` small linears):
        ggml_tensor * q_raw = build_lora_mm(L.wq,        qin);
        ggml_tensor * k_raw = build_lora_mm(L.wk,        kin);
        ggml_tensor * v_raw = build_lora_mm(L.wv,        vin);
        ggml_tensor * g_raw = build_lora_mm(L.wqkv_gate, qin);
        ggml_tensor * r_raw = build_lora_mm(L.ink2_attn_rel,  qin);
        ggml_tensor * sinks = build_lora_mm(L.ink2_attn_sink, qin);

        // 4. K & V ShortConvolution (`static` on local visits, `dynamic` on global visits):
        ggml_tensor * k_sc = nullptr;
        ggml_tensor * v_sc = nullptr;
        if (is_local) {
            k_sc = emit_sconv_static(k_raw, st_k, L.ink2_attn_k_sconv, 1024);
            v_sc = emit_sconv_static(v_raw, st_v, L.ink2_attn_v_sconv, 1024);
        } else {
            ggml_tensor * dw_k = build_lora_mm(L.ink2_attn_k_wproj, kin);
            ggml_tensor * dw_v = build_lora_mm(L.ink2_attn_v_wproj, vin);
            k_sc = emit_sconv_dynamic(k_raw, st_k, L.ink2_attn_k_static_w, dw_k);
            v_sc = emit_sconv_dynamic(v_raw, st_v, L.ink2_attn_v_static_w, dw_v);
        }

        // 5. K L2-Clip & V Head-RMSNorm -> Store into `llama_kv_cache_iswa` -> Relative+Sink GQA-3 Attention:
        ggml_tensor * k_cur = emit_prep_kv(k_sc, 0);
        ggml_tensor * v_cur = emit_prep_kv(v_sc, 1);
        ggml_build_forward_expand(gf, q_raw);
        ggml_build_forward_expand(gf, k_cur);
        ggml_build_forward_expand(gf, v_cur);

        const auto * mctx_cur = is_local ? mctx_iswa->get_swa() : mctx_iswa->get_base();
        ggml_tensor * k_idxs  = is_local ? inp_attn->get_k_idxs_swa() : inp_attn->get_k_idxs();
        ggml_tensor * v_idxs  = is_local ? inp_attn->get_v_idxs_swa() : inp_attn->get_v_idxs();
        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, v_idx));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, v_idx));

        ggml_tensor * k_cache = mctx_cur->get_k(ctx0, v_idx);
        ggml_tensor * v_cache = mctx_cur->get_v(ctx0, v_idx);
        ggml_tensor * pos_pk  = is_local ? t_pos_swa : t_pos_base;

        ggml_tensor * attn_args[7] = {
            ggml_cont(ctx0, q_raw), k_cache, v_cache,
            ggml_cont(ctx0, r_raw), ggml_cont(ctx0, sinks), ggml_cont(ctx0, g_raw), pos_pk
        };
        ggml_tensor * attn_gated = ggml_custom_4d(
            ctx0, GGML_TYPE_F32, 3072, n_tokens, 1, 1,
            attn_args, 7, inkling2_model_detail::cpu_op_attn_rel_sink, GGML_N_TASKS_MAX,
            const_cast<ggml_tensor **>(&L.ink2_attn_q_clip));
        inkling2_model_detail::tag_ink2_op(attn_gated, GGML_CUDA_INK2_OP_ATTN_REL_SINK);
        attn_gated->op_params[10] = is_local ? 1 : 0;

        ggml_tensor * wo_out   = build_lora_mm(L.wo, attn_gated);
        ggml_tensor * attn_out = emit_sconv_static(wo_out, st_a, L.ink2_attn_sconv, n_embd);

        // 6. Block Boundary / Residual Update (`h_mid`):
        ggml_tensor * h_mid = nullptr;
        if (is_bound) {
            blks[n_blk] = cur_h;
            h_mid       = attn_out;
            n_blk      += 1;
        } else {
            h_mid = ggml_add(ctx0, cur_h, attn_out);
        }

        // 7. Pre-MLP BlockAttentionResidual + Optional LoopMessageGate + Norm:
        ggml_tensor * mlp_q   = ggml_view_1d(ctx0, model.ink2_mlp_res_queries, n_embd, (size_t) phys_li * model.ink2_mlp_res_queries->nb[1]);
        ggml_tensor * mlp_res = ggml_reshape_2d(ctx0, emit_block_attn_res(h_mid, mlp_q, blks, n_blk, 1, 0), n_embd, n_tokens);
        if (msgs_4 != nullptr) {
            mlp_res = emit_loop_msg_gate(mlp_res, msg_slot(3), L.ink2_ctrl_down[3], L.ink2_ctrl_forget[3], L.ink2_ctrl_gain[3]);
        }
        ggml_tensor * mlp_in = build_norm(mlp_res, L.ffn_norm, nullptr, LLM_NORM_RMS, v_idx);

        // 8. Routed MoE (`EXT0` plugin + Loop-Aware `L17 -> L06` Streaming Prefetch!):
        ggml_tensor * gate_next  = nullptr;
        ggml_tensor * probs_next = nullptr;
        const bool is_loop_wrap  = (mstream != nullptr && pass_idx >= 0 && pass_idx < n_passes - 1 && phys_li == loop_end - 1 && next_phys_li == loop_beg);
        if (mstream != nullptr && next_phys_li >= 0) {
            const int v_next = inkling2_model_detail::phys_layer_to_first_visit(next_phys_li, loop_beg, loop_end, n_passes);
            gate_next  = model.layers[v_next].ffn_gate_inp;
            probs_next = model.layers[v_next].ffn_exp_probs_b;
            if (is_loop_wrap) {
                mstream->cur_pred_il_override = next_phys_li;
                mstream->cur_loop_pass        = pass_idx;
            } else {
                mstream->cur_pred_il_override = -2;
            }
        }
        ggml_tensor * moe_raw = build_moe_ffn(
            mlp_in,
            L.ffn_gate_inp,
            L.ffn_up_exps,
            L.ffn_gate_exps,
            L.ffn_down_exps,
            L.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SITU,
            hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            phys_li,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            gate_next, probs_next);
        if (mstream != nullptr) {
            mstream->cur_pred_il_override = -2;
        }

        ggml_tensor * mlp_out = emit_sconv_static(moe_raw, st_m, L.ink2_ffn_sconv, n_embd);
        cur_h = ggml_add(ctx0, h_mid, mlp_out);

        // 9. Commit 4 ShortConvolution States + Rollback Snapshots (`slot = 0..n_wr-1`):
        auto append_hist = [&](ggml_tensor * st_in, ggml_tensor * x_2d, int64_t C) -> ggml_tensor * {
            ggml_tensor * x_bf = ggml_cast(ctx0, ggml_cast(ctx0, ggml_cont(ctx0, x_2d), GGML_TYPE_BF16), GGML_TYPE_F32);
            ggml_tensor * x_3d = ggml_reshape_3d(ctx0, x_bf, C, n_seq_t, n_seqs);
            ggml_tensor * x_tr = ggml_cont(ctx0, ggml_transpose(ctx0, x_3d)); // [n_seq_t, C, n_seqs]
            return ggml_concat(ctx0, st_in, x_tr, 0); // [3 + n_seq_t, C, n_seqs]
        };
        ggml_tensor * hk   = append_hist(st_k, k_raw,  1024);
        ggml_tensor * hv   = append_hist(st_v, v_raw,  1024);
        ggml_tensor * ha   = append_hist(st_a, wo_out, n_embd);
        ggml_tensor * hm   = append_hist(st_m, moe_raw, n_embd);
        ggml_tensor * h_kv = ggml_concat(ctx0, hk, hv, 1);
        ggml_tensor * h_am = ggml_concat(ctx0, ha, hm, 1);
        ggml_tensor * hall = ggml_concat(ctx0, h_kv, h_am, 1); // [3 + n_seq_t, 6656, n_seqs]

        const size_t row_sz = ggml_row_size(conv_state->type, d_conv * c_tot);
        for (int64_t slot = 0; slot < n_wr; ++slot) {
            ggml_tensor * snap = ggml_view_3d(
                ctx0, hall, d_conv, c_tot, n_seqs, hall->nb[1], hall->nb[2],
                (size_t)(hall->ne[0] - d_conv - slot) * ggml_element_size(hall));
            ggml_tensor * dst_view = ggml_view_2d(
                ctx0, conv_state, d_conv * c_tot, n_seqs, conv_state->nb[1],
                ((size_t) slot * rs_sz + rs_hd) * row_sz);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, snap, dst_view));
        }
    };

    // Execute 48-Visit Schedule (`0..5`, `3 x 6..17`, `18..23`):
    for (int v = 0; v < loop_beg; ++v) {
        const int next_li = (v + 1 < loop_beg) ? (v + 1) : ((loop_len > 0) ? loop_beg : loop_end);
        run_visit_graph(v, v, -1, next_li, nullptr);
    }

    if (loop_len > 0) {
        ggml_tensor * pre_h    = cur_h;
        ggml_tensor * pre_blk0 = blks[0];
        const int     pre_nblk = n_blk;

        ggml_tensor * prv_blk0 = nullptr;
        ggml_tensor * prev_h   = nullptr;

        for (int p = 0; p < n_passes; ++p) {
            if (p > 0) {
                prv_blk0 = blks[std::max(0, n_blk - 1)];
                prev_h   = cur_h;
                cur_h    = pre_h;
                blks[0]  = pre_blk0;
                n_blk    = pre_nblk;
            }
            for (int j = 0; j < loop_len; ++j) {
                const int v_idx   = loop_beg + p * loop_len + j;
                const int phys_li = loop_beg + j;
                int next_li       = -1;
                if (j + 1 < loop_len) {
                    next_li = phys_li + 1;
                } else if (p + 1 < n_passes) {
                    next_li = loop_beg; // Loop wrap L17 -> L06!
                } else if (loop_end < (int) hparams.ink2_phys_block_count) {
                    next_li = loop_end;
                }

                ggml_tensor * msgs = nullptr;
                if (p > 0) {
                    ggml_tensor * mq_j = ggml_view_2d(
                        ctx0, model.ink2_loop_msg_queries, n_embd, 4,
                        model.ink2_loop_msg_queries->nb[1], (size_t) j * model.ink2_loop_msg_queries->nb[2]);
                    ggml_tensor * pblks[1] = { prv_blk0 };
                    msgs = emit_block_attn_res(prev_h, mq_j, pblks, 1, 4, 1);
                }
                run_visit_graph(v_idx, phys_li, p, next_li, msgs);
            }
        }
    }

    const int coda_v0 = loop_beg + n_passes * loop_len;
    const int n_phys  = (int) hparams.ink2_phys_block_count;
    for (int li = loop_end; li < n_phys; ++li) {
        const int v_idx   = coda_v0 + (li - loop_end);
        const int next_li = (li + 1 < n_phys) ? (li + 1) : -1;
        run_visit_graph(v_idx, li, -1, next_li, nullptr);
    }

    // Final BlockAttentionResidual + Output Norm + Vocab Softcap/Mask:
    ggml_tensor * h_pre_fin = ggml_reshape_2d(
        ctx0, emit_block_attn_res(cur_h, model.ink2_output_res_query, blks, n_blk, 1, 1), n_embd, n_tokens);
    ggml_tensor * h_fin = build_norm(h_pre_fin, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    h_fin = maybe_stash_mtp_tap(h_fin, 3);

    if (inp_out_ids) {
        h_fin = ggml_get_rows(ctx0, h_fin, inp_out_ids);
    }
    res->t_embd = h_fin;

    if (!cparams.embeddings) {
        ggml_tensor * raw_logits = build_lora_mm(model.output, h_fin);
        ggml_tensor * sm_args[1] = { ggml_cont(ctx0, raw_logits) };
        ggml_tensor * cur = ggml_custom_4d(
            ctx0, GGML_TYPE_F32, n_vocab, raw_logits->ne[1], 1, 1,
            sm_args, 1, inkling2_model_detail::cpu_op_softcap_mask_head, GGML_N_TASKS_MAX, nullptr);
        inkling2_model_detail::tag_ink2_op(cur, GGML_CUDA_INK2_OP_SOFTCAP_MASK_HEAD);
        cur->op_params[10] = (int32_t) hparams.ink2_unpadded_vocab_size;
        const float cap = hparams.f_final_logit_softcapping > 0.0f ? hparams.f_final_logit_softcapping : 30.0f;
        std::memcpy(&cur->op_params[11], &cap, sizeof(float));
        ggml_set_name(cur, "result_output");

        res->t_logits = cur;
        ggml_build_forward_expand(gf, cur);
    } else {
        ggml_build_forward_expand(gf, h_fin);
    }
}

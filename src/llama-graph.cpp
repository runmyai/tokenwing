#include "llama-graph.h"

#include <algorithm>
#include <atomic>

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-stream.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-sampler.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <tuple>
#include <sstream>
#include <string>
#include <unordered_set>

// dedup helpers

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    // flash attention requires an f16 mask
    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

// impl

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd && embd && !ubatch->is_mixed()) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }

    if (ubatch->is_mixed() && embd) {
        GGML_ASSERT(mixed_tokens && mixed_slots && mixed_embd && "mixed token/embd ubatch is not supported here");

        std::vector<int32_t> ids;
        std::vector<int64_t> slots;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            if (!ubatch->type[i]) {
                ids.push_back(ubatch->token[i]);
                slots.push_back(i);
            }
        }
        GGML_ASSERT((int64_t) ids.size() == mixed_tokens->ne[0]);
        GGML_ASSERT(n_embd == mixed_embd->ne[0]);

        ggml_backend_tensor_set(mixed_tokens, ids.data(),    0, ggml_nbytes(mixed_tokens));
        ggml_backend_tensor_set(mixed_slots,  slots.data(),  0, ggml_nbytes(mixed_slots));
        ggml_backend_tensor_set(mixed_embd,   ubatch->embd,  0, ggml_nbytes(mixed_embd));
    }

    if (scale_rows) {
        const int64_t n_tokens = ubatch->n_tokens;

        std::vector<float> data(n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const bool is_embd = !ubatch->token || (ubatch->is_mixed() && ubatch->type[i]);
            data[i] = is_embd ? 1.0f : scale_tok;
        }
        ggml_backend_tensor_set(scale_rows, data.data(), 0, ggml_nbytes(scale_rows));
    }
}

// number of token rows of the mixed path, a non-mixed ubatch is sized for the worst case
static int64_t llm_graph_n_tok_rows(const llama_ubatch & ubatch) {
    if (!ubatch.is_mixed()) {
        return ubatch.n_tokens;
    }
    int64_t n = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        n += !ubatch.type[i];
    }
    return n;
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);
    res &= (!mixed_tokens) || mixed_tokens->ne[0] == llm_graph_n_tok_rows(params.ubatch);
    res &= (!mixed_embd)   || mixed_embd->ne[1]   == params.ubatch.n_tokens;
    res &= (!scale_rows) || scale_rows->ne[1] == params.ubatch.n_tokens;

    return res;
}

void llm_graph_input_embd_h::set_input(const llama_ubatch * ubatch) {
    ASSERT_EMBD_OR_TOKEN(*ubatch);

    const int64_t n_tokens = ubatch->n_tokens;

    if (ubatch->token) {
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    } else {
        // note: mtmd embedding input goes through here
        GGML_ASSERT(ubatch->embd);
        GGML_ASSERT(n_embd == embd->ne[0]);

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }

    GGML_ASSERT(ubatch->embd_state && "this graph requires a state embedding, see llama_batch_ext_set_embd_state()");
    GGML_ASSERT(n_embd_state == h->ne[0]);

    ggml_backend_tensor_set(h, ubatch->embd_state, 0, n_tokens*n_embd_state*ggml_element_size(h));
}

bool llm_graph_input_embd_h::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token)      || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)       || (embd   && embd->ne[1]   == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd_state) || (h      && h->ne[1]      == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && cparams.causal_attn)
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    // the state read in place: still one state, the cell's own
    res &= !s_inplace || (mctx->get_n_rs() == 1 && mctx->s_copy(0) == (int32_t) mctx->get_head());

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

template <typename T>
static void print_mask(const T * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = llama_cast<float>(data[i * n_kv + j]);
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](auto * data, int64_t ne, int n_swa, llama_swa_type swa_type) {
        using T = std::remove_reference_t<decltype(*data)>;
        std::fill(data, data + ne, llama_cast<T>(-INFINITY));

        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (cparams.causal_attn && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = llama_cast<T>(hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f);
            }
        }

        if (debug) {
            print_mask(data, n_tokens, n_kv, n_swa, swa_type);
        }
    };

    GGML_ASSERT(self_kq_mask);
    GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
    if (self_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    } else {
        fill_mask((float       *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
        if (self_kq_mask_swa->type == GGML_TYPE_F16) {
            fill_mask((ggml_fp16_t *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        } else {
            fill_mask((float       *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        }
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    // the mask is left unallocated when the graph only stores K/V without attending
    // (e.g. DFlash's KV-injection pass)
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->set_input_v_rot(self_v_rot);
    }
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    return can_reuse_impl(params);
}

bool llm_graph_input_attn_k::can_reuse_impl(const llm_graph_params & params) {
    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

llm_graph_input_attn_kv_msa::llm_graph_input_attn_kv_msa(
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        const llama_kv_cache_msa_context * mctx) :
    llm_graph_input_attn_kv(hparams, cparams, mctx->get_base()),
    mctx_msa(mctx) {
}

void llm_graph_input_attn_kv_msa::set_input(const llama_ubatch * ubatch) {
    llm_graph_input_attn_kv::set_input(ubatch);

    if (self_k_idxs_idx) {
        mctx_msa->get_idx()->set_input_k_idxs(self_k_idxs_idx, ubatch);
    }
}

bool llm_graph_input_attn_kv_msa::can_reuse(const llm_graph_params & params) {
    mctx_msa = static_cast<const llama_kv_cache_msa_context *>(params.mctx);

    // the parent class operates on the base cache context
    this->mctx = mctx_msa->get_base();

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    if (self_k_idxs_idx) {
        res &= self_k_idxs_idx->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(self_kq_mask, this->mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    // leave unallocated when the top-k layers use the indexer mask
    if (self_kq_mask_mla->buffer) {
        mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);
    }

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);

    // left unallocated when the indexer does not use the rotation
    if (self_k_rot_lid && self_k_rot_lid->buffer) {
        mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
    }
}

bool llm_graph_input_attn_k_dsa::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsa_context *>(params.mctx);

    return can_reuse_impl(params);
}

bool llm_graph_input_attn_k_dsa::can_reuse_impl(const llm_graph_params & params) {
    bool res = true;

    res &= self_k_idxs_mla->ne[0] == params.ubatch.n_tokens;
    res &= self_k_idxs_lid->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask_mla, mctx->get_mla(), params.ubatch, params.cparams);
    res &= can_reuse_kq_mask(self_kq_mask_lid, mctx->get_lid(), params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa_iswa::set_input(const llama_ubatch * ubatch) {
    inp_dsa->set_input(ubatch);
    inp_swa->set_input(ubatch);
}

bool llm_graph_input_attn_k_dsa_iswa::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsa_iswa_context *>(params.mctx);

    inp_dsa->mctx = mctx->get_dsa();
    inp_swa->mctx = mctx->get_swa();

    bool res = true;

    res &= inp_dsa->can_reuse_impl(params);
    res &= inp_swa->can_reuse_impl(params);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
        if (self_v_idxs) {
            mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);
        }
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
        if (self_v_idxs_swa) {
            mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);
        }
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot && self_v_rot->buffer) {
        mctx->get_base()->set_input_v_rot(self_v_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }

    if (self_v_rot_swa && self_v_rot_swa->buffer) {
        mctx->get_swa()->set_input_v_rot(self_v_rot_swa);
    }
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

void llm_graph_input_attn_k_iswa::set_input(const llama_ubatch * ubatch) {
    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot && self_k_rot->buffer) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_k_rot_swa && self_k_rot_swa->buffer) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }
}

bool llm_graph_input_attn_k_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

static void dsv4_set_i64(ggml_tensor * dst, const std::vector<int64_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_i32(ggml_tensor * dst, const std::vector<int32_t> & src) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());
    ggml_backend_tensor_set(dst, src.data(), 0, src.size()*ggml_element_size(dst));
}

static void dsv4_set_kq_mask(
        ggml_tensor * dst,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (!dst || !dst->buffer) {
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);
    GGML_ASSERT(dst->ne[0] == plan.n_kv);
    GGML_ASSERT(dst->ne[1] == (int64_t) n_tokens/n_stream);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == n_stream);
    GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) n_tokens);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    if (dst->type == GGML_TYPE_F32) {
        float * data = (float *) dst->data;

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? 0.0f : -INFINITY;
            }
        }
    } else if (dst->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *) dst->data;
        const ggml_fp16_t fp16_ninf = llama_cast<ggml_fp16_t>(-INFINITY);
        const ggml_fp16_t fp16_zero = llama_cast<ggml_fp16_t>(0.0f);

        for (int64_t i = 0; i < (int64_t) n_tokens; ++i) {
            const int32_t n_visible = plan.n_visible[i];

            for (int64_t j = 0; j < dst->ne[0]; ++j) {
                data[i*dst->ne[0] + j] = j < n_visible ? fp16_zero : fp16_ninf;
            }
        }
    }
}

static ggml_tensor * dsv4_build_raw_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens%n_stream == 0);

    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static bool dsv4_can_reuse_raw_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_dsv4_raw_context * mctx,
        const llama_ubatch & ubatch,
        int64_t n_stream) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;

    GGML_ASSERT(n_stream > 0);

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static bool dsv4_compress_debug() {
    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    return debug;
}

static void dsv4_set_comp_inputs(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        bool debug,
        uint32_t n_tokens,
        int64_t n_stream) {
    dsv4_set_i32(inp.state_pos, plan.state_pos);
    dsv4_set_i32(inp.state_persist_src_idxs, plan.state_persist_src_idxs);
    dsv4_set_i32(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs);
    dsv4_set_i32(inp.state_restore_src_idxs, plan.state_restore_src_idxs);
    dsv4_set_i32(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs);
    dsv4_set_i32(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs);
    dsv4_set_i32(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs);
    dsv4_set_i32(inp.state_read_idxs, plan.state_read_idxs);
    dsv4_set_i64(inp.state_write_idxs, plan.state_write_idxs);
    dsv4_set_i32(inp.state_write_pos, plan.state_write_pos);
    dsv4_set_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    if (debug || dsv4_compress_debug()) {
        LLAMA_LOG_INFO("%s: %s n_tokens=%u, n_stream=%d, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, name, n_tokens, (int) n_stream,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
    }
}

static bool dsv4_can_reuse_tensor_1d(ggml_tensor * t, int64_t ne0) {
    return (t == nullptr && ne0 == 0) || (t != nullptr && t->ne[0] == ne0);
}

static bool dsv4_can_reuse_kq_mask(
        ggml_tensor * t,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    if (plan.n_kv == 0) {
        return t == nullptr;
    }

    GGML_ASSERT(n_stream > 0);

    return t != nullptr &&
           t->ne[0] == plan.n_kv &&
           t->ne[1] == (int64_t) n_tokens/n_stream &&
           t->ne[2] == 1 &&
           t->ne[3] == n_stream;
}

static bool dsv4_can_reuse_comp_input(
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        uint32_t n_tokens,
        int64_t n_stream) {
    bool res = true;
    res &= dsv4_can_reuse_tensor_1d(inp.state_pos, plan.state_pos.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_src_idxs, plan.state_persist_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_persist_dst_idxs, plan.state_persist_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_src_idxs, plan.state_restore_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_restore_dst_idxs, plan.state_restore_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_src_idxs, plan.state_snapshot_src_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_snapshot_dst_idxs, plan.state_snapshot_dst_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_read_idxs, plan.state_read_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_idxs, plan.state_write_idxs.size());
    res &= dsv4_can_reuse_tensor_1d(inp.state_write_pos, plan.state_write_pos.size());
    res &= dsv4_can_reuse_kq_mask(inp.kq_mask, plan, n_tokens, n_stream);

    return res;
}

static ggml_tensor * dsv4_build_input_1d(
        ggml_context * ctx,
        ggml_type type,
        int64_t ne0,
        const std::string & name) {
    if (ne0 == 0) {
        return nullptr;
    }

    ggml_tensor * res = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_input(res);
    ggml_set_name(res, name.c_str());

    return res;
}

static void dsv4_build_comp_inputs(
        ggml_context * ctx,
        llm_graph_input_dsv4::comp_input & inp,
        const llama_kv_cache_dsv4_context::comp_plan & plan,
        const char * name,
        const llama_cparams & cparams,
        int64_t n_stream) {
    inp.state_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_pos.size(), std::string("dsv4_") + name + "_state_pos");
    inp.state_persist_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_src_idxs.size(), std::string("dsv4_") + name + "_state_persist_src_idxs");
    inp.state_persist_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_persist_dst_idxs.size(), std::string("dsv4_") + name + "_state_persist_dst_idxs");
    inp.state_restore_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_src_idxs.size(), std::string("dsv4_") + name + "_state_restore_src_idxs");
    inp.state_restore_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_restore_dst_idxs.size(), std::string("dsv4_") + name + "_state_restore_dst_idxs");
    inp.state_snapshot_src_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_src_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_src_idxs");
    inp.state_snapshot_dst_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_snapshot_dst_idxs.size(), std::string("dsv4_") + name + "_state_snapshot_dst_idxs");
    inp.state_read_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_read_idxs.size(), std::string("dsv4_") + name + "_state_read_idxs");
    inp.state_write_idxs = dsv4_build_input_1d(ctx, GGML_TYPE_I64, plan.state_write_idxs.size(), std::string("dsv4_") + name + "_state_write_idxs");
    inp.state_write_pos = dsv4_build_input_1d(ctx, GGML_TYPE_I32, plan.state_write_pos.size(), std::string("dsv4_") + name + "_state_write_pos");

    if (plan.n_kv > 0) {
        const int64_t n_tokens = (int64_t) plan.n_visible.size();

        GGML_ASSERT(n_stream > 0);
        GGML_ASSERT(n_tokens%n_stream == 0);

        inp.kq_mask = ggml_new_tensor_4d(ctx, (strcmp(name, "lid") != 0 && cparams.flash_attn) || (strcmp(name, "lid") == 0 && cparams.fused_lid) ? GGML_TYPE_F16 : GGML_TYPE_F32, plan.n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp.kq_mask);
        ggml_set_name(inp.kq_mask, (std::string("dsv4_") + name + "_kq_mask").c_str());
    }
}

void llm_graph_input_dsv4_raw::set_input(const llama_ubatch * ubatch) {
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs);
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (self_k_rot) {
        mctx->set_input_k_rot(self_k_rot);
    }
}

void llm_graph_input_dsv4::set_input(const llama_ubatch * ubatch) {
    const auto & plan_csa = mctx->get_csa_plan(*ubatch);
    const auto & plan_hca = mctx->get_hca_plan(*ubatch);
    const auto & plan_lid = mctx->get_lid_plan(*ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    inp_raw->mctx = mctx->get_raw();
    inp_raw->set_input(ubatch);

    dsv4_set_comp_inputs(inp_csa, plan_csa, "csa", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_hca, plan_hca, "hca", debug > 0, ubatch->n_tokens, n_stream);
    dsv4_set_comp_inputs(inp_lid, plan_lid, "lid", debug > 0, ubatch->n_tokens, n_stream);

    if (inp_csa.k_rot && inp_csa.k_rot->buffer) {
        mctx->get_csa()->set_input_k_rot(inp_csa.k_rot);
    }

    if (inp_hca.k_rot && inp_hca.k_rot->buffer) {
        mctx->get_hca()->set_input_k_rot(inp_hca.k_rot);
    }

    if (inp_lid.k_rot && inp_lid.k_rot->buffer) {
        mctx->get_lid()->set_input_k_rot(inp_lid.k_rot);
    }
}

bool llm_graph_input_dsv4::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    this->mctx = mctx;
    inp_raw->mctx = mctx->get_raw();

    bool res = true;

    const auto & plan_csa = mctx->get_csa_plan(params.ubatch);
    const auto & plan_hca = mctx->get_hca_plan(params.ubatch);
    const auto & plan_lid = mctx->get_lid_plan(params.ubatch);
    const int64_t n_stream = plan_csa.n_stream;

    const auto * raw_ctx = mctx->get_raw();
    inp_raw->mctx = raw_ctx;

    if (inp_raw->self_k_idxs && inp_raw->self_k_idxs->buffer) {
        res &= inp_raw->self_k_idxs->ne[0] == raw_ctx->get_n_write();
    }
    if (inp_raw->self_kq_mask && inp_raw->self_kq_mask->buffer) {
        res &= dsv4_can_reuse_raw_kq_mask(inp_raw->self_kq_mask, raw_ctx, params.ubatch, n_stream);
    }

    res &= dsv4_can_reuse_comp_input(inp_csa, plan_csa, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_hca, plan_hca, params.ubatch.n_tokens, n_stream);
    res &= dsv4_can_reuse_comp_input(inp_lid, plan_lid, params.ubatch.n_tokens, n_stream);

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    const auto fill_mask = [&](auto * data) {
        using T = std::remove_reference_t<decltype(*data)>;
        for (int i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first");
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[i*n_enc + j] = llama_cast<T>(f);
            }
        }
    };

    if (cross_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) cross_kq_mask->data);
    } else {
        fill_mask((float *) cross_kq_mask->data);
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    // An input no node of this graph reads has no buffer (a layer-major group graph without an attention or a
    // recurrent layer) and is not filled
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    }
    if (inp_attn->self_v_idxs && inp_attn->self_v_idxs->buffer) {
        mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }

    // leave the mask unallocated if all attention layers have their own (e.g. qwen4exp QSA)
    if (inp_attn->self_kq_mask && inp_attn->self_kq_mask->buffer) {
        mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot && inp_attn->self_k_rot->buffer) {
        mctx->get_attn()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot && inp_attn->self_v_rot->buffer) {
        mctx->get_attn()->set_input_v_rot(inp_attn->self_v_rot);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy && inp_rs->s_copy->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    // the state read in place: still one state, the cell's own
    res &= !inp_rs->s_inplace || (mctx->get_recr()->get_n_rs() == 1 && mctx->get_recr()->s_copy(0) == (int32_t) mctx->get_recr()->get_head());

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    // An input no node of this graph reads has no buffer (a layer-major group graph without an attention or a
    // recurrent layer) and is not filled - the KQ mask is n_kv x n_tokens, ~12 ms per fill at 8k
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    }

    if (inp_attn->self_kq_mask && inp_attn->self_kq_mask->buffer) {
        mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy && inp_rs->s_copy->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    // the state read in place: still one state, the cell's own
    res &= !inp_rs->s_inplace || (mctx->get_recr()->get_n_rs() == 1 && mctx->get_recr()->s_copy(0) == (int32_t) mctx->get_recr()->get_head());

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }

    if (inp_attn->self_kq_mask && inp_attn->self_kq_mask->buffer) {
        attn_ctx->get_base()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);
    }

    if (inp_attn->self_kq_mask_swa && inp_attn->self_kq_mask_swa->buffer) {
        attn_ctx->get_swa()->set_input_kq_mask(inp_attn->self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot) {
        attn_ctx->get_base()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        attn_ctx->get_base()->set_input_v_rot(inp_attn->self_v_rot);
    }

    if (inp_attn->self_k_rot_swa) {
        attn_ctx->get_swa()->set_input_k_rot(inp_attn->self_k_rot_swa);
    }

    if (inp_attn->self_v_rot_swa) {
        attn_ctx->get_swa()->set_input_v_rot(inp_attn->self_v_rot_swa);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, attn_ctx->get_base(), params.ubatch, params.cparams);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    // the state read in place: still one state, the cell's own
    res &= !inp_rs->s_inplace || (mctx->get_recr()->get_n_rs() == 1 && mctx->get_recr()->s_copy(0) == (int32_t) mctx->get_recr()->get_head());

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_h_nextn     = nullptr;
    t_lm_hidden   = nullptr;

    t_layer_inp.resize(LLAMA_MAX_LAYERS + 1);
    std::fill(t_layer_inp.begin(), t_layer_inp.end(), nullptr);

    t_sampled.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();

    params = {};

    inputs.clear();
    fused_nodes.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

void llm_graph_result::set_outputs(const llm_graph_params & params) {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    if (t_h_nextn != nullptr) {
        ggml_set_output(t_h_nextn);
    }
    {
        const auto & embeddings_layer_inp = params.cparams.embeddings_layer_inp;
        for (size_t il = 0; il < embeddings_layer_inp.size(); ++il) {
            if (embeddings_layer_inp[il]) {
                GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null");
                ggml_set_output(t_layer_inp[il]);
            }
        }
    }
    for (auto * tensor : t_sampled) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_probs) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_sampled_logits) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
    for (auto * tensor : t_candidates) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::add_fused_node(llm_graph_fused_node result) {
    fused_nodes.push_back(result);
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer()),
    n_layer_nextn    (hparams.n_layer_nextn),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used()),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    mstream          (params.mstream),
    prec_policy      (params.prec_policy),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
        lm_phase = params.lm_phase;
        lm_hid   = params.lm_hid;
        lm_x     = params.lm_x;
        lm_moe   = params.lm_moe;
        lm_ids   = params.lm_ids;
        lm_w     = params.lm_w;
        lm_row0  = params.lm_row0;
        lm_chunk = params.lm_chunk;
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}

ggml_backend_t llm_graph_context::moe_backend_of(ggml_backend_dev_t dev) const {
    for (int b = 0; dev != nullptr && sched != nullptr && b < ggml_backend_sched_get_n_backends(sched); b++) {
        ggml_backend_t be = ggml_backend_sched_get_backend(sched, b);
        if (ggml_backend_get_device(be) == dev) {
            return be;
        }
    }
    return nullptr;
}

void llm_graph_context::moe_pin(ggml_tensor * t, ggml_backend_dev_t dev) const {
    if (ggml_backend_t be = moe_backend_of(dev); be != nullptr && t != nullptr && ggml_backend_supports_op(be, t)) {
        ggml_backend_sched_set_tensor_backend(sched, t, be);
    }
}

void llm_graph_context::cb_tag(ggml_tensor * cur, const char * name, int il, const char * tag) const {
    if (!cb_func) {
        return;
    }

    if (!tag) {
        cb_func(ubatch, cur, name, il);
        return;
    }

    char buf[128];
    snprintf(buf, sizeof(buf), "%s_%s", name, tag);
    cb_func(ubatch, cur, buf, il);
}



ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    if (prec_policy) {
        prec_policy->apply(res);
    }

    if (w->type == GGML_TYPE_NVFP4) {
        ggml_prec_set_acc(res, GGML_PREC_BF16);
    }

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

llm_graph_amax_src1 llm_graph_context::build_amax_src1_f16(
          ggml_tensor * cur) const {
    llm_graph_amax_src1 result = {
        /*.res     =*/ cur,
        /*.scale_s =*/ nullptr,
    };

    const int64_t n_embd = cur->ne[0];

    // amax = max|cur| along the rows for each [expert, token] -> [1, n_expert, n_tokens]
    ggml_tensor * ab = ggml_abs(ctx0, cur);
    ggml_tensor * amax = ggml_pool_1d(ctx0, ab, GGML_OP_POOL_MAX, n_embd, n_embd, 0);

    // ref: https://github.com/ggml-org/llama.cpp/pull/26223
    constexpr float k_log2_e = 1.4426950408889634f; // 1 / ln(2)
    constexpr float k_ln_2   = 0.6931471805599453f;

    ggml_tensor * e;
    // note: this can technically be:
    // e = ggml_scale(ctx0, amax, 1.0f/32768.0f);
    // but we leave x32 headroom for the F16 sums of 32 values in Q8 activation blocks
    e = ggml_scale(ctx0, amax, 1.0f/1024.0f);
    e = ggml_log  (ctx0, e);
    e = ggml_scale(ctx0, e, k_log2_e);
    e = ggml_ceil (ctx0, e);
    e = ggml_clamp(ctx0, e, 0.0f, 127.0f);
    e = ggml_scale(ctx0, e, k_ln_2);
    e = ggml_exp  (ctx0, e);

    result = {
        /*.res     =*/ ggml_div(ctx0, cur, e),
        /*.scale_s =*/ e,
    };

    return result;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids,
          ggml_tensor * w_s,
          ggml_tensor * ids_scale) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);

    if (prec_policy) {
        prec_policy->apply(res);
    }

    if (w->type == GGML_TYPE_NVFP4) {
        ggml_prec_set_acc(res, GGML_PREC_BF16);
    }

    if (w_s) {
        // w_s holds one scale per expert, so it is gathered with the original expert ids even when
        // the GEMM ids are remapped pool slots (MoE expert streaming)
        ggml_tensor * ids_s = ids_scale ? ids_scale : ids;

        const int64_t n_expert = w_s->ne[0];
        const int64_t n_tokens = cur->ne[2];
        ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, ids_s);
        res = ggml_mul(ctx0, res, s);
    }
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const {
    return build_qkv(layer, cur,
            n_embd_head, n_head,
            n_embd_head, n_head_kv,
            n_embd_head, n_head_kv,
            il);
}

llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape) const {
    const int64_t n_embd_q = n_embd_head_q * n_head_q;
    const int64_t n_embd_k = n_embd_head_k * n_head_k;

    ggml_tensor * Qcur, * Kcur, * Vcur;

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        } else if (layer.wq_b && layer.wk_b && layer.wv_b) {
            // Fused weights may coexist with separate Q/K/V biases in legacy or custom GGUFs.
            ggml_tensor * qkv_b = ggml_concat(ctx0, ggml_concat(ctx0, layer.wq_b, layer.wk_b, 0), layer.wv_b, 0);
            qkv = ggml_add(ctx0, qkv, qkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(qkv, "wqkv_clamped", il);
        }
        if (reshape) {
            Qcur = ggml_view_3d(ctx0, qkv, n_embd_head_q, n_head_q, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_q), qkv->nb[1], 0);
            Kcur = ggml_view_3d(ctx0, qkv, n_embd_head_k, n_head_k, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_k), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_3d(ctx0, qkv, n_embd_head_v, n_head_v, n_tokens,
                ggml_row_size(qkv->type, n_embd_head_v), qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        } else {
            Qcur = ggml_view_2d(ctx0, qkv, n_embd_q, n_tokens, qkv->nb[1], 0);
            Kcur = ggml_view_2d(ctx0, qkv, n_embd_k, n_tokens, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q));
            Vcur = ggml_view_2d(ctx0, qkv, n_embd_head_v * n_head_v, n_tokens, qkv->nb[1],
                ggml_row_size(qkv->type, n_embd_q + n_embd_k));
        }
        if (!reshape) {
            Qcur = ggml_cont(ctx0, Qcur);
            Kcur = ggml_cont(ctx0, Kcur);
            Vcur = ggml_cont(ctx0, Vcur);
        }
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        if (reshape) {
            cb(Qcur, "Qcur", il);
        }
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            if (reshape) {
                cb(Qcur, "Qcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Qcur, "Qcur_clamped", il);
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        if (reshape) {
            cb(Kcur, "Kcur", il);
        }
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            if (reshape) {
                cb(Kcur, "Kcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Kcur, "Kcur_clamped", il);
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        if (reshape) {
            cb(Vcur, "Vcur", il);
        }
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            if (reshape) {
                cb(Vcur, "Vcur", il);
            }
        }
        if (reshape && hparams.f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Vcur, "Vcur_clamped", il);
        }
        if (reshape) {
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_q, n_head_q, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head_k, n_head_k, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head_v, n_head_v, n_tokens);
        }
    }

    if (reshape) {
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
    }

    return { Qcur, Kcur, Vcur };
}


ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    // NVFP4 support is currently restricted to
    // 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
    // 2) bias absense (*_s would be applied after bias addition, which is incorrect)
    // TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
    auto has_lora = [this](ggml_tensor * w) {
        if (!w) {
            return false;
        }
        for (const auto & lora : *loras) {
            if (lora.first->get_weight(w) != nullptr) {
                return true;
            }
        }
        return false;
    };

    GGML_ASSERT(!up_s   || !up_b   || !up   || up->type   != GGML_TYPE_NVFP4);
    GGML_ASSERT(!gate_s || !gate_b || !gate || gate->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!down_s || !down_b || !down || down->type != GGML_TYPE_NVFP4);
    GGML_ASSERT(!up_s   || !up   || up->type   != GGML_TYPE_NVFP4 || !has_lora(up));
    GGML_ASSERT(!gate_s || !gate || gate->type != GGML_TYPE_NVFP4 || !has_lora(gate));
    GGML_ASSERT(!down_s || !down || down->type != GGML_TYPE_NVFP4 || !has_lora(down));

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                if (il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        if (arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
                            cur = ggml_swiglu_clamp(ctx0, cur, tmp, limit);
                        } else {
                            tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                            cb(tmp, "ffn_up_clamped", il);
                            ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                            cb(gate_act, "ffn_silu", il);
                            gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                            cb(gate_act, "ffn_silu_clamped", il);
                            cur = ggml_mul(ctx0, gate_act, tmp);
                        }
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            if (gate && type_gate == LLM_FFN_PAR) {
                // same alpha/limit constants as gpt-oss
                const float alpha = 1.702f;
                const float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, tmp, alpha, limit);
                cb(cur, "ffn_swiglu_oai", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                GGML_ABORT("LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        case LLM_FFN_SITU:
            GGML_ABORT("not yet supported");
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            ggml_prec_set_acc(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in,
         ggml_tensor * gate_inp_next,
         ggml_tensor * exp_probs_b_next,
         ggml_tensor * pred_in,
         ggml_tensor * gate_inp_next2,   // the router of the layer *two* ahead
         ggml_tensor * exp_probs_b_next2,
         ggml_tensor * gate_inp_next3,   // lead 3: the router of the layer *three* ahead
         ggml_tensor * exp_probs_b_next3,
         const llm_graph_rec_leads * rec // what the recorder needs of the next layers
         ) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s,
        selected_experts_in,
        gate_inp_next,
        exp_probs_b_next,
        pred_in,
        gate_inp_next2,
        exp_probs_b_next2,
        gate_inp_next3,
        exp_probs_b_next3,
        rec
    );
}

// the router of a MoE layer: gate_inp is applied to cur, then the gating of gating_op is applied
ggml_tensor * llm_graph_context::build_moe_probs(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * logits_in,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor ** out_logits,
         const char    * name_tag) const {
    ggml_tensor * logits = logits_in;

    if (logits == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_prec_set_acc(logits, GGML_PREC_F32);
        }
        cb_tag(logits, "ffn_moe_logits", il, name_tag);
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb_tag(logits, "ffn_moe_logits_biased", il, name_tag);
    }

    if (out_logits) {
        *out_logits = logits;   // what llama4 and grovemoe select their experts on
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
            {
                probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb_tag(probs, "ffn_moe_probs", il, name_tag);

    return probs;
}

// Debug (MOE_STREAM_TEST_FUSED_CHECK=1): compares the fused selection (a) with the argsort chain (b) row by row on the host and
// prints the rows that differ, with the selection scores (c = probs + bias) of both id lists. Counts only; no effect on
// the ids the graph uses.
static void fused_check_op(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c,
        int ith, int nth, void * userdata) {
    GGML_UNUSED(dst); GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    static std::atomic<int64_t> n_rows{0}, n_order{0}, n_set{0};
    static std::atomic<int>     n_printed{0};
    const int il = (int) (intptr_t) userdata;
    const int64_t k = a->ne[0];
    for (int64_t r = 0; r < a->ne[1]; r++) {
        const int32_t * ia = (const int32_t *) ((const char *) a->data + r*a->nb[1]);
        const int32_t * ib = (const int32_t *) ((const char *) b->data + r*b->nb[1]);
        const float   * sc = (const float   *) ((const char *) c->data + r*c->nb[1]);
        bool same = true;
        for (int64_t j = 0; j < k; j++) {
            same = same && ia[j] == ib[j];
        }
        n_rows++;
        if (same) {
            continue;
        }
        bool in_range = true;
        for (int64_t j = 0; j < k; j++) {
            in_range = in_range && ia[j] >= 0 && ia[j] < c->ne[0] && ib[j] >= 0 && ib[j] < c->ne[0];
        }
        if (!in_range) {
            fprintf(stderr, "MOE_STREAM_TEST_FUSED_CHECK: layer %d row %lld: an id outside [0, %lld)\n", il, (long long) r, (long long) c->ne[0]);
            continue;
        }
        std::vector<int32_t> sa(ia, ia + k), sb(ib, ib + k);
        std::sort(sa.begin(), sa.end());
        std::sort(sb.begin(), sb.end());
        const bool set_same = sa == sb;
        (set_same ? n_order : n_set)++;
        if (n_printed++ < 40) {
            std::string s = format("MOE_STREAM_TEST_FUSED_CHECK: layer %d row %lld %s:", il, (long long) r, set_same ? "same set, other order" : "OTHER SET");
            for (int64_t j = 0; j < k; j++) { s += format(" %d(%.9g)", ia[j], sc[ia[j]]); }
            s += " | chain:";
            for (int64_t j = 0; j < k; j++) { s += format(" %d(%.9g)", ib[j], sc[ib[j]]); }
            fprintf(stderr, "%s\n", s.c_str());
        }
    }
    static std::atomic<int64_t> n_calls{0};
    if ((++n_calls % 512) == 0) {
        fprintf(stderr, "MOE_STREAM_TEST_FUSED_CHECK: %lld rows compared, %lld same set in another order, %lld other set\n",
                (long long) n_rows.load(), (long long) n_order.load(), (long long) n_set.load());
    }
}

// the expert selection: selection bias, expert groups and top k on the router probabilities
ggml_tensor * llm_graph_context::build_moe_topk(
         ggml_tensor * probs,
         ggml_tensor * logits,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
                 int   il,
         const char    * name_tag) const {
    const int64_t n_tokens = probs->ne[1];

    // --moe-stream-fused-route: the selection bias, the expert groups and the top k as one op. It takes the same
    // probs and bias as the chain below and returns the same experts in the same order (largest first), so the weights
    // that build_moe_ffn gathers from probs with these ids are the same numbers. Not for the archs that select on
    // something else than probs (+ bias).
    // Only for grouped routing: without expert groups llama.cpp's own topk-moe CUDA fusion already does the selection
    // (and the weights) in one kernel, and this op would replace it. Every precondition of ggml_moe_group_topk is checked
    // here, so a model it does not fit builds the chain as before instead of hitting an assert.
    static thread_local bool fused_bypass = false;   // MOE_STREAM_TEST_FUSED_CHECK: build the chain as well
    const int64_t n_groups_hp = hparams.n_expert_groups;
    const int64_t n_used_hp   = hparams.n_group_used;
    const bool fused_ok = !fused_bypass && mstream && mstream->p1.fused_route && n_tokens > 0 &&
            arch != LLM_ARCH_LLAMA4 && arch != LLM_ARCH_GROVEMOE &&
            n_groups_hp > 1 && n_groups_hp <= 64 && n_expert % n_groups_hp == 0 &&
            n_used_hp >= 1 && n_used_hp <= n_groups_hp && 2 <= n_expert / n_groups_hp &&
            n_expert_used >= 1 && n_expert_used <= n_used_hp * (n_expert / n_groups_hp) &&
            n_expert <= 1024 && probs->type == GGML_TYPE_F32 && probs->ne[0] == n_expert && ggml_is_contiguous(probs) &&
            (exp_probs_b == nullptr || (exp_probs_b->type == GGML_TYPE_F32 && ggml_is_contiguous(exp_probs_b) &&
                                        // Or a bias per row (the combined guess step's targets)
                                        (ggml_nelements(exp_probs_b) == n_expert || ggml_nelements(exp_probs_b) == n_expert*n_tokens)));
    if (fused_ok) {
        ggml_tensor * selected_experts = ggml_moe_group_topk(ctx0, probs, exp_probs_b, (int) n_groups_hp, (int) n_used_hp,
                /* k_group = */ 2, (int) n_expert_used);   // [n_expert_used, n_tokens]
        cb_tag(selected_experts, "ffn_moe_topk", il, name_tag);

        static const bool fused_check = getenv("MOE_STREAM_TEST_FUSED_CHECK") && atoi(getenv("MOE_STREAM_TEST_FUSED_CHECK")) != 0;
        if (fused_check) {
            fused_bypass = true;
            ggml_tensor * chain = build_moe_topk(probs, logits, exp_probs_b, n_expert, n_expert_used, il, name_tag);
            fused_bypass = false;
            ggml_format_name(chain, "fused_check_chain-%d", il);   // not ffn_moe_topk: tools key on that name
            ggml_tensor * sel = exp_probs_b ? ggml_add(ctx0, probs, exp_probs_b) : probs;
            ggml_build_forward_expand(gf, ggml_map_custom3(ctx0, selected_experts, chain, sel, fused_check_op, 1,
                        (void *) (intptr_t) il));
        }
        return selected_experts;
    }

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb_tag(selection_probs, "ffn_moe_probs_biased", il, name_tag);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens] (upstream)
        cb_tag(selection_probs, "ffn_moe_probs_biased", il, name_tag);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb_tag(expert_groups, "ffn_moe_group_topk", il, name_tag);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb_tag(selection_probs, "ffn_moe_probs_masked", il, name_tag);
    }

    // select experts
    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
    // --moe-stream-cpu-topk 1: on a machine with no GPU device the CPU argsort picks only the n_expert_used ids that the view
    // reads, in one pass (the same ids in the same order; ties and NaN sort the row as before). A GPU machine's graphs are unchanged
    if (mstream != nullptr && mstream->p1.cpu_topk && !mstream->el_has_gpu && n_expert_used <= 64) {
        selected_experts->src[0]->op_params[1] = (int32_t) n_expert_used;
    }
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
    cb_tag(selected_experts, "ffn_moe_topk", il, name_tag);

    return selected_experts;
}

// --moe-stream-wordstart D: the word-start guesses of source layer il (a layer before the first streamed one, not
// streamed: a VRAM MoE layer or a dense layer) for its la_targets X = il + d, built exactly as a streamed layer's far guesses are
// (build_moe_ffn's build_far, and the combined step with --moe-stream-guess-combined 1 when the source has 2+ targets and its
// stacks were built): with --moe-stream-pred-shared 1 the base normed with X's ffn_norm gain, without it cur_in (the source's FFN
// input after its own ffn_norm); X's router, bias, gating and group rule; the decode guess's width. Returns the guess columns
// [width, n_targets] in la_targets order (nullptr: no streamed target). With --moe-stream-record the logits go into the source's
// lead slots too (d <= 4), as a streamed source's. Graphs of one token only (the caller checks)
// The guess wiring every MoE model shares (moved out of bailingmoe3.cpp word for word; the same nodes in the
// same order, so Ling's graph is unchanged). See llm_graph_moe_guess in llama-graph.h.
llm_graph_moe_guess llm_graph_context::moe_guess_routers(const llama_model & model, int il) const {
    llm_graph_moe_guess g;
    if (!mstream) {
        return g;
    }
    const auto moe_at = [&](int j) { return j < (int) model.layers.size() && (uint32_t) j >= hparams.n_layer_dense_lead; };
    // the router of the next MoE layer, for the layer-ahead expert prefetch
    if (moe_at(il + 1)) {
        g.gate_inp_next = model.layers[il + 1].ffn_gate_inp;
        g.probs_b_next  = model.layers[il + 1].ffn_exp_probs_b;
        // --moe-stream-correction: the corrected matrix W'(il + 1, 1) (its gain folded in) when the file has this pair
        if (mstream->corr_gate(il + 1, 1)) {
            g.gate_inp_next = mstream->corr_gate(il + 1, 1);
        }
    }
    // And the routers two and three ahead. Their guesses are issued on this layer's own router input, so they are not delayed
    // by this layer's shared expert, and the layers in between correct them
    if (moe_at(il + 2)) {
        g.gate_inp_next2 = model.layers[il + 2].ffn_gate_inp;
        g.probs_b_next2  = model.layers[il + 2].ffn_exp_probs_b;
    }
    if (moe_at(il + 3)) {
        g.gate_inp_next3 = model.layers[il + 3].ffn_gate_inp;
        g.probs_b_next3  = model.layers[il + 3].ffn_exp_probs_b;
    }
    // --moe-stream-pred-shared: the prediction for the next layer is made from the residual plus *this* layer's
    // shared expert, normed with the next layer's own ffn_norm gain - what the next router will see, minus the routed part that
    // only exists once this layer's experts have run. The model builds the shared expert before build_moe_ffn for it
    g.want_shared = mstream->pred_shared && g.gate_inp_next;
    return g;
}

void llm_graph_context::moe_guess_finish(const llama_model & model, int il, ggml_tensor * ffn_inp, ggml_tensor * shared,
                                         llm_graph_moe_guess & g) const {
    if (!mstream) {
        return;
    }
    // residual + shared expert, before the norm
    moe_guess_finish_in(model, il, g.want_shared && shared ? ggml_add(ctx0, ffn_inp, shared) : nullptr, nullptr, g);
}

void llm_graph_context::moe_guess_finish_in(const llama_model & model, int il, ggml_tensor * base,
                                            const llm_graph_guess_in_fn & guess_in, llm_graph_moe_guess & g) const {
    if (!mstream) {
        return;
    }
    if (g.want_shared && base) {
        g.pred_base = base;
        // The model's own guess input, or the base normed with the next layer's gain. On a model with ffn_norm, a corrected
        // pair's norm weight is the ones (the gain is inside W'); on a model with its own guess input, W' = W_X + Bt reads guess_in directly
        g.pred_in   = guess_in ? guess_in(il + 1)
                    : build_norm(g.pred_base, mstream->corr_unit(il + 1, 1) ? mstream->corr_unit(il + 1, 1) : model.layers[il + 1].ffn_norm,
                            nullptr, LLM_NORM_RMS, il + 1);
    }
    // The routers and norm gains of the next four layers, for the recorder's guess logits. Up to
    // --moe-stream-lookahead layers ahead (and their selection bias) for the far guesses, built the same way
    g.rec_on = mstream->rec_on() || mstream->lookahead > 1;
    if (g.rec_on) {
        g.rec.base = g.pred_base;
        const int d_max = std::max(4, (int) mstream->lookahead);
        for (int d = 1; d <= d_max && d <= llm_graph_rec_leads::MAX_D; d++) {
            if (il + d < (int) model.layers.size() && (uint32_t) (il + d) >= hparams.n_layer_dense_lead) {
                g.rec.gate   [d] = model.layers[il + d].ffn_gate_inp;
                g.rec.norm   [d] = model.layers[il + d].ffn_norm;
                g.rec.probs_b[d] = model.layers[il + d].ffn_exp_probs_b;
                if (mstream->corr_gate(il + d, d)) {   // the corrected matrix (gain folded in) and the ones
                    g.rec.gate[d] = mstream->corr_gate(il + d, d);
                    g.rec.norm[d] = mstream->corr_unit(il + d, d);
                }
                // The model's own guess input for this target, only where a one-token graph reads it (the far guesses up to
                // --moe-stream-lookahead, the recorder's leads): each is a few ops, and unused ones would only fill the graph's context
                const bool used = n_tokens == 1 && ((mstream->prefetch_on() && d <= (int) mstream->lookahead) ||
                                                    (mstream->rec_on() && d <= llama_moe_stream::REC_LEADS && mstream->rec_lead_avail(il, d)));
                if (guess_in && g.pred_base && d >= 2 && used) {
                    g.rec.in[d] = guess_in(il + d);
                }
            }
        }
        if (guess_in && g.pred_base) {
            g.rec.in[1] = g.pred_in;
        }
    }
}

// --moe-stream-wordstart D: in a graph of one token the last D layers before the first streamed one (dense or VRAM MoE
// layers) build their guesses for the streamed layers 1..N ahead (build_ws_guess); one CPU op after the last of them hands them all
// to the waiting list (llama_moe_stream_wordstart). With D 0 the graph is unchanged
llm_graph_moe_ws llm_graph_context::moe_ws_begin() const {
    llm_graph_moe_ws st;
    st.graph = mstream && mstream->ws_on() && mstream->prefetch_on() && n_tokens == 1;
    // With the GPU producer on every source, each source's guess is marked (the GPU writes it into the source's slot) and
    // the word-start CPU op is not built
    st.s2g   = st.graph && mstream->ws_gpu && mstream->ne_gpu_graph();
    return st;
}

void llm_graph_context::moe_ws_layer(const llama_model & model, int il, ggml_tensor * out, ggml_tensor * ws_base,
                                     ggml_tensor * ffn_in_n, llm_graph_moe_ws & st, const llm_graph_guess_in_fn & guess_in) const {
    if (!st.graph || !mstream->ws_src(il)) {
        return;
    }
    if ((uint32_t) il < hparams.n_layer_dense_lead) {
        ws_base = out;   // a dense layer's base: the residual + its FFN output, the layer's output
    }
    llm_graph_rec_leads ws_leads;
    for (int d = 1; d <= (int) mstream->lookahead && d <= llm_graph_rec_leads::MAX_D; d++) {
        if (il + d < (int) model.layers.size() && (uint32_t) (il + d) >= hparams.n_layer_dense_lead) {
            ws_leads.gate   [d] = model.layers[il + d].ffn_gate_inp;
            ws_leads.norm   [d] = model.layers[il + d].ffn_norm;
            ws_leads.probs_b[d] = model.layers[il + d].ffn_exp_probs_b;
            if (mstream->corr_gate(il + d, d)) {   // a word-start source is in the correction file only if the trained pair has that source layer
                ws_leads.gate[d] = mstream->corr_gate(il + d, d);
                ws_leads.norm[d] = mstream->corr_unit(il + d, d);
            }
            if (guess_in && mstream->pred_shared && ws_base) {   // the model's own guess input (from its base)
                ws_leads.in[d] = guess_in(il + d);
            }
        }
    }
    ggml_tensor * cols = build_ws_guess(il, mstream->pred_shared ? ws_base : nullptr, ffn_in_n, ws_leads,
            (llama_expert_gating_func_type) hparams.expert_gating_func, n_expert);
    if (cols && st.s2g) {   // the mark, in this source's launch
        if (cols->view_src != nullptr || !ggml_is_contiguous(cols)) {
            cols = ggml_cont(ctx0, cols);
        }
        mstream->ws_mark_node(il, cols);
        moe_pin(cols, mstream->ne_guess_dev(il));   // (the GPU of its targets' routers: its event is there)
        ggml_build_forward_expand(gf, cols);
    } else if (cols) {
        st.cols = st.cols ? ggml_concat(ctx0, st.cols, cols, 1) : cols;
    }
    if (il == mstream->ws_S0 - 1 && st.cols) {   // the last source: the one CPU op of the word
        ggml_tensor * op = ggml_map_custom1(ctx0, st.cols, llama_moe_stream_wordstart, 1, mstream);
        ggml_format_name(op, "moe_wordstart");
        ggml_build_forward_expand(gf, op);
    }
}

ggml_tensor * llm_graph_context::build_ws_guess(
                     int   il,
             ggml_tensor * base,
             ggml_tensor * cur_in,
             const llm_graph_rec_leads & rec,
            llama_expert_gating_func_type gating_op,
                 int64_t   n_expert) const {
    int32_t ds[llama_moe_stream::LOOKAHEAD_MAX];
    const int32_t n_t = mstream->la_targets(il, ds);
    if (n_t == 0) {
        return nullptr;
    }
    const uint32_t width = mstream->guess_width(1);
    for (int32_t t = 0; t < n_t; t++) {
        GGML_ASSERT(ds[t] <= llm_graph_rec_leads::MAX_D && rec.gate[ds[t]] != nullptr);
        GGML_ASSERT(!mstream->pred_shared || rec.in[ds[t]] != nullptr || (base != nullptr && rec.norm[ds[t]] != nullptr));
    }
    ggml_tensor * lg_d[llm_graph_rec_leads::MAX_D + 1] = { nullptr };   // the logits per distance (the recorder's leads)

    // the separate ops (build_far's), without the combined step or for its bit check
    const bool comb_on = mstream->la_comb && mstream->ws_comb(il) && n_t >= 2;
    const bool sep_on  = !comb_on || mstream->la_gchk != 0;
    ggml_tensor * sep_lg[llm_graph_rec_leads::MAX_D + 1] = { nullptr };
    ggml_tensor * sep_pr[llm_graph_rec_leads::MAX_D + 1] = { nullptr };
    ggml_tensor * sep_id[llm_graph_rec_leads::MAX_D + 1] = { nullptr };
    ggml_tensor * cols = nullptr;
    if (sep_on) {
        for (int32_t t = 0; t < n_t; t++) {
            const int d = ds[t];
            const int X = il + d;
            ggml_tensor * rin = cur_in;
            if (mstream->pred_shared) {
                rin = rec.in[d] ? rec.in[d] : build_norm(base, rec.norm[d], nullptr, LLM_NORM_RMS, X);
            }
            ggml_tensor * lg = build_lora_mm(rec.gate[d], rin);   // [n_expert, 1]
            if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
                ggml_prec_set_acc(lg, GGML_PREC_F32);
            }
            cb_tag(lg, "ffn_moe_logits", X, "pred");
            ggml_tensor * f_logits = nullptr;
            ggml_tensor * f_probs  = build_moe_probs(rin, rec.gate[d], /* gate_inp_b = */ nullptr, /* logits_in = */ lg, gating_op, X,
                    &f_logits, /* name_tag = */ "pred");
            ggml_tensor * f_top    = ggml_cont(ctx0, build_moe_topk(f_probs, f_logits, rec.probs_b[d], n_expert, width, X,
                    /* name_tag = */ "pred"));
            sep_lg[d] = lg;
            sep_pr[d] = f_probs;
            sep_id[d] = f_top;
            lg_d[d]   = lg;
            cols = cols ? ggml_concat(ctx0, cols, f_top, 1) : f_top;
        }
    }

    // The combined step (the same code as a streamed layer's: one norm pinned to the first target's layer device, the stacked
    // gains, one GGML_OP_MUL_MAT_MULTI, the gating, the top-k with a bias per row)
    if (comb_on) {
        ggml_tensor * as[llama_moe_stream::LOOKAHEAD_MAX];
        for (int32_t t = 0; t < n_t; t++) {
            as[t] = rec.gate[ds[t]];
        }
        ggml_tensor * v = cur_in;
        if (mstream->pred_shared) {
            GGML_ASSERT(mstream->ws_gain(il) != nullptr);
            ggml_tensor * nrm = ggml_rms_norm(ctx0, base, hparams.f_norm_rms_eps);
            cb(nrm, "norm", il + ds[0]);
            ggml_format_name(nrm, "moe_comb_norm-%d", il);
            v = ggml_mul(ctx0, mstream->ws_gain(il), nrm);   // [n_embd, 1, n_t]
            cb_tag(v, "ffn_moe_pred_in", il, "pred_comb");
        }
        ggml_tensor * comb_lg = ggml_mul_mat_multi(ctx0, as, n_t, v);
        if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            ggml_prec_set_acc(comb_lg, GGML_PREC_F32);
        }
        cb_tag(comb_lg, "ffn_moe_logits", il, "pred_comb");
        ggml_tensor * c_logits = nullptr;
        ggml_tensor * comb_pr  = build_moe_probs(v, nullptr, /* gate_inp_b = */ nullptr, /* logits_in = */ comb_lg, gating_op, il,
                &c_logits, /* name_tag = */ "pred_comb");
        ggml_tensor * comb_id  = build_moe_topk(comb_pr, c_logits, mstream->ws_bias(il), n_expert, width, il, /* name_tag = */ "pred_comb");
        if (!ggml_is_contiguous(comb_id)) {
            comb_id = ggml_cont(ctx0, comb_id);
        }
        for (int32_t t = 0; t < n_t; t++) {
            lg_d[ds[t]] = ggml_view_2d(ctx0, comb_lg, n_expert, 1, comb_lg->nb[1], (size_t) t*comb_lg->nb[1]);
        }
        // TEST ONLY (MOE_STREAM_TEST_GUESS_CHECK): every row against the separate guess for the same target (the combined step's check op)
        if (mstream->la_gchk != 0) {
            for (int32_t t = 0; t < n_t; t++) {
                const int d = ds[t];
                ggml_tensor * args[6] = {
                    sep_lg[d], sep_pr[d], sep_id[d],
                    ggml_view_2d(ctx0, comb_lg, n_expert, 1, comb_lg->nb[1], (size_t) t*comb_lg->nb[1]),
                    ggml_view_2d(ctx0, comb_pr, n_expert, 1, comb_pr->nb[1], (size_t) t*comb_pr->nb[1]),
                    ggml_view_2d(ctx0, comb_id, comb_id->ne[0], 1, comb_id->nb[1], (size_t) t*comb_id->nb[1]),
                };
                ggml_tensor * chk = ggml_custom_4d(ctx0, GGML_TYPE_I32, 1, 1, 1, 1, args, 6, llama_moe_stream_guess_check, 1,
                        mstream->gchk_ud(il, d));
                ggml_format_name(chk, "moe_guess_check-%d-d%d", il, d);
                ggml_build_forward_expand(gf, chk);
            }
        }
        cols = comb_id;
    }

    // --moe-stream-record: the logits of each guess into the source's lead slot (lead d = the router of il + d)
    if (mstream->rec_on()) {
        for (int32_t t = 0; t < n_t; t++) {
            const int d = ds[t];
            if (d > llama_moe_stream::REC_LEADS || !mstream->rec_lead_avail(il, d)) {
                continue;
            }
            ggml_tensor * ldst = ggml_view_2d(ctx0, mstream->rec_logits_tensor(), n_expert, 1, n_expert*sizeof(float),
                    mstream->rec_logits_offset(il, d)*sizeof(float));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, lg_d[d]), ldst));
        }
    }
    return cols;
}

// --moe-stream-cpu-weighted-sum: the routing-weight multiply and the sum over the chosen experts of a layer
// whose MoE tail runs on the CPU, as one CPU op instead of ggml_mul + n_used views + n_used - 1 ggml_add (one barrier
// instead of n_used). The same bits as that chain: every product is rounded to float before its add (no FMA: GCC
// contracts a*b + c into an FMA by default at -O3, so contraction is turned off for this function), the sum starts with
// the first product itself (not 0.0f + it, which would turn a -0.0f into +0.0f), and the experts are added in the
// chain's order j = 0 .. n_used - 1. The threads split the n_embd rows. src[0] = experts [n_embd, n_expert_used,
// n_tokens], src[1] = weights [1, n_expert_used, n_tokens], both f32; userdata = n_used (the layer's n_expert_used).
#if defined(__clang__)
#define LLAMA_WSUM_NO_FP_CONTRACT
#elif defined(__GNUC__)
#define LLAMA_WSUM_NO_FP_CONTRACT __attribute__((optimize("fp-contract=off")))
#else
#define LLAMA_WSUM_NO_FP_CONTRACT
#endif

// The expert weights a layer's down matmul reads are in host memory (a view counts as its source)
static bool llama_moe_wsum_weights_on_host(const ggml_tensor * w) {
    const ggml_tensor * src = w && w->view_src ? w->view_src : w;
    return src && src->buffer && ggml_backend_buffer_is_host(src->buffer);
}

static void llama_moe_hybrid_tail_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    auto * ms = (llama_moe_stream *) userdata;
    if (ms) {
        ms->compute_hybrid_tail_cpu(dst->src[0], dst->src[1], dst, ith, nth);
    }
}

static LLAMA_WSUM_NO_FP_CONTRACT void llama_moe_weighted_sum_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const ggml_tensor * e = dst->src[0];
    const ggml_tensor * w = dst->src[1];
    const int64_t n_used   = (int64_t) (intptr_t) userdata;
    const int64_t n_embd   = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];

    const int64_t dr = (n_embd + nth - 1)/nth;
    const int64_t r0 = std::min<int64_t>(dr*ith, n_embd);
    const int64_t r1 = std::min<int64_t>(r0 + dr, n_embd);

    for (int64_t t = 0; t < n_tokens; t++) {
        const char * eb  = (const char *) e->data + t*e->nb[2];
        const char * wb  = (const char *) w->data + t*w->nb[2];
        float      * out = (float *) ((char *) dst->data + t*dst->nb[1]);

        const float   w0 = *(const float *) wb;
        const float * e0 = (const float *) eb;
        for (int64_t r = r0; r < r1; r++) {
            out[r] = e0[r]*w0;
        }
        for (int64_t j = 1; j < n_used; j++) {
            const float   wj = *(const float *) (wb + j*w->nb[1]);
            const float * ej = (const float *) (eb + j*e->nb[1]);
            for (int64_t r = r0; r < r1; r++) {
                const float pr = ej[r]*wj;   // rounded to float, as ggml_mul stores it
                out[r] = out[r] + pr;        // as ggml_add
            }
        }
    }
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s,
         ggml_tensor * selected_experts_in,
         ggml_tensor * gate_inp_next,
         ggml_tensor * exp_probs_b_next,
         ggml_tensor * pred_in,
         ggml_tensor * gate_inp_next2,   // the router of the layer *two* ahead
         ggml_tensor * exp_probs_b_next2,
         ggml_tensor * gate_inp_next3,   // lead 3: the router of the layer *three* ahead
         ggml_tensor * exp_probs_b_next3,
         const llm_graph_rec_leads * rec // what the recorder needs of the next layers
         ) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    // probs_in used to be the router *logits* of a layer whose matmul this graph does not build
    // (a model that shares one router between layers), and the bias and the gating were applied to it.
    // It is now used as the final probabilities, which changed gemma4, nemotron-h and nemotron-h-moe.
    // It goes in as logits_in, so the bias and the gating are applied to it again.
    // The expert graph of a layer-major call takes the router's picks and weights of the whole call from phase 1
    ggml_tensor * selected_experts = selected_experts_in;
    ggml_tensor * probs  = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_tensor * weights_lm = nullptr;
    ggml_tensor * weights_rt = nullptr;   // the router's weights (every other case)
    if (lm_phase == 2) {
        GGML_ASSERT(lm_ids && lm_w && lm_hid && mstream && mstream->layer(il) && lm_chunk > 0 && !weight_before_ffn);
        selected_experts = lm_ids;
        weights_lm       = ggml_reshape_3d(ctx0, lm_w, 1, n_expert_used, n_tokens);
    } else {
    probs = build_moe_probs(cur, gate_inp, gate_inp_b, probs_in, gating_op, il, &logits);

    // select experts
    if (selected_experts == nullptr) {
        // The logits were not passed here, so llama4 selected its experts from a null tensor
        // (the segfault of test-llama-archs) and grovemoe took sigmoid(probs) instead of sigmoid(logits)
        selected_experts = build_moe_topk(probs, logits, exp_probs_b, n_expert, n_expert_used, il);
    }

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    weights_rt = weights;

    // The router's picks and weights go into the call's tensors at this ubatch's rows; no experts, no guess
    if (lm_phase == 1) {
        GGML_ASSERT(lm_ids && lm_w);
        ggml_tensor * ids_dst = ggml_view_2d(ctx0, lm_ids, n_expert_used, n_tokens, lm_ids->nb[1], lm_row0*lm_ids->nb[1]);
        ggml_tensor * w_dst   = ggml_view_2d(ctx0, lm_w,   n_expert_used, n_tokens, lm_w->nb[1],   lm_row0*lm_w->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, selected_experts), ids_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens), w_dst));
        return nullptr;
    }
    }   // (lm_phase != 2)
    ggml_tensor * weights = weights_lm ? weights_lm : weights_rt;

    // GPU-producer decode graph (--moe-stream-needed-early), grouped routing with the fused route op: the topk-moe fusion cannot fire here and
    // the CPU piece needs none of the weights (it gets only the ids and the router input), so the chain is not expanded in front of
    // the router mark: it is expanded after the mark and the guess (below, or after that block when the layer gets no mark), where
    // it runs in the GPU window and is ready for the VRAM pass and the join. Same ops on the same inputs.
    // MOE_STREAM_TEST_WEIGHTS_EARLY=1 (test only) keeps the old order for the A/B.
    static const bool weights_early_test = getenv("MOE_STREAM_TEST_WEIGHTS_EARLY") && atoi(getenv("MOE_STREAM_TEST_WEIGHTS_EARLY")) != 0;
    const bool weights_deferred = !weights_early_test && mstream && mstream->p1.fused_route && hparams.n_expert_groups > 1 &&
            n_tokens == 1 && mstream->ne_gpu_graph() && mstream->ne_rd_layer_on(il);
    bool weights_expanded = false;

    //call early so that topk-moe can be used
    if (!weights_deferred) {
        ggml_build_forward_expand(gf, weights);
        weights_expanded = true;
    }

    // the router input of this layer, [n_embd, n_tokens]; every wave of the layer reads it as is
    ggml_tensor * cur_in = cur;

    // the expert matmuls of one wave: ids_gemm holds the pool slots of this wave (the pairs of the
    // other waves get the all-zero slot, so their contribution is exactly 0) and weights the routing
    // weights of the ubatch. Summed over the k expert uses of every token, as without waves.
    // A view of a weight tensor, so that two waves of the same layer are two distinct sources for
    // the scheduler. The view covers the same memory (the pool is refilled between waves, and the ids
    // are absolute slot indices), it only gives each wave its own copy to take.
    // --moe-stream-elastic / --moe-stream-shares: a pool tensor is allocated for its cap. A matmul the GPU may run
    // (a GPU wave, or the scheduler's offload of a one-wave 32+ row ubatch) gets a view of the first start-size slots + the zero
    // slices only - the size reserved at start-up, whatever the layer's size now: in such a call the engine keeps every expert of
    // the layer and its zero ids in that range (llama_moe_stream::el_low_start). Every wave gets its own view
    const bool    wave_el = mstream != nullptr && mstream->layer(il) != nullptr && mstream->el_has_gpu;
    const int64_t wave_ns = wave_el ? mstream->wave_view_slots() + n_expert_used : 0;   // (TEST ONLY wide reserve)
    // One view per (pool, wave, gpu) in this layer, so that the chunks of a wave share one source and the scheduler copies
    // the wave's experts once (the plain path: one call per wave and projection - the same views as before)
    std::map<std::tuple<const ggml_tensor *, int, bool>, ggml_tensor *> wave_view_memo;
    const auto wave_view = [&](ggml_tensor * t, int wave, bool gpu) -> ggml_tensor * {
        if (t == nullptr) {
            return t;
        }
        const auto key = std::make_tuple((const ggml_tensor *) t, wave, gpu);
        const auto it = wave_view_memo.find(key);
        if (it != wave_view_memo.end()) {
            return it->second;
        }
        ggml_tensor * v = t;
        if (gpu && wave_ns > 0 && t->ne[2] > wave_ns) {
            v = ggml_view_3d(ctx0, t, t->ne[0], t->ne[1], wave_ns, t->nb[1], t->nb[2], 0);
        } else if (wave > 0) {
            v = ggml_view_tensor(ctx0, t);
        }
        wave_view_memo[key] = v;
        return v;
    };

    // --moe-stream-gpu-wave-skip-zero: a MUL_MAT_ID of more than one token whose weight tensor ends in the zero slices
    // carries their count (the CUDA mmq path then skips their pairs and writes 0); anything else (a scale / bias between, LoRA) stays as
    // it is
    const auto mark_zero_tail = [&](ggml_tensor * t, const ggml_tensor * w) {
        if (mstream == nullptr || t == nullptr || w == nullptr || t->op != GGML_OP_MUL_MAT_ID || t->src[0] != w) {
            return;
        }
        const int32_t nz = mstream->zero_tail_of(w, il);
        if (nz > 0) {
            t->op_params[GGML_MOE_ZERO_TAIL_PARAM] = GGML_MOE_ZERO_TAIL_MAGIC | nz;
            mstream->n_zt_marked += 1;
        } else if (mstream->p1.gpu_wave_skip_zero) {
            mstream->n_zt_refused += 1;
        }
    };

    // Tail waves: this layer's GPU waves read the gather copy's tensors at the tail's places (set by
    // the wave loop below, with the tail-wave flag on the wave's ids op)
    bool tailw_w = false;

    // x_c = a chunk of the expert input ([n_embd, nt] rows of the call) with its ids / weights views - the same chain on
    // the chunk only (the plain path: the whole ubatch)
    bool ca_mark_wave = true;   // a layer-major wave marks only its last chunk (below)
    const auto build_expert_wave = [&](ggml_tensor * ids_gemm, ggml_tensor * weights, int wave = 0, ggml_tensor * x_c = nullptr) -> ggml_tensor * {
        ggml_tensor * cur = x_c ? x_c : cur_in;
        const int64_t nt  = x_c ? x_c->ne[1] : n_tokens;

        // When the expert matmuls of a wave graph run on the GPU, every wave reads its own view of
        // the pool, so that the scheduler has a separate host-weight source per wave and copies it when
        // that wave's split runs. With the flag off (the default) there is no view and the graph is the
        // plain wave graph.
        //
        // A repacked pool cannot be on this path at all. A CPU_REPACK buffer type has no cpy_tensor
        // and no get_tensor, so a view of it is not a weight the scheduler can copy to the device: with
        // --moe-stream-cpu-repack 1 and --moe-stream-gpu-waves 512 the wave graphs came out with 13
        // distinct experts per ubatch-layer instead of 115 on tiny, and the text after the prompt was
        // "!!!!!!!!". The wave matmuls then run on the host, which is correct (measured), so the
        // guard is "no per-wave view": the graph is the plain wave graph, with host waves.
        // In the layer-major expert graph every chunk decides as the graph did - by n_ubatch, not its own size - so
        // a short last chunk reads each wave's own pool view, never wave 0's copy
        const int64_t nt_gw = lm_phase == 2 ? lm_chunk : nt;
        const bool gpu_w = mstream != nullptr && mstream->gpu_waves > 0 && nt_gw >= (int) mstream->gpu_waves &&
                !(mstream->cpu_repack_on.load());
        const int  w     = gpu_w ? wave : 0;

        // A ONE-wave prompt ubatch of 32+ rows is offloaded to the GPU with a copy of the whole pool
        // (op_offload), so its copy is limited to the switched-on slots too - the same matmul on the same bytes. Never a
        // CPU-wave ubatch (a view there let the scheduler move the CPU waves' matmuls to the GPU: another text) and
        // never a repacked pool (a view of it would lose its repack traits)
        // (the engine's one rule, el_gpu_view - the same as its call-level low-only decision)
        const bool cp_w = wave_el && mstream->el_gpu_view(nt_gw);
        ggml_tensor * gate_up_w = wave_view(gate_up_exps, w, cp_w);
        ggml_tensor * up_w      = wave_view(up_exps,      w, cp_w);
        ggml_tensor * gate_w    = wave_view(gate_exps,    w, cp_w);
        ggml_tensor * down_w    = wave_view(down_exps,    w, cp_w);
        if (tailw_w && gpu_w) {   // tail waves: the engine copies the wave's experts there; the scheduler copies nothing
            const int n_w = mstream->n_waves(n_tokens);   // the set of this step (two sets alternate by wave too)
            gate_w = mstream->tailw_tensor(0, il, wave, n_w);   // the tail tensors of this layer's format
            up_w   = mstream->tailw_tensor(1, il, wave, n_w);
            down_w = mstream->tailw_tensor(2, il, wave, n_w);
            if (auto * wc = mstream->wave_ctx_of(il, wave)) {
                wc->tail_gate = gate_w;   // (the copy-ahead check compares it with the gather's tensor)
            }
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);

        if (weight_before_ffn) {
            // repeat cur to [n_embd, n_expert_used, nt]
            ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, nt, 1);
            cur = ggml_mul(ctx0, repeated, weights);
            cb(cur, "ffn_moe_weighted", il);
        }

        ggml_tensor * up = nullptr;
        ggml_tensor * experts = nullptr;
        ggml_tensor * cur_gate_mm = nullptr;   // the gate matmul's node (before its scale / bias)

        if (gate_up_exps) {
            // merged gate_up path: one mul_mat_id, then split into gate and up views
            ggml_tensor * gate_up = build_lora_mm_id(gate_up_w, cur, ids_gemm, up_exps_s, selected_experts); // [n_ff*2, n_expert_used, nt]
            cb(gate_up, "ffn_moe_gate_up", il);

            if (up_exps_s) {
                cb(gate_up, "ffn_moe_gate_up_scaled", il);
            }

            if (gate_up_exps_b) {
                gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
                cb(gate_up, "ffn_moe_gate_up_biased", il);
            }

            const int64_t n_ff = gate_up->ne[0] / 2;
            cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
            cb(cur, "ffn_moe_gate", il);
            up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
            cb(up, "ffn_moe_up", il);
        } else {
            // separate gate and up path
            up = build_lora_mm_id(up_w, cur, ids_gemm, up_exps_s, selected_experts); // [n_ff, n_expert_used, nt]
            cb(up, "ffn_moe_up", il);

            if (up_exps_s) {
                cb(up, "ffn_moe_up_scaled", il);
            }

            if (up_exps_b) {
                up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
                cb(up, "ffn_moe_up_biased", il);
            }

            if (gate_exps) {
                cur = build_lora_mm_id(gate_w, cur, ids_gemm, gate_exps_s, selected_experts); // [n_ff, n_expert_used, nt]
                cur_gate_mm = cur;
                cb(cur, "ffn_moe_gate", il);
            } else {
                cur = up;
            }

            if (gate_exps_s) {
                cb(cur, "ffn_moe_gate_scaled", il);
            }

            if (gate_exps_b) {
                cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
                cb(cur, "ffn_moe_gate_biased", il);
            }
        }

        const bool has_gate = gate_exps || gate_up_exps;

        switch (type_op) {
            case LLM_FFN_SILU:
                if (gate_exps) {
                    if (il >= 0) {
                        const float limit = hparams.swiglu_clamp_exp[il];
                        constexpr float eps = 1e-6f;
                        if (limit > eps) {
                            if (arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0) || arch == LLM_ARCH_HY_V4) {
                                cur = ggml_swiglu_clamp(ctx0, cur, up, limit);
                            } else {
                                up = ggml_clamp(ctx0, up, -limit, limit);
                                cb(up, "ffn_moe_up_clamped", il);
                                ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                                cb(gate_act, "ffn_moe_silu", il);
                                gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                                cb(gate_act, "ffn_moe_silu_clamped", il);
                                cur = ggml_mul(ctx0, gate_act, up);
                            }
                            cb(cur, "ffn_moe_swiglu_limited", il);
                            break;
                        }
                    }
                }

                if (has_gate) {
                    cur = ggml_swiglu_split(ctx0, cur, up);
                    cb(cur, "ffn_moe_swiglu", il);
                } else {
                    cur = ggml_silu(ctx0, cur);
                    cb(cur, "ffn_moe_silu", il);
                } break;
            case LLM_FFN_SITU:
                {
                    // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
                    GGML_ASSERT(has_gate);
                    const float beta = hparams.situ_beta;
                    const float lb   = hparams.situ_linear_beta;

                    ggml_tensor * act = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/beta)), beta);
                    act = ggml_mul(ctx0, act, ggml_sigmoid(ctx0, cur));
                    if (lb > 0.0f) {
                        up = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, up, 1.0f/lb)), lb);
                    }
                    cur = ggml_mul(ctx0, act, up);
                    cb(cur, "ffn_moe_situ", il);
                } break;
            case LLM_FFN_GELU:
                if (has_gate) {
                    cur = ggml_geglu_split(ctx0, cur, up);
                    cb(cur, "ffn_moe_geglu", il);
                } else {
                    cur = ggml_gelu(ctx0, cur);
                    cb(cur, "ffn_moe_gelu", il);
                } break;
            case LLM_FFN_SWIGLU_OAI_MOE:
                {
                    // TODO: move to hparams?
                    constexpr float alpha = 1.702f;
                    constexpr float limit = 7.0f;
                    cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                    cb(cur, "ffn_moe_swiglu_oai", il);
                } break;
            case LLM_FFN_RELU:
                if (has_gate) {
                    cur = ggml_reglu_split(ctx0, cur, up);
                    cb(cur, "ffn_moe_reglu", il);
                } else {
                    cur = ggml_relu(ctx0, cur);
                    cb(cur, "ffn_moe_relu", il);
                } break;
            case LLM_FFN_RELU_SQR:
                if (has_gate) {
                    // TODO: add support for gated squared relu
                    GGML_ABORT("fatal error: gated squared relu not implemented");
                } else {
                    cur = ggml_relu(ctx0, cur);
                    cur = ggml_sqr(ctx0, cur);
                    cb(cur, "ffn_moe_relu_sqr", il);
                } break;
            default:
                GGML_ABORT("fatal error");
        }

        // Copy-ahead, one set: the node after this wave's gate and up matmuls tells the copier their places are free
        if (tailw_w && gpu_w && ca_mark_wave) {
            mstream->ca_mark_node(cur, il);
        }

        // Upstream #30861: the down projection's input rescaled per expert and token by a power of two (it can exceed the
        // F16 range on these archs), undone on the output with the same factor - as upstream's build_moe_ffn does;
        // the amax op cannot be split across devices, so these architectures are denied -sm tensor [TAG_MOE_AMAX_SCALE]
        llm_graph_amax_src1 amax_src1 = { nullptr, nullptr };

        if (arch == LLM_ARCH_MISTRAL4 || arch == LLM_ARCH_MELLUM || arch == LLM_ARCH_FLINT) {
            // --moe-stream-cpu-down-scale 1: left out where the down matmul certainly runs on the CPU and its kernel quantizes
            // the input with a float block scale - the same bits there (llama_moe_stream::down_scale_skip); kept everywhere else
            const bool ds_skip = mstream != nullptr && mstream->layer(il) != nullptr &&
                                 mstream->down_scale_skip(down_exps->type, nt, gpu_w, cparams.op_offload);
            if (mstream != nullptr && mstream->p1.cpu_down_scale && mstream->layer(il) != nullptr) {
                (ds_skip ? mstream->ds_built_skip : mstream->ds_built_keep)++;
            }
            if (!ds_skip) {
                amax_src1 = build_amax_src1_f16(cur);
                cur = amax_src1.res;
            } else if (mstream->ds_count) {   // test only: count the rows the step would have scaled (reads the input, changes nothing)
                cur = ggml_map_custom1_inplace(ctx0, cur, llama_moe_stream_down_scale_count, 1, mstream);
            }
        }
        experts = build_lora_mm_id(down_w, cur, ids_gemm, down_exps_s, selected_experts); // [n_embd, n_expert_used, nt]
        if (gpu_w && nt >= 2) {   // --moe-stream-gpu-wave-skip-zero: the zero tails of this GPU wave's three matmuls
            mark_zero_tail(experts, down_w);
            mark_zero_tail(up, up_w);
            mark_zero_tail(cur_gate_mm, gate_w);
        }
        cb(experts, "ffn_moe_down", il);
        if (amax_src1.scale_s) {
            experts = ggml_mul(ctx0, experts, amax_src1.scale_s);
        }

        if (down_exps_s) {
            cb(experts, "ffn_moe_down_scaled", il);
        }

        if (down_exps_b) {
            experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
            cb(experts, "ffn_moe_down_biased", il);
        }

        // --moe-stream-cpu-weighted-sum: in a layer whose tail runs on the CPU, the weight multiply and the sum
        // are one CPU op (llama_moe_weighted_sum_cpu, the same bits). Decided here, when the graph is built:
        //  - the layer AND what consumes its output (the next layer, or the output after the last) are on the CPU device
        //    (mstream->p1.wsum_cpu_tail, from the model's layer devices: -ngl 0, a CPU-only machine, layers -ngl N leaves on
        //    the CPU). Not the experts' buffer: streamed experts are always in host pools, in GPU layers too, and there the
        //    sum is one GPU kernel - a CPU-only op would move it to the CPU and change the splits. Not the layer alone:
        //    the scheduler pulls the tail of the last CPU layer before a GPU layer onto the GPU (tiny -ngl 12, layer 12);
        //  - not a layer with the VRAM expert pass (its device output is added to this sum);
        //  - the expert weights the down matmul reads are in host memory: a CPU-device layer whose
        //    experts sit in VRAM (a whole VRAM layer of --moe-stream-gpu-layers(-list) with -ngl 0) runs that matmul on the
        //    GPU, and a CPU op would move its sum;
        //  - and no device backend would take this layer's expert matmul by op offload (the scheduler's own test,
        //    ggml_backend_supports_op + ggml_backend_offload_op: a big prompt batch with a GPU present), since then
        //    the plain sum runs on that device;
        //  - weights after the experts (!weight_before_ffn), after any down bias, more than one expert, the shapes as the chain
        //    expects. Otherwise the plain chain. The wave path builds each wave's sum here too, so it is covered the same way.
        if (!weight_before_ffn && mstream && mstream->p1.cpu_wsum && hparams.n_expert_used(il) > 1 &&
                experts->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32 && experts->nb[0] == sizeof(float) &&
                ggml_n_dims(weights) <= 3 && weights->ne[0] == 1 && weights->ne[1] == experts->ne[1] &&
                weights->ne[2] == experts->ne[2] && (int64_t) hparams.n_expert_used(il) <= experts->ne[1] &&
                il >= 0 && (size_t) il < mstream->p1.wsum_cpu_tail.size() && mstream->p1.wsum_cpu_tail[il] &&
                !mstream->uma_gpu_pool && !mstream->vram_serves(il) && llama_moe_wsum_weights_on_host(down_exps)) {
            bool on_cpu = true;
            if (cparams.op_offload) {
                for (int b = 0; b < ggml_backend_sched_get_n_backends(sched) && on_cpu; b++) {
                    ggml_backend_t     be  = ggml_backend_sched_get_backend(sched, b);
                    ggml_backend_dev_t dev = ggml_backend_get_device(be);
                    if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU &&
                            ggml_backend_supports_op(be, experts) && ggml_backend_offload_op(be, experts)) {
                        on_cpu = false;
                    }
                }
            }
            if (on_cpu) {
                ggml_tensor * args[2] = { experts, weights };
                ggml_tensor * moe_out = ggml_custom_4d(ctx0, GGML_TYPE_F32, n_embd, nt, 1, 1, args, 2,
                        llama_moe_weighted_sum_cpu, GGML_N_TASKS_MAX, (void *) (intptr_t) hparams.n_expert_used(il));
                cb(moe_out, "ffn_moe_wsum", il);
                ggml_build_forward_expand(gf, moe_out);
                return moe_out;
            }
        }

        if (!weight_before_ffn) {
            experts = ggml_mul(ctx0, experts, weights);
            cb(experts, "ffn_moe_weighted", il);
        }

        ggml_build_forward_expand(gf, experts);

        ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

        assert(n_expert_used > 0);

        // order the views before the adds
        // Use per-layer n_expert_used to bound the graph even during warmup (avoids
        // the large-add-nodes issue for uniform arches; for Puzzle the per-layer
        // value is correct). ref: https://github.com/ggml-org/llama.cpp/pull/14753
        const uint32_t n_expert_used_il = hparams.n_expert_used(il);
        for (uint32_t i = 0; i < n_expert_used_il; ++i) {
            cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, nt, experts->nb[2], i*experts->nb[1]);

            ggml_build_forward_expand(gf, cur_experts[i]);
        }

        // aggregate experts
        ggml_tensor * moe_out = cur_experts[0];

        for (uint32_t i = 1; i < n_expert_used_il; ++i) {
            moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

            ggml_build_forward_expand(gf, moe_out);
        }

            if (n_expert_used_il == 1) {
                // avoid returning a non-contiguous tensor
                moe_out = ggml_cont(ctx0, moe_out);
            }

            if (mstream && mstream->p1.hybrid_tail > 0 && tailw_w && gpu_w && nt >= 8 && nt <= 2048 &&
                    !weight_before_ffn && gate_exps != nullptr && !gate_up_exps &&
                    !up_exps_s && !gate_exps_s && !down_exps_s && !up_exps_b && !gate_exps_b && !down_exps_b &&
                    type_op == LLM_FFN_SILU && (il < 0 || hparams.swiglu_clamp_exp[il] <= 1e-6f)) {
                ggml_tensor * x_wave = ggml_reshape_3d(ctx0, x_c ? x_c : cur_in, n_embd, 1, nt);
                ids_gemm->op_params[2] = 0x48544149;
                ggml_tensor * h_args[3] = { x_wave, weights, ids_gemm };
                ggml_tensor * cpu_tail  = ggml_custom_4d(ctx0, GGML_TYPE_F32, n_embd, nt, 1, 1, h_args, 3,
                        llama_moe_hybrid_tail_cpu, GGML_N_TASKS_MAX, mstream);
                ggml_format_name(cpu_tail, "ffn_moe_htail-%d", il);
                moe_out = ggml_add(ctx0, moe_out, cpu_tail);
                ggml_build_forward_expand(gf, moe_out);
            }

            return moe_out;
        };

    // --moe-stream-vram-experts: the device pass over the VRAM expert pool - the same chain as the host's (gate and up
    // from one merged tensor of the VRAM pool, with the layer's swiglu clamp), ids from llama_moe_stream_remap_vram, then the
    // routing weights and the sum over the k pairs, as the host path does. Its output is added to the host's ffn_moe_out.
    // Not before the weights: there the add changed what the CUDA backend fuses (the host's weighted sum ran unfused, in
    // another summation order), so even an all-zero device pass changed the text (MOE_STREAM_VRAM_TEST_EMPTY).
    // x_c / w_c = a chunk of the call's expert input and weights (ids_v its view) - the same pass on the chunk
    // abs = the ids are absolute places (the vp_table paths) - the view is never shifted there
    const auto build_vram_chain = [&](ggml_tensor * ids_v, ggml_tensor * x_c = nullptr, ggml_tensor * w_c = nullptr, bool abs = false) -> ggml_tensor * {
        const int64_t nt = x_c ? x_c->ne[1] : n_tokens;
        ggml_tensor * x  = ggml_reshape_3d(ctx0, x_c ? x_c : cur_in, n_embd, 1, nt);
        ggml_tensor * w_gu = mstream->vx_pool_view(ctx0, il, 0, abs);
        ggml_tensor * w_dn = mstream->vx_pool_view(ctx0, il, 1, abs);
        ggml_tensor * gu = ggml_mul_mat_id(ctx0, w_gu, x, ids_v);   // [2*n_ff, n_expert_used, n_tokens]
        cb(gu, "ffn_moe_vram_gate_up", il);
        if (nt >= 2) {
            mark_zero_tail(gu, w_gu);
        }

        const int64_t n_ff = gu->ne[0]/2;
        ggml_tensor * g = ggml_view_3d(ctx0, gu, n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], 0);
        ggml_tensor * u = ggml_view_3d(ctx0, gu, n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], n_ff*gu->nb[0]);

        ggml_tensor * act = nullptr;
        const float limit = il >= 0 ? hparams.swiglu_clamp_exp[il] : 0.0f;
        if (limit > 1e-6f) {   // the host path's clamp for a separate gate tensor (llm_graph_context::build_moe_ffn above)
            if (arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0) || arch == LLM_ARCH_HY_V4) {
                act = ggml_swiglu_clamp(ctx0, g, u, limit);
            } else {
                // the CUDA clamp and silu kernels read their input as one flat array, so the halves of the merged
                // gate|up output are made contiguous first (the host path's gate and up are separate tensors)
                g = ggml_cont(ctx0, g);
                u = ggml_cont(ctx0, u);
                u = ggml_clamp(ctx0, u, -limit, limit);
                ggml_tensor * a = ggml_silu(ctx0, g);
                a   = ggml_clamp(ctx0, a, -INFINITY, limit);
                act = ggml_mul(ctx0, a, u);
            }
        } else {
            act = ggml_swiglu_split(ctx0, g, u);
        }
        cb(act, "ffn_moe_vram_swiglu", il);

        // The down projection's input rescaled by a power of two and undone on the output, as the host path does
        // (upstream #30861: it can exceed the F16 range of the Q8 activation blocks' sums on these archs; Flint is one of them)
        llm_graph_amax_src1 vx_amax = { nullptr, nullptr };
        if (arch == LLM_ARCH_MISTRAL4 || arch == LLM_ARCH_MELLUM || arch == LLM_ARCH_FLINT) {
            vx_amax = build_amax_src1_f16(act);
            act = vx_amax.res;
        }
        ggml_tensor * dn = ggml_mul_mat_id(ctx0, w_dn, act, ids_v);   // [n_embd, n_expert_used, n_tokens]
        cb(dn, "ffn_moe_vram_down", il);
        if (nt >= 2) {
            mark_zero_tail(dn, w_dn);   // (on the matmul itself, before the rescale)
        }
        if (vx_amax.scale_s) {
            dn = ggml_mul(ctx0, dn, vx_amax.scale_s);
        }

        dn = ggml_mul(ctx0, dn, w_c ? w_c : weights);
        cb(dn, "ffn_moe_vram_weighted", il);
        ggml_build_forward_expand(gf, dn);

        const uint32_t n_used = hparams.n_expert_used(il);
        ggml_tensor * views[LLAMA_MAX_EXPERTS] = { nullptr };
        for (uint32_t i = 0; i < n_used; ++i) {   // the views before the adds, as build_expert_wave orders them
            views[i] = ggml_view_2d(ctx0, dn, n_embd, nt, dn->nb[2], i*dn->nb[1]);
            ggml_build_forward_expand(gf, views[i]);
        }
        ggml_tensor * out = views[0];
        for (uint32_t i = 1; i < n_used; ++i) {
            out = ggml_add(ctx0, out, views[i]);
            ggml_build_forward_expand(gf, out);
        }
        if (n_used == 1) {
            out = ggml_cont(ctx0, out);
        }
        cb(out, "ffn_moe_vram_out", il);
        return out;
    };

    // MoE expert streaming: make the selected experts resident in this layer's pool (loading the
    //   misses) and rewrite the ids that the expert GEMMs use to pool slots. Only the ids change,
    //   so the routing itself - which experts, in which order, with which weights - is untouched.
    //
    //   A ubatch that selects more distinct experts than the pool holds is split into waves: the
    //   remap of wave w loads the experts of that wave, and the pairs of the other waves get the
    //   all-zero slot, so their contribution is exactly 0 and the waves sum up. The wave count is
    //   known only at run time, which is why this cannot be one op for the whole layer.
    ggml_tensor * ids_gemm = selected_experts;

    if (mstream) {
        if (auto * msl = mstream->layer(il)) {
            // layer-ahead prefetch: the router of the next MoE layer on this layer's router input,
            // so that the experts it picks can be read while this layer computes
            ggml_tensor * pred    = nullptr;
            ggml_tensor * pred_tr = nullptr;   // the guessed top-8 of layer pred_il, for the counter
            ggml_tensor * pred_p  = nullptr;   // the sigmoid scores of the guess's top 16 (decode, chance table)
            ggml_tensor * pred_lg = nullptr;   // the router logits that guess is made from
            ggml_tensor * far_lg[llm_graph_rec_leads::MAX_D + 1] = { nullptr };   // the far guesses' logits
            std::vector<std::pair<ggml_tensor *, ggml_tensor *>> rec_copies;   // (source, host slot) copies, expanded below

            // With --moe-stream-needed-early 1 (the in-graph form of a CPU layer, a graph of one token), op A takes the router's
            // ids right here - expanded into the graph before anything below, so before the guess, the far guesses and (with
            // --moe-stream-pred-shared) the shared expert they read: the CPU backend runs the nodes in this order, so the needed
            // reads are queued before that work starts. Op B (below, in place of the remap op) takes op A's output too
            ggml_tensor * ig_ids = nullptr;
            ggml_tensor * ig_a   = nullptr;
            if (msl->ne_ingraph && n_tokens == 1 && mstream->prefetch_on()) {
                ig_ids = ggml_cont(ctx0, selected_experts);   // the top_k output is a view (the remap op's ids below)
                ig_a   = ggml_map_custom1(ctx0, ig_ids, llama_moe_stream_remap_a, 1, mstream->wave_ctx_of(il, 0));
                ggml_format_name(ig_a, "moe_ig_a-%d", il);
                ggml_build_forward_expand(gf, ig_a);
            }

            // --moe-stream-lookahead N (a graph of one token): the far guesses for the layers il+2 .. il+N, built
            // exactly like the recorder's lead d below (the counts were made on those): with --moe-stream-pred-shared
            // the residual + shared expert normed with layer X's ffn_norm gain, without it this layer's router input; then X's
            // router, X's bias, the same probs / top-k as the 1-ahead guess, the same width. Column d-1 of the remap op's guess
            // input is distance d; the columns stop at the first distance with no router (past the last layer, or a missing
            // shared-expert input) - the engine skips a column whose target is not a streamed layer
            const bool far_on = mstream->prefetch_on() && mstream->lookahead > 1 && n_tokens == 1 && rec;
            // A column only for a streamed target - the engine's la_targets, which the engine reads the columns by
            int32_t la_ds[llama_moe_stream::LOOKAHEAD_MAX];
            const int32_t la_nt = far_on ? mstream->la_targets(il, la_ds) : 0;
            // The separate guesses' logits / probabilities / ids per distance (for the bit check)
            ggml_tensor * sep_lg[llm_graph_rec_leads::MAX_D + 1] = { nullptr };
            ggml_tensor * sep_pr[llm_graph_rec_leads::MAX_D + 1] = { nullptr };
            ggml_tensor * sep_id[llm_graph_rec_leads::MAX_D + 1] = { nullptr };
            auto build_far = [&](ggml_tensor * cols, uint32_t width) {
                for (int32_t t = 0; t < la_nt; t++) {
                    const int d = la_ds[t];
                    if (d < 2) {
                        continue;   // the 1-ahead guess is built by its own block
                    }
                    const int X = il + d;
                    if (rec->gate[d] == nullptr) {
                        break;
                    }
                    ggml_tensor * rin = cur;
                    if (mstream->pred_shared) {
                        if (rec->in[d]) {   // the model's own guess input
                            rin = rec->in[d];
                        } else if (!rec->base || !rec->norm[d]) {
                            break;
                        } else {
                            rin = build_norm(rec->base, rec->norm[d], nullptr, LLM_NORM_RMS, X);
                        }
                    }
                    ggml_tensor * lg = build_lora_mm(rec->gate[d], rin);   // [n_expert, 1]
                    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
                        ggml_prec_set_acc(lg, GGML_PREC_F32);
                    }
                    cb_tag(lg, "ffn_moe_logits", X, "pred");
                    far_lg[d] = lg;
                    ggml_tensor * f_logits = nullptr;
                    ggml_tensor * f_probs  = build_moe_probs(rin, rec->gate[d], /* gate_inp_b = */ nullptr, /* logits_in = */ lg,
                            gating_op, X, &f_logits, /* name_tag = */ "pred");
                    ggml_tensor * f_top    = ggml_cont(ctx0, build_moe_topk(f_probs, f_logits, rec->probs_b[d], n_expert, width, X,
                            /* name_tag = */ "pred"));
                    sep_lg[d] = lg;
                    sep_pr[d] = f_probs;
                    sep_id[d] = f_top;
                    // [width, columns]; a graph whose next layer is not streamed starts with its first far column
                    cols = cols ? ggml_concat(ctx0, cols, f_top, 1) : f_top;
                }
                return cols;
            };
            // A layer whose next layer is not streamed (a VRAM layer in between, or the MTP layer) builds no
            // 1-ahead guess, but its far guesses for the streamed layers il+2 .. il+N are built all the same (the plain decode
            // guess only: not with the chance table or a two-layer lead, whose op shapes are the 1-ahead guess's)
            const bool far_only = far_on && msl->pred_il < 0 && la_nt > 0 && !mstream->chance_on() && mstream->prefetch_lead < 2;

            // --moe-stream-guess-combined 1: the layer's guesses for its la_targets in one step, bit for bit the separate
            // ops': one rms_norm of the base, one broadcast mul by the targets' stacked gains (each element the same two
            // float products as the separate fused norm, which computes scale * x * gain left to right), the router products in
            // one GGML_OP_MUL_MAT_MULTI (each row by the backend's own MUL_MAT code on that router), then the gating, the bias and
            // the top-k per row (the group top-k with a bias per row). Without --moe-stream-pred-shared the input is this layer's
            // router input for every row, as the separate guesses'. Only the plain decode guess (not with the chance table or a
            // two-layer lead), a layer with 2+ targets whose stacks were built (llama_model::load_tensors)
            const bool comb_on = far_on && mstream->la_comb && msl->la_comb_ok && la_nt >= 2 && !mstream->chance_on() &&
                                 mstream->prefetch_lead < 2;
            const bool sep_on  = !comb_on || mstream->la_gchk != 0;   // the separate ops: without the step, or for the bit check
            ggml_tensor * comb_lg = nullptr;   // [n_expert, la_nt]
            ggml_tensor * comb_pr = nullptr;
            ggml_tensor * comb_id = nullptr;   // [width, la_nt]
            if (comb_on) {
                ggml_tensor * as[llama_moe_stream::LOOKAHEAD_MAX];
                for (int32_t t = 0; t < la_nt; t++) {
                    as[t] = rec->gate[la_ds[t]];
                    GGML_ASSERT(as[t] != nullptr);
                }
                ggml_tensor * v = cur;
                if (mstream->pred_shared) {
                    GGML_ASSERT(rec->base && msl->la_gain_stack);
                    ggml_tensor * nrm = ggml_rms_norm(ctx0, rec->base, hparams.f_norm_rms_eps);
                    // Pinned as the separate guesses' norms are - build_norm tags its norm "norm" with the target
                    // layer and a decode graph forces a "norm" onto that layer's device (llama-context.cpp); here the first
                    // target's (the routers' buffer type is the same for all targets of a combined layer). Unpinned, the
                    // scheduler could run it on the CPU (which sums the squares in double, CUDA in float). Then its own name,
                    // which the bit check prints with the backend it ran on
                    cb(nrm, "norm", il + la_ds[0]);
                    ggml_format_name(nrm, "moe_comb_norm-%d", il);
                    v = ggml_mul(ctx0, msl->la_gain_stack, nrm);   // [n_embd, 1, la_nt]: row t = target t's gain x the norm
                    cb_tag(v, "ffn_moe_pred_in", il, "pred_comb");
                }
                comb_lg = ggml_mul_mat_multi(ctx0, as, la_nt, v);
                if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
                    ggml_prec_set_acc(comb_lg, GGML_PREC_F32);
                }
                cb_tag(comb_lg, "ffn_moe_logits", il, "pred_comb");
                ggml_tensor * c_logits = nullptr;
                comb_pr = build_moe_probs(v, nullptr, /* gate_inp_b = */ nullptr, /* logits_in = */ comb_lg, gating_op, il, &c_logits,
                        /* name_tag = */ "pred_comb");
                comb_id = build_moe_topk(comb_pr, c_logits, msl->la_bias_stack, n_expert, mstream->guess_width(n_tokens), il,
                        /* name_tag = */ "pred_comb");
                // --moe-stream-guess-skip-full 1 (a machine with no GPU device): the combined step and its top-k skip the rows
                // of targets that hold every expert in RAM - the shared array's address and each row's target layer in their op params
                // (ggml-cpu reads them). A GPU machine's graph is unchanged (its guess skip is skip-full on the GPU)
                if (mstream->p1.guess_skip_full && !mstream->el_has_gpu && la_nt <= 4) {
                    uint8_t * full = mstream->gsf_ptr();
                    ggml_tensor * as_t = comb_id;   // the top-k's ARGSORT under the view (and the cont)
                    while (as_t && as_t->op != GGML_OP_ARGSORT) {
                        as_t = as_t->view_src ? as_t->view_src : as_t->src[0];
                    }
                    if (as_t && as_t->ne[1] == la_nt) {
                        comb_lg->op_params[9] = GGML_CPU_GSF_MAGIC;
                        memcpy(&comb_lg->op_params[10], &full, sizeof(full));
                        as_t->op_params[2] = GGML_CPU_GSF_MAGIC;
                        memcpy(&as_t->op_params[3], &full, sizeof(full));
                        for (int32_t t = 0; t < la_nt; t++) {
                            comb_lg->op_params[4 + t]  = il + la_ds[t];   // (slots 4..7: 15 carries the layer of the GPU producer's marks)
                            as_t->op_params[5 + t]     = il + la_ds[t];
                        }
                    }
                }
                // --moe-stream-guess-skip-full-gpu 1 (a GPU machine): the same tag for the GPU's combined step and its top-k, with
                // the GPU copy of the 'full' bytes (skip-full's layout in the op params; its own magic, so a CPU never reads a GPU address)
                // (the bytes live on the VRAM pool's GPU: a guess that runs on another GPU may not read them; a CPU guess ignores the GPU tag)
                const ggml_backend_dev_t g9_rd = la_nt > 0 ? mstream->router_dev_of(il + la_ds[0]) : nullptr;
                const bool g9_here = g9_rd == nullptr || g9_rd == mstream->pool_dev;
                if (mstream->p1.guess_skip_full && mstream->p1.guess_skip_full_gpu && mstream->el_has_gpu && la_nt <= 4 &&
                        mstream->g9_dev_ptr() != nullptr && !g9_here) {
                    static bool said = false;
                    if (!said) {
                        said = true;
                        fprintf(stderr, "llama_moe_stream: --moe-stream-guess-skip-full-gpu: off for the layers whose guess runs on another GPU than "
                                "the VRAM pool's (%s): their guess reads every row (first: layer %d)\n",
                                mstream->pool_dev ? ggml_backend_dev_name(mstream->pool_dev) : "none", il);
                    }
                }
                if (mstream->p1.guess_skip_full && mstream->p1.guess_skip_full_gpu && mstream->el_has_gpu && la_nt <= 4 &&
                        mstream->g9_dev_ptr() != nullptr && g9_here) {
                    uint8_t * full = mstream->g9_dev_ptr();
                    ggml_tensor * as_t = comb_id;
                    while (as_t && as_t->op != GGML_OP_ARGSORT) {
                        as_t = as_t->view_src ? as_t->view_src : as_t->src[0];
                    }
                    if (as_t && as_t->ne[1] == la_nt) {
                        comb_lg->op_params[9] = GGML_GPU_GSF_MAGIC;
                        memcpy(&comb_lg->op_params[10], &full, sizeof(full));
                        as_t->op_params[2] = GGML_GPU_GSF_MAGIC;
                        memcpy(&as_t->op_params[3], &full, sizeof(full));
                        for (int32_t t = 0; t < la_nt; t++) {
                            comb_lg->op_params[4 + t]  = il + la_ds[t];
                            as_t->op_params[5 + t]     = il + la_ds[t];
                        }
                    }
                }
                if (!ggml_is_contiguous(comb_id)) {
                    comb_id = ggml_cont(ctx0, comb_id);
                }
                for (int32_t t = 0; t < la_nt; t++) {   // the recorder's lead d is the same tensor (a row view)
                    far_lg[la_ds[t]] = ggml_view_2d(ctx0, comb_lg, n_expert, 1, comb_lg->nb[1], (size_t) t*comb_lg->nb[1]);
                }
            }
            // TEST ONLY (MOE_STREAM_TEST_GUESS_CHECK): every row of the combined step against the separate guess for the
            // same target - the logits, the probabilities and the ids, on the host (memcmp; a difference stops the run)
            auto build_checks = [&]() {
                for (int32_t t = 0; t < la_nt; t++) {
                    const int d = la_ds[t];
                    GGML_ASSERT(sep_lg[d] && sep_pr[d] && sep_id[d]);
                    const int64_t width_c = comb_id->ne[0];
                    ggml_tensor * args[6] = {
                        sep_lg[d], sep_pr[d], sep_id[d],
                        ggml_view_2d(ctx0, comb_lg, n_expert, 1, comb_lg->nb[1], (size_t) t*comb_lg->nb[1]),
                        ggml_view_2d(ctx0, comb_pr, n_expert, 1, comb_pr->nb[1], (size_t) t*comb_pr->nb[1]),
                        ggml_view_2d(ctx0, comb_id, width_c, 1, comb_id->nb[1], (size_t) t*comb_id->nb[1]),
                    };
                    ggml_tensor * chk = ggml_custom_4d(ctx0, GGML_TYPE_I32, 1, 1, 1, 1, args, 6, llama_moe_stream_guess_check, 1,
                            mstream->gchk_ud(il, d));
                    ggml_format_name(chk, "moe_guess_check-%d-d%d", il, d);
                    ggml_build_forward_expand(gf, chk);
                }
            };

            if (mstream->prefetch_on() && msl->pred_il >= 0 && gate_inp_next) {
                // name_tag "pred": these tensors carry the *predicted* ids of the next layer, and
                // a trace tool that keys on the name must not record them as that layer's real ones
                ggml_tensor * pred_logits = nullptr;   // the next layer selects on its logits, too
                // pred_in is the residual plus this layer's shared expert, normed with the
                // *next* layer's gain (Colibri's shared-expert correction).
                // Without it the prediction runs on this layer's own router input, whose gain
                // is the next layer's only if the two ffn_norm gains are the same (measured
                // that ratio as 1.04 +- 0.04 on average, i.e. that part is worth almost nothing here).
                ggml_tensor * pred_cur = pred_in ? pred_in : cur;
                ggml_tensor * pred_probs = nullptr;
                if (sep_on) {
                    pred_probs = build_moe_probs(pred_cur, gate_inp_next, /* gate_inp_b = */ nullptr,
                            /* logits_in = */ nullptr, gating_op, msl->pred_il, &pred_logits,
                            /* name_tag = */ "pred");
                    sep_lg[1] = pred_logits;
                    sep_pr[1] = pred_probs;
                }
                if (comb_on) {   // row 0 of the combined step is this guess (la_targets starts with distance 1 here)
                    GGML_ASSERT(la_ds[0] == 1);
                    pred_logits = ggml_view_2d(ctx0, comb_lg, n_expert, 1, comb_lg->nb[1], 0);
                    pred_probs  = ggml_view_2d(ctx0, comb_pr, n_expert, 1, comb_pr->nb[1], 0);
                }
                pred_lg = pred_logits;
                // With --moe-stream-prefetch-lead 2 this tensor is the *correction* of the early
                // guess for this layer, so it covers prefetch_m2 experts and the engine cancels the
                // early reads it drops; with lead 1 it is the engine's own one-layer-ahead guess.
                // With a chance table, a decode graph's guess is the router's top 16 in its own order
                // (the order the table was measured in) with each candidate's sigmoid score, and the engine reads the
                // candidates whose measured chance is high enough instead of a fixed top prefetch_m
                const bool chance16 = mstream->chance_on() && n_tokens == 1;
                // With --moe-stream-guess-top K a graph of one token (a decode word: the waiting list's guess) is K wide
                const uint32_t width = chance16 ? (uint32_t) llama_moe_stream::P1_N_CAND
                        : mstream->prefetch_lead >= 2 && mstream->prefetch_m2 > 0
                        ? mstream->prefetch_m2 : mstream->guess_width(n_tokens);
                if (sep_on) {
                    pred = ggml_cont(ctx0, build_moe_topk(sep_pr[1], sep_lg[1], exp_probs_b_next,
                                n_expert, width, msl->pred_il, /* name_tag = */ "pred"));
                    sep_id[1] = pred;
                    if (chance16) {
                        pred_p = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, pred_probs, 1, n_expert, n_tokens), pred);   // [1, 16, 1]
                    }

                    // --moe-stream-lookahead N (a graph of one token): the far guesses for the layers il+2 .. il+N
                    // (build_far, above)
                    if (far_on) {
                        pred = build_far(pred, width);
                    }
                }
                if (comb_on) {   // the combined step's ids are the guess input (and the recorder's leads its rows)
                    if (mstream->la_gchk != 0) {
                        build_checks();
                    }
                    for (int32_t t = 0; t < la_nt; t++) {
                        far_lg[la_ds[t]] = ggml_view_2d(ctx0, comb_lg, n_expert, 1, comb_lg->nb[1], (size_t) t*comb_lg->nb[1]);
                    }
                    pred = comb_id;
                }
                // The GPU producer of --moe-stream-needed-early - in a one-token graph, the guess of a
                // layer that posts to the helper is marked: the GPU backend writes it into the layer's pinned slot, then the call
                // counter, then records the layer's "guess done" event, right after this node (the remap op's input is unchanged)
                // (only in the graph of a one-token decode call with the GPU producer; graph reuse keys on it, llama-context)
                // (on the FINAL guess tensor - the 1-ahead top-k, plus the far columns or the combined step's ids -
                // of N columns x the slot's width; only a layer whose targets' routers are all on the producer's GPU)
                if (mstream->ne_gpu_graph() && n_tokens == 1 && msl->ne_post_on && msl->ne_gpu_ok && !chance16 &&
                        !(mstream->prefetch_lead >= 2) &&
                        (int64_t) mstream->ne_width*mstream->la_cols_of(il) == ggml_nelements(pred) &&
                        pred->type == GGML_TYPE_I32 && ggml_is_contiguous(pred)) {   // what the producer copies
                    pred->flags |= (int32_t) LLAMA_MOE_STREAM_TENSOR_GUESS_MARK;
                    pred->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = il;
                    moe_pin(pred, mstream->ne_guess_dev(il));   // its event's GPU (with several GPUs the scheduler could move it)
                }

                // Measurement only (MOE_STREAM_LEAD2_COST=1): the 2-ahead guess that 2-ahead reads would need - the
                // router of the layer two ahead on this layer's guess input (one RMS norm standing in for that layer's own norm),
                // its top 16 and their scores - computed and left unused, so that its device cost can be measured on its own
                static const bool lead2_cost = getenv("MOE_STREAM_LEAD2_COST") && atoi(getenv("MOE_STREAM_LEAD2_COST")) != 0;
                if (lead2_cost && chance16 && gate_inp_next2) {
                    ggml_tensor * l2_logits = nullptr;
                    ggml_tensor * l2_in     = ggml_rms_norm(ctx0, pred_cur, hparams.f_norm_rms_eps);
                    ggml_tensor * l2_probs  = build_moe_probs(l2_in, gate_inp_next2, /* gate_inp_b = */ nullptr,
                            /* logits_in = */ nullptr, gating_op, msl->pred_il, &l2_logits, /* name_tag = */ "pred2cost");
                    ggml_tensor * l2_top    = ggml_cont(ctx0, build_moe_topk(l2_probs, l2_logits, exp_probs_b_next2,
                            n_expert, llama_moe_stream::P1_N_CAND, msl->pred_il, /* name_tag = */ "pred2cost"));
                    ggml_build_forward_expand(gf, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, l2_probs, 1, n_expert, n_tokens), l2_top));
                }

                // The guess measured against the next layer's real choice. The prefetch only holds
                // prefetch_m ids, which is 4 by default, so recall@8 needs its own top-8 of the same
                // predicted scores. It is copied into a host tensor the remap op owns and scheduled
                // before it: reading the device tensor from inside the op (ggml_backend_tensor_get)
                // deadlocked the run, and reading it a node later would race the copy.
                // The buffer holds recall_tokens tokens; a bigger graph (a ubatch or a layer-major call above it) skips
                // the measurement instead of writing past the buffer (the remap op then counts nothing for it: pred8 = nullptr)
                if (mstream->pred_recall && mstream->recall_buf() && cur->ne[1] <= llama_moe_stream::recall_tokens) {
                    ggml_tensor * top8 = ggml_cont(ctx0, build_moe_topk(pred_probs, pred_logits,
                                exp_probs_b_next, n_expert, n_expert_used, msl->pred_il,
                                /* name_tag = */ "pred_recall"));

                    pred_tr = ggml_view_2d(ctx0, mstream->recall_buf(), n_expert_used, cur->ne[1],
                            n_expert_used*sizeof(int32_t), 0);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, top8, pred_tr));
                }

                // Record the vector this guess is made on, so that a learned corrector can be
                // trained on what the engine's guess actually sees (the normed residual + shared output
                // with --moe-stream-pred-shared) instead of the plain router input. The guess's own top-k
                // goes with it, so the file can be checked against the engine without the model. Both
                // copies land in ONE fixed host slot at a per-layer offset: a captured CUDA graph bakes
                // the destination address in, so a per-word offset would make every replay write word 0.
                // n_tokens == 1 only, because the slot holds one word and only generated words are
                // recorded - a prompt ubatch would need one slot per token.
                if (mstream->pred_input_vec_tensor() && n_tokens == 1) {
                    ggml_tensor * vdst = ggml_view_2d(ctx0, mstream->pred_input_vec_tensor(),
                            (int64_t) mstream->n_embd, 1, (int64_t) mstream->n_embd*sizeof(float),
                            mstream->pred_input_vec_offset(il)*sizeof(float));

                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, pred_cur), vdst));

                    if (rec && rec->in[2] && mstream->pred_input_vec2_tensor()) {
                        mstream->mark_pred_input_vec2();
                        ggml_tensor * v2dst = ggml_view_2d(ctx0, mstream->pred_input_vec2_tensor(),
                                (int64_t) mstream->n_embd, 1, (int64_t) mstream->n_embd*sizeof(float),
                                mstream->pred_input_vec_offset(il)*sizeof(float));

                        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, rec->in[2]), v2dst));
                    }

                    ggml_tensor * gtop = ggml_cont(ctx0, build_moe_topk(pred_probs, pred_logits,
                                exp_probs_b_next, n_expert, n_expert_used, msl->pred_il,
                                /* name_tag = */ "pred_input_topk"));
                    ggml_tensor * gdst = ggml_view_2d(ctx0, mstream->pred_input_guess_tensor(),
                            n_expert_used, 1, n_expert_used*sizeof(int32_t),
                            mstream->pred_input_guess_offset(il)*sizeof(int32_t));

                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, gtop, gdst));
                }

                // --moe-stream-pred-scores: the guess's scores for EVERY expert, sigmoid(W.h_hat) + bias, copied
                // into the host buffer the engine owns. One slot per generated word and model layer,
                // decode graphs only (a multi-token graph would need one word per token). The copy is
                // the last thing this graph does with the prediction, so it does not change the run.
                if (mstream->pred_scores_buf_tensor() && n_tokens == 1) {
                    ggml_tensor * z = ggml_cont(ctx0, exp_probs_b_next ? ggml_add(ctx0, pred_probs, exp_probs_b_next) : pred_probs);   // a model without a selection bias has no bias to add
                    ggml_tensor * dst = ggml_view_2d(ctx0, mstream->pred_scores_buf_tensor(),
                            n_expert, 1, n_expert*sizeof(float),
                            mstream->pred_scores_offset(il)*sizeof(float));

                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, z, dst));
                }
            } else if (far_only) {
                if (sep_on) {
                    pred = build_far(nullptr, mstream->guess_width(n_tokens));   // the far guesses alone
                }
                if (comb_on) {   // the combined step (its rows are the far targets only here)
                    if (mstream->la_gchk != 0) {
                        build_checks();
                    }
                    for (int32_t t = 0; t < la_nt; t++) {
                        far_lg[la_ds[t]] = ggml_view_2d(ctx0, comb_lg, n_expert, 1, comb_lg->nb[1], (size_t) t*comb_lg->nb[1]);
                    }
                    pred = comb_id;
                }
            }

            // --moe-stream-record: the side-by-side recording. Decode graphs only (one token: the host
            // slots hold one word, like the recall counting's and the guess input recording's). The routing weights this layer really uses, and the
            // raw logits of the routers of the next four layers on the guess input of THIS layer - lead d is
            // the router of layer il+d. The d = 1 logits are the very tensor the engine's own guess is made
            // from when prefetch is on (so the recording and the engine cannot disagree); the others are
            // built here the way lead 1 is: with --moe-stream-pred-shared the residual + shared expert normed
            // with the TARGET router's ffn_norm gain, without it this layer's own router input. All of it is
            // copied into the engine's host slots and nothing is handed to the prefetcher.
            if (mstream->rec_on() && n_tokens == 1) {
                const int64_t k_rec = n_expert_used;

                ggml_tensor * wdst = ggml_view_1d(ctx0, mstream->rec_w_tensor(), k_rec,
                        mstream->rec_w_offset(il)*sizeof(float));

                rec_copies.emplace_back(ggml_cont(ctx0, weights), wdst);

                // TEST ONLY (MOE_STREAM_TEST_REC_RLOGITS=1): the REAL router logits of this layer (what
                // its own router produced on its own input when it ran), next to the guess logits - the ridge's
                // training target. Same fixed host slot pattern; the engine never reads it.
                if (mstream->rec_rlogits() && logits != nullptr) {
                    ggml_tensor * rdst = ggml_view_1d(ctx0, mstream->rec_rlogits_tensor(), n_expert,
                            mstream->rec_rlogits_offset(il)*sizeof(float));

                    rec_copies.emplace_back(ggml_cont(ctx0, logits), rdst);
                }

                ggml_tensor * lead[llama_moe_stream::REC_LEADS + 1] = { nullptr };

                for (int d = 1; d <= llama_moe_stream::REC_LEADS; d++) {
                    const int X = il + d;

                    if (!rec || !mstream->rec_lead_avail(il, d) || rec->gate[d] == nullptr) {
                        continue;
                    }

                    if (d == 1 && pred_lg) {
                        lead[d] = pred_lg;
                        continue;
                    }
                    if (far_lg[d]) {   // the far guess's own logits - the recording and the guess are one tensor
                        lead[d] = far_lg[d];
                        continue;
                    }

                    ggml_tensor * rin = cur;

                    if (mstream->pred_shared) {
                        if (d == 1 && pred_in) {
                            rin = pred_in;
                        } else if (rec->in[d]) {   // the model's own guess input
                            rin = rec->in[d];
                        } else if (rec->base && rec->norm[d]) {
                            rin = build_norm(rec->base, rec->norm[d], nullptr, LLM_NORM_RMS, X);
                        } else {
                            continue;
                        }
                    }

                    ggml_tensor * lg = build_lora_mm(rec->gate[d], rin);   // [n_expert, 1]
                    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
                        ggml_prec_set_acc(lg, GGML_PREC_F32);
                    }
                    char rec_tag[8];
                    snprintf(rec_tag, sizeof(rec_tag), "d%d", d);
                    cb_tag(lg, "rec_logits", il, rec_tag);
                    lead[d] = lg;
                }

                // one copy per run of consecutive leads: [n_expert, run] into the slot of the first of them
                for (int d = 1; d <= llama_moe_stream::REC_LEADS; ) {
                    if (!lead[d]) {
                        d++;
                        continue;
                    }

                    int      d1  = d;
                    ggml_tensor * cat = ggml_cont(ctx0, lead[d]);

                    while (d1 + 1 <= llama_moe_stream::REC_LEADS && lead[d1 + 1]) {
                        d1++;
                        cat = ggml_concat(ctx0, cat, lead[d1], 1);
                    }

                    ggml_tensor * ldst = ggml_view_2d(ctx0, mstream->rec_logits_tensor(), n_expert, d1 - d + 1,
                            n_expert*sizeof(float), mstream->rec_logits_offset(il, d)*sizeof(float));

                    rec_copies.emplace_back(cat, ldst);
                    d = d1 + 1;
                }
            }

            // The early guess of a two-layer lead. Layer L+2's router runs on *this* layer's
            // router input, so nothing delays it - in particular not this layer's shared expert, which
            // is the input the --moe-stream-pred-shared correction has to wait for. The layer in between corrects it (the
            // block above, with role 2), which is where the better input is used.
            ggml_tensor * pred2_early = nullptr;   // the op whose output --early-order waits for

            if (mstream->prefetch_on() && mstream->prefetch_lead >= 2 && msl->pred_il2 >= 0 &&
                gate_inp_next2 && gate_inp_next) {
                ggml_tensor * pred2_logits = nullptr;
                ggml_tensor * pred2_probs = build_moe_probs(cur, gate_inp_next2, /* gate_inp_b = */ nullptr,
                        /* logits_in = */ nullptr, gating_op, msl->pred_il2, &pred2_logits,
                        /* name_tag = */ "pred2");
                ggml_tensor * pred2 = ggml_cont(ctx0, build_moe_topk(pred2_probs, pred2_logits,
                            exp_probs_b_next2, n_expert, mstream->prefetch_m, msl->pred_il2,
                            /* name_tag = */ "pred2"));
                // Lead 3: with a three-layer chain this step is the *middle* one for the layer two
                // ahead (the layer before already guessed it from its own router input), so it refines
                // without cancelling: the layer in between still has the better input to cancel with.
                const int32_t role2 = mstream->prefetch_lead >= 3 ? 3 : 1;
                pred2 = ggml_map_custom1(ctx0, pred2, llama_moe_stream_pred2, 1,
                        (void *) mstream->pred2_ctx_of(il, role2));
                ggml_build_forward_expand(gf, pred2);
                pred2_early = pred2;
            }

            // Lead 3: the chain's first step for the layer *three* ahead, again on this layer's own
            // router input, so nothing delays it. Two further steps correct it: this layer's two-ahead
            // prediction is its middle refinement (above), and the layer in between's one-ahead
            // prediction is the final correction, which is the one that cancels.
            if (mstream->prefetch_on() && mstream->prefetch_lead >= 3 && msl->pred_il3 >= 0 &&
                gate_inp_next3) {
                ggml_tensor * pred3_logits = nullptr;
                ggml_tensor * pred3_probs = build_moe_probs(cur, gate_inp_next3, /* gate_inp_b = */ nullptr,
                        /* logits_in = */ nullptr, gating_op, msl->pred_il3, &pred3_logits,
                        /* name_tag = */ "pred3");
                ggml_tensor * pred3 = ggml_cont(ctx0, build_moe_topk(pred3_probs, pred3_logits,
                            exp_probs_b_next3, n_expert, mstream->prefetch_m, msl->pred_il3,
                            /* name_tag = */ "pred3"));
                pred3 = ggml_map_custom1(ctx0, pred3, llama_moe_stream_pred2, 1,
                        (void *) mstream->pred3_ctx_of(il));
                ggml_build_forward_expand(gf, pred3);
                pred2_early = pred3;   // --early-order: with a chain of three this is the first guess
            }

            ggml_tensor * ids_cont = ig_ids ? ig_ids : ggml_cont(ctx0, selected_experts); // the top_k output is a view

            // The "router done" mark of the GPU producer. In a one-token graph with the GPU producer, a layer whose
            // router runs on the producer's GPU marks its router's ids with the router input (what the CPU experts read) as src[1]:
            // the GPU backend copies both into the layer's pinned router slot right after this node, then the call's number, then
            // records "router done". Expanded here, before the guess (and, with --moe-stream-pred-shared, the shared expert it
            // reads), so in the GPU split they run after the producer, while the CPU split - which then no longer takes the guess
            // (the helper gets it from the guess slot) - already runs from the slot. Only when the guess is absent (the last
            // streamed layer) or marked, so that nothing else of it crosses into the CPU split.
            bool guess_left = false;   // the remap op does not take the guess
            ggml_tensor * vram_out_early = nullptr;   // the device pass built in the layer's GPU launch
            if (mstream->ne_gpu_graph() && n_tokens == 1 && mstream->ne_rd_layer_on(il) && !pred_p &&
                    (pred == nullptr || (pred->flags & (int32_t) LLAMA_MOE_STREAM_TENSOR_GUESS_MARK)) &&
                    mstream->rd_fits(il, ids_cont, cur_in)) {   // else no mark, the plain path for this layer
                ids_cont->flags |= (int32_t) LLAMA_MOE_STREAM_TENSOR_ROUTER_MARK;
                ids_cont->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = il;
                ids_cont->src[1] = cur_in;   // CONT reads only src[0]; the producer copies src[1] too, and it orders cur_in first
                moe_pin(ids_cont, mstream->ne_router_dev(il));   // the mark runs on its router's GPU, where its event is (several GPUs)
                ggml_build_forward_expand(gf, ids_cont);
                if (pred) {
                    ggml_build_forward_expand(gf, pred);
                    guess_left = true;
                }
                if (weights_deferred) {   // Behind the mark and the guess, in front of the VRAM pass that multiplies by them
                    ggml_build_forward_expand(gf, weights);
                    weights_expanded = true;
                }
                // --moe-stream-vram-parallel 1 or 2 (this layer's pass moves, vp_in_launch): the VRAM pool's device pass in this layer's GPU
                // launch, after the router producer ("router done") and the guess, so the GPU runs it while the CPU split - which
                // starts from the router slot - computes the RAM experts. Its ids come from the layer's table on the GPU: row e of
                // the table = the expert's place k times when held, else the zero slices z0 + 0 .. z0 + k - 1; get_rows of the
                // router's k ids gives a k x k block whose diagonal is the device ids (pair j's own column, so no token names a slice
                // twice). The join stays where it was: ffn_moe_out = the host's weighted sum + this pass's sum, after the CPU split.
                if (mstream->vp_in_launch(il) && mstream->vram_serves(il)) {
                    const bool vx_ok = !weight_before_ffn && !up_exps_b && !gate_exps_b && !down_exps_b && !gate_up_exps &&
                                       gate_exps && up_exps && type_op == LLM_FFN_SILU && !up_exps_s && !gate_exps_s &&
                                       !down_exps_s && !gate_up_exps_b;
                    if (!vx_ok) {
                        GGML_ABORT("--moe-stream-vram-parallel: layer %d's MoE has no device pass (see --moe-stream-vram-experts)", il);
                    }
                    size_t        off = 0;
                    ggml_tensor * tab = mstream->vp_table_of(il, off);
                    tab = ggml_view_2d(ctx0, tab, tab->ne[0], tab->ne[1], tab->nb[1], off);    // this layer's [k, n_expert]
                    ggml_tensor * rows = ggml_get_rows(ctx0, tab, ids_cont);                    // [k, k]: row j = table[ids[j]]
                    GGML_ASSERT(rows->ne[0] >= rows->ne[1] && "vram-parallel: the id table is narrower than the layer's k");
                    cb(rows, "ffn_moe_vp_rows", il);
                    ggml_tensor * diag = ggml_view_2d(ctx0, rows, 1, rows->ne[1], (rows->ne[0] + 1)*ggml_element_size(rows), 0);
                    diag = ggml_cont(ctx0, diag);
                    if (mstream->ne_router_dev(il) != mstream->pool_dev) {   // several GPUs: the pass on the pool's GPU, the ids copied there
                        moe_pin(rows, mstream->pool_dev);
                        moe_pin(diag, mstream->pool_dev);
                    }
                    diag = ggml_reshape_2d(ctx0, diag, rows->ne[1], 1);       // [k, 1]: pair j's device id
                    cb(diag, "ffn_moe_topk_vram", il);
                    vram_out_early = build_vram_chain(diag, nullptr, nullptr, true);   // (absolute places)
                    ggml_build_forward_expand(gf, vram_out_early);
                }
                run_moe_window();   // the builder's window nodes, last
            } else if (lm_phase != 2 && mstream->ps_wants(n_tokens) && mstream->vp_in_launch(il) && !pred_p && mstream->vram_serves(il) &&
                    (pred == nullptr || !(pred->flags & (int32_t) LLAMA_MOE_STREAM_TENSOR_GUESS_MARK)) &&
                    mstream->ps_fits(il, ids_cont, cur_in, pred, weights)) {
                // --moe-stream-prompt-split 1: a CPU-wave prompt ubatch - the router mark hands the CPU split
                // the router's ids, its input and the guess through the shared slot (the scheduler's input hook), so the CPU split starts
                // without waiting for the GPU stream, and the VRAM pass is built here, in the layer's GPU launch: the GPU runs it while
                // the CPU computes the RAM experts. Its ids: the layer's table on the GPU, the diagonal per token ([k, k, n_tok] -> [k, n_tok])
                ids_cont->flags |= (int32_t) LLAMA_MOE_STREAM_TENSOR_ROUTER_MARK;
                ids_cont->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = il;
                ids_cont->src[1] = cur_in;   // the CPU split's other inputs, packed in this order (the producer stops at the first null)
                ids_cont->src[2] = pred ? pred : weights;
                ids_cont->src[3] = pred ? weights : nullptr;
                moe_pin(ids_cont, mstream->ne_router_dev(il));   // its guess source may come from the next layer's GPU: copied here, not moved there
                ggml_build_forward_expand(gf, ids_cont);
                weights_expanded = true;   // (a source of the mark: computed before it)
                const bool vx_ok = !weight_before_ffn && !up_exps_b && !gate_exps_b && !down_exps_b && !gate_up_exps &&
                                   gate_exps && up_exps && type_op == LLM_FFN_SILU && !up_exps_s && !gate_exps_s &&
                                   !down_exps_s && !gate_up_exps_b;
                if (!vx_ok) {
                    GGML_ABORT("--moe-stream-prompt-split: layer %d's MoE has no device pass (see --moe-stream-vram-experts)", il);
                }
                size_t        off = 0;
                ggml_tensor * tab = mstream->vp_table_of(il, off);
                tab = ggml_view_2d(ctx0, tab, tab->ne[0], tab->ne[1], tab->nb[1], off);    // this layer's [k, n_expert]
                ggml_tensor * rows = ggml_get_rows(ctx0, tab, ggml_reshape_1d(ctx0, ids_cont, ggml_nelements(ids_cont)));   // [k, k x n_tok]
                GGML_ASSERT(rows->ne[0] >= ids_cont->ne[0] && "prompt-split: the id table is narrower than the layer's k");
                rows = ggml_reshape_3d(ctx0, rows, rows->ne[0], ids_cont->ne[0], ids_cont->ne[1]);   // rows[:, j, t] = table[ids[j, t]]
                cb(rows, "ffn_moe_ps_rows", il);
                ggml_tensor * diag = ggml_view_3d(ctx0, rows, 1, rows->ne[1], rows->ne[2], (rows->ne[0] + 1)*ggml_element_size(rows),
                                                  rows->nb[2], 0);                          // [1, k, n_tok]: pair (j, t)'s own column j
                diag = ggml_cont(ctx0, diag);
                if (mstream->ne_router_dev(il) != mstream->pool_dev) {   // several GPUs: the pass on the pool's GPU, the ids copied there
                    moe_pin(rows->view_src ? rows->view_src : rows, mstream->pool_dev);   // (the get_rows under the reshape)
                    moe_pin(diag, mstream->pool_dev);
                }
                diag = ggml_reshape_2d(ctx0, diag, rows->ne[1], rows->ne[2]);   // [k, n_tok]: the device ids
                cb(diag, "ffn_moe_topk_vram", il);
                vram_out_early = build_vram_chain(diag, nullptr, nullptr, true);   // (absolute places)
                ggml_build_forward_expand(gf, vram_out_early);
            }
            if (weights_deferred && !weights_expanded) {   // This layer got no mark - the chain after the guess blocks, still before the remap op and the other CPU nodes
                ggml_build_forward_expand(gf, weights);
                weights_expanded = true;
            }
            run_moe_window();   // a layer without a mark - the builder's window nodes in front of the remap op
            if (mstream->ne_gpu_graph() && n_tokens == 1) {
                msl->vp_built        = vram_out_early != nullptr;   // for the end-of-run line
                msl->ne_guess_left   = guess_left;   // read by the remap op of this graph
                msl->ne_guess_marked = pred && (pred->flags & (int32_t) LLAMA_MOE_STREAM_TENSOR_GUESS_MARK);
            }

            // A copy into a host slot runs on the CPU (its destination lives there), so every copy is
            // expanded here, in front of the remap op, and AFTER every device node it or the remap op needs
            // (the ids, the guess, every recorded source): expanding one copy expands its sources just before
            // it, so a copy-by-copy loop made device, CPU, device, CPU runs - two extra switches per layer
            // Sources first, then all copies: one run of device nodes, one of CPU nodes.
            if (!rec_copies.empty()) {
                ggml_build_forward_expand(gf, ids_cont);

                if (pred) {
                    ggml_build_forward_expand(gf, pred);
                }

                if (pred_p) {
                    ggml_build_forward_expand(gf, pred_p);   // a device op, so before the host copies too
                }

                for (const auto & c : rec_copies) {
                    ggml_build_forward_expand(gf, c.first);
                }

                for (const auto & c : rec_copies) {
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, c.first, c.second));
                }
            }

            // --moe-stream-early-order: nothing in the data dependencies makes the early-guess op
            // run *before* this layer's expert work, so the scheduler is free to issue it whenever the
            // node becomes ready - which is the difference between a guess that arrives in time and one
            // that arrives after the layer that wanted it. Making the ids depend on the op's output
            // forces the order. The dependency is exact: sum the guess's ids (I32), square them (so the
            // value is exactly 0 whatever they were) and add that scalar to the ids, which ggml_add
            // repeats over the tensor. Integer arithmetic, so the matmul sees the same indices.
            //
            // (the ids themselves are left to ggml_add, which makes the contiguous copy the
            // ggml_cont below used to make; adding a tensor of zeroes to them changes nothing)
            if (mstream->early_order && pred2_early) {
                // The guess op's output summed and scaled to an *exact* zero, then added to the ids,
                // so the matmul sees the same indices but cannot be scheduled before the guess ran.
                // Cast because a F32 sum of an I32 tensor has no CUDA kernel; the sum is over expert
                // indices, so it is finite and 0 * it is exactly 0 whatever it is.
                ggml_tensor * sm = ggml_sum(ctx0, ggml_cast(ctx0, pred2_early, GGML_TYPE_F32));
                ggml_tensor * z0 = ggml_scale(ctx0, sm, 0.0f);
                selected_experts = ggml_add(ctx0, selected_experts, ggml_cast(ctx0, z0, GGML_TYPE_I32));
            }

            // the zero slot only zeroes the matmul output: a per-expert bias would survive it, and
            // llama4 scales the input by the weights before the matmuls, so neither may use waves
            const bool can_wave = !weight_before_ffn && !up_exps_b && !gate_exps_b && !down_exps_b;
            const int  n_waves  = can_wave ? mstream->n_waves(n_tokens) : 1;
            // Tail waves: GPU waves of a graph whose tail is vacated (separate gate / up / down pools, no per-expert scales)
            // (copy-ahead: also a one-wave prompt when every expert is in RAM - the engine's copy instead of the scheduler's whole pool)
            tailw_w = (n_waves > 1 || mstream->ca_one_wave) && !gate_up_exps && gate_exps && up_exps && !up_exps_s && !gate_exps_s && !down_exps_s &&
                   mstream->tailw_graph(lm_phase == 2 ? lm_chunk : n_tokens);

            ggml_tensor * moe_out = nullptr;
            ggml_tensor * vram_ids = nullptr;

            // The expert graph of a layer-major call - each wave's matmuls in chunks of lm_chunk tokens, each chunk's
            // result added into its rows of the call's residual (lm_hid) or routed-expert accumulator (lm_moe); the chain orders the waves
            ggml_tensor * lm_acc = lm_moe ? lm_moe : lm_hid;
            const int64_t lm_nc = lm_phase == 2 ? (n_tokens + lm_chunk - 1)/lm_chunk : 0;
            std::vector<ggml_tensor *> lm_h_cur((size_t) lm_nc, nullptr);
            const auto lm_add_chunk = [&](int64_t c, ggml_tensor * add) {
                const int64_t c0 = c*lm_chunk, nc = std::min<int64_t>(lm_chunk, n_tokens - c0);
                ggml_tensor * dst = ggml_view_2d(ctx0, lm_acc, n_embd, nc, lm_acc->nb[1], c0*lm_acc->nb[1]);
                if (lm_moe && !lm_h_cur[(size_t) c]) {
                    lm_h_cur[(size_t) c] = ggml_cpy(ctx0, add, dst);
                } else {
                    ggml_tensor * prev     = lm_h_cur[(size_t) c] ? lm_h_cur[(size_t) c] : dst;
                    ggml_tensor * prev_f32 = prev->type == GGML_TYPE_F32 ? prev : ggml_cast(ctx0, prev, GGML_TYPE_F32);
                    lm_h_cur[(size_t) c]   = ggml_cpy(ctx0, ggml_add(ctx0, prev_f32, add), dst);
                }
                ggml_build_forward_expand(gf, lm_h_cur[(size_t) c]);
            };
            const auto lm_chunk_views = [&](int64_t c, ggml_tensor * ids_all, ggml_tensor ** ids_c, ggml_tensor ** x_c, ggml_tensor ** w_c) {
                const int64_t c0 = c*lm_chunk, nc = std::min<int64_t>(lm_chunk, n_tokens - c0);
                *ids_c = ggml_view_2d(ctx0, ids_all, ids_all->ne[0], nc, ids_all->nb[1], c0*ids_all->nb[1]);
                *x_c   = ggml_view_2d(ctx0, lm_x, n_embd, nc, lm_x->nb[1], c0*lm_x->nb[1]);
                *w_c   = ggml_view_3d(ctx0, weights, 1, weights->ne[1], nc, weights->nb[1], weights->nb[2], c0*weights->nb[2]);
            };

            // With GPU waves, each wave's ids (and the VRAM ids) go to the GPU ONCE through a cont on the
            // GPU backend and the chunks take views of that - not one async host copy per chunk view, which the next wave's remap op
            // could overwrite before it ran (--moe-stream-async-h2d)
            ggml_backend_t lm_gpu_be = nullptr;   // (the VRAM pool's GPU: the wave matmuls of the RAM pools run there)
            if (lm_phase == 2 && mstream->gpu_waves > 0 && lm_chunk >= (int64_t) mstream->gpu_waves && !mstream->cpu_repack_on.load() &&
                    cparams.op_offload) {
                lm_gpu_be = moe_backend_of(mstream->pool_dev);
            }
            const auto lm_ids_on_dev = [&](ggml_tensor * ids_all) -> ggml_tensor * {
                if (!lm_gpu_be) {
                    return ids_all;
                }
                ggml_tensor * t = ggml_cont(ctx0, ids_all);
                ggml_backend_sched_set_tensor_backend(sched, t, lm_gpu_be);
                return t;
            };

            for (int w = 0; w < n_waves; w++) {
                auto * wctx = mstream->wave_ctx_of(il, w);

                wctx->pred8 = pred_tr;   // read by the remap op of this layer, scored by the next

                ids_gemm = ig_a   // op B (with the guess, or without one when none was built in this graph)
                    ? (pred && !guess_left && !pred_p
                       ? ggml_map_custom3(ctx0, ids_cont, pred, ig_a, llama_moe_stream_remap_b, 1, wctx)
                       : ggml_map_custom2(ctx0, ids_cont, ig_a, llama_moe_stream_remap_b1, 1, wctx))
                    : pred_p
                    ? ggml_map_custom3(ctx0, ids_cont, pred, pred_p, llama_moe_stream_remap3, 1, wctx)   // with the chance table
                    : pred && !guess_left
                    ? ggml_map_custom2(ctx0, ids_cont, pred, llama_moe_stream_remap2, 1, wctx)
                    : ggml_map_custom1(ctx0, ids_cont, llama_moe_stream_remap,  1, wctx);

                // In a window call a one-wave piece too - the window moves room above the start
                // size, which the start-size view a GPU offload copies does not hold (256+ slots per layer: a 32-row piece is one wave)
                // and in every call where the tail rule says so (RAM slots >= 8 x the offload minimum batch:
                // a one-wave ubatch's GPU copy would be a whole start-size view that the reserve does not cover)
                if (n_waves > 1 || mstream->cpu_repack_on.load() || (mstream->win_host_call && mstream->el_has_gpu) || (n_waves == 1 && mstream->onewave_host(n_tokens))) {
                    // the experts of this wave are loaded while the graph computes, so the device may
                    // not run their matmuls: the pool it would copy once per graph is filled wave by
                    // wave. llama_context keeps ops that read host weights on the host for this graph.
                    ids_gemm->flags |= (uint32_t) LLAMA_MOE_STREAM_TENSOR_WAVE;
                }
                if (tailw_w) {
                    ids_gemm->flags |= (uint32_t) LLAMA_MOE_STREAM_TENSOR_TAIL;   // the op gathers this wave's experts
                }

                cb(ids_gemm, "ffn_moe_topk_stream", il);

                // The VRAM expert pool's device pass, once per layer (wave 0). Its ids op is a CPU op and is expanded
                // right after the host ids op (then both ids ops and the host matmuls are one CPU region and
                // the device chain joins the CUDA region that already runs this layer's weighting, aggregation and shared
                // expert - expanded on its own it costs a graph split per layer). The chain is expanded after the waves.
                if (w == 0 && mstream->vram_serves(il) && vram_out_early == nullptr) {
                    const bool vx_ok = can_wave && !gate_up_exps && gate_exps && up_exps && type_op == LLM_FFN_SILU &&
                                       !up_exps_s && !gate_exps_s && !down_exps_s && !gate_up_exps_b;
                    if (!vx_ok) {
                        GGML_ABORT("--moe-stream-vram-experts: layer %d's MoE (merged gate_up %d, gate %d, op %d, scales %d/%d/%d, "
                                "biases or weights before the FFN %d) has no device pass; the VRAM expert pool supports a gated "
                                "SiLU MoE with separate gate and up tensors and no per-expert scale or bias", il,
                                gate_up_exps != nullptr, gate_exps != nullptr, (int) type_op, up_exps_s != nullptr,
                                gate_exps_s != nullptr, down_exps_s != nullptr, (int) !can_wave);
                    }
                    ggml_build_forward_expand(gf, ids_gemm);

                    ggml_tensor * ids_vram = ggml_map_custom1(ctx0, ids_cont, llama_moe_stream_remap_vram, 1,
                            mstream->wave_ctx_of(il, 0));
                    cb(ids_vram, "ffn_moe_topk_vram", il);
                    ggml_build_forward_expand(gf, ids_vram);
                    vram_ids = ids_vram;
                }

                if (lm_phase == 2) {   // this wave's chunks, all before the next wave's remap op
                    ggml_tensor * ids_dev = lm_ids_on_dev(ids_gemm);
                    for (int64_t c = 0; c < lm_nc; c++) {
                        ggml_tensor * ids_c, * x_c, * w_c;
                        lm_chunk_views(c, ids_dev, &ids_c, &x_c, &w_c);
                        ca_mark_wave = c + 1 == lm_nc;   // (the wave's last chunk)
                        lm_add_chunk(c, build_expert_wave(ids_c, w_c, w, x_c));
                    }
                    ca_mark_wave = true;
                    continue;
                }

                // the experts of this wave are read before its matmuls run, and the next wave may
                // evict them, so the graph has to hold the matmuls of this wave already
                ggml_tensor * out = build_expert_wave(ids_gemm, weights, w);


                ggml_build_forward_expand(gf, out);

                if (moe_out) {
                    moe_out = ggml_add(ctx0, moe_out, out);
                    ggml_build_forward_expand(gf, moe_out);
                } else {
                    moe_out = out;
                }
            }

            if (lm_phase == 2) {   // the VRAM pass per chunk, added into the residual / accumulator; the graph's result is lm_hid / lm_moe_chunks
                if (vram_ids) {
                    ggml_tensor * vids_dev = lm_ids_on_dev(vram_ids);
                    for (int64_t c = 0; c < lm_nc; c++) {
                        ggml_tensor * ids_c, * x_c, * w_c;
                        lm_chunk_views(c, vids_dev, &ids_c, &x_c, &w_c);
                        lm_add_chunk(c, build_vram_chain(ids_c, x_c, w_c));
                    }
                }
                lm_moe_chunks = std::move(lm_h_cur);
                GGML_ASSERT(vram_out_early == nullptr);
                return nullptr;
            }

            // The device pass's weighted sum, added once per layer after the host's (whose graph is then exactly the
            // one without the flag, fusions included: an all-zero device pass gives the same bits)
            if (vram_ids) {
                moe_out = ggml_add(ctx0, moe_out, build_vram_chain(vram_ids));
                ggml_build_forward_expand(gf, moe_out);
            } else if (vram_out_early) {   // the same add, the device pass's sum built earlier in the launch
                moe_out = ggml_add(ctx0, moe_out, vram_out_early);
                ggml_build_forward_expand(gf, moe_out);
            }

            cb(moe_out, "ffn_moe_out", il);

            return moe_out;
        }
    }

    ggml_tensor * moe_out = build_expert_wave(ids_gemm, weights);

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd, float tok_scale) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    // mixed path (ubatch.is_mixed()): set_rows the token rows into a copy of the embd rows, with its own inputs as select branches must not share tensors
    // TODO: use inp->tokens and inp->embd once ggml_build_forward_select allows it
    const bool has_mixed = llm_arch_supports_mixed_batch(arch) && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT;

    const int64_t n_tok_rows = has_mixed ? llm_graph_n_tok_rows(ubatch) : 0;

    // we have 3 standard paths to produce the input embeddings for the first layer:
    // - embd0: extract from the token embeddings weight (`tok_embd`) using the input token ids
    // - embd1: pass raw embeddings, skipping the `tok_embd`
    // - embd2: mixed path of both tokens ids + raw embeddings (if supported)
    ggml_tensor * embd0 = nullptr;
    ggml_tensor * embd1 = nullptr;
    ggml_tensor * embd2 = nullptr;

    // construct the input tensors
    {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
        cb(inp->embd, "inp_embd", -1);
        ggml_set_input(inp->embd);

        if (has_mixed) {
            inp->mixed_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tok_rows);
            cb(inp->mixed_tokens, "inp_mixed_tokens", -1);
            ggml_set_input(inp->mixed_tokens);
        }
    }

    // the embeddings placeholders for the 3 paths
    // we use ggml_build_forward_order to make the GET_ROWS ops stick at the beginning of the compute graph
    // this way the embeddings remain in the host buffer, and the GET_ROWS run before any other computations
    {
        embd0 = ggml_get_rows(ctx0, tok_embd, inp->tokens);
        ggml_build_forward_order(gf, embd0);

        embd1 = inp->embd;

        if (has_mixed) {
            embd2 = ggml_get_rows(ctx0, tok_embd, inp->mixed_tokens);
            ggml_build_forward_order(gf, embd2);
        }
    }

    // helper for extracting token embeddings with lora and padding
    // TODO: when lora is active, this is likely going to cause issues similar to https://github.com/ggml-org/llama.cpp/pull/30160
    //       need to add lora tests and refactor the logic to make the lora GET_ROWS go at the front of the graph
    auto build_tok = [&](ggml_tensor * cur, ggml_tensor * ids) {
        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, ids)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }

        return cur;
    };

    // select one of the 3 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 3> inps = {};

    // token embeddings path (ubatch.token != nullptr)
    inps[0] = build_tok(embd0, inp->tokens);

    // vector embeddings path (ubatch.embd != nullptr)
    inps[1] = embd1;

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    if (has_mixed) {
        inp->mixed_slots = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tok_rows);
        cb(inp->mixed_slots, "inp_mixed_slots", -1);
        ggml_set_input(inp->mixed_slots);

        inp->mixed_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
        cb(inp->mixed_embd, "inp_mixed_embd", -1);
        ggml_set_input(inp->mixed_embd);

        // note: set_rows writes into its destination, so it gets a copy of the input
        ggml_tensor * embd_mixed = build_tok(embd2, inp->mixed_tokens);
        inps[2] = ggml_set_rows(ctx0, ggml_dup(ctx0, inp->mixed_embd), embd_mixed, inp->mixed_slots);

        assert(ggml_are_same_shape (inps[0], inps[2]));
        assert(ggml_are_same_stride(inps[0], inps[2]));
    }

    const int idx = ubatch.is_mixed() ? 2 : ubatch.token ? 0 : 1;

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), has_mixed ? 3 : 2, idx);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // NOTE: For deepstack models, only apply scale to token inputs (ie text-only input).
    //  Raw embeddings are assumed to be multimodal inputs that should not be scaled.
    const bool scale_tok_only = hparams.f_embedding_scale != 0.0f && hparams.n_deepstack_layers > 0;

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f && !scale_tok_only) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    // scale the token rows only, applied after the select so that the graph is the same for any batch contents
    inp->scale_tok = tok_scale*(scale_tok_only ? hparams.f_embedding_scale : 1.0f);
    if (inp->scale_tok != 1.0f) {
        inp->scale_rows = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, ubatch.n_tokens);
        cb(inp->scale_rows, "inp_scale_rows", -1);
        ggml_set_input(inp->scale_rows);

        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_mul(ctx0, cur, inp->scale_rows);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);
    cb(cur, "inp_pos", -1);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);
    cb(cur, "inp_attn_scale", -1);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);
    ggml_set_name(cur, "out_ids");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);
    ggml_set_name(cur, "mean");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);
    ggml_set_name(cur, "cls");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);
    ggml_set_name(cur, "cross_embd");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "pos_bucket_enc");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "pos_bucket_dec");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
             int64_t   n_kv_max,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    // the stream dim steps over one stream's worth of dim-2 (tokens): (ne[2]/n_stream) rows of stride nb[2].
    // q->nb[3]/n_stream only equals that for a contiguous q; nope-only MLA (glm5-next) passes a permuted
    // q_absorbed where nb[3] != ne[2]*nb[2], so using nb[3] read another head's queries for streams s >= 1.
    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[2]*(q->ne[2]/n_stream), 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    ggml_tensor * cur;

    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        const bool qsa_compact = kq_mask != nullptr && kq_mask->type == GGML_TYPE_I32 && sinks == nullptr;
        cur = ggml_flash_attn_ext(ctx0, q, k, v, qsa_compact ? nullptr : kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        if (qsa_compact) {
            cur->src[4] = kq_mask;
        }
        GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX);
        ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<int32_t>(n_kv_max));
        ggml_prec_set_acc(cur, GGML_PREC_F32);

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_prec_set_acc(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);
    cb(inp->self_kq_mask, "self_kq_mask", -1);

    inp->self_kq_mask_cnv = inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_cur->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->build_input_v_rot(ctx0);

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(v_mla == nullptr);

    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_fused_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
                int   il) const {
    top_k = ggml_reshape_4d(ctx0, top_k, top_k->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);

    ggml_tensor * cur = ggml_top_k_mask(ctx0, kq_mask, top_k, nullptr, nullptr, nullptr);
    res->add_fused_node({LLM_FUSED_OP_TOP_K_MASK, cur, il});

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx->get_mla();

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs_mla();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    // top_k indexes the cells, and the indexer mask of the same type is built the same way
    // let's use it, so when no other layer reads the MLA mask is not allocated
    ggml_tensor * kq_mask = inp->get_kq_mask_mla();
    if (inp->get_kq_mask_lid()->type == kq_mask->type) {
        kq_mask = inp->get_kq_mask_lid();
    }

    ggml_tensor * kq_mask_top_k = nullptr;
    if (cparams.fused_top_k_mask) {
        kq_mask_top_k = build_fused_top_k_mask(kq_mask, top_k, il);
    } else {
        // prepare new kq mask - starts filled with -INFINITY
        ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

        // reshape KQ mask into tensor with rows of size 1:
        // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
        kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

        // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
        ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

        // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
        // this will be our source of zero values for unmasking top k mask elements
        ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
        zeros = ggml_fill(ctx0, zeros, 0.0f);

        // modify KQ mask by unmasking elements that are in top_k indices
        // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
        kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

        // reshape to restore the original shape of KQ mask:
        // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
        kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

        // combine with the original kq mask
        kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, top_k->ne[0], kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;

    if (k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    const auto * mctx_iswa = inp->mctx;
    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // MLA-style attention: the cached K is used as V
    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (k_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, k_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k_dsa> build_attn_inp_k_dsa_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_dsa_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_mla = build_attn_inp_kq_mask(ctx0, mctx_cur->get_mla(), ubatch, cparams);
        inp->self_kq_mask_mla_cnv = inp->self_kq_mask_mla;
    }

    {
        inp->self_k_idxs_lid = mctx_cur->get_lid()->build_input_k_idxs(ctx0, ubatch);

        // ensure that mask type matches fused lightning indexer use (requires f16 mask)
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = cparams.fused_lid;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);
    }

    return inp;
}

llm_graph_input_attn_k_dsa * llm_graph_context::build_attn_inp_k_dsa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_context *>(mctx);

    auto inp = build_attn_inp_k_dsa_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k_dsa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_dsa_iswa * llm_graph_context::build_attn_inp_k_dsa_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_iswa_context *>(mctx);

    auto inp_dsa = build_attn_inp_k_dsa_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_dsa());

    // build_attn_inp_k_impl rejects SWA caches, so construct the input directly
    auto inp_swa = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur->get_swa());

    inp_swa->self_k_idxs = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

    inp_swa->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
    inp_swa->self_kq_mask_cnv = inp_swa->self_kq_mask;

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa_iswa>(std::move(inp_dsa), std::move(inp_swa), mctx_cur);

    return (llm_graph_input_attn_k_dsa_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_kv_msa * llm_graph_context::build_attn_inp_kv_msa(bool msa_enabled) const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_msa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_msa>(hparams, cparams, mctx_cur);

    const auto * mctx_base = mctx_cur->get_base();
    const auto * mctx_idx  = mctx_cur->get_idx();

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_base->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_base->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_base, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_base->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_base->build_input_v_rot(ctx0);

    if (msa_enabled) {
        inp->self_k_idxs_idx = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    }

    return (llm_graph_input_attn_kv_msa *) res->add_input(std::move(inp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->get_base()->build_input_v_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);
    inp->self_v_rot_swa = mctx_cur->get_swa()->build_input_v_rot(ctx0);

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_attn_k_iswa * llm_graph_context::build_attn_inp_k_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);

    return (llm_graph_input_attn_k_iswa *) res->add_input(std::move(inp));
}

llm_graph_input_dsv4 * llm_graph_context::build_inp_dsv4() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);
    const auto * raw_ctx  = mctx_cur->get_raw();

    auto inp_raw = std::make_unique<llm_graph_input_dsv4_raw>(cparams, raw_ctx);

    const int64_t n_stream = mctx_cur->get_csa_plan(ubatch).n_stream;

    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "DSV4 expects SWA raw cache");

    inp_raw->self_k_idxs = raw_ctx->build_input_k_idxs(ctx0, ubatch);
    inp_raw->self_kq_mask = dsv4_build_raw_kq_mask(ctx0, raw_ctx, ubatch, cparams, n_stream);
    inp_raw->self_kq_mask_cnv = inp_raw->self_kq_mask;

    inp_raw->self_k_rot = raw_ctx->build_input_k_rot(ctx0);
    auto inp = std::make_unique<llm_graph_input_dsv4>(cparams, std::move(inp_raw), mctx_cur);

    dsv4_build_comp_inputs(ctx0, inp->inp_csa, mctx_cur->get_csa_plan(ubatch), "csa", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_hca, mctx_cur->get_hca_plan(ubatch), "hca", cparams, n_stream);
    dsv4_build_comp_inputs(ctx0, inp->inp_lid, mctx_cur->get_lid_plan(ubatch), "lid", cparams, n_stream);
    inp->inp_csa.k_rot = mctx_cur->get_csa()->build_input_k_rot(ctx0);
    inp->inp_hca.k_rot = mctx_cur->get_hca()->build_input_k_rot(ctx0);
    inp->inp_lid.k_rot = mctx_cur->get_lid()->build_input_k_rot(ctx0);

    return (llm_graph_input_dsv4 *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy,
        ggml_tensor * state_copy_main,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows,
               bool   expand_main) const {

    GGML_UNUSED(rs_size);
    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // one gather of the states i0..n_rs (ubatch states then extra states), sized by n_rs so the reserve covers every split
    // {state_size, rs_size} -> {state_size, n_rs - i0}
    const int64_t i0 = n_rs - state_copy->ne[0];

    ggml_tensor * output_states;
    ggml_tensor * states_extra;
    if (expand_main) {
        ggml_tensor * states_all = ggml_get_rows(ctx0, states, state_copy);

        output_states = get_state_rows ?
            get_state_rows(ctx0, states, state_copy_main) :
            ggml_view_2d(ctx0, states_all, state_size, n_seqs, states_all->nb[1], 0);
        ggml_build_forward_expand(gf, output_states);

        states_extra = ggml_view_2d(ctx0, states_all, state_size, n_rs - n_seqs, states_all->nb[1], (n_seqs - i0)*states_all->nb[1]);
    } else {
        // --moe-stream-cpu-state-inplace: a caller that reads the state in place leaves the main copy out of the
        // graph (it is computed only if something still uses it, after the reads below - as the caller's fallback); only the
        // extra states are gathered
        output_states = (get_state_rows ? get_state_rows : ggml_get_rows)(ctx0, states, state_copy_main);

        ggml_tensor * state_copy_extra = ggml_view_1d(ctx0, state_copy, n_rs - n_seqs, (n_seqs - i0)*state_copy->nb[0]);
        states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    }

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);
    ggml_set_name(inp->s_copy, "rs_s_copy");

    inp->s_copy_main = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_tail = ggml_view_1d(ctx0, inp->s_copy, n_rs - 1, inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

void llm_graph_context::run_moe_window() const {
    if (moe_window_nodes) {
        const std::function<void()> f = std::move(moe_window_nodes);
        moe_window_nodes = nullptr;
        f();
    }
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows,
               bool   expand_main) const {
    const auto * kv_state = inp->mctx;

    // a custom getter reads the states of the ubatch straight from the cache, so the gather skips the first
    // state: it still holds the n_rs - n_seqs extra states and copies no state of a single sequence ubatch
    ggml_tensor * state_copy = get_state_rows ? inp->s_copy_tail : inp->s_copy;

    return build_rs(s, state_copy, inp->s_copy_main, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows, expand_main);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        inp_attn->self_kq_mask_cnv = inp_attn->self_kq_mask;
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        inp_attn->self_kq_mask_swa_cnv = inp_attn->self_kq_mask_swa;
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_2_b,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_2_b || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_2_b) {
        cur = ggml_add(ctx0, cur, dense_2_b);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b,
        ggml_tensor * cls_norm) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (arch == LLM_ARCH_MODERN_BERT) {
                    // modern bert gte reranker builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    if (arch == LLM_ARCH_MODERN_BERT) {
                        cur = ggml_gelu(ctx0, cur);
                    } else {
                        cur = ggml_tanh(ctx0, cur);
                    }
                    if (cls_norm) {
                        // head norm
                        cur = build_norm(cur, cls_norm, NULL, LLM_NORM, -1);
                    }
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, std::vector<uint32_t>> sampling_rows;
    uint32_t n_rows = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.output[i]) {
            sampling_rows[ubatch.seq_id[i][0]].push_back(n_rows++);
        }
    }

    res->t_sampled.resize(n_rows, nullptr);
    res->t_sampled_probs.resize(n_rows, nullptr);
    res->t_sampled_logits.resize(n_rows, nullptr);
    res->t_candidates.resize(n_rows, nullptr);

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    // the padding row gives the chains without a row of the ubatch a valid input, even with no output at all
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    for (const auto & entry : samplers) {
        if (entry.second->iface->backend_reset) {
            entry.second->iface->backend_reset(entry.second);
        }
    }

    static const std::vector<uint32_t> no_rows;

    // every sampler builds n_outputs_max_per_seq chains, like the reserve, so the graph keeps its topology
    // whatever rows the ubatch outputs: a chain without a row works on the first row and is not selected
    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
        const auto & rows = it != sampling_rows.end() ? it->second : no_rows;

        for (uint32_t i = 0; i < cparams.n_outputs_max_per_seq; ++i) {
            const bool     active = i < rows.size();
            const uint32_t row    = active ? rows[i] : 0;
            const int      i_out  = active ? 1       : 0;

            ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], row * logits_t->nb[1]);
            ggml_format_name(logits_seq, "logits_seq_%d_%u", seq_id, i);

            struct llama_sampler_data data = {
                /*.logits       =*/ logits_seq,
                /*.probs        =*/ nullptr,
                /*.sampled      =*/ nullptr,
                /*.candidates   =*/ nullptr,
            };

            assert(sampler->iface->backend_apply);
            sampler->iface->backend_apply(sampler, ctx0, gf, &data);

            if (data.sampled != nullptr) {
                if (active) {
                    res->t_sampled[row] = data.sampled;
                }
                outs[1] = data.sampled;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.probs != nullptr) {
                if (active) {
                    res->t_sampled_probs[row] = data.probs;
                }
                outs[1] = data.probs;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.logits != nullptr) {
                if (active) {
                    res->t_sampled_logits[row] = data.logits;
                }
                outs[1] = data.logits;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }

            if (data.candidates != nullptr) {
                if (active) {
                    res->t_candidates[row] = data.candidates;
                }
                outs[1] = data.candidates;
                ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
            }
        }
    }

    // TODO: Call backend_accept after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
        if (it == sampling_rows.end()) {
            continue;
        }

        for (uint32_t row : it->second) {
            ggml_tensor * selected_token = res->t_sampled[row];
            if (selected_token != nullptr && sampler->iface->backend_accept) {
                sampler->iface->backend_accept(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}

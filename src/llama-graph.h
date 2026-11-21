#pragma once

struct llama_moe_stream;

#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-hparams.h"
#include "llama-adapter.h"

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <memory>
#include <set>
#include <functional>
#include <map>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

struct llama_cparams;
struct llama_layer;
struct llama_model;
struct llama_prec_policy;

struct llama_memory_context_i;

class llama_kv_cache_context;
class llama_kv_cache_dsa_context;
class llama_kv_cache_dsa_iswa_context;
class llama_kv_cache_msa_context;
class llama_kv_cache_dsv4_raw_context;
class llama_kv_cache_dsv4_context;
class llama_kv_cache_iswa_context;
class llama_memory_recurrent_context;
class llama_memory_hybrid_context;
class llama_memory_hybrid_iswa_context;

// certain models (typically multi-modal) can produce different types of graphs
enum llm_graph_type {
    LLM_GRAPH_TYPE_DEFAULT,
    LLM_GRAPH_TYPE_ENCODER,
    LLM_GRAPH_TYPE_DECODER,
    LLM_GRAPH_TYPE_DECODER_MTP,
};

enum llm_fused_op {
    LLM_FUSED_OP_FLASH_ATTN,
    LLM_FUSED_OP_GDN_AR,
    LLM_FUSED_OP_GDN_CH,
    LLM_FUSED_OP_LIGHTNING_INDEXER,
    LLM_FUSED_OP_DSV4_HC_PRE,
    LLM_FUSED_OP_DSV4_HC_COMB,
    LLM_FUSED_OP_DSV4_HC_POST,
    LLM_FUSED_OP_TOP_K_MASK,
};

enum llm_ffn_op_type : int {
    LLM_FFN_NONE = 0,           // sentinel: unset; archs must assign before use
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
    LLM_FFN_GEGLU,
    LLM_FFN_REGLU,
    LLM_FFN_SWIGLU_OAI_MOE,
    LLM_FFN_SITU,           // kimi-k3
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};

// TODO: tmp - need something better to pass the data from the encoder to the decoder
struct llama_cross {
    // the output embeddings from the encoder as a ggml tensor
    // TODO: this needs more work to be correct, for now copy the embeddings data to host memory
    //       ref: https://github.com/ggml-org/llama.cpp/pull/11213#discussion_r1969892524
    //ggml_tensor * t_embd = nullptr;

    int64_t n_embd = 0;
    int64_t n_enc  = 0;

    // embeddings data copied to host memory (tmp)
    std::vector<float> v_embd;

    // needed to construct the cross-attention mask in the decoder
    std::vector<std::set<llama_seq_id>> seq_ids_enc;
};

struct llm_graph_params;

//
// llm_graph_input
//

class llm_graph_input_i {
public:
    llm_graph_input_i() {
        const char * LLAMA_GRAPH_INPUT_DEBUG = getenv("LLAMA_GRAPH_INPUT_DEBUG");
        debug = LLAMA_GRAPH_INPUT_DEBUG ? atoi(LLAMA_GRAPH_INPUT_DEBUG) : 0;
    }

    virtual ~llm_graph_input_i() = default;

    virtual void set_input(const llama_ubatch * ubatch) = 0;

    // return true if the resulting input tensors using the provided graph parameters would be
    //   the same as the previous input tensors that we have currently stored in the object
    virtual bool can_reuse(const llm_graph_params & params) {
        // returning false here by default will prevent from reusing the graph if the check
        //   for the input type has not been implemented yet
        GGML_UNUSED(params);
        return false;
    }
protected:
    // env: LLAMA_GRAPH_INPUT_DEBUG
    int debug = 0;
};

using llm_graph_input_ptr = std::unique_ptr<llm_graph_input_i>;

class llm_graph_input_embd : public llm_graph_input_i {
public:
    llm_graph_input_embd(int64_t n_embd) : n_embd(n_embd) {}
    virtual ~llm_graph_input_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens       = nullptr; // I32 [n_batch]
    ggml_tensor * embd         = nullptr; // F32 [n_embd, n_batch]
    ggml_tensor * mixed_tokens = nullptr; // I32 [n_tok_rows], mixed path: ids of the token rows
    ggml_tensor * mixed_slots  = nullptr; // I64 [n_tok_rows], mixed path: batch index of the token rows
    ggml_tensor * mixed_embd   = nullptr; // F32 [n_embd, n_batch], mixed path: embd rows, token rows are overwritten
    ggml_tensor * scale_rows   = nullptr; // F32 [1, n_batch], per-row scale: scale_tok for token rows, 1 for embd rows

    float scale_tok = 1.0f;

    const int64_t n_embd = 0;
};

// similar to llm_graph_input_embd but with an additional hidden state input, fed from ubatch.embd_state
class llm_graph_input_embd_h : public llm_graph_input_i {
public:
    llm_graph_input_embd_h(int64_t n_embd, int64_t n_embd_state) : n_embd(n_embd), n_embd_state(n_embd_state) {}
    virtual ~llm_graph_input_embd_h() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens = nullptr; // I32 [n_batch]
    ggml_tensor * embd   = nullptr; // F32 [n_embd, n_batch]
    ggml_tensor * h      = nullptr; // F32 [n_embd_state, n_batch]

    const int64_t n_embd       = 0;
    const int64_t n_embd_state = 0;
};

class llm_graph_input_pos : public llm_graph_input_i {
public:
    llm_graph_input_pos(uint32_t n_pos_per_embd) : n_pos_per_embd(n_pos_per_embd) {}
    virtual ~llm_graph_input_pos() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * pos = nullptr; // I32 [n_batch]

    const uint32_t n_pos_per_embd = 1;
};

// temperature tuning, used by llama4
class llm_graph_input_attn_temp : public llm_graph_input_i {
public:
    llm_graph_input_attn_temp(uint32_t n_attn_temp_floor_scale, float f_attn_temp_scale, float f_attn_temp_offset)
        : n_attn_temp_floor_scale(n_attn_temp_floor_scale), f_attn_temp_scale(f_attn_temp_scale), f_attn_temp_offset(f_attn_temp_offset) {}
    virtual ~llm_graph_input_attn_temp() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * attn_scale = nullptr; // F32 [n_batch]

    const uint32_t n_attn_temp_floor_scale;
    const float    f_attn_temp_scale;
    const float    f_attn_temp_offset;
};

class llm_graph_input_pos_bucket : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket(const llama_hparams & hparams) : hparams(hparams) {}
    virtual ~llm_graph_input_pos_bucket() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_batch, n_batch]

    const llama_hparams hparams;
};

class llm_graph_input_pos_bucket_kv : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket_kv(
            const llama_hparams & hparams,
            const llama_kv_cache_context * mctx) : hparams(hparams), mctx(mctx) {}
    virtual ~llm_graph_input_pos_bucket_kv() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_kv, n_batch]

    const llama_hparams hparams;

    const llama_kv_cache_context * mctx;
};

class llm_graph_input_out_ids : public llm_graph_input_i {
public:
    llm_graph_input_out_ids(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            uint32_t n_outputs) : hparams(hparams), cparams(cparams), n_outputs(n_outputs) {}
    virtual ~llm_graph_input_out_ids() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * out_ids; // I32 [n_outputs]

    const llama_hparams hparams;
    const llama_cparams cparams;

    const uint32_t n_outputs;
};

class llm_graph_input_mean : public llm_graph_input_i {
public:
    llm_graph_input_mean(const llama_cparams & cparams) : cparams(cparams) {}
    virtual ~llm_graph_input_mean() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * mean; // F32 [n_batch, n_batch]

    const llama_cparams cparams;
};

class llm_graph_input_cls : public llm_graph_input_i {
public:
    llm_graph_input_cls(const llama_cparams & cparams, const llm_arch arch) : cparams(cparams), arch(arch) {}
    virtual ~llm_graph_input_cls() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cls; // I32 [n_batch]

    const llama_cparams cparams;
    const llm_arch arch;
};

class llm_graph_input_rs : public llm_graph_input_i {
public:
    llm_graph_input_rs(const llama_memory_recurrent_context * mctx) : mctx(mctx) {}
    virtual ~llm_graph_input_rs() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * s_copy;  // I32 [n_rs]

    // views of s_copy, computed once per graph
    // and shared across layers which use build_rs
    ggml_tensor * s_copy_main; // I32 [n_seqs]
    ggml_tensor * s_copy_tail; // I32 [n_rs - 1]

    const llama_memory_recurrent_context * mctx;

    // used in view offsets, need to match for valid graph reuse
    uint32_t head;
    int32_t rs_z;

    // a model read the ubatch's state straight from its cache row (one sequence whose state copy maps row head to itself):
    // the graph can be reused only while that still holds
    bool s_inplace = false;
};

class llm_graph_input_cross_embd : public llm_graph_input_i {
public:
    llm_graph_input_cross_embd(
            const llama_cross * cross) : cross(cross) {}
    virtual ~llm_graph_input_cross_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cross_embd; // F32 [n_embd, n_outputs_enc]

    const llama_cross * cross;
};

class llm_graph_input_attn_no_cache : public llm_graph_input_i {
public:
    llm_graph_input_attn_no_cache(const llama_hparams & hparams, const llama_cparams & cparams) :
        hparams(hparams),
        cparams(cparams) {
    }
    ~llm_graph_input_attn_no_cache() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    // n_tokens == n_batch
    ggml_tensor * self_kq_mask         = nullptr; // F32/F16 [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //         [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32/F16 [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //         [n_tokens, n_batch/n_stream, 1, n_stream]

    const llama_hparams hparams;
    const llama_cparams cparams;
};

class llm_graph_input_attn_kv : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_kv() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_v_idxs() const { return self_v_idxs; }

    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    ggml_tensor * self_kq_mask     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    // note: assumes v_rot^2 == I
    ggml_tensor * self_k_rot = nullptr;
    ggml_tensor * self_v_rot = nullptr;

    // note: these have to be copies because in order to be able to reuse a graph, its inputs
    //       need to carry these parameters with them. otherwise, they can point to freed
    //       llm_graph_params from a previous batch, causing stack-use-after-return
    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;
};

// V-less input for the KV cache
// ref: https://github.com/ggml-org/llama.cpp/pull/19067
class llm_graph_input_attn_k : public llm_graph_input_i {
public:
    llm_graph_input_attn_k(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // like can_reuse, but does not re-bind mctx
    bool can_reuse_impl(const llm_graph_params & params);

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }

    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;
};

class llm_graph_input_attn_k_dsa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_dsa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_dsa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_dsa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    // like can_reuse, but does not re-bind mctx
    bool can_reuse_impl(const llm_graph_params & params);

    ggml_tensor * get_k_idxs_mla() const { return self_k_idxs_mla; }
    ggml_tensor * get_k_idxs_lid() const { return self_k_idxs_lid; }

    ggml_tensor * get_kq_mask_mla() const { return self_kq_mask_mla_cnv; }
    ggml_tensor * get_kq_mask_lid() const { return self_kq_mask_lid; }

    ggml_tensor * self_k_idxs_mla = nullptr; // I64 [n_batch]
    ggml_tensor * self_k_idxs_lid = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask_mla     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_mla_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_lid     = nullptr; // F32     [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_lid_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_k_rot_lid = nullptr;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_dsa_context * mctx;
};

// DSA input (full-attention layers + indexer) with K-only input for the SWA layers
class llm_graph_input_attn_k_dsa_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_dsa_iswa(
            std::unique_ptr<llm_graph_input_attn_k_dsa> inp_dsa,
            std::unique_ptr<llm_graph_input_attn_k>     inp_swa,
            const llama_kv_cache_dsa_iswa_context *     mctx) :
        inp_dsa(std::move(inp_dsa)),
        inp_swa(std::move(inp_swa)),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_dsa_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    llm_graph_input_attn_k_dsa * get_dsa() const { return inp_dsa.get(); }
    llm_graph_input_attn_k     * get_swa() const { return inp_swa.get(); }

    std::unique_ptr<llm_graph_input_attn_k_dsa> inp_dsa;
    std::unique_ptr<llm_graph_input_attn_k>     inp_swa;

    const llama_kv_cache_dsa_iswa_context * mctx;
};

// standard K/V attention input against the base cache, plus destination indices for the indexer key cache
class llm_graph_input_attn_kv_msa : public llm_graph_input_attn_kv {
public:
    llm_graph_input_attn_kv_msa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_msa_context * mctx);
    ~llm_graph_input_attn_kv_msa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs_idx() const { return self_k_idxs_idx; }

    ggml_tensor * self_k_idxs_idx = nullptr; // I64 [n_batch]

    const llama_kv_cache_msa_context * mctx_msa;
};

class llm_graph_input_attn_kv_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv_iswa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_iswa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_kv_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs()     const { return self_k_idxs; }
    ggml_tensor * get_v_idxs()     const { return self_v_idxs; }
    ggml_tensor * get_k_idxs_swa() const { return self_k_idxs_swa; }
    ggml_tensor * get_v_idxs_swa() const { return self_v_idxs_swa; }

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    ggml_tensor * self_k_idxs     = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs     = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]
    ggml_tensor * self_k_idxs_swa = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs_swa = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    ggml_tensor * self_kq_mask         = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_k_rot = nullptr;
    ggml_tensor * self_v_rot = nullptr;

    ggml_tensor * self_k_rot_swa = nullptr;
    ggml_tensor * self_v_rot_swa = nullptr;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_iswa_context * mctx;
};

class llm_graph_input_attn_k_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_iswa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_iswa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs()     const { return self_k_idxs; }
    ggml_tensor * get_k_idxs_swa() const { return self_k_idxs_swa; }

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    ggml_tensor * self_k_idxs     = nullptr; // I64 [n_batch]
    ggml_tensor * self_k_idxs_swa = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask         = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_k_rot = nullptr;
    ggml_tensor * self_k_rot_swa = nullptr;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_iswa_context * mctx;
};

// DSV4 raw graph inputs are SWA-only, but their mask may be stream-shaped
// so raw K can be concatenated with DSV4 compressed K in one attention op.
class llm_graph_input_dsv4_raw {
public:
    llm_graph_input_dsv4_raw(
            const llama_cparams & cparams,
            const llama_kv_cache_dsv4_raw_context * mctx) :
        cparams(cparams),
        mctx(mctx) {
    }

    void set_input(const llama_ubatch * ubatch);

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_k_rot = nullptr;

    const llama_cparams cparams;

    const llama_kv_cache_dsv4_raw_context * mctx;
};

class llm_graph_input_dsv4 : public llm_graph_input_i {
public:
    struct comp_input {
        ggml_tensor * state_pos        = nullptr; // I32 [n_state]
        ggml_tensor * state_persist_src_idxs = nullptr; // I32 [n_state_persist]
        ggml_tensor * state_persist_dst_idxs = nullptr; // I32 [n_state_persist]
        ggml_tensor * state_restore_src_idxs = nullptr; // I32 [n_state_restore]
        ggml_tensor * state_restore_dst_idxs = nullptr; // I32 [n_state_restore]
        ggml_tensor * state_snapshot_src_idxs = nullptr; // I32 [n_state_snapshot]
        ggml_tensor * state_snapshot_dst_idxs = nullptr; // I32 [n_state_snapshot]
        ggml_tensor * state_read_idxs  = nullptr; // I32 [ratio*n_state_write]
        ggml_tensor * state_write_idxs = nullptr; // I64 [n_state_write]
        ggml_tensor * state_write_pos  = nullptr; // I32 [n_state_write]

        ggml_tensor * kq_mask    = nullptr; // F32 [n_kv, n_batch/n_stream, 1, n_stream]

        ggml_tensor * k_rot      = nullptr;
    };

    llm_graph_input_dsv4(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_dsv4_raw> inp_raw,
            const llama_kv_cache_dsv4_context * mctx) :
        inp_raw(std::move(inp_raw)),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_dsv4() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    llm_graph_input_dsv4_raw * get_raw() const { return inp_raw.get(); }
    const comp_input & get_csa() const { return inp_csa; }
    const comp_input & get_hca() const { return inp_hca; }
    const comp_input & get_lid() const { return inp_lid; }

    std::unique_ptr<llm_graph_input_dsv4_raw> inp_raw;

    comp_input inp_csa;
    comp_input inp_hca;
    comp_input inp_lid;

    const llama_cparams cparams;

    const llama_kv_cache_dsv4_context * mctx;
};

class llm_graph_input_attn_cross : public llm_graph_input_i {
public:
    llm_graph_input_attn_cross(const llama_cross * cross) : cross(cross) {}
    ~llm_graph_input_attn_cross() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * get_kq_mask_cross() const { return cross_kq_mask_cnv; }

    ggml_tensor * cross_kq_mask     = nullptr; // F32/F16 [n_outputs_enc, n_batch, 1, 1]
    ggml_tensor * cross_kq_mask_cnv = nullptr; // F32/F16 [n_outputs_enc, n_batch, 1, 1]

    const llama_cross * cross = nullptr;
};

class llm_graph_input_mem_hybrid : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv> inp_attn,
            std::unique_ptr<llm_graph_input_rs>      inp_rs,
            const llama_memory_hybrid_context *      mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_kv * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_context * mctx;
};

class llm_graph_input_mem_hybrid_k : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_k(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_k> inp_attn,
            std::unique_ptr<llm_graph_input_rs>      inp_rs,
            const llama_memory_hybrid_context *      mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_k() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_k> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_k * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_context * mctx;
};

class llm_graph_input_mem_hybrid_iswa : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_iswa(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn,
            std::unique_ptr<llm_graph_input_rs>          inp_rs,
            const llama_memory_hybrid_iswa_context *     mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn;
    std::unique_ptr<llm_graph_input_rs>          inp_rs;

    llm_graph_input_attn_kv_iswa * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs           * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_iswa_context * mctx;
};

class llm_graph_input_sampling : public llm_graph_input_i {
public:
    llm_graph_input_sampling(std::map<llama_seq_id, llama_sampler *> samplers) :
        samplers(std::move(samplers)) { }
    virtual ~llm_graph_input_sampling() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    std::map<llama_seq_id, llama_sampler *> samplers;
};

//
// llm_graph_result
//

// these objects deliver the result from the graph build process back to the llama_context
// note that the input tensors created for the graph are referenced here - the goal is to be able to populate their
//   specific data, by calling the set_inputs() method
// along with the input tensors, the object also provides commonly used outputs tensors, such as logits, embeddings, etc.
//   these are used by the llama_context to extact the relevant data, based on the compute parameters

// callback that allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
using llm_graph_cb = std::function<void(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il)>;

class llm_graph_result;

struct llama_moe_stream;

struct llm_graph_params {
    llm_arch arch = LLM_ARCH_UNKNOWN;

    llama_hparams hparams;
    llama_cparams cparams;

    llama_ubatch ubatch; // note: intentionally make a copy

    llm_graph_type gtype;

    ggml_backend_sched_t sched;
    ggml_backend_t backend_cpu;

    const llama_adapter_cvec     * cvec;
    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_cross            * cross;

    // MoE expert streaming state of the model, null when not enabled
    llama_moe_stream * mstream = nullptr;

    // A snapshot of the streaming engine's graph generation, bumped when the set of layers whose MoE carries the
    // VRAM expert pool's device pass changes, so a graph built before that is not reused after it (a reused graph
    // without the device pass gave the held experts a zero slice and nobody computed them). A snapshot, not the pointer's
    // value: both graphs of a comparison point at the same engine.
    uint32_t mstream_gen = 0;

    const llama_prec_policy * prec_policy = nullptr;

    std::map<llama_seq_id, llama_sampler *> samplers;

    static bool samplers_equal(
          const std::map<llama_seq_id, llama_sampler *> & lhs,
          const std::map<llama_seq_id, llama_sampler *> & rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (const auto & [seq_id, sampler] : lhs) {
            auto it = rhs.find(seq_id);
            if (it == rhs.end() || it->second != sampler) {
                return false;
            }
        }
        return true;
    }

    uint32_t n_outputs;

    llm_graph_cb cb;

    llm_graph_result * res;

    // Layer-major prompt (--moe-stream-layer-major 1): build only the layers [lm_il0, lm_il1) (lm_il1 < 0 = to the
    // last); a graph that starts after layer 0 takes the hidden state as its input (ubatch.embd), and a graph that ends before the
    // last layer outputs every token's hidden state (llm_graph_result::t_lm_hidden) instead of the logits
    int32_t lm_il0 = 0;
    int32_t lm_il1 = -1;
    // The call's hidden state [n_embd * hc, T] (hc = the hyper-connection streams, 1 for a model without them) in a device
    // buffer (llama_context); a group graph of this ubatch reads its input from rows [lm_row0, lm_row0 + n_tokens) when it starts
    // after layer 0 and writes its output there when it ends early - nothing goes through host memory and nothing waits between
    // the graphs. nullptr = the host path (ubatch.embd)
    ggml_tensor * lm_hid  = nullptr;
    int64_t       lm_row0 = 0;
    // lm_phase 1 = a streamed layer's first part for one ubatch (attention + router + shared expert; writes the residual +
    // shared expert into lm_hid, the normed expert input into lm_x and the router's picks / weights into lm_ids / lm_w, at its rows);
    // 2 = the expert graph of a streamed layer for the whole call (the usual wave loop over lm_ids, its matmuls in chunks of lm_chunk
    // tokens, each chunk's routed output added into lm_hid); 0 = whole layers.
    // A model with hyper-connection streams (qwen4exp) writes only the residual into lm_hid in lm_phase 1, and the FFN combine's
    // inject weights into lm_inj; in lm_phase 2 the routed output is summed into lm_moe, and per chunk the model adds the shared
    // expert and runs the combine into lm_hid
    int32_t       lm_phase = 0;
    ggml_tensor * lm_x     = nullptr;
    ggml_tensor * lm_moe   = nullptr;   // optional [n_embd, T] routed-expert accumulator when hidden state has hyper-connection streams
    ggml_tensor * lm_inj   = nullptr;   // optional [hc, T] FFN hyper-connection scatter weights from lm_phase 1
    ggml_tensor * lm_ids   = nullptr;
    ggml_tensor * lm_w     = nullptr;
    int64_t       lm_chunk = 0;

    // return true if the "other" params would result in a graph with the same topology as with the current params
    //   having the same topology allows us to reuse the graph in some cases
    bool allow_reuse(const llm_graph_params & other) const {
        // first check the ubatch
        bool can_reuse_ubatch =
            ubatch.equal_seqs() == other.ubatch.equal_seqs() &&
            ubatch.n_tokens     == other.ubatch.n_tokens &&
            ubatch.n_seq_tokens == other.ubatch.n_seq_tokens &&
            ubatch.n_seqs       == other.ubatch.n_seqs &&
            ubatch.n_seqs_unq   == other.ubatch.n_seqs_unq &&
            ubatch.is_mixed()   == other.ubatch.is_mixed() &&
            (
                (!ubatch.token && !other.ubatch.token) ||
                (!ubatch.embd  && !other.ubatch.embd)  ||
                (ubatch.token && other.ubatch.token && ubatch.embd && other.ubatch.embd)
            ) &&
            (!ubatch.embd_state == !other.ubatch.embd_state);

        // when we split the batch using "equal_seqs" we have to verify that the participating sequences are the same
        //   the reason is because the set of attention streams would be different for different sequences
        if (can_reuse_ubatch && ubatch.equal_seqs()) {
            if (!ubatch.data) {
                // if the old ubatch does not own it's data, then we cannot guarantee that it is still alive, and
                //   therefore we cannot perform the sequence id check. normally should never happen
                can_reuse_ubatch = false;
            } else {
                for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                    can_reuse_ubatch &= ubatch.seq_id_unq[s] == other.ubatch.seq_id_unq[s];
                }
            }
        }

        if (!can_reuse_ubatch) {
            return false;
        }

        if (n_outputs != other.n_outputs) {
            return false;
        }

        if (mstream_gen != other.mstream_gen) {
            return false;
        }

        // Never reuse a graph that reads / writes the call's device hidden state - it is allocated per call, so
        // a new one can get the old one's address while the graph's views still point into the freed buffer
        if (lm_il0 != other.lm_il0 || lm_il1 != other.lm_il1 || lm_hid != nullptr || other.lm_hid != nullptr) {
            return false;
        }

        if (!samplers_equal(samplers, other.samplers)) {
            return false;
        }

        if (samplers.size() > 0) {
            if (!ubatch.data || !other.ubatch.data) {
                return false;
            }

            // check that the outputs are the same for all samplers
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (ubatch.output[i]    != other.ubatch.output[i] ||
                    ubatch.seq_id[i][0] != other.ubatch.seq_id[i][0]) {
                    return false;
                }
            }
        }

        // TODO: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448035248
        if (cparams.nextn_layer_offset != other.cparams.nextn_layer_offset) {
            return false;
        }

        return
            cparams.embeddings              == other.cparams.embeddings              &&
            cparams.embeddings_nextn        == other.cparams.embeddings_nextn        &&
            cparams.embeddings_nextn_masked == other.cparams.embeddings_nextn_masked &&
            cparams.causal_attn             == other.cparams.causal_attn             &&
            arch  == other.arch  &&
            gtype == other.gtype &&
            cvec  == other.cvec  &&
            loras == other.loras &&
            cross == other.cross;
    }
};

struct llm_graph_fused_node {
    llm_fused_op op;
    ggml_tensor * tensor;
    int il;
};

class llm_graph_result {
public:
    llm_graph_result(int64_t max_nodes);

    virtual ~llm_graph_result() = default;

    ggml_tensor * get_inp_tokens()  const { return t_inp_tokens; }
    ggml_tensor * get_logits()      const { return t_logits; }
    ggml_tensor * get_embd()        const { return t_embd; }
    ggml_tensor * get_embd_pooled() const { return t_embd_pooled; }
    ggml_tensor * get_h_nextn()     const { return t_h_nextn; }

    ggml_tensor * get_layer_inp(int il) const { return t_layer_inp[il]; }

    ggml_cgraph  * get_gf()  const { return gf; }
    ggml_context * get_ctx() const { return ctx_compute.get(); }

    int64_t get_max_nodes() const;

    void reset();

    void set_inputs(const llama_ubatch * ubatch);
    void set_outputs(const llm_graph_params & params);

    // try to update the existing graph result using the new graph parameters in order to reuse it
    // this can only be done if we determine that the resulting graph using the new graph parameters
    //   would be identical to the existing graph. in that case, we simply have to update the memory
    //   contexts of the input tensors of the graph and we can reuse it for another computation
    // return true if the graph was updated and can be reused
    bool can_reuse(const llm_graph_params & params);

    llm_graph_input_i * add_input(llm_graph_input_ptr input);

    void add_fused_node(llm_graph_fused_node result);

    const std::vector<llm_graph_fused_node> & get_fused_nodes() const { return fused_nodes; }

    void set_params(const llm_graph_params & params);

    // important graph nodes
    ggml_tensor * t_inp_tokens  = nullptr;
    ggml_tensor * t_inp_embd    = nullptr; // [n_embd_inp, n_tokens]
    ggml_tensor * t_logits      = nullptr;
    ggml_tensor * t_embd        = nullptr;
    ggml_tensor * t_embd_pooled = nullptr;
    ggml_tensor * t_h_nextn     = nullptr; // [n_embd, n_outputs] hidden state before final output norm
    ggml_tensor * t_lm_hidden   = nullptr; // [n_embd, n_tokens] the hidden state after a layer group that ends early
    // llama_context's op-offload decision for this graph (kept with it: a reused graph keeps its split) and a layer-major
    // expert graph's chunk rows (0 = none) - the engine reads them before every compute (experts_on_cpu)
    bool          ms_op_offload = true;
    int64_t       ms_chunk      = 0;

    std::vector<ggml_tensor *> t_layer_inp;

    std::vector<ggml_tensor *> t_sampled;
    std::vector<ggml_tensor *> t_sampled_probs;
    std::vector<ggml_tensor *> t_sampled_logits;
    std::vector<ggml_tensor *> t_candidates;

    std::vector<llm_graph_input_ptr> inputs;
    std::vector<llm_graph_fused_node> fused_nodes;

    ggml_context_ptr ctx_compute;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;

    ggml_cgraph * gf;

    int64_t max_nodes;

private:
    // keep a copy of the previous graph parameters
    // we will use this to determine whether the graph can be reused by comparing them with the new parameters
    // note: these are updated after constructing the new graph
    llm_graph_params params;

    // env: LLAMA_GRAPH_RESULT_DEBUG
    int debug = 0;
};

using llm_graph_result_ptr = std::unique_ptr<llm_graph_result>;

//
// llm_graph_context
//

// used in build_rs to properly order writes and avoid unnecessary copies
using llm_graph_get_rows_fn = std::function<ggml_tensor * (ggml_context *, ggml_tensor * states, ggml_tensor * ids)>;

struct llm_graph_qkv {
    ggml_tensor * q; // [n_embd_head, n_head,    n_tokens]
    ggml_tensor * k; // [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * v; // [n_embd_head, n_head_kv, n_tokens]
};

// --moe-stream-record: the tensors the recorder runs the routers of the next four layers with. base is
// the residual plus this layer's shared expert, NOT normed; gate[d] and norm[d] are the router and the
// ffn_norm gain of layer il + d (null when there is no such layer).
// --moe-stream-lookahead N: the same tensors for distances up to 8 (the far guesses are built exactly like the
// recorder's lead d), plus probs_b[d] = layer il + d's selection bias for their top-k.
struct llm_graph_rec_leads {
    static constexpr int MAX_D = 32;
    ggml_tensor * base    = nullptr;
    ggml_tensor * gate   [MAX_D + 1] = { nullptr };
    ggml_tensor * gate_b [MAX_D + 1] = { nullptr };
    ggml_tensor * norm   [MAX_D + 1] = { nullptr };
    ggml_tensor * probs_b[MAX_D + 1] = { nullptr };
    // the guess input for layer il + d, built by the model (a model whose routers do not read "base normed with ffn_norm", e.g.
    // hyper-connection models); null: base normed with norm[d]
    ggml_tensor * in     [MAX_D + 1] = { nullptr };
};

// A model's own guess input for target layer X from its base (llm_graph_moe_guess::pred_base); see moe_guess_finish_in
using llm_graph_guess_in_fn = std::function<ggml_tensor * (int X)>;

// The guesses a model hands to build_moe_ffn for the expert streaming, built once in shared code for every
// model (Ling's wiring, moved out of bailingmoe3.cpp). First call (moe_guess_routers): the routers 1-3 MoE layers ahead (the correction file's corrected
// matrix for the next one); if want_shared, the model builds its shared expert BEFORE build_moe_ffn (--moe-stream-pred-shared) and
// passes it to the second call (moe_guess_finish): the guess input (residual + shared expert, normed with the next layer's gain) and the far
// guesses' leads (--moe-stream-lookahead, the recorder). Without --moe-stream everything stays null: the stock graph.
struct llm_graph_moe_guess {
    ggml_tensor * gate_inp_next  = nullptr;   // the next MoE layer's router: the 1-ahead guess
    ggml_tensor * probs_b_next   = nullptr;
    ggml_tensor * gate_inp_next2 = nullptr;   // two and three ahead
    ggml_tensor * probs_b_next2  = nullptr;
    ggml_tensor * gate_inp_next3 = nullptr;
    ggml_tensor * probs_b_next3  = nullptr;
    bool          want_shared    = false;     // --moe-stream-pred-shared and a next MoE layer: build the shared expert first
    ggml_tensor * pred_base      = nullptr;   // the residual + this layer's shared expert (the recorder's; a word-start source's base)
    ggml_tensor * pred_in        = nullptr;   // pred_base normed with the next layer's gain
    bool          rec_on         = false;     // rec is filled (the recorder, or --moe-stream-lookahead > 1)
    llm_graph_rec_leads rec;
};

// The word-start guesses' state across a one-token graph's layers
struct llm_graph_moe_ws {
    bool          graph = false;   // a graph of one token with --moe-stream-wordstart D > 0 and prefetch on
    bool          s2g   = false;   // every source on needed-early's GPU mark: each source's guess is marked, the word-start CPU op is not built
    ggml_tensor * cols  = nullptr; // the sources' guess columns so far, in layer order
};

struct llm_graph_amax_src1 {
    ggml_tensor * res;     // cur rescaled
    ggml_tensor * scale_s; // scale to undo on the matmul result, or nullptr
};

struct llm_graph_context {
    const llm_arch arch;

    const llama_hparams & hparams;
    const llama_cparams & cparams;
    const llama_ubatch  & ubatch;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_layer_nextn;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int64_t n_tokens;
    const int64_t n_outputs;
    const int32_t n_ctx_orig; // yarn

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    ggml_backend_sched_t sched;

    ggml_backend_t backend_cpu; // TODO: needed by build_attn_mha, figure out a way to remove?

    const llama_adapter_cvec     * cvec;
    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_cross            * cross;

    // MoE expert streaming state of the model, null when not enabled
    llama_moe_stream * mstream;

    // Nodes a model builder wants in the GPU window of the MoE layer it is about to build (the idle time
    // after the router mark while the CPU computes the experts). Set before build_moe_ffn; build_moe_ffn runs it once, after the mark
    // and the VRAM pass (or, in a layer without a mark, in front of the remap op); the builder calls run_moe_window() after
    // build_moe_ffn for the layers that never reached it. Empty = nothing to do.
    mutable std::function<void()> moe_window_nodes;
    void run_moe_window() const;

    const llama_prec_policy * prec_policy;

    std::map<llama_seq_id, llama_sampler *> samplers;

    const llm_graph_cb & cb_func;

    llm_graph_result * res;

    ggml_context * ctx0 = nullptr;
    ggml_cgraph  * gf   = nullptr;

    // What build_moe_ffn needs of a layer-major call (llm_graph_params::lm_*)
    int32_t       lm_phase = 0;
    ggml_tensor * lm_hid   = nullptr;
    ggml_tensor * lm_x     = nullptr;
    ggml_tensor * lm_moe   = nullptr;
    ggml_tensor * lm_ids   = nullptr;
    ggml_tensor * lm_w     = nullptr;
    int64_t       lm_row0  = 0;
    int64_t       lm_chunk = 0;
    // filled by build_moe_ffn in lm_phase 2: each chunk's last copy into lm_moe (or lm_hid without it); a model builder with
    // lm_moe set continues from these nodes (shared expert + combine)
    mutable std::vector<ggml_tensor *> lm_moe_chunks;

    // Auto word-start tracking for non-bespoke MoE architectures (skipped when moe_ws_begin() is called explicitly):
    mutable bool             ws_bespoke     = false;
    mutable llm_graph_moe_ws ws_auto_st;
    mutable int32_t          ws_auto_next   = -1;
    mutable ggml_tensor *    ws_last_ffn_in = nullptr;
    mutable int32_t          ws_last_ffn_il = -1;

    ggml_tensor * build_gg_shares(
             ggml_tensor * probs_2d,
             ggml_tensor * ids_2d,
            llama_expert_gating_func_type gating_op) const;

    llm_graph_context(const llm_graph_params & params);
    virtual ~llm_graph_context() = default;

    void cb(ggml_tensor * cur, const char * name, int il) const;

    // MoE streaming: run node t on the backend of device dev (none: as the scheduler decides) - needed-early's marked nodes must run on
    // the GPU that holds the marks, the GPU pass of a layer-major call on the VRAM pool's GPU
    ggml_backend_t moe_backend_of(ggml_backend_dev_t dev) const;
    void           moe_pin(ggml_tensor * t, ggml_backend_dev_t dev) const;
    // A tensor that belongs to the predictor of the next layer gets its own name, so that a trace
    // tool that keys on the name (llama-moe-trace) does not record the predicted ids as the real ones
    void cb_tag(ggml_tensor * cur, const char * name, int il, const char * tag) const;

    // true when the last layer must be narrowed to the output rows before the nextn hidden state is captured
    bool crop_before_nextn(const ggml_tensor * inp_out_ids) const {
        return inp_out_ids != nullptr && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);
    }

    // true when the nextn hidden state must be narrowed to the output rows after it is captured
    bool crop_after_nextn(const ggml_tensor * inp_out_ids) const {
        return inp_out_ids != nullptr && cparams.embeddings_nextn && !cparams.embeddings_nextn_masked;
    }

    //
    // common
    //

    ggml_tensor * build_cvec(
             ggml_tensor * cur,
                     int   il) const;

    // do mat_mul, while optionally apply lora and per-tensor scale
    ggml_tensor * build_lora_mm(
              ggml_tensor * w,
              ggml_tensor * cur,
              ggml_tensor * w_s = nullptr) const;

    // do mat_mul_id, while optionally apply lora and per-expert scale
    // ids_scale: ids to gather the per-expert scale with, when the GEMM ids are remapped cache
    //            slots instead of expert ids (MoE expert streaming)
    ggml_tensor * build_lora_mm_id(
              ggml_tensor * w,   // ggml_tensor * as
              ggml_tensor * cur, // ggml_tensor * b
              ggml_tensor * ids,
              ggml_tensor * w_s     = nullptr,
              ggml_tensor * ids_scale = nullptr) const;

    // rescale `cur` (the src1 of a matmul) by a power of two, computed from its maximum absolute
    // keep both individual values and Q8 activation block sums within F16 range
    llm_graph_amax_src1 build_amax_src1_f16(
              ggml_tensor * cur) const;

    ggml_tensor * build_norm(
             ggml_tensor * cur,
             ggml_tensor * mw,
             ggml_tensor * mb,
           llm_norm_type   type,
                     int   il) const;


    // compute Q, K, V projections with optional bias and reshape
    // supports both fused wqkv and separate wq/wk/wv paths
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const;

    // Set reshape to false to return contiguous projections before clamp/reshape.
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head_q,
                  int64_t   n_head_q,
                  int64_t   n_embd_head_k,
                  int64_t   n_head_k,
                  int64_t   n_embd_head_v,
                  int64_t   n_head_v,
                      int   il,
                     bool   reshape = true) const;

    ggml_tensor * build_ffn(
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
                     int   il) const;

    // the router of a MoE layer: the expert probabilities of cur with the gating of gating_op,
    // [n_expert, n_tokens]. logits_in replaces the gate_inp matmul when it is not null.
    // out_logits gets the router output with the bias applied and before the gating, which is
    // what llama4 and grovemoe select their experts on (the two callers used to pass a null there).
    ggml_tensor * build_moe_probs(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * gate_inp_b,
             ggml_tensor * logits_in,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor ** out_logits = nullptr,
             const char    * name_tag   = nullptr) const;

    // the expert selection: bias, expert groups and top k on the router probabilities,
    // [n_expert_used, n_tokens]. logits is the router output before the gating, llama4 only.
    ggml_tensor * build_moe_topk(
             ggml_tensor * probs,
             ggml_tensor * logits,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
                     int   il,
             const char    * name_tag = nullptr) const;

    // --moe-stream-wordstart D: the word-start guesses of source layer il for its streamed targets il + d (d <= N), as
    // a streamed layer's far guesses; returns the guess columns [width, n_targets], or nullptr when it has no streamed target
    // The shared guess wiring (llm_graph_moe_guess / llm_graph_moe_ws above); every MoE model calls these
    llm_graph_moe_guess moe_guess_routers(const llama_model & model, int il) const;
    void                moe_guess_finish(const llama_model & model, int il, ggml_tensor * ffn_inp, ggml_tensor * shared,
                                         llm_graph_moe_guess & g) const;
    // the same for a model that builds its own base (residual + shared expert, in its own form; nullptr: no shared-expert
    // guess) and, with guess_in, its own guess input per target layer (instead of the base normed with that layer's ffn_norm)
    void                moe_guess_finish_in(const llama_model & model, int il, ggml_tensor * base,
                                            const llm_graph_guess_in_fn & guess_in, llm_graph_moe_guess & g) const;
    llm_graph_moe_ws    moe_ws_begin() const;
    // after layer il's output `out` (residual added): a word-start source builds its guesses; ws_base = the MoE source's
    // pred_base (nullptr for a dense layer: its output is the base), ffn_in_n = its normed FFN input
    void                moe_ws_layer(const llama_model & model, int il, ggml_tensor * out, ggml_tensor * ws_base,
                                     ggml_tensor * ffn_in_n, llm_graph_moe_ws & st,
                                     const llm_graph_guess_in_fn & guess_in = nullptr) const;

    ggml_tensor * build_ws_guess(
                     int   il,
             ggml_tensor * base,
             ggml_tensor * cur_in,
             const llm_graph_rec_leads & rec,
            llama_expert_gating_func_type gating_op,
                 int64_t   n_expert,
             ggml_tensor ** out_shares = nullptr) const;

    // build MoE FFN without bias tensors
    ggml_tensor * build_moe_ffn(
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
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * up_exps_s = nullptr,
             ggml_tensor * gate_exps_s = nullptr,
             ggml_tensor * down_exps_s = nullptr,
             ggml_tensor * selected_experts_in = nullptr,
    // router of the next MoE layer, for the layer-ahead expert prefetch of MoE expert streaming
             ggml_tensor * gate_inp_next = nullptr,
             ggml_tensor * exp_probs_b_next = nullptr,
    // What the *next* router is run on instead of this layer's normalised input, when the
    // prediction should see the shared expert's output. nullptr = this layer's own router input.
             ggml_tensor * pred_in = nullptr,
    // The router of the layer *two* ahead, for a two-layer lead (--moe-stream-prefetch-lead 2):
    // it is run on this layer's own router input, so the early guess is not delayed by anything
             ggml_tensor * gate_inp_next2 = nullptr,
             ggml_tensor * exp_probs_b_next2 = nullptr,
             ggml_tensor * gate_inp_next3 = nullptr,   // lead 3: the router of the layer *three* ahead
             ggml_tensor * exp_probs_b_next3 = nullptr, // lead 3
    // --moe-stream-record: what the recorder needs of the layers after this one
             const llm_graph_rec_leads * rec = nullptr
             ) const;

    ggml_tensor * build_moe_ffn(
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
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * gate_up_exps_b = nullptr,
             ggml_tensor * up_exps_s = nullptr,
             ggml_tensor * gate_exps_s = nullptr,
             ggml_tensor * down_exps_s = nullptr,
             ggml_tensor * selected_experts_in = nullptr,
             ggml_tensor * gate_inp_next = nullptr,
             ggml_tensor * exp_probs_b_next = nullptr,
             ggml_tensor * pred_in = nullptr,
             ggml_tensor * gate_inp_next2 = nullptr,
             ggml_tensor * exp_probs_b_next2 = nullptr,
             ggml_tensor * gate_inp_next3 = nullptr,   // lead 3: the router of the layer *three* ahead
             ggml_tensor * exp_probs_b_next3 = nullptr, // lead 3
    // --moe-stream-record: what the recorder needs of the layers after this one
             const llm_graph_rec_leads * rec = nullptr
             ) const;

    //
    // inputs
    //

    // tok_scale: applied to token rows only
    ggml_tensor * build_inp_embd(ggml_tensor * tok_embd, float tok_scale = 1.0f) const;
    ggml_tensor * build_inp_pos() const;
    ggml_tensor * build_inp_attn_scale() const;
    ggml_tensor * build_inp_out_ids() const;
    ggml_tensor * build_inp_mean() const;
    ggml_tensor * build_inp_cls() const;

    ggml_tensor * build_inp_cross_embd() const;
    ggml_tensor * build_inp_pos_bucket_enc() const;
    ggml_tensor * build_inp_pos_bucket_dec() const;
    ggml_tensor * build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const;

    //
    // attention
    //

    ggml_tensor * build_attn_mha(
            ggml_tensor * q,       // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k,       // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v,       // [n_embd_head_v, n_head_v, n_tokens] (v_trans = false)
            ggml_tensor * kq_b,
            ggml_tensor * kq_mask,
            ggml_tensor * sinks,   // [n_head_q]
            ggml_tensor * v_mla,   // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                int64_t   n_kv_max,
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_no_cache * build_attn_inp_no_cache() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_no_cache * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_kv * build_attn_inp_kv() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v] // TODO: remove
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k  * build_attn_inp_k() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_k * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k_dsa * build_attn_inp_k_dsa() const;

    llm_graph_input_attn_k_dsa_iswa * build_attn_inp_k_dsa_iswa() const;

    llm_graph_input_attn_kv_msa * build_attn_inp_kv_msa(bool msa_enabled) const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_k_dsa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
            ggml_tensor * top_k, // [n_indexer_top_k, n_tokens]
                  float   kq_scale,
                    int   il) const;

    // ggml_top_k_mask with top_k viewed in the row layout of kq_mask, see ggml.h
    ggml_tensor * build_fused_top_k_mask(
            ggml_tensor * kq_mask, // [n_kv, n_tokens/n_stream, 1, n_stream]
            ggml_tensor * top_k,   // [n_top, n_tokens]
                    int   il) const;

    llm_graph_input_attn_kv_iswa * build_attn_inp_kv_iswa() const;

    llm_graph_input_dsv4 * build_inp_dsv4() const;

    // note: if k_cur or v_cur are not provided, they will not be stored in the memory
    ggml_tensor * build_attn(
            llm_graph_input_attn_kv_iswa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens] optional
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens] optional
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k_iswa * build_attn_inp_k_iswa() const;

    // note: if k_cur is not provided, it will not be stored in the memory
    // note: the K cache is used as V (MLA-style attention)
    ggml_tensor * build_attn(
            llm_graph_input_attn_k_iswa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens] optional
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens] optional
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_cross * build_attn_inp_cross() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_cross * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    //
    // recurrent
    //

    // TODO: move this implementation to llama_memory_recurrent.
    //       this is analogous to llama_kv_cache::cpy_k / cpy_v
    //       when moving, avoid passing `ggml_cgraph` - only pass `ggml_context`. would likely need to split the
    //         implementation in 2 separate methods. the goal is to avoid calling `ggml_build_forward_expand` in
    //         `llama_memory_recurrent`
    ggml_tensor * build_rs(
            ggml_tensor * s,
            ggml_tensor * state_copy,
            ggml_tensor * state_copy_main,
                int32_t   state_size,
                int32_t   n_seqs,
               uint32_t   n_rs,
               uint32_t   rs_head,
               uint32_t   rs_size,
                int32_t   rs_zero,
            const llm_graph_get_rows_fn & get_state_rows = nullptr,
                   bool   expand_main = true) const;   // false = the main copy is not forced into the graph

    llm_graph_input_rs * build_rs_inp() const;

    ggml_tensor * build_rs(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
                int32_t   state_size,
                int32_t   n_seqs,
            const llm_graph_get_rows_fn & get_state_rows = nullptr,
                   bool   expand_main = true) const;   // --moe-stream-cpu-state-inplace

    ggml_tensor * build_rwkv_token_shift_load(
        llm_graph_input_rs * inp,
        const llama_ubatch & ubatch,
                       int   il) const;

    ggml_tensor * build_rwkv_token_shift_store(
             ggml_tensor * token_shift,
      const llama_ubatch & ubatch,
                     int   il) const;
    //
    // hybrid
    //

    llm_graph_input_mem_hybrid * build_inp_mem_hybrid() const;
    llm_graph_input_mem_hybrid_k * build_inp_mem_hybrid_k() const;

    llm_graph_input_mem_hybrid_iswa * build_inp_mem_hybrid_iswa() const;

    //
    // pooling
    //

    void build_pooling(
            ggml_tensor * cls,
            ggml_tensor * cls_b,
            ggml_tensor * cls_out,
            ggml_tensor * cls_out_b,
            ggml_tensor * cls_norm) const;

    //
    // sampling (backend sampling)
    //

    void build_sampling() const;

    //
    // dense (out)
    //

    void build_dense_out(
            ggml_tensor * dense_2,
            ggml_tensor * dense_2_b,
            ggml_tensor * dense_3) const;
};

// TODO: better name
int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional);

#include "llama-context.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-recurrent.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-moe-stream.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <atomic>
#include <set>
#include <tuple>

//
// llama_context
//

// A compute buffer that cannot be made is counted as device memory when the context has a GPU backend (the reserve does not say which
// buffer failed; the GPU's is the large one)
static bool llama_backends_have_gpu(const std::vector<ggml_backend_ptr> & backends) {
    for (const auto & b : backends) {
        if (ggml_backend_dev_type(ggml_backend_get_device(b.get())) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            return true;
        }
    }
    return false;
}

// A device the engine's per-context decisions count as a GPU (the pool's GPU, its tail, the async copy, the staging memory): a
// discrete GPU, or an integrated one when --moe-stream-uma 1 is on (without it an integrated GPU stays out of them, as before)
static bool llama_ctx_is_gpu(ggml_backend_dev_t dev, const llama_moe_stream * ms) {
    const auto dt = ggml_backend_dev_type(dev);
    return dt == GGML_BACKEND_DEVICE_TYPE_GPU || (dt == GGML_BACKEND_DEVICE_TYPE_IGPU && ms != nullptr && ms->uma_active);
}

// The GPU of the compute buffer the last reserve could not allocate (null: not known, or not a GPU)
static ggml_backend_dev_t llama_reserve_failed_dev() {
    ggml_backend_buffer_type_t buft = ggml_gallocr_last_failed_buft();
    ggml_backend_dev_t dev = buft != nullptr ? ggml_backend_buft_get_device(buft) : nullptr;
    return dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU ? dev : nullptr;
}

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

// GGML_PROFILE_HANDOFF (read once by ggml-base): the context-level timers of a call (graph build / reuse,
// set-input, the scheduler compute, the output readback, the host block in synchronize) and the decode call's begin / end
static const bool g_llama_handoff = ggml_handoff_on();

// MOE_STREAM_PROFILE_TOKENS read once per process, not on every decode call
static bool moe_stream_profile_tokens_on() {
    static const bool on = getenv("MOE_STREAM_PROFILE_TOKENS") != nullptr;
    return on;
}

static const llm_fused_op_probe llm_fused_op_top_k_mask_probe = {
    /*.op               =*/ LLM_FUSED_OP_TOP_K_MASK,
    /*.name             =*/ "fused top-k mask",
    /*.n_tokens_per_seq =*/ 1,
};
// The model graphs that can build a layer range for --moe-stream-layer-major (llm_graph_params lm_*): Ling-3.0
// (bailingmoe3.cpp), Flint (flint.cpp) and qwen4exp (qwen4exp.cpp). Flint's sliding-window cache replays an earlier ubatch with
// that ubatch's own cell records (llama_kv_cache_context::cells_rec); qwen4exp's hybrid-idx memory replays its k-pool layout prefixes
static bool llama_arch_layer_major(llm_arch arch) {
    return arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_FLINT || arch == LLM_ARCH_QWEN4EXP;
}

static std::vector<int64_t> lm_call_bytes_per_tok(const llama_hparams & hp, bool split = true) {
    const int64_t hc  = std::max<int64_t>(1, hp.dsv4_hc_mult);
    const int64_t big = (int64_t) hp.n_embd * 4;
    const int64_t sm  = (int64_t) hp.n_expert_used() * 4;
    if (!split) {
        return { hc * big };
    }
    if (hc > 1) {
        return { hc * big, big, big, hc * 4, sm, sm };
    }
    return { big, big, sm, sm };
}

static int64_t lm_call_bytes_sum(const llama_hparams & hp, bool split = true) {
    int64_t s = 0;
    for (int64_t b : lm_call_bytes_per_tok(hp, split)) {
        s += b;
    }
    return s;
}

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    // MTP uses the embd input for the hidden state
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd(),
                llm_arch_supports_mixed_batch(model.arch) && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT)) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // only the indexer models build this mask, skip the probe for the others
    cparams.fused_top_k_mask = true;
    cparams.auto_ftkm        = hparams.indexer_top_k > 0;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    // With --moe-stream-layer-major 1 the engine raises n_batch so a whole prompt arrives in one call (a layer-major
    // call reads each streamed layer's experts about once); --moe-stream-layer-major-batch N caps the raise. n_ubatch keeps the
    // user's value (the line below reads params.n_batch, not the raised one). Tools that cut a prompt themselves read llama_n_batch().
    if (const auto * ms = ctx_moe_stream(); ms && ms->p1.layer_major && cparams.causal_attn && llama_arch_layer_major(model.arch)) {
        const uint32_t cap = ms->p1.layer_major_batch > 0 ? std::min(cparams.n_ctx, ms->p1.layer_major_batch) : cparams.n_ctx;
        if (cap > cparams.n_batch) {
            fprintf(stderr, "%s: --moe-stream-layer-major 1 raises n_batch %u -> %u (%s) so a prompt of up to %u tokens is one call\n",
                    __func__, cparams.n_batch, cap, ms->p1.layer_major_batch > 0 ? "--moe-stream-layer-major-batch" : "the context size", cap);
            cparams.n_batch = cap;
        }
    }

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
            /*.no_alloc  =*/ model.no_alloc(),
        };

        // MOE_STREAM_TEST_CTX_ALLOC_FAIL=1 (test only): with whole GPU MoE layers, the first context's memory fails as if device memory
        // were short, once per process, so the reload with one whole layer fewer (common) can be tested
        static std::atomic<bool> test_ctx_fail_done{false};
        static const bool test_ctx_fail = getenv("MOE_STREAM_TEST_CTX_ALLOC_FAIL") && atoi(getenv("MOE_STREAM_TEST_CTX_ALLOC_FAIL")) != 0;
        if (test_ctx_fail && !model.no_alloc() && ctx_moe_stream() && !ctx_moe_stream()->gpu_layers.empty() && !test_ctx_fail_done.exchange(true)) {
            llama_memory_note_alloc_failure(true, ctx_moe_stream()->pool_dev);   // (as if on the GPU of the VRAM pool)
            throw std::runtime_error("failed to allocate buffer for kv cache (TEST ONLY: MOE_STREAM_TEST_CTX_ALLOC_FAIL)");
        }

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        // Not with streamed MoE layers: its copies of every input that crosses devices (4) would hold the RAM pools that the prompt matmuls
        // copy to a GPU (85-234 GiB asked for before the retry without it); whole layers only keep it
        if (pipeline_parallel && ctx_moe_stream() != nullptr && !ctx_moe_stream()->il_of_moe.empty()) {
            pipeline_parallel = false;
            fprintf(stderr, "%s: pipeline parallelism: off - %zu MoE layers are streamed (its copies of every input that crosses devices would "
                    "include the RAM pools)\n", __func__, ctx_moe_stream()->il_of_moe.size());
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        sched_reserve();

        // --moe-stream-vram-experts: the VRAM expert pool is allocated here, after the KV cache and the compute
        // buffers, so the free VRAM it checks against is what is really left; the compute buffers are then reserved again,
        // because the graphs now carry the pool's device pass. The placement line is printed for every streaming run.
        if (auto * ms = ctx_moe_stream()) {
            ggml_backend_dev_t         gdev  = nullptr;
            ggml_backend_buffer_type_t gbuft = nullptr;
            ggml_backend_t             gback = ms_pool_backend();   // the GPU of the VRAM pool (--moe-stream-vram-device, else the first GPU)
            if (gback == nullptr && !backends.empty() && getenv("MOE_STREAM_TEST_VRAM_ON_CPU") && atoi(getenv("MOE_STREAM_TEST_VRAM_ON_CPU")) != 0) {
                gback = backends.back().get();   // TEST ONLY: exercise vx_setup / chunked VRAM pools / vram_follow on CPU-only hosts
            }
            if (gback != nullptr) {
                gdev  = ggml_backend_get_device(gback);
                gbuft = ggml_backend_get_default_buffer_type(gback);
            }
            if (ms->uma_active && ms->uma_dev != nullptr) {
                for (auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == ms->uma_dev) {
                        ms->uma_backend = backend.get();
                        break;
                    }
                }
            }
            {   // the pool's tail needs no pipeline copies (the scheduler's n_copies is 1), op offload, and every GPU prompt wave on the
                // pool's GPU: one GPU, or with several the scheduler's op offload limited to the pool's GPU (sched_new)
                // MOE_STREAM_TEST_MULTI_GPU_NO_TAIL=1 (test only): with several GPUs no tail (no tail waves, no copy-ahead)
                int n_gpu_t = 0;
                for (auto & backend : backends) {
                    n_gpu_t += llama_ctx_is_gpu(ggml_backend_get_device(backend.get()), ms) ? 1 : 0;
                }
                static const bool test_no_tail = getenv("MOE_STREAM_TEST_MULTI_GPU_NO_TAIL") && atoi(getenv("MOE_STREAM_TEST_MULTI_GPU_NO_TAIL")) != 0;
                ms->n_gpu_ctx    = n_gpu_t;
                ms->tail_test_mg = n_gpu_t > 1 && test_no_tail;
                ms->tail_allowed = (n_gpu_t == 1 || (n_gpu_t > 1 && gback != nullptr && !ms->tail_test_mg)) && !cparams.pipeline_parallel &&
                                   cparams.op_offload;
            }
            {   // the op scratch pool's size. 8 bytes per hidden element per ubatch row is a CALIBRATION on Ling-3.0-flash
                // (18.0 KiB per row measured = 7.2-7.6 B per hidden element at -ub 2048, rounded up), not a derivation: the batch-sized users of the
                // pool (the quantized input of the expert matmul, the fp16 weights / activations of cuBLAS) also grow with top-k and the FFN width, so
                // another model sets --moe-stream-vram-scratch (or the autotuner will); a request above the mapped size still works, the pool grows.
                // Flash attention's fp16 copies of a quantized K / V are NOT in this pool: they sit in the attention node's output inside the compute
                // buffer (fattn-common.cuh:47-84, fattn.cu:720-756, ggml-cuda.cu:939-943), which the closed loop measures.
                const int64_t ub_s = (int64_t) std::min<uint32_t>(cparams.n_ctx, cparams.n_ubatch);
                ms->vx_prewarm(gback, ub_s*(int64_t) model.hparams.n_embd*8);
            }   // the backend's lazy cuBLAS memory exists before vx_setup reads the free VRAM
            // A layer-major call's per-call tensors (the residual and the normed expert input in F32, the router's
            // picks and weights) live in the VRAM pool. Not a tail extension reserved here for a call
            // of n_batch tokens (-c with layer-major: ~1 GB on Flint at 65k, so a 6 GB card could not start with the pool) - each call
            // takes the places its real size needs from the pool's expert places (lm_lend_place), which VRAM follow's fill fills again
            ms->lm_tail_req = 0;
            if (ms->vx_setup(gdev, gbuft, (int64_t) std::min<uint32_t>(cparams.n_ctx, cparams.n_ubatch), (int64_t) cparams.n_ctx,
                        [this]() {   // (a trial: a failure here shrinks the pool - E1 does not halve -ub inside it)
                            sched_need_reserve = true;
                            rtrial_in_trial = true;
                            rtrial_trial_n += 1;
                            try {
                                sched_reserve();
                            } catch (...) {
                                rtrial_in_trial = false;
                                throw;
                            }
                            rtrial_in_trial = false;
                        },
                        [this]() {   // the first compute buffers go before the pool with its tail is allocated
                            synchronize();
                            for (auto & res : gf_res_prev) {
                                res.reset();
                            }
                            gf_res_reserve.reset();
                            gf_res_prev_active = nullptr;
                            sched.reset();
                            sched_need_reserve = true;
                        })) {
                sched_need_reserve = true;
                sched_reserve();
            }
            ms->vx_report(gdev);
            llama_memory_note_alloc_failure(false);   // (the pool sizing's trial reserves fail on purpose; they are not the context's failure)
            // The whole-layer fit leaves the pool's GPU's --moe-stream-vram-free to the pool. When the pool could not keep it (whole layers
            // took too much) and that GPU holds a whole layer, this context fails as a device allocation does: common loads the model again
            // with one whole layer fewer, until the keep-free fits or no whole layer is left there (then the run keeps less free, as before)
            if (ms->vx_keep_short && ms->whole_layer_to_drop(ms->pool_dev) >= 0 && !model.no_alloc()) {
                size_t f = 0, t = 0;
                ggml_backend_dev_memory(ms->pool_dev, &f, &t);
                fprintf(stderr, "%s: --moe-stream-vram-free: %.0f MiB free on %s next to the whole GPU MoE layers, under the %.0f MiB to keep "
                        "free - one whole layer fewer\n", __func__, f/1048576.0, ggml_backend_dev_name(ms->pool_dev), ms->p1.vram_free/1048576.0);
                llama_memory_note_alloc_failure(true, ms->pool_dev);
                throw std::runtime_error("the VRAM pool cannot keep --moe-stream-vram-free free next to the whole GPU MoE layers");
            }
            if (ms->lend_on && ms->lend_owner == this) {   // the final reserve's decision
                const auto U = ms->lend_places_at((int64_t) cparams.n_ctx);
                int64_t nU = 0;
                for (const int32_t u : U) { nU += u; }
                std::string g;
                for (size_t i = 0; i < ms->lend_kv_pts.size(); i++) {
                    g += format("%s%lld: %.0f..%.0f", i ? ", " : "", (long long) ms->lend_kv_pts[i], ms->lend_grid[i].front()/1048576.0,
                                ms->lend_grid[i].back()/1048576.0);
                }
                fprintf(stderr, "%s: VRAM lending for prompts: on - a prompt call of more than %d tokens plans its GPU compute buffer (ggml's own plan, MiB by "
                        "context for %lld..%lld tokens: %s) into lent VRAM pool places (%lld places at -c %u, the front of each group's held "
                        "places, refilled after the prompt); the real compute buffer %.1f MiB\n", __func__, ms->p1.vram_lend_short,
                        (long long) ms->lend_t_pts.front(), (long long) ms->lend_t_pts.back(), g.c_str(), (long long) nU, cparams.n_ctx,
                        ms->lend_real/1048576.0);
            } else if (ms->vx_ready && !ms->p1.vram_lend_prompt) {
                fprintf(stderr, "%s: VRAM lending for prompts: off (--moe-stream-vram-lend-prompt 0, the default): a prompt's big GPU tensors stay in the "
                        "compute buffer\n", __func__);
            } else if (ms->vx_ready && ms->lend_test_off) {
                fprintf(stderr, "%s: VRAM lending for prompts: off (TEST ONLY: MOE_STREAM_TEST_LEND_OFF)\n", __func__);
            } else if (ms->vx_ready && ms->p1.vram_follow && ms->p1.vram_free == 0) {
                fprintf(stderr, "%s: VRAM lending for prompts: off - the VRAM pool's count is pinned (--moe-stream-vram-experts without --moe-stream-vram-free): "
                        "a smaller compute buffer would give it no places\n", __func__);
            }

            // A layer-major call's tensors take the pool's expert places for the call (lm_lend_place). When even every place
            // cannot hold a call of n_batch tokens (each tensor is one range in one pool group: Flint on a 6 GB card at 32k, 13 places per layer),
            // nor the free VRAM, the call is capped at what fits (a longer prompt then comes in several calls; the tools cut it by n_batch).
            // With the pool off (not even 1 per layer fits) the call's tensors take a GPU buffer of their own: the cap by
            // free VRAM alone (with no cap, a prompt over ~5k tokens failed that allocation with a 131k context on a 6 GB card)
            const bool lm_gpu_own = !ms->lm_lend_on() && gdev != nullptr && ggml_backend_dev_type(model.dev_layer(0)) != GGML_BACKEND_DEVICE_TYPE_CPU;
            if (ms->p1.layer_major && cparams.n_batch > cparams.n_ubatch && llama_arch_layer_major(model.arch) && (ms->lm_lend_on() || lm_gpu_own)) {
                const auto    bpt       = lm_call_bytes_per_tok(model.hparams);
                const int64_t bpt_total = lm_call_bytes_sum(model.hparams);
                size_t vfree = 0, vtot = 0;
                // (the own buffer is made on layer 0's device - with several GPUs maybe not the VRAM pool's)
                ggml_backend_dev_memory(lm_gpu_own ? model.dev_layer(0) : gdev, &vfree, &vtot);
                const int64_t t_lend = ms->lm_lend_max_tokens(bpt);
                // (a margin for growth: with the pool off, the engine's own growth estimate - the lazy CUDA memory still to come, as the
                // pool's fit uses it - instead of a fixed 512 MiB, which on a 4.6 GB start left 0 tokens for 221 MiB free. Only
                // where vx_prewarm created the cuBLAS handle and mapped the scratch pool before this check: the estimate leaves them out
                // (~130 MiB, created later without the VRAM pool's flags), so otherwise the 512 MiB stays)
                const int64_t margin = lm_gpu_own && ms->vx_prewarmed ? ms->vx_growth_bytes(cparams.n_ubatch, cparams.n_ctx, nullptr) :
                                       ((int64_t) 512 << 20);
                const int64_t t_free = std::max<int64_t>(0, (int64_t) vfree - margin)/bpt_total;
                const int64_t t_fit  = std::max(t_lend, t_free)/cparams.n_ubatch*cparams.n_ubatch;
                if (t_fit < (int64_t) cparams.n_batch) {
                    const uint32_t nb = (uint32_t) std::max<int64_t>(cparams.n_ubatch, t_fit);
                    if (lm_gpu_own) {
                        fprintf(stderr, "%s: the VRAM pool is off - a layer-major call's tensors (a GPU buffer of their own) fit the "
                                "free VRAM (%.0f MiB, less %.1f MiB of growth) for %lld tokens: n_batch %u -> %u (a longer prompt comes in several "
                                "calls)\n", __func__, vfree/1048576.0, margin/1048576.0, (long long) t_free, cparams.n_batch, nb);
                    } else {
                        fprintf(stderr, "%s: a layer-major call's tensors fit the VRAM pool's places for %lld tokens (free VRAM %.0f MiB for "
                                "%lld): n_batch %u -> %u (a longer prompt comes in several calls)\n", __func__, (long long) t_lend, vfree/1048576.0,
                                (long long) t_free, cparams.n_batch, nb);
                    }
                    cparams.n_batch = nb;
                }
            }

            // Which MoE layers --moe-stream-cpu-weighted-sum can fuse, and why the others
            // cannot - the same rules as build_moe_ffn's tail, except the per-graph one (a prompt batch that a GPU backend
            // would take by op offload keeps the plain sum), which depends on the batch
            if (ms->p1.cpu_wsum) {
                ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                const uint32_t n_trunk = model.hparams.n_layer();
                std::string why[5];   // fuse, GPU layer, last CPU layer before a GPU layer, experts not in host memory, VRAM pass
                int n_moe = 0, cnt[5] = { 0 };
                for (uint32_t il = 0; il < n_trunk && il < model.layers.size(); il++) {
                    const ggml_tensor * dw = model.layers[il].ffn_down_exps;
                    if (!dw || model.hparams.n_expert_used(il) <= 1) {
                        continue;   // not an MoE layer with a sum
                    }
                    n_moe++;
                    const ggml_tensor * dsrc = dw->view_src ? dw->view_src : dw;
                    const ggml_backend_dev_t next = il + 1 < n_trunk ? model.dev_layer(il + 1) : model.dev_output();
                    const int k = model.dev_layer(il) != cpu_dev ? 1
                                : next != cpu_dev ? 2
                                : (ms->uma_gpu_pool || !(dsrc->buffer && ggml_backend_buffer_is_host(dsrc->buffer))) ? 3
                                : ms->vram_serves((int32_t) il) ? 4 : 0;
                    cnt[k]++;
                    why[k] += (why[k].empty() ? "" : ",") + std::to_string(il);
                }
                static const char * const names[5] = { "fuse", "GPU layer", "last CPU layer before a GPU layer",
                                                        "experts not in host memory", "VRAM experts pass" };
                std::string line = format("--moe-stream-cpu-weighted-sum: %d of %d MoE layers fuse", cnt[0], n_moe);
                for (int k = 1; k < 5; k++) {
                    if (cnt[k] > 0) {
                        line += format("; %d not: %s (layers %s)", cnt[k], names[k], why[k].c_str());
                    }
                }
                fprintf(stderr, "%s: %s\n", __func__, line.c_str());
            }
        }

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }

    // The GPU half of the read-back check. The experts of a whole MoE layer are read by
    // the stock loader out of the model file into VRAM, and nothing else in the engine ever looks at
    // them, so this is the only place where a wrong offset in that path would be caught.
    if (ctx_moe_stream()) {
        llama_moe_stream_verify_gpu_tier(this);
    }

    // The context (and the VRAM expert pool) is built - idle-time loading may start after its pause (off: nothing); with
    // --moe-stream-idle-load 2 the RAM pools are filled here first (not for a model loaded without memory)
    if (auto * ms = ctx_moe_stream()) {
        ms->idle_arm(/* may_fill = */ !model.no_alloc());
    }
}

llama_context::~llama_context() {
    if (auto * ms = ctx_moe_stream(); ms && ms->lend_owner == this) {
        ms->lend_owner = nullptr;   // (a later context may lend)
        ms->lend_on    = false;
        ms->lend_free_pieces(lend_bufs);
        ms->lend_free_pieces(lend_bufs_old);
    }
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();
    if (auto * ms = ctx_moe_stream(); ms != nullptr) {   // only this context's backend (another context may still run on the model)
        for (auto & backend : backends) {
            if (ms->uma_backend == backend.get()) {
                ms->uma_backend = nullptr;
            }
        }
    }
    for (auto ev : lm_evs) {   // the per-layer events of the window's signals
        if (ev) {
            ggml_backend_event_free(ev);
        }
    }

    if (ctx_moe_stream() != nullptr && sched) {   // what the scheduler's re-plans and the VRAM lending place hook cost
        int64_t pl = 0, sy = 0, pn = 0; int nr = 0;
        ggml_backend_sched_get_alloc_times(sched.get(), &pl, &sy, &pn, &nr);
        int why[4];
        ggml_gallocr_get_realloc_reasons(why);
        fprintf(stderr, "llama_context: the scheduler re-planned %d graphs (GPU wait before them %.1f ms, planning %.1f ms); VRAM lending's place "
                "hook %.1f ms; the allocator's reasons (all its checks, every scheduler of the process): node count %d, leaf count %d, a tensor "
                "with memory of its own in the last plan has none now %d, a tensor bigger than its place %d\n", nr, sy/1000.0, pn/1000.0,
                pl/1000.0, why[0], why[1], why[2], why[3]);
    }
    if (!prompt_calls.empty()) {   // how a prompt was cut into calls (layer-major: L)
        std::string c;
        for (size_t i = 0; i < prompt_calls.size() && i < 24; i++) {
            c += format("%s%lld%s", i ? ", " : "", (long long) prompt_calls[i].first, prompt_calls[i].second ? " L" : "");
        }
        fprintf(stderr, "llama_context: %lld prompt calls: %s%s\n", (long long) n_prompt_calls, c.c_str(), n_prompt_calls > 24 ? ", ..." : "");
    }
    ggml_handoff_dump();   // the per-word summary (nothing unless GGML_PROFILE_HANDOFF is set)

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!model.no_alloc() && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }

    if (cparams.auto_ftkm) {
        LLAMA_LOG_INFO("%s: resolving fused top-k mask support:\n", func);
        resolve(llm_fused_op_top_k_mask_probe, cparams.fused_top_k_mask);
        cparams.auto_ftkm = false;
    }
}

static int llama_graph_n_input_tensors(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    for (const auto & [tensor, nodes] : users) {
        GGML_ASSERT(tensor->op == GGML_OP_NONE);
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

void llama_context::sched_new(size_t max_nodes, bool parallel) {
    sched.reset(ggml_backend_sched_new(
            backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, parallel, cparams.op_offload));

    ggml_backend_sched_set_realloc_callback(sched.get(), sched_graph_realloc, this);
    // --moe-stream-vram-device, or streamed layers with several GPUs: the prompt matmuls of the RAM pools on the pool's GPU only (without
    // the flag that is the first GPU, where the offload went anyway; with it fixed, the pool's tail can take their copies)
    if (auto * ms = ctx_moe_stream(); ms && (ms->pool_dev_set || (!ms->il_of_moe.empty() && sched_n_gpu() > 1))) {
        ggml_backend_sched_set_op_offload_backend(sched.get(), ms_pool_backend());
    }
    if (auto * ms = ctx_moe_stream()) {   // this context's prompt graphs may plan into lent pool places
        static const bool lend_off = getenv("MOE_STREAM_TEST_LEND_OFF") && atoi(getenv("MOE_STREAM_TEST_LEND_OFF")) != 0;
        ms->lend_test_off = lend_off;
        ms->lend_free_pieces(lend_bufs);   // (a new scheduler: nothing set on it; the old one's allocator never frees given pieces)
        ms->lend_free_pieces(lend_bufs_old);
        lend_U.clear();
        lend_gen += 1;
        if (ms->p1.vram_lend_prompt) {   // (--moe-stream-vram-lend-prompt 1; off = the plain compute buffer)
            // One context lends - the first that sets it up; a second context on the same model (an MTP draft, another server
            // context) keeps its compute buffer as usual, so two contexts never lend the same places
            if (ms->lend_owner == nullptr || ms->lend_owner == this) {
                ms->lend_owner = this;
            } else {
                static bool said = false;
                if (!said) {
                    said = true;
                    fprintf(stderr, "%s: VRAM lending for prompts: off for this context - another context on this model lends the VRAM pool's places "
                            "(this one keeps its compute buffer)\n", __func__);
                }
            }
        }
    }
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve && !sampler_reserve_due()) {
        sampling.changed = false;   // the samplers changed back to what the reserve was made with
        return;
    }

    sched_need_reserve = false;
    // The samplers this reserve is made with
    sampling.changed = false;
    sampling.reserved.clear();
    for (const auto & [seq_id, smpl] : sampling.samplers) {
        sampling.reserved[seq_id] = llama_sampler_backend_shape(smpl);
    }

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);   // (halved below when the reserve does not fit)

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_prev_active = nullptr;

    sched_new(max_nodes, cparams.pipeline_parallel);

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // VRAM lending for prompts: with the VRAM pool sized from free VRAM, a prompt call's GPU plan lives in lent pool places and the real
    // compute buffer holds only the reply graphs (the prompt graphs below are then planned, not allocated)
    bool lend_now = false;
    if (auto * ms = ctx_moe_stream(); ms && ms->lend_owner == this) {
        lend_clear();
        ms->lend_on = false;
        if (ms->lend_possible() && !model.no_alloc() && !cparams.pipeline_parallel) {
            lend_now = lend_decide(mctx.get());
        }
    }

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * ms_w = ctx_moe_stream();
        const bool wide_fail = ms_w && ms_w->el_wide && ms_w->el_test_wide_fail;   // TEST ONLY: the wide reserve "fails"
        // MOE_STREAM_TEST_PROMPT_RESERVE_RESERVE_ABOVE=<rows> (test only) - a prompt reserve of more rows "fails"
        // (not in vx_setup's trials); MOE_STREAM_TEST_PROMPT_RESERVE_TRIAL_FAIL=<n> (test only): the n-th trial reserve of the pool sizing "fails"
        static const int64_t rtrial_test = getenv("MOE_STREAM_TEST_PROMPT_RESERVE_RESERVE_ABOVE") ? atoll(getenv("MOE_STREAM_TEST_PROMPT_RESERVE_RESERVE_ABOVE")) : 0;
        static const int     rtrial_trial_fail = getenv("MOE_STREAM_TEST_PROMPT_RESERVE_TRIAL_FAIL") ? atoi(getenv("MOE_STREAM_TEST_PROMPT_RESERVE_TRIAL_FAIL")) : 0;
        const bool rtrial_force = rtrial_in_trial ? (rtrial_trial_fail > 0 && rtrial_trial_n == rtrial_trial_fail) : (rtrial_test > 0 && n_tokens > rtrial_test);
        if (rtrial_force && rtrial_in_trial) {
            fprintf(stderr, "%s: TEST ONLY (MOE_STREAM_TEST_PROMPT_RESERVE_TRIAL_FAIL): trial reserve %d of the pool sizing fails\n", __func__, rtrial_trial_n);
        }
        auto * gf = wide_fail || rtrial_force ? nullptr : graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.no_alloc() || lend_now, model.no_alloc() ? backend_buf_exp_size.data() : nullptr);   // (VRAM lending for prompts: planned only)
        if (!gf && ms_w && ms_w->el_wide) {   // the wide prompt-wave reserve did not fit - the start size
            ms_w->el_wide_off(wide_fail ? "failed (TEST ONLY: MOE_STREAM_TEST_WIDE_FAIL)" : "did not fit in VRAM");
            gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                    model.no_alloc(), model.no_alloc() ? backend_buf_exp_size.data() : nullptr);
        }
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched_new(max_nodes, false);
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            // The compute buffers of a prompt ubatch do not fit -> half the ubatch until they do (prompts slower),
            // one line each; only then the error
            while (!gf && !rtrial_in_trial && cparams.n_ubatch > 16) {
                const uint32_t ub_old = cparams.n_ubatch;
                cparams.n_ubatch = std::max<uint32_t>(16, ub_old/2);
                n_tokens         = std::min(cparams.n_ctx, cparams.n_ubatch);
                n_outputs_pp     = std::min(n_tokens, cparams.n_outputs_max);
                fprintf(stderr, "%s: the compute buffers for -ub %u did not fit%s - -ub %u instead (prompts slower)\n", __func__, ub_old,
                        rtrial_test > 0 ? " (TEST ONLY: MOE_STREAM_TEST_PROMPT_RESERVE_RESERVE_ABOVE)" : "", cparams.n_ubatch);
                gf = rtrial_test > 0 && n_tokens > rtrial_test ? nullptr : graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                        model.no_alloc(), model.no_alloc() ? backend_buf_exp_size.data() : nullptr);
            }
            if (!gf) {
                llama_memory_note_alloc_failure(llama_backends_have_gpu(backends), llama_reserve_failed_dev());
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_pp = this->n_input_tensors;
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.no_alloc());
        if (!gf) {
            llama_memory_note_alloc_failure(llama_backends_have_gpu(backends), llama_reserve_failed_dev());
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_tg = this->n_input_tensors;
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    if (!lend_now) {   // (VRAM lending for prompts: the prompt graphs plan into lent pieces - the real buffer stays the reply graphs')
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve(n_tokens, 1,      n_outputs_pp, mctx.get(), model.no_alloc());
                break;
            default:
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.no_alloc());
        };

        if (!gf) {
            llama_memory_note_alloc_failure(llama_backends_have_gpu(backends), llama_reserve_failed_dev());
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    lm_measure_expert_graph();   // the expert graph's room per chunk (the -ub 1 fix)

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.no_alloc()) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                __func__,
                diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                val(n_nodes_pp, n_nodes_tg).c_str(),
                val(n_splits_pp, n_splits_tg).c_str(),
                val(n_inputs_pp, n_inputs_tg).c_str(),
                val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    // --moe-stream-nonpool-measured 1: the first context's own KV cache and host compute buffers, before a slot holds an expert. The buffers
    // are written once here, so their pages count in the process's memory from now on and every later look can subtract them whole
    if (auto * ms = ctx_moe_stream(); ms && ms->p1.nonpool && !ms->np_started && !model.no_alloc()) {
        int64_t kv_host = 0, compute_host = 0;
        if (memory) {
            for (const auto & [buft, size] : memory->memory_breakdown()) {
                kv_host += ggml_backend_buft_is_host(buft) ? (int64_t) size : 0;
            }
        }
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            compute_host += ggml_backend_buft_is_host(backend_buft[i]) ? (int64_t) ggml_backend_sched_get_buffer_size(sched.get(), backend_ptrs[i]) : 0;
        }
        ggml_backend_sched_clear_host_buffers(sched.get());
        ms->np_start(kv_host, compute_host);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    const int64_t t_ho = g_llama_handoff ? ggml_time_us() : 0;
    auto * ms_tt = ctx_moe_stream() && ctx_moe_stream()->vf_test_tick_timers ? ctx_moe_stream() : nullptr;   // TEST ONLY
    const int64_t t_tt = ms_tt ? ggml_time_us() : 0;
    const int64_t t_prof = ggml_prof_enabled() ? ggml_time_us() : 0;
    ggml_backend_sched_synchronize(sched.get());
    if (t_prof) {   // --moe-stream-timing: the host's block on the device's last work
        ggml_prof_add(GGML_PROF_SYNC, ggml_time_us() - t_prof);
    }
    if (ms_tt) {
        ms_tt->vf_tt_sync_us += ggml_time_us() - t_tt;   // MOE_STREAM_TEST_VF_TICK_TIMERS: the caller's GPU wait
    }
    if (g_llama_handoff) {
        ggml_handoff_ctx_add(GGML_HANDOFF_C_SYNC, ggml_time_us() - t_ho);
    }

    // The graph of the last recorded word is done (a no-op unless --moe-stream-record is on)
    if (auto * mstream = ctx_moe_stream()) {
        mstream->rec_word_end(ggml_time_us());
    }

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

llama_moe_stream * llama_context::ctx_moe_stream() const {
    return cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT ? get_model().moe_stream() : nullptr;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    if (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) {
        // these flags change the graph shape
        sched_need_reserve = true;
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    // no scheduler reserve needed because graph shapes must not depend on causal_attn, a flip only rebuilds the graph
    //sched_need_reserve = true;
}

bool llama_context::get_causal_attn() const {
    return cparams.causal_attn;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sampler_changed();
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sampler_changed();

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sampler_changed();
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sampler_changed();

    return true;
}

// Upstream re-reserves the scheduler on every sampler change; llama-server clears a slot's sampler at its release and sets
// a new chain at its next launch, so with --backend-sampling every request paid a full sched_reserve (48-55 ms on Ling Flash). The
// reserve depends on the samplers only through the nodes they add to the graph, so it runs when the samplers' graph fingerprints
// (llama_sampler_backend_shape) differ from the ones it was last made with - a changed chain (other sampling settings), an added or a
// removed one still re-reserves at the next decode. LLAMA_TEST_SAMPLER_RESERVE_ALWAYS=1 (test only): every change re-reserves (upstream)
void llama_context::sampler_changed() {
    static const bool always = getenv("LLAMA_TEST_SAMPLER_RESERVE_ALWAYS") && atoi(getenv("LLAMA_TEST_SAMPLER_RESERVE_ALWAYS")) != 0;
    if (always) {
        sched_need_reserve = true;
    }
    sampling.changed = true;
    // Upstream's reserve also dropped graph reuse; with the reserve skipped a new chain at the freed chain's address
    // (graph reuse compares sampler pointers) could reuse a graph the old chain built - no graph is reused across a sampler change
    // (the next build resets it; nothing is freed here, so a graph still running is untouched)
    gf_res_prev_active = nullptr;
}

bool llama_context::sampler_reserve_due() const {
    if (!sampling.changed) {
        return false;
    }
    if (sampling.samplers.size() != sampling.reserved.size()) {
        return true;
    }
    for (const auto & [seq_id, smpl] : sampling.samplers) {
        const auto it = sampling.reserved.find(seq_id);
        if (it == sampling.reserved.end() || it->second != llama_sampler_backend_shape(smpl)) {
            return true;
        }
    }
    return false;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

// TEST ONLY (MOE_STREAM_TEST_REQ_TIMES=1, src/llama-moe-stream.h): a prompt call's process_ubatch parts, summed per call
static struct { bool on = false; int64_t n = 0, n_build = 0, apply = 0, build = 0, inputs = 0, compute = 0, total = 0, tail = 0, bg = 0, al = 0; } g_tt_ub;   // tail, bg, al: tail_ubatch, build_graph, sched alloc

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    const int64_t tt0 = g_tt_ub.on ? ggml_time_us() : 0;   // TEST ONLY
    if (g_tt_ub.on) { g_tt_ub.n += 1; }
    // A later layer group replays the first apply(); MOE_STREAM_TEST_LM_NOREPLAY=1 (test-only, the control that must fail)
    // applies again
    static const bool lm_noreplay = getenv("MOE_STREAM_TEST_LM_NOREPLAY") && atoi(getenv("MOE_STREAM_TEST_LM_NOREPLAY")) != 0;
    if (mctx && !(lm_reapply && !lm_noreplay ? mctx->reapply() : mctx->apply())) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    const int64_t t_ho_graph = g_llama_handoff ? ggml_time_us() : 0;
    const int64_t tt1 = g_tt_ub.on ? ggml_time_us() : 0;

    // A GPU-wave graph's pool copies live in the VRAM pool's tail - before the graph is built or reused, no expert
    // may be held there (a no-op for every other ubatch and with the feature off)
    if (auto * ms_t = ctx_moe_stream(); ms_t && ms_t->tail_on) {
        ms_t->tail_ubatch(lm_phase == 2 ? (int64_t) cparams.n_ubatch : (int64_t) ubatch.n_tokens);   // layer-major: by the chunk
    }
    if (auto * ms_p = ctx_moe_stream(); ms_p && ms_p->ps_slot != nullptr) {
        ms_p->ps_ubatch(lm_phase == 2 ? 0 : (int64_t) ubatch.n_tokens);   // before the graph (never a layer-major expert graph)
    }
    if (g_tt_ub.on) { g_tt_ub.tail += ggml_time_us() - tt1; }

    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    // (VRAM lending for prompts: a graph is reused only in the lend it was allocated in - its tensors' memory is the pieces of that lend)
    if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams) && lend_gen_alloc == lend_gen) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;

        if (g_llama_handoff) {
            ggml_handoff_ctx_add(GGML_HANDOFF_C_REUSE, ggml_time_us() - t_ho_graph);
        }
    } else {
        gf_res_prev_active = nullptr;
        res->reset();
        if (g_tt_ub.on) { g_tt_ub.n_build += 1; }

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
        // The GPU producer's "router done" - the CPU split of a marked layer gets its inputs from the router slot
        if (ctx_moe_stream() && ctx_moe_stream()->ne_gpu_on()) {
            ggml_backend_sched_set_input_hook(sched.get(), llama_moe_stream_sched_input_hook, ctx_moe_stream());
        }
        // The GPU producer for the word-start sources - the split hook queues each source for the helper once its launch is issued
        if (ctx_moe_stream()) {
            ctx_moe_stream()->psplit_no_cb = ctx_moe_stream()->psplit_no_cb || cparams.cb_eval != nullptr;
        }
        if (ctx_moe_stream() && (ctx_moe_stream()->ws_gpu || ctx_moe_stream()->p1.prompt_split)) {   // (prompt split: the split end)
            ggml_backend_sched_set_split_hook(sched.get(), llama_moe_stream_sched_split_hook, ctx_moe_stream());
        }
        // No pipeline parallelism (the scheduler's events), or it stays off. With several GPUs the scheduler keeps a pending copy per GPU
        // and waits for each before a CPU split writes the CPU compute buffer (MOE_STREAM_TEST_MULTI_GPU_NO_ASYNC_H2D=1, test only: off there)
        if (ctx_moe_stream() && ctx_moe_stream()->p1.async_h2d) {
            static const bool test_no_h2d = getenv("MOE_STREAM_TEST_MULTI_GPU_NO_ASYNC_H2D") && atoi(getenv("MOE_STREAM_TEST_MULTI_GPU_NO_ASYNC_H2D")) != 0;
            int n_gpu = 0;
            for (auto & backend : backends) {
                n_gpu += llama_ctx_is_gpu(ggml_backend_get_device(backend.get()), ctx_moe_stream()) ? 1 : 0;
            }
            // Inputs can qualify only when the CPU backend's compute buffer type is the GPU's pinned host buffer type
            bool host_pinned = false;
            for (size_t b = 0; b < backend_ptrs.size() && b < backend_buft.size(); b++) {
                if (ggml_backend_dev_type(ggml_backend_get_device(backend_ptrs[b])) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    for (auto & backend : backends) {
                        ggml_backend_dev_t d = ggml_backend_get_device(backend.get());
                        if (llama_ctx_is_gpu(d, ctx_moe_stream())) {
                            host_pinned = host_pinned || backend_buft[b] == ggml_backend_dev_host_buffer_type(d);
                        }
                    }
                }
            }
            const bool mg_off = n_gpu > 1 && test_no_h2d;
            // Only builds made with CUDA 13 or newer (with the CUDA 12 runtime the copies made replies several times slower): a GPU
            // backend that offers ggml_backend_cuda_async_h2d_ok says it (MOE_STREAM_TEST_OLD_CUDA=1, test only: answers no)
            static const bool test_old_cuda = getenv("MOE_STREAM_TEST_OLD_CUDA") && atoi(getenv("MOE_STREAM_TEST_OLD_CUDA")) != 0;
            bool old_cuda = false;
            for (auto & backend : backends) {
                ggml_backend_dev_t d = ggml_backend_get_device(backend.get());
                ggml_backend_reg_t reg = llama_ctx_is_gpu(d, ctx_moe_stream()) ? ggml_backend_dev_backend_reg(d) : nullptr;
                auto fn = reg ? (bool (*)(void)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_async_h2d_ok") : nullptr;
                old_cuda = old_cuda || (fn != nullptr && (test_old_cuda || !fn()));
            }
            const bool ok = n_gpu >= 1 && !cparams.pipeline_parallel && host_pinned && !mg_off && !old_cuda;
            ggml_backend_sched_set_async_host_input(sched.get(), ok);
            static int said = -1;   // (printed again whenever the decision changes, e.g. for another context)
            if (said != (ok ? 1 : 0)) {
                said = ok ? 1 : 0;
                fprintf(stderr, "%s: async-h2d (--moe-stream-async-h2d 1): %s\n", __func__, ok
                        ? (n_gpu > 1 ? format("on - CPU -> GPU split inputs in the GPUs' pinned host buffer go as async H2D copies on each GPU's "
                                              "stream; a CPU split waits for every GPU with a copy still queued (%d GPUs)", n_gpu).c_str()
                                     : "on - CPU -> GPU split inputs in the GPU's pinned host buffer go as async H2D copies on the GPU's stream")
                        : mg_off ? "off - TEST ONLY: MOE_STREAM_TEST_MULTI_GPU_NO_ASYNC_H2D=1 with several GPUs"
                        : old_cuda ? (test_old_cuda ? "off - this build uses CUDA 12 or older; the async copy needs a CUDA 13 build "
                                                      "(TEST ONLY: MOE_STREAM_TEST_OLD_CUDA=1)"
                                                    : "off - this build uses CUDA 12 or older; the async copy needs a CUDA 13 build")
                        : format("off - it needs a GPU, no pipeline parallelism and the CPU compute buffer in the GPU's pinned "
                                 "host buffer (%d GPUs, pipeline parallel %d, pinned %d)", n_gpu, cparams.pipeline_parallel ? 1 : 0,
                                 host_pinned ? 1 : 0).c_str());
            }
        }

        //const auto t_start_us = ggml_time_us();

        const int64_t tt_bg = g_tt_ub.on ? ggml_time_us() : 0;
        gf = model.build_graph(gparams);
        if (g_tt_ub.on) { g_tt_ub.bg += ggml_time_us() - tt_bg; }

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        // The expert pools of the MoE stream live in host memory and the engine fills them while the
        // graph computes: a ubatch that needs more distinct experts than the pool holds is split
        // into waves and every wave after the first overwrites slots that the copy of the pool on the
        // device does not have yet - ggml copies a host weights tensor to a device once per graph.
        // So a graph with expert waves keeps the ops that read host weights, the expert matmuls,
        // on the host, next to the pool they read.
        {
            bool moe_waves = false;

            for (int i = 0; i < ggml_graph_n_nodes(gf) && !moe_waves; i++) {
                moe_waves = (ggml_graph_node(gf, i)->flags & (uint32_t) LLAMA_MOE_STREAM_TENSOR_WAVE) != 0;
            }

            // A graph with waves keeps its host-weight matmuls on the host, because the pool it
            // would copy is filled wave by wave *during* the graph. --moe-stream-gpu-waves N lifts that
            // for batches of at least N tokens, where the copies are amortised over the tokens.
            // Not with a repacked pool. op_offload sends the expert matmuls to the device, which
            // needs the pool copied there, and a CPU_REPACK buffer type has no cpy_tensor: with
            // --moe-stream-cpu-repack 1 and gpu-waves on, a 4k prompt named 20 distinct experts per
            // ubatch-layer instead of 389 and the text was "!!!!!!!!". This is the gate that has to hold;
            // the per-wave views in llama-graph.cpp are the matching half.
            // (the engine's one rule: waves_on_gpu)
            bool gpu_waves = false;
            if (moe_waves) {
                if (auto * mstream = ctx_moe_stream()) {
                    const int64_t n_w = lm_phase == 2 ? cparams.n_ubatch : ubatch.n_tokens;   // layer-major: by the chunk
                    gpu_waves = mstream->waves_on_gpu(n_w);
                    mstream->n_onewave_host += lm_phase != 2 && mstream->onewave_host(n_w) ? 1 : 0;   // a graph built with them on the CPU
                }
            }
            ggml_backend_sched_set_op_offload(sched.get(), cparams.op_offload && (!moe_waves || gpu_waves));
            res->ms_op_offload = cparams.op_offload && (!moe_waves || gpu_waves);   // the engine's experts_on_cpu reads it
            res->ms_chunk      = lm_phase == 2 ? (int64_t) cparams.n_ubatch : 0;

            // This graph's pool copies (waves on the GPU) get their memory in the VRAM pool's tail
            if (auto * ms_t = ctx_moe_stream(); ms_t && ms_t->tail_on) {
                const bool want = moe_waves && gpu_waves && ms_t->tail_hook_ready();
                ms_t->tail_real = want;
                ggml_backend_sched_set_copy_hook(sched.get(), want ? llama_moe_stream_sched_copy_hook : nullptr, ms_t);
            }
        }

        const int64_t tt_al = g_tt_ub.on ? ggml_time_us() : 0;
        bool alloc_ok = ggml_backend_sched_alloc_graph(sched.get(), gf);
        // Never a failed graph - one whose plan did not fit lends (more), then the real compute buffer
        if (auto * ms_g = ctx_moe_stream(); !alloc_ok && ms_g && ms_g->lend_owner == this && ms_g->lend_on) {
            alloc_ok = lend_alloc_retry();
        }
        if (alloc_ok) {
            lend_gen_alloc = lend_gen;
        }
        if (g_tt_ub.on) { g_tt_ub.al += ggml_time_us() - tt_al; }
        if (!alloc_ok) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        // TEST ONLY (MOE_STREAM_TEST_GUESS_CHECK): where the combined guess step's norm runs, once per graph build
        if (const auto * ms = ctx_moe_stream(); ms && ms->la_gchk != 0) {
            std::string where;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * t = ggml_graph_node(gf, i);
                int il = -1;
                if (sscanf(ggml_get_name(t), "moe_comb_norm-%d", &il) == 1) {
                    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched.get(), t);
                    where += format("%s %d:%s", where.empty() ? "" : ",", il, b ? ggml_backend_name(b) : "?");
                }
            }
            if (!where.empty()) {
                fprintf(stderr, "MOE_STREAM_TEST_GUESS_CHECK: graph of %u token(s): the combined norm of layer:backend%s\n",
                        ubatch.n_tokens, where.c_str());
            }
        }

        // TEST ONLY (MOE_STREAM_TEST_IG_ORDER=1, read once): the in-graph form's node order, once per graph build - for
        // every op A ("moe_ig_a-<il>"): its op B (the node that reads it), the nodes only op B's guess input needs (its ancestors
        // that are not ancestors of the ids: the guess, the far guesses and, with --moe-stream-pred-shared, the shared expert),
        // the layer's shared expert by name ("ffn_up/gate/down-<il>") and the nodes that read op B (the expert matmuls). The
        // CPU backend runs the nodes in this order: op A must be below the guess-only and shared-expert nodes, op B below its
        // readers
        static const bool ig_order = getenv("MOE_STREAM_TEST_IG_ORDER") && atoi(getenv("MOE_STREAM_TEST_IG_ORDER")) != 0;
        if (ig_order && ctx_moe_stream()) {
            const int nn = ggml_graph_n_nodes(gf);
            std::unordered_map<const ggml_tensor *, int> idx;
            for (int i = 0; i < nn; i++) {
                idx[ggml_graph_node(gf, i)] = i;
            }
            auto anc = [&](const ggml_tensor * root, std::unordered_set<const ggml_tensor *> & out) {
                std::vector<const ggml_tensor *> st = { root };
                while (!st.empty()) {
                    const ggml_tensor * t = st.back();
                    st.pop_back();
                    if (!t || !out.insert(t).second) {
                        continue;
                    }
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        if (t->src[j]) {
                            st.push_back(t->src[j]);
                        }
                    }
                    if (t->view_src) {
                        st.push_back(t->view_src);
                    }
                }
            };
            int n_layers = 0, n_bad = 0;
            std::string detail;
            for (int i = 0; i < nn; i++) {
                ggml_tensor * a = ggml_graph_node(gf, i);
                int il = -1;
                if (sscanf(ggml_get_name(a), "moe_ig_a-%d", &il) != 1) {
                    continue;
                }
                n_layers++;
                int ib = -1;
                for (int j = i + 1; j < nn && ib < 0; j++) {
                    ggml_tensor * t = ggml_graph_node(gf, j);
                    for (int k = 0; k < GGML_MAX_SRC; k++) {
                        if (t->src[k] == a) {
                            ib = j;
                        }
                    }
                }
                if (ib < 0) {
                    n_bad++;
                    detail += format(" [layer %d: no op B]", il);
                    continue;
                }
                ggml_tensor * b = ggml_graph_node(gf, ib);
                int g_lo = INT_MAX, g_hi = -1, g_n = 0;   // guess-only nodes
                if (b->src[1] && b->src[1] != a && b->src[2] == a) {   // op B with a guess: (ids, guess, op A)
                    std::unordered_set<const ggml_tensor *> ai, ag;
                    anc(b->src[0], ai);
                    anc(b->src[1], ag);
                    for (const ggml_tensor * t : ag) {
                        auto it = idx.find(t);
                        if (it != idx.end() && !ai.count(t)) {
                            g_lo = std::min(g_lo, it->second);
                            g_hi = std::max(g_hi, it->second);
                            g_n++;
                        }
                    }
                }
                int s_lo = INT_MAX, r_lo = INT_MAX, r_n = 0;   // shared expert by name; op B's readers
                for (int j = 0; j < nn; j++) {
                    ggml_tensor * t = ggml_graph_node(gf, j);
                    int sil = -1;
                    char nm[16] = {0};
                    if (sscanf(ggml_get_name(t), "ffn_%15[a-z]-%d", nm, &sil) == 2 && sil == il &&
                            (!strcmp(nm, "up") || !strcmp(nm, "gate") || !strcmp(nm, "down"))) {
                        s_lo = std::min(s_lo, j);
                    }
                    for (int k = 0; k < GGML_MAX_SRC; k++) {
                        if (t->src[k] == b) {
                            r_lo = std::min(r_lo, j);
                            r_n++;
                        }
                    }
                }
                const bool ok = (g_n == 0 || i < g_lo) && (s_lo == INT_MAX || i < s_lo) && r_n > 0 && ib < r_lo;
                n_bad += ok ? 0 : 1;
                if (!ok || n_layers <= 3) {
                    detail += format(" [layer %d: A %d, guess-only %d nodes %s, shared expert from %s, B %d, its %d readers from %d%s]",
                            il, i, g_n, g_n ? format("%d..%d", g_lo, g_hi).c_str() : "-",
                            s_lo == INT_MAX ? "(not in the graph by that name)" : std::to_string(s_lo).c_str(), ib, r_n,
                            r_lo == INT_MAX ? -1 : r_lo, ok ? "" : " WRONG ORDER");
                }
            }
            if (n_layers > 0) {
                fprintf(stderr, "MOE_STREAM_TEST_IG_ORDER: graph of %u token(s), %d nodes: %d in-graph layers, %d in the wrong order;%s\n",
                        ubatch.n_tokens, nn, n_layers, n_bad, detail.c_str());
            }
        }

        // The compute buffers the run really uses, so it can be compared with the reserve
        if (n_reserve_printed && !printed_compute_bufs) {
            printed_compute_bufs = true;

            std::string per_backend;

            for (const auto & backend_ptr : backends) {
                ggml_backend_t backend = backend_ptr.get();
                per_backend += format(" %s %.1f MiB", ggml_backend_name(backend),
                            ggml_backend_sched_get_buffer_size(sched.get(), backend)/1048576.0);
            }

            fprintf(stderr, "%s: the run's compute buffers (n_tokens %d):%s\n", __func__,
                    ubatch.n_tokens, per_backend.c_str());
        }

        gf_res_prev_active = res;

        if (g_llama_handoff) {
            ggml_handoff_ctx_add(GGML_HANDOFF_C_BUILD, ggml_time_us() - t_ho_graph);
        }
    }

    const int64_t tt2 = g_tt_ub.on ? ggml_time_us() : 0;
    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();
        const int64_t t_ho = g_llama_handoff ? ggml_time_us() : 0;

        // With --moe-stream-async-h2d the previous compute's last async H2D copies may still be queued
        // on the GPU, reading the CPU compute buffer that set_inputs and this graph's CPU splits write (nothing else synchronizes
        // between two computes) - wait for them first. Between decode words the GPU is already idle here (the logits were read)
        if (ctx_moe_stream() && ctx_moe_stream()->p1.async_h2d && ggml_backend_sched_get_async_host_input(sched.get())) {
            const int64_t t_tt = ctx_moe_stream()->vf_test_tick_timers ? ggml_time_us() : 0;
            ggml_backend_sched_wait_async_host_inputs(sched.get());   // wait only for pending async_h2d DMAs, not whole GPU stream
            if (t_tt) {
                ctx_moe_stream()->vf_tt_sync_us += ggml_time_us() - t_tt;   // TEST ONLY (MOE_STREAM_TEST_VF_TICK_TIMERS)
            }
        }

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        if (g_llama_handoff) {
            ggml_handoff_ctx_add(GGML_HANDOFF_C_SET_INPUT, ggml_time_us() - t_ho);
        }
        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const int64_t t_ho_compute = g_llama_handoff ? ggml_time_us() : 0;
    const int64_t tt3 = g_tt_ub.on ? ggml_time_us() : 0;
    if (auto * ms = ctx_moe_stream()) {   // this graph's op-offload decision (built now or reused), for experts_on_cpu
        ms->graph_op_offload = res->ms_op_offload;
        ms->graph_chunk      = res->ms_chunk;
    }
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (g_llama_handoff) {
        ggml_handoff_ctx_add(GGML_HANDOFF_C_COMPUTE, ggml_time_us() - t_ho_compute);
    }
    if (g_tt_ub.on) {
        const int64_t tt4 = ggml_time_us();
        g_tt_ub.apply += tt1 - tt0;  g_tt_ub.build += tt2 - tt1;  g_tt_ub.inputs += tt3 - tt2;  g_tt_ub.compute += tt4 - tt3;
        g_tt_ub.total += tt4 - tt0;
    }
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch_ext & batch_inp) {
    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != hparams.n_embd_inp_enc()) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always output all tokens and skip position continuity checks (output_all=true)
    if (!balloc->init(batch_inp, model.vocab, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

// A per-distance counter summed over the distances (index 0 holds what has no distance)
static int64_t llama_moe_stream_la_sum(const int64_t (&a)[9]) {
    int64_t t = 0;
    for (int i = 0; i < 9; i++) {
        t += a[i];
    }
    return t;
}

// The profile line's list counters (lookahead, protect named, the word-start op summed over its targets:
// added, bound by a reader, bound at the pick, removed unread; the picks it named READY / loading / on the list / none; the in-graph
// form: layers, op A -> op B time in us, needed reads op A queued)
static void llama_moe_stream_la_now(const llama_moe_stream_stats & ms, int64_t (&o)[40]) {
    const int64_t v[22] = { llama_moe_stream_la_sum(ms.dec.n_la_named), llama_moe_stream_la_sum(ms.dec.n_la_new),
            llama_moe_stream_la_sum(ms.dec.n_la_vote), llama_moe_stream_la_sum(ms.dec.n_la_pace),
            llama_moe_stream_la_sum(ms.dec.n_la_pool), llama_moe_stream_la_sum(ms.dec.n_la_bind),
            llama_moe_stream_la_sum(ms.dec.n_la_bind_pick), llama_moe_stream_la_sum(ms.dec.n_la_used),
            llama_moe_stream_la_sum(ms.dec.n_la_used_late), ms.dec.n_dr_became, ms.dec.n_dr_bound, ms.dec.n_dr_again,
            ms.dec.n_dr_removed, ms.dec.n_mw_back, ms.dec.n_mw_promoted, ms.dec.n_mw_cancelled, ms.dec.n_mw_again,
            ms.dec.n_pn_refused, ms.dec.n_pn_noroom, ms.dec.n_pn_kept, ms.dec.n_pn_kept_picked, ms.dec.n_pn_kept_unpicked };
    for (int i = 0; i < 22; i++) {
        o[i] = v[i];
    }
    for (int i = 22; i < 40; i++) {
        o[i] = 0;
    }
    for (int j = 0; j < 8; j++) {
        o[22] += ms.n_ws_add[j];
        o[23] += ms.n_ws_bind[j];
        o[24] += ms.n_ws_bind_pick[j];
        o[25] += ms.n_ws_rm[j];
        for (int i = 0; i < 4; i++) {
            o[26 + i] += ms.n_ws_pick[j][i];
        }
    }
    o[30] = ms.dec.n_ig_calls;   // in-graph layers, op A -> op B time (us), needed reads op A queued
    o[31] = ms.dec.t_ig_ab_us;
    o[32] = ms.dec.n_ig_reads;
    o[33] = ms.n_ws_post;   // the word-start source slots - posts, the helper's adds done, posts with a column too late
    o[34] = ms.n_ws_done;
    o[35] = ms.n_ws_late;
    o[36] = ms.dec.n_pr_early;   // gate/up computed before their down landed, read-gap events and their time, split reads
    o[37] = ms.dec.n_pr_gap;
    o[38] = ms.dec.t_pr_gap_us;
    o[39] = ms.dec.n_pr_split;
}

// --moe-stream-layer-major 1: may this call run layer by layer? Only the cases that were checked; everything else
// runs the plain path. MOE_STREAM_TEST_LM_GROUP=G (test-only) runs it with groups of G layers even without the setting (G = 0 = off).
// Calls llama_moe_stream::lm_layer_done for every streamed layer of [il_from, il_to), after its expert work of this call is
// queued. When the engine asks for events (lm_events), one per layer is recorded on the VRAM pool GPU's stream right here: once it
// has completed, nothing of this call reads the layer's RAM slots (null on a CPU-only run: the work has run when the graph returns)
void llama_context::lm_signal_done(int il_from, int il_to) {
    auto * ms = ctx_moe_stream();
    if (ms == nullptr || !ms->lm_in_call) {
        return;
    }
    const int n_layer = (int) model.hparams.n_layer();
    for (int il = il_from; il < il_to && il < n_layer; il++) {
        if (ms->layer(il) == nullptr) {
            continue;
        }
        ggml_backend_event_t ev = nullptr;
        if (ms->lm_events) {
            if (lm_ev_be == nullptr) {   // the GPU that reads the RAM slots in a prompt (the VRAM pool's: its prompt matmuls run there)
                lm_ev_be = ms_pool_backend();
            }
            if (lm_ev_be != nullptr) {
                if ((int) lm_evs.size() < n_layer) {
                    lm_evs.resize((size_t) n_layer, nullptr);
                }
                if (lm_evs[(size_t) il] == nullptr) {
                    lm_evs[(size_t) il] = ggml_backend_event_new(ggml_backend_get_device(lm_ev_be));
                }
                ev = lm_evs[(size_t) il];
                if (ev != nullptr) {
                    ggml_backend_event_record(ev, lm_ev_be);
                } else {
                    ggml_backend_synchronize(lm_ev_be);   // (review: no event on this backend - the layer's work is waited for here)
                }
            }
        }
        ms->lm_layer_done(il, ev);
    }
}

int llama_context::sched_n_gpu() const {
    int n = 0;
    for (const auto & backend : backends) {
        n += llama_ctx_is_gpu(ggml_backend_get_device(backend.get()), ctx_moe_stream()) ? 1 : 0;
    }
    return n;
}

ggml_backend_t llama_context::ms_pool_backend() const {
    const auto * ms = ctx_moe_stream();
    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (ms != nullptr && ms->pool_dev != nullptr ? dev == ms->pool_dev : llama_ctx_is_gpu(dev, ms)) {
            return backend.get();
        }
    }
    return nullptr;
}

bool llama_context::lm_layer_major_ok(uint32_t n_tokens_all) {
    static const int env_g = getenv("MOE_STREAM_TEST_LM_GROUP") ? atoi(getenv("MOE_STREAM_TEST_LM_GROUP")) : -1;
    const auto * mstream = ctx_moe_stream();
    const bool want = env_g > 0 || (env_g < 0 && mstream && mstream->p1.layer_major);
    if (!want || n_tokens_all <= cparams.n_ubatch || lm_force_off) {
        return false;   // off, or one ubatch: nothing to reorder (or this call's layer-major attempt failed)
    }
    if (!llama_arch_layer_major(model.arch) ||                      // the model builders that take a layer range (Ling-3.0, Flint, qwen4exp)
        cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP ||
        cparams.embeddings || !memory ||
        std::find(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(), true) !=
            cparams.embeddings_layer_inp.end()) {   // a draft (EAGLE3 / DFlash) reading layer inputs: a group graph has only its own
        return false;
    }
    int n_seq = 0;   // several sequences in one call (llama-server -np 2+): not tested yet
    for (int s = 0; s < (int) cparams.n_seq_max && s < LLAMA_MAX_SEQ; ++s) {
        n_seq += balloc->seq_pos_min(s) >= 0 ? 1 : 0;
    }
    return n_seq == 1;
}

// What a prompt call adds to the host memory that the RAM plan made at load does not hold
//  (1) layer-major's call tensors (decode() below) when they live in host memory (the layers' device is the CPU): the hidden state, and
//      the normed expert input and the router's picks and weights - 16 KB per word on Flint;
//  (2) what the memory keeps for the call's replays (a sliding-window cache's cell records and masks, llama-kv-cache.cpp);
//  (3) a pageable host compute buffer (a CPU-only run; with a GPU it is the GPU's pinned host memory, which holds its RAM from the
//      start): the reserve sized it for the whole context (the KQ masks at -c) and n_ubatch rows, but a page holds RAM only once a graph
//      writes it - so its share for the call's rows is counted until a call has written it (once; again only for what it grows by).
//      Flint case 3: 547 MiB, measured missing from the first count (a 4k layer-major prompt under a tight cap was killed after giving
//      back 11 slots per layer).
// An upper bound of each; 0 for a decode call (the caller).
int64_t llama_context::rg_call_need(uint32_t n_tokens_all, const llama_memory_context_i & mctx) {
    const auto & hparams = model.hparams;
    const size_t elt = cparams.flash_attn ? 2 : 4;   // the mask's element (build_attn_inp_kq_mask)
    int64_t need = 0;
    if (lm_layer_major_ok(n_tokens_all) && ggml_backend_dev_type(model.dev_layer(0)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        need += (int64_t) n_tokens_all * lm_call_bytes_sum(hparams, !lm_group_test());
    }
    need += (int64_t) mctx.replay_keep_bytes(elt);
    if (backend_cpu != nullptr && ggml_backend_sched_get_buffer_type(sched.get(), backend_cpu) == ggml_backend_cpu_buffer_type()) {
        // a graph's tensors grow with its ubatch's rows (the activations, the masks' rows): a call of fewer rows than n_ubatch writes
        // about that share of the buffer (a 58-word prompt counted at the whole 547 MiB gave back a slot it did not need)
        const int64_t cb   = (int64_t) ggml_backend_sched_get_buffer_size(sched.get(), backend_cpu);
        const int64_t rows = std::min<int64_t>(n_tokens_all, cparams.n_ubatch);
        const int64_t est  = (int64_t) ((double) cb*rows/std::max<uint32_t>(1, cparams.n_ubatch));
        need += std::max<int64_t>(0, est - rg_cb_counted);
        rg_cb_counted = std::max(rg_cb_counted, est);
    }
    return need;
}

int llama_context::lm_group() const {
    static const int env_g = getenv("MOE_STREAM_TEST_LM_GROUP") ? atoi(getenv("MOE_STREAM_TEST_LM_GROUP")) : -1;
    return env_g > 0 ? env_g : 1;
}

// MOE_STREAM_TEST_LM_GROUP=G (test only) runs the whole-layer form - groups of G whole layers, each ubatch its own expert
// stage, so the text is exactly the per-ubatch path's; unset = lm_phase 1 per ubatch + one expert graph per streamed layer
bool llama_context::lm_group_test() const {
    static const bool env_set = getenv("MOE_STREAM_TEST_LM_GROUP") && atoi(getenv("MOE_STREAM_TEST_LM_GROUP")) > 0;
    return env_set;
}

// A layer-major call that fails or is stopped leaves its ubatches finished up to different layers, so every position of the
// call is removed (the per-ubatch path removes only the failed ubatch's). A hybrid / recurrent memory cannot cut a sequence in the middle
// (seq_rm returns false and the hybrid memory then cuts nothing, its KV part included): such a sequence is cleared whole, and the
// call reports a compute error (false) instead of an abort, so the caller starts the sequence again (llama-server: the request
// fails and the slot's prompt is cleared; an abort would make it retry the batch on top of the cleared memory).
bool llama_context::lm_remove_call() {
    bool cut = true;
    for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const llama_pos p0 = balloc->seq_pos_min(s);
        if (p0 < 0) {
            continue;
        }
        LLAMA_LOG_WARN("%s: layer-major: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, p0);
        if (!memory->seq_rm(s, p0, -1)) {
            LLAMA_LOG_WARN("%s: layer-major: the memory cannot cut seq_id = %d at %d - the whole sequence is cleared\n", __func__, s, p0);
            memory->seq_rm(s, -1, -1);
            cut = false;
        }
    }
    return cut;
}

int llama_context::decode(const llama_batch_ext & batch_inp) {
    // TEST ONLY: one TT line per step of a prompt call, the process_ubatch sums at its end (after end_call: declared first)
    struct tt_call_t {
        bool on;
        ~tt_call_t() {
            if (on) {
                moe_tt("D ub_n", g_tt_ub.n);         moe_tt("D ub_n_build", g_tt_ub.n_build);
                moe_tt("D ub_apply_us", g_tt_ub.apply);  moe_tt("D ub_build_us", g_tt_ub.build);
                moe_tt("D ub_inputs_us", g_tt_ub.inputs); moe_tt("D ub_compute_us", g_tt_ub.compute);
                moe_tt("D ub_total_us", g_tt_ub.total);   moe_tt("D ub_tail_us", g_tt_ub.tail);
                moe_tt("D ub_buildgraph_us", g_tt_ub.bg); moe_tt("D ub_alloc_us", g_tt_ub.al);   moe_tt("D exit");
                g_tt_ub.on = false;
            }
        }
    } tt_call = { moe_tt_on() && batch_inp.tokens.size() > 1 };
    if (tt_call.on) {
        g_tt_ub = {};
        g_tt_ub.on = true;
        moe_tt("D enter", (long long) batch_inp.tokens.size());
    }
    // MOE_STREAM_PROFILE_TOKENS prints one line per decode call with the wall time of the call
    // and what it was spent on, so that a slow token can be attributed (host compute, device work,
    // copies, syncs). Nothing is measured or printed when the variable is not set.
    struct call_timer {
        bool     on;
        int64_t  t0;
        int64_t  n_tok;
        int64_t  p0[GGML_PROF_N_SLOTS];
        int64_t  stall0 = 0, op0 = 0;
        const llama_context * ctx = nullptr;
        // The decode causes of the not-READY experts and their read-backs, at the start of the call
        int64_t  cause0[4] = {0, 0, 0, 0}, tcause0[4] = {0, 0, 0, 0}, rb80 = 0, rb640 = 0;
        // The waiting list's entries added / bound / removed by reason, and the goal number, at the start of the call
        int64_t  pl0[12] = {0}, goal0[2] = {0, 0};   // + [9] bound at the pick, [10] bound after the target ran, [11] stale cancels
        // Helper posted / done / too late, post -> adds done (us), word-end drains and their wait (us)
        int64_t  ne0[6] = {0, 0, 0, 0, 0, 0};
        // Reader turns / taken while behind, entries skipped while behind, guess-inflight waits and their wait (us)
        int64_t  b20[5] = {0, 0, 0, 0, 0};
        // Ranks above k - entries added, bound, used by their target, of them late; + bound at the pick and the
        // seven removal reasons, so a call's line adds up (added = bound + bound at the pick + removed: the list is empty at its end)
        int64_t  gx0[12] = {0};
        // lookahead, summed over the distances: names / new / votes / skipped by pace / in the pool; bound / at the pick /
        // used / late; dropped became / bound / named again / removed; mid-way back / promoted / cancelled / named again
        int64_t  la0[40] = {0};
        ~call_timer() {
            if (!on) {
                return;
            }
            // The engine's stall and its remap op for this call alone, so that a call with
            // more than one token can be split like the per-token lines
            int64_t stall = 0, op = 0;
            int64_t nc[4] = {0, 0, 0, 0}, tc[4] = {0, 0, 0, 0}, rb8 = 0, rb64 = 0;
            int64_t pl[12] = {0}, goal[2] = {0, 0};
            int64_t ne[6] = {0, 0, 0, 0, 0, 0};
            int64_t b2[5] = {0, 0, 0, 0, 0};
            int64_t gx[12] = {0};
            int64_t la[40] = {0};                                                   // lookahead; protect-named: + pn; word-start: + ws; needed-early: + ig; proj-ready: + pr
            llama_moe_stream_stats ms;
            if (ctx && llama_moe_stream_get_stats(ctx, &ms)) {
                stall = ms.t_stall_us - stall0;
                op    = ms.t_op_us   - op0;
                for (int i = 0; i < 4; i++) {
                    nc[i] = ms.dec.n_cause[i]    - cause0[i];
                    tc[i] = ms.dec.t_cause_us[i] - tcause0[i];
                }
                rb8  = ms.dec.n_rb8  - rb80;
                rb64 = ms.dec.n_rb64 - rb640;
                pl[0] = ms.dec.n_pl_add  - pl0[0];
                pl[1] = ms.dec.n_pl_bind - pl0[1];
                for (int i = 0; i < MOE_STREAM_PL_NREASON; i++) {
                    pl[2 + i] = ms.dec.n_pl_rm[i] - pl0[2 + i];
                }
                pl[9]  = ms.dec.n_pl_bind_pick    - pl0[9];
                pl[10] = ms.dec.n_pl_bind_after   - pl0[10];
                pl[11] = ms.dec.n_pl_stale_cancel - pl0[11];
                ne[0] = ms.dec.n_ne_post - ne0[0];
                ne[1] = ms.dec.n_ne_done - ne0[1];
                ne[2] = ms.dec.n_ne_late - ne0[2];
                ne[3] = ms.dec.t_ne_delay_us    - ne0[3];
                ne[4] = ms.dec.n_ne_drain_calls - ne0[4];
                ne[5] = ms.dec.t_ne_drain_us    - ne0[5];
                b2[0] = ms.dec.n_turns      - b20[0];
                b2[1] = ms.dec.n_bh_turns   - b20[1];
                b2[2] = ms.dec.n_bh_skip    - b20[2];
                b2[3] = ms.dec.n_gi_wait    - b20[3];
                b2[4] = ms.dec.t_gi_wait_us - b20[4];
                gx[0] = ms.dec.n_gx_add       - gx0[0];
                gx[1] = ms.dec.n_gx_bind      - gx0[1];
                gx[2] = ms.dec.n_gx_used      - gx0[2];
                gx[3] = ms.dec.n_gx_used_late - gx0[3];
                gx[4] = ms.dec.n_gx_bind_pick - gx0[4];
                for (int i = 0; i < 7; i++) {
                    gx[5 + i] = ms.dec.n_gx_rm[i] - gx0[5 + i];
                }
                int64_t la_now[40];
                llama_moe_stream_la_now(ms, la_now);
                for (int i = 0; i < 40; i++) {
                    la[i] = la_now[i] - la0[i];
                }
                goal[0] = ms.dec.n_goal_ready - goal0[0];
                goal[1] = ms.dec.n_goal_cand  - goal0[1];
            }
            // The stall is this call's own (llama_moe_stream::end_call() ran before this), then the call's
            // not-READY experts by cause (late guess / no guess for the layer / named, not read / not named), the lateness booked to
            // each, and the read-backs of the needed ones
            fprintf(stderr,
                    "moe profile: %d tokens in %.3f ms (host %.3f, device %.3f, copies %.3f, syncs %.3f, "
                    "remap op %.3f, stall %.3f), late/no-guess/nnr/nn %lld/%lld/%lld/%lld "
                    "(%.3f/%.3f/%.3f/%.3f ms), rb8 %lld, rb64 %lld, list +%lld bound %lld rm %lld/%lld/%lld/%lld/%lld/%lld/%lld, "
                    "pick %lld after %lld stalecx %lld, goal %lld/%lld, helper %lld/%lld/%lld (post -> adds done %.3f ms mean, "
                    "drain %lld %.3f ms), turns %lld behind %lld skipped %lld, gi-wait %lld %.3f ms, extra +%lld bound %lld used %lld/%lld pick %lld rm %lld/%lld/%lld/%lld/%lld/%lld/%lld, "
                    "la %lld/%lld/%lld/%lld/%lld bind %lld/%lld/%lld/%lld dr %lld/%lld/%lld/%lld mw %lld/%lld/%lld/%lld "
                    "pn %lld/%lld/%lld/%lld/%lld ws %lld/%lld/%lld/%lld pick %lld/%lld/%lld/%lld ig %lld/%.3f/%lld wsh %lld/%lld/%lld pr %lld/%lld/%.3f/%lld\n",
                    (int) n_tok, (ggml_time_us() - t0)/1000.0,
                    (ggml_prof_get(GGML_PROF_CPU_COMPUTE) - p0[GGML_PROF_CPU_COMPUTE])/1000.0,
                    (ggml_prof_get(GGML_PROF_GPU_COMPUTE) - p0[GGML_PROF_GPU_COMPUTE])/1000.0,
                    (ggml_prof_get(GGML_PROF_COPY)         - p0[GGML_PROF_COPY])        /1000.0,
                    (ggml_prof_get(GGML_PROF_SYNC)         - p0[GGML_PROF_SYNC])        /1000.0,
                    op/1000.0, stall/1000.0,
                    (long long) nc[0], (long long) nc[1], (long long) nc[2], (long long) nc[3],
                    tc[0]/1000.0, tc[1]/1000.0, tc[2]/1000.0, tc[3]/1000.0,
                    (long long) rb8, (long long) rb64,
                    (long long) pl[0], (long long) pl[1], (long long) pl[2], (long long) pl[3], (long long) pl[4],
                    (long long) pl[5], (long long) pl[6], (long long) pl[7], (long long) pl[8],
                    (long long) pl[9], (long long) pl[10], (long long) pl[11],
                    (long long) goal[0], (long long) goal[1], (long long) ne[0], (long long) ne[1], (long long) ne[2],
                    ne[1] > 0 ? ne[3]/1000.0/(double) ne[1] : 0.0, (long long) ne[4], ne[5]/1000.0,
                    (long long) b2[0], (long long) b2[1], (long long) b2[2], (long long) b2[3], b2[4]/1000.0,
                    (long long) gx[0], (long long) gx[1], (long long) gx[2], (long long) gx[3], (long long) gx[4],
                    (long long) gx[5], (long long) gx[6], (long long) gx[7], (long long) gx[8], (long long) gx[9], (long long) gx[10],
                    (long long) gx[11],
                    (long long) la[0], (long long) la[1], (long long) la[2], (long long) la[3], (long long) la[4], (long long) la[5],
                    (long long) la[6], (long long) la[7], (long long) la[8], (long long) la[9], (long long) la[10], (long long) la[11],
                    (long long) la[12], (long long) la[13], (long long) la[14], (long long) la[15], (long long) la[16],
                    (long long) la[17], (long long) la[18], (long long) la[19], (long long) la[20], (long long) la[21],
                    (long long) la[22], (long long) la[23], (long long) la[24], (long long) la[25],
                    (long long) la[26], (long long) la[27], (long long) la[28], (long long) la[29],
                    (long long) la[30], la[31]/1000.0, (long long) la[32],
                    (long long) la[33], (long long) la[34], (long long) la[35],
                    (long long) la[36], (long long) la[37], la[38]/1000.0, (long long) la[39]);
        }
    } prof_timer = { moe_stream_profile_tokens_on(), ggml_time_us(), 0,
                     { ggml_prof_get(0), ggml_prof_get(1), ggml_prof_get(2), ggml_prof_get(3) },
                     0, 0, this };
    prof_timer.n_tok = (int64_t) batch_inp.tokens.size();

    // --moe-stream-timing 1: this call's wall time and what the profiler's timers booked during it (host backend compute, device
    // kernels, cross-backend copies, backend syncs), added to the engine's totals of the call's kind (prompt or decode) when the call ends on any
    // path. Nothing runs unless the setting is on.
    struct timing_call {
        llama_moe_stream * ms = nullptr;
        int     kind = -1;    // 0 prompt, 1 decode: set where the call is classified; a call that never gets there is not booked
        int64_t t0 = 0;
        int64_t p0[GGML_PROF_N_SLOTS] = {};
        ~timing_call() {
            if (ms && kind >= 0) {
                ms->timing_add(kind, ggml_time_us() - t0,
                        ggml_prof_get(GGML_PROF_CPU_COMPUTE) - p0[GGML_PROF_CPU_COMPUTE],
                        ggml_prof_get(GGML_PROF_GPU_COMPUTE) - p0[GGML_PROF_GPU_COMPUTE],
                        ggml_prof_get(GGML_PROF_COPY)        - p0[GGML_PROF_COPY],
                        ggml_prof_get(GGML_PROF_SYNC)        - p0[GGML_PROF_SYNC]);
            }
        }
    } tm_call;
    if (auto * mstream_tm = ctx_moe_stream(); mstream_tm && mstream_tm->p1.timing) {
        tm_call.ms = mstream_tm;
        tm_call.t0 = ggml_time_us();
        for (int i = 0; i < GGML_PROF_N_SLOTS; i++) {
            tm_call.p0[i] = ggml_prof_get(i);
        }
    }

    // A one-token call is a word, any other call is prompt work; the end is booked on every return path
    struct handoff_call {
        int64_t t0 = 0;
        ~handoff_call() {
            if (t0) {
                ggml_handoff_call_end(ggml_time_us() - t0);
            }
        }
    } ho_call;
    if (g_llama_handoff) {
        ggml_handoff_call_begin(batch_inp.tokens.size() == 1 ? GGML_HANDOFF_DECODE : GGML_HANDOFF_PROMPT,
                                (int) batch_inp.tokens.size());
        ho_call.t0 = ggml_time_us();
    }
    if (prof_timer.on) {   // the counters are only read when the line is printed
        llama_moe_stream_stats ms;
        if (llama_moe_stream_get_stats(this, &ms)) {
            prof_timer.stall0 = ms.t_stall_us;
            prof_timer.op0    = ms.t_op_us;
            for (int i = 0; i < 4; i++) {
                prof_timer.cause0[i]  = ms.dec.n_cause[i];
                prof_timer.tcause0[i] = ms.dec.t_cause_us[i];
            }
            prof_timer.rb80  = ms.dec.n_rb8;
            prof_timer.rb640 = ms.dec.n_rb64;
            prof_timer.pl0[0] = ms.dec.n_pl_add;
            prof_timer.pl0[1] = ms.dec.n_pl_bind;
            for (int i = 0; i < MOE_STREAM_PL_NREASON; i++) {
                prof_timer.pl0[2 + i] = ms.dec.n_pl_rm[i];
            }
            prof_timer.pl0[9]  = ms.dec.n_pl_bind_pick;
            prof_timer.pl0[10] = ms.dec.n_pl_bind_after;
            prof_timer.pl0[11] = ms.dec.n_pl_stale_cancel;
            prof_timer.ne0[0] = ms.dec.n_ne_post;
            prof_timer.ne0[1] = ms.dec.n_ne_done;
            prof_timer.ne0[2] = ms.dec.n_ne_late;
            prof_timer.ne0[3] = ms.dec.t_ne_delay_us;
            prof_timer.ne0[4] = ms.dec.n_ne_drain_calls;
            prof_timer.ne0[5] = ms.dec.t_ne_drain_us;
            prof_timer.b20[0] = ms.dec.n_turns;
            prof_timer.b20[1] = ms.dec.n_bh_turns;
            prof_timer.b20[2] = ms.dec.n_bh_skip;
            prof_timer.b20[3] = ms.dec.n_gi_wait;
            prof_timer.b20[4] = ms.dec.t_gi_wait_us;
            prof_timer.gx0[0] = ms.dec.n_gx_add;
            prof_timer.gx0[1] = ms.dec.n_gx_bind;
            prof_timer.gx0[2] = ms.dec.n_gx_used;
            prof_timer.gx0[3] = ms.dec.n_gx_used_late;
            prof_timer.gx0[4] = ms.dec.n_gx_bind_pick;
            for (int i = 0; i < 7; i++) {
                prof_timer.gx0[5 + i] = ms.dec.n_gx_rm[i];
            }
            int64_t la_now[40];
            llama_moe_stream_la_now(ms, la_now);
            for (int i = 0; i < 40; i++) {
                prof_timer.la0[i] = la_now[i];
            }
            prof_timer.goal0[0] = ms.dec.n_goal_ready;
            prof_timer.goal0[1] = ms.dec.n_goal_cand;
        }
    }

    // At the end of the call - after the graph has run on the host, before the profile line above reads the
    // counters (a destructor declared later runs first) and before the caller can - the engine books the overlap waits and
    // the word's lateness to this call. Every return path of decode() passes here.
    struct moe_call_end {
        llama_moe_stream * ms;
        bool               complete;   // every ubatch of the call ran (set after the ubatch loop)
        ~moe_call_end() {
            if (ms) {
                ms->end_call(complete);
            }
        }
    } moe_end = { ctx_moe_stream(), false };
    if (ctx_moe_stream()) {
        ctx_moe_stream()->idle_call_start();   // before anything of the call runs, the queued idle reads are taken back
        ctx_moe_stream()->el_enter();   // before the graph is reserved, the pressure thread waits for the call's end
    }
    if (tt_call.on) { moe_tt("D idle_el"); }

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    // The cuts armed by llama_set_call_cuts are for this call only, for its one sequence
    cc_active = false;
    balloc->cut_pos.clear();   // (a call that returned before its split made its ubatches leaves none behind)
    if (!cc_cuts.empty()) {
        bool one = !output_all && cc_seq >= 0 && balloc->seq_pos_min(cc_seq) >= 0;
        for (int s = 0; s < LLAMA_MAX_SEQ && one; ++s) {
            one = s == cc_seq || balloc->seq_pos_min(s) < 0;
        }
        if (one) {
            balloc->cut_pos = cc_cuts;
            cc_active = !cc_keep.empty();
            cc_layer_done.assign(cc_keep.size(), std::vector<bool>(cc_recr()->r_l.size(), false));
            cc_done.clear();
        } else {
            LLAMA_LOG_WARN("%s: layer-major: the armed cuts need a call of one sequence (seq %d) without all outputs - not taken\n",
                    __func__, cc_seq);
            cc_keep.clear();
            cc_done.clear();
        }
        cc_cuts.clear();
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();
    embd_batch_idxs.clear();

    if (tt_call.on) { moe_tt("D reserve_b", sched_need_reserve); }
    sched_reserve();
    if (tt_call.on) { moe_tt("D reserve_e"); }

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }
    balloc->cut_pos.clear();   // the ubatches are made
    if (tt_call.on) { moe_tt("D mctx"); }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // Tell the MoE expert streaming engine which phase the tokens of this batch belong to, so
    // that its stall, reads and misses are per phase. A decode step is a single token that continues
    // a sequence (pos > 0); a batch of several tokens, or the first token of a sequence, is prompt
    // processing - even when the prompt is split into ubatches of one token each.
    if (auto * mstream = ctx_moe_stream()) {
        const auto & b = balloc->get_batch();

        // A small batch that continues exactly where the last decode batch ended is decoding as
        // well - that is what a speculative draft of a few tokens is - so the engine's counters follow
        // the tokens, not the shape of the call. With several sequences in flight the rule can put a
        // batch in the wrong of the two buckets; nothing else depends on it.
        // A draft check asks for an output at every token; a prompt's last piece (llama-server sends the last few tokens of a
        // prompt as their own call: 5,476 + 2,048 + 4) asks for one - so such a piece is a prompt call and its end is marked
        // (before, set_prompt_end never fired in llama-server: no prompt-end wave order, weights or end fill there)
        const bool cont = b.pos[0] > 0;
        const bool dec  = cont && (n_tokens_all == 1 ||
                        (n_tokens_all <= 8 && b.pos[0] >= mstream->decode_next_pos && n_outputs_all == n_tokens_all));
        tm_call.kind = dec ? 1 : 0;   // (the call's timing goes where the engine's counters of prompt and decode go)

        mstream->vf_word_ctx = cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP;   // VRAM follow's reply words
        if (mstream->vp) {   // the id-table rows of this call start go on this context's stream of the VRAM pool's GPU
            mstream->vp_backend = ms_pool_backend();
        }
        if (mstream->vf_ev_on) {   // VRAM follow's tick records its event on this context's stream of VRAM follow's own GPU
            mstream->vf_ctx_be = nullptr;   // (none: the tick keeps the device-wide wait)
            for (auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == mstream->vf_dev) {
                    mstream->vf_ctx_be = backend.get();
                    break;
                }
            }
        }
        if (mstream->tailw_tensor(0) != nullptr) {   // the gather copies go on this context's stream of the tail's GPU
            mstream->tailw_be = nullptr;
            for (auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == mstream->vx_dev) {
                    mstream->tailw_be = backend.get();
                    break;
                }
            }
        }
        if (mstream->lend_owner == this && mstream->lend_on) {   // before set_decode (VRAM follow's refill of the places at a reply word)
            mstream->lend_call_begin();   // (no front places yet - they come with the call's tensors and its lend)
            lend_kv = (int64_t) b.pos[0] + (int64_t) n_tokens_all;
            lend_again = false;
            if (dec) {
                lend_clear();
            }
        }
        mstream->rg_need = dec ? 0 : rg_call_need(n_tokens_all, *mctx);   // (before the call's start)
        mstream->call_n_out = (int32_t) n_outputs_all;   // (the word-end check of the waiting list)
        mstream->set_decode(dec, (int32_t) n_tokens_all, this);

        // The expert pools, their pins and the counters belong to the model, so a second context
        // on the same model shares them. That is fine as long as their graphs do not overlap (the
        // server runs every slot on one thread and an MTP draft decodes after its target), and this
        // says so once when it happens, because nothing in the engine can detect an overlap.
        if (mstream->n_ctx_shared > 0 && !mstream->n_ctx_warned) {
            mstream->n_ctx_warned = true;
            LLAMA_LOG_WARN("%s: %d contexts share this streaming model's expert pools; their graphs must "
                    "not run at the same time, which is what llama-server and MTP do\n",
                    __func__, mstream->n_ctx_shared);
        }
        if (dec) {
            mstream->decode_next_pos = b.pos[0] + (int32_t) n_tokens_all;
        } else {
            // A prompt batch starts a new sequence, so the position a decode step continues from
            // is the one of that sequence. Without this the value survives a memory clear and a
            // server's next conversation is judged by a position that is not its own.
            mstream->decode_next_pos = 0;
        }
    }

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }
    if (tt_call.on) { moe_tt("D ubatches_b"); }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    // Layer-major (--moe-stream-layer-major 1): a call of several ubatches runs layer group by layer group - every ubatch
    // through layers [0, G), then every ubatch through [G, 2G), ... (G = 1, or MOE_STREAM_TEST_LM_GROUP=G), each ubatch's hidden
    // state kept in host memory between groups; the loop below then runs the last group with the usual output handling. The memory is
    // applied once per ubatch (the first group) and replayed for the later groups (reapply). Whole layers keep each ubatch's own expert
    // stage, so the text is the per-ubatch path's. Anything it does not support runs the per-ubatch path (lm_layer_major_ok).
    struct lm_reset_t {   // back to whole graphs on every way out of decode(), and the call's tensors freed (up to about 60 KiB per token)
        llama_context & c;
        ~lm_reset_t() {
            if (auto * ms = c.ctx_moe_stream(); ms && ms->lm_in_call) {
                ms->lm_call_end(c.lm_complete);   // on every way out of a layer-major call
            }
            c.lm_complete = false;
            c.lm_il0 = 0; c.lm_il1 = -1; c.lm_reapply = false; c.lm_on = false; c.lm_row0 = 0; c.lm_phase = 0;
            c.lm_h.clear(); c.lm_h.shrink_to_fit();
            if (c.lm_hctx) {   // the per-call tensors go once the call's graphs are done with them (in the tail's
                               // extension: before the next call refills the tail)
                ggml_backend_sched_synchronize(c.sched.get());
                c.lm_hid = c.lm_x = c.lm_moe = c.lm_inj = c.lm_ids = c.lm_w = nullptr; c.lm_hbuf.reset(); c.lm_hctx.reset();
            }
        }
    } lm_reset { *this };
    lm_on = lm_layer_major_ok(n_tokens_all);
    // The borrowed room for the call's REAL context - a layer-major call whose tensors plus the prompt lending's budget at the context it
    // reaches do not fit the VRAM pool's held places is processed as layer-major pieces that do (each starts where the last ended, so a
    // sliding window stays whole). Only where the pieces before the last ask for no output (a prompt's), and no token carries an embedding
    // or a state; else the call runs as it is (the pool gives way, never a failure)
    if (auto * ms = ctx_moe_stream(); lm_on && !lm_force_off && ms && ms->lend_owner == this && ms->lend_on && ms->lm_lend_on()) {
        const int64_t kv0 = (int64_t) balloc->get_batch().pos[0];
        const int64_t t_fit = ms->lm_lend_max_tokens(lm_call_bytes_per_tok(model.hparams), kv0)/cparams.n_ubatch*cparams.n_ubatch;
        if (t_fit >= (int64_t) cparams.n_ubatch && t_fit < (int64_t) n_tokens_all) {
            bool ok = true;
            for (int64_t i = 0; i < t_fit && ok; i++) {
                const auto & tk = batch_inp.tokens[(size_t) i];
                ok = !tk.output && !tk.has_embd && !tk.has_state;
            }
            if (ok) {
                static int said = 0;
                if (said++ < 3) {
                    fprintf(stderr, "%s: VRAM lending: a layer-major call of %u tokens from position %lld does not fit the VRAM pool's places at "
                            "the context it reaches - processed as pieces of up to %lld tokens\n", __func__, n_tokens_all, (long long) kv0,
                            (long long) t_fit);
                }
                llama_batch_ext b1 = batch_inp;
                b1.tokens.resize((size_t) t_fit);
                llama_batch_ext b2 = batch_inp;
                b2.tokens.erase(b2.tokens.begin(), b2.tokens.begin() + t_fit);
                lm_on = false;
                n_queued_tokens -= n_tokens_all;   // (the pieces count their own tokens: not twice in the prompt's count)
                const int r1 = decode(b1);
                if (r1 != 0) {
                    return r1;
                }
                return decode(b2);
            }
        }
    }
    const int lm_g = lm_on ? lm_group() : 0;
    if (auto * ms_c = ctx_moe_stream(); ms_c && !ms_c->is_decode() && n_tokens_all > 1) {   // the prompt calls (the exit line)
        n_prompt_calls += 1;
        if (prompt_calls.size() < 64) {   // (the first 64 only: a long-running server's list stays bounded; the count goes on)
            prompt_calls.push_back({ (int64_t) n_tokens_all, lm_on });
        }
    }
    // A layer-major call that fails after its positions were removed exactly (lm_remove_call() == true) runs
    // again ubatch by ubatch - the same batch, layer-major off for it - with one line, instead of failing the call. A user abort, or a
    // memory that cleared the whole sequence (Ling's hybrid one), keeps the failure result. MOE_STREAM_TEST_PROMPT_SPLIT_LM_FAIL=1 (test only): the
    // first layer-major call's rewind "fails" once
    // (only where the retry gets the same context - a call that started at position 0, or a memory without a sliding
    // window. A failed call that started later has already written its first ubatches over the window cells from before the call, and
    // seq_rm cannot bring them back: the retry would run with up to n_swa tokens missing their window - wrong text, silently. There
    // the error goes back as before, and the caller processes the prompt again)
    const auto lm_retry = [&](bool cut, ggml_status st, const char * what, int & r) -> bool {
        if (lend_again) {   // (a full card - the caller processes the prompt again, the lend doubled)
            lend_again    = false;
            lm_reprocess = true;
            return false;
        }
        if (!cut || st == GGML_STATUS_ABORTED || lm_force_off) {
            return false;
        }
        if (model.hparams.is_swa_any()) {
            bool at0 = true;
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                at0 = at0 && balloc->seq_pos_min(s) <= 0;   // (-1: the sequence is not in the call)
            }
            if (!at0) {
                fprintf(stderr, "%s: layer-major: %s - the call did not start at position 0 and the memory has a sliding window (its cells from "
                        "before the call are overwritten): decode returns 3 - the caller processes the prompt again from its start\n", __func__, what);
                lm_reprocess = true;
                return false;
            }
        }
        fprintf(stderr, "%s: layer-major: %s - this call runs ubatch by ubatch instead (no layer-major for it; slower)\n", __func__, what);
        if (auto * ms = ctx_moe_stream(); ms && ms->lm_in_call) {
            ms->lm_call_end(false);
        }
        lm_complete = false; lm_il0 = 0; lm_il1 = -1; lm_reapply = false; lm_on = false; lm_row0 = 0; lm_phase = 0;
        lm_h.clear();
        if (lm_hctx) {
            ggml_backend_sched_synchronize(sched.get());
            lm_hid = lm_x = lm_moe = lm_inj = lm_ids = lm_w = nullptr; lm_hbuf.reset(); lm_hctx.reset();
        }
        lm_force_off = true;
        r = decode(batch_inp);
        lm_force_off = false;
        return true;
    };
    // (MOE_STREAM_TEST_PROMPT_SPLIT_LM_FAIL=1: the first layer-major call; =2: the first one that does not start at position 0)
    static int psplit_test = getenv("MOE_STREAM_TEST_PROMPT_SPLIT_LM_FAIL") ? atoi(getenv("MOE_STREAM_TEST_PROMPT_SPLIT_LM_FAIL")) : 0;
    const bool psplit_here = psplit_test == 1 || (psplit_test == 2 && balloc->get_batch().pos[0] > 0);
    // One expert graph per streamed layer (MOE_STREAM_TEST_LM_GROUP=G keeps whole layers). Only where the expert graph's waves run where its
    // tensors are: on a CPU-only run, or with GPU waves for an n_ubatch chunk (CPU waves in a GPU run would pull the
    // whole call's expert input into the CPU compute buffer, uncounted) - otherwise whole layers
    bool lm_split = lm_on && !lm_group_test();
    if (lm_split) {
        const auto * ms = ctx_moe_stream();
        const bool cpu_only = ggml_backend_dev_type(model.dev_layer(0)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        const bool gpu_waves_chunk = ms && ms->gpu_waves > 0 && cparams.n_ubatch >= ms->gpu_waves && !ms->cpu_repack_on.load() &&
                                     cparams.op_offload;
        lm_split = ms != nullptr && (cpu_only || gpu_waves_chunk);
        // The expert graph holds the whole call in chunks of n_ubatch rows, per wave; it may hold only what the graph
        // result's room takes (nodes and object memory, measured at start: lm_measure_expert_graph) - a longer call runs layer-major
        // in whole-layer groups (step 2a) instead, with one line (before: -ub 1 on CPU-only aborted in ggml_new_object)
        if (lm_split && !lm_xg_fits(n_tokens_all)) {
            lm_split = false;
            static int said = 0;
            if (said++ < 3) {
                fprintf(stderr, "%s: layer-major: a call of %u tokens is %lld chunks of -ub %u - more than one expert graph holds (its room, "
                        "at %lld waves) - it runs layer by layer in whole-layer groups instead\n", __func__, n_tokens_all,
                        (long long) ((n_tokens_all + cparams.n_ubatch - 1)/cparams.n_ubatch), cparams.n_ubatch,
                        (long long) std::max<int64_t>(lm_xg.w, ms->n_waves(n_tokens_all)));
            }
        }
    }
    int lm_last0 = 0;   // the first layer of the last group (0 = one group = the plain order)
    if (lm_on) {
        lm_last0 = (((int) hparams.n_layer() - 1) / lm_g) * lm_g;
        // With one expert graph per streamed layer, the last layer too runs lm_phase 1 per ubatch + one expert graph (its
        // experts read once per call, not once per ubatch: Flash 8k, layer 41 read 926 experts against 388-462 in the other layers);
        // the loop below then builds only the output per ubatch. MOE_STREAM_TEST_LM_LAST_GROUP=1 (test only) keeps the last group per ubatch
        static const bool test_last_group = getenv("MOE_STREAM_TEST_LM_LAST_GROUP") && atoi(getenv("MOE_STREAM_TEST_LM_LAST_GROUP")) != 0;
        if (lm_split && !test_last_group) {
            lm_last0 = (int) hparams.n_layer();
        }
        lm_h.clear();
        // The call's hidden state [n_embd * hc, T] (hc = the hyper-connection streams, 1 without them) on the device of the layers
        // (the CPU on a CPU-only run): the group graphs read and write their ubatch's rows there, so no graph waits for the one before
        // it and nothing goes to the host. With the expert graphs split off (lm_split) it adds the normed expert input [n_embd, T] and
        // the router's picks and weights [k, T], and with hyper-connection streams also the routed-expert sum lm_moe [n_embd, T] and
        // the FFN combine's inject weights lm_inj [hc, T]; with a GPU expert pool that has a tail they all live in the tail's
        // extension (vacated for this call's GPU-wave graphs first), else in a buffer of their own
        bool lm_output_lent = false;
        auto restore_lent_output = [&]() {
            if (lm_output_lent && !lm_output_backup.empty() && model.output) {
                synchronize();
                ggml_backend_tensor_set(model.output, lm_output_backup.data(), 0, lm_output_backup.size());
                lm_output_lent = false;
            }
        };
        struct lent_output_raii_t {
            decltype(restore_lent_output) & fn;
            ~lent_output_raii_t() { fn(); }
        } lent_output_guard{ restore_lent_output };
        if (lm_last0 > 0) {
            auto * ms = ctx_moe_stream();
            if (ms) { ms->clear_lm_routed_masks(); }
            const int64_t k  = hparams.n_expert_used();
            const int64_t hc = std::max<int64_t>(1, hparams.dsv4_hc_mult);
            const ggml_type hid_type = (model.prefill_lm_bf16_explicit() &&
                    ggml_backend_dev_type(model.dev_layer(0)) != GGML_BACKEND_DEVICE_TYPE_CPU)
                ? GGML_TYPE_BF16 : GGML_TYPE_F32;
            const auto build_lm_tensors = [&](ggml_type ht) {
                ggml_init_params ip = { 6*ggml_tensor_overhead(), nullptr, true };
                lm_hctx.reset(ggml_init(ip));
                lm_hid = ggml_new_tensor_2d(lm_hctx.get(), ht, hparams.n_embd*hc, n_tokens_all);
                ggml_set_name(lm_hid, "lm_hidden");
                lm_x = lm_moe = lm_inj = lm_ids = lm_w = nullptr;
                if (lm_split) {
                    lm_x   = ggml_new_tensor_2d(lm_hctx.get(), GGML_TYPE_F32, hparams.n_embd, n_tokens_all);
                    if (hc > 1) {
                        lm_moe = ggml_new_tensor_2d(lm_hctx.get(), GGML_TYPE_F32, hparams.n_embd, n_tokens_all);
                        lm_inj = ggml_new_tensor_2d(lm_hctx.get(), GGML_TYPE_F32, hc, n_tokens_all);
                        ggml_set_name(lm_moe, "lm_moe_out");
                        ggml_set_name(lm_inj, "lm_hc_inject");
                    }
                    lm_ids = ggml_new_tensor_2d(lm_hctx.get(), GGML_TYPE_I32, k, n_tokens_all);
                    lm_w   = ggml_new_tensor_2d(lm_hctx.get(), GGML_TYPE_F32, k, n_tokens_all);
                    ggml_set_name(lm_x, "lm_expert_in"); ggml_set_name(lm_ids, "lm_ids"); ggml_set_name(lm_w, "lm_weights");
                }
            };
            build_lm_tensors(hid_type);
            const auto lm_all_tensors = [&]() {
                return std::array<ggml_tensor *, 6>{ lm_hid, lm_x, lm_moe, lm_inj, lm_ids, lm_w };
            };
            bool placed = false;
            if (ms && ms->tail_on) {
                const size_t al = 256;
                size_t total = 0;
                for (ggml_tensor * t : lm_all_tensors()) {
                    if (t) { total += GGML_PAD(ggml_nbytes(t), al); }
                }
                // vacates the tail's copy places when this call's ubatches are GPU-wave graphs, and the extension places these tensors
                // take (only those, by the same byte count lm_tail_region checks)
                ms->tail_ubatch((int64_t) cparams.n_ubatch, (int64_t) total);
                ggml_backend_buffer_t tbuf = nullptr;
                char * taddr = nullptr;
                if (ms->lm_tail_region((int64_t) total, &tbuf, &taddr)) {
                    size_t off = 0;
                    for (ggml_tensor * t : lm_all_tensors()) {
                        if (t) {
                            GGML_ASSERT(ggml_backend_tensor_alloc(tbuf, t, taddr + off) == GGML_STATUS_SUCCESS);
                            off += GGML_PAD(ggml_nbytes(t), al);
                        }
                    }
                    placed = true;
                }
            }
            if (!placed && ms && ms->lm_lend_on() && ggml_backend_dev_type(model.dev_layer(0)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                // Expert places of the VRAM pool lent for this call, each tensor in one pool group (with or without a tail)
                std::vector<ggml_tensor *> tv;
                std::vector<int64_t> sz;
                for (ggml_tensor * t : lm_all_tensors()) {
                    if (t) {
                        tv.push_back(t);
                        sz.push_back((int64_t) ggml_nbytes(t));
                    }
                }
                std::vector<ggml_backend_buffer_t> bb(tv.size(), nullptr);
                std::vector<char *> aa(tv.size(), nullptr);
                ms->lend_call_kv = (int64_t) balloc->get_batch().pos[0] + (int64_t) n_tokens_all;   // (the context this call reaches)
                if (ms->lm_lend_place(sz, bb.data(), aa.data())) {
                    for (size_t i = 0; i < tv.size(); i++) {
                        GGML_ASSERT(ggml_backend_tensor_alloc(bb[i], tv[i], aa[i]) == GGML_STATUS_SUCCESS);
                    }
                    placed = true;
                }
            }
            ggml_backend_dev_t dev = model.dev_layer(0);
            const auto try_lend_output = [&]() -> bool {
                if (!lm_split || !model.prefill_lend_output_on() ||
                        !model.output || model.output == model.tok_embd ||
                        !model.output->buffer || ggml_backend_buffer_is_host(model.output->buffer) ||
                        !ggml_is_contiguous(model.output)) {
                    return false;
                }
                const size_t al = 256;
                size_t aux_bytes = 0;
                for (ggml_tensor * t : { lm_x, lm_moe, lm_inj, lm_ids, lm_w }) {
                    if (t) { aux_bytes += GGML_PAD(ggml_nbytes(t), al); }
                }
                const size_t out_bytes = ggml_nbytes(model.output);
                const size_t hid_bytes = GGML_PAD(ggml_nbytes(lm_hid), al);
                if (aux_bytes == 0 || aux_bytes > out_bytes) {
                    return false;
                }
                ggml_backend_buffer_ptr hbuf_only(ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), hid_bytes));
                if (!hbuf_only) {
                    return false;
                }
                char * hbase = (char *) ggml_backend_buffer_get_base(hbuf_only.get());
                GGML_ASSERT(ggml_backend_tensor_alloc(hbuf_only.get(), lm_hid, hbase) == GGML_STATUS_SUCCESS);
                synchronize();
                if (lm_output_backup.size() != out_bytes) {
                    lm_output_backup.resize(out_bytes);
                    ggml_backend_tensor_get(model.output, lm_output_backup.data(), 0, out_bytes);
                }
                lm_output_lent = true;
                lm_hbuf = std::move(hbuf_only);
                size_t off = 0;
                for (ggml_tensor * t : { lm_x, lm_moe, lm_inj, lm_ids, lm_w }) {
                    if (t) {
                        t->buffer = model.output->buffer;
                        t->data   = (char *) model.output->data + off;
                        off += GGML_PAD(ggml_nbytes(t), al);
                    }
                }
                return true;
            };
            if (!placed && model.prefill_lend_output_explicit()) {
                placed = try_lend_output();
            }
            if (!placed) {
                lm_hbuf.reset(ggml_backend_alloc_ctx_tensors_from_buft(lm_hctx.get(), ggml_backend_dev_buffer_type(dev)));
                if (lm_hbuf) {
                    placed = true;
                }
            }
            // Automatic OOM rescue under prefill-fast: shrink lm_hid to BF16 (if not already) and/or lend model.output
            if (!placed && model.prefill_lm_bf16_on() && lm_hid->type == GGML_TYPE_F32 &&
                    ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                build_lm_tensors(GGML_TYPE_BF16);
                lm_hbuf.reset(ggml_backend_alloc_ctx_tensors_from_buft(lm_hctx.get(), ggml_backend_dev_buffer_type(dev)));
                if (lm_hbuf) {
                    placed = true;
                }
            }
            if (!placed) {
                placed = try_lend_output();
            }
            if (!placed) {
                if (true) {
                    LLAMA_LOG_ERROR("%s: layer-major: could not allocate the call's tensors (%.1f MiB of hidden state on %s)\n", __func__,
                            ggml_nbytes(lm_hid)/1048576.0, ggml_backend_dev_name(dev));
                    lm_hid = lm_x = lm_moe = lm_inj = lm_ids = lm_w = nullptr; lm_hctx.reset();
                    if (!lm_force_off) {   // nothing has run yet, the memory is as before the call - ubatch by ubatch
                        fprintf(stderr, "%s: layer-major: the call's tensors did not fit on %s - this call runs ubatch by ubatch instead (no "
                                "layer-major for it; slower)\n", __func__, ggml_backend_dev_name(dev));
                        lm_on = false;
                        lm_force_off = true;
                        const int r = decode(batch_inp);
                        lm_force_off = false;
                        return r;
                    }
                    return -2;   // nothing has run yet: the memory is as before the call
                }
            }
            static int said_where = -1;
            if (said_where != (int) placed) {
                said_where = (int) placed;
                size_t call_bytes = 0;
                for (ggml_tensor * t : lm_all_tensors()) {
                    if (t) { call_bytes += ggml_nbytes(t); }
                }
                fprintf(stderr, "%s: layer-major: the call's tensors (%d tokens, %.1f MiB) %s\n", __func__, (int) n_tokens_all,
                        call_bytes/1048576.0,
                        placed ? "in the VRAM pool (its tail / expert places lent for the call)" : (std::string("in a buffer of their own on ") + ggml_backend_dev_name(dev)).c_str());
            }
        }
        if (auto * ms = ctx_moe_stream(); ms && ms->lend_owner == this && ms->lend_on && !ms->is_decode()) {
            lend_call_start(false, (int64_t) n_tokens_all, lend_kv);   // a layer-major prompt call lends after the front places
        }
        if (auto * ms = ctx_moe_stream(); ms && lm_last0 > 0) {   // the window's call start (the call-level prompt-end rule)
            static const bool pe_all = getenv("MOE_STREAM_PROMPT_END_ALL") && atoi(getenv("MOE_STREAM_PROMPT_END_ALL")) != 0;
            ms->lm_call_begin(n_tokens_all, !ms->is_decode() && n_tokens_all > 1 && n_outputs_all > 0 &&
                    (pe_all || n_tokens_all < cparams.n_batch));
        }
        for (int il0 = 0; il0 < lm_last0; il0 += lm_g) {
            if (auto * ms = ctx_moe_stream()) {
                ms->lm_layer_begin(il0);
            }
            const bool psplit_now = psplit_here && il0 > 0;   // (test only, once)
            if (il0 > 0 && (psplit_now || !mctx->rewind())) {   // (the memories of the models llama_arch_layer_major admits can)
                psplit_test = 0;
                LLAMA_LOG_ERROR("%s: layer-major: the memory cannot go back to the first ubatch\n", __func__);
                const bool cut = lm_remove_call();
                int r = 0;
                if (lm_retry(cut, GGML_STATUS_FAILED, psplit_now ? "TEST ONLY (MOE_STREAM_TEST_PROMPT_SPLIT_LM_FAIL): the memory 'cannot go back'"
                                                             : "the memory cannot go back to the first ubatch", r)) {
                    return r;
                }
                if (lm_reprocess) { lm_reprocess = false; return 3; }   // (the caller processes the prompt again)
                return -3;
            }
            lm_il0     = il0;
            lm_il1     = il0 + lm_g;
            lm_reapply = il0 > 0;
            // A streamed layer runs lm_phase 1 per ubatch, then one expert graph for the whole call (below)
            lm_phase   = lm_split && ctx_moe_stream() && ctx_moe_stream()->layer(il0) != nullptr ? 1 : 0;
            llama_ubatch lm_ub_first;
            size_t u = 0;
            int64_t lm_tok_prev = 0;
            do {
                const auto & ub = mctx->get_ubatch();
                llama_ubatch ubx = ub;   // the input after layer 0 comes from lm_hid (graph_params), not the ubatch
                lm_row0 = lm_tok_prev;
                if (u == 0) {
                    lm_ub_first = ub;
                }
                n_outputs = ub.n_tokens;   // (no output rows in these graphs; process_ubatch only sizes the graph by it)
                if (auto * mstream = ctx_moe_stream()) {
                    // the same prompt-end rule as the loop below, for every layer group: the last ubatch of a short prompt
                    // batch with an output orders its waves so the reply's experts stay (otherwise the groups before the last see
                    // whatever the previous call left)
                    static const bool pe_all = getenv("MOE_STREAM_PROMPT_END_ALL") && atoi(getenv("MOE_STREAM_PROMPT_END_ALL")) != 0;
                    int32_t n_out_ub = 0;
                    for (uint32_t i = 0; i < ub.n_tokens; ++i) {
                        n_out_ub += ub.output[i] != 0;
                    }
                    mstream->set_prompt_end(!mstream->is_decode() && n_tokens_all > 1 && n_out_ub > 0 &&
                            lm_tok_prev + (int64_t) ub.n_tokens == (int64_t) n_tokens_all &&
                            (pe_all || n_tokens_all < cparams.n_batch));
                }
                lm_tok_prev += ub.n_tokens;
                ggml_status st = GGML_STATUS_SUCCESS;
                const auto * res_g = process_ubatch(ubx, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), st);
                if (!res_g || !res_g->t_lm_hidden) {
                    // a later group's failure leaves earlier ubatches finished only up to that group: the whole call goes
                    LLAMA_LOG_ERROR("%s: layer-major: layer group [%d, %d) of ubatch %zu failed\n", __func__, lm_il0, lm_il1, u);
                    const bool cut = lm_remove_call();
                    if (int r = 0; lm_retry(cut, st, "a layer group failed", r)) {
                        return r;
                    }
                    if (lm_reprocess) { lm_reprocess = false; return 3; }   // (the caller processes the prompt again)
                    switch (st) {
                        case GGML_STATUS_ABORTED:      return cut ? 2 : -3;
                        case GGML_STATUS_ALLOC_FAILED: return -2;
                        default:                       return -3;
                    }
                }
                cc_copy(ub, lm_il0, lm_il1);   // this group's recurrent rows at a kept cut
                u++;   // no synchronize - the next graph runs after this one on the same stream
            } while (mctx->next());

            if (lm_phase == 1) {
                // The layer's expert graph - one "ubatch" of the whole call for the engine (its remap ops see the call's
                // picks [k, T]), no memory, no input; the GPU-wave, tail and VRAM-pass decisions go by n_ubatch (process_ubatch)
                lm_phase = 2;
                lm_row0  = 0;
                llama_ubatch ube = lm_ub_first;
                ube.n_tokens     = n_tokens_all;
                ube.n_seq_tokens = n_tokens_all;
                n_outputs = n_tokens_all;
                if (auto * mstream = ctx_moe_stream()) {   // the call's prompt-end rule (the usual one, for the whole call)
                    static const bool pe_all = getenv("MOE_STREAM_PROMPT_END_ALL") && atoi(getenv("MOE_STREAM_PROMPT_END_ALL")) != 0;
                    mstream->set_prompt_end(!mstream->is_decode() && n_tokens_all > 1 && n_outputs_all > 0 &&
                            (pe_all || n_tokens_all < cparams.n_batch));
                }
                ggml_status st = GGML_STATUS_SUCCESS;
                ctx_moe_stream()->lm_expert_graph = true;    // the tail's copy hook: a wave's chunks share one pool copy
                const auto * res_e = process_ubatch(ube, ctx_type_to_graph_type(cparams.ctx_type), nullptr, st);
                ctx_moe_stream()->lm_expert_graph = false;
                if (!res_e) {
                    LLAMA_LOG_ERROR("%s: layer-major: the expert graph of layer %d failed\n", __func__, il0);
                    const bool cut = lm_remove_call();
                    if (int r = 0; lm_retry(cut, st, "a layer's expert graph failed", r)) {
                        return r;
                    }
                    if (lm_reprocess) { lm_reprocess = false; return 3; }   // (the caller processes the prompt again)
                    switch (st) {
                        case GGML_STATUS_ABORTED:      return cut ? 2 : -3;
                        case GGML_STATUS_ALLOC_FAILED: return -2;
                        default:                       return -3;
                    }
                }
            }
            lm_phase = 0;
            lm_signal_done(il0, il0 + lm_g);   // the group's streamed layers are done with this call
            // MOE_STREAM_TEST_LM_DUMP=1 (test only, read once): the call's hidden state after each layer group, per n_ubatch piece
            static const bool test_dump = getenv("MOE_STREAM_TEST_LM_DUMP") && atoi(getenv("MOE_STREAM_TEST_LM_DUMP")) != 0;
            if (test_dump && lm_hid) {
                synchronize();
                std::vector<float> h(ggml_nelements(lm_hid));
                if (lm_hid->type == GGML_TYPE_BF16) {
                    std::vector<ggml_bf16_t> hb(ggml_nelements(lm_hid));
                    ggml_backend_tensor_get(lm_hid, hb.data(), 0, ggml_nbytes(lm_hid));
                    for (size_t i = 0; i < hb.size(); ++i) { h[i] = ggml_bf16_to_fp32(hb[i]); }
                } else {
                    ggml_backend_tensor_get(lm_hid, h.data(), 0, ggml_nbytes(lm_hid));
                }
                const int64_t ne0 = lm_hid->ne[0], T = lm_hid->ne[1];
                for (int64_t r0 = 0; r0 < T; r0 += cparams.n_ubatch) {
                    const int64_t r1 = std::min<int64_t>(T, r0 + cparams.n_ubatch);
                    int64_t n_nan = 0; double mx = 0, ss = 0;
                    for (int64_t i = r0*ne0; i < r1*ne0; i++) {
                        const float v = h[(size_t) i];
                        if (!std::isfinite(v)) { n_nan++; continue; }
                        mx = std::max(mx, (double) std::fabs(v)); ss += (double) v*v;
                    }
                    fprintf(stderr, "LM_DUMP layer %d rows %lld-%lld: non-finite %lld max %.4g rms %.4g\n", il0, (long long) r0, (long long) r1,
                            (long long) n_nan, mx, std::sqrt(ss/std::max<int64_t>(1, (r1 - r0)*ne0)));
                }
            }
        }
        restore_lent_output();
        if (lm_last0 > 0 && !mctx->rewind()) {
            LLAMA_LOG_ERROR("%s: layer-major: the memory cannot go back to the first ubatch\n", __func__);
            const bool cut = lm_remove_call();
            if (int r = 0; lm_retry(cut, GGML_STATUS_FAILED, "the memory cannot go back to the first ubatch", r)) {
                return r;
            }
            if (lm_reprocess) { lm_reprocess = false; return 3; }   // (the caller processes the prompt again)
            return -3;
        }
        lm_il0     = lm_last0;
        lm_il1     = -1;
        lm_reapply = lm_last0 > 0;
        if (auto * ms = ctx_moe_stream(); ms && lm_last0 < (int) hparams.n_layer()) {
            ms->lm_layer_begin(lm_last0);   // the last group (per ubatch, as without layer-major)
        }
        static bool said = false;
        if (!said) {
            said = true;
            fprintf(stderr, "%s: layer-major prompt (--moe-stream-layer-major 1, %s): "
                    "groups of %d layers (%d layers, %s), ubatch %u\n", __func__,
                    lm_split ? "a streamed layer's experts once per call" : "MOE_STREAM_TEST_LM_GROUP: each ubatch keeps its own expert stage",
                    lm_g, (int) hparams.n_layer(), lm_last0 >= (int) hparams.n_layer() ? "every layer in the layer loop, then the output per ubatch" :
                    format("the last group from layer %d", lm_last0).c_str(), cparams.n_ubatch);
        }
    }
    if (auto * ms = ctx_moe_stream(); !lm_on && ms && ms->lend_owner == this && ms->lend_on && !ms->is_decode()) {
        lend_call_start(false, (int64_t) n_tokens_all, lend_kv);   // a prompt call that is not layer-major lends at its start too
    }
    do {
        const auto & ubatch = mctx->get_ubatch();

        // The last layer group takes the hidden state of the groups before it (step 2a: its rows of lm_hid)
        const llama_ubatch * ub_run = &ubatch;
        if (lm_on && lm_last0 > 0) {
            lm_row0 = n_tokens_prev;
        }

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        // Tell the expert streaming engine whether this ubatch ends a prompt, so that its waves can
        // leave in RAM what the reply is likely to need: the last ubatch of a prompt batch that asks for an output and is
        // shorter than n_batch. Tools that cut a prompt into n_batch pieces ask for the last token of every piece
        // (common_batch_get_one), so only the short last piece is the prompt's end; a prompt that is an exact multiple
        // of n_batch is missed. MOE_STREAM_PROMPT_END_ALL=1 drops the length test (llama-perplexity's pieces are all
        // full, and that is how the numerics of the order are measured).
        if (auto * mstream = ctx_moe_stream()) {
            static const bool pe_all = getenv("MOE_STREAM_PROMPT_END_ALL") && atoi(getenv("MOE_STREAM_PROMPT_END_ALL")) != 0;
            mstream->set_prompt_end(!mstream->is_decode() && n_tokens_all > 1 && n_outputs > 0 &&
                    n_tokens_prev + (int64_t) ubatch.n_tokens == (int64_t) n_tokens_all &&
                    (pe_all || n_tokens_all < cparams.n_batch));
        }

        ggml_status status;

        const auto * res = process_ubatch(*ub_run, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res && lm_on && lm_last0 > 0) {
            // The earlier ubatches have not run the last layer group yet -> the whole call goes (as in a group's failure)
            const bool cut = lm_remove_call();
            if (int r = 0; lm_retry(cut, status, "the last layer group failed", r)) {
                return r;
            }
            if (lm_reprocess) { lm_reprocess = false; return 3; }   // (the caller processes the prompt again)
            switch (status) {
                case GGML_STATUS_ABORTED:      return cut ? 2 : -3;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                default:                       return -3;
            }
        }

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }
            if (lend_again) {   // (a full card - the caller processes the prompt again, the lend doubled)
                lend_again = false;
                return 3;
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        cc_copy(ubatch, lm_on ? lm_last0 : 0, INT_MAX);   // the recurrent rows of this graph's layers at a kept cut

        const int64_t t_ho_output = g_llama_handoff ? ggml_time_us() : 0;   // the readback calls below

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        // A streamed model's output is checked for non-finite values (--moe-stream-out-check N: every prompt call and every Nth reply
        // call; 0 = never), on 256 logits of the last output row: a wrong expert byte becomes NaN in every later row, and the call fails
        // here with one line (its sequences' memory dropped, so nothing reuses it) instead of giving a wrong answer
        if (auto * ms_o = ctx_moe_stream(); ms_o && t_logits && n_outputs > 0 && ms_o->out_check_due(ubatch.n_tokens > 1)) {
            ggml_backend_t be_o = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            const int64_t  nv   = t_logits->ne[0];
            const int64_t  nr   = std::min<int64_t>(256, nv);
            std::vector<float> lv((size_t) nr);
            if (be_o != nullptr) {
                const auto t_o0 = std::chrono::steady_clock::now();
                ggml_backend_tensor_get_async(be_o, t_logits, lv.data(), (size_t) (n_outputs - 1)*t_logits->nb[1], (size_t) nr*sizeof(float));
                ggml_backend_synchronize(be_o);
                int64_t bad = 0;
                for (const float v : lv) {
                    bad += std::isfinite(v) ? 0 : 1;
                }
                const int kind = ubatch.n_tokens > 1 ? 0 : 1;
                ms_o->out_checked[kind] += 1;
                ms_o->out_us[kind]      += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_o0).count();
                if (bad > 0 || ms_o->out_test_nan) {
                    LLAMA_LOG_ERROR("%s: the streamed model's output holds non-finite values (%lld of %lld logits of the call's last row are NaN / Inf%s) - "
                            "the expert computation went wrong; this call fails instead of giving a wrong answer\n", __func__, (long long) bad,
                            (long long) nr, ms_o->out_test_nan ? ", MOE_STREAM_TEST_OUT_NAN" : "");
                    fprintf(stderr, "llama_moe_stream: the output check failed this call (NaN / Inf logits); its sequences' memory is dropped\n");
                    ms_o->out_failed += 1;
                    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                        for (int32_t q = 0; q < ubatch.n_seq_id[i]; q++) {
                            memory->seq_rm(ubatch.seq_id[i][q], -1, -1);
                        }
                    }
                    return -3;
                }
            }
        }

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                extract_all_idxs = extract_all_idxs || !masked;
            }
        }

        if (extract_all_idxs) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev);
            const auto & batch_idxs = ubatch.data->batch_idxs;
            embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
        }

        if (g_llama_handoff) {
            ggml_handoff_ctx_add(GGML_HANDOFF_C_OUTPUT, ggml_time_us() - t_ho_output);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    moe_end.complete = true;   // the graph of every ubatch ran to its end
    if (tt_call.on) { moe_tt("D ubatches_e"); }
    if (lm_on && lm_last0 > 0) {   // the last group's streamed layers are done with this call
        lm_signal_done(lm_last0, (int) hparams.n_layer());
        lm_complete = true;
    }

    if (cc_active) {   // a kept cut is complete when every recurrent layer copied its rows there
        cc_active = false;
        const auto * recr = cc_recr();
        for (size_t k = 0; k < cc_keep.size(); ++k) {
            bool all = true;
            for (size_t il = 0; il < recr->r_l.size() && all; ++il) {
                all = (recr->r_l[il] == nullptr && recr->s_l[il] == nullptr) || cc_layer_done[k][il];
            }
            if (all) {
                cc_done.push_back(cc_keep[k]);
            }
        }
        fprintf(stderr, "%s: layer-major: %zu of %zu checkpoints taken inside this call (%u tokens)\n", __func__, cc_done.size(),
                cc_keep.size(), n_tokens_all);
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();
    if (tt_call.on) { moe_tt("D outputs_e"); }

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        GGML_ASSERT(row_floats == model.hparams.n_embd);
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        // Tensor-split backends require a zero source offset.
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
        extracted = true;
    }
    return extracted;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
    for (size_t i = 0; i < embd_batch_idxs.size(); ++i) {
        while (embd_batch_idxs[i] != (int32_t) i) {
            const int32_t j = embd_batch_idxs[i];
            GGML_ASSERT(j >= 0 && (size_t) j < embd_batch_idxs.size());
            if (embd_nextn.has_data() && !cparams.embeddings_nextn_masked) {
                for (size_t k = 0; k < n_embd_out; ++k) {
                    std::swap(embd_nextn.data[i*n_embd_out + k], embd_nextn.data[j*n_embd_out + k]);
                }
            }
            for (auto & layer : embd_layer_inp) {
                if (layer.has_data()) {
                    for (size_t k = 0; k < n_embd; ++k) {
                        std::swap(layer.data[i*n_embd + k], layer.data[j*n_embd + k]);
                    }
                }
            }
            std::swap(embd_batch_idxs[i], embd_batch_idxs[j]);
        }
    }
    embd_batch_idxs.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4 ||
        // The MoE expert streaming adds nodes to every streamed layer's graph (the remap op, the pool views, the guess, the
        // waves) whatever the architecture: Flint's graph ran out of the 8-per-tensor budget below at its first prompt
        ctx_moe_stream() != nullptr) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    // every sampler builds n_outputs_max_per_seq chains, see llm_graph_context::build_sampling
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        res += llama_sampler_backend_n_nodes(sampler) * cparams.n_outputs_max_per_seq;
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];
    if (!res) {
        res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    return res.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_input_tensors = llama_graph_n_input_tensors(gf);
    this->n_outputs = save_n_outputs;

    // The reserved graph has to be scheduled with the rule the real graph will use. It used
    // to keep whatever op_offload was last set, which for a ubatch with expert waves is true: the
    // scheduler then puts the wave matmuls on the device and sizes the device compute buffer for a copy
    // of the pool per layer, which the run never makes (the rule keeps them on the host below the
    // gpu-waves threshold). That is a plausible reason for "5 whole layers fit at -ub 16 but not at
    // -ub 512": the reserve asked for VRAM the run does not use.
    // MOE_STREAM_RESERVE_OP_OFFLOAD=1 forces the old behaviour, so the cost of the bug is measurable.
    bool moe_waves = false;

    for (int i = 0; i < ggml_graph_n_nodes(gf) && !moe_waves; i++) {
        moe_waves = (ggml_graph_node(gf, i)->flags & (uint32_t) LLAMA_MOE_STREAM_TENSOR_WAVE) != 0;
    }

    // Same rule as the real graph above: a repacked pool cannot be copied to the device, so the
    // wave matmuls stay on the host and the reserved device buffer must not be sized for them
    // (the engine's one rule, waves_on_gpu)
    bool gpu_waves = false;
    if (moe_waves) {
        if (const auto * mstream = ctx_moe_stream()) {
            gpu_waves = mstream->waves_on_gpu(n_tokens);
        }
    }

    const bool reserve_force = getenv("MOE_STREAM_RESERVE_OP_OFFLOAD") != nullptr;
    const bool reserve_offload = reserve_force ? cparams.op_offload
                                               : (cparams.op_offload && (!moe_waves || gpu_waves));

    ggml_backend_sched_set_op_offload(sched.get(), reserve_offload);

    // The reserved graph is scheduled with the same hook, so the compute buffer is sized without the pool copies
    if (auto * ms_t = ctx_moe_stream(); ms_t && ms_t->tail_on) {
        ms_t->tail_real = false;
        ggml_backend_sched_set_copy_hook(sched.get(), moe_waves && gpu_waves && ms_t->tail_hook_ready() ? llama_moe_stream_sched_copy_hook : nullptr, ms_t);
    }

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    // what the reserve asked for, per backend, next to what the run uses (printed once, below)
    if (!split_only && ctx_moe_stream() != nullptr && !getenv("MOE_STREAM_QUIET_RESERVE")) {   // not on stock runs
        std::string per_backend;

        for (const auto & backend_ptr : backends) {
            ggml_backend_t backend = backend_ptr.get();
            per_backend += format(" %s %.1f MiB", ggml_backend_name(backend),
                        ggml_backend_sched_get_buffer_size(sched.get(), backend)/1048576.0);
        }

        // stderr, like the rest of the engine's decisions: the tools drop the library's INFO lines
        fprintf(stderr, "%s: reserve n_tokens %u: waves %s, gpu-waves %s, op_offload %d%s: %s\n",
                __func__, n_tokens, moe_waves ? "yes" : "no", gpu_waves ? "yes" : "no",
                (int) reserve_offload, reserve_force ? " (forced by MOE_STREAM_RESERVE_OP_OFFLOAD)" : "",
                per_backend.c_str());
        n_reserve_printed += 1;
    }

    return gf;
}

// The room a layer-major expert graph takes. It holds, for each wave, one set of nodes per chunk
// of n_ubatch rows, and the waves grow with the call (n_waves(n_tokens)): at -ub 1 a 512-token CPU-only call is 512 chunks x 3 waves and
// overflowed the graph result's object memory (ggml_new_object). Every streamed layer's expert graph is built - not allocated - at m and
// m + 1 chunks, m the fewest chunks with the most waves, in a scratch result with room to spare: the difference / waves is one chunk-
// wave's nodes and object memory, the rest the base. decode() splits a call's experts only while base + chunks x waves x that fits the
// real result's room (its max nodes; its context memory less its graph object), else the call runs in whole-layer groups. Measured for
// this model, -ub, context and pool, not a fixed count
void llama_context::lm_measure_expert_graph() {
    lm_xg = {};
    auto * ms = ctx_moe_stream();
    const uint32_t ub = std::min(cparams.n_ctx, cparams.n_ubatch);
    if (!ms || !ms->p1.layer_major || !llama_arch_layer_major(model.arch) || cparams.n_batch < 2*ub || !gf_res_reserve) {
        return;
    }
    const auto & hp = model.hparams;
    const int64_t k = hp.n_expert_used();
    const int w_max = ms->n_waves(INT32_MAX);
    int64_t m = 1;
    while (ms->n_waves(m*(int64_t) ub) < w_max && m < (int64_t) hp.n_expert) {
        m++;
    }
    const int64_t room_nodes = gf_res_reserve->get_max_nodes(), scratch_nodes = 4*room_nodes + 4096;
    std::unique_ptr<llm_graph_result> res(new llm_graph_result(scratch_nodes));
    const int64_t scratch_gmem = (int64_t) ggml_graph_overhead_custom((size_t) scratch_nodes, false);   // (the graph object, not counted)
    const int64_t hc = std::max<int64_t>(1, hp.dsv4_hc_mult);
    ggml_init_params ip = { 8*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr hctx(ggml_init(ip));
    const auto save = std::make_tuple(lm_il0, lm_il1, lm_phase, lm_row0, lm_hid, lm_x, lm_moe, lm_inj, lm_ids, lm_w, lm_reapply);
    const auto n_outputs_save = n_outputs;
    int64_t base_n = 0, per_n = 0, base_m = 0, per_m = 0;
    int w[2] = { 0, 0 };
    for (int c = 0; c < 2; c++) {
        w[c] = ms->n_waves((m + c)*(int64_t) ub);
    }
    for (int il = 0; il < (int) hp.n_layer(); il++) {
        if (ms->layer(il) == nullptr) {
            continue;
        }
        int64_t nn[2] = { 0, 0 }, mm[2] = { 0, 0 };
        for (int c = 0; c < 2; c++) {
            const int64_t T = (m + c)*(int64_t) ub;
            // stand-ins of the call's tensors, T rows (the expert graph takes its row count from them)
            ggml_tensor * hid = ggml_new_tensor_2d(hctx.get(), GGML_TYPE_F32, hp.n_embd*hc, T);
            ggml_tensor * x   = ggml_new_tensor_2d(hctx.get(), GGML_TYPE_F32, hp.n_embd, T);
            ggml_tensor * moe = hc > 1 ? ggml_new_tensor_2d(hctx.get(), GGML_TYPE_F32, hp.n_embd, T) : nullptr;
            ggml_tensor * inj = hc > 1 ? ggml_new_tensor_2d(hctx.get(), GGML_TYPE_F32, hc, T) : nullptr;
            ggml_tensor * ids = ggml_new_tensor_2d(hctx.get(), GGML_TYPE_I32, k, T);
            ggml_tensor * wt  = ggml_new_tensor_2d(hctx.get(), GGML_TYPE_F32, k, T);
            llama_batch_allocr balloc_x(hp.n_pos_per_embd());
            llama_ubatch ubatch = balloc_x.ubatch_reserve((uint32_t) T, 1);
            ubatch_prepare_reserve(ubatch, (uint32_t) T, sampling.samplers, cparams.n_outputs_max_per_seq);
            lm_il0 = il; lm_il1 = il + 1; lm_phase = 2; lm_row0 = 0; lm_reapply = true;
            lm_hid = hid; lm_x = x; lm_moe = moe; lm_inj = inj; lm_ids = ids; lm_w = wt;
            n_outputs = (int32_t) T;
            res->reset();
            const auto gparams = graph_params(res.get(), ubatch, nullptr, ctx_type_to_graph_type(cparams.ctx_type));
            ms->lm_expert_graph = true;
            ggml_cgraph * gf = model.build_graph(gparams);
            ms->lm_expert_graph = false;
            nn[c] = gf ? ggml_graph_n_nodes(gf) : 0;
            mm[c] = (int64_t) ggml_used_mem(res->get_ctx()) - scratch_gmem;
            ggml_reset(hctx.get());
        }
        // nodes(chunks, waves) = base + chunks x waves x per; m and m + 1 chunks at w[0], w[1] waves (the same, the most)
        const int64_t cw0 = m*w[0], cw1 = (m + 1)*w[1];
        if (cw1 > cw0) {
            const int64_t pn = (nn[1] - nn[0] + cw1 - cw0 - 1)/(cw1 - cw0), pm = (mm[1] - mm[0] + cw1 - cw0 - 1)/(cw1 - cw0);
            per_n  = std::max(per_n,  pn);
            per_m  = std::max(per_m,  pm);
            base_n = std::max(base_n, nn[0] - cw0*pn);
            base_m = std::max(base_m, mm[0] - cw0*pm);
        }
    }
    std::tie(lm_il0, lm_il1, lm_phase, lm_row0, lm_hid, lm_x, lm_moe, lm_inj, lm_ids, lm_w, lm_reapply) = save;
    n_outputs = n_outputs_save;
    if (per_n <= 0 && per_m <= 0) {
        fprintf(stderr, "%s: layer-major: an expert graph does not grow with its chunks (%lld nodes, %lld B) - no chunk limit\n", __func__,
                (long long) base_n, (long long) base_m);
        return;
    }
    lm_xg.on     = true;
    lm_xg.ub     = ub;
    lm_xg.w      = std::max(1, w_max);
    lm_xg.base_n = base_n; lm_xg.per_n = per_n; lm_xg.room_n = room_nodes;
    lm_xg.base_m = base_m; lm_xg.per_m = per_m; lm_xg.room_m = (int64_t) (ggml_tensor_overhead()*(size_t) room_nodes);
    // the longest call that still splits, for the line (at the most waves)
    int64_t lo = 0, hi = cparams.n_batch;
    while (lo < hi) {
        const int64_t mid = (lo + hi + 1)/2;
        if (lm_xg_fits(mid)) { lo = mid; } else { hi = mid - 1; }
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "%s: layer-major: an expert graph takes %lld + %lld nodes and %lld + %lld B of objects per chunk and wave (-ub %u, up to %d "
                "waves; the graph's room %lld nodes, %lld B): a layer-major call splits its experts up to %lld tokens; a longer call runs "
                "in whole-layer groups\n", __func__, (long long) base_n, (long long) per_n, (long long) base_m, (long long) per_m, ub,
                w_max, (long long) lm_xg.room_n, (long long) lm_xg.room_m, (long long) lo);
    }
}

// Does a layer-major call of n_tok tokens fit one expert graph (base + chunks x waves x per, both in nodes and objects;
// one chunk-wave of each kept free: the last chunk of a call may differ a little in shape). Always at the most waves: a chunk's VRAM
// pass is once per chunk, not per wave, so a call with fewer waves must not count less than per x its waves. The most waves at start,
// or the call's own when more: the give-back of pool slots (set_decode, before this) narrows the pools, so a call can have more waves than at start
bool llama_context::lm_xg_fits(int64_t n_tok) const {
    if (!lm_xg.on) {
        return true;
    }
    const int64_t cw = ((n_tok + lm_xg.ub - 1)/lm_xg.ub)*std::max<int64_t>(lm_xg.w, ctx_moe_stream()->n_waves(n_tok));
    return (lm_xg.per_n <= 0 || lm_xg.base_n + (cw + 1)*lm_xg.per_n <= lm_xg.room_n) &&
           (lm_xg.per_m <= 0 || lm_xg.base_m + (cw + 1)*lm_xg.per_m <= lm_xg.room_m);
}

// The graphs a layer-major prompt call runs, built one by one for a reserve (each() then measures it): per kind of layer (its
// phase - 1 = attention + router + shared expert of a streamed layer, 0 = a whole layer -, sliding window or not, recurrent or not, dense
// lead or not) the first such layer's graph for one ubatch, and one expert graph of a few ubatches' rows (its tensors are per
// n_ubatch chunk: its peak does not grow with the call). The call's own tensors (hidden state, expert input, picks, weights) are
// stand-ins in the pool's buffer - a reserve never writes them. false = a graph could not be built / each() said stop
bool llama_context::lend_lm_graphs(const llama_memory_context_i * mctx, const std::function<bool(void)> & each) {
    auto * ms = ctx_moe_stream();
    const auto & hp = model.hparams;
    const bool gpu_waves_chunk = ms->gpu_waves > 0 && cparams.n_ubatch >= ms->gpu_waves && !ms->cpu_repack_on.load() && cparams.op_offload;
    const bool split = gpu_waves_chunk;   // (as decode's lm_split on a GPU run)
    const uint32_t ub = std::min(cparams.n_ctx, cparams.n_ubatch);
    const int64_t  R  = std::max<int64_t>(ub, std::min<int64_t>({ (int64_t) cparams.n_batch, 4*(int64_t) ub, (int64_t) cparams.n_ctx })/ub*ub);
    const int64_t  k  = hp.n_expert_used();
    const int64_t  hc = std::max<int64_t>(1, hp.dsv4_hc_mult);
    ggml_init_params ip = { 6*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr hctx(ggml_init(ip));
    ggml_tensor * hid = ggml_new_tensor_2d(hctx.get(), GGML_TYPE_F32, hp.n_embd*hc, R);
    ggml_tensor * x   = split ? ggml_new_tensor_2d(hctx.get(), GGML_TYPE_F32, hp.n_embd, R) : nullptr;
    ggml_tensor * moe = split && hc > 1 ? ggml_new_tensor_2d(hctx.get(), GGML_TYPE_F32, hp.n_embd, R) : nullptr;
    ggml_tensor * inj = split && hc > 1 ? ggml_new_tensor_2d(hctx.get(), GGML_TYPE_F32, hc, R) : nullptr;
    ggml_tensor * ids = split ? ggml_new_tensor_2d(hctx.get(), GGML_TYPE_I32, k, R) : nullptr;
    ggml_tensor * w   = split ? ggml_new_tensor_2d(hctx.get(), GGML_TYPE_F32, k, R) : nullptr;
    for (ggml_tensor * t : { hid, x, moe, inj, ids, w }) {
        if (t) {
            t->buffer = ms->vx_groups[0].buf;          // (a stand-in: the call's tensors live in the pool's places or a GPU buffer)
            t->data   = ms->vx_groups[0].gate_up->data;
        }
    }
    const auto save = std::make_tuple(lm_il0, lm_il1, lm_phase, lm_row0, lm_hid, lm_x, lm_moe, lm_inj, lm_ids, lm_w, lm_reapply);
    const int32_t n_outputs_save = n_outputs;
    std::set<std::array<int, 5>> kinds;
    int first_streamed = -1;
    bool ok = true;
    for (int il = 0; il < (int) hp.n_layer() && ok; il++) {
        const int ph = split && ms->layer(il) != nullptr ? 1 : 0;
        if (ph == 1 && first_streamed < 0) {
            first_streamed = il;
        }
        if (!kinds.insert({ ph, hp.is_swa(il) ? 1 : 0, hp.is_recr(il) ? 1 : 0, (uint32_t) il < hp.n_layer_dense_lead ? 1 : 0, hp.is_ple(il) ? 1 : 0 }).second) {
            continue;
        }
        lm_il0 = il; lm_il1 = il + 1; lm_phase = ph; lm_row0 = R - ub; lm_reapply = il > 0;
        lm_hid = hid; lm_x = x; lm_moe = moe; lm_inj = inj; lm_ids = ids; lm_w = w;
        ok = each();
    }
    if (ok && first_streamed >= 0) {   // the expert graph: one "ubatch" of R rows, no memory
        lm_il0 = first_streamed; lm_il1 = first_streamed + 1; lm_phase = 2; lm_row0 = 0; lm_reapply = true;
        lm_hid = hid; lm_x = x; lm_moe = moe; lm_inj = inj; lm_ids = ids; lm_w = w;
        ms->lm_expert_graph = true;
        lm_lend_rows = R;
        ok = each();
        lm_lend_rows = 0;
        ms->lm_expert_graph = false;
    }
    std::tie(lm_il0, lm_il1, lm_phase, lm_row0, lm_hid, lm_x, lm_moe, lm_inj, lm_ids, lm_w, lm_reapply) = save;
    n_outputs = n_outputs_save;
    return ok;
}

// The GPU plans of a prompt call's graphs (size queries, or reserved for real with alloc): the one-ubatch whole-model graph at
// each call size of t_pts (by_t[j]), and with layer-major the graphs such a call runs (lm: the most of them). With pieces set, a size is
// what stays in the real buffer. stays (alloc): the GPU tensors of the reserved graphs that did not go into a piece
bool llama_context::lend_measure(const llama_memory_context_i * mctx, const std::vector<int64_t> & t_pts, std::vector<int64_t> & by_t,
                                int64_t & lm, bool alloc, std::map<std::string, std::pair<int64_t, int>> * stays) {
    auto * ms = ctx_moe_stream();
    std::vector<size_t> sz(backend_ptrs.size(), 0);
    by_t.assign(t_pts.size(), 0);
    lm = 0;
    const auto look = [&](ggml_cgraph * gf) {
        if (!stays || !gf) {
            return;
        }
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            const ggml_tensor * t = ggml_graph_node(gf, i);
            if (t->view_src || !t->buffer || ggml_backend_buffer_get_type(t->buffer) != backend_buft[(size_t) lend_be] ||
                    ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
                    std::find(lend_bufs.begin(), lend_bufs.end(), t->buffer) != lend_bufs.end() ||
                    std::find(lend_trial.begin(), lend_trial.end(), t->buffer) != lend_trial.end()) {
                continue;
            }
            std::string n = t->name;
            const size_t d = n.rfind('-');
            if (d != std::string::npos && d + 1 < n.size() && isdigit((unsigned char) n[d + 1])) {
                n = n.substr(0, d);
            }
            auto & e = (*stays)[n];
            e.first   = std::max<int64_t>(e.first, (int64_t) ggml_nbytes(t));
            e.second += 1;
        }
    };
    const uint32_t ub = std::min(cparams.n_ctx, cparams.n_ubatch);
    for (size_t j = 0; j < t_pts.size(); j++) {
        const uint32_t T = (uint32_t) t_pts[j];
        ggml_cgraph * gf = graph_reserve(T, std::min<uint32_t>(cparams.n_seq_max, T), std::min(T, cparams.n_outputs_max), mctx, !alloc,
                                         alloc ? nullptr : sz.data());
        if (!gf) {
            return false;
        }
        by_t[j] = alloc ? 0 : (int64_t) sz[(size_t) lend_be];
        look(gf);
    }
    if (ms->p1.layer_major && memory && llama_arch_layer_major(model.arch) && cparams.n_batch > cparams.n_ubatch) {
        const bool ok = lend_lm_graphs(mctx, [&]() {
            const uint32_t rows = lm_lend_rows > 0 ? (uint32_t) lm_lend_rows : ub;
            ggml_cgraph * gf = graph_reserve(rows, 1, rows, lm_phase == 2 ? nullptr : mctx, !alloc, alloc ? nullptr : sz.data());
            if (!gf) {
                return false;
            }
            lm = std::max<int64_t>(lm, alloc ? 0 : (int64_t) sz[(size_t) lend_be]);
            look(gf);
            return true;
        });
        if (!ok) {
            return false;
        }
    }
    return true;
}

// VRAM lending for prompts, at a reserve (the pool's sizing trials too): does this context lend, and how much? The prompt graphs' plans are measured on
// a grid of context lengths (-c/8, -c/4, -c/2, -c) and call sizes (--moe-stream-vram-lend-short, ub/4, ub/2, ub). The real compute
// buffer is sized for the reply graphs, a short call at -c, and what of the full plan stays in it: a trial plan at -c into the pieces
// the lend gives (partial when the pool cannot give it all) - a tensor no piece can hold stays in the real buffer. Off - one line,
// the plain compute buffer - only when the pool's pieces would hold none of the plan
bool llama_context::lend_decide(const llama_memory_context_i * mctx) {
    auto * ms = ctx_moe_stream();
    lend_be = -1;
    for (size_t i = 0; i < backend_ptrs.size(); i++) {
        if (ggml_backend_get_device(backend_ptrs[i]) == ms->vx_dev) {
            lend_be = (int) i;
        }
    }
    if (ms->lend_from_ptr == nullptr && ms->vx_dev != nullptr) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ms->vx_dev);
        ms->lend_from_ptr = reg ? (ggml_backend_buffer_t (*)(ggml_backend_buffer_type_t, void *, size_t))
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_buffer_from_ptr") : nullptr;
    }
    const auto off = [&](const std::string & w) {
        ms->lend_on = false;
        static std::string said;   // (also in the pool's sizing trials: the last reserve may be one of them)
        if (said != w) {
            said = w;
            fprintf(stderr, "llama_context: VRAM lending: off for this context - %s: the plain compute buffer%s\n", w.c_str(),
                    rtrial_in_trial ? " (in a pool-sizing trial)" : "");
        }
        return false;
    };
    if (lend_be < 0) {
        return off("the VRAM pool's GPU is not one of this context's backends");
    }
    if (ms->lend_from_ptr == nullptr) {
        return off("this GPU backend cannot make a buffer over the pool's memory");
    }
    static const double test_low = getenv("MOE_STREAM_TEST_LEND_LOW") ? std::max(0.01, atof(getenv("MOE_STREAM_TEST_LEND_LOW"))) : 1.0;
    const uint32_t ub = std::min(cparams.n_ctx, cparams.n_ubatch);
    const int64_t  Ts = std::min<int64_t>(ms->p1.vram_lend_short, ub);
    std::vector<int64_t> t_pts = { Ts, (int64_t) ub/4, (int64_t) ub/2, (int64_t) ub }, kv_pts;
    std::sort(t_pts.begin(), t_pts.end());
    t_pts.erase(std::remove_if(t_pts.begin(), t_pts.end(), [&](int64_t t) { return t < Ts; }), t_pts.end());
    t_pts.erase(std::unique(t_pts.begin(), t_pts.end()), t_pts.end());
    for (uint32_t k = 1; k <= 8; k++) {   // (eighths of the context: the next point above a call is at most -c/8 further)
        const int64_t kv = k == 8 ? (int64_t) cparams.n_ctx : std::max<int64_t>(256, (int64_t) cparams.n_ctx*k/8/256*256);
        if (kv_pts.empty() || kv > kv_pts.back()) {
            kv_pts.push_back(kv);
        }
    }
    std::vector<std::vector<int64_t>> grid(kv_pts.size());
    for (size_t i = 0; i < kv_pts.size(); i++) {
        llama_memory_context_ptr mk;
        const llama_memory_context_i * mc = mctx;
        if (memory && kv_pts[i] < (int64_t) cparams.n_ctx) {
            const uint32_t old = llama_kv_cache_reserve_cap((uint32_t) kv_pts[i]);
            mk = memory->init_full();
            llama_kv_cache_reserve_cap(old);
            mc = mk.get();
        }
        std::vector<int64_t> by_t;
        int64_t lm = 0;
        if (!mc || !lend_measure(mc, t_pts, by_t, lm, false, nullptr)) {
            return off("a prompt graph could not be measured");
        }
        by_t.back() = std::max(by_t.back(), lm);   // (the full-ubatch column: a layer-major call's graphs too)
        for (size_t j = 1; j < by_t.size(); j++) {
            by_t[j] = std::max(by_t[j], by_t[j - 1]);   // (the plan never shrinks with more tokens: the bound stays monotone)
        }
        if (i > 0) {
            for (size_t j = 0; j < by_t.size(); j++) {
                by_t[j] = std::max(by_t[j], grid[i - 1][j]);   // (nor with more context)
            }
        }
        grid[i] = by_t;
    }
    std::vector<size_t> szt(backend_ptrs.size(), 0);
    const uint32_t ns = cparams.n_seq_max;
    if (!graph_reserve(ns, ns, ns, mctx, true, szt.data())) {
        return off("a reply graph could not be measured");
    }
    const int64_t tg = (int64_t) szt[(size_t) lend_be], shrt = grid.back().front(), full = grid.back().back();
    const int64_t small = std::max(tg, shrt);
    if (full <= small) {
        return off("the prompt graphs need no more GPU memory than the reply graphs and a short call");
    }
    ms->lend_kv_pts = kv_pts;
    ms->lend_t_pts  = t_pts;
    ms->lend_grid   = grid;
    ms->lend_low    = 1.0;
    ms->lend_on     = true;
    ms->lend_call_begin();    // (the trial's pieces from the first held place)
    // the margin of the pieces over the plan, measured: the plan at -c planned into the pieces of 1.0 / 1.1 / 1.25 x itself; the smallest
    // margin that leaves no more in the real buffer than 1.25 does (the plan's tensors do not tile the pieces perfectly; what does not fit
    // a piece stays in the real buffer)
    const auto left_at = [&](double slack, int64_t & left) {
        ms->lend_slack = slack;
        const auto U = ms->lend_places_for(ms->lend_need(INT32_MAX, (int64_t) cparams.n_ctx));
        int64_t nU = 0;
        for (const int32_t u : U) { nU += u; }
        ms->lend_free_pieces(lend_trial);
        if (nU == 0 || !ms->lend_pieces(U, lend_trial)) {
            lend_trial.clear();
            return false;
        }
        ggml_backend_sched_set_fixed_chunks(sched.get(), backend_ptrs[(size_t) lend_be], lend_trial.data(), (int) lend_trial.size(), true);
        std::vector<int64_t> by_t;
        int64_t lm = 0;
        const bool ok = lend_measure(mctx, t_pts, by_t, lm, false, nullptr);
        left = lm;
        for (const int64_t b : by_t) { left = std::max(left, b); }
        ggml_backend_sched_set_fixed_chunks(sched.get(), backend_ptrs[(size_t) lend_be], nullptr, 0, true);
        return ok;
    };
    int64_t left = 0, left_max = 0;
    if (!left_at(1.25, left_max)) {
        ms->lend_free_pieces(lend_trial);
        return off(lend_trial.empty() ? "the VRAM pool has no held places to lend" : "a prompt graph could not be planned into the pieces");
    }
    double slack = 1.25;
    left = left_max;
    for (const double sl : { 1.0, 1.1 }) {
        int64_t l = 0;
        if (left_at(sl, l) && l <= left_max + (1 << 20)) {
            slack = sl;
            left  = l;
            break;
        }
    }
    if (!left_at(slack, left)) {   // (the pieces of the chosen margin, for the reserve below)
        ms->lend_free_pieces(lend_trial);
        return off("a prompt graph could not be planned into the pieces");
    }
    if (left >= full) {
        ms->lend_free_pieces(lend_trial);
        return off("no tensor of the prompt plan fits a piece the VRAM pool can lend");
    }
    bool ok = true;
    std::vector<int64_t> unused;
    int64_t lm_unused = 0;
    if (!graph_reserve((uint32_t) Ts, std::min<uint32_t>(cparams.n_seq_max, (uint32_t) Ts), std::min((uint32_t) Ts, cparams.n_outputs_max), mctx)) {
        ms->lend_free_pieces(lend_trial);
        return off("a short prompt call's graph could not be reserved");   // (the short call at -c, in the real buffer only)
    }
    std::map<std::string, std::pair<int64_t, int>> stays;
    ggml_backend_sched_set_fixed_chunks(sched.get(), backend_ptrs[(size_t) lend_be], lend_trial.data(), (int) lend_trial.size(), true);
    ok = lend_measure(mctx, t_pts, unused, lm_unused, true, &stays);
    ggml_backend_sched_set_fixed_chunks(sched.get(), backend_ptrs[(size_t) lend_be], nullptr, 0, true);
    ms->lend_free_pieces(lend_trial);
    if (!ok) {
        return off("a prompt graph could not be reserved with its GPU buffer in the pieces");
    }
    ms->lend_low  = test_low;
    ms->lend_real = std::max(small, left);
    if (!rtrial_in_trial) {
        std::vector<std::pair<int64_t, std::string>> v;
        for (const auto & [n, e] : stays) {
            if (e.first >= (1 << 20)) {
                v.push_back({ e.first, format("%s %.1f MiB%s", n.c_str(), e.first/1048576.0, e.second > 1 ? format(" (x%d)", e.second).c_str() : "") });
            }
        }
        std::sort(v.rbegin(), v.rend());
        std::string top;
        for (size_t k = 0; k < v.size() && k < 4; k++) {
            top += (k ? ", " : "") + v[k].second;
        }
        fprintf(stderr, "llama_context: VRAM lending: at -c %u, %.1f MiB of the prompt plan's %.1f MiB stays in the real compute buffer (%s); "
                "the real buffer %.1f MiB (the reply graphs %.1f, a %lld-token call %.1f); the pieces %.2fx the plan (measured)%s\n", cparams.n_ctx,
                left/1048576.0, full/1048576.0, top.empty() ? "nothing above 1 MiB" : top.c_str(), ms->lend_real/1048576.0, tg/1048576.0,
                (long long) Ts, shrt/1048576.0, ms->lend_slack,
                test_low != 1.0 ? format(" - TEST ONLY: the estimates x %.2f (MOE_STREAM_TEST_LEND_LOW)", test_low).c_str() : "");
    }
    return true;
}

// Set the pieces of a lend on the scheduler (the next graph re-plans into them; the real buffer may grow only when asked); the
// last set's buffers are freed now
void llama_context::lend_set(std::vector<ggml_backend_buffer_t> && bufs, const std::vector<int32_t> & U, bool may_grow) {
    auto * ms = ctx_moe_stream();
    ms->lend_free_pieces(lend_bufs_old);
    lend_bufs_old = std::move(lend_bufs);
    lend_bufs     = std::move(bufs);
    lend_U        = U;
    GGML_ASSERT(ggml_backend_sched_set_fixed_chunks(sched.get(), backend_ptrs[(size_t) lend_be], lend_bufs.data(), (int) lend_bufs.size(), may_grow));
    lend_no_grow = !may_grow;
    lend_gen += 1;
}

// The real compute buffer again (the next graph re-plans into it)
void llama_context::lend_clear() {
    auto * ms = ctx_moe_stream();
    if ((lend_bufs.empty() && !lend_no_grow) || ms == nullptr) {
        return;
    }
    ggml_backend_sched_set_fixed_chunks(sched.get(), backend_ptrs[(size_t) lend_be], nullptr, 0, true);   // (and the real buffer may grow again)
    lend_no_grow = false;
    if (!lend_bufs.empty()) {
        ms->lend_free_pieces(lend_bufs_old);
        lend_bufs_old = std::move(lend_bufs);
    }
    lend_bufs.clear();
    lend_U.clear();
    lend_start.clear();
    lend_gen += 1;
}

// VRAM lending, decode: a reply word gives the places back (before set_decode: the fills refill them); a prompt call of at most
// --moe-stream-vram-lend-short tokens, or whose plan fits the real buffer, runs there; any longer prompt call (layer-major or not) lends
// need(T, kv) at its start, before any graph of the call - after the front places. The same places again: the pieces set now
// stay (no re-plan; the places vacated again, the end fill may have refilled them). A lend that cannot give the plan lends what it can
// (the rest stays in the real buffer)
void llama_context::lend_call_start(bool dec, int64_t T, int64_t kv) {
    auto * ms = ctx_moe_stream();
    lend_kv = kv;
    lend_T  = T;
    lend_start_failed = false;
    if (dec) {
        lend_clear();
        return;
    }
    // a prompt call that does not lend runs on the real buffer as it is - not grown (a plan bigger than it fails, and the retry lends)
    const auto plain_no_grow = [&]() {
        if (lend_no_grow && lend_bufs.empty()) {
            return;   // (already so: no re-plan)
        }
        lend_clear();
        ggml_backend_sched_set_fixed_chunks(sched.get(), backend_ptrs[(size_t) lend_be], nullptr, 0, false);
        lend_no_grow = true;
        lend_gen += 1;
    };
    const int64_t need = ms->lend_need(T, kv);
    if (T <= ms->p1.vram_lend_short || need <= ms->lend_real) {
        plain_no_grow();
        return;
    }
    ms->n_lend_calls += 1;
    std::vector<int32_t> cap(ms->vx_groups.size(), 0);   // (after this call's front places)
    for (size_t g = 0; g < cap.size(); g++) {
        const int32_t st = g < ms->lend_start_g.size() ? ms->lend_start_g[g] : ms->vx_groups[g].p0;
        cap[g] = ms->vx_groups[g].h_end() - st;
    }
    const auto U = ms->lend_places_for(need, &cap);
    std::vector<ggml_backend_buffer_t> pcs;
    if (!ms->lend_lend(U, pcs)) {
        plain_no_grow();
        lend_start_failed = true;
        ms->n_lend_fallback += 1;
        if (ms->n_lend_fallback <= 3) {
            fprintf(stderr, "%s: VRAM lending: a prompt call of %lld tokens at context %lld: the VRAM pool could not lend %.1f MiB%s - this call "
                    "uses the real compute buffer\n", __func__, (long long) T, (long long) kv, need/1048576.0,
                    getenv("MOE_STREAM_TEST_LEND_LEND_FAIL") ? " (TEST ONLY: MOE_STREAM_TEST_LEND_LEND_FAIL)" : "");
        }
        return;
    }
    if (U == lend_U && ms->lend_start_g == lend_start && !lend_bufs.empty()) {
        ms->lend_free_pieces(pcs);   // (the same places: the pieces set now stay - no re-plan)
        return;
    }
    lend_set(std::move(pcs), U, false);
    lend_start = ms->lend_start_g;
}

// A graph whose allocation failed - its plan needed more of the real buffer than it has (the estimate was too low), or the
// real buffer could not grow. No lend in the middle of a call (only at a call's start; a mid-call lend crashed in the
// forced-low test): this call's real buffer may grow (the pieces set stay), and later calls lend 25 % more (the
// slack learns, up to 4x); last, the real buffer alone. One line each (the first few)
bool llama_context::lend_alloc_retry() {
    auto * ms = ctx_moe_stream();
    static const bool test_no_grow = getenv("MOE_STREAM_TEST_LEND_NO_GROW") && atoi(getenv("MOE_STREAM_TEST_LEND_NO_GROW")) != 0;   // TEST ONLY
    // the same split graph planned again (never split again: the failed allocation's split already pointed its nodes at the scheduler's
    // input copies - a second split of it reads those as its own tensors, and the graph that reaches the GPU is not the allocated one)
    const auto again = [&]() {
        return ggml_backend_sched_alloc_graph_again(sched.get());
    };
    ms->n_lend_grow += 1;
    if (!test_no_grow) {   // 1. this call's real buffer may grow (the pieces set stay); later calls lend 25 % more
        const size_t before = ggml_backend_sched_get_buffer_size(sched.get(), backend_ptrs[(size_t) lend_be]);
        if (!lend_bufs.empty()) {
            GGML_ASSERT(ggml_backend_sched_set_fixed_chunks(sched.get(), backend_ptrs[(size_t) lend_be], lend_bufs.data(), (int) lend_bufs.size(), true));
        } else {
            ggml_backend_sched_set_fixed_chunks(sched.get(), backend_ptrs[(size_t) lend_be], nullptr, 0, true);
        }
        lend_no_grow = false;
        lend_gen += 1;
        if (again()) {
            ms->lend_slack = std::min(4.0, ms->lend_slack*1.25);
            if (ms->n_lend_grow <= 3) {
                const size_t after = ggml_backend_sched_get_buffer_size(sched.get(), backend_ptrs[(size_t) lend_be]);
                fprintf(stderr, "%s: VRAM lending: a graph's plan needed more than the lent places and the real buffer hold (the estimate was low%s) - "
                        "the real compute buffer grew by %.1f MiB for this call; later calls lend %.2fx the plan\n", __func__,
                        ms->lend_low != 1.0 ? ": TEST ONLY MOE_STREAM_TEST_LEND_LOW" : "",
                        after > before ? (after - before)/1048576.0 : 0.0, ms->lend_slack);
            }
            return true;
        }
    }
    // 2. the card is full (the real buffer cannot grow) and no lend happens in the middle of a call: the call ends cleanly - decode
    // returns 3 (the caller processes the prompt again; llama-server does, as for the layer-major retry guard) with the lend doubled for the next try;
    // when no bigger lend can follow, -2 (an error the caller reports; never an abort, never a loop): the lend already was every place
    // the pool can give, the call lends nothing (a reply word, a short call, or the pool could not lend at its start), or the lend
    // factor is already at its largest - a caller that processes every ended call again (llama-server) would otherwise never stop
    std::vector<int32_t> cap(ms->vx_groups.size(), 0);
    int64_t all = 0;
    for (size_t g = 0; g < cap.size(); g++) {
        const int32_t st = g < ms->lend_start_g.size() ? ms->lend_start_g[g] : ms->vx_groups[g].p0;
        cap[g] = std::max(0, ms->vx_groups[g].h_end() - st);
        all   += (int64_t) cap[g]*(int64_t) (ms->vx_groups[g].gate_up->nb[2] + ms->vx_groups[g].down->nb[2]);
    }
    const bool no_lend = lend_bufs.empty() && (ms->is_decode() || lend_T <= ms->p1.vram_lend_short || lend_start_failed);
    const bool at_max  = no_lend || ms->lend_slack >= 64.0 || (!lend_bufs.empty() && lend_U == ms->lend_places_for(all, &cap));
    if (!at_max) {
        ms->lend_slack = std::min(64.0, ms->lend_slack*2.0);
        lend_again     = true;
    }
    ms->n_lend_fallback += 1;
    if (ms->n_lend_fallback <= 5) {
        fprintf(stderr, "%s: VRAM lending: a graph's plan does not fit the lent places and the real compute buffer cannot grow%s - %s\n", __func__,
                test_no_grow ? " (TEST ONLY: MOE_STREAM_TEST_LEND_NO_GROW)" : "",
                no_lend ? "this call cannot lend from the VRAM pool: the call fails (decode -2)" :
                at_max  ? "no bigger lend is possible: the call fails (decode -2)" :
                format("the call ends, decode returns 3 (the caller processes the prompt again) and lends %.2fx the plan", ms->lend_slack).c_str());
    }
    lend_clear();
    return false;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    llm_graph_params p = {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.mstream     =*/ ctx_moe_stream(),
        /*.mstream_gen =*/ ctx_moe_stream() ? ctx_moe_stream()->graph_gen ^ (ctx_moe_stream()->ne_gpu_graph() ? 0x80000000u : 0u) : 0u,   // a needed-early one-token graph is not reused in another kind of call
        /*.prec_policy =*/ &model.prec_policy,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
    p.lm_il0 = lm_il0;
    p.lm_hid  = lm_il1 >= 0 || lm_il0 > 0 ? lm_hid : nullptr;   // (only the group graphs of a layer-major call)
    p.lm_row0 = lm_row0;
    p.lm_phase = lm_phase;
    p.lm_x     = lm_phase != 0 ? lm_x   : nullptr;
    p.lm_moe   = lm_phase != 0 ? lm_moe : nullptr;
    p.lm_inj   = lm_phase != 0 ? lm_inj : nullptr;
    p.lm_ids   = lm_phase != 0 ? lm_ids : nullptr;
    p.lm_w     = lm_phase != 0 ? lm_w   : nullptr;
    p.lm_chunk = cparams.n_ubatch;
    p.lm_il1 = lm_il1;
    return p;
}

void llama_context::sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data) {
    // reallocating after the backend assignments changed is expected, trace only the unexpected ones
    if (backend_ids_changed) {
        return;
    }

    auto * lctx = static_cast<llama_context *>(user_data);

    if (graph_size_changed) {
        lctx->n_reallocs_diff++;
    } else {
        lctx->n_reallocs_same++;
    }

    LLAMA_LOG_INFO("%s: graph reallocation: nodes = %d, splits = %d, n_same = %d, n_diff = %d\n",
            __func__, ggml_graph_n_nodes(gf), ggml_backend_sched_get_n_splits(sched),
            lctx->n_reallocs_same, lctx->n_reallocs_diff);
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void discard() override {
        rinfos.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
                    }
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                const size_t   el   = ggml_element_size(src_t);
                const int64_t n_el = (int64_t) (n_copy / el);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                ggml_backend_tensor_copy(src_v, dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io->discard();
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

//
// Checkpoints inside one call (llama-server's context checkpoints without cutting the prompt into calls)
//

llama_memory_recurrent * llama_context::cc_recr() const {
    auto * hy = dynamic_cast<llama_memory_hybrid *>(memory.get());
    return hy ? hy->get_mem_recr() : nullptr;
}

// only where the call runs layer by layer (Ling-3.0 with --moe-stream-layer-major 1), with a recurrent part and no rollback planes;
// elsewhere the server keeps cutting the prompt (the default)
bool llama_context::call_checkpoints_ok() const {
    const auto * ms   = ctx_moe_stream();
    const auto * recr = cc_recr();
    return ms != nullptr && ms->p1.layer_major && model.arch == LLM_ARCH_BAILINGMOE3 && recr != nullptr && recr->n_rs_seq == 0 &&
           cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP && !cparams.embeddings;
}

// the staging of n_keep checkpoints inside a call: grows to the most asked for, then kept (the RAM plan counts --ctx-checkpoints + 1 of
// them). llama-server asks for --ctx-checkpoints of them at load (llama_call_checkpoints_reserve), so the first request does
// not pay the pinned allocation (26.8 ms on Ling Flash)
bool llama_context::cc_reserve(size_t n_keep) {
    auto * recr = cc_recr();
    if (recr == nullptr) {
        return false;
    }
    if (cc_bytes == 0) {   // the staging layout: the sequence cell's row of every recurrent tensor (its order does not matter)
        size_t off = 0;
        for (const auto * v : { &recr->r_l, &recr->p_l, &recr->s_l }) {
            for (const ggml_tensor * t : *v) {
                if (t != nullptr) {
                    cc_off[t] = off;
                    off += GGML_PAD(t->nb[1], 64);
                }
            }
        }
        cc_bytes = off;
    }
    const size_t need = n_keep*cc_bytes;
    const size_t have = cc_buf ? ggml_backend_buffer_get_size(cc_buf.get()) : cc_host.size();
    if (need > have) {
        synchronize();   // (no copy of an earlier call may still be writing into the old staging)
        cc_buf.reset();
        cc_host.clear();
        for (auto & be : backends) {   // pinned memory of the first GPU: the device-to-host copies then run without a host wait
            ggml_backend_dev_t dev = ggml_backend_get_device(be.get());
            if (dev && llama_ctx_is_gpu(dev, ctx_moe_stream())) {
                if (ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(dev)) {
                    cc_buf.reset(ggml_backend_buft_alloc_buffer(hbt, need));
                    if (cc_buf) {
                        break;
                    }
                }
            }
        }
        if (!cc_buf) {
            cc_host.resize(need);
        }
        fprintf(stderr, "%s: layer-major: %.1f MiB of staging for %zu checkpoints inside a call (%s)\n", __func__, need/1048576.0,
                n_keep, cc_buf ? ggml_backend_buffer_name(cc_buf.get()) : "host memory");
    }
    return true;
}

bool llama_context::set_call_cuts(llama_seq_id seq_id, const llama_pos * cuts, const bool * keep, int32_t n) {
    cc_cuts.clear();
    cc_keep.clear();
    cc_seq = -1;
    if (!call_checkpoints_ok() || seq_id < 0 || cuts == nullptr || n <= 0) {
        return false;
    }
    for (int32_t i = 0; i < n; ++i) {
        cc_cuts.push_back(cuts[i]);
        if (keep != nullptr && keep[i]) {
            cc_keep.push_back(cuts[i]);
        }
    }
    std::sort(cc_cuts.begin(), cc_cuts.end());
    cc_cuts.erase(std::unique(cc_cuts.begin(), cc_cuts.end()), cc_cuts.end());
    std::sort(cc_keep.begin(), cc_keep.end());
    cc_keep.erase(std::unique(cc_keep.begin(), cc_keep.end()), cc_keep.end());
    if (!cc_reserve(cc_keep.size())) {
        cc_cuts.clear();
        cc_keep.clear();
        return false;
    }
    cc_seq = seq_id;
    return true;
}

uint8_t * llama_context::cc_stage(size_t i_keep) {
    uint8_t * base = cc_buf ? (uint8_t *) ggml_backend_buffer_get_base(cc_buf.get()) : cc_host.data();
    return base + i_keep*cc_bytes;
}

// after the graph of ubatch ub for layers [il_from, il_to): when ub ends right before a kept cut, the sequence cell's rows of those
// layers' recurrent tensors go to that cut's staging - on the backend that holds them, behind the graph in its stream (a GPU: an
// async copy into pinned memory, no wait; the CPU: its graph has finished, a plain copy)
void llama_context::cc_copy(const llama_ubatch & ub, int il_from, int il_to) {
    if (!cc_active || ub.n_tokens == 0) {
        return;
    }
    const llama_pos end = ub.pos[ub.n_tokens - 1] + 1;
    const auto it = std::lower_bound(cc_keep.begin(), cc_keep.end(), end);
    if (it == cc_keep.end() || *it != end) {
        return;
    }
    const size_t k = (size_t) (it - cc_keep.begin());
    auto * recr = cc_recr();
    const int32_t cell = (size_t) cc_seq < recr->cells.size() ? recr->cells[cc_seq].tail : -1;
    if (cell < 0) {
        return;
    }
    uint8_t * dst = cc_stage(k);
    const int n_l = (int) recr->r_l.size();
    for (int il = std::max(0, il_from); il < std::min(il_to, n_l); ++il) {
        bool any = false;
        for (ggml_tensor * t : { recr->r_l[il], recr->p_l[il], recr->s_l[il] }) {
            if (t == nullptr) {
                continue;
            }
            ggml_backend_t be = nullptr;
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
            for (auto & b : backends) {
                if (ggml_backend_get_device(b.get()) == dev) {
                    be = b.get();
                    break;
                }
            }
            if (be != nullptr) {
                ggml_backend_tensor_get_async(be, t, dst + cc_off.at(t), (size_t) cell*t->nb[1], t->nb[1]);
            } else {
                ggml_backend_tensor_get(t, dst + cc_off.at(t), (size_t) cell*t->nb[1], t->nb[1]);
            }
            any = true;
        }
        if (any) {
            cc_layer_done[k][il] = true;
        }
    }
}

// the bytes of llama_state_seq_get_data_ext(..., PARTIAL_ONLY) with the staged rows: the usual writer runs on the memory as it is, the
// tensor reads come from the staging and the cell's position is the cut's (pos - 1) while it writes - the format cannot drift apart
class llama_io_write_staged : public llama_io_write_i {
public:
    llama_io_write_staged(uint8_t * p, size_t len, const uint8_t * stage, const std::map<const ggml_tensor *, size_t> & off, int32_t cell)
        : ptr(p), buf_size(len), stage(stage), off(off), cell(cell) {}

    void write(const void * src, size_t size) override {
        if (ptr != nullptr) {
            if (size > buf_size) {
                throw std::runtime_error("unexpectedly reached end of buffer");
            }
            memcpy(ptr, src, size);
            ptr += size;
            buf_size -= size;
        }
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const auto it = off.find(tensor);
        if (it == off.end() || size != tensor->nb[1] || offset != (size_t) cell*tensor->nb[1]) {
            throw std::runtime_error(format("layer-major: the checkpoint asks for %s at %zu (%zu bytes), not the staged row",
                    tensor->name, offset, size));
        }
        write(stage + it->second, size);
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size;
    size_t size_written = 0;
    const uint8_t * stage;
    const std::map<const ggml_tensor *, size_t> & off;
    int32_t cell;
};

size_t llama_context::get_call_checkpoint(llama_pos pos, uint8_t * dst, size_t size) {
    synchronize();
    const auto it = std::lower_bound(cc_keep.begin(), cc_keep.end(), pos);
    if (it == cc_keep.end() || *it != pos || std::find(cc_done.begin(), cc_done.end(), pos) == cc_done.end()) {
        return 0;
    }
    auto * recr = cc_recr();
    const int32_t cell = recr->cells[cc_seq].tail;
    if (cell < 0) {
        return 0;
    }
    llama_io_write_staged io(dst, dst ? size : 0, cc_stage((size_t) (it - cc_keep.begin())), cc_off, cell);
    llama_pos & cpos = recr->cells[cell].pos;
    const llama_pos pos_now = cpos;
    cpos = pos - 1;
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&cc_seq, sizeof(cc_seq));
        memory->state_write(io, cc_seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    } catch (const std::exception & err) {
        cpos = pos_now;
        LLAMA_LOG_ERROR("%s: %s\n", __func__, err.what());
        return 0;
    }
    cpos = pos_now;
    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    data.n_reallocs_same = n_reallocs_same;
    data.n_reallocs_diff = n_reallocs_diff;

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;

    // note: the reallocation counters are not reset - they track the lifetime totals for this context
}

int64_t llama_context::memory_bytes() const {
    int64_t n = 0;
    if (memory) {
        for (const auto & kv : memory->memory_breakdown()) {
            n += (int64_t) kv.second;
        }
    }
    return n;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.no_alloc()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;

        // the graph changes without flash attention, need to reserve again
        sched_need_reserve = true;
        sched_reserve();
    }

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_prev_active = nullptr;
            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    llama_memory_note_alloc_failure(false);

    try {
        auto * ctx = new llama_context(*model, params);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

bool llama_flash_attn_enabled(const llama_context * ctx) {
    return ctx->get_cparams().flash_attn;
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

bool llama_get_causal_attn(const llama_context * ctx) {
    return ctx->get_causal_attn();
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

bool llama_call_checkpoints_ok(llama_context * ctx) {
    return ctx->call_checkpoints_ok();
}

bool llama_set_call_cuts(llama_context * ctx, llama_seq_id seq_id, const llama_pos * cuts, const bool * keep, int32_t n) {
    return ctx->set_call_cuts(seq_id, cuts, keep, n);
}

bool llama_call_checkpoints_reserve(llama_context * ctx, int32_t n_keep) {
    return ctx->call_checkpoints_ok() && n_keep > 0 && ctx->cc_reserve((size_t) n_keep);
}

size_t llama_get_call_checkpoint(llama_context * ctx, llama_pos pos, uint8_t * dst, size_t size) {
    return ctx->get_call_checkpoint(pos, dst, size);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
    LLAMA_LOG_INFO("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);

    llama_moe_stream_print_stats(ctx);
}

// The rotation sizes of the attention cache of the known memory kinds (our own fix for --cache-disk)
bool llama_get_kv_rot(const llama_context * ctx, uint32_t * n_rot_k, uint32_t * n_rot_v, uint32_t * n_rot_k_default, uint32_t * n_rot_v_default) {
    *n_rot_k = *n_rot_v = *n_rot_k_default = *n_rot_v_default = 0;
    const llama_memory_i * mem = ctx ? ctx->get_memory() : nullptr;
    const llama_kv_cache * kv = nullptr;
    if (const auto * m = dynamic_cast<const llama_kv_cache *>(mem)) {
        kv = m;
    } else if (const auto * m = dynamic_cast<const llama_kv_cache_iswa *>(mem)) {
        kv = m->get_base();
    } else if (const auto * m = dynamic_cast<const llama_memory_hybrid *>(mem)) {   // also llama_memory_hybrid_idx
        kv = m->get_mem_attn();
    } else if (const auto * m = dynamic_cast<const llama_memory_hybrid_iswa *>(mem)) {
        kv = m->get_mem_attn() ? m->get_mem_attn()->get_base() : nullptr;
    }
    if (!kv) {
        return false;
    }
    *n_rot_k         = kv->get_n_rot_k();
    *n_rot_v         = kv->get_n_rot_v();
    *n_rot_k_default = kv->get_n_rot_k(true);
    *n_rot_v_default = kv->get_n_rot_v(true);
    return true;
}

// The experts of a layer the VRAM expert pool holds now (counted from the live table; while a swap is made it can be one off)
static int32_t llama_moe_stream_vram_held(const llama_moe_stream_layer & sl) {
    int32_t n = 0;
    for (const int32_t s : sl.vx_slot) {
        n += s >= 0 ? 1 : 0;
    }
    return n;
}

int32_t llama_moe_stream_get_layer_stats(const llama_context * ctx, llama_moe_stream_layer_stats * out, int32_t n_max) {
    const auto * mstream = ctx ? ctx->get_model().moe_stream() : nullptr;

    if (!mstream) {
        return 0;
    }

    int32_t n = 0;
    for (const auto & slp : mstream->layers) {
        if (!slp) {
            continue;
        }
        if (out && n < n_max) {
            llama_moe_stream_layer_stats & o = out[n];
            o = {};
            o.il           = slp->il;
            o.ram_slots    = (int32_t) slp->share;
            o.vram_experts = llama_moe_stream_vram_held(*slp);
            for (int kind = 0; kind < 2; kind++) {
                const auto & c = kind == 0 ? slp->pre : slp->dec;
                auto &       d = kind == 0 ? o.pre    : o.dec;
                d.n_hit      = c.n_hit;
                d.n_miss     = c.n_miss;
                d.n_vram_hit = c.n_vram_hit;
                d.t_stall_us = c.t_stall_us;
                d.bytes_read = c.bytes_read;
            }
        }
        n++;
    }
    return n;
}

int32_t llama_moe_stream_gpu_layers(const llama_model * model, int32_t * out, int32_t n_max) {
    const auto * mstream = model ? model->moe_stream() : nullptr;

    if (!mstream) {
        return 0;
    }

    const auto & pick = mstream->gpu_layers_pick;
    for (size_t i = 0; out && i < pick.size() && (int32_t) i < n_max; i++) {
        out[i] = pick[i];
    }

    return (int32_t) pick.size();
}

bool llama_moe_stream_ctx_failed_on_gpu(void) {
    return llama_memory_alloc_failed_on_device();
}

int32_t llama_moe_stream_gpu_layer_to_drop(const llama_model * model) {
    const auto * mstream = model ? model->moe_stream() : nullptr;
    return mstream ? mstream->whole_layer_to_drop(llama_memory_alloc_failed_dev()) : -1;
}

bool llama_moe_stream_get_layout(const llama_context * ctx, llama_moe_stream_layout * out) {
    const auto * mstream = ctx ? ctx->get_model().moe_stream() : nullptr;

    if (!mstream || !out) {
        return false;
    }

    *out = {};
    out->n_layer       = (int32_t) mstream->layers.size();
    out->n_expert      = (int32_t) mstream->n_expert_total;
    out->n_expert_used = (int32_t) mstream->n_expert_used;
    out->ram_slots     = (int32_t) mstream->n_slots;
    out->vram_experts  = mstream->vx_ready ? (int32_t) mstream->p1.vram_experts : 0;
    out->vram_follow   = mstream->vf_word_end_fn != nullptr ? 1 : 0;
    out->n_gpu_layers  = (int32_t) mstream->gpu_layers.size();
    for (size_t i = 0; i < mstream->gpu_layers.size() && i < LLAMA_MOE_STREAM_MAX_GPU_LAYERS; i++) {
        out->gpu_layers[i] = mstream->gpu_layers[i];
    }
    out->ram_pool_bytes  = mstream->plan().pool_bytes;
    out->vram_pool_bytes = mstream->vx_ready ? mstream->vx_bytes : 0;
    out->kv_bytes        = ctx->memory_bytes();
    out->timing          = mstream->p1.timing ? 1 : 0;
    for (const auto & slp : mstream->layers) {
        if (slp) {
            out->n_streamed++;
            if (out->block_bytes == 0) {
                out->block_bytes = (int64_t) slp->block_bytes;
            }
        }
    }
    if (mstream->el_on) {
        const uint32_t mn = mstream->el_min_share_pub.load(std::memory_order_relaxed);
        const uint32_t eff_mn = mn > 0 ? mn : mstream->n_slots;
        out->elastic_on             = 1;
        out->elastic_slots_min      = (int32_t) eff_mn;
        out->elastic_slots_soft_min = mstream->el_soft_min_layer;
        out->elastic_given_bytes    = mstream->el_net_given_bytes.load(std::memory_order_relaxed);
        out->elastic_band_bytes     = mstream->el_dz_now.load(std::memory_order_relaxed) * out->block_bytes;
        out->elastic_free_bytes     = mstream->el_last_room_bytes.load(std::memory_order_relaxed);
        out->elastic_state          = (mstream->el_soft_min_layer > 0 && (int32_t) eff_mn < mstream->el_soft_min_layer) ? 2
                                    : mstream->el_taking ? 3
                                    : (eff_mn < mstream->n_slots || out->elastic_given_bytes > 0) ? 1 : 0;
    }
    return true;
}

bool llama_moe_stream_get_stats(const llama_context * ctx, llama_moe_stream_stats * out) {
    const auto * mstream = ctx ? ctx->get_model().moe_stream() : nullptr;

    if (!mstream) {
        return false;
    }

    llama_moe_stream_stats res = {};
    res.n_slots       = (int32_t) mstream->n_slots;
    res.n_expert_used = (int32_t) mstream->n_expert_used;

    for (const auto & slp : mstream->layers) {
        if (!slp) {
            continue;
        }

        res.n_layers++;
        // Per target index among the streamed layers (the word-start op reaches the first N of them)
        {
            const int32_t j = slp->il >= 0 && (size_t) slp->il < mstream->moe_index_of.size() ? mstream->moe_index_of[slp->il] : -1;
            if (j == 0) {
                res.n_first_miss = slp->dec.n_miss;
            }
            if (j >= 0 && j < 8) {
                res.n_ws_add      [j] = slp->dec.n_ws_add;
                res.n_ws_bind     [j] = slp->dec.n_ws_bind;
                res.n_ws_bind_pick[j] = slp->dec.n_ws_bind_pick;
                for (int i = 0; i < MOE_STREAM_PL_NREASON; i++) {
                    res.n_ws_rm[j] += slp->dec.n_ws_rm[i];
                }
                for (int i = 0; i < 4; i++) {
                    res.n_ws_pick[j][i] = slp->dec.n_ws_pick[i];
                }
            }
        }
        for (int phase = 0; phase < 2; phase++) {
            const auto & c = phase == 0 ? slp->pre : slp->dec;
            auto * d       = phase == 0 ? &res.pre : &res.dec;

            d->n_calls      += c.n_calls;
            d->n_distinct   += c.n_distinct;
            d->n_hit        += c.n_hit;
            d->n_miss       += c.n_miss;
            d->n_vram_hit   += c.n_vram_hit;
            d->n_hit_pref   += c.n_hit_pref;
            d->n_pref       += c.n_pref;
            d->n_pref_waste += c.n_pref_waste;
            d->n_pref_late  += c.n_pref_late;
            d->n_reads      += c.n_reads;
            d->n_chunks     += c.n_chunks;
            d->n_cancel_wait += c.n_cancel_wait;
            d->n_read_retry += c.n_read_retry;
            d->t_stall_us   += c.t_stall_us;
            d->t_cb_wait_us  += c.t_cb_wait_us;
            d->t_wait_all_us += c.t_wait_all_us;
            d->n_copyback   += c.n_copyback;
            d->n_cb_late    += c.n_cb_late;
            d->t_op_us      += c.t_op_us;
            d->bytes_read   += c.bytes_read;
            d->t_read_us    += c.t_read_us;
            // The two-layer lead's cancels and the read bytes by who asked for them
            d->n_pref_cancel        += c.n_pref_cancel;
            d->n_pref_cancel_bytes  += c.n_pref_cancel_bytes;
            d->bytes_demand         += c.bytes_demand;
            d->bytes_prefetch       += c.bytes_prefetch;
            // What the drive was already doing when the compute needed a read
            d->n_inflight_at_demand      += c.n_inflight_at_demand;
            d->n_inflight_at_demand_pref += c.n_inflight_at_demand_pref;
            for (int b = 0; b < 17; b++) {
                d->n_inflight_hist[b] += c.n_inflight_hist[b];
            }
            d->n_pref_capped += c.n_pref_capped;
            // The achieved lead of the guesses, and the guess op's cost on the graph thread
            for (int b = 0; b < 225; b++) {
                d->lead_hist[b] += c.lead_hist[b];
            }
            d->lead_sum_us   += c.lead_sum_us;
            d->lead_max_us    = std::max(d->lead_max_us, c.lead_max_us);
            d->t_pred_us      += c.t_pred_us;
            d->t_pred_lock_us += c.t_pred_lock_us;
            d->n_pred_ops     += c.n_pred_ops;
            // The causes of the not-READY experts and their read-backs
            for (int i = 0; i < 4; i++) {
                d->n_cause[i]    += c.n_cause[i];
                d->t_cause_us[i] += c.t_cause_us[i];
            }
            d->n_rb8        += c.n_rb8;
            d->n_rb8_guess  += c.n_rb8_guess;
            d->n_rb64       += c.n_rb64;
            d->n_rb64_guess += c.n_rb64_guess;
            d->n_rb_prompt  += c.n_rb_prompt;
            // The waiting list and the goal number
            d->n_pl_add       += c.n_pl_add;
            d->n_pl_bind      += c.n_pl_bind;
            // The public array has one entry per reason the engine has
            static_assert(sizeof(d->n_pl_rm)/sizeof(d->n_pl_rm[0]) == MOE_STREAM_PL_NREASON, "llama_moe_stream_stats.n_pl_rm size");
            for (int i = 0; i < MOE_STREAM_PL_NREASON; i++) {
                d->n_pl_rm[i] += c.n_pl_rm[i];
            }
            d->t_pl_delay_us  += c.t_pl_delay_us;
            for (int i = 0; i < 100; i++) {
                d->pl_delay_hist[i] += c.pl_delay_hist[i];
            }
            d->n_pl_used      += c.n_pl_used;
            d->n_pl_used_late += c.n_pl_used_late;
            d->n_pl_bind_pick          += c.n_pl_bind_pick;
            d->n_pl_bind_after         += c.n_pl_bind_after;
            d->n_pl_stale_cancel       += c.n_pl_stale_cancel;
            d->n_pl_stale_cancel_bytes += c.n_pl_stale_cancel_bytes;
            d->n_miss_stale            += c.n_miss_stale;
            // The guess helper
            d->n_ne_post     += c.n_ne_post;
            d->n_ne_done     += c.n_ne_done;
            d->n_ne_late     += c.n_ne_late;
            d->t_ne_delay_us += c.t_ne_delay_us;
            for (int i = 0; i < 9; i++) {
                d->n_ne_late_d[i] += c.n_ne_late_d[i];
            }
            for (int i = 0; i < 100; i++) {
                d->ne_delay_hist[i] += c.ne_delay_hist[i];
            }
            // The entries skipped while behind and what became of them
            d->n_bh_skip        += c.n_bh_skip;
            d->n_bh_skip_bound  += c.n_bh_skip_bound;
            d->n_bh_skip_picked += c.n_bh_skip_picked;
            d->n_bh_skip_stale  += c.n_bh_skip_stale;
            d->n_bh_skip_other  += c.n_bh_skip_other;
            // The guess's extra ranks (the public array has one entry per removal reason, as n_pl_rm)
            static_assert(sizeof(d->n_gx_rm)/sizeof(d->n_gx_rm[0]) == MOE_STREAM_PL_NREASON, "llama_moe_stream_stats.n_gx_rm size");
            d->n_gx_add       += c.n_gx_add;
            d->n_gx_bind      += c.n_gx_bind;
            for (int i = 0; i < MOE_STREAM_PL_NREASON; i++) {
                d->n_gx_rm[i] += c.n_gx_rm[i];
            }
            d->n_gx_used      += c.n_gx_used;
            d->n_gx_used_late += c.n_gx_used_late;
            d->n_gx_bind_pick += c.n_gx_bind_pick;
            d->n_gx_stale_cx  += c.n_gx_stale_cx;
            // Lookahead
            for (int i = 0; i < 9; i++) {
                d->n_la_named[i]     += c.n_la_named[i];
                d->n_la_new[i]       += c.n_la_new[i];
                d->n_la_vote[i]      += c.n_la_vote[i];
                d->n_la_pace[i]      += c.n_la_pace[i];
                d->n_la_pool[i]      += c.n_la_pool[i];
                d->n_la_bind[i]      += c.n_la_bind[i];
                d->n_la_bind_pick[i] += c.n_la_bind_pick[i];
                d->n_la_used[i]      += c.n_la_used[i];
                d->n_la_used_late[i] += c.n_la_used_late[i];
            }
            d->n_dr_became    += c.n_dr_became;
            d->n_dr_bound     += c.n_dr_bound;
            d->n_dr_again     += c.n_dr_again;
            d->n_dr_removed   += c.n_dr_removed;
            d->n_mw_back      += c.n_mw_back;
            d->n_mw_promoted  += c.n_mw_promoted;
            d->n_mw_cancelled += c.n_mw_cancelled;
            d->n_mw_again     += c.n_mw_again;
            d->n_pn_refused       += c.n_pn_refused;
            d->n_pn_noroom        += c.n_pn_noroom;
            d->n_pn_kept          += c.n_pn_kept;
            d->n_pn_kept_picked   += c.n_pn_kept_picked;
            d->n_pn_kept_unpicked += c.n_pn_kept_unpicked;
            d->n_goal_cand    += c.n_goal_cand;
            d->n_goal_ready   += c.n_goal_ready;
            d->n_ig_calls     += c.n_ig_calls;
            d->t_ig_ab_us     += c.t_ig_ab_us;
            d->n_ig_reads     += c.n_ig_reads;
            d->t_ig_a_us      += c.t_ig_a_us;
            d->n_pr_early     += c.n_pr_early;
            d->n_pr_gap       += c.n_pr_gap;
            d->t_pr_gap_us    += c.t_pr_gap_us;
            d->n_pr_split     += c.n_pr_split;
        }
        res.n_calls       += slp->pre.n_calls + slp->dec.n_calls;
        res.n_tokens      += slp->pre.n_tok + slp->dec.n_tok;
        res.n_distinct    += slp->pre.n_distinct + slp->dec.n_distinct;
        res.n_hit         += slp->pre.n_hit + slp->dec.n_hit;
        res.n_miss        += slp->pre.n_miss + slp->dec.n_miss;
        res.n_hit_pref    += slp->pre.n_hit_pref + slp->dec.n_hit_pref;
        res.n_pref        += slp->pre.n_pref + slp->dec.n_pref;
        res.n_pref_waste  += slp->pre.n_pref_waste + slp->dec.n_pref_waste;
        res.n_pref_late   += slp->pre.n_pref_late + slp->dec.n_pref_late;
        res.n_oracle_cmp  += slp->n_oracle_cmp;
        res.n_oracle_miss += slp->n_oracle_miss;
        res.n_or_cmp_dec  += slp->n_or_cmp_dec;
        res.n_or_miss_dec += slp->n_or_miss_dec;
        res.n_wave_ubatches += slp->n_wave_ubatches;
        res.n_wave_experts  += slp->n_wave_experts;
        res.n_wave_reads    += slp->n_wave_reads;
        res.n_wave_stages   += slp->n_wave_stages;
        res.n_reads       += slp->pre.n_reads + slp->dec.n_reads;
        res.n_cancel_wait += slp->pre.n_cancel_wait + slp->dec.n_cancel_wait;
        res.n_read_retry  += slp->pre.n_read_retry + slp->dec.n_read_retry;
        res.t_stall_us    += slp->pre.t_stall_us + slp->dec.t_stall_us;
        // A use the VRAM expert pool served is neither a RAM hit nor a miss, but it is an expert use
        res.n_vram_hit    += slp->pre.n_vram_hit + slp->dec.n_vram_hit;
        res.n_expert_uses += slp->pre.n_hit + slp->pre.n_miss + slp->dec.n_hit + slp->dec.n_miss +
                             slp->pre.n_vram_hit + slp->dec.n_vram_hit;
        res.bytes_read    += slp->pre.bytes_read + slp->dec.bytes_read;
        res.t_read_us     += slp->pre.t_read_us + slp->dec.t_read_us;
        res.t_op_us       += slp->pre.t_op_us + slp->dec.t_op_us;
        res.t_read_win_us  = mstream->t_read_win_us;
    }

    // every streamed layer sees every token
    if (res.n_layers > 0) {
        res.n_tokens /= res.n_layers;
    }

    // The per-phase token counts are counted once per batch by set_decode(), so they are the
    // real tokens of that phase - the divisor of every per-token figure the tool prints
    res.pre.n_tokens = mstream->n_tok_outer_pre;
    res.dec.n_tokens = mstream->n_tok_outer;
    if (mstream->p1.timing) {   // --moe-stream-timing: the totals of the prompt calls and of the decode calls (engine-wide)
        for (int kind = 0; kind < 2; kind++) {
            auto * d = kind == 0 ? &res.pre : &res.dec;
            const std::atomic<int64_t> * t = mstream->tm_us[kind];
            d->n_tm_calls   = t[0].load(std::memory_order_relaxed);
            d->t_tm_wall_us = t[1].load(std::memory_order_relaxed);
            d->t_tm_host_us = t[2].load(std::memory_order_relaxed);
            d->t_tm_dev_us  = t[3].load(std::memory_order_relaxed);
            d->t_tm_copy_us = t[4].load(std::memory_order_relaxed);
            d->t_tm_sync_us = t[5].load(std::memory_order_relaxed);
        }
    }
    res.dec.n_ne_drain_calls = mstream->n_ne_drain_calls;   // engine-wide
    res.n_rd_split     = mstream->n_rd_split;       // engine-wide (decode calls only)
    res.n_rd_ready     = mstream->n_rd_ready;
    res.n_rd_event     = mstream->n_rd_event;
    res.t_rd_us        = mstream->t_rd_us;
    res.n_rd_uncovered = mstream->n_rd_uncovered;
    res.n_ne_ready     = mstream->n_ne_ready;
    res.n_ne_event     = mstream->n_ne_event;
    res.n_rd_unfit     = mstream->n_rd_unfit;
    {
        std::lock_guard<std::mutex> lock(mstream->mtx);   // VRAM follow's counters are engine-wide, under the engine's lock
        res.n_vf_words = mstream->vf_words;             res.n_vf_ticks = mstream->vf_ticks;
        res.n_vf_swaps = mstream->vf_swaps;             res.n_vf_switches = mstream->vf_switches;
        res.n_vf_copies = mstream->vf_copies.load();
        res.n_vf_skip_loading = mstream->vf_skip_loading; res.n_vf_skip_notram = mstream->vf_skip_notram;
        res.n_vf_skip_marks = mstream->vf_skip_marks;   res.n_vf_switch_wait = mstream->vf_switch_wait;
        res.n_vf_mark_mismatch = mstream->vf_mark_mismatch; res.n_vf_marked_prompt = mstream->vf_marked_prompt;
        res.n_vf_src_ram = mstream->vf_src_ram;         res.n_vf_src_ssd = mstream->vf_src_ssd;
        res.n_vf_src_waited = mstream->vf_src_waited;   res.n_vf_verified = mstream->vf_verified;
        res.n_vf_hysteresis_stop = mstream->vf_hysteresis_stop; res.n_vf_gpu_syncs = mstream->vf_gpu_syncs;
        res.t_vf_gpu_wait_us = mstream->vf_gpu_wait_us; res.n_vf_drains = mstream->vf_drains;
        std::lock_guard<std::mutex> lk(mstream->vf_mtx);
        res.n_vf_queued = (int64_t) mstream->vf_queue.size() + (int64_t) mstream->vf_cb_queue.size() + mstream->vf_busy;
        res.n_vf_copies_outside = mstream->vf_copies_outside;
    }
    res.dec.t_ne_drain_us    = mstream->t_ne_drain_us;
    res.dec.n_bh_words       = mstream->n_bh_words;     // engine-wide
    res.dec.n_turns          = mstream->n_turns;
    res.dec.n_bh_turns       = mstream->n_bh_turns;
    res.dec.n_gi_wait        = mstream->n_gi_wait;
    res.dec.t_gi_wait_us     = mstream->t_gi_wait_us;
    // The word-start op (engine-wide)
    res.first_streamed = mstream->il_of_moe.empty() ? -1 : mstream->il_of_moe.front();
    res.ws_D           = mstream->ws_D;
    res.n_ws_words     = mstream->n_ws_words;
    res.n_ws_skip      = mstream->n_ws_skip;
    for (int a = 0; a < 9; a++) {
        for (int b = 0; b < 9; b++) {
            res.n_ws_guess[a][b] = mstream->n_ws_guess[a][b];
            res.n_ws_late_d[b]  += mstream->n_ws_late_d[a][b];   // the source slots, summed over the sources
        }
        res.n_ws_post     += mstream->n_ws_post[a];
        res.n_ws_done     += mstream->n_ws_done[a];
        res.n_ws_late     += mstream->n_ws_late[a];
        res.t_ws_delay_us += mstream->t_ws_delay_us[a];
    }

    if (out) {
        *out = res;
    }

    return true;
}

// The router parameters of the streamed layers as a json object, for the recording's meta.json: the
// grouping, the weight scale and the selection bias (exp_probs_b) of every streamed layer, which is what
// turns the recorded raw logits back into the engine's ranking. The engine itself does not see the model.
static std::string moe_stream_router_json(const llama_model & model, const llama_moe_stream & ms) {
    const auto & hp = model.hparams;

    std::string js = "{";
    js += format("\"gating\": \"%s\", \"gating_id\": %u, \"n_expert_groups\": %u, \"n_group_used\": %u, "
                 "\"weights_norm\": %s, \"weights_scale\": %.9g, \"n_layer_dense_lead\": %u, \"bias\": [",
                 hp.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID ? "sigmoid" :
                 hp.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX ? "softmax" : "other",
                 (unsigned) hp.expert_gating_func, (unsigned) hp.n_expert_groups, (unsigned) hp.n_group_used,
                 hp.expert_weights_norm ? "true" : "false", (double) hp.expert_weights_scale,
                 (unsigned) hp.n_layer_dense_lead);

    bool first = true;

    for (const int32_t il : ms.il_of_moe) {
        const ggml_tensor * b = (size_t) il < model.layers.size() ? model.layers[il].ffn_exp_probs_b : nullptr;

        js += first ? "" : ", ";
        first = false;

        if (!b || b->type != GGML_TYPE_F32) {
            js += "null";
            continue;
        }

        std::vector<float> v((size_t) ggml_nelements(b));

        ggml_backend_tensor_get(b, v.data(), 0, v.size()*sizeof(float));

        js += "[";
        for (size_t i = 0; i < v.size(); i++) {
            js += format("%s%.9g", i ? ", " : "", (double) v[i]);
        }
        js += "]";
    }

    js += "]}";

    return js;
}

void llama_moe_stream_print_stats(const llama_context * ctx) {
    if (ctx && ctx->ctx_moe_stream()) {
        ctx->ctx_moe_stream()->print_stats();

        if (ctx->ctx_moe_stream()->rec_on()) {
            ctx->ctx_moe_stream()->rec_write(moe_stream_router_json(ctx->get_model(), *ctx->ctx_moe_stream()));
        }
    }
}

void llama_moe_stream_idle(const llama_context * ctx) {
    if (ctx && ctx->ctx_moe_stream()) {
        ctx->ctx_moe_stream()->el_reply_end();
    }
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: return ctx->decode(*batch);
    }
    return -1;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}

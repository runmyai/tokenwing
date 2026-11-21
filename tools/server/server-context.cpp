#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"

// TEST ONLY: MOE_STREAM_TEST_REQ_TIMES=1 - the server's steps of a request as "TT <us> <tag> <arg>" lines on ggml_time_us (the
// prompt clock of slot.stats), next to the engine's (src/llama-moe-stream.h moe_tt). g_srv_tt: a prompt is between its launch and its first
// token (the decode-path marks print only then). Off: one test of a static bool per step
static bool g_srv_tt = false;
static bool srv_tt_on() {
    static const bool on = getenv("MOE_STREAM_TEST_REQ_TIMES") && atoi(getenv("MOE_STREAM_TEST_REQ_TIMES")) != 0;
    return on;
}
static void srv_tt(const char * tag, long long arg = 0) {
    if (srv_tt_on()) {
        fprintf(stderr, "TT %lld %s %lld\n", (long long) ggml_time_us(), tag, arg);
    }
}
#define SRV_TT(tag, arg) do { if (g_srv_tt) { srv_tt(tag, arg); } } while (0)
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "gguf.h"

#include <algorithm>
#include <cstddef>
#include <cinttypes>
#include <exception>
#include <memory>
#include <filesystem>
#include <random>
#include <utility>
#include <fstream>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// a checkpoint restore dropped tokens the target had accepted - re-accept them rather than verify again
static std::vector<llama_token> server_accept_replay(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    for (size_t i = 0; i < draft.size(); ++i) {
        // the token is discarded - the call is what advances the sampler over this position
        common_sampler_sample(smpl, ctx, idxs[i]);
        common_sampler_accept(smpl, draft[i], true);
        result.push_back(draft[i]);
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = is_replay || dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept && (is_replay || !llama_vocab_is_eog(vocab, draft[i]))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const bool is_replay_target = is_replay && i + 1 == draft.size();
            common_sampler_accept(smpl, draft[i], is_replay_target);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
// llama_decode returned 3 - a layer-major call of the engine failed after it had written over the sliding window's cells from
// before the call, so the sequence was cut at the call's start; the slots that were processing a prompt start it again (thrown by
// decode(), caught by the batch loop: nothing of this batch is processed further)
struct server_reprocess {};

enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    common_batch view; // the rendered sub-batch [off, off + n_tokens), see render()

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
        bool is_prompt; // for stats tracking
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    bool has_embd = false;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    void init(llama_context * ctx, int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        view = common_batch(ctx);
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output, is_prompt });
        return true;
    }

    bool add(int32_t id_slot, const std::vector<float> & embd_in, llama_pos pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, pos, output, is_prompt });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        embd.clear();
        view.clear();
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        has_embd          = false;
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    // render the sub-batch [off, off + n_tokens) into view, index i in view is index off + i here
    void render(int32_t off, int32_t n_tokens) {
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        view.clear();
        for (int32_t i = off; i < off + n_tokens; i++) {
            const auto & t = tokens[i];
            if (has_embd) {
                // text embeddings broadcast the same position across the M-RoPE sections
                const llama_pos pos[GGML_MROPE_SECTIONS] = { t.pos, t.pos, t.pos, 0 };
                view.add_embd({ embd.data() + (size_t) i * n_embd, 1, (size_t) n_embd }, pos, t.id_slot, t.output);
            } else {
                view.add(t.token, t.pos, t.id_slot, t.output);
            }
        }
    }
};

// --cache-disk: a model file's identity - the sum of its files' sizes (all shards of a split model, as the
// engine's llama_moe_stream_model_bytes() counts them; 0 = unknown, a size could not be read) and a hash of the first file's
// GGUF header (every byte before the tensor data). Never throws.
static void server_cache_disk_model_id(const std::string & path, uint64_t & bytes, uint64_t & hash) {
    bytes = 0;
    hash  = 0;
    if (path.empty()) {
        return;
    }
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gp);
    int n_split = 1;
    if (gctx) {
        const int64_t k = gguf_find_key(gctx, "split.count");
        if (k >= 0 && gguf_get_kv_type(gctx, k) == GGUF_TYPE_UINT16) {
            n_split = std::max<int>(1, gguf_get_val_u16(gctx, k));
        }
        const size_t n_hdr = gguf_get_data_offset(gctx);
        std::vector<uint8_t> hdr(n_hdr);
        if (FILE * f = fopen(path.c_str(), "rb")) {
            if (fread(hdr.data(), 1, n_hdr, f) == n_hdr) {
                hash = server_cache_disk::fnv1a(hdr.data(), hdr.size());
            }
            fclose(f);
        }
        gguf_free(gctx);
    }
    std::vector<std::string> files;
    if (n_split > 1) {
        char prefix[4096];
        if (llama_split_prefix(prefix, sizeof(prefix), path.c_str(), 0, n_split) > 0) {
            for (int i = 0; i < n_split; i++) {
                char shard[4096];
                if (llama_split_path(shard, sizeof(shard), prefix, i, n_split) > 0) {
                    files.push_back(shard);
                }
            }
        }
        if ((int) files.size() != n_split) {
            files.clear();   // the shard names could not be made: unknown
        }
    } else {
        files.push_back(path);
    }
    for (const auto & fpath : files) {
        std::error_code ec;
        const uint64_t sz = std::filesystem::file_size(fpath, ec);
        if (ec) {
            bytes = 0;   // unknown, not an error value added to the fingerprint
            return;
        }
        bytes += sz;
    }
}

// --cache-disk: what a saved state's bytes and meaning depend on: the model's identity (above), a draft model's
// identity and KV types, the control vectors, the context size, the KV cache types, the sequences, a draft context, the
// flash-attention setting as resolved by the context, --ctx-checkpoints and llama.cpp's state format. LoRA adapters can
// change per request, so their set is kept per entry (server_cache_disk::lora_hash_of), not here.
static server_cache_disk::config server_cache_disk_config(const common_params & params, llama_context * ctx, llama_context * ctx_dft) {
    const bool has_draft = ctx_dft != nullptr;
    server_cache_disk::config c;
    server_cache_disk_model_id(params.model.path, c.model_bytes, c.gguf_hash);
    if (c.model_bytes == 0) {
        SRV_WRN("%s", "cache-disk: the model files' size could not be read; the fingerprint uses the GGUF header only\n");
    }
    if (has_draft) {
        server_cache_disk_model_id(params.speculative.draft.mparams.path, c.draft_bytes, c.draft_hash);
        c.type_k_dft = (int32_t) params.speculative.draft.cache_type_k;
        c.type_v_dft = (int32_t) params.speculative.draft.cache_type_v;
    }
    for (const auto & cv : params.control_vectors) {
        if (c.ctrl_hash == 0) {
            c.ctrl_hash = 1469598103934665603ull;
        }
        c.ctrl_hash = server_cache_disk::fnv1a(cv.fname.data(), cv.fname.size(), c.ctrl_hash);
        c.ctrl_hash = server_cache_disk::fnv1a(&cv.strength, sizeof(cv.strength), c.ctrl_hash);
    }
    if (c.ctrl_hash != 0) {
        c.ctrl_hash = server_cache_disk::fnv1a(&params.control_vector_layer_start, sizeof(int32_t), c.ctrl_hash);
        c.ctrl_hash = server_cache_disk::fnv1a(&params.control_vector_layer_end,   sizeof(int32_t), c.ctrl_hash);
    }
    {
        // RoPE / YaRN settings and --override-kv change what the model computes, so what a state means
        uint64_t h = 1469598103934665603ull;
        const float   fr[6] = { params.rope_freq_base, params.rope_freq_scale, params.yarn_ext_factor, params.yarn_attn_factor,
                                params.yarn_beta_fast, params.yarn_beta_slow };
        const int32_t ir[2] = { params.yarn_orig_ctx, (int32_t) params.rope_scaling_type };
        h = server_cache_disk::fnv1a(fr, sizeof(fr), h);
        h = server_cache_disk::fnv1a(ir, sizeof(ir), h);
        for (const auto & kv : params.kv_overrides) {
            if (kv.key[0] == 0) {
                continue;   // the list's terminator
            }
            h = server_cache_disk::fnv1a(kv.key, strnlen(kv.key, sizeof(kv.key)), h);
            h = server_cache_disk::fnv1a(&kv.tag, sizeof(kv.tag), h);
            switch (kv.tag) {   // only the union member the tag names (a bool sets 1 of its 8 bytes)
                case LLAMA_KV_OVERRIDE_TYPE_INT:   h = server_cache_disk::fnv1a(&kv.val_i64,  sizeof(kv.val_i64),  h); break;
                case LLAMA_KV_OVERRIDE_TYPE_FLOAT: h = server_cache_disk::fnv1a(&kv.val_f64,  sizeof(kv.val_f64),  h); break;
                case LLAMA_KV_OVERRIDE_TYPE_BOOL:  h = server_cache_disk::fnv1a(&kv.val_bool, sizeof(kv.val_bool), h); break;
                case LLAMA_KV_OVERRIDE_TYPE_STR:   h = server_cache_disk::fnv1a(kv.val_str, strnlen(kv.val_str, sizeof(kv.val_str)), h); break;
            }
        }
        c.ovr_hash = h;
    }

    c.n_ctx      = llama_n_ctx(ctx);
    c.type_k     = (int32_t) params.cache_type_k;
    c.type_v     = (int32_t) params.cache_type_v;
    c.n_seq      = llama_n_seq_max(ctx);
    c.has_draft  = has_draft ? 1 : 0;
    c.flash_attn = llama_flash_attn_enabled(ctx) ? 1 : 0;
    c.flash_attn_dft = has_draft ? (llama_flash_attn_enabled(ctx_dft) ? 1 : 0) : -1;
    c.n_ckpt_max = params.n_ctx_checkpoints;
    c.state_ver  = LLAMA_STATE_SEQ_VERSION;
    c.kv_unified = params.kv_unified ? 1 : 0;   // an SWA model's partial state depends on these three
    c.swa_full   = params.swa_full ? 1 : 0;
    c.n_ubatch   = llama_n_ubatch(ctx);
    // The rotation the states are saved with (our own fix of upstream #28498); -1 = a memory kind llama_get_kv_rot does not know
    {
        uint32_t k = 0, v = 0, kd = 0, vd = 0;
        if (llama_get_kv_rot(ctx, &k, &v, &kd, &vd)) {
            c.rot_k = (int32_t) k; c.rot_v = (int32_t) v; c.rot_k_def = (int32_t) kd; c.rot_v_def = (int32_t) vd;
        }
        if (ctx_dft) {
            if (llama_get_kv_rot(ctx_dft, &k, &v, &kd, &vd)) {
                c.rot_k_dft = (int32_t) k; c.rot_v_dft = (int32_t) v; c.rot_k_dft_def = (int32_t) kd; c.rot_v_dft_def = (int32_t) vd;
            }
        } else {
            c.rot_k_dft = c.rot_v_dft = c.rot_k_dft_def = c.rot_v_dft_def = 0;   // no draft context: nothing rotated
        }
    }
    return c;
}

// ---------------------------------------------------------------------------------------------------------------------
// The streaming engine's figures as JSON: a request's timings (timings.moe_stream) and GET /moe-stream/stats

using moe_layers_t = std::vector<llama_moe_stream_layer_stats>;

// the counters of every streamed layer, in layer order
static moe_layers_t moe_stream_layers_now(const llama_context * ctx) {
    const int32_t n = llama_moe_stream_get_layer_stats(ctx, nullptr, 0);
    moe_layers_t v((size_t) std::max<int32_t>(n, 0));
    if (n > 0) {
        llama_moe_stream_get_layer_stats(ctx, v.data(), n);
    }
    return v;
}

static double moe_pct(int64_t a, int64_t b) {
    return b > 0 ? 100.0*(double) a/(double) b : 0.0;
}

// The figures of one stretch of work: the engine's counters `ms` minus `b`, the ones at its start (all zero = since the model
// was loaded), per layer `ln` minus `lb`. `timing` = --moe-stream-timing.
static json moe_stream_figures_json(const llama_moe_stream_stats & ms, const llama_moe_stream_stats & b,
        const moe_layers_t & ln, const moe_layers_t & lb, bool timing) {
    // the figures of the prompt or of the reply
    const auto part = [](const auto & now, const auto & base, bool reply) {
        const int64_t n_tok  = now.n_tokens   - base.n_tokens;
        const int64_t n_hit  = now.n_hit      - base.n_hit;
        const int64_t n_miss = now.n_miss     - base.n_miss;
        const int64_t n_vram = now.n_vram_hit - base.n_vram_hit;
        const int64_t n_use  = n_hit + n_miss + n_vram;
        const double  mib    = (double) (now.bytes_read  - base.bytes_read)/1024.0/1024.0;
        const double  stall  = (double) (now.t_stall_us  - base.t_stall_us)/1000.0;
        // The guesses: reads started for experts a layer was expected to need. In time = landed before the layer asked, late = the
        // layer asked while the read was in flight (it waited), wasted = left RAM unused, cancelled = stopped before they ran. A
        // missed expert is one the layer needed that no guess had brought: it is read on demand
        const int64_t n_cov    = now.n_hit_pref  - base.n_hit_pref;
        const int64_t n_late   = now.n_pref_late - base.n_pref_late;
        const int64_t n_cancel = (now.n_pref_cancel - base.n_pref_cancel) + (now.n_pl_stale_cancel - base.n_pl_stale_cancel);
        json guesses = {
            { "issued",        now.n_pref - base.n_pref },
            { "in_time",       n_cov - n_late },
            { "late",          n_late },
            { "late_wait_ms",  reply ? json((double) (now.t_cause_us[0] - base.t_cause_us[0])/1000.0) : json() },
            { "missed",        n_miss },
            { "wasted",        now.n_pref_waste - base.n_pref_waste },
            { "cancelled",     n_cancel },
        };
        if (reply) {   // the engine tells why a reply word's expert was read on demand, and the 2x2 guesser census
            guesses["missed_by_cause"] = {
                { "no_guess",       now.n_cause[1] - base.n_cause[1] },
                { "not_named",      now.n_cause[3] - base.n_cause[3] },
                { "named_not_read", now.n_cause[2] - base.n_cause[2] },
            };
            guesses["mem_right"]  = now.n_gg_mem_right  - base.n_gg_mem_right;
            guesses["mem_wrong"]  = now.n_gg_mem_wrong  - base.n_gg_mem_wrong;
            guesses["ssd_right"]  = now.n_gg_ssd_right  - base.n_gg_ssd_right;
            guesses["ssd_wrong"]  = now.n_gg_ssd_wrong  - base.n_gg_ssd_wrong;
            guesses["skip_floor"] = now.n_gg_skip_floor - base.n_gg_skip_floor;
            guesses["skip_vic"]   = now.n_gg_skip_vic   - base.n_gg_skip_vic;
            guesses["defer_q"]    = now.n_gg_defer_q    - base.n_gg_defer_q;
            guesses["pn_skip"]    = now.n_gg_pn_skip    - base.n_gg_pn_skip;
        }
        return json {
            { "tokens",             n_tok },
            { "hit_rate_percent",   n_hit + n_miss > 0 ? 100.0*(double) n_hit/(double) (n_hit + n_miss) : 0.0 },
            { "misses",             n_miss },
            { "mib_read",           mib },
            { "stall_ms",           stall },
            { "misses_per_token",   n_tok > 0 ? (double) n_miss/(double) n_tok : 0.0 },
            { "mib_per_token",      n_tok > 0 ? mib/(double) n_tok : 0.0 },
            { "stall_ms_per_token", n_tok > 0 ? stall/(double) n_tok : 0.0 },
            { "expert_uses",        n_use },
            { "ram_hit_percent",    moe_pct(n_hit,  n_use) },
            { "vram_hit_percent",   moe_pct(n_vram, n_use) },
            { "miss_percent",       moe_pct(n_miss, n_use) },
            { "guesses",            guesses },
        };
    };

    const int64_t n_hit  = ms.n_hit  - b.n_hit;
    const int64_t n_miss = ms.n_miss - b.n_miss;
    const int64_t n_vram = ms.n_vram_hit - b.n_vram_hit;
    const int64_t n_use  = n_hit + n_miss + n_vram;
    const int64_t n_cov    = ms.n_hit_pref  - b.n_hit_pref;
    const int64_t n_late   = ms.n_pref_late - b.n_pref_late;
    const int64_t n_cancel = (ms.pre.n_pref_cancel - b.pre.n_pref_cancel) + (ms.dec.n_pref_cancel - b.dec.n_pref_cancel) +
                             (ms.pre.n_pl_stale_cancel - b.pre.n_pl_stale_cancel) + (ms.dec.n_pl_stale_cancel - b.dec.n_pl_stale_cancel);

    json out = {
        { "layers",           ms.n_layers },
        { "slots",            ms.n_slots },
        { "hit_rate_percent", n_hit + n_miss > 0 ? 100.0*(double) n_hit/(double) (n_hit + n_miss) : 0.0 },
        { "prompt",           part(ms.pre, b.pre, false) },
        { "reply",            part(ms.dec, b.dec, true) },
        { "prompt_tokens",    ms.pre.n_tokens - b.pre.n_tokens },
        { "words",            ms.dec.n_tokens - b.dec.n_tokens },
        { "expert_uses",      n_use },
        { "ram_hit_percent",  moe_pct(n_hit,  n_use) },
        { "vram_hit_percent", moe_pct(n_vram, n_use) },
        { "miss_percent",     moe_pct(n_miss, n_use) },
        { "misses",           n_miss },
        { "mib_read",         (double) (ms.bytes_read - b.bytes_read)/1024.0/1024.0 },
        { "ssd_wait_ms",      (double) (ms.t_stall_us - b.t_stall_us)/1000.0 },
        // prompt and reply together; prompt and reply each have their own (prompt.guesses, reply.guesses)
        { "guesses", {
            { "issued",       ms.n_pref - b.n_pref },
            { "in_time",      n_cov - n_late },
            { "late",         n_late },
            { "late_wait_ms", (double) (ms.dec.t_cause_us[0] - b.dec.t_cause_us[0])/1000.0 },   // reply words only
            { "missed",       n_miss },
            { "wasted",       ms.n_pref_waste - b.n_pref_waste },
            { "cancelled",    n_cancel },
            { "mem_right",    ms.dec.n_gg_mem_right  - b.dec.n_gg_mem_right },
            { "mem_wrong",    ms.dec.n_gg_mem_wrong  - b.dec.n_gg_mem_wrong },
            { "ssd_right",    ms.dec.n_gg_ssd_right  - b.dec.n_gg_ssd_right },
            { "ssd_wrong",    ms.dec.n_gg_ssd_wrong  - b.dec.n_gg_ssd_wrong },
            { "skip_floor",   ms.dec.n_gg_skip_floor - b.dec.n_gg_skip_floor },
            { "skip_vic",     ms.dec.n_gg_skip_vic   - b.dec.n_gg_skip_vic },
            { "defer_q",      ms.dec.n_gg_defer_q    - b.dec.n_gg_defer_q },
            { "pn_skip",      ms.dec.n_gg_pn_skip    - b.dec.n_gg_pn_skip },
        } },
        { "vram", {
            { "follow_swaps",     ms.n_vf_swaps  - b.n_vf_swaps },
            { "follow_copies",    ms.n_vf_copies - b.n_vf_copies },
            { "copybacks",        (ms.pre.n_copyback - b.pre.n_copyback) + (ms.dec.n_copyback - b.dec.n_copyback) },
            { "copyback_late",    (ms.pre.n_cb_late  - b.pre.n_cb_late)  + (ms.dec.n_cb_late  - b.dec.n_cb_late) },
            { "copyback_wait_ms", (double) ((ms.pre.t_cb_wait_us - b.pre.t_cb_wait_us) + (ms.dec.t_cb_wait_us - b.dec.t_cb_wait_us))/1000.0 },
        } },
    };

    // per streamed layer
    json per_layer = json::array();
    for (size_t i = 0; i < ln.size(); i++) {
        const llama_moe_stream_layer_stats & n = ln[i];
        const llama_moe_stream_layer_stats * p = i < lb.size() && lb[i].il == n.il ? &lb[i] : nullptr;
        const auto d = [&](auto get) -> int64_t { return get(n) - (p ? get(*p) : 0); };
        const int64_t h  = d([](const auto & s) -> int64_t { return s.pre.n_hit      + s.dec.n_hit; });
        const int64_t m  = d([](const auto & s) -> int64_t { return s.pre.n_miss     + s.dec.n_miss; });
        const int64_t v  = d([](const auto & s) -> int64_t { return s.pre.n_vram_hit + s.dec.n_vram_hit; });
        const int64_t st = d([](const auto & s) -> int64_t { return s.pre.t_stall_us + s.dec.t_stall_us; });
        per_layer.push_back({
            { "layer",        n.il },
            { "hits",         h },
            { "misses",       m },
            { "vram_hits",    v },
            { "stall_ms",     (double) st/1000.0 },
            { "ram_slots",    n.ram_slots },
            { "vram_experts", n.vram_experts },
            { "reply", {
                { "hits",      d([](const auto & s) -> int64_t { return s.dec.n_hit; }) },
                { "misses",    d([](const auto & s) -> int64_t { return s.dec.n_miss; }) },
                { "vram_hits", d([](const auto & s) -> int64_t { return s.dec.n_vram_hit; }) },
                { "stall_ms",  (double) d([](const auto & s) -> int64_t { return s.dec.t_stall_us; })/1000.0 },
            } },
        });
    }
    out["per_layer"] = per_layer;

    // where the time of a reply word goes (--moe-stream-timing 1)
    if (timing) {
        const int64_t words = ms.dec.n_tokens - b.dec.n_tokens;
        const int64_t calls = ms.dec.n_tm_calls - b.dec.n_tm_calls;
        const double  wall  = (double) (ms.dec.t_tm_wall_us - b.dec.t_tm_wall_us)/1000.0;
        const double  host  = (double) (ms.dec.t_tm_host_us - b.dec.t_tm_host_us)/1000.0;
        const double  dev   = (double) (ms.dec.t_tm_dev_us  - b.dec.t_tm_dev_us)/1000.0;
        const double  copy  = (double) (ms.dec.t_tm_copy_us - b.dec.t_tm_copy_us)/1000.0;
        const double  sync  = (double) (ms.dec.t_tm_sync_us - b.dec.t_tm_sync_us)/1000.0;
        const double  ssd   = (double) (ms.dec.t_stall_us   - b.dec.t_stall_us)/1000.0;
        // The host backend's compute includes the time the engine waits for reads (its remap op runs there), so the CPU work is the
        // rest of it. The device's kernels run beside the host's work: gpu_ms is device busy time, not time added to the word
        const double  cpu   = std::max(0.0, host - ssd);
        const double  rest  = wall - host - copy - sync;
        const double  w     = words > 0 ? (double) words : 0.0;
        const auto    pw    = [w](double ms_total) { return w > 0.0 ? ms_total/w : 0.0; };
        out["time_per_word"] = {
            { "words",       words },
            { "calls",       calls },
            { "ms_per_word", pw(wall) },
            { "ssd_wait_ms", pw(ssd) },
            { "cpu_ms",      pw(cpu) },
            { "gpu_ms",      pw(dev) },
            { "handoff_ms",  pw(copy + sync) },
            { "rest_ms",     pw(rest) },
            { "total_ms", {
                { "all",     wall },
                { "ssd_wait", ssd },
                { "cpu",     cpu },
                { "gpu",     dev },
                { "handoff", copy + sync },
                { "rest",    rest },
            } },
            { "handoff_copies_ms", pw(copy) },
            { "handoff_syncs_ms",  pw(sync) },
        };
    }
    return out;
}

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // The cuts of the call being decoded (position, its n_tokens, whether a checkpoint is taken there)
    std::vector<llama_pos> cc_pos;
    std::vector<int64_t>   cc_ntok;
    std::vector<uint8_t>   cc_keep;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;

    // draft candidates per token in spec_draft; only draft-simple and draft-mtp fill it
    std::vector<std::vector<llama_token_data>> spec_draft_q;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;
    std::mt19937 spec_synth_rng;

    // TODO: move members that belong to the task (such as `generated`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    common_chat_input generated;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    // --cache-disk: the chat (X-Conversation-Id) of the last task this slot ran, "" = none; it marks the slot's
    // content and the RAM-cache state saved from it, so POST /cache-disk/erase finds them
    std::string chat_id;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }
        cur->conv_id = chat_id;

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id, &chat_id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
        chat_id.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    server_slot_stats stats;

    // The streaming engine's counters when the request that is running now started, so the
    // per-request line is a difference and not everything since the model was loaded
    llama_moe_stream_stats moe_base{};
    std::vector<llama_moe_stream_layer_stats> moe_base_layers;   // the same, per streamed layer
    bool moe_timing = false;   // --moe-stream-timing: the time-per-word figures are in the request's timings

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()
    std::function<void(server_slot &)>       callback_on_end;   // called first on every release, task still set

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;

        last_nl_pos    = 0;
        generated      = {};
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();

        n_predict_max = -1;

        llama_set_sampler(ctx_tgt, id, nullptr);
        srv_tt("S release_sampler", id);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    bool can_split() const {
        GGML_ASSERT(task);
        // MTP supports splitting - uses task->need_embd() not need_embd()
        if (!task->need_embd()) {
            return true;
        }
        // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
        if (!llama_get_memory(ctx_tgt)) {
            return false;
        }
        // context can be chunked/split if the pooling type is LAST
        const auto pooling = llama_pooling_type(ctx_tgt);
        if (pooling == LLAMA_POOLING_TYPE_LAST) {
            return true;
        }
        // causal rerankers read the last token and have a KV cache, so they can also be chunked/split.
        if (pooling == LLAMA_POOLING_TYPE_RANK && llama_get_causal_attn(ctx_tgt)) {
            return true;
        }
        return false;
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    // at temp 0 both p and q are point masses, so rejection is the same as sample-and-match
    bool use_spec_rejection() const {
        return task && task->params.sampling.temp > 0.0f;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                add_ok &= batch.add(id, inp_embd, prompt.tokens.pos_next(), true, false);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            // nor of the stateless slots - no task ever reuses it
            if (task->is_child() || task->is_stateless()) {
                prompt_clear();
            }

            if (callback_on_end) {
                callback_on_end(*this);
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    // The streaming engine's counters for this request (the figures print_timings() logs, prompt and reply apart), for the
    // timings of the final response; null when --moe-stream is off. Read once, when the request ends.
    json moe_stream_json() const {
        llama_moe_stream_stats ms;
        if (!llama_moe_stream_get_stats(ctx_tgt, &ms)) {
            return json();
        }
        return moe_stream_figures_json(ms, moe_base, moe_stream_layers_now(ctx_tgt), moe_base_layers, moe_timing);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        if (smpl) {   // where this request's words were picked (with --backend-sampling: on the GPU)
            int64_t n_gpu = 0, n_cpu = 0;
            common_sampler_pick_counts(smpl.get(), &n_gpu, &n_cpu);
            SLT_INF(*this, "    picked words = %lld on the GPU (backend sampling), %lld on the CPU\n", (long long) n_gpu, (long long) n_cpu);
        }

        // The streaming engine's counters, per request. The tools print them at exit (common_perf_print),
        // but a server runs for days, so the only place they are useful is next to the timings of the request
        // they belong to.
        // The difference between this request's counters and the ones it started from, and the
        // denominators of the request itself (its prompt tokens and its generated tokens), so the
        // figures are per request and the decode ones are per generated token. The hit rate is
        // n_hit/(n_hit+n_miss) of the request, the same as common/sampling.cpp's.
        {
            llama_moe_stream_stats ms;
            const llama_moe_stream_stats & b = moe_base;

            if (llama_moe_stream_get_stats(ctx_tgt, &ms)) {
                const int64_t n_hit      = ms.n_hit      - b.n_hit;
                const int64_t n_miss     = ms.n_miss     - b.n_miss;
                const int64_t n_dec_tok  = ms.dec.n_tokens - b.dec.n_tokens;
                const int64_t n_pre_tok  = ms.pre.n_tokens - b.pre.n_tokens;
                const int64_t dec_bytes  = ms.dec.bytes_read - b.dec.bytes_read;
                const int64_t dec_stall  = ms.dec.t_stall_us - b.dec.t_stall_us;
                const int64_t pre_stall  = ms.pre.t_stall_us - b.pre.t_stall_us;

                SLT_INF(*this,
                        "      moe stream: %d layers x %d slots, this request %lld prompt + %lld decode tokens, "
                        "hit rate %.2f%%, %.2f misses/decode token, %.2f MiB/decode token read, "
                        "stall %.2f ms/decode token, %.2f ms/prompt token\n",
                        ms.n_layers, ms.n_slots,
                        (long long) n_pre_tok, (long long) n_dec_tok,
                        n_hit + n_miss > 0 ? 100.0*(double) n_hit/(double) (n_hit + n_miss) : 0.0,
                        n_dec_tok > 0 ? (double) n_miss/(double) n_dec_tok : 0.0,
                        n_dec_tok > 0 ? (double) dec_bytes/(double) n_dec_tok/1024.0/1024.0 : 0.0,
                        n_dec_tok > 0 ? (double) dec_stall/(double) n_dec_tok/1000.0 : 0.0,
                        n_pre_tok > 0 ? (double) pre_stall/(double) n_pre_tok/1000.0 : 0.0);

                {   // the margin to the process's own memory cap, after every request (Linux cgroup v1 or v2)
                    int64_t cmx = 0, cpk = 0, cmg = 0;
                    if (llama_moe_stream_cap_margin(&cmx, &cpk, &cmg)) {
                        // the peak counts file cache too (a model another program read earlier); with 16 MiB or more of it the use now without it
                        int64_t cun = 0, cca = 0;
                        if (llama_moe_stream_cap_now(&cun, &cca) && cca >= ((int64_t) 16 << 20)) {
                            SLT_INF(*this, "      moe stream: process memory cap %.0f MiB, peak use %.0f MiB, margin to the cap %.0f MiB; the peak counts file cache: "
                                    "in use now %.0f MiB without %.0f MiB of file cache the kernel can free, room now %.0f MiB\n",
                                    (double) cmx/1048576.0, (double) cpk/1048576.0, (double) cmg/1048576.0,
                                    (double) cun/1048576.0, (double) cca/1048576.0, (double) (cmx - cun)/1048576.0);
                        } else {
                            SLT_INF(*this, "      moe stream: process memory cap %.0f MiB, peak use %.0f MiB, margin to the cap %.0f MiB\n",
                                    (double) cmx/1048576.0, (double) cpk/1048576.0, (double) cmg/1048576.0);
                        }
                    }
                }

                // The same causes and read-backs as common/sampling.cpp's run line, for this request's decode words
                if (n_dec_tok > 0) {
                    const double nw = (double) n_dec_tok;
                    double nc[4], tc[4];
                    for (int i = 0; i < 4; i++) {
                        nc[i] = (double) (ms.dec.n_cause[i]    - b.dec.n_cause[i])/nw;
                        tc[i] = (double) (ms.dec.t_cause_us[i] - b.dec.t_cause_us[i])/1000.0/nw;
                    }
                    SLT_INF(*this,
                            "      moe stream: causes [decode] per word: late %.2f (%.2f ms), no-guess %.2f (%.2f ms), "
                            "named-not-read %.2f (%.2f ms), not-named %.2f (%.2f ms); read-backs <=8: %.2f (by guess %.2f), "
                            "<=64: %.2f (by guess %.2f); evicted by the prompt: %.2f\n",
                            nc[0], tc[0], nc[1], tc[1], nc[2], tc[2], nc[3], tc[3],
                            (double) (ms.dec.n_rb8        - b.dec.n_rb8)/nw,
                            (double) (ms.dec.n_rb8_guess  - b.dec.n_rb8_guess)/nw,
                            (double) (ms.dec.n_rb64       - b.dec.n_rb64)/nw,
                            (double) (ms.dec.n_rb64_guess - b.dec.n_rb64_guess)/nw,
                            (double) (ms.dec.n_rb_prompt  - b.dec.n_rb_prompt)/nw);

                    // The goal number and the waiting list, for this request's decode words
                    const int64_t g_c = ms.dec.n_goal_cand  - b.dec.n_goal_cand;
                    const int64_t g_r = ms.dec.n_goal_ready - b.dec.n_goal_ready;
                    const int64_t p_a = ms.dec.n_pl_add     - b.dec.n_pl_add;
                    const int64_t p_b = ms.dec.n_pl_bind    - b.dec.n_pl_bind;
                    const int64_t p_u = ms.dec.n_pl_used    - b.dec.n_pl_used;
                    const int64_t p_k = ms.dec.n_pl_bind_pick  - b.dec.n_pl_bind_pick;
                    const int64_t p_r = ms.dec.n_pl_bind_after - b.dec.n_pl_bind_after;
                    const int64_t p_c = ms.dec.n_pl_stale_cancel - b.dec.n_pl_stale_cancel;
                    const int64_t p_m = ms.dec.n_pl_stale_cancel_bytes - b.dec.n_pl_stale_cancel_bytes;
                    if (g_c > 0) {
                        SLT_INF(*this, "      moe stream: goal [decode]: %.2f%% of the 1-ahead guess's top-k READY when it was made\n",
                                100.0*(double) g_r/(double) g_c);
                    }
                    // The guess helper, for this request's decode words
                    const int64_t h_p = ms.dec.n_ne_post - b.dec.n_ne_post;
                    if (h_p > 0) {
                        const int64_t h_d = ms.dec.n_ne_done - b.dec.n_ne_done;
                        int64_t hh[100];
                        for (int i = 0; i < 100; i++) {
                            hh[i] = ms.dec.ne_delay_hist[i] - b.dec.ne_delay_hist[i];
                        }
                        std::string late_d;   // with lookahead, the columns skipped as too late by distance
                        for (int d = 1; d <= 8; d++) {
                            const int64_t v = ms.dec.n_ne_late_d[d] - b.dec.n_ne_late_d[d];
                            if (v > 0) {
                                late_d += string_format("%sd%d %.2f", late_d.empty() ? " (columns by distance: " : ", ", d, (double) v/nw);
                            }
                        }
                        late_d += late_d.empty() ? "" : ")";
                        SLT_INF(*this, "      moe stream: helper [decode] per word: posted %.2f, adds done %.2f, guess too late %.2f%s; "
                                "post -> adds done %.3f ms mean, p90 %s; word-end drain: %lld calls waited, %.3f ms\n",
                                (double) h_p/nw, (double) h_d/nw, (double) (ms.dec.n_ne_late - b.dec.n_ne_late)/nw, late_d.c_str(),
                                h_d > 0 ? (double) (ms.dec.t_ne_delay_us - b.dec.t_ne_delay_us)/1000.0/(double) h_d : 0.0,
                                common_moe_pl_p90(hh, h_d).c_str(),
                                (long long) (ms.dec.n_ne_drain_calls - b.dec.n_ne_drain_calls),
                                (double) (ms.dec.t_ne_drain_us - b.dec.t_ne_drain_us)/1000.0);
                    }
                    // The GPU producer's "router done", for this request's decode words
                    if (ms.n_rd_split - b.n_rd_split > 0 || ms.n_rd_uncovered - b.n_rd_uncovered > 0 ||
                            ms.n_ne_ready + ms.n_ne_event - b.n_ne_ready - b.n_ne_event > 0) {
                        SLT_INF(*this, "      moe stream: router done [decode] per word: CPU splits from the slot %.2f (number already "
                                "there %.2f, after the event %.2f; %.3f ms in the hook), not covered %.2f; helper: number already there "
                                "%.2f, after the event %.2f; layers left unmarked at graph builds %lld\n",
                                (double) (ms.n_rd_split - b.n_rd_split)/nw, (double) (ms.n_rd_ready - b.n_rd_ready)/nw,
                                (double) (ms.n_rd_event - b.n_rd_event)/nw, (double) (ms.t_rd_us - b.t_rd_us)/1000.0/nw,
                                (double) (ms.n_rd_uncovered - b.n_rd_uncovered)/nw, (double) (ms.n_ne_ready - b.n_ne_ready)/nw,
                                (double) (ms.n_ne_event - b.n_ne_event)/nw, (long long) (ms.n_rd_unfit - b.n_rd_unfit));
                    }
                    // VRAM follow (--moe-stream-vram-follow 1), for this request's reply words
                    if (ms.n_vf_words - b.n_vf_words > 0) {
                        const double xw = (double) (ms.n_vf_words - b.n_vf_words);
                        SLT_INF(*this, "      moe stream: VRAM follow [decode] per word: VRAM picks %.2f, swaps %.3f, copies %.3f, switches %.3f; "
                                "skipped (source loading %.3f, not in RAM %.3f, mark cap %.3f); %lld ticks, %lld switches put off, %lld drains; "
                                "mark mismatches %lld, marks at a prompt %lld (engine-wide); copies so far %lld (queued or running now "
                                "%lld, landed outside a call %lld)\n",
                                (double) (ms.dec.n_vram_hit - b.dec.n_vram_hit)/nw, (double) (ms.n_vf_swaps - b.n_vf_swaps)/xw,
                                (double) (ms.n_vf_copies - b.n_vf_copies)/xw, (double) (ms.n_vf_switches - b.n_vf_switches)/xw,
                                (double) (ms.n_vf_skip_loading - b.n_vf_skip_loading)/xw,
                                (double) (ms.n_vf_skip_notram - b.n_vf_skip_notram)/xw,
                                (double) (ms.n_vf_skip_marks - b.n_vf_skip_marks)/xw, (long long) (ms.n_vf_ticks - b.n_vf_ticks),
                                (long long) (ms.n_vf_switch_wait - b.n_vf_switch_wait), (long long) (ms.n_vf_drains - b.n_vf_drains),
                                (long long) ms.n_vf_mark_mismatch,
                                (long long) ms.n_vf_marked_prompt, (long long) ms.n_vf_copies, (long long) ms.n_vf_queued,
                                (long long) ms.n_vf_copies_outside);
                    }
                    // The ranks above k (from guess-top, or prefetch-m above k), for this request's decode words. Every
                    // removal reason, and "bound, not picked, not cancelled" (= bound - used - cancelled by drop stale)
                    const int64_t gx_a = ms.dec.n_gx_add - b.dec.n_gx_add;
                    if (gx_a > 0) {
                        const int64_t gx_b  = ms.dec.n_gx_bind      - b.dec.n_gx_bind;
                        const int64_t gx_u  = ms.dec.n_gx_used      - b.dec.n_gx_used;
                        const int64_t gx_ul = ms.dec.n_gx_used_late - b.dec.n_gx_used_late;
                        const int64_t gx_cx = ms.dec.n_gx_stale_cx  - b.dec.n_gx_stale_cx;
                        double rm[7];   // [0] picked [1] present [2] no room [3] word end [4] list full [5] aborted [6] stale
                        for (int i = 0; i < 7; i++) {
                            rm[i] = (double) (ms.dec.n_gx_rm[i] - b.dec.n_gx_rm[i])/nw;
                        }
                        SLT_INF(*this, "      moe stream: extra ranks [decode] (ranks above k, from guess-top or prefetch-m above k) per word: "
                                "added %.2f, bound by a reader %.2f, removed unread: stale %.2f, word end %.2f, no room %.2f, present %.2f, "
                                "list full %.2f, aborted %.2f; picked by the target: READY %.2f, loading %.2f, on the list (bound at the pick "
                                "%.2f, no room %.2f); bound and not picked %.2f (cancelled by drop stale %.2f; bound, not picked, not cancelled "
                                "%.2f); needed reads saved per extra read %.3f\n",
                                (double) gx_a/nw, (double) gx_b/nw, rm[6], rm[3], rm[2], rm[1], rm[4], rm[5],
                                (double) (gx_u - gx_ul)/nw, (double) gx_ul/nw,
                                (double) (ms.dec.n_gx_bind_pick - b.dec.n_gx_bind_pick)/nw, rm[0], (double) (gx_b - gx_u)/nw,
                                (double) gx_cx/nw, (double) (gx_b - gx_u - gx_cx)/nw,
                                gx_b > 0 ? (double) (gx_u - gx_ul)/(double) gx_b : 0.0);
                    }
                    // Lookahead N > 1, for this request's decode words (per distance; dropped entries; the mid-way rule)
                    if (ms.dec.n_la_named[2] - b.dec.n_la_named[2] > 0) {
                        std::string per_d;
                        for (int d = 1; d <= 8; d++) {
                            const int64_t nm = ms.dec.n_la_named[d] - b.dec.n_la_named[d];
                            if (nm == 0) {
                                continue;
                            }
                            const int64_t used = ms.dec.n_la_used[d] - b.dec.n_la_used[d];
                            const int64_t late = ms.dec.n_la_used_late[d] - b.dec.n_la_used_late[d];
                            per_d += string_format("%sd%d: named %.2f = new %.2f + vote %.2f + pace %.2f + pool %.2f; bound %.2f, at the "
                                    "pick %.2f, picked by the target %.2f (READY %.2f, loading %.2f)", per_d.empty() ? "" : "; ", d,
                                    (double) nm/nw, (double) (ms.dec.n_la_new[d] - b.dec.n_la_new[d])/nw,
                                    (double) (ms.dec.n_la_vote[d] - b.dec.n_la_vote[d])/nw,
                                    (double) (ms.dec.n_la_pace[d] - b.dec.n_la_pace[d])/nw,
                                    (double) (ms.dec.n_la_pool[d] - b.dec.n_la_pool[d])/nw,
                                    (double) (ms.dec.n_la_bind[d] - b.dec.n_la_bind[d])/nw,
                                    (double) (ms.dec.n_la_bind_pick[d] - b.dec.n_la_bind_pick[d])/nw,
                                    (double) used/nw, (double) (used - late)/nw, (double) late/nw);
                        }
                        SLT_INF(*this, "      moe stream: lookahead [decode] per word (new by the distance that added the entry, bound / "
                                "picked by its latest distance at the bind: only the totals over the distances add up): %s; dropped: "
                                "became %.2f, bound while dropped (its read starts moved back) %.2f, named again %.2f, removed at the "
                                "target's start %.2f; mid-way: started reads with a queued piece moved back %.2f; of all moved-back "
                                "reads: picked by the target while loading %.2f, cancelled at its start %.2f, named again %.2f\n",
                                per_d.c_str(),
                                (double) (ms.dec.n_dr_became - b.dec.n_dr_became)/nw, (double) (ms.dec.n_dr_bound - b.dec.n_dr_bound)/nw,
                                (double) (ms.dec.n_dr_again - b.dec.n_dr_again)/nw, (double) (ms.dec.n_dr_removed - b.dec.n_dr_removed)/nw,
                                (double) (ms.dec.n_mw_back - b.dec.n_mw_back)/nw, (double) (ms.dec.n_mw_promoted - b.dec.n_mw_promoted)/nw,
                                (double) (ms.dec.n_mw_cancelled - b.dec.n_mw_cancelled)/nw, (double) (ms.dec.n_mw_again - b.dec.n_mw_again)/nw);
                    }
                    // The in-graph form of CPU layers, for this request's decode words
                    if (ms.dec.n_ig_calls - b.dec.n_ig_calls > 0) {
                        const double ig_n = (double) (ms.dec.n_ig_calls - b.dec.n_ig_calls);
                        const double ig_t = (double) (ms.dec.t_ig_ab_us - b.dec.t_ig_ab_us)/1000.0;
                        const double ig_a = (double) (ms.dec.t_ig_a_us - b.dec.t_ig_a_us)/1000.0;   // op A itself
                        SLT_INF(*this, "      moe stream: in-graph [decode] per word: layers %.2f, op A -> op B %.3f ms per layer (%.3f ms "
                                "per word summed: how much earlier the needed reads were queued; of it op A itself %.3f ms per layer, "
                                "the work between A and B %.3f), needed reads queued by op A %.2f\n",
                                ig_n/nw, ig_t/ig_n, ig_t/nw, ig_a/ig_n, (ig_t - ig_a)/ig_n, (double) (ms.dec.n_ig_reads - b.dec.n_ig_reads)/nw);
                    }
                    // The per-projection ready flags, for this request's decode words
                    if (ms.dec.n_pr_split - b.dec.n_pr_split > 0) {
                        const double pr_g = (double) (ms.dec.n_pr_gap - b.dec.n_pr_gap);
                        SLT_INF(*this, "      moe stream: proj-ready [decode] per word: needed reads split (gate+up, then down) %.2f, "
                                "gate/up projections computed before their expert's down landed %.2f, waited-on experts whose down landed "
                                "after their gate+up %.2f (%.3f ms mean gap)\n",
                                (double) (ms.dec.n_pr_split - b.dec.n_pr_split)/nw, (double) (ms.dec.n_pr_early - b.dec.n_pr_early)/nw,
                                pr_g/nw, pr_g > 0 ? (double) (ms.dec.t_pr_gap_us - b.dec.t_pr_gap_us)/1000.0/pr_g : 0.0);
                    }
                    // Protect named, for this request's decode words
                    if (ms.dec.n_pn_refused - b.dec.n_pn_refused > 0) {
                        SLT_INF(*this, "      moe stream: protect-named [decode] per word: guess victim choices the rule changed (the "
                                "rule-free victim was a named expert) %.2f, of them with no other victim (the entry skipped for no "
                                "room) %.2f; experts kept from eviction %.2f, of them picked by their layer (a read saved) %.2f, not "
                                "picked %.2f\n",
                                (double) (ms.dec.n_pn_refused - b.dec.n_pn_refused)/nw, (double) (ms.dec.n_pn_noroom - b.dec.n_pn_noroom)/nw,
                                (double) (ms.dec.n_pn_kept - b.dec.n_pn_kept)/nw,
                                (double) (ms.dec.n_pn_kept_picked - b.dec.n_pn_kept_picked)/nw,
                                (double) (ms.dec.n_pn_kept_unpicked - b.dec.n_pn_kept_unpicked)/nw);
                    }
                    // The first streamed layer's needed reads and the word-start op, for this request's decode words
                    if (ms.first_streamed >= 0) {
                        std::string ws;
                        if (ms.ws_D > 0) {
                            std::string g, t;
                            for (int j = 1; j <= ms.ws_D && j <= 8; j++) {
                                for (int d = 1; d <= 8; d++) {
                                    const int64_t v = ms.n_ws_guess[j][d] - b.n_ws_guess[j][d];
                                    if (v > 0) {
                                        g += string_format("%slayer %d d%d %.2f", g.empty() ? "" : ", ", ms.first_streamed - j, d,
                                                (double) v/nw);
                                    }
                                }
                            }
                            for (int j = 0; j < 8; j++) {
                                int64_t p[4];
                                for (int i = 0; i < 4; i++) {
                                    p[i] = ms.n_ws_pick[j][i] - b.n_ws_pick[j][i];
                                }
                                const int64_t pk  = p[0] + p[1] + p[2] + p[3];
                                const int64_t add = ms.n_ws_add[j] - b.n_ws_add[j];
                                if (add == 0 && pk == 0) {
                                    continue;
                                }
                                t += string_format("%sstreamed #%d: added %.2f, bound %.2f, at the pick %.2f, removed unread %.2f; picks "
                                        "named %.2f = READY %.2f + loading %.2f + on the list %.2f + none %.2f", t.empty() ? "" : "; ", j,
                                        (double) add/nw, (double) (ms.n_ws_bind[j] - b.n_ws_bind[j])/nw,
                                        (double) (ms.n_ws_bind_pick[j] - b.n_ws_bind_pick[j])/nw, (double) (ms.n_ws_rm[j] - b.n_ws_rm[j])/nw,
                                        (double) pk/nw, (double) p[0]/nw, (double) p[1]/nw, (double) p[2]/nw, (double) p[3]/nw);
                            }
                            ws = string_format("; wordstart D %d: words with adds %lld, op runs with none %lld; guesses per word: %s; per "
                                    "target: %s", ms.ws_D, (long long) (ms.n_ws_words - b.n_ws_words), (long long) (ms.n_ws_skip - b.n_ws_skip),
                                    g.empty() ? "none" : g.c_str(), t.empty() ? "none" : t.c_str());
                            // The source slots (needed-early 1), for this request
                            const int64_t wp = ms.n_ws_post - b.n_ws_post, wd = ms.n_ws_done - b.n_ws_done;
                            if (wp > 0) {
                                std::string ld;
                                for (int d = 1; d <= 8; d++) {
                                    const int64_t v = ms.n_ws_late_d[d] - b.n_ws_late_d[d];
                                    if (v > 0) {
                                        ld += string_format("%sd%d %.2f", ld.empty() ? " (columns by distance: " : ", ", d, (double) v/nw);
                                    }
                                }
                                ld += ld.empty() ? "" : ")";
                                ws += string_format("; source slots per word: posted %.2f, adds done %.2f (posts the helper took; at N > 1 also a post whose columns were "
                                        "all too late), posts with a column too late %.2f%s, "
                                        "post -> adds done %.3f ms mean", (double) wp/nw, (double) wd/nw,
                                        (double) (ms.n_ws_late - b.n_ws_late)/nw, ld.c_str(),
                                        wd > 0 ? (double) (ms.t_ws_delay_us - b.t_ws_delay_us)/1000.0/(double) wd : 0.0);
                            }
                        }
                        SLT_INF(*this, "      moe stream: first streamed layer %d [decode]: needed reads per word %.3f%s\n",
                                ms.first_streamed, (double) (ms.n_first_miss - b.n_first_miss)/nw, ws.c_str());
                    }
                    // The behind rule / guess-inflight, for this request's decode words
                    const int64_t b_t = ms.dec.n_turns - b.dec.n_turns;
                    const int64_t b_w = ms.dec.n_gi_wait - b.dec.n_gi_wait;
                    if (b_t > 0 || b_w > 0) {
                        const int64_t b_bt = ms.dec.n_bh_turns - b.dec.n_bh_turns;
                        SLT_INF(*this, "      moe stream: behind [decode]: %lld words behind at least once, %.2f%% of the reader turns "
                                "taken while behind (%lld of %lld); entries skipped while behind %lld: bound later %lld, picked by "
                                "their target %lld, removed at their target's start %lld, other %lld; guess-inflight waits %lld, "
                                "%.3f ms\n",
                                (long long) (ms.dec.n_bh_words - b.dec.n_bh_words),
                                b_t > 0 ? 100.0*(double) b_bt/(double) b_t : 0.0, (long long) b_bt, (long long) b_t,
                                (long long) (ms.dec.n_bh_skip        - b.dec.n_bh_skip),
                                (long long) (ms.dec.n_bh_skip_bound  - b.dec.n_bh_skip_bound),
                                (long long) (ms.dec.n_bh_skip_picked - b.dec.n_bh_skip_picked),
                                (long long) (ms.dec.n_bh_skip_stale  - b.dec.n_bh_skip_stale),
                                (long long) (ms.dec.n_bh_skip_other  - b.dec.n_bh_skip_other),
                                (long long) b_w, (double) (ms.dec.t_gi_wait_us - b.dec.t_gi_wait_us)/1000.0);
                    }
                    if (p_a > 0) {
                        constexpr int n_rm = (int) (sizeof(ms.dec.n_pl_rm)/sizeof(ms.dec.n_pl_rm[0]));
                        static_assert(n_rm == 7, "the request line prints 7 removal reasons");
                        double  rm[n_rm];
                        int64_t hist[100];   // this request's add -> bind buckets, for the p90
                        for (int i = 0; i < n_rm; i++) {
                            rm[i] = (double) (ms.dec.n_pl_rm[i] - b.dec.n_pl_rm[i])/nw;
                        }
                        for (int i = 0; i < 100; i++) {
                            hist[i] = ms.dec.pl_delay_hist[i] - b.dec.pl_delay_hist[i];
                        }
                        SLT_INF(*this,
                                "      moe stream: list [decode] per word: added %.2f, bound %.2f (add -> bind %.3f ms mean, p90 %s), "
                                "removed without a read: picked %.2f, present %.2f, no room %.2f, word end %.2f, list full %.2f, "
                                "aborted %.2f, stale %.2f; bound at the pick %.2f, bound after its target ran %.2f; bound and picked %.2f, "
                                "bound unused %.2f; stale started, cancelled %.2f (%.2f MiB)\n",
                                (double) p_a/nw, (double) p_b/nw,
                                p_b - p_k > 0 ? (double) (ms.dec.t_pl_delay_us - b.dec.t_pl_delay_us)/1000.0/(double) (p_b - p_k) : 0.0,
                                common_moe_pl_p90(hist, p_b - p_k).c_str(),   // the delay leaves out the binds at the pick
                                rm[0], rm[1], rm[2], rm[3], rm[4], rm[5], rm[6], (double) p_k/nw, (double) p_r/nw, (double) p_u/nw,
                                (double) (p_b - p_u - p_k)/nw, (double) p_c/nw, (double) p_m/nw/1024.0/1024.0);
                    }
                }
            }
        }

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        SLT_INF(*this,
                "  graph reallocs = %10d    / %5d        (same size) / (changed size)\n",
                llama_perf_context(ctx_tgt).n_reallocs_same, llama_perf_context(ctx_tgt).n_reallocs_diff);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated.empty() ? debug_generated_text : generated.text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                struct cb_data_t {
                    common_speculative * spec;
                    llama_context * ctx;
                } cb_data = { slot.spec, slot.ctx_tgt };

                static auto cb = [](const mtmd_helper_embd_batch * b, void * user_data) {
                    const auto * data = static_cast<cb_data_t *>(user_data);

                    common_batch batch(data->ctx);
                    for (int32_t i = 0; i < b->n_tokens; ++i) {
                        llama_pos pos[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
                        for (int32_t j = 0; j < b->n_pos; ++j) {
                            pos[j] = b->pos[j * b->n_tokens + i];
                        }
                        batch.add_embd({ b->embd + (size_t) i * b->n_embd, 1, (size_t) b->n_embd }, pos, b->seq_id, false);
                    }

                    return common_speculative_process(data->spec, batch) ? 0 : 1;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    &cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    mbatch.reset(mtmd_batch_init(mctx));
    res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return try_decode();
}

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    mtmd_helper_cap mtmd_cap = {};
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

    // The disk tier, for POST /cache-disk/* while the server sleeps - only under the task queue's lock
    // (server_queue::run_while_sleeping), when the main loop is not running
    server_cache_disk * cache_disk_while_sleeping() {
        return cache_disk.get();
    }

    // The chats pinned for the disk tier's size limit; thread-safe, and kept while the server sleeps
    server_cache_disk_pins & cache_disk_pinned() {
        return cd_pins;
    }

    // The streaming engine's figures for GET /moe-stream/stats, read from the HTTP thread while a request runs: the engine's counters
    // are only read (never locked against the compute), the request each slot is running is remembered under moe_stats_mtx, and so
    // is the context pointer, which destroy() takes away under the same lock before the context is freed.
    // status: 0 = answered, 1 = the model does not stream, 2 = no model loaded now (the server sleeps)
    json moe_stats_json(int & status) {
        std::lock_guard<std::mutex> lock(moe_stats_mtx);
        status = 2;
        if (ctx_tgt == nullptr) {
            return json();
        }
        llama_moe_stream_stats now;
        llama_moe_stream_layout lay;
        if (!llama_moe_stream_get_stats(ctx_tgt, &now) || !llama_moe_stream_get_layout(ctx_tgt, &lay)) {
            status = 1;
            return json();
        }
        status = 0;
        const bool         timing = lay.timing != 0;
        const moe_layers_t ln     = moe_stream_layers_now(ctx_tgt);

        // the request that started last and has not ended, else the one that ended last
        const moe_stats_req * cur = nullptr;
        for (const auto & r : moe_reqs) {
            if (r.active && (cur == nullptr || r.seq > cur->seq)) {
                cur = &r;
            }
        }
        json request;
        if (cur != nullptr) {
            request = moe_stream_figures_json(now, cur->base, ln, cur->base_layers, timing);
            request["active"] = true;
        } else if (!moe_last.is_null()) {
            request = moe_last;
        }

        json ram_slots = json::array(), vram_experts = json::array(), layer_ids = json::array();
        int64_t n_ram = 0, n_vram = 0;
        for (const auto & l : ln) {
            layer_ids.push_back(l.il);
            ram_slots.push_back(l.ram_slots);
            vram_experts.push_back(l.vram_experts);
            n_ram  += l.ram_slots;
            n_vram += l.vram_experts;
        }
        json gpu_layers = json::array();
        for (int i = 0; i < lay.n_gpu_layers && i < LLAMA_MOE_STREAM_MAX_GPU_LAYERS; i++) {
            gpu_layers.push_back(lay.gpu_layers[i]);
        }
        const double mib = 1.0/1048576.0;
        return json {
            { "moe_stream", true },
            { "timing",     timing },
            { "request",    request },
            { "total",      moe_stream_figures_json(now, llama_moe_stream_stats{}, ln, moe_layers_t{}, timing) },
            { "layout", {
                { "layers",                 lay.n_layer },
                { "streamed_layers",        lay.n_streamed },
                { "streamed_layer_ids",     layer_ids },
                { "whole_gpu_layers",       gpu_layers },
                { "experts_per_layer",      lay.n_expert },
                { "experts_per_word",       lay.n_expert_used },
                { "expert_mib",             (double) lay.block_bytes*mib },
                { "ram_slots_per_layer",    ram_slots },
                { "ram_slots_total",        n_ram },
                { "vram_experts_per_layer", vram_experts },
                { "vram_experts_total",     n_vram },
                { "vram_follow",            lay.vram_follow != 0 },
                { "ram_pool_mib",           (double) lay.ram_pool_bytes*mib },
                { "vram_pool_mib",          (double) lay.vram_pool_bytes*mib },
                { "kv_cache_mib",           (double) lay.kv_bytes*mib },
                { "elastic", {
                    { "enabled",              lay.elastic_on != 0 },
                    { "state",                lay.elastic_state == 2 ? "squeezed_warn" : lay.elastic_state == 3 ? "reclaiming" : lay.elastic_state == 1 ? "shared" : "normal" },
                    { "slots_min_per_layer",  lay.elastic_slots_min },
                    { "slots_soft_min",       lay.elastic_slots_soft_min },
                    { "given_mib",            (double) lay.elastic_given_bytes*mib },
                    { "discardable_band_mib", (double) lay.elastic_band_bytes*mib },
                    { "os_free_room_mib",     lay.elastic_free_bytes >= 0 ? (double) lay.elastic_free_bytes*mib : -1.0 },
                } },
            } },
        };
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF
    // 1 = also snapshot the start of a new part when it only covers the few tokens before the last user
    // message (the near-duplicate of the user-start snapshot made right after); test-only, read once at start-up
    bool ckpt_near_dup = false; // env: LLAMA_SERVER_CKPT_NEAR_DUP

    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    // The chat states on disk (--cache-disk), null = off
    std::unique_ptr<server_cache_disk> cache_disk;
    server_cache_disk_pins cd_pins;   // POST /cache-disk/pins

    // GET /moe-stream/stats: what each slot's running request started from, and the figures of the last request that ended
    struct moe_stats_req {
        bool                   active = false;
        int64_t                seq    = 0;
        llama_moe_stream_stats base{};
        moe_layers_t           base_layers;
    };
    std::mutex                 moe_stats_mtx;   // guards ctx_tgt's lifetime for the route, moe_reqs, moe_last
    std::vector<moe_stats_req> moe_reqs;        // per slot
    int64_t                    moe_req_seq = 0;
    json                       moe_last;

    // a request starts / ends on a slot (the main loop's thread)
    void moe_stats_request_start(const server_slot & slot) {
        std::lock_guard<std::mutex> lock(moe_stats_mtx);
        if (moe_reqs.size() <= (size_t) slot.id) {
            moe_reqs.resize((size_t) slot.id + 1);
        }
        moe_stats_req & r = moe_reqs[(size_t) slot.id];
        r.active      = true;
        r.seq         = ++moe_req_seq;
        r.base        = slot.moe_base;
        r.base_layers = slot.moe_base_layers;
    }

    void moe_stats_request_end(const server_slot & slot) {
        std::lock_guard<std::mutex> lock(moe_stats_mtx);
        if ((size_t) slot.id < moe_reqs.size() && moe_reqs[(size_t) slot.id].active) {
            moe_reqs[(size_t) slot.id].active = false;
            json last = slot.moe_stream_json();   // the figures the request ended with
            if (!last.is_null()) {
                last["active"] = false;
                moe_last = std::move(last);
            }
        }
    }

    // POST /cache-disk/erase and /clear never wait; a request that arrived before one of them (a lower task id:
    // ids grow in arrival order) - answering now, or still deferred - does not save its state at its reply's end, and its
    // slot's content is cleared then, so a deleted chat cannot come back. One entry per erased chat id (a few bytes each).
    struct cd_mark {
        int     id   = -1;   // the erase task's id
        int64_t t_us = 0;    // when it ran (ggml_time_us)
    };
    std::map<std::string, cd_mark> cd_erased_before;
    int                            cd_cleared_before = -1;

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        std::lock_guard<std::mutex> moe_lock(moe_stats_mtx);   // GET /moe-stream/stats never sees a freed context
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // get the memory usage of mmproj, also used to check image_max_tokens against n_ubatch
        mtmd_memory_usage mmproj_usage = {{}, -1, false};
        int64_t mmproj_usage_t_us = 0;
        if (has_mmproj) {
            int64_t t_start = ggml_time_us();
            mmproj_usage = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            mmproj_usage_t_us = ggml_time_us() - t_start;
        }

        // optionally fit mmproj memory usage
        // on resume params_base already holds the margin from the first load
        if (has_mmproj && params_base.fit_params && !is_resume) {
            const auto & mmproj_mem = mmproj_usage.backend_mem_usage;
            const int64_t t_elapsed = mmproj_usage_t_us;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        // This server's snapshots and --cache-disk restore buffer go into the engine's RAM plan
        params_base.moe_stream_srv = {
            /*.n_ctx_checkpoints =*/ params_base.n_ctx_checkpoints,
            /*.n_ctx             =*/ (uint32_t) std::max(0, params_base.n_ctx),
            /*.n_seq             =*/ (uint32_t) std::max(1, params_base.n_parallel),
            /*.n_ubatch          =*/ (uint32_t) std::max(1, params_base.n_ubatch),
            /*.type_k            =*/ (int32_t) params_base.cache_type_k,
            /*.type_v            =*/ (int32_t) params_base.cache_type_v,
            /*.kv_unified        =*/ params_base.kv_unified,
            /*.swa_full          =*/ params_base.swa_full,
            /*.state_files       =*/ !params_base.cache_disk_dir.empty(),
            /*.other_rest        =*/ params_base.moe_stream_other_rest,
            /*.call_checkpoints  =*/ params_base.moe_stream_layer_major == 1 && params_base.n_parallel <= 1 &&
                                     params_base.n_ctx_checkpoints > 0 && !has_spec && !params_base.speculative.has_synth(),
        };
        params_base.moe_stream_srv_set = true;

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            // non-causal models need the whole image in one ubatch
            {
                const int n_ubatch = llama_n_ubatch(ctx_tgt);
                if (mmproj_usage.use_non_causal && mmproj_usage.image_max_tokens > n_ubatch) {
                    SRV_WRN("cap image_max_tokens (original=%d) to n_ubatch (%d) because model needs non-causal attention on image\n", mmproj_usage.image_max_tokens, n_ubatch);
                    SRV_WRN("%s\n", "increase n_ubatch (-ub) to increase vision token budget");
                    mparams.image_max_tokens = n_ubatch;
                    mparams.image_min_tokens = std::min(mparams.image_min_tokens, n_ubatch);
                }
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        mtmd_cap = mtmd_helper_get_capabilities(model_tgt, mctx);

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                        n_ctx_capped, n_ctx_train);
            }
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot(), params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot();

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            // Every way a request ends (a reply, a cancel, an error, an embedding / rerank) runs the erase /
            // clear check, so a slot answering when its chat was erased never keeps that chat's state
            slot.moe_timing = params_base.moe_stream_timing == 1;
            slot.callback_on_end = [this](server_slot & slot) {
                moe_stats_request_end(slot);
                cache_disk_covered(slot);
                cd_prune_marks();
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        {
            const char * LLAMA_SERVER_CKPT_NEAR_DUP = getenv("LLAMA_SERVER_CKPT_NEAR_DUP");
            ckpt_near_dup = LLAMA_SERVER_CKPT_NEAR_DUP && atoi(LLAMA_SERVER_CKPT_NEAR_DUP) != 0;

            if (ckpt_near_dup) {
                SRV_WRN("%s", "LLAMA_SERVER_CKPT_NEAR_DUP = 1: the near-duplicate context checkpoint before the last user message is made (test only)\n");
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(ctx_tgt, std::max(n_batch, params_base.n_parallel), n_embd);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.cache_disk_checkpoints > 0 && params_base.cache_disk_dir.empty()) {
            SRV_ERR("%s", "--cache-disk-checkpoints needs --cache-disk DIR (the snapshots are files of that directory)\n");
            return false;
        }
        if (!params_base.cache_disk_dir.empty()) {
            cache_disk = std::make_unique<server_cache_disk>();
            // Off where this server makes no context snapshots (--ctx-checkpoints 0, or a model whose
            // memory can all be cut back: no recurrent layers, no SWA window) - the same rule as the prompt loop's do_checkpoint
            const bool can_snap = params_base.n_ctx_checkpoints > 0 && (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS || n_swa > 0);
            cache_disk->max_snaps = can_snap ? params_base.cache_disk_checkpoints : 0;
            if (params_base.cache_disk_checkpoints > 0 && !can_snap) {
                SRV_WRN("cache-disk: --cache-disk-checkpoints %d is off: this server makes no context snapshots (%s)\n",
                        params_base.cache_disk_checkpoints, params_base.n_ctx_checkpoints <= 0 ? "--ctx-checkpoints 0" :
                        "the model's memory can all be cut back, so a request goes back without them");
            }
            cache_disk->n_swa     = n_swa;
            cache_disk->pins      = &cd_pins;
            if (!cache_disk->init(params_base.cache_disk_dir, params_base.cache_disk_gb,
                                  server_cache_disk_config(params_base, ctx_tgt, slots.empty() ? nullptr : slots[0].ctx_dft))) {
                cache_disk.reset();
                if (sleeping) {
                    // waking from --sleep-idle-seconds: the directory went bad while the server slept; the tier stays off
                    // instead of stopping a running server
                    SRV_WRN("cache-disk: '%s' cannot be used after waking; the disk tier is off until a restart\n",
                            params_base.cache_disk_dir.c_str());
                } else {
                    // at start-up a chat-state directory that cannot be used is refused: the server does not run without
                    // what it was asked for
                    SRV_ERR("cache-disk: refusing --cache-disk '%s' (see above); start without it or give a writable directory\n",
                            params_base.cache_disk_dir.c_str());
                    return false;
                }
            }
        }

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
            // The staging of the checkpoints a call takes inside it, now (the RAM plan counts it) - not at the first request
            if (!ckpt_copy_old() && call_ckpt_ok_any()) {
                llama_call_checkpoints_reserve(ctx_tgt, params_base.n_ctx_checkpoints);
            }
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mtmd_cap.vision,
                /* allow_audio           */ mtmd_cap.audio,
                /* allow_video           */ mtmd_cap.video,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning").get<bool>();
                bool specified = params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported) {
                    SRV_TRC("preserve_reasoning kwarg: %s\n",
                            it == params_base.default_template_kwargs.end() ? "unset (template default)" : it->second.c_str());
                } else {
                    SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
                }
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                // a busy slot is returned untouched, the caller defers the task
                if (ret->is_processing()) {
                    return ret;
                }

                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find an empty slot, or the one that has been least recently used
        if (ret == nullptr) {
            const auto free_slots = get_free_slots(1, -1);

            if (!free_slots.empty()) {
                ret = free_slots[0];

                SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt.n_tokens(), ret->t_last_used);

                update_cache = true;
            }
        }

        if (ret) {
            const bool is_completion = task.type == SERVER_TASK_TYPE_COMPLETION;

            // The disk tier's candidate, chosen before anything is restored so that one source is used.
            // When the slot's own content does not serve this task (the RAM tier's update condition) any entry may serve
            // it; otherwise only the request's own chat file (e.g. -np 1, the slot holds chat B and chat A comes back
            // with the same system prompt - the similarity pick keeps B's slot, but A's file covers much more of A)
            server_cache_disk::pick disk_pick;
            bool saved_for_disk = false;   // the slot's content went to the RAM tier before a chat-file restore
            if (cache_disk && is_completion && task.params.cache_prompt && !ret->is_processing()) {
                // (a slot named by id_slot can still be answering: the caller defers the task, nothing is restored now)
                disk_pick = cache_disk->find(ret->prompt, task.tokens, task.conv_id, task_lora_hash(task),
                                             /*chat_only =*/ !update_cache);
            }

            update_cache = update_cache && prompt_cache;

            // cache prompts only for the tasks that reuse them
            update_cache = update_cache && !task.is_stateless();

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                // One source per restore - the RAM tier's state, unless the disk's shares more of the request
                if (disk_pick.ok() && (int64_t) disk_pick.lcp > (int64_t) prompt_cache->peek(ret->prompt, task.tokens)) {
                    SRV_TRC("%s", "the disk tier's state shares more of the request than the RAM tier's\n");
                } else {
                    disk_pick = {};
                    if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                        ret->prompt_clear();
                    }
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            } else if (disk_pick.ok() && prompt_cache && is_completion) {
                // the chat-file restore replaces content the RAM tier's update did not keep: keep it as an LRU switch would
                ret->prompt_save(*prompt_cache);
                prompt_cache->update();
                saved_for_disk = true;
            }

            if (disk_pick.ok()) {
                // restore() frees the slot's own snapshots once the file's header has passed and before
                // its buffers are allocated (a header refusal keeps them)
                const auto rr = cache_disk->restore(disk_pick, ret->prompt, task.tokens, ret->ctx_tgt, ret->ctx_dft, ret->id, task.conv_id);
                if (rr == server_cache_disk::RESTORE_DONE) {
                    // the slot now holds that file's state, whatever happens to this task (it may still be deferred)
                    ret->chat_id = cache_disk->index[disk_pick.i].is_chat ? task.conv_id : std::string();
                }
                if (rr == server_cache_disk::RESTORE_FAILED) {
                    ret->prompt_clear();
                }
                // Also on a refusal when the slot's content was just saved to the RAM cache - a refusal after the
                // header (a damaged body) has already freed the slot's snapshots, and that copy has them
                if (rr != server_cache_disk::RESTORE_DONE && (update_cache || saved_for_disk)) {
                    // the file was refused or failed: the RAM tier's state, which it was preferred over (or the slot's own
                    // content, saved there just before), is still there
                    if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                        ret->prompt_clear();
                    }
                    prompt_cache->update();
                }
            }
        }

        return ret;
    }

    // After a reply (its final response is sent), the slot's state to --cache-disk, finished before the next
    // request is taken; not for child tasks of a multi-completion request (the parent's state is the one kept)
    // (and not when the request asked for no prompt cache: cache_prompt false, or for the disk to be left as it is: cache_disk_save false)
    // True when an erase / clear that arrived after this slot's request covers it; the slot's content is then
    // cleared (it is that chat's state). Called by cache_disk_save at a reply's end and on every release (callback_on_end)
    bool cache_disk_covered(server_slot & slot) {
        if (!slot.task || (cd_cleared_before < 0 && cd_erased_before.empty())) {
            return false;
        }
        const int id_task = slot.task->is_child() ? slot.task->id_parent : slot.task->id;
        const auto it = cd_erased_before.find(slot.chat_id);
        if (!(id_task < cd_cleared_before || (it != cd_erased_before.end() && id_task < it->second.id))) {
            return false;
        }
        if (slot.prompt.n_tokens() > 0 || !slot.chat_id.empty()) {
            SLT_INF(slot, "cache-disk: not saved: %s after this request arrived; the slot is cleared\n",
                    id_task < cd_cleared_before ? "the states were cleared" : "the chat was erased");
            slot.prompt_clear();
        }
        return true;
    }

    // An erase mark is needed only while a request that arrived before it (a lower task id) can still be
    // running, deferred or about to be posted: dropped when no waiting or running task has a lower id, and the mark is
    // over 10 s old (a request that already has its id but is still being prepared on an HTTP thread is posted within ms)
    void cd_prune_marks() {
        if (cd_erased_before.empty()) {
            return;
        }
        int lo = queue_tasks.lowest_waiting_id();
        for (const auto & sl : slots) {
            if (sl.is_processing() && sl.task) {
                lo = std::min(lo, sl.task->is_child() ? sl.task->id_parent : sl.task->id);
            }
        }
        const int64_t now = ggml_time_us();
        for (auto it = cd_erased_before.begin(); it != cd_erased_before.end();) {
            const bool done = lo >= it->second.id && now - it->second.t_us > 10*1000*1000;
            it = done ? cd_erased_before.erase(it) : std::next(it);
        }
    }

    void cache_disk_save(server_slot & slot) {
        if (!cache_disk || !slot.task) {
            return;
        }
        // an erase / clear that arrived after this request: no save, and the slot's content goes (it is that chat's state)
        if (cache_disk_covered(slot)) {
            return;
        }
        if (slot.task->is_child() || slot.task->type != SERVER_TASK_TYPE_COMPLETION || !slot.task->params.cache_prompt ||
                !slot.task->params.cache_disk_save) {
            return;
        }
        // The "start of message" snapshot of the request just answered (made at its last user message's
        // start) to its own file, now - the SSD is idle between replies; with another slot answering it is not written
        bool    snap_fresh = false;   // this reply wrote a new snapshot file
        int64_t snap_n     = -1;
        if (cache_disk->max_snaps > 0 && !slot.task->conv_id.empty()) {
            const int32_t pos_user = slot.task->params.message_spans.last_user_message_pos();
            const common_prompt_checkpoint * cs = nullptr;
            for (const auto & c : slot.prompt.checkpoints) {
                if (pos_user >= 0 && c.n_tokens == pos_user) {
                    cs = &c;
                }
            }
            bool busy = false;
            for (const auto & other : slots) {
                busy = busy || (other.id != slot.id && other.is_processing());
            }
            if (cs && busy) {
                SLT_INF(slot, "cache-disk: snapshot not written: slot(s) answering, the SSD is theirs (n_tokens = %" PRId64 ")\n", cs->n_tokens);
            } else if (cs) {
                cache_disk->write_snapshot(*cs, slot.prompt.tokens, slot.task->conv_id, server_cache_disk::lora_hash_of(slot.lora), &snap_fresh);
                snap_n = cs->n_tokens;
            } else if (pos_user > 0) {
                // Say so whenever a reply ends without that snapshot (none was made, or the RAM cap dropped it).
                // max_snaps is 0 where the server makes no snapshots, so this line is not printed there
                const bool on_disk = cache_disk->has_snapshot(slot.task->conv_id, pos_user, slot.prompt.tokens);
                SLT_INF(slot, "cache-disk: no snapshot at the last user message's start (n_tokens = %d) to write%s\n", pos_user,
                        on_disk ? ": it is already on the disk" : ": it is not in RAM (an edit of this message re-reads from an earlier one)");
            }
        }
        const bool saved = cache_disk->save(slot.prompt, slot.ctx_tgt, slot.ctx_dft, slot.id, slot.task->conv_id,
                                            server_cache_disk::lora_hash_of(slot.lora), ggml_time_us());
        if (snap_fresh && !saved) {
            // No chat file lists it (the save did not fit, failed, ...): no orphan
            cache_disk->remove_snapshot(slot.task->conv_id, snap_n, "its chat file was not saved");
        }
    }

    // The LoRA set a task runs with (as launch_slot_with_task sets it), for matching saved states
    uint64_t task_lora_hash(const server_task & task) const {
        return server_cache_disk::lora_hash_of(task.params.lora.empty() ? params_base.lora_adapters : construct_lora_list(task.params.lora));
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        g_srv_tt = srv_tt_on();
        SRV_TT("S launch", (long long) task.n_tokens());
        slot.chat_id = task.conv_id;   // "" unless --cache-disk is on and the request names its chat

        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }
            SRV_TT("S sampler_set", use_backend_sampling);

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            slot.smpl.reset();
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        slot.task = std::make_unique<const server_task>(std::move(task));

        // The streaming engine's counters are cumulative over the life of the model, so the
        // per-request line has to subtract what they were when this request started. The snapshot is
        // here, where the request starts, and it is used in print_timings().
        if (!llama_moe_stream_get_stats(ctx_tgt, &slot.moe_base)) {
            slot.moe_base = llama_moe_stream_stats{};
            slot.moe_base_layers.clear();
        } else {
            slot.moe_base_layers = moe_stream_layers_now(ctx_tgt);
            moe_stats_request_start(slot);
        }

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated.append(token_str, result.tok);
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated.text) < slot.generated.text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated.text.size());

            const std::string str_test = slot.generated.text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated.truncate(pos + stop_pos);
                pos = std::min(slot.n_sent_text, slot.generated.text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated.text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated.text.size() && (slot.generated.text[pos] == ' ' || slot.generated.text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated.text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated.truncate(pos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated.text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = slot.generated.substr(slot.n_sent_text - tkn.text_to_send.size(), tkn.text_to_send.size());
            res->tokens.assign(1, tkn.tok);
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated.text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = {};
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->stats.moe_stream = slot.moe_stream_json();
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.tokens[i].seq_id);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.tokens[i].seq_id);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
        }

        std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot * a, const server_slot * b) {
            if ((a->prompt.n_tokens() == 0) != (b->prompt.n_tokens() == 0)) {
                return a->prompt.n_tokens() == 0;
            }
            return a->t_last_used < b->t_last_used;
        });

        free_slots.resize(std::min(free_slots.size(), n_slots_needed));

        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            // keep the state that the child task is about to overwrite
            if (prompt_cache && slot->prompt_save(*prompt_cache)) {
                prompt_cache->update();
            }

            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // The rest of the prompt joins the call and its checkpoints are taken inside it - only where the engine can
    // (llama_call_checkpoints_ok: --moe-stream-layer-major 1, no rollback snapshots) and the call holds this one sequence with
    // nothing else (no draft context / speculative decoding, one slot, no alora split, no embeddings); otherwise the plain cuts.
    // LLAMA_SERVER_NO_CALL_CKPT=1 (test only) keeps the plain cuts (the bit-exact check's reference)
    bool call_ckpt_ok_any() const {   // the parts that do not depend on the slot (also checked at load)
        static const bool test_off = getenv("LLAMA_SERVER_NO_CALL_CKPT") && atoi(getenv("LLAMA_SERVER_NO_CALL_CKPT")) != 0;
        return !test_off && llama_call_checkpoints_ok(ctx_tgt) && ctx_dft == nullptr && !spec && params_base.n_parallel <= 1;
    }
    bool call_ckpt_ok(const server_slot & slot) const {
        return call_ckpt_ok_any() && !slot.need_embd() && !lora_all_alora(slot.lora);
    }
    // TEST ONLY: LLAMA_SERVER_TEST_CKPT_COPY_OLD=1 - the in-call checkpoints the old way (a fresh zeroed vector, then a
    // second copy into the checkpoint; the staging allocated at the first request)
    static bool ckpt_copy_old() {
        static const bool on = getenv("LLAMA_SERVER_TEST_CKPT_COPY_OLD") && atoi(getenv("LLAMA_SERVER_TEST_CKPT_COPY_OLD")) != 0;
        return on;
    }

    // before each decode attempt: the pending cuts of the slot (one slot only, call_ckpt_ok) armed for that call - its ubatches never
    // cross them, and a cut the call reaches is taken
    void arm_call_cuts() {
        for (auto & slot : slots) {
            if (slot.cc_pos.empty()) {
                continue;
            }
            std::unique_ptr<bool[]> keep(new bool[slot.cc_keep.size()]);
            for (size_t i = 0; i < slot.cc_keep.size(); ++i) {
                keep[i] = slot.cc_keep[i] != 0;
            }
            if (!llama_set_call_cuts(ctx_tgt, slot.id, slot.cc_pos.data(), keep.get(), (int32_t) slot.cc_pos.size())) {
                SLT_WRN(slot, "%s", "the engine refused the call's cuts - the prompt runs without them\n");
                slot.cc_pos.clear();
                slot.cc_ntok.clear();
                slot.cc_keep.clear();
            }
        }
    }

    // after a decode of batch tokens [off, off + n_tokens): the checkpoints the engine took inside it (the pending cuts up to its last
    // position + 1), made with the usual bookkeeping (the cap, superseding, the disk keep) in the order the usual batches would make them;
    // cuts past it stay pending for the next call of the same batch (a retry with a smaller n_batch)
    void make_call_checkpoints(int32_t off, int32_t n_tokens) {
        llama_pos last = -1;
        for (int32_t i = off; i < off + n_tokens && i < batch.size(); ++i) {
            last = std::max(last, batch.tokens[i].pos);
        }
        for (auto & slot : slots) {
            size_t n_done = 0;
            for (size_t i = 0; i < slot.cc_pos.size(); ++i) {
                const llama_pos p = slot.cc_pos[i];
                if (p > last + 1) {
                    break;
                }
                n_done = i + 1;
                if (!slot.cc_keep[i]) {
                    continue;
                }
                const size_t n = llama_get_call_checkpoint(ctx_tgt, p, nullptr, 0);
                if (n == 0) {
                    SLT_WRN(slot, "the checkpoint at n_tokens = %" PRId64 " was not taken inside the call - none made there\n",
                            slot.cc_ntok[i]);
                    continue;
                }
                if (ckpt_copy_old()) {   // TEST ONLY: the old copies
                    std::vector<uint8_t> data(n);
                    if (llama_get_call_checkpoint(ctx_tgt, p, data.data(), n) != n) {
                        SLT_WRN(slot, "the checkpoint at n_tokens = %" PRId64 " was not taken inside the call - none made there\n",
                                slot.cc_ntok[i]);
                        continue;
                    }
                    create_checkpoint(slot, slot.prompt.n_tokens() - slot.cc_ntok[i], p - 1, p - 1, &data);
                    continue;
                }
                // Written once, straight into the checkpoint's own buffer (create_checkpoint gives it the evicted one's)
                create_checkpoint(slot, slot.prompt.n_tokens() - slot.cc_ntok[i], p - 1, p - 1, nullptr, p, n);
                // TEST ONLY: LLAMA_SERVER_TEST_CKPT_VERIFY=1 - the bytes also read the old way (a fresh vector) and compared
                static const bool verify = getenv("LLAMA_SERVER_TEST_CKPT_VERIFY") && atoi(getenv("LLAMA_SERVER_TEST_CKPT_VERIFY")) != 0;
                if (verify && !slot.prompt.checkpoints.empty() &&
                        slot.prompt.checkpoints.back().n_tokens == slot.cc_ntok[i]) {
                    std::vector<uint8_t> ref(n);
                    const bool same = llama_get_call_checkpoint(ctx_tgt, p, ref.data(), n) == n && slot.prompt.checkpoints.back().data_tgt == ref;
                    SLT_INF(slot, "checkpoint verify: n_tokens = %" PRId64 ", %zu bytes: %s\n", slot.prompt.checkpoints.back().n_tokens, n,
                            same ? "SAME" : "DIFFERENT");
                }
            }
            slot.cc_pos.erase(slot.cc_pos.begin(), slot.cc_pos.begin() + n_done);
            slot.cc_ntok.erase(slot.cc_ntok.begin(), slot.cc_ntok.begin() + n_done);
            slot.cc_keep.erase(slot.cc_keep.begin(), slot.cc_keep.begin() + n_done);
        }
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max,
            const std::vector<uint8_t> * data_call = nullptr,   // the bytes taken inside the call
            llama_pos call_pos = -1, size_t call_n = 0) {       // or read from the call's checkpoint at call_pos (call_n bytes)
        const int id_task = slot.task->id;
        // The largest buffer of a checkpoint erased below becomes the new checkpoint's (its pages are already there: a fresh
        // 74.9 MiB buffer cost page faults + zeroing + a second copy, ~75 ms per message for two on Ling Flash)
        std::vector<uint8_t> reuse;
        const auto take = [&](common_prompt_checkpoint & c) {
            if (c.data_tgt.capacity() > reuse.capacity()) {
                reuse = std::move(c.data_tgt);
            }
        };
        const int64_t t_ckpt0 = ggml_time_us();   // the copy's time, in the trace line

        // With --cache-disk-checkpoints, the snapshot at the last user message's start is the one written to
        // the disk at the reply's end; a long message makes two more (n - min(n_batch, 4 + n_ubatch), n - 4), so the cap below
        // drops the next oldest instead of it (an edit near the end of a long last message then restarts from its start)
        const int64_t n_keep_disk = cache_disk && cache_disk->max_snaps > 0 && !slot.task->conv_id.empty()
                ? (int64_t) slot.task->params.message_spans.last_user_message_pos() : -1;
        const auto keep_for_disk = [&](const common_prompt_checkpoint & c) { return n_keep_disk > 0 && c.n_tokens == n_keep_disk; };

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        // only when the list is full, otherwise short prompts keep just the oldest checkpoint
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin();
                slot.prompt.checkpoints.size() + 1 >= (size_t) params_base.n_ctx_checkpoints &&
                it != slot.prompt.checkpoints.end(); ) {
            if (it->id_task != id_task && last >= 0 && it->n_tokens <= last + params_base.checkpoint_min_step && !keep_for_disk(*it)) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

                take(*it);
                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = it->n_tokens;
            ++it;
        }

        if (params_base.n_ctx_checkpoints == 1 && slot.prompt.checkpoints.size() == 1 &&
                keep_for_disk(slot.prompt.checkpoints.front()) && slot.prompt.checkpoints.front().n_tokens != slot.prompt.n_tokens() - n_tokens_cur) {
            // A1 / A3 at --ctx-checkpoints 1: the one place holds the snapshot the disk needs (the last user message's start), so
            // the newer one is not made (a regenerate then re-reads the last message; without --cache-disk-checkpoints the
            // newer one replaces it as before)
            SLT_TRC(slot, "context checkpoint at n_tokens = %" PRId64 " not made: --ctx-checkpoints 1 holds the last user message's start for --cache-disk-checkpoints\n",
                    (int64_t) (slot.prompt.n_tokens() - n_tokens_cur));
            return;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            auto victim = slot.prompt.checkpoints.begin();
            if (keep_for_disk(*victim) && slot.prompt.checkpoints.size() > 1) {
                ++victim;   // The next oldest goes, the one the disk needs stays
            }
            const auto & cur = *victim;

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)%s\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024,
                    victim != slot.prompt.checkpoints.begin() ? " - the one at the last user message's start is kept for --cache-disk-checkpoints" : "");

            take(*victim);
            slot.prompt.checkpoints.erase(victim);
        }

        // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        {
            const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if (it->n_tokens == n_tokens_new) {
                    SLT_TRC(slot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", it->n_tokens);
                    take(*it);
                    it = slot.prompt.checkpoints.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        if (!ckpt_copy_old()) {
            cur.data_tgt = std::move(reuse);   // (update_tgt's resize keeps it too)
        }
        if (call_pos >= 0) {
            cur.data_tgt.resize(call_n);   // (the same size as the evicted one's: nothing is zeroed)
            // (make_call_checkpoints read this size from the same state just before - the same deterministic write -
            // so this read cannot fail in practice; if it did, the checkpoints erased above for it are gone too, unlike the old path)
            if (llama_get_call_checkpoint(ctx_tgt, call_pos, cur.data_tgt.data(), call_n) != call_n) {
                SLT_WRN(slot, "the checkpoint at n_tokens = %" PRId64 " could not be read from the call - none made there "
                        "(the older checkpoints it replaced are gone)\n", cur.n_tokens);
                slot.prompt.checkpoints.pop_back();
                return;
            }
        } else if (data_call != nullptr) {
            cur.data_tgt = *data_call;
        } else {
            cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }
        // A reused buffer larger than this checkpoint (a model whose partial state size varies, e.g. SWA) would hold
        // memory its size() - the cache budget's measure - does not show; Ling's recurrent state has one size, so this never copies there
        if (cur.data_tgt.capacity() > cur.data_tgt.size()) {
            cur.data_tgt.shrink_to_fit();
        }
        cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // LLAMA_SERVER_CKPT_DUMP=<dir> (test only): each checkpoint's bytes in <dir>/ckpt_<n_tokens>.bin (the layer-major bit-exact check)
        if (const char * dump = getenv("LLAMA_SERVER_CKPT_DUMP")) {
            const std::string fn = string_format("%s/ckpt_%" PRId64 ".bin", dump, cur.n_tokens);
            if (FILE * f = fopen(fn.c_str(), "wb")) {
                fwrite(cur.data_tgt.data(), 1, cur.data_tgt.size(), f);
                fclose(f);
            }
        }
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB, %.2f ms)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024, (ggml_time_us() - t_ckpt0) / 1000.0);
    }

    // checkpoints are appended to the slot save file, after the llama state payload
    // they cannot be recreated from the final state alone (a recurrent state cannot be rewound)
    static constexpr uint32_t SLOT_CKPT_MAGIC   = 0x504b4353; // "SCKP"
    static constexpr uint32_t SLOT_CKPT_VERSION = 1;

    static bool ckpt_read(std::ifstream & ifs, void * dst, size_t size, size_t & n_read) {
        if (!ifs.read((char *) dst, size)) {
            return false;
        }
        n_read += size;
        return true;
    }

    static bool ckpt_read_buf(std::ifstream & ifs, std::vector<uint8_t> & buf, size_t n_avail, size_t & n_read) {
        uint64_t n = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckpt_read(ifs, &n, sizeof(n), n_read) || n > n_avail - n_read) {
            return false;
        }
        buf.resize(n);
        return n == 0 || ckpt_read(ifs, buf.data(), n, n_read);
    }

    static void ckpt_write(std::ofstream & ofs, const void * src, size_t size, size_t & n_written) {
        ofs.write((const char *) src, size);
        n_written += size;
    }

    static void ckpt_write_buf(std::ofstream & ofs, const std::vector<uint8_t> & buf, size_t & n_written) {
        const uint64_t n = buf.size();
        ckpt_write(ofs, &n, sizeof(n), n_written);
        if (n > 0) {
            ckpt_write(ofs, buf.data(), n, n_written);
        }
    }

    // returns false if the appendix could not be written completely
    bool save_slot_checkpoints(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        if (slot.prompt.checkpoints.empty()) {
            return true;
        }
        std::ofstream ofs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_CKPT_MAGIC;
        const uint32_t version = SLOT_CKPT_VERSION;
        const uint32_t count   = (uint32_t) slot.prompt.checkpoints.size();
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write(ofs, &count,   sizeof(count),   n_written);
        for (const auto & cur : slot.prompt.checkpoints) {
            ckpt_write(ofs, &cur.n_tokens, sizeof(cur.n_tokens), n_written);
            ckpt_write(ofs, &cur.pos_min,  sizeof(cur.pos_min),  n_written);
            ckpt_write(ofs, &cur.pos_max,  sizeof(cur.pos_max),  n_written);
            ckpt_write_buf(ofs, cur.data_tgt,  n_written);
            ckpt_write_buf(ofs, cur.data_dft,  n_written);
            ckpt_write_buf(ofs, cur.data_spec, n_written);
        }
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s' - the appendix is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended %u context checkpoint(s) (%.3f MiB) to '%s'\n",
                count, (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable appendix
    size_t load_slot_checkpoints(const std::string & filepath, size_t offset, server_slot & slot) const {
        std::ifstream ifs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset; // bytes after the llama state payload
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        uint32_t count   = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_CKPT_MAGIC) {
            return 0;
        }
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_CKPT_VERSION ||
            !ckpt_read(ifs, &count,   sizeof(count),   n_read)) {
            SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        std::list<common_prompt_checkpoint> checkpoints;
        for (uint32_t i = 0; i < count; ++i) {
            common_prompt_checkpoint cur;
            cur.id_task = -1; // not created by a task - marks a checkpoint restored from a slot file
            if (!ckpt_read(ifs, &cur.n_tokens, sizeof(cur.n_tokens), n_read) ||
                !ckpt_read(ifs, &cur.pos_min,  sizeof(cur.pos_min),  n_read) ||
                !ckpt_read(ifs, &cur.pos_max,  sizeof(cur.pos_max),  n_read) ||
                !ckpt_read_buf(ifs, cur.data_tgt,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_dft,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_spec, n_avail, n_read)) {
                SLT_WRN(slot, "truncated context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            // a saved checkpoint always holds a target state - an empty blob would roll back without restoring anything
            if (cur.data_tgt.empty()) {
                SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            checkpoints.push_back(std::move(cur));
            if (checkpoints.size() > (size_t) params_base.n_ctx_checkpoints) {
                checkpoints.pop_front();
            }
        }
        // the slot file does not check the draft context - test-load one draft checkpoint, drop the draft data if it does not fit
        if (ctx_dft != nullptr && !checkpoints.empty() && !checkpoints.back().data_dft.empty()) {
            const bool ok = checkpoints.back().load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
            if (!ok) {
                SLT_WRN(slot, "draft context checkpoint data in '%s' does not match the draft context - dropped\n", filepath.c_str());
                for (auto & cur : checkpoints) {
                    cur.clear_dft();
                }
            }
        }
        slot.prompt.checkpoints = std::move(checkpoints);
        SLT_INF(slot, "restored %zu context checkpoint(s) from '%s'\n", slot.prompt.checkpoints.size(), filepath.c_str());
        return n_read;
    }

    // the live draft state follows the checkpoints: the full draft context and the speculative state
    // the llama state payload holds the target context only
    static constexpr uint32_t SLOT_DRAFT_MAGIC   = 0x54464453; // "SDFT"
    static constexpr uint32_t SLOT_DRAFT_VERSION = 1;

    // returns false if the draft state could not be written completely
    bool save_slot_draft(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (ctx_dft != nullptr) {
            data_dft.resize(llama_state_seq_get_size_ext(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE));
            llama_state_seq_get_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        common_speculative_get_state(spec.get(), slot.id, data_spec);
        if (data_dft.empty() && data_spec.empty()) {
            return true;
        }
        std::ofstream ofs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_DRAFT_MAGIC;
        const uint32_t version = SLOT_DRAFT_VERSION;
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write_buf(ofs, data_dft,  n_written);
        ckpt_write_buf(ofs, data_spec, n_written);
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s' - the draft state is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended the draft state (%.3f MiB) to '%s'\n", (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable draft state
    // the draft starts empty unless the file holds a draft state that fits the draft context
    size_t load_slot_draft(const std::string & filepath, size_t offset, server_slot & slot) const {
        if (ctx_dft != nullptr) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
        }
        std::ifstream ifs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset;
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_DRAFT_MAGIC) {
            return 0;
        }
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_DRAFT_VERSION ||
            !ckpt_read_buf(ifs, data_dft,  n_avail, n_read) ||
            !ckpt_read_buf(ifs, data_spec, n_avail, n_read)) {
            SLT_WRN(slot, "invalid draft state in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        if (!data_dft.empty()) {
            if (ctx_dft == nullptr ||
                llama_state_seq_set_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) != data_dft.size()) {
                if (ctx_dft != nullptr) {
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
                }
                SLT_WRN(slot, "draft state in '%s' does not match the draft context - dropped\n", filepath.c_str());
                return n_read;
            }
        }
        if (!data_spec.empty()) {
            common_speculative_set_state(spec.get(), slot.id, data_spec);
        }
        SLT_INF(slot, "restored the draft state (%.3f MiB) from '%s'\n", (float) n_read / 1024 / 1024, filepath.c_str());
        return n_read;
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_ckpt = 0;
                    if (!save_slot_checkpoints(filepath, *slot, nwrite_ckpt)) {
                        send_error(task, "Unable to save slot: incomplete context checkpoints", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_dft = 0;
                    if (!save_slot_draft(filepath, *slot, nwrite_dft)) {
                        send_error(task, "Unable to save slot: incomplete draft state", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite + nwrite_ckpt + nwrite_dft;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                        slot->chat_id.clear();   // a /slots file's state belongs to no --cache-disk chat
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    // nread is the end offset of the llama state payload within the file
                    const size_t nread_ckpt = load_slot_checkpoints(filepath, nread, *slot);
                    const size_t nread_dft  = load_slot_draft(filepath, nread + nread_ckpt, *slot);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread + nread_ckpt + nread_dft;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_CACHE_DISK_ERASE:
            case SERVER_TASK_TYPE_CACHE_DISK_CLEAR:
                {
                    // The app's "delete chat" / "delete all": the engine's saved states, never the chat text. It
                    // runs at once on this loop (after any save in progress: saves run here too) and never waits or defers,
                    // so it takes no slot's turn. A request that arrived before it and is still answering or deferred is
                    // marked through its task id: it does not save at its reply's end, and its slot is cleared then
                    const bool is_erase = task.type == SERVER_TASK_TYPE_CACHE_DISK_ERASE;
                    const std::string & chat = task.conv_id;
                    if (!cache_disk) {
                        send_error(task, "the disk tier is off (its directory could not be used after waking)", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    if (is_erase) {
                        cd_mark & m = cd_erased_before[chat];
                        m.id   = std::max(m.id, task.id);
                        m.t_us = ggml_time_us();
                    } else {
                        cd_cleared_before = std::max(cd_cleared_before, task.id);
                        // entries the clear covers (an id below it) are no longer needed
                        for (auto it = cd_erased_before.begin(); it != cd_erased_before.end();) {
                            it = it->second.id < cd_cleared_before ? cd_erased_before.erase(it) : std::next(it);
                        }
                    }

                    json data;
                    size_t n_ram   = 0;
                    size_t n_slots = 0;
                    size_t n_busy  = 0;
                    if (prompt_cache) {
                        auto & st = prompt_cache->states;
                        const size_t n0 = st.size();
                        if (is_erase) {
                            st.remove_if([&](const server_prompt_cache_state & x) { return x.conv_id == chat; });
                        } else {
                            st.clear();
                        }
                        n_ram = n0 - st.size();
                    }
                    for (auto & slot : slots) {
                        if (is_erase && slot.chat_id != chat) {
                            continue;
                        }
                        if (slot.is_processing()) {
                            n_busy++;   // not saved at its reply's end, cleared then (cache_disk_save)
                        } else if (slot.prompt.n_tokens() > 0) {
                            slot.prompt_clear();
                            n_slots++;
                        }
                    }
                    cd_prune_marks();
                    if (is_erase) {
                        const int64_t n_bytes = cache_disk->erase_chat(chat);
                        if (n_bytes < 0) {
                            send_error(task, "the chat's state file could not be deleted (see the server log)", ERROR_TYPE_SERVER);
                            break;
                        }
                        // The chat's resumable-stream buffer goes too; not cancelled. Only after the
                        // files are gone (a failed erase keeps it)
                        const size_t n_streams = server_stream_drop_session(chat);
                        SRV_INF("cache-disk: erase %s: file %.1f MB, %zu RAM-cache states, %zu slots cleared, %zu answering (cleared at their end), %zu stream sessions\n",
                                server_cache_disk::chat_name(chat).c_str(), n_bytes/1e6, n_ram, n_slots, n_busy, n_streams);
                        data = {
                            { "erased_file",  n_bytes > 0 },
                            { "file_bytes",   n_bytes },
                            { "ram_states",   n_ram },
                            { "slots",        n_slots },
                            { "slots_answering", n_busy },
                            { "stream_sessions", n_streams },
                        };
                    } else {
                        const auto [n_files, n_bytes] = cache_disk->clear();
                        const size_t n_streams = server_stream_drop_all_sessions();
                        SRV_INF("cache-disk: clear: %zu files (%.1f MB), %zu RAM-cache states, %zu slots cleared, %zu answering (cleared at their end), %zu stream sessions\n",
                                n_files, n_bytes/1e6, n_ram, n_slots, n_busy, n_streams);
                        data = {
                            { "files",      n_files },
                            { "file_bytes", n_bytes },
                            { "ram_states", n_ram },
                            { "slots",      n_slots },
                            { "slots_answering", n_busy },
                            { "stream_sessions", n_streams },
                        };
                    }
                    auto res = std::make_unique<server_task_result_cache_disk>();
                    res->id   = task.id;
                    res->data = std::move(data);
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_CACHE_DISK_FORK:
                {
                    // The app's "new branch": the saved files of chat conv_id_from are copied to chat conv_id, replacing that
                    // chat's own. Runs at once on this loop, like erase: after any save in progress, taking no slot's turn
                    const std::string & to = task.conv_id;
                    if (!cache_disk) {
                        send_error(task, "the disk tier is off (its directory could not be used after waking)", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }
                    const server_cache_disk::fork_result fr = cache_disk->fork_chat(task.conv_id_from, to);
                    if (!fr.ok()) {
                        send_error(task, fr.error_text(), fr.found ? ERROR_TYPE_SERVER : ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    // the new chat's earlier states are gone, as after an erase of it: a request of it that arrived before
                    // does not save over the copy, and the RAM states and idle slots that hold it are dropped
                    cd_mark & m = cd_erased_before[to];
                    m.id   = std::max(m.id, task.id);
                    m.t_us = ggml_time_us();
                    size_t n_ram   = 0;
                    size_t n_slots = 0;
                    size_t n_busy  = 0;
                    if (prompt_cache) {
                        auto & st = prompt_cache->states;
                        const size_t n0 = st.size();
                        st.remove_if([&](const server_prompt_cache_state & x) { return x.conv_id == to; });
                        n_ram = n0 - st.size();
                    }
                    for (auto & slot : slots) {
                        if (slot.chat_id != to) {
                            continue;
                        }
                        if (slot.is_processing()) {
                            n_busy++;
                        } else if (slot.prompt.n_tokens() > 0) {
                            slot.prompt_clear();
                            n_slots++;
                        }
                    }
                    cd_prune_marks();
                    const size_t n_streams = server_stream_drop_session(to);
                    json data = fr.to_json();
                    data["ram_states"]      = n_ram;
                    data["slots"]           = n_slots;
                    data["slots_answering"] = n_busy;
                    data["stream_sessions"] = n_streams;
                    auto res = std::make_unique<server_task_result_cache_disk>();
                    res->id   = task.id;
                    res->data = std::move(data);
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            SRV_TT("S pre_decode_b", 0);
            pre_decode();
            SRV_TT("S pre_decode_e", batch.size());
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch.render(off, n_tokens);
                SRV_TT("S arm_b", n_tokens);
                arm_call_cuts();   // each attempt (a retry with a smaller n_batch decodes the rest in several calls)
                SRV_TT("S arm_e", 0);
                bool ok = decode(n_batch, off);
                SRV_TT("S decode_e", ok);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    make_call_checkpoints(off, n_tokens);
                    SRV_TT("S make_ckpts_e", 0);

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const server_reprocess &) {
                break;   // (the slots start their prompts again at the next update)
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off);
                SRV_TT("S post_decode_e", 0);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    // stale candidates: a replay never reads them, a new draft refills them
                    slot.spec_draft_q.clear();

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        const bool spec_reject = slot.use_spec_rejection();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .pos0     = */ slot.prompt.tokens.pos_next(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                            /* .result_q = */ spec_reject ? &slot.spec_draft_q : nullptr,
                            /* .temp     = */ slot.task->params.sampling.temp,
                            /* .seed     = */ slot.task->params.sampling.seed,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.stats.n_draft_tokens += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    GGML_ASSERT(ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.id);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start();
                        SRV_TT("S prompt_start", (long long) slot.stats.t_start);

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not support logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt && !slot.task->is_stateless()) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // number of cached tokens shared with the new prompt, before any checkpoint is restored
                            const int n_prefix = n_past;

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
                            // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
                            const llama_pos pos_eval = pos_next - (has_new_tokens ? 0 : 1);

                            // a state or a checkpoint is useful when its pos_min is below this threshold
                            const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                // A state that starts at position 0 holds every position, so it is usable whatever the threshold (the
                                // checkpoint search below already accepts pos_min == 0). Before, a chat shorter than the sliding window
                                // (threshold clamped to 0) had its state rejected and went back to an old checkpoint every turn
                                if (pos_min >= pos_min_thold && pos_min > 0) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    // The chat's "start of message" snapshots on the disk count too - the
                                    // newest one the same rule allows, if it is newer than the RAM one; read only to load it
                                    bool used_disk = false;
                                    common_prompt_checkpoint * ram_ck = do_reset ? nullptr : &*it;   // a std::list: stays valid
                                    if (cache_disk && cache_disk->max_snaps > 0 && !slot.chat_id.empty()) {
                                        const int64_t ram_n = do_reset ? -1 : it->n_tokens;
                                        const size_t di = cache_disk->find_snapshot(slot.chat_id, server_cache_disk::lora_hash_of(slot.lora),
                                                ram_n, pos_next, pos_min_thold, slot.task->tokens);
                                        int64_t e_n = 0;
                                        int32_t e_min = 0, e_max = 0;
                                        if (di != SIZE_MAX && cache_disk->snapshot_where(di, e_n, e_min, e_max)) {
                                            // Never more than --ctx-checkpoints in RAM, also while the file is read - first the RAM
                                            // ones this restore drops anyway (past its position; a RAM restore would drop them too),
                                            // then the oldest while the list is full (not the one a failed read falls back to)
                                            const int32_t pos_next_d = std::min(pos_next, std::max(e_min + 1, e_max));
                                            for (auto ci = slot.prompt.checkpoints.begin(); ci != slot.prompt.checkpoints.end();) {
                                                if (ci->pos_max > pos_next_d && &*ci != ram_ck) {
                                                    SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB) before the disk snapshot is read\n",
                                                            ci->pos_min, ci->pos_max, ci->n_tokens, n_swa, pos_next_d, (float) ci->size() / 1024 / 1024);
                                                    ci = slot.prompt.checkpoints.erase(ci);
                                                } else {
                                                    ++ci;
                                                }
                                            }
                                            while (slot.prompt.checkpoints.size() >= (size_t) std::max(1, params_base.n_ctx_checkpoints)) {
                                                auto ci = slot.prompt.checkpoints.begin();
                                                if (&*ci == ram_ck) {
                                                    ++ci;
                                                }
                                                if (ci == slot.prompt.checkpoints.end()) {
                                                    // Only the fallback is left and the cap is reached (--ctx-checkpoints 1):
                                                    // it goes too, so the cap holds while the file is read (a failed read then re-reads)
                                                    ci = slot.prompt.checkpoints.begin();
                                                    ram_ck = nullptr;
                                                    do_reset = true;
                                                }
                                                SLT_TRC(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB) before the disk snapshot is read\n",
                                                        ci->pos_min, ci->pos_max, ci->n_tokens, (float) ci->size() / 1024 / 1024);
                                                slot.prompt.checkpoints.erase(ci);
                                            }
                                        }
                                        common_prompt_checkpoint dc;
                                        bool loaded = di != SIZE_MAX && cache_disk->read_snapshot(di, slot.chat_id, dc);
                                        if (loaded) {
                                            // Checked like restore() does (load_tgt / load_dft abort on a size mismatch):
                                            // a state llama.cpp does not accept clears the slot, and the request re-reads
                                            loaded = llama_state_seq_set_data_ext(ctx_tgt, dc.data_tgt.data(), dc.data_tgt.size(), slot.id,
                                                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == dc.data_tgt.size();
                                            if (loaded && ctx_dft && !dc.data_dft.empty()) {
                                                loaded = llama_state_seq_set_data_ext(ctx_dft, dc.data_dft.data(), dc.data_dft.size(), slot.id,
                                                        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == dc.data_dft.size();
                                            }
                                            if (!loaded) {
                                                cache_disk->refuse_snapshot(di, "llama.cpp did not accept its state (a size mismatch); the slot is cleared");
                                                // The request goes on in this slot, so it keeps its chat id (an erase
                                                // during the reply must still find it, and its end must still check the erase marks)
                                                const std::string keep_chat = slot.chat_id;
                                                slot.prompt_clear();
                                                slot.chat_id = keep_chat;
                                                ram_ck   = nullptr;
                                                do_reset = true;
                                                pos_next = 0;
                                                n_past   = 0;
                                            }
                                        }
                                        if (loaded) {
                                            common_speculative_set_state(spec.get(), slot.id, dc.data_spec);

                                            pos_next = std::min(pos_next, std::max(dc.pos_min + 1, dc.pos_max));
                                            n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) dc.n_tokens);
                                            const int64_t dc_n = dc.n_tokens;
                                            const int32_t dc_min = dc.pos_min, dc_max = dc.pos_max;
                                            // it stays in the RAM list as a RAM restore's checkpoint does (the batch splitting below reads
                                            // the list, so a missing one changes the batches and the float order); the list is cut back to
                                            // --ctx-checkpoints after the invalidated ones are erased below
                                            auto pos = slot.prompt.checkpoints.begin();
                                            while (pos != slot.prompt.checkpoints.end() && pos->n_tokens < dc.n_tokens) {
                                                ++pos;
                                            }
                                            slot.prompt.checkpoints.insert(pos, std::move(dc));
                                            SLT_INF(slot, "cache-disk: restored the disk snapshot at n_tokens = %" PRId64 " (pos_min = %d, pos_max = %d, n_past = %d)%s; %zu snapshot(s) in RAM with it\n",
                                                    dc_n, dc_min, dc_max, n_past, do_reset ? "" : ", newer than the RAM one", slot.prompt.checkpoints.size());
                                            used_disk = true;
                                            do_reset  = false;
                                        }
                                    }

                                    if (!used_disk && ram_ck == nullptr) {
                                        do_reset = true;   // The fallback was dropped for the cap
                                    }
                                    if (!do_reset && !used_disk) {
                                        // restore the context checkpoint (upstream #26004: load_* return false instead of aborting)
                                        if (!ram_ck->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
                                            !ram_ck->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                                            if (ram_ck->id_task != -1) {
                                                GGML_ABORT("failed to restore context checkpoint\n");
                                            }
                                            // restored from a slot file, not guaranteed to load - fall back to full prompt re-processing
                                            SLT_WRN(slot, "%s", "failed to load context checkpoint restored from a slot file\n");
                                            do_reset = true;
                                        }
                                    }
                                    if (!do_reset && !used_disk) {
                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.id, ram_ck->data_spec);

                                        pos_next = std::min(pos_next, std::max(ram_ck->pos_min + 1, ram_ck->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) ram_ck->n_tokens);
                                        SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", ram_ck->pos_min, ram_ck->pos_max, ram_ck->n_tokens, n_past, (float) ram_ck->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next, or that cover tokens not shared with the new prompt
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next || cur.n_tokens > n_prefix) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                                // Never more than --ctx-checkpoints in RAM after a disk snapshot joined the list
                                while (slot.prompt.checkpoints.size() > (size_t) std::max(1, params_base.n_ctx_checkpoints)) {
                                    const auto & cur = slot.prompt.checkpoints.front();
                                    SLT_TRC(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                                            cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
                                    slot.prompt.checkpoints.pop_front();
                                }
                                // The chat's disk snapshots past this point that were made for another history
                                if (cache_disk && cache_disk->max_snaps > 0 && !slot.chat_id.empty() && slot.task->params.cache_disk_save) {
                                    cache_disk->drop_snapshots_after(slot.chat_id, n_past, slot.task->tokens);
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    slot.mem.seq_rm(slot.id, p0, -1);

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // The batch filling, as a function of the batch's start value of do_checkpoint and of the
                    // checkpoint list the batch sees (the first batch: the real list; the later batches of one call: the list as it
                    // will be once the checkpoints before them are made) - the usual rules unchanged
                    const bool do_ckpt_start = do_checkpoint;
                    const auto fill_batch = [&](const bool do_ckpt, const bool ckpts_empty, const int64_t ckpt_last) {
                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next(),
                            /* output    = */ slot.need_embd(),
                            /* is_prompt = */ true);
                        slot.prompt.tokens.push_back(cur_tok);

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_ckpt && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();

                            if (pos == last_user_pos || ckpts_empty || pos > ckpt_last + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_ckpt) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }
                    };
                    fill_batch(do_ckpt_start, slot.prompt.checkpoints.empty(),
                            slot.prompt.checkpoints.empty() ? 0 : slot.prompt.checkpoints.back().n_tokens);

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed (by this first batch; the slot's state is set below, after
                    // the rest of a one-call prompt is added)
                    const bool done_first = slot.prompt.n_tokens() == slot.task->n_tokens();
                    if (!done_first) {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    // The near-duplicate. When the cached part ends a few tokens before the last user message (the
                    // turn separator after the previous reply), this batch covers only those tokens and ends at the user message's
                    // start, where the next batch makes a checkpoint anyway (the last user message is exempt from the spacing
                    // rule; the next token is text, so no media chunk skips it). A checkpoint at this batch's start would be the
                    // same state minus those few tokens: it is the oldest of the three a message makes, so a small
                    // --ctx-checkpoints drops it again before the request ends, and it would only serve a later request that
                    // differs inside the separator itself. Not made (LLAMA_SERVER_CKPT_NEAR_DUP=1 makes it, as before)
                    if (do_checkpoint && !ckpt_near_dup && n_tokens_cur <= 16 &&
                            last_user_pos >= 0 && slot.prompt.n_tokens() == last_user_pos &&
                            slot.prompt.n_tokens() < slot.task->n_tokens() &&
                            input_tokens[slot.prompt.n_tokens()] != LLAMA_TOKEN_NULL) {
                        SLT_TRC(slot, "no context checkpoint at n_tokens = %" PRId64 ": the last user message starts %d tokens later (its checkpoint is made next)\n",
                                (int64_t) n_tokens_start, (int) n_tokens_cur);
                        do_checkpoint = false;
                    }

                    // With --moe-stream-layer-major 1 the rest of the prompt joins this call. The batches the usual
                    // loop would make next are made here with the same rules (each starts from do_checkpoint's start value and sees
                    // the checkpoints made before it); their starts become piece cuts of the call, and where a batch would start
                    // with a checkpoint, the engine takes it inside the call (llama_set_call_cuts) and it is made after the decode
                    slot.cc_pos.clear();
                    slot.cc_ntok.clear();
                    slot.cc_keep.clear();
                    // (only when the rest of the prompt fits in this batch: a later batch of the usual loop starts from an empty batch,
                    // so a capped n_batch would make other batches than the usual loop does)
                    if (!done_first && call_ckpt_ok(slot) && input_tokens[slot.prompt.n_tokens()] != LLAMA_TOKEN_NULL &&
                            batch.size() + (slot.task->n_tokens() - slot.prompt.n_tokens()) <= n_batch) {
                        bool    ck_empty = slot.prompt.checkpoints.empty() && !do_checkpoint;
                        int64_t ck_last  = do_checkpoint ? (int64_t) n_tokens_start :
                                           (slot.prompt.checkpoints.empty() ? 0 : slot.prompt.checkpoints.back().n_tokens);
                        while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch &&
                                input_tokens[slot.prompt.n_tokens()] != LLAMA_TOKEN_NULL) {
                            const int64_t   s_b = slot.prompt.n_tokens();
                            const llama_pos p_b = slot.prompt.tokens.pos_next();
                            fill_batch(do_ckpt_start, ck_empty, ck_last);
                            const int64_t   e_b = slot.prompt.n_tokens();
                            if (e_b == s_b) {
                                break;
                            }
                            // the usual decision for a batch [s_b, e_b) (above): its start is not the prompt's start, so the memory's
                            // pos_min is p_b - 1 >= 0; no media chunk inside it
                            const bool near_end = slot.task->n_tokens() < e_b + n_ubatch;
                            bool d = do_ckpt_start;
                            if (e_b != slot.task->n_tokens() && !spans.is_user_start(s_b) && !near_end) {
                                d = false;
                            }
                            d = d && (ck_empty || s_b == last_user_pos || near_end || s_b > ck_last + params_base.checkpoint_min_step);
                            if (d && !ckpt_near_dup && e_b - s_b <= 16 && last_user_pos >= 0 && e_b == last_user_pos &&
                                    e_b < slot.task->n_tokens() && input_tokens[e_b] != LLAMA_TOKEN_NULL) {
                                d = false;
                            }
                            slot.cc_pos.push_back(p_b);
                            slot.cc_ntok.push_back(s_b);
                            slot.cc_keep.push_back(d ? 1 : 0);
                            if (d) {
                                ck_empty = false;
                                ck_last  = s_b;
                            }
                        }
                        // a checkpoint that --ctx-checkpoints would erase before this call's end is not taken: the last
                        // n_ctx_checkpoints of the call stay, and the last user message's start when --cache-disk-checkpoints keeps it
                        const int64_t n_keep_disk = cache_disk && cache_disk->max_snaps > 0 && !slot.task->conv_id.empty() ? last_user_pos : -1;
                        // (create_checkpoint keeps the disk one plus the newest n_ctx_checkpoints - 1 others)
                        int n_left = std::max(0, params_base.n_ctx_checkpoints);
                        for (size_t i = 0; i < slot.cc_keep.size(); ++i) {
                            if (slot.cc_keep[i] && slot.cc_ntok[i] == n_keep_disk) {
                                n_left = std::max(0, n_left - 1);
                            }
                        }
                        for (size_t i = slot.cc_keep.size(); i-- > 0; ) {
                            if (slot.cc_keep[i] && slot.cc_ntok[i] != n_keep_disk) {
                                if (n_left > 0) {
                                    n_left--;
                                } else {
                                    slot.cc_keep[i] = 0;
                                }
                            }
                        }
                    }

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        SRV_TT("S init_sampler_b", 0);
                        slot.init_sampler();
                        SRV_TT("S init_sampler_e", 0);
                    }

                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        SRV_TT("S ckpt_b", 0);
                        create_checkpoint(slot, slot.prompt.n_tokens() - n_tokens_start, pos_min, pos_max);
                        SRV_TT("S ckpt_e", 0);
                    }

                    if (!slot.cc_pos.empty()) {   // armed right before each decode attempt (arm_call_cuts)
                        int n_kept = 0;
                        for (const uint8_t k : slot.cc_keep) {
                            n_kept += k ? 1 : 0;
                        }
                        SLT_INF(slot, "one call of %" PRId64 " tokens (%d pieces cut at the usual batch starts), %d checkpoints taken inside it\n",
                                (int64_t) (slot.prompt.n_tokens() - n_tokens_start), (int) slot.cc_pos.size(), n_kept);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;
        for (int i = off; i < off + batch.view.size(); ++i) {
            has_output |= batch.tokens[i].output;
        }

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        queue_tasks.yield_to_queue([&]() {
            SRV_TT("S process_b", batch.view.size());
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            SRV_TT("S process_e", ret);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
                SRV_TT("S sync_e", 0);
            }
        });

        if (ret == 3) {   // the engine could not keep the call's context - its prompt is processed again from the start
            for (auto & slot : slots) {
                if (slot.is_processing() && (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT)) {
                    SLT_WRN(slot, "%s", "the engine ended this call (a layer-major call failed over a sliding window, or VRAM lending found the VRAM "
                            "full) - the prompt is processed again from its start\n");
                    slot.prompt_clear();
                    slot.state   = SLOT_STATE_STARTED;
                    slot.i_batch = -1;
                    slot.stats   = {};   // (the prompt's timers and counts start again with it)
                }
            }
            throw server_reprocess();
        }

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch.view.size(), has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (spec) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch.view);
            });

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off) {
        // for checking if a given batch index is inside the current sub-batch
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                // n_predict 0: the prompt is evaluated into the cache and nothing is generated
                if (slot.n_predict_max == 0) {
                    slot.stats.update_prompt_last();
                    slot.stop           = STOP_TYPE_LIMIT;
                    slot.has_next_token = false;
                    slot.i_batch        = -1;
                    slot.print_timings();
                    send_final_response(slot);
                    cache_disk_save(slot);
                    slot.release();
                    if (std::none_of(slots.begin(), slots.end(), [](const server_slot & s) { return s.is_processing(); })) {
                        llama_moe_stream_idle(ctx_tgt);   // a reply's end and no other reply running (elastic RAM's take-back)
                    }
                    return;
                }

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                SRV_TT("S sample_b", 0);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
                SRV_TT("S sample_e", 0);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
                SRV_TT("S first_token", (long long) slot.stats.t_prompt_last);
                g_srv_tt = false;
            }

            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                cache_disk_save(slot);
                slot.release();
                if (std::none_of(slots.begin(), slots.end(), [](const server_slot & s) { return s.is_processing(); })) {
                    llama_moe_stream_idle(ctx_tgt);   // a reply's end and no other reply running (elastic RAM's take-back)
                }

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // batch indices of the draft tokens, used to get the token probabilities below
            auto spec_i_batch = std::move(slot.spec_i_batch);

            // verify and try to accept the draft
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(spec_i_batch.size() == n_draft + 1);
                GGML_ASSERT(slot.spec_draft_q.empty() || (slot.spec_draft_q.size() == slot.spec_draft.size()));
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());

                // drafters that fill no distribution fall back here
                const bool use_rejection = slot.use_spec_rejection() && !slot.spec_draft_q.empty();

                std::vector<llama_token> accepted;
                if (!synth_probs.empty()) {
                    // synthetic acceptance replaces verification entirely, so it comes first
                    accepted = server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                } else if (slot.spec_is_replay && slot.use_spec_rejection()) {
                    accepted = server_accept_replay(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft);
                } else if (use_rejection) {
                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft, slot.spec_draft_q);
                } else {
                    accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft);
                }

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                // the recurrent snapshots are only guaranteed for batches of up to n_rs_seq + 1 tokens, so a longer draft uses its checkpoint
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_draft > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        GGML_ASSERT(ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                        if (slot.ctx_dft) {
                            GGML_ASSERT(ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                        }

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // post_sampling_probs is not supported with speculative decoding
                if (slot.task->params.sampling.n_probs > 0 && !slot.task->params.post_sampling_probs) {
                    populate_token_probs(slot, result, false, params_base.special, spec_i_batch[i]);
                }

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    cache_disk_save(slot);
                    slot.release();
                    if (std::none_of(slots.begin(), slots.end(), [](const server_slot & s) { return s.is_processing(); })) {
                        llama_moe_stream_idle(ctx_tgt);   // only when no reply is running
                    }

                    return;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model
    int n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return std::min(res, llama_model_n_ctx_train(model_tgt));
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info              */ std::string(llama_build_info()),
        /* model_name              */ impl->model_name,
        /* model_aliases           */ impl->model_aliases,
        /* model_tags              */ impl->model_tags,
        /* model_path              */ impl->params_base.model.path,
        /* model_output_modalities */ server_model_output_modalities(),
        /* has_mtmd                */ impl->mctx != nullptr,
        /* has_inp_image           */ impl->chat_params.allow_image,
        /* has_inp_audio           */ impl->chat_params.allow_audio,
        /* has_inp_video           */ impl->chat_params.allow_video,
        /* json_ui_settings        */ impl->json_ui_settings,
        /* slot_n_ctx              */ impl->n_ctx_slot(),
        /* pooling_type            */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params             */ impl->chat_params,
        /* chat_template_caps      */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str           */ bos_token_str,
        /* eos_token_str           */ eos_token_str,
        /* fim_pre_token           */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token           */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token           */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token           */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token           */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token           */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog          */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type        */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens    */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train       */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp        */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params          */ llama_model_n_params(impl->model_tgt),
        /* model_size              */ llama_model_size(impl->model_tgt),
        /* model_ftype             */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const common_chat_session & chat_session) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.apply_chat_session(chat_session);

            task.id_slot = json_value(data, "id_slot", -1);
            if (!params.cache_disk_dir.empty()) {
                task.conv_id = server_stream_conv_id_from_headers(req.headers);   // the --cache-disk chat key
            }
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks), chat_session);
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & usage   = arr[0]["usage"];
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }

                // OAI counts generated tokens across all choices
                int64_t n_completion_tokens = 0;
                for (auto & item : arr) {
                    n_completion_tokens += item["usage"]["completion_tokens"].get<int64_t>();
                }
                usage["completion_tokens"] = n_completion_tokens;
                usage["total_tokens"]      = usage["prompt_tokens"].get<int64_t>() + n_completion_tokens;

                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        // The first wait also ends when the request's stream session is cancelled (DELETE /v1/stream for its
        // X-Conversation-Id - an app leaving the chat): the reader is destroyed on return, which cancels the task, so a request
        // still in the queue is removed at once instead of running its prompt's first batch later. A resumable stream's dropped
        // connection is unchanged (the same req.should_stop as before)
        auto * res_cancel = res.get();
        const auto first_stop = [&req, res_cancel]() {
            return req.should_stop() || res_cancel->stream_cancelled();
        };
        auto first_result = rd.next(first_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(first_stop());
            if (!req.should_stop()) {
                // the client still listens: say why the stream ends without output
                SRV_INF("%s", "request cancelled before its first result (its stream session was cancelled)\n");
                res->error(format_error_response("cancelled: the conversation's stream was stopped (DELETE /v1/stream) or replaced by a newer request of the same conversation", ERROR_TYPE_INVALID_REQUEST));
            }
            return res; // connection is closed, or the stream session was cancelled
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });

    // A cancelled stream session (DELETE /v1/stream, an app's switch) wakes the requests waiting for results, so
    // one that still waits for its first result (in the queue, or before its prompt's first batch) sees the cancel at once
    server_response * qr = &queue_results;
    server_stream_set_cancel_hook([qr]() { qr->wake_waiters(); });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"architecture", server_model_architecture_json(
            meta.has_inp_image,
            meta.has_inp_audio,
            meta.has_inp_video,
            meta.model_output_modalities)},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    // GET /moe-stream/stats: the streaming engine's figures now, readable at any time (also while a request runs, and it does not wake
    // a sleeping server)
    this->get_moe_stream_stats = [this](const server_http_req &) {
        auto res = create_response(true);
        int status = 2;
        json data = ctx_server.moe_stats_json(status);
        if (status == 1) {
            res->error(format_error_response("This server does not stream experts from the SSD. Start it with `--moe-stream`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        if (status == 2) {
            res->error(format_error_response("No model is loaded right now (the server is asleep)", ERROR_TYPE_UNAVAILABLE));
            return res;
        }
        res->ok(data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    // POST /cache-disk/erase (the chat in X-Conversation-Id), POST /cache-disk/clear, POST /cache-disk/pins and POST /cache-disk/fork
    this->post_cache_disk_erase = [this](const server_http_req & req) {
        return handle_cache_disk(req, true);
    };

    this->post_cache_disk_clear = [this](const server_http_req & req) {
        return handle_cache_disk(req, false);
    };

    this->post_cache_disk_pins = [this](const server_http_req & req) {
        return handle_cache_disk_pins(req);
    };

    this->post_cache_disk_fork = [this](const server_http_req & req) {
        return handle_cache_disk_fork(req);
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT,
            session);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP,
            session);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR,
            session);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC,
            session);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        common_chat_session session;   // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i], ctx_server.init_opt);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

// A localhost origin (as server-http.cpp's origin_is_localhost: localhost, 127.0.0.1, ::1).
// An Origin header as a browser sends it - "scheme://host[:port]", http or https, no user part, no path, a
// port of digits only (so "http://localhost:8080.evil" is not localhost); false = not a valid origin
static bool cd_origin_parse(const std::string & origin_in, std::string & norm, std::string & host) {
    std::string o = origin_in;
    std::transform(o.begin(), o.end(), o.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    while (!o.empty() && o.back() == '/') {
        o.pop_back();
    }
    const size_t p = o.find("://");
    if (p == std::string::npos || (o.compare(0, p, "http") != 0 && o.compare(0, p, "https") != 0)) {
        return false;
    }
    const std::string auth = o.substr(p + 3);
    if (auth.empty() || auth.find_first_of("/@?#\\ ") != std::string::npos) {
        return false;
    }
    std::string port;
    if (auth[0] == '[') {
        const size_t e = auth.find(']');
        if (e == std::string::npos) {
            return false;
        }
        host = auth.substr(1, e - 1);
        const std::string after = auth.substr(e + 1);
        if (!after.empty()) {
            if (after[0] != ':') {
                return false;
            }
            port = after.substr(1);
            if (port.empty()) {
                return false;
            }
        }
    } else {
        const size_t c = auth.find(':');
        host = auth.substr(0, c);
        if (c != std::string::npos) {
            port = auth.substr(c + 1);
            if (port.empty()) {
                return false;
            }
        }
    }
    if (host.empty() || (!port.empty() && (port.size() > 5 || port.find_first_not_of("0123456789") != std::string::npos))) {
        return false;
    }
    norm = o;
    return true;
}

// without --api-key: a localhost page, or an origin the user listed in --cors-origins (when it is a list, not "*" / "localhost")
static bool cd_origin_allowed(const std::string & origin, const std::string & cors_origins) {
    std::string norm, host;
    if (!cd_origin_parse(origin, norm, host)) {
        return false;
    }
    if (host == "localhost" || host == "127.0.0.1" || host == "::1") {
        return true;
    }
    if (cors_origins == "*" || cors_origins == "localhost") {
        return false;
    }
    size_t i = 0;
    while (i <= cors_origins.size()) {
        size_t j = cors_origins.find(',', i);
        if (j == std::string::npos) {
            j = cors_origins.size();
        }
        std::string item = cors_origins.substr(i, j - i);
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        std::string item_norm, item_host;
        if (!item.empty() && cd_origin_parse(item, item_norm, item_host) && item_norm == norm) {
            return true;
        }
        i = j + 1;
    }
    return false;
}

// the Origin header of a request ("" = none)
static std::string cd_request_origin(const server_http_req & req) {
    std::string origin;
    for (const auto & [k, v] : req.headers) {
        std::string kl = k;
        std::transform(kl.begin(), kl.end(), kl.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        if (kl == "origin") {
            origin = v;
        }
    }
    return origin;
}

std::unique_ptr<server_res_generator> server_routes::handle_cache_disk(const server_http_req & req, bool erase) {
    // never wakes a sleeping model (--sleep-idle-seconds): "not supported" and a missing chat id are answered at once, and
    // while the server sleeps the files are deleted here, under the task queue's lock (the slots and the RAM prompt cache are
    // rebuilt empty when it wakes, so there is nothing else to delete)
    auto res = create_response(true);
    if (params.cache_disk_dir.empty()) {
        res->error(format_error_response("This server does not keep chat states on disk. Start it with `--cache-disk DIR`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }
    // Without --api-key, a web page could send these (llama-server answers any origin by default: CORS "*", and
    // it reflects the request's Origin), so a request that carries a browser Origin which is not localhost - or
    // one the user listed in --cors-origins - is refused; a request without an Origin (the app's backend, curl) and a
    // localhost page work. With --api-key the key protects them
    if (params.api_keys.empty()) {
        const std::string origin = cd_request_origin(req);
        if (!origin.empty() && !cd_origin_allowed(origin, params.cors_origins)) {
            SRV_WRN("cache-disk: refused POST /cache-disk/%s from origin '%s' (not localhost, not listed in --cors-origins, and the "
                    "server has no --api-key)\n", erase ? "erase" : "clear", origin.c_str());
            res->error(format_error_response("POST /cache-disk/erase and /clear are refused from a web page that is not on localhost "
                    "or listed in --cors-origins when the server has no --api-key", ERROR_TYPE_PERMISSION));
            return res;
        }
    }
    const std::string chat = erase ? server_stream_conv_id_from_headers(req.headers) : std::string();
    if (erase && chat.empty()) {
        res->error(format_error_response("POST /cache-disk/erase needs the chat id in the X-Conversation-Id header", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    {
        json data;
        bool failed = false;
        const bool asleep = queue_tasks.run_while_sleeping([&]() {
            server_cache_disk * cd = ctx_server.cache_disk_while_sleeping();
            if (!cd) {
                failed = true;
                return;
            }
            if (erase) {
                const int64_t n_bytes = cd->erase_chat(chat);
                failed = n_bytes < 0;
                if (!failed) {
                    server_stream_drop_session(chat);   // only after the files are gone
                }
                data = { { "erased_file", n_bytes > 0 }, { "file_bytes", n_bytes }, { "ram_states", 0 }, { "slots", 0 },
                         { "slots_answering", 0 }, { "asleep", true } };
            } else {
                const auto [n_files, n_bytes] = cd->clear();
                server_stream_drop_all_sessions();
                data = { { "files", n_files }, { "file_bytes", n_bytes }, { "ram_states", 0 }, { "slots", 0 },
                         { "slots_answering", 0 }, { "asleep", true } };
            }
            SRV_INF("cache-disk: %s while the server sleeps (the model is not loaded for it)\n", erase ? "erase" : "clear");
        });
        if (asleep) {
            if (failed) {
                res->error(format_error_response("the state could not be deleted, or the disk tier is off (see the server log)", ERROR_TYPE_SERVER));
            } else {
                res->ok(data);
            }
            return res;
        }
    }
    queue_tasks.wait_until_no_sleep();   // not sleeping, or a wake is already pending: the task runs on the main loop
    auto & rd = res->rd;
    {
        server_task task(erase ? SERVER_TASK_TYPE_CACHE_DISK_ERASE : SERVER_TASK_TYPE_CACHE_DISK_CLEAR);
        task.conv_id = chat;
        task.id = rd.get_new_id();
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_cache_disk*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

// POST /cache-disk/pins {"pinned": ["<chat id>", ...]}: replaces the set of chats whose saved states the size limit deletes last.
// Answered at once, also while the server sleeps (the set is kept by the server, not the disk tier)
std::unique_ptr<server_res_generator> server_routes::handle_cache_disk_pins(const server_http_req & req) {
    auto res = create_response(true);
    if (params.cache_disk_dir.empty()) {
        res->error(format_error_response("This server does not keep chat states on disk. Start it with `--cache-disk DIR`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }
    if (params.api_keys.empty()) {   // as for erase / clear: a web page that is not on localhost is refused
        const std::string origin = cd_request_origin(req);
        if (!origin.empty() && !cd_origin_allowed(origin, params.cors_origins)) {
            SRV_WRN("cache-disk: refused POST /cache-disk/pins from origin '%s' (not localhost, not listed in --cors-origins, and the "
                    "server has no --api-key)\n", origin.c_str());
            res->error(format_error_response("POST /cache-disk/pins is refused from a web page that is not on localhost "
                    "or listed in --cors-origins when the server has no --api-key", ERROR_TYPE_PERMISSION));
            return res;
        }
    }
    std::vector<std::string> ids;
    try {
        const json body = json::parse(req.body);
        if (!body.is_object() || !body.contains("pinned") || !body.at("pinned").is_array()) {
            throw std::runtime_error("no pinned list");
        }
        for (const auto & v : body.at("pinned")) {
            if (!v.is_string()) {
                throw std::runtime_error("a pinned chat id is not a string");
            }
            ids.push_back(v.get<std::string>());
        }
    } catch (const std::exception &) {
        res->error(format_error_response("POST /cache-disk/pins needs a JSON body {\"pinned\": [\"<chat id>\", ...]}", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    const size_t n = ctx_server.cache_disk_pinned().set(ids);
    SRV_INF("cache-disk: %zu chats pinned (their files are deleted last by the size limit)\n", n);
    res->ok(json { { "pinned", n } });
    return res;
}

// POST /cache-disk/fork {"from": "<chat id>", "to": "<chat id>"}: every saved entry of chat `from` (its state and snapshots)
// also becomes an entry of chat `to`, replacing the entries `to` had. Runs as a task on the main loop (after any save in
// progress), or directly while the server sleeps (the model is not loaded for it)
std::unique_ptr<server_res_generator> server_routes::handle_cache_disk_fork(const server_http_req & req) {
    auto res = create_response(true);
    if (params.cache_disk_dir.empty()) {
        res->error(format_error_response("This server does not keep chat states on disk. Start it with `--cache-disk DIR`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }
    if (params.api_keys.empty()) {   // as for erase / clear: a web page that is not on localhost is refused
        const std::string origin = cd_request_origin(req);
        if (!origin.empty() && !cd_origin_allowed(origin, params.cors_origins)) {
            SRV_WRN("cache-disk: refused POST /cache-disk/fork from origin '%s' (not localhost, not listed in --cors-origins, and the "
                    "server has no --api-key)\n", origin.c_str());
            res->error(format_error_response("POST /cache-disk/fork is refused from a web page that is not on localhost "
                    "or listed in --cors-origins when the server has no --api-key", ERROR_TYPE_PERMISSION));
            return res;
        }
    }
    std::string from;
    std::string to;
    try {
        const json body = json::parse(req.body);
        if (!body.is_object() || !body.contains("from") || !body.at("from").is_string() || !body.contains("to") || !body.at("to").is_string()) {
            throw std::runtime_error("no from / to");
        }
        from = body.at("from").get<std::string>();
        to   = body.at("to").get<std::string>();
        if (from.empty() || to.empty()) {
            throw std::runtime_error("an empty chat id");
        }
    } catch (const std::exception &) {
        res->error(format_error_response("POST /cache-disk/fork needs a JSON body {\"from\": \"<chat id>\", \"to\": \"<chat id>\"} "
                "with two chat ids that are not empty", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    if (from == to) {
        res->error(format_error_response("POST /cache-disk/fork: \"from\" and \"to\" are the same chat id", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    {
        json data;
        bool failed = false;
        server_cache_disk::fork_result fr;
        const bool asleep = queue_tasks.run_while_sleeping([&]() {
            server_cache_disk * cd = ctx_server.cache_disk_while_sleeping();
            if (!cd) {
                failed = true;
                return;
            }
            fr = cd->fork_chat(from, to);
            if (fr.ok()) {
                server_stream_drop_session(to);   // only after the copy is made
                data = fr.to_json();
                data["ram_states"] = 0;
                data["slots"] = 0;
                data["slots_answering"] = 0;
                data["asleep"] = true;
            }
            SRV_INF("%s", "cache-disk: fork while the server sleeps (the model is not loaded for it)\n");
        });
        if (asleep) {
            if (failed) {
                res->error(format_error_response("the disk tier is off", ERROR_TYPE_SERVER));
            } else if (!fr.ok()) {
                res->error(format_error_response(fr.error_text(), fr.found ? ERROR_TYPE_SERVER : ERROR_TYPE_INVALID_REQUEST));
            } else {
                res->ok(data);
            }
            return res;
        }
    }
    queue_tasks.wait_until_no_sleep();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_CACHE_DISK_FORK);
        task.conv_id      = to;
        task.conv_id_from = from;
        task.id = rd.get_new_id();
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_cache_disk*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    // same shapes as tokenize_input_prompts(), plus OAI content: { "content": [ { "type": "text"|"image_url"|"input_audio"|"input_video", ... } ] }
    auto tokenize_entry = [&](const json & p) {
        if (p.is_object() && p.contains("content")) {
            return tokenize_oai_content_array(ctx_server.vocab, ctx_server.mctx, meta->chat_params, p.at("content"), true, true, ctx_server.init_opt);
        }
        return tokenize_input_subprompt(ctx_server.vocab, ctx_server.mctx, p, true, true, ctx_server.init_opt);
    };

    std::vector<server_tokens> tokenized_prompts;
    if (prompt.is_array() && !json_is_array_and_contains_numbers(prompt)) {
        for (const auto & p : prompt) {
            tokenized_prompts.push_back(tokenize_entry(p));
        }
    } else {
        tokenized_prompts.push_back(tokenize_entry(prompt));
    }
    if (tokenized_prompts.empty()) {
        res->error(format_error_response("\"input\" must not be empty", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    common_chat_session session; // dummy, unused
    json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (ctx_server.mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt, true).size();
    } else {
        n_tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}

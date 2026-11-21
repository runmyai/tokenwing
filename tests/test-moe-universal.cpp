// Universal MoE Streaming, 2x2 Guesser Census & Guess-Gate Parity Suite
// Synthesizes 8 distinct MoE GGUF topologies on the fly using public gguf/llama APIs
// and verifies bit-exact greedy token & logit parity, 0 hangs, and clean fallbacks.

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct gguf_builder {
    ggml_context * gctx = nullptr;
    gguf_context * uctx = nullptr;
    std::vector<std::vector<uint8_t>> blobs;

    explicit gguf_builder(uint32_t align_bytes = 4096, size_t max_tensors = 512) {
        ggml_init_params ip = { ggml_tensor_overhead() * max_tensors, nullptr, true };
        gctx = ggml_init(ip);
        if (align_bytes == 32) {
            uctx = gguf_init_empty();
        } else {
            gguf_context * seed = gguf_init_empty();
            gguf_set_val_u32(seed, "general.alignment", align_bytes);
            const std::string sp = (fs::temp_directory_path() / ("tw_align_seed_" + std::to_string(align_bytes) + ".gguf")).string();
            gguf_write_to_file(seed, sp.c_str(), true);
            gguf_free(seed);
            gguf_init_params up = { true, nullptr };
            uctx = gguf_init_from_file(sp.c_str(), up);
            std::remove(sp.c_str());
        }
    }

    ~gguf_builder() {
        if (uctx) { gguf_free(uctx); }
        if (gctx) { ggml_free(gctx); }
    }

    void add_vocab(uint32_t n_vocab) {
        gguf_set_val_str(uctx, "tokenizer.ggml.model", "llama");
        std::vector<std::string> toks(n_vocab);
        std::vector<const char *> ptrs(n_vocab);
        std::vector<float> scores(n_vocab, 0.0f);
        std::vector<int32_t> types(n_vocab, 1);
        for (uint32_t i = 0; i < n_vocab; i++) {
            toks[i] = "t" + std::to_string(i);
            ptrs[i] = toks[i].c_str();
        }
        types[0] = 3; types[1] = 3; types[2] = 3;
        gguf_set_arr_str(uctx, "tokenizer.ggml.tokens", ptrs.data(), n_vocab);
        gguf_set_arr_data(uctx, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), n_vocab);
        gguf_set_arr_data(uctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), n_vocab);
        gguf_set_val_u32(uctx, "tokenizer.ggml.bos_token_id", 1);
        gguf_set_val_u32(uctx, "tokenizer.ggml.eos_token_id", 2);
    }

    void add_f32_1d(const std::string & name, int64_t n0, uint32_t seed, float scale = 0.08f, bool ones = false, bool ramp = false) {
        ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, n0);
        ggml_set_name(t, name.c_str());
        blobs.emplace_back((size_t) n0 * sizeof(float));
        float * p = (float *) blobs.back().data();
        std::mt19937 rng(seed ^ (uint32_t) std::hash<std::string>{}(name));
        std::normal_distribution<float> dist(0.0f, scale);
        for (int64_t i = 0; i < n0; i++) {
            p[i] = ones ? (1.0f + 0.02f * dist(rng)) : ramp ? (0.15f * (float) i) : dist(rng);
        }
        t->data = p;
        gguf_add_tensor(uctx, t);
    }

    void add_f32_2d(const std::string & name, int64_t n0, int64_t n1, uint32_t seed, float scale = 0.06f, bool norm_ones = false) {
        ggml_tensor * t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n0, n1);
        ggml_set_name(t, name.c_str());
        const int64_t ne = n0 * n1;
        blobs.emplace_back((size_t) ne * sizeof(float));
        float * p = (float *) blobs.back().data();
        std::mt19937 rng(seed ^ (uint32_t) std::hash<std::string>{}(name));
        std::normal_distribution<float> dist(0.0f, scale);
        for (int64_t i = 0; i < ne; i++) {
            p[i] = norm_ones ? (1.0f + 0.02f * dist(rng)) : dist(rng);
        }
        if (name.find("gate_inp.weight") != std::string::npos) {
            // Break ties deterministically across experts
            for (int64_t r = 0; r < n1; r++) {
                p[r * n0 + (r % n0)] += 0.35f + 0.03f * (float) r;
            }
        }
        t->data = p;
        gguf_add_tensor(uctx, t);
    }

    void add_f16_3d(const std::string & name, int64_t n0, int64_t n1, int64_t n2, uint32_t seed, float scale = 0.05f) {
        ggml_tensor * t = ggml_new_tensor_3d(gctx, GGML_TYPE_F16, n0, n1, n2);
        ggml_set_name(t, name.c_str());
        const int64_t ne = n0 * n1 * n2;
        blobs.emplace_back((size_t) ne * sizeof(ggml_fp16_t));
        ggml_fp16_t * p = (ggml_fp16_t *) blobs.back().data();
        std::mt19937 rng(seed ^ (uint32_t) std::hash<std::string>{}(name));
        std::normal_distribution<float> dist(0.0f, scale);
        for (int64_t i = 0; i < ne; i++) {
            p[i] = ggml_fp32_to_fp16(dist(rng));
        }
        t->data = p;
        gguf_add_tensor(uctx, t);
    }

    void write(const std::string & path) {
        if (!gguf_write_to_file(uctx, path.c_str(), false)) {
            throw std::runtime_error("gguf_write_to_file failed: " + path);
        }
    }
};

struct run_result {
    std::vector<float>       first_logits;
    std::vector<llama_token> gen_tokens;
    llama_moe_stream_stats   stats = {};
    bool                     has_stats = false;
};

static run_result eval_model(const std::string & path, const llama_model_params & mparams, int n_gen = 8) {
    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (!model) {
        throw std::runtime_error("llama_model_load_from_file failed for " + path);
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 64;
    cparams.n_batch         = 32;
    cparams.n_ubatch        = 32;
    cparams.n_threads       = 2;
    cparams.n_threads_batch = 2;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        throw std::runtime_error("llama_init_from_model failed for " + path);
    }

    std::vector<llama_token> prompt = { 1, 5, 9, 13, 17, 21 };
    llama_batch batch = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
    if (llama_decode(ctx, batch) != 0) {
        llama_free(ctx);
        llama_model_free(model);
        throw std::runtime_error("prompt llama_decode failed for " + path);
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits0 = llama_get_logits_ith(ctx, -1);
    run_result out;
    out.first_logits.assign(logits0, logits0 + n_vocab);

    auto greedy_pick = [&](const float * lg) -> llama_token {
        llama_token best = 0;
        for (int32_t v = 1; v < n_vocab; v++) {
            if (lg[v] > lg[best]) { best = v; }
        }
        return best;
    };

    llama_token cur_tok = greedy_pick(logits0);
    out.gen_tokens.push_back(cur_tok);
    for (int step = 1; step < n_gen; step++) {
        llama_batch b1 = llama_batch_get_one(&cur_tok, 1);
        if (llama_decode(ctx, b1) != 0) {
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("decode step failed for " + path);
        }
        cur_tok = greedy_pick(llama_get_logits_ith(ctx, -1));
        out.gen_tokens.push_back(cur_tok);
    }

    out.has_stats = llama_moe_stream_get_stats(ctx, &out.stats);
    llama_free(ctx);
    llama_model_free(model);
    return out;
}

static void check_parity(const char * label, const run_result & ref, const run_result & got) {
    if (ref.first_logits.size() != got.first_logits.size()) {
        throw std::runtime_error(std::string(label) + ": logit size mismatch");
    }
    double max_abs = 0.0;
    for (size_t i = 0; i < ref.first_logits.size(); i++) {
        if (!std::isfinite(got.first_logits[i])) {
            throw std::runtime_error(std::string(label) + ": non-finite logit");
        }
        max_abs = std::max(max_abs, (double) std::fabs(ref.first_logits[i] - got.first_logits[i]));
    }
    // Expert weights are F16 (1 ULP ~= 9.8e-4), so CUDA cuBLAS FP16 vs CPU SIMD FP32 accumulation can differ by ~0.6 F16 ULP.
    if (max_abs > 2e-3) {
        throw std::runtime_error(std::string(label) + ": logit max_abs=" + std::to_string(max_abs) + " > 2e-3");
    }
    if (ref.gen_tokens != got.gen_tokens) {
        throw std::runtime_error(std::string(label) + ": generated greedy token sequence mismatch");
    }
}

static void build_qwen3moe_gguf(const std::string & path, uint32_t align_bytes = 4096, int shard_mode = 0, const std::string & path2 = "") {
    const uint32_t n_vocab = 64, n_embd = 64, n_head = 2, n_ff_exp = 64, n_layer = 4, n_exp = 8, n_used = 2;
    const uint32_t head_k  = n_embd / n_head;

    auto populate = [&](gguf_builder & b, int shard_idx) {
        gguf_set_val_str(b.uctx, "general.architecture", "qwen3moe");
        gguf_set_val_u32(b.uctx, "general.alignment", align_bytes);
        if (align_bytes == 32) {
            gguf_set_val_str(b.uctx, "general.description", "unaligned-32b-padding-test-blob-1234567");
        }
        if (shard_idx >= 0) {
            gguf_set_val_u16(b.uctx, "split.no", (uint16_t) shard_idx);
            gguf_set_val_u16(b.uctx, "split.count", 2);
            gguf_set_val_i32(b.uctx, "split.tensors.count", 3 + (int32_t) n_layer * 12);
        }
        b.add_vocab(n_vocab);
        gguf_set_val_u32(b.uctx, "qwen3moe.context_length", 128);
        gguf_set_val_u32(b.uctx, "qwen3moe.embedding_length", n_embd);
        gguf_set_val_u32(b.uctx, "qwen3moe.block_count", n_layer);
        gguf_set_val_u32(b.uctx, "qwen3moe.feed_forward_length", n_ff_exp * n_used);
        gguf_set_val_u32(b.uctx, "qwen3moe.expert_feed_forward_length", n_ff_exp);
        gguf_set_val_u32(b.uctx, "qwen3moe.attention.head_count", n_head);
        gguf_set_val_u32(b.uctx, "qwen3moe.attention.head_count_kv", n_head);
        gguf_set_val_f32(b.uctx, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-5f);
        gguf_set_val_u32(b.uctx, "qwen3moe.expert_count", n_exp);
        gguf_set_val_u32(b.uctx, "qwen3moe.expert_used_count", n_used);

        if (shard_idx <= 0) {
            b.add_f32_2d("token_embd.weight", n_embd, n_vocab, 101);
        }
        for (uint32_t il = 0; il < n_layer; il++) {
            if (shard_idx == 0 && il >= 2) { continue; }
            if (shard_idx == 1 && il <  2) { continue; }
            const std::string p = "blk." + std::to_string(il) + ".";
            b.add_f32_1d(p + "attn_norm.weight",   n_embd, 200 + il, 0.02f, true);
            b.add_f32_2d(p + "attn_q.weight",      n_embd, n_embd, 210 + il);
            b.add_f32_2d(p + "attn_k.weight",      n_embd, n_embd, 220 + il);
            b.add_f32_2d(p + "attn_v.weight",      n_embd, n_embd, 230 + il);
            b.add_f32_2d(p + "attn_output.weight", n_embd, n_embd, 240 + il);
            b.add_f32_1d(p + "attn_q_norm.weight", head_k, 250 + il, 0.02f, true);
            b.add_f32_1d(p + "attn_k_norm.weight", head_k, 260 + il, 0.02f, true);
            b.add_f32_1d(p + "ffn_norm.weight",    n_embd, 270 + il, 0.02f, true);
            b.add_f32_2d(p + "ffn_gate_inp.weight", n_embd, n_exp, 280 + il, 0.12f);
            b.add_f16_3d(p + "ffn_gate_exps.weight", n_embd, n_ff_exp, n_exp, 290 + il);
            b.add_f16_3d(p + "ffn_up_exps.weight",   n_embd, n_ff_exp, n_exp, 300 + il);
            b.add_f16_3d(p + "ffn_down_exps.weight", n_ff_exp, n_embd, n_exp, 310 + il);
        }
        if (shard_idx != 0) {
            b.add_f32_1d("output_norm.weight", n_embd, 401, 0.02f, true);
            b.add_f32_2d("output.weight",      n_embd, n_vocab, 402);
        }
    };

    if (shard_mode == 0) {
        gguf_builder b(align_bytes);
        populate(b, -1);
        b.write(path);
    } else {
        gguf_builder b0(align_bytes), b1(align_bytes);
        populate(b0, 0);
        populate(b1, 1);
        b0.write(path);
        b1.write(path2);
    }
}

static void build_interleaved_llama4_gguf(const std::string & path) {
    const uint32_t n_vocab = 64, n_embd = 64, n_head = 2, n_ff = 96, n_ff_exp = 64, n_layer = 6, n_exp = 8, n_used = 2;
    gguf_builder b;
    gguf_set_val_str(b.uctx, "general.architecture", "llama4");
    gguf_set_val_u32(b.uctx, "general.alignment", 4096);
    b.add_vocab(n_vocab);
    gguf_set_val_u32(b.uctx, "llama4.context_length", 128);
    gguf_set_val_u32(b.uctx, "llama4.embedding_length", n_embd);
    gguf_set_val_u32(b.uctx, "llama4.block_count", n_layer);
    gguf_set_val_u32(b.uctx, "llama4.feed_forward_length", n_ff);
    gguf_set_val_u32(b.uctx, "llama4.expert_feed_forward_length", n_ff_exp);
    gguf_set_val_u32(b.uctx, "llama4.interleave_moe_layer_step", 2);
    gguf_set_val_u32(b.uctx, "llama4.attention.head_count", n_head);
    gguf_set_val_u32(b.uctx, "llama4.attention.head_count_kv", n_head);
    gguf_set_val_f32(b.uctx, "llama4.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(b.uctx, "llama4.expert_count", n_exp);
    gguf_set_val_u32(b.uctx, "llama4.expert_used_count", n_used);

    b.add_f32_2d("token_embd.weight", n_embd, n_vocab, 501);
    for (uint32_t il = 0; il < n_layer; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const bool is_moe = ((il + 1) % 2 == 0);   // layers 1, 3, 5 MoE; 0, 2, 4 dense!
        b.add_f32_1d(p + "attn_norm.weight",   n_embd, 510 + il, 0.02f, true);
        b.add_f32_2d(p + "attn_q.weight",      n_embd, n_embd, 520 + il);
        b.add_f32_2d(p + "attn_k.weight",      n_embd, n_embd, 530 + il);
        b.add_f32_2d(p + "attn_v.weight",      n_embd, n_embd, 540 + il);
        b.add_f32_2d(p + "attn_output.weight", n_embd, n_embd, 550 + il);
        b.add_f32_1d(p + "ffn_norm.weight",    n_embd, 560 + il, 0.02f, true);
        if (is_moe) {
            b.add_f32_2d(p + "ffn_gate_inp.weight",   n_embd, n_exp, 570 + il, 0.12f);
            b.add_f16_3d(p + "ffn_gate_exps.weight",  n_embd, n_ff_exp, n_exp, 580 + il);
            b.add_f16_3d(p + "ffn_up_exps.weight",    n_embd, n_ff_exp, n_exp, 590 + il);
            b.add_f16_3d(p + "ffn_down_exps.weight",  n_ff_exp, n_embd, n_exp, 600 + il);
            b.add_f32_2d(p + "ffn_gate_shexp.weight", n_embd, n_ff_exp, 610 + il);
            b.add_f32_2d(p + "ffn_up_shexp.weight",   n_embd, n_ff_exp, 620 + il);
            b.add_f32_2d(p + "ffn_down_shexp.weight", n_ff_exp, n_embd, 630 + il);
        } else {
            b.add_f32_2d(p + "ffn_gate.weight", n_embd, n_ff, 640 + il);
            b.add_f32_2d(p + "ffn_up.weight",   n_embd, n_ff, 650 + il);
            b.add_f32_2d(p + "ffn_down.weight", n_ff, n_embd, 660 + il);
        }
    }
    b.add_f32_1d("output_norm.weight", n_embd, 690, 0.02f, true);
    b.add_f32_2d("output.weight",      n_embd, n_vocab, 691);
    b.write(path);
}

static void build_fused_2proj_qwen35moe_gguf(const std::string & path) {
    const uint32_t n_vocab = 64, n_embd = 64, n_head = 2, n_ff_exp = 64, n_layer = 4, n_exp = 8, n_used = 2;
    const uint32_t head_k  = n_embd / n_head;
    const int32_t  rope_sec[4] = { 8, 8, 8, 8 };
    gguf_builder b;
    gguf_set_val_str(b.uctx, "general.architecture", "qwen35moe");
    gguf_set_val_u32(b.uctx, "general.alignment", 4096);
    b.add_vocab(n_vocab);
    gguf_set_val_u32(b.uctx, "qwen35moe.context_length", 128);
    gguf_set_val_u32(b.uctx, "qwen35moe.embedding_length", n_embd);
    gguf_set_val_u32(b.uctx, "qwen35moe.block_count", n_layer);
    gguf_set_val_u32(b.uctx, "qwen35moe.feed_forward_length", n_ff_exp);
    gguf_set_val_u32(b.uctx, "qwen35moe.expert_feed_forward_length", n_ff_exp);
    gguf_set_val_u32(b.uctx, "qwen35moe.expert_shared_feed_forward_length", n_ff_exp);
    gguf_set_val_u32(b.uctx, "qwen35moe.attention.head_count", n_head);
    gguf_set_val_u32(b.uctx, "qwen35moe.attention.head_count_kv", n_head);
    gguf_set_val_f32(b.uctx, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(b.uctx, "qwen35moe.full_attention_interval", 1);   // all full-attn layers
    gguf_set_arr_data(b.uctx, "qwen35moe.rope.dimension_sections", GGUF_TYPE_INT32, rope_sec, 4);
    gguf_set_val_u32(b.uctx, "qwen35moe.ssm.conv_kernel", 4);
    gguf_set_val_u32(b.uctx, "qwen35moe.ssm.inner_size", n_embd);
    gguf_set_val_u32(b.uctx, "qwen35moe.ssm.state_size", 32);
    gguf_set_val_u32(b.uctx, "qwen35moe.ssm.time_step_rank", 2);
    gguf_set_val_u32(b.uctx, "qwen35moe.ssm.group_count", 2);
    gguf_set_val_u32(b.uctx, "qwen35moe.expert_count", n_exp);
    gguf_set_val_u32(b.uctx, "qwen35moe.expert_used_count", n_used);

    b.add_f32_2d("token_embd.weight", n_embd, n_vocab, 701);
    for (uint32_t il = 0; il < n_layer; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        b.add_f32_1d(p + "attn_norm.weight",      n_embd, 710 + il, 0.02f, true);
        b.add_f32_1d(p + "post_attention_norm.weight", n_embd, 715 + il, 0.02f, true);
        b.add_f32_2d(p + "attn_q.weight",         n_embd, n_embd * 2, 720 + il);
        b.add_f32_2d(p + "attn_k.weight",         n_embd, n_embd, 730 + il);
        b.add_f32_2d(p + "attn_v.weight",         n_embd, n_embd, 740 + il);
        b.add_f32_2d(p + "attn_output.weight",    n_embd, n_embd, 750 + il);
        b.add_f32_1d(p + "attn_q_norm.weight",    head_k, 760 + il, 0.02f, true);
        b.add_f32_1d(p + "attn_k_norm.weight",    head_k, 770 + il, 0.02f, true);
        b.add_f32_2d(p + "ffn_gate_inp.weight",   n_embd, n_exp, 780 + il, 0.12f);
        // 2-projection fused gate_up + down (NO separate gate_exps / up_exps!)
        b.add_f16_3d(p + "ffn_gate_up_exps.weight", n_embd, n_ff_exp * 2, n_exp, 790 + il);
        b.add_f16_3d(p + "ffn_down_exps.weight",    n_ff_exp, n_embd, n_exp, 800 + il);
        b.add_f32_1d(p + "ffn_gate_inp_shexp.weight", n_embd, 810 + il);
        b.add_f32_2d(p + "ffn_gate_shexp.weight", n_embd, n_ff_exp, 820 + il);
        b.add_f32_2d(p + "ffn_up_shexp.weight",   n_embd, n_ff_exp, 830 + il);
        b.add_f32_2d(p + "ffn_down_shexp.weight", n_ff_exp, n_embd, 840 + il);
    }
    b.add_f32_1d("output_norm.weight", n_embd, 890, 0.02f, true);
    b.add_f32_2d("output.weight",      n_embd, n_vocab, 891);
    b.write(path);
}

static void build_openai_moe_gguf(const std::string & path) {
    const uint32_t n_vocab = 64, n_embd = 64, n_head = 2, n_ff_exp = 64, n_layer = 4, n_exp = 8, n_used = 2;
    gguf_builder b;
    gguf_set_val_str(b.uctx, "general.architecture", "gpt-oss");
    gguf_set_val_u32(b.uctx, "general.alignment", 4096);
    b.add_vocab(n_vocab);
    gguf_set_val_u32(b.uctx, "gpt-oss.context_length", 128);
    gguf_set_val_u32(b.uctx, "gpt-oss.embedding_length", n_embd);
    gguf_set_val_u32(b.uctx, "gpt-oss.block_count", n_layer);
    gguf_set_val_u32(b.uctx, "gpt-oss.feed_forward_length", n_ff_exp);
    gguf_set_val_u32(b.uctx, "gpt-oss.expert_feed_forward_length", n_ff_exp);
    gguf_set_val_u32(b.uctx, "gpt-oss.attention.head_count", n_head);
    gguf_set_val_u32(b.uctx, "gpt-oss.attention.head_count_kv", n_head);
    gguf_set_val_u32(b.uctx, "gpt-oss.attention.sliding_window", 32);
    gguf_set_val_f32(b.uctx, "gpt-oss.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(b.uctx, "gpt-oss.expert_count", n_exp);
    gguf_set_val_u32(b.uctx, "gpt-oss.expert_used_count", n_used);

    b.add_f32_2d("token_embd.weight", n_embd, n_vocab, 901);
    for (uint32_t il = 0; il < n_layer; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        b.add_f32_1d(p + "attn_norm.weight",           n_embd, 910 + il, 0.02f, true);
        b.add_f32_1d(p + "post_attention_norm.weight", n_embd, 915 + il, 0.02f, true);
        b.add_f32_2d(p + "attn_q.weight",              n_embd, n_embd, 920 + il);
        b.add_f32_2d(p + "attn_k.weight",              n_embd, n_embd, 930 + il);
        b.add_f32_2d(p + "attn_v.weight",              n_embd, n_embd, 940 + il);
        b.add_f32_2d(p + "attn_output.weight",         n_embd, n_embd, 950 + il);
        b.add_f32_1d(p + "attn_output.bias",           n_embd, 955 + il, 0.01f);
        b.add_f32_1d(p + "attn_sinks.weight",          n_head, 960 + il, 0.05f);
        b.add_f32_2d(p + "ffn_gate_inp.weight",        n_embd, n_exp, 970 + il, 0.12f);
        b.add_f32_1d(p + "ffn_gate_inp.bias",          n_exp,  975 + il, 0.02f, false, true);
        b.add_f16_3d(p + "ffn_gate_exps.weight",       n_embd, n_ff_exp, n_exp, 980 + il);
        b.add_f16_3d(p + "ffn_up_exps.weight",         n_embd, n_ff_exp, n_exp, 985 + il);
        b.add_f16_3d(p + "ffn_down_exps.weight",       n_ff_exp, n_embd, n_exp, 990 + il);
        b.add_f32_2d(p + "ffn_gate_exps.bias",         n_ff_exp, n_exp, 992 + il, 0.01f);
        b.add_f32_2d(p + "ffn_up_exps.bias",           n_ff_exp, n_exp, 994 + il, 0.01f);
        b.add_f32_2d(p + "ffn_down_exps.bias",         n_embd,   n_exp, 996 + il, 0.01f);
    }
    b.add_f32_1d("output_norm.weight", n_embd, 998, 0.02f, true);
    b.add_f32_2d("output.weight",      n_embd, n_vocab, 999);
    b.write(path);
}

static void build_grouped_bailingmoe2_gguf(const std::string & path) {
    const uint32_t n_vocab = 64, n_embd = 64, n_head = 2, n_ff = 96, n_ff_exp = 64, n_layer = 5, n_exp = 8, n_used = 2;
    const uint32_t head_k  = n_embd / n_head;
    gguf_builder b;
    gguf_set_val_str(b.uctx, "general.architecture", "bailingmoe2");
    gguf_set_val_u32(b.uctx, "general.alignment", 4096);
    b.add_vocab(n_vocab);
    gguf_set_val_u32(b.uctx, "bailingmoe2.context_length", 128);
    gguf_set_val_u32(b.uctx, "bailingmoe2.embedding_length", n_embd);
    gguf_set_val_u32(b.uctx, "bailingmoe2.block_count", n_layer);
    gguf_set_val_u32(b.uctx, "bailingmoe2.leading_dense_block_count", 1);
    gguf_set_val_u32(b.uctx, "bailingmoe2.feed_forward_length", n_ff);
    gguf_set_val_u32(b.uctx, "bailingmoe2.expert_feed_forward_length", n_ff_exp);
    gguf_set_val_u32(b.uctx, "bailingmoe2.expert_shared_feed_forward_length", n_ff_exp);
    gguf_set_val_u32(b.uctx, "bailingmoe2.expert_shared_count", 1);
    gguf_set_val_u32(b.uctx, "bailingmoe2.attention.head_count", n_head);
    gguf_set_val_u32(b.uctx, "bailingmoe2.attention.head_count_kv", n_head);
    gguf_set_val_f32(b.uctx, "bailingmoe2.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(b.uctx, "bailingmoe2.expert_count", n_exp);
    gguf_set_val_u32(b.uctx, "bailingmoe2.expert_used_count", n_used);
    gguf_set_val_u32(b.uctx, "bailingmoe2.expert_group_count", 2);
    gguf_set_val_u32(b.uctx, "bailingmoe2.expert_group_used_count", 1);
    gguf_set_val_u32(b.uctx, "bailingmoe2.expert_gating_func", 2);   // sigmoid
    gguf_set_val_bool(b.uctx, "bailingmoe2.expert_weights_norm", true);
    gguf_set_val_f32(b.uctx, "bailingmoe2.expert_weights_scale", 1.0f);

    b.add_f32_2d("token_embd.weight", n_embd, n_vocab, 1101);
    for (uint32_t il = 0; il < n_layer; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        b.add_f32_1d(p + "attn_norm.weight",   n_embd, 1110 + il, 0.02f, true);
        b.add_f32_2d(p + "attn_qkv.weight",    n_embd, n_embd * 3, 1120 + il);
        b.add_f32_2d(p + "attn_output.weight", n_embd, n_embd, 1130 + il);
        b.add_f32_1d(p + "attn_q_norm.weight", head_k, 1140 + il, 0.02f, true);
        b.add_f32_1d(p + "attn_k_norm.weight", head_k, 1150 + il, 0.02f, true);
        b.add_f32_1d(p + "ffn_norm.weight",    n_embd, 1160 + il, 0.02f, true);
        if (il == 0) {
            b.add_f32_2d(p + "ffn_gate.weight", n_embd, n_ff, 1170);
            b.add_f32_2d(p + "ffn_up.weight",   n_embd, n_ff, 1171);
            b.add_f32_2d(p + "ffn_down.weight", n_ff, n_embd, 1172);
        } else {
            b.add_f32_2d(p + "ffn_gate_inp.weight", n_embd, n_exp, 1180 + il, 0.12f);
            b.add_f32_1d(p + "exp_probs_b.bias",    n_exp,  1185 + il, 0.02f, false, true);
            b.add_f16_3d(p + "ffn_gate_exps.weight", n_embd, n_ff_exp, n_exp, 1190 + il);
            b.add_f16_3d(p + "ffn_up_exps.weight",   n_embd, n_ff_exp, n_exp, 1200 + il);
            b.add_f16_3d(p + "ffn_down_exps.weight", n_ff_exp, n_embd, n_exp, 1210 + il);
            b.add_f32_2d(p + "ffn_gate_shexp.weight", n_embd, n_ff_exp, 1220 + il);
            b.add_f32_2d(p + "ffn_up_shexp.weight",   n_embd, n_ff_exp, 1230 + il);
            b.add_f32_2d(p + "ffn_down_shexp.weight", n_ff_exp, n_embd, 1240 + il);
        }
    }
    b.add_f32_1d("output_norm.weight", n_embd, 1290, 0.02f, true);
    b.add_f32_2d("output.weight",      n_embd, n_vocab, 1291);
    b.write(path);
}

static void build_hc_qwen4exp_gguf(const std::string & path) {
    const uint32_t n_vocab = 64, n_embd = 64, n_head = 2, n_ff_exp = 64, n_layer = 4, n_exp = 8, n_used = 2;
    const uint32_t hc = 4, hc_lr = 8, hc_dim = hc * n_embd, head_k = n_embd / n_head, idx_dim = 32;
    const int32_t  rope_sec[4] = { 8, 8, 8, 8 };
    gguf_builder b;
    gguf_set_val_str(b.uctx, "general.architecture", "qwen4exp");
    gguf_set_val_u32(b.uctx, "general.alignment", 4096);
    b.add_vocab(n_vocab);
    gguf_set_val_u32(b.uctx, "qwen4exp.context_length", 128);
    gguf_set_val_u32(b.uctx, "qwen4exp.embedding_length", n_embd);
    gguf_set_val_u32(b.uctx, "qwen4exp.block_count", n_layer);
    gguf_set_val_u32(b.uctx, "qwen4exp.feed_forward_length", n_ff_exp);
    gguf_set_val_u32(b.uctx, "qwen4exp.expert_feed_forward_length", n_ff_exp);
    gguf_set_val_u32(b.uctx, "qwen4exp.expert_shared_feed_forward_length", n_ff_exp);
    gguf_set_val_u32(b.uctx, "qwen4exp.attention.head_count", n_head);
    gguf_set_val_u32(b.uctx, "qwen4exp.attention.head_count_kv", n_head);
    gguf_set_val_f32(b.uctx, "qwen4exp.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(b.uctx, "qwen4exp.full_attention_interval", 1);
    gguf_set_arr_data(b.uctx, "qwen4exp.rope.dimension_sections", GGUF_TYPE_INT32, rope_sec, 4);
    gguf_set_val_u32(b.uctx, "qwen4exp.ssm.conv_kernel", 4);
    gguf_set_val_u32(b.uctx, "qwen4exp.ssm.inner_size", n_embd);
    gguf_set_val_u32(b.uctx, "qwen4exp.ssm.state_size", 32);
    gguf_set_val_u32(b.uctx, "qwen4exp.ssm.time_step_rank", 2);
    gguf_set_val_u32(b.uctx, "qwen4exp.ssm.group_count", 2);
    gguf_set_val_u32(b.uctx, "qwen4exp.hyper_connection.count", hc);
    gguf_set_val_u32(b.uctx, "qwen4exp.hyper_connection.low_rank", hc_lr);
    gguf_set_val_u32(b.uctx, "qwen4exp.attention.indexer.head_count", n_head);
    gguf_set_val_u32(b.uctx, "qwen4exp.attention.indexer.key_length", idx_dim);
    gguf_set_val_u32(b.uctx, "qwen4exp.attention.indexer.top_k", 16);
    gguf_set_val_u32(b.uctx, "qwen4exp.expert_count", n_exp);
    gguf_set_val_u32(b.uctx, "qwen4exp.expert_used_count", n_used);

    b.add_f32_2d("token_embd.weight",     n_embd, n_vocab, 1301);
    b.add_f32_2d("output_hc_norm.weight", n_embd, hc, 1302, 0.02f, true);
    b.add_f32_2d("output_hc_down.weight", hc_dim, hc_lr, 1303);
    b.add_f32_2d("output_hc_up.weight",   hc_lr, hc_dim, 1304);
    b.add_f32_2d("output.weight",         n_embd, n_vocab, 1305);

    for (uint32_t il = 0; il < n_layer; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        b.add_f32_2d(p + "hc_attn_norm.weight",   n_embd, hc, 1310 + il, 0.02f, true);
        b.add_f32_2d(p + "hc_attn_up.weight",     hc_lr, hc_dim, 1312 + il);
        b.add_f32_2d(p + "hc_attn_down.weight",   hc_dim, hc_lr, 1314 + il);
        b.add_f32_2d(p + "hc_attn_inject.weight", hc_dim, hc, 1316 + il);
        b.add_f32_2d(p + "hc_ffn_norm.weight",    n_embd, hc, 1318 + il, 0.02f, true);
        b.add_f32_2d(p + "hc_ffn_up.weight",      hc_lr, hc_dim, 1320 + il);
        b.add_f32_2d(p + "hc_ffn_down.weight",    hc_dim, hc_lr, 1322 + il);
        b.add_f32_2d(p + "hc_ffn_inject.weight",  hc_dim, hc, 1324 + il);
        b.add_f32_2d(p + "attn_q.weight",         n_embd, n_embd * 2, 1326 + il);
        b.add_f32_2d(p + "attn_k.weight",         n_embd, n_embd, 1328 + il);
        b.add_f32_2d(p + "attn_v.weight",         n_embd, n_embd, 1330 + il);
        b.add_f32_2d(p + "attn_output.weight",    n_embd, n_embd, 1332 + il);
        b.add_f32_1d(p + "attn_q_norm.weight",    head_k, 1334 + il, 0.02f, true);
        b.add_f32_1d(p + "attn_k_norm.weight",    head_k, 1336 + il, 0.02f, true);
        b.add_f32_2d(p + "indexer.q_proj.weight", n_embd, n_head * idx_dim, 1338 + il);
        b.add_f32_2d(p + "indexer.k_proj.weight", n_embd, idx_dim, 1340 + il);
        b.add_f32_1d(p + "indexer.q_norm.weight", idx_dim, 1342 + il, 0.02f, true);
        b.add_f32_1d(p + "indexer.k_norm.weight", idx_dim, 1344 + il, 0.02f, true);
        b.add_f32_2d(p + "ffn_gate_inp.weight",   n_embd, n_exp, 1346 + il, 0.12f);
        b.add_f16_3d(p + "ffn_gate_exps.weight",  n_embd, n_ff_exp, n_exp, 1348 + il);
        b.add_f16_3d(p + "ffn_up_exps.weight",    n_embd, n_ff_exp, n_exp, 1350 + il);
        b.add_f16_3d(p + "ffn_down_exps.weight",  n_ff_exp, n_embd, n_exp, 1352 + il);
        b.add_f32_1d(p + "ffn_gate_inp_shexp.weight", n_embd, 1354 + il);
        b.add_f32_2d(p + "ffn_gate_shexp.weight", n_embd, n_ff_exp, 1356 + il);
        b.add_f32_2d(p + "ffn_up_shexp.weight",   n_embd, n_ff_exp, 1358 + il);
        b.add_f32_2d(p + "ffn_down_shexp.weight", n_ff_exp, n_embd, 1360 + il);
    }
    b.write(path);
}

static llama_model_params make_stream_params(int32_t slots = 4, bool guess_gate = false, bool vram_try = false) {
    llama_model_params p = llama_model_default_params();
    p.moe_stream               = "model";
    p.moe_stream_slots         = slots;
    p.moe_stream_gpu_layers    = 0;
    p.moe_stream_io_threads    = 2;
    p.moe_stream_policy        = "decay";
    p.moe_stream_evict_halflife = 32.0f;
    p.moe_stream_plan          = 1;
    p.moe_stream_drop_stale    = 1;
    p.moe_stream_needed_early  = 1;
    p.moe_stream_lookahead     = 3;
    p.moe_stream_guess_pace    = 1;
    p.moe_stream_guess_combined = 1;
    p.moe_stream_protect_named = 1;
    p.moe_stream_wordstart     = 1;
    p.moe_stream_promote       = 1;
    p.moe_stream_guess_inflight = 2;
    p.moe_stream_pred_shared   = 1;
    p.moe_stream_verify        = 1;
    if (guess_gate) {
        p.moe_stream_guess_gate     = 1;
        p.moe_stream_ssd_latency_us = 400;
        p.moe_stream_ssd_gbps       = 4.5f;
    }
    if (vram_try) {
        p.moe_stream_vram_experts  = 2;
        p.moe_stream_vram_follow   = 1;
        p.moe_stream_vram_parallel = 1;
    }
    return p;
}

int main() {
    setvbuf(stderr, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);
    llama_backend_init();
    const fs::path tmp = fs::temp_directory_path() / ("tw_moe_universal_" + std::to_string(std::random_device{}()));
    fs::create_directories(tmp);

    auto run_suite = [&]() {
        llama_model_params base_p = llama_model_default_params();

        // 1. Standard 3-proj MoE (qwen3moe) + census + guess-gate + .guesser.json persistence
        {
            const std::string f = (tmp / "t1_qwen3moe.gguf").string();
            build_qwen3moe_gguf(f, 4096);
            const auto ref  = eval_model(f, base_p);
            const auto st0  = eval_model(f, make_stream_params(4, false, true));
            const auto st1  = eval_model(f, make_stream_params(4, true,  true));
            const auto st1b = eval_model(f, make_stream_params(4, true,  true));   // reloads .guesser.json
            check_parity("T1-qwen3moe-base", ref, st0);
            check_parity("T1-qwen3moe-gg1",  ref, st1);
            check_parity("T1-qwen3moe-gg1-reload", ref, st1b);
            const int64_t c0 = st0.stats.dec.n_gg_mem_right + st0.stats.dec.n_gg_mem_wrong +
                               st0.stats.dec.n_gg_ssd_right + st0.stats.dec.n_gg_ssd_wrong;
            if (!st0.has_stats || c0 <= 0) {
                throw std::runtime_error("T1: 2x2 guesser census counters were zero");
            }
            if (!fs::exists(f + ".guesser.json")) {
                throw std::runtime_error("T1: .guesser.json was not persisted");
            }
            fprintf(stderr, "[PASS] T1 standard 3-proj qwen3moe (census=%lld, .guesser.json persisted)\n", (long long) c0);
        }

        // 2. Interleaved dense + MoE (llama4: blocks 0,2,4 dense; 1,3,5 MoE) + safe vx_compat fallback
        {
            const std::string f = (tmp / "t2_llama4_interleaved.gguf").string();
            build_interleaved_llama4_gguf(f);
            const auto ref = eval_model(f, base_p);
            const auto st1 = eval_model(f, make_stream_params(4, true, true));   // vram_try=true tests vx_compat clean fallback!
            check_parity("T2-llama4-interleaved", ref, st1);
            fprintf(stderr, "[PASS] T2 interleaved dense+MoE llama4 (multi-span lookahead + vx_compat fallback)\n");
        }

        // 3. 2-projection fused ffn_gate_up_exps.weight + ffn_down_exps.weight (qwen35moe) + openai-moe biased
        {
            const std::string f_fused = (tmp / "t3a_qwen35moe_fused2proj.gguf").string();
            build_fused_2proj_qwen35moe_gguf(f_fused);
            const auto ref_a = eval_model(f_fused, base_p);
            const auto st_a  = eval_model(f_fused, make_stream_params(4, true, true));
            check_parity("T3a-qwen35moe-fused-gate-up", ref_a, st_a);

            const std::string f_oai = (tmp / "t3b_openai_moe_biased.gguf").string();
            build_openai_moe_gguf(f_oai);
            const auto ref_b = eval_model(f_oai, base_p);
            const auto st_b  = eval_model(f_oai, make_stream_params(4, true, true));
            check_parity("T3b-openai-moe-biased", ref_b, st_b);
            fprintf(stderr, "[PASS] T3 2-proj fused ffn_gate_up_exps (qwen35moe) & expert-biased (openai-moe)\n");
        }

        // 4. Grouped routing + selection bias + shared expert + leading dense (bailingmoe2)
        {
            const std::string f = (tmp / "t4_bailingmoe2_grouped.gguf").string();
            build_grouped_bailingmoe2_gguf(f);
            const auto ref = eval_model(f, base_p);
            auto sp = make_stream_params(4, true, true);
            sp.moe_stream_fused_route = 1;
            const auto st1 = eval_model(f, sp);
            check_parity("T4-bailingmoe2-grouped-bias", ref, st1);
            fprintf(stderr, "[PASS] T4 grouped routing + exp_probs_b + shared expert + wordstart (bailingmoe2)\n");
        }

        // 5. 4-stream hyper-connection MoE (qwen4exp)
        {
            const std::string f = (tmp / "t5_qwen4exp_hc4.gguf").string();
            build_hc_qwen4exp_gguf(f);
            const auto ref = eval_model(f, base_p);
            const auto st1 = eval_model(f, make_stream_params(4, true, false));
            check_parity("T5-qwen4exp-hc4", ref, st1);
            fprintf(stderr, "[PASS] T5 4-stream hyper-connection MoE (qwen4exp)\n");
        }

        // 6. Multi-shard split GGUF (--moe-stream model across 2 shards)
        {
            const std::string s1 = (tmp / "t6_split-00001-of-00002.gguf").string();
            const std::string s2 = (tmp / "t6_split-00002-of-00002.gguf").string();
            build_qwen3moe_gguf(s1, 4096, 1, s2);
            const auto ref = eval_model(s1, base_p);
            const auto st1 = eval_model(s1, make_stream_params(4, true, false));
            check_parity("T6-multishard-split", ref, st1);
            fprintf(stderr, "[PASS] T6 multi-shard split GGUF Direct-I/O + verify\n");
        }

        // 7. 32B-aligned (non-4096B-aligned) stock GGUF bounce-buffer Direct-I/O + default LRU policy with guess-gate 1
        {
            const std::string f = (tmp / "t7_unaligned_32b.gguf").string();
            build_qwen3moe_gguf(f, 32);
            const auto ref = eval_model(f, base_p);
            auto sp = make_stream_params(4, true, false);
            sp.moe_stream_policy = "lru";
            const auto st1 = eval_model(f, sp);
            check_parity("T7-32b-unaligned-stock", ref, st1);
            fprintf(stderr, "[PASS] T7 32B-unaligned stock GGUF bounce-buffer Direct-I/O (policy=lru)\n");
        }

        // 8. Slow-I/O congestion + CPU-router host-fallback stress test (zero hangs)
        {
            const std::string f = (tmp / "t1_qwen3moe.gguf").string();
            const auto ref = eval_model(f, base_p, 12);
            auto sp = make_stream_params(3, true, false);
            sp.n_gpu_layers             = 0;       // force host router / host gg_shares_t fallback
            sp.moe_stream_guess_inflight = 1;
            sp.moe_stream_ssd_latency_us = 5000;    // heavy simulated SSD congestion
            sp.moe_stream_ssd_gbps       = 0.4f;
            const auto st1 = eval_model(f, sp, 12);
            check_parity("T8-slow-io-host-fallback", ref, st1);
            fprintf(stderr, "[PASS] T8 slow-I/O congestion + host-shares fallback (skip_floor=%lld, skip_vic=%lld, defer_q=%lld)\n",
                    (long long) st1.stats.dec.n_gg_skip_floor,
                    (long long) st1.stats.dec.n_gg_skip_vic,
                    (long long) st1.stats.dec.n_gg_defer_q);
        }
    };

    int rc = 0;
    try {
        run_suite();
        fprintf(stderr, "ALL 8 UNIVERSAL MOE TOPOLOGIES PASSED!\n");
    } catch (const std::exception & ex) {
        fprintf(stderr, "FAIL: %s\n", ex.what());
        rc = 1;
    }
    std::error_code ec;
    fs::remove_all(tmp, ec);
    llama_backend_free();
    return rc;
}

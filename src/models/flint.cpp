#include "models.h"

void llama_model_flint::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);

    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
    ml.get_arr(LLM_KV_ATTENTION_ROPE_PATTERN, hparams.rope_pattern);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_flint::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp        = hparams.n_ff_exp();

    // The word table is read lazily under --moe-stream (llama_arch_marks_token_embd_lazy in
    // llama-model.cpp lists Flint); only with a separate output.weight - a tied table would be read in full by every word
    const bool untied = ml.get_weight(tn(LLM_TENSOR_OUTPUT, "weight").str().c_str()) != nullptr;
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, untied ? TENSOR_READ_LAZY : 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), {n_head}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            // MoE branch
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        hparams.n_ff_shexp, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_flint::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_flint::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    // Layer-major (--moe-stream-layer-major 1, as in bailingmoe3.cpp): lm_phase 2 = the expert graph of streamed layer lm_il0 for
    // the whole call - no input, no memory: build_moe_ffn reads the call's normed expert input and router picks / weights and adds the
    // routed output into the call's residual
    if (params.lm_phase == 2) {
        const int il = params.lm_il0;
        build_moe_ffn(params.lm_x,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il, nullptr, nullptr,
                model.layers[il].ffn_up_exps_s,
                model.layers[il].ffn_gate_exps_s,
                model.layers[il].ffn_down_exps_s);
        return;
    }

    ggml_tensor * cur;

    // A layer group after the first reads its input from the call's hidden state on the device (params.lm_hid)
    ggml_tensor * inpL = params.lm_hid && params.lm_il0 > 0
        ? ggml_view_2d(ctx0, params.lm_hid, n_embd, n_tokens, params.lm_hid->nb[1], params.lm_row0*params.lm_hid->nb[1])
        : build_inp_embd(model.tok_embd);
    if (inpL->type != GGML_TYPE_F32) {
        inpL = ggml_cast(ctx0, inpL, GGML_TYPE_F32);
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    // Only the layers [lm_b, lm_e) (layer-major); the output graph is the one without an end layer (lm_il1 < 0: the call's last
    // loop, or not layer-major); lm_b == n_layer = the output alone
    const int  lm_b    = params.lm_il0;
    const int  lm_e    = params.lm_il1 < 0 ? n_layer : std::min((int) n_layer, (int) params.lm_il1);
    const bool lm_last = params.lm_il1 < 0;
    GGML_ASSERT(lm_b >= 0 && (lm_b < lm_e || (lm_b == n_layer && lm_last)));

    ggml_tensor * inp_out_ids = lm_last ? build_inp_out_ids() : nullptr;

    // The expert streaming's word-start guesses (shared code, llama-graph.cpp; nothing without --moe-stream-wordstart)
    llm_graph_moe_ws ws = moe_ws_begin();

    for (int il = lm_b; il < lm_e; ++il) {
        res->t_layer_inp[il] = inpL;
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            if (hparams.has_rope(il)) {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            }
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, model.layers[il].attn_sinks, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);
        ggml_tensor * ffn_in_n = cur;       // a word-start source's guess input
        ggml_tensor * ws_base  = nullptr;   // and an MoE source's base (the residual + the shared expert)

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // The guesses for the next layers (shared code, llama-graph.cpp; all null without --moe-stream). With
            // --moe-stream-pred-shared the shared expert is built first and the guess made from the residual plus it
            llm_graph_moe_guess g = moe_guess_routers(model, il);
            ggml_tensor * ffn_shexp = nullptr;
            if (g.want_shared) {
                ffn_shexp = build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                        model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                        model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            }
            moe_guess_finish(model, il, ffn_inp, ffn_shexp, g);
            ws_base = g.pred_base;

            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il, nullptr, nullptr,
                model.layers[il].ffn_up_exps_s,
                model.layers[il].ffn_gate_exps_s,
                model.layers[il].ffn_down_exps_s,
                /* selected_experts_in = */ nullptr,
                g.gate_inp_next, g.probs_b_next, g.pred_in,
                g.gate_inp_next2, g.probs_b_next2, g.gate_inp_next3, g.probs_b_next3,
                g.rec_on ? &g.rec : nullptr);
            if (moe_out) {   // null in a layer-major lm_phase 1 graph (the routed experts come in the layer's expert graph)
                cb(moe_out, "ffn_moe_out", il);
            }

            {
                if (!ffn_shexp) {   // built above when the guess needed it first
                    ffn_shexp = build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                        model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                        model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                }
                cb(ffn_shexp, "ffn_shexp", il);

                if (moe_out == nullptr) {   // layer-major lm_phase 1: the routed experts come later, in the layer's expert graph
                    GGML_ASSERT(params.lm_phase == 1 && params.lm_x);
                    ggml_tensor * x_dst = ggml_view_2d(ctx0, params.lm_x, n_embd, n_tokens, params.lm_x->nb[1],
                            params.lm_row0*params.lm_x->nb[1]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, cur, x_dst));   // cur: this layer's normed expert input
                    cur = ffn_shexp;
                } else {
                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                }
                cb(cur, "ffn_out", il);
            }
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        moe_ws_layer(model, il, cur, ws_base, ffn_in_n, ws);   // a word-start source's guesses (nothing otherwise)

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    if (lm_b > 0 || !lm_last) {
        // As in bailingmoe3.cpp: the positions are expanded as a leaf in every group graph
        ggml_build_forward_expand(gf, inp_pos);
    }

    if (!lm_last) {   // a layer group that ends early outputs every token's hidden state instead of the logits
        if (params.lm_hid) {   // written into the call's hidden state on the device (its own rows)
            ggml_tensor * dst = ggml_view_2d(ctx0, params.lm_hid, n_embd, n_tokens, params.lm_hid->nb[1],
                    params.lm_row0*params.lm_hid->nb[1]);
            res->t_lm_hidden = ggml_cpy(ctx0, inpL, dst);
            ggml_build_forward_expand(gf, res->t_lm_hidden);
            return;
        }
        ggml_set_output(inpL);
        res->t_lm_hidden = inpL;
        ggml_build_forward_expand(gf, inpL);
        return;
    }

    if (lm_b == n_layer && inp_out_ids) {
        // The output alone after a layer-major call - the last layer ran every row in the layer loop, so the output rows are
        // kept here (a whole graph drops them inside the last layer)
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // upstream's Flint (#30928) was written after #30445, whose build_logits() expands the LM head; that change is not
    // brought, so the model expands it itself like every other model here
    ggml_build_forward_expand(gf, cur);
}

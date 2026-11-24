#include "llama-model.h"

#include "llama-arch.h"
#include "llama-ext.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-cparams.h"
#include "llama-model-loader.h"
#include "llama-moe-stream.h"
#include "llama-arch.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include "llama.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static llama_model * llama_model_mapping(llm_arch arch, const llama_model_params & params) {
    switch (arch) {
        case LLM_ARCH_CLIP:
            return new llama_model_clip(params);
        case LLM_ARCH_LLAMA:
            return new llama_model_llama(params);
        case LLM_ARCH_LLAMA4:
            return new llama_model_llama4(params);
        case LLM_ARCH_LLAMA_EMBED:
            return new llama_model_llama_embed(params);
        case LLM_ARCH_MAINCODER:
            return new llama_model_maincoder(params);
        case LLM_ARCH_TALKIE:
            return new llama_model_talkie(params);
        case LLM_ARCH_DECI:
            return new llama_model_deci(params);
        case LLM_ARCH_BAICHUAN:
            return new llama_model_baichuan(params);
        case LLM_ARCH_FALCON:
            return new llama_model_falcon(params);
        case LLM_ARCH_GROK:
            return new llama_model_grok(params);
        case LLM_ARCH_STARCODER:
            return new llama_model_starcoder(params);
        case LLM_ARCH_REFACT:
            return new llama_model_refact(params);
        case LLM_ARCH_BERT:
            return new llama_model_bert(params);
        case LLM_ARCH_JINA_BERT_V2:
            return new llama_model_jina_bert_v2(params);
        case LLM_ARCH_JINA_BERT_V3:
            return new llama_model_jina_bert_v3(params);
        case LLM_ARCH_NOMIC_BERT:
            return new llama_model_nomic_bert(params);
        case LLM_ARCH_NOMIC_BERT_MOE:
            return new llama_model_nomic_bert_moe(params);
        case LLM_ARCH_MODERN_BERT:
            return new llama_model_modern_bert(params);
        case LLM_ARCH_NEO_BERT:
            return new llama_model_neo_bert(params);
        case LLM_ARCH_EUROBERT:
            return new llama_model_eurobert(params);
        case LLM_ARCH_BLOOM:
            return new llama_model_bloom(params);
        case LLM_ARCH_MPT:
            return new llama_model_mpt(params);
        case LLM_ARCH_STABLELM:
            return new llama_model_stablelm(params);
        case LLM_ARCH_MELLUM:
            return new llama_model_mellum(params);
        case LLM_ARCH_NANBEIGE:
            return new llama_model_nanbeige(params);
        case LLM_ARCH_QWEN:
            return new llama_model_qwen(params);
        case LLM_ARCH_QWEN2:
            return new llama_model_qwen2(params);
        case LLM_ARCH_DREAM:
            return new llama_model_dream(params);
        case LLM_ARCH_LLADA:
            return new llama_model_llada(params);
        case LLM_ARCH_LLADA_MOE:
            return new llama_model_llada_moe(params);
        case LLM_ARCH_RND1:
            return new llama_model_rnd1(params);
        case LLM_ARCH_QWEN2VL:
            return new llama_model_qwen2vl(params);
        case LLM_ARCH_QWEN2MOE:
            return new llama_model_qwen2moe(params);
        case LLM_ARCH_QWEN3:
            return new llama_model_qwen3(params);
        case LLM_ARCH_FLINT:
            return new llama_model_flint(params);
        case LLM_ARCH_QWEN3MOE:
            return new llama_model_qwen3moe(params);
        case LLM_ARCH_QWEN3VL:
            return new llama_model_qwen3vl(params);
        case LLM_ARCH_QWEN3VLMOE:
            return new llama_model_qwen3vlmoe(params);
        case LLM_ARCH_QWEN3TTS:
            return new llama_model_qwen3tts(params);
        case LLM_ARCH_POCKETTTS:
            return new llama_model_pockettts(params);
        case LLM_ARCH_PHI2:
            return new llama_model_phi2(params);
        case LLM_ARCH_PHI3:
            return new llama_model_phi3(params);
        case LLM_ARCH_PHIMOE:
            return new llama_model_phimoe(params);
        case LLM_ARCH_PLAMO:
            return new llama_model_plamo(params);
        case LLM_ARCH_PLAMO2:
            return new llama_model_plamo2(params);
        case LLM_ARCH_PLAMO3:
            return new llama_model_plamo3(params);
        case LLM_ARCH_GPT2:
            return new llama_model_gpt2(params);
        case LLM_ARCH_CODESHELL:
            return new llama_model_codeshell(params);
        case LLM_ARCH_ORION:
            return new llama_model_orion(params);
        case LLM_ARCH_INTERNLM2:
            return new llama_model_internlm2(params);
        case LLM_ARCH_MINICPM3:
            return new llama_model_minicpm3(params);
        case LLM_ARCH_GEMMA:
            return new llama_model_gemma(params);
        case LLM_ARCH_GEMMA2:
            return new llama_model_gemma2(params);
        case LLM_ARCH_GEMMA3:
            return new llama_model_gemma3(params);
        case LLM_ARCH_GEMMA3N:
            return new llama_model_gemma3n(params);
        case LLM_ARCH_GEMMA4:
            return new llama_model_gemma4(params);
        case LLM_ARCH_GEMMA4_ASSISTANT:
            return new llama_model_gemma4_assistant(params);
        case LLM_ARCH_GEMMA_EMBEDDING:
            return new llama_model_gemma_embedding(params);
        case LLM_ARCH_STARCODER2:
            return new llama_model_starcoder2(params);
        case LLM_ARCH_MAMBA:
            return new llama_model_mamba(params);
        case LLM_ARCH_MAMBA2:
            return new llama_model_mamba2(params);
        case LLM_ARCH_MAPLE:
            return new llama_model_maple(params);
        case LLM_ARCH_JAMBA:
            return new llama_model_jamba(params);
        case LLM_ARCH_XVERSE:
            return new llama_model_xverse(params);
        case LLM_ARCH_COMMAND_R:
            return new llama_model_command_r(params);
        case LLM_ARCH_COHERE2:
            return new llama_model_cohere2(params);
        case LLM_ARCH_COHERE2MOE:
            return new llama_model_cohere2moe(params);
        case LLM_ARCH_DBRX:
            return new llama_model_dbrx(params);
        case LLM_ARCH_OLMO:
            return new llama_model_olmo(params);
        case LLM_ARCH_OLMO2:
            return new llama_model_olmo2(params);
        case LLM_ARCH_OLMOE:
            return new llama_model_olmoe(params);
        case LLM_ARCH_MUSE_GLIMMER:
            return new llama_model_muse_glimmer(params);
        case LLM_ARCH_OPENELM:
            return new llama_model_openelm(params);
        case LLM_ARCH_GPTNEOX:
            return new llama_model_gptneox(params);
        case LLM_ARCH_ARCTIC:
            return new llama_model_arctic(params);
        case LLM_ARCH_DEEPSEEK:
            return new llama_model_deepseek(params);
        case LLM_ARCH_DEEPSEEK2:
            return new llama_model_deepseek2(params);
        case LLM_ARCH_DEEPSEEK2OCR:
            return new llama_model_deepseek2ocr(params);
        case LLM_ARCH_DEEPSEEK32:
            return new llama_model_deepseek32(params);
        case LLM_ARCH_DOTS3NOTE:
            return new llama_model_dots3note(params);
        case LLM_ARCH_DEEPSEEK4:
            return new llama_model_deepseek4(params);
        case LLM_ARCH_GLM_DSA:
            return new llama_model_glm_dsa(params);
        case LLM_ARCH_MISTRAL4:
            return new llama_model_mistral4(params);
        case LLM_ARCH_CHATGLM:
            return new llama_model_chatglm(params);
        case LLM_ARCH_GLM4:
            return new llama_model_glm4(params);
        case LLM_ARCH_GLM4_MOE:
            return new llama_model_glm4_moe(params);
        case LLM_ARCH_BITNET:
            return new llama_model_bitnet(params);
        case LLM_ARCH_T5:
            return new llama_model_t5(params);
        case LLM_ARCH_T5ENCODER:
            return new llama_model_t5encoder(params);
        case LLM_ARCH_JAIS:
            return new llama_model_jais(params);
        case LLM_ARCH_JAIS2:
            return new llama_model_jais2(params);
        case LLM_ARCH_NEMOTRON:
            return new llama_model_nemotron(params);
        case LLM_ARCH_NEMOTRON_H:
            return new llama_model_nemotron_h(params);
        case LLM_ARCH_NEMOTRON_H_MOE:
            return new llama_model_nemotron_h_moe(params);
        case LLM_ARCH_EXAONE:
            return new llama_model_exaone(params);
        case LLM_ARCH_EXAONE4:
            return new llama_model_exaone4(params);
        case LLM_ARCH_EXAONE_MOE:
            return new llama_model_exaone_moe(params);
        case LLM_ARCH_RWKV6:
            return new llama_model_rwkv6(params);
        case LLM_ARCH_RWKV6QWEN2:
            return new llama_model_rwkv6qwen2(params);
        case LLM_ARCH_RWKV7:
            return new llama_model_rwkv7(params);
        case LLM_ARCH_ARWKV7:
            return new llama_model_arwkv7(params);
        case LLM_ARCH_GRANITE:
            return new llama_model_granite(params);
        case LLM_ARCH_GRANITE_MOE:
            return new llama_model_granite_moe(params);
        case LLM_ARCH_GRANITE_SWITCH:
            return new llama_model_granite_switch(params);
        case LLM_ARCH_MINICPM:
            return new llama_model_minicpm(params);
        case LLM_ARCH_GRANITE_HYBRID:
            return new llama_model_granite_hybrid(params);
        case LLM_ARCH_GRANITE_SWA:
            return new llama_model_granite_swa(params);
        case LLM_ARCH_CHAMELEON:
            return new llama_model_chameleon(params);
        case LLM_ARCH_WAVTOKENIZER_DEC:
            return new llama_model_wavtokenizer_dec(params);
        case LLM_ARCH_PLM:
            return new llama_model_plm(params);
        case LLM_ARCH_BAILINGMOE:
            return new llama_model_bailingmoe(params);
        case LLM_ARCH_BAILINGMOE2:
            return new llama_model_bailingmoe2(params);
        case LLM_ARCH_BAILINGMOE3:
            return new llama_model_bailingmoe3(params);
        case LLM_ARCH_SEED_OSS:
            return new llama_model_seed_oss(params);
        case LLM_ARCH_DOTS1:
            return new llama_model_dots1(params);
        case LLM_ARCH_ARCEE:
            return new llama_model_arcee(params);
        case LLM_ARCH_AFMOE:
            return new llama_model_afmoe(params);
        case LLM_ARCH_LAGUNA:
            return new llama_model_laguna(params);
        case LLM_ARCH_ERNIE4_5:
            return new llama_model_ernie4_5(params);
        case LLM_ARCH_ERNIE4_5_MOE:
            return new llama_model_ernie4_5_moe(params);
        case LLM_ARCH_PADDLEOCR:
            return new llama_model_paddleocr(params);
        case LLM_ARCH_HUNYUAN_MOE:
            return new llama_model_hunyuan_moe(params);
        case LLM_ARCH_HUNYUAN_VL:
            return new llama_model_hunyuan_vl(params);
        case LLM_ARCH_HUNYUAN_DENSE:
            return new llama_model_hunyuan_dense(params);
        case LLM_ARCH_HY_V3:
            return new llama_model_hy_v3(params);
        case LLM_ARCH_HY_V4:
            return new llama_model_hy_v4(params);
        case LLM_ARCH_SMOLLM3:
            return new llama_model_smollm3(params);
        case LLM_ARCH_OPENAI_MOE:
            return new llama_model_openai_moe(params);
        case LLM_ARCH_FALCON_H1:
            return new llama_model_falcon_h1(params);
        case LLM_ARCH_LFM2:
            return new llama_model_lfm2(params);
        case LLM_ARCH_LFM2MOE:
            return new llama_model_lfm2moe(params);
        case LLM_ARCH_SMALLTHINKER:
            return new llama_model_smallthinker(params);
        case LLM_ARCH_GROVEMOE:
            return new llama_model_grovemoe(params);
        case LLM_ARCH_APERTUS:
            return new llama_model_apertus(params);
        case LLM_ARCH_MINIMAX_01:
            return new llama_model_minimax_01(params);
        case LLM_ARCH_MINIMAX_M2:
            return new llama_model_minimax_m2(params);
        case LLM_ARCH_MINIMAX_M3:
            return new llama_model_minimax_m3(params);
        case LLM_ARCH_HRM_TEXT:
            return new llama_model_hrm_text(params);
        case LLM_ARCH_INKLING2:
            return new llama_model_inkling2(params);
        case LLM_ARCH_COGVLM:
            return new llama_model_cogvlm(params);
        case LLM_ARCH_PANGU_EMBED:
            return new llama_model_pangu_embed(params);
        case LLM_ARCH_QWEN3NEXT:
            return new llama_model_qwen3next(params);
        case LLM_ARCH_QWEN35:
            return new llama_model_qwen35(params);
        case LLM_ARCH_QWEN35MOE:
            return new llama_model_qwen35moe(params);
        case LLM_ARCH_QWEN4EXP:
            return new llama_model_qwen4exp(params);
        case LLM_ARCH_MISTRAL3:
            return new llama_model_mistral3(params);
        case LLM_ARCH_EAGLE3:
            return new llama_model_eagle3(params);
        case LLM_ARCH_DFLASH:
            return new llama_model_dflash(params);
        case LLM_ARCH_MIMO2:
            return new llama_model_mimo2(params);
        case LLM_ARCH_KIMI_LINEAR:
            return new llama_model_kimi_linear(params);
        case LLM_ARCH_KIMI_K3:
            return new llama_model_kimi_k3(params);
        case LLM_ARCH_STEP35:
            return new llama_model_step35(params);
        case LLM_ARCH_SPARK2_5:
            return new llama_model_spark2_5(params);
        default:
            throw std::runtime_error(std::string("unsupported model architecture: '") + llm_arch_name(arch) + "'");
    }

}

llama_model * llama_model_create(llm_arch arch, const llama_model_params & params) {
    llama_model * model = llama_model_mapping(arch, params);

    if (model != nullptr) {
        model->arch = arch;
        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR && !llm_arch_supports_sm_tensor(arch)) {
            throw std::runtime_error(std::string("LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '") + llm_arch_name(arch) + "'");
        }
    }

    return model;
}

llama_model * llama_model_create(llama_model_loader & ml, const llama_model_params & params) {
    llm_arch arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }

    return llama_model_create(arch, params);
}

struct ggml_backend_meta_split_state llama_meta_device_get_split_state(const struct ggml_tensor * tensor, void * userdata) {
    const llama_meta_device_get_split_state_userdata * ud = (const llama_meta_device_get_split_state_userdata *) userdata;
    const llama_hparams & hparams = ud->model->hparams;
    const std::string tensor_name = tensor->name;
    const bool is_dsv4 = ud->model->arch == LLM_ARCH_DEEPSEEK4 ||
        (ud->model->arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0);

    static const std::regex pattern_q_weight        ("blk\\.\\d*\\.attn_q.weight");
    static const std::regex pattern_kv_weight       ("blk\\.\\d*\\.attn_(k|v).weight");
    static const std::regex pattern_qkv_weight      ("blk\\.\\d*\\.attn_qkv.weight");
    static const std::regex pattern_q_bias          ("blk\\.\\d*\\.attn_q\\.bias");
    static const std::regex pattern_kv_bias         ("blk\\.\\d*\\.attn_(k|v)\\.bias");
    static const std::regex pattern_qkv_bias        ("blk\\.\\d*\\.attn_qkv.bias");
    static const std::regex pattern_qk_norm         ("blk\\.\\d*\\.attn_(q|k)_norm\\.weight");
    static const std::regex pattern_kv_cache        ("cache_(k|v)_l\\d*");
    static const std::regex pattern_idx_cache       ("cache_idx_(k|v)_l\\d*");
    static const std::regex pattern_dsv4_state      ("dsv4_(csa|hca|lid)_state_(kv|score)_l\\d*");
    static const std::regex pattern_attn_sinks      ("blk\\.\\d*\\.attn_sinks.weight");
    static const std::regex pattern_attn_out_weight ("blk\\.\\d*\\.attn_output.weight");
    static const std::regex pattern_attn_out_bias   ("blk\\.\\d*\\.attn_output.bias");
    static const std::regex pattern_attn_out_a_weight("blk\\.\\d*\\.attn_output_a\\.weight");
    static const std::regex pattern_attn_out_b_weight("blk\\.\\d*\\.attn_output_b\\.weight");
    static const std::regex pattern_attn_q_b_weight ("blk\\.\\d*\\.attn_q_b\\.weight");
    static const std::regex pattern_attn_gate_weight("blk\\.\\d*\\.attn_gate.weight");

    static const std::regex pattern_ssm_dt          ("blk\\.\\d*\\.ssm_dt.bias");
    static const std::regex pattern_ssm_a           ("blk\\.\\d*\\.ssm_a");
    static const std::regex pattern_ssm_alpha       ("blk\\.\\d*\\.ssm_alpha.weight");
    static const std::regex pattern_ssm_beta        ("blk\\.\\d*\\.ssm_beta.weight");
    static const std::regex pattern_ssm_beta_alpha  ("blk\\.\\d*\\.ssm_ba.weight");
    static const std::regex pattern_r_cache         ("cache_r_l\\d*");
    static const std::regex pattern_ple_r_cache     ("cache_ple_r_l\\d*");
    static const std::regex pattern_s_cache         ("cache_s_l\\d*");
    static const std::regex pattern_ssm_conv1d      ("blk\\.\\d*\\.ssm_conv1d.weight");
    static const std::regex pattern_ssm_out_weight  ("blk\\.\\d*\\.ssm_out.weight");

    static const std::regex pattern_ffn_up_weight     ("blk\\.\\d*\\.ffn_up(_exps)?.weight");
    static const std::regex pattern_ffn_up_bias       ("blk\\.\\d*\\.ffn_up(_exps)?.bias");
    static const std::regex pattern_ffn_gate_weight   ("blk\\.\\d*\\.ffn_gate(_exps)?.weight");
    static const std::regex pattern_ffn_gate_bias     ("blk\\.\\d*\\.ffn_gate(_exps)?.bias");
    static const std::regex pattern_ffn_gate_up_weight("blk\\.\\d*\\.ffn_gate_up(_exps)?.weight");
    static const std::regex pattern_ffn_down_weight   ("blk\\.\\d*\\.ffn_down(_exps)?.weight");
    static const std::regex pattern_ffn_down_bias         ("blk\\.\\d*\\.ffn_down.bias");
    static const std::regex pattern_ffn_down_exps_bias    ("blk\\.\\d*\\.ffn_down_exps.bias");
    static const std::regex pattern_ffn_up_shexp_weight   ("blk\\.\\d*\\.ffn_up_shexp.weight");
    static const std::regex pattern_ffn_gate_shexp_weight ("blk\\.\\d*\\.ffn_gate_shexp.weight");
    static const std::regex pattern_ffn_down_shexp_weight ("blk\\.\\d*\\.ffn_down_shexp.weight");

    static const std::regex pattern_output_weight("output\\.weight");
    static const std::regex pattern_output_bias  ("output\\.bias");

    struct tensor_config {
        ggml_backend_meta_split_axis axis;

        const ggml_tensor * tensor_axis_0;

        uint32_t il;
        size_t   rotation; // when assigning tensor slices, rotate how the rounding is done for more even allocation
    };

    auto get_tensor_config_impl = [&](
                const ggml_backend_meta_split_axis axis, const std::string & suffix = "", const std::string & suffix_fallback = "") -> tensor_config {
        // the layers in a tensor can be inhomogeneous, if the pattern is cleanly divided by the number of GPUs there can be aliasing effects,
        //     count only the same type of previous layers to avoid this
        auto get_il_eff = [&](const size_t il){
            size_t ret = 0;
            const bool il_is_recr = hparams.is_recr(il);
            const bool il_is_swa  = hparams.is_swa(il);
            for (size_t il_prev = 0; il_prev < il; il_prev++) {
                ret += hparams.is_recr(il_prev) == il_is_recr && hparams.is_swa(il_prev) == il_is_swa;
            }
            return ret;
        };

        uint32_t il;
        std::string prefix;
        size_t rotation;
        if (tensor_name.substr(0, 4) == "blk.") {
            const size_t length_prefix = tensor_name.find('.', 4);
            GGML_ASSERT(length_prefix != std::string::npos);
            prefix = tensor_name.substr(0, length_prefix + 1);
            il = std::stoull(tensor_name.substr(4, length_prefix));
            rotation = get_il_eff(il) % ud->n_devices;
        } else if (tensor_name.substr(0, 6) == "cache_") {
            const size_t layer_index_start = tensor_name.find("_l", 6);
            GGML_ASSERT(layer_index_start != std::string::npos);
            il = std::stoull(tensor_name.substr(layer_index_start + 2));
            prefix = "blk." + std::to_string(il) + ".";
            rotation = get_il_eff(il) % ud->n_devices;
        } else {
            il = 0;
            rotation = hparams.n_layer() % ud->n_devices;
        }
        const ggml_tensor * tensor_axis_0 = suffix.empty() ? tensor : ud->model->get_tensor((prefix + suffix).c_str());
        if (tensor_axis_0 == nullptr) {
            GGML_ASSERT(!suffix_fallback.empty());
            tensor_axis_0 = ud->model->get_tensor((prefix + suffix_fallback).c_str());
        }
        GGML_ASSERT(tensor_axis_0 != nullptr);
        return {axis, tensor_axis_0, il, rotation};
    };

    auto get_tensor_config = [&]() -> tensor_config {
        if (ud->model->arch == LLM_ARCH_HRM_TEXT || ud->model->arch == LLM_ARCH_INKLING2) {
            // aliased cache slots cannot satisfy the meta-split invariants, so replicate all tensors
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, tensor, 0, 0};
        }
        if (is_dsv4) {
            if (std::regex_match(tensor_name, pattern_kv_cache) ||
                    std::regex_match(tensor_name, pattern_dsv4_state)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            }
            if (std::regex_match(tensor_name, pattern_attn_sinks)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output_a.weight");
            }
            if (std::regex_match(tensor_name, pattern_attn_q_b_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output_a.weight");
            }
            if (std::regex_match(tensor_name, pattern_attn_out_a_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_2);
            }
            if (std::regex_match(tensor_name, pattern_attn_out_b_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
            }
            if (std::regex_match(tensor_name, pattern_ffn_up_shexp_weight) ||
                    std::regex_match(tensor_name, pattern_ffn_gate_shexp_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down_shexp.weight");
            }
            if (std::regex_match(tensor_name, pattern_ffn_down_shexp_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down_shexp.weight");
            }
        }

        // the qsa indexer has one key head and its projections are mirrored, so its cache cannot be split
        if (std::regex_match(tensor_name, pattern_idx_cache)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        // the PLE table is model-level and its conv is mirrored, so every device runs the whole conv and needs the whole history
        if (std::regex_match(tensor_name, pattern_ple_r_cache)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        // standard attention
        if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_kv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_q_bias) || std::regex_match(tensor_name, pattern_kv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qkv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if ( std::regex_match(tensor_name, pattern_qkv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qk_norm)) {
            return get_tensor_config_impl(tensor->ne[1] == 1 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_kv_cache) || std::regex_match(tensor_name, pattern_attn_sinks)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }
        if (std::regex_match(tensor_name, pattern_attn_out_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta) ||
                std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_r_cache) || std::regex_match(tensor_name, pattern_s_cache)) {
            if (ud->model->arch == LLM_ARCH_LFM2 || ud->model->arch == LLM_ARCH_LFM2MOE) {
                // the LFM2 shortconv block runs fully mirrored, so its conv state must be mirrored too
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED, "");
            }
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_conv1d)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_bias) || std::regex_match(tensor_name, pattern_ffn_gate_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_exps_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_PARTIAL, "ffn_down_exps.weight");
        }

        // output
        if (std::regex_match(tensor_name, pattern_output_weight)) {
            if (is_dsv4) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            }
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        }
        if (std::regex_match(tensor_name, pattern_output_bias)) {
            const ggml_tensor * output_weight = ud->model->get_tensor("output.weight");
            GGML_ASSERT(output_weight != nullptr);
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // everything else
        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    };

    auto get_split_segments = [&](int axis, uint32_t il) -> std::vector<std::pair<int64_t, uint32_t>> {
        // TODO: clarify why this is necessary specifically for these models
        // TODO: deduplicate condition [TAG_SPLIT_QGATE_QWEN]
        if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE ||
                ud->model->arch == LLM_ARCH_QWEN4EXP) {

            // fused full attention layers with Q gate tensors that need n_embd doubled:
            if (!hparams.is_recr(il) && (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias))) {
                const int64_t n_embd      = hparams.n_head(il) * hparams.n_embd_head_k(il) * 2;
                const int64_t n_embd_gqa  = hparams.n_embd_v_gqa(il);
                GGML_ASSERT(hparams.n_embd_k_gqa(il) == n_embd_gqa);
                GGML_ASSERT(tensor->ne[axis] == n_embd + 2*n_embd_gqa);
                return {{n_embd, 1}, {n_embd_gqa, 2}};
            }

            const int64_t head_k_dim = hparams.ssm_d_state;
            const int64_t head_v_dim = hparams.ssm_d_state;
            const int64_t n_k_heads  = hparams.ssm_n_group;
            const int64_t n_v_heads  = hparams.ssm_dt_rank;
            const int64_t key_dim    = head_k_dim * n_k_heads;
            const int64_t value_dim  = head_v_dim * n_v_heads;

            // both Qwen 3 Next and Qwen 3.5 support n_v_heads > n_k_heads but the broadcasting pattern is different:
            //   - Qwen 3 Next: [k0_v0, k0_v1, k1_v2, k1_v3] (this is the default split pattern)
            //   - Qwen 3.5:    [k0_v0, k1_v1, k0_v2, k1_v3] (needs segmenting of V on the scale of K to get the correct pattern)
            if (ud->model->arch == LLM_ARCH_QWEN3NEXT) {
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, 2}, {value_dim, 1}};
                }
                if (std::regex_match(tensor_name, pattern_r_cache)) {
                    return {{key_dim * (hparams.ssm_d_conv - 1), 2}, {value_dim * (hparams.ssm_d_conv - 1), 1}};
                }
            } else {
                const int64_t head_ratio = n_v_heads / n_k_heads;
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, 2 + head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_attn_gate_weight) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                    return {{key_dim, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                        std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                    return {{n_k_heads, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_r_cache)) {
                    return {{key_dim * (hparams.ssm_d_conv - 1), 2 + head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_s_cache)) {
                    return {{n_k_heads * head_v_dim * head_v_dim, head_ratio}};
                }
            }

            // the FFN is the same for Qwen 3 Next and Qwen 3.5:
            if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
                const int64_t n_ff_exp = hparams.n_ff_exp(il);
                GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
                return {{n_ff_exp, 2}};
            }
            return {{tensor->ne[axis], 1}};
        }

        if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
            const int64_t n_embd_q = hparams.n_head(il) * hparams.n_embd_head_k(il);
            const int64_t n_embd_k = hparams.n_embd_k_gqa(il);
            const int64_t n_embd_v = hparams.n_embd_v_gqa(il);
            GGML_ASSERT(tensor->ne[axis] == n_embd_q + n_embd_k + n_embd_v);
            if (n_embd_k == n_embd_v) {
                return {{n_embd_q, 1}, {n_embd_k, 2}};
            }
            // uneven K/V head sizes (e.g. MiMo d_k=192 d_v=128): split K and V as separate
            // segments so each device gets whole heads of both
            return {{n_embd_q, 1}, {n_embd_k, 1}, {n_embd_v, 1}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias)) {
            const int64_t n_ff = hparams.n_ff(il);
            // some models such as Phi 3 have fused up + gate tensors named "up" tensors, which need to be segmented
            if (tensor->ne[axis] == 2*n_ff) {
                return {{n_ff, 2}};
            }
            return {{tensor->ne[axis], 1}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            const int64_t n_ff_exp = hparams.n_ff_exp(il);
            GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
            return {{n_ff_exp, 2}};
        }
        return {{tensor->ne[axis], 1}};
    };

    auto get_split_granularity = [&](int64_t blck_size, uint32_t il, const std::vector<std::pair<int64_t, uint32_t>> & segments) -> std::vector<int64_t> {
        // for better performance it may make sense to round up blck_size to a higher power of 2 so that more efficient kernels can be used
        if (hparams.is_recr(il)) {
            // linear attention
            const int64_t head_dim        = hparams.ssm_d_state;
            const int64_t blck_size_perf  = std::lcm(blck_size, 128);
            const int64_t granularity_qkv = std::lcm(blck_size_perf, head_dim);
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_attn_gate_weight) ||
                    std::regex_match(tensor_name, pattern_ssm_conv1d) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv);
            }
            if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                    std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv / head_dim);
            }
            if (std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
                return std::vector<int64_t>(segments.size(), 2 * (granularity_qkv / head_dim));
            }
            if (std::regex_match(tensor_name, pattern_r_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * (hparams.ssm_d_conv - 1));
            }
            if (std::regex_match(tensor_name, pattern_s_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * head_dim);
            }
        } else {
            // regular attention
            const uint32_t n_gqa    = hparams.n_gqa(il);
            const uint32_t n_embd_q = n_gqa * hparams.n_embd_head_k(il);

            // to handle head sizes like 80, only increase granularity while it doesn't cause underutilization
            int64_t blck_size_perf = blck_size;
            while (blck_size_perf < 128 && blck_size_perf*ud->n_devices < n_embd_q) {
                blck_size_perf *= 2;
            }

            const int64_t granularity_q    = std::lcm(n_embd_q, blck_size_perf);
            const int64_t granularity_head = granularity_q / hparams.n_embd_head_k(il); // for tensors with one value per head
            if (std::regex_match(tensor_name, pattern_attn_sinks)) {
                GGML_ASSERT(segments.size() == 1);
                if (is_dsv4) {
                    return {hparams.n_head(il) / hparams.dsv4_o_group_count};
                }
                return {granularity_head};
            }

            if (is_dsv4) {
                if (std::regex_match(tensor_name, pattern_attn_q_b_weight)) {
                    GGML_ASSERT(segments.size() == 1);
                    // the grouped output projection requires each device to hold whole groups of heads
                    const int64_t n_head_group = hparams.n_head(il) / hparams.dsv4_o_group_count;
                    return {n_head_group * hparams.n_embd_head_k(il)};
                }
                if (std::regex_match(tensor_name, pattern_attn_out_a_weight)) {
                    GGML_ASSERT(segments.size() == 1);
                    return {1};
                }
                if (std::regex_match(tensor_name, pattern_attn_out_b_weight)) {
                    GGML_ASSERT(segments.size() == 1);
                    // the boundaries must align with wo_a's per-group split, so quant blocks must not straddle groups
                    GGML_ASSERT(hparams.dsv4_o_lora_rank % blck_size == 0);
                    return {hparams.dsv4_o_lora_rank};
                }
            }
            if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_q_bias)) {
                GGML_ASSERT(segments.size() == 1);
                // some models have Q gate tensors, for those cases the granularity needs to be doubled:
                // TODO: deduplicate condition [TAG_SPLIT_QGATE_QWEN]
                if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE ||
                        ud->model->arch == LLM_ARCH_QWEN4EXP) {
                    return {std::lcm(2*n_embd_q, blck_size_perf)};
                }
                return {granularity_q};
            }
            if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_head * hparams.n_embd_head_v(il)};
            }
            if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
                GGML_ASSERT(segments.size() == 1);
                if (tensor->ne[1] == hparams.n_head(il)) {
                    return {granularity_head};
                }
                return {granularity_q};
            }

            const int64_t granularity_kv = granularity_q / n_gqa;
            // the V head size can differ from the K head size (e.g. MiMo d_k=192 d_v=128):
            // align V tensors to whole V heads at the same head-index scale as Q and K so all
            // three stay in lockstep per device
            const int64_t granularity_v  = (granularity_kv / hparams.n_embd_head_k(il)) * hparams.n_embd_head_v(il);
            if (std::regex_match(tensor_name, pattern_kv_weight) ||
                std::regex_match(tensor_name, pattern_kv_bias) ||
                std::regex_match(tensor_name, pattern_kv_cache)) {
                GGML_ASSERT(segments.size() == 1);
                const bool is_v = tensor_name.find("attn_v") != std::string::npos || tensor_name.find("cache_v") != std::string::npos;
                return {is_v ? granularity_v : granularity_kv};
            }
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
                // fused full attention layers need Q gate tensors handled like above:
                // TODO: deduplicate condition [TAG_SPLIT_QGATE_QWEN]
                if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE ||
                        ud->model->arch == LLM_ARCH_QWEN4EXP) {
                    return {std::lcm(2*n_embd_q, blck_size_perf), granularity_kv};
                }
                if (segments.size() == 3) {
                    // uneven K/V head sizes: per-segment granularity, V aligned to whole V heads
                    return {granularity_q, granularity_kv, granularity_v};
                }
                GGML_ASSERT(segments.size() == 2);
                return {granularity_q, granularity_kv};
            }
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_weight) || std::regex_match(tensor_name, pattern_ffn_gate_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_up_weight) ||
                std::regex_match(tensor_name, pattern_ffn_down_weight) ||
                std::regex_match(tensor_name, pattern_ffn_up_shexp_weight) ||
                std::regex_match(tensor_name, pattern_ffn_gate_shexp_weight) ||
                std::regex_match(tensor_name, pattern_ffn_down_shexp_weight)) {
            const int64_t blck_size_perf = std::lcm(blck_size, 128);
            GGML_ASSERT(segments.size() == 1);
            return {blck_size_perf};
        }

        // everything else
        GGML_ASSERT(segments.size() == 1);
        return {1};
    };

    ggml_backend_meta_split_state split_state;
    memset(&split_state, 0, sizeof(split_state));
    tensor_config tc = get_tensor_config();
    split_state.axis = tc.axis;
    if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
        const int64_t blck_size = ggml_blck_size(tc.tensor_axis_0->type);
        const float * tensor_split = ud->model->tensor_split();
        std::vector<float> tensor_split_scan;
        tensor_split_scan.reserve(ud->n_devices);
        for (size_t j = 0; j < ud->n_devices; j++) {
            tensor_split_scan.push_back(tensor_split == nullptr ? 0.0f : tensor_split[(j + tc.rotation) % ud->n_devices]);
            if (j > 0) {
                tensor_split_scan[j] += tensor_split_scan[j - 1];
            }
        }
        const std::vector<std::pair<int64_t, uint32_t>> segments = get_split_segments(split_state.axis, tc.il);
        const std::vector<int64_t> granularity = get_split_granularity(blck_size, tc.il, segments);
        for (size_t is = 0; is < segments.size(); is++) {
            const int64_t  ne_s = segments[is].first;
            const uint32_t nr_s = segments[is].second;
            const int64_t  g_s  = granularity[is];
            int64_t low = 0;
            size_t j = 0;
            for (; j < ud->n_devices - 1; j++) {
                int64_t high = tensor_split_scan.back() == 0.0f ?
                    ne_s * (j+1)/ud->n_devices : ne_s * tensor_split_scan[j]/tensor_split_scan.back();
                if (high % g_s != 0) {
                    high -= high % g_s;
                }
                split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = high - low;
                low = high;
            }
            split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = ne_s - low;
            split_state.nr[is] = nr_s;
        }
        split_state.n_segments = segments.size();
    } else {
        memset(split_state.ne, 0, sizeof(split_state.ne));
        split_state.nr[0] = 1;
        split_state.n_segments = 1;
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            GGML_ASSERT(tc.tensor_axis_0 != tensor);
            const ggml_backend_meta_split_state source_split_state = llama_meta_device_get_split_state(tc.tensor_axis_0, userdata);
            GGML_ASSERT(source_split_state.axis >= 0 && source_split_state.axis < GGML_MAX_DIMS);
            for (size_t j = 0; j < ud->n_devices; j++) {
                for (size_t is = 0; is < source_split_state.n_segments; is++) {
                    split_state.ne[j] += source_split_state.ne[is*ud->n_devices + j] * source_split_state.nr[is];
                }
            }
        }
    }
    return split_state;
    GGML_UNUSED(userdata);
}

// The archs whose create_tensor marks token_embd TENSOR_READ_LAZY (models/bailingmoe3.cpp). Only an arch whose input table is not
// also its output matrix may be listed (a tied table would be read in full by every word's output projection). Ling = bailingmoe3;
// the other archs (Qwen3.x ... tied-embedding check first) with their ports. bailingmoe3 marks it only
// when the file has a separate output.weight (a tied table would be read in full by every word) - the accounting and the plan test that too
static bool llama_arch_marks_token_embd_lazy(llm_arch a) {
    return a == LLM_ARCH_BAILINGMOE3 || a == LLM_ARCH_FLINT ||   // Flint (its own output.weight, flint.cpp)
           a == LLM_ARCH_QWEN4EXP;                               // qwen4exp (the same test in qwen4exp.cpp)
}

// The other tables outside the layers whose create_tensor marks them TENSOR_READ_LAZY: the per-layer token table of qwen4exp (its
// n-gram table, tens of GiB, often a file of its own) and of gemma4. Only the rows a token uses are read, from the mapping; the
// table is never held in RAM. Must agree with the arch's create_tensor flags (gemma3n loads the same tensor in full)
static bool llama_arch_marks_tensor_lazy(llm_arch a, const char * name) {
    return strcmp(name, "per_layer_token_embd.weight") == 0 && (a == LLM_ARCH_QWEN4EXP || a == LLM_ARCH_GEMMA4);
}

// AUTO lazy mode falls back to OFF when any device cannot mmap (iGPUs, #28160). The loader and
// llama_moe_stream_get_plan ask the same two backend calls through this one function, so the plan cannot disagree with the load
static bool llama_devices_all_mmap(const std::vector<ggml_backend_dev_t> & devs) {
    for (ggml_backend_dev_t d : devs) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(d, &props);
        if (!props.caps.mmap_support) {
            return false;
        }
    }
    return true;
}

const char * llm_type_name(llm_type type) {
    switch (type) {
        case LLM_TYPE_14M:           return "14M";
        case LLM_TYPE_17M:           return "17M";
        case LLM_TYPE_22M:           return "22M";
        case LLM_TYPE_33M:           return "33M";
        case LLM_TYPE_47M:           return "47M";
        case LLM_TYPE_60M:           return "60M";
        case LLM_TYPE_70M:           return "70M";
        case LLM_TYPE_80M:           return "80M";
        case LLM_TYPE_109M:          return "109M";
        case LLM_TYPE_137M:          return "137M";
        case LLM_TYPE_140M:          return "140M";
        case LLM_TYPE_149M:          return "149M";
        case LLM_TYPE_160M:          return "160M";
        case LLM_TYPE_190M:          return "190M";
        case LLM_TYPE_220M:          return "220M";
        case LLM_TYPE_230M:          return "230M";
        case LLM_TYPE_250M:          return "250M";
        case LLM_TYPE_256M:          return "256M";
        case LLM_TYPE_270M:          return "270M";
        case LLM_TYPE_335M:          return "335M";
        case LLM_TYPE_350M:          return "350M";
        case LLM_TYPE_360M:          return "360M";
        case LLM_TYPE_395M:          return "395M";
        case LLM_TYPE_410M:          return "410M";
        case LLM_TYPE_450M:          return "450M";
        case LLM_TYPE_475M:          return "475M";
        case LLM_TYPE_558M:          return "558M";
        case LLM_TYPE_700M:          return "700M";
        case LLM_TYPE_770M:          return "770M";
        case LLM_TYPE_780M:          return "780M";
        case LLM_TYPE_950M:          return "950M";
        case LLM_TYPE_0_3B:          return "0.3B";
        case LLM_TYPE_0_5B:          return "0.5B";
        case LLM_TYPE_0_6B:          return "0.6B";
        case LLM_TYPE_0_8B:          return "0.8B";
        case LLM_TYPE_1B:            return "1B";
        case LLM_TYPE_1_2B:          return "1.2B";
        case LLM_TYPE_1_3B:          return "1.3B";
        case LLM_TYPE_1_4B:          return "1.4B";
        case LLM_TYPE_1_5B:          return "1.5B";
        case LLM_TYPE_1_6B:          return "1.6B";
        case LLM_TYPE_1_7B:          return "1.7B";
        case LLM_TYPE_1_8B:          return "1.8B";
        case LLM_TYPE_2B:            return "2B";
        case LLM_TYPE_2_6B:          return "2.6B";
        case LLM_TYPE_2_8B:          return "2.8B";
        case LLM_TYPE_2_9B:          return "2.9B";
        case LLM_TYPE_3B:            return "3B";
        case LLM_TYPE_4B:            return "4B";
        case LLM_TYPE_6B:            return "6B";
        case LLM_TYPE_6_9B:          return "6.9B";
        case LLM_TYPE_7B:            return "7B";
        case LLM_TYPE_8B:            return "8B";
        case LLM_TYPE_9B:            return "9B";
        case LLM_TYPE_11B:           return "11B";
        case LLM_TYPE_12B:           return "12B";
        case LLM_TYPE_13B:           return "13B";
        case LLM_TYPE_14B:           return "14B";
        case LLM_TYPE_15B:           return "15B";
        case LLM_TYPE_16B:           return "16B";
        case LLM_TYPE_20B:           return "20B";
        case LLM_TYPE_26B:           return "26B";
        case LLM_TYPE_27B:           return "27B";
        case LLM_TYPE_30B:           return "30B";
        case LLM_TYPE_31B:           return "31B";
        case LLM_TYPE_32B:           return "32B";
        case LLM_TYPE_34B:           return "34B";
        case LLM_TYPE_35B:           return "35B";
        case LLM_TYPE_36B:           return "36B";
        case LLM_TYPE_40B:           return "40B";
        case LLM_TYPE_65B:           return "65B";
        case LLM_TYPE_70B:           return "70B";
        case LLM_TYPE_120B:          return "120B";
        case LLM_TYPE_142B:          return "142B";
        case LLM_TYPE_236B:          return "236B";
        case LLM_TYPE_290B:          return "290B";
        case LLM_TYPE_314B:          return "314B";
        case LLM_TYPE_405B:          return "405B";
        case LLM_TYPE_456B:          return "456B";
        case LLM_TYPE_671B:          return "671B";
        case LLM_TYPE_SMALL:         return "0.1B";
        case LLM_TYPE_MEDIUM:        return "0.4B";
        case LLM_TYPE_LARGE:         return "0.8B";
        case LLM_TYPE_XL:            return "1.5B";
        case LLM_TYPE_A1_7B:         return "A1.7B";
        case LLM_TYPE_A2_7B:         return "A2.7B";
        case LLM_TYPE_8x7B:          return "8x7B";
        case LLM_TYPE_8x22B:         return "8x22B";
        case LLM_TYPE_16x12B:        return "16x12B";
        case LLM_TYPE_16x3_8B:       return "16x3.8B";
        case LLM_TYPE_10B_128x3_66B: return "10B+128x3.66B";
        case LLM_TYPE_57B_A14B:      return "57B.A14B";
        case LLM_TYPE_17B_16E:       return "17Bx16E (Scout)";
        case LLM_TYPE_17B_128E:      return "17Bx128E (Maverick)";
        case LLM_TYPE_A13B:          return "A13B";
        case LLM_TYPE_1B_A400M:      return "1B.A400M";
        case LLM_TYPE_3B_A800M:      return "3B.A800M";
        case LLM_TYPE_7B_A1B:        return "7B.A1B";
        case LLM_TYPE_8B_A1B:        return "8B.A1B";
        case LLM_TYPE_7_9B_A1_3B:    return "7.9B.A1.3B";
        case LLM_TYPE_12B_A2_5B:     return "12B.A2.5B";
        case LLM_TYPE_16B_A1B:       return "16B.A1B";
        case LLM_TYPE_21B_A3B:       return "21B.A3B";
        case LLM_TYPE_24B_A2B:       return "24B.A2B";
        case LLM_TYPE_26B_A4B:       return "26B.A4B";
        case LLM_TYPE_30B_A3B:       return "30B.A3B";
        case LLM_TYPE_31B_A3_5B:     return "31B.A3.5B";
        case LLM_TYPE_32B_A9B:       return "32B.A9B";
        case LLM_TYPE_35B_A3B:       return "35B.A3B";
        case LLM_TYPE_48B_A3B:       return "48B.A3B";
        case LLM_TYPE_75B_A9B:       return "75B.A9B";
        case LLM_TYPE_80B_A3B:       return "80B.A3B";
        case LLM_TYPE_A3B:           return "A3B";
        case LLM_TYPE_100B_A6B:      return "100B.A6B";
        case LLM_TYPE_102B_A12B:     return "102B.A12B";
        case LLM_TYPE_106B_A12B:     return "106B.A12B";
        case LLM_TYPE_118B_A8B:      return "118B.A8B";
        case LLM_TYPE_120B_A12B:     return "120B.A12B";
        case LLM_TYPE_122B_A10B:     return "122B.A10B";
        case LLM_TYPE_124B_A5_1B:    return "124B.A5.1B";
        case LLM_TYPE_196B_A11B:     return "196B.A11B";
        case LLM_TYPE_230B_A10B:     return "230B.A10B";
        case LLM_TYPE_428B_A23B:     return "428B.A23B";
        case LLM_TYPE_235B_A22B:     return "235B.A22B";
        case LLM_TYPE_288B_A19B:     return "288B.A19B";
        case LLM_TYPE_300B_A47B:     return "300B.A47B";
        case LLM_TYPE_310B_A15B:     return "310B.A15B";
        case LLM_TYPE_355B_A32B:     return "355B.A32B";
        case LLM_TYPE_397B_A17B:     return "397B.A17B";
        case LLM_TYPE_685B_A37B:     return "685B.A37B";
        case LLM_TYPE_744B_A40B:     return "744B.A40B";
        case LLM_TYPE_2_8T_A50B:     return "2.8T.A50B";
        case LLM_TYPE_E2B:           return "E2B";
        case LLM_TYPE_E4B:           return "E4B";
        default:                     return "?B";
    }
}

static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type) {
    switch (type) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX: return "softmax";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID: return "sigmoid";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS: return "sqrtsoftplus";
        default:                                    return "unknown";
    }
}

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LLAMA_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type) {
    return LLAMA_ROPE_SCALING_TYPES.at(rope_scaling_type);
}

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

// Maps the GGUF `<arch>.hidden_activation` string to the FFN op type used by the
// graph builders. Only gated activations that map cleanly to llm_ffn_op_type are
// listed; unrecognized values fall back to GeGLU, which matches the historical
// default for ModernBert-style architectures.
static const std::map<std::string, llm_ffn_op_type> LLM_FFN_OP_TYPES_FROM_STRING = {
    { "gelu",   LLM_FFN_GEGLU  },
    { "geglu",  LLM_FFN_GEGLU  },
    { "silu",   LLM_FFN_SWIGLU },
    { "swish",  LLM_FFN_SWIGLU },
    { "swiglu", LLM_FFN_SWIGLU },
    { "relu",   LLM_FFN_RELU   },
    { "reglu",  LLM_FFN_REGLU  },
};

llm_ffn_op_type llm_ffn_op_type_from_string(const std::string & name, llm_ffn_op_type fallback) {
    const auto it = LLM_FFN_OP_TYPES_FROM_STRING.find(name);
    if (it != LLM_FFN_OP_TYPES_FROM_STRING.end()) {
        return it->second;
    }
    return fallback;
}

// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<llama_device> & devices, bool use_extra_bufts, bool no_host) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    if (!no_host) {
        for (const auto & dev : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev.dev);
            if (buft) {
                buft_list.emplace_back(dev.dev, buft);
                break;
            }
        }
    }

    // add extra buffer types
    if (use_extra_bufts) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }

        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }

    return buft_list;
}

// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, llama_split_mode split_mode, const float * tensor_split) {
    buft_list_t buft_list;

    // add the device split buffer type if requested and available
    if (split_mode == LLAMA_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (ggml_backend_split_buffer_type_fn) {
            size_t dev_index = [&]() {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                    if (ggml_backend_reg_dev_get(reg, i) == dev) {
                        return i;
                    }
                }
                throw std::runtime_error(format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
            }();
            auto * buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
            if (buft != nullptr) {
                buft_list.emplace_back(dev, buft);
            }
        } else {
            throw std::runtime_error(format("device %s does not support split buffers", ggml_backend_dev_name(dev)));
        }
    }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    // add the device extra buffer type (if any)
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg) {
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    return buft_list;
}

struct llama_model::impl {
    impl() = default;
    ~impl() = default;

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored as well as the corresponding buffers:
    std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> ctxs_bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    bool has_tensor_overrides;

    // MoE expert SSD streaming state, null when not enabled
    std::unique_ptr<llama_moe_stream> moe_stream;

    std::vector<float> tensor_split_owned;
};

bool llama_prec_policy::apply(ggml_tensor * res) const {
    if (!res || !res->src[0]) {
        return false;
    }

    const auto it = prec_src1.find(res->src[0]);
    if (it == prec_src1.end()) {
        return false;
    }

    return ggml_prec_set_src(res, it->second, 1);
}

void llama_prec_policy::load(llama_model_loader & ml, const llama_model & model) {
    std::vector<std::string> tensor_names;
    if (!ml.get_arr(LLM_KV_GENERAL_TENSOR_EXTRA_NAME, tensor_names, false)) {
        return;
    }

    const gguf_context * ctx = ml.metadata;
    const std::string key = ml.llm_kv(LLM_KV_GENERAL_TENSOR_EXTRA_PREC_A4);
    const int kid = gguf_find_key(ctx, key.c_str());
    if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, kid) != GGUF_TYPE_BOOL) {
        throw std::runtime_error(format("%s must be a bool array", key.c_str()));
    }

    const size_t n_values = gguf_get_arr_n(ctx, kid);
    if (n_values != tensor_names.size()) {
        throw std::runtime_error(format(
            "%s tensor/value length mismatch (%zu vs %zu)",
            key.c_str(), tensor_names.size(), n_values));
    }

    // tensors that can not use 4-bit activations, keep src1 at higher precision
    const int8_t * values = (const int8_t *) gguf_get_arr_data(ctx, kid);
    std::unordered_set<std::string> want;
    for (size_t i = 0; i < n_values; ++i) {
        if (values[i] == 0) {
            want.insert(tensor_names[i]);
        }
    }

    // resolve names to tensor pointers
    for (const auto & [name, w] : model.tensors_by_name) {
        if (want.count(name)) {
            prec_src1.emplace(w, GGML_PREC_Q8);
        }
    }
}

static thread_local bool llama_model_alloc_host = false;   // the last weight-buffer failure was a host buffer's
bool llama_model_last_alloc_failed_on_host() { return llama_model_alloc_host; }
// the whole GPU MoE layer that load would give up: the last picked on the device whose buffer failed; -1 = none there, -2 = not known
static thread_local int32_t     llama_model_alloc_drop = -2;
static thread_local std::string llama_model_auto_gl_list;
static thread_local bool        llama_model_wl_retry = false;
int32_t llama_model_last_alloc_drop_layer() { return llama_model_alloc_drop; }
const std::string & llama_model_last_auto_gl_list() { return llama_model_auto_gl_list; }
void llama_model_set_wl_retry(bool v) { llama_model_wl_retry = v; }

llama_model::llama_model(const llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {
    llama_model_alloc_drop = -2;   // (a failure of this load says which whole layer to give up)
    llama_model_auto_gl_list.clear();
    if (params.tensor_split != nullptr) {
        // llama_model_params stores tensor_split as a borrowed pointer, but the model
        // may need it later for tensor-parallel KV-cache split metadata.
        pimpl->tensor_split_owned.assign(params.tensor_split, params.tensor_split + llama_max_devices());
        this->params.tensor_split = pimpl->tensor_split_owned.data();
    }
    pimpl->has_tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}

llama_model::~llama_model() {
    for (auto * lora : loras) {
        delete lora;
    }
}

void llama_model_base::load_stats(llama_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes = ml.n_bytes;
}

void llama_model_base::load_hparams(llama_model_loader & ml) {
    const gguf_context * ctx = ml.metadata;

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    // everything past this point is not vocab-related
    // for CLIP models, we only need to load tensors, no hparams
    if (hparams.vocab_only || ml.get_arch() == LLM_ARCH_CLIP) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,          hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,        hparams.n_embd);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_OUT,    hparams.n_embd_out_impl, false);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,        hparams.causal_attn,     false);
    ml.get_key(LLM_KV_POOLING_TYPE,            hparams.pooling_type,    false);
    ml.get_key(LLM_KV_BLOCK_COUNT,             hparams.n_layer_all);
    GGML_ASSERT(hparams.n_layer_all > 0 && hparams.n_layer_all <= LLAMA_MAX_LAYERS);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,    hparams.n_layer_nextn,   false);
    GGML_ASSERT(hparams.n_layer_nextn <= hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_COUNT,            hparams.n_expert,        false);
    std::fill(hparams.n_expert_used_arr.begin(), hparams.n_expert_used_arr.end(), 0);
    ml.get_key_or_arr(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,      hparams.n_expert_groups, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT, hparams.n_group_used,    false);

    if (arch == LLM_ARCH_HUNYUAN_VL || arch == LLM_ARCH_HUNYUAN_DENSE) {
        if (hparams.n_expert <= 1) {
            hparams.n_expert = 0;
            std::fill(hparams.n_expert_used_arr.begin(), hparams.n_expert_used_arr.end(), 0);
        }
    }

    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        ml.get_key(LLM_KV_FEATURES_LENGTH,  hparams.n_embd);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH, hparams.n_embd_out_impl);

        ml.get_key(LLM_KV_POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd);
        ml.get_key(LLM_KV_POSNET_BLOCK_COUNT,      hparams.posnet.n_layer);

        ml.get_key(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
        ml.get_key(LLM_KV_CONVNEXT_BLOCK_COUNT,      hparams.convnext.n_layer);

        GGML_ASSERT(hparams.posnet.n_layer   <= hparams.n_layer_all);
        GGML_ASSERT(hparams.convnext.n_layer <= hparams.n_layer_all);
    }

    // models may route a different number of experts per layer, so validate the maximum
    uint32_t n_expert_used_max = hparams.n_expert_used_max();

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(n_expert_used_max <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(n_expert_used_max > 0);
        GGML_ASSERT(hparams.n_expert_groups < hparams.n_expert);
        if (hparams.n_expert_groups > 1) {
            GGML_ASSERT(hparams.n_expert % hparams.n_expert_groups == 0);
            GGML_ASSERT(hparams.n_group_used > 0);
            GGML_ASSERT(hparams.n_group_used < hparams.n_expert_groups);
        }
    } else {
        GGML_ASSERT(n_expert_used_max == 0);
        GGML_ASSERT(hparams.n_expert_groups == 0);
    }

    std::fill(hparams.n_head_arr.begin(),       hparams.n_head_arr.end(),       0);
    std::fill(hparams.n_head_kv_arr.begin(),    hparams.n_head_kv_arr.end(),    0);
    std::fill(hparams.n_ff_arr.begin(),         hparams.n_ff_arr.end(),         0);
    std::fill(hparams.n_ff_exp_arr.begin(),     hparams.n_ff_exp_arr.end(),     0);

    std::fill(hparams.rope_sections.begin(), hparams.rope_sections.end(), 0);
    std::fill(hparams.rope_pattern.begin(),  hparams.rope_pattern.end(), 1);
    std::fill(hparams.is_swa_impl.begin(),   hparams.is_swa_impl.end(), 0);
    std::fill(hparams.is_recr_impl.begin(),  hparams.is_recr_impl.end(),  llm_arch_is_recurrent(ml.get_arch()) ? 1 : 0);
    std::fill(hparams.is_indexer_full_impl.begin(), hparams.is_indexer_full_impl.end(), 0);

    std::fill(hparams.xielu_alpha_n.begin(), hparams.xielu_alpha_n.end(), 0.0f);
    std::fill(hparams.xielu_alpha_p.begin(), hparams.xielu_alpha_p.end(), 0.0f);
    std::fill(hparams.xielu_beta.begin(),    hparams.xielu_beta.end(), 0.0f);
    std::fill(hparams.xielu_eps.begin(),     hparams.xielu_eps.end(), 0.0f);

    std::fill(hparams.swiglu_clamp_exp.begin(),   hparams.swiglu_clamp_exp.end(),   0.0f);
    std::fill(hparams.swiglu_clamp_shexp.begin(), hparams.swiglu_clamp_shexp.end(), 0.0f);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer_all, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer_all, false);

    // Populate deepstack_mapping_arr - initialized to -1 (no deepstack)
    std::fill(hparams.deepstack_mapping_arr.begin(), hparams.deepstack_mapping_arr.end(), -1);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer_all, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // TODO: Handle SWA metadata similarly when models start implementing it
    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);
    ml.get_key(LLM_KV_ROPE_SCALING_ALPHA,       hparams.rope_scaling_alpha, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k_full, false);

        hparams.n_embd_head_v_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_full, false);

        // sanity check for n_rot (optional)
        hparams.n_rot_full = hparams.n_embd_head_k_full;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot_full, false);

        if (arch == LLM_ARCH_LLAMA || arch == LLM_ARCH_DECI || arch == LLM_ARCH_FALCON || arch == LLM_ARCH_LLAMA_EMBED) {
            if (hparams.n_rot_full != hparams.n_embd_head_k_full) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot_full, hparams.n_embd_head_k_full));
            }
        }
    } else {
        hparams.n_rot_full = 0;
        hparams.n_embd_head_k_full = 0;
        hparams.n_embd_head_v_full = 0;
    }

    // head size and n_rot for SWA layers
    {
        hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
        hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA, hparams.n_embd_head_k_swa, false);
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa, false);

        hparams.n_rot_swa = hparams.n_rot_full;
        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);
    }

    // for classifier models
    ml.get_arr(LLM_KV_CLASSIFIER_OUTPUT_LABELS, classifier_labels, false);
    if (!classifier_labels.empty()) {
        hparams.n_cls_out = classifier_labels.size();
    }

    // per-arch hparams
    load_arch_hparams(ml);

    pimpl->n_bytes = ml.n_bytes;

    pimpl->desc_str = arch_name() + " " + type_name() + " " + ml.ftype_name();

    pimpl->ftype = ml.ftype;

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_model_rope_type(this);
}

void llama_model_base::load_vocab(llama_model_loader & ml) {
    const auto kv = LLM_KV(arch);

    vocab.load(ml, kv);
}

// The allowance the pools leave for the allocations that are not weights (see the constant)
// Static - only this file uses it
static int64_t llama_moe_stream_other_reserve() {
    return MOE_STREAM_OTHER_RESERVE;
}

// A server's memory next to the engine, from the model's own dimensions and the server's settings
// (the same function for the load and the plan)
struct llama_moe_stream_srv_terms {
    int64_t snap    = 0;   // one context snapshot: the memory parts that cannot be rolled back (running states, SWA window)
    int64_t kv_tok  = 0;   // the attention layers' cache bytes per token of a saved state
    int64_t ckpts   = 0;   // --ctx-checkpoints x slots x snap
    int64_t restore = 0;   // with --cache-disk: the largest state file in RAM (a full context, its snapshots), one at a time
    int64_t need    = 0;   // the peak of both - the restoring slot's own snapshots are freed before the file is
                           // read, so that slot holds max(its snapshots, the restore buffer), the other slots their snapshots
    int64_t stage   = 0;   // the in-call checkpoints' staging (--ctx-checkpoints + 1 snapshots), kept
};

// The bytes per token, on one attention layer, of a sparse-attention indexer's own cache (llama_memory_hybrid_idx, which
// create_memory builds for qwen4exp: one key row of indexer_head_size x the k-pool row's slots per token, in the K cache type, no
// V); 0 for a model without one. The KV cache sizes below add it on the layers that have it (the attention layers)
static int64_t llama_moe_stream_idx_row(const llama_hparams & hp, llm_arch arch, ggml_type tk) {
    if (arch != LLM_ARCH_QWEN4EXP || hp.indexer_head_size == 0) {
        return 0;
    }
    return (int64_t) ggml_row_size(tk, (int64_t) hp.indexer_head_size*(hp.indexer_kpool > 0 ? hp.indexer_kpool_row : 1));
}

static llama_moe_stream_srv_terms llama_moe_stream_srv_compute(const llama_hparams & hp, llm_arch arch, const llama_moe_stream_srv & srv) {
    llama_moe_stream_srv_terms t;
    const uint32_t n_seq     = srv.n_seq > 0 ? srv.n_seq : 1;
    const uint32_t n_ctx     = srv.n_ctx > 0 ? srv.n_ctx : hp.n_ctx_train;
    const uint32_t n_ctx_seq = srv.kv_unified ? n_ctx : std::max(1u, n_ctx / n_seq);
    const ggml_type tk = (ggml_type) srv.type_k;
    const ggml_type tv = (ggml_type) srv.type_v;

    bool any_recr = false;
    for (uint32_t il = 0; il < hp.n_layer(); ++il) {
        if (hp.is_recr(il)) {
            // the recurrent memory keeps its states in F32 (llama_model::create_memory: recurrent_type_r / _s)
            t.snap += (int64_t) ggml_row_size(GGML_TYPE_F32, hp.n_embd_r()) + (int64_t) ggml_row_size(GGML_TYPE_F32, hp.n_embd_s());
            any_recr = true;
            continue;
        }
        const int64_t kv_row = (int64_t) ggml_row_size(tk, hp.n_embd_k_gqa(il)) + (int64_t) ggml_row_size(tv, hp.n_embd_v_gqa(il)) +
                llama_moe_stream_idx_row(hp, arch, tk);   // (the indexer's rows are saved with the state too)
        if (hp.is_swa(il) && !srv.swa_full) {
            // the window cannot be rolled back. A sequence's state writes only the cells inside the window
            // (llama_kv_cache::state_write skips the SWA-masked ones), n_swa per layer - not n_swa + n_ubatch, the cache's size;
            // and its rows are in the snapshot, so they are not counted again per token of a saved state
            t.snap += kv_row * (int64_t) std::min<uint32_t>(n_ctx_seq, hp.n_swa);
            continue;
        }
        t.kv_tok += kv_row;
    }
    // a model whose memory can all be rolled back makes no snapshots
    t.ckpts   = t.snap > 0 ? (int64_t) std::max(0, srv.n_ctx_checkpoints) * n_seq * t.snap : 0;
    t.restore = srv.state_files ? t.snap * (1 + std::max(0, srv.n_ctx_checkpoints)) + t.kv_tok * n_ctx_seq : 0;
    const int64_t per_slot = n_seq > 0 ? t.ckpts / n_seq : 0;
    t.need    = t.ckpts - per_slot + std::max(per_slot, t.restore);
    // in-call checkpoints copy recurrent rows (llama_context::call_checkpoints_ok needs the hybrid memory's recurrent part),
    // so a model without recurrent layers (Flint's sliding window) never stages them
    t.stage   = srv.call_checkpoints && any_recr && t.snap > 0 ? (int64_t) (std::max(0, srv.n_ctx_checkpoints) + 1) * t.snap : 0;
    t.need   += t.stage;
    return t;
}

// One layer's KV cache rows or recurrent state for the context the model is loaded for (llama_model_params' moe_stream_ctx*), at
// the sizes llama_kv_cache, llama_kv_cache_iswa and llama_memory_recurrent allocate for the plain, SWA and hybrid layouts; the
// whole-layer VRAM check charges it to the layer's device. A sparse-attention indexer's own cache (qwen4exp) is counted with its layer;
// other caches of their own (a compressor's state) are not
static int64_t llama_moe_stream_layer_ctx_bytes(const llama_hparams & hp, llm_arch arch, uint32_t il, const llama_model_params & p) {
    const bool     unified   = (p.moe_stream_ctx_flags & 1) != 0;
    const bool     swa_full  = (p.moe_stream_ctx_flags & 2) != 0;
    const uint32_t n_seq     = std::max(1u, p.moe_stream_ctx_seq);
    const uint32_t n_ubatch  = p.moe_stream_ubatch > 0 ? (uint32_t) p.moe_stream_ubatch : 512u;
    const uint32_t n_ctx     = GGML_PAD(p.moe_stream_ctx > 0 ? p.moe_stream_ctx : hp.n_ctx_train, 256);   // (llama_context pads it so)
    const uint32_t n_ctx_seq = unified ? n_ctx : GGML_PAD(n_ctx/n_seq, 256);
    const uint32_t n_stream  = unified ? 1 : n_seq;
    const bool     hybrid    = llm_arch_is_hybrid(arch);
    if (llm_arch_is_recurrent(arch) || (hybrid && hp.is_recr(il))) {
        // the recurrent memory keeps its states in F32, max(1, n_seq) x (1 + the rollback rows) rows per layer
        const uint32_t rs_seq = llm_arch_supports_rs_rollback(arch) ? p.moe_stream_ctx_rs_seq : 0;
        int64_t row = (int64_t) ggml_row_size(GGML_TYPE_F32, hp.n_embd_r()) + (int64_t) ggml_row_size(GGML_TYPE_F32, hp.n_embd_s());
        if (hp.ple_conv_state() > 0 && hp.is_ple(il)) {
            row += (int64_t) ggml_row_size(GGML_TYPE_F32, hp.ple_conv_state());
        }
        return row*(int64_t) n_seq*(1 + rs_seq);
    }
    if (!hp.has_kv(il)) {
        return 0;
    }
    if (!hybrid && hp.n_layer_nextn > 0 && hp.n_layer() > 0 && hp.router_layer < 0 && il >= hp.n_layer()) {
        return 0;   // the MTP block's cache belongs to the draft context
    }
    const bool     swa     = hp.swa_type != LLAMA_SWA_TYPE_NONE && hp.is_swa(il) && !swa_full;
    const uint32_t kv_size = swa ? GGML_PAD(std::min(n_ctx_seq, hp.n_swa*(unified ? n_seq : 1) + n_ubatch), 256) : n_ctx_seq;
    const int64_t  k = (int64_t) ggml_row_size((ggml_type) p.moe_stream_ctx_type_k, hp.n_embd_k_gqa(il));
    const int64_t  v = hp.is_mla() ? 0 : (int64_t) ggml_row_size((ggml_type) p.moe_stream_ctx_type_v, hp.n_embd_v_gqa(il));   // (MLA: V is a view of K)
    const int64_t  idx = swa ? 0 : llama_moe_stream_idx_row(hp, arch, (ggml_type) p.moe_stream_ctx_type_k);   // the indexer cache: as long as the KV cache
    return (k + v + idx)*(int64_t) kv_size*(int64_t) n_stream;
}

// the bytes to add to the non-pool memory for it (0 when it fits in the reserve's headroom), and the start-up line
static int64_t llama_moe_stream_srv_extra(const llama_moe_stream_srv_terms & t, const llama_moe_stream_srv & srv, const char * func) {
    const int64_t rest     = srv.other_rest > 0 ? srv.other_rest : MOE_STREAM_OTHER_REST;   // A8
    const int64_t headroom = MOE_STREAM_OTHER_RESERVE - rest;
    const int64_t need     = t.need;
    const int64_t extra    = std::max<int64_t>(0, need - headroom);
    const double  M = 1024.0*1024.0;
    fprintf(stderr, "%s: the server's memory in the RAM plan: the reserve %.0f MiB = the rest %.0f MiB (%s) + headroom "
            "%.0f MiB; snapshots %d x %u slot(s) x %.1f MiB = %.1f MiB, the state-file restore buffer %.1f MiB (%s; it replaces the "
            "restoring slot's snapshots, which are freed first), the in-call checkpoints' staging %.1f MiB -> at most %.1f MiB at once; %s\n",
            func, MOE_STREAM_OTHER_RESERVE/M, rest/M,
            srv.other_rest > 0 ? "--moe-stream-other-rest" : "a measured default: Ling Flash 16k -np 1; --moe-stream-other-rest sets it",
            headroom/M, std::max(0, srv.n_ctx_checkpoints),
            srv.n_seq > 0 ? srv.n_seq : 1u, t.snap/M, t.ckpts/M, t.restore/M,
            srv.state_files ? "--cache-disk" : "no --cache-disk", t.stage/M, need/M,
            extra == 0 ? format("fits, %.0f MiB of headroom left: the pools are unchanged", (headroom - need)/M).c_str()
                       : format("does NOT fit: the non-pool memory grows by %.1f MiB (fewer slots)", extra/M).c_str());
    return extra;
}

// --moe-stream-nonpool-measured 1: what the plan holds back outside the pools, in place of the fixed reserve + the server's term beyond its
// headroom: the record of this run's own needs (KV cache and host compute buffers as the run measured them, the rest of the process in its replies)
// + the growth allowance + the server's snapshot terms in full (a computed reserve has no headroom) + --cache-ram. No record for these settings:
// the fixed reserve's numbers. The same function for the load and the plan
// --moe-stream-correction FILE: where the correction file is. The word `model` is the correction appended to the model file (a TokenWing
// model file made with a correction); a model file without one gives an empty path (no correction) and, when `say`, one line.
struct llama_moe_corr_ref {
    std::string path;
    uint64_t    base = 0, size = 0;
};

static llama_moe_corr_ref llama_moe_correction_ref(const char * value, const std::string & model_path, bool say) {
    llama_moe_corr_ref ref;

    ref.path = value ? value : "";

    if (ref.path == MOE_STREAM_MODEL_WORD) {
        if (llama_moe_model_file_blob(model_path, "correction", ref.base, ref.size)) {
            ref.path = model_path;
        } else {
            ref.path.clear();
            if (say) {
                fprintf(stderr, "llama_moe_correction: NOT used (the plain guess runs): --moe-stream-correction model, but '%s' has no "
                        "correction data (add one with tokenwing-tools/experts/make_model_file.py --correction)\n", model_path.c_str());
            }
        }
    }

    return ref;
}

static int64_t llama_moe_stream_np_bytes(const llama_moe_stream_nonpool & np, const char * sidecar, int64_t srv_need, int64_t srv_extra, const char * func) {
    const llama_moe_np_entry e = llama_moe_np_read(sidecar, np.key);
    const double M = 1024.0*1024.0;
    if (!e.found) {
        fprintf(stderr, "%s: non-pool memory: no record for these settings (key %s): the fixed reserve; the run writes the record\n", func, np.key);
        return MOE_STREAM_OTHER_RESERVE + srv_extra;
    }
    const int64_t bytes = e.kv + e.compute + e.rest + np.growth + srv_need + np.cache_ram;
    fprintf(stderr, "%s: non-pool memory from the record (key %s, %lld samples): KV in RAM %.1f + host compute buffers %.1f + the rest %.1f + growth %.1f + "
            "the server's snapshots %.1f + --cache-ram %.1f = %.1f MiB (the fixed reserve would be %.1f)\n", func, np.key, (long long) e.n, e.kv/M, e.compute/M, e.rest/M,
            np.growth/M, srv_need/M, np.cache_ram/M, bytes/M, (MOE_STREAM_OTHER_RESERVE + srv_extra)/M);
    return bytes;
}

// the routed expert weights of a MoE layer, which the expert pool replaces
static bool llama_moe_stream_is_exps(const std::string & name) {
    for (const char * suffix : { ".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight" }) {
        const size_t len = strlen(suffix);
        if (name.size() >= len && name.compare(name.size() - len, len, suffix) == 0) {
            return true;
        }
    }

    return false;
}

bool llama_model_base::load_tensors(llama_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const bool use_mlock      = params.load_mode == LLAMA_LOAD_MODE_MLOCK || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
    const auto & tensor_split = params.tensor_split;

    const int n_layer_all = hparams.n_layer_all;
    const int n_gpu_layers = this->n_gpu_layers();

    const bool use_mmap_buffer = true;

    this->ml = &ml; // to be used by create_tensor() and load_arch_tensors()

    if (ml.use_mmap && params.load_mode == LLAMA_LOAD_MODE_AUTO) {
        for (const auto & dev : devices) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev.dev, &props);
            if (!props.caps.mmap_support) {
                ml.use_mmap = false;
                break;
            }
        }
    }

    // --moe-stream on a model it cannot stream - not a MoE model, not loaded from a file, --no-alloc - runs as
    // plain llama.cpp, with one line. MOE_STREAM_TEST_E4_DENSE=1 (test only): take the model for a dense one
    if (params.moe_stream) {
        static const bool test_dense = getenv("MOE_STREAM_TEST_E4_DENSE") && atoi(getenv("MOE_STREAM_TEST_E4_DENSE")) != 0;
        const char * why = test_dense || hparams.n_expert == 0 || hparams.n_expert_used() == 0 ? "the model is not a MoE model"
                         : ml.file_paths.empty() || ml.file_paths[0].empty() ? "the model is not loaded from a file"
                         : ml.no_alloc ? "--no-alloc is set" : nullptr;
        if (why != nullptr) {
            fprintf(stderr, "%s: --moe-stream: off - %s%s; the model runs without streaming (plain llama.cpp)\n", __func__, why,
                    test_dense ? " (TEST ONLY: MOE_STREAM_TEST_E4_DENSE)" : "");
            params.moe_stream = nullptr;
        }
    }

    // resolve AUTO on systems without mmap support (e.g. iGPUs): fall back to OFF; see #28160
    // With --moe-stream the word table (a marked tensor of 64 MiB or more) is read lazily in AUTO mode
    if (params.moe_stream) {
        ml.lazy.token_embd_min_bytes = llama_model_loader::lazy_read::moe_stream_min_bytes();
    }
    if (ml.lazy.mode == LLAMA_LAZY_MODE_AUTO) {
        std::vector<ggml_backend_dev_t> lazy_devs;
        for (const auto & dev : devices) {
            lazy_devs.push_back(dev.dev);
        }
        if (!llama_devices_all_mmap(lazy_devs)) {
            ml.lazy.mode = LLAMA_LAZY_MODE_OFF;
        }
    }

    const char * load_mode_name = params.load_mode == LLAMA_LOAD_MODE_AUTO
        ? llama_load_mode_name(ml.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE)
        : llama_load_mode_name(params.load_mode);

    LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (load_mode = %s)\n",
        __func__, load_mode_name);

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts, params.no_host);
    for (const auto & dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev.dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev.dev, std::move(buft_list));
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        throw std::runtime_error(format("%s: no CPU backend found", __func__));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i].dev;
            size_t total;
            size_t free;
            ggml_backend_dev_memory(dev, &free, &total);

            // devices can return 0 bytes for free and total memory if they do not
            // have any to report. in this case, we will use the host memory as a fallback
            // fixes: https://github.com/ggml-org/llama.cpp/issues/18577
            if (free == 0 && total == 0) {
                ggml_backend_dev_memory(cpu_dev, &free, &total);
            }
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] = split_sum != 0.0f ? splits[i] / split_sum : float(i + 1) / n_devices();
    }

    const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);
    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        const bool is_swa = il < n_layer_all && hparams.is_swa(il);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer_all);
    for (int il = 0; il < n_layer_all; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer_all);

    {
        const bool fast = params.moe_stream_prefill_fast == 1;
        uint32_t cuda_pf_flags = 0;
        if (fast)                                   { cuda_pf_flags |= (1u << 5); }
        if (fast || params.moe_stream_mmq_moe_opt   == 1) { cuda_pf_flags |= (1u << 0); }
        if (fast || params.moe_stream_gpu_moe_fused == 1) { cuda_pf_flags |= (1u << 1); }
        if (fast || params.moe_stream_qsa_fused     == 1) { cuda_pf_flags |= (1u << 2); }
        if (fast || params.moe_stream_hc_fused      == 1) { cuda_pf_flags |= (1u << 3); }
        if (fast || params.moe_stream_ple_fast      == 1) { cuda_pf_flags |= (1u << 4); }
        ggml_backend_reg_t creg = ggml_backend_reg_by_name("CUDA");
        if (!creg) { creg = ggml_backend_reg_by_name("ROCm"); }
        if (creg) {
            auto * set_pf = (void (*)(uint32_t)) ggml_backend_reg_get_proc_address(creg, "ggml_backend_cuda_set_prefill_opts");
            if (set_pf) {
                set_pf(cuda_pf_flags);
            }
        }
    }

    if (params.moe_stream) {
        // (a model these do not hold for has --moe-stream cleared above, with its line)

        if (pimpl->has_tensor_overrides) {
            LLAMA_LOG_WARN("%s: -ot/--cpu-moe overrides do not apply to streamed expert tensors\n", __func__);
        }

        // More than one GPU: the device names of --moe-stream-vram-device and --moe-stream-gpu-layers-device, resolved against this
        // model's GPUs (--device decides which and in what order). A name that is not one of them falls back, with one line each
        std::string gpu_names;
        ggml_backend_dev_t first_gpu  = nullptr;
        ggml_backend_dev_t first_igpu = nullptr;
        int n_gpu_devs = 0;
        for (const auto & dv : devices) {
            const auto t = ggml_backend_dev_type(dv.dev);
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                gpu_names += format("%s%s", gpu_names.empty() ? "" : ",", ggml_backend_dev_name(dv.dev));
                n_gpu_devs++;
                first_gpu  = first_gpu  == nullptr && t == GGML_BACKEND_DEVICE_TYPE_GPU  ? dv.dev : first_gpu;
                first_igpu = first_igpu == nullptr && t == GGML_BACKEND_DEVICE_TYPE_IGPU ? dv.dev : first_igpu;
            }
        }
        if (first_gpu == nullptr && params.moe_stream_uma == 1) {
            first_gpu = first_igpu;   // --moe-stream-uma 1: an integrated GPU can be the engine's GPU (without it, it stays out as before)
        }
        const auto gpu_by_name = [&](const std::string & n, bool igpu_ok) -> ggml_backend_dev_t {
            for (const auto & dv : devices) {
                const auto t = ggml_backend_dev_type(dv.dev);
                if ((t == GGML_BACKEND_DEVICE_TYPE_GPU || (igpu_ok && t == GGML_BACKEND_DEVICE_TYPE_IGPU)) && n == ggml_backend_dev_name(dv.dev)) {
                    return dv.dev;
                }
            }
            return nullptr;
        };
        ggml_backend_dev_t ms_pool_dev = first_gpu;
        bool ms_pool_dev_set = false;
        if (params.moe_stream_vram_device && params.moe_stream_vram_device[0]) {
            if (ggml_backend_dev_t d = gpu_by_name(params.moe_stream_vram_device, /* igpu_ok = */ params.moe_stream_uma == 1)) {
                ms_pool_dev     = d;
                ms_pool_dev_set = true;
            } else {
                fprintf(stderr, "%s: --moe-stream-vram-device %s is not a GPU of this model (its GPUs: %s) - %s\n", __func__,
                        params.moe_stream_vram_device, gpu_names.empty() ? "none" : gpu_names.c_str(),
                        first_gpu ? format("the VRAM pool stays on %s", ggml_backend_dev_name(first_gpu)).c_str() : "not used");
            }
        }
        // --moe-stream-gpu-layers-device: the device of each named layer's experts. The layers themselves are the list (common fills it with
        // the named layers; a reload with one layer fewer shortens the list, not this); a layer named for a device that is not a GPU of the
        // model leaves the list (it is streamed)
        const auto split_by = [](const std::string & str, char c) {
            std::vector<std::string> out;
            size_t p = 0;
            for (size_t q; (q = str.find(c, p)) != std::string::npos; p = q + 1) {
                out.push_back(str.substr(p, q - p));
            }
            out.push_back(str.substr(p));
            return out;
        };
        // --moe-stream-vram-free as a list per GPU: each GPU keeps its value free (a GPU not named keeps moe_stream_vram_free: the
        // list's value without a name, or its largest). Without a list: 0 here, the single value sizes the pool alone as before
        const bool keep_list = params.moe_stream_vram_free_dev && params.moe_stream_vram_free_dev[0];
        std::map<std::string, int64_t> keep_named;
        if (keep_list) {
            std::string bad_keep;
            for (const std::string & item : split_by(params.moe_stream_vram_free_dev, ',')) {
                const size_t eq = item.find('=');
                if (eq == std::string::npos) {
                    continue;
                }
                keep_named[item.substr(0, eq)] = std::max<int64_t>(0, atoll(item.c_str() + eq + 1));
                if (gpu_by_name(item.substr(0, eq), /* igpu_ok = */ true) == nullptr) {
                    bad_keep += format("%s%s", bad_keep.empty() ? "" : ", ", item.substr(0, eq).c_str());
                }
            }
            if (!bad_keep.empty()) {
                fprintf(stderr, "%s: --moe-stream-vram-free: %s - not a GPU of this model (its GPUs: %s), not used\n", __func__,
                        bad_keep.c_str(), gpu_names.empty() ? "none" : gpu_names.c_str());
            }
        }
        const auto vram_keep = [&](ggml_backend_dev_t d) -> int64_t {
            if (!keep_list || d == nullptr) {
                return 0;
            }
            const auto it = keep_named.find(ggml_backend_dev_name(d));
            return it != keep_named.end() ? it->second : std::max<int64_t>(0, params.moe_stream_vram_free);
        };
        // What the whole-layer fits leave free on a device: its --moe-stream-vram-free value, except on the VRAM pool's GPU. There the pool
        // keeps it free itself, measured once the context exists, and loads the model again with one whole layer fewer when whole layers
        // left too little (keeping it here as well turned one whole layer into pool places, with the same VRAM free at the end)
        const auto fit_keep = [&](ggml_backend_dev_t d) -> int64_t {
            if (d == nullptr || d == ms_pool_dev) {
                return 0;
            }
            return keep_list ? vram_keep(d) : std::max<int64_t>(0, params.moe_stream_vram_free);
        };
        // the compute buffers' margin per token of the physical batch (the auto-fit's, and the whole-layer check's estimate)
        static const double fit_mib_per_token = getenv("MOE_STREAM_FIT_MIB_PER_TOKEN")
            ? atof(getenv("MOE_STREAM_FIT_MIB_PER_TOKEN")) : 0.5;
        // What each GPU device that runs layers holds besides the weights and the context: the compute buffers' estimate (the per-token
        // margin x -ub + the attention masks, F16 [KV cells x -ub]), the backend's cuBLAS handle + workspace (98 MiB measured on CUDA) and
        // its op scratch pool (the engine's own size, ubatch x hidden size x 8 bytes) - the whole-layer check's terms, also used by "auto"
        const int64_t wl_ub = params.moe_stream_ubatch > 0 ? params.moe_stream_ubatch : 512;
        const bool offload_kv = (params.moe_stream_ctx_flags & 4) != 0;
        int64_t wl_compute = 0, wl_prewarm = 0;
        {
            const int64_t ub = wl_ub;
            const uint32_t n_seq_c   = std::max(1u, params.moe_stream_ctx_seq);
            const uint32_t n_ctx_c   = GGML_PAD(params.moe_stream_ctx > 0 ? params.moe_stream_ctx : hparams.n_ctx_train, 256);
            const uint32_t n_ctx_seq = (params.moe_stream_ctx_flags & 1) ? n_ctx_c : GGML_PAD(n_ctx_c/n_seq_c, 256);
            const int64_t kv_full = (int64_t) n_ctx_seq;
            const int64_t kv_swa  = hparams.swa_type != LLAMA_SWA_TYPE_NONE ? std::min<int64_t>(kv_full,
                    GGML_PAD((int64_t) hparams.n_swa*((params.moe_stream_ctx_flags & 1) ? n_seq_c : 1) + ub, 256)) : 0;
            bool any_full = false, any_swa = false;
            for (uint32_t il = 0; il < hparams.n_layer(); il++) {
                if (!hparams.is_recr(il) && hparams.has_kv(il)) {
                    (hparams.swa_type != LLAMA_SWA_TYPE_NONE && hparams.is_swa(il) && !(params.moe_stream_ctx_flags & 2) ? any_swa : any_full) = true;
                }
            }
            wl_compute = (int64_t) (fit_mib_per_token*1048576.0)*ub +
                         (any_full ? kv_full*ub*2 : 0) + (any_swa ? kv_swa*ub*2 : 0);
            wl_prewarm = ((int64_t) 98 << 20) +
                         (params.moe_stream_vram_scratch >= 0 ? params.moe_stream_vram_scratch
                                                              : std::min<int64_t>(n_ctx_c, ub)*(int64_t) hparams.n_embd*8);
        }
        const bool wl_spec = params.moe_stream_gpu_layers_device && params.moe_stream_gpu_layers_device[0];
        std::vector<ggml_backend_dev_t> whole_dev_flag(n_layer_all, nullptr);
        std::string gl_list_eff = params.moe_stream_gpu_layers_list ? params.moe_stream_gpu_layers_list : "";
        std::vector<ggml_backend_dev_t> auto_devs;   // the devices named "auto", in the order given
        std::vector<int32_t>            spec_order;  // the layers named with a device, in the order given
        if (wl_spec) {
            std::set<int32_t> bad;
            std::string bad_say;
            for (const std::string & part : split_by(params.moe_stream_gpu_layers_device, ';')) {
                const size_t eq = part.find('=');
                if (eq == std::string::npos) {
                    continue;   // (common checked the syntax; an API caller's empty part is skipped)
                }
                const std::string dname = part.substr(0, eq);
                ggml_backend_dev_t d = gpu_by_name(dname, /* igpu_ok = */ true);
                if (part.substr(eq + 1) == "auto") {
                    if (d != nullptr) {
                        auto_devs.push_back(d);
                    } else {
                        bad_say += format("%s%s (auto)", bad_say.empty() ? "" : "; ", dname.c_str());
                    }
                    continue;
                }
                std::string bl;
                for (const std::string & item : split_by(part.substr(eq + 1), ',')) {
                    int a = -1, b = -1;
                    if (sscanf(item.c_str(), "%d-%d", &a, &b) < 2) {
                        b = a;
                    }
                    for (int il = std::max(a, 0); il <= b && il < 4096; il++) {
                        if (d != nullptr && il < n_layer_all) {
                            if (std::find(spec_order.begin(), spec_order.end(), il) == spec_order.end()) {
                                spec_order.push_back(il);
                            }
                            whole_dev_flag[il] = d;
                        } else if (d == nullptr) {
                            bad.insert(il);
                            bl += format("%s%d", bl.empty() ? "" : ",", il);
                        }
                    }
                }
                if (d == nullptr) {
                    bad_say += format("%s%s (layers %s)", bad_say.empty() ? "" : "; ", dname.c_str(), bl.c_str());
                }
            }
            if (!bad.empty() || bad_say.find("(auto)") != std::string::npos) {
                std::string kept;
                for (const std::string & item : split_by(gl_list_eff, ',')) {
                    if (!item.empty() && bad.count(atoi(item.c_str())) == 0) {
                        kept += (kept.empty() ? "" : ",") + item;
                    }
                }
                gl_list_eff = kept;
                fprintf(stderr, "%s: --moe-stream-gpu-layers-device: %s - not a GPU of this model (its GPUs: %s): those layers are streamed "
                        "instead\n", __func__, bad_say.c_str(), gpu_names.empty() ? "none" : gpu_names.c_str());
            }
        }
        // --moe-stream-gpu-layers-device DEV=auto: the first device named auto takes as many of the last streamed layers as fit in its free
        // VRAM, keeping its --moe-stream-vram-free value free (a run without gaps, from the end), the next the layers before those. Each
        // layer costs its experts + its other weights (+ its KV cache / recurrent state when the context is in VRAM), each device the
        // check's fixed terms (compute buffers, cuBLAS + scratch) and the first one the output weights; the bytes come from the model
        // file, the engine's check below then runs with the exact sizes and streams what still does not fit. The engine then places the
        // layers: each whole layer runs completely on its device, every other GPU layer on the pool's GPU, the output with the last layer.
        // A list given back by a reload (one layer fewer) is kept as it is; a reload that gave up every layer (whole-layer count 0) picks none
        static const bool no_auto_mgpu = getenv("MOE_STREAM_NO_AUTO_MULTIGPU") && atoi(getenv("MOE_STREAM_NO_AUTO_MULTIGPU")) != 0;
        bool auto_gl_budget = false;
        std::vector<int32_t> user_gl_init;
        if (!wl_spec && !no_auto_mgpu && ms_pool_dev != nullptr &&
                (params.moe_stream_auto_mask & (1u << 6)) != 0 &&
                (params.moe_stream_vram_experts > 0 || params.moe_stream_vram_free > 0) &&
                (params.moe_stream_gpu_layers < 0 || (n_gpu_devs > 1 && params.moe_stream_gpu_layers != 0))) {
            auto_gl_budget = true;
            if (n_gpu_devs > 1) {
                for (size_t i = devices.size(); i-- > 0; ) {
                    const auto t = ggml_backend_dev_type(devices[i].dev);
                    if ((t == GGML_BACKEND_DEVICE_TYPE_GPU || (params.moe_stream_uma == 1 && t == GGML_BACKEND_DEVICE_TYPE_IGPU)) &&
                            devices[i].dev != ms_pool_dev) {
                        auto_devs.push_back(devices[i].dev);
                    }
                }
            }
            if (!llama_model_wl_retry && !gl_list_eff.empty()) {
                for (const std::string & item : split_by(gl_list_eff, ',')) {
                    if (!item.empty()) {
                        user_gl_init.push_back(atoi(item.c_str()));
                    }
                }
                if (n_gpu_devs > 1) {
                    gl_list_eff.clear();   // let auto_devs pack secondary GPUs first, then prepend non-overlapping pool-GPU whole layers
                }
            }
        }
        if ((!auto_devs.empty() || auto_gl_budget) && ms_pool_dev != nullptr) {
            const int il_end = n_layer_all - (int) hparams.n_layer_nextn;   // the trunk's end (the MTP block follows the last layer)
            std::vector<int64_t> exp_b(n_layer_all, 0), oth_b(n_layer_all, 0);
            int64_t out_b = 0;
            for (const auto & it : ml.weights_map) {
                int il = -1;
                sscanf(it.first.c_str(), "blk.%d.", &il);
                const bool nextn = hparams.n_layer_nextn > 0 && il >= il_end;
                if (nextn && !ml.load_mtp) {
                    continue;
                }
                const int64_t b = (int64_t) ggml_nbytes(it.second.tensor);
                if (il >= 0 && il < n_layer_all) {
                    (llama_moe_stream_is_exps(it.first.c_str()) ? exp_b : oth_b)[il] += b;
                } else if (strncmp(it.first.c_str(), "output", 6) == 0) {
                    out_b += b;
                }
            }
            const bool pick = gl_list_eff.empty() && params.moe_stream_gpu_layers != 0;
            std::string say;
            int il = il_end - 1;
            for (size_t k = 0; k < auto_devs.size(); k++) {
                ggml_backend_dev_t d = auto_devs[k];
                size_t f = 0, tot = 0;
                ggml_backend_dev_memory(d, &f, &tot);
                int64_t room = (int64_t) f - fit_keep(d) - wl_compute - wl_prewarm - (k == 0 ? out_b : 0);
                int n = 0, lo = -1, hi = -1;
                int64_t used = 0;
                for (; il >= (int) hparams.n_layer_dense_lead; il--) {
                    if (pimpl->dev_layer[il].dev == cpu_dev) {
                        break;   // (the layers below -ngl stay on the CPU)
                    }
                    if (exp_b[il] == 0 || whole_dev_flag[il] != nullptr) {
                        continue;   // (a layer without experts, or one named for a device, is passed over)
                    }
                    const int64_t cost = exp_b[il] + oth_b[il] + (offload_kv ? llama_moe_stream_layer_ctx_bytes(hparams, arch, (uint32_t) il, params) : 0);
                    if (cost > room) {
                        break;
                    }
                    room -= cost;
                    used += exp_b[il];
                    whole_dev_flag[il] = d;
                    hi = hi < 0 ? il : hi;
                    lo = il;
                    if (pick) {
                        gl_list_eff += format("%s%d", gl_list_eff.empty() ? "" : ",", il);   // (descending: a short device gives up its first layer)
                    }
                    n++;
                }
                say += format("%s%s %s (%.2f GiB of experts, %.0f MiB of %.0f free kept free)", say.empty() ? "" : "; ", ggml_backend_dev_name(d),
                        n > 0 ? format("takes layers %d-%d", lo, hi).c_str() : "takes no layer", used/1073741824.0,
                        fit_keep(d)/1048576.0, f/1048576.0);
            }
            if (auto_gl_budget && pick && user_gl_init.empty() && params.moe_stream_gpu_layers < 0) {
                size_t f_pool = 0, tot_pool = 0;
                ggml_backend_dev_memory(ms_pool_dev, &f_pool, &tot_pool);
                const int64_t pool_keep = keep_list ? vram_keep(ms_pool_dev) : std::max<int64_t>(0, params.moe_stream_vram_free);
                int64_t pool_oth = 0, pool_ctx = 0;
                const bool sec_took_last = !auto_devs.empty() && il_end > 0 && whole_dev_flag[il_end - 1] != nullptr && whole_dev_flag[il_end - 1] != ms_pool_dev;
                if (!sec_took_last && pimpl->dev_output.dev != cpu_dev) {
                    pool_oth += out_b;
                }
                std::set<int64_t> distinct_mib;
                for (int x = 0; x < n_layer_all; x++) {
                    if (pimpl->dev_layer[x].dev == cpu_dev) {
                        continue;
                    }
                    const ggml_backend_dev_t owner = x >= il_end && il_end > 0
                                                   ? (whole_dev_flag[il_end - 1] != nullptr ? whole_dev_flag[il_end - 1] : ms_pool_dev)
                                                   : (whole_dev_flag[x] != nullptr ? whole_dev_flag[x] : ms_pool_dev);
                    if (owner == ms_pool_dev) {
                        pool_oth += oth_b[x];
                        if (offload_kv) {
                            pool_ctx += llama_moe_stream_layer_ctx_bytes(hparams, arch, (uint32_t) x, params);
                        }
                    }
                    if (x < il_end && x >= (int) hparams.n_layer_dense_lead && exp_b[x] > 0 && whole_dev_flag[x] == nullptr) {
                        distinct_mib.insert((exp_b[x] + 524288) >> 20);
                    }
                }
                const int64_t reclaim_cap  = std::max<int64_t>(0, params.moe_stream_gpu_layers_lend_reclaim);
                const int64_t lend_reclaim = (params.moe_stream_vram_lend_prompt != 0 && params.moe_stream_vram_follow == 1)
                                           ? std::min<int64_t>(wl_compute, reclaim_cap) : 0;
                const int64_t room_after_fixed = (int64_t) f_pool - pool_keep - wl_compute - wl_prewarm - pool_oth - pool_ctx;
                const int64_t eff_room = room_after_fixed + lend_reclaim;
                const int64_t vx_floor = std::max<int64_t>(0, params.moe_stream_gpu_layers_floor);
                const float   cap_frac = std::min(1.0f, std::max(0.0f, params.moe_stream_gpu_layers_cap));
                const int64_t wl_cap   = std::min<int64_t>(eff_room - vx_floor, (int64_t) (eff_room * (double) cap_frac));
                int64_t pool_used = 0;
                std::vector<int32_t> pool_picked;
                if (eff_room > vx_floor && wl_cap > 0) {
                    if (distinct_mib.size() == 1) {
                        const int32_t uni_max = params.moe_stream_gpu_layers_uniform_max < 0
                                              ? (hparams.ink2_phys_block_count > 0 ? INT32_MAX : std::max<int32_t>(0, params.moe_stream_wordstart))
                                              : (params.moe_stream_gpu_layers_uniform_max == 0 ? INT32_MAX : params.moe_stream_gpu_layers_uniform_max);
                        const int64_t uni_budget = (params.moe_stream_gpu_layers_uniform_max == 0 || hparams.ink2_phys_block_count > 0) ? wl_cap : (eff_room - vx_floor);
                        std::vector<int> cand_x;
                        if (hparams.ink2_phys_block_count > 0) {
                            for (int x = (int) hparams.ink2_loop_start; x < (int) hparams.ink2_loop_end && x < il_end; x++) { cand_x.push_back(x); }
                            for (int x = (int) hparams.n_layer_dense_lead; x < (int) hparams.ink2_loop_start && x < il_end; x++) { cand_x.push_back(x); }
                            for (int x = (int) hparams.ink2_loop_end; x < (int) hparams.ink2_phys_block_count && x < il_end; x++) { cand_x.push_back(x); }
                        } else {
                            for (int x = (int) hparams.n_layer_dense_lead; x < il_end; x++) { cand_x.push_back(x); }
                        }
                        for (const int x : cand_x) {
                            if ((int) pool_picked.size() >= uni_max) {
                                break;
                            }
                            if (pimpl->dev_layer[x].dev == cpu_dev || exp_b[x] == 0 || whole_dev_flag[x] != nullptr) {
                                continue;
                            }
                            if (pool_used + exp_b[x] > uni_budget) {
                                break;
                            }
                            pool_used += exp_b[x];
                            whole_dev_flag[x] = ms_pool_dev;
                            pool_picked.push_back(x);
                        }
                    } else if (distinct_mib.size() > 1) {
                        for (int x = il_end - 1; x >= (int) hparams.n_layer_dense_lead; x--) {
                            if (pimpl->dev_layer[x].dev == cpu_dev || exp_b[x] == 0 || whole_dev_flag[x] != nullptr) {
                                continue;
                            }
                            if (pool_used + exp_b[x] > wl_cap) {
                                continue;
                            }
                            pool_used += exp_b[x];
                            whole_dev_flag[x] = ms_pool_dev;
                            pool_picked.push_back(x);
                        }
                        std::sort(pool_picked.begin(), pool_picked.end());
                    }
                }
                for (const int32_t x : pool_picked) {
                    user_gl_init.push_back(x);
                }
                if (!pool_picked.empty()) {
                    std::string plist;
                    for (const int32_t x : pool_picked) {
                        plist += format("%s%d", plist.empty() ? "" : ",", x);
                    }
                    fprintf(stderr, "%s: auto whole-layer budget on %s: takes layers %s (%.2f GiB of experts, %.2f GiB available after weights/ctx)\n",
                            __func__, ggml_backend_dev_name(ms_pool_dev), plist.c_str(), pool_used/1073741824.0, eff_room/1073741824.0);
                }
            }
            if (pick) {
                for (const int32_t x : spec_order) {   // the layers named with a device come first, in the order given
                    gl_list_eff = format("%d", x) + (gl_list_eff.empty() ? "" : ",") + gl_list_eff;
                }
                if (!user_gl_init.empty()) {
                    std::string u_pref;
                    std::set<int32_t> seen;
                    for (const int32_t x : user_gl_init) {
                        if (x >= 0 && x < n_layer_all && (whole_dev_flag[x] == nullptr || whole_dev_flag[x] == ms_pool_dev) && seen.insert(x).second) {
                            u_pref += format("%s%d", u_pref.empty() ? "" : ",", x);
                        }
                    }
                    if (!u_pref.empty()) {
                        gl_list_eff = u_pref + (gl_list_eff.empty() ? "" : ",") + gl_list_eff;
                    }
                }
            }
            // the placement: a layer of the list on its device, the other GPU layers on the pool's GPU
            std::set<int32_t> in_list;
            for (const std::string & item : split_by(gl_list_eff, ',')) {
                if (!item.empty()) {
                    in_list.insert(atoi(item.c_str()));
                }
            }
            for (int x = 0; x < n_layer_all; x++) {
                if (pimpl->dev_layer[x].dev == cpu_dev) {
                    continue;
                }
                ggml_backend_dev_t t = x >= il_end && il_end > 0 ? pimpl->dev_layer[il_end - 1].dev
                                     : in_list.count(x) && whole_dev_flag[x] != nullptr ? whole_dev_flag[x] : ms_pool_dev;
                pimpl->dev_layer[x] = { t, &pimpl->gpu_buft_list.at(t) };
            }
            if (pimpl->dev_output.dev != cpu_dev && il_end > 0) {
                pimpl->dev_output = pimpl->dev_layer[il_end - 1];
            }
            if (!auto_devs.empty()) {
                fprintf(stderr, "%s: --moe-stream-gpu-layers-device auto: %s; the other GPU layers run on %s (the pool's GPU)%s\n", __func__,
                        say.c_str(), ggml_backend_dev_name(ms_pool_dev), pick ? "" : format(", whole layers as given: %s",
                        gl_list_eff.empty() ? "none" : gl_list_eff.c_str()).c_str());
            }
        }
        // the whole-layer count the engine is given: none when the device list was given or auto_gl_budget picked (not the legacy auto-fit)
        const int32_t gpu_layers_n = ((wl_spec || (auto_gl_budget && params.moe_stream_gpu_layers < 0)) && gl_list_eff.empty())
                                   ? 0 : params.moe_stream_gpu_layers;

        // What this process will need in RAM besides the expert pools: the weights of the layers
        // that stay on the host, and the tensors outside the layers (token embeddings, output). Walked
        // from the model file, so it is a measurement of this model and this offload split, not a guess.
        // The pools are sized from MemAvailable minus this, so a model that is fully offloaded leaves it
        // near zero and a mostly-CPU one leaves tens of GiB. The expert tensors of a streamed layer are
        // skipped: a pool takes their place, and --moe-stream-gpu-layers turns them into VRAM.
        int64_t other_bytes = 0;
        int64_t np_known = 0, np_planned = 0, np_srv_need = 0, np_delta = 0;   // the non-pool bytes known exactly (weights in RAM), the plan's own part
        int64_t gpu_weight_bytes = 0;
        std::map<ggml_backend_dev_t, int64_t> gpu_weight_dev;   // the same per GPU device (the whole-layer check)
        {
            // Walk the loader's weights map, not ml.metadata - the metadata lists only
            // the FIRST shard's tensors, so a split GGUF was under-counted
            for (const auto & it : ml.weights_map) {
                const char * tn_name = it.first.c_str();
                const int64_t tn_bytes = (int64_t) ggml_nbytes(it.second.tensor);
                int il = -1;

                sscanf(tn_name, "blk.%d.", &il);

                // The expert tensors of a streamed layer are in neither bucket: a pool takes their
                // place, or the layer becomes a whole layer in VRAM, which the auto-fit divides
                // for. The MTP/nextn block is the exception: the engine never streams it (the sidecar
                // entry is skipped), so its experts are ordinary weights and have to be counted, or the
                // auto-fit spends VRAM on a block that is already there when --spec-type draft-mtp loads it.
                const bool nextn = hparams.n_layer_nextn > 0 &&
                        il >= (int) (n_layer_all - hparams.n_layer_nextn);

                // The MTP block is only loaded with --spec-type draft-mtp (bailingmoe3 gives it
                // TENSOR_SKIP otherwise), so it was counted as 1.55 GiB of GPU weights that were not there
                if (nextn && !ml.load_mtp) {
                    continue;
                }

                // The word table of an arch that marks it TENSOR_READ_LAZY (bailingmoe3.cpp) is mapped from the file, not held
                // in RAM: clean file pages the OS drops first - not counted against the pools (the same rule decides in
                // llama_moe_stream_get_plan; the arch list is the one place that must agree with the arch's create_tensor flags)
                if (il < 0 && strcmp(tn_name, "token_embd.weight") == 0 && llama_arch_marks_token_embd_lazy(arch) &&
                        ml.weights_map.count("output.weight") > 0 &&   // a tied table is not lazy (bailingmoe3.cpp makes the same test)
                        ml.lazy.add(tn_name, it.second.tensor, nullptr)) {
                    fprintf(stderr, "%s: the word table %s (%.1f MiB) is read lazily - mapped from the file, MADV_RANDOM, not counted as RAM "
                            "of ours (-lzm off or MOE_STREAM_TEST_TOKEN_TABLE_RAM=1 keeps it in RAM)\n", __func__, tn_name, (double) tn_bytes/1048576.0);
                    continue;
                }

                // Any other table the arch reads lazily (the same decision as the loader's: lazy mode, size): neither RAM nor VRAM of
                // ours. Counted, a 27 GiB n-gram table left no room for the pools (CPU) or filled the GPU's weight sum (-ngl)
                if (il < 0 && llama_arch_marks_tensor_lazy(arch, tn_name) && ml.lazy.add(tn_name, it.second.tensor, nullptr)) {
                    fprintf(stderr, "%s: %s (%.1f MiB) is read lazily - mapped from the file, only the rows a token uses are read, "
                            "not counted as memory of ours\n", __func__, tn_name, (double) tn_bytes/1048576.0);
                    continue;
                }

                if (llama_moe_stream_is_exps(tn_name) && !nextn) {
                    continue;
                }

                // A tensor outside the blocks sits on its own device - token_embd on the input
                // device, output/output_norm on the output device. The old test (dev_output AND dev_input not
                // CPU) was never true, so output.weight (0.40 GiB on Flash, on the GPU) was counted as RAM.
                bool on_gpu;
                ggml_backend_dev_t tn_dev;
                if (il >= 0) {
                    on_gpu = il < (int) pimpl->dev_layer.size() && pimpl->dev_layer[il].dev != cpu_dev;
                    tn_dev = on_gpu ? pimpl->dev_layer[il].dev : cpu_dev;
                } else if (strncmp(tn_name, "token_embd", 10) == 0) {
                    on_gpu = pimpl->dev_input.dev != cpu_dev;
                    tn_dev = pimpl->dev_input.dev;
                } else {
                    on_gpu = pimpl->dev_output.dev != cpu_dev;
                    tn_dev = pimpl->dev_output.dev;
                }

                // With --moe-stream-guess-combined 1 (and lookahead N > 1) the combined guess step stacks
                // a MoE layer's ffn_norm gain (with --moe-stream-pred-shared 1) and selection bias once for each layer that guesses
                // it (at most N), on its router's buffer type - counted with the layer's own weights, i.e. in the bucket its router
                // is counted in (an upper bound, ~1.9 MiB on Flash at N 4); with combined 0 nothing is stacked and nothing counted
                int64_t stack_bytes = 0;
                const int lookahead_n = params.moe_stream_lookahead > 1 && params.moe_stream_guess_combined == 1 ?
                        params.moe_stream_lookahead : 0;
                if (lookahead_n > 0 && il >= (int) hparams.n_layer_dense_lead && !nextn &&
                        ((params.moe_stream_pred_shared == 1 && strstr(tn_name, ".ffn_norm.weight")) || strstr(tn_name, ".exp_probs_b.bias"))) {
                    stack_bytes = tn_bytes*lookahead_n;
                }

                if (on_gpu) {
                    gpu_weight_bytes += tn_bytes + stack_bytes;
                    gpu_weight_dev[tn_dev] += tn_bytes + stack_bytes;
                    if (il < 0 && strcmp(tn_name, "output.weight") == 0 && prefill_lend_output_explicit()) {
                        other_bytes += tn_bytes;   // --moe-stream-lend-output 1 keeps one session-level RAM backup of output.weight
                    }
                } else {
                    other_bytes += tn_bytes + stack_bytes;
                }
            }

            // --moe-stream-correction: the file's tensors are allocated at model load on the router's device, so they are part of the weights that stay
            // there: counted per target layer in the bucket its router is counted in (the same rule as the stacks above). The settings the load refuses for count nothing.
            if (params.moe_stream_correction && params.moe_stream_correction[0] && params.moe_stream_pred_shared == 1 && params.moe_stream_prefetch != 0 &&
                    params.moe_stream_prefetch_lead < 2 && !(params.moe_stream_chance_model && params.moe_stream_chance_model[0])) {
                llama_moe_correction_plan_in cin;
                const llama_moe_corr_ref cref = llama_moe_correction_ref(params.moe_stream_correction, ml.file_paths[0], /* say = */ false);
                cin.path = cref.path;  cin.base = cref.base;  cin.size = cref.size;  cin.arch = arch_name();
                cin.n_embd = hparams.n_embd;  cin.n_expert = hparams.n_expert;
                cin.n_layer_all = n_layer_all;  cin.n_layer_dense = hparams.n_layer_dense_lead;  cin.n_layer_nextn = hparams.n_layer_nextn;
                cin.lookahead = params.moe_stream_lookahead;
                int64_t corr_gpu = 0, corr_ram = 0;
                for (const auto & [x, b] : llama_moe_correction_plan(cin)) {
                    const bool x_gpu = x < (int) pimpl->dev_layer.size() && pimpl->dev_layer[x].dev != cpu_dev;
                    (x_gpu ? corr_gpu : corr_ram) += b;
                    if (x_gpu) {
                        gpu_weight_dev[pimpl->dev_layer[x].dev] += b;
                    }
                }
                gpu_weight_bytes += corr_gpu;
                other_bytes      += corr_ram;
                if (corr_gpu + corr_ram > 0) {
                    fprintf(stderr, "%s: the correction file's tensors are counted in the weight plan: %.1f MiB on the GPU side, %.1f MiB in RAM (%s)\n",
                            __func__, (double) corr_gpu/1048576.0, (double) corr_ram/1048576.0, params.moe_stream_correction);
                }
            }

            if (params.moe_stream_uma == 1) {
                for (const auto & d : devices) {
                    if (llama_moe_stream_dev_is_uma(d.dev)) {
                        const int64_t uma_w = gpu_weight_dev[d.dev];
                        if (uma_w > 0) {
                            other_bytes += uma_w;
                            fprintf(stderr, "%s: unified-memory mode (--moe-stream-uma 1): %.2f GiB of non-expert weights on %s counted in the shared RAM budget\n",
                                    __func__, (double) uma_w/1024.0/1024.0/1024.0, ggml_backend_dev_name(d.dev));
                        }
                        break;
                    }
                }
            }

            LLAMA_LOG_DEBUG("%s: %zu bytes of this model's weights stay in RAM next to the expert pools\n",
                    __func__, (size_t) other_bytes);

            // The pools are sized before the loader reads any weight, so the rest of the process
            // cannot be measured yet - the KV cache, the compute buffers and the staging buffers are
            // not allocated either. Add a reserve for them (measured: 0.8 GiB on Flash at 4k ctx) and
            // say so, because this number decides how many slots the pools get.
            np_known = other_bytes;
            other_bytes += llama_moe_stream_other_reserve();

            // A server's snapshots and its state-file restore buffer, inside the reserve's headroom
            int64_t srv_bytes = 0;
            if (params.moe_stream_srv) {
                srv_bytes = llama_moe_stream_srv_extra(llama_moe_stream_srv_compute(hparams, arch, *params.moe_stream_srv),
                        *params.moe_stream_srv, __func__);
                other_bytes += srv_bytes;
                np_srv_need = llama_moe_stream_srv_compute(hparams, arch, *params.moe_stream_srv).need;
            }
            if (params.moe_stream_nonpool) {   // from the record, or the fixed reserve without one
                np_planned = llama_moe_stream_np_bytes(*params.moe_stream_nonpool, llama_moe_stream_sidecar_arg(params.moe_stream, ml.file_paths[0]).c_str(),
                        np_srv_need, srv_bytes, __func__);
                np_delta   = np_planned - (llama_moe_stream_other_reserve() + srv_bytes);
                other_bytes += np_delta;
            }

            // The recorder keeps up to --moe-stream-record-words decode words in RAM, so
            // the pools must leave room for them. Counted at the cap and for every MoE layer (an upper bound).
            int64_t record_bytes = 0;
            if (params.moe_stream_record && params.moe_stream_record[0]) {
                const int64_t n_moe = (int64_t) hparams.n_layer() - (int64_t) hparams.n_layer_dense_lead - (int64_t) hparams.n_layer_nextn;

                record_bytes = llama_moe_stream_record_bytes(n_moe > 0 ? n_moe : 0, (int64_t) hparams.n_layer_all,
                        (int64_t) hparams.n_expert, (int64_t) hparams.n_expert_used(hparams.n_layer_dense_lead),
                        params.moe_stream_record_words > 0 ? params.moe_stream_record_words : MOE_STREAM_RECORD_DEFAULT_WORDS,
                        params.moe_stream_io_threads > 0 ? params.moe_stream_io_threads : 8,
                        /* helper = */ params.moe_stream_needed_early == 1,
                        params.moe_stream_guess_top,
                        params.moe_stream_lookahead > 0 ? params.moe_stream_lookahead : 1,
                        params.moe_stream_guess_pace > 0 ? params.moe_stream_guess_pace : 1);
                other_bytes += record_bytes;
                fprintf(stderr, "%s: --moe-stream-record keeps up to %d decode words in RAM = %.2f GiB, counted in the "
                        "non-pool memory\n", __func__,
                        params.moe_stream_record_words > 0 ? (int) params.moe_stream_record_words : (int) MOE_STREAM_RECORD_DEFAULT_WORDS,
                        (double) record_bytes/1024.0/1024.0/1024.0);
            }
            // The guess helper's slot buffer (N columns of the decode guess's width + a counter per layer):
            // only with --moe-stream-needed-early 1, at its real size, as the plan counts it (the engine default adds nothing)
            const int64_t n_moe_layer_slots = hparams.ink2_phys_block_count > 0 ? (int64_t) hparams.ink2_phys_block_count : (int64_t) hparams.n_layer_all;
            int64_t ne_bytes = 0;
            if (params.moe_stream_needed_early == 1) {
                const int64_t ne_la = params.moe_stream_lookahead > 1 ? params.moe_stream_lookahead : 1;
                const int64_t ne_w  = params.moe_stream_guess_top > 0 ? params.moe_stream_guess_top
                                                                      : (params.moe_stream_prefetch_m > 0 ? params.moe_stream_prefetch_m : (int32_t) hparams.n_expert_used());   // 0 = the model's top-k
                ne_bytes = llama_moe_stream_ne_bytes(n_moe_layer_slots, ne_la, ne_w)
                         // + the GPU producer's pinned buffer (guess slots, call cell, router slots), as the plan counts it
                         + llama_moe_stream_ne2_bytes(n_moe_layer_slots, ne_la, ne_w, (int64_t) hparams.n_embd,
                                                      (int64_t) hparams.n_expert_used(),
                                                      // + the prompt split's shared router slot (gpu-waves - 1 rows)
                                                      params.moe_stream_prompt_split == 1 && params.moe_stream_gpu_waves > 1
                                                          ? (int64_t) params.moe_stream_gpu_waves - 1 : 0);
                other_bytes += ne_bytes;
            }
            // The wave plan's / slot check's shared tables, at the training context (the plan counts the same)
            const int64_t wp_bytes = llama_moe_stream_wp_bytes((int64_t) hparams.n_ctx_train, (int64_t) hparams.n_expert_used(),
                                                               (int64_t) hparams.n_expert);
            other_bytes += wp_bytes;
            // stderr, not the library logger: the tools drop the library's INFO lines (the same reason
            // "moe profile" is printed with fprintf), and this is the number that decides the pool size
            fprintf(stderr, "%s: non-pool memory: %.2f GiB of model weights stay in RAM, %.2f GiB reserved for "
                    "the KV cache, the compute buffers and the CUDA host allocations (the measured worst "
                    "case of 36 runs + 0.25 GiB)%s, %.2f MiB for the wave plan's tables (%u tokens), %.2f GiB in total. The pools get "
                    "MemAvailable minus that minus --moe-stream-ram-free\n",
                    __func__, (other_bytes - np_delta - llama_moe_stream_other_reserve() - record_bytes - srv_bytes - ne_bytes - wp_bytes)/1024.0/1024.0/1024.0,
                    (double) (llama_moe_stream_other_reserve() + np_delta)/1024.0/1024.0/1024.0,
                    // The server's term beyond the reserve is named, not counted as weights
                    srv_bytes > 0 ? format(", %.2f GiB more for the server's snapshots / restore buffer", srv_bytes/1024.0/1024.0/1024.0).c_str() : "",
                    wp_bytes/1024.0/1024.0, hparams.n_ctx_train,
                    (double) other_bytes/1024.0/1024.0/1024.0);
            // The same to the byte, with the guess helper's slot buffer named (0 unless --moe-stream-needed-early 1)
            fprintf(stderr, "%s: non-pool memory: %lld bytes (of them the guess helper's slot buffers %lld: the host producer's + the GPU producer's pinned one)\n", __func__,
                    (long long) other_bytes, (long long) ne_bytes);
        }

        const int n_moe_layers = hparams.ink2_phys_block_count > 0 ? (int) hparams.ink2_phys_block_count : n_layer_all;

        llama_moe_stream::model_info mi;
        mi.arch          = llm_arch_name(arch);
        mi.n_layer_all   = (uint32_t) n_moe_layers;
        mi.n_layer_nextn = hparams.n_layer_nextn;
        mi.n_layer_dense = hparams.n_layer_dense_lead;
        mi.n_expert      = hparams.n_expert;
        mi.n_expert_used = hparams.n_expert_used();
        mi.n_embd        = (uint32_t) hparams.n_embd;   // the router input's width
        mi.n_ctx_train   = hparams.n_ctx_train;         // the wave plan's tables

        // What whole MoE layers can have of VRAM. Free on the device, minus the weights that are
        // already going there, minus a reserve for the KV cache, the compute buffers and the driver.
        // The context is not built yet, so the KV cache and the compute buffers are not known here;
        // --moe-stream-vram-reserve covers them and defaults to what they measured at 4k ctx.
        int64_t vram_reserve = 0;

        // Only when the auto-fit decides - a --moe-stream-gpu-layers-list places the layers itself, and
        // this line then printed a plan nobody followed (the placement is printed after the context is built)
        const bool gpu_list = !gl_list_eff.empty();
        if (gpu_layers_n < 0 && !gpu_list && !devices.empty()) {
            size_t vram_free = 0;
            size_t vram_total = 0;
            ggml_backend_dev_memory(devices[0].dev, &vram_free, &vram_total);

            // --moe-stream-vram-reserve is a constant, but what the reserve has to cover
            // is not: the device compute buffer grows with the ubatch (measured 322.0 MiB at -ub 512
            // against 1306.3 MiB at -ub 2048), and the whole-layer fit runs before the context exists.
            // A constant margin therefore over-fills VRAM at a large -ub and cublasCreate fails there
            // (on the IQ4_XS file). So the margin grows with the physical batch: MOE_STREAM_FIT_MIB_PER_TOKEN
            // MiB per token on top of the user's reserve, 0 restores the constant.
            const int64_t ubatch_margin = (int64_t) (fit_mib_per_token*1024*1024) *
                (int64_t) (params.moe_stream_ubatch > 0 ? params.moe_stream_ubatch : 512);
            const int64_t vram_reserve_total = params.moe_stream_vram_reserve + ubatch_margin;

            // a --moe-stream-vram-free list: the device's own value stays free too (0 without a list; 0 on the VRAM pool's GPU: fit_keep)
            const int64_t keep0 = fit_keep(devices[0].dev);
            vram_reserve = (int64_t) vram_free - gpu_weight_bytes - vram_reserve_total - keep0;

            fprintf(stderr, "%s: VRAM for whole MoE layers: %.2f GiB free on %s, %.2f GiB of weights go there, "
                    "%.2f GiB reserved for the KV cache, the compute buffers and the driver "
                    "(%.2f GiB of it %.2f MiB x -ub %d), %s%.2f GiB left\n",
                    __func__, (double) vram_free/1024.0/1024.0/1024.0,
                    ggml_backend_dev_name(devices[0].dev),
                    (double) gpu_weight_bytes/1024.0/1024.0/1024.0,
                    (double) vram_reserve_total/1024.0/1024.0/1024.0,
                    (double) ubatch_margin/1024.0/1024.0/1024.0, fit_mib_per_token,   // this argument was missing
                    (int) (params.moe_stream_ubatch > 0 ? params.moe_stream_ubatch : 512),
                    keep0 > 0 ? format("%.2f GiB kept free (--moe-stream-vram-free), ", keep0/1024.0/1024.0/1024.0).c_str() : "",
                    (double) vram_reserve/1024.0/1024.0/1024.0);
        }

        // An explicit list or count: what each GPU device holds besides the whole layers, for the engine's check before the weights
        // load (the layers that do not fit stay streamed). Charged to each GPU device that holds layers: the other weights there, the
        // KV cache and recurrent state of its layers, the compute buffers' estimate (the per-token margin above x -ub + the attention
        // masks, F16 [KV cells x -ub]), the backend's cuBLAS handle + workspace (98 MiB measured on CUDA) and its op scratch pool (the
        // engine's own size, ubatch x hidden size x 8 bytes). The VRAM pool's minimum and the growth go to the first GPU device (the
        // pool's) in the engine, which knows the streamed layers. A device that reports no memory is not checked
        std::vector<llama_moe_stream_p1::wl_dev> wl_devs;
        std::vector<int32_t>                     wl_dev_of;
        // (the auto-fit's or auto-gl-budget's picks are checked per device before the weights load)
        if ((gpu_list || gpu_layers_n > 0 || (gpu_layers_n < 0 && n_gpu_devs > 1) || auto_gl_budget) && !devices.empty()) {
            wl_dev_of.assign(n_moe_layers, -1);
            const bool pool_on = params.moe_stream_vram_experts > 0 || params.moe_stream_vram_free > 0;
            for (const auto & dv : devices) {
                const auto t = ggml_backend_dev_type(dv.dev);
                if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    continue;
                }
                size_t f = 0, tot = 0;
                ggml_backend_dev_memory(dv.dev, &f, &tot);
                if (f == 0 && tot == 0) {
                    continue;
                }
                llama_moe_stream_p1::wl_dev w;
                w.name    = ggml_backend_dev_name(dv.dev);
                w.free    = (int64_t) f;
                w.weights = gpu_weight_dev.count(dv.dev) ? gpu_weight_dev.at(dv.dev) : 0;
                w.pool    = dv.dev == ms_pool_dev;   // (the first GPU unless --moe-stream-vram-device names another)
                // a device that runs a layer, holds a whole layer's experts (--moe-stream-gpu-layers-device: maybe not its layer's device)
                // or holds the VRAM pool (its device pass and the prompt matmuls) gets the compute and cuBLAS terms
                bool has_layer = w.pool && pool_on;
                for (int il = 0; il < n_layer_all; il++) {
                    const bool here = il < (int) pimpl->dev_layer.size() && pimpl->dev_layer[il].dev == dv.dev;
                    if (here) {
                        has_layer = true;
                        if (offload_kv) {
                            w.ctx += llama_moe_stream_layer_ctx_bytes(hparams, arch, (uint32_t) il, params);
                        }
                    }
                    if (il < n_moe_layers && (whole_dev_flag[il] != nullptr ? whole_dev_flag[il] == dv.dev : here)) {
                        wl_dev_of[il] = (int32_t) wl_devs.size();
                        has_layer = true;
                    }
                }
                if (has_layer) {
                    w.compute = wl_compute;
                    w.prewarm = wl_prewarm;
                }
                w.keep = fit_keep(dv.dev);
                wl_devs.push_back(w);
            }
        }

        // A split GGUF is several files (llama.cpp loads them all); the sidecar recorded the sum
        size_t model_bytes = 0;
        for (const auto & fp : ml.file_paths) {
            llama_file f(fp.c_str(), "rb");
            model_bytes += f.size();
        }

        llama_moe_stream_p1 p1;
        p1.io_wake_chunk       = params.moe_stream_io_wake == 1;
        p1.promote             = params.moe_stream_promote != 0;
        p1.guess_cap           = params.moe_stream_guess_cap;
        p1.cap_refill          = params.moe_stream_guess_cap_refill == 1;
        p1.evict_halflife      = params.moe_stream_evict_halflife > 0.0f ? params.moe_stream_evict_halflife : 64.0f;
        p1.evict_credit        = params.moe_stream_evict_credit;
        p1.prompt_end          = params.moe_stream_prompt_end == 1;
        p1.prompt_end_halflife = params.moe_stream_prompt_end_halflife > 0.0f ? params.moe_stream_prompt_end_halflife : 128.0f;
        p1.prompt_values       = params.moe_stream_prompt_values == 1;
        p1.chance_model        = params.moe_stream_chance_model ? params.moe_stream_chance_model : "";
        p1.read_threshold      = params.moe_stream_read_threshold > 0.0f ? params.moe_stream_read_threshold : 0.3f;
        p1.cpu_skip_zero       = params.moe_stream_cpu_skip_zero == 1;
        p1.cpu_x4              = params.moe_stream_cpu_x4;               // 0, 1 or 2
        p1.cpu_x4_pf           = params.moe_stream_cpu_x4_pf;
        p1.cpu_fuse            = params.moe_stream_cpu_fuse == 1;
        p1.cpu_dispatch        = params.moe_stream_cpu_dispatch == 1;
        p1.cpu_pieces          = params.moe_stream_cpu_pieces > 1 ? params.moe_stream_cpu_pieces : 1;
        p1.cpu_conv1           = params.moe_stream_cpu_conv1 == 1;
        p1.cpu_wsum            = params.moe_stream_cpu_wsum == 1;
        p1.cpu_down_scale      = params.moe_stream_cpu_down_scale == 1;
        p1.cpu_topk            = params.moe_stream_cpu_topk == 1;
        p1.cpu_q8_tiled        = params.moe_stream_cpu_q8_tiled;
        p1.cpu_fa              = params.moe_stream_cpu_fa;               // (0 / 1; 2 falls back to 0 in llama_moe_stream)
        p1.cpu_fuse_small      = params.moe_stream_cpu_fuse_small & (1 | 2 | 64);
        p1.guess_skip_full     = params.moe_stream_guess_skip_full == 1;
        p1.guess_skip_full_gpu = params.moe_stream_guess_skip_full_gpu == 1;
        p1.cpu_mmid_merge      = params.moe_stream_cpu_mmid_merge == 1;
        p1.cpu_iqk             = params.moe_stream_cpu_iqk;
        if (p1.cpu_wsum) {
            // A layer's MoE tail runs on the CPU only if the layer AND its consumer are CPU layers: the scheduler
            // pulls the weight-free tail of a CPU layer onto the GPU when the next layer is there (tiny -ngl 12: layer 12's
            // ffn_moe_weighted ran on CUDA0), and a CPU-only op there would move that sum back to the CPU
            p1.wsum_cpu_tail.assign(pimpl->dev_layer.size(), 0);
            for (size_t il = 0; il < pimpl->dev_layer.size(); il++) {
                const ggml_backend_dev_t next = il + 1 < hparams.n_layer() ? pimpl->dev_layer[il + 1].dev : pimpl->dev_output.dev;
                p1.wsum_cpu_tail[il] = pimpl->dev_layer[il].dev == cpu_dev && next == cpu_dev ? 1 : 0;
            }
        }
        p1.cpu_state_inplace   = params.moe_stream_cpu_state_inplace == 1;
        p1.cpu_small_ops       = params.moe_stream_cpu_small_ops == 1;
        if (p1.cpu_state_inplace) {
            // The in-place delta step is a CPU-only op, so only layers on the CPU device (their recurrent cache
            // is in host memory there) build it; a GPU layer keeps GGML_OP_GATED_DELTA_NET and its copies
            p1.state_cpu_layer.assign(pimpl->dev_layer.size(), 0);
            std::string list;
            int n_cpu = 0, n_recr = 0;
            for (size_t il = 0; il < pimpl->dev_layer.size() && il < hparams.n_layer(); il++) {
                if (!hparams.is_recr(il)) {
                    continue;
                }
                n_recr++;
                if (pimpl->dev_layer[il].dev == cpu_dev) {
                    p1.state_cpu_layer[il] = 1;
                    n_cpu++;
                    list += (list.empty() ? "" : ",") + std::to_string(il);
                }
            }
            fprintf(stderr, "%s: --moe-stream-cpu-state-inplace: %d of %d recurrent layers on the CPU device (%s): their delta "
                    "step reads and writes the state in place (one sequence per batch, no snapshot slots; otherwise as before)\n",
                    __func__, n_cpu, n_recr, list.empty() ? "none" : list.c_str());
        }
        p1.fused_route         = params.moe_stream_fused_route == 1;
        p1.slot_check          = params.moe_stream_slot_check != 0;
        p1.wave_plan           = params.moe_stream_wave_plan == 1;
        p1.wave_victim         = params.moe_stream_wave_victim > 0 || (params.moe_stream_wave_victim < 0 && params.moe_stream_prefill_fast == 1);
        p1.conv_silu           = params.moe_stream_conv_silu == 1;
        p1.hc_share            = params.moe_stream_hc_share == 1;
        p1.rs_window           = params.moe_stream_rs_window == 1;
        p1.ple_late            = params.moe_stream_ple_late == 1;
        p1.hc_silu             = params.moe_stream_hc_silu == 1;
        p1.hc_q8               = params.moe_stream_hc_q8 == 1;
        p1.rs_inplace          = params.moe_stream_rs_inplace == 1;
        p1.gpu_share_q8        = params.moe_stream_gpu_share_q8 == 1;
        p1.vram_experts        = params.moe_stream_vram_experts > 0 ? (uint32_t) params.moe_stream_vram_experts : 0;
        p1.vram_experts_file   = params.moe_stream_vram_experts_file ? params.moe_stream_vram_experts_file : "";
        p1.vram_experts_auto   = (params.moe_stream_auto_mask & (1u << 3)) != 0;
        p1.vram_experts_margin = params.moe_stream_vram_experts_margin > 0 ? params.moe_stream_vram_experts_margin
                                                                           : (int64_t) 512*1024*1024;
        p1.vram_free           = params.moe_stream_vram_free > 0 ? params.moe_stream_vram_free : 0;
        if (keep_list && ms_pool_dev != nullptr) {
            p1.vram_free = std::max<int64_t>(1, vram_keep(ms_pool_dev));   // the pool's GPU's own value (a list turns the pool on: 0 = off)
            fprintf(stderr, "%s: --moe-stream-vram-free per GPU: the VRAM pool on %s keeps %.0f MiB free\n", __func__,
                    ggml_backend_dev_name(ms_pool_dev), vram_keep(ms_pool_dev)/1048576.0);
        }
        p1.vram_scratch        = params.moe_stream_vram_scratch;   // -1 = derived
        p1.vram_growth         = params.moe_stream_vram_growth > 0 ? params.moe_stream_vram_growth : 0;  // 0 = derived
        {
            const llama_moe_corr_ref cref = llama_moe_correction_ref(params.moe_stream_correction, ml.file_paths[0],
                    /* say = */ (params.moe_stream_auto_mask & (1u << 2)) == 0);
            p1.correction_file     = cref.path;
            p1.correction_base     = cref.base;
            p1.correction_size     = cref.size;
        }
        p1.idle_load           = params.moe_stream_idle_load >= 1;
        p1.fill_start          = params.moe_stream_idle_load == 2;
        p1.usage_file          = params.moe_stream_usage_file ? params.moe_stream_usage_file : "";
        p1.idle_pause_ms       = params.moe_stream_idle_pause_ms > 0 ? params.moe_stream_idle_pause_ms : 500;
        p1.read_ahead          = params.moe_stream_read_ahead == 1;
        p1.read_ahead_min      = params.moe_stream_read_ahead_min > 0 ? params.moe_stream_read_ahead_min : 4096;
        p1.window              = params.moe_stream_window == 1;
        p1.window_keep         = params.moe_stream_window_keep > 0.0f ? std::min(1.0f, params.moe_stream_window_keep) : 1.0f;
        const bool is_looped_moe = hparams.ink2_loop_passes > 1;
        p1.prompt_defer        = params.moe_stream_prompt_defer == 1;
        p1.prompt_split        = !is_looped_moe && params.moe_stream_prompt_split == 1;
        p1.prompt_split_places = params.moe_stream_prompt_split_places >= 2 ? params.moe_stream_prompt_split_places : 128;
        p1.tail_wave_w         = params.moe_stream_tail_wave_w > 0 ? params.moe_stream_tail_wave_w : 0;
        const bool pf_fast     = params.moe_stream_prefill_fast == 1;
        p1.gpu_wave_skip_zero  = pf_fast || params.moe_stream_gpu_wave_skip_zero == 1;
        p1.gpu_skip_zero       = pf_fast || params.moe_stream_gpu_skip_zero == 1;
        p1.prefill_fast        = pf_fast;
        p1.mmq_moe_opt         = pf_fast || params.moe_stream_mmq_moe_opt   == 1;
        p1.gpu_moe_fused       = pf_fast || params.moe_stream_gpu_moe_fused == 1;
        p1.pinned_ring         = params.moe_stream_pinned_ring > 0 ? params.moe_stream_pinned_ring : 0;
        p1.lm_bf16             = pf_fast || params.moe_stream_lm_bf16      == 1;
        p1.lend_output         = params.moe_stream_lend_output == 1;
        p1.qsa_fused           = pf_fast || params.moe_stream_qsa_fused    == 1;
        p1.hc_fused            = pf_fast || params.moe_stream_hc_fused     == 1;
        p1.ple_fast            = pf_fast || params.moe_stream_ple_fast     == 1;
        p1.plan                = params.moe_stream_plan == 1;
        p1.plan_bind_early     = params.moe_stream_plan_bind_early == 1; // test only
        p1.drop_stale          = params.moe_stream_drop_stale == 1;
        p1.read_threshold_set  = params.moe_stream_read_threshold > 0.0f; // the engine's plan-1 guard
        p1.needed_early        = !is_looped_moe && params.moe_stream_needed_early == 1;
        p1.behind_backlog      = params.moe_stream_behind_backlog;
        p1.behind_rank         = params.moe_stream_behind_rank;
        p1.guess_inflight      = params.moe_stream_guess_inflight;
        p1.guess_top           = params.moe_stream_guess_top;
        p1.lookahead           = params.moe_stream_lookahead > 0 ? params.moe_stream_lookahead : 1;
        p1.guess_pace          = params.moe_stream_guess_pace > 0 ? params.moe_stream_guess_pace : 1;
        p1.plan_old_order      = params.moe_stream_plan_old_order == 1;
        p1.plan_extra_near     = params.moe_stream_plan_extra_near == 1;
        p1.guess_combined      = params.moe_stream_guess_combined == 1;
        p1.protect_named       = params.moe_stream_protect_named == 1;
        p1.wordstart           = is_looped_moe ? 0 : params.moe_stream_wordstart;
        p1.proj_ready          = params.moe_stream_proj_ready == 1;
        p1.vram_follow          = params.moe_stream_vram_follow == 1;
        p1.vram_follow_tick     = (uint32_t) params.moe_stream_vram_follow_tick;
        p1.vram_follow_swaps    = (uint32_t) params.moe_stream_vram_follow_swaps;
        p1.vram_follow_recent   = (uint32_t) params.moe_stream_vram_follow_recent;
        p1.vram_follow_longterm = (uint32_t) params.moe_stream_vram_follow_longterm;
        p1.vram_follow_ratio    = params.moe_stream_vram_follow_ratio;
        p1.vram_follow_margin   = (uint32_t) params.moe_stream_vram_follow_margin;
        p1.vram_follow_fixed    = params.moe_stream_vram_follow_fixed == 1;
        p1.vram_follow_pinned   = params.moe_stream_vram_follow_pinned == 1;
        p1.vram_parallel        = !is_looped_moe && params.moe_stream_vram_parallel >= 1;
        p1.vram_parallel_cross  = !is_looped_moe && params.moe_stream_vram_parallel == 2;
        p1.vram_copyback        = params.moe_stream_vram_copyback == 1;
        p1.async_h2d            = params.moe_stream_async_h2d == 1;
        p1.vram_chunk           = std::max<int64_t>(0, params.moe_stream_vram_chunk);
        p1.uma                  = params.moe_stream_uma == 1;
        if (p1.uma) {
            for (const auto & d : devices) {
                if (llama_moe_stream_dev_is_uma(d.dev)) {
                    p1.uma_dev = d.dev;
                    break;
                }
            }
        }
        {
            const bool has_rocm = ggml_backend_reg_by_name("ROCm") != nullptr;
            p1.hip_mmvq = params.moe_stream_hip_mmvq == 1 || (has_rocm && (params.moe_stream_auto_mask & (1u << 4)) != 0);
            p1.hip_sync = params.moe_stream_hip_sync > 0 ? params.moe_stream_hip_sync
                        : (has_rocm && (params.moe_stream_auto_mask & (1u << 5)) != 0) ? 1 : 0;
        }
        p1.elastic              = params.moe_stream_elastic == 1;
        // one leave-free number: --moe-stream-ram-margin, else --moe-stream-ram-free, else the OS default (el_parse_opts) - the start's
        // fill, elastic RAM's floor and the prompt guard all keep the same amount free
        const int64_t keep_free = params.moe_stream_ram_margin > 0 ? params.moe_stream_ram_margin
                                : params.moe_stream_ram_free > 0   ? params.moe_stream_ram_free : 0;
        p1.ram_margin           = keep_free;
        p1.ram_margin_set       = keep_free > 0;
        p1.elastic_opts         = params.moe_stream_elastic_opts ? params.moe_stream_elastic_opts : "";
        p1.shares               = params.moe_stream_shares == 1;
        p1.share_file           = params.moe_stream_share_file ? params.moe_stream_share_file : "";
        p1.share_opts           = params.moe_stream_share_opts ? params.moe_stream_share_opts : "";
        p1.layer_major          = !is_looped_moe && params.moe_stream_layer_major == 1;
        p1.layer_major_batch    = params.moe_stream_layer_major_batch > 0 ? (uint32_t) params.moe_stream_layer_major_batch : 0;
        p1.vram_end_fill        = std::max(0, std::min(2, params.moe_stream_vram_end_fill));
        p1.copy_ahead           = is_looped_moe ? 0 : params.moe_stream_copy_ahead;   // copy-ahead for GPU prompts (0 / 1 / 2)
        p1.vram_lend_prompt     = params.moe_stream_vram_lend_prompt != 0;
        p1.vram_lend_short      = std::max(1, params.moe_stream_vram_lend_short);
        if ((params.moe_stream_auto_mask & 1u) != 0 && ms_pool_dev == nullptr) {
            p1.vram_free = 0;
            p1.vram_experts = 0;
            p1.vram_experts_file.clear();
            p1.vram_follow = false;
            p1.vram_parallel = false;
            p1.vram_parallel_cross = false;
            p1.vram_copyback = false;
            p1.vram_end_fill = 0;
            p1.copy_ahead = 0;
            p1.vram_lend_prompt = false;
            p1.prompt_split = false;
        }
        if ((params.moe_stream_auto_mask & 2u) != 0 && ms_pool_dev != nullptr && p1.copy_ahead == 1) {
            size_t f_p = 0, tot_p = 0;
            ggml_backend_dev_memory(ms_pool_dev, &f_p, &tot_p);
            const int64_t one_below = std::max<int64_t>(0, params.moe_stream_copy_ahead_one_below);
            if (one_below > 0 && tot_p > 0 && (int64_t) tot_p < one_below) {
                p1.copy_ahead = 2;
            }
        }
        p1.timing               = params.moe_stream_timing == 1;
        p1.out_check            = std::max(0, params.moe_stream_out_check);
        p1.nonpool              = params.moe_stream_nonpool != nullptr;
        p1.wl_devs              = std::move(wl_devs);
        p1.wl_dev_of            = std::move(wl_dev_of);
        p1.wl_ubatch            = params.moe_stream_ubatch > 0 ? params.moe_stream_ubatch : 512;
        p1.wl_by_device         = wl_spec || !auto_devs.empty();
        uint32_t eff_gpu_waves  = (!is_looped_moe && params.moe_stream_gpu_waves > 0) ? (uint32_t) params.moe_stream_gpu_waves : 0;
        if (is_looped_moe && params.moe_stream_gpu_waves > 0 && ms_pool_dev != nullptr &&
                p1.vram_free > 0 && p1.vram_follow && !params.moe_stream_cpu_repack && !p1.elastic && !p1.shares) {
            eff_gpu_waves = ((params.moe_stream_auto_mask & (1u << 6)) != 0 && params.moe_stream_gpu_waves >= 64)
                          ? 64u : (uint32_t) params.moe_stream_gpu_waves;
            if (p1.tail_wave_w <= 0) {
                p1.tail_wave_w = 64;
            }
            if (p1.pinned_ring <= 0 && pf_fast && !params.moe_stream_pin) {
                p1.pinned_ring = (int64_t) 256 << 20;
            }
        }
        pimpl->moe_stream = std::make_unique<llama_moe_stream>(
                params.moe_stream,
                params.moe_stream_log ? params.moe_stream_log : "",
                ml.file_paths[0],
                mi,
                model_bytes,
                params.moe_stream_slots > 0 ? (uint32_t) params.moe_stream_slots : 0,
                params.moe_stream_io_threads > 0 ? (uint32_t) params.moe_stream_io_threads : 8,
                params.moe_stream_prefetch != 0,
                params.moe_stream_prefetch_m > 0 ? (uint32_t) params.moe_stream_prefetch_m : hparams.n_expert_used(),   // 0 = the model's top-k (was 8)
                params.moe_stream_max_waves > 0 ? (uint32_t) params.moe_stream_max_waves : 0,
                params.moe_stream_ram,
                params.moe_stream_oracle ? params.moe_stream_oracle : "",
                params.moe_stream_oracle_lead >= 0 ? (uint32_t) params.moe_stream_oracle_lead : 1,
                params.moe_stream_chunk > 0 ? (uint32_t) params.moe_stream_chunk : 0,
                params.moe_stream_chunk_prefetch > 0 ? (uint32_t) params.moe_stream_chunk_prefetch : 0,
                eff_gpu_waves,
                gpu_layers_n > 0 ? (uint32_t) gpu_layers_n : 0,
                gl_list_eff,
                /* gpu_layers_auto = */ gpu_layers_n < 0,
                vram_reserve,
                params.moe_stream_gpu_layers_mode ? params.moe_stream_gpu_layers_mode : "hot",
                llama_moe_stream_start_ram_free(params.moe_stream_elastic, params.moe_stream_slots, params.moe_stream_ram_free,
                        params.moe_stream_ram_margin, params.moe_stream_elastic_opts),
                other_bytes,
                params.moe_stream_verify,
                params.moe_stream_cpu_repack != 0,
                params.moe_stream_hugepages != 0,
                params.moe_stream_pin != 0,
                params.moe_stream_policy ? params.moe_stream_policy : "lru",
                params.moe_stream_pred_shared != 0,
                params.moe_stream_pred_recall != 0,
                params.moe_stream_prefetch_lead > 0 ? (uint32_t) params.moe_stream_prefetch_lead : 1,
                params.moe_stream_prefetch_m2 > 0 ? (uint32_t) params.moe_stream_prefetch_m2 : 0,
                params.moe_stream_demand_last != 0,
                params.moe_stream_probation != 0,
                params.moe_stream_prefetch_inflight > 0 ? (int32_t) params.moe_stream_prefetch_inflight : 0,
                params.moe_stream_early_order != 0,                        // force the guess first
                params.moe_stream_overlap != 0,
                params.moe_stream_pred_scores,
                params.moe_stream_pred_input,
                params.moe_stream_pred_input_every,
                params.moe_stream_record ? params.moe_stream_record : "",
                params.moe_stream_record_words,
                p1);
        {   // the GPUs of the pool and of the whole layers' experts (set before the weights are created: create_tensor reads whole_dev)
            auto & ms = *pimpl->moe_stream;
            if (hparams.ink2_loop_end > hparams.ink2_loop_start && hparams.ink2_loop_passes > 1) {
                ms.set_loop_range((int32_t) hparams.ink2_loop_start, (int32_t) hparams.ink2_loop_end);
            }
            ms.pool_dev     = ms_pool_dev;
            ms.pool_dev_set = ms_pool_dev_set;
            ms.whole_dev.assign(n_moe_layers, nullptr);
            for (const int32_t il : ms.gpu_layers) {
                if (il >= 0 && il < n_moe_layers) {
                    ms.whole_dev[il] = whole_dev_flag[il] != nullptr ? whole_dev_flag[il] : pimpl->dev_layer[il].dev;
                }
            }
            if (auto_gl_budget) {
                llama_model_auto_gl_list.clear();
                for (const int32_t il : ms.gpu_layers) {
                    llama_model_auto_gl_list += format("%s%d", llama_model_auto_gl_list.empty() ? "" : ",", il);
                }
            }
            // A named whole layer that did not fit on its GPU runs completely on the pool's GPU (its experts are streamed there):
            // left on its card, its attention would sit on one GPU and its experts on another, a hand-over there and back in every
            // layer, and that card's prompt buffers have no room (it is full of whole layers)
            if (!ms.wl_dropped.empty() && ms_pool_dev != nullptr) {
                int on_pool = -1;   // a layer already on the pool's GPU: its device and buffer types are copied
                for (size_t i = 0; i < pimpl->dev_layer.size(); i++) {
                    if (pimpl->dev_layer[i].dev == ms_pool_dev) { on_pool = (int) i; break; }
                }
                std::string moved;
                for (const int32_t il : ms.wl_dropped) {
                    if (on_pool >= 0 && il >= 0 && il < n_layer_all && pimpl->dev_layer[il].dev != ms_pool_dev) {
                        pimpl->dev_layer[il] = pimpl->dev_layer[on_pool];
                        moved += format("%s%d", moved.empty() ? "" : ",", il);
                    }
                }
                if (!moved.empty()) {
                    fprintf(stderr, "%s: layer(s) %s did not fit whole on their GPU: they run completely on %s, the pool's GPU (attention and "
                            "context memory too)\n", __func__, moved.c_str(), ggml_backend_dev_name(ms_pool_dev));
                }
            }
            // MOE_STREAM_TEST_WHOLE_OTHER_CARD=L1,L2,.. (test only): these whole layers count as being on another GPU for the choice of the
            // layer a reload gives up (their weights stay where they are)
            if (const char * e = getenv("MOE_STREAM_TEST_WHOLE_OTHER_CARD")) {
                for (const std::string & item : split_by(e, ',')) {
                    const int il = atoi(item.c_str());
                    if (!item.empty() && il >= 0 && il < n_layer_all && ms.whole_dev[il] != nullptr) {
                        ms.whole_test_other.resize(n_layer_all, 0);
                        ms.whole_test_other[il] = 1;
                        fprintf(stderr, "%s: MOE_STREAM_TEST_WHOLE_OTHER_CARD (TEST ONLY): whole layer %d counts as being on another GPU\n", __func__, il);
                    }
                }
            }
            if (ms_pool_dev_set || wl_spec) {
                std::string wl;
                for (const int32_t il : ms.gpu_layers_pick) {
                    wl += format("%s%d: %s", wl.empty() ? "" : ", ", il, ggml_backend_dev_name(ms.whole_dev[il]));
                }
                fprintf(stderr, "%s: devices: the VRAM pool and the prompt matmuls of the RAM pools on %s%s; whole MoE layers' experts: %s\n", __func__,
                        ms_pool_dev ? ggml_backend_dev_name(ms_pool_dev) : "no GPU", ms_pool_dev_set ? " (--moe-stream-vram-device)" : " (the first GPU)",
                        wl.empty() ? "none" : wl.c_str());
            }
        }
        if (params.moe_stream_nonpool) {
            auto & ms    = *pimpl->moe_stream;
            ms.np_key     = params.moe_stream_nonpool->key;
            ms.np_sidecar = llama_moe_stream_sidecar_arg(params.moe_stream, ml.file_paths[0]);
            ms.np_planned = np_planned;
            ms.np_known   = np_known;
            ms.np_growth  = params.moe_stream_nonpool->growth;
            ms.np_srv_need = np_srv_need;
            ms.np_cache_ram = params.moe_stream_nonpool->cache_ram;
            ms.np_pinned  = params.moe_stream_nonpool->pinned;
        }
    }

    const auto TENSOR_NOT_REQUIRED = llama_model_loader::TENSOR_NOT_REQUIRED;

    // create tensors for the weights
    {
        // TODO: move to a separate function
        const auto tn = LLM_TN(arch);

        const int64_t n_expert = hparams.n_expert;

        if (n_expert > 0 && hparams.n_expert_used_max() == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        layers.resize(n_layer_all);

        // call the per-model loading function
        load_arch_tensors(ml);

        // generic pass: load optional per-tensor/per-expert ".scale" tensors (e.g. NVFP4 scale2)
        // this avoids having to add scale loading to every architecture
        for (int i = 0; i < n_layer_all; ++i) {
            auto & layer = layers[i];

            // attention weight scales (per-tensor, shape {1})
            if (!layer.wq_s && layer.wq) {
                layer.wq_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_s && layer.wk) {
                layer.wk_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_s && layer.wv) {
                layer.wv_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_s && layer.wo) {
                layer.wo_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_s && layer.wqkv) {
                layer.wqkv_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_s && layer.wqkv_gate) {
                layer.wqkv_gate_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // dense FFN weight scales (per-tensor, shape {1})
            if (!layer.ffn_gate_s && layer.ffn_gate) {
                layer.ffn_gate_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_s && layer.ffn_down) {
                layer.ffn_down_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_s && layer.ffn_up) {
                layer.ffn_up_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // MoE expert weight scales (per-expert, shape {n_expert})
            if (!layer.ffn_gate_exps_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }

            // recurrent / linear-attention weight scales (per-tensor, shape {1})
            if (!layer.ssm_in_s && layer.ssm_in) {
                layer.ssm_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_s && layer.ssm_out) {
                layer.ssm_out_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_s && layer.ssm_alpha) {
                layer.ssm_alpha_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_s && layer.ssm_beta) {
                layer.ssm_beta_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.eh_proj_s && layer.nextn.eh_proj) {
                layer.nextn.eh_proj_s = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.shared_head_head_s && layer.nextn.shared_head_head) {
                layer.nextn.shared_head_head_s = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // input scales
            if (!layer.wq_in_s && layer.wq) {
                layer.wq_in_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_in_s && layer.wk) {
                layer.wk_in_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_in_s && layer.wv) {
                layer.wv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_in_s && layer.wo) {
                layer.wo_in_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_in_s && layer.wqkv) {
                layer.wqkv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_in_s && layer.wqkv_gate) {
                layer.wqkv_gate_in_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_in_s && layer.ffn_gate) {
                layer.ffn_gate_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_in_s && layer.ffn_down) {
                layer.ffn_down_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_in_s && layer.ffn_up) {
                layer.ffn_up_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_exps_in_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_in_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_in_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_in_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_in_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_in_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_in_in_s && layer.ssm_in) {
                layer.ssm_in_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_in_s && layer.ssm_out) {
                layer.ssm_out_in_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_in_s && layer.ssm_alpha) {
                layer.ssm_alpha_in_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_in_s && layer.ssm_beta) {
                layer.ssm_beta_in_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.eh_proj_in_s && layer.nextn.eh_proj) {
                layer.nextn.eh_proj_in_s = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.nextn.shared_head_head_in_s && layer.nextn.shared_head_head) {
                layer.nextn.shared_head_head_in_s = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
        }
        // output scales
        if (output && output->type == GGML_TYPE_NVFP4) {
            // weight scale
            if (!output_s) {
                output_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "scale"), {1}, TENSOR_NOT_REQUIRED);
            }
            // input scale
            if (!output_in_s) {
                output_in_s = create_tensor(tn(LLM_TENSOR_OUTPUT, "input_scale"), {1}, TENSOR_NOT_REQUIRED);
            }
        }
    }

    ml.done_getting_tensors();

    if (per_layer_tok_embd && ml.lazy.has(per_layer_tok_embd)) {
        LLAMA_LOG_INFO("%s: enabling prefetch for '%s'\n", __func__, per_layer_tok_embd->name);

        can_prefetch.insert(per_layer_tok_embd);
    }

    // Tied NVFP4 output is valid when no separate LM-head scale tensors are present.
    // If sidecar scales exist, the output weight must be an actual output tensor.
    GGML_ASSERT(!(output && tok_embd &&
            strcmp(output->name, tok_embd->name) == 0 &&
            output->type == GGML_TYPE_NVFP4 &&
            (output_s || output_in_s)));
    // populate tensors_by_name
    for (auto & [_, ctx_ptr] : ml.ctx_map) {
        for (auto * cur = ggml_get_first_tensor(ctx_ptr.get()); cur != NULL; cur = ggml_get_next_tensor(ctx_ptr.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    // page-lock the pools. This is the first point where they exist: the pools are created
    // with their layer's first weight tensor, and their memory is ours to pin. The device is known
    // (the fit above used it), and pinning does not change any accounting - it only stops the kernel
    // from moving or swapping these pages.
    if (pimpl->moe_stream && params.moe_stream_pin) {
        ggml_backend_reg_t reg = devices.empty() ? nullptr
            : ggml_backend_dev_backend_reg(devices[0].dev);

        pimpl->moe_stream->pin_pools(reg);
    }

    // per-tensor activation precision policy
    prec_policy.load(ml, *this);

    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    // note: the buffers are owned by pimpl->ctxs_bufs, this is a raw-pointer view used to load the data
    std::vector<std::pair<ggml_context *, std::vector<ggml_backend_buffer_t>>> ctx_bufs;
    ctx_bufs.reserve(ml.ctx_map.size());

    // reserve hint: each context creates one buffer per file, or more when its tensors are spread
    //     across disjoint ranges of a file (see get_mapping_ranges)
    pimpl->ctxs_bufs.reserve(ml.ctx_map.size() * ml.files.size());

    for (auto & [ctx_key, ctx_ptr] : ml.ctx_map) {
        ggml_backend_buffer_type_t buft = ctx_key.buft;
        ggml_context * ctx = ctx_ptr.get();

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!dev) {
                throw std::runtime_error(format("%s: no CPU backend found", __func__));
            }
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

        std::vector<ggml_backend_buffer_ptr> bufs;

        // a lazy context is mapped whatever the load mode, but the memory-fit pass maps nothing
        const bool is_lazy_mapped = ctx_key.lazy && !ml.no_alloc;

        if ((ml.use_mmap || is_lazy_mapped) && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            GGML_ASSERT(!ml.no_alloc);
            const size_t max_size = ggml_get_max_tensor_size(ctx);

            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // map the mmap regions that contain the tensors of this context
                // this is important for metal with apple silicon: a backend buffer pins everything it
                //     wraps, so the regions owned by the other contexts (lazy tensors read on demand
                //     from disk, MoE tensors kept on the CPU, etc) are left out and stay out of RAM
                // this also allows using partial offloading when the model size exceeds the metal buffer
                //     size, but not the RAM size
                void * addr = ml.mappings.at(idx)->addr();

                for (const auto & range : ml.get_mapping_ranges(idx, ctx)) {
                    ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + range.first, range.last - range.first, max_size);
                    if (buf == nullptr) {
                        llama_model_alloc_host = ggml_backend_buft_is_host(buft);
                        llama_model_alloc_drop = pimpl->moe_stream ? pimpl->moe_stream->whole_layer_to_drop(ggml_backend_buft_get_device(buft)) : -2;
                        throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                    }
                    bufs.emplace_back(buf);
                }
            }
        } else {
            ggml_backend_buffer_t buf;
            if (ml.no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf; // set dummy buffer for weights so that the backend scheduler won't try to allocate them
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer
            }
            if (buf == nullptr) {
                llama_model_alloc_host = ggml_backend_buft_is_host(buft);   // a host buffer's failure is not helped by fewer GPU layers
                llama_model_alloc_drop = pimpl->moe_stream ? pimpl->moe_stream->whole_layer_to_drop(ggml_backend_buft_get_device(buft)) : -2;
                throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            // lock the buffers that hold the tensors: a multi_buffer has no base, and its parts are pageable memory when pinning fails
            if (use_mlock && !ml.no_alloc && ggml_backend_buffer_is_host(buf)) {
                std::vector<ggml_backend_buffer_t> lock_bufs;
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    if (t->buffer != nullptr && std::find(lock_bufs.begin(), lock_bufs.end(), t->buffer) == lock_bufs.end()) {
                        lock_bufs.push_back(t->buffer);
                    }
                }
                for (ggml_backend_buffer_t lock_buf : lock_bufs) {
                    pimpl->mlock_bufs.emplace_back(new llama_mlock);
                    auto & mlock_buf = pimpl->mlock_bufs.back();
                    mlock_buf->init   (ggml_backend_buffer_get_base(lock_buf));
                    mlock_buf->grow_to(ggml_backend_buffer_get_size(lock_buf));
                }
            }
            bufs.emplace_back(buf);
        }

        std::vector<ggml_backend_buffer_t> buf_ptrs;
        buf_ptrs.reserve(bufs.size());

        for (auto & buf : bufs) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            buf_ptrs.emplace_back(buf.get());
        }

        pimpl->ctxs_bufs.emplace_back(std::move(ctx_ptr), std::move(bufs));

        ctx_bufs.emplace_back(ctx, std::move(buf_ptrs));
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, n_layer_all);

        int n_repeating = n_gpu;
        if (n_repeating > 0) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
            n_repeating--;
        }
        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_repeating);

        const int max_backend_supported_layers = n_layer_all + 1;
        const int max_offloadable_layers       = n_layer_all + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & [_, bufs] : pimpl->ctxs_bufs) {
        for (auto & buf: bufs) {
            LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
        }
    }

    if (ml.no_alloc) {
        return true;
    }

    // without mmap, load non-host buffers first: their tensors go through a staging buffer, which is cheapest while the fewest weights are resident
    if (!ml.use_mmap) {
        std::stable_partition(ctx_bufs.begin(), ctx_bufs.end(), [](const auto & x) {
            const auto & bufs = x.second;
            return !bufs.empty() && !ggml_backend_buffer_is_host(bufs[0]);
        });
    }

    // load tensor data
    for (auto & [ctx, bufs] : ctx_bufs) {
        if (!ml.load_all_data(ctx, bufs, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback, params.progress_callback_user_data)) {
            return false;
        }
    }

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    if (pimpl->moe_stream) {
        // The GPU that runs each layer's router (null: the CPU), for needed-early's marks and the GPU guess's per-device rules
        {
            // A streamed layer L hands its 1-ahead guess to the helper only if the guess runs in
            // a GPU split. The guess is L+1's router matmul (build_moe_ffn's "pred" probs), which the scheduler places with its
            // weight, so the rule is where L+1's ffn_gate_inp landed (that also follows -ngl and any -ot override of the router).
            // From the buffer type's DEVICE - a GPU or integrated-GPU device = a
            // GPU split - and not a host buffer: an AMX / CPU_REPACK buffer's device is the CPU (and AMX's is_host is nullptr,
            // so is_host alone called it a device buffer), and a pinned CUDA_Host buffer's device is the CUDA device although the
            // scheduler runs its matmul on the CPU (CUDA's supports_buft takes CUDA buffers only). The CPU keeps the direct
            // add (its in-graph form is needed-early's op A / op B split)
            // The exact test - a GPU device of the model that supports the router weight's buffer type runs
            // the router (and the guess) of a decode word. On a discrete GPU this is the older rule's answer (its own buffers: yes; CPU and
            // CUDA_Host: no); an integrated GPU also supports the pinned CUDA_Host buffers, where the older rule said "host = CPU"
            std::vector<uint8_t> router_gpu(layers.size(), 0);
            std::vector<ggml_backend_dev_t> router_dev(layers.size(), nullptr);   // per layer, for the GPU producer
            for (size_t il = 0; il < layers.size(); il++) {
                const ggml_tensor * w = layers[il].ffn_gate_inp;
                if (w == nullptr || w->buffer == nullptr) {
                    continue;
                }
                ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(w->buffer);
                for (const llama_device & ld : devices) {
                    ggml_backend_dev_t dev = ld.dev;
                    const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
                    if ((t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) && ggml_backend_dev_supports_buft(dev, buft)) {
                        router_gpu[il] = 1;
                        router_dev[il] = dev;
                        break;
                    }
                }
            }
            // MOE_STREAM_TEST_ROUTER_OTHER_CARD=L1,L2,.. (test only): these layers' routers count as running on another GPU than every
            // other layer's (as with a layer on a second card), so the per-layer fallbacks of the marks can be seen on one GPU
            if (const char * e = getenv("MOE_STREAM_TEST_ROUTER_OTHER_CARD")) {
                std::string said;
                ggml_backend_dev_t other = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);   // (a device that is none of the GPUs)
                for (const char * c = e; *c != '\0'; ) {
                    char * end = nullptr;
                    const long il = strtol(c, &end, 10);
                    if (end == c) {
                        break;
                    }
                    if (il >= 0 && (size_t) il < router_dev.size() && router_dev[(size_t) il] != nullptr) {
                        router_dev[(size_t) il] = other;
                        said += format("%s%ld", said.empty() ? "" : ",", il);
                    }
                    c = *end == ',' ? end + 1 : end;
                }
                fprintf(stderr, "%s: MOE_STREAM_TEST_ROUTER_OTHER_CARD (TEST ONLY): the routers of layers %s count as running on another GPU\n",
                        __func__, said.empty() ? "none" : said.c_str());
            }
            pimpl->moe_stream->router_dev_all = router_dev;
            if (hparams.ink2_loop_passes <= 1 && params.moe_stream_needed_early == 1) {
                pimpl->moe_stream->ne_set_router_gpu(router_gpu);
                pimpl->moe_stream->ne_gpu_setup(router_dev);   // the GPU producers, or the host producer stays
            }
        }
        pimpl->moe_stream->ws_test_check();   // a test switch that would change nothing stops the run
        // --moe-stream-guess-combined 1: the combined guess step's stacked inputs per streamed layer - the gains
        // (ffn_norm) and selection biases (exp_probs_b) of its la_targets, in that order, copied once into small tensors on the
        // buffer type of those targets' routers (so the step runs where the separate guesses ran). A layer whose targets' routers
        // sit on different buffer types or differ in type / shape keeps the separate ops; a layer with one target too (nothing
        // to combine)
        // --moe-stream-correction FILE: the corrected guess matrices, before the combined step's stacks (their gain rows
        // are ones for a corrected pair)
        if (!pimpl->moe_stream->p1.correction_file.empty()) {
            pimpl->moe_stream->corr.load(*this, *pimpl->moe_stream, pimpl->moe_stream->p1.correction_file,
                    pimpl->moe_stream->p1.correction_base, pimpl->moe_stream->p1.correction_size);
        }
        if (pimpl->moe_stream->la_comb) {
            llama_moe_stream & ms = *pimpl->moe_stream;
            const int64_t n_embd_g = hparams.n_embd;
            const int64_t n_exp_g  = hparams.n_expert;
            std::map<ggml_backend_buffer_type_t, std::vector<int32_t>> by_buft;
            std::vector<uint8_t> with_bias(layers.size(), 0);
            int32_t n_ws_comb = 0;   // of the layers with the step, word-start sources (--moe-stream-wordstart)
            for (int32_t il = 0; il < (int32_t) layers.size(); il++) {
                if (!ms.layer(il) && !ms.ws_src(il)) {
                    continue;
                }
                int32_t ds[llama_moe_stream::LOOKAHEAD_MAX];
                const int32_t n = ms.la_targets(il, ds);
                if (n < 2) {
                    continue;
                }
                bool ok = true, has_b = false, no_b = false;
                ggml_backend_buffer_type_t bt = nullptr;
                const ggml_tensor * w0 = nullptr;
                for (int32_t t = 0; t < n && ok; t++) {
                    const llama_layer & L = layers[il + ds[t]];
                    const ggml_tensor * w = ms.corr_gate(il + ds[t], ds[t]) ? ms.corr_gate(il + ds[t], ds[t]) : L.ffn_gate_inp;
                    // The combined norm runs on the first target's layer device, each separate norm on its own
                    // target's (the "norm" pin) - so every target must be on the same layer device, whatever the routers' buffers
                    if ((size_t) (il + ds[t]) >= pimpl->dev_layer.size() || pimpl->dev_layer[il + ds[t]].dev != pimpl->dev_layer[il + ds[0]].dev) {
                        ok = false;
                        break;
                    }
                    if (w == nullptr || w->buffer == nullptr) {
                        ok = false;
                        break;
                    }
                    ggml_backend_buffer_type_t wbt = ggml_backend_buffer_get_type(w->buffer);
                    ok = ok && (bt == nullptr || wbt == bt) && (w0 == nullptr || (w->type == w0->type && ggml_are_same_shape(w, w0)));
                    bt = bt ? bt : wbt;
                    w0 = w0 ? w0 : w;
                    if (ms.pred_shared) {
                        ok = ok && L.ffn_norm && L.ffn_norm->type == GGML_TYPE_F32 && ggml_nelements(L.ffn_norm) == n_embd_g;
                    }
                    if (L.ffn_exp_probs_b) {
                        has_b = true;
                        ok = ok && L.ffn_exp_probs_b->type == GGML_TYPE_F32 && ggml_nelements(L.ffn_exp_probs_b) == n_exp_g;
                    } else {
                        no_b = true;
                    }
                }
                // The routers' backend must run GGML_OP_MUL_MAT_MULTI (CPU and CUDA do; Vulkan, Metal, SYCL do not) -
                // asked on a dummy op over the real routers, as the scheduler would ask
                if (ok && !(has_b && no_b)) {
                    ggml_backend_dev_t dev = ggml_backend_buft_get_device(bt);
                    // The plain CPU buffer type has no device (ggml-backend.cpp's ggml_backend_cpu_buffer_type, "FIXME"), so routers
                    // in host memory (every CPU-only run: Flint's and Ling's CPU-only commands) were refused here and never used the step;
                    // a host buffer type without a device is the CPU device's
                    if (dev == nullptr && ggml_backend_buft_is_host(bt)) {
                        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                    }
                    ggml_init_params dp = { ggml_tensor_overhead()*4 + ggml_graph_overhead_custom(4, false), nullptr, true };
                    // No probe context = this layer keeps the separate guess ops;
                    // MOE_STREAM_TEST_E6_PROBE=1 (test only): every probe fails
                    static const bool test_e6 = getenv("MOE_STREAM_TEST_E6_PROBE") && atoi(getenv("MOE_STREAM_TEST_E6_PROBE")) != 0;
                    ggml_context * dctx = test_e6 ? nullptr : ggml_init(dp);
                    if (dctx == nullptr) {
                        static bool said = false;
                        if (!said) {
                            said = true;
                            fprintf(stderr, "%s: guess-combined: the backend probe could not be built%s - off for the layers it could not "
                                    "probe (their guesses are made one by one, the same guesses)\n", __func__, test_e6 ? " (TEST ONLY: MOE_STREAM_TEST_E6_PROBE)" : "");
                        }
                        ms.la_comb_sep += 1;
                        continue;
                    }
                    ggml_tensor * as[llama_moe_stream::LOOKAHEAD_MAX];
                    for (int32_t t = 0; t < n; t++) {
                        as[t] = ms.corr_gate(il + ds[t], ds[t]) ? ms.corr_gate(il + ds[t], ds[t]) : layers[il + ds[t]].ffn_gate_inp;
                    }
                    ggml_tensor * col = ggml_new_tensor_3d(dctx, GGML_TYPE_F32, as[0]->ne[0], 1, n);
                    ggml_tensor * op  = ggml_mul_mat_multi(dctx, as, n, col);
                    ok = dev != nullptr && ggml_backend_dev_supports_op(dev, op);
                    ggml_free(dctx);
                }
                if (!ok || (has_b && no_b)) {
                    ms.la_comb_sep += 1;
                    continue;
                }
                with_bias[il] = has_b ? 1 : 0;
                by_buft[bt].push_back(il);
            }
            size_t stack_bytes = 0;
            for (auto & [bt, list] : by_buft) {
                ggml_init_params ip = { ggml_tensor_overhead()*list.size()*2 + 256, nullptr, true };
                ggml_context * sctx = ggml_init(ip);
                std::vector<std::pair<ggml_tensor *, ggml_tensor *>> st(list.size(), { nullptr, nullptr });
                for (size_t k = 0; k < list.size(); k++) {
                    int32_t ds[llama_moe_stream::LOOKAHEAD_MAX];
                    const int32_t n = ms.la_targets(list[k], ds);
                    if (ms.pred_shared) {
                        st[k].first = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, n_embd_g, 1, n);
                        ggml_format_name(st[k].first, "moe_guess_gains-%d", list[k]);
                    }
                    if (with_bias[list[k]]) {
                        st[k].second = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, n_exp_g, n);
                        ggml_format_name(st[k].second, "moe_guess_biases-%d", list[k]);
                    }
                }
                ggml_backend_buffer_t sbuf = nullptr;
                if (ggml_get_first_tensor(sctx) != nullptr) {
                    sbuf = ggml_backend_alloc_ctx_tensors_from_buft(sctx, bt);
                    if (sbuf == nullptr) {   // these layers keep the separate guess ops
                        fprintf(stderr, "%s: guess-combined: the stacked gains / biases (%zu layers) could not be allocated on %s - "
                                "off for them (their guesses are made one by one, the same guesses)\n", __func__, list.size(), ggml_backend_buft_name(bt));
                        ms.la_comb_sep += (int32_t) list.size();
                        ggml_free(sctx);
                        continue;
                    }
                    ggml_backend_buffer_set_usage(sbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    stack_bytes += ggml_backend_buffer_get_size(sbuf);
                }
                std::vector<uint8_t> tmp;
                for (size_t k = 0; k < list.size(); k++) {
                    int32_t ds[llama_moe_stream::LOOKAHEAD_MAX];
                    const int32_t n = ms.la_targets(list[k], ds);
                    for (int32_t t = 0; t < n; t++) {
                        const llama_layer & L = layers[list[k] + ds[t]];
                        for (int j = 0; j < 2; j++) {
                            ggml_tensor * dst = j == 0 ? st[k].first : st[k].second;
                            // A corrected pair's gain is folded into its matrix W', so its gain row is the ones
                            const ggml_tensor * src = j == 0 ? (ms.corr_unit(list[k] + ds[t], ds[t]) ? ms.corr_unit(list[k] + ds[t], ds[t]) : L.ffn_norm)
                                                             : L.ffn_exp_probs_b;
                            if (dst == nullptr) {
                                continue;
                            }
                            const size_t nb = ggml_nbytes(src);
                            tmp.resize(nb);
                            ggml_backend_tensor_get(src, tmp.data(), 0, nb);
                            ggml_backend_tensor_set(dst, tmp.data(), (size_t) t*nb, nb);
                        }
                    }
                    ms.la_set_stack(list[k], st[k].first, st[k].second);
                    ms.la_comb_layers += 1;
                    n_ws_comb += ms.ws_src(list[k]) ? 1 : 0;
                }
                ms.la_stack_ctx.push_back(sctx);
                if (sbuf) {
                    ms.la_stack_buf.push_back(sbuf);
                }
            }
            fprintf(stderr, "%s: guess-combined: %d streamed layers build their guesses in one step (%.1f KiB of stacked gains / "
                    "biases on %zu buffer type(s)); %d layers with 2+ targets keep the separate ops (routers on different buffer "
                    "types or layer devices, another type / shape, or a backend without the op)%s\n", __func__,
                    ms.la_comb_layers - n_ws_comb, stack_bytes/1024.0, by_buft.size(), ms.la_comb_sep,
                    ms.ws_on() ? format("; word-start sources with the step: %d", n_ws_comb).c_str() : "");
        }
        pimpl->moe_stream->verify_pools();
        pimpl->moe_stream->print_repack_state();
    }

    return true;
}

ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;

    // --moe-stream-gpu-layers-device: a whole MoE layer's routed experts on the GPU it names, the rest of the layer where its layer is
    // (an -ot override of the tensor still wins, in the loader)
    if (pimpl->moe_stream && tn.bid >= 0 && (size_t) tn.bid < pimpl->moe_stream->whole_dev.size() && tn.suffix != nullptr &&
            strcmp(tn.suffix, "weight") == 0) {
        ggml_backend_dev_t wd = pimpl->moe_stream->whole_dev[(size_t) tn.bid];
        if (wd != nullptr && wd != pimpl->dev_layer.at(tn.bid).dev && pimpl->gpu_buft_list.count(wd) > 0 &&
                llama_moe_stream_is_exps(tn.str()) && !pimpl->moe_stream->has_layer(tn.bid)) {
            buft_list_layer = &pimpl->gpu_buft_list.at(wd);
        }
    }

    // a streamed layer has no exps tensors in the model file: the pool of expert slots takes their
    // place, and the loader is told to skip them so that the size accounting adds up
    if (pimpl->moe_stream && tn.bid >= 0 && buft_list_layer != nullptr && tn.suffix != nullptr &&
        strcmp(tn.suffix, "weight") == 0 && (flags & (TENSOR_DUPLICATED | TENSOR_SKIP | TENSOR_SKIP_IF_VIRTUAL)) == 0) {
        const std::string name = tn.str();

        if (llama_moe_stream_is_exps(name) && pimpl->moe_stream->has_layer(tn.bid)) {
            const auto * w = ml.get_weight(name.c_str());

            bool dims_ok = w != nullptr && w->tensor->ne[2] == hparams.n_expert;
            if (dims_ok) {
                size_t dim = 0;
                for (int64_t d : ne) {
                    dims_ok = dims_ok && d == w->tensor->ne[dim++];
                }
            }

            if (dims_ok) {
                ml.create_tensor(
                    hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
                    tn, ne, flags | TENSOR_STREAMED);

                return pimpl->moe_stream->create_pool(tn.bid, name, w->tensor);
            }
        }
    }

    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
        tn, ne, flags);
}

std::string llama_model::arch_name() const {
    return llm_arch_name(arch);
}

std::string llama_model::type_name() const {
    return llm_type_name(type);
}

std::string llama_model::desc() const {
    return pimpl->desc_str;
}

llama_ftype llama_model::ftype() const {
    return pimpl->ftype;
}

size_t llama_model::size() const {
    return pimpl->n_bytes;
}

size_t llama_model::n_tensors() const {
    return tensors_by_name.size();
}

size_t llama_model::n_devices() const {
    return devices.size();
}

const float * llama_model::tensor_split() const {
    return params.tensor_split;
}

uint32_t llama_model::n_gpu_layers() const {
    // note: plus 1 for the "output" layer
    return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer_all + 1;
}

llama_split_mode llama_model::split_mode() const {
    return params.split_mode;
}

bool llama_model::no_alloc() const {
    return params.no_alloc;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_model::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        if (no_alloc()) {
            GGML_ASSERT(bufs.size() == 1);
            ggml_backend_buffer_t buf = bufs[0].get();
            GGML_ASSERT(ggml_backend_buffer_get_base(buf) == nullptr);
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            for (const auto & buf : bufs) {
                // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
                ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
            }
        }
    }
    return ret;
}

uint64_t llama_model::n_elements() const {
    return pimpl->n_elements;
}

void llama_model::print_info() const {
    const std::string rope_scaling_type = llama_rope_scaling_type_name(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<int32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<int32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        } else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LLAMA_LOG_INFO("%s: arch                  = %s\n",     __func__, arch_name().c_str());
    LLAMA_LOG_INFO("%s: vocab_only            = %d\n",     __func__, hparams.vocab_only);
    LLAMA_LOG_INFO("%s: no_alloc              = %d\n",     __func__, no_alloc());

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train           = %u\n",     __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd_inp            = %u\n",     __func__, hparams.n_embd_inp());
        LLAMA_LOG_INFO("%s: n_embd                = %u\n",     __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_embd_out            = %u\n",     __func__, hparams.n_embd_out());
        LLAMA_LOG_INFO("%s: n_layer               = %u\n",     __func__, hparams.n_layer());
        LLAMA_LOG_INFO("%s: n_layer_all           = %u\n",     __func__, hparams.n_layer_all);
        LLAMA_LOG_INFO("%s: n_head                = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv             = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_rot                 = %u\n",     __func__, hparams.n_rot_full);
        LLAMA_LOG_INFO("%s: n_swa                 = %u\n",     __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: is_swa_any            = %u\n",     __func__, hparams.is_swa_any());
        LLAMA_LOG_INFO("%s: non_causal_type       = %d\n",     __func__, hparams.non_causal_type);
        LLAMA_LOG_INFO("%s: n_embd_head_k         = %u\n",     __func__, hparams.n_embd_head_k_full);
        LLAMA_LOG_INFO("%s: n_embd_head_v         = %u\n",     __func__, hparams.n_embd_head_v_full);
        LLAMA_LOG_INFO("%s: n_gqa                 = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_gqa(il);        }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_embd_k_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_embd_v_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: f_norm_eps            = %.1e\n",   __func__, hparams.f_norm_eps);
        LLAMA_LOG_INFO("%s: f_norm_rms_eps        = %.1e\n",   __func__, hparams.f_norm_rms_eps);
        LLAMA_LOG_INFO("%s: f_clamp_kqv           = %.1e\n",   __func__, hparams.f_clamp_kqv);
        LLAMA_LOG_INFO("%s: f_max_alibi_bias      = %.1e\n",   __func__, hparams.f_max_alibi_bias);
        LLAMA_LOG_INFO("%s: f_logit_scale         = %.1e\n",   __func__, hparams.f_logit_scale);
        LLAMA_LOG_INFO("%s: f_attn_scale          = %.1e\n",   __func__, hparams.f_attention_scale);
        LLAMA_LOG_INFO("%s: f_attn_value_scale    = %.4f\n",   __func__, hparams.f_attn_value_scale);
        LLAMA_LOG_INFO("%s: n_ff                  = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer_all).c_str());
        LLAMA_LOG_INFO("%s: n_expert              = %u\n",     __func__, hparams.n_expert);
        LLAMA_LOG_INFO("%s: n_expert_used         = %u\n",     __func__, hparams.n_expert_used());
        LLAMA_LOG_INFO("%s: n_expert_groups       = %d\n",     __func__, hparams.n_expert_groups);
        LLAMA_LOG_INFO("%s: n_group_used          = %d\n",     __func__, hparams.n_group_used);
        LLAMA_LOG_INFO("%s: causal attn           = %d\n",     __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type          = %d\n",     __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type             = %d\n",     __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling          = %s\n",     __func__, rope_scaling_type.c_str());
        LLAMA_LOG_INFO("%s: freq_base_train       = %.1f\n",   __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train      = %g\n",     __func__, hparams.rope_freq_scale_train);
        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
            LLAMA_LOG_INFO("%s: freq_base_swa         = %.1f\n",   __func__, hparams.rope_freq_base_train_swa);
            LLAMA_LOG_INFO("%s: freq_scale_swa        = %g\n",     __func__, hparams.rope_freq_scale_train_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_k_swa     = %u\n",     __func__, hparams.n_embd_head_k_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_v_swa     = %u\n",     __func__, hparams.n_embd_head_v_swa);
            LLAMA_LOG_INFO("%s: n_rot_swa             = %u\n",     __func__, hparams.n_rot_swa);
        }
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn       = %u\n",     __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_yarn_log_mul     = %.4f\n",   __func__, hparams.rope_yarn_log_mul);
        LLAMA_LOG_INFO("%s: rope_finetuned        = %s\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
        if (arch == LLM_ARCH_GRANITE &&
            std::any_of(hparams.deepstack_mapping_arr.begin(),
                        hparams.deepstack_mapping_arr.end(),
                        [](const auto & entry) { return entry >= 0; })) {
            LLAMA_LOG_INFO("%s: deepstack_mapping_arr = %s\n", __func__,
                           print_f([&](uint32_t il) { return hparams.deepstack_mapping_arr[il]; },
                           hparams.n_layer_all).c_str());
        }
        // MRoPE (Multi-axis Rotary Position Embedding) sections
        if (const auto & s = hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
            LLAMA_LOG_INFO("%s: mrope sections        = [%d, %d, %d, %d]\n", __func__, s[0], s[1], s[2], s[3]);
        }
        if (!classifier_labels.empty()) {
            LLAMA_LOG_INFO("%s: n_cls_out             = %u\n", __func__, hparams.n_cls_out);

            size_t i = 0;
            for (const auto & label : classifier_labels) {
                LLAMA_LOG_INFO("%s: cls_label[%2zu]         = %s\n", __func__, i++, label.c_str());
            }
        }

        if (arch == LLM_ARCH_MAMBA ||
                arch == LLM_ARCH_MAMBA2 ||
                arch == LLM_ARCH_JAMBA ||
                arch == LLM_ARCH_FALCON_H1 ||
                arch == LLM_ARCH_PLAMO2 ||
                arch == LLM_ARCH_GRANITE_HYBRID ||
                arch == LLM_ARCH_QWEN3NEXT ||
                arch == LLM_ARCH_QWEN35 ||
                arch == LLM_ARCH_QWEN35MOE ||
                arch == LLM_ARCH_NEMOTRON_H ||
                arch == LLM_ARCH_NEMOTRON_H_MOE) {
            LLAMA_LOG_INFO("%s: ssm_d_conv            = %u\n",     __func__, hparams.ssm_d_conv);
            LLAMA_LOG_INFO("%s: ssm_d_inner           = %u\n",     __func__, hparams.ssm_d_inner);
            LLAMA_LOG_INFO("%s: ssm_d_state           = %u\n",     __func__, hparams.ssm_d_state);
            LLAMA_LOG_INFO("%s: ssm_dt_rank           = %u\n",     __func__, hparams.ssm_dt_rank);
            LLAMA_LOG_INFO("%s: ssm_n_group           = %u\n",     __func__, hparams.ssm_n_group);
            LLAMA_LOG_INFO("%s: ssm_dt_b_c_rms        = %d\n",     __func__, hparams.ssm_dt_b_c_rms);
        }

        LLAMA_LOG_INFO("%s: model type            = %s\n",     __func__, type_name().c_str());
        if (pimpl->n_elements >= 1e12) {
            LLAMA_LOG_INFO("%s: model params          = %.2f T\n", __func__, pimpl->n_elements*1e-12);
        } else if (pimpl->n_elements >= 1e9) {
            LLAMA_LOG_INFO("%s: model params          = %.2f B\n", __func__, pimpl->n_elements*1e-9);
        } else if (pimpl->n_elements >= 1e6) {
            LLAMA_LOG_INFO("%s: model params          = %.2f M\n", __func__, pimpl->n_elements*1e-6);
        } else {
            LLAMA_LOG_INFO("%s: model params          = %.2f K\n", __func__, pimpl->n_elements*1e-3);
        }

        // general kv
        LLAMA_LOG_INFO("%s: general.name          = %s\n",    __func__, name.c_str());

        if (arch == LLM_ARCH_DEEPSEEK) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
        }

        if (arch == LLM_ARCH_DEEPSEEK2 || arch == LLM_ARCH_DEEPSEEK2OCR ||
                arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA ||
                arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_MISTRAL4 ||
                arch == LLM_ARCH_HY_V4) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_lora_q              = %d\n",     __func__, hparams.n_lora_q);
            LLAMA_LOG_INFO("%s: n_lora_kv             = %d\n",     __func__, hparams.n_lora_kv);
            LLAMA_LOG_INFO("%s: n_embd_head_k_mla     = %d\n",     __func__, hparams.n_embd_head_k_mla());
            LLAMA_LOG_INFO("%s: n_embd_head_v_mla     = %d\n",     __func__, hparams.n_embd_head_v_mla());
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
        }

        if (arch == LLM_ARCH_QWEN2MOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n",     __func__, hparams.n_ff_shexp);
        }

        if (arch == LLM_ARCH_MELLUM ||
                arch == LLM_ARCH_COHERE2MOE ||
                arch == LLM_ARCH_QWEN3MOE ||
                arch == LLM_ARCH_OPENAI_MOE ||
                arch == LLM_ARCH_QWEN3VLMOE ||
                arch == LLM_ARCH_RND1) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
        }

        if (arch == LLM_ARCH_MINICPM ||
                arch == LLM_ARCH_GRANITE ||
                arch == LLM_ARCH_GRANITE_MOE ||
                arch == LLM_ARCH_GRANITE_HYBRID ||
                arch == LLM_ARCH_GRANITE_SWITCH ||
                arch == LLM_ARCH_NEMOTRON_H_MOE) {
            LLAMA_LOG_INFO("%s: f_embedding_scale     = %f\n", __func__, hparams.f_embedding_scale);
            LLAMA_LOG_INFO("%s: f_residual_scale      = %f\n", __func__, hparams.f_residual_scale);
            LLAMA_LOG_INFO("%s: f_attention_scale     = %f\n", __func__, hparams.f_attention_scale);
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n", __func__, hparams.n_ff_shexp);
        }

        if (arch == LLM_ARCH_BAILINGMOE) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
        }

        if (arch == LLM_ARCH_BAILINGMOE2 || arch == LLM_ARCH_BAILINGMOE3) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n",     __func__, hparams.n_ff_shexp);
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
            LLAMA_LOG_INFO("%s: n_layer_nextn         = %d\n",     __func__, hparams.n_layer_nextn);
        }

        if (arch == LLM_ARCH_SMALLTHINKER || arch == LLM_ARCH_LFM2MOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
        }

        if (arch == LLM_ARCH_GROVEMOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp());
            LLAMA_LOG_INFO("%s: n_ff_chexp            = %d\n",     __func__, hparams.n_ff_chexp);
            LLAMA_LOG_INFO("%s: n_group_experts       = %d\n",     __func__, hparams.n_group_experts);
            LLAMA_LOG_INFO("%s: expert_group_scale    = %.2f\n",   __func__, hparams.expert_group_scale);
        }
    }

    vocab.print_info();
}

ggml_backend_dev_t llama_model::dev_layer(int il) const {
    return pimpl->dev_layer.at(il).dev;
}

ggml_backend_dev_t llama_model::dev_output() const {
    return pimpl->dev_output.dev;
}

template<typename F>
static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error(format("failed to create ggml context"));
    }

    ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, 0) };
    ggml_tensor * op_tensor = fn(ctx.get());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op_tensor->src[i] != nullptr) {
            assert(op_tensor->src[i]->buffer == nullptr);
            op_tensor->src[i]->buffer = buf.get();
        }
    }

    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);

    return op_supported;
}

template<typename F>
static ggml_backend_buffer_type_t select_buft(const buft_list_t & buft_list, const F & fn) {
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (buft_supported(cur_buft, cur_dev, fn)) {
            return cur_buft;
        }
    }

    throw std::runtime_error(format("no suitable buffer type found"));
}

ggml_backend_buffer_type_t llama_model::select_buft(int il) const {
    return ::select_buft(
            *pimpl->dev_layer.at(il).buft_list,
            [&](ggml_context * ctx) {
                ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                ggml_tensor * layer_dir = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                return ggml_add(ctx, cur, layer_dir);
            });
}

llama_moe_stream * llama_model::moe_stream() const {
    return pimpl->moe_stream.get();
}

// read a scalar u32 of the model metadata, false when it is missing or another type
static bool moe_stream_kv_u32(const gguf_context * ctx, const std::string & key, uint32_t & out) {
    const int id = gguf_find_key(ctx, key.c_str());

    if (id < 0) {
        return false;
    }

    if (gguf_get_kv_type(ctx, id) == GGUF_TYPE_UINT32) {
        out = gguf_get_val_u32(ctx, id);
        return true;
    }

    if (gguf_get_kv_type(ctx, id) == GGUF_TYPE_ARRAY) {
        // a per-layer array, the plan needs the largest value
        if (gguf_get_arr_type(ctx, id) != GGUF_TYPE_UINT32) {
            return false;
        }

        const size_t n = gguf_get_arr_n(ctx, id);
        const auto * v = (const uint32_t *) gguf_get_arr_data(ctx, id);

        for (size_t i = 0; i < n; i++) {
            out = std::max(out, v[i]);
        }

        return n > 0;
    }

    return false;
}

bool llama_moe_stream_get_plan(
        const char * model_path,
        const char * sidecar_json,
        int64_t     ram_budget,
        int32_t     n_slots,
        int32_t     io_threads,
        int32_t     prefetch,
        int32_t     prefetch_m,
        int32_t     max_waves,
        const char * oracle,
        int32_t     oracle_lead,
        int64_t     ram_free,
        bool        cpu_repack,
        int32_t     record_words,
        int32_t     gpu_layers,
        const char * gpu_layers_list,
        const char * gpu_layers_mode,
        const struct llama_moe_stream_srv * srv,
        int32_t     guess_top,
        int32_t     lookahead,
        int32_t     guess_pace,
        int32_t     needed_early,
        bool        hugepages,
        bool        pin,
        int32_t     lazy_mode,
        const char * correction,
        int32_t     pred_shared,
        int32_t     ps_rows,
        const struct llama_moe_stream_nonpool * nonpool,
        struct llama_moe_stream_plan * out) {
    if (!model_path || !sidecar_json) {
        LLAMA_LOG_ERROR("%s: a model and a sidecar are needed\n", __func__);
        return false;
    }

    std::unique_ptr<gguf_context, void (*)(gguf_context *)> ctx(
            gguf_init_from_file(model_path, { /* no_alloc = */ true, /* ctx = */ nullptr }), gguf_free);

    if (!ctx) {
        LLAMA_LOG_ERROR("%s: cannot read the metadata of '%s'\n", __func__, model_path);
        return false;
    }

    const int id_arch = gguf_find_key(ctx.get(), "general.architecture");

    if (id_arch < 0) {
        LLAMA_LOG_ERROR("%s: '%s' has no general.architecture\n", __func__, model_path);
        return false;
    }

    const std::string arch = gguf_get_val_str(ctx.get(), id_arch);
    const LLM_KV       kv(llm_arch_from_string(arch));

    llama_moe_stream::model_info mi;
    mi.arch = arch;

    if (!moe_stream_kv_u32(ctx.get(), kv(LLM_KV_BLOCK_COUNT), mi.n_layer_all) ||
        !moe_stream_kv_u32(ctx.get(), kv(LLM_KV_EXPERT_COUNT), mi.n_expert)) {
        LLAMA_LOG_ERROR("%s: '%s' is not a model with blocks and routed experts\n", __func__, model_path);
        return false;
    }
    {
        uint32_t phys_blocks = 0;
        if (moe_stream_kv_u32(ctx.get(), kv(LLM_KV_INK2_PHYS_BLOCK_COUNT), phys_blocks) && phys_blocks > 0) {
            mi.n_layer_all = phys_blocks;
        }
    }

    moe_stream_kv_u32(ctx.get(), kv(LLM_KV_NEXTN_PREDICT_LAYERS),     mi.n_layer_nextn);
    moe_stream_kv_u32(ctx.get(), kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT), mi.n_layer_dense);
    moe_stream_kv_u32(ctx.get(), kv(LLM_KV_EXPERT_USED_COUNT),         mi.n_expert_used);
    moe_stream_kv_u32(ctx.get(), kv(LLM_KV_EMBEDDING_LENGTH),          mi.n_embd);   // the GPU producer's router slots
    moe_stream_kv_u32(ctx.get(), kv(LLM_KV_CONTEXT_LENGTH),            mi.n_ctx_train);   // the wave plan's tables

    if (mi.n_layer_nextn > mi.n_layer_all) {
        LLAMA_LOG_ERROR("%s: '%s' has %u nextn blocks of %u\n", __func__, model_path, mi.n_layer_nextn, mi.n_layer_all);
        return false;
    }

    try {
        // the same manager with no I/O threads and no pools: everything it computes is the plan
        // The plan has no loader, so the size of a split model comes from the shard files
        const size_t model_bytes = llama_moe_stream_model_bytes(model_path);

        // The load adds the weights that stay in RAM to the non-pool memory and the
        // plan did not, so a run could pass the plan gate and then fail at load. The plan has no device
        // list; token_embd is the one tensor that always stays on the CPU (the input layer), so it is
        // counted here. With -ngl lower than the layer count the load keeps more in RAM than this.
        int64_t plan_ram_weights = 0;
        {
            const int64_t it_embd = gguf_find_tensor(ctx.get(), "token_embd.weight");
            if (it_embd >= 0) {
                plan_ram_weights = (int64_t) gguf_get_tensor_size(ctx.get(), it_embd);
                // The word table of an arch that marks it lazy is mapped from the file (as in the load): not RAM of ours.
                // The same rule as the loader's lazy_read::add - mode ON always, AUTO above the --moe-stream minimum, OFF never
                // (point C) the plan has no device list of the load: the registered GPU / iGPU devices stand for it; with none registered the plan
                // cannot tell, and counts the table as RAM (the safe side)
                // the loader's rule (llama.cpp): RPC devices and GPUs, iGPUs only when there is no (non-RPC) GPU
                std::vector<ggml_backend_dev_t> plan_devs, plan_gpus, plan_igpus;
                for (size_t di = 0; di < ggml_backend_dev_count(); di++) {
                    ggml_backend_dev_t pd = ggml_backend_dev_get(di);
                    const bool is_rpc = strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(pd)), "RPC") == 0;
                    if (ggml_backend_dev_type(pd) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                        (is_rpc ? plan_devs : plan_gpus).push_back(pd);
                    } else if (ggml_backend_dev_type(pd) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                        plan_igpus.push_back(pd);
                    }
                }
                plan_devs.insert(plan_devs.end(), plan_gpus.begin(), plan_gpus.end());
                if (plan_gpus.empty()) {
                    plan_devs.insert(plan_devs.end(), plan_igpus.begin(), plan_igpus.end());
                }
                const bool plan_lazy_mode_ok = lazy_mode == LLAMA_LAZY_MODE_ON ||
                        (lazy_mode == LLAMA_LAZY_MODE_AUTO && ggml_backend_dev_count() > 0 && llama_devices_all_mmap(plan_devs));
                if (llama_arch_marks_token_embd_lazy(llm_arch_from_string(arch)) && gguf_find_tensor(ctx.get(), "output.weight") >= 0 &&
                        plan_lazy_mode_ok && llama_mmap::SUPPORTED &&
                        (lazy_mode == LLAMA_LAZY_MODE_ON || (size_t) plan_ram_weights > llama_model_loader::lazy_read::moe_stream_min_bytes())) {
                    plan_ram_weights = 0;
                }
            }
        }

        // A server's snapshots and restore buffer, from the model's dimensions as the load computes them
        // (a vocab-only load reads the header's hyperparameters, no tensor)
        int64_t plan_srv_bytes = 0, plan_srv_need = 0;
        if (srv) {
            // the load's own first steps (llama.cpp's llama_model_load: the loader on the header, the architecture, the
            // hyperparameters) and nothing more - no device, no tensor
            llama_model_params mp = llama_model_default_params();
            mp.no_alloc = true;
            std::vector<std::string> splits;
            llama_model_loader ml(nullptr, nullptr, nullptr, model_path, splits, nullptr, mp.load_mode,
                    /* check_tensors = */ false, /* no_alloc = */ true, /* load_mtp = */ false, nullptr, nullptr);
            std::unique_ptr<llama_model> m(llama_model_create(ml, mp));
            auto * mb = dynamic_cast<llama_model_base *>(m.get());
            if (!mb) {
                LLAMA_LOG_ERROR("%s: cannot read the hyperparameters of '%s'\n", __func__, model_path);
                return false;
            }
            mb->load_hparams(ml);
            plan_srv_bytes = llama_moe_stream_srv_extra(llama_moe_stream_srv_compute(mb->hparams, mb->arch, *srv), *srv, __func__);
            plan_srv_need  = llama_moe_stream_srv_compute(mb->hparams, mb->arch, *srv).need;
        }
        const int64_t plan_np_bytes = nonpool ? llama_moe_stream_np_bytes(*nonpool, llama_moe_stream_sidecar_arg(sidecar_json, model_path).c_str(), plan_srv_need, plan_srv_bytes, __func__)
                                              : llama_moe_stream_other_reserve() + plan_srv_bytes;

        // --moe-stream-record's RAM (record_words > 0), so that the plan gate sees what the run will hold
        int64_t plan_record_bytes = 0;
        if (record_words > 0) {
            const int64_t n_moe = (int64_t) mi.n_layer_all - (int64_t) mi.n_layer_dense - (int64_t) mi.n_layer_nextn;

            plan_record_bytes = llama_moe_stream_record_bytes(n_moe > 0 ? n_moe : 0, (int64_t) mi.n_layer_all,
                    (int64_t) mi.n_expert, (int64_t) mi.n_expert_used, record_words,
                    io_threads > 0 ? io_threads : 8,
                    /* helper = */ true,    // the plan is not told --moe-stream-needed-early, so it counts the
                                            // helper's buffer (24 MiB) as if it were on: an upper bound, like the rest of its count
                    guess_top,
                    lookahead > 0 ? lookahead : 1, guess_pace > 0 ? guess_pace : 1);
        }

        // The correction file's tensors (--moe-stream-correction). The plan has no device list, like the token table above: they count as RAM when no
        // discrete GPU is registered (a CPU-only machine); with a GPU the plan keeps the routers there (as for every block's weights). NOTE: Metal reports itself as a GPU device
        // (ggml-metal.cpp ggml_backend_metal_device_get_type), so on an Apple machine neither this plan nor the load-time walk counts the file as RAM although the memory is shared - to do
        int64_t plan_corr_bytes = 0;
        if (correction && correction[0] && pred_shared == 1 && prefetch != 0) {
            bool plan_has_gpu = false;
            for (size_t di = 0; di < ggml_backend_dev_count(); di++) {
                ggml_backend_dev_t pd = ggml_backend_dev_get(di);
                plan_has_gpu = plan_has_gpu || (ggml_backend_dev_type(pd) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                        strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(pd)), "RPC") != 0);
            }
            if (!plan_has_gpu) {
                llama_moe_correction_plan_in cin;
                const llama_moe_corr_ref cref = llama_moe_correction_ref(correction, model_path, /* say = */ false);   // (the word `model`: the data in the model file, as the load reads it)
                cin.path = cref.path;  cin.base = cref.base;  cin.size = cref.size;
                cin.arch = arch;  cin.n_embd = (int64_t) mi.n_embd;  cin.n_expert = (int64_t) mi.n_expert;
                cin.n_layer_all = mi.n_layer_all;  cin.n_layer_dense = mi.n_layer_dense;  cin.n_layer_nextn = mi.n_layer_nextn;
                cin.lookahead = lookahead > 1 ? lookahead : 1;
                for (const auto & [x, b] : llama_moe_correction_plan(cin)) {
                    plan_corr_bytes += b;
                }
            }
        }

        llama_moe_stream ms(
                sidecar_json,
                "",
                model_path,
                mi,
                model_bytes,
                n_slots > 0 ? (uint32_t) n_slots : 0,
                /* io_threads = */ 0,
                prefetch != 0,
                prefetch_m > 0 ? (uint32_t) prefetch_m : mi.n_expert_used,   // 0 = the model's top-k (was 8)
                max_waves > 0 ? (uint32_t) max_waves : 0,
                ram_budget,
                oracle ? oracle : "",
                oracle_lead >= 0 ? (uint32_t) oracle_lead : 1,
                /* chunk_demand = */ 0,
                /* chunk_prefetch = */ 0,   // the plan only reads, it never issues a read
                /* gpu_waves = */ 0,
                // A list or an explicit count places whole layers in VRAM the way the real load does (the constructor
                // clears their sidecar entries before the slot budget is derived), so the pools cover the streamed layers
                // only. The auto-fit (-1) needs the free VRAM, which a plan does not have: it counts every MoE layer there.
                gpu_layers > 0 ? (uint32_t) gpu_layers : 0,
                gpu_layers_list ? gpu_layers_list : "",
                /* gpu_layers_auto = */ false,
                /* vram_reserve = */ 0,
                gpu_layers_mode && gpu_layers_mode[0] ? gpu_layers_mode : "hot",
                ram_free,
                // the plan has no buffer list, so it cannot walk the model's weights; it uses the same
                // allowance the engine adds on top of them
                /* other_bytes = */ plan_np_bytes + plan_ram_weights + plan_record_bytes + plan_corr_bytes +   // + the correction file
                                    // The guess helper's slot buffer, only with --moe-stream-needed-early 1,
                                    // at its real size (as the load counts it: the engine default adds nothing).
                                    // The combined step's stacked gains / biases are not here: the plan keeps every block's
                                    // weights on the GPU, and so would they be (they sit on the routers' buffer type)
                                    (needed_early == 1 ? llama_moe_stream_ne_bytes((int64_t) mi.n_layer_all,
                                            lookahead > 1 ? lookahead : 1,
                                            guess_top > 0 ? guess_top : (prefetch_m > 0 ? prefetch_m : (int32_t) mi.n_expert_used)) : 0) +
                                    // + the GPU producer's pinned buffer, as the load counts it
                                    (needed_early == 1 ? llama_moe_stream_ne2_bytes((int64_t) mi.n_layer_all,
                                            lookahead > 1 ? lookahead : 1,
                                            guess_top > 0 ? guess_top : (prefetch_m > 0 ? prefetch_m : (int32_t) mi.n_expert_used),
                                            (int64_t) mi.n_embd, (int64_t) mi.n_expert_used,
                                            ps_rows > 0 ? (int64_t) ps_rows : 0) : 0) +   // + the shared slot, as the load
                                    // The wave plan's / slot check's shared tables, as the load counts them
                                    llama_moe_stream_wp_bytes((int64_t) mi.n_ctx_train, (int64_t) mi.n_expert_used, (int64_t) mi.n_expert),
                /* verify      = */ false,     // a plan reads nothing
                /* cpu_repack  = */ cpu_repack,// a plan allocates no pools, but the repack's layout check is in there
                /* hugepages   = */ hugepages,  // a plan allocates nothing, but the option decides whether the zero slices are charged
                /* pin         = */ pin,
                /* policy      = */ "lru",     // the plan does not evict
                /* pred_shared = */ false,    // the plan builds no graph
                /* prefetch_lead = */ 1, /* prefetch_m2 = */ 0,
                /* demand_last = */ false, /* probation = */ false,   // no reads in a plan
                /* prefetch_inflight = */ 0,   // a plan issues no reads
                /* early_order = */ false,     // a plan builds no graph
                /* overlap     = */ false,    // the plan builds no graph and runs no matmul
                /* pred_recall = */ false,     // a plan counts nothing
                /* pred_scores = */ "",      // a plan writes nothing
                /* pred_input  = */ "",       // a plan records nothing
                /* pred_input_every = */ 0,
                /* record_dir  = */ "",         // a plan records nothing
                /* record_words = */ 0);

        // the plan holds what the engine would use, with the same defaults as the engine
        ms.plan().io_threads   = io_threads > 0 ? (uint32_t) io_threads : 8;
        ms.plan().max_waves     = max_waves > 0 ? max_waves : 0;

        const moe_stream_plan_in & p = ms.plan();

        if (out) {
            *out = {};

            snprintf(out->model_path, sizeof(out->model_path), "%s", p.model_path.c_str());
            snprintf(out->sidecar,    sizeof(out->sidecar),    "%s", p.sidecar.c_str());
            snprintf(out->arch,       sizeof(out->arch),       "%s", p.arch.c_str());
            out->n_layer_all   = (int32_t) p.n_layer_all;
            out->n_layer_nextn = (int32_t) p.n_layer_nextn;
            out->n_layer_dense = (int32_t) p.n_layer_dense;
            out->n_streamed    = p.n_streamed;
            out->n_gpu_layers  = (int32_t) p.gpu_layers.size();
            for (size_t i = 0; i < p.gpu_layers.size() && i < sizeof(out->gpu_layers)/sizeof(out->gpu_layers[0]); i++) {
                out->gpu_layers[i] = p.gpu_layers[i];
            }
            out->n_expert      = (int32_t) p.n_expert;
            out->n_expert_used = (int32_t) p.n_expert_used;
            out->block_bytes   = (int64_t) p.block_bytes;
            out->n_slots       = (int32_t) p.n_slots;
            out->ram_budget    = p.ram_budget;
            out->ram_free      = p.ram_free;
            out->other_bytes   = p.other_bytes;
            out->pool_bytes    = p.pool_bytes;
            out->mem_available = p.mem_available;
            out->io_threads    = (int32_t) p.io_threads;
            out->prefetch      = p.prefetch;
            out->prefetch_m    = (int32_t) p.prefetch_m;
            out->max_waves     = (int32_t) p.max_waves;
            snprintf(out->oracle, sizeof(out->oracle), "%s", p.oracle.c_str());
            out->oracle_lead   = (int32_t) p.oracle_lead;
        }

        return true;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return false;
    }
}

bool llama_model::has_tensor_overrides() const {
    return pimpl->has_tensor_overrides;
}

const ggml_tensor * llama_model::get_tensor(const char * name) const {
    auto it = std::find_if(tensors_by_name.begin(), tensors_by_name.end(),
            [name](const std::pair<std::string, ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}

float llama_model::get_rope_freq_base (const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
}

float llama_model::get_rope_freq_scale(const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
}

ggml_tensor * llama_model::get_rope_factors(const llama_cparams & cparams, int il) const {
    const uint32_t n_ctx_seq = cparams.n_ctx_seq;

    // choose long/short freq factors based on the context size
    if (layers[il].rope_freqs != nullptr) {
        return layers[il].rope_freqs;
    }

    if (n_ctx_seq > hparams.n_ctx_orig_yarn) {
        return layers[il].rope_long;
    }

    return layers[il].rope_short;
}

llama_memory_i * llama_model::create_memory(const llama_memory_params & params, const llama_cparams & cparams) const {
    llama_memory_i * res;

    switch (arch) {
        // Models that need specific instantiation should be handled in the
        // switch statement
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_EUROBERT:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
            {
                res = nullptr;
            } break;
        case LLM_ARCH_MINIMAX_M3:
            {
                // sparse (MSA) layers carry an indexer key cache, but leading dense layers do not
                llama_kv_cache::layer_filter_cb filter_idx =
                    [&](int32_t il) { return (uint32_t) il >= hparams.n_layer_dense_lead; };

                res = new llama_kv_cache_msa(
                        *this,
                        params.no_alloc,
                        params.type_k,
                        params.type_v,
                        !cparams.flash_attn,
                        cparams.offload_kqv,
                        cparams.kv_unified,
                        cparams.n_ctx_seq,
                        cparams.n_seq_max,
                        hparams.n_swa,
                        hparams.swa_type,
                        nullptr,
                        filter_idx,
                        nullptr);
            } break;
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_DEEPSEEK32:
            {
                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && hparams.n_layer_nextn > 0) {
                    // The NextN/MTP draft head runs dense MLA (no DSA indexer), so the
                    // MTP context uses a plain attention KV cache holding only the
                    // nextn layer(s) - same pattern as the hybrid Qwen3.5 MTP context.
                    llama_kv_cache::layer_filter_cb filter =
                        [&](uint32_t il) { return il >= hparams.n_layer(); };

                    res = new llama_kv_cache(
                            *this,
                            params.no_alloc,
                            hparams,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            hparams.n_swa,
                            hparams.swa_type,
                            nullptr,
                            filter,
                            nullptr,
                            nullptr);
                } else {
                    // Main context: DSA cache for the trunk layers only - the nextn
                    // layer(s) are never attended by the trunk graph.
                    llama_kv_cache::layer_filter_cb filter_mla = nullptr;
                    if (hparams.n_layer_nextn > 0) {
                        filter_mla = [&](uint32_t il) { return il < hparams.n_layer(); };
                    }
                    llama_kv_cache::layer_filter_cb filter_lid = [&](uint32_t il) { return il < hparams.n_layer() && (arch != LLM_ARCH_GLM_DSA || hparams.is_indexer_full(il)); };

                    res = new llama_kv_cache_dsa(
                            *this,
                            params.no_alloc,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            hparams.n_swa,
                            hparams.swa_type,
                            filter_mla,
                            filter_lid,
                            nullptr);
                }
            } break;
        case LLM_ARCH_HY_V4:
            {
                if (hparams.indexer_top_k == 0) {
                    // full-attention checkpoint: no indexer, so no indexer key cache
                    res = new llama_kv_cache(
                            *this,
                            params.no_alloc,
                            hparams,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            hparams.n_swa,
                            hparams.swa_type,
                            nullptr,
                            nullptr,
                            nullptr,
                            nullptr);
                } else {
                    // only "full" layers own an indexer, so the shared layers need no indexer cache
                    llama_kv_cache::layer_filter_cb filter_lid = [&](uint32_t il) { return hparams.is_indexer_full(il); };

                    res = new llama_kv_cache_dsa(
                            *this,
                            params.no_alloc,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            hparams.n_swa,
                            hparams.swa_type,
                            nullptr,
                            filter_lid,
                            nullptr);
                }
            } break;
        case LLM_ARCH_DOTS3NOTE:
            {
                GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && hparams.n_layer_nextn > 0) {
                    // MTP draft context: plain attention KV cache holding only the nextn layer
                    llama_kv_cache::layer_filter_cb filter =
                        [&](uint32_t il) { return il >= hparams.n_layer(); };

                    res = new llama_kv_cache(
                            *this,
                            params.no_alloc,
                            hparams,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            hparams.n_swa,
                            hparams.swa_type,
                            nullptr,
                            filter,
                            nullptr,
                            nullptr);
                } else {
                    // main context: DSA cache for the trunk full-attention layers plus a window-sized SWA cache
                    llama_kv_cache::layer_filter_cb filter_mla = nullptr;
                    if (hparams.n_layer_nextn > 0) {
                        filter_mla = [&](uint32_t il) { return il < hparams.n_layer(); };
                    }
                    llama_kv_cache::layer_filter_cb filter_lid = [&](uint32_t il) { return il < hparams.n_layer() && hparams.is_indexer_full(il); };

                    res = new llama_kv_cache_dsa_iswa(
                            *this,
                            params.no_alloc,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            filter_mla,
                            filter_lid,
                            nullptr);
                }
            } break;
        case LLM_ARCH_DEEPSEEK4:
            {
                GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

                if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                    const llama_memory_i::layer_filter_cb filter_mtp = [&](int32_t il) {
                        return il >= (int32_t) hparams.n_layer();
                    };

                    res = new llama_kv_cache_iswa(
                            *this,
                            params.no_alloc,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            nullptr,
                            filter_mtp,
                            nullptr,
                            nullptr);
                } else {
                    res = new llama_kv_cache_dsv4(
                            *this,
                            params.no_alloc,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            cparams.n_rs_seq,
                            nullptr,
                            nullptr);
                }
            } break;
        case LLM_ARCH_DFLASH:
            {
                // DSV4 DSpark stages store a single MLA-style K per position (window = the draft ring)
                if (hparams.dsv4_hc_mult > 0) {
                    GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE);

                    res = new llama_kv_cache_iswa(
                            *this,
                            params.no_alloc,
                            params.type_k,
                            params.type_v,
                            !cparams.flash_attn,
                            cparams.offload_kqv,
                            params.swa_full,
                            cparams.kv_unified,
                            cparams.n_ctx_seq,
                            cparams.n_seq_max,
                            cparams.n_ubatch,
                            nullptr,
                            nullptr,
                            nullptr,
                            nullptr);
                    break;
                }
            }
            [[fallthrough]];
        // Models that need standard caching should rely on recurrent/hybrid
        // checks
        default:
            {
                // Dense MTP heads use a plain attention KV cache instead of the hybrid wrapper.
                const bool mtp_on_hybrid_qwen =
                    params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
                    (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ||
                     arch == LLM_ARCH_BAILINGMOE3);

                const bool mtp_on_hybrid_nemotron =
                    params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && arch == LLM_ARCH_NEMOTRON_H_MOE;

                if (llm_arch_is_recurrent(arch)) {
                    res = new llama_memory_recurrent(
                            *this,
                            params.no_alloc,
                            GGML_TYPE_F32,
                            GGML_TYPE_F32,
                            cparams.offload_kqv,
                            std::max((uint32_t) 1, cparams.n_seq_max),
                            cparams.n_seq_max,
                            cparams.n_rs_seq,
                            nullptr);
                } else if (llm_arch_is_hybrid(arch) && !mtp_on_hybrid_qwen && !mtp_on_hybrid_nemotron) {
                    // The main difference between hybrid architectures is the
                    // layer filters, so pick the right one here
                    llama_memory_hybrid::layer_filter_cb filter_attn = nullptr;
                    llama_memory_hybrid::layer_filter_cb filter_recr = nullptr;
                    // only the sparse-attention architectures use llama_memory_hybrid_idx
                    // a null filter_idx means the GGUF has no indexer tensors
                    llama_memory_hybrid::layer_filter_cb filter_idx  = nullptr;
                    const bool needs_mem_idx = (arch == LLM_ARCH_QWEN4EXP);
                    if (arch == LLM_ARCH_FALCON_H1 || arch == LLM_ARCH_INKLING2) {
                        filter_attn = [&](uint32_t) { return true; };
                        filter_recr = [&](uint32_t) { return true; };
                    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
                        filter_attn = [&](uint32_t il) {
                            return !hparams.is_recr(il) && hparams.n_ff(il) == 0;
                        };
                        filter_recr = [&](uint32_t il) {
                            return hparams.is_recr(il) && hparams.n_ff(il) == 0;
                        };
                    } else if (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_MINIMAX_01) {
                        filter_attn = [&](uint32_t il) {
                            return il < hparams.n_layer() && !hparams.is_recr(il);
                        };
                        filter_recr = [&](uint32_t il) {
                            return il < hparams.n_layer() && hparams.is_recr(il);
                        };

                        if (arch == LLM_ARCH_QWEN4EXP && hparams.indexer_head_size > 0) {
                            // QSA runs on the dense-attention layers only
                            filter_idx = [&](uint32_t il) {
                                return il < hparams.n_layer() && !hparams.is_recr(il);
                            };
                        }

                        // the MTP draft context holds the MTP block alone: its attention and indexer, no recurrent layer
                        if (arch == LLM_ARCH_QWEN4EXP && params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                            filter_attn = [&](uint32_t il) { return il >= hparams.n_layer(); };
                            filter_recr = [&](uint32_t)    { return false; };
                            if (filter_idx) {
                                filter_idx = [&](uint32_t il) { return il >= hparams.n_layer(); };
                            }
                        }
                    }

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        const ggml_type eff_tk = (arch == LLM_ARCH_INKLING2 && params.type_k == GGML_TYPE_F16) ? GGML_TYPE_BF16 : params.type_k;
                        const ggml_type eff_tv = (arch == LLM_ARCH_INKLING2 && params.type_v == GGML_TYPE_F16) ? GGML_TYPE_BF16 : params.type_v;
                        // Use hybrid-iswa for hybrid models with SWA
                        res = new llama_memory_hybrid_iswa(
                            /* model             */ *this,
                            /* no_alloc          */ params.no_alloc,
                            /* attn_type_k       */ eff_tk,
                            /* attn_type_v       */ eff_tv,
                            /* attn_v_trans      */ (arch == LLM_ARCH_INKLING2 ? false : !cparams.flash_attn),
                            /* attn_swa_full     */ params.swa_full,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_ubatch     */ cparams.n_ubatch,
                            /* recurrent_type_r  */ GGML_TYPE_F32,
                            /* recurrent_type_s  */ GGML_TYPE_F32,
                            /* recurrent_rs_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* n_rs_seq          */ cparams.n_rs_seq,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr));
                    } else if (needs_mem_idx) {
                        // sparse attention over a per-token indexer cache, in its own memory type
                        res = new llama_memory_hybrid_idx(
                            /* model             */ *this,
                            /* no_alloc          */ params.no_alloc,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_swa        */ hparams.n_swa,
                            /* attn_swa_type     */ hparams.swa_type,
                            /* recurrent_type_k  */ GGML_TYPE_F32,
                            /* recurrent_type_v  */ GGML_TYPE_F32,
                            /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* n_rs_seq          */ cparams.n_rs_seq,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr),
                            /* filter_idx        */ std::move(filter_idx));
                    } else {
                        res = new llama_memory_hybrid(
                            /* model             */ *this,
                            /* no_alloc          */ params.no_alloc,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_swa        */ hparams.n_swa,
                            /* attn_swa_type     */ hparams.swa_type,
                            /* recurrent_type_k  */ GGML_TYPE_F32,
                            /* recurrent_type_v  */ GGML_TYPE_F32,
                            /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* n_rs_seq          */ cparams.n_rs_seq,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr));
                    }
                } else {
                    llama_kv_cache::layer_filter_cb filter = nullptr;
                    llama_memory_i::layer_reuse_cb reuse = nullptr;
                    llama_kv_cache::layer_share_cb share = nullptr;

                    if (arch == LLM_ARCH_GEMMA3N || arch == LLM_ARCH_GEMMA4) {
                        reuse = [&](uint32_t il) {
                            GGML_ASSERT(hparams.n_layer_kv_from_start >= 2);

                            if (il >= (uint32_t)hparams.n_layer_kv_from_start) {
                                return hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
                            }

                            return -1;
                        };
                    }

                    if (mtp_on_hybrid_qwen || mtp_on_hybrid_nemotron) {
                        filter = [&](uint32_t il) { return il >= hparams.n_layer(); };
                    }

                    // don't filter when n_layer_nextn is repurposed for a router layer the trunk attends
                    // or when a model is entirely n_layer_nextn layers and has no trunk
                    if (hparams.n_layer_nextn > 0 && hparams.n_layer() > 0 && hparams.router_layer < 0) {
                        if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
                            filter = [&](uint32_t il) { return il >= hparams.n_layer(); };
                        } else {
                            filter = [&](uint32_t il) { return il <  hparams.n_layer(); };
                        }
                    }

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        GGML_ASSERT(hparams.is_swa_any());

                        if (arch == LLM_ARCH_GEMMA4_ASSISTANT) {
                            llama_memory_t mem_other = llama_get_memory(cparams.ctx_other);

                            share = [&](int32_t il) {
                                const llama_model * model_other = llama_get_model(cparams.ctx_other);

                                if (hparams.is_swa(il)) {
                                    return llama_model_n_layer(model_other) - 2;
                                }

                                return llama_model_n_layer(model_other) - 1;
                            };

                            res = new llama_kv_cache_iswa(
                                    *this,
                                    params.no_alloc,
                                    params.type_k,
                                    params.type_v,
                                    !cparams.flash_attn,
                                    cparams.offload_kqv,
                                    params.swa_full,
                                    cparams.kv_unified,
                                    cparams.n_ctx_seq,
                                    cparams.n_seq_max,
                                    cparams.n_ubatch,
                                    mem_other,
                                    filter,
                                    reuse,
                                    share);
                        } else {
                            res = new llama_kv_cache_iswa(
                                    *this,
                                    params.no_alloc,
                                    params.type_k,
                                    params.type_v,
                                    !cparams.flash_attn,
                                    cparams.offload_kqv,
                                    params.swa_full,
                                    cparams.kv_unified,
                                    cparams.n_ctx_seq,
                                    cparams.n_seq_max,
                                    cparams.n_ubatch,
                                    nullptr,
                                    filter,
                                    reuse,
                                    share);
                        }
                    } else {
                        GGML_ASSERT(!hparams.is_swa_any());

                        res = new llama_kv_cache(
                                *this,
                                params.no_alloc,
                                hparams,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                cparams.kv_unified,
                                cparams.n_ctx_seq,
                                cparams.n_seq_max,
                                hparams.n_swa,
                                hparams.swa_type,
                                nullptr,
                                filter,
                                nullptr,
                                nullptr);
                    }
                }
            }
    }

    return res;
}

ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b, cls_norm);

    // add backend sampling layers (if any)
    llm->build_sampling();

    // if the gguf model was converted with --sentence-transformers-dense-modules
    // there will be two additional dense projection layers
    // dense linear projections are applied after pooling
    // TODO: move reranking logic here and generalize
    llm->build_dense_out(dense_2_out_layers, dense_2_out_layers_b, dense_3_out_layers);

    llm->res->set_outputs(params);

    return llm->res->get_gf();
}


//
// interface implementation
//

llama_model_params llama_model_default_params() {
    llama_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ -1,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.load_mode                   =*/ LLAMA_LOAD_MODE_AUTO,
        /*.lazy_mode                   =*/ LLAMA_LAZY_MODE_AUTO,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.moe_stream                  =*/ nullptr,
        /*.moe_stream_slots            =*/ 0,
        /*.moe_stream_log              =*/ nullptr,
        /*.moe_stream_io_threads       =*/ 0,
        /*.moe_stream_gpu_waves       =*/ 512,    // the default, 0 = host matmuls for wave graphs
        /*.moe_stream_gpu_layers      =*/ -1,     // the default, negative = fit whole layers to the VRAM
        /*.moe_stream_vram_reserve    =*/ (int64_t) (3.15 * 1024 * 1024 * 1024),   // 2 GiB, +1.15 GiB since the honest VRAM count
        /*.moe_stream_gpu_layers_mode =*/ nullptr,
        /*.moe_stream_gpu_layers_list =*/ nullptr,
        /*.moe_stream_ram_free      =*/ 0,
        /*.moe_stream_verify        =*/ 1,      // the read-back check is on by default
        /*.moe_stream_cpu_repack    =*/ false,  // the repacked CPU layout for the pools; off by default
        /*.moe_stream_chunk            =*/ 0,
        /*.moe_stream_chunk_prefetch   =*/ 0,
        /*.moe_stream_prefetch         =*/ -1,
        /*.moe_stream_prefetch_m       =*/ 0,
        /*.moe_stream_max_waves        =*/ 0,
        /*.moe_stream_hugepages        =*/ 0,   // off until measured
        /*.moe_stream_pin              =*/ 0,   // off until the pin test passes
        /*.moe_stream_policy           =*/ nullptr,  // "lru"
        /*.moe_stream_pred_shared      =*/ 0,   // off by default (adds a device wait)
        /*.moe_stream_pred_recall      =*/ 0,   // no recall counter
        /*.moe_stream_pred_scores      =*/ nullptr, // no score file
        /*.moe_stream_pred_input       =*/ nullptr, // no router-input recording
        /*.moe_stream_pred_input_every =*/ 0,      // 1 word in 8
        /*.moe_stream_record           =*/ nullptr, // no side-by-side recording
        /*.moe_stream_record_words     =*/ 0,      // the default cap
        /*.moe_stream_prefetch_lead    =*/ 1,   // one layer of lead, the engine as it was
        /*.moe_stream_prefetch_m2      =*/ 0,   // the correction's width follows prefetch_m
        /*.moe_stream_demand_last      =*/ 0,   // a demand read overtakes a prefetch (as it does)
        /*.moe_stream_probation        =*/ 0,   // uncovered prefetches are protected, not evicted
        /*.moe_stream_prefetch_inflight=*/ 0,   // no cap on the guessed reads in flight
        /*.moe_stream_early_order     =*/ 0,   // the graph is ordered as it was
        /*.moe_stream_overlap          =*/ 0,   // off until measured to win and stay bit-identical
        /*.moe_stream_ubatch           =*/ 512, // the -ub the VRAM reserve is sized for
        /*.moe_stream_ctx              =*/ 0,   // the training context
        /*.moe_stream_ctx_seq          =*/ 1,
        /*.moe_stream_ctx_type_k       =*/ GGML_TYPE_F16,
        /*.moe_stream_ctx_type_v       =*/ GGML_TYPE_F16,
        /*.moe_stream_ctx_rs_seq       =*/ 0,
        /*.moe_stream_ctx_flags        =*/ 4,   // the KV cache on the layers' devices
        /*.moe_stream_ram              =*/ 0,
        /*.moe_stream_plan_only        =*/ 0,
        /*.moe_stream_oracle           =*/ nullptr,
        /*.moe_stream_oracle_lead      =*/ -1,
        /*.moe_stream_io_wake          =*/ 0,
        /*.moe_stream_promote          =*/ 0,
        /*.moe_stream_guess_cap        =*/ -1,
        /*.moe_stream_guess_cap_refill =*/ 0,
        /*.moe_stream_evict_halflife   =*/ 0.0f,
        /*.moe_stream_evict_credit     =*/ 0.0f,
        /*.moe_stream_prompt_end       =*/ 0,
        /*.moe_stream_prompt_end_halflife =*/ 0.0f,
        /*.moe_stream_prompt_values    =*/ 0,
        /*.moe_stream_chance_model     =*/ nullptr,
        /*.moe_stream_read_threshold   =*/ 0.0f,
        /*.moe_stream_cpu_skip_zero    =*/ 0,   // the zero slices are computed, as before
        /*.moe_stream_cpu_x4           =*/ 0,   // one row per call, as before
        /*.moe_stream_cpu_x4_pf        =*/ -1,  // the next 4-row group (used only with the 4-row kernel)
        /*.moe_stream_cpu_fuse         =*/ 0,   // separate gate, up and swiglu ops, as before
        /*.moe_stream_cpu_dispatch     =*/ 0,   // mul_mat_id's own split, as before
        /*.moe_stream_cpu_pieces       =*/ 1,   // one work piece per thread and expert, as before
        /*.moe_stream_cpu_conv1        =*/ 0,   // the activation row split over the threads, as before
        /*.moe_stream_cpu_wsum         =*/ 0,   // the mul and the add chain, as before
        /*.moe_stream_cpu_state_inplace =*/ 0,  // the state copied before and after the delta step, as before
        /*.moe_stream_cpu_mmid_merge   =*/ 0,   // one window per 64-row group, as before
        /*.moe_stream_cpu_iqk          =*/ 0,   // our own expert kernels, as before
        /*.moe_stream_guess_skip_full  =*/ 0,   // every guess runs, as before
        /*.moe_stream_guess_skip_full_gpu =*/ 0,   // the GPU's guess as before
        /*.moe_stream_cpu_fuse_small   =*/ 0,   // one step per node, as before
        /*.moe_stream_cpu_fa           =*/ 0,   // the plain CPU flash attention
        /*.moe_stream_cpu_q8_tiled     =*/ 0,   // q8_0 on its own (llamafile / vec_dot) path, as before
        /*.moe_stream_cpu_topk         =*/ 0,   // the full argsort, as before
        /*.moe_stream_cpu_down_scale   =*/ 0,   // the down-input scale step everywhere, as before
        /*.moe_stream_cpu_small_ops    =*/ 0,   // the full sort and the plane split, as before
        /*.moe_stream_fused_route      =*/ 0,   // the argsort chain, as before
        /*.moe_stream_slot_check       =*/ 1,   // on
        /*.moe_stream_wave_plan        =*/ 0,   // off
        /*.moe_stream_wave_victim      =*/ -1,  // auto (on with prefill_fast)
        /*.moe_stream_conv_silu        =*/ 0,   // SiLU after the reshape, as before
        /*.moe_stream_hc_share         =*/ 0,   // every combine computes its own weights, as before
        /*.moe_stream_rs_window        =*/ 0,   // the state fetch and the conv-state write where the layer needs them, as before
        /*.moe_stream_ple_late         =*/ 0,   // the n-gram rows gathered first, with the token row, as before
        /*.moe_stream_hc_silu          =*/ 0,   // the mix's scale and SiLU as two kernels, as before
        /*.moe_stream_hc_q8            =*/ 0,   // the mix's matvecs quantize their inputs, as before
        /*.moe_stream_rs_inplace       =*/ 0,   // the delta step reads a gathered copy of its state, as before
        /*.moe_stream_gpu_share_q8     =*/ 0,   // every matvec quantizes its input, as before
        /*.moe_stream_vram_experts     =*/ 0,   // off
        /*.moe_stream_vram_experts_file   =*/ nullptr,
        /*.moe_stream_vram_experts_margin =*/ 0,   // 512 MiB
        /*.moe_stream_vram_free        =*/ 0,   // off (the pinned N decides)
        /*.moe_stream_vram_scratch     =*/ -1,  // derived from the model
        /*.moe_stream_vram_growth      =*/ 0,   // derived
        /*.moe_stream_correction       =*/ nullptr,   // the plain guess
        /*.moe_stream_idle_load        =*/ 0,         // off
        /*.moe_stream_usage_file       =*/ nullptr,   // next to the sidecar
        /*.moe_stream_idle_pause_ms    =*/ 0,         // 500 ms
        /*.moe_stream_read_ahead       =*/ 0,         // off
        /*.moe_stream_read_ahead_min   =*/ 0,         // 4096 tokens
        /*.moe_stream_window           =*/ 0,         // off
        /*.moe_stream_window_keep      =*/ 0.0f,      // 1
        /*.moe_stream_prompt_defer     =*/ 0,         // off
        /*.moe_stream_prompt_split     =*/ 0,         // off
        /*.moe_stream_prompt_split_places =*/ 128,
        /*.moe_stream_gpu_wave_skip_zero =*/ 0,       // off
        /*.moe_stream_gpu_skip_zero    =*/ 0,   // off
        /*.moe_stream_plan             =*/ 0,   // the plain guess path
        /*.moe_stream_plan_bind_early  =*/ 0,   // test only
        /*.moe_stream_drop_stale       =*/ 0,   // off
        /*.moe_stream_needed_early     =*/ 0,   // off
        /*.moe_stream_behind_backlog   =*/ 0,   // off
        /*.moe_stream_behind_rank      =*/ 0,   // off
        /*.moe_stream_guess_inflight   =*/ 0,   // no limit
        /*.moe_stream_nonpool          =*/ nullptr,   // the fixed reserve
        /*.moe_stream_srv              =*/ nullptr,   // not a server
        /*.moe_stream_guess_top        =*/ 0,   // off (the guess width is prefetch_m)
        /*.moe_stream_lookahead        =*/ 2,   // the layers 1..2 ahead
        /*.moe_stream_guess_pace       =*/ 1,
        /*.moe_stream_plan_old_order   =*/ 0,   // test only
        /*.moe_stream_plan_extra_near  =*/ 0,   // test only
        /*.moe_stream_guess_combined   =*/ 0,   // the separate ops
        /*.moe_stream_protect_named    =*/ 0,   // off
        /*.moe_stream_wordstart        =*/ 0,   // off
        /*.moe_stream_proj_ready       =*/ 0,   // off
        /*.moe_stream_vram_follow      =*/ 0,   // off (the VRAM experts stay put)
        /*.moe_stream_vram_follow_tick =*/ 8,
        /*.moe_stream_vram_follow_swaps =*/ 64,
        /*.moe_stream_vram_follow_recent =*/ 256,
        /*.moe_stream_vram_follow_longterm =*/ 1024,
        /*.moe_stream_vram_follow_ratio =*/ 1.25f,
        /*.moe_stream_vram_follow_margin =*/ 4,
        /*.moe_stream_vram_follow_fixed =*/ 0,
        /*.moe_stream_vram_follow_pinned =*/ 1,
        /*.moe_stream_vram_parallel =*/ 0,   // off (the device pass after the CPU experts)
        /*.moe_stream_vram_copyback =*/ 0,   // off (an expert leaving VRAM follow's set is dropped)
        /*.moe_stream_async_h2d     =*/ 0,   // off (CPU -> GPU split inputs copied synchronously)
        /*.moe_stream_elastic       =*/ 0,         // fixed pools, as before
        /*.moe_stream_ram_margin    =*/ 0,         // 500 MiB when elastic
        /*.moe_stream_elastic_opts  =*/ nullptr,
        /*.moe_stream_shares        =*/ 0,         // equal shares, as before
        /*.moe_stream_share_file    =*/ nullptr,
        /*.moe_stream_share_opts    =*/ nullptr,
        /*.moe_stream_layer_major   =*/ 0,         // off (one graph per ubatch)
        /*.moe_stream_layer_major_batch =*/ 0,     // the raise goes up to the context size
        /*.moe_stream_vram_end_fill =*/ 0,         // off (the tail refilled at the first reply word)
        /*.moe_stream_copy_ahead    =*/ 0,         // off (no copy ahead; one-wave prompts take the scheduler's pool copy)
        /*.moe_stream_tail_wave_w   =*/ 0,         // off (plain wave width)
        /*.moe_stream_vram_lend_prompt =*/ 0,      // off (the prompt's GPU tensors in the compute buffer)
        /*.moe_stream_vram_lend_short  =*/ 64,     // prompt calls of <= 64 tokens in the real compute buffer
        /*.moe_stream_timing           =*/ 0,      // no timing work
        /*.moe_stream_out_check        =*/ 16,     // every prompt call and every 16th reply call
        /*.moe_stream_vram_device      =*/ nullptr, // the first GPU holds the VRAM pool
        /*.moe_stream_gpu_layers_device =*/ nullptr, // whole layers' experts on their layer's own device
        /*.moe_stream_vram_free_dev    =*/ nullptr, // moe_stream_vram_free is the pool's GPU's only
        /*.quant_plugin                =*/ nullptr,
        /*.moe_stream_vram_chunk       =*/ 0,      // off (one buffer group per format)
        /*.moe_stream_uma              =*/ 0,      // off (discrete RAM / VRAM pools)
        /*.moe_stream_hip_mmvq         =*/ 0,      // off
        /*.moe_stream_hip_sync         =*/ 0,      // off
        /*.moe_stream_prefill_fast     =*/ 0,      // off
        /*.moe_stream_mmq_moe_opt      =*/ 0,      // off
        /*.moe_stream_gpu_moe_fused    =*/ 0,      // off
        /*.moe_stream_pinned_ring      =*/ 0,      // off
        /*.moe_stream_lm_bf16          =*/ 0,      // off
        /*.moe_stream_lend_output      =*/ 0,      // off
        /*.moe_stream_qsa_fused        =*/ 0,      // off
        /*.moe_stream_hc_fused         =*/ 0,      // off
        /*.moe_stream_ple_fast         =*/ 0,      // off
        /*.moe_stream_gpu_layers_floor        =*/ 2500LL << 20, // 2500 MiB pool floor
        /*.moe_stream_gpu_layers_lend_reclaim =*/ 640LL << 20,  // 640 MiB max prompt-lend reclaim
        /*.moe_stream_copy_ahead_one_below    =*/ 10LL << 30,   // 10 GiB single-set threshold
        /*.moe_stream_gpu_layers_cap          =*/ 0.52f,        // 52 % max whole-layer fraction
        /*.moe_stream_gpu_layers_uniform_max  =*/ 0,            // bounded by cap & floor
        /*.moe_stream_auto_mask        =*/ 0,      // no umbrella auto-enabled bits
        /*.vocab_only                  =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
        /*.no_host                     =*/ false,
        /*.no_alloc                    =*/ false,
        /*.load_mtp                    =*/ false,
    };

    return result;
}

const llama_vocab * llama_model_get_vocab(const llama_model * model) {
    return &model->vocab;
}

void llama_free_model(llama_model * model) {
    llama_model_free(model);
}

void llama_model_free(llama_model * model) {
    delete model;
}

int32_t llama_model_n_ctx_train(const llama_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t llama_model_n_embd(const llama_model * model) {
    return model->hparams.n_embd;
}

int32_t llama_model_n_embd_inp(const llama_model * model) {
    return model->hparams.n_embd_inp();
}

int32_t llama_model_n_embd_out(const llama_model * model) {
    return model->hparams.n_embd_out();
}

int32_t llama_model_n_layer(const llama_model * model) {
    return model->hparams.n_layer();
}

int32_t llama_model_n_layer_nextn(const llama_model * model) {
    return model->hparams.n_layer_nextn;
}

int32_t llama_model_dflash_selector_top_k(const llama_model * model) {
    return model->hparams.dflash_selector_top_k;
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

int32_t llama_model_n_head_kv(const llama_model * model) {
    return model->hparams.n_head_kv();
}

int32_t llama_model_n_swa(const llama_model * model) {
    // dsv4 kv-cache has SWA but it cannot be used as a rollback because of
    // other compression ratios, so we return 0 here
    if (model->arch == LLM_ARCH_DEEPSEEK4) {
        return 0;
    }
    return model->hparams.n_swa;
}


uint32_t llama_model_n_cls_out(const struct llama_model * model) {
    return model->hparams.n_cls_out;
}

const char * llama_model_cls_label(const struct llama_model * model, uint32_t i) {
    if (i < model->classifier_labels.size()) {
        return model->classifier_labels[i].c_str();
    }

    return nullptr;
}

// deprecated
int32_t llama_n_ctx_train(const llama_model * model) {
    return llama_model_n_ctx_train(model);
}

// deprecated
int32_t llama_n_embd(const llama_model * model) {
    return llama_model_n_embd(model);
}

// deprecated
int32_t llama_n_layer(const llama_model * model) {
    return llama_model_n_layer(model);
}

// deprecated
int32_t llama_n_head(const llama_model * model) {
    return llama_model_n_head(model);
}

llama_rope_type llama_model_rope_type(const llama_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_CLIP:
        case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ:
        case LLM_ARCH_MPT:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_BLOOM:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_MAMBA2:
        case LLM_ARCH_JAMBA:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_JAIS:
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_RWKV6QWEN2:
        case LLM_ARCH_RWKV7:
        case LLM_ARCH_ARWKV7:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_NEMOTRON_H:
        case LLM_ARCH_NEMOTRON_H_MOE:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_INKLING2:
            return LLAMA_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_DECI:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_XVERSE:
        case LLM_ARCH_COMMAND_R:
        case LLM_ARCH_COHERE2:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_OLMO:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK2OCR:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_MUSE_GLIMMER:
        case LLM_ARCH_PLM:
        case LLM_ARCH_CHATGLM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_GRANITE_HYBRID:
        case LLM_ARCH_GRANITE_SWITCH:
        case LLM_ARCH_GRANITE_SWA:
        case LLM_ARCH_CHAMELEON:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_SMOLLM3:
        case LLM_ARCH_ARCEE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_EAGLE3:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_MAINCODER:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_NANBEIGE:
        case LLM_ARCH_POCKETTTS:
            return LLAMA_ROPE_TYPE_NORM;
        case LLM_ARCH_BAILINGMOE3:
            // VL files carry mrope sections; text-only files keep NORM rope
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NORM;
        // HY_V4 rotates consecutive pairs, matching the reference implementation
        case LLM_ARCH_HY_V4:
            return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_FALCON_H1:
        case LLM_ARCH_GROK:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_EUROBERT:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_FLINT:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_PLAMO2:
        case LLM_ARCH_PLAMO3:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_GEMMA3:
        case LLM_ARCH_GEMMA3N:
        case LLM_ARCH_GEMMA4:
        case LLM_ARCH_GEMMA4_ASSISTANT:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_ORION:
        case LLM_ARCH_NEMOTRON:
        case LLM_ARCH_EXAONE:
        case LLM_ARCH_EXAONE4:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_MINICPM3:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_JAIS2:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_HUNYUAN_DENSE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_LFM2:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_SEED_OSS:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_APERTUS:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_COGVLM:
        case LLM_ARCH_PANGU_EMBED:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_SPARK2_5:
        case LLM_ARCH_TALKIE:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_MAPLE:
        case LLM_ARCH_HRM_TEXT:
            return LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_DFLASH:
            // drafts for M-RoPE targets carry rope sections and follow the target's temporal dim
            if (const auto & s = model->hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
                return LLAMA_ROPE_TYPE_MROPE;
            }
            // DSV4 DSpark drafters use DeepSeek-V4's normal RoPE; legacy DFlash backbones are NeoX
            return model->hparams.dsv4_hc_mult > 0 ? LLAMA_ROPE_TYPE_NORM : LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_QWEN2VL:
        case LLM_ARCH_PADDLEOCR:
            return LLAMA_ROPE_TYPE_MROPE;
        case LLM_ARCH_QWEN3VL:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_QWEN3TTS:
            return LLAMA_ROPE_TYPE_IMROPE;

        case LLM_ARCH_GLM4:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NORM;
        case LLM_ARCH_GLM4_MOE:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_HUNYUAN_VL:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

float llama_model_rope_freq_scale_train(const llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t llama_model_meta_val_str(const llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_count(const llama_model * model) {
    return (int)model->gguf_kv.size();
}

const char * llama_model_meta_key_str(llama_model_meta_key key) {
    switch (key) {
        case LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE:        return "general.sampling.sequence";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_K:           return "general.sampling.top_k";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_P:           return "general.sampling.top_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIN_P:           return "general.sampling.min_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY: return "general.sampling.xtc_probability";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD:   return "general.sampling.xtc_threshold";
        case LLAMA_MODEL_META_KEY_SAMPLING_TEMP:            return "general.sampling.temp";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N:  return "general.sampling.penalty_last_n";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT:  return "general.sampling.penalty_repeat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT:        return "general.sampling.mirostat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU:    return "general.sampling.mirostat_tau";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA:    return "general.sampling.mirostat_eta";
        default:                                            return nullptr;
    }
}

int32_t llama_model_meta_key_by_index(const llama_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_model_meta_val_str_by_index(const llama_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_desc(const llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s", model->desc().c_str());
}

llama_ftype llama_model_ftype(const llama_model * model) {
    return model->ftype();
}

uint64_t llama_model_size(const llama_model * model) {
    return model->size();
}

const char * llama_model_chat_template(const llama_model * model, const char * name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        // one-off fix for very popular models (so we are not flooded with issues)
        // do not extend this list unless absolutely necessary
        // Mistral-Small-2503 does not have built-in chat template
        llama_vocab_pre_type pre_type = model->vocab.get_pre_type();
        if (!name && pre_type == LLAMA_VOCAB_PRE_TYPE_TEKKEN && model->layers.size() == 40) {
            return "mistral-v7-tekken";
        }

        return nullptr;
    }

    return it->second.c_str();
}

uint64_t llama_model_n_params(const llama_model * model) {
    return model->n_elements();
}

bool llama_model_has_encoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_EAGLE3:
        case LLM_ARCH_DFLASH:    return true;
        default:                 return false;
    }
}

bool llama_model_has_decoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5ENCODER: return false;
        default:                 return true;
    }
}

llama_token llama_model_decoder_start_token(const llama_model * model) {
    return model->hparams.dec_start_token_id;
}

bool llama_model_is_recurrent(const llama_model * model) {
    return llm_arch_is_recurrent(model->arch);
}

bool llama_model_is_hybrid(const llama_model * model) {
    return llm_arch_is_hybrid(model->arch);
}

bool llama_model_is_diffusion(const llama_model * model) {
    return llm_arch_is_diffusion(model->arch);
}

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model) {
    return model->tensors_by_name;
}

int32_t llama_model_n_expert(const struct llama_model * model) {
    return model->hparams.n_expert;
}

int32_t llama_model_n_devices(const struct llama_model * model) {
    return (int32_t)model->devices.size();
}

ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i) {
    if (i < 0 || i >= (int)model->devices.size()) {
        return nullptr;
    }
    return model->devices[i].dev;
}

void llama_model_set_attn_rot(struct llama_model * model, bool value) {
    model->attn_rot = value;
}

//
// llama_model_base
//

llama_model_base::llama_model_base(const struct llama_model_params & params) : llama_model(params), model(this), tn(model->arch),
    TENSOR_DUPLICATED     (llama_model_loader::TENSOR_DUPLICATED),
    TENSOR_NOT_REQUIRED   (llama_model_loader::TENSOR_NOT_REQUIRED),
    TENSOR_SKIP           (llama_model_loader::TENSOR_SKIP),
    TENSOR_SKIP_IF_VIRTUAL(llama_model_loader::TENSOR_SKIP_IF_VIRTUAL),
    TENSOR_STREAMED       (llama_model_loader::TENSOR_STREAMED),
    TENSOR_ALLOW_RESHAPE  (llama_model_loader::TENSOR_ALLOW_RESHAPE),
    TENSOR_READ_LAZY      (llama_model_loader::TENSOR_READ_LAZY) {}

ggml_tensor * llama_model_base::create_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    GGML_ASSERT(ml != nullptr);
    return create_tensor(*ml, tn, ne, flags);
}

void llama_model_base::create_tensor_gate_up_exps(llama_layer & layer, int bid, int64_t n_embd_, int64_t n_ff_, int64_t n_expert_, int flags) {
    if (flags & TENSOR_SKIP) {
        const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;

        create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", bid), {n_embd_, n_ff_ * 2, n_expert_}, skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,    "weight", bid), {n_embd_, n_ff_,     n_expert_}, skip);
        create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,      "weight", bid), {n_embd_, n_ff_,     n_expert_}, skip);
        return;
    }

    layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", bid), {n_embd_, n_ff_ * 2, n_expert_}, TENSOR_NOT_REQUIRED);
    if (layer.ffn_gate_up_exps == nullptr) {
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
    }
}

void llama_model_base::create_tensor_qkv(llama_layer & layer, int bid,
        int64_t n_embd_, int64_t n_embd_q_, int64_t n_embd_k_, int64_t n_embd_v_,
        int flags) {
    const int64_t n_embd_qkv = n_embd_q_ + n_embd_k_ + n_embd_v_;

    if (flags & TENSOR_SKIP) {
        const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;

        create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias",   bid), {n_embd_qkv},          skip | TENSOR_SKIP_IF_VIRTUAL);
        create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", bid), {n_embd_, n_embd_q_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", bid), {n_embd_, n_embd_k_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", bid), {n_embd_, n_embd_v_},  skip);
        create_tensor(tn(LLM_TENSOR_ATTN_Q,   "bias",   bid), {n_embd_q_},           skip);
        create_tensor(tn(LLM_TENSOR_ATTN_K,   "bias",   bid), {n_embd_k_},           skip);
        create_tensor(tn(LLM_TENSOR_ATTN_V,   "bias",   bid), {n_embd_v_},           skip);
        return;
    }

    layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
    if (layer.wqkv) {
        layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", bid), {n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
        // Fused weights may coexist with separate Q/K/V biases in legacy or custom GGUFs.
        if (!layer.wqkv_b) {
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", bid), {n_embd_q_}, TENSOR_NOT_REQUIRED);
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", bid), {n_embd_k_}, TENSOR_NOT_REQUIRED);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", bid), {n_embd_v_}, TENSOR_NOT_REQUIRED);
        }
    } else {
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", bid), {n_embd_, n_embd_q_}, flags);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", bid), {n_embd_, n_embd_k_}, flags);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", bid), {n_embd_, n_embd_v_}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", bid), {n_embd_q_}, TENSOR_NOT_REQUIRED);
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", bid), {n_embd_k_}, TENSOR_NOT_REQUIRED);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", bid), {n_embd_v_}, TENSOR_NOT_REQUIRED);
    }
}

llama_model_base::nextn_flags_t llama_model_base::nextn_flags(llama_model_loader & ml, llm_tensor trunk_probe) const {
    int trunk = 0;
    int mtp   = 0;

    // a file without the first trunk layer is MTP-only, a file without the first NextN layer is trunk-only
    if (hparams.n_layer_nextn > 0) {
        if (ml.get_weight(tn(trunk_probe, "weight", 0).str().c_str()) == nullptr) {
            trunk = TENSOR_NOT_REQUIRED;
        }
        if (ml.get_weight(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", hparams.n_layer()).str().c_str()) == nullptr) {
            mtp = TENSOR_NOT_REQUIRED;
        }
    }

    if (!ml.load_mtp) {
        mtp |= TENSOR_SKIP;
    }

    return { trunk, mtp };
}

void llama_model_base::load_swa_pattern(llama_model_loader & ml, uint32_t n_pattern, bool dense_first) {
    if (ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, false)) {
        return;
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern, false);
    hparams.set_swa_pattern(n_pattern, dense_first);
}

const int32_t * llama_model_target_layer_ids(const struct llama_model * model) {
    const auto & v = model->target_layer_ids;
    return v.empty() ? nullptr : v.data();
}

uint32_t llama_model_target_layer_ids_n(const struct llama_model * model) {
    return (uint32_t) model->target_layer_ids.size();
}

uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out) {
    if (model->vocab.n_tokens() == 0 || model->tok_embd == nullptr) {
        return 0;
    }

    const ggml_tensor * tensor = model->tok_embd;
    const size_t nelements = ggml_nelements(tensor);
    GGML_ASSERT(nelements <= UINT32_MAX); // for the return type

    if (out == nullptr) {
        return (uint32_t) nelements;
    }

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, out, 0, nelements * sizeof(float));
        return (uint32_t) nelements;
    }

    std::vector<uint8_t> buf(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, buf.data(), 0, buf.size());

    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) buf.data(), out, nelements);
    } else if (tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) buf.data(), out, nelements);
    } else if (ggml_is_quantized(tensor->type) && traits->to_float != nullptr) {
        traits->to_float(buf.data(), out, nelements);
    } else {
        GGML_ABORT("unsupported tensor type for dequantization: %s", ggml_type_name(tensor->type));
    }

    return (uint32_t) nelements;
}

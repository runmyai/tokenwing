// Various helper functions and utilities

#pragma once

#include "llama-cpp.h"

#include "ggml-opt.h"
#include "ggml.h"
#include "llama.h"

#include <array>
#include <list>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstdio>

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#else
#define DIRECTORY_SEPARATOR '/'
#endif // _WIN32

#define COMMON_RECURSIVE_LIMIT 128

#define COM_DBG(fmt, ...) LOG_DBG("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_TRC(fmt, ...) LOG_TRC("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_INF(fmt, ...) LOG_INF("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_WRN(fmt, ...) LOG_WRN("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_ERR(fmt, ...) LOG_ERR("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define die(msg)          do { fputs("error: " msg "\n", stderr);                exit(1); } while (0)
#define die_fmt(fmt, ...) do { fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); exit(1); } while (0)

struct common_time_meas {
    common_time_meas(int64_t & t_acc, bool disable = false);
    ~common_time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

struct common_adapter_lora_info {
    std::string path;
    float scale;

    std::string task_name;
    std::string prompt_prefix;

    struct llama_adapter_lora * ptr;
};

using llama_tokens = std::vector<llama_token>;

struct common_control_vector_load_info;

//
// CPU utils
//

struct common_cpu_params {
    int      n_threads                   = -1;
    bool     cpumask[GGML_MAX_N_THREADS] = {false}; // CPU affinity mask.
    bool     mask_valid                  = false;   // Default: any CPU
    enum ggml_sched_priority  priority   = GGML_SCHED_PRIO_NORMAL;  // Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)
    bool     strict_cpu                  = false;   // Use strict CPU placement
    uint32_t poll                        = 50;      // Polling (busywait) level (0 - no polling, 100 - mostly polling)
};

int32_t common_cpu_get_num_physical_cores();
int32_t common_cpu_get_num_math();

//
// Common params
//

enum llama_example {
    LLAMA_EXAMPLE_BATCHED,
    LLAMA_EXAMPLE_DEBUG,
    LLAMA_EXAMPLE_COMMON,
    LLAMA_EXAMPLE_SPECULATIVE,
    LLAMA_EXAMPLE_COMPLETION,
    LLAMA_EXAMPLE_CLI,
    LLAMA_EXAMPLE_EMBEDDING,
    LLAMA_EXAMPLE_PERPLEXITY,
    LLAMA_EXAMPLE_RETRIEVAL,
    LLAMA_EXAMPLE_PASSKEY,
    LLAMA_EXAMPLE_IMATRIX,
    LLAMA_EXAMPLE_BENCH,
    LLAMA_EXAMPLE_SERVER,
    LLAMA_EXAMPLE_CVECTOR_GENERATOR,
    LLAMA_EXAMPLE_EXPORT_LORA,
    LLAMA_EXAMPLE_MTMD,
    LLAMA_EXAMPLE_LOOKUP,
    LLAMA_EXAMPLE_PARALLEL,
    LLAMA_EXAMPLE_TTS,
    LLAMA_EXAMPLE_DIFFUSION,
    LLAMA_EXAMPLE_FINETUNE,
    LLAMA_EXAMPLE_FIT_PARAMS,
    LLAMA_EXAMPLE_RESULTS,
    LLAMA_EXAMPLE_EXPORT_GRAPH_OPS,
    LLAMA_EXAMPLE_MOE_TRACE,
    LLAMA_EXAMPLE_DOWNLOAD,
    LLAMA_EXAMPLE_TOKENIZE,

    LLAMA_EXAMPLE_COUNT,
};

enum common_sampler_type {
    COMMON_SAMPLER_TYPE_NONE        = 0,
    COMMON_SAMPLER_TYPE_DRY         = 1,
    COMMON_SAMPLER_TYPE_TOP_K       = 2,
    COMMON_SAMPLER_TYPE_TOP_P       = 3,
    COMMON_SAMPLER_TYPE_MIN_P       = 4,
  //COMMON_SAMPLER_TYPE_TFS_Z       = 5,
    COMMON_SAMPLER_TYPE_TYPICAL_P   = 6,
    COMMON_SAMPLER_TYPE_TEMPERATURE = 7,
    COMMON_SAMPLER_TYPE_XTC         = 8,
    COMMON_SAMPLER_TYPE_INFILL      = 9,
    COMMON_SAMPLER_TYPE_PENALTIES   = 10,
    COMMON_SAMPLER_TYPE_TOP_N_SIGMA = 11,
    COMMON_SAMPLER_TYPE_ADAPTIVE_P  = 12,
};

// dimensionality reduction methods, used by cvector-generator
enum dimre_method {
    DIMRE_METHOD_PCA,
    DIMRE_METHOD_MEAN,
};

enum common_conversation_mode {
    COMMON_CONVERSATION_MODE_DISABLED = 0,
    COMMON_CONVERSATION_MODE_ENABLED  = 1,
    COMMON_CONVERSATION_MODE_AUTO     = 2,
};

enum common_grammar_trigger_type {
    COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
};

struct common_grammar_trigger {
    common_grammar_trigger_type type;
    std::string value;
    llama_token token = LLAMA_TOKEN_NULL;
};

enum common_params_sampling_config : uint64_t {
    COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS        = 1 << 0,
    COMMON_PARAMS_SAMPLING_CONFIG_TOP_K           = 1 << 1,
    COMMON_PARAMS_SAMPLING_CONFIG_TOP_P           = 1 << 2,
    COMMON_PARAMS_SAMPLING_CONFIG_MIN_P           = 1 << 3,
    COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY = 1 << 4,
    COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD   = 1 << 5,
    COMMON_PARAMS_SAMPLING_CONFIG_TEMP            = 1 << 6,
    COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N  = 1 << 7,
    COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT  = 1 << 8,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT        = 1 << 9,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU    = 1 << 10,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA    = 1 << 11,
};

enum common_speculative_type {
    COMMON_SPECULATIVE_TYPE_NONE,          // no speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE,  // standalone draft model speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3,  // Eagle3 speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_MTP,     // Multi-token prediction
    COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,  // DFlash speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK,  // DSpark speculative decoding (DFlash + Markov head)
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,  // simple self-speculative decoding based on n-grams
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,   // self-speculative decoding with n-gram keys only
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, // self-speculative decoding with n-gram keys and 4 m-gram values
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,   // self-speculative decoding with 3-level n-gram cache
    COMMON_SPECULATIVE_TYPE_COUNT          // number of types, unknown type
};

// Grammar type enumeration
enum common_grammar_type {
    COMMON_GRAMMAR_TYPE_NONE,           // no grammar set
    COMMON_GRAMMAR_TYPE_USER,           // user-provided GBNF (--grammar / "grammar" API field)
    COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,  // auto-generated from JSON schema (--json-schema / "json_schema" API field)
    COMMON_GRAMMAR_TYPE_TOOL_CALLS,     // auto-generated by chat template parser for function calling
};

// Grammar variant struct with type and grammar string
struct common_grammar {
    common_grammar_type type = COMMON_GRAMMAR_TYPE_NONE;
    std::string grammar;

    // Default constructor - no grammar
    common_grammar() = default;

    // Constructor with type and grammar string
    common_grammar(common_grammar_type t, std::string g) : type(t), grammar(std::move(g)) {
        GGML_ASSERT(type != COMMON_GRAMMAR_TYPE_NONE || !grammar.empty());
    }

    // Check if a grammar is set
    bool empty() const { return type == COMMON_GRAMMAR_TYPE_NONE || grammar.empty(); }
};

// Returns the raw grammar string, or empty string if no grammar is set.
inline const std::string & common_grammar_value(const common_grammar & g) {
    return g.grammar;
}

// Returns true when the generation_prompt should be prefilled into the grammar sampler.
// Only output-format and tool-call grammars need prefill; user-supplied grammars must not be prefilled.
inline bool common_grammar_needs_prefill(const common_grammar & g) {
    return g.type == COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT
        || g.type == COMMON_GRAMMAR_TYPE_TOOL_CALLS;
}

// sampling parameters
struct common_params_sampling {
    uint32_t seed = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampler

    int32_t n_prev             = 64;     // number of previous tokens to remember
    int32_t n_probs            = 0;      // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t min_keep           = 0;      // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t top_k              = 40;     // <= 0 to use vocab size
    float   top_p              = 0.95f;  // 1.0 = disabled
    float   min_p              = 0.05f;  // 0.0 = disabled
    float   xtc_probability    = 0.00f;  // 0.0 = disabled
    float   xtc_threshold      = 0.10f;  // > 0.5 disables XTC
    float   typ_p              = 1.00f;  // typical_p, 1.0 = disabled
    float   temp               = 0.80f;  // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float   dynatemp_range     = 0.00f;  // 0.0 = disabled
    float   dynatemp_exponent  = 1.00f;  // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t penalty_last_n     = 64;     // last n tokens to penalize (0 = disable penalty)
    float   penalty_repeat     = 1.00f;  // 1.0 = disabled
    float   penalty_freq       = 0.00f;  // 0.0 = disabled
    float   penalty_present    = 0.00f;  // 0.0 = disabled
    float   dry_multiplier     = 0.0f;   // 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
    float   dry_base           = 1.75f;  // 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
    int32_t dry_allowed_length = 2;      // tokens extending repetitions beyond this receive penalty
    int32_t dry_penalty_last_n = 64;     // how many tokens to scan for repetitions (0 = disable penalty)
    float   adaptive_target    = -1.0f;  // select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    float   adaptive_decay     = 0.90f;  // EMA decay for adaptation; history ≈ 1/(1-decay) tokens (0.0 - 0.99)
    int32_t mirostat           = 0;      // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   top_n_sigma        = -1.00f; // -1.0 = disabled
    float   mirostat_tau       = 5.00f;  // target entropy
    float   mirostat_eta       = 0.10f;  // learning rate
    bool    ignore_eos         = false;
    bool    no_perf            = false;  // disable performance metrics
    bool    timing_per_token   = false;

    uint64_t user_sampling_config = 0; // bitfield to track user-specified samplers

    std::vector<std::string> dry_sequence_breakers = {"\n", ":", "\"", "*"};     // default sequence breakers for DRY

    std::vector<enum common_sampler_type> samplers = {
        COMMON_SAMPLER_TYPE_PENALTIES,
        COMMON_SAMPLER_TYPE_DRY,
        COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TYPICAL_P,
        COMMON_SAMPLER_TYPE_TOP_P,
        COMMON_SAMPLER_TYPE_MIN_P,
        COMMON_SAMPLER_TYPE_XTC,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };

    common_grammar                      grammar;          // optional grammar constraint (user / output-format / tool-calls)
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers; // optional triggers (for lazy grammars)
    std::set<llama_token>               preserved_tokens;

    std::vector<llama_logit_bias> logit_bias;     // logit biases to apply
    std::vector<llama_logit_bias> logit_bias_eog; // pre-calculated logit biases for EOG tokens

    // The assistant generation prompt already prefilled into the prompt.
    // Fed to the grammar sampler (to advance past pre-existing tokens) and used
    // to determine the reasoning budget sampler's initial state.
    // Only applied when the grammar is of output-format or tool-calls type.
    std::string generation_prompt;

    // reasoning budget sampler parameters
    // these are populated by the server/CLI based on chat template params
    int32_t                   reasoning_budget_tokens   = -1;  // -1 = disabled, >= 0 = token budget
    std::vector<llama_token>  reasoning_budget_start;          // start tag token sequence
    std::vector<llama_tokens> reasoning_budget_end;            // end tag token sequences; the first tag is used as the forcing sequence
    std::vector<llama_token>  reasoning_budget_forced;         // forced sequence (message + first end tag)
    std::string               reasoning_budget_message;        // message injected before end tag when budget exhausted
    bool                      reasoning_control = false;       // create the budget sampler on demand so reasoning can be ended at runtime

    bool backend_sampling = false;

    // print the parameters into a string
    std::string print() const;
};

struct common_params_model {
    std::string path        = ""; // model local path
    std::string url         = ""; // model url to download
    std::string hf_repo     = ""; // HF repo
    std::string hf_file     = ""; // HF file
    std::string docker_repo = ""; // Docker repo

    std::string get_name() const {
        if (!hf_repo.empty()) {
            return hf_repo;
        }
        if (!docker_repo.empty()) {
            return docker_repo;
        }
        return path;
    }

    bool empty() const {
        return get_name().empty();
    }
};

// draft-model-based speculative decoding parameters
struct common_params_speculative_draft {
    int32_t n_max = 3; // maximum number of tokens to draft during speculative decoding
    int32_t n_min = 0; // minimum number of draft tokens to use for speculative decoding

    float p_split = 0.1f; // speculative decoding split probability
    float p_min   = 0.0f; // minimum speculative decoding probability (greedy)

    bool backend_sampling = true; // offload draft sampling to the backend (default: on)

    bool probabilistic = false; // sample the draft and verify by rejection, instead of argmax and match

    common_params_model mparams;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    int32_t n_gpu_layers = -1; // number of layers to store in VRAM for the draft model (-1 - use default)

    int32_t n_ubatch = 0; // physical batch size of the draft context (0 - use the target's n_ubatch)

    ggml_type cache_type_k = GGML_TYPE_F16; // KV cache data type for the K
    ggml_type cache_type_v = GGML_TYPE_F16; // KV cache data type for the V
    int32_t   kv_hadamard  = -1;            // -1 = auto, 0 = force off, 1 = force on Walsh-Hadamard KV rotation

    common_cpu_params cpuparams;
    common_cpu_params cpuparams_batch;

    std::vector<ggml_backend_dev_t> devices; // devices to use for offloading

    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
};

struct common_params_speculative_ngram_mod {
    int32_t n_match = 24;

    int32_t n_max = 64;
    int32_t n_min = 48;
};

struct common_params_speculative_ngram_map {
    uint16_t size_n   = 12; // ngram size for lookup
    uint16_t size_m   = 48; // mgram size for speculative tokens
    uint16_t min_hits = 1;  // minimum hits at ngram/mgram lookup for mgram to be proposed
};

struct common_params_speculative_ngram_cache {
    std::string lookup_cache_static;  // path of static ngram cache file for lookup decoding
    std::string lookup_cache_dynamic; // path of dynamic ngram cache file for lookup decoding
};

struct common_params_speculative {
    std::vector<enum common_speculative_type> types = { COMMON_SPECULATIVE_TYPE_NONE };

    double synth_len = -1.0;
    std::vector<double> synth_rates;

    // used by Simple, MTP, Eagle3, etc. - all methods that require some kind of draft model
    common_params_speculative_draft draft;

    common_params_speculative_ngram_mod ngram_mod;
    common_params_speculative_ngram_map ngram_simple;
    common_params_speculative_ngram_map ngram_map_k;
    common_params_speculative_ngram_map ngram_map_k4v;

    common_params_speculative_ngram_cache ngram_cache;

    bool has_dft() const {
        return !draft.mparams.empty();
    }

    bool has_synth() const {
        return synth_len != -1.0 || !synth_rates.empty();
    }

    uint32_t need_n_rs_seq() const {
        bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 || t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });

        return needs_rs_seq ? draft.n_max : 0u;
    }
};

struct common_params_diffusion {
    int32_t steps         = 128;
    bool    visual_mode   = false;

    float   eps           = 0;        // epsilon for timesteps
    int32_t block_length  = 0;        // block length for generation

    int32_t algorithm     = 4;        // default algorithm: low-confidence
    float   alg_temp      = 0.0f;     // algorithm temperature

    float   cfg_scale     = 0;        // classifier-free guidance scale
    bool    add_gumbel_noise = false; // add gumbel noise to the logits if temp > 0.0
};

// reasoning API response format (not to be confused as chat template's reasoning format)
// only used by server
enum common_reasoning_format {
    COMMON_REASONING_FORMAT_NONE,
    COMMON_REASONING_FORMAT_AUTO,            // Same as deepseek, using `message.reasoning_content`
    COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY, // Extract thinking tag contents and return as `message.reasoning_content`, or leave inline in <think> tags in stream mode
    COMMON_REASONING_FORMAT_DEEPSEEK,        // Extract thinking tag contents and return as `message.reasoning_content`, including in streaming deltas.
    // do not extend this enum unless you absolutely have to
    // in most cases, use COMMON_REASONING_FORMAT_AUTO
    // see: https://github.com/ggml-org/llama.cpp/pull/15408
};


struct lr_opt {
    float    lr0          = 1e-5; // learning rate at first epoch
    float    lr_min       = -1;
    float    decay_epochs = -1;   // if >0, the learning rate starts at lr0 and decays to lr_min after this many epochs
    float    scale_epoch  = 0;
    float    wd           = 0;
    unsigned epochs       = 2;

    unsigned epoch; // set by optimizer outer (epochs) loop
    // learning rate decay - constant LR per epoch only for now
    float get_lr(float e) const;
    float get_lr() const { return get_lr(epoch); }
    // must call after arg parse, before get_lr
    void init();
};

struct ggml_opt_optimizer_params common_opt_lr_pars(void * userdata);

struct common_params {
    int32_t n_predict             =    -1; // max. number of new tokens to predict, -1 == no limit
    int32_t n_ctx                 =     0; // context size, 0 == context the model was trained with
    int32_t n_batch               =  2048; // logical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_ubatch              =   512; // physical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep                =     0; // number of tokens to keep from initial prompt
    int32_t n_chunks              =    -1; // max number of chunks to process (-1 = unlimited)
    int32_t n_parallel            =     1; // number of parallel sequences to decode
    int32_t n_sequences           =     1; // number of sequences to decode
    int32_t n_outputs_max         =     0; // max outputs in a batch (0 = n_batch)
    int32_t n_outputs_max_per_seq =     1; // max outputs per sequence
    int32_t grp_attn_n            =     1; // group-attention factor
    int32_t grp_attn_w            =   512; // group-attention width
    int32_t n_print               =    -1; // print token count every n tokens (-1 = disabled)
    float   rope_freq_base        =  0.0f; // RoPE base frequency
    float   rope_freq_scale       =  0.0f; // RoPE frequency scaling factor
    float   yarn_ext_factor       = -1.0f; // YaRN extrapolation mix factor
    float   yarn_attn_factor      = -1.0f; // YaRN magnitude scaling factor
    float   yarn_beta_fast        = -1.0f; // YaRN low correction dim
    float   yarn_beta_slow        = -1.0f; // YaRN high correction dim
    int32_t yarn_orig_ctx         =     0; // YaRN original context length

    // offload params
    std::vector<ggml_backend_dev_t> devices; // devices to use for offloading

    int32_t n_gpu_layers       = -1;    // number of layers to store in VRAM, -1 is auto, <= -2 is all
    int32_t main_gpu           = 0;     // the GPU that is used for scratch and small tensors
    float   tensor_split[128]  = {0};   // how split tensors should be distributed across GPUs
    bool    fit_params         = true;  // whether to fit unset model/context parameters to free device memory
    bool    fit_params_print   = false; // print the estimated required memory to run the model
    int32_t fit_params_min_ctx = 4096;  // minimum context size to set when trying to reduce memory use

    // margin per device in bytes for fitting parameters to free memory:
    std::vector<size_t> fit_params_target = std::vector<size_t>(llama_max_devices(), 1024 * 1024*1024);

    enum llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER; // how to split the model across GPUs
    enum llama_load_mode  load_mode  = LLAMA_LOAD_MODE_AUTO; // how to load the model

    enum llama_lazy_mode lazy_mode = LLAMA_LAZY_MODE_AUTO; // on-demand reading of tensors marked by the arch

    common_cpu_params cpuparams;
    common_cpu_params cpuparams_batch;

    ggml_backend_sched_eval_callback cb_eval = nullptr;
    void * cb_eval_user_data                 = nullptr;

    ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;

    enum llama_rope_scaling_type rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
    enum llama_pooling_type      pooling_type      = LLAMA_POOLING_TYPE_UNSPECIFIED; // pooling type for embeddings
    enum llama_attention_type    attention_type    = LLAMA_ATTENTION_TYPE_UNSPECIFIED; // attention type for embeddings
    enum llama_flash_attn_type   flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_AUTO; // whether to use Flash Attention

    struct common_params_sampling    sampling;
    struct common_params_speculative speculative;
    struct common_params_diffusion   diffusion;

    struct common_params_model model;

    std::set<std::string> model_alias;     // model aliases                                                 // NOLINT
    std::set<std::string> model_tags;      // model tags (informational, not used for routing)              // NOLINT
    std::string hf_token             = ""; // HF token (aka bearer token)                                   // NOLINT
    std::string prompt               = "";                                                                  // NOLINT
    std::string system_prompt        = "";                                                                  // NOLINT
    std::string prompt_file          = ""; // store the external prompt file name                           // NOLINT
    std::string path_prompt_cache    = ""; // path to file for saving/loading prompt eval state             // NOLINT
    std::string input_prefix         = ""; // string to prefix user inputs with                             // NOLINT
    std::string input_suffix         = ""; // string to suffix user inputs with                             // NOLINT
    std::string logits_file          = ""; // file for saving *all* logits                                  // NOLINT
    std::string path_prompts_log_dir = ""; // directory with logged prompts                                 // NOLINT

    // llama-debug specific options
    std::string logits_output_dir = "data"; // directory for saving logits output files                     // NOLINT
    bool        save_logits       = false;  // whether to save logits to files                              // NOLINT
    std::vector<std::string> tensor_filter; // filter tensor names for debug output (regex)                 // NOLINT

    std::vector<std::string> in_files;   // all input files
    std::vector<std::string> antiprompt; // strings upon which more user input is prompted (a.k.a. reverse prompts)
    std::vector<llama_model_kv_override> kv_overrides;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;

    bool lora_init_without_apply = false; // only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_adapter_lora_apply)
    std::vector<common_adapter_lora_info> lora_adapters; // lora adapter path with user defined scale

    std::vector<common_control_vector_load_info> control_vectors; // control vector with user defined scale

    int32_t verbosity                  = 3;  // LOG_LEVEL_INFO
    int32_t control_vector_layer_start = -1; // layer range for control vector
    int32_t control_vector_layer_end   = -1; // layer range for control vector
    bool    offline                    = false;

    int32_t ppl_stride      = 0;     // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
    int32_t ppl_output_type = 0;     // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
                                     //                                       (which is more convenient to use for plotting)
                                     //
    bool   hellaswag        = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
    size_t hellaswag_tasks  = 400;   // number of tasks to use when computing the HellaSwag score

    bool   winogrande       = false; // compute Winogrande score over random tasks from datafile supplied in prompt
    size_t winogrande_tasks = 0;     // number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

    bool   multiple_choice  = false;  // compute TruthfulQA score over random tasks from datafile supplied in prompt
    size_t multiple_choice_tasks = 0; // number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

    bool   kl_divergence    = false; // compute KL divergence

    bool check             = false; // check rather than generate results for llama-results

    bool usage             = false; // print usage
    bool completion        = false; // print source-able completion script
    bool use_color         = false; // use color to distinguish generations and inputs
    bool special           = false; // enable special token output
    bool interactive       = false; // interactive mode
    bool interactive_first = false; // wait for user input immediately
    bool prompt_cache_all  = false; // save user input and generations to prompt cache
    bool prompt_cache_ro   = false; // open the prompt cache read-only and do not update it

    bool escape            = true;  // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    bool multiline_input   = false; // reverse the usage of `\`
    bool simple_io         = false; // improves compatibility with subprocesses and limited consoles
    bool cont_batching     = true;  // insert new sequences for decoding on-the-fly
    bool no_perf           = false; // disable performance metrics
    bool show_timings      = true;  // show timing information on CLI
    bool ctx_shift         = false; // context shift on infinite text generation
    bool swa_full          = false; // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
    bool kv_unified        = false; // enable unified KV cache

    bool input_prefix_bos  = false; // prefix BOS to user inputs, preceding input_prefix
    bool verbose_prompt    = false; // print prompt tokens before generation
    bool display_prompt    = true;  // print prompt before generation
    bool no_kv_offload     = false; // disable KV offloading
    bool warmup            = true;  // warmup run
    bool check_tensors     = false; // validate tensor data
    bool no_op_offload     = false; // globally disable offload host tensor operations to device
    bool no_extra_bufts    = false; // disable extra buffer types (used for weight repacking)
    bool no_host           = false; // bypass host buffer allowing extra buffers to be used

    bool single_turn       = false; // single turn chat conversation

    ggml_type cache_type_k = GGML_TYPE_F16; // KV cache data type for the K
    ggml_type cache_type_v = GGML_TYPE_F16; // KV cache data type for the V
    int32_t   kv_hadamard  = -1;            // -1 = auto, 0 = force off, 1 = force on Walsh-Hadamard KV rotation

    common_conversation_mode conversation_mode = COMMON_CONVERSATION_MODE_AUTO;

    // multimodal models (see tools/mtmd)
    struct common_params_model mmproj;
    bool mmproj_use_gpu = true;                 // use GPU for multimodal model
    ggml_backend_dev_t mmproj_device = nullptr; // GPU device to use for multimodal model
    bool no_mmproj = false;                     // explicitly disable multimodal model
    std::vector<std::string> image;             // path to image file(s) ; TODO: change the name to "media"
    int image_min_tokens = -1;
    int image_max_tokens = -1;
    int mtmd_batch_max_tokens = 1024;

    // for video input
    float       video_fps                   = 4.0f;
    int64_t     video_timestamp_interval_ms = 5000;
    std::string video_ffmpeg_bin_dir        = "";

    // finetune
    struct lr_opt lr;
    enum ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    float val_split = 0.05f; // fraction of the data used for the validation set

    // embedding
    bool embedding         = false; // get only sentence embedding
    int32_t embd_normalize = 2;     // normalisation for embeddings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
    std::string embd_out   = "";    // empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix
    std::string embd_sep   = "\n";  // separator of embeddings
    std::string cls_sep    = "\t";  // separator of classification sequences

    // server params
    int32_t port                = 8080;          // server listens on this network port
    bool    reuse_port          = false;         // allow multiple sockets to bind to the same port
    int32_t timeout_read        = 3600;          // http read timeout in seconds
    int32_t timeout_write       = timeout_read;  // http write timeout in seconds
    int32_t sse_ping_interval   = 30;            // SSE ping interval in seconds
    int32_t n_threads_http      = -1;    // number of threads to process HTTP requests (TODO: support threadpool)
    int32_t n_cache_reuse       = 0;     // min chunk size to reuse from the cache via KV shifting
    bool    cache_prompt        = true;  // whether to enable prompt caching
    bool    cache_idle_slots    = true;  // save and clear idle slots upon starting a new task
    int32_t n_ctx_checkpoints   = 2;     // max number of context checkpoints per slot (2 here, upstream 32:
                                         // 32 x 74.9 MiB on Ling outgrew the RAM margin)
    int32_t kv_unified_per_slot = 0;     // max context per parallel slot; 0 = unset
    int32_t checkpoint_min_step = 8192;  // minimum spacing between context checkpoints
    int32_t cache_ram_mib       = 8192;  // -1 = no limit, 0 - disable, 1 = 1 MiB, etc.
    std::string cache_disk_dir;           // the server's chat states on disk, "" = off
    float   cache_disk_gb       = 8.0f;  // size limit of that directory (GB), the oldest-written entry goes first
    int32_t cache_disk_checkpoints = 0;  // "start of message" snapshots per chat kept as files, 0 = off
    bool    cache_ram_explicit  = false; // set by --cache-ram, decision models disable the cache otherwise

    std::string public_path   = "";                                                                         // NOLINT
    std::string api_prefix    = "";                                                                         // NOLINT
    std::string chat_template = "";                                                                         // NOLINT
    std::vector<std::string> hostnames = {"127.0.0.1"};
    bool use_jinja = true;                                                                                  // NOLINT

    // server CORS params
    std::string cors_origins = "*";
    std::string cors_methods = "GET, POST, DELETE, OPTIONS";
    std::string cors_headers = "*";
    bool cors_credentials = true;
    bool cors_origins_explicit = false; // for --agent option

    bool enable_chat_template = true;
    bool force_pure_content_parser = false;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    int enable_reasoning = -1; // -1 = auto, 0 = disable, 1 = enable
    bool prefill_assistant = true; // if true, any trailing assistant message will be prefilled into the response
    int sleep_idle_seconds = -1;   // if >0, server will sleep after this many seconds of idle time

    std::vector<std::string> api_keys;

    std::string ssl_file_key  = "";                                                                         // NOLINT
    std::string ssl_file_cert = "";                                                                         // NOLINT

    std::map<std::string, std::string> default_template_kwargs;
    bool preserve_reasoning_specified = false;

    // CLI params
    std::string server_base; // if set, connect to this server instead of starting a new one

    // UI configs
    bool ui = true;
    bool ui_mcp_proxy = false;
    std::string ui_config_json;

    // "advanced" endpoints are disabled by default for better security
    bool endpoint_slots   = true;
    bool endpoint_props   = false; // only control POST requests, not GET
    bool endpoint_metrics = false;

    bool usage_stats = true; // llama-server: one anonymous usage record a day

    // enable built-in tools
    std::vector<std::string> server_tools;
    std::string server_tools_runtime;

    // MCP server configs (Cursor-compatible JSON)
    std::string mcp_servers_config;   // path to JSON file with MCP server definitions
    std::string mcp_servers_json;     // inline JSON with MCP server definitions

    // router server configs
    std::string models_dir    = "";     // directory containing models for the router server
    std::string models_preset = "";     // directory containing model presets for the router server
    int models_max = 4;                 // maximum number of models to load simultaneously
    bool models_autoload = true;        // automatically load models when requested via the router server
    std::string models_preset_hf = "";  // show a warning about remote presets on router loaded (if not empty)

    bool log_json = false;

    std::string slot_save_path;
    std::string media_path; // path to directory for loading media files

    float slot_prompt_similarity = 0.1f;

    // batched-bench params
    bool is_pp_shared   = false;
    bool is_tg_separate = false;

    std::vector<int32_t> n_pp;
    std::vector<int32_t> n_tg;
    std::vector<int32_t> n_pl;

    // retrieval params
    std::vector<std::string> context_files; // context files to embed

    int32_t chunk_size = 64; // chunk size for context embedding

    std::string chunk_separator = "\n"; // chunk separator for context embedding

    // passkey params
    int32_t n_junk = 250; // number of times to repeat the junk text
    int32_t i_pos  = -1;  // position of the passkey in the junk text

    // imatrix params
    int32_t n_out_freq  = 10; // output the imatrix every n_out_freq iterations
    int32_t n_save_freq =  0; // save the imatrix every n_save_freq iterations
    int32_t i_chunk     =  0; // start processing from this chunk
    int8_t  imat_dat    =  0; // whether the legacy imatrix.dat format should be output (gguf <= 0 < dat)

    bool process_output  = false; // collect data for the output tensor
    bool compute_ppl     = true;  // whether to compute perplexity
    bool show_statistics = false; // show imatrix statistics per tensor
    bool parse_special   = false; // whether to parse special tokens during imatrix tokenization

    // cvector-generator params
    int n_pca_batch = 100;
    int n_pca_iterations = 1000;
    dimre_method cvector_dimre_method = DIMRE_METHOD_PCA;
    std::string cvector_positive_file = "tools/cvector-generator/positive.txt";
    std::string cvector_negative_file = "tools/cvector-generator/negative.txt";

    bool spm_infill = false; // suffix/prefix/middle pattern for infill

    // batched-bench params
    bool batched_bench_output_jsonl = false;

    // tokenize params
    bool tokenize_ids        = false; // if true, only print the token IDs
    bool tokenize_stdin      = false; // if true, read the prompt from stdin
    bool tokenize_no_bos     = false; // if true, do not add the BOS token
    bool tokenize_show_count = false; // if true, print the total token count

    // common params
    std::string out_file; // output filename for all example programs
    bool moe_trace_dump_router_input = false; // llama-moe-trace: also dump the router input
    // llama-moe-trace: also dump the top-N router scores/ids per layer,
    // 0 = off. The selection bias is added, because that is what the router selects on.
    int32_t moe_trace_routing_topk = 0;
    std::string moe_stream = "";      // sidecar metadata path, empty = disabled
    std::string moe_stream_log = "";  // per-token miss log path
    // The defaults (gpu-layers auto, gpu-waves 512) have to
    // live here: a tool that sets nothing but --moe-stream used to get neither feature
    int32_t     moe_stream_slots = 0;  // expert slots per MoE layer, 0 = follow the RAM budget
    int32_t     moe_stream_io_threads = 0;  // threads that read the sidecar, 0 = 8
    // Split a read into chunks of this many bytes so the I/O threads serve it in parallel,
    // 0 = one read per expert (what the engine did before)
    int32_t     moe_stream_chunk = 0;
    int32_t     moe_stream_chunk_prefetch = 0;
    // Min tokens for GPU matmuls in a wave graph; the default is 512, 0 = off
    int32_t     moe_stream_gpu_waves = 512;
    // Whole MoE layers kept on the GPU, 0 = off, negative = fit to the free VRAM; the default
    // is -1 (auto)
    int32_t     moe_stream_gpu_layers = -1;
    // VRAM kept for KV cache, compute buffers, driver. 2 GiB was tuned while the fit
    // silently counted 1.15 GiB too much as used (the unloaded MTP block minus output.weight); with honest counting the
    // default carries that slack itself, so the placement stays the measured one (3 layers here; a 4th measured slower)
    int64_t     moe_stream_vram_reserve = (int64_t) (3.15 * 1024 * 1024 * 1024);
    // "hot" is the default. "last" and "hot" were measured at KLD 0.043 and 0.073 against no whole
    // layer, and compared against the wrong reference (stock --no-op-offload, where only the prompt
    // matmuls move); the fair one is stock's routed experts on the host instead of the GPU, 0.1528 on tiny
    // Both orders are inside it, so the faster one is the default.
    std::string moe_stream_gpu_layers_mode; // "hot" (default) or "last" layer order for the auto-fit
    std::string moe_stream_gpu_layers_list; // explicit layer list, takes precedence over the count
    int64_t     moe_stream_ram_free = 0;  // bytes of RAM to keep free, 0 = the OS default (llama_moe_stream_start_ram_free), negative = no margin
    // Compare one filled pool slot per layer (and one GPU whole-layer expert) against the
    // model file at the first token, so wrong weights are named there and not in the output
    int32_t     moe_stream_verify = 1;   // 0 = no read-back check, N>1 = check N layers
    // Keep the expert pools in llama.cpp's repacked CPU layout (mxfp4_8x8_q8_0 on x86). Every read
    // is converted on the I/O thread. false = the plain layout.
    // OFF by default. With a repacked pool and a ubatch that needs waves (n_tokens >= 512 and more
    // distinct experts than the pool has slots) the engine names only ~20 distinct experts per
    // ubatch-layer instead of ~390 and the text after the prompt is garbage, so the measured 3x prefill
    // was a wrong-output path. Decode (-ub 16, no waves) is fine. Do not turn this on for a prompt run
    // until that is fixed.
    bool        moe_stream_cpu_repack = false;
    int32_t     moe_stream_prefetch = -1;   // layer-ahead prefetch, -1 = on
    int32_t     moe_stream_prefetch_m = 0;  // experts predicted per token, 0 = the model's top-k (was 8)
    int32_t     moe_stream_max_waves = 0;  // cap the expert waves per ubatch and layer, 0 = the safe bound
    int32_t     moe_stream_hugepages = 0;    // 2 MiB-aligned pools + madvise(MADV_HUGEPAGE)
    int32_t     moe_stream_pin = 0;          // cudaHostRegister the pools
    std::string moe_stream_policy = "";      // "lru" (default), "lfu", "slru"
    int32_t     moe_stream_pred_shared = 0;  // predict from the residual + this layer's shared expert
    int32_t     moe_stream_pred_recall = 0;  // count recall@8 of the layer-ahead guess
    std::string moe_stream_pred_scores;      // write the guess's scores for all experts here
    std::string moe_stream_pred_input;       // record the guess's router input here
    int32_t     moe_stream_pred_input_every = 0; // 1 generated word in N, 0 = the default 8
    std::string moe_stream_record;           // write the side-by-side recording here
    int32_t     moe_stream_record_words = 0; // the most decode words it keeps, 0 = the default (1024)
    // The prefetcher's two layers of lead (1 = the engine's own R(1,M)), the width of the
    // correction the layer in between applies to the early guess (0 = the same as prefetch_m), and the
    // two queue policies of step 3 (demand_last reverses the priority the engine already has, so that
    // what it is worth can be measured; probation makes an unused prefetch the first eviction victim)
    int32_t     moe_stream_prefetch_lead = 1;
    int32_t     moe_stream_prefetch_m2   = 0;
    int32_t     moe_stream_demand_last   = 0;
    int32_t     moe_stream_probation     = 0;
    int32_t     moe_stream_prefetch_inflight = 0;  // cap on guessed reads in flight, 0 = none
    int32_t     moe_stream_early_order = 0;       // force the early guess before the layer's work
    int32_t     moe_stream_io_wake = 0;           // 0 = wake one I/O thread per read, 1 = one per chunk
    int32_t     moe_stream_promote = 0;           // 1 = a needed expert's queued guess chunks move to the demand queue
    int32_t     moe_stream_guess_cap = -1;        // < 0 = off; else max guess chunks in service while a needed read is active
    int32_t     moe_stream_guess_cap_refill = 0;  // 1 = a finished guess chunk wakes a thread for the freed place
    float       moe_stream_evict_halflife = 0.0f; // policy decay: words, 0 = 64
    float       moe_stream_evict_credit = 0.0f;   // policy decay: router-interest credit, 0 = off
    int32_t     moe_stream_prompt_end = 0;        // 1 = recency wave order on a prompt's last ubatch
    float       moe_stream_prompt_end_halflife = 0.0f; // tokens, 0 = 128
    int32_t     moe_stream_prompt_values = 0;     // 1 = policy decay counts a prompt's expert uses as decode words
    std::string moe_stream_chance_model = "";     // probability table for the 1-ahead reads, "" = off
    float       moe_stream_read_threshold = 0.0f; // 0 = 0.3
    int32_t     moe_stream_cpu_skip_zero = 0;     // 1 = write 0 for the waves' zero slices instead of computing them
    int32_t     moe_stream_cpu_x4        = 0;     // 4-row MXFP4 kernel (same bits): 0 off, 1 256-bit, 2 512-bit registers
    int32_t     moe_stream_cpu_x4_pf     = -1;    // its prefetch: -1 the next 4-row group, 0 none, N blocks ahead in the row
    int32_t     moe_stream_cpu_fuse      = 0;     // 1 = fused up+gate+swiglu step for one token (same bits)
    int32_t     moe_stream_cpu_dispatch  = 0;     // 1 = ik's expert split for one-token mul_mat_id (same bits)
    int32_t     moe_stream_cpu_pieces    = 1;     // work pieces per thread and expert, one-token expert steps (same bits)
    int32_t     moe_stream_cpu_conv1     = 0;     // 1 = the one-token activation row converted by one thread (same bits)
    int32_t     moe_stream_cpu_wsum      = 0;     // 1 = CPU layers' expert weighted sum as one op (same bits)
    int32_t     moe_stream_cpu_state_inplace = 0; // 1 = CPU layers' delta step on the state in place (same bits)
    int32_t     moe_stream_cpu_mmid_merge  = 0;   // 1 = the tiled MUL_MAT_ID's groups of a thread as one window (same bits)
    int32_t     moe_stream_cpu_iqk         = 0;   // N > 0 = ik's expert kernels at prompt sizes for experts with N+ rows
    int32_t     moe_stream_guess_skip_full = 0;   // 1 = no guess for target layers that hold every expert in RAM
    int32_t     moe_stream_guess_skip_full_gpu = 0;   // 1 = the GPU's guess skips them too
    int32_t     moe_stream_cpu_fuse_small = 0;    // a mask of fewer small CPU steps (1, 2, 64) - the same results
    int32_t     moe_stream_cpu_fa        = 0;     // CPU flash attention with q8_0 KV: 1 = tiled prompts + the GQA decode split
    int32_t     moe_stream_cpu_q8_tiled  = 0;     // N > 0 = q8_0 weights on the CPU's tiled matmul (1 = from 12 rows, else from N rows; prompts)
    int32_t     moe_stream_cpu_topk      = 0;     // 1 = the router / guess top k in one pass with no GPU device (same ids)
    int32_t     moe_stream_cpu_down_scale = 0;    // 1 = skip Flint's down-input scale step where it is exact on the CPU
    int32_t     moe_stream_cpu_small_ops = 0;     // 1 = the expert pick and CONCAT on all threads (same bits)
    int32_t     moe_stream_fused_route = 0;       // 1 = grouped expert selection as one op (router + guesses)
    int32_t     moe_stream_slot_check  = 1;       // 1 = a pair with no RAM slot stops the run, 0 = off (emergency only)
    int32_t     moe_stream_wave_plan   = 0;       // 1 = a multi-wave ubatch's waves from a plan built at wave 0
    int32_t     moe_stream_wave_victim = 0;       // 1 = wave marks last the call (earlier-wave victims first)
    int32_t     moe_stream_conv_silu   = 0;       // 1 = KDA conv + SiLU fused on CUDA (same bits)
    int32_t     moe_stream_hc_share    = 0;       // 1 = an FFN hyper-connection combine's weights computed once (same bits)
    int32_t     moe_stream_rs_window   = 0;       // 1 = delta-net state fetch / conv-state write in the GPU window (same bits)
    int32_t     moe_stream_ple_late    = 0;       // 1 = qwen4exp's n-gram rows gathered right before the layer that reads them
    int32_t     moe_stream_hc_silu     = 0;       // 1 = a hyper-connection mix's scale + SiLU as one CUDA kernel (same bits)
    int32_t     moe_stream_hc_q8       = 0;       // 1 = a hyper-connection mix's norm (and SiLU) write the matvecs' q8_1 copies
    int32_t     moe_stream_rs_inplace  = 0;       // 1 = qwen4exp's delta step reads its state row in place (same bits)
    int32_t     moe_stream_gpu_share_q8 = 0;      // 1 = CUDA matvecs share one q8_1 copy of the same input (same bits)
    int32_t     moe_stream_vram_experts = 0;      // N most-picked experts per streamed layer kept in VRAM, 0 = off
    std::string moe_stream_vram_experts_file = ""; // their ranked list (tokenwing vram-experts v1)
    int64_t     moe_stream_vram_experts_margin = 0; // VRAM that must stay free after the pool, bytes, 0 = 512 MiB
    int64_t     moe_stream_vram_free = 0;           // end-state keep-free that sizes the pool's N, bytes, 0 = off
    int64_t     moe_stream_vram_scratch = -1;       // the op scratch pool mapped before the VRAM check, bytes; -1 = derived from the model, 0 = not mapped
    int64_t     moe_stream_vram_growth = 0;         // what the CUDA runtime adds lazily after the load-time check, bytes, 0 = derived
    std::string moe_stream_correction = "";         // the trained correction file of the router-trick guess, "" = off
    int32_t     moe_stream_idle_load = 0;           // 1 = idle-time loading of empty RAM slots from the usage table; 2 = also every RAM slot filled before the model counts as loaded
    std::string moe_stream_usage_file = "";         // the usage table's path, "" = <sidecar>.usage.json
    int32_t     moe_stream_idle_pause_ms = 0;       // no call for this long = idle, 0 = 500
    int32_t     moe_stream_read_ahead = 0;          // 1 = read the next layer's most-used experts during a big prompt call
    int32_t     moe_stream_read_ahead_min = 0;      // only in calls of at least this many tokens, 0 = 4096
    int32_t     moe_stream_window = 0;              // 1 = the RAM as one window over the layers in a big layer-major prompt call
    float       moe_stream_window_keep = 0.0f;      // a finished layer keeps this share during the call, 0 = 1
    int32_t     moe_stream_prompt_defer = 0;        // 1 = CPU-matmul prompt calls leave the wait for their reads to mul_mat_id
    int32_t     moe_stream_prompt_split = 0;        // 1 = a CPU-wave prompt's VRAM pass beside the CPU split
    int32_t     moe_stream_prompt_split_places = 128;   // the tail places a prompt call borrows
    int32_t     moe_stream_gpu_wave_skip_zero = 0;  // 1 = GPU prompt matmuls skip the zero-slice pairs (they write 0)
    int32_t     moe_stream_gpu_skip_zero = 0;     // 1 = the GPU mmvq writes 0 for the VRAM pool's zero slices
    int32_t     moe_stream_plan = 0;              // 1 = the waiting list with late binding (decode words)
    int32_t     moe_stream_plan_bind_early = 0;   // test only - every list entry is bound when it is added
    int32_t     moe_stream_drop_stale = 0;        // with plan 1, unpicked entries / started reads dropped at the target
    int32_t     moe_stream_needed_early = 0;      // with plan 1, the 1-ahead guess's adds made by a helper thread
    int32_t     moe_stream_behind_backlog = 0;    // with plan 1, "behind" = at least this many reads not finished (0 = off)
    int32_t     moe_stream_behind_rank    = 0;    // while behind, entries ranked above this wait (0 = off)
    int32_t     moe_stream_guess_inflight = 0;    // with plan 1, guess pieces in service while no needed read is (0 = no limit)
    llama_moe_stream_srv moe_stream_srv = {};     // set by llama-server (its snapshots / restore buffer in the
    int32_t     moe_stream_nonpool_measured = 0;                 // 1 = the non-pool memory from a record of this run's own needs
    int64_t     moe_stream_nonpool_growth   = (int64_t) 128 << 20;   // on top of the record
    std::string moe_stream_nonpool_args;                         // the --moe-stream flags of the command line that shape the graph (the record's key)
    llama_moe_stream_nonpool moe_stream_nonpool = {};            // filled in common_init
    bool        moe_stream_nonpool_set = false;
    bool        moe_stream_srv_set = false;       //   engine's RAM plan); other tools leave it unset
    int64_t     moe_stream_other_rest = 0;        // llama-server, the measured rest of the process in bytes, 0 = default
    int32_t     moe_stream_guess_top      = 0;    // with plan 1, ranks 1..K of a decode word's 1-ahead guess on the list (0 = off)
    int32_t     moe_stream_lookahead      = 2;    // guesses for the layers 1..N ahead (1 = the 1-ahead guess only)
    bool        moe_stream_lookahead_set  = false; // given on the command line: a default that cannot run falls back without a line
    int32_t     moe_stream_guess_pace     = 1;    // new entries per far guess and target
    int32_t     moe_stream_plan_old_order = 0;    // test only - the list's old order
    int32_t     moe_stream_plan_extra_near = 0;   // test only - order A for the extra ranks
    int32_t     moe_stream_guess_combined = 0;    // a layer's N guesses in one step (bit-identical)
    int32_t     moe_stream_protect_named = 0;     // a named READY expert is no guess read's victim
    int32_t     moe_stream_wordstart = 0;         // word-start guesses from the last D layers before the first streamed one
    int32_t     moe_stream_proj_ready = 0;        // per-projection ready flags, a needed read = gate+up then down
    int32_t     moe_stream_vram_follow = 0;       // the VRAM experts move with the text
    int32_t     moe_stream_elastic     = 0;       // elastic RAM
    int64_t     moe_stream_ram_margin  = 0;       // RAM left free when growing (0 = 500 MiB)
    std::string moe_stream_elastic_opts = "";     // key=value tunables
    int32_t     moe_stream_shares      = 0;       // moving slot shares per layer
    std::string moe_stream_share_file  = "";      // a starting split
    std::string moe_stream_share_opts  = "";      // key=value tunables
    int32_t     moe_stream_layer_major = 0;       // a prompt call of 2+ ubatches runs layer by layer
    int32_t     moe_stream_layer_major_batch = 0; // the cap of the engine's n_batch raise (0 = the context size)
    int32_t     moe_stream_vram_end_fill     = 0; // the GPU tail refilled at a prompt's end (0 = off, 1, 2)
    int32_t     moe_stream_copy_ahead        = 0; // GPU prompts copy the next layer's experts ahead (0 = off, 1)
    int32_t     moe_stream_tail_wave_w       = 0; // > 0 = cap GPU tail wave width to N experts (0 = off)
    int32_t     moe_stream_vram_lend_prompt  = 0; // a prompt's big GPU tensors in lent VRAM pool places (0 = off, 1)
    int32_t     moe_stream_vram_lend_short   = 64; // prompt calls of at most N tokens in the real compute buffer
    int32_t     moe_stream_timing            = 0; // 1 = measure where a call's time goes (host, device, copies, syncs), report it per word
    int32_t     moe_stream_out_check         = 16; // the output's NaN / Inf check: every prompt call + every Nth reply call, 0 = off
    std::string moe_stream_vram_device;           // the GPU that holds the VRAM expert pool, "" = the first GPU
    std::string moe_stream_gpu_layers_device;     // whole MoE layers per GPU, "DEV=LIST;DEV=LIST" (repeated flags joined with ';'); LIST may be "auto"
    std::string moe_stream_vram_free_dev;         // --moe-stream-vram-free per GPU, "DEV=bytes,DEV=bytes"; "" = one value for the pool's GPU
    int64_t     moe_stream_vram_chunk        = 0; // max bytes per VRAM expert pool buffer group (1 = auto <= 2 GiB, 0 = off)
    int32_t     moe_stream_uma               = 0; // unified-memory zero-copy streaming mode (0 = off, 1)
    int32_t     moe_stream_hip_mmvq          = 0; // RDNA2/3/3.5/4 short-row & RDNA3.5 multi-warp MMVQ under ROCm/HIP (0 = off, 1)
    int32_t     moe_stream_hip_sync          = 0; // ROCm/HIP CPU-GPU handoff spin window in us (1 = auto 400 us, 0 = off)
    int32_t     moe_stream_guess_gate        = 0; // bind-time victim- & queue-priced guesser gate (0 = off, 1)
    int32_t     moe_stream_ssd_latency_us    = 0; // single-read SSD latency in us for congestion pricing (0 = auto)
    float       moe_stream_ssd_gbps          = 0.0f; // sequential/chunk SSD bandwidth in GB/s for congestion pricing (0 = auto)
    int32_t     moe_stream_vram_follow_tick = 8;
    int32_t     moe_stream_vram_follow_swaps = 64;
    int32_t     moe_stream_vram_follow_recent = 256;
    int32_t     moe_stream_vram_follow_longterm = 1024;
    float       moe_stream_vram_follow_ratio = 1.25f;
    int32_t     moe_stream_vram_follow_margin = 4;
    int32_t     moe_stream_vram_follow_fixed = 0;   // test only
    int32_t     moe_stream_vram_follow_pinned = 1;
    int32_t     moe_stream_vram_parallel = 0;     // the device pass during the CPU experts
    int32_t     moe_stream_vram_copyback = 0;     // VRAM follow's leaving experts copied back into RAM
    int32_t     moe_stream_async_h2d = 0;         // CPU -> GPU split inputs as async H2D copies
    int32_t     moe_stream_overlap = 0;      // compute the resident experts during the miss reads
    int64_t     moe_stream_ram = 0;        // RAM budget for the pools in bytes, 0 = slots decide
    bool        moe_stream_plan_only = false; // print the plan and stop
    std::string moe_stream_oracle = "";    // trace prefix to replay instead of predicting
    int32_t     moe_stream_oracle_lead = -1; // oracle lookahead in MoE layers, -1 = 1
    std::string quant_plugin = "";         // shared library (.so / .dll) registering GGML_TYPE_EXT0..EXT5
    // optional callback for model loading progress and cancellation:
    // called with a progress value between 0.0 and 1.0.
    // return false from callback to abort model loading or true to continue
    llama_progress_callback load_progress_callback = NULL;
    void *                  load_progress_callback_user_data = NULL;
    bool no_alloc = false; // Don't allocate model buffers

    // TTS params
    std::string tts_lang = "";
    std::string tts_speaker_file = "";

    bool is_gen_docs = false; // whether we are running inside llama-gen-docs
};

// call once at the start of a program if it uses libcommon
// initializes the logging system and prints info about the build
void common_init();

void common_params_print_info(const common_params & params, bool print_devices = true);
std::string common_params_get_system_info(const common_params & params);

bool parse_cpu_range(const std::string & range, bool(&boolmask)[GGML_MAX_N_THREADS]);
bool parse_cpu_mask(const std::string & mask, bool(&boolmask)[GGML_MAX_N_THREADS]);
void postprocess_cpu_params(common_cpu_params & cpuparams, const common_cpu_params * role_model = nullptr);
bool set_process_priority(enum ggml_sched_priority prio);

//
// String utils
//

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define LLAMA_COMMON_ATTRIBUTE_FORMAT(...)
#endif

LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
std::string string_format(const char * fmt, ...);

std::string string_strip(const std::string & str);
std::string string_get_sortable_timestamp();
std::string string_lcs(std::string_view a, std::string_view b);

std::string string_join(const std::vector<std::string> & values, const std::string & separator);
std::vector<std::string> string_split(const std::string & str, const std::string & delimiter);
std::string string_repeat(const std::string & str, size_t n);

void string_replace_all(std::string & s, const std::string & search, const std::string & replace);

std::string regex_escape(const std::string & s);

template<class T>
static std::vector<T> string_split(const std::string & str, char delim) {
    static_assert(!std::is_same<T, std::string>::value, "Please use the specialized version for std::string");
    std::vector<T> values;
    std::istringstream str_stream(str);
    std::string token;
    while (std::getline(str_stream, token, delim)) {
        T value;
        std::istringstream token_stream(token);
        if (!(token_stream >> value)) {
            throw std::invalid_argument("invalid value: \"" + token + "\"");
        }
        values.push_back(value);
    }
    return values;
}

template<>
inline std::vector<std::string> string_split<std::string>(const std::string & str, char delim)
{
    std::vector<std::string> parts;
    size_t begin_pos = 0;
    size_t delim_pos = str.find(delim);
    while (delim_pos != std::string::npos) {
        std::string part = str.substr(begin_pos, delim_pos - begin_pos);
        parts.emplace_back(part);
        begin_pos = delim_pos + 1;
        delim_pos = str.find(delim, begin_pos);
    }
    parts.emplace_back(str.substr(begin_pos));
    return parts;
}

// remove when moving to c++20
inline bool string_starts_with(std::string_view str, std::string_view prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
}

// remove when moving to c++20
inline bool string_starts_with(std::string_view str, char prefix) {
    return !str.empty() && str.front() == prefix;
}

// remove when moving to c++20
inline bool string_ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool string_remove_suffix(std::string & str, std::string_view suffix) {
    if (string_ends_with(str, suffix)) {
        str.resize(str.size() - suffix.size());
        return true;
    }
    return false;
}

inline size_t string_find_partial_stop(std::string_view str, std::string_view stop) {
    if (!str.empty() && !stop.empty()) {
        const size_t max_len = std::min(str.size(), stop.size());
        const char last_char = str.back();
        for (size_t len = max_len; len > 0; --len) {
            if (stop[len - 1] == last_char) {
                if (string_ends_with(str, stop.substr(0, len))) {
                    return str.size() - len;
                }
            }
        }
    }
    return std::string::npos;
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides);
void string_process_escapes(std::string & input);

std::string string_from(bool value);
std::string string_from(const std::vector<int> & values);
std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens);

bool glob_match(const std::string & pattern, const std::string & str);

//
// Unicode utils
//

#ifdef _WIN32
std::wstring utf8_to_wstring(const std::string & str);
std::string  wstring_to_utf8(const std::wstring & str);
#endif

// returns the path as a UTF-8 string, preserving its separators
std::string fs_path_to_utf8(const std::filesystem::path & path);

//
// Environment utils
//

// portable environment access, an unset variable reads as an empty string
// and setting an empty value unsets the variable
std::string common_get_env(const std::string & name);
void        common_set_env(const std::string & name, const std::string & value);

// reads a path from the environment, an unset variable gives an empty path
std::filesystem::path common_get_path_from_env(const std::string & name);

//
// Filesystem utils
//

bool fs_validate_filename(const std::string & filename, bool allow_subdirs = false);
bool fs_is_directory(const std::string & path);

// some old libstdc++ versions don't follow symlinks here, so adding a trailing "/" fixes it: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=101510
inline bool common_create_directories(const std::filesystem::path & path, std::error_code & ec) {
#if defined(__linux__)
    return std::filesystem::create_directories(path / "", ec);
#else
    return std::filesystem::create_directories(path, ec);
#endif
}

std::filesystem::path fs_get_cache_directory();
std::filesystem::path fs_get_cache_file(const std::string & filename);
std::filesystem::path fs_get_config_directory();

void fs_write_atomic(const std::filesystem::path & path, const std::string & data);

//
// TTY utils
//

// Auto-detect if colors can be enabled based on terminal and environment
bool tty_can_use_colors();
bool tty_enable_ansi(); // false when stdout or stderr is a console that cannot render ANSI sequences

// Check if the given file is attached to a terminal
bool common_is_tty(FILE * file);

//
// Model utils
//

struct common_sampler;

// note: defines the model, context, samplers, ets. lifetimes
struct common_init_result {
    // set when --moe-stream-plan-only printed the plan and stopped: there is no context and that is fine
    bool plan_only = false;

    common_init_result(common_params & params, bool model_only = false);
    ~common_init_result();

    llama_model * model();
    llama_context * context();

    common_sampler * sampler(llama_seq_id seq_id);
    void reset_samplers();

    std::vector<llama_adapter_lora_ptr> & lora();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_init_result_ptr = std::unique_ptr<common_init_result>;

common_init_result_ptr common_init_from_params(common_params & params, bool model_only = false);

struct llama_model_params   common_model_params_to_llama  (      common_params & params);
struct llama_context_params common_context_params_to_llama(const common_params & params);

// clear LoRA adapters from context, then apply new list of adapters
void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora);

// model endpoint from env
std::string common_get_model_endpoint();

// for testing purposes
char * common_get_model_or_exit(int, char*[]);

//
// Threadpool utils
//

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params);

struct common_threadpools {
    common_threadpools() = default;
    ~common_threadpools();

    common_threadpools(const common_threadpools &) = delete;
    common_threadpools & operator=(const common_threadpools &) = delete;

    void init(llama_context * ctx, const common_params & params);

private:
    ggml_threadpool * threadpool       = nullptr;
    ggml_threadpool * threadpool_batch = nullptr;

    decltype(ggml_threadpool_free) * free_fn = nullptr;
};

//
// Context utils
//

enum common_context_seq_rm_type {
    COMMON_CONTEXT_SEQ_RM_TYPE_NO           = 0, // seq_rm not supported (e.g. no memory module)
    COMMON_CONTEXT_SEQ_RM_TYPE_PART         = 1, // can seq_rm partial sequences
    COMMON_CONTEXT_SEQ_RM_TYPE_FULL         = 2, // can seq_rm full sequences only
    COMMON_CONTEXT_SEQ_RM_TYPE_RS = 3, // can seq_rm partial sequences, bounded by n_rs_seq
};

// check if the llama_context can remove sequences
// note: clears the memory of the context
common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx);

struct common_memory {
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    void init(llama_context * ctx_tgt, llama_context * ctx_dft = nullptr);

    // aborts execution on failure
    void seq_rm (llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) const;
    void seq_cp (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) const;
};

//
// Batch utils
//

// wrapper around llama_batch_ext that provide getter functions for downstream code
// entries can exceed n_batch, use get_sub_batch() to decode them in chunks
struct common_batch {
    struct token {
        llama_token  id;
        std::array<llama_pos, GGML_MROPE_SECTIONS> pos; // only pos[0] is used for text tokens
        llama_seq_id seq_id; // the first sequence id, see add_seq()
        bool         output;
        llama_embd   embd; // non-owning view of the data passed to add_embd()/set_embd(), data == NULL if none
        llama_embd   state; // non-owning view of the data passed to set_embd_state(), data == NULL if none
        std::vector<llama_seq_id> seq_ids_extra; // see add_seq()
    };

    std::vector<token> tokens; // mirror of the entries, tokens[i] describes batch index i
    llama_batch_ext_ptr batch;

    int32_t n_pos = 1; // positions per embedding entry, GGML_MROPE_SECTIONS for MROPE/IMROPE

    common_batch() = default;
    common_batch(struct llama_context * ctx);

    llama_batch_ext * get() { return get_sub_batch(0, size()); }

    // render entries [off, off + n) into batch, the result is overwritten by the next call
    llama_batch_ext * get_sub_batch(int32_t off, int32_t n);

    // content type of the batch, all entries carry the same combination
    bool has_token() const { return !tokens.empty() && tokens[0].id != LLAMA_TOKEN_NULL; }
    bool has_embd () const { return !tokens.empty() && tokens[0].embd.data != nullptr; }

    void clear();

    // returns the batch index
    int32_t add(llama_token id, llama_pos pos, llama_seq_id seq_id, bool output);

    // same, with the entry shared by all seq_ids (must not be empty)
    int32_t add(llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool output);

    // add the entry at idx to another sequence, tokens[idx].seq_id keeps the first one
    bool add_seq(int32_t idx, llama_seq_id seq_id);

    bool set_output(int32_t idx, bool value);

    // attach a token embedding to the entry at idx, can only be set once per entry
    bool set_embd(int32_t idx, llama_embd embd);

    // attach a state embedding (e.g. the target hidden state for MTP) to the entry at idx, can only be set once per entry
    bool set_embd_state(int32_t idx, llama_embd state);

    // add an embedding-only entry (no token id)
    // pos points to n_pos positions
    int32_t add_embd(llama_embd embd, const llama_pos * pos, llama_seq_id seq_id, bool output);

    int32_t size() const { return (int32_t) tokens.size(); }
};

// create a single-sequence batch from a list of tokens
// positions continue from the memory, last token always have output_logits set to true
common_batch common_batch_get_one(struct llama_context * ctx, const llama_token * tokens, int32_t n_tokens);
common_batch common_batch_get_one(struct llama_context * ctx, const llama_tokens & tokens);

// decodes a single batch of tokens for a prompt and manages session tokens
//
// Note: We save state before the last token so that we can replay it to ensure
// compatibility with all memory types. Recurrent/hybrid models cannot remove
// tokens from memory, so this approach works across all model architectures.
bool common_prompt_batch_decode(
              struct llama_context * ctx,
                const llama_tokens & all_tokens,
                               int   n_new,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state);

// replays the last token after loading state to regenerate logits
// used after loading session state to ensure the sampling context has valid logits
bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos);

//
// Vocab utils
//

// tokenizes a string into a vector of tokens
// should work similar to Python's `tokenizer.encode`
std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

// tokenizes a token into a piece, optionally renders special/control tokens
// should work similar to Python's `tokenizer.id_to_piece`
std::string common_token_to_piece(
        const struct llama_context * ctx,
                       llama_token   token,
                       bool          special = true);

std::string common_token_to_piece(
          const struct llama_vocab * vocab,
                       llama_token   token,
                       bool          special = true);

// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
// optionally renders special/control tokens
std::string common_detokenize(
            const struct llama_context * ctx,
        const std::vector<llama_token> & tokens,
                                  bool   special = true);

std::string common_detokenize(
              const struct llama_vocab * vocab,
        const std::vector<llama_token> & tokens,
                                  bool   special = true);

//
// Embedding utils
//

// TODO: replace embd_norm with an enum
void common_embd_normalize(const float * inp, float * out, int n, int embd_norm);

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n);

//
// Control vector utils
//

struct common_control_vector_data {
    int n_embd;

    // stores data for layers [1, n_layer] where n_layer = data.size() / n_embd
    std::vector<float> data;
};

struct common_control_vector_load_info {
    float strength;

    std::string fname;
};

// Load control vectors, scale each by strength, and add them together.
// Direction layer indices must be in [1, n_layer].
// On error, returns {-1, empty}
common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos, int32_t n_layer);

//
// Split utils
//

namespace {

const char * const LLM_KV_SPLIT_NO            = "split.no";
const char * const LLM_KV_SPLIT_COUNT         = "split.count";
const char * const LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count";

}

//
// FFN offload utils
//

const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

const char * const LLM_FFN_DENSE_REGEX = "\\.ffn_(up|down|gate)\\.";

inline std::string llm_ffn_block_regex(int idx, const char * ffn_regex) {
    return string_format("blk\\.%d%s", idx, ffn_regex);
}

inline llama_model_tensor_buft_override llm_ffn_exps_cpu_override() {
    return { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() };
}

inline void llm_add_n_cpu_ffn_overrides(int n, const char * ffn_regex, std::vector<llama_model_tensor_buft_override> & overrides) {
    // keep strings alive and avoid leaking memory by storing them in a static list
    static std::list<std::string> buft_override_strings;
    for (int i = 0; i < n; ++i) {
        buft_override_strings.push_back(llm_ffn_block_regex(i, ffn_regex));
        overrides.push_back({buft_override_strings.back().c_str(), ggml_backend_cpu_buffer_type()});
    }
}

//
// training utils
//

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride);

// "adamw" or "sgd" (case insensitive)
enum ggml_opt_optimizer_type common_opt_get_optimizer(const char *);

//
// prompt utils
//

struct common_prompt_checkpoint {
    int64_t n_tokens;

    // (optional) id of the task that created the checkpoint
    int id_task = -1;

    llama_pos pos_min;
    llama_pos pos_max;

    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;

    // (optional) speculative-decoding implementation state stashed with the checkpoint
    // (e.g. eagle3's deferred-boundary g_embd row)
    std::vector<uint8_t> data_spec;

    size_t size() const;

    bool empty() const;
    void clear();

    void update_pos(
            int64_t n_tokens,
            llama_pos pos_min,
            llama_pos pos_max);

    void update_tgt(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags);

    void update_dft(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags);

    // return false if the state could not be restored
    bool load_tgt(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags) const;

    bool load_dft(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags) const;

    void clear_tgt();
    void clear_dft();
};

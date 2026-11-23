#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"
#include "speculative.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

static void test(void) {
    common_params params;

    auto assert_output_limits = [](int32_t n_batch, int32_t n_parallel, int32_t n_draft,
                                   int32_t total, int32_t per_seq) {
        const auto limits = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
        assert(limits.total == total);
        assert(limits.per_seq == per_seq);
    };

    assert_output_limits(16, 2,  3, 8, 4);
    assert_output_limits(16, 2, -1, 2, 1);
    assert_output_limits( 6, 2,  3, 6, 4);
    assert_output_limits( 2, 1,  3, 2, 2);
    assert_output_limits(
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max());

    {
        common_params_speculative spec;
        spec.synth_len = 3.4;

        auto assert_invalid = [](const common_params_speculative & value, int32_t n_max) {
            try {
                common_speculative_synth_rates_resolve(&value, n_max);
                assert(false);
            } catch (const std::invalid_argument &) {
            }
        };

        const auto rates = common_speculative_synth_rates_resolve(&spec, 4);
        assert(rates.size() == 4);
        assert(std::abs(rates[0] - 0.80581) < 1e-5);
        assert(std::abs(rates[1] - 0.64933) < 1e-5);
        assert(std::abs(rates[2] - 0.52323) < 1e-5);
        assert(std::abs(rates[3] - 0.42163) < 1e-5);
        assert(std::abs(1.0 + rates[0] + rates[1] + rates[2] + rates[3] - 3.4) < 1e-8);

        spec.synth_len = 1.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({0.0, 0.0, 0.0, 0.0}));

        spec.synth_len = 5.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({1.0, 1.0, 1.0, 1.0}));

        spec.synth_len = 5.1;
        assert_invalid(spec, 4);

        spec.synth_len = std::numeric_limits<double>::quiet_NaN();
        assert_invalid(spec, 4);

        spec.synth_len = 0.0;
        assert_invalid(spec, 4);

        spec.synth_len = -1.0;
        spec.synth_rates = {0.8, 0.6, 0.4};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        assert(common_speculative_synth_rates_resolve(&spec, 4) == spec.synth_rates);

        spec.synth_rates = {0.8, 0.9, 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, std::numeric_limits<double>::quiet_NaN(), 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, -0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        spec.synth_len = 3.0;
        assert_invalid(spec, 4);
    }

    {
        common_params base;
        base.n_parallel = 4;
        base.n_outputs_max_per_seq = 8;
        base.n_ubatch = 512;

        {
            const auto draft = common_base_params_to_speculative(base);
            assert(draft.n_outputs_max == 4);
            assert(draft.n_outputs_max_per_seq == 1);
            assert(draft.n_ubatch == 512); // inherits the target by default
        }

        base.speculative.draft.n_ubatch = 8;
        {
            const auto draft = common_base_params_to_speculative(base);
            assert(draft.n_ubatch == 8);
        }
    }

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    {
        common_params penalty_params;
        assert(penalty_params.sampling.penalty_last_n == 64);
        assert(penalty_params.sampling.dry_penalty_last_n == 64);

        argv = {"binary_name", "--repeat-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--dry-penalty-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "0"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "nan"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
        const char * nonfinite_values[] = {"nan", "inf", "-inf"};
        for (const char * option : penalty_options) {
            for (const char * value : nonfinite_values) {
                argv = {"binary_name", option, value};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
            }
        }
    }

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    argv = {"binary_name", "-lm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --moe-stream-vram-free: one size (the pool's GPU) or a list per GPU; --moe-stream-gpu-layers-device with "auto"
    {
        common_params p;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream-vram-free", "200MiB"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_COMMON));
        assert(p.moe_stream_vram_free == 200ll << 20 && p.moe_stream_vram_free_dev.empty());
    }
    {
        common_params p;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream-vram-free", "CUDA0=1GiB,CUDA1=512MiB"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_COMMON));
        assert(p.moe_stream_vram_free_dev == "CUDA0=1073741824,CUDA1=536870912");
        assert(p.moe_stream_vram_free == 1ll << 30);   // no default: the largest named
    }
    {
        common_params p;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream-vram-free", "CUDA1=0,256MiB"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_COMMON));
        assert(p.moe_stream_vram_free_dev == "CUDA1=0" && p.moe_stream_vram_free == 256ll << 20);
    }
    {
        common_params p;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream-vram-free", "CUDA0=0"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_COMMON));
        assert(p.moe_stream_vram_free == 1);   // a list keeps the pool on
    }
    for (const char * bad : {"CUDA0=", "=1GiB", "1GiB,2GiB", "CUDA0=1XB"}) {
        common_params p;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream-vram-free", bad};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_COMMON));
    }
    {
        common_params p;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream-gpu-layers-device", "CUDA1=30-32", "--moe-stream-gpu-layers-device", "CUDA2=5"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_COMMON));
        assert(p.moe_stream_gpu_layers_list == "30,31,32,5");
    }
    {
        common_params p;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream-gpu-layers-device", "CUDA1=auto;CUDA2=auto"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_COMMON));
        assert(p.moe_stream_gpu_layers_device == "CUDA1=auto;CUDA2=auto" && p.moe_stream_gpu_layers_list.empty());
    }
    {
        common_params p;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream-gpu-layers-device", "CUDA1=auto"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_COMMON));
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream-gpu-layers-device", "CUDA1=autox"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_COMMON));
    }

    // --moe-stream umbrella groups: default ON when --moe-stream is set, cleanly overridable
    {
        common_params p;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream", "model"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_COMMON));
        assert(p.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);
        assert(p.n_ubatch == 2048);
        assert(p.moe_stream_prefill_fast == 1);
        assert(p.moe_stream_layer_major == 1);
        assert(p.moe_stream_layer_major_batch == 32768);
        assert(p.moe_stream_wave_plan == 1);
        assert(p.moe_stream_read_ahead == 1);
        assert(p.moe_stream_window == 1);
        assert(p.moe_stream_prompt_defer == 1);
        assert(p.moe_stream_gpu_waves == 512);
        assert(p.moe_stream_gpu_wave_skip_zero == 1);
        assert(p.moe_stream_prompt_split == 1);
        assert(p.moe_stream_vram_end_fill == 1);
        assert(p.moe_stream_copy_ahead == 1);
        assert(p.moe_stream_vram_lend_prompt == 1);
        assert(p.moe_stream_overlap == 1);
        assert(p.moe_stream_proj_ready == 1);
        assert(p.moe_stream_io_wake == 1);
        assert(p.moe_stream_chunk_prefetch == 512 * 1024);
        assert(p.moe_stream_promote == 1);
        assert(p.moe_stream_guess_cap == 2);
        assert(p.moe_stream_pred_shared == 1);
        assert(p.moe_stream_plan == 1);
        assert(p.moe_stream_drop_stale == 1);
        assert(p.moe_stream_needed_early == 1);
        assert(p.moe_stream_lookahead == 2);
        assert(p.moe_stream_guess_combined == 1);
        assert(p.moe_stream_wordstart == 2);
        assert(p.moe_stream_protect_named == 1);
        assert(p.moe_stream_guess_inflight == 4);
        assert(p.moe_stream_correction == "model");
        assert(p.moe_stream_cpu_skip_zero == 1);
        assert(p.moe_stream_cpu_x4 == 1);
        assert(p.moe_stream_cpu_iqk == 8);
        assert(p.moe_stream_policy == "decay");
        assert(p.moe_stream_prompt_end == 1);
        assert(p.moe_stream_idle_load == 2);
        assert(p.moe_stream_vram_free == (200LL << 20));
        assert(p.moe_stream_vram_experts_file == "model");
        assert(p.moe_stream_vram_follow == 1);
        assert(p.moe_stream_vram_parallel == 1);
        assert(p.moe_stream_vram_copyback == 1);
        assert((p.moe_stream_auto_mask & 0x7Fu) == 0x7Fu);
        assert(p.moe_stream_gpu_layers_floor == (2500LL << 20));
        assert(p.moe_stream_gpu_layers_lend_reclaim == (640LL << 20));
        assert(p.moe_stream_copy_ahead_one_below == (10LL << 30));
        assert(p.moe_stream_gpu_layers_uniform_max == -1);
    }

    // 262k context auto-scales batch/ubatch under --moe-stream model
    {
        common_params p;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream", "model", "-c", "262144"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_COMMON));
        assert(p.n_batch == 1024);
        assert(p.n_ubatch == 1024);
    }

    // Server auto-defaults under --moe-stream model (and explicit overrides)
    {
        common_params p;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream", "model"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_SERVER));
        assert(p.n_parallel == 1);
        assert(p.cache_ram_mib == 0);
        assert(p.cache_idle_slots == false);
        assert(p.sampling.backend_sampling == true);

        common_params p2;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream", "model", "-np", "4", "-cram", "1024"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p2, LLAMA_EXAMPLE_SERVER));
        assert(p2.n_parallel == 4);
        assert(p2.cache_ram_mib == 1024);
    }

    // Group disables, internal overrides, and elastic->window guard
    {
        common_params p;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream", "model",
                "--moe-stream-fast-prefill", "0", "--moe-stream-vram-cache", "0",
                "--moe-stream-fast-cpu", "0", "--moe-stream-ram-cache", "0"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p, LLAMA_EXAMPLE_COMMON));
        assert(p.moe_stream_prefill_fast == 0);
        assert(p.moe_stream_layer_major == 0);
        assert(p.moe_stream_vram_free == 0);
        assert(p.moe_stream_vram_follow == 0);
        assert(p.moe_stream_cpu_iqk == 0);
        assert(p.moe_stream_idle_load == 0);
        assert((p.moe_stream_auto_mask & (1u << 6)) == 0);

        common_params p_ov;
        argv = {"binary_name", "-m", "m.gguf", "--moe-stream", "model",
                "--moe-stream-copy-ahead", "2", "--moe-stream-gpu-layers", "0", "--moe-stream-elastic", "1",
                "--moe-stream-gpu-layers-floor", "3GiB", "--moe-stream-gpu-layers-cap", "0.40",
                "--moe-stream-gpu-layers-uniform-max", "4", "--moe-stream-gpu-layers-lend-reclaim", "512MiB",
                "--moe-stream-copy-ahead-one-below", "12GiB"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), p_ov, LLAMA_EXAMPLE_COMMON));
        assert(p_ov.moe_stream_copy_ahead == 2);
        assert(p_ov.moe_stream_gpu_layers == 0);
        assert(p_ov.moe_stream_elastic == 1);
        assert(p_ov.moe_stream_window == 0);
        assert(p_ov.moe_stream_gpu_layers_floor == (3LL << 30));
        assert(p_ov.moe_stream_gpu_layers_uniform_max == 4);
        assert(p_ov.moe_stream_gpu_layers_lend_reclaim == (512LL << 20));
        assert(p_ov.moe_stream_copy_ahead_one_below == (12LL << 30));
        assert((p_ov.moe_stream_auto_mask & (1u << 1)) == 0);
    }

    // is_internal flag separation for --help vs --help-internal
    {
        common_params p;
        auto ctx_arg = common_params_parser_init(p, LLAMA_EXAMPLE_COMMON);
        auto find_internal = [&](const char * name, bool & found, bool & internal) {
            found = false;
            for (const auto & opt : ctx_arg.options) {
                for (const char * a : opt.args) {
                    if (a && strcmp(a, name) == 0) {
                        found = true;
                        internal = opt.is_internal;
                        return;
                    }
                }
            }
        };
        bool f1 = false, i1 = true, f2 = false, i2 = true, f3 = false, i3 = false, f4 = false, i4 = false;
        find_internal("--moe-stream", f1, i1);
        find_internal("--moe-stream-fast-prefill", f2, i2);
        find_internal("--moe-stream-copy-ahead", f3, i3);
        find_internal("--moe-stream-gpu-layers-floor", f4, i4);
        assert(f1 && !i1);
        assert(f2 && !i2);
        assert(f3 && i3);
        assert(f4 && i4);
    }

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_len == 3.4);
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-rates", "0.8,0.6,0.2"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_rates == std::vector<double>({0.8, 0.6, 0.2}));
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4x"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
    }

    argv = {"binary_name", "-lm", "none"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);

    argv = {"binary_name", "-lm", "mmap"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    argv = {"binary_name", "-lm", "mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    argv = {"binary_name", "-lm", "mmap+mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    argv = {"binary_name", "-lm", "dio"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_LOAD_MODE", "none", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}

int main(void) {
    try {
        test();
    } catch (std::exception & e) {
        fprintf(stderr, "test-arg-parser: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}

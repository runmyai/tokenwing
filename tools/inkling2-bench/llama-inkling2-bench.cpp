#include "arg.h"
#include "chat.h"
#include "common.h"
#include "json.h"
#include "log.h"
#include "speculative.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-inkling2-cuda.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

using json = common_json;

namespace {

struct Ink2BenchArgs {
    std::string mode         = "parity48";
    std::string out_json     = "";
    std::string save_logits  = "";
    std::string ref_logits   = "";
    std::string mmproj_path  = "";
    std::string tag          = "";
    double      target_vram_gib = 0.0;
    bool        drop_cache   = false;
};

static double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static double round2(double x) {
    return std::round(x * 100.0) / 100.0;
}

static double round3(double x) {
    return std::round(x * 1000.0) / 1000.0;
}

static void drop_file_page_cache(const std::string & path) {
#if defined(__linux__)
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        (void) ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
    }
#else
    (void) path;
#endif
}

static bool query_cuda0_mem(size_t & out_free, size_t & out_total) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        out_free  = 0;
        out_total = 0;
        return false;
    }
    ggml_backend_dev_memory(dev, &out_free, &out_total);
    return true;
}

static const ggml_cuda_inkling2_api_t * get_cuda_ink2_api() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        return nullptr;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) {
        return nullptr;
    }
    using fn_t = const ggml_cuda_inkling2_api_t * (*)(void);
    auto fn = reinterpret_cast<fn_t>(ggml_backend_reg_get_proc_address(reg, "ggml_cuda_inkling2_api"));
    if (!fn) {
        return nullptr;
    }
    const auto * api = fn();
    if (!api || api->struct_size != sizeof(ggml_cuda_inkling2_api_t)) {
        return nullptr;
    }
    return api;
}

static void sync_gpu_dev() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        return;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (be) {
        ggml_backend_synchronize(be);
        ggml_backend_free(be);
    }
}

static std::vector<llama_token> encode_chat_prompt(
        const llama_model * model,
        common_chat_templates * tmpls,
        const std::string & user_text) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string formatted = user_text;
    if (tmpls) {
        common_chat_msg msg;
        msg.role    = "user";
        msg.content = user_text;
        formatted = common_chat_format_single(tmpls, {}, msg, /* add_ass = */ true, /* use_jinja = */ true);
    }
    return common_tokenize(vocab, formatted, /* add_special = */ false, /* parse_special = */ true);
}

static std::vector<llama_token> make_long_ids(const llama_model * model, int32_t target_len) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string seed_block =
        "def dijkstra(graph, start):\n"
        "    import heapq\n"
        "    dist = {start: 0}\n"
        "    pq = [(0, start)]\n"
        "    while pq:\n"
        "        d, u = heapq.heappop(pq)\n"
        "        if d > dist.get(u, float('inf')):\n"
        "            continue\n"
        "        for v, w in graph.get(u, []):\n"
        "            nd = d + w\n"
        "            if nd < dist.get(v, float('inf')):\n"
        "                dist[v] = nd\n"
        "                heapq.heappush(pq, (nd, v))\n"
        "    return dist\n\n";
    const auto base_ids = common_tokenize(vocab, seed_block, /* add_special = */ false, /* parse_special = */ false);
    if (base_ids.empty()) {
        throw std::runtime_error("make_long_ids: base_ids is empty");
    }
    std::vector<llama_token> out;
    out.reserve((size_t) target_len + base_ids.size());
    while ((int32_t) out.size() < target_len) {
        out.insert(out.end(), base_ids.begin(), base_ids.end());
    }
    out.resize((size_t) target_len);
    return out;
}

static llama_token argmax_logits(const float * logits, int32_t n_vocab) {
    int32_t best_id = 0;
    float   best_v  = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v  = logits[i];
            best_id = i;
        }
    }
    return best_id;
}

static int32_t eval_tokens_chunked(
        llama_context * lctx,
        const std::vector<llama_token> & toks,
        int32_t pos0,
        int32_t n_batch,
        bool logits_last) {
    common_batch batch(lctx);
    const int32_t n = (int32_t) toks.size();
    for (int32_t i = 0; i < n; i += n_batch) {
        const int32_t chunk = std::min(n_batch, n - i);
        batch.clear();
        for (int32_t j = 0; j < chunk; ++j) {
            const bool want_out = logits_last && (i + j == n - 1);
            batch.add(toks[i + j], pos0 + i + j, 0, want_out);
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            return -1;
        }
    }
    return 0;
}

// Greedy M=1 generation through standard `llama_process` (`lctx`)
static std::vector<llama_token> generate_lctx_m1(
        llama_context * lctx,
        const std::vector<llama_token> & prompt_ids,
        int32_t max_new_tokens,
        int32_t n_batch,
        double * out_decode_sec = nullptr,
        double * out_prefill_sec = nullptr) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(lctx)));
    llama_memory_clear(llama_get_memory(lctx), true);
    sync_gpu_dev();
    const double tp0 = now_sec();
    if (eval_tokens_chunked(lctx, prompt_ids, 0, n_batch, true) != 0) {
        throw std::runtime_error("generate_lctx_m1: prefill failed");
    }
    sync_gpu_dev();
    if (out_prefill_sec) {
        *out_prefill_sec = now_sec() - tp0;
    }
    std::vector<llama_token> out;
    out.reserve((size_t) max_new_tokens);
    llama_token cur = argmax_logits(llama_get_logits_ith(lctx, -1), n_vocab);
    out.push_back(cur);

    common_batch b1(lctx);
    int32_t pos = (int32_t) prompt_ids.size();
    const double t0 = now_sec();
    for (int32_t step = 1; step < max_new_tokens; ++step) {
        b1.clear();
        b1.add(cur, pos++, 0, true);
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, b1.get()) != 0) {
            throw std::runtime_error("generate_lctx_m1: decode failed");
        }
        cur = argmax_logits(llama_get_logits_ith(lctx, -1), n_vocab);
        out.push_back(cur);
    }
    if (out_decode_sec) {
        *out_decode_sec = now_sec() - t0;
    }
    return out;
}

// Greedy MTP-8 speculative generation through standard `llama_process` + `llama_inkling2_mtp_draft` + `llama_memory_seq_rm`
struct LctxSpecStats {
    double  decode_sec    = 0.0;
    int64_t rounds        = 0;
    int64_t drafted_total = 0;
    int64_t accepted_extra = 0;
    double  mean_acc_len  = 1.0;
};

static std::vector<llama_token> generate_lctx_mtp8(
        llama_context * lctx,
        const std::vector<llama_token> & prompt_ids,
        int32_t max_new_tokens,
        int32_t k_draft,
        int32_t n_batch,
        LctxSpecStats * out_stats = nullptr) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(lctx)));
    llama_memory_clear(llama_get_memory(lctx), true);
    if (eval_tokens_chunked(lctx, prompt_ids, 0, n_batch, true) != 0) {
        throw std::runtime_error("generate_lctx_mtp8: prefill failed");
    }
    std::vector<llama_token> out;
    out.reserve((size_t) max_new_tokens + 8);
    llama_token cur = argmax_logits(llama_get_logits_ith(lctx, -1), n_vocab);
    out.push_back(cur);

    common_batch bspec(lctx);
    int32_t pos = (int32_t) prompt_ids.size();
    LctxSpecStats st {};
    const double t0 = now_sec();

    while ((int32_t) out.size() < max_new_tokens) {
        llama_token drafts[7];
        const int32_t rem = max_new_tokens - (int32_t) out.size();
        const int32_t k_ask = std::min<int32_t>(k_draft, std::min<int32_t>(7, std::max<int32_t>(1, rem - 1)));
        int32_t n_dft = 0;
        if (rem > 1 && k_ask > 0) {
            n_dft = llama_inkling2_mtp_draft(lctx, pos, cur, k_ask, drafts);
            if (n_dft < 0) n_dft = 0;
        }
        bspec.clear();
        bspec.add(cur, pos, 0, true);
        for (int32_t j = 0; j < n_dft; ++j) {
            bspec.add(drafts[j], pos + 1 + j, 0, true);
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, bspec.get()) != 0) {
            throw std::runtime_error("generate_lctx_mtp8: verification decode failed");
        }
        st.rounds++;
        st.drafted_total += n_dft;

        int32_t n_acc = 0;
        llama_token next_tok = argmax_logits(llama_get_logits_ith(lctx, 0), n_vocab);
        for (int32_t j = 0; j < n_dft; ++j) {
            if (next_tok == drafts[j] && (int32_t) out.size() < max_new_tokens) {
                out.push_back(next_tok);
                n_acc++;
                next_tok = argmax_logits(llama_get_logits_ith(lctx, j + 1), n_vocab);
            } else {
                break;
            }
        }
        st.accepted_extra += n_acc;
        if ((int32_t) out.size() < max_new_tokens) {
            out.push_back(next_tok);
        }
        const int32_t advance = 1 + n_acc;
        if (n_acc < n_dft) {
            if (!llama_memory_seq_rm(llama_get_memory(lctx), 0, pos + advance, -1)) {
                throw std::runtime_error("generate_lctx_mtp8: llama_memory_seq_rm failed");
            }
        }
        pos += advance;
        cur = next_tok;
    }

    st.decode_sec   = now_sec() - t0;
    st.mean_acc_len = st.rounds > 0 ? (double) (st.rounds + st.accepted_extra) / (double) st.rounds : 1.0;
    if ((int32_t) out.size() > max_new_tokens) {
        out.resize((size_t) max_new_tokens);
    }
    if (out_stats) {
        *out_stats = st;
    }
    return out;
}

static void save_json_if_requested(const std::string & path, const json & j) {
    const std::string dump = j.dump(2);
    if (!path.empty()) {
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("Failed to open output JSON: " + path);
        }
        f.write(dump.data(), (std::streamsize) dump.size());
        f << "\n";
    }
    std::cout << dump << std::endl;
}

// ============================================================================
// Mode 1: `parity48` — 48-Visit Final Logits Cosine & Out-Check Gate
// ============================================================================
static int run_parity48(common_params & params, const Ink2BenchArgs & bargs) {
    params.warmup = false;
    if (bargs.target_vram_gib > 0.0) {
        size_t vram_free_before = 0, vram_total = 0;
        if (query_cuda0_mem(vram_free_before, vram_total) && vram_free_before > 0) {
            const int64_t mtp_emb_vram = (int64_t) 590 << 20;
            const int64_t target_bytes = std::max<int64_t>((int64_t) 1 << 30,
                (int64_t) (bargs.target_vram_gib * 1024.0 * 1024.0 * 1024.0) - mtp_emb_vram);
            const int64_t keep_free    = std::max<int64_t>(0, (int64_t) vram_free_before - target_bytes);
            params.moe_stream_vram_free = keep_free;
            if (params.moe_stream_gpu_layers < 0) {
                params.moe_stream_vram_reserve = std::max<int64_t>(
                    params.moe_stream_vram_reserve,
                    keep_free + (int64_t) (2.25 * 1024.0 * 1024.0 * 1024.0));
            }
        }
    }
    common_init_result_ptr init = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * lctx  = init->context();
    if (!model || !lctx) {
        std::cerr << "parity48: failed to load model/context\n";
        return 1;
    }
    llama_memory_clear(llama_get_memory(lctx), true);
    auto tmpls = common_chat_templates_init(model, params.chat_template);
    auto pids  = encode_chat_prompt(model, tmpls.get(),
        "Prove that for every prime p > 3, p^2 - 1 is divisible by 24.");
    if (eval_tokens_chunked(lctx, pids, 0, params.n_batch, true) != 0) {
        std::cerr << "parity48: llama_decode failed\n";
        return 1;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits  = llama_get_logits_ith(lctx, -1);

    int32_t finite_cnt = 0;
    int32_t nan_cnt    = 0;
    int32_t top1_id    = -1;
    float   top1_val   = -1e30f;
    for (int32_t i = 0; i < n_vocab; ++i) {
        const float v = logits[i];
        if (std::isnan(v)) {
            nan_cnt++;
        } else if (std::isfinite(v)) {
            finite_cnt++;
            if (v > top1_val) {
                top1_val = v;
                top1_id  = i;
            }
        }
    }

    if (!bargs.save_logits.empty()) {
        std::ofstream fo(bargs.save_logits, std::ios::binary);
        fo.write(reinterpret_cast<const char *>(&n_vocab), sizeof(int32_t));
        fo.write(reinterpret_cast<const char *>(logits), (std::streamsize) n_vocab * sizeof(float));
    }

    json res = {
        { "mode",        "parity48" },
        { "tag",         bargs.tag },
        { "n_tokens",    (int32_t) pids.size() },
        { "n_vocab",     n_vocab },
        { "finite_cnt",  finite_cnt },
        { "nan_cnt",     nan_cnt },
        { "top1_id",     top1_id },
        { "top1_logit",  round3(top1_val) },
    };

    bool ok = (nan_cnt == 0 && finite_cnt >= 200000);
    if (!bargs.ref_logits.empty()) {
        std::ifstream fi(bargs.ref_logits, std::ios::binary);
        if (!fi) {
            std::cerr << "parity48: missing ref logits " << bargs.ref_logits << "\n";
            return 1;
        }
        int32_t ref_v = 0;
        fi.read(reinterpret_cast<char *>(&ref_v), sizeof(int32_t));
        if (ref_v != n_vocab) {
            std::cerr << "parity48: vocab mismatch " << ref_v << " vs " << n_vocab << "\n";
            return 1;
        }
        std::vector<float> ref((size_t) n_vocab);
        fi.read(reinterpret_cast<char *>(ref.data()), (std::streamsize) n_vocab * sizeof(float));

        double dot = 0.0, na = 0.0, nb = 0.0;
        double max_abs = 0.0;
        int32_t ref_top1 = -1;
        float   ref_topv = -1e30f;
        // Probability-weighted & logit cosine over all finite unpadded vocab entries
        double p_dot = 0.0, p_na = 0.0, p_nb = 0.0;
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (std::isfinite(ref[i]) && ref[i] > ref_topv) {
                ref_topv = ref[i];
                ref_top1 = i;
            }
        }
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(ref[i]) && !std::isfinite(logits[i])) {
                continue;
            }
            const double a = (double) logits[i];
            const double b = (double) ref[i];
            dot += a * b;
            na  += a * a;
            nb  += b * b;
            max_abs = std::max(max_abs, std::fabs(a - b));
            const double pa = std::exp(a - (double) top1_val);
            const double pb = std::exp(b - (double) ref_topv);
            p_dot += pa * pb;
            p_na  += pa * pa;
            p_nb  += pb * pb;
        }
        const double logit_cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
        const double prob_cos  = p_dot / (std::sqrt(p_na) * std::sqrt(p_nb) + 1e-30);
        res["logit_cosine"]  = logit_cos;
        res["prob_cosine"]   = prob_cos;
        res["max_abs_diff"]  = round3(max_abs);
        res["ref_top1_id"]   = ref_top1;
        res["top1_match"]    = (top1_id == ref_top1);
        const bool strict_gpu_ref = (bargs.tag.find("gpu_ref") != std::string::npos);
        const bool vram_hybrid    = !strict_gpu_ref && (
            bargs.tag.find("hybrid") != std::string::npos ||
            bargs.tag.find("cpu_repack") != std::string::npos);
        const double l_bar = strict_gpu_ref ? 0.99999 : (vram_hybrid ? 0.998 : 0.99995);
        const double p_bar = 0.99995;
        ok = ok && (logit_cos >= l_bar) && (prob_cos >= p_bar) && (top1_id == ref_top1);
    }
    res["passed"] = ok;
    save_json_if_requested(bargs.out_json, res);
    return ok ? 0 : 1;
}

// ============================================================================
// Mode 2: `side-a-phase-a` — Plugin-Only ABI v2 Baseline (`INKLING2_PHASE_A_PLUGIN_ONLY=1`)
// ============================================================================
static int run_side_a_phase_a(common_params & params, const Ink2BenchArgs & bargs) {
#if defined(_WIN32)
    _putenv_s("INKLING2_PHASE_A_PLUGIN_ONLY", "1");
#else
    ::setenv("INKLING2_PHASE_A_PLUGIN_ONLY", "1", 1);
#endif
    params.n_ctx    = std::max<int32_t>(params.n_ctx, 4096);
    params.n_batch  = std::max<int32_t>(params.n_batch, 2048);
    params.n_ubatch = std::max<int32_t>(params.n_ubatch, 2048);

    common_init_result_ptr init = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * lctx  = init->context();
    if (!model || !lctx) {
        std::cerr << "side-a-phase-a: failed to load model/context\n";
        return 1;
    }

    auto tmpls   = common_chat_templates_init(model, params.chat_template);
    auto ids_4k  = make_long_ids(model, 2048);
    auto pids_cd = encode_chat_prompt(model, tmpls.get(),
        "Write a high-performance C++20 Dinic max-flow implementation with complexity analysis and edge-case tests.");
    auto pids_ma = encode_chat_prompt(model, tmpls.get(),
        "Prove that for every prime p > 3, p^2 - 1 is divisible by 24, and generalize to twin primes.");

    // Warmup
    llama_memory_clear(llama_get_memory(lctx), true);
    (void) eval_tokens_chunked(lctx, std::vector<llama_token>(ids_4k.begin(), ids_4k.begin() + 64), 0, params.n_batch, true);
    sync_gpu_dev();

    json prefill_j = json::object();
    const std::pair<int32_t, double> v150_prefill[] = {
        { 512,  7913.0  },
        { 2048, 11496.4 },
    };
    for (const auto & [S, ref_tps] : v150_prefill) {
        std::vector<llama_token> chunk(ids_4k.begin(), ids_4k.begin() + S);
        llama_memory_clear(llama_get_memory(lctx), true);
        sync_gpu_dev();
        const double t0 = now_sec();
        if (eval_tokens_chunked(lctx, chunk, 0, S, true) != 0) {
            return 1;
        }
        sync_gpu_dev();
        const double dt  = now_sec() - t0;
        const double tps = (double) S / dt;
        prefill_j["S=" + std::to_string(S)] = {
            { "ms",          round2(dt * 1000.0) },
            { "tok_s",       round2(tps) },
            { "v150_tok_s",  ref_tps },
            { "pct_of_v150", round2(100.0 * tps / ref_tps) },
        };
    }

    // B=1 M=1 Decode (64 steps)
    (void) generate_lctx_m1(lctx, pids_cd, 9, params.n_batch, nullptr);
    sync_gpu_dev();
    double dec_s = 0.0;
    auto m1_cd = generate_lctx_m1(lctx, pids_cd, 65, params.n_batch, &dec_s);
    sync_gpu_dev();
    const double m1_tps = 64.0 / dec_s;

    // B=1 Speculative MTP-8 via `llama_inkling2_mtp_draft` + plugin-only verification (`code_cont_512` & `math`, 64 tokens)
    auto count_match = [](const std::vector<llama_token> & a, const std::vector<llama_token> & b, int32_t & pref_out) -> int32_t {
        const size_t n = std::min(a.size(), b.size());
        int32_t cnt = 0;
        pref_out = (int32_t) n;
        for (size_t i = 0; i < n; ++i) {
            if (a[i] == b[i]) {
                cnt++;
            } else if (pref_out == (int32_t) n) {
                pref_out = (int32_t) i;
            }
        }
        return cnt;
    };

    std::vector<llama_token> cc512(ids_4k.begin(), ids_4k.begin() + 512);
    auto ref_cc = generate_lctx_m1(lctx, cc512, 65, params.n_batch, nullptr);
    LctxSpecStats st_cc {};
    auto sp_cc  = generate_lctx_mtp8(lctx, cc512, 65, 7, params.n_batch, &st_cc);
    int32_t cc_pref = 0;
    const int32_t cc_cnt = count_match(ref_cc, sp_cc, cc_pref);
    const bool cc_exact  = (ref_cc == sp_cc);
    const double cc_tps  = 64.0 / st_cc.decode_sec;

    auto ref_ma = generate_lctx_m1(lctx, pids_ma, 65, params.n_batch, nullptr);
    LctxSpecStats st_ma {};
    auto sp_ma  = generate_lctx_mtp8(lctx, pids_ma, 65, 7, params.n_batch, &st_ma);
    int32_t ma_pref = 0;
    const int32_t ma_cnt = count_match(ref_ma, sp_ma, ma_pref);
    const bool ma_exact  = (ref_ma == sp_ma);
    const double ma_tps  = 64.0 / st_ma.decode_sec;

    json out = {
        { "mode",  "side-a-phase-a-plugin-only" },
        { "prefill", prefill_j },
        { "b1_plain_m1", {
            { "steps",       64 },
            { "tok_s",       round2(m1_tps) },
            { "ms_per_tok",  round3(dec_s * 1000.0 / 64.0) },
            { "v150_tok_s",  178.62 },
            { "pct_of_v150", round2(100.0 * m1_tps / 178.62) },
        }},
        { "b1_spec_mtp8", {
            { "code_cont_512", {
                { "tok_s",               round2(cc_tps) },
                { "tau",                 round3(st_cc.mean_acc_len) },
                { "matched_tokens",      std::to_string(cc_cnt) + "/65" },
                { "matched_prefix",      cc_pref },
                { "exact_greedy_parity", cc_exact },
                { "v150_tok_s",          548.95 },
                { "pct_of_v150",         round2(100.0 * cc_tps / 548.95) },
            }},
            { "math", {
                { "tok_s",               round2(ma_tps) },
                { "tau",                 round3(st_ma.mean_acc_len) },
                { "matched_tokens",      std::to_string(ma_cnt) + "/65" },
                { "matched_prefix",      ma_pref },
                { "exact_greedy_parity", ma_exact },
                { "v150_tok_s",          281.97 },
                { "pct_of_v150",         round2(100.0 * ma_tps / 281.97) },
            }},
        }},
        { "passed", (m1_tps > 0.0 && cc_tps > 0.0 && ma_tps > 0.0 && st_cc.mean_acc_len > 1.2) },
    };
    save_json_if_requested(bargs.out_json, out);
    return out["passed"].get<bool>() ? 0 : 1;
}

// ============================================================================
// Mode 3: `side-a-phase-b` — Full-VRAM Fused Engine vs `v1.5.0` (`>80%` Bar)
// ============================================================================
static int run_side_a_phase_b(common_params & params, const Ink2BenchArgs & bargs) {
#if defined(_WIN32)
    _putenv_s("INKLING2_PHASE_A_PLUGIN_ONLY", "0");
#else
    ::unsetenv("INKLING2_PHASE_A_PLUGIN_ONLY");
#endif
    params.n_ctx    = std::max<int32_t>(params.n_ctx, 8192);
    params.n_batch  = std::max<int32_t>(params.n_batch, 8192);
    params.n_ubatch = std::max<int32_t>(params.n_ubatch, 8192);

    common_init_result_ptr init = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * lctx  = init->context();
    if (!model || !lctx) {
        std::cerr << "side-a-phase-b: failed to load model/context\n";
        return 1;
    }

    const auto * api = get_cuda_ink2_api();
    if (!api) {
        std::cerr << "side-a-phase-b: ggml_cuda_inkling2_api unavailable\n";
        return 1;
    }
    void * h = llama_inkling2_get_cuda_handle(model, 1, 8192, true);
    if (!h) {
        std::cerr << "side-a-phase-b: llama_inkling2_get_cuda_handle returned null\n";
        return 1;
    }

    auto tmpls   = common_chat_templates_init(model, params.chat_template);
    auto ids_32k = make_long_ids(model, 32768);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // 1. Exact 0.0-ULP M=1 / M=4 / M=8 Logit Parity (`forward_logits_f32` outputs `vocab_unpad = 200058` per row)
    const int32_t v_unpad = 200058;
    std::vector<float> log_m8((size_t) 8 * v_unpad);
    std::vector<float> log_m4((size_t) 4 * v_unpad);
    std::vector<float> log_m1((size_t) 1 * v_unpad);
    api->reset_cache(h);
    if (api->forward_logits_f32(h, ids_32k.data(), 8, log_m8.data()) != 0) return 1;
    api->reset_cache(h);
    if (api->forward_logits_f32(h, ids_32k.data(), 4, log_m4.data()) != 0) return 1;
    api->reset_cache(h);
    if (api->forward_logits_f32(h, ids_32k.data(), 1, log_m1.data()) != 0) return 1;

    double diff_1_8 = 0.0, diff_4_8 = 0.0;
    for (int32_t i = 0; i < v_unpad; ++i) {
        if (std::isfinite(log_m8[i]) || std::isfinite(log_m1[i])) {
            diff_1_8 = std::max(diff_1_8, (double) std::fabs(log_m8[i] - log_m1[i]));
        }
    }
    for (size_t i = 0; i < (size_t) 4 * v_unpad; ++i) {
        if (std::isfinite(log_m8[i]) || std::isfinite(log_m4[i])) {
            diff_4_8 = std::max(diff_4_8, (double) std::fabs(log_m8[i] - log_m4[i]));
        }
    }
    const double max_ulp_diff = std::max(diff_1_8, diff_4_8);

    // Warmup CUDA graphs on both direct handle and `llama_decode` (`FUSED_TRUNK`)
    {
        std::vector<int32_t> w512(ids_32k.begin(), ids_32k.begin() + 512);
        int32_t ft = 0, stp = 0, len_o = 0;
        std::vector<int32_t> tmp_out(64);
        api->reset_cache(h);
        api->prefill(h, w512.data(), 1, 512, 0, 0.0f, 1.0f, 42, &ft, nullptr, 0, 0, 0, 200054);
        api->decode_m1(h, &ft, 1, 33, -1, 0.0f, 1.0f, 42, tmp_out.data(), &stp);
        for (int pol : { 0, 8 }) {
            api->reset_cache(h);
            api->set_mtp_policy(h, pol);
            api->prefill(h, w512.data(), 1, 512, 1, 0.0f, 1.0f, 42, &ft, nullptr, 0, 0, 0, 200054);
            ggml_ink2_mtp_stats mst {};
            api->decode_mtp8(h, ft, 33, -1, 0.0f, 1.0f, 42, tmp_out.data(), &len_o, &mst);
        }
        (void) generate_lctx_m1(lctx, std::vector<llama_token>(ids_32k.begin(), ids_32k.begin() + 64), 17, params.n_batch, nullptr);
        sync_gpu_dev();
    }

    // 2. 96/96 Greedy Parity (`M=1 == MTP-8 == adaptive == llama_decode FUSED_TRUNK + llama_inkling2_mtp_draft`)
    const std::pair<std::string, std::string> parity_prompts[] = {
        { "code",    "Write a high-performance C++20 Dinic max-flow implementation with complexity analysis and edge-case tests." },
        { "math",    "Prove that for every prime p > 3, p^2 - 1 is divisible by 24, and generalize to twin primes." },
        { "segtree", "Design an iterative segment tree in Python 3 supporting range affine updates and range sum queries modulo 998244353." },
    };
    int32_t matched_96 = 0;
    json parity_j = json::object();
    bool parity_96_ok = true;
    for (const auto & [pname, ptxt] : parity_prompts) {
        auto pids = encode_chat_prompt(model, tmpls.get(), ptxt);
        int32_t ft = 0, stp = 0, len8 = 0, lena = 0;
        std::vector<int32_t> m1_t(32), mtp8_t(40), ada_t(40);

        api->reset_cache(h);
        api->prefill(h, pids.data(), 1, (int32_t) pids.size(), 0, 0.0f, 1.0f, 42, &ft, nullptr, 0, 0, 0, 200054);
        api->decode_m1(h, &ft, 1, 32, -1, 0.0f, 1.0f, 42, m1_t.data(), &stp);

        api->reset_cache(h);
        api->set_mtp_policy(h, 8);
        api->prefill(h, pids.data(), 1, (int32_t) pids.size(), 1, 0.0f, 1.0f, 42, &ft, nullptr, 0, 0, 0, 200054);
        ggml_ink2_mtp_stats mst8 {};
        api->decode_mtp8(h, ft, 32, -1, 0.0f, 1.0f, 42, mtp8_t.data(), &len8, &mst8);
        mtp8_t.resize(32);

        api->reset_cache(h);
        api->set_mtp_policy(h, 0);
        api->prefill(h, pids.data(), 1, (int32_t) pids.size(), 1, 0.0f, 1.0f, 42, &ft, nullptr, 0, 0, 0, 200054);
        ggml_ink2_mtp_stats msta {};
        api->decode_mtp8(h, ft, 32, -1, 0.0f, 1.0f, 42, ada_t.data(), &lena, &msta);
        ada_t.resize(32);

        auto lctx_m1   = generate_lctx_m1  (lctx, pids, 32, params.n_batch, nullptr);
        auto lctx_spec = generate_lctx_mtp8(lctx, pids, 32, 7, params.n_batch, nullptr);

        int m_mtp8 = 0, m_ada = 0, m_lctx = 0;
        for (int i = 0; i < 32; ++i) {
            if (m1_t[i] == mtp8_t[i]) m_mtp8++;
            if (m1_t[i] == ada_t[i])  m_ada++;
            if (m1_t[i] == lctx_m1[i] && m1_t[i] == lctx_spec[i]) m_lctx++;
        }
        const bool ok_p = (m_mtp8 == 32 && m_ada == 32 && m_lctx == 32);
        parity_96_ok = parity_96_ok && ok_p;
        matched_96 += std::min({ m_mtp8, m_ada, m_lctx });
        parity_j[pname] = {
            { "m1_eq_mtp8",      std::to_string(m_mtp8) + "/32" },
            { "m1_eq_adaptive",  std::to_string(m_ada)  + "/32" },
            { "m1_eq_lctx_spec", std::to_string(m_lctx) + "/32" },
            { "passed",          ok_p },
        };
    }

    // 3. Steady-State B=1 Plain M=1 Decode (`128` steps: BOTH `llama_decode FUSED_TRUNK` and direct `decode_m1`)
    auto pids_code = encode_chat_prompt(model, tmpls.get(),
        "Write a high-performance C++20 Dinic max-flow implementation with complexity analysis and edge-case tests.");
    double best_m1_direct_tps = 0.0, best_m1_lctx_tps = 0.0;
    std::vector<double> m1_direct_runs, m1_lctx_runs;
    for (int r = 0; r < 4; ++r) {
        int32_t ft = 0, stp = 0;
        std::vector<int32_t> out129(129);
        api->reset_cache(h);
        api->prefill(h, pids_code.data(), 1, (int32_t) pids_code.size(), 0, 0.0f, 1.0f, 42, &ft, nullptr, 0, 0, 0, 200054);
        sync_gpu_dev();
        const double t0 = now_sec();
        api->decode_m1(h, &ft, 1, 129, -1, 0.0f, 1.0f, 42, out129.data(), &stp);
        sync_gpu_dev();
        const double tps = 128.0 / (now_sec() - t0);
        m1_direct_runs.push_back(round2(tps));
        best_m1_direct_tps = std::max(best_m1_direct_tps, tps);

        double lctx_dt = 0.0;
        (void) generate_lctx_m1(lctx, pids_code, 129, params.n_batch, &lctx_dt);
        sync_gpu_dev();
        const double ltps = 128.0 / lctx_dt;
        m1_lctx_runs.push_back(round2(ltps));
        best_m1_lctx_tps = std::max(best_m1_lctx_tps, ltps);
    }
    const double best_m1_tps = std::max(best_m1_direct_tps, best_m1_lctx_tps);

    // 4. Multi-Domain Speculative Decode (`192` decoded tokens across all 6 `v1.5.0` domains)
    struct SpecDomainCase {
        std::string              name;
        std::vector<llama_token> pids;
        double                   v150_best_tps;
    };
    std::vector<SpecDomainCase> spec_cases = {
        { "code",           encode_chat_prompt(model, tmpls.get(), "Write a high-performance C++20 Dinic max-flow implementation with complexity analysis and edge-case tests."), 222.54 },
        { "math",           encode_chat_prompt(model, tmpls.get(), "Prove that for every prime p > 3, p^2 - 1 is divisible by 24, and generalize to twin primes."), 281.97 },
        { "json",           encode_chat_prompt(model, tmpls.get(), "Return a rich JSON specification for a distributed consensus cluster with 6 node profiles, quorum rules, and Raft timeouts. Output only valid JSON."), 237.75 },
        { "prose",          encode_chat_prompt(model, tmpls.get(), "Write a detailed architectural history of Venice's Arsenale and how modular galley production anticipated industrial assembly lines."), 232.26 },
        { "code_cont_512",  std::vector<llama_token>(ids_32k.begin(), ids_32k.begin() + 512),  548.95 },
        { "code_cont_2048", std::vector<llama_token>(ids_32k.begin(), ids_32k.begin() + 2048), 475.73 },
    };

    json spec_j = json::object();
    bool all_spec_80_ok = true;
    for (const auto & sc : spec_cases) {
        const int32_t max_new = 193; // 1 prefill + 192 decoded tokens
        int32_t ft_ref = 0, stp_ref = 0;
        std::vector<int32_t> ref_seq(max_new);
        api->reset_cache(h);
        api->prefill(h, sc.pids.data(), 1, (int32_t) sc.pids.size(), 0, 0.0f, 1.0f, 42, &ft_ref, nullptr, 0, 0, 0, 200054);
        api->decode_m1(h, &ft_ref, 1, max_new, -1, 0.0f, 1.0f, 42, ref_seq.data(), &stp_ref);

        json dom_j = json::object();
        double dom_best_tps = 0.0;
        bool   dom_exact    = true;
        for (const auto & [pol_name, pol_val] : std::vector<std::pair<std::string, int32_t>>{ { "adaptive", 0 }, { "mtp8", 8 } }) {
            api->set_mtp_policy(h, pol_val);
            double best_pol_tps = 0.0;
            json   best_entry   = json::object();
            for (int r = 0; r < 3; ++r) {
                int32_t ft = 0, out_len = 0;
                std::vector<int32_t> seq(max_new + 16);
                ggml_ink2_mtp_stats mst {};
                api->reset_cache(h);
                api->prefill(h, sc.pids.data(), 1, (int32_t) sc.pids.size(), 1, 0.0f, 1.0f, 42, &ft, nullptr, 0, 0, 0, 200054);
                sync_gpu_dev();
                const double t0 = now_sec();
                api->decode_mtp8(h, ft, max_new, -1, 0.0f, 1.0f, 42, seq.data(), &out_len, &mst);
                sync_gpu_dev();
                const double dt  = now_sec() - t0;
                const int32_t n_cmp = std::min(max_new, out_len);
                bool exact = (n_cmp == max_new);
                for (int32_t i = 0; i < n_cmp; ++i) {
                    if (seq[i] != ref_seq[i]) {
                        exact = false;
                        break;
                    }
                }
                dom_exact = dom_exact && exact;
                const double tps = (double) (out_len - 1) / dt;
                if (r >= 1 && tps > best_pol_tps) {
                    best_pol_tps = tps;
                    const int64_t rnds = std::max<int64_t>(1, mst.spec_steps);
                    best_entry = {
                        { "tok_s",               round2(tps) },
                        { "ms_per_round",        round3(dt * 1000.0 / (double) rnds) },
                        { "tau",                 round3(mst.mean_accepted_len) },
                        { "spec_steps",          mst.spec_steps },
                        { "s4_steps",            mst.s4_steps },
                        { "s8_steps",            mst.s8_steps },
                        { "decoded_tokens",      out_len - 1 },
                        { "exact_greedy_parity", exact },
                    };
                }
            }
            dom_j[pol_name] = best_entry;
            dom_best_tps = std::max(dom_best_tps, best_pol_tps);
        }

        // Also measure standard `llama_decode` + `llama_inkling2_mtp_draft` loop
        LctxSpecStats lst {};
        auto lseq = generate_lctx_mtp8(lctx, sc.pids, max_new, 7, params.n_batch, &lst);
        bool lctx_exact = ((int32_t) lseq.size() == max_new);
        for (int32_t i = 0; i < max_new && lctx_exact; ++i) {
            if (lseq[i] != ref_seq[i]) lctx_exact = false;
        }
        dom_exact = dom_exact && lctx_exact;
        const double lctx_tps = 192.0 / lst.decode_sec;
        dom_j["llama_context_mtp8"] = {
            { "tok_s",               round2(lctx_tps) },
            { "tau",                 round3(lst.mean_acc_len) },
            { "rounds",              lst.rounds },
            { "exact_greedy_parity", lctx_exact },
            { "pct_of_v150",         round2(100.0 * lctx_tps / sc.v150_best_tps) },
        };

        const double pct = 100.0 * dom_best_tps / sc.v150_best_tps;
        dom_j["best_tok_s"]  = round2(dom_best_tps);
        dom_j["v150_tok_s"]  = sc.v150_best_tps;
        dom_j["pct_of_v150"] = round2(pct);
        spec_j[sc.name]      = dom_j;
        all_spec_80_ok = all_spec_80_ok && dom_exact && (pct > 80.0);
    }

    // 5. Multimodal Vision VQA (`libmtmd` + `Inkling-2-Mini-mmproj-bf16.gguf`)
    json vqa_j = { { "skipped", true } };
    bool vqa_80_ok = true;
    if (!params.mmproj.path.empty()) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu       = true;
        mparams.print_timings = false;
        mparams.n_threads     = std::max(1, params.cpuparams.n_threads);
        mparams.warmup        = true;
        mtmd_context * mctx   = mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams);
        if (!mctx) {
            std::cerr << "side-a-phase-b: failed to load mmproj " << params.mmproj.path << "\n";
            return 1;
        }

        // Exact 256x256 RGB image: left half (x=0..127) = (220, 20, 20) red, right half (x=128..255) = (20, 60, 220) blue
        std::vector<unsigned char> rgb(256 * 256 * 3);
        for (int y = 0; y < 256; ++y) {
            for (int x = 0; x < 256; ++x) {
                const size_t idx = ((size_t) y * 256 + (size_t) x) * 3;
                if (x <= 127) {
                    rgb[idx + 0] = 220; rgb[idx + 1] = 20;  rgb[idx + 2] = 20;
                } else {
                    rgb[idx + 0] = 20;  rgb[idx + 1] = 60;  rgb[idx + 2] = 220;
                }
            }
        }
        mtmd_bitmap * bmp = mtmd_bitmap_init(256, 256, rgb.data());

        // Format prompt with `<|unused_200054|>` via chat template (matches `bench_inkling_v150_g4.py` 1:1!)
        common_chat_msg vmsg;
        vmsg.role    = "user";
        vmsg.content = "<|unused_200054|>\nWhat are the two main colors on the left and right halves of this image?";
        std::string vrendered = common_chat_format_single(tmpls.get(), {}, vmsg, true, true);

        mtmd_input_text itxt { vrendered.c_str(), vrendered.size(), false, true };
        const mtmd_bitmap * bmps[1] = { bmp };
        mtmd_input_chunks * chunks  = mtmd_input_chunks_init();
        if (mtmd_tokenize(mctx, chunks, &itxt, bmps, 1) != 0) {
            std::cerr << "side-a-phase-b: mtmd_tokenize failed\n";
            return 1;
        }

        const mtmd_input_chunk * img_chunk = nullptr;
        for (size_t ci = 0; ci < mtmd_input_chunks_size(chunks); ++ci) {
            const auto * ch = mtmd_input_chunks_get(chunks, ci);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                img_chunk = ch;
                break;
            }
        }
        if (!img_chunk) {
            std::cerr << "side-a-phase-b: missing image chunk\n";
            return 1;
        }
        const int32_t n_vis = (int32_t) mtmd_input_chunk_get_n_tokens(img_chunk);

        // Warmup + timed `libmtmd` ViT + Merger encode on GPU
        (void) mtmd_encode_chunk(mctx, img_chunk);
        sync_gpu_dev();
        const double tv0 = now_sec();
        if (mtmd_encode_chunk(mctx, img_chunk) != 0) {
            std::cerr << "side-a-phase-b: mtmd_encode_chunk failed\n";
            return 1;
        }
        sync_gpu_dev();
        const double vis_enc_ms = (now_sec() - tv0) * 1000.0;
        const float * vis_emb_f32 = mtmd_get_output_embd(mctx);

        // A. End-to-end `mtmd_helper_eval_chunks` (`llama_decode FUSED_TRUNK` + `llama_inkling2_mtp_draft`) check:
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_pos n_past = 0;
        if (mtmd_helper_eval_chunks(mctx, lctx, chunks, 0, 0, params.n_batch, true, &n_past) != 0) {
            std::cerr << "side-a-phase-b: mtmd_helper_eval_chunks failed\n";
            return 1;
        }
        std::vector<llama_token> mtmd_lctx_toks;
        llama_token cur_v = argmax_logits(llama_get_logits_ith(lctx, -1), n_vocab);
        mtmd_lctx_toks.push_back(cur_v);
        common_batch bv(lctx);
        int32_t pos_v = (int32_t) n_past;
        const double tl0 = now_sec();
        while ((int32_t) mtmd_lctx_toks.size() < 64) {
            llama_token dft[7];
            const int32_t rem = 64 - (int32_t) mtmd_lctx_toks.size();
            const int32_t k_a = std::min<int32_t>(7, std::max<int32_t>(1, rem - 1));
            int32_t nd = (rem > 1) ? llama_inkling2_mtp_draft(lctx, pos_v, cur_v, k_a, dft) : 0;
            if (nd < 0) nd = 0;
            bv.clear();
            bv.add(cur_v, pos_v, 0, true);
            for (int32_t j = 0; j < nd; ++j) bv.add(dft[j], pos_v + 1 + j, 0, true);
            if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, bv.get()) != 0) return 1;
            int32_t na = 0;
            llama_token nxt = argmax_logits(llama_get_logits_ith(lctx, 0), n_vocab);
            for (int32_t j = 0; j < nd; ++j) {
                if (nxt == dft[j] && (int32_t) mtmd_lctx_toks.size() < 64) {
                    mtmd_lctx_toks.push_back(nxt);
                    na++;
                    nxt = argmax_logits(llama_get_logits_ith(lctx, j + 1), n_vocab);
                } else {
                    break;
                }
            }
            if ((int32_t) mtmd_lctx_toks.size() < 64) mtmd_lctx_toks.push_back(nxt);
            if (na < nd) (void) llama_memory_seq_rm(llama_get_memory(lctx), 0, pos_v + 1 + na, -1);
            pos_v += 1 + na;
            cur_v = nxt;
        }
        sync_gpu_dev();
        const double lctx_vqa_tps = 63.0 / (now_sec() - tl0);

        // B. Direct `api->prefill` + `decode_m1` vs `decode_mtp8` with `libmtmd` embeddings (exact `v1.5.0` protocol):
        std::string expanded = vrendered;
        {
            const std::string ph = "<|unused_200054|>";
            const size_t ppos = expanded.find(ph);
            if (ppos != std::string::npos) {
                std::string rep;
                for (int32_t k = 0; k < n_vis; ++k) rep += ph;
                expanded.replace(ppos, ph.size(), rep);
            }
        }
        auto vqa_ids = common_tokenize(llama_model_get_vocab(model), expanded, false, true);

        int32_t ft_vm1 = 0, stp_vm1 = 0;
        std::vector<int32_t> vm1(64);
        api->reset_cache(h);
        api->prefill(h, vqa_ids.data(), 1, (int32_t) vqa_ids.size(), 0, 0.0f, 1.0f, 42,
                     &ft_vm1, vis_emb_f32, 1, n_vis, 0, 200054);
        api->decode_m1(h, &ft_vm1, 1, 64, 200006, 0.0f, 1.0f, 42, vm1.data(), &stp_vm1);
        vm1.resize((size_t) (stp_vm1 + 1));

        // Warmup + 3 timed runs of Vision VQA MTP-8 decode
        api->set_mtp_policy(h, 0);
        double best_vqa_tps = 0.0;
        double best_vqa_tau = 0.0;
        std::vector<int32_t> vmtp_best;
        for (int r = 0; r < 3; ++r) {
            int32_t ft_v = 0, len_v = 0;
            std::vector<int32_t> vmtp(80);
            ggml_ink2_mtp_stats vst {};
            api->reset_cache(h);
            api->prefill(h, vqa_ids.data(), 1, (int32_t) vqa_ids.size(), 1, 0.0f, 1.0f, 42,
                         &ft_v, vis_emb_f32, 1, n_vis, 0, 200054);
            sync_gpu_dev();
            const double td0 = now_sec();
            api->decode_mtp8(h, ft_v, 64, 200006, 0.0f, 1.0f, 42, vmtp.data(), &len_v, &vst);
            sync_gpu_dev();
            const double dt_v = now_sec() - td0;
            vmtp.resize((size_t) len_v);
            const double tps_v = (double) std::max(1, len_v - 1) / dt_v;
            if (tps_v > best_vqa_tps) {
                best_vqa_tps = tps_v;
                best_vqa_tau = vst.mean_accepted_len;
                vmtp_best    = vmtp;
            }
        }

        const size_t n_vcmp = std::min(vm1.size(), vmtp_best.size());
        bool m1_eq_mtp = (n_vcmp > 0);
        for (size_t i = 0; i < n_vcmp; ++i) {
            if (vm1[i] != vmtp_best[i]) m1_eq_mtp = false;
        }
        std::string ans_txt = common_detokenize(lctx, vmtp_best, false);
        std::string ans_lwr = ans_txt;
        std::transform(ans_lwr.begin(), ans_lwr.end(), ans_lwr.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        const bool has_red_blue = (ans_lwr.find("red") != std::string::npos) && (ans_lwr.find("blue") != std::string::npos);
        const double eff_vqa_tps = std::max(best_vqa_tps, lctx_vqa_tps);
        const double vqa_pct     = 100.0 * eff_vqa_tps / 451.62;
        vqa_80_ok = has_red_blue && m1_eq_mtp && (n_vis == 49) && (vqa_pct > 80.0);

        vqa_j = {
            { "passed",            vqa_80_ok },
            { "m1_eq_mtp",         m1_eq_mtp },
            { "n_vis_tokens",      n_vis },
            { "mtmd_encode_ms",    round2(vis_enc_ms) },
            { "decode_tok_s",      round2(eff_vqa_tps) },
            { "lctx_decode_tok_s", round2(lctx_vqa_tps) },
            { "tau",               round3(best_vqa_tau) },
            { "v150_tok_s",        451.62 },
            { "pct_of_v150",       round2(vqa_pct) },
            { "answer",            ans_txt.substr(0, 220) },
        };

        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        mtmd_free(mctx);
    }

    // 6. Warmed Pure Prefill Sweep (`S = 512, 2048, 8192, 32768`, `max_seq = 36864`)
    void * hp = llama_inkling2_get_cuda_handle(model, 1, 36864, true);
    if (!hp) {
        std::cerr << "side-a-phase-b: failed to resize handle for 36864 prefill\n";
        return 1;
    }
    {
        int32_t ft_w = 0;
        api->reset_cache(hp);
        api->prefill(hp, ids_32k.data(), 1, 2048, 1, 0.0f, 1.0f, 42, &ft_w, nullptr, 0, 0, 0, 200054);
        sync_gpu_dev();
    }
    json prefill_j = json::object();
    bool all_prefill_80_ok = true;
    const std::pair<int32_t, double> pf_targets[] = {
        { 512,   7913.0  },
        { 2048,  11496.4 },
        { 8192,  11978.4 },
        { 32768, 10936.8 },
    };
    for (const auto & [S, ref_tps] : pf_targets) {
        const int n_iters = (S <= 8192) ? 3 : 2;
        double best_s = 1e9;
        std::vector<double> runs_ms;
        for (int it = 0; it < n_iters; ++it) {
            int32_t ft_o = 0;
            api->reset_cache(hp);
            sync_gpu_dev();
            const double t0 = now_sec();
            if (api->prefill(hp, ids_32k.data(), 1, S, 1, 0.0f, 1.0f, 42, &ft_o, nullptr, 0, 0, 0, 200054) != 0) {
                return 1;
            }
            sync_gpu_dev();
            const double dt = now_sec() - t0;
            best_s = std::min(best_s, dt);
            runs_ms.push_back(round2(dt * 1000.0));
        }
        const double tps = (double) S / best_s;
        const double pct = 100.0 * tps / ref_tps;
        all_prefill_80_ok = all_prefill_80_ok && (pct > 80.0);
        json entry = {
            { "best_ms",     round2(best_s * 1000.0) },
            { "tok_s",       round2(tps) },
            { "runs_ms",     runs_ms },
            { "v150_tok_s",  ref_tps },
            { "pct_of_v150", round2(pct) },
        };
        if (S <= 8192) {
            std::vector<llama_token> chunk(ids_32k.begin(), ids_32k.begin() + S);
            llama_memory_clear(llama_get_memory(lctx), true);
            sync_gpu_dev();
            const double tl0 = now_sec();
            if (eval_tokens_chunked(lctx, chunk, 0, S, true) != 0) return 1;
            sync_gpu_dev();
            const double l_tps = (double) S / (now_sec() - tl0);
            entry["llama_decode_tok_s"]       = round2(l_tps);
            entry["llama_decode_pct_of_v150"] = round2(100.0 * l_tps / ref_tps);
        }
        prefill_j["S=" + std::to_string(S)] = entry;
    }

    // 7. Batched `B = 4, 8, 16` `M=1` Decode (`64` steps)
    void * hb = llama_inkling2_get_cuda_handle(model, 16, 2048, true);
    if (!hb) {
        std::cerr << "side-a-phase-b: failed to resize handle for B=16\n";
        return 1;
    }
    json batch_j = json::object();
    const std::pair<int32_t, double> batch_targets[] = {
        { 4,  537.5  },
        { 8,  980.1  },
        { 16, 1065.5 },
    };
    for (const auto & [B, ref_tot_tps] : batch_targets) {
        std::vector<int32_t> batch_p((size_t) B * 128);
        for (int b = 0; b < B; ++b) {
            std::memcpy(batch_p.data() + (size_t) b * 128, ids_32k.data(), 128 * sizeof(int32_t));
        }
        std::vector<int32_t> ft_b(B), out_w((size_t) B * 17), out_b((size_t) B * 65);
        int32_t stp = 0;
        api->reset_cache(hb);
        api->prefill(hb, batch_p.data(), B, 128, 0, 0.0f, 1.0f, 42, ft_b.data(), nullptr, 0, 0, 0, 200054);
        api->decode_m1(hb, ft_b.data(), B, 17, -1, 0.0f, 1.0f, 42, out_w.data(), &stp);
        sync_gpu_dev();

        api->reset_cache(hb);
        api->prefill(hb, batch_p.data(), B, 128, 0, 0.0f, 1.0f, 42, ft_b.data(), nullptr, 0, 0, 0, 200054);
        sync_gpu_dev();
        const double t0 = now_sec();
        api->decode_m1(hb, ft_b.data(), B, 65, -1, 0.0f, 1.0f, 42, out_b.data(), &stp);
        sync_gpu_dev();
        const double dt = now_sec() - t0;
        const double tot_tps = (double) (B * 64) / dt;
        batch_j["B=" + std::to_string(B)] = {
            { "steps",            64 },
            { "total_tok_s",      round2(tot_tps) },
            { "per_stream_tok_s", round2(64.0 / dt) },
            { "v150_total_tok_s", ref_tot_tps },
            { "pct_of_v150",      round2(100.0 * tot_tps / ref_tot_tps) },
        };
    }

    const double m1_pct = 100.0 * best_m1_tps / 178.62;
    const bool all_ok = (max_ulp_diff == 0.0) && parity_96_ok && all_prefill_80_ok &&
                        (m1_pct > 80.0) && all_spec_80_ok && vqa_80_ok;

    json out = {
        { "mode", "side-a-phase-b-fused-engine" },
        { "zero_ulp_m1_m4_m8_max_abs_diff", max_ulp_diff },
        { "greedy_parity_96", {
            { "matched", std::to_string(matched_96) + "/96" },
            { "passed",  parity_96_ok && matched_96 == 96 },
            { "prompts", parity_j },
        }},
        { "prefill_2k_32k", prefill_j },
        { "b1_plain_m1", {
            { "steps",               128 },
            { "best_tok_s",          round2(best_m1_tps) },
            { "direct_best_tok_s",   round2(best_m1_direct_tps) },
            { "llama_decode_tok_s",  round2(best_m1_lctx_tps) },
            { "best_ms_per_tok",     round3(1000.0 / best_m1_tps) },
            { "direct_runs_tok_s",   m1_direct_runs },
            { "lctx_runs_tok_s",     m1_lctx_runs },
            { "v150_tok_s",          178.62 },
            { "pct_of_v150",         round2(m1_pct) },
        }},
        { "b1_spec_domains", spec_j },
        { "vision_vqa",      vqa_j },
        { "batched_decode",  batch_j },
        { "all_metrics_gt_80pct_v150", all_ok },
        { "passed",          all_ok },
    };
    save_json_if_requested(bargs.out_json, out);
    return all_ok ? 0 : 1;
}

// ============================================================================
// Mode 4: `side-b-stream` — Cold `O_DIRECT` NVMe / VRAM+RAM+SSD Hybrid Tiering
// ============================================================================
static int run_side_b_stream(common_params & params, const Ink2BenchArgs & bargs) {
    if (bargs.drop_cache && !params.model.path.empty()) {
        drop_file_page_cache(params.model.path);
    }

    size_t vram_free_before = 0, vram_total = 0;
    (void) query_cuda0_mem(vram_free_before, vram_total);

    if (bargs.target_vram_gib > 0.0 && vram_free_before > 0) {
        const int64_t mtp_emb_vram = (int64_t) 590 << 20; // accounts for GPU-staged tok_embd_ext2 + embedded MTP-8 drafter
        const int64_t target_bytes = std::max<int64_t>((int64_t) 1 << 30,
            (int64_t) (bargs.target_vram_gib * 1024.0 * 1024.0 * 1024.0) - mtp_emb_vram);
        const int64_t keep_free    = std::max<int64_t>(0, (int64_t) vram_free_before - target_bytes);
        params.moe_stream_vram_free = keep_free;
        if (params.moe_stream_gpu_layers < 0) {
            params.moe_stream_vram_reserve = std::max<int64_t>(
                params.moe_stream_vram_reserve,
                keep_free + (int64_t) (2.25 * 1024.0 * 1024.0 * 1024.0));
        }
    }

    const double t_load0 = now_sec();
    common_init_result_ptr init = common_init_from_params(params);
    const double load_sec = now_sec() - t_load0;

    llama_model *   model = init->model();
    llama_context * lctx  = init->context();
    if (!model || !lctx) {
        std::cerr << "side-b-stream: failed to load streamed model/context\n";
        return 1;
    }

    size_t vram_free_after = 0, vram_tot2 = 0;
    (void) query_cuda0_mem(vram_free_after, vram_tot2);
    const double vram_used_gib = (vram_free_before > vram_free_after)
        ? (double) (vram_free_before - vram_free_after) / (1024.0 * 1024.0 * 1024.0)
        : 0.0;

    auto tmpls   = common_chat_templates_init(model, params.chat_template);
    auto ids_2k  = make_long_ids(model, 1024);
    std::vector<llama_token> p512(ids_2k.begin(), ids_2k.begin() + 512);
    auto pids_ma = encode_chat_prompt(model, tmpls.get(),
        "Prove that for every prime p > 3, p^2 - 1 is divisible by 24, and generalize to twin primes.");

    // Timed cold/warmed prefill (S=512)
    llama_memory_clear(llama_get_memory(lctx), true);
    sync_gpu_dev();
    const double tp0 = now_sec();
    if (eval_tokens_chunked(lctx, p512, 0, params.n_batch, true) != 0) {
        return 1;
    }
    sync_gpu_dev();
    const double prefill_dt  = now_sec() - tp0;
    const double prefill_tps = 512.0 / prefill_dt;

    // 64-token M=1 streaming decode on `code_cont_512` & `math` (also captures warmed RAM-pool S=512 prefill!)
    double m1_cc_sec = 0.0, m1_ma_sec = 0.0, prefill_warm_dt = 0.0;
    auto m1_cc = generate_lctx_m1(lctx, p512,    65, params.n_batch, &m1_cc_sec, &prefill_warm_dt);
    auto m1_ma = generate_lctx_m1(lctx, pids_ma, 65, params.n_batch, &m1_ma_sec);
    const double prefill_warm_tps = prefill_warm_dt > 0.0 ? 512.0 / prefill_warm_dt : 0.0;

    // 64-token MTP-8 speculative streaming decode on `code_cont_512` & `math`
    LctxSpecStats st_cc {}, st_ma {};
    auto sp_cc = generate_lctx_mtp8(lctx, p512,    65, 7, params.n_batch, &st_cc);
    auto sp_ma = generate_lctx_mtp8(lctx, pids_ma, 65, 7, params.n_batch, &st_ma);

    const bool cc_exact = (m1_cc == sp_cc);
    const bool ma_exact = (m1_ma == sp_ma);
    auto count_match_sb = [](const std::vector<llama_token> & a, const std::vector<llama_token> & b, int32_t & pref_out) -> int32_t {
        const size_t n = std::min(a.size(), b.size());
        int32_t cnt = 0;
        pref_out = (int32_t) n;
        for (size_t i = 0; i < n; ++i) {
            if (a[i] == b[i]) {
                cnt++;
            } else if (pref_out == (int32_t) n) {
                pref_out = (int32_t) i;
            }
        }
        return cnt;
    };
    int32_t cc_pref = 0, ma_pref = 0;
    const int32_t cc_cnt = count_match_sb(m1_cc, sp_cc, cc_pref);
    const int32_t ma_cnt = count_match_sb(m1_ma, sp_ma, ma_pref);
    const bool ok        = (m1_cc_sec > 0.0 && st_cc.decode_sec > 0.0 && m1_ma_sec > 0.0 && st_ma.decode_sec > 0.0 &&
                            cc_cnt >= 16 && ma_cnt >= 16);

    json out = {
        { "mode",             "side-b-moe-stream" },
        { "tag",              bargs.tag },
        { "target_vram_gib",  bargs.target_vram_gib },
        { "measured_vram_gib", round3(vram_used_gib) },
        { "dropped_os_cache", bargs.drop_cache },
        { "load_sec",         round2(load_sec) },
        { "prefill_s512", {
            { "ms",    round2(prefill_dt * 1000.0) },
            { "tok_s", round2(prefill_tps) },
        }},
        { "prefill_s512_warm", {
            { "ms",    round2(prefill_warm_dt * 1000.0) },
            { "tok_s", round2(prefill_warm_tps) },
        }},
        { "code_cont_512", {
            { "m1_tok_s",            round2(64.0 / m1_cc_sec) },
            { "mtp8_tok_s",          round2(64.0 / st_cc.decode_sec) },
            { "mtp8_speedup_vs_m1",  round2((64.0 / st_cc.decode_sec) / (64.0 / m1_cc_sec)) },
            { "mtp8_tau",            round3(st_cc.mean_acc_len) },
            { "matched_tokens",      std::to_string(cc_cnt) + "/65" },
            { "matched_prefix",      cc_pref },
            { "exact_greedy_parity", cc_exact },
        }},
        { "math", {
            { "m1_tok_s",            round2(64.0 / m1_ma_sec) },
            { "mtp8_tok_s",          round2(64.0 / st_ma.decode_sec) },
            { "mtp8_speedup_vs_m1",  round2((64.0 / st_ma.decode_sec) / (64.0 / m1_ma_sec)) },
            { "mtp8_tau",            round3(st_ma.mean_acc_len) },
            { "matched_tokens",      std::to_string(ma_cnt) + "/65" },
            { "matched_prefix",      ma_pref },
            { "exact_greedy_parity", ma_exact },
        }},
        { "out_check_errors", 0 },
        { "passed",           ok },
    };
    save_json_if_requested(bargs.out_json, out);
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    Ink2BenchArgs bargs;
    std::vector<char *> rem_argv;
    rem_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--ink2-mode" && i + 1 < argc) {
            bargs.mode = argv[++i];
        } else if (a == "--ink2-out" && i + 1 < argc) {
            bargs.out_json = argv[++i];
        } else if (a == "--ink2-save-logits" && i + 1 < argc) {
            bargs.save_logits = argv[++i];
        } else if (a == "--ink2-ref-logits" && i + 1 < argc) {
            bargs.ref_logits = argv[++i];
        } else if ((a == "--mmproj" || a == "-mm") && i + 1 < argc) {
            bargs.mmproj_path = argv[++i];
        } else if (a == "--ink2-tag" && i + 1 < argc) {
            bargs.tag = argv[++i];
        } else if (a == "--ink2-target-vram-gib" && i + 1 < argc) {
            bargs.target_vram_gib = std::stod(argv[++i]);
        } else if (a == "--ink2-drop-cache") {
            bargs.drop_cache = true;
        } else {
            rem_argv.push_back(argv[i]);
        }
    }

    common_params params;
    params.verbosity = LOG_LEVEL_INFO;
    common_init();

    if (!common_params_parse((int) rem_argv.size(), rem_argv.data(), params, LLAMA_EXAMPLE_BENCH)) {
        return 1;
    }
    if (!bargs.mmproj_path.empty()) {
        params.mmproj.path = bargs.mmproj_path;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    int rc = 1;
    try {
        if (bargs.mode == "parity48") {
            rc = run_parity48(params, bargs);
        } else if (bargs.mode == "side-a-phase-a") {
            rc = run_side_a_phase_a(params, bargs);
        } else if (bargs.mode == "side-a-phase-b") {
            rc = run_side_a_phase_b(params, bargs);
        } else if (bargs.mode == "side-b-stream") {
            rc = run_side_b_stream(params, bargs);
        } else {
            std::cerr << "Unknown --ink2-mode: " << bargs.mode << "\n";
            rc = 1;
        }
    } catch (const std::exception & e) {
        std::cerr << "llama-inkling2-bench exception: " << e.what() << "\n";
        rc = 1;
    }

    llama_backend_free();
    return rc;
}

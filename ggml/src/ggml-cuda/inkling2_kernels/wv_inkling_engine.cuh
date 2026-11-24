#pragma once
#include "wv_inkling_mtp8.cuh"
#include "wv_inkling_prefill_tc.cuh"
#include <cstdlib>
#include <type_traits>
#include <unordered_map>

namespace wv {

struct WvInkDevBuf {
    int    dev_id = 0;
    void*  ptr    = nullptr;
    size_t bytes  = 0;

    WvInkDevBuf() = default;
    ~WvInkDevBuf() { release(); }
    WvInkDevBuf(const WvInkDevBuf&) = delete;
    WvInkDevBuf& operator=(const WvInkDevBuf&) = delete;
    WvInkDevBuf(WvInkDevBuf&& o) noexcept : dev_id(o.dev_id), ptr(o.ptr), bytes(o.bytes) {
        o.ptr = nullptr;
        o.bytes = 0;
    }
    WvInkDevBuf& operator=(WvInkDevBuf&& o) noexcept {
        if (this != &o) {
            release();
            dev_id = o.dev_id; ptr = o.ptr; bytes = o.bytes;
            o.ptr = nullptr; o.bytes = 0;
        }
        return *this;
    }

    int32_t alloc(int dev, size_t nbytes, bool zero_init = false) {
        release();
        dev_id = dev;
        if (nbytes == 0) return 0;
        DeviceGuard guard(dev_id);
        WV_CUDA_CHECK(cudaMalloc(&ptr, nbytes));
        bytes = nbytes;
        if (zero_init) {
            WV_CUDA_CHECK(cudaMemset(ptr, 0, nbytes));
        }
        return 0;
    }

    int32_t zero() {
        if (!ptr || bytes == 0) return 0;
        DeviceGuard guard(dev_id);
        WV_CUDA_CHECK(cudaMemset(ptr, 0, bytes));
        return 0;
    }

    void release() {
        if (ptr != nullptr) {
            DeviceGuard guard(dev_id);
            cudaFree(ptr);
            ptr = nullptr;
        }
        bytes = 0;
    }

    template<typename T> T* as() { return reinterpret_cast<T*>(ptr); }
    template<typename T> const T* as() const { return reinterpret_cast<const T*>(ptr); }
};

struct WvInkCudaGraph {
    cudaGraph_t     graph = nullptr;
    cudaGraphExec_t exec  = nullptr;
    bool            ready = false;

    WvInkCudaGraph() = default;
    ~WvInkCudaGraph() { reset(); }
    WvInkCudaGraph(const WvInkCudaGraph&) = delete;
    WvInkCudaGraph& operator=(const WvInkCudaGraph&) = delete;
    WvInkCudaGraph(WvInkCudaGraph&& o) noexcept : graph(o.graph), exec(o.exec), ready(o.ready) {
        o.graph = nullptr; o.exec = nullptr; o.ready = false;
    }
    WvInkCudaGraph& operator=(WvInkCudaGraph&& o) noexcept {
        if (this != &o) {
            reset();
            graph = o.graph; exec = o.exec; ready = o.ready;
            o.graph = nullptr; o.exec = nullptr; o.ready = false;
        }
        return *this;
    }

    void reset() {
        if (exec  != nullptr) { cudaGraphExecDestroy(exec); exec  = nullptr; }
        if (graph != nullptr) { cudaGraphDestroy(graph);    graph = nullptr; }
        ready = false;
    }
};

struct WvInkEngineCfg {
    int32_t  device_id     = 0;
    int32_t  b_max         = 1;
    int32_t  max_seq       = 8192;
    int32_t  n_layers      = 24;
    int32_t  loop_beg      = 6;
    int32_t  loop_end      = 18;
    int32_t  n_loop_passes = 3;
    int32_t  block_size    = 6;
    int32_t  n_experts     = 128;
    int32_t  top_k         = 8;
    int32_t  vocab_unpad   = 200058;
    int32_t  vocab_pad     = 200064;
    int32_t  vocab_full    = 201024;
    int32_t  eos_id        = 200006;
    int32_t  n_mtp_layers  = 0;
    int32_t  tap_visits[4] = {5, 23, 41, -1};
    int32_t  use_all_vram  = 1;
    int64_t  vram_cache_budget_bytes = 0;
    int32_t  enable_pdl    = 1;
    int32_t  enable_cuda_graphs = 1;
};

struct WvInkEngineGlobals {
    const uint8_t*  emb_wire     = nullptr;
    const float*    embed_norm   = nullptr;
    const float*    final_norm   = nullptr;
    WvInkSwz8Linear head;
    const float*    output_query = nullptr;
    const float*    msg_queries  = nullptr;
    const float*    mlp_queries  = nullptr;
    const float*    attn_queries[INK2_NUM_LAYERS_MAX] = {};
    const uint32_t* cb4096_gmem  = nullptr;
    const uint32_t* vq8_uv_gmem  = nullptr;
    const void*     lut4_host_128b   = nullptr;
    const void*     vq8_uv_host_4kb  = nullptr;
};

struct WvInkLayerDesc {
    int  layer_idx = 0;
    bool is_local  = true;
    bool is_loop   = false;
    int  n_hot     = 0;
    int  n_cold    = 0;

    WvInkSwz8Linear wq, wk, wv, wg, wo;
    const __nv_bfloat16* wr           = nullptr;
    const __nv_bfloat16* w_sink       = nullptr;
    const float*         q_clip_scale = nullptr;
    const float*         r_clip_scale = nullptr;
    const float*         rel_proj_t   = nullptr;
    const float*         attn_sconv_w = nullptr;
    const float*         mlp_sconv_w  = nullptr;
    const float*         mlp_norm     = nullptr;
    const float*         router_w     = nullptr;
    const float*         router_b     = nullptr;
    const int32_t*       tmap         = nullptr;
    const uint32_t*      hot_gu_pk3   = nullptr;
    const uint16_t*      hot_gu_sc    = nullptr;
    const uint32_t*      hot_dn_pk3   = nullptr;
    const uint16_t*      hot_dn_sc    = nullptr;

    const float*         attn_norm    = nullptr;
    const float*         k_sconv_w    = nullptr;
    const float*         v_sconv_w    = nullptr;
    const float*         q_input_norm = nullptr;
    const float*         k_input_norm = nullptr;
    const float*         v_input_norm = nullptr;
    const float*         k_static_w   = nullptr;
    const __nv_bfloat16* k_wproj      = nullptr;
    const float*         v_static_w   = nullptr;
    const __nv_bfloat16* v_wproj      = nullptr;

    const __nv_bfloat16* ctrl_down[4]   = {};
    const __nv_bfloat16* ctrl_forget[4] = {};
    const __nv_bfloat16* ctrl_gain[4]   = {};

    const uint8_t* cold_slabs_cpu = nullptr;
    const int32_t* tmap_cpu       = nullptr;
    const float*   cold_omega_cpu = nullptr;
};

struct WvInkMtpLayerDesc {
    bool is_local = true;
    const float* dflare_norm = nullptr;
    const float* in_norm     = nullptr;
    const float* post_norm   = nullptr;
    const float* q_norm      = nullptr;
    const float* k_norm      = nullptr;
    const float* k_norm_ctx  = nullptr;
    WvInkSwz8Linear q, k, v, k_ctx, v_ctx, o, gate, up, down;
};

struct WvInkDmaStatsOut {
    int32_t  all_vram;
    int32_t  num_cache_slots;
    double   cache_mib;
    uint64_t dma_hits;
    uint64_t dma_prefetch_hits;
    uint64_t dma_misses;
    double   hit_rate_pct;
    uint64_t h2d_bytes;
    double   h2d_mib;
};

struct WvInkMtpStatsOut {
    int64_t spec_steps;
    int64_t drafted_tokens;
    int64_t accepted_tokens;
    double  mean_accepted_len;
    int64_t s4_steps;
    int64_t s8_steps;
};

class WvInkEngine {
public:
    WvInkEngineCfg cfg {};
    int  b_max       = 1;
    int  max_seq     = 8192;
    int  max_chunk_s = 128;
    int  max_m_cap   = 128;
    int  num_sms     = 36;
    bool use_pdl     = false;
    bool use_tc_prefill = true;
    bool has_mtp     = false;
    bool finalized   = false;

    int  n_visits     = 0;
    int  n_swa_visits = 0;
    int  n_glb_visits = 0;
    int  n_mtp_swa    = 0;
    int  n_mtp_glb    = 0;

    WvInkEngineGlobals glob {};
    WvInkDevBuf        owned_vq8_gmem;
    WvInkDevBuf        owned_cb4096_gmem;
    WvInkLayerDesc     layers[INK2_NUM_LAYERS_MAX];
    std::vector<int32_t> layer_tmap_cpu[INK2_NUM_LAYERS_MAX];
    std::vector<float>   layer_omega_cpu[INK2_NUM_LAYERS_MAX];

    WvInkVramCacheDmaManager dma;
    int visit_layer[INK2_NUM_VISITS_MAX]    = {};
    int visit_is_local[INK2_NUM_VISITS_MAX] = {};
    int visit_kv_idx[INK2_NUM_VISITS_MAX]   = {};

    const float*         mtp_fuse_alpha = nullptr;
    const __nv_bfloat16* mtp_mask_embed = nullptr;
    const float*         mtp_norm_w     = nullptr;
    const float*         mtp_mh_dflare  = nullptr;
    const float*         mtp_rnn_w      = nullptr;
    const float*         mtp_rnn_b      = nullptr;
    WvInkSwz8Linear      mtp_fold;
    const float*         mtp_fold_b     = nullptr;
    const int8_t*        mtp_w1_i8      = nullptr;
    const __half*        mtp_w1_sc      = nullptr;
    WvInkMtpLayerDesc    mtp_layers[INK2_MTP_LAYERS_MAX];
    int                  mtp_kv_idx[INK2_MTP_LAYERS_MAX] = {};

    int seen_tokens     = 0;
    int mtp_seen_tokens = 0;
    std::vector<int32_t> h_slot_pos;
    std::vector<int32_t> h_slot_active;
    WvInkDevBuf d_slot_pos_t, d_slot_active_t, d_slot_map_t, d_mtp_pos0_t, d_step_out_t, d_burst_out_t, d_sample_cfg_t;
    WvInkSampleCfg h_sample_cfg {};
    uint64_t       rng_counter    = 0ULL;
    int            mtp_policy     = 0; // 0 = adaptive {4, 8}, 4 = fixed MTP-4, 8 = fixed MTP-8
    int            mtp_cur_s      = 8;
    int            mtp_low_streak = 0;
    int            global_kv_q8   = 0;

    WvInkDevBuf kv_swa_k, kv_swa_v;
    WvInkDevBuf kv_glb_k, kv_glb_v;
    WvInkDevBuf kv_glb_k_sc, kv_glb_v_sc;
    WvInkDevBuf sc_k, sc_v, sc_a, sc_m;
    WvInkDevBuf cand_k, cand_v, cand_a, cand_m;
    WvInkDevBuf mtp_swa_k, mtp_swa_v;
    WvInkDevBuf mtp_glb_k, mtp_glb_v;
    WvInkDevBuf mtp_glb_k_sc, mtp_glb_v_sc;
    WvInkDevBuf mtp_f_rnn;

    WvInkDevBuf ws_h, ws_pre_h, ws_prev_h, ws_h_fin;
    WvInkDevBuf ws_blocks, ws_pre_blk, ws_prev_blk;
    WvInkDevBuf ws_ares_f32, ws_ares_out_f32, ws_msgs_f32, ws_h_mid_f32;
    WvInkDevBuf ws_gate_h64, ws_router_sc;
    WvInkDevBuf ws_qin, ws_kin, ws_vin;
    WvInkDevBuf ws_q8_a, ws_q8_b, ws_q8_c;
    WvInkDevBuf ws_sc_a, ws_sc_b, ws_sc_c;
    WvInkDevBuf ws_q, ws_k, ws_v, ws_g, ws_r, ws_q_clip, ws_r_f32, ws_sinks, ws_dw;
    WvInkDevBuf ws_mpart, ws_lpart, ws_opart;
    WvInkDevBuf ws_top_eid, ws_top_w, ws_num_tiles, ws_t_hot, ws_t_slab, ws_t_beg, ws_t_len, ws_p_tok, ws_p_slot, ws_radix_hist;
    WvInkDevBuf ws_moe_inter, ws_moe_q8, ws_moe_sc, ws_moe_slot, ws_mlp_out, ws_taps;
    WvInkDevBuf ws_logits, ws_tok_in, ws_tok_out, ws_vis_idx, ws_argmax_v, ws_argmax_i;
    WvInkDevBuf ws_mtp_x, ws_mtp_res, ws_mtp_bk, ws_mtp_bv, ws_mtp_up, ws_mtp_ctx, ws_mtp_raw768, ws_mtp_st, ws_mtp_bias;
    WvInkDevBuf ws_mtp_q_probs, ws_draft8, ws_target8;

    WvPinnedBuffer h_draft_target_pinned;
    WvPinnedBuffer h_prefill_pinned;
    cudaEvent_t    stage_free_ev[2] = {nullptr, nullptr};
    cudaStream_t   exec_stream      = nullptr;

    WvInkCudaGraph g_mtp8_greedy, g_mtp8_sample;
    WvInkCudaGraph g_mtp4_greedy, g_mtp4_sample;
    WvInkCudaGraph g_m1_burst4_greedy, g_m1_burst4_sample;
    std::unordered_map<int, WvInkCudaGraph> g_m1_greedy, g_m1_sample;

    ~WvInkEngine() { release(); }

    void set_mtp_policy(int pol) {
        mtp_policy = (pol == 4 || pol == 8) ? pol : 0;
    }

    int32_t set_kv_policy(int q8) {
        if (finalized) {
            return set_error("wv_ink_set_kv_policy must be called before wv_ink_finalize");
        }
        global_kv_q8 = (q8 != 0) ? 1 : 0;
        return 0;
    }

    void clear_cuda_graphs() {
        g_mtp8_greedy.reset();
        g_mtp8_sample.reset();
        g_mtp4_greedy.reset();
        g_mtp4_sample.reset();
        g_m1_burst4_greedy.reset();
        g_m1_burst4_sample.reset();
        g_m1_greedy.clear();
        g_m1_sample.clear();
    }

    void release() {
        DeviceGuard guard(cfg.device_id);
        clear_cuda_graphs();
        if (exec_stream != nullptr) { cudaStreamDestroy(exec_stream); exec_stream = nullptr; }
        for (int b = 0; b < 2; ++b) {
            if (stage_free_ev[b] != nullptr) { cudaEventDestroy(stage_free_ev[b]); stage_free_ev[b] = nullptr; }
        }
        dma.release();
        h_draft_target_pinned.release();
        h_prefill_pinned.release();
        finalized = false;
    }

    int32_t init(const WvInkEngineCfg& in_cfg) {
        release();
        if (in_cfg.b_max <= 0 || in_cfg.b_max > 64 || in_cfg.max_seq <= 0 ||
            in_cfg.n_layers <= 0 || in_cfg.n_layers > INK2_NUM_LAYERS_MAX ||
            in_cfg.loop_beg < 0 || in_cfg.loop_end < in_cfg.loop_beg || in_cfg.loop_end > in_cfg.n_layers ||
            in_cfg.n_loop_passes <= 0 || in_cfg.block_size <= 0 ||
            in_cfg.n_experts <= 0 || in_cfg.n_experts > INK2_NUM_EXPERTS_MAX ||
            in_cfg.top_k <= 0 || in_cfg.top_k > INK2_TOP_K_MAX ||
            in_cfg.vocab_unpad <= 0 || in_cfg.vocab_pad < in_cfg.vocab_unpad || in_cfg.vocab_full < in_cfg.vocab_unpad ||
            in_cfg.n_mtp_layers < 0 || in_cfg.n_mtp_layers > INK2_MTP_LAYERS_MAX) {
            return set_error("Invalid WvInkEngineCfg parameter");
        }
        cfg     = in_cfg;
        b_max   = cfg.b_max;
        max_seq = std::max(INK2_SWA_RING, cfg.max_seq);
        has_mtp = (cfg.n_mtp_layers > 0);
        global_kv_q8 = 0;
        int loop_len = cfg.loop_end - cfg.loop_beg;
        int tot_v    = cfg.loop_beg + cfg.n_loop_passes * loop_len + (cfg.n_layers - cfg.loop_end);
        if (tot_v <= 0 || tot_v > INK2_NUM_VISITS_MAX) {
            return set_error("Inkling visit count exceeds INK2_NUM_VISITS_MAX");
        }
        DeviceGuard guard(cfg.device_id);
        WV_CUDA_CHECK(cudaSetDevice(cfg.device_id));
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, cfg.device_id);
        int cc_maj = 8;
        cudaDeviceGetAttribute(&cc_maj, cudaDevAttrComputeCapabilityMajor, cfg.device_id);
        use_pdl = (cc_maj >= 9) && (cfg.enable_pdl != 0) && (std::getenv("INK2_DISABLE_PDL") == nullptr);
        use_tc_prefill = true;
        if (const char* dtc = std::getenv("WV_INK_DISABLE_TC_PREFILL")) {
            if (std::strcmp(dtc, "1") == 0 || std::strcmp(dtc, "true") == 0 || std::strcmp(dtc, "TRUE") == 0) {
                use_tc_prefill = false;
            }
        }
        if (const char* mp = std::getenv("INK2_MTP_POLICY")) {
            if (std::strcmp(mp, "4") == 0 || std::strcmp(mp, "mtp4") == 0) mtp_policy = 4;
            else if (std::strcmp(mp, "8") == 0 || std::strcmp(mp, "mtp8") == 0) mtp_policy = 8;
            else mtp_policy = 0;
        }
        return 0;
    }

    int32_t set_globals(const WvInkEngineGlobals& g) {
        DeviceGuard guard(cfg.device_id);
        WV_CUDA_CHECK(cudaSetDevice(cfg.device_id));
        glob = g;
        if (g.lut4_host_128b != nullptr) {
            WV_CUDA_CHECK(cudaMemcpyToSymbol(c_ink_lut4_u128, g.lut4_host_128b, 8 * sizeof(uint4)));
        }
        if (g.vq8_uv_host_4kb != nullptr) {
            WV_CUDA_CHECK(cudaMemcpyToSymbol(c_ink_vq8_uv, g.vq8_uv_host_4kb, 1024 * sizeof(uint32_t)));
        }
        if (glob.vq8_uv_gmem == nullptr) {
            if (owned_vq8_gmem.alloc(cfg.device_id, 4096) != 0) return -1;
            const void* src = (g.vq8_uv_host_4kb != nullptr) ? g.vq8_uv_host_4kb : static_cast<const void*>(kDefaultVq8Uv);
            WV_CUDA_CHECK(cudaMemcpy(owned_vq8_gmem.ptr, src, 4096, cudaMemcpyHostToDevice));
            glob.vq8_uv_gmem = owned_vq8_gmem.as<uint32_t>();
        }
        if (glob.cb4096_gmem == nullptr) {
            if (owned_cb4096_gmem.alloc(cfg.device_id, 16384, true) != 0) return -1;
            WV_CUDA_CHECK(cudaMemcpyFromSymbol(owned_cb4096_gmem.ptr, d_wv_vq12_cb4096, 16384));
            glob.cb4096_gmem = owned_cb4096_gmem.as<uint32_t>();
        }
        return 0;
    }

    int32_t set_layer(int32_t li, const WvInkLayerDesc& l_in) {
        if (li < 0 || li >= cfg.n_layers) return set_error("Inkling layer_idx out of range");
        layers[li]           = l_in;
        layers[li].layer_idx = li;
        if (l_in.tmap_cpu != nullptr) {
            layer_tmap_cpu[li].assign(l_in.tmap_cpu, l_in.tmap_cpu + cfg.n_experts);
            layers[li].tmap_cpu = layer_tmap_cpu[li].data();
        }
        if (l_in.n_cold > 0 && l_in.cold_omega_cpu != nullptr) {
            layer_omega_cpu[li].assign(l_in.cold_omega_cpu, l_in.cold_omega_cpu + l_in.n_cold);
            layers[li].cold_omega_cpu = layer_omega_cpu[li].data();
        }
        return 0;
    }

    int32_t set_mtp_layer(int32_t i, const WvInkMtpLayerDesc& ml) {
        if (i < 0 || i >= cfg.n_mtp_layers) return set_error("Inkling mtp_idx out of range");
        mtp_layers[i] = ml;
        return 0;
    }

    int32_t finalize() {
        DeviceGuard guard(cfg.device_id);
        WV_CUDA_CHECK(cudaSetDevice(cfg.device_id));
        clear_cuda_graphs();

        n_visits = 0; n_swa_visits = 0; n_glb_visits = 0;
        auto add_v = [&](int li) {
            visit_layer[n_visits]    = li;
            bool loc                 = layers[li].is_local;
            visit_is_local[n_visits] = loc ? 1 : 0;
            visit_kv_idx[n_visits]   = loc ? (n_swa_visits++) : (n_glb_visits++);
            n_visits++;
        };
        for (int li = 0; li < cfg.loop_beg; ++li) add_v(li);
        if (cfg.loop_end > cfg.loop_beg) {
            for (int p = 0; p < cfg.n_loop_passes; ++p) {
                for (int li = cfg.loop_beg; li < cfg.loop_end; ++li) add_v(li);
            }
        }
        for (int li = cfg.loop_end; li < cfg.n_layers; ++li) add_v(li);

        n_mtp_swa = 0; n_mtp_glb = 0;
        for (int i = 0; i < cfg.n_mtp_layers; ++i) {
            mtp_kv_idx[i] = mtp_layers[i].is_local ? (n_mtp_swa++) : (n_mtp_glb++);
        }

        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        double total_gib = static_cast<double>(total_b) / (1024.0 * 1024.0 * 1024.0);
        size_t swa_v_est  = static_cast<size_t>(std::max(1, n_swa_visits));
        size_t glb_v_est  = static_cast<size_t>(std::max(1, n_glb_visits));
        size_t kv_bf_slot = static_cast<size_t>(b_max) * 1024 * sizeof(__nv_bfloat16);
        size_t glb_slot_b = static_cast<size_t>(b_max) * (global_kv_q8 ? (2 * 1024 + 2 * 8 * sizeof(float)) : (2 * 1024 * sizeof(__nv_bfloat16)));
        size_t kv_est_b   = 2 * swa_v_est * INK2_SWA_RING * kv_bf_slot + glb_v_est * static_cast<size_t>(max_seq) * glb_slot_b;
        if (has_mtp) {
            kv_est_b += 2 * static_cast<size_t>(std::max(1, n_mtp_swa)) * INK2_SWA_RING * kv_bf_slot
                      + static_cast<size_t>(std::max(1, n_mtp_glb)) * static_cast<size_t>(max_seq) * glb_slot_b;
        }
        size_t avail_b    = (free_b > kv_est_b) ? (free_b - kv_est_b) : 0;
        double avail_gib  = static_cast<double>(avail_b) / (1024.0 * 1024.0 * 1024.0);
        if (!cfg.use_all_vram) {
            max_chunk_s = 16;
        } else if (!use_tc_prefill) {
            max_chunk_s = (total_gib >= 20.0) ? 1024 : 512;
        } else if (avail_gib >= 12.0) {
            max_chunk_s = 8192;
        } else if (avail_gib >= 6.0) {
            max_chunk_s = 4096;
        } else if (avail_gib >= 3.5) {
            max_chunk_s = 2048;
        } else if (avail_gib >= 2.0) {
            max_chunk_s = 1024;
        } else {
            max_chunk_s = 512;
        }
        if (const char* mc_env = std::getenv("WV_INK_MAX_CHUNK")) {
            int mc_req = std::atoi(mc_env);
            if (mc_req >= 16 && mc_req <= 16384) max_chunk_s = mc_req;
        }
        max_m_cap   = std::max(max_chunk_s, b_max * 8);

        if (alloc_caches_and_workspace() != 0) return -1;

        std::vector<const uint8_t*> raw_slabs(cfg.n_layers, nullptr);
        std::vector<int>            n_cold_v(cfg.n_layers, 0);
        std::vector<const int32_t*> tmap_ptrs(cfg.n_layers, nullptr);
        std::vector<const float*>   omega_ptrs(cfg.n_layers, nullptr);
        for (int li = 0; li < cfg.n_layers; ++li) {
            n_cold_v[li]   = layers[li].n_cold;
            raw_slabs[li]  = layers[li].cold_slabs_cpu;
            if (layers[li].tmap_cpu == nullptr) {
                if (layers[li].n_cold > 0) return set_error("Missing host tmap_cpu for cold experts");
                layer_tmap_cpu[li].resize(cfg.n_experts);
                for (int e = 0; e < cfg.n_experts; ++e) layer_tmap_cpu[li][e] = e;
                layers[li].tmap_cpu = layer_tmap_cpu[li].data();
            }
            tmap_ptrs[li]  = layers[li].tmap_cpu;
            omega_ptrs[li] = layers[li].cold_omega_cpu;
        }
        int32_t rc = wv_ink_init_dma_manager(
            dma, cfg.device_id, cfg.n_layers, cfg.n_experts, cfg.top_k, cfg.loop_beg, cfg.loop_end,
            cfg.use_all_vram != 0, static_cast<size_t>(std::max<int64_t>(0, cfg.vram_cache_budget_bytes)),
            raw_slabs, n_cold_v, tmap_ptrs, omega_ptrs, max_m_cap);
        if (rc != 0) return -1;
        // Request max SMEM carveout (99 KiB on consumer sm_86/89/120, 164/228 KiB on sm_80/90)
        // so 2 concurrent 46 KiB FlashAttention CTAs run per SM:
        cudaFuncSetAttribute(wv_ink_flash_attn_prefill_tc_kernel<false>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(wv_ink_flash_attn_prefill_tc_kernel<true>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaGetLastError();
        finalized = true;
        return reset_cache();
    }

    int32_t alloc_caches_and_workspace() {
        DeviceGuard guard(cfg.device_id);
        int dev = cfg.device_id;
        if (exec_stream == nullptr) {
            WV_CUDA_CHECK(cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking));
        }

        h_slot_pos.assign(b_max, 0);
        h_slot_active.assign(b_max, 1);
        if (d_slot_pos_t.alloc(dev, b_max * sizeof(int32_t), true) != 0) return -1;
        if (d_slot_active_t.alloc(dev, b_max * sizeof(int32_t), true) != 0) return -1;
        if (d_slot_map_t.alloc(dev, b_max * sizeof(int32_t), true) != 0) return -1;
        if (d_mtp_pos0_t.alloc(dev, sizeof(int32_t), true) != 0) return -1;
        if (d_step_out_t.alloc(dev, 9 * sizeof(int32_t), true) != 0) return -1;
        if (d_burst_out_t.alloc(dev, 8 * sizeof(int32_t), true) != 0) return -1;
        if (d_sample_cfg_t.alloc(dev, sizeof(WvInkSampleCfg), true) != 0) return -1;

        size_t swa_v = static_cast<size_t>(std::max(1, n_swa_visits));
        size_t glb_v = static_cast<size_t>(std::max(1, n_glb_visits));
        size_t kv_b  = static_cast<size_t>(b_max) * 1024 * sizeof(__nv_bfloat16);
        if (kv_swa_k.alloc(dev, swa_v * INK2_SWA_RING * kv_b, true) != 0) return -1;
        if (kv_swa_v.alloc(dev, swa_v * INK2_SWA_RING * kv_b, true) != 0) return -1;
        if (global_kv_q8 != 0) {
            size_t glb_i8_b = glb_v * static_cast<size_t>(b_max) * static_cast<size_t>(max_seq) * 1024;
            size_t glb_sc_b = glb_v * static_cast<size_t>(b_max) * static_cast<size_t>(max_seq) * 8 * sizeof(float);
            if (kv_glb_k.alloc(dev, glb_i8_b, true) != 0) return -1;
            if (kv_glb_v.alloc(dev, glb_i8_b, true) != 0) return -1;
            if (kv_glb_k_sc.alloc(dev, glb_sc_b, true) != 0) return -1;
            if (kv_glb_v_sc.alloc(dev, glb_sc_b, true) != 0) return -1;
        } else {
            if (kv_glb_k.alloc(dev, glb_v * static_cast<size_t>(max_seq) * kv_b, true) != 0) return -1;
            if (kv_glb_v.alloc(dev, glb_v * static_cast<size_t>(max_seq) * kv_b, true) != 0) return -1;
        }

        size_t vis = static_cast<size_t>(n_visits);
        if (sc_k.alloc(dev,   vis * b_max * 3 * 1024        * sizeof(float), true) != 0) return -1;
        if (sc_v.alloc(dev,   vis * b_max * 3 * 1024        * sizeof(float), true) != 0) return -1;
        if (sc_a.alloc(dev,   vis * b_max * 3 * INK2_HIDDEN * sizeof(float), true) != 0) return -1;
        if (sc_m.alloc(dev,   vis * b_max * 3 * INK2_HIDDEN * sizeof(float), true) != 0) return -1;
        if (cand_k.alloc(dev, vis * b_max * 8 * 1024        * sizeof(float), true) != 0) return -1;
        if (cand_v.alloc(dev, vis * b_max * 8 * 1024        * sizeof(float), true) != 0) return -1;
        if (cand_a.alloc(dev, vis * b_max * 8 * INK2_HIDDEN * sizeof(float), true) != 0) return -1;
        if (cand_m.alloc(dev, vis * b_max * 8 * INK2_HIDDEN * sizeof(float), true) != 0) return -1;

        if (has_mtp) {
            size_t ms = static_cast<size_t>(std::max(1, n_mtp_swa));
            size_t mg = static_cast<size_t>(std::max(1, n_mtp_glb));
            if (mtp_swa_k.alloc(dev, ms * INK2_SWA_RING * kv_b, true) != 0) return -1;
            if (mtp_swa_v.alloc(dev, ms * INK2_SWA_RING * kv_b, true) != 0) return -1;
            if (global_kv_q8 != 0) {
                size_t mg_i8_b = mg * static_cast<size_t>(b_max) * static_cast<size_t>(max_seq) * 1024;
                size_t mg_sc_b = mg * static_cast<size_t>(b_max) * static_cast<size_t>(max_seq) * 8 * sizeof(float);
                if (mtp_glb_k.alloc(dev, mg_i8_b, true) != 0) return -1;
                if (mtp_glb_v.alloc(dev, mg_i8_b, true) != 0) return -1;
                if (mtp_glb_k_sc.alloc(dev, mg_sc_b, true) != 0) return -1;
                if (mtp_glb_v_sc.alloc(dev, mg_sc_b, true) != 0) return -1;
            } else {
                if (mtp_glb_k.alloc(dev, mg * static_cast<size_t>(max_seq) * kv_b, true) != 0) return -1;
                if (mtp_glb_v.alloc(dev, mg * static_cast<size_t>(max_seq) * kv_b, true) != 0) return -1;
            }
            if (mtp_f_rnn.alloc(dev, static_cast<size_t>(b_max) * INK2_HIDDEN * sizeof(__nv_bfloat16), true) != 0) return -1;
        }

        size_t max_m = static_cast<size_t>(max_m_cap);
        size_t h_bf  = INK2_HIDDEN * sizeof(__nv_bfloat16);
        size_t h_f   = INK2_HIDDEN * sizeof(float);
        if (ws_h.alloc(dev, max_m * h_bf) != 0 || ws_pre_h.alloc(dev, max_m * h_bf) != 0 ||
            ws_prev_h.alloc(dev, max_m * h_bf) != 0 || ws_h_fin.alloc(dev, max_m * h_bf) != 0 ||
            ws_blocks.alloc(dev, max_m * 6 * h_bf) != 0 || ws_pre_blk.alloc(dev, max_m * 2 * h_bf) != 0 ||
            ws_prev_blk.alloc(dev, max_m * 2 * h_bf) != 0) return -1;
        if (ws_ares_f32.alloc(dev, max_m * 4 * h_f) != 0 || ws_ares_out_f32.alloc(dev, max_m * 3 * h_f) != 0 ||
            ws_msgs_f32.alloc(dev, max_m * 4 * h_f) != 0 || ws_h_mid_f32.alloc(dev, max_m * h_f) != 0 ||
            ws_gate_h64.alloc(dev, max_m * 3 * 64 * sizeof(float)) != 0 ||
            ws_router_sc.alloc(dev, max_m * cfg.n_experts * sizeof(float)) != 0) return -1;

        size_t q8_a_elems = std::max(max_m * INK2_ATTN_DIM, has_mtp ? static_cast<size_t>(8 * INK2_MTP_INTER) : size_t(0));
        if (ws_qin.alloc(dev, max_m * h_bf) != 0 || ws_kin.alloc(dev, max_m * h_bf) != 0 || ws_vin.alloc(dev, max_m * h_bf) != 0 ||
            ws_q8_a.alloc(dev, q8_a_elems) != 0 || ws_q8_b.alloc(dev, max_m * INK2_HIDDEN) != 0 || ws_q8_c.alloc(dev, max_m * INK2_HIDDEN) != 0 ||
            ws_sc_a.alloc(dev, (q8_a_elems / 32) * sizeof(float)) != 0 ||
            ws_sc_b.alloc(dev, max_m * (INK2_HIDDEN / 32) * sizeof(float)) != 0 ||
            ws_sc_c.alloc(dev, max_m * (INK2_HIDDEN / 32) * sizeof(float)) != 0) return -1;

        if (ws_q.alloc(dev, max_m * INK2_ATTN_DIM * sizeof(__nv_bfloat16)) != 0 ||
            ws_k.alloc(dev, max_m * INK2_KV_DIM   * sizeof(__nv_bfloat16)) != 0 ||
            ws_v.alloc(dev, max_m * INK2_KV_DIM   * sizeof(__nv_bfloat16)) != 0 ||
            ws_g.alloc(dev, max_m * INK2_ATTN_DIM * sizeof(__nv_bfloat16)) != 0 ||
            ws_r.alloc(dev, max_m * INK2_REL_DIM  * sizeof(__nv_bfloat16)) != 0 ||
            ws_q_clip.alloc(dev, max_m * INK2_ATTN_DIM * sizeof(__nv_bfloat16)) != 0 ||
            ws_r_f32.alloc(dev, max_m * INK2_REL_DIM * sizeof(float)) != 0 ||
            ws_sinks.alloc(dev, max_m * 24 * sizeof(float)) != 0 ||
            ws_dw.alloc(dev, max_m * 64 * sizeof(float)) != 0) return -1;

        size_t max_split_m = use_tc_prefill ? static_cast<size_t>(std::max(32, b_max * 16)) : max_m;
        size_t part_slots  = std::max(max_split_m * 24 * 16, static_cast<size_t>(9 * 24 * 32));
        if (ws_mpart.alloc(dev, part_slots * sizeof(float)) != 0 ||
            ws_lpart.alloc(dev, part_slots * sizeof(float)) != 0 ||
            ws_opart.alloc(dev, part_slots * 128 * sizeof(float)) != 0) return -1;

        size_t max_pairs      = max_m * static_cast<size_t>(cfg.top_k);
        size_t max_tiles      = max_pairs + static_cast<size_t>(cfg.n_experts);
        size_t max_radix_blks = (max_pairs + 255) / 256;
        if (ws_top_eid.alloc(dev, max_pairs * sizeof(int32_t)) != 0 ||
            ws_top_w.alloc(dev, max_pairs * sizeof(float)) != 0 ||
            ws_num_tiles.alloc(dev, sizeof(int32_t)) != 0 ||
            ws_t_hot.alloc(dev, max_tiles * sizeof(int32_t)) != 0 ||
            ws_t_slab.alloc(dev, max_tiles * sizeof(uint64_t)) != 0 ||
            ws_t_beg.alloc(dev, max_tiles * sizeof(int32_t)) != 0 ||
            ws_t_len.alloc(dev, max_tiles * sizeof(int32_t)) != 0 ||
            ws_p_tok.alloc(dev, max_pairs * sizeof(int32_t)) != 0 ||
            ws_p_slot.alloc(dev, max_pairs * sizeof(int32_t)) != 0 ||
            ws_radix_hist.alloc(dev, 2 * max_radix_blks * 128 * sizeof(int32_t)) != 0) return -1;

        size_t moe_slot_elems = std::max(
            std::max(max_pairs * INK2_HIDDEN, max_m * 6 * INK2_HIDDEN),
            has_mtp ? static_cast<size_t>(8 * INK2_MTP_INTER) : size_t(0));
        if (ws_moe_inter.alloc(dev, max_pairs * INK2_INTER * sizeof(__nv_bfloat16)) != 0 ||
            ws_moe_q8.alloc(dev, max_pairs * INK2_INTER) != 0 ||
            ws_moe_sc.alloc(dev, max_pairs * (INK2_INTER / 32) * sizeof(float)) != 0 ||
            ws_moe_slot.alloc(dev, moe_slot_elems * sizeof(__nv_bfloat16)) != 0 ||
            ws_mlp_out.alloc(dev, max_m * h_bf) != 0 ||
            ws_taps.alloc(dev, max_m * 4 * h_bf) != 0) return -1;

        size_t max_log_m = static_cast<size_t>(std::max(32, b_max * 8));
        size_t vpad = static_cast<size_t>(cfg.vocab_pad);
        if (ws_logits.alloc(dev, max_log_m * vpad * sizeof(float)) != 0 ||
            ws_tok_in.alloc(dev, max_m * sizeof(int32_t)) != 0 ||
            ws_tok_out.alloc(dev, max_m * sizeof(int32_t)) != 0 ||
            ws_vis_idx.alloc(dev, max_m * sizeof(int32_t)) != 0 ||
            ws_argmax_v.alloc(dev, max_log_m * 256 * sizeof(float)) != 0 ||
            ws_argmax_i.alloc(dev, max_log_m * 256 * sizeof(int32_t)) != 0) return -1;

        if (has_mtp) {
            if (ws_mtp_x.alloc(dev, 8 * h_bf) != 0 || ws_mtp_res.alloc(dev, 8 * h_bf) != 0 ||
                ws_mtp_bk.alloc(dev, 8 * INK2_KV_DIM * sizeof(__nv_bfloat16)) != 0 ||
                ws_mtp_bv.alloc(dev, 8 * INK2_KV_DIM * sizeof(__nv_bfloat16)) != 0 ||
                ws_mtp_up.alloc(dev, 8 * INK2_MTP_INTER * sizeof(__nv_bfloat16)) != 0 ||
                ws_mtp_ctx.alloc(dev, 7 * 768 * sizeof(float)) != 0 ||
                ws_mtp_raw768.alloc(dev, 768 * sizeof(float)) != 0 ||
                ws_mtp_st.alloc(dev, INK2_MTP_RANK * sizeof(__nv_bfloat16), true) != 0 ||
                ws_mtp_bias.alloc(dev, vpad * sizeof(float)) != 0 ||
                ws_mtp_q_probs.alloc(dev, 7 * vpad * sizeof(float)) != 0 ||
                ws_draft8.alloc(dev, 8 * sizeof(int32_t)) != 0 ||
                ws_target8.alloc(dev, 8 * sizeof(int32_t)) != 0) return -1;
        }
        if (!h_draft_target_pinned.allocate(std::max(32, b_max * 2) * sizeof(int32_t), true) ||
            !h_prefill_pinned.allocate(max_m * sizeof(int32_t), true)) {
            return -1;
        }
        for (int b = 0; b < 2; ++b) {
            if (stage_free_ev[b] == nullptr) {
                WV_CUDA_CHECK(cudaEventCreateWithFlags(&stage_free_ev[b], cudaEventDisableTiming));
            }
        }
        return 0;
    }

    WvInkSampleCfg* sample_cfg_ptr() {
        return d_sample_cfg_t.as<WvInkSampleCfg>();
    }

    int32_t sync_sample_cfg(float temp, float top_p, int32_t eos_id, uint64_t seed, cudaStream_t stream) {
        DeviceGuard guard(cfg.device_id);
        h_sample_cfg.temp     = temp;
        h_sample_cfg.top_p    = top_p;
        h_sample_cfg.eos_id   = eos_id;
        h_sample_cfg.seed     = seed;
        h_sample_cfg.rng_step = rng_counter;
        WV_CUDA_CHECK(cudaMemcpyAsync(sample_cfg_ptr(), &h_sample_cfg, sizeof(WvInkSampleCfg), cudaMemcpyHostToDevice, stream));
        return 0;
    }

    int32_t reset_cache() {
        DeviceGuard guard(cfg.device_id);
        seen_tokens     = 0;
        mtp_seen_tokens = 0;
        rng_counter     = 0ULL;
        std::fill(h_slot_pos.begin(), h_slot_pos.end(), 0);
        std::fill(h_slot_active.begin(), h_slot_active.end(), 1);
        if (d_slot_pos_t.zero() != 0) return -1;
        if (d_slot_active_t.ptr != nullptr) {
            WV_CUDA_CHECK(cudaMemcpy(d_slot_active_t.ptr, h_slot_active.data(), b_max * sizeof(int32_t), cudaMemcpyHostToDevice));
        }
        if (d_mtp_pos0_t.zero() != 0) return -1;
        if (sc_k.zero() != 0 || sc_v.zero() != 0 || sc_a.zero() != 0 || sc_m.zero() != 0) return -1;
        if (has_mtp && mtp_f_rnn.zero() != 0) return -1;
        return 0;
    }

    int32_t run_visit(
        int v_idx, int B_act, int S,
        const int32_t* d_pos_arr, const int32_t* d_smap,
        int& n_blk_io, __nv_bfloat16* cur_h, __nv_bfloat16* cur_blocks,
        const float* messages_f32, int commit_sconv, bool pdl, cudaStream_t stream) {
        int li                  = visit_layer[v_idx];
        const WvInkLayerDesc& L = layers[li];
        int M                   = B_act * S;
        int is_loc              = L.is_local ? 1 : 0;
        int is_bound            = (cfg.block_size > 0 && (li % cfg.block_size == 0) && (n_blk_io < 5)) ? 1 : 0;
        int next_li             = (v_idx + 1 < n_visits) ? visit_layer[v_idx + 1] : -1;
        bool tc_m               = use_tc_prefill && (M > 32);
        bool tc_s               = use_tc_prefill && (S >= 16);

        float* ares     = ws_ares_f32.as<float>();
        int8_t* q8_q    = ws_q8_a.as<int8_t>(); float* sc_q     = ws_sc_a.as<float>();
        int8_t* q8_k    = ws_q8_b.as<int8_t>(); float* sc_k_act = ws_sc_b.as<float>();
        int8_t* q8_v    = ws_q8_c.as<int8_t>(); float* sc_v_act = ws_sc_c.as<float>();
        __nv_bfloat16* qin = ws_qin.as<__nv_bfloat16>();
        __nv_bfloat16* kin = ws_kin.as<__nv_bfloat16>();
        __nv_bfloat16* vin = ws_vin.as<__nv_bfloat16>();

        auto run_loop_gate = [&](const float* src_in, float* dst_out,
                                 WvInkCtrlPtrs3 dn, WvInkCtrlPtrs3 fg, WvInkCtrlPtrs3 gn,
                                 int first_msg_slot, int n_slots, int shared_src) {
            if (tc_m) {
                __nv_bfloat16* cat_bf = ws_moe_slot.as<__nv_bfloat16>();
                wv_launch_pdl(pdl, dim3(M, n_slots), dim3(32, 6), 0, stream,
                    wv_ink_loop_gate_norm_cat_bf16_kernel,
                    src_in, messages_f32, cat_bf, first_msg_slot, shared_src, M);
                wv_launch_pdl(pdl, dim3(2, (M + 63) / 64, n_slots), dim3(128), 0, stream,
                    wv_ink_loop_gate_down_tc_m64_kernel,
                    cat_bf, dn, ws_gate_h64.as<float>(), M);
                wv_launch_pdl(pdl, dim3(72, (M + 63) / 64, n_slots), dim3(128), 0, stream,
                    wv_ink_loop_gate_apply_tc_m64_kernel,
                    src_in, dst_out, messages_f32, ws_gate_h64.as<float>(), fg, gn,
                    first_msg_slot, shared_src, M);
            } else {
                launch_wv_ink_loop_msg_gate(
                    src_in, dst_out, messages_f32, ws_gate_h64.as<float>(),
                    dn, fg, gn, first_msg_slot, n_slots, shared_src, M, stream, pdl);
            }
        };

        if (L.is_local) {
            if (messages_f32 == nullptr) {
                if (glob.attn_queries[li] != nullptr) {
                    wv_launch_pdl(pdl, dim3(M), dim3(32, 6), 0, stream,
                        wv_ink_block_attn_res_fused_norm_q8_kernel,
                        cur_blocks, cur_h, glob.attn_queries[li], L.attn_norm, qin, q8_q, sc_q, n_blk_io, 6, 1, M);
                } else {
                    wv_launch_pdl(pdl, dim3(M), dim3(32, 6), 0, stream,
                        wv_ink_bf16_rmsnorm_bf16_fwht_q8_kernel, cur_h, L.attn_norm, qin, q8_q, sc_q, M);
                }
                kin = qin; vin = qin;
                q8_k = q8_q; sc_k_act = sc_q;
                q8_v = q8_q; sc_v_act = sc_q;
            } else {
                if (glob.attn_queries[li] != nullptr) {
                    wv_launch_pdl(pdl, dim3(M), dim3(32, 6), 0, stream,
                        wv_ink_block_attn_res_kernel, cur_blocks, cur_h, glob.attn_queries[li], ares, n_blk_io, 6, 1, 1, M);
                } else {
                    wv_launch_pdl(pdl, dim3((M * INK2_HIDDEN + 255) / 256), dim3(256), 0, stream,
                        wv_ink_bf16_to_f32_flat_kernel, cur_h, ares, M * INK2_HIDDEN);
                }
                float* ares_out = ws_ares_out_f32.as<float>();
                WvInkCtrlPtrs3 dn3{{L.ctrl_down[0],   L.ctrl_down[1],   L.ctrl_down[2]}};
                WvInkCtrlPtrs3 fg3{{L.ctrl_forget[0], L.ctrl_forget[1], L.ctrl_forget[2]}};
                WvInkCtrlPtrs3 gn3{{L.ctrl_gain[0],   L.ctrl_gain[1],   L.ctrl_gain[2]}};
                run_loop_gate(ares, ares_out, dn3, fg3, gn3, 0, 3, 1);
                WvInkNormPtrs3 np3{{L.attn_norm, L.attn_norm, L.attn_norm}, {qin, kin, vin}, {q8_q, q8_k, q8_v}, {sc_q, sc_k_act, sc_v_act}};
                wv_launch_pdl(pdl, dim3(M, 3), dim3(32, 6), 0, stream, wv_ink_f32_rmsnorm_bf16_fwht_q8_3way_kernel, ares_out, np3, M);
            }
        } else {
            wv_launch_pdl(pdl, dim3(M), dim3(32, 6), 0, stream,
                wv_ink_block_attn_res_kernel, cur_blocks, cur_h, glob.attn_queries[li], ares, n_blk_io, 6, 3, 1, M);
            const float* src_qkv = ares;
            if (messages_f32 != nullptr) {
                float* ares_out = ws_ares_out_f32.as<float>();
                WvInkCtrlPtrs3 dn3{{L.ctrl_down[0],   L.ctrl_down[1],   L.ctrl_down[2]}};
                WvInkCtrlPtrs3 fg3{{L.ctrl_forget[0], L.ctrl_forget[1], L.ctrl_forget[2]}};
                WvInkCtrlPtrs3 gn3{{L.ctrl_gain[0],   L.ctrl_gain[1],   L.ctrl_gain[2]}};
                run_loop_gate(ares, ares_out, dn3, fg3, gn3, 0, 3, 0);
                src_qkv = ares_out;
            }
            WvInkNormPtrs3 np3{{L.q_input_norm, L.k_input_norm, L.v_input_norm}, {qin, kin, vin}, {q8_q, q8_k, q8_v}, {sc_q, sc_k_act, sc_v_act}};
            wv_launch_pdl(pdl, dim3(M, 3), dim3(32, 6), 0, stream, wv_ink_f32_rmsnorm_bf16_fwht_q8_3way_kernel, src_qkv, np3, M);
        }

        __nv_bfloat16* q_raw = ws_q.as<__nv_bfloat16>();
        __nv_bfloat16* k_raw = ws_k.as<__nv_bfloat16>();
        __nv_bfloat16* v_raw = ws_v.as<__nv_bfloat16>();
        __nv_bfloat16* g_raw = ws_g.as<__nv_bfloat16>();
        __nv_bfloat16* r_raw = ws_r.as<__nv_bfloat16>();
        if (tc_m) {
            wv_launch_pdl(pdl, dim3(128, (M + 63) / 64), dim3(256), 0, stream,
                wv_ink_swz8_fused_qkvg_m64_n64_kernel,
                L.wq, L.wk, L.wv, L.wg, q8_q, sc_q, q8_k, sc_k_act, q8_v, sc_v_act,
                q_raw, k_raw, v_raw, g_raw, M);
            wv_launch_pdl(pdl, dim3(13, (M + 63) / 64), dim3(128), 0, stream,
                wv_ink_wr_sink_tc_m64_kernel, qin, L.wr, L.w_sink, r_raw, ws_sinks.as<float>(), M);
        } else {
            launch_wv_ink_swz8_fused_qkvg_wr_sink(
                L.wq, L.wk, L.wv, L.wg, L.wr, L.w_sink,
                q8_q, sc_q, q8_k, sc_k_act, q8_v, sc_v_act, qin,
                q_raw, k_raw, v_raw, g_raw, r_raw, ws_sinks.as<float>(), M, stream, pdl);
        }

        float* k_st   = sc_k.as<float>() + static_cast<size_t>(v_idx) * b_max * 3 * 1024;
        float* v_st   = sc_v.as<float>() + static_cast<size_t>(v_idx) * b_max * 3 * 1024;
        float* k_cand = commit_sconv ? nullptr : (cand_k.as<float>() + static_cast<size_t>(v_idx) * b_max * 8 * 1024);
        float* v_cand = commit_sconv ? nullptr : (cand_v.as<float>() + static_cast<size_t>(v_idx) * b_max * 8 * 1024);
        __nv_bfloat16* k_conv_out = ws_kin.as<__nv_bfloat16>();
        __nv_bfloat16* v_conv_out = ws_vin.as<__nv_bfloat16>();
        int kv_i     = visit_kv_idx[v_idx];
        int kv_cap   = is_loc ? INK2_SWA_RING : max_seq;
        int is_q8_gl = (!is_loc && global_kv_q8 != 0) ? 1 : 0;
        size_t kv_st = static_cast<size_t>(kv_i) * b_max * kv_cap * 1024;
        void* kc_ptr = is_q8_gl
            ? static_cast<void*>(kv_glb_k.as<int8_t>() + kv_st)
            : static_cast<void*>((is_loc ? kv_swa_k : kv_glb_k).as<__nv_bfloat16>() + kv_st);
        void* vc_ptr = is_q8_gl
            ? static_cast<void*>(kv_glb_v.as<int8_t>() + kv_st)
            : static_cast<void*>((is_loc ? kv_swa_v : kv_glb_v).as<__nv_bfloat16>() + kv_st);
        float* kc_sc_ptr = is_q8_gl ? (kv_glb_k_sc.as<float>() + static_cast<size_t>(kv_i) * b_max * kv_cap * 8) : nullptr;
        float* vc_sc_ptr = is_q8_gl ? (kv_glb_v_sc.as<float>() + static_cast<size_t>(kv_i) * b_max * kv_cap * 8) : nullptr;
        __nv_bfloat16* q_clip = ws_q_clip.as<__nv_bfloat16>();
        if (tc_s) {
            int s_blks32 = (S + 31) / 32;
            if (L.is_local) {
                wv_launch_pdl(pdl, dim3(8, s_blks32, B_act), dim3(256), 0, stream,
                    wv_ink_kv_static_sconv_prefill_kernel,
                    k_raw, v_raw, L.k_sconv_w, L.v_sconv_w, k_st, v_st, k_cand, v_cand,
                    d_smap, k_conv_out, v_conv_out, B_act, S);
                if (commit_sconv) {
                    wv_launch_pdl(pdl, dim3(8, B_act), dim3(256), 0, stream,
                        wv_ink_commit_kv_sconv_tail_kernel, k_raw, v_raw, k_st, v_st, d_smap, B_act, S);
                }
                wv_launch_pdl(pdl, dim3(M), dim3(32, 8), 0, stream,
                    wv_ink_prep_qkv_rel_swa_chunk_kernel,
                    q_raw, k_conv_out, v_conv_out, r_raw, L.q_clip_scale, L.r_clip_scale,
                    q_clip, ws_r_f32.as<float>(), k_raw, v_raw, M);
                wv_launch_pdl(pdl, dim3(8, s_blks32, B_act), dim3(192), 0, stream,
                    wv_ink_flash_attn_prefill_tc_kernel<false>,
                    q_clip, ws_r_f32.as<float>(), L.rel_proj_t,
                    kc_ptr, vc_ptr, static_cast<const float*>(nullptr), static_cast<const float*>(nullptr),
                    k_raw, v_raw, d_pos_arr, d_smap, ws_sinks.as<float>(), g_raw,
                    q8_q, sc_q, B_act, S, kv_cap, 1);
                wv_launch_pdl(pdl, dim3(std::min(S, INK2_SWA_WIN), B_act), dim3(128), 0, stream,
                    wv_ink_commit_swa_kv_tail_kernel,
                    k_raw, v_raw, static_cast<__nv_bfloat16*>(kc_ptr), static_cast<__nv_bfloat16*>(vc_ptr),
                    d_pos_arr, d_smap, B_act, S, kv_cap);
            } else {
                if (tc_m) {
                    wv_launch_pdl(pdl, dim3(2, (M + 63) / 64), dim3(128), 0, stream,
                        wv_ink_kv_wproj_tc_m64_kernel, kin, vin, L.k_wproj, L.v_wproj, ws_dw.as<float>(), M);
                } else {
                    wv_launch_pdl(pdl, dim3(8, M), dim3(32, 8), 0, stream,
                        wv_ink_kv_wproj_kernel, kin, vin, L.k_wproj, L.v_wproj, ws_dw.as<float>(), M);
                }
                wv_launch_pdl(pdl, dim3(8, s_blks32, B_act), dim3(256), 0, stream,
                    wv_ink_kv_dynamic_sconv_prefill_kernel,
                    k_raw, v_raw, ws_dw.as<float>(), L.k_static_w, L.v_static_w,
                    k_st, v_st, k_cand, v_cand, d_smap, k_conv_out, v_conv_out, B_act, S);
                if (commit_sconv) {
                    wv_launch_pdl(pdl, dim3(8, B_act), dim3(256), 0, stream,
                        wv_ink_commit_kv_sconv_tail_kernel, k_raw, v_raw, k_st, v_st, d_smap, B_act, S);
                }
                wv_launch_pdl(pdl, dim3(M), dim3(32, 8), 0, stream,
                    wv_ink_prep_qkv_rel_kvcache_kernel,
                    q_raw, k_conv_out, v_conv_out, r_raw, L.q_clip_scale, L.r_clip_scale,
                    q_clip, ws_r_f32.as<float>(), kc_ptr, vc_ptr, kc_sc_ptr, vc_sc_ptr,
                    d_pos_arr, d_smap, S, kv_cap, 0, is_q8_gl, M);
                if (is_q8_gl) {
                    wv_launch_pdl(pdl, dim3(8, s_blks32, B_act), dim3(192), 0, stream,
                        wv_ink_flash_attn_prefill_tc_kernel<true>,
                        q_clip, ws_r_f32.as<float>(), L.rel_proj_t,
                        kc_ptr, vc_ptr, kc_sc_ptr, vc_sc_ptr,
                        static_cast<const __nv_bfloat16*>(nullptr), static_cast<const __nv_bfloat16*>(nullptr),
                        d_pos_arr, d_smap, ws_sinks.as<float>(), g_raw,
                        q8_q, sc_q, B_act, S, kv_cap, 0);
                } else {
                    wv_launch_pdl(pdl, dim3(8, s_blks32, B_act), dim3(192), 0, stream,
                        wv_ink_flash_attn_prefill_tc_kernel<false>,
                        q_clip, ws_r_f32.as<float>(), L.rel_proj_t,
                        kc_ptr, vc_ptr, static_cast<const float*>(nullptr), static_cast<const float*>(nullptr),
                        static_cast<const __nv_bfloat16*>(nullptr), static_cast<const __nv_bfloat16*>(nullptr),
                        d_pos_arr, d_smap, ws_sinks.as<float>(), g_raw,
                        q8_q, sc_q, B_act, S, kv_cap, 0);
                }
            }
        } else {
            if (L.is_local && S <= 8) {
                wv_launch_pdl(pdl, dim3(8, B_act), dim3(32, 4), 0, stream,
                    wv_ink_prep_qkv_rel_kvcache_fused_swa_sm_kernel,
                    q_raw, k_raw, v_raw, r_raw, L.k_sconv_w, L.v_sconv_w, k_st, v_st, k_cand, v_cand,
                    L.q_clip_scale, L.r_clip_scale, q_clip, ws_r_f32.as<float>(),
                    static_cast<__nv_bfloat16*>(kc_ptr), static_cast<__nv_bfloat16*>(vc_ptr), d_pos_arr, d_smap, kv_cap, B_act, S, commit_sconv);
            } else {
                if (L.is_local) {
                    wv_launch_pdl(pdl, dim3(8, B_act), dim3(256), 0, stream,
                        wv_ink_kv_static_sconv_kernel,
                        k_raw, v_raw, L.k_sconv_w, L.v_sconv_w, k_st, v_st, k_cand, v_cand,
                        d_smap, k_conv_out, v_conv_out, B_act, S, commit_sconv);
                } else {
                    wv_launch_pdl(pdl, dim3(8, M), dim3(32, 8), 0, stream,
                        wv_ink_kv_wproj_kernel, kin, vin, L.k_wproj, L.v_wproj, ws_dw.as<float>(), M);
                    wv_launch_pdl(pdl, dim3(8, B_act), dim3(256), 0, stream,
                        wv_ink_kv_dynamic_sconv_kernel,
                        k_raw, v_raw, ws_dw.as<float>(), L.k_static_w, L.v_static_w,
                        k_st, v_st, k_cand, v_cand, d_smap, k_conv_out, v_conv_out, B_act, S, commit_sconv);
                }
                wv_launch_pdl(pdl, dim3(M), dim3(32, 8), 0, stream,
                    wv_ink_prep_qkv_rel_kvcache_kernel,
                    q_raw, k_conv_out, v_conv_out, r_raw, L.q_clip_scale, L.r_clip_scale,
                    q_clip, ws_r_f32.as<float>(), kc_ptr, vc_ptr, kc_sc_ptr, vc_sc_ptr,
                    d_pos_arr, d_smap, S, kv_cap, is_loc, is_q8_gl, M);
            }

            int n_splits = is_loc ? 4 : ((max_seq <= 8192) ? 8 : 16);
            if (!is_loc && is_q8_gl && B_act == 1 && S >= 2 && S <= 9) {
                n_splits = (max_seq <= 8192) ? 16 : 32;
                auto launch_s_shared = [&](auto nw_tag, auto q8_tag) {
                    constexpr int NW_V = decltype(nw_tag)::value;
                    constexpr bool Q8_V = decltype(q8_tag)::value;
                    wv_launch_pdl(pdl, dim3(8, n_splits), dim3(32, NW_V), 0, stream,
                        wv_ink_flash_gqa3_s_shared_split_kernel<NW_V, Q8_V>,
                        q_clip, ws_r_f32.as<float>(), L.rel_proj_t, kc_ptr, vc_ptr, kc_sc_ptr, vc_sc_ptr,
                        d_pos_arr, d_smap, ws_mpart.as<float>(), ws_lpart.as<float>(), ws_opart.as<float>(),
                        S, kv_cap, n_splits);
                };
                if (S <= 4) {
                    if (is_q8_gl) launch_s_shared(std::integral_constant<int, 4>{}, std::true_type{});
                    else          launch_s_shared(std::integral_constant<int, 4>{}, std::false_type{});
                } else if (S <= 8) {
                    if (is_q8_gl) launch_s_shared(std::integral_constant<int, 8>{}, std::true_type{});
                    else          launch_s_shared(std::integral_constant<int, 8>{}, std::false_type{});
                } else {
                    if (is_q8_gl) launch_s_shared(std::integral_constant<int, 9>{}, std::true_type{});
                    else          launch_s_shared(std::integral_constant<int, 9>{}, std::false_type{});
                }
            } else {
                wv_launch_pdl(pdl, dim3(M * 8, n_splits), dim3(32, 4), 0, stream,
                    wv_ink_flash_gqa3_rel_sink_split_kernel,
                    q_clip, ws_r_f32.as<float>(), L.rel_proj_t, kc_ptr, vc_ptr, kc_sc_ptr, vc_sc_ptr,
                    d_pos_arr, d_smap, ws_mpart.as<float>(), ws_lpart.as<float>(), ws_opart.as<float>(),
                    S, kv_cap, is_loc, is_q8_gl, n_splits);
            }
            wv_launch_pdl(pdl, dim3((M * 24 + 3) / 4), dim3(32, 4), 0, stream,
                wv_ink_flash_gqa_combine_gate_fwht_q8_kernel,
                ws_mpart.as<float>(), ws_lpart.as<float>(), ws_opart.as<float>(),
                ws_sinks.as<float>(), g_raw, q8_q, sc_q, M * 24, n_splits);
        }

        if (use_tc_prefill && M > 48) {
            wv_launch_pdl(pdl, dim3((INK2_HIDDEN + 63) / 64, (M + 63) / 64), dim3(256), 0, stream,
                wv_ink_swz8_mma_linear_m64_n64_kernel<true, false>,
                L.wo.pk, L.wo.sc, static_cast<const float*>(nullptr),
                q8_q, sc_q, static_cast<void*>(qin), INK2_ATTN_DIM, INK2_HIDDEN, M);
        } else {
            launch_wv_ink_swz8_linear<true>(L.wo, nullptr, q8_q, sc_q, qin, INK2_ATTN_DIM, INK2_HIDDEN, M, stream, pdl);
        }
        float* a_st   = sc_a.as<float>() + static_cast<size_t>(v_idx) * b_max * 3 * INK2_HIDDEN;
        float* a_cand = commit_sconv ? nullptr : (cand_a.as<float>() + static_cast<size_t>(v_idx) * b_max * 8 * INK2_HIDDEN);
        float* h_mid  = ws_h_mid_f32.as<float>();
        if (tc_s) {
            wv_launch_pdl(pdl, dim3(9, (S + 31) / 32, B_act), dim3(256), 0, stream,
                wv_ink_attn_sconv_res_update_prefill_kernel,
                qin, cur_h, L.attn_sconv_w, a_st, a_cand, d_smap, cur_blocks, h_mid,
                B_act, S, n_blk_io, 6, is_bound);
            if (commit_sconv) {
                wv_launch_pdl(pdl, dim3(9, B_act), dim3(256), 0, stream,
                    wv_ink_commit_h_sconv_tail_kernel, qin, a_st, d_smap, B_act, S);
            }
        } else {
            unsigned sc_gx  = (B_act == 1) ? 36u : 9u;
            unsigned sc_thr = (B_act == 1) ? 64u : 256u;
            wv_launch_pdl(pdl, dim3(sc_gx, B_act), dim3(sc_thr), 0, stream,
                wv_ink_attn_sconv_res_update_kernel,
                qin, cur_h, L.attn_sconv_w, a_st, a_cand, d_smap, cur_blocks, h_mid,
                B_act, S, n_blk_io, 6, is_bound, commit_sconv);
        }
        if (is_bound) n_blk_io += 1;

        if (messages_f32 == nullptr) {
            wv_launch_pdl(pdl, dim3(M), dim3(32, 6), 0, stream,
                wv_ink_block_attn_res_fused_norm_q8_kernel,
                cur_blocks, h_mid, glob.mlp_queries + static_cast<size_t>(li) * INK2_HIDDEN, L.mlp_norm,
                qin, q8_q, sc_q, n_blk_io, 6, 0, M);
        } else {
            wv_launch_pdl(pdl, dim3(M), dim3(32, 6), 0, stream,
                wv_ink_block_attn_res_kernel,
                cur_blocks, h_mid, glob.mlp_queries + static_cast<size_t>(li) * INK2_HIDDEN, ares,
                n_blk_io, 6, 1, 0, M);
            float* mlp_out = ws_ares_out_f32.as<float>();
            WvInkCtrlPtrs3 dn1{{L.ctrl_down[3],   nullptr, nullptr}};
            WvInkCtrlPtrs3 fg1{{L.ctrl_forget[3], nullptr, nullptr}};
            WvInkCtrlPtrs3 gn1{{L.ctrl_gain[3],   nullptr, nullptr}};
            run_loop_gate(ares, mlp_out, dn1, fg1, gn1, 3, 1, 0);
            wv_launch_pdl(pdl, dim3(M), dim3(32, 6), 0, stream,
                wv_ink_f32_rmsnorm_bf16_fwht_q8_kernel, mlp_out, L.mlp_norm, qin, q8_q, sc_q, M);
        }

        __nv_bfloat16* moe_inter = ws_moe_inter.as<__nv_bfloat16>();
        __nv_bfloat16* moe_slot  = ws_moe_slot.as<__nv_bfloat16>();
        int tot_b128 = M * cfg.top_k * 9;
        if (M == 1) {
            launch_wv_ink_router_topk(
                qin, L.router_w, L.router_b, ws_router_sc.as<float>(),
                ws_top_eid.as<int32_t>(), ws_top_w.as<float>(), cfg.n_experts, cfg.top_k, 1, stream, pdl, num_sms);
            if (!dma.all_vram && dma.has_cold_dma) {
                if (wv_ink_resolve_and_prefetch_visit_dma(dma, v_idx, li, next_li, ws_top_eid.as<int32_t>(), cfg.top_k, 1, stage_free_ev, stream) != 0) return -1;
            }
            wv_launch_pdl(pdl, dim3(72, cfg.top_k), dim3(128), 0, stream,
                wv_ink_moe_gate_up_swiglu_dp4a_m1_kernel,
                q8_q, sc_q, glob.cb4096_gmem, glob.vq8_uv_gmem, L.hot_gu_pk3, L.hot_gu_sc,
                ws_top_eid.as<int32_t>(), L.tmap,
                dma.all_vram ? dma.d_all_vram_slabs[li] : nullptr,
                dma.all_vram ? nullptr : dma.d_resolved_ptrs, moe_inter);
            wv_launch_pdl(pdl, dim3((tot_b128 + 3) / 4), dim3(32, 4), 0, stream,
                wv_ink_bf16_fwht128_q8_kernel, moe_inter, ws_moe_q8.as<int8_t>(), ws_moe_sc.as<float>(), tot_b128);
            wv_launch_pdl(pdl, dim3(72, cfg.top_k), dim3(128), 0, stream,
                wv_ink_moe_down_dp4a_m1_kernel,
                ws_moe_q8.as<int8_t>(), ws_moe_sc.as<float>(), glob.cb4096_gmem, glob.vq8_uv_gmem,
                L.hot_dn_pk3, L.hot_dn_sc, ws_top_eid.as<int32_t>(), L.tmap,
                dma.all_vram ? dma.d_all_vram_slabs[li] : nullptr,
                dma.all_vram ? nullptr : dma.d_resolved_ptrs, moe_slot);
        } else if (tc_m) {
            wv_launch_pdl(pdl, dim3((cfg.n_experts + 31) / 32, (M + 63) / 64), dim3(128), 0, stream,
                wv_ink_router_gemm_tc_m64_kernel, qin, L.router_w, ws_router_sc.as<float>(), cfg.n_experts, M);
            wv_launch_pdl(pdl, dim3(M), dim3(32), 0, stream,
                wv_ink_router_select_topk_kernel,
                ws_router_sc.as<float>(), L.router_b, ws_top_eid.as<int32_t>(), ws_top_w.as<float>(),
                cfg.n_experts, cfg.top_k, M);
            if (!dma.all_vram && dma.has_cold_dma) {
                if (wv_ink_resolve_and_prefetch_visit_dma(dma, v_idx, li, next_li, ws_top_eid.as<int32_t>(), cfg.top_k, M, stage_free_ev, stream) != 0) return -1;
            }
            int tot_p  = M * cfg.top_k;
            int n_blks = (tot_p + 255) / 256;
            int32_t* blk_cnt = ws_radix_hist.as<int32_t>();
            int32_t* blk_off = blk_cnt + static_cast<size_t>(n_blks) * 128;
            wv_launch_pdl(pdl, dim3(n_blks), dim3(256), 0, stream,
                wv_ink_radix_hist_stage1_kernel, ws_top_eid.as<int32_t>(), blk_cnt, tot_p, cfg.n_experts);
            wv_launch_pdl(pdl, dim3(1), dim3(128), 0, stream,
                wv_ink_radix_prefix_tiles64_kernel,
                blk_cnt, blk_off, L.tmap,
                dma.all_vram ? dma.d_all_vram_slabs[li] : nullptr,
                dma.all_vram ? nullptr : dma.d_resolved_ptrs,
                ws_num_tiles.as<int32_t>(), ws_t_hot.as<int32_t>(), ws_t_slab.as<uint64_t>(),
                ws_t_beg.as<int32_t>(), ws_t_len.as<int32_t>(), n_blks, cfg.n_experts);
            wv_launch_pdl(pdl, dim3(n_blks), dim3(256), 0, stream,
                wv_ink_radix_scatter_stage3_kernel,
                ws_top_eid.as<int32_t>(), blk_off, ws_p_tok.as<int32_t>(), ws_p_slot.as<int32_t>(),
                tot_p, cfg.top_k, cfg.n_experts);
            int max_tiles64 = (tot_p >> 6) + cfg.n_experts;
            wv_launch_pdl(pdl, dim3(36, max_tiles64), dim3(256), 0, stream,
                wv_ink_moe_gate_up_swiglu_m64_n64_kernel,
                q8_q, sc_q, glob.cb4096_gmem, glob.vq8_uv_gmem, L.hot_gu_pk3, L.hot_gu_sc,
                ws_num_tiles.as<int32_t>(), ws_t_hot.as<int32_t>(), ws_t_slab.as<uint64_t>(),
                ws_t_beg.as<int32_t>(), ws_t_len.as<int32_t>(), ws_p_tok.as<int32_t>(), ws_p_slot.as<int32_t>(), moe_inter);
            wv_launch_pdl(pdl, dim3((tot_b128 + 3) / 4), dim3(32, 4), 0, stream,
                wv_ink_bf16_fwht128_q8_kernel, moe_inter, ws_moe_q8.as<int8_t>(), ws_moe_sc.as<float>(), tot_b128);
            wv_launch_pdl(pdl, dim3(36, max_tiles64), dim3(256), 0, stream,
                wv_ink_moe_down_m64_n64_kernel,
                ws_moe_q8.as<int8_t>(), ws_moe_sc.as<float>(), glob.cb4096_gmem, glob.vq8_uv_gmem,
                L.hot_dn_pk3, L.hot_dn_sc,
                ws_num_tiles.as<int32_t>(), ws_t_hot.as<int32_t>(), ws_t_slab.as<uint64_t>(),
                ws_t_beg.as<int32_t>(), ws_t_len.as<int32_t>(), ws_p_slot.as<int32_t>(), moe_slot);
        } else {
            if (dma.all_vram && M <= 32) {
                wv_launch_pdl(pdl, dim3((cfg.n_experts + 1) / 2, (M + 7) / 8), dim3(32, 2), 0, stream,
                    wv_ink_router_gemv_tiled_kernel, qin, L.router_w, ws_router_sc.as<float>(), cfg.n_experts, M);
                wv_launch_pdl(pdl, dim3(1), dim3(1024), 0, stream,
                    wv_ink_router_select_and_group_allvram_kernel,
                    ws_router_sc.as<float>(), L.router_b, L.tmap, dma.d_all_vram_slabs[li],
                    ws_top_eid.as<int32_t>(), ws_top_w.as<float>(),
                    ws_num_tiles.as<int32_t>(), ws_t_hot.as<int32_t>(), ws_t_slab.as<uint64_t>(),
                    ws_t_beg.as<int32_t>(), ws_t_len.as<int32_t>(), ws_p_tok.as<int32_t>(), ws_p_slot.as<int32_t>(),
                    cfg.n_experts, cfg.top_k, M);
            } else {
                launch_wv_ink_router_topk(
                    qin, L.router_w, L.router_b, ws_router_sc.as<float>(),
                    ws_top_eid.as<int32_t>(), ws_top_w.as<float>(), cfg.n_experts, cfg.top_k, M, stream, pdl, num_sms);
                if (!dma.all_vram && dma.has_cold_dma) {
                    if (wv_ink_resolve_and_prefetch_visit_dma(dma, v_idx, li, next_li, ws_top_eid.as<int32_t>(), cfg.top_k, M, stage_free_ev, stream) != 0) return -1;
                }
                wv_launch_pdl(pdl, dim3(1), dim3(128), 0, stream,
                    wv_ink_build_expert_groups_kernel,
                    ws_top_eid.as<int32_t>(), L.tmap,
                    dma.all_vram ? dma.d_all_vram_slabs[li] : nullptr,
                    dma.all_vram ? nullptr : dma.d_resolved_ptrs,
                    ws_num_tiles.as<int32_t>(), ws_t_hot.as<int32_t>(), ws_t_slab.as<uint64_t>(),
                    ws_t_beg.as<int32_t>(), ws_t_len.as<int32_t>(), ws_p_tok.as<int32_t>(), ws_p_slot.as<int32_t>(),
                    cfg.n_experts, cfg.top_k, M);
            }
            int gy = std::max(2, (6 * num_sms) / 72);
            if (M <= 8) {
                wv_launch_pdl(pdl, dim3(72, gy), dim3(128), 0, stream,
                    wv_ink_moe_gate_up_swiglu_mma_m8_kernel,
                    q8_q, sc_q, glob.cb4096_gmem, glob.vq8_uv_gmem, L.hot_gu_pk3, L.hot_gu_sc,
                    ws_num_tiles.as<int32_t>(), ws_t_hot.as<int32_t>(), ws_t_slab.as<uint64_t>(),
                    ws_t_beg.as<int32_t>(), ws_t_len.as<int32_t>(), ws_p_tok.as<int32_t>(), ws_p_slot.as<int32_t>(), moe_inter);
                wv_launch_pdl(pdl, dim3((tot_b128 + 3) / 4), dim3(32, 4), 0, stream,
                    wv_ink_bf16_fwht128_q8_kernel, moe_inter, ws_moe_q8.as<int8_t>(), ws_moe_sc.as<float>(), tot_b128);
                wv_launch_pdl(pdl, dim3(72, gy), dim3(128), 0, stream,
                    wv_ink_moe_down_mma_m8_kernel,
                    ws_moe_q8.as<int8_t>(), ws_moe_sc.as<float>(), glob.cb4096_gmem, glob.vq8_uv_gmem,
                    L.hot_dn_pk3, L.hot_dn_sc,
                    ws_num_tiles.as<int32_t>(), ws_t_hot.as<int32_t>(), ws_t_slab.as<uint64_t>(),
                    ws_t_beg.as<int32_t>(), ws_t_len.as<int32_t>(), ws_p_slot.as<int32_t>(), moe_slot);
            } else {
                wv_launch_pdl(pdl, dim3(72, gy), dim3(128), 0, stream,
                    wv_ink_moe_gate_up_swiglu_kernel,
                    q8_q, sc_q, glob.cb4096_gmem, glob.vq8_uv_gmem, L.hot_gu_pk3, L.hot_gu_sc,
                    ws_num_tiles.as<int32_t>(), ws_t_hot.as<int32_t>(), ws_t_slab.as<uint64_t>(),
                    ws_t_beg.as<int32_t>(), ws_t_len.as<int32_t>(), ws_p_tok.as<int32_t>(), ws_p_slot.as<int32_t>(), moe_inter);
                wv_launch_pdl(pdl, dim3((tot_b128 + 3) / 4), dim3(32, 4), 0, stream,
                    wv_ink_bf16_fwht128_q8_kernel, moe_inter, ws_moe_q8.as<int8_t>(), ws_moe_sc.as<float>(), tot_b128);
                wv_launch_pdl(pdl, dim3(72, gy), dim3(128), 0, stream,
                    wv_ink_moe_down_kernel,
                    ws_moe_q8.as<int8_t>(), ws_moe_sc.as<float>(), glob.cb4096_gmem, glob.vq8_uv_gmem,
                    L.hot_dn_pk3, L.hot_dn_sc,
                    ws_num_tiles.as<int32_t>(), ws_t_hot.as<int32_t>(), ws_t_slab.as<uint64_t>(),
                    ws_t_beg.as<int32_t>(), ws_t_len.as<int32_t>(), ws_p_slot.as<int32_t>(), moe_slot);
            }
        }
        if (!dma.all_vram && dma.has_cold_dma) {
            WV_CUDA_CHECK(cudaEventRecord(stage_free_ev[v_idx & 1], stream));
        }

        float* m_st   = sc_m.as<float>() + static_cast<size_t>(v_idx) * b_max * 3 * INK2_HIDDEN;
        float* m_cand = commit_sconv ? nullptr : (cand_m.as<float>() + static_cast<size_t>(v_idx) * b_max * 8 * INK2_HIDDEN);
        if (tc_s) {
            __nv_bfloat16* mlp_out_bf = ws_mlp_out.as<__nv_bfloat16>();
            wv_launch_pdl(pdl, dim3(9, (M + 3) / 4), dim3(256), 0, stream,
                wv_ink_moe_combine_prefill_kernel, moe_slot, ws_top_w.as<float>(), mlp_out_bf, cfg.top_k, M);
            wv_launch_pdl(pdl, dim3(9, (S + 31) / 32, B_act), dim3(256), 0, stream,
                wv_ink_mlp_sconv_res_add_prefill_kernel,
                mlp_out_bf, L.mlp_sconv_w, m_st, m_cand, d_smap, h_mid, cur_h, B_act, S);
            if (commit_sconv) {
                wv_launch_pdl(pdl, dim3(9, B_act), dim3(256), 0, stream,
                    wv_ink_commit_h_sconv_tail_kernel, mlp_out_bf, m_st, d_smap, B_act, S);
            }
        } else {
            unsigned mc_gx  = (B_act == 1) ? 36u : 9u;
            unsigned mc_thr = (B_act == 1) ? 64u : 256u;
            wv_launch_pdl(pdl, dim3(mc_gx, B_act), dim3(mc_thr), 0, stream,
                wv_ink_moe_combine_sconv_res_add_kernel,
                moe_slot, ws_top_w.as<float>(), L.mlp_sconv_w, m_st, m_cand, d_smap, h_mid, cur_h,
                cfg.top_k, B_act, S, commit_sconv);
        }
        return 0;
    }

    void commit_mtp_taps_gpu(
        const __nv_bfloat16* taps_ptr, int B_act, int S_grid, const int32_t* d_n_acc, bool pdl, cudaStream_t stream) {
        if (!has_mtp || S_grid <= 0) return;
        int M      = B_act * S_grid;
        int8_t* q8 = ws_q8_a.as<int8_t>();
        float*  sc = ws_sc_a.as<float>();
        __nv_bfloat16* kc_raw = ws_k.as<__nv_bfloat16>();
        __nv_bfloat16* vc_raw = ws_v.as<__nv_bfloat16>();
        const int32_t* d_mp0  = d_mtp_pos0_t.as<int32_t>();

        for (int i = 0; i < cfg.n_mtp_layers; ++i) {
            const WvInkMtpLayerDesc& ML = mtp_layers[i];
            wv_launch_pdl(pdl, dim3(M), dim3(32, 6), 0, stream,
                wv_ink_mtp_mix_dflare_norm_kernel, taps_ptr, mtp_fuse_alpha, i, ML.dflare_norm, nullptr, q8, sc, M);
            if (M <= 8) {
                wv_launch_pdl(pdl, dim3(128), dim3(64), 0, stream,
                    wv_ink_swz8_fused_mtp_kv_ctx_kernel, ML.k_ctx, ML.v_ctx, q8, sc, kc_raw, vc_raw, M);
            } else if (use_tc_prefill && M > 48) {
                wv_launch_pdl(pdl, dim3((INK2_KV_DIM + 63) / 64, (M + 63) / 64), dim3(256), 0, stream,
                    wv_ink_swz8_mma_linear_m64_n64_kernel<true, false>,
                    ML.k_ctx.pk, ML.k_ctx.sc, static_cast<const float*>(nullptr),
                    q8, sc, static_cast<void*>(kc_raw), INK2_HIDDEN, INK2_KV_DIM, M);
                wv_launch_pdl(pdl, dim3((INK2_KV_DIM + 63) / 64, (M + 63) / 64), dim3(256), 0, stream,
                    wv_ink_swz8_mma_linear_m64_n64_kernel<true, false>,
                    ML.v_ctx.pk, ML.v_ctx.sc, static_cast<const float*>(nullptr),
                    q8, sc, static_cast<void*>(vc_raw), INK2_HIDDEN, INK2_KV_DIM, M);
            } else {
                launch_wv_ink_swz8_linear<true>(ML.k_ctx, nullptr, q8, sc, kc_raw, INK2_HIDDEN, INK2_KV_DIM, M, stream, pdl);
                launch_wv_ink_swz8_linear<true>(ML.v_ctx, nullptr, q8, sc, vc_raw, INK2_HIDDEN, INK2_KV_DIM, M, stream, pdl);
            }

            int is_loc   = ML.is_local ? 1 : 0;
            int is_q8_gl = (!is_loc && global_kv_q8 != 0) ? 1 : 0;
            int kv_cap   = is_loc ? INK2_SWA_RING : max_seq;
            int kv_i     = mtp_kv_idx[i];
            size_t off   = static_cast<size_t>(kv_i) * b_max * kv_cap * 1024;
            void* k_dst  = is_q8_gl
                ? static_cast<void*>(mtp_glb_k.as<int8_t>() + off)
                : static_cast<void*>((is_loc ? mtp_swa_k : mtp_glb_k).as<__nv_bfloat16>() + off);
            void* v_dst  = is_q8_gl
                ? static_cast<void*>(mtp_glb_v.as<int8_t>() + off)
                : static_cast<void*>((is_loc ? mtp_swa_v : mtp_glb_v).as<__nv_bfloat16>() + off);
            float* k_dst_sc = is_q8_gl ? (mtp_glb_k_sc.as<float>() + static_cast<size_t>(kv_i) * b_max * kv_cap * 8) : nullptr;
            float* v_dst_sc = is_q8_gl ? (mtp_glb_v_sc.as<float>() + static_cast<size_t>(kv_i) * b_max * kv_cap * 8) : nullptr;
            wv_launch_pdl(pdl, dim3(M), dim3(32, 8), 0, stream,
                wv_ink_mtp_commit_ctx_kv_kernel,
                kc_raw, vc_raw, ML.k_norm_ctx, k_dst, v_dst, k_dst_sc, v_dst_sc,
                d_mp0, d_n_acc, S_grid, kv_cap, is_loc, is_q8_gl, M);
        }

        if (d_n_acc != nullptr) {
            wv_launch_pdl(pdl, dim3(1), dim3(32, 6), 0, stream,
                wv_ink_mtp_mix_dflare_norm_last_acc_kernel,
                taps_ptr, d_n_acc, mtp_fuse_alpha, mtp_mh_dflare, mtp_f_rnn.as<__nv_bfloat16>());
        } else {
            for (int b = 0; b < B_act; ++b) {
                const __nv_bfloat16* last_tap = taps_ptr + static_cast<size_t>(b * S_grid + S_grid - 1) * 4 * INK2_HIDDEN;
                wv_launch_pdl(pdl, dim3(1), dim3(32, 6), 0, stream,
                    wv_ink_mtp_mix_dflare_norm_kernel,
                    last_tap, mtp_fuse_alpha, 3, mtp_mh_dflare,
                    mtp_f_rnn.as<__nv_bfloat16>() + static_cast<size_t>(b) * INK2_HIDDEN, nullptr, nullptr, 1);
            }
        }
    }

    void draft_k_block_gpu(int K_draft, bool do_sample, bool pdl, cudaStream_t stream) {
        __nv_bfloat16* x_blk = ws_mtp_x.as<__nv_bfloat16>();
        __nv_bfloat16* r_blk = ws_mtp_res.as<__nv_bfloat16>();
        int8_t* q8           = ws_q8_a.as<int8_t>();
        float*  sc           = ws_sc_a.as<float>();
        const int32_t* d_mp0 = d_mtp_pos0_t.as<int32_t>();

        wv_launch_pdl(pdl, dim3(1), dim3(32, 6), 0, stream,
            wv_ink_mtp_embed_init_block8_kernel, ws_draft8.as<int32_t>(), glob.emb_wire, glob.embed_norm, mtp_mask_embed, x_blk, cfg.vocab_full);

        __nv_bfloat16* q_buf  = ws_q.as<__nv_bfloat16>();
        __nv_bfloat16* bk_buf = ws_mtp_bk.as<__nv_bfloat16>();
        __nv_bfloat16* bv_buf = ws_mtp_bv.as<__nv_bfloat16>();
        __nv_bfloat16* g_mlp  = ws_moe_slot.as<__nv_bfloat16>();
        __nv_bfloat16* u_mlp  = ws_mtp_up.as<__nv_bfloat16>();

        for (int i = 0; i < cfg.n_mtp_layers; ++i) {
            const WvInkMtpLayerDesc& ML = mtp_layers[i];
            wv_launch_pdl(pdl, dim3(8), dim3(32, 6), 0, stream,
                wv_ink_mtp_res_accum_rmsnorm_fwht_q8_kernel, x_blk, r_blk, ML.in_norm, nullptr, q8, sc, (i == 0) ? 1 : 0, 8);

            wv_launch_pdl(pdl, dim3(160), dim3(128), 0, stream,
                wv_ink_swz8_fused_mtp_qkv_m8_kernel, ML.q, ML.k, ML.v, q8, sc, q_buf, bk_buf, bv_buf);
            wv_launch_pdl(pdl, dim3(8), dim3(32, 8), 0, stream,
                wv_ink_mtp_qkv_norm_rope_kernel, q_buf, bk_buf, ML.q_norm, ML.k_norm, d_mp0, 8);

            int is_loc   = ML.is_local ? 1 : 0;
            int is_q8_gl = (!is_loc && global_kv_q8 != 0) ? 1 : 0;
            int kv_cap   = is_loc ? INK2_SWA_RING : max_seq;
            int kv_i     = mtp_kv_idx[i];
            size_t off   = static_cast<size_t>(kv_i) * b_max * kv_cap * 1024;
            if (is_loc) {
                const __nv_bfloat16* k_ctx = mtp_swa_k.as<__nv_bfloat16>() + off;
                const __nv_bfloat16* v_ctx = mtp_swa_v.as<__nv_bfloat16>() + off;
                wv_launch_pdl(pdl, dim3(8 * 24), dim3(32, 4), 0, stream,
                    wv_ink_mtp_bidir_gqa_fwht_q8_kernel, q_buf, k_ctx, v_ctx, bk_buf, bv_buf, q8, sc, d_mp0, kv_cap, is_loc);
            } else {
                int n_splits = (max_seq <= 8192) ? 16 : 32;
                const void* k_ctx = is_q8_gl
                    ? static_cast<const void*>(mtp_glb_k.as<int8_t>() + off)
                    : static_cast<const void*>(mtp_glb_k.as<__nv_bfloat16>() + off);
                const void* v_ctx = is_q8_gl
                    ? static_cast<const void*>(mtp_glb_v.as<int8_t>() + off)
                    : static_cast<const void*>(mtp_glb_v.as<__nv_bfloat16>() + off);
                const float* k_ctx_sc = is_q8_gl ? (mtp_glb_k_sc.as<float>() + static_cast<size_t>(kv_i) * b_max * kv_cap * 8) : nullptr;
                const float* v_ctx_sc = is_q8_gl ? (mtp_glb_v_sc.as<float>() + static_cast<size_t>(kv_i) * b_max * kv_cap * 8) : nullptr;
                if (is_q8_gl) {
                    wv_launch_pdl(pdl, dim3(8, n_splits), dim3(32, 8), 0, stream,
                        wv_ink_mtp_bidir_gqa3_s_shared_split_kernel<true>,
                        q_buf, k_ctx, v_ctx, k_ctx_sc, v_ctx_sc, bk_buf, bv_buf, d_mp0,
                        ws_mpart.as<float>(), ws_lpart.as<float>(), ws_opart.as<float>(), kv_cap, n_splits);
                } else {
                    wv_launch_pdl(pdl, dim3(8, n_splits), dim3(32, 8), 0, stream,
                        wv_ink_mtp_bidir_gqa3_s_shared_split_kernel<false>,
                        q_buf, k_ctx, v_ctx, k_ctx_sc, v_ctx_sc, bk_buf, bv_buf, d_mp0,
                        ws_mpart.as<float>(), ws_lpart.as<float>(), ws_opart.as<float>(), kv_cap, n_splits);
                }
                wv_launch_pdl(pdl, dim3(48), dim3(32, 4), 0, stream,
                    wv_ink_mtp_flash_gqa_combine_fwht_q8_kernel,
                    ws_mpart.as<float>(), ws_lpart.as<float>(), ws_opart.as<float>(), q8, sc, 8 * 24, n_splits);
            }
            launch_wv_ink_swz8_linear<true>(ML.o, nullptr, q8, sc, x_blk, INK2_ATTN_DIM, INK2_HIDDEN, 8, stream, pdl);

            wv_launch_pdl(pdl, dim3(8), dim3(32, 6), 0, stream,
                wv_ink_mtp_res_accum_rmsnorm_fwht_q8_kernel, x_blk, r_blk, ML.post_norm, nullptr, q8, sc, 0, 8);
            wv_launch_pdl(pdl, dim3(576), dim3(128), 0, stream,
                wv_ink_swz8_fused_mtp_gu_m8_kernel, ML.gate, ML.up, q8, sc, g_mlp, u_mlp);
            int tot_b128 = 8 * (INK2_MTP_INTER / 128);
            wv_launch_pdl(pdl, dim3((tot_b128 + 3) / 4), dim3(32, 4), 0, stream,
                wv_ink_mtp_silu_mul_fwht_q8_kernel, g_mlp, u_mlp, q8, sc, tot_b128);
            launch_wv_ink_swz8_linear<true>(ML.down, nullptr, q8, sc, x_blk, INK2_MTP_INTER, INK2_HIDDEN, 8, stream, pdl);
        }

        wv_launch_pdl(pdl, dim3(8), dim3(32, 6), 0, stream,
            wv_ink_mtp_res_accum_rmsnorm_fwht_q8_kernel, x_blk, r_blk, mtp_norm_w, x_blk, q8, sc, 0, 8);

        float* base_logits = ws_logits.as<float>();
        launch_wv_ink_swz8_linear<false, false>(
            glob.head, nullptr, q8 + INK2_HIDDEN, sc + 72, base_logits, INK2_HIDDEN, cfg.vocab_pad, K_draft, stream, pdl, true);

        wv_launch_pdl(pdl, dim3(96, K_draft), dim3(32, 8), 0, stream,
            wv_ink_mtp_markov_ctx_kernel,
            x_blk, mtp_f_rnn.as<__nv_bfloat16>(), mtp_rnn_w, mtp_rnn_b, ws_mtp_ctx.as<float>());

        int8_t* lat_q8 = ws_q8_b.as<int8_t>();
        float*  lat_sc = ws_sc_b.as<float>();
        float*  bias_l = ws_mtp_bias.as<float>();
        int32_t* d8    = ws_draft8.as<int32_t>();
        for (int t = 0; t < K_draft; ++t) {
            wv_launch_pdl(pdl, dim3(96), dim3(32, 8), 0, stream,
                wv_ink_mtp_markov_rnn_gemv_kernel,
                d8 + t, mtp_w1_i8, mtp_w1_sc, mtp_rnn_w,
                ws_mtp_ctx.as<float>() + t * 768, ws_mtp_st.as<__nv_bfloat16>(), ws_mtp_raw768.as<float>(), t, cfg.vocab_full);
            wv_launch_pdl(pdl, dim3(1), dim3(256), 0, stream,
                wv_ink_mtp_markov_rnn_act_q8_kernel,
                ws_mtp_raw768.as<float>(), ws_mtp_st.as<__nv_bfloat16>(), lat_q8, lat_sc, t);
            const float* logits_t = base_logits + static_cast<size_t>(t) * cfg.vocab_pad;
            if (!do_sample) {
                wv_launch_pdl(pdl, dim3(256), dim3(128), 0, stream,
                    wv_ink_mtp_fold_combine_argmax_part_kernel,
                    mtp_fold, mtp_fold_b, lat_q8, lat_sc, logits_t,
                    ws_argmax_v.as<float>(), ws_argmax_i.as<int32_t>(), cfg.vocab_unpad);
                wv_launch_pdl(pdl, dim3(1), dim3(256), 0, stream,
                    wv_ink_argmax_vocab_final_kernel,
                    ws_argmax_v.as<float>(), ws_argmax_i.as<int32_t>(), d8 + 1 + t);
            } else {
                launch_wv_ink_swz8_linear<false, false>(
                    mtp_fold, mtp_fold_b, lat_q8, lat_sc, bias_l, INK2_MTP_RANK, cfg.vocab_pad, 1, stream, pdl, true);
                wv_launch_pdl(pdl, dim3(1), dim3(512), 0, stream,
                    wv_ink_mtp_combine_softcap_bias_sample_topp_kernel,
                    logits_t, bias_l, ws_mtp_q_probs.as<float>() + static_cast<size_t>(t) * cfg.vocab_pad,
                    d8 + 1 + t, sample_cfg_ptr(), t, cfg.vocab_pad, cfg.vocab_unpad);
            }
        }
    }

    void maybe_stash_tap(int v_idx, __nv_bfloat16* cur_h, __nv_bfloat16* taps, int M, int c_grd, bool pdl, cudaStream_t stream) {
        for (int t = 0; t < 3; ++t) {
            if (cfg.tap_visits[t] == v_idx) {
                wv_launch_pdl(pdl, dim3(c_grd), dim3(256), 0, stream, wv_ink_stash_tap_kernel, cur_h, taps, t, M);
            }
        }
    }

    int32_t forward_chunk(
        const int32_t* d_ids, int B_act, int S,
        const int32_t* d_pos_arr, const int32_t* d_smap,
        bool capture_taps, int commit_sconv, bool pdl, cudaStream_t stream,
        const int32_t* d_vis_idx = nullptr, const __nv_bfloat16* d_vis_emb = nullptr, int N_vis = 0) {
        int M = B_act * S;
        __nv_bfloat16* cur_h   = ws_h.as<__nv_bfloat16>();
        __nv_bfloat16* pre_h   = ws_pre_h.as<__nv_bfloat16>();
        __nv_bfloat16* prev_h  = ws_prev_h.as<__nv_bfloat16>();
        __nv_bfloat16* h_fin   = ws_h_fin.as<__nv_bfloat16>();
        __nv_bfloat16* blks    = ws_blocks.as<__nv_bfloat16>();
        __nv_bfloat16* pre_blk = ws_pre_blk.as<__nv_bfloat16>();
        __nv_bfloat16* prv_blk = ws_prev_blk.as<__nv_bfloat16>();
        __nv_bfloat16* taps    = ws_taps.as<__nv_bfloat16>();

        wv_launch_pdl(pdl, dim3(M), dim3(32, 6), 0, stream,
            wv_ink_embed_gather_norm_kernel, d_ids, glob.emb_wire, glob.embed_norm, cur_h, M, cfg.vocab_full);
        if (d_vis_idx != nullptr && d_vis_emb != nullptr && N_vis > 0) {
            wv_launch_pdl(pdl, dim3(M), dim3(256), 0, stream,
                wv_ink_scatter_vision_embeds_kernel, d_vis_idx, d_vis_emb, cur_h, M, N_vis);
        }

        int n_blk = 0;
        int c_grd = (M * INK2_HIDDEN + 255) / 256;

        for (int v_idx = 0; v_idx < cfg.loop_beg; ++v_idx) {
            if (capture_taps) maybe_stash_tap(v_idx, cur_h, taps, M, c_grd, pdl, stream);
            if (run_visit(v_idx, B_act, S, d_pos_arr, d_smap, n_blk, cur_h, blks, nullptr, commit_sconv, pdl, stream) != 0) return -1;
        }

        int loop_len = cfg.loop_end - cfg.loop_beg;
        if (loop_len > 0) {
            if (cfg.n_loop_passes > 1) {
                wv_launch_pdl(pdl, dim3(c_grd), dim3(256), 0, stream, wv_ink_save_prelude_kernel, cur_h, blks, pre_h, pre_blk, M, 6);
            }
            int n_blk_after_prelude = n_blk;
            float* msgs = ws_msgs_f32.as<float>();
            for (int p = 0; p < cfg.n_loop_passes; ++p) {
                if (p > 0) {
                    int prv_src_slot = std::max(0, n_blk - 1);
                    wv_launch_pdl(pdl, dim3(c_grd), dim3(256), 0, stream,
                        wv_ink_loop_pass_reset_kernel, cur_h, prev_h, pre_h, blks, prv_blk, pre_blk, M, 6, prv_src_slot);
                    n_blk = n_blk_after_prelude;
                }
                for (int j = 0; j < loop_len; ++j) {
                    int v_idx = cfg.loop_beg + p * loop_len + j;
                    if (capture_taps) maybe_stash_tap(v_idx, cur_h, taps, M, c_grd, pdl, stream);
                    bool use_msg = (p > 0) && (glob.msg_queries != nullptr);
                    if (use_msg) {
                        wv_launch_pdl(pdl, dim3(M), dim3(32, 6), 0, stream,
                            wv_ink_block_attn_res_kernel,
                            prv_blk, prev_h, glob.msg_queries + static_cast<size_t>(j) * 4 * INK2_HIDDEN, msgs, 1, 2, 4, 1, M);
                    }
                    if (run_visit(v_idx, B_act, S, d_pos_arr, d_smap, n_blk, cur_h, blks, use_msg ? msgs : nullptr, commit_sconv, pdl, stream) != 0) return -1;
                }
            }
        }

        int post_start = cfg.loop_beg + cfg.n_loop_passes * loop_len;
        for (int v_idx = post_start; v_idx < n_visits; ++v_idx) {
            if (capture_taps) maybe_stash_tap(v_idx, cur_h, taps, M, c_grd, pdl, stream);
            if (run_visit(v_idx, B_act, S, d_pos_arr, d_smap, n_blk, cur_h, blks, nullptr, commit_sconv, pdl, stream) != 0) return -1;
        }

        wv_launch_pdl(pdl, dim3(M), dim3(32, 6), 0, stream,
            wv_ink_block_attn_res_fused_norm_q8_kernel,
            blks, cur_h, glob.output_query, glob.final_norm, h_fin, ws_q8_a.as<int8_t>(), ws_sc_a.as<float>(), n_blk, 6, 1, M);
        if (capture_taps) {
            wv_launch_pdl(pdl, dim3(c_grd), dim3(256), 0, stream, wv_ink_stash_tap_kernel, h_fin, taps, 3, M);
        }
        return 0;
    }

    int32_t emit_m1_full_step_gpu(int B_act, bool do_sample, cudaStream_t stream) {
        bool pdl = use_pdl && dma.all_vram;
        if (forward_chunk(ws_tok_in.as<int32_t>(), B_act, 1, d_slot_pos_t.as<int32_t>(), nullptr, false, 1, pdl, stream) != 0) return -1;
        if (!do_sample) {
            wv_launch_pdl(pdl, dim3(256, B_act), dim3(128), 0, stream,
                wv_ink_swz8_gemv_m1_greedy_part_kernel,
                glob.head.pk, glob.head.sc, static_cast<const float*>(nullptr),
                ws_q8_a.as<int8_t>(), ws_sc_a.as<float>(),
                ws_argmax_v.as<float>(), ws_argmax_i.as<int32_t>(), cfg.vocab_unpad);
            wv_launch_pdl(pdl, dim3(B_act), dim3(256), 0, stream,
                wv_ink_argmax_vocab_final_kernel,
                ws_argmax_v.as<float>(), ws_argmax_i.as<int32_t>(), ws_tok_in.as<int32_t>());
        } else {
            launch_wv_ink_swz8_linear<false, true>(
                glob.head, nullptr, ws_q8_a.as<int8_t>(), ws_sc_a.as<float>(),
                ws_logits.as<float>(), INK2_HIDDEN, cfg.vocab_pad, B_act, stream, pdl, true);
            wv_launch_pdl(pdl, dim3(B_act), dim3(512), 0, stream,
                wv_ink_sample_topp_vocab_kernel,
                ws_logits.as<float>(), ws_tok_in.as<int32_t>(), sample_cfg_ptr(), cfg.vocab_pad, cfg.vocab_unpad);
        }
        wv_launch_pdl(pdl, dim3(1), dim3(64), 0, stream,
            wv_ink_advance_m1_slots_kernel,
            d_slot_pos_t.as<int32_t>(), d_slot_active_t.as<int32_t>(), sample_cfg_ptr(), B_act, 1);
        return 0;
    }

    int32_t emit_mtpk_full_step_gpu(int S_ver, bool do_sample, cudaStream_t stream) {
        bool pdl    = use_pdl && dma.all_vram;
        int K_draft = S_ver - 1;
        draft_k_block_gpu(K_draft, do_sample, pdl, stream);
        if (forward_chunk(ws_draft8.as<int32_t>(), 1, S_ver, d_slot_pos_t.as<int32_t>(), nullptr, true, 0, pdl, stream) != 0) return -1;
        launch_wv_ink_swz8_linear<false, true>(
            glob.head, nullptr, ws_q8_a.as<int8_t>(), ws_sc_a.as<float>(),
            ws_logits.as<float>(), INK2_HIDDEN, cfg.vocab_pad, S_ver, stream, pdl, true);
        if (!do_sample) {
            launch_wv_ink_argmax_vocab(
                ws_logits.as<float>(), ws_argmax_v.as<float>(), ws_argmax_i.as<int32_t>(),
                ws_target8.as<int32_t>(), S_ver, cfg.vocab_pad, cfg.vocab_unpad, pdl, stream);
            wv_launch_pdl(pdl, dim3(1), dim3(32), 0, stream,
                wv_ink_mtpk_greedy_verify_kernel,
                ws_draft8.as<int32_t>(), ws_target8.as<int32_t>(), sample_cfg_ptr(), d_step_out_t.as<int32_t>(), S_ver);
        } else {
            wv_launch_pdl(pdl, dim3(S_ver), dim3(512), 0, stream,
                wv_ink_logits_to_topp_probs_kernel,
                ws_logits.as<float>(), sample_cfg_ptr(), cfg.vocab_pad, cfg.vocab_unpad);
            wv_launch_pdl(pdl, dim3(1), dim3(512), 0, stream,
                wv_ink_mtpk_spec_verify_kernel,
                ws_draft8.as<int32_t>(), ws_logits.as<float>(), ws_mtp_q_probs.as<float>(),
                sample_cfg_ptr(), d_step_out_t.as<int32_t>(), cfg.vocab_pad, cfg.vocab_unpad, S_ver);
        }
        wv_launch_pdl(pdl, dim3(26, n_visits), dim3(256), 0, stream,
            wv_ink_commit_all_sconv_cand_kernel,
            sc_k.as<float>(), cand_k.as<float>(),
            sc_v.as<float>(), cand_v.as<float>(),
            sc_a.as<float>(), cand_a.as<float>(),
            sc_m.as<float>(), cand_m.as<float>(),
            d_step_out_t.as<int32_t>(), b_max, n_visits);
        commit_mtp_taps_gpu(ws_taps.as<__nv_bfloat16>(), 1, S_ver, d_step_out_t.as<int32_t>(), pdl, stream);
        wv_launch_pdl(pdl, dim3(1), dim3(1), 0, stream,
            wv_ink_advance_mtp8_counters_kernel,
            d_slot_pos_t.as<int32_t>(), d_mtp_pos0_t.as<int32_t>(),
            d_step_out_t.as<int32_t>(), sample_cfg_ptr());
        return 0;
    }

    int32_t emit_mtp8_full_step_gpu(bool do_sample, cudaStream_t stream) {
        return emit_mtpk_full_step_gpu(8, do_sample, stream);
    }

    int32_t prepare_vision_device(
        const void* vis_bf16, int n_vis, bool vis_is_device,
        const int32_t* h_ids, int B_act, int S_tot, int image_token_id,
        WvInkDevBuf& tmp_vis_gpu, const __nv_bfloat16** out_d_vis,
        std::vector<int32_t>& vis_map_out) {
        vis_map_out.clear();
        *out_d_vis = nullptr;
        if (vis_bf16 == nullptr || n_vis <= 0) return 0;
        int64_t tot_ph = 0;
        size_t total_toks = static_cast<size_t>(B_act) * S_tot;
        for (size_t i = 0; i < total_toks; ++i) {
            if (h_ids[i] == image_token_id) tot_ph++;
        }
        bool per_row_broadcast = (B_act > 1 && tot_ph == static_cast<int64_t>(B_act) * n_vis);
        if (tot_ph != n_vis && !per_row_broadcast) {
            return set_error("Multimodal placeholder token count mismatch with vision_embeds rows");
        }
        vis_map_out.assign(total_toks, -1);
        int vis_cur = 0;
        for (int b = 0; b < B_act; ++b) {
            if (per_row_broadcast) vis_cur = 0;
            for (int s = 0; s < S_tot; ++s) {
                if (h_ids[static_cast<size_t>(b) * S_tot + s] == image_token_id) {
                    if (vis_cur >= n_vis) return set_error("Row image placeholder count mismatch");
                    vis_map_out[static_cast<size_t>(b) * S_tot + s] = vis_cur++;
                }
            }
            if (per_row_broadcast && vis_cur != n_vis) {
                return set_error("Row image placeholder count != vision_embeds rows");
            }
        }
        if (vis_is_device) {
            *out_d_vis = reinterpret_cast<const __nv_bfloat16*>(vis_bf16);
        } else {
            size_t nb = static_cast<size_t>(n_vis) * INK2_HIDDEN * sizeof(__nv_bfloat16);
            if (tmp_vis_gpu.alloc(cfg.device_id, nb) != 0) return -1;
            WV_CUDA_CHECK(cudaMemcpyAsync(tmp_vis_gpu.ptr, vis_bf16, nb, cudaMemcpyHostToDevice, exec_stream));
            *out_d_vis = tmp_vis_gpu.as<__nv_bfloat16>();
        }
        return 0;
    }

    int32_t upload_vis_chunk_if_needed(
        const std::vector<int32_t>& vis_map,
        int B_act, int S_tot, int s0, int sc,
        std::vector<int32_t>& chunk_buf,
        const int32_t** out_d_vidx) {
        *out_d_vidx = nullptr;
        if (vis_map.empty()) return 0;
        chunk_buf.resize(static_cast<size_t>(B_act) * sc);
        bool any_vis = false;
        for (int b = 0; b < B_act; ++b) {
            for (int s = 0; s < sc; ++s) {
                int32_t v = vis_map[static_cast<size_t>(b) * S_tot + (s0 + s)];
                chunk_buf[static_cast<size_t>(b) * sc + s] = v;
                if (v >= 0) any_vis = true;
            }
        }
        if (!any_vis) return 0;
        WV_CUDA_CHECK(cudaMemcpyAsync(
            ws_vis_idx.ptr, chunk_buf.data(),
            chunk_buf.size() * sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
        WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
        *out_d_vidx = ws_vis_idx.as<int32_t>();
        return 0;
    }

    int32_t prefill(
        const int32_t* ids_b_s_host, int B_act, int S_tot,
        bool enable_mtp_commit, float temp, float top_p, uint64_t seed,
        int32_t* out_next_toks_host,
        const void* vis_bf16 = nullptr, int n_vis = 0, bool vis_is_device = false, int image_token_id = 200054) {
        if (!finalized) return set_error("Inkling engine not finalized");
        if (B_act <= 0 || B_act > b_max || S_tot <= 0 || seen_tokens + S_tot > max_seq) {
            return set_error("Inkling prefill dims exceed cache capacity");
        }
        DeviceGuard guard(cfg.device_id);
        WV_CUDA_CHECK(cudaDeviceSynchronize());
        WvInkDevBuf tmp_vis_gpu;
        const __nv_bfloat16* d_vis_ptr = nullptr;
        std::vector<int32_t> vis_map, vis_chunk;
        if (prepare_vision_device(vis_bf16, n_vis, vis_is_device, ids_b_s_host, B_act, S_tot, image_token_id, tmp_vis_gpu, &d_vis_ptr, vis_map) != 0) {
            return -1;
        }
        bool do_mtp   = has_mtp && enable_mtp_commit;
        bool do_samp  = (temp > 0.0f);
        int  step_cap = std::max(1, std::min(max_chunk_s, max_m_cap / B_act));
        if (sync_sample_cfg(temp, top_p, cfg.eos_id, seed, exec_stream) != 0) return -1;

        int32_t* h_chunk = reinterpret_cast<int32_t*>(h_prefill_pinned.ptr);
        for (int s0 = 0; s0 < S_tot; s0 += step_cap) {
            int sc = std::min(step_cap, S_tot - s0);
            if (B_act == 1) {
                WV_CUDA_CHECK(cudaMemcpyAsync(ws_tok_in.ptr, ids_b_s_host + s0,
                    static_cast<size_t>(sc) * sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
            } else {
                if (s0 > 0) WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
                for (int b = 0; b < B_act; ++b) {
                    std::memcpy(h_chunk + static_cast<size_t>(b) * sc,
                                ids_b_s_host + static_cast<size_t>(b) * S_tot + s0,
                                static_cast<size_t>(sc) * sizeof(int32_t));
                }
                WV_CUDA_CHECK(cudaMemcpyAsync(ws_tok_in.ptr, h_chunk,
                    static_cast<size_t>(B_act * sc) * sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
            }
            const int32_t* d_vidx = nullptr;
            if (upload_vis_chunk_if_needed(vis_map, B_act, S_tot, s0, sc, vis_chunk, &d_vidx) != 0) return -1;
            if (forward_chunk(ws_tok_in.as<int32_t>(), B_act, sc, d_slot_pos_t.as<int32_t>(), nullptr, do_mtp, 1, false, exec_stream, d_vidx, d_vis_ptr, n_vis) != 0) return -1;
            if (s0 + sc == S_tot) {
                for (int b = 0; b < B_act; ++b) {
                    int m_last = b * sc + (sc - 1);
                    launch_wv_ink_swz8_linear<false, true>(
                        glob.head, nullptr,
                        ws_q8_a.as<int8_t>() + static_cast<size_t>(m_last) * INK2_HIDDEN,
                        ws_sc_a.as<float>()  + static_cast<size_t>(m_last) * 72,
                        ws_logits.as<float>() + static_cast<size_t>(b) * cfg.vocab_pad,
                        INK2_HIDDEN, cfg.vocab_pad, 1, exec_stream, false);
                }
                if (!do_samp) {
                    launch_wv_ink_argmax_vocab(
                        ws_logits.as<float>(), ws_argmax_v.as<float>(), ws_argmax_i.as<int32_t>(),
                        ws_tok_out.as<int32_t>(), B_act, cfg.vocab_pad, cfg.vocab_unpad, false, exec_stream);
                } else {
                    wv_launch_pdl(false, dim3(B_act), dim3(512), 0, exec_stream,
                        wv_ink_sample_topp_vocab_kernel,
                        ws_logits.as<float>(), ws_tok_out.as<int32_t>(), sample_cfg_ptr(), cfg.vocab_pad, cfg.vocab_unpad);
                    rng_counter += 1ULL;
                }
            }
            if (do_mtp) {
                commit_mtp_taps_gpu(ws_taps.as<__nv_bfloat16>(), B_act, sc, nullptr, false, exec_stream);
                mtp_seen_tokens += sc;
                wv_launch_pdl(false, dim3(1), dim3(1), 0, exec_stream,
                    wv_ink_add_i32_kernel, d_mtp_pos0_t.as<int32_t>(), 1, sc);
            }
            seen_tokens += sc;
            for (int b = 0; b < B_act; ++b) h_slot_pos[b] = seen_tokens;
            wv_launch_pdl(false, dim3(1), dim3(std::min(64, B_act)), 0, exec_stream,
                wv_ink_add_i32_kernel, d_slot_pos_t.as<int32_t>(), B_act, sc);
        }
        WV_CUDA_CHECK(cudaMemcpyAsync(out_next_toks_host, ws_tok_out.ptr, B_act * sizeof(int32_t), cudaMemcpyDeviceToHost, exec_stream));
        WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
        return 0;
    }

    int32_t decode_m1(
        const int32_t* first_toks_host, int B_act, int max_new_tokens, int32_t eos_id,
        float temp, float top_p, uint64_t seed,
        int32_t* out_ids_b_t_host, int32_t* out_steps_done) {
        if (!finalized) return set_error("Inkling engine not finalized");
        if (B_act <= 0 || B_act > b_max || max_new_tokens <= 0) return set_error("Invalid decode_m1 args");
        DeviceGuard guard(cfg.device_id);
        bool do_samp = (temp > 0.0f);
        std::vector<int32_t> out_flat;
        out_flat.reserve(static_cast<size_t>(B_act) * max_new_tokens);
        for (int b = 0; b < B_act; ++b) out_flat.push_back(first_toks_host[b]);

        if (sync_sample_cfg(temp, top_p, eos_id, seed, exec_stream) != 0) return -1;
        WV_CUDA_CHECK(cudaMemcpyAsync(ws_tok_in.ptr, first_toks_host, static_cast<size_t>(B_act) * sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
        int32_t* h_step = reinterpret_cast<int32_t*>(h_draft_target_pinned.ptr);
        WvInkCudaGraph& g  = do_samp ? g_m1_sample[B_act] : g_m1_greedy[B_act];
        WvInkCudaGraph& gb = do_samp ? g_m1_burst4_sample : g_m1_burst4_greedy;

        int step = 1;
        while (step < max_new_tokens) {
            if (seen_tokens + 1 > max_seq) break;
            if (B_act == 1 && dma.all_vram && cfg.enable_cuda_graphs != 0 &&
                (max_new_tokens - step) >= 8 && (seen_tokens + 8) <= max_seq) {
                if (!gb.ready) {
                    bool pdl = use_pdl && dma.all_vram;
                    WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
                    WV_CUDA_CHECK(cudaStreamBeginCapture(exec_stream, cudaStreamCaptureModeThreadLocal));
                    for (int bk = 0; bk < 8; ++bk) {
                        if (emit_m1_full_step_gpu(1, do_samp, exec_stream) != 0) {
                            cudaGraph_t tmp_g = nullptr;
                            cudaStreamEndCapture(exec_stream, &tmp_g);
                            if (tmp_g) cudaGraphDestroy(tmp_g);
                            return -1;
                        }
                        wv_launch_pdl(pdl, dim3(1), dim3(1), 0, exec_stream,
                            wv_ink_m1_record_burst_step_kernel,
                            ws_tok_in.as<int32_t>(), d_slot_active_t.as<int32_t>(),
                            sample_cfg_ptr(), d_burst_out_t.as<int32_t>(), bk);
                    }
                    WV_CUDA_CHECK(cudaStreamEndCapture(exec_stream, &gb.graph));
                    WV_CUDA_CHECK(cudaGraphInstantiateWithFlags(&gb.exec, gb.graph, 0));
                    gb.ready = true;
                }
                WV_CUDA_CHECK(cudaGraphLaunch(gb.exec, exec_stream));
                WV_CUDA_CHECK(cudaMemcpyAsync(h_step, d_burst_out_t.ptr, 8 * sizeof(int32_t), cudaMemcpyDeviceToHost, exec_stream));
                WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));

                bool hit_eos = false;
                int  emitted = 0;
                for (int bk = 0; bk < 8; ++bk) {
                    out_flat.push_back(h_step[bk]);
                    emitted++;
                    if (h_step[bk] == eos_id) { hit_eos = true; break; }
                }
                seen_tokens  += emitted;
                h_slot_pos[0] = seen_tokens;
                rng_counter  += 8ULL;
                step         += emitted;
                if (hit_eos) break;
                continue;
            }

            if (dma.all_vram && cfg.enable_cuda_graphs != 0) {
                if (!g.ready) {
                    WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
                    WV_CUDA_CHECK(cudaStreamBeginCapture(exec_stream, cudaStreamCaptureModeThreadLocal));
                    if (emit_m1_full_step_gpu(B_act, do_samp, exec_stream) != 0) {
                        cudaGraph_t tmp_g = nullptr;
                        cudaStreamEndCapture(exec_stream, &tmp_g);
                        if (tmp_g) cudaGraphDestroy(tmp_g);
                        return -1;
                    }
                    WV_CUDA_CHECK(cudaStreamEndCapture(exec_stream, &g.graph));
                    WV_CUDA_CHECK(cudaGraphInstantiateWithFlags(&g.exec, g.graph, 0));
                    g.ready = true;
                }
                WV_CUDA_CHECK(cudaGraphLaunch(g.exec, exec_stream));
            } else {
                if (emit_m1_full_step_gpu(B_act, do_samp, exec_stream) != 0) return -1;
            }
            seen_tokens += 1;
            for (int b = 0; b < B_act; ++b) h_slot_pos[b] = seen_tokens;
            rng_counter += 1ULL;
            step        += 1;

            WV_CUDA_CHECK(cudaMemcpyAsync(h_step, ws_tok_in.ptr, static_cast<size_t>(B_act) * sizeof(int32_t), cudaMemcpyDeviceToHost, exec_stream));
            WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
            bool all_eos = true;
            for (int b = 0; b < B_act; ++b) {
                out_flat.push_back(h_step[b]);
                if (h_step[b] != eos_id) all_eos = false;
            }
            if (B_act == 1 && all_eos) break;
        }
        int steps_done = static_cast<int>(out_flat.size()) / B_act;
        for (int b = 0; b < B_act; ++b) {
            for (int t = 0; t < steps_done; ++t) {
                out_ids_b_t_host[static_cast<size_t>(b) * steps_done + t] = out_flat[static_cast<size_t>(t) * B_act + b];
            }
        }
        *out_steps_done = steps_done;
        return 0;
    }

    int32_t init_mtp8_stream(int32_t first_tok, int32_t eos_id, float temp, float top_p, uint64_t seed) {
        if (!finalized || !has_mtp) return set_error("MTP-8 weights not loaded or engine not finalized");
        DeviceGuard guard(cfg.device_id);
        h_slot_active[0] = 1;
        mtp_cur_s        = (mtp_policy == 8) ? 8 : 4;
        mtp_low_streak   = 0;
        WV_CUDA_CHECK(cudaMemcpyAsync(d_slot_active_t.ptr, &h_slot_active[0], sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
        if (sync_sample_cfg(temp, top_p, eos_id, seed, exec_stream) != 0) return -1;
        WV_CUDA_CHECK(cudaMemcpyAsync(ws_draft8.ptr, &first_tok, sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
        WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
        return 0;
    }

    int32_t _step_mtp8_raw(bool do_samp, int32_t* h_out, int32_t* out_n_acc, int* out_s_used = nullptr) {
        DeviceGuard guard(cfg.device_id);
        int s_ver = (mtp_policy == 4) ? 4 : ((mtp_policy == 8) ? 8 : mtp_cur_s);
        if (!dma.all_vram) s_ver = 8;
        if (out_s_used != nullptr) *out_s_used = s_ver;

        if (dma.all_vram && cfg.enable_cuda_graphs != 0) {
            WvInkCudaGraph& g = (s_ver == 4)
                ? (do_samp ? g_mtp4_sample : g_mtp4_greedy)
                : (do_samp ? g_mtp8_sample : g_mtp8_greedy);
            if (!g.ready) {
                WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
                WV_CUDA_CHECK(cudaStreamBeginCapture(exec_stream, cudaStreamCaptureModeThreadLocal));
                if (emit_mtpk_full_step_gpu(s_ver, do_samp, exec_stream) != 0) {
                    cudaGraph_t tmp_g = nullptr;
                    cudaStreamEndCapture(exec_stream, &tmp_g);
                    if (tmp_g) cudaGraphDestroy(tmp_g);
                    return -1;
                }
                WV_CUDA_CHECK(cudaStreamEndCapture(exec_stream, &g.graph));
                WV_CUDA_CHECK(cudaGraphInstantiateWithFlags(&g.exec, g.graph, 0));
                g.ready = true;
            }
            WV_CUDA_CHECK(cudaGraphLaunch(g.exec, exec_stream));
        } else {
            if (emit_mtpk_full_step_gpu(s_ver, do_samp, exec_stream) != 0) return -1;
        }
        WV_CUDA_CHECK(cudaMemcpyAsync(h_out, d_step_out_t.ptr, (s_ver + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, exec_stream));
        WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
        int n_acc        = h_out[0];
        seen_tokens     += n_acc;
        mtp_seen_tokens += n_acc;
        h_slot_pos[0]    = seen_tokens;
        rng_counter     += 1ULL;
        if (mtp_policy == 0 && dma.all_vram) {
            if (s_ver == 8) {
                if (n_acc <= 2) {
                    mtp_cur_s      = 4;
                    mtp_low_streak = 0;
                } else if (n_acc == 3) {
                    mtp_low_streak++;
                    if (mtp_low_streak >= 2) { mtp_cur_s = 4; mtp_low_streak = 0; }
                } else {
                    mtp_low_streak = 0;
                }
            } else {
                if (n_acc >= 4) {
                    mtp_cur_s      = 8;
                    mtp_low_streak = 0;
                }
            }
        }
        *out_n_acc = n_acc;
        return 0;
    }

    int32_t step_mtp8(float temp, int32_t* out_tokens_8_host, int32_t* out_n_acc) {
        if (!finalized || !has_mtp) return set_error("MTP-8 weights not loaded");
        if (seen_tokens + 8 > max_seq) { *out_n_acc = 0; return 0; }
        int32_t* h_out = reinterpret_cast<int32_t*>(h_draft_target_pinned.ptr);
        int32_t n_acc  = 0;
        if (_step_mtp8_raw((temp > 0.0f), h_out, &n_acc) != 0) return -1;
        for (int k = 0; k < n_acc; ++k) out_tokens_8_host[k] = h_out[1 + k];
        *out_n_acc = n_acc;
        return 0;
    }

    int32_t decode_mtp8(
        int32_t first_tok, int max_new_tokens, int32_t eos_id, float temp, float top_p, uint64_t seed,
        int32_t* out_tokens_host, int32_t* out_len, WvInkMtpStatsOut* out_stats) {
        if (!finalized || !has_mtp) return set_error("MTP-8 weights not loaded");
        if (max_new_tokens <= 0) return set_error("max_new_tokens must be > 0");
        bool do_samp = (temp > 0.0f);
        std::vector<int32_t> gen;
        gen.reserve(static_cast<size_t>(max_new_tokens) + 8);
        gen.push_back(first_tok);
        int32_t cur_anchor   = first_tok;
        int64_t spec_steps   = 0, drafted_tot = 0, accepted_tot = 0;
        int64_t s4_steps     = 0, s8_steps    = 0;

        if (init_mtp8_stream(first_tok, eos_id, temp, top_p, seed) != 0) return -1;
        int32_t* h_out = reinterpret_cast<int32_t*>(h_draft_target_pinned.ptr);

        while (static_cast<int>(gen.size()) < max_new_tokens && cur_anchor != eos_id && seen_tokens + 8 <= max_seq) {
            int s_used    = 8;
            int32_t n_acc = 0;
            if (_step_mtp8_raw(do_samp, h_out, &n_acc, &s_used) != 0) return -1;
            if (s_used == 4) s4_steps += 1;
            else             s8_steps += 1;
            int emitted = 0;
            for (int k = 0; k < n_acc; ++k) {
                if (static_cast<int>(gen.size()) < max_new_tokens) {
                    gen.push_back(h_out[1 + k]);
                    emitted++;
                    if (h_out[1 + k] == eos_id) break;
                } else {
                    break;
                }
            }
            cur_anchor    = gen.back();
            spec_steps   += 1;
            drafted_tot  += (s_used - 1);
            accepted_tot += emitted;
            if (emitted < n_acc) break;
        }
        std::memcpy(out_tokens_host, gen.data(), gen.size() * sizeof(int32_t));
        *out_len = static_cast<int32_t>(gen.size());
        if (out_stats != nullptr) {
            out_stats->spec_steps        = spec_steps;
            out_stats->drafted_tokens    = drafted_tot;
            out_stats->accepted_tokens   = accepted_tot;
            out_stats->mean_accepted_len = (spec_steps > 0) ? (static_cast<double>(accepted_tot) / static_cast<double>(spec_steps)) : 1.0;
            out_stats->s4_steps          = s4_steps;
            out_stats->s8_steps          = s8_steps;
        }
        return 0;
    }

    int32_t reset_slot(int b) {
        if (!finalized || b < 0 || b >= b_max) return set_error("slot out of range");
        DeviceGuard guard(cfg.device_id);
        for (int v = 0; v < n_visits; ++v) {
            size_t row_kv = (static_cast<size_t>(v) * b_max + b) * 3 * 1024;
            size_t row_h  = (static_cast<size_t>(v) * b_max + b) * 3 * INK2_HIDDEN;
            WV_CUDA_CHECK(cudaMemsetAsync(sc_k.as<float>() + row_kv, 0, 3 * 1024 * sizeof(float), exec_stream));
            WV_CUDA_CHECK(cudaMemsetAsync(sc_v.as<float>() + row_kv, 0, 3 * 1024 * sizeof(float), exec_stream));
            WV_CUDA_CHECK(cudaMemsetAsync(sc_a.as<float>() + row_h,  0, 3 * INK2_HIDDEN * sizeof(float), exec_stream));
            WV_CUDA_CHECK(cudaMemsetAsync(sc_m.as<float>() + row_h,  0, 3 * INK2_HIDDEN * sizeof(float), exec_stream));
        }
        h_slot_pos[b]    = 0;
        h_slot_active[b] = 0;
        WV_CUDA_CHECK(cudaMemcpyAsync(d_slot_pos_t.as<int32_t>() + b,    &h_slot_pos[b],    sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
        WV_CUDA_CHECK(cudaMemcpyAsync(d_slot_active_t.as<int32_t>() + b, &h_slot_active[b], sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
        WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
        return 0;
    }

    int32_t deactivate_slot(int b) {
        if (!finalized || b < 0 || b >= b_max) return set_error("slot out of range");
        DeviceGuard guard(cfg.device_id);
        h_slot_pos[b]    = 0;
        h_slot_active[b] = 0;
        WV_CUDA_CHECK(cudaMemcpyAsync(d_slot_pos_t.as<int32_t>() + b,    &h_slot_pos[b],    sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
        WV_CUDA_CHECK(cudaMemcpyAsync(d_slot_active_t.as<int32_t>() + b, &h_slot_active[b], sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
        WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
        return 0;
    }

    int32_t prefill_slot(
        int b, const int32_t* ids_s_host, int S_tot,
        float temp, float top_p, uint64_t seed, int32_t* out_tok0,
        const void* vis_bf16 = nullptr, int n_vis = 0, bool vis_is_device = false, int image_token_id = 200054) {
        if (!finalized || b < 0 || b >= b_max || S_tot <= 0 || S_tot > max_seq) {
            return set_error("prefill_slot args out of range");
        }
        DeviceGuard guard(cfg.device_id);
        if (reset_slot(b) != 0) return -1;
        WvInkDevBuf tmp_vis_gpu;
        const __nv_bfloat16* d_vis_ptr = nullptr;
        std::vector<int32_t> vis_map, vis_chunk;
        if (prepare_vision_device(vis_bf16, n_vis, vis_is_device, ids_s_host, 1, S_tot, image_token_id, tmp_vis_gpu, &d_vis_ptr, vis_map) != 0) {
            return -1;
        }
        WV_CUDA_CHECK(cudaMemcpyAsync(d_slot_map_t.ptr, &b, sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
        if (sync_sample_cfg(temp, top_p, cfg.eos_id, seed, exec_stream) != 0) return -1;

        for (int s0 = 0; s0 < S_tot; s0 += max_chunk_s) {
            int sc = std::min(max_chunk_s, S_tot - s0);
            WV_CUDA_CHECK(cudaMemcpyAsync(ws_tok_in.ptr, ids_s_host + s0,
                static_cast<size_t>(sc) * sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
            const int32_t* d_vidx = nullptr;
            if (upload_vis_chunk_if_needed(vis_map, 1, S_tot, s0, sc, vis_chunk, &d_vidx) != 0) return -1;
            if (forward_chunk(ws_tok_in.as<int32_t>(), 1, sc, d_slot_pos_t.as<int32_t>(), d_slot_map_t.as<int32_t>(), false, 1, false, exec_stream, d_vidx, d_vis_ptr, n_vis) != 0) return -1;
            h_slot_pos[b] += sc;
            wv_launch_pdl(false, dim3(1), dim3(1), 0, exec_stream,
                wv_ink_add_i32_kernel, d_slot_pos_t.as<int32_t>() + b, 1, sc);
            if (s0 + sc == S_tot) {
                int m_last = sc - 1;
                launch_wv_ink_swz8_linear<false, true>(
                    glob.head, nullptr,
                    ws_q8_a.as<int8_t>() + static_cast<size_t>(m_last) * INK2_HIDDEN,
                    ws_sc_a.as<float>()  + static_cast<size_t>(m_last) * 72,
                    ws_logits.as<float>(), INK2_HIDDEN, cfg.vocab_pad, 1, exec_stream, false);
                if (temp <= 0.0f) {
                    launch_wv_ink_argmax_vocab(
                        ws_logits.as<float>(), ws_argmax_v.as<float>(), ws_argmax_i.as<int32_t>(),
                        ws_tok_out.as<int32_t>(), 1, cfg.vocab_pad, cfg.vocab_unpad, false, exec_stream);
                } else {
                    wv_launch_pdl(false, dim3(1), dim3(512), 0, exec_stream,
                        wv_ink_sample_topp_vocab_kernel,
                        ws_logits.as<float>(), ws_tok_out.as<int32_t>(), sample_cfg_ptr(), cfg.vocab_pad, cfg.vocab_unpad);
                    rng_counter += 1ULL;
                }
            }
        }
        int32_t tok0 = 0;
        h_slot_active[b] = 1;
        WV_CUDA_CHECK(cudaMemcpyAsync(d_slot_active_t.as<int32_t>() + b, &h_slot_active[b], sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
        WV_CUDA_CHECK(cudaMemcpyAsync(&tok0, ws_tok_out.ptr, sizeof(int32_t), cudaMemcpyDeviceToHost, exec_stream));
        WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
        *out_tok0 = tok0;
        return 0;
    }

    int32_t step_continuous_m1(
        const int32_t* cur_toks_host, int B_span, bool upload_cur_toks,
        float temp, float top_p, uint64_t seed, int32_t* out_toks_host) {
        if (!finalized || B_span < 1 || B_span > b_max) return set_error("B_span out of range");
        DeviceGuard guard(cfg.device_id);
        bool do_samp = (temp > 0.0f);
        if (upload_cur_toks) {
            if (cur_toks_host == nullptr) return set_error("Null cur_toks_host when upload_cur_toks is set");
            if (sync_sample_cfg(temp, top_p, cfg.eos_id, seed, exec_stream) != 0) return -1;
            WV_CUDA_CHECK(cudaMemcpyAsync(ws_tok_in.ptr, cur_toks_host,
                static_cast<size_t>(B_span) * sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
        }
        WvInkCudaGraph& g = do_samp ? g_m1_sample[B_span] : g_m1_greedy[B_span];
        if (dma.all_vram && cfg.enable_cuda_graphs != 0) {
            if (!g.ready) {
                WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
                WV_CUDA_CHECK(cudaStreamBeginCapture(exec_stream, cudaStreamCaptureModeThreadLocal));
                if (emit_m1_full_step_gpu(B_span, do_samp, exec_stream) != 0) {
                    cudaGraph_t tmp_g = nullptr;
                    cudaStreamEndCapture(exec_stream, &tmp_g);
                    if (tmp_g) cudaGraphDestroy(tmp_g);
                    return -1;
                }
                WV_CUDA_CHECK(cudaStreamEndCapture(exec_stream, &g.graph));
                WV_CUDA_CHECK(cudaGraphInstantiateWithFlags(&g.exec, g.graph, 0));
                g.ready = true;
            }
            WV_CUDA_CHECK(cudaGraphLaunch(g.exec, exec_stream));
        } else {
            if (emit_m1_full_step_gpu(B_span, do_samp, exec_stream) != 0) return -1;
        }
        for (int b = 0; b < B_span; ++b) {
            if (h_slot_active[b] > 0) h_slot_pos[b] += 1;
        }
        rng_counter += 1ULL;
        WV_CUDA_CHECK(cudaMemcpyAsync(out_toks_host, ws_tok_in.ptr, static_cast<size_t>(B_span) * sizeof(int32_t), cudaMemcpyDeviceToHost, exec_stream));
        WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
        return 0;
    }

    int32_t forward_all_logits_f32(
        const int32_t* ids_s_host, int S_tot, float* out_logits_s_v_host,
        const void* vis_bf16 = nullptr, int n_vis = 0, bool vis_is_device = false, int image_token_id = 200054) {
        if (!finalized || S_tot <= 0 || S_tot > max_seq) return set_error("forward_all_logits_f32 out of range");
        DeviceGuard guard(cfg.device_id);
        if (reset_cache() != 0) return -1;
        WvInkDevBuf tmp_vis_gpu;
        const __nv_bfloat16* d_vis_ptr = nullptr;
        std::vector<int32_t> vis_map, vis_chunk;
        if (prepare_vision_device(vis_bf16, n_vis, vis_is_device, ids_s_host, 1, S_tot, image_token_id, tmp_vis_gpu, &d_vis_ptr, vis_map) != 0) {
            return -1;
        }

        for (int s0 = 0; s0 < S_tot; s0 += max_chunk_s) {
            int sc = std::min(max_chunk_s, S_tot - s0);
            WV_CUDA_CHECK(cudaMemcpyAsync(ws_tok_in.ptr, ids_s_host + s0,
                static_cast<size_t>(sc) * sizeof(int32_t), cudaMemcpyHostToDevice, exec_stream));
            const int32_t* d_vidx = nullptr;
            if (upload_vis_chunk_if_needed(vis_map, 1, S_tot, s0, sc, vis_chunk, &d_vidx) != 0) return -1;
            if (forward_chunk(ws_tok_in.as<int32_t>(), 1, sc, d_slot_pos_t.as<int32_t>(), nullptr, false, 1, false, exec_stream, d_vidx, d_vis_ptr, n_vis) != 0) return -1;
            seen_tokens += sc;
            h_slot_pos[0] = seen_tokens;
            wv_launch_pdl(false, dim3(1), dim3(1), 0, exec_stream,
                wv_ink_add_i32_kernel, d_slot_pos_t.as<int32_t>(), 1, sc);
            for (int sub = 0; sub < sc; sub += 32) {
                int sb = std::min(32, sc - sub);
                launch_wv_ink_swz8_linear<false, true>(
                    glob.head, nullptr,
                    ws_q8_a.as<int8_t>() + static_cast<size_t>(sub) * INK2_HIDDEN,
                    ws_sc_a.as<float>()  + static_cast<size_t>(sub) * 72,
                    ws_logits.as<float>(), INK2_HIDDEN, cfg.vocab_pad, sb, exec_stream, false);
                WV_CUDA_CHECK(cudaMemcpy2DAsync(
                    out_logits_s_v_host + static_cast<size_t>(s0 + sub) * cfg.vocab_unpad,
                    static_cast<size_t>(cfg.vocab_unpad) * sizeof(float),
                    ws_logits.as<float>(),
                    static_cast<size_t>(cfg.vocab_pad) * sizeof(float),
                    static_cast<size_t>(cfg.vocab_unpad) * sizeof(float),
                    static_cast<size_t>(sb),
                    cudaMemcpyDeviceToHost, exec_stream));
            }
        }
        WV_CUDA_CHECK(cudaStreamSynchronize(exec_stream));
        return 0;
    }

    void reset_dma_stats() { dma.reset_stats(); }

    void get_dma_stats(WvInkDmaStatsOut* out) const {
        uint64_t tot = dma.dma_hits + dma.dma_misses;
        out->all_vram          = dma.all_vram ? 1 : 0;
        out->num_cache_slots   = dma.num_cache_slots;
        out->cache_mib         = static_cast<double>(dma.cache_bytes) / (1024.0 * 1024.0);
        out->dma_hits          = dma.dma_hits;
        out->dma_prefetch_hits = dma.dma_prefetch_hits;
        out->dma_misses        = dma.dma_misses;
        out->hit_rate_pct      = (tot > 0) ? (100.0 * static_cast<double>(dma.dma_hits) / static_cast<double>(tot)) : 100.0;
        out->h2d_bytes         = dma.h2d_bytes;
        out->h2d_mib           = static_cast<double>(dma.h2d_bytes) / (1024.0 * 1024.0);
    }
};

} // namespace wv

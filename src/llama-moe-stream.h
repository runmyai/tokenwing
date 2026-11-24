#pragma once

// The achieved lead of a guess, in 50 us buckets up to 11.2 ms, then one overflow bucket. A layer
// of decode is 1.1-1.8 ms (measured), so 50 us is a fiftieth of the thing being measured.
#define MOE_STREAM_LEAD_NB 225

// Why a waiting-list entry left the list without being bound (the counters' index and the recorder's DROP code)
enum {
    MOE_STREAM_PL_PICKED  = 0,   // its target's remap picked the expert (a needed read is issued for it there)
    MOE_STREAM_PL_PRESENT = 1,   // at a reader's turn the expert was already in the target pool (READY or LOADING) or VRAM-held
    MOE_STREAM_PL_NOROOM  = 2,   // at a reader's turn the target pool had no slot a guess may take
    MOE_STREAM_PL_WORDEND = 3,   // the word ended (its target had run, or was the last trunk layer in a call with no output row)
    MOE_STREAM_PL_FULL    = 4,   // the list was full when it was added (never added)
    MOE_STREAM_PL_ABORTED = 5,   // the decode call stopped part-way (abort callback, failed compute) before its target ran
    MOE_STREAM_PL_STALE   = 6,   // --moe-stream-drop-stale 1: its target ran and did not pick it
    MOE_STREAM_PL_NREASON = 7,
};

#include "llama-mmap.h"
#include "llama-moe-correction.h"
#include "llama-moe-wave-plan.h"

#include "ggml.h"
#include "ggml-cpu.h"   // the CPU kernels' input types (down_scale_skip)
#include "ggml-backend.h"
#include "ggml-cpp.h"   // the owning pointers of the GPU copy of the full bytes

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#include <malloc.h>
#endif

// TEST ONLY: MOE_STREAM_TEST_REQ_TIMES=1 - one stderr line "TT <us> <tag> <arg>" per step of a request (llama-server writes the
// same lines from its side), on ggml_time_us = the server's own prompt clock; read once. Off: one test of a static bool per step
inline bool moe_tt_on() {
    static const bool on = getenv("MOE_STREAM_TEST_REQ_TIMES") && atoi(getenv("MOE_STREAM_TEST_REQ_TIMES")) != 0;
    return on;
}
inline void moe_tt(const char * tag, long long arg = 0) {
    if (moe_tt_on()) {
        fprintf(stderr, "TT %lld %s %lld\n", (long long) ggml_time_us(), tag, arg);
    }
}

bool llama_moe_stream_dev_is_uma(ggml_backend_dev_t dev);

// The CPU backend's MoE entry table (ggml-cpu.h, struct ggml_cpu_moe_api), found once through the "CPU" registry: the engine
// never links to ggml-cpu, so it also runs when the CPU backend is a module picked at start for the CPU's level
const ggml_cpu_moe_api * llama_cpu_moe_api();

// SSD streaming of MoE routed expert weights
//
// Streamed layers do not materialize their ffn_{gate,up,down}_exps tensors: the routed experts
// are read from a repacked sidecar file (tokenwing-tools/experts/expert_repack.py) into a per-layer pool of
// n_slots experts, which lives in host memory so that the expert GEMMs run on the CPU. An
// id-remapping custom op runs on the CPU right after the router top-k: it makes every expert
// that the ubatch selected resident and rewrites the ids to pool slots. The remap only relabels
// ids, so the routing - which experts, in which order, with which weights - is untouched and the
// output is identical to a run without streaming.
//
// The reads run on a small pool of I/O threads so that the misses of a layer are in flight at the
// same time, and the graph hands the op a second input: the experts that the next layer's router
// predicts from this layer's router input (the layer-ahead predictor). Those are
// prefetched, without waiting for them, while the current layer's experts are computed.

struct llama_moe_stream;

// where the sidecar keeps one layer's experts
struct moe_stream_block {
    size_t base_offset = 0;  // sidecar offset of expert 0 of this layer
    size_t block_bytes = 0;  // sidecar bytes per expert of this layer
    // The sidecar stores this layer's experts in llama.cpp's repacked CPU layout, so a read of
    // them is a copy into the slot and not a conversion
    bool   repacked = false;
    // The experts are read from the model file itself (--moe-stream model): the three projections of an expert are in
    // three places, so a block is virtual (gate | up | down, as the slot holds them) and each projection says where its
    // expert 0 is in the file (file_off). base_offset is not used then.
    bool   by_proj = false;

    struct projection {
        size_t      offs  = 0;  // offset inside the block
        size_t      size  = 0;  // bytes of one expert slice
        size_t      file_off = 0;  // by_proj: file offset of expert 0 of this projection, expert e is at file_off + e*size
        std::string name;      // name of the GGUF tensor it was cut from
        // Where that tensor is in the model, so the engine can read the same bytes again
        // and compare a streamed slot against the file the sidecar claims to come from. Expert e of it
        // is at src_off + e*size in shard src_shard. has_src is false for a sidecar built before this
        // was recorded, and then the read-back check compares the slot against the sidecar only.
        size_t      src_off  = 0;
        int32_t     src_shard = 0;
        // false when the metadata does not say where the tensor is in the model, and also when it does
        // not say which *shard*: a sidecar of a split GGUF built before the shard was recorded
        // would otherwise be compared against the wrong file (which the check reported on the IQ4_XS
        // sidecar: layer 41, 1,044,480 bytes, first at +0)
        bool        has_src   = false;
    } gate, up, down;
};

// one streamed projection (gate, up or down) of one layer
struct llama_moe_stream_weight {
    ggml_tensor * pool = nullptr;  // [ne0, ne1, n_slots], host memory
    void        * mem  = nullptr;  // page-aligned base of the pool
    size_t        offs = 0;        // offset of this projection inside the sidecar expert block
    size_t        size = 0;        // bytes of one expert slice
    bool          plugin_repack = false;   // slots hold the quant plugin's own layout (a plugin type, --moe-stream-cpu-repack 1)
};

// How a layer's pool picks the slot to overwrite. "lru" is the only one the engine had;
// "lfu" is the decayed frequency of vllm-expert-cache (every frequency halved every 64 accesses, so the
// score tracks recent use and not the whole history) and "slru" is the two-segment LRU of nvmoe's
// segmented_host_lru.hpp (a hit promotes a slot out of probation, victims come from probation first).
enum llama_moe_stream_policy {
    MOE_STREAM_POLICY_LRU  = 0,
    MOE_STREAM_POLICY_LFU  = 1,
    MOE_STREAM_POLICY_SLRU = 2,
    MOE_STREAM_POLICY_DECAY = 3,   // lowest decayed use count
};

// Options; the defaults keep the plain path
struct llama_moe_stream_p1 {
    bool        io_wake_chunk  = false;   // an enqueue wakes one I/O thread per chunk, not one per read
    bool        promote        = false;   // a needed expert's queued guess chunks move to the demand queue
    int32_t     guess_cap      = -1;      // max guess chunks in service while a needed read is active, < 0 = off
    bool        cap_refill     = false;   // with guess_cap: a finished guess chunk wakes a thread for the freed place
    float       evict_halflife = 64.0f;   // policy "decay": words after which a use counts half
    float       evict_credit   = 0.0f;    // policy "decay": value for a 1-ahead top-16 candidate that was not chosen
    bool        prompt_end     = false;   // order the waves of a prompt's last ubatch by recent use
    bool        prompt_values  = false;   // policy "decay": a prompt's expert uses count as decode words would
    float       prompt_end_halflife = 128.0f;   // tokens
    std::string chance_model;             // probability table for the 1-ahead reads, "" = off
    float       read_threshold = 0.3f;    // read a 1-ahead candidate when its probability is at least this
    bool        cpu_skip_zero  = false;   // the CPU expert matmul writes 0 for the zero slices instead of computing them
    int         cpu_x4         = 0;       // 4-row MXFP4 kernel (same bits): 0 off, 1 256-bit, 2 512-bit registers
    int         cpu_x4_pf      = -1;      // its prefetch: -1 the next 4-row group, 0 none, N blocks ahead
    bool        cpu_fuse       = false;   // fused gate+up+swiglu step for one token (same bits)
    bool        cpu_dispatch   = false;   // ik's expert split for one-token mul_mat_id (same bits)
    int         cpu_pieces     = 1;       // work pieces per thread and expert, one-token CPU expert steps (same bits)
    bool        cpu_conv1      = false;   // the one-token activation row converted by one thread (same bits)
    bool        cpu_wsum       = false;   // a CPU layer's expert weighted sum as one op (same bits)
    int         cpu_iqk        = 0;       // the CPU's picked-expert matmul at prompt sizes uses ik's kernels from this many rows per expert (0 = off)
    bool        cpu_mmid_merge = false;   // the tiled MUL_MAT_ID's groups of a thread as one window (ggml_cpu_set_mmid_merge)
    bool        guess_skip_full = false;  // no guess for a target layer that holds every expert in RAM (READY)
    bool        nonpool = false;              // --moe-stream-nonpool-measured 1
    bool        guess_skip_full_gpu = false;  // the GPU's guess skips them too (the 'full' bytes copied to the GPU per call)
    int         cpu_fuse_small = 0;       // fewer small CPU steps, a mask (1 router chain, 2 guess chain, 64 same-input matmuls)
    int         cpu_fa         = 0;       // CPU flash attention with q8_0 KV: 1 = tiled prompts + the GQA decode split
    int32_t     cpu_q8_tiled   = 0;       // > 0 = q8_0 weights on the CPU's tiled matmul, 1 = from 12 rows, else from N rows (ggml_cpu_set_q8_tiled)
    bool        cpu_topk       = false;   // the router / guess top k in one pass on a machine with no GPU device (same ids)
    bool        cpu_down_scale = false;   // skip the down-input scale step where it is exact on the CPU (same bits)
    std::vector<uint8_t> wsum_cpu_tail;   // per layer, 1 = the layer and what consumes its output (the next
                                          //       layer, or the output after the last one) are on the CPU device
    bool        cpu_state_inplace = false; // CPU layers' delta step reads / writes the state in place (same bits)
    std::vector<uint8_t> state_cpu_layer;   // per layer, 1 = a recurrent layer on the CPU device
    bool        cpu_small_ops  = false;   // the expert pick on the kept groups, CONCAT split (same bits)
    bool        fused_route    = false;   // the grouped expert selection (router and guesses) as one ggml_moe_group_topk op
    bool        slot_check     = true;    // a pair with no RAM slot stops the run (one-wave calls; every pair once over a
                                          //       ubatch's waves); false = off (emergency only)
    bool        wave_plan      = false;   // a multi-wave call's waves come from a plan built once at wave 0
                                          //       (residents first, packed by pair count under the wave cap), not from first-use rank
    bool        wave_victim    = false;   // a slot's wave mark (slot_wave) lasts the whole multi-wave call, so
                                          //       pick_victim's "an expert of an earlier wave of this call" branch can fire
    bool        conv_silu      = false;   // Ling's KDA conv: SiLU before the reshape, so CUDA fuses conv + SiLU
    bool        hc_share       = false;   // hyper-connection models: an FFN combine's 2*sigmoid(inject) weights computed once,
                                          //       for the guess's shared-expert combine and the layer's own combine
    bool        rs_window      = false;   // qwen4exp reply graphs with the GPU producer: the next delta-net layer's state fetch
                                          //       and this layer's conv-state write in the MoE layer's GPU window
    bool        ple_late       = false;   // qwen4exp: the n-gram embedding rows gathered right before the layer that reads them
    bool        hc_silu        = false;   // hyper-connection models: a mix's scale + SiLU tagged for one CUDA kernel
    bool        hc_q8          = false;   // hyper-connection models: a mix's norm (and fused SiLU) write the matvecs' q8_1 copies
    bool        rs_inplace     = false;   // qwen4exp reply graphs of one sequence: the delta step reads its state row in place
    bool        gpu_share_q8   = false;   // CUDA matvecs reading the same input bytes share one q8_1 copy
    uint32_t    vram_experts   = 0;       // the first N experts of the list of every streamed layer live in VRAM, 0 = off
    std::string vram_experts_file;        // the ranked list per layer ("tokenwing vram-experts v1")
    int64_t     vram_experts_margin = (int64_t) 512*1024*1024;   // VRAM that must stay free after the pool
    int64_t     vram_free = 0;      // --moe-stream-vram-free: end-state keep-free that sizes N; 0 = off
    int64_t     vram_scratch = -1;  // --moe-stream-vram-scratch: the op scratch pool mapped before the check, bytes; -1 = derived by llama_context, 0 = not mapped
    int64_t     vram_growth = 0;    // --moe-stream-vram-growth: the fit's reserve for lazy allocations; 0 = derived
    bool        mtp_on_pool_dev = false;   // --spec-type draft-mtp loads NextN on the pool's GPU; ctx_dft is created after vx_setup
    int64_t     mtp_ctx_bytes   = 0;       // NextN KV/recurrent bytes ctx_dft allocates on the pool's GPU after vx_setup
    std::string correction_file;   // --moe-stream-correction FILE: the trained correction of the router-trick guess, "" = off
    uint64_t    correction_base = 0, correction_size = 0;   // the file is the `size` bytes at `base` of correction_file (the word `model`: data appended to the model file); 0 = the whole file
    bool        idle_load      = false;   // --moe-stream-idle-load 1 or 2 (idle-time loading)
    bool        fill_start     = false;   // --moe-stream-idle-load 2: every RAM slot filled before the model counts as loaded
    std::string usage_file;               // --moe-stream-usage-file, "" = <sidecar>.usage.json
    int32_t     idle_pause_ms  = 500;     // --moe-stream-idle-pause
    bool        read_ahead     = false;   // --moe-stream-read-ahead 1
    int32_t     read_ahead_min = 4096;    // --moe-stream-read-ahead-min (tokens per call)
    bool        window         = false;   // --moe-stream-window 1
    float       window_keep    = 1.0f;    // --moe-stream-window-keep F
    bool        prompt_defer   = false;   // --moe-stream-prompt-defer 1
    bool        prompt_split   = false;   // --moe-stream-prompt-split 1
    int32_t     prompt_split_places = 128;   // --moe-stream-prompt-split-places N
    bool        gpu_wave_skip_zero = false;   // --moe-stream-gpu-wave-skip-zero 1
    bool        gpu_skip_zero  = false;   // the GPU one-token mmvq writes 0 for the VRAM pool's zero slices
    bool        prefill_fast   = false;   // --moe-stream-prefill-fast 1
    bool        mmq_moe_opt    = false;   // --moe-stream-mmq-moe-opt 1
    bool        gpu_moe_fused  = false;   // --moe-stream-gpu-moe-fused 1
    int64_t     pinned_ring    = 0;       // --moe-stream-pinned-ring SIZE
    bool        lm_bf16        = false;   // --moe-stream-lm-bf16 1
    bool        lend_output    = false;   // --moe-stream-lend-output 1
    bool        qsa_fused      = false;   // --moe-stream-qsa-fused 1
    bool        hc_fused       = false;   // --moe-stream-hc-fused 1
    bool        ple_fast       = false;   // --moe-stream-ple-fast 1
    bool        plan           = false;   // --moe-stream-plan 1, the waiting list with late binding (decode words)
    bool        plan_bind_early = false;  // test only - every list entry is bound the moment it is added
    bool        drop_stale     = false;   // at its target's start, entries it did not pick go, and their started reads'
                                          //           unstarted pieces are dropped
    bool        read_threshold_set = false;   // --moe-stream-read-threshold was given (the engine's plan-1 guard)
    bool        needed_early   = false;   // --moe-stream-needed-early 1 - a decode word's 1-ahead guess is handed to the
                                          //           helper thread (the host producer: posted by the remap op, for GPU-split layers only;
                                          //           which layers: ne_set_router_gpu(), after the weights are placed)
    int32_t     behind_backlog = 0;       // --moe-stream-behind-backlog B (0 = off): "behind" = >= B reads not finished
    int32_t     behind_rank    = 0;       // --moe-stream-behind-rank R (0 = off): while behind, entries ranked above R wait
    int32_t     guess_inflight = 0;       // --moe-stream-guess-inflight G (0 = no limit): guess pieces in service while no
                                          //           needed read is
    int32_t     guess_top      = 0;       // --moe-stream-guess-top K (0 = off): a decode word's 1-ahead guess is K wide
    int32_t     lookahead      = 1;       // --moe-stream-lookahead N (1 = the default): guesses for the layers 1..N ahead
    int32_t     guess_pace     = 1;       // --moe-stream-guess-pace B: new entries per far guess and target
    bool        plan_old_order = false;   // --moe-stream-plan-old-order 1 (test only): the list's order before chunk 8
    bool        plan_extra_near = false;  // --moe-stream-plan-extra-near 1 (test only): order A for the extra ranks
    bool        guess_combined = false;   // --moe-stream-guess-combined 1: a layer's N guesses in one step
    bool        protect_named  = false;   // --moe-stream-protect-named 1: a named READY expert is no guess read's victim
    bool        proj_ready     = false;   // --moe-stream-proj-ready 1: a ready flag per projection, gate / up read first
    int32_t     wordstart      = 0;       // --moe-stream-wordstart D (0 = off): word-start guesses from the last D
                                          //          layers before the first streamed one
    // VRAM follow: the moving VRAM expert set - the VRAM expert pool shared by all streamed layers and refilled while a
    // reply is written by AI-der's rule (qwen38_tier.c:743-793, tier.h:9-14); every number a setting, the defaults AI-der's shipped ones
    bool        vram_follow          = false;   // --moe-stream-vram-follow 1 (needs --moe-stream-vram-experts N)
    uint32_t    vram_follow_tick     = 8;       // a tick every N reply words
    uint32_t    vram_follow_swaps    = 64;      // up to N swaps per tick
    uint32_t    vram_follow_recent   = 256;     // the recent count is halved every N reply words
    uint32_t    vram_follow_longterm = 1024;    // the long-term count is halved every N reply words
    double      vram_follow_ratio    = 1.25;    // a swap needs hot > floor(ratio x cold) + margin (AI-der: cold + cold/4 + 4)
    uint32_t    vram_follow_margin   = 4;
    bool        vram_follow_fixed    = false;   // test only: the held set a function of the text only (synchronous switches)
    bool        vram_follow_pinned   = true;    // the uploader copies through a pinned bounce (13.6 against 12.3 GB/s)
    // --moe-stream-vram-parallel 1 - in a GPU-producer one-token decode graph the VRAM pool's device
    // pass runs in the layer's GPU launch after "router done" (ids from a per-layer table in VRAM), during the CPU experts; with
    // several GPUs only in layers whose router runs on the pool's GPU. 2 (vram_parallel_cross) - also in the other layers: their pass
    // runs on the pool's GPU and its inputs and sum cross between the cards
    bool        vram_parallel        = false;
    bool        vram_parallel_cross  = false;
    // --moe-stream-vram-copyback 1 - an expert leaving VRAM follow's set is copied back into its RAM pool
    // (an empty slot, or a keep value above the pool's lowest), and held experts' picks count in the keep value
    bool        vram_copyback        = false;
    // --moe-stream-async-h2d 1 - CPU -> GPU split inputs in the GPU's pinned host buffer as async
    // H2D copies on the GPU's stream, no host wait (llama-context turns it on in the scheduler: one GPU, no pipeline parallelism)
    bool        async_h2d            = false;
    int32_t     tail_wave_w          = 0;       // --moe-stream-tail-wave-w N (0 = off): narrow GPU-staged PCIe wave width
    int64_t     vram_chunk           = 0;       // --moe-stream-vram-chunk SIZE (1 = auto <= 2 GiB, > 1 = bytes, 0 = off)
    bool        uma                  = false;   // --moe-stream-uma 1
    ggml_backend_dev_t uma_dev       = nullptr; // active unified-memory GPU device selected at model load
    bool        hip_mmvq             = false;   // --moe-stream-hip-mmvq 1
    int32_t     hip_sync             = 0;       // --moe-stream-hip-sync N (1 = auto 400 us, > 1 = us, 0 = off)
    // Elastic RAM and moving slot shares (both off = fixed, equal pools)
    bool        elastic              = false;
    int64_t     ram_margin           = 0;       // 0 = OS zero-paging default (1.25 GiB Linux / 1.5 GiB Win & macOS)
    bool        ram_margin_set       = false;   // the user set keep-free
    std::string elastic_opts;
    bool        shares               = false;
    std::string share_file;
    std::string share_opts;
    bool        layer_major = false;   // --moe-stream-layer-major 1 (llama_context's prompt loop; the engine only carries it)
    uint32_t    layer_major_batch = 0; // --moe-stream-layer-major-batch (the cap of llama_context's n_batch raise; 0 = n_ctx)
    bool        vram_lend_prompt  = false; // --moe-stream-vram-lend-prompt 1
    int32_t     vram_lend_short   = 64;    // --moe-stream-vram-lend-short N (prompt calls of <= N tokens in the real buffer)
    bool        timing            = false; // --moe-stream-timing 1 (where a call's time goes; see llama_moe_stream::tm_us)
    int32_t     out_check         = 16;    // --moe-stream-out-check N (the output's NaN / Inf check: every prompt call + every Nth reply call; 0 = off)
    int32_t     copy_ahead        = 0;     // --moe-stream-copy-ahead 0|1|2|3 (1: two copy sets when a set costs <= 1/4 of the pool's room, else one; 2: one set; 3: two)
    int32_t     vram_end_fill     = 0; // --moe-stream-vram-end-fill 0|1|2 (the GPU tail refilled at a prompt-end call's end
                                       //   with the prompt-end weights; 2 = also one tick for the per-layer places; 0 = off)
    // The VRAM check of an explicit whole-layer list / count: per GPU device what it must hold besides the whole layers (bytes, from the
    // model load); the VRAM pool's minimum and the growth after the check follow the streamed layers and are added by the engine on the
    // pool's device. Empty = no check (the auto-fit, the plan, no GPU)
    struct wl_dev {
        std::string name;
        int64_t free    = 0;   // free VRAM before the weights load
        int64_t weights = 0;   // the weights that go there and are not routed experts of streamed or whole layers
        int64_t ctx     = 0;   // the KV cache and the recurrent state of the layers there
        int64_t compute = 0;   // the compute buffers' estimate at the physical batch (with the attention mask)
        int64_t prewarm = 0;   // the backend's cuBLAS handle + workspace and its op scratch pool
        int64_t keep    = 0;   // VRAM kept free on it (a --moe-stream-vram-free list; 0 without one)
        bool    pool    = false;   // the VRAM pool's device
    };
    std::vector<wl_dev>  wl_devs;
    std::vector<int32_t> wl_dev_of;   // per block: the index in wl_devs of the device that holds the layer's experts when whole, -1 = not a GPU
    int64_t              wl_ubatch = 512;   // the physical batch (whether GPU-wave graphs, and so the pool's tail, can run)
    bool                 wl_by_device = false;   // the list came from --moe-stream-gpu-layers-device (the check line names it)
    bool                 vram_experts_auto = false; // auto-enabled --moe-stream-vram-experts-file model (silent fallback if absent)
};

// a slot is empty, has a read in flight, or holds the expert it was loaded with
enum llama_moe_stream_slot_state {
    MOE_STREAM_SLOT_EMPTY   = 0,
    MOE_STREAM_SLOT_LOADING = 1,
    MOE_STREAM_SLOT_READY   = 2,
};

struct llama_moe_stream_layer;

// per-layer streaming state, also the userdata of the id-remapping custom op
struct llama_moe_stream_layer {
    llama_moe_stream * mgr = nullptr;

    int32_t  il       = -1;  // model layer index
    int32_t  moe_index= -1;  // index among the streamed MoE layers, -1 for none
    int32_t  pred_il  = -1;  // layer whose experts are prefetched while this layer runs, -1 for none
    // The layer *two* ahead, for a two-layer lead (--moe-stream-prefetch-lead 2): its experts are
    // guessed here, on this layer's router input, and corrected when the next layer runs.
    int32_t  pred_il2 = -1;
    // Lead 3: the layer *three* ahead. The chain for a target X is then: guessed at X-3 from
    // X-3's router input (role 1), refined at X-2 (role 3, no cancels), and corrected at X-1 (role 2,
    // the only step that cancels).
    int32_t  pred_il3 = -1;
    uint32_t n_expert = 0;
    uint32_t n_slots  = 0;

    size_t base_offset = 0;  // sidecar offset of expert 0 of this layer
    size_t block_bytes = 0;  // sidecar bytes per expert of this layer
    bool   repacked   = false;  // the sidecar stores this layer's experts already repacked
    bool   by_proj    = false;  // the experts are read from the model file: each projection has its own place (file_off)
    size_t file_off[3] = { 0, 0, 0 };  // by_proj: file offset of expert 0 of gate, up, down; expert e is at file_off[p] + e*weights[p].size

    std::vector<llama_moe_stream_weight> weights;  // gate, up, down

    // residency, guarded by mgr->mtx
    std::vector<int32_t>  slot_expert;    // [n_slots]  expert id, or -1 when empty
    std::vector<uint8_t>  slot_state;     // [n_slots]  llama_moe_stream_slot_state
    std::vector<uint64_t> slot_last_use;  // [n_slots]  LRU stamp
    std::vector<uint64_t> slot_tailw_gen; // [n_slots]  atomic H2D sequence stamp for page-locked tail-wave copies
    std::vector<uint64_t> slot_ca_gen;    // [n_slots]  atomic H2D sequence stamp for page-locked copy-ahead copies
    // The readiness table ggml's mul_mat_id polls, one flag per slice of the pool tensor
    // (n_slots real slots, then the n_expert_used all-zero slices which are ready from the start).
    // Plain uint32 because it is written with atomic stores from another thread; the accesses are the
    // atomic_ref in moe_slot_ready() below.
    std::vector<uint32_t> slot_ready;
    std::vector<int64_t>  slot_ready_us;  // when the I/O thread last published each slot ready (hand-off timer)
    // --moe-stream-proj-ready 1 (allocated only then): one ready flag per projection (0 gate, 1 up, 2 down) and slice, each
    // registered with its own pool, published (release store) when every byte of that projection has landed; the bytes of each
    // projection still to land for the slot's read in flight; when each flag was published. slot_ready / slot_state stay "all three"
    std::vector<uint32_t> proj_ready[3];
    std::vector<int64_t>  proj_ready_us[3];
    std::vector<int64_t>  proj_left[3];
    int32_t               read_failed = -1;   // -1 while fine; the layer index when a read failed

    // instrumentation. issue_us[slot] is when the read that fills the slot was issued, so
    // the wait can be reported as the full issue->land window and as the part of it the op paid.
    std::vector<int64_t>  issue_us;      // [n_slots]
    int64_t               t_land_us = 0;   // sum over the waited slots of (land - issue)
    int64_t               n_land_samples = 0;
    // The two other policies need one score and one segment per slot. slot_freq is
    // halved every MOE_STREAM_LFU_DECAY accesses (vllm-expert-cache does the same every 64 steps) and
    // slot_seg is 0 = probationary, 1 = protected (only slru uses it).
    std::vector<uint64_t> slot_freq;     // [n_slots]  decayed access count
    std::vector<uint8_t>  slot_seg;      // [n_slots]  0 = probation, 1 = protected (slru)
    uint64_t              freq_epoch   = 0;       // accesses since the last halving
    std::vector<int32_t>  expert_slot;    // [n_expert] pool slot, or -1 when not resident
    // 1 = expert_slot / slot_state / slot_expert / slot_cancel / slot_off changed since gsf_call_start last looked at this
    // layer (set right after every write of them, under mtx); gsf_full_cached = that look's answer (gsf_layer_full)
    uint8_t               gsf_dirty       = 1;
    uint8_t               gsf_full_cached = 0;
    std::vector<uint8_t>  slot_prefetch;  // [n_slots]  the residency came from a prefetch
    // The prefetch came from a two-layer-lead *early* guess, so the correction at the next layer
    // may cancel it while its read has not started (an unstarted read only needs slot_cancel set).
    std::vector<uint8_t>  slot_early;     // [n_slots]  a two-layer-lead early guess is filling it
    std::vector<uint8_t>  slot_covered;   // [n_slots]  a demand hit was already counted
    std::vector<uint8_t>  slot_idle;      // [n_slots]  filled by an idle-time load and not used since (only with --moe-stream-idle-load 1)
    std::vector<uint8_t>  slot_ra;        // [n_slots]  filled by a read-ahead and not used since (only with --moe-stream-read-ahead 1)
    // TEST ONLY (MOE_STREAM_TEST_WINDOW_ENDSTATE=1): the prompt's pe128 weights (every use weighs 2^-(tokens to the prompt's end
    // / 128), over all its calls), what RAM / VRAM held when its last call ended, and the reply's first es_max_words words against them
    std::vector<float>    es_w;           // [n_expert]
    std::vector<uint8_t>  es_ram, es_vram, es_top, es_seen;   // [n_expert] at the prompt's end; es_seen: used by a counted reply word
    int64_t               es_seq = -1;    // the prompt es_w belongs to
    int32_t               es_share = 0, es_words = 0;
    int64_t               es_dump_word = 0;   // TEST ONLY: reply words written to the dump since the prompt's end
    int64_t               es_n_ram = 0, es_n_top_ram = 0, es_n_union = 0, es_n_top = 0;
    int64_t               es_p = 0, es_p_ram = 0, es_p_top = 0, es_p_vram = 0, es_p_hit = 0, es_p_vhit = 0, es_p_lost = 0;
    int64_t               es_d = 0, es_d_ram = 0, es_d_top = 0, es_d_vram = 0;
    // --moe-stream-window 1: this layer in the current window call
    int8_t                win_state = 0;    // 0 not run yet in this call, 1 running (its expert stage), 2 finished
    uint32_t              win_home  = 0;    // its share at rest (the start size n_slots), which it has again at the call's end
    int32_t               win_union = 0;    // the call's distinct experts in this layer not held in VRAM (so far, with pieces)
    int64_t               win_tok   = 0;    // the call's tokens this layer has seen (all at once, or piece by piece)
    std::vector<float>    win_score;        // [n_expert] the keep rule's score of the call's picks (pe128 by default; 0 = not picked)
    std::vector<int32_t>  win_refill;       // experts of its keep set no longer in RAM when it finished (read back after the call)
    int64_t               win_dbg[4] = {};  // TEST ONLY (window log): the end's keep step - ready, K, dropped, not droppable
    ggml_backend_event_t  win_ev = nullptr; // lm_layer_done's event: the GPU has finished with its slots once it completed
    std::vector<uint8_t>  slot_cancel;    // [n_slots]  skip this read, the pool needs the sl
    std::vector<int64_t>  slot_issue_us;  // when this slot's read was issued (ggml clock)ot
    std::vector<const void *> ovl_pools;  // this layer's pools in the CPU backend's overlap registry
    std::vector<uint8_t>  slot_chunk_bad;  // [n_slots]  a chunk of the current read was skipped or failed
    std::vector<int32_t>  slot_pending;   // [n_slots]  chunk jobs of the read that fills it, left
    std::vector<int32_t>  slot_gen;       // [n_slots]  generation of the read that owns the slot
    std::vector<uint8_t>  slot_bl;        // [n_slots]  the slot's read is counted in the backlog (behind rule)
    int32_t               gen_counter = 0;
    std::vector<int64_t>  slot_wave;      // [n_slots]  the wave the resident expert belongs to, -1 for none
    // Policy decay: the decayed decode use count of every expert and the word it was last updated,
    // the word a guess read was issued in (a guess for a layer that has not run in that word is not a victim), and
    // the last decode word this layer's remap ran in
    std::vector<float>    p1_dval;        // [n_expert]
    std::vector<int64_t>  p1_dword;       // [n_expert]
    std::vector<int64_t>  p1_issue_word;  // [n_slots]
    int64_t               p1_remap_word = -1;
    std::vector<float>    p1_pe_w;        // [n_expert] recency-weighted uses in a prompt's last ubatch (scratch)
    uint64_t              p1_pe_seq = 0;  // the call (llama_moe_stream::call_seq) that last computed p1_pe_w
    bool                  rs_restamp = false;  // a prompt-end call (with the wave plan) waits for its re-stamp at the first reply word
    std::vector<int32_t>  rs_slots;            // scratch
    // The 1-ahead guess's top 16 for this layer, made by the layer before in decode word p1_cand_word
    int32_t               p1_cand[16]  = {0};
    int32_t               p1_n_cand    = 0;
    int64_t               p1_cand_word = -1;
    int64_t               use_counter = 0;
    // What the last failed read was, so the error names the layer, the expert, the byte range and
    // how many tries it had
    int     read_errno = 0;      // errno of a failed read, 0 = none
    int32_t read_layer  = -1;
    int32_t read_expert = -1;
    int64_t read_off    = 0;
    int32_t read_len    = 0;
    int32_t read_tries  = 0;

    // scratch of the remap callback
    std::vector<int32_t> uniq;        // distinct experts of the ubatch, first-use order
    // --moe-stream-wave-plan 1: which call's plan this layer runs (the plan itself is the engine's shared wp_tab)
    int32_t              wp_next = -1;   // the wave the stored plan expects next, -1 = none
    int64_t              wp_ntok = 0;    // the call's tokens
    uint64_t             wp_hash = 0;    // the call's ids (a later wave of another call must not read this plan)
    std::vector<int64_t> tok_stamp;   // [n_expert] stamp of the (call, token) that last used it
    int64_t              dup_epoch = 0;      // last stamp handed out
    std::vector<int32_t> wait_slots;
    std::vector<int32_t> wait_experts;  // the expert each waiting slot was for
    int64_t              n_slot_early = 0;  // slots holding an early guess right now (per-layer print)  // slots this call waits for
    std::vector<uint8_t>  keep;       // [n_slots] slots this call is holding, not evictable
    // [keep's size] the wave of this call an early copy is for (expert_ready with keep_wave):
    // its keep[] outlasts every remap op's clear until that wave's op of this layer (or the call's end); -1 = none; empty until first used
    std::vector<int16_t>  ahead_w;
    int32_t               n_ahead = 0;   // marks in ahead_w
    int64_t               wv_w    = 0;   // the wave width of this layer's last remap op (wave_experts without the plan)
    // This layer's guessed experts in the running prompt piece, (expert, count) most-used
    // first - counted at the streamed layer before it from the prompt graph's 1-ahead guess; ended at this layer's wave-0 op
    std::vector<std::pair<int32_t, int32_t>> psplit_guess;
    uint8_t               pin_listed = 0;   // on llama_moe_stream::pin_list (keep[] / slot_wave[] may hold marks)
    // The pool tensor holds cap + n_zero slices. n_slots (above) is the first index of the 8-slice
    // zero block: every switched-on slot is below it, every slot at or above n_slots + n_zero is switched off. Slots below
    // n_slots can be switched off too (slot_off: EMPTY, never taken, pages returned). share = the switched-on slots. With both
    // features off: cap = n_slots, share = n_slots, nothing is ever off (the plain layout).
    uint32_t              cap   = 0;
    uint32_t              share = 0;
    std::vector<uint8_t>  slot_off;       // [cap + n_zero]
    std::vector<int32_t>  el_drop_e;      // experts a give-back dropped (refill candidates; most recent last)
    std::vector<float>    el_drop_v;      // their keep values when dropped
    std::vector<uint8_t>  el_refill;      // [cap + n_zero] the slot's read is a take-back's refill (q_refill, the lowest read priority)
    // A ghost list (the last gh_cap experts evicted in decode words: a needed read of one = one more slot would have saved
    // it) and a victim zone (the zone_n least valuable residents at the last check: a hit on one = one fewer slot would have
    // cost it); both counted per check and averaged over --moe-stream-share-opts half
    bool                  el_low_only = false;   // this prompt call reads into slots below the start size only
    std::vector<uint8_t>  el_zero_known;         // [cap + n_zero] the slot's bytes are all zeros (the low-only setup skips them)
    std::vector<uint8_t>  el_land;               // [cap + n_zero] a read issued into a slot holding no RAM, not landed yet (promised)
    std::vector<uint8_t>  el_res;                // [cap + n_zero] the slot holds RAM (written since its pages were last returned)
    std::vector<uint8_t>  dz;                    // [cap + n_zero] the slot is DISCARDABLE (the OS may take its pages; a use locks it)
    std::vector<std::vector<intptr_t>> dz_saved; // [cap + n_zero] a discardable slot's pages' first words (the cookie took them)
    std::vector<uint8_t>  dz_wr;                 // [cap + n_zero] claimed from the band - its next O_DIRECT read writes its pages first
    bool                  el_pool_mixed = false; // one of the layer's pools is not fresh memory (every slot counted as holding RAM)
    std::vector<int64_t>  gh_seq;         // [n_expert] the eviction number of the expert's last decode eviction (-1 = none)
    int64_t               gh_n     = 0;
    uint32_t              gh_cap   = 8;
    int64_t               gh_raw   = 0, zn_raw = 0;
    int64_t               gh_seen  = 0;   // slot shares' hits guard: ghost hits seen since the share last changed
    double                gh_score = 0.0, zn_score = 0.0;
    std::vector<uint8_t>  zone;           // [n_expert] the victim zone's experts at the last slot shares check
    uint32_t              zone_n   = 0;
    // The VRAM follow protection mark of each slot - a slot whose mark is set is never a victim and never
    // written into. Read only by the VRAM follow version of pick_victim; set only by VRAM follow.
    std::vector<uint8_t>  vf_mark;    // [n_slots]
    // VRAM follow: AI-der's counts per expert of this layer (+1 per pick, prompts and replies; halved by reply words), the
    // experts a decided swap brings in (until its switch), and how many of this layer's slots carry a mark now
    std::vector<uint32_t> vf_recent;  // [n_expert]
    // Diagnostic (MOE_STREAM_TEST_PER_LAYER=1 prints them at exit): this layer's tail waves, the experts they copied,
    // and the experts it held in the VRAM pool at its first prompt gather
    int64_t               tailw_n_waves = 0, tailw_n_copied = 0;
    int32_t               tailw_held_first = -1;
    std::vector<uint32_t> vf_long;    // [n_expert]
    std::vector<uint8_t>  vf_in;      // [n_expert]
    int32_t               vf_n_marks = 0;
    int32_t               vf_n_in    = 0;   // incoming swaps of this layer not switched yet (the fixed mode's cap)
    // --moe-stream-vram-copyback 1: per slot 1 = a copy-back from the VRAM pool in flight (LOADING, source "copy":
    // no I/O thread settles it, the uploader does), 2 = copied back and not picked since (counters); the copy-backs of this layer
    // in flight (they take room as the marks do); per expert the last-use stamp of its held picks (for lru)
    std::vector<uint8_t>  slot_cb;    // [n_slots]
    int32_t               vf_n_cb    = 0;
    std::vector<uint64_t> vf_cb_stamp;   // [n_expert]
    // --moe-stream-plan 1: this layer's experts that have an entry on the waiting list, the decode word in which a
    // list entry bound each slot (-1 = none, to count used / unused), and the decode word this layer's remap last ran in
    std::vector<uint8_t>  pl_listed;      // [n_expert]
    std::vector<int64_t>  pl_bound_word;  // [n_slots]
    std::vector<int32_t>  pl_bound_rank;  // [n_slots]  the rank (0-based) of the entry that bound it, with pl_bound_word
    // --moe-stream-lookahead N > 1 (allocated only then). Per expert of this TARGET layer, this word's guesses for it:
    // the word the marks belong to (older = none), the votes (distances that named it), the latest distance and rank that named
    // it; the latest (= smallest) distance guessed for this layer in this word - an entry is live while its latest distance is
    // that one, dropped otherwise. Per slot: the latest distance of the entry that bound it (with pl_bound_word), and the
    // mid-way mark (a started read the latest guess dropped: its queued pieces wait behind every live entry)
    std::vector<int64_t>  la_word;        // [n_expert]
    std::vector<uint8_t>  la_votes;       // [n_expert]
    std::vector<int8_t>   la_d;           // [n_expert]
    std::vector<int16_t>  la_rank;        // [n_expert]
    int64_t               la_latest_word = -1;
    int32_t               la_latest_d    = 0;
    std::vector<int8_t>   pl_bound_d;     // [n_slots]
    std::vector<uint8_t>  la_back;        // [n_slots] (set only for a read with a queued piece - nothing else can move)
    std::vector<uint8_t>  la_qmark;       // [n_slots] scratch: a slot that may move back / has a queued piece (0 outside)
    int64_t               la_src_far[9] = {0};   // names of the far guesses made AT this layer, by distance (decode)
    // --moe-stream-guess-combined 1: the combined step's stacked inputs for this source layer - its la_targets' ffn_norm
    // gains [n_embd, 1, n] (only with --moe-stream-pred-shared 1) and selection biases [n_expert, n] (null if the model has none),
    // on the device of the targets' routers; la_comb_ok = the step can be built for this layer (else it keeps the separate ops)
    ggml_tensor *         la_gain_stack = nullptr;
    ggml_tensor *         la_bias_stack = nullptr;
    bool                  la_comb_ok    = false;
    int32_t               la_cols = 1;    // the remap op's guess input columns in this call (set by the op; distance = column + 1)
    int64_t               pl_ran_word = -1;
    std::vector<int32_t>  pl_bound_slots; // slots a list entry bound for this layer since its last remap (capacity n_slots)
    // A slot drop stale cancelled (its read will settle EMPTY: a piece was skipped) - a later pick of its expert while
    // it still loads is a miss, not a late guess; and the scratch of drop stale's one pass over q_prefetch
    std::vector<uint8_t>  pl_stale_cx;    // [n_slots]
    std::vector<uint8_t>  pl_cx_cand;     // [n_slots] scratch: a candidate of this remap's drop-stale scan
    std::vector<int64_t>  pl_cx_bytes;    // [n_slots] scratch: its queued bytes
    // --moe-stream-needed-early 1: the decode word in which this layer's remap started (the helper skips a guess whose
    // target has started), whether this layer hands its 1-ahead guess to the helper (decided once at start-up), the postings
    // the helper consumed (the counter rule), and the word / time of the pending one
    int64_t               ne_started_word = -1;
    bool                  ne_post_on      = false;
    uint32_t              ne_consumed     = 0;
    int64_t               ne_post_word    = -1;
    int64_t               ne_post_us      = 0;
    // Set when a GPU-producer one-token graph is built - its guess was built AND marked AND left the remap op, so
    // the remap op (which then gets no guess) marks the layer for the helper; false = no guess of this layer reaches the helper
    bool                  ne_guess_left   = false;
    // Set with it - the GPU-producer graph marked this layer's guess (the GPU writes the slot), whether or not the
    // guess also left the remap op; the remap op then marks the layer for the helper (ne_mark), never ne_post (both would write it)
    bool                  ne_guess_marked = false;
    // The GPU producer may mark this layer's guess - all its targets' routers (la_targets, or pred_il) are on one GPU of the
    // producer's backend (the one-card rule: on the producer's GPU); otherwise its guess keeps the host producer (ne_post) and its split
    // the plain input path (decided once, ne_gpu_setup)
    bool                  ne_gpu_ok        = false;
    // --moe-stream-needed-early 1: the in-graph form - a layer with a guess whose own router runs on the CPU and that
    // does not post to the helper (decided once, ne_set_router_gpu): in a one-token graph its remap op is split into op A
    // (right after the router: the needed reads) and op B (after the shared expert and the guess: the adds and the rest).
    // ig_a_done: op A did the needed part in this call (op B then does only the rest); ig_t_a0: when op A started
    bool                  ne_ingraph      = false;
    bool                  ig_a_done       = false;
    int64_t               ig_t_a0         = 0;
    std::vector<int32_t> pref_ids;    // experts predicted for the next layer
    std::vector<int32_t> pref_cnt;    // [n_expert] how many tokens predicted them
    std::vector<int32_t> pref_pos;    // [n_expert] the best (lowest) position the experts had in a token's guess, valid while pref_cnt > 0
    std::vector<uint8_t>  touched;    // [n_expert]
    std::vector<uint8_t>  was_res;    // [n_expert] resident before the current call

    // --moe-stream-vram-experts: the experts of this layer the VRAM expert pool holds. vx_slot is the expert's
    // slot in its group's pool (absolute, the id the device mul_mat_id uses), -1 = not held; set by vx_setup() and, with VRAM follow
    // (--moe-stream-vram-follow 1), changed only at word boundaries (end_call, set_decode: outside any graph, under mtx); read
    // under mtx and by the device ids op inside a graph. A held expert never gets a RAM slot, a read or a guess.
    std::vector<int32_t>  vx_slot;    // [n_expert], empty when the feature is off
    std::vector<int32_t>  vx_list;    // the first N experts of the list file, in list order
    int32_t               vx_group = -1;     // index into llama_moe_stream::vx_groups, -1 = no VRAM experts
    int32_t               vx_li    = -1;     // this layer's index in its group (its slice of the group's vp_table)
    int32_t               vx_fmt_group = -1; // parent format index across chunked sub-groups
    int32_t               vx_fmt_li    = -1; // this layer's ordinal within its parent format group
    bool                  vp_built = false;   // the last one-token graph built this layer's device pass in its launch
    std::vector<int32_t>  vx_seen;    // remap scratch: the distinct VRAM-held experts of a call
    bool vx_held(int32_t e) const { return !vx_slot.empty() && vx_slot[e] >= 0; }

    // stats, one set per phase: prompt processing and decode, so that ms/token means the same
    // thing on both sides of the phase change (the old single set divided the stall by all
    // tokens while the tool's "ms per token" counts decode tokens only)
    struct counters {
        int64_t n_calls = 0, n_tok = 0, n_distinct = 0, n_hit = 0, n_miss = 0;
        int64_t n_vram_hit = 0;   // expert uses the VRAM expert pool served (neither a RAM hit nor a miss)
        // copy-backs from the VRAM pool started (= the recorder's ISSUE bit7), and picks of a copied-back expert while
        // its copy was still in flight (= truth_res 4; a pick after it settled is an ordinary hit)
        int64_t n_copyback = 0, n_cb_late = 0;
        int64_t n_hit_pref = 0, n_pref = 0, n_pref_waste = 0, n_pref_late = 0, n_reads = 0;
        int64_t t_stall_us = 0;
        // Time picks waited on a copy-back from the VRAM pool in flight (not in t_stall_us, which stays SSD-only),
        // and the old booking of the same waits (SSD + copy-back together, as t_stall_us was before) to check the split against
        int64_t t_cb_wait_us = 0, t_wait_all_us = 0;
        int64_t bytes_read = 0, t_read_us = 0;
        int64_t n_pred_cmp = 0, n_pred_hit = 0;  // the guessed top-8 against the real one, per phase
        int64_t n_chunks = 0;      // chunk jobs issued, n_reads counts logical expert reads
        int64_t t_op_us = 0;      // wall time in the remap op, the stall included
        int64_t n_cancel_wait = 0; // prefetches a demand had to cancel and wait for
        int64_t n_read_retry = 0;  // chunks that failed and were read again
        // A correction's cancels, and the bytes they would have cost, plus the prefetch
        // and demand bytes split, so that "demand misses x 3,133,440 B = demand MiB" can be checked
        // on its own (the MiB/token line mixes both, which is why the two disagreed by 1.5-97x on
        // earlier runs - the ratio was the prefetch traffic).
        int64_t n_pref_cancel = 0;
        int64_t n_pref_cancel_bytes = 0;
        int64_t bytes_prefetch = 0;
        int64_t bytes_demand = 0;
        // The number of reads already in flight every time a needed read was issued,
        // as a sum (the mean is the useful figure) and as 17 buckets of 4 so that the distribution is
        // visible, and the same for the guessed reads alone
        int64_t n_inflight_at_demand = 0;
    // The cost a needed read pays for the guesses that were ahead of it. t_read_us is what the
    // read itself took; these are submit->pickup (did it wait for an I/O thread) and submit->land (what the
    // compute actually waited for), and the excess over the 0.76 ms an idle drive gives.
    int64_t n_demand_submitted    = 0;
    int64_t t_submit_to_pick_us   = 0;
    int64_t n_waited_thread       = 0;   // picked up more than 50 us after being queued
    int64_t t_submit_to_land_us   = 0;
    int64_t n_land_gt_idle        = 0;   // submit->land slower than the idle 0.76 ms
    int64_t t_excess_us           = 0;   // summed (submit->land - 760 us) over those reads
    int64_t n_gahead[4]           = {0, 0, 0, 0};   // needed reads by guessed-reads-ahead 0/1/2/3+
    int64_t t_excess_gahead[4]    = {0, 0, 0, 0};   // their summed excess
    int64_t land_hist[12]         = {0};  // submit->land in 0.5 ms buckets, the last one is >= 5.5 ms
    // Where a read's time goes, split at the three hand-overs the
    // engine controls. pick = submit -> an I/O thread took the job; pread = the read itself; lock = the read
    // finished -> the I/O thread got the engine's lock back (only then can the slot be marked ready). Needed
    // reads (dem) and guess chunks (gc) separately; a skipped (cancelled/stale) job is not counted. Counters only.
    int64_t g12_dem_n = 0, g12_dem_pick_us = 0, g12_dem_pread_us = 0, g12_dem_lock_us = 0;
    int64_t g12_gc_n  = 0, g12_gc_pick_us  = 0, g12_gc_pread_us  = 0, g12_gc_lock_us  = 0;
    int64_t g12_dem_pread_hist[24] = {0};  // 0.1 ms buckets, the last one is >= 2.3 ms
    // The hand-off from "published ready" to "the computation saw it" - the spin wait in mul_mat_id
    // (overlap) and the engine's own condition-variable wait in remap(); count, summed us, max us
    int64_t g12_ho_spin_n = 0, g12_ho_spin_us = 0, g12_ho_spin_max = 0;
    int64_t g12_ho_cv_n   = 0, g12_ho_cv_us   = 0, g12_ho_cv_max   = 0;
    int64_t g12_dem_lock_hist[24]  = {0};  // 0.025 ms buckets, the last one is >= 0.575 ms
        int64_t n_inflight_at_demand_pref = 0;
        int64_t n_inflight_hist[17] = { 0 };
        int64_t n_pref_capped = 0;      // guesses the cap did not let through
        // How far ahead a guess actually was. The sample is taken where it matters: when the
        // target layer's remap asks for that expert, minus the moment the read was issued. Negative
        // cannot happen (a read is issued before the layer that wants it), so the value is the time
        // the read had to finish in, and the bucket histogram gives p10/p50/p90.
        int64_t lead_hist[MOE_STREAM_LEAD_NB] = { 0 };
        int64_t lead_sum_us = 0;         // and the mean, for a sanity check against the histogram
        int64_t lead_max_us = 0;
        // The early-guess op's own cost on the graph thread, and how much of it was waiting for
        // the engine's lock (it is not remap(), so it has to take it)
        int64_t t_pred_us = 0;
        int64_t t_pred_lock_us = 0;
        int64_t n_pred_ops = 0;
        // Why each picked expert of a decode word was not READY when its layer's remap op started - [0] its read was
        // in flight (a guess late), [1] no 1-ahead guess was made for the layer in that word (the first streamed layer), [2] the
        // guess named it but it was neither in RAM nor being read, [3] the guess did not name it - and the layer's lateness
        // (op start -> the land of its last such expert) booked to that expert's class. Read-backs: [1]-[3] experts this
        // pool had evicted <= 8 / <= 64 decode words before, and how many of those evictions a guess read made.
        int64_t n_cause[4]    = {0, 0, 0, 0};
        int64_t t_cause_us[4] = {0, 0, 0, 0};
        int64_t n_rb8 = 0, n_rb8_guess = 0, n_rb64 = 0, n_rb64_guess = 0;
        int64_t n_rb_prompt = 0;   // [1]-[3] experts that the last prompt's processing evicted from this pool
        // --moe-stream-plan 1 (booked to the target layer, decode words): waiting-list entries added, bound (a slot
        // taken and the read queued), removed by reason (MOE_STREAM_PL_*), the add -> bind delay (sum, and 100 us buckets
        // with the last one >= 9.9 ms), the bound ones the target picked (and of them, still in flight at its remap)
        int64_t n_pl_add = 0, n_pl_bind = 0;
        int64_t n_pl_rm[MOE_STREAM_PL_NREASON] = {0};
        int64_t t_pl_delay_us = 0;
        int64_t pl_delay_hist[100] = {0};
        int64_t n_pl_used = 0, n_pl_used_late = 0;
        // Entries bound at their target's pick (the late-guess path; of the named-not-read), bound after
        // their target had run (a read nobody can use in this word), and - drop stale - started guess reads the target did not
        // pick whose unstarted pieces were dropped, with those pieces' bytes
        int64_t n_pl_bind_pick = 0, n_pl_bind_after = 0, n_pl_stale_cancel = 0, n_pl_stale_cancel_bytes = 0;
        // Picks of an expert whose read drop stale had cancelled and that was still loading - counted as misses
        // (n_miss), not as late guesses; each is re-read by the lost-read recovery (a re-issue, ISSUE bit5)
        int64_t n_miss_stale = 0;
        // Needed-early (booked to the producer layer, decode words): guesses posted to the helper, consumed (adds done), skipped
        // because the target had started ("guess too late"), and the post -> adds-done delay (sum; 100 us buckets, the last open)
        int64_t n_ne_post = 0, n_ne_done = 0, n_ne_late = 0, t_ne_delay_us = 0;
        int64_t n_ne_late_d[9] = {0};   // with lookahead, guess columns the helper skipped as too late, by distance
        // --moe-stream-protect-named 1 (decode; booked only when the rule changed the choice): guess victim
        // choices the rule changed (the rule-free victim was a named READY expert), of them with no other victim (the entry
        // skipped for no room), the experts so kept (once per word), and of those the target picked (a read saved) / did not
        int64_t n_pn_refused = 0, n_pn_noroom = 0, n_pn_kept = 0, n_pn_kept_picked = 0, n_pn_kept_unpicked = 0;
        // --moe-stream-wordstart D (booked to the target layer, decode words): the word-start op's list entries - added,
        // bound by a reader / at the target's pick, removed unread by reason (as n_pl_rm); and the target's picks that a word-start
        // guess named in this word, by their state at the target's start: READY, still loading, still on the list, none of these
        int64_t n_ws_add = 0, n_ws_bind = 0, n_ws_bind_pick = 0;
        int64_t n_ws_rm[MOE_STREAM_PL_NREASON] = {0};
        int64_t n_ws_pick[4] = {0, 0, 0, 0};
        int64_t ne_delay_hist[100] = {0};
        // Behind rule (booked to the target layer): list entries skipped while the drive was behind (each once), and what became
        // of them: bound later by a reader, picked by their target (a needed read or a bind at the pick), removed at their
        // target's start (drop stale), other (present / no room at a later turn, word end, a stopped call)
        int64_t n_bh_skip = 0, n_bh_skip_bound = 0, n_bh_skip_picked = 0, n_bh_skip_stale = 0, n_bh_skip_other = 0;
        // --moe-stream-guess-top K (booked to the target layer): entries of the guess's extra ranks (k+1..K) - added,
        // bound by a reader, removed unread by reason, picked by the target while bound this word (used; still loading = late),
        // bound at the target's pick, and their bound reads that drop stale cancelled at the target's start
        int64_t n_gx_add = 0, n_gx_bind = 0, n_gx_used = 0, n_gx_used_late = 0, n_gx_bind_pick = 0, n_gx_stale_cx = 0;
        int64_t n_gx_rm[MOE_STREAM_PL_NREASON] = {0};
        // --moe-stream-lookahead N > 1 (booked to the target layer), per distance d (index 1..8): the guess's names,
        // of them new entries / votes on an existing entry / skipped by the pace / already in the pool (READY / LOADING) or
        // VRAM-held; entries by their latest distance: bound by a reader, bound at the target's pick, picked by the target
        // while bound this word (used; still loading = late)
        int64_t n_la_named[9] = {0}, n_la_new[9] = {0}, n_la_vote[9] = {0}, n_la_pace[9] = {0}, n_la_pool[9] = {0};
        int64_t n_la_bind[9] = {0}, n_la_bind_pick[9] = {0}, n_la_used[9] = {0}, n_la_used_late[9] = {0};
        // dropped entries (named only by farther guesses than the latest): became dropped, bound while dropped (such a
        // read starts moved back), named again, removed unread at their target's start; mid-way: started reads with a queued
        // piece moved back, and of every moved-back read (these + the dropped binds): picked by the target while still loading
        // (promoted), cancelled at the target's start, named again
        int64_t n_dr_became = 0, n_dr_bound = 0, n_dr_again = 0, n_dr_removed = 0;
        int64_t n_mw_back = 0, n_mw_promoted = 0, n_mw_cancelled = 0, n_mw_again = 0;
        // The 1-ahead guess's top-k experts (both paths), and how many of them were READY when it was made
        int64_t n_goal_cand = 0, n_goal_ready = 0;
        // The in-graph form (decode words of one token): op A / op B pairs that split the call, the op A start -> op B
        // start time summed (how much earlier the needed reads were queued than a single op would queue them), and the
        // needed reads op A queued
        int64_t n_ig_calls = 0, t_ig_ab_us = 0, n_ig_reads = 0;
        int64_t t_ig_a_us = 0;   // op A's own run, inside t_ig_ab_us (the rest = the work between A and B)
        // --moe-stream-proj-ready 1 (decode): gate / up projections whose compute started while their expert's down had not
        // landed (from mul_mat_id), experts the matmul waited on whose gate and up landed before their down and the sum of those
        // gaps (down ready - the later of gate / up ready), and the needed reads issued as two requests (one extra request each)
        int64_t n_pr_early = 0, n_pr_gap = 0, t_pr_gap_us = 0, n_pr_split = 0;
    };

    counters pre, dec;           // prompt processing, decode
    bool     decode_phase = false;   // the batch the current call belongs to is a decode step
    // the counters the current call writes to
    // The guessed top-8 of this layer, made by the previous layer's remap op in the same graph,
    // and whether one is waiting to be scored
    std::vector<int32_t> guess;
    bool                guess_valid = false;

    // This decode word's real top-k of THIS layer, stashed by this layer's own remap op so that
    // the op of the last streamed layer can write the whole word out in one piece
    std::vector<int32_t> real_k;

    // --moe-stream-record: bookkeeping of the event recorder, empty unless it is on. "clock" is the
    // decode word in progress + 1 (0 while the prompt is processed).
    std::vector<int32_t> rec_uses;       // [n_slots] demands since the read that filled the slot
    std::vector<int32_t> rec_last;       // [n_slots] clock of the last demand, -1 = never demanded
    std::vector<int32_t> rec_read;       // [n_slots] clock at which the read was issued
    std::vector<int64_t> rec_bytes;      // [n_slots] bytes the read in flight has read so far
    std::vector<int32_t> rec_sv_clock;   // [n_slots] the eviction record the read in flight consumed (a READBACK), -1 = none
    std::vector<uint8_t> rec_sv_by;      // [n_slots]  ... its "evicted by a needed read" flag
    std::vector<int32_t> rec_sv_resid;   // [n_slots]  ... its "resident clocks"
    std::vector<int32_t> rec_ev_clock;   // [n_expert] clock of the expert's last eviction, -1 = none since it was read
    std::vector<uint8_t> rec_ev_by;      // [n_expert] 1 = a needed read evicted it, 0 = a guess
    std::vector<int32_t> rec_ev_resid;   // [n_expert] clocks it had been resident before that eviction

    // Per-word lateness counters (counters only, always on; the words are llama_moe_stream::n_tok_outer, the decode words so far): the word in
    // which the 1-ahead guess for this layer was last made and named each expert, the word of each expert's last eviction
    // from this pool (-1 = never) and whether a guess read took its slot; the not-READY experts of the word's remap op and
    // their classes (counters::n_cause), whose lateness end_call() books once the word's graph is done
    int64_t              b0_guess_word = -1;
    std::vector<int64_t> b0_named_word;   // [n_expert]
    std::vector<int64_t> ws_named_word;   // [n_expert] the word in which a word-start guess named it (wordstart only)
    // --moe-stream-protect-named 1: the word in which a guess read's victim choice passed this expert over because
    // a guess named it (the rule kept it), and those experts in this word (checked at the layer's start: picked or not)
    std::vector<int64_t> pn_kept_word;    // [n_expert]
    std::vector<int32_t> pn_kept;         // capacity n_expert
    std::vector<int64_t> b0_ev_word;      // [n_expert]
    std::vector<uint8_t> b0_ev_guess;     // [n_expert]
    std::vector<int64_t> b0_ev_prompt;    // [n_expert] the prompt (llama_moe_stream::b0_prompt_seq) whose processing
                                          //            evicted it, -1 = its last eviction was in a decode word (or none)
    int64_t              b0_t0 = 0;       // the remap op's start
    int32_t              b0_n  = 0;       // entries of b0_e / b0_cls in use
    std::vector<int32_t> b0_e;            // [n_expert_used]
    std::vector<uint8_t> b0_cls;          // [n_expert_used]
    // A call of this layer left a wait to ggml's mul_mat_id (--moe-stream-overlap), whose time is still in the CPU
    // backend's registry; take_overlap_waits() collects it
    bool                 ovl_pending   = false;
    bool                 in_remap_fill = false;   // active only during remap_needed's fill loop (Belady touched[] valid)

    counters & c() { return decode_phase ? dec : pre; }
    const counters & c() const { return decode_phase ? dec : pre; }

    int64_t n_tok_all = 0;                                  // prompt + decode, for the oracle (the ubatch's rows: ub_rows_of)
    int64_t n_oracle_cmp = 0, n_oracle_miss = 0;             // all tokens
    int64_t n_or_cmp_dec = 0, n_or_miss_dec = 0;             // one token per call, i.e. decode

    // waves: a ubatch that needs more distinct experts than the pool holds is split into waves
    int64_t n_wave_ubatches = 0;     // ubatches that needed more than one wave
    int64_t n_wave_experts  = 0;     // distinct experts those ubatches selected, summed per ubatch
    int64_t n_wave_reads    = 0;     // expert reads they issued, summed per ubatch
    int64_t n_wave_stages   = 0;     // wave stages run, summed
};

// The file layout's alignment: a TokenWing model file's tensors start on it, buffers are allocated on it and read chunks are
// multiples of it
constexpr size_t MOE_STREAM_ALIGN = 4096;

// An expert slice (and so a pool slot) must be a multiple of this, the smallest sector a drive has. The drive's own direct-read
// alignment is read at start; a slice that is not a multiple of it (a 512-multiple on a drive that needs 4096) is read through a copy
constexpr size_t MOE_STREAM_SLICE_ALIGN = 512;

// The pools are allocated 2 MiB-aligned and madvise(MADV_HUGEPAGE)d, so that THP can back
// them with 2 MiB pages. A 3,133,440 B expert slice then spans 2 huge pages instead of ~765 of 4 KiB.
constexpr size_t MOE_STREAM_HUGE_ALIGN = 2*1024*1024;

// Accesses between the halvings of the LFU scores (vllm-expert-cache: 64 steps)
constexpr uint64_t MOE_STREAM_LFU_DECAY = 64;

// what the machine must still have free after the pools are allocated
constexpr int64_t MOE_STREAM_MIN_AVAIL = 4LL*1024*1024*1024;   // the old fixed margin
// With elastic RAM, how long after the fill at start (--moe-stream-idle-load 2) it still gives back only for a swap-out or free RAM under
// the headroom (no early give-back or trim from its rate estimates: the load's own reads and page faults are not another program); the
// first request ends it sooner
constexpr int64_t MOE_STREAM_EL_FILL_SETTLE_US = 10LL*1000*1000;   // 10 s
// What the pools have to leave for everything else in the process that the model file does not
// describe: the KV cache, the compute buffers, the ggml context, and the CUDA driver/runtime host
// allocations - which are made *after* this number is needed, which is why it cannot be measured here.
//
// Calibrated, not estimated: the peak RSS minus the pool bytes of 36 completed Flash runs
// (3.7-21.5 GiB of pools) is 0.57-1.26 GiB, worst case on a
// prefill run at -ub 512. The default is the largest of those plus 0.25 GiB. Raising it costs 8 slots
// per layer per GiB on Flash; lowering it risks an OOM kill.
constexpr int64_t MOE_STREAM_OTHER_RESERVE = (int64_t) (1.51*1024*1024*1024);   // 1.51 GiB, measured
// The part of that reserve the rest of the process really used, measured with llama-server on Ling
// Flash at 16k context (a typical command, 2 snapshots, --cache-disk): RSS - the pools' resident
// part - the weights in RAM - the 2 snapshots, worst over 15 turns and a restore = 1,093 MiB. What is left of the reserve
// (the headroom) holds a server's snapshots and its state-file restore buffer; only what does not fit is added
constexpr int64_t MOE_STREAM_OTHER_REST    = (int64_t) 1093*1024*1024;

// --moe-stream-nonpool-measured 1: one line of <sidecar>.nonpool.txt - what a run with these settings measured outside its pools
struct llama_moe_np_entry {
    bool    found = false;
    int64_t kv = 0, compute = 0, rest = 0, n = 0;
};
llama_moe_np_entry llama_moe_np_read(const char * sidecar, const char * key);   // no file / no such key: found = false

// One drive event, 32 bytes, the record of events.bin
struct moe_rec_event {
    int64_t t_us;
    uint8_t type;      // 1 ISSUE, 2 LAND, 3 EVICT, 4 READBACK
    uint8_t flags;     // bit0 guess, bit1 decode phase, bit2 cancelled, bit3 early, bit4 unused-guess victim, bit5 re-issue, bit6 failed
    int16_t layer;
    int16_t expert;
    int16_t slot;
    int32_t word;      // decode word in progress, -1 in the prompt phase
    int32_t a, b, c;
};
static_assert(sizeof(moe_rec_event) == 32, "events.bin records are 32 bytes");

// event buffers: [0] = the graph thread (ISSUE, EVICT, READBACK, and the list's ADD / DROP / BIND), [1 + ith] = I/O thread ith
// (LAND), and with --moe-stream-needed-early 1 [1 + n_io] = the guess helper (the ADD / DROP / BIND of the guesses it takes, and -
// with --moe-stream-plan-bind-early 1, test only - the EVICT / READBACK / ISSUE of a bind it makes; at the graph thread's cap: those
// events moved there from buffer 0). Reserved at start-up, never grown during the run; a full buffer
// counts its drops (written in the header of events.bin). rec_write merges them by time; at equal times graph -> helper -> I/O.
constexpr size_t MOE_STREAM_REC_EV_CAP_GRAPH = 786432;
constexpr size_t MOE_STREAM_REC_EV_CAP_IO    = 131072;

// The RAM --moe-stream-record needs for up to `words` decode words, counted into the engine's non-pool
// memory (the plan and pool sizing see it): per word and streamed layer the logits of the four
// leads (f32), the two residency bitmaps, the real top-k (id 4 B + weight 4 B + state 1 B), and two int64
// timestamps; per word the word index and t_word; plus the host slots (logits and weights of every layer).
constexpr int32_t MOE_STREAM_RECORD_DEFAULT_WORDS = 1024;

// The graph thread's (and the helper's) event caps grow with --moe-stream-guess-top K: ceil(K / k) times (K <= k: 1).
// And with --moe-stream-lookahead N / --moe-stream-guess-pace B - a target's entries per word are at most its 1-ahead
// guess's width plus B new ones per far guess: ceil((max(K, k) + (N - 1) B) / k)
inline int64_t llama_moe_stream_rec_ev_factor(int64_t guess_top, int64_t k, int64_t lookahead = 1, int64_t pace = 1) {
    const int64_t w = std::max(guess_top, k) + (lookahead > 1 ? (lookahead - 1)*pace : 0);
    return w > k && k > 0 ? (w + k - 1)/k : 1;
}

// The guess helper's slot buffer (--moe-stream-needed-early 1): per model layer N columns of `width` ids + the counter
inline int64_t llama_moe_stream_ne_bytes(int64_t n_layer_all, int64_t lookahead, int64_t width) {
    return n_layer_all*(std::max<int64_t>(lookahead, 1)*width + 1)*(int64_t) sizeof(int32_t);
}

// The wave plan's and the slot check's tables for one multi-wave call of n_tok tokens (ONE shared set, the running layer's):
// the pair-wave table (2 B per pair), the slot check's count (4 B per token), the per-expert arrays (cnt, w_first, n_first: 4 B each;
// pin, wp_res: 1 B each). The RAM plan counts it at the model's training context: a remap call never holds more than a ubatch, and a
// ubatch is at most n_batch <= n_ctx tokens (a -c above the training context grows it past the count by 20 B per token)
inline int64_t llama_moe_stream_wp_bytes(int64_t n_tok, int64_t k, int64_t n_expert) {
    return n_tok*(2*k + 4) + n_expert*(3*4 + 2);
}

// The GPU producer's pinned buffer (needed-early 1 on a GPU): the guess slots (the same layout as ne_buf: n_layer_all slots
// of N columns x width ids + the counter), the call cell (its own 64-byte line), one 64-byte line of router counter per layer, and the
// router slots (the router's ids + its input, each at a 16-byte offset). The engine allocates exactly this (ne_gpu_setup), and the
// RAM plan and the load count it on top of the host-producer slot vector with --moe-stream-needed-early 1 (the GPU producer or not, so the plan,
// which cannot know, equals the load: at most ~0.5 MiB too much when the host producer stays)
// --moe-stream-prompt-split: + one shared router slot for a CPU-wave prompt ubatch of up to ps_rows rows (its ids, input,
// guess and routing weights), 0 = none
inline int64_t llama_moe_stream_ps_cap(int64_t ps_rows, int64_t width, int64_t n_embd, int64_t n_expert_used) {
    return ps_rows <= 0 ? 0 : 2*((n_expert_used*4*ps_rows + 15)/16*16) + (n_embd*4*ps_rows + 15)/16*16 + (width*4*ps_rows + 15)/16*16;
}
inline int64_t llama_moe_stream_ne2_bytes(int64_t n_layer_all, int64_t lookahead, int64_t width, int64_t n_embd, int64_t n_expert_used,
                                          int64_t ps_rows = 0) {
    const int64_t b_guess = llama_moe_stream_ne_bytes(n_layer_all, lookahead, width);
    const int64_t rd_cap  = (n_expert_used*4 + 15)/16*16 + (n_embd*4 + 15)/16*16;
    return (b_guess + 63)/64*64 + 64 + n_layer_all*64 + n_layer_all*rd_cap + llama_moe_stream_ps_cap(ps_rows, width, n_embd, n_expert_used);
}

inline int64_t llama_moe_stream_record_bytes(int64_t n_moe, int64_t n_layer_all, int64_t E, int64_t k, int64_t words,
                                             int64_t n_io = 8, bool helper = false, int64_t guess_top = 0,
                                             int64_t lookahead = 1, int64_t pace = 1) {
    const int64_t per_layer = 4*E*4 + 4*(E/8)*2 + k*(4 + 4 + 1) + 16 + 8;   // + t_snap
    const int64_t per_word  = n_moe*per_layer + 8 + 16;
    // the event buffers at their caps (+ the guess helper's with --moe-stream-needed-early 1) and the
    // per-slot/per-expert bookkeeping of every layer (~1 MiB)
    const int64_t cap_g     = (int64_t) MOE_STREAM_REC_EV_CAP_GRAPH*llama_moe_stream_rec_ev_factor(guess_top, k, lookahead, pace);
    const int64_t events    = (cap_g*(helper ? 2 : 1) + n_io*(int64_t) MOE_STREAM_REC_EV_CAP_IO)*(int64_t) sizeof(moe_rec_event);

    return words*per_word + n_layer_all*(4*E*4 + k*4) + events + (int64_t) 1048576;
}

// the same plan with the paths as strings, filled by the constructor
struct moe_stream_plan_in {
    std::string model_path;
    std::string sidecar;
    std::string arch;
    std::string oracle;
    uint32_t n_layer_all   = 0;
    uint32_t n_layer_nextn = 0;
    uint32_t n_layer_dense = 0;
    int32_t  n_streamed    = 0;
    int32_t  n_gpu_layers  = 0;     // MoE layers kept on the GPU instead of streamed
    std::vector<int32_t> gpu_layers; // which ones
    uint32_t n_expert      = 0;
    uint32_t n_expert_used = 0;
    size_t   block_bytes   = 0;
    uint32_t n_slots       = 0;
    int64_t  ram_budget    = 0;
    int64_t  ram_free      = 0;    // RAM the engine keeps free for everything else
    int64_t  other_bytes   = 0;    // bytes the rest of the process needs
    int64_t  pool_bytes    = 0;
    int64_t  mem_available = 0;
    uint32_t io_threads    = 0;
    bool     prefetch      = true;
    uint32_t prefetch_m    = 8;
    uint32_t max_waves     = 0;   // the cap of the waves, 0 = the bound that is always enough
    uint32_t oracle_lead   = 1;
};

// one expert read
struct llama_moe_stream_job {
    llama_moe_stream_layer * sl = nullptr;
    int32_t slot   = -1;
    int32_t expert = -1;
    bool     prefetch = false;  // a demand read is served before a prefetch
    // The byte range of the expert block this job reads. One read is one job with the whole
    // block (chunk_bytes == 0) or n_chunks jobs that cover it; the slot is ready when the last of
    // them completes. gen is the slot's generation: a job whose generation is stale belongs to a
    // read the pool gave up, and must neither read nor change the slot.
    int64_t  off    = 0;
    int32_t  len    = 0;
    int32_t  gen    = 0;
    // When this read was queued, and how many guessed reads were already in the drive at that
    // moment. Counters only - nothing here changes what is read or when.
    int64_t  t_submit_us = 0;
    uint32_t gahead      = 0;   // min(n_inflight_pref, 3): guessed reads in the drive at submit
    // This job carries the read-back check of its layer. It runs on the I/O thread that just wrote
    // the slot, where the bytes are known to be this expert's and the slot is pinned by the job - which
    // is the only place the check can be made against a slot that really holds what it should.
    bool     verify = false;
    bool     from_demand = false;   // taken from q_demand (a needed read or a promoted guess chunk)
    bool     pr_down     = false;   // the down request of a needed read split as gate+up, then down
    bool     dz_touch    = false;   // the slot was discardable - write one word per page before the read
    bool     idle        = false;   // an idle-time load (q_idle, the lowest priority; kept out of the call counters)
    bool     refill      = false;   // a take-back's refill read (q_refill: after every needed and guessed read, a few at a time)
};

// the userdata of one wave's remapping op: the layer and which wave of the ubatch it loads
struct llama_moe_stream_wave {
    llama_moe_stream_layer * sl = nullptr;
    int32_t w = 0;
    ggml_tensor * pred8 = nullptr;   // this graph's guessed top-8 of layer pred_il, for the counter
    // What this op is for. 0 = this layer's demand ids (and the layer-ahead prediction of the
    // next one, which is what the engine has always done), 1 = the first guess of the target layer,
    // 2 = the correction of that layer's guess from the layer in between, the only step that cancels
    // the early reads it drops, 3 = an intermediate refinement of the chain (lead 3): it adds reads but
    // cancels nothing, because a better step for the same target follows it.
    int32_t role = 0;
    // Which target an early-guess op aims at - 2 = pred_il2 (the layer two ahead),
    // 3 = pred_il3 (three ahead). The chain's later steps come from remap() as they always did.
    int32_t tlead = 2;
    const ggml_tensor * tail_gate = nullptr;   // the tail tensor this wave's gate matmul reads in the graph built last (copy-ahead check)
};

struct llama_moe_stream {
    // the model description that the plan needs, all of it from the GGUF header
    struct model_info {
        std::string arch;
        uint32_t n_layer_all   = 0;
        uint32_t n_layer_nextn = 0;
        uint32_t n_layer_dense = 0;
        uint32_t n_expert      = 0;
        uint32_t n_expert_used = 0;
        uint32_t n_embd        = 0;   // the width of the vector a router is run on
        uint32_t n_ctx_train   = 0;   // the bound the RAM plan counts the wave plan's tables at
    };

    llama_moe_stream(
            const std::string & json_path,    // sidecar metadata (the "<sidecar>.json" of expert_repack.py)
            const std::string & log_path,    // optional per-token miss log
            const std::string & model_path,  // model the sidecar was built from
            const model_info &  mi,
            size_t              size_total,  // the model's bytes, the sum of a split GGUF's shards
            uint32_t            n_slots,     // 0 = derive them from ram_budget
            uint32_t            n_io_threads,// 0 = no I/O threads (plan only)
            bool                prefetch,
            uint32_t            prefetch_m,
            uint32_t            max_waves,  // 0 = the bound that is always enough
            int64_t             ram_budget,  // 0 = no budget
            const std::string & oracle,      // trace prefix for oracle replay, empty = off
            uint32_t            oracle_lead,// lead in MoE layers
            uint32_t            chunk_demand,   // 0 = one read per demand miss
            uint32_t            chunk_prefetch, // 0 = one read per prefetch
            uint32_t            gpu_waves,      // min tokens for GPU matmuls in a wave graph, 0 = off
            uint32_t            gpu_layers,     // keep this many whole MoE layers' experts on the GPU, 0 = off
            const std::string & gpu_layers_list, // explicit comma-separated layer list, "" = none
            bool                gpu_layers_auto,// fit as many whole layers as the free VRAM takes
            int64_t              vram_reserve,  // VRAM free after the model, KV and compute buffers
            const std::string & gpu_layers_mode,// "hot" (the default) or "last", which layers to pick
            int64_t              ram_free,   // RAM to keep free; 0 = the default, <0 = keep no margin
            int64_t              other_bytes,// what the rest of the process needs (non-pool weights)
            int32_t             verify,     // the read-back check: 0 = off, 1 = every layer, N = N
            bool                cpu_repack_in, // keep the pools in the repacked CPU layout
            bool                hugepages,     // 2 MiB-aligned pools + madvise(MADV_HUGEPAGE)
            bool                pin,           // cudaHostRegister the pools (page-locked)
            const std::string & policy,        // "lru", "lfu" or "slru"
            bool                pred_shared,   // predict from the residual + this layer's shared expert
            bool                pred_recall_in,// count recall@8 of the layer-ahead guess
            uint32_t            prefetch_lead_in, // MoE layers of lead, 1 = the engine as it was
            uint32_t            prefetch_m2_in,   // the correction's width, 0 = prefetch_m
            bool                demand_last_in,
            bool                probation_in,
            int32_t             prefetch_inflight_in, // cap on guessed reads in flight
            bool                early_order_in,    // force the early guess to run before the layer
            bool                overlap_in,        // compute the resident experts during the reads
            const std::string & pred_scores_in,    // write the guess's scores to this prefix, "" = off
            const std::string & pred_input_in,     // record the router input to this prefix, "" = off
            int32_t             pred_input_every_in, // 1 word in N, 0 = the default 8
            const std::string & record_dir_in,     // write the side-by-side recording here, "" = off
            int32_t             record_words_in,   // the most decode words it keeps, 0 = the default
            const llama_moe_stream_p1 & p1_in = llama_moe_stream_p1()   // options
    );
    ~llama_moe_stream();

    // the plan, valid after the constructor ran
    const moe_stream_plan_in & plan() const { return mplan; }
          moe_stream_plan_in & plan()       { return mplan; }

    // returns the pool tensor for one projection, creating it on first use
    ggml_tensor * create_pool(int32_t il, const std::string & name, const ggml_tensor * meta);

    // true if the sidecar holds an expert block for this layer
    bool has_layer(int32_t il) const;

    // every streamed layer must have gotten a pool for all three projections, otherwise the
    // remapping op would hand slot ids to an expert tensor that holds expert ids
    void verify_pools() const;

    // What the pools got (the repack has no kernel for some types/shapes, and then it is off)
    void print_repack_state() const;

    // With --moe-stream-needed-early 1, per model layer whether its router weight (ffn_gate_inp) is in a device
    // buffer (llama_model, once the weights are placed). A streamed layer posts its guess to the helper only if its target's
    // router is: the guess is that router's matmul, and it then runs in a GPU split. Prints which layers post.
    void ne_set_router_gpu(const std::vector<uint8_t> & router_gpu);

    // the streaming state of one layer, null when the layer is not streamed
    llama_moe_stream_layer * layer(int32_t il) const {
        return il >= 0 && (size_t) il < layers.size() ? layers[il].get() : nullptr;
    }

    // the remap userdata of wave w of layer il, for the graph
    llama_moe_stream_wave * wave_ctx_of(int32_t il, int32_t w) {
        if (il < 0 || (size_t) il >= layers.size() || !layers[il] || wave_base[il] < 0 ||
            (size_t) (wave_base[il] + w) >= wave_ctx.size()) {
            return nullptr;
        }

        llama_moe_stream_wave & c = wave_ctx[wave_base[il] + w];

        c.sl = layers[il].get();
        c.w  = w;

        return &c;
    }

    // The userdata of the early-guess op (llama_moe_stream_pred2). One context is enough: the op
    // reads it inside its own compute, and the graph's other op for this layer has its own context
    // (wave_ctx_of), so the two cannot overwrite each other.
    // One context per layer, not one for the whole graph: the early guess is a graph node per layer,
    // so a single context left every one of them holding the layer that was built last - all the early
    // guesses went into one pool, the correction found no early slot to cancel, and the vote scratch
    // (pref_cnt / pref_ids) was shared between ops running on different threads.
    llama_moe_stream_wave * pred3_ctx_of(int32_t il) {
        if (il < 0 || (size_t) il >= layers.size() || !layers[il]) {
            return nullptr;
        }
        if (pred3_ctx.size() < layers.size()) {
            pred3_ctx.resize(layers.size());
        }
        pred3_ctx[il].sl     = layers[il].get();
        pred3_ctx[il].w      = 0;
        pred3_ctx[il].role   = 1;   // the chain's first step for the layer three ahead
        pred3_ctx[il].tlead  = 3;
        return &pred3_ctx[il];
    }

    // role 1 when the chain for the layer two ahead has two steps (lead 2), role 3 when it has three
    // (lead 3), because then this layer's guess is the *middle* one and must not cancel anything
    llama_moe_stream_wave * pred2_ctx_of(int32_t il, int32_t role = 1) {
        if (il < 0 || (size_t) il >= layers.size() || !layers[il]) {
            return nullptr;
        }
        if (pred2_ctx.size() < layers.size()) {
            pred2_ctx.resize(layers.size());
        }
        pred2_ctx[il].sl    = layers[il].get();
        pred2_ctx[il].w     = 0;
        pred2_ctx[il].role  = role;
        pred2_ctx[il].tlead = 2;
        return &pred2_ctx[il];
    }

    void print_stats() const;

    // --moe-stream-vram-experts: one VRAM pool per (type, shape) group of streamed layers - Ling has one group -
    // with N slots per layer and n_expert_used all-zero slices at the end (pair p of a token names zero slice p, so a
    // token never names a slice twice, which the CUDA mmq ids path needs). gate and up share one tensor
    // [n_embd, 2*n_ff, n] (the merged pool: expert s is its gate in rows [0, n_ff) and its up in [n_ff, 2*n_ff)).
    struct vx_group_t {
        ggml_type     t_gate = GGML_TYPE_COUNT, t_down = GGML_TYPE_COUNT;
        int64_t       ne_gate[2] = {0, 0}, ne_down[2] = {0, 0};
        int32_t       fmt_group = 0;          // parent format group index across chunked sub-groups
        int32_t       n_real    = 0;          // sum of held places across layers of this sub-group; zero slices are [z_end(), z_end() + n_zero_g)
        // Copy-ahead, one zero block: the tail group's places are [tail: n_tail | held: n_real | zero slices] -
        // the zero slices at the tensor's end for every reader (replies, the id tables, the CUDA zero table, the prompt zero skip); a graph
        // whose waves run through the (emptied) tail reads the views [p0, end), which start after the tail. Other groups: [held | zero]
        int32_t       n_tail    = 0;
        int32_t       p0        = 0;          // the first held place (n_tail in the tail group, else 0)
        std::vector<int32_t> layers;
        ggml_tensor * gate_up = nullptr;
        ggml_tensor * down    = nullptr;
        // Copy-ahead: the device pass's views [p0, p0 + n_real + n_zero) - the held places and the zero slices, not the tail (a
        // MUL_MAT_ID launches work for every expert of its weight tensor: over the tail's places it cost the prompt pass 2.7x)
        ggml_tensor * vp_gate_up = nullptr;
        ggml_tensor * vp_down    = nullptr;
        int32_t       n_zero_g  = 0;
        int32_t       z_end()    const { return p0 + n_real; }        // the first zero slice (in the whole tensor)
        int32_t       h_end()    const { return p0 + n_real; }        // the end of the held places (= the zero slices' start)
        int32_t       n_places() const { return z_end() + n_zero_g; } // the whole tensor's places
        // --moe-stream-vram-parallel: [n_expert_used, n_expert, layers] int32 - row e of a layer = the expert's place
        // k times when held, else the zero slices n_real + 0 .. n_real + k - 1 (so no token names a slice twice); the graph
        // takes get_rows(row of each picked expert) and the diagonal = the device ids, all on the GPU
        ggml_tensor * vp_table = nullptr;
        ggml_context * ctx    = nullptr;
        ggml_backend_buffer_t buf = nullptr;
    };
    std::vector<vx_group_t> vx_groups;
    std::map<int32_t, std::vector<int32_t>> vx_file_layers;   // the list file, by model layer
    std::map<int32_t, std::vector<int64_t>> vx_file_scores;   // optional per-expert priority scores from the list file
    std::map<int32_t, size_t>               vx_file_len;      // the file's part of a layer's list when the rest was appended in expert-id order (absent = all of it)
    bool                                    vx_non_uniform = false;
    bool        vx_ready     = false;   // the pool is allocated (and filled): the graphs carry the device pass
    bool     (* vx_zero_unreg)(const void * pool) = nullptr;   // --moe-stream-gpu-skip-zero: the CUDA table's unregister
    bool        vx_test_empty = false;  // MOE_STREAM_VRAM_TEST_EMPTY=1: pool and device pass, but no expert held
    int64_t     vx_bytes     = 0;
    double      vx_fill_s    = 0.0;
    // A snapshot of this counter goes into llm_graph_params, so a graph built before the set of layers with a
    // device pass changed is not reused after it (vx_setup bumps it)
    uint32_t    graph_gen    = 0;
    // VRAM follow: the moving set. A decided swap is a job: at word boundary 1 (a tick, end_call) the leaving expert stops
    // being held and its place is reserved for the incoming one, whose RAM slot gets the mark; at word boundary 2 (a later call's
    // start, set_decode) the incoming expert is switched in once its copy has landed: its RAM slot found through expert_slot[e] at
    // that moment is released and the mark cleared. The fixed mode (test only) makes every switch at the next call start and copies
    // waiting for the uploader's copy (or copying itself when the source was not READY at the tick); the real mode switches only
    // copies that have landed. The copies of a READY source run on the uploader thread (its own backend instance = its own stream).
    struct vf_job_t {
        int32_t il = -1, e = -1;         // the incoming expert
        int32_t leave_il = -1, leave_e = -1;   // the leaving one (OLD_ORDER keeps it held until the switch)
        int32_t group = -1, place = -1;
        int32_t src = -1;                // the RAM slot the copy reads (marked; -1 = none: the fixed mode reads at the switch)
        bool    marked = false;
        bool    queued = false;          // the uploader copies it (a READY source at the tick)
        int32_t delay = 0;               // MOE_STREAM_TEST_VF_DELAY_SWITCH: call starts left before the switch may happen
        std::atomic<bool> done{false};   // the copy has landed in the place
        // The leaving expert's copy-back into its RAM pool, done BEFORE the incoming copy writes the place
        int32_t cb_slot = -1;            // the slot it takes in layer leave_il (-1 = no copy-back: off, no room, or value too low)
        int32_t cb_gen  = 0;             // that slot's generation when it was taken
        int64_t cb_t0   = 0;             // when it was taken (the ISSUE event's time)
        int64_t cb_seq  = 0;             // the call count when it was taken (test only CB_HOLD_US)
        std::atomic<bool> cb_done{true}; // the copy-back has settled (true when there is none)
    };
    std::vector<std::unique_ptr<vf_job_t>> vf_jobs;   // decided, not switched yet (mtx)
    std::vector<std::vector<int32_t>> vf_place_il, vf_place_e;   // [group][place]: the owner, -1 = reserved for a job
    // the uploader (real mode)
    std::thread                vf_thread;
    mutable std::mutex         vf_mtx;
    std::condition_variable    vf_cv, vf_cv_done;
    std::deque<vf_job_t *>     vf_queue;
    std::deque<vf_job_t *>     vf_cb_queue;           // the jobs whose copy-back is due - the uploader takes these first
    bool                       vf_stop = false;
    int32_t                    vf_busy = 0;           // jobs taken by the uploader and not finished (vf_mtx)
    ggml_backend_t             vf_be = nullptr;       // a second backend instance of the GPU: the uploader's own stream
    ggml_backend_buffer_t      vf_bounce_buf = nullptr;
    uint8_t *                  vf_bounce = nullptr;   // one expert block, pinned (--moe-stream-vram-follow-pinned 1)
    static constexpr int       VAC_D2H_HALF = 8;      // double-buffered pipelined vacate D2H (8 experts/half = 16 total)
    ggml_backend_buffer_t      vac_d2h_buf = nullptr;
    uint8_t *                  vac_d2h_ptr = nullptr;
    size_t                     vac_d2h_cap = 0;
    ggml_backend_event_t       vac_d2h_ev[2] = {};
    std::atomic<bool>          vf_in_call{false};     // a call is between set_decode and end_call (the idle line)
    // The tick's wait on the GPU, not on the host. At a tick the engine records vf_ev on the calling context's
    // GPU stream (vf_ctx_be, set by llama-context every call) after the word's graph; every copy INTO a place waits for it (the
    // uploader's stream with ggml_backend_event_wait, a synchronous copy with ggml_backend_event_synchronize), so no thread waits
    // at the tick. Chosen once at start-up (vf_ev_on); with more than one context on the model (n_ctx_shared > 0) or the synchronous
    // test paths the tick keeps the device-wide wait. MOE_STREAM_TEST_VF_DEVICE_WAIT=1 (test only) keeps it always.
    bool                       vf_ev_on = false;
    ggml_backend_t             vf_ctx_be = nullptr;   // the calling context's GPU backend (its compute stream)
    ggml_backend_event_t       vf_ev = nullptr;
    bool                       vf_ev_tick = false;    // this tick recorded vf_ev: its copies are handed over after the record
    bool                       vf_ev_multi_said = false;
    int64_t                    vf_ev_ticks = 0;       // ticks that waited on the GPU (vf_gpu_syncs counts the device-wide ones)
    // TEST ONLY: MOE_STREAM_TEST_VF_EARLY_EVENT=1 records vf_ev at the tick word's start instead of at its end (the
    // failure test); MOE_STREAM_TEST_VF_PLACE_CHECK=N queues, at each event tick, N us of GPU delay on the compute stream, then a read of
    // the first freed place, before the record - the next call start compares it with the leaving expert's bytes and stops the
    // run on a difference; MOE_STREAM_TEST_VF_TICK_TIMERS=1 prints one line per reply word (end of call, scheduler waits)
    int32_t                    vf_test_early_ev = 0, vf_test_pc_us = 0, vf_test_tick_timers = 0;
    void (*vf_test_spin)(ggml_backend_t, int64_t) = nullptr;   // ggml-cuda's test delay kernel (NVIDIA builds)
    ggml_backend_buffer_t      vf_pc_buf = nullptr;   // pinned: the place as the GPU read it
    ggml_backend_event_t       vf_pc_ev  = nullptr;   // recorded right after that read (its own: the next call may be another context)
    std::vector<uint8_t>       vf_pc_want;            // the leaving expert's bytes (from the sidecar)
    bool                       vf_pc_pending = false;
    int32_t                    vf_pc_il = -1, vf_pc_e = -1, vf_pc_place = -1, vf_pc_group = -1;
    int64_t                    vf_pc_checks = 0;
    int64_t                    vf_tt_sync_us = 0;     // TICK_TIMERS: the scheduler's waits since the last reply word's line
    void vf_place_check_queue(const std::vector<vf_job_t *> & made);   // mtx held, at an event tick, before the record
    void vf_place_check_done();                                        // no lock: a queued check's result
    // the two policies, chosen once at start-up (the word path never tests the fixed flag)
    void (llama_moe_stream::*vf_word_end_fn)()            = nullptr;   // end_call, mtx held: decay + tick (boundary 1)
    void (llama_moe_stream::*vf_call_start_fn)(bool decode, int32_t n_tok) = nullptr; // set_decode: the switches (boundary 2)
    template <bool FIXED> void vf_word_end_t();
    template <bool FIXED> void vf_call_start_t(bool decode, int32_t n_tok);
    template <bool FIXED> void vf_tick();
    template <bool FIXED> bool vf_switch(std::unique_lock<std::mutex> & lock, vf_job_t & j, bool force);
    void vf_setup(ggml_backend_dev_t dev);                      // after vx_setup: owners, counts, the uploader
    void vf_copy(vf_job_t & j, const uint8_t * gate, const uint8_t * up, const uint8_t * down, bool async);
    void vf_copy_sync(std::unique_lock<std::mutex> & lock, vf_job_t & j);   // the fixed mode / SYNC_COPY / OLD_ORDER
    void vf_verify(const vf_job_t & j);
    // --moe-stream-vram-copyback 1: take the leaving expert's RAM slot (mtx held, the tick), copy the place's bytes
    // GPU -> (pinned bounce ->) the slot (no engine lock: the slot is LOADING and only the copy-back writes it), settle it READY
    // (mtx held: the I/O threads' settle), and the keep value an expert leaving VRAM has
    void vf_cb_take(vf_job_t & j);
    void cb_claim(llama_moe_stream_layer & B, int32_t v, int32_t e, vf_job_t & j);
    bool tail_claim_cb_slot(int32_t g, int32_t p, int32_t il, int32_t e, vf_job_t & out_j);
    bool tail_copy_back(int32_t g, int32_t p, int32_t il, int32_t e);
    void vf_cb_copy(vf_job_t & j, bool async);
    void vf_cb_settle(vf_job_t & j);
    float vf_keep_value(const llama_moe_stream_layer & sl, int32_t e) const;
    void vf_loop();
    void vf_stop_uploader();
    // test only, read once at start-up
    bool    vf_no_tick = false, vf_ignore_mark = false, vf_sync_copy = false, vf_old_order = false, vf_verify_on = false;
    int32_t vf_delay_switch = 0;
    int32_t vf_verify_level = 0;   // MOE_STREAM_TEST_VF_VERIFY=N: 1 = every switched place; 2 = also 16 held places per call start
    int64_t vf_vcheck_pos = 0, vf_verified_held = 0;
    // This call is a one-token decode call (a reply word: the counts' clock and the ticks); the device
    // waits before a tick, outside the engine lock, and their time; calls that needed the room the marks take (drained first)
    bool    vf_call_one = false;
    bool    vf_word_ctx = true;   // llama-context says whether this call's context counts reply words (an MTP
                                  // draft context does not: its one-token drafts would otherwise drive the ticks)
    bool    vf_old_waves = false;   // MOE_STREAM_TEST_VF_OLD_WAVES (test only): the old n_waves()
    int64_t vf_gpu_wait_us = 0, vf_drains = 0;
    // The pool's TAIL: the last tail_n places of the VRAM pool tensor are
    // ordinary places at word time; while a GPU-wave graph runs they are the scheduler's pool copies' memory (a preset target, the
    // compute buffer no longer holds them). tail_ubatch (llama_context, before every ubatch) vacates them first - no expert held there,
    // its id-table row on the zero slices - and a call that is not a GPU-wave call fills the free places again (vf_fill_t).
    bool     tail_allowed = false;   // llama_context: no pipeline parallelism, op offload on, the pool's GPU takes every GPU prompt wave (set before vx_setup)
    int32_t  n_gpu_ctx    = 0;       // llama_context: the GPU backends of the context (set before vx_setup)
    bool     tail_test_mg = false;   // MOE_STREAM_TEST_MULTI_GPU_NO_TAIL=1 with several GPUs (test only): no tail
    bool     tail_on      = false;   // the pool has a tail and the scheduler hook may use it (decided in vx_setup)
    bool     tail_vacated = false;   // no expert is held in the tail's first tail_vac_n places and no job targets them (mtx)
    int32_t  tail_vac_n   = 0;       // the places emptied from tail_first (the copy places, + what a layer-major call's tensors
                                     // need); 0 = none, >= tail_copy_n while tail_vacated (mtx)
    bool     tail_preset  = false;   // the scheduler hook gives GPU-wave pool copies a target in the tail (MOE_STREAM_TEST_PRESET_COPY=0: no)
    bool     tail_refill_due = false; // a vacate happened: the next call that is not a GPU-wave call fills the free places once
    // The tail is on although a one-wave ubatch of at least the offload minimum rows
    // exists (RAM slots >= 8 x GGML_OP_OFFLOAD_MIN_BATCH, default 32). Such a ubatch keeps its expert matmuls on the CPU in every call
    // (the graph flags its ids op as a wave, waves_on_gpu and el_gpu_view say no), so no whole-pool copy the reserve does not cover is
    // made. The minimum is read as at least 1 (ggml offloads every op at 0); at 1 a decode word is such a ubatch too
    bool     tail_onewave_host = false;
    int64_t  tail_onewave_max  = 0;   // the largest such ubatch (vx_setup; the print, el_call_low)
    int64_t  n_onewave_host    = 0;   // graphs built with such a ubatch (llama_context; the exit line)
    static int64_t offload_min1() { return std::max<int64_t>(1, offload_min_batch()); }
    bool     el_win_no_whole_copy(int64_t u) const {
#ifdef _WIN32
        return el_win_holes && !tailw_graph(u);
#else
        (void) u;
        return false;
#endif
    }
    bool     onewave_host(int64_t u) const { return (tail_onewave_host || el_win_no_whole_copy(u)) && u >= offload_min1() && n_waves(u) == 1 &&
                                                    !(ca_one_wave && tailw_graph(u) && gpu_waves > 0 && u >= (int64_t) gpu_waves); }   // (those take tail waves)
    bool tail_hook_ready() const {
        return tail_preset && (size_t) tail_group < vx_groups.size() &&
               vx_groups[(size_t) tail_group].gate_up != nullptr && vx_groups[(size_t) tail_group].down != nullptr;
    }
    bool     tail_real    = false;   // llama_context: the graph being split will run (not a reserve): the tail must be vacated now
    int32_t  tail_n = 0, tail_first = 0, tail_group = 0;   // places, the first one, its pool group
    // The tail grows by lm_tail_ext places for a layer-major call's per-call tensors (llama_context asks for
    // lm_tail_req bytes before vx_setup). The wave copies use the first tail_copy_n places (gate then up, back to back, then the down
    // tensor's own); the rest of the gate|up tensor's tail is the region lm_tail_region hands out while the tail is vacated
    int64_t  lm_tail_req = 0;
    int32_t  lm_tail_ext = 0, tail_copy_n = 0;
    bool     lm_tail_region(int64_t bytes, ggml_backend_buffer_t * buf, char ** addr);
    int32_t  lm_tail_need(int64_t bytes) const;   // the extension places a call's tensors of `bytes` take (at most lm_tail_ext)
    // No extension is reserved at start for a call of n_batch words (-c with layer-major: ~1 GB on Flint at 65k, so the 6 GB
    // case could not start with the pool); a call's tensors take, for that call, expert places of the pool - as many as the real prompt
    // needs, each tensor in one pool group (the last places of its expert range; the tail group's end before the tail); their experts step
    // aside as the tail's do, and VRAM follow's fill fills them again (vf_fill_t looks at every group's lent places)
    std::vector<int32_t> lm_lend_g, lm_lend_max_g;   // mtx, per pool group: places [end - n, end) vacated now / the most ever lent
    int64_t  n_lm_lend = 0;
    // (copy-ahead: the end of every group's held places - the zero slices follow, so the lent places never take one)
    int32_t  lm_group_end(size_t g) const { return vx_groups[g].h_end(); }
    // llama_context, after tail_ubatch: VRAM places for a layer-major call's tensors of these sizes (one buffer + address each); false =
    // not every one fits (the caller's own buffer then)
    bool     lm_lend_place(const std::vector<int64_t> & sizes, ggml_backend_buffer_t * bufs, char ** addrs);
    // the most tokens a layer-major call's tensors (per-token byte sizes in bytes_per_tok) can take in the pool's places (0 = no lending)
    int64_t  lm_lend_max_tokens(const std::vector<int64_t> & bytes_per_tok, int64_t kv0 = 0) const;   // (kv0: the call's first position)
    int64_t  lend_call_kv = 0;   // llama_context: the context (KV cells) the running call reaches (the lent room for it)
    bool     lm_lend_on() const { return vx_ready && !vx_groups.empty() && vf_fill_fn != nullptr && vf_drain_fn != nullptr && lm_tail_ext <= 0; }
    // VRAM lending for prompts: a prompt call's GPU compute buffer is planned by ggml's allocator as
    // always - its layout is ggml's own, so it is safe for any fusion - and only its memory is lent places of the VRAM pool: per group
    // the FRONT of the held places [p0, p0 + U), as two pieces (the gate / up part and the down part), each a buffer over the pool's
    // memory that llama_context sets on the scheduler (ggml_backend_sched_set_fixed_chunks). The call's own tensors stay at the end of
    // the held places, so the two never meet (its room = the held places less U at the call's context). The reply graphs keep a small
    // real compute buffer. Lent from a prompt call's start to the next call's start; the fills (the end fill, the first word's refill)
    // refill the places, which no graph uses then
    // possible: the pool is sized from free VRAM (a pinned --moe-stream-vram-experts N gains nothing from a smaller compute buffer) and
    // VRAM follow refills the places
    bool     lend_possible() const { return p1.vram_lend_prompt && !lend_test_off && p1.vram_follow && p1.vram_free > 0 && !vx_groups.empty() &&
                                           vx_groups[0].buf && vx_groups[0].gate_up && vx_dev != nullptr && lm_tail_ext <= 0; }
    bool     lend_test_off = false;          // MOE_STREAM_TEST_LEND_OFF=1 (test only): the plain way
    const void * lend_owner = nullptr;       // the one context that lends (a second context on the model keeps its compute buffer)
    bool     lend_on = false;                // llama_context's last reserve: this context lends (the plan fits the pool's pieces)
    // the prompt graphs' GPU plan measured at start on a grid (C): lend_grid[i][j] bytes at lend_kv_pts[i] KV cells and lend_t_pts[j] call
    // tokens (both ascending; the last call size = n_ubatch, its column also holds the layer-major graphs). A call's need = the next
    // measured point at or above it (the plan only grows with the tokens and the context: a safe upper bound), + 2 % + 2 MiB, x slack
    std::vector<int64_t> lend_kv_pts, lend_t_pts;
    std::vector<std::vector<int64_t>> lend_grid;
    double   lend_slack = 1.0;               // the pieces' bytes over the plan's (fragmentation over the pieces)
    double   lend_low   = 1.0;               // MOE_STREAM_TEST_LEND_LOW=f (test only): the estimate x f (the retry's check)
    int64_t  lend_real  = 0;                 // the real compute buffer: the reply graphs, a short call at -c, what stays in it at -c
    int64_t  lend_need(int64_t T, int64_t kv) const {
        if (!lend_on || lend_grid.empty()) { return 0; }
        size_t i = 0, j = 0;
        while (i + 1 < lend_kv_pts.size() && lend_kv_pts[i] < kv) { i++; }
        while (j + 1 < lend_t_pts.size()  && lend_t_pts[j]  < T)  { j++; }
        const int64_t v = lend_grid[i][j];
        return (int64_t) (((double) v*1.02 + 2.0*1048576.0)*lend_slack*lend_low);
    }
    // per group, the front places that hold `bytes` in pieces: the groups with the most held bytes first (the fewest, largest pieces),
    // each up to `cap_left[g]` places (empty = all its held places); when they cannot hold it all, as much as they can (partial: the
    // rest of the plan stays in the real buffer)
    std::vector<int32_t> lend_places_for(int64_t bytes, const std::vector<int32_t> * cap_left = nullptr) const;
    std::vector<int32_t> lend_places_at(int64_t kv) const { return lend_on ? lend_places_for(lend_need(INT32_MAX, kv)) : std::vector<int32_t>(); }
    // the pieces over the front places U (no vacate: the reserve's trial plan), or lent for a call (lend_lend: the experts there step
    // aside, VRAM follow drained first); false = not possible (then nothing changed)
    bool     lend_pieces(const std::vector<int32_t> & U, std::vector<ggml_backend_buffer_t> & out) const;
    // (shift = false: a retry for a graph already built - its views keep their start; the new places are inside them, empty)
    bool     lend_lend(const std::vector<int32_t> & U, std::vector<ggml_backend_buffer_t> & out, bool shift = true);
    void     lend_free_pieces(std::vector<ggml_backend_buffer_t> & bufs) const;
    ggml_backend_buffer_t (* lend_from_ptr)(ggml_backend_buffer_type_t, void *, size_t) = nullptr;   // ggml-cuda's buffer over pool memory
    std::vector<int32_t> lend_lent_g;        // per group: the front places vacated for VRAM lending since the last fill (the fills scan them)
    int64_t  n_lend_lends = 0, n_lend_calls = 0, n_lend_fallback = 0, n_lend_grow = 0, n_lend_stepped = 0, lend_lent_max = 0;
    void     lend_vacate(size_t g, int32_t p_lo, int32_t p_hi, int32_t * n_stepped);   // mtx held, VRAM follow drained
    int      vx_empty_layers(int * n_streamed, int * lo, int * hi) const;   // diagnostic (mtx held)
    bool     lm_expert_graph = false;   // llama_context: the graph being built and split is a layer-major expert graph
    int64_t  tail_cap[3] = { 0, 0, 0 };                   // bytes of the target of a gate / up / down copy
    int64_t  tail_ub = 0;                                  // the largest ubatch: min(n_ctx, n_ubatch)
    std::unordered_map<const ggml_tensor *, int8_t>  tail_proj;   // a layer's pool tensor -> 0 gate / 1 up / 2 down
    // Per pool group (format), where its GPU-wave copies go: byte offsets of gate / up / down from the start of the tail
    // group's gate|up tail (gate, up, and down when tail_dn_r1[g]: a down slice too big for the down tail) or its down tail; the copies
    // are tail_tc slices wide (the RAM pool view; one group: tail_copy_n)
    std::unordered_map<const ggml_tensor *, int32_t> tail_pgroup;  // a layer's pool tensor -> its pool group
    std::vector<std::array<int64_t, 3>> tail_off_g;
    std::vector<uint8_t>                tail_dn_r1;
    // The output check (llama_context, --moe-stream-out-check N): every prompt call and every Nth reply call; MOE_STREAM_TEST_OUT_NAN=1
    // (test only) fails every checked call as if its logits were NaN (the failure path)
    // The rows of the running ubatch (the first streamed layer's op sets it): a layer cut to the output rows - the last layer of a
    // prompt call, where the graph keeps only the rows that give logits - still advances its token count by the ubatch's rows
    int64_t  ub_rows = 0;
    int64_t  ub_rows_of(const llama_moe_stream_layer & L, int64_t n_tok, bool note = true) {
        if (L.moe_index == 0) {
            if (note) {
                ub_rows = n_tok;
            }
            return n_tok;
        }
        return std::max(n_tok, ub_rows);
    }
    // The output check (--moe-stream-out-check N): reply calls counted, calls failed; per kind ([0] prompt calls, [1] reply calls) the
    // calls checked and the wall time inside the checks (the copy, the wait for the device - the rest of the call's GPU work - the scan)
    int64_t  out_calls = 0, out_failed = 0, out_checked[2] = { 0, 0 }, out_us[2] = { 0, 0 };
    bool     out_test_nan = getenv("MOE_STREAM_TEST_OUT_NAN") && atoi(getenv("MOE_STREAM_TEST_OUT_NAN")) != 0;
    bool     out_check_due(bool prompt) {
        if (p1.out_check <= 0) {
            return false;
        }
        out_calls += prompt ? 0 : 1;
        return prompt || out_calls % p1.out_check == 0;
    }
    bool     vx_test_pad_check = getenv("MOE_STREAM_TEST_PAD_CHECK") && atoi(getenv("MOE_STREAM_TEST_PAD_CHECK")) != 0;   // (test only)
    void     vx_pad_check(const char * where);   // MOE_STREAM_TEST_PAD_CHECK: the held places' row padding (mtx held)
    // The held places' row padding. A quantized matmul whose rows need padding (CUDA / ROCm: to 512 values) reads the first vx_pad[g][k]
    // bytes of the place after the one it computes (k: 0 = the gate|up tensor, 1 = down). Those bytes must be zeros or blocks of the
    // group's own format whenever a held place before them can be read (a NaN scale of other bytes there makes the result NaN). Places
    // whose memory has other uses for a while (the tail's copies, places lent to a prompt call's buffers or tensors) are marked in
    // vx_dirty when that use starts; no such use writes the first bytes of the first place of its range (whose predecessor may be held);
    // the fills that make places held again first write zeros into every marked free place (vx_pad_clean); a switched-in expert clears
    // its place's mark. Groups whose rows need no padding have no marks and take none of these steps
    std::vector<std::array<int64_t, 2>> vx_pad;      // [group]: the row padding of gate|up / down (0 = none)
    std::vector<std::vector<uint8_t>>   vx_dirty;    // [group][place] (only for groups with padding)
    int64_t  vx_n_dirty = 0;                         // marked places in all groups
    int64_t  vx_pad_cleans = 0, vx_pad_zeroed = 0, vx_pad_writes = 0, vx_pad_us = 0, vx_pad_targeted = 0;   // the exit line
    bool     vx_test_no_place_pad = getenv("MOE_STREAM_TEST_NO_PLACE_PAD") && atoi(getenv("MOE_STREAM_TEST_NO_PLACE_PAD")) != 0;   // (test only: off)
    bool     vx_test_pad_plant    = getenv("MOE_STREAM_TEST_PAD_PLANT") && atoi(getenv("MOE_STREAM_TEST_PAD_PLANT")) != 0;     // (test only)
    int64_t  vx_pad_planted = 0;
    // the bytes a use of place memory leaves alone at the start of its range's first place (0 for a group without padding, or off in a test)
    int64_t  vx_pad_skip(size_t g, int k) const { return !vx_test_no_place_pad && g < vx_pad.size() ? vx_pad[g][(size_t) k] : 0; }
    void     vx_pad_init();                                         // vf_setup: the padding per group, the marks
    void     vx_pad_mark(size_t g, int32_t p_lo, int32_t p_hi, int32_t p_kept, int k_only = -1, bool plant = true);   // mtx held: places [p_lo, p_hi) get another use
    void     vx_pad_clean();                                        // mtx held, a fill: zeros into every marked free place
    std::vector<std::array<int64_t, 3>> tail_pad_g;   // per group: the row padding (bytes read past a copy's end) of gate / up / down
    int64_t                             tail_tc = 0;
    std::vector<uint8_t>                tailw_read;   // scratch [tail_tc + 1]: the places a wave's ids name
    std::vector<uint8_t>                tailw_zero_src;   // zeros for the row padding writes
    bool     tailw_test_no_pad = getenv("MOE_STREAM_TEST_NO_ROW_PAD") && atoi(getenv("MOE_STREAM_TEST_NO_ROW_PAD")) != 0;   // (test only: the writes off)
    int64_t                             tailw_pads = 0, tailw_pad_bytes = 0;   // row-padding writes of the gathers (places' first bytes, copies' ends)
    std::unordered_map<const ggml_tensor *, int32_t> tail_uses;   // this graph split: consumers of a pool-derived source
    std::unordered_map<const ggml_tensor *, int32_t> tail_mm_uses;   // of them, MUL_MAT_IDs that read it as their weights
    int32_t  tail_split_used[3] = { -1, -1, -1 };         // the split that last took projection k's target
    int64_t  tail_n_vac_last = 0, tail_n_vac_max = 0;   // places emptied (the last vacate's range, the largest)
    int64_t  tail_n_cb = 0, tail_cb_us = 0, tail_cb_noroom = 0;   // held experts copied back to RAM at a vacate, the time, none free
    bool     vx_keep_short = false;   // the VRAM pool's sizing ended with less than --moe-stream-vram-free free on its GPU
    int64_t  tail_n_vacate = 0, tail_n_vac_places = 0, tail_n_fill_calls = 0, tail_n_fill_jobs = 0, tail_n_fill_noram = 0;
    int64_t  tail_n_taken = 0, tail_n_declined[8] = { 0 }; // the hook: copies given a target / refused, by reason
    int64_t  tail_n_taken_real = 0;                        // of the copies given a target: in graphs that were computed (tail_real)
    bool     tail_decline_said = false;
    std::map<int32_t, std::vector<int32_t>> vx_rank_lists;   // the whole list file by layer (the tail's extras, the fill's tie-break)
    std::vector<std::vector<int16_t>>       vx_rank_of;      // [layer][expert] position in that list, -1 = not in it
    static int64_t offload_min_batch() { static const int64_t v = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32; return v; }
    bool tail_graph(int64_t n_tok) const;       // would a ubatch of n_tok tokens build a GPU-wave graph (pool copies with a preset target)
    bool tail_batch(int64_t n_tok_all) const;   // does a call of n_tok_all tokens hold such a ubatch
    void tail_ubatch(int64_t n_tokens, int64_t lm_bytes = 0);   // llama_context, before every ubatch's graph: vacates the tail's copy
                                                                 // places for a GPU-wave graph (+ the extension a layer-major call's
                                                                 // tensors of lm_bytes need)
    bool tail_copy_hook(ggml_backend_t be, const ggml_tensor * src, ggml_tensor * copy, const ggml_tensor * consumer, int i_split,
            const ggml_cgraph * graph);
    struct vf_cand_t { uint64_t key; int32_t il, e, rank; };
    // The candidates for the free places of pool group gi, best first (at most `want`)
    void vf_fill_rank(size_t gi, size_t want, std::vector<vf_cand_t> & out) const;
    template <bool FIXED> void    vf_drain_all(std::unique_lock<std::mutex> & lock);
    template <bool FIXED> void    tail_vacate_t(std::unique_lock<std::mutex> & lock, int32_t n_places);
    template <bool FIXED> int32_t vf_fill_t(std::vector<vf_job_t *> * made);   // jobs for the free places, not dispatched when `made` is given
    void vf_dispatch(vf_job_t * jp);                                           // the copy of a job without a copy-back
    void (llama_moe_stream::*tail_vacate_fn)(std::unique_lock<std::mutex> &, int32_t) = nullptr;
    int32_t (llama_moe_stream::*vf_fill_fn)(std::vector<vf_job_t *> *)      = nullptr;
    void (llama_moe_stream::*vf_drain_fn)(std::unique_lock<std::mutex> &)    = nullptr;   // used by lm_lend_place
    void    (llama_moe_stream::*vf_tick_fn)()                                = nullptr;   // the end fill's tick (value 2)
    // --moe-stream-vram-end-fill: the tail's free places filled at the end of a prompt-end
    // call (end_call), ranked by the call's prompt-end weights (p1_pe_w, the stand-in for the window's keep values) instead of VRAM follow's
    // counts; vf_rank_pe switches vf_fill_rank / vf_tick to those weights while it runs
    bool     vf_rank_pe = false;
    int64_t  n_end_fill = 0, n_end_fill_jobs = 0, n_end_tick_jobs = 0, n_end_fill_skip = 0;
    uint64_t vf_key(const llama_moe_stream_layer & sl, uint32_t e) const;   // the rank key (counts, or the prompt-end weights)
    void     end_fill();                                                     // mtx held, end_call of a complete prompt-end call
    // The moving RAM window. llama_context's layer
    // signals of a layer-major call; the window acts on them. lm_layer_done's event (null on the CPU path) is recorded on the
    // compute stream right after the layer's expert graph: the layer's RAM slots are not read by this call once it has completed
    bool     lm_events   = false;   // llama_context records the per-layer event (the window or MOE_STREAM_TEST_WIN_CLOBBER asks for it)
    bool     lm_in_call  = false;   // between lm_call_begin and lm_call_end
    bool     lm_call_pe  = false;   // the call ends a prompt (the call-level prompt-end rule)
    int64_t  lm_call_tok = 0;
    int32_t  lm_cur_il   = -1;      // the layer being computed (lm_layer_begin)
    bool     lm_test_clobber = false;   // MOE_STREAM_TEST_WIN_CLOBBER=1 (test only): a finished layer's slots overwritten and emptied
    bool     lm_test_reads   = false;   // MOE_STREAM_TEST_LM_READS=1 (test only): the reads per streamed layer of each layer-major call
    std::vector<int64_t> lm_reads0;     // [n_layer] the layer's reads at lm_call_begin (lm_test_reads)
    int64_t  n_lm_clobbered = 0;
    void     lm_call_begin(int64_t n_tok, bool prompt_end);
    void     lm_layer_begin(int32_t il);
    void     lm_layer_done(int32_t il, ggml_backend_event_t ev);
    void     lm_call_end(bool complete);
    // The gather copy (tail waves): a GPU-wave graph's expert matmuls read three tensors at the tail's copy places (where the scheduler's
    // pool copies go by default), and the wave's remap op - after its slot ids are final - maps them to places (a slot below the wave width
    // keeps its number, others take a free place; the zero ids the zero places) and queues the copies "place p <- RAM slot s" on the
    // compute stream (tailw_be), merged into runs. A wave's experts may then sit in any slot of the pool. On with the window (tailw_want, set
    // by it) and a tail; MOE_STREAM_TEST_WIN_GATHER=1 (test only) turns it on without the window, MOE_STREAM_TEST_WIN_NOGATHER=1 (test
    // only) keeps the scheduler's copy. The scheduler copies nothing for these matmuls
    bool          tailw_want = false, tailw_test_off = false;
    bool          tailw_test_move = false;   // MOE_STREAM_TEST_WIN_TAIL_MOVE=1 (test only): no slot keeps its number (the free-place path)
    bool          tailw_ready = false;       // vx_setup: tail waves can run here (a CUDA GPU, pageable pools, a pool slice = a VRAM place)
    std::string   tailw_why;                 // why not
    bool          pools_registered = false;   // pin_pools page-locked at least one pool
    ggml_context * tailw_ctx = nullptr;
    ggml_tensor *  tailw_t[3] = { nullptr, nullptr, nullptr };   // gate / up / down, [ne0, ne1, tail_copy_n] at the tail's copy places
    // One tail-wave tensor set per pool group (format), [ne0, ne1, tail_tc] at that group's copy places (they share the tail's bytes);
    // tailw_t = the tail group's set (the "tail waves are on" tests); tailw_zero_group = the group whose zero places the tail holds now (-1 = none)
    std::vector<std::array<ggml_tensor *, 3>> tailw_tg;
    int32_t        tailw_zero_group = -1;
    // --moe-stream-copy-ahead: the tail's copy places are two full-width sets (nothing of a batch's room is given up);
    // a streamed layer uses set (its index among the streamed layers) % 2, and tailw_tg holds [set x pool group] (set 1 after set 0). Layer
    // L's tail gather hands the next streamed layer's READY experts to the VRAM-follow uploader, which copies them into the other set behind an event
    // recorded there (so behind L-1's matmuls, that set's last reader); L+1's gather waits for the uploader's event and copies only the
    // rest. Prompts whose experts all fit one wave (every streamed layer holds all its experts in RAM) - they take tail waves too
    bool     ca_want = false;          // vx_setup: the setting, tail waves can run and every streamed layer holds all its experts
    // Every streamed layer holds all its experts in RAM (case 1) - a VRAM-held expert keeps its RAM copy (the slots exist
    // anyway), so emptying the tail or lending places costs no copy-back and the refill finds them in RAM. MOE_STREAM_TEST_NO_KEEP_RAM=1
    // (test only) = the old rule (a held expert never has a RAM slot)
    bool     vx_keep_ram = false;
    int64_t  n_keep_ram_skip_cb = 0;   // copy-backs not needed (the leaving expert is still in RAM)
    bool     ca_one_wave = false;      // one-wave GPU prompts of >= gpu_waves rows take tail waves (the tail exists for them)
    bool     ca_on = false;            // the two sets exist and the VRAM-follow uploader runs: jobs are made
    int32_t  ca_sets = 1;              // copy-place sets in the tail (2 with copy-ahead)
    int32_t  ca_set_places = 0;        // places of one set (tail_copy_n = ca_sets x this)
    std::vector<int32_t> ca_pos;       // [il]: the layer's index among the streamed layers, -1 = none
    int32_t  tailw_zero_group_s[2] = { -1, -1 };   // per set: the pool group whose zero places it holds (-1 = none since the vacate)
    // The job's experts go into the SAME set - the places the current wave reads - so the
    // uploader copies only their gate/up parts, and only after the GPU mark behind the current wave's gate/up activation (wait_il's
    // counter = wait_val); their down parts are copied by the next gather on the compute stream (behind the current wave's down)
    bool     ca_p2 = false;            // vx_setup: one-set mode (--moe-stream-copy-ahead 2, or 1 where not every expert is in RAM)
    bool     ca_done_ok = false;       // ne_gpu_setup: the "gate/up done" marks are registered (needed-early's GPU producer, NVIDIA)
    ggml_backend_buffer_t ca_pin_buf = nullptr;      // pinned: [layer] counters, then [layer] expected values (64 bytes each)
    uint32_t * ca_done_ctr = nullptr, * ca_done_exp = nullptr;
    std::vector<uint32_t> ca_done_want;              // graph thread, [il]: the value the layer's last mark will write (0 = none)
    uint32_t ca_done_seq = 0;
    uint32_t * ca_ctr(int32_t il) const { return ca_done_ctr + (size_t) il*16; }
    uint32_t * ca_exp(int32_t il) const { return ca_done_exp + (size_t) il*16; }
    void     ca_mark_node(ggml_tensor * t, int32_t il);   // graph build: the wave's gate/up activation tells the copier (one-set mode)
    struct ca_job_t {
        int32_t il = -1, w = 0;            // the layer and wave the places are for
        int32_t n_w = 1;                   // the call's waves (the set of a step)
        // Follow-up copies: the step's experts not READY when the job was made; the uploader adds the ones that land
        // before the step's gather starts (ca_follow_stop) into the set's free places, and hands (expert, place, slot) back in `added`
        std::vector<int32_t> pend;
        std::vector<uint8_t> used;         // the set's places taken by the job
        std::vector<std::array<int32_t, 3>> added;
        std::vector<std::pair<int32_t, int32_t>> list;   // (place, slot)
        bool    queued = false, issued = false;
        bool    gu_only = false;           // one-set mode: gate / up parts only, after wait_il's mark wrote wait_val
        bool    abandoned = false;         // the mark did not come in 10 s - nothing was copied
        int32_t wait_il = -1;
        uint32_t wait_val = 0;
        int64_t bytes = 0, us = 0, wait_us = 0;
    };
    ca_job_t ca_job;                   // vf_mtx: the one job out
    ggml_backend_event_t ca_ev_c = nullptr, ca_ev_a = nullptr;   // the compute stream -> the uploader, the uploader -> the compute stream
    static constexpr size_t H2D_EV_RING = 64;
    ggml_backend_event_t  tailw_ev_ring[H2D_EV_RING] = {};
    ggml_backend_event_t  ca_ev_ring[H2D_EV_RING]    = {};
    std::mutex            h2d_ev_mtx;
    std::atomic<uint64_t> tailw_h2d_seq{0};
    std::atomic<uint64_t> tailw_h2d_acked{0};
    std::atomic<uint64_t> ca_h2d_seq{0};
    std::atomic<uint64_t> ca_h2d_acked{0};
    void                  wait_slot_h2d_done(const llama_moe_stream_layer & sl, int32_t slot);
    void                (*unpin_pool_fn)(void *) = nullptr;
    std::vector<uint8_t>  pool_pinned;
    int32_t  ca_job_il = -1;           // graph thread: the layer the job's places are for
    int32_t  ca_job_w  = 0;            // graph thread: and the wave (one-set mode: the next wave of the same layer, or wave 0)
    int32_t  ca_job_set = 0;           // graph thread: and the set its places are in (the gather's "mine" check)
    void     ca_no_job_out(const char * where);   // mtx held, before a graph-thread wait on the uploader
    std::atomic<bool> ca_cancel{false};   // ca_end_call: a one-set job still waiting for its mark copies nothing
    std::atomic<bool> ca_follow_stop{false};   // the job's step gather has started - no more follow-up copies
    std::atomic<bool> ca_step_started{false};  // the job's step remap op has started - follow-ups set no keep mark from then on
    std::atomic<bool> ca_follow_hold{false};   // from the step's mark end to its pin pass - follow-ups wait
    std::atomic<int64_t> ca_follow_held{0};    // follow-up checks the hold put off
    int64_t  ca_follow = 0, ca_follow0 = 0;    // experts copied as follow-ups (landed after their job was made)
    bool     ca_two_room = false;              // vx_setup: one set costs at most a quarter of the pool's room
    bool     ca_follow_on = true;              // follow-up copies (TEST ONLY MOE_STREAM_TEST_CA_NO_FOLLOW=1: off)
    bool     ra_onewave = true;                // read-ahead in one-wave calls (TEST ONLY MOE_STREAM_TEST_RA_NO_ONEWAVE=1: off)
    // TEST ONLY: MOE_STREAM_TEST_CA_ABORT=<n> - the first n prompt calls with tail waves stop part-way, right after the middle
    // streamed layer's gather (its "gate/up done" value is set, its GPU split never runs) through an abort callback on the context
    int32_t  ca_test_abort_left = 0;
    bool     ca_test_abort_set = false;
    std::atomic<bool> ca_test_abort_now{false};
    void     ca_test_abort_install(const void * ctx);
    std::vector<uint8_t> ca_stale;        // graph thread: [il] the layer's marks failed once - the tail gather copies its next steps
    int64_t  ca_fallbacks = 0;
    bool     ca_is_stale(int32_t il) const { return il >= 0 && (size_t) il < ca_stale.size() && ca_stale[(size_t) il]; }
    void     ca_set_stale(int32_t il, const char * why);
    int64_t  ca_too_late = 0, ca_left = 0, ca_gu_wait_us = 0, ca_too_late0 = 0, ca_left0 = 0, ca_xgroup = 0;   // not READY at the job / copied by the tail gather itself
    std::vector<int32_t> ca_place, ca_src, ca_list_e;   // graph thread: [expert] -> place / RAM slot of the job; the job's experts
    std::vector<int32_t> ca_es, ca_slots;               // scratch
    int64_t  ca_jobs = 0, ca_experts = 0, ca_bytes = 0, ca_us = 0, ca_found = 0, ca_wait_us = 0, ca_checked = 0, ca_late = 0;
    int64_t  ca_jobs0 = 0, ca_experts0 = 0, ca_found0 = 0, ca_wait_us0 = 0;   // at the call's start (the per-call line)
    bool     ca_test_check = false;    // MOE_STREAM_TEST_COPY_AHEAD_CHECK=1 (test only): every early-copied place read back and compared
    int64_t  ca_zero_checked = 0, ca_zero_bad = 0, ca_place_checked = 0, ca_place_bad = 0;
    int64_t  ca_graph_checked = 0, ca_graph_bad = 0, ca_prev_checked = 0, ca_prev_bad = 0;   // (CHECK=1) the graph's tail tensor = the gather's; the last step's places intact at the next gather
    struct ca_prev_t { const ggml_tensor * t[3]; int32_t il, wave, set, place; uint64_t h[3]; };
    std::vector<ca_prev_t> ca_prev;   // (CHECK=1) the last gather's places and their bytes' hashes   // (CHECK=1) zero-place projections a gather trusted, read back; those not all zero
    bool     ca_test_nowait = false;   // MOE_STREAM_TEST_COPY_AHEAD_NOWAIT=1 (test only, the control): no wait for the uploader
    int32_t  ca_test_set = -1;         // MOE_STREAM_TEST_CA_SET=0|1 (test only): every step uses that set (with CA_NO_JOBS)
    bool     ca_test_no_jobs = false;  // MOE_STREAM_TEST_CA_NO_JOBS=1 (test only): no jobs - every gather copies its step itself (the sets still alternate)
    // TEST ONLY (MOE_STREAM_TEST_COPY_AHEAD_CHECK=2): the read-back that keeps the timing - at each gather the found places are
    // sampled (one 4 KiB block in every 64 KiB of each projection tensor, and every place's last 4 KiB) by strided GPU -> host copies on
    // the compute stream right before the matmul that reads them (stream order: what the matmul reads, before any later writer); the same
    // bytes of the RAM slots are copied by the graph thread; a checker thread compares them after the step's event. No host wait
    // except a full ring (counted)
    struct ck_step_t {
        int32_t il = -1, wave = 0;
        std::vector<std::array<int32_t, 3>> found;   // (place, slot, expert)
        ggml_backend_buffer_t buf = nullptr;          // pinned host memory (the GPU samples)
        uint8_t * gpu = nullptr;
        size_t    cap = 0;
        std::vector<uint8_t> ram;                     // the same layout, from the RAM slots
        size_t    nb[3] = { 0, 0, 0 }, rows[3] = { 0, 0, 0 }, tw[3] = { 0, 0, 0 }, o_rows[3] = { 0, 0, 0 }, o_tail[3] = { 0, 0, 0 };
        int32_t   n_parts = 3;
        ggml_backend_event_t ev = nullptr;
        bool      busy = false;
    };
    bool (*ck_get2d)(ggml_backend_t, void *, size_t, const void *, size_t, size_t, size_t) = nullptr;
    std::vector<ck_step_t> ck_ring;
    size_t   ck_head = 0, ck_tail = 0;
    mutable std::mutex ck_mtx;
    mutable std::condition_variable ck_cv;
    std::thread ck_thread;
    bool     ck_stop = false;
    std::atomic<int64_t> ck_places{0}, ck_bytes{0}, ck_steps{0};
    int64_t  ck_wait_us = 0;
    void     ck_enqueue(llama_moe_stream_layer & sl, int32_t wave, const std::array<ggml_tensor *, 3> & g1s,
                        const std::vector<std::array<int32_t, 3>> & found, int32_t n_parts);
    void     ck_loop();
    void     ck_drain() const;   // every enqueued step checked (the exit line counts them)
    // With two sets each step of a call - a wave of a layer, layer by layer - copies into the other set: step = the layer's
    // index among the streamed layers x the call's waves + the wave (one-wave prompts: the layer's index)
    int32_t  ca_set_of(int32_t il, int32_t wave, int32_t n_w) const {
        return ca_sets > 1 && ca_test_set >= 0 ? ca_test_set : ca_sets > 1 && il >= 0 && (size_t) il < ca_pos.size() && ca_pos[(size_t) il] >= 0 ?
            (int32_t) (((int64_t) ca_pos[(size_t) il]*std::max<int32_t>(n_w, 1) + wave) % 2) : 0;
    }
    void     ca_run_job(ca_job_t & j);   // the uploader thread
    bool     ca_finish_job();            // the graph thread: wait until the job is issued, then drop it; true = it was issued
    void     ca_end_call();              // a call's end: a job still out is finished and its copies have landed
    ggml_backend_t tailw_be  = nullptr;      // llama_context: this call's backend on the tail's GPU (its compute stream)
    bool           tailw_zero_ok = false;    // the tail's zero places hold zeros (cleared by every vacate and fill)
    std::vector<int32_t> tailw_place_of;     // scratch [pool slots]: the place a slot got in this wave, -1 = none
    std::vector<uint8_t> tailw_used;         // scratch [tail_copy_n]
    std::vector<std::pair<int32_t, int32_t>> tailw_list;   // scratch: (place, slot) of this wave
    int64_t        tailw_waves = 0, tailw_runs = 0, tailw_calls = 0, tailw_bytes = 0, tailw_us = 0, tailw_experts = 0, tailw_moved = 0;
    int64_t        tailw_waves0 = 0, tailw_calls0 = 0, tailw_bytes0 = 0, tailw_us0 = 0, tailw_experts0 = 0;   // at this call's start (the per-call line)
    bool tailw_graph(int64_t n_tok) const {   // a ubatch (or a layer-major chunk) of n_tok rows reads its GPU-wave experts through the tail gather
        return tailw_want && !tailw_test_off && tailw_t[0] != nullptr && tail_graph(n_tok);
    }
    ggml_tensor * tailw_tensor(int k) const { return tailw_t[k]; }
    ggml_tensor * tailw_tensor(int k, int32_t il, int32_t wave, int32_t n_w) const {   // the tail tensor of layer il's format (copy-ahead: and set)
        const int32_t g = il >= 0 && (size_t) il < layers.size() && layers[il] ? layers[il]->vx_group : -1;
        const size_t  i = (size_t) ca_set_of(il, wave, n_w)*vx_groups.size() + (size_t) g;
        return g >= 0 && i < tailw_tg.size() ? tailw_tg[i][(size_t) k] : tailw_t[k];
    }
    void tailw_gather(llama_moe_stream_layer & sl, int32_t * out, int64_t n, int32_t wave);   // the remap op, after its slot ids are final

    // Pinned H2D staging ring + routed-only mask + CPU hybrid tail co-execution
    static constexpr int  PRING_SLICES = 4;
    ggml_backend_buffer_t pring_buf    = nullptr;
    char *                pring_ptr    = nullptr;
    size_t                pring_cap    = 0;
    size_t                pring_head   = 0;
    int                   pring_slice  = 0;
    ggml_backend_event_t  pring_ev[PRING_SLICES] = {};
    bool                  pring_ev_rec[PRING_SLICES] = {};
    int64_t               pring_wraps = 0, pring_bytes = 0;
    ggml_backend_buffer_t tailw_zero_pin_buf = nullptr;
    uint8_t *             tailw_zero_pin_ptr = nullptr;
    size_t                tailw_zero_pin_cap = 0;
    ggml_backend_buffer_t vf_pring_buf       = nullptr;
    char *                vf_pring_ptr       = nullptr;
    size_t                vf_pring_cap       = 0;
    size_t                vf_pring_head      = 0;
    int                   vf_pring_slice     = 0;
    ggml_backend_event_t  vf_pring_ev[PRING_SLICES] = {};
    bool                  vf_pring_ev_rec[PRING_SLICES] = {};
    void                  pring_ensure();
    void                  tailw_set_async_staged(ggml_tensor * D, const char * src, size_t dst_off, size_t nbytes);
    void                  vf_pring_ensure();
    void                  vf_set_async_staged(ggml_tensor * D, const char * src, size_t dst_off, size_t nbytes);

    std::vector<std::vector<uint8_t>> lm_routed_mask;   // [il][expert]: 1 if routed in lm_phase 1
    void set_lm_routed_ids(int32_t il, const int32_t * ids_host, int64_t n);
    void clear_lm_routed_masks() { lm_routed_mask.clear(); }

    // Three pieces for --moe-stream-g1-ahead; nothing calls them
    // unless that setting or the test switches below are on
    uint32_t tailw_wave_w = 0;   // (c) the wave width of a tail-wave graph (graph and remap op alike), 0 = the plain width; option A sets the half places
    int n_waves_narrow(int64_t n_tok) const;
    // (a) which experts of a list are READY in this layer's RAM: slots[i] = the slot of es[i], -1 = not READY (absent, being read,
    // cancelled, switched off, a low-only call's slot above the start size - wave_plan's misses). (b) keep_wave >= 0: every READY one is
    // kept for that wave of this call (no eviction, no window move, no drop) until that wave's remap op of this layer, or the call's end.
    // Returns the READY count. expert_ready takes mtx; the _locked form is for a caller that holds it
    int32_t expert_ready(int32_t il, const int32_t * es, int32_t n, int32_t * slots, int32_t keep_wave = -1);
    int32_t expert_ready_locked(llama_moe_stream_layer & sl, const int32_t * es, int32_t n, int32_t * slots, int32_t keep_wave);
    // wave w's experts of this layer's running call (the stored wave plan, else the first-use rank range); false = not known (no
    // multi-wave call of this layer has run its wave 0, or w past the plan). Takes mtx
    bool    wave_experts(int32_t il, int32_t w, std::vector<int32_t> & out);
    std::vector<int32_t> ahead_layers;   // layers with n_ahead > 0 (mtx)
    std::vector<int32_t> ahead_free;     // slots of the running op's layer whose marks it ended (freed before its resident pins)
    bool    ahead_on   = false;          // a keep was ever asked for (the call start / end release runs only then)
    bool    ahead_test = false;          // TEST ONLY: MOE_STREAM_TEST_AHEAD=1 - every wave op of a multi-wave call keeps the next wave's READY experts
    int64_t n_ahead_kept = 0, n_ahead_found = 0, n_ahead_left = 0, n_ahead_ready = 0, n_ahead_asked = 0;
    void    ahead_restore(llama_moe_stream_layer & sl, int32_t wave);   // mtx held, a remap op after its pin clear
    void    ahead_release_all();                                       // mtx held, a call's start / end
    // Prompt split's pieces (a) + (c) (--moe-stream-prompt-split); nothing runs while psplit_on is false
    bool    psplit_on = false;                // set by the setting (or MOE_STREAM_TEST_PROMPT_SPLIT)
    int32_t psplit_test = 0, psplit_test_borrow = 0;   // TEST ONLY: MOE_STREAM_TEST_PROMPT_SPLIT=1 (count) / 2 (+ keep the top 16 of the next layer for its
                                               // wave 0); MOE_STREAM_TEST_PROMPT_SPLIT_BORROW=N (borrow N tail places at each prompt call's first op)
    // (a) layer il's guessed experts of the running prompt piece, (expert, count) most-used first; false = none (takes mtx)
    bool    prompt_guess(int32_t il, std::vector<std::pair<int32_t, int32_t>> & out);
    void    psplit_count(llama_moe_stream_layer & nx, const int32_t * pred, int64_t w, int64_t n_tok);   // mtx held
    void    psplit_check(llama_moe_stream_layer & sl);   // mtx held, sl's wave-0 op: the guess against the real picks, then ended
    // (c) borrow up to n tail places, lowest VRAM follow key first (empty ones first), never one holding an expert of keep_il's guesses; the held
    // expert steps aside, the place is marked borrowed (vf_place_il = -2) until tail_borrow_end (end_call): then free, tail_refill_due set
    // - the end fill or the first word's refill fills them. Returns the places' count (takes mtx)
    int32_t tail_borrow(int32_t n, const std::vector<int32_t> * keep_il, std::vector<int32_t> & places);
    void    tail_borrow_end();   // mtx held
    void    psplit_call_start();     // takes mtx
    std::vector<int32_t> psplit_borrowed;   // places borrowed in this call (mtx)
    bool    psplit_layer_ran = false;   // a streamed layer's op of this call has run: tail_borrow refuses (mtx)
    int64_t n_psplit_lists = 0, n_psplit_guessed = 0, n_psplit_hit = 0, n_psplit_real = 0, n_psplit_top16 = 0, n_psplit_top16_hit = 0;
    int64_t n_psplit_borrowed = 0, n_psplit_borrowed_held = 0, n_psplit_returned = 0, n_psplit_calls = 0;
    // Elastic RAM and moving slot shares - src/llama-moe-elastic.cpp
    bool     el_on = false, sh_on = false;
    int64_t  el_step = (int64_t) 512 << 20, el_growmin = -1;   // el_growmin = take-again (-1 = auto: keep + 1 GiB deadband)
    bool     el_growmin_set = false;
    double   el_cool_s = 30.0;                     // opts settle= (cool=): no take-back until nothing needed RAM for this long
    int32_t  el_stall_us = 100000;
    uint32_t el_ceiling = 0;                       // 0 = what all RAM holds
    int64_t  el_refill_step = (int64_t) 256 << 20; // opts refillstep=: max bytes reclaimed per paced take-back step
    int32_t  el_refill_ms = 5000;                  // opts refillms=: min milliseconds between consecutive take-back steps
    int64_t  el_last_take_step_us = 0;             // mtx: timestamp of last take-back step
    double   el_soft_min_pct = 25.0;               // opts softmin=: % of n_expert below which squeezed_warn state activates
    bool     el_soft_min_set = false;
    int32_t  el_soft_min_layer = 0;                // computed soft comfort floor per streamed layer
    bool     el_squeezed_warn_said = false;        // mtx: logged once per squeeze excursion below el_soft_min_layer
    bool     el_win_holes = false;                 // Windows: a layer has decommitted slots below wave_view_slots()
    std::atomic<int64_t>  el_last_room_bytes{-1};  // lock-free snapshot of free headroom for layout telemetry
    std::atomic<int64_t>  el_last_room_raw{-1};    // the elastic thread's last room reading (el_room), for a word's take-back
    std::atomic<int64_t>  el_last_avail{-1};       // its MemAvailable at that look
    std::atomic<int64_t>  el_last_look_us{0};      // when it looked
    int64_t  el_word_take_try_us = 0;              // mtx: the last take-back try at a word's end
    int64_t  el_n_word_take = 0;                   // mtx: take-back steps made at a word's end
    std::atomic<uint32_t> el_min_share_pub{0};     // lock-free snapshot of smallest active share per streamed layer
    std::atomic<int64_t>  el_net_given_bytes{0};   // lock-free snapshot of net bytes lent below initial pool size
    double   sh_step_pct = 1.0, sh_gap = 1.5, sh_half = 256.0, sh_floor_pct = 25.0, sh_ceil_pct = 300.0;
    int32_t  sh_every = 64, sh_hits = 32, sh_cool = 512;
    uint32_t sh_min = 24;
    std::atomic<int64_t> el_req{0};                // bytes the pressure alarm asked to give back, not given yet
    std::atomic<int64_t> el_events{0};             // pressure events (real + test)
    bool     el_in_call = false;                   // mtx: a decode() call is between set_decode and end_call
    bool     el_hold_gives = false;                // mtx: a prompt call with GPU-wave copies (tail waves, copy-ahead) runs - slots are switched
                                                   // off at its end, not during it (the band still frees RAM at any moment)
    int64_t  el_n_held_gives = 0;                  // give-backs put off to such a call's end
    int64_t  el_last_give_us = 0;                  // mtx
    int64_t  el_last_big_give_us = 0;              // mtx: the last give-back asked of settlefall or more (the trim waits after it)
    int64_t  el_last_change_word = -1000000;       // mtx: the decode word of the last total change (slot shares wait)
    uint32_t el_min_share = 0;                     // mtx: the smallest share (the wave bound)
    uint32_t el_total = 0;                         // mtx: the sum of the shares
    int64_t  el_given_slots = 0, el_taken_slots = 0, el_n_give = 0, el_n_take = 0, el_n_refill = 0, el_n_refill_cx = 0;
    int64_t  el_events_unanswered = 0;             // mtx: alarms in a row without a give-back while a layer was above its floor
    int64_t  el_events_seen = 0, el_gives_seen = 0;
    size_t   el_rr = 0;                            // mtx: the equal cut's rotating start
    uint32_t el_grow_max = UINT32_MAX;             // mtx: the most slots a layer may grow to (no bound)
    uint32_t el_wave_w = 0;                        // mtx: a prompt wave's width (prompt wave cap; 0 = n_slots, features off)
    bool     el_has_gpu = false;                   // a GPU device: prompt matmuls may run there (prompt wave cap)
    bool     sh_test_log = false;                  // TEST ONLY: MOE_STREAM_TEST_SHARE_LOG (one line per slot shares check)
    bool     el_test_lowskip = false;              // TEST ONLY: MOE_STREAM_TEST_LOWONLY_SKIP (the slot check's failure test)
    int64_t  el_n_low_calls = 0, el_n_low_drop = 0;
    bool     el_swap_on = true;                    // --moe-stream-elastic-opts swap=0|1: the swap-out signal
    // Keep-free = p1.ram_margin, take-again = el_growmin; a give-back brings the free RAM to the band's middle
    // (never more than one step above keep-free: take-again 1000GiB = "never take back" made the middle ~500 GiB)
    int64_t  el_committed() const;                 // mtx: switched-on EMPTY slots' bytes (promised to the next reads, no RAM yet)
    int64_t  el_x_committed() const;               // mtx: the same as elastic RAM's decisions count it (0 in the fake squeeze)
    void     el_serve_in_call();                   // mtx held, a call runs: free what may be freed now (the eviction's rules)
    bool     el_test_free_pinned = false;          // TEST ONLY: MOE_STREAM_TEST_FREE_PINNED (the failure test: a pinned slot in a free)
    bool     el_test_keep_skip = false;            // TEST ONLY: MOE_STREAM_TEST_KEEP_SKIP (the failure test: a take-back past keep-free)
    int64_t  el_n_now_give = 0, el_n_now_slots = 0, el_n_idle_take = 0;   // mtx: frees during a call; take-backs from the thread
    // Obey the OS - opts fill=0|1, relearn=S, part=SIZE
    // The rule (the default): free RAM = keep-free; give back only (a) under keep-free or (b) on PSI, in
    // proportion; swap-outs of idle memory are logged only (or answered by discarding our band, near keep-free)
    int64_t  el_gap = (int64_t) 512 << 20;         // opts gap=: a give-back brings free RAM to keep + gap (default 512 MiB deadband)
    bool     el_gap_set = false;
    int64_t  el_psi_floor = (int64_t) 2 << 30;     // opts psifloor=: a PSI event is a need only with less free (2.0 GiB)
    // Discardable expert slots
    bool     el_dz_opt = true;                     // opts discard=0|1
    bool     el_cache_opt = true;                  // opts cache=0|1: old file cache counts as free room (Linux)
    bool     el_dz_ok = false;                     // the OS has them here (the probe; set by el_parse_opts)
    bool     el_dz_on = false;                     // and every pool is fresh (create_pool): the band runs (a give-back takes it last)
    std::atomic<int64_t> el_dz_now{0};             // discardable slots now (changed under mtx; read by el_room before it)
    std::atomic<int64_t> el_win_cshort{0};         // Windows: el_room's room minus the commit room at its last read (>= 0; el_x_committed)
    int64_t  dz_n_unlock = 0, dz_n_lock_ok = 0, dz_n_lock_fail = 0, dz_n_claim = 0, dz_n_taken = 0, dz_n_relock = 0, dz_n_relock_fail = 0;
    int64_t  n_dz_unlock_fail = 0;                  // the OS refused an unlock (discardable slots off from then on)
    int64_t  n_fb_noslot = 0, n_fb_noslot_freed = 0; // "no free slot" fallbacks (kept sets released, VRAM follow drained) - usually a bug
    int64_t  n_fb_wplan = 0;                        // wave plans that failed -> the plain waves for that layer and call
    int64_t  n_fb_commit = 0;                       // slots the OS would not commit (left switched off)
    int32_t  fb_test_noslot = -1;                   // TEST ONLY (MOE_STREAM_TEST_NO_FREE_SLOT=<layer>): the first pick there finds no slot
    int64_t  fb_test_commit = 0;                    // TEST ONLY (MOE_STREAM_TEST_COMMIT_FAIL=N): every Nth slot commit fails
    int32_t  fb_test_commit_part = 0;               // TEST ONLY (MOE_STREAM_TEST_COMMIT_FAIL_PART=0|1|2): at which projection
    int64_t  n_fb_commit_released = 0;              // projections committed before a refusal and given back
    int64_t  dz_n_probe = 0, dz_n_probe_slots = 0, dz_n_hits = 0, dz_n_falls = 0;
    // Taking band slots back ahead of use: a guess that names an expert sitting in a band slot queues it here (mtx), a reader takes it back
    // before the layer's remap op reaches it; the remap op's own take-back stays as the fallback
    struct dz_ahead_t { llama_moe_stream_layer * sl; int32_t slot, expert; };
    std::deque<dz_ahead_t> dz_ahead_q;
    int64_t  dz_n_ahead = 0, dz_n_ahead_lost = 0, dz_n_onpath = 0;   // taken back by a reader (the OS had taken it), by the remap op
    int64_t  el_band_age = 1024;                   // opts age=: reply words a slot's expert must go unused before it joins the band
    int64_t  dz_age_used = 0;                      // the age the last upkeep used (in the layer's expert uses; the stats line)
    double   dz_probe_ms_sum = 0.0, dz_probe_ms_max = 0.0, dz_hold_ms_max = 0.0, dz_lock_us_sum = 0.0, dz_unlock_us_sum = 0.0;
    int64_t  el_n_settle_wait = 0;                 // idle looks a take-back waited for the settle rule
    int64_t  el_others_last = -1;                  // the other programs' memory at the last look (the settle rule)
    int      el_own_cg = -1;                       // our cgroup holds only this process (1), not (0), not read yet (-1)
    int64_t  el_os_others();                       // OS layer: MemTotal - MemFree - our own                       // the free RAM at the last look (a fall = an app growing - no take-back)
    int64_t  el_settle_fall = (int64_t) 64 << 20;  // opts settlefall=: a fall between two looks that counts as "free RAM falling"
    // Meta's Senpai way (proactive reclaim): the elastic RAM thread trims OTHER programs' old FILE cache in small steps
    // (memory.reclaim swappiness=0; our own cgroup protected) while the engine is under its ceiling or free RAM under keep-free, and
    // takes the freed RAM back as slots; it stops at once when a program waits on memory (PSI: the system, the desktop session) and backs
    // off before trying again
    bool     el_tr_opt = true;                     // opts trim=0|1
    bool     el_tr_ok = false;                     // it can run here (el_parse_opts: the cgroup, memory.reclaim, the swappiness= argument)
    std::string el_tr_why;                         // why not (the start-up line)
    std::vector<std::string> el_tr_targets;        // where (the first: our cgroup's parent, our cgroup protected)
    std::string el_tr_guard;                       // our cgroup's memory.min
    int64_t  el_tr_step0 = (int64_t) 64 << 20;     // opts trimstep=: the first step (and after a stop)
    int64_t  el_tr_max = (int64_t) 1 << 30;        // opts trimmax=: the largest step (doubling while nobody waits)
    int64_t  el_tr_psi_us = 1000;                  // opts trimpsi=: memory-stall time between two looks (us, system or session) that stops it
    double   el_tr_back_s = 10.0, el_tr_back_max_s = 160.0;   // opts trimback= / trimbackmax=: the pause after a stop (doubling)
    int64_t  el_tr_step = 0, el_tr_next_us = 0, el_tr_sys_last = -1, el_tr_ses_last = -1;
    double   el_tr_back_now = 0.0;
    bool     el_tr_after = false;
    int      el_tr_last_why = -1;                  // TEST ONLY (LOOK_LOG): the last decision's reason                  // the last look trimmed (a stall at this look stops it)
    int64_t  el_tr_n = 0, el_tr_n_again = 0, el_tr_n_stop = 0, el_tr_bytes = 0, el_tr_got = 0;   // trims, EAGAIN, stops, asked, MemFree gained
    double   el_tr_cpu_ms = 0.0, el_tr_wall_ms = 0.0, el_tr_wall_max = 0.0;
    int64_t  el_tr_decide(int64_t now, int64_t x, int64_t memfree, int64_t cgroom, int64_t committed, bool give, int64_t sys_us, int64_t ses_us);   // mtx held
    int64_t  el_tr_run(int64_t bytes, bool * none_left = nullptr, bool keep_mode = false);   // elastic RAM thread, no lock: one trim; the cache's drop
    void     el_short(int64_t bytes, int reason, int64_t allowed = -1);   // elastic RAM thread, no lock: a shortfall's order - our empties, old file cache, our filled slots
    int64_t  el_tr_keep(int64_t bytes, bool parent_ok = false);             // elastic RAM thread, no lock: trim FIRST for keep-free; MemFree gained
    int64_t  el_tr_n_skip = 0;                     // trim writes that failed on a target (skipped, not fatal)
    int64_t  el_tr_n_keep = 0, el_tr_keep_bytes = 0, el_tr_keep_short = 0, el_tr_n_inuse = 0;
    // Elastic RAM discards our band itself while the system swaps other apps out (opts selfdiscard=0|1, selfnear=SIZE)
    bool     el_sd_opt = true, el_sd_ok = false;
    int64_t  el_sd_near = (int64_t) 512 << 20;
    int64_t  el_sd_n = 0, el_sd_slots = 0, el_sd_n_far = 0;
    double   el_sd_ms = 0.0;
    bool     el_test_self_discard = false;     // TEST ONLY: MOE_STREAM_TEST_SELF_DISCARD
    void     el_self_discard(int64_t bytes);   // the elastic RAM thread (the lock per layer in short turns): `bytes` of our band discarded, coldest first
    int64_t  el_n_refill_val = 0;                  // idle refills by the decayed value (after the dropped list)
    bool     el_give_empty = false;                // mtx: el_give / el_drop_layer take switched-on EMPTY slots only (empties first)
    int64_t  el_tr_n_empty = 0, el_tr_empty_bytes = 0;   // keep-free shortfalls met first by switching off our empty slots
    int64_t  el_tr_kp = 0;                         // the last decision's keep-free part
    int64_t  el_tr_act_last = -1, el_tr_keep_last_got = 0, el_tr_keep_last_us = 0, el_tr_keep_hold_us = 0;   // the keep-free trim's "old" check   // keep-free trims, their MemFree gained, give-backs after them
    // The safety net: the engine never gives back below min= (opts min=SIZE; auto: a quarter of the start size
    // where the trim cannot run, the built-in floor 2k + 4 per layer where it can)
    int64_t  el_min_opt = -1;                      // -1 = auto
    int32_t  el_min_layer = 0;                     // its floor per streamed layer (el_floor)
    bool     el_test_discard = false, el_test_discard_nolock = false, el_test_unlock_every = false;   // TEST ONLY
    bool     el_test_thp_always = false, el_test_thp_keep = false, el_test_look_log = false;                                    // TEST ONLY
    bool     dz_unlock(llama_moe_stream_layer & sl, int32_t s);   // mtx: a filled, droppable slot becomes discardable
    void     dz_ahead(llama_moe_stream_layer & sl, int32_t s);    // mtx: a guess named the expert of slot s - taken back ahead if in the band
    void     dz_take_ahead(std::unique_lock<std::mutex> & lock);  // a reader, mtx held: the queued take-backs, one lock turn each
    bool     slot_use(llama_moe_stream_layer & sl, int32_t s);    // mtx: about to be READ - lock it (false: the OS took it, it is EMPTY now)
    void     dz_claim(llama_moe_stream_layer & sl, int32_t s);    // mtx: about to be WRITTEN - an ordinary slot again (never O_DIRECT into it)
    void     dz_forget(llama_moe_stream_layer & sl, int32_t s);   // mtx: its discardable state dropped (its pages go back with the caller)
    int64_t  dz_band(int64_t budget, int64_t * taken = nullptr, bool reconcile_win = false);  // elastic RAM thread (takes the lock per layer)
    double   el_band_pct = 20.0;                                  // opts band=: the discardable band, % of each layer's switched-on slots
    int64_t  el_band_pace = 64;                                   // opts bandpace=: the most slots the upkeep marks / unmarks per look
    bool     el_dz_touch = true;                                  // an O_DIRECT read into a claimed slot writes its pages first (Linux < 5.17)
    int64_t  el_psi_base = (int64_t) 128 << 20;    // opts psi=: a PSI event's give-back (doubling while the pressure goes on)
    int32_t  el_psi_run = 0; int64_t el_psi_last_us = 0;   // the PSI events in a row (each within 10 s of the last)
    int64_t  el_n_need_a = 0, el_n_need_pred = 0, el_n_need_psi = 0, el_n_swap_idle = 0;   // mtx: the give-backs by reason; logged swap-outs
    int64_t  el_target_rule() const { return p1.ram_margin + el_gap; }
    void     el_ask_r(int64_t bytes, int reason, int64_t allowed);   // mtx held: el_ask with its reason (the checks)
    bool     el_test_give_noreason = false, el_test_give_oversize = false;   // TEST ONLY: the failure tests
    bool     el_test_psi_off = false;   // TEST ONLY: MOE_STREAM_TEST_PSI_OFF - the PSI triggers not armed (identity runs)
    // Events + lightweight quarter-second headroom check (250 ms active/idle; 25 ms while free RAM falls >= fastrate)
    int32_t  el_idle_look_ms = 250;
    int64_t  el_fast_rate = (int64_t) 128 << 20;   // bytes/s
    int64_t  el_last_call_end_us = 0;              // mtx: the last call's end (idle = no call for 2 s)
    int64_t  el_n_looks_idle = 0;
    int64_t  el_last_obj_us = 0;                   // mtx: the last need that was no give-back (a PSI need, a fall): a take-back waits el_cool_s
    std::atomic<int64_t> el_quiet_until_us{0};     // while the fill at start runs and a short settle after it: no give-back or trim from rate estimates
    bool     el_call_known = true;                 // mtx: decode_phase describes the running call (false from el_enter to el_at_call_start)
    void     el_check_wave_need(const char * where) const;   // mtx held: the check - a prompt call's layers keep its wave width
    int32_t  el_test_fake_swap = 0;                // TEST ONLY: MOE_STREAM_TEST_FAKE_SWAP=N
    bool     el_test_give_one = false;             // TEST ONLY: the failure test
    int64_t  el_swap_look(int64_t pages, int64_t x);   // mtx held: logged (never answered); returns 0 (the failure test: a step)
    void     el_ask(int64_t bytes);                                               // mtx held: give back (now if idle, else at the call's end)
    int64_t  el_os_avail();                                                       // elastic RAM's MemAvailable (or the fake squeeze's)
    int64_t  el_os_pswpout();                                                     // elastic RAM's swap-out count (or the fake squeeze's)
    int64_t  el_os_own();                                                         // this process's resident bytes (or the fake squeeze's)
    // --moe-stream-elastic-opts pace / drop / lookmin / lookmax / ratewin: the looks paced by the other programs' rate
    double   el_pace = 2.0;                        // the safety factor on the rate (0 = unpaced: lookmax far, lookmin near)
    int64_t  el_drop = 0;                          // the most free RAM may fall between two looks (0 = step / 2)
    int32_t  el_look_min_ms = 25, el_look_max_ms = 250, el_rate_win_ms = 1000;
    bool     el_test_pace_off = false;             // TEST ONLY: MOE_STREAM_TEST_PACE_OFF (the failure test: unpaced, the check armed)
    int64_t  el_n_early = 0, el_n_fast_looks = 0;  // early give-backs; looks at the fastest pace
    double   el_rate_max = 0.0;                    // the fastest other-program rate seen (bytes/s)
    // An ask adds only the part still missing (el_ask); every boundary give through el_serve_pending (its check)
    void     el_serve_pending();                   // mtx held, a boundary: give what is pending
    int64_t  el_ask_peak = 0;                      // mtx: the largest single ask since nothing was pending (the check)
    int64_t  el_n_ask_cut = 0;                     // mtx: bytes the asks did not add (already pending)
    int64_t  el_pass_req = 0;                      // mtx: the pending at the start of a prompt ubatch's pass (its below-start shares)
    int64_t  el_n_prompt_hi = 0, el_n_prompt_lo = 0;   // mtx: slots prompt layer ops gave above / below the start size
    void     el_pend_note();                       // mtx held: the time from an ask until nothing is pending
    int64_t  el_pend_t0 = 0, el_pend_n = 0;
    double   el_pend_sum_ms = 0.0, el_pend_max_ms = 0.0;
    bool     el_test_pending_ignore = false;       // TEST ONLY: MOE_STREAM_TEST_PENDING_IGNORE (the failure test: every ask adds it all)
    int64_t  el_room();                                                           // the free room (MemAvailable, the cgroup's room)
    std::atomic<bool> el_taking{false};                                            // a take-back in steps is going on
    int64_t  el_memtotal = -1;                                                    // MemTotal, read at start-up
    int64_t  el_n_zero_done = 0, el_n_zero_skip = 0;                              // the low-only zeroing
    bool     el_test_lowneed_skip = false, el_test_zero_always = false, sh_test_lowfirst = false;   // TEST ONLY
    int64_t  el_low_keep() const;                                                 // the slots below the start size every layer keeps
    bool     el_givable_any() const;                                              // mtx: a slot some layer may give back
    void     el_release_slot(llama_moe_stream_layer & sl, int32_t s);             // mtx: its pages back to the system
    void     el_release_slots_batch(std::vector<std::pair<llama_moe_stream_layer *, int32_t>> & batch);
    int64_t  el_take_steps = 0, el_take_t0_us = 0, el_take_step_us = 0, el_n_trigger = 0;
    std::atomic<int64_t> el_looks{0};
    double   el_thread_cpu_s = 0.0, el_thread_wall_s = 0.0;
    int32_t  el_test_fake_squeeze = 0;                                            // TEST ONLY
    int32_t  el_test_fake_a0 = 0;                                                 // TEST ONLY: the pretend machine's MemAvailable at the first look (MiB)
    bool     el_test_give_ignore = false;                                         // TEST ONLY: the failure test
    std::atomic<int64_t> el_fk_t0_us{0}, el_fk_swp{0}, el_fk_swapped{0};
    int64_t  el_fk_a0 = 0, el_fk_given0 = 0, el_fk_taken0 = 0;
    bool     el_quiet_drop = false;                // mtx: the low-only drop (counted on its own line, not as an eviction)
    int32_t  el_refill_io = 2;                     // opts refillio=: refill reads in flight at most (behind every other read)
    int32_t  n_refill_run = 0;                     // mtx: refill pieces in a reader now
    void     el_refill_drain(llama_moe_stream_layer * only, int32_t s_lo, int32_t s_hi);   // mtx: queued refill reads taken back
    std::atomic<int64_t> el_n_ignored{0};          // pressure events with more than margin + step free (not a shortage)
    int64_t  el_n_lower = 0, el_n_raise = 0;       // zero-block moves (graph rebuilds)
    bool     el_min_dirty = false;                 // mtx: a prompt's layer op gave back; settled at the call's end
    int64_t  el_prompt_min = 0;                    // mtx: the share the layer op being served must keep
    int64_t  sh_moves = 0, sh_checks = 0, sh_next_word = 0;
    int32_t  sh_last_a = -1, sh_last_b = -1;
    std::map<std::pair<int32_t, int32_t>, int64_t> sh_pair_word;   // the last move of a pair (either direction)
    std::vector<std::string> sh_log;               // one line per move (printed at the end)
    std::thread el_thread;
    int      el_stop_fd = -1;
    int      el_pipe_fds[2] = { -1, -1 };
    void *   el_win_stop_ev = nullptr;
    void *   el_win_mem_ev  = nullptr;
    void *   el_mac_src     = nullptr;
    // TEST ONLY (read once at start-up): MOE_STREAM_TEST_ELASTIC_CHECK=1 checks every switch-off (eligible slots only) and every
    // share change (floors, ceilings, the sum, each share = its switched-on slots); _SHRINK_INUSE=1 lets the first give-back take
    // an in-use slot (the failure test); MOE_STREAM_TEST_FAKE_ALARM=N a pressure event every N decode words; _IGNORE_ALARM=1 drops the
    // requests (the failure test of the alarm check: 3 unanswered events stop the run); MOE_STREAM_TEST_SHARE_FORCE=1 a move at every check
    // (no gap / hits / twice / cool-down); _SHARE_FLOOR_BREAK=1 moves ignore the floor (the failure test of the floor check)
    bool     el_check = false, el_test_inuse = false, el_test_ignore = false, sh_test_force = false, sh_test_floor = false;
    int32_t  el_test_fake = 0;
    void     el_parse_opts();                      // constructor: the tunables and the test switches
    uint32_t el_cap_slots(uint32_t n_slots_start, size_t block_total, int32_t n_streamed, uint32_t n_expert) const;   // a pool's cap
    void     el_start();                           // the first call: the pressure thread
    void     el_stop();
    void     el_loop();                            // the pressure thread
    bool     slot_is_leased(const llama_moe_stream_layer & sl, int32_t s) const {
        return sl.keep[s] || sl.vf_mark[s] || (sl.slot_cb[s] & 1) != 0 ||
               sl.slot_pending[s] != 0 || sl.pl_bound_word[s] >= 0;
    }
    bool     el_droppable(const llama_moe_stream_layer & sl, int32_t s) const;   // mtx
    int32_t  el_floor(const llama_moe_stream_layer & sl) const;
    int32_t  el_ceil (const llama_moe_stream_layer & sl) const;
    int64_t  el_drop_layer(llama_moe_stream_layer & sl, int64_t n, bool record, int64_t low_min = 0, int32_t s_lo = 0, int32_t s_hi = -1,
                           bool empty_only = false);   // mtx: switch off the n least valuable (empty_only: only EMPTY slots)
    int64_t  el_grow_layer(llama_moe_stream_layer & sl, int64_t n, bool boundary); // mtx: switch on n (raise the zero block only at a boundary)
    void     el_move_zero(llama_moe_stream_layer & sl, uint32_t zb);              // mtx, boundary: the zero block to [zb, zb + n_zero)
    void     el_lower_zero(llama_moe_stream_layer & sl);                          // mtx, boundary
    int64_t  el_give(int64_t bytes, bool boundary, llama_moe_stream_layer * only);   // mtx
    struct el_mem { int64_t room = -1, avail = -1; };                            // read through the OS layer before the lock
    el_mem   el_read_mem();
    void     el_take(int mode, const el_mem & m);                                 // mtx, boundary: 0 a request, 1 a reply's end, 3 idle (elastic RAM thread)
    int64_t  el_os_memfree();
    int64_t  el_old_file_cache();   // Linux: file cache the kernel drops without a write (0 elsewhere)
    void     el_at_call_start(bool decode, int32_t n_tok);                        // set_decode
    void     el_at_call_end(bool complete);                                       // end_call, mtx held
    void     el_at_layer(llama_moe_stream_layer & sl, bool decode, int32_t wave, int64_t n_tok);   // remap_needed, mtx: a prompt's give-back
    void     el_finish_prompt_gives();                                            // mtx, boundary
    void     el_reply_end();                                                      // API: llama_moe_stream_idle
    // A prompt call's own host memory, outside the RAM plan, against the room under the memory limit
    int64_t  rg_need      = 0;   // the next call's own host memory (llama_context sets it just before set_decode)
    uint32_t rg_off       = 0;   // mtx: routed-expert slots per streamed layer given back for prompts' memory (switched off)
    int64_t  rg_wait      = 0;   // decode calls before the next take-back look (0 = at the next one)
    int64_t  n_rg_give    = 0, n_rg_back = 0;
    uint32_t rg_off_max   = 0;
    uint32_t rest_share() const { return n_slots - rg_off - np_off; }   // a layer's share at rest (the window's home size)
    void     rg_call_start(bool decode, int32_t n_tok);   // set_decode, after the window's call start
    // --moe-stream-nonpool-measured 1: the memory outside the pools from the run's own needs; a record next to the sidecar, one line per key
    bool        np_started = false;        // the first context's np_start ran
    std::string np_key, np_sidecar;        // the record's key; the sidecar's path
    int64_t     np_planned = 0, np_known = 0, np_growth = 0, np_srv_need = 0, np_cache_ram = 0;   // load: the plan's own part; weights in RAM; the terms on top
    int64_t     np_kv = 0, np_compute = 0, np_rest = -1, np_rest_saved = -1, np_peak = -1;        // measured
    int64_t     np_n = 0;                  // reply samples taken
    int32_t     np_wait = 0;               // decode calls before the next look
    uint32_t    np_off = 0;                // slots per layer switched off at the start check (never taken back)
    bool        np_pinned = false;         // --moe-stream-slots N: the check reports, it gives nothing back
    void        np_start(int64_t kv_host, int64_t compute_host);   // llama_context, after the first reserve
    void        np_call_start(bool decode);                        // set_decode
    int64_t     np_pool_ram();                                     // the pools' pages that hold RAM
    void        np_print() const;
    void     el_enter();                                                          // llama_context::decode()'s first step
    void     el_low_start(std::unique_lock<std::mutex> & lock);                   // prompt wave cap: a GPU-capable prompt call's start
    void     el_low_end();                                                        // prompt wave cap: the call's end, mtx
    void     el_low_call_start(int32_t n_tok);                                     // prompt wave cap: set_decode, after VRAM follow
    bool     el_gpu_view(int64_t u) const;                                       // prompt wave cap: may this ubatch's matmuls be GPU-copied
    bool     el_call_low(int64_t n_tok) const;                                    // prompt wave cap: may any ubatch of this call be
    void     el_recount();                                                        // mtx: total, min share
    void     el_check_shares(const char * where) const;                           // test only
    void     sh_tick();                                                           // end_call, mtx: slot shares' check every sh_every words
    void     sh_set_zone(llama_moe_stream_layer & sl);
    void     sh_apply_file();                                                     // first call, mtx
    void     el_print_stats() const;
    // The layers whose keep[] / slot_wave[] may hold marks since the last remap op cleared them - every writer of
    // a mark notes its layer here (pin_mark, mtx held), and the next op clears only these layers instead of all of them. The state
    // after the clear is the same as clearing every layer: a layer never noted holds no mark (both arrays start clean, and
    // drop_slot only resets slot_wave). MOE_STREAM_TEST_PIN_CHECK=1 (test only) verifies that at every op; MOE_STREAM_TEST_PIN_SKIP=N
    // (the check's failure test) drops the N-th note
    std::vector<int32_t> pin_list;
    bool    pin_check = false;
    int64_t pin_test_skip = 0, pin_notes = 0, pin_checks = 0;
    void pin_mark(llama_moe_stream_layer & x) {
        if (x.pin_listed) {
            return;
        }
        x.pin_listed = 1;
        if (pin_test_skip > 0 && ++pin_notes == pin_test_skip) {
            return;   // TEST ONLY (PIN_SKIP): this layer is never listed again (its marks are never cleared) - the check must stop the run
        }
        pin_list.push_back(x.il);
    }
    void pin_check_clean() const;   // TEST ONLY (PIN_CHECK)
    int32_t vf_test_slow_copy_us = 0;   // MOE_STREAM_TEST_VF_SLOW_COPY_US: the uploader sleeps this long before each copy
    int32_t vf_test_slow_us = 0;   // MOE_STREAM_TEST_VF_SLOW_READ_US: every guess read sleeps this long first (the I/O thread)
    int32_t vf_test_cb_hold_us = 0;   // MOE_STREAM_TEST_VF_CB_HOLD_US=N: the uploader holds a due copy-back until the next
                                      // call has started and N us more, so a reply's last copy-backs are still in flight at the next
                                      // prompt's drain (the server may run small calls of the prompt's tail first)
    std::atomic<int64_t> vf_call_seq{0};   // calls started (set_decode)
    int32_t vf_test_cb_stale = 0;  // MOE_STREAM_TEST_VF_CB_STALE=N: the N-th copy-back skips its host copy (the slot keeps
                                   // its old bytes) and settles READY - the read-back check on copied-back slots must stop the run
    std::atomic<int64_t> vf_cb_seen{0};   // copy-backs copied so far (the uploader / the sync path; for the failure test's N)
    // counters (engine-wide; decode = reply words)
    int64_t vf_words = 0, vf_ticks = 0, vf_swaps = 0, vf_switches = 0, vf_skip_loading = 0, vf_skip_notram = 0, vf_skip_marks = 0;
    int64_t vf_switch_wait = 0, vf_mark_mismatch = 0, vf_marked_prompt = 0, vf_src_ram = 0, vf_src_ssd = 0, vf_src_waited = 0;
    int64_t vf_verified = 0, vf_hysteresis_stop = 0;
    // copy-backs (taken, skipped: no slot / value not above the pool's lowest / the room cap), copied-back experts
    // picked again while in RAM (of them while the copy was in flight), evicted again unused, checked by the read-back check; at a
    // drained call the copy-backs still in flight (must stay 0); the keep value of every expert leaving VRAM (decay policy)
    int64_t vf_cb = 0, vf_cb_skip_room = 0, vf_cb_skip_value = 0, vf_cb_skip_cap = 0, vf_cb_picked = 0, vf_cb_late = 0;
    int64_t vf_cb_unused = 0, vf_cb_verified = 0, vf_cb_prompt = 0, vf_cb_drained = 0, vf_leave_n = 0, vf_cb_evict = 0;
    double  vf_leave_val = 0.0;
    std::atomic<int64_t> vf_copies{0};
    int64_t vf_copies_outside = 0;   // the uploader's copies that landed while no call ran (vf_mtx)
    // the GPU's tail of the word that ends a tick may still read a place the tick frees: everything on the device runs before the
    // first swap of a tick is decided (the CUDA backend's device-wide wait; the sampler would wait for the word anyway)
    bool (* vf_dev_sync)(ggml_backend_dev_t dev) = nullptr;
    ggml_backend_dev_t vf_dev = nullptr;
    int64_t vf_gpu_syncs = 0;
    // --moe-stream-vram-parallel 1: the device pass in the layer's GPU launch. On only when the
    // pool exists and the GPU producer is active (decided at vx_setup); the table is written at setup, at a tick (leaving experts) and at a
    // switch (incoming experts). A row is written stream-ordered on the calling context's GPU backend (vp_backend),
    // set by llama-context before every call), so it lands after that context's last graph and before its next one - no host wait,
    // nothing held under the engine lock; without a backend (no context yet) the old device-wide wait is the fallback
    bool    vp = false;
    ggml_backend_t vp_backend = nullptr;
    int64_t vp_writes = 0, vp_syncs = 0, t_vp_sync_us = 0, t_vp_write_us = 0, vp_skips = 0;
    int32_t vp_test_stale = 0, vp_switch_n = 0;   // MOE_STREAM_TEST_VP_STALE_TABLE=N (test only): skip the N-th incoming row write
    bool vp_on() const { return vp && vx_ready; }
    // [n_layer]: the layer's device pass runs in its GPU launch (decided at vx_setup): its router runs on the pool's GPU, or
    // --moe-stream-vram-parallel 2 lets the pass cross from another GPU; vp_n_launch = how many streamed layers with "router done" have it
    std::vector<uint8_t> vp_launch;
    int32_t vp_n_launch = 0;
    bool vp_in_launch(int32_t il) const { return vp_on() && il >= 0 && (size_t) il < vp_launch.size() && vp_launch[(size_t) il] != 0; }
    ggml_tensor * vp_table_of(int32_t il, size_t & offset) const {   // the group's table and this layer's byte offset in it
        const auto * sl = layer(il);
        const auto & G  = vx_groups[sl->vx_group];
        offset = (size_t) sl->vx_li*G.vp_table->nb[2];
        return G.vp_table;
    }
    void vp_row(llama_moe_stream_layer & sl, int32_t e, bool incoming);   // write expert e's row of its layer's table
    void vp_check();                                                      // TEST ONLY (VRAM follow verify): the tables = vx_slot
    bool vram_serves(int32_t il) const {
        const auto * sl = layer(il);
        return vx_ready && sl && sl->vx_group >= 0;
    }
    // Copy-ahead: the graph being built runs its waves through the emptied tail (tail_ubatch, before every graph is built or
    // reused): its device pass reads the views without the tail and names zero slices A (remap_vram); else the whole tensor, zero slices B
    bool          vp_tail_graph = false;
    ggml_tensor * vx_pool_gate_up(int32_t il) const { if (!vram_serves(il)) { return nullptr; } const auto & G = vx_groups[layer(il)->vx_group]; return vp_tail_graph ? G.vp_gate_up : G.gate_up; }
    ggml_tensor * vx_pool_down(int32_t il)    const { if (!vram_serves(il)) { return nullptr; } const auto & G = vx_groups[layer(il)->vx_group]; return vp_tail_graph ? G.vp_down    : G.down; }
    // the zero slices' first id as the graph's device pass names it (the views start at p0)
    // Front lending: the device pass's view of a group starts at vx_view_first - the first held place
    // in a graph through the emptied tail (else 0), plus the places lent at the front for this call where the view stays one block (a
    // tail graph of the tail group, any graph of a group without a tail; never on the vp_table paths, abs = true: absolute places)
    int32_t       vx_view_first(int32_t il, bool abs = false) const {
        if (!vram_serves(il)) { return 0; }
        const int32_t g = layer(il)->vx_group;
        const auto &  G = vx_groups[(size_t) g];
        const int32_t sh = !abs && (vp_tail_graph || G.n_tail == 0) && (size_t) g < lend_shift_g.size() ? lend_shift_g[(size_t) g] : 0;
        return (vp_tail_graph ? G.p0 : 0) + sh;
    }
    int32_t       vx_zero_first(int32_t il)   const { return vram_serves(il) ? vx_groups[layer(il)->vx_group].z_end() - vx_view_first(il) : -1; }
    int32_t       vx_id_base(int32_t il)      const { return vx_view_first(il); }
    // the device pass's weight tensor for this graph (part 0 = gate / up, 1 = down): the prebuilt tensor / view, or a view built in the
    // graph's context when the front lending shifts it
    ggml_tensor * vx_pool_view(ggml_context * ctx, int32_t il, int part, bool abs) const;
    std::vector<int32_t> lend_start_g;       // per group, this call's first place for the lending pieces (p0 + the front places lent to the call's tensors)
    std::vector<int32_t> lend_shift_g;       // per group, this call's places lent at the front (the call's tensors and the compute buffer): the view's shift
    void          lend_call_begin();         // llama_context, a call's start: no front places yet (graph_gen moves when the shift does)
    void          lend_set_shift(size_t g, int32_t v);
    // Parse the list file (constructor), allocate + fill the pool on the device and check the free VRAM (llama_context,
    // after the KV cache and the first compute-buffer reserve; once per model). Returns false when the feature is off.
    void vx_load_list();
    void vx_flags_off(const char * why);   // never refuse: the pool and what needs it off, said once
    void vx_free_pool();                   // after the pool was made
    void vx_lists_by_id(const char * why); // no usable experts file - the id order
    bool vx_pool_off = false;              // the pool was switched off by a fallback (the exit / report lines)
    bool vx_setup(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft, int64_t n_ubatch, int64_t n_ctx,
            const std::function<void()> & reserve_again = nullptr, const std::function<void()> & release_sched = nullptr);
    // The lazy VRAM that the load-time check can see, the auto-sized pool, the end-state line
    void vx_prewarm(ggml_backend_t backend, int64_t scratch_derived);          // before vx_setup: ask the CUDA backend for its cuBLAS handle + workspace now
    int64_t vx_growth_bytes(int64_t n_ubatch, int64_t n_ctx, std::string * why) const;   // the fit's reserve for what is still lazy
    void vx_end_state();                              // after a prompt: the free VRAM now against the load-time check
    bool        vx_auto        = false;     // --moe-stream-vram-free: N was chosen by the fit
    uint32_t    vx_list_min    = 0;         // the shortest list over the streamed layers (auto mode; with the id-order rest appended)
    uint32_t    vx_file_min    = 0;         // the shortest list in the file itself over the streamed layers (for the log)
    size_t      vx_file_part(int32_t il, size_t n) const { const auto it = vx_file_len.find(il); return it == vx_file_len.end() ? n : std::min(n, it->second); }
    uint32_t    vx_extra       = 0;         // the first vx_extra layers of the (single) pool group hold N + 1 places (auto pool: steps of one place, not of one place in every layer)
    ggml_backend_dev_t vx_dev  = nullptr;   // the pool's device, for the end-state line
    int64_t     vx_free_check  = 0;         // free VRAM the placement line (vx_report) read after the pool and the 2nd reserve, bytes
    int64_t     vx_keep_ask    = 0;         // the end-state keep-free asked for (--moe-stream-vram-free), bytes, 0 = none
    int64_t     vx_growth_used = 0;         // the growth the fit reserved, bytes
    bool        vx_prewarmed = false;   // vx_prewarm created the backend's cuBLAS handle before the VRAM check
    int64_t     vx_scratch_premapped = 0;   // bytes the op scratch pool took when vx_prewarm mapped it (0 = not mapped at start)
    int64_t     vx_first_reserve = 0;       // the size the first compute buffers had when they were released before the pool (log only)
    int64_t     vx_scratch_lazy = 0;        // what the growth sum counts for the scratch pool when it is not mapped at start (the derived size, also with --moe-stream-vram-scratch 0)
    int64_t     vx_scratch_want = 0;        // the scratch the biggest prompt batch needs (derived or --moe-stream-vram-scratch), bytes
    bool        vx_graphs_on = true;        // the backend uses CUDA graphs (their instance chunks are lazy VRAM)
    bool        vx_pool_vmm = true;         // the backend's op scratch pool is the VMM pool
    size_t    (*vx_pool_held)(ggml_backend_t, size_t *) = nullptr;   // the legacy scratch pool's held bytes (backend entry; nullptr = none)
    ggml_backend_t vx_pool_backend = nullptr;
    std::string vx_scratch_why;             // how vx_scratch_want was chosen
    int64_t     vx_end_min_free = INT64_MAX; // the lowest free VRAM seen after a prompt
    bool        vx_end_seen    = false;
    void vx_report(ggml_backend_dev_t dev);   // the placement line after the compute buffers, and the margin check

    // llama_context says whether the tokens of the batch it is about to process continue a
    // sequence (a decode step) or fill one (prompt processing); the remap op files its counters,
    // its stall and its bytes under that name, so that a per-token figure is unambiguous. The
    // batch's token count goes here too, so that a token is counted once and not once per layer
    // (the per-layer counters made a per-token figure 40x too small)
    //
    // Ctx is the llama_context the batch belongs to. The pools, the pins and the counters belong
    // to the model, so two contexts of one model share them: the engine says so once, when it sees the
    // second one, because it can only be correct while their graphs do not overlap (llama-server runs
    // every slot on one thread and an MTP draft decodes after its target, both verified).
    void set_decode(bool decode, int32_t n_tok = 0, const void * ctx = nullptr) {
        if (!decode) { moe_tt("E set_decode_b", n_tok); }   // TEST ONLY (prompt calls)
        if (ca_test_abort_left > 0 && !ca_test_abort_set && ctx != nullptr) {
            ca_test_abort_install(ctx);   // TEST ONLY (MOE_STREAM_TEST_CA_ABORT)
        }
        if (ahead_on) {   // a keep a call never released (no end_call) does not reach this one
            std::lock_guard<std::mutex> lock(mtx);
            ahead_release_all();
        }
        if (psplit_on) {
            psplit_call_start();
        }
        win_call_seq  += 1;      // a window pass belongs to one call
        win_call_ntok  = n_tok;
        rec_batch_start();   // when llama_decode accepted this batch
        call_seq += 1;               // the call that p1_pe_w values belong to
        p1_prompt_end_call = false;  // set by set_prompt_end(true) during the call
        if (rec_ev_on) {
            rec_word.store(decode ? n_tok_outer : -1, std::memory_order_relaxed);   // before this batch is counted
        }
        // A new prompt number only when a prompt call follows a decode call (or is the
        // first call), so a prompt split into several calls (e.g. 4k tokens at -b 2048) keeps one number
        b0_prompt_seq += !decode && (decode_phase || b0_prompt_seq == 0) ? 1 : 0;
        const bool vx_was_decode = decode_phase;
        decode_phase = decode;
        if (decode && !vx_was_decode && vx_ready) {
            vx_end_state();   // once per request, when its prompt is done: what the VRAM looks like at the end state
        }
        // The helper takes the guesses only of a decode call of one token - the word number moves once per call,
        // so in a call of several 1-token ubatches (a speculative / MTP verify of one sequence at -ub 1) the second ubatch would
        // post for a target the first one already started; every other call shape keeps the direct add
        ne_call_one = decode && n_tok == 1;
        if (ne_on && ne_call_one) {
            ne_call_start();   // this call's number, before its graph is enqueued
        }
        (decode ? n_tok_outer : n_tok_outer_pre) += n_tok;
        for (auto & slp : layers) {
            if (slp) {
                slp->decode_phase = decode;
            }
        }
        if (ctx != nullptr && ctx != ctx_last) {
            if (ctx_last != nullptr) {
                n_ctx_shared = n_ctx + 1;   // the caller says it once, llama-context.cpp has the logger
            }
            if (n_ctx < LLAMA_MOE_STREAM_MAX_CTX) {
                ctx_seen[n_ctx++] = ctx;
            }
            ctx_last = ctx;
        }
        if (el_on || sh_on) {
            el_at_call_start(decode, n_tok);   // a give-back / take-back at a call's start (before its graph)
        }
        if (vf_call_start_fn != nullptr) {
            // TEST ONLY: the tick's event at the start of the word whose end ticks (too early) - only that call, so a
            // later call's record cannot stand in for it (that would be after the word's GPU work again)
            if (vf_test_early_ev && vf_ev_on && vf_ctx_be != nullptr && decode && n_tok == 1 &&
                    (vf_words + 1) % p1.vram_follow_tick == 0) {
                ggml_backend_event_record(vf_ev, vf_ctx_be);
            }
            vf_in_call.store(true);
            if (vf_test_cb_hold_us > 0) {
                vf_call_seq.fetch_add(1);   // test only (CB_HOLD_US): a new call has started
            }
            (this->*vf_call_start_fn)(decode, n_tok);   // VRAM follow: word boundary 2 - the switches, before this call's graph
            if (!decode) { moe_tt("E vf_call_start", n_tok); }
        }
        if (win_on) {
            win_call_start(decode, n_tok);   // before the prompt wave cap's call start (both read the window-call rule)
            if (!decode) { moe_tt("E win_call_start", n_tok); }
        }
        if (!el_on && !sh_on) {
            rg_call_start(decode, n_tok);   // slots given back for a prompt's own memory / taken back (after the window's restore)
        }
        if (p1.nonpool) {
            np_call_start(decode);
        }
        if (el_on || sh_on || win_on) {
            el_low_call_start(n_tok);   // prompt wave cap: after VRAM follow's call start (a prompt call drains its marks / copy-backs)
            if (!decode) { moe_tt("E el_low_call_start", n_tok); }
        }                               // (a window call can leave kept experts above the start size)
        if (p1.guess_skip_full) {
            gsf_call_start(decode, n_tok);   // which target layers are full this word (after the give-back / take-back)
            if (p1.guess_skip_full_gpu) {
                g9_call_start();   // the bytes to the GPU, stream-ordered before this call's graph
            }
        }
        if (oracle_list && ne_on && ne_call_one) {
            ne_oracle_list_start();   // the oracle waiting-list test (test only): after every other call-start hook
        }
        if (!decode) { moe_tt("E set_decode_e", n_tok); }
    }
    bool is_decode() const { return decode_phase; }
    uint64_t call_seq = 0;             // llama_decode calls so far
    bool     p1_prompt_end_call = false;   // a prompt-end ubatch (or expert graph) ran in this call

    // llama_context says, before every ubatch, whether it is the last ubatch of a prompt (see its rule
    // in llama_context::decode). Only then is the order of the ubatch's waves worth choosing for the reply.
    void set_prompt_end(bool end) {
        static const bool test_log = getenv("MOE_STREAM_TEST_PE_LOG") && atoi(getenv("MOE_STREAM_TEST_PE_LOG")) != 0;   // TEST ONLY
        if (test_log && end) {
            fprintf(stderr, "llama_moe_stream: PE_LOG the prompt's end marked (call %llu)\n", (unsigned long long) call_seq);
        }
        p1_prompt_end_ub = end;
        p1_prompt_end_call = p1_prompt_end_call || end;
        n_p1_prompt_end_ub += end ? 1 : 0;
    }

    bool prefetch_on() const { return do_prefetch; }

    std::vector<llama_moe_stream_wave> pred2_ctx;   // one per layer (the two-ahead early guess)
    std::vector<llama_moe_stream_wave> pred3_ctx;   // one per layer (the three-ahead guess)

    uint32_t n_slots = 0;
    uint32_t el_cap  = 0;   // the pools' cap (slots in the pool tensor); = n_slots with the features off
    // GPU prompt waves follow the current size. VRAM stays fixed at start: the GPU prompt-wave
    // view (and its compute buffer) is reserved for the widest a layer can reach - el_wide_w = the pools' cap, at most the layer's
    // experts; the waves follow the current smallest share up to it (more slots = fewer waves); no low-only calls. If that reserve
    // fails, the start size (low-only calls for a grown layer)
    bool     el_wide = false, el_wide_opt = false; // opts wide=0|1 (off by default)
    uint32_t el_wide_w = 0;
    bool     el_test_wide_fail = false;            // TEST ONLY: MOE_STREAM_TEST_WIDE_FAIL (the wide reserve "fails": the fallback)
    int64_t  wave_view_slots() const { return el_wide ? (int64_t) el_wide_w : (int64_t) n_slots; }
    void     el_wide_off(const char * why);        // the fallback to the start size
    uint32_t el_lo_end(const llama_moe_stream_layer & sl, bool prompt) const;   // the slots a prompt's guard counts
    // a call that must keep its slots like a prompt: a prompt, a call whose kind is not known yet, or a decode call of more
    // than one token - MTP verify / speculative calls need every token's experts, so an in-call give-back may not cut a layer below that
    bool     el_prompt_now() const { return el_in_call && (!decode_phase || !el_call_known || el_call_ntok > 1); }
    int32_t  el_call_ntok = 0;                     // mtx: the running call's tokens (el_at_call_start)
    uint32_t n_expert_total = 0;
    uint32_t n_expert_used = 0;
    // the all-zero slices every pool gets on top of its slots: one per pair of a token, so that the
    // ids of a wave can give every pair outside it a slice of its own
    uint32_t n_zero = 0;
    uint32_t max_waves = 0;  // the cap of n_waves(), 0 = none
    uint32_t prefetch_m = 8;
    // --moe-stream-guess-top K (0 = off). With K > 0 a graph of one token builds its 1-ahead guess K wide (the decode
    // words the waiting list serves); every other graph keeps prefetch_m. The graph (build_moe_ffn), the remap op's shape check
    // and its call of prefetch_next / ne_post all take the width from guess_width(), so they cannot disagree.
    uint32_t guess_top = 0;
    // --moe-stream-lookahead N (1 = the default, nothing below runs) and --moe-stream-guess-pace B; the list's order is one
    // comparator picked at start-up (order B, order A = --moe-stream-plan-extra-near 1, the old order = --moe-stream-plan-old-order 1)
    uint32_t lookahead  = 1;
    uint32_t guess_pace = 1;
    static constexpr uint32_t LOOKAHEAD_MAX = 8;
    static constexpr uint32_t GUESS_TOP_MAX = 32;
    uint32_t guess_width(int64_t n_tokens) const { return guess_top > 0 && n_tokens == 1 ? guess_top : prefetch_m; }
    // The targets of layer il's guesses with lookahead N - the distances d = 1..N whose layer il + d is a streamed
    // layer, in order (a VRAM or MTP layer is skipped, past the last layer stops). One column of the remap op's guess input per
    // target, in this order: the graph builds exactly these (stopping at the first missing router) and the engine reads them
    // the same way, so no column is built for a layer that is not streamed. Returns the count; ds gets the distances.
    int32_t la_targets(int32_t il, int32_t * ds) const {
        int32_t n = 0;
        for (int32_t d = 1; d <= (int32_t) lookahead; d++) {
            const int32_t X = il + d;
            if (X >= (int32_t) layers.size()) {
                break;
            }
            if (layers[X]) {
                ds[n++] = d;
            }
        }
        return n;
    }
    // The columns of layer il's decode guess (and of its slot that the producer and ne_post fill)
    int32_t la_cols_of(int32_t il) const {
        int32_t ds[LOOKAHEAD_MAX];
        return la_on ? la_targets(il, ds) : 1;
    }
    // How many MoE layers of lead the prefetcher has, and the width of the correction that the
    // layer in between applies to the early guess. lead 1 is the default (R(1,m)); lead 2
    // issues R(2,m) for layer L+2 at layer L and refines it with R(1,m2) at layer L+1, cancelling the
    // early reads that have not started and that the refinement does not name (0.760 recall at
    // (8,8) and 0.855 at (16,12), against 0.757 for lead 1 at M=8).
    uint32_t prefetch_lead = 1;
    uint32_t prefetch_m2   = 0;   // 0 = the same width as prefetch_m
    // The two queue policies, each behind a flag so it can be measured alone. By default a
    // demand read always overtakes a queued prefetch (the worker pops q_demand first), so
    // demand_last *reverses* that to measure what the priority is worth; probation makes a prefetched
    // slot that nobody used the first eviction victim (by default pick_victim protects them).
    bool     demand_last  = false;
    bool     probation    = false;
    // Make this layer's expert matmul *depend* on the early-guess op, so the guess is issued
    // before the layer's own expert work instead of whenever the scheduler finds it ready
    bool     early_order  = false;

    // --moe-stream-plan 1: the waiting list. A decode word's 1-ahead guess adds an
    // entry per named expert that is not in its target's pool; a reader thread binds the first entry when it has nothing else
    // to do (takes the victim slot then, not at queue time); the target's remap removes the entries it picked, and the word's
    // end removes the rest, so the list is empty at every word boundary. Guarded by mtx; capacity fixed at start-up.
    struct pl_entry {
        int32_t il;      // target layer
        int32_t e;       // expert
        int32_t lead;    // layers ahead of the layer whose guess added it
        int32_t rank;    // its true rank in that guess (0 = best)
        int64_t word;    // n_tok_outer when it was added
        int64_t t_add;   // when it was added (ggml clock)
        uint8_t bh_skip = 0;   // a reader skipped it while the drive was behind (counted once)
        // With lookahead N > 1: copied from the target's marks before every sort (lead / rank = the latest distance / rank)
        uint8_t  votes = 1;
        uint8_t  live  = 1;
        uint32_t seq   = 0;    // the add order: the sort's last key, so the order is deterministic
        uint8_t  ws    = 0;    // added by the word-start op (its counters)
    };
    std::vector<pl_entry> pl_list;
    size_t                pl_cap = 0;
    // The model's last trunk layer: a graph cuts it to the call's output rows before its experts, so in a call that asks
    // for no output row its experts never run - its list entries of that word are dropped at the word's end, not a bug
    // (llama-server sends such a call of one token when it cuts a prompt's last 4 tokens off for a context checkpoint and
    // one uncached token comes before them; with pos > 0 and one token it counts as a decode word)
    int32_t               pl_il_crop = -1;
    int32_t               call_n_out = -1;   // the output rows the current call asks for (llama_context sets it before set_decode; -1 = not told)
    bool                  plan_on = false;
    bool                  drop_stale = false;   // --moe-stream-drop-stale 1 (with plan 1)

    // --moe-stream-needed-early 1: the guess hand-over to the helper thread.
    // The slot buffer: one slot per model layer of ne_stride int32 = the 1-ahead guess's ids in rank order (ne_width = the decode
    // guess's width guess_width(1): --moe-stream-prefetch-m, or K with --moe-stream-guess-top K - the remap op's second
    // input in a one-token call; the GPU producer's GPU writer copies the same number of ids), then one uint32 call counter (= the producer's
    // consumed count + 1 when posted). Allocated once at start-up; ne_slot() is the one accessor (the GPU producer makes it pinned and
    // the GPU writes it). The host producer is the remap op itself (a host-side stand-in), its signal a condition variable.
    bool                  ne_on = false;
    // Lookahead N > 1: a slot holds ne_cols = N columns of ne_width ids - the remap op's guess input, one column per
    // la_targets(il) target in that order (a layer uses only its first la_targets(il) columns), then the counter at the same
    // place in every slot (after N columns); the GPU producer's GPU writer copies the same layout
    int32_t               ne_width = 0, ne_cols = 1, ne_stride = 0;
    std::vector<int32_t>  ne_buf;
    int32_t *  ne_slot(int32_t il) { return (ne_pin != nullptr ? ne_pin : ne_buf.data()) + (size_t) il*ne_stride; }
    uint32_t & ne_counter(int32_t il) { return *reinterpret_cast<uint32_t *>(ne_slot(il) + (size_t) ne_cols*ne_width); }
    bool                  la_from_helper = false;   // la_guesses runs for the helper (on_guess_ready): the too-late skip applies
    std::vector<int32_t>  ne_q;          // posted layers, in posting order (a ring, capacity n_layer)
    size_t                ne_q_head = 0, ne_q_n = 0;
    int32_t               ne_pending = 0;   // posted and not yet consumed
    bool                  ne_stop = false;
    bool                  ne_call_one = false;   // the current call is a decode call of one token (set_decode)
    std::condition_variable ne_cv;          // a slot was posted, or the helper must stop
    std::condition_variable ne_done_cv;     // the helper consumed the last pending slot
    std::thread           ne_thread;
    int32_t               ne_test_delay_us = 0;   // MOE_STREAM_TEST_HELPER_DELAY_US (test only, read once at start-up)
    int                   rec_ev_buf_cur = 0;     // the recorder buffer list events go to (the helper: its own)
    int                   rec_ev_buf_helper = 0;
    int64_t               n_ne_drain_calls = 0, t_ne_drain_us = 0;   // end_call had to wait for the helper
    template <bool DELAY> void ne_loop();          // the helper thread
    void   ne_wait_posted(std::unique_lock<std::mutex> & lock);   // the condition variable; the GPU producer: the event
    void   ne_post(llama_moe_stream_layer & sl, const int32_t * pred, int64_t n_pred);   // mtx held (the stand-in producer)
    void   on_guess_ready(int32_t il);             // mtx held: the plan-1 adds, or a skip when the target has started

    // The GPU producer - the GPU writes the guess into the layer's slot (now pinned host memory,
    // the GPU device's host buffer type), then the counter = this call's number, then records the layer's "guess done" event.
    // The remap op only says "layer L enqueued" (ne_mark), the helper waits on the event (ne_wait_posted) and checks the counter
    // as in the host producer. Set up once the weights are placed; if pinned memory, the backend's mark functions or a single GPU device
    // are missing, the host producer stays (a start-up line says which).
    // The counter rule of both stages compares the slot with ne_call_seq, a number that moves once per
    // one-token decode call with needed-early on (ne_call_start), so a slot left by an aborted call can never pass.
    bool                  ne_gpu = false;
    ggml_backend_dev_t    ne_mark_dev = nullptr;   // the registration's GPU (the marks' only GPU with the one-card rule)
    // per layer, the GPU whose stream runs its marked node (a guess / word-start mark: where its targets' routers are; a router mark:
    // its router's GPU; null: no mark of that kind). Every marked node is pinned to it; one GPU: all equal ne_mark_dev
    std::vector<ggml_backend_dev_t> ne_dev_guess, ne_dev_router;
    ggml_backend_dev_t    ne_guess_dev(int32_t il) const  { return il >= 0 && (size_t) il < ne_dev_guess.size()  && ne_dev_guess[(size_t) il]  ? ne_dev_guess[(size_t) il]  : ne_mark_dev; }
    ggml_backend_dev_t    ne_router_dev(int32_t il) const { return il >= 0 && (size_t) il < ne_dev_router.size() && ne_dev_router[(size_t) il] ? ne_dev_router[(size_t) il] : ne_mark_dev; }
    uint32_t              ne_call_seq = 0;         // mtx: this call's number (0 = no call yet; the slots start at 0)
    ggml_backend_buffer_t ne_pin_buf = nullptr;
    int32_t *             ne_pin = nullptr;        // [n_layer_all * ne_stride]: ids, then the counter (the same layout as ne_buf)
    uint32_t *            ne_call_cell = nullptr;  // pinned: ne_call_seq, read by the GPU producers when they run
    bool (*ne_mark_wait)(int32_t, int32_t)    = nullptr;   // the backend's "wait for layer L's event of a kind"
    void (*ne_mark_unregister)(const void *)  = nullptr;
    bool (*ne_mark_sync)(void)                = nullptr;   // every producer device ran everything enqueued
    bool   ne_gpu_on() const { return ne_gpu; }
    // after ne_set_router_gpu: router_dev[il] = the GPU device that runs layer il's router (nullptr: not a GPU)
    void   ne_gpu_setup(const std::vector<ggml_backend_dev_t> & router_dev);
    void   ne_oracle_list_start();                         // the oracle waiting-list test (test only)

    // --moe-stream-guess-skip-full 1: the shared "full" array - one byte per model layer (index = the model layer; 0 for a
    // layer that is not streamed): 1 = the layer holds every expert in RAM, READY (n_slots >= n_expert, every expert_slot[e] a READY
    // slot). Written only here, under mtx, at the start of every call (set_decode's call-start hooks, after the give-back / take-back,
    // before the call's graph runs); constant while a graph runs; all 0 in a prompt call. Host memory, pinned for the GPU when there is
    // one (a GPU device's host buffer type), allocated once, never moved: a graph may carry its address. Readers: the CPU's combined
    // guess (MUL_MAT_MULTI) and its top-k skip the rows of full targets (llama-graph.cpp passes the address and the rows' targets in
    // their op params); the GPU skip-full guess will read the same bytes
    uint8_t *             gsf_full     = nullptr;
    // --moe-stream-guess-skip-full-gpu 1: the 'full' bytes on the GPU - one byte per model layer, copied from gsf_full at every
    // call's start on the calling context's GPU stream (vp_backend), before the call's graph; the guess kernels read them there
    ggml_context_ptr        g9_ctx;   // (freed with the engine, by the owning pointers)
    ggml_backend_buffer_ptr g9_buf;
    ggml_tensor *         g9_t   = nullptr;
    uint8_t *             g9_dev = nullptr;
    uint8_t *             g9_dev_ptr() const { return g9_dev; }
    void                  g9_call_start();
    ggml_backend_buffer_t gsf_buf      = nullptr;   // the pinned host buffer behind gsf_full (GPU machines)
    std::vector<uint8_t>  gsf_host;                 // or plain memory (no GPU)
    bool                  gsf_test_all  = false;    // TEST ONLY: MOE_STREAM_TEST_GUESS_FULL_ALL=1 - every streamed layer counts as full
    int64_t               gsf_test_word = -1;       // TEST ONLY: MOE_STREAM_TEST_GUESS_EMPTY_AT=W:L - at reply word W, layer L loses one
    int32_t               gsf_test_layer = -1;      //   expert (drop_slot) if it is full, or (with FULL_ALL) stops counting as full
    int64_t               gsf_words = 0, gsf_full_sum = 0, gsf_full_max = 0, gsf_changes = 0;
    int64_t               gsf_rescans = 0, gsf_t_us = 0, gsf_lag_skipped = 0;   // layers looked at again, call-start time, la_guess skips
    bool                  gsf_test_check  = false;  // TEST ONLY: MOE_STREAM_TEST_GSF_CHECK=1 - every call start compares the cached answer with a rescan
    bool                  gsf_test_rescan = false;  // TEST ONLY: MOE_STREAM_TEST_GSF_RESCAN=1 - every layer looked at again at every call (as first built)
    uint8_t *             gsf_ptr();                // the array (allocated on first use; the same address for the whole run)
    void                  gsf_call_start(bool decode, int64_t n_tok);
    bool                  gsf_layer_full(const llama_moe_stream_layer & sl) const;
    void   ne_call_start();                                // a one-token decode call starts (set_decode)
    void   ne_mark(llama_moe_stream_layer & sl);           // mtx held: the remap op says "layer L enqueued for this call"

    // The GPU producer's "router done". In a one-token decode graph of a GPU-producer call the
    // router's ids of a layer whose router runs on the producer's GPU carry LLAMA_MOE_STREAM_TENSOR_ROUTER_MARK and, as src[1],
    // the router input the CPU experts read; right after that node the GPU writes both into the layer's pinned "router" slot,
    // then the call's number, then records the layer's "router done" event. The guess and (with --moe-stream-pred-shared) the
    // shared expert come after it in the same GPU split, and the remap op no longer takes the guess, so the scheduler's input hook
    // (rd_input_hook) can give the layer's CPU split its inputs from the slot as soon as the router is done, while the GPU still
    // runs the shared expert and the guess.
    bool                  ne_rd = false;            // router slots registered
    std::vector<uint8_t>  ne_rd_layer;              // [n_layer_all]: the layer has a router slot
    uint8_t *             ne_rd_slots = nullptr;    // pinned: [n_layer_all * ne_rd_cap]
    uint32_t *            ne_rd_ctr   = nullptr;    // pinned: [n_layer_all * 16] (one 64-byte line per layer; [il*16] is used)
    int64_t               ne_rd_cap   = 0;          // bytes of one router slot
    // --moe-stream-prompt-split: a CPU-wave prompt ubatch of 2 .. ps_rows rows hands its CPU split the
    // router's ids, input and guess through one shared pinned slot (the router mark, as a one-token GPU-producer call does), and its VRAM
    // pass is built in the layer's GPU launch right after the router, so the GPU runs it while the CPU split computes the rest
    uint8_t *             ps_slot = nullptr;
    int64_t               ps_cap  = 0, ps_rows = 0;
    bool                  ps_ub   = false;          // llama_context, per ubatch: this ubatch is a prompt-split one (the hook's rule)
    int64_t               ps_ubatches = 0, ps_splits = 0;
    bool                  ps_unfit_said = false;
    bool ps_wants(int64_t n_tok) const {            // the graph builder and ps_ubatch: the same rule
        return ps_slot != nullptr && ne_gpu && vp_on() && vp_n_launch > 0 && !decode_phase && n_tok > 1 && n_tok <= ps_rows && !waves_on_gpu(n_tok);
    }
    // --moe-stream-gpu-wave-skip-zero: n_zero when the LAST n_zero experts of this weight tensor (a GPU wave's tail tensor or
    // g1-ahead set view, a layer's pool or its start-size view, the VRAM pool) are the all-zero slices, else 0 (then the matmul is not marked)
    int32_t zero_tail_of(const ggml_tensor * w, int32_t il) const;
    int64_t n_zt_marked = 0, n_zt_refused = 0;
    bool ps_fits(int32_t il, const ggml_tensor * ids, const ggml_tensor * hidden, const ggml_tensor * guess, const ggml_tensor * w);   // the slot holds them
    void ps_ubatch(int64_t n_tok) {                 // llama_context, before each ubatch's graph: a new call number per such ubatch (the
        ps_ub = ps_wants(n_tok);                    // slot's counter must not pass with an earlier ubatch's write)
        if (ps_ub) {
            ps_ubatches += 1;
            ne_call_start();
            psplit_ubatch_start();
        }
    }
    // In a prompt-split ubatch, at layer L's remap op the next streamed layer's
    // guessed experts (prompt_guess, most-used first) that are READY (expert_ready, kept until that layer's op) - k of them,
    // k = E x c / (c + d) with c the CPU time per expert measured on that layer's last CPU split and d the uploader's copy time per
    // expert - are copied by the VRAM-follow uploader into one half of a ring of borrowed tail places (tail_borrow, once per call) while L's
    // CPU split runs. At that split's end (the scheduler's split hook) the compute stream waits for the copies, L's held set and table
    // rows are cleared and L+1's set: its VRAM pass (built beside its CPU split by the prompt split) computes them, its CPU split skips them
    bool    psplit_build = false;                       // vx_setup: the setting + a tail + the VRAM-follow uploader + the VRAM table
    std::vector<int32_t> psplit_ring;                   // the places borrowed for this call (two halves)
    struct psplit_job_t {
        int32_t il = -1;
        std::vector<std::array<int32_t, 3>> list;   // (expert, RAM slot, place)
        int64_t seq = 0, us = 0;
        bool    queued = false, issued = false;     // vf_mtx
    };
    psplit_job_t psplit_job;
    int64_t  psplit_seq = 0;
    int32_t  psplit_half = 0;                           // the ring half the next job copies into
    ggml_backend_event_t psplit_ev = nullptr;           // recorded on the uploader's stream after a job's copies
    int32_t  psplit_held_il = -1;                       // the layer whose experts are held in borrowed places now
    std::vector<std::pair<int32_t, int32_t>> psplit_held;   // (expert, place) of that layer
    std::vector<std::pair<int32_t, int32_t>> psplit_g;  // scratch: prompt_guess
    std::vector<int32_t> psplit_cand, psplit_slots;         // scratch
    std::vector<double>  psplit_c_us;                   // [n_layer] CPU time per expert of the layer's last CPU split (us)
    double   psplit_d_us = 220.0;                       // the uploader's copy time per expert (us), measured
    int32_t  psplit_remap_il = -1;                      // the layer whose CPU split runs now (set at its wave-0 remap op)
    int64_t  psplit_t_remap = 0, psplit_remap_cpu_e = 0;
    int64_t  n_psplit_jobs = 0, n_psplit_copied = 0, n_psplit_wanted = 0, n_psplit_not_ready = 0, n_psplit_wait_us = 0, n_psplit_up_us = 0, n_psplit_held = 0;
    void     psplit_ubatch_start();                     // ps_ubatch: the call's ring (once)
    void     psplit_after_remap(llama_moe_stream_layer & sl, int64_t n_tok, int32_t wave);   // graph thread, after a remap op
    void     psplit_split_end(ggml_backend_t be);       // the scheduler's split hook (a split was issued; on the CPU: computed)
    void     psplit_run_job(psplit_job_t & j);              // the VRAM-follow uploader thread
    void     psplit_finish_job(bool hold);              // wait for the job's copies (host), the compute stream waits for them; hold = name them
    void     psplit_release_held();                     // the held layer's experts back to RAM experts (vx_slot, table rows)
    void     psplit_end_call();                         // end_call, before the borrowed places go back
    bool     psplit_holds(int32_t il, int32_t e) const {   // e is held in a borrowed place for layer il now (it keeps its RAM copy)
        if (il != psplit_held_il) {
            return false;
        }
        for (const auto & h : psplit_held) {
            if (h.first == e) {
                return true;
            }
        }
        return false;
    }
    bool     psplit_no_cb = false;                      // llama_context: an eval callback is set (its split hooks fire mid-layer): no prompt split
    bool     psplit_said  = false;
    // the counters (main thread; decode calls): CPU splits given their inputs from a slot, of them with the number already there
    // (no CUDA call) / after the event wait, the time in the hook, the splits that took the plain path because an input was not in
    // the slot; the helper's waits on "guess done" by the same rule
    int64_t               n_rd_split = 0, n_rd_ready = 0, n_rd_event = 0, t_rd_us = 0, n_rd_uncovered = 0;
    int64_t               n_ne_ready = 0, n_ne_event = 0;
    bool                  rd_uncovered_said = false;
    // Graph builds that left a layer without the router mark because its inputs do not fit the slot (not contiguous,
    // not 4-byte, too large), counted per layer and build; said once
    int64_t               n_rd_unfit = 0;
    bool                  rd_unfit_said = false;
    // the graph builder's checks of a router mark: true = mark it; false = the plain path for that layer (counted, said once)
    bool   rd_fits(int32_t il, const ggml_tensor * ids, const ggml_tensor * hidden);
    // the graph of the current call is a GPU-producer one-token decode graph (its marks; graph reuse keys on it, llama-context)
    bool   ne_gpu_graph() const { return ne_gpu && ne_call_one; }
    bool   ne_rd_layer_on(int32_t il) const { return ne_rd && il >= 0 && (size_t) il < ne_rd_layer.size() && ne_rd_layer[il] != 0; }
    uint32_t * rd_counter(int32_t il) { return ne_rd_ctr + (size_t) il*16; }
    // the scheduler's input hook (ggml_backend_sched_set_input_hook): a CPU split whose inputs hold a router mark gets them all
    // from the layer's router slot (input_copies == nullptr: only "are they all there"); false = the plain path
    bool   rd_input_hook(ggml_backend_t split_backend, int n_inputs, ggml_tensor ** inputs, ggml_tensor ** input_copies);
    bool                  vf_hook = false;   // the VRAM follow version of pick_victim (MOE_STREAM_TEST_VF_HOOK_ON=1: test only)
    // what an added entry does: wait on the list (late binding) or bind at once (--moe-stream-plan-bind-early, test only);
    // chosen once at start-up
    void (llama_moe_stream::*pl_on_add)(llama_moe_stream_layer & nx, const pl_entry & en) = nullptr;
    void   pl_add_late (llama_moe_stream_layer & nx, const pl_entry & en);   // mtx held
    void   pl_add_early(llama_moe_stream_layer & nx, const pl_entry & en);   // mtx held
    void   pl_bind(llama_moe_stream_layer & nx, const pl_entry & en, int32_t s, int buf, bool at_pick = false);   // mtx held
    void   pl_drop(llama_moe_stream_layer & nx, const pl_entry & en, int reason, int buf);  // mtx held
    // a reader's turn: bind the first entry that still needs a read; mtx held. CHK (test only): the order check
    template <bool CHK> bool pl_take(int ith);
    // The same, skipping entries ranked above R while the drive is behind; mtx held
    template <bool CHK> bool pl_take_behind(int ith);
    bool   behind_now() const { return bh_on && bl_n >= bh_B; }   // the drive is behind (>= B reads not finished)
    // An entry a reader that is behind leaves on the list (ranked above R; with lookahead, a dropped entry)
    bool   pl_skip_behind(const pl_entry & en) const { return en.rank >= bh_R || (la_on && !la_old_order && !en.live); }

    // --moe-stream-lookahead N > 1. All mtx held.
    bool     la_on = false;
    bool     la_old_order = false;    // test only: one class, no mid-way move
    uint32_t pl_seq = 0;
    bool (llama_moe_stream::*la_less)(const pl_entry &, const pl_entry &) const = nullptr;
    bool   la_less_b  (const pl_entry & a, const pl_entry & b) const;   // order B (the default)
    bool   la_less_a  (const pl_entry & a, const pl_entry & b) const;   // order A (--moe-stream-plan-extra-near 1)
    bool   la_less_old(const pl_entry & a, const pl_entry & b) const;   // the old order (--moe-stream-plan-old-order 1)
    void   la_guesses(llama_moe_stream_layer & sl, const int32_t * pred, int32_t width, int32_t n_cols);   // a layer's guesses
    // the same for any source layer src (sl = nullptr for a word-start source, which is not streamed)
    void   la_guesses_src(int32_t src, llama_moe_stream_layer * sl, const int32_t * pred, int32_t width, int32_t n_cols);
    void   la_guess(llama_moe_stream_layer & nx, int32_t d, const int32_t * ids, int32_t width);
    void   la_sort();
    // TEST ONLY (MOE_STREAM_TEST_LIST_DUMP=W, read once at start-up): the sort that also prints the list for the decode
    // words W..W+2, and the order check (below) - counted over the whole run, printed at the end
    void (llama_moe_stream::*la_sort_fn)() = &llama_moe_stream::la_sort;
    void   la_sort_dump();
    int64_t la_dump_word = -1;
    // a reader: the first queued guess piece it may take now; CHK (test only): every hand-out checked
    template <bool CHK> bool la_pop_prefetch(llama_moe_stream_job & job);
    bool   la_live_bindable() const;   // a live list entry that a reader may bind now (the behind rule applied)
    // Decode ubatches of one token in a call of several tokens took the direct 1-ahead path; la_guesses calls
    // (and of them in a call of several tokens: none by construction)
    int64_t n_la_direct_ub = 0, n_la_guess_calls = 0, n_la_guess_multi = 0;

    // TEST ONLY (with MOE_STREAM_TEST_LIST_DUMP): the order check, independent of the list's own fields and
    // comparators. A shadow log of the word's guesses (target, distance, the names in rank order), appended before any entry is
    // touched; at every reader's bind every entry's key is recomputed from it (votes, latest distance / rank, live, extra) and
    // compared with the stored fields, the bound entry must be first in the recomputed order among the entries the reader may
    // bind, a dropped entry only when no live one may be bound; at every hand-out of a queued guess piece, its read's moved-back
    // state is recomputed (and a moved-back piece only when no live entry may be bound)
    struct la_log_rec { int32_t il, d, off, n; };
    struct la_chk_val { bool found = false, live = false; int32_t d = 0, rank = 0, votes = 0, n_d = 0; };
    bool   la_extra_near = false;      // order A (the check's own key needs it)
    int64_t la_chk_word = -1;
    std::vector<la_log_rec> la_log;
    std::vector<int32_t>    la_log_ids;
    std::vector<std::vector<int32_t>> la_log_by_il;   // [n_layer] the target's records in arrival order
    void   la_chk_log(int32_t X, int32_t d, const int32_t * ids, int32_t width);   // one guess
    la_chk_val la_chk_recompute(int32_t il, int32_t e) const;
    bool   la_chk_may_bind(const la_chk_val & v, bool behind) const;
    bool   la_chk_before(const la_chk_val & va, const pl_entry & a, const la_chk_val & vb, const pl_entry & b) const;
    bool   la_chk_fields(const pl_entry & en, const la_chk_val & v);   // counts; true = equal
    bool   la_chk_back(const llama_moe_stream_layer & sl, int32_t s) const;
    void   la_chk_bind(const pl_entry & en, bool behind);
    void   la_chk_out(const llama_moe_stream_job & job, bool back);
    int64_t n_lc_binds = 0, n_lc_binds_behind = 0, n_lc_binds_dropped = 0, n_lc_fields = 0, n_lc_back_out = 0, n_lc_fwd_out = 0;
    int64_t n_lc_unchecked = 0;
    int64_t v_lc_fields = 0, v_lc_log = 0, v_lc_order = 0, v_lc_skip = 0, v_lc_dropped = 0, v_lc_back_live = 0, v_lc_back_flag = 0;
    int64_t v_lc_fwd_back = 0;
    // TEST ONLY (MOE_STREAM_TEST_LA_BAD_ORDER=n, read once at start-up): order B with key n inverted (1 target, 2 live, 3 extra,
    // 4 votes, 5 rank) - the check's failure test
    bool   la_less_bad(const pl_entry & a, const pl_entry & b) const;
    int    la_bad_key = 0;
    // --moe-stream-guess-combined 1 (with N > 1): a layer's N guesses in one step (the graph builds it). Per streamed
    // layer the stacked inputs it needs (the N targets' ffn_norm gains [n_embd, 1, n] and selection biases [n_expert, n], in
    // la_targets order; built once after the weights are loaded, on the device of the layer's routers; null = the layer keeps
    // the separate ops); MOE_STREAM_TEST_GUESS_CHECK (test only): 1 = the bit check, 2 = its failure test
    bool   la_comb = false;
    int    la_gchk = 0;
    bool   pn_on   = false;   // --moe-stream-protect-named 1 (pick_victim's PN version; the counters at a layer's start)
    std::vector<ggml_context *>         la_stack_ctx;   // the stacks' contexts and buffers (freed with the engine)
    std::vector<ggml_backend_buffer_t>  la_stack_buf;
    int32_t la_comb_layers = 0, la_comb_sep = 0;        // layers with the combined step / with targets but kept separate
    void   la_set_stack(int32_t il, ggml_tensor * gain, ggml_tensor * bias) {
        if (!layers[il]) {   // a word-start source
            ws_gain_stack[il] = gain;
            ws_bias_stack[il] = bias;
            ws_comb_ok   [il] = 1;
            return;
        }
        layers[il]->la_gain_stack = gain;
        layers[il]->la_bias_stack = bias;
        layers[il]->la_comb_ok    = true;
    }

    // --moe-stream-wordstart D > 0: the word-start sources ws_first .. ws_S0 - 1 = the last D layers before the first
    // streamed layer ws_S0 (D clamped to the layers before it; set once at start-up). A graph of one token builds each source's
    // guesses for its la_targets (llm_graph_context::build_ws_guess) and one CPU op after the last source
    // (llama_moe_stream_wordstart) adds them to the waiting list, source by source in layer order, through the streamed layers'
    // add code (la_guesses_src; with N = 1 prefetch_next's plan path). ws_adding marks the entries and names it makes. mtx held
    int32_t ws_D = 0, ws_S0 = -1, ws_first = -1;
    // --moe-stream-correction: the corrected guess matrices W'(X, d), and (when ffn_norm is folded) the ones that replace X's
    // gain in that guess (built at model load by corr.load; nullptr = the plain guess). A corrected pair's guess is
    // W' . rms_norm(base) with no gain, or W' . guess_in(X) on a hyper-connection model
    llama_moe_correction corr;
    ggml_tensor * corr_gate(int32_t X, int32_t d) const { return corr.gate(X, d); }
    ggml_tensor * corr_unit(int32_t X, int32_t d) const { return corr.gate(X, d) != nullptr ? corr.unit(X) : nullptr; }
    bool    ws_on() const { return ws_D > 0; }
    bool    ws_src(int32_t il) const { return ws_D > 0 && il >= ws_first && il < ws_S0; }
    bool    ws_adding = false;
    std::vector<ggml_tensor *> ws_gain_stack, ws_bias_stack;   // [n_layer] a source's combined-step stacks (the combined guess's, built with them)
    std::vector<uint8_t>       ws_comb_ok;                     // [n_layer] the source builds the combined step
    ggml_tensor * ws_gain(int32_t il) const { return ws_gain_stack.empty() ? nullptr : ws_gain_stack[il]; }
    ggml_tensor * ws_bias(int32_t il) const { return ws_bias_stack.empty() ? nullptr : ws_bias_stack[il]; }
    bool          ws_comb(int32_t il) const { return !ws_comb_ok.empty() && ws_comb_ok[il] != 0; }
    void    ws_add(const int32_t * cols, int32_t width, int64_t n_cols);   // the op's work (takes mtx)
    // decode, engine-wide: words whose op made its adds, op runs that made none (not a decode call of one token), and the guesses
    // (columns) made per source (index S0 - V, 1..8) and distance (1..8)
    int64_t n_ws_words = 0, n_ws_skip = 0;
    int64_t n_ws_guess[9][9] = {};
    // --moe-stream-needed-early 1: a source whose word-start targets' routers are all in a
    // device buffer gets a slot in the helper's slot buffer (the same layout as a streamed layer's: its la_targets columns of ne_width
    // ids, then the call counter after ne_cols columns; ne_buf / ne_pin already hold one slot per model layer) - ws_slot[V] = 1, decided
    // once (ne_set_router_gpu); the other sources keep the word-start direct add. The host producer: the word-start op posts each slot-source's
    // columns (ws_post, in source order) and the helper makes its adds (ws_on_guess_ready: the word-start add code, the too-late rule). The
    // source's post word / time, and per source (index S0 - V) the posts, the adds done, the posts with a column too late, the columns
    // too late by distance, the post -> adds done time
    std::vector<uint8_t> ws_slot;                // [n_layer]
    std::vector<int64_t> ws_post_word, ws_post_us;   // [n_layer]
    int64_t n_ws_post[9] = {}, n_ws_done[9] = {}, n_ws_late[9] = {}, t_ws_delay_us[9] = {};
    int64_t n_ws_late_d[9][9] = {};
    bool    ws_bad_counter = false;              // MOE_STREAM_TEST_WS_BAD_COUNTER=1 (test only, read once): the first post's counter is wrong
    bool    ws_bad_counter_set = false;          // the switch as read at start-up (ws_bad_counter is consumed by the first post)
    void    ws_post(int32_t V, const int32_t * cols, int32_t width, int32_t n_t);   // mtx held
    void    ws_on_guess_ready(int32_t V);                                           // mtx held (the helper)
    // When EVERY source with a target has all its
    // targets' routers on the producer's GPU (ws_gpu, decided once in ne_gpu_setup; otherwise every source keeps the word-start op), a GPU-producer
    // one-token graph marks each source's guess (ws_mark_node: the guess mark, the source's slot registered in the mark table),
    // so the GPU writes the columns + the call's number into ne_slot(V) and records V's "guess done" event inside V's launch; the word-start CPU
    // op is not built. The scheduler's split hook (ws_split_issued) queues each source for the helper after the split holding its
    // mark was issued, in source order S0 - D .. S0 - 1 ("launch enqueued before waited"); the helper waits on the event (the
    // counter-first rule), checks the counter, then ws_on_guess_ready. ws_gpu_next / ws_gpu_posted: the main thread's, per call
    bool    ws_gpu = false;
    // MOE_STREAM_TEST_WS_INELIGIBLE is read in ne_gpu_setup; set without a GPU-producer setup it would change nothing, so
    // ws_test_check (after the load) stops the run
    bool    ws_inel_parsed = false, ws_inel_used = false;
    void    ws_test_check() const;
    int32_t ws_gpu_n = 0, ws_gpu_next = 0, ws_gpu_posted = 0;
    int64_t n_ws_gpu_ready = 0, n_ws_gpu_event = 0, n_ws_gpu_none = 0;   // the helper's waits; GPU-producer calls with no source issued
    void    ws_mark_node(int32_t V, ggml_tensor * cols);   // the graph builder: mark source V's guess (it must fit the slot)
    void    ws_split_issued(ggml_backend_t backend, ggml_cgraph * g);   // the split hook (main thread; takes mtx when it queues)
    // the bit check's custom op (MOE_STREAM_TEST_GUESS_CHECK): its user data per (layer, distance), made once at start-up
    struct gchk_ud_t { llama_moe_stream * mgr; int32_t il, d; };
    std::vector<gchk_ud_t> la_gchk_ud;
    gchk_ud_t * gchk_ud(int32_t il, int32_t d) { return &la_gchk_ud[(size_t) il*(LOOKAHEAD_MAX + 1) + (size_t) d]; }
    std::atomic<int64_t> la_gchk_rows{0};
    std::atomic<int64_t> la_gchk_rows_ws{0};   // of them the word-start sources' rows
    std::atomic<bool>    la_gchk_flipped{false};

    // The behind rule and guess-inflight. All under mtx. bh_on / gi_on and the reader
    // loop's version are chosen once at start-up: with all three settings 0 the readers run the plain loop (io_loop_t<false>).
    bool                  bh_on = false;          // --moe-stream-behind-backlog B and -rank R both > 0
    int32_t               bh_B = 0, bh_R = 0;
    int32_t               gi_G = 0;               // --moe-stream-guess-inflight (0 = no limit)
    int32_t               bl_n = 0;               // reads not finished: +1 when enqueue starts one, -1 when it settles (or its
                                                  // slot is dropped / re-issued first) - once per read, through slot_bl
    int64_t               bh_word_last = -1;      // the last decode word counted in n_bh_words
    int64_t               n_bh_words = 0, n_turns = 0, n_bh_turns = 0, n_gi_wait = 0, t_gi_wait_us = 0;
    void   bl_add(llama_moe_stream_layer & sl, int32_t slot);   // enqueue: the slot's new read joins the backlog
    void   bl_end(llama_moe_stream_layer & sl, int32_t slot);   // the slot's read settled or was given up: it leaves
    void   bh_note_word() {                                     // the drive is behind now: count the decode word once
        if (decode_phase && bh_word_last != n_tok_outer) {
            bh_word_last = n_tok_outer;
            n_bh_words  += 1;
        }
    }
    // coordination 3.4: true when the list has no entry (always with --moe-stream-plan 0); mtx held
    bool   waiting_list_empty() const { return pl_list.empty(); }

    // A read is split into chunks of this many bytes so that the I/O threads serve it in
    // parallel; 0 = one read per expert, which is what the engine did before
    uint32_t chunk_demand   = 0;
    uint32_t chunk_prefetch = 0;
    // The smallest batch whose wave graphs run their expert matmuls on the GPU
    uint32_t gpu_waves      = 0;
    // Compute the experts whose bytes have landed while the missing ones are still being read.
    // Default off: it moves the wait out of remap() and into ggml's mul_mat_id, and it must be measured
    // (and stay bit-identical) before it can be on by default.
    bool        overlap     = false;
    // 2 MiB-aligned + MADV_HUGEPAGE pools, page-locked pools (cudaHostRegister), and the
    // eviction policy of a layer's pool ("lru", "lfu" or "slru")
    bool        hugepages   = false;
    bool        pin         = false;
    int         policy      = MOE_STREAM_POLICY_LRU;
    // The host buffer the graph copies a guessed top-8 into (one per graph, reused per layer)
    ggml_tensor * recall_buf() const { return recall_t; }
    // The recall buffer holds this many tokens of guessed ids (a graph with more tokens skips the measurement)
    static constexpr int64_t recall_tokens = 8192;

    // Where the graph writes this layer's guess scores, and which layer's remap op appends them to
    // the output. ONE fixed slot, not one per word: llama.cpp replays a captured CUDA graph for decode,
    // so a per-word host offset would be baked into the capture and every replay would write to word 0
    // (that is what the first version did). The last streamed layer's op reads the slot after every
    // layer of the graph has written it, and appends it to pred_scores_out.
    // Where the graph writes this layer's router input for the current decode word, and the guess
    // it makes from it. ONE fixed slot per array, for the same reason as the scores above: the offset a
    // captured CUDA graph writes at is baked into the capture. The slot holds ONE token, so the graph
    // only builds the copies in decode graphs (n_tokens == 1) - a prompt ubatch would need one slot per
    // token, which is 43 x 2560 floats each.
    ggml_tensor * pred_input_vec_tensor()   const { return pred_input_vec_t; }
    ggml_tensor * pred_input_vec2_tensor()  const { return pred_input_vec2_t; }
    void          mark_pred_input_vec2()          { pred_input_vec2_used = true; }
    size_t        pred_input_vec_offset(int32_t il) const { return (size_t) il*n_embd; }
    ggml_tensor * pred_input_guess_tensor() const { return pred_input_guess_t; }
    size_t        pred_input_guess_offset(int32_t il) const { return (size_t) il*n_expert_used; }

    ggml_tensor * pred_scores_buf_tensor() const { return pred_scores_t; }
    size_t        pred_scores_offset(int32_t il) const { return (size_t) il*pred_scores_n_expert; }
    uint32_t      pred_scores_n_expert = 0;

    // The prediction for layer pred_il is built from the residual plus this layer's shared expert
    bool        pred_shared = false;
    // A counter of how much of the guessed top-8 the real top-8 of the next layer keeps, and a dump
    // of the guess's scores for every expert of every generated word
    bool                    pred_recall   = false;
    ggml_tensor *           recall_t      = nullptr;      // where the graph copies a guessed top-8
    struct ggml_context *   recall_ctx    = nullptr;
    void *                  recall_buf_mem    = nullptr;
    ggml_backend_buffer_t   recall_buf_handle = nullptr;
    // The vector the next layer's router is run on, for 1 decode word in N, so that a learned
    // corrector can be trained on what the *engine's* guess sees instead of the plain router input
    int64_t                 n_embd           = 0;   // the router input's width
    std::string             pred_input_path;
    int32_t                 pred_input_every = 8;   // 1 word in N
    int32_t                 n_pred_input     = 0;   // words stored so far
    int32_t                 pred_input_max   = 0;   // words the output vectors may hold
    float *                 pred_input_vec_mem    = nullptr;
    ggml_tensor *           pred_input_vec_t      = nullptr;
    struct ggml_context *   pred_input_vec_ctx    = nullptr;
    ggml_backend_buffer_t   pred_input_vec_buf    = nullptr;
    float *                 pred_input_vec2_mem   = nullptr;
    ggml_tensor *           pred_input_vec2_t     = nullptr;
    struct ggml_context *   pred_input_vec2_ctx   = nullptr;
    ggml_backend_buffer_t   pred_input_vec2_buf   = nullptr;
    bool                    pred_input_vec2_used  = false;
    int32_t *               pred_input_guess_mem  = nullptr;
    ggml_tensor *           pred_input_guess_t    = nullptr;
    struct ggml_context *   pred_input_guess_ctx  = nullptr;
    ggml_backend_buffer_t   pred_input_guess_buf  = nullptr;
    std::vector<float>      pred_input_out_vec;    // [n_words, n_layer, n_embd] as it is appended
    std::vector<float>      pred_input_out_vec2;   // [n_words, n_layer, n_embd] lead-2 guess input when distinct
    std::vector<int32_t>    pred_input_out_guess;  // [n_words, n_layer, n_expert_used], the engine's guess
    std::vector<int32_t>    pred_input_out_real;   // [n_words, n_layer, n_expert_used], the real top-k
    std::vector<int64_t>    pred_input_rows;       // the decode word index of each stored word
    int32_t                 pred_input_last = -1;  // the last streamed layer: its op appends the word

    std::string             pred_scores_path;
    float *                 pred_scores_mem  = nullptr;
    ggml_tensor *           pred_scores_t    = nullptr;
    struct ggml_context *   pred_scores_ctx  = nullptr;
    ggml_backend_buffer_t   pred_scores_buf  = nullptr;
    int32_t                 n_pred_scores    = 0;    // words written so far
    int32_t                 pred_scores_last = -1; // the last streamed layer: its remap op appends
    std::vector<float>      pred_scores_out;     // [n_words, n_layer, n_expert] as it is appended
    int32_t                 pred_scores_max  = 0;    // words the buffer holds
    std::vector<int64_t>    pred_scores_row;         // the trace row of each stored word

    // ---- --moe-stream-record DIR: the side-by-side recording
    //
    // Per decode word and streamed layer: the real top-k with its weights and whether each expert was in
    // RAM, the raw router logits of the leads 1..4 (computed by the graph, copied into ONE fixed host slot
    // per source layer - a captured CUDA graph bakes the destination address in, see the recall-counting / guess-input notes
    // above), the residency of the next four layers' pools, and timestamps. Nothing here is passed to the
    // prefetcher or changes what is read, moved or evicted.
    static constexpr int32_t REC_LEADS = 4;
    bool          rec_on() const { return rec_logit_t != nullptr; }
    ggml_tensor * rec_logits_tensor() const { return rec_logit_t; }
    // lead d (1..REC_LEADS) of SOURCE layer il: [n_expert] floats of the router of il+d on il's guess input
    size_t        rec_logits_offset(int32_t il, int32_t d) const { return ((size_t) il*REC_LEADS + (size_t) (d - 1))*n_expert_total; }
    ggml_tensor * rec_w_tensor() const { return rec_w_t; }
    size_t        rec_w_offset(int32_t il) const { return (size_t) il*n_expert_used; }
    // TEST ONLY (MOE_STREAM_TEST_REC_RLOGITS=1): the REAL router logits of every streamed layer (what its
    // own router produced on its own input when it ran), by TARGET layer - the ridge's training target. Off by default.
    bool          rec_rlogits() const { return rec_rlogit_t != nullptr; }
    ggml_tensor * rec_rlogits_tensor() const { return rec_rlogit_t; }
    size_t        rec_rlogits_offset(int32_t il) const { return (size_t) il*n_expert_total; }
    // is there a streamed layer il and a streamed layer il+d (so that the graph computes the logits)? Or a word-start
    // source il whose guess for the streamed layer il+d the graph builds (d <= --moe-stream-lookahead; only with the prefetch
    // on - with --moe-stream-prefetch 0 the graph builds no word-start guess, bailingmoe3.cpp ws_graph)
    bool          rec_lead_avail(int32_t il, int32_t d) const {
        const int32_t X = il + d;
        return il >= 0 && X < (int32_t) sidecar_has.size() &&
               (sidecar_has[il] || (ws_src(il) && do_prefetch && d <= (int32_t) lookahead)) && sidecar_has[X];
    }
    void rec_stage_state(llama_moe_stream_layer & sl, const int32_t * ids, int64_t k);   // mtx held
    void rec_stage_done(llama_moe_stream_layer & sl, const int32_t * ids, int64_t k, int64_t t0, int64_t t1);
    void rec_append(llama_moe_stream_layer & sl);
    void rec_word_end(int64_t t_us);   // llama_context::synchronize(): the last recorded word's graph is done
    void rec_write(const std::string & router_json);
    void rec_batch_start() { if (rec_logit_t) { rec_t_call = ggml_time_us(); } }

    std::string             rec_dir;
    int32_t                 rec_max_words = 0;
    int32_t                 rec_n_dropped = 0;      // words whose rows were incomplete or over the cap
    int32_t                 rec_written   = -1;     // words in the files on disk
    int32_t                 rec_first = -1, rec_last = -1;   // the first / last streamed layer
    int64_t                 rec_t_call = 0;         // when llama_decode accepted the batch of the word in progress
    float *                 rec_logit_mem = nullptr;    // [n_layer, REC_LEADS, n_expert], by SOURCE layer
    ggml_tensor *           rec_logit_t   = nullptr;
    struct ggml_context *   rec_logit_ctx = nullptr;
    ggml_backend_buffer_t   rec_logit_buf = nullptr;
    float *                 rec_w_mem     = nullptr;    // [n_layer, k] the routing weights of the real top-k
    ggml_tensor *           rec_w_t       = nullptr;
    struct ggml_context *   rec_w_ctx     = nullptr;
    ggml_backend_buffer_t   rec_w_buf     = nullptr;
    float *                 rec_rlogit_mem = nullptr;   // [n_layer, n_expert], the real router logits (TEST ONLY flag)
    ggml_tensor *           rec_rlogit_t   = nullptr;
    struct ggml_context *   rec_rlogit_ctx = nullptr;
    ggml_backend_buffer_t   rec_rlogit_buf = nullptr;
    // staged by the remap op of each layer, appended by the last streamed layer's op (graph thread only)
    std::vector<int32_t>    rec_s_ids;      // [n_layer, k]
    std::vector<uint8_t>    rec_s_res;      // [n_layer, k] 0 not in the pool, 1 landed, 2 in flight
    std::vector<uint8_t>    rec_s_bits;     // [n_layer, REC_LEADS, n_expert/8] in the pool of layer il+d
    std::vector<uint8_t>    rec_s_bitsr;    // the same, landed only
    std::vector<int64_t>    rec_s_t0, rec_s_t1;   // [n_layer] remap op start / end
    std::vector<int64_t>    rec_s_word;     // [n_layer] the decode word the stage belongs to
    // the output, in the file layout
    std::vector<int64_t>    rec_o_words;
    std::vector<int32_t>    rec_o_truth;
    std::vector<float>      rec_o_truth_w;
    std::vector<uint8_t>    rec_o_truth_res;
    std::vector<float>      rec_o_logits;
    std::vector<float>      rec_o_rlogits;   // the real router logits, [W, Ls, n_expert] (TEST ONLY flag)
    std::vector<uint8_t>    rec_o_bits, rec_o_bitsr;
    std::vector<int64_t>    rec_o_t_layer, rec_o_t_layer_end, rec_o_t_word, rec_o_t_snap;
    std::vector<int64_t>    rec_s_tsnap;    // [n_layer] when the pool state of the word was read (under the lock)

    // ---- half 2: the drive events
    bool                    rec_ev_on = false;
    std::atomic<int64_t>    rec_word{-1};   // decode word in progress, -1 in the prompt phase
    std::vector<std::vector<moe_rec_event>> rec_ev;   // [0] graph thread, [1 + ith] I/O thread ith, [1 + n_io] the guess helper (needed-early)
    std::vector<size_t>     rec_ev_cap;
    std::vector<int64_t>    rec_ev_dropped;
    int64_t                 rec_ev_written = -1;      // events in the file on disk
    int32_t rec_clock() const { return (int32_t) (rec_word.load(std::memory_order_relaxed) + 1); }
    // mtx held by every caller
    void rec_emit(int buf, uint8_t type, uint8_t flags, int32_t layer, int32_t expert, int32_t slot,
                  int32_t a, int32_t b, int32_t c, int64_t t_us);
    void rec_use(llama_moe_stream_layer & sl, int32_t slot) {
        if (rec_ev_on) {
            sl.rec_uses[slot] += 1;
            sl.rec_last[slot]  = rec_clock();
        }
    }
    // The MoE layers whose routed experts are not streamed but loaded onto the GPU (no pool, no
    // remap op, no prefetch: the stock build_moe_ffn path over all n_expert experts)
    std::vector<int32_t>   gpu_layers;
    std::vector<int32_t>   gpu_layers_pick;   // the same layers in the order they were picked (a list's own order)
    // --moe-stream-gpu-layers-device: the named whole layers that did not fit on their GPU. They are streamed, and llama_model moves
    // each of them completely (attention, its context memory) to the pool's GPU, so no layer has its attention on one card and its
    // experts on another
    std::vector<int32_t>   wl_dropped;
    // More than one GPU. pool_dev: the GPU of the VRAM expert pool (--moe-stream-vram-device, else the first GPU of the model; null =
    // none), set at load before the per-device check; pool_dev_set: the flag named it (the prompt matmuls of the RAM pools then run on that
    // GPU only). whole_dev[il]: the device that holds whole layer il's experts (--moe-stream-gpu-layers-device, else the layer's own device).
    // router_dev_all[il]: the GPU that runs layer il's router (null: the CPU, or not known), set after the weights load
    ggml_backend_dev_t              pool_dev     = nullptr;
    bool                            pool_dev_set = false;
    std::vector<ggml_backend_dev_t> whole_dev;
    std::vector<uint8_t>            whole_test_other;   // MOE_STREAM_TEST_WHOLE_OTHER_CARD (test only): counts as on another GPU for the drop
    std::vector<ggml_backend_dev_t> router_dev_all;
    ggml_backend_dev_t router_dev_of(int32_t il) const { return il >= 0 && (size_t) il < router_dev_all.size() ? router_dev_all[(size_t) il] : nullptr; }
    // The whole layer to give up after an allocation failed on device `failed`: the last picked whose experts are there (the last picked
    // when `failed` is null); -1 = none there
    int32_t whole_layer_to_drop(ggml_backend_dev_t failed) const;
    // The VRAM budget the auto-fit divides by the cost of one whole layer, and the order it
    // takes them in ("hot": the layers a census measured as reading the most, "last": the deepest; the default is "hot")
    int64_t     vram_reserve    = 0;
    std::string gpu_layers_mode = "hot";    // the constructor turns an empty value into "hot"
    // The RAM the engine leaves free for everything else, and the part of the budget that is not
    // ours (the weights of the layers we do not stream, the KV cache, the compute buffers)
    int64_t     ram_free   = 0;
    int64_t     other_bytes = 0;

    std::vector<std::unique_ptr<llama_moe_stream_layer>> layers;  // [n_layer], null when not streamed

    std::vector<moe_stream_block> sidecar_blocks;  // [n_layer]
    std::vector<uint8_t>          sidecar_has;     // [n_layer]
    std::vector<int32_t>          moe_index_of;    // [n_layer] index among the streamed layers, -1
    std::vector<int32_t>          il_of_moe;       // [n_streamed] the layer of a MoE index

    // declared before the file, which opens it
    std::string sidecar;       // path of the sidecar data file
    std::string sidecar_json;  // path of the sidecar metadata

    llama_file file;  // the sidecar, opened with O_DIRECT

    // --moe-stream model: the experts are read from the model file itself, which carries what the experts file would
    // (tokenwing-tools/experts/make_model_file.py)
    bool        model_file = false;
    std::string model_file_path;   // the model's GGUF (where its appended data is)
    std::vector<std::array<uint64_t, 9>> mf_sums;       // [n_layer] FNV-1a 64 of the first, middle, last expert's gate, up, down
    std::vector<uint8_t>                 mf_has_sums;   // [n_layer]
    std::map<std::string, std::pair<uint64_t, uint64_t>> mf_blobs;   // appended data: name -> offset, size
    // the named data appended to the model file as a string; false when the model file is not a TokenWing model file or has no such data
    bool model_file_blob(const char * name, std::string & out) const;
    // the checksum check of one layer's sample experts against the file's metadata (stops the run on a mismatch)
    void verify_model_file_layer(llama_moe_stream_layer & sl);
    // The read-back check's read of a model file is several pieces on several threads: for the read that carries the check, the pieces
    // still to land per (layer, slot). The thread that lands the last one runs the check. Guarded by verify_mtx.
    std::map<std::pair<int32_t, int32_t>, int32_t> mf_left;

    std::vector<ggml_context *>    ctxs;  // one per pool tensor
    std::vector<ggml_backend_buffer_t> bufs;
    std::vector<void *>                  pool_mem;  // page-aligned allocations owned by us
    std::vector<size_t>                  pool_size; // bytes of each of them (for --moe-stream-pin)
    std::vector<uint8_t>                 pool_fresh; // allocated fresh by the OS layer (freed by it)
    // The issue->land window of every read the compute threads waited for, in us, so the
    // run prints the distribution instead of the design assuming it
    std::vector<int64_t>                 land_hist_us;
    bool                                 land_hist = false;   // MOE_STREAM_LAND_HIST=1

    FILE * log = nullptr;

    // I/O thread pool, all of it guarded by mtx
    mutable std::mutex       mtx;
    std::condition_variable  cv_work;   // a job was queued or the threads are asked to stop
    std::condition_variable  cv_done;   // a job finished
    std::deque<llama_moe_stream_job> q_demand;
    std::deque<llama_moe_stream_job> q_prefetch;
    std::deque<llama_moe_stream_job> q_idle;   // idle-time loads, taken only when nothing else waits
    std::deque<llama_moe_stream_job> q_refill; // elastic RAM's refill reads: after every needed and guessed read, el_refill_io at a time
    std::vector<std::thread>    io_threads;
    std::vector<std::unique_ptr<llama_file>> io_files;
    int io_fd(int ith) const {
        return ith >= 0 && (size_t) ith < io_files.size() && io_files[(size_t) ith]
            ? io_files[(size_t) ith]->file_id()
            : file.file_id();
    }
    bool                        io_stop = false;
    int32_t                     n_inflight = 0;      // reads in flight right now
    // Measured: a needed read takes 0.76 ms on an idle drive and ~4.7 ms with
    // eight others queued, so each outstanding *guess* makes the stall worse: prefetch_inflight caps
    // how many guessed reads may be outstanding, and a needed read never waits for the cap (it goes
    // to the front of the queue and is not counted against it).
    int32_t                     prefetch_inflight = 0;
    int32_t                     n_inflight_pref = 0;   // guessed reads in flight right now
    int32_t                     n_inflight_dem  = 0;   // jobs taken from q_demand in flight right now
    int64_t                     n_p1_promoted   = 0;   // guess chunks moved to q_demand
    int64_t                     n_p1_capped     = 0;   // times a thread waited because of guess_cap
    llama_moe_stream_p1         p1;                    // options
    std::vector<float>          p1_decay_tbl;          // 2^(-d/evict_halflife) for a distance of d decode words
    float p1_decay(int64_t d) const {                  // 0 past the table (64 half-lives)
        return d <= 0 ? 1.0f : d < (int64_t) p1_decay_tbl.size() ? p1_decay_tbl[(size_t) d] : 0.0f;
    }
    std::vector<float>          pe_decay_tbl;          // 2^(-d/prompt_end_halflife) precomputed to avoid subnormal exp2 traps
    float pe_decay(int64_t d) const {
        return d <= 0 ? 1.0f : d < (int64_t) pe_decay_tbl.size() ? pe_decay_tbl[(size_t) d] : 0.0f;
    }
    // The chance table (--moe-stream-chance-model): the chance that a
    // guess candidate is picked by its layer, by lead, rank, sigmoid score bin and band of layers
    static constexpr int P1_N_CAND = 16;
    struct p1_chance_table {
        int32_t            n_lead = 0, n_rank = 0, n_bin = 0, n_band = 0, band_div = 1;
        std::vector<float> edges;   // n_bin + 1 score bin edges
        std::vector<float> P;       // [n_lead][n_rank][n_bin][n_band]
    } p1_ch;
    bool chance_on() const { return !p1_ch.P.empty(); }
    float p1_chance(int32_t lead, int32_t rank, float p, int32_t target_il);   // mtx held
    std::vector<int32_t>        p1_sidx;                    // streamed-layer index by model layer, -1 = not streamed
    const float *               p1_pred_p = nullptr;        // step 1.4: the scores of the guess remap3 is handling
    int64_t                     n_p1_ch_tok = 0, n_p1_ch_cand = 0, n_p1_ch_read = 0, n_p1_ch_res = 0, n_p1_ch_below = 0;
    int64_t                     n_p1_ch_rank_read[16] = {0};
    int64_t                     n_p1_credit = 0;            // router-interest credits given
    bool                        p1_prompt_end_ub = false;   // this ubatch ends a prompt (llama_context)
    int64_t                     n_p1_prompt_end_ub = 0;     // ubatches llama_context flagged as a prompt end
    int64_t                     n_p1_prompt_end_sorted = 0; // layer calls whose waves were put in recency order
    int64_t                     n_dbg_wave_ok = 0;          // slot check: multi-wave layer ubatches whose waves passed it
    // --moe-stream-wave-plan 1: plans built (one per multi-wave layer call), by order, split experts; the index
    // rows checked against the wave's slot ids (MOE_STREAM_TEST_WAVE_ROWS=1, test only: per-pair index rows built
    // and compared in every wave without being used); MOE_STREAM_TEST_WAVE_SLACK=<rows> (test only):
    // a cap of the average + this slack pairs per wave (without it no cap: each expert in one wave)
    int64_t                     n_wp_build = 0, n_wp_a = 0, n_wp_b = 0, n_wp_split = 0, n_wp_rows_ok = 0, n_wp_rows = 0, n_wp_pad = 0;
    int64_t                     n_wp_pin = 0, n_wp_pe = 0;   // residents pinned to a prompt-end call's end; prompt-end plans
    int64_t                     n_rs_pe_sorted = 0, n_rs_restamped = 0, n_rs_restamp_layers = 0;
    // ONE plan and one slot-check count for the layer whose call runs now (the layers' wave ops run one at a time, in
    // graph order), not one per layer: at 16k tokens 20 B per token instead of 37 x that. wp_tab_il / dbg_wave_il = the owner; a
    // later wave of another layer finds it is not its own and stops the run (as a plan of another call does)
    moe_wave_plan               wp_tab;
    int32_t                     wp_tab_il   = -1;
    std::vector<uint8_t>        wp_res;          // [n_expert] scratch: in RAM (or being read) at wave 0
    std::vector<int32_t>        dbg_wave_cnt;    // slot check: per token of the running call, the pairs that got a real slot
    int32_t                     dbg_wave_il = -1;
    // Victims taken through pick_victim's "an expert of an earlier wave of this call" branch (counted always; printed
    // with the wave-plan line or MOE_STREAM_TEST_WAVE_VICTIM=1); --moe-stream-wave-victim 1 keeps those wave marks for the whole call
    int64_t                     n_victim_wave = 0;

    // Idle-time loading (--moe-stream-idle-load 1). A usage table (picks per expert
    // per streamed layer, counted live at each call's wave 0) is kept next to the sidecar on the internal NVMe (the first start reads a
    // seed counted from recordings) and saved by an atomic replace. After the context is built, and after every call once no call has
    // come for idle_pause_us, the filler reads the table's top experts into EMPTY RAM slots only - never an eviction - at the lowest
    // read priority (q_idle); a call's start takes every queued idle read back at once (the reads already in the drive finish).
    // All under mtx except the file writes. Off (the default): none of it runs and nothing is counted.
    bool                        idle_on        = false;
    std::string                 idle_path;          // the live table
    std::string                 idle_seed_path;     // the seed (read only when the live table does not exist)
    int64_t                     idle_pause_us  = 500000;
    std::vector<std::vector<uint32_t>> us_cnt;      // [n_layer_in][n_expert] (empty for a layer that is not streamed)
    bool                        us_dirty = false;
    std::string                 us_from  = "none";  // where the table came from at start-up
    std::string                 idle_arch;          // the model's architecture, written into the table
    std::thread                 idle_thread;
    std::condition_variable     idle_cv;            // arm / call start / call end / quit (mtx)
    bool                        idle_quit  = false;
    bool                        idle_armed = false;
    int32_t                     idle_calls = 0;     // calls running now (any context of the model)
    int64_t                     idle_last_end_us = 0;
    int64_t                     n_idle_rounds = 0, n_idle_reads = 0, n_idle_bytes = 0, n_idle_drained = 0, n_idle_used = 0;
    int64_t                     n_idle_unused = 0, n_idle_not_empty = 0, n_idle_stopped = 0, n_idle_saves = 0, n_idle_save_fail = 0;
    int64_t                     t_idle_last_ms = 0, n_idle_last = 0;
    int64_t                     n_idle_loading = 0;     // idle reads queued or in a worker (mtx)
    int64_t                     idle_call_seq  = 0;     // calls started so far (a round stops for good at the next one)
    int64_t                     n_idle_failed = 0, n_idle_start_waits = 0, t_idle_start_wait_us = 0;
    // An idle load's last use stays older than every slot already used in its layer (the layer's stamps and use_counter
    // are shifted up first: no wrap), rank 1 the newest preload, also after it lands. MOE_STREAM_TEST_IDLE_STAMP_OLD=1 (test only, read
    // once in idle_setup) gives the stamps before the fix (use_counter - 1 - r at the issue, wrapping; the landing's own stamp after it)
    bool                        idle_stamp_old = false;
    int64_t                     n_idle_shifts  = 0;   // layer shifts made for the fixed stamps
    // --moe-stream-idle-load 2: the fill at start. idle_arm runs one idle round on the calling thread before the context returns
    // (no call can come meanwhile); it waits for the reads to land instead of polling, keeps 2 reads per I/O thread queued, and the
    // first fill read of each layer carries the read-back check (a model that fits whole would have no needed read to carry it)
    bool                        fs_done    = false;   // the fill at start has run (once per model, not per context)
    bool                        fs_running = false;   // mtx: the fill at start is issuing reads (enqueue lets its reads carry the check)
    std::vector<int32_t>        fs_verify_slot;       // [n_layer] mtx: the slot whose fill read carries the layer's check, -1 = none
    std::vector<uint8_t>        verify_checked;       // [n_layer] verify_mtx: the layer's check has run (only sized for the fill at start)
    void fs_verify_settled(llama_moe_stream_layer & sl, int32_t slot, bool landed);   // mtx held: a fill read's claim kept or given back
    void idle_setup();                           // the constructor's part: the table, the thread
    void idle_arm(bool may_fill = true);         // llama_context, once the context (and the VRAM pool) is built; may_fill = false: no fill at start
    void idle_call_start();                      // llama_context::decode, before anything of the call runs
    void idle_call_end();                        // end_call
    void idle_loop();
    void idle_round(std::unique_lock<std::mutex> & lock, bool start = false);   // start: the fill at start (see fs_done)
    void idle_drain();                           // mtx held: every queued idle read taken back (its slot settles EMPTY)
    void idle_used(llama_moe_stream_layer & sl, int32_t s) {   // mtx held: a call used an expert an idle load (or a read-ahead) brought in
        if (!sl.slot_idle.empty() && sl.slot_idle[s]) {
            sl.slot_idle[s] = 0;
            n_idle_used += 1;
        }
        if (!sl.slot_ra.empty() && sl.slot_ra[s]) {
            sl.slot_ra[s] = 0;
            n_ra_used += 1;
        }
    }
    // --moe-stream-read-ahead 1: in a prompt call of at least read_ahead_min tokens,
    // layer L's wave-0 remap op (after its own needed reads are queued) queues guess-priority reads of the next streamed layer's most-used
    // experts (the usage table, up to a wave's width) into that layer's own EMPTY slots (evicting a resident saves no read when
    // almost every expert is in the union); never a pinned, switched-off, VRAM-follow-source or copy-back slot. Needs --moe-stream-wave-plan 1:
    // the next layer's plan then finds them in RAM (residents first). Shares the usage table with idle loading (us_on = either one).
    bool    ra_on = false;
    bool    us_on = false;         // the usage table is counted (idle loading or read-ahead)
    bool    us_loaded = false;
    int64_t n_ra_calls = 0, n_ra_reads = 0, n_ra_held = 0, n_ra_used = 0, n_ra_unused = 0;
    void ra_issue(llama_moe_stream_layer & sl);   // mtx held
    // TEST ONLY (MOE_STREAM_TEST_WINDOW_ENDSTATE=1): the RAM end state of a prompt against its pe128 top share and against what
    // the reply's first words use - counters only, nothing the engine does changes
    bool    es_on = false;
    FILE *  es_dump = nullptr;     // TEST ONLY (MOE_STREAM_TEST_WINDOW_ENDSTATE_DUMP): flushed at each prompt's end and at exit
    int32_t es_max_words = 64;
    int64_t es_seq_cur = 0;        // prompts so far (a prompt call after a reply word starts the next one)
    bool    es_in_reply = false;
    int64_t es_snap_seq = -1;      // the prompt the last snapshot belongs to
    int32_t es_layers_done = 0;
    void es_count(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n, int64_t n_tok);   // mtx held, a call's wave 0
    void es_snapshot();            // end_call of a prompt call (takes mtx)
    void es_print();               // mtx held

    // --moe-stream-window 1: in a layer-major prompt call of at
    // least read_ahead_min tokens whose layers need more than one wave, the pools are one window over the layers with the same total.
    // It reuses the slot shares pool layout (each pool reserved for all n_expert slots; slots switched on / off, pages returned or moved; the
    // zero block moved) - none of elastic RAM's or slot shares' decisions. All under mtx.
    //  - a layer's wave-0 remap op (win_layer_start): the layers before it are finished (win_finish: they keep the top `share x keep`
    //    of the call's experts by the keep rule's score, the rest are dropped: free room); the layer gets back at least its share;
    //  - after its own needed reads are queued (win_issue): the next 2 streamed layers' most-used experts (usage table rank, up to this
    //    layer's union size, not held in VRAM) are read at a guess's priority into room taken in this order: the target's own empty
    //    slots; empty slots of finished layers; empty slots of the layers farthest ahead; (for the next layer only) residents of the
    //    layers farthest ahead, lowest table rank first; (next layer only, last) the lowest-scored kept experts of finished layers,
    //    which are read back after the call;
    //  - the call's end (win_end): exact keep sets, every layer back to its share, the missing kept experts read back, zero blocks low.
    bool    win_on     = false;
    bool    win_active = false;            // a window call is running
    bool    win_check  = false;            // MOE_STREAM_TEST_WINDOW_CHECK=1 (test only)
    int32_t win_rule   = 0;                // MOE_STREAM_TEST_WINDOW_KEEP: 0 pe (pe128), 1 recency (the latest pick), 2 count
    int64_t win_budget = 0;                // the switched-on slots of all streamed layers at the call's start (never exceeded)
    std::vector<int32_t> win_pos;          // [layers.size()] the position in il_of_moe, -1 = not streamed
    int64_t n_win_calls = 0, n_win_reads = 0, n_win_moves = 0, n_win_pagemoves = 0, n_win_released = 0, n_win_borrowed = 0;
    int64_t n_win_keep_taken = 0, n_win_refill = 0, n_win_dropped = 0, n_win_kept = 0, n_win_missing = 0, n_win_grow_short = 0;
    int64_t n_win_end_short = 0, n_win_checks = 0, n_win_start_off = 0, n_win_stale = 0, n_win_drained = 0, n_win_passes = 0;
    int64_t win_call_seq = 0, win_call_ntok = 0;   // set_decode: calls so far, this call's tokens
    int64_t win_pass_call = -1, win_skip_call = -1;
    bool    win_page_move = false;         // the OS layer may move pages (Linux, enough map entries allowed: os_page_moves_ok)
    bool    win_gpu_said  = false;
    int64_t t_win_us = 0;                  // the window's own time under the lock
    int64_t t_win_end_wait_us = 0;
    int64_t n_win_refill_skip = 0;
    bool    win_gpu = false;               // GPU prompt waves possible (a GPU, --moe-stream-gpu-waves > 0): they need tail waves
    bool    win_test_compact = false;
    bool    win_log = false;               // MOE_STREAM_TEST_WINDOW_LOG=1 (test only)      // MOE_STREAM_TEST_WINDOW_COMPACT=1 (test only): compact at a pass's end without a GPU
    int64_t n_win_relocated = 0, n_win_compact_left = 0;
    void    win_relocate(llama_moe_stream_layer & sl, int32_t h, int32_t l);
    void    win_compact(llama_moe_stream_layer & sl);
    bool    win_keep_take = false;         // MOE_STREAM_TEST_WINDOW_KEEP_TAKE=1 (test only): the last layers' read-ahead may take kept experts
    bool    win_refill_on = false;         // MOE_STREAM_TEST_WINDOW_REFILL=1 (test only): read back evicted kept experts after a call
    bool    win_need_compact = false;      // the end's compaction is needed (a one-wave GPU offload, or GPU waves without tail waves)
    int64_t n_win_compact_short = 0;
    int64_t n_win_short_why[7] = {}, n_win_short_ready = 0, n_win_short_rwhy[8] = {}, n_win_incomplete = 0;
    int64_t n_win_stale_bound = 0;
    bool    win_gpu_blocked(int64_t n_tok) const;         // win_end: waiting for read-aheads already in a reader thread
    // A window call's expert matmuls stay on the host - also a one-wave piece the scheduler would copy to the GPU
    // as the start-size view (256+ slots per layer); set per prompt call at its start. A layer's waves cover what the graph's wave
    // count expects: room above the budget when nothing can be taken (win_over), given back at the next prompt call's start
    bool    win_host_call = false;
    bool    win_test_no_takeback = false, win_test_end_off = false;
    int32_t win_test_short = 0;            // MOE_STREAM_TEST_WINDOW_SHORT=N (test only): a starting layer gives its room away down to N slots   // MOE_STREAM_TEST_WINDOW_END_OFF=1 (test only): see win_close  // MOE_STREAM_TEST_WINDOW_NO_TAKEBACK=1 (test only): no room taken back at a layer's start
    int64_t win_over = 0, n_win_over = 0, n_win_cover = 0, n_win_call_fix = 0;
    bool    win_call(int64_t n_tok) const;                                                      // is a prompt call of n_tok a window call
    void    win_call_start(bool decode, int32_t n_tok);                                         // set_decode
    void    win_cover(llama_moe_stream_layer & sl, int64_t d);                                  // mtx held: d more slots for its waves
    void    win_restore(bool drop_ready);
    void    win_page_move_failed();                                                       // mtx held: every layer back to its share
    void    win_setup();                                                                        // the constructor
    void    win_layer_start(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n, int64_t n_tok);   // wave 0, before the plan
    void    win_issue(llama_moe_stream_layer & sl);                                             // wave 0, after the needed reads
    void    win_finish(llama_moe_stream_layer & sl, bool end);
    void    win_end(bool complete);                                                             // end_call (takes mtx)
    void    win_close(bool complete);                                                           // mtx held: the pass is over
    void    win_drain(llama_moe_stream_layer * only, const std::vector<uint8_t> * keep_e);      // mtx held: queued window reads back
    int64_t win_take(llama_moe_stream_layer & sl, int64_t need, bool borrow, bool keep_take);
    bool    win_move_room(llama_moe_stream_layer & a_sl, int32_t a, llama_moe_stream_layer & t_sl,
                          std::vector<std::pair<llama_moe_stream_layer *, int32_t>> * defer_rel = nullptr);
    bool    win_commit_slot(llama_moe_stream_layer & sl, int32_t s);
    bool    win_commit_range(llama_moe_stream_layer & sl, uint32_t a, uint32_t b);
    bool    win_was_decommitted(const llama_moe_stream_layer & sl, int32_t s) const;
    void    win_check_budget(const char * where) const;
    void    win_print_stats() const;
    bool us_save(std::unique_lock<std::mutex> & lock);   // the table to its file (the write outside the lock)
    void us_load();
    bool                        wp_rows_check = false;
    int64_t                     wp_slack      = -1;
    // MOE_STREAM_TEST_DROP_SLOT (test only): 1 = the slot check sees one pair of the first one-wave decode call of the
    // first streamed layer as slot-less, 2 = one pair of the first multi-wave ubatch. Changes nothing but what the check sees.
    int32_t                     test_drop_slot = 0;
    bool                        test_drop_done = false;     // the one pair has been chosen
    int32_t                     test_drop_il   = -1;        // =2: the layer whose multi-wave ubatch carries it, while it runs
    int64_t                     t_read_win_us = 0;   // wall time with at least one read in flight
    int64_t                     t_read_win_t0 = 0;   // when the first read of a window started

    bool do_prefetch = true;
    bool decode_phase = false;  // set by llama_context before every batch
    // The start-up read-back check. One filled pool slot of every streamed layer (or of the
    // first verify_layers of them) is compared against the bytes of the model file, and one expert of
    // one whole GPU layer against the same, so that wrong weights are named at the first token instead
    // of showing up as strange output days later. One slot costs one expert read plus one memcmp;
    // --moe-stream-verify 0 turns the check off, N>1 checks only the first N layers.
    bool        verify = true;
    int32_t     verify_layers = -1;   // -1 = every streamed layer
    // Keep the pools in llama.cpp's repacked CPU layout (mxfp4_8x8_q8_0 on x86), which is the same
    // number of bytes in a different order and lets the expert MUL_MAT_ID use the repacked kernels.
    // The sidecar stays plain: each read lands in a per-thread scratch buffer and is repacked into the
    // slot, which costs one pass over the slice on an I/O thread (measured 0.11-0.26 ms per expert) and
    // buys 2.2x on the matmul (measured 0.187 -> 0.084 ms for 8 experts and one token).
    bool        cpu_repack = true;    // false = the plain layout
    // What the sidecar stores. "plain" is every sidecar built before this field existed (the field is optional);
    // "mxfp4_8x8" means the experts are already in the repacked CPU layout, so a read copies them into
    // the slot with no conversion at all. A repacked sidecar with the flag off is refused: the pool
    // would be plain and the repacked weights would be read into it as if they were plain.
    std::string sidecar_layout = "plain";
    // Per streamed layer, since only MXFP4 can be repacked (Ling's MTP block is Q4_0): a plain
    // layer in a repacked file is copied into a plain pool, a repacked one into a repacked pool. Read
    // only by the loader before the I/O threads start.
    std::vector<uint8_t> block_repacked;
    // How many of the streamed layers the sidecar stores repacked (all of them, for Ling's
    // 40 MXFP4 layers). Read once by the pool setup, after the blocks are parsed.
    int32_t         n_repacked_blocks = 0;
    // what the pools actually got (the type may have no kernels). Atomic because the I/O threads read it
    // on every job and it is written by the loader, later, on another thread: a plain bool let -O3 keep
    // it in a register, so the reads kept taking the plain path into a repacked pool (garbage output).
    std::atomic<bool> cpu_repack_on{false};
    std::atomic<bool> plugin_repack_on{false};   // at least one pool is repacked by the quant plugin (for the messages)
    bool              gpu_waves_warned = false;   // the repacked-pool warning is printed once
    // THE rule "the expert waves of a graph judged by n_tok rows run on the GPU" - n_tok =
    // the ubatch's rows, or a layer-major expert graph's chunk (n_ubatch). The graph's own wave flag and cparams.op_offload are the
    // caller's (llama_context ANDs them; in the engine tail_allowed holds op offload). Both copies in llama-context.cpp call it.
    // Milestone 3, 2(a): never a one-wave ubatch the tail rule keeps on the host (onewave_host; false in every run before 2(a), whose
    // tail was refused there) - else a --moe-stream-gpu-waves at or under its rows would offload it with the whole pool
    bool waves_on_gpu(int64_t n_tok) const {
        return gpu_waves > 0 && n_tok >= (int64_t) gpu_waves && !cpu_repack_on.load() && !onewave_host(n_tok) && !el_win_no_whole_copy(n_tok);
    }
    // THE rule "this remap op's expert matmul runs on the CPU". It reads llama_context's own
    // decision for the graph being computed (graph_op_offload = op offload && (no wave flag || waves_on_gpu), set before every compute,
    // reused graphs too - review: re-deriving it per layer missed a last layer cut to the output rows) and the scheduler's threshold: with
    // op offload on, a matmul of at least the offload minimum rows (ne[2]: the op's rows, a layer-major chunk's at most) goes to the GPU -
    // GPU waves, or the whole-pool copy of a one-wave ubatch that waits for no read (a GPU machine with the tail
    // off, --moe-stream-gpu-waves 0, 2+ GPUs, elastic / shares). No GPU: always the CPU. (The threshold is the CUDA / Vulkan / Metal / SYCL
    // one, ne[2] >= GGML_OP_OFFLOAD_MIN_BATCH; the one-word rule assumes the same)
    bool    uma_active          = false; // --moe-stream-uma 1 active on a unified-memory device
    bool    uma_gpu_pool        = false; // shared/per-layer RAM pool buffer is zero-copy mapped onto uma_dev
    bool    uma_fallback_warned = false; // printed once if zero-copy GPU pool wrapping falls back to host CPU pool
    size_t  uma_pool_align      = 0;     // pool base & byte-size alignment used (65536 B when uma_active)
    ggml_backend_dev_t uma_dev = nullptr;
    ggml_backend_t     uma_backend = nullptr; // llama_context: active UMA GPU backend for wave > 0 slot barrier
    int32_t mark_spin_test_us = 0;     // TEST ONLY (MOE_STREAM_TEST_MARK_SPIN_US): host counter spin-poll window
    std::atomic<int64_t> n_mark_spin_hits{0};
    bool    graph_op_offload = true;   // llama_context, before every compute (true until it says: the safe side)
    // --moe-stream-cpu-down-scale: whether a layer's down-input scale step (#30861) can be left out of the graph being built.
    // The down matmul must certainly run on the CPU: no GPU device; or not a GPU wave and the scheduler cannot move it to a GPU (op
    // offload off for the context, or fewer rows than the offload minimum: ne[2] >= GGML_OP_OFFLOAD_MIN_BATCH goes to the GPU). A
    // graph with CPU waves would also stay on the CPU, but its offload flag is set by llama_context after the build, so the rule keeps
    // the step there (the safe side). And the down weights' CPU kernel must quantize the input with a float block scale (Q8_K): then
    // x / 2^n gives the same integers and a block scale 2^n smaller, all exact, and the output times 2^n is the same float - the scale
    // is exactly 2^n while expf(n ln2) is (n <= 12 with glibc: every row's max |x| <= 2^22). A q8_0 down (fp16 block scale) keeps it.
    bool down_scale_skip(ggml_type down_type, int64_t n_rows, bool gpu_wave, bool op_offload) const {
        if (uma_gpu_pool || !p1.cpu_down_scale || llama_cpu_moe_api()->get_type_traits_cpu(down_type)->vec_dot_type != GGML_TYPE_Q8_K) {
            return false;
        }
        if (!el_has_gpu) {
            return true;    // no GPU device: every matmul runs on the CPU (gpu_wave is only the row rule: --moe-stream-gpu-waves is 512 by default)
        }
        return !gpu_wave && (!op_offload || n_rows < offload_min1());
    }
    // Test only (MOE_STREAM_TEST_DOWN_SCALE_COUNT=1): rows of the skipped layers' down input, the rows whose scale would not
    // have been 1 (max |x| > 1024) and those above the exact limit (max |x| > 2^22); the graph builds that skipped / kept the step
    bool                 ds_count = false;   // the counting op: llama_moe_stream_down_scale_count (llama-moe-stream.cpp)
    std::atomic<int64_t> ds_rows{0}, ds_rows_scaled{0}, ds_rows_inexact{0};
    int64_t              ds_built_skip = 0, ds_built_keep = 0;
    int64_t graph_chunk      = 0;      // llama_context: a layer-major expert graph's chunk rows, 0 = none
    bool experts_on_cpu(int64_t n_tok) const {
        if (cpu_repack_on.load()) {
            return true;
        }
        if (uma_gpu_pool) {
            return false;
        }
        if (!el_has_gpu || !graph_op_offload) {
            return true;
        }
        if (tailw_graph(graph_chunk > 0 ? graph_chunk : n_tok)) {
            return false;   // tail waves read the gather copy in VRAM at any row count (--moe-stream-gpu-waves under the minimum)
        }
        const int64_t n_rows = graph_chunk > 0 ? std::min<int64_t>(n_tok, graph_chunk) : n_tok;
        return n_rows < offload_min1();
    }
    // Whether a read of a *plain* layer has to be converted, i.e. the pool is repacked. A layer
    // the sidecar already stores repacked is copied into the slot as it is (see block_repacked).
    bool              convert_on_read = false;
    int64_t     n_repack_us = 0;      // what the repack cost, summed
    int64_t     n_repack_n = 0;
    // The model is a split GGUF (…-00001-of-00002.gguf), so a projection's shard matters
    bool        model_is_split = false;
    std::vector<uint8_t> verify_done;  // [n_layer] this layer's slot has been compared
    std::vector<int32_t>  verify_layers_streamed;  // the layers the boundary check walks
    // How many layers the check may still cover, and the lock that hands the check to one job per
    // layer (the remap path claims it, the I/O thread runs it)
    std::atomic<int32_t>  verify_claims{0};
    bool                  verify_claims_init = false;   // the count is taken at the first read of the run
    bool                  verify_boundaries_done = false;
    std::mutex  verify_mtx;            // the check reads a model file and a scratch buffer of its own
    // These were std::vectors filled from a posix_memalign'd block, which copied the
    // bytes into the vector's own (16-byte aligned) storage and leaked the aligned block - O_DIRECT reads went
    // into an unaligned buffer. An owned aligned buffer now.
    struct aligned_buf {
        uint8_t * p = nullptr;
        size_t    n = 0;
        aligned_buf() = default;
        aligned_buf(const aligned_buf &) = delete;
        aligned_buf & operator=(const aligned_buf &) = delete;
#ifdef _WIN32
        ~aligned_buf() { _aligned_free(p); }
#else
        ~aligned_buf() { free(p); }
#endif
    };
    mutable aligned_buf verify_buf;     // page-aligned scratch: the sidecar's bytes
    mutable aligned_buf verify_buf2;    // and the model's
    aligned_buf         vf_stage;       // VRAM follow: a sidecar read of an incoming expert (the fixed mode; O_DIRECT)
    mutable std::vector<llama_file *> verify_files;  // one per model shard, opened on first use
    int64_t     n_verify = 0, bytes_verify = 0, t_verify_us = 0, n_verify_bad = 0;
    // A failed read is retried this many times (MOE_STREAM_READ_RETRIES, default 3) with this pause
    // between the tries (MOE_STREAM_READ_RETRY_US, default 20 ms): one bad sector on an SSD that runs for
    // days should not end the process. MOE_STREAM_READ_FAIL_ONCE=N makes the N-th read of the process
    // fail on purpose, which is how the retry path is tested.
    int     read_retries   = 3;
    int     read_retry_us  = 20;
    int64_t fail_at        = 0;      // 0 = no injected failure
    std::atomic<int64_t>   n_read_started{0};
    std::atomic<int64_t>   n_read_retry_atomic{0};
    // The tokens of each phase, counted once per batch by set_decode(), not once per layer
    int64_t n_tok_outer_pre = 0;
    int64_t n_tok_outer     = 0;
    // --moe-stream-timing 1: for [0] prompt calls and [1] decode calls, the calls and where their time went - wall time,
    // host backend compute, device kernels, cross-backend copies, backend syncs (microseconds). Added once per call, when it ends
    // (llama_context::decode); atomic because the server reads them from its HTTP thread while a request runs
    std::atomic<int64_t> tm_us[2][6] = {};
    void timing_add(int kind, int64_t wall, int64_t host, int64_t dev, int64_t copy, int64_t sync) {
        std::atomic<int64_t> * t = tm_us[kind];
        t[0].fetch_add(1,    std::memory_order_relaxed);
        t[1].fetch_add(wall, std::memory_order_relaxed);
        t[2].fetch_add(host, std::memory_order_relaxed);
        t[3].fetch_add(dev,  std::memory_order_relaxed);
        t[4].fetch_add(copy, std::memory_order_relaxed);
        t[5].fetch_add(sync, std::memory_order_relaxed);
    }
    int64_t b0_prompt_seq   = 0;   // prompt calls so far (the evictions of the last one are counted apart)
    // The position a speculative draft of a few tokens would continue from
    int32_t  decode_next_pos = 0;
    // The contexts that have run a batch on this model, so the engine can say once that they
    // share the pools (see set_decode)
    static constexpr int LLAMA_MOE_STREAM_MAX_CTX = 8;
    const void * ctx_seen[LLAMA_MOE_STREAM_MAX_CTX] = { nullptr };
    const void * ctx_last = nullptr;
    int          n_ctx    = 0;
    int          n_ctx_shared = 0;   // how many contexts have run a batch on this model
    bool         n_ctx_warned = false; // the caller warns once about it

    // one remap userdata per (layer, wave), the graph picks the ones it needs
    std::vector<llama_moe_stream_wave> wave_ctx;
    std::vector<int32_t>              wave_base;    // [n_layer] where a layer's wave ctx starts

    // oracle replay: the ids a recorded run picked, indexed [token, MoE layer, k]
    bool                   oracle_on = false;
    bool                   oracle_list = false;   // (test only, MOE_STREAM_TEST_ORACLE_LIST): the oracle feeds the waiting list
    uint32_t               oracle_lead = 1;
    uint32_t               oracle_n_tok = 0;
    uint32_t               oracle_n_moe = 0;
    uint32_t               oracle_k = 0;
    std::vector<int32_t>   oracle_ids;
    std::string            oracle_path;

    moe_stream_plan_in    mplan;

    // internals
    template <int BEHIND> void io_loop_t(int ith);    // BEHIND = 1 the behind rule / guess-inflight / lookahead version, 2 = the
                                                  // same + the test-only order check (MOE_STREAM_TEST_LIST_DUMP); chosen at start-up
    // Lead / rank / extra flags describe a guess for the event recorder (lead = layers ahead of the layer that
    // issued it, rank = position in its guess list); they change nothing in the read
    void   enqueue(llama_moe_stream_layer & sl, int32_t expert, int32_t slot, bool prefetch,
                   int32_t lead = 0, int32_t rank = -1, uint8_t rflags = 0, bool pr_split_ok = false, bool idle = false, bool refill = false);
    // A remap call's split needed reads queue their gate+up job at once and hold the down job here; the call
    // queues the held downs after its last needed read (or before it releases the lock to wait), so every gate+up of the call
    // is ahead of its downs. mtx held.
    std::vector<llama_moe_stream_job> pr_down_hold;
    void   pr_flush_downs(bool early);
    // target_il is the layer whose experts are prefetched (pred_il for the one-layer lead,
    // pred_il2 for the early guess), width how many experts the vote covers, and role 1 = early guess
    // (mark the slots so a correction can cancel them) or 2 = correction (cancel the early reads it
    // drops before issuing its own). src_il = the guessing layer when it is not sl (a word-start source, which is not
    // streamed: sl is then the target, whose scratch the call uses), -1 = sl.
    void   prefetch_next(llama_moe_stream_layer & sl, const int32_t * pred, int64_t n_pred, int64_t n_tok,
                         int32_t target_il, int32_t width, int32_t role, int32_t src_il = -1);
    // t_op0: when the remap op started (the lateness of a decode word's layer is measured from it)
    void   remap(llama_moe_stream_layer & sl, const int32_t * ids, const int32_t * pred,
            int64_t n, int64_t n_tok, int32_t wave, int32_t * out_slots, int64_t t_op0);
    // remap() in its three parts, mtx held by the caller (lock): F.1 step 1 (the layer starts; its picks, the needed reads
    // queued, its list entries; returns the needed reads queued), F.2 step 2 (the guess's adds), and the rest (the wait for READY
    // without overlap, the slot ids). remap() calls them in sequence; the in-graph form calls the first from op A and the other
    // two from op B (remap_a / remap_b; remap_b returns its own start - the layer's start for the recorder and the lateness counters)
    int64_t remap_needed(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n, int64_t n_tok, int32_t wave, int64_t t_op0,
            std::unique_lock<std::mutex> & lock);
    void   remap_guess (llama_moe_stream_layer & sl, const int32_t * pred, int64_t n_tok, int32_t wave);
    void   remap_finish(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n, int64_t n_tok, int32_t wave, int32_t * out_slots,
            std::unique_lock<std::mutex> & lock);
    // The pairs a wave may hold (no cap unless a test sets one), and wave w's three index tensors from the
    // stored plan (slot_w / tok_w / dst_w, cap rows; mtx held, after remap_finish of that wave; only the test check
    // MOE_STREAM_TEST_WAVE_ROWS reads them). Returns the rows used.
    int64_t wave_cap(int64_t n_tok) const;
    int64_t wave_rows(const llama_moe_stream_layer & sl, const int32_t * ids, int64_t n_tok, int32_t wave, int32_t * slot_w,
            int32_t * tok_w, int32_t * dst_w) const;
    void   remap_a(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n, int64_t n_tok, int32_t wave, int64_t t_op0);
    int64_t remap_b(llama_moe_stream_layer & sl, const int32_t * ids, const int32_t * pred, int64_t n, int64_t n_tok, int32_t wave,
            int32_t * out_slots, int64_t t_op0);
    // llama_context::decode() calls this at the end of every call, after the graph has run on the host (the CPU
    // splits run synchronously inside the graph compute) and before anything reads the counters: every layer's overlap wait
    // and the lateness of the word's not-READY experts are booked to the call that had them, not to the next one
    // complete = the call's graph ran to its end; a call stopped part-way removes its list entries (MOE_STREAM_PL_ABORTED)
    // instead of stopping the run
    void   end_call(bool complete);
    void   take_overlap_waits(llama_moe_stream_layer & sl);   // mtx held
    // the number of waves a ubatch of n_tokens tokens can need: one when everything fits
    int    n_waves(int64_t n_tok) const;
    int    n_waves_base(int64_t n_tok) const;   // the count at the start-size width (n_waves = this unless tailw_wave_w narrows a tail-wave graph)
    void   rs_restamp(llama_moe_stream_layer & sl);   // the prompt-end call's experts re-stamped by weight at the first reply word
    // May remap() leave the wait to ggml's mul_mat_id (--moe-stream-overlap)?
    // Only when that kernel really runs on this call's pool and reads the ready flags: not with a repacked
    // pool (its kernel never reads them), not in a wave graph (gpu-waves copy the pool to the device, and
    // the next wave needs the slots back), and not for a ubatch the scheduler may offload to the device.
    bool   defer_wait(int64_t n_tok, bool multi) const;
    bool   defer_wait_call(int64_t n_tok, bool multi) const;   // + --moe-stream-prompt-defer (prompt calls on the CPU)
    int64_t n_prompt_defer = 0, n_prompt_defer_gpu = 0;         // prompt remap ops that left their wait to mul_mat_id / that
                                                                //   did not because the GPU may run their matmul
    // a slot this call may overwrite. An empty or a ready slot wins, then - with cancel - a prefetch
    // that is in flight, which the caller then has to wait for. protect keeps the experts that a
    // prefetch brought in and that were not used yet, so that only a demand may take them. -1 when
    // there is nothing this call may take.
    // Prefetch = true when the caller is the prefetcher, which is what makes probation apply
    // ONE function built in two versions - FOLLOW = true also skips a slot whose VRAM follow mark is set. The
    // version is chosen by vf_hook, a start-up constant; with VRAM follow off no mark is read. Every victim choice goes through here.
    // And PN = --moe-stream-protect-named 1 (also a start-up constant): four versions, the off ones read nothing new
    template <bool FOLLOW, bool PN>
    int32_t pick_victim_t(const llama_moe_stream_layer & sl, bool cancel, bool protect, bool prefetch) const;
    // the scan itself; with PN, *pn_skipped says the rule passed over a named expert (the counters are booked by the caller)
    template <bool FOLLOW, bool PN>
    int32_t pick_victim_core(const llama_moe_stream_layer & sl, bool cancel, bool protect, bool prefetch, bool * pn_skipped) const;
    // the version, chosen once at start-up (the constructor sets it from vf_hook)
    int32_t (llama_moe_stream::*pick_victim_fn)(const llama_moe_stream_layer &, bool, bool, bool) const = nullptr;
    int32_t pick_victim_or_fallback(llama_moe_stream_layer & sl, int32_t wave, bool wplan, bool multi, std::unique_lock<std::mutex> & lock);
    int32_t pick_victim(const llama_moe_stream_layer & sl, bool cancel, bool protect, bool prefetch = false) const {
        return (this->*pick_victim_fn)(sl, cancel, protect, prefetch);
    }

    // Register the pools with the driver's page-locked pool (cudaHostRegister through the
    // backend's "ggml_backend_register_host_buffer"), so that RAM->VRAM copies run at full speed and
    // the pools cannot be swapped out. `reg` is the device's backend registry, nullptr = no CUDA.
    bool   pin_pools(ggml_backend_reg_t reg);

    // publish/inspect a slot's readiness, and the registry the CPU backend looks the table up in
    static std::atomic<uint32_t> & moe_slot_ready(llama_moe_stream_layer & sl, int32_t s) {
        return *reinterpret_cast<std::atomic<uint32_t> *>(&sl.slot_ready[s]);
    }
    void   register_pool(void * pool_data, size_t bytes, size_t slice_bytes, int n_as, int n_real, llama_moe_stream_layer * sl,
                         int pi = -1);  // the projection (0 gate, 1 up, 2 down) - its own flags with proj-ready (-1: the slot flag)
    // --moe-stream-proj-ready 1, chosen once at start-up: on; test only - MOE_STREAM_TEST_DOWN_DELAY_US (a needed read's down
    // request waits that long on its I/O thread before it reads), MOE_STREAM_TEST_PROJ_CHECK=1 (the flag invariants checked at every
    // decode layer start: EMPTY = 0 0 0, READY = 1 1 1, LOADING = a projection's flag 0 while bytes of it are still to land)
    bool    pr_on = false;
    int32_t pr_down_delay_us = 0;
    bool    pr_check = false;
    bool    pr_noclear = false;   // MOE_STREAM_TEST_PROJ_CHECK=2 (test only): the clears are skipped (the check's failure test)
    int64_t n_pr_checks = 0;
    // Test only with MOE_STREAM_TEST_PROJ_CHECK: remap calls with 2+ split needed reads, gate+up jobs queued after a
    // down job of the same call (must stay 0 outside an early flush), and the calls whose downs were queued early (the call waited)
    int64_t n_pr_ord_calls = 0, n_pr_ord_bad = 0, n_pr_ord_early = 0;
    int32_t pr_call_splits = 0;
    bool    pr_call_down_out = false, pr_call_early = false;
    void    pr_clear(llama_moe_stream_layer & sl, int32_t slot);    // mtx held: the three flags 0 (relaxed, as the overlap slot flag)
    void    pr_check_layer(llama_moe_stream_layer & sl);            // mtx held: MOE_STREAM_TEST_PROJ_CHECK
    void   unregister_pool(void * pool_data);

    // The issue timestamp of a read, and one histogram sample
    void   stamp_issue(llama_moe_stream_layer & sl, int32_t slot);
    void   add_hist(int64_t us);

    // Policy bookkeeping and comparison
    void   note_use(llama_moe_stream_layer & sl, int32_t s, bool demand);
    void   slot_settle_ready(llama_moe_stream_layer & sl, int32_t slot);   // a slot's bytes are in: READY (mtx held)
    bool   worse_than(const llama_moe_stream_layer & sl, int32_t a, int32_t b) const;
    // taker: recorder event; to_vram: VRAM follow's switch releases the slot of an expert that moved to the VRAM pool - not an
    // eviction (the recorder's EVICT carries bit6, no b0 eviction mark, no unused-guess count, no read-back later)
    void   drop_slot(llama_moe_stream_layer & sl, int32_t slot, int32_t taker = -1, bool taker_guess = false, bool to_vram = false,
                     bool by_copyback = false);
    void   promote_chunks(llama_moe_stream_layer & sl, int32_t slot);   // mtx held
    float  p1_value(const llama_moe_stream_layer & sl, int32_t slot) const;   // decayed value of a slot's expert
    void   p1_use(llama_moe_stream_layer & sl, int32_t e, float amount);      // add to an expert's value
    bool   p1_pending_guess(const llama_moe_stream_layer & sl, int32_t slot) const;   // guess whose layer has not run
    // One sample of a guess's achieved lead (issued -> the target layer asking for that expert)
    void   note_lead(llama_moe_stream_layer & sl, int32_t slot);
    // Abort with everything that is known about the read that failed every try (never returns)
    [[noreturn]] void fail_read(const llama_moe_stream_layer & sl, const char * when);
    // Called from a compute thread that was waiting for a slot whose read failed
    void   fail_slot_error();
    void   prefetch_oracle(llama_moe_stream_layer & sl, int64_t t0, int64_t n_tok);
    void   oracle_check(llama_moe_stream_layer & sl, int64_t t0, const int32_t * ids, int64_t n, int64_t k, bool decode);
    void   load_oracle(const std::string & prefix);
    size_t pool_total_bytes(uint32_t n_slots) const;
    // Repack one plain expert slice into a pool slot. Called from the I/O threads right after the
    // read, with the plain bytes in scratch (thread n), so the pool always holds the repacked layout the
    // kernel expects and the sidecar stays plain.
    void   repack_slot(llama_moe_stream_layer & sl, int32_t pi, int32_t slot, const uint8_t * plain, int ith);
    // The plain bytes land in the I/O thread's own scratch (one page-aligned expert block each)
    uint8_t * repack_scratch(int ith);
    // One scratch per I/O thread (a single one shared under a mutex serialized all the reads and
    // tripled the stall), page-aligned so the O_DIRECT read can target it
    // raw page-aligned blocks, not std::vector data: a vector copies into a malloc'd buffer that is not
    // page-aligned, and an O_DIRECT read into one of those fails with EFAULT
    std::vector<void *> repack_bufs;
    uint32_t             repack_bufs_n  = 0;   // how many I/O threads need one
    size_t               repack_buf_bytes = 0;
    std::mutex                       repack_bufs_mtx;
    // One 2-D view of a pool slot per (layer, projection), so the repack is a ggml_backend_tensor_set
    // on a tensor whose nbytes is one expert slice (the repack's set_tensor asserts that)
    // [n_io][3*n_layer], one per I/O thread: the view's data pointer is the slot the thread is
    // filling, and the repack reads it, so a view shared by the threads is a race
    std::vector<ggml_context *> repack_ctxs;
    std::vector<ggml_tensor *>  repack_views;
    size_t                     repack_view_stride = 0;   // 3*n_layer
    // The same view over a scratch buffer instead of a pool slot, for the read-back check when the
    // bytes it compares are not a slot (writing the repacked model bytes into slot 0 of the pool would
    // corrupt a live expert)
    mutable void *                     verify_repack_mem = nullptr;
    mutable size_t                     verify_repack_bytes = 0;   // its size: the largest slice checked so far
    mutable std::vector<ggml_context *> repack_verify_ctxs;
    mutable std::vector<ggml_tensor *>  repack_verify_views;
    // Compare the bytes of one projection of one expert of one layer in the model file
    // with what is in memory (a pool slot, or an expert of a whole GPU layer that the loader read).
    // Returns an empty string when they are the same, else what differs.
    std::string compare_model(int32_t il, int p, int32_t expert, const void * have, size_t size,
            const char * what, bool have_repacked) const;
    // One filled slot of this layer against the model, once per layer. mtx must be held.
    void   verify_slot(llama_moe_stream_layer & sl, int32_t slot, int32_t expert);
    // How many layers the check covers
    int64_t n_verify_layers() const;
    // The first and the last expert of the first and of the last streamed layer, compared
    // between the sidecar and the model. A repack that starts a slice late or loses its last page only
    // shows up at the ends of a tensor (one was found), and the router picks expert 511 of a layer
    // about 0.2 % of the time, so the check cannot leave it to luck.
    void   verify_boundaries(const std::vector<int32_t> & streamed);
    // The same for one expert of one whole GPU layer, whose bytes the stock loader read
    // into VRAM (the caller has already copied them back to the host).
    void   verify_gpu_tensor(int32_t il, const std::string & proj, const void * data, size_t size);
    // What the check has done so far, for the one line the run prints
    std::string verify_summary() const;
};

// The GPU half of the read-back check: the experts of the whole MoE layers the engine
// keeps in VRAM come from the model file through the stock loader, so nothing else in the engine ever
// looks at them. Called once, after the model is loaded, with a context that can name its tensors.
void llama_moe_stream_verify_gpu_tier(struct llama_context * ctx);

// The bytes of a model file, or the sum of the shards of a split one (the sidecar records that
// sum, and the fingerprint check compares against it)
size_t llama_moe_stream_model_bytes(const char * model_path);

// --moe-stream model: the experts are read from the model file itself. The word `model` is also what the path flags
// of the data that such a file can carry (--moe-stream-correction, --moe-stream-vram-experts-file) take to say "the copy in the model file".
#define MOE_STREAM_MODEL_WORD "model"

// the path the engine's own files are named after: the model file for --moe-stream model, else the experts file's metadata path as given
std::string llama_moe_stream_sidecar_arg(const std::string & moe_stream, const std::string & model_path);

// where the named data appended to a TokenWing model file is (correction, vram_experts, usage_seed, layer_order); false
// when the file is not one or has no such data
bool llama_moe_model_file_blob(const std::string & model_path, const char * name, uint64_t & offset, uint64_t & size);

// the plan of --moe-stream for a model and a sidecar, without loading or allocating anything
bool llama_moe_stream_print_plan(
        const std::string & model_path,
        const std::string & sidecar_json,
        int64_t             ram_budget,
        uint32_t            n_slots,
        uint32_t            io_threads,
        bool                prefetch,
        uint32_t            prefetch_m,
        const std::string & oracle,
        uint32_t            oracle_lead);

// A tensor with this flag is the id-remapping op of one of several expert waves of its layer. ggml
// copies a host weights tensor to a device once per graph, so a graph that has such an op keeps the
// ops that read host weights - the expert matmuls - on the host, next to the pool they read.
#define LLAMA_MOE_STREAM_TENSOR_WAVE (1 << 7)

// With --moe-stream-needed-early 1 and the GPU producer, the 1-ahead guess of a one-token graph
// carries this flag and its layer index in its last op param; the GPU backend writes it into the layer's pinned slot, then the
// call counter, then records the layer's "guess done" event (ggml-cuda/moe-mark.cu). The bit is handed to the backend at
// registration, so ggml has no flag of its own for it.
#define LLAMA_MOE_STREAM_TENSOR_GUESS_MARK (1 << 8)

// The router's ids of a one-token GPU-producer graph, with the layer in the last op param and the router
// input as src[1]: the GPU backend copies both into the layer's pinned router slot, then the call's number, then records the
// layer's "router done" event (ggml-cuda/moe-mark.cu); the scheduler's input hook gives the CPU split its inputs from that slot
#define LLAMA_MOE_STREAM_TENSOR_ROUTER_MARK (1 << 9)

// Tail waves: a GPU wave's ids op whose matmuls read the tail tensors - after its slot ids are final
// the op maps them to tail places and queues the copies (llama_moe_stream::tailw_gather)
#define LLAMA_MOE_STREAM_TENSOR_TAIL (1 << 10)
// Copy-ahead, one set: a wave's gate/up activation in a tail wave graph - after the GPU runs it, a "done" mark writes the
// layer's counter (ggml-cuda/moe-mark.cu, kind 2), so the VRAM-follow uploader may copy the next wave's gate/up parts into the places it read
#define LLAMA_MOE_STREAM_TENSOR_DONE_MARK (1 << 11)

// The scheduler's input hook (user_data = the engine)
bool llama_moe_stream_sched_input_hook(void * user_data, ggml_backend_t split_backend, int n_inputs, ggml_tensor ** inputs,
                                       ggml_tensor ** input_copies);
// The scheduler's copy hook (user_data = the engine): a preset target in the VRAM pool's tail for a GPU-wave pool copy
bool llama_moe_stream_sched_copy_hook(void * user_data, ggml_backend_t dst_backend, const ggml_tensor * src, ggml_tensor * copy,
                                      const ggml_tensor * consumer, int i_split, const ggml_cgraph * graph);
// The scheduler's split hook (user_data = the engine): the word-start sources whose marked guess the split held
void llama_moe_stream_sched_split_hook(void * user_data, ggml_backend_t split_backend, ggml_cgraph * graph);

// callbacks of the id-remapping custom op inserted by build_moe_ffn. The second one also gets the
// experts that the next layer's router predicts (b), the first one is used when prefetch is off.
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);
void llama_moe_stream_remap2(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata);
// --moe-stream-needed-early 1, the in-graph form of a CPU layer: op A (a = the router's ids; the needed reads, right after
// the router) and op B (a = the ids, b = the guess, c = op A's output, which only orders it after op A; the guess's adds and the
// rest of the remap; dst = the slot ids the expert matmul reads); op B without a guess (b = op A's output). userdata as remap2
void llama_moe_stream_remap_a(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);
void llama_moe_stream_remap_b(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c,
                              int ith, int nth, void * userdata);
void llama_moe_stream_remap_b1(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata);
// TEST ONLY (MOE_STREAM_TEST_GUESS_CHECK): the bit check of the combined guess step - one row (one target): the separate
// guess's logits / probabilities / ids against the combined step's (memcmp); a difference stops the run. src 0-2 separate, 3-5
// combined; userdata = llama_moe_stream::gchk_ud(il, d)
void llama_moe_stream_guess_check(ggml_tensor * dst, int ith, int nth, void * userdata);
// --moe-stream-wordstart D: the word-start op - a = the sources' guess columns [guess_width(1), columns], source by source
// in layer order, each source's la_targets in order; adds them to the waiting list (decode calls of one token only).
// userdata = the llama_moe_stream
void llama_moe_stream_wordstart(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);
// remap2 with the guess's sigmoid scores (c, F32, one per entry of b) for the chance table
void llama_moe_stream_remap3(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c,
                             int ith, int nth, void * userdata);
// The ids of the VRAM expert pool's device pass (the slot of a VRAM-held expert, else zero slice j of the token's row)
void llama_moe_stream_remap_vram(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);
// The early guess of a two-layer lead (role 1 of llama_moe_stream_wave)
void llama_moe_stream_pred2(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// Test only (MOE_STREAM_TEST_DOWN_SCALE_COUNT=1): an in-place op on a skipped layer's down input that only reads it - per row,
// whether its max |x| is above 1024 (the scale step would have scaled it) and above 2^22 (the skip would not be bit-exact)
void llama_moe_stream_down_scale_count(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata);

std::string llama_moe_os_read_file_range(const std::string & path, uint64_t off, uint64_t size);

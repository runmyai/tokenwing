#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"

class llama_memory_recurrent;
#include "ggml-opt.h"

#include <array>
#include <functional>
#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();
    void sampler_changed();               // set_sampler changed sampling.samplers (a reserve only if their graph changed)
    bool sampler_reserve_due() const;

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;
    struct llama_moe_stream * ctx_moe_stream() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    bool get_causal_attn() const;

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch_ext & batch_inp);
    int decode(const llama_batch_ext & batch_inp);

    // compat version
    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);

    // Checkpoints inside one call (llama_set_call_cuts / llama_get_call_checkpoint)
    bool   call_checkpoints_ok() const;
    bool   cc_reserve(size_t n_keep);   // the staging for n_keep checkpoints inside a call (llama-server: at load)
    bool   set_call_cuts(llama_seq_id seq_id, const llama_pos * cuts, const bool * keep, int32_t n);
    size_t get_call_checkpoint(llama_pos pos, uint8_t * dst, size_t size);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    // bytes of the context's memory (the KV cache and the recurrent state), all buffer types; fixed once the context exists
    int64_t memory_bytes() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    bool extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_result * get_gf_res_prev();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // create a new backend scheduler for a graph of the specified size and register the callbacks below
    void sched_new(size_t max_nodes, bool parallel);
    // ggml_backend_sched realloc callback, counts and traces the graph reallocations of this context
    static void sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * graph, bool backend_ids_changed, bool graph_size_changed, void * user_data);
    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        // The backend samplers' graph fingerprints the scheduler was last reserved with, and whether set_sampler changed the
        // samplers since; sched_reserve() runs for them only when the fingerprints differ
        std::map<llama_seq_id, uint64_t> reserved;
        bool                             changed = false;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    // graph reallocations during inference where the backend assignments did not change, split by whether the
    // graph size is the same as the previously scheduled one (unexpected) or different
    int32_t n_reallocs_same = 0;
    int32_t n_reallocs_diff = 0;
    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // --moe-stream-layer-major 1: whether this call runs layer by layer, the layer range of the graphs being built,
    // whether the memory replays its ubatch (reapply) instead of applying it, and each ubatch's hidden state between layer groups
    // (freed at the end of the call)
    bool    lm_on      = false;
    int32_t lm_il0     = 0;
    int32_t lm_il1     = -1;
    bool    lm_reapply = false;
    std::vector<std::vector<float>> lm_h;
    // The call's hidden state on the device of the layers (one tensor [n_embd, T]), allocated per call
    ggml_context_ptr          lm_hctx;
    ggml_backend_buffer_ptr   lm_hbuf;
    std::vector<uint8_t>      lm_output_backup;   // session-level host backup of model.output (--moe-stream-lend-output 1)
    ggml_tensor *             lm_hid  = nullptr;
    int64_t                   lm_row0 = 0;
    // The phase of the graph being built (0 whole layers, 1 phase 1, 2 the expert graph) and the call's other tensors:
    // the normed expert input [n_embd, T], the router's picks [k, T] (I32) and weights [k, T]; in the tail's extension when the pool
    // has one (lm_hbuf is then empty), else in lm_hbuf with lm_hid
    int32_t                   lm_phase = 0;
    ggml_tensor *             lm_x   = nullptr;
    ggml_tensor *             lm_moe = nullptr;
    ggml_tensor *             lm_inj = nullptr;
    ggml_tensor *             lm_ids = nullptr;
    ggml_tensor *             lm_w   = nullptr;
    // The moving window's signals: the call reached its end (lm_call_end's `complete`),
    // the per-layer events recorded after each streamed layer's expert work when the engine asks for them (lm_events), and the GPU
    // backend they are recorded on
    bool                              lm_complete = false;
    std::vector<ggml_backend_event_t> lm_evs;
    ggml_backend_t                    lm_ev_be = nullptr;
    void lm_signal_done(int il_from, int il_to);   // lm_layer_done for the streamed layers of [il_from, il_to)

    // This context's backend of the VRAM pool's GPU (--moe-stream-vram-device, else the first GPU); null = no such backend
    ggml_backend_t ms_pool_backend() const;
    int            sched_n_gpu() const;   // this context's GPU backends

    // The cuts armed for the next call (cc_cuts all, cc_keep the ones whose state is copied), the call that runs them
    // (cc_active), the staging (one block of cc_bytes per kept cut: the sequence cell's rows of every recurrent tensor, at
    // cc_off[tensor]; pinned host memory of the first GPU device when there is one), the layers copied per kept cut, and the cuts
    // whose copy is complete after the call (cc_done; read after a synchronize)
    llama_seq_id                                  cc_seq = -1;
    std::vector<llama_pos>                        cc_cuts;
    std::vector<llama_pos>                        cc_keep;
    bool                                          cc_active = false;
    ggml_backend_buffer_ptr                       cc_buf;
    std::vector<uint8_t>                          cc_host;
    size_t                                        cc_bytes = 0;
    std::map<const ggml_tensor *, size_t>         cc_off;
    std::vector<std::vector<bool>>                cc_layer_done;
    std::vector<llama_pos>                        cc_done;

    llama_memory_recurrent * cc_recr() const;
    uint8_t * cc_stage(size_t i_keep);
    void cc_copy(const llama_ubatch & ub, int il_from, int il_to);   // after a ubatch's graph for layers [il_from, il_to)

    bool lm_layer_major_ok(uint32_t n_tokens_all);
    bool lm_force_off = false;   // this call runs ubatch by ubatch (a layer-major attempt of it failed)
    // VRAM lending for prompts: the lent pieces of the VRAM pool under the GPU's compute buffer
    int      lend_be = -1;                                // the backend (index in backend_ptrs) whose buffer lends: the pool's GPU
    std::vector<ggml_backend_buffer_t> lend_bufs;         // the pieces set on the scheduler now (empty = the real compute buffer)
    std::vector<ggml_backend_buffer_t> lend_bufs_old;     // the last set's buffers, freed one set later (a finished graph's tensors name them)
    std::vector<int32_t> lend_U;                          // the places per group of the pieces set now
    std::vector<int32_t> lend_start;                      // and their first places (after the call's front places)
    uint64_t lend_gen = 0, lend_gen_alloc = 0;             // bumped at every set / clear; the gen the last graph was allocated in (its reuse)
    int64_t  lend_kv = 0;                                 // the context the running call reaches
    int64_t  lm_lend_rows = 0;                            // (lend_lm_graphs: the expert graph's rows for the measure)
    bool     lend_measure(const llama_memory_context_i * mctx, const std::vector<int64_t> & t_pts, std::vector<int64_t> & by_t, int64_t & lm,
                         bool alloc, std::map<std::string, std::pair<int64_t, int>> * stays);   // the prompt graphs' GPU plans
    std::vector<ggml_backend_buffer_t> lend_trial;        // the reserve's trial pieces (no vacate)
    int64_t  lend_T = 0;                                  // the running call's tokens
    bool     lend_no_grow = false;                        // the real buffer may not grow now (a prompt call)
    bool     lend_again = false;                          // a full card ended the call: decode returns 3 (processed again)
    bool     lend_start_failed = false;                   // the running prompt call's lend at its start failed (it runs on the real buffer)
    std::vector<std::pair<int64_t, bool>> prompt_calls;  // the first 64 prompt calls' tokens (and layer-major or not), for the exit line
    int64_t  n_prompt_calls = 0;                         // (all of them)
    bool     lend_lm_graphs(const llama_memory_context_i * mctx, const std::function<bool(void)> & each);   // the layer-major graphs, one by one
    bool     lend_decide(const llama_memory_context_i * mctx); // at a reserve: lend or not, the plan's a + b x kv, the trial into the pieces
    void     lend_set(std::vector<ggml_backend_buffer_t> && bufs, const std::vector<int32_t> & U, bool may_grow);
    void     lend_clear();
    void     lend_call_start(bool dec, int64_t T, int64_t kv);   // decode: a prompt call lends (before its graphs), a reply word gives back
    bool     lend_alloc_retry();                           // a graph whose allocation failed: the real buffer grows / the call ends
    void lm_measure_expert_graph();   // the room one layer-major expert graph takes per chunk and wave
    bool lm_xg_fits(int64_t n_tok) const;
    struct { bool on = false; int64_t ub = 1, w = 1, base_n = 0, per_n = 0, room_n = 0, base_m = 0, per_m = 0, room_m = 0; } lm_xg;
    bool lm_reprocess = false;   // prompt split could not retry (a sliding window, not at 0): decode returns 3, the caller processes again
    bool rtrial_in_trial  = false;   // a reserve of vx_setup's pool sizing (its failure shrinks the pool: no -ub halving)
    int  rtrial_trial_n   = 0;       // (the trial reserves so far: MOE_STREAM_TEST_PROMPT_RESERVE_TRIAL_FAIL)
    int  lm_group() const;
    bool lm_group_test() const;
    bool lm_remove_call();

    // A prompt call's own host memory outside the expert streaming engine's RAM plan (the engine gives back RAM slots for it)
    int64_t rg_call_need(uint32_t n_tokens_all, const llama_memory_context_i & mctx);
    int64_t rg_cb_counted = 0;   // the pageable host compute buffer's bytes already counted (they hold RAM after that call)

    // The reserve prints the compute buffers it asked for, the first real graph prints the
    // ones it uses, and the two are the measurement of whether the reserve reserves for the wave rule
    int n_reserve_printed     = 0;
    bool printed_compute_bufs = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    // the compute plan that needs to be prepared for ggml_graph_compute()
    // since https://github.com/ggml-org/ggml/issues/287
    struct ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

        int n_threads;
        struct ggml_threadpool * threadpool;

        // abort ggml_graph_compute when true
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // use only reference implementations
        bool use_ref;
    };

    // numa strategies
    enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED   = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE    = 2,
        GGML_NUMA_STRATEGY_NUMACTL    = 3,
        GGML_NUMA_STRATEGY_MIRROR     = 4,
        GGML_NUMA_STRATEGY_COUNT
    };

    GGML_BACKEND_API void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
    GGML_BACKEND_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    GGML_BACKEND_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);

    GGML_BACKEND_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);

    GGML_BACKEND_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

    GGML_BACKEND_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    GGML_BACKEND_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

    GGML_BACKEND_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    GGML_BACKEND_API struct ggml_threadpool *      ggml_threadpool_new           (struct ggml_threadpool_params  * params);
    GGML_BACKEND_API void                          ggml_threadpool_free          (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API int                           ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

    // ggml_graph_plan() has to be called before ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
                  const struct ggml_cgraph * cgraph,
                                       int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                    struct ggml_threadpool * threadpool /* = NULL */ );
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

    // same as ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);

    //
    // system info
    //

    // x86
    GGML_BACKEND_API int ggml_cpu_has_sse3       (void);
    GGML_BACKEND_API int ggml_cpu_has_ssse3      (void);
    GGML_BACKEND_API int ggml_cpu_has_avx        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx_vnni   (void);
    GGML_BACKEND_API int ggml_cpu_has_avx2       (void);
    GGML_BACKEND_API int ggml_cpu_has_bmi2       (void);
    GGML_BACKEND_API int ggml_cpu_has_f16c       (void);
    GGML_BACKEND_API int ggml_cpu_has_fma        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512     (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vbmi(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vnni(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_bf16(void);
    GGML_BACKEND_API int ggml_cpu_has_amx_int8   (void);
    // ARM
    GGML_BACKEND_API int ggml_cpu_has_neon       (void);
    GGML_BACKEND_API int ggml_cpu_has_arm_fma    (void);
    GGML_BACKEND_API int ggml_cpu_has_fp16_va    (void);
    GGML_BACKEND_API int ggml_cpu_has_dotprod    (void);
    GGML_BACKEND_API int ggml_cpu_has_matmul_int8(void);
    GGML_BACKEND_API int ggml_cpu_has_sve        (void);
    GGML_BACKEND_API int ggml_cpu_get_sve_cnt    (void);  // sve vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_sme        (void);
    GGML_BACKEND_API int ggml_cpu_has_sme2       (void);
    // other
    GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void);
    GGML_BACKEND_API int ggml_cpu_get_rvv_vlen   (void);  // risc-v vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_vsx        (void);
    GGML_BACKEND_API int ggml_cpu_has_vxe        (void);
    GGML_BACKEND_API int ggml_cpu_has_wasm_simd  (void);
    GGML_BACKEND_API int ggml_cpu_has_llamafile  (void);

    // Internal types and functions exposed for tests and benchmarks

    typedef void (*ggml_vec_dot_t)  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx,
                                       const void * GGML_RESTRICT y, size_t by, int nrc);

    struct ggml_type_traits_cpu {
        ggml_from_float_t        from_float;
        ggml_vec_dot_t           vec_dot;
        enum ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
    };

    GGML_BACKEND_API const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);

    GGML_BACKEND_API void ggml_cpu_init(void);

    //
    // CPU backend
    //

    GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);

    GGML_BACKEND_API bool ggml_backend_is_cpu                (ggml_backend_t backend);
    GGML_BACKEND_API void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);

    // The CPU MoE switches. Process-wide: every model in the process uses the last value set (two models: the last
    // one wins). Each graph compute takes one snapshot of them before its threads start.
    // ggml_cpu_set_mxfp4_kernel: the kernel mul_mat_id uses on plain MXFP4: 0 = ggml_vec_dot_mxfp4_q8_0, one row per call
    // (default); 1 = the 4-row AVX-512 VNNI kernel with 256-bit registers; 2 = the same with 512-bit registers. Same bits
    // as kernel 0. prefetch_blocks: > 0 = that many 17-byte blocks ahead in the row, < 0 = the same block of the next
    // 4-row group, 0 = none (kernels 1 and 2 only; x86 only, other CPUs keep kernel 0).
    GGML_BACKEND_API void ggml_cpu_set_mxfp4_kernel(int kernel, int prefetch_blocks);
    // ggml_cpu_set_moe_fuse: run mul_mat_id(gate) + mul_mat_id(up) + swiglu (or Ling's clamped silu/clamp/mul) of one token
    // as one fused step (same bits); 0 = off (default)
    GGML_BACKEND_API void ggml_cpu_set_moe_fuse(int on);

    // ggml_cpu_set_q8_tiled(N), N > 0 = q8_0 weights take the tiled matmul (ggml-cpu/tiled), 1 = from 12 activation rows, else from N, with
    // Q8_K activations and a float scale per 32 values (rounding differs from the Q8_0 path; 1 row is unchanged). Off by default;
    // GGML_CPU_Q8_TILED=1 also switches it on (test tools). x86 with AVX2 or AVX-512 VNNI only
    GGML_BACKEND_API void ggml_cpu_set_q8_tiled(int on);

    // ggml_cpu_set_fa_kernel(1) = CPU flash attention with a q8_0 KV cache takes the tiled prompt path (K / V tiles
    // converted to f32 once per tile) and, for one query row, the KV split with all q heads of a KV head per K / V row read.
    // 0 = as before (default); GGML_CPU_FA_KERNEL=1 also switches it on (test tools)
    GGML_BACKEND_API void ggml_cpu_set_fa_kernel(int kernel);

    // ggml_cpu_set_fuse_small(mask) = fewer small CPU steps, the same results: 1 = a router chain (sigmoid, top-k, get_rows,
    // sum_rows, clamp, div) of a reply as one step on one thread, 2 = a guess chain (sigmoid, top-k, cont) the same, 64 = consecutive
    // one-row matmuls on the same input with one conversion and two barriers. 0 = off (default). The counts of the steps taken:
    GGML_BACKEND_API void ggml_cpu_set_fuse_small(int mask);

    // ggml_cpu_set_mmid_merge(1) = in the tiled MUL_MAT_ID (prompts), a thread's consecutive 64-row weight groups of an
    // expert run as one window of up to 256 rows, so the expert's activation rows are unpacked once per thread instead of once per
    // group. The same bits (every value is summed in the same order). 0 = as before (default); GGML_CPU_MMID_MERGE=1 also (test tools)
    GGML_BACKEND_API void ggml_cpu_set_mmid_merge(int on);
    GGML_BACKEND_API int  ggml_cpu_get_mmid_merge(void);
    GGML_BACKEND_API int64_t ggml_cpu_get_mmid_merge_count(void);   // the groups whose activation unpack was saved (all graphs)

    // ggml_cpu_set_iqk(N) = MUL_MAT_ID at prompt sizes (32+ tokens, x86 with AVX2 or AVX-512) uses ik_llama.cpp's kernels for the experts that
    // get N or more rows (Q2_0, Q4_0, Q8_0: max(8, N/4)), in the formats where they win (Q3_K, Q4_K, Q5_K, IQ4_XS, IQ3_S, Q2_0, Q4_0, Q8_0; not Q6_K). Rounding-level differences.
    // 0 = off (default); GGML_CPU_IQK=N also (test tools). Counts: experts computed by the port, experts it left to ggml's own path, their rows, experts started before their bytes landed (a stream test: must be 0)
    GGML_BACKEND_API void ggml_cpu_set_iqk(int min_rows);
    GGML_BACKEND_API int  ggml_cpu_get_iqk(void);
    GGML_BACKEND_API void ggml_cpu_get_iqk_counts(int64_t * experts, int64_t * experts_other, int64_t * rows, int64_t * not_landed);

    // A MUL_MAT_MULTI / ARGSORT node may carry a "skip full targets" tag (op params, written by llama's combined guess): the
    // rows whose target layer's byte in that array is 1 are not computed (the matmul writes 0, the top-k writes the ids 0..k-1). The
    // count of matmul rows skipped (all graphs of the process):
    GGML_BACKEND_API int64_t ggml_cpu_get_guess_skip_rows(void);
    #define GGML_CPU_GSF_MAGIC 0x5C7
    GGML_BACKEND_API void ggml_cpu_get_fuse_small_counts(int64_t * router, int64_t * guess, int64_t * mm);
    GGML_BACKEND_API int  ggml_cpu_get_fa_kernel(void);
    GGML_BACKEND_API int  ggml_cpu_get_q8_tiled(void);
    GGML_BACKEND_API int64_t ggml_cpu_moe_fuse_count(void);   // fused steps run so far
    // ggml_cpu_set_moe_dispatch(1) = a one-token mul_mat_id (decode) runs ik_llama.cpp's expert split
    // (GGML_EXPERT_CHUNKING): one chunk queue over all its experts, one chunk of nr0/nth rows (16-row multiples) per
    // thread and expert, ready experts first, whole activation rows quantized per thread. Same bits as off (the
    // same kernel on the same rows; only which thread computes which rows changes); 0 = off (default)
    GGML_BACKEND_API void ggml_cpu_set_moe_dispatch(int on);
    GGML_BACKEND_API int64_t ggml_cpu_moe_dispatch_count(void);   // one-token mul_mat_id ops run that way so far
    // GGML_MOE_CPU_TIMERS=1 (test only): the CPU expert step's counters as one line of text; returns 0 (and writes nothing) when off
    GGML_BACKEND_API int      ggml_cpu_m11_report(char * buf, size_t n);
    // The one-token expert steps (the fused gate/up step and the dispatch split): ggml_cpu_set_moe_pieces(P) cuts
    // each expert into P work pieces per thread instead of 1 (16-row boundaries; the threads take pieces as they finish, so the
    // last piece of a step is shorter); ggml_cpu_set_moe_conv1(1) converts each activation row on one thread instead of a
    // slice per thread (upstream #29308; one token = one row = thread 0). Same bits as the defaults (the same kernel on the same rows) - 1 / 0 = as before
    GGML_BACKEND_API void ggml_cpu_set_moe_pieces(int n);
    GGML_BACKEND_API void ggml_cpu_set_moe_conv1(int on);
    // The CPU track: ggml_cpu_set_track(bits) - CPU-only work-split changes, each the same bits as off
    //   GGML_CPU_TRACK_ROWS (1): GET_ROWS (f32) with fewer rows than threads splits each row's columns over the threads
    //   GGML_CPU_TRACK_PICK (2): MOE_GROUP_TOPK takes the top k from the kept groups only (a partial selection); whenever two of
    //                            the k + 1 best scores are equal (or a score is NaN) it runs the full sort instead
    //   GGML_CPU_TRACK_CAT  (4): CONCAT (f32) with fewer planes than threads splits its rows over the threads
    // 0 = as before (default). The environment variable GGML_CPU_TRACK=<bits> sets the start value (test-backend-ops uses it).
    // ggml-cpu takes one snapshot per graph compute.
    #define GGML_CPU_TRACK_ROWS 1
    #define GGML_CPU_TRACK_PICK 2
    #define GGML_CPU_TRACK_CAT  4
    GGML_BACKEND_API void ggml_cpu_set_track(int bits);
    // The CPU track: the gated delta-net step of one token batch with the state read in place and written in place
    // (built by the model as a custom op on CPU layers, --moe-stream-cpu-state-inplace). dst = the attention output
    // [S_v, H, n_tokens, n_seqs] f32; dst->src: 0 q, 1 k, 2 v, 3 g, 4 beta (as GGML_OP_GATED_DELTA_NET), 5 the cache's
    // states [S_v*S_v*H, rows] f32 (read at row ids[s]), 6 ids (i32 [n_seqs]), 7 the destination rows [S_v*S_v*H, n_seqs] f32
    // (a view of the same cache). The same numbers as GGML_OP_GATED_DELTA_NET with K = 1 + the copy into the cache: every
    // pass of that op works row by row of the state, so each row goes through all of them at once. test_skip_write != 0
    // (TEST ONLY): the new state is not kept (the old one is put back) - a skipped state write must change the text.
    GGML_BACKEND_API void ggml_cpu_gated_delta_net_inplace(struct ggml_tensor * dst, int ith, int nth, int test_skip_write);
    // The readiness registry of the MoE streaming engine's expert pools (src/llama-moe-stream.cpp is the
    // consumer). Declared here so the compiler checks the definitions in ggml-cpu.c against the callers.
    GGML_BACKEND_API void ggml_cpu_moe_slots_register(const void * pool, size_t bytes, size_t slice_bytes, int n_as,
                                                      const uint32_t * ready, const int32_t * err,
                                                      void (* on_error)(void *, int32_t), void * ctx,
                                                      const int64_t * ready_us);
    GGML_BACKEND_API void ggml_cpu_moe_slots_unregister(const void * pool);
    GGML_BACKEND_API int64_t ggml_cpu_moe_slots_take_wait(const void * pool);   // the overlap stall
    // The copy-back flags of a pool's slots (bit0 = a copy-back from the VRAM pool in flight), and the waits on them
    // since the last take (cb_us) with the old all-in-one booking (all_us); waits on those slices are not in take_wait's stall
    GGML_BACKEND_API void ggml_cpu_moe_slots_set_cb(const void * pool, const uint8_t * cb, int n);
    GGML_BACKEND_API void ggml_cpu_moe_slots_take_wait_cb(const void * pool, int64_t * cb_us, int64_t * all_us);
    GGML_BACKEND_API void ggml_cpu_moe_slots_take_handoff(const void * pool, int64_t * sum_us, int64_t * n, int64_t * max_us);
    GGML_BACKEND_API void ggml_cpu_moe_slots_skip_zero(const void * pool, int first_zero);   // --moe-stream-cpu-skip-zero
    // --moe-stream-proj-ready 1: a gate / up pool learns its expert's down pool flags (`later`, the same slot index);
    // mul_mat_id then counts, once per expert and op, a projection whose compute starts while that expert's down has not landed.
    // take_early returns that count since the last take and resets it
    GGML_BACKEND_API void    ggml_cpu_moe_slots_set_later(const void * pool, const uint32_t * later);
    GGML_BACKEND_API int64_t ggml_cpu_moe_slots_take_early(const void * pool);
    GGML_BACKEND_API void ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
    GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

    GGML_BACKEND_API void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref);

    GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);

    GGML_BACKEND_API void ggml_cpu_fp32_to_fp32(const float *,       float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_i32 (const float *,     int32_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_fp16(const float *, ggml_fp16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *, float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_bf16(const float *, ggml_bf16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_bf16_to_fp32(const ggml_bf16_t *, float *, int64_t);

#ifdef __cplusplus
}
#endif

#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"
#include "unary-ops.h"
#include "binary-ops.h"
#include "vec.h"
#include "ops.h"
#include "ggml.h"
#include "common.h"
#include "tiled/tiled.h"
#include "iqk/iqk_port.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
float ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
float ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
float ggml_table_f32_ue4m3[1 << 8];

#if defined(__ARM_ARCH)
struct ggml_arm_arch_features_type {
    int sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#if defined(__riscv)
struct ggml_riscv_arch_features_type {
    int rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_row_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0] = {
        .from_float               = quantize_row_q2_0,
        .vec_dot                  = ggml_vec_dot_q2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_row_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_row_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_row_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_i32,
    },
};

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

//
// Threading defs
//

typedef pthread_t          ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;

    // The CPU MoE switches as this graph sees them: ggml_graph_compute copies them here before any thread starts,
    // so every thread of one graph takes the same decision (the fused step has a barrier inside)
    int moe_fuse;      // 1 = fused gate+up step
    int fuse_small;    // ggml_cpu_set_fuse_small's mask (1 router chain, 2 guess chain, 64 same-input matmuls)
    int mxfp4_x4;      // 0 = 1-row MXFP4 kernel, else the 4-row kernel's register width (256 or 512)
    int mxfp4_x4_pf;   // its prefetch mode (> 0 blocks ahead in the row, < 0 the next 4-row group, 0 none)
    int moe_dispatch;  // 1 = ik's expert split for one-token mul_mat_id
    int moe_pieces;    // work pieces per thread and expert in the one-token expert steps (1 = as before)
    int moe_conv1;     // 1 = the one-token activation row is converted by one thread
    int cpu_track;     // the CPU track: GGML_CPU_TRACK_* bits (ggml_cpu_set_track)

    // Hand-off profiler (GGML_PROFILE_HANDOFF): per-node host stamps of this graph, allocated only when the timers are on
    int       ho_on;       // 1 = this graph runs the timed thread loop
    int       ho_nth;      // the threads ho_ts has room for (cplan->n_threads)
    int64_t * ho_wt;       // [thread][node] the thread's slot waits inside the node
    int64_t * ho_ts;       // [thread][node][start, end] in us (one thread's stamps together: no false sharing);
                           // thread 0's start is 0 for a node that did not run
    int     * ho_fused;    // [node] nodes the CPU backend fused into it (thread 0)
    size_t    ho_cap;      // nodes ho_ts / ho_fused have room for
    int64_t   ho_t_final;  // thread 0 after the last barrier
    struct ggml_handoff_cpu_node * ho_nodes;   // the records handed to ggml_handoff_cpu_graph
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    struct ggml_threadpool * threadpool;
    int ith;
};

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#elif defined(__riscv)
static inline void ggml_thread_cpu_relax(void) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #else
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#else
static inline void ggml_thread_cpu_relax(void) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct ggml_state {
    struct ggml_numa_nodes numa;
};

static struct ggml_state g_state = {0};

void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value) {
    atomic_store_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

int ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value) {
    return atomic_fetch_add_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct ggml_numa_node * node = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    if (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

#if defined(__ARM_ARCH)
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static void ggml_init_arm_arch_features(void) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#else
static void ggml_init_arm_arch_features(void) {}
#endif
#endif // __ARM_ARCH

#if defined(__riscv) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
static void ggml_init_riscv_arch_features(void) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#else
static void ggml_init_riscv_arch_features(void) {}
#endif

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        return;
    }

    // If tiled is supported, it will execute the full op here and we return
    if (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = ggml_is_contiguous(src1);

    if (!params->use_ref && src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        const bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        const int64_t bs = ggml_blck_size(vec_dot_type);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    const void * src1_row  = (const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10;
                    void       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;

                    const int64_t ne10_block_size = (ne10_block_end - ne10_block_start) * bs;

                    if (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_to_fp32((const ggml_fp16_t *) src1_row, (float *) wdata_row, ne10_block_size);
                    } else if (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_to_fp32((const ggml_bf16_t *) src1_row, (float *) wdata_row, ne10_block_size);
                    } else {
                        from_float((const float *) src1_row, wdata_row, ne10_block_size);
                    }
                }
            }
        }
    #endif
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE
    if (!params->use_ref && src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

// The CPU MoE switches (ggml_cpu_set_mxfp4_kernel, ggml_cpu_set_moe_fuse). Process-wide: with two models in one
// process the last value set wins. Atomic; ggml_graph_compute snapshots them into the threadpool and the ops read only that.
static _Atomic int g_ggml_cpu_mxfp4_x4    = 0;   // 0 = off, 256 or 512 = the 4-row kernel's register width
static _Atomic int g_ggml_cpu_mxfp4_x4_pf = 0;
static _Atomic int g_ggml_cpu_moe_fuse    = 0;
static bool        g_ggml_cpu_tiled_forced = false;   // GGML_CPU_TILED_MM_FORCE=1 (set in ggml_cpu_init): no fusing, the
                                                      // unfused path would take the tiled kernel

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define GGML_CPU_HAVE_MXFP4_X4 1   // the 4-row kernels live in arch/x86/quants.c
#endif

#ifdef GGML_CPU_HAVE_MXFP4_X4
typedef void (*ggml_vec_dot_x4_t)(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, int width, int pf);
// The 4-row kernel of a weight type (each gives the same bits as its type's 1-row vec_dot), NULL for a type
// without one. One switch (--moe-stream-cpu-x4 / ggml_cpu_set_mxfp4_kernel) turns them all on. Q5_K's 4-row kernel
// (ggml_vec_dot_q5_K_q8_K_x4) is not used: ggml's 1-row one already reads at the speed of reading the bytes alone (kernel benchmark)
static ggml_vec_dot_x4_t ggml_cpu_vec_dot_x4(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_MXFP4:  return ggml_vec_dot_mxfp4_q8_0_x4;
        case GGML_TYPE_IQ4_XS: return ggml_vec_dot_iq4_xs_q8_K_x4;
        default:               return NULL;
    }
}
#endif

void ggml_cpu_set_mxfp4_kernel(int kernel, int prefetch_blocks) {
#ifdef GGML_CPU_HAVE_MXFP4_X4
    atomic_store_explicit(&g_ggml_cpu_mxfp4_x4, kernel == 1 ? 256 : kernel == 2 ? 512 : 0, memory_order_relaxed);
    atomic_store_explicit(&g_ggml_cpu_mxfp4_x4_pf, prefetch_blocks, memory_order_relaxed);
#else
    GGML_UNUSED(kernel);
    GGML_UNUSED(prefetch_blocks);   // other CPUs keep the 1-row kernel
#endif
}

void ggml_cpu_set_moe_fuse(int on) {
    atomic_store_explicit(&g_ggml_cpu_moe_fuse, on ? 1 : 0, memory_order_relaxed);
}

// q8_0 on the tiled matmul (ggml_cpu_set_q8_tiled, see ggml-cpu.h); GGML_CPU_Q8_TILED=1 = on from the start (test tools)
static _Atomic int g_ggml_cpu_q8_tiled = -1;   // -1 = not set yet: the environment decides at the first read

void ggml_cpu_set_q8_tiled(int on) {
    atomic_store_explicit(&g_ggml_cpu_q8_tiled, on > 0 ? on : 0, memory_order_relaxed);
}

// A thread's MUL_MAT_ID groups as one tiled window (ggml_cpu_set_mmid_merge, see ggml-cpu.h); GGML_CPU_MMID_MERGE=1 for
// the test tools
static _Atomic int g_ggml_cpu_mmid_merge = -1;   // -1 = not set yet: the environment decides at the first read
static _Atomic int64_t g_ggml_cpu_mmid_merge_n = 0;   // activation unpacks saved (one per merged group)

void ggml_cpu_mmid_merge_count_add(int64_t n);   // tiled.cpp
void ggml_cpu_mmid_merge_count_add(int64_t n) {
    atomic_fetch_add_explicit(&g_ggml_cpu_mmid_merge_n, n, memory_order_relaxed);
}

int64_t ggml_cpu_get_mmid_merge_count(void) {
    return atomic_load_explicit(&g_ggml_cpu_mmid_merge_n, memory_order_relaxed);
}

void ggml_cpu_set_mmid_merge(int on) {
    atomic_store_explicit(&g_ggml_cpu_mmid_merge, on ? 1 : 0, memory_order_relaxed);
}

int ggml_cpu_get_mmid_merge(void) {
    int v = atomic_load_explicit(&g_ggml_cpu_mmid_merge, memory_order_relaxed);
    if (v < 0) {
        const char * env = getenv("GGML_CPU_MMID_MERGE");
        v = env != NULL && atoi(env) == 1 ? 1 : 0;
        int expected = -1;
        atomic_compare_exchange_strong(&g_ggml_cpu_mmid_merge, &expected, v);
        v = atomic_load_explicit(&g_ggml_cpu_mmid_merge, memory_order_relaxed);
    }
    return v;
}

// ik_llama.cpp's expert kernels for MUL_MAT_ID at prompt sizes (ggml_cpu_set_iqk, see ggml-cpu.h); GGML_CPU_IQK=N for the test tools
static _Atomic int     g_ggml_cpu_iqk = -1;   // -1 = not set yet: the environment decides at the first read
static _Atomic int64_t g_ggml_cpu_iqk_n[4] = { 0, 0, 0, 0 };   // experts computed by the port, by ggml's own path in a node it could take, rows, computed before their bytes landed

void ggml_cpu_set_iqk(int min_rows) {
    atomic_store_explicit(&g_ggml_cpu_iqk, MAX(0, min_rows), memory_order_relaxed);
}

int ggml_cpu_get_iqk(void) {
    int v = atomic_load_explicit(&g_ggml_cpu_iqk, memory_order_relaxed);
    if (v < 0) {
        const char * env = getenv("GGML_CPU_IQK");
        v = env != NULL ? MAX(0, atoi(env)) : 0;
        int expected = -1;
        atomic_compare_exchange_strong(&g_ggml_cpu_iqk, &expected, v);
        v = atomic_load_explicit(&g_ggml_cpu_iqk, memory_order_relaxed);
    }
    return v;
}

void ggml_cpu_get_iqk_counts(int64_t * experts, int64_t * experts_other, int64_t * rows, int64_t * not_landed) {
    *experts       = atomic_load_explicit(&g_ggml_cpu_iqk_n[0], memory_order_relaxed);
    *experts_other = atomic_load_explicit(&g_ggml_cpu_iqk_n[1], memory_order_relaxed);
    *rows          = atomic_load_explicit(&g_ggml_cpu_iqk_n[2], memory_order_relaxed);
    *not_landed    = atomic_load_explicit(&g_ggml_cpu_iqk_n[3], memory_order_relaxed);
}

// which MUL_MAT_ID nodes the port may take: prompt sizes only (replies stay on ggml's own path), an F32 result and activations
#define GGML_IQK_MIN_TOKENS 32
static bool ggml_cpu_iqk_node(const struct ggml_tensor * dst) {
    const struct ggml_tensor * a = dst->src[0], * b = dst->src[1];
    return ggml_cpu_get_iqk() > 0 && dst->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && b->ne[3] == 1 && b->ne[2] >= GGML_IQK_MIN_TOKENS &&
           ggml_is_contiguous(b) && iqk_port_node_ok(a->type, a->ne[0], a->ne[1]);
}

// Matmul rows of full guess targets skipped (see ggml-cpu.h)
static _Atomic int64_t g_ggml_cpu_gsf_rows = 0;

int64_t ggml_cpu_get_guess_skip_rows(void) {
    return atomic_load_explicit(&g_ggml_cpu_gsf_rows, memory_order_relaxed);
}

// Fewer small CPU steps (ggml_cpu_set_fuse_small, see ggml-cpu.h); the counts of the runs taken (all graphs of the process)
static _Atomic int     g_ggml_cpu_fuse_small = 0;
static _Atomic int64_t g_ggml_cpu_fuse_small_n[3] = { 0, 0, 0 };   // router chains, guess chains, same-input matmul runs

void ggml_cpu_set_fuse_small(int mask) {
    atomic_store_explicit(&g_ggml_cpu_fuse_small, mask & (1 | 2 | 64), memory_order_relaxed);
}

void ggml_cpu_get_fuse_small_counts(int64_t * router, int64_t * guess, int64_t * mm) {
    *router = atomic_load_explicit(&g_ggml_cpu_fuse_small_n[0], memory_order_relaxed);
    *guess  = atomic_load_explicit(&g_ggml_cpu_fuse_small_n[1], memory_order_relaxed);
    *mm     = atomic_load_explicit(&g_ggml_cpu_fuse_small_n[2], memory_order_relaxed);
}

// The CPU flash attention kernel choice (ggml_cpu_set_fa_kernel, see ggml-cpu.h); GGML_CPU_FA_KERNEL=1 for the test tools
static _Atomic int g_ggml_cpu_fa_kernel = -1;   // -1 = not set yet: the environment decides at the first read

void ggml_cpu_set_fa_kernel(int kernel) {
    atomic_store_explicit(&g_ggml_cpu_fa_kernel, kernel == 1 ? 1 : 0, memory_order_relaxed);
}

int ggml_cpu_get_fa_kernel(void) {
    int v = atomic_load_explicit(&g_ggml_cpu_fa_kernel, memory_order_relaxed);
    if (v < 0) {
        const char * env = getenv("GGML_CPU_FA_KERNEL");
        v = env != NULL && atoi(env) == 1 ? 1 : 0;
        int expected = -1;
        atomic_compare_exchange_strong(&g_ggml_cpu_fa_kernel, &expected, v);
        v = atomic_load_explicit(&g_ggml_cpu_fa_kernel, memory_order_relaxed);
    }
    return v;
}

int ggml_cpu_get_q8_tiled(void) {
    int v = atomic_load_explicit(&g_ggml_cpu_q8_tiled, memory_order_relaxed);
    if (v < 0) {
        const char * env = getenv("GGML_CPU_Q8_TILED");
        v = env != NULL ? MAX(0, atoi(env)) : 0;
        int expected = -1;
        atomic_compare_exchange_strong(&g_ggml_cpu_q8_tiled, &expected, v);
        v = atomic_load_explicit(&g_ggml_cpu_q8_tiled, memory_order_relaxed);
    }
    return v;
}

// ik's expert split for one-token mul_mat_id (ggml_cpu_set_moe_dispatch, see ggml-cpu.h)
static _Atomic int     g_ggml_cpu_moe_dispatch = 0;
static int             g_ggml_cpu_handoff = 0;   // GGML_PROFILE_HANDOFF, set once in ggml_cpu_init
static _Atomic int64_t g_ggml_cpu_moe_dispatch_count = 0;

void ggml_cpu_set_moe_dispatch(int on) {
    atomic_store_explicit(&g_ggml_cpu_moe_dispatch, on ? 1 : 0, memory_order_relaxed);
}

int64_t ggml_cpu_moe_dispatch_count(void) {
    return atomic_load_explicit(&g_ggml_cpu_moe_dispatch_count, memory_order_relaxed);
}

// ===== CPU expert-step timers (GGML_MOE_CPU_TIMERS=1, test only; read once; with it off the cost is one predictable branch per expert op) =====
// What the one-token expert step does per word on the CPU, and the TAIL of a layer that waits for the SSD: the fused gate+up op and the down
// op that follows it are timed together. Layer = one (fused op, down op) pair. A layer WAITED when some expert of the call landed after the fused
// op started and a thread spun for it (the thread records the landing stamp, written by the I/O thread, in ggml_moe_wait_ready). The tail = the end of the down op (the last thread to
// finish it) minus the last landing of any gate / up / down slice of the call. All counters sum over layers; the engine prints them on their own line.
struct ggml_m11_acc {
    _Atomic int64_t ops, experts_cpu, ids_zero;
    _Atomic int64_t n_wait, n_nowait;
    _Atomic int64_t tail_n, tail_sum, tail_max;           // layers where some thread spun for a slice that landed after the fused op started: end of the down op - that landing
    _Atomic int64_t tail_f_n, tail_f_sum, tail_d_n, tail_d_sum;   // ... split by where the last waited landing fell: inside the fused op / after it (the down op's own waits)
    _Atomic int64_t f_wait, d_wait, f_nowait, d_nowait;   // op durations (us): waiting / no-wait layers
    _Atomic int64_t idle_f, idle_d;                       // threads idle at the end of the fused / down op: sum over threads of (the last thread's end - own end)
    _Atomic int64_t gap;                                  // end of the fused op -> start of the down op (barrier + setup)
};
static struct ggml_m11_acc g_m11;
static struct {
    int64_t f_t0, d_t0;
    _Atomic int64_t last_wait_land;                       // the latest ready stamp of a slice a thread actually spun for in this layer (0 = nobody waited)
    _Atomic int64_t f_end_max, f_end_sum, d_end_max, d_end_sum;
    int fused_seen, pending;
} g_m11_cur;

static int ggml_m11_flag(void) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("GGML_MOE_CPU_TIMERS");
        on = e && atoi(e) ? 1 : 0;
    }
    return on;
}

static void ggml_m11_max(_Atomic int64_t * p, int64_t v) {
    int64_t cur = atomic_load_explicit(p, memory_order_relaxed);
    while (v > cur && !atomic_compare_exchange_weak_explicit(p, &cur, v, memory_order_relaxed, memory_order_relaxed)) {
    }
}

// thread 0, at the next fused op's start (a barrier separates it from the previous down op): book the previous layer
static void ggml_m11_settle(int nth) {
    if (!g_m11_cur.pending) {
        return;
    }
    g_m11_cur.pending = 0;
    const int64_t f_end = atomic_load_explicit(&g_m11_cur.f_end_max, memory_order_relaxed);
    const int64_t d_end = atomic_load_explicit(&g_m11_cur.d_end_max, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_m11.idle_f, nth*f_end - atomic_load_explicit(&g_m11_cur.f_end_sum, memory_order_relaxed), memory_order_relaxed);
    atomic_fetch_add_explicit(&g_m11.idle_d, nth*d_end - atomic_load_explicit(&g_m11_cur.d_end_sum, memory_order_relaxed), memory_order_relaxed);
    atomic_fetch_add_explicit(&g_m11.gap, g_m11_cur.d_t0 - f_end, memory_order_relaxed);
    const int64_t lw = atomic_load_explicit(&g_m11_cur.last_wait_land, memory_order_relaxed);
    if (lw > g_m11_cur.f_t0) {   // some thread spun for a slice that landed after this layer's fused op started
        atomic_fetch_add_explicit(&g_m11.n_wait, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_m11.f_wait, f_end - g_m11_cur.f_t0, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_m11.d_wait, d_end - g_m11_cur.d_t0, memory_order_relaxed);
        const int64_t tail = d_end - lw;
        if (tail >= 0) {
            atomic_fetch_add_explicit(&g_m11.tail_n, 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_m11.tail_sum, tail, memory_order_relaxed);
            ggml_m11_max(&g_m11.tail_max, tail);
            if (lw < f_end) {
                atomic_fetch_add_explicit(&g_m11.tail_f_n, 1, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_m11.tail_f_sum, tail, memory_order_relaxed);
            } else {
                atomic_fetch_add_explicit(&g_m11.tail_d_n, 1, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_m11.tail_d_sum, tail, memory_order_relaxed);
            }
        }
    } else {
        atomic_fetch_add_explicit(&g_m11.n_nowait, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_m11.f_nowait, f_end - g_m11_cur.f_t0, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_m11.d_nowait, d_end - g_m11_cur.d_t0, memory_order_relaxed);
    }
}

int ggml_cpu_m11_report(char * buf, size_t n) {
    if (!ggml_m11_flag() || n == 0) {
        return 0;
    }
    #define M(x) ((double) atomic_load_explicit(&g_m11.x, memory_order_relaxed))
    const double ops = M(ops) > 0 ? M(ops) : 1, nw = M(n_wait) > 0 ? M(n_wait) : 1, nn = M(n_nowait) > 0 ? M(n_nowait) : 1, tn = M(tail_n) > 0 ? M(tail_n) : 1;
    const double tf = M(tail_f_n) > 0 ? M(tail_f_n) : 1, td = M(tail_d_n) > 0 ? M(tail_d_n) : 1;
    const double lay = M(n_wait) + M(n_nowait) > 0 ? M(n_wait) + M(n_nowait) : 1;
    snprintf(buf, n, "cpu: fused ops %.0f, experts on the CPU %.0f (%.2f per op), zero-slice ids %.0f (%.2f per op), layers where a thread spun for a slice that landed after the fused op started %.0f of %.0f (%.1f %%); "
            "tail (down op end - the last landing a thread spun for) mean %.1f us max %.0f us over %.0f, of them %.0f landed inside the fused op (tail mean %.1f us) and %.0f after it (tail mean %.1f us); "
            "fused op mean %.1f us (no wait, %.0f) / %.1f us (wait, %.0f); down op mean %.1f us (no wait) / %.1f us (wait); "
            "thread idle at the end of the fused op %.1f us, of the down op %.1f us (per layer, summed over threads); gap fused end -> down start %.1f us",
            M(ops), M(experts_cpu), M(experts_cpu)/ops, M(ids_zero), M(ids_zero)/ops, M(n_wait), M(n_wait) + M(n_nowait), 100.0*M(n_wait)/lay,
            M(tail_sum)/tn, M(tail_max), M(tail_n), M(tail_f_n), M(tail_f_sum)/tf, M(tail_d_n), M(tail_d_sum)/td,
            M(f_nowait)/nn, M(n_nowait), M(f_wait)/nw, M(n_wait), M(d_nowait)/nn, M(d_wait)/nw,
            M(idle_f)/lay, M(idle_d)/lay, M(gap)/lay);
    #undef M
    return 1;
}

// The one-token expert steps' work split (ggml_cpu_set_moe_pieces / _conv1, see ggml-cpu.h)
static _Atomic int g_ggml_cpu_moe_pieces = 1;
static _Atomic int g_ggml_cpu_moe_conv1  = 0;

void ggml_cpu_set_moe_pieces(int n) {
    atomic_store_explicit(&g_ggml_cpu_moe_pieces, n > 1 ? n : 1, memory_order_relaxed);
}

void ggml_cpu_set_moe_conv1(int on) {
    atomic_store_explicit(&g_ggml_cpu_moe_conv1, on ? 1 : 0, memory_order_relaxed);
}

// The CPU track (ggml_cpu_set_track, see ggml-cpu.h); start value from GGML_CPU_TRACK (ggml_cpu_init)
static _Atomic int g_ggml_cpu_track = 0;

void ggml_cpu_set_track(int bits) {
    atomic_store_explicit(&g_ggml_cpu_track, bits & (GGML_CPU_TRACK_ROWS | GGML_CPU_TRACK_PICK | GGML_CPU_TRACK_CAT),
            memory_order_relaxed);
}

int ggml_cpu_track_bits(const struct ggml_compute_params * params) {
    return params->threadpool ? params->threadpool->cpu_track : 0;
}

static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata,
    const int x4,          // 0 or the 4-row kernel's width (the graph's snapshot)
    const int x4_pf) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;
#ifdef GGML_CPU_HAVE_MXFP4_X4
    const ggml_vec_dot_x4_t x4fn = x4 ? ggml_cpu_vec_dot_x4(type) : NULL;   // the type's 4-row kernel, if it has one
#endif

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                int64_t ir0 = iir0;
#ifdef GGML_CPU_HAVE_MXFP4_X4
                // 4 rows per call for plain MXFP4; and IQ4_XS (same bits as the 1-row kernels, see ggml_cpu_vec_dot_x4)
                if (x4fn) {
                    for (; ir0 + 3 < iir0 + blck_0 && ir0 + 3 < ir0_end; ir0 += 4) {
                        x4fn(ne00, &tmp[ir0 - iir0], src0_cur + ir0*nb01, nb01, src1_col, x4, x4_pf);
                    }
                }
#else
                GGML_UNUSED(x4);
                GGML_UNUSED(x4_pf);
#endif
                for (; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

// The readiness table of a streamed MoE pool, registered by the engine (src/llama-moe-stream.cpp)
// so that mul_mat_id can compute the experts whose bytes have landed while the ones still being read
// are waited for. The table is per pool tensor: `ready` has one flag per expert slice (the pool's slots
// followed by its all-zero slices, which are ready from the start), and `err` is the layer's read-error
// flag: a spinner must never wait forever, so it checks it.
//
// Why a registry and not an op parameter: ggml_mul_mat_id allocates no op_params, so there is nowhere to
// pass a pointer through the public API, and src0->data is stable for the life of the pool. This is the
// shape upstream is moving toward (ggml_backend_cuda_register_host_buffer, PR #26659).
// The registry is growable, not a fixed array. It was 64 entries and it SILENTLY dropped the
// overflow - tiny streams 23 layers x 3 projections (gate, up, down: each has its own pool) = 69
// registrations, so 5 pools were computed with no wait at all and the run produced different text. A
// dropped registration is not a slowdown, it is a wrong answer, so the store must never fail quietly.
#define GGML_MOE_MAX_POOLS_HARD 4096   // a leak of registrations trips this, loudly, at load time

struct ggml_moe_pool_slots {
    const void * pool;
    size_t       bytes;        // the whole allocation: [pool, pool+bytes) is this pool
    size_t       slice_bytes;  // bytes of one slice, so a view's offset divides into a slot index
    int          n_as;
    const uint32_t * ready;   // written with a release store, read with an acquire load; NULL = the engine waits
                              // itself and the pool is registered only for the zero skip
    // CPU zero skip (--moe-stream-cpu-skip-zero): slices [first_zero, n_as) are the pool's all-zero reserve; mul_mat_id writes
    // 0 for the pairs that name one instead of computing them. -1 = off (every slice is computed, as before).
    int          first_zero;
    const int32_t * err;      // -1 while fine
    void       (* on_error)(void * ctx, int32_t layer);
    void * ctx;
    // The longest time one thread of the last call(s) waited in
    // ggml_moe_wait_ready, in us - with --moe-stream-overlap that wait is the SSD stall, and the engine's
    // stall counter never saw it. The engine takes it (and resets it) at the layer's next remap.
    _Atomic int64_t wait_max_us;
    // The hand-off - from the I/O thread's ready stamp (ready_us[a], written before the release store of
    // ready[a]) to the moment a spinning thread saw the flag. Sum, count and max since the engine's last take.
    const int64_t * ready_us;
    _Atomic int64_t ho_sum_us;
    _Atomic int64_t ho_n;
    _Atomic int64_t ho_max_us;
    // Per-projection ready flags (--moe-stream-proj-ready 1): for a gate / up pool, the ready flags of the same expert's down pool (NULL otherwise),
    // and the projections whose compute started while their expert's down had not landed (counted once per expert and op)
    const uint32_t * later;
    _Atomic int64_t  n_early;
    // The engine's copy-back flags of the pool's slots (bit0 = a copy-back from the VRAM pool in flight; NULL =
    // none): a wait that starts on such a slice is no SSD wait - it is booked in wait_cb_max_us (the longest per-thread sum, as the
    // stall), and wait_all_max_us keeps the old booking (both together) so the split can be checked against it
    const uint8_t *  cb;
    int              cb_n;
    _Atomic int64_t  wait_cb_max_us;
    _Atomic int64_t  wait_all_max_us;
};

static struct ggml_moe_pool_slots * g_ggml_moe_pools = NULL;
static int g_ggml_moe_n_pools = 0;
static int g_ggml_moe_cap     = 0;
static int g_ggml_moe_tries   = 0;   // attempted registrations, to compare against what was stored

// growable storage. Registration happens while the model loads, before any op runs, so the pointers the
// op looks up below cannot be invalidated by a later regrow.
static struct ggml_moe_pool_slots * ggml_moe_pool_new(void) {
    if (g_ggml_moe_n_pools == g_ggml_moe_cap) {
        const int cap = g_ggml_moe_cap ? g_ggml_moe_cap*2 : 128;
        void * p = realloc(g_ggml_moe_pools, (size_t) cap*sizeof(*g_ggml_moe_pools));
        if (!p) {
            fprintf(stderr, "cannot store the pool registration (%d pools): out of memory\n",
                    g_ggml_moe_n_pools);
            abort();
        }
        g_ggml_moe_pools = p;
        g_ggml_moe_cap   = cap;
    }
    if (g_ggml_moe_n_pools >= GGML_MOE_MAX_POOLS_HARD) {
        fprintf(stderr, "%d pool registrations, which is more than any model has: slots are being\n"
                "      registered without being unregistered\n", g_ggml_moe_n_pools);
        abort();
    }
    return &g_ggml_moe_pools[g_ggml_moe_n_pools++];
}

void ggml_cpu_moe_slots_register(const void * pool, size_t bytes, size_t slice_bytes, int n_as, const uint32_t * ready,
                                 const int32_t * err, void (* on_error)(void *, int32_t), void * ctx, const int64_t * ready_us) {
    for (int i = 0; i < g_ggml_moe_n_pools; i++) {
        if (g_ggml_moe_pools[i].pool == pool) {
            return;   // already registered: a pool is registered once per process
        }
    }

    g_ggml_moe_tries++;

    struct ggml_moe_pool_slots * e = ggml_moe_pool_new();

    e->pool        = pool;
    e->bytes       = bytes;
    e->slice_bytes = slice_bytes;
    e->n_as        = n_as;
    e->ready    = ready;
    e->first_zero = -1;
    e->err      = err;
    e->on_error = on_error;
    e->ctx      = ctx;
    atomic_store_explicit(&e->wait_max_us, 0, memory_order_relaxed);
    e->ready_us = ready_us;
    e->later    = NULL;   // set by ggml_cpu_moe_slots_set_later
    e->cb       = NULL;   // set by ggml_cpu_moe_slots_set_cb
    e->cb_n     = 0;
    atomic_store_explicit(&e->wait_cb_max_us,  0, memory_order_relaxed);
    atomic_store_explicit(&e->wait_all_max_us, 0, memory_order_relaxed);
    atomic_store_explicit(&e->n_early, 0, memory_order_relaxed);
    atomic_store_explicit(&e->ho_sum_us, 0, memory_order_relaxed);
    atomic_store_explicit(&e->ho_n,      0, memory_order_relaxed);
    atomic_store_explicit(&e->ho_max_us, 0, memory_order_relaxed);
}

// Turn the zero skip on for a registered pool: slices from first_zero on are all zeros (the waves' reserve), so
// their pairs get 0 written instead of a matmul. -1 turns it off. An unregistered pool is an engine bug: abort.
void ggml_cpu_moe_slots_skip_zero(const void * pool, int first_zero) {
    for (int i = 0; i < g_ggml_moe_n_pools; i++) {
        if (g_ggml_moe_pools[i].pool == pool) {
            if (first_zero > g_ggml_moe_pools[i].n_as) {
                fprintf(stderr, "zero skip from slice %d but the pool has %d slices\n", first_zero, g_ggml_moe_pools[i].n_as);
                abort();
            }
            g_ggml_moe_pools[i].first_zero = first_zero;
            return;
        }
    }
    fprintf(stderr, "zero skip asked for a pool that is not registered (%p)\n", pool);
    abort();
}

// A gate / up pool's link to its down pool's flags (see later)
void ggml_cpu_moe_slots_set_later(const void * pool, const uint32_t * later) {
    for (int i = 0; i < g_ggml_moe_n_pools; i++) {
        if (g_ggml_moe_pools[i].pool == pool) {
            g_ggml_moe_pools[i].later = later;
            return;
        }
    }
    fprintf(stderr, "set_later asked for a pool that is not registered (%p)\n", pool);
    abort();
}

// The pool's slots' copy-back flags (one byte per slot, n of them; the zero slices after them have none)
void ggml_cpu_moe_slots_set_cb(const void * pool, const uint8_t * cb, int n) {
    for (int i = 0; i < g_ggml_moe_n_pools; i++) {
        if (g_ggml_moe_pools[i].pool == pool) {
            g_ggml_moe_pools[i].cb   = cb;
            g_ggml_moe_pools[i].cb_n = n;
            return;
        }
    }
    fprintf(stderr, "set_cb asked for a pool that is not registered (%p)\n", pool);
    abort();
}

// The copy-back waits and the old all-in-one stall since the last take, and reset them
void ggml_cpu_moe_slots_take_wait_cb(const void * pool, int64_t * cb_us, int64_t * all_us) {
    *cb_us = *all_us = 0;
    for (int i = 0; i < g_ggml_moe_n_pools; i++) {
        if (g_ggml_moe_pools[i].pool == pool) {
            *cb_us  = atomic_exchange_explicit(&g_ggml_moe_pools[i].wait_cb_max_us,  0, memory_order_relaxed);
            *all_us = atomic_exchange_explicit(&g_ggml_moe_pools[i].wait_all_max_us, 0, memory_order_relaxed);
            return;
        }
    }
}

// Book one op's per-thread waits - the SSD part into wait_max_us (the stall, as before), the copy-back part into
// wait_cb_max_us, and both together into wait_all_max_us (the old booking); each the longest of any of the op's threads
static void ggml_moe_book_wait(const struct ggml_moe_pool_slots * e, int64_t ssd_us, int64_t cb_us) {
    struct ggml_moe_pool_slots * em = g_ggml_moe_pools + (e - g_ggml_moe_pools);   // the same entry, writable
    _Atomic int64_t * ms[3] = { &em->wait_max_us, &em->wait_cb_max_us, &em->wait_all_max_us };
    const int64_t     vs[3] = { ssd_us, cb_us, ssd_us + cb_us };
    for (int j = 0; j < 3; j++) {
        if (vs[j] > 0) {
            int64_t cur = atomic_load_explicit(ms[j], memory_order_relaxed);
            while (vs[j] > cur && !atomic_compare_exchange_weak_explicit(ms[j], &cur, vs[j], memory_order_relaxed, memory_order_relaxed)) {
            }
        }
    }
}

int64_t ggml_cpu_moe_slots_take_early(const void * pool) {
    for (int i = 0; i < g_ggml_moe_n_pools; i++) {
        if (g_ggml_moe_pools[i].pool == pool) {
            return atomic_exchange_explicit(&g_ggml_moe_pools[i].n_early, 0, memory_order_relaxed);
        }
    }
    return 0;
}

// The compute of slice a of pool e starts now - count it when the same expert's down has not landed (counters only)
static inline void ggml_moe_note_early(const struct ggml_moe_pool_slots * e, int a) {
    if (e && e->later && a < e->n_as &&
            atomic_load_explicit((const _Atomic uint32_t *) &e->later[a], memory_order_relaxed) == 0) {
        struct ggml_moe_pool_slots * em = g_ggml_moe_pools + (e - g_ggml_moe_pools);   // the same entry, writable
        atomic_fetch_add_explicit(&em->n_early, 1, memory_order_relaxed);
    }
}

// The hand-off figures since the last take, and reset them (see ho_sum_us)
void ggml_cpu_moe_slots_take_handoff(const void * pool, int64_t * sum_us, int64_t * n, int64_t * max_us) {
    *sum_us = *n = *max_us = 0;
    for (int i = 0; i < g_ggml_moe_n_pools; i++) {
        if (g_ggml_moe_pools[i].pool == pool) {
            *sum_us = atomic_exchange_explicit(&g_ggml_moe_pools[i].ho_sum_us, 0, memory_order_relaxed);
            *n      = atomic_exchange_explicit(&g_ggml_moe_pools[i].ho_n,      0, memory_order_relaxed);
            *max_us = atomic_exchange_explicit(&g_ggml_moe_pools[i].ho_max_us, 0, memory_order_relaxed);
            return;
        }
    }
}

// The stall of the calls since the last take, and reset it (see wait_max_us)
int64_t ggml_cpu_moe_slots_take_wait(const void * pool) {
    for (int i = 0; i < g_ggml_moe_n_pools; i++) {
        if (g_ggml_moe_pools[i].pool == pool) {
            return atomic_exchange_explicit(&g_ggml_moe_pools[i].wait_max_us, 0, memory_order_relaxed);
        }
    }
    return 0;
}

void ggml_cpu_moe_slots_unregister(const void * pool) {
    for (int i = 0; i < g_ggml_moe_n_pools; i++) {
        if (g_ggml_moe_pools[i].pool == pool) {
            g_ggml_moe_pools[i] = g_ggml_moe_pools[g_ggml_moe_n_pools - 1];
            g_ggml_moe_n_pools--;
            return;
        }
    }
}

// Wait until slice `a` of this pool is published ready. The load is an acquire load, so the pool
// bytes the I/O thread wrote before its release store are visible to this thread; without it the
// compiler is free to hoist the reads of the expert above the flag and the op would compute the
// *previous* expert of the slot - the worst failure in this project, and a silent one.
// How many not-ready slices one call may skip before the two-pass path gives up and waits inline.
// A pool holds n_slots + n_expert_used slices (148 on Flash at 140 slots) but a call only ever touches
// the ones its id tensor names, so this is a backstop, not the expected size.
#define MOE_PASS_MAX_SKIPPED 256

// Overlap diagnostic (see the use in ggml_compute_forward_mul_mat_id). Its on-switch g_ggml_moe_diag_on was
// never set (nothing reads MOE_PASS_DIAG) and was read only into an unused variable: removed
static bool        g_ggml_moe_diag_printed  = false;
static bool        g_ggml_moe_diag_printed2 = false;
static bool        g_ggml_moe_diag_printed3 = false;
static uint64_t    g_ggml_moe_diag[3][GGML_MAX_N_THREADS] = {{0}};
static pthread_mutex_t g_ggml_moe_diag_mu = PTHREAD_MUTEX_INITIALIZER;

// returns how long this thread waited, in us (0 when the slice was ready). A wait that started on a copy-back from
// the VRAM pool in flight (the engine's flag, read at the first spin) is added to *cb_us instead and 0 is returned (no SSD wait)
static _Thread_local int64_t t_ho_slot_wait_us = 0;   // hand-off profiler: this thread's waits for slots, summed
static int64_t ggml_moe_wait_ready(const struct ggml_moe_pool_slots * e, int a, int ith, int64_t * cb_us) {
    int spins = 0;
    int64_t t0 = 0;
    bool cbw = false;

    while (atomic_load_explicit((const _Atomic uint32_t *) &e->ready[a], memory_order_acquire) == 0) {
        if (e->err && atomic_load_explicit((const _Atomic int32_t *) e->err, memory_order_relaxed) != -1) {
            if (e->on_error && ith == 0) {
                e->on_error(e->ctx, atomic_load_explicit((const _Atomic int32_t *) e->err, memory_order_relaxed));
            }
            // every thread leaves, so the process can report the failure once instead of hanging
            return 0;
        }
        if (spins == 0) {
            t0 = ggml_time_us();
            cbw = e->cb && a < e->cb_n && (__atomic_load_n(&e->cb[a], __ATOMIC_RELAXED) & 1) != 0;
        }

        // _mm_pause() on x86, yield on aarch64. The window is ~1 ms, so polling with a pause is right:
        // a long sleep would add its own latency to the tail of every layer that waits.
        ggml_thread_cpu_relax();
        spins++;
    }
    if (spins == 0) {
        return 0;
    }
    const int64_t t1 = ggml_time_us();
    t_ho_slot_wait_us += t1 - t0;   // only after a wait (the clock is read for it anyway)
    if (e->ready_us && ggml_m11_flag()) {   // the landing this thread spun for
        ggml_m11_max(&g_m11_cur.last_wait_land, e->ready_us[a]);
    }
    // How late this thread saw the flag after the I/O thread stamped it (counters only)
    if (e->ready_us) {
        const int64_t ho = t1 - e->ready_us[a];
        if (ho >= 0 && ho < 1000000) {
            struct ggml_moe_pool_slots * em = g_ggml_moe_pools + (e - g_ggml_moe_pools);   // the same entry, writable
            atomic_fetch_add_explicit(&em->ho_sum_us, ho, memory_order_relaxed);
            atomic_fetch_add_explicit(&em->ho_n, 1, memory_order_relaxed);
            int64_t cur = atomic_load_explicit(&em->ho_max_us, memory_order_relaxed);
            while (ho > cur && !atomic_compare_exchange_weak_explicit(&em->ho_max_us, &cur, ho, memory_order_relaxed, memory_order_relaxed)) {
            }
        }
    }
    if (ggml_trace_on()) {   // GGML_TRACE - this thread's wait for the slot, and when the read had landed
        const int64_t arg = (int64_t) (e - g_ggml_moe_pools)*65536 + a;   // the pool entry (3 per streamed layer: gate, up, down) and the slot
        ggml_trace_event(GGML_TRACE_LANE(3, ith), cbw ? "slot wait (copy-back)" : "slot wait", t0, t1, arg);
        if (e->ready_us && !cbw) {
            ggml_trace_event(GGML_TRACE_LANE(7, 0), "read landed", e->ready_us[a], e->ready_us[a], arg);
        }
    }
    if (cbw) {
        *cb_us += t1 - t0;
        return 0;
    }
    return t1 - t0;
}

static bool ggml_cpu_mmid_one_token_ok(const struct ggml_tensor * dst, int nth);
static void ggml_compute_forward_mul_mat_id_one_token(const struct ggml_compute_params * params, struct ggml_tensor * dst);

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    if (params->threadpool->moe_dispatch && ggml_cpu_mmid_one_token_ok(dst, params->nth)) {
        ggml_compute_forward_mul_mat_id_one_token(params, dst);   // ik's expert split (same bits)
        return;
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    char * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    // ik's kernels read the activations in their own layout (Q8_K with one more field per block, or Q8_2_X4): a second copy of the rows, once per call
    const bool   iqk_node = !params->use_ref && ggml_cpu_iqk_node(dst);
    const int    iqk_kind = iqk_node ? iqk_port_b_kind(type) : 0;
    const size_t iqk_row  = iqk_node ? iqk_port_b_row_size(type, ne10) : 0;
    char *       iqk_rows = iqk_node ? incr_ptr_aligned(&wdata_cur, iqk_row*ne11*ne12, 64) : NULL;

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    if (iqk_node && vec_dot_type == GGML_TYPE_Q8_K) {   // ik's quantizer, for our layout and (iq4_xs, iq3_s) for its own
                        iqk_port_q8k_quant_range((const float *) ((const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11), wdata + i13*nbw3 + i12*nbw2 + i11*nbw1,
                                                 iqk_kind == 1 ? iqk_rows + (i11 + i12*ne11)*iqk_row : NULL, ne10_block_start, ne10_block_end);
                    } else {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                    }
                }
            }
        }
#endif
    }

    if (iqk_kind == 2) {   // this thread's activation rows, quantized for ik's kernels
        for (int64_t r = ith; r < ne11*ne12; r += nth) {
            const int64_t i12 = r/ne11, i11 = r - i12*ne11;
            iqk_port_q8_2_x4_row((const float *) ((const char *) src1->data + i12*nb12 + i11*nb11), iqk_rows + r*iqk_row, ne10);
        }
    }

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // Is this a streamed pool, and which slices are ready?
    // Find the pool this tensor's data lives in, and where inside it. Equality is not enough: the
    // engine registers the allocation (llama-moe-stream.cpp register_pool(mem, ...)) while the op's
    // src0 can be a view that points into it, and an equality lookup that misses silently disables every
    // wait in the project - remap() returns early, ggml never waits, and mul_mat_id computes slices whose
    // bytes have not landed (that is the identity break, and it produced different text, not a crash).
    const struct ggml_moe_pool_slots * moe_slots = NULL;
    int    moe_slot0 = 0;

    for (int i = 0; i < g_ggml_moe_n_pools; i++) {
        const struct ggml_moe_pool_slots * e = &g_ggml_moe_pools[i];
        if ((const char *) src0->data >= (const char *) e->pool &&
            (const char *) src0->data <  (const char *) e->pool + e->bytes) {
            moe_slots  = e;
            moe_slot0  = (int) (((const char *) src0->data - (const char *) e->pool) / (int64_t) e->slice_bytes);
            break;
        }
    }

    // A pool registered only for the zero skip has no ready flags: the engine waited for its reads in remap()
    const bool moe_wait = moe_slots && moe_slots->ready;
    // Experts cur_a >= moe_zero_a of this op are all-zero slices (n_as when the skip is off or not a pool)
    const int  moe_zero_a = moe_slots && moe_slots->first_zero >= 0 ? MAX(0, moe_slots->first_zero - moe_slot0) : n_as;

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    // A zero slice gives exactly 0 in every output row, so its pairs get 0 written (dst is reused memory) and
    // the expert loop below skips them. Every thread sees the grouped rows after the barrier, so the writes are
    // split over the threads; the expert loop never writes these rows, so nothing races.
    for (int cur_a = moe_zero_a; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];
        for (int64_t r = ith; r < cne1; r += nth) {
            const struct mmid_row_mapping m = MMID_MATRIX_ROW(cur_a, r);
            memset((char *) dst->data + m.i1*nb1 + m.i2*nb2, 0, ne0*sizeof(float));
        }
    }

    // Two passes over the same loop. Pass 0 computes every slice whose bytes have landed and
    // SKIPS the rest without waiting; pass 1 walks the ones pass 0 skipped, waiting for each in turn.
    // That is the ready-first order, expressed without a permutation array:
    // the cur_a loop is already walked in the same order by every thread, and a slice never becomes
    // un-ready inside one call (remap() holds every slot it needs with keep[]), so "was ready in pass 0"
    // is a stable property of the slice. The difference is the whole point: without it, the first
    // not-ready slice would block every thread before the ready ones behind it were computed.
    //
    // "which slices pass 0 skipped" has to be remembered per thread, not re-read in pass 1. The first
    // version re-read the flag and changed tiny's text: a slice that was not ready for EVERY thread's
    // pass 0 and became ready before pass 1 was then skipped by every thread's pass 1 as well, so no
    // thread computed it and its output column kept whatever was in dst. The threads' views differ
    // because they read the flag at different times, so only the thread's own record is authoritative.
    int   moe_skipped[MOE_PASS_MAX_SKIPPED];
    int   n_moe_skipped = 0;
    int64_t moe_waited_us = 0;   // this thread's wait for slices that had not landed
    int64_t moe_waited_cb = 0;   // of it, the waits on a copy-back from the VRAM pool (not SSD)
    bool  moe_no_overlap = false;   // set if the list cannot hold them: fall back to waiting inline
    if (ith == 0 && moe_wait && !g_ggml_moe_diag_printed) {
        g_ggml_moe_diag_printed = true;
        fprintf(stderr, "overlap diag: %d pools stored of %d registrations attempted (%d layers x 3"
                " projections fit, so anything above that was being dropped):", g_ggml_moe_n_pools, g_ggml_moe_tries,
                GGML_MOE_MAX_POOLS_HARD/3);
        for (int i = 0; i < g_ggml_moe_n_pools && i < 4; i++) {
            fprintf(stderr, " [%p +%zu B, slice %zu B, n_as %d]",
                    (const void *) g_ggml_moe_pools[i].pool, g_ggml_moe_pools[i].bytes,
                    g_ggml_moe_pools[i].slice_bytes, g_ggml_moe_pools[i].n_as);
        }
        fprintf(stderr, "\n");
    }
    if (ith == 0 && moe_wait && !g_ggml_moe_diag_printed2) {
        g_ggml_moe_diag_printed2 = true;
        fprintf(stderr, "overlap diag: first op: src0->data=%p ne02=%" PRId64 " ne01=%" PRId64
                " nb02=%zu matched=%d slot0=%d\n",
                (const void *) src0->data, src0->ne[2], src0->ne[1], src0->nb[2],
                moe_slots != NULL, moe_slot0);
    }

    static int moe_pass_mode = -1;   // 1 = wait inline (no reordering), 2 = two-pass; set once
    if (moe_pass_mode < 0) {
        const char * m = getenv("MOE_PASS_MODE");
        moe_pass_mode = m ? atoi(m) : 2;
        if (moe_pass_mode != 1 && moe_pass_mode != 2) {
            moe_pass_mode = 2;
        }
    }

    for (int moe_pass = 0; moe_pass < (moe_wait && !moe_no_overlap && moe_pass_mode == 2 ? 2 : 1); ++moe_pass) {
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        if (cur_a >= moe_zero_a) {
            continue;   // an all-zero slice, its rows were written as 0 above (in both passes of the overlap)
        }

        if (moe_wait) {
            // the pool's trailing all-zero slices have no read behind them
            const int  a_idx   = moe_slot0 + cur_a;
            int  a_ready = a_idx < moe_slots->n_as
                ? atomic_load_explicit((const _Atomic uint32_t *) &moe_slots->ready[a_idx], memory_order_acquire)
                : 1;

            if (moe_pass == 0 && !a_ready && moe_pass_mode == 1) {
                moe_waited_us += ggml_moe_wait_ready(moe_slots, a_idx, ith, &moe_waited_cb);
                a_ready = 1;
            }

            if (moe_pass == 0 && !a_ready) {
                if (n_moe_skipped >= MOE_PASS_MAX_SKIPPED) {
                    // the record is full: wait for this slice and compute it now. (Note:
                    // this used to set moe_no_overlap and break, which ended pass 0 here and - the pass
                    // count reads the flag - skipped pass 1 too, so this slice, the ones after it and every
                    // recorded one were never computed. It needs > 256 not-ready slices on one thread.)
                    moe_waited_us += ggml_moe_wait_ready(moe_slots, a_idx, ith, &moe_waited_cb);
                    a_ready = 1;
                } else {
                    moe_skipped[n_moe_skipped++] = cur_a;
                    continue;
                }
            }

            // Overlap diagnostic (MOE_PASS_DIAG=1): the identity bug was silent because a slice computed
            // from bytes that had not landed still produces a plausible number. Count what each thread
            // did, per class, so one tiny run says whether pass 0 and pass 1 covered the same slices.
            {
                g_ggml_moe_diag[moe_pass][ith]++;
                if (moe_pass == 0 && !a_ready) {
                    g_ggml_moe_diag[2][ith]++;
                }
            }

            if (moe_pass == 1) {
                // this thread's pass 0 record decides, never the flag: it may have flipped since
                bool mine = false;
                for (int k = 0; k < n_moe_skipped; ++k) {
                    if (moe_skipped[k] == cur_a) {
                        mine = true;
                        break;
                    }
                }
                if (!mine) {
                    continue;
                }
                moe_waited_us += ggml_moe_wait_ready(moe_slots, moe_slot0 + cur_a, ith, &moe_waited_cb);
            }
        }

        if (moe_wait && ith == 0) {
            ggml_moe_note_early(moe_slots, moe_slot0 + cur_a);   // thread 0 reaches each expert once (counters only)
        }

        // ik's kernels, where they beat ours: enough rows routed to this expert (its bytes have landed: the wait above; each thread
        // does its share of the weight rows, no barrier). Otherwise - and whenever they cannot take it - ggml's own path
        if (iqk_node) {
            const int  iqk_min    = ggml_cpu_get_iqk();
            const bool legacy_min = type == GGML_TYPE_Q2_0 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0;
            const bool take       = cne1 >= (legacy_min ? MAX(8, iqk_min/4) : iqk_min) && iqk_port_can(type, cne1, ne00);
            if (take) {
                if (moe_wait && moe_slot0 + cur_a < moe_slots->n_as &&   // what the streaming check reads: the flag at the moment the kernels start
                        !atomic_load_explicit((const _Atomic uint32_t *) &moe_slots->ready[moe_slot0 + cur_a], memory_order_acquire)) {
                    atomic_fetch_add_explicit(&g_ggml_cpu_iqk_n[3], 1, memory_order_relaxed);
                }
                // the expert's weight rows in chunks, each thread starting with its own and then taking what is left: fast cores do more (no wait per expert)
                const int nchunk = MAX(nth, MIN(2*nth, (int) (ne01/32)));
                atomic_int * ck_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
                for (int ck = ith; ck < nchunk; ck = atomic_fetch_add_explicit(ck_ctr, 1, memory_order_relaxed)) {
                    iqk_port_mmid(ne01, cne1, ne00, (int) ne11, type, (const char *) src0->data + cur_a*nb02, nb01, iqk_rows, iqk_row,
                                  (float *) dst->data, nb1, nb2, &MMID_MATRIX_ROW(cur_a, 0), ck, nchunk);
                }
                if (ith == 0) {
                    atomic_fetch_add_explicit(&g_ggml_cpu_iqk_n[0], 1, memory_order_relaxed);
                    atomic_fetch_add_explicit(&g_ggml_cpu_iqk_n[2], cne1, memory_order_relaxed);
                }
                continue;
            }
            if (ith == 0) {
                atomic_fetch_add_explicit(&g_ggml_cpu_iqk_n[1], 1, memory_order_relaxed);
            }
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        if (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (const int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata,
                params->threadpool->mxfp4_x4, params->threadpool->mxfp4_x4_pf
            );

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
    }   // moe_pass

    // the op's stall is the longest wait of any of its threads (the copy-back waits apart)
    if (moe_wait && (moe_waited_us > 0 || moe_waited_cb > 0)) {
        ggml_moe_book_wait(moe_slots, moe_waited_us, moe_waited_cb);
    }

    if (ith == 0 && moe_wait && !g_ggml_moe_diag_printed3) {
        g_ggml_moe_diag_printed3 = true;
        pthread_mutex_lock(&g_ggml_moe_diag_mu);
        fprintf(stderr, "overlap diag: ne01=%" PRId64 " n_as=%d moe_no_overlap=%d\n",
                ne01, n_as, (int) moe_no_overlap);
        for (int p = 0; p < 2; p++) {
            uint64_t tot = 0;
            for (int t = 0; t < nth; t++) {
                tot += g_ggml_moe_diag[p][t];
            }
            fprintf(stderr, "overlap diag:   pass %d computed %" PRIu64 " slices over %d threads\n", p, tot, nth);
        }
        uint64_t sk = 0;
        for (int t = 0; t < nth; t++) {
            sk += g_ggml_moe_diag[2][t];
        }
        fprintf(stderr, "overlap diag:   pass 0 skipped %" PRIu64 " (sum over threads; pass0+pass1 should be n_as*threads)\n", sk);
        memset(g_ggml_moe_diag, 0, sizeof(g_ggml_moe_diag));
        pthread_mutex_unlock(&g_ggml_moe_diag_mu);
    }
}

/////////////////////////////////

// The CPU thread loop is compiled twice (timed / untimed, below). The two functions it calls per node are
// forced inline, so each copy keeps them inlined as the single loop did (one call site each)
#if defined(_MSC_VER)
#define GGML_HO_INLINE   __forceinline
#define GGML_HO_NOINLINE __declspec(noinline)
#else
#define GGML_HO_INLINE   inline __attribute__((always_inline))
#define GGML_HO_NOINLINE __attribute__((noinline))
#endif

static GGML_HO_NOINLINE void ggml_compute_forward_mul_mat_multi(struct ggml_compute_params * params, struct ggml_tensor * dst);

// Row i of a MUL_MAT_MULTI as the MUL_MAT node it stands for: as[i] times column i of b (or its one column),
// into row i of dst - the shapes and strides of a separate [k, m] x [k, 1] -> [m, 1] node, the op params copied (precision, hint)
static void ggml_mul_mat_multi_row(const struct ggml_tensor * dst, int i, struct ggml_tensor * row, struct ggml_tensor * col) {
    const int n = ggml_get_op_params_i32(dst, 8);
    const struct ggml_tensor * b = dst->src[n];
    *col = *b;
    col->ne[2] = 1;
    col->ne[3] = 1;
    col->nb[2] = col->nb[1];
    col->nb[3] = col->nb[1];
    col->data  = (char *) b->data + (b->ne[2] > 1 ? (size_t) i*b->nb[2] : 0);
    col->view_src  = NULL;
    col->view_offs = 0;
    *row = *dst;
    row->op = GGML_OP_MUL_MAT;
    memset(row->src, 0, sizeof(row->src));
    row->src[0] = dst->src[i];
    row->src[1] = col;
    row->ne[1] = 1;
    row->nb[2] = row->nb[1];
    row->nb[3] = row->nb[1];
    row->data  = (char *) dst->data + (size_t) i*dst->nb[1];
    row->view_src  = NULL;
    row->view_offs = 0;
}

static GGML_HO_INLINE void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;
        case GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } break;
        case GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } break;
        case GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } break;
        case GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } break;
        case GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } break;
        case GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } break;
        case GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } break;
        case GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } break;
        case GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } break;
        case GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } break;
        case GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } break;
        case GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } break;
        case GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } break;
        case GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                ggml_compute_forward_count_equal(params, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } break;
        case GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } break;
        case GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case GGML_OP_MUL_MAT_MULTI:
            {
                ggml_compute_forward_mul_mat_multi(params, tensor);
            } break;
        case GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } break;
        case GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } break;
        case GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } break;
        case GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } break;
        case GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } break;
        case GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } break;
        case GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } break;
        case GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } break;
        case GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } break;
        case GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } break;
        case GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } break;
        case GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } break;
        case GGML_OP_TOP_K:
            {
                ggml_compute_forward_top_k(params, tensor);
            } break;
        case GGML_OP_MOE_GROUP_TOPK:
            {
                ggml_compute_forward_moe_group_topk(params, tensor);
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } break;
        case GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } break;
        case GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } break;
        case GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } break;
        case GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } break;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } break;
        case GGML_OP_TOP_K_MASK:
            {
                ggml_compute_forward_top_k_mask(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_RESHAPE:
            {
                // nop
            } break;
        case GGML_OP_PERMUTE:
            {
                // nop
            } break;
        case GGML_OP_VIEW:
            {
                // nop
            } break;
        case GGML_OP_TRANSPOSE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// GGML_OP_MUL_MAT_MULTI - row by row, each row through the very dispatch of a MUL_MAT node (the extra buffer
// types first, then ggml_compute_forward_mul_mat), so row i is bit for bit the separate MUL_MAT's result. All threads run every
// row; a barrier between rows, because a MUL_MAT shares the work buffer and the chunk counter among the threads.
// Op params [9] = GGML_CPU_GSF_MAGIC, [10..11] = the "full" array's address, [4..7] = rows 0..3's target layer - a row whose
// target is full is not computed: thread 0 writes 0 into it (the array is constant while a graph runs, so every thread skips the same
// rows; the barrier stays between computed rows only)
static GGML_HO_NOINLINE void ggml_compute_forward_mul_mat_multi(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const int n = ggml_get_op_params_i32(dst, 8);
    const uint8_t * full = NULL;
    if (ggml_get_op_params_i32(dst, 9) == GGML_CPU_GSF_MAGIC) {
        memcpy(&full, (const int32_t *) dst->op_params + 10, sizeof(full));
    }
    bool first = true;
    for (int i = 0; i < n; i++) {
        if (full && i < 4 && full[ggml_get_op_params_i32(dst, 4 + i)]) {
            if (params->ith == 0) {
                memset((char *) dst->data + i*dst->nb[1], 0, dst->ne[0]*sizeof(float));
                atomic_fetch_add_explicit(&g_ggml_cpu_gsf_rows, 1, memory_order_relaxed);
            }
            continue;
        }
        struct ggml_tensor row, col;
        ggml_mul_mat_multi_row(dst, i, &row, &col);
        if (!first) {
            ggml_barrier(params->threadpool);
        }
        first = false;
        ggml_compute_forward(params, &row);
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static GGML_HO_INLINE void set_numa_thread_affinity(int thread_n) {   // one call site per thread-loop copy
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
        case GGML_OP_TOP_K_MASK:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(node)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_SILU_BACK:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_MUL_MAT_MULTI:   // each row is a MUL_MAT
        case GGML_OP_OUT_PROD:
        case GGML_OP_SET_ROWS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
                // The CPU track (GGML_CPU_TRACK_ROWS): an f32 GET_ROWS with fewer rows than threads is split by
                // columns over every thread (ggml_compute_forward_get_rows_f32); say so, so a graph of that op alone gets them too
                if (node->op == GGML_OP_GET_ROWS &&
                        (atomic_load_explicit(&g_ggml_cpu_track, memory_order_relaxed) & GGML_CPU_TRACK_ROWS) &&
                        node->src[0]->type == GGML_TYPE_F32 && ggml_nelements(node->src[1]) < n_threads &&
                        node->src[0]->ne[0] >= 16*n_threads) {
                    n_tasks = n_threads;
                }
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } break;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_COL2IM_1D:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ROLL:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_MOE_GROUP_TOPK:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                const int64_t n_heads = node->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } break;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM2:
            {
                struct ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM3:
            {
                struct ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CUSTOM:
            {
                struct ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  break;
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio != GGML_SCHED_PRIO_LOW) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (int) GetLastError());
            return false;
        }
        #endif
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__linux__)
// TODO: this may not work on BSD, to be verified

static bool ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_BATCH; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

void ggml_threadpool_free(struct ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    struct ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    free(threadpool->ho_wt);
    free(threadpool->ho_ts);      // NULL unless the hand-off timers ran on this pool
    free(threadpool->ho_fused);
    free(threadpool->ho_nodes);

    const size_t workers_size = sizeof(struct ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void ggml_threadpool_resume_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

void ggml_threadpool_pause(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void ggml_threadpool_resume(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#if defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    size_t work_size = 0;

    struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            switch (node->op) {
                case GGML_OP_CPY:
                case GGML_OP_DUP:
                    {
                        if (ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (node->src[0]->type == GGML_TYPE_F32 && node->src[1] && node->src[1]->type == GGML_TYPE_I32) ||
                            (node->src[0]->type == GGML_TYPE_I32 && node->src[1] && node->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ADD:
                case GGML_OP_ADD_ID:
                case GGML_OP_ADD1:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ACC:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_COUNT_EQUAL:
                    {
                        cur = ggml_type_size(node->type)*n_tasks;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        const enum ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;
                        if (node->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, node);
                    } break;
                case GGML_OP_MUL_MAT_MULTI:   // one row's MUL_MAT at a time (all rows have the same shapes)
                    {
                        struct ggml_tensor row, col;
                        ggml_mul_mat_multi_row(node, 0, &row, &col);
                        const enum ggml_type vec_dot_type = type_traits_cpu[row.src[0]->type].vec_dot_type;
                        if (col.type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(&col));
                        }
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, &row);
                    } break;
                case GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[1];
                        const struct ggml_tensor * ids = node->src[2];
                        const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        const int n_as = src0->ne[2];
                        // src1
                        if (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, node);
                        if (ggml_cpu_iqk_node(node)) {   // the activations in ik's layout
                            cur = GGML_PAD(cur, 64);
                            cur += iqk_port_b_row_size(src0->type, src1->ne[0])*src1->ne[1]*src1->ne[2] + 64;
                        }
                    } break;
                case GGML_OP_OUT_PROD:
                    {
                        if (ggml_is_quantized(node->src[0]->type) ||
                            node->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SET_ROWS:
                    {
                        if (node->src[0]->type == GGML_TYPE_F16 && node->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SOFT_MAX:
                case GGML_OP_ROPE:
                case GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(node->src[0]->ne[3] == 1);
                        GGML_ASSERT(node->src[1]->ne[2] == 1);
                        GGML_ASSERT(node->src[1]->ne[3] == 1);

                        const int64_t ne00 = node->src[0]->ne[0];  // K
                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
                        const int64_t ne10 = node->src[1]->ne[0];  // L
                        const int64_t ne11 = node->src[1]->ne[1];  // Cin

                        if ((node->src[0]->type == GGML_TYPE_F16 ||
                             node->src[0]->type == GGML_TYPE_BF16) &&
                            node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += sizeof(ggml_fp16_t)*ne10*ne11;
                        } else if (node->src[0]->type == GGML_TYPE_F32 &&
                                   node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(float)*ne00*ne01*ne02;
                            cur += sizeof(float)*ne10*ne11;
                        } else {
                            GGML_ABORT("fatal error");
                        }
                    } break;
                case GGML_OP_CONV_2D:
                case GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        const int64_t ne00 = node->src[0]->ne[0]; // W
                        const int64_t ne01 = node->src[0]->ne[1]; // H
                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        const int64_t ne10 = node->src[1]->ne[0]; // W
                        const int64_t ne11 = node->src[1]->ne[1]; // H
                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In
                        const int64_t ne13 = node->src[1]->ne[3]; // Batch

                        GGML_ASSERT(node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(node->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(node->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(node->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } break;
                case GGML_OP_TOP_K:
                    {
                        cur += sizeof(int32_t)*node->src[0]->ne[0]*n_tasks;
                    } break;
                case GGML_OP_MOE_GROUP_TOPK:   // per thread the scores, group scores and sort indices
                    {
                        cur += (2*node->src[0]->ne[0]*sizeof(float) + 2*64*sizeof(float) + CACHE_LINE_SIZE)*n_tasks;
                    } break;
                case GGML_OP_FLASH_ATTN_EXT:
                    {
                        const int64_t neq2 = node->src[0]->ne[2]; // number of query heads
                        const int64_t DK = node->src[1]->ne[0];
                        const int64_t DV = node->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = sizeof(float)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = sizeof(float)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));
                        if (node->src[1]->type == GGML_TYPE_Q8_0 && node->src[1]->ne[2] > 0) {   // the GQA decode chunk's scratch
                            decode += sizeof(float)*n_tasks*(CACHE_LINE_SIZE_F32*2 +
                                      ggml_cpu_fa_q8_0_gqa_scratch(MAX(1, neq2/node->src[1]->ne[2]), DK, DV));
                        }

                        cur += MAX(prefill, decode);
                    } break;
                case GGML_OP_FLASH_ATTN_BACK:
                    {
                        const int64_t    D = node->src[0]->ne[0];
                        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        if (node->src[1]->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_BF16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } break;

                case GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } break;
                case GGML_OP_GATED_DELTA_NET:
                    {
                        const int64_t S_v = node->src[2]->ne[0];
                        const int64_t K   = ggml_get_op_params_i32(node, 0);
                        const int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * sizeof(float) * n_tasks;
                    } break;
                case GGML_OP_COUNT:
                    {
                        GGML_ABORT("fatal error");
                    }
                case GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        const int64_t ne10 = node->src[1]->ne[0];
                        cur += sizeof(float)*ne10*n_tasks;
                    } break;
                default:
                    break;
            }
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}


// Fused MoE up+gate for one token (ggml_cpu_set_moe_fuse). The nodes mul_mat_id(gate), mul_mat_id(up) and
// swiglu_split(gate, up) run as one step: the activation is quantized once, one chunk queue covers all experts (ready
// ones first, like the overlap path), and each chunk computes its gate and up rows with the same kernel as mul_mat_id and then
// ggml_vec_swiglu_f32 on them. Chunks start and end on multiples of 16 rows, so the swiglu takes the same SIMD path as
// the GLU op on the whole row: the same bits. The idea is ik_llama.cpp's fused up/gate op (MIT).
static _Atomic int64_t g_ggml_cpu_moe_fuse_count = 0;   // fused steps run, so a caller can check the setting took effect

int64_t ggml_cpu_moe_fuse_count(void) {
    return atomic_load_explicit(&g_ggml_cpu_moe_fuse_count, memory_order_relaxed);
}

static const struct ggml_moe_pool_slots * ggml_moe_pool_find(const void * data, int * slot0) {
    for (int i = 0; i < g_ggml_moe_n_pools; i++) {
        const struct ggml_moe_pool_slots * e = &g_ggml_moe_pools[i];
        if ((const char *) data >= (const char *) e->pool && (const char *) data < (const char *) e->pool + e->bytes) {
            *slot0 = (int) (((const char *) data - (const char *) e->pool) / (int64_t) e->slice_bytes);
            return e;
        }
    }
    *slot0 = 0;
    return NULL;
}

// what the fused step computes: y = silu(gate) * up, or with Ling's clamped layers
// y = clamp(silu(gate), cg) * clamp(up, cu) (llama-graph.cpp: SILU, CLAMP, CLAMP, MUL when swiglu_clamp_exp > 0)
struct ggml_moe_fuse {
    struct ggml_tensor * dst;
    const struct ggml_tensor * gate;   // mul_mat_id nodes
    const struct ggml_tensor * up;
    bool  clamped;
    float cg_min, cg_max, cu_min, cu_max;
    int   n_nodes;                     // nodes covered, from node_n
};

// the two mul_mat_id must share input and ids, plain layouts, one token, rows in multiples of 16, and the step's own
// limits must hold for this thread count (else the normal ops run: the step never aborts on a graph it accepted)
static bool ggml_cpu_moe_pair_ok(const struct ggml_tensor * n0, const struct ggml_tensor * n1, int nth) {
    const struct ggml_tensor * a0 = n0->src[0];
    const struct ggml_tensor * a1 = n1->src[0];
    const struct ggml_tensor * b  = n0->src[1];
    const struct ggml_tensor * ids = n0->src[2];
    if (n1->src[1] != b || n1->src[2] != ids || a0->type != a1->type || a0->extra || a1->extra ||
        !ggml_are_same_shape(a0, a1) || a0->nb[1] != a1->nb[1] || a0->nb[2] != a1->nb[2]) {
        return false;
    }
    // b must be quantized into wdata: an F32 b (vec_dot_type F32) would be read in place, and the allocator may give its
    // memory to the step's own output
    if (type_traits_cpu[a0->type].vec_dot_type == b->type) {
        return false;
    }
    // ids: contiguous int32, at most 64 per token (the order list); chunk rows: at most the 4096 of the row scratch
    if (ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->ne[0] > 64 ||
        GGML_PAD((a0->ne[1] + nth - 1) / nth, 16) > 4096) {
        return false;
    }
    return b->type == GGML_TYPE_F32 && b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1 && ids->ne[1] == 1 &&
        ggml_is_contiguous(b) && a0->ne[1] % 16 == 0 && a0->nb[0] == ggml_type_size(a0->type) &&
        type_traits_cpu[a0->type].vec_dot != NULL && n0->type == GGML_TYPE_F32 && n1->type == GGML_TYPE_F32 &&
        n0->ne[0] == a0->ne[1] && ggml_is_contiguous_1(n0) && ggml_is_contiguous_1(n1);
}

// Forced inline, one call site per thread-loop copy, as the single loop inlined it
static GGML_HO_INLINE bool ggml_cpu_moe_up_gate_match(const struct ggml_cgraph * cgraph, int node_n, const struct ggml_compute_params * params,
                                       struct ggml_moe_fuse * f) {
    if (!params->threadpool->moe_fuse) {
        return false;
    }
    const int nth = params->nth;
    struct ggml_tensor ** nd = cgraph->nodes + node_n;
    const int left = cgraph->n_nodes - node_n;

    // pattern 1: mul_mat_id, mul_mat_id, swiglu_split (either order of the two)
    if (left >= 3 && nd[0]->op == GGML_OP_MUL_MAT_ID && nd[1]->op == GGML_OP_MUL_MAT_ID && nd[2]->op == GGML_OP_GLU &&
        ggml_get_glu_op(nd[2]) == GGML_GLU_OP_SWIGLU && nd[2]->src[1] != NULL &&
        ((nd[2]->src[0] == nd[0] && nd[2]->src[1] == nd[1]) || (nd[2]->src[0] == nd[1] && nd[2]->src[1] == nd[0])) &&
        (nd[0]->flags & nd[1]->flags & nd[2]->flags & GGML_TENSOR_FLAG_COMPUTE) &&
        ggml_node_has_n_uses(cgraph, node_n, 1) && ggml_node_has_n_uses(cgraph, node_n + 1, 1) &&
        ggml_cpu_moe_pair_ok(nd[2]->src[0], nd[2]->src[1], nth) && nd[2]->type == GGML_TYPE_F32 && ggml_is_contiguous_1(nd[2])) {
        f->dst = nd[2]; f->gate = nd[2]->src[0]; f->up = nd[2]->src[1];
        f->clamped = false; f->n_nodes = 3;
        return true;
    }

    // pattern 2 (Ling's clamped layers): gate = mul_mat_id, silu(gate), clamp, up = mul_mat_id, clamp(up), mul
    if (left >= 6 && nd[0]->op == GGML_OP_MUL_MAT_ID && nd[1]->op == GGML_OP_UNARY && ggml_get_unary_op(nd[1]) == GGML_UNARY_OP_SILU &&
        nd[1]->src[0] == nd[0] && nd[2]->op == GGML_OP_CLAMP && nd[2]->src[0] == nd[1] &&
        nd[3]->op == GGML_OP_MUL_MAT_ID && nd[4]->op == GGML_OP_CLAMP && nd[4]->src[0] == nd[3] &&
        nd[5]->op == GGML_OP_MUL && ((nd[5]->src[0] == nd[2] && nd[5]->src[1] == nd[4]) || (nd[5]->src[0] == nd[4] && nd[5]->src[1] == nd[2]))) {
        for (int i = 0; i < 6; i++) {
            if ((nd[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || nd[i]->type != GGML_TYPE_F32 || !ggml_is_contiguous_1(nd[i]) ||
                !ggml_are_same_shape(nd[i], nd[0]) || (i < 5 && !ggml_node_has_n_uses(cgraph, node_n + i, 1))) {
                return false;
            }
        }
        if (!ggml_cpu_moe_pair_ok(nd[0], nd[3], nth)) {
            return false;
        }
        f->dst = nd[5]; f->gate = nd[0]; f->up = nd[3];
        f->clamped = true; f->n_nodes = 6;
        memcpy(&f->cg_min, (const float *) nd[2]->op_params + 0, sizeof(float));
        memcpy(&f->cg_max, (const float *) nd[2]->op_params + 1, sizeof(float));
        memcpy(&f->cu_min, (const float *) nd[4]->op_params + 0, sizeof(float));
        memcpy(&f->cu_max, (const float *) nd[4]->op_params + 1, sizeof(float));
        // the zero-slice branch writes 0, which is the ops' result only if 0 survives both clamps
        if (!(f->cg_min <= 0.0f && 0.0f <= f->cg_max && f->cu_min <= 0.0f && 0.0f <= f->cu_max)) {
            return false;
        }
        return true;
    }
    return false;
}

// ik_llama.cpp's expert split for a one-token mul_mat_id (its GGML_EXPERT_CHUNKING in
// ggml_compute_forward_mul_mat_id: one atomic chunk counter over all used experts, chunks_per_expert = nth, so each thread
// streams one long contiguous row range per expert; the activation rows are quantized whole, one row per thread), in
// the shape of the fused step: ready experts first and a wait for the rest, the zero slices written as 0.
// Each row is computed by the same kernel as in mul_mat_id (vec_dot, or the 4-row kernel on 4-row groups that start
// at a 16-row boundary), so the output has the same bits. ik_llama.cpp: Copyright (c) 2024-2025 The ik_llama.cpp authors,
// MIT license (SPDX-License-Identifier: MIT).
static bool ggml_cpu_mmid_one_token_ok(const struct ggml_tensor * dst, int nth) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids  = dst->src[2];
    const enum ggml_type vdt = type_traits_cpu[src0->type].vec_dot_type;
    GGML_UNUSED(nth);
    return ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) && ids->ne[1] == 1 && ids->ne[0] <= 64 &&
        src1->type == GGML_TYPE_F32 && vdt != GGML_TYPE_F32 && src1->ne[2] == 1 && src1->ne[3] == 1 && ggml_is_contiguous(src1) &&
        src0->ne[3] == 1 && !src0->extra && src0->nb[0] == ggml_type_size(src0->type) && type_traits_cpu[src0->type].vec_dot != NULL &&
        src0->ne[1] % 16 == 0 && src0->ne[0] % ggml_blck_size(vdt) == 0 &&
        dst->type == GGML_TYPE_F32 && dst->nb[0] == sizeof(float) && dst->ne[0] == src0->ne[1] && dst->ne[1] == ids->ne[0] && dst->ne[2] == 1;
}

static void ggml_compute_forward_mul_mat_id_one_token(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids  = dst->src[2];

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;
    const enum ggml_type vec_dot_type = type_traits_cpu[type].vec_dot_type;
    ggml_vec_dot_t const vec_dot = type_traits_cpu[type].vec_dot;
    ggml_from_float_t const from_float = type_traits_cpu[vec_dot_type].from_float;

    const int64_t ne00  = src0->ne[0];
    const int64_t nr0   = src0->ne[1];
    const int     n_as  = (int) src0->ne[2];
    const size_t  nb01  = src0->nb[1];
    const size_t  nb02  = src0->nb[2];
    const int     n_ids = (int) ids->ne[0];
    const int64_t ne11  = src1->ne[1];

    const size_t row_size = ggml_row_size(vec_dot_type, ne00);

    // the activation rows: whole rows per thread when there are enough of them (ik), else each row split by blocks
    // (as mul_mat_id); blocks are quantized independently, so either split gives the same bytes
    char * wdata = params->wdata;
    {
        const int64_t bs = ggml_blck_size(vec_dot_type);
        const int64_t nb = ne00/bs;
        // --moe-stream-cpu-conv1: whole rows, one thread per row (one token's single row: thread 0 alone)
        const bool    whole = ne11 >= nth || params->threadpool->moe_conv1;
        for (int64_t i11 = 0; i11 < ne11; i11++) {
            int64_t b0 = 0, b1 = nb;
            if (whole) {
                if (i11 % nth != ith) {
                    continue;
                }
            } else {
                b0 = (ith*nb)/nth;
                b1 = ((ith + 1)*nb)/nth;
            }
            const float * x = (const float *) ((const char *) src1->data + i11*src1->nb[1]);
            if (b1 > b0) {
                from_float(x + b0*bs, wdata + i11*row_size + b0*ggml_type_size(vec_dot_type), (b1 - b0)*bs);
            }
        }
    }
    int * order = (int *) (wdata + GGML_PAD(ne11*row_size, 64));   // [n_ids]: id positions, ready experts first

    int s0 = 0;
    const struct ggml_moe_pool_slots * e = ggml_moe_pool_find(src0->data, &s0);
    const bool wait   = e && e->ready;
    const int  zero_a = e && e->first_zero >= 0 ? MAX(0, e->first_zero - s0) : n_as;

    const int32_t * id = (const int32_t *) ids->data;
    // ik: chunks_per_expert = nth; --moe-stream-cpu-pieces P: P times as many, on 16-row boundaries (P = 1: as before)
    const int     npc = nth*MAX(1, params->threadpool->moe_pieces);
    const int64_t dr  = GGML_PAD((nr0 + npc - 1) / npc, 16);
    const int     cpi = (int) ((nr0 + dr - 1) / dr);

    if (ith == 0) {
        bool rdy[64];
        for (int k = 0; k < n_ids; k++) {
            const int a = id[k];
            GGML_ASSERT(a >= 0 && a < n_as);
            rdy[k] = a >= zero_a || !wait || s0 + a >= e->n_as ||
                atomic_load_explicit((const _Atomic uint32_t *) &e->ready[s0 + a], memory_order_acquire);   // read once
        }
        int n = 0;
        for (int pass = 0; pass < 2; pass++) {
            for (int k = 0; k < n_ids; k++) {
                if (rdy[k] == (pass == 0)) {
                    order[n++] = k;
                }
            }
        }
        GGML_ASSERT(n == n_ids);
        ggml_threadpool_chunk_set(params->threadpool, nth);
        atomic_fetch_add_explicit(&g_ggml_cpu_moe_dispatch_count, 1, memory_order_relaxed);
        if (g_m11_cur.fused_seen && ggml_m11_flag() && strncmp(dst->name, "ffn_moe_down", 12) == 0) {   // the down op of the layer just started
            g_m11_cur.fused_seen = 0;
            g_m11_cur.d_t0 = ggml_time_us();
            g_m11_cur.pending  = 1;
        }
    }
    ggml_barrier(params->threadpool);

    const int x4    = params->threadpool->mxfp4_x4;
    const int x4_pf = params->threadpool->mxfp4_x4_pf;
#ifdef GGML_CPU_HAVE_MXFP4_X4
    const ggml_vec_dot_x4_t x4fn = x4 ? ggml_cpu_vec_dot_x4(type) : NULL;   // MXFP4 and IQ4_XS
#endif
    int64_t waited = 0, waited_cb = 0;   // (the copy-back waits apart)
    const int n_chunks = n_ids * cpi;

    for (int c = ith; c < n_chunks; c = ggml_threadpool_chunk_add(params->threadpool, 1)) {
        const int k  = order[c / cpi];
        const int a  = id[k];
        const int64_t r0 = (int64_t) (c % cpi) * dr;
        const int64_t r1 = MIN(r0 + dr, nr0);
        float * y = (float *) ((char *) dst->data + (size_t) k*dst->nb[1]);
        if (a >= zero_a) {
            memset(y + r0, 0, (size_t) (r1 - r0)*sizeof(float));   // an all-zero slice
            continue;
        }
        if (wait && s0 + a < e->n_as) {
            waited += ggml_moe_wait_ready(e, s0 + a, ith, &waited_cb);
            if (c % cpi == 0) {
                ggml_moe_note_early(e, s0 + a);   // the expert's first chunk (counters only)
            }
        }
        const char * p = (const char *) src0->data + (size_t) a*nb02;
        const char * q = wdata + (k % ne11)*row_size;
        int64_t r = r0;
#ifdef GGML_CPU_HAVE_MXFP4_X4
        if (x4fn) {
            for (; r + 3 < r1; r += 4) {
                x4fn((int) ne00, y + r, p + r*nb01, nb01, q, x4, x4_pf);
            }
        }
#else
        GGML_UNUSED(x4);
        GGML_UNUSED(x4_pf);
#endif
        for (; r < r1; r++) {
            vec_dot((int) ne00, y + r, 0, p + r*nb01, 0, q, 0, 1);
        }
    }

    if (g_m11_cur.pending && ggml_m11_flag()) {   // this thread's end of the down op
        const int64_t te = ggml_time_us();
        ggml_m11_max(&g_m11_cur.d_end_max, te);
        atomic_fetch_add_explicit(&g_m11_cur.d_end_sum, te, memory_order_relaxed);
    }
    if (wait && (waited > 0 || waited_cb > 0)) {   // the op's stall is the longest wait of any of its threads, as in mul_mat_id
        ggml_moe_book_wait(e, waited, waited_cb);
    }
}

static void ggml_compute_forward_moe_up_gate_fused(
        const struct ggml_compute_params * params,
        const struct ggml_moe_fuse * f) {

    struct ggml_tensor * glu = f->dst;   // the output: [n_ff, n_expert_used, 1]
    const struct ggml_tensor * gate = f->gate;
    const struct ggml_tensor * up   = f->up;
    const struct ggml_tensor * ag   = gate->src[0];
    const struct ggml_tensor * au   = up->src[0];
    const struct ggml_tensor * b    = gate->src[1];
    const struct ggml_tensor * ids  = gate->src[2];

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = ag->type;
    const enum ggml_type vec_dot_type = type_traits_cpu[type].vec_dot_type;
    ggml_vec_dot_t const vec_dot = type_traits_cpu[type].vec_dot;

    const int64_t ne00  = ag->ne[0];
    const int64_t nr0   = ag->ne[1];
    const int     n_as  = (int) ag->ne[2];
    const int     n_ids = (int) ids->ne[0];
    const size_t  nb01  = ag->nb[1];
    const size_t  nb02  = ag->nb[2];

    // the quantized activation, exactly as mul_mat_id makes it (the same block split over the threads)
    char * wdata = params->wdata;
    const void * qb = b->data;
    size_t used = 0;
    if (b->type != vec_dot_type) {
        ggml_from_float_t const from_float = type_traits_cpu[vec_dot_type].from_float;
        const size_t bs = ggml_blck_size(vec_dot_type);
        // --moe-stream-cpu-conv1: the whole row on thread 0 (upstream #29308: no cache line written by two cores)
        const bool    one = params->threadpool->moe_conv1 != 0;
        const int64_t s = one ? 0                         : (ith * ne00/bs) / nth;
        const int64_t e = one ? (ith == 0 ? ne00/bs : 0) : ((ith + 1) * ne00/bs) / nth;
        if (e > s) {
            from_float((const float *) b->data + s*bs, wdata + s*ggml_type_size(vec_dot_type), (e - s)*bs);
        }
        qb   = wdata;
        used = GGML_PAD(ggml_row_size(vec_dot_type, ne00), 64);
    }
    int * order = (int *) (wdata + used);   // [n_ids]: id positions, ready experts first

    int sg = 0, su = 0;
    const struct ggml_moe_pool_slots * eg = ggml_moe_pool_find(ag->data, &sg);
    const struct ggml_moe_pool_slots * eu = ggml_moe_pool_find(au->data, &su);
    const bool wait  = (eg && eg->ready) || (eu && eu->ready);
    const int  zero_a = eg && eu && eg->first_zero >= 0 && eu->first_zero >= 0 && eg->first_zero - sg == eu->first_zero - su
        ? MAX(0, eg->first_zero - sg) : n_as;

    const int32_t * id = (const int32_t *) ids->data;
    // chunks per id: ggml's decode split (one per thread), on 16-row boundaries; --moe-stream-cpu-pieces P: P per thread
    const int     npc    = nth*MAX(1, params->threadpool->moe_pieces);
    const int64_t dr     = GGML_PAD((nr0 + npc - 1) / npc, 16);
    const int     cpi    = (int) ((nr0 + dr - 1) / dr);

    if (ith == 0) {
        // each flag is read ONCE: reading it again in the second pass dropped a slice that landed in between from both
        // passes, left the tail of order[] unset and crashed a test run (the overlap path warns about exactly this)
        bool rdy[64];   // n_expert_used is 8 for Ling, 10 for Qwen3.8
        GGML_ASSERT(n_ids <= (int) (sizeof(rdy)/sizeof(rdy[0])));
        for (int k = 0; k < n_ids; k++) {
            const int a = id[k];
            GGML_ASSERT(a >= 0 && a < n_as);
            rdy[k] = a >= zero_a ||
                ((!eg || !eg->ready || sg + a >= eg->n_as || atomic_load_explicit((const _Atomic uint32_t *) &eg->ready[sg + a], memory_order_acquire)) &&
                 (!eu || !eu->ready || su + a >= eu->n_as || atomic_load_explicit((const _Atomic uint32_t *) &eu->ready[su + a], memory_order_acquire)));
        }
        int n = 0;
        for (int pass = 0; pass < 2; pass++) {
            for (int k = 0; k < n_ids; k++) {
                if (rdy[k] == (pass == 0)) {
                    order[n++] = k;
                }
            }
        }
        GGML_ASSERT(n == n_ids);
        ggml_threadpool_chunk_set(params->threadpool, nth);
        atomic_fetch_add_explicit(&g_ggml_cpu_moe_fuse_count, 1, memory_order_relaxed);
        if (ggml_m11_flag()) {   // a new layer starts; book the previous one
            ggml_m11_settle(nth);
            g_m11_cur.f_t0 = ggml_time_us();
            atomic_store_explicit(&g_m11_cur.f_end_max, 0, memory_order_relaxed);
            atomic_store_explicit(&g_m11_cur.f_end_sum, 0, memory_order_relaxed);
            atomic_store_explicit(&g_m11_cur.d_end_max, 0, memory_order_relaxed);
            atomic_store_explicit(&g_m11_cur.d_end_sum, 0, memory_order_relaxed);
            atomic_store_explicit(&g_m11_cur.last_wait_land, 0, memory_order_relaxed);
            int64_t nz = 0;
            for (int k = 0; k < n_ids; k++) {
                if (id[k] >= zero_a) {
                    nz++;
                }
            }
            g_m11_cur.fused_seen = 1;
            atomic_fetch_add_explicit(&g_m11.ops, 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_m11.experts_cpu, n_ids - nz, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_m11.ids_zero, nz, memory_order_relaxed);
        }
    }
    ggml_barrier(params->threadpool);

    float tg[4096];   // gate and up rows of one chunk
    float tu[4096];
    GGML_ASSERT(dr <= 4096);

    const int x4    = params->threadpool->mxfp4_x4;   // the graph's snapshot
    const int x4_pf = params->threadpool->mxfp4_x4_pf;
#ifdef GGML_CPU_HAVE_MXFP4_X4
    const ggml_vec_dot_x4_t x4fn = x4 ? ggml_cpu_vec_dot_x4(type) : NULL;   // MXFP4 and IQ4_XS
#endif
    int64_t waited_g = 0, waited_u = 0, waited_cb_g = 0, waited_cb_u = 0;   // (the copy-back waits apart)
    const int n_chunks = n_ids * cpi;

    for (int c = ith; c < n_chunks; c = ggml_threadpool_chunk_add(params->threadpool, 1)) {
        const int k  = order[c / cpi];
        const int a  = id[k];
        const int64_t r0 = (int64_t) (c % cpi) * dr;
        const int64_t r1 = MIN(r0 + dr, nr0);
        float * y = (float *) ((char *) glu->data + (size_t) k*glu->nb[1]);
        // Bounds, loudly (one segfault in 5 runs, inside the zero branch)
        if (k < 0 || k >= n_ids || r0 < 0 || r1 > nr0 || (const char *) (y + r1) > (const char *) glu->data + ggml_nbytes(glu)) {
            fprintf(stderr, "fused step out of bounds: ith %d c %d/%d k %d a %d r %lld..%lld nr0 %lld\n",
                    ith, c, n_chunks, k, a, (long long) r0, (long long) r1, (long long) nr0);
            GGML_ABORT("fused step out of bounds");
        }

        if (a >= zero_a) {
            memset(y + r0, 0, (size_t) (r1 - r0)*sizeof(float));   // a zero slice gives silu(0)*0 = +0
            continue;
        }
        if (wait) {
            if (eg && eg->ready && sg + a < eg->n_as) waited_g += ggml_moe_wait_ready(eg, sg + a, ith, &waited_cb_g);
            if (eu && eu->ready && su + a < eu->n_as) waited_u += ggml_moe_wait_ready(eu, su + a, ith, &waited_cb_u);
            if (c % cpi == 0) {   // the expert's first chunk - its gate and up compute start now (counters only)
                ggml_moe_note_early(eg, sg + a);
                ggml_moe_note_early(eu, su + a);
            }
        }
        const char * pg = (const char *) ag->data + (size_t) a*nb02;
        const char * pu = (const char *) au->data + (size_t) a*nb02;
        int64_t r = r0;
#ifdef GGML_CPU_HAVE_MXFP4_X4
        if (x4fn) {
            for (; r + 3 < r1; r += 4) {
                x4fn((int) ne00, tg + (r - r0), pg + r*nb01, nb01, qb, x4, x4_pf);
                x4fn((int) ne00, tu + (r - r0), pu + r*nb01, nb01, qb, x4, x4_pf);
            }
        }
#else
        GGML_UNUSED(x4);
        GGML_UNUSED(x4_pf);
#endif
        for (; r < r1; r++) {
            vec_dot((int) ne00, tg + (r - r0), 0, pg + r*nb01, 0, qb, 0, 1);
            vec_dot((int) ne00, tu + (r - r0), 0, pu + r*nb01, 0, qb, 0, 1);
        }
        if (!f->clamped) {
            ggml_vec_swiglu_f32((int) (r1 - r0), y + r0, tg, tu);
        } else {
            // the SILU, CLAMP, CLAMP and MUL ops element by element: silu on 16-aligned rows takes the op's SIMD path,
            // clamp is the op's MAX(MIN(x, max), min) and mul a plain product, so the same bits
            ggml_vec_silu_f32((int) (r1 - r0), tg, tg);
            for (int64_t i = 0; i < r1 - r0; i++) {
                const float g = MAX(MIN(tg[i], f->cg_max), f->cg_min);
                const float u = MAX(MIN(tu[i], f->cu_max), f->cu_min);
                y[r0 + i] = g*u;
            }
        }
    }

    if (ggml_m11_flag()) {   // this thread's end of the fused op
        const int64_t te = ggml_time_us();
        ggml_m11_max(&g_m11_cur.f_end_max, te);
        atomic_fetch_add_explicit(&g_m11_cur.f_end_sum, te, memory_order_relaxed);
    }
    // the op's stall per pool is the longest wait of any of its threads, as in mul_mat_id
    const struct ggml_moe_pool_slots * es[2] = { eg, eu };
    const int64_t ws[2] = { waited_g, waited_u }, wc[2] = { waited_cb_g, waited_cb_u };
    for (int j = 0; j < 2; j++) {
        if (es[j] && es[j]->ready && (ws[j] > 0 || wc[j] > 0)) {
            ggml_moe_book_wait(es[j], ws[j], wc[j]);
        }
    }
}

// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
static bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

// ggml_cpu_set_fuse_small: fewer small CPU steps. Only nodes of this CPU graph (one split of the scheduler) are seen here,
// so a GPU graph is never touched. Both fusions run each node's own kernel on the same data, so the results are the same bits.
//  1 / 2: a router chain (from the router's SIGMOID "ffn_moe_probs-L": ARGSORT, GET_ROWS, SUM_ROWS, CLAMP, DIV) or a guess chain
//         (from a guess's SIGMOID "ffn_moe_probs_pred...": ARGSORT, CONT) of small nodes (<= 8 rows, as in a reply): thread 0
//         computes them one after another with one thread each (every op here splits by rows, so one thread does all of them the
//         same way), the other threads wait once at the end - one barrier instead of one per node
// 64:     consecutive one-row MUL_MATs on the same F32 input (Q / K / V, a shared expert's gate / up): the input is converted once,
//         each thread computes its fixed share of every matmul's chunks (the same chunk plan as ggml_compute_forward_mul_mat; which
//         thread computes a row does not change it), one barrier after the conversion and one at the end
static bool ggml_fs_empty(const struct ggml_tensor * t) {
    return ggml_op_is_empty(t->op) || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
}

static bool ggml_fs_chain_op(const struct ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_UNARY:    return ggml_get_unary_op(t) == GGML_UNARY_OP_SIGMOID;
        case GGML_OP_ARGSORT:
        case GGML_OP_GET_ROWS:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_CLAMP:
        case GGML_OP_DIV:
        case GGML_OP_CONT:     break;
        default:               return false;
    }
    return ggml_nrows(t) <= 8 && t->ne[0] <= 4096 && (!t->src[0] || ggml_nrows(t->src[0]) <= 8);
}

static int ggml_cpu_fuse_small_chain(const struct ggml_cgraph * cgraph, const int node_n, const struct ggml_compute_params * params, const int mask) {
    const struct ggml_tensor * node = cgraph->nodes[node_n];
    if (node->op != GGML_OP_UNARY || ggml_get_unary_op(node) != GGML_UNARY_OP_SIGMOID || strncmp(node->name, "ffn_moe_probs", 13) != 0) {
        return 0;
    }
    const bool guess = strstr(node->name, "pred") != NULL;
    if (!(mask & (guess ? 2 : 1)) || !ggml_fs_chain_op(node)) {
        return 0;
    }
    int last = node_n, n_real = 1;
    for (int j = node_n + 1; j < cgraph->n_nodes; j++) {
        const struct ggml_tensor * t = cgraph->nodes[j];
        if (ggml_fs_empty(t)) {
            continue;
        }
        if (!ggml_fs_chain_op(t) || (t->op == GGML_OP_UNARY && t != node)) {
            break;   // a node outside the chain, or the next chain's SIGMOID
        }
        last = j;
        n_real++;
    }
    if (n_real < 2) {
        return 0;
    }
    if (params->ith == 0) {
        struct ggml_compute_params p1 = *params;
        p1.ith = 0;
        p1.nth = 1;
        for (int j = node_n; j <= last; j++) {
            struct ggml_tensor * t = cgraph->nodes[j];
            if (!ggml_fs_empty(t)) {
                ggml_compute_forward(&p1, t);
            }
        }
        atomic_fetch_add_explicit(&g_ggml_cpu_fuse_small_n[guess ? 1 : 0], 1, memory_order_relaxed);
    }
    return last - node_n;
}

static bool ggml_fs_mm_ok(const struct ggml_tensor * t) {
    if (t->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(t, 1) != 0) {
        return false;   // (op param 1: the Hadamard hint)
    }
    const struct ggml_tensor * a = t->src[0];
    const struct ggml_tensor * b = t->src[1];
    const enum ggml_type vdt = type_traits_cpu[a->type].vec_dot_type;
    return a->extra == NULL && b->type == GGML_TYPE_F32 && ggml_is_contiguous(b) && b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1 &&
           a->ne[2] == 1 && a->ne[3] == 1 && vdt != GGML_TYPE_F32 && type_traits_cpu[vdt].from_float != NULL && t->type == GGML_TYPE_F32 &&
           a->nb[0] == ggml_type_size(a->type);
}

static int ggml_cpu_fuse_small_mm(const struct ggml_cgraph * cgraph, const int node_n, const struct ggml_compute_params * params) {
    struct ggml_tensor * node = cgraph->nodes[node_n];
    if (!ggml_fs_mm_ok(node)) {
        return 0;
    }
    const enum ggml_type vdt = type_traits_cpu[node->src[0]->type].vec_dot_type;
    struct ggml_tensor * mm[16];
    int n_mm = 0, last = node_n;
    mm[n_mm++] = node;
    for (int j = node_n + 1; j < cgraph->n_nodes && n_mm < 16; j++) {
        struct ggml_tensor * t = cgraph->nodes[j];
        if (ggml_fs_empty(t)) {
            continue;
        }
        if (!ggml_fs_mm_ok(t) || t->src[1] != node->src[1] || type_traits_cpu[t->src[0]->type].vec_dot_type != vdt) {
            break;
        }
        mm[n_mm++] = t;
        last = j;
    }
    if (n_mm < 2) {
        return 0;
    }
    const int ith = params->ith, nth = params->nth;
    // the input converted once (as ggml_compute_forward_mul_mat: every thread a block range of the row)
    {
        const struct ggml_tensor * src1 = node->src[1];
        const int64_t ne10 = src1->ne[0];
        const int64_t bs = ggml_blck_size(vdt);
        const int64_t b0 = (ith * ne10/bs) / nth;
        const int64_t b1 = ((ith + 1) * ne10/bs) / nth;
        type_traits_cpu[vdt].from_float((const float *) src1->data + b0*bs, (char *) params->wdata + b0*ggml_type_size(vdt), (b1 - b0) * bs);
    }
    ggml_barrier(params->threadpool);
    for (int m = 0; m < n_mm; m++) {
        struct ggml_tensor * dst = mm[m];
        const struct ggml_tensor * src0 = dst->src[0];
        const int64_t nr0 = dst->ne[0];
        const int64_t nr1 = dst->ne[1] * dst->ne[2] * dst->ne[3];
        int chunk_size = (nr0 == 1 || nr1 == 1) ? 64 : 16;
        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
        if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }
        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;
        for (int64_t c = ith; c < nchunk0 * nchunk1; c += nth) {
            const int64_t ith0 = c % nchunk0;
            const int64_t ith1 = c / nchunk0;
            const int64_t ir0_start = dr0 * ith0, ir0_end = MIN(ir0_start + dr0, nr0);
            const int64_t ir1_start = dr1 * ith1, ir1_end = MIN(ir1_start + dr1, nr1);
            int64_t nrv = type_traits_cpu[src0->type].nrows;
            if ((nr0 % 2 != 0) || (dst->src[1]->ne[1] % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
                nrv = 1;
            }
            ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, nrv, ir0_start, ir0_end, ir1_start, ir1_end);
        }
    }
    if (ith == 0) {
        atomic_fetch_add_explicit(&g_ggml_cpu_fuse_small_n[2], 1, memory_order_relaxed);
    }
    return last - node_n;
}

static GGML_HO_INLINE int ggml_cpu_try_fuse_ops(
        const struct ggml_cgraph * cgraph,
        const int node_n,
        const struct ggml_compute_params * params,
        const struct ggml_cplan * cplan) {

    if (ggml_cpu_disable_fusion || cplan->use_ref) {
        return 0;
    }

    struct ggml_tensor * node = cgraph->nodes[node_n];

    const int fs = params->threadpool->fuse_small;
    if (fs) {
        if (fs & (1 | 2)) {
            const int n = ggml_cpu_fuse_small_chain(cgraph, node_n, params, fs);
            if (n > 0) {
                return n;
            }
        }
        if (fs & 64) {
            const int n = ggml_cpu_fuse_small_mm(cgraph, node_n, params);
            if (n > 0) {
                return n;
            }
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID) {
        // mul_mat_id(gate) + mul_mat_id(up) + swiglu_split -> one step (off unless ggml_cpu_set_moe_fuse(1))
        struct ggml_moe_fuse fz;
        const bool fused = ggml_cpu_moe_up_gate_match(cgraph, node_n, params, &fz);
        // MOE_STREAM_TEST_FUSE_TRACE=1: per MoE matmul, did the fused step engage, and what follows it if not
        static int trace = -1;
        if (trace < 0) {
            trace = getenv("MOE_STREAM_TEST_FUSE_TRACE") ? 400 : 0;
        }
        if (trace > 0 && params->ith == 0 && params->threadpool->moe_fuse && node->src[2] && node->src[2]->ne[1] == 1 &&
            (strncmp(node->name, "ffn_moe_up", 10) == 0 || strncmp(node->name, "ffn_moe_gate", 12) == 0)) {
            trace--;
            const struct ggml_tensor * n1 = node_n + 1 < cgraph->n_nodes ? cgraph->nodes[node_n + 1] : NULL;
            const struct ggml_tensor * n2 = node_n + 2 < cgraph->n_nodes ? cgraph->nodes[node_n + 2] : NULL;
            fprintf(stderr, "fuse trace: %-18s %s | next %s (%s) | next2 %s (%s) | n_nodes %d at %d\n", node->name,
                    fused ? (fz.clamped ? "FUSED (clamped)" : "FUSED") : "not fused", n1 ? n1->name : "-", n1 ? ggml_op_desc(n1) : "-", n2 ? n2->name : "-",
                    n2 ? ggml_op_desc(n2) : "-", cgraph->n_nodes, node_n);
        }
        if (fused) {
            ggml_compute_forward_moe_up_gate_fused(params, &fz);
            return fz.n_nodes - 1;
        }
    }

    if (node->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        const enum ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ggml_can_fuse(cgraph, node_n, fuse_ops, 2)) {
            struct ggml_tensor * mul_node = cgraph->nodes[node_n + 1];
            const struct ggml_tensor * mul_w = (mul_node->src[0] == node)
                ? mul_node->src[1] : mul_node->src[0];
            if (node->src[0]->type  == GGML_TYPE_F32 &&
                mul_node->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == node->ne[0]   &&
                mul_w->nb[0]        == sizeof(float)) {

                ggml_compute_forward_rms_norm_mul_fused(params, node, mul_node);
                return 1;
            }
        }
    }

    return 0;
}

// The thread loop is one body compiled twice (forced inline, `ho` a constant in each copy), so the untimed
// copy has no timer code in it and the timed copy cannot drift from it

static GGML_HO_INLINE thread_ret_t ggml_graph_compute_thread_body(void * data, const bool ho) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;
    int64_t * const ho_ts  = ho ? tp->ho_ts + (size_t) state->ith*tp->ho_cap*2 : NULL;   // this thread's [node][2]
    int64_t * const ho_wt  = ho ? tp->ho_wt + (size_t) state->ith*tp->ho_cap : NULL;     // this thread's [node]

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#else
    set_numa_thread_affinity(state->ith);
#endif

    struct ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        if (ggml_op_is_empty(node->op)) {
            // skip NOPs
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        const int ho_node = node_n;
        const int64_t ho_w0 = ho ? t_ho_slot_wait_us : 0;
        if (ho) {
            ho_ts[(size_t) ho_node*2 + 0] = ggml_time_us();
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);
        }

        if (ho) {
            ho_ts[(size_t) ho_node*2 + 1] = ggml_time_us();
            ho_wt[ho_node] = t_ho_slot_wait_us - ho_w0;
            if (state->ith == 0) {
                tp->ho_fused[ho_node] = n_fused > 0 ? n_fused : 0;
            }
        }

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

    if (ho && state->ith == 0) {
        tp->ho_t_final = ggml_time_us();
    }

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    return 0;
}

// two separate functions, so the untimed one is compiled on its own and can be compared with the code without the timers
static GGML_HO_NOINLINE thread_ret_t ggml_graph_compute_thread_plain(void * data) {
    return ggml_graph_compute_thread_body(data, false);
}

static GGML_HO_NOINLINE thread_ret_t ggml_graph_compute_thread_timed(void * data) {
    return ggml_graph_compute_thread_body(data, true);
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    return state->threadpool->ho_on ? ggml_graph_compute_thread_timed(data) : ggml_graph_compute_thread_plain(data);
}

// Before the threads start - room for this graph's stamps, thread 0's starts cleared
static void ggml_cpu_handoff_prepare(struct ggml_threadpool * tp, const struct ggml_cgraph * cgraph, int n_threads) {
    const size_t n = (size_t) cgraph->n_nodes;
    if (tp->ho_cap < n || tp->ho_nth != n_threads) {
        free(tp->ho_ts);
        free(tp->ho_fused);
        free(tp->ho_nodes);
        free(tp->ho_wt);
        tp->ho_wt    = (int64_t *) calloc((n ? n : 1) * n_threads, sizeof(int64_t));
        tp->ho_ts    = (int64_t *) malloc(sizeof(int64_t) * 2 * (n ? n : 1) * n_threads);
        tp->ho_fused = (int *) malloc(sizeof(int) * (n ? n : 1));
        tp->ho_nodes = (struct ggml_handoff_cpu_node *) malloc(sizeof(struct ggml_handoff_cpu_node) * (n ? n : 1));
        GGML_ASSERT(tp->ho_ts && tp->ho_fused && tp->ho_nodes && tp->ho_wt);
        tp->ho_cap = n;
        tp->ho_nth = n_threads;
    }
    for (size_t i = 0; i < n; i++) {
        tp->ho_ts[i*2 + 0] = 0;   // thread 0's row
    }
    tp->ho_t_final = 0;
}

// After the threads joined - one record per node that ran, handed to ggml-base
static void ggml_cpu_handoff_flush(struct ggml_threadpool * tp, const struct ggml_cgraph * cgraph, int n_threads,
                                   int64_t t_enter) {
    const int64_t t_exit = ggml_time_us();
    const size_t  cap    = tp->ho_cap;
#define HO_TS(node, thread, k) tp->ho_ts[((size_t) (thread)*cap + (size_t) (node))*2 + (k)]
    int n_rec = 0, prev = -1;
    int64_t fork_us = 0, fork_stagger = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const int64_t s0 = HO_TS(i, 0, 0);
        if (s0 == 0) {
            continue;   // not run (empty op, no compute flag, or fused into an earlier node)
        }
        if (prev >= 0) {
            tp->ho_nodes[n_rec - 1].wall_us = s0 - HO_TS(prev, 0, 0);
        }
        int64_t smin = INT64_MAX, smax = INT64_MIN, emin = INT64_MAX, emax = INT64_MIN, wsum = 0, wmax = 0;
        int64_t slot_wmax = 0, esum = 0;
        for (int t = 0; t < n_threads; t++) {
            const int64_t st = HO_TS(i, t, 0);
            const int64_t en = HO_TS(i, t, 1);
            smin = st < smin ? st : smin; smax = st > smax ? st : smax;
            emin = en < emin ? en : emin; emax = en > emax ? en : emax;
            wsum += en - st;
            wmax  = en - st > wmax ? en - st : wmax;
            const int64_t w = tp->ho_wt[(size_t) t*cap + (size_t) i];
            slot_wmax = w > slot_wmax ? w : slot_wmax;
            esum += en;
        }
        const struct ggml_tensor * node = cgraph->nodes[i];
        struct ggml_handoff_cpu_node * r = &tp->ho_nodes[n_rec++];
        r->name        = node->name;
        r->op          = ggml_op_desc(node);
        r->is_mmid     = node->op == GGML_OP_MUL_MAT_ID;
        r->n_fused     = tp->ho_fused[i];
        r->wall_us     = 0;
        r->stagger_us  = smax - smin;
        r->work_sum_us = wsum;
        r->work_max_us = wmax;
        r->spread_us   = emax - emin;
        r->slot_wait_max_us = slot_wmax;
        r->idle_us          = emax - (esum + n_threads/2)/n_threads;   // mean over the threads of (last end - its end)
        if (prev < 0) {
            fork_us      = s0 - t_enter;
            fork_stagger = smax - smin;
        }
        prev = i;
    }
    if (prev >= 0) {
        tp->ho_nodes[n_rec - 1].wall_us = tp->ho_t_final - HO_TS(prev, 0, 0);
    }
#undef HO_TS
    ggml_handoff_cpu_graph(t_exit - t_enter, fork_us, prev >= 0 ? t_exit - tp->ho_t_final : 0, fork_stagger, n_threads,
                           tp->ho_nodes, n_rec);
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
static inline bool ggml_graph_compute_thread_ready(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    int n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        return true;
    }

    return false;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    int n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

static struct ggml_threadpool * ggml_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
        threadpool->ho_on            = 0;
        threadpool->ho_nth           = 0;
        threadpool->ho_ts            = NULL;
        threadpool->ho_fused         = NULL;
        threadpool->ho_cap           = 0;
        threadpool->ho_t_final       = 0;
        threadpool->ho_nodes         = NULL;
        threadpool->ho_wt            = NULL;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
    struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    for (int j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#else // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    return threadpool;
}

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {
    return ggml_threadpool_new_impl(tpp, NULL, NULL);
}

enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // One value of the CPU MoE switches for the whole graph, taken before any thread starts
    threadpool->moe_fuse    = atomic_load_explicit(&g_ggml_cpu_moe_fuse, memory_order_relaxed) && !g_ggml_cpu_tiled_forced;
    threadpool->fuse_small  = g_ggml_cpu_tiled_forced ? 0 : atomic_load_explicit(&g_ggml_cpu_fuse_small, memory_order_relaxed);
    threadpool->mxfp4_x4    = atomic_load_explicit(&g_ggml_cpu_mxfp4_x4, memory_order_relaxed);
    threadpool->mxfp4_x4_pf = atomic_load_explicit(&g_ggml_cpu_mxfp4_x4_pf, memory_order_relaxed);
    threadpool->moe_dispatch = atomic_load_explicit(&g_ggml_cpu_moe_dispatch, memory_order_relaxed) && !g_ggml_cpu_tiled_forced;
    threadpool->moe_pieces   = atomic_load_explicit(&g_ggml_cpu_moe_pieces, memory_order_relaxed);
    threadpool->moe_conv1    = atomic_load_explicit(&g_ggml_cpu_moe_conv1, memory_order_relaxed);
    threadpool->cpu_track    = atomic_load_explicit(&g_ggml_cpu_track, memory_order_relaxed);   // the CPU track

    // The hand-off timers (GGML_PROFILE_HANDOFF, read once in ggml_cpu_init)
    int64_t ho_t_enter = 0;
    threadpool->ho_on = g_ggml_cpu_handoff;
    if (g_ggml_cpu_handoff) {
        ggml_cpu_handoff_prepare(threadpool, cgraph, n_threads);
        ho_t_enter = ggml_time_us();   // after the stamps' preparation, so the fork is the threads' start only
    }

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            int ith = omp_get_thread_num();

            ggml_thread_apply_priority(threadpool->prio);
            if (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads);
        n_threads = threadpool->n_threads;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    if (g_ggml_cpu_handoff) {
        ggml_cpu_handoff_flush(threadpool, cgraph,
                atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK, ho_t_enter);
    }

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    return ggml_graph_compute(cgraph, &cplan);
}

void ggml_cpu_fp32_to_fp32(const float * x, float * y, int64_t n) {
    memcpy(y, x, n * sizeof(float));
}

void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#elif defined(__VXE__) || defined(__VXE2__)
    for (; i + 7 < n; i += 8) {
        const uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        const uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

void ggml_cpu_fp32_to_bf16(const float * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

void ggml_cpu_fp32_to_i32(const float * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = x[i];
    }
}

void ggml_cpu_bf16_to_fp32(const ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (const __m256i *)(x + i))),
                                16)));
    }
#endif
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (const __m128i *)(x + i))),
                                16)));
    }
#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((const __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_rvv_vlen(void) {
#if defined(__riscv) && defined(__riscv_v_intrinsic)
    return ggml_riscv_arch_features.rvv_vlen;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_llamafile(void) {
#if defined(GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme2(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    return 1;
#else
    return 0;
#endif
}

void ggml_cpu_init(void) {
    // needed to initialize ggml_time
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            if (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#else
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#if defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            const char * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != NULL && atoi(env) == 1);
        }

        {
            // The tiled kernel's test switch (tiled/tiled.cpp reads the same variable): with it forced, the unfused
            // path takes the tiled kernel, so the fused step would not compute what that path computes
            const char * env = getenv("GGML_CPU_TILED_MM_FORCE");
            g_ggml_cpu_tiled_forced = (env != NULL && atoi(env) == 1);
        }

        // The hand-off timers, one value for the whole run (ggml-base reads GGML_PROFILE_HANDOFF once)
        g_ggml_cpu_handoff = ggml_handoff_on() ? 1 : 0;
        {   // the CPU track: the start value of ggml_cpu_set_track (test-backend-ops runs the new paths with it)
            const char * env = getenv("GGML_CPU_TRACK");
            if (env) {
                ggml_cpu_set_track(atoi(env));
            }
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}

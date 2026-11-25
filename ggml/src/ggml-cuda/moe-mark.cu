#include "moe-mark.cuh"

#include <algorithm>
#include <atomic>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

static inline void ggml_moe_mark_spin_pause(void) {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
}

// acquire load of a 32-bit word the GPU writes into pinned host memory after __threadfence_system()
static inline uint32_t ggml_moe_mark_load_acquire_u32(const uint32_t * p) {
#if defined(_MSC_VER) && !defined(__clang__)
    const uint32_t v = *static_cast<const volatile uint32_t *>(p);
    std::atomic_signal_fence(std::memory_order_acquire);
#if defined(_M_ARM64) || defined(_M_ARM64EC)
    std::atomic_thread_fence(std::memory_order_acquire);
#endif
    return v;
#else
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}

#if defined(GGML_USE_HIP)
static std::atomic<int32_t> g_hip_sync_spin_us{0};

void ggml_hip_set_sync_spin_us(int32_t spin_us) {
    const int32_t eff = spin_us == 1 ? 400 : std::max<int32_t>(0, spin_us);
    g_hip_sync_spin_us.store(eff, std::memory_order_relaxed);
}

int32_t ggml_hip_get_sync_spin_us(void) {
    static const int32_t env_us = []() -> int32_t {
        const char * v = getenv("GGML_HIP_SYNC_SPIN_US");
        if (v == nullptr) {
            return -1;
        }
        const int a = atoi(v);
        return a == 1 ? 400 : std::max(0, a);
    }();
    return env_us >= 0 ? env_us : g_hip_sync_spin_us.load(std::memory_order_relaxed);
}

void ggml_hip_spin_pause(void) {
    ggml_moe_mark_spin_pause();
}

static inline uint32_t ggml_hip_load_acquire_u32(const uint32_t * p) {
    return ggml_moe_mark_load_acquire_u32(p);
}
#elif !defined(GGML_USE_MUSA)
static inline int32_t ggml_cuda_get_mark_spin_us(void) {
    static const int32_t env_us = []() -> int32_t {
        const char * v = getenv("GGML_CUDA_MARK_SPIN_US");
        if (v == nullptr) {
            return 350;
        }
        const int a = atoi(v);
        return a == 1 ? 350 : std::max(0, a);
    }();
    return env_us;
}
#endif

#if !defined(GGML_USE_MUSA)

// See moe-mark.cuh. One table per kind for the process (one engine registers; each entry on its own device);
// written at start-up and at teardown, read by the graph thread.
struct ggml_cuda_moe_mark_entry {
    uint8_t        * slot     = nullptr;   // pinned
    int64_t          cap      = 0;         // bytes of data the slot holds
    uint32_t       * counter  = nullptr;   // pinned
    const uint32_t * expected = nullptr;   // pinned: the host's number of this call
    cudaEvent_t      ev       = nullptr;
    int              device   = -1;        // the CUDA device whose stream runs the marked node (the event is created there)
#if defined(GGML_USE_HIP)
    cudaEvent_t      tmp      = nullptr;   // the point in the compute stream the side stream waits for (see below)
#endif
};

#if defined(GGML_USE_HIP)
// HIP refuses a sync of an event while the stream that last recorded it captures a graph, and the refused sync ends that
// capture. The guess helper waits on a mark's event while the compute stream may capture the next graph, so a mark recorded
// outside a capture goes through a side stream per device that first waits for the compute stream (a mark inside a capture
// is an external record node, which HIP lets the host wait on)
static cudaStream_t g_moe_mark_side[GGML_CUDA_MAX_DEVICES] = {};
#endif

struct ggml_cuda_moe_mark_kind_state {
    std::vector<ggml_cuda_moe_mark_entry> e;
    uint32_t bit  = 0;
    int32_t  skip = -1;
};

static ggml_cuda_moe_mark_kind_state g_moe_mark[GGML_CUDA_MOE_MARK_N];
static uint32_t                      g_moe_mark_bits   = 0;
static const void *                  g_moe_mark_owner  = nullptr;
static std::vector<int>              g_moe_mark_devices;       // every device with an entry (the sync after an aborted call)

// the sources a mark copies (the node, then for a router mark its src[1..]), packed at 16-byte offsets
#define MOE_MARK_MAX_SRC 4
struct moe_mark_srcs {
    const int32_t * src[MOE_MARK_MAX_SRC];
    int32_t         n_words[MOE_MARK_MAX_SRC];
    int32_t         dst_word[MOE_MARK_MAX_SRC];
    int32_t         n_src;
    // GGML_CUDA_MOE_MARK_IDS=1: source 0 is the router's top-k view itself (rows of v0_ne0 words, v0_nb1 words apart), copied
    // into the slot AND into the ids CONT node's own memory (v0_cont), whose separate copy kernel is then not launched
    int32_t       * v0_cont;
    int32_t         v0_ne0;
    int32_t         v0_nb1;
};

// one block: the data, a system-scope fence, then the counter - a reader that sees this call's counter sees this call's data.
// 256 threads copy 16 bytes at a time where source and slot are both 16-byte aligned (the slot's offsets
// are; the rest word by word); the call's number - pinned host memory, a PCIe read - is read first, so its wait overlaps the copy (the
// host writes it before the graph is launched, no kernel does); every writer fences its own writes system-wide before the barrier
// (as before), then thread 0 writes the counter.
static constexpr int MOE_MARK_THREADS = 256;
static __global__ void k_moe_mark(const moe_mark_srcs s, int32_t * dst, const uint32_t * expected, uint32_t * counter) {
    uint32_t want = 0;
    if (threadIdx.x == 0) {
        want = *((const volatile uint32_t *) expected);
    }
    ggml_cuda_pdl_sync();   // PDL-launched - the sources are read only after the kernel before (the ids copy) ended
    int k0 = 0;
    if (s.v0_cont != nullptr) {   // the strided view, word by word, into the slot and the CONT node
        int32_t * d = dst + s.dst_word[0];
        for (int i = threadIdx.x; i < s.n_words[0]; i += blockDim.x) {
            const int32_t v = s.src[0][(i / s.v0_ne0) * s.v0_nb1 + i % s.v0_ne0];
            d[i]         = v;
            s.v0_cont[i] = v;
        }
        k0 = 1;
    }
    for (int k = k0; k < s.n_src; k++) {
        const int32_t * src = s.src[k];
        int32_t       * d   = dst + s.dst_word[k];
        const int       n   = s.n_words[k];
        int             i0  = 0;
        if ((((uintptr_t) src | (uintptr_t) d) & 15) == 0) {
            const int4 * s4 = (const int4 *) src;
            int4       * d4 = (int4 *) d;
            for (int i = threadIdx.x; i < n/4; i += blockDim.x) {
                d4[i] = s4[i];
            }
            i0 = n/4*4;
        }
        for (int i = i0 + threadIdx.x; i < n; i += blockDim.x) {
            d[i] = src[i];
        }
    }
    __threadfence_system();   // every writer orders its own data writes system-wide before the barrier
    __syncthreads();
    if (threadIdx.x == 0) {
        *((volatile uint32_t *) counter) = want;
    }
}

bool ggml_cuda_moe_mark_register(const void * owner, int device, int32_t n_layers,
                                 const ggml_cuda_moe_mark_table * tables, int n_tables, const int32_t * const * dev_of,
                                 const char ** why) {
    if (g_moe_mark_owner != nullptr && g_moe_mark_owner != owner) {
        *why = "another model in this process already uses needed-early (one per process)";
        return false;
    }
    ggml_cuda_moe_mark_unregister(owner);
    std::vector<int> devices;
    for (int k = 0; k < n_tables && k < GGML_CUDA_MOE_MARK_N; k++) {
        const ggml_cuda_moe_mark_table & t = tables[k];
        ggml_cuda_moe_mark_kind_state & st = g_moe_mark[k];
        st.e.assign((size_t) n_layers, ggml_cuda_moe_mark_entry());
        bool any = false;
        for (int32_t il = 0; il < n_layers; il++) {
            if (t.slots == nullptr || t.slots[il] == nullptr) {
                continue;
            }
            ggml_cuda_moe_mark_entry & e = st.e[il];
            e.slot     = (uint8_t *) t.slots[il];
            e.cap      = t.caps[il];
            e.counter  = t.counters[il];
            e.expected = t.expected[il];
            e.device   = dev_of != nullptr && dev_of[k] != nullptr && dev_of[k][il] >= 0 ? dev_of[k][il] : device;
            if (std::find(devices.begin(), devices.end(), e.device) == devices.end()) {
                devices.push_back(e.device);
            }
            ggml_cuda_set_device(e.device);   // the event belongs to the device whose stream records it
            if (cudaEventCreateWithFlags(&e.ev, cudaEventDisableTiming) != cudaSuccess
#if defined(GGML_USE_HIP)
                    || cudaEventCreateWithFlags(&e.tmp, cudaEventDisableTiming) != cudaSuccess
                    || (g_moe_mark_side[e.device] == nullptr &&
                        cudaStreamCreateWithFlags(&g_moe_mark_side[e.device], cudaStreamNonBlocking) != cudaSuccess)
#endif
                    ) {
                (void) cudaGetLastError();
                ggml_cuda_moe_mark_unregister(owner);
                *why = "the GPU backend could not create the events";
                return false;
            }
            any = true;
        }
        st.bit  = any ? t.flag_bit : 0;
        st.skip = t.skip_layer;
    }
    ggml_cuda_set_device(device);   // (the registration's device stays the current one, as before)
    g_moe_mark_owner   = owner;
    g_moe_mark_devices = devices;
    g_moe_mark_bits    = 0;
    for (const auto & st : g_moe_mark) {
        g_moe_mark_bits |= st.bit;
    }
    return true;
}

bool ggml_cuda_moe_mark_wait(int32_t kind, int32_t il) {
    if (kind < 0 || kind >= GGML_CUDA_MOE_MARK_N || il < 0 || (size_t) il >= g_moe_mark[kind].e.size() ||
            g_moe_mark[kind].e[il].ev == nullptr) {
        GGML_LOG_ERROR("%s: layer %d has no registered %s event\n", __func__, il, kind == GGML_CUDA_MOE_MARK_ROUTER ? "\"router done\"" :
                kind == GGML_CUDA_MOE_MARK_DONE ? "\"gate/up done\"" : "\"guess done\"");
        return false;
    }
    const ggml_cuda_moe_mark_entry & e = g_moe_mark[kind].e[il];
#if defined(GGML_USE_HIP)
    const int32_t spin_us = ggml_hip_get_sync_spin_us();
#elif !defined(GGML_USE_MUSA)
    const int32_t spin_us = ggml_cuda_get_mark_spin_us();
#else
    const int32_t spin_us = 0;
#endif
    // k_moe_mark issues __threadfence_system() before writing *e.counter = want into pinned host memory.
    // Bounded user-space polling avoids CUDA/HSA driver + container syscall latency when the mark is ready or arrives soon.
    if (spin_us > 0 && e.counter != nullptr && e.expected != nullptr) {
        const uint32_t want = *((const volatile uint32_t *) e.expected);
        if (ggml_moe_mark_load_acquire_u32(e.counter) == want) {
            return true;
        }
        const int64_t t0 = ggml_time_us();
        for (uint32_t it = 0;; ++it) {
            ggml_moe_mark_spin_pause();
            if (ggml_moe_mark_load_acquire_u32(e.counter) == want) {
                return true;
            }
            if ((it & 31u) == 31u && ggml_time_us() - t0 >= (int64_t) spin_us) {
                break;
            }
        }
    }
    const cudaError_t err = cudaEventSynchronize(e.ev);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: layer %d: CUDA error %s\n", __func__, il, cudaGetErrorString(err));
        return false;
    }
    return true;
}

void ggml_cuda_moe_mark_unregister(const void * owner) {
    if (g_moe_mark_owner != nullptr && g_moe_mark_owner != owner) {
        return;   // Another engine's table
    }
    g_moe_mark_bits = 0;
    for (auto & st : g_moe_mark) {
        for (auto & e : st.e) {
            if (e.ev != nullptr) {
                (void) cudaEventDestroy(e.ev);
            }
#if defined(GGML_USE_HIP)
            if (e.tmp != nullptr) {
                (void) cudaEventDestroy(e.tmp);
            }
#endif
        }
        st.e.clear();
        st.bit  = 0;
        st.skip = -1;
    }
#if defined(GGML_USE_HIP)
    for (cudaStream_t & s : g_moe_mark_side) {
        if (s != nullptr) {
            (void) cudaStreamDestroy(s);
            s = nullptr;
        }
    }
#endif
    g_moe_mark_owner  = nullptr;
    g_moe_mark_devices.clear();
}

uint32_t ggml_cuda_moe_mark_bits(void) {
    return g_moe_mark_bits;
}

// After a call aborted part-way, every producer kernel it enqueued (on any device with an entry) has run before the next call's
// number is written
bool ggml_cuda_moe_mark_sync(void) {
    for (const int d : g_moe_mark_devices) {
        ggml_cuda_set_device(d);
        const cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            GGML_LOG_ERROR("%s: CUDA error %s\n", __func__, cudaGetErrorString(err));
            return false;
        }
    }
    return true;
}

// GGML_CUDA_MOE_MARK_IDS=1 - a router mark whose node is the CONT of the top-k view (I32, rows contiguous) takes the CONT's
// work: the mark kernel reads the view and writes the CONT node's memory too, so the CONT is not launched (its readers - the VRAM
// pass's get_rows, the scheduler's copies - read the same bytes, written earlier in the same stream)
static bool ggml_cuda_moe_mark_ids_on() {
    static const char * v = getenv("GGML_CUDA_MOE_MARK_IDS");
    return v != nullptr ? atoi(v) != 0 : ggml_cuda_prefill_fast_defaults_on();
}

bool ggml_cuda_moe_mark_takes_node(const ggml_tensor * node) {
    if (!ggml_cuda_moe_mark_ids_on() || (node->flags & g_moe_mark[GGML_CUDA_MOE_MARK_ROUTER].bit) == 0 ||
            g_moe_mark[GGML_CUDA_MOE_MARK_ROUTER].bit == 0 || node->op != GGML_OP_CONT) {
        return false;
    }
    const ggml_tensor * v = node->src[0];
    return v != nullptr && v->type == GGML_TYPE_I32 && node->type == GGML_TYPE_I32 && v->nb[0] == 4 && v->nb[1] % 4 == 0 &&
           v->ne[2] == 1 && v->ne[3] == 1 && ggml_is_contiguous(node) && ggml_nelements(node) == ggml_nelements(v) &&
           node->ne[0] == v->ne[0] && v->data != nullptr && node->data != nullptr;
}

void ggml_cuda_moe_mark_after(const ggml_cgraph * cgraph, int i0, int i1, cudaStream_t stream, bool capturing, int device) {
    for (int i = i0; i <= i1 && i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        for (int k = 0; k < GGML_CUDA_MOE_MARK_N; k++) {
            ggml_cuda_moe_mark_kind_state & st = g_moe_mark[k];
            if ((node->flags & st.bit) == 0) {
                continue;
            }
            const int32_t il = node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1];
            if (il < 0 || (size_t) il >= st.e.size() || st.e[il].slot == nullptr) {
                GGML_ABORT("%s: node %s is marked for layer %d, which has no registered slot", __func__, node->name, il);
            }
            ggml_cuda_moe_mark_entry & e = st.e[il];
            if (device != e.device) {
                GGML_ABORT("%s: node %s (layer %d) runs on CUDA%d, but its needed-early event belongs to CUDA%d", __func__, node->name,
                        il, device, e.device);
            }
            // the sources: the node itself; a router mark also copies its src[1..] (the other inputs of the layer's CPU split)
            moe_mark_srcs s = {};
            int64_t off = 0;
            for (int j = 0; j < MOE_MARK_MAX_SRC; j++) {
                const ggml_tensor * t = k == GGML_CUDA_MOE_MARK_DONE ? nullptr :   // a "done" mark copies nothing
                                        j == 0 ? node : k == GGML_CUDA_MOE_MARK_ROUTER ? node->src[j] : nullptr;
                if (t == nullptr) {
                    break;
                }
                GGML_ASSERT(ggml_is_contiguous(t) && ggml_type_size(t->type) == 4 && t->data != nullptr);
                const int64_t nb = ggml_nbytes(t);
                if (off + nb > e.cap) {
                    GGML_ABORT("%s: layer %d's slot holds %lld bytes, the mark's sources need more (%s, %lld bytes at %lld)",
                            __func__, il, (long long) e.cap, t->name, (long long) nb, (long long) off);
                }
                s.src[s.n_src]      = (const int32_t *) t->data;
                s.n_words[s.n_src]  = (int32_t) (nb / 4);
                s.dst_word[s.n_src] = (int32_t) (off / 4);
                s.n_src++;
                off += (nb + 15) / 16 * 16;
            }
            if (k == GGML_CUDA_MOE_MARK_ROUTER && ggml_cuda_moe_mark_takes_node(node)) {   // source 0 = the view
                s.src[0]   = (const int32_t *) node->src[0]->data;
                s.v0_cont  = (int32_t *) node->data;
                s.v0_ne0   = (int32_t) node->src[0]->ne[0];
                s.v0_nb1   = (int32_t) (node->src[0]->nb[1] / 4);
            }
            if (il == st.skip) {
                continue;   // test only: this layer's producer never runs - its counter check must stop the run
            }
            // With the PDL attribute, so it starts while the ids copy ends (the copy signals at its top)
            ggml_cuda_kernel_launch(k_moe_mark, ggml_cuda_kernel_launch_params(1, MOE_MARK_THREADS, 0, stream),
                                    s, (int32_t *) e.slot, e.expected, e.counter);
            if (capturing) {
                CUDA_CHECK(cudaEventRecordWithFlags(e.ev, stream, cudaEventRecordExternal));
            } else {
#if defined(GGML_USE_HIP)
                CUDA_CHECK(cudaEventRecord(e.tmp, stream));
                CUDA_CHECK(cudaStreamWaitEvent(g_moe_mark_side[e.device], e.tmp, 0));
                CUDA_CHECK(cudaEventRecord(e.ev, g_moe_mark_side[e.device]));
#else
                CUDA_CHECK(cudaEventRecord(e.ev, stream));
#endif
            }
        }
    }
}

// test only: see moe-mark.cuh
#if !defined(GGML_USE_HIP)
static __global__ void k_moe_test_spin(uint64_t ns) {
    uint64_t t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (;;) {
        uint64_t t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (t - t0 >= ns) {
            break;
        }
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA   // __nanosleep needs Volta or newer; older cards just spin
        __nanosleep(1000);
#endif
    }
}

bool ggml_cuda_moe_test_spin(cudaStream_t stream, int64_t us) {
    k_moe_test_spin<<<1, 1, 0, stream>>>((uint64_t) us*1000);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#else // HIP: no %globaltimer register; the test delay does nothing (not exported)
bool ggml_cuda_moe_test_spin(cudaStream_t, int64_t) { return false; }
#endif

#else // MUSA: no producer (the host-side stand-in stays: the register function is not exported)

bool ggml_cuda_moe_test_spin(cudaStream_t, int64_t) { return false; }

bool ggml_cuda_moe_mark_register(const void *, int, int32_t, const ggml_cuda_moe_mark_table *, int, const int32_t * const *, const char ** why) {
    *why = "this GPU backend has no producer";
    return false;
}
bool ggml_cuda_moe_mark_wait(int32_t, int32_t) { return false; }
void ggml_cuda_moe_mark_unregister(const void *) {}
void ggml_cuda_moe_mark_after(const ggml_cgraph *, int, int, cudaStream_t, bool, int) {}
uint32_t ggml_cuda_moe_mark_bits(void) { return 0; }
bool ggml_cuda_moe_mark_sync(void) { return true; }
bool ggml_cuda_moe_mark_takes_node(const ggml_tensor *) { return false; }

#endif

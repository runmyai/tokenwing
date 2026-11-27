// autotuner: measures the parts of this PC that decide TokenWing's settings, without loading a model, and prints one
// JSON line. The streamed experts' CPU matmul reads quantized weights from RAM, so its speed is bound by the memory system:
// a quantized matrix-vector product over more weights than any CPU cache holds runs at 1, 2, ... physical-core threads on a
// quiet CPU, and the reply thread count (-t) is the smallest within 5 % of the best, less the cores the engine's other work
// needs (the SSD readers' measured CPU time, and one for the thread that drives the GPU). Prompts are compute-bound, so their
// count (-tb) is every physical core but the GPU thread's. On CPUs with two kinds of cores (Intel P / E cores, ARM big /
// little, several memory nodes) it measures again on one thread per fast core and names them (--cpu-mask, and --cpu-range
// when they are numbered in one run from 0) only when that is clearly faster.
//
// With --file it reads the model file directly (no file cache) at 1, 2, 4 ... 32 reads in flight, each the size of one
// expert's projection in that model, in three alternating passes, and recommends the SSD reader count
// (--moe-stream-io-threads): the fewest within 5 % of the best. It also lists the graphics cards the engine's backends see
// (discrete ones, and the integrated ones apart) with their memory and the speed they read quantized weights at.
//
//   autotuner [--seconds S] [--file MODEL.gguf] [--self-test]

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#if defined(_M_X64) || defined(_M_IX86)
#include <intrin.h>
#endif
#elif defined(__APPLE__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>
#else
#include <fcntl.h>
#include <sched.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

namespace {

#ifdef _WIN32
static std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    if (n <= 0) return std::wstring(s.begin(), s.end());
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}
#endif

// Calling thread's consumed CPU time (user + kernel) in seconds, for measuring exact `read_cores` of Direct-I/O workers.
static double thread_cpu_seconds() {
#ifdef _WIN32
    FILETIME create_t{}, exit_t{}, kernel_t{}, user_t{};
    if (!GetThreadTimes(GetCurrentThread(), &create_t, &exit_t, &kernel_t, &user_t)) return 0.0;
    const uint64_t k = ((uint64_t) kernel_t.dwHighDateTime << 32) | kernel_t.dwLowDateTime;
    const uint64_t u = ((uint64_t) user_t.dwHighDateTime   << 32) | user_t.dwLowDateTime;
    return (double) (k + u) * 1e-7;
#elif defined(CLOCK_THREAD_CPUTIME_ID)
    struct timespec ts{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) {
        return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
    }
    return 0.0;
#else
    return 0.0;
#endif
}

// Saves and restores the calling thread's OS affinity mask (ggml_threadpool_new pins workers[0] on the caller thread).
struct ThreadAffinityGuard {
#ifdef _WIN32
    GROUP_AFFINITY prev{};
    bool ok = false;
    ThreadAffinityGuard() { ok = GetThreadGroupAffinity(GetCurrentThread(), &prev) != 0; }
    ~ThreadAffinityGuard() { if (ok) SetThreadGroupAffinity(GetCurrentThread(), &prev, nullptr); }
#elif defined(__linux__)
    cpu_set_t prev;
    bool ok = false;
    ThreadAffinityGuard() { CPU_ZERO(&prev); ok = (sched_getaffinity(0, sizeof(prev), &prev) == 0); }
    ~ThreadAffinityGuard() { if (ok) sched_setaffinity(0, sizeof(prev), &prev); }
#else
    ThreadAffinityGuard() = default;
    ~ThreadAffinityGuard() = default;
#endif
};

static double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back((char) c); }
        else if (c >= 0x20) { out.push_back((char) c); }
    }
    return out;
}

// Direct zero-plugin GGUF v2/v3 metadata & tensor-offset reader.
// Works on standard GGUFs AND custom --quant-plugin GGUFs (GGML_TYPE_EXT0..EXT5 / WV-v5) without loading any plugin DLL,
// because exact tensor byte sizes are derived from sorted GGUF data offsets rather than ggml_blck_size().
struct GgufModelInfo {
    size_t   piece_bytes   = 2u << 20;   // median 4 KiB-aligned single expert projection (gate, up, or down)
    uint32_t n_experts     = 0;
    uint32_t n_used        = 0;
    bool     parsed        = false;
};

// The model's header, read in order. On Windows without the file cache: a buffered read leaves the file with a cache section for
// some seconds, and while it is there Windows sends the storage probe's unbuffered reads to the drive one at a time (8 readers:
// 1.3 GB/s instead of 5.0), so the probe right after the header would measure that and pick too few I/O threads.
struct HdrFile {
#ifdef _WIN32
    HANDLE   h = INVALID_HANDLE_VALUE;
    uint8_t * buf = nullptr;   // page-aligned: unbuffered reads need aligned memory, offsets and sizes
    uint64_t size = 0, buf_at = 0, buf_len = 0, pos = 0;
#else
    FILE * f = nullptr;
#endif
};
#ifdef _WIN32
static constexpr size_t HDR_BUF = 1u << 20;
#endif

static HdrFile * hdr_open(const std::string & path, uint64_t * out_size) {
    auto * r = new HdrFile;
#ifdef _WIN32
    const std::wstring wp = utf8_to_wide(path);
    r->h = CreateFileW(wp.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                       FILE_FLAG_NO_BUFFERING, nullptr);
    LARGE_INTEGER sz{};
    r->buf = (uint8_t *) VirtualAlloc(nullptr, HDR_BUF, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (r->h == INVALID_HANDLE_VALUE || !GetFileSizeEx(r->h, &sz) || !r->buf) {
        if (r->buf) VirtualFree(r->buf, 0, MEM_RELEASE);
        if (r->h != INVALID_HANDLE_VALUE) CloseHandle(r->h);
        delete r;
        return nullptr;
    }
    r->size = (uint64_t) sz.QuadPart;
    *out_size = r->size;
#else
    r->f = fopen(path.c_str(), "rb");
    if (!r->f) { delete r; return nullptr; }
    struct stat st{};
    *out_size = (fstat(fileno(r->f), &st) == 0 && st.st_size > 0) ? (uint64_t) st.st_size : 0;
#endif
    return r;
}

static bool hdr_read(HdrFile * r, void * dst, size_t n) {
#ifdef _WIN32
    auto * out = (uint8_t *) dst;
    while (n > 0) {
        if (r->pos < r->buf_at || r->pos >= r->buf_at + r->buf_len) {   // the aligned 1 MiB around pos
            r->buf_at = r->pos / 4096 * 4096;
            OVERLAPPED ov{};
            ov.Offset     = (DWORD) (r->buf_at & 0xffffffffu);
            ov.OffsetHigh = (DWORD) (r->buf_at >> 32);
            DWORD got = 0;
            if (!ReadFile(r->h, r->buf, (DWORD) HDR_BUF, &got, &ov) || r->buf_at + got <= r->pos) return false;
            r->buf_len = got;
        }
        const size_t take = (size_t) std::min<uint64_t>(n, r->buf_at + r->buf_len - r->pos);
        memcpy(out, r->buf + (r->pos - r->buf_at), take);
        out += take; r->pos += take; n -= take;
    }
    return true;
#else
    return fread(dst, 1, n, r->f) == n;
#endif
}

static bool hdr_skip(HdrFile * r, uint64_t n) {
#ifdef _WIN32
    if (n > r->size - std::min(r->pos, r->size)) return false;
    r->pos += n;
    return true;
#else
    return fseeko(r->f, (off_t) n, SEEK_CUR) == 0;
#endif
}

static int64_t hdr_tell(HdrFile * r) {
#ifdef _WIN32
    return (int64_t) r->pos;
#else
    return (int64_t) ftello(r->f);
#endif
}

static void hdr_close(HdrFile * r) {
#ifdef _WIN32
    VirtualFree(r->buf, 0, MEM_RELEASE);
    CloseHandle(r->h);
#else
    fclose(r->f);
#endif
    delete r;
}

static GgufModelInfo inspect_gguf(const std::string & path) {
    GgufModelInfo info;
    if (path.empty()) return info;

    uint64_t file_size = 0;
    HdrFile * f = hdr_open(path, &file_size);
    if (!f || file_size < 64) { if (f) hdr_close(f); return info; }

    auto rd = [&](void * dst, size_t n) -> bool { return hdr_read(f, dst, n); };
    auto skip = [&](uint64_t n) -> bool { return hdr_skip(f, n); };
    auto rd_str = [&](std::string & out) -> bool {
        uint64_t len = 0;
        if (!rd(&len, 8) || len > (1ull << 20)) return false;
        out.resize((size_t) len);
        return len == 0 || rd(out.data(), (size_t) len);
    };
    auto scalar_sz = [](uint32_t t) -> size_t {
        switch (t) {
            case 0: case 1: case 7: return 1;
            case 2: case 3:         return 2;
            case 4: case 5: case 6: return 4;
            case 10: case 11: case 12: return 8;
            default: return 0;
        }
    };
    auto skip_val = [&](auto & self, uint32_t t) -> bool {
        if (const size_t sz = scalar_sz(t)) return skip(sz);
        if (t == 8) { uint64_t l = 0; return rd(&l, 8) && l <= (1ull << 26) && skip(l); }
        if (t == 9) {
            uint32_t et = 0; uint64_t cnt = 0;
            if (!rd(&et, 4) || !rd(&cnt, 8) || cnt > (1ull << 26)) return false;
            if (const size_t esz = scalar_sz(et)) return skip(cnt * esz);
            for (uint64_t i = 0; i < cnt; i++) { if (!self(self, et)) return false; }
            return true;
        }
        return false;
    };

    char magic[4] = {};
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!rd(magic, 4) || memcmp(magic, "GGUF", 4) != 0 || !rd(&version, 4) || version < 2 || version > 3 ||
        !rd(&n_tensors, 8) || !rd(&n_kv, 8) || n_tensors > (1ull << 20) || n_kv > (1ull << 20)) {
        hdr_close(f);
        return info;
    }

    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key;
        uint32_t vtype = 0;
        if (!rd_str(key) || !rd(&vtype, 4)) { hdr_close(f); return info; }
        auto rd_u32_like = [&](uint32_t & dst) -> bool {
            if (vtype == 4 || vtype == 5) return rd(&dst, 4);
            if (vtype == 10 || vtype == 11) { uint64_t v = 0; if (!rd(&v, 8)) return false; dst = (uint32_t) v; return true; }
            return skip_val(skip_val, vtype);
        };
        if (key == "general.alignment") {
            uint32_t a = 32;
            if (!rd_u32_like(a)) { hdr_close(f); return info; }
            if (a >= 4 && (a & (a - 1)) == 0) alignment = a;
        } else if (key.size() > 13 && key.compare(key.size() - 13, 13, ".expert_count") == 0) {
            if (!rd_u32_like(info.n_experts)) { hdr_close(f); return info; }
        } else if (key.size() > 18 && key.compare(key.size() - 18, 18, ".expert_used_count") == 0) {
            if (!rd_u32_like(info.n_used)) { hdr_close(f); return info; }
        } else {
            if (!skip_val(skip_val, vtype)) { hdr_close(f); return info; }
        }
    }

    struct RawTensor { std::string name; uint64_t ne2 = 1; uint64_t offset = 0; bool is_exp = false; };
    std::vector<RawTensor> tensors;
    tensors.reserve((size_t) n_tensors);
    std::vector<uint64_t> offsets;
    offsets.reserve((size_t) n_tensors + 1);

    for (uint64_t i = 0; i < n_tensors; i++) {
        RawTensor rt;
        uint32_t n_dims = 0, ttype = 0;
        uint64_t ne[4] = {1, 1, 1, 1};
        if (!rd_str(rt.name) || !rd(&n_dims, 4) || n_dims == 0 || n_dims > 4) { hdr_close(f); return info; }
        for (uint32_t d = 0; d < n_dims; d++) { if (!rd(&ne[d], 8)) { hdr_close(f); return info; } }
        if (!rd(&ttype, 4) || !rd(&rt.offset, 8)) { hdr_close(f); return info; }
        rt.ne2 = ne[2];
        rt.is_exp = (rt.name.find("_exps.weight") != std::string::npos && rt.ne2 > 1);
        offsets.push_back(rt.offset);
        tensors.push_back(std::move(rt));
    }

    const int64_t hdr_end = hdr_tell(f);
    hdr_close(f);
    if (hdr_end <= 0) return info;

    const uint64_t data_start = (((uint64_t) hdr_end) + alignment - 1) / alignment * alignment;
    if (data_start >= file_size) return info;
    const uint64_t data_bytes = file_size - data_start;

    offsets.push_back(data_bytes);
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());

    std::vector<double> pieces;
    for (const auto & rt : tensors) {
        if (!rt.is_exp) continue;
        auto it = std::upper_bound(offsets.begin(), offsets.end(), rt.offset);
        if (it == offsets.end() || *it <= rt.offset) continue;
        const uint64_t tbytes = *it - rt.offset;
        const size_t p = (size_t) (((tbytes / rt.ne2) + 4095ull) / 4096ull * 4096ull);
        if (p >= 4096 && p <= (64u << 20)) pieces.push_back((double) p);
        if (info.n_experts == 0) info.n_experts = (uint32_t) rt.ne2;
    }

    info.parsed = true;
    if (!pieces.empty()) {
        const size_t med = ((size_t) std::llround(median_of(pieces)) + 4095u) / 4096u * 4096u;
        info.piece_bytes  = std::max<size_t>(4096u, med);
    }
    return info;
}

// Encodes CPU indices (< GGML_MAX_N_THREADS, max 512) as an MSB-first hex mask matching common.cpp:parse_cpu_mask().
static std::string build_cpu_mask_hex(const std::vector<int> & cpus) {
    int max_c = -1;
    std::vector<bool> bits(GGML_MAX_N_THREADS, false);
    for (int c : cpus) {
        if (c >= 0 && c < GGML_MAX_N_THREADS && c < 512) {
            bits[c] = true;
            max_c = std::max(max_c, c);
        }
    }
    if (max_c < 0) return "";
    const int nibbles = (max_c / 4) + 1;
    static const char hex[] = "0123456789abcdef";
    std::string out = "0x";
    out.reserve(2 + nibbles);
    for (int d = nibbles - 1; d >= 0; d--) {
        const int base = d * 4;
        int v = (bits[base + 3] ? 8 : 0) | (bits[base + 2] ? 4 : 0) | (bits[base + 1] ? 2 : 0) | (bits[base] ? 1 : 0);
        out.push_back(hex[v]);
    }
    return out;
}

static std::vector<int> parse_cpulist(const std::string & s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        const auto dash = tok.find('-');
        if (dash == std::string::npos) {
            out.push_back(std::atoi(tok.c_str()));
        } else {
            const int lo = std::atoi(tok.substr(0, dash).c_str());
            const int hi = std::atoi(tok.substr(dash + 1).c_str());
            for (int c = lo; c <= hi && c < 1024; c++) out.push_back(c);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// Reads Linux cgroup v2 (`cpu.max` = "<quota> <period>") CPU ceiling, or 0 when uncapped.
static int parse_cgroup_cpu_cap(const std::string & cgroup_dir) {
    std::ifstream f(cgroup_dir + "/cpu.max");
    std::string sq;
    uint64_t period = 0;
    if (!(f >> sq >> period) || sq.empty() || sq == "max" || period == 0) return 0;
    const uint64_t quota = std::strtoull(sq.c_str(), nullptr, 10);
    return quota > 0 ? std::max(1, (int) ((quota + period - 1) / period)) : 0;
}

struct CpuTopology {
    int              logical_cores  = 1;
    int              physical_cores = 1;
    std::vector<int> phys_primary;      // 1 primary logical CPU per physical core (on memory-attached, allowed CPUs)
    std::vector<int> fast_primary;      // candidate subset (Intel P-core, ARM big-core, or Socket-0 NUMA; kept only if >5% faster)
    std::vector<int> fast_logical;      // all logical CPUs belonging to candidate fast cores
    std::string      fast_range;        // "0-N" when fast_logical is contiguous from 0, else ""
    std::string      fast_mask;         // hex mask of fast_primary (1 per physical core; handles HT, Arrow Lake, ARM, NUMA)
};

static void finalize_topology_masks(CpuTopology & topo, bool allow_masks = true) {
    std::sort(topo.fast_logical.begin(), topo.fast_logical.end());
    topo.fast_logical.erase(std::unique(topo.fast_logical.begin(), topo.fast_logical.end()), topo.fast_logical.end());
    std::sort(topo.fast_primary.begin(), topo.fast_primary.end());
    topo.fast_primary.erase(std::unique(topo.fast_primary.begin(), topo.fast_primary.end()), topo.fast_primary.end());
    topo.fast_range.clear();
    topo.fast_mask.clear();
    if (!allow_masks || topo.fast_primary.empty() || (int) topo.fast_primary.size() >= topo.physical_cores) {
        topo.fast_primary.clear();
        topo.fast_logical.clear();
        return;
    }
    if (!topo.fast_logical.empty() && topo.fast_logical.front() == 0 &&
        (int) topo.fast_logical.size() == topo.fast_logical.back() + 1) {
        topo.fast_range = "0-" + std::to_string(topo.fast_logical.back());
    }
    topo.fast_mask = build_cpu_mask_hex(topo.fast_primary);
}

// Platform-independent sysfs + cgroup topology parser (used live on Linux and fixture-tested in `--self-test` everywhere).
static void parse_sysfs_cpu_topology(const std::string & sys_devices, const std::string & cgroup_dir,
                                     const std::vector<int> & allowed_cpus, CpuTopology & topo) {
    std::vector<int> mem_cpus, node0_cpus;
    int mem_node_count = 0;
    {
        std::ifstream fmem(sys_devices + "/system/node/has_memory");
        std::string smem;
        if (std::getline(fmem, smem) && !smem.empty()) {
            const auto nodes = parse_cpulist(smem);
            mem_node_count = (int) nodes.size();
            for (size_t idx = 0; idx < nodes.size(); idx++) {
                std::ifstream fn(sys_devices + "/system/node/node" + std::to_string(nodes[idx]) + "/cpulist");
                std::string sn;
                if (std::getline(fn, sn)) {
                    for (int c : parse_cpulist(sn)) {
                        mem_cpus.push_back(c);
                        if (idx == 0) node0_cpus.push_back(c);
                    }
                }
            }
            std::sort(mem_cpus.begin(), mem_cpus.end());
            mem_cpus.erase(std::unique(mem_cpus.begin(), mem_cpus.end()), mem_cpus.end());
            std::sort(node0_cpus.begin(), node0_cpus.end());
        }
    }

    std::ifstream fon(sys_devices + "/system/cpu/online");
    std::string son;
    std::getline(fon, son);
    std::vector<int> online = !son.empty() ? parse_cpulist(son) : std::vector<int>{};
    if (!allowed_cpus.empty()) {
        std::vector<int> inter;
        for (int c : online) {
            if (std::binary_search(allowed_cpus.begin(), allowed_cpus.end(), c)) inter.push_back(c);
        }
        online.swap(inter);
    }
    if (!online.empty()) topo.logical_cores = (int) online.size();

    for (int c : online) {
        if (!mem_cpus.empty() && !std::binary_search(mem_cpus.begin(), mem_cpus.end(), c)) continue;
        std::ifstream fsib(sys_devices + "/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        std::string ssib;
        if (std::getline(fsib, ssib)) {
            int first_allowed = -1;
            for (int s : parse_cpulist(ssib)) {
                if (std::binary_search(online.begin(), online.end(), s)) { first_allowed = s; break; }
            }
            if (first_allowed == c && c < GGML_MAX_N_THREADS) topo.phys_primary.push_back(c);
        } else if (c < GGML_MAX_N_THREADS) {
            topo.phys_primary.push_back(c);
        }
    }
    if (!topo.phys_primary.empty()) topo.physical_cores = (int) topo.phys_primary.size();

    const int cg_cap = parse_cgroup_cpu_cap(cgroup_dir);
    if (cg_cap > 0) {
        topo.physical_cores = std::min(topo.physical_cores, cg_cap);
        topo.logical_cores  = std::min(topo.logical_cores, cg_cap);
        if ((int) topo.phys_primary.size() > cg_cap) topo.phys_primary.resize((size_t) cg_cap);
    }

    // Candidate 1: Intel x86 hybrid (`cpu_core/cpus` + `cpu_atom/cpus`).
    std::ifstream fp(sys_devices + "/cpu_core/cpus"), fe(sys_devices + "/cpu_atom/cpus");
    std::string sp, se;
    std::getline(fp, sp); std::getline(fe, se);
    if (!sp.empty() && !se.empty()) {
        for (int c : parse_cpulist(sp)) {
            if (c < GGML_MAX_N_THREADS && std::binary_search(online.begin(), online.end(), c)) {
                topo.fast_logical.push_back(c);
                if (std::binary_search(topo.phys_primary.begin(), topo.phys_primary.end(), c)) {
                    topo.fast_primary.push_back(c);
                }
            }
        }
    } else {
        // Candidate 2: ARM64 big.LITTLE / DynamIQ (`cpu_capacity`, e.g. DGX Spark GB10 X925+A725, Snapdragon, RK3588).
        int lo_cap = 1 << 30, hi_cap = 0;
        std::vector<std::pair<int, int>> caps;
        for (int c : topo.phys_primary) {
            std::ifstream fc(sys_devices + "/system/cpu/cpu" + std::to_string(c) + "/cpu_capacity");
            int cap = 0;
            if (fc >> cap && cap > 0) {
                lo_cap = std::min(lo_cap, cap);
                hi_cap = std::max(hi_cap, cap);
                caps.push_back({ c, cap });
            }
        }
        if (hi_cap > 0 && lo_cap < hi_cap) {
            for (const auto & kv : caps) {
                if (kv.second == hi_cap) {
                    topo.fast_primary.push_back(kv.first);
                    topo.fast_logical.push_back(kv.first);
                }
            }
        } else if (mem_node_count > 1 && !node0_cpus.empty()) {
            // Candidate 3: Multi-socket NUMA (Socket 0 local cores vs cross-socket; kept only if measured >5% faster).
            for (int c : topo.phys_primary) {
                if (std::binary_search(node0_cpus.begin(), node0_cpus.end(), c)) {
                    topo.fast_primary.push_back(c);
                }
            }
        }
    }
    finalize_topology_masks(topo, true);
}

static CpuTopology discover_cpu_topology() {
    CpuTopology topo;
    topo.physical_cores = std::max(1, (int) common_cpu_get_num_physical_cores());
    topo.logical_cores  = std::max(topo.physical_cores, (int) std::thread::hardware_concurrency());

#ifdef _WIN32
    topo.logical_cores = std::max<int>(topo.logical_cores, (int) GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
    const WORD n_groups = GetActiveProcessorGroupCount();
    DWORD_PTR proc_mask = 0, sys_mask = 0;
    const bool has_proc_mask = (n_groups <= 1) && (GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask) != 0) && proc_mask != 0;

    bool check_eff_class = false;
#if defined(_M_X64) || defined(_M_IX86)
    int id[4] = {};
    __cpuid(id, 0);
    char vendor[13] = {};
    memcpy(vendor, &id[1], 4); memcpy(vendor + 4, &id[3], 4); memcpy(vendor + 8, &id[2], 4);
    check_eff_class = (strcmp(vendor, "GenuineIntel") == 0);   // AMD Zen 5c carries EfficiencyClass < Zen 5
#elif defined(_M_ARM64)
    check_eff_class = true;                                    // Snapdragon / Windows-on-ARM big.LITTLE
#endif

    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len > 0) {
        std::vector<char> buf(len);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
            int lo_cls = 255, hi_cls = 0, phys_total = 0;
            for (DWORD off = 0; off < len;) {
                auto * p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
                bool core_allowed = false;
                for (WORD g = 0; g < p->Processor.GroupCount; g++) {
                    const WORD grp = p->Processor.GroupMask[g].Group;
                    const KAFFINITY m = p->Processor.GroupMask[g].Mask & (has_proc_mask && grp == 0 ? (KAFFINITY) proc_mask : ~(KAFFINITY) 0);
                    for (int b = 0; b < (int) (8 * sizeof(KAFFINITY)); b++) {
                        if (m & ((KAFFINITY) 1 << b)) {
                            const int cpu = (int) grp * 64 + b;
                            if (!core_allowed && cpu < GGML_MAX_N_THREADS) topo.phys_primary.push_back(cpu);
                            core_allowed = true;
                            break;
                        }
                    }
                }
                if (core_allowed) {
                    lo_cls = std::min<int>(lo_cls, p->Processor.EfficiencyClass);
                    hi_cls = std::max<int>(hi_cls, p->Processor.EfficiencyClass);
                    phys_total++;
                }
                off += p->Size;
            }
            if (phys_total > 0) topo.physical_cores = phys_total;
            if (has_proc_mask) {
                int allowed_log = 0;
                for (int b = 0; b < (int) (8 * sizeof(DWORD_PTR)); b++) { if (proc_mask & ((DWORD_PTR) 1 << b)) allowed_log++; }
                if (allowed_log > 0) topo.logical_cores = allowed_log;
            }
            if (check_eff_class && lo_cls < hi_cls) {
                for (DWORD off = 0; off < len;) {
                    auto * p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
                    if (p->Processor.EfficiencyClass == hi_cls) {
                        bool got_primary = false;
                        for (WORD g = 0; g < p->Processor.GroupCount; g++) {
                            const WORD grp = p->Processor.GroupMask[g].Group;
                            const KAFFINITY m = p->Processor.GroupMask[g].Mask & (has_proc_mask && grp == 0 ? (KAFFINITY) proc_mask : ~(KAFFINITY) 0);
                            for (int b = 0; b < (int) (8 * sizeof(KAFFINITY)); b++) {
                                if (m & ((KAFFINITY) 1 << b)) {
                                    const int cpu = (int) grp * 64 + b;
                                    if (cpu < GGML_MAX_N_THREADS) {
                                        topo.fast_logical.push_back(cpu);
                                        if (!got_primary) { topo.fast_primary.push_back(cpu); got_primary = true; }
                                    }
                                }
                            }
                        }
                    }
                    off += p->Size;
                }
            }
        }
    }
    // ggml on Windows uses single-group 64-bit SetThreadAffinityMask; never emit --cpu-mask on >64-CPU multi-group hosts.
    finalize_topology_masks(topo, n_groups <= 1 && topo.logical_cores <= 64);
#elif defined(__linux__)
    std::vector<int> allowed;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int c = 0; c < CPU_SETSIZE; c++) { if (CPU_ISSET(c, &set)) allowed.push_back(c); }
    }
    parse_sysfs_cpu_topology("/sys/devices", "/sys/fs/cgroup", allowed, topo);
#elif defined(__APPLE__)
    int pcpu = 0;
    size_t sz = sizeof(int);
    if (sysctlbyname("hw.physicalcpu", &pcpu, &sz, nullptr, 0) == 0 && pcpu > 0) topo.physical_cores = pcpu;
    finalize_topology_masks(topo, false);   // macOS QoS schedules P-cores automatically without explicit affinity
#else
    finalize_topology_masks(topo, false);
#endif
    return topo;
}

struct StorageProbe {
    bool   readable       = false;
    int    io_threads     = 8;
    double rd_gbps[6]     = {};
    double rd_spread[6]   = {};
    double io_cpu_cores   = 0.0;
    double latency_us     = 0.0;   // lone single-piece read latency at QD=1
};

// Synchronized single-window read probe across `qd` concurrent Direct-I/O readers.
// Every worker opens its handle and aligned buffer BEFORE signaling ready, then starts & stops on one shared deadline,
// recording both bytes transferred and OS thread CPU seconds (`read_cores`).
static bool probe_qd_once(const std::string & path, size_t blk, int qd, double seconds, uint64_t pass_seed,
                          double * out_gbps, double * out_cores) {
    struct WorkerInit {
        bool ok = false;
        uint64_t bytes = 0;
        double cpu_s = 0.0;
    };
    std::vector<WorkerInit> state(qd);
    std::mutex mtx;
    std::condition_variable cv_ready, cv_go;
    int ready_cnt = 0;
    bool go = false, abort_run = false;
    std::chrono::steady_clock::time_point deadline;

    std::vector<std::thread> workers;
    workers.reserve(qd);
    for (int i = 0; i < qd; i++) {
        workers.emplace_back([&, i] {
#ifdef _WIN32
            const std::wstring wp = utf8_to_wide(path);
            HANDLE h = CreateFileW(wp.c_str(), GENERIC_READ,
                                   FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                   nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
            LARGE_INTEGER fsz{};
            const bool h_ok = (h != INVALID_HANDLE_VALUE) && GetFileSizeEx(h, &fsz) && (uint64_t) fsz.QuadPart > 2 * blk;
            const uint64_t size = h_ok ? (uint64_t) fsz.QuadPart : 0;
            void * buf = h_ok ? VirtualAlloc(nullptr, blk, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) : nullptr;
            const bool init_ok = h_ok && buf != nullptr;
#else
            int fd = -1;
            bool drop_cache = false;
#if defined(O_DIRECT)
            fd = open(path.c_str(), O_RDONLY | O_DIRECT);
#endif
            if (fd < 0) {
                fd = open(path.c_str(), O_RDONLY);
                drop_cache = (fd >= 0);
#ifdef __APPLE__
                if (fd >= 0) { fcntl(fd, F_NOCACHE, 1); drop_cache = false; }
#endif
            }
            struct stat st{};
            const bool f_ok = (fd >= 0) && (fstat(fd, &st) == 0) && (uint64_t) st.st_size > 2 * blk;
            const uint64_t size = f_ok ? (uint64_t) st.st_size : 0;
            void * buf = nullptr;
            const bool init_ok = f_ok && (posix_memalign(&buf, 4096, blk) == 0) && buf != nullptr;
#endif
            {
                std::lock_guard<std::mutex> lk(mtx);
                state[i].ok = init_ok;
                if (!init_ok) abort_run = true;
                if (++ready_cnt == qd) cv_ready.notify_one();
            }
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_go.wait(lk, [&] { return go; });
            }

            if (init_ok && !abort_run) {
                uint64_t rng = 0x9e3779b97f4a7c15ull ^ (pass_seed + (uint64_t) (i + 1) * 0xbf58476d1ce4e5b9ull);
                const uint64_t n_blocks = (size - blk) / 4096ull;
                const double cpu0 = thread_cpu_seconds();
                uint64_t total_b = 0;
                bool read_ok = true;
                while (std::chrono::steady_clock::now() < deadline) {
                    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                    const uint64_t off = (rng % (n_blocks ? n_blocks : 1)) * 4096ull;
#ifdef _WIN32
                    OVERLAPPED ov{};
                    ov.Offset     = (DWORD) (off & 0xffffffffu);
                    ov.OffsetHigh = (DWORD) (off >> 32);
                    DWORD got = 0;
                    if (!ReadFile(h, buf, (DWORD) blk, &got, &ov) || got == 0) { read_ok = false; break; }
                    total_b += got;
#else
                    const ssize_t got = pread(fd, buf, blk, (off_t) off);
                    if (got <= 0) { read_ok = false; break; }
#if defined(POSIX_FADV_DONTNEED) && !defined(__APPLE__)
                    if (drop_cache) (void) posix_fadvise(fd, (off_t) off, got, POSIX_FADV_DONTNEED);
#endif
                    total_b += (uint64_t) got;
#endif
                }
                state[i].cpu_s = std::max(0.0, thread_cpu_seconds() - cpu0);
                state[i].bytes = total_b;
                state[i].ok    = read_ok && total_b > 0;
            }

#ifdef _WIN32
            if (buf) VirtualFree(buf, 0, MEM_RELEASE);
            if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
#else
            if (buf) free(buf);
            if (fd >= 0) close(fd);
#endif
        });
    }

    {
        std::unique_lock<std::mutex> lk(mtx);
        cv_ready.wait(lk, [&] { return ready_cnt == qd; });
    }
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lk(mtx);
        deadline = t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
        go = true;
    }
    cv_go.notify_all();
    for (auto & w : workers) w.join();
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (abort_run || wall <= 0.0) return false;

    uint64_t sum_bytes = 0;
    double   sum_cpu   = 0.0;
    for (const auto & s : state) {
        if (!s.ok) return false;
        sum_bytes += s.bytes;
        sum_cpu   += s.cpu_s;
    }
    *out_gbps  = (double) sum_bytes / wall / 1e9;
    *out_cores = sum_cpu / wall;
    return true;
}

// 3 alternating forward/reverse passes (eliminates monotonic SLC/thermal drift & single-pass 30% knee noise).
// On slow media (< 150 MB/s at QD=1: spinning HDDs, USB thumb drives, throttled shares), caps probing at QD <= 4
// so 32-thread synchronous reads never stall for 10-20s or thrash disk heads.
static StorageProbe probe_storage(const std::string & path, size_t piece, int phys_cores) {
    StorageProbe res;
    if (path.empty()) return res;
    static constexpr int qds[6] = { 1, 2, 4, 8, 16, 32 };
    static constexpr int passes = 3;
    std::vector<double> rates[6], cores[6];

    // Check QD=1 first to detect slow HDD / USB media (< 0.15 GB/s).
    double g0 = 0.0, c0 = 0.0;
    if (!probe_qd_once(path, piece, 1, 0.07, 11ull, &g0, &c0)) return res;
    const int n_qds = (g0 < 0.15) ? 3 : 6;   // QD={1,2,4} on slow media, full QD={1..32} on SSD/NVMe

    for (int p = 0; p < passes; p++) {
        for (int step = 0; step < n_qds; step++) {
            const int j = (p % 2 == 0) ? step : (n_qds - 1 - step);
            if (p == 0 && j == 0) {
                rates[0].push_back(g0);
                cores[0].push_back(c0);
                continue;
            }
            double gbps = 0.0, cpu_c = 0.0;
            if (!probe_qd_once(path, piece, qds[j], 0.07, (uint64_t) (p * 97 + j * 13), &gbps, &cpu_c)) {
                return res;
            }
            rates[j].push_back(gbps);
            cores[j].push_back(cpu_c);
        }
    }

    int top_j = 0;
    for (int j = 0; j < n_qds; j++) {
        res.rd_gbps[j] = median_of(rates[j]);
        const auto mm  = std::minmax_element(rates[j].begin(), rates[j].end());
        res.rd_spread[j] = 0.5 * (*mm.second - *mm.first);
        if (res.rd_gbps[j] > res.rd_gbps[top_j]) top_j = j;
    }
    for (int j = n_qds; j < 6; j++) {
        res.rd_gbps[j]   = res.rd_gbps[n_qds - 1];
        res.rd_spread[j] = res.rd_spread[n_qds - 1];
    }
    if (res.rd_gbps[top_j] <= 0.0) return res;

    res.readable = true;
    res.latency_us = res.rd_gbps[0] > 0.0 ? ((double) piece / (res.rd_gbps[0] * 1e9)) * 1e6 : 0.0;

    const double top_rate = res.rd_gbps[top_j];
    const int min_readers = (phys_cores <= 4 || n_qds == 3) ? 2 : 4;
    const int max_readers = (n_qds == 3) ? 4 : 16;
    int chosen_j = top_j;
    for (int j = 0; j < n_qds; j++) {
        const double tol = std::min(res.rd_spread[j] + res.rd_spread[top_j], 0.08 * top_rate);
        if (res.rd_gbps[j] + tol >= 0.95 * top_rate) {
            chosen_j = j;
            break;
        }
    }
    res.io_threads = std::min(max_readers, std::max(min_readers, qds[chosen_j]));
    int io_idx = 0;
    for (int j = 0; j < n_qds; j++) { if (qds[j] <= res.io_threads) io_idx = j; }
    res.io_cpu_cores = median_of(cores[io_idx]);
    return res;
}

// Populates Q4_K blocks (144 bytes each: fp16 d, fp16 dmin, 12B scales, 128B nibbles) with deterministic non-zero bits
// so OS zero-page / Windows Memory Compression cannot compress the buffer, while keeping all FP16 scales normal & finite.
static void fill_q4k_noncompressible(void * dst, size_t nbytes, uint64_t seed) {
    auto * p = static_cast<uint8_t *>(dst);
    uint64_t s = seed ? seed : 0x2545f4914f6cdd1dull;
    const size_t n_blocks = nbytes / 144;
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t * blk = p + b * 144;
        const uint16_t d    = 0x2c00u;   // 0.0625 in IEEE-754 FP16 (exact normal, zero NaN/denormal risk)
        const uint16_t dmin = 0x2800u;   // 0.03125 in IEEE-754 FP16
        memcpy(blk + 0, &d, 2);
        memcpy(blk + 2, &dmin, 2);
        for (size_t off = 4; off < 144; off += 8) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            memcpy(blk + off, &s, std::min<size_t>(8, 144 - off));
        }
    }
}

// GB/s one GPU reads quantized weights at: four 4096 x 28672 Q4_K matrices (~66 MB each, more than any GPU cache) times a
// vector on that GPU, the median of the runs in `seconds`. 0 when the GPU cannot take them. The engine fills the fastest
// memory first, so the caller ranks the cards by this
static double gpu_gbps(ggml_backend_dev_t d, double seconds) {
    const int64_t K = 4096, N = 28672, M = 4;
    ggml_init_params ip = { /*.mem_size =*/ 16*ggml_tensor_overhead() + ggml_graph_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    size_t bytes = 0;
    for (int i = 0; i < M; i++) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N);
        bytes += ggml_nbytes(w);
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, w, x));
    }
    double gbps = 0;
    ggml_backend_t be = ggml_backend_dev_init(d, nullptr);
    ggml_backend_buffer_t buf = be ? ggml_backend_alloc_ctx_tensors(ctx, be) : nullptr;
    if (buf) {
        ggml_backend_buffer_clear(buf, 0);
        ggml_backend_graph_compute(be, gf);   // warm-up
        std::vector<double> runs;
        const auto t0 = std::chrono::steady_clock::now();
        do {
            const auto a = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(be, gf);
            ggml_backend_synchronize(be);
            runs.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds || runs.size() < 3);
        gbps = bytes/median_of(std::move(runs))/1e9;
        ggml_backend_buffer_free(buf);
    }
    if (be) { ggml_backend_free(be); }
    ggml_free(ctx);
    return gbps;
}

struct TuneGpuInfo {
    std::string name;
    size_t      mib  = 0;
    double      gbps = 0.0;
};

struct TuneRec {
    std::string              vram_device;
    std::vector<std::string> args;
};

static TuneRec build_recommended_args(
        const std::vector<TuneGpuInfo> & gpus,
        const std::vector<TuneGpuInfo> & igpus,
        int threads,
        int threads_batch,
        const std::string & fast_mask,
        bool has_side_load,
        int io_threads) {
    TuneRec rec;
    rec.args.push_back("--moe-stream");
    rec.args.push_back("model");
    rec.args.push_back("-t");
    rec.args.push_back(std::to_string(threads));
    if (threads_batch > threads) {
        rec.args.push_back("-tb");
        rec.args.push_back(std::to_string(threads_batch));
    }
    if (!fast_mask.empty()) {
        rec.args.push_back("--cpu-mask");
        rec.args.push_back(fast_mask);
        rec.args.push_back("--cpu-strict");
        rec.args.push_back("1");
    }
    if (has_side_load && io_threads > 0) {
        rec.args.push_back("--moe-stream-io-threads");
        rec.args.push_back(std::to_string(io_threads));
    }
    if (!gpus.empty()) {
        size_t best_idx = 0;
        for (size_t i = 1; i < gpus.size(); i++) {
            const long long g_best = std::llround(gpus[best_idx].gbps);
            const long long g_cur  = std::llround(gpus[i].gbps);
            if (g_cur > g_best || (g_cur == g_best && gpus[i].mib > gpus[best_idx].mib)) {
                best_idx = i;
            }
        }
        rec.vram_device = gpus[best_idx].name;
        if (best_idx != 0 && !rec.vram_device.empty()) {
            rec.args.push_back("--moe-stream-vram-device");
            rec.args.push_back(rec.vram_device);
        }
    } else {
        for (const auto & ig : igpus) {
            if (ig.mib >= 24576) {
                rec.vram_device = ig.name;
                rec.args.push_back("--moe-stream-uma");
                rec.args.push_back("1");
                break;
            }
        }
    }
    return rec;
}

} // namespace

static int run_self_test() {
    // 1. Verify MSB-first hex mask encoder against TokenWing's common.cpp:parse_cpu_mask() across HT & Arrow Lake layouts.
    for (const auto & vec : std::vector<std::vector<int>>{
            {0, 2, 4, 6, 8, 10},                 // 6 HT P-cores (i7-12800H primary threads -> 0x555)
            {0, 1, 10, 11, 12, 13, 14, 15},      // Intel Arrow Lake split P-cores (0xfc03)
            {0, 63, 64, 127},                    // Multi-group >64 logical CPUs
        }) {
        const std::string hex = build_cpu_mask_hex(vec);
        bool mask[GGML_MAX_N_THREADS] = {};
        if (hex.empty() || !parse_cpu_mask(hex, mask)) {
            fprintf(stderr, "autotuner self-test FAIL: parse_cpu_mask rejected %s\n", hex.c_str());
            return 1;
        }
        for (int c = 0; c < GGML_MAX_N_THREADS; c++) {
            const bool want = std::find(vec.begin(), vec.end(), c) != vec.end();
            if (mask[c] != want) {
                fprintf(stderr, "autotuner self-test FAIL: mask mismatch at CPU %d for %s\n", c, hex.c_str());
                return 1;
            }
        }
    }

    // 3. Verify zero-plugin GGUF v3 offset reader on a synthetic EXT0 (type 112) expert tensor header.
    const std::string tmp = "autotuner_selftest_ext0.gguf";
    {
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) return 1;
        auto wr = [&](const void * p, size_t n) { fwrite(p, 1, n, f); };
        auto wr_u32 = [&](uint32_t v) { wr(&v, 4); };
        auto wr_u64 = [&](uint64_t v) { wr(&v, 8); };
        auto wr_str = [&](const char * s) { uint64_t l = strlen(s); wr_u64(l); wr(s, (size_t) l); };
        wr("GGUF", 4); wr_u32(3); wr_u64(2); wr_u64(2);
        wr_str("qwen4exp.expert_count");      wr_u32(4); wr_u32(512);
        wr_str("qwen4exp.expert_used_count"); wr_u32(4); wr_u32(10);
        // Tensor 0: blk.0.ffn_gate_exps.weight, type=112 (GGML_TYPE_EXT0), 512 experts * 180 KiB = 94,371,840 B at offset 0
        wr_str("blk.0.ffn_gate_exps.weight"); wr_u32(3); wr_u64(2048); wr_u64(768); wr_u64(512); wr_u32(112); wr_u64(0);
        // Tensor 1: output_norm.weight, F32, 2048 B at offset 94,371,840
        wr_str("output_norm.weight"); wr_u32(1); wr_u64(512); wr_u32(0); wr_u64(94371840ull);
        const long pos = ftell(f);
        const long pad = ((pos + 31) / 32) * 32 - pos;
        std::vector<char> zeros((size_t) pad + 4096, 0);
        wr(zeros.data(), zeros.size());
        // Sparse/seek extend to total data size 94,371,840 + 2048 so we write almost zero disk bytes.
#ifdef _WIN32
        _fseeki64(f, ((pos + 31) / 32) * 32 + 94371840ll + 2047ll, SEEK_SET);
#else
        fseeko(f, (off_t) (((pos + 31) / 32) * 32 + 94371840ll + 2047ll), SEEK_SET);
#endif
        char z = 0; wr(&z, 1);
        fclose(f);
    }
    const GgufModelInfo gi = inspect_gguf(tmp);
    std::remove(tmp.c_str());
    if (!gi.parsed || gi.n_experts != 512 || gi.n_used != 10 || gi.piece_bytes != 180u * 1024u) {
        fprintf(stderr, "autotuner self-test FAIL: GGUF EXT0 piece=%zu exp=%u used=%u\n", gi.piece_bytes, gi.n_experts, gi.n_used);
        return 1;
    }

    // 4. Verify sysfs + cgroup v2 topology & RAM parsers across Docker container, Arrow Lake, ARM64 GB10, and NUMA fixtures.
    namespace fs = std::filesystem;
    const fs::path fx = "autotuner_selftest_sysfs";
    auto write_txt = [](const fs::path & p, const std::string & s) {
        fs::create_directories(p.parent_path());
        std::ofstream(p) << s << "\n";
    };
    {
        // Fixture A: 8-CPU / 15 GiB cgroup container on a 24-CPU Arrow Lake host (split P-cores 0-1,10-15; E-cores 2-9,16-23)
        fs::remove_all(fx);
        write_txt(fx / "sys/system/cpu/online", "0-23");
        write_txt(fx / "sys/cpu_core/cpus", "0-1,10-15");
        write_txt(fx / "sys/cpu_atom/cpus", "2-9,16-23");
        for (int c = 0; c < 24; c++) write_txt(fx / ("sys/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"), std::to_string(c));
        write_txt(fx / "cg/cpu.max", "max 100000");

        CpuTopology t_arl;
        parse_sysfs_cpu_topology((fx / "sys").string(), (fx / "cg").string(), {}, t_arl);
        if (t_arl.physical_cores != 24 || !t_arl.fast_range.empty() || t_arl.fast_mask != "0xfc03") {
            fprintf(stderr, "autotuner self-test FAIL: ArrowLake/cgroup phys=%d range='%s' mask='%s'\n",
                    t_arl.physical_cores, t_arl.fast_range.c_str(), t_arl.fast_mask.c_str());
            fs::remove_all(fx);
            return 1;
        }
        // Apply container quota cpu.max = 6 CPUs on the same host
        write_txt(fx / "cg/cpu.max", "600000 100000");
        CpuTopology t_cg;
        parse_sysfs_cpu_topology((fx / "sys").string(), (fx / "cg").string(), {0, 1, 2, 3, 4, 5, 6, 7}, t_cg);
        if (t_cg.physical_cores != 6) {
            fprintf(stderr, "autotuner self-test FAIL: cgroup cpu.max cap=%d\n", t_cg.physical_cores);
            fs::remove_all(fx);
            return 1;
        }

        // Fixture B: ARM64 big.LITTLE (4x Cortex-X925 @ 1024 + 4x Cortex-A725 @ 512)
        fs::remove_all(fx);
        write_txt(fx / "sys/system/cpu/online", "0-7");
        for (int c = 0; c < 8; c++) {
            write_txt(fx / ("sys/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"), std::to_string(c));
            write_txt(fx / ("sys/system/cpu/cpu" + std::to_string(c) + "/cpu_capacity"), c < 4 ? "1024" : "512");
        }
        CpuTopology t_arm;
        parse_sysfs_cpu_topology((fx / "sys").string(), (fx / "cg").string(), {}, t_arm);
        if (t_arm.physical_cores != 8 || t_arm.fast_range != "0-3" || t_arm.fast_mask != "0xf") {
            fprintf(stderr, "autotuner self-test FAIL: ARM64 big.LITTLE range='%s' mask='%s'\n",
                    t_arm.fast_range.c_str(), t_arm.fast_mask.c_str());
            fs::remove_all(fx);
            return 1;
        }

        // Fixture C: Multi-socket + memoryless NUMA (nodes 0 & 2 have memory; node 1 is memoryless compute die)
        fs::remove_all(fx);
        write_txt(fx / "sys/system/cpu/online", "0-11");
        write_txt(fx / "sys/system/node/has_memory", "0,2");
        write_txt(fx / "sys/system/node/node0/cpulist", "0-3");
        write_txt(fx / "sys/system/node/node1/cpulist", "4-7");
        write_txt(fx / "sys/system/node/node2/cpulist", "8-11");
        for (int c = 0; c < 12; c++) write_txt(fx / ("sys/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"), std::to_string(c));
        CpuTopology t_numa;
        parse_sysfs_cpu_topology((fx / "sys").string(), (fx / "cg").string(), {}, t_numa);
        if (t_numa.physical_cores != 8 || t_numa.fast_mask != "0xf") {
            fprintf(stderr, "autotuner self-test FAIL: NUMA phys=%d mask='%s'\n", t_numa.physical_cores, t_numa.fast_mask.c_str());
            fs::remove_all(fx);
            return 1;
        }
        fs::remove_all(fx);
    }

    // 5. Verify recommended_vram_device + recommended_args builder across multi-GPU, hybrid P-core, and UMA topologies.
    {
        const auto contains_pair = [](const std::vector<std::string> & a, const std::string & k, const std::string & v) {
            for (size_t i = 0; i + 1 < a.size(); i++) {
                if (a[i] == k && a[i + 1] == v) return true;
            }
            return false;
        };

        const TuneRec r_mgpu = build_recommended_args(
            { {"CUDA0", 12288, 350.0}, {"CUDA1", 24576, 820.0} },
            {},
            6, 12, "0xfc03", true, 8);
        if (r_mgpu.vram_device != "CUDA1" ||
                !contains_pair(r_mgpu.args, "--moe-stream", "model") ||
                !contains_pair(r_mgpu.args, "-t", "6") ||
                !contains_pair(r_mgpu.args, "-tb", "12") ||
                !contains_pair(r_mgpu.args, "--cpu-mask", "0xfc03") ||
                !contains_pair(r_mgpu.args, "--moe-stream-io-threads", "8") ||
                !contains_pair(r_mgpu.args, "--moe-stream-vram-device", "CUDA1")) {
            fprintf(stderr, "autotuner self-test FAIL: mgpu recommended_args mismatch (dev=%s)\n", r_mgpu.vram_device.c_str());
            return 1;
        }

        const TuneRec r_uma = build_recommended_args(
            {},
            { {"Vulkan0", 65536, 210.0} },
            8, 8, "", false, 0);
        if (r_uma.vram_device != "Vulkan0" ||
                !contains_pair(r_uma.args, "--moe-stream-uma", "1") ||
                std::find(r_uma.args.begin(), r_uma.args.end(), "-tb") != r_uma.args.end()) {
            fprintf(stderr, "autotuner self-test FAIL: UMA recommended_args mismatch (dev=%s)\n", r_uma.vram_device.c_str());
            return 1;
        }
    }

    printf("{\"self_test\": \"PASS\", \"ext0_piece_kib\": %zu, \"fixtures\": 5}\n", gi.piece_bytes >> 10);
    return 0;
}

int main(int argc, char ** argv) {
    common_init();
    // Silence all backend/driver diagnostic logs so stdout contains strictly the single JSON line.
    ggml_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);

    double seconds = 0.14;
    std::string file;
    bool self_test = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seconds") && i + 1 < argc) {
            seconds = std::max(0.03, atof(argv[++i]));
        } else if (!strcmp(argv[i], "--file") && i + 1 < argc) {
            file = argv[++i];
        } else if (!strcmp(argv[i], "--self-test")) {
            self_test = true;
        }
    }
    if (self_test) return run_self_test();

    const CpuTopology topo = discover_cpu_topology();
    const int phys = topo.physical_cores;

    // Load all compiled/dynamic backends once up front (supports both static builds and GGML_BACKEND_DL=ON).
    ggml_backend_load_all();

    // the cards the engine's backends see: discrete GPUs, and apart from them the integrated ones (they share the system's RAM;
    // the memory they report is the part of it they can use - a DGX Spark's GB10 and a Strix Halo run the models far faster on it)
    std::vector<std::pair<ggml_backend_dev_t, size_t>> gpu_devs, igpu_devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto dtype = ggml_backend_dev_type(d);
        if (dtype != GGML_BACKEND_DEVICE_TYPE_GPU && dtype != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(d, &free_b, &total_b);
        (dtype == GGML_BACKEND_DEVICE_TYPE_GPU ? gpu_devs : igpu_devs).push_back({ d, total_b >> 20 });
    }
    // a thread drives the GPU whenever the engine runs on one: a discrete card, or an integrated GPU large enough to run the
    // model (24 GiB, the caller's rule)
    bool runs_on_gpu = !gpu_devs.empty();
    for (const auto & g : igpu_devs) runs_on_gpu = runs_on_gpu || g.second >= 24576;

    // 1. Metadata + synchronized multi-pass Direct-I/O storage curve (measures both knee io_threads and OS io_cpu_cores).
    const GgufModelInfo gm = inspect_gguf(file);
    const size_t piece = gm.piece_bytes;
    const StorageProbe sp = probe_storage(file, piece, phys);
    const int io = sp.io_threads;

    // 2. Initialize CPU backend via registry proc addresses (links cleanly under both GGML_BACKEND_DL=ON and OFF).
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t be = cpu_dev ? ggml_backend_dev_init(cpu_dev, nullptr) : nullptr;
    if (!be) {
        fprintf(stderr, "autotuner: cannot initialize CPU backend\n");
        return 1;
    }
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    using set_n_threads_fn   = void (*)(ggml_backend_t, int);
    using tp_new_fn          = struct ggml_threadpool * (*)(struct ggml_threadpool_params *);
    using tp_free_fn         = void (*)(struct ggml_threadpool *);
    using set_tp_fn          = void (*)(ggml_backend_t, struct ggml_threadpool *);
    auto fn_set_threads = (set_n_threads_fn) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads");
    auto fn_tp_new      = (tp_new_fn)        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_new");
    auto fn_tp_free     = (tp_free_fn)       ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_free");
    auto fn_set_tp      = (set_tp_fn)        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_set_threadpool");

    // 4 (or 8 on >=24-core 3D V-Cache / Threadripper hosts) Q4_K matrices (~340 MB or ~680 MB) so working set > 2x LLC.
    const int64_t K = 4096, N = 36864;
    const int M = (phys >= 24) ? 8 : 4;
    ggml_init_params ip = { /*.mem_size =*/ 128 * ggml_tensor_overhead() + 4 * ggml_graph_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    std::vector<ggml_tensor *> w;
    ggml_cgraph * gf1 = ggml_new_graph(ctx);   // 1S graph for (S, 2S, 4S) marginal linearity check
    ggml_cgraph * gf2 = ggml_new_graph(ctx);   // 2S graph
    ggml_cgraph * gf  = ggml_new_graph(ctx);   // full MS graph (> 2x LLC)
    for (int i = 0; i < M; i++) {
        w.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N));
        ggml_tensor * prod = ggml_mul_mat(ctx, w.back(), x);
        if (i < M / 4) ggml_build_forward_expand(gf1, prod);
        if (i < M / 2) ggml_build_forward_expand(gf2, prod);
        ggml_build_forward_expand(gf, prod);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) {
        fprintf(stderr, "autotuner: cannot allocate test matrices\n");
        ggml_backend_free(be); ggml_free(ctx);
        return 1;
    }
    std::vector<float> x_init(K, 0.125f);
    ggml_backend_tensor_set(x, x_init.data(), 0, K * sizeof(float));
    size_t bytes = 0;
    for (int i = 0; i < M; i++) {
        const size_t nb = ggml_nbytes(w[i]);
        fill_q4k_noncompressible(w[i]->data, nb, 0x9e3779b97f4a7c15ull + (uint64_t) i * 104729ull);
        bytes += nb;
    }

    auto run_graph_median = [&](ggml_cgraph * g, int t, const std::vector<int> * pin_cpus, double budget_s) -> double {
        ThreadAffinityGuard guard;
        ggml_threadpool_params tp = ggml_threadpool_params_default(t);
        if (pin_cpus && !pin_cpus->empty()) {
            for (int c : *pin_cpus) {
                if (c >= 0 && c < GGML_MAX_N_THREADS) tp.cpumask[c] = true;
            }
        }
        struct ggml_threadpool * pool = fn_tp_new ? fn_tp_new(&tp) : nullptr;
        if (fn_set_threads) fn_set_threads(be, t);
        if (fn_set_tp && pool) fn_set_tp(be, pool);

        ggml_backend_graph_compute(be, g);   // warm-up (wakes worker pool & TLBs)
        std::vector<double> times;
        times.reserve(16);
        const auto t0 = std::chrono::steady_clock::now();
        do {
            const auto a = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(be, g);
            times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < budget_s || times.size() < 3);

        if (fn_set_tp) fn_set_tp(be, nullptr);
        if (fn_tp_free && pool) fn_tp_free(pool);
        return median_of(std::move(times));
    };

    auto measure = [&](int t, const std::vector<int> * pin_cpus) -> double {
        const double med_s = run_graph_median(gf, t, pin_cpus, seconds);
        return med_s > 0.0 ? (double) bytes / med_s / 1e9 : 0.0;
    };

    auto saturation = [](const std::vector<double> & gb, int n) -> int {
        const double best = *std::max_element(gb.begin() + 1, gb.begin() + n + 1);
        for (int t = 1; t <= n; t++) {
            if (gb[t] >= 0.95 * best) return t;
        }
        return n;
    };

    // Sweep 1..phys unpinned on an idle machine (deterministic, free of synthetic barrier convoy stalls).
    std::vector<double> gbps(phys + 1, 0.0);
    for (int t = 1; t <= phys; t++) {
        gbps[t] = measure(t, nullptr);
    }
    const int sat = saturation(gbps, phys);

    // Verify (S, 2S, 4S) scaling at `sat` threads to confirm working set escaped L3 / 3D V-Cache.
    const double t_1s = run_graph_median(gf1, sat, nullptr, 0.05);
    const double t_2s = run_graph_median(gf2, sat, nullptr, 0.05);
    const double t_4s = run_graph_median(gf,  sat, nullptr, 0.05);
    const bool cache_escaped = (t_1s > 0.0) && (t_2s > 1.35 * t_1s) && (t_4s > 1.65 * t_2s);

    // Analytical core reservation: OS-measured Direct-I/O reader CPU cores + GPU submission/spin thread.
    const int io_reserve  = sp.readable ? std::max(1, (int) std::ceil(sp.io_cpu_cores)) : 1;
    const int gpu_reserve = runs_on_gpu ? 1 : 0;
    int threads = std::max(1, std::min(sat, phys - io_reserve - gpu_reserve));

    // Hybrid P-core test (1 primary hardware thread per physical P-core -> prevents SMT sibling collisions and handles
    // non-contiguous Intel Arrow Lake layouts such as 0-1,10-15 via `fast_mask`).
    std::string fast_out;
    std::string fast_mask_out;
    std::vector<double> gbps_fast;
    const int n_fast_phys = (int) topo.fast_primary.size();
    if (n_fast_phys > 0 && n_fast_phys < phys) {
        gbps_fast.assign(n_fast_phys + 1, 0.0);
        for (int t = 1; t <= n_fast_phys; t++) {
            gbps_fast[t] = measure(t, &topo.fast_primary);
        }
        const int sat_f  = saturation(gbps_fast, n_fast_phys);
        const int n_slow = std::max(0, phys - n_fast_phys);
        // Candidate cores (Intel P-core, ARM big-core, or Socket-0 NUMA) are kept ONLY when measured > 5% faster.
        const int p_reserve = std::max(0, (io_reserve + gpu_reserve) - n_slow);
        const int tf = std::max(1, std::min(sat_f, n_fast_phys - p_reserve));
        if (gbps_fast[tf] > 1.05 * gbps[threads]) {
            threads       = tf;
            fast_out      = topo.fast_range;
            fast_mask_out = topo.fast_mask;
        }
    }

    // Prompt prefill (`-tb`) is compute-bound (tiled GEMM / iqk), so it scales across all physical cores minus GPU driver.
    const int threads_batch = std::max(threads, std::max(1, phys - gpu_reserve));
    auto list_json = [&](const std::vector<std::pair<ggml_backend_dev_t, size_t>> & devs, std::vector<TuneGpuInfo> & info) {
        std::string out;
        for (const auto & g : devs) {
            const std::string name = ggml_backend_dev_name(g.first);
            const double cur_gbps  = gpu_gbps(g.first, seconds);
            info.push_back({ name, g.second, cur_gbps });
            char gb[32];
            snprintf(gb, sizeof(gb), "%.0f", cur_gbps);
            out += (out.empty() ? "" : ", ") + std::string("{\"name\": \"") + json_escape(name) +
                   "\", \"desc\": \"" + json_escape(ggml_backend_dev_description(g.first)) +
                   "\", \"mib\": " + std::to_string(g.second) + ", \"gbps\": " + gb + "}";
        }
        return out;
    };
    std::vector<TuneGpuInfo> gpu_info, igpu_info;
    const std::string gpus = list_json(gpu_devs, gpu_info), igpus = list_json(igpu_devs, igpu_info);
    const TuneRec rec = build_recommended_args(
        gpu_info, igpu_info, threads, threads_batch, fast_mask_out, !file.empty() && sp.readable, io);

    printf("{\"gpus\": [%s], \"igpus\": [%s], \"physical_cores\": %d, \"logical_cores\": %d, "
           "\"threads\": %d, \"threads_batch\": %d, \"saturation\": %d, "
           "\"fast_cpus\": \"%s\", \"fast_mask\": \"%s\", \"cache_escaped\": %s, \"gbps\": [",
           gpus.c_str(), igpus.c_str(), phys, topo.logical_cores,
           threads, threads_batch, sat,
           fast_out.c_str(), fast_mask_out.c_str(), cache_escaped ? "true" : "false");
    for (int t = 1; t <= phys; t++) {
        printf("%s%.1f", t > 1 ? ", " : "", gbps[t]);
    }
    printf("]");
    if (!gbps_fast.empty()) {
        printf(", \"fast_cores\": \"%s\", \"gbps_fast\": [",
               (!topo.fast_range.empty() ? topo.fast_range : topo.fast_mask).c_str());
        for (size_t t = 1; t < gbps_fast.size(); t++) {
            printf("%s%.1f", t > 1 ? ", " : "", gbps_fast[t]);
        }
        printf("]");
    }
    if (!file.empty()) {
        printf(", \"side_load\": %s, \"io_threads\": %d, \"io_cores\": %.2f, \"latency_us\": %.1f, "
               "\"piece_kib\": %zu, \"read_gbps\": [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
               sp.readable ? "true" : "false", io, sp.io_cpu_cores, sp.latency_us,
               piece >> 10,
               sp.rd_gbps[0], sp.rd_gbps[1], sp.rd_gbps[2], sp.rd_gbps[3], sp.rd_gbps[4], sp.rd_gbps[5]);
    } else {
        printf(", \"side_load\": false");
    }
    printf(", \"recommended_vram_device\": \"%s\", \"recommended_args\": [", json_escape(rec.vram_device).c_str());
    for (size_t i = 0; i < rec.args.size(); i++) {
        printf("%s\"%s\"", i > 0 ? ", " : "", json_escape(rec.args[i]).c_str());
    }
    printf("]}\n");

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return 0;
}

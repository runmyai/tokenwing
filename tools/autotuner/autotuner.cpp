// autotuner: measures the parts of this PC that decide TokenWing's settings, without loading a model, and prints one
// JSON line. Reads GGUF metadata + tensor offsets directly (including multi-shard splits, 2-proj gate_up_exps, and
// custom GGML_TYPE_EXT0..EXT5 plugin quants), measures CPU memory-bound decode threads (-t), compute-bound batch
// threads (-tb), hybrid P-core / NUMA masks, Direct-I/O storage curves, GPU weight bandwidth + PCIe H2D latency/BW,
// system RAM + cgroup limits, and synthesizes deterministic conflict-free flags for any MoE architecture and setup.
//
//   autotuner [--seconds S] [--file MODEL.gguf] [--ctx TOKENS] [--ram-free MIB] [--self-test]

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
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi.h>
#if defined(_M_X64) || defined(_M_IX86)
#include <intrin.h>
#endif
#elif defined(__APPLE__)
#include <fcntl.h>
#include <mach/mach.h>
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

static std::string wide_to_utf8(const wchar_t * ws) {
    if (!ws || !*ws) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, ws, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::string out(n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws, -1, out.data(), n - 1, nullptr, nullptr);
    return out;
}
#endif

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

struct DxgiAdapterInfo {
    std::string desc;
    size_t      dedicated_mib = 0;
    size_t      shared_mib    = 0;
    uint32_t    vendor_id     = 0;
};

static std::vector<DxgiAdapterInfo> discover_dxgi_adapters() {
    std::vector<DxgiAdapterInfo> out;
#ifdef _WIN32
    HMODULE h_dxgi = LoadLibraryW(L"dxgi.dll");
    if (!h_dxgi) return out;
    using create_fn = HRESULT (WINAPI *)(REFIID, void **);
    auto fn = (create_fn) (void *) GetProcAddress(h_dxgi, "CreateDXGIFactory1");
    IDXGIFactory1 * fac = nullptr;
    if (fn && SUCCEEDED(fn(__uuidof(IDXGIFactory1), (void **) &fac)) && fac) {
        IDXGIAdapter1 * ad = nullptr;
        for (UINT idx = 0; fac->EnumAdapters1(idx, &ad) != DXGI_ERROR_NOT_FOUND; idx++) {
            DXGI_ADAPTER_DESC1 d{};
            if (SUCCEEDED(ad->GetDesc1(&d)) && (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
                DxgiAdapterInfo info;
                info.desc          = wide_to_utf8(d.Description);
                info.dedicated_mib = (size_t) (d.DedicatedVideoMemory >> 20);
                info.shared_mib    = (size_t) (d.SharedSystemMemory    >> 20);
                info.vendor_id     = d.VendorId;
                out.push_back(std::move(info));
            }
            ad->Release();
        }
        fac->Release();
    }
    FreeLibrary(h_dxgi);
#endif
    return out;
}

struct HostRamInfo {
    size_t total_mib = 16384;
    size_t avail_mib = 12288;
    bool   has_swap  = true;
};

static size_t parse_cgroup_mem_cap_mib(const std::string & cgroup_dir, size_t * out_used_mib = nullptr) {
    std::ifstream fmax(cgroup_dir + "/memory.max");
    std::string smax;
    if (!(fmax >> smax) || smax.empty() || smax == "max") return 0;
    const uint64_t cap_b = std::strtoull(smax.c_str(), nullptr, 10);
    if (cap_b == 0 || cap_b >= (1ull << 60)) return 0;
    if (out_used_mib) {
        uint64_t cur_b = 0;
        std::ifstream fcur(cgroup_dir + "/memory.current");
        if (fcur >> cur_b) *out_used_mib = (size_t) (cur_b >> 20);
    }
    return (size_t) (cap_b >> 20);
}

static HostRamInfo discover_host_ram(const std::string & proc_meminfo = "/proc/meminfo",
                                     const std::string & cgroup_dir   = "/sys/fs/cgroup") {
    HostRamInfo info;
#ifdef _WIN32
    (void) proc_meminfo; (void) cgroup_dir;
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        info.total_mib = (size_t) (ms.ullTotalPhys >> 20);
        info.avail_mib = (size_t) (ms.ullAvailPhys >> 20);
        info.has_swap  = ms.ullTotalPageFile > ms.ullTotalPhys + (256ull << 20);
    }
#elif defined(__APPLE__)
    (void) proc_meminfo; (void) cgroup_dir;
    uint64_t mem_b = 0;
    size_t sz = sizeof(mem_b);
    if (sysctlbyname("hw.memsize", &mem_b, &sz, nullptr, 0) == 0 && mem_b > 0) {
        info.total_mib = (size_t) (mem_b >> 20);
        info.avail_mib = (size_t) ((mem_b * 3 / 4) >> 20);
    }
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vmstat{};
    vm_size_t page_sz = 4096;
    host_page_size(mach_host_self(), &page_sz);
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vmstat, &count) == KERN_SUCCESS) {
        const uint64_t avail_pages = (uint64_t) vmstat.free_count + (uint64_t) vmstat.inactive_count;
        info.avail_mib = (size_t) ((avail_pages * (uint64_t) page_sz) >> 20);
    }
#else
    std::ifstream fm(proc_meminfo);
    std::string k, unit;
    uint64_t val = 0, mtotal_kb = 0, mavail_kb = 0, swap_kb = 0;
    while (fm >> k >> val) {
        std::getline(fm, unit);
        if (k == "MemTotal:")      mtotal_kb = val;
        else if (k == "MemAvailable:") mavail_kb = val;
        else if (k == "SwapTotal:")    swap_kb   = val;
    }
    if (mtotal_kb > 0) {
        info.total_mib = (size_t) (mtotal_kb >> 10);
        info.avail_mib = (size_t) ((mavail_kb > 0 ? mavail_kb : mtotal_kb * 3 / 4) >> 10);
        info.has_swap  = swap_kb > 0;
    }
    size_t cg_used = 0;
    const size_t cg_cap = parse_cgroup_mem_cap_mib(cgroup_dir, &cg_used);
    if (cg_cap > 0 && cg_cap < info.total_mib) {
        info.total_mib = cg_cap;
        const size_t cg_avail = cg_cap > cg_used ? cg_cap - cg_used : 0;
        info.avail_mib = std::min(info.avail_mib, cg_avail > 0 ? cg_avail : cg_cap);
    }
#endif
    return info;
}

struct GgufModelInfo {
    std::string arch;
    uint32_t    context_length     = 0;
    uint32_t    n_layers           = 0;
    uint32_t    n_moe_layers       = 0;
    uint32_t    n_experts          = 0;
    uint32_t    n_used             = 0;
    uint32_t    n_proj             = 3;       // 2 for fused ffn_gate_up_exps + down, else 3
    uint32_t    n_shards           = 1;
    uint32_t    hc_count           = 1;
    uint32_t    exp_type           = 0;
    size_t      piece_bytes        = 2u << 20;
    size_t      expert_bytes       = 0;       // single routed expert across all projections of 1 layer
    size_t      non_expert_mib     = 0;
    bool        has_shared_exp     = false;
    bool        has_exp_bias       = false;
    bool        has_ple            = false;
    bool        has_recurrent      = false;
    bool        needs_quant_plugin = false;
    bool        x4_quant_ok        = false;
    bool        vx_compat          = true;
    bool        tokenwing_format   = false;
    std::string correction_arg;               // "model", adjacent path, or ""
    std::string vram_experts_arg;             // "model", adjacent path, or ""
    bool        parsed             = false;
};

static FILE * open_cfile_ro(const std::string & path, uint64_t * out_size) {
#ifdef _WIN32
    const std::wstring wp = utf8_to_wide(path);
    FILE * f = _wfopen(wp.c_str(), L"rb");
    if (!f) return nullptr;
    _fseeki64(f, 0, SEEK_END);
    const int64_t sz = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    *out_size = sz > 0 ? (uint64_t) sz : 0;
    return f;
#else
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return nullptr;
    struct stat st{};
    *out_size = (fstat(fileno(f), &st) == 0 && st.st_size > 0) ? (uint64_t) st.st_size : 0;
    return f;
#endif
}

static std::vector<std::string> gguf_shard_paths(const std::string & path, std::string * out_stem = nullptr) {
    std::vector<std::string> shards = { path };
    std::string stem = path;
    if (stem.size() > 5 && stem.compare(stem.size() - 5, 5, ".gguf") == 0) {
        stem.resize(stem.size() - 5);
    }
    // Check "-00001-of-NNNNN" suffix
    if (stem.size() > 15 && stem.compare(stem.size() - 9, 4, "-of-") == 0 && stem[stem.size() - 15] == '-') {
        const int idx = std::atoi(stem.substr(stem.size() - 14, 5).c_str());
        const int cnt = std::atoi(stem.substr(stem.size() - 5,  5).c_str());
        const std::string prefix = stem.substr(0, stem.size() - 15);
        if (idx == 1 && cnt >= 2 && cnt <= 256) {
            shards.clear();
            for (int s = 1; s <= cnt; s++) {
                char buf[32];
                snprintf(buf, sizeof(buf), "-%05d-of-%05d.gguf", s, cnt);
                const std::string sp = prefix + buf;
                if (s == 1 || std::filesystem::exists(std::filesystem::u8path(sp))) {
                    shards.push_back(sp);
                }
            }
            stem = prefix;
        }
    }
    if (stem.size() > 10 && stem.compare(stem.size() - 10, 10, ".tokenwing") == 0) {
        stem.resize(stem.size() - 10);
    }
    if (out_stem) *out_stem = stem;
    return shards;
}

static GgufModelInfo inspect_gguf(const std::string & path) {
    GgufModelInfo info;
    if (path.empty()) return info;

    std::string stem;
    const std::vector<std::string> shards = gguf_shard_paths(path, &stem);
    info.n_shards = (uint32_t) shards.size();

    std::vector<double> pieces;
    std::map<int, uint64_t> layer_exp_bytes;
    std::set<int> moe_layers_seen;
    uint64_t non_exp_bytes_total = 0;
    bool has_blob_corr = false, has_blob_vram = false;
    bool saw_gate_up_fused = false;

    for (size_t si = 0; si < shards.size(); si++) {
        uint64_t file_size = 0;
        FILE * f = open_cfile_ro(shards[si], &file_size);
        if (!f || file_size < 64) { if (f) fclose(f); if (si == 0) return info; else continue; }

        auto rd = [&](void * dst, size_t n) -> bool { return fread(dst, 1, n, f) == n; };
        auto skip = [&](uint64_t n) -> bool {
#ifdef _WIN32
            return _fseeki64(f, (int64_t) n, SEEK_CUR) == 0;
#else
            return fseeko(f, (off_t) n, SEEK_CUR) == 0;
#endif
        };
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
            fclose(f);
            if (si == 0) return info; else continue;
        }

        uint64_t alignment = 32;
        uint64_t first_blob_off = UINT64_MAX;
        auto ends_with = [](const std::string & s, const char * suf) {
            const size_t n = strlen(suf);
            return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
        };

        for (uint64_t i = 0; i < n_kv; i++) {
            std::string key;
            uint32_t vtype = 0;
            if (!rd_str(key) || !rd(&vtype, 4)) { fclose(f); return info; }
            auto rd_u32_like = [&](uint32_t & dst) -> bool {
                if (vtype == 4 || vtype == 5) return rd(&dst, 4);
                if (vtype == 10 || vtype == 11) { uint64_t v = 0; if (!rd(&v, 8)) return false; dst = (uint32_t) v; return true; }
                return skip_val(skip_val, vtype);
            };
            if (key == "general.architecture" && vtype == 8) {
                if (!rd_str(info.arch)) { fclose(f); return info; }
            } else if (key == "general.alignment") {
                uint32_t a = 32;
                if (!rd_u32_like(a)) { fclose(f); return info; }
                if (a >= 4 && (a & (a - 1)) == 0) alignment = a;
            } else if (key == "tokenwing.format" && vtype == 8) {
                std::string tf;
                if (!rd_str(tf)) { fclose(f); return info; }
                info.tokenwing_format = (tf == "tokenwing model v1");
            } else if (key.rfind("tokenwing.blob.", 0) == 0 && ends_with(key, ".offset") && vtype == 10) {
                uint64_t boff = 0;
                if (!rd(&boff, 8)) { fclose(f); return info; }
                first_blob_off = std::min(first_blob_off, boff);
                if (key == "tokenwing.blob.correction.offset")   has_blob_corr = true;
                if (key == "tokenwing.blob.vram_experts.offset") has_blob_vram = true;
            } else if (ends_with(key, ".context_length")) {
                uint32_t v = 0; if (!rd_u32_like(v)) { fclose(f); return info; }
                if (info.context_length == 0) info.context_length = v;
            } else if (ends_with(key, ".block_count")) {
                uint32_t v = 0; if (!rd_u32_like(v)) { fclose(f); return info; }
                if (info.n_layers == 0) info.n_layers = v;
            } else if (ends_with(key, ".expert_count")) {
                if (!rd_u32_like(info.n_experts)) { fclose(f); return info; }
            } else if (ends_with(key, ".expert_used_count")) {
                if (!rd_u32_like(info.n_used)) { fclose(f); return info; }
            } else if (ends_with(key, ".expert_shared_count")) {
                uint32_t sc = 0; if (!rd_u32_like(sc)) { fclose(f); return info; }
                if (sc > 0) info.has_shared_exp = true;
            } else if (ends_with(key, ".hyper_connection.count")) {
                if (!rd_u32_like(info.hc_count)) { fclose(f); return info; }
            } else if (ends_with(key, ".ssm.conv_kernel")) {
                uint32_t ck = 0; if (!rd_u32_like(ck)) { fclose(f); return info; }
                if (ck > 0) info.has_recurrent = true;
            } else if (ends_with(key, ".ple.ngram_size")) {
                uint32_t pn = 0; if (!rd_u32_like(pn)) { fclose(f); return info; }
                if (pn > 0) info.has_ple = true;
            } else {
                if (!skip_val(skip_val, vtype)) { fclose(f); return info; }
            }
        }

        struct RawTensor { std::string name; uint64_t ne2 = 1; uint32_t ttype = 0; uint64_t offset = 0; bool is_exp = false; bool is_lazy = false; };
        std::vector<RawTensor> tensors;
        tensors.reserve((size_t) n_tensors);
        std::vector<uint64_t> offsets;
        offsets.reserve((size_t) n_tensors + 1);

        for (uint64_t i = 0; i < n_tensors; i++) {
            RawTensor rt;
            uint32_t n_dims = 0;
            uint64_t ne[4] = {1, 1, 1, 1};
            if (!rd_str(rt.name) || !rd(&n_dims, 4) || n_dims == 0 || n_dims > 4) { fclose(f); return info; }
            for (uint32_t d = 0; d < n_dims; d++) { if (!rd(&ne[d], 8)) { fclose(f); return info; } }
            if (!rd(&rt.ttype, 4) || !rd(&rt.offset, 8)) { fclose(f); return info; }
            rt.ne2     = ne[2];
            rt.is_exp  = (rt.name.find("_exps.weight") != std::string::npos && rt.ne2 > 1);
            rt.is_lazy = (rt.name == "per_layer_token_embd.weight");
            if (rt.is_lazy) info.has_ple = true;
            if (rt.name.find("_shexp.weight") != std::string::npos) info.has_shared_exp = true;
            if (rt.name.find("_exps.bias")    != std::string::npos) info.has_exp_bias   = true;
            if (rt.name.find("ffn_gate_up_exps.weight") != std::string::npos) saw_gate_up_fused = true;
            if (rt.name.find("ssm_conv1d") != std::string::npos || rt.name.find("attn_kda") != std::string::npos) info.has_recurrent = true;
            offsets.push_back(rt.offset);
            tensors.push_back(std::move(rt));
        }

#ifdef _WIN32
        const int64_t hdr_end = _ftelli64(f);
#else
        const off_t hdr_end = ftello(f);
#endif
        fclose(f);
        if (hdr_end <= 0) continue;

        const uint64_t data_start = (((uint64_t) hdr_end) + alignment - 1) / alignment * alignment;
        if (data_start >= file_size) continue;
        const uint64_t eff_end = std::min(file_size, first_blob_off);
        if (eff_end <= data_start) continue;
        const uint64_t data_bytes = eff_end - data_start;

        offsets.push_back(data_bytes);
        std::sort(offsets.begin(), offsets.end());
        offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());

        for (const auto & rt : tensors) {
            auto it = std::upper_bound(offsets.begin(), offsets.end(), rt.offset);
            if (it == offsets.end() || *it <= rt.offset) continue;
            const uint64_t tbytes = *it - rt.offset;
            if (rt.is_exp) {
                const size_t p = (size_t) (((tbytes / rt.ne2) + 4095ull) / 4096ull * 4096ull);
                if (p >= 4096 && p <= (128u << 20)) pieces.push_back((double) p);
                if (info.n_experts == 0) info.n_experts = (uint32_t) rt.ne2;
                if (info.exp_type == 0)  info.exp_type  = rt.ttype;
                if (rt.ttype >= (uint32_t) GGML_TYPE_EXT0 && rt.ttype <= (uint32_t) GGML_TYPE_EXT5) {
                    info.needs_quant_plugin = true;
                }
                if (rt.name.compare(0, 4, "blk.") == 0) {
                    const int il = std::atoi(rt.name.c_str() + 4);
                    moe_layers_seen.insert(il);
                    layer_exp_bytes[il] += tbytes / rt.ne2;
                }
            } else if (!rt.is_lazy) {
                non_exp_bytes_total += tbytes;
            }
        }
        info.parsed = true;
    }

    if (!info.parsed) return info;
    info.n_proj       = saw_gate_up_fused ? 2 : 3;
    info.n_moe_layers = !moe_layers_seen.empty() ? (uint32_t) moe_layers_seen.size() : info.n_layers;
    info.non_expert_mib = (size_t) ((non_exp_bytes_total + (1ull << 20) - 1) >> 20);
    if (!pieces.empty()) {
        const size_t med = ((size_t) std::llround(median_of(pieces)) + 4095u) / 4096u * 4096u;
        info.piece_bytes = std::max<size_t>(4096u, med);
    }
    if (!layer_exp_bytes.empty()) {
        std::vector<double> le;
        for (const auto & kv : layer_exp_bytes) le.push_back((double) kv.second);
        info.expert_bytes = (size_t) std::llround(median_of(std::move(le)));
    } else {
        info.expert_bytes = info.piece_bytes * info.n_proj;
    }
    info.x4_quant_ok = (info.exp_type == (uint32_t) GGML_TYPE_Q4_0   ||
                        info.exp_type == (uint32_t) GGML_TYPE_Q3_K   ||
                        info.exp_type == (uint32_t) GGML_TYPE_IQ4_XS ||
                        info.exp_type == (uint32_t) GGML_TYPE_MXFP4  ||
                        info.needs_quant_plugin);
    const bool non_silu_arch = (info.arch == "llama4" || info.arch == "gpt-oss" || info.arch == "gemma4" ||
                                info.arch == "grok"   || info.arch == "kimi-k3" || info.arch == "minimax-m3" ||
                                info.arch == "nemotron_h_moe" || info.arch == "smallthinker");
    info.vx_compat = (info.n_proj == 3) && !info.has_exp_bias && !non_silu_arch;

    auto first_existing = [](const std::vector<std::string> & cands) -> std::string {
        for (const auto & c : cands) {
            if (!c.empty() && std::filesystem::exists(std::filesystem::u8path(c))) return c;
        }
        return {};
    };
    info.correction_arg   = first_existing({ path + ".correction", stem + ".correction" });
    if (info.correction_arg.empty() && has_blob_corr) info.correction_arg = "model";
    info.vram_experts_arg = first_existing({ path + ".vram-experts.json", stem + ".vram-experts.json" });
    if (info.vram_experts_arg.empty() && has_blob_vram) info.vram_experts_arg = "model";
    return info;
}

#if (defined(__x86_64__) || defined(__i386__)) && !defined(_MSC_VER)
#include <cpuid.h>
#endif

static void detect_x86_simd(bool * out_avx2, bool * out_avx512_vnni) {
    *out_avx2 = false;
    *out_avx512_vnni = false;
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int id0[4] = {}, id7[4] = {};
    __cpuid(id0, 0);
    if (id0[0] >= 7) {
        __cpuidex(id7, 7, 0);
        *out_avx2        = (id7[1] & (1 << 5)) != 0;
        *out_avx512_vnni = ((id7[1] & (1 << 16)) != 0) && ((id7[2] & (1 << 11)) != 0);
    }
#elif defined(__x86_64__) || defined(__i386__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_max(0, nullptr) >= 7 && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        *out_avx2        = (ebx & (1u << 5)) != 0;
        *out_avx512_vnni = ((ebx & (1u << 16)) != 0) && ((ecx & (1u << 11)) != 0);
    }
#endif
}

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

static int parse_cgroup_cpu_cap(const std::string & cgroup_dir) {
    std::ifstream f(cgroup_dir + "/cpu.max");
    std::string sq;
    uint64_t period = 0;
    if (!(f >> sq >> period) || sq.empty() || sq == "max" || period == 0) return 0;
    const uint64_t quota = std::strtoull(sq.c_str(), nullptr, 10);
    return quota > 0 ? std::max(1, (int) ((quota + period - 1) / period)) : 0;
}

struct CpuTopology {
    int              logical_cores   = 1;
    int              physical_cores  = 1;
    bool             avx2            = false;
    bool             avx512_vnni     = false;
    std::vector<int> phys_primary;
    std::vector<int> fast_primary;
    std::vector<int> fast_logical;
    std::string      fast_range;
    std::string      fast_mask;
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
    detect_x86_simd(&topo.avx2, &topo.avx512_vnni);
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
    check_eff_class = (strcmp(vendor, "GenuineIntel") == 0);
#elif defined(_M_ARM64)
    check_eff_class = true;
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
    finalize_topology_masks(topo, false);
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
    double latency_us     = 0.0;
    double top_gbps       = 0.0;
};

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

static StorageProbe probe_storage(const std::string & path, size_t piece, int phys_cores) {
    StorageProbe res;
    if (path.empty()) return res;
    static constexpr int qds[6] = { 1, 2, 4, 8, 16, 32 };
    static constexpr int passes = 3;
    std::vector<double> rates[6], cores[6];

    double g0 = 0.0, c0 = 0.0;
    if (!probe_qd_once(path, piece, 1, 0.07, 11ull, &g0, &c0)) return res;
    const int n_qds = (g0 < 0.15) ? 3 : 6;

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

    res.readable   = true;
    res.top_gbps   = res.rd_gbps[top_j];
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

static void fill_q4k_noncompressible(void * dst, size_t nbytes, uint64_t seed) {
    auto * p = static_cast<uint8_t *>(dst);
    uint64_t s = seed ? seed : 0x2545f4914f6cdd1dull;
    const size_t n_blocks = nbytes / 144;
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t * blk = p + b * 144;
        const uint16_t d    = 0x2c00u;
        const uint16_t dmin = 0x2800u;
        memcpy(blk + 0, &d, 2);
        memcpy(blk + 2, &dmin, 2);
        for (size_t off = 4; off < 144; off += 8) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            memcpy(blk + off, &s, std::min<size_t>(8, 144 - off));
        }
    }
}

struct GpuDevInfo {
    std::string name;
    std::string desc;
    std::string backend;
    size_t      mib         = 0;
    double      gbps        = 0.0;
    double      pcie_gbps   = 0.0;
    double      pcie_lat_us = 0.0;
};

static GpuDevInfo probe_gpu_dev(ggml_backend_dev_t d, size_t mib, bool is_discrete, double seconds) {
    GpuDevInfo info;
    info.name = ggml_backend_dev_name(d);
    info.desc = ggml_backend_dev_description(d);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d);
    info.backend = reg ? ggml_backend_reg_name(reg) : "";
    info.mib  = mib;

    const int64_t K = 4096, N = 28672, M = 4;
    ggml_init_params ip = { 24 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    size_t bytes = 0;
    for (int i = 0; i < M; i++) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N);
        bytes += ggml_nbytes(w);
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, w, x));
    }
    const size_t h2d_bytes = 16u << 20;
    ggml_tensor * h2d_dst = is_discrete ? ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) h2d_bytes) : nullptr;

    ggml_backend_t be = ggml_backend_dev_init(d, nullptr);
    ggml_backend_buffer_t buf = be ? ggml_backend_alloc_ctx_tensors(ctx, be) : nullptr;
    if (buf) {
        ggml_backend_buffer_clear(buf, 0);
        ggml_backend_graph_compute(be, gf);
        std::vector<double> runs;
        const auto t0 = std::chrono::steady_clock::now();
        do {
            const auto a = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(be, gf);
            ggml_backend_synchronize(be);
            runs.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds || runs.size() < 3);
        const double med_s = median_of(std::move(runs));
        if (med_s > 0.0) info.gbps = (double) bytes / med_s / 1e9;

        if (is_discrete && h2d_dst) {
            ggml_backend_buffer_type_t hbuft = ggml_backend_dev_host_buffer_type(d);
            ggml_backend_buffer_t hbuf = hbuft ? ggml_backend_buft_alloc_buffer(hbuft, h2d_bytes) : nullptr;
            std::vector<uint8_t> fallback_host;
            void * hptr = hbuf ? ggml_backend_buffer_get_base(hbuf) : nullptr;
            if (!hptr) {
                fallback_host.assign(h2d_bytes, 0x5au);
                hptr = fallback_host.data();
            } else {
                memset(hptr, 0x5a, h2d_bytes);
            }
            ggml_backend_tensor_set(h2d_dst, hptr, 0, h2d_bytes);
            ggml_backend_synchronize(be);
            std::vector<double> lat_us, bw_s;
            for (int r = 0; r < 5; r++) {
                const auto a = std::chrono::steady_clock::now();
                ggml_backend_tensor_set(h2d_dst, hptr, 0, 4096);
                ggml_backend_synchronize(be);
                lat_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count());
            }
            for (int r = 0; r < 5; r++) {
                const auto a = std::chrono::steady_clock::now();
                ggml_backend_tensor_set(h2d_dst, hptr, 0, h2d_bytes);
                ggml_backend_synchronize(be);
                bw_s.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
            }
            info.pcie_lat_us = median_of(std::move(lat_us));
            const double med_bw = median_of(std::move(bw_s));
            if (med_bw > 0.0) info.pcie_gbps = (double) h2d_bytes / med_bw / 1e9;
            if (hbuf) ggml_backend_buffer_free(hbuf);
        }
        ggml_backend_buffer_free(buf);
    }
    if (be) ggml_backend_free(be);
    ggml_free(ctx);
    return info;
}

struct SynthPlan {
    std::string                        tier;
    int                                est_slots_per_layer = 0;
    std::map<std::string, std::string> env;
    std::vector<std::string>           flags;
};

static bool backend_is_cuda(const GpuDevInfo & g) {
    return g.backend.find("CUDA") != std::string::npos || g.name.rfind("CUDA", 0) == 0;
}
static bool backend_is_rocm(const GpuDevInfo & g) {
    return g.backend.find("ROCm") != std::string::npos || g.backend.find("HIP") != std::string::npos ||
           g.name.rfind("ROCm", 0) == 0 || g.name.rfind("HIP", 0) == 0;
}
static bool backend_is_vulkan(const GpuDevInfo & g) {
    return g.backend.find("Vulkan") != std::string::npos || g.name.rfind("Vulkan", 0) == 0;
}

static SynthPlan synthesize_plan(const HostRamInfo & ram, const CpuTopology & topo,
                                 int threads, int threads_batch,
                                 const std::string & fast_mask, const std::string & fast_cpus,
                                 const StorageProbe & sp,
                                 const std::vector<GpuDevInfo> & gpus_in,
                                 const std::vector<GpuDevInfo> & igpus_in,
                                 const GgufModelInfo & gm,
                                 int req_ctx, int req_ram_free_mib, bool is_windows) {
    SynthPlan plan;
    auto sort_cards = [](std::vector<GpuDevInfo> v) {
        std::stable_sort(v.begin(), v.end(), [](const GpuDevInfo & a, const GpuDevInfo & b) {
            if (std::fabs(a.gbps - b.gbps) > 1e-3) return a.gbps > b.gbps;
            return a.mib > b.mib;
        });
        return v;
    };
    const auto gpus  = sort_cards(gpus_in);
    const auto igpus = sort_cards(igpus_in);

    size_t max_gpu_mib = 0, max_igpu_mib = 0;
    for (const auto & g : gpus)  max_gpu_mib  = std::max(max_gpu_mib,  g.mib);
    for (const auto & g : igpus) max_igpu_mib = std::max(max_igpu_mib, g.mib);

    if (gpus.size() >= 2) {
        plan.tier = "multi_gpu";
    } else if (gpus.size() == 1 && max_gpu_mib >= 10240) {
        plan.tier = "discrete_large";
    } else if (gpus.size() == 1 && max_gpu_mib >= 4096) {
        plan.tier = "discrete_small";
    } else if (gpus.empty() && max_igpu_mib >= 16384) {
        plan.tier = "uma";
    } else {
        plan.tier = "cpu_only";
    }

    const int ctx = req_ctx > 0 ? req_ctx
                  : (gm.context_length > 0 ? (int) std::min<uint32_t>(gm.context_length, 32768u) : 16384);
    const int ub  = ctx >= 262144 ? 1024 : 2048;
    const int def_free_mib = !ram.has_swap ? 2048 : (is_windows ? 1536 : 1280);
    const int ram_free_mib = req_ram_free_mib > 0 ? req_ram_free_mib : def_free_mib;

    auto add = [&](const std::string & a) { plan.flags.push_back(a); };
    auto add2 = [&](const std::string & k, const std::string & v) { plan.flags.push_back(k); plan.flags.push_back(v); };

    const bool has_gpu_tier = (plan.tier != "cpu_only");
    const GpuDevInfo * primary_gpu = !gpus.empty() ? &gpus.back() : (!igpus.empty() ? &igpus.front() : nullptr);
    const bool is_cuda   = has_gpu_tier && primary_gpu && backend_is_cuda(*primary_gpu);
    const bool is_rocm   = has_gpu_tier && primary_gpu && backend_is_rocm(*primary_gpu);
    const bool is_vulkan = has_gpu_tier && primary_gpu && backend_is_vulkan(*primary_gpu);

    if (is_cuda && plan.tier != "uma") {
        plan.env["GGML_CUDA_COPY_KERNEL"]     = "65536";
        plan.env["GGML_CUDA_POOL1D_ROW"]      = "1";
        plan.env["GGML_CUDA_FUSE_AMAX"]       = "1";
        plan.env["GGML_CUDA_MOE_MARK_IDS"]    = "1";
        plan.env["GGML_CUDA_FUSE_NORM_QUANT"] = "1";
    }

    add2("--load-mode", "dio");
    add2("-fit", "off");
    add2("-ngl", has_gpu_tier ? "99" : "0");
    add2("-c", std::to_string(ctx));
    add2("-ctk", "q8_0");
    add2("-ctv", "q8_0");
    add2("-fa", "on");
    add2("-np", "1");
    add2("-t", std::to_string(threads));
    if (threads_batch > threads) add2("-tb", std::to_string(threads_batch));
    if (!fast_mask.empty())      add2("--cpu-mask", fast_mask);
    else if (!fast_cpus.empty()) add2("--cpu-range", fast_cpus);
    if (topo.logical_cores > 1 && (!fast_mask.empty() || !fast_cpus.empty())) {
        add2("--cpu-range-batch", "0-" + std::to_string(topo.logical_cores - 1));
    }
    add2("-b",  std::to_string(ub));
    add2("-ub", std::to_string(ub));
    add2("--cache-ram", "0");
    add2("--ctx-checkpoints", "2");

    // Non-MoE dense GGUF fallback: clean standard flags without --moe-stream.
    if (gm.parsed && gm.n_experts == 0) {
        if (plan.tier == "cpu_only") add2("--device", "none");
        else if (primary_gpu && !primary_gpu->name.empty()) add2("--device", primary_gpu->name);
        return plan;
    }

    // Estimate per-layer RAM slots to select --moe-stream-guess-gate vs --moe-stream-guess-skip-full.
    const size_t host_weights_mib = (has_gpu_tier && plan.tier != "uma") ? 0 : gm.non_expert_mib;
    const size_t overhead_mib     = 1536 + (size_t) ram_free_mib + host_weights_mib;
    const size_t pool_mib         = ram.avail_mib > overhead_mib ? (ram.avail_mib - overhead_mib) : 512;
    const uint32_t n_moe          = std::max<uint32_t>(1, gm.n_moe_layers > 0 ? gm.n_moe_layers : 32);
    const size_t exp_b            = std::max<size_t>(4096, gm.expert_bytes > 0 ? gm.expert_bytes : gm.piece_bytes * 3);
    const int raw_slots           = (int) (((uint64_t) pool_mib << 20) / ((uint64_t) n_moe * exp_b));
    const int min_slots           = (int) std::max<uint32_t>(6, 3 * (gm.n_used > 0 ? gm.n_used : 2));
    plan.est_slots_per_layer      = gm.n_experts > 0
            ? std::min<int>((int) gm.n_experts, std::max<int>(min_slots, raw_slots))
            : std::max<int>(min_slots, raw_slots);

    add2("--moe-stream", "model");
    add2("--moe-stream-ram-free", std::to_string(ram_free_mib) + "MiB");
    const int io = sp.readable ? sp.io_threads : 8;
    add2("--moe-stream-io-threads", std::to_string(io));
    add2("--moe-stream-overlap", "1");
    add2("--moe-stream-proj-ready", "1");
    add2("--moe-stream-chunk-prefetch", "512");
    add2("--moe-stream-io-wake", "chunk");
    add2("--moe-stream-promote", "1");
    add2("--moe-stream-guess-cap", "2");
    add2("--moe-stream-policy", "decay");
    add2("--moe-stream-prompt-end", "recency");

    const int la = (gm.parsed && gm.n_moe_layers <= 1) ? 1
                 : (sp.readable && sp.latency_us > 0.0 && sp.latency_us < 75.0 && sp.top_gbps > 11.0) ? 1 : 2;
    add2("--moe-stream-plan", "1");
    add2("--moe-stream-drop-stale", "1");
    add2("--moe-stream-needed-early", "1");
    add2("--moe-stream-lookahead", std::to_string(la));
    if (la >= 2 && gm.hc_count <= 1) add2("--moe-stream-guess-combined", "1");
    add2("--moe-stream-wordstart", std::to_string(la >= 2 ? 2 : 1));
    add2("--moe-stream-protect-named", "1");
    add2("--moe-stream-guess-inflight", std::to_string(std::min(4, io)));

    if (!gm.parsed || gm.has_shared_exp) {
        if (gm.has_shared_exp) add2("--moe-stream-pred-shared", "1");
        if (gm.has_shared_exp && !gm.correction_arg.empty()) add2("--moe-stream-correction", gm.correction_arg);
    }

    if (gm.n_experts == 0 || plan.est_slots_per_layer < (int) gm.n_experts) {
        add2("--moe-stream-guess-gate", "1");
        if (sp.readable && sp.latency_us > 0.0) {
            add2("--moe-stream-ssd-latency-us", std::to_string(std::max(1, (int) std::llround(sp.latency_us))));
        }
        if (sp.readable && sp.top_gbps > 0.0) {
            char bw[32];
            snprintf(bw, sizeof(bw), "%.2f", sp.top_gbps);
            add2("--moe-stream-ssd-gbps", bw);
        }
    } else {
        add2("--moe-stream-guess-skip-full", "1");
        if (is_cuda) add2("--moe-stream-guess-skip-full-gpu", "1");
    }

    add2("--moe-stream-cpu-skip-zero", "1");
    add2("--moe-stream-cpu-fuse", "1");
    add2("--moe-stream-cpu-dispatch", "1");
    add2("--moe-stream-cpu-topk", "1");
    add2("--moe-stream-cpu-q8-tiled", "1");
    add2("--moe-stream-cpu-fa", "1");
    add2("--moe-stream-cpu-mmid-merge", "1");
    if (topo.avx512_vnni && (!gm.parsed || gm.x4_quant_ok)) add2("--moe-stream-cpu-x4", "1");
    if (topo.avx2) add2("--moe-stream-cpu-iqk", "8");

    add2("--moe-stream-wave-plan", "1");
    add2("--moe-stream-read-ahead", "1");
    add2("--moe-stream-idle-load", "2");
    add2("--moe-stream-prompt-defer", "1");
    if (gm.arch == "bailingmoe3" || gm.arch == "flint" || gm.arch == "qwen4exp") {
        add2("--moe-stream-layer-major", "1");
        add2("--moe-stream-window", "1");
        if (ctx >= 65536) add2("--moe-stream-layer-major-batch", "32768");
    }
    if (gm.hc_count > 1) {
        add2("--moe-stream-hc-share", "1");
        add2("--moe-stream-hc-silu",  "1");
        if (is_cuda) add2("--moe-stream-hc-q8", "1");
    }
    if (gm.has_ple) add2("--moe-stream-ple-late", "1");
    if (gm.arch == "bailingmoe3") add2("--moe-stream-conv-silu", "1");

    if (plan.tier == "cpu_only") {
        add2("--moe-stream-gpu-layers", "0");
        add2("--moe-stream-cpu-weighted-sum", "1");
        add2("--moe-stream-cpu-small-ops", "1");
        if (gm.has_recurrent) add2("--moe-stream-cpu-state-inplace", "1");
        add2("--device", "none");
    } else if (plan.tier == "uma") {
        add2("--moe-stream-uma", "1");
        if (primary_gpu && !primary_gpu->name.empty()) add2("--device", primary_gpu->name);
    } else {
        if (plan.tier == "discrete_small") add2("--moe-stream-gpu-layers", "0");
        add2("--moe-stream-elastic", "1");
        add2("--moe-stream-gpu-waves", "512");
        add2("--moe-stream-gpu-wave-skip-zero", "1");
        add2("--moe-stream-prompt-split", "1");
        if (!gm.parsed || gm.vx_compat) {
            add2("--moe-stream-vram-free", "200MiB");
            if (!gm.vram_experts_arg.empty()) add2("--moe-stream-vram-experts-file", gm.vram_experts_arg);
            add2("--moe-stream-gpu-skip-zero", "1");
            add2("--moe-stream-vram-follow", "1");
            add2("--moe-stream-vram-copyback", "1");
            add2("--moe-stream-vram-end-fill", "1");
            add2("--moe-stream-copy-ahead", "1");
            if (max_gpu_mib >= 8192) add2("--moe-stream-vram-lend-prompt", "1");
        }
        if (is_cuda) {
            add2("--moe-stream-gpu-share-q8", "1");
            if (!gm.parsed || gm.vx_compat) add2("--moe-stream-vram-parallel", "1");
            add2("--moe-stream-async-h2d", is_windows ? "0" : "1");
            if (gm.arch == "qwen4exp" && gm.has_recurrent) {
                add2("--moe-stream-rs-window",  "1");
                add2("--moe-stream-rs-inplace", "1");
            }
        } else if (is_rocm) {
            add2("--moe-stream-hip-mmvq", "1");
            add2("--moe-stream-hip-sync", "1");
        } else if (is_vulkan) {
            add2("--moe-stream-vram-chunk", "1");
        }
        add("--backend-sampling");
        if (plan.tier == "multi_gpu") {
            const auto & pool = gpus.back();
            std::string dev_list = pool.name;
            for (size_t i = 0; i + 1 < gpus.size(); i++) dev_list += "," + gpus[i].name;
            add2("--device", dev_list);
            add2("--moe-stream-vram-device", pool.name);
            for (size_t i = 0; i + 1 < gpus.size(); i++) {
                add2("--moe-stream-gpu-layers-device", gpus[i].name + "=auto");
            }
        } else if (primary_gpu && !primary_gpu->name.empty()) {
            add2("--device", primary_gpu->name);
        }
    }
    return plan;
}

} // namespace

static bool has_flag_pair(const std::vector<std::string> & flags, const std::string & k, const std::string & v) {
    for (size_t i = 0; i + 1 < flags.size(); i++) {
        if (flags[i] == k && flags[i + 1] == v) return true;
    }
    return false;
}

static bool has_flag_key(const std::vector<std::string> & flags, const std::string & k) {
    return std::find(flags.begin(), flags.end(), k) != flags.end();
}

static int run_self_test() {
    for (const auto & vec : std::vector<std::vector<int>>{
            {0, 2, 4, 6, 8, 10},
            {0, 1, 10, 11, 12, 13, 14, 15},
            {0, 63, 64, 127},
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

    // 2. Verify zero-plugin GGUF v3 reader on EXT0 + trailer blob AND 2-shard 2-proj fused gate_up GGUF.
    const std::string tmp = "autotuner_selftest_ext0.gguf";
    {
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) return 1;
        auto wr = [&](const void * p, size_t n) { fwrite(p, 1, n, f); };
        auto wr_u32 = [&](uint32_t v) { wr(&v, 4); };
        auto wr_u64 = [&](uint64_t v) { wr(&v, 8); };
        auto wr_str = [&](const char * s) { uint64_t l = strlen(s); wr_u64(l); wr(s, (size_t) l); };
        wr("GGUF", 4); wr_u32(3); wr_u64(2); wr_u64(5);
        wr_str("general.architecture");           wr_u32(8); wr_str("qwen4exp");
        wr_str("qwen4exp.expert_count");          wr_u32(4); wr_u32(512);
        wr_str("qwen4exp.expert_used_count");     wr_u32(4); wr_u32(10);
        wr_str("qwen4exp.hyper_connection.count"); wr_u32(4); wr_u32(4);
        wr_str("tokenwing.blob.correction.offset"); wr_u32(10); wr_u64(94375936ull);
        wr_str("blk.0.ffn_gate_exps.weight"); wr_u32(3); wr_u64(2048); wr_u64(768); wr_u64(512); wr_u32((uint32_t) GGML_TYPE_EXT0); wr_u64(0);
        wr_str("blk.0.ffn_gate_shexp.weight"); wr_u32(2); wr_u64(512); wr_u64(2); wr_u32(0); wr_u64(94371840ull);
        const long pos = ftell(f);
        const long base = ((pos + 31) / 32) * 32;
#ifdef _WIN32
        _fseeki64(f, 94380031ll + base, SEEK_SET);
#else
        fseeko(f, (off_t) (94380031ll + base), SEEK_SET);
#endif
        char z = 0; wr(&z, 1);
        fclose(f);
    }
    const GgufModelInfo gi = inspect_gguf(tmp);
    std::remove(tmp.c_str());
    if (!gi.parsed || gi.n_experts != 512 || gi.n_used != 10 || gi.piece_bytes != 180u * 1024u ||
        !gi.needs_quant_plugin || gi.hc_count != 4 || !gi.has_shared_exp || gi.correction_arg != "model") {
        fprintf(stderr, "autotuner self-test FAIL: GGUF EXT0 inspect piece=%zu exp=%u plugin=%d hc=%u corr='%s'\n",
                gi.piece_bytes, gi.n_experts, (int) gi.needs_quant_plugin, gi.hc_count, gi.correction_arg.c_str());
        return 1;
    }

    // 2b. 2-shard split + 2-proj fused ffn_gate_up_exps.weight
    const std::string s1 = "autotuner_selftest_2p-00001-of-00002.gguf";
    const std::string s2 = "autotuner_selftest_2p-00002-of-00002.gguf";
    auto write_2p_shard = [&](const std::string & p, int il) {
        FILE * f = fopen(p.c_str(), "wb");
        if (!f) return false;
        auto wr = [&](const void * b, size_t n) { fwrite(b, 1, n, f); };
        auto wr_u32 = [&](uint32_t v) { wr(&v, 4); };
        auto wr_u64 = [&](uint64_t v) { wr(&v, 8); };
        auto wr_str = [&](const char * s) { uint64_t l = strlen(s); wr_u64(l); wr(s, (size_t) l); };
        wr("GGUF", 4); wr_u32(3); wr_u64(2); wr_u64(3);
        wr_str("general.architecture");        wr_u32(8); wr_str("qwen35moe");
        wr_str("qwen35moe.expert_count");      wr_u32(4); wr_u32(64);
        wr_str("qwen35moe.expert_used_count"); wr_u32(4); wr_u32(8);
        const std::string pre = "blk." + std::to_string(il) + ".";
        wr_str((pre + "ffn_gate_up_exps.weight").c_str()); wr_u32(3); wr_u64(1024); wr_u64(512); wr_u64(64); wr_u32((uint32_t) GGML_TYPE_Q4_K); wr_u64(0);
        wr_str((pre + "ffn_down_exps.weight").c_str());    wr_u32(3); wr_u64(256);  wr_u64(1024); wr_u64(64); wr_u32((uint32_t) GGML_TYPE_Q4_K); wr_u64(18874368ull);
        const long pos = ftell(f);
        const long base = ((pos + 31) / 32) * 32;
#ifdef _WIN32
        _fseeki64(f, base + 28311551ll, SEEK_SET);
#else
        fseeko(f, (off_t) (base + 28311551ll), SEEK_SET);
#endif
        char z = 0; wr(&z, 1);
        fclose(f);
        return true;
    };
    if (!write_2p_shard(s1, 0) || !write_2p_shard(s2, 1)) return 1;
    const GgufModelInfo gi2 = inspect_gguf(s1);
    std::remove(s1.c_str()); std::remove(s2.c_str());
    if (!gi2.parsed || gi2.n_shards != 2 || gi2.n_proj != 2 || gi2.n_moe_layers != 2 || gi2.vx_compat) {
        fprintf(stderr, "autotuner self-test FAIL: 2-shard 2-proj shards=%u proj=%u moe_l=%u vx=%d\n",
                gi2.n_shards, gi2.n_proj, gi2.n_moe_layers, (int) gi2.vx_compat);
        return 1;
    }

    // 3. Verify sysfs + cgroup v2 CPU & RAM parsers.
    namespace fs = std::filesystem;
    const fs::path fx = "autotuner_selftest_sysfs";
    auto write_txt = [](const fs::path & p, const std::string & s) {
        fs::create_directories(p.parent_path());
        std::ofstream(p) << s << "\n";
    };
    {
        fs::remove_all(fx);
        write_txt(fx / "sys/system/cpu/online", "0-23");
        write_txt(fx / "sys/cpu_core/cpus", "0-1,10-15");
        write_txt(fx / "sys/cpu_atom/cpus", "2-9,16-23");
        for (int c = 0; c < 24; c++) write_txt(fx / ("sys/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"), std::to_string(c));
        write_txt(fx / "cg/cpu.max", "max 100000");
        write_txt(fx / "cg/memory.max", std::to_string(16ull << 30));
        write_txt(fx / "cg/memory.current", std::to_string(4ull << 30));

        CpuTopology t_arl;
        parse_sysfs_cpu_topology((fx / "sys").string(), (fx / "cg").string(), {}, t_arl);
        size_t cg_used = 0;
        const size_t cg_mem = parse_cgroup_mem_cap_mib((fx / "cg").string(), &cg_used);
        if (t_arl.physical_cores != 24 || !t_arl.fast_range.empty() || t_arl.fast_mask != "0xfc03" ||
            cg_mem != 16384 || cg_used != 4096) {
            fprintf(stderr, "autotuner self-test FAIL: ArrowLake/cgroup phys=%d mask='%s' mem=%zu\n",
                    t_arl.physical_cores, t_arl.fast_mask.c_str(), cg_mem);
            fs::remove_all(fx);
            return 1;
        }
        write_txt(fx / "cg/cpu.max", "600000 100000");
        CpuTopology t_cg;
        parse_sysfs_cpu_topology((fx / "sys").string(), (fx / "cg").string(), {0, 1, 2, 3, 4, 5, 6, 7}, t_cg);
        if (t_cg.physical_cores != 6) { fs::remove_all(fx); return 1; }

        fs::remove_all(fx);
        write_txt(fx / "sys/system/cpu/online", "0-7");
        for (int c = 0; c < 8; c++) {
            write_txt(fx / ("sys/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"), std::to_string(c));
            write_txt(fx / ("sys/system/cpu/cpu" + std::to_string(c) + "/cpu_capacity"), c < 4 ? "1024" : "512");
        }
        CpuTopology t_arm;
        parse_sysfs_cpu_topology((fx / "sys").string(), (fx / "cg").string(), {}, t_arm);
        if (t_arm.physical_cores != 8 || t_arm.fast_range != "0-3" || t_arm.fast_mask != "0xf") { fs::remove_all(fx); return 1; }

        fs::remove_all(fx);
        write_txt(fx / "sys/system/cpu/online", "0-11");
        write_txt(fx / "sys/system/node/has_memory", "0,2");
        write_txt(fx / "sys/system/node/node0/cpulist", "0-3");
        write_txt(fx / "sys/system/node/node1/cpulist", "4-7");
        write_txt(fx / "sys/system/node/node2/cpulist", "8-11");
        for (int c = 0; c < 12; c++) write_txt(fx / ("sys/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"), std::to_string(c));
        CpuTopology t_numa;
        parse_sysfs_cpu_topology((fx / "sys").string(), (fx / "cg").string(), {}, t_numa);
        fs::remove_all(fx);
        if (t_numa.physical_cores != 8 || t_numa.fast_mask != "0xf") return 1;
    }

    // 4. Verify deterministic flag synthesis across CUDA, ROCm, Vulkan, Multi-GPU, UMA, and CPU-only.
    {
        HostRamInfo ram_partial = { 32768, 24576, true };
        HostRamInfo ram_huge    = { 524288, 491520, true };
        CpuTopology topo_x86;
        topo_x86.physical_cores = 16; topo_x86.logical_cores = 32; topo_x86.avx2 = true; topo_x86.avx512_vnni = true;
        StorageProbe sp_nvme;
        sp_nvme.readable = true; sp_nvme.io_threads = 8; sp_nvme.latency_us = 185.0; sp_nvme.top_gbps = 5.40;
        GgufModelInfo gm_qwen = gi;
        gm_qwen.n_moe_layers = 48; gm_qwen.expert_bytes = 2290u * 1024u; gm_qwen.has_ple = true; gm_qwen.has_recurrent = true;

        GpuDevInfo cuda24 = { "CUDA0", "RTX 4090", "CUDA", 24576, 890.0, 24.5, 9.0 };
        const SynthPlan p_cuda = synthesize_plan(ram_partial, topo_x86, 8, 15, "0xff", "", sp_nvme, {cuda24}, {}, gm_qwen, 16384, 0, false);
        if (p_cuda.tier != "discrete_large" || !has_flag_pair(p_cuda.flags, "--moe-stream-guess-gate", "1") ||
            !has_flag_pair(p_cuda.flags, "--moe-stream-hc-q8", "1") || !has_flag_pair(p_cuda.flags, "--moe-stream-ple-late", "1") ||
            p_cuda.env.count("GGML_CUDA_MOE_MARK_IDS") == 0) {
            fprintf(stderr, "autotuner self-test FAIL: CUDA discrete_large plan\n");
            return 1;
        }

        // Full-RAM regime -> --moe-stream-guess-skip-full 1 instead of guess-gate
        const SynthPlan p_full = synthesize_plan(ram_huge, topo_x86, 8, 15, "", "", sp_nvme, {cuda24}, {}, gm_qwen, 16384, 0, false);
        if (!has_flag_pair(p_full.flags, "--moe-stream-guess-skip-full", "1") || has_flag_key(p_full.flags, "--moe-stream-guess-gate")) {
            fprintf(stderr, "autotuner self-test FAIL: full-RAM skip-full plan\n");
            return 1;
        }

        // Multi-GPU (faster card filled with auto whole layers, slower card holds VRAM pool)
        GpuDevInfo cuda12 = { "CUDA1", "RTX 3060", "CUDA", 12288, 320.0, 12.0, 11.0 };
        const SynthPlan p_mgpu = synthesize_plan(ram_partial, topo_x86, 8, 15, "", "", sp_nvme, {cuda12, cuda24}, {}, gm_qwen, 16384, 0, true);
        if (p_mgpu.tier != "multi_gpu" || !has_flag_pair(p_mgpu.flags, "--moe-stream-vram-device", "CUDA1") ||
            !has_flag_pair(p_mgpu.flags, "--moe-stream-gpu-layers-device", "CUDA0=auto") ||
            !has_flag_pair(p_mgpu.flags, "--moe-stream-async-h2d", "0")) {
            fprintf(stderr, "autotuner self-test FAIL: Multi-GPU Windows plan\n");
            return 1;
        }

        // ROCm Small GPU + 2-proj non-vx_compat GGUF -> hip-mmvq/hip-sync on, no vram-free
        GpuDevInfo rocm8 = { "ROCm0", "RX 7600", "ROCm", 8192, 260.0, 13.0, 14.0 };
        const SynthPlan p_rocm = synthesize_plan(ram_partial, topo_x86, 8, 15, "", "", sp_nvme, {rocm8}, {}, gi2, 8192, 0, false);
        if (p_rocm.tier != "discrete_small" || !has_flag_pair(p_rocm.flags, "--moe-stream-hip-mmvq", "1") ||
            !has_flag_pair(p_rocm.flags, "--moe-stream-hip-sync", "1") || has_flag_key(p_rocm.flags, "--moe-stream-vram-free")) {
            fprintf(stderr, "autotuner self-test FAIL: ROCm 2-proj plan\n");
            return 1;
        }

        // Vulkan + UMA + CPU-only
        GpuDevInfo vk16  = { "Vulkan0", "Arc A770", "Vulkan", 16384, 480.0, 13.0, 15.0 };
        GpuDevInfo uma64 = { "Vulkan0", "Radeon 8060S", "Vulkan", 65536, 210.0, 0.0, 0.0 };
        const SynthPlan p_vk  = synthesize_plan(ram_partial, topo_x86, 8, 15, "", "", sp_nvme, {vk16}, {}, gm_qwen, 8192, 0, false);
        const SynthPlan p_uma = synthesize_plan(ram_partial, topo_x86, 8, 15, "", "", sp_nvme, {}, {uma64}, gm_qwen, 8192, 0, false);
        const SynthPlan p_cpu = synthesize_plan(ram_partial, topo_x86, 8, 16, "", "", sp_nvme, {}, {}, gm_qwen, 8192, 0, false);
        if (!has_flag_pair(p_vk.flags,  "--moe-stream-vram-chunk", "1") ||
            !has_flag_pair(p_uma.flags, "--moe-stream-uma", "1") || has_flag_key(p_uma.flags, "--moe-stream-elastic") ||
            !has_flag_pair(p_cpu.flags, "--device", "none") || !has_flag_pair(p_cpu.flags, "--moe-stream-cpu-weighted-sum", "1")) {
            fprintf(stderr, "autotuner self-test FAIL: Vulkan/UMA/CPU-only plan\n");
            return 1;
        }
    }

    (void) discover_dxgi_adapters();
    printf("{\"self_test\": \"PASS\", \"ext0_piece_kib\": %zu, \"shards_2p\": %u, \"fixtures\": 10}\n",
           gi.piece_bytes >> 10, gi2.n_shards);
    return 0;
}

int main(int argc, char ** argv) {
    common_init();
    ggml_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);

    double seconds = 0.14;
    std::string file;
    int req_ctx = 0;
    int req_ram_free_mib = 0;
    bool self_test = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seconds") && i + 1 < argc) {
            seconds = std::max(0.03, atof(argv[++i]));
        } else if (!strcmp(argv[i], "--file") && i + 1 < argc) {
            file = argv[++i];
        } else if ((!strcmp(argv[i], "--ctx") || !strcmp(argv[i], "-c")) && i + 1 < argc) {
            req_ctx = std::max(512, atoi(argv[++i]));
        } else if (!strcmp(argv[i], "--ram-free") && i + 1 < argc) {
            req_ram_free_mib = std::max(0, atoi(argv[++i]));
        } else if (!strcmp(argv[i], "--self-test")) {
            self_test = true;
        }
    }
    if (self_test) return run_self_test();

    const HostRamInfo host_ram = discover_host_ram();
    const CpuTopology topo     = discover_cpu_topology();
    const int phys = topo.physical_cores;

    ggml_backend_load_all();

    std::vector<std::pair<ggml_backend_dev_t, size_t>> raw_gpus, raw_igpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto dtype = ggml_backend_dev_type(d);
        if (dtype != GGML_BACKEND_DEVICE_TYPE_GPU && dtype != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(d, &free_b, &total_b);
        (dtype == GGML_BACKEND_DEVICE_TYPE_GPU ? raw_gpus : raw_igpus).push_back({ d, total_b >> 20 });
    }
    const auto dxgi_list = discover_dxgi_adapters();

    bool runs_on_gpu = !raw_gpus.empty();
    for (const auto & g : raw_igpus) runs_on_gpu = runs_on_gpu || g.second >= 16384;

    const GgufModelInfo gm = inspect_gguf(file);
    const size_t piece = gm.piece_bytes;
    const StorageProbe sp = probe_storage(file, piece, phys);
    const int io = sp.io_threads;

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t be = cpu_dev ? ggml_backend_dev_init(cpu_dev, nullptr) : nullptr;
    if (!be) {
        fprintf(stderr, "autotuner: cannot initialize CPU backend\n");
        return 1;
    }
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    using set_n_threads_fn = void (*)(ggml_backend_t, int);
    using tp_new_fn        = struct ggml_threadpool * (*)(struct ggml_threadpool_params *);
    using tp_free_fn       = void (*)(struct ggml_threadpool *);
    using set_tp_fn        = void (*)(ggml_backend_t, struct ggml_threadpool *);
    auto fn_set_threads = (set_n_threads_fn) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads");
    auto fn_tp_new      = (tp_new_fn)        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_new");
    auto fn_tp_free     = (tp_free_fn)       ggml_backend_reg_get_proc_address(cpu_reg, "ggml_threadpool_free");
    auto fn_set_tp      = (set_tp_fn)        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_set_threadpool");

    const int64_t K = 4096, N = 36864;
    const int M = (phys >= 24) ? 8 : 4;
    ggml_init_params ip = { 128 * ggml_tensor_overhead() + 4 * ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    std::vector<ggml_tensor *> w;
    ggml_cgraph * gf1 = ggml_new_graph(ctx);
    ggml_cgraph * gf2 = ggml_new_graph(ctx);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
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
        ggml_backend_graph_compute(be, g);
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

    std::vector<double> gbps(phys + 1, 0.0);
    for (int t = 1; t <= phys; t++) gbps[t] = measure(t, nullptr);
    const int sat = saturation(gbps, phys);

    const double t_1s = run_graph_median(gf1, sat, nullptr, 0.05);
    const double t_2s = run_graph_median(gf2, sat, nullptr, 0.05);
    const double t_4s = run_graph_median(gf,  sat, nullptr, 0.05);
    const bool cache_escaped = (t_1s > 0.0) && (t_2s > 1.35 * t_1s) && (t_4s > 1.65 * t_2s);

    const int io_reserve  = sp.readable ? std::max(1, (int) std::ceil(sp.io_cpu_cores)) : 1;
    const int gpu_reserve = runs_on_gpu ? 1 : 0;
    int threads = std::max(1, std::min(sat, phys - io_reserve - gpu_reserve));

    std::string fast_out, fast_mask_out;
    std::vector<double> gbps_fast;
    const int n_fast_phys = (int) topo.fast_primary.size();
    if (n_fast_phys > 0 && n_fast_phys < phys) {
        gbps_fast.assign(n_fast_phys + 1, 0.0);
        for (int t = 1; t <= n_fast_phys; t++) gbps_fast[t] = measure(t, &topo.fast_primary);
        const int sat_f  = saturation(gbps_fast, n_fast_phys);
        const int n_slow = std::max(0, phys - n_fast_phys);
        const int p_reserve = std::max(0, (io_reserve + gpu_reserve) - n_slow);
        const int tf = std::max(1, std::min(sat_f, n_fast_phys - p_reserve));
        if (gbps_fast[tf] > 1.05 * gbps[threads]) {
            threads       = tf;
            fast_out      = topo.fast_range;
            fast_mask_out = topo.fast_mask;
        }
    }
    const int threads_batch = std::max(threads, std::max(1, phys - gpu_reserve));

    std::vector<GpuDevInfo> gpu_infos, igpu_infos;
    for (const auto & g : raw_gpus)  gpu_infos.push_back(probe_gpu_dev(g.first, g.second, true,  seconds));
    for (const auto & g : raw_igpus) igpu_infos.push_back(probe_gpu_dev(g.first, g.second, false, seconds));

    auto list_json = [](const std::vector<GpuDevInfo> & devs, bool disc) {
        std::string out;
        for (const auto & g : devs) {
            char gb[32], pcie[96] = {};
            snprintf(gb, sizeof(gb), "%.0f", g.gbps);
            if (disc) {
                snprintf(pcie, sizeof(pcie), ", \"pcie_gbps\": %.1f, \"pcie_lat_us\": %.1f", g.pcie_gbps, g.pcie_lat_us);
            }
            out += (out.empty() ? "" : ", ") + std::string("{\"name\": \"") + json_escape(g.name) +
                   "\", \"desc\": \"" + json_escape(g.desc) +
                   "\", \"backend\": \"" + json_escape(g.backend) +
                   "\", \"mib\": " + std::to_string(g.mib) + ", \"gbps\": " + gb + pcie + "}";
        }
        return out;
    };
    const std::string gpus_s = list_json(gpu_infos, true), igpus_s = list_json(igpu_infos, false);

#ifdef _WIN32
    const bool is_win = true;
#else
    const bool is_win = false;
#endif
    const SynthPlan plan = synthesize_plan(host_ram, topo, threads, threads_batch, fast_mask_out, fast_out,
                                           sp, gpu_infos, igpu_infos, gm, req_ctx, req_ram_free_mib, is_win);

    printf("{\"gpus\": [%s], \"igpus\": [%s], \"ram_total_mib\": %zu, \"ram_avail_mib\": %zu, "
           "\"physical_cores\": %d, \"logical_cores\": %d, \"avx2\": %s, \"avx512_vnni\": %s, "
           "\"threads\": %d, \"threads_batch\": %d, \"saturation\": %d, "
           "\"fast_cpus\": \"%s\", \"fast_mask\": \"%s\", \"cache_escaped\": %s, \"gbps\": [",
           gpus_s.c_str(), igpus_s.c_str(), host_ram.total_mib, host_ram.avail_mib,
           phys, topo.logical_cores, topo.avx2 ? "true" : "false", topo.avx512_vnni ? "true" : "false",
           threads, threads_batch, sat,
           fast_out.c_str(), fast_mask_out.c_str(), cache_escaped ? "true" : "false");
    for (int t = 1; t <= phys; t++) printf("%s%.1f", t > 1 ? ", " : "", gbps[t]);
    printf("]");
    if (!dxgi_list.empty()) {
        printf(", \"dxgi_gpus\": [");
        for (size_t i = 0; i < dxgi_list.size(); i++) {
            printf("%s{\"desc\": \"%s\", \"dedicated_mib\": %zu, \"shared_mib\": %zu}",
                   i > 0 ? ", " : "", json_escape(dxgi_list[i].desc).c_str(),
                   dxgi_list[i].dedicated_mib, dxgi_list[i].shared_mib);
        }
        printf("]");
    }
    if (!gbps_fast.empty()) {
        printf(", \"fast_cores\": \"%s\", \"gbps_fast\": [",
               (!topo.fast_range.empty() ? topo.fast_range : topo.fast_mask).c_str());
        for (size_t t = 1; t < gbps_fast.size(); t++) printf("%s%.1f", t > 1 ? ", " : "", gbps_fast[t]);
        printf("]");
    }
    if (!file.empty()) {
        printf(", \"side_load\": %s, \"io_threads\": %d, \"io_cores\": %.2f, \"latency_us\": %.1f, "
               "\"piece_kib\": %zu, \"read_gbps\": [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
               sp.readable ? "true" : "false", io, sp.io_cpu_cores, sp.latency_us,
               piece >> 10,
               sp.rd_gbps[0], sp.rd_gbps[1], sp.rd_gbps[2], sp.rd_gbps[3], sp.rd_gbps[4], sp.rd_gbps[5]);
        if (gm.parsed) {
            printf(", \"model\": {\"arch\": \"%s\", \"context_length\": %u, \"n_layers\": %u, \"n_moe_layers\": %u, "
                   "\"n_experts\": %u, \"n_used\": %u, \"n_proj\": %u, \"n_shards\": %u, \"expert_kib\": %zu, "
                   "\"non_expert_mib\": %zu, \"needs_quant_plugin\": %s, \"vx_compat\": %s}",
                   json_escape(gm.arch).c_str(), gm.context_length, gm.n_layers, gm.n_moe_layers,
                   gm.n_experts, gm.n_used, gm.n_proj, gm.n_shards, gm.expert_bytes >> 10,
                   gm.non_expert_mib, gm.needs_quant_plugin ? "true" : "false", gm.vx_compat ? "true" : "false");
        }
    } else {
        printf(", \"side_load\": false");
    }
    printf(", \"tier\": \"%s\", \"est_slots_per_layer\": %d, \"recommended_env\": {",
           json_escape(plan.tier).c_str(), plan.est_slots_per_layer);
    size_t ei = 0;
    for (const auto & kv : plan.env) {
        printf("%s\"%s\": \"%s\"", ei++ > 0 ? ", " : "", json_escape(kv.first).c_str(), json_escape(kv.second).c_str());
    }
    printf("}, \"recommended_flags\": [");
    for (size_t fi = 0; fi < plan.flags.size(); fi++) {
        printf("%s\"%s\"", fi > 0 ? ", " : "", json_escape(plan.flags[fi]).c_str());
    }
    printf("]}\n");

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return 0;
}

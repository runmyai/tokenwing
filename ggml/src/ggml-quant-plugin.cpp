#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-quant-plugin.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

extern "C" void ggml_internal_set_ext_type_traits(
        enum ggml_type    type,
        const char *      type_name,
        int64_t           blck_size,
        size_t            type_size,
        ggml_to_float_t   to_float,
        ggml_from_float_t from_float_ref);

namespace {

std::mutex                                  g_plugin_mu;
std::atomic<const ggml_quant_plugin_api *>  g_plugin_api{ nullptr };
std::string                                 g_plugin_path;
bool                                        g_ext_has_cuda[GGML_QUANT_PLUGIN_MAX_TYPES] = {};
std::vector<void (*)(const ggml_quant_plugin_api *)> g_cpu_hooks;

#ifdef _WIN32
HMODULE g_plugin_handle = nullptr;

HMODULE load_shared_lib(const char * utf8_path, std::string & err_out) {
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8_path, -1, nullptr, 0);
    if (wlen <= 0) {
        err_out = "invalid UTF-8 path";
        return nullptr;
    }
    std::wstring wpath((size_t) wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8_path, -1, &wpath[0], wlen);

    const DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
    HMODULE h = LoadLibraryW(wpath.c_str());
    const DWORD gle = h ? 0 : GetLastError();
    SetErrorMode(old_mode);

    if (!h) {
        err_out = "LoadLibraryW failed (error " + std::to_string((unsigned long) gle) + ")";
    }
    return h;
}

void * resolve_sym(HMODULE h, const char * name) {
    return reinterpret_cast<void *>(GetProcAddress(h, name));
}
#else
void * g_plugin_handle = nullptr;

void * load_shared_lib(const char * path, std::string & err_out) {
    dlerror();
    void * h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        const char * e = dlerror();
        err_out = e ? e : "dlopen failed";
    }
    return h;
}

void * resolve_sym(void * h, const char * name) {
    return dlsym(h, name);
}
#endif

} // namespace

bool ggml_quant_plugin_load(const char * path) {
    if (path == nullptr || path[0] == '\0') {
        return true;
    }

    std::vector<void (*)(const ggml_quant_plugin_api *)> hooks_to_run;
    const ggml_quant_plugin_api * loaded_api = nullptr;

    {
        std::lock_guard<std::mutex> lock(g_plugin_mu);
        const ggml_quant_plugin_api * cur = g_plugin_api.load(std::memory_order_relaxed);
        if (cur != nullptr) {
            if (g_plugin_path == path) {
                return true;
            }
            GGML_LOG_ERROR("%s: quant plugin '%s' is already loaded; cannot also load '%s'\n",
                __func__, g_plugin_path.c_str(), path);
            return false;
        }

        std::string load_err;
        auto h = load_shared_lib(path, load_err);
        if (!h) {
            GGML_LOG_ERROR("%s: failed to open '%s': %s\n", __func__, path, load_err.c_str());
            return false;
        }

        auto get_api = reinterpret_cast<ggml_quant_plugin_get_api_t>(resolve_sym(h, GGML_QUANT_PLUGIN_ENTRY_NAME));
        if (!get_api) {
            GGML_LOG_ERROR("%s: '%s' does not export '%s'\n", __func__, path, GGML_QUANT_PLUGIN_ENTRY_NAME);
            return false;
        }

        const ggml_quant_plugin_api * api = get_api();
        if (!api || api->abi_version != GGML_QUANT_PLUGIN_ABI_VERSION) {
            GGML_LOG_ERROR("%s: '%s' ABI version mismatch (got %u, expected %u)\n",
                __func__, path, api ? api->abi_version : 0u, (unsigned) GGML_QUANT_PLUGIN_ABI_VERSION);
            return false;
        }

        if (api->init != nullptr) {
            const int32_t rc = api->init();
            if (rc != 0) {
                const char * why = api->last_error ? api->last_error() : "init failed";
                GGML_LOG_ERROR("%s: '%s' init failed (%d): %s\n", __func__, path, (int) rc, why ? why : "");
                return false;
            }
        }

        for (int32_t i = 0; i < GGML_QUANT_PLUGIN_MAX_TYPES; ++i) {
            ggml_quant_plugin_type_info info = {};
            if (!api->get_type_info || api->get_type_info(i, &info) != 0) {
                const char * why = api->last_error ? api->last_error() : "get_type_info failed";
                GGML_LOG_ERROR("%s: '%s' get_type_info(%d) failed: %s\n", __func__, path, (int) i, why ? why : "");
                return false;
            }
            if (!info.enabled) {
                continue;
            }
            if (info.blck_size <= 0 || info.type_size == 0 || info.type_name == nullptr) {
                GGML_LOG_ERROR("%s: '%s' returned invalid traits for slot %d\n", __func__, path, (int) i);
                return false;
            }
            ggml_internal_set_ext_type_traits(
                (ggml_type) (GGML_TYPE_EXT0 + i),
                info.type_name,
                info.blck_size,
                info.type_size,
                reinterpret_cast<ggml_to_float_t>(info.to_float),
                reinterpret_cast<ggml_from_float_t>(info.from_float_ref));
            g_ext_has_cuda[i] = (info.has_cuda != 0);
        }

        g_plugin_handle = h;
        g_plugin_path   = path;
        g_plugin_api.store(api, std::memory_order_release);
        loaded_api   = api;
        hooks_to_run = g_cpu_hooks;

        GGML_LOG_INFO("%s: loaded '%s' (%s) from '%s'\n",
            __func__,
            api->plugin_name ? api->plugin_name : "quant-plugin",
            api->plugin_version ? api->plugin_version : "0",
            path);
    }

    for (auto fn : hooks_to_run) {
        fn(loaded_api);
    }
    return true;
}

const struct ggml_quant_plugin_api * ggml_quant_plugin_get(void) {
    return g_plugin_api.load(std::memory_order_acquire);
}

bool ggml_quant_plugin_has_cuda(enum ggml_type type, bool is_id) {
    if (type < GGML_TYPE_EXT0 || type > GGML_TYPE_EXT5) {
        return false;
    }
    const ggml_quant_plugin_api * api = g_plugin_api.load(std::memory_order_acquire);
    if (!api || !g_ext_has_cuda[type - GGML_TYPE_EXT0]) {
        return false;
    }
    return is_id ? (api->mul_mat_id_cuda != nullptr) : (api->mul_mat_cuda != nullptr);
}

void ggml_quant_plugin_register_cpu_hook(void (*hook)(const struct ggml_quant_plugin_api * api)) {
    if (!hook) {
        return;
    }
    const ggml_quant_plugin_api * cur = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_plugin_mu);
        for (auto existing : g_cpu_hooks) {
            if (existing == hook) {
                return;
            }
        }
        g_cpu_hooks.push_back(hook);
        cur = g_plugin_api.load(std::memory_order_acquire);
    }
    if (cur != nullptr) {
        hook(cur);
    }
}

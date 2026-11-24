#include "wv_ink_plugin_internal.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

static const uint32_t kDefaultLut4U32[8][4] = {
    {0xC8B6A181u, 0xFAEFE3D6u, 0x2A1D1106u, 0x7F5F4A38u},
    {0xCDBCA681u, 0xFBF1E6DBu, 0x251A0F05u, 0x7F5A4433u},
    {0xD1C1AA81u, 0xFCF3E9DEu, 0x22170D04u, 0x7F563F2Fu},
    {0xD7C7AF81u, 0xFCF5ECE3u, 0x1D140B04u, 0x7F513929u},
    {0xD8C7AE81u, 0xFDF5EDE4u, 0x1C130B03u, 0x7F523928u},
    {0xAE9E9081u, 0xF3E0CEBEu, 0x4232200Du, 0x7F706252u},
    {0xB8A69481u, 0xFAEBDBCAu, 0x36251506u, 0x7F6C5A48u},
    {0xB4A39281u, 0xF8E7D6C5u, 0x3B2A1908u, 0x7F6E5D4Cu},
};

static std::mutex          g_err_mtx;
static std::string         g_last_error;
static std::vector<uint8_t> g_aux_storage;
static wv_ink_host_tables  g_htab;

static void reset_default_tables() {
    std::memset(&g_htab, 0, sizeof(g_htab));
    std::memcpy(g_htab.lut4, kDefaultLut4U32, sizeof(kDefaultLut4U32));
}

static int32_t api_init(void) {
    reset_default_tables();
    wv_ink_cpu_init_dispatch();
#ifdef WV_INK_PLUGIN_ENABLE_CUDA
    wv_ink_cuda_load_aux(nullptr, 0, &g_htab);
#endif
    return 0;
}

static void api_shutdown(void) {
#ifdef WV_INK_PLUGIN_ENABLE_CUDA
    wv_ink_cuda_free_aux();
#endif
    g_aux_storage.clear();
    reset_default_tables();
}

static const char * api_last_error(void) {
    std::lock_guard<std::mutex> lk(g_err_mtx);
    return g_last_error.c_str();
}

static int32_t api_load_aux_blob(const void * data, size_t size) {
    reset_default_tables();
    g_aux_storage.clear();
    if (!data || size == 0) {
#ifdef WV_INK_PLUGIN_ENABLE_CUDA
        return wv_ink_cuda_load_aux(nullptr, 0, &g_htab);
#else
        return 0;
#endif
    }
    if (size < sizeof(wv_qwv2_aux_header)) {
        wv_ink_set_error("load_aux_blob: blob smaller than QWV2 64-byte header");
        return -1;
    }
    wv_qwv2_aux_header hdr;
    std::memcpy(&hdr, data, sizeof(hdr));
    if (hdr.magic != WV_QWV2_AUX_MAGIC || hdr.version != WV_QWV2_AUX_VERSION || hdr.total_bytes != size) {
        wv_ink_set_error("load_aux_blob: invalid QWV2 magic/version/total_bytes");
        return -1;
    }
    if (hdr.cb4096_off + 16384ull > size || hdr.vq8_uv_off + 4096ull > size ||
        hdr.lut4_off + 128ull > size || hdr.tails_off > size) {
        wv_ink_set_error("load_aux_blob: codebook/tails offset out of bounds");
        return -1;
    }

    const uint8_t * base = static_cast<const uint8_t *>(data);
    g_aux_storage.assign(base, base + size);
    std::memcpy(g_htab.cb4096, g_aux_storage.data() + hdr.cb4096_off, 16384);
    std::memcpy(g_htab.vq8_uv, g_aux_storage.data() + hdr.vq8_uv_off, 4096);
    std::memcpy(g_htab.lut4,   g_aux_storage.data() + hdr.lut4_off,   128);
    g_htab.aux_base = g_aux_storage.data();
    g_htab.aux_size = g_aux_storage.size();

#ifdef WV_INK_PLUGIN_ENABLE_CUDA
    return wv_ink_cuda_load_aux(g_htab.aux_base, g_htab.aux_size, &g_htab);
#else
    return 0;
#endif
}

static int32_t api_get_type_info(int32_t ext_idx, struct ggml_quant_plugin_type_info * out_info) {
    if (!out_info || ext_idx < 0 || ext_idx >= GGML_QUANT_PLUGIN_MAX_TYPES) {
        return -1;
    }
    std::memset(out_info, 0, sizeof(*out_info));
    if (ext_idx > 3) {
        out_info->enabled = 0;
        return 0;
    }
    const char * no_cuda = std::getenv("WV_PLUGIN_NO_CUDA");
    const char * repack_mode = std::getenv("WV_PLUGIN_CPU_REPACK_MODE");
    const bool cuda_off = (no_cuda && no_cuda[0] == '1');
    const bool cpu_repack_only = (repack_mode && repack_mode[0] == '1');

    out_info->enabled      = 1;
    out_info->vec_dot_type = WV_INK_GGML_TYPE_F32;
#ifdef WV_INK_PLUGIN_ENABLE_CUDA
    out_info->has_cuda = cuda_off ? 0 : 1;
#else
    (void)cuda_off;
    out_info->has_cuda = 0;
#endif

    switch (ext_idx) {
        case 0:
            out_info->type_name  = "wv6_vq8_sb6";
            out_info->blck_size  = 64;
            out_info->type_size  = sizeof(wv_ink_block_vq8);
            out_info->to_float   = wv_ink_to_float_ext0;
            out_info->vec_dot    = wv_ink_get_vec_dot_ext0();
            out_info->vec_dot_x4 = wv_ink_get_vec_dot_x4_ext0();
            if (cpu_repack_only) {
                out_info->has_cuda = 0;
            }
            return 0;
        case 1:
            out_info->type_name  = "wv6_vq12_3b31";
            out_info->blck_size  = 128;
            out_info->type_size  = sizeof(wv_ink_block_vq12);
            out_info->to_float   = wv_ink_to_float_ext1;
            out_info->vec_dot    = wv_ink_get_vec_dot_ext1();
            out_info->vec_dot_x4 = wv_ink_get_vec_dot_x4_ext1();
            return 0;
        case 2:
            out_info->type_name  = "wv6_4b38";
            out_info->blck_size  = 128;
            out_info->type_size  = sizeof(wv_ink_block_4b38);
            out_info->to_float   = wv_ink_to_float_ext2;
            out_info->vec_dot    = wv_ink_get_vec_dot_ext2();
            out_info->vec_dot_x4 = wv_ink_get_vec_dot_x4_ext2();
            return 0;
        case 3:
            out_info->type_name  = "wv6_swz8_g32";
            out_info->blck_size  = 32;
            out_info->type_size  = sizeof(wv_ink_block_swz8);
            out_info->to_float   = wv_ink_to_float_ext3;
            out_info->vec_dot    = wv_ink_get_vec_dot_ext3();
            out_info->vec_dot_x4 = wv_ink_get_vec_dot_x4_ext3();
            return 0;
        default:
            return -1;
    }
}

static const struct ggml_quant_plugin_api g_wv_ink_api = {
    GGML_QUANT_PLUGIN_ABI_VERSION,
    "tw_vq_plugin",
    "2.1.0",
    api_init,
    api_shutdown,
    api_last_error,
    api_get_type_info,
    api_load_aux_blob,
#ifdef WV_INK_PLUGIN_ENABLE_CUDA
    wv_ink_mul_mat_cuda,
    wv_ink_mul_mat_id_cuda,
#else
    nullptr,
    nullptr,
#endif
    nullptr, // converter_create
    nullptr, // converter_destroy
    nullptr, // converter_probe_tensor
    nullptr, // converter_planned_aux_size
    nullptr, // converter_transcode_tensor
    nullptr, // converter_finalize_aux
    wv_ink_repack_expert_slot_cpu,
};

} // namespace

const wv_ink_host_tables * wv_ink_get_host_tables(void) {
    return &g_htab;
}

void wv_ink_set_error(const char * msg) {
    std::lock_guard<std::mutex> lk(g_err_mtx);
    g_last_error = msg ? msg : "";
}

#if defined(_WIN32)
#define WV_INK_EXPORT extern "C" __declspec(dllexport)
#else
#define WV_INK_EXPORT extern "C" __attribute__((visibility("default")))
#endif

WV_INK_EXPORT const struct ggml_quant_plugin_api * ggml_quant_plugin_get_api(void) {
    return &g_wv_ink_api;
}

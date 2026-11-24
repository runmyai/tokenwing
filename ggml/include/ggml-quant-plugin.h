#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Version 2: the plugin types moved to ids 112-117 (a plugin built for version 1 wrote other ids into model files)
#define GGML_QUANT_PLUGIN_ABI_VERSION 2
#define GGML_QUANT_PLUGIN_MAX_TYPES   6
#define GGML_QUANT_PLUGIN_ENTRY_NAME  "ggml_quant_plugin_get_api"
// The ggml type id of plugin slot 0 (GGML_TYPE_EXT0); slot i is GGML_QUANT_PLUGIN_TYPE_FIRST + i. A plugin's converter
// returns these ids as out_type for the tensors it converts.
#define GGML_QUANT_PLUGIN_TYPE_FIRST  112

enum ggml_quant_plugin_tensor_action {
    GGML_QUANT_PLUGIN_TENSOR_KEEP      = 0,
    GGML_QUANT_PLUGIN_TENSOR_REHEADER  = 1,
    GGML_QUANT_PLUGIN_TENSOR_TRANSCODE = 2,
};

typedef void (*ggml_quant_plugin_to_float_t)(const void * x, float * y, int64_t k);
typedef void (*ggml_quant_plugin_from_float_t)(const float * x, void * y, int64_t k);
typedef void (*ggml_quant_plugin_vec_dot_t)(
        int n, float * s, size_t bs,
        const void * vx, size_t bx,
        const void * vy, size_t by, int nrc);
typedef void (*ggml_quant_plugin_vec_dot_x4_t)(
        int n, float * s,
        const void * vx, size_t bx,
        const void * vy, int width, int pf);

struct ggml_quant_plugin_type_info {
    int32_t     enabled;
    const char * type_name;
    int64_t     blck_size;
    size_t      type_size;
    int32_t     vec_dot_type;
    int32_t     has_cuda;
    ggml_quant_plugin_to_float_t     to_float;
    ggml_quant_plugin_from_float_t   from_float_ref;
    ggml_quant_plugin_vec_dot_t      vec_dot;
    ggml_quant_plugin_vec_dot_x4_t   vec_dot_x4;
};

struct ggml_quant_plugin_tensor_plan {
    int32_t  action;
    int32_t  out_type;
    int32_t  out_n_dims;
    int32_t  reserved;
    uint64_t out_ne[4];
    uint64_t src_nbytes;
    uint64_t out_nbytes;
};

struct ggml_quant_plugin_api {
    uint32_t    abi_version;
    const char * plugin_name;
    const char * plugin_version;

    int32_t (*init)(void);
    void    (*shutdown)(void);
    const char * (*last_error)(void);

    int32_t (*get_type_info)(int32_t ext_idx, struct ggml_quant_plugin_type_info * out_info);
    int32_t (*load_aux_blob)(const void * data, size_t size);

    // Optional CUDA kernels (return 0 on success, non-zero on error).
    int32_t (*mul_mat_cuda)(
            int32_t ggml_type_id,
            const void * vx, const float * vy, float * dst,
            int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
            size_t  nb01, size_t  nb02, size_t  nb03,
            int64_t ne11, int64_t ne12, int64_t ne13,
            size_t  nb11, size_t  nb12, size_t  nb13,
            size_t  nb1,  size_t  nb2,  size_t  nb3,
            int32_t device_id, void * cuda_stream);

    // ids: int32 [n_expert_used, ne12] at byte strides ids_nb0 / ids_nb1; the result of pair (j, t) goes to dst + j*nb1 + t*nb2.
    // first_zero >= 0: the experts from first_zero on are all-zero slices (the streaming engine's reserve), whose result is 0.
    int32_t (*mul_mat_id_cuda)(
            int32_t ggml_type_id,
            const void * vx, const float * vy, const int32_t * ids, float * dst,
            int64_t ne00, int64_t ne01, int64_t ne02,
            size_t  nb01, size_t  nb02,
            int64_t ne11, int64_t ne12,
            size_t  nb11, size_t  nb12,
            int64_t n_expert_used,
            size_t  ids_nb0, size_t ids_nb1,
            size_t  nb1,     size_t nb2,
            int32_t first_zero,
            int32_t device_id, void * cuda_stream);

    // Optional offline GGUF conversion interface for make_model_file.py.
    void *   (*converter_create)(void);
    void     (*converter_destroy)(void * conv);
    int32_t  (*converter_probe_tensor)(
            void * conv, const char * name, int32_t src_type,
            int32_t n_dims, const uint64_t * src_ne,
            const void * head_bytes, size_t head_len, uint64_t max_span_bytes,
            struct ggml_quant_plugin_tensor_plan * out_plan);
    uint64_t (*converter_planned_aux_size)(void * conv);
    int32_t  (*converter_transcode_tensor)(
            void * conv, const char * name, int32_t src_type,
            int32_t n_dims, const uint64_t * src_ne,
            const void * src_bytes, size_t src_len,
            void * dst_bytes, size_t dst_len);
    int32_t  (*converter_finalize_aux)(
            void * conv, void * out_aux, size_t out_aux_cap, size_t * out_aux_written);

    // Optional in-slot repacking of the streamed experts (--moe-stream-cpu-repack 1). Used only for this plugin's own
    // types, and only for a type with no GPU kernel here (has_cuda 0): the engine copies RAM slots to a GPU in several
    // places and no GPU kernel is promised to read the repacked layout. Each expert slice read from the model is passed
    // through it into its RAM slot, same byte count. The repacked bytes are read only by this type's vec_dot and
    // vec_dot_x4, at any row, one or four rows at a time. Called from several I/O threads at once.
    // When dst_slot == NULL and src_plain == NULL, probes whether (ggml_type_id, ne0, ne1)
    // supports equal-byte in-slot repacking (returns 0 if supported, non-zero otherwise).
    // Extended probe: when dst_slot == NULL, src_plain == NULL, ne0 == -1, ne1 == 0, returns 1 if
    // the plugin's CUDA kernels (mul_mat_cuda / mul_mat_id_cuda) also natively read the repacked layout.
    int32_t  (*repack_expert_slot)(
            int32_t ggml_type_id, void * dst_slot, const void * src_plain,
            int64_t ne0, int64_t ne1);
};

typedef const struct ggml_quant_plugin_api * (*ggml_quant_plugin_get_api_t)(void);

#ifdef __cplusplus
}
#endif

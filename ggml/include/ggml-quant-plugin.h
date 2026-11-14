#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_QUANT_PLUGIN_ABI_VERSION 1
#define GGML_QUANT_PLUGIN_MAX_TYPES   4
#define GGML_QUANT_PLUGIN_ENTRY_NAME  "ggml_quant_plugin_get_api"

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

struct ggml_quant_plugin_type_info {
    int32_t     enabled;
    const char * type_name;
    int64_t     blck_size;
    size_t      type_size;
    int32_t     vec_dot_type;
    int32_t     has_cuda;
    ggml_quant_plugin_to_float_t   to_float;
    ggml_quant_plugin_from_float_t from_float_ref;
    ggml_quant_plugin_vec_dot_t    vec_dot;
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
};

typedef const struct ggml_quant_plugin_api * (*ggml_quant_plugin_get_api_t)(void);

#ifdef __cplusplus
}
#endif

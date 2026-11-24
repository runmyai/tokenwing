#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct      ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct             ggml_backend * ggml_backend_t;

// Tensor allocator
struct ggml_tallocr {
    ggml_backend_buffer_t buffer;
    void * base;
    size_t alignment;
    size_t offset;
};

GGML_API struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer);
GGML_API enum ggml_status    ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor);

// Graph allocator
/*
  Example usage:
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    // optional: create a worst-case graph and reserve the buffers to avoid reallocations
    ggml_gallocr_reserve(galloc, build_graph(max_batch));

    // allocate the graph
    struct ggml_cgraph * graph = build_graph(batch);
    ggml_gallocr_alloc_graph(galloc, graph);

    printf("compute buffer size: %zu bytes\n", ggml_gallocr_get_buffer_size(galloc, 0));

    // evaluate the graph
    ggml_backend_graph_compute(backend, graph);
*/

// special tensor flags for use with the graph allocator:
//   ggml_set_input(): all input tensors are allocated at the beginning of the graph in non-overlapping addresses
//   ggml_set_output(): output tensors are never freed and never overwritten

typedef struct ggml_gallocr * ggml_gallocr_t;

GGML_API ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft);
GGML_API ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs);
GGML_API void           ggml_gallocr_free(ggml_gallocr_t galloc);

// pre-allocate buffers from a measure graph - does not allocate or modify the graph
// call with a worst-case graph to avoid buffer reallocations
// not strictly required for single buffer usage: ggml_gallocr_alloc_graph will reallocate the buffers automatically if needed
// returns false if the buffer allocation failed
// ggml_gallocr_resrve_n_size writes the buffer sizes per galloc buffer that would be allocated by ggml_gallocr_reserve_n to sizes
GGML_API bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
GGML_API void ggml_gallocr_reserve_n_size(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids,
    size_t * sizes);
typedef void (*ggml_gallocr_before_realloc_fn)(void * user_data);

GGML_API bool ggml_gallocr_reserve_n(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids);
GGML_API bool ggml_gallocr_reserve_n_ex(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids,
    ggml_gallocr_before_realloc_fn before_realloc,
    void * before_realloc_user_data);
// the buffer type of the buffer the last reserve could not allocate, NULL when it allocated everything (or failed for another reason)
GGML_API ggml_backend_buffer_type_t ggml_gallocr_last_failed_buft(void);

// automatic reallocation if the topology changes when using a single buffer
// returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)
// Diagnostic: automatic re-plans by reason (node count, leaf count, pre-allocated before / not now, too small)
GGML_API void ggml_gallocr_get_realloc_reasons(int why[4]);

// Buffer buffer_id plans its tensors into the given caller-owned buffers (each of its own fixed size) and the
// allocator's own buffer: a tensor takes the best-fitting free space in a given buffer, and only when none can hold it the own buffer,
// which grows as needed - so a plan always fits; a size query reports what the own buffer needs. The own buffer is kept across the
// switch; n_chunks = 0 brings back the plain way. Either change makes the next alloc_graph re-plan. The chunks must outlive their use.
// own_may_grow = false (also with n_chunks = 0): a plan that needs more of the allocator's own buffer than it has fails (the reserve
// returns false) instead of growing it
GGML_API bool ggml_gallocr_set_fixed_chunks(ggml_gallocr_t galloc, int buffer_id, ggml_backend_buffer_t * chunks, int n_chunks, bool own_may_grow);
GGML_API bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
GGML_API bool ggml_gallocr_alloc_graph_n(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids);

GGML_API size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);
// Write zeros over the allocator's buffers that live in host memory (so their pages are resident from now on)
GGML_API void   ggml_gallocr_clear_host_buffers(ggml_gallocr_t galloc);

// Utils
// Create a buffer and allocate all the tensors in a ggml_context

// returns the size of the buffer that would be allocated by ggml_backend_alloc_ctx_tensors_from_buft. returns 0 on failure
GGML_API size_t                       ggml_backend_alloc_ctx_tensors_from_buft_size(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
// returns NULL on failure or if all tensors in ctx are already allocated or zero-sized
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif

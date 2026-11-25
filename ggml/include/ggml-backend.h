#pragma once

#include "ggml.h"
#include "ggml-alloc.h"

#ifdef GGML_BACKEND_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BACKEND_BUILD
#            define GGML_BACKEND_API __declspec(dllexport) extern
#        else
#            define GGML_BACKEND_API __declspec(dllimport) extern
#        endif
#    else
#        define GGML_BACKEND_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define GGML_BACKEND_API extern
#endif

#ifdef  __cplusplus
extern "C" {
#endif

    typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
    typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
    typedef struct ggml_backend_event * ggml_backend_event_t;
    typedef struct ggml_backend * ggml_backend_t;
    typedef void * ggml_backend_graph_plan_t;
    typedef struct ggml_backend_reg * ggml_backend_reg_t;
    typedef struct ggml_backend_device * ggml_backend_dev_t;


    //
    // Backend buffer type
    //

    GGML_API const char *          ggml_backend_buft_name            (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer    (ggml_backend_buffer_type_t buft, size_t size);
    GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n  (ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors);
    GGML_API size_t                ggml_backend_buft_get_alignment   (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_max_size    (ggml_backend_buffer_type_t buft);
    GGML_API size_t                ggml_backend_buft_get_alloc_size  (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
    GGML_API size_t                ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors);
    GGML_API bool                  ggml_backend_buft_is_host         (ggml_backend_buffer_type_t buft);
    GGML_API ggml_backend_dev_t    ggml_backend_buft_get_device      (ggml_backend_buffer_type_t buft);

    //
    // Backend buffer
    //

    enum ggml_backend_buffer_usage {
        GGML_BACKEND_BUFFER_USAGE_ANY = 0,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
        GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2,
    };

    GGML_API const char *                   ggml_backend_buffer_name          (ggml_backend_buffer_t buffer);
    GGML_API void                           ggml_backend_buffer_free          (ggml_backend_buffer_t buffer);
    GGML_API void *                         ggml_backend_buffer_get_base      (ggml_backend_buffer_t buffer);
    GGML_API size_t                         ggml_backend_buffer_get_size      (ggml_backend_buffer_t buffer);
    GGML_API enum ggml_status               ggml_backend_buffer_init_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    GGML_API size_t                         ggml_backend_buffer_get_alignment (ggml_backend_buffer_t buffer);
    GGML_API size_t                         ggml_backend_buffer_get_max_size  (ggml_backend_buffer_t buffer);
    GGML_API size_t                         ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor);
    GGML_API void                           ggml_backend_buffer_clear         (ggml_backend_buffer_t buffer, uint8_t value);
    GGML_API bool                           ggml_backend_buffer_is_host       (ggml_backend_buffer_t buffer);
    GGML_API void                           ggml_backend_buffer_set_usage     (ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
    GGML_API enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage     (ggml_backend_buffer_t buffer);
    GGML_API ggml_backend_buffer_type_t     ggml_backend_buffer_get_type      (ggml_backend_buffer_t buffer);
    GGML_API void                           ggml_backend_buffer_reset         (ggml_backend_buffer_t buffer);

    // tensor copy between different backends
    GGML_API void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst);

    //
    // Backend (stream)
    //

    GGML_API ggml_guid_t  ggml_backend_guid(ggml_backend_t backend);
    GGML_API const char * ggml_backend_name(ggml_backend_t backend);
    GGML_API void         ggml_backend_free(ggml_backend_t backend);

    GGML_API ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend);
    GGML_API ggml_backend_buffer_t      ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);
    GGML_API size_t                     ggml_backend_get_alignment(ggml_backend_t backend);
    GGML_API size_t                     ggml_backend_get_max_size(ggml_backend_t backend);

    GGML_API void ggml_backend_tensor_set_async   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get_async   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_set_2d_async(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);

    // "offset" refers to the offset in tensor->data for setting/getting data
    GGML_API void ggml_backend_tensor_set   (      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_get   (const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    GGML_API void ggml_backend_tensor_set_2d(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
    GGML_API void ggml_backend_tensor_memset(      struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);

    GGML_API void ggml_backend_synchronize(ggml_backend_t backend);

    GGML_API ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API void                      ggml_backend_graph_plan_free  (ggml_backend_t backend, ggml_backend_graph_plan_t plan);

    GGML_API enum ggml_status ggml_backend_graph_plan_compute (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
    GGML_API enum ggml_status ggml_backend_graph_compute      (ggml_backend_t backend, struct ggml_cgraph * cgraph);
    GGML_API enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);

    // NOTE: will be removed, use device version instead
    GGML_API bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op);
    GGML_API bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft);
    GGML_API bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op);

    // asynchronous copy
    // the copy is performed after all the currently queued operations in backend_src
    // backend_dst will wait for the copy to complete before performing other operations
    // automatic fallback to sync copy if async is not supported
    GGML_API void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

    GGML_API ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend);

    //
    // Events
    //

    GGML_API ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device);
    GGML_API void                 ggml_backend_event_free(ggml_backend_event_t event);
    GGML_API void                 ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend);
    GGML_API void                 ggml_backend_event_synchronize(ggml_backend_event_t event);
    GGML_API void                 ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event);

    //
    // Backend device
    //

    enum ggml_backend_dev_type {
        // CPU device using system memory
        GGML_BACKEND_DEVICE_TYPE_CPU,
        // GPU device using dedicated memory
        GGML_BACKEND_DEVICE_TYPE_GPU,
        // integrated GPU device using host memory
        GGML_BACKEND_DEVICE_TYPE_IGPU,
        // accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
        GGML_BACKEND_DEVICE_TYPE_ACCEL,
        // "meta" device wrapping multiple other devices for tensor parallelism
        GGML_BACKEND_DEVICE_TYPE_META,
    };

    // functionality supported by the device
    struct ggml_backend_dev_caps {
        // asynchronous operations
        bool async;
        // pinned host buffer
        bool host_buffer;
        // creating buffers from host ptr
        bool buffer_from_host_ptr;
        // event synchronization
        bool events;
        // mmap is supported for loading
        bool mmap_support;
    };

    // all the device properties
    struct ggml_backend_dev_props {
        // device name
        const char * name;
        // device description
        const char * description;
        // device free memory in bytes
        size_t memory_free;
        // device total memory in bytes
        size_t memory_total;
        // device type
        enum ggml_backend_dev_type type;
        // device id
        //   for PCI devices, this should be the lower-case PCI bus id formatted as "domain:bus:device.function" (e.g. "0000:c1:00.0")
        //   if the id is unknown, this should be NULL
        const char * device_id;
        // device capabilities
        struct ggml_backend_dev_caps caps;
    };

    GGML_API const char *                  ggml_backend_dev_name(ggml_backend_dev_t device);
    GGML_API const char *                  ggml_backend_dev_description(ggml_backend_dev_t device);
    GGML_API void                          ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total);
    GGML_API enum ggml_backend_dev_type    ggml_backend_dev_type(ggml_backend_dev_t device);
    GGML_API void                          ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props);
    GGML_API ggml_backend_reg_t            ggml_backend_dev_backend_reg(ggml_backend_dev_t device);
    GGML_API ggml_backend_t                ggml_backend_dev_init(ggml_backend_dev_t device, const char * params);
    GGML_API ggml_backend_buffer_type_t    ggml_backend_dev_buffer_type(ggml_backend_dev_t device);
    GGML_API ggml_backend_buffer_type_t    ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device);
    GGML_API ggml_backend_buffer_t         ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size);

    GGML_API bool                          ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op);
    GGML_API bool                          ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft);
    GGML_API bool                          ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op);

    //
    // Backend (reg)
    //

    GGML_API const char *       ggml_backend_reg_name(ggml_backend_reg_t reg);
    GGML_API size_t             ggml_backend_reg_dev_count(ggml_backend_reg_t reg);
    GGML_API ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index);
    GGML_API void *             ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name);

    // Common functions that may be obtained using ggml_backend_reg_get_proc_address

    // Context management and operations for faster communication between backends, used for tensor parallelism (meta backend)
    typedef void * (*ggml_backend_comm_init_t)(ggml_backend_t * backends, size_t n_backends);
    typedef void   (*ggml_backend_comm_free_t)(void * comm_ctx);
    typedef bool   (*ggml_backend_comm_allreduce_tensor_t)(void * comm_ctx, struct ggml_tensor ** tensors);

    // Split buffer type for tensor parallelism (old)
    typedef ggml_backend_buffer_type_t   (*ggml_backend_split_buffer_type_t)(int main_device, const float * tensor_split);
    // Set the number of threads for the backend
    typedef void                         (*ggml_backend_set_n_threads_t)(ggml_backend_t backend, int n_threads);
    // Get additional buffer types provided by the device (returns a NULL-terminated array)
    typedef ggml_backend_buffer_type_t * (*ggml_backend_dev_get_extra_bufts_t)(ggml_backend_dev_t device);
    // Set the abort callback for the backend
    typedef void                         (*ggml_backend_set_abort_callback_t)(ggml_backend_t backend, ggml_abort_callback abort_callback, void * abort_callback_data);
    // Get a list of feature flags supported by the backend (returns a NULL-terminated array)
    struct ggml_backend_feature {
        const char * name;
        const char * value;
    };
    typedef struct ggml_backend_feature * (*ggml_backend_get_features_t)(ggml_backend_reg_t reg);

    //
    // Backend registry
    //

    GGML_API void ggml_backend_register(ggml_backend_reg_t reg);

    GGML_API void ggml_backend_device_register(ggml_backend_dev_t device);

    // Backend (reg) enumeration
    GGML_API size_t             ggml_backend_reg_count(void);
    GGML_API ggml_backend_reg_t ggml_backend_reg_get(size_t index);
    GGML_API ggml_backend_reg_t ggml_backend_reg_by_name(const char * name);

    // Device enumeration
    GGML_API size_t             ggml_backend_dev_count(void);
    GGML_API ggml_backend_dev_t ggml_backend_dev_get(size_t index);
    GGML_API ggml_backend_dev_t ggml_backend_dev_by_name(const char * name);
    GGML_API ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type);

    // Direct backend (stream) initialization
    // = ggml_backend_dev_init(ggml_backend_dev_by_name(name), params)
    GGML_API ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params);
    // = ggml_backend_dev_init(ggml_backend_dev_by_type(type), params)
    GGML_API ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params);
    // = ggml_backend_dev_init(ggml_backend_dev_by_type(GPU) OR ggml_backend_dev_by_type(CPU), NULL)
    GGML_API ggml_backend_t ggml_backend_init_best(void);

    // Load a backend from a dynamic library and register it
    GGML_API ggml_backend_reg_t ggml_backend_load(const char * path);
    // Unload a backend if loaded dynamically and unregister it
    GGML_API void               ggml_backend_unload(ggml_backend_reg_t reg);
    // Load all known backends from dynamic libraries
    GGML_API void               ggml_backend_load_all(void);
    GGML_API void               ggml_backend_load_all_from_path(const char * dir_path);

    //
    // Backend scheduler
    //

    // The backend scheduler allows for multiple backend devices to be used together
    // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
    // The backends are selected based on:
    // - the backend that supports the operation
    // - the location of the pre-allocated tensors (e.g. the weights)
    /*
      Example usage:

        // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be assigned
        // preferably to run on the same backend as the buffer
        ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, GGML_DEFAULT_GRAPH_SIZE, false, true);

        // initialize buffers from a max size graph (optional)
        reserve_graph = build_graph(sched, max_batch_size);

        // manually assign nodes to a backend (optional, should not be needed in most cases)
        struct ggml_tensor * node = ggml_mul_mat(ctx, ...);
        ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);

        ggml_backend_sched_reserve(sched, reserve_graph);

        // compute
        graph = build_graph(sched); // the graph and its tensors are single-use in terms of allocation, multi-use in terms of computation
        for (int i = 0; i < 10; ++i) {
            ggml_backend_sched_graph_compute(sched, graph); // on the first iteration the graph is allocated automatically
        }

        // if there are graph inputs:
        graph = build_graph(sched); // get a new graph that is not allocated (the metadata for the old graph is freed once ggml_free is called)
        ggml_backend_sched_reset(sched); // clear the allocation of the previous graph
        ggml_backend_sched_alloc_graph(sched, graph); // explicitly allocate the new graph but do not execute it
        ggml_backend_tensor_set(input_tensor, ...); // copy data to the newly allocated graph tensors
        ggml_backend_sched_graph_compute(sched, graph); // execute the graph

        // as an alternative to the above it is also possible to assign the inputs to a dedicated context and
        // allocate them statically via ggml_backend_alloc_ctx_tensors
    }
    */

    typedef struct ggml_backend_sched * ggml_backend_sched_t;

    // Evaluation callback for each node in the graph (set with ggml_backend_sched_set_eval_callback)
    // when ask == true, the scheduler wants to know if the user wants to observe this node
    // this allows the scheduler to batch nodes together in order to evaluate them in a single call
    //
    // when ask == false, the scheduler is passing the node tensor to the user for observation
    // if the user returns false, the scheduler will cancel the graph compute
    //
    typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);

    // Callback when the buffers reserved for a graph are not sufficient and have to be reserved again
    // it happens on the graph allocation path and costs a synchronization of the backends
    // `sched` is the scheduler
    // `graph` is the compute graph being allocated
    // `backend_ids_changed` is true if the backend assignment of the graph nodes changed since the last allocation
    // `graph_size_changed` is true if the graph size changed since the last allocation
    typedef void (*ggml_backend_sched_realloc_callback)(ggml_backend_sched_t sched, struct ggml_cgraph * graph, bool backend_ids_changed, bool graph_size_changed, void * user_data);
    // Initialize a backend scheduler, backends with low index are given priority over backends with high index
    GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload);
    GGML_API void                 ggml_backend_sched_free(ggml_backend_sched_t sched);

    // Initialize backend buffers from a measure graph
    GGML_API void                 ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes);
    GGML_API bool                 ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph); // returns success

    GGML_API int                  ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched);
    GGML_API ggml_backend_t       ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i);

    // Get the number of splits of the last graph
    GGML_API int                  ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
    GGML_API int                  ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);

    GGML_API ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend);
    GGML_API size_t                     ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend);
    GGML_API void                       ggml_backend_sched_clear_host_buffers(ggml_backend_sched_t sched);

    GGML_API void                 ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
    GGML_API ggml_backend_t       ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node);

    // Split graph without allocating it
    GGML_API void                 ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);

    // Allocate and compute graph on the backend scheduler
    GGML_API bool                 ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph); // returns success
    // After a failed ggml_backend_sched_alloc_graph (e.g. once the buffers' limits were changed): plan and allocate the same graph again
    // without splitting it again. The split already pointed the graph's nodes at the scheduler's input copies, so the graph must never be
    // passed to ggml_backend_sched_alloc_graph a second time: build it again instead. false (nothing done) when the scheduler's last graph
    // was not one that ggml_backend_sched_alloc_graph failed to allocate (a reset, a reserve or a split came since)
    GGML_API bool                 ggml_backend_sched_alloc_graph_again(ggml_backend_sched_t sched); // returns success
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API void                 ggml_backend_sched_synchronize(ggml_backend_sched_t sched);

    // Reset all assignments and allocators - must be called before changing the node backends or allocating a new graph.
    // This in effect deallocates all tensors that were previously allocated and leaves them with dangling pointers.
    // The correct way to use this API is to discard the deallocated tensors and create new ones.
    GGML_API void                 ggml_backend_sched_reset(ggml_backend_sched_t sched);

    // Set a callback to be called for each resulting node during graph compute
    GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);

    // An input provider for a split. Called before a split's inputs are copied: first with
    // input_copies == NULL ("will you provide these inputs?" - it copies nothing and waits for nothing; it may count), then, if it
    // said yes, with the split backend's copies to fill; it then returns true when it filled every one of them, and the scheduler
    // copies none of them itself.
    typedef bool (*ggml_backend_sched_input_hook_t)(void * user_data, ggml_backend_t split_backend, int n_inputs,
                                                     struct ggml_tensor ** inputs, struct ggml_tensor ** input_copies);
    GGML_API void                 ggml_backend_sched_set_input_hook(ggml_backend_sched_t sched, ggml_backend_sched_input_hook_t hook, void * user_data);
    // Called after the scheduler issued a split (on a GPU backend: its graph is enqueued; on the CPU: computed;
    // with an eval callback, once per part issued), with the split's backend and the graph it issued
    typedef void (*ggml_backend_sched_split_hook_t)(void * user_data, ggml_backend_t split_backend, struct ggml_cgraph * graph);
    GGML_API void                 ggml_backend_sched_set_split_hook(ggml_backend_sched_t sched, ggml_backend_sched_split_hook_t hook, void * user_data);
    // A preset target for the copy of a split input. Called when the scheduler makes the copy of `src`
    // (a tensor in another backend's buffer) for the split `i_split` of `dst_backend`, and once at the start of every graph split
    // with src == NULL (`graph` = the graph being split; the hook may scan it). To use its own memory the hook sets `copy->view_src` and
    // `copy->view_offs` (a view of a tensor that already has a buffer, big enough for `copy`'s bytes) and returns true: the allocator
    // then does not allocate the copy, and the scheduler copies the data into it as into any copy. false = the copy is made as always.
    typedef bool (*ggml_backend_sched_copy_hook_t)(void * user_data, ggml_backend_t dst_backend, const struct ggml_tensor * src,
                                                    struct ggml_tensor * copy, const struct ggml_tensor * consumer, int i_split,
                                                    const struct ggml_cgraph * graph);
    GGML_API void                 ggml_backend_sched_set_copy_hook(ggml_backend_sched_t sched, ggml_backend_sched_copy_hook_t hook, void * user_data);
    // The buffer of `backend` plans its tensors only inside the given caller-owned buffers (ggml_gallocr_set_fixed_chunks):
    // n_chunks = 0 brings back the scheduler's own buffer. The next graph re-plans; a plan that does not fit them fails (the graph's
    // allocation reports it). false = not possible (the backend is not the scheduler's, or its buffer type is shared)
    GGML_API bool                 ggml_backend_sched_set_fixed_chunks(ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_buffer_t * chunks, int n_chunks, bool own_may_grow);
    // Diagnostic: the place hook's time (0: the hook is gone), and the automatic re-plans' GPU wait and planning time, since the scheduler was made
    GGML_API void                 ggml_backend_sched_get_alloc_times(ggml_backend_sched_t sched, int64_t * place_us, int64_t * sync_us, int64_t * plan_us, int * n_realloc);
    // Copy a split input that a finished CPU split wrote into the split GPU's pinned host buffer with an
    // async H2D on that GPU's stream, without the host waits (no effect with pipeline parallelism)
    GGML_API void                 ggml_backend_sched_set_async_host_input(ggml_backend_sched_t sched, bool on);
    GGML_API bool                 ggml_backend_sched_get_async_host_input(ggml_backend_sched_t sched);   //
    GGML_API void                 ggml_backend_sched_wait_async_host_inputs(ggml_backend_sched_t sched);

    // whether ops that read host weights may run on a device backend, with a copy of those weights.
    // Read when a graph is assigned, so it may be changed between graphs
    GGML_API void                 ggml_backend_sched_set_op_offload(ggml_backend_sched_t sched, bool op_offload);
    // the one backend such an op may be offloaded to (it is not offloaded when that backend declines it); NULL = the first backend
    // in the scheduler's order that takes it (the default)
    GGML_API void                 ggml_backend_sched_set_op_offload_backend(ggml_backend_sched_t sched, ggml_backend_t backend);

    // Where the time of a graph goes, measured only when GGML_PROFILE is set, printed by
    // ggml_prof_dump(). The slots accumulate microseconds, the counts are plain integers.
    enum ggml_prof_slot {
        GGML_PROF_CPU_COMPUTE = 0, // wall time inside the host backend's graph compute
        GGML_PROF_GPU_COMPUTE,     // device time of the kernels, from CUDA events
        GGML_PROF_COPY,            // wall time in the cross-backend tensor copies
        GGML_PROF_SYNC,            // wall time waiting for a backend or an event
        GGML_PROF_N_SLOTS,
    };
    enum ggml_prof_count {
        GGML_PROF_N_GRAPHS   = 0, // calls of ggml_backend_sched_compute_splits
        GGML_PROF_N_SPLITS,       // the backend pieces those graphs were split into
        GGML_PROF_N_SPLIT_INPUTS, // the cross-backend inputs the scheduler had to bring over
        GGML_PROF_N_COPIES,       // tensor copies between backends
        GGML_PROF_N_SYNCS,        // waits for a backend or an event that block the host
        GGML_PROF_N_CUDA_CAPTURED, GGML_PROF_N_CUDA_REPLAYED, GGML_PROF_N_CUDA_DISABLED,
        GGML_PROF_N_CUDA_DISABLED_P0, GGML_PROF_N_CUDA_DISABLED_P1,
        GGML_PROF_N_CUDA_DISABLED_P2, GGML_PROF_N_CUDA_DISABLED_P3,
        GGML_PROF_N_COUNTS,
    };
    // Turns the timers on (or off) at run time; GGML_PROFILE sets the start value. Call it before the first graph is computed
    GGML_API void                 ggml_prof_enable(int on);
    GGML_API bool                 ggml_prof_enabled(void);
    GGML_API void                 ggml_prof_add(int slot, int64_t us);
    GGML_API void                 ggml_prof_count(int what, int64_t n);
    // a CUDA graph was captured (kind 0), replayed (kind 1) or not used (kind 2, with a reason 0..3:
    // policy, incompatible, warmup, properties changed)
    GGML_API void                 ggml_prof_cuda_graph(int kind, int reason);
    GGML_API void                 ggml_prof_dump(void);
    // the running total of a slot, so that a caller can report the difference over its own interval
    GGML_API int64_t              ggml_prof_get(int slot);

    // GGML_PROFILE_HANDOFF=1 - host-clock timers of the CPU <-> GPU hand-off (a measuring tool, off when
    // unset). The variable is read once when ggml-base loads; with it unset the scheduler and the CPU graph loop run
    // their unchanged code (the timed versions are separate copies chosen once per graph). With it set, the timers only
    // read ggml_time_us() around calls that exist anyway: no sync, copy or change of order is added. Lines on stderr
    // start with "handoff:"; ggml_handoff_dump() prints the per-word summary.
    enum ggml_handoff_call {           // what the current llama_decode call is (set by the caller of the scheduler)
        GGML_HANDOFF_PROMPT = 0,       // a call with more than one token
        GGML_HANDOFF_DECODE,           // a one-token call (a word)
        GGML_HANDOFF_OTHER,            // scheduler work outside a decode call
        GGML_HANDOFF_N_CALL,
    };
    enum ggml_handoff_ctx {            // the context-level timers of a call (microseconds)
        GGML_HANDOFF_C_BUILD = 0,      // graph build + allocation (no reuse)
        GGML_HANDOFF_C_REUSE,          // the reuse check when the previous graph is reused
        GGML_HANDOFF_C_SET_INPUT,      // set_inputs
        GGML_HANDOFF_C_COMPUTE,        // ggml_backend_sched_graph_compute_async
        GGML_HANDOFF_C_OUTPUT,         // the logits / embeddings readback calls in decode
        GGML_HANDOFF_C_SYNC,           // llama_context::synchronize (the host block on the device's last work)
        GGML_HANDOFF_N_CTX,
    };
    // the CPU graph loop's per-node record (ggml-cpu): the node as thread 0 saw it and the spread over the threads
    struct ggml_handoff_cpu_node {
        const char * name;             // the node's name (the layer suffix "-<il>" is dropped when aggregated)
        const char * op;               // ggml_op_desc of the node
        int          is_mmid;          // GGML_OP_MUL_MAT_ID
        int          n_fused;          // nodes the CPU backend fused into this one
        int64_t      wall_us;          // thread 0: start of this node -> start of the next one (incl. the barrier)
        int64_t      stagger_us;       // first thread start -> last thread start
        int64_t      work_sum_us;      // sum over the threads of (end - start)
        int64_t      work_max_us;      // the longest thread
        int64_t      spread_us;        // first thread finish -> last thread finish
        int64_t      slot_wait_max_us; // the longest wait of one thread for an expert slot inside the node
        int64_t      idle_us;          // mean over the threads of (the last thread's end - its end)
    };
    // GGML_TRACE=<file> - a timeline of the engine as Chrome / Perfetto trace JSON (a measuring
    // tool, off when unset; GGML_TRACE_MAX_MB caps the file, default 512). The variable is read once, on the first call of
    // ggml_trace_on(); with it unset the instrumented places run their unchanged code (one start-up-chosen check each). Events
    // use the same clock as ggml_time_us() (CLOCK_MONOTONIC, microseconds), so they line up with the recorder. A lane is
    // GGML_TRACE_LANE(kind, sub): kind 1 scheduler splits / waits / copies (sub 0), 2 GPU pieces issue -> done (sub = device),
    // 3 CPU threads' waits for expert slots (sub = thread), 4 SSD reads (sub = I/O thread), 5 remap op, 6 guess helper,
    // 7 reads as the CPU saw them land. arg is free (see the caller).
    #define GGML_TRACE_LANE(kind, sub) ((kind)*1000 + (sub))
    // A MUL_MAT_ID node whose weight tensor's LAST n experts are all-zero
    // slices carries GGML_MOE_ZERO_TAIL_MAGIC | n in op_params[GGML_MOE_ZERO_TAIL_PARAM]; a backend may then skip their pairs and write 0
    // in their output rows (what the zero slices give). Unset = 0 = the op as before
    #define GGML_MOE_ZERO_TAIL_PARAM ((int) (GGML_MAX_OP_PARAMS/sizeof(int32_t)) - 2)
    #define GGML_MOE_ZERO_TAIL_MAGIC 0x5A700000
    // A MUL_MAT_MULTI / ARGSORT node of the GPU's combined guess may carry this tag in the same op-param slots as skip-full's
    // CPU tag (GGML_CPU_GSF_MAGIC) - but its array address is GPU memory, so only the CUDA backend reads it (the CPU never does)
    #define GGML_GPU_GSF_MAGIC 0x5C9
    #define GGML_MOE_ZERO_TAIL_MASK  0xFFF00000
    // A node a model's graph builder marks for a backend fusion carries GGML_FUSE_TAG_MAGIC | bits in op_params[GGML_FUSE_TAG_PARAM]
    // (unset = 0 = the op as before; a backend that does not know the tag ignores it). GGML_FUSE_TAG_SCALE_SILU on a SCALE: the
    // SCALE and the UNARY SILU right after it may run as one kernel (the same float steps). GGML_FUSE_TAG_Q8_1 on that SCALE or on an
    // RMS_NORM followed by its MUL: the fused kernel may also write the q8_1 copy of its output that a quantized matvec reading it uses
    #define GGML_FUSE_TAG_PARAM ((int) (GGML_MAX_OP_PARAMS/sizeof(int32_t)) - 3)
    #define GGML_FUSE_TAG_MAGIC 0x7A600000
    #define GGML_FUSE_TAG_MASK  0xFFF00000
    #define GGML_FUSE_TAG_SCALE_SILU 1
    #define GGML_FUSE_TAG_Q8_1       2
    GGML_API bool                 ggml_trace_on(void);
    GGML_API void                 ggml_trace_event(int lane, const char * name, int64_t t0_us, int64_t t1_us, int64_t arg);
    GGML_API bool                 ggml_handoff_on(void);
    GGML_API void                 ggml_handoff_call_begin(int kind, int n_tokens);
    GGML_API void                 ggml_handoff_call_end(int64_t decode_us);   // the decode call returned
    GGML_API void                 ggml_handoff_ctx_add(int what, int64_t us);
    // one CPU graph: its wall time in ggml_graph_compute, thread 0's fork (entry -> first node) and join (last
    // barrier -> return), the first node's start stagger over the threads, and its nodes
    GGML_API void                 ggml_handoff_cpu_graph(int64_t total_us, int64_t fork_us, int64_t join_us,
                                                         int64_t fork_stagger_us, int n_threads,
                                                         const struct ggml_handoff_cpu_node * nodes, int n_nodes);
    GGML_API void                 ggml_handoff_dump(void);

    // Set a callback to be called when the compute buffers of a graph are being reallocated
    GGML_API void                 ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, void * user_data);

    //
    // Meta backend
    //

#define GGML_BACKEND_META_MAX_DEVICES 16

    enum ggml_backend_meta_split_axis {
        // tensor split by tensor dimensions:
        GGML_BACKEND_SPLIT_AXIS_0 = 0,
        GGML_BACKEND_SPLIT_AXIS_1 = 1,
        GGML_BACKEND_SPLIT_AXIS_2 = 2,
        GGML_BACKEND_SPLIT_AXIS_3 = 3,

        GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10, // all values on all backends
        GGML_BACKEND_SPLIT_AXIS_PARTIAL  = 11, // each backend has a partial sum

        // for internal bookkeeping only:
        GGML_BACKEND_SPLIT_AXIS_NONE    = 98,
        GGML_BACKEND_SPLIT_AXIS_UNKNOWN = 99,
    };
    GGML_API const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis);

    struct ggml_backend_meta_split_state {
        enum ggml_backend_meta_split_axis axis;

        // for tensors with axis >= 0 && axis < GGML_MAX_DIMS:
        //   - each device has a slice of the tensor along the split axis
        //   - most tensors have n_segments == 1 and a contiguous slice of the tensor data
        //   - some tensors have an inhomogenenous data layout along the split axis,
        //     those tensors are divided into segments which are each individually split across devices
        //   - ne has one entry per segment and device and that segment repeats nr times,
        //     in total when accounting for repetitions the segments add up to ggml_tensor::ne for that axis,
        //     the outer/inner loops are over segments/devices like [seg0_dev0_r0, seg0_dev1_r0, seg0_dev0_r1, seg0_dev1_r1, seg1_dev0_r0, seg1_dev1_r0],
        //   - for example, a transformer may have a fused QKV matrix rather than 3 matrices, those would be 3 separate segments
        //     that each need to be split individually across devices so that each device gets a slice of Q, K, and V,
        //     the Q matrix can be larger than the K and V matrices so this can either be expressed as 3 segments or as 2 segments
        //     where the segment for K/V repeats twice
        int64_t  ne[16*GGML_BACKEND_META_MAX_DEVICES];
        uint32_t nr[16];
        uint32_t n_segments;
    };

    // function to assign split states for statically allocated tensors, compute tensor split states will be assigned to be compatible:
    typedef struct ggml_backend_meta_split_state(*ggml_backend_meta_get_split_state_t)(const struct ggml_tensor * tensor, void * userdata);

    // create a new meta device from "simple" devices, meta buffer type/buffer/backend is then derived from this:
    // TODO: this looks a bit strange - a backend API creates a device. I think we should try
    //       express this as a backend registry functionality instead
    GGML_API ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud);

    //
    // Utils
    //

    struct ggml_backend_graph_copy {
        ggml_backend_buffer_t buffer;
        struct ggml_context * ctx_allocated;
        struct ggml_context * ctx_unallocated;
        struct ggml_cgraph * graph;
    };

    // Copy a graph to a different backend
    GGML_API struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph);
    GGML_API void                           ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy);

    typedef bool (*ggml_backend_eval_callback)(int node_index, struct ggml_tensor * t1, struct ggml_tensor * t2, void * user_data);

    // Compare the output of two backends
    GGML_API bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes);

    // Tensor initialization
    GGML_API enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr);
    GGML_API enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor);

    // CPU buffer types are always available
    GGML_API ggml_backend_buffer_t      ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);

    // The CPU_REPACK buffer type and a buffer of that type over memory the caller owns. The repacked
    // kernels (mxfp4_8x8_q8_0 and friends, ggml/src/ggml-cpu/repack.cpp) are chosen by the buffer type of
    // src0, so weights that the engine fills itself - the MoE expert pools - can use them by being
    // wrapped in this buffer. The caller keeps the alignment (the pools are filled with O_DIRECT reads,
    // so they need 4096-byte alignment and the type's own allocator only promises TENSOR_ALIGNMENT).
    GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);
    GGML_BACKEND_API ggml_backend_buffer_t      ggml_backend_cpu_repack_buffer_from_ptr(void * ptr, size_t size);
    GGML_API ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);

#ifdef  __cplusplus
}
#endif

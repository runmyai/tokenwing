// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <atomic>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <mutex>
#include <map>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <condition_variable>
#include <deque>
#include <string_view>
#include <thread>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

// The hand-off profile prints the async_h2d copy kind only when the setting was used;
// the hook-provided CPU splits that had to synchronize the GPU first (the graph-shape guard of async-h2d) are counted
static bool                 g_ho_async_h2d_on = false;
static std::atomic<int64_t> g_async_guard_syncs{0};
static std::atomic<int64_t> g_async_multi_syncs{0};   // several GPUs: a CPU split waited for another GPU's pending async copies


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
struct ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    int    first; // first tensor index (inclusive)
    int    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

static ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    int    first        = 0;

    for (int i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        struct ggml_tensor * t = tensors[i];
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } else {
            cur_buf_size += this_size;
        }
    }

    if (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    return plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
static ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        if (buffer == NULL) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            return NULL;
        }

        struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        struct ggml_tensor * t_failed = NULL;
        for (int j = item.first; j < item.last; j++) {
            struct ggml_tensor * t = tensors[j];
            if (t->data == NULL) {
                if (t->view_src == NULL) {
                    if (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                } else if (t->buffer == NULL) {
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            } else {
                if (t->view_src != NULL && t->buffer == NULL) {
                    // view of a pre-allocated tensor
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            }
        }
        if (t_failed != NULL) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            return NULL;
        }

        buffers.push_back(buffer);
    }

    if (buffers.empty()) {
        return NULL;
    }

    if (buffers.size() == 1) {
        return buffers[0];
    }

    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
static size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    return total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.alloc_buffer_n) {
        return buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        return size;
    }
    return ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.get_alloc_size_n) {
        return buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } else if (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

// ------------------------------------------------------------------------------------------
// The profiler, a no-op unless GGML_PROFILE is set
// -1 = not decided yet (the variable decides on the first use), 0 / 1 = set by ggml_prof_enable() or read from the variable
static std::atomic<int> g_prof_state{-1};

static bool ggml_prof_on(void) {
    int on = g_prof_state.load(std::memory_order_relaxed);
    if (on < 0) {
        const char * e = getenv("GGML_PROFILE");
        on = e && atoi(e) ? 1 : 0;
        g_prof_state.store(on, std::memory_order_relaxed);
    }
    return on == 1;
}

void ggml_prof_enable(int on) {
    g_prof_state.store(on ? 1 : 0, std::memory_order_relaxed);
}

bool ggml_prof_enabled(void) {
    return ggml_prof_on();
}

static int64_t g_prof_us[GGML_PROF_N_SLOTS] = {0};
static int64_t g_prof_n[GGML_PROF_N_COUNTS] = {0};
static std::vector<std::pair<std::string, int64_t>> g_prof_cuda_graph;

void ggml_prof_add(int slot, int64_t us) {
    if (ggml_prof_on() && slot >= 0 && slot < GGML_PROF_N_SLOTS) {
        g_prof_us[slot] += us;
    }
}

void ggml_prof_count(int what, int64_t n) {
    if (ggml_prof_on() && what >= 0 && what < GGML_PROF_N_COUNTS) {
        g_prof_n[what] += n;
    }
}

void ggml_prof_cuda_graph(int kind, int reason) {
    if (ggml_prof_on()) {
        g_prof_n[GGML_PROF_N_CUDA_CAPTURED + kind] += 1;
        if (kind == 2) {   // only a graph that was not used has a reason
            g_prof_n[GGML_PROF_N_CUDA_DISABLED_P0 + reason] += 1;
        }
    }
}

int64_t ggml_prof_get(int slot) {
    return slot >= 0 && slot < GGML_PROF_N_SLOTS ? g_prof_us[slot] : 0;
}

void ggml_prof_dump(void) {
    if (!ggml_prof_on()) {
        return;
    }
    // stderr, not GGML_LOG_INFO: ggml's own logger prints nothing unless the application sets it up
    static const char * names[GGML_PROF_N_SLOTS] = {
        "host backend compute", "device kernels (cuda events)", "cross-backend copies", "backend syncs",
    };
    static const char * reasons[4] = {
        "not used by the policy", "the graph is not compatible", "warmup not complete", "properties changed",
    };
    fprintf(stderr, "ggml profile: %lld graphs, %lld splits in them (%.2f per graph), %lld cross-backend "
            "inputs, %lld copies, %lld blocking syncs\n",
            (long long) g_prof_n[GGML_PROF_N_GRAPHS], (long long) g_prof_n[GGML_PROF_N_SPLITS],
            (double) g_prof_n[GGML_PROF_N_SPLITS]/g_prof_n[GGML_PROF_N_GRAPHS],
            (long long) g_prof_n[GGML_PROF_N_SPLIT_INPUTS], (long long) g_prof_n[GGML_PROF_N_COPIES],
            (long long) g_prof_n[GGML_PROF_N_SYNCS]);
    for (int i = 0; i < GGML_PROF_N_SLOTS; i++) {
        fprintf(stderr, "ggml profile:   %-28s %8.3f s\n", names[i], g_prof_us[i]/1e6);
    }
    fprintf(stderr, "ggml profile:   %-28s %8lld captured, %lld replayed, %lld not used\n", "cuda graphs",
            (long long) g_prof_n[GGML_PROF_N_CUDA_CAPTURED], (long long) g_prof_n[GGML_PROF_N_CUDA_REPLAYED],
            (long long) g_prof_n[GGML_PROF_N_CUDA_DISABLED]);
    for (int i = 0; i < 4; i++) {
        fprintf(stderr, "ggml profile:   %-28s %8lld\n", reasons[i],
                (long long) g_prof_n[GGML_PROF_N_CUDA_DISABLED_P0 + i]);
    }
    fflush(stderr);
}

// ------------------------------------------------------------------------------------------
// The hand-off timers (GGML_PROFILE_HANDOFF=1). Read once, on the first call (a function-local static,
// so the order of static initialisers across libraries does not matter); the scheduler's timed loop is a separate
// instantiation of ggml_backend_sched_compute_splits_impl, so with the variable unset nothing here runs. Every number is
// host time from ggml_time_us() around calls the untimed code makes anyway.
static bool ggml_handoff_read_env(void) {
    const char * e = getenv("GGML_PROFILE_HANDOFF");
    return e && atoi(e) != 0;
}

bool ggml_handoff_on(void) {
    static const bool on = ggml_handoff_read_env();
    return on;
}

// ------------------------------------------------------------------------------------------
// GGML_TRACE=<file>, the timeline (see ggml-backend.h). An event is only pushed into a buffer under one
// mutex (no file I/O, no allocation but the name's first interning); a WRITER THREAD swaps the buffer every 200 ms (or when 32,768 events are
// waiting) and writes it as Chrome trace "X" events, so no engine thread ever waits for the file (an earlier version wrote
// from the engine's threads, under the engine's lock, and stalled a decode word by 12-16 ms). A run that is killed loses at most the last 200 ms:
// the JSON array may be unterminated. Everything here runs only when the variable is set.
namespace {
struct ggml_trace_ev { int64_t t0, t1, arg; const char * name; int lane; };
struct ggml_trace_state {
    FILE *                         f = nullptr;
    std::mutex                     mu;
    std::vector<ggml_trace_ev>     buf;                 // producers push here (mu)
    std::vector<ggml_trace_ev>     out;                 // the writer thread's batch (only that thread, and the exit handler after the join)
    std::condition_variable        cv;
    std::thread                    th;
    bool                           stop = false;        // mu
    std::unordered_set<std::string_view> names;  // interned event names (node names change with the graph); the views point into name_store
    std::deque<std::string>        name_store;   // stable addresses
    std::set<int>                  lanes;
    size_t                         bytes = 0, max_bytes = 512u << 20;
    int64_t                        n_events = 0, n_dropped = 0;
    std::atomic<bool>              full{false};
};
ggml_trace_state * g_trace = nullptr;

void ggml_trace_json_name(FILE * f, const char * s) {
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') { fputc('\\', f); fputc(*s, f); } else if ((unsigned char) *s >= 0x20) { fputc(*s, f); }
    }
}

const char * ggml_trace_lane_name(int kind) {
    switch (kind) {
        case 1: return "scheduler (host thread): splits, waits, copies";
        case 2: return "GPU piece: issued -> done (device)";
        case 3: return "CPU thread: wait for an expert slot";
        case 4: return "SSD read (I/O thread)";
        case 5: return "remap op";
        case 6: return "guess helper";
        case 7: return "read landed (as the waiting CPU saw it)";
        default: return "lane";
    }
}

// the writer thread only (st->out is its own); no lock held while it writes
void ggml_trace_write(ggml_trace_state * st) {
    if (!st->f) { st->out.clear(); return; }
    for (const ggml_trace_ev & e : st->out) {
        if (st->full) { st->n_dropped++; continue; }
        st->n_events++;
        if (st->lanes.insert(e.lane).second) {
            const int kind = e.lane/1000, sub = e.lane%1000;
            st->bytes += fprintf(st->f, ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":%d,\"args\":{\"name\":\"%s #%d\"}}"
                                        ",\n{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":1,\"tid\":%d,\"args\":{\"sort_index\":%d}}",
                                 e.lane, ggml_trace_lane_name(kind), sub, e.lane, e.lane);
        }
        st->bytes += fprintf(st->f, ",\n{\"name\":\"");
        ggml_trace_json_name(st->f, e.name);
        st->bytes += strlen(e.name) + fprintf(st->f, "\",\"ph\":\"X\",\"ts\":%lld,\"dur\":%lld,\"pid\":1,\"tid\":%d,\"args\":{\"a\":%lld}}",
                             (long long) e.t0, (long long) (e.t1 > e.t0 ? e.t1 - e.t0 : 0), e.lane, (long long) e.arg);
        if (st->bytes >= st->max_bytes) {
            st->full = true;
            fprintf(stderr, "ggml_trace: the file reached GGML_TRACE_MAX_MB (%zu MiB), later events are dropped\n", st->max_bytes >> 20);
        }
    }
    st->out.clear();
    fflush(st->f);
}

void ggml_trace_writer(ggml_trace_state * st) {
    std::unique_lock<std::mutex> lock(st->mu);
    for (;;) {
        st->cv.wait_for(lock, std::chrono::milliseconds(200), [&] { return st->stop || st->buf.size() >= 32768; });
        st->out.swap(st->buf);   // O(1); the producers go on with the (empty, already reserved) old out
        const bool stop = st->stop;
        lock.unlock();
        ggml_trace_write(st);
        lock.lock();
        if (stop) { break; }
    }
}

void ggml_trace_atexit(void) {
    ggml_trace_state * st = g_trace;
    if (!st) { return; }
    {
        std::lock_guard<std::mutex> lock(st->mu);
        st->stop = true;
    }
    st->cv.notify_all();
    if (st->th.joinable()) { st->th.join(); }   // the writer writes what is left, then ends; later events are dropped
    if (st->f) {
        fprintf(st->f, "\n]\n");
        fclose(st->f);
        st->f = nullptr;
        fprintf(stderr, "ggml_trace: %lld events written (%lld dropped)\n", (long long) st->n_events, (long long) st->n_dropped);
    }
}

bool ggml_trace_read_env(void) {
    const char * p = getenv("GGML_TRACE");
    if (!p || !*p) { return false; }
    FILE * f = fopen(p, "w");
    if (!f) { fprintf(stderr, "ggml_trace: cannot open '%s'\n", p); return false; }
    g_trace = new ggml_trace_state();
    g_trace->f = f;
    g_trace->buf.reserve(1u << 17);   // so that a push never reallocates under the mutex
    g_trace->out.reserve(1u << 17);
    if (const char * m = getenv("GGML_TRACE_MAX_MB")) { if (atoi(m) > 0) { g_trace->max_bytes = (size_t) atoi(m) << 20; } }
    fprintf(f, "[\n{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"args\":{\"name\":\"TokenWing (ts = CLOCK_MONOTONIC us)\"}}");
    g_trace->th = std::thread(ggml_trace_writer, g_trace);
    atexit(ggml_trace_atexit);
    fprintf(stderr, "ggml_trace: on, writing '%s' (cap %zu MiB)\n", p, g_trace->max_bytes >> 20);
    return true;
}
} // namespace

bool ggml_trace_on(void) {
    static const bool on = ggml_trace_read_env();
    return on;
}

void ggml_trace_event(int lane, const char * name, int64_t t0_us, int64_t t1_us, int64_t arg) {
    if (!ggml_trace_on()) { return; }
    ggml_trace_state * st = g_trace;
    bool wake = false;
    {
        std::lock_guard<std::mutex> lock(st->mu);
        if (st->full || st->stop) { st->n_dropped++; return; }
        auto it = st->names.find(std::string_view(name));   // no string is built for a name that is already known
        if (it == st->names.end()) {
            st->name_store.emplace_back(name);
            it = st->names.insert(std::string_view(st->name_store.back())).first;
        }
        const char * nm = it->data();   // null-terminated: the view covers a whole std::string
        st->buf.push_back({ t0_us, t1_us, arg, nm, lane });
        wake = st->buf.size() == 32768;   // once, when the batch is big; the writer wakes by itself every 200 ms
    }
    if (wake) { st->cv.notify_one(); }
}

// the places in compute_splits where the host waits, and the kinds of copies
enum ggml_handoff_wait_kind {
    HO_W_NOINPUT = 0, // a split with no inputs waits for the previous split's backend
    HO_W_BATCHED,     // the batched wait (GGML_SCHED_SYNC_BATCH)
    HO_W_USER_INPUT,  // before a user input (GGML_TENSOR_FLAG_INPUT) is copied
    HO_W_WAR,         // the split backend finished reading the input copy that is overwritten (write after read)
    HO_W_MOE_INPUT,   // the used-experts path: the backend that holds the expert weights
    HO_W_MOE_IDS,     // the used-experts path: the ids readback (get_async + sync)
    HO_W_DEP_IN,      // a cross-backend input without an async copy: its producer backend
    HO_W_DEP_SPLIT,   // the same path: the split backend (event or sync) before the copy
    HO_W_HOOK,        // the input hook provided the split's inputs (its wait for the producer + the copies)
    HO_W_GUARD,       // the async-h2d graph-shape guard synchronized the GPU before the hook
    HO_W_N,
};
static const char * const g_ho_wait_names[HO_W_N] = {
    "noinput", "batched", "user_input", "war", "moe_input", "moe_ids", "dep_in", "dep_split", "hook", "guard",
};
// async_fail = a cpy_tensor_async attempt that returned false (the sync path follows; its time is not a copy)
enum ggml_handoff_copy_kind { HO_C_USER = 0, HO_C_EXPERTS, HO_C_ASYNC, HO_C_SYNC, HO_C_ASYNC_FAIL, HO_C_ASYNC_H2D, HO_C_N };
static const char * const g_ho_copy_names[HO_C_N] = { "user_input", "experts", "async", "sync", "async_fail", "async_h2d" };
static const char * const g_ho_call_names[GGML_HANDOFF_N_CALL] = { "prompt", "decode", "other" };
static const char * const g_ho_ctx_names[GGML_HANDOFF_N_CTX] = {
    "build", "reuse", "set_input", "compute", "output", "sync",
};
static const int64_t HO_LONG_US = 20; // a wait at least this long counts as "long" (the host really blocked)

struct ggml_handoff_split_rec {        // one split in one call
    int64_t host_us, compute_us, cpu_agg_us;
    int64_t wn[HO_W_N], wus[HO_W_N], wlong[HO_W_N];
    int64_t cn[HO_C_N], cus[HO_C_N];
};

struct ggml_handoff_split_acc {        // one split index over the calls of a kind
    std::string backend, first;
    bool        is_cpu = false;
    int64_t     n = 0, name_changes = 0;
    ggml_handoff_split_rec s = {};
};

struct ggml_handoff_node_acc {
    std::string op, name;
    int         is_mmid = 0;
    int64_t     n = 0, n_fused = 0, wall = 0, stagger = 0, work_sum = 0, work_max = 0, spread = 0, thread_n = 0;
};

struct ggml_handoff_kind_acc {
    int64_t calls = 0, tokens = 0;
    int64_t ctx_us[GGML_HANDOFF_N_CTX] = {}, ctx_n[GGML_HANDOFF_N_CTX] = {};
    int64_t decode_us = 0, cycle_us = 0, cycles = 0, cycle_inside_us = 0, print_us = 0, sync_in_us = 0, sync_in_n = 0;
    int64_t sched_n = 0, sched_us = 0, splits_us = 0, splits_n = 0, splits_shape_changes = 0, book_us = 0;
    int     first_n_splits = -1;
    ggml_handoff_split_rec tot = {};   // all splits summed
    int64_t cmp_cpu_us = 0, cmp_cpu_n = 0, cmp_gpu_us = 0, cmp_gpu_n = 0;
    int64_t cpu_graphs = 0, cpu_total_us = 0, cpu_fork_us = 0, cpu_join_us = 0, cpu_fork_stagger_us = 0,
            cpu_nodes_us = 0, cpu_agg_us = 0;
    int64_t prof_n[GGML_PROF_N_COUNTS] = {};
    std::vector<ggml_handoff_split_acc> splits;
    std::unordered_map<std::string, ggml_handoff_node_acc> nodes;
    std::unordered_map<std::string, ggml_handoff_node_acc> ops;   // the same, summed per op type (unnamed nodes included)
};

struct ggml_handoff_call_state {       // the call in progress (one "handoff: call" line each)
    bool    active  = false;   // a call line is pending (from the call's begin to the next call's begin or the dump)
    bool    in_call = false;   // between the call's begin and end: scheduler / context work is booked to the call
    int64_t sync_in_us = 0;    // synchronize() inside decode() (part of decode_us); ctx_us[SYNC] = the ones after it
    int     kind = GGML_HANDOFF_OTHER, n_tok = 0, reused = -1, n_splits = 0;
    int64_t index = 0, t0 = 0, decode_us = -1;
    int64_t ctx_us[GGML_HANDOFF_N_CTX] = {};
    int64_t sched_us = 0, splits_us = 0, cmp_cpu_us = 0, cmp_gpu_us = 0, cpu_graph_us = 0, cpu_nodes_us = 0,
            cpu_agg_us = 0, book_us = 0;
    ggml_handoff_split_rec tot = {};
    int64_t prof0[GGML_PROF_N_COUNTS] = {};
};

static std::mutex                g_ho_mutex;
static ggml_handoff_kind_acc     g_ho[GGML_HANDOFF_N_CALL];
static ggml_handoff_call_state   g_hc;
static int64_t                   g_ho_n_calls[GGML_HANDOFF_N_CALL] = {};
// per thread: the timed scheduler loop's per-split records, and the CPU graphs' bookkeeping time on this
// thread (the CPU backend computes on the scheduler's thread), so two contexts on two threads do not share them
static thread_local std::vector<ggml_handoff_split_rec> t_ho_recs;
static thread_local int64_t                              t_ho_cpu_agg_us = 0;

// where scheduler and context work is booked: the call between its begin and end, else "other" (mutex held)
// Decode words' expert-step and remap nodes, apart for nodes where a thread waited for an expert slot inside the
// expert step (ggml-cpu's slot wait: an SSD read or a VRAM copy-back still landing - that idle is the wait) and nodes with no such wait
// (that idle is the threads' imbalance): [n, idle us, wall us] x 2, work_max us. A remap op's own waits (the engine's lock / its
// read waits without --moe-stream-overlap) are not slot waits: its idle is the other threads waiting for thread 0, whatever it did
static std::map<std::string, std::array<int64_t, 7>> g_ho_idle;

static int ggml_handoff_kind_now(void) {
    return g_hc.in_call ? g_hc.kind : GGML_HANDOFF_OTHER;
}

static void ggml_handoff_rec_add(ggml_handoff_split_rec & a, const ggml_handoff_split_rec & b) {
    a.host_us += b.host_us; a.compute_us += b.compute_us; a.cpu_agg_us += b.cpu_agg_us;
    for (int i = 0; i < HO_W_N; i++) { a.wn[i] += b.wn[i]; a.wus[i] += b.wus[i]; a.wlong[i] += b.wlong[i]; }
    for (int i = 0; i < HO_C_N; i++) { a.cn[i] += b.cn[i]; a.cus[i] += b.cus[i]; }
}

static int64_t ggml_handoff_rec_wait_us(const ggml_handoff_split_rec & r) {
    int64_t s = 0; for (int i = 0; i < HO_W_N; i++) s += r.wus[i]; return s;
}
static int64_t ggml_handoff_rec_wait_n(const ggml_handoff_split_rec & r) {
    int64_t s = 0; for (int i = 0; i < HO_W_N; i++) s += r.wn[i]; return s;
}
static int64_t ggml_handoff_rec_copy_us(const ggml_handoff_split_rec & r) {
    int64_t s = 0; for (int i = 0; i < HO_C_N; i++) s += r.cus[i]; return s;
}
static int64_t ggml_handoff_rec_copy_n(const ggml_handoff_split_rec & r) {
    int64_t s = 0; for (int i = 0; i < HO_C_N; i++) s += r.cn[i]; return s;
}

// close the call in progress: book its cycle (this call's start -> the next call's start) and print its line.
// Called with the mutex held.
static void ggml_handoff_finish_call(int64_t t_next) {
    if (!g_hc.active) {
        return;
    }
    ggml_handoff_kind_acc & k = g_ho[g_hc.kind];
    const int64_t cycle = t_next > 0 ? t_next - g_hc.t0 : -1;
    if (cycle >= 0) {
        k.cycle_us        += cycle;
        k.cycles          += 1;
        // ctx_us[SYNC] holds only the synchronize() calls after decode(); the ones inside are in decode_us
        k.cycle_inside_us += std::max<int64_t>(g_hc.decode_us, 0) + g_hc.ctx_us[GGML_HANDOFF_C_SYNC];
    }
    const ggml_handoff_split_rec & t = g_hc.tot;
    fprintf(stderr, "handoff: call i=%lld kind=%s n_tok=%d cycle_us=%lld decode_us=%lld build_us=%lld reuse_us=%lld "
            "reused=%d set_input_us=%lld compute_us=%lld sched_us=%lld splits_us=%lld n_splits=%d waits=%lld wait_us=%lld "
            "copies=%lld copy_us=%lld cmp_cpu_us=%lld cmp_gpu_us=%lld cpu_graph_us=%lld cpu_nodes_us=%lld cpu_agg_us=%lld "
            "book_us=%lld output_us=%lld sync_us=%lld sync_in_us=%lld w_us=",
            (long long) g_hc.index, g_ho_call_names[g_hc.kind], g_hc.n_tok, (long long) cycle, (long long) g_hc.decode_us,
            (long long) g_hc.ctx_us[GGML_HANDOFF_C_BUILD], (long long) g_hc.ctx_us[GGML_HANDOFF_C_REUSE], g_hc.reused,
            (long long) g_hc.ctx_us[GGML_HANDOFF_C_SET_INPUT], (long long) g_hc.ctx_us[GGML_HANDOFF_C_COMPUTE],
            (long long) g_hc.sched_us, (long long) g_hc.splits_us, g_hc.n_splits,
            (long long) ggml_handoff_rec_wait_n(t), (long long) ggml_handoff_rec_wait_us(t),
            (long long) ggml_handoff_rec_copy_n(t), (long long) ggml_handoff_rec_copy_us(t),
            (long long) g_hc.cmp_cpu_us, (long long) g_hc.cmp_gpu_us, (long long) g_hc.cpu_graph_us,
            (long long) g_hc.cpu_nodes_us, (long long) g_hc.cpu_agg_us, (long long) g_hc.book_us,
            (long long) g_hc.ctx_us[GGML_HANDOFF_C_OUTPUT], (long long) g_hc.ctx_us[GGML_HANDOFF_C_SYNC],
            (long long) g_hc.sync_in_us);
    for (int i = 0; i < HO_W_N; i++) {
        fprintf(stderr, "%s%lld", i ? "," : "", (long long) t.wus[i]);
    }
    fprintf(stderr, " w_n=");
    for (int i = 0; i < HO_W_N; i++) {
        fprintf(stderr, "%s%lld", i ? "," : "", (long long) t.wn[i]);
    }
    fprintf(stderr, "\n");
    g_hc.active = false;
}

void ggml_handoff_call_begin(int kind, int n_tokens) {
    if (!ggml_handoff_on()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_ho_mutex);
    const int64_t t_now = ggml_time_us();
    const int prev_kind = g_hc.kind;
    const bool had_prev = g_hc.active;
    ggml_handoff_finish_call(t_now);
    if (had_prev) {
        g_ho[prev_kind].print_us += ggml_time_us() - t_now;   // the line above, booked to no call
    }
    kind = kind >= 0 && kind < GGML_HANDOFF_N_CALL ? kind : GGML_HANDOFF_OTHER;
    g_hc = ggml_handoff_call_state();
    g_hc.active  = true;
    g_hc.in_call = true;
    g_hc.kind   = kind;
    g_hc.n_tok  = n_tokens;
    g_hc.index  = g_ho_n_calls[kind]++;
    for (int i = 0; i < GGML_PROF_N_COUNTS; i++) {
        g_hc.prof0[i] = g_prof_n[i];
    }
    g_ho[kind].calls  += 1;
    g_ho[kind].tokens += n_tokens;
    g_hc.t0 = ggml_time_us();
}

void ggml_handoff_call_end(int64_t decode_us) {
    if (!ggml_handoff_on()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_ho_mutex);
    if (g_hc.in_call) {
        g_hc.decode_us = decode_us;
        ggml_handoff_kind_acc & k = g_ho[g_hc.kind];
        k.decode_us += decode_us;
        for (int i = 0; i < GGML_PROF_N_COUNTS; i++) {
            k.prof_n[i] += g_prof_n[i] - g_hc.prof0[i];   // GGML_PROFILE's own counters over this call (0 when it is unset)
        }
        g_hc.in_call = false;   // from here on only synchronize() is booked to this call, the rest to "other"
    }
}

void ggml_handoff_ctx_add(int what, int64_t us) {
    if (!ggml_handoff_on() || what < 0 || what >= GGML_HANDOFF_N_CTX) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_ho_mutex);
    if (what == GGML_HANDOFF_C_SYNC && g_hc.in_call) {
        // A synchronize() inside decode() (embd_seq, output_reserve, ...) is part of decode_us - booked apart
        g_ho[g_hc.kind].sync_in_us += us;
        g_ho[g_hc.kind].sync_in_n  += 1;
        g_hc.sync_in_us += us;
        return;
    }
    // A synchronize() after decode() and before the next call (the host block on the call's last device work)
    // belongs to the call; any other work outside a call is "other"
    const bool to_call = g_hc.in_call || (what == GGML_HANDOFF_C_SYNC && g_hc.active);
    ggml_handoff_kind_acc & k = g_ho[to_call ? g_hc.kind : GGML_HANDOFF_OTHER];
    k.ctx_us[what] += us;
    k.ctx_n[what]  += 1;
    if (to_call) {
        g_hc.ctx_us[what] += us;
        if (what == GGML_HANDOFF_C_BUILD) g_hc.reused = 0;
        if (what == GGML_HANDOFF_C_REUSE) g_hc.reused = 1;
    }
}

// a name as one token of a "handoff:" line (tensor names can hold spaces, e.g. "x (view)")
static std::string ggml_handoff_token(const std::string & name) {
    std::string s = name.empty() ? "-" : name;
    std::replace(s.begin(), s.end(), ' ', '_');
    return s;
}

static std::string ggml_handoff_strip_layer(const char * name) {
    // "ffn_moe_down-12" -> "ffn_moe_down", "ffn_moe_down-12 (copy)" -> "ffn_moe_down (copy)"
    std::string s = name ? name : "";
    const size_t dash = s.rfind('-');
    if (dash != std::string::npos) {
        size_t e = dash + 1;
        while (e < s.size() && s[e] >= '0' && s[e] <= '9') e++;
        if (e > dash + 1) {
            s.erase(dash, e - dash);
        }
    }
    return s;
}

void ggml_handoff_cpu_graph(int64_t total_us, int64_t fork_us, int64_t join_us, int64_t fork_stagger_us, int n_threads,
                            const struct ggml_handoff_cpu_node * nodes, int n_nodes) {
    if (!ggml_handoff_on()) {
        return;
    }
    const int64_t t_agg0 = ggml_time_us();
    std::lock_guard<std::mutex> lock(g_ho_mutex);
    ggml_handoff_kind_acc & k = g_ho[ggml_handoff_kind_now()];
    int64_t nodes_us = 0;
    if (ggml_handoff_kind_now() == GGML_HANDOFF_DECODE) {
        for (int i = 0; i < n_nodes; i++) {
            const ggml_handoff_cpu_node & nd = nodes[i];
            if (!nd.is_mmid && (nd.op == nullptr || strncmp(nd.op, "MAP_CUSTOM", 10) != 0)) {
                continue;
            }
            auto & a = g_ho_idle[std::string(nd.op ? nd.op : "?") + " " + ggml_handoff_strip_layer(nd.name)];
            const int c = nd.slot_wait_max_us > 0 ? 3 : 0;
            a[c + 0] += 1; a[c + 1] += nd.idle_us; a[c + 2] += nd.wall_us; a[6] += nd.work_max_us;
        }
    }
    for (int i = 0; i < n_nodes; i++) {
        const ggml_handoff_cpu_node & nd = nodes[i];
        std::string key = std::string(nd.op ? nd.op : "?") + " " + ggml_handoff_strip_layer(nd.name);
        ggml_handoff_node_acc & a = k.nodes[key];
        if (a.n == 0) {
            a.op      = ggml_handoff_token(nd.op ? nd.op : "?");
            a.name    = ggml_handoff_token(ggml_handoff_strip_layer(nd.name));
            a.is_mmid = nd.is_mmid;
        }
        a.n        += 1;
        a.n_fused  += nd.n_fused;
        a.wall     += nd.wall_us;
        a.stagger  += nd.stagger_us;
        a.work_sum += nd.work_sum_us;
        a.work_max += nd.work_max_us;
        a.spread   += nd.spread_us;
        a.thread_n += n_threads;
        nodes_us   += nd.wall_us;

        ggml_handoff_node_acc & o = k.ops[nd.op ? nd.op : "?"];   // per op type, e.g. all ADD nodes of a word
        if (o.n == 0) {
            o.op      = ggml_handoff_token(nd.op ? nd.op : "?");
            o.is_mmid = nd.is_mmid;
        }
        o.n        += 1;
        o.n_fused  += nd.n_fused;
        o.wall     += nd.wall_us;
        o.stagger  += nd.stagger_us;
        o.work_sum += nd.work_sum_us;
        o.work_max += nd.work_max_us;
        o.spread   += nd.spread_us;
        o.thread_n += n_threads;
    }
    k.cpu_graphs          += 1;
    k.cpu_total_us        += total_us;
    k.cpu_fork_us         += fork_us;
    k.cpu_join_us         += join_us;
    k.cpu_fork_stagger_us += fork_stagger_us;
    k.cpu_nodes_us        += nodes_us;
    const int64_t agg = ggml_time_us() - t_agg0;
    k.cpu_agg_us          += agg;
    t_ho_cpu_agg_us       += agg;   // read by this thread's timed scheduler loop
    if (g_hc.in_call) {
        g_hc.cpu_graph_us += total_us;
        g_hc.cpu_nodes_us += nodes_us;
        g_hc.cpu_agg_us   += agg;
    }
}

// one compute_splits call: its splits' records (from the timed loop)
static void ggml_handoff_sched_done(ggml_backend_sched_t sched, const std::vector<ggml_handoff_split_rec> & recs,
                                    int64_t sched_us);

static void ggml_handoff_print_kind(int kind) {
    const ggml_handoff_kind_acc & k = g_ho[kind];
    if (k.calls == 0) {
        return;
    }
    const double W = (double) k.calls;   // per call (= per word for decode)
    const char * kn = g_ho_call_names[kind];
    fprintf(stderr, "handoff: == summary kind=%s calls=%lld tokens=%lld (values are per call%s) ==\n", kn,
            (long long) k.calls, (long long) k.tokens, kind == GGML_HANDOFF_DECODE ? " = per word" : "");
    const double cyc = k.cycles ? k.cycle_us/1e3/k.cycles : 0.0;
    const double outside = k.cycles ? (k.cycle_us - k.cycle_inside_us)/1e3/k.cycles : 0.0;
    double ctx_sum = (double) k.sync_in_us;   // synchronize() inside decode() is part of decode_us
    for (int i = 0; i < GGML_HANDOFF_N_CTX; i++) {
        if (i != GGML_HANDOFF_C_SYNC) ctx_sum += k.ctx_us[i];
    }
    fprintf(stderr, "handoff: ctx kind=%s cycle_ms=%.4f cycles=%lld decode_ms=%.4f outside_ms=%.4f decode_untimed_ms=%.4f "
            "print_ms=%.4f sync_in_ms=%.4f sync_in_n=%lld builds=%lld reuse_hits=%lld", kn, cyc, (long long) k.cycles,
            k.decode_us/1e3/W, outside, (k.decode_us - ctx_sum)/1e3/W, k.print_us/1e3/W, k.sync_in_us/1e3/W,
            (long long) k.sync_in_n,
            (long long) k.ctx_n[GGML_HANDOFF_C_BUILD], (long long) k.ctx_n[GGML_HANDOFF_C_REUSE]);
    for (int i = 0; i < GGML_HANDOFF_N_CTX; i++) {
        fprintf(stderr, " %s_ms=%.4f", g_ho_ctx_names[i], k.ctx_us[i]/1e3/W);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "handoff: sched kind=%s graphs=%.3f sched_ms=%.4f splits=%.3f splits_host_ms=%.4f between_splits_ms=%.4f "
            "shape_changes=%lld cmp_cpu_ms=%.4f cmp_cpu_n=%.3f cmp_gpu_ms=%.4f cmp_gpu_n=%.3f wait_ms=%.4f waits=%.3f "
            "copy_ms=%.4f copies=%.3f book_ms=%.4f\n", kn, k.sched_n/W, k.sched_us/1e3/W, k.splits_n/W, k.splits_us/1e3/W,
            (k.sched_us - k.splits_us)/1e3/W, (long long) k.splits_shape_changes, k.cmp_cpu_us/1e3/W, k.cmp_cpu_n/W,
            k.cmp_gpu_us/1e3/W, k.cmp_gpu_n/W, ggml_handoff_rec_wait_us(k.tot)/1e3/W, ggml_handoff_rec_wait_n(k.tot)/W,
            ggml_handoff_rec_copy_us(k.tot)/1e3/W, ggml_handoff_rec_copy_n(k.tot)/W, k.book_us/1e3/W);
    for (int i = 0; i < HO_W_N; i++) {
        fprintf(stderr, "handoff: wait kind=%s type=%s n=%.3f ms=%.4f long_n=%.3f\n", kn, g_ho_wait_names[i],
                k.tot.wn[i]/W, k.tot.wus[i]/1e3/W, k.tot.wlong[i]/W);
    }
    for (int i = 0; i < HO_C_N; i++) {
        if (i == HO_C_ASYNC_H2D && !g_ho_async_h2d_on) {
            continue;   // the async_h2d kind only when --moe-stream-async-h2d is on
        }
        fprintf(stderr, "handoff: copy kind=%s type=%s n=%.3f ms=%.4f\n", kn, g_ho_copy_names[i], k.tot.cn[i]/W,
                k.tot.cus[i]/1e3/W);
    }
    for (size_t i = 0; i < k.splits.size(); i++) {
        const ggml_handoff_split_acc & s = k.splits[i];
        if (s.n == 0) {
            continue;
        }
        const double other = (s.s.host_us - s.s.compute_us - ggml_handoff_rec_wait_us(s.s) - ggml_handoff_rec_copy_us(s.s))/1e3/W;
        fprintf(stderr, "handoff: split kind=%s idx=%zu backend=%s cpu=%d first=%s n=%lld name_changes=%lld host_ms=%.4f "
                "compute_ms=%.4f wait_ms=%.4f copy_ms=%.4f other_ms=%.4f waits=", kn, i, s.backend.c_str(), s.is_cpu ? 1 : 0,
                s.first.c_str(), (long long) s.n, (long long) s.name_changes, s.s.host_us/1e3/W, s.s.compute_us/1e3/W,
                ggml_handoff_rec_wait_us(s.s)/1e3/W, ggml_handoff_rec_copy_us(s.s)/1e3/W, other);
        bool any = false;
        for (int w = 0; w < HO_W_N; w++) {
            if (s.s.wn[w]) {
                fprintf(stderr, "%s%s:%.3f/%.4f", any ? "," : "", g_ho_wait_names[w], s.s.wn[w]/W, s.s.wus[w]/1e3/W);
                any = true;
            }
        }
        fprintf(stderr, "%s copies=", any ? "" : "-");
        any = false;
        for (int c = 0; c < HO_C_N; c++) {
            if (s.s.cn[c]) {
                fprintf(stderr, "%s%s:%.3f/%.4f", any ? "," : "", g_ho_copy_names[c], s.s.cn[c]/W, s.s.cus[c]/1e3/W);
                any = true;
            }
        }
        fprintf(stderr, "%s\n", any ? "" : "-");
    }
    if (k.cpu_graphs > 0) {
        fprintf(stderr, "handoff: cpugraph kind=%s graphs=%.3f total_ms=%.4f nodes_wall_ms=%.4f fork_ms=%.4f join_ms=%.4f "
                "fork_stagger_us=%.2f agg_ms=%.4f\n", kn, k.cpu_graphs/W, k.cpu_total_us/1e3/W, k.cpu_nodes_us/1e3/W,
                k.cpu_fork_us/1e3/W, k.cpu_join_us/1e3/W, (double) k.cpu_fork_stagger_us/k.cpu_graphs, k.cpu_agg_us/1e3/W);
        std::vector<const ggml_handoff_node_acc *> v;
        for (const auto & it : k.nodes) {
            v.push_back(&it.second);
        }
        std::sort(v.begin(), v.end(), [](const ggml_handoff_node_acc * a, const ggml_handoff_node_acc * b) {
            return a->wall > b->wall;
        });
        int shown = 0;
        for (const ggml_handoff_node_acc * a : v) {
            if (shown >= 25 && !a->is_mmid) {
                continue;   // the top 25 by wall time, and every MUL_MAT_ID row
            }
            shown++;
            fprintf(stderr, "handoff: cpunode kind=%s mmid=%d op=%s name=%s n=%.3f fused=%.3f wall_ms=%.4f wall_us_each=%.2f "
                    "stagger_us=%.2f spread_us=%.2f work_avg_us=%.2f work_max_us=%.2f\n", kn, a->is_mmid, a->op.c_str(),
                    a->name.c_str(), a->n/W, a->n_fused/W, a->wall/1e3/W, (double) a->wall/a->n, (double) a->stagger/a->n,
                    (double) a->spread/a->n, (double) a->work_sum/a->thread_n, (double) a->work_max/a->n);
        }
    }
    // Every op type's total (a node = one barrier of the CPU graph loop)
    std::vector<const ggml_handoff_node_acc *> vo;
    for (const auto & it : k.ops) {
        vo.push_back(&it.second);
    }
    std::sort(vo.begin(), vo.end(), [](const ggml_handoff_node_acc * a, const ggml_handoff_node_acc * b) {
        return a->wall > b->wall;
    });
    for (const ggml_handoff_node_acc * a : vo) {
        fprintf(stderr, "handoff: cpuop kind=%s op=%s n=%.3f fused=%.3f wall_ms=%.4f wall_us_each=%.2f stagger_us=%.2f "
                "spread_us=%.2f work_avg_us=%.2f work_max_us=%.2f\n", kn, a->op.c_str(), a->n/W, a->n_fused/W, a->wall/1e3/W,
                (double) a->wall/a->n, (double) a->stagger/a->n, (double) a->spread/a->n, (double) a->work_sum/a->thread_n,
                (double) a->work_max/a->n);
    }
    if (ggml_prof_on()) {
        const int64_t cap = k.prof_n[GGML_PROF_N_CUDA_CAPTURED], rep = k.prof_n[GGML_PROF_N_CUDA_REPLAYED],
                      dis = k.prof_n[GGML_PROF_N_CUDA_DISABLED];
        fprintf(stderr, "handoff: cudagraph kind=%s captured=%.3f replayed=%.3f not_used=%.3f replay_share=%.2f%% "
                "(GGML_PROFILE; not used: policy %.3f, incompatible %.3f, warmup %.3f, properties %.3f)\n", kn, cap/W, rep/W,
                dis/W, cap + rep + dis ? 100.0*rep/(cap + rep + dis) : 0.0, k.prof_n[GGML_PROF_N_CUDA_DISABLED_P0]/W,
                k.prof_n[GGML_PROF_N_CUDA_DISABLED_P1]/W, k.prof_n[GGML_PROF_N_CUDA_DISABLED_P2]/W,
                k.prof_n[GGML_PROF_N_CUDA_DISABLED_P3]/W);
        fprintf(stderr, "handoff: gprof kind=%s graphs=%.3f splits=%.3f split_inputs=%.3f copies=%.3f syncs=%.3f "
                "(GGML_PROFILE's counters over the same calls)\n", kn, k.prof_n[GGML_PROF_N_GRAPHS]/W,
                k.prof_n[GGML_PROF_N_SPLITS]/W, k.prof_n[GGML_PROF_N_SPLIT_INPUTS]/W, k.prof_n[GGML_PROF_N_COPIES]/W,
                k.prof_n[GGML_PROF_N_SYNCS]/W);
    } else {
        fprintf(stderr, "handoff: cudagraph kind=%s n/a (set GGML_PROFILE=1 for the CUDA-graph counters)\n", kn);
    }
}

void ggml_handoff_dump(void) {
    if (!ggml_handoff_on()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_ho_mutex);
    ggml_handoff_finish_call(-1);   // the last call has no next call: no cycle
    fprintf(stderr, "handoff: waits counted as long when >= %lld us; ms values are per call (decode: per word)\n",
            (long long) HO_LONG_US);
    fprintf(stderr, "handoff: the totals are process-wide: with several contexts (server slots, a draft model) their calls are "
            "booked together\n");
    ggml_handoff_print_kind(GGML_HANDOFF_DECODE);
    {   // the idle of the threads, per decode word
        const double w = g_ho_n_calls[GGML_HANDOFF_DECODE] > 0 ? (double) g_ho_n_calls[GGML_HANDOFF_DECODE] : 1.0;
        for (const auto & kv : g_ho_idle) {
            const auto & a = kv.second;
            fprintf(stderr, "handoff: idle kind=decode node=%s no_slot_wait_n=%.2f idle_us_each=%.2f wall_us_each=%.2f "
                    "slot_wait_n=%.2f idle_us_each=%.2f wall_us_each=%.2f work_max_us_each=%.2f idle_ms_no_slot_wait=%.4f "
                    "idle_ms_slot_wait=%.4f\n", kv.first.c_str(),
                    a[0]/w, a[0] ? (double) a[1]/a[0] : 0.0, a[0] ? (double) a[2]/a[0] : 0.0,
                    a[3]/w, a[3] ? (double) a[4]/a[3] : 0.0, a[3] ? (double) a[5]/a[3] : 0.0,
                    (a[0] + a[3]) ? (double) a[6]/(a[0] + a[3]) : 0.0, a[1]/1000.0/w, a[4]/1000.0/w);
        }
        g_ho_idle.clear();
    }
    ggml_handoff_print_kind(GGML_HANDOFF_PROMPT);
    ggml_handoff_print_kind(GGML_HANDOFF_OTHER);
    fprintf(stderr, "handoff: == end ==\n");
    fflush(stderr);
    for (auto & k : g_ho) {   // a second dump (a second context) prints only what came after
        k = ggml_handoff_kind_acc();
    }
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

#define GGML_SCHED_USER_INP_RING 4

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;
    bool alloc_failed; // the last ggml_backend_sched_alloc_graph split its graph and could not allocate it (ggml_backend_sched_alloc_graph_again may plan it again)

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    ggml_backend_sched_input_hook_t input_hook;
    void * input_hook_user_data;
    ggml_backend_sched_split_hook_t split_hook;
    void * split_hook_user_data;
    ggml_backend_sched_copy_hook_t copy_hook;
    void * copy_hook_user_data;
    int64_t t_place_us, t_realloc_sync_us, t_realloc_plan_us;   // the place hook's time; a re-plan's GPU wait / its planning
    int     n_realloc;
    // A split input that a finished CPU split wrote into the GPU's pinned host
    // buffer is copied with an async H2D on the GPU split's own stream - no host wait (see ggml_backend_sched_async_host_input_ok)
    bool async_host_input;
    int  n_async_gpus;    // the GPU backends of this scheduler (async_host_input's per-GPU rule with more than one)
    int  test_h2d_late;   // GGML_SCHED_TEST_H2D_LATE (test only): the async copies are enqueued after the split's graph
    ggml_backend_event_t  async_h2d_ev[GGML_SCHED_MAX_BACKENDS];
    bool                  async_h2d_pending[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_t user_inp_buf[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_USER_INP_RING];
    size_t                user_inp_cap[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_USER_INP_RING];
    ggml_backend_event_t  user_inp_ev[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_USER_INP_RING];
    bool                  user_inp_ev_pending[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_USER_INP_RING];
    int                   user_inp_cur[GGML_SCHED_MAX_BACKENDS];
    // Per split, the GPU split that produced the inputs an input hook provides (-2 =
    // not looked up yet, -1 = not found), looked up once per graph split; valid until the graph is split again
    int * hook_src_split;
    int   hook_src_cap;
    bool  hook_src_valid;   // hook_src_split holds this graph split's lookups (false after the graph is split again)

    ggml_backend_sched_realloc_callback callback_realloc;
    void * callback_realloc_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;
    int  op_offload_only;   // the backend index op offload may use, -1 = the first that takes the op

    int debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int graph_size;
    int prev_graph_size;

    int debug_realloc;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                const struct ggml_tensor * src_base = src;
                while (src_base->view_src != NULL) {
                    src_base = src_base->view_src;
                }
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 &&
                    ggml_backend_buffer_is_host(src->buffer) &&
                    src_base->extra == NULL && src->buffer->iface.get_tensor != NULL) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (sched->op_offload_only >= 0 && b != sched->op_offload_only) {
                            continue;
                        }
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;
    sched->alloc_failed = false;
    sched->hook_src_valid = false;
    if (sched->copy_hook != NULL && sched->n_copies == 1) {   // a new graph is split
        sched->copy_hook(sched->copy_hook_user_data, NULL, NULL, NULL, NULL, -1, graph);
    }

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            // A preset target (a view of memory the hook owns). Never with pipeline copies
                            if (sched->copy_hook != NULL && sched->n_copies == 1) {
                                sched->copy_hook(sched->copy_hook_user_data, backend, src, tensor_copy, node, i_split, graph);
                            }
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    if (sched->n_copies > 1) {
        for (int i = 0; i < graph->n_leafs; i++) {
            struct ggml_tensor * leaf = graph->leafs[i];
            if ((leaf->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
                continue;
            }

            const size_t leaf_id = hash_id(leaf);
            const int leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            if (tensor_id_copy(leaf_id, leaf_backend_id, 0) == NULL) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * tensor_copy;
                    if (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } else {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_input(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            int n_graph_inputs = sched->n_graph_inputs++;
            if (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    struct ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](void * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            if (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    int n_dep_nodes = 0;
    for (const auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_graph_size = sched->graph_size;
    sched->graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    int n_dep_nodes_added = 0;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            if (alloc_deps.empty()) {
                continue;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            if (it != alloc_deps.end()) {
                const std::vector<ggml_tensor *> & keep = it->second;
                for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    struct ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    assert(graph_copy->size > graph_copy->n_nodes);
                    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static void ggml_backend_sched_mark_backend_synced(ggml_backend_sched_t sched, int id) {
    if (id < 0 || id >= sched->n_backends) {
        return;
    }
    sched->async_h2d_pending[id] = false;
    for (int r = 0; r < GGML_SCHED_USER_INP_RING; r++) {
        sched->user_inp_ev_pending[id][r] = false;
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    // Try active plan + 16-slot multi-topology plan cache before replanning
    if (!ggml_gallocr_alloc_graph_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        bool backend_ids_changed = false;
        for (int i = 0; i < sched->graph.n_nodes; i++) {
            if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
                sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
        if (!backend_ids_changed) {
            for (int i = 0; i < sched->graph.n_leafs; i++) {
                if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                    sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                    backend_ids_changed = true;
                    break;
                }
            }
        }
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        const bool graph_size_changed = sched->prev_graph_size != sched->graph_size;

        if (sched->callback_realloc != NULL) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_size_changed, sched->callback_realloc_user_data);
        }

        if (sched->debug_realloc > 0) {
            const bool unexpected = !backend_ids_changed && !graph_size_changed;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // Only synchronize backends if a physical vbuffer actually needs to grow/reallocate!
        struct sched_realloc_sync_ctx {
            ggml_backend_sched_t sched;
            int64_t sync_us;
        } sctx = { sched, 0 };
        auto before_buf_realloc = [](void * ud) {
            auto * c = (sched_realloc_sync_ctx *) ud;
            const int64_t t_s0 = ggml_time_us();
            for (int i = 0; i < c->sched->n_backends; i++) {
                ggml_backend_synchronize(c->sched->backends[i]);
                ggml_backend_sched_mark_backend_synced(c->sched, i);
            }
            c->sync_us += ggml_time_us() - t_s0;
        };

        const int64_t t_p0 = ggml_time_us();
        if (!ggml_gallocr_reserve_n_ex(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, before_buf_realloc, &sctx)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
        const int64_t t_p1 = ggml_time_us();
        sched->t_realloc_sync_us += sctx.sync_us;
        sched->t_realloc_plan_us += (t_p1 - t_p0) - sctx.sync_us;
        sched->n_realloc         += 1;
        if (!ggml_gallocr_alloc_graph_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

// Book one compute_splits call of the timed loop (GGML_PROFILE_HANDOFF)
static void ggml_handoff_sched_done(ggml_backend_sched_t sched, const std::vector<ggml_handoff_split_rec> & recs,
                                    int64_t sched_us) {
    const int64_t t_book0 = ggml_time_us();
    std::lock_guard<std::mutex> lock(g_ho_mutex);
    ggml_handoff_kind_acc & k = g_ho[ggml_handoff_kind_now()];
    const int n = std::min<int>(sched->n_splits, (int) recs.size());
    if (k.first_n_splits < 0) {
        k.first_n_splits = n;
    } else if (k.first_n_splits != n) {
        k.splits_shape_changes++;
    }
    if ((int) k.splits.size() < n) {
        k.splits.resize(n);
    }
    k.sched_n  += 1;
    k.sched_us += sched_us;
    k.splits_n += n;
    if (g_hc.in_call) {
        g_hc.sched_us += sched_us;
        g_hc.n_splits += n;
    }
    for (int i = 0; i < n; i++) {
        const ggml_backend_sched_split & split = sched->splits[i];
        ggml_backend_t       backend = sched->backends[split.backend_id];
        ggml_backend_dev_t   dev     = ggml_backend_get_device(backend);
        const bool           is_cpu  = dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        const char *         first   = split.graph.n_nodes > 0 ? split.graph.nodes[0]->name : "";
        ggml_handoff_split_acc & a = k.splits[i];
        if (a.n == 0) {
            a.backend = ggml_handoff_token(ggml_backend_name(backend));
            a.first   = ggml_handoff_token(first);
            a.is_cpu  = is_cpu;
        } else if (a.first != ggml_handoff_token(first) || a.backend != ggml_handoff_token(ggml_backend_name(backend))) {
            a.name_changes++;
        }
        a.n++;
        ggml_handoff_rec_add(a.s, recs[i]);
        ggml_handoff_rec_add(k.tot, recs[i]);
        k.splits_us += recs[i].host_us;
        (is_cpu ? k.cmp_cpu_us : k.cmp_gpu_us) += recs[i].compute_us;
        (is_cpu ? k.cmp_cpu_n  : k.cmp_gpu_n)  += 1;
        if (g_hc.in_call) {
            ggml_handoff_rec_add(g_hc.tot, recs[i]);
            g_hc.splits_us += recs[i].host_us;
            (is_cpu ? g_hc.cmp_cpu_us : g_hc.cmp_gpu_us) += recs[i].compute_us;
        }
    }
    const int64_t book = ggml_time_us() - t_book0;   // this function, inside the context's compute timer
    k.book_us += book;
    if (g_hc.in_call) {
        g_hc.book_us += book;
    }
}

// HO = the hand-off timers (GGML_PROFILE_HANDOFF). Two instances; the dispatcher below picks one per call
// from a value read at start-up, so the untimed instance is the plain code with no timer branch in it.
// A split input may be copied with an async H2D on the split's GPU stream and no host wait when
// - its source backend is the CPU and the split's backend a GPU, and the scheduler keeps one copy of the inputs (no events);
// - its bytes are in that GPU's own pinned host buffer type (the CPU compute buffer llama-context gives the CPU backend) - an async
//   copy from it is a real DMA on the stream, ordered after the GPU work already queued there (so the split's previous reader of the
//   destination is done: no WAR wait) and before the split's graph launched next;
// - the CPU split that wrote it has finished: the CPU backend computes synchronously, before the scheduler gets here.
// The source must stay unchanged until the DMA has run: every later CPU write to that buffer in the same compute is a later CPU split,
// which starts only after a GPU event recorded later on the same stream (needed-early's "router done") or a stream synchronize (a GPU ->
// input copy) - both after this copy in stream order; across computes llama-context synchronizes before set_inputs (because
// nothing else does between two computes)
// The scheduler copies a host weight to a backend ONCE per graph (at the first split that reads it; later splits read the
// same copy), so a partial copy must hold the experts of every MUL_MAT_ID that reads the copy, in this split and in the later ones (a
// layer-major expert graph has one per chunk of the call; elsewhere there is one). One pass over the graph, made at the first partial copy:
// per copy the ids of its users, or "whole" when another kind of node reads it (or a view of it); per computed node the split that computes it.
struct ggml_backend_sched_moe_users {
    bool built = false;
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> ids;    // copy -> the ids tensors of its MUL_MAT_ID users
    std::unordered_map<const ggml_tensor *, bool>                       whole;  // copy -> another use: copy all of it
    std::unordered_map<const ggml_tensor *, int>                        split_of;   // computed node -> its split
    std::unordered_map<const ggml_tensor *, std::pair<ggml_tensor *, int>> src_of;  // a split's input copy -> (the input, its backend id)
};

// the ids tensors to read for the partial copy of input_cpy made at split split_id, each as (tensor, backend id) with an input copy replaced
// by the tensor it copies; false = copy the whole tensor (another kind of use, or ids that are computed in this split or a later one)
static bool ggml_backend_sched_moe_copy_ids(ggml_backend_sched_t sched, ggml_backend_sched_moe_users & U, int split_id,
        const ggml_tensor * input_cpy, std::vector<std::pair<ggml_tensor *, int>> & out) {
    if (!U.built) {
        U.built = true;
        for (int s = 0; s < sched->n_splits; s++) {
            const auto & sp = sched->splits[s];
            for (int k = 0; k < sp.n_inputs; k++) {
                const ggml_tensor * cpy = sched->hv_tensor_copies[hash_id(sp.inputs[k])*sched->n_backends*sched->n_copies +
                                                                  sp.backend_id*sched->n_copies + sched->cur_copy];
                if (cpy) {
                    U.src_of[cpy] = { sp.inputs[k], sched->hv_tensor_backend_ids[hash_id(sp.inputs[k])] };
                    U.ids.emplace(cpy, std::vector<ggml_tensor *>());
                }
            }
        }
        for (int s = 0; s < sched->n_splits; s++) {
            const ggml_cgraph & g = sched->splits[s].graph;
            for (int i = 0; i < g.n_nodes; i++) {
                ggml_tensor * n = g.nodes[i];
                if (n->op != GGML_OP_NONE && n->op != GGML_OP_VIEW && n->op != GGML_OP_RESHAPE && n->op != GGML_OP_PERMUTE &&
                        n->op != GGML_OP_TRANSPOSE) {
                    U.split_of[n] = s;
                }
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    const ggml_tensor * sj = n->src[j];
                    if (sj == nullptr) {
                        continue;
                    }
                    const ggml_tensor * c = U.ids.count(sj) ? sj : (sj->view_src && U.ids.count(sj->view_src) ? sj->view_src : nullptr);
                    if (c == nullptr) {
                        continue;
                    }
                    if (n->op != GGML_OP_MUL_MAT_ID || j != 0 || sj != c) {
                        U.whole[c] = true;
                    } else {
                        auto & v = U.ids[c];
                        if (std::find(v.begin(), v.end(), n->src[2]) == v.end()) {
                            v.push_back(n->src[2]);
                        }
                    }
                }
            }
        }
    }
    out.clear();
    const auto it = U.ids.find(input_cpy);
    if (it == U.ids.end() || it->second.empty() || U.whole.count(input_cpy)) {
        return false;
    }
    for (ggml_tensor * ids : it->second) {
        int be = -1;   // -1: the split's backend
        const auto sit = U.src_of.find(ids);
        if (sit != U.src_of.end()) {
            ids = sit->second.first;
            be  = sit->second.second;
        }
        for (const ggml_tensor * t : { (const ggml_tensor *) ids, (const ggml_tensor *) ids->view_src }) {
            const auto nit = t ? U.split_of.find(t) : U.split_of.end();
            if (nit != U.split_of.end() && nit->second >= split_id) {
                return false;   // not computed yet when this copy is made
            }
        }
        out.push_back({ ids, be });
    }
    return true;
}

static bool ggml_backend_sched_async_host_input_ok(ggml_backend_t input_backend, ggml_backend_t split_backend,
                                                   const struct ggml_tensor * input, const struct ggml_tensor * input_cpy) {
    if (ggml_backend_dev_type(ggml_backend_get_device(input_backend)) != GGML_BACKEND_DEVICE_TYPE_CPU ||
            ggml_backend_dev_type(ggml_backend_get_device(split_backend)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    // CUDA and the same backend built for AMD (HIP, "ROCm"): the copy is ordered on the split's stream there too. Upstream kept HIP
    // out of the same change for a prompt-speed loss, not for correctness; MUSA is not taken, and Vulkan needs a sync between H2D
    // copies and the graph
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(split_backend));
    if (reg == NULL || (strcmp(ggml_backend_reg_name(reg), "CUDA") != 0 && strcmp(ggml_backend_reg_name(reg), "ROCm") != 0)) {
        return false;
    }
    // A view of a user input (written by set_inputs, not by a CPU split) and host weights (op offload's expert copies) keep
    // the plain path
    if (((input->view_src ? input->view_src->flags : input->flags) & GGML_TENSOR_FLAG_INPUT) != 0) {
        return false;
    }
    ggml_backend_buffer_t buf = input->view_src ? input->view_src->buffer : input->buffer;
    if (buf != NULL && ggml_backend_buffer_get_usage(buf) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(split_backend));
    return buf != NULL && host != NULL && ggml_backend_buffer_get_type(buf) == host && input->data != NULL &&
           split_backend->iface.set_tensor_async != NULL && ggml_is_contiguous(input) && ggml_is_contiguous(input_cpy) &&
           ggml_nbytes(input) == ggml_nbytes(input_cpy);
}

static bool ggml_backend_sched_user_input_ring_ok(ggml_backend_t input_backend, ggml_backend_t split_backend,
                                                  const struct ggml_tensor * input, const struct ggml_tensor * input_cpy) {
    if (ggml_backend_dev_type(ggml_backend_get_device(input_backend)) != GGML_BACKEND_DEVICE_TYPE_CPU ||
            ggml_backend_dev_type(ggml_backend_get_device(split_backend)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(split_backend));
    if (reg == NULL || (strcmp(ggml_backend_reg_name(reg), "CUDA") != 0 && strcmp(ggml_backend_reg_name(reg), "ROCm") != 0)) {
        return false;
    }
    ggml_backend_buffer_t buf = input->view_src ? input->view_src->buffer : input->buffer;
    ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(split_backend));
    const size_t nbytes = ggml_nbytes(input);
    return buf != NULL && ggml_backend_buffer_is_host(buf) && host != NULL && input->data != NULL &&
           split_backend->iface.set_tensor_async != NULL && ggml_is_contiguous(input) && ggml_is_contiguous(input_cpy) &&
           nbytes == ggml_nbytes(input_cpy) && nbytes <= (size_t) 64 * 1024 * 1024;
}

// The split whose launch holds the event an input hook waits on before it
// provides split k's inputs - the hook's contract is to wait until the inputs it provides are produced, i.e. for the LATEST
// earlier split whose graph holds one of them (or its view source); inputs that no earlier split holds are skipped. Looked up once per
// graph split (the cache is reset when the graph is split again); -1 when no earlier split holds any of them (then the guard syncs)
static int ggml_backend_sched_hook_src_split(ggml_backend_sched_t sched, int k) {
    if (!sched->hook_src_valid || sched->hook_src_cap < sched->n_splits) {
        if (sched->hook_src_cap < sched->n_splits) {
            free(sched->hook_src_split);
            sched->hook_src_cap   = sched->n_splits;
            sched->hook_src_split = (int *) malloc(sizeof(int) * (size_t) sched->hook_src_cap);
            GGML_ASSERT(sched->hook_src_split != NULL);
        }
        for (int i = 0; i < sched->hook_src_cap; i++) {
            sched->hook_src_split[i] = -2;
        }
        sched->hook_src_valid = true;
    }
    if (sched->hook_src_split[k] != -2) {
        return sched->hook_src_split[k];
    }
    const struct ggml_backend_sched_split * sp = &sched->splits[k];
    int j_max = -1;
    for (int in = 0; in < sp->n_inputs; in++) {
        const struct ggml_tensor * t = sp->inputs[in];
        int j = -1;
        for (int i = k - 1; i >= 0 && j < 0; i--) {
            const struct ggml_cgraph * g = &sched->splits[i].graph;
            for (int n = 0; n < g->n_nodes; n++) {
                if (g->nodes[n] == t || (t->view_src != NULL && g->nodes[n] == t->view_src)) {
                    j = i;
                    break;
                }
            }
        }
        j_max = std::max(j_max, j);   // (an input found nowhere is skipped, j = -1)
    }
    sched->hook_src_split[k] = j_max;
    return sched->hook_src_split[k];
}

template <bool HO>
static enum ggml_status ggml_backend_sched_compute_splits_impl(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // GGML_COUNT_SPLITS prints how many graphs (and how many cross-backend inputs) the scheduler
    // runs per compute: the number of CPU<->GPU boundaries a graph has is what the MoE expert
    // streaming engine pays per token (measured: 82 of them for one Flash decode token)
    // Read once per process, not on every compute (otherwise 2 getenv calls per compute)
    static const bool dbg_splits = getenv("GGML_COUNT_SPLITS") != nullptr;
    // the profiler (GGML_PROFILE or ggml_prof_enable): decided per compute, so a run-time switch is seen at the next graph
    const bool prof = ggml_prof_on();
    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<std::pair<ggml_tensor *, int>> moe_ids_list;
    ggml_backend_sched_moe_users moe_users;
    std::vector<int32_t> ids;
    std::vector<std::pair<struct ggml_tensor *, struct ggml_tensor *>> h2d_late;
    // Per backend: the last split of this compute that received an async H2D copy on that backend's stream (-1: none pending). With one
    // GPU only that GPU's entry is ever set (the guard's rule as before); with several GPUs a CPU split waits for every GPU with a copy
    // still pending before it writes the CPU compute buffer, unless the input hook's event (later on that same GPU) covers it
    int last_async_split[GGML_SCHED_MAX_BACKENDS];
    size_t user_inp_off[GGML_SCHED_MAX_BACKENDS];
    bool   user_inp_used[GGML_SCHED_MAX_BACKENDS];
    for (int b = 0; b < GGML_SCHED_MAX_BACKENDS; b++) {
        last_async_split[b] = -1;
        user_inp_off[b]     = 0;
        user_inp_used[b]    = false;
    }
    // A host synchronize of a backend finishes its copies too: nothing is left for the guard there
    auto async_synced = [&](ggml_backend_t b) {
        const int id = ggml_backend_sched_backend_id(sched, b);
        if (id >= 0) {
            last_async_split[id] = -1;
            ggml_backend_sched_mark_backend_synced(sched, id);
        }
    };
    auto record_async_h2d_ev = [&](int bid, ggml_backend_t b) {
        if (sched->async_h2d_ev[bid] == NULL) {
            sched->async_h2d_ev[bid] = ggml_backend_event_new(ggml_backend_get_device(b));
        }
        if (sched->async_h2d_ev[bid] != NULL) {
            ggml_backend_event_record(sched->async_h2d_ev[bid], b);
        }
        sched->async_h2d_pending[bid] = true;
    };
    const bool async_multi = sched->async_host_input && sched->n_async_gpus > 1;
    auto is_cpu_backend = [](ggml_backend_t b) {
        return ggml_backend_dev_type(ggml_backend_get_device(b)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    };
    std::vector<ggml_bitset_t> used_ids;
    std::vector<ggml_tensor *> hook_cpys;   // the input hook's copies of one split

    int prev_backend_id = -1;

    // One record per split of this call (timed instance only; per thread)
    std::vector<ggml_handoff_split_rec> * ho_recs = nullptr;
    int64_t ho_t_sched0 = 0;
    if constexpr (HO) {
        ho_t_sched0 = ggml_time_us();
        ho_recs     = &t_ho_recs;
        ho_recs->assign(sched->n_splits, ggml_handoff_split_rec{});
    }
    // GGML_TRACE - the timed instance (HO) also runs when only the trace is on; it books the numbers as
    // always (nothing prints unless GGML_PROFILE_HANDOFF is set) and sends every wait / copy / split to the trace
    const bool tr       = HO && ggml_trace_on();
    int        tr_split = 0;
    auto ho_wait = [&](ggml_handoff_split_rec * r, int kind, int64_t t_start) {
        const int64_t t_now = ggml_time_us();
        const int64_t us = t_now - t_start;
        r->wn[kind]  += 1;
        r->wus[kind] += us;
        r->wlong[kind] += us >= HO_LONG_US ? 1 : 0;
        if (tr) {
            char nb[64]; snprintf(nb, sizeof(nb), "wait %s", g_ho_wait_names[kind]);
            ggml_trace_event(GGML_TRACE_LANE(1, 0), nb, t_start, t_now, tr_split);
        }
    };
    auto ho_copy = [&](ggml_handoff_split_rec * r, int kind, int64_t t_start) {
        const int64_t t_now = ggml_time_us();
        r->cn[kind]  += 1;
        r->cus[kind] += t_now - t_start;
        if (tr) {
            char nb[64]; snprintf(nb, sizeof(nb), "copy %s", g_ho_copy_names[kind]);
            ggml_trace_event(GGML_TRACE_LANE(1, 0), nb, t_start, t_now, tr_split);
        }
    };
    GGML_UNUSED(ho_wait);
    GGML_UNUSED(ho_copy);

    if (prof) {
        ggml_prof_count(GGML_PROF_N_GRAPHS,   1);
        ggml_prof_count(GGML_PROF_N_SPLITS,   sched->n_splits);
    }

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        ggml_handoff_split_rec * ho_r = nullptr;
        int64_t ho_t_split0 = 0, ho_t = 0;
        if constexpr (HO) {
            ho_r        = &(*ho_recs)[split_id];
            ho_t_split0 = ggml_time_us();
            tr_split    = split_id;
        }
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        if (dbg_splits && split_id == 0) {
            // once per compute, when the shape of the graph changes
            int64_t n_in = 0;
            for (int sj = 0; sj < sched->n_splits; sj++) {
                n_in += sched->splits[sj].n_inputs;
            }
            static int last_splits = -1;
            static long long last_inputs = -1;
            if (sched->n_splits != last_splits || n_in != last_inputs) {
                last_splits = sched->n_splits;
                last_inputs = n_in;
                fprintf(stderr, "GGML_SPLITS: %d graphs, %lld cross-backend inputs\n",
                        sched->n_splits, (long long) n_in);
            }
        }
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            const int64_t t0_prof = ggml_time_us();
            ggml_prof_count(GGML_PROF_N_SYNCS, 1);
            if constexpr (HO) { ho_t = ggml_time_us(); }
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
            if constexpr (HO) { ho_wait(ho_r, HO_W_NOINPUT, ho_t); }
            ggml_prof_add(GGML_PROF_SYNC, ggml_time_us() - t0_prof);
        }

        // Batched wait (the idea is neurall/llama.cpp's fork, ggml-backend.cpp ~2113-2141 there): a split
        // with many cross-backend inputs - our decode graph has 166 of them and 352 syncs per
        // token - pays one *blocking* ggml_backend_synchronize() per input on the backend that produced
        // it. All of those inputs come from one or two backends, and one wait per backend is enough: after
        // a backend's first wait, every tensor it produced is final. So hoist the waits out of the loop.
        // Default off: the scheduler is where races live, and the right tests are GGML_SCHED_SANITIZE
        // (mainline #26167) and token identity, not tok/s.
        static const bool sync_batch = getenv("GGML_SCHED_SYNC_BATCH") != nullptr &&
                atoi(getenv("GGML_SCHED_SYNC_BATCH")) != 0;

        if (sync_batch && split->n_inputs >= 1) {   // with one input the per-input wait below was skipped and nothing waited
            ggml_backend_t synced[GGML_SCHED_MAX_BACKENDS] = { nullptr };
            int n_synced = 0;

            for (int input_id = 0; input_id < split->n_inputs; input_id++) {
                ggml_backend_t ib = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
                bool seen = false;

                for (int i = 0; i < n_synced; i++) {
                    seen = seen || synced[i] == ib;
                }

                if (seen || n_synced == GGML_SCHED_MAX_BACKENDS) {
                    continue;
                }

                // the split's own device needs no wait: its work is ordered on its own stream, and
                // blocking the host on it would be a sync the code below does not do
                if (ib == split_backend) {
                    continue;
                }

                ggml_prof_count(GGML_PROF_N_SYNCS, 1);
                if constexpr (HO) { ho_t = ggml_time_us(); }
                ggml_backend_synchronize(ib);
                if constexpr (HO) { ho_wait(ho_r, HO_W_BATCHED, ho_t); }
                synced[n_synced++] = ib;
            }
        }

        // An input provider (the MoE streaming engine's "router done" slots) may provide all of this
        // split's inputs; asked first without the copies (it copies and waits for nothing), then asked to fill them
        bool inputs_provided = false;
        if (sched->input_hook != NULL && split->n_inputs > 0 &&
                sched->input_hook(sched->input_hook_user_data, split_backend, split->n_inputs, split->inputs, NULL)) {
            hook_cpys.resize(split->n_inputs);
            for (int input_id = 0; input_id < split->n_inputs; input_id++) {
                hook_cpys[input_id] = tensor_copy(split->inputs[input_id], split_backend_id, sched->cur_copy);
            }
            // The hook waits only for the event of the split that produced its inputs.
            // If an async H2D copy went into a LATER split, that event does not cover it, and this CPU split - the hook's own copies
            // into its inputs first (the check runs before them) - could write the copy's source before the DMA
            // ran: synchronize that split's GPU first (counted; 0 on ordinary graphs, where the router of the next layer sits in the
            // split that reads the copy)
            // (with several GPUs: also every GPU with a copy pending whose split is not the hook's - an event on another GPU does not
            // cover it; a covered one is done once the hook has waited, so it is cleared)
            int hook_src = -2;
            for (int b = 0; b < sched->n_backends; b++) {
                if (last_async_split[b] < 0) {
                    continue;
                }
                if (hook_src == -2) {
                    hook_src = ggml_backend_sched_hook_src_split(sched, split_id);
                }
                const int  src_b     = hook_src >= 0 ? splits[hook_src].backend_id : -1;
                const bool other_gpu = src_b >= 0 && src_b != b && !is_cpu_backend(sched->backends[src_b]);
                if (hook_src < last_async_split[b] || other_gpu) {
                    if constexpr (HO) { ho_t = ggml_time_us(); }
                    ggml_backend_synchronize(sched->backends[b]);
                    if constexpr (HO) { ho_wait(ho_r, HO_W_GUARD, ho_t); }
                    g_async_guard_syncs.fetch_add(1);
                    last_async_split[b] = -1;
                } else if (async_multi) {
                    last_async_split[b] = -1;
                }
            }
            if constexpr (HO) { ho_t = ggml_time_us(); }
            const int64_t t0_hook = prof ? ggml_time_us() : 0;
            inputs_provided = sched->input_hook(sched->input_hook_user_data, split_backend, split->n_inputs, split->inputs,
                    hook_cpys.data());
            if (prof && inputs_provided) {   // the provider waits for its producer's event and fills the inputs
                ggml_prof_add(GGML_PROF_SYNC, ggml_time_us() - t0_hook);
            }
            if constexpr (HO) { if (inputs_provided) { ho_wait(ho_r, HO_W_HOOK, ho_t); } }
        }

        // Several GPUs: a CPU split that the hook does not provide writes the CPU compute buffer (its input copies, its compute) - every
        // GPU with an async copy still pending (whose source may be in that buffer) is waited for first
        if (async_multi && !inputs_provided && is_cpu_backend(split_backend)) {
            for (int b = 0; b < sched->n_backends; b++) {
                if (last_async_split[b] < 0) {
                    continue;
                }
                if constexpr (HO) { ho_t = ggml_time_us(); }
                ggml_backend_synchronize(sched->backends[b]);
                if constexpr (HO) { ho_wait(ho_r, HO_W_GUARD, ho_t); }
                g_async_multi_syncs.fetch_add(1);
                last_async_split[b] = -1;
            }
        }

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs && !inputs_provided; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            ggml_prof_count(GGML_PROF_N_SPLIT_INPUTS, 1);

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                bool staged_async = false;
                if (sched->events[split_backend_id][sched->cur_copy] == NULL &&
                        ggml_backend_sched_user_input_ring_ok(input_backend, split_backend, input, input_cpy)) {
                    const int b = split_backend_id;
                    const int r = sched->user_inp_cur[b];
                    const size_t nbytes = ggml_nbytes(input);
                    bool ring_ready = true;
                    if (!user_inp_used[b]) {
                        if (sched->user_inp_ev_pending[b][r] && sched->user_inp_ev[b][r] != NULL) {
                            const int64_t t0_sync = ggml_time_us();
                            if constexpr (HO) { ho_t = ggml_time_us(); }
                            ggml_backend_event_synchronize(sched->user_inp_ev[b][r]);
                            sched->user_inp_ev_pending[b][r] = false;
                            if constexpr (HO) { ho_wait(ho_r, HO_W_USER_INPUT, ho_t); }
                            ggml_prof_add(GGML_PROF_SYNC, ggml_time_us() - t0_sync);
                        }
                        if (sched->user_inp_ev[b][r] == NULL) {
                            sched->user_inp_ev[b][r] = ggml_backend_event_new(ggml_backend_get_device(split_backend));
                        }
                        size_t total_need = 0;
                        for (int sj = split_id; sj < sched->n_splits; sj++) {
                            if (splits[sj].backend_id != b) {
                                continue;
                            }
                            for (int ij = 0; ij < splits[sj].n_inputs; ij++) {
                                struct ggml_tensor * sinp = splits[sj].inputs[ij];
                                if (sinp->flags & GGML_TENSOR_FLAG_INPUT) {
                                    total_need += (ggml_nbytes(sinp) + 255) & ~(size_t) 255;
                                }
                            }
                        }
                        if (total_need > sched->user_inp_cap[b][r]) {
                            ggml_backend_buffer_free(sched->user_inp_buf[b][r]);
                            sched->user_inp_buf[b][r] = NULL;
                            sched->user_inp_cap[b][r] = 0;
                            const size_t alloc_cap = std::max<size_t>(total_need * 2, (size_t) 4 * 1024 * 1024);
                            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(split_backend));
                            sched->user_inp_buf[b][r] = ggml_backend_buft_alloc_buffer(host_buft, alloc_cap);
                            if (sched->user_inp_buf[b][r] != NULL) {
                                sched->user_inp_cap[b][r] = alloc_cap;
                            }
                        }
                        ring_ready = sched->user_inp_buf[b][r] != NULL && sched->user_inp_ev[b][r] != NULL;
                    } else {
                        ring_ready = sched->user_inp_buf[b][r] != NULL && sched->user_inp_ev[b][r] != NULL &&
                                     ((user_inp_off[b] + 255) & ~(size_t) 255) + nbytes <= sched->user_inp_cap[b][r];
                    }
                    if (ring_ready) {
                        const int64_t t0_cpy = ggml_time_us();
                        ggml_prof_count(GGML_PROF_N_COPIES, 1);
                        if constexpr (HO) { ho_t = ggml_time_us(); }
                        const size_t off = (user_inp_off[b] + 255) & ~(size_t) 255;
                        uint8_t * dst = (uint8_t *) ggml_backend_buffer_get_base(sched->user_inp_buf[b][r]) + off;
                        if (nbytes > 0) {
                            memcpy(dst, input->data, nbytes);
                            ggml_backend_tensor_set_async(split_backend, input_cpy, dst, 0, nbytes);
                            ggml_backend_event_record(sched->user_inp_ev[b][r], split_backend);
                            sched->user_inp_ev_pending[b][r] = true;
                        }
                        user_inp_off[b]  = off + nbytes;
                        user_inp_used[b] = true;
                        staged_async     = true;
                        if constexpr (HO) { ho_copy(ho_r, HO_C_USER, ho_t); }
                        ggml_prof_add(GGML_PROF_COPY, ggml_time_us() - t0_cpy);
                    }
                }
                if (!staged_async) {
                    // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                    const int64_t t0_sync = ggml_time_us();
                    ggml_prof_count(GGML_PROF_N_SYNCS, 1);
                    if constexpr (HO) { ho_t = ggml_time_us(); }
                    if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                        ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                    } else {
                        ggml_backend_synchronize(split_backend);
                        async_synced(split_backend);
                    }
                    if constexpr (HO) { ho_wait(ho_r, HO_W_USER_INPUT, ho_t); }
                    ggml_prof_add(GGML_PROF_SYNC, ggml_time_us() - t0_sync);
                    const int64_t t0_cpy = ggml_time_us();
                    ggml_prof_count(GGML_PROF_N_COPIES, 1);
                    if constexpr (HO) { ho_t = ggml_time_us(); }
                    ggml_backend_tensor_copy(input, input_cpy);
                    if constexpr (HO) { ho_copy(ho_r, HO_C_USER, ho_t); }
                    ggml_prof_add(GGML_PROF_COPY, ggml_time_us() - t0_cpy);
                }
            } else if (sched->async_host_input && ggml_backend_sched_async_host_input_ok(input_backend, split_backend, input, input_cpy)) {
                // The async H2D on the split's stream, no host wait (see the conditions above)
                const int64_t t0_cpy = ggml_time_us();
                ggml_prof_count(GGML_PROF_N_COPIES, 1);
                if constexpr (HO) { ho_t = ggml_time_us(); }
                if (sched->test_h2d_late) {
                    h2d_late.push_back({ input, input_cpy });   // TEST ONLY: enqueued after the split's graph below
                } else {
                    ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, ggml_nbytes(input));
                    record_async_h2d_ev(split_backend_id, split_backend);
                }
                last_async_split[split_backend_id] = split_id;
                if constexpr (HO) { ho_copy(ho_r, HO_C_ASYNC_H2D, ho_t); }   // (its own kind)
                ggml_prof_add(GGML_PROF_COPY, ggml_time_us() - t0_cpy);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                const int64_t t0_wait = ggml_time_us();
                ggml_prof_count(GGML_PROF_N_SYNCS, 1);
                if constexpr (HO) { ho_t = ggml_time_us(); }
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                    async_synced(split_backend);
                }
                if constexpr (HO) { ho_wait(ho_r, HO_W_WAR, ho_t); }
                ggml_prof_add(GGML_PROF_SYNC, ggml_time_us() - t0_wait);

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    ) && ggml_backend_sched_moe_copy_ids(sched, moe_users, split_id, input_cpy, moe_ids_list)) {   // every user's ids

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    if (!sync_batch) {   // the batched wait above already did this
                        if constexpr (HO) { ho_t = ggml_time_us(); }
                        const int64_t t0_mi = prof ? ggml_time_us() : 0;
                        ggml_backend_synchronize(input_backend);
                        async_synced(input_backend);
                        if (prof) { ggml_prof_add(GGML_PROF_SYNC, ggml_time_us() - t0_mi); }
                        if constexpr (HO) { ho_wait(ho_r, HO_W_MOE_INPUT, ho_t); }
                    }

                    // get the ids (of every MUL_MAT_ID node that reads this copy - the union of their experts)
                    if (moe_ids_list.size() == 1 && ggml_nelements(moe_ids_list[0].first) == 0) {
                        continue;
                    }

                    if (moe_ids_list.size() != 1 || moe_ids_list[0].first != prev_ids_tensor) {
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (const auto & idb : moe_ids_list) {
                            // an ids tensor that is an input copy (of this split or a later one) is read from the tensor it copies
                            ggml_tensor * ids_tensor = idb.first;
                            ggml_backend_t ids_backend = idb.second >= 0 ? sched->backends[idb.second] : split_backend;

                            ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                            if constexpr (HO) { ho_t = ggml_time_us(); }
                            const int64_t t0_mids = prof ? ggml_time_us() : 0;
                            ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                            ggml_backend_synchronize(ids_backend);
                            if (prof) { ggml_prof_add(GGML_PROF_SYNC, ggml_time_us() - t0_mids); }
                            if constexpr (HO) { ho_wait(ho_r, HO_W_MOE_IDS, ho_t); }

                            // find the used experts
                            for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                                for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                    int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                    GGML_ASSERT(id >= 0 && id < n_expert);
                                    ggml_bitset_set(used_ids.data(), id);
                                }
                            }
                        }

                        prev_ids_tensor = moe_ids_list.size() == 1 ? moe_ids_list[0].first : nullptr;
                    }
                    if (std::none_of(used_ids.begin(), used_ids.end(), [](ggml_bitset_t b) { return b != 0; })) {
                        continue;
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        const int64_t t0_prof = ggml_time_us();
                        ggml_prof_count(GGML_PROF_N_COPIES, 1);
                        if constexpr (HO) { ho_t = ggml_time_us(); }
                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                        if constexpr (HO) { ho_copy(ho_r, HO_C_EXPERTS, ho_t); }
                        ggml_prof_add(GGML_PROF_COPY, ggml_time_us() - t0_prof);
                    };

                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if constexpr (HO) { ho_t = ggml_time_us(); }
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        if constexpr (HO) { ho_copy(ho_r, HO_C_ASYNC_FAIL, ho_t); ho_t = ggml_time_us(); }
                        int64_t t0_dep = prof ? ggml_time_us() : 0;   // the profiler: the waits for the producer and the split, then the copy
                        if (!sync_batch) {   // the batched wait above already did this
                            if constexpr (HO) { ho_t = ggml_time_us(); }
                            ggml_backend_synchronize(input_backend);
                            async_synced(input_backend);
                            if constexpr (HO) { ho_wait(ho_r, HO_W_DEP_IN, ho_t); }
                        }
                        if constexpr (HO) { ho_t = ggml_time_us(); }
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                            async_synced(split_backend);
                        }
                        if constexpr (HO) { ho_wait(ho_r, HO_W_DEP_SPLIT, ho_t); ho_t = ggml_time_us(); }
                        if (prof) {
                            const int64_t t1_dep = ggml_time_us();
                            ggml_prof_add(GGML_PROF_SYNC, t1_dep - t0_dep);
                            t0_dep = t1_dep;
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                        if (prof) { ggml_prof_add(GGML_PROF_COPY, ggml_time_us() - t0_dep); }
                        if constexpr (HO) { ho_copy(ho_r, HO_C_SYNC, ho_t); }
                    } else if constexpr (HO) {
                        ho_copy(ho_r, HO_C_ASYNC, ho_t);
                    }
                }
            }
        }

        int64_t ho_agg0 = 0;
        if constexpr (HO) {
            ho_t    = ggml_time_us();
            ho_agg0 = t_ho_cpu_agg_us;   // the CPU graph's own bookkeeping (ggml_handoff_cpu_graph) is taken out below
        }
        // The split hook, after a split has been issued (on a GPU: enqueued)
        const bool split_hook = sched->split_hook != NULL;
        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
            if (split_hook) {
                sched->split_hook(sched->split_hook_user_data, split_backend, &split->graph);
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }
                if (split_hook) {
                    sched->split_hook(sched->split_hook_user_data, split_backend, &gv);
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }
        // TEST ONLY (GGML_SCHED_TEST_H2D_LATE): the copies after the graph that reads them - on both paths
        if (!h2d_late.empty()) {
            for (const auto & lc : h2d_late) {
                ggml_backend_tensor_set_async(split_backend, lc.second, lc.first->data, 0, ggml_nbytes(lc.first));
            }
            record_async_h2d_ev(split_backend_id, split_backend);
            h2d_late.clear();
        }

        if constexpr (HO) {
            ho_r->cpu_agg_us  = t_ho_cpu_agg_us - ho_agg0;
            ho_r->compute_us += ggml_time_us() - ho_t - ho_r->cpu_agg_us;
        }

        // record the event of this split
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;

        if constexpr (HO) {
            const int64_t t_split1 = ggml_time_us();
            ho_r->host_us = t_split1 - ho_t_split0 - ho_r->cpu_agg_us;
            if (tr) {   // the whole split, as the host spent it (a CPU split computes in here; a GPU split is only issued)
                char nb[160];
                snprintf(nb, sizeof(nb), "split %s %s", ggml_backend_name(split_backend),
                         split->graph.n_nodes > 0 ? split->graph.nodes[0]->name : "");
                ggml_trace_event(GGML_TRACE_LANE(1, 0), nb, ho_t_split0, t_split1, split_id);
            }
        }
    }

    for (int b = 0; b < sched->n_backends; b++) {
        if (user_inp_used[b]) {
            sched->user_inp_cur[b] = (sched->user_inp_cur[b] + 1) % GGML_SCHED_USER_INP_RING;
        }
    }

    if constexpr (HO) {
        int64_t agg = 0;   // the CPU graphs' bookkeeping, already taken out of the splits
        for (int i = 0; i < sched->n_splits; i++) {
            agg += (*ho_recs)[i].cpu_agg_us;
        }
        if (ggml_handoff_on()) {   // trace-only runs need the events, not the bookkeeping
            ggml_handoff_sched_done(sched, *ho_recs, ggml_time_us() - ho_t_sched0 - agg);
        }
    }

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    static const bool timed = ggml_handoff_on() || ggml_trace_on();   // chosen once at start-up
    return timed ? ggml_backend_sched_compute_splits_impl<true>(sched)
                        : ggml_backend_sched_compute_splits_impl<false>(sched);
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->graph_size      = 0;
    sched->prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;
    sched->op_offload_only = -1;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
        ggml_backend_event_free(sched->async_h2d_ev[b]);
        for (int r = 0; r < GGML_SCHED_USER_INP_RING; r++) {
            ggml_backend_event_free(sched->user_inp_ev[b][r]);
            ggml_backend_buffer_free(sched->user_inp_buf[b][r]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hook_src_split);
    if (sched->async_host_input) {
        fprintf(stderr, "ggml_backend_sched: async-h2d: hook-provided CPU splits that had to synchronize the GPU first (the "
                "graph-shape guard): %lld\n", (long long) g_async_guard_syncs.load());
        if (sched->n_async_gpus > 1) {
            fprintf(stderr, "ggml_backend_sched: async-h2d with %d GPUs: CPU splits not provided by the hook that synchronized a GPU with a "
                    "copy still queued first: %lld\n", sched->n_async_gpus, (long long) g_async_multi_syncs.load());
        }
    }
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
    sched->alloc_failed = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        sched->alloc_failed = true;
        return false;
    }

    sched->is_alloc = true;

    return true;
}

bool ggml_backend_sched_alloc_graph_again(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    if (sched->is_alloc) {
        return true;
    }
    if (!sched->alloc_failed) {
        return false;   // (no graph split by a failed ggml_backend_sched_alloc_graph: a reset, a reserve or a successful allocation came since)
    }

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->alloc_failed = false;
    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
        ggml_backend_sched_mark_backend_synced(sched, i);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_wait_async_host_inputs(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int b = 0; b < sched->n_backends; b++) {
        if (!sched->async_h2d_pending[b]) {
            continue;
        }
        if (sched->async_h2d_ev[b] != NULL) {
            ggml_backend_event_synchronize(sched->async_h2d_ev[b]);
        } else {
            ggml_backend_synchronize(sched->backends[b]);
        }
        sched->async_h2d_pending[b] = false;
    }
}

void ggml_backend_sched_set_op_offload(ggml_backend_sched_t sched, bool op_offload) {
    GGML_ASSERT(sched);
    sched->op_offload = op_offload;
}

void ggml_backend_sched_set_op_offload_backend(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    sched->op_offload_only = backend != NULL ? ggml_backend_sched_backend_id(sched, backend) : -1;
}

// See async_host_input. Only with one copy of the inputs (no pipeline parallelism: no events), which the
// caller checks too (single GPU)
bool ggml_backend_sched_get_async_host_input(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->async_host_input;
}

void ggml_backend_sched_set_async_host_input(ggml_backend_sched_t sched, bool on) {
    GGML_ASSERT(sched);
    sched->async_host_input = on && sched->n_copies == 1;
    sched->n_async_gpus     = 0;
    for (int b = 0; b < sched->n_backends; b++) {
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(ggml_backend_get_device(sched->backends[b]));
        sched->n_async_gpus += t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU ? 1 : 0;
    }
    g_ho_async_h2d_on = g_ho_async_h2d_on || sched->async_host_input;
    static const int late = getenv("GGML_SCHED_TEST_H2D_LATE") ? atoi(getenv("GGML_SCHED_TEST_H2D_LATE")) : 0;
    sched->test_h2d_late = sched->async_host_input ? late : 0;
}

void ggml_backend_sched_set_input_hook(ggml_backend_sched_t sched, ggml_backend_sched_input_hook_t hook, void * user_data) {
    GGML_ASSERT(sched);
    sched->input_hook           = hook;
    sched->input_hook_user_data = user_data;
}

void ggml_backend_sched_set_copy_hook(ggml_backend_sched_t sched, ggml_backend_sched_copy_hook_t hook, void * user_data) {
    GGML_ASSERT(sched);
    sched->copy_hook           = hook;
    sched->copy_hook_user_data = user_data;
}

void ggml_backend_sched_get_alloc_times(ggml_backend_sched_t sched, int64_t * place_us, int64_t * sync_us, int64_t * plan_us, int * n_realloc) {
    *place_us  = sched->t_place_us;
    *sync_us   = sched->t_realloc_sync_us;
    *plan_us   = sched->t_realloc_plan_us;
    *n_realloc = sched->n_realloc;
}

bool ggml_backend_sched_set_fixed_chunks(ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_buffer_t * chunks, int n_chunks, bool own_may_grow) {
    GGML_ASSERT(sched);
    const int id = ggml_backend_sched_backend_id(sched, backend);
    if (id < 0) {
        return false;
    }
    return ggml_gallocr_set_fixed_chunks(sched->galloc, id, chunks, n_chunks, own_may_grow);
}

void ggml_backend_sched_set_split_hook(ggml_backend_sched_t sched, ggml_backend_sched_split_hook_t hook, void * user_data) {
    GGML_ASSERT(sched);
    sched->split_hook           = hook;
    sched->split_hook_user_data = user_data;
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

void ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_clear_host_buffers(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ggml_gallocr_clear_host_buffers(sched->galloc);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

bool ggml_op_alloc_size_may_expand(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_ARGMAX:
            return true;
        default:
            return false;
    }
}

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}

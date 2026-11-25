#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define GGML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FILL:
        case GGML_OP_SCALE:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_SILU_BACK:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_CLAMP:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_MAX_BACK:
            return true;

        default:
            return false;
    }
}

static size_t aligned_offset(const void * buffer, size_t offset, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1))); // power of 2
    size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
    return offset + align;
}

// tallocr

struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    void * base = ggml_backend_buffer_get_base(buffer);
    size_t align = ggml_backend_buffer_get_alignment(buffer);

    assert(align && !(align & (align - 1))); // power of 2

    struct ggml_tallocr talloc = (struct ggml_tallocr) {
        /*.buffer    = */ buffer,
        /*.base      = */ base,
        /*.alignment = */ align,
        /*.offset    = */ aligned_offset(base, 0, align),
    };
    return talloc;
}

enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    if (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ABORT("not enough space in the buffer");
    }

    void * addr = (char *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    assert(((uintptr_t)addr % talloc->alignment) == 0);

    return ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
struct buffer_address {
    int chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

static const struct buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

static bool ggml_buffer_address_less(struct buffer_address a, struct buffer_address b) {
    return a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

struct free_block {
    size_t offset;
    size_t size;
};

struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
    size_t max_size;
};

struct ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    struct tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    int n_chunks;
    // n_fixed > 0 = chunks [0, n_fixed) are the caller's pieces, each of its fixed size (ggml_gallocr_set_fixed_chunks),
    // and chunk n_fixed is the allocator's own growable chunk (the real buffer): a tensor takes the best-fitting free space in a piece,
    // and only when no piece can hold it the own chunk (which grows as needed) - so a plan always fits
    size_t fixed[GGML_VBUFFER_MAX_CHUNKS];
    int n_fixed;
    bool failed;   // (unused since the own chunk: kept for the reserve's check)

#ifdef GGML_ALLOCATOR_DEBUG
    struct {
        const struct ggml_tensor * tensor;
        struct buffer_address addr;
    } allocated_tensors[1024];
#endif
};

static void ggml_dyn_tallocr_insert_block(struct tallocr_chunk * chunk, size_t offset, size_t size) {
    GGML_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
    // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
    int insert_pos = 0;
    while (insert_pos < chunk->n_free_blocks && chunk->free_blocks[insert_pos].offset < offset) {
        insert_pos++;
    }
    // shift all blocks from insert_pos onward to make room for the new block
    for (int i = chunk->n_free_blocks; i > insert_pos; i--) {
        chunk->free_blocks[i] = chunk->free_blocks[i-1];
    }
    // insert the new block
    chunk->free_blocks[insert_pos].offset = offset;
    chunk->free_blocks[insert_pos].size = size;
    chunk->n_free_blocks++;
}

static void ggml_dyn_tallocr_remove_block(struct tallocr_chunk * chunk, int idx) {
    // shift all elements after idx by 1 to the left, overwriting the element at idx
    for (int i = idx; i < chunk->n_free_blocks - 1; i++) {
        chunk->free_blocks[i] = chunk->free_blocks[i+1];
    }
    chunk->n_free_blocks--;
}

static int ggml_dyn_tallocr_new_chunk(struct ggml_dyn_tallocr * alloc, size_t min_size) {
    if (alloc->n_chunks >= GGML_VBUFFER_MAX_CHUNKS || alloc->n_fixed > 0) {
        return -1;
    }
    struct tallocr_chunk * chunk = calloc(1, sizeof(struct tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    if (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }
    alloc->chunks[alloc->n_chunks] = chunk;
    alloc->n_chunks++;
    return alloc->n_chunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
static void add_allocated_tensor(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].tensor == NULL) {
            alloc->allocated_tensors[i].tensor = tensor;
            alloc->allocated_tensors[i].addr = addr;
            return;
        }
    }
    GGML_ABORT("out of allocated_tensors");
}
static void remove_allocated_tensor(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, const struct ggml_tensor * tensor) {
    for (int i = 0; i < 1024; i++) {
        if (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.offset) {
            alloc->allocated_tensors[i].tensor = NULL;
            return;
        }
    }
    GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

// An allocation with fixed pieces - the best-fitting free block of any piece (every block of a piece is bounded by
// its capacity), else the own chunk (its non-last blocks first, then its last block, which grows)
static struct buffer_address ggml_dyn_tallocr_alloc_fixed(struct ggml_dyn_tallocr * alloc, size_t size) {
    int best_c = -1, best_b = -1;
    size_t best_size = SIZE_MAX;
    for (int c = 0; c < alloc->n_fixed; ++c) {
        struct tallocr_chunk * chunk = alloc->chunks[c];
        for (int i = 0; i < chunk->n_free_blocks; i++) {
            const size_t bs = chunk->free_blocks[i].size;
            if (bs >= size && bs < best_size) {
                best_c = c; best_b = i; best_size = bs;
            }
        }
    }
    if (best_c < 0) {
        struct tallocr_chunk * own = alloc->chunks[alloc->n_fixed];
        for (int i = 0; i < own->n_free_blocks - 1; i++) {
            const size_t bs = own->free_blocks[i].size;
            if (bs >= size && bs < best_size) {
                best_c = alloc->n_fixed; best_b = i; best_size = bs;
            }
        }
        if (best_c < 0) {
            best_c = alloc->n_fixed;
            best_b = own->n_free_blocks - 1;   // (the last block: virtually endless)
        }
    }
    struct tallocr_chunk * chunk = alloc->chunks[best_c];
    struct free_block    * block = &chunk->free_blocks[best_b];
    GGML_ASSERT(block->size >= size);
    struct buffer_address  addr  = {.chunk = best_c, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    if (block->size == 0) {
        ggml_dyn_tallocr_remove_block(chunk, best_b);
    }
    chunk->max_size = MAX(chunk->max_size, addr.offset + size);
    return addr;
}

static struct buffer_address ggml_dyn_tallocr_alloc(struct ggml_dyn_tallocr * alloc, size_t size, const struct ggml_tensor * tensor) {
    size = aligned_offset(NULL, size, alloc->alignment);
    if (alloc->n_fixed > 0) {
        GGML_UNUSED(tensor);
        return ggml_dyn_tallocr_alloc_fixed(alloc, size);
    }

    AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

    int best_fit_chunk = -1;
    int best_fit_block = -1;
    size_t max_avail = 0;

    // find the best fitting free block besides the last block, within any chunk
    for (int c = 0; c < alloc->n_chunks; ++c) {
        struct tallocr_chunk * chunk = alloc->chunks[c];
        size_t best_fit_size = SIZE_MAX;
        for (int i = 0; i < chunk->n_free_blocks - 1; i++) {
            struct free_block * block = &chunk->free_blocks[i];
            max_avail = MAX(max_avail, block->size);
            if (block->size >= size && block->size <= best_fit_size) {
                best_fit_chunk = c;
                best_fit_block = i;
                best_fit_size = block->size;
            }
        }
    }

    if (best_fit_block == -1) {
        // no suitable block found, try the last block (this may grow a chunks size)
        int64_t best_reuse = INT64_MIN;
        for (int c = 0; c < alloc->n_chunks; ++c) {
            struct tallocr_chunk * chunk = alloc->chunks[c];
            if (chunk->n_free_blocks > 0) {
                struct free_block * block = &chunk->free_blocks[chunk->n_free_blocks - 1];
                max_avail = MAX(max_avail, block->size);
                int64_t reuse_factor = chunk->max_size - block->offset - size;
                // reuse_factor < 0 : amount of extra memory that needs to be allocated
                // reuse_factor = 0 : allocated free space exactly matches tensor size
                // reuse_factor > 0 : superfluous memory that will remain unused
                bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
                bool better_fit = reuse_factor >= 0 && reuse_factor < best_reuse;
                if (block->size >= size && (better_reuse || better_fit)) {
                    best_fit_chunk = c;
                    best_fit_block = chunk->n_free_blocks - 1;
                    best_reuse = reuse_factor;
                }
            }
        }
    }

    if (best_fit_block == -1) {
        // none of the existing chunks have enough space left
        best_fit_chunk = ggml_dyn_tallocr_new_chunk(alloc, size);
        best_fit_block = 0;
    }
    if (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }

    struct tallocr_chunk * chunk = alloc->chunks[best_fit_chunk];
    struct free_block    * block = &chunk->free_blocks[best_fit_block];
    struct buffer_address  addr  = {.chunk = best_fit_chunk, .offset = block->offset };
    block->offset += size;
    block->size -= size;
    if (block->size == 0) {
        // remove block if empty
        ggml_dyn_tallocr_remove_block(chunk, best_fit_block);
    }

    AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
    add_allocated_tensor(alloc, addr, tensor);
    size_t cur_max = addr.offset + size;
    if (cur_max > chunk->max_size) {
        // sort allocated_tensors by chunk/offset
        for (int i = 0; i < 1024; i++) {
            for (int j = i + 1; j < 1024; j++) {
                if (ggml_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
                    const struct ggml_tensor * tmp_tensor = alloc->allocated_tensors[i].tensor;
                    struct buffer_address tmp_addr = alloc->allocated_tensors[i].addr;
                    alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
                    alloc->allocated_tensors[i].addr = alloc->allocated_tensors[j].addr;
                    alloc->allocated_tensors[j].tensor = tmp_tensor;
                    alloc->allocated_tensors[j].addr = tmp_addr;
                }
            }
        }
        GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
        for (int i = 0; i < 1024; i++) {
            if (alloc->allocated_tensors[i].tensor) {
                GGML_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].addr.chunk,
                    alloc->allocated_tensors[i].addr.offset,
                    alloc->allocated_tensors[i].addr.offset + ggml_nbytes(alloc->allocated_tensors[i].tensor),
                    ggml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
            }
        }
        GGML_LOG_DEBUG("\n");
    }
#endif

    chunk->max_size = MAX(chunk->max_size, addr.offset + size);

    return addr;

    GGML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
static void ggml_dyn_tallocr_free_bytes(struct ggml_dyn_tallocr * alloc, struct buffer_address addr, size_t size) {
    size = aligned_offset(NULL, size, alloc->alignment);
    if (addr.chunk < 0) {
        return;   // (a tensor a failed fixed-chunk plan could not place)
    }

    struct tallocr_chunk * chunk = alloc->chunks[addr.chunk];

    // see if we can merge with an existing block
    for (int i = 0; i < chunk->n_free_blocks; i++) {
        struct free_block * block = &chunk->free_blocks[i];
        // check if ptr is at the end of the block
        if (block->offset + block->size == addr.offset) {
            block->size += size;
            // check if we can merge with the next block
            if (i < chunk->n_free_blocks - 1) {
                struct free_block * next = &chunk->free_blocks[i+1];
                if (block->offset + block->size == next->offset) {
                    block->size += next->size;
                    ggml_dyn_tallocr_remove_block(chunk, i+1);
                }
            }
            return;
        }
        // check if ptr is at the beginning of the block
        if (addr.offset + size == block->offset) {
            block->offset = addr.offset;
            block->size += size;
            // check if we can merge with the previous block
            if (i > 0) {
                struct free_block * prev = &chunk->free_blocks[i-1];
                if (prev->offset + prev->size == block->offset) {
                    prev->size += block->size;
                    ggml_dyn_tallocr_remove_block(chunk, i);
                }
            }
            return;
        }
    }
    // otherwise, add a new block
    ggml_dyn_tallocr_insert_block(chunk, addr.offset, size);
}

static void ggml_dyn_tallocr_reset(struct ggml_dyn_tallocr * alloc) {
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS; i++) {
        free(alloc->chunks[i]);
        alloc->chunks[i] = NULL;
    }
    alloc->n_chunks = 0;
    alloc->failed   = false;
    for (int c = 0; c < alloc->n_fixed; c++) {   // (the pieces, empty, at their fixed sizes)
        struct tallocr_chunk * chunk = calloc(1, sizeof(struct tallocr_chunk));
        chunk->n_free_blocks = 1;
        chunk->free_blocks[0].offset = 0;
        chunk->free_blocks[0].size   = alloc->fixed[c];
        alloc->chunks[c] = chunk;
        alloc->n_chunks++;
    }
    if (alloc->n_fixed > 0) {   // (and the own growable chunk after them)
        struct tallocr_chunk * chunk = calloc(1, sizeof(struct tallocr_chunk));
        chunk->n_free_blocks = 1;
        chunk->free_blocks[0].offset = 0;
        chunk->free_blocks[0].size   = SIZE_MAX/2;
        alloc->chunks[alloc->n_fixed] = chunk;
        alloc->n_chunks++;
    }

#ifdef GGML_ALLOCATOR_DEBUG
    for (int i = 0; i < 1024; i++) {
        alloc->allocated_tensors[i].tensor = NULL;
    }
#endif
}

static struct ggml_dyn_tallocr * ggml_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
    struct ggml_dyn_tallocr * alloc = (struct ggml_dyn_tallocr *)malloc(sizeof(struct ggml_dyn_tallocr));

    *alloc = (struct ggml_dyn_tallocr) {
        /*.alignment      = */ alignment,
        /*.max_chunk_size = */ MIN(max_buffer_size, SIZE_MAX/2), // clamp to avoid overflows
        /*.chunks         = */ {NULL},
        /*.n_chunks       = */ 0,
        /*.fixed          = */ {0},
        /*.n_fixed        = */ 0,
        /*.failed         = */ false,
#ifdef GGML_ALLOCATOR_DEBUG
        /*.allocated_tensors = */ {{0}},
#endif
    };

    ggml_dyn_tallocr_reset(alloc);

    return alloc;
}

static void ggml_dyn_tallocr_free(struct ggml_dyn_tallocr * alloc) {
    for (int i = 0; i < alloc->n_chunks; ++i) {
        free(alloc->chunks[i]);
    }
    free(alloc);
}

static size_t ggml_dyn_tallocr_max_size(struct ggml_dyn_tallocr * alloc, int chunk) {
    return chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

struct vbuffer {
    ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
    bool external;   // the chunks belong to the caller (ggml_gallocr_set_fixed_chunks): never freed here
};

static void ggml_vbuffer_free(struct vbuffer * buf) {
    if (buf == NULL) {
        return;
    }
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS && !buf->external; ++i) {
        ggml_backend_buffer_free(buf->chunks[i]);
    }
    free(buf);
}

static size_t ggml_vbuffer_chunk_size(struct vbuffer * buf, int chunk) {
    return buf->chunks[chunk] ? ggml_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

static size_t ggml_vbuffer_size(struct vbuffer * buf) {
    size_t size = 0;
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        size += ggml_backend_buffer_get_size(buf->chunks[i]);
    }
    return size;
}

static struct vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, const struct ggml_dyn_tallocr * talloc, enum ggml_backend_buffer_usage usage) {
    struct vbuffer * buf = (struct vbuffer *)calloc(1, sizeof(struct vbuffer));
    if (buf == NULL) {
        return NULL;
    }

    for (int n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        if (buf->chunks[n] == NULL) {
            ggml_vbuffer_free(buf);
            return NULL;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    return buf;
}

static void ggml_vbuffer_tensor_alloc(struct vbuffer * buf, struct ggml_tensor * tensor, struct buffer_address buf_addr) {
    void * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    void * addr = (char *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}

static void ggml_vbuffer_reset(struct vbuffer * buf) {
    for (int i = 0; i < GGML_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
        ggml_backend_buffer_reset(buf->chunks[i]);
    }
}


/////////////////////////////////////

// graph allocator

struct hash_node {
    int n_children;
    int n_views;
    int buffer_id;
    int topo_idx;
    struct buffer_address addr;
    bool allocated;
};

struct tensor_alloc {
    int buffer_id;
    struct buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

struct leaf_alloc {
    struct tensor_alloc leaf;
};

struct node_alloc {
    struct tensor_alloc dst;
    struct tensor_alloc src[GGML_MAX_SRC];
};

#define GGML_GALLOCR_MAX_PLANS 16

struct ggml_gallocr_plan {
    struct node_alloc * node_allocs;
    int n_nodes;
    struct leaf_alloc * leaf_allocs;
    int n_leafs;
    uint64_t topo_hash;
    uint64_t topo_hash_nobuf;
    uint64_t last_used;
};

struct ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    struct vbuffer ** buffers; // [n_buffers]
    struct ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    int n_buffers;
    // A buffer that plans into fixed chunks has the caller's chunks in buffers[i] (external) and its own buffer kept
    // here (not freed: the switch back costs no allocation); force_replan = the next alloc_graph re-plans (a switch, or a failed plan)
    struct vbuffer ** own_buffers; // [n_buffers]
    bool force_replan;
    bool * own_no_grow; // [n_buffers]: a plan that needs more of the buffer's own memory than it has fails (the caller lends instead)

    struct ggml_hash_set hash_set;
    struct hash_node * hash_values; // [hash_set.size]

    struct node_alloc * node_allocs; // [node_allocs_cap]
    int n_nodes;
    int node_allocs_cap;

    struct leaf_alloc * leaf_allocs; // [leaf_allocs_cap]
    int n_leafs;
    int leaf_allocs_cap;

    uint64_t active_topo_hash;
    uint64_t active_topo_hash_nobuf;
    uint64_t plan_clock;
    struct ggml_gallocr_plan plan_cache[GGML_GALLOCR_MAX_PLANS];
};

static void ggml_gallocr_clear_plan_cache(ggml_gallocr_t galloc) {
    for (int k = 0; k < GGML_GALLOCR_MAX_PLANS; k++) {
        free(galloc->plan_cache[k].node_allocs);
        free(galloc->plan_cache[k].leaf_allocs);
        memset(&galloc->plan_cache[k], 0, sizeof(galloc->plan_cache[k]));
    }
    galloc->active_topo_hash = 0;
    galloc->active_topo_hash_nobuf = 0;
}

static int ggml_gallocr_topo_id(ggml_gallocr_t galloc, struct ggml_tensor * t, int * next_id) {
    if (t == NULL) {
        return 0;
    }
    size_t idx = ggml_hash_insert(&galloc->hash_set, t);
    if (idx != GGML_HASHSET_ALREADY_EXISTS) {
        int id = ++(*next_id);
        galloc->hash_values[idx].topo_idx = id;
        return id;
    }
    idx = ggml_hash_find(&galloc->hash_set, t);
    return galloc->hash_values[idx].topo_idx;
}

static uint64_t ggml_gallocr_graph_topo_hash(ggml_gallocr_t galloc, const struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    size_t max_t = (size_t)graph->n_leafs + (size_t)graph->n_nodes * (GGML_MAX_SRC + 2) + 16;
    size_t min_hash_size = max_t + max_t / 4;
    if (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.keys != NULL);
        free(galloc->hash_values);
        galloc->hash_values = malloc(sizeof(struct hash_node) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_values != NULL);
    }
    ggml_hash_set_reset(&galloc->hash_set);
    int next_id = 0;

    uint64_t h = 1469598103934665603ULL;
    const uint64_t prime = 1099511628211ULL;
    h = (h ^ (uint64_t)(uint32_t)graph->n_nodes) * prime;
    h = (h ^ (uint64_t)(uint32_t)graph->n_leafs) * prime;

    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int lid = ggml_gallocr_topo_id(galloc, leaf, &next_id);
        int vs_id = ggml_gallocr_topo_id(galloc, leaf->view_src, &next_id);
        int req_lb = leaf_buffer_ids != NULL ? leaf_buffer_ids[i] : 0;
        uint64_t ltag = ((uint64_t)(uint32_t)lid << 32) ^
                        ((uint64_t)(uint32_t)vs_id << 16) ^
                        ((uint64_t)(uint32_t)(req_lb + 2) << 8) ^
                        ((uint64_t)((leaf->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) != 0 ? (leaf->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) : 0) << 2) ^
                        ((uint64_t)(leaf->buffer != NULL) << 1) ^
                        (uint64_t)(leaf->data != NULL);
        h = (h ^ ltag) * prime;
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int nid = ggml_gallocr_topo_id(galloc, node, &next_id);
        int vs_id = ggml_gallocr_topo_id(galloc, node->view_src, &next_id);
        int req_b = node_buffer_ids != NULL ? node_buffer_ids[i] : 0;
        const bool can_inp = ggml_op_can_inplace(node->op);
        uint64_t tag = ((uint64_t)(uint32_t)nid << 32) ^
                       ((uint64_t)(uint32_t)vs_id << 16) ^
                       ((uint64_t)node->op << 8) ^
                       ((uint64_t)node->type << 2) ^
                       ((uint64_t)(node->view_offs == 0) << 1) ^
                       (uint64_t)(node->data != NULL);
        tag ^= ((uint64_t)(uint32_t)(req_b + 2) << 48) ^
               ((uint64_t)(node->buffer != NULL) << 47) ^
               ((uint64_t)((node->flags & GGML_TENSOR_FLAG_INPUT) != 0) << 46) ^
               ((uint64_t)((node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) << 45);
        h = (h ^ tag) * prime;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * s = node->src[j];
            if (s == NULL) {
                continue;
            }
            int sid = ggml_gallocr_topo_id(galloc, s, &next_id);
            int svs_id = ggml_gallocr_topo_id(galloc, s->view_src, &next_id);
            const bool same_lay = can_inp && ggml_are_same_layout(node, s);
            uint64_t stag = ((uint64_t)(uint32_t)sid << 32) ^
                            ((uint64_t)(uint32_t)svs_id << 16) ^
                            ((uint64_t)(j + 1) << 8) ^
                            ((uint64_t)same_lay << 5) ^
                            ((uint64_t)((s->flags & GGML_TENSOR_FLAG_INPUT) != 0) << 4) ^
                            ((uint64_t)((s->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) << 3) ^
                            ((uint64_t)(s->buffer != NULL) << 2) ^
                            ((uint64_t)(s->data != NULL) << 1) ^
                            (uint64_t)(s->view_src != NULL && s->view_src->data == s->data);
            h = (h ^ stag) * prime;
        }
    }
    return h ? h : 1ULL;
}

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs) {
    ggml_gallocr_t galloc = (ggml_gallocr_t)calloc(1, sizeof(struct ggml_gallocr));
    GGML_ASSERT(galloc != NULL);

    galloc->bufts = calloc(n_bufs, sizeof(ggml_backend_buffer_type_t));
    GGML_ASSERT(galloc->bufts != NULL);

    galloc->buffers = calloc(n_bufs, sizeof(struct vbuffer *));
    GGML_ASSERT(galloc->buffers != NULL);

    galloc->buf_tallocs = calloc(n_bufs, sizeof(struct ggml_dyn_tallocr *));
    GGML_ASSERT(galloc->buf_tallocs != NULL);

    galloc->own_buffers = calloc(n_bufs, sizeof(struct vbuffer *));
    GGML_ASSERT(galloc->own_buffers != NULL);

    galloc->own_no_grow = calloc(n_bufs, sizeof(bool));
    GGML_ASSERT(galloc->own_no_grow != NULL);

    for (int i = 0; i < n_bufs; i++) {
        galloc->bufts[i] = bufts[i];
        galloc->buffers[i] = NULL;

        // check if the same buffer type is used multiple times and reuse the same allocator
        for (int j = 0; j < i; j++) {
            if (bufts[i] == bufts[j]) {
                galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
                break;
            }
        }

        if (galloc->buf_tallocs[i] == NULL) {
            size_t alignment = ggml_backend_buft_get_alignment(bufts[i]);
            size_t max_size = ggml_backend_buft_get_max_size(bufts[i]);
            galloc->buf_tallocs[i] = ggml_dyn_tallocr_new(alignment, max_size);
        }
    }
    galloc->n_buffers = n_bufs;

    return galloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    return ggml_gallocr_new_n(&buft, 1);
}

void ggml_gallocr_free(ggml_gallocr_t galloc) {
    if (galloc == NULL) {
        return;
    }

    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers != NULL) {
            // skip if already freed
            bool freed = false;
            for (int j = 0; j < i; j++) {
                if (galloc->buffers[j] == galloc->buffers[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) {
                ggml_vbuffer_free(galloc->buffers[i]);
            }
        }
        if (galloc->buf_tallocs != NULL) {
            // skip if already freed
            bool freed = false;
            for (int j = 0; j < i; j++) {
                if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                    freed = true;
                    break;
                }
            }
            if (!freed) {
                ggml_dyn_tallocr_free(galloc->buf_tallocs[i]);
            }
        }
    }

    for (int i = 0; i < galloc->n_buffers; i++) {   // (a buffer kept aside while fixed chunks were set)
        ggml_vbuffer_free(galloc->own_buffers[i]);
    }
    ggml_gallocr_clear_plan_cache(galloc);
    free(galloc->own_buffers);
    free(galloc->own_no_grow);
    ggml_hash_set_free(&galloc->hash_set);
    free(galloc->hash_values);
    free(galloc->bufts);
    free(galloc->buffers);
    free(galloc->buf_tallocs);
    free(galloc->node_allocs);
    free(galloc->leaf_allocs);
    free(galloc);
}

typedef struct ggml_gallocr * ggml_gallocr_t;

static struct hash_node * ggml_gallocr_hash_get(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    size_t i = ggml_hash_find_or_insert(&galloc->hash_set, t);
    return &galloc->hash_values[i];
}

static bool ggml_gallocr_is_own(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return ggml_gallocr_hash_get(galloc, t)->allocated;
}

static bool ggml_gallocr_is_allocated(ggml_gallocr_t galloc, struct ggml_tensor * t) {
    return t->data != NULL // tensor data already set externally
        || t->buffer // tensor on external buffer (but not yet allocated)
        || ggml_gallocr_is_own(galloc, t); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
static void ggml_gallocr_free_extra_space(ggml_gallocr_t galloc, struct ggml_tensor * node, struct ggml_tensor * parent) {
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);

    size_t parent_size = ggml_backend_buft_get_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
    size_t node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);

    GGML_ASSERT(parent_size >= node_size);

    // note: we want after the freeing the chunks to continue to be aligned
    struct ggml_dyn_tallocr * p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    parent_size = aligned_offset(NULL, parent_size, p_alloc->alignment);
    node_size = aligned_offset(NULL, node_size, p_alloc->alignment);

    if (parent_size > node_size) {
        struct buffer_address p_addr = p_hn->addr;
        p_addr.offset += node_size;
        size_t extra_size = parent_size - node_size;
        AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, node->name);
        ggml_dyn_tallocr_free_bytes(p_alloc, p_addr, extra_size);
    }
}

// GGML_ALLOC_DUMP_TOP=N (diagnostic, read once; unset or 0 = nothing, no cost). For every graph the
// scheduler reserves (a decode word, the prompt graph, the wave graph) it prints, per buffer, how many bytes THIS graph needs (the
// allocator is reset for each reserve, so this is the graph's own peak, not the running maximum of the buffer) next to the size
// the buffer has now, and the N largest tensors the graph allocated (the largest allocations, not necessarily alive at the peak).
// Printed to stderr (the tools drop the library's INFO lines).
#define GGML_ALLOC_DUMP_MAX 32
struct ggml_alloc_dump_item { size_t size; int buffer_id; int op; char name[GGML_MAX_NAME]; };
static struct ggml_alloc_dump_item g_alloc_dump[GGML_ALLOC_DUMP_MAX];
static int ggml_alloc_dump_n(void) {
    static int n = -1;
    if (n < 0) {
        const char * e = getenv("GGML_ALLOC_DUMP_TOP");
        n = e ? atoi(e) : 0;
        n = n < 0 ? 0 : n > GGML_ALLOC_DUMP_MAX ? GGML_ALLOC_DUMP_MAX : n;
    }
    return n;
}
// the same allocations grouped by name without the layer number ("ffn_moe_down-17" -> "ffn_moe_down"): how many, the biggest one, the sum
#define GGML_ALLOC_DUMP_CLASSES 512
struct ggml_alloc_dump_class { char base[40]; int n; size_t one; size_t total; int buffer_id; };
static struct ggml_alloc_dump_class g_alloc_class[GGML_ALLOC_DUMP_CLASSES];
static int g_alloc_class_n = 0;
static void ggml_alloc_dump_class_add(const struct ggml_tensor * node, int buffer_id, size_t size) {
    char base[40];
    snprintf(base, sizeof(base), "%s", node->name);
    size_t l = strlen(base);
    size_t e = l;
    while (e > 0 && base[e-1] >= '0' && base[e-1] <= '9') { e--; }
    if (e < l && e > 0 && base[e-1] == '-') { base[e-1] = 0; }
    for (int i = 0; i < g_alloc_class_n; i++) {
        if (g_alloc_class[i].buffer_id == buffer_id && strcmp(g_alloc_class[i].base, base) == 0) {
            g_alloc_class[i].n++; g_alloc_class[i].total += size;
            if (size > g_alloc_class[i].one) { g_alloc_class[i].one = size; }
            return;
        }
    }
    if (g_alloc_class_n < GGML_ALLOC_DUMP_CLASSES) {
        struct ggml_alloc_dump_class * c = &g_alloc_class[g_alloc_class_n++];
        snprintf(c->base, sizeof(c->base), "%s", base);
        c->n = 1; c->one = size; c->total = size; c->buffer_id = buffer_id;
    }
}
static void ggml_alloc_dump_note(const struct ggml_tensor * node, int buffer_id, size_t size, int n) {
    int worst = 0;   // the smallest of the kept ones is replaced
    for (int i = 1; i < n; i++) {
        if (g_alloc_dump[i].size < g_alloc_dump[worst].size) {
            worst = i;
        }
    }
    if (size > g_alloc_dump[worst].size) {
        g_alloc_dump[worst].size      = size;
        g_alloc_dump[worst].buffer_id = buffer_id;
        g_alloc_dump[worst].op        = (int) node->op;
        snprintf(g_alloc_dump[worst].name, GGML_MAX_NAME, "%s", node->name);
    }
}

static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);

    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_impl_is_view(node)) {
        hn->allocated = true;
        assert(hn->addr.offset == 0);

        // try to reuse a parent's buffer (inplace)
        if (ggml_op_can_inplace(node->op)) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                struct ggml_tensor * parent = node->src[i];
                if (parent == NULL) {
                    continue;
                }

                // if the node's data is external, then we cannot re-use it
                if (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
                    continue;
                }

                // outputs cannot be reused
                if (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != NULL && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, node->name);
                    continue;
                }

                if (!ggml_are_same_layout(node, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, node->name);
                    continue;
                }

                struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
                if (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    if (ggml_impl_is_view(parent)) {
                        struct ggml_tensor * view_src = parent->view_src;
                        struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                            assert(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->addr = p_hn->addr;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            ggml_gallocr_free_extra_space(galloc, node, view_src);
                            return;
                        }
                    } else {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->addr = p_hn->addr;
                        p_hn->allocated = false; // avoid freeing the parent
                        ggml_gallocr_free_extra_space(galloc, node, parent);
                        return;
                    }
                }
            }
        }
        // allocate tensor from the buffer
        struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, node);
        hn->buffer_id = buffer_id;
        hn->addr = ggml_dyn_tallocr_alloc(alloc, size, node);
        if (ggml_alloc_dump_n() > 0) {
            ggml_alloc_dump_note(node, buffer_id, size, ggml_alloc_dump_n());
            ggml_alloc_dump_class_add(node, buffer_id, size);
        }
    }
}

static void ggml_gallocr_free_node(ggml_gallocr_t galloc, struct ggml_tensor * node) {
    // graph outputs are never freed
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        AT_PRINTF("not freeing output %s\n", node->name);
        return;
    }

    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    int buffer_id = hn->buffer_id;
    struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
    size_t size = ggml_backend_buft_get_alloc_size(buft, node);

    AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
        __func__, node->name, hn->addr.chunk, hn->addr.offset, size, alloc->chunks[hn->addr.chunk]->n_free_blocks);
#ifdef GGML_ALLOCATOR_DEBUG
    remove_allocated_tensor(alloc, hn->addr, node);
#endif

    ggml_dyn_tallocr_free_bytes(alloc, hn->addr, size);
    hn->allocated = false;
}

static int get_node_buffer_id(const int * node_buffer_ids, int i) {
    return node_buffer_ids ? node_buffer_ids[i] : 0;
}

static void ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_values, 0, sizeof(struct hash_node) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
    }

    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        if (ggml_impl_is_view(node) && node->op != GGML_OP_NONE) {
            struct ggml_tensor * view_src = node->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_gallocr_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                ggml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
            }
        }
    }

    // allocate tensors
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int buffer_id = get_node_buffer_id(node_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            ggml_gallocr_allocate_node(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_node(galloc, node, buffer_id);

        AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(node), node->name);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            AT_PRINTF("%s", parent->name);
            if (j < GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");

        // update parents
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            if (p_hn->n_children == 0 && p_hn->n_views == 0) {
                if (ggml_impl_is_view(parent)) {
                    struct ggml_tensor * view_src = parent->view_src;
                    struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_node(galloc, view_src);
                    }
                }
                else if (p_hn->allocated) {
                    ggml_gallocr_free_node(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
    }
}

// the buffer type of the last buffer a reserve could not allocate (see ggml_gallocr_last_failed_buft)
static ggml_backend_buffer_type_t ggml_gallocr_failed_buft = NULL;

ggml_backend_buffer_type_t ggml_gallocr_last_failed_buft(void) {
    return ggml_gallocr_failed_buft;
}

static bool ggml_gallocr_reserve_n_impl(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, bool no_alloc,
        ggml_gallocr_before_realloc_fn before_realloc, void * before_realloc_user_data) {
    ggml_gallocr_failed_buft = NULL;
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    if (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.keys != NULL);

        free(galloc->hash_values);
        galloc->hash_values = malloc(sizeof(struct hash_node) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_values != NULL);
    }

    // reset allocators
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    if (ggml_alloc_dump_n() > 0) {
        memset(g_alloc_dump, 0, sizeof(g_alloc_dump));
        g_alloc_class_n = 0;
    }
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);
    galloc->force_replan = false;
    for (int i = 0; i < galloc->n_buffers; i++) {   // (a plan into fixed chunks that does not fit fails here)
        if (galloc->buf_tallocs[i]->failed) {
            galloc->force_replan = true;   // (the partial plan is never used)
            return false;
        }
    }
    if (ggml_alloc_dump_n() > 0) {
        fprintf(stderr, "GGML_ALLOC_DUMP: reserve (%s) of a graph with %d nodes, %d leafs:\n", no_alloc ? "size query" : "allocating", graph->n_nodes, graph->n_leafs);
        for (int i = 0; i < galloc->n_buffers; i++) {
            size_t need = 0;
            for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
                need += galloc->buf_tallocs[i]->chunks[c]->max_size;
            }
            fprintf(stderr, "GGML_ALLOC_DUMP:   buffer %d %-14s this graph needs %9.3f MiB, the buffer is now %9.3f MiB (%d chunks)\n", i,
                    ggml_backend_buft_name(galloc->bufts[i]), need/1048576.0, galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i])/1048576.0 : 0.0,
                    galloc->buf_tallocs[i]->n_chunks);
        }
        for (int k = 0; k < ggml_alloc_dump_n(); k++) {   // largest first
            int best = -1;
            for (int i = 0; i < ggml_alloc_dump_n(); i++) {
                if (g_alloc_dump[i].size > 0 && (best < 0 || g_alloc_dump[i].size > g_alloc_dump[best].size)) {
                    best = i;
                }
            }
            if (best < 0) {
                break;
            }
            fprintf(stderr, "GGML_ALLOC_DUMP:     %9.3f MiB  buffer %d  %-16s %s\n", g_alloc_dump[best].size/1048576.0, g_alloc_dump[best].buffer_id,
                    ggml_op_name((enum ggml_op) g_alloc_dump[best].op), g_alloc_dump[best].name);
            g_alloc_dump[best].size = 0;
        }
        fprintf(stderr, "GGML_ALLOC_DUMP:   by name (every allocation of the graph, not only those alive at the peak): count, biggest one, sum\n");
        for (int k = 0; k < ggml_alloc_dump_n(); k++) {   // by the biggest single allocation
            int best = -1;
            for (int i = 0; i < g_alloc_class_n; i++) {
                if (g_alloc_class[i].one > 0 && (best < 0 || g_alloc_class[i].one > g_alloc_class[best].one)) {
                    best = i;
                }
            }
            if (best < 0) {
                break;
            }
            fprintf(stderr, "GGML_ALLOC_DUMP:     %-34s buffer %d  x%-4d biggest %9.3f MiB  sum %10.3f MiB\n", g_alloc_class[best].base, g_alloc_class[best].buffer_id,
                    g_alloc_class[best].n, g_alloc_class[best].one/1048576.0, g_alloc_class[best].total/1048576.0);
            g_alloc_class[best].one = 0;
        }
    }

    // set the node_allocs from the hash table
    if (galloc->node_allocs_cap < graph->n_nodes) {
        free(galloc->node_allocs);
        galloc->node_allocs = calloc(graph->n_nodes, sizeof(struct node_alloc));
        GGML_ASSERT(galloc->node_allocs != NULL);
        galloc->node_allocs_cap = graph->n_nodes;
    }
    galloc->n_nodes = graph->n_nodes;
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        if (node->view_src || node->data) {
            node_alloc->dst.buffer_id = -1;
            node_alloc->dst.addr = GGML_BUFFER_ADDRESS_INVALID;
            node_alloc->dst.size_max = 0;
        } else {
            struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
            node_alloc->dst.buffer_id = hn->buffer_id;
            node_alloc->dst.addr = hn->addr;
            node_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (!src || src->view_src || src->data) {
                node_alloc->src[j].buffer_id = -1;
                node_alloc->src[j].addr = GGML_BUFFER_ADDRESS_INVALID;
                node_alloc->src[j].size_max = 0;
            } else {
                struct hash_node * hn = ggml_gallocr_hash_get(galloc, src);
                node_alloc->src[j].buffer_id = hn->buffer_id;
                node_alloc->src[j].addr = hn->addr;
                node_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
    if (galloc->leaf_allocs_cap < graph->n_leafs) {
        free(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, sizeof(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != NULL);
        galloc->leaf_allocs_cap = graph->n_leafs;
    }
    galloc->n_leafs = graph->n_leafs;
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct hash_node * hn = ggml_gallocr_hash_get(galloc, leaf);
        if (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.addr = GGML_BUFFER_ADDRESS_INVALID;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } else {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.addr = hn->addr;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }

    bool synced_before_realloc = false;
    bool did_buf_realloc = false;

    // reallocate buffers if needed
    for (int i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        for (int j = 0; j < i; j++) {
            if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                break;
            }
        }

        if (galloc->buf_tallocs[i]->n_fixed > 0) {
            // (the pieces are the caller's; the own chunk is the allocator's own buffer, grown when the plan needs more)
            const int    nf   = galloc->buf_tallocs[i]->n_fixed;
            const size_t need = galloc->buf_tallocs[i]->chunks[nf]->max_size;
            struct vbuffer * own = galloc->own_buffers[i];
            const size_t cur  = own && own->chunks[0] ? ggml_backend_buffer_get_size(own->chunks[0]) : 0;
            if (need > cur && galloc->own_no_grow[i] && !no_alloc) {
                galloc->force_replan = true;   // (the caller asked: no growth of the real buffer - it lends more instead)
                return false;
            }
            if (need > cur) {
                if (!no_alloc && before_realloc != NULL && !synced_before_realloc) {
                    before_realloc(before_realloc_user_data);
                    synced_before_realloc = true;
                }
                did_buf_realloc = true;
                ggml_vbuffer_free(own);
                galloc->own_buffers[i] = NULL;
                galloc->buffers[i]->chunks[nf] = NULL;
                if (!no_alloc) {
                    own = (struct vbuffer *) calloc(1, sizeof(struct vbuffer));
                    GGML_ASSERT(own != NULL);
                    own->chunks[0] = ggml_backend_buft_alloc_buffer(galloc->bufts[i], need);
                    if (own->chunks[0] == NULL) {
                        free(own);
                        GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), need);
                        ggml_gallocr_failed_buft = galloc->bufts[i];
                        galloc->force_replan = true;
                        return false;
                    }
                    ggml_backend_buffer_set_usage(own->chunks[0], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                    galloc->own_buffers[i] = own;
                }
            }
            galloc->buffers[i]->chunks[nf] = galloc->own_buffers[i] ? galloc->own_buffers[i]->chunks[0] : NULL;
            continue;
        }
        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        bool realloc = galloc->buffers[i] == NULL;
        size_t new_size = 0;
        for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            size_t cur_chunk_size = galloc->buffers[i] ? ggml_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
            size_t new_chunk_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
            new_size += new_chunk_size;
            if (new_chunk_size > cur_chunk_size) {
                realloc = true;
            }
        }
        if (realloc && galloc->own_no_grow[i] && !no_alloc && galloc->buffers[i] != NULL) {
            galloc->force_replan = true;   // (the caller asked: no growth of this buffer - it lends instead)
            return false;
        }
        if (realloc) {
            if (!no_alloc && before_realloc != NULL && !synced_before_realloc && galloc->buffers[i] != NULL) {
                before_realloc(before_realloc_user_data);
                synced_before_realloc = true;
            }
            did_buf_realloc = true;
#ifndef NDEBUG
            {
                size_t cur_size = galloc->buffers[i] ? ggml_vbuffer_size(galloc->buffers[i]) : 0;
                if (cur_size > 0) {
                    GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n",
                        __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
                }
            }
#endif
            ggml_vbuffer_free(galloc->buffers[i]);
            if (no_alloc) {
                galloc->buffers[i] = NULL;
            } else {
                galloc->buffers[i] = ggml_vbuffer_alloc(galloc->bufts[i], galloc->buf_tallocs[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (galloc->buffers[i] == NULL) {
                    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                    ggml_gallocr_failed_buft = galloc->bufts[i];
                    galloc->force_replan = true;   // (the plan is kept but its buffer is gone: the next alloc_graph must plan and allocate again)
                    return false;
                }
            }
        }
    }

    if (no_alloc) {
        galloc->active_topo_hash = 0;
        return true;
    }

    if (did_buf_realloc) {
        ggml_gallocr_clear_plan_cache(galloc);
    }

    const uint64_t th = ggml_gallocr_graph_topo_hash(galloc, graph, node_buffer_ids, leaf_buffer_ids);
    const uint64_t th_nobuf = (node_buffer_ids == NULL && leaf_buffer_ids == NULL)
        ? th
        : ggml_gallocr_graph_topo_hash(galloc, graph, NULL, NULL);
    galloc->active_topo_hash = th;
    galloc->active_topo_hash_nobuf = th_nobuf;

    // Save or update matching topology slot in plan_cache
    int slot_idx = -1;
    for (int k = 0; k < GGML_GALLOCR_MAX_PLANS; k++) {
        if (galloc->plan_cache[k].topo_hash == th &&
            galloc->plan_cache[k].n_nodes == graph->n_nodes &&
            galloc->plan_cache[k].n_leafs == graph->n_leafs) {
            slot_idx = k;
            break;
        }
    }
    if (slot_idx < 0) {
        slot_idx = 0;
        for (int k = 1; k < GGML_GALLOCR_MAX_PLANS; k++) {
            if (galloc->plan_cache[k].topo_hash == 0) {
                slot_idx = k;
                break;
            }
            if (galloc->plan_cache[k].last_used < galloc->plan_cache[slot_idx].last_used) {
                slot_idx = k;
            }
        }
    }
    struct ggml_gallocr_plan * sp = &galloc->plan_cache[slot_idx];
    if (sp->n_nodes != graph->n_nodes) {
        free(sp->node_allocs);
        sp->node_allocs = graph->n_nodes > 0 ? (struct node_alloc *) malloc((size_t)graph->n_nodes * sizeof(struct node_alloc)) : NULL;
        GGML_ASSERT(graph->n_nodes == 0 || sp->node_allocs != NULL);
        sp->n_nodes = graph->n_nodes;
    }
    if (sp->n_leafs != graph->n_leafs) {
        free(sp->leaf_allocs);
        sp->leaf_allocs = graph->n_leafs > 0 ? (struct leaf_alloc *) malloc((size_t)graph->n_leafs * sizeof(struct leaf_alloc)) : NULL;
        GGML_ASSERT(graph->n_leafs == 0 || sp->leaf_allocs != NULL);
        sp->n_leafs = graph->n_leafs;
    }
    if (graph->n_nodes > 0) {
        memcpy(sp->node_allocs, galloc->node_allocs, (size_t)graph->n_nodes * sizeof(struct node_alloc));
    }
    if (graph->n_leafs > 0) {
        memcpy(sp->leaf_allocs, galloc->leaf_allocs, (size_t)graph->n_leafs * sizeof(struct leaf_alloc));
    }
    sp->topo_hash = th;
    sp->topo_hash_nobuf = th_nobuf;
    sp->last_used = ++galloc->plan_clock;

    return true;
}

void ggml_gallocr_reserve_n_size(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids, size_t * sizes) {
    const bool ok = ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ true, NULL, NULL);
    bool any_fixed = false;
    for (int i = 0; i < galloc->n_buffers; i++) {
        any_fixed = any_fixed || galloc->buf_tallocs[i]->n_fixed > 0;
    }
    GGML_ASSERT(ok || any_fixed);
    for (int i = 0; i < galloc->n_buffers && !ok; i++) {
        sizes[i] = galloc->buf_tallocs[i]->failed ? SIZE_MAX : 0;
    }
    for (int i = 0; i < galloc->n_buffers && ok; i++) {
        if (galloc->buf_tallocs[i]->n_fixed > 0) {   // (with pieces, what the own buffer needs - the rest is lent)
            sizes[i] = galloc->buf_tallocs[i]->chunks[galloc->buf_tallocs[i]->n_fixed]->max_size;
            continue;
        }
        sizes[i] = 0;
        for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
            sizes[i] += galloc->buf_tallocs[i]->chunks[c]->max_size;
        }
    }
}

bool ggml_gallocr_reserve_n_ex(
        ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids,
        ggml_gallocr_before_realloc_fn before_realloc, void * before_realloc_user_data) {
    return ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false, before_realloc, before_realloc_user_data);
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    return ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false, NULL, NULL);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph *graph) {
    return ggml_gallocr_reserve_n(galloc, graph, NULL, NULL);
}

static void ggml_gallocr_init_tensor(ggml_gallocr_t galloc, struct ggml_tensor * tensor, struct tensor_alloc * tensor_alloc) {
    int buffer_id = tensor_alloc->buffer_id;
    assert(tensor->data || tensor->view_src || ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);

    if (tensor->view_src != NULL) {
        if (tensor->buffer == NULL) {
            assert(tensor_alloc->addr.offset == SIZE_MAX);
            if (tensor->view_src->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
            ggml_backend_view_init(tensor);
        }
    } else {
        if (tensor->data == NULL) {
            assert(tensor_alloc->addr.offset != SIZE_MAX);
            assert(ggml_backend_buft_get_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
            ggml_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
        } else {
            if (tensor->buffer == NULL) {
                // this tensor was allocated without ggml-backend
                return;
            }
        }
    }
}

// Diagnostic: why an automatic re-plan happened - 0 node count, 1 leaf count, 2 a tensor that had memory of its own in the
// last plan (pre-allocated) has none now, 3 a tensor is bigger than its place in the last plan
static int g_gallocr_why[4];
void ggml_gallocr_get_realloc_reasons(int why[4]) { for (int i = 0; i < 4; i++) { why[i] = g_gallocr_why[i]; } }

static bool ggml_gallocr_tensor_fits_talloc(ggml_gallocr_t galloc, struct ggml_tensor * node, const struct tensor_alloc * talloc,
                                            int req_buf_id, bool count_why) {
    if (node->view_src != NULL) {
        if (node->buffer == NULL && talloc->addr.offset != SIZE_MAX) {
            return false;
        }
        return true;
    }
    if (node->data != NULL) {
        return true;
    }
    if (talloc->buffer_id < 0 || talloc->addr.offset == SIZE_MAX) {
        if (count_why) {
            g_gallocr_why[2]++;
        }
        return false;
    }
    if (req_buf_id >= 0 && talloc->buffer_id != req_buf_id && galloc->bufts[talloc->buffer_id] != galloc->bufts[req_buf_id]) {
        return false;
    }
    const size_t node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], node);
    if (talloc->size_max < node_size) {
        if (count_why) {
            g_gallocr_why[3]++;
        }
        return false;
    }
    return true;
}

static bool ggml_gallocr_plan_fits_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph,
                                         const struct node_alloc * node_allocs, int n_nodes,
                                         const struct leaf_alloc * leaf_allocs, int n_leafs,
                                         const int * node_buffer_ids, const int * leaf_buffer_ids,
                                         bool count_why) {
    if (n_nodes != graph->n_nodes) {
        if (count_why) {
            g_gallocr_why[0]++;
        }
        return false;
    }
    if (n_leafs != graph->n_leafs) {
        if (count_why) {
            g_gallocr_why[1]++;
        }
        return false;
    }
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        const struct node_alloc * na = &node_allocs[i];
        const int req_b = node_buffer_ids != NULL ? node_buffer_ids[i] : -1;
        if (!ggml_gallocr_tensor_fits_talloc(galloc, node, &na->dst, req_b, count_why)) {
            return false;
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            if (!ggml_gallocr_tensor_fits_talloc(galloc, src, &na->src[j], -1, count_why)) {
                return false;
            }
        }
    }
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        const int req_lb = leaf_buffer_ids != NULL ? leaf_buffer_ids[i] : -1;
        if (!ggml_gallocr_tensor_fits_talloc(galloc, leaf, &leaf_allocs[i].leaf, req_lb, count_why)) {
            return false;
        }
    }
    return true;
}

static bool ggml_gallocr_try_restore_cached_plan(ggml_gallocr_t galloc, struct ggml_cgraph * graph,
                                                 const int * node_buffer_ids, const int * leaf_buffer_ids) {
    if (galloc->force_replan) {
        return false;
    }
    const bool nobuf = (node_buffer_ids == NULL && leaf_buffer_ids == NULL);
    const uint64_t th = ggml_gallocr_graph_topo_hash(galloc, graph, node_buffer_ids, leaf_buffer_ids);
    if ((nobuf ? galloc->active_topo_hash_nobuf : galloc->active_topo_hash) == th &&
        ggml_gallocr_plan_fits_graph(galloc, graph, galloc->node_allocs, galloc->n_nodes,
                                     galloc->leaf_allocs, galloc->n_leafs, node_buffer_ids, leaf_buffer_ids, false)) {
        return true;
    }
    for (int k = 0; k < GGML_GALLOCR_MAX_PLANS; k++) {
        struct ggml_gallocr_plan * sp = &galloc->plan_cache[k];
        if ((nobuf ? sp->topo_hash_nobuf : sp->topo_hash) != th || sp->n_nodes != graph->n_nodes || sp->n_leafs != graph->n_leafs) {
            continue;
        }
        if (!ggml_gallocr_plan_fits_graph(galloc, graph, sp->node_allocs, sp->n_nodes,
                                          sp->leaf_allocs, sp->n_leafs, node_buffer_ids, leaf_buffer_ids, false)) {
            continue;
        }
        if (galloc->node_allocs_cap < sp->n_nodes) {
            free(galloc->node_allocs);
            galloc->node_allocs = (struct node_alloc *) calloc((size_t)sp->n_nodes, sizeof(struct node_alloc));
            GGML_ASSERT(galloc->node_allocs != NULL);
            galloc->node_allocs_cap = sp->n_nodes;
        }
        if (galloc->leaf_allocs_cap < sp->n_leafs) {
            free(galloc->leaf_allocs);
            galloc->leaf_allocs = (struct leaf_alloc *) calloc((size_t)sp->n_leafs, sizeof(struct leaf_alloc));
            GGML_ASSERT(galloc->leaf_allocs != NULL);
            galloc->leaf_allocs_cap = sp->n_leafs;
        }
        if (sp->n_nodes > 0) {
            memcpy(galloc->node_allocs, sp->node_allocs, (size_t)sp->n_nodes * sizeof(struct node_alloc));
        }
        if (sp->n_leafs > 0) {
            memcpy(galloc->leaf_allocs, sp->leaf_allocs, (size_t)sp->n_leafs * sizeof(struct leaf_alloc));
        }
        galloc->n_nodes = sp->n_nodes;
        galloc->n_leafs = sp->n_leafs;
        galloc->active_topo_hash = sp->topo_hash;
        galloc->active_topo_hash_nobuf = sp->topo_hash_nobuf;
        sp->last_used = ++galloc->plan_clock;
        return true;
    }
    // Count diagnostic reason once against active plan when all cached plans missed
    (void) ggml_gallocr_plan_fits_graph(galloc, graph, galloc->node_allocs, galloc->n_nodes,
                                        galloc->leaf_allocs, galloc->n_leafs, node_buffer_ids, leaf_buffer_ids, true);
    return false;
}

bool ggml_gallocr_alloc_graph_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    if (!ggml_gallocr_try_restore_cached_plan(galloc, graph, node_buffer_ids, leaf_buffer_ids)) {
        if (galloc->n_buffers == 1 && node_buffer_ids == NULL && leaf_buffer_ids == NULL) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            if (!ggml_gallocr_reserve(galloc, graph)) {
                return false;
            }
        } else {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            return false;
        }
    }

    // reset buffers
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers[i] != NULL) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            ggml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, node, &node_alloc->dst);
    }

    return true;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    return ggml_gallocr_alloc_graph_n(galloc, graph, NULL, NULL);
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

    if (galloc->buffers[buffer_id] == NULL) {
        return 0;
    }

    for (int i = 0; i < buffer_id; i++) {
        if (galloc->buffers[i] == galloc->buffers[buffer_id]) {
            // this buffer is the same as a previous one due to the same buffer type being used multiple times
            // only return the buffer size the first time it appears to avoid double counting
            return 0;
        }
    }

    return ggml_vbuffer_size(galloc->buffers[buffer_id]);
}

// See ggml-alloc.h
void ggml_gallocr_clear_host_buffers(ggml_gallocr_t galloc) {
    for (int i = 0; i < galloc->n_buffers; i++) {
        bool seen = galloc->buffers[i] == NULL;
        for (int j = 0; j < i && !seen; j++) {
            seen = galloc->buffers[j] == galloc->buffers[i];
        }
        for (int c = 0; c < GGML_VBUFFER_MAX_CHUNKS && !seen; c++) {
            ggml_backend_buffer_t b = galloc->buffers[i]->chunks[c];
            if (b != NULL && ggml_backend_buffer_is_host(b)) {
                ggml_backend_buffer_clear(b, 0);
            }
        }
    }
}

// See ggml-alloc.h
bool ggml_gallocr_set_fixed_chunks(ggml_gallocr_t galloc, int buffer_id, ggml_backend_buffer_t * chunks, int n_chunks, bool own_may_grow) {
    GGML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);
    if (n_chunks < 0 || n_chunks > GGML_VBUFFER_MAX_CHUNKS) {
        return false;
    }
    struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    for (int j = 0; j < galloc->n_buffers; j++) {
        if (j != buffer_id && galloc->buf_tallocs[j] == alloc) {
            return false;   // (a buffer type used by two buffer ids shares one allocator: not supported)
        }
    }
    if (alloc->n_fixed > 0) {   // the caller's chunks go; the allocator's own buffer comes back
        ggml_vbuffer_free(galloc->buffers[buffer_id]);   // (external: only the struct)
        galloc->buffers[buffer_id]     = galloc->own_buffers[buffer_id];
        galloc->own_buffers[buffer_id] = NULL;
        alloc->n_fixed = 0;
    }
    if (n_chunks > 0) {
        if (n_chunks > GGML_VBUFFER_MAX_CHUNKS - 1) {
            return false;   // (one chunk is the own buffer's)
        }
        struct vbuffer * vb = (struct vbuffer *) calloc(1, sizeof(struct vbuffer));
        GGML_ASSERT(vb != NULL);
        vb->external = true;   // (it owns nothing: the pieces are the caller's, its last chunk the own buffer's)
        for (int c = 0; c < n_chunks; c++) {
            vb->chunks[c]   = chunks[c];
            alloc->fixed[c] = ggml_backend_buffer_get_size(chunks[c]);
        }
        alloc->n_fixed = n_chunks;
        galloc->own_buffers[buffer_id] = galloc->buffers[buffer_id];
        galloc->buffers[buffer_id]     = vb;
        vb->chunks[n_chunks] = galloc->own_buffers[buffer_id] ? galloc->own_buffers[buffer_id]->chunks[0] : NULL;
    }
    galloc->own_no_grow[buffer_id] = !own_may_grow;   // (with pieces or without: the own buffer stays as it is)
    ggml_gallocr_clear_plan_cache(galloc);
    galloc->force_replan = true;
    return true;
}

// utils

static struct ggml_tensor ** ggml_backend_alloc_ctx_tensors_from_buft_collect(
        struct ggml_context * ctx, int * n_tensors) {
    int n = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        n++;
    }
    *n_tensors = n;
    if (n == 0) {
        return NULL;
    }

    struct ggml_tensor ** tensors = (struct ggml_tensor **) malloc(n * sizeof(struct ggml_tensor *));
    if (tensors == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, n * sizeof(struct ggml_tensor *));
        return NULL;
    }
    int i = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        tensors[i++] = t;
    }
    return tensors;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    int n_tensors = 0;
    struct ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    if (tensors == NULL) {
        return NULL;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer_n(buft, tensors, n_tensors);
    free(tensors);
    return buffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    int n_tensors = 0;
    struct ggml_tensor ** tensors = ggml_backend_alloc_ctx_tensors_from_buft_collect(ctx, &n_tensors);
    if (tensors == NULL) {
        return 0;
    }

    size_t nbytes_total = ggml_backend_buft_get_alloc_size_n(buft, tensors, n_tensors);
    free(tensors);
    return nbytes_total;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend) {
    return ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}

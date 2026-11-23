#include "pool1d.cuh"

static __global__ void pool1d_nchw_kernel(
        const int iw, const int ow,
        const int kw, const int sw, const int pw,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= parallel_elements) {
        return;
    }

    const int nc = idx / ow;
    const int cur_ow = idx % ow;

    const float * i_ptr = src + nc * iw;
    float * o_ptr = dst + nc * ow;

    const int start = cur_ow * sw - pw;
    const int b = max(0, start);
    const int e = min(iw, start + kw);

    float res;
    switch (op) {
        case GGML_OP_POOL_AVG: res = 0.0f;     break;
        case GGML_OP_POOL_MAX: res = -FLT_MAX; break;
        default: return;
    }

    int count = 0;
    for (int i = b; i < e; i++) {
#if __CUDA_ARCH__ >= 350
        float cur = __ldg(i_ptr + i);
#else
        float cur = i_ptr[i];
#endif
        switch (op) {
            case GGML_OP_POOL_AVG: res += cur;                break;
            case GGML_OP_POOL_MAX: res = max(res, cur);       break;
            default: break;
        }
        count++;
    }

    if (op == GGML_OP_POOL_AVG) {
        res = (count > 0) ? (res / count) : 0.0f;
    }

    o_ptr[cur_ow] = res;
}

// GGML_CUDA_POOL1D_ROW=1 (default off): a MAX pool over whole rows (k0 == s0 == IW, p0 == 0, OW == 1 - Flint's
// down-input scale step, #30861's amax: one value per [expert, token] row of 512) as one warp per row with a shuffle max, in place of
// one thread walking the whole row. Max is order-free, so the result is the same bits; -FLT_MAX is the start value as above.
static __global__ void pool1d_row_max_kernel(const int iw, const int nr, const float * src, float * dst) {
    constexpr int ws = ggml_cuda_get_physical_warp_size();
    const int row  = blockIdx.x * (blockDim.x / ws) + threadIdx.x / ws;
    const int lane = threadIdx.x % ws;
    if (row >= nr) {
        return;
    }
    const float * i_ptr = src + (int64_t) row * iw;
    float res = -FLT_MAX;
    for (int i = lane; i < iw; i += ws) {
        res = max(res, i_ptr[i]);
    }
    res = warp_reduce_max<ws>(res);
    if (lane == 0) {
        dst[row] = res;
    }
}

static bool ggml_cuda_pool1d_row_on() {
    static const char * e = getenv("GGML_CUDA_POOL1D_ROW");
    return e != nullptr ? atoi(e) != 0 : ggml_cuda_prefill_fast_defaults_on();
}

static void pool1d_nchw_kernel_f32_f32_cuda(
        const int iw, const int ow,
        const int kw, const int sw, const int pw,
        const int parallel_elements,
        const float * src, float * dst, const enum ggml_op_pool op,
        cudaStream_t stream) {
    const int num_blocks = (parallel_elements + CUDA_POOL1D_BLOCK_SIZE - 1) / CUDA_POOL1D_BLOCK_SIZE;
    dim3 block_nums(num_blocks);
    pool1d_nchw_kernel<<<block_nums, CUDA_POOL1D_BLOCK_SIZE, 0, stream>>>(iw, ow, kw, sw, pw, parallel_elements, src, dst, op);
}

void ggml_cuda_op_pool1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];

    const int64_t IW = src0->ne[0];
    const int64_t OW = dst->ne[0];
    const int64_t nr = ggml_nrows(src0);

    const int parallel_elements = (int)(nr * OW);

    if (op == GGML_OP_POOL_MAX && k0 == IW && s0 == IW && p0 == 0 && OW == 1 && ggml_cuda_pool1d_row_on()) {
        const int ws             = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
        const int rows_per_block = 256 / ws;
        pool1d_row_max_kernel<<<(int) ((nr + rows_per_block - 1) / rows_per_block), rows_per_block * ws, 0, stream>>>((int) IW, (int) nr, src0_d, dst_d);
        return;
    }

    pool1d_nchw_kernel_f32_f32_cuda(IW, OW, k0, s0, p0, parallel_elements, src0_d, dst_d, op, stream);
}

#include "moe-amax.cuh"

// One warp per row of x ([n, rows]): m = max |x| (start -FLT_MAX, as POOL_1D's kernels), then e with the same float
// steps in the same order as the separate kernels (scale.cu: s * x + b; unary.cu: logf / ceilf / expf; clamp.cu: fminf(fmaxf()));
// lane 0 writes e, every lane writes its part of x / e (binbcast's op_div: a / b). The build's math flags are the same for every .cu.
static __global__ void k_moe_amax_div(const float * x, float * __restrict__ e_out, float * y,   // x and y are the same memory when the DIV is in place
        const int n, const int nrows,
        const float s1, const float b1, const float s2, const float b2, const float cmin, const float cmax, const float s3, const float b3) {
    constexpr int ws = ggml_cuda_get_physical_warp_size();
    const int row  = blockIdx.x * (blockDim.x / ws) + threadIdx.x / ws;
    const int lane = threadIdx.x % ws;
    if (row >= nrows) {
        return;
    }
    const float * xr = x + (int64_t) row * n;
    float m = -FLT_MAX;
    for (int i = lane; i < n; i += ws) {
        m = max(m, fabsf(xr[i]));
    }
    m = warp_reduce_max<ws>(m);

    float e = s1 * m + b1;
    e = logf(e);
    e = s2 * e + b2;
    e = ceilf(e);
    e = fminf(fmaxf(e, cmin), cmax);
    e = s3 * e + b3;
    e = expf(e);

    if (lane == 0) {
        e_out[row] = e;
    }
    float * yr = y + (int64_t) row * n;
    for (int i = lane; i < n; i += ws) {
        yr[i] = xr[i] / e;
    }
}

bool ggml_cuda_fuse_amax_on() {
    static const char * v = getenv("GGML_CUDA_FUSE_AMAX");
    return v != nullptr ? atoi(v) != 0 : ggml_cuda_prefill_fast_defaults_on();
}

static float op_param_f(const ggml_tensor * t, int k) {
    float v;
    memcpy(&v, (const float *) t->op_params + k, sizeof(float));
    return v;
}

// nodes: the 10 matched nodes, in order (abs, pool, scale, log, scale, ceil, clamp, scale, exp, div)
void ggml_cuda_op_moe_amax_fused(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes) {
    const ggml_tensor * x   = nodes[0]->src[0];
    ggml_tensor       * ex  = nodes[8];
    ggml_tensor       * div = nodes[9];
    const int n     = (int) x->ne[0];
    const int nrows = (int) ggml_nrows(x);
    const int ws    = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rpb   = 256 / ws;
    k_moe_amax_div<<<(nrows + rpb - 1) / rpb, rpb * ws, 0, ctx.stream()>>>((const float *) x->data, (float *) ex->data, (float *) div->data,
        n, nrows,
        op_param_f(nodes[2], 0), op_param_f(nodes[2], 1), op_param_f(nodes[4], 0), op_param_f(nodes[4], 1),
        op_param_f(nodes[6], 0), op_param_f(nodes[6], 1), op_param_f(nodes[7], 0), op_param_f(nodes[7], 1));
}

#include "gated-delta-net-mma.cuh"
#include "moe-fused-prefill.cuh"
#include "mma.cuh"

#include <climits>
#include <mutex>

namespace ggml_cuda_gated_delta_net_mma {
constexpr int D = 128;
using bf16      = nv_bfloat16;

template <int R, int C> struct matrix {
    bf16 hi[R * C], lo[R * C];

    static __device__ __forceinline__ int index(int r, int c) {
#ifdef GGML_USE_HIP
        return r * C + c;
#else
        return 2 * ggml_cuda_mma::swizzle<C / 2, nv_bfloat162>(r * (C / 2) + c / 2, r) + c % 2;
#endif // GGML_USE_HIP
    }

    __device__ __forceinline__ void store2(int r, int c, float2 x) {
#ifdef GGML_USE_MUSA
        store(r, c, x.x);
        store(r, c + 1, x.y);
#else
        const int    i                            = index(r, c);
        const auto   h                            = __float22bfloat162_rn(x);
        const float2 rounded                      = __bfloat1622float2(h);
        *reinterpret_cast<nv_bfloat162 *>(hi + i) = h;
        *reinterpret_cast<nv_bfloat162 *>(lo + i) =
            __float22bfloat162_rn(make_float2(x.x - rounded.x, x.y - rounded.y));
#endif // GGML_USE_MUSA
    }

    __device__ __forceinline__ void store(int r, int c, float x) {
        const int  i = index(r, c);
        const bf16 h = __float2bfloat16(x);
        hi[i]        = h;
        lo[i]        = __float2bfloat16(x - __bfloat162float(h));
    }
};

template <int C, int V> struct alignas(128) shared {
    matrix<C, D> q, k;
    matrix<C, C> p, inverse;
    matrix<C, V> delta;

    union {
        matrix<D, V> state;
        matrix<C, V> solved;
    } scratch;

    float lower[C * C], prefix[C], beta[C];

    static __device__ __forceinline__ int lower_index(int r, int c) { return r * C + c; }
};

#if defined(AMPERE_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA3))
using namespace ggml_cuda_mma;
#ifdef GGML_USE_HIP
using acc    = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;
using a_tile = tile<16, 8, nv_bfloat162, DATA_LAYOUT_I_MAJOR_MIRRORED>;
using b_tile = a_tile;
#else
using acc    = tile<16, 8, float>;
using a_tile = tile<16, 8, nv_bfloat162>;
using b_tile = tile<8, 8, nv_bfloat162>;
#endif // GGML_USE_HIP

template <class tile_t, bool TRANS, int R, int C>
__device__ __forceinline__ tile_t load(const matrix<R, C> & m, int row, int col, bool low) {
    tile_t       t;
    const bf16 * data = low ? m.lo : m.hi;
#ifdef GGML_USE_HIP
#pragma unroll
    for (int i = 0; i < tile_t::ne; ++i) {
        const int  r = row + tile_t::get_i(i), c = col + 2 * tile_t::get_j(i);
        const bf16 x = data[TRANS ? m.index(c, r) : m.index(r, c)];
        const bf16 y = data[TRANS ? m.index(c + 1, r) : m.index(r, c + 1)];
        t.x[i]       = __float22bfloat162_rn(make_float2(__bfloat162float(x), __bfloat162float(y)));
    }
#else
    const auto * packed = reinterpret_cast<const nv_bfloat162 *>(data);
    if constexpr (TRANS) {
        load_ldmatrix_trans_swizzled<C / 2>(t, packed, col * (C / 2) + row / 2);
    } else {
        load_ldmatrix_swizzled<C / 2>(t, packed, row * (C / 2) + col / 2);
    }
#endif // GGML_USE_HIP
    return t;
}

// B is addressed as B-transpose. Preserve both BF16 first-order residual terms.
template <int K, bool TA = false, bool TB = false, int AR, int AC, int BR, int BC>
__device__ __forceinline__ void product(acc & c, const matrix<AR, AC> & a, const matrix<BR, BC> & b, int row, int col) {
#pragma unroll
    for (int k = 0; k < K; k += 16) {
        const auto ah = load<a_tile, TA>(a, row, k, false);
        const auto al = load<a_tile, TA>(a, row, k, true);
        const auto bh = load<b_tile, TB>(b, col, k, false);
        const auto bl = load<b_tile, TB>(b, col, k, true);
        mma(c, al, bh);
        mma(c, ah, bl);
        mma(c, ah, bh);
    }
}

template <int R, int C> __device__ __forceinline__ void store(matrix<R, C> & m, const acc & x, int r, int c) {
#pragma unroll
#ifdef GGML_USE_HIP
    for (int i = 0; i < acc::ne; ++i) {
        m.store(r + acc::get_i(i), c + acc::get_j(i), x.x[i]);
    }
#else
    for (int i = 0; i < acc::ne; i += 2) {
        m.store2(r + acc::get_i(i), c + acc::get_j(i), make_float2(x.x[i], x.x[i + 1]));
    }
#endif // GGML_USE_HIP
}
#endif // defined(AMPERE_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA3))

// Each block owns V independent value columns and advances by C tokens.
// Keep recurrent state in FP32; pack high/residual BF16 operands inside the block.
template <int C, int V, int BLOCKS = 1>
static __global__ __launch_bounds__(256, BLOCKS) void gdn_single(ggml_cuda_gdn_mma_args a) {
#if defined(AMPERE_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA3))
    static_assert(ggml_cuda_get_physical_warp_size() == 32, "GDN MMA requires 32-lane warps");
    constexpr int WARPS = 8;
#ifdef GGML_USE_HIP
    constexpr int N = 16;
#else
    constexpr int N = 8;
#endif // GGML_USE_HIP
    static_assert(C >= 16 && (C & (C - 1)) == 0 && V % N == 0 && D % V == 0, "invalid GDN tile");
    extern __shared__ __align__(128) unsigned char bytes[];
    auto &                                         s    = *reinterpret_cast<shared<C, V> *>(bytes);
    const int                                      lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
    const int                                      slice = blockIdx.x % (D / V);
    const int                                      h     = (blockIdx.x / (D / V)) % a.H;
    const int                                      seq   = blockIdx.x / ((D / V) * a.H);
    const int                                      v0    = slice * V;
    const int64_t                                  soff  = ((int64_t) seq * a.H + h) * D * D;
    const int64_t                                  qoff  = (seq / a.rq3) * a.sq3 + (h % a.H_k) * a.sq1;
    const int64_t                                  voff  = seq * a.sv3 + h * a.sv1 + v0;
    const int64_t                                  goff  = seq * a.sb3 + h * a.sb1;
    float *                                        out   = a.dst + ((int64_t) seq * a.n_tokens * a.H + h) * D + v0;
    constexpr int                                  STATE_TILES = (D / 16) * (V / N);
    constexpr int                                  PER_WARP    = STATE_TILES / WARPS;
    static_assert(STATE_TILES % WARPS == 0, "state must divide across warps");
    acc state[PER_WARP];
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
        const int tile_index = warp + j * WARPS, r = (tile_index / (V / N)) * 16, c = (tile_index % (V / N)) * N;
#pragma unroll
        for (int i = 0; i < acc::ne; ++i) {
            state[j].x[i] = a.state[soff + (v0 + c + acc::get_j(i)) * D + r + acc::get_i(i)];
        }
    }
    auto read_qk = [&](int64_t t, float4 & q, float4 & k) {
        t += warp;
        q = k = make_float4(0, 0, 0, 0);
        if (t < a.n_tokens) {
            const int64_t off = qoff + t * a.sq2 + 4 * lane;
            if ((((uintptr_t) (a.q + off) | (uintptr_t) (a.k + off)) & 15) == 0) {
                q = *reinterpret_cast<const float4 *>(a.q + off);
                k = *reinterpret_cast<const float4 *>(a.k + off);
            } else {
                q = make_float4(a.q[off], a.q[off + 1], a.q[off + 2], a.q[off + 3]);
                k = make_float4(a.k[off], a.k[off + 1], a.k[off + 2], a.k[off + 3]);
            }
        }
    };
    // Carry the first eight input rows across chunks to hide global-load latency.
    constexpr bool PREFETCH = C == 16 && V == 32;
    float4         next_q, next_k;
    if constexpr (PREFETCH) {
        read_qk(0, next_q, next_k);
    }
    for (int64_t t0 = 0; t0 < a.n_tokens; t0 += C) {
        const int valid = min((int64_t) C, a.n_tokens - t0);
#pragma unroll
        for (int j = 0; j < C / 8; ++j) {
            const int t = warp + 8 * j, d = 4 * lane;
            float4    q, k;
            if (PREFETCH && j == 0) {
                q = next_q;
                k = next_k;
            } else {
                read_qk(t0 + 8 * j, q, k);
            }
            s.q.store2(t, d, make_float2(q.x * a.scale, q.y * a.scale));
            s.q.store2(t, d + 2, make_float2(q.z * a.scale, q.w * a.scale));
            s.k.store2(t, d, make_float2(k.x, k.y));
            s.k.store2(t, d + 2, make_float2(k.z, k.w));
        }
        // Load gates in parallel before the FP32 prefix scan.
        if (warp == 0) {
            float carry = 0.f;
            for (int base = 0; base < C; base += 32) {
                const int   t    = base + lane;
                float       g    = t < valid ? a.g[goff + (t0 + t) * a.sb2] : 0.f;
                const float beta = t < valid ? a.beta[goff + (t0 + t) * a.sb2] : 0.f;
#pragma unroll
                for (int offset = 1; offset < 32; offset *= 2) {
                    const float previous = __shfl_up_sync(0xffffffff, g, offset, 32);
                    if (lane >= offset) {
                        g += previous;
                    }
                }
                g += carry;
                if (t < C) {
                    s.prefix[t] = g;
                    s.beta[t]   = beta;
                }
                carry = __shfl_sync(0xffffffff, g, 31, 32);
            }
        }
#pragma unroll
        for (int j = 0; j < PER_WARP; ++j) {
            const int ti = warp + j * WARPS;
            store(s.scratch.state, state[j], (ti / (V / N)) * 16, (ti % (V / N)) * N);
        }
        __syncthreads();
        if constexpr (PREFETCH) {
            if (t0 + C < a.n_tokens) {
                read_qk(t0 + C, next_q, next_k);
            }
        }
        // V32 overlaps Gram work with one output tile per projection warp.
        constexpr bool SPLIT_PROJECTION = C == 16 && V == 32;
        acc            base_output[((C / 16) * (V / N) + WARPS - 1) / WARPS];
        // delta = beta * (v - decay * K S), base output = decay * Q S
        auto           project = [&]() {
            if constexpr (V == 32) {
                if (warp < 4) {
                    return;
                }
            }
            for (int ti = (V == 32 ? warp - 4 : warp); ti < (C / 16) * (V / N); ti += (V == 32 ? 4 : WARPS)) {
                const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
                acc       qs, ks;
                product<D, false, true>(qs, s.q, s.scratch.state, r, c);
                product<D, false, true>(ks, s.k, s.scratch.state, r, c);
#pragma unroll
                for (int i = 0; i < acc::ne; ++i) {
                    const int   t = r + acc::get_i(i), v = c + acc::get_j(i);
                    const float decay = expf(s.prefix[t]);
                    const float vv    = t < valid ? a.v[voff + (t0 + t) * a.sv2 + v] : 0.f;
                    s.delta.store(t, v, s.beta[t] * (vv - decay * ks.x[i]));
                    base_output[ti / WARPS].x[i] = decay * qs.x[i];
                }
            }
        };
        // A = beta * strict_tril(K K^T * decay), P = tril(Q K^T * decay)
        for (int ti = warp; ti < 2 * (C / 16) * (C / N) && (V != 32 || warp < 4); ti += (V == 32 ? 4 : WARPS)) {
            const bool gram = ti < (C / 16) * (C / N);
            const int  t = ti % ((C / 16) * (C / N)), r = (t / (C / N)) * 16, c = (t % (C / N)) * N;
            acc        x;
            if (gram) {
                product<D>(x, s.k, s.k, r, c);
            } else {
                product<D>(x, s.q, s.k, r, c);
            }
#pragma unroll
            for (int i = 0; i < acc::ne; ++i) {
                const int   rr = r + acc::get_i(i), cc = c + acc::get_j(i);
                const float v = rr < valid && cc <= rr ? x.x[i] * expf(s.prefix[rr] - s.prefix[cc]) : 0.f;
                if (gram) {
                    if (cc < rr) {
                        s.lower[s.lower_index(rr, cc)] = v * s.beta[rr];
                    }
                } else {
                    s.p.store(rr, cc, v);
                }
            }
        }
        if constexpr (V == 32) {
            project();
        }
        __syncthreads();
        // inverse = (I + A)^-1 by forward substitution
        for (int row = warp; row < C; row += WARPS) {
            float x[(C + 31) / 32];
#pragma unroll
            for (int j = 0; j < (C + 31) / 32; ++j) {
                const int col = lane + 32 * j;
                x[j]          = col == row ? 1.f : (col < row ? -s.lower[s.lower_index(row, col)] : 0.f);
            }
            for (int pivot = row - 1; pivot > 0; --pivot) {
                const float xp = __shfl_sync(0xffffffff, x[pivot / 32], pivot % 32, 32);
#pragma unroll
                for (int j = 0; j < (C + 31) / 32; ++j) {
                    const int col = lane + 32 * j;
                    if (col < pivot) {
                        x[j] -= xp * s.lower[s.lower_index(pivot, col)];
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < (C + 31) / 32; ++j) {
                if (lane + 32 * j < C) {
                    s.inverse.store(row, lane + 32 * j, x[j]);
                }
            }
        }
        if constexpr (V != 32) {
            project();
        }
        __syncthreads();
        // solved = inverse * delta
        for (int ti = warp; ti < (C / 16) * (V / N); ti += WARPS) {
            const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
            acc       x;
            product<C, false, true>(x, s.inverse, s.delta, r, c);
            store(s.scratch.solved, x, r, c);
        }
        __syncthreads();
        // out = base output + P * solved
        for (int ti = (SPLIT_PROJECTION ? warp - 4 : warp); ti < (C / 16) * (V / N) && (!SPLIT_PROJECTION || warp >= 4);
             ti += WARPS) {
            const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
            acc       x;
            product<C, false, true>(x, s.p, s.scratch.solved, r, c);
#pragma unroll
            for (int i = 0; i < acc::ne; ++i) {
                const int t = r + acc::get_i(i), v = c + acc::get_j(i);
                if (t < valid) {
                    out[(t0 + t) * a.H * D + v] = base_output[ti / WARPS].x[i] + x.x[i];
                }
            }
        }
        // S = decay(chunk) * S + K^T * (solved * decay to chunk end)
        for (int i = tid; i < C * V; i += 256) {
            const int   t = i / V, v = i % V, j = s.scratch.solved.index(t, v);
            const float x = __bfloat162float(s.scratch.solved.hi[j]) + __bfloat162float(s.scratch.solved.lo[j]);
            s.delta.store(t, v, x * expf(s.prefix[valid - 1] - s.prefix[t]));
        }
        __syncthreads();
        const float decay = expf(s.prefix[valid - 1]);
#pragma unroll
        for (int j = 0; j < PER_WARP; ++j) {
            const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
#pragma unroll
            for (int i = 0; i < acc::ne; ++i) {
                state[j].x[i] *= decay;
            }
            product<C, true, true>(state[j], s.k, s.delta, r, c);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
        const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
#pragma unroll
        for (int i = 0; i < acc::ne; ++i) {
            a.state_out[soff + (v0 + c + acc::get_j(i)) * D + r + acc::get_i(i)] = state[j].x[i];
        }
    }
#else
    GGML_UNUSED_VARS(a);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA3))
}

template <int C, int V> struct alignas(128) shared_kda {
    matrix<C, D> q_dec, k_dec;
    matrix<C, C> p, inverse;
    matrix<C, V> delta;

    union {
        matrix<D, V> state;
        matrix<C, V> solved;
        struct {
            matrix<C, D> q_pos;
            matrix<C, D> k_neg;
        } gp;
    } scratch;

    float g_pref[C * D];
    float lower[C * C];
    float beta[C];
    int   wide_gate;

    static __device__ __forceinline__ int lower_index(int r, int c) { return r * C + c; }
};

template <int C, int V, int BLOCKS = 1>
static __global__ __launch_bounds__(256, BLOCKS) void gdn_kda_single(ggml_cuda_gdn_mma_args a) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(AMPERE_MMA_AVAILABLE)
    static_assert(ggml_cuda_get_physical_warp_size() == 32, "GDN KDA MMA requires 32-lane warps");
    static_assert(C == 16 && (V == 32 || V == 128), "GDN KDA MMA supports C=16, V in {32, 128}");
    constexpr int WARPS = 8;
    constexpr int N     = 8;
    extern __shared__ __align__(128) unsigned char kda_bytes[];
    auto &        s     = *reinterpret_cast<shared_kda<C, V> *>(kda_bytes);
    const int     lane  = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
    const int     slice = blockIdx.x % (D / V);
    const int     h     = (blockIdx.x / (D / V)) % a.H;
    const int     seq   = blockIdx.x / ((D / V) * a.H);
    const int     v0    = slice * V;
    const int64_t soff  = ((int64_t) seq * a.H + h) * D * D;
    const int64_t qoff  = (seq / a.rq3) * a.sq3 + (h % a.H_k) * a.sq1;
    const int64_t voff  = seq * a.sv3 + h * a.sv1 + v0;
    const int64_t goff  = seq * a.sb3 + h * a.sb1;
    float *       out   = a.dst + ((int64_t) seq * a.n_tokens * a.H + h) * D + v0;

    constexpr int STATE_TILES = (D / 16) * (V / N);
    constexpr int PER_WARP    = STATE_TILES / WARPS;
    acc state[PER_WARP];
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
        const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
#pragma unroll
        for (int i = 0; i < acc::ne; ++i) {
            state[j].x[i] = a.state[soff + (v0 + c + acc::get_j(i)) * D + r + acc::get_i(i)];
        }
    }

    auto read_f4 = [](const float * ptr) -> float4 {
        if ((((uintptr_t) ptr) & 15) == 0) {
            return *reinterpret_cast<const float4 *>(ptr);
        }
        return make_float4(ptr[0], ptr[1], ptr[2], ptr[3]);
    };

    for (int64_t t0 = 0; t0 < a.n_tokens; t0 += C) {
        const int valid = (int) min((int64_t) C, a.n_tokens - t0);
        float4 q_reg[2], k_reg[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int t = warp + 8 * j, d = 4 * lane;
            q_reg[j] = k_reg[j] = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 gv = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < valid) {
                const int64_t qo = qoff + (t0 + t) * a.sq2 + d;
                const int64_t go = (goff + (t0 + t) * a.sb2) * D + d;
                q_reg[j] = read_f4(a.q + qo);
                k_reg[j] = read_f4(a.k + qo);
                gv       = read_f4(a.g + go);
            }
            s.g_pref[t * D + d + 0] = gv.x;
            s.g_pref[t * D + d + 1] = gv.y;
            s.g_pref[t * D + d + 2] = gv.z;
            s.g_pref[t * D + d + 3] = gv.w;
        }
        if (warp == 0 && lane < C) {
            s.beta[lane] = lane < valid ? a.beta[goff + (t0 + lane) * a.sb2] : 0.f;
            if (lane == 0) {
                s.wide_gate = 0;
            }
        }
        __syncthreads();

        if (tid < D) {
            const int d = tid;
            float sum = 0.f;
            const float g0 = s.g_pref[d];
            bool bad = false;
#pragma unroll
            for (int t = 0; t < C; ++t) {
                const float gt = s.g_pref[t * D + d];
                if (gt > 1e-4f) {
                    bad = true;
                }
                sum += gt;
                s.g_pref[t * D + d] = sum;
            }
            const float gmid  = s.g_pref[((valid - 1) / 2) * D + d];
            const float glast = s.g_pref[(valid - 1) * D + d];
            if (bad || fabsf(g0 - gmid) > 60.0f || fabsf(glast - gmid) > 60.0f || sum > 24.0f) {
                atomicOr(&s.wide_gate, 1);
            }
        }
        __syncthreads();
        if (!s.wide_gate) {
            const int mid = (valid - 1) / 2;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int t = warp + 8 * j, d = 4 * lane;
                const float m0 = s.g_pref[mid * D + d + 0], m1 = s.g_pref[mid * D + d + 1];
                const float m2 = s.g_pref[mid * D + d + 2], m3 = s.g_pref[mid * D + d + 3];
                const float g0 = s.g_pref[t * D + d + 0],   g1 = s.g_pref[t * D + d + 1];
                const float g2 = s.g_pref[t * D + d + 2],   g3 = s.g_pref[t * D + d + 3];
                const float ep0 = t < valid ? expf(g0 - m0) : 0.f, ep1 = t < valid ? expf(g1 - m1) : 0.f;
                const float ep2 = t < valid ? expf(g2 - m2) : 0.f, ep3 = t < valid ? expf(g3 - m3) : 0.f;
                const float en0 = t < valid ? expf(m0 - g0) : 0.f, en1 = t < valid ? expf(m1 - g1) : 0.f;
                const float en2 = t < valid ? expf(m2 - g2) : 0.f, en3 = t < valid ? expf(m3 - g3) : 0.f;
                s.scratch.gp.q_pos.store2(t, d,     make_float2(q_reg[j].x * a.scale * ep0, q_reg[j].y * a.scale * ep1));
                s.scratch.gp.q_pos.store2(t, d + 2, make_float2(q_reg[j].z * a.scale * ep2, q_reg[j].w * a.scale * ep3));
                s.k_dec.store2(t, d,     make_float2(k_reg[j].x * ep0, k_reg[j].y * ep1));
                s.k_dec.store2(t, d + 2, make_float2(k_reg[j].z * ep2, k_reg[j].w * ep3));
                s.scratch.gp.k_neg.store2(t, d,     make_float2(k_reg[j].x * en0, k_reg[j].y * en1));
                s.scratch.gp.k_neg.store2(t, d + 2, make_float2(k_reg[j].z * en2, k_reg[j].w * en3));
            }
            __syncthreads();
            if (warp < 4) {
                const bool gram = warp < 2;
                const int  c    = (warp % 2) * N;
                acc        x;
                if (gram) {
                    product<D>(x, s.k_dec, s.scratch.gp.k_neg, 0, c);
                } else {
                    product<D>(x, s.scratch.gp.q_pos, s.scratch.gp.k_neg, 0, c);
                }
#pragma unroll
                for (int i = 0; i < acc::ne; ++i) {
                    const int   rr = acc::get_i(i), cc = c + acc::get_j(i);
                    const float val = (rr < valid && cc <= rr) ? x.x[i] : 0.f;
                    if (gram) {
                        if (cc < rr) {
                            s.lower[s.lower_index(rr, cc)] = val * s.beta[rr];
                        }
                    } else {
                        s.p.store(rr, cc, val);
                    }
                }
            }
            __syncthreads();
        } else {
            for (int idx = warp; idx < C * C; idx += WARPS) {
                const int rr = idx / C, cc = idx % C;
                float dq = 0.f, dk = 0.f;
                if (rr < valid && cc <= rr) {
                    const int64_t orr = qoff + (t0 + rr) * a.sq2 + 4 * lane;
                    const int64_t occ = qoff + (t0 + cc) * a.sq2 + 4 * lane;
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        const int   d   = 4 * lane + m;
                        const float dec = expf(s.g_pref[rr * D + d] - s.g_pref[cc * D + d]);
                        const float kc  = a.k[occ + m] * dec;
                        dq += (a.q[orr + m] * a.scale) * kc;
                        if (cc < rr) {
                            dk += a.k[orr + m] * kc;
                        }
                    }
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        dq += __shfl_xor_sync(0xffffffff, dq, off, 32);
                        dk += __shfl_xor_sync(0xffffffff, dk, off, 32);
                    }
                }
                if (lane == 0) {
                    s.p.store(rr, cc, (rr < valid && cc <= rr) ? dq : 0.f);
                    if (rr < valid && cc < rr) {
                        s.lower[s.lower_index(rr, cc)] = dk * s.beta[rr];
                    }
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int t = warp + 8 * j, d = 4 * lane;
            const float e0 = t < valid ? expf(s.g_pref[t * D + d + 0]) : 0.f;
            const float e1 = t < valid ? expf(s.g_pref[t * D + d + 1]) : 0.f;
            const float e2 = t < valid ? expf(s.g_pref[t * D + d + 2]) : 0.f;
            const float e3 = t < valid ? expf(s.g_pref[t * D + d + 3]) : 0.f;
            s.q_dec.store2(t, d,     make_float2(q_reg[j].x * a.scale * e0, q_reg[j].y * a.scale * e1));
            s.q_dec.store2(t, d + 2, make_float2(q_reg[j].z * a.scale * e2, q_reg[j].w * a.scale * e3));
            s.k_dec.store2(t, d,     make_float2(k_reg[j].x * e0, k_reg[j].y * e1));
            s.k_dec.store2(t, d + 2, make_float2(k_reg[j].z * e2, k_reg[j].w * e3));
        }
#pragma unroll
        for (int j = 0; j < PER_WARP; ++j) {
            const int ti = warp + j * WARPS;
            store(s.scratch.state, state[j], (ti / (V / N)) * 16, (ti % (V / N)) * N);
        }
        __syncthreads();
        constexpr int PROJ_TILES = V / N;
        acc base_output[(PROJ_TILES + WARPS - 1) / WARPS];
        if constexpr (V == 32) {
            if (warp < 4) {
                const int c = warp * N;
                acc qs, ks;
                product<D, false, true>(qs, s.q_dec, s.scratch.state, 0, c);
                product<D, false, true>(ks, s.k_dec, s.scratch.state, 0, c);
#pragma unroll
                for (int i = 0; i < acc::ne; ++i) {
                    const int   t  = acc::get_i(i), v = c + acc::get_j(i);
                    const float vv = t < valid ? a.v[voff + (t0 + t) * a.sv2 + v] : 0.f;
                    s.delta.store(t, v, s.beta[t] * (vv - ks.x[i]));
                    base_output[0].x[i] = qs.x[i];
                }
            } else {
                for (int row = warp - 4; row < C; row += 4) {
                    float x = (lane == row) ? 1.f : (lane < row ? -s.lower[s.lower_index(row, lane)] : 0.f);
                    for (int pivot = row - 1; pivot > 0; --pivot) {
                        const float xp = __shfl_sync(0xffffffff, x, pivot, 32);
                        if (lane < pivot) {
                            x -= xp * s.lower[s.lower_index(pivot, lane)];
                        }
                    }
                    if (lane < C) {
                        s.inverse.store(row, lane, x);
                    }
                }
            }
        } else {
            for (int b = 0, ti = warp; ti < PROJ_TILES; ti += WARPS, ++b) {
                const int c = ti * N;
                acc qs, ks;
                product<D, false, true>(qs, s.q_dec, s.scratch.state, 0, c);
                product<D, false, true>(ks, s.k_dec, s.scratch.state, 0, c);
#pragma unroll
                for (int i = 0; i < acc::ne; ++i) {
                    const int   t  = acc::get_i(i), v = c + acc::get_j(i);
                    const float vv = t < valid ? a.v[voff + (t0 + t) * a.sv2 + v] : 0.f;
                    s.delta.store(t, v, s.beta[t] * (vv - ks.x[i]));
                    base_output[b].x[i] = qs.x[i];
                }
            }
            for (int row = warp; row < C; row += WARPS) {
                float x = (lane == row) ? 1.f : (lane < row ? -s.lower[s.lower_index(row, lane)] : 0.f);
                for (int pivot = row - 1; pivot > 0; --pivot) {
                    const float xp = __shfl_sync(0xffffffff, x, pivot, 32);
                    if (lane < pivot) {
                        x -= xp * s.lower[s.lower_index(pivot, lane)];
                    }
                }
                if (lane < C) {
                    s.inverse.store(row, lane, x);
                }
            }
        }
        __syncthreads();

        for (int ti = warp; ti < PROJ_TILES; ti += WARPS) {
            const int c = ti * N;
            acc x;
            product<C, false, true>(x, s.inverse, s.delta, 0, c);
            store(s.scratch.solved, x, 0, c);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int t = warp + 8 * j, d = 4 * lane;
            const int last = valid - 1;
            const float e0 = t < valid ? expf(s.g_pref[last * D + d + 0] - s.g_pref[t * D + d + 0]) : 0.f;
            const float e1 = t < valid ? expf(s.g_pref[last * D + d + 1] - s.g_pref[t * D + d + 1]) : 0.f;
            const float e2 = t < valid ? expf(s.g_pref[last * D + d + 2] - s.g_pref[t * D + d + 2]) : 0.f;
            const float e3 = t < valid ? expf(s.g_pref[last * D + d + 3] - s.g_pref[t * D + d + 3]) : 0.f;
            s.k_dec.store2(t, d,     make_float2(k_reg[j].x * e0, k_reg[j].y * e1));
            s.k_dec.store2(t, d + 2, make_float2(k_reg[j].z * e2, k_reg[j].w * e3));
        }
        __syncthreads();

        for (int b = 0, ti = warp; ti < PROJ_TILES; ti += WARPS, ++b) {
            const int c = ti * N;
            acc x;
            product<C, false, true>(x, s.p, s.scratch.solved, 0, c);
#pragma unroll
            for (int i = 0; i < acc::ne; ++i) {
                const int t = acc::get_i(i), v = c + acc::get_j(i);
                if (t < valid) {
                    out[(t0 + t) * a.H * D + v] = base_output[b].x[i] + x.x[i];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < PER_WARP; ++j) {
            const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
#pragma unroll
            for (int i = 0; i < acc::ne; ++i) {
                state[j].x[i] *= expf(s.g_pref[(valid - 1) * D + r + acc::get_i(i)]);
            }
            product<C, true, true>(state[j], s.k_dec, s.scratch.solved, r, c);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
        const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
#pragma unroll
        for (int i = 0; i < acc::ne; ++i) {
            a.state_out[soff + (v0 + c + acc::get_j(i)) * D + r + acc::get_i(i)] = state[j].x[i];
        }
    }
#else
    GGML_UNUSED_VARS(a);
    NO_DEVICE_CODE;
#endif
}

template <int C, int V, int BLOCKS = 1> bool init_kda(int device) {
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool           available[GGML_CUDA_MAX_DEVICES] = {};
    std::call_once(once[device], [device] {
        ggml_cuda_set_device(device);
        if (sizeof(shared_kda<C, V>) > ggml_cuda_info().devices[device].smpbo) {
            return;
        }
        const auto status = cudaFuncSetAttribute((const void *) gdn_kda_single<C, V, BLOCKS>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(shared_kda<C, V>));
        available[device] = status == cudaSuccess;
        if (status != cudaSuccess) {
            (void) cudaGetLastError();
        }
    });
    return available[device];
}

template <int C, int V, int BLOCKS = 1> bool init(int device) {
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool           available[GGML_CUDA_MAX_DEVICES] = {};
    std::call_once(once[device], [device] {
        ggml_cuda_set_device(device);
        if (sizeof(shared<C, V>) > ggml_cuda_info().devices[device].smpbo) {
            return;
        }
        const auto status = cudaFuncSetAttribute((const void *) gdn_single<C, V, BLOCKS>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(shared<C, V>));
        available[device] = status == cudaSuccess;
        if (status != cudaSuccess) {
            (void) cudaGetLastError();
        }
    });
    return available[device];
}

enum class gdn_path { ar, slice32_one_block, slice32_two_blocks, slice64, whole_head };

static gdn_path select_gdn_mma_path(int device, const ggml_cuda_gdn_mma_args & a) {
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    const bool opt = ggml_cuda_mmq_moe_opt_enabled();
    const bool ok_heads = opt
        ? (a.H_k > 0 && a.H >= a.H_k && (a.H % a.H_k) == 0)
        : (!a.kda && a.H_k == 16 && (a.H == 16 || a.H == 32 || a.H == 48 || a.H == 64));
    if (!a.eligible || !ok_heads || a.n_tokens < 64 ||
        a.n_tokens > INT64_MAX - 32 || a.rq3 <= 0 || a.n_seqs <= 0 || a.n_seqs > INT_MAX / (a.H * 4)) {
        return gdn_path::ar;
    }
    const auto & info = ggml_cuda_info().devices[device];
#ifdef GGML_USE_HIP
    if (a.kda || a.H_k != 16 || (a.H != 16 && a.H != 32 && a.H != 48 && a.H != 64)) {
        return gdn_path::ar;
    }
    const bool extra_arch = opt && GGML_CUDA_CC_IS_RDNA4(info.cc);
    if ((!GGML_CUDA_CC_IS_RDNA3_5(info.cc) && !extra_arch) || a.n_tokens < 2048) {
        return gdn_path::ar;
    }
    return init<16, 64>(device) ? gdn_path::slice64 : gdn_path::ar;
#elif defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(info);
    return gdn_path::ar;
#else
    // Enable only architectures benchmarked for this schedule. MMA support alone does not imply a speedup.
    const bool base_arch  = info.cc == GGML_CUDA_CC_AMPERE_GA10X || info.cc == GGML_CUDA_CC_ADA_LOVELACE ||
                            info.cc == GGML_CUDA_CC_BLACKWELL    || info.cc == GGML_CUDA_CC_DGX_SPARK;
    const bool extra_arch = opt &&
                            (info.cc == GGML_CUDA_CC_AMPERE || info.cc == GGML_CUDA_CC_HOPPER || info.cc == 1000);
    if (!ampere_mma_available(info.cc) || (!base_arch && !extra_arch)) {
        return gdn_path::ar;
    }
    if (a.kda) {
        if (a.H * a.n_seqs * 2 <= info.nsm) {
            if (a.H * a.n_seqs * 4 <= info.nsm) {
                return init_kda<16, 32, 1>(device) ? gdn_path::slice32_one_block : gdn_path::ar;
            }
            return init_kda<16, 32, 2>(device) ? gdn_path::slice32_two_blocks : gdn_path::ar;
        }
        return init_kda<16, 128, 1>(device) ? gdn_path::whole_head : gdn_path::ar;
    }
    // Value splitting pays off when whole heads leave at least half the SMs idle.
    if (a.H * a.n_seqs * 2 <= info.nsm) {
        // A grid that already fits in one wave benefits from a larger register budget.
        if (a.H * a.n_seqs * 4 <= info.nsm) {
            return init<16, 32, 1>(device) ? gdn_path::slice32_one_block : gdn_path::ar;
        }
        return init<16, 32, 2>(device) ? gdn_path::slice32_two_blocks : gdn_path::ar;
    }
    return init<16, 128>(device) ? gdn_path::whole_head : gdn_path::ar;
#endif // GGML_USE_HIP
}
}  // namespace
using namespace ggml_cuda_gated_delta_net_mma;

bool ggml_cuda_gdn_mma_launch(int device, const ggml_cuda_gdn_mma_args & a, cudaStream_t stream) {
    if (!a.state_out) {
        return false;
    }
    const auto path = select_gdn_mma_path(device, a);
    if (path == gdn_path::ar) {
        return false;
    }
#ifdef GGML_USE_HIP
    gdn_single<16, 64><<<a.H * a.n_seqs * 2, dim3(32, 8), sizeof(shared<16, 64>), stream>>>(a);
#else
    if (a.kda) {
        if (path == gdn_path::slice32_two_blocks) {
            gdn_kda_single<16, 32, 2><<<a.H * a.n_seqs * 4, dim3(32, 8), sizeof(shared_kda<16, 32>), stream>>>(a);
        } else if (path == gdn_path::slice32_one_block) {
            gdn_kda_single<16, 32, 1><<<a.H * a.n_seqs * 4, dim3(32, 8), sizeof(shared_kda<16, 32>), stream>>>(a);
        } else {
            gdn_kda_single<16, 128, 1><<<a.H * a.n_seqs, dim3(32, 8), sizeof(shared_kda<16, 128>), stream>>>(a);
        }
    } else if (path == gdn_path::slice32_two_blocks) {
        gdn_single<16, 32, 2><<<a.H * a.n_seqs * 4, dim3(32, 8), sizeof(shared<16, 32>), stream>>>(a);
    } else if (path == gdn_path::slice32_one_block) {
        gdn_single<16, 32, 1><<<a.H * a.n_seqs * 4, dim3(32, 8), sizeof(shared<16, 32>), stream>>>(a);
    } else {
        gdn_single<16, 128><<<a.H * a.n_seqs, dim3(32, 8), sizeof(shared<16, 128>), stream>>>(a);
    }
#endif // GGML_USE_HIP
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#pragma once
#include "wv_inkling_moe_dma.cuh"

namespace wv {

static constexpr int INK2_MTP_BLOCK = 8;
static constexpr int INK2_MTP_INTER = 9216; // 72 * 128
static constexpr int INK2_MTP_RANK  = 256;  // 2 * 128

// Mixes taps [M, 4, 2304] with fuse_alpha[row, 0..3] + RMSNorm(dflare_norm) -> BF16 + optional FWHT128+Q8_32:
// Grid <<<M, dim3(32, 6)>>>
__global__ void wv_ink_mtp_mix_dflare_norm_kernel(
    const __nv_bfloat16* __restrict__ taps_bf16,  // [M, 4, 2304]
    const float* __restrict__ fuse_alpha,         // [4, 4]
    int row,
    const float* __restrict__ dflare_norm,        // [2304]
    __nv_bfloat16* __restrict__ out_bf16,         // [M, 2304] or nullptr
    int8_t* __restrict__ Y_q8,                    // [M, 2304] or nullptr
    float* __restrict__ Y_sc,                     // [M, 72] or nullptr
    int M) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m >= M) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x, wid = threadIdx.y;
    float a0 = fuse_alpha[row * 4 + 0], a1 = fuse_alpha[row * 4 + 1];
    float a2 = fuse_alpha[row * 4 + 2], a3 = fuse_alpha[row * 4 + 3];

    const __nv_bfloat16* tm = taps_bf16 + static_cast<size_t>(m) * 4 * INK2_HIDDEN;
    __shared__ float s_wsum[6];
    float mix[3][4];
    float sq = 0.0f;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int base = (wid + it * 6) * 128 + lane * 4;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int d = base + r;
            float v = a0 * __bfloat162float(tm[0 * INK2_HIDDEN + d])
                    + a1 * __bfloat162float(tm[1 * INK2_HIDDEN + d])
                    + a2 * __bfloat162float(tm[2 * INK2_HIDDEN + d])
                    + a3 * __bfloat162float(tm[3 * INK2_HIDDEN + d]);
            mix[it][r] = v;
            sq += v * v;
        }
    }
    sq = wv_warp_sum_f32(sq);
    if (lane == 0) s_wsum[wid] = sq;
    __syncthreads();

    float inv = rsqrtf((s_wsum[0]+s_wsum[1]+s_wsum[2]+s_wsum[3]+s_wsum[4]+s_wsum[5]) * (1.0f / 2304.0f) + 1e-6f);
    __nv_bfloat16* yb = (out_bf16 != nullptr) ? (out_bf16 + static_cast<size_t>(m) * INK2_HIDDEN) : nullptr;
    int8_t* q8_m      = (Y_q8 != nullptr)     ? (Y_q8 + static_cast<size_t>(m) * INK2_HIDDEN)     : nullptr;
    float*  sc_m      = (Y_sc != nullptr)     ? (Y_sc + static_cast<size_t>(m) * 72)              : nullptr;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int b128 = wid + it * 6;
        int base = b128 * 128 + lane * 4;
        float4 nw = reinterpret_cast<const float4*>(dflare_norm + base)[0];
        __nv_bfloat16 b0 = __float2bfloat16(mix[it][0] * inv * nw.x);
        __nv_bfloat16 b1 = __float2bfloat16(mix[it][1] * inv * nw.y);
        __nv_bfloat16 b2 = __float2bfloat16(mix[it][2] * inv * nw.z);
        __nv_bfloat16 b3 = __float2bfloat16(mix[it][3] * inv * nw.w);
        if (yb != nullptr) { yb[base + 0] = b0; yb[base + 1] = b1; yb[base + 2] = b2; yb[base + 3] = b3; }
        if (q8_m != nullptr && sc_m != nullptr) {
            float f0 = __bfloat162float(b0), f1 = __bfloat162float(b1);
            float f2 = __bfloat162float(b2), f3 = __bfloat162float(b3);
            wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
            wv_quant_q8_32_warp(f0, f1, f2, f3, lane, q8_m + b128 * 128, sc_m + b128 * 4);
        }
    }
    wv_pdl_trigger();
}

__device__ __forceinline__ void wv_ink_apply_neox_rope_pair(float& x_lo, float& x_hi, int d_lo, int pos) {
    float inv_f = powf(1000000.0f, -static_cast<float>(2 * d_lo) * (1.0f / 128.0f));
    float ang   = static_cast<float>(pos) * inv_f;
    float cs    = cosf(ang), sn = sinf(ang);
    float r_lo  = x_lo * cs - x_hi * sn;
    float r_hi  = x_hi * cs + x_lo * sn;
    x_lo = r_lo;
    x_hi = r_hi;
}

// Commits MTP context K (k_norm_ctx [128] + neox RoPE 1e6) & V into MTP KV cache:
// Grid <<<M, dim3(32, 8)>>>
__global__ void wv_ink_mtp_commit_ctx_kv_kernel(
    const __nv_bfloat16* __restrict__ k_raw,      // [B, S, 1024]
    const __nv_bfloat16* __restrict__ v_raw,      // [B, S, 1024]
    const float* __restrict__ k_norm_ctx,         // [128]
    void* __restrict__ k_cache,                   // [B_max, kv_cap, 8, 128] bf16 or int8
    void* __restrict__ v_cache,                   // [B_max, kv_cap, 8, 128] bf16 or int8
    float* __restrict__ k_cache_sc,               // [B_max, kv_cap, 8] or nullptr
    float* __restrict__ v_cache_sc,               // [B_max, kv_cap, 8] or nullptr
    const int32_t* __restrict__ d_mtp_pos0,       // [1]
    const int32_t* __restrict__ d_n_acc,          // [1] or nullptr (commits all S)
    int S, int kv_cap, int is_local, int kv_is_q8, int M) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m >= M) { wv_pdl_trigger(); return; }
    int b = m / S, s = m - b * S;
    if (d_n_acc != nullptr && s >= *d_n_acc) { wv_pdl_trigger(); return; }
    if (is_local && d_n_acc == nullptr && s < S - INK2_SWA_WIN) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x; // 0..31
    int hkv  = threadIdx.y; // 0..7
    int pos  = (*d_mtp_pos0) + s;
    int slot = is_local ? (pos % kv_cap) : pos;
    if (slot < 0 || slot >= kv_cap) { wv_pdl_trigger(); return; }

    const __nv_bfloat16* kh = k_raw + static_cast<size_t>(m) * 1024 + hkv * 128;
    const __nv_bfloat16* vh = v_raw + static_cast<size_t>(m) * 1024 + hkv * 128;
    float k0 = __bfloat162float(kh[lane]),      k1 = __bfloat162float(kh[lane + 32]);
    float k2 = __bfloat162float(kh[lane + 64]), k3 = __bfloat162float(kh[lane + 96]);
    float inv = rsqrtf(wv_warp_sum_f32(k0*k0 + k1*k1 + k2*k2 + k3*k3) * (1.0f / 128.0f) + 1e-6f);
    k0 = __bfloat162float(__float2bfloat16(k0 * inv * k_norm_ctx[lane]));
    k1 = __bfloat162float(__float2bfloat16(k1 * inv * k_norm_ctx[lane + 32]));
    k2 = __bfloat162float(__float2bfloat16(k2 * inv * k_norm_ctx[lane + 64]));
    k3 = __bfloat162float(__float2bfloat16(k3 * inv * k_norm_ctx[lane + 96]));
    wv_ink_apply_neox_rope_pair(k0, k2, lane, pos);
    wv_ink_apply_neox_rope_pair(k1, k3, lane + 32, pos);

    if (!is_local && kv_is_q8 != 0) {
        float v0 = __bfloat162float(vh[lane]),      v1 = __bfloat162float(vh[lane + 32]);
        float v2 = __bfloat162float(vh[lane + 64]), v3 = __bfloat162float(vh[lane + 96]);
        auto q_repack = [&](float x0, float x1, float x2, float x3, float& out_sc) -> uint32_t {
            float amax = wv_warp_max_f32(fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3))));
            out_sc = amax * (1.0f / 127.0f);
            float inv_a = (amax > 1e-30f) ? (127.0f / amax) : 0.0f;
            auto q1 = [&](float val) -> uint32_t {
                int qi = __float2int_rn(val * inv_a);
                qi = max(-127, min(127, qi));
                return static_cast<uint32_t>(static_cast<int8_t>(qi)) & 0xFFu;
            };
            uint32_t p = q1(x0) | (q1(x1) << 8) | (q1(x2) << 16) | (q1(x3) << 24);
            int grp_sh = (lane >> 3) << 3;
            uint32_t out = 0u;
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                uint32_t w = __shfl_sync(0xffffffffu, p, ((lane << 2) + r) & 31);
                out |= ((w >> grp_sh) & 0xFFu) << (8 * r);
            }
            return out;
        };
        float k_sc = 0.0f, v_sc = 0.0f;
        uint32_t k_pk = q_repack(k0, k1, k2, k3, k_sc);
        uint32_t v_pk = q_repack(v0, v1, v2, v3, v_sc);
        size_t tok_h = (static_cast<size_t>(b) * kv_cap + static_cast<size_t>(slot)) * 8 + hkv;
        static_cast<uint32_t*>(k_cache)[tok_h * 32 + lane] = k_pk;
        static_cast<uint32_t*>(v_cache)[tok_h * 32 + lane] = v_pk;
        if (lane == 0) {
            k_cache_sc[tok_h] = k_sc;
            v_cache_sc[tok_h] = v_sc;
        }
    } else {
        __nv_bfloat16* k_bf = static_cast<__nv_bfloat16*>(k_cache);
        __nv_bfloat16* v_bf = static_cast<__nv_bfloat16*>(v_cache);
        size_t dst_off = (static_cast<size_t>(b) * kv_cap + static_cast<size_t>(slot)) * 1024 + hkv * 128;
        k_bf[dst_off + lane]      = __float2bfloat16(k0); k_bf[dst_off + lane + 32] = __float2bfloat16(k1);
        k_bf[dst_off + lane + 64] = __float2bfloat16(k2); k_bf[dst_off + lane + 96] = __float2bfloat16(k3);
        v_bf[dst_off + lane]      = vh[lane];             v_bf[dst_off + lane + 32] = vh[lane + 32];
        v_bf[dst_off + lane + 64] = vh[lane + 64];        v_bf[dst_off + lane + 96] = vh[lane + 96];
    }
    wv_pdl_trigger();
}

// CUDA-Graph-safe update of mtp_f_rnn [2304] from taps[(*d_n_acc) - 1]: Grid <<<1, dim3(32, 6)>>>
__global__ void wv_ink_mtp_mix_dflare_norm_last_acc_kernel(
    const __nv_bfloat16* __restrict__ taps_bf16,  // [8, 4, 2304]
    const int32_t* __restrict__ d_n_acc,          // [1] (1..8)
    const float* __restrict__ fuse_alpha,         // [4, 4] (row = 3)
    const float* __restrict__ dflare_norm,        // [2304]
    __nv_bfloat16* __restrict__ out_f_rnn) {      // [2304]
    wv_pdl_wait();
    int last_s = (*d_n_acc) - 1;
    if (last_s < 0) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x, wid = threadIdx.y;
    float a0 = fuse_alpha[12], a1 = fuse_alpha[13], a2 = fuse_alpha[14], a3 = fuse_alpha[15];
    const __nv_bfloat16* tm = taps_bf16 + static_cast<size_t>(last_s) * 4 * INK2_HIDDEN;
    __shared__ float s_wsum[6];
    float mix[3][4];
    float sq = 0.0f;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int base = (wid + it * 6) * 128 + lane * 4;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int d = base + r;
            float v = a0 * __bfloat162float(tm[0 * INK2_HIDDEN + d])
                    + a1 * __bfloat162float(tm[1 * INK2_HIDDEN + d])
                    + a2 * __bfloat162float(tm[2 * INK2_HIDDEN + d])
                    + a3 * __bfloat162float(tm[3 * INK2_HIDDEN + d]);
            mix[it][r] = v;
            sq += v * v;
        }
    }
    sq = wv_warp_sum_f32(sq);
    if (lane == 0) s_wsum[wid] = sq;
    __syncthreads();
    float inv = rsqrtf((s_wsum[0]+s_wsum[1]+s_wsum[2]+s_wsum[3]+s_wsum[4]+s_wsum[5]) * (1.0f / 2304.0f) + 1e-6f);
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int base = (wid + it * 6) * 128 + lane * 4;
        float4 nw = reinterpret_cast<const float4*>(dflare_norm + base)[0];
        out_f_rnn[base + 0] = __float2bfloat16(mix[it][0] * inv * nw.x);
        out_f_rnn[base + 1] = __float2bfloat16(mix[it][1] * inv * nw.y);
        out_f_rnn[base + 2] = __float2bfloat16(mix[it][2] * inv * nw.z);
        out_f_rnn[base + 3] = __float2bfloat16(mix[it][3] * inv * nw.w);
    }
    wv_pdl_trigger();
}

// Initializes 8-token MTP draft block (s==0: emb_anchor [B, 2304], s==1..7: mask_embed [2304]):
// Grid <<<(B * 8 * 2304 + 255) / 256, 1>>>, Block 256
__global__ void wv_ink_mtp_init_block8_kernel(
    const __nv_bfloat16* __restrict__ emb_anchor, // [B, 2304]
    const __nv_bfloat16* __restrict__ mask_embed, // [2304]
    __nv_bfloat16* __restrict__ x_blk,            // [B, 8, 2304]
    int B) {
    wv_pdl_wait();
    int idx = blockIdx.x * 256 + threadIdx.x;
    int tot = B * 8 * INK2_HIDDEN;
    if (idx >= tot) { wv_pdl_trigger(); return; }
    int d  = idx % INK2_HIDDEN;
    int bs = idx / INK2_HIDDEN;
    int b  = bs >> 3, s = bs & 7;
    __nv_bfloat16 v = (s == 0) ? emb_anchor[static_cast<size_t>(b) * INK2_HIDDEN + d] : mask_embed[d];
    x_blk[idx] = v;
    wv_pdl_trigger();
}

// Residual update s_f32 = (first ? x : x + res); res_bf16 = bf16(s_f32):
__global__ void wv_ink_mtp_res_accum_kernel(
    const __nv_bfloat16* __restrict__ x_bf16,
    __nv_bfloat16* __restrict__ res_bf16,
    float* __restrict__ s_out_f32,
    int is_first,
    int total_elems) {
    wv_pdl_wait();
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= total_elems) { wv_pdl_trigger(); return; }
    float xv = __bfloat162float(x_bf16[idx]);
    float sv = is_first ? xv : (xv + __bfloat162float(res_bf16[idx]));
    res_bf16[idx]  = __float2bfloat16(sv);
    s_out_f32[idx] = sv;
    wv_pdl_trigger();
}

// Applies q_norm [128] & k_norm [128] + neox RoPE (1e6) at (*d_mtp_pos0) + s (s = 0..7):
// Grid <<<B * 8, dim3(32, 8)>>>
__global__ void wv_ink_mtp_qkv_norm_rope_kernel(
    __nv_bfloat16* __restrict__ q_io,             // [B * 8, 24, 128]
    __nv_bfloat16* __restrict__ bk_io,            // [B * 8, 8, 128]
    const float* __restrict__ q_norm,             // [128]
    const float* __restrict__ k_norm,             // [128]
    const int32_t* __restrict__ d_mtp_pos0,       // [1]
    int total_m) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m >= total_m) { wv_pdl_trigger(); return; }
    int s    = m & 7;
    int pos  = (*d_mtp_pos0) + s;
    int lane = threadIdx.x; // 0..31
    int wid  = threadIdx.y; // 0..7

    for (int sub = 0; sub < 3; ++sub) {
        int hq = wid * 3 + sub;
        __nv_bfloat16* qp = q_io + static_cast<size_t>(m) * 3072 + hq * 128;
        float q0 = __bfloat162float(qp[lane]),      q1 = __bfloat162float(qp[lane + 32]);
        float q2 = __bfloat162float(qp[lane + 64]), q3 = __bfloat162float(qp[lane + 96]);
        float inv = rsqrtf(wv_warp_sum_f32(q0*q0 + q1*q1 + q2*q2 + q3*q3) * (1.0f / 128.0f) + 1e-6f);
        q0 = __bfloat162float(__float2bfloat16(q0 * inv * q_norm[lane]));
        q1 = __bfloat162float(__float2bfloat16(q1 * inv * q_norm[lane + 32]));
        q2 = __bfloat162float(__float2bfloat16(q2 * inv * q_norm[lane + 64]));
        q3 = __bfloat162float(__float2bfloat16(q3 * inv * q_norm[lane + 96]));
        wv_ink_apply_neox_rope_pair(q0, q2, lane, pos);
        wv_ink_apply_neox_rope_pair(q1, q3, lane + 32, pos);
        qp[lane]      = __float2bfloat16(q0); qp[lane + 32] = __float2bfloat16(q1);
        qp[lane + 64] = __float2bfloat16(q2); qp[lane + 96] = __float2bfloat16(q3);
    }

    __nv_bfloat16* kp = bk_io + static_cast<size_t>(m) * 1024 + wid * 128;
    float k0 = __bfloat162float(kp[lane]),      k1 = __bfloat162float(kp[lane + 32]);
    float k2 = __bfloat162float(kp[lane + 64]), k3 = __bfloat162float(kp[lane + 96]);
    float inv_k = rsqrtf(wv_warp_sum_f32(k0*k0 + k1*k1 + k2*k2 + k3*k3) * (1.0f / 128.0f) + 1e-6f);
    k0 = __bfloat162float(__float2bfloat16(k0 * inv_k * k_norm[lane]));
    k1 = __bfloat162float(__float2bfloat16(k1 * inv_k * k_norm[lane + 32]));
    k2 = __bfloat162float(__float2bfloat16(k2 * inv_k * k_norm[lane + 64]));
    k3 = __bfloat162float(__float2bfloat16(k3 * inv_k * k_norm[lane + 96]));
    wv_ink_apply_neox_rope_pair(k0, k2, lane, pos);
    wv_ink_apply_neox_rope_pair(k1, k3, lane + 32, pos);
    kp[lane]      = __float2bfloat16(k0); kp[lane + 32] = __float2bfloat16(k1);
    kp[lane + 64] = __float2bfloat16(k2); kp[lane + 96] = __float2bfloat16(k3);
    wv_pdl_trigger();
}

// Bidirectional (is_causal = false) Online GQA over [committed MTP KV ++ 8-token block KV] + FWHT128 + Q8_32:
// Grid <<<B * 8 * 24, 1>>>, Block dim3(32, 4)
__global__ void wv_ink_mtp_bidir_gqa_fwht_q8_kernel(
    const __nv_bfloat16* __restrict__ q_all,      // [B * 8, 24, 128]
    const __nv_bfloat16* __restrict__ k_ctx,      // [B_max, kv_cap, 8, 128]
    const __nv_bfloat16* __restrict__ v_ctx,      // [B_max, kv_cap, 8, 128]
    const __nv_bfloat16* __restrict__ bk_blk,     // [B, 8, 8, 128]
    const __nv_bfloat16* __restrict__ bv_blk,     // [B, 8, 8, 128]
    int8_t* __restrict__ O_q8,                    // [B * 8, 3072]
    float* __restrict__ O_sc,                     // [B * 8, 96]
    const int32_t* __restrict__ d_mtp_pos0,       // [1]
    int kv_cap, int is_local) {
    wv_pdl_wait();
    int mh   = blockIdx.x;
    int m    = mh / 24;
    int hq   = mh - m * 24;
    int hkv  = hq / 3;
    int b    = m >> 3;
    int lane = threadIdx.x, wid = threadIdx.y;

    const __nv_bfloat16* qp = q_all + static_cast<size_t>(mh) * 128 + lane * 4;
    constexpr float kScale = 0.08838834764831845f;
    float q0 = __bfloat162float(qp[0]) * kScale, q1 = __bfloat162float(qp[1]) * kScale;
    float q2 = __bfloat162float(qp[2]) * kScale, q3 = __bfloat162float(qp[3]) * kScale;

    int pos0    = *d_mtp_pos0;
    int ctx_beg = is_local ? max(0, pos0 - INK2_SWA_WIN) : 0;
    int n_ctx   = pos0 - ctx_beg;
    int tot_kv  = n_ctx + 8;

    float m_max = -1e30f, l_sum = 0.0f;
    float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
    for (int idx = wid; idx < tot_kv; idx += 4) {
        const __nv_bfloat16* kt;
        const __nv_bfloat16* vt;
        if (idx < n_ctx) {
            int kp   = ctx_beg + idx;
            int slot = is_local ? (kp % kv_cap) : kp;
            size_t off = (static_cast<size_t>(b) * kv_cap + slot) * 1024 + hkv * 128 + lane * 4;
            kt = k_ctx + off;
            vt = v_ctx + off;
        } else {
            int bs = idx - n_ctx;
            size_t off = (static_cast<size_t>(b * 8 + bs) * 8 + hkv) * 128 + lane * 4;
            kt = bk_blk + off;
            vt = bv_blk + off;
        }
        float qk = wv_warp_sum_f32(
            q0 * __bfloat162float(kt[0]) + q1 * __bfloat162float(kt[1]) +
            q2 * __bfloat162float(kt[2]) + q3 * __bfloat162float(kt[3]));
        float mn = fmaxf(m_max, qk);
        float al = (m_max > -1e29f) ? expf(m_max - mn) : 0.0f;
        float wt = expf(qk - mn);
        l_sum = l_sum * al + wt;
        m_max = mn;
        o0 = o0 * al + wt * __bfloat162float(vt[0]); o1 = o1 * al + wt * __bfloat162float(vt[1]);
        o2 = o2 * al + wt * __bfloat162float(vt[2]); o3 = o3 * al + wt * __bfloat162float(vt[3]);
    }

    __shared__ float s_m[4], s_l[4], s_o[4][128];
    if (lane == 0) { s_m[wid] = m_max; s_l[wid] = l_sum; }
    s_o[wid][lane * 4 + 0] = o0; s_o[wid][lane * 4 + 1] = o1;
    s_o[wid][lane * 4 + 2] = o2; s_o[wid][lane * 4 + 3] = o3;
    __syncthreads();

    if (wid == 0) {
        float fm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
        float fl = 0.0f, fo0 = 0.f, fo1 = 0.f, fo2 = 0.f, fo3 = 0.f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (s_l[w] > 0.0f) {
                float a = expf(s_m[w] - fm);
                fl  += s_l[w] * a;
                fo0 += s_o[w][lane * 4 + 0] * a; fo1 += s_o[w][lane * 4 + 1] * a;
                fo2 += s_o[w][lane * 4 + 2] * a; fo3 += s_o[w][lane * 4 + 3] * a;
            }
        }
        float inv = 1.0f / fmaxf(fl, 1e-20f);
        float f0 = __bfloat162float(__float2bfloat16(fo0 * inv));
        float f1 = __bfloat162float(__float2bfloat16(fo1 * inv));
        float f2 = __bfloat162float(__float2bfloat16(fo2 * inv));
        float f3 = __bfloat162float(__float2bfloat16(fo3 * inv));
        wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
        wv_quant_q8_32_warp(f0, f1, f2, f3, lane, O_q8 + static_cast<size_t>(mh) * 128, O_sc + mh * 4);
    }
    wv_pdl_trigger();
}

// S-Shared Bidirectional Split-KV Flash-GQA for MTP Global layer (`B == 1, 8` draft tokens):
// Streams `k_ctx`/`v_ctx` (`INT8` `__dp4a` or `BF16`) ONCE per split across all 8 draft tokens and 3 GQA heads!
// Grid `<<<dim3(8, n_splits), dim3(32, 8)>>>`
template<bool KV_Q8>
__global__ void wv_ink_mtp_bidir_gqa3_s_shared_split_kernel(
    const __nv_bfloat16* __restrict__ q_all,      // [8, 24, 128]
    const void* __restrict__ k_ctx,               // [1, kv_cap, 8, 128] bf16 or int8
    const void* __restrict__ v_ctx,               // [1, kv_cap, 8, 128] bf16 or int8
    const float* __restrict__ k_ctx_sc,           // [1, kv_cap, 8] or nullptr
    const float* __restrict__ v_ctx_sc,           // [1, kv_cap, 8] or nullptr
    const __nv_bfloat16* __restrict__ bk_blk,     // [1, 8, 8, 128]
    const __nv_bfloat16* __restrict__ bv_blk,     // [1, 8, 8, 128]
    const int32_t* __restrict__ d_mtp_pos0,       // [1]
    float* __restrict__ m_part,                   // [8, 24, n_splits]
    float* __restrict__ l_part,                   // [8, 24, n_splits]
    float* __restrict__ o_part,                   // [8, 24, n_splits, 128]
    int kv_cap, int n_splits) {
    wv_pdl_wait();
    int hkv    = blockIdx.x; // 0..7
    int sp     = blockIdx.y; // 0..n_splits-1
    int hq0    = hkv * 3;
    int lane   = threadIdx.x; // 0..31
    int wid    = threadIdx.y; // 0..7 (draft token s = wid)
    int n_ctx  = min(kv_cap, max(0, *d_mtp_pos0));
    int tot_kv = n_ctx + 8;
    int chunk  = (tot_kv + n_splits - 1) / n_splits;
    int p0     = sp * chunk;
    int p1     = min(tot_kv, p0 + chunk);

    if (p0 >= p1) {
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            int mh = wid * 24 + hq0 + g;
            size_t idx_sp = static_cast<size_t>(mh) * n_splits + sp;
            if (lane == 0) { m_part[idx_sp] = -1e30f; l_part[idx_sp] = 0.0f; }
            float* od = o_part + idx_sp * 128 + lane * 4;
            od[0] = 0.0f; od[1] = 0.0f; od[2] = 0.0f; od[3] = 0.0f;
        }
        wv_pdl_trigger();
        return;
    }

    __shared__ union {
        struct { uint32_t k[8][32]; uint32_t v[8][32]; float ks[8]; float vs[8]; } q8;
        struct { uint32_t kl[8][32], kh[8][32]; uint32_t vl[8][32], vh[8][32]; } bf;
    } s_kv;

    constexpr float kScale = 0.08838834764831845f;
    float    qv[3][4];
    uint32_t q_pk[3] = {0u, 0u, 0u};
    float    q_sc[3] = {0.f, 0.f, 0.f};
    float    m_max[3], l_sum[3], o_acc[3][4];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        const __nv_bfloat16* qp = q_all + static_cast<size_t>(wid * 24 + hq0 + g) * 128 + lane * 4;
        qv[g][0] = __bfloat162float(qp[0]) * kScale;
        qv[g][1] = __bfloat162float(qp[1]) * kScale;
        qv[g][2] = __bfloat162float(qp[2]) * kScale;
        qv[g][3] = __bfloat162float(qp[3]) * kScale;
        if constexpr (KV_Q8) {
            q_pk[g] = wv_ink_pack_q8_head128_warp(qv[g][0], qv[g][1], qv[g][2], qv[g][3], q_sc[g]);
        }
        m_max[g] = -1e30f; l_sum[g] = 0.0f;
        o_acc[g][0] = 0.0f; o_acc[g][1] = 0.0f; o_acc[g][2] = 0.0f; o_acc[g][3] = 0.0f;
    }

    const uint32_t* kb_i8      = KV_Q8 ? (static_cast<const uint32_t*>(k_ctx) + static_cast<size_t>(hkv) * 32 + lane) : nullptr;
    const uint32_t* vb_i8      = KV_Q8 ? (static_cast<const uint32_t*>(v_ctx) + static_cast<size_t>(hkv) * 32 + lane) : nullptr;
    const float*    ks_base    = KV_Q8 ? (k_ctx_sc + hkv) : nullptr;
    const float*    vs_base    = KV_Q8 ? (v_ctx_sc + hkv) : nullptr;
    const __nv_bfloat16* kb_bf = (!KV_Q8) ? (static_cast<const __nv_bfloat16*>(k_ctx) + hkv * 128 + lane * 4) : nullptr;
    const __nv_bfloat16* vb_bf = (!KV_Q8) ? (static_cast<const __nv_bfloat16*>(v_ctx) + hkv * 128 + lane * 4) : nullptr;

    int hi_ctx = min(p1, n_ctx);
    for (int kp_base = p0; kp_base < hi_ctx; kp_base += 8) {
        int tile_cnt = min(8, hi_ctx - kp_base);
        if (wid < tile_cnt) {
            int kp = kp_base + wid;
            if constexpr (KV_Q8) {
                size_t tok_off = static_cast<size_t>(kp) * 256;
                s_kv.q8.k[wid][lane] = wv_ld_stream_u32(kb_i8 + tok_off);
                s_kv.q8.v[wid][lane] = wv_ld_stream_u32(vb_i8 + tok_off);
                if (lane == 0) {
                    size_t sc_off = static_cast<size_t>(kp) * 8;
                    s_kv.q8.ks[wid] = ks_base[sc_off];
                    s_kv.q8.vs[wid] = vs_base[sc_off];
                }
            } else {
                size_t s_off = static_cast<size_t>(kp) * 1024;
                uint2 kl2 = *reinterpret_cast<const uint2*>(kb_bf + s_off);
                uint2 vl2 = *reinterpret_cast<const uint2*>(vb_bf + s_off);
                s_kv.bf.kl[wid][lane] = kl2.x; s_kv.bf.kh[wid][lane] = kl2.y;
                s_kv.bf.vl[wid][lane] = vl2.x; s_kv.bf.vh[wid][lane] = vl2.y;
            }
        }
        __syncthreads();

        for (int t = 0; t < tile_cnt; ++t) {
            if constexpr (KV_Q8) {
                uint32_t k_pk = s_kv.q8.k[t][lane];
                uint32_t v_pk = s_kv.q8.v[t][lane];
                float k_sc    = s_kv.q8.ks[t];
                float v_sc    = s_kv.q8.vs[t];
                float vt0, vt1, vt2, vt3;
                wv_ink_unpack_s8x4_f32(v_pk, vt0, vt1, vt2, vt3);
                #pragma unroll
                for (int g = 0; g < 3; ++g) {
                    int32_t dp  = __dp4a(static_cast<int32_t>(q_pk[g]), static_cast<int32_t>(k_pk), 0);
                    float qk    = wv_warp_sum_f32(static_cast<float>(dp)) * (q_sc[g] * k_sc);
                    float m_new = fmaxf(m_max[g], qk);
                    float alpha = (m_max[g] > -1e29f) ? expf(m_max[g] - m_new) : 0.0f;
                    float wt    = expf(qk - m_new);
                    float wt_v  = wt * v_sc;
                    l_sum[g]    = l_sum[g] * alpha + wt;
                    m_max[g]    = m_new;
                    o_acc[g][0] = o_acc[g][0] * alpha + wt_v * vt0;
                    o_acc[g][1] = o_acc[g][1] * alpha + wt_v * vt1;
                    o_acc[g][2] = o_acc[g][2] * alpha + wt_v * vt2;
                    o_acc[g][3] = o_acc[g][3] * alpha + wt_v * vt3;
                }
            } else {
                uint32_t kl_w = s_kv.bf.kl[t][lane], kh_w = s_kv.bf.kh[t][lane];
                uint32_t vl_w = s_kv.bf.vl[t][lane], vh_w = s_kv.bf.vh[t][lane];
                float2 kt01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kl_w));
                float2 kt23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kh_w));
                float2 vt01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vl_w));
                float2 vt23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vh_w));
                #pragma unroll
                for (int g = 0; g < 3; ++g) {
                    float qk = qv[g][0] * kt01.x + qv[g][1] * kt01.y + qv[g][2] * kt23.x + qv[g][3] * kt23.y;
                    qk = wv_warp_sum_f32(qk);
                    float m_new = fmaxf(m_max[g], qk);
                    float alpha = (m_max[g] > -1e29f) ? expf(m_max[g] - m_new) : 0.0f;
                    float wt    = expf(qk - m_new);
                    l_sum[g]    = l_sum[g] * alpha + wt;
                    m_max[g]    = m_new;
                    o_acc[g][0] = o_acc[g][0] * alpha + wt * vt01.x;
                    o_acc[g][1] = o_acc[g][1] * alpha + wt * vt01.y;
                    o_acc[g][2] = o_acc[g][2] * alpha + wt * vt23.x;
                    o_acc[g][3] = o_acc[g][3] * alpha + wt * vt23.y;
                }
            }
        }
        __syncthreads();
    }

    int p2_beg = max(p0, n_ctx);
    for (int idx = p2_beg; idx < p1; ++idx) {
        int bs = idx - n_ctx;
        size_t off = (static_cast<size_t>(bs) * 8 + hkv) * 128 + lane * 4;
        const __nv_bfloat16* kt = bk_blk + off;
        const __nv_bfloat16* vt = bv_blk + off;
        float kt0 = __bfloat162float(kt[0]), kt1 = __bfloat162float(kt[1]);
        float kt2 = __bfloat162float(kt[2]), kt3 = __bfloat162float(kt[3]);
        float vt0 = __bfloat162float(vt[0]), vt1 = __bfloat162float(vt[1]);
        float vt2 = __bfloat162float(vt[2]), vt3 = __bfloat162float(vt[3]);
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            float qk = qv[g][0] * kt0 + qv[g][1] * kt1 + qv[g][2] * kt2 + qv[g][3] * kt3;
            qk = wv_warp_sum_f32(qk);
            float m_new = fmaxf(m_max[g], qk);
            float alpha = (m_max[g] > -1e29f) ? expf(m_max[g] - m_new) : 0.0f;
            float wt    = expf(qk - m_new);
            l_sum[g]    = l_sum[g] * alpha + wt;
            m_max[g]    = m_new;
            o_acc[g][0] = o_acc[g][0] * alpha + wt * vt0;
            o_acc[g][1] = o_acc[g][1] * alpha + wt * vt1;
            o_acc[g][2] = o_acc[g][2] * alpha + wt * vt2;
            o_acc[g][3] = o_acc[g][3] * alpha + wt * vt3;
        }
    }

    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        int mh = wid * 24 + hq0 + g;
        size_t idx_sp = static_cast<size_t>(mh) * n_splits + sp;
        if (lane == 0) { m_part[idx_sp] = m_max[g]; l_part[idx_sp] = l_sum[g]; }
        float* od = o_part + idx_sp * 128 + lane * 4;
        od[0] = o_acc[g][0]; od[1] = o_acc[g][1]; od[2] = o_acc[g][2]; od[3] = o_acc[g][3];
    }
    wv_pdl_trigger();
}

// Combines `n_splits` MTP Global Flash-GQA partials + BF16 round + FWHT-128 + Q8_32:
// Grid `<<<(total_mh + 3) / 4, dim3(32, 4)>>>`
__global__ void wv_ink_mtp_flash_gqa_combine_fwht_q8_kernel(
    const float* __restrict__ m_part,             // [8, 24, n_splits]
    const float* __restrict__ l_part,             // [8, 24, n_splits]
    const float* __restrict__ o_part,             // [8, 24, n_splits, 128]
    int8_t* __restrict__ O_q8,                    // [8, 3072]
    float* __restrict__ O_sc,                     // [8, 96]
    int total_mh,
    int n_splits) {
    wv_pdl_wait();
    int mh = blockIdx.x * blockDim.y + threadIdx.y;
    if (mh >= total_mh) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x;
    size_t sp_base = static_cast<size_t>(mh) * n_splits;
    float m_tot = -1e30f;
    for (int s = 0; s < n_splits; ++s) {
        if (l_part[sp_base + s] > 0.0f) {
            m_tot = fmaxf(m_tot, m_part[sp_base + s]);
        }
    }
    float denom = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    if (m_tot > -1e29f) {
        for (int s = 0; s < n_splits; ++s) {
            float ls = l_part[sp_base + s];
            if (ls > 0.0f) {
                float a = expf(m_part[sp_base + s] - m_tot);
                denom += ls * a;
                const float* op = o_part + (sp_base + s) * 128 + lane * 4;
                o0 += op[0] * a; o1 += op[1] * a; o2 += op[2] * a; o3 += op[3] * a;
            }
        }
    }
    float inv = 1.0f / fmaxf(denom, 1e-20f);
    float f0 = __bfloat162float(__float2bfloat16(o0 * inv));
    float f1 = __bfloat162float(__float2bfloat16(o1 * inv));
    float f2 = __bfloat162float(__float2bfloat16(o2 * inv));
    float f3 = __bfloat162float(__float2bfloat16(o3 * inv));
    wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
    wv_quant_q8_32_warp(f0, f1, f2, f3, lane, O_q8 + static_cast<size_t>(mh) * 128, O_sc + mh * 4);
    wv_pdl_trigger();
}

// Standard MTP SwiGLU bf16(silu(gate) * up) + FWHT128 + Q8_32 ([M, 9216]):
// Grid <<<(total_blks128 + 3) / 4, dim3(32, 4)>>>
__global__ void wv_ink_mtp_silu_mul_fwht_q8_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int8_t* __restrict__ Y_q8,
    float* __restrict__ Y_sc,
    int total_blks128) {
    wv_pdl_wait();
    int blk_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (blk_id >= total_blks128) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x;
    size_t off = static_cast<size_t>(blk_id) * 128 + lane * 4;
    float f[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        float g = __bfloat162float(gate[off + r]);
        float u = __bfloat162float(up[off + r]);
        f[r] = __bfloat162float(__float2bfloat16((g / (1.0f + expf(-g))) * u));
    }
    wv_fwht128_warp_inplace(f[0], f[1], f[2], f[3], lane);
    wv_quant_q8_32_warp(f[0], f[1], f[2], f[3], lane, Y_q8 + static_cast<size_t>(blk_id) * 128, Y_sc + blk_id * 4);
    wv_pdl_trigger();
}

// Computes ctx [7, 768] = [h_draft(2304), f_rnn(2304)] @ rnn_w[:, 512:]^T + rnn_b:
// Grid <<<96, 7>>>, Block dim3(32, 8)
__global__ void wv_ink_mtp_markov_ctx_kernel(
    const __nv_bfloat16* __restrict__ h_blk8,     // [8, 2304]
    const __nv_bfloat16* __restrict__ f_rnn,      // [2304]
    const float* __restrict__ rnn_w,              // [768, 5120]
    const float* __restrict__ rnn_b,              // [768]
    float* __restrict__ ctx_out) {                // [7, 768]
    wv_pdl_wait();
    int r    = blockIdx.x * 8 + threadIdx.y;      // 0..767
    int t    = blockIdx.y;                        // 0..6
    int lane = threadIdx.x;
    if (r >= 768 || t >= 7) { wv_pdl_trigger(); return; }

    const __nv_bfloat16* hd = h_blk8 + static_cast<size_t>(1 + t) * INK2_HIDDEN;
    const float* wr         = rnn_w + static_cast<size_t>(r) * 5120 + 512;
    float acc = 0.0f;
    for (int i = lane; i < INK2_HIDDEN; i += 32) {
        acc += wr[i] * __bfloat162float(hd[i]) + wr[INK2_HIDDEN + i] * __bfloat162float(f_rnn[i]);
    }
    acc = wv_warp_sum_f32(acc);
    if (lane == 0) {
        ctx_out[t * 768 + r] = acc + rnn_b[r];
    }
    wv_pdl_trigger();
}

// Multi-CTA MarkovHead RNN step Stage A (<<<96, dim3(32, 8)>>>):
__global__ void wv_ink_mtp_markov_rnn_gemv_kernel(
    const int32_t* __restrict__ prev_tok_ptr,     // points to draft_blk8[t]
    const int8_t* __restrict__ mh_w1_i8,          // [vocab_full, 256]
    const __half* __restrict__ mh_w1_sc,          // [vocab_full, 8]
    const float* __restrict__ rnn_w,              // [768, 5120]
    const float* __restrict__ ctx_t,              // [768]
    const __nv_bfloat16* __restrict__ state_bf16, // [256] (zeroed when t == 0)
    float* __restrict__ raw768_out,               // [768]
    int step_t,
    int vocab_full) {
    wv_pdl_wait();
    int lane = threadIdx.x;
    int wid  = threadIdx.y;
    int r    = blockIdx.x * 8 + wid; // 0..767
    int tid  = wid * 32 + lane;      // 0..255

    __shared__ float s_inp[512];
    int tok = *prev_tok_ptr;
    if (tok < 0 || tok >= vocab_full) tok = 0;
    float st_v = (step_t == 0) ? 0.0f : __bfloat162float(state_bf16[tid]);
    float w1_s = __half2float(__ldg(mh_w1_sc + static_cast<size_t>(tok) * 8 + wid));
    float w1_v = static_cast<float>(__ldg(mh_w1_i8 + static_cast<size_t>(tok) * 256 + tid)) * w1_s;
    s_inp[tid]       = st_v;
    s_inp[256 + tid] = __bfloat162float(__float2bfloat16(w1_v));
    __syncthreads();

    const float4* wr4 = reinterpret_cast<const float4*>(rnn_w + static_cast<size_t>(r) * 5120);
    float acc = 0.0f;
    #pragma unroll
    for (int k4 = lane; k4 < 128; k4 += 32) {
        float4 wv4 = __ldg(wr4 + k4);
        float4 iv  = reinterpret_cast<const float4*>(s_inp)[k4];
        acc += wv4.x * iv.x + wv4.y * iv.y + wv4.z * iv.z + wv4.w * iv.w;
    }
    acc = wv_warp_sum_f32(acc);
    if (lane == 0) {
        raw768_out[r] = __bfloat162float(__float2bfloat16(acc + ctx_t[r]));
    }
    wv_pdl_trigger();
}

// Multi-CTA MarkovHead RNN step Stage B (<<<1, 256>>>):
__global__ void wv_ink_mtp_markov_rnn_act_q8_kernel(
    const float* __restrict__ raw768_in,          // [768]
    __nv_bfloat16* __restrict__ state_bf16,       // [256]
    int8_t* __restrict__ lat_q8,                  // [256]
    float* __restrict__ lat_sc,                   // [8]
    int step_t) {
    wv_pdl_wait();
    int tid  = threadIdx.x; // 0..255
    int lane = tid & 31, wid = tid >> 5;
    float st_v = (step_t == 0) ? 0.0f : __bfloat162float(state_bf16[tid]);
    float g  = raw768_in[tid];
    float c  = raw768_in[256 + tid];
    float gb = __bfloat162float(__float2bfloat16(1.0f / (1.0f + expf(-g))));
    float cb = __bfloat162float(__float2bfloat16(tanhf(c)));
    float f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f;
    if (wid < 2) {
        int base = wid * 128 + lane * 4;
        f0 = __bfloat162float(__float2bfloat16(tanhf(raw768_in[512 + base + 0])));
        f1 = __bfloat162float(__float2bfloat16(tanhf(raw768_in[512 + base + 1])));
        f2 = __bfloat162float(__float2bfloat16(tanhf(raw768_in[512 + base + 2])));
        f3 = __bfloat162float(__float2bfloat16(tanhf(raw768_in[512 + base + 3])));
        wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
    }
    state_bf16[tid] = __float2bfloat16(gb * st_v + (1.0f - gb) * cb);
    if (wid < 2) {
        wv_quant_q8_32_warp(f0, f1, f2, f3, lane, lat_q8 + wid * 128, lat_sc + wid * 4);
    }
    wv_pdl_trigger();
}

// Parallel Softcap In-Place: Grid <<<(valid_vocab + 255) / 256, 256>>>
__global__ void wv_ink_mtp_combine_softcap_inplace_kernel(
    float* __restrict__ base_logits_io,
    const float* __restrict__ bias_logits_t,
    int valid_vocab) {
    wv_pdl_wait();
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < valid_vocab) {
        float s  = __bfloat162float(__float2bfloat16(base_logits_io[i] + bias_logits_t[i]));
        float sc = 30.0f * tanhf(s * (1.0f / 30.0f));
        base_logits_io[i] = sc;
    }
    wv_pdl_trigger();
}

// Fused MTP q [3072] + k [1024] + v [1024] (160 CTAs x 128, CTA_M = 8 mma.m16n8k32) at M = 8:
__global__ __launch_bounds__(128, 6)
void wv_ink_swz8_fused_mtp_qkv_m8_kernel(
    WvInkSwz8Linear wq, WvInkSwz8Linear wk, WvInkSwz8Linear w_v,
    const int8_t* __restrict__ X_q8, const float* __restrict__ X_sc,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out) {
    __shared__ uint32_t s_X[8][68];
    wv_pdl_wait();
    int cta     = blockIdx.x; // 0..159 (96 q + 32 k + 32 v)
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int lane_n  = lane >> 2;
    int c_k     = lane & 3;
    int r0      = lane >> 2;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;

    const uint2*   w_pk;
    const __half*  w_sc;
    __nv_bfloat16* dst;
    int n_tile8, N;
    if (cta < 96) {
        w_pk = wq.pk; w_sc = wq.sc; dst = q_out; n_tile8 = cta * 4 + warp_id; N = INK2_ATTN_DIM;
    } else if (cta < 128) {
        w_pk = wk.pk; w_sc = wk.sc; dst = k_out; n_tile8 = (cta - 96) * 4 + warp_id; N = INK2_KV_DIM;
    } else {
        w_pk = w_v.pk; w_sc = w_v.sc; dst = v_out; n_tile8 = (cta - 128) * 4 + warp_id; N = INK2_KV_DIM;
    }

    int out_n0 = n_tile8 * 8 + col_n0;
    constexpr int n_kc32    = INK2_HIDDEN >> 5;
    constexpr float kInv127 = 1.0f / 127.0f;
    const uint2*  pk_base   = w_pk + static_cast<size_t>(n_tile8) * n_kc32 * 32 + lane;
    const __half* sc_base   = w_sc + static_cast<size_t>(n_tile8) * n_kc32 * 8 + lane_n;

    float acc0 = 0.0f, acc1 = 0.0f;
    int ld_m = threadIdx.x >> 4, ld_q = threadIdx.x & 15;
    for (int kc256 = 0; kc256 < 18; kc256 += 2) {
        uint4 xv = reinterpret_cast<const uint4*>(X_q8 + static_cast<size_t>(ld_m) * INK2_HIDDEN + kc256 * 128)[ld_q];
        int c4 = ld_q * 4;
        s_X[ld_m][c4 + 0] = xv.x; s_X[ld_m][c4 + 1] = xv.y;
        s_X[ld_m][c4 + 2] = xv.z; s_X[ld_m][c4 + 3] = xv.w;
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            int kc128 = kc256 + c;
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int kc32 = kc128 * 4 + sub, s_off = (c * 4 + sub) * 8 + c_k;
                uint2 pkv = __ldg(pk_base + kc32 * 32);
                float ws_self = __half2float(__ldg(sc_base + kc32 * 8)) * kInv127;
                float ws0 = __shfl_sync(0xffffffffu, ws_self, col_n0 * 4);
                float ws1 = __shfl_sync(0xffffffffu, ws_self, col_n1 * 4);
                float xs0 = X_sc[static_cast<size_t>(r0) * n_kc32 + kc32];
                int d0, d1, d2, d3;
                wv_mma_m16n8k32_s8(d0, d1, d2, d3, s_X[r0][s_off], 0u, s_X[r0][s_off + 4], 0u, pkv.x, pkv.y);
                acc0 += static_cast<float>(d0) * (ws0 * xs0);
                acc1 += static_cast<float>(d1) * (ws1 * xs0);
            }
        }
        __syncthreads();
    }
    size_t row_off = static_cast<size_t>(r0) * N;
    dst[row_off + out_n0]     = __float2bfloat16(acc0);
    dst[row_off + out_n0 + 1] = __float2bfloat16(acc1);
    wv_pdl_trigger();
}

// Fused MTP gate [9216] + up [9216] (576 CTAs x 128, CTA_M = 8 mma.m16n8k32) at M = 8:
__global__ __launch_bounds__(128, 6)
void wv_ink_swz8_fused_mtp_gu_m8_kernel(
    WvInkSwz8Linear w_gate, WvInkSwz8Linear w_up,
    const int8_t* __restrict__ X_q8, const float* __restrict__ X_sc,
    __nv_bfloat16* __restrict__ g_out,
    __nv_bfloat16* __restrict__ u_out) {
    __shared__ uint32_t s_X[8][68];
    wv_pdl_wait();
    int cta     = blockIdx.x; // 0..575 (288 gate + 288 up)
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int lane_n  = lane >> 2;
    int c_k     = lane & 3;
    int r0      = lane >> 2;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;
    bool is_g   = (cta < 288);
    int n_tile8 = (is_g ? cta : (cta - 288)) * 4 + warp_id;
    const uint2*   w_pk = is_g ? w_gate.pk : w_up.pk;
    const __half*  w_sc = is_g ? w_gate.sc : w_up.sc;
    __nv_bfloat16* dst  = is_g ? g_out     : u_out;

    int out_n0 = n_tile8 * 8 + col_n0;
    constexpr int n_kc32    = INK2_HIDDEN >> 5;
    constexpr float kInv127 = 1.0f / 127.0f;
    const uint2*  pk_base   = w_pk + static_cast<size_t>(n_tile8) * n_kc32 * 32 + lane;
    const __half* sc_base   = w_sc + static_cast<size_t>(n_tile8) * n_kc32 * 8 + lane_n;

    float acc0 = 0.0f, acc1 = 0.0f;
    int ld_m = threadIdx.x >> 4, ld_q = threadIdx.x & 15;
    for (int kc256 = 0; kc256 < 18; kc256 += 2) {
        uint4 xv = reinterpret_cast<const uint4*>(X_q8 + static_cast<size_t>(ld_m) * INK2_HIDDEN + kc256 * 128)[ld_q];
        int c4 = ld_q * 4;
        s_X[ld_m][c4 + 0] = xv.x; s_X[ld_m][c4 + 1] = xv.y;
        s_X[ld_m][c4 + 2] = xv.z; s_X[ld_m][c4 + 3] = xv.w;
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            int kc128 = kc256 + c;
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int kc32 = kc128 * 4 + sub, s_off = (c * 4 + sub) * 8 + c_k;
                uint2 pkv = __ldg(pk_base + kc32 * 32);
                float ws_self = __half2float(__ldg(sc_base + kc32 * 8)) * kInv127;
                float ws0 = __shfl_sync(0xffffffffu, ws_self, col_n0 * 4);
                float ws1 = __shfl_sync(0xffffffffu, ws_self, col_n1 * 4);
                float xs0 = X_sc[static_cast<size_t>(r0) * n_kc32 + kc32];
                int d0, d1, d2, d3;
                wv_mma_m16n8k32_s8(d0, d1, d2, d3, s_X[r0][s_off], 0u, s_X[r0][s_off + 4], 0u, pkv.x, pkv.y);
                acc0 += static_cast<float>(d0) * (ws0 * xs0);
                acc1 += static_cast<float>(d1) * (ws1 * xs0);
            }
        }
        __syncthreads();
    }
    size_t row_off = static_cast<size_t>(r0) * INK2_MTP_INTER;
    dst[row_off + out_n0]     = __float2bfloat16(acc0);
    dst[row_off + out_n0 + 1] = __float2bfloat16(acc1);
    wv_pdl_trigger();
}

// Stochastic MTP-8 Draft Sampler (t in 0..6): Grid <<<1, 512>>>
__global__ void wv_ink_mtp_combine_softcap_sample_topp_kernel(
    const float* __restrict__ base_logits_t,
    float* __restrict__ q_probs_t,
    int32_t* __restrict__ out_tok_ptr,
    const WvInkSampleCfg* __restrict__ cfg,
    int step_t,
    int stride_v,
    int valid_vocab) {
    wv_pdl_wait();
    int tid = threadIdx.x;
    auto lfn = [&](int i) -> float { return base_logits_t[i]; };
    WvInkTopPCutoff cut = wv_ink_compute_topp_cutoff_cta512(lfn, valid_vocab, cfg->temp, cfg->top_p);
    for (int i = tid; i < stride_v; i += 512) {
        if (i < valid_vocab) {
            float lv = lfn(i);
            q_probs_t[i] = wv_ink_topp_is_kept(lv, i, cut) ? (expf((lv - cut.m_max) * cut.inv_temp) * cut.inv_kept_Z) : 0.0f;
        } else {
            q_probs_t[i] = 0.0f;
        }
    }
    __syncthreads();
    float u01 = wv_ink_rng_uniform01(cfg->seed, cfg->rng_step, static_cast<uint32_t>(step_t));
    int tok   = wv_ink_sample_cdf_cta512([&](int i) { return q_probs_t[i]; }, valid_vocab, 1.0f, u01, cut.argmax_idx);
    if (tid == 0) *out_tok_ptr = tok;
    wv_pdl_trigger();
}

__global__ void wv_ink_advance_m1_slots_kernel(
    int32_t* __restrict__ d_slot_pos,
    const int32_t* __restrict__ d_slot_active,
    WvInkSampleCfg* __restrict__ cfg,
    int B_act, int S) {
    wv_pdl_wait();
    int b = threadIdx.x;
    if (b < B_act) {
        int act = (d_slot_active != nullptr) ? d_slot_active[b] : 1;
        if (act > 0) d_slot_pos[b] += S;
    }
    if (b == 0 && cfg != nullptr) cfg->rng_step += 1ULL;
    wv_pdl_trigger();
}

__global__ void wv_ink_advance_mtp8_counters_kernel(
    int32_t* __restrict__ d_slot_pos,
    int32_t* __restrict__ d_mtp_pos0,
    const int32_t* __restrict__ d_step_out,
    WvInkSampleCfg* __restrict__ cfg) {
    wv_pdl_wait();
    if (threadIdx.x == 0) {
        int n = d_step_out[0];
        d_slot_pos[0] += n;
        d_mtp_pos0[0] += n;
        if (cfg != nullptr) cfg->rng_step += 1ULL;
    }
    wv_pdl_trigger();
}

// Fused MTP Anchor Embed Gather+RMSNorm + 8-Token Block Init ([emb_anchor, mask_embed x 7] in 1 launch):
// Grid <<<1, dim3(32, 6)>>>
__global__ void wv_ink_mtp_embed_init_block8_kernel(
    const int32_t* __restrict__ tok_id_ptr,
    const uint8_t* __restrict__ emb_wire,
    const float* __restrict__ embed_norm,
    const __nv_bfloat16* __restrict__ mask_embed, // [2304]
    __nv_bfloat16* __restrict__ x_blk,            // [8, 2304]
    int vocab_full) {
    wv_pdl_wait();
    int lane = threadIdx.x, wid = threadIdx.y;
    int tok  = *tok_id_ptr;
    if (tok < 0 || tok >= vocab_full) tok = 0;

    const uint8_t* row_p = emb_wire + static_cast<size_t>(tok) * (18 * 70);

    __shared__ float s_wsum[6];
    float reg_e[3][4];
    float p_sq = 0.0f;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int b128   = wid + it * 6;
        float ws   = 0.0f;
        int32_t w4 = wv_ink_unpack_wv6_4b38_vec4(row_p, b128, lane, ws);
        float f0   = static_cast<float>(static_cast<int8_t>(w4 & 0xFF)) * ws;
        float f1   = static_cast<float>(static_cast<int8_t>((w4 >> 8) & 0xFF)) * ws;
        float f2   = static_cast<float>(static_cast<int8_t>((w4 >> 16) & 0xFF)) * ws;
        float f3   = static_cast<float>(static_cast<int8_t>((w4 >> 24) & 0xFF)) * ws;
        wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
        f0 = __bfloat162float(__float2bfloat16(f0));
        f1 = __bfloat162float(__float2bfloat16(f1));
        f2 = __bfloat162float(__float2bfloat16(f2));
        f3 = __bfloat162float(__float2bfloat16(f3));
        reg_e[it][0] = f0; reg_e[it][1] = f1; reg_e[it][2] = f2; reg_e[it][3] = f3;
        p_sq += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
    }
    p_sq = wv_warp_sum_f32(p_sq);
    if (lane == 0) s_wsum[wid] = p_sq;
    __syncthreads();
    float inv_rms = rsqrtf((s_wsum[0]+s_wsum[1]+s_wsum[2]+s_wsum[3]+s_wsum[4]+s_wsum[5]) * (1.0f / 2304.0f) + 1e-6f);

    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int base  = (wid + it * 6) * 128 + lane * 4;
        float4 nw = __ldg(reinterpret_cast<const float4*>(embed_norm + base));
        x_blk[base + 0] = __float2bfloat16(reg_e[it][0] * inv_rms * nw.x);
        x_blk[base + 1] = __float2bfloat16(reg_e[it][1] * inv_rms * nw.y);
        x_blk[base + 2] = __float2bfloat16(reg_e[it][2] * inv_rms * nw.z);
        x_blk[base + 3] = __float2bfloat16(reg_e[it][3] * inv_rms * nw.w);
        uint2 mv = *reinterpret_cast<const uint2*>(mask_embed + base);
        #pragma unroll
        for (int s = 1; s < 8; ++s) {
            *reinterpret_cast<uint2*>(x_blk + static_cast<size_t>(s) * INK2_HIDDEN + base) = mv;
        }
    }
    wv_pdl_trigger();
}

// Fused MTP Residual Accumulate + RMSNorm + BF16 + FWHT-128 + Q8_32: Grid <<<M, dim3(32, 6)>>>
__global__ void wv_ink_mtp_res_accum_rmsnorm_fwht_q8_kernel(
    const __nv_bfloat16* __restrict__ x_bf16,
    __nv_bfloat16* __restrict__ res_bf16,
    const float* __restrict__ w_norm,
    __nv_bfloat16* __restrict__ y_bf16,
    int8_t* __restrict__ Y_q8,
    float* __restrict__ Y_sc,
    int is_first,
    int M) {
    wv_pdl_wait();
    int m = blockIdx.x;
    if (m >= M) { wv_pdl_trigger(); return; }
    int lane = threadIdx.x, wid = threadIdx.y;

    const __nv_bfloat16* xm = x_bf16   + static_cast<size_t>(m) * INK2_HIDDEN;
    __nv_bfloat16*       rm = res_bf16 + static_cast<size_t>(m) * INK2_HIDDEN;
    __shared__ float s_wsum[6];
    float reg_s[3][4];
    float p_sq = 0.0f;

    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int base = (wid + it * 6) * 128 + lane * 4;
        float x0 = __bfloat162float(xm[base + 0]), x1 = __bfloat162float(xm[base + 1]);
        float x2 = __bfloat162float(xm[base + 2]), x3 = __bfloat162float(xm[base + 3]);
        float s0 = is_first ? x0 : (x0 + __bfloat162float(rm[base + 0]));
        float s1 = is_first ? x1 : (x1 + __bfloat162float(rm[base + 1]));
        float s2 = is_first ? x2 : (x2 + __bfloat162float(rm[base + 2]));
        float s3 = is_first ? x3 : (x3 + __bfloat162float(rm[base + 3]));
        rm[base + 0] = __float2bfloat16(s0); rm[base + 1] = __float2bfloat16(s1);
        rm[base + 2] = __float2bfloat16(s2); rm[base + 3] = __float2bfloat16(s3);
        reg_s[it][0] = s0; reg_s[it][1] = s1; reg_s[it][2] = s2; reg_s[it][3] = s3;
        p_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
    }
    p_sq = wv_warp_sum_f32(p_sq);
    if (lane == 0) s_wsum[wid] = p_sq;
    __syncthreads();
    float inv_rms = rsqrtf((s_wsum[0]+s_wsum[1]+s_wsum[2]+s_wsum[3]+s_wsum[4]+s_wsum[5]) * (1.0f / 2304.0f) + 1e-6f);

    __nv_bfloat16* yb = (y_bf16 != nullptr) ? (y_bf16 + static_cast<size_t>(m) * INK2_HIDDEN) : nullptr;
    int8_t* q8_m      = Y_q8 + static_cast<size_t>(m) * INK2_HIDDEN;
    float*  sc_m      = Y_sc + static_cast<size_t>(m) * 72;
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        int b128 = wid + it * 6;
        int base = b128 * 128 + lane * 4;
        float4 nw = __ldg(reinterpret_cast<const float4*>(w_norm + base));
        __nv_bfloat16 b0 = __float2bfloat16(reg_s[it][0] * inv_rms * nw.x);
        __nv_bfloat16 b1 = __float2bfloat16(reg_s[it][1] * inv_rms * nw.y);
        __nv_bfloat16 b2 = __float2bfloat16(reg_s[it][2] * inv_rms * nw.z);
        __nv_bfloat16 b3 = __float2bfloat16(reg_s[it][3] * inv_rms * nw.w);
        if (yb != nullptr) { yb[base + 0] = b0; yb[base + 1] = b1; yb[base + 2] = b2; yb[base + 3] = b3; }
        float f0 = __bfloat162float(b0), f1 = __bfloat162float(b1);
        float f2 = __bfloat162float(b2), f3 = __bfloat162float(b3);
        wv_fwht128_warp_inplace(f0, f1, f2, f3, lane);
        wv_quant_q8_32_warp(f0, f1, f2, f3, lane, q8_m + b128 * 128, sc_m + b128 * 4);
    }
    wv_pdl_trigger();
}

// Fused MTP k_ctx [1024] + v_ctx [1024] (128 CTAs x 64, 2 warps/CTA, kc512) for M <= 8:
__global__ __launch_bounds__(64, 8)
void wv_ink_swz8_fused_mtp_kv_ctx_kernel(
    WvInkSwz8Linear wk_ctx, WvInkSwz8Linear wv_ctx,
    const int8_t* __restrict__ X_q8, const float* __restrict__ X_sc,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int M) {
    __shared__ uint32_t s_X[8][132];
    wv_pdl_wait();
    if (M <= 0) { wv_pdl_trigger(); return; }
    int cta     = blockIdx.x; // 0..127 (64 k_ctx + 64 v_ctx)
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int lane_n  = lane >> 2;
    int c_k     = lane & 3;
    int r0      = lane >> 2;
    int col_n0  = c_k * 2, col_n1 = col_n0 + 1;
    bool is_k   = (cta < 64);
    int n_tile8 = (is_k ? cta : (cta - 64)) * 2 + warp_id;
    const uint2*   w_pk = is_k ? wk_ctx.pk : wv_ctx.pk;
    const __half*  w_sc = is_k ? wk_ctx.sc : wv_ctx.sc;
    __nv_bfloat16* dst  = is_k ? k_out     : v_out;

    int out_n0 = n_tile8 * 8 + col_n0;
    constexpr int n_kc32    = INK2_HIDDEN >> 5;
    constexpr float kInv127 = 1.0f / 127.0f;
    const uint2*  pk_base   = w_pk + static_cast<size_t>(n_tile8) * n_kc32 * 32 + lane;
    const __half* sc_base   = w_sc + static_cast<size_t>(n_tile8) * n_kc32 * 8 + lane_n;

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int kc512 = 0; kc512 < 18; kc512 += 4) {
        int rem_c = min(4, 18 - kc512);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int idx   = u * 64 + threadIdx.x;
            int ld_m  = idx >> 5;
            int ld_q  = idx & 31;
            bool ok_k = (ld_q < rem_c * 8);
            uint4 xv  = (ld_m < M && ok_k)
                ? reinterpret_cast<const uint4*>(X_q8 + static_cast<size_t>(ld_m) * INK2_HIDDEN + static_cast<size_t>(kc512) * 128)[ld_q]
                : make_uint4(0, 0, 0, 0);
            int c4 = ld_q * 4;
            s_X[ld_m][c4 + 0] = xv.x; s_X[ld_m][c4 + 1] = xv.y;
            s_X[ld_m][c4 + 2] = xv.z; s_X[ld_m][c4 + 3] = xv.w;
        }
        __syncthreads();
        for (int c = 0; c < rem_c; ++c) {
            int kc128 = kc512 + c;
            #pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                int kc32 = kc128 * 4 + sub, s_off = (c * 4 + sub) * 8 + c_k;
                uint2 pkv = __ldg(pk_base + kc32 * 32);
                float ws_self = __half2float(__ldg(sc_base + kc32 * 8)) * kInv127;
                float ws0 = __shfl_sync(0xffffffffu, ws_self, col_n0 * 4);
                float ws1 = __shfl_sync(0xffffffffu, ws_self, col_n1 * 4);
                float xs0 = (r0 < M) ? X_sc[static_cast<size_t>(r0) * n_kc32 + kc32] : 0.0f;
                int d0, d1, d2, d3;
                wv_mma_m16n8k32_s8(d0, d1, d2, d3, s_X[r0][s_off], 0u, s_X[r0][s_off + 4], 0u, pkv.x, pkv.y);
                acc0 += static_cast<float>(d0) * (ws0 * xs0);
                acc1 += static_cast<float>(d1) * (ws1 * xs0);
            }
        }
        __syncthreads();
    }
    if (r0 < M) {
        size_t row_off = static_cast<size_t>(r0) * INK2_KV_DIM;
        dst[row_off + out_n0]     = __float2bfloat16(acc0);
        dst[row_off + out_n0 + 1] = __float2bfloat16(acc1);
    }
    wv_pdl_trigger();
}

// Fused MTP Combine + Softcap30 + Greedy Argmax Stage 1: Grid <<<64, 256>>>
__global__ void wv_ink_mtp_combine_softcap_argmax_part_kernel(
    const float* __restrict__ base_logits_t,
    const float* __restrict__ bias_logits_t,
    float* __restrict__ part_v,
    int32_t* __restrict__ part_i,
    int valid_vocab) {
    wv_pdl_wait();
    int blk = blockIdx.x;
    int tid = threadIdx.x;
    float best_v = -1e30f;
    int   best_i = 0;
    for (int i = blk * 256 + tid; i < valid_vocab; i += 64 * 256) {
        float s = __bfloat162float(__float2bfloat16(base_logits_t[i] + bias_logits_t[i]));
        float v = 30.0f * tanhf(s * (1.0f / 30.0f));
        if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
    }
    int lane = tid & 31, wid = tid >> 5;
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best_v, mask);
        int   oi = __shfl_xor_sync(0xffffffffu, best_i, mask);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    __shared__ float s_v[8];
    __shared__ int   s_i[8];
    if (lane == 0) { s_v[wid] = best_v; s_i[wid] = best_i; }
    __syncthreads();
    if (tid == 0) {
        float fv = s_v[0]; int fi = s_i[0];
        for (int w = 1; w < 8; ++w) {
            if (s_v[w] > fv || (s_v[w] == fv && s_i[w] < fi)) { fv = s_v[w]; fi = s_i[w]; }
        }
        part_v[blk] = fv;
        part_i[blk] = fi;
    }
    wv_pdl_trigger();
}

// Single-Launch Fused Markov mtp_fold ([V, 256]) + Combine + Softcap30 + Greedy Argmax Stage 1: Grid <<<64, 256>>>
__global__ __launch_bounds__(128, 6)
void wv_ink_mtp_fold_combine_argmax_part_kernel(
    WvInkSwz8Linear w_fold,
    const float* __restrict__ fold_bias,
    const int8_t* __restrict__ lat_q8,
    const float* __restrict__ lat_sc,
    const float* __restrict__ base_logits_t,
    float* __restrict__ part_v,
    int32_t* __restrict__ part_i,
    int valid_vocab) {
    wv_pdl_wait();
    int blk   = blockIdx.x;  // 0..255
    int tid   = threadIdx.x; // 0..127
    int wid   = tid >> 5;    // 0..3
    int lane  = tid & 31;    // 0..31
    int col_n = lane >> 2;   // 0..7
    int c_k   = lane & 3;    // 0..3

    int32_t x_lo[8], x_hi[8];
    float   xs[8];
    #pragma unroll
    for (int kc32 = 0; kc32 < 8; ++kc32) {
        const int32_t* xp = reinterpret_cast<const int32_t*>(lat_q8 + kc32 * 32);
        x_lo[kc32] = xp[c_k];
        x_hi[kc32] = xp[4 + c_k];
        xs[kc32]   = lat_sc[kc32];
    }

    constexpr float kInv127 = 1.0f / 127.0f;
    float best_v = -1e30f;
    int   best_i = 0;
    int n_tiles8 = (valid_vocab + 7) >> 3;
    for (int n_tile8 = blk * 4 + wid; n_tile8 < n_tiles8; n_tile8 += 256 * 4) {
        const uint2*  pk_base = w_fold.pk + static_cast<size_t>(n_tile8) * 8 * 32 + lane;
        const __half* sc_base = w_fold.sc + static_cast<size_t>(n_tile8) * 8 * 8 + col_n;
        float acc = 0.0f;
        #pragma unroll
        for (int kc32 = 0; kc32 < 8; ++kc32) {
            uint2 pkv = wv_ld_stream_u64(pk_base + kc32 * 32);
            float ws  = __half2float(__ldg(sc_base + kc32 * 8)) * kInv127;
            int32_t d = __dp4a(x_lo[kc32], static_cast<int32_t>(pkv.x),
                        __dp4a(x_hi[kc32], static_cast<int32_t>(pkv.y), 0));
            d += __shfl_xor_sync(0xffffffffu, d, 2);
            d += __shfl_xor_sync(0xffffffffu, d, 1);
            acc += static_cast<float>(d) * (ws * xs[kc32]);
        }
        int out_col = n_tile8 * 8 + col_n;
        if (c_k == 0 && out_col < valid_vocab) {
            float bv = acc + __ldg(fold_bias + out_col);
            float s  = __bfloat162float(__float2bfloat16(__ldg(base_logits_t + out_col) + bv));
            float v  = 30.0f * tanhf(s * (1.0f / 30.0f));
            if (v > best_v || (v == best_v && out_col < best_i)) {
                best_v = v;
                best_i = out_col;
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best_v, mask);
        int   oi = __shfl_xor_sync(0xffffffffu, best_i, mask);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    __shared__ float s_v[4];
    __shared__ int   s_i[4];
    if (lane == 0) { s_v[wid] = best_v; s_i[wid] = best_i; }
    __syncthreads();
    if (tid == 0) {
        float fv = s_v[0]; int fi = s_i[0];
        for (int w = 1; w < 4; ++w) {
            if (s_v[w] > fv || (s_v[w] == fv && s_i[w] < fi)) { fv = s_v[w]; fi = s_i[w]; }
        }
        part_v[blk] = fv;
        part_i[blk] = fi;
    }
    wv_pdl_trigger();
}

// Fused MTP Combine + Softcap30 + Stochastic Top-p Draft Sampler: Grid <<<1, 512>>>
__global__ void wv_ink_mtp_combine_softcap_bias_sample_topp_kernel(
    const float* __restrict__ base_logits_t,
    const float* __restrict__ bias_logits_t,
    float* __restrict__ q_probs_t,
    int32_t* __restrict__ out_tok_ptr,
    const WvInkSampleCfg* __restrict__ cfg,
    int step_t,
    int stride_v,
    int valid_vocab) {
    wv_pdl_wait();
    int tid = threadIdx.x;
    auto lfn = [&](int i) -> float {
        float s = __bfloat162float(__float2bfloat16(base_logits_t[i] + bias_logits_t[i]));
        return 30.0f * tanhf(s * (1.0f / 30.0f));
    };
    WvInkTopPCutoff cut = wv_ink_compute_topp_cutoff_cta512(lfn, valid_vocab, cfg->temp, cfg->top_p);
    for (int i = tid; i < stride_v; i += 512) {
        if (i < valid_vocab) {
            float lv = lfn(i);
            q_probs_t[i] = wv_ink_topp_is_kept(lv, i, cut) ? (expf((lv - cut.m_max) * cut.inv_temp) * cut.inv_kept_Z) : 0.0f;
        } else {
            q_probs_t[i] = 0.0f;
        }
    }
    __syncthreads();
    float u01 = wv_ink_rng_uniform01(cfg->seed, cfg->rng_step, static_cast<uint32_t>(step_t));
    int tok   = wv_ink_sample_cdf_cta512([&](int i) { return q_probs_t[i]; }, valid_vocab, 1.0f, u01, cut.argmax_idx);
    if (tid == 0) *out_tok_ptr = tok;
    wv_pdl_trigger();
}

} // namespace wv

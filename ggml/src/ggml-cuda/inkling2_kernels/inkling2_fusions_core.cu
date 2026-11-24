/* Clean Modular Inkling-2 CUDA Engine Fusions & Weight Unpack Core for RunMyAi TokenWing. */

#ifndef WV_OBFUSCATE_SYMBOLS
#define WV_OBFUSCATE_SYMBOLS
#endif
#include "wv_sym_obf.h"
#include "wv_os_port.h"
#include "wv_common.cuh"
#include "wv_inkling_engine.cuh"
#include "inkling2_core_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#ifdef WV_OBFUSCATE_SYMBOLS
#define wv_ink_twf_unpack_ext3_kernel          s_twf_00
#define wv_ink_twf_f32_to_bf16_kernel          s_twf_01
#define wv_ink_twf_unpack_q8_0_kernel          s_twf_02
#define wv_ink_twf_unpack_hot_gu_kernel        s_twf_03
#define wv_ink_twf_unpack_hot_dn_kernel        s_twf_04
#define wv_ink_twf_unpack_cold_gu_kernel       s_twf_05
#define wv_ink_twf_unpack_cold_dn_kernel       s_twf_06
#define wv_ink_twf_block_attn_res_op_kernel    s_twf_07
#define wv_ink_twf_loop_msg_gate_op_kernel     s_twf_08
#define wv_ink_twf_sconv_static_op_kernel      s_twf_09
#define wv_ink_twf_sconv_dynamic_op_kernel     s_twf_10
#define wv_ink_twf_prep_kv_op_kernel           s_twf_11
#define wv_ink_twf_attn_rel_sink_op_kernel     s_twf_12
#define wv_ink_twf_softcap_mask_head_kernel    s_twf_13
#define wv_ink_twf_gather_q8_rows_kernel       s_twf_14
#define wv_ink_twf_stash_f32_tap_kernel        s_twf_15
#define wv_ink_twf_set_i32_scalar_kernel       s_twf_16
#endif

namespace wv {
std::string& tls_last_error() {
    static thread_local std::string s_err;
    return s_err;
}
} // namespace wv

namespace twf_detail {

using namespace wv;

static constexpr uint32_t kQwv2AuxMagic = 0x32565751u; // "QWV2"

struct Qwv2AuxHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t n_layers;
    uint32_t n_experts;
    uint64_t total_bytes;
    uint64_t cb4096_off;
    uint64_t vq8_uv_off;
    uint64_t lut4_off;
    uint64_t tmap_off;
    uint64_t tails_off;
};

struct Ink2BlkPtrs5 {
    const float* p[5];
};

__device__ __forceinline__ uint16_t ld_u16_le_u8(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

__device__ __forceinline__ uint32_t ld_u32_le_u8(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ uint8_t ld_hot_vq12_wire_byte(
    const uint8_t* __restrict__ ext0_row,
    const uint8_t* __restrict__ aux_blob,
    uint32_t aux_tail_off,
    int in_slot_cap,
    int j) {
    if (j < 68) {
        return ext0_row[4 + j];
    }
    if (j < in_slot_cap) {
        return ext0_row[6 + j];
    }
    return aux_blob[static_cast<size_t>(aux_tail_off) + static_cast<size_t>(j - in_slot_cap)];
}

// ============================================================================
// 1. GPU Weight Unpack Kernels (`EXT3`, `F32->BF16`, `Q8_0`, `EXT0 + QWV2`)
// ============================================================================

// Grid <<<dim3(k32, n_rows8), 32>>>
__global__ __launch_bounds__(32, 8)
void wv_ink_twf_unpack_ext3_kernel(
    const uint8_t* __restrict__ ext3_rows,
    uint2* __restrict__ dst_pk,
    __half* __restrict__ dst_sc,
    int n_rows8,
    int k32) {
    int b32  = blockIdx.x;
    int r8   = blockIdx.y;
    int lane = threadIdx.x; // 0..31
    if (b32 >= k32 || r8 >= n_rows8) return;

    size_t row_stride = static_cast<size_t>(k32) * 34;
    if (lane < 8) {
        const uint8_t* blk = ext3_rows + static_cast<size_t>(r8 * 8 + lane) * row_stride + static_cast<size_t>(b32) * 34;
        uint16_t d_u16 = ld_u16_le_u8(blk);
        dst_sc[(static_cast<size_t>(r8) * k32 + b32) * 8 + lane] = __ushort_as_half(d_u16);
    }

    int a8  = lane >> 2;  // 0..7
    int c_k = lane & 3;   // 0..3
    const uint8_t* qs = ext3_rows + static_cast<size_t>(r8 * 8 + a8) * row_stride + static_cast<size_t>(b32) * 34 + 2;
    uint32_t w0 = ld_u32_le_u8(qs + (0 * 4 + c_k) * 4);
    uint32_t w1 = ld_u32_le_u8(qs + (1 * 4 + c_k) * 4);
    dst_pk[(static_cast<size_t>(r8) * k32 + b32) * 32 + lane] = make_uint2(w0, w1);
}

__global__ __launch_bounds__(256, 4)
void wv_ink_twf_f32_to_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    size_t n_elems) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_elems) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

// Grid <<<v_rows, 256>>> (8 blocks of 32 per 256-dim row)
__global__ __launch_bounds__(256, 4)
void wv_ink_twf_unpack_q8_0_kernel(
    const uint8_t* __restrict__ q8_rows,
    int8_t* __restrict__ dst_i8,
    __half* __restrict__ dst_sc,
    int v_rows) {
    int r = blockIdx.x;
    if (r >= v_rows) return;
    int tid = threadIdx.x; // 0..255
    int b8  = tid >> 5;    // 0..7
    int l32 = tid & 31;    // 0..31
    const uint8_t* blk = q8_rows + (static_cast<size_t>(r) * 8 + b8) * 34;
    if (l32 == 0) {
        dst_sc[static_cast<size_t>(r) * 8 + b8] = __ushort_as_half(ld_u16_le_u8(blk));
    }
    dst_i8[static_cast<size_t>(r) * 256 + tid] = static_cast<int8_t>(blk[2 + l32]);
}

// Unpack Hot VQ-4096 Gate+Up (`[Eh, 288, 18]`): Grid <<<dim3(18, 288, 128), 32>>>
__global__ __launch_bounds__(32, 8)
void wv_ink_twf_unpack_hot_gu_kernel(
    const uint8_t* __restrict__ gate_ext0,
    const uint8_t* __restrict__ up_ext0,
    const uint8_t* __restrict__ aux_blob,
    const int32_t* __restrict__ tmap,
    uint32_t* __restrict__ dst_pk3,
    uint16_t* __restrict__ dst_sc,
    int n_experts) {
    int ei = blockIdx.z;
    if (ei >= n_experts) return;
    int tv = tmap[ei];
    if (tv < 0) return;

    int b128  = blockIdx.x; // 0..17
    int r8    = blockIdx.y; // 0..287
    int lane  = threadIdx.x; // 0..31
    int g144  = r8 >> 1;
    int is_up = r8 & 1;
    const uint8_t* exp_base = (is_up ? up_ext0 : gate_ext0) + static_cast<size_t>(ei) * 746496u;
    constexpr int row_ext0    = 648;
    constexpr int in_slot_cap = 642;

    if (lane < 8) {
        int r_src = g144 * 8 + lane;
        const uint8_t* rp = exp_base + static_cast<size_t>(r_src) * row_ext0;
        uint32_t tail_off = ld_u32_le_u8(rp);
        int jb = b128 * 53;
        uint8_t m0 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, jb + 0);
        uint8_t m1 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, jb + 1);
        uint8_t m2 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, jb + 2);
        uint8_t m3 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, jb + 3);
        uint8_t m4 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, jb + 4);
        uint16_t* sc_out = dst_sc + ((static_cast<size_t>(tv) * 288 + r8) * 18 + b128) * 20;
        sc_out[lane]     = static_cast<uint16_t>(m0) | (static_cast<uint16_t>(m1) << 8);
        sc_out[8 + lane] = static_cast<uint16_t>(m2) | (static_cast<uint16_t>(m3) << 8);
        reinterpret_cast<uint8_t*>(sc_out + 16)[lane] = m4;
    }

    int a8 = lane >> 2;
    int l4 = lane & 3;
    int r_src = g144 * 8 + a8;
    const uint8_t* rp = exp_base + static_cast<size_t>(r_src) * row_ext0;
    uint32_t tail_off = ld_u32_le_u8(rp);
    int q_base = b128 * 53 + 5;
    uint32_t idx8[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int p16 = j * 2 + (l4 >> 1);
        uint8_t b0 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, q_base + 3 * p16 + 0);
        uint8_t b1 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, q_base + 3 * p16 + 1);
        uint8_t b2 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, q_base + 3 * p16 + 2);
        idx8[j] = ((l4 & 1) == 0)
            ? (static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1 & 0x0Fu) << 8))
            : ((static_cast<uint32_t>(b1) >> 4) | (static_cast<uint32_t>(b2) << 4));
    }
    uint32_t u0 = idx8[0] | (idx8[1] << 12) | ((idx8[2] & 0xFFu) << 24);
    uint32_t u1 = (idx8[2] >> 8) | (idx8[3] << 4) | (idx8[4] << 16) | ((idx8[5] & 0x0Fu) << 28);
    uint32_t u2 = (idx8[5] >> 4) | (idx8[6] << 8) | (idx8[7] << 20);
    uint32_t* pk_out = dst_pk3 + ((static_cast<size_t>(tv) * 288 + r8) * 18 + b128) * 96;
    pk_out[0 * 32 + lane] = u0;
    pk_out[1 * 32 + lane] = u1;
    pk_out[2 * 32 + lane] = u2;
}

// Unpack Hot VQ-4096 Down (`[Eh, 288, 9]`): Grid <<<dim3(9, 288, 128), 32>>>
__global__ __launch_bounds__(32, 8)
void wv_ink_twf_unpack_hot_dn_kernel(
    const uint8_t* __restrict__ down_ext0,
    const uint8_t* __restrict__ aux_blob,
    const int32_t* __restrict__ tmap,
    uint32_t* __restrict__ dst_pk3,
    uint16_t* __restrict__ dst_sc,
    int n_experts) {
    int ei = blockIdx.z;
    if (ei >= n_experts) return;
    int tv = tmap[ei];
    if (tv < 0) return;

    int b128 = blockIdx.x; // 0..8
    int r8   = blockIdx.y; // 0..287
    int lane = threadIdx.x; // 0..31
    const uint8_t* exp_base = down_ext0 + static_cast<size_t>(ei) * 746496u;
    constexpr int row_ext0    = 324;
    constexpr int in_slot_cap = 318;

    if (lane < 8) {
        int r_src = r8 * 8 + lane;
        const uint8_t* rp = exp_base + static_cast<size_t>(r_src) * row_ext0;
        uint32_t tail_off = ld_u32_le_u8(rp);
        int jb = b128 * 53;
        uint8_t m0 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, jb + 0);
        uint8_t m1 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, jb + 1);
        uint8_t m2 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, jb + 2);
        uint8_t m3 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, jb + 3);
        uint8_t m4 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, jb + 4);
        uint16_t* sc_out = dst_sc + ((static_cast<size_t>(tv) * 288 + r8) * 9 + b128) * 20;
        sc_out[lane]     = static_cast<uint16_t>(m0) | (static_cast<uint16_t>(m1) << 8);
        sc_out[8 + lane] = static_cast<uint16_t>(m2) | (static_cast<uint16_t>(m3) << 8);
        reinterpret_cast<uint8_t*>(sc_out + 16)[lane] = m4;
    }

    int a8 = lane >> 2;
    int l4 = lane & 3;
    int r_src = r8 * 8 + a8;
    const uint8_t* rp = exp_base + static_cast<size_t>(r_src) * row_ext0;
    uint32_t tail_off = ld_u32_le_u8(rp);
    int q_base = b128 * 53 + 5;
    uint32_t idx8[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int p16 = j * 2 + (l4 >> 1);
        uint8_t b0 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, q_base + 3 * p16 + 0);
        uint8_t b1 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, q_base + 3 * p16 + 1);
        uint8_t b2 = ld_hot_vq12_wire_byte(rp, aux_blob, tail_off, in_slot_cap, q_base + 3 * p16 + 2);
        idx8[j] = ((l4 & 1) == 0)
            ? (static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1 & 0x0Fu) << 8))
            : ((static_cast<uint32_t>(b1) >> 4) | (static_cast<uint32_t>(b2) << 4));
    }
    uint32_t u0 = idx8[0] | (idx8[1] << 12) | ((idx8[2] & 0xFFu) << 24);
    uint32_t u1 = (idx8[2] >> 8) | (idx8[3] << 4) | (idx8[4] << 16) | ((idx8[5] & 0x0Fu) << 28);
    uint32_t u2 = (idx8[5] >> 4) | (idx8[6] << 8) | (idx8[7] << 20);
    uint32_t* pk_out = dst_pk3 + ((static_cast<size_t>(tv) * 288 + r8) * 9 + b128) * 96;
    pk_out[0 * 32 + lane] = u0;
    pk_out[1 * 32 + lane] = u1;
    pk_out[2 * 32 + lane] = u2;
}

// Unpack Cold VQ8 Gate+Up (`[Ec, 2239488]`): Grid <<<dim3(36, 288, 128), 32>>>
__global__ __launch_bounds__(32, 8)
void wv_ink_twf_unpack_cold_gu_kernel(
    const uint8_t* __restrict__ gate_ext0,
    const uint8_t* __restrict__ up_ext0,
    const int32_t* __restrict__ tmap,
    uint8_t* __restrict__ dst_cold_slabs,
    int n_experts) {
    int ei = blockIdx.z;
    if (ei >= n_experts) return;
    int tv = tmap[ei];
    if (tv >= 0) return;
    int ci = ~tv;

    int b64   = blockIdx.x; // 0..35
    int r8    = blockIdx.y; // 0..287
    int lane  = threadIdx.x; // 0..31
    int g144  = r8 >> 1;
    int is_up = r8 & 1;
    const uint8_t* exp_base = (is_up ? up_ext0 : gate_ext0) + static_cast<size_t>(ei) * 746496u;
    uint8_t* slab = dst_cold_slabs + static_cast<size_t>(ci) * INK2_COLD_SLAB_BYTES;

    if (lane < 8) {
        const uint8_t* blk = exp_base + static_cast<size_t>(g144 * 8 + lane) * 648u + static_cast<size_t>(b64) * 18u;
        uint16_t* sc_out = reinterpret_cast<uint16_t*>(slab + INK2_OFF_GU_SC);
        sc_out[(static_cast<size_t>(r8) * 36 + b64) * 8 + lane] = ld_u16_le_u8(blk);
    }

    int a8  = lane >> 2;
    int b_2 = (lane >> 1) & 1;
    int c_2 = lane & 1;
    const uint8_t* qs = exp_base + static_cast<size_t>(g144 * 8 + a8) * 648u + static_cast<size_t>(b64) * 18u + 2;
    uint16_t lo16 = ld_u16_le_u8(qs + (c_2 * 4 + 0 * 2 + b_2) * 2);
    uint16_t hi16 = ld_u16_le_u8(qs + (c_2 * 4 + 1 * 2 + b_2) * 2);
    uint32_t* pk_out = reinterpret_cast<uint32_t*>(slab + INK2_OFF_GU_PK);
    pk_out[(static_cast<size_t>(r8) * 36 + b64) * 32 + lane] = static_cast<uint32_t>(lo16) | (static_cast<uint32_t>(hi16) << 16);
}

// Unpack Cold VQ8 Down (`[Ec, 2239488]`): Grid <<<dim3(18, 288, 128), 32>>>
__global__ __launch_bounds__(32, 8)
void wv_ink_twf_unpack_cold_dn_kernel(
    const uint8_t* __restrict__ down_ext0,
    const int32_t* __restrict__ tmap,
    uint8_t* __restrict__ dst_cold_slabs,
    int n_experts) {
    int ei = blockIdx.z;
    if (ei >= n_experts) return;
    int tv = tmap[ei];
    if (tv >= 0) return;
    int ci = ~tv;

    int b64  = blockIdx.x; // 0..17
    int r8   = blockIdx.y; // 0..287
    int lane = threadIdx.x; // 0..31
    const uint8_t* exp_base = down_ext0 + static_cast<size_t>(ei) * 746496u;
    uint8_t* slab = dst_cold_slabs + static_cast<size_t>(ci) * INK2_COLD_SLAB_BYTES;

    if (lane < 8) {
        const uint8_t* blk = exp_base + static_cast<size_t>(r8 * 8 + lane) * 324u + static_cast<size_t>(b64) * 18u;
        uint16_t* sc_out = reinterpret_cast<uint16_t*>(slab + INK2_OFF_DN_SC);
        sc_out[(static_cast<size_t>(r8) * 18 + b64) * 8 + lane] = ld_u16_le_u8(blk);
    }

    int a8  = lane >> 2;
    int b_2 = (lane >> 1) & 1;
    int c_2 = lane & 1;
    const uint8_t* qs = exp_base + static_cast<size_t>(r8 * 8 + a8) * 324u + static_cast<size_t>(b64) * 18u + 2;
    uint16_t lo16 = ld_u16_le_u8(qs + (c_2 * 4 + 0 * 2 + b_2) * 2);
    uint16_t hi16 = ld_u16_le_u8(qs + (c_2 * 4 + 1 * 2 + b_2) * 2);
    uint32_t* pk_out = reinterpret_cast<uint32_t*>(slab + INK2_OFF_DN_PK);
    pk_out[(static_cast<size_t>(r8) * 18 + b64) * 32 + lane] = static_cast<uint32_t>(lo16) | (static_cast<uint32_t>(hi16) << 16);
}

// ============================================================================
// 2. Modular CUDA Micro-Ops (`Phase A` + `--moe-stream` SSD Tiering Path)
// ============================================================================

// Grid <<<M, dim3(32, 6)>>>
__global__ __launch_bounds__(192, 2)
void wv_ink_twf_block_attn_res_op_kernel(
    const float* __restrict__ cur_h,
    const float* __restrict__ query_f32,
    Ink2BlkPtrs5 blks,
    float* __restrict__ out_f32,
    int n_blk,
    int Q,
    int round_cur_bf16,
    int M) {
    int m    = blockIdx.x;
    int lane = threadIdx.x;
    int wid  = threadIdx.y; // 0..5
    int n_src = n_blk + 1;
    bool active = (m < M);

    __shared__ float s_wsum[6][6];
    __shared__ float s_dot[4][6][6];
    __shared__ float s_prob[4][6];

    float vec[6][3][4];
    #pragma unroll
    for (int s = 0; s < 6; ++s) {
        float sq = 0.0f;
        if (active && s < n_src) {
            const float* src_s = (s < n_blk)
                ? ((s == 0) ? blks.p[0] : ((s == 1) ? blks.p[1] : ((s == 2) ? blks.p[2] : ((s == 3) ? blks.p[3] : blks.p[4]))))
                : cur_h;
            bool do_bf16 = (s < n_blk) || (round_cur_bf16 != 0);
            #pragma unroll
            for (int it = 0; it < 3; ++it) {
                int base = (wid + it * 6) * 128 + lane * 4;
                float4 xv = reinterpret_cast<const float4*>(src_s + static_cast<size_t>(m) * INK2_HIDDEN + base)[0];
                float x0 = do_bf16 ? __bfloat162float(__float2bfloat16(xv.x)) : xv.x;
                float x1 = do_bf16 ? __bfloat162float(__float2bfloat16(xv.y)) : xv.y;
                float x2 = do_bf16 ? __bfloat162float(__float2bfloat16(xv.z)) : xv.z;
                float x3 = do_bf16 ? __bfloat162float(__float2bfloat16(xv.w)) : xv.w;
                vec[s][it][0] = x0; vec[s][it][1] = x1; vec[s][it][2] = x2; vec[s][it][3] = x3;
                sq += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
            }
        }
        sq = wv_warp_sum_f32(sq);
        if (lane == 0) s_wsum[s][wid] = sq;
    }
    __syncthreads();

    if (active) {
        #pragma unroll
        for (int s = 0; s < 6; ++s) {
            if (s < n_src) {
                float tot = s_wsum[s][0] + s_wsum[s][1] + s_wsum[s][2] + s_wsum[s][3] + s_wsum[s][4] + s_wsum[s][5];
                float inv = rsqrtf(tot * (1.0f / 2304.0f) + 1e-6f);
                #pragma unroll
                for (int it = 0; it < 3; ++it) {
                    vec[s][it][0] *= inv; vec[s][it][1] *= inv; vec[s][it][2] *= inv; vec[s][it][3] *= inv;
                }
            }
        }
    }

    constexpr float kInvSqrtD = 1.0f / 48.0f;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        float qv[3][4];
        if (active && q < Q) {
            const float* qp = query_f32 + static_cast<size_t>(q) * INK2_HIDDEN;
            #pragma unroll
            for (int it = 0; it < 3; ++it) {
                int base = (wid + it * 6) * 128 + lane * 4;
                float4 ql = reinterpret_cast<const float4*>(qp + base)[0];
                qv[it][0] = ql.x * kInvSqrtD; qv[it][1] = ql.y * kInvSqrtD;
                qv[it][2] = ql.z * kInvSqrtD; qv[it][3] = ql.w * kInvSqrtD;
            }
        }
        #pragma unroll
        for (int s = 0; s < 6; ++s) {
            float d = 0.0f;
            if (active && q < Q && s < n_src) {
                #pragma unroll
                for (int it = 0; it < 3; ++it) {
                    d += qv[it][0] * vec[s][it][0] + qv[it][1] * vec[s][it][1]
                       + qv[it][2] * vec[s][it][2] + qv[it][3] * vec[s][it][3];
                }
            }
            d = wv_warp_sum_f32(d);
            if (lane == 0) s_dot[q][s][wid] = d;
        }
    }
    __syncthreads();

    if (active && wid == 0 && lane == 0) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (q < Q) {
                float mx = -1e30f;
                float lg[6];
                #pragma unroll
                for (int s = 0; s < 6; ++s) {
                    if (s < n_src) {
                        lg[s] = s_dot[q][s][0] + s_dot[q][s][1] + s_dot[q][s][2] + s_dot[q][s][3] + s_dot[q][s][4] + s_dot[q][s][5];
                        if (lg[s] > mx) mx = lg[s];
                    }
                }
                float sm = 0.0f;
                #pragma unroll
                for (int s = 0; s < 6; ++s) {
                    if (s < n_src) {
                        lg[s] = expf(lg[s] - mx);
                        sm += lg[s];
                    }
                }
                float inv_s = 1.0f / fmaxf(sm, 1e-20f);
                #pragma unroll
                for (int s = 0; s < 6; ++s) {
                    if (s < n_src) s_prob[q][s] = lg[s] * inv_s;
                }
            }
        }
    }
    __syncthreads();

    if (active) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (q < Q) {
                float* dst = out_f32 + (static_cast<size_t>(q) * M + m) * INK2_HIDDEN;
                #pragma unroll
                for (int it = 0; it < 3; ++it) {
                    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
                    #pragma unroll
                    for (int s = 0; s < 6; ++s) {
                        if (s < n_src) {
                            float p = s_prob[q][s];
                            o0 += p * vec[s][it][0]; o1 += p * vec[s][it][1];
                            o2 += p * vec[s][it][2]; o3 += p * vec[s][it][3];
                        }
                    }
                    int base = (wid + it * 6) * 128 + lane * 4;
                    reinterpret_cast<float4*>(dst + base)[0] = make_float4(o0, o1, o2, o3);
                }
            }
        }
    }
}

// Single-CTA Fused LoopMessageGate (`0` global workspace!): Grid <<<M, 256>>>
__global__ __launch_bounds__(256, 2)
void wv_ink_twf_loop_msg_gate_op_kernel(
    const float* __restrict__ src_in,
    const float* __restrict__ msg_in,
    const __nv_bfloat16* __restrict__ ctrl_down_bf16,
    const float* __restrict__ ctrl_forget_f32,
    const float* __restrict__ ctrl_gain_f32,
    float* __restrict__ dst_out,
    int M) {
    int m    = blockIdx.x;
    int tid  = threadIdx.x; // 0..255
    int wid  = tid >> 5;    // 0..7
    int lane = tid & 31;    // 0..31
    bool active = (m < M);

    __shared__ __align__(16) __nv_bfloat16 s_cat[4608];
    __shared__ float s_dsum[8], s_msum[8];
    __shared__ float s_h64[64];

    const float* sp = active ? (src_in + static_cast<size_t>(m) * INK2_HIDDEN) : src_in;
    const float* mp = active ? (msg_in + static_cast<size_t>(m) * INK2_HIDDEN) : msg_in;
    float d_sq = 0.0f, m_sq = 0.0f;
    if (active) {
        for (int i = tid; i < INK2_HIDDEN; i += 256) {
            float dv = sp[i], mv = mp[i];
            d_sq += dv * dv;
            m_sq += mv * mv;
        }
    }
    d_sq = wv_warp_sum_f32(d_sq);
    m_sq = wv_warp_sum_f32(m_sq);
    if (lane == 0) {
        s_dsum[wid] = d_sq;
        s_msum[wid] = m_sq;
    }
    __syncthreads();

    if (active) {
        float inv_d = rsqrtf((s_dsum[0]+s_dsum[1]+s_dsum[2]+s_dsum[3]+s_dsum[4]+s_dsum[5]+s_dsum[6]+s_dsum[7]) * (1.0f / 2304.0f) + 1e-6f);
        float inv_m = rsqrtf((s_msum[0]+s_msum[1]+s_msum[2]+s_msum[3]+s_msum[4]+s_msum[5]+s_msum[6]+s_msum[7]) * (1.0f / 2304.0f) + 1e-6f);
        for (int i = tid; i < INK2_HIDDEN; i += 256) {
            s_cat[i]               = __float2bfloat16(sp[i] * inv_d);
            s_cat[INK2_HIDDEN + i] = __float2bfloat16(mp[i] * inv_m);
        }
    }
    __syncthreads();

    // 8 warps compute 8 rows each -> 64 rows of ctrl_down @ s_cat
    #pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
        int r = wid * 8 + sub;
        float acc = 0.0f;
        if (active) {
            const __nv_bfloat16* wr = ctrl_down_bf16 + static_cast<size_t>(r) * 4608;
            for (int k = lane * 4; k < 4608; k += 128) {
                uint2 wv_tmp = __ldg(reinterpret_cast<const uint2*>(wr + k));
                uint2 cv     = *reinterpret_cast<const uint2*>(s_cat + k);
                const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv_tmp);
                const __nv_bfloat162* cb = reinterpret_cast<const __nv_bfloat162*>(&cv);
                float2 w0 = __bfloat1622float2(wb[0]), w1 = __bfloat1622float2(wb[1]);
                float2 c0 = __bfloat1622float2(cb[0]), c1 = __bfloat1622float2(cb[1]);
                acc += w0.x * c0.x + w0.y * c0.y + w1.x * c1.x + w1.y * c1.y;
            }
        }
        acc = wv_warp_sum_f32(acc);
        if (lane == 0) s_h64[r] = acc;
    }
    __syncthreads();

    // 16 groups x 16 sub_lanes -> 9 iterations of 16 channels = 2304 channels
    int grp      = tid >> 4;  // 0..15
    int sub_lane = tid & 15;  // 0..15
    int k4       = sub_lane * 4;
    float4 hv    = *reinterpret_cast<const float4*>(s_h64 + k4);
    for (int blk16 = 0; blk16 < 144; ++blk16) {
        int ch = blk16 * 16 + grp;
        float f_acc = 0.0f, g_acc = 0.0f;
        if (active) {
            float4 fw = __ldg(reinterpret_cast<const float4*>(ctrl_forget_f32 + static_cast<size_t>(ch) * 64 + k4));
            float4 gw = __ldg(reinterpret_cast<const float4*>(ctrl_gain_f32   + static_cast<size_t>(ch) * 64 + k4));
            // Match BF16 rounding of forget/gain weights in WvInkEngine:
            float f0 = __bfloat162float(__float2bfloat16(fw.x)), f1 = __bfloat162float(__float2bfloat16(fw.y));
            float f2 = __bfloat162float(__float2bfloat16(fw.z)), f3 = __bfloat162float(__float2bfloat16(fw.w));
            float g0 = __bfloat162float(__float2bfloat16(gw.x)), g1 = __bfloat162float(__float2bfloat16(gw.y));
            float g2 = __bfloat162float(__float2bfloat16(gw.z)), g3 = __bfloat162float(__float2bfloat16(gw.w));
            f_acc = f0 * hv.x + f1 * hv.y + f2 * hv.z + f3 * hv.w;
            g_acc = g0 * hv.x + g1 * hv.y + g2 * hv.z + g3 * hv.w;
        }
        #pragma unroll
        for (int mask = 8; mask >= 1; mask >>= 1) {
            f_acc += __shfl_xor_sync(0xffffffffu, f_acc, mask);
            g_acc += __shfl_xor_sync(0xffffffffu, g_acc, mask);
        }
        if (active && sub_lane == 0) {
            dst_out[static_cast<size_t>(m) * INK2_HIDDEN + ch] = (1.0f - tanhf(f_acc)) * sp[ch] + tanhf(g_acc) * mp[ch];
        }
    }
}

// Static ShortConvolution (`C = 1024` or `2304`): Grid <<<dim3((C + 255) / 256, B), 256>>>
__global__ __launch_bounds__(256, 4)
void wv_ink_twf_sconv_static_op_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ state_3_c_b,
    const float* __restrict__ w4_c,
    float* __restrict__ y_out,
    int C, int S, int B) {
    int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int b = blockIdx.y;
    if (c >= C || b >= B || S <= 0) return;

    float4 w4 = reinterpret_cast<const float4*>(w4_c + static_cast<size_t>(c) * 4)[0];
    const float* st = state_3_c_b + (static_cast<size_t>(b) * C + c) * 3;
    float h0 = st[0], h1 = st[1], h2 = st[2];
    constexpr float kInvSqrt5 = 0.44721359549995793928f;
    const float* xb = x_in  + static_cast<size_t>(b) * S * C;
    float*       yb = y_out + static_cast<size_t>(b) * S * C;

    for (int s = 0; s < S; ++s) {
        float x_cur = __bfloat162float(__float2bfloat16(xb[static_cast<size_t>(s) * C + c]));
        float conv  = w4.x * h0 + w4.y * h1 + w4.z * h2 + w4.w * x_cur;
        yb[static_cast<size_t>(s) * C + c] = __bfloat162float(__float2bfloat16((conv + x_cur) * kInvSqrt5));
        h0 = h1; h1 = h2; h2 = x_cur;
    }
}

// Dynamic ShortConvolution (`C = 1024`): Grid <<<dim3(4, B), 256>>>
__global__ __launch_bounds__(256, 4)
void wv_ink_twf_sconv_dynamic_op_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ state_3_c_b,
    const float* __restrict__ static_w_c_4,
    const float* __restrict__ dw_32_s_b,
    float* __restrict__ y_out,
    int S, int B) {
    int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int b = blockIdx.y;
    if (c >= 1024 || b >= B || S <= 0) return;

    int h = c >> 7; // 0..7
    float sw0 = static_w_c_4[0 * 1024 + c];
    float sw1 = static_w_c_4[1 * 1024 + c];
    float sw2 = static_w_c_4[2 * 1024 + c];
    float sw3 = static_w_c_4[3 * 1024 + c];

    const float* st = state_3_c_b + (static_cast<size_t>(b) * 1024 + c) * 3;
    float lag3 = st[0], lag2 = st[1], lag1 = st[2];
    const float* xb = x_in  + static_cast<size_t>(b) * S * 1024;
    float*       yb = y_out + static_cast<size_t>(b) * S * 1024;

    for (int s = 0; s < S; ++s) {
        float x_cur = __bfloat162float(__float2bfloat16(xb[static_cast<size_t>(s) * 1024 + c]));
        float4 dw   = reinterpret_cast<const float4*>(dw_32_s_b + static_cast<size_t>(b * S + s) * 32 + h * 4)[0];
        float y     = (sw0 + dw.x) * x_cur + (sw1 + dw.y) * lag1 + (sw2 + dw.z) * lag2 + (sw3 + dw.w) * lag3;
        yb[static_cast<size_t>(s) * 1024 + c] = __bfloat162float(__float2bfloat16((y + x_cur) * (1.0f / 3.0f)));
        lag3 = lag2; lag2 = lag1; lag1 = x_cur;
    }
}

// K L2-clip (`is_v == 0`) or V head-RMSNorm (`is_v == 1`) with BF16 round: Grid <<<M, dim3(32, 8)>>>
__global__ __launch_bounds__(256, 4)
void wv_ink_twf_prep_kv_op_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ y_out,
    int is_v,
    int M) {
    int m = blockIdx.x;
    if (m >= M) return;
    int lane = threadIdx.x; // 0..31
    int wid  = threadIdx.y; // 0..7
    int off  = wid * 128 + lane * 4;
    float4 xv = reinterpret_cast<const float4*>(x_in + static_cast<size_t>(m) * 1024 + off)[0];
    float f0 = __bfloat162float(__float2bfloat16(xv.x));
    float f1 = __bfloat162float(__float2bfloat16(xv.y));
    float f2 = __bfloat162float(__float2bfloat16(xv.z));
    float f3 = __bfloat162float(__float2bfloat16(xv.w));
    float sq = wv_warp_sum_f32(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3);
    constexpr float kClipQK = 22.627416997969522f;
    float scale = (is_v != 0)
        ? rsqrtf(sq * (1.0f / 128.0f) + 1e-6f)
        : fminf(1.0f, kClipQK / (sqrtf(sq) + 1e-6f));
    float o0 = __bfloat162float(__float2bfloat16(f0 * scale));
    float o1 = __bfloat162float(__float2bfloat16(f1 * scale));
    float o2 = __bfloat162float(__float2bfloat16(f2 * scale));
    float o3 = __bfloat162float(__float2bfloat16(f3 * scale));
    reinterpret_cast<float4*>(y_out + static_cast<size_t>(m) * 1024 + off)[0] = make_float4(o0, o1, o2, o3);
}

__device__ __forceinline__ float4 ld_kv_vec4_f32(
    const uint8_t* base_bytes, int kv_type, int lane) {
    int off = lane * 4;
    if (kv_type == 30) { // GGML_TYPE_BF16
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(base_bytes) + off;
        return make_float4(__bfloat162float(p[0]), __bfloat162float(p[1]),
                           __bfloat162float(p[2]), __bfloat162float(p[3]));
    }
    if (kv_type == 1) { // GGML_TYPE_F16
        const __half* p = reinterpret_cast<const __half*>(base_bytes) + off;
        return make_float4(__half2float(p[0]), __half2float(p[1]),
                           __half2float(p[2]), __half2float(p[3]));
    }
    return reinterpret_cast<const float4*>(reinterpret_cast<const float*>(base_bytes) + off)[0];
}

// Relative-Projected + Dynamic-Sink + Tau + Gate GQA-3 Attention over `llama_kv_cache_iswa`:
// Grid <<<M * 24, dim3(32, 4)>>>
__global__ __launch_bounds__(128, 4)
void wv_ink_twf_attn_rel_sink_op_kernel(
    const float* __restrict__ q_raw,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    int kv_type,
    int n_kv,
    int n_stream,
    size_t nb1_k, size_t nb2_k, size_t nb3_k,
    size_t nb1_v, size_t nb2_v, size_t nb3_v,
    const float* __restrict__ r_raw,
    const float* __restrict__ q_clip_scale,
    const float* __restrict__ r_clip_scale,
    const float* __restrict__ rel_proj_t,
    const float* __restrict__ sinks_f32,
    const float* __restrict__ gate_raw,
    const int32_t* __restrict__ pos_pack,
    float* __restrict__ dst_out,
    int is_local,
    int M) {
    int mh    = blockIdx.x;
    int m     = mh / 24;
    int hq    = mh - m * 24;
    int hkv   = hq / 3;
    int lane  = threadIdx.x; // 0..31
    int wid   = threadIdx.y; // 0..3
    bool active = (m < M);

    __shared__ float s_r16[16];
    __shared__ float s_m[4], s_l[4], s_o[4][128];

    int q_pos = active ? pos_pack[m] : 0;
    int q_st  = active ? pos_pack[M + m] : 0; // stream index for token m
    const int32_t* k_pos_all = pos_pack + 2 * M;
    float tau = is_local ? 1.0f : (1.0f + 0.1f * logf(fmaxf(1.0f, static_cast<float>(q_pos + 1) * (1.0f / 8192.0f))));
    constexpr float kClipQK = 22.627416997969522f;
    constexpr float kScale  = 0.08838834764831845f; // 128^-0.5

    // Prepare clipped + tau-scaled Q in registers:
    float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;
    if (active) {
        int off_q = hq * 128 + lane * 4;
        float4 qv = reinterpret_cast<const float4*>(q_raw + static_cast<size_t>(m) * 3072 + off_q)[0];
        float4 sc = reinterpret_cast<const float4*>(q_clip_scale + off_q)[0];
        q0 = __bfloat162float(__float2bfloat16(qv.x)) * sc.x;
        q1 = __bfloat162float(__float2bfloat16(qv.y)) * sc.y;
        q2 = __bfloat162float(__float2bfloat16(qv.z)) * sc.z;
        q3 = __bfloat162float(__float2bfloat16(qv.w)) * sc.w;
    }
    float q_sq = wv_warp_sum_f32(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    float q_cf = fminf(1.0f, kClipQK / (sqrtf(q_sq) + 1e-6f));
    q0 = __bfloat162float(__float2bfloat16(q0 * q_cf));
    q1 = __bfloat162float(__float2bfloat16(q1 * q_cf));
    q2 = __bfloat162float(__float2bfloat16(q2 * q_cf));
    q3 = __bfloat162float(__float2bfloat16(q3 * q_cf));
    if (!is_local) {
        q0 = __bfloat162float(__float2bfloat16(q0 * tau));
        q1 = __bfloat162float(__float2bfloat16(q1 * tau));
        q2 = __bfloat162float(__float2bfloat16(q2 * tau));
        q3 = __bfloat162float(__float2bfloat16(q3 * tau));
    }
    q0 *= kScale; q1 *= kScale; q2 *= kScale; q3 *= kScale;

    // Prepare clipped + tau-scaled R [16] in s_r16:
    if (wid == 0) {
        int d = lane & 15;
        float rv = (active && lane < 16)
            ? (__bfloat162float(__float2bfloat16(r_raw[static_cast<size_t>(m) * 384 + hq * 16 + d])) * r_clip_scale[hq * 16 + d])
            : 0.0f;
        float r_sq = rv * rv;
        #pragma unroll
        for (int mask = 8; mask >= 1; mask >>= 1) r_sq += __shfl_xor_sync(0xffffffffu, r_sq, mask);
        float r_cf = fminf(1.0f, 8.0f / (sqrtf(r_sq) + 1e-6f));
        __nv_bfloat16 rb = __float2bfloat16(rv * r_cf);
        if (!is_local) rb = __float2bfloat16(__bfloat162float(rb) * tau);
        if (lane < 16) s_r16[lane] = __bfloat162float(rb);
    }
    __syncthreads();

    float4 r0 = reinterpret_cast<const float4*>(s_r16 + 0)[0];
    float4 r1 = reinterpret_cast<const float4*>(s_r16 + 4)[0];
    float4 r2 = reinterpret_cast<const float4*>(s_r16 + 8)[0];
    float4 r3 = reinterpret_cast<const float4*>(s_r16 + 12)[0];

    float m_max = -1e30f, l_sum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    int st_clamped = (q_st >= 0 && q_st < n_stream) ? q_st : 0;
    const int32_t* k_pos_st = k_pos_all + static_cast<size_t>(st_clamped) * n_kv;
    const uint8_t* k_st_base = k_cache + static_cast<size_t>(st_clamped) * nb3_k + static_cast<size_t>(hkv) * nb1_k;
    const uint8_t* v_st_base = v_cache + static_cast<size_t>(st_clamped) * nb3_v + static_cast<size_t>(hkv) * nb1_v;

    for (int kv_i = wid; kv_i < n_kv; kv_i += 4) {
        int kp = active ? k_pos_st[kv_i] : -1;
        int dist = q_pos - kp;
        bool valid = active && (kp >= 0) && (dist >= 0) && (!is_local || dist < INK2_SWA_WIN);
        float4 kv4 = valid ? ld_kv_vec4_f32(k_st_base + static_cast<size_t>(kv_i) * nb2_k, kv_type, lane)
                           : make_float4(0.f, 0.f, 0.f, 0.f);
        float qk = wv_warp_sum_f32(q0 * kv4.x + q1 * kv4.y + q2 * kv4.z + q3 * kv4.w);
        if (valid) {
            if (dist < 1024) {
                const float4* rp = reinterpret_cast<const float4*>(rel_proj_t + dist * 16);
                float4 p_0 = rp[0], p_1 = rp[1], p_2 = rp[2], p_3 = rp[3];
                qk += r0.x*p_0.x + r0.y*p_0.y + r0.z*p_0.z + r0.w*p_0.w
                    + r1.x*p_1.x + r1.y*p_1.y + r1.z*p_1.z + r1.w*p_1.w
                    + r2.x*p_2.x + r2.y*p_2.y + r2.z*p_2.z + r2.w*p_2.w
                    + r3.x*p_3.x + r3.y*p_3.y + r3.z*p_3.z + r3.w*p_3.w;
            }
            float m_new = fmaxf(m_max, qk);
            float alpha = (m_max > -1e29f) ? expf(m_max - m_new) : 0.0f;
            float wt    = expf(qk - m_new);
            l_sum = l_sum * alpha + wt;
            m_max = m_new;
            float4 vv4 = ld_kv_vec4_f32(v_st_base + static_cast<size_t>(kv_i) * nb2_v, kv_type, lane);
            o0 = o0 * alpha + wt * vv4.x;
            o1 = o1 * alpha + wt * vv4.y;
            o2 = o2 * alpha + wt * vv4.z;
            o3 = o3 * alpha + wt * vv4.w;
        }
    }

    if (lane == 0) { s_m[wid] = m_max; s_l[wid] = l_sum; }
    s_o[wid][lane * 4 + 0] = o0; s_o[wid][lane * 4 + 1] = o1;
    s_o[wid][lane * 4 + 2] = o2; s_o[wid][lane * 4 + 3] = o3;
    __syncthreads();

    if (active && wid == 0) {
        float sink  = sinks_f32[static_cast<size_t>(m) * 24 + hq];
        float m_tot = sink;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (s_l[w] > 0.0f) m_tot = fmaxf(m_tot, s_m[w]);
        }
        float denom = expf(sink - m_tot);
        float fo0 = 0.f, fo1 = 0.f, fo2 = 0.f, fo3 = 0.f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (s_l[w] > 0.0f) {
                float a = expf(s_m[w] - m_tot);
                denom += s_l[w] * a;
                fo0 += s_o[w][lane * 4 + 0] * a;
                fo1 += s_o[w][lane * 4 + 1] * a;
                fo2 += s_o[w][lane * 4 + 2] * a;
                fo3 += s_o[w][lane * 4 + 3] * a;
            }
        }
        float inv_d = 1.0f / fmaxf(denom, 1e-20f);
        float a0 = __bfloat162float(__float2bfloat16(fo0 * inv_d));
        float a1 = __bfloat162float(__float2bfloat16(fo1 * inv_d));
        float a2 = __bfloat162float(__float2bfloat16(fo2 * inv_d));
        float a3 = __bfloat162float(__float2bfloat16(fo3 * inv_d));

        int off_q = hq * 128 + lane * 4;
        float4 gv = reinterpret_cast<const float4*>(gate_raw + static_cast<size_t>(m) * 3072 + off_q)[0];
        auto apply_gate = [](float a, float g_in) -> float {
            float g  = __bfloat162float(__float2bfloat16(g_in));
            float sg = 1.0f / (1.0f + expf(-g));
            return __bfloat162float(__float2bfloat16(a * sg));
        };
        float4 out4 = make_float4(
            apply_gate(a0, gv.x), apply_gate(a1, gv.y),
            apply_gate(a2, gv.z), apply_gate(a3, gv.w));
        reinterpret_cast<float4*>(dst_out + static_cast<size_t>(m) * 3072 + off_q)[0] = out4;
    }
}

// Grid <<<dim3((out_vocab_full + 255) / 256, n_rows), 256>>>
__global__ __launch_bounds__(256, 4)
void wv_ink_twf_softcap_mask_head_kernel(
    const float* __restrict__ src_logits,
    float* __restrict__ dst_logits,
    int in_stride_vocab,
    int out_vocab_full,
    int vocab_unpad,
    float softcap,
    int apply_softcap,
    int n_rows) {
    int v = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int r = blockIdx.y;
    if (v >= out_vocab_full || r >= n_rows) return;
    float out_v;
    if (v < vocab_unpad && v < in_stride_vocab) {
        float x = src_logits[static_cast<size_t>(r) * in_stride_vocab + v];
        out_v = (apply_softcap != 0 && softcap > 0.0f) ? (softcap * tanhf(x / softcap)) : x;
    } else {
        out_v = -__int_as_float(0x7f800000u); // -INFINITY
    }
    dst_logits[static_cast<size_t>(r) * out_vocab_full + v] = out_v;
}

// Gather selected rows of `(q8 [M, 2304], sc [M, 72])`: Grid <<<n_out, 256>>>
__global__ __launch_bounds__(256, 4)
void wv_ink_twf_gather_q8_rows_kernel(
    const int8_t* __restrict__ src_q8,
    const float* __restrict__ src_sc,
    const int32_t* __restrict__ out_ids,
    int8_t* __restrict__ dst_q8,
    float* __restrict__ dst_sc,
    int M, int n_out) {
    int o = blockIdx.x;
    if (o >= n_out) return;
    int m = out_ids[o];
    if (m < 0 || m >= M) m = 0;
    int tid = threadIdx.x;
    const uint4* sq = reinterpret_cast<const uint4*>(src_q8 + static_cast<size_t>(m) * INK2_HIDDEN);
    uint4*       dq = reinterpret_cast<uint4*>(dst_q8 + static_cast<size_t>(o) * INK2_HIDDEN);
    for (int i = tid; i < (INK2_HIDDEN / 16); i += 256) {
        dq[i] = sq[i];
    }
    if (tid < 72) {
        dst_sc[static_cast<size_t>(o) * 72 + tid] = src_sc[static_cast<size_t>(m) * 72 + tid];
    }
}

// Stash F32 hidden state `[M, 2304]` into `ws_taps [M, 4, 2304]` BF16 and copy `src -> dst` pass-through:
__global__ __launch_bounds__(256, 4)
void wv_ink_twf_stash_f32_tap_kernel(
    const float* __restrict__ src_f32,
    float* __restrict__ dst_f32,
    __nv_bfloat16* __restrict__ taps_bf16,
    int tap_slot,
    int M) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t tot = static_cast<size_t>(M) * INK2_HIDDEN;
    if (idx < tot) {
        float v = src_f32[idx];
        if (dst_f32 != src_f32) {
            dst_f32[idx] = v;
        }
        if (taps_bf16 != nullptr && tap_slot >= 0 && tap_slot < 4) {
            size_t m = idx / INK2_HIDDEN;
            size_t c = idx - m * INK2_HIDDEN;
            taps_bf16[(m * 4 + static_cast<size_t>(tap_slot)) * INK2_HIDDEN + c] = __float2bfloat16(v);
        }
    }
}

__global__ __launch_bounds__(32, 8)
void wv_ink_twf_set_i32_scalar_kernel(int32_t* dst, int32_t val) {
    if (threadIdx.x == 0) *dst = val;
}

// ============================================================================
// 3. Persistent Per-Model Handle (`Unpacked GPU Weights` + `WvInkEngine`)
// ============================================================================

struct UnpackedSwz8 {
    WvInkDevBuf pk_buf;
    WvInkDevBuf sc_buf;
    WvInkSwz8Linear view;

    int32_t unpack_from_dev(int dev, const void* ext3_ptr, int n_valid_rows, int K, cudaStream_t stream) {
        if (!ext3_ptr) return -1;
        int r8  = n_valid_rows / 8;
        int k32 = K / 32;
        if (pk_buf.alloc(dev, static_cast<size_t>(r8) * k32 * 32 * sizeof(uint2)) != 0) return -1;
        if (sc_buf.alloc(dev, static_cast<size_t>(r8) * k32 * 8 * sizeof(__half)) != 0) return -1;

        cudaPointerAttributes attr {};
        cudaError_t qerr = cudaPointerGetAttributes(&attr, ext3_ptr);
        (void) cudaGetLastError();
        if (qerr == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
            wv_ink_twf_unpack_ext3_kernel<<<dim3(k32, r8), 32, 0, stream>>>(
                static_cast<const uint8_t*>(ext3_ptr),
                pk_buf.as<uint2>(), sc_buf.as<__half>(), r8, k32);
        } else {
            const size_t ext3_bytes = static_cast<size_t>(n_valid_rows) * k32 * 34;
            WvInkDevBuf tmp_ext3;
            if (tmp_ext3.alloc(dev, ext3_bytes) != 0) return -1;
            WV_CUDA_CHECK(cudaMemcpyAsync(tmp_ext3.ptr, ext3_ptr, ext3_bytes, cudaMemcpyHostToDevice, stream));
            wv_ink_twf_unpack_ext3_kernel<<<dim3(k32, r8), 32, 0, stream>>>(
                tmp_ext3.as<uint8_t>(),
                pk_buf.as<uint2>(), sc_buf.as<__half>(), r8, k32);
            WV_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        view.pk = pk_buf.as<uint2>();
        view.sc = sc_buf.as<__half>();
        return 0;
    }
};

struct UnpackedLayerWeights {
    int n_hot  = 0;
    int n_cold = 0;
    std::vector<int32_t> h_tmap;
    WvInkDevBuf d_tmap;
    UnpackedSwz8 wq, wk, wv, wg, wo;
    WvInkDevBuf ctrl_forget_bf16[4];
    WvInkDevBuf ctrl_gain_bf16[4];
    WvInkDevBuf hot_gu_pk3, hot_gu_sc, hot_dn_pk3, hot_dn_sc;
    WvInkDevBuf cold_slabs;
};

struct UnpackedMtpLayerWeights {
    UnpackedSwz8 wq, wk, wv, wk_ctx, wv_ctx, wo, gate, up, down;
};

struct Ink2CudaHandle {
    const void* model_key = nullptr;
    int32_t     device_id = 0;
    bool        moe_unpacked = false;
    ggml_ink2_model_desc cached_desc {};

    WvInkDevBuf d_cb4096;
    WvInkDevBuf d_vq8_uv;
    WvInkDevBuf d_emb_wire_owned;
    const void* d_emb_wire_eff = nullptr;
    std::vector<std::unique_ptr<WvInkDevBuf>> staged_bufs;
    std::vector<uint8_t> h_lut4_128b;
    std::vector<uint8_t> h_vq8_4kb;

    const void* stage_raw_bytes(const void* ptr, size_t nbytes, cudaStream_t st) {
        if (!ptr || nbytes == 0) return nullptr;
        cudaPointerAttributes attr {};
        cudaError_t qerr = cudaPointerGetAttributes(&attr, ptr);
        (void) cudaGetLastError();
        if (qerr == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
            return ptr;
        }
        auto buf = std::make_unique<WvInkDevBuf>();
        if (buf->alloc(device_id, nbytes) != 0) return nullptr;
        if (cudaMemcpyAsync(buf->ptr, ptr, nbytes, cudaMemcpyHostToDevice, st) != cudaSuccess) return nullptr;
        const void* out = buf->ptr;
        staged_bufs.push_back(std::move(buf));
        return out;
    }

    const float* stage_f32(const float* ptr, size_t count, cudaStream_t st) {
        return static_cast<const float*>(stage_raw_bytes(ptr, count * sizeof(float), st));
    }

    static int32_t ensure_dev_bytes(
        int dev, const void* ptr, size_t nbytes,
        WvInkDevBuf& owned_buf, const void*& out_dev_ptr, cudaStream_t st) {
        if (!ptr || nbytes == 0) {
            out_dev_ptr = nullptr;
            return -1;
        }
        cudaPointerAttributes attr {};
        cudaError_t qerr = cudaPointerGetAttributes(&attr, ptr);
        (void) cudaGetLastError();
        if (qerr == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
            out_dev_ptr = ptr;
            return 0;
        }
        if (owned_buf.alloc(dev, nbytes) != 0) {
            out_dev_ptr = nullptr;
            return -1;
        }
        WV_CUDA_CHECK(cudaMemcpyAsync(owned_buf.ptr, ptr, nbytes, cudaMemcpyHostToDevice, st));
        out_dev_ptr = owned_buf.ptr;
        return 0;
    }

    UnpackedSwz8 head;
    UnpackedLayerWeights layers[24];
    UnpackedSwz8 mtp_fold;
    WvInkDevBuf  mtp_w1_i8, mtp_w1_sc;
    UnpackedMtpLayerWeights mtp_layers[3];

    WvInkEngine eng;

    // Host-side ubatch tracking (`notify_ubatch`) + deferred speculative commit state:
    int32_t cur_ubatch_b    = 1;
    int32_t cur_ubatch_s    = 1;
    int32_t cur_ubatch_pos0 = 0;
    bool    cur_needs_reset = false;
    std::vector<int32_t> cur_seq_ids;
    std::vector<int32_t> cur_seq_pos0;

    int32_t pending_spec_pos0  = 0;
    int32_t pending_spec_s     = 0;
    bool    pending_spec_sconv = false;
    bool    pending_spec_mtp   = false;

    // Captured M=1 CUDA Graph inside `llama_decode` (`FUSED_TRUNK`):
    WvInkCudaGraph g_decode_m1;
    const void*    g_m1_tok_ptr = nullptr;
    const void*    g_m1_pos_ptr = nullptr;
    const void*    g_m1_dst_ptr = nullptr;

    ~Ink2CudaHandle() {
        // Clear borrowed pointers in eng.dma before destruction so eng.dma doesn't double-cudaFree:
        for (int l = 0; l < INK2_NUM_LAYERS_MAX; ++l) {
            eng.dma.d_all_vram_slabs[l] = nullptr;
        }
    }

    int32_t flush_pending_spec(int32_t next_pos0, cudaStream_t stream) {
        if (pending_spec_s <= 0) return 0;
        DeviceGuard guard(device_id);
        if (next_pos0 <= 0) {
            pending_spec_s     = 0;
            pending_spec_sconv = false;
            pending_spec_mtp   = false;
            return 0;
        }
        int32_t n_acc = std::max(0, std::min(pending_spec_s, next_pos0 - pending_spec_pos0));
        if (n_acc > 0) {
            wv_ink_twf_set_i32_scalar_kernel<<<1, 32, 0, stream>>>(eng.d_step_out_t.as<int32_t>(), n_acc);
            if (pending_spec_sconv) {
                wv_ink_commit_all_sconv_cand_kernel<<<dim3(26, eng.n_visits), 256, 0, stream>>>(
                    eng.sc_k.as<float>(), eng.cand_k.as<float>(),
                    eng.sc_v.as<float>(), eng.cand_v.as<float>(),
                    eng.sc_a.as<float>(), eng.cand_a.as<float>(),
                    eng.sc_m.as<float>(), eng.cand_m.as<float>(),
                    eng.d_step_out_t.as<int32_t>(), eng.b_max, eng.n_visits);
            }
            if (pending_spec_mtp && eng.has_mtp) {
                wv_ink_twf_set_i32_scalar_kernel<<<1, 32, 0, stream>>>(eng.d_mtp_pos0_t.as<int32_t>(), pending_spec_pos0);
                eng.commit_mtp_taps_gpu(
                    eng.ws_taps.as<__nv_bfloat16>(), 1, pending_spec_s,
                    eng.d_step_out_t.as<int32_t>(), false, stream);
                int32_t new_mp0 = pending_spec_pos0 + n_acc;
                wv_ink_twf_set_i32_scalar_kernel<<<1, 32, 0, stream>>>(eng.d_mtp_pos0_t.as<int32_t>(), new_mp0);
                eng.mtp_seen_tokens = new_mp0;
            }
            eng.seen_tokens = pending_spec_pos0 + n_acc;
            if (!eng.h_slot_pos.empty()) eng.h_slot_pos[0] = eng.seen_tokens;
        }
        pending_spec_s     = 0;
        pending_spec_sconv = false;
        pending_spec_mtp   = false;
        return 0;
    }

    int32_t build_or_resize_engine(int32_t req_b_max, int32_t req_max_seq) {
        DeviceGuard guard(device_id);
        for (int l = 0; l < INK2_NUM_LAYERS_MAX; ++l) {
            eng.dma.d_all_vram_slabs[l] = nullptr;
        }
        g_decode_m1.reset();
        g_m1_tok_ptr = g_m1_pos_ptr = g_m1_dst_ptr = nullptr;

        WvInkEngineCfg ec;
        ec.device_id     = device_id;
        ec.b_max         = std::max(1, req_b_max);
        ec.max_seq       = std::max(INK2_SWA_RING, req_max_seq);
        ec.n_layers      = cached_desc.n_phys_layers;
        ec.loop_beg      = cached_desc.loop_beg;
        ec.loop_end      = cached_desc.loop_end;
        ec.n_loop_passes = cached_desc.n_loop_passes;
        ec.block_size    = cached_desc.block_size;
        ec.n_experts     = cached_desc.n_experts;
        ec.top_k         = cached_desc.top_k;
        ec.vocab_unpad   = cached_desc.vocab_unpad;
        ec.vocab_pad     = cached_desc.vocab_pad;
        ec.vocab_full    = cached_desc.vocab_full;
        ec.eos_id        = cached_desc.eos_id;
        ec.n_mtp_layers  = cached_desc.n_mtp_layers;
        for (int i = 0; i < 4; ++i) ec.tap_visits[i] = cached_desc.tap_visits[i];
        ec.use_all_vram            = moe_unpacked ? 1 : 0;
        ec.vram_cache_budget_bytes = 0;
        ec.enable_pdl              = 1;
        ec.enable_cuda_graphs      = 1;

        if (eng.init(ec) != 0) return -1;
        if (ec.max_seq >= 16384) {
            if (eng.set_kv_policy(1) != 0) return -1;
        }

        WvInkEngineGlobals g;
        g.emb_wire        = static_cast<const uint8_t*>(d_emb_wire_eff);
        g.embed_norm      = cached_desc.tok_norm_f32;
        g.final_norm      = cached_desc.output_norm_f32;
        g.head            = head.view;
        g.output_query    = cached_desc.output_res_query_f32;
        g.msg_queries     = cached_desc.loop_msg_queries_f32;
        g.mlp_queries     = cached_desc.mlp_res_queries_f32;
        for (int i = 0; i < 24; ++i) g.attn_queries[i] = cached_desc.attn_res_queries_f32[i];
        g.cb4096_gmem     = d_cb4096.as<uint32_t>();
        g.vq8_uv_gmem     = d_vq8_uv.as<uint32_t>();
        g.lut4_host_128b  = h_lut4_128b.empty() ? nullptr : h_lut4_128b.data();
        g.vq8_uv_host_4kb = h_vq8_4kb.empty()   ? nullptr : h_vq8_4kb.data();
        if (eng.set_globals(g) != 0) return -1;

        for (int li = 0; li < cached_desc.n_phys_layers; ++li) {
            const ggml_ink2_layer_ptrs& src_l = cached_desc.layers[li];
            UnpackedLayerWeights& ul          = layers[li];
            WvInkLayerDesc ld;
            ld.layer_idx = li;
            ld.is_local  = (src_l.is_local != 0);
            ld.is_loop   = (src_l.is_loop != 0);
            ld.n_hot     = moe_unpacked ? ul.n_hot  : 0;
            ld.n_cold    = moe_unpacked ? ul.n_cold : 0;
            if (moe_unpacked) {
                ld.wq = ul.wq.view; ld.wk = ul.wk.view; ld.wv = ul.wv.view;
                ld.wg = ul.wg.view; ld.wo = ul.wo.view;
                ld.wr            = static_cast<const __nv_bfloat16*>(src_l.wr_bf16);
                ld.w_sink        = static_cast<const __nv_bfloat16*>(src_l.w_sink_bf16);
                ld.q_clip_scale  = src_l.q_clip_f32;
                ld.r_clip_scale  = src_l.r_clip_f32;
                ld.rel_proj_t    = src_l.rel_proj_t_f32;
                ld.attn_sconv_w  = src_l.attn_sconv_f32;
                ld.mlp_sconv_w   = src_l.ffn_sconv_f32;
                ld.mlp_norm      = src_l.ffn_norm_f32;
                ld.router_w      = src_l.router_w_f32;
                ld.router_b      = src_l.router_b_f32;
                ld.tmap          = ul.d_tmap.as<int32_t>();
                ld.hot_gu_pk3    = ul.hot_gu_pk3.as<uint32_t>();
                ld.hot_gu_sc     = ul.hot_gu_sc.as<uint16_t>();
                ld.hot_dn_pk3    = ul.hot_dn_pk3.as<uint32_t>();
                ld.hot_dn_sc     = ul.hot_dn_sc.as<uint16_t>();
                ld.attn_norm     = src_l.attn_norm_f32;
                ld.k_sconv_w     = src_l.k_sconv_f32;
                ld.v_sconv_w     = src_l.v_sconv_f32;
                ld.q_input_norm  = src_l.q_in_norm_f32;
                ld.k_input_norm  = src_l.k_in_norm_f32;
                ld.v_input_norm  = src_l.v_in_norm_f32;
                ld.k_static_w    = src_l.k_static_w_f32;
                ld.k_wproj       = static_cast<const __nv_bfloat16*>(src_l.k_wproj_bf16);
                ld.v_static_w    = src_l.v_static_w_f32;
                ld.v_wproj       = static_cast<const __nv_bfloat16*>(src_l.v_wproj_bf16);
                for (int c = 0; c < 4; ++c) {
                    ld.ctrl_down[c]   = static_cast<const __nv_bfloat16*>(src_l.ctrl_down_bf16[c]);
                    ld.ctrl_forget[c] = ul.ctrl_forget_bf16[c].as<__nv_bfloat16>();
                    ld.ctrl_gain[c]   = ul.ctrl_gain_bf16[c].as<__nv_bfloat16>();
                }
                ld.tmap_cpu = ul.h_tmap.empty() ? nullptr : ul.h_tmap.data();
            }
            if (eng.set_layer(li, ld) != 0) return -1;
        }

        if (cached_desc.n_mtp_layers > 0) {
            eng.mtp_fuse_alpha = cached_desc.mtp_fuse_alpha_f32;
            eng.mtp_mask_embed = static_cast<const __nv_bfloat16*>(cached_desc.mtp_mask_embed_bf16);
            eng.mtp_norm_w     = cached_desc.mtp_norm_f32;
            eng.mtp_mh_dflare  = cached_desc.mtp_mh_dflare_f32;
            eng.mtp_rnn_w      = cached_desc.mtp_rnn_w_f32;
            eng.mtp_rnn_b      = cached_desc.mtp_rnn_b_f32;
            eng.mtp_fold       = mtp_fold.view;
            eng.mtp_fold_b     = cached_desc.mtp_fold_b_f32;
            eng.mtp_w1_i8      = mtp_w1_i8.as<int8_t>();
            eng.mtp_w1_sc      = mtp_w1_sc.as<__half>();
            for (int mi = 0; mi < cached_desc.n_mtp_layers; ++mi) {
                const ggml_ink2_mtp_layer_ptrs& sm = cached_desc.mtp_layers[mi];
                UnpackedMtpLayerWeights& um        = mtp_layers[mi];
                WvInkMtpLayerDesc md;
                md.is_local    = (sm.is_local != 0);
                md.dflare_norm = sm.dflare_norm_f32;
                md.in_norm     = sm.in_norm_f32;
                md.post_norm   = sm.post_norm_f32;
                md.q_norm      = sm.q_norm_f32;
                md.k_norm      = sm.k_norm_f32;
                md.k_norm_ctx  = sm.k_norm_ctx_f32;
                md.q     = um.wq.view;     md.k     = um.wk.view;     md.v = um.wv.view;
                md.k_ctx = um.wk_ctx.view; md.v_ctx = um.wv_ctx.view; md.o = um.wo.view;
                md.gate  = um.gate.view;   md.up    = um.up.view;     md.down = um.down.view;
                if (eng.set_mtp_layer(mi, md) != 0) return -1;
            }
        }

        if (eng.finalize() != 0) return -1;
        if (moe_unpacked) {
            eng.dma.all_vram = true;
            for (int li = 0; li < cached_desc.n_phys_layers; ++li) {
                eng.dma.d_all_vram_slabs[li] = layers[li].cold_slabs.as<uint8_t>();
            }
        }
        return 0;
    }

    int32_t init_and_unpack(const ggml_ink2_model_desc* desc) {
        if (!desc) return set_error("Null ggml_ink2_model_desc");
        model_key    = desc->model_key;
        device_id    = desc->device_id;
        moe_unpacked = (desc->unpack_full_vram_moe != 0);
        cached_desc  = *desc;

        DeviceGuard guard(device_id);
        WV_CUDA_CHECK(cudaSetDevice(device_id));
        cudaStream_t st = nullptr;
        WV_CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

        auto cleanup_err = [&](const char* msg) -> int32_t {
            if (st) cudaStreamDestroy(st);
            return msg ? set_error(msg) : -1;
        };

        // 1. Parse QWV2 aux_blob_host for codebooks & tmaps:
        if (!desc->aux_blob_host || desc->aux_blob_size < sizeof(Qwv2AuxHeader)) {
            return cleanup_err("Missing or truncated QWV2 quant_plugin_aux blob");
        }
        const uint8_t* aux_h = static_cast<const uint8_t*>(desc->aux_blob_host);
        Qwv2AuxHeader hdr {};
        std::memcpy(&hdr, aux_h, sizeof(hdr));
        if (hdr.magic != kQwv2AuxMagic || hdr.version != 2) {
            return cleanup_err("Invalid QWV2 aux_blob magic/version");
        }

        if (d_cb4096.alloc(device_id, 16384) != 0 || d_vq8_uv.alloc(device_id, 4096) != 0) {
            return cleanup_err(nullptr);
        }
        WV_CUDA_CHECK(cudaMemcpyAsync(d_cb4096.ptr, aux_h + hdr.cb4096_off, 16384, cudaMemcpyHostToDevice, st));
        WV_CUDA_CHECK(cudaMemcpyAsync(d_vq8_uv.ptr, aux_h + hdr.vq8_uv_off, 4096,  cudaMemcpyHostToDevice, st));
        WV_CUDA_CHECK(cudaMemcpyToSymbolAsync(d_wv_vq12_cb4096, aux_h + hdr.cb4096_off, 16384, 0, cudaMemcpyHostToDevice, st));
        h_lut4_128b.assign(aux_h + hdr.lut4_off,   aux_h + hdr.lut4_off   + 128);
        h_vq8_4kb.assign  (aux_h + hdr.vq8_uv_off, aux_h + hdr.vq8_uv_off + 4096);

        // 2. Upload `tok_embd_ext2` if host-resident (~241.5 MiB) & unpack `output.weight` + `mtp.*`:
        const size_t emb_bytes = static_cast<size_t>(desc->vocab_full) * (INK2_HIDDEN / 128) * 70;
        if (ensure_dev_bytes(device_id, desc->tok_embd_ext2, emb_bytes, d_emb_wire_owned, d_emb_wire_eff, st) != 0) {
            return cleanup_err("Failed to stage tok_embd_ext2 on CUDA device");
        }
        if (head.unpack_from_dev(device_id, desc->output_ext3, desc->vocab_pad, INK2_HIDDEN, st) != 0) {
            return cleanup_err(nullptr);
        }
        // Ensure global norm/query vectors reside on GPU even under partial/CPU tensor placement:
        cached_desc.tok_norm_f32         = stage_f32(cached_desc.tok_norm_f32, INK2_HIDDEN, st);
        cached_desc.output_norm_f32      = stage_f32(cached_desc.output_norm_f32, INK2_HIDDEN, st);
        cached_desc.output_res_query_f32 = stage_f32(cached_desc.output_res_query_f32, INK2_HIDDEN, st);
        if (cached_desc.loop_msg_queries_f32 && cached_desc.loop_end > cached_desc.loop_beg) {
            cached_desc.loop_msg_queries_f32 = stage_f32(
                cached_desc.loop_msg_queries_f32,
                static_cast<size_t>(cached_desc.loop_end - cached_desc.loop_beg) * 4 * INK2_HIDDEN, st);
        }
        if (cached_desc.mlp_res_queries_f32 && cached_desc.n_phys_layers > 0) {
            cached_desc.mlp_res_queries_f32 = stage_f32(
                cached_desc.mlp_res_queries_f32,
                static_cast<size_t>(cached_desc.n_phys_layers) * INK2_HIDDEN, st);
        }
        for (int li = 0; li < cached_desc.n_phys_layers && li < 24; ++li) {
            if (cached_desc.attn_res_queries_f32[li]) {
                size_t nq = (cached_desc.layers[li].is_local ? 1u : 3u) * INK2_HIDDEN;
                cached_desc.attn_res_queries_f32[li] = stage_f32(cached_desc.attn_res_queries_f32[li], nq, st);
            }
        }

        WvInkDevBuf d_w1_tmp;
        if (desc->n_mtp_layers > 0) {
            cached_desc.mtp_fuse_alpha_f32  = stage_f32(cached_desc.mtp_fuse_alpha_f32, 16u, st);
            cached_desc.mtp_mask_embed_bf16 = stage_raw_bytes(cached_desc.mtp_mask_embed_bf16, INK2_HIDDEN * sizeof(__nv_bfloat16), st);
            cached_desc.mtp_norm_f32        = stage_f32(cached_desc.mtp_norm_f32, INK2_HIDDEN, st);
            cached_desc.mtp_mh_dflare_f32   = stage_f32(cached_desc.mtp_mh_dflare_f32, INK2_HIDDEN, st);
            cached_desc.mtp_rnn_w_f32       = stage_f32(cached_desc.mtp_rnn_w_f32, 768u * 5120u, st);
            cached_desc.mtp_rnn_b_f32       = stage_f32(cached_desc.mtp_rnn_b_f32, 768u, st);
            cached_desc.mtp_fold_b_f32      = stage_f32(cached_desc.mtp_fold_b_f32, static_cast<size_t>(desc->vocab_pad), st);

            if (mtp_fold.unpack_from_dev(device_id, desc->mtp_fold_ext3, desc->vocab_pad, INK2_MTP_RANK, st) != 0) {
                return cleanup_err(nullptr);
            }
            int w1_rows = (desc->mtp_mh_w1_rows > 0) ? desc->mtp_mh_w1_rows : desc->vocab_full;
            if (mtp_w1_i8.alloc(device_id, static_cast<size_t>(w1_rows) * 256, true) != 0 ||
                mtp_w1_sc.alloc(device_id, static_cast<size_t>(w1_rows) * 8 * sizeof(__half), true) != 0) {
                return cleanup_err(nullptr);
            }
            if (desc->mtp_mh_w1_q8_0 != nullptr) {
                const void* d_w1_src = nullptr;
                if (ensure_dev_bytes(device_id, desc->mtp_mh_w1_q8_0, static_cast<size_t>(w1_rows) * 8 * 34, d_w1_tmp, d_w1_src, st) != 0) {
                    return cleanup_err("Failed to stage mtp_mh_w1_q8_0 on CUDA device");
                }
                wv_ink_twf_unpack_q8_0_kernel<<<w1_rows, 256, 0, st>>>(
                    static_cast<const uint8_t*>(d_w1_src),
                    mtp_w1_i8.as<int8_t>(), mtp_w1_sc.as<__half>(), w1_rows);
            }
            for (int mi = 0; mi < desc->n_mtp_layers; ++mi) {
                ggml_ink2_mtp_layer_ptrs& sm = cached_desc.mtp_layers[mi];
                sm.dflare_norm_f32 = stage_f32(sm.dflare_norm_f32, INK2_HIDDEN, st);
                sm.in_norm_f32     = stage_f32(sm.in_norm_f32,     INK2_HIDDEN, st);
                sm.post_norm_f32   = stage_f32(sm.post_norm_f32,   INK2_HIDDEN, st);
                sm.q_norm_f32      = stage_f32(sm.q_norm_f32,      128u,        st);
                sm.k_norm_f32      = stage_f32(sm.k_norm_f32,      128u,        st);
                sm.k_norm_ctx_f32  = stage_f32(sm.k_norm_ctx_f32,  128u,        st);

                UnpackedMtpLayerWeights& um        = mtp_layers[mi];
                if (um.wq.unpack_from_dev    (device_id, sm.wq_ext3,       INK2_ATTN_DIM,  INK2_HIDDEN,    st) != 0 ||
                    um.wk.unpack_from_dev    (device_id, sm.wk_ext3,       INK2_KV_DIM,    INK2_HIDDEN,    st) != 0 ||
                    um.wv.unpack_from_dev    (device_id, sm.wv_ext3,       INK2_KV_DIM,    INK2_HIDDEN,    st) != 0 ||
                    um.wk_ctx.unpack_from_dev(device_id, sm.wk_ctx_ext3,   INK2_KV_DIM,    INK2_HIDDEN,    st) != 0 ||
                    um.wv_ctx.unpack_from_dev(device_id, sm.wv_ctx_ext3,   INK2_KV_DIM,    INK2_HIDDEN,    st) != 0 ||
                    um.wo.unpack_from_dev    (device_id, sm.wo_ext3,       INK2_HIDDEN,    INK2_ATTN_DIM,  st) != 0 ||
                    um.gate.unpack_from_dev  (device_id, sm.ffn_gate_ext3, INK2_MTP_INTER, INK2_HIDDEN,    st) != 0 ||
                    um.up.unpack_from_dev    (device_id, sm.ffn_up_ext3,   INK2_MTP_INTER, INK2_HIDDEN,    st) != 0 ||
                    um.down.unpack_from_dev  (device_id, sm.ffn_down_ext3, INK2_HIDDEN,    INK2_MTP_INTER, st) != 0) {
                    return cleanup_err(nullptr);
                }
            }
        }

        // 3. Full-VRAM trunk + MoE unpack (`-ngl 99` Phase B fast path):
        WvInkDevBuf d_aux_tmp;
        if (moe_unpacked) {
            if (d_aux_tmp.alloc(device_id, desc->aux_blob_size) != 0) {
                return cleanup_err(nullptr);
            }
            WV_CUDA_CHECK(cudaMemcpyAsync(d_aux_tmp.ptr, aux_h, desc->aux_blob_size, cudaMemcpyHostToDevice, st));
            const int32_t* all_tmaps = reinterpret_cast<const int32_t*>(aux_h + hdr.tmap_off);

            for (int li = 0; li < desc->n_phys_layers; ++li) {
                const ggml_ink2_layer_ptrs& sl = desc->layers[li];
                UnpackedLayerWeights& ul       = layers[li];
                ul.h_tmap.assign(all_tmaps + li * desc->n_experts, all_tmaps + (li + 1) * desc->n_experts);
                ul.n_hot = 0; ul.n_cold = 0;
                for (int32_t tv : ul.h_tmap) {
                    if (tv >= 0) ul.n_hot++;
                    else         ul.n_cold++;
                }
                if (ul.d_tmap.alloc(device_id, desc->n_experts * sizeof(int32_t)) != 0) return cleanup_err(nullptr);
                WV_CUDA_CHECK(cudaMemcpyAsync(ul.d_tmap.ptr, ul.h_tmap.data(), desc->n_experts * sizeof(int32_t), cudaMemcpyHostToDevice, st));

                if (ul.wq.unpack_from_dev(device_id, sl.wq_ext3, INK2_ATTN_DIM, INK2_HIDDEN,   st) != 0 ||
                    ul.wk.unpack_from_dev(device_id, sl.wk_ext3, INK2_KV_DIM,   INK2_HIDDEN,   st) != 0 ||
                    ul.wv.unpack_from_dev(device_id, sl.wv_ext3, INK2_KV_DIM,   INK2_HIDDEN,   st) != 0 ||
                    ul.wg.unpack_from_dev(device_id, sl.wg_ext3, INK2_ATTN_DIM, INK2_HIDDEN,   st) != 0 ||
                    ul.wo.unpack_from_dev(device_id, sl.wo_ext3, INK2_HIDDEN,   INK2_ATTN_DIM, st) != 0) {
                    return cleanup_err(nullptr);
                }

                if (sl.is_loop) {
                    size_t cf_elems = 2304u * 64u;
                    for (int c = 0; c < 4; ++c) {
                        if (sl.ctrl_forget_f32[c] && sl.ctrl_gain_f32[c]) {
                            if (ul.ctrl_forget_bf16[c].alloc(device_id, cf_elems * sizeof(__nv_bfloat16)) != 0 ||
                                ul.ctrl_gain_bf16[c].alloc  (device_id, cf_elems * sizeof(__nv_bfloat16)) != 0) {
                                return cleanup_err(nullptr);
                            }
                            unsigned grd = static_cast<unsigned>((cf_elems + 255) / 256);
                            wv_ink_twf_f32_to_bf16_kernel<<<grd, 256, 0, st>>>(
                                sl.ctrl_forget_f32[c], ul.ctrl_forget_bf16[c].as<__nv_bfloat16>(), cf_elems);
                            wv_ink_twf_f32_to_bf16_kernel<<<grd, 256, 0, st>>>(
                                sl.ctrl_gain_f32[c], ul.ctrl_gain_bf16[c].as<__nv_bfloat16>(), cf_elems);
                        }
                    }
                }

                if (ul.n_hot > 0) {
                    size_t gu_pk_b = static_cast<size_t>(ul.n_hot) * 288 * 18 * 96 * sizeof(uint32_t);
                    size_t gu_sc_b = static_cast<size_t>(ul.n_hot) * 288 * 18 * 20 * sizeof(uint16_t);
                    size_t dn_pk_b = static_cast<size_t>(ul.n_hot) * 288 * 9  * 96 * sizeof(uint32_t);
                    size_t dn_sc_b = static_cast<size_t>(ul.n_hot) * 288 * 9  * 20 * sizeof(uint16_t);
                    if (ul.hot_gu_pk3.alloc(device_id, gu_pk_b) != 0 ||
                        ul.hot_gu_sc.alloc (device_id, gu_sc_b) != 0 ||
                        ul.hot_dn_pk3.alloc(device_id, dn_pk_b) != 0 ||
                        ul.hot_dn_sc.alloc (device_id, dn_sc_b) != 0) {
                        return cleanup_err(nullptr);
                    }
                    wv_ink_twf_unpack_hot_gu_kernel<<<dim3(18, 288, desc->n_experts), 32, 0, st>>>(
                        static_cast<const uint8_t*>(sl.gate_exps_ext0),
                        static_cast<const uint8_t*>(sl.up_exps_ext0),
                        d_aux_tmp.as<uint8_t>(), ul.d_tmap.as<int32_t>(),
                        ul.hot_gu_pk3.as<uint32_t>(), ul.hot_gu_sc.as<uint16_t>(), desc->n_experts);
                    wv_ink_twf_unpack_hot_dn_kernel<<<dim3(9, 288, desc->n_experts), 32, 0, st>>>(
                        static_cast<const uint8_t*>(sl.down_exps_ext0),
                        d_aux_tmp.as<uint8_t>(), ul.d_tmap.as<int32_t>(),
                        ul.hot_dn_pk3.as<uint32_t>(), ul.hot_dn_sc.as<uint16_t>(), desc->n_experts);
                }

                if (ul.n_cold > 0) {
                    size_t cold_b = static_cast<size_t>(ul.n_cold) * INK2_COLD_SLAB_BYTES;
                    if (ul.cold_slabs.alloc(device_id, cold_b) != 0) return cleanup_err(nullptr);
                    wv_ink_twf_unpack_cold_gu_kernel<<<dim3(36, 288, desc->n_experts), 32, 0, st>>>(
                        static_cast<const uint8_t*>(sl.gate_exps_ext0),
                        static_cast<const uint8_t*>(sl.up_exps_ext0),
                        ul.d_tmap.as<int32_t>(), ul.cold_slabs.as<uint8_t>(), desc->n_experts);
                    wv_ink_twf_unpack_cold_dn_kernel<<<dim3(18, 288, desc->n_experts), 32, 0, st>>>(
                        static_cast<const uint8_t*>(sl.down_exps_ext0),
                        ul.d_tmap.as<int32_t>(), ul.cold_slabs.as<uint8_t>(), desc->n_experts);
                }
            }
        }

        WV_CUDA_CHECK(cudaStreamSynchronize(st));
        cudaStreamDestroy(st);
        d_aux_tmp.release();

        return build_or_resize_engine(desc->b_max, desc->max_seq);
    }

    int32_t forward_chunk_from_embd_f32(
        const float* d_embd_f32, int B_act, int S,
        const int32_t* d_pos_arr, const int32_t* d_smap,
        bool capture_taps, int commit_sconv, bool pdl, cudaStream_t stream) {
        int M = B_act * S;
        __nv_bfloat16* cur_h   = eng.ws_h.as<__nv_bfloat16>();
        __nv_bfloat16* pre_h   = eng.ws_pre_h.as<__nv_bfloat16>();
        __nv_bfloat16* prev_h  = eng.ws_prev_h.as<__nv_bfloat16>();
        __nv_bfloat16* h_fin   = eng.ws_h_fin.as<__nv_bfloat16>();
        __nv_bfloat16* blks    = eng.ws_blocks.as<__nv_bfloat16>();
        __nv_bfloat16* pre_blk = eng.ws_pre_blk.as<__nv_bfloat16>();
        __nv_bfloat16* prv_blk = eng.ws_prev_blk.as<__nv_bfloat16>();
        __nv_bfloat16* taps    = eng.ws_taps.as<__nv_bfloat16>();

        size_t tot = static_cast<size_t>(M) * INK2_HIDDEN;
        wv_ink_twf_f32_to_bf16_kernel<<<static_cast<unsigned>((tot + 255) / 256), 256, 0, stream>>>(
            d_embd_f32, cur_h, tot);

        int n_blk = 0;
        int c_grd = (M * INK2_HIDDEN + 255) / 256;
        for (int v_idx = 0; v_idx < eng.cfg.loop_beg; ++v_idx) {
            if (capture_taps) eng.maybe_stash_tap(v_idx, cur_h, taps, M, c_grd, pdl, stream);
            if (eng.run_visit(v_idx, B_act, S, d_pos_arr, d_smap, n_blk, cur_h, blks, nullptr, commit_sconv, pdl, stream) != 0) return -1;
        }
        int loop_len = eng.cfg.loop_end - eng.cfg.loop_beg;
        if (loop_len > 0) {
            if (eng.cfg.n_loop_passes > 1) {
                wv_launch_pdl(pdl, dim3(c_grd), dim3(256), 0, stream, wv_ink_save_prelude_kernel, cur_h, blks, pre_h, pre_blk, M, 6);
            }
            int n_blk_after_prelude = n_blk;
            float* msgs = eng.ws_msgs_f32.as<float>();
            for (int p = 0; p < eng.cfg.n_loop_passes; ++p) {
                if (p > 0) {
                    int prv_src_slot = std::max(0, n_blk - 1);
                    wv_launch_pdl(pdl, dim3(c_grd), dim3(256), 0, stream,
                        wv_ink_loop_pass_reset_kernel, cur_h, prev_h, pre_h, blks, prv_blk, pre_blk, M, 6, prv_src_slot);
                    n_blk = n_blk_after_prelude;
                }
                for (int j = 0; j < loop_len; ++j) {
                    int v_idx = eng.cfg.loop_beg + p * loop_len + j;
                    if (capture_taps) eng.maybe_stash_tap(v_idx, cur_h, taps, M, c_grd, pdl, stream);
                    bool use_msg = (p > 0) && (eng.glob.msg_queries != nullptr);
                    if (use_msg) {
                        wv_launch_pdl(pdl, dim3(M), dim3(32, 6), 0, stream,
                            wv_ink_block_attn_res_kernel,
                            prv_blk, prev_h, eng.glob.msg_queries + static_cast<size_t>(j) * 4 * INK2_HIDDEN, msgs, 1, 2, 4, 1, M);
                    }
                    if (eng.run_visit(v_idx, B_act, S, d_pos_arr, d_smap, n_blk, cur_h, blks, use_msg ? msgs : nullptr, commit_sconv, pdl, stream) != 0) return -1;
                }
            }
        }
        int post_start = eng.cfg.loop_beg + eng.cfg.n_loop_passes * loop_len;
        for (int v_idx = post_start; v_idx < eng.n_visits; ++v_idx) {
            if (capture_taps) eng.maybe_stash_tap(v_idx, cur_h, taps, M, c_grd, pdl, stream);
            if (eng.run_visit(v_idx, B_act, S, d_pos_arr, d_smap, n_blk, cur_h, blks, nullptr, commit_sconv, pdl, stream) != 0) return -1;
        }
        wv_launch_pdl(pdl, dim3(M), dim3(32, 6), 0, stream,
            wv_ink_block_attn_res_fused_norm_q8_kernel,
            blks, cur_h, eng.glob.output_query, eng.glob.final_norm, h_fin, eng.ws_q8_a.as<int8_t>(), eng.ws_sc_a.as<float>(), n_blk, 6, 1, M);
        if (capture_taps) {
            wv_launch_pdl(pdl, dim3(c_grd), dim3(256), 0, stream, wv_ink_stash_tap_kernel, h_fin, taps, 3, M);
        }
        return 0;
    }
};

static std::mutex g_ink2_mtx;
static std::vector<Ink2CudaHandle*> g_ink2_handles;

// ============================================================================
// 4. `ggml_cuda_inkling2_api_t` C Callbacks
// ============================================================================

static void* api_ensure_handle(const ggml_ink2_model_desc* desc) {
    if (!desc) return nullptr;
    std::lock_guard<std::mutex> lk(g_ink2_mtx);
    bool want_moe = (desc->unpack_full_vram_moe != 0);
    for (size_t i = 0; i < g_ink2_handles.size(); ++i) {
        Ink2CudaHandle* h = g_ink2_handles[i];
        if (h->model_key == desc->model_key && h->device_id == desc->device_id && h->moe_unpacked == want_moe) {
            int req_b = std::max(h->eng.b_max, desc->b_max);
            int req_s = std::max(h->eng.max_seq, desc->max_seq);
            if (req_b > h->eng.b_max || req_s > h->eng.max_seq) {
                if (h->build_or_resize_engine(req_b, req_s) != 0) return nullptr;
            }
            return h;
        }
    }
    auto* h = new (std::nothrow) Ink2CudaHandle();
    if (!h) return nullptr;
    if (h->init_and_unpack(desc) != 0) {
        delete h;
        return nullptr;
    }
    g_ink2_handles.push_back(h);
    return h;
}

static void api_destroy_by_model(const void* model_key) {
    std::lock_guard<std::mutex> lk(g_ink2_mtx);
    for (size_t i = 0; i < g_ink2_handles.size();) {
        if (g_ink2_handles[i]->model_key == model_key) {
            delete g_ink2_handles[i];
            g_ink2_handles.erase(g_ink2_handles.begin() + static_cast<ptrdiff_t>(i));
        } else {
            ++i;
        }
    }
}

static int32_t api_reset_cache(void* handle) {
    if (!handle) return -1;
    auto* h = static_cast<Ink2CudaHandle*>(handle);
    h->pending_spec_s     = 0;
    h->pending_spec_sconv = false;
    h->pending_spec_mtp   = false;
    return h->eng.reset_cache();
}

static int32_t api_seq_rm(void* handle, int32_t seq_id, int32_t p0, int32_t p1) {
    if (!handle) return -1;
    auto* h = static_cast<Ink2CudaHandle*>(handle);
    if (p0 <= 0 && (p1 < 0 || p1 >= h->eng.seen_tokens)) {
        if (seq_id <= 0 && h->eng.b_max == 1) {
            return api_reset_cache(handle);
        }
        if (seq_id >= 0 && seq_id < h->eng.b_max) {
            return h->eng.reset_slot(seq_id);
        }
    }
    if (seq_id == 0 && p0 > 0 && h->pending_spec_s > 0 && p1 < 0) {
        DeviceGuard guard(h->device_id);
        if (h->flush_pending_spec(p0, h->eng.exec_stream) != 0) return -1;
        WV_CUDA_CHECK(cudaStreamSynchronize(h->eng.exec_stream));
    }
    return 0;
}

static int32_t api_set_mtp_policy(void* handle, int32_t policy) {
    if (!handle) return -1;
    static_cast<Ink2CudaHandle*>(handle)->eng.set_mtp_policy(policy);
    return 0;
}

static int32_t api_prefill(
    void* handle, const int32_t* ids_b_s_host, int32_t b_act, int32_t s_tot,
    int32_t enable_mtp_commit, float temp, float top_p, uint64_t seed, int32_t* out_next_toks_host,
    const void* vis_bf16_or_f32, int32_t vis_is_f32, int32_t n_vis, int32_t vis_is_device, int32_t image_token_id) {
    if (!handle) return -1;
    auto* h = static_cast<Ink2CudaHandle*>(handle);
    h->pending_spec_s = 0;
    h->pending_spec_sconv = false;
    h->pending_spec_mtp = false;

    WvInkDevBuf vis_conv;
    const void* vis_bf16 = vis_bf16_or_f32;
    bool is_dev          = (vis_is_device != 0);
    if (vis_bf16_or_f32 != nullptr && n_vis > 0 && vis_is_f32 != 0) {
        DeviceGuard guard(h->device_id);
        size_t n_el = static_cast<size_t>(n_vis) * INK2_HIDDEN;
        if (vis_conv.alloc(h->device_id, n_el * sizeof(__nv_bfloat16)) != 0) return -1;
        if (is_dev) {
            wv_ink_twf_f32_to_bf16_kernel<<<static_cast<unsigned>((n_el + 255) / 256), 256, 0, h->eng.exec_stream>>>(
                static_cast<const float*>(vis_bf16_or_f32), vis_conv.as<__nv_bfloat16>(), n_el);
        } else {
            WvInkDevBuf tmp_f32;
            if (tmp_f32.alloc(h->device_id, n_el * sizeof(float)) != 0) return -1;
            WV_CUDA_CHECK(cudaMemcpyAsync(tmp_f32.ptr, vis_bf16_or_f32, n_el * sizeof(float), cudaMemcpyHostToDevice, h->eng.exec_stream));
            wv_ink_twf_f32_to_bf16_kernel<<<static_cast<unsigned>((n_el + 255) / 256), 256, 0, h->eng.exec_stream>>>(
                tmp_f32.as<float>(), vis_conv.as<__nv_bfloat16>(), n_el);
            WV_CUDA_CHECK(cudaStreamSynchronize(h->eng.exec_stream));
        }
        vis_bf16 = vis_conv.ptr;
        is_dev   = true;
    }
    return h->eng.prefill(
        ids_b_s_host, b_act, s_tot, enable_mtp_commit != 0, temp, top_p, seed,
        out_next_toks_host, vis_bf16, n_vis, is_dev, image_token_id);
}

static int32_t api_decode_m1(
    void* handle, const int32_t* first_toks_host, int32_t b_act, int32_t max_new_tokens,
    int32_t eos_id, float temp, float top_p, uint64_t seed, int32_t* out_ids_b_t_host, int32_t* out_steps_done) {
    if (!handle) return -1;
    auto* h = static_cast<Ink2CudaHandle*>(handle);
    return h->eng.decode_m1(first_toks_host, b_act, max_new_tokens, eos_id, temp, top_p, seed, out_ids_b_t_host, out_steps_done);
}

static int32_t api_decode_mtp8(
    void* handle, int32_t first_tok, int32_t max_new_tokens, int32_t eos_id,
    float temp, float top_p, uint64_t seed, int32_t* out_tokens_host, int32_t* out_len, ggml_ink2_mtp_stats* out_stats) {
    if (!handle) return -1;
    auto* h = static_cast<Ink2CudaHandle*>(handle);
    WvInkMtpStatsOut st {};
    int32_t rc = h->eng.decode_mtp8(first_tok, max_new_tokens, eos_id, temp, top_p, seed, out_tokens_host, out_len, &st);
    if (rc == 0 && out_stats != nullptr) {
        out_stats->spec_steps        = st.spec_steps;
        out_stats->drafted_tokens    = st.drafted_tokens;
        out_stats->accepted_tokens   = st.accepted_tokens;
        out_stats->mean_accepted_len = st.mean_accepted_len;
        out_stats->s4_steps          = st.s4_steps;
        out_stats->s8_steps          = st.s8_steps;
    }
    return rc;
}

static int32_t api_prefill_slot(
    void* handle, int32_t slot_b, const int32_t* ids_s_host, int32_t s_tot,
    float temp, float top_p, uint64_t seed, int32_t* out_tok0) {
    if (!handle) return -1;
    return static_cast<Ink2CudaHandle*>(handle)->eng.prefill_slot(slot_b, ids_s_host, s_tot, temp, top_p, seed, out_tok0);
}

static int32_t api_step_continuous_m1(
    void* handle, const int32_t* cur_toks_host, int32_t b_span, int32_t upload_cur_toks,
    float temp, float top_p, uint64_t seed, int32_t* out_toks_host) {
    if (!handle) return -1;
    return static_cast<Ink2CudaHandle*>(handle)->eng.step_continuous_m1(
        cur_toks_host, b_span, upload_cur_toks != 0, temp, top_p, seed, out_toks_host);
}

static int32_t api_mtp_draft_block(
    void* handle, int32_t pos0, int32_t anchor_tok, int32_t k_draft, int32_t* out_draft_toks_host) {
    if (!handle || !out_draft_toks_host) return -1;
    auto* h = static_cast<Ink2CudaHandle*>(handle);
    if (!h->eng.finalized || !h->eng.has_mtp) return set_error("MTP-8 not available on Ink2CudaHandle");
    int k_clamped = std::max(1, std::min(7, static_cast<int>(k_draft)));
    DeviceGuard guard(h->device_id);
    cudaStream_t st = h->eng.exec_stream;

    if (h->pending_spec_s > 0) {
        if (h->flush_pending_spec(pos0, st) != 0) return -1;
    }
    wv_ink_twf_set_i32_scalar_kernel<<<1, 32, 0, st>>>(h->eng.d_mtp_pos0_t.as<int32_t>(), pos0);
    wv_ink_twf_set_i32_scalar_kernel<<<1, 32, 0, st>>>(h->eng.ws_draft8.as<int32_t>(), anchor_tok);
    h->eng.draft_k_block_gpu(k_clamped, false, h->eng.use_pdl, st);
    WV_CUDA_CHECK(cudaMemcpyAsync(
        out_draft_toks_host, h->eng.ws_draft8.as<int32_t>() + 1,
        static_cast<size_t>(k_clamped) * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
    WV_CUDA_CHECK(cudaStreamSynchronize(st));
    return k_clamped;
}

static void api_notify_ubatch(
    void* handle, int32_t b_act, int32_t s_tokens,
    const int32_t* host_seq_ids, const int32_t* host_pos0_per_seq) {
    if (!handle) return;
    auto* h = static_cast<Ink2CudaHandle*>(handle);
    h->cur_ubatch_b    = std::max(1, b_act);
    h->cur_ubatch_s    = std::max(1, s_tokens);
    h->cur_ubatch_pos0 = (host_pos0_per_seq != nullptr) ? host_pos0_per_seq[0] : 0;
    h->cur_needs_reset = (h->cur_ubatch_b == 1 && h->cur_ubatch_pos0 == 0 && (h->eng.seen_tokens > 0 || h->pending_spec_s > 0));
    if (host_seq_ids != nullptr && b_act > 0) {
        h->cur_seq_ids.assign(host_seq_ids, host_seq_ids + b_act);
    }
    if (host_pos0_per_seq != nullptr && b_act > 0) {
        h->cur_seq_pos0.assign(host_pos0_per_seq, host_pos0_per_seq + b_act);
    }
    if (!h->moe_unpacked && h->eng.finalized) {
        DeviceGuard guard(h->device_id);
        if (h->cur_needs_reset) {
            h->cur_needs_reset = false;
            h->pending_spec_s  = 0;
            (void) h->eng.reset_cache();
        } else if (h->pending_spec_s > 0 && h->cur_ubatch_b == 1) {
            (void) h->flush_pending_spec(h->cur_ubatch_pos0, h->eng.exec_stream);
            (void) cudaStreamSynchronize(h->eng.exec_stream);
        }
    }
}

static int32_t api_forward_logits_f32(
    void* handle, const int32_t* ids_s_host, int32_t s_tot, float* out_logits_s_v_host) {
    if (!handle) return -1;
    return static_cast<Ink2CudaHandle*>(handle)->eng.forward_all_logits_f32(ids_s_host, s_tot, out_logits_s_v_host);
}

static const char* api_last_error(void*) {
    return wv::tls_last_error().c_str();
}

static const ggml_cuda_inkling2_api_t g_ink2_api = {
    sizeof(ggml_cuda_inkling2_api_t),
    api_ensure_handle,
    api_destroy_by_model,
    api_reset_cache,
    api_seq_rm,
    api_set_mtp_policy,
    api_prefill,
    api_decode_m1,
    api_decode_mtp8,
    api_prefill_slot,
    api_step_continuous_m1,
    api_mtp_draft_block,
    api_notify_ubatch,
    api_forward_logits_f32,
    api_last_error,
};

} // namespace twf_detail

// ============================================================================
// 5. Exported C Bridge Functions
// ============================================================================

extern "C" {

const struct ggml_cuda_inkling2_api_t* ink2_core_get_api(void) {
    return &twf_detail::g_ink2_api;
}

int32_t ink2_core_op_block_attn_res(
    const float* d_cur_h, const float* d_queries, const float* const d_blks[5],
    float* d_out, int32_t n_blk, int32_t n_q, int32_t round_cur_bf16, int32_t m_tokens, cudaStream_t stream) {
    if (m_tokens <= 0 || n_q <= 0) return 0;
    twf_detail::Ink2BlkPtrs5 bp {};
    for (int i = 0; i < 5; ++i) bp.p[i] = (i < n_blk && d_blks != nullptr) ? d_blks[i] : d_cur_h;
    twf_detail::wv_ink_twf_block_attn_res_op_kernel<<<dim3(m_tokens), dim3(32, 6), 0, stream>>>(
        d_cur_h, d_queries, bp, d_out, n_blk, n_q, round_cur_bf16, m_tokens);
    return 0;
}

int32_t ink2_core_op_loop_msg_gate(
    const float* d_src, const float* d_msg, const void* d_ctrl_down_bf16,
    const float* d_ctrl_forget_f32, const float* d_ctrl_gain_f32,
    float* d_out, int32_t m_tokens, cudaStream_t stream) {
    if (m_tokens <= 0) return 0;
    twf_detail::wv_ink_twf_loop_msg_gate_op_kernel<<<dim3(m_tokens), 256, 0, stream>>>(
        d_src, d_msg, static_cast<const __nv_bfloat16*>(d_ctrl_down_bf16),
        d_ctrl_forget_f32, d_ctrl_gain_f32, d_out, m_tokens);
    return 0;
}

int32_t ink2_core_op_sconv_static(
    const float* d_x_in, const float* d_state_3_c_b, const float* d_w4_c,
    float* d_out, int32_t channels, int32_t s_tokens, int32_t b_seqs, cudaStream_t stream) {
    if (channels <= 0 || s_tokens <= 0 || b_seqs <= 0) return 0;
    dim3 grd((channels + 255) / 256, b_seqs);
    twf_detail::wv_ink_twf_sconv_static_op_kernel<<<grd, 256, 0, stream>>>(
        d_x_in, d_state_3_c_b, d_w4_c, d_out, channels, s_tokens, b_seqs);
    return 0;
}

int32_t ink2_core_op_sconv_dynamic(
    const float* d_x_in, const float* d_state_3_c_b, const float* d_static_w_c_4,
    const float* d_dw_32_s_b, float* d_out, int32_t s_tokens, int32_t b_seqs, cudaStream_t stream) {
    if (s_tokens <= 0 || b_seqs <= 0) return 0;
    twf_detail::wv_ink_twf_sconv_dynamic_op_kernel<<<dim3(4, b_seqs), 256, 0, stream>>>(
        d_x_in, d_state_3_c_b, d_static_w_c_4, d_dw_32_s_b, d_out, s_tokens, b_seqs);
    return 0;
}

int32_t ink2_core_op_prep_kv(
    const float* d_in, float* d_out, int32_t is_v, int32_t m_tokens, cudaStream_t stream) {
    if (m_tokens <= 0) return 0;
    twf_detail::wv_ink_twf_prep_kv_op_kernel<<<dim3(m_tokens), dim3(32, 8), 0, stream>>>(
        d_in, d_out, is_v, m_tokens);
    return 0;
}

int32_t ink2_core_op_attn_rel_sink(
    const float* d_q_raw, const void* d_k_cache, const void* d_v_cache,
    int32_t kv_ggml_type, int32_t n_kv, int32_t n_stream,
    size_t nb1_k, size_t nb2_k, size_t nb3_k,
    size_t nb1_v, size_t nb2_v, size_t nb3_v,
    const float* d_r_raw, const float* d_q_clip, const float* d_r_clip,
    const float* d_rel_proj_t, const float* d_sinks, const float* d_gate_raw,
    const int32_t* d_pos_pack, float* d_out, int32_t is_local, int32_t m_tokens, cudaStream_t stream) {
    if (m_tokens <= 0) return 0;
    twf_detail::wv_ink_twf_attn_rel_sink_op_kernel<<<dim3(m_tokens * 24), dim3(32, 4), 0, stream>>>(
        d_q_raw, static_cast<const uint8_t*>(d_k_cache), static_cast<const uint8_t*>(d_v_cache),
        kv_ggml_type, n_kv, n_stream, nb1_k, nb2_k, nb3_k, nb1_v, nb2_v, nb3_v,
        d_r_raw, d_q_clip, d_r_clip, d_rel_proj_t, d_sinks, d_gate_raw,
        d_pos_pack, d_out, is_local, m_tokens);
    return 0;
}

int32_t ink2_core_op_softcap_mask_head(
    const float* d_logits_in, float* d_logits_out,
    int32_t in_stride_vocab, int32_t out_vocab_full, int32_t vocab_unpad,
    float softcap, int32_t n_rows, cudaStream_t stream) {
    if (n_rows <= 0) return 0;
    dim3 grd((out_vocab_full + 255) / 256, n_rows);
    twf_detail::wv_ink_twf_softcap_mask_head_kernel<<<grd, 256, 0, stream>>>(
        d_logits_in, d_logits_out, in_stride_vocab, out_vocab_full, vocab_unpad, softcap, 1, n_rows);
    return 0;
}

int32_t ink2_core_op_mtp_stash_tap(
    void* handle, const float* d_src_f32, float* d_dst_f32,
    int32_t tap_slot, int32_t m_tokens, cudaStream_t stream) {
    if (m_tokens <= 0) return 0;
    auto* h = static_cast<twf_detail::Ink2CudaHandle*>(handle);
    __nv_bfloat16* taps_ptr = nullptr;
    if (h != nullptr && h->eng.has_mtp && m_tokens <= h->eng.max_m_cap) {
        cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
        (void) cudaStreamIsCapturing(stream, &cap_st);
        if (tap_slot == 0 && cap_st == cudaStreamCaptureStatusNone) {
            if (h->cur_needs_reset) {
                h->cur_needs_reset = false;
                h->eng.reset_cache();
            } else if (h->pending_spec_s > 0) {
                h->flush_pending_spec(h->cur_ubatch_pos0, stream);
            }
        }
        taps_ptr = h->eng.ws_taps.as<__nv_bfloat16>();
    }
    size_t tot = static_cast<size_t>(m_tokens) * wv::INK2_HIDDEN;
    twf_detail::wv_ink_twf_stash_f32_tap_kernel<<<static_cast<unsigned>((tot + 255) / 256), 256, 0, stream>>>(
        d_src_f32, d_dst_f32, taps_ptr, tap_slot, m_tokens);

    if (taps_ptr != nullptr && tap_slot == 3 && h->cur_ubatch_b == 1) {
        if (m_tokens >= 2 && m_tokens <= 8) {
            h->pending_spec_pos0  = h->cur_ubatch_pos0;
            h->pending_spec_s     = m_tokens;
            h->pending_spec_sconv = false;
            h->pending_spec_mtp   = true;
        } else {
            twf_detail::wv_ink_twf_set_i32_scalar_kernel<<<1, 32, 0, stream>>>(
                h->eng.d_mtp_pos0_t.as<int32_t>(), h->cur_ubatch_pos0);
            h->eng.commit_mtp_taps_gpu(taps_ptr, 1, m_tokens, nullptr, false, stream);
            int32_t next_mp0 = h->cur_ubatch_pos0 + m_tokens;
            twf_detail::wv_ink_twf_set_i32_scalar_kernel<<<1, 32, 0, stream>>>(
                h->eng.d_mtp_pos0_t.as<int32_t>(), next_mp0);
            h->eng.mtp_seen_tokens = next_mp0;
            h->eng.seen_tokens     = next_mp0;
        }
    }
    return 0;
}

int32_t ink2_core_op_fused_trunk(
    void* handle,
    const void* d_src0_tok_or_embd,
    int32_t has_tokens_src0,
    const int32_t* d_pos,
    const int32_t* d_seq_ids,
    int32_t single_seq_id,
    const int32_t* d_out_ids,
    int32_t b_act,
    int32_t s_tokens,
    int32_t n_outputs,
    float* d_logits_out,
    cudaStream_t stream) {
    if (!handle || !d_src0_tok_or_embd || !d_pos || b_act <= 0 || s_tokens <= 0) return -1;
    auto* h = static_cast<twf_detail::Ink2CudaHandle*>(handle);
    wv::WvInkEngine& eng = h->eng;
    wv::DeviceGuard guard(h->device_id);

    if (h->cur_needs_reset) {
        h->cur_needs_reset = false;
        h->pending_spec_s  = 0;
        eng.reset_cache();
    } else if (h->pending_spec_s > 0 && b_act == 1) {
        if (h->flush_pending_spec(h->cur_ubatch_pos0, stream) != 0) return -1;
    }

    bool pdl = eng.use_pdl && eng.dma.all_vram;

    // Fast Path 1: Steady-state B=1, S=1 token decode (`CUDA Graph` captured directly on `stream`!):
    if (b_act == 1 && s_tokens == 1 && single_seq_id == 0 && n_outputs == 1 &&
        d_out_ids == nullptr && has_tokens_src0 != 0 && eng.dma.all_vram && eng.cfg.enable_cuda_graphs != 0) {
        const int32_t* d_tok = static_cast<const int32_t*>(d_src0_tok_or_embd);
        auto emit_m1_body = [&](cudaStream_t s_cap) -> int32_t {
            bool cap_taps = eng.has_mtp;
            if (eng.forward_chunk(d_tok, 1, 1, d_pos, nullptr, cap_taps, 1, pdl, s_cap) != 0) return -1;
            wv::launch_wv_ink_swz8_linear<false, true>(
                eng.glob.head, nullptr, eng.ws_q8_a.as<int8_t>(), eng.ws_sc_a.as<float>(),
                eng.ws_logits.as<float>(), wv::INK2_HIDDEN, eng.cfg.vocab_pad, 1, s_cap, pdl);
            dim3 sm_grd((eng.cfg.vocab_full + 255) / 256, 1);
            twf_detail::wv_ink_twf_softcap_mask_head_kernel<<<sm_grd, 256, 0, s_cap>>>(
                eng.ws_logits.as<float>(), d_logits_out,
                eng.cfg.vocab_pad, eng.cfg.vocab_full, eng.cfg.vocab_unpad, 30.0f, 0, 1);
            if (cap_taps) {
                // Run after vocab head because `commit_mtp_taps_gpu` reuses `ws_q8_a` / `ws_sc_a`!
                WV_CUDA_CHECK(cudaMemcpyAsync(eng.d_mtp_pos0_t.ptr, d_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice, s_cap));
                eng.commit_mtp_taps_gpu(eng.ws_taps.as<__nv_bfloat16>(), 1, 1, nullptr, pdl, s_cap);
            }
            return 0;
        };

        if (!h->g_decode_m1.ready ||
            h->g_m1_tok_ptr != d_src0_tok_or_embd ||
            h->g_m1_pos_ptr != d_pos ||
            h->g_m1_dst_ptr != d_logits_out) {
            h->g_decode_m1.reset();
            cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(stream, &cap_st);
            if (cap_st == cudaStreamCaptureStatusNone) {
                WV_CUDA_CHECK(cudaStreamSynchronize(stream));
                WV_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                if (emit_m1_body(stream) != 0) {
                    cudaGraph_t tmp_g = nullptr;
                    cudaStreamEndCapture(stream, &tmp_g);
                    if (tmp_g) cudaGraphDestroy(tmp_g);
                    return -1;
                }
                WV_CUDA_CHECK(cudaStreamEndCapture(stream, &h->g_decode_m1.graph));
                WV_CUDA_CHECK(cudaGraphInstantiateWithFlags(&h->g_decode_m1.exec, h->g_decode_m1.graph, 0));
                h->g_decode_m1.ready = true;
                h->g_m1_tok_ptr = d_src0_tok_or_embd;
                h->g_m1_pos_ptr = d_pos;
                h->g_m1_dst_ptr = d_logits_out;
            } else {
                if (emit_m1_body(stream) != 0) return -1;
            }
        }
        if (h->g_decode_m1.ready) {
            WV_CUDA_CHECK(cudaGraphLaunch(h->g_decode_m1.exec, stream));
        }
        eng.seen_tokens     = h->cur_ubatch_pos0 + 1;
        eng.mtp_seen_tokens = eng.seen_tokens;
        if (!eng.h_slot_pos.empty()) eng.h_slot_pos[0] = eng.seen_tokens;
        return 0;
    }

    // Fast Path 2: Speculative verification (`B=1, 2 <= S <= 8`):
    if (b_act == 1 && single_seq_id == 0 && s_tokens >= 2 && s_tokens <= 8) {
        bool cap_taps = eng.has_mtp;
        int rc = (has_tokens_src0 != 0)
            ? eng.forward_chunk(static_cast<const int32_t*>(d_src0_tok_or_embd), 1, s_tokens, d_pos, nullptr, cap_taps, 0, pdl, stream)
            : h->forward_chunk_from_embd_f32(static_cast<const float*>(d_src0_tok_or_embd), 1, s_tokens, d_pos, nullptr, cap_taps, 0, pdl, stream);
        if (rc != 0) return -1;

        h->pending_spec_pos0  = h->cur_ubatch_pos0;
        h->pending_spec_s     = s_tokens;
        h->pending_spec_sconv = true;
        h->pending_spec_mtp   = cap_taps;

        if (n_outputs > 0 && d_logits_out != nullptr) {
            const int8_t* q8_src = eng.ws_q8_a.as<int8_t>();
            const float*  sc_src = eng.ws_sc_a.as<float>();
            if (d_out_ids != nullptr && n_outputs < s_tokens) {
                twf_detail::wv_ink_twf_gather_q8_rows_kernel<<<n_outputs, 256, 0, stream>>>(
                    q8_src, sc_src, d_out_ids, eng.ws_q8_b.as<int8_t>(), eng.ws_sc_b.as<float>(), s_tokens, n_outputs);
                q8_src = eng.ws_q8_b.as<int8_t>();
                sc_src = eng.ws_sc_b.as<float>();
            }
            wv::launch_wv_ink_swz8_linear<false, true>(
                eng.glob.head, nullptr, q8_src, sc_src,
                eng.ws_logits.as<float>(), wv::INK2_HIDDEN, eng.cfg.vocab_pad, n_outputs, stream, pdl);
            dim3 sm_grd((eng.cfg.vocab_full + 255) / 256, n_outputs);
            twf_detail::wv_ink_twf_softcap_mask_head_kernel<<<sm_grd, 256, 0, stream>>>(
                eng.ws_logits.as<float>(), d_logits_out,
                eng.cfg.vocab_pad, eng.cfg.vocab_full, eng.cfg.vocab_unpad, 30.0f, 0, n_outputs);
        }
        return 0;
    }

    // Fast Path 3: General Chunked TC Prefill / Multi-Sequence Batched Forward (`S = 1..32768`, `B = 1..b_max`):
    int step_cap = std::max(1, std::min(eng.max_chunk_s, eng.max_m_cap / b_act));
    const int32_t* d_smap = nullptr;
    if (b_act > 1 || single_seq_id > 0) {
        if (d_seq_ids != nullptr) {
            d_smap = d_seq_ids;
        } else if (single_seq_id >= 0) {
            twf_detail::wv_ink_twf_set_i32_scalar_kernel<<<1, 32, 0, stream>>>(eng.d_slot_map_t.as<int32_t>(), single_seq_id);
            d_smap = eng.d_slot_map_t.as<int32_t>();
        }
        if (!h->cur_seq_pos0.empty() && static_cast<int>(h->cur_seq_pos0.size()) >= b_act) {
            for (int b = 0; b < b_act; ++b) {
                int sid = (b < static_cast<int>(h->cur_seq_ids.size())) ? h->cur_seq_ids[b] : b;
                if (sid >= 0 && sid < eng.b_max) eng.h_slot_pos[sid] = h->cur_seq_pos0[b];
            }
            WV_CUDA_CHECK(cudaMemcpyAsync(
                eng.d_slot_pos_t.ptr, eng.h_slot_pos.data(),
                static_cast<size_t>(eng.b_max) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        }
    }

    bool do_mtp = eng.has_mtp && (b_act == 1) && (single_seq_id <= 0);
    for (int s0 = 0; s0 < s_tokens; s0 += step_cap) {
        int sc = std::min(step_cap, s_tokens - s0);
        const int32_t* chunk_pos = (b_act == 1 && single_seq_id <= 0) ? (d_pos + s0) : eng.d_slot_pos_t.as<int32_t>();

        if (has_tokens_src0 != 0) {
            const int32_t* d_tok_base = static_cast<const int32_t*>(d_src0_tok_or_embd);
            const int32_t* d_tok_ch   = d_tok_base + s0;
            if (b_act > 1 && s_tokens > sc) {
                for (int b = 0; b < b_act; ++b) {
                    WV_CUDA_CHECK(cudaMemcpyAsync(
                        eng.ws_tok_in.as<int32_t>() + static_cast<size_t>(b) * sc,
                        d_tok_base + static_cast<size_t>(b) * s_tokens + s0,
                        static_cast<size_t>(sc) * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
                }
                d_tok_ch = eng.ws_tok_in.as<int32_t>();
            }
            if (eng.forward_chunk(d_tok_ch, b_act, sc, chunk_pos, d_smap, do_mtp, 1, false, stream) != 0) return -1;
        } else {
            const float* d_emb_base = static_cast<const float*>(d_src0_tok_or_embd);
            const float* d_emb_ch   = d_emb_base + static_cast<size_t>(s0) * wv::INK2_HIDDEN;
            if (h->forward_chunk_from_embd_f32(d_emb_ch, b_act, sc, chunk_pos, d_smap, do_mtp, 1, false, stream) != 0) return -1;
        }

        // Emit logits BEFORE `commit_mtp_taps_gpu` (`commit_mtp_taps_gpu` reuses `ws_q8_a` / `ws_sc_a`!):
        if (s0 + sc == s_tokens && n_outputs > 0 && d_logits_out != nullptr) {
            int m_chunk = b_act * sc;
            const int8_t* q8_src = eng.ws_q8_a.as<int8_t>();
            const float*  sc_src = eng.ws_sc_a.as<float>();
            if (d_out_ids != nullptr && s0 == 0 && n_outputs <= m_chunk) {
                twf_detail::wv_ink_twf_gather_q8_rows_kernel<<<n_outputs, 256, 0, stream>>>(
                    q8_src, sc_src, d_out_ids, eng.ws_q8_b.as<int8_t>(), eng.ws_sc_b.as<float>(), m_chunk, n_outputs);
                q8_src = eng.ws_q8_b.as<int8_t>();
                sc_src = eng.ws_sc_b.as<float>();
            } else if (s0 > 0 && b_act == 1 && n_outputs == 1) {
                // Multi-chunk prefill (`S > max_chunk_s`) with last-token output:
                q8_src += static_cast<size_t>(sc - 1) * wv::INK2_HIDDEN;
                sc_src += static_cast<size_t>(sc - 1) * 72;
            }
            for (int o0 = 0; o0 < n_outputs; o0 += 32) {
                int ob = std::min(32, n_outputs - o0);
                wv::launch_wv_ink_swz8_linear<false, true>(
                    eng.glob.head, nullptr,
                    q8_src + static_cast<size_t>(o0) * wv::INK2_HIDDEN,
                    sc_src + static_cast<size_t>(o0) * 72,
                    eng.ws_logits.as<float>(), wv::INK2_HIDDEN, eng.cfg.vocab_pad, ob, stream, false);
                dim3 sm_grd((eng.cfg.vocab_full + 255) / 256, ob);
                twf_detail::wv_ink_twf_softcap_mask_head_kernel<<<sm_grd, 256, 0, stream>>>(
                    eng.ws_logits.as<float>(),
                    d_logits_out + static_cast<size_t>(o0) * eng.cfg.vocab_full,
                    eng.cfg.vocab_pad, eng.cfg.vocab_full, eng.cfg.vocab_unpad, 30.0f, 0, ob);
            }
        }

        if (do_mtp) {
            WV_CUDA_CHECK(cudaMemcpyAsync(eng.d_mtp_pos0_t.ptr, d_pos + s0, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
            eng.commit_mtp_taps_gpu(eng.ws_taps.as<__nv_bfloat16>(), 1, sc, nullptr, false, stream);
            eng.mtp_seen_tokens = h->cur_ubatch_pos0 + s0 + sc;
            twf_detail::wv_ink_twf_set_i32_scalar_kernel<<<1, 32, 0, stream>>>(
                eng.d_mtp_pos0_t.as<int32_t>(), eng.mtp_seen_tokens);
        }

        if (b_act > 1 || single_seq_id > 0) {
            for (int b = 0; b < b_act; ++b) {
                int sid = (b < static_cast<int>(h->cur_seq_ids.size())) ? h->cur_seq_ids[b] : b;
                if (sid >= 0 && sid < eng.b_max) eng.h_slot_pos[sid] += sc;
            }
            if (s0 + sc < s_tokens) {
                WV_CUDA_CHECK(cudaMemcpyAsync(
                    eng.d_slot_pos_t.ptr, eng.h_slot_pos.data(),
                    static_cast<size_t>(eng.b_max) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            }
        } else {
            eng.seen_tokens = h->cur_ubatch_pos0 + s0 + sc;
            if (!eng.h_slot_pos.empty()) eng.h_slot_pos[0] = eng.seen_tokens;
        }
    }
    return 0;
}

} // extern "C"

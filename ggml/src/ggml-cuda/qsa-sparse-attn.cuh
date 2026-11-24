#pragma once

#include "common.cuh"

bool ggml_cuda_qsa_fused_enabled();
void ggml_cuda_set_qsa_fused(bool on);

bool ggml_cuda_hc_fused_enabled();
void ggml_cuda_set_hc_fused(bool on);

bool ggml_cuda_ple_fast_enabled();
void ggml_cuda_set_ple_fast(bool on);

// Builds sorted compact cell indices [n_sel, n_tokens] (int32, -1 sentinel for inactive slots)
// directly from (top_k, score, pool_idxs, tail_idxs) without allocating an [n_kv, n_tokens] mask.
void ggml_cuda_qsa_build_cells(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         top_k,
        const ggml_tensor *         score,
        const ggml_tensor *         pool_idxs,
        const ggml_tensor *         tail_idxs,
        int32_t                     n_kv,
        ggml_tensor *               dst_cells);

bool ggml_cuda_qsa_sparse_attn_supported(
        const ggml_tensor * q,
        const ggml_tensor * k,
        const ggml_tensor * v,
        const ggml_tensor * sel_cells);

void ggml_cuda_qsa_sparse_attn(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         q,
        const ggml_tensor *         k,
        const ggml_tensor *         v,
        const ggml_tensor *         sel_cells,
        ggml_tensor *               dst,
        float                       scale);

bool ggml_cuda_hc_up_pre_supported(
        const ggml_tensor * up_mm,
        const ggml_tensor * hc_pre);

void ggml_cuda_hc_up_pre_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         up_mm,
        ggml_tensor *               hc_pre);

void ggml_cuda_ple_gather_dequant(
        ggml_backend_cuda_context & ctx,
        const void *                uniq_raw_d,
        ggml_type                   ple_type,
        const int32_t *             map_idx_d,
        float *                     dst_d,
        int                         head_dim,
        int                         n_rows_out);

bool ggml_cuda_ple_conv_silu_supported(
        const ggml_cgraph *  cgraph,
        int                  node_idx,
        const ggml_tensor ** out_padded,
        const ggml_tensor ** out_conv_w,
        int *                out_kern,
        int *                out_dil,
        int *                out_span);

void ggml_cuda_ple_dilated_conv_silu(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         padded,
        const ggml_tensor *         conv_w,
        ggml_tensor *               dst,
        int                         kern,
        int                         dil);

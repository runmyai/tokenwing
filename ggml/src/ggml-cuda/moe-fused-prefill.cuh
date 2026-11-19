#pragma once

#include "common.cuh"

bool ggml_cuda_mmq_moe_opt_enabled();
void ggml_cuda_set_mmq_moe_opt(bool on);

bool ggml_cuda_moe_fused_prefill_enabled();
void ggml_cuda_set_moe_fused_prefill(bool on);

bool ggml_cuda_moe_fused_prefill_supported(
        const ggml_tensor * up,
        const ggml_tensor * gate,
        const ggml_tensor * glu,
        const ggml_tensor * down,
        int                 device);

void ggml_cuda_moe_fused_prefill(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         up,
        const ggml_tensor *         gate,
        const ggml_tensor *         glu,
        ggml_tensor *               down,
        void (*mark_after_gu)(void *) = nullptr,
        void *                      mark_ud = nullptr);

bool ggml_cuda_moe_weighted_sum_supported(
        const ggml_cgraph * cgraph,
        int                 node_idx,
        int *               out_n_nodes);

void ggml_cuda_moe_weighted_sum(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         experts,
        const ggml_tensor *         weights,
        ggml_tensor *               dst,
        int                         n_expert_used);

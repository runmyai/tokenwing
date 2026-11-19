#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

// --moe-stream-gpu-wave-skip-zero: after mm_ids_helper, the pairs of the last n_zero experts (all-zero slices) get 0
// in their output rows and an empty column range, so the matmul does not compute them (the multi-token form of --moe-stream-gpu-skip-zero's mmvq skip). The
// maps (ids_src1 / ids_dst) stay complete, so the input quantization reads only valid entries
int  ggml_cuda_moe_zero_tail_of(const ggml_tensor * dst);   // n_zero from the node's op_params, 0 = none
void ggml_cuda_moe_zero_tail_apply(int32_t * expert_bounds, const int32_t * ids_dst, float * dst, int n_experts, int n_zero, int ne0,
        int64_t stride_col_dst, int n_cols_max, cudaStream_t stream, bool skip_zero_rows = false, bool save_orig_end = false);
void ggml_cuda_moe_zero_tail_rows_cached(const int32_t * expert_bounds, const int32_t * ids_dst, float * dst, int n_experts, int n_zero,
        int ne0, int64_t stride_col_dst, int n_cols_max, cudaStream_t stream);

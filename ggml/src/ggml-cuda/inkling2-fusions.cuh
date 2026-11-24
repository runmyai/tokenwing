#pragma once

#include "common.cuh"
#include "ggml-inkling2-cuda.h"

bool ggml_cuda_inkling2_supports_op(const ggml_tensor * op);
void ggml_cuda_inkling2_compute_forward(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
const struct ggml_cuda_inkling2_api_t * ggml_cuda_inkling2_api(void);

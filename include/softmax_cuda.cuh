#pragma once

#include <cuda_runtime_api.h>

void launch_softmax_basic(
    const float* device_input,
    float* device_output,
    int rows,
    int cols,
    cudaStream_t stream = nullptr
);

void launch_softmax_optimized(
    const float* device_input,
    float* device_output,
    int rows,
    int cols,
    int threads_per_block,
    cudaStream_t stream = nullptr
);

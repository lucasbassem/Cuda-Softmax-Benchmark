#include "softmax_cuda.cuh"
#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace {

__global__ void softmax_basic_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const std::size_t offset =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
    float row_max = -CUDART_INF_F;
    for (int col = 0; col < cols; ++col) {
        row_max = fmaxf(row_max, input[offset + col]);
    }

    float denominator = 0.0F;
    for (int col = 0; col < cols; ++col) {
        const float value = expf(input[offset + col] - row_max);
        output[offset + col] = value;
        denominator += value;
    }

    const float inverse_denominator = 1.0F / denominator;
    for (int col = 0; col < cols; ++col) {
        output[offset + col] *= inverse_denominator;
    }
}

__global__ void softmax_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    extern __shared__ float shared[];

    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int lane = threadIdx.x;
    const std::size_t offset =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);

    float local_max = -CUDART_INF_F;
    for (int col = lane; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, input[offset + col]);
    }

    shared[lane] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared[lane] = fmaxf(shared[lane], shared[lane + stride]);
        }
        __syncthreads();
    }

    const float row_max = shared[0];
    float local_sum = 0.0F;
    for (int col = lane; col < cols; col += blockDim.x) {
        const float value = expf(input[offset + col] - row_max);
        output[offset + col] = value;
        local_sum += value;
    }

    shared[lane] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared[lane] += shared[lane + stride];
        }
        __syncthreads();
    }

    const float inverse_denominator = 1.0F / shared[0];
    for (int col = lane; col < cols; col += blockDim.x) {
        output[offset + col] *= inverse_denominator;
    }
}

bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

}  // namespace

void launch_softmax_basic(
    const float* device_input,
    float* device_output,
    int rows,
    int cols,
    cudaStream_t stream
) {
    constexpr int threads = 128;
    const int blocks = (rows + threads - 1) / threads;
    softmax_basic_kernel<<<blocks, threads, 0, stream>>>(device_input, device_output, rows, cols);
}

void launch_softmax_optimized(
    const float* device_input,
    float* device_output,
    int rows,
    int cols,
    int threads_per_block,
    cudaStream_t stream
) {
    if (!is_power_of_two(threads_per_block) || threads_per_block < 32 || threads_per_block > 1024) {
        throw std::invalid_argument(
            "threads_per_block must be a power of two between 32 and 1024"
        );
    }

    const std::size_t shared_bytes =
        static_cast<std::size_t>(threads_per_block) * sizeof(float);
    softmax_optimized_kernel<<<rows, threads_per_block, shared_bytes, stream>>>(
        device_input,
        device_output,
        rows,
        cols
    );
}

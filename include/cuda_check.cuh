#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

inline void cuda_check_impl(cudaError_t status, const char* expression, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::ostringstream message;
    message << "CUDA call failed: " << expression << " at " << file << ':' << line
            << " (" << cudaGetErrorName(status) << ": " << cudaGetErrorString(status) << ')';
    throw std::runtime_error(message.str());
}

#define CUDA_CHECK(expression) cuda_check_impl((expression), #expression, __FILE__, __LINE__)

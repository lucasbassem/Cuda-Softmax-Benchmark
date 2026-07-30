#pragma once

#include <cstddef>
#include <vector>

struct ErrorMetrics {
    double max_absolute_error{};
    double max_relative_error{};
    double max_row_sum_error{};
};

void softmax_cpu(const float* input, float* output, std::size_t rows, std::size_t cols);

ErrorMetrics compare_softmax(
    const std::vector<float>& reference,
    const std::vector<float>& candidate,
    std::size_t rows,
    std::size_t cols,
    double relative_epsilon = 1.0e-8
);

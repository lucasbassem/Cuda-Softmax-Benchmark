#include "softmax.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

void softmax_cpu(const float* input, float* output, std::size_t rows, std::size_t cols) {
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("softmax_cpu received a null pointer");
    }
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("softmax_cpu requires positive rows and columns");
    }

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t offset = row * cols;
        float row_max = -std::numeric_limits<float>::infinity();

        for (std::size_t col = 0; col < cols; ++col) {
            row_max = std::max(row_max, input[offset + col]);
        }

        double denominator = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            const double exponent = std::exp(
                static_cast<double>(input[offset + col]) - static_cast<double>(row_max)
            );
            output[offset + col] = static_cast<float>(exponent);
            denominator += exponent;
        }

        const double inverse_denominator = 1.0 / denominator;
        for (std::size_t col = 0; col < cols; ++col) {
            output[offset + col] = static_cast<float>(
                static_cast<double>(output[offset + col]) * inverse_denominator
            );
        }
    }
}

ErrorMetrics compare_softmax(
    const std::vector<float>& reference,
    const std::vector<float>& candidate,
    std::size_t rows,
    std::size_t cols,
    double relative_epsilon
) {
    if (reference.size() != candidate.size()) {
        throw std::invalid_argument("Reference and candidate sizes differ");
    }
    if (reference.size() != rows * cols) {
        throw std::invalid_argument("Tensor shape does not match vector size");
    }

    ErrorMetrics metrics{};

    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double expected = static_cast<double>(reference[index]);
        const double actual = static_cast<double>(candidate[index]);
        const double absolute_error = std::abs(expected - actual);
        const double relative_error = absolute_error / std::max(std::abs(expected), relative_epsilon);

        metrics.max_absolute_error = std::max(metrics.max_absolute_error, absolute_error);
        metrics.max_relative_error = std::max(metrics.max_relative_error, relative_error);
    }

    for (std::size_t row = 0; row < rows; ++row) {
        double row_sum = 0.0;
        for (std::size_t col = 0; col < cols; ++col) {
            row_sum += static_cast<double>(candidate[row * cols + col]);
        }
        metrics.max_row_sum_error = std::max(metrics.max_row_sum_error, std::abs(row_sum - 1.0));
    }

    return metrics;
}

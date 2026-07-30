#include "softmax.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const std::vector<float> input{
        10000.0F,
        9999.0F,
        -10000.0F,
        -1.0F,
        0.0F,
        1.0F,
    };
    std::vector<float> output(input.size());
    softmax_cpu(input.data(), output.data(), 2, 3);

    for (int row = 0; row < 2; ++row) {
        double row_sum = 0.0;
        for (int col = 0; col < 3; ++col) {
            const float value = output[static_cast<std::size_t>(row * 3 + col)];
            if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
                std::cerr << "Invalid probability encountered\n";
                return 1;
            }
            row_sum += static_cast<double>(value);
        }

        if (std::abs(row_sum - 1.0) > 1.0e-6) {
            std::cerr << "Row does not sum to one\n";
            return 2;
        }
    }

    std::cout << "CPU softmax validation passed\n";
    return 0;
}

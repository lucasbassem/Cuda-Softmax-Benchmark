#include "cuda_check.cuh"
#include "softmax.hpp"
#include "softmax_cuda.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Shape {
    int rows{};
    int cols{};
};

struct Options {
    std::vector<Shape> shapes{{1024, 128}, {1024, 512}, {4096, 1024}, {8192, 2048}};
    int warmup_iterations{20};
    int benchmark_iterations{100};
    int threads_per_block{256};
    std::uint32_t seed{2026};
    std::string output_path{"results/benchmark_results.csv"};
};

struct DeviceBuffer {
    float* pointer{nullptr};

    explicit DeviceBuffer(std::size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(float)));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : pointer(other.pointer) {
        other.pointer = nullptr;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (pointer != nullptr) {
                cudaFree(pointer);
            }
            pointer = other.pointer;
            other.pointer = nullptr;
        }
        return *this;
    }

    ~DeviceBuffer() {
        if (pointer != nullptr) {
            cudaFree(pointer);
        }
    }
};

struct EventPair {
    cudaEvent_t start{};
    cudaEvent_t stop{};

    EventPair() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;

    ~EventPair() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

struct BenchmarkRow {
    int rows{};
    int cols{};
    std::size_t elements{};
    double cpu_ms{};
    double basic_cuda_ms{};
    double optimized_cuda_ms{};
    double basic_speedup_vs_cpu{};
    double optimized_speedup_vs_cpu{};
    double optimized_speedup_vs_basic{};
    ErrorMetrics basic_error{};
    ErrorMetrics optimized_error{};
};

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        if (!item.empty()) {
            parts.push_back(item);
        }
    }
    return parts;
}

Shape parse_shape(const std::string& text) {
    const auto separator = text.find('x');
    if (separator == std::string::npos) {
        throw std::invalid_argument("Shape must use ROWSxCOLS format: " + text);
    }

    const int rows = std::stoi(text.substr(0, separator));
    const int cols = std::stoi(text.substr(separator + 1));
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Shape dimensions must be positive: " + text);
    }
    return Shape{rows, cols};
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Options:\n"
        << "  --sizes ROWSxCOLS,...   Tensor sizes (default: 1024x128,1024x512,4096x1024,8192x2048)\n"
        << "  --warmup N              Warmup launches per CUDA kernel (default: 20)\n"
        << "  --iterations N          Timed iterations (default: 100)\n"
        << "  --threads N             Optimized-kernel block size, power of two [32,1024] (default: 256)\n"
        << "  --seed N                Random seed (default: 2026)\n"
        << "  --output PATH           CSV output path (default: results/benchmark_results.csv)\n"
        << "  --help                  Show this help message\n";
}

Options parse_options(int argc, char** argv) {
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto require_value = [&](const std::string& flag) -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + flag);
            }
            return argv[++index];
        };

        if (argument == "--sizes") {
            options.shapes.clear();
            for (const auto& shape_text : split(require_value(argument), ',')) {
                options.shapes.push_back(parse_shape(shape_text));
            }
            if (options.shapes.empty()) {
                throw std::invalid_argument("--sizes must contain at least one shape");
            }
        } else if (argument == "--warmup") {
            options.warmup_iterations = std::stoi(require_value(argument));
        } else if (argument == "--iterations") {
            options.benchmark_iterations = std::stoi(require_value(argument));
        } else if (argument == "--threads") {
            options.threads_per_block = std::stoi(require_value(argument));
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(require_value(argument)));
        } else if (argument == "--output") {
            options.output_path = require_value(argument);
        } else if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }

    if (options.warmup_iterations < 0 || options.benchmark_iterations <= 0) {
        throw std::invalid_argument("Warmup must be non-negative and iterations must be positive");
    }

    return options;
}

std::string sanitize_csv_field(std::string value) {
    std::replace(value.begin(), value.end(), ',', ';');
    std::replace(value.begin(), value.end(), '\n', ' ');
    std::replace(value.begin(), value.end(), '\r', ' ');
    return value;
}

std::string cuda_runtime_version() {
    int version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&version));
    std::ostringstream text;
    text << version / 1000 << '.' << (version % 1000) / 10;
    return text.str();
}

std::string cuda_driver_version() {
    int version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&version));
    std::ostringstream text;
    text << version / 1000 << '.' << (version % 1000) / 10;
    return text.str();
}

std::vector<float> generate_input(std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0F, 3.0F);
    std::vector<float> input(count);
    for (float& value : input) {
        value = distribution(generator);
    }
    return input;
}

double benchmark_cpu(
    const std::vector<float>& input,
    std::vector<float>& output,
    std::size_t rows,
    std::size_t cols,
    int iterations
) {
    const int cpu_iterations = std::max(1, std::min(iterations, 10));
    softmax_cpu(input.data(), output.data(), rows, cols);

    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < cpu_iterations; ++iteration) {
        softmax_cpu(input.data(), output.data(), rows, cols);
    }
    const auto stop = std::chrono::steady_clock::now();

    const auto elapsed = std::chrono::duration<double, std::milli>(stop - start).count();
    return elapsed / static_cast<double>(cpu_iterations);
}

template <typename LaunchFunction>
double benchmark_cuda_kernel(
    LaunchFunction&& launch,
    int warmup_iterations,
    int benchmark_iterations
) {
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    EventPair events;
    CUDA_CHECK(cudaEventRecord(events.start));
    for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(events.stop));
    CUDA_CHECK(cudaEventSynchronize(events.stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, events.start, events.stop));
    return static_cast<double>(elapsed_ms) / static_cast<double>(benchmark_iterations);
}

void copy_device_output(
    const DeviceBuffer& device_output,
    std::vector<float>& host_output
) {
    CUDA_CHECK(cudaMemcpy(
        host_output.data(),
        device_output.pointer,
        host_output.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
}

BenchmarkRow run_shape(
    const Shape& shape,
    const Options& options,
    std::uint32_t shape_seed
) {
    const std::size_t rows = static_cast<std::size_t>(shape.rows);
    const std::size_t cols = static_cast<std::size_t>(shape.cols);
    if (rows > std::numeric_limits<std::size_t>::max() / cols) {
        throw std::overflow_error("Tensor shape is too large");
    }
    const std::size_t elements = rows * cols;

    auto input = generate_input(elements, shape_seed);
    std::vector<float> cpu_output(elements);
    std::vector<float> basic_output(elements);
    std::vector<float> optimized_output(elements);

    const double cpu_ms = benchmark_cpu(
        input,
        cpu_output,
        rows,
        cols,
        options.benchmark_iterations
    );

    DeviceBuffer device_input(elements);
    DeviceBuffer device_output(elements);
    CUDA_CHECK(cudaMemcpy(
        device_input.pointer,
        input.data(),
        elements * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    const auto basic_launch = [&]() {
        launch_softmax_basic(
            device_input.pointer,
            device_output.pointer,
            shape.rows,
            shape.cols
        );
    };
    const double basic_ms = benchmark_cuda_kernel(
        basic_launch,
        options.warmup_iterations,
        options.benchmark_iterations
    );
    copy_device_output(device_output, basic_output);

    const auto optimized_launch = [&]() {
        launch_softmax_optimized(
            device_input.pointer,
            device_output.pointer,
            shape.rows,
            shape.cols,
            options.threads_per_block
        );
    };
    const double optimized_ms = benchmark_cuda_kernel(
        optimized_launch,
        options.warmup_iterations,
        options.benchmark_iterations
    );
    copy_device_output(device_output, optimized_output);

    return BenchmarkRow{
        shape.rows,
        shape.cols,
        elements,
        cpu_ms,
        basic_ms,
        optimized_ms,
        cpu_ms / basic_ms,
        cpu_ms / optimized_ms,
        basic_ms / optimized_ms,
        compare_softmax(cpu_output, basic_output, rows, cols),
        compare_softmax(cpu_output, optimized_output, rows, cols),
    };
}

void print_row(const BenchmarkRow& row) {
    std::cout << std::fixed << std::setprecision(6)
              << row.rows << 'x' << row.cols
              << " | CPU " << row.cpu_ms << " ms"
              << " | basic " << row.basic_cuda_ms << " ms"
              << " | optimized " << row.optimized_cuda_ms << " ms"
              << " | opt/basic " << row.optimized_speedup_vs_basic << "x"
              << " | opt max abs " << row.optimized_error.max_absolute_error
              << " | opt max rel " << row.optimized_error.max_relative_error
              << '\n';
}

void write_csv(
    const std::string& path,
    const cudaDeviceProp& properties,
    const Options& options,
    const std::vector<BenchmarkRow>& rows
) {
    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not open output file: " + path);
    }

    output << "gpu,cuda_runtime,cuda_driver,compute_capability,fast_math,threads_per_block,"
              "warmup_iterations,benchmark_iterations,rows,cols,elements,cpu_ms,basic_cuda_ms,"
              "optimized_cuda_ms,basic_speedup_vs_cpu,optimized_speedup_vs_cpu,"
              "optimized_speedup_vs_basic,basic_max_abs_error,basic_max_rel_error,"
              "basic_max_row_sum_error,optimized_max_abs_error,optimized_max_rel_error,"
              "optimized_max_row_sum_error\n";

    output << std::setprecision(10);
    for (const auto& row : rows) {
        output
            << sanitize_csv_field(properties.name) << ','
            << cuda_runtime_version() << ','
            << cuda_driver_version() << ','
            << properties.major << '.' << properties.minor << ','
#ifdef SOFTMAX_FAST_MATH
            << "true,"
#else
            << "false,"
#endif
            << options.threads_per_block << ','
            << options.warmup_iterations << ','
            << options.benchmark_iterations << ','
            << row.rows << ','
            << row.cols << ','
            << row.elements << ','
            << row.cpu_ms << ','
            << row.basic_cuda_ms << ','
            << row.optimized_cuda_ms << ','
            << row.basic_speedup_vs_cpu << ','
            << row.optimized_speedup_vs_cpu << ','
            << row.optimized_speedup_vs_basic << ','
            << row.basic_error.max_absolute_error << ','
            << row.basic_error.max_relative_error << ','
            << row.basic_error.max_row_sum_error << ','
            << row.optimized_error.max_absolute_error << ','
            << row.optimized_error.max_relative_error << ','
            << row.optimized_error.max_row_sum_error << '\n';
    }
}

void verify_accuracy(const std::vector<BenchmarkRow>& rows) {
    constexpr double absolute_tolerance = 2.0e-5;
    constexpr double row_sum_tolerance = 2.0e-5;

    for (const auto& row : rows) {
        const bool basic_failed =
            row.basic_error.max_absolute_error > absolute_tolerance ||
            row.basic_error.max_row_sum_error > row_sum_tolerance;
        const bool optimized_failed =
            row.optimized_error.max_absolute_error > absolute_tolerance ||
            row.optimized_error.max_row_sum_error > row_sum_tolerance;

        if (basic_failed || optimized_failed) {
            std::ostringstream message;
            message << "Accuracy validation failed for " << row.rows << 'x' << row.cols
                    << ". Inspect the CSV for error metrics.";
            throw std::runtime_error(message.str());
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw std::runtime_error("No CUDA-capable GPU was detected");
        }

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        std::cout << "GPU: " << properties.name << " (compute capability "
                  << properties.major << '.' << properties.minor << ")\n"
                  << "CUDA runtime: " << cuda_runtime_version()
                  << " | driver: " << cuda_driver_version() << '\n'
                  << "Benchmarking " << options.shapes.size() << " tensor shapes...\n";

        std::vector<BenchmarkRow> benchmark_rows;
        benchmark_rows.reserve(options.shapes.size());

        for (std::size_t index = 0; index < options.shapes.size(); ++index) {
            const auto row = run_shape(
                options.shapes[index],
                options,
                options.seed + static_cast<std::uint32_t>(index)
            );
            benchmark_rows.push_back(row);
            print_row(row);
        }

        verify_accuracy(benchmark_rows);
        write_csv(options.output_path, properties, options, benchmark_rows);

        std::cout << "Accuracy checks passed. Results written to "
                  << options.output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

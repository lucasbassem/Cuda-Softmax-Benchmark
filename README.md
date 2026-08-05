# GPU Softmax Kernel and Inference Benchmark

[![CPU validation](https://github.com/lucasbassem/Cuda-Softmax-Benchmark/actions/workflows/cpu-validation.yml/badge.svg)](https://github.com/lucasbassem/Cuda-Softmax-Benchmark/actions/workflows/cpu-validation.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![CUDA](https://img.shields.io/badge/CUDA-C%2B%2B-76B900)
![License](https://img.shields.io/badge/license-MIT-green)

A reproducible GPU-performance project that implements numerically stable
row-wise softmax in C++ and CUDA, validates correctness against independent
references, and measures both custom-kernel and framework-level inference
performance.

## Verified results

### CUDA kernel optimization

On an NVIDIA Tesla T4, the optimized shared-memory kernel reduced latency from
**0.626 ms to 0.050 ms** for a `1024 × 512` tensor:

- **12.44× faster** than the basic CUDA kernel
- **97.60× faster** than the serial C++ reference on the same workload
- Maximum absolute error: **2.38e-7**

Across the full tested suite, the optimized kernel reached up to **230.88×**
speedup over the serial CPU reference. The maximum CPU and basic-CUDA speedups
occur at different shapes and are reported separately.

![CUDA softmax speedup](results/speedup.png)

| Shape | Serial CPU | Basic CUDA | Optimized CUDA | Optimized vs CPU | Optimized vs basic |
|---|---:|---:|---:|---:|---:|
| `1024 × 128` | 1.189 ms | 0.147 ms | 0.040 ms | 30.06× | 3.72× |
| `1024 × 512` | 4.911 ms | 0.626 ms | 0.050 ms | 97.60× | **12.44×** |
| `4096 × 1024` | 40.537 ms | 1.888 ms | 0.193 ms | 210.06× | 9.78× |
| `8192 × 2048` | 162.082 ms | 4.820 ms | 0.702 ms | **230.88×** | 6.87× |

[Raw CUDA data](results/benchmark_results.csv) ·
[Full CUDA report](results/RESULTS.md)

### ONNX Runtime CUDA inference

A pretrained ResNet-18 model was exported from PyTorch to ONNX and benchmarked
with GPU-resident input/output through ONNX Runtime I/O binding.

At batch size 1, ONNX Runtime reduced median latency from **3.858 ms to
2.105 ms**, a **45.4% reduction**, and increased throughput from **259.2 to
475.1 images/second**, a **1.83× improvement**.

Performance was effectively tied at batch size 8, and PyTorch was approximately
0.9% faster at batch size 32. The project therefore reports the batch-size-1
result rather than implying a universal ONNX speedup.

![ONNX Runtime comparison](results/onnx_runtime_comparison.png)

| Batch | PyTorch median | ONNX median | ONNX speedup | PyTorch images/s | ONNX images/s | Max abs. error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.858 ms | 2.105 ms | **1.833×** | 259.2 | 475.1 | 6e-6 |
| 8 | 9.184 ms | 9.117 ms | 1.007× | 871.1 | 877.5 | 8e-6 |
| 32 | 26.023 ms | 26.266 ms | 0.991× | 1229.7 | 1218.3 | 1.3e-5 |

[Raw ONNX data](results/onnx_runtime_cuda_benchmark.csv) ·
[Full ONNX report](results/ONNX_RESULTS.md)

## What was implemented

### Serial C++ reference

`src/cpu_softmax.cpp` computes stable row-wise softmax by subtracting each row
maximum before exponentiation. It uses double-precision intermediates for the
denominator and serves as the numerical reference for both CUDA kernels.

### Basic CUDA kernel

`softmax_basic_kernel` assigns one GPU thread to each row. The thread performs
the maximum, exponential sum, and normalization sequentially. This is a valid
but intentionally under-parallelized GPU baseline.

### Optimized CUDA kernel

`softmax_optimized_kernel` assigns one block to each row:

1. Threads traverse columns using a strided access pattern.
2. Shared-memory reduction computes the row maximum.
3. Threads exponentiate assigned elements and accumulate partial sums.
4. A second shared-memory reduction computes the denominator.
5. Threads normalize their assigned outputs.

The launch wrapper validates a power-of-two block size from 32 through 1024
threads.

### Framework inference benchmark

`python/onnx_runtime_benchmark.py` exports ResNet-18 with a dynamic batch
dimension, validates the ONNX graph, confirms that CUDA is the primary ONNX
Runtime provider, benchmarks PyTorch and ONNX Runtime, and records numerical
error.

## Measurement methodology

### CUDA softmax

- GPU: NVIDIA Tesla T4
- CUDA runtime: 12.8
- CUDA driver API version: 13.0
- Compute capability: 7.5
- Build: fast math enabled
- Threads per block: 256
- Warm-up iterations: 20
- Timed iterations: 100
- Timing: CUDA events
- Host-to-device and device-to-host transfers excluded from kernel latency
- Correctness metrics: maximum absolute error, relative error, and row-sum error

The CPU comparison uses a **single-threaded reference implementation**. It is
included as a correctness baseline and a transparent serial comparison, not as
a claim against an optimized multithreaded CPU library.

### ONNX Runtime

- Model: pretrained torchvision ResNet-18
- Input: FP32, `[batch, 3, 224, 224]`
- Batch sizes: 1, 8, and 32
- Warm-up iterations: 10
- Timed iterations: 50
- PyTorch timing: CUDA events
- ONNX Runtime timing: synchronized wall-clock measurement
- ONNX input/output: GPU-resident through I/O binding
- Correctness: maximum absolute output difference from PyTorch

The original ONNX run preserved the measurements and active CUDA provider but
not complete package and driver metadata. Rerunning the included script writes
`results/onnx_runtime_environment.json`.

## Build

### Requirements

- NVIDIA GPU with a CUDA-capable driver
- CUDA Toolkit with `nvcc`
- CMake 3.24+
- Ninja or another CMake generator
- C++17 compiler supported by the CUDA Toolkit
- Python 3.10+

### Linux or WSL2

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cmake --preset release
cmake --build --preset release
```

### Windows PowerShell

Use a Visual Studio Developer PowerShell with the CUDA Toolkit on `PATH`:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cmake --preset release
cmake --build --preset release
```

## Run

Run the standard benchmark suite:

```bash
python scripts/run_benchmarks.py
```

Reproduce the published fast-math CUDA configuration:

```bash
python scripts/run_benchmarks.py \
  --preset release-fast-math \
  --sizes 1024x128,1024x512,4096x1024,8192x2048 \
  --warmup 20 \
  --iterations 100 \
  --threads 256
```

Run the ONNX Runtime benchmark:

```bash
python python/onnx_runtime_benchmark.py \
  --output-dir results \
  --batch-sizes 1 8 32 \
  --warmup 10 \
  --iterations 50
```

## Tests

CPU and Python validation:

```bash
pytest -q

g++ -std=c++17 -Wall -Wextra -Wpedantic \
  -Iinclude src/cpu_softmax.cpp tests/cpu_softmax_test.cpp \
  -o cpu_softmax_test

./cpu_softmax_test
```

CUDA smoke test after building:

```bash
ctest --preset release
```

Deeper memory and synchronization checking:

```bash
compute-sanitizer ./build/release/softmax_benchmark \
  --sizes 16x31,32x257 \
  --warmup 1 \
  --iterations 2
```

GitHub Actions runs Python tests, validates the committed benchmark tables, and
compiles/runs the CPU reference test on each push and pull request. GPU
benchmarks remain hardware-executed because standard hosted runners do not
provide a CUDA GPU.

## Repository structure

```text
.
├── .github/workflows/       # CPU and result-integrity CI
├── include/                 # C++ and CUDA interfaces
├── python/                  # PyTorch, ONNX, plotting, and report tools
├── results/                 # Published raw data, reports, and figures
├── scripts/                 # End-to-end benchmark orchestration
├── src/                     # CPU reference, CUDA kernels, benchmark CLI
├── tests/                   # CPU, Python, and published-result tests
├── CMakeLists.txt
├── CMakePresets.json
└── requirements.txt
```

## Current scope and next experiments

The optimized kernel is intentionally understandable rather than
architecture-specialized. Useful extensions include:

- Warp-shuffle reductions
- Vectorized loads for aligned column counts
- Shape-specialized kernels
- FP16 or BF16 input with FP32 accumulation
- Effective-memory-bandwidth and occupancy analysis with Nsight Compute
- PyTorch custom-operator integration
- Backward-pass implementation and gradient validation
- Comparison against production framework kernels on identical inputs

## License

MIT. See [LICENSE](LICENSE).

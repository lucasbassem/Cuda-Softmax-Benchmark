# Validation status

Validation performed in the project-generation environment:

- Python source compilation: passed.
- PyTorch explicit stable-softmax comparison: passed.
- Pytest suite: 3 tests passed.
- Standalone C++17 CPU reference compilation with `-Wall -Wextra -Wpedantic`: passed.
- C++ CPU numerical-stability and row-sum test: passed.
- PyTorch CPU benchmark smoke run on `32x257` and `64x513`: passed.
- Plot and Markdown report scripts: validated with schema-compatible sample CSV data.

CUDA compilation and GPU execution were not available because this environment has neither `nvcc` nor an NVIDIA GPU. Therefore, no CUDA latency, speedup, or GPU accuracy numbers are claimed in this repository. Run `python scripts/run_benchmarks.py` on the target NVIDIA system to complete GPU validation and generate measured results.

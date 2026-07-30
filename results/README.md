# Benchmark results

This directory is populated by `python scripts/run_benchmarks.py`.

Expected generated files:

- `benchmark_results.csv`: CPU, basic CUDA, and optimized CUDA measurements plus accuracy metrics and hardware metadata.
- `pytorch_results.csv`: PyTorch reference timing and accuracy measurements.
- `latency.png`: latency comparison chart.
- `speedup.png`: optimized-versus-basic CUDA speedup chart.

No GPU measurements are committed by default because results must reflect the actual hardware, driver, CUDA version, compiler configuration, and benchmark settings used.

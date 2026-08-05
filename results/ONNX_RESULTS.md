# ONNX Runtime CUDA benchmark results

## Summary

A pretrained ResNet-18 model was exported from PyTorch to ONNX with a dynamic batch dimension and benchmarked using ONNX Runtime's CUDA Execution Provider.

At batch size 1, ONNX Runtime reduced median latency from **3.858 ms** to **2.105 ms**, a **45.4% latency reduction**, and increased throughput from **259.2** to **475.1 images/second**, a **1.83x speedup**.

Performance was effectively tied at batch size 8. At batch size 32, PyTorch was approximately 0.9% faster. The results support a batch-size-1 latency claim, not a universal ONNX speedup claim.

## Measured results

| Batch | PyTorch median (ms) | PyTorch p95 (ms) | PyTorch images/s | ONNX median (ms) | ONNX p95 (ms) | ONNX images/s | ONNX speedup | Latency reduction | Max abs. error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.858 | 4.727 | 259.2 | 2.105 | 3.238 | 475.1 | 1.833x | 45.44% | 0.000006 |
| 8 | 9.184 | 9.347 | 871.1 | 9.117 | 9.239 | 877.5 | 1.007x | 0.73% | 0.000008 |
| 32 | 26.023 | 26.934 | 1229.7 | 26.266 | 28.593 | 1218.3 | 0.991x | -0.93% | 0.000013 |

## Methodology

- Model: pretrained torchvision ResNet-18
- Input: float32 tensors with shape `[batch_size, 3, 224, 224]`
- Batch sizes: 1, 8, and 32
- Warm-up iterations: 10
- Timed iterations: 50
- PyTorch timing: CUDA events
- ONNX Runtime timing: synchronized wall-clock timing
- ONNX input/output: GPU-resident through I/O binding
- Correctness: maximum absolute difference against PyTorch output
- Active provider: `CUDAExecutionProvider`

## Reproducibility note

The original pasted output preserved the result table and active provider, but not complete GPU, driver, CUDA, PyTorch, and ONNX Runtime metadata. Rerun `python/onnx_runtime_benchmark.py` on the original benchmark environment before the final application. The script writes `onnx_runtime_environment.json` automatically.

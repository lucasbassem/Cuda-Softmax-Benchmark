#!/usr/bin/env python3
"""Generate CUDA softmax latency and speedup charts from the raw CSV."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
data = pd.read_csv(RESULTS / "cuda_softmax_benchmark.csv")
labels = [f"{r}×{c}" for r, c in zip(data["rows"], data["cols"])]

x = range(len(data))
width = 0.25

plt.figure(figsize=(9, 5.5))
plt.bar([i - width for i in x], data["cpu_ms"], width=width, label="CPU")
plt.bar(list(x), data["basic_cuda_ms"], width=width, label="Basic CUDA")
plt.bar([i + width for i in x], data["optimized_cuda_ms"], width=width, label="Optimized CUDA")
plt.yscale("log")
plt.xticks(list(x), labels, rotation=20)
plt.ylabel("Median latency (ms, log scale)")
plt.xlabel("Tensor shape")
plt.title("Softmax latency by implementation")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / "cuda_softmax_latency.png", dpi=180)
plt.close()

plt.figure(figsize=(9, 5.5))
plt.plot(labels, data["optimized_speedup_vs_cpu"], marker="o", label="Optimized CUDA vs CPU")
plt.plot(labels, data["optimized_speedup_vs_basic"], marker="o", label="Optimized CUDA vs basic CUDA")
plt.ylabel("Speedup (×)")
plt.xlabel("Tensor shape")
plt.title("Optimized CUDA softmax speedup")
plt.xticks(rotation=20)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / "cuda_softmax_speedup.png", dpi=180)
plt.close()

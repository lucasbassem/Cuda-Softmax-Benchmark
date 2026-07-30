#!/usr/bin/env python3
"""Create benchmark plots from the C++/CUDA CSV output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No benchmark rows found in {path}")
    return rows


def plot_latency(rows: list[dict[str, str]], output: Path) -> None:
    labels = [f"{row['rows']}x{row['cols']}" for row in rows]
    positions = list(range(len(labels)))
    width = 0.25

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar([x - width for x in positions], [float(row["cpu_ms"]) for row in rows], width, label="CPU")
    axis.bar(positions, [float(row["basic_cuda_ms"]) for row in rows], width, label="Basic CUDA")
    axis.bar([x + width for x in positions], [float(row["optimized_cuda_ms"]) for row in rows], width, label="Optimized CUDA")
    axis.set_yscale("log")
    axis.set_ylabel("Latency per softmax call (ms, log scale)")
    axis.set_xlabel("Tensor shape")
    axis.set_title("Softmax Latency by Implementation")
    axis.set_xticks(positions, labels, rotation=20)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_speedup(rows: list[dict[str, str]], output: Path) -> None:
    labels = [f"{row['rows']}x{row['cols']}" for row in rows]
    speedups = [float(row["optimized_speedup_vs_basic"]) for row in rows]

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, speedups)
    axis.axhline(1.0, linewidth=1)
    axis.set_ylabel("Speedup (basic CUDA / optimized CUDA)")
    axis.set_xlabel("Tensor shape")
    axis.set_title("Optimized Kernel Speedup")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/benchmark_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    args = parser.parse_args()

    rows = load_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latency_path = args.output_dir / "latency.png"
    speedup_path = args.output_dir / "speedup.png"
    plot_latency(rows, latency_path)
    plot_speedup(rows, speedup_path)
    print(f"Wrote {latency_path}")
    print(f"Wrote {speedup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

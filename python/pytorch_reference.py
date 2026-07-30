#!/usr/bin/env python3
"""PyTorch reference implementation and timing baseline for row-wise softmax."""

from __future__ import annotations

import argparse
import csv
import math
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass(frozen=True)
class Shape:
    rows: int
    cols: int

    @property
    def elements(self) -> int:
        return self.rows * self.cols


def parse_shapes(value: str) -> list[Shape]:
    shapes: list[Shape] = []
    for item in value.split(","):
        try:
            rows_text, cols_text = item.lower().split("x", maxsplit=1)
            rows, cols = int(rows_text), int(cols_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid shape {item!r}; expected ROWSxCOLS"
            ) from exc
        if rows <= 0 or cols <= 0:
            raise argparse.ArgumentTypeError("Shape dimensions must be positive")
        shapes.append(Shape(rows, cols))
    if not shapes:
        raise argparse.ArgumentTypeError("At least one shape is required")
    return shapes


def stable_softmax_reference(tensor: torch.Tensor) -> torch.Tensor:
    """Explicit stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))."""
    shifted = tensor - tensor.amax(dim=-1, keepdim=True)
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=-1, keepdim=True)


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float, float]:
    difference = (reference - candidate).abs()
    denominator = reference.abs().clamp_min(1.0e-8)
    max_absolute = difference.max().item()
    max_relative = (difference / denominator).max().item()
    max_row_sum_error = (candidate.sum(dim=-1) - 1.0).abs().max().item()
    return max_absolute, max_relative, max_row_sum_error


def benchmark_cuda(operation, warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))

    return statistics.mean(samples), statistics.pstdev(samples)


def benchmark_cpu(operation, iterations: int) -> tuple[float, float]:
    samples: list[float] = []
    cpu_iterations = max(1, min(iterations, 20))
    operation()
    for _ in range(cpu_iterations):
        start = time.perf_counter_ns()
        operation()
        stop = time.perf_counter_ns()
        samples.append((stop - start) / 1.0e6)
    return statistics.mean(samples), statistics.pstdev(samples)


def iter_rows(
    shapes: Iterable[Shape],
    device: torch.device,
    warmup: int,
    iterations: int,
    seed: int,
) -> Iterable[dict[str, object]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    for shape in shapes:
        source = torch.randn(
            (shape.rows, shape.cols),
            dtype=torch.float32,
            device=device,
        ) * 3.0

        explicit = stable_softmax_reference(source)
        builtin = torch.softmax(source, dim=-1)
        max_abs, max_rel, row_sum_error = error_metrics(explicit, builtin)

        operation = lambda: torch.softmax(source, dim=-1)
        if device.type == "cuda":
            mean_ms, std_ms = benchmark_cuda(operation, warmup, iterations)
        else:
            mean_ms, std_ms = benchmark_cpu(operation, iterations)

        yield {
            "framework": "PyTorch",
            "torch_version": torch.__version__,
            "device": str(device),
            "hardware": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor() or platform.machine()
            ),
            "rows": shape.rows,
            "cols": shape.cols,
            "elements": shape.elements,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "max_abs_error_vs_explicit": max_abs,
            "max_rel_error_vs_explicit": max_rel,
            "max_row_sum_error": row_sum_error,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=parse_shapes,
        default=parse_shapes("1024x128,1024x512,4096x1024,8192x2048"),
        help="Comma-separated ROWSxCOLS tensor shapes",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pytorch_results.csv"),
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but torch.cuda.is_available() is false")
    resolved_device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(resolved_device)

    rows = list(iter_rows(args.sizes, device, args.warmup, args.iterations, args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['rows']}x{row['cols']} | {row['device']} | "
            f"{row['mean_ms']:.6f} ms ± {row['std_ms']:.6f} | "
            f"max abs {row['max_abs_error_vs_explicit']:.3e} | "
            f"max rel {row['max_rel_error_vs_explicit']:.3e}"
        )
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

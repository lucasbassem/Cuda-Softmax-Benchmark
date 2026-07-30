#!/usr/bin/env python3
"""Render a commit-ready Markdown report from benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
import platform
from datetime import datetime, timezone
from pathlib import Path


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def format_number(value: str, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-results",
        type=Path,
        default=Path("results/benchmark_results.csv"),
    )
    parser.add_argument(
        "--pytorch-results",
        type=Path,
        default=Path("results/pytorch_results.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/RESULTS.md"),
    )
    args = parser.parse_args()

    cuda_rows = load_csv(args.cuda_results)
    pytorch_rows = load_csv(args.pytorch_results)
    first = cuda_rows[0]
    torch_by_shape = {
        (row["rows"], row["cols"]): row for row in pytorch_rows
    }

    lines = [
        "# Measured benchmark results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Hardware and configuration",
        "",
        f"- GPU: {first['gpu']}",
        f"- Compute capability: {first['compute_capability']}",
        f"- CUDA runtime: {first['cuda_runtime']}",
        f"- CUDA driver API version: {first['cuda_driver']}",
        f"- Optimized kernel block size: {first['threads_per_block']} threads",
        f"- Warmup iterations: {first['warmup_iterations']}",
        f"- Timed iterations: {first['benchmark_iterations']}",
        f"- CUDA fast math: {first['fast_math']}",
        f"- PyTorch: {pytorch_rows[0]['torch_version']}",
        f"- CPU: {platform.processor() or platform.machine()}",
        f"- Operating system: {platform.platform()}",
        f"- Python: {platform.python_version()}",
        "",
        "## Latency and speedup",
        "",
        "| Shape | CPU ms | Basic CUDA ms | Optimized CUDA ms | PyTorch CUDA ms | Optimized vs. basic | Optimized vs. CPU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in cuda_rows:
        torch_row = torch_by_shape.get((row["rows"], row["cols"]))
        torch_ms = format_number(torch_row["mean_ms"]) if torch_row else "n/a"
        lines.append(
            "| "
            f"{row['rows']}x{row['cols']} | "
            f"{format_number(row['cpu_ms'])} | "
            f"{format_number(row['basic_cuda_ms'])} | "
            f"{format_number(row['optimized_cuda_ms'])} | "
            f"{torch_ms} | "
            f"{format_number(row['optimized_speedup_vs_basic'], 3)}x | "
            f"{format_number(row['optimized_speedup_vs_cpu'], 3)}x |"
        )

    lines.extend(
        [
            "",
            "## Accuracy",
            "",
            "| Shape | Basic max abs. | Basic max rel. | Optimized max abs. | Optimized max rel. | Optimized row-sum error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cuda_rows:
        lines.append(
            "| "
            f"{row['rows']}x{row['cols']} | "
            f"{float(row['basic_max_abs_error']):.3e} | "
            f"{float(row['basic_max_rel_error']):.3e} | "
            f"{float(row['optimized_max_abs_error']):.3e} | "
            f"{float(row['optimized_max_rel_error']):.3e} | "
            f"{float(row['optimized_max_row_sum_error']):.3e} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation checklist",
            "",
            "Document observations only after inspecting the generated numbers and profiler output:",
            "",
            "- Identify the tensor sizes where block-level parallelism amortizes launch and synchronization overhead.",
            "- Explain whether the basic kernel is limited by serialized per-row work.",
            "- Compare the custom kernel with PyTorch without claiming parity; framework kernels may use architecture- and shape-specific implementations.",
            "- Record any fast-math tradeoff separately rather than mixing configurations.",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

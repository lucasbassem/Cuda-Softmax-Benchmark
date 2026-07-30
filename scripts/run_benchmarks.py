#!/usr/bin/env python3
"""Build the CUDA executable, run both benchmark suites, and create plots."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="release")
    parser.add_argument("--sizes", default="1024x128,1024x512,4096x1024,8192x2048")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=256)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if shutil.which("cmake") is None:
        parser.error("cmake was not found on PATH")
    if shutil.which("nvcc") is None:
        parser.error("nvcc was not found on PATH; install the NVIDIA CUDA Toolkit")

    run(["cmake", "--preset", args.preset], root)
    run(["cmake", "--build", "--preset", args.preset], root)

    executable_name = "softmax_benchmark.exe" if sys.platform == "win32" else "softmax_benchmark"
    executable = root / "build" / args.preset / executable_name
    if not executable.exists():
        parser.error(f"Expected benchmark executable at {executable}")

    run(
        [
            str(executable),
            "--sizes",
            args.sizes,
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--threads",
            str(args.threads),
            "--output",
            "results/benchmark_results.csv",
        ],
        root,
    )
    run(
        [
            sys.executable,
            "python/pytorch_reference.py",
            "--device",
            "cuda",
            "--sizes",
            args.sizes,
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--output",
            "results/pytorch_results.csv",
        ],
        root,
    )
    run(
        [
            sys.executable,
            "python/plot_results.py",
            "--input",
            "results/benchmark_results.csv",
            "--output-dir",
            "results",
        ],
        root,
    )
    run(
        [
            sys.executable,
            "python/render_results_markdown.py",
            "--cuda-results",
            "results/benchmark_results.csv",
            "--pytorch-results",
            "results/pytorch_results.csv",
            "--output",
            "results/RESULTS.md",
        ],
        root,
    )

    print("Benchmark suite completed. Review the results/ directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

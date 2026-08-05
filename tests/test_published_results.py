from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_rows(filename: str) -> list[dict[str, str]]:
    with (RESULTS / filename).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, f"{filename} contains no benchmark rows"
    return rows


def assert_close(actual: float, expected: float, tolerance: float = 1.0e-6) -> None:
    assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def test_cuda_result_formulas_and_accuracy() -> None:
    rows = load_rows("benchmark_results.csv")

    for row in rows:
        cpu_ms = float(row["cpu_ms"])
        basic_ms = float(row["basic_cuda_ms"])
        optimized_ms = float(row["optimized_cuda_ms"])

        assert cpu_ms > 0
        assert basic_ms > 0
        assert optimized_ms > 0

        assert_close(float(row["basic_speedup_vs_cpu"]), cpu_ms / basic_ms)
        assert_close(float(row["optimized_speedup_vs_cpu"]), cpu_ms / optimized_ms)
        assert_close(float(row["optimized_speedup_vs_basic"]), basic_ms / optimized_ms)

        assert float(row["basic_max_abs_error"]) <= 2.0e-5
        assert float(row["basic_max_row_sum_error"]) <= 2.0e-5
        assert float(row["optimized_max_abs_error"]) <= 2.0e-5
        assert float(row["optimized_max_row_sum_error"]) <= 2.0e-5


def test_published_same_workload_cuda_claim() -> None:
    rows = load_rows("benchmark_results.csv")
    row = next(
        item
        for item in rows
        if int(item["rows"]) == 1024 and int(item["cols"]) == 512
    )

    assert_close(float(row["optimized_speedup_vs_cpu"]), 97.60322716)
    assert_close(float(row["optimized_speedup_vs_basic"]), 12.44275471)
    assert float(row["optimized_max_abs_error"]) <= 2.4e-7


def test_onnx_result_formulas_and_accuracy() -> None:
    rows = load_rows("onnx_runtime_cuda_benchmark.csv")

    for row in rows:
        batch_size = int(row["batch_size"])
        pytorch_median = float(row["pytorch_median_ms"])
        onnx_median = float(row["onnx_median_ms"])

        assert batch_size > 0
        assert pytorch_median > 0
        assert onnx_median > 0

        assert_close(
            float(row["pytorch_throughput_images_per_second"]),
            batch_size * 1000.0 / pytorch_median,
            tolerance=1.0e-5,
        )
        assert_close(
            float(row["onnx_throughput_images_per_second"]),
            batch_size * 1000.0 / onnx_median,
            tolerance=1.0e-5,
        )
        assert_close(
            float(row["onnx_speedup_vs_pytorch"]),
            pytorch_median / onnx_median,
            tolerance=1.0e-5,
        )
        assert_close(
            float(row["onnx_latency_reduction_percent"]),
            (1.0 - onnx_median / pytorch_median) * 100.0,
            tolerance=1.0e-5,
        )
        assert float(row["max_absolute_error"]) <= 1.3e-5


def test_published_batch_one_onnx_claim() -> None:
    rows = load_rows("onnx_runtime_cuda_benchmark.csv")
    row = next(item for item in rows if int(item["batch_size"]) == 1)

    assert_close(float(row["onnx_speedup_vs_pytorch"]), 1.832863)
    assert_close(float(row["onnx_latency_reduction_percent"]), 45.440536)

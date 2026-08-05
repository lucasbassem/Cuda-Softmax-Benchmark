#!/usr/bin/env python3
"""Benchmark PyTorch CUDA against ONNX Runtime CUDA using ResNet-18."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torchvision.models as models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable.")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "ONNX Runtime CUDAExecutionProvider is unavailable. "
            "Install a compatible onnxruntime-gpu build."
        )


def export_model(model: torch.nn.Module, path: Path) -> None:
    dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
    torch.onnx.export(
        model,
        (dummy_input,),
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=18,
    )
    onnx.checker.check_model(onnx.load(path))


def create_session(model_path: Path) -> ort.InferenceSession:
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        preload()

    session = ort.InferenceSession(
        str(model_path),
        providers=[
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ],
    )
    active = session.get_providers()
    if not active or active[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            f"CUDAExecutionProvider is not the primary provider: {active}"
        )
    return session


def collect_environment(
    session: ort.InferenceSession,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.cuda.get_device_properties(0)
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(device.total_memory),
        "gpu_compute_capability": f"{device.major}.{device.minor}",
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "onnxruntime_available_providers": ort.get_available_providers(),
        "onnxruntime_active_providers": session.get_providers(),
        "batch_sizes": args.batch_sizes,
        "warmup_iterations": args.warmup,
        "benchmark_iterations": args.iterations,
        "random_seed": args.seed,
        "model": "torchvision.models.resnet18",
        "weights": "ResNet18_Weights.DEFAULT",
        "opset_version": 18,
        "input_shape": "[batch_size, 3, 224, 224]",
    }


def benchmark(
    model: torch.nn.Module,
    session: ort.InferenceSession,
    batch_sizes: list[int],
    warmup_iterations: int,
    benchmark_iterations: int,
) -> pd.DataFrame:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    rows: list[dict[str, float | int]] = []

    print("Active providers:", session.get_providers())

    for batch_size in batch_sizes:
        print(f"Benchmarking batch size {batch_size}...")
        torch_input = torch.randn(
            batch_size,
            3,
            224,
            224,
            device="cuda",
            dtype=torch.float32,
        ).contiguous()

        with torch.inference_mode():
            for _ in range(warmup_iterations):
                model(torch_input)

        torch.cuda.synchronize()
        pytorch_times: list[float] = []

        with torch.inference_mode():
            for _ in range(benchmark_iterations):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                pytorch_output = model(torch_input)
                end_event.record()
                end_event.synchronize()
                pytorch_times.append(float(start_event.elapsed_time(end_event)))

        io_binding = session.io_binding()
        io_binding.bind_input(
            name=input_name,
            device_type="cuda",
            device_id=0,
            element_type=np.float32,
            shape=tuple(torch_input.shape),
            buffer_ptr=torch_input.data_ptr(),
        )
        io_binding.bind_output(
            name=output_name,
            device_type="cuda",
            device_id=0,
        )

        for _ in range(warmup_iterations):
            session.run_with_iobinding(io_binding)

        torch.cuda.synchronize()
        onnx_times: list[float] = []

        for _ in range(benchmark_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            session.run_with_iobinding(io_binding)
            torch.cuda.synchronize()
            onnx_times.append((time.perf_counter() - start) * 1000.0)

        onnx_output = io_binding.copy_outputs_to_cpu()[0]
        pytorch_output_np = pytorch_output.detach().cpu().numpy()

        pytorch_median = float(np.median(pytorch_times))
        onnx_median = float(np.median(onnx_times))

        rows.append(
            {
                "batch_size": batch_size,
                "pytorch_median_ms": pytorch_median,
                "pytorch_p95_ms": float(np.percentile(pytorch_times, 95)),
                "pytorch_throughput_images_per_second": (
                    batch_size * 1000.0 / pytorch_median
                ),
                "onnx_median_ms": onnx_median,
                "onnx_p95_ms": float(np.percentile(onnx_times, 95)),
                "onnx_throughput_images_per_second": (
                    batch_size * 1000.0 / onnx_median
                ),
                "onnx_speedup_vs_pytorch": pytorch_median / onnx_median,
                "onnx_latency_reduction_percent": (
                    1.0 - onnx_median / pytorch_median
                )
                * 100.0,
                "max_absolute_error": float(
                    np.max(np.abs(pytorch_output_np - onnx_output))
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("Warm-up and timed iterations must be positive.")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise ValueError("All batch sizes must be positive.")

    require_cuda()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "resnet18_dynamic.onnx"
    csv_path = args.output_dir / "onnx_runtime_cuda_benchmark.csv"
    metadata_path = args.output_dir / "onnx_runtime_environment.json"

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    ).eval().to("cuda")

    export_model(model, model_path)
    session = create_session(model_path)
    dataframe = benchmark(
        model,
        session,
        args.batch_sizes,
        args.warmup,
        args.iterations,
    )

    dataframe.to_csv(csv_path, index=False)
    metadata_path.write_text(
        json.dumps(collect_environment(session, args), indent=2) + "\n",
        encoding="utf-8",
    )

    print(dataframe.to_string(index=False))
    print("Saved:", csv_path)
    print("Saved:", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
from pathlib import Path

import torch


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

import pytorch_reference as reference  # noqa: E402


def test_stable_softmax_matches_torch() -> None:
    torch.manual_seed(7)
    values = torch.randn(32, 257, dtype=torch.float32) * 20.0
    expected = reference.stable_softmax_reference(values)
    candidate = torch.softmax(values, dim=-1)
    assert torch.allclose(expected, candidate, atol=1.0e-6, rtol=1.0e-5)


def test_large_logits_remain_finite() -> None:
    values = torch.tensor([[10_000.0, 9_999.0, -10_000.0]], dtype=torch.float32)
    output = reference.stable_softmax_reference(values)
    assert torch.isfinite(output).all()
    assert torch.allclose(output.sum(dim=-1), torch.ones(1), atol=1.0e-6)


def test_error_metrics_detect_exact_match() -> None:
    values = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    max_abs, max_rel, row_sum = reference.error_metrics(values, values.clone())
    assert max_abs == 0.0
    assert max_rel == 0.0
    assert row_sum == 0.0

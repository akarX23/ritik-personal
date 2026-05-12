"""T008 – Unit tests for MnistClassifier.

Tests must FAIL before src/model.py is implemented.
"""

import pytest
import torch


def test_output_shape():
    from src.model import MnistClassifier

    model = MnistClassifier()
    x = torch.randn(8, 784)
    out = model(x)
    assert out.shape == (8, 10), f"Expected (8, 10), got {out.shape}"


def test_single_sample_shape():
    from src.model import MnistClassifier

    model = MnistClassifier()
    x = torch.randn(1, 784)
    out = model(x)
    assert out.shape == (1, 10)


def test_output_dtype_is_float():
    from src.model import MnistClassifier

    model = MnistClassifier()
    x = torch.randn(4, 784)
    out = model(x)
    assert out.dtype == torch.float32


def test_forward_is_deterministic():
    from src.model import MnistClassifier

    torch.manual_seed(0)
    model = MnistClassifier()
    model.eval()
    x = torch.randn(4, 784)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)

"""T034 – Unit tests for data.py split sizes and DataLoader types.

Tests must FAIL before src/data.py is implemented.
"""

import pytest
from torch.utils.data import DataLoader


def test_returns_three_dataloaders(tmp_path):
    from src.data import load_mnist

    result = load_mnist(tmp_path, batch_size=64)
    assert len(result) == 3
    for loader in result:
        assert isinstance(loader, DataLoader)


def test_train_loader_has_batches(tmp_path):
    from src.data import load_mnist

    train_loader, _, _ = load_mnist(tmp_path, batch_size=64)
    batch = next(iter(train_loader))
    images, labels = batch
    # shape: (batch, 1, 28, 28) and (batch,)
    assert images.shape[0] <= 64
    assert labels.shape[0] <= 64


def test_test_loader_length_is_standard(tmp_path):
    """MNIST test set has 10 000 samples."""
    from src.data import load_mnist

    _, _, test_loader = load_mnist(tmp_path, batch_size=256)
    total = sum(len(labels) for _, labels in test_loader)
    assert total == 10_000


def test_val_plus_train_equals_60000(tmp_path):
    from src.data import load_mnist

    train_loader, val_loader, _ = load_mnist(tmp_path, batch_size=256)
    total_train = sum(len(l) for _, l in train_loader)
    total_val = sum(len(l) for _, l in val_loader)
    assert total_train + total_val == 60_000

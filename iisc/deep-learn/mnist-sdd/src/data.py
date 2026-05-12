"""MNIST dataset loading utilities.

Returns train/validation/test DataLoader objects backed by
torchvision.datasets.MNIST. No custom augmentation is applied; only the
standard pixel-normalisation transform is used.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)


def load_mnist(
    data_dir: str | Path,
    batch_size: int,
    val_fraction: float = 0.1,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Download (if needed) and load MNIST into three DataLoaders.

    Parameters
    ----------
    data_dir:
        Directory where MNIST data will be stored / read from.
    batch_size:
        Mini-batch size for all loaders.
    val_fraction:
        Fraction of the official 60 000 training samples used for validation.
    num_workers:
        Subprocesses for data loading.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
        ]
    )

    full_train = datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    loader_kwargs: dict = {"batch_size": batch_size, "num_workers": num_workers}

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

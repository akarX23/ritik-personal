"""MnistClassifier – fixed fully-connected architecture for MNIST.

Architecture: 784 → 256 (ReLU) → 128 (ReLU) → 10
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MnistClassifier(nn.Module):
    """Fully-connected MNIST classifier with two hidden layers.

    Input must be a flat tensor of shape ``(batch, 784)``.
    Output is raw logits of shape ``(batch, 10)`` — no softmax applied.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Flat input tensor, shape ``(batch, 784)``.

        Returns
        -------
        torch.Tensor
            Logits, shape ``(batch, 10)``.
        """
        return self.net(x)

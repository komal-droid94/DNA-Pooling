"""1D CNN for DNA sequence classification.

How it maps to biology:
- Conv1d filters slide along the one-hot sequence; each filter acts like a learnable
  position weight matrix (PWM) that scores how well each window matches a motif.
- Max pooling keeps the strongest match in a region and throws away its exact position.
  That is what we want: a binding site counts whether it sits at base 20 or base 180.
- Global max pooling ("is the motif anywhere?") makes the model fully translation-invariant.

The `pooling` argument lets the experiments swap max pooling for average pooling or none.
"""

import torch
from torch import nn


class DNACNN(nn.Module):
    def __init__(self, seq_len: int = 200, n_filters: int = 32, kernel_size: int = 12,
                 pooling: str = "max", pool_size: int = 4, dropout: float = 0.2):
        super().__init__()
        if pooling not in {"max", "avg", "none"}:
            raise ValueError("pooling must be 'max', 'avg', or 'none'")
        self.pooling = pooling
        self.conv1 = nn.Conv1d(4, n_filters, kernel_size)          # motif scanners
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.conv2 = nn.Conv1d(n_filters, n_filters, 5, padding=2)  # combinations of motifs
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        conv_len = seq_len - kernel_size + 1
        if pooling == "none":
            # No pooling: flatten every position. The classifier then has a separate weight
            # for "motif at position 37" vs "motif at position 38" (not translation-invariant).
            self.local_pool = nn.Identity()
            self.global_pool = None
            self.fc = nn.Linear(n_filters * conv_len, 1)
        else:
            pool_cls = nn.MaxPool1d if pooling == "max" else nn.AvgPool1d
            self.local_pool = pool_cls(pool_size)
            self.global_pool = nn.AdaptiveMaxPool1d(1) if pooling == "max" else nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(n_filters, 1)

    def first_layer_activations(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_filters, positions) scores from the motif-scanning layer."""
        return self.act(self.bn1(self.conv1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.first_layer_activations(x)
        h = self.local_pool(h)
        h = self.act(self.conv2(h))
        h = self.drop(h)
        h = h.flatten(1) if self.global_pool is None else self.global_pool(h).squeeze(-1)
        return self.fc(h).squeeze(-1)  # logits; apply sigmoid for probabilities

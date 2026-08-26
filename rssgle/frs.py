from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class AffineResidualHead(nn.Module):
    """A single affine residual mapping used in the deployment graph."""

    def __init__(self, feature_dim: int, classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, classes)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features.float())


class NormalizedResidualHead(nn.Module):
    """Training-time affine residual head with frozen feature normalization."""

    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        classes: int,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", feature_mean.detach().float().clone())
        self.register_buffer("feature_scale", feature_scale.detach().float().clone())
        self.linear = nn.Linear(feature_mean.numel(), classes)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features.float() - self.feature_mean) / self.feature_scale
        return self.linear(normalized)

    def forward(self, features: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        return base_logits.detach().float() + self.residual(features)


@dataclass(frozen=True)
class RawAffine:
    weight: torch.Tensor
    bias: torch.Tensor


def normalized_to_raw(head: NormalizedResidualHead) -> RawAffine:
    """Convert W((f-mu)/sigma)+b to an affine mapping of raw features."""

    weight = head.linear.weight.detach().float() / head.feature_scale.unsqueeze(0)
    bias = head.linear.bias.detach().float() - weight @ head.feature_mean
    return RawAffine(weight, bias)


def fold_frs_heads(
    ordinary: NormalizedResidualHead,
    rotation: NormalizedResidualHead,
    alpha: float = 0.4,
) -> AffineResidualHead:
    """Fold ordinary and rotation residual heads into one exact affine map."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    original = normalized_to_raw(ordinary)
    rotated = normalized_to_raw(rotation)
    if original.weight.shape != rotated.weight.shape:
        raise ValueError("residual head shapes must match")
    classes, feature_dim = original.weight.shape
    folded = AffineResidualHead(feature_dim, classes)
    with torch.no_grad():
        folded.linear.weight.copy_(
            original.weight + alpha * (rotated.weight - original.weight)
        )
        folded.linear.bias.copy_(original.bias + alpha * (rotated.bias - original.bias))
    return folded

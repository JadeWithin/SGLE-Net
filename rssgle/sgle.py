from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SGLEConfig:
    radius: int = 3
    affinity_temperature: float = 1.0
    message_strength: float = 1.5
    epsilon: float = 1e-12


def shift_neighbour(
    value: torch.Tensor,
    row_offset: int,
    column_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift an HxW tensor without wraparound and return its validity mask."""

    height, width = value.shape[:2]
    shifted = torch.zeros_like(value)
    valid = torch.zeros((height, width), dtype=torch.bool, device=value.device)
    src_r0, src_r1 = max(0, row_offset), min(height, height + row_offset)
    src_c0, src_c1 = max(0, column_offset), min(width, width + column_offset)
    dst_r0, dst_r1 = max(0, -row_offset), min(height, height - row_offset)
    dst_c0, dst_c1 = max(0, -column_offset), min(width, width - column_offset)
    shifted[dst_r0:dst_r1, dst_c0:dst_c1] = value[src_r0:src_r1, src_c0:src_c1]
    valid[dst_r0:dst_r1, dst_c0:dst_c1] = True
    return shifted, valid


def sgle_refine(
    cube_hwb: torch.Tensor,
    logits_hwc: torch.Tensor,
    config: SGLEConfig = SGLEConfig(),
) -> torch.Tensor:
    """Apply spectrally gated neighbourhood log-evidence refinement."""

    if cube_hwb.ndim != 3 or logits_hwc.ndim != 3:
        raise ValueError("cube and logits must be [H,W,B] and [H,W,C]")
    if cube_hwb.shape[:2] != logits_hwc.shape[:2]:
        raise ValueError("cube and logits spatial dimensions do not match")
    if config.radius < 1 or config.affinity_temperature <= 0:
        raise ValueError("radius and affinity temperature must be positive")

    cube = cube_hwb.float()
    logits = logits_hwc.float()
    height, width, _classes = logits.shape
    offsets = [
        (row, column)
        for row in range(-config.radius, config.radius + 1)
        for column in range(-config.radius, config.radius + 1)
        if row != 0 or column != 0
    ]

    distances: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    for row, column in offsets:
        neighbour, valid = shift_neighbour(cube, row, column)
        distance = (neighbour - cube).square().mean(dim=2)
        distances.append(distance.masked_fill(~valid, float("nan")))
        valids.append(valid)

    local_scale = torch.nanquantile(torch.stack(distances), 0.5, dim=0)
    local_scale = local_scale.clamp_min(config.epsilon)
    log_probability = F.log_softmax(logits, dim=2)
    message = torch.zeros_like(logits)
    weight_sum = torch.zeros((height, width), device=logits.device)

    for (row, column), distance, valid in zip(offsets, distances, valids):
        neighbour_log, _ = shift_neighbour(log_probability, row, column)
        affinity = torch.exp(
            -torch.nan_to_num(distance, nan=0.0)
            / (config.affinity_temperature * local_scale)
        ) * valid.float()
        message += affinity.unsqueeze(2) * neighbour_log
        weight_sum += affinity

    message /= weight_sum.clamp_min(config.epsilon).unsqueeze(2)
    return logits + config.message_strength * message

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .data import PixelSplit, flat_indices
from .frs import NormalizedResidualHead
from .metrics import classification_metrics
from .model import MSSSBackbone


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 300
    eval_interval: int = 5
    patience_checks: int = 12
    learning_rate: float = 3e-4
    residual_steps: int = 600
    residual_learning_rate: float = 0.03


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _indices(split: PixelSplit, name: str, width: int, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(flat_indices(split, name, width), dtype=torch.long, device=device)


def train_backbone(
    model: MSSSBackbone,
    cube: torch.Tensor,
    split: PixelSplit,
    config: TrainConfig,
) -> MSSSBackbone:
    """Train the base full-scene network with validation early stopping."""

    device = cube.device
    _height, width, _bands = cube.shape
    train_index = _indices(split, "train", width, device)
    val_index = _indices(split, "val", width, device)
    train_labels = torch.as_tensor(split.labels["train"], dtype=torch.long, device=device)
    val_labels = torch.as_tensor(split.labels["val"], dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(cube).reshape(-1, model.classes)
            loss = F.cross_entropy(logits[train_index], train_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if epoch != 1 and epoch % config.eval_interval:
            continue
        model.eval()
        with torch.inference_mode():
            val_logits = model(cube).reshape(-1, model.classes)[val_index]
            val_prediction = val_logits.argmax(dim=1).cpu().numpy()
            validation = classification_metrics(
                val_prediction,
                val_labels.cpu().numpy(),
                model.classes,
            )
            score = float(
                np.mean([validation[metric] for metric in ("oa", "aa", "kappa")])
            )
        if score > best_score + 1e-8:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience_checks:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model


def train_residual_head(
    features: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    classes: int,
    config: TrainConfig,
) -> NormalizedResidualHead:
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-4)
    head = NormalizedResidualHead(mean, scale, classes).to(features.device)
    counts = torch.bincount(labels, minlength=classes).float().clamp_min(1.0)
    class_weight = counts.sum() / (classes * counts)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.residual_learning_rate, weight_decay=1e-3
    )
    for _ in range(config.residual_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(features, base_logits), labels, weight=class_weight)
        loss.backward()
        optimizer.step()
    head.eval()
    return head


@torch.no_grad()
def training_view_tensors(
    model: MSSSBackbone,
    cube: torch.Tensor,
    split: PixelSplit,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ordinary and four-quarter-turn training tensors for FRS."""

    height, width, _bands = cube.shape
    labels_grid = torch.full((height, width), -1, dtype=torch.long, device=cube.device)
    rows = torch.as_tensor(split.rows["train"], dtype=torch.long, device=cube.device)
    cols = torch.as_tensor(split.cols["train"], dtype=torch.long, device=cube.device)
    labels = torch.as_tensor(split.labels["train"], dtype=torch.long, device=cube.device)
    labels_grid[rows, cols] = labels

    feature = model.features(cube)
    logits = model.classifier(feature)
    index = rows * width + cols
    ordinary_features = feature.reshape(-1, model.feature_dim)[index]
    ordinary_logits = logits.reshape(-1, model.classes)[index]

    rotation_features = []
    rotation_logits = []
    rotation_labels = []
    for turns in (0, 1, 2, 3):
        rotated_cube = torch.rot90(cube, turns, dims=(0, 1)).contiguous()
        rotated_label_grid = torch.rot90(labels_grid, turns, dims=(0, 1)).contiguous()
        valid = rotated_label_grid.reshape(-1) >= 0
        rotated_feature = model.features(rotated_cube)
        rotated_base = model.classifier(rotated_feature)
        rotation_features.append(rotated_feature.reshape(-1, model.feature_dim)[valid])
        rotation_logits.append(rotated_base.reshape(-1, model.classes)[valid])
        rotation_labels.append(rotated_label_grid.reshape(-1)[valid])
    return (
        ordinary_features,
        ordinary_logits,
        labels,
        torch.cat(rotation_features),
        torch.cat(rotation_logits),
        torch.cat(rotation_labels),
    )

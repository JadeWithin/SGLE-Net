from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import yaml


@dataclass(frozen=True)
class PixelSplit:
    rows: dict[str, np.ndarray]
    cols: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    label_ids: tuple[int, ...]

    @property
    def classes(self) -> int:
        return len(self.label_ids)


def load_config(path: Path, dataset: str) -> dict[str, object]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if dataset not in values:
        raise KeyError(f"unknown dataset {dataset!r}; choose from {sorted(values)}")
    return dict(values[dataset])


def load_dataset(data_root: Path, config: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    cube_path = data_root / str(config["cube_file"])
    gt_path = data_root / str(config["gt_file"])
    if not cube_path.is_file() or not gt_path.is_file():
        raise FileNotFoundError(f"missing {cube_path} or {gt_path}")

    if config["layout"] == "mat":
        cube = scipy.io.loadmat(cube_path)[str(config["cube_key"])]
        ground_truth = scipy.io.loadmat(gt_path)[str(config["gt_key"])]
    elif config["layout"] == "longkou_bsq":
        bands = int(config["bands"])
        rows = int(config["rows"])
        cols = int(config["cols"])
        cube = np.memmap(cube_path, dtype="<f4", mode="r", shape=(bands, rows, cols))
        cube = cube.transpose(1, 2, 0)
        ground_truth = np.memmap(gt_path, dtype="u1", mode="r", shape=(rows, cols))
    else:
        raise ValueError(f"unsupported layout: {config['layout']}")

    cube = np.asarray(cube)
    ground_truth = np.asarray(ground_truth)
    if cube.ndim != 3 or ground_truth.ndim != 2 or cube.shape[:2] != ground_truth.shape:
        raise ValueError(f"invalid cube/GT shapes: {cube.shape}, {ground_truth.shape}")
    if not np.isfinite(cube).all():
        raise ValueError("cube contains non-finite values")
    return cube, ground_truth


def stratified_pixel_split(
    cube: np.ndarray,
    ground_truth: np.ndarray,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> PixelSplit:
    """Create the deterministic per-class split used in the experiments."""

    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fraction must be below one")
    label_ids = tuple(int(value) for value in np.unique(ground_truth) if value > 0)
    generator = np.random.default_rng(seed)
    pieces = {name: [] for name in ("train", "val", "test")}

    for class_index, label_id in enumerate(label_ids):
        rows, cols = np.nonzero(ground_truth == label_id)
        order = generator.permutation(len(rows))
        train_count = max(1, int(round(len(rows) * train_fraction)))
        val_count = max(1, int(round(len(rows) * validation_fraction)))
        selections = {
            "train": order[:train_count],
            "val": order[train_count : train_count + val_count],
            "test": order[train_count + val_count :],
        }
        for name, selection in selections.items():
            pieces[name].append(
                (
                    rows[selection].astype(np.int64),
                    cols[selection].astype(np.int64),
                    np.full(len(selection), class_index, dtype=np.int64),
                )
            )

    split_rows: dict[str, np.ndarray] = {}
    split_cols: dict[str, np.ndarray] = {}
    split_labels: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        split_rows[name] = np.concatenate([item[0] for item in pieces[name]])
        split_cols[name] = np.concatenate([item[1] for item in pieces[name]])
        split_labels[name] = np.concatenate([item[2] for item in pieces[name]])
        permutation = generator.permutation(len(split_labels[name]))
        split_rows[name] = split_rows[name][permutation]
        split_cols[name] = split_cols[name][permutation]
        split_labels[name] = split_labels[name][permutation]

    spectra = np.asarray(cube[split_rows["train"], split_cols["train"]], dtype=np.float32)
    mean = spectra.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(spectra.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    return PixelSplit(split_rows, split_cols, split_labels, mean, std, label_ids)


def normalize_cube(cube: np.ndarray, split: PixelSplit) -> np.ndarray:
    value = np.asarray(cube, dtype=np.float32)
    return np.ascontiguousarray((value - split.mean) / split.std, dtype=np.float32)


def flat_indices(split: PixelSplit, partition: str, width: int) -> np.ndarray:
    return split.rows[partition] * width + split.cols[partition]

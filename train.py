from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rssgle.data import (
    flat_indices,
    load_config,
    load_dataset,
    normalize_cube,
    stratified_pixel_split,
)
from rssgle.frs import fold_frs_heads
from rssgle.metrics import classification_metrics
from rssgle.model import MSSSBackbone
from rssgle.sgle import SGLEConfig, sgle_refine
from rssgle.training import (
    TrainConfig,
    set_seed,
    train_backbone,
    train_residual_head,
    training_view_tensors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate RSSGLE")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("indianpines", "longkou", "sdfc", "houston2013"),
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--residual-steps", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    dataset_config = load_config(args.config, args.dataset)
    raw_cube, ground_truth = load_dataset(args.data_root, dataset_config)
    split = stratified_pixel_split(
        raw_cube,
        ground_truth,
        float(dataset_config["train_fraction"]),
        float(dataset_config["validation_fraction"]),
        args.seed,
    )
    cube = torch.as_tensor(normalize_cube(raw_cube, split), device=device)
    model = MSSSBackbone(cube.shape[-1], split.classes).to(device)
    training_config = TrainConfig(
        max_epochs=args.max_epochs,
        residual_steps=args.residual_steps,
    )
    model = train_backbone(model, cube, split, training_config)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    ordinary_f, ordinary_z, ordinary_y, rotation_f, rotation_z, rotation_y = (
        training_view_tensors(model, cube, split)
    )
    ordinary_head = train_residual_head(
        ordinary_f, ordinary_z, ordinary_y, split.classes, training_config
    )
    rotation_head = train_residual_head(
        rotation_f, rotation_z, rotation_y, split.classes, training_config
    )
    folded_head = fold_frs_heads(ordinary_head, rotation_head, alpha=0.4).to(device)

    with torch.inference_mode():
        features = model.features(cube)
        base_logits = model.classifier(features)
        pre_sgle = base_logits + folded_head(features)
        final_logits = sgle_refine(
            cube,
            pre_sgle,
            SGLEConfig(radius=3, affinity_temperature=1.0, message_strength=1.5),
        )

    _height, width, _bands = cube.shape
    test_index_np = flat_indices(split, "test", width)
    test_index = torch.as_tensor(test_index_np, dtype=torch.long, device=device)
    prediction = (
        final_logits.reshape(-1, split.classes)[test_index].argmax(dim=1).cpu().numpy()
    )
    metrics = classification_metrics(prediction, split.labels["test"], split.classes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dataset": args.dataset,
            "seed": args.seed,
            "backbone": model.state_dict(),
            "folded_frs": folded_head.state_dict(),
            "bands": int(cube.shape[-1]),
            "classes": split.classes,
            "configuration": {"r": 3, "T": 1.0, "lambda": 1.5, "alpha": 0.4},
        },
        args.output_dir / "checkpoint.pt",
    )
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        prediction=prediction.astype(np.int16),
        targets=split.labels["test"].astype(np.int16),
        rows=split.rows["test"],
        cols=split.cols["test"],
    )
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "metrics": metrics,
        "configuration": {"r": 3, "T": 1.0, "lambda": 1.5, "alpha": 0.4},
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

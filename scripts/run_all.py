from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = ("indianpines", "longkou", "sdfc", "houston2013")
SEEDS = tuple(range(20260727, 20260737))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the four-dataset ten-seed RSSGLE study"
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    for dataset in args.datasets:
        for seed in args.seeds:
            output = args.output_root / dataset / f"seed_{seed}"
            if (output / "metrics.json").is_file():
                print(f"skip completed {dataset}/{seed}")
                continue
            command = [
                sys.executable,
                str(root / "train.py"),
                "--dataset",
                dataset,
                "--data-root",
                str(args.data_root),
                "--output-dir",
                str(output),
                "--seed",
                str(seed),
                "--device",
                args.device,
            ]
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()

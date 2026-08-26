# RSSGLE

Official PyTorch implementation of **RSSGLE: Rotation-Stabilized Spectrally
Gated Log-Evidence Network for Hyperspectral Image Classification**.

RSSGLE contains three stages:

1. a full-scene multi-scale spectral-spatial (MSSS) representation network;
2. foldable rotation-view stabilization (FRS), which folds two training-time
   affine residual heads into one deployment head; and
3. parameter-free spectrally gated log-evidence (SGLE) refinement.

The implementation follows the manuscript configuration: neighbourhood radius
`r=3`, affinity temperature `T=1.0`, message strength `lambda=1.5`, and FRS
blending coefficient `alpha=0.4` on all four datasets.

## Repository layout

```text
rssgle/                 Core model, FRS, SGLE, data and metric code
train.py                End-to-end training and evaluation entry point
configs/                Dataset file names, MAT keys and split fractions
scripts/run_all.py      Ten-seed experiment launcher
tests/                  Unit tests for model shape, SGLE and FRS folding
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/JadeWithin/RSSGLE.git
cd RSSGLE
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Install the CUDA build of PyTorch appropriate for your system when GPU support
is required. CPU execution is also supported for smoke tests and small scenes.

## Data

The datasets are not redistributed. Download them from their providers and
place the files under one data directory using these names:

| Dataset | Cube | Ground truth |
|---|---|---|
| Indian Pines | `Indian_pines_corrected.mat` | `Indian_pines_gt.mat` |
| WHU-Hi-LongKou | `WHU-Hi-LongKou.bsq` | `WHU-Hi-LongKou_gt.bsq` |
| SDFC | `SDFC.mat` | `SDFC_gt.mat` |
| Houston 2013 | `Houston2013_standard.mat` | `Houston2013_standard_gt.mat` |

Dataset sources:

- [Indian Pines hyperspectral scene](https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html)
- [WHU-Hi benchmark](https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm)
- [Houston 2013 IEEE GRSS Data Fusion Contest](https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/)

Users are responsible for following the providers' licenses and terms. Dataset
paths and MAT keys can be changed in `configs/datasets.yaml`.

## Train and evaluate

Run one dataset/seed:

```bash
python train.py \
  --dataset indianpines \
  --data-root /path/to/data \
  --seed 20260727 \
  --output-dir outputs/indianpines_seed20260727
```

The script performs the manuscript's staged procedure: it trains the MSSS
carrier, freezes its feature extractor, fits the ordinary and four-rotation
residual heads on training pixels, analytically folds the residual heads,
applies SGLE, and reports OA, AA and Kappa on the held-out test set. Validation
data are used only for early stopping; the test set is never used for model
selection.

Run the ten formal seeds:

```bash
python scripts/run_all.py --data-root /path/to/data --output-root outputs
```

## Implementation notes

- HSI tensors use `[H, W, B]`; dense logits use `[H, W, C]`.
- SGLE introduces no learnable parameters.
- Deployment does not rotate the input and does not use multi-view inference.
- FRS folding is tested numerically in `tests/test_frs.py`.
- No dataset, checkpoint, password, token or machine-specific path is included.

## License

This project is released under the [MIT License](LICENSE).

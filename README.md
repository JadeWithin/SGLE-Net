

## Repository layout

```text
rssgle/                 Core model, FRS, SGLE, data and metric code
train.py                End-to-end training and evaluation entry point
configs/                Dataset file names, MAT keys and split fractions
scripts/run_all.py      Ten-seed experiment launcher
tests/                  Unit tests for model shape, SGLE and FRS folding
```



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



## Implementation notes

- HSI tensors use `[H, W, B]`; dense logits use `[H, W, C]`.
- SGLE introduces no learnable parameters.
- Deployment does not rotate the input and does not use multi-view inference.
- FRS folding is tested numerically in `tests/test_frs.py`.
- No dataset, checkpoint, password, token or machine-specific path is included.

## License

This project is released under the [MIT License](LICENSE).

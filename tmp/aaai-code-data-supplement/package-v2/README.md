# VecGAD Anonymous Reproduction Supplement

This supplement provides an anonymous reproduction implementation of VecGAD.
It contains the main method and the strict direction--magnitude controls. Each
invocation runs one dataset and one seed.

## Contents

- `run.py`: single-seed training and fixed-endpoint evaluation entry point.
- `vecgad.py`: Transformer encoder, attention readout, pseudo-anomaly generator,
  reconstruction objective, HSC objective, and classifier.
- `data.py`: dataset loading, fixed data split, and graph normalization.
- `tokenization.py`: incremental sparse multi-hop tokenization.
- `controls.py`: strict direction--magnitude controls.
- `reproduction.sh`: a command catalog for the seven datasets.
- `environment.yml`: the reference software environment.
- `REVIEW_USE.md`: anonymous review-use notice.

## Environment

The reference environment uses Python 3.8, PyTorch 2.0.0, and CUDA 11.8 on an
NVIDIA GPU.

```bash
conda env create -f environment.yml
conda activate vecgad-review
```

The code retains a CPU fallback for basic execution, but the submitted
configurations are intended for GPU execution.

## Data

Dataset files are not included. Prepare them under `dataset/` before running a
command. Amazon, Reddit, Photo, Elliptic, T-Finance, and Tolokers use the
corresponding `.mat` files expected by `data.py`. DGraph uses
`dataset/dgraphfin.npz` and `dataset/dgraphfin_adj_list`.

## Reproduction

Run the following command to display the seven single-seed command lines, then
copy the command for the desired dataset:

```bash
bash reproduction.sh
```

All commands use training seed 0 and data-split seed 42. Change only `--seed`
to run another training seed while keeping the split fixed.

The strict controls apply to Amazon, Elliptic, and Tolokers. In the corresponding
dataset command, set `--control` to one of `random_dir`, `random_mag`,
`random_both`, or `constant_mag`; `full` is the unmodified method.

# VecGAD Protocol Card

## Supervision Boundary

VecGAD is framed as semi-supervised graph anomaly detection:

- labeled data during training: a small subset of normal nodes;
- anomaly labels: not used during training;
- evaluation: AUROC and AUPRC on true labels after training.

The paper states a 5 percent training ratio for the main table.

## Datasets In Main Table

VecGAD reports seven datasets:

| Dataset | Nodes | Edges | Attributes | Anomaly rate |
|---|---:|---:|---:|---:|
| Amazon | 11,944 | 4,398,392 | 25 | 9.5% |
| Reddit | 10,984 | 168,016 | 64 | 3.3% |
| Photo | 7,535 | 119,043 | 745 | 9.2% |
| Elliptic | 203,769 | 234,355 | 166 | 9.8% |
| T-Finance | 39,357 | 21,222,543 | 10 | 4.6% |
| Tolokers | 11,758 | 519,000 | 10 | 21.8% |
| DGraph | 3,700,550 | 4,300,999 | 17 | 1.3% |

## Main 5 Percent Results

| Dataset | AUROC | AUPRC |
|---|---:|---:|
| Amazon | 0.9391 | 0.8064 |
| Reddit | 0.5782 | 0.0441 |
| Photo | 0.8960 | 0.6210 |
| Elliptic | 0.7627 | 0.2813 |
| T-Finance | 0.8988 | 0.6448 |
| Tolokers | 0.6612 | 0.3103 |
| DGraph | 0.6006 | 0.0057 |

Note: `docs/experiments.md` currently records Amazon as `0.9344` and Elliptic
as `0.7447`, likely from a prior table revision. Verify source-table alignment
before final paper comparison.

## Hyperparameters Mentioned In Paper

- Transformer layers: 3.
- Attention heads: 2.
- Hidden dimension: 256.
- Propagation steps searched in `{2, 3, ..., 20}`.
- Shell radii searched with inner radius in `{0.3, 0.5}` and outer radius in
  `{0.5, 1, 2}`.
- Learning rate searched in `{1e-4, 3e-4, 5e-4}`.
- Dropout: 0.4.
- Weight decay: 0.

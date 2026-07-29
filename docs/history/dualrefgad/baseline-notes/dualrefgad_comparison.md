# VecGAD Versus DualRefGAD Mechanism Comparison

## Shared Boundary

Both methods are intended to work under a known-normal semi-supervised boundary:

- use labeled normal nodes during training;
- avoid anomaly labels for construction, training, checkpoint selection, and
  hyperparameter selection;
- compute AUROC/AUPRC after training as diagnostics.

## Main Difference

| Axis | VecGAD | DualRefGAD Current Mainline |
|---|---|---|
| Evidence object | reconstruction discrepancy vector | target plus normal/deviation reference tokens |
| Deviation semantics | normal explanation failure direction | descriptor-selected deviation references |
| Training contrast | labeled normals vs generated hard negatives | known-normal low score, weak high-unlabeled ranking, consistency |
| Scalar risk | avoids scalarizing residual before generation | response matrix summaries can become scalar anchors |
| Photo failure risk | controlled by discrepancy direction and shell | `mat_mean` and reader may read valid evidence backward |

## What Not To Copy Blindly

- Do not import VecGAD as a black-box replacement for DualRefGAD.
- Do not use VecGAD's Photo score to justify dataset-specific gates.
- Do not turn reconstruction residual magnitude into another universal scalar
  teacher without auditing direction and regime.

## Useful Translation Hypothesis

DualRefGAD may need a normal-deviation-conditioned readout:

1. use reference membership or witness sets as support/evidence context;
2. use normal explanation failure to decide the direction or shape of readout;
3. avoid universal `mean` response aggregation on Photo-like regimes.

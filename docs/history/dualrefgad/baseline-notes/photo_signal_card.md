# VecGAD Photo Signal Card

## Why Photo Matters

Photo is the largest discrepancy between VecGAD and the current DualRefGAD
main table:

| Method | Photo AUROC | Photo AUPRC |
|---|---:|---:|
| VecGAD | 0.8960 | 0.6210 |
| DualRefGAD mainline | 0.4528 | 0.0805 |

Photo is an e-commerce graph where nodes are items and edges represent
co-purchase or co-review relationships. The VecGAD appendix describes Amazon
and Photo anomalies as localized attributed deviations.

## VecGAD Interpretation

VecGAD's Photo success suggests the useful signal may be:

- normal explanation failure in high-dimensional attributes;
- direction of failure, not only magnitude;
- raw identity preservation during multi-hop structural tokenization;
- hard-negative supervision generated from normal-node discrepancy directions.

## DualRefGAD Tension

Existing DualRefGAD records suggest:

- deviation-reference membership is not useless on Photo;
- `mat_mean` response aggregation is stably reversed or harmful;
- wide witness evidence partially recovers signal;
- fixed range/shape summaries outperform the current trainable reader;
- linear residual over fixed shape summaries failed to repair the gap.

Working interpretation:

> Photo likely contains useful normal-deviation evidence, but the current
> DualRefGAD response readout compresses it into the wrong scalar semantics.

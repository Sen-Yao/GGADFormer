# VecGAD Method Card

## Core Idea

VecGAD treats reconstruction residuals as vectorized directional evidence, not
as a scalar anomaly score. The reconstruction discrepancy vector indicates how
a node fails to be reconstructed by a normality-oriented autoencoder. VecGAD
then uses that direction to synthesize pseudo-anomalies as hard negatives.

## Mechanism

1. Build graph token sequences from offline propagation:
   - use normalized adjacency with self loops;
   - compute multi-hop propagated features;
   - inject residual identity preservation so each token retains a portion of
     the raw node features.
2. Encode each node's token sequence with a Transformer autoencoder.
3. Decode the compact embedding back to the token sequence.
4. Define the reconstruction discrepancy vector:
   - source: original token sequence minus reconstructed token sequence;
   - semantics: directional normal-explanation failure.
5. Project the discrepancy vector into embedding space.
6. Generate pseudo-anomaly embeddings by perturbing normal-node embeddings along
   the projected discrepancy direction.
7. Use a hyperspherical shell constraint to keep generated pseudo-anomalies:
   - outside the normal manifold;
   - not so far away that they become trivial negatives.
8. Train a binary classifier on labeled normal nodes versus generated
   pseudo-anomalies.

## Important Distinction

VecGAD is not merely "autoencoder reconstruction error." Its key claim is that
the direction of reconstruction failure contains anomaly semantics that scalar
residual scores discard.

## Components To Audit

- Residual-identity tokenization: whether Photo needs raw attribute identity
  preserved against over-smoothing.
- Discrepancy vector: whether Photo anomalies are directional failures relative
  to normal reconstruction.
- Hard-negative generation: whether DualRefGAD lacks a comparable training
  contrast.
- Hyperspherical shell: whether magnitude control matters more than readout
  capacity.

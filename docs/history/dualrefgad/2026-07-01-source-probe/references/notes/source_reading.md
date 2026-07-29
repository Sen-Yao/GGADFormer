# VecGAD Source Reading

Status: source inspection completed.

## Source Snapshot

- Repository: `https://github.com/Sen-Yao/GGADFormer.git`
- Local clone: `references/code/GGADFormer`
- Inspected commit: `28bce1a83bc87d7cd1d2dce423da7c79b296c5b7`
- Main files read:
  - `VecGAD.py`
  - `run.py`
  - `utils.py`
  - `ablation.py`
  - `docs/VecGAD.md`
  - `reproduction.sh`

The current branch has already been renamed from GGADFormer to VecGAD. The
older `reconstruct-outlier` branch is useful historically, but the main branch
contains the more explicit residual-guided hard-negative implementation.

## Mechanism Found In Source

VecGAD first builds NAGphormer-style tokens:

```text
t_i = [x_i, propagated_1(i), ..., propagated_k(i)]
```

`utils.py` constructs the token tensor with shape `[N, pp_k + 1, d]`.
Propagation uses `progregate_alpha` to preserve raw features while adding
multi-hop structural context.

The model then encodes these tokens with a Transformer encoder:

```text
h_i = E(t_i)
```

In `VecGAD.py`, the important components are:

```text
token_projection: raw token dim -> embedding dim
token_decoder: embedding -> flattened token sequence
reconstruction_proj: flattened token residual -> embedding-space direction
```

The core hard-negative step is:

```text
reconstructed_tokens = D_tok(h)
e_tok = reconstructed_tokens - input_tokens
e_emb = P(e_tok)
h_negative = h_normal + beta * e_emb
```

This is the key transferable idea. Reconstruction error is not used primarily
as a scalar anomaly score. It is kept as a structured vector, projected into
embedding space, and used as a direction for generating pseudo anomalies.

## Training Objective

During VecGAD training, each mini-batch identifies local known-normal nodes
from `normal_for_train_idx`. It then creates generated outlier embeddings from
a sampled subset of those known-normal embeddings.

The classifier is trained with BCE:

```text
known normal -> 0
generated outlier -> 1
```

The total loss combines:

```text
BCE classification loss
+ token/embedding reconstruction loss
+ ring/shell constraint on generated outliers
```

At inference, VecGAD does not generate pseudo anomalies. It uses the learned
classifier logits as the deployable anomaly score.

## Boundary / Shell Constraint

VecGAD constrains generated negatives to live in a ring around a center:

```text
R_min <= ||h_negative - center||_2 <= R_max
```

This matters because generated negatives that are too close to known normals
are not useful, while generated negatives that are too far away become trivial
separation examples. The shell is the mechanism that tries to make generated
examples boundary-like rather than arbitrary noise.

## Built-in Controls

`ablation.py` already implements the exact controls we should imitate:

- `random_dir`: keep residual magnitude but randomize direction.
- `random_mag`: keep residual direction but randomize magnitude.
- `random_both`: randomize both direction and magnitude.
- `constant_mag`: keep residual direction but use mean magnitude.

These controls are unusually useful for DualRefGAD because our failure mode is
not "does any signal exist"; it is "can a legal objective turn a structured
signal into a deployable anomaly score." Direction/magnitude controls test
whether the residual itself carries information, instead of merely injecting
regularizing noise.

## Photo Relevance

`reproduction.sh` includes a 5 percent `photo` run:

```text
--dataset=photo
--train_rate=0.05
--pp_k=6
--progregate_alpha=0.05
--rec_loss_weight=1
--ring_R_min=0.3
--ring_R_max=1
```

The docs report VecGAD Photo performance around AUC `0.8183 +/- 0.0202` and AP
`0.4756 +/- 0.0585` under the 5 percent setting. This does not prove the
DualRefGAD analogue will work, but it does rule out the weak explanation that
Photo is impossible simply because 5 percent supervision is too low or because
Transformer-style models cannot learn under this protocol.

## Mapping To DualRefGAD

The safe DualRefGAD transfer is not to copy VecGAD's exact token autoencoder.
The transfer should be:

```text
normal-calibrated evidence object
-> structured residual direction
-> shell-constrained hard negatives
-> BCE deployable anomaly score
```

This is different from the failed `R_D` story:

- Failed route: choose context nodes and hope the Transformer reads
  target-conditioned witness evidence.
- VecGAD-style route: train the model to separate known normals from generated
  boundary negatives whose direction is derived from normal-explanation
  residuals.

The probe should use the strongest already-known legal evidence objects first,
especially compact relation summaries and fixed diagnostic features from the
Photo diagnostic-to-deployable gap audit. It should avoid resurrecting raw
`R_D` or raw `R_N` token sequences as the residual source.

## Immediate Design Implication

A valid DualRefGAD probe must include at least these arms:

```text
residual_guided
random_dir
random_mag
random_both
scalar_magnitude_only
no_shell
known_normal_center vs all_node_center
```

If `residual_guided` beats random direction/magnitude controls under the same
evidence object and split, then the route has a real mechanism. If not, the
residual-guided story should be demoted quickly instead of becoming another
long reader-interface loop.

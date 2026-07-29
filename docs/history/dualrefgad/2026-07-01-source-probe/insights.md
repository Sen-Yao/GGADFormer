# Insights

Status: closed as source-level feasibility probe.

## Starting Interpretation

This investigation should not be treated as a request to copy VecGAD. The
problem is narrower:

```text
Can DualRefGAD convert already-detected diagnostic evidence into a legal
known-normal anomaly score by adopting residual-guided boundary negatives?
```

Prior constraints:

- Photo `R_D` / `R_N` / `E_N` context-reader probes failed causal controls.
- Photo diagnostic evidence is strong but not deployable under current
  known-normal scoring.
- VecGAD's reported Photo strength under 5 percent normal labels means Photo is
  not intrinsically impossible under this supervision level.

## Source-level Conclusion

The VecGAD source supports a worthwhile next probe, but for a precise reason:

```text
It offers an objective/protocol mechanism for converting normal-explanation
residuals into deployable anomaly scores.
```

It does not rescue the old `R_D` story directly. The failed `R_D` route tried
to select witness nodes and rely on a reader to discover target-witness
contrast. VecGAD's route is different: it creates generated boundary negatives
from known-normal nodes and trains the score head to separate known normals
from those hard negatives.

This matters for Photo because our strongest recent evidence is:

```text
diagnostic signal exists,
but legal downstream objectives fail to exploit it.
```

VecGAD's residual-guided hard-negative mechanism attacks that exact gap. It
turns a structured residual into a training signal, instead of asking a
Transformer reader to infer the score from context tokens.

## What Transfers

The transferable unit is not VecGAD's exact implementation. The transferable
unit is:

```text
structured residual direction
+ shell-constrained pseudo anomaly generation
+ BCE score-head training
+ direction/magnitude controls
```

For DualRefGAD, the analogous residual should be derived from a legal
normal-calibrated evidence object, not from raw `R_D(v)` or raw `R_N(v)` node
sequences. A first probe should use compact evidence that prior investigations
already found to preserve signal better than full response matrices or scalar
readouts:

- target descriptor / raw embedding;
- compact known-normal relation summaries;
- pre-matrix reference-set relation summaries;
- response-profile summaries only if used compactly;
- robust normal-manifold residual features from the diagnostic-to-deployable
  audit.

## What Does Not Transfer Cleanly

Three risks remain.

First, VecGAD's residual comes from reconstructing NAGphormer-style propagated
tokens. DualRefGAD's residual object is less obvious. If we define it poorly,
the probe becomes another hand-built heuristic.

Second, shell constraints are not cosmetic. Without a shell, generated
negatives can be too easy or too destructive. DualRefGAD must test shell
variants explicitly rather than assume a fixed radius works.

Third, success must beat direction/magnitude controls. If a random direction
or random magnitude performs similarly, then the method is merely regularized
noise injection, not residual-guided hard-negative learning.

## Recommended Next Probe

Build a minimal runner-backed probe with these arms on Photo first:

```text
A0 target_only_score_head
A1 fixed_evidence_supervised_oracle_diagnostic_control
A2 known_normal_autoencoder_scalar_residual_score
A3 residual_guided_hard_negative
A4 random_dir_control
A5 random_mag_control
A6 random_both_control
A7 residual_guided_no_shell
A8 residual_guided_known_normal_center
A9 residual_guided_all_node_center
```

The important readout is not only final AUC/AP. The probe should answer:

```text
Does a vector residual direction provide information beyond magnitude/noise?
Does shell-constrained generation improve deployable known-normal scoring?
Does this close part of Photo's diagnostic-to-deployable gap?
```

## Decision

This route is higher priority than another `R_D` constructor or reader-interface
iteration. The reason is not that VecGAD is fashionable; it is that the recent
DualRefGAD failure stack has narrowed the bottleneck to deployable objective
conversion. VecGAD-style residual-guided hard negatives are a direct test of
that bottleneck.

However, this investigation itself is only source-level evidence. It should be
closed here and handed off to a separate runner-backed implementation probe if
the user wants to test it.

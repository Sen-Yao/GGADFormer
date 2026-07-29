# VecGAD-style Residual-guided Hard Negative Probe

Date: 2026-07-01

Status: closed as source-level feasibility probe

## Research Question

Can DualRefGAD move beyond the failed Photo `R_D` context-reader story by
borrowing VecGAD's core mechanism:

```text
normal-explanation residual direction
-> boundary hard negatives
-> deployable known-normal anomaly score
```

The goal of this investigation is to inspect the actual VecGAD/GGADFormer
source code, identify the residual-guided pseudo-anomaly mechanism, and decide
whether a DualRef-specific probe is scientifically and technically worthwhile.

## Lineage

This investigation follows:

- `2026-07-01-dualrefgad-literature-route-solvency-audit`
- `2026-07-01-dualrefgad-photo-diagnostic-to-deployable-signal-gap-audit`
- `2026-06-30-dualrefgad-rd-target-conditioned-context-causal-audit`
- `2026-06-30-dualrefgad-rn-known-normal-context-readability-causal-audit`
- `2026-06-29-dualrefgad-photo-failure-autopsy`

The immediate motivation is:

```text
Photo has strong diagnostic evidence,
but current legal known-normal scoring remains weak.
```

## Scope

In scope:

- Clone the user's VecGAD/GGADFormer GitHub source into
  `references/code/`.
- Read source code paths that implement token construction, reconstruction
  discrepancy, pseudo-anomaly generation, shell/boundary constraints, and
  classifier training.
- Map VecGAD components onto possible DualRefGAD analogues.
- Produce a feasibility discussion and recommended probe design.

Out of scope for this first step:

- Launching runner experiments.
- Adding mainline DualRefGAD code.
- Claiming method success from literature/source inspection alone.

## Deliverables

- `references/code/`: cloned VecGAD/GGADFormer source snapshot.
- `references/notes/source_reading.md`: source-level mechanism notes.
- `insights.md`: feasibility conclusion and probe design.
- `PROGRESS.md`: auditable activity log.

## Conclusion

Source inspection supports a follow-up runner-backed probe. The transferable
idea is not the exact VecGAD architecture, but:

```text
structured normal-explanation residual
-> shell-constrained generated hard negatives
-> BCE deployable anomaly score
```

This is a better match to the current DualRefGAD bottleneck than another
`R_D` constructor/readout loop, because recent Photo investigations already
showed that strong diagnostic evidence exists but is not being converted into
legal downstream scores.

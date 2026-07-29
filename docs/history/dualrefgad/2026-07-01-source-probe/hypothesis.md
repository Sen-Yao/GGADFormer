# Hypotheses

## H1: Residual direction is the missing deployability bridge on Photo

Claim:

```text
Photo's diagnostic evidence becomes more deployable if DualRefGAD preserves a
vector/structured normal-explanation residual and turns it into boundary hard
negative training.
```

Support would require a future runner-backed probe where residual-guided hard
negatives beat random/noise negatives and current legal known-normal scoring.

## H2: VecGAD's transferable idea is mechanism, not architecture

Claim:

```text
DualRefGAD should not copy VecGAD's exact autoencoder. It should transfer the
mechanism: normal reconstruction/explanation failure direction -> pseudo
anomaly or boundary negative objective.
```

Source inspection should identify whether VecGAD's gains depend on a compact,
portable set of ideas or on a tightly coupled implementation.

## H3: A naive scalar residual is insufficient

Claim:

```text
If the residual is scalarized before generating negatives, the method collapses
back to the failed scalar-score loop.
```

A valid probe should include vector residual versus scalar residual magnitude
controls.

## H4: The probe is worth implementing only if controls are clear

Claim:

```text
The successor experiment is worthwhile only if it can test residual direction,
shell/boundary placement, and legal known-normal deployability separately.
```

If source inspection cannot support clean controls, this route should be
demoted before touching mainline code.

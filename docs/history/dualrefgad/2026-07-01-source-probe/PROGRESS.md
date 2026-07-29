# Progress

## 2026-07-01

- Confirmed main DualRefGAD repository was clean before opening this
  investigation.
- Fixed stale `Status: active` markers in two closed ignored investigation
  notes.
- Created clean-state checkpoint commit:
  `b20ad1b Record clean state before VecGAD hard negative probe`.
- Created investigation skeleton.
- Located the user's VecGAD/GGADFormer source:
  `https://github.com/Sen-Yao/GGADFormer.git`.
- Cloned source into:
  `references/code/GGADFormer`.
- Inspected cloned source at commit:
  `28bce1a83bc87d7cd1d2dce423da7c79b296c5b7`.
- Read and summarized:
  - `VecGAD.py`
  - `run.py`
  - `utils.py`
  - `ablation.py`
  - `docs/VecGAD.md`
  - `reproduction.sh`
- Recorded mechanism mapping in `references/notes/source_reading.md`.
- Recorded feasibility decision in `insights.md`.

Closure:

```text
Close this as a source-level feasibility probe.
Recommended successor is a runner-backed DualRefGAD residual-guided
hard-negative implementation probe with direction/magnitude/shell controls.
```

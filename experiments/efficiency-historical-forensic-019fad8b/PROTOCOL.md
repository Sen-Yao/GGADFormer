# VecGAD historical efficiency 2x2 forensic protocol

This is an internal diagnostic. It is not manuscript or supplement evidence unless the
user explicitly promotes it after reviewing every cell.

## Question

The submitted PDF reports VecGAD epoch times read from historical tqdm progress bars,
while the later HCCS-90 audit uses CUDA-synchronized wall-clock timing. This forensic
tests whether the discrepancy can be attributed to either of two frozen factors:

1. historical source commit: `e071ae6646451d94fc8e8c9e88305eb76c393089` or
   `5bf8205b0d4c54d583b13c547ae62122ffdf2f6a`;
2. execution batch mode: the batch size logged by the historical W&B sweep or the
   full-batch interpretation stated in the PDF.

## Frozen scope

- Methods: VecGAD only.
- Datasets: Amazon and T-Finance.
- Repeats: three fresh child processes per cell.
- Design: 2 source commits x 2 batch modes x 2 datasets x 3 repeats = 24 trials.
- Runtime: HCCS-90, GPUs 0--7, RTX 4090 24 GB. Physical GPU index is metadata,
  not an experimental factor.
- Each fresh process runs 10 warm-up epochs and one 30-epoch measured block.
- Seed is fixed at 0 in every repeat so process-to-process timing variance is not
  mixed with a seed factor.
- No evaluation, checkpointing, or external W&B logging is executed inside the
  measured child loop. The historical even-epoch `.item()` calls are retained because
  they were evaluated before historical `wandb.log` calls and therefore forced CUDA
  synchronization in the original progress-bar path.

## Historical configuration provenance

- Amazon: W&B `HCCS/GGADFormer/8ylmsq7q`, `batch_size=1024`, `pp_k=5`,
  `progregate_alpha=0.3`, `num_epoch=100`, `peak_lr=3e-4`, `end_lr=1e-4`.
- T-Finance: W&B `HCCS/GGADFormer/iqxjqsdl`, source commit `e071ae66`,
  `batch_size=8192`, `pp_k=7`, `progregate_alpha=0.3`, `num_epoch=40`,
  `peak_lr=5e-4`, `end_lr=1e-4`.
- Full-batch replaces only `batch_size` with the observed node count: 11,944 for
  Amazon and 39,357 for T-Finance.
- Commit `e071ae66` uses `num_workers=4`, `persistent_workers=True`, and
  `pin_memory=True`.
- Commit `5bf8205b` uses `num_workers=0`, `persistent_workers=False`, and
  `pin_memory=False`.

## Timing estimands

Each measured block records both estimands without selecting between them after seeing
results:

1. `tqdm_terminal_rate_it_per_second`: the terminal exponential-moving-average rate
   exposed by tqdm immediately after the 30th update and before the final explicit
   CUDA synchronization. This is the automated analogue of reading `it/s` from the
   historical progress bar. It is intentionally retained as a diagnostic of the old
   measurement surface and is not treated as canonical elapsed time.
2. `synchronized_block_seconds`: wall-clock elapsed time from a CUDA-synchronized
   start to a CUDA-synchronized end around all 30 epochs. Canonical per-epoch time is
   this value divided by 30.

The measured block includes DataLoader/sampler work, host-to-device transfer, forward,
loss, backward, optimizer, scheduler, the historical even-epoch scalar extraction,
and tqdm rendering. Dataset loading and offline tokenization are outside the block;
tokenization time is recorded separately.

## Validity and interpretation

- The harness imports `GGADFormer.py` and `utils.py` directly from clean detached
  worktrees at the declared commits and records their SHA-256 digests.
- A trial is valid only when source commit, loader contract, dataset node count,
  batch mode, optimizer-step count, runtime identity, W&B identity, and all 30 measured
  epochs match the registry.
- All 24 assignments must reach a declared terminal status before aggregation.
- No batch fallback, worker adjustment, K/alpha change, hidden-dimension change, or
  result-driven rerun is allowed.
- The result may diagnose a historical measurement mechanism. It cannot establish an
  L40 reproduction because the original L40 environment is unavailable.

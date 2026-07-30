# Elliptic loss-weight unification formal protocol

## Objective and boundary

This experiment tests whether Elliptic can use the paper-level unified loss
weights `lambda_rec_emb=0.1` and `ring_loss_weight=1` without losing practical
equivalence against a same-code, same-protocol control.

Only two variant identities are allowed:

| Variant | `lambda_rec_emb` | `ring_loss_weight` |
|---|---:|---:|
| `control_2_20` | 2.0 | 20.0 |
| `unified_0p1_1` | 0.1 | 1.0 |

All other effective hyperparameters, split identity, seeds, sampler, code path,
training budget, and metric policy are fixed. Each `variant x seed` is an
independent native W&B run, for 10 expected valid trials.

The fixed endpoint is the only decision metric: use W&B history row
`_step=150`, fields `AUC` and `AP`, reported as `AUC.last` and `AP.last`.
Do not use best epoch, `AUC.max`, `AP.max`, early stopping, or hand-picked runs.

## Predeclared decision rule

The unified candidate passes practical equivalence only if both mean drops
relative to same-protocol control are within the fixed thresholds:

- AUROC mean drop <= `0.01`;
- AUPRC mean drop <= `0.02`.

The final report must also check whether the candidate remains above GGAD on
the current main-table Elliptic values: AUROC `0.7006`, AUPRC `0.2565`.

## Fixed command family

Control:

```bash
python run.py --batch_size=32768 --dataset=elliptic --end_lr=0.0003 --lambda_rec_emb=2 --num_epoch=150 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=7 --progregate_alpha=0.6 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=20 --seed=<seed> --train_rate=0.05 --warmup_updates=50
```

Unified candidate:

```bash
python run.py --batch_size=32768 --dataset=elliptic --end_lr=0.0003 --lambda_rec_emb=0.1 --num_epoch=150 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=7 --progregate_alpha=0.6 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=<seed> --train_rate=0.05 --warmup_updates=50
```

`sweep.yaml` intentionally grids only `variant x seed`. The committed
`run-sweep-trial.py` wrapper maps each variant to the only legal full argv, so
W&B cannot create unplanned cross-products such as `0.1/20` or `2/1`.

## Execution contract

- Execution host is fixed to `HCCS-90`; no fallback host is allowed.
- W&B destination is fixed to `HCCS/GGADFormer`.
- Code is run from a clean detached worktree at the committed full SHA recorded
  in `manifest.yaml`.
- Runtime must reuse the existing HCCS-90 VecGAD conda environment after live
  import/help/smoke checks.
- Data must bind the existing read-only Elliptic cache, expected at
  `/root/gpufree-data/linziyao/DualRefGAD/dataset/elliptic.mat`; missing or
  unreadable data fails closed.
- Use only GPUs with no pre-existing compute PID at the immediate launch probe.
- Use one tmux pane/native W&B agent per selected GPU, `remain-on-exit=on`.
- Set `WANDB_DISABLE_CODE=true` and `WANDB_CONSOLE=off`. Do not upload raw data,
  source code, credentials, checkpoints, or undeclared artifacts.
- Use only one Codex one-shot heartbeat for long-run follow-up. Do not use
  Experiment Console, GPUHub, custom watchers, polling scripts, or recurring
  automation.

## Completion criteria

All 10 expected trials must be W&B `finished`; agent exit codes must be zero;
every run must match dataset, variant, seed, hyperparameters, code SHA, host,
GPU, and fixed-final-epoch policy; and W&B history must contain final step 150.
Any crash, missing history, failed run, identity mismatch, or unexpected extra
trial blocks formal aggregation until diagnosed.

Final results must be written to `results.json`, W&B replay evidence to
`authoritative-sweep.json`, and remote log hashes to `remote-log-sha256.txt`.
This experiment does not modify the paper main table, Supplement, or
`reproduction.sh`.

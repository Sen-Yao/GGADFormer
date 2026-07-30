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

## Formal outcome

W&B sweep: [HCCS/GGADFormer/l6ubfjxt](https://wandb.ai/HCCS/GGADFormer/sweeps/l6ubfjxt).
All 10 runs finished successfully. Every run matched the committed code SHA,
dataset, variant, seed, effective configuration, host, GPU, split identity and
metric policy; every W&B history contained exactly one final row at `_step=150`.

| Variant | Seed | Run | AUC.last | AP.last |
|---|---:|---|---:|---:|
| `control_2_20` | 0 | [27vg0m3g](https://wandb.ai/HCCS/GGADFormer/runs/27vg0m3g) | 0.8138869401 | 0.4031653396 |
| `control_2_20` | 1 | [d5818dpg](https://wandb.ai/HCCS/GGADFormer/runs/d5818dpg) | 0.7605318736 | 0.2320502575 |
| `control_2_20` | 2 | [4iq0q6ew](https://wandb.ai/HCCS/GGADFormer/runs/4iq0q6ew) | 0.7481187174 | 0.3138529234 |
| `control_2_20` | 3 | [fmplzj9d](https://wandb.ai/HCCS/GGADFormer/runs/fmplzj9d) | 0.7779013920 | 0.3234603053 |
| `control_2_20` | 4 | [20ni08r3](https://wandb.ai/HCCS/GGADFormer/runs/20ni08r3) | 0.7398626576 | 0.2020723209 |
| `unified_0p1_1` | 0 | [y9wz9q9x](https://wandb.ai/HCCS/GGADFormer/runs/y9wz9q9x) | 0.5740399702 | 0.1028485481 |
| `unified_0p1_1` | 1 | [chlavgv0](https://wandb.ai/HCCS/GGADFormer/runs/chlavgv0) | 0.5882983557 | 0.1073180959 |
| `unified_0p1_1` | 2 | [8re70ocx](https://wandb.ai/HCCS/GGADFormer/runs/8re70ocx) | 0.6195653164 | 0.1146246739 |
| `unified_0p1_1` | 3 | [l53pk0pa](https://wandb.ai/HCCS/GGADFormer/runs/l53pk0pa) | 0.6146241323 | 0.1141297082 |
| `unified_0p1_1` | 4 | [cz8v5hlo](https://wandb.ai/HCCS/GGADFormer/runs/cz8v5hlo) | 0.5770277853 | 0.1067040926 |

| Variant | Mean AUC.last | Sample std | Mean AP.last | Sample std |
|---|---:|---:|---:|---:|
| `control_2_20` | 0.7680603161 | 0.0293529342 | 0.2949202293 | 0.0797862053 |
| `unified_0p1_1` | 0.5947111120 | 0.0211859590 | 0.1091250237 | 0.0050944347 |

The paired mean differences (`unified - control`) are `-0.1733492041` for
AUROC and `-0.1857952056` for AUPRC. Both exceed the predeclared tolerated
drops (`0.01` and `0.02`), so the unified candidate fails practical
equivalence on Elliptic under this protocol. Its mean AUROC and AUPRC are also
below GGAD's current main-table references (`0.7006/0.2565`), so it does not
retain the runner-up ranking on either metric.

This is a bounded conclusion about the jointly changed `0.1/1` candidate under
the declared Elliptic protocol. No intermediate loss-weight search was run,
and the result does not identify which of the two changed weights is causal.
There were no scientific protocol deviations. A transient W&B TLS EOF occurred
during evidence collection; the identical direct-HTTPS query succeeded on a
bounded retry. The experiment completed before a one-shot heartbeat was needed.

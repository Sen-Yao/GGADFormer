# Tolokers `lambda_rec_emb` 统一权重正式实验协议

## 目标与边界

本实验检验 Tolokers 是否可以跟随论文统一 loss 权重设置，将
`lambda_rec_emb` 从当前 Tolokers control 的 `5` 降为 `0.1`。`lambda_HSC`
对应实现参数 `ring_loss_weight`，本实验中 control 与 candidate 均保持为 `1`。

实验只比较以下两个配置身份：

| Variant | `lambda_rec_emb` | `ring_loss_weight` |
|---|---:|---:|
| `control` | 5.0 | 1.0 |
| `unified_0p1_1` | 0.1 | 1.0 |

除 `lambda_rec_emb` 外，两个配置的 dataset、split、seed、optimizer、epoch、
batch size、propagation、warmup、ring 半径和其他 loss 权重完全相同。每个
variant 运行 seeds `0,1,2,3,4`，共 10 个独立 W&B trials。

固定训练终点为唯一裁决 metric：报告 `AUC.last` 和 `AP.last`，不使用
`AUC.max`、`AP.max`、best epoch 或手选 run。

## 预声明裁决门槛

`unified_0p1_1` 相对同 seed、同协议 `control` 的 mean drop 满足以下条件时，
Tolokers 可视为通过实用等效门槛：

- AUROC mean drop 不超过 `0.01`；
- AUPRC mean drop 不超过 `0.02`。

门槛只用于结果裁决，不用于提前停止、调参或改变任何配置。最终报告必须同时
给出逐 seed 原始值、mean、sample std (ddof=1)、paired differences 和 paired
mean difference。

## 固定命令族

Control:

```bash
python run.py --batch_size=1024 --dataset=tolokers --end_lr=0.00001 --lambda_rec_emb=5 --num_epoch=100 --outlier_beta=0.3 --peak_lr=0.0001 --pp_k=10 --progregate_alpha=0.9 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=<seed> --train_rate=0.05 --warmup_updates=5
```

Unified candidate:

```bash
python run.py --batch_size=1024 --dataset=tolokers --end_lr=0.00001 --lambda_rec_emb=0.1 --num_epoch=100 --outlier_beta=0.3 --peak_lr=0.0001 --pp_k=10 --progregate_alpha=0.9 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=<seed> --train_rate=0.05 --warmup_updates=5
```

`sweep.yaml` implements the same protocol as one native W&B grid sweep over
`lambda_rec_emb in [5, 0.1]` and `seed in [0,1,2,3,4]`.

## Execution Contract

- 唯一执行主机：`HCCS-90`；不回退到 HCCS-25、HCCS-85 或其他机器。
- W&B destination：`HCCS/GGADFormer`。
- 代码 SHA：`bb798db0e32615abd8504da7ccb21a124102b363`。
- 运行 worktree：HCCS-90 task-owned durable path 下从上述 SHA 创建的 clean
  detached worktree。
- Python runtime：复用 HCCS-90 上已审计的
  `/root/gpufree-data/linziyao/.conda/envs/VecGAD-28bce1a8`。
- 数据：只读复用
  `/root/gpufree-data/linziyao/DualRefGAD/dataset/tolokers.mat`。
- GPU：只使用 launch 前 live probe 中无 pre-existing compute PID 的 GPU；不杀死
  或信号任何外来进程。
- tmux：每个 GPU 一个 task-owned pane/agent，`remain-on-exit=on`，W&B native
  agent 分配 trial。
- W&B：设置 `WANDB_DISABLE_CODE=true`、`WANDB_CONSOLE=off`，不上传源码、原始数据、
  credentials、checkpoint 或未声明 artifact。
- 监控：长运行只使用 Codex one-shot heartbeat，不使用 Experiment Console、
  GPUHub、自研 watcher、循环轮询脚本或 recurring automation。

## 完成条件

10 个 expected trials 全部 W&B `finished`，且每个 run 的 dataset、variant、
seed、lambda、code SHA、host、GPU、fixed-final-epoch metric policy、history 和
tmux exit code 与 manifest 一致。任何 failed/crashed run、missing artifact、
identity mismatch 或非 0 agent exit code 都使本实验进入诊断，不做正式聚合。


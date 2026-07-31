# Elliptic unified-loss bundle factorial

## 研究问题

父 sweep `k5lbpsg9` 已证明，在历史 mixed bundle 与固定 step 200 下，
`unified_0p1_1` 相对 matched `control_2_20` 同时通过 AUROC `-0.01` 与
AUPRC `-0.02` 实用等效门槛。本实验只在 unified `0.1/1` 下拆分 mixed
bundle，用 2x2 factorial 区分 optimization bundle 与 propagation bundle 的
主效应和交互作用。

## 冻结因子

- `optimization=current`：`batch_size=32768`、`end_lr=0.0003`、
  `num_epoch=150`；
- `optimization=mixed`：`batch_size=8192`、`end_lr=0.0001`、
  `num_epoch=200`；
- `propagation=current`：`pp_k=7`、`progregate_alpha=0.6`；
- `propagation=mixed`：`pp_k=8`、`progregate_alpha=0.8`。

四个 cell 均运行 seeds `0..4`，共 20 个独立 native W&B trials。其他参数固定：
`lambda_rec_emb=0.1`、`ring_loss_weight=1`、`peak_lr=0.0005`、
`warmup_updates=50`、`outlier_beta=0.3`、`rec_loss_weight=1`、
`ring_R_min=0.3`、`ring_R_max=1`、`train_rate=0.05`。

## 终点与估计量

每个 trial 只取其 optimization cell 的固定训练终点：`current` 为 step 150，
`mixed` 为 step 200。禁止使用 `AUC.max`、`AP.max`、best epoch、early stopping
或跨 cell 手选 checkpoint。

对 AUROC 与 AUPRC 分别报告四个 cell 的 mean 和 sample std (`ddof=1`)；在
同 seed 内计算 optimization 主效应、propagation 主效应、两类 simple effects
以及 difference-in-differences interaction，再跨五个 seed 聚合。该 factorial
用于机制归因，不预声明新的性能晋升门槛，也不据结果追加中间配置。


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

## 固定终点结果

W&B sweep [`bghcjp76`](https://wandb.ai/HCCS/GGADFormer/sweeps/bghcjp76)
的 20 个 run 均为 `finished`。显式 `scan_history` 已验证：current optimization
的五个 seed 在 `_step=150` 取值，mixed optimization 的五个 seed 在
`_step=200` 取值；所有 run 的完整 performance history、逐 epoch/逐 batch
diagnostic history、config、seed、代码 SHA、GPU、pane 与 agent exit identity
均通过核验。四个 cell 的五 seed 聚合为：

| Optimization | Propagation | AUROC mean +/- std | AUPRC mean +/- std |
|---|---|---:|---:|
| `current` | `current` | 0.594711 +/- 0.021186 | 0.109125 +/- 0.005094 |
| `current` | `mixed` | 0.607488 +/- 0.042961 | 0.114572 +/- 0.013370 |
| `mixed` | `current` | 0.647282 +/- 0.042396 | 0.130835 +/- 0.022100 |
| `mixed` | `mixed` | 0.638637 +/- 0.058220 | 0.126635 +/- 0.022010 |

离散度均为 sample standard deviation (`ddof=1`)。同 seed 配对后的 factorial
效应如下，正值表示从 current 切换到 mixed 后指标增加：

| Paired effect | mean delta AUROC | mean delta AUPRC |
|---|---:|---:|
| optimization main effect | +0.041860 | +0.016886 |
| propagation main effect | +0.002066 | +0.000624 |
| optimization at current propagation | +0.052571 | +0.021710 |
| optimization at mixed propagation | +0.031148 | +0.012063 |
| propagation at current optimization | +0.012777 | +0.005447 |
| propagation at mixed optimization | -0.008645 | -0.004200 |
| interaction (difference in differences) | -0.021423 | -0.009647 |
| full mixed minus full current | +0.043925 | +0.017510 |

## 诊断解释

固定终点、按 pseudo count 加权的 HSC 诊断显示，current optimization 下
current/mixed propagation 的 shell-hit 分别为 `0.0000/0.0752`，outer
violation 为 `1.0000/0.9248`；切换到 mixed optimization 后，两种 propagation
的 shell-hit 升至 `0.9719/0.9772`，outer violation 降至 `0.0281/0.0228`。
相应地，HSC raw loss 从 `0.5374/0.2560` 降至 `0.0098/0.0064`，BCE raw
loss 从 `0.1110/0.1008` 降至 `0.0009/0.0007`。REC combined raw loss 的
四 cell 均值依次为 `4.4392`、`4.4058`、`1.6940`、`2.8638`。

这些观测与 optimization bundle 的较大性能主效应方向一致：较小 batch、较低
end LR 和更长训练组成的 bundle 改变了固定终点的 HSC 几何状态，并解释了历史
mixed 配置为何能让 `0.1/1` 获得相对更高的绝对性能。传播 bundle 的平均贡献很小，
且在 mixed optimization 下略为负；负交互项进一步表明两个 bundle 不能按独立、
可加的增益理解。但本设计同时改变 batch size、end LR 和 epoch，并使用各 cell
预声明的不同固定终点，因此不能把改善单独归因于其中任一变量，也不能把终点诊断
写成因果证明。

## 结论边界

父 sweep 已表明，在完整 historical mixed bundle 内，`unified_0p1_1` 相对 matched
`control_2_20` 的 paired mean difference 为 AUROC `-0.007251`、AUPRC
`-0.003843`，同时通过预声明等效门槛。本 factorial 进一步支持：该 bundle 的性能
恢复主要来自 optimization bundle，而不是 `K/alpha` propagation bundle。最佳均值
cell 是 mixed optimization + current propagation，但其 `0.647282/0.130835` 仍明显
低于当前协议 `control_2_20` 的 `0.768060/0.294920`，也未同时高于 GGAD 的
`0.7006/0.2565`。因此，历史 mixed 结果能帮助定位 `0.1/1` 更接近 matched mixed
control 的条件，却没有恢复当前协议 control 的绝对性能。

本阶段没有运行新的 loss 权重、中间 bundle、best epoch 或结果驱动搜索；也没有修改
论文主表、Supplement 或 `reproduction.sh`。若进一步拆分 batch size、end LR 与
epoch，需要另行预声明新的正交实验，不能由当前结果自动触发。

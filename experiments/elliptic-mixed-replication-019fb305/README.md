# Elliptic historical mixed-bundle paired replication

## 科学问题

历史 W&B run `xvocbgqu` 表明，固定 `lambda_rec_emb=0.1`、
`ring_loss_weight=1` 时，较小 batch、较长训练、较低 end LR 与不同传播配置
可能显著改善 Elliptic AUROC。本实验在当前 VecGAD 科学基线
`655d6293bb76633bc6aa6fd21166a49c3b91d504` 上正式复现该 mixed bundle，
并在同一协议下运行 `2/20` control，避免跨协议比较。

## 冻结设计

- 两个 variant：`control_2_20` 与 `unified_0p1_1`；
- seeds `0..4`，共 10 个独立 native W&B trials；
- 两组共享 `batch_size=8192`、`num_epoch=200`、`end_lr=1e-4`、
  `peak_lr=5e-4`、`pp_k=8`、`progregate_alpha=0.8`；
- primary endpoint 固定为 W&B history `_step=200` 的 `AUC.last/AP.last`；
- step 130/150 仅保留在完整 history 中，不用于选择、停止或裁决；
- candidate 相对 matched control 的 mean drop 门槛固定为 AUROC `0.01`、
  AUPRC `0.02`；两项都通过才允许进入后续 optimization/propagation bundle
  拆分实验。

## 诊断记录

训练目标、forward、backward、optimizer、scheduler 与 RNG 路径保持 scientific
base 不变。execution commit 只在每次已有 batch update 后，从 detach 的
`emb/outlier_emb/loss` 读取并向 W&B 记录：

- HSC shell-hit、inner/outer violation 与伪异常到 batch center 的均值距离；
- raw/weighted BCE、combined REC、HSC 与真实 weighted total；
- 每个 batch 的 pseudo count。

所有诊断 key 使用 `diagnostic/batch_<0..5>/...`，不上传 raw data、源码、
checkpoint 或 custom artifact。

## 后续 gate

只有本实验 joint practical equivalence 通过，才准备第二个独立 sweep，把
historical/current optimization bundle 与 propagation bundle 做 2x2 拆分。
本 sweep 不包含该拆分，也不搜索新的 loss 权重或 best epoch。

## 固定终点结果

W&B sweep `k5lbpsg9` 的 10 个 run 均为 `finished`，且显式
`scan_history` 验证了每个 run 的 `_step=200`。逐 seed 结果如下：

| variant | seed | AUROC | AUPRC |
|---|---:|---:|---:|
| `control_2_20` | 0 | 0.710262 | 0.158011 |
| `control_2_20` | 1 | 0.580432 | 0.103992 |
| `control_2_20` | 2 | 0.604321 | 0.118523 |
| `control_2_20` | 3 | 0.644071 | 0.126697 |
| `control_2_20` | 4 | 0.690352 | 0.145167 |
| `unified_0p1_1` | 0 | 0.705038 | 0.150515 |
| `unified_0p1_1` | 1 | 0.690945 | 0.149668 |
| `unified_0p1_1` | 2 | 0.626636 | 0.116799 |
| `unified_0p1_1` | 3 | 0.602243 | 0.113237 |
| `unified_0p1_1` | 4 | 0.568320 | 0.102956 |

`control_2_20` 的 AUROC/AUPRC 为
`0.645888 +/- 0.055083` / `0.130478 +/- 0.021404`，
`unified_0p1_1` 为
`0.638637 +/- 0.058220` / `0.126635 +/- 0.022010`，其中离散度为
sample standard deviation (`ddof=1`)。同 seed 的 candidate-control mean
difference 为 AUROC `-0.007251`、AUPRC `-0.003843`，分别满足预声明的
`-0.01` 与 `-0.02` 门槛，因此 joint practical equivalence 通过。

## 结论边界

该结果支持的窄结论是：在历史 mixed bundle 与固定 step 200 下，把两个权重
从 `2/20` 同时降为 `0.1/1` 没有造成超过预声明容忍度的额外平均损失。它不支持
“mixed bundle 已恢复主表性能”：两组绝对性能都明显低于当前主表 VecGAD，
candidate 也未高于 GGAD 的 `0.7006/0.2565`。因此，历史 `xvocbgqu` 的次级结果
可以作为“优化与传播协议会显著改变 0.1/1 表现”的线索，但不能单独证明
`0.1/1` 已接近当前协议 control。

固定终点的 HSC 诊断显示，candidate 的 shell-hit 从 `0.963838` 升至
`0.977203`，outer violation 从 `0.036162` 降至 `0.022797`；同时 HSC weighted
loss 从 `0.071056` 降至 `0.006397`。这些量说明降权后伪异常几何并未失控，
但它们是机制诊断，不构成性能因果归因。由于 joint gate 已通过，下一阶段按
预声明规则运行 unified `0.1/1` 下 optimization bundle x propagation bundle
的 2x2 factorial，并保持 seeds `0..4` 与固定各 cell 训练终点。

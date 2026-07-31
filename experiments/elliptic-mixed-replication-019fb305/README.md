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

# Tolokers 严格方向--幅值控制：统一 `lambda_rec_emb=0.1`

本协议只重跑 Tolokers 的 Strict Direction--Magnitude Controls，用权威主实验配置
替换历史 strict sweep 中的 `lambda_rec_emb=5`。除这一参数外，训练配置和严格
控制实现均保持预声明不变。

正式证据单元为五种变体 `none`、`random_dir`、`random_mag`、
`random_both`、`constant_mag` 与 seeds 0--4 的笛卡尔积，共 25 个原生 W&B
trial。指标使用固定训练终点的 `AUC.last` 和 `AP.last`，聚合报告均值与
sample standard deviation（`ddof=1`），不允许 best-seed 或 best-epoch 选择。

严格控制代码绑定到 `fdb150b7927f26f2e8b5270365a324d844dc8b98`：

- `random_dir` 保留原模长并使用独立 RNG 采样随机单位方向；
- `random_mag` 保留方向并对当前生成子集的模长执行非零 cyclic permutation；
- `random_both` 使用随机单位方向和同一 exact magnitude permutation 机制；
- `constant_mag` 保留方向并使用当前生成子集的平均模长；
- 方向/幅值 RNG seed 分别为 `seed * 1000003 + 1729` 和
  `seed * 1000003 + 7919`。

W&B 目的地为 `HCCS/GGADFormer`。仅发送 config、seed、运行状态、metadata、
AUROC 和 AP；不上传源码、原始数据、checkpoint、控制台输出或未声明 artifact。


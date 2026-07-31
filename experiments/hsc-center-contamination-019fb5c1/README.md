# HSC 中心污染压力实验

本实验只改变 VecGAD 训练时 HSC 使用的 batch 中心，不改变采样器、初始化、训练顺序、伪异常生成、BCE、重构损失、分类器或测试评分。实际代码中的 Default 中心是当前重采样 mini-batch 的 embedding 均值，不是一次性计算的整图均值。

受控中心为：

```text
c_B(q) = (1 - q) * mean(h_i | y_i = 0) + q * mean(h_i | y_i = 1)
```

条件为 `default`、`q0`、`q10`、`q20`、`q30`、`q40`。其中 `default` 保留原实现；其他条件只允许真实标签进入 oracle 中心构造，不允许标签进入其他 loss 或 scoring。任一 batch 缺少正常类或异常类时 fail closed。

正式设计固定为 Amazon、Tolokers，seed `0..4`，data split seed `42`，共 60 个独立 W&B trial。每个 dataset/seed 的 6 个条件必须具有相同的初始化哈希、训练 batch trace、伪异常 source trace、诊断 batch trace 和诊断 source trace。

主指标是固定训练终点的 `AUC.last`、`AP.last`。每个 run 保存但不上传 final checkpoint，并从该 checkpoint 重放两次固定诊断，要求完全一致。诊断记录 ShellHit、inner violation、outer violation、mean HSC loss、相对 Default 中心位移、相对 oracle normal 中心位移和采样异常比例。

汇总仅报告均值、sample standard deviation、相对 Default 的逐 seed paired delta 和 95% paired t confidence interval；不设置结果判定门槛。

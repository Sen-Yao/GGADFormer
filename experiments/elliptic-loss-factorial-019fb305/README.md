# Elliptic loss-weight factorial follow-up

## 科学问题

上一轮受控实验表明，将 `lambda_rec_emb` 与 `ring_loss_weight` 从 `2/20`
同时降至 `0.1/1` 会显著降低 Elliptic 性能。本 follow-up 补齐两个混合角点，
用完整 2x2 factorial 区分两个权重的简单效应与交互效应。

| Factorial cell | `lambda_rec_emb` | `ring_loss_weight` | Evidence |
|---|---:|---:|---|
| `control_2_20` | 2.0 | 20.0 | frozen sweep `l6ubfjxt` |
| `emb_only_0p1_20` | 0.1 | 20.0 | this sweep, seeds 0-4 |
| `ring_only_2_1` | 2.0 | 1.0 | this sweep, seeds 0-4 |
| `unified_0p1_1` | 0.1 | 1.0 | frozen sweep `l6ubfjxt` |

本轮只运行两个新增角点，各 seeds `0..4`，共 10 个独立 W&B trials。
上一轮 control/unified 结果不重跑、不选择性替换，也不参与 W&B assignment。

## 冻结分析

固定使用 `_step=150` 的 `AUC` 与 `AP`，不使用 max/best epoch。
对每个 seed 计算：

- `emb_only_0p1_20 - control_2_20`：在 `ring_loss_weight=20` 时降低
  `lambda_rec_emb` 的简单效应；
- `unified_0p1_1 - ring_only_2_1`：在 `ring_loss_weight=1` 时降低
  `lambda_rec_emb` 的简单效应；
- `ring_only_2_1 - control_2_20`：在 `lambda_rec_emb=2` 时降低
  `ring_loss_weight` 的简单效应；
- `unified_0p1_1 - emb_only_0p1_20`：在 `lambda_rec_emb=0.1` 时降低
  `ring_loss_weight` 的简单效应；
- 交互项：`unified - emb_only - ring_only + control`。

每个 cell 报告 mean 与 sample std (`ddof=1`)，每个效应报告逐 seed 差值与
paired mean。该实验用于描述两个权重在这两个离散水平上的影响，不外推连续响应曲线，
也不据结果追加其他权重搜索。

## 固定协议

除两项权重构成的 factorial cell 外，命令与上一轮完全一致：

```bash
python run.py --batch_size=32768 --dataset=elliptic --end_lr=0.0003 \
  --lambda_rec_emb=<0.1-or-2> --num_epoch=150 --outlier_beta=0.3 \
  --peak_lr=0.0005 --pp_k=7 --progregate_alpha=0.6 \
  --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 \
  --ring_loss_weight=<1-or-20> --seed=<0..4> --train_rate=0.05 \
  --warmup_updates=50
```

运行主机固定为 HCCS-90，W&B destination 固定为 `HCCS/GGADFormer`。
训练代码绑定 `655d6293bb76633bc6aa6fd21166a49c3b91d504` 的独立 clean detached
worktree。不得上传 raw data、source code、credentials、checkpoint 或未声明 artifact。


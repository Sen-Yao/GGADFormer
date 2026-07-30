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

## 正式结果

所有指标均来自 W&B history 的固定 `_step=150`；下表为五个 seeds 的
mean +/- sample std (`ddof=1`)。

| Factorial cell | AUROC | AUPRC |
|---|---:|---:|
| `control_2_20` | 0.7681 +/- 0.0294 | 0.2949 +/- 0.0798 |
| `emb_only_0p1_20` | 0.6031 +/- 0.0403 | 0.1150 +/- 0.0158 |
| `ring_only_2_1` | 0.4893 +/- 0.1367 | 0.0924 +/- 0.0238 |
| `unified_0p1_1` | 0.5947 +/- 0.0212 | 0.1091 +/- 0.0051 |

| Paired effect (low minus high) | mean delta AUROC | mean delta AUPRC |
|---|---:|---:|
| `lambda_rec_emb`, at `ring_loss_weight=20` | -0.1650 | -0.1799 |
| `lambda_rec_emb`, at `ring_loss_weight=1` | +0.1054 | +0.0167 |
| `ring_loss_weight`, at `lambda_rec_emb=2` | -0.2787 | -0.2025 |
| `ring_loss_weight`, at `lambda_rec_emb=0.1` | -0.0083 | -0.0059 |
| interaction, difference of differences | +0.2704 | +0.1966 |
| `lambda_rec_emb` marginal descriptive effect | -0.0298 | -0.0816 |
| `ring_loss_weight` marginal descriptive effect | -0.1435 | -0.1042 |

结果表明两个权重存在强交互，不能把任一权重解释为与另一权重无关的单调控制项。
在另一权重保持高值时，单独降低任一权重都会显著损害性能；当
`lambda_rec_emb=0.1` 时，再降低 `ring_loss_weight` 的均值影响接近零；当
`ring_loss_weight=1` 时，降低 `lambda_rec_emb` 反而改善均值。正交角点因此支持
“两个目标的相对尺度共同决定训练状态”这一有界解释，而不支持“某一个权重单独造成
全部退化”。

该交互并未挽回统一配置：`unified_0p1_1` 相对同协议 control 的 paired mean
仍下降 AUROC 0.1733、AUPRC 0.1858，未通过预声明实用等效门槛，也低于 GGAD
的 0.7006/0.2565。四个角点中只有 `control_2_20` 同时高于 GGAD；不据此搜索
中间权重或修改论文表格。

## 证据

- 新增 sweep: `HCCS/GGADFormer/rmhd15po`，10/10 finished；
- 冻结 prior sweep: `HCCS/GGADFormer/l6ubfjxt`；
- `authoritative-sweep.json`: W&B config 与 `_step,AUC,AP` history；
- `results.json`: 四角点聚合、逐 seed 配对效应和交互项；
- `replay.json`: 独立本地重放，状态 `passed`；
- `remote-log-sha256.txt`: HCCS-90 agent 日志与状态文件哈希。

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

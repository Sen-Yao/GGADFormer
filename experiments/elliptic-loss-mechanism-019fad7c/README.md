# Elliptic loss mechanism probe

本目录记录 Elliptic 数据集 `lambda_rec_emb` 与 `ring_loss_weight` 四个固定配置的轻量机制诊断。它不修改主文或 Supplement，不寻找更高性能配置，也不把 seed 0 诊断提升为正式科学结论。

## 冻结问题

1. 审计 Elliptic 使用 `2/20` 而其他数据集默认 `0.1/1` 的来源：实现错误、协议/数据漂移、历史选择、测试反馈，或仍无法判定。
2. 在完全相同初始化、batch 采样和 pseudo-source 轨迹下，描述四个固定 cell 的 loss scale、gradient geometry、HSC shell activation 与 collapse 指标如何分化。

## Scope v1

| 项目 | 状态 | 证据边界 |
|---|---|---|
| Phase 1 实现与协议审计 | required | 只读代码、Git 历史、cache/split、W&B 历史与 authority 对账 |
| Phase 2 四 cell 机制探针 | required | Elliptic；`control_2_20`、`emb_only_0p1_20`、`ring_only_2_1`、`unified_0p1_1`；每 cell seed 0 |
| Tolokers 正对照 | excluded | Grill 中未纳入首轮 |
| 连续调参或第 3 阶段 sweep | prohibited | 结果不能触发新配置、LR/radius/schedule 变化或隐蔽复刻 |

固定训练终点为 epoch 150。测试 AUROC/AUPRC 继续实时记录，但不得用于停止、checkpoint、配置选择或扩展实验。正式路径关闭原有 best-AUC/AP state 记录，仅保留 final-epoch 描述。

## 诊断面

- 每个 optimizer update：raw/weighted BCE、token REC、embedding REC、combined REC、HSC 与真实总目标；inner/outer/shell-hit 比例；正常节点和伪异常到 batch center 的距离分位数；embedding 与 reconstruction-displacement norm；logit/score 分布；弱 collapse 指标。
- 第一个 batch 的 epoch `0,1,2,5,10,20,50,100,150`：四个 primitive loss 的 raw/weighted gradient norm、pairwise cosine、连接参数比例与 weighted-total norm。
- 每个 run：初始 model-state SHA-256、完整 batch-global-index trace SHA-256、pseudo-source-global-index trace SHA-256、JSONL SHA-256。

JSONL 保存 302 个逐 update 记录。W&B 每个 epoch 以 `diagnostic/batch_0/*` 与 `diagnostic/batch_1/*` 两组标量提交一次，使两次 update 都可查看，同时保持 AUC/AP 的 W&B `_step` 等于训练 epoch，固定终点仍为 `_step=150`。

## Fail-closed 边界

以下情况使执行无效并停止解释：代码/数据/split/目标/score 方向/metric identity 不一致，非法或重复 cell，非 seed 0，非有限诊断值，epoch/update/gradient 覆盖不完整，W&B 与本地 JSONL hash 不一致，四个 run 的初始化或采样 trace hash 不一致，异常被吞掉或 agent 非零退出。

没有 AUROC/AUPRC 数值门槛，也没有其他结果驱动 gate。单 seed 只提供机制诊断证据。

## 数据流授权

用户已明确授权在 HCCS 执行，并向 `HCCS/GGADFormer` 发送 config、seed、运行状态、metadata、AUROC/AUPRC、scalar diagnostics 与 trace hashes。禁止发送原始数据、源码、凭据、checkpoint 或 custom artifact；launcher 设置 `WANDB_DISABLE_CODE=true` 与 `WANDB_CONSOLE=off`，collector/replay 只写本地文件。

## 执行单元

一个 native W&B sweep，四个独立 run，每个 run 对应一个 cell 与 seed 0。HCCS-85 计划使用 GPU `0,1,2,3`，每个 tmux pane 只消费一个 assignment。正式 launch 前仍需重新核对 occupancy、GPU compute PID、clean detached code worktree、runtime、dataset hash 和 W&B project access。

权威入口：

- `manifest.yaml`: Git、host、runtime、data、W&B、trial 与状态绑定。
- `run-sweep-trial.py`: 四个合法 cell 到固定 argv 的唯一映射。
- `collect-evidence.py`: W&B/history/JSONL/trace 全覆盖校验。
- `replay-results.py`: 从已收集 evidence 与原始 JSONL 独立重算固定终点和覆盖身份。
- `PHASE1_AUDIT.md`: 第一阶段审计结论与未解决项。

## 完成状态

W&B sweep `mufbddb1` 已完成四个 seed-0 run；collector、trace identity 校验与独立 replay 均通过。固定 epoch-150 AUROC/AUPRC 为：

| cell | AUROC | AUPRC |
|---|---:|---:|
| `control_2_20` | 0.8030 | 0.3796 |
| `emb_only_0p1_20` | 0.6089 | 0.1198 |
| `ring_only_2_1` | 0.4330 | 0.0776 |
| `unified_0p1_1` | 0.5688 | 0.1017 |

最终机制裁决见 `DIAGNOSTIC_REPORT.md`。`mechanism-analysis.py` 从冻结 evidence 确定性生成 `mechanism-summary.json`；它不访问 W&B、不选择 best epoch，也不包含性能门槛。当前科学 verdict 为 `partially-tested`：Phase 1/2 的冻结 required scope 已完成，但单 seed 不能证明因果机制，且跨数据集依赖仍未解决。

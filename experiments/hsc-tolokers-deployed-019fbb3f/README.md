# Tolokers HSC 中心污染正式重跑

本目录是 investigation `2026-08-01-vecgad-tolokers-hsc-deployed` 的协议、执行清单与有界证据归档。它只重跑 Tolokers，纠正旧 sweep `25agh73h` 使用过时 Tolokers 配置的问题；旧 sweep 和旧 SHA `89e96d5ac8b088c9e4bcc85c32ff3e568f735438` 只保留为 protocol-drift 反例，不参与本次结果聚合，也不被覆盖。

## 治理状态

- Governance: `governed`
- Lifecycle: `closed`
- Coverage: `canonical`
- Scientific verdict: `resolved_protocol_specific_asymmetric_sensitivity`
- 当前 task: `019fbc4f-1fe7-7860-aee9-130f66dc70cd`
- 父 task: `019fbb3f-2b0c-7d53-b852-457514993d8c`

Scope v1 在正式运行前冻结：

| ID | 范围项 | 类别 | 完成证据 | 状态 |
|---|---|---|---|---|
| `M1` | 六个 center condition 与五个训练 seed 的完整笛卡尔积 | `required` | `canonical-coverage`：30/30 unique valid | `completed` |
| `M2` | 固定终点 AUROC/AUPRC、mean、sample std (`ddof=1`) 与 paired-seed difference | `required` | collector 与独立 W&B replay 一致 | `completed` |
| `M3` | 初始化、训练 batch、伪异常 source、诊断 batch/source 的 pairing audit | `required` | 同 seed 六条件哈希全部一致 | `completed` |
| `M4` | 证据归档、Supplement 更新与 HCCS-85 释放 | `required` | manifest completed、证据 commit、无本任务 PID | `completed` |

## 数据流授权

用户已在本 task 中明确授权在 HCCS-85 执行，并向 `HCCS/GGADFormer` 发送实验 config、seed、run/sweep 状态、必要的 code/host/GPU/pairing metadata、AUROC 和 AUPRC。禁止上传源码、原始数据、凭据、checkpoint、节点级结果、embedding 或未声明 artifact。`WANDB_DISABLE_CODE=true`，checkpoint 与逐 run diagnostic 只保存在 task-owned HCCS 路径并用于本地验证。

## 受控变量

实验只改变训练时 HSC 使用的 batch center。`default` 使用当前重采样 mini-batch 的 embedding 均值。其余条件使用

```text
c_B(q) = (1 - q) * mean(h_i | y_i = 0) + q * mean(h_i | y_i = 1)
```

其中 `q0,q10,q20,q30,q40` 分别对应 `q=0,0.1,0.2,0.3,0.4`。oracle 标签只进入上述 center 构造，不进入伪异常生成、BCE、reconstruction、classifier 或测试 anomaly scorer；oracle batch 缺正常或异常节点时 fail closed。center 不 detach，因此干预覆盖端到端训练中的 HSC 梯度路径。

同一训练 seed 的六个条件固定 `data_split_seed=42`，并要求以下哈希完全相同：

- `initial_model_sha256`
- `training_batch_trace_sha256`
- `pseudo_source_trace_sha256`
- `HSC.diagnostic_batch_trace_sha256`
- `HSC.diagnostic_source_trace_sha256`

## Tolokers 权威配置

```text
batch_size=1024
dataset=tolokers
end_lr=0.00001
lambda_rec_emb=0.1
num_epoch=100
outlier_beta=0.3
peak_lr=0.0001
pp_k=10
progregate_alpha=0.9
rec_loss_weight=1
ring_R_max=1
ring_R_min=0.3
ring_loss_weight=1
train_rate=0.05
warmup_updates=5
sample_rate=0.15
data_split_seed=42
```

实验轴为 condition `default,q0,q10,q20,q30,q40` 与训练 seed `0,1,2,3,4`，共 30 个独立 native W&B trials。主指标只取固定 step 100 的 `AUC.last` 和 `AP.last`；不使用 best epoch、best seed 或事后筛选。每组报告算术平均与 sample standard deviation (`ddof=1`)。

## 正式执行契约

- 唯一执行主机为 `HCCS-85`，GPU 只使用 launch 前无 compute PID 的卡。
- 正式运行从 pre-launch committed full SHA 创建 clean detached remote worktree。
- 使用 native W&B grid sweep；每个 GPU 对应一个 task-specific tmux pane，`remain-on-exit=on`。
- 30/30 finished 不是充分条件；还必须通过 config/code/final-step/local-evidence/pairing/agent-exit 校验。
- `collect-evidence.py` 从 W&B history 与本地 hash-bound diagnostics 收集证据；`replay-results.py` 再次直接查询 W&B，独立重算逐 run fixed-endpoint 指标、六组 mean/std 与 paired differences。
- 只有 collector 和 replay 均通过后，才允许改共享 `supplement.tex`。

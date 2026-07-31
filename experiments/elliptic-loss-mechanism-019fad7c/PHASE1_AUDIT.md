# Phase 1 audit

## 当前裁决

### 实现错误

未发现代码会因 `dataset=elliptic` 自动选择 `lambda_rec_emb=2`、`ring_loss_weight=20`。parser 默认仍是 `0.1/1`；`run.py` 的 variant 逻辑只标记已经传入的权重，不参与选择。

真实 VecGAD 训练目标是：

\[
L = w_{BCE}L_{BCE} + w_{REC}(\lambda_{tok}L_{tok}+\lambda_{emb}L_{emb}) + w_{HSC}L_{HSC}.
\]

冻结 code SHA 中 `run.py` 的 training loss 使用上述加权目标；旧 `batched_total_loss` 则直接相加 raw BCE、combined REC 和 HSC，因此旧训练日志不能回答哪个损失实际主导优化。REC 在 `VecGAD.py` 内部已经包含 `lambda_rec_emb`，诊断实现必须避免再次误乘。HSC 是伪异常到 sampled-batch center 的 `[0.3,1]` shell hinge；embedding REC 的 re-encoded target 在原实现中 detach，诊断忠实保留该单边梯度语义。

裁决：尚无证据把 `2/20` 例外归因于自动分支或 loss 公式实现 bug；旧日志口径确有审计缺陷，但不改变真实 optimizer objective。

### 数据与协议漂移

本轮只读核验的 Elliptic cache SHA-256 为 `2f502df4b87be8f8b5ed5ef8378876125c92b06afbc5b38ee58fe4b56b1b2023`，节点数 46,564。`data_split_seed=42`、`train_rate=0.05`、`val_rate=0.1`；train/validation/test 数量分别为 2,328/4,656/39,580，已知正常训练节点为 2,096。Weighted sampler 给 known-normal 与 unknown pool 相同总质量，`num_samples=46,564`、batch size 32,768，每 epoch 两个 update，epoch `0..150` 共 302 个 update。pseudo-source 来自每个 sampled batch 中的 known-normal 节点。

测试标签只用于每 10 epoch 的 AUROC/AUPRC 描述，不进入 optimizer。原代码曾维护 best AUC/AP state，但没有恢复或保存为最终 checkpoint；正式诊断路径已完全关闭这段 best state 记录，固定使用 epoch 150。

裁决：当前新控制证据之间未发现 cache/split/score/metric/final-step 身份漂移；历史 `0.7876/0.3027` 注释尚未与当前可查询 W&B authority 对上，不能作为现协议结果。

### 历史选择与测试反馈

Git `-S` 追溯显示 `2/20` 首次出现在 commit `6888ce43c5dd467957678d7b9c91dfe57874f159`（2026-04-07）。同一 diff 同时写入 Elliptic 命令和 `AUC=0.7876, AP=0.3027` 注释，但 commit message 仅为 `chore: add some visualization`，没有选择依据、validation 记录或防 test-feedback 协议。

这支持“历史上与结果一起固化、很可能是 outcome-guided 的数据集特定选择”；它不能证明作者明确使用了 test labels 做 hyperparameter search。没有发现能够区分 validation feedback、test feedback、一次性经验选择或更早未保存实验的直接 provenance。

裁决：`2/20` 应分类为来源未充分记录的历史数据集特定选择；“明确 test-feedback”仍未证实。

## 三套 performance authority

| authority | AUROC/AUPRC | 对账结果 |
|---|---:|---|
| submitted main table | `0.7627/0.2813` | 与当前可查询的 `39e3dk75`、`v7cug4b2` `ablation_mode=none` 五 seed final history aggregate 一致：`0.7627034/0.2812562` |
| `reproduction.sh` 注释 | `0.7876/0.3027` | 当前可查询 `39e3dk75` 不复现；分类为 stale/unreconciled |
| 新控制实验 `l6ubfjxt` | `0.7681/0.2949` | code SHA `655d629...`、五 seed、固定 step 150 的独立控制 evidence |

这三者不可混称同一 run。submitted table 的 authority 是五 seed aggregate；新控制是较晚 code/protocol 身份下的可审计比较；历史注释没有可复核的 run lineage。

## Phase 2 能回答与不能回答的内容

四个 seed-0 run 只用于区分以下候选机制：loss contribution scale、gradient conflict/alignment、HSC 激活异常、collapse/geometry 轨迹与不同 optimization basin 的描述性证据。即使某项诊断与 AUROC/AUPRC 同步，也不能单独证明因果，更不能外推连续 lambda 响应面或真实跨数据集依赖。

“真实数据集依赖”在本轮最多保持为与既有 Tolokers 结果相容的解释；由于 Tolokers 正对照被排除，本轮不能新建立跨数据集机制结论。

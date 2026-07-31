# Elliptic loss 权重机制诊断报告

## 结论摘要

Elliptic 的 `lambda_rec_emb=2`、`ring_loss_weight=20` 不是由当前代码针对数据集自动选择的。parser 默认仍是 `0.1/1`，真实 optimizer objective 正确使用外层 loss 权重，而 embedding REC 又在 `VecGAD.py` 内部乘以 `lambda_rec_emb`。旧 `batched_total_loss` 日志确实把三项直接相加，不能反映真实优化尺度，但这属于日志审计缺陷，不是训练目标实现错误。

`2/20` 最早在 commit `6888ce43c5dd467957678d7b9c91dfe57874f159` 中与 `AUC=0.7876, AP=0.3027` 注释同时写入。仓库没有保存选择依据、validation 记录或防 test-feedback 协议。因此最稳妥的分类是：**来源未充分记录的历史数据集特定选择**。现有证据既不能证明它来自明确的 test-label 调参，也不能证明它完全独立于结果反馈。

本轮四个 seed-0 固定配置显示，巨大性能差异与一个强交互相伴：高 HSC 权重把伪异常压入 `[0.3,1]` shell；低 HSC 权重则让几乎所有伪异常仍在半径 1 之外，并使 known-normal 与伪异常 BCE 任务变得很容易，但这种容易分离没有迁移到真实 Elliptic 异常。高 embedding REC 只在高 HSC 下有利。该结果支持“伪异常任务难度/几何与真实异常不匹配”的解释，并显示晚期梯度冲突和不同优化 basin；它不能以单 seed 证明完整因果机制，也不能外推连续响应面或跨数据集依赖。

## 证据身份与协议

- 训练代码：`65014dd11bed01b761aa7c3889c7718b7950884d`，clean detached worktree。
- W&B：`HCCS/GGADFormer/mufbddb1`，四个 run 均 `FINISHED`。
- 数据：`elliptic.mat` SHA-256 `2f502df4b87be8f8b5ed5ef8378876125c92b06afbc5b38ee58fe4b56b1b2023`。
- 固定 split：`data_split_seed=42`，train/validation/test 为 `2328/4656/39580`，其中 known-normal train 为 `2096`。
- 固定执行：每个 cell 仅 seed 0，epoch `0..150`，每 epoch 两个 batch，共 302 次 optimizer update；最终指标只取 epoch 150。
- 四个 run 的初始模型、batch trace、pseudo-source trace SHA-256 完全一致。初始模型 hash 为 `2f6c0f54...`，batch trace 为 `1ee65df6...`，pseudo-source trace 为 `4e8d5772...`。
- 测试 AUROC/AUPRC 虽实时记录，但未用于停止、checkpoint、配置选择或扩展实验。

## Phase 1：协议与实现审计

### 真实 objective 与日志口径

训练实际优化：

\[
L=w_{BCE}L_{BCE}+w_{REC}(\lambda_{tok}L_{tok}+\lambda_{emb}L_{emb})+w_{HSC}L_{HSC}.
\]

`run.py:333` 使用该加权目标；`VecGAD.py:372-379` 证明 REC 已在模型内部包含 `lambda_rec_emb`。`run.py:405` 的旧 `batched_total_loss` 只是 raw BCE、combined REC、raw HSC 之和，因此不能用于判断真实梯度主导关系。诊断路径分别记录四个 primitive loss 及其真实 weighted contribution，并继续对原始 `loss` 执行 backward，没有重构或替换训练目标。

HSC 的实现位于 `VecGAD.py:334-341`：伪异常到 sampled-batch center 的距离低于 `0.3` 或高于 `1` 才产生 hinge。伪异常来源是 sampled batch 中的 known-normal 节点；`run.py:317-346` 同时记录其全局节点 trace。embedding REC 的 re-encoded target 在 `VecGAD.py:344` detach，因此诊断中的 embedding REC 梯度忠实反映原实现的单边梯度。

### 数据、split 与评价协议

`utils.py:99-131` 用 `data_split_seed` 固定 split，并只从 train split 中抽取 known-normal。`run.py:238-247` 给 known-normal 与 unknown pool 相同总采样质量，batch size 32,768 在 46,564 个节点上产生每 epoch 两次 update。`run.py:501-539` 只在测试阶段计算 higher-logit-is-more-anomalous 的 AUROC/AUPRC。

本轮新控制证据之间没有发现 cache、split、known-normal、pseudo-source、score direction、metric definition、final-step、batch/update 或 RNG 身份漂移。当前 parser 默认仍为 `lambda_rec_emb=0.1`、`ring_loss_weight=1`（`run.py:681-684`），`infer_run_variant` 只标记调用者已经传入的权重，不会自动为 Elliptic 改成 `2/20`。

### 三套 performance authority

| authority | AUROC/AUPRC | 裁决 |
|---|---:|---|
| submitted main table | `0.7627/0.2813` | 与可查询的 `39e3dk75`、`v7cug4b2` 中 `ablation_mode=none` 五 seed final aggregate `0.7627034/0.2812562` 一致 |
| `reproduction.sh` 注释 | `0.7876/0.3027` | 当前 `39e3dk75` 不复现，缺少可审计 run lineage，分类为 stale/unreconciled |
| 新五 seed 控制 `l6ubfjxt` | `0.7681/0.2949` | 较晚 code SHA `655d629...` 下固定 step=150 的独立控制证据 |

三者不是同一执行实例，不应互相替代。主表不能修改；本报告只澄清证据来源。

## Phase 2：四个固定配置

| cell | `lambda_emb/HSC` | run | epoch-150 AUROC | epoch-150 AUPRC |
|---|---:|---|---:|---:|
| `control_2_20` | `2/20` | `a6w0waem` | 0.8030 | 0.3796 |
| `emb_only_0p1_20` | `0.1/20` | `9gnmdde9` | 0.6089 | 0.1198 |
| `ring_only_2_1` | `2/1` | `q94sewkd` | 0.4330 | 0.0776 |
| `unified_0p1_1` | `0.1/1` | `9n7w76kh` | 0.5688 | 0.1017 |

本轮 seed-0 difference-in-differences 为 AUROC `+0.3299`、AUPRC `+0.2839`。它与既有五 seed factorial 的同方向强交互一致，但仍只说明四个离散 cell 的非加性交互，不能重建连续 lambda 响应面。

### 1. 初始尺度：权重确实重塑优化方向

epoch 0 第一个 batch 的 raw loss、raw gradient 和 pairwise cosine 在四个 cell 中逐值相同，证明差异不是初始化或采样漂移。加权后：

| 分量 | raw gradient norm | 权重较低时 | 权重较高时 |
|---|---:|---:|---:|
| embedding REC | 9.774 | 0.977 (`0.1`) | 19.548 (`2`) |
| HSC | 11.437 | 11.437 (`1`) | 228.737 (`20`) |

初始 embedding REC/HSC cosine 为 `+0.384`。因此“从第一步起两者直接冲突导致差异”不成立。另一方面，HSC=20 的初始 weighted gradient 远大于 BCE (`0.754`) 和两类 REC，说明权重不是对同一轨迹的小幅校准，而是显著改变早期优化方向。这里的“尺度失配”是已证实的数值事实，但不能简单叫作错误，因为高权重 cell 反而是当前最佳 cell。

### 2. HSC 激活：低权重没有把伪异常送入 shell

epoch 150 对两个 batch 按伪异常数量加权后：

| cell | shell-hit | outer violation | 伪异常到 center 平均距离 |
|---|---:|---:|---:|
| `2/20` | 83.5% | 16.5% | 0.939 |
| `0.1/20` | 94.2% | 5.8% | 0.844 |
| `2/1` | 2.9% | 97.1% | 1.186 |
| `0.1/1` | 0.0% | 100.0% | 1.488 |

四个 cell 的 inner violation 都为 0。低 HSC cell 的 loss 与梯度仍然存在，代码路径没有失活；只是权重 1 的力不足以在固定预算内把多数伪异常压到半径 1 内。因此这是**预期的优化结果/激活不足**，不是 HSC 实现分支失败。

### 3. 伪任务越容易，真实异常表现反而越差

epoch 150 的训练几何呈现反向关系：

| cell | raw BCE | pseudo mean - normal mean logit gap | AUROC/AUPRC |
|---|---:|---:|---:|
| `2/20` | 0.362 | 0.261 | 0.803/0.380 |
| `0.1/20` | 0.342 | 0.463 | 0.609/0.120 |
| `2/1` | 0.080 | 4.402 | 0.433/0.078 |
| `0.1/1` | 0.067 | 4.682 | 0.569/0.102 |

低 HSC cell 能几乎完美地区分 known-normal 与生成伪异常，却不能识别真实测试异常。这支持一个较弱但直接的解释：HSC=1 允许伪异常停留在离中心过远、过于容易的区域，分类头学习了一个不迁移到真实 Elliptic 异常的代理任务。HSC=20 迫使伪异常靠近 shell，维持更困难的训练边界；这与较好真实异常排序相伴。

该证据仍不是完整的 pseudo-real distribution audit：本轮没有把真实异常 embedding 分布用于训练机制诊断，也没有用 Wasserstein/KS 等距离量化伪异常与真实异常的分布相似度。因此报告采用“支持错配/难度假设”，不写成已证明的因果机制。

### 4. 高 embedding REC 只在高 HSC 下有利

固定 HSC=20 时，将 `lambda_emb` 从 0.1 提到 2，AUROC/AUPRC 增加 `+0.1941/+0.2598`；固定 HSC=1 时，同样变化反而为 `-0.1358/-0.0240`。这排除了“embedding REC 独立、单调改善 Elliptic”的解释。

一个与观测相容的机制是：高 HSC 先把伪异常维持在较难的 shell 几何中，高 embedding REC 再通过重构约束限制表示过快收缩到简单分隔解。epoch 150 的 `2/20` embedding REC/HSC cosine 已变为 `-0.462`，说明二者晚期存在竞争；但由于该竞争是训练状态产生后的结果，不能据此断言它就是性能提升的原因。

### 5. 梯度冲突与 optimization basin 都是晚期、状态依赖现象

epoch 150 第一个 batch 的代表性冲突包括：

- `2/20`：embedding REC/HSC cosine `-0.462`；
- `2/1`：BCE/HSC cosine `-0.821`；
- `0.1/1`：BCE/HSC cosine `-0.695`。

同一 pair 在初始状态并不冲突，因此不能把一个静态 gradient-conflict 标签当作四 cell 差异的充分解释。更合理的描述是：权重改变早期轨迹，轨迹进入不同表示状态后，再产生不同的晚期冲突结构。

固定每 10 epoch 的测试曲线也显示不同 basin。`2/20` 的 AUROC 从 epoch 120 的 `0.3486` 跳到 130 的 `0.7008`、140 的 `0.7906` 和 150 的 `0.8030`；`0.1/1` 在 90-110 约为 `0.62`，随后下降到 150 的 `0.5688`。这些曲线只用于描述 basin，不构成 best-checkpoint 选择依据。

### 6. Collapse 指标没有给出单一解释

四个 cell 的 score variance、embedding centered RMS 与 reconstruction displacement 都是有限且非零的，没有发现代码级全常数输出或完全表示塌缩。高 HSC cell 的伪异常 centered RMS 更小，但其中既包含最佳的 `2/20`，也包含明显较弱的 `0.1/20`；因此现有弱 collapse 指标不足以解释性能排序。

## 分类裁决

| 候选解释 | 当前裁决 | 证据边界 |
|---|---|---|
| 实现错误 | 未发现会自动选择 `2/20` 或改变真实 objective 的 bug | 旧 `batched_total_loss` 是日志缺陷，不是 optimizer objective |
| 协议/数据漂移 | 新控制证据内未发现 | 历史 `0.7876/0.3027` lineage 仍未对上 |
| 历史选择 | 支持 | `2/20` 与结果注释在同一历史提交固化，选择理由缺失 |
| 明确 test feedback | 未解决 | 没有 validation/test 选择日志，不能从结果注释反推明确过程 |
| loss 尺度差异 | 强支持 | 初始 weighted gradient 相差 20 倍，但不能称为单纯错误 |
| 梯度冲突 | 部分支持、非初始原因 | 冲突晚期出现且依赖 cell/状态，因果方向未识别 |
| HSC 激活异常 | 支持“低权重激活不足”，不支持实现失活 | HSC=1 几乎全为 outer violation，梯度和 loss 仍非零 |
| optimization basin | 支持描述性差异 | 单 seed、固定曲线，不做 checkpoint 选择 |
| 伪异常与真实异常错配 | 支持一个较弱机制假设 | 伪任务更容易而真实指标更差；未做完整分布对齐检验 |
| 真实数据集依赖 | 未解决 | Phase 2 只有 Elliptic；Tolokers 正对照被明确排除 |

## 不可外推项

1. 本轮是 seed-0 机制探针，不是五 seed 正式机制结论；性能数值不替代 `l6ubfjxt` 或 `rmhd15po` 的正式五 seed evidence。
2. 四个离散 cell 只能支持强交互，不能推出最优连续权重、单调性、阈值或响应面。
3. 不允许用曲线选择 best checkpoint，也不允许据结果增加 LR、radius、schedule、Tolokers 或新 lambda 配置。
4. 没有跨数据集机制诊断，因此不能把 `2/20` 写成“Elliptic 本质上需要”的普遍规律。
5. 报告不修改主文、Supplement 或 `reproduction.sh`；历史注释保留为被审计对象。

## 可重放材料

- `authoritative-sweep.json`：W&B config、summary、完整选定 history 与诊断身份。
- `results.json`：四 cell 固定终点、trace identity 与 JSONL hash。
- `diagnostics/*.jsonl`：每 run 302 个 optimizer update，加 start/end 共 304 条。
- `replay-results.py` / `replay.json`：固定终点和覆盖独立 replay，SHA-256 `9c41f203...`。
- `mechanism-analysis.py` / `mechanism-summary.json`：本报告使用的确定性机制提取。
- `collector-live.log`：terminal W&B collector 的保留输出。
- `remote-log-sha256.txt`：四个 agent 的 start/finish/exit/log 内容寻址记录。

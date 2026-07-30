# VecGAD TODOLIST

更新日期：2026-07-29

## 使用说明

- **A 区由 Codex 执行**：记录 grill 已冻结、需要落实到 AAAI 主稿或补充材料的迁移与精简修改。A 区复选框表示实际落稿状态，而不是是否已接受该决定。
- **B 区写给 Ziyao**：记录完成基本模板迁移和正文精简后，为提高实验严谨性或 provenance 完整性需要推进的四项研究路线。启动时间、计算资源和最终 protocol 由 Ziyao 决定；Codex 可按后续指令协助执行、审计并回填论文。
- B1--B3 不阻塞当前的基础稿件迁移，但在 AAAI 最终结果冻结前必须处理或明确接受其残余风险；B4 仅为可选的 provenance 完善项，不阻塞当前任务。

## A. Codex 执行队列：AAAI 稿件迁移与精简

- [x] 将 AAAI 工作标题冻结为 `Leveraging Vectorized Reconstruction Discrepancy for Label-Efficient Graph Anomaly Detection`，不使用 `VecGAD:` 方法名前缀；落稿时同步修改 PDF 和 OpenReview，确保两处标题完全一致。该修改只在 KDD 版标题的基础上补入 `Reconstruction`，不改变任务、方法或贡献定位。
- [x] 将 OpenReview TL;DR 冻结为 `VecGAD leverages vectorized reconstruction discrepancy to guide pseudo-anomaly generation within a hyperspherical shell for semi-supervised graph anomaly detection with scarce normal labels.`。仅将原文 `under label scarcity` 改为 `with scarce normal labels`，以明确对齐 labeled-normal-only 监督设定；TL;DR 保留 VecGAD 名称。
- [ ] 准备并提交独立的匿名 `Code and Data Supplement` 压缩包。不在主稿或 Technical Supplement 中放置 GitHub、OpenReview、W&B 或其他外部仓库链接；压缩包移除 Git 历史、作者姓名、机构、仓库 URL、历史投稿记录、访问凭据和可识别运行日志；仅保留评审复现 VecGAD 必需的代码、依赖说明、数据获取/预处理说明和匿名化实验配置。打包后另做姓名、URL、凭据、绝对路径和元数据扫描。
- [x] 将正文全部 section/subsection 标题改为 Title Case；修复参考文献中可见的句号、逗号前空格，并清理 `02_RW.tex`、主结果表和消融表的行尾空格。最终 Overleaf 主稿编译为 8 页，正文约占 6.6 页、随后为 References；日志为 `Errors 0 / Warnings 0`，仅保留 4 条非阻塞 underfull typesetting 信息。
- [x] 按已确认实现恢复方法解释：明确 hop token 离线预计算及 node-level mini-batch 优化、$\mathcal B$/$\mathcal B_N$/$\mathcal G_{\mathcal B}$ 的监督范围、未标注节点仅参与 batch-wide reconstruction 与 sampled-batch centroid、非对称 dual-space reconstruction 与 stop-gradient，以及伪异常仅用于训练。该恢复不加入效率、可扩展性或未冻结结果 claim。
- [x] 按已确认实现恢复实验协议：固定 data-split seed 42，采用 5\% training-index / 10\% unused holdout-index / 85\% evaluation split，明确 `idx_val` 不用于验证、调参、模型选择或评估；补回 transductive graph/features access、无异常标签训练、等总采样质量且有放回的 weighted sampler，以及所有数据集固定 $\rho=0.15$、$\beta=0.3$。主表同时确认 TAM Reddit AUROC 为 `0.5764`、AnomalyDAE Elliptic AUROC 为 `0.5522`，并移除 DAGAD 实验行。
- [x] 按后续冻结决定，从 AAAI 2027 正文主表和实验比较口径中移除 DAGAD 数值；Related Work 仍可将 DAGAD 作为相关生成方法讨论。
- [x] 实验设置改为八个 baseline，并同步删除 DAGAD 的 protocol caveat 和所有基于其表格结果的正文比较；GGAD 和 RHO 继续明确为原生面向 labeled-normal-only semi-supervised setting，其余 baseline 使用 task-specific adaptations。
- [x] 将 VecGAD 的 Reddit 主表结果冻结并写为五随机种子均值 AUROC `0.5782`、AUPRC `0.0441`。
- [x] 使用已冻结的 Tolokers label-ratio 数据替换当前旧版 Training Ratio Analysis 图和对应数据源。新图仅报告五随机种子 AUROC 均值，不绘制误差带，比较 GGAD、RHO 与 VecGAD 在 $R\in\{1\%,2\%,3\%,5\%,7\%,10\%,15\%\}$ 下的表现：GGAD=`0.5125/0.5133/0.5137/0.5098/0.5374/0.5306/0.5411`，RHO=`0.5137/0.4823/0.4885/0.5098/0.4879/0.5763/0.5849`，VecGAD=`0.5876/0.5992/0.6505/0.6496/0.6637/0.6714/0.6702`。删除当前三数据集、五比例、双指标图及其与新数据冲突的分析，但保持 Ablation Study 的标题和顶会论文常用行文组织。
- [x] 将新版 label-ratio 分析限定为均值层面的标签稀缺鲁棒性证据：说明 VecGAD 在全部七个比例下均取得高于 GGAD 和 RHO 的 mean AUROC，在仅使用 1% 节点训练时仍保持明确优势；性能从 1% 到 3% 快速提高，之后相对稳定。可写该模式与 RDV-guided pseudo-anomaly generation 在标签稀缺下丰富监督的设计目标一致，但不得声称该实验单独隔离或证明了该组件的因果贡献；因不报告方差，不使用 `statistically significant`。删除关于 Tolokers 异常由属性主导等未经独立验证的数据集机制推断。
- [x] 统一修正引言、图注和摘要对 label scarcity 的证据叙事。删除 GGAD/RHO 在比例低于 5% 后 `significant/severe/monotonic degradation` 的表述，因为已冻结均值在 5% 到 1% 间基本持平；图注仅说明 VecGAD 在全部比例（含 1%）均取得更高 mean AUROC。引言改为现有方法在低标签区间表现有限或不稳定，提示其可能未充分提取稀缺正常标签中的监督价值；摘要将 `fail to achieve competitive performance with gradually decreasing size of labeled samples` 最小改为 `fail to maintain competitive performance under low-label regimes`，并将后续判断改为 `current approaches may not fully capture the underlying patterns of normality under label scarcity`。摘要的 RDV 句删除 `precisely`，仅保留 `captures how each node deviates from normality`。保留 label scarcity 作为 research gap，但将 reconstruction overfitting、generation unreliability 等未直接隔离验证的因果解释改为带 `may` 的设计分析。
- [x] 将摘要最后一句改为：`Extensive experiments on seven datasets demonstrate that VecGAD achieves strong overall performance compared with existing methods, while label-ratio analysis further shows its clear advantage under scarce normal supervision.` 删除 `significantly outperforms state-of-the-art methods under various label-scarce settings`，避免在不报告方差或显著性检验、Elliptic 存在例外且多比例比较仅覆盖 Tolokers 时形成全面统计显著或全数据集多比例领先的暗示。
- [x] 为满足 AAAI 2027 正文页数限制，将 $K$ 与 $\alpha$ 的敏感性图及详细分析移入补充材料；正文仅保留搜索范围和一句总体结论，并优先保留主结果、Ablation Study 与 label-ratio 实验。
- [x] 按 2026-07-29 的最新用户授权，在不将 B3 作为本次恢复前置条件的情况下，将 KDD 版本的五数据集效率表和 `Efficiency Analysis` 恢复到 AAAI 正文。完整保留 dataset statistics、GPU memory、time per epoch、L40 48GB、full-batch 和 OOM 口径；将原三段分析适度压缩为两段，最大限度保留 KDD 原有结构、边界、故事性和 novelty，仅微调明显过强的绝对 claim。删除 `Performance Comparison` 中重复的 OOM/scalability 段落，不加入 DGraph 效率 claim。恢复前的 AAAI 稿件和本决策账本备份于 `tmp/aaai-migration/backups/2026-07-29-pre-efficiency-restoration/`；更早删除的历史材料仍保留于 `tmp/aaai-migration/backups/2026-07-28-efficiency-pre-b3/`。
- [x] 压缩正文的 Anomaly Scoring and Joint Optimization：保留异常分数定义、正常/伪异常标签规则和总目标函数；标准 BCE 的多行展开公式、重复的逐 loss 解释及完整优化细节移入补充材料。
- [x] 压缩 HSC 小节中的重复叙述：保留一次“扰动过小/过大”的动机、球壳定义、完整 HSC 公式和一次 pushing/pulling 解释，删除第三次重复总结，不改变公式、机制或 claim 主线。
- [x] 保留 RDV 作为有效方向信号的核心 claim，但降低 Dual-Space Reconstruction Constraint 的绝对化措辞：将 `guarantees`、`strictly reflects true deviation` 和 `precise direction` 改为鼓励双空间一致性、减少编码失配并提供更稳定的方向信号；有效性由右半图和严格受控消融共同支撑。
- [x] 修正 `W/O Directional Guide` 的定义与结论边界：该变体保留原扰动幅度 $\|\mathbf p_i\|_2$，仅将学习到的方向替换为随机单位方向，因此不得描述为将向量偏差缩减为标量。基于当前三个数据集均出现性能下降，只声称 learned perturbation direction 提供了超越 perturbation magnitude alone 的有用 node-dependent directional information；删除 `specific anomaly semantics`、`semantically incoherent pseudo-anomalies` 以及该消融验证真实异常语义的表述。最终措辞仍需在 B1 严格受控消融复现后按最终数值确认。
- [x] 将 HSC 与 REC 的消融结论改为不对称表述。按当前表，移除 HSC 在 Amazon、Elliptic、Tolokers 上均造成较大下降，可暂写其在当前结果中表现出更一致的影响；移除 REC 主要在 Elliptic 上造成明显下降，在 Amazon 和 Tolokers 上影响较小，只说明其 empirical contribution 在当前 protocol 下更具 dataset dependence。删除二者均在大多数数据集 `significantly` 下降、均为 `indispensable` 的概括；所有精确结论在 B1 严格复现后按最终数值确认。
- [x] 调整主结果表、消融和 RDV 可视化的证据分工。主表中的跨方法比较只用于说明 VecGAD 整体框架的表现，其相对 reconstruction-based baselines 的提升可写为与利用 vectorized reconstruction discrepancies 的动机一致，但不得直接 `verify` RDV 的因果作用；`W/O Directional Guide` 受控消融用于更直接支撑 learned direction 超越 magnitude alone 的贡献，右半图仅作为方向信号的直观辅助证据。
- [x] 将 HSC 中心从“全图质心”校正为当前加权采样 batch 的质心 $\mathbf{c}_{\mathcal{B}}=|\mathcal{B}|^{-1}\sum_{v_i\in\mathcal{B}}\mathbf{h}_i$，并将 HSC 的求和范围对应到该 batch 生成的伪异常；模型和既有实验结果不变。补充材料说明 `WeightedRandomSampler` 的 batch 构成，正文不再使用 `global centroid` 表述。
- [x] 将伪异常生成公式校正为 $\tilde{\mathbf{h}}_i=\mathbf{h}_i+\beta\mathbf{p}_i$，其中 $\beta$ 是全局扰动缩放系数；所有数据集固定使用 $\beta=0.3$，在补充材料中作为固定实现参数报告而不列为搜索参数。正文说明 $\mathbf{p}_i$ 提供数据依赖的扰动向量、$\beta$ 设置基础尺度、HSC 进一步约束伪异常相对于 batch centroid 的最终径向位置。
- [x] 明确伪异常仅由当前 batch 内已标注正常节点的随机子集生成：令 $\mathcal{B}_N=\mathcal{B}\cap\mathcal{V}_L^N$，从中抽取 $\mathcal{G}_{\mathcal{B}}\subset\mathcal{B}_N$，满足 $|\mathcal{G}_{\mathcal{B}}|=\lfloor\rho|\mathcal{B}_N|\rfloor$，并仅对 $i\in\mathcal{G}_{\mathcal{B}}$ 应用伪异常生成和 HSC 求和；BCE 的正常项仍使用 $\mathcal{B}_N$。所有数据集固定 $\rho=0.15$，补充材料报告该值和完整 `WeightedRandomSampler` 协议。
- [x] 保留现有非对称 Dual-Space Reconstruction Constraint，不删除重编码分支的 stop-gradient，也不重跑实验。正文将 $\mathcal{L}_{\mathrm{REC}}$ 校正为与实现一致的两项形式：token MSE 覆盖整个 sampled batch；embedding consistency 使用未平方的 $L_2$ 距离，仅覆盖 $\mathcal{G}_{\mathcal{B}}$，并写明 $\operatorname{sg}(\mathcal{E}(\hat{\mathbf{T}}_i))$；同时保留 $\lambda_{\mathrm{tok}}$ 与 $\lambda_{\mathrm{emb}}$。相邻文字仅做必要调整，不声称该设计优于对称训练、能够防止 collapse 或保证优化稳定性。
- [x] 将 RDV 定义的符号方向校正为与实现一致的 $\mathbf{R}_i=\hat{\mathbf{T}}_i-\mathbf{T}_i$，不修改代码、不重跑实验。将原始 $\mathbf{R}_i$ 称为保留结构化符号信息的 reconstruction discrepancy signal，由 $\mathbf p_i=\mathcal P(\mathbf R_i)$ 将其转换为 embedding-space perturbation vector；不再声称原始 RDV 本身就是精确异常方向。
- [x] 保留 Hyperspherical Shell Constraint 的名称、公式结构和“鼓励 informative hard negatives”的核心作用，但不再将 $R_{\min}$ 解释为真实正常流形或决策边界。正文在公式处定义 batch centroid $\mathbf c_{\mathcal B}$，后文自然简称 reference center；$R_{\min}$ 只表示鼓励最小分离，$R_{\max}$ 限制过度偏移。固定 $R_{\min}=0.3,R_{\max}=1$ 仅在补充材料报告。
- [x] 将传播算子改写为代码使用的 $\hat{\mathbf A}=\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}+\mathbf I$，其中 $\mathbf D$ 是原始 $\mathbf A$ 的度矩阵；同时将 Graph Tokenization 校正为递归残差传播 $\mathbf X^{(0)}=\mathbf X$、$\mathbf X^{(k)}=(1-\alpha)\hat{\mathbf A}\mathbf X^{(k-1)}+\alpha\mathbf X^{(0)}$，并令 $\mathbf t_i^{(k)}=\mathbf X_{i,:}^{(k)}$。该修改已落实到本地 AAAI LaTeX 和 Overleaf，在线编译通过；恢复已确认的方法解释与实验协议后，当前主稿为 8 页，正文约占 6.6 页、随后为参考文献；不改变现有非 DGraph 数据路径的代码或既有结果。
- [x] 保留 `W/O $\alpha$-residual` 的消融名称，但将其严格解释为设置 $\alpha=0$、移除每一步对 $\mathbf X^{(0)}$ 的显式原始特征重注入，而不是完全消除 identity information。参数分析将 `pure propagation without identity` 改为 `propagation without explicit original-feature reinjection`，并说明 $\hat{\mathbf A}$ 中的 $\mathbf I$ 仍保留 self-information path；结论只声称显式重注入有助于将多跳 token 锚定到原始节点属性，不声称该消融移除了全部节点自身信息。代码、实验设置和结果不变。
- [x] 重写 $\alpha$ 参数分析的机制解释，并将其限定为有图滤波依据的弱 claim。由 $\mathbf X^{(k)}=((1-\alpha)\hat{\mathbf A})^k\mathbf X^{(0)}+\alpha\sum_{j=0}^{k-1}((1-\alpha)\hat{\mathbf A})^j\mathbf X^{(0)}$ 说明：较小的 $\alpha$ 相对强化反复传播项，较大的 $\alpha$ 抑制高阶传播并加强原始属性锚定；不同最优值可能反映 anomaly-relevant attributes 与 graph neighborhoods 之间不同的 feature-topology alignment。补充材料可据此解释 Amazon 偏好中等值、Elliptic 偏好较强 residual reinjection、Tolokers 在较宽范围内相对稳定；正文只保留一句总体结论。不得继续使用 `Amazon anomalies are local`、`Elliptic relies on long-range structure` 或其他仅凭敏感性曲线无法支持的数据集异常机制断言。
- [x] 重写 $K$ 参数分析的机制解释，并限定为“有效结构尺度”的弱 claim。说明 $K$ 决定 token 序列包含的最高传播阶数和模型可访问的最大结构尺度；增大 $K$ 可能提供额外邻域上下文，也可能因反复传播使高阶 token 更同质化或混入无关的远距离信息。由于全部 token 同时参与 reconstruction objective，即使 Transformer 能降低某些 hop 的注意力，高阶噪声也不能被假定为完全忽略。不同最优 $K$ 只提示不同数据集具有不同的 useful structural-context scale；删除 `Amazon fraud patterns are highly localized`、`Elliptic relies on extended structural patterns`、`Tolokers is attribute-dominant` 等敏感性曲线无法单独支持的异常机制断言。
- [x] 参数分析迁移时，Amazon 的权威结果以 `figs/propagation_steps/propagation_steps.py` 和 `figs/residual_weight/residual_weight.py` 中的数据为准；`docs/VecFormer/experiments.md` 仅作为历史文档，不因其数值冲突修改当前 Amazon 曲线，也不据此触发重跑。
- [x] 修正 Ablation Study 对 `W/O $\alpha$-residual` 的表外解释：删除 T-Finance 及其高密度导致表示快速同质化的叙述，因为当前消融表只报告 Amazon、Elliptic 和 Tolokers。暂改为：`Removing the explicit residual reinjection causes performance degradation across all three datasets, with the largest drop observed on Amazon, supporting its role in anchoring multi-hop representations to the original node attributes.` B1 严格受控重跑后，再依据最终数值确认 `all three datasets` 与 `largest drop observed on Amazon` 是否仍成立。
- [x] 在正文用一句话明确 Transformer encoder 的实际 Readout：对最后一层注意力权重在多头维度取平均，并使用以 0-hop token 为 query 的注意力分布对全部 hop token 表征进行加权聚合；不在论文中讨论未参与前向计算的 `self.read_out` 层。
- [x] 将 Autoencoding Structure 的 backbone 表述最小校正为“a Transformer encoder and a lightweight MLP decoder”；摘要中的“Transformer-based autoencoder framework”继续保留。
- [x] 恢复 KDD PDF 中 VecGAD 的正式展开 `Vectorized Discrepancy-Guided Graph Anomaly Detection (VecGAD)`；将当前引言误写的 `Vectorized Discrepancy-Guided Transformer` 改回该名称，避免缩写不成立。同时将方法章节标题中的旧名 `THE PROPOSED VECFORMER` 改为 VecGAD 对应标题。该项仅修正迁移漂移，不改变方法定位。
- [x] 将结论中的 `Experiments on eight datasets` 修正为 `Experiments on seven datasets`，与正文实验设置和主结果表一致。
- [x] 删除结论中缺乏受控证据的 `reduced effectiveness on extremely sparse graphs` 及相应因果解释。恢复为设计层的弱 limitation：当前 tokenization 在同一数据集内对所有节点统一使用 dataset-level propagation parameters，可能无法充分适配节点之间不同的 feature-topology alignment；将 node-adaptive propagation weights 作为未来方向。该表述不否定按数据集调优 $\alpha$ 与 $K$，也不声称固定参数已被实验证明会造成性能下降。

## B. Ziyao 后续研究路线：最终严谨性工作

### B1. VecGAD 结果与消融的严格复现

- [ ] 在 AAAI 2027 最终实验结果冻结前，完成 VecGAD 内部结果的严格复现与 provenance 审计。
  - 对 Amazon、Elliptic 和 Tolokers 完成严格受控的五随机种子消融复现。
  - 对每个数据集，以权威完整模型配置为唯一基础配置，运行完整模型及 `W/O HSC`、`W/O REC`、`W/O alpha-residual`、`W/O Directional Guide` 四个变体。
  - T-Finance 主表 authority 已裁决为 W&B sweep `HCCS/GGADFormer/n30dxpp2`：科学代码 SHA `e071ae6646451d94fc8e8c9e88305eb76c393089`，seeds `0-4`，`AUC.last`/`AP.last` 均值 `0.897484608080049/0.6460347053997909`，sample std (ddof=1) `0.006994945403890782/0.019944971027181832`，`ring_R_min=0.3`。该 sweep 相对原 `iqxjqsdl` 只改变 `ring_R_min: 0.5 -> 0.3`；旧 sweep 仅保留为受控基准 lineage，不再作为论文主表 authority。主表、实验分析和 `reproduction.sh` 已按四位小数 `0.8975/0.6460` 统一。
  - 每个变体只改变其定义对应的单一因素；数据划分、seeds `0-4`、epoch 数、学习率、batch size、`alpha`、`K`、壳层半径和其余 loss 权重必须与该数据集的完整模型一致。
  - 统一报告固定训练轮数结束时的 `AUC.last` 和 `AP.last` 五种子均值，不使用测试集最优 epoch 指标。
  - 在 W&B 中保留完整模型和全部变体的永久 run/sweep 链接、完整配置、逐 seed 原始结果及聚合值，并核验 Tolokers 完整模型结果的原始 provenance。
  - 复现完成前，当前消融表不视为最终受控证据，不据此继续增强模块因果性 claim。
  - 复现完成后，统一更新主结果表、消融表、实验分析及 LaTeX provenance 注释，确保完整模型数值在不同表格中一致。
  - 将 DGraph 的预处理统一为论文及其余数据集使用的 $\hat{\mathbf A}=\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}+\mathbf I$，其中 $\mathbf D$ 由加入单位阵前的原始邻接矩阵计算。
  - 使用统一算子和 DGraph 的权威完整模型配置运行 seeds `0-4`，保存 source commit、命令、逐 seed 结果和永久 W&B provenance，并据此更新 DGraph 的 VecGAD AUROC/AUPRC。完成前，当前 DGraph 精度仅视为旧算子下的历史结果。
  - 后续 DGraph 效率与可行性重跑也必须使用该统一算子，避免精度表和效率表对应不同实现。

### B2. DAGAD 严格复现

- [ ] 在 AAAI 2027 最终实验结果冻结前，完成 DAGAD 的严格复现与 provenance 审计。
  - 使用当前论文的 `5%/95%` 数据划分协议，至少运行 seeds `0-4`。
  - 固定并记录 DAGAD 官方代码来源、source commit、环境、命令、逐数据集配置和数据划分 artifact。
  - 明确记录训练、调参和最终评估各阶段可访问的标签；不得仅凭方法名推断 protocol 等价。
  - 在 W&B 中显式记录 `method=DAGAD`，保存每个 run 和 sweep 的永久链接、AUROC/AUPRC 原始值及五种子均值。
  - 复核当前暂用的 Amazon、Reddit、Photo、Elliptic、T-Finance、Tolokers 数值，并实测确认 DGraph 是否确为 OOM。
  - 若复现值与暂用值不一致，更新正文主表、实验分析和所有引用该结果的材料。

当前暂用结果：

| Metric | Amazon | Reddit | Photo | Elliptic | T-Finance | Tolokers | DGraph |
|---|---:|---:|---:|---:|---:|---:|---:|
| AUROC | 0.7950 | 0.5639 | 0.6368 | 0.8528 | 0.5194 | 0.4136 | OOM |
| AUPRC | 0.2378 | 0.0438 | 0.1268 | 0.4514 | 0.0448 | 0.1795 | OOM |

状态：上述 DAGAD 数值不用于当前 AAAI 稿件，仅保留为历史候选结果；只有完成严格复现和 provenance 审计后，才讨论是否在后续版本恢复。

### B3. 统一效率实验与硬件口径

- [ ] 作为后续版本的严谨性工作，完成历史计时 provenance 审计，并在统一 GPU 环境下重跑 VecGAD、GGAD 和 RHO 的可复现权威效率基准；具体 GPU 型号在执行本 TODO 时根据实际可用资源确定。按 2026-07-29 的最新用户授权，本项不再阻塞当前 AAAI 稿件恢复 KDD 历史效率表；完成后以统一重测结果替换或复核当前恢复的历史结果。
  - 最终横向训练时间比较仅覆盖 Tolokers、Amazon、T-Finance 和 Elliptic，Questions 不再纳入。DGraph 不进入训练速度排名，而是作为独立的 full-batch 可行性实验：验证 VecGAD 能够完成训练，并验证 GGAD/RHO 在同一 GPU 显存预算下不可行。
  - 尽可能恢复旧 `22--84 ms/epoch` 与新 T-Finance、Elliptic、DGraph 计时数据对应的机器、环境、source commit、配置、命令、计时代码和 W&B run/sweep；恢复前仅将二者视为历史观测，不直接作为最终论文证据。
  - 若历史 provenance 无法完整恢复，或两组历史结果的实验口径不一致，以统一 GPU 重跑结果作为 AAAI 稿件唯一权威的拓展性结果。
  - 执行时为 VecGAD、GGAD 和 RHO 使用同一台机器、同一型号 GPU 和统一计时协议，并完整记录 GPU 型号、显存、软件环境及运行条件。
  - 每个方法在每个数据集上使用生成正文主结果表权威精度结果的最终配置，不为效率测试另行统一隐藏维度、层数、传播步数或其他模型容量相关超参数。
  - 补充材料记录各方法的关键效率相关配置和参数量，并将结论限定为“论文实际报告模型的效率比较”，不表述为完全相同模型容量下的纯架构对照。
  - 三种方法全部强制采用 full-batch。GGAD/RHO 通过节点分区或子图拆分实现的训练路径不纳入该比较，因为其改变了完整图上的训练执行方式，不能与 VecGAD 的 full-batch 结果视为同一口径。
  - 论文中的效率与 OOM 结论必须明确限定于 full-batch setting，不外推为 GGAD/RHO 在任意子图拆分实现下均不可运行。
  - 可以保留 VecGAD 面向大规模图及百万节点图的 full-batch scalability claim。对 DGraph 只声称可行性优势，即 VecGAD 可以训练而 GGAD/RHO 在相同 GPU 条件下不可行；不得将其表述为 VecGAD 在 DGraph 上具有更快训练速度。
  - 正文效率段仅用一句话陈述 DGraph 的 full-batch 可行性结论，不将 DGraph 加入四数据集训练速度表。补充材料使用独立状态表报告 GGAD `OOM`、RHO `OOM`、VecGAD `Completed`，并给出 GPU 型号与显存、各方法 source commit 以及 VecGAD 峰值显存。
  - 权威重跑必须固定各方法的 source commit、逐数据集配置及“完整训练 epoch”的计时边界；GPU 计时前后执行同步，完成预热后测量多个 epoch，并保存原始日志与永久 W&B provenance。
  - 四数据集横向效率表对 VecGAD、GGAD 和 RHO 同时报告峰值 GPU 显存与完整训练 epoch 时间；峰值显存作为 full-batch scalability 和 OOM 结论的直接量化证据保留。
  - 上述四数据集显存/epoch 时间比较以紧凑表格保留在正文，作为 scalability claim 的直接证据；Tokenization 时间、详细计时协议、关键配置、参数量和 DGraph 独立状态表放入补充材料。
  - Tokenization 作为一次性预处理单独计时，与完整训练 epoch 时间并列报告；不额外计算或报告端到端总训练成本。
  - 不再使用精确的 `<1% of training time` claim，也不要求通过总 epoch 数证明该阈值。正文仅在数据支持时将其弱化为对“一次性预处理开销较小”的定性描述，并让读者直接依据所报告的原始时间判断。
  - 正文主结果表中的运行时间采用 CPU 后端测得，不与拓展性实验的 GPU 结果直接比较。
  - 拓展性实验所用 GPU 型号暂不预设，在实验完成后于表注或实验设置中明确说明最终硬件条件。
  - 现有 GPU OOM 历史记录为：GGAD 在 Questions、Elliptic 上 OOM；RHO 在 Questions、T-Finance、Elliptic 上 OOM。由于 OOM 依赖具体硬件，最终稿中的四数据集横向结果与 DGraph 独立可行性结论均以选定统一 GPU 上的重测结果为准。
  - 在落稿前复核所有运行时间的单位、统计范围和硬件来源，避免将 CPU 主表结果与统一 GPU 拓展性结果混为同一实验口径。
  - 预热 epoch 数、正式计时 epoch 数以及采用均值或中位数作为表中单值，延后到实际执行本 TODO 时根据届时硬件、负载与运行成本讨论确定；当前不冻结。

### B4. 历史 `figs` 数据的 sweep provenance 追溯

- [ ] 人工尝试为 `figs/` 下论文图表使用的各个数据点恢复精确 W&B sweep/run provenance。
  - 优先记录每张图、每个数据集和每组参数对应的永久 sweep/run 链接、配置、seed/聚合方式、source commit 与指标字段。
  - 对 $K$ 与 $\alpha$ 参数分析，Amazon 当前数值继续以 `figs/propagation_steps/propagation_steps.py` 和 `figs/residual_weight/residual_weight.py` 为权威；`docs/VecFormer/experiments.md` 仅用于辅助检索历史线索，不作为数值裁决来源。
  - 若找到原始 sweep，在代码注释或论文 LaTeX 注释中补充可复查链接；只有发现明确的数据抄录错误时，再返回讨论是否修改图表。
  - 本项属于 provenance 完善和锦上添花，不阻塞当前 AAAI 模板迁移、正文精简或既定稿件修改。

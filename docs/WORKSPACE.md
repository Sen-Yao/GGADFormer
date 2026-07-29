# VecGAD 工作区定义

更新日期：2026-07-20

## 定义

当前目录是 VecGAD 的独立主工作区，服务于四类工作：

1. VecGAD 方法与代码的维护；
2. 可复现实验、数据协议和结果审计；
3. 论文主稿与补充材料的准备；
4. KDD 2026 拒稿后的 AAAI 2027 复投迭代。

已确认的项目事实：

- 方法名：`VecGAD`；旧项目名为 `GGADFormer`。
- 方法提出者：Ziyao Lin。
- KDD 2026 投稿状态：拒稿，公开讨论页为 <https://openreview.net/forum?id=fB79pLB4RN#discussion>。
- 当前目标 venue：AAAI 2027 Main Technical Track。
- canonical code remote：<https://github.com/Sen-Yao/GGADFormer>。
- 当前迁移基线提交：`28bce1a83bc87d7cd1d2dce423da7c79b296c5b7`。

完整作者名单、当前单位和互惠审稿人候选尚不能从匿名 KDD PDF 或代码仓库可靠恢复，必须由当前投稿团队确认，不能由历史元数据推断。

## 事实优先级

发生冲突时按以下顺序处理：

1. 当前冻结的实验 protocol、runner 结果和最终论文主稿；
2. 当前代码及对应提交；
3. `docs/submission/` 中带来源和日期的投稿记录；
4. `literature/vecgad/paper.pdf` 中的 KDD 版本内容；
5. `docs/history/dualrefgad/` 中的历史分析。

DualRefGAD 历史资料记录的是“VecGAD 对 DualRefGAD 有何启发”，不自动构成 VecGAD 自身的贡献、结论或实验事实。

## 目录边界

- 根目录代码、`VecGAD.py`、`model.py`、`run.py`：VecGAD 当前实现。
- `docs/VecGAD.md`：仓库原有的方法和实验说明。
- `docs/submission/`：venue 流程、提交字段、历史文案和复用判断。
- `docs/history/dualrefgad/`：从 DualRefGAD 原样复制的历史研究记录。
- `literature/vecgad/`：本地 KDD PDF、元数据和阅读卡；被 Git 忽略。
- `dataset/`、`wandb/`、`logs/`：本地数据与运行产物；不进入 Git。

## 当前已知的不一致

GitHub `README.md`、`docs/VecGAD.md`、KDD PDF 和 DualRefGAD 的历史 protocol card 中存在部分结果数字不一致，例如 Photo 的 AUROC/AUPRC。提交 AAAI 前必须冻结唯一 protocol，并从正式结果 artifact 重新生成论文表格、摘要结果句和 README 数字。

Reddit 结果已经作为例外完成冻结：VecGAD 的权威五随机种子均值为 AUROC `0.5782`、AUPRC `0.0441`。其来源为 W&B project `HCCS/GGADFormer` 的 sweep `qs4t9byw`（`end_lr=1e-4`，seeds `0-4`）；后续稿件与材料不得用其他历史 Reddit 数值覆盖该结果。

在完成该审计前：

- 不把任一历史表格称为 AAAI 最终结果；
- 不在摘要中写未经统一验证的提升幅度；
- 不用单个数据集或单个版本的 best score 支撑稳定性结论。

DAGAD 已按后续冻结决定从当前 AAAI 主表和实验比较口径中移除；其历史候选数值与未来严格复现要求仅保留在根目录 [`TODOLIST.md`](../TODOLIST.md)。

## 投稿入口

AAAI 2027 的时间线、摘要截止前必填项和 VecGAD 填写建议见 [`submission/AAAI-2027.md`](submission/AAAI-2027.md)。KDD 文案原文和复用审计见 [`submission/KDD-2026.md`](submission/KDD-2026.md)。

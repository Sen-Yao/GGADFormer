# 假设与判定边界

## H1：已部署 batch mean 对受控异常侧位移保持稳定

在固定 Tolokers 已部署配置、split、五个训练 seed 和固定训练终点下，仅改变 HSC center 后，`default` 与 `q0..q40` 的 AUROC/AUPRC 差异能够反映 center contamination 的训练效应，而不是初始化、batch 顺序或伪异常 source 漂移。

支持条件是同 seed 的 pairing hashes 全部一致，且 `q` 条件相对 `default` 的 paired differences 呈现可重复方向。反证条件是配对身份不一致、结果只来自 best epoch/seed，或差异方向在五个 seed 中高度不稳定。

## H2：normal-only center 不一定优于 sampled-batch mean

`q0` 将 center 固定为 batch 正常节点均值，但这不等价于更好的 HSC 几何参考。若 `q0-default` 在固定终点上同时降低 AUROC 与 AUPRC，则当前协议只支持“移除异常侧贡献不能改善 HSC reference”这一有界结论。

## 声明边界

本实验是 Tolokers 单数据集、固定配置、五 seed 的正式 protocol。它可以更新 Supplement 中 Tolokers 的中心污染结果与对应叙述，但不能据此推断所有数据集、所有 HSC 参数或所有 center estimator 的普遍鲁棒性。

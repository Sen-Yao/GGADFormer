# Reddit/Photo alpha=0.1 严格配对重跑

## 目标与边界

本协议审计 Technical Supplement 中 `progregate_alpha=0.1` 与已报告 Reddit/Photo 结果配置不一致的问题。它不预先裁决论文 authority，也不构成公开预注册；它只在启动前冻结可执行、可审计的 20 个 trial 身份。

配对设计为：

- Reddit：历史对照 `alpha=0` vs 候选 `alpha=0.1`；
- Photo：历史对照 `alpha=0.05` vs 候选 `alpha=0.1`；
- 每组固定 seeds `0--4`，共 20 个独立 W&B runs；
- 同一数据集内，对照与候选除 `progregate_alpha` 外配置完全一致。

## 指标口径

主结果只使用第 200 epoch 的 `AUC.last` 和 `AP.last`，并报告五种子均值、样本标准差及同 seed 配对差值。同时从完整 history 独立计算 `AUC.max`/`AP.max` 及各自对应 epoch，仅作训练动态诊断，不拼接为同一个“最佳 epoch”结果。

## 有效性与重试

只有 crashed/failed、非零退出、终点 history 缺失，或配置、Git SHA、数据哈希不匹配时，才允许保留原 run 与诊断后以相同身份重跑。任何已完成且协议有效的低分、离群值或较差 seed 均不得替换或排除。

## 执行契约

- 科学基线为 `0f81d27555c67a25067e85153a51d9cf9693db87`；
- 正式执行使用已提交的 detached Git worktree；
- HCCS-25/85/90 完成实时 Git/runtime/data/W&B/GPU preflight 后确定性选主机；
- 一个 sweep 终生绑定一台主机和 manifest 中的显式 GPU 索引；
- wrapper 设置 `PYTHONHASHSEED=<seed>` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，但由于科学基线未启用 `torch.use_deterministic_algorithms(True)`，不声称 bitwise-identical trajectory；
- 使用 native W&B sweep 和每 GPU 一个 tmux pane；
- W&B 只传输配置、seed、运行状态、AUROC/AUPRC 及允许的运行元数据，不上传原始数据、源码、凭据、checkpoint 或未声明 artifact。

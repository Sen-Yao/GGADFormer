# 进展记录

## 2026-08-01 - investigation 创建与 protocol 冻结

- 建立 governed Scope v1；30 个 condition-seed trial、独立 replay、证据归档、Supplement 更新和 HCCS-85 释放均为 required。
- 旧 sweep `25agh73h` 使用 `K=3`、`progregate_alpha=0.3`、`lambda_rec_emb=0.5`、`num_epoch=70`、`R_min=R_max=0.5`、`ring_loss_weight=20`，仅保留为无效 protocol lineage，不参与本次聚合。
- 本次 Tolokers 配置固定为论文已部署/default 参数；只改变训练时 HSC center。
- 当前未创建新 W&B sweep，未启动正式 agent，未产生新 runs。

## 2026-08-01 - execution code 冻结

- execution code SHA: `d8fdc7a2e0f6c7cfceedbc163f03b0d3a2a287bd`。
- tree SHA: `400e545e35300f5d21cf394d3827ed32d6d015d8`。
- Mac 上 12 个 wrapper/evidence tests 通过，Python compile 与 YAML 解析通过。
- Mac runtime 没有 `torch`，因此 `tests/test_hsc_center.py` 留待 HCCS-85 的已审计 runtime 执行；正式 launch gate 在该测试和 live preflight 通过前保持关闭。

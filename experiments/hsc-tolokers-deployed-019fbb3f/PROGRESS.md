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

## 2026-08-01 - HCCS-85 preflight 与 native sweep 创建

- HCCS-85 live probe：`gpufree-container`、tmux 3.2a、8 张 RTX 4090、launch 前无 compute PID；数据 cache SHA 为 `d6ec349...27a1d0`。
- 远端 clean detached worktree 为 execution SHA `d8fdc7a...a287bd`，source bundle SHA 为 `7131f284...4d82747c`。
- HSC unit/evidence tests 共 20 个在 HCCS runtime 通过；dataset `loadmat` inventory 通过；deterministic q10 非正式 smoke 通过。
- native W&B sweep 已创建且仅创建一次：`HCCS/GGADFormer/txc1ymqu`。此时 manifest 仍保持 `formal_launch_allowed=false`，等待 selected-host 最终 re-probe 与 launch checkpoint。

## 2026-08-01 - formal launch gate opened

- selected-host targeted re-probe 通过：HCCS-85 `gpufree-container`，8 个目标 GPU 无 compute PID，tmux 3.2a；证据为 `launch-preflight.json`。
- occupancy 保持当前 investigation 的 `reserved`，HCCS-90 的并行 task 保持不触碰。
- manifest `formal_launch_allowed=true`；下一步只允许把 occupancy 转为 `running`，创建 task-specific tmux panes 并记录 pane/GPU identity。

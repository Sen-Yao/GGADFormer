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

## 2026-08-01 - formal sweep launched

- occupancy 已转为 `running`，native sweep `txc1ymqu` 于 `2026-08-01T08:48:05Z` 启动。
- tmux session `vecgad_hsc_tolokers_019fbb3f` 有 8 个 panes，分别绑定 GPU `0..7`；agent count 总和为 30。
- 首批 8 个 W&B runs 均为 `running`，config 轴从 `default/seed 0..4` 与 `q0/seed 0..2` 开始；8 张 GPU 均出现对应 task-owned compute PID。
- 当前只记录运行身份，不解释中间指标。最终结论仍要求 30/30 terminal valid、agent exit 0、collector 与独立 replay 全部通过。

## 2026-08-01 - 30/30 terminal validation

- 30 个 condition-seed runs 全部 `finished`，missing/duplicate/unexpected 均为 0；native sweep 已 graceful stop 并转为 `FINISHED`。
- 8 个 agents 全部写出 terminal record 且 exit code 均为 0，正式训练于 `2026-08-01T08:52:51Z` 前结束。
- 首次 collector 通过后，W&B 延迟物化了每 run 一个 provider-generated `wandb-history` artifact；初次 replay 因原 validator 的 artifact-empty 假设而 fail closed。
- validator commit `244617470ce17b7d8d96cf27df23a7558fcd4447` 将边界收紧为只接受 `run-<id>-history:v0`、`type=wandb-history`、单文件 `0000.parquet`、无 metadata/used artifacts 的后端 history manifest；任何其他 artifact 仍失败。训练 execution SHA 未改变。
- HCCS-85 clean detached validation worktree 上的 9 个 tests 通过。最终 collector 与独立 replay 均通过，证据位于 `evidence-final/`。

## 2026-08-01 - scientific interpretation and supplement update

- Default 为 AUROC `0.6640 +/- 0.0062`、AUPRC `0.3148 +/- 0.0073`；`q=0` 显著退化，`q=0.1` 两指标在五个 paired seeds 中均小幅下降，`q=0.2--0.4` 两指标均在五个 seeds 中改善。
- `q=0.4` 的平均 center shift from Default 为 `0.1062307525`；旧文中的 Tolokers `0.254` 已更正。
- Supplement 只更新 Tolokers AUROC/AUPRC 列、paired-seed 叙述、Tolokers center-shift 与不可见 provenance 注释；未渲染或编译 PDF。
- HCCS-85 释放与 final closure commit 尚待完成。

## 2026-08-01 - evidence checkpoint and HCCS-85 release

- 高精度 authoritative/results/replay 证据、REPORT、insights 与 validated manifest 已提交并推送为 `f1dc0b69e642f26bffa25de7cb963cc79557084c`。
- 精确 tmux session、execution/validation detached worktrees、dataset/cache symlinks 与 artifact-audit scratch 已清理；没有终止或修改其他 tmux sessions。
- `2026-08-01T09:52:57Z` live release audit 确认 sweep 仍为 `FINISHED` 30/30、GPU 无 compute PID、无本任务 process/session/worktree/symlink，远端 final evidence hashes 未变化。
- HCCS-85 occupancy 已更新为 `free`；investigation Scope v1 的 M1--M4 全部完成并关闭。

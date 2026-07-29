# T-Finance `ring_R_min=0.3` 正式复现实验协议

## 目标与边界

本实验以 W&B 权威 sweep `HCCS/GGADFormer/iqxjqsdl` 为唯一基线，复现
T-Finance seeds `0,1,2,3,4`，只将 `ring_R_min` 从 `0.5` 改为 `0.3`。
每个 seed 是一个独立 W&B trial，固定训练 40 epochs，报告 `AUC.last` 和
`AP.last`；不选择最优 epoch、seed 或 metric。

科学执行代码固定为权威 runs 共同记录的完整 Git SHA：

```text
e071ae6646451d94fc8e8c9e88305eb76c393089
```

该 SHA 是 `origin/main` 的祖先。当前项目 HEAD `9eb3c669e2de9f11644b0e4a4a9d705354238726`
及历史 HCCS-90 部署 `28bce1a83bc87d7cd1d2dce423da7c79b296c5b7`
在 `run.py`、`utils.py` 和模型实现上均已与权威代码发生科学变化，不能替代权威 SHA。

禁止修改科学逻辑、数据 split、metric 定义、模型、优化器、采样、评估位置或
冻结论文结果。禁止上传原始数据、源码、凭据、checkpoint 或未声明 artifact。

## 权威证据

`authoritative-sweep.json` 是 2026-07-29 通过 W&B public API 的只读查询结果，
包含 sweep config、五个 finished run 的 ID、source commit、真实 argv、
`AUC.last`/`AP.last`、runtime 和全部已记录评估 history（steps 0/10/20/30/40）。

权威五 seed 结果为：

| Seed | Run ID | AUC.last | AP.last |
|---:|---|---:|---:|
| 0 | `u23q4y5p` | 0.8976451645322565 | 0.6438040560862622 |
| 1 | `4tlo98yu` | 0.9030300142950548 | 0.6323573841312963 |
| 2 | `b7q1e0sg` | 0.9006482553969007 | 0.6620247970736751 |
| 3 | `mkfgrxpw` | 0.8938798023817345 | 0.6410958853668718 |
| 4 | `6snuqhrf` | 0.8988293637624499 | 0.6303765186745383 |

独立重算得到 AUC mean `0.8988065200736793`、sample std (ddof=1)
`0.0034224153818109976`；AP mean `0.6419317282665288`、sample std
`0.012581003469923047`。与任务给定的十位小数均值一致。

## 配置不变量

`resolved-config-baseline.json` 和 `resolved-config-rmin03.json` 保存完整 resolved
config 与 seed trial axis。二者的 canonical JSON diff 必须只有：

```text
fixed_config.ring_R_min: 0.5 -> 0.3
```

`sweep.yaml` 保持权威 sweep 的 `program`、`method`、metric 和参数集合不变，
只应用上述单项值变化。创建 sweep 时必须从解析后的 YAML 创建一次，不得把路径
字符串传给 API；调用超时后必须先查询 W&B，禁止盲目再次创建。

## HCCS-90 执行契约

- 唯一执行主机：`HCCS-90`；不回退或迁移到其他 HCCS。
- 代码 worktree：task-owned durable path 下从 `e071ae...` 创建的 clean detached worktree。
- Python runtime：复用经重新审计的 user-owned VecGAD Python 3.8 / Torch 2.0
  cu118 / DGL 1.1.3 环境，不覆盖环境或 package cache。
- 数据：只读复用现有 `t_finance.mat`，inventory 和 SHA-256 必须与 manifest 一致。
- GPU：预声明 indices `0,1,2,3,4`，每个 pane 一个 GPU、一个 `wandb agent
  HCCS/GGADFormer/<sweep-id> --count 1`。启动前再次确认每个 GPU 没有 compute PID。
- tmux：task-owned session，`remain-on-exit=on`；记录 pane ID、GPU、首个 agent
  启动时间、退出码和日志路径。
- agent launcher：每个 pane 精确执行 `launch-agent.sh <gpu> n30dxpp2`；该脚本只负责
  环境变量、native agent 和 task-owned 审计日志，不修改科学代码或监控进程。
- W&B：获得本任务具体写授权后才能创建新 sweep。设置 `WANDB_DISABLE_CODE=true`
  和 `WANDB_CONSOLE=off`，不上传源码或 console；不声明或上传 artifact。
- 监控：长运行仅使用一个 Codex one-shot heartbeat，不创建 watcher 或 recurring automation。

## 完成条件

五个 seeds 均须 W&B `finished`，每个 run 的 config、commit、history、summary 与
manifest 一致，且本地独立聚合/回放通过。任何 crashed/failed、缺失 trial、配置
漂移、identity 不一致或非 0 退出都使正式证据无效并进入诊断。完成 handoff 前
occupancy 保持 `running`；确认无 task-owned 残留进程后才通过 helper 释放。

## 最终结果

新 sweep：`HCCS/GGADFormer/n30dxpp2`。五个 run 均 finished、step 40、agent
exit code 0，且完整 resolved config 与候选配置逐字段相等。

| Seed | Run ID | AUC.last | AP.last |
|---:|---|---:|---:|
| 0 | `uvluzm4a` | 0.9007241725551484 | 0.6576483240310368 |
| 1 | `1smy9jeq` | 0.9023773477030808 | 0.6179107309759142 |
| 2 | `xrqpnqbu` | 0.9019033590636678 | 0.6673770657857934 |
| 3 | `knhyf3m7` | 0.8855968833883806 | 0.6535736747492829 |
| 4 | `88j4a5va` | 0.8968212776899674 | 0.6336637314569272 |

AUC mean/sample std 为 `0.897484608080049 / 0.006994945403890782`，相对基线
mean delta 为 `-0.001321911993630298`。AP mean/sample std 为
`0.6460347053997909 / 0.019944971027181832`，相对基线 mean delta 为
`+0.004102977133262131`。完整 history、逐 seed delta、时间和验证结论见
`results.json`；远端日志 hash 见 `remote-log-sha256.txt`。

运维偏差：tmux `remain-on-exit` 仅保留 `%8`，正常退出后的 `%9-%12` 被 tmux
回收。五个 panes 的 ID、timestamp、完整日志、exitcode 和 W&B identity 均已保留
并通过 hash/配置回放，因此该偏差无科学影响。

## 论文 authority 裁决

2026-07-29，用户明确指定以正式五 seed sweep `HCCS/GGADFormer/n30dxpp2`
替代 `iqxjqsdl` 作为论文 T-Finance 主结果 authority。论文主表及正文采用五 seed
`AUC.last`/`AP.last` 均值的四位小数 `0.8975/0.6460`；`reproduction.sh`
同步采用 `ring_R_min=0.3` 并记录本 sweep 的完整 provenance。`iqxjqsdl` 继续作为
单变量对照基准保留，不改写其历史配置或结果。

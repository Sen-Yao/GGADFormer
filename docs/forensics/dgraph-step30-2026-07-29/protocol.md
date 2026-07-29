# DGraph 历史 step-30 取证协议

状态：prepared

本协议只回答一个有界问题：在历史 source commit、历史 resolved config 和用户指定的 DGraph 数据候选下，仅改变训练 `seed`，能否在 seeds `0,1,2,3,4` 的精确 W&B logging step `30` 得到可核验的 AUC/AP。结果只能称为“历史 step-30 forensic evidence”，不能替代当前统一协议、fixed-last 结果、论文冻结结果或正式复现。

## 历史 run 事实

- W&B：`HCCS/GGADFormer/43edp77a`，display name `lively-valley-69270`，无 sweep，最终状态 `crashed`。
- source commit：`08fa68eb149ab1bb9972d4e141e3c0a3857024f7`；本仓库与 `origin/main` 均可解析，且它是当前 `9eb3c669e2de9f11644b0e4a4a9d705354238726` 的祖先。
- W&B metadata command：`python run.py --dataset dgraph --num_epoch=1000 --peak_lr=1e-4 --end_lr=5e-5 --batch_size=65536 --train_rate=0.05`。
- 历史环境：Python `3.8.20`、Torch `2.0.0`、DGL `1.1.3`、NumPy `1.21.6`、SciPy `1.7.3`、scikit-learn `1.0.2`、wandb `0.22.2`；单个训练进程实际使用 NVIDIA L40 `cuda:0`。
- 历史 step `30`：AUC `0.6005969753665018`，AP `0.005738912433357575`，`_runtime=3700.853709692` 秒，learning rate `0.000062`。
- 历史 run 在 epoch 187 附近收到 `KeyboardInterrupt`；最后 W&B history step 为 `184`。这不是 source code 或数据格式异常。

完整 51 项 resolved config 见 [resolved-config.json](resolved-config.json)，文件 SHA-256 为 `b3c621682b42a9cb89fa2e5c86fcd709c1cc2b64df7b666d6bf1eb9cafa00e50`。

## loader 与 step 语义

commit `08fa68eb...` 的 `utils.py::load_dgraph` 固定读取工作目录下：

- `./dataset/dgraphfin.npz`：使用 `x` 和 `y`，并将 `y == 1` 定义为异常标签；
- `./dataset/dgraphfin_adj_list`：用 `pickle.load` 读取邻接表，要求 key 覆盖 `0..N-1`，随后无向化、去权、加自环并做对称归一化。

`run.py` 执行 `for epoch in range(num_epoch + 1)`，每个 epoch 先完成一次训练更新，再调用 scheduler；当 `epoch % 10 == 0` 时在固定 `idx_test` 上计算 AUC/AP，并显式 `wandb.log(..., step=epoch)`。因此 step `30` 是完成 epoch 编号 `0..30`、即 31 次训练迭代后的测试指标。scheduler 的 `tot_updates` 实际传入 `args.num_epoch`，所以必须保持 `num_epoch=1000`；把它改为 31 会改变 step `0..30` 的 learning-rate 轨迹。

所有 seed 使用固定 `data_split_seed=42`、`train_rate=0.05`、`val_rate=0.1`。测试集是剩余 85% 节点；AUC 使用 `roc_auc_score`，AP 使用 `average_precision_score(..., pos_label=1)`，分数越大越异常。

## 用户指定数据候选

Capability URL 不写入公共 Git；下表的 share 1/2 与用户委托中给出的两个 Nextcloud 分享按顺序一一对应。两个 share 的 WebDAV root 都是单文件对象，没有其他目录或文件被枚举或下载。

| source | 文件 | WebDAV bytes | mtime (UTC) | SHA-256 |
| --- | --- | ---: | --- | --- |
| share 1 | `dgraphfin.npz` | 680317982 | 2022-04-19 09:12:31 | `95470dab2c48523f7118a92204c090de37a957bb053bd5841c7bdba09558ba85` |
| share 2 | `dgraphfin_adj_list` | 399146049 | 2026-01-26 07:30:13 | `b82e16aed09f00985e26e596ff894e7e43cc99d5a45357e13075a4ffa95387c1` |

本地 Git 外暂存目录为 `/private/tmp/vecgad-dgraph-forensic.cnDDEj`。NPZ 全 archive CRC 通过；`x` shape=`(3700550,17)`、dtype=`float64`，`y` shape=`(3700550,)`、dtype=`int64`，两者均为有限值，`y == 1` 共 15,509 个节点。adjacency 通过受限 unpickler 检查：仅允许 `collections.defaultdict`、`builtins.set`、`numpy.core.multiarray.scalar` 和 `numpy.dtype`，对象为 `defaultdict(set)`，3,700,550 个连续 key，邻居范围 `0..3700549`，无非法值或尾随字节。

NPZ 的 `edge_index` shape=`(4300999,2)`。它包含 3,997,260 条无向唯一边、无自环；双向展开正好得到 adjacency 的 7,994,520 个邻居项，NPZ 输入边在 adjacency 中的缺失数为 0。这个一致性证明两个 share 属于同一 DGraph 图实例。限制是历史 W&B run 未记录原始数据 hash，因此无法证明这些字节与 2025 年 run 的本地文件逐字节相同；share 2 的 Nextcloud mtime 也晚于历史 run。

## HCCS-90 执行约束

预备 live probe（2026-07-29 10:50 CST）观察到：host `gpufree-container`，8x RTX 4090，driver `580.126.09`，全部 GPU utilization 0%、约 24 GiB free、无 compute PID；`/root/gpufree-data` 约 5.1 TiB 可用，RAM 约 491 GiB 可用。cooperative occupancy 在同一轮为 `free`。既存 tmux pane 全部 dead；已知 smoke checkout 是 clean detached `28bce1a8...`，env 为 Python 3.8.20 / Torch 2.0.0+cu118 / DGL 1.1.3，Photo smoke 成功。

这个 snapshot 不是未来空闲保证。正式 claim 前必须重新读取 occupancy 并重新 live probe；只要 HCCS-90 不再是 `free`、出现 T-Finance 或其他 owner、probe 超时，或选定 GPU 出现 compute PID，就停止，不上传、不改环境、不训练。

预声明的新路径：

- data：`/root/gpufree-data/linziyao/VecGAD-dgraph-data-95470dab2c48-b82e16aed09f`；
- run root：`/root/gpufree-data/linziyao/VecGAD-dgraph-step30-forensic-08fa68eb`；
- detached worktree：`<run root>/worktree`；
- env：`/root/gpufree-data/linziyao/.conda/envs/VecGAD-forensic-08fa68eb`；
- tmux session：`vecgad_dgraph_step30_08fa68eb_20260729`；
- GPUs：`0,1,2,3,4`，分别绑定 seeds `0,1,2,3,4`。

所有路径必须不存在后再创建，禁止覆盖 smoke、VecGAD、DualRefGAD 或其他任务资料。worktree 必须从完整 source SHA 创建为 clean detached worktree；`dataset/` 中只放指向独立 data cache 的两个 ignored symlink。新 env 从已验证 smoke env clone 后只把 wandb 对齐到 `0.22.2`，不得修改原 env。

## 运行与停止协议

每个 seed 的训练命令保持历史非默认参数，仅改变 `--seed`：

```bash
python run.py --dataset dgraph --num_epoch=1000 --peak_lr=1e-4 --end_lr=5e-5 --batch_size=65536 --train_rate=0.05 --seed=<0..4>
```

每个 seed 独立 pane、独立 log 和独立 `WANDB_DIR`。由于用户尚未授权向 W&B 新写入 config/metrics/source，运行强制使用 `WANDB_MODE=offline`；不得 sync，不创建 cloud run/project/sweep，不发送源码、数据或 artifact。wandb 0.22.2 本地 smoke 已证明 `run-*.wandb` 可通过 SDK 自带 `DataStore`/protobuf reader 精确回放 `_step=30` 的 AUC/AP。

使用一个 Codex one-shot heartbeat 做跟进，不部署 watcher 或持续轮询。每次检查只读回放各 task-owned offline run：某 seed 出现同时含 `_step=30`、`AUC` 和 `AP` 的 history row 后，允许对该 seed pane 发送 `Ctrl-C`，然后核对本地记录已持久化。不得把 step `40` 或其他 step 替代 step `30`。若在不改科学逻辑的条件下无法可靠持久化或停止，则停止整个任务并记录 blocker，不修补 loader 或训练代码。

## 证据边界

即使五 seed 全部完成，本次结果仍有三项不可消除的偏差：原 run 使用 L40 而本任务使用 RTX 4090；原 run 未记录数据 hash；本任务不向 W&B cloud 写入新 run，只保留本地 offline evidence。因此结果只用于历史轨迹取证，不得晋升为当前统一 operator、正式论文复现或模型稳定性结论。

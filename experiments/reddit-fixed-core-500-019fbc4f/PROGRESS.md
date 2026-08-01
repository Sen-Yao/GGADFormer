# Progress

## 2026-08-01 准备检查点

- lineage：`origin/main=0f81d27`；最近严格 Reddit 协议为其后继 `7a0540e`，本任务从后者开始。
- 旧证据边界：`ry7lvaiy` 的固定核心五种子为 AUROC 0.551455 ± 0.007220、AUPRC 0.040330 ± 0.001947；`v1a7ab4r` 因 test-metric 驱动不作为独立验证。
- 三机 live probe：HCCS-25 有外来 compute PID，只剩 GPU 2--6；HCCS-85/90 均有 8 张无 compute PID 的 RTX 4090，完整 preflight 通过。
- deterministic scorer：HCCS-85 与 HCCS-90 同为 8 GPU / 660.64 TFLOPS 估计值，固定 tie-break 选择 HCCS-85。
- occupancy：已按用户预授权完成 `pending_approval -> reserved`，owner 为当前 task `019fbc4f-3c4e-7701-a1de-37f373693859`。
- protocol：500 条记录硬上限已冻结；当前 created/consumed 为 0。
- 下一决策：验证并提交 prepared protocol；在 clean detached worktree 完成一个计费 smoke 后，只有 smoke 身份、数据、指标和固定核心均有效才创建一次 screening sweep。

当前材料不包含结果判断，screening/promotion/test 均未开始。

- prepared commit `081c246b1da213d093174375991b3582f4429b6e` was superseded before launch by `c20a37f327921f40c833be5e535bfc9a3f7cafc8`, which withholds test labels during validation-only selection; manifest binding commit pending before HCCS detached launch.
- HCCS-85 launch preflight revalidated at 2026-08-01T08:29:05Z: all eight GPUs clean, detached SHA/data/runtime/W&B identities exact. Smoke run identity reserved as `fcfqzf5w`.
- HCCS-85 was claimed by another owner before occupancy start; it was not launched or modified. Full rerank at 2026-08-01T08:30:47Z excluded it and selected HCCS-90, which passed the same 8-GPU/data/runtime/W&B preflight. Launcher host identity was made manifest-bound before relaunch.
- Smoke `fcfqzf5w` finished valid on HCCS-90 GPU 0 at 2026-08-01T08:35:46Z: fixed-core config exact, validation-only metrics `Val/AUC.last=0.5257899142079934`, `Val/AP.last=0.04450442254016716`, no test metrics, no uploaded artifacts. Budget consumed is 1/500; remaining 499.
- The first tmux shell expansion error was pre-assignment and consumed no W&B record; it is fenced in `manifest.yaml`.
- Native screening sweep `2cibydx2` was created exactly once from parsed YAML at 2026-08-01T08:40:00Z. Its grid is 192 configs × seeds 0--1 = 384 validation-only records; launch is pending exact occupancy rebinding.

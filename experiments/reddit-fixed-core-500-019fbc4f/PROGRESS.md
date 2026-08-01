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

- prepared commit: `081c246b1da213d093174375991b3582f4429b6e`; manifest binding commit pending before HCCS detached launch.

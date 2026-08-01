# Reddit 固定核心 500-run 超参数扫描

## 研究问题

在固定 `dataset=reddit`、`progregate_alpha=0.0`、`lambda_rec_emb=0.1`、`ring_R_max=1.0` 的前提下，能否通过模型中性的训练超参数搜索，得到可靠且接近 PDF 17 所报 AUROC 0.5782 / AUPRC 0.0441 的五种子结果。

本协议继承严格 Reddit lineage `7a0540e`，但不把旧的 test-driven sweep `v1a7ab4r` 当作独立验证。W&B 目的地固定为 `HCCS/GGADFormer`。

## 标签与指标边界

`utils.load_mat` 用 `data_split_seed=42` 固定 5% train、10% validation 和剩余 test。训练仍只使用 train 中已知 normal 标签。screening 和 promotion 运行 `evaluation_protocol=validation_only`：代码不会构造 test loader，也不会读取 `idx_test` 的指标。只有 promotion 排名冻结前六个配置后，confirmation 才运行 `frozen_test`，并且只在各 trial 的固定最终 epoch 读取一次 test。

配置选择顺序预声明为：

1. screening 对每个配置运行 seeds 0、1，以 `Val/AUC.last` 均值排序，`Val/AP.last` 均值作固定 tie-break；
2. 前 12 个配置用新的 validation seeds 2--6 做 promotion，仍按同一规则排序；
3. 冻结前 6 个配置后，用 test seeds 0--4 做 confirmation；
4. 正式报告只使用固定最终 epoch 的均值和样本标准差。禁止 per-seed best epoch、跨 seed 拼接或根据 test 改写候选。

validation 标签用于超参数选择是本协议的明确科研假设；其目的不是估计最终泛化性能。test 只承担冻结后的确认职责。

## Scope v1

| ID | Scope item | Class | Completion evidence | Trigger / budget | State |
|---|---|---|---|---|---|
| S1 | validation-only screening | required | screening | 192 configs × 2 seeds = 384 | complete |
| P1 | fresh-seed promotion | required | canonical-coverage | top 12 × 5 seeds = 60 | complete |
| C1 | frozen test confirmation | required | promotion | top 6 × 5 seeds = 30 | complete |
| R1 | technical retries | adaptive | implementation validity | crashed/identity-invalid only; at most 25 | not-triggered |

最终三轴状态为 `Lifecycle: complete`、`Coverage: required-complete`、`Scientific verdict: mixed`。固定核心扫描已完整覆盖预声明范围；AUROC 与 PDF 17 接近，但 AUPRC 仍有差距，因此不作全面复现成功的表述。

## Promotion 冻结结果

Promotion sweep `55chbpyh` 的 60 条 validation-only 记录全部有效。按预声明的 mean `Val/AUC.last`、mean `Val/AP.last`、`config_id` 顺序独立重放后，confirmation 冻结为：

| Rank | Config | Val AUROC mean ± sample std | Val AP mean ± sample std |
|---:|---|---:|---:|
| 1 | `cfg-117` | 0.585104 ± 0.020637 | 0.052359 ± 0.012751 |
| 2 | `cfg-058` | 0.541091 ± 0.044289 | 0.043003 ± 0.008546 |
| 3 | `cfg-183` | 0.535944 ± 0.042194 | 0.036757 ± 0.005250 |
| 4 | `cfg-177` | 0.535779 ± 0.040434 | 0.036414 ± 0.004542 |
| 5 | `cfg-018` | 0.532823 ± 0.050233 | 0.042230 ± 0.006503 |
| 6 | `cfg-016` | 0.528330 ± 0.038136 | 0.036518 ± 0.004086 |

完整 validation 证据位于 `promotion-results.json`。该冻结过程未读取 test；六个配置固定后，confirmation 才能用 seeds 0--4 在固定最终 epoch 各读取一次 test。

## Confirmation 固定最终结果

Confirmation sweep `219k2jj2` 的 30 条记录全部有效。下表严格保留 promotion validation 冻结顺序；Test 指标没有用于重排、挑选配置或选择 epoch。标准差为五个预声明 seeds 0--4 的样本标准差。

| Validation freeze rank | Config | Test AUROC mean ± sample std | Test AP mean ± sample std |
|---:|---|---:|---:|
| 1 | `cfg-117` | 0.579653 ± 0.017170 | 0.041303 ± 0.003952 |
| 2 | `cfg-058` | 0.521017 ± 0.048216 | 0.037056 ± 0.004736 |
| 3 | `cfg-183` | 0.546887 ± 0.027083 | 0.036153 ± 0.003326 |
| 4 | `cfg-177` | 0.544546 ± 0.038492 | 0.037006 ± 0.005304 |
| 5 | `cfg-018` | 0.497024 ± 0.021790 | 0.034359 ± 0.002773 |
| 6 | `cfg-016` | 0.559765 ± 0.029880 | 0.038347 ± 0.004380 |

预冻结第一名 `cfg-117` 相对 PDF 17 的 AUROC 0.5782 高 0.001453，AUPRC 0.0441 低 0.002797。它相对严格固定核心历史控制 `ry7lvaiy` 的 AUROC 0.551455 / AUPRC 0.040330 分别高 0.028198 和 0.000973，但 AUROC 的 seed 间离散程度也更大。因此证据支持“在不可变固定核心下找到 AUROC 接近 PDF 的配置”，不支持“完整复现 PDF 两项指标”或跨数据集泛化结论。

完整逐 run 证据、固定最终 step、聚合值和摘要 SHA 位于 `confirmation-results.json`。审计前所有 run 的 application artifact 清单为 0。一次 `scan_history` 审计调用使 W&B 服务端为 `3kstc07c` 物化了内部 `wandb-history` 对象；它不是训练或客户端上传，也没有引入授权范围外的数据类别。此后终态审计不再使用该 API，并在 manifest 中单列该副作用。

## 搜索空间

screening registry 用固定 `SEARCH_SEED=20260801` 生成 192 个唯一配置，包含当前声明控制配置，并覆盖：

- batch size：256、512、1024、2048；
- peak LR：1e-4、2e-4、3e-4、5e-4、8e-4；
- end LR：1e-5、3e-5、1e-4；
- warmup updates：5、20、50；
- epochs：100、150、200、250；
- `pp_k`：4、6、8、10、12；
- `outlier_beta`：0.1、0.2、0.3、0.5；
- reconstruction/ring loss weight：各 0.25、0.5、1、2；
- `ring_R_min`：0.1、0.3、0.5、0.7；
- AdamW weight decay：0、1e-6、1e-5、1e-4。

Transformer 宽度、深度和方法结构保持不变。每个 resolved config 都由 `protocol.py` 和 `run.py` 的 parsed-argument guard 双重断言固定核心四项。

## 500-run 硬预算

| Phase | W&B records |
|---|---:|
| smoke | 1 |
| screening | 384 |
| promotion | 60 |
| confirmation | 30 |
| technical retry reserve | 25 |
| hard total | 500 |

任何 smoke、失败或重试都会消耗一个 ordinal。低分、离群值或科学上不理想的有效 run 不得重跑。manifest 中 `next_record_ordinal` 在任何会创建或分配新 run 的操作前检查；ordinal 501 永远拒绝。

最终科学消耗为 475/500：smoke 1、screening 384、promotion 60、confirmation 30，未发生失败或重试；25 条技术重试预留保持未使用，并在 investigation 完成后不再分配。

## 外部数据授权

用户明确授权本 investigation 在 HCCS 执行，并向 `HCCS/GGADFormer` 发送 config、seed、run status、必要 metadata、AUROC 和 AP。原始数据、源码、凭据、checkpoint 和未声明 artifact 不上传。trial 配置关闭 training-loss、console、source-code 和 model artifact 上传。

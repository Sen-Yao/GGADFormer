# Tolokers HSC Center-Contamination Rerun

## Verdict

The deployed-configuration rerun is complete and replaces the old Tolokers HSC center-contamination results from sweep `25agh73h`. Sweep `txc1ymqu` contains exactly 30 finished runs: six center conditions by five training seeds. The fixed epoch-100 endpoint, all authoritative configuration fields, execution SHA, local checkpoint and diagnostic hashes, and within-seed pairing hashes passed both collection and an independent W&B replay.

The response is asymmetric rather than monotonically robust. The normal-only center (`q=0`) substantially reduces both metrics for every seed. `q=0.1` also produces a small reduction for every paired seed, whereas `q=0.2`, `q=0.3`, and `q=0.4` improve AUROC and AUPRC for every paired seed. These results do not establish the deployed sampled-batch mean as optimal; they rule out a naive normal-only replacement and delimit the sensitivity observed under the tested anomaly-side shifts.

## Fixed-Endpoint Results

| Center | AUROC mean +/- sample std | AUPRC mean +/- sample std |
|---|---:|---:|
| Default | 0.6640012397 +/- 0.0061786313 | 0.3147889299 +/- 0.0072722700 |
| q=0 | 0.4573195807 +/- 0.0923393647 | 0.2016444258 +/- 0.0366723430 |
| q=0.1 | 0.6584054531 +/- 0.0088217635 | 0.3099454772 +/- 0.0092965628 |
| q=0.2 | 0.6811287013 +/- 0.0040228599 | 0.3279394030 +/- 0.0019236629 |
| q=0.3 | 0.6888928849 +/- 0.0044337680 | 0.3338384804 +/- 0.0025702252 |
| q=0.4 | 0.6930377674 +/- 0.0044580697 | 0.3372791182 +/- 0.0024038944 |

Mean paired differences versus Default are `-0.2066816590/-0.1131445041` for `q=0`, `-0.0055957866/-0.0048434526` for `q=0.1`, `+0.0171274616/+0.0131504731` for `q=0.2`, `+0.0248916452/+0.0190495505` for `q=0.3`, and `+0.0290365277/+0.0224901883` for `q=0.4` (AUROC/AUPRC). The `q=0.4` mean center shift from Default is `0.1062307525`.

## Lineage And Evidence

- Execution host: `HCCS-85` (`gpufree-container`, 8 x RTX 4090)
- Task branch: `codex/hsc-tolokers-deployed-4193`
- Training execution SHA: `d8fdc7a2e0f6c7cfceedbc163f03b0d3a2a287bd`
- Evidence-validator SHA: `244617470ce17b7d8d96cf27df23a7558fcd4447`
- Sweep: `https://wandb.ai/HCCS/GGADFormer/sweeps/txc1ymqu`
- Authoritative evidence: `evidence-final/authoritative-sweep.json` (`b939db30d26a7a1468d82e48f672764d86ab063e73519030bef4d5ec549f536d`)
- Aggregated results: `evidence-final/results.json` (`5ac960ea8203c51cbee3877ad95ef08592854472006e31ca31ba96a6b024ad57`)
- Independent replay: `evidence-final/replay-results.json` (`3d46dec31d16d709221f0667b84336d4c0358d8fc5d372998f738f701d554a53`)
- Remote agent log hashes: `remote-log-sha256.txt` (`09d3452284843c5e2b67b7ea1d6fc4f835d4ce8ec77cbce8815957e072de665b`)

W&B asynchronously materialized one provider-generated `wandb-history` artifact per run after the first collector query. The final validators require its exact provider history structure and independently confirm there are no user-uploaded or used artifacts. No source, raw data, credentials, checkpoints, embeddings, or node-level results were uploaded.

## Closure

Evidence checkpoint `f1dc0b69e642f26bffa25de7cb963cc79557084c` is pushed on the task branch. The shared supplement was updated without rendering or compiling a PDF. The task-specific tmux session, detached worktrees, processes, and symlinks were removed; a live release audit found no task process or GPU compute PID. HCCS-85 occupancy was released to `free` at `2026-08-01T09:52:57Z`.

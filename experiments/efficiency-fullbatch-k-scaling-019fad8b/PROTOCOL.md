# VecGAD full-batch efficiency audit protocol

- Execution host: HCCS-90, GPUs 0--7 (RTX 4090, 24 GB). Eight native W&B agents run concurrently. Physical GPU index is recorded but is not an experimental factor.
- W&B destination: `HCCS/VecGAD`. Only config, status, timing, resource metadata, and run identity are uploaded. Raw datasets, source, credentials, and raw result files remain on HCCS.
- Full-batch sweep: Amazon and T-Finance; VecGAD, GGAD, and official RHO at commit `a394d5575dea6745215b15e0453e1f925ffcc1f2`; three fresh-process repeats per cell, giving 18 trials.
- All methods use a 5% training ratio. VecGAD/GGAD use data-split seed 42. The RHO harness uses a dedicated split RNG with seed 42 so model seed 0 is not repurposed as the split seed.
- Each measured epoch must contain exactly one optimizer step. VecGAD uses a single sampled batch of size `N`; GGAD uses its full-graph path; RHO uses upstream `batch_size=0`, which executes the complete N-by-N InfoNCE path without internal splitting.
- Dataset reads are outside the preparation timer. `offline.seconds` starts after data loading and ends when the first training epoch can begin. `offline.tokenization_seconds` isolates VecGAD propagation and token stacking.
- Each trial warms up for 10 complete epochs and records the next 30. Epoch time includes sampling/DataLoader work, host-to-device transfer, forward, loss, backward, optimizer, and scheduler. Evaluation, checkpointing, W&B logging, and terminal rendering are excluded.
- CUDA timing is synchronized. CPU RSS and CUDA allocated/reserved baseline, absolute peak, and delta are recorded separately for preparation and training.
- An OOM is terminal only after two fresh child-process CUDA OOMs under the identical W&B assignment. Harness/infrastructure errors fail the W&B run and are never converted to OOM.
- No batch-size fallback, graph partition, hidden-dimension adjustment, or result-driven configuration change is permitted.
- The existing mini-batch diagnostic audit is separate evidence and must not be merged with this sweep.
- The T-Finance K-scaling profile is a separate successor sweep because concurrent CPU SpMM trials would confound tokenization timing.

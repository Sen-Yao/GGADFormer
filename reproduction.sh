# Commit: 458664a84bb37d617e86b950deed9642b16e543c

## Amazon

## AUC=0.9533, AP=0.8171

python run.py --batch_size=1024 --dataset=Amazon --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=100 --peak_lr=0.0003 --pp_k=5 --progregate_alpha=0.4 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

## Reddit

## Epoch=201/200, AUC=0.5827, AP=0.0464

python run.py --batch_size=1024 --dataset=reddit --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=200 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=10 --progregate_alpha=0 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

# photo

## AUC=0.8972, AP=0.6439

# W&B provenance for the paper's 5-seed Photo result (alpha=0.05, seeds 0-4):
# https://wandb.ai/HCCS/GGADFormer/sweeps/v98ueupn

python run.py --batch_size=128 --dataset=photo --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=200 --peak_lr=0.0005 --pp_k=6 --progregate_alpha=0.05 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

## Elliptic

# W&B provenance for the paper's 5-seed Elliptic result (seeds 0-4):
# https://wandb.ai/HCCS/GGADFormer/sweeps/39e3dk75

## AUC=0.7876, AP=0.3027

python run.py --batch_size=32768 --dataset=elliptic --end_lr=0.0003 --lambda_rec_emb=2 --num_epoch=150 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=7 --progregate_alpha=0.6 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=20 --seed=0 --train_rate=0.05 --warmup_updates=50

# Epoch=136/200, AUC=0.7459, AP=0.197

python run.py --batch_size=8192 --dataset=elliptic --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=150 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=8 --progregate_alpha=0.8 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

## T-Finance

# W&B provenance: https://wandb.ai/HCCS/GGADFormer/sweeps/n30dxpp2
# Scientific code: e071ae6646451d94fc8e8c9e88305eb76c393089
# Seeds 0-4, AUC.last/AP.last means: 0.897484608080049 / 0.6460347053997909
# Sample std (ddof=1): 0.006994945403890782 / 0.019944971027181832
# Main-table values after four-decimal rounding: 0.8975 / 0.6460

python run.py --batch_size=8192 --dataset=t_finance --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=40 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=7 --progregate_alpha=0.3 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

## Tolokers

# W&B provenance for the unified Tolokers 5-seed result (lambda_rec_emb=0.1,
# ring_loss_weight=1, seeds 0-4):
# https://wandb.ai/HCCS/GGADFormer/sweeps/2acum2mg
# Scientific code: bb798db0e32615abd8504da7ccb21a124102b363
# Seeds 0-4, AUC.last/AP.last means: 0.6659020323844487 / 0.31507833656938
# Sample std (ddof=1): 0.004186228825945539 / 0.007207500868862045
# Audit manifest: experiments/tolokers-lrec-unification-019fb2b1/manifest.yaml

python run.py --batch_size=1024 --dataset=tolokers --end_lr=0.00001 --lambda_rec_emb=0.1 --num_epoch=100 --outlier_beta=0.3 --peak_lr=0.0001 --pp_k=10 --progregate_alpha=0.9 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=5

## DGraph

# Historical W&B provenance only; this is not formal five-seed evidence:
# https://wandb.ai/HCCS/GGADFormer/runs/43edp77a (lively-valley-69270)
# https://wandb.ai/HCCS/GGADFormer/runs/m042886o
# Both are crashed seed-0 runs from commit 08fa68eb149ab1bb9972d4e141e3c0a3857024f7.
# Their step-30 AUC.max/AP.max are 0.6005969754/0.0057389124; seeds 1-4 and
# sweep lineage are absent. B1 requires a unified-operator five-seed rerun.

python run.py --batch_size=65536 --dataset=dgraph --end_lr=0.00001 --lambda_rec_emb=0.1 --num_epoch=200 --outlier_beta=0.3 --peak_lr=0.00005 --pp_k=10 --progregate_alpha=0.9 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=5

python run.py --dataset dgraph --num_epoch=100 --peak_lr=1e-4 --end_lr=5e-5

# ---

CUDA_VISIBLE_DEVICES=5 python run.py --batch_size=8192 --dataset=elliptic --end_lr=0.0001 --lambda_rec_emb=0.05 --num_epoch=150 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=7 --progregate_alpha=0.6 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

python run.py --batch_size=8192 --dataset=elliptic --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=200 --outlier_beta=0.3 --peak_lr=0.0003 --pp_k=8 --progregate_alpha=0.8 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

# Epoch=136/200, AUC=0.7459, AP=0.197

python run.py --batch_size=8192 --dataset=elliptic --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=200 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=8 --progregate_alpha=0.8 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50


CUDA_VISIBLE_DEVICES=5 python run.py --batch_size=1024 --dataset=tolokers --end_lr=0.00001 --lambda_rec_emb=0.1 --num_epoch=100 --outlier_beta=0.3 --peak_lr=0.0001 --pp_k=10 --progregate_alpha=0.9 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=5  --visualize=True

# Historical ablation retained with its original lambda_rec_emb=5 setting.

CUDA_VISIBLE_DEVICES=5 python run.py --batch_size=1024 --dataset=tolokers --end_lr=0.00001 --lambda_rec_emb=5 --num_epoch=100 --outlier_beta=0.3 --peak_lr=0.0001 --pp_k=10 --progregate_alpha=0.9 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=5  --rec_error_filter_ratio=0.5

# Commit: 458664a84bb37d617e86b950deed9642b16e543c

## Amazon

## AUC=0.9533, AP=0.8171

python run.py --batch_size=1024 --dataset=Amazon --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=100 --peak_lr=0.0003 --pp_k=5 --progregate_alpha=0.4 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

## Reddit

## Epoch=201/200, AUC=0.5827, AP=0.0464

python run.py --batch_size=1024 --dataset=reddit --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=200 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=10 --progregate_alpha=0 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

# photo

## AUC=0.8972, AP=0.6439

python run.py --batch_size=128 --dataset=photo --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=200 --peak_lr=0.0005 --pp_k=6 --progregate_alpha=0.05 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

## Elliptic

## AUC=0.7876, AP=0.3027

python run.py --batch_size=32768 --dataset=elliptic --end_lr=0.0003 --lambda_rec_emb=2 --num_epoch=150 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=7 --progregate_alpha=0.6 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=20 --seed=0 --train_rate=0.05 --warmup_updates=50

# Epoch=136/200, AUC=0.7459, AP=0.197

python run.py --batch_size=8192 --dataset=elliptic --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=150 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=8 --progregate_alpha=0.8 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

## T-Finance

# AUC=0.8786, AP=0.4704

python run.py --batch_size=8192 --dataset=t_finance --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=40 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=7 --progregate_alpha=0.3 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

## Tolokers

# Epoch=201/200, AUC=0.6683, AP=0.3120

python run.py --batch_size=1024 --dataset=tolokers --end_lr=0.00001 --lambda_rec_emb=5 --num_epoch=100 --outlier_beta=0.3 --peak_lr=0.0001 --pp_k=10 --progregate_alpha=0.9 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=5

## DGraph

python run.py --batch_size=65536 --dataset=dgraph --end_lr=0.00001 --lambda_rec_emb=0.1 --num_epoch=200 --outlier_beta=0.3 --peak_lr=0.00005 --pp_k=10 --progregate_alpha=0.9 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=5

python run.py --dataset dgraph --num_epoch=100 --peak_lr=1e-4 --end_lr=5e-5

# ---

CUDA_VISIBLE_DEVICES=5 python run.py --batch_size=8192 --dataset=elliptic --end_lr=0.0001 --lambda_rec_emb=0.05 --num_epoch=150 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=7 --progregate_alpha=0.6 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

python run.py --batch_size=8192 --dataset=elliptic --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=200 --outlier_beta=0.3 --peak_lr=0.0003 --pp_k=8 --progregate_alpha=0.8 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50

# Epoch=136/200, AUC=0.7459, AP=0.197

python run.py --batch_size=8192 --dataset=elliptic --end_lr=0.0001 --lambda_rec_emb=0.1 --num_epoch=200 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=8 --progregate_alpha=0.8 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50


CUDA_VISIBLE_DEVICES=5 python run.py --batch_size=8192 --dataset=elliptic --end_lr=0.0001 --lambda_rec_emb=5 --num_epoch=150 --outlier_beta=0.3 --peak_lr=0.0005 --pp_k=8 --progregate_alpha=0.8 --rec_loss_weight=1 --ring_R_max=1 --ring_R_min=0.3 --ring_loss_weight=1 --seed=0 --train_rate=0.05 --warmup_updates=50
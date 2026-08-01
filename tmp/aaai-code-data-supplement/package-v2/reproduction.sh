#!/usr/bin/env bash

# This file only prints a command catalog. It never launches training.
cat <<'COMMANDS'
Amazon
python run.py --dataset=Amazon --seed=0 --data_split_seed=42 --train_rate=0.05 --batch_size=1024 --num_epoch=100 --peak_lr=0.0003 --end_lr=0.0001 --warmup_updates=50 --pp_k=5 --progregate_alpha=0.4 --sample_rate=0.15 --outlier_beta=0.3 --lambda_rec_tok=1 --lambda_rec_emb=0.1 --bce_loss_weight=1 --rec_loss_weight=1 --ring_loss_weight=1 --ring_R_min=0.3 --ring_R_max=1 --control=full

Reddit
python run.py --dataset=reddit --seed=0 --data_split_seed=42 --train_rate=0.05 --batch_size=1024 --num_epoch=200 --peak_lr=0.0005 --end_lr=0.0001 --warmup_updates=50 --pp_k=10 --progregate_alpha=0.1 --sample_rate=0.15 --outlier_beta=0.3 --lambda_rec_tok=1 --lambda_rec_emb=0.1 --bce_loss_weight=1 --rec_loss_weight=1 --ring_loss_weight=1 --ring_R_min=0.3 --ring_R_max=1 --control=full

Photo
python run.py --dataset=photo --seed=0 --data_split_seed=42 --train_rate=0.05 --batch_size=128 --num_epoch=200 --peak_lr=0.0005 --end_lr=0.0001 --warmup_updates=50 --pp_k=6 --progregate_alpha=0.1 --sample_rate=0.15 --outlier_beta=0.3 --lambda_rec_tok=1 --lambda_rec_emb=0.1 --bce_loss_weight=1 --rec_loss_weight=1 --ring_loss_weight=1 --ring_R_min=0.3 --ring_R_max=1 --control=full

Elliptic
python run.py --dataset=elliptic --seed=0 --data_split_seed=42 --train_rate=0.05 --batch_size=32768 --num_epoch=150 --peak_lr=0.0005 --end_lr=0.0003 --warmup_updates=50 --pp_k=7 --progregate_alpha=0.6 --sample_rate=0.15 --outlier_beta=0.3 --lambda_rec_tok=1 --lambda_rec_emb=2 --bce_loss_weight=1 --rec_loss_weight=1 --ring_loss_weight=20 --ring_R_min=0.3 --ring_R_max=1 --control=full

T-Finance
python run.py --dataset=t_finance --seed=0 --data_split_seed=42 --train_rate=0.05 --batch_size=8192 --num_epoch=40 --peak_lr=0.0005 --end_lr=0.0001 --warmup_updates=50 --pp_k=7 --progregate_alpha=0.3 --sample_rate=0.15 --outlier_beta=0.3 --lambda_rec_tok=1 --lambda_rec_emb=0.1 --bce_loss_weight=1 --rec_loss_weight=1 --ring_loss_weight=1 --ring_R_min=0.3 --ring_R_max=1 --control=full

Tolokers
python run.py --dataset=tolokers --seed=0 --data_split_seed=42 --train_rate=0.05 --batch_size=1024 --num_epoch=100 --peak_lr=0.0001 --end_lr=0.00001 --warmup_updates=5 --pp_k=10 --progregate_alpha=0.9 --sample_rate=0.15 --outlier_beta=0.3 --lambda_rec_tok=1 --lambda_rec_emb=0.1 --bce_loss_weight=1 --rec_loss_weight=1 --ring_loss_weight=1 --ring_R_min=0.3 --ring_R_max=1 --control=full

DGraph
python run.py --dataset=dgraph --seed=0 --data_split_seed=42 --train_rate=0.05 --batch_size=65536 --num_epoch=200 --peak_lr=0.00005 --end_lr=0.00001 --warmup_updates=5 --pp_k=10 --progregate_alpha=0.9 --sample_rate=0.15 --outlier_beta=0.3 --lambda_rec_tok=1 --lambda_rec_emb=0.1 --bce_loss_weight=1 --rec_loss_weight=1 --ring_loss_weight=1 --ring_R_min=0.3 --ring_R_max=1 --control=full
COMMANDS

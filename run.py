import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from model import Model
from VecGAD import VecGAD
from SGT import SGT
from utils import *

from sklearn.metrics import roc_auc_score
import os
import random
import subprocess
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time
import torch.utils.data as Data

import wandb
from visualization import create_tsne_visualization, visualize_attention_weights, visualize_reconstruction_analysis
from utils import send_notification, calculate_graph_statistics
from ablation_rec_error import evaluate_with_rec_error_filter
from hsc_center import HSC_CENTER_Q, compute_center_components, compute_shell_statistics

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
# os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# Set argument

def get_git_head_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_index_trace(digest, trace_name, epoch, batch_index, indices):
    values = indices.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
    digest.update(
        f"{trace_name}:{epoch}:{batch_index}:{values.size}\n".encode("ascii")
    )
    digest.update(values.tobytes())


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_and_reload_final_checkpoint(model, args):
    checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_id = wandb.run.id if wandb.run is not None else "local"
    checkpoint_path = checkpoint_dir / f"{run_id}.pt"
    identity = {
        "run_id": run_id,
        "dataset": args.dataset,
        "hsc_center_condition": args.hsc_center_condition,
        "seed": args.seed,
        "data_split_seed": args.data_split_seed,
        "code_sha": os.environ.get("CODE_SHA") or get_git_head_sha(),
        "protocol_identity": os.environ.get("PROTOCOL_ID", "unrecorded"),
        "final_training_epoch": args.num_epoch,
    }
    torch.save({"identity": identity, "model_state_dict": model.state_dict()}, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    payload = torch.load(checkpoint_path, map_location=model.device)
    if payload.get("identity") != identity:
        raise RuntimeError("final checkpoint identity mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return checkpoint_path, checkpoint_sha256, identity


def run_hsc_diagnostic_replay(model, batch_data_train, weights, normal_global_idx, args, device):
    diagnostic_seed = 1_000_000 + args.data_split_seed * 100 + args.seed
    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(diagnostic_seed)
    source_generator = torch.Generator(device="cpu")
    source_generator.manual_seed(diagnostic_seed + 1)
    sampler = Data.WeightedRandomSampler(
        weights,
        num_samples=len(batch_data_train),
        replacement=True,
        generator=sampler_generator,
    )
    loader = Data.DataLoader(
        batch_data_train,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )

    replay_batch_trace = hashlib.sha256()
    replay_source_trace = hashlib.sha256()
    totals = {
        "count": 0,
        "shell_count": 0,
        "inner_count": 0,
        "outer_count": 0,
        "hsc_loss_sum": 0.0,
        "center_shift_from_default_sum": 0.0,
        "center_shift_from_normal_sum": 0.0,
        "anomaly_count": 0,
        "node_count": 0,
        "batch_count": 0,
    }

    cuda_devices = [device.index] if device.type == "cuda" else []
    model.eval()
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(diagnostic_seed + 2)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(diagnostic_seed + 2)
        with torch.no_grad():
            for batch_index, item in enumerate(loader):
                input_tokens = item[0].to(device)
                batch_labels = item[1].to(device)
                batch_global_indices = item[2].to(device)
                update_index_trace(
                    replay_batch_trace,
                    "diagnostic_batch",
                    0,
                    batch_index,
                    batch_global_indices,
                )

                is_known_normal = torch.isin(batch_global_indices, normal_global_idx)
                local_normal_idx = torch.nonzero(
                    is_known_normal, as_tuple=False
                ).squeeze(-1)
                source_count = int(local_normal_idx.numel() * args.sample_rate)
                if source_count == 0:
                    raise RuntimeError("diagnostic replay selected no pseudo-anomaly sources")
                permutation = torch.randperm(
                    local_normal_idx.numel(), generator=source_generator
                ).to(device)
                local_source_idx = local_normal_idx[permutation[:source_count]]
                update_index_trace(
                    replay_source_trace,
                    "diagnostic_source",
                    0,
                    batch_index,
                    batch_global_indices[local_source_idx],
                )

                emb = model.TransformerEncoder(input_tokens)
                centers = compute_center_components(
                    emb, batch_labels, args.hsc_center_condition
                )
                outlier_emb, _, _, _ = model.build_pseudo_outliers(
                    input_tokens, emb, local_source_idx, args
                )
                shell = compute_shell_statistics(
                    outlier_emb,
                    centers.selected,
                    args.ring_R_min,
                    args.ring_R_max,
                )
                for key in (
                    "count",
                    "shell_count",
                    "inner_count",
                    "outer_count",
                    "hsc_loss_sum",
                ):
                    totals[key] += shell[key]
                totals["center_shift_from_default_sum"] += float(
                    torch.norm(centers.selected - centers.default, p=2).item()
                )
                totals["center_shift_from_normal_sum"] += float(
                    torch.norm(centers.selected - centers.normal, p=2).item()
                )
                totals["anomaly_count"] += int((batch_labels == 1).sum().item())
                totals["node_count"] += int(batch_labels.numel())
                totals["batch_count"] += 1

    if totals["count"] == 0 or totals["batch_count"] == 0:
        raise RuntimeError("diagnostic replay produced no HSC observations")
    count = totals["count"]
    batch_count = totals["batch_count"]
    return {
        "diagnostic_seed": diagnostic_seed,
        "pseudo_anomaly_count": count,
        "batch_count": batch_count,
        "ShellHit": totals["shell_count"] / count,
        "inner_violation": totals["inner_count"] / count,
        "outer_violation": totals["outer_count"] / count,
        "mean_hsc_loss": totals["hsc_loss_sum"] / count,
        "center_shift_from_default": totals["center_shift_from_default_sum"] / batch_count,
        "center_shift_from_normal": totals["center_shift_from_normal_sum"] / batch_count,
        "sampled_anomaly_fraction": totals["anomaly_count"] / totals["node_count"],
        "batch_trace_sha256": replay_batch_trace.hexdigest(),
        "source_trace_sha256": replay_source_trace.hexdigest(),
    }


def infer_run_variant(args):
    if abs(args.lambda_rec_emb - 2.0) < 1e-12 and abs(args.ring_loss_weight - 20.0) < 1e-12:
        return "control_2_20"
    if abs(args.lambda_rec_emb - 5.0) < 1e-12 and abs(args.ring_loss_weight - 1.0) < 1e-12:
        return "control"
    if abs(args.lambda_rec_emb - 0.1) < 1e-12 and abs(args.ring_loss_weight - 1.0) < 1e-12:
        return "unified_0p1_1"
    return f"lambda_rec_emb={args.lambda_rec_emb:g};ring_loss_weight={args.ring_loss_weight:g}"


def build_wandb_audit_config(args):
    wandb_entity = os.environ.get("WANDB_ENTITY", "HCCS")
    wandb_project = os.environ.get("WANDB_PROJECT", "GGADFormer")
    code_sha = os.environ.get("CODE_SHA") or get_git_head_sha()
    gpu_index = os.environ.get("GPU_INDEX") or os.environ.get("CUDA_VISIBLE_DEVICES") or str(args.device)

    return {
        "variant": infer_run_variant(args),
        "hsc_center_condition": args.hsc_center_condition,
        "hsc_oracle_q": HSC_CENTER_Q[args.hsc_center_condition],
        "hsc_label_usage_scope": "oracle center construction only; excluded from all other losses and scoring",
        "pair_id": f"{args.dataset}:seed={args.seed}:data_split_seed={args.data_split_seed}",
        "protocol_identity": os.environ.get("PROTOCOL_ID", "unrecorded"),
        "split_protocol_identity": (
            f"{args.dataset}:train_rate={args.train_rate}:val_rate=0.1:"
            f"data_split_seed={args.data_split_seed}"
        ),
        "code_sha": code_sha,
        "execution_host": os.environ.get("EXECUTION_HOST", "unknown"),
        "gpu_index": gpu_index,
        "fixed_final_epoch_metric_policy": "AUC.last/AP.last at fixed training endpoint",
        "hsc_diagnostic_policy": "final checkpoint; fixed weighted-sampler replay; sample-weighted shell metrics",
        "wandb_entity": wandb_entity,
        "wandb_project": wandb_project,
    }


def train(args):
    # Set random seed
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # 设置设备
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available() and args.device >= 0:
        print(f'CUDA device name: {torch.cuda.get_device_name(args.device)}')
        print(f'CUDA device memory: {torch.cuda.get_device_properties(args.device).total_memory / 1024**3:.1f} GB')
    else:
        print('Using CPU for computation')

    # Load and preprocess data
    if args.dataset == 'dgraph':
        adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, _, _, normal_for_train_idx, normal_for_generation_idx = load_dgraph(train_rate=args.train_rate, val_rate=0.1, args=args)
        concated_input_features = nagphormer_tokenization(features, adj, args)
        model = VecGAD(features.shape[1], args.embedding_dim, 'prelu', args)
        features = features.to(device)
        adj = adj.to(device)
        labels = torch.tensor(labels).to(device)

        num_nodes = features.shape[0]
        ft_size = features.shape[1]
    else:
        adj, features, labels, all_idx, idx_train, idx_val, \
        idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, 0.1, args=args)

        if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
            features, _ = preprocess_features(features)
        else:
            features = features.todense()


        num_nodes = features.shape[0]
        ft_size = features.shape[1]
        if args.model_type == 'GGAD':
            raw_adj = adj
            #print(adj.sum())
            raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
            raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
            raw_adj = raw_adj.to(device)

        adj = normalize_adj(adj)
        adj = (adj + sp.eye(adj.shape[0])).todense()
        features = torch.FloatTensor(features[np.newaxis])
        # adj = torch.FloatTensor(adj[np.newaxis])
        features = torch.FloatTensor(features)
        adj = torch.FloatTensor(adj)
        # adj = adj.to_sparse_csr()
        adj = torch.FloatTensor(adj[np.newaxis])
        labels = torch.FloatTensor(labels[np.newaxis])

        # 将数据移动到指定设备
        if args.model_type != 'VecGAD':
            features = features.to(device)
            adj = adj.to(device)
            labels = labels.to(device)

        # concated_input_features.shape: torch.Size([1, node_num, 2 * feature_dim])

        # idx_train = torch.LongTensor(idx_train)
        # idx_val = torch.LongTensor(idx_val)
        # idx_test = torch.LongTensor(idx_test)

        # Initialize model and optimiser

        if args.model_type == 'VecGAD':
            concated_input_features = nagphormer_tokenization(features.squeeze(0), adj.squeeze(0), args)
            model = VecGAD(ft_size, args.embedding_dim, 'prelu', args)
        elif args.model_type == 'SGT':
            concated_input_features = preprocess_sample_features(args, features.squeeze(0), adj.squeeze(0)).to(device)
            model = SGT(n_layers=args.GT_num_layers,
                input_dim=concated_input_features.shape[-1],
                hidden_dim=args.embedding_dim,
                n_class=2,
                num_heads=args.GT_num_heads,
                ffn_dim=args.GT_ffn_dim,
                dropout_rate=args.GT_dropout,
                attention_dropout_rate=args.GT_attention_dropout,
                args=args).to(device)
        elif args.model_type == 'GGAD':
            concated_input_features = features.to(device)
            model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout, args)
        else:
            raise ValueError(f"Invalid model type: {args.model_type}")

    # 计算图的平均最短路径和有效直径（基于采样估算）
    # avg_sp, eff_diameter = calculate_graph_statistics(adj, n_samples=1000)

    initial_model_sha256 = model_state_sha256(model)
    training_batch_trace = hashlib.sha256()
    pseudo_source_trace = hashlib.sha256()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=int(0.1 * args.num_epoch) if args.warmup_updates == -1 else args.warmup_updates,
        tot_updates=args.num_epoch,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )

    # 损失函数设置
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
    xent = nn.CrossEntropyLoss()

    auc = 0
    ap = 0
    best_AUC = 0
    best_AP = 0
    best_model_state = None
    best_epoch = 0
    
    if args.model_type == "VecGAD":
        labels = labels.squeeze(0)

        all_node_indices = torch.arange(num_nodes)

        # 在半监督场景中，模型训练时允许访问全图的 feature 和被 normal_for_train_idx 允许的那些 label
        # 为了形式统一，这里将全图的 label 也提供给 Dataset，但是在实际训练中，只有 normal_for_train_idx 的那些 label 允许被使用！
        # 其中 all_node_indices 是用于计算 batch 内部的 normal_for_train_idx 的
        batch_data_train = Data.TensorDataset(concated_input_features, labels, all_node_indices)
        batch_data_val = Data.TensorDataset(concated_input_features[idx_val], labels[idx_val])
        batch_data_test = Data.TensorDataset(concated_input_features[idx_test], labels[idx_test])

        # 对于训练集需要分层采样

        all_indices = set(range(num_nodes))
        known_indices = set(normal_for_train_idx)
        unknown_indices = list(all_indices - known_indices)

        weights = torch.zeros(num_nodes)
        weights[normal_for_train_idx] = 1.0 / len(normal_for_train_idx)
        weights[unknown_indices] = 1.0 / len(unknown_indices)

        # 基于权重，实例化一个采样器
        # replacement=True 允许重复采样，这对于过采样少数类至关重要
        sampler = Data.WeightedRandomSampler(weights, num_samples=num_nodes, replacement=True)


        train_data_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=False)
        val_data_loader = Data.DataLoader(batch_data_val, batch_size=args.batch_size, shuffle = False)
        test_data_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle = False)

        normal_for_train_idx = torch.tensor(normal_for_train_idx, dtype=torch.long, device=device)


    # Train model
    print(f"Start training! Total epochs: {args.num_epoch}")
    pbar = tqdm(total=args.num_epoch, desc='Training')
    total_time = 0
    for epoch in range(args.num_epoch + 1):
        dynamic_weights = get_dynamic_loss_weights(epoch, args)
        start_time = time.time()
        train_flag = True
        model.train()
        if args.model_type == "VecGAD":
            batched_bce_loss = 0
            batched_rec_loss = 0
            batched_ring_loss = 0
            # start_time = time.time()
            for batch_idx, item in enumerate(train_data_loader):
                # print(f"time to start batch {time.time() - start_time}")
                concated_input_features = item[0].to(device)
                labels = item[1].to(device)
                batch_global_indices = item[2].to(device)
                update_index_trace(
                    training_batch_trace,
                    "training_batch",
                    epoch,
                    batch_idx,
                    batch_global_indices,
                )

                optimizer.zero_grad()
                is_known_normal_mask = torch.isin(batch_global_indices, normal_for_train_idx)
                local_normal_for_train_idx = torch.nonzero(is_known_normal_mask, as_tuple=False).squeeze(-1)
                emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, loss_rec, loss_ring = model(concated_input_features, None,
                                                                    labels, local_normal_for_train_idx,
                                                                    train_flag, args)
                update_index_trace(
                    pseudo_source_trace,
                    "pseudo_source",
                    epoch,
                    batch_idx,
                    batch_global_indices[model.last_normal_for_generation_idx],
                )
                    # BCE loss
                lbl = torch.unsqueeze(torch.cat(
                    (torch.zeros(len(local_normal_for_train_idx)), torch.ones(len(outlier_emb)))),
                    1).unsqueeze(0)
                lbl = lbl.to(device)  # 将标签移动到指定设备
                loss_bce = b_xent(logits, lbl)
                loss_bce = torch.mean(loss_bce)

                diff_attribute = torch.pow(outlier_emb - noised_normal_for_generation_emb, 2)
                # loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

                loss = dynamic_weights['bce_loss_weight'] * loss_bce + dynamic_weights['rec_loss_weight'] * loss_rec + dynamic_weights['ring_loss_weight'] * loss_ring

                loss.backward()
                optimizer.step()
                batched_bce_loss += loss_bce
                batched_rec_loss += loss_rec
                batched_ring_loss += loss_ring

            batched_total_loss = batched_bce_loss + batched_rec_loss + batched_ring_loss
            end_time = time.time()
            total_time += end_time - start_time
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 更新进度条信息
            pbar.set_postfix({
                'Time': f'{total_time:.1f}s',
                'Epoch': f'{epoch+1}/{args.num_epoch}',
                'AUC': f'{auc:.4f}',
                'AP': f'{ap:.4f}'
            })
            pbar.update(1)
            if epoch % 2 == 0:
                wandb.log({ "batched_total_loss": batched_total_loss.item(),
                            "bce_loss": batched_bce_loss.item(),
                            "rec_loss": batched_rec_loss.item(),
                            "ring_loss": batched_ring_loss.item(),
                            "learning_rate": current_lr}, step=epoch)
        else:
            optimizer.zero_grad()

            # print("start forward")
            emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, _, con_loss, proj_loss, reconstruction_loss = model(concated_input_features, adj,
                                                                    normal_for_generation_idx, normal_for_train_idx,
                                                                    train_flag, args)

            # BCE loss
            lbl = torch.unsqueeze(torch.cat(
                (torch.zeros(len(normal_for_train_idx)), torch.ones(len(outlier_emb)))),
                1).unsqueeze(0)
            lbl = lbl.to(device)  # 将标签移动到指定设备

            loss_bce = b_xent(logits, lbl)
            loss_bce = torch.mean(loss_bce)
            if args.model_type == 'GGAD':
            # Local affinity margin loss
                emb = torch.squeeze(emb)

                emb_inf = torch.norm(emb, dim=-1, keepdim=True)
                emb_inf = torch.pow(emb_inf, -1)
                emb_inf[torch.isinf(emb_inf)] = 0.
                emb_norm = emb * emb_inf

                sim_matrix = torch.mm(emb_norm, emb_norm.T)
                raw_adj = torch.squeeze(raw_adj)
                similar_matrix = sim_matrix * raw_adj

                r_inv = torch.pow(torch.sum(raw_adj, 0), -1)
                r_inv[torch.isinf(r_inv)] = 0.
                affinity = torch.sum(similar_matrix, 0) * r_inv

                affinity_normal_mean = torch.mean(affinity[normal_for_train_idx])
                affinity_abnormal_mean = torch.mean(affinity[normal_for_generation_idx])

                loss_margin = (args.confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)
            else:
                loss_margin = torch.tensor(0.0)

            diff_attribute = torch.pow(outlier_emb - noised_normal_for_generation_emb, 2)
            loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

            loss = dynamic_weights['margin_loss_weight'] * loss_margin + dynamic_weights['bce_loss_weight'] * loss_bce + dynamic_weights['rec_loss_weight'] * loss_rec + dynamic_weights['con_loss_weight'] * con_loss + dynamic_weights['proj_loss_weight'] * proj_loss + dynamic_weights['reconstruction_loss_weight'] * reconstruction_loss

            loss.backward()
            optimizer.step()
            end_time = time.time()
            total_time += end_time - start_time
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 更新进度条信息
            pbar.set_postfix({
                'Time': f'{total_time:.1f}s',
                'Epoch': f'{epoch+1}/{args.num_epoch}',
                'AUC': f'{auc:.4f}',
                'AP': f'{ap:.4f}'
            })
            pbar.update(1)
            if epoch % 2 == 0:
                wandb.log({ "margin_loss": loss_margin.item(),
                            "bce_loss": loss_bce.item(),
                            "rec_loss": loss_rec.item(),
                            "con_loss": con_loss.item(),
                            "proj_loss": proj_loss.item(),
                            "train_loss": loss.item(),
                            "reconstruction_loss": reconstruction_loss.item(),
                            "learning_rate": current_lr}, step=epoch)
        lr_scheduler.step()
        if epoch % 10 == 0:
            model.eval()
            train_flag = False

            if args.model_type == "VecGAD":
                all_batched_logits = []
                with torch.no_grad():
                    for _, item in enumerate(test_data_loader):
                        concated_input_features = item[0].to(device)
                        labels = item[1].to(device)
                        emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, loss_rec, loss_ring = model(concated_input_features, None, None, None,
                                                                                train_flag, args)
                        all_batched_logits.append(logits.squeeze(0))
                    # Concatenate all batched logits
                    concatenated_logits = torch.cat(all_batched_logits, dim=0)
                    logits = np.squeeze(concatenated_logits.cpu().detach().numpy())
            else: 
                emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, _, con_loss, proj_loss, reconstruction_loss = model(concated_input_features, adj, normal_for_generation_idx, normal_for_train_idx,
                                                                        train_flag, args)
                logits = np.squeeze(logits[:, idx_test, :].cpu().detach().numpy())
            
            # ===== 重构误差过滤消融实验 =====
            # 当 rec_error_filter_ratio != 1.0 时，使用过滤后的节点计算AUROC/AUPRC
            if getattr(args, 'rec_error_filter_ratio', 1.0) != 1.0 and args.model_type == "VecGAD":
                filtered_results = evaluate_with_rec_error_filter(
                    model, test_data_loader, ano_label, idx_test,
                    args, device, args.rec_error_filter_ratio
                )
                auc = filtered_results['auroc']
                ap = filtered_results['auprc']
            else:
                auc = roc_auc_score(ano_label[idx_test], logits)
                ap = average_precision_score(ano_label[idx_test], logits, average='macro', pos_label=1, sample_weight=None)
            
            wandb.log({"AUC": auc, "AP": ap}, step=epoch)
            
            # 检查是否为最佳模型
            if auc > best_AUC and ap > best_AP:
                best_AUC = auc
                best_AP = ap
                best_model_state = model.state_dict().copy()
                best_epoch = epoch

    pbar.close()  # 关闭进度条
    print(f"Training done! Total time: {total_time:.2f} seconds")

    if args.model_type == "VecGAD":
        checkpoint_path, checkpoint_sha256, checkpoint_identity = save_and_reload_final_checkpoint(
            model, args
        )
        diagnostics = run_hsc_diagnostic_replay(
            model,
            batch_data_train,
            weights,
            normal_for_train_idx,
            args,
            device,
        )
        repeated_diagnostics = run_hsc_diagnostic_replay(
            model,
            batch_data_train,
            weights,
            normal_for_train_idx,
            args,
            device,
        )
        if diagnostics != repeated_diagnostics:
            raise RuntimeError("final-checkpoint HSC diagnostic replay is not deterministic")
        diagnostic_record = {
            "schema_version": 1,
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "initial_model_sha256": initial_model_sha256,
            "training_batch_trace_sha256": training_batch_trace.hexdigest(),
            "pseudo_source_trace_sha256": pseudo_source_trace.hexdigest(),
            "final_model_state_sha256": model_state_sha256(model),
            "diagnostic_replay_repeat_verified": True,
            "final_metrics": {
                "AUC.last": float(auc),
                "AP.last": float(ap),
                "final_step": args.num_epoch,
            },
            "hsc_diagnostics": diagnostics,
        }
        diagnostic_dir = Path(os.environ.get("DIAGNOSTIC_DIR", "diagnostics"))
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        run_id = wandb.run.id if wandb.run is not None else "local"
        diagnostic_path = diagnostic_dir / f"{run_id}.json"
        diagnostic_path.write_text(
            json.dumps(diagnostic_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        diagnostic_sha256 = sha256_file(diagnostic_path)

        if wandb.run is not None:
            wandb.run.summary.update({
                "run_valid": True,
                "fixed_endpoint_epoch": args.num_epoch,
                "initial_model_sha256": initial_model_sha256,
                "training_batch_trace_sha256": training_batch_trace.hexdigest(),
                "pseudo_source_trace_sha256": pseudo_source_trace.hexdigest(),
                "checkpoint_sha256": checkpoint_sha256,
                "diagnostic_sha256": diagnostic_sha256,
                "HSC.diagnostic_replay_repeat_verified": True,
                "HSC.ShellHit": diagnostics["ShellHit"],
                "HSC.inner_violation": diagnostics["inner_violation"],
                "HSC.outer_violation": diagnostics["outer_violation"],
                "HSC.mean_loss": diagnostics["mean_hsc_loss"],
                "HSC.center_shift_from_default": diagnostics["center_shift_from_default"],
                "HSC.center_shift_from_normal": diagnostics["center_shift_from_normal"],
                "HSC.sampled_anomaly_fraction": diagnostics["sampled_anomaly_fraction"],
                "HSC.diagnostic_batch_trace_sha256": diagnostics["batch_trace_sha256"],
                "HSC.diagnostic_source_trace_sha256": diagnostics["source_trace_sha256"],
            })
    
    # 在最后一次eval时进行重构误差分析可视化
    if args.visualize and args.model_type == "VecGAD":
        print("\n=== Starting Final Evaluation Reconstruction Analysis ===")
        model.eval()
        train_flag = False
        
        # 收集所有测试集的数据进行重构分析
        all_test_input_tokens = []
        all_test_labels = []
        all_test_embeddings = []
        
        with torch.no_grad():
            for _, item in enumerate(test_data_loader):
                concated_input_features = item[0].to(device)
                labels = item[1].to(device)
                
                # 获取模型嵌入
                emb = model.TransformerEncoder(concated_input_features)
                
                all_test_input_tokens.append(concated_input_features.cpu())
                all_test_labels.append(labels.cpu())
                all_test_embeddings.append(emb.squeeze(0).cpu())
        
        # 合并所有batch的数据
        all_test_input_tokens = torch.cat(all_test_input_tokens, dim=0)
        all_test_labels = torch.cat(all_test_labels, dim=0)
        all_test_embeddings = torch.cat(all_test_embeddings, dim=0)
        
        print(f"测试集总节点数: {len(all_test_labels)}")
        print(f"正常点数量: {(all_test_labels == 0).sum().item()}")
        print(f"异常点数量: {(all_test_labels == 1).sum().item()}")
        
        # 调用重构误差分析可视化
        visualize_reconstruction_analysis(
            model=model,
            input_tokens=all_test_input_tokens.to(device),
            labels=all_test_labels,
            ano_label=ano_label,
            idx_test=idx_test,
            epoch=args.num_epoch,  # 使用最后一个epoch
            args=args,
            device=device
        )
    
    # tsne可视化已在visualize_reconstruction_analysis中完成，此处不再重复
if __name__ == "__main__":


    # 定义一个辅助函数，把各种字符串转成 Python 的 bool
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset', type=str,
                        default='reddit')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_split_seed', type=int, default=42)
    parser.add_argument('--train_rate', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=8192)

    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--proj_dim', type=int, default=64)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
    parser.add_argument('--auc_test_rounds', type=int, default=256)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--var', type=float, default=0.0)
    parser.add_argument('--confidence_margin', type=float, default=2)
    parser.add_argument('--outlier_beta', type=float, default=0.3)
    parser.add_argument('--sample_rate', type=float, default=0.15)
    
    parser.add_argument('--model_type', type=str, default='VecGAD')
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--pp_k', type=int, default=6)
    parser.add_argument('--progregate_alpha', type=float, default=0.2)
    parser.add_argument('--sample_num_p', type=int, default=7)
    parser.add_argument('--sample_num_n', type=int, default=7)
    parser.add_argument('--sample_size', type=int, default=10000)

    parser.add_argument('--GT_ffn_dim', type=int, default=256)
    parser.add_argument('--GT_dropout', type=float, default=0.4)
    parser.add_argument('--GT_attention_dropout', type=float, default=0.4)
    parser.add_argument('--GT_num_heads', type=int, default=2)
    parser.add_argument('--GT_num_layers', type=int, default=3)

    parser.add_argument('--proj_R_max', type=float, default=0.5)
    parser.add_argument('--proj_R_min', type=float, default=0.1)
    parser.add_argument('--ring_R_max', type=float, default=1)
    parser.add_argument('--ring_R_min', type=float, default=0.3)

    parser.add_argument('--rec_loss_weight', type=float, default=1)
    parser.add_argument('--bce_loss_weight', type=float, default=1.0)
    parser.add_argument('--margin_loss_weight', type=float, default=0)
    parser.add_argument('--con_loss_weight', type=float, default=0.1)
    parser.add_argument('--proj_loss_weight', type=float, default=0)
    parser.add_argument('--reconstruction_loss_weight', type=float, default=1.0)
    parser.add_argument('--ring_loss_weight', type=float, default=1.0)

    parser.add_argument('--lambda_rec_tok', type=float, default=1.0)
    parser.add_argument('--lambda_rec_emb', type=float, default=0.1)
    
    parser.add_argument('--con_loss_temp', type=float, default=10)
    parser.add_argument('--GNA_temp', type=float, default=1)
    

    parser.add_argument('--warmup_updates', type=int, default=50)
    parser.add_argument('--tot_updates', type=int, default=1000)
    parser.add_argument('--peak_lr', type=float, default=1e-4)    
    parser.add_argument('--end_lr', type=float, default=1e-4)

    parser.add_argument('--warmup_epoch', type=int, default=20)

    # Ablation Study (perturbation + h_mean center computation + token fusion)
    parser.add_argument('--ablation_mode', type=str, default='none',
                        choices=['none',
                                 'random_dir', 'random_mag', 'random_both', 'constant_mag',
                                 'h_mean_labeled_normal', 'h_mean_trimmed',
                                 'gprgnn_weighted_sum'],
                        help='Ablation mode: none (original model), '
                             'random_dir/random_mag/random_both/constant_mag (perturbation ablation), '
                             'h_mean_labeled_normal (center from labeled normal nodes only), '
                             'h_mean_trimmed (trimmed mean, drop furthest 10%% nodes), '
                             'gprgnn_weighted_sum (GPRGNN-style learnable weighted sum fusion instead of Transformer)')

    parser.add_argument(
        '--hsc_center_condition',
        type=str,
        default='default',
        choices=list(HSC_CENTER_Q),
        help='HSC center intervention: default batch mean or oracle q mixture.',
    )
    
    # Ablation Study: Reconstruction Error Filter
    parser.add_argument('--rec_error_filter_ratio', type=float, default=1.0,
                        help='重构误差过滤比例，选择重构误差最小的该比例节点参与评估。'
                             '默认1.0表示使用全部节点，0.5表示只使用重构误差最小的50%%节点。'
                             '当值为1.0时，保持原有评估逻辑不变。')



    args = parser.parse_args()

    if args.hsc_center_condition != 'default' and args.ablation_mode != 'none':
        parser.error('oracle HSC center conditions require --ablation_mode=none')
    if args.num_epoch is None or args.num_epoch % 10 != 0:
        parser.error('--num_epoch must be a multiple of 10 for fixed AUC.last/AP.last')

    if args.dataset in ['reddit', 'photo']:
        args.mean = 0.02
        args.var = 0.01
    else:
        args.mean = 0.0
        args.var = 0.0


    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "HCCS"),
        # Set the wandb project where this run will be logged.
        project=os.environ.get("WANDB_PROJECT", "GGADFormer"),
        # Track hyperparameters and run metadata.
        config=args,
    )
    wandb.config.update(build_wandb_audit_config(args), allow_val_change=True)

    wandb.define_metric("AUC", summary="max")
    wandb.define_metric("AP", summary="max")
    wandb.define_metric("AUC", summary="last")
    wandb.define_metric("AP", summary="last")
    print('Dataset: ', args.dataset)
        
    try:
        train(args)
    except Exception:
        if wandb.run is not None:
            wandb.run.summary["run_valid"] = False
        wandb.finish(exit_code=1)
        raise
    wandb.finish(exit_code=0)

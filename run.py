import torch.nn as nn

from model import Model
from GGADFormer import GGADFormer
from SGT import SGT
from utils import *

from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time
import torch.utils.data as Data

import wandb
from visualization import create_tsne_visualization, visualize_attention_weights, visualize_reconstruction_analysis
from utils import send_notification, calculate_graph_statistics

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
# os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# Set argument

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
        model = GGADFormer(features.shape[1], args.embedding_dim, 'prelu', args)
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
        if args.model_type != 'GGADFormer':
            features = features.to(device)
            adj = adj.to(device)
            labels = labels.to(device)

        # concated_input_features.shape: torch.Size([1, node_num, 2 * feature_dim])

        # idx_train = torch.LongTensor(idx_train)
        # idx_val = torch.LongTensor(idx_val)
        # idx_test = torch.LongTensor(idx_test)

        # Initialize model and optimiser

        if args.model_type == 'GGADFormer':
            concated_input_features = nagphormer_tokenization(features.squeeze(0), adj.squeeze(0), args)
            model = GGADFormer(ft_size, args.embedding_dim, 'prelu', args)
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
    
    if args.model_type == "GGADFormer":
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
        if args.model_type == "GGADFormer":
            batched_bce_loss = 0
            batched_rec_loss = 0
            batched_ring_loss = 0
            # start_time = time.time()
            for batch_idx, item in enumerate(train_data_loader):
                # print(f"time to start batch {time.time() - start_time}")
                concated_input_features = item[0].to(device)
                labels = item[1].to(device)
                batch_global_indices = item[2].to(device)

                optimizer.zero_grad()
                is_known_normal_mask = torch.isin(batch_global_indices, normal_for_train_idx)
                local_normal_for_train_idx = torch.nonzero(is_known_normal_mask, as_tuple=False).squeeze(-1)
                emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, loss_rec, loss_ring = model(concated_input_features, None,
                                                                    None, local_normal_for_train_idx,
                                                                    train_flag, args)
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

            if args.model_type == "GGADFormer":
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
                    auc = roc_auc_score(ano_label[idx_test], logits)
                    ap = average_precision_score(ano_label[idx_test], logits, average='macro', pos_label=1, sample_weight=None)
            else: 
                emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, _, con_loss, proj_loss, reconstruction_loss = model(concated_input_features, adj, normal_for_generation_idx, normal_for_train_idx,
                                                                        train_flag, args)
                logits = np.squeeze(logits[:, idx_test, :].cpu().detach().numpy())
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
    
    # 在最后一次eval时进行重构误差分析可视化
    if args.visualize and args.model_type == "GGADFormer":
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
    
    parser.add_argument('--model_type', type=str, default='GGADFormer')
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



    args = parser.parse_args()

    if args.dataset in ['reddit', 'photo']:
        args.mean = 0.02
        args.var = 0.01
    else:
        args.mean = 0.0
        args.var = 0.0


    run = wandb.init(
        entity="HCCS",
        # Set the wandb project where this run will be logged.
        project="GGADFormer",
        # Track hyperparameters and run metadata.
        config=args,
    )

    wandb.define_metric("AUC", summary="max")
    wandb.define_metric("AP", summary="max")
    wandb.define_metric("AUC", summary="last")
    wandb.define_metric("AP", summary="last")
    print('Dataset: ', args.dataset)
        
    try:
        train(args)
        start_time = time.time()
        wandb.finish()
        
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"显存不足!：{e}")
        send_notification(f"【VecFormer】出现显存不足!：{e}")
        wandb.log({"AUC.max": 0})
        wandb.log({"AP.max": 0})
        start_time = time.time()
        wandb.finish()
    
    except Exception as e:
        import traceback
        print(f"其他错误：{e}")
        traceback.print_exc()  # 打印详细的错误堆栈，包括出错的代码行
        wandb.log({"AUC.max": 0})
        start_time = time.time()
        wandb.finish()
    print(f"WandB finish took {time.time() - start_time:.2f} seconds")
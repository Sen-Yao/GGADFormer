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

import wandb
from visualization import create_tsne_visualization

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
    adj, features, labels, all_idx, idx_train, idx_val, \
    idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset)

    if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()

    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    raw_adj = adj
    #print(adj.sum())
    adj = normalize_adj(adj)

    raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    # adj = torch.FloatTensor(adj[np.newaxis])
    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(adj)
    # adj = adj.to_sparse_csr()
    adj = torch.FloatTensor(adj[np.newaxis])
    raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])

    # 将数据移动到指定设备
    features = features.to(device)
    adj = adj.to(device)
    raw_adj = raw_adj.to(device)
    labels = labels.to(device)


    # concated_input_features.shape: torch.Size([1, node_num, 2 * feature_dim])

    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    # Initialize model and optimiser

    if args.model_type == 'GGADFormer':
        progregated_features = node_neighborhood_feature(adj.squeeze(0), features.squeeze(0), args.pp_k, args.progregate_alpha).to(args.device).unsqueeze(0)
        concated_input_features = torch.concat((features.to(args.device), progregated_features), dim=2)
        model = GGADFormer(ft_size, args.embedding_dim, 'prelu', args)
    elif args.model_type == 'SGT':
        concated_input_features = preprocess_sample_features(args, features.squeeze(0), adj.squeeze(0)).to(args.device)
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
        concated_input_features = features.to(args.device)
        model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout, args)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 损失函数设置
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
    xent = nn.CrossEntropyLoss()

    best_AUC = 0
    best_AP = 0
    best_model_state = None
    best_epoch = 0

    # Train model
    print(f"Start training! Total epochs: {args.num_epoch}")
    with tqdm(total=args.num_epoch, desc='Training', ncols=100) as pbar:
        total_time = 0
        for epoch in range(args.num_epoch):
            start_time = time.time()
            model.train()
            optimiser.zero_grad()

            # Train model
            train_flag = True
            emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, _, con_loss = model(concated_input_features, adj,
                                                                    normal_for_generation_idx, normal_for_train_idx,
                                                                    train_flag, args)
            if epoch % 10 == 0:
                # save data for tsne
                pass

                # tsne_data_path = 'draw/tfinance/tsne_data_{}.mat'.format(str(epoch))
                # io.savemat(tsne_data_path, {'emb': np.array(emb.cpu().detach()), 'ano_label': ano_label,
                #                             'abnormal_label_idx': np.array(abnormal_label_idx),
                #                             'normal_label_idx': np.array(normal_label_idx)})

            # BCE loss
            lbl = torch.unsqueeze(torch.cat(
                (torch.zeros(len(normal_for_train_idx)), torch.ones(len(outlier_emb)))),
                1).unsqueeze(0)
            lbl = lbl.to(device)  # 将标签移动到指定设备

            loss_bce = b_xent(logits, lbl)
            loss_bce = torch.mean(loss_bce)

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

            # if epoch % 10 == 0:
            #     real_abnormal_label_idx = np.array(all_idx)[np.argwhere(ano_label == 1).squeeze()].tolist()
            #     real_normal_label_idx = np.array(all_idx)[np.argwhere(ano_label == 0).squeeze()].tolist()
            #     overlap = list(set(real_abnormal_label_idx) & set(real_normal_label_idx))
            #
            #     real_affinity, index = torch.sort(affinity[real_abnormal_label_idx])
            #     real_affinity = real_affinity[:300]
            #     draw_pdf(np.array(affinity[real_normal_label_idx].detach().cpu()),
            #              np.array(affinity[abnormal_label_idx].detach().cpu()),
            #              np.array(real_affinity.detach().cpu()), args.dataset, epoch)

            loss_margin = (args.confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)

            diff_attribute = torch.pow(outlier_emb - noised_normal_for_generation_emb, 2)
            loss_rec = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))

            loss = args.margin_loss_weight * loss_margin + 1 * loss_bce + 1 * loss_rec + args.con_loss_weight * con_loss

            loss.backward()
            optimiser.step()
            end_time = time.time()
            total_time += end_time - start_time
            
            # 更新进度条信息
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Time': f'{total_time:.1f}s',
                'Epoch': f'{epoch+1}/{args.num_epoch}'
            })
            pbar.update(1)
            if epoch % 2 == 0:
                logits = np.squeeze(logits.cpu().detach().numpy())
                lbl = np.squeeze(lbl.cpu().detach().numpy())
                auc = roc_auc_score(lbl, logits)
                # print('Traininig {} AUC:{:.4f}'.format(args.dataset, auc))
                # AP = average_precision_score(lbl, logits, average='macro', pos_label=1, sample_weight=None)
                # print('Traininig AP:', AP)

                # print("Epoch:", '%04d' % (epoch), "train_loss_margin=", "{:.5f}".format(loss_margin.item()))
                # print("Epoch:", '%04d' % (epoch), "train_loss_bce=", "{:.5f}".format(loss_bce.item()))
                # print("Epoch:", '%04d' % (epoch), "rec_loss=", "{:.5f}".format(loss_rec.item()))
                # print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()))
                # print("=====================================================================")
                wandb.log({ "bce_loss": loss_bce.item(),
                            "rec_loss": loss_rec.item(),
                            "con_loss": con_loss.item(),
                            "train_loss": loss.item()}, step=epoch)
            if epoch % 10 == 0 and epoch != 0:
                model.eval()
                train_flag = False
                emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, _, con_loss = model(concated_input_features, adj, normal_for_generation_idx, normal_for_train_idx,
                                                                        train_flag, args)
                # evaluation on the valid and test node
                # 在eval阶段，我们需要为所有节点生成logits
                # 首先获取所有节点的嵌入
                all_embeddings = emb.squeeze(0)  # [num_nodes, embedding_dim]
                
                # 为所有节点生成logits
                all_logits = []
                for i in range(all_embeddings.shape[0]):
                    node_emb = all_embeddings[i:i+1, :].unsqueeze(0)  # [1, 1, embedding_dim]
                    f_1 = model.fc1(node_emb)
                    f_1 = model.act(f_1)
                    f_2 = model.fc2(f_1)
                    f_2 = model.act(f_2)
                    node_logit = model.fc3(f_2)
                    all_logits.append(node_logit.squeeze())
                
                all_logits = torch.stack(all_logits)  # [num_nodes]
                logits = all_logits[idx_test].cpu().detach().numpy()
                auc = roc_auc_score(ano_label[idx_test], logits)
                # print('Testing {} AUC:{:.4f}'.format(args.dataset, auc))
                AP = average_precision_score(ano_label[idx_test], logits, average='macro', pos_label=1, sample_weight=None)
                # print('Testing AP:', AP)
                wandb.log({"AUC": auc, "AP": AP}, step=epoch)
                
                # 检查是否为最佳模型
                if auc > best_AUC and AP > best_AP:
                    best_AUC = auc
                    best_AP = AP
                    best_model_state = model.state_dict().copy()
                    best_epoch = epoch

    print(f"Training done! Total time: {total_time:.2f} seconds")
    if args.visualize and args.model_type != 'SGT':
        # 加载最佳模型进行tsne可视化
        if best_model_state is not None:
            
            model.load_state_dict(best_model_state)
            model.eval()
            print("running the best model...")
            # 获取所有节点的嵌入
            train_flag = False
            emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, agg_attention_weights = model(concated_input_features, adj, 
                                                                                            normal_for_generation_idx, normal_for_train_idx,
                                                                                            train_flag, args)
            
            # 准备tsne数据
            # 获取原始特征（去掉batch维度）
            original_features = concated_input_features.squeeze(0)  # [num_nodes, feature_dim]
            
            # 获取嵌入（去掉batch维度）

            embeddings = emb.squeeze(0)  # [num_nodes, embedding_dim]
            
            # 获取真实标签（去掉batch维度）
            true_labels = labels.squeeze(0)  # [num_nodes]

            if outlier_emb is None:
                print("outlier_emb is None")
                outlier_emb_len = 0
            else:
                outlier_emb_len = len(outlier_emb)
            
            # 创建节点类型标签
            node_types = []
            for i in range(nb_nodes + outlier_emb_len):
                if i >= nb_nodes:
                    node_types.append("generated_anomaly")
                elif true_labels[i] == 1:
                    # 真实异常点
                    node_types.append("anomaly") 
                else:
                    node_types.append("normal")
            
            # 创建tsne可视化
            create_tsne_visualization(original_features, embeddings, true_labels, node_types, best_epoch, device,
                                    normal_for_train_idx, normal_for_generation_idx, outlier_emb)
            
            # 可视化注意力权重
            print("开始分析注意力权重...")
            # 获取邻接矩阵（去掉batch维度）
            adj_matrix_np = adj.squeeze(0).detach().cpu().numpy()
            attention_stats = visualize_attention_weights(agg_attention_weights, labels, normal_for_train_idx, 
                                                        normal_for_generation_idx, outlier_emb, best_epoch, 
                                                        args.dataset, device, adj_matrix_np)
            print("注意力权重分析完成！")
        
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset', type=str,
                        default='reddit')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
    parser.add_argument('--auc_test_rounds', type=int, default=256)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--var', type=float, default=0.0)
    parser.add_argument('--confidence_margin', type=float, default=0.7)
    
    parser.add_argument('--model_type', type=str, default='GGADFormer')
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--pp_k', type=int, default=2)
    parser.add_argument('--progregate_alpha', type=float, default=0.1)
    parser.add_argument('--sample_num_p', type=int, default=7)
    parser.add_argument('--sample_num_n', type=int, default=7)
    parser.add_argument('--sample_size', type=int, default=10000)

    parser.add_argument('--GT_ffn_dim', type=int, default=128)
    parser.add_argument('--GT_dropout', type=float, default=0.5)
    parser.add_argument('--GT_attention_dropout', type=float, default=0.5)
    parser.add_argument('--GT_num_heads', type=int, default=1)
    parser.add_argument('--GT_num_layers', type=int, default=3)

    parser.add_argument('--margin_loss_weight', type=float, default=1.0)
    parser.add_argument('--con_loss_weight', type=float, default=1.0)
    parser.add_argument('--con_loss_temp', type=float, default=10)



    args = parser.parse_args()

    if args.lr is None:
        if args.dataset in ['Amazon']:
            args.lr = 1e-3
        elif args.dataset in ['t_finance']:
            args.lr = 1e-3
        elif args.dataset in ['reddit']:
            args.lr = 1e-3
        elif args.dataset in ['photo']:
            args.lr = 1e-3
        elif args.dataset in ['elliptic']:
            args.lr = 1e-3

    if args.num_epoch is None:
        if args.dataset in ['photo']:
            args.num_epoch = 80
        if args.dataset in ['elliptic']:
            args.num_epoch = 150
        if args.dataset in ['reddit']:
            args.num_epoch = 300
        elif args.dataset in ['t_finance']:
            args.num_epoch = 500
        elif args.dataset in ['Amazon']:
            args.num_epoch = 800
    if args.dataset in ['reddit', 'photo']:
        args.mean = 0.02
        args.var = 0.01
    else:
        args.mean = 0.0
        args.var = 0.0


    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="HCCS",
        # Set the wandb project where this run will be logged.
        project="GGADFormer",
        # Track hyperparameters and run metadata.
        config=args,
    )

    wandb.define_metric("AUC", summary="max")
    wandb.define_metric("AP", summary="max")
    print('Dataset: ', args.dataset)
        
    try:
        train(args)
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"显存不足!：{e}")
        wandb.log({"AUC.max": 0})
        wandb.log({"AP.max": 0})
        wandb.finish()
    
    except Exception as e:
        import traceback
        print(f"其他错误：{e}")
        traceback.print_exc()  # 打印详细的错误堆栈，包括出错的代码行
        wandb.log({"AUC.max": 0})
        wandb.finish()
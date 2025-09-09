import torch.nn as nn

from model import Model
from GGADFormer import GGADFormer
from SGT import SGT
from RAGFormer import RAGFormer
from utils import *

from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time

import wandb
from visualization import create_tsne_visualization, visualize_attention_weights

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
    idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, 0.1, args=args)

    if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()


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

    if args.model_type == 'RAGFormer':
        concated_input_features = features.to(args.device)
        model = RAGFormer(ft_size, args.embedding_dim, 'prelu', args)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=int(0.1 * args.num_epoch),
        tot_updates=args.num_epoch,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )

    # 损失函数设置
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).to(device))
    xent = nn.CrossEntropyLoss()

    best_AUC = 0
    best_AP = 0
    best_model_state = None
    best_epoch = 0

    # Train model
    print(f"Start training! Total epochs: {args.num_epoch}")
    pbar = tqdm(total=args.num_epoch, desc='Training')
    total_time = 0
    for epoch in range(args.num_epoch):
        dynamic_weights = get_dynamic_loss_weights(epoch, args)
        start_time = time.time()
        model.train()
        optimizer.zero_grad()

        # Train model
        train_flag = True
        emb, anomaly_scores, reconstruction_loss = model(concated_input_features, adj,
                                                                normal_for_generation_idx, normal_for_train_idx,
                                                                train_flag, args)
        loss = reconstruction_loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        end_time = time.time()
        total_time += end_time - start_time
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新进度条信息
        postfix_dict = {
            'Time': f'{total_time:.1f}s',
            'Epoch': f'{epoch+1}/{args.num_epoch}'
        }
        
        pbar.update(1)
        pbar.set_postfix(postfix_dict)
        if epoch % 2 == 0:
            # print('Traininig {} AUC:{:.4f}'.format(args.dataset, auc))
            # AP = average_precision_score(lbl, logits, average='macro', pos_label=1, sample_weight=None)
            # print('Traininig AP:', AP)

            # print("Epoch:", '%04d' % (epoch), "train_loss_margin=", "{:.5f}".format(loss_margin.item()))
            # print("Epoch:", '%04d' % (epoch), "train_loss_bce=", "{:.5f}".format(loss_bce.item()))
            # print("Epoch:", '%04d' % (epoch), "rec_loss=", "{:.5f}".format(loss_rec.item()))
            # print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()))
            # print("=====================================================================")
            wandb.log({ 
                        "rec_loss": reconstruction_loss.item(),
                        "train_loss": loss.item(),
                        "learning_rate": current_lr}, step=epoch)
        if epoch % 10 == 0 and epoch != 0:
            model.eval()
            train_flag = False
            emb, anomaly_scores, reconstruction_loss = model(concated_input_features, adj, normal_for_generation_idx, normal_for_train_idx,
                                                                    train_flag, args)
            anomaly_scores = anomaly_scores[idx_test].cpu().detach().numpy()
            auc = roc_auc_score(ano_label[idx_test], anomaly_scores)
            # print('Testing {} AUC:{:.4f}'.format(args.dataset, auc))
            AP = average_precision_score(ano_label[idx_test], anomaly_scores, average='macro', pos_label=1, sample_weight=None)
            # print('Testing AP:', AP)
            wandb.log({"AUC": auc, "AP": AP}, step=epoch)
            
            # 添加AUC和AP到进度条后缀
            postfix_dict['AUC'] = f'{auc:.4f}'
            postfix_dict['AP'] = f'{AP:.4f}'
            pbar.set_postfix(postfix_dict)
            
            # 检查是否为最佳模型
            if auc > best_AUC and AP > best_AP:
                best_AUC = auc
                best_AP = AP
                best_model_state = model.state_dict().copy()
                best_epoch = epoch

    pbar.close()  # 关闭进度条
    print(f"Training done! Total time: {total_time:.2f} seconds")
    if args.visualize:
        # 加载最佳模型进行tsne可视化
        if best_model_state is not None:
            model.eval()
            # 为了获取人造异常点的嵌入，设置train_flag为True
            train_flag = True

            # 先运行最后一个 epoch 的模型
            emb_last_epoch, _, _ = model(concated_input_features, adj, 
                                                                                normal_for_generation_idx, normal_for_train_idx,
                                                                                train_flag, args)
            # 再运行最佳模型的模型
            model.load_state_dict(best_model_state)
            emb_best_epoch, _, _ = model(concated_input_features, adj, 
                                                                                            normal_for_generation_idx, normal_for_train_idx,
                                                                                            train_flag, args)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset', type=str,
                        default='reddit')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_split_seed', type=int, default=0)
    parser.add_argument('--train_rate', type=float, default=0.3)

    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--proj_dim', type=int, default=32)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
    parser.add_argument('--auc_test_rounds', type=int, default=256)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--var', type=float, default=0.0)
    parser.add_argument('--confidence_margin', type=float, default=0.7)
    parser.add_argument('--sample_rate', type=float, default=0.15)
    
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

    parser.add_argument('--rec_loss_weight', type=float, default=1.0)
    parser.add_argument('--bce_loss_weight', type=float, default=1.0)
    parser.add_argument('--margin_loss_weight', type=float, default=0)
    parser.add_argument('--con_loss_weight', type=float, default=1.0)
    parser.add_argument('--gna_loss_weight', type=float, default=1.0)
    
    parser.add_argument('--con_loss_temp', type=float, default=10)
    parser.add_argument('--GNA_temp', type=float, default=1)
    

    parser.add_argument('--warmup_updates', type=int, default=100)
    parser.add_argument('--tot_updates', type=int, default=1000)
    parser.add_argument('--peak_lr', type=float, default=1e-4)    
    parser.add_argument('--end_lr', type=float, default=0)

    parser.add_argument('--warmup_epoch', type=int, default=20)

    # RAGFormer
    parser.add_argument('--mask_ratio', type=float, default=0.1)
    parser.add_argument('--mask_edge', type=float, default=0.1)

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
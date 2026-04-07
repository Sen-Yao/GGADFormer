import time
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
from collections import Counter
import pandas as pd
import os
from tqdm import tqdm
import pickle
import datetime
import requests

from torch.optim.lr_scheduler import _LRScheduler

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset, train_rate=0.3, val_rate=0.1, args=None):

    """Load .mat dataset."""
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']


    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    # labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
    # num_classes = np.max(labels) + 1
    # labels = dense_to_one_hot(labels, num_classes)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    
    # 使用data_split_seed来控制数据集划分的随机性
    if args is not None and hasattr(args, 'data_split_seed'):
        # 保存当前的全局随机种子状态
        original_random_seed = random.getstate()
        original_np_random_seed = np.random.get_state()
        
        # 设置数据划分专用的随机种子
        random.seed(args.data_split_seed)
        np.random.seed(args.data_split_seed)
        
        random.shuffle(all_idx)
        
        # 恢复全局随机种子状态
        random.setstate(original_random_seed)
        np.random.set_state(original_np_random_seed)
    else:
        random.shuffle(all_idx)
    
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    # idx_test = all_idx[num_train:]
    print('Training', Counter(np.squeeze(ano_labels[idx_train])))
    print('Test', Counter(np.squeeze(ano_labels[idx_test])))
    # Sample some labeled normal nodes
    all_normal_label_idx = [i for i in idx_train if ano_labels[i] == 0]
    rate = 1  #  change train_rate to 0.3 0.5 0.6  0.8
    # normal_for_train_idx 为用于训练的正常的节点索引
    normal_for_train_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * rate)]
    print('Training rate', rate)

    # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.2)]
    # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.25)]
    # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.15)]
    # normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * 0.10)]

    # contamination
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # add_rate = 0.1 * len(real_abnormal_id)
    # random.shuffle(real_abnormal_id)
    # add_abnormal_id = real_abnormal_id[:int(add_rate)]
    # normal_label_idx = normal_label_idx + add_abnormal_id
    # idx_test = np.setdiff1d(idx_test, add_abnormal_id, False)

    # contamination 
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # add_rate = 0.05 * len(real_abnormal_id)  #0.05 0.1  0.15
    # remove_rate = 0.15 * len(real_abnormal_id)
    # random.shuffle(real_abnormal_id)
    # add_abnormal_id = real_abnormal_id[:int(add_rate)]
    # remove_abnormal_id = real_abnormal_id[:int(remove_rate)]
    # normal_label_idx = normal_label_idx + add_abnormal_id
    # idx_test = np.setdiff1d(idx_test, remove_abnormal_id, False)

    # camouflage
    # real_abnormal_id = np.array(all_idx)[np.argwhere(ano_labels == 1).squeeze()].tolist()
    # normal_feat = np.mean(feat[normal_label_idx], 0)
    # replace_rate = 0.05 * normal_feat.shape[1]
    # feat[real_abnormal_id, :int(replace_rate)] = normal_feat[:, :int(replace_rate)]

    # random.shuffle(normal_for_train_idx)
    # 0.05 for Amazon and 0.15 for other datasets

    # 选择一部分正常节点用于生成异常节点
    # normal_for_generation_idx 为用于生成异常节点的正常节点索引
    if dataset in ['Amazon']:
        normal_for_generation_idx = normal_for_train_idx[: int(len(normal_for_train_idx) * args.sample_rate)]  
    else:
        normal_for_generation_idx = normal_for_train_idx[: int(len(normal_for_train_idx) * args.sample_rate)]  
    return adj, feat, ano_labels, all_idx, idx_train, idx_val, idx_test, ano_labels, None, None, normal_for_train_idx, normal_for_generation_idx

def load_dgraph(prefix='./dataset/', train_rate=0.3, val_rate=0.1, args=None):
    print("Loading DGraph, this may take a while...")
    f = np.load(prefix + 'dgraphfin.npz')
    label = torch.tensor(f['y']).float()
    label = (label == 1).int().unsqueeze(0).numpy()  # shape: (1, N)
    ano_labels = np.squeeze(np.array(label))
    attr = torch.tensor(f['x']).float()      # shape: (N, D)

    with open(prefix + 'dgraphfin_adj_list', 'rb') as file:
        adj_list = pickle.load(file)

    N = len(adj_list)
    assert set(adj_list.keys()) == set(range(N)), "Keys must be 0 to N-1"

    # 构建原始邻接矩阵（可能不对称）
    row = []
    col = []
    for i in range(N):
        for j in adj_list[i]:
            if 0 <= j < N:
                row.append(i)
                col.append(j)

    data = np.ones(len(row), dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(N, N))

    # 强制对称化
    adj = adj + adj.T
    adj.data[:] = 1.0  # 保持无权

    # 添加自环
    adj = adj + sp.eye(N, format='csr')

    # 对称归一化：\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}
    degrees = np.array(adj.sum(1)).flatten()  # \tilde{D}_{ii} = sum_j \tilde{A}_{ij}
    deg_inv_sqrt = np.power(degrees, -0.5)
    deg_inv_sqrt[np.isinf(degrees)] = 0.0  # 处理孤立节点（理论上不会出现，因有自环）
    
    D_inv_sqrt = sp.diags(deg_inv_sqrt)  # 对角矩阵
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt  # 仍是 CSR

    # 转为 PyTorch COO
    adj_norm_coo = adj_norm.tocoo()
    indices = torch.LongTensor(np.vstack((adj_norm_coo.row, adj_norm_coo.col)))
    values = torch.FloatTensor(adj_norm_coo.data)
    adj_coo = torch.sparse_coo_tensor(indices, values, adj_norm.shape)

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    
    # 使用data_split_seed来控制数据集划分的随机性
    if args is not None and hasattr(args, 'data_split_seed'):
        # 保存当前的全局随机种子状态
        original_random_seed = random.getstate()
        original_np_random_seed = np.random.get_state()
        
        # 设置数据划分专用的随机种子
        random.seed(args.data_split_seed)
        np.random.seed(args.data_split_seed)
        
        random.shuffle(all_idx)
        
        # 恢复全局随机种子状态
        random.setstate(original_random_seed)
        np.random.set_state(original_np_random_seed)
    else:
        random.shuffle(all_idx)
    
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    # idx_test = all_idx[num_train:]
    print('Training', Counter(np.squeeze(ano_labels[idx_train])))
    print('Test', Counter(np.squeeze(ano_labels[idx_test])))
    # Sample some labeled normal nodes
    all_normal_label_idx = [i for i in idx_train if ano_labels[i] == 0]
    rate = 1  #  change train_rate to 0.3 0.5 0.6  0.8
    # normal_for_train_idx 为用于训练的正常的节点索引
    normal_for_train_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * rate)]
    print('Training rate', rate)

    # 选择一部分正常节点用于生成异常节点
    # normal_for_generation_idx 为用于生成异常节点的正常节点索引
    normal_for_generation_idx = normal_for_train_idx[: int(len(normal_for_train_idx) * args.sample_rate)]  
    return adj_coo, attr, label, all_idx, idx_train, idx_val, idx_test, ano_labels, None, None, normal_for_train_idx, normal_for_generation_idx

def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1,
                                                           max_nodes_per_seed=subgraph_size * 3)
    subv = []

    for i, trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace), sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
                                                                      max_nodes_per_seed=subgraph_size * 5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size * 3]
        subv[i].append(i)

    return subv

def node_neighborhood_feature(adj, features, k, alpha=0.1):

    x_0 = features
    for i in range(k):
        # print(f"features.shape: {features.shape}, adj.shape: {adj.shape}")
        if isinstance(adj, torch.Tensor):
            features = (1-alpha) * torch.mm(adj, features) + alpha * x_0
        else:
            # sparse adj, maybe DGraph
            features = (1 - alpha) * torch.spmm(adj, features) + alpha * x_0

    return features

import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm

matplotlib.use('Agg')
plt.rcParams['figure.dpi'] = 300  # 图片像素
plt.rcParams['figure.figsize'] = (8.5, 7.5)
# plt.rcParams['figure.figsize'] = (10.5, 9.5)
from matplotlib.backends.backend_pdf import PdfPages


def draw_pdf(message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0 = np.mean(message_all[0])  # 计算均值
    sigma_0 = np.std(message_all[0])
    # print('The mean of normal {}'.format(mu_0))
    # print('The std of normal {}'.format(sigma_0))
    mu_1 = np.mean(message_all[1])  # 计算均值
    sigma_1 = np.std(message_all[1])
    # print('The mean of abnormal {}'.format(mu_1))
    # print('The std of abnormal {}'.format(sigma_1))
    mu_2 = np.mean(message_all[2])  # 计算均值
    sigma_2 = np.std(message_all[2])
    # print('The mean of abnormal {}'.format(mu_2))
    # print('The std of abnormal {}'.format(sigma_2))
    n, bins, patches = plt.hist(message_all, bins=30, density=True, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = norm.pdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
    y_1 = norm.pdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
    y_2 = norm.pdf(bins, mu_2, sigma_2)  # 拟合一条最佳正态分布曲线y
    # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # 绘制y的曲线
    # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # 绘制y的曲线
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.ylim(0, 20)

    # plt.xlabel('RAW-based Affinity', fontsize=25)
    # plt.xlabel('TAM-based Affinity', fontsize=25)
    # plt.ylabel('Number of Samples', size=25)
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    # from matplotlib.pyplot import MultipleLocator
    # x_major_locator = MultipleLocator(0.02)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.legend(loc='upper left', fontsize=30)
    # plt.title('Amazon'.format(dataset), fontsize=25)
    # plt.title('BlogCatalog', fontsize=50)
    plt.savefig('fig/{}/{}_{}.pdf'.format(dataset, dataset, epoch))
    plt.close()


def draw_pdf_methods(method, message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0 = np.mean(message_all[0])  # 计算均值
    sigma_0 = np.std(message_all[0])
    # print('The mean of normal {}'.format(mu_0))
    # print('The std of normal {}'.format(sigma_0))
    mu_1 = np.mean(message_all[1])  # 计算均值
    sigma_1 = np.std(message_all[1])
    # print('The mean of abnormal {}'.format(mu_1))
    # print('The std of abnormal {}'.format(sigma_1))
    mu_2 = np.mean(message_all[2])  # 计算均值
    sigma_2 = np.std(message_all[2])
    # print('The mean of abnormal {}'.format(mu_2))
    # print('The std of abnormal {}'.format(sigma_2))

    n, bins, patches = plt.hist(message_all, bins=30, density=True, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = norm.pdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
    y_1 = norm.pdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
    y_2 = norm.pdf(bins, mu_2, sigma_2)  # 拟合一条最佳正态分布曲线y
    # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # 绘制y的曲线
    # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # 绘制y的曲线
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.ylim(0, 8)

    # plt.xlabel('RAW-based Affinity', fontsize=25)
    # plt.xlabel('TAM-based Affinity', fontsize=25)
    # plt.ylabel('Number of Samples', size=25)

    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    # plt.legend(loc='upper left', fontsize=30)
    # plt.title('Amazon'.format(dataset), fontsize=25)
    # plt.title('BlogCatalog', fontsize=50)
    plt.savefig('fig/{}/{}2/{}_{}.svg'.format(method, dataset, dataset, epoch))
    plt.close()



def node_seq_feature(features, pk, nk, sample_batch):
    """
    跟据特征矩阵采样正负样本

    Args:
        features: 特征矩阵, size = (N, d)
        pk: 采样正样本的个数
        nk: 采样负样本的个数
        sample_batch: 每次采样的batch大小
    Returns:
        nodes_features: 采样之后的特征矩阵, size = (N, 1, K+1, d)
    """

    nodes_features_p = torch.empty(features.shape[0], pk+1, features.shape[1])

    nodes_features_n = torch.empty(features.shape[0], nk+1, features.shape[1])

    x = features + torch.zeros_like(features)
    
    x = torch.nn.functional.normalize(x, dim=1)

    # 构建 batch 采样
    total_batch = int(features.shape[0]/sample_batch)

    rest_batch = int(features.shape[0]%sample_batch)

    for index_batch in tqdm(range(total_batch)):

        # x_batch, [b,d]
        # x1_batch, [b,d]
        # 切片操作左闭右开
        x_batch = x[(index_batch)*sample_batch:(index_batch+1)*sample_batch,:]

        
        s = torch.matmul(x_batch, x.transpose(1, 0))


        # Begin sampling positive samples
        # print(s.shape)
        for i in range(sample_batch):
            s[i][(index_batch)*(sample_batch) + i] = -1000        #将得到的相似度矩阵的对角线的值置为负的最大值

        topk_values, topk_indices = torch.topk(s, pk, dim=1)

        for index in range(sample_batch):

            nodes_features_p[(index_batch)*sample_batch + index, 0, :] = features[(index_batch)*sample_batch + index]
            for i in range(pk):
                nodes_features_p[(index_batch)*sample_batch + index, i+1, :] = features[topk_indices[index][i]]

        # Begin sampling positive samples
        if nk > 0:
            all_idx = [i for i in range(s.shape[1])]

            for index in tqdm(range(sample_batch)):

                nce_idx = list(set(all_idx) - set(topk_indices[index].tolist()))
                

                nce_indices = np.random.choice(nce_idx, nk, replace=True)

                nodes_features_n[(index_batch)*sample_batch + index, 0, :] = features[(index_batch)*sample_batch + index]
                for i in range(nk):
                    nodes_features_n[(index_batch)*sample_batch + index, i+1, :] = features[nce_indices[i]]

    if rest_batch > 0:

        x_batch = x[(total_batch)*sample_batch:(total_batch)*sample_batch + rest_batch,:]
        x = x
        # print(f"x_batch.shape: {x_batch.shape}, x.shape: {x.shape}")

        s = torch.matmul(x_batch, x.transpose(1, 0))

        print("------------begin sampling positive samples------------")

        #采正样本
        for i in range(rest_batch):
            s[i][(total_batch)*sample_batch + i] = -1000         #将得到的相似度矩阵的对角线的值置为负的最大值

        topk_values, topk_indices = torch.topk(s, pk, dim=1)
        # print(topk_indices.shape)

        for index in range(rest_batch):
            nodes_features_p[(total_batch)*sample_batch + index, 0, :] = features[(total_batch)*sample_batch + index]
            for i in range(pk):
                nodes_features_p[(total_batch)*sample_batch + index, i+1, :] = features[topk_indices[index][i]]


        print("------------begin sampling negative samples------------")

        #采负样本
        if nk > 0:
            all_idx = [i for i in range(s.shape[1])]

            for index in tqdm(range(rest_batch)):

                nce_idx = list(set(all_idx) - set(topk_indices[index].tolist()))
                

                nce_indices = np.random.choice(nce_idx, nk, replace=False)
                # print(nce_indices)

                # print((index_batch)*sample_batch + index)
                # print(nce_indices)


                nodes_features_n[(total_batch)*sample_batch + index, 0, :] = features[(total_batch)*sample_batch + index]
                for i in range(nk):
                    nodes_features_n[(total_batch)*sample_batch + index, i+1, :] = features[nce_indices[i]]


    nodes_features = torch.concat((nodes_features_p, nodes_features_n), dim=1)
    

    # print(nodes_features_p.shape)
    # print(nodes_features_n.shape)
    # print(nodes_features.shape)

    return nodes_features

def preprocess_sample_features(args, features, adj):
    """
    基于节点序列采样方法，准备预处理特征矩阵
    Args:
        args: 输入的训练参数
        features: 特征矩阵, size = (N, d)
    Returns:
        features: 预处理后的特征矩阵, size = (N, args.sample_num_p+1 + args.sample_num_n+1, d)
    """
    # 检查./pretrain目录是否存在，不存在则创建
    pretrain_dir = './pretrain'
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)

    data_file = './pretrain/pre_sample/'+args.dataset +'_'+str(args.sample_num_p)+'_'+str(args.sample_num_n)+"_"+str(args.pp_k)+'.pt'
    if os.path.isfile(data_file):
        processed_features = torch.load(data_file, map_location='cpu')

    else:
        processed_features = node_seq_feature(features, args.sample_num_p, args.sample_num_n, args.sample_size)  # return (N, hops+1, d)

        if args.pp_k > 0:

            data_file_ppr = './pretrain/pre_features'+args.dataset +'_'+str(args.pp_k)+'.pt'

            if os.path.isfile(data_file_ppr):
                ppr_features = torch.load(data_file_ppr, map_location='cpu')

            else:
                ppr_features = node_neighborhood_feature(adj, features, args.pp_k, args.progregate_alpha)  # return (N, d)
                # store the data 
                torch.save(ppr_features, data_file_ppr)

            ppr_processed_features = node_seq_feature(ppr_features, args.sample_num_p, args.sample_num_n, args.sample_size)

            processed_features = torch.concat((processed_features, ppr_processed_features), dim=1)

        # store the data
        # 检查父目录是否存在，如果不存在则创建
        if not os.path.exists(os.path.dirname(data_file)):
            os.makedirs(os.path.dirname(data_file))
        torch.save(processed_features, data_file)
    # return (N, sample_num_p+1 + args.sample_num_n+1, d)
    return processed_features

def nagphormer_tokenization(features, adj, args):
    """
    基于Nagphormer的tokenization方法，准备预处理特征矩阵
    Args:
        features: 特征矩阵, size = (N, d)
        adj: 邻接矩阵, size = (N, N)
    Returns:
        features: 预处理后的特征矩阵, size = (N, args.pp_k+1, d)
    """
    print("Tokenizating")
    start_time = time.time()

    nodes_features = features.unsqueeze(1)
    for hop in range(args.pp_k):

        steped_nodes_features = node_neighborhood_feature(adj, features, hop+1, args.progregate_alpha)
        nodes_features = torch.concat((nodes_features, steped_nodes_features.unsqueeze(1)), dim=1)
    print(f"Tokenization time: {time.time() - start_time:.4f}s")
    
    # 计算并输出 nodes_features 空间开销
    num_elements = nodes_features.shape[0] * nodes_features.shape[1] * nodes_features.shape[2]
    memory_bytes = num_elements * 4  # float32 = 4 bytes
    if memory_bytes >= 1024**3:
        memory_str = f"{memory_bytes / 1024**3:.2f} GB"
    elif memory_bytes >= 1024**2:
        memory_str = f"{memory_bytes / 1024**2:.2f} MB"
    else:
        memory_str = f"{memory_bytes / 1024:.2f} KB"
    print(f"nodes_features shape: {nodes_features.shape}, memory usage: {memory_str}")
    
    return nodes_features

class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False


def get_dynamic_loss_weights(epoch, args):
    """
    根据当前epoch和warmup_epoch计算动态损失权重
    
    Args:
        epoch: 当前epoch
        args: 参数对象，包含各种损失权重的目标值
    
    Returns:
        dict: 包含各种损失权重的字典
    """
    if epoch < args.warmup_epoch:
        # warmup阶段：只开启community_loss和正常节点内部的对比损失
        return {
            'margin_loss_weight': args.margin_loss_weight,
            'bce_loss_weight': args.bce_loss_weight,
            'rec_loss_weight': args.rec_loss_weight,
            'con_loss_weight': args.con_loss_weight,
            'proj_loss_weight': 0.0,
            'reconstruction_loss_weight': args.reconstruction_loss_weight,
            'ring_loss_weight': args.ring_loss_weight
        }
    else:
        # 超过warmup后，使用线性插值平滑地恢复到目标值
        progress = min(1.0, (epoch - args.warmup_epoch) / args.warmup_epoch)  # 在warmup_epoch个epoch内平滑过渡
        
        return {
            'margin_loss_weight': args.margin_loss_weight,
            'bce_loss_weight': args.bce_loss_weight,
            'rec_loss_weight': args.rec_loss_weight,
            'con_loss_weight': args.con_loss_weight,
            'proj_loss_weight': progress * args.proj_loss_weight,
            'reconstruction_loss_weight': args.reconstruction_loss_weight,
            'ring_loss_weight': args.ring_loss_weight
        }
    

def send_notification(content):  
    print(f"发送通知: {content}")  
    payload = {"text": content, "timestamp": str(datetime.now())} 
    try:  
        requests.post(os.environ['WANDB_NOTIFY_URL'], json=payload, timeout=10)  
    except Exception as e:  
        print(f"发送通知失败: {e}")  


# 全局变量用于并行BFS计算（避免pickle问题）
_adj_sparse_global = None


def _bfs_from_source_global(source):
    """从单个源节点执行BFS，返回到所有可达节点的距离（全局函数，用于并行计算）"""
    distances = {}
    visited = set([source])
    queue = [(source, 0)]
    
    while queue:
        node, dist = queue.pop(0)
        # 获取邻居
        neighbors = _adj_sparse_global[node].nonzero()[1]
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                new_dist = dist + 1
                distances[neighbor] = new_dist
                queue.append((neighbor, new_dist))
    
    return list(distances.values())


def _init_worker(adj):
    """初始化worker进程，设置全局邻接矩阵"""
    global _adj_sparse_global
    _adj_sparse_global = adj


def calculate_graph_statistics(adj, n_samples=1000, n_workers=None):
    """
    计算图的平均最短路径和有效直径（基于采样估算）
    
    参数:
        adj: 邻接矩阵（scipy sparse matrix, torch tensor, 或 numpy array）
        n_samples: 采样源节点数量（默认1000，如果节点数少则使用全部节点）
        n_workers: 并行工作进程数（默认使用所有CPU核心）
    
    返回:
        avg_shortest_path: 平均最短路径长度（仅统计可达节点对）
        effective_diameter: 有效直径（90%可达节点对的最大距离）
    """
    from multiprocessing import Pool, cpu_count
    from scipy.sparse import csr_matrix, issparse
    import warnings
    
    # 转换为 scipy sparse matrix
    if isinstance(adj, torch.Tensor):
        adj_np = adj.detach().cpu().numpy()
        if adj_np.ndim == 3:
            adj_np = adj_np.squeeze(0)  # 移除 batch 维度
        adj_sparse = csr_matrix(adj_np)
    elif issparse(adj):
        adj_sparse = adj if adj.format == 'csr' else adj.tocsr()
    else:
        adj_sparse = csr_matrix(adj)
    
    n_nodes = adj_sparse.shape[0]
    
    # 确保邻接矩阵是无向的（取对称）
    adj_sparse = adj_sparse.maximum(adj_sparse.transpose())
    
    # 移除自环
    adj_sparse.setdiag(0)
    adj_sparse.eliminate_zeros()
    
    # 检查边数
    n_edges = adj_sparse.nnz // 2  # 无向图边数
    print(f"图统计: 节点数={n_nodes}, 边数={n_edges}")
    
    if n_edges == 0:
        warnings.warn("图没有边，无法计算最短路径统计量")
        return float('inf'), float('inf')
    
    # 确定采样节点数
    n_samples = min(n_samples, n_nodes)
    
    # 随机采样源节点
    np.random.seed(42)  # 可重复性
    sampled_nodes = np.random.choice(n_nodes, size=n_samples, replace=False)
    
    # 并行或串行执行BFS
    if n_workers is None:
        n_workers = cpu_count()
    
    print(f"开始计算最短路径统计量（采样 {n_samples} 个节点，{n_workers} 个并行进程）...")
    start_time = time.time()
    
    # 设置全局邻接矩阵（用于串行计算）
    global _adj_sparse_global
    _adj_sparse_global = adj_sparse
    
    all_distances = []
    if n_workers > 1 and n_samples > 10:
        # 并行计算 - 使用全局函数和初始化函数
        with Pool(processes=n_workers, initializer=_init_worker, initargs=(adj_sparse,)) as pool:
            results = pool.map(_bfs_from_source_global, sampled_nodes)
            for dists in results:
                all_distances.extend(dists)
    else:
        # 串行计算（小图或单进程）
        for source in tqdm(sampled_nodes, desc="计算最短路径"):
            dists = _bfs_from_source_global(source)
            all_distances.extend(dists)
    
    elapsed = time.time() - start_time
    
    if len(all_distances) == 0:
        warnings.warn("图可能不连通，无法计算最短路径统计量")
        return float('inf'), float('inf')
    
    all_distances = np.array(all_distances)
    
    # 计算平均最短路径
    avg_shortest_path = np.mean(all_distances)
    
    # 计算有效直径（90%分位数）
    effective_diameter = np.percentile(all_distances, 90)
    
    print(f"图统计计算完成，耗时 {elapsed:.2f}s")
    print(f"  - 采样节点对数: {len(all_distances)}")
    print(f"  - 平均最短路径: {avg_shortest_path:.4f}")
    print(f"  - 有效直径 (90%): {effective_diameter:.4f}")
    
    return avg_shortest_path, effective_diameter

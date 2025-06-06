import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
from collections import Counter

import os
from tqdm import tqdm
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


def load_mat(dataset, train_rate=0.3, val_rate=0.1):

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
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    # idx_test = all_idx[num_train:]
    print('Training', Counter(np.squeeze(ano_labels[idx_train])))
    print('Test', Counter(np.squeeze(ano_labels[idx_test])))
    # Sample some labeled normal nodes
    all_normal_label_idx = [i for i in idx_train if ano_labels[i] == 0]
    rate = 0.5  #  change train_rate to 0.3 0.5 0.6  0.8
    normal_label_idx = all_normal_label_idx[: int(len(all_normal_label_idx) * rate)]
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

    random.shuffle(normal_label_idx)
    # 0.05 for Amazon and 0.15 for other datasets
    if dataset in ['Amazon']:
        abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * 0.05)]  
    else:
        abnormal_label_idx = normal_label_idx[: int(len(normal_label_idx) * 0.15)]  
    return adj, feat, ano_labels, all_idx, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels, normal_label_idx, abnormal_label_idx


def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_array(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph


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


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

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
    n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # 拟合一条最佳正态分布曲线y
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

    n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # 拟合一条最佳正态分布曲线y
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


def node_neighborhood_feature(adj, features, k):

    x_0 = features
    for i in range(k):
        # print(f"features.shape: {features.shape}, adj.shape: {adj.shape}")
        features = 0.9 * torch.mm(adj, features) + 0.1 * x_0

    return features

def preprocess_sample_features(args, features, adj):
    """
    基于节点序列采样方法，准备预处理特征矩阵
    Args:
        args: 输入的训练参数
        features: 特征矩阵, size = (N, d)
    Returns:
        features: 预处理后的特征矩阵, size = (N, args.sample_num_p+1 + args.sample_num_n+1, d)
    """
    data_file = './pre_sample/'+args.dataset +'_'+str(args.sample_num_p)+'_'+str(args.sample_num_n)+"_"+str(args.pp_k)+'.pt'
    if os.path.isfile(data_file):
        processed_features = torch.load(data_file)

    else:
        processed_features = node_seq_feature(features, args.sample_num_p, args.sample_num_n, args.sample_size)  # return (N, hops+1, d)

        if args.pp_k > 0:

            data_file_ppr = './pre_features'+args.dataset +'_'+str(args.pp_k)+'.pt'

            if os.path.isfile(data_file_ppr):
                ppr_features = torch.load(data_file_ppr)

            else:
                ppr_features = node_neighborhood_feature(adj, features, args.pp_k)  # return (N, d)
                # store the data 
                torch.save(ppr_features, data_file_ppr)

            ppr_processed_features = node_seq_feature(ppr_features, args.sample_num_p, args.sample_num_n, args.sample_size)

            processed_features = torch.concat((processed_features, ppr_processed_features), dim=1)

        # store the data
        # 检查父目录是否存在，如果不存在则创建
        if not os.path.exists(os.path.dirname(data_file)):
            os.makedirs(os.path.dirname(data_file))
        torch.save(processed_features, data_file)
    return processed_features

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
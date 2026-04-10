"""
重构误差过滤消融实验工具函数库

功能：在评估DOMINANT和AnomalyDAE模型时，将测试集节点按重构误差排序，
     只选择重构误差最小的一定比例节点参与AUROC/AUPRC计算。

目的：验证当重构误差较小时，模型能否保持高质量的异常检测能力。

使用方法：在 dominant.py 或 anomalyDAE.py 中通过 --rec_error_filter_ratio 参数调用
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_rec_error_dominant(model, features, adj, idx, device):
    """
    计算DOMINANT模型每个节点的属性重构误差
    
    Args:
        model: DOMINANT模型
        features: 节点特征 [1, N, feat_dim]
        adj: 邻接矩阵 [1, N, N]
        idx: 需要计算重构误差的节点索引
        device: 设备
        
    Returns:
        rec_errors: 每个节点的属性重构误差 [len(idx)]
    """
    from model_domaint import neighList_to_edgeList
    
    model.eval()
    with torch.no_grad():
        adj_sq = torch.squeeze(adj)
        features_sq = torch.squeeze(features)
        edge_index = neighList_to_edgeList(adj_sq)
        edge_index = torch.tensor(np.array(edge_index)).T
        if features.device != torch.device('cpu'):
            edge_index = edge_index.cuda()
        
        # 获取重构特征
        x_, s_ = model.model_enc(features_sq, edge_index)
        
        # 计算属性重构误差
        diff_attr = torch.pow(features_sq[idx, :] - x_[idx, :], 2)
        attr_error = torch.sqrt(torch.sum(diff_attr, 1))
        
    return attr_error.cpu().numpy()


def compute_rec_error_anomalydae(model, features, adj, idx, device, weight=0.5):
    """
    计算AnomalyDAE模型每个节点的属性+结构重构误差
    
    Args:
        model: AnomalyDAE模型
        features: 节点特征 [1, N, feat_dim]
        adj: 邻接矩阵 [1, N, N]
        idx: 需要计算重构误差的节点索引
        device: 设备
        weight: 属性重构误差的权重 (1-weight为结构重构误差权重)
        
    Returns:
        rec_errors: 每个节点的综合重构误差 [len(idx)]
        attr_errors: 每个节点的属性重构误差 [len(idx)]
        stru_errors: 每个节点的结构重构误差 [len(idx)]
    """
    from model_AnomalyDAE import neighList_to_edgeList
    
    model.eval()
    with torch.no_grad():
        adj_sq = torch.squeeze(adj)
        features_sq = torch.squeeze(features)
        edge_index = neighList_to_edgeList(adj_sq)
        edge_index = torch.tensor(np.array(edge_index)).T
        if features.device != torch.device('cpu'):
            edge_index = edge_index.cuda()
        
        # 获取重构特征和邻接矩阵
        x_, s_ = model.model_enc(features_sq, edge_index)
        
        # 计算属性重构误差
        diff_attr = torch.pow(features_sq[idx, :] - x_[idx, :], 2)
        attr_error = torch.sqrt(torch.sum(diff_attr, 1))
        
        # 计算结构重构误差
        diff_stru = torch.pow(adj_sq[idx, :] - s_[idx, :], 2)
        stru_error = torch.sqrt(torch.sum(diff_stru, 1))
        
        # 综合重构误差
        combined_error = weight * attr_error + (1 - weight) * stru_error
        
    return combined_error.cpu().numpy(), attr_error.cpu().numpy(), stru_error.cpu().numpy()


def evaluate_with_rec_error_filter(score, rec_errors, labels, rec_error_filter_ratio=1.0):
    """
    带重构误差过滤的评估函数
    
    Args:
        score: 模型输出的异常分数 [N]
        rec_errors: 重构误差 [N]
        labels: 标签 [N]
        rec_error_filter_ratio: 重构误差过滤比例，选择重构误差最小的该比例节点
        
    Returns:
        results: 包含AUROC、AUPRC及详细统计信息的字典
    """
    # 转换为numpy处理
    if torch.is_tensor(score):
        score = score.detach().cpu().numpy()
    if torch.is_tensor(rec_errors):
        rec_errors = rec_errors.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    score = np.array(score)
    rec_errors = np.array(rec_errors)
    labels = np.array(labels)
    
    # 按重构误差排序（升序，选择重构误差最小的节点）
    sorted_indices = np.argsort(rec_errors)
    
    # 计算保留的节点数量
    total_nodes = len(sorted_indices)
    keep_count = int(total_nodes * rec_error_filter_ratio)
    keep_indices = sorted_indices[:keep_count]
    
    # 过滤后的score和labels
    filtered_score = score[keep_indices]
    filtered_labels = labels[keep_indices]
    
    # 统计信息
    normal_count = np.sum(filtered_labels == 0)
    anomaly_count = np.sum(filtered_labels == 1)
    
    # 计算AUROC和AUPRC
    if len(np.unique(filtered_labels)) > 1:
        auroc = roc_auc_score(filtered_labels, filtered_score)
        auprc = average_precision_score(filtered_labels, filtered_score, average='macro')
    else:
        auroc = 0.0
        auprc = 0.0
        print(f"Warning: 过滤后只有单一标签类别，无法计算AUROC/AUPRC")
    
    # 重构误差统计
    filtered_rec_errors = rec_errors[keep_indices]
    rec_error_mean = np.mean(filtered_rec_errors)
    rec_error_std = np.std(filtered_rec_errors)
    rec_error_min = np.min(filtered_rec_errors)
    rec_error_max = np.max(filtered_rec_errors)
    
    # 正常节点和异常节点的重构误差分别统计
    normal_rec_errors = rec_errors[keep_indices][filtered_labels == 0]
    anomaly_rec_errors = rec_errors[keep_indices][filtered_labels == 1]
    
    normal_rec_mean = np.mean(normal_rec_errors) if len(normal_rec_errors) > 0 else 0.0
    anomaly_rec_mean = np.mean(anomaly_rec_errors) if len(anomaly_rec_errors) > 0 else 0.0
    
    results = {
        'auroc': auroc,
        'auprc': auprc,
        'rec_error_filter_ratio': rec_error_filter_ratio,
        'total_nodes': total_nodes,
        'filtered_nodes': keep_count,
        'filtered_normal_count': normal_count,
        'filtered_anomaly_count': anomaly_count,
        'rec_error_mean': rec_error_mean,
        'rec_error_std': rec_error_std,
        'rec_error_min': rec_error_min,
        'rec_error_max': rec_error_max,
        'normal_rec_error_mean': normal_rec_mean,
        'anomaly_rec_error_mean': anomaly_rec_mean,
        'anomaly_ratio_in_filtered': anomaly_count / keep_count if keep_count > 0 else 0.0,
    }
    
    return results


def print_rec_error_filter_results(results, epoch=None):
    """
    打印重构误差过滤评估结果
    
    Args:
        results: evaluate_with_rec_error_filter返回的结果字典
        epoch: 当前epoch（可选）
    """
    epoch_str = f"Epoch {epoch}: " if epoch is not None else ""
    print(f"{epoch_str}Rec Error Filter Ratio: {results['rec_error_filter_ratio']:.2%}")
    print(f"  Filtered Nodes: {results['filtered_nodes']}/{results['total_nodes']}")
    print(f"  Normal/Anomaly: {results['filtered_normal_count']}/{results['filtered_anomaly_count']}")
    print(f"  AUROC: {results['auroc']:.4f}, AUPRC: {results['auprc']:.4f}")
    print(f"  Rec Error Mean: {results['rec_error_mean']:.4f} ± {results['rec_error_std']:.4f}")
    print(f"  Normal Rec Error: {results['normal_rec_error_mean']:.4f}, Anomaly Rec Error: {results['anomaly_rec_error_mean']:.4f}")
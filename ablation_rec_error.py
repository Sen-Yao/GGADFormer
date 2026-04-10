"""
重构误差过滤消融实验工具函数库

功能：在评估时，将测试集节点按重构误差排序，
     只选择重构误差最小的一定比例节点参与AUROC/AUPRC计算。

目的：验证当重构误差较小时，GGADFormer能否保持高质量的异常检测能力。

使用方法：在 run.py 中通过 --rec_error_filter_ratio 参数调用
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_node_rec_error(model, input_tokens, args, device):
    """
    计算每个节点的重构误差
    
    Args:
        model: GGADFormer模型
        input_tokens: 输入tokens [N, pp_k+1, feature_dim]
        args: 参数配置
        device: 设备
        
    Returns:
        rec_errors: 每个节点的重构误差 [N]
    """
    n_in = model.n_in
    
    with torch.no_grad():
        # 获取模型编码
        emb = model.TransformerEncoder(input_tokens)  # [1, N, embedding_dim]
        
        # 重构tokens
        reconstructed_tokens = model.token_decoder(emb).squeeze(0)  # [N, (pp_k+1)*n_in]
        
        # 计算重构误差（L2范数）
        original_tokens_flat = input_tokens.view(-1, (args.pp_k+1) * n_in)  # [N, (pp_k+1)*n_in]
        reconstruction_error = reconstructed_tokens - original_tokens_flat  # [N, (pp_k+1)*n_in]
        
        # 每个节点的重构误差（L2范数）
        rec_errors = torch.norm(reconstruction_error, p=2, dim=1)  # [N]
    
    return rec_errors


def evaluate_with_rec_error_filter(model, test_data_loader, ano_label, idx_test, 
                                   args, device, rec_error_filter_ratio=1.0):
    """
    带重构误差过滤的评估函数
    
    Args:
        model: 模型
        test_data_loader: 测试数据加载器
        ano_label: 全图异常标签
        idx_test: 测试集索引
        args: 参数配置
        device: 设备
        rec_error_filter_ratio: 重构误差过滤比例，选择重构误差最小的该比例节点
        
    Returns:
        results: 包含AUROC、AUPRC及详细统计信息的字典
    """
    model.eval()
    
    all_logits = []
    all_rec_errors = []
    
    with torch.no_grad():
        for batch_idx, item in enumerate(test_data_loader):
            input_tokens = item[0].to(device)
            labels = item[1].to(device)
            
            # 获取模型输出
            emb, emb_combine, logits, _, _, _, _ = model(
                input_tokens, None, None, None, False, args
            )
            
            # 计算每个节点的重构误差
            rec_errors = compute_node_rec_error(model, input_tokens, args, device)
            
            # 收集结果
            all_logits.append(logits.squeeze(0).cpu())
            all_rec_errors.append(rec_errors.cpu())
    
    # 合并所有batch的结果
    all_logits = torch.cat(all_logits, dim=0)  # [N_test]
    all_rec_errors = torch.cat(all_rec_errors, dim=0)  # [N_test]
    
    # 获取测试集标签
    test_labels = ano_label[idx_test]
    
    # 转换为numpy处理
    logits_np = all_logits.numpy()
    rec_errors_np = all_rec_errors.numpy()
    labels_np = test_labels
    
    # 按重构误差排序（升序，选择重构误差最小的节点）
    sorted_indices = np.argsort(rec_errors_np)
    
    # 计算保留的节点数量
    total_nodes = len(sorted_indices)
    keep_count = int(total_nodes * rec_error_filter_ratio)
    keep_indices = sorted_indices[:keep_count]
    
    # 过滤后的logits和labels
    filtered_logits = logits_np[keep_indices]
    filtered_labels = labels_np[keep_indices]
    
    # 统计信息
    normal_count = np.sum(filtered_labels == 0)
    anomaly_count = np.sum(filtered_labels == 1)
    
    # 计算AUROC和AUPRC
    if len(np.unique(filtered_labels)) > 1:
        auroc = roc_auc_score(filtered_labels, filtered_logits)
        auprc = average_precision_score(filtered_labels, filtered_logits, average='macro')
    else:
        auroc = 0.0
        auprc = 0.0
        print(f"Warning: 过滤后只有单一标签类别，无法计算AUROC/AUPRC")
    
    # 重构误差统计
    filtered_rec_errors = rec_errors_np[keep_indices]
    rec_error_mean = np.mean(filtered_rec_errors)
    rec_error_std = np.std(filtered_rec_errors)
    rec_error_min = np.min(filtered_rec_errors)
    rec_error_max = np.max(filtered_rec_errors)
    
    # 正常节点和异常节点的重构误差分别统计
    normal_rec_errors = rec_errors_np[keep_indices][filtered_labels == 0]
    anomaly_rec_errors = rec_errors_np[keep_indices][filtered_labels == 1]
    
    normal_rec_mean = np.mean(normal_rec_errors) if len(normal_rec_errors) > 0 else 0.0
    anomaly_rec_mean = np.mean(anomaly_rec_errors) if len(anomaly_rec_errors) > 0 else 0.0
    
    results = {
        'auroc': auroc,
        'auprc': auprc,
        'rec_error_filter_ratio': rec_error_filter_ratio,
        'total_test_nodes': total_nodes,
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
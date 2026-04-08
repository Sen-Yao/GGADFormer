"""
消融实验模块

将各消融变体封装为独立函数，避免 GGADFormer.forward 臃肿。
当前仅针对重构误差扰动向量 (perturbation) 的消融，
未来可扩展为针对注意力机制、编码器等方面的消融。

所有消融模式的缩放因子均从当前 batch 数据中自动提取，
无需人工指定超参数，消除审稿人对参数选取的质疑。
"""

import torch
import torch.nn.functional as F


def apply_perturbation_ablation(reconstruction_error_proj, ablation_mode):
    """
    针对重构误差扰动向量的消融实验。

    所有模长统计量从当前 batch 的 reconstruction_error_proj 自动计算，
    无需人为指定缩放因子。

    Args:
        reconstruction_error_proj: 原始重构误差投影 [batch, embedding_dim]
        ablation_mode: 消融模式
            - 'none': 不做消融，直接返回原向量
            - 'random_dir': 保留模长，随机方向
            - 'random_mag': 保留方向，随机模长（采样范围由数据决定）
            - 'random_both': 完全随机向量（均值和方差与原向量一致）
            - 'constant_mag': 保留方向，用 batch 平均模长替换

    Returns:
        处理后的 reconstruction_error_proj
    """
    if ablation_mode is None or ablation_mode == 'none':
        return reconstruction_error_proj

    # 从当前 batch 自动提取模长统计量
    norms = torch.norm(reconstruction_error_proj, p=2, dim=1, keepdim=True)
    mean_norm = norms.mean()
    std_norm = norms.std()

    if ablation_mode == 'random_dir':
        # 保留 learned magnitude，randomize direction
        random_dir = F.normalize(torch.randn_like(reconstruction_error_proj), p=2, dim=1)
        return norms * random_dir

    elif ablation_mode == 'random_mag':
        # 保留 learned direction，randomize magnitude
        # 使用高斯分布，保持均值和方差与原向量一致
        direction = F.normalize(reconstruction_error_proj, p=2, dim=1)
        # 从 N(mean_norm, std_norm^2) 采样，并截断负值
        random_mag = torch.randn_like(norms) * std_norm + mean_norm
        random_mag = torch.clamp(random_mag, min=0)  # 确保非负
        return direction * random_mag

    elif ablation_mode == 'random_both':
        # 完全随机的 perturbation
        # 方向和模长都独立随机化，但保持模长的统计量一致
        random_vec = torch.randn_like(reconstruction_error_proj)
        random_dirs = F.normalize(random_vec, p=2, dim=1)
        random_mags = torch.randn_like(norms) * std_norm + mean_norm
        random_mags = torch.clamp(random_mags, min=0.01)  # 避免0模长
        return random_dirs * random_mags

    elif ablation_mode == 'constant_mag':
        # 保留 learned direction，用 batch 平均模长替换
        # 控制了模长的统计量，仅测试方向信息的贡献
        direction = F.normalize(reconstruction_error_proj, p=2, dim=1)
        return direction * mean_norm

    else:
        # 非本消融的 mode，返回原值（让 h_mean 等其他消融处理）
        return reconstruction_error_proj


def apply_h_mean_ablation(emb, ablation_mode, normal_for_train_idx=None):
    """
    针对全局中心点 h_mean 计算方式的消融实验。

    Args:
        emb: 节点嵌入表征, 形状 [1, N, embedding_dim]
        ablation_mode: 消融模式
            - 'none': 原始模型，所有节点的嵌入平均（global_mean）
            - 'h_mean_labeled_normal': 仅使用有标签的正常节点嵌入平均
            - 'h_mean_trimmed': Trimmed mean，去掉离中心最远的 10% 节点后取平均
        normal_for_train_idx: 有标签的正常节点索引（仅在 'h_mean_labeled_normal' 模式下使用）

    Returns:
        h_mean: 全局中心点, 形状 [1, 1, embedding_dim]
    """
    # 默认：原始模型，所有节点嵌入平均
    h_mean = torch.mean(emb, dim=1, keepdim=True)

    if ablation_mode is None or ablation_mode == 'none':
        return h_mean

    elif ablation_mode == 'h_mean_labeled_normal':
        # 仅使用有标签的正常节点嵌入平均
        if normal_for_train_idx is None or len(normal_for_train_idx) == 0:
            # 退回到 global_mean
            return h_mean
        normal_emb = emb[:, normal_for_train_idx, :]  # [1, num_normal, embedding_dim]
        return torch.mean(normal_emb, dim=1, keepdim=True)

    elif ablation_mode == 'h_mean_trimmed':
        # Trimmed mean：先计算粗略中心，去掉离中心最远的 10% 节点，再重新计算均值
        distances = torch.norm(emb - h_mean, p=2, dim=2).squeeze(0)  # [N]
        # 确定裁剪数量（去掉最远的 10%）
        num_nodes = distances.size(0)
        num_trim = max(1, int(num_nodes * 0.1))
        # 找到距离最近的 (1 - 0.1) 比例的节点
        _, sorted_indices = torch.sort(distances)
        keep_indices = sorted_indices[:num_nodes - num_trim]
        # 对保留的节点取平均
        trimmed_emb = emb[:, keep_indices, :]  # [1, num_kept, embedding_dim]
        return torch.mean(trimmed_emb, dim=1, keepdim=True)

    else:
        # 非本消融的 mode，返回默认值（让 perturbation 消融处理）
        return h_mean

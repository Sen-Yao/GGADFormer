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
        raise ValueError(f"Unknown ablation_mode: '{ablation_mode}'. "
                         f"Supported: 'none', 'random_dir', 'random_mag', 'random_both', 'constant_mag'")
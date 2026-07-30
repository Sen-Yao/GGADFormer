"""
消融实验模块

将各消融变体封装为独立函数，避免 GGADFormer.forward 臃肿。
当前仅针对重构误差扰动向量 (perturbation) 的消融，
未来可扩展为针对注意力机制、编码器等方面的消融。

所有消融模式的缩放因子均从当前 batch 数据中自动提取，
无需人工指定超参数，消除审稿人对参数选取的质疑。

新增：GPRGNN式邻域融合消融实验
对比 Transformer 融合与可学习权重线性组合的有效性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _randn_like_with_generator(tensor, generator):
    if generator is None:
        return torch.randn_like(tensor)
    return torch.randn(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        generator=generator,
    )


def normalize_direction(vectors, eps=1e-12):
    """
    L2-normalize row vectors without producing NaNs for zero rows.
    """
    norms = torch.norm(vectors, p=2, dim=1, keepdim=True)
    return vectors / norms.clamp_min(eps)


def permute_magnitudes_cyclic(magnitudes, generator=None):
    """
    Permute node magnitudes by a non-zero cyclic shift within the current
    generation subset.

    For n >= 2 this preserves the exact magnitude multiset and creates no fixed
    points. For n < 2 the identity is explicit because no association can be
    broken.
    """
    if magnitudes.dim() != 2 or magnitudes.size(1) != 1:
        raise ValueError(
            f"magnitudes must have shape [n, 1], got {tuple(magnitudes.shape)}"
        )

    n = magnitudes.size(0)
    if n < 2:
        return magnitudes.clone()

    if generator is None:
        shift_tensor = torch.randint(1, n, (1,), device=magnitudes.device)
    else:
        shift_tensor = torch.randint(
            1, n, (1,), device=magnitudes.device, generator=generator
        )
    shift = int(shift_tensor.item())
    return torch.roll(magnitudes, shifts=shift, dims=0)


def apply_perturbation_ablation(
    reconstruction_error_proj,
    ablation_mode,
    direction_generator=None,
    magnitude_generator=None,
):
    """
    针对重构误差扰动向量的消融实验。

    所有模长统计量从当前 batch 的 reconstruction_error_proj 自动计算，
    无需人为指定缩放因子。

    Args:
        reconstruction_error_proj: 原始重构误差投影 [batch, embedding_dim]
        ablation_mode: 消融模式
            - 'none': 不做消融，直接返回原向量
            - 'random_dir': 保留模长，随机方向
            - 'random_mag': 保留方向，使用当前生成子集内的精确模长置换
            - 'random_both': 随机方向 + 同一精确模长置换机制
            - 'constant_mag': 保留方向，用 batch 平均模长替换
        direction_generator: random_dir/random_both 使用的独立显式 RNG
        magnitude_generator: random_mag/random_both 使用的独立显式 RNG

    Returns:
        处理后的 reconstruction_error_proj
    """
    if ablation_mode is None or ablation_mode == 'none':
        return reconstruction_error_proj

    # 从当前生成子集自动提取模长；后续置换不 detach，保持 projection head 梯度路径。
    norms = torch.norm(reconstruction_error_proj, p=2, dim=1, keepdim=True)
    mean_norm = norms.mean()

    if ablation_mode == 'random_dir':
        # 保留 learned magnitude，randomize direction
        random_vec = _randn_like_with_generator(
            reconstruction_error_proj, direction_generator
        )
        random_dir = normalize_direction(random_vec)
        return norms * random_dir

    elif ablation_mode == 'random_mag':
        # 保留 learned direction，用精确置换后的 donor magnitude 打破 node-magnitude association。
        direction = normalize_direction(reconstruction_error_proj)
        permuted_mag = permute_magnitudes_cyclic(norms, magnitude_generator)
        return direction * permuted_mag

    elif ablation_mode == 'random_both':
        # 随机方向与 random_mag 完全相同的精确模长置换边界语义。
        random_vec = _randn_like_with_generator(
            reconstruction_error_proj, direction_generator
        )
        random_dirs = normalize_direction(random_vec)
        permuted_mag = permute_magnitudes_cyclic(norms, magnitude_generator)
        return random_dirs * permuted_mag

    elif ablation_mode == 'constant_mag':
        # 保留 learned direction，用 batch 平均模长替换
        # 控制了模长的统计量，仅测试方向信息的贡献
        direction = normalize_direction(reconstruction_error_proj)
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


class GPRGNNFusion(nn.Module):
    """
    GPRGNN式可学习权重融合模块
    
    参考：*"Adaptive Universal Generalized PageRank Graph Neural Network"* (ICLR 2021)
    
    使用一组可学习的标量权重 {γ_k}_{k=0}^{K} 对各跳 token 进行加权求和：
    h_i = sum_{k=0}^{K} γ_k * t_i^{(k)}
    
    权重通过 softmax 归一化保证和为1。
    """
    
    def __init__(self, num_hops):
        """
        Args:
            num_hops: token序列的跳数 (K+1)，即 args.pp_k + 1
        """
        super(GPRGNNFusion, self).__init__()
        # 可学习权重参数 γ_k，初始化为0使得 softmax 后均匀分布
        self.gamma = nn.Parameter(torch.zeros(num_hops))
    
    def forward(self, tokens):
        """
        对多跳 token 序列进行加权求和
        
        Args:
            tokens: 投影后的 token 序列，形状 [N, K+1, embedding_dim]
        
        Returns:
            emb: 融合后的节点嵌入，形状 [N, embedding_dim]
        """
        # softmax 归一化权重
        weights = F.softmax(self.gamma, dim=0)  # [K+1]
        
        # 加权求和：h = sum(weights[k] * tokens[:, k, :])
        # weights.unsqueeze(0).unsqueeze(2) -> [1, K+1, 1]
        # tokens -> [N, K+1, embedding_dim]
        # 结果 -> [N, embedding_dim]
        emb = torch.sum(weights.unsqueeze(0).unsqueeze(2) * tokens, dim=1)
        
        return emb
    
    def get_weights(self):
        """获取当前的归一化权重（用于分析）"""
        return F.softmax(self.gamma, dim=0).detach().cpu().numpy()


def create_token_fusion_module(ablation_mode, num_hops, embedding_dim, device):
    """
    根据 ablation_mode 创建相应的 token 融合模块
    
    Args:
        ablation_mode: 消融模式
            - 'none' / 'transformer': 使用 Transformer 融合，返回 None
            - 'gprgnn_weighted_sum': 使用 GPRGNN 式可学习权重求和
        num_hops: token 序列的跳数 (K+1)
        embedding_dim: 嵌入维度（当前 GPRGNN 融合不需要，保留参数一致性）
        device: 设备
    
    Returns:
        fusion_module: GPRGNNFusion 模块或 None
    """
    if ablation_mode == 'gprgnn_weighted_sum':
        module = GPRGNNFusion(num_hops)
        return module.to(device)
    
    # 其他模式使用 Transformer，不需要额外模块
    return None


def should_use_gprgnn_fusion(ablation_mode):
    """
    判断是否应使用 GPRGNN 融合方式
    
    Args:
        ablation_mode: 消融模式
    
    Returns:
        bool: 是否使用 GPRGNN 融合
    """
    return ablation_mode == 'gprgnn_weighted_sum'

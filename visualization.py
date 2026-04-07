from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import wandb
import numpy as np
import torch
import pandas as pd

def create_tsne_visualization(features, emb_last_epoch, labels, epoch, normal_for_train_idx, outlier_emb_last_epoch, args):
    """
    创建tsne可视化并保存到wandb
    
    Args:
        features: 原始特征 [num_nodes, feature_dim]
        emb_last_epoch: 模型生成的嵌入 [1, num_nodes, embedding_dim]
        labels: 真实标签 [num_nodes]
        epoch: 当前epoch
        normal_for_train_idx: 用于训练的正常节点索引（可为None）
        outlier_emb_last_epoch: 生成的异常节点嵌入（可为None）
        args: 参数配置
    """
    # 准备tsne数据
    # 获取原始特征
    if features.dim() > 2:
        features = features.squeeze(0)
    
    # 获取嵌入（去掉batch维度）
    if emb_last_epoch.dim() > 2:
        embeddings_last_epoch = emb_last_epoch.squeeze(0)  # [batch_size, embedding_dim]
    else:
        embeddings_last_epoch = emb_last_epoch
    
    # 获取真实标签（去掉batch维度）
    if labels.dim() > 1:
        labels = labels.squeeze(0)  # [batch_size]

    # 处理outlier_emb可能为None的情况
    outlier_emb_len = 0 if outlier_emb_last_epoch is None else len(outlier_emb_last_epoch)
    
    # 先将数据移到CPU，避免CUDA错误
    print(f"\n\nStarting tsne visualization...")
    features = features.cpu().detach().numpy()
    embeddings_last_epoch = embeddings_last_epoch.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels
    
    # 创建节点类型标签
    node_types = []
    print(f"nb_nodes: {args.batch_size}, outlier_emb_len: {outlier_emb_len}")
    for i in range(args.batch_size):
        if i < len(labels_np) and labels_np[i] == 1:
            # 真实异常点
            node_types.append("anomaly") 
        else:
            node_types.append("normal")
    
    # 添加生成的异常节点类型标签
    for i in range(outlier_emb_len):
        node_types.append("pesudo")
    
    # 创建用于可视化的嵌入和标签
    # 对于生成的离群点，我们需要将其添加到嵌入空间中，但不在特征空间中
    if outlier_emb_last_epoch is not None and len(outlier_emb_last_epoch) > 0:
        # 确保outlier_emb是numpy数组
        embeddings_last_epoch = np.concatenate([embeddings_last_epoch, outlier_emb_last_epoch.cpu().detach().numpy()], axis=0)

    

    # 创建tsne可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='random', learning_rate=200.0)
    
    # 对过滤后的原始特征进行tsne
    features_2d = tsne.fit_transform(features)
    
    # 对过滤后的嵌入进行tsne
    embeddings_2d_last_epoch = tsne.fit_transform(embeddings_last_epoch)
    # 创建wandb表格
    # 原始特征空间的tsne
    feature_table_data = []
    for i in range(len(features_2d)):
        feature_table_data.append([
            float(features_2d[i, 0]),
            float(features_2d[i, 1]),
            node_types[i]
        ])
    
    feature_table = wandb.Table(
        columns=["TSNE_X", "TSNE_Y", "Node_Type"],
        data=feature_table_data
    )
    
    # 嵌入空间的tsne
    embedding_table_data_last_epoch = []
    for i in range(len(embeddings_2d_last_epoch)):
        # 对于生成的异常节点，使用默认标签和类型
        if i < len(node_types):
            node_type = node_types[i]
        else:
            node_type = "generated_anomaly"
        
        embedding_table_data_last_epoch.append([
            float(embeddings_2d_last_epoch[i, 0]),
            float(embeddings_2d_last_epoch[i, 1]),
            node_type
        ])
    
    embedding_table_last_epoch = wandb.Table(
        columns=["TSNE_X", "TSNE_Y", "Node_Type"],
        data=embedding_table_data_last_epoch
    )
    
    # 记录到wandb
    wandb.log({
        f"tsne_features": feature_table,
        f"tsne_embeddings_last_epoch": embedding_table_last_epoch,
    })
    print("tsne visualization done!\n")


def visualize_attention_weights(agg_attention_weights, labels, normal_for_train_idx, normal_for_generation_idx, 
                               outlier_emb, epoch, dataset_name, device, adj_matrix=None, args=None):
    """
    分析注意力权重，将原始注意力数据保存到wandb table中
    
    Args:
        agg_attention_weights: 注意力权重矩阵 [1, num_nodes, num_nodes]
        labels: 真实标签 [1, num_nodes]
        normal_for_train_idx: 用于训练的正常节点索引
        normal_for_generation_idx: 用于生成异常节点的正常节点索引
        outlier_emb: 生成的异常节点嵌入
        epoch: 当前epoch
        dataset_name: 数据集名称
        device: 设备
        adj_matrix: 邻接矩阵，用于判断节点间的邻居关系
    """
    print(f"\n\nStarting attention weights visualization...")
    if args.model_type != 'SGT':
        # 将注意力权重从GPU移到CPU并转换为numpy
        attention_weights = agg_attention_weights.squeeze(0).detach().cpu().numpy()  # [num_nodes, num_nodes]
        labels_np = labels.squeeze(0).detach().cpu().numpy()  # [num_nodes]
        
        # 获取节点数量
        num_nodes = attention_weights.shape[0]
        
        # 计算生成的异常节点数量
        if outlier_emb is None:
            outlier_emb_len = 0
        else:
            outlier_emb_len = len(outlier_emb)
        
        # 创建节点类型标签
        node_types = []
        for i in range(num_nodes):
            if i >= num_nodes - outlier_emb_len:
                node_types.append("generated_anomaly")
            elif labels_np[i] == 1:
                node_types.append("anomaly")
            else:
                node_types.append("normal")
        
        # 将节点类型转换为numpy数组
        node_types = np.array(node_types)
        
        # 分别获取正常节点和异常节点的索引
        normal_indices = np.where(node_types == "normal")[0]
        anomaly_indices = np.where(node_types == "anomaly")[0]
        generated_anomaly_indices = np.where(node_types == "generated_anomaly")[0]
        
        # print(f"节点统计: 正常={len(normal_indices)}, 异常={len(anomaly_indices)}, 生成异常={len(generated_anomaly_indices)}")
        
        # 1. 从正常点中选取前len(anomaly_indices)个，保证正常点和异常点数量相似
        if len(anomaly_indices) > 0 and len(normal_indices) > 0:
            max_sample_num = 50
            sampled_normal_count = min(len(anomaly_indices), len(normal_indices))
            sampled_normal_indices = normal_indices[:max_sample_num]
            sampled_anomaly_indices = anomaly_indices[:max_sample_num]
            
            # print(f"采样节点数量: {max_sample_num}")
            '''
            # 2. 记录每个采样出来的正常点关于其他全部正常点的注意力
            normal_to_normal_data = []
            for i, source_node in enumerate(sampled_normal_indices):
                for j, target_node in enumerate(sampled_normal_indices):
                    attention_value = attention_weights[source_node, target_node]
                    normal_to_normal_data.append([
                        "normal", "normal", source_node, target_node, attention_value
                    ])
            print("Finished saving normal to normal attention weights")
            # 3. 记录每个采样出来的正常点关于其他全部异常点的注意力
            normal_to_anomaly_data = []
            for i, source_node in enumerate(sampled_normal_indices):
                for j, target_node in enumerate(sampled_anomaly_indices):
                    attention_value = attention_weights[source_node, target_node]
                    normal_to_anomaly_data.append([
                        "normal", "abnormal", source_node, target_node, attention_value
                    ])
            print("Finished saving normal to anomaly attention weights")
            # 4. 记录每个异常点关于其他全部异常点的注意力
            anomaly_to_anomaly_data = []
            for i, source_node in enumerate(sampled_anomaly_indices):
                for j, target_node in enumerate(sampled_anomaly_indices):
                    attention_value = attention_weights[source_node, target_node]
                    anomaly_to_anomaly_data.append([
                        "abnormal", "abnormal", source_node, target_node, attention_value
                    ])
            '''
            
            # 记录前一百个正常点和异常点的注意力
            all_to_all_data = []
            # 拼接采样的正常点和异常点索引
            all_sampled_indices = np.concatenate([sampled_normal_indices, sampled_anomaly_indices])
            for i, source_node in enumerate(all_sampled_indices):
                for j, target_node in enumerate(all_sampled_indices):
                    attention_value = attention_weights[source_node, target_node]
                    all_to_all_data.append([
                        i,
                        j,
                        "normal" if source_node in sampled_normal_indices else "abnormal", 
                        "normal" if target_node in sampled_normal_indices else "abnormal", 
                        source_node, 
                        target_node, 
                        attention_value
                    ])
            df = pd.DataFrame(all_to_all_data, columns=["index_1", "index_2", "source_type", "target_type", "source_node", "target_node", "attention_weight"])

            # 按 source_type 分组并计算方差
            variance_by_type = df.groupby('source_type')['attention_weight'].var().to_dict()

            print(variance_by_type)
            print("Saving attention weights to wandb table...")
            # 保存到wandb table
            wandb.log({
                f"attention_tables/all_to_all_epoch_{epoch}": wandb.Table(
                    columns=["index_1", "index_2", "source_type", "target_type", "source_node", "target_node", "attention_weight"],
                    data=all_to_all_data
                )
            })
            
            # print(f"  正常->正常: {len(normal_to_normal_data)} 个数据点")
            # print(f"  正常->异常: {len(normal_to_anomaly_data)} 个数据点")
            # print(f"  异常->异常: {len(anomaly_to_anomaly_data)} 个数据点")
            
            # 添加画图功能：记录选定节点关于其他节点的注意力
            
            # 获取邻接矩阵信息（从原始数据中获取）
            # 这里需要从外部传入邻接矩阵，暂时使用一个简单的判断方法
            # 在实际使用时，您需要将邻接矩阵作为参数传入
            print(f"Analyzing attention weights from single node...")
            # 选择一个正常节点和一个异常节点进行分析

            if len(normal_indices) > 0 and len(anomaly_indices) > 0:
                selected_normal_node = normal_indices[0]  # 选择第一个正常节点
                selected_anomaly_node = anomaly_indices[0]  # 选择第一个异常节点
                
                # 分析正常节点的注意力分布
                normal_node_attention_data = []
                for target_node in range(num_nodes):
                    attention_value = attention_weights[selected_normal_node, target_node]
                    target_type = "normal" if target_node in normal_indices else ("anomaly" if target_node in anomaly_indices else "generated_anomaly")
                    
                    # 判断是否为邻居

                    is_neighbor = adj_matrix[selected_normal_node, target_node] > 0                
                    normal_node_attention_data.append([
                        target_node,  target_type, 
                        attention_value, is_neighbor
                    ])
                
                # 分析异常节点的注意力分布
                anomaly_node_attention_data = []
                for target_node in range(num_nodes):
                    attention_value = attention_weights[selected_anomaly_node, target_node]
                    target_type = "normal" if target_node in normal_indices else ("anomaly" if target_node in anomaly_indices else "generated_anomaly")
                    
                    # 判断是否为邻居
                    is_neighbor = adj_matrix[selected_anomaly_node, target_node] > 0
                    
                    anomaly_node_attention_data.append([
                        target_node,  target_type, 
                        attention_value, is_neighbor
                    ])
                
                # 保存到wandb table
                print(f"Saving single node attention weights to wandb table...")
                wandb.log({
                    f"attention_analysis/normal_node_attention_epoch_{epoch}": wandb.Table(
                        columns=["target_node", "target_type", "attention_weight", "is_neighbor"],
                        data=normal_node_attention_data
                    ),
                    f"attention_analysis/anomaly_node_attention_epoch_{epoch}": wandb.Table(
                        columns=["target_node", "target_type", "attention_weight", "is_neighbor"],
                        data=anomaly_node_attention_data
                    )
                })
            
        else:
            print("Warning: No enough normal nodes or anomaly nodes for analysis")
            wandb.log({
                f"attention_analysis/num_normal_nodes": len(normal_indices),
                f"attention_analysis/num_anomaly_nodes": len(anomaly_indices),
                f"attention_analysis/num_generated_anomaly_nodes": len(generated_anomaly_indices),
                f"attention_analysis/sampled_normal_count": 0,
                f"attention_analysis/normal_to_normal_pairs": 0,
                f"attention_analysis/normal_to_anomaly_pairs": 0,
                f"attention_analysis/anomaly_to_anomaly_pairs": 0,
            })
            print("Attention weights visualization done!")
            return {
                'normal_indices': normal_indices,
                'anomaly_indices': anomaly_indices,
                'generated_anomaly_indices': generated_anomaly_indices,
                'node_types': node_types
            }
    else:
        # 对于SGT模型，注意力权重比较复杂，暂时没有实现可视化
        return None


def visualize_reconstruction_analysis(model, input_tokens, labels, ano_label, 
                                       idx_test, epoch, args, device):
    """
    在最后一次eval时进行重构误差分析可视化
    
    功能：
    1. 计算正常点和异常点的reconstruction_error
    2. 计算归一化后的||RDV||_2（批内归一化后计算L2范数）
    3. t-SNE降维可视化重构误差分布
    4. 计算正常和异常内部的Silhouette Score
    
    Args:
        model: GGADFormer模型
        input_tokens: 输入tokens [N, pp_k+1, feature_dim]
        labels: 真实标签 [N]（batch内的标签）
        ano_label: 全图的真实异常标签 [total_nodes]
        idx_test: 测试集索引
        epoch: 当前epoch
        args: 参数配置
        device: 设备
    """
    print("\n\nStarting reconstruction error analysis visualization...")
    
    # 获取模型参数
    n_in = model.n_in
    
    with torch.no_grad():
        # 1. 获取模型编码
        emb = model.TransformerEncoder(input_tokens)  # [1, N, embedding_dim]
        
        # 2. 重构tokens
        reconstructed_tokens = model.token_decoder(emb).squeeze(0)  # [N, (pp_k+1)*n_in]
        
        # 3. 计算重构误差
        original_tokens_flat = input_tokens.view(-1, (args.pp_k+1) * n_in)  # [N, (pp_k+1)*n_in]
        reconstruction_error = reconstructed_tokens - original_tokens_flat  # [N, (pp_k+1)*n_in]
        
        # 4. 投影到embedding维度得到RDV
        RDV = model.reconstruction_proj(reconstruction_error)  # [N, embedding_dim]
    
    # 5. 批内归一化RDV（batch normalization）
    RDV_mean = RDV.mean(dim=0, keepdim=True)
    RDV_std = RDV.std(dim=0, keepdim=True) + 1e-8  # 防止除零
    RDV_normalized = (RDV - RDV_mean) / RDV_std  # [N, embedding_dim]
    
    # 6. 计算归一化后的||RDV||_2
    rdv_norm = torch.norm(RDV_normalized, p=2, dim=1)  # [N]
    
    # 转换为numpy进行处理
    RDV_normalized_np = RDV_normalized.cpu().numpy()
    rdv_norm_np = rdv_norm.cpu().numpy()
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    # 识别正常点和异常点索引
    normal_indices = np.where(labels_np == 0)[0]
    anomaly_indices = np.where(labels_np == 1)[0]
    
    print(f"节点统计: 正常点={len(normal_indices)}, 异常点={len(anomaly_indices)}")
    
    if len(normal_indices) == 0 or len(anomaly_indices) == 0:
        print("Warning: 正常点或异常点数量为0，跳过可视化")
        return
    
    # 7. 分别统计正常点和异常点的归一化||RDV||_2分布
    normal_rdv_norm = rdv_norm_np[normal_indices]
    anomaly_rdv_norm = rdv_norm_np[anomaly_indices]
    
    print(f"正常点 ||RDV||_2 统计: mean={np.mean(normal_rdv_norm):.4f}, std={np.std(normal_rdv_norm):.4f}")
    print(f"异常点 ||RDV||_2 统计: mean={np.mean(anomaly_rdv_norm):.4f}, std={np.std(anomaly_rdv_norm):.4f}")
    
    # 8. t-SNE降维可视化
    print("Performing t-SNE dimensionality reduction...")
    # 调整perplexity，确保不超过样本数
    perplexity = min(30, len(RDV_normalized_np) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                init='random', learning_rate=200.0)
    RDV_2d = tsne.fit_transform(RDV_normalized_np)  # [N, 2]
    
    # 9. 计算Silhouette Score
    # 正常点内部的聚类紧密度（使用所有正常点作为一类）
    if len(normal_indices) >= 2:
        # 对于Silhouette Score，需要至少2个样本
        # 计算正常点之间的平均距离作为紧密度度量
        normal_points_2d = RDV_2d[normal_indices]
        # 使用 intra-cluster distance 衡量紧密度
        normal_center = np.mean(normal_points_2d, axis=0)
        normal_intra_dist = np.mean(np.linalg.norm(normal_points_2d - normal_center, axis=1))
        
        # 计算Silhouette Score（正常点作为一类）
        normal_labels_for_silhouette = np.zeros(len(normal_indices))
        if len(anomaly_indices) >= 1:
            # 创建二分类标签用于计算silhouette
            all_points_for_normal = np.vstack([normal_points_2d, RDV_2d[anomaly_indices[:min(len(anomaly_indices), len(normal_indices))]]])
            all_labels_for_normal = np.array([0] * len(normal_indices) + [1] * min(len(anomaly_indices), len(normal_indices)))
            if len(set(all_labels_for_normal)) > 1:
                normal_silhouette = silhouette_score(all_points_for_normal, all_labels_for_normal)
            else:
                normal_silhouette = 0.0
        else:
            normal_silhouette = 0.0
        
        print(f"正常点 Silhouette Score: {normal_silhouette:.4f}")
        print(f"正常点 intra-cluster distance: {normal_intra_dist:.4f}")
    else:
        normal_silhouette = 0.0
        normal_intra_dist = 0.0
        print("Warning: 正常点数量不足，无法计算Silhouette Score")
    
    # 异常点内部的聚类紧密度
    if len(anomaly_indices) >= 2:
        anomaly_points_2d = RDV_2d[anomaly_indices]
        anomaly_center = np.mean(anomaly_points_2d, axis=0)
        anomaly_intra_dist = np.mean(np.linalg.norm(anomaly_points_2d - anomaly_center, axis=1))
        
        # 计算Silhouette Score（异常点作为一类）
        anomaly_labels_for_silhouette = np.ones(len(anomaly_indices))
        if len(normal_indices) >= 1:
            all_points_for_anomaly = np.vstack([anomaly_points_2d, RDV_2d[normal_indices[:min(len(normal_indices), len(anomaly_indices))]]])
            all_labels_for_anomaly = np.array([1] * len(anomaly_indices) + [0] * min(len(normal_indices), len(anomaly_indices)))
            if len(set(all_labels_for_anomaly)) > 1:
                anomaly_silhouette = silhouette_score(all_points_for_anomaly, all_labels_for_anomaly)
            else:
                anomaly_silhouette = 0.0
        else:
            anomaly_silhouette = 0.0
        
        print(f"异常点 Silhouette Score: {anomaly_silhouette:.4f}")
        print(f"异常点 intra-cluster distance: {anomaly_intra_dist:.4f}")
    else:
        anomaly_silhouette = 0.0
        anomaly_intra_dist = 0.0
        print("Warning: 异常点数量不足，无法计算Silhouette Score")
    
    # 10. 创建可视化图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：t-SNE可视化
    ax1 = axes[0]
    ax1.scatter(RDV_2d[normal_indices, 0], RDV_2d[normal_indices, 1], 
                c='blue', alpha=0.6, s=20, label=f'Normal ({len(normal_indices)})')
    ax1.scatter(RDV_2d[anomaly_indices, 0], RDV_2d[anomaly_indices, 1], 
                c='red', alpha=0.6, s=20, label=f'Anomaly ({len(anomaly_indices)})')
    ax1.set_title(f't-SNE Visualization of RDV\nNormal Silhouette: {normal_silhouette:.3f}, Anomaly Silhouette: {anomaly_silhouette:.3f}')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：||RDV||_2分布直方图
    ax2 = axes[1]
    bins = 30
    ax2.hist(normal_rdv_norm, bins=bins, alpha=0.6, color='blue', 
             label='Normal', density=True)
    ax2.hist(anomaly_rdv_norm, bins=bins, alpha=0.6, color='red', 
             label='Anomaly', density=True)
    ax2.set_title(f'Distribution of Normalized ||RDV||_2\n'
                  f'Normal: μ={np.mean(normal_rdv_norm):.3f}, σ={np.std(normal_rdv_norm):.3f}\n'
                  f'Anomaly: μ={np.mean(anomaly_rdv_norm):.3f}, σ={np.std(anomaly_rdv_norm):.3f}')
    ax2.set_xlabel('Normalized ||RDV||_2')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_dir = './figs/reconstruction_analysis'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'rdv_analysis_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {save_path}")
    
    # 11. 记录到wandb
    # 创建RDV统计表格
    rdv_stats_table = wandb.Table(
        columns=["Metric", "Normal", "Anomaly"],
        data=[
            ["Count", len(normal_indices), len(anomaly_indices)],
            ["||RDV||_2 Mean", np.mean(normal_rdv_norm), np.mean(anomaly_rdv_norm)],
            ["||RDV||_2 Std", np.std(normal_rdv_norm), np.std(anomaly_rdv_norm)],
            ["||RDV||_2 Min", np.min(normal_rdv_norm), np.min(anomaly_rdv_norm)],
            ["||RDV||_2 Max", np.max(normal_rdv_norm), np.max(anomaly_rdv_norm)],
            ["Silhouette Score", normal_silhouette, anomaly_silhouette],
            ["Intra-cluster Distance", normal_intra_dist, anomaly_intra_dist],
        ]
    )
    
    # 创建t-SNE坐标表格
    tsne_table_data = []
    for i in range(len(RDV_2d)):
        node_type = "normal" if i in normal_indices else "anomaly"
        tsne_table_data.append([
            i,
            float(RDV_2d[i, 0]),
            float(RDV_2d[i, 1]),
            node_type,
            float(rdv_norm_np[i])
        ])
    
    tsne_table = wandb.Table(
        columns=["Index", "TSNE_X", "TSNE_Y", "Node_Type", "RDV_Norm"],
        data=tsne_table_data
    )
    
    # 记录图片和表格到wandb
    wandb.log({
        f"reconstruction_analysis/RDV_stats_epoch_{epoch}": rdv_stats_table,
        f"reconstruction_analysis/TSNE_table_epoch_{epoch}": tsne_table,
        f"reconstruction_analysis/RDV_figure_epoch_{epoch}": wandb.Image(save_path),
        f"reconstruction_analysis/normal_silhouette": normal_silhouette,
        f"reconstruction_analysis/anomaly_silhouette": anomaly_silhouette,
        f"reconstruction_analysis/normal_rdv_norm_mean": np.mean(normal_rdv_norm),
        f"reconstruction_analysis/anomaly_rdv_norm_mean": np.mean(anomaly_rdv_norm),
    })
    
    print("Reconstruction error analysis visualization done!\n")
    
    return {
        'normal_silhouette': normal_silhouette,
        'anomaly_silhouette': anomaly_silhouette,
        'normal_rdv_norm_mean': np.mean(normal_rdv_norm),
        'anomaly_rdv_norm_mean': np.mean(anomaly_rdv_norm),
        'RDV_2d': RDV_2d,
        'normal_indices': normal_indices,
        'anomaly_indices': anomaly_indices
    }

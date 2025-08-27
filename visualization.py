from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import wandb
import numpy as np
import torch

def create_tsne_visualization(features, embeddings, labels, node_types, epoch, device, 
                             normal_for_train_idx, normal_for_generation_idx, outlier_emb=None):
    """
    创建tsne可视化并保存到wandb
    
    Args:
        features: 原始特征 [num_nodes, feature_dim]
        embeddings: 模型生成的嵌入 [num_nodes, embedding_dim]
        labels: 真实标签 [num_nodes]
        node_types: 节点类型标签 [num_nodes]
        epoch: 当前epoch
        device: 设备
        normal_for_train_idx: 用于训练的正常节点索引
        normal_for_generation_idx: 用于生成异常节点的正常节点索引
        outlier_emb: 生成的异常节点嵌入 [num_generated_anomalies, embedding_dim]
    """
    # 将数据移到CPU并转换为numpy，使用detach()避免梯度问题
    features = features.cpu().detach().numpy()
    embeddings = embeddings.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    num_nodes = features.shape[0]
    
    # 创建用于可视化的嵌入和标签
    # 对于生成的离群点，我们需要将其添加到嵌入空间中，但不在特征空间中
    if outlier_emb is not None and len(outlier_emb) > 0:
        # 确保outlier_emb是numpy数组
        if torch.is_tensor(outlier_emb):
            outlier_emb = outlier_emb.cpu().detach().numpy()
        elif not isinstance(outlier_emb, np.ndarray):
            outlier_emb = np.array(outlier_emb)
        embeddings = np.concatenate([embeddings, outlier_emb], axis=0)
    
    # 创建过滤索引：只包含异常节点和属于normal_for_train_idx的正常节点
    filter_indices = []
    for i in range(num_nodes):
        if labels[i] == 1:  # 异常节点
            filter_indices.append(i)
        elif i in normal_for_train_idx:  # 属于训练集的正常节点
            filter_indices.append(i)
    
    # 过滤特征和标签
    filtered_features = features[filter_indices]
    filtered_labels = labels[filter_indices]
    filtered_node_types = [node_types[i] for i in filter_indices]
    
    # 过滤嵌入（只过滤原始节点，不包括生成的异常节点）
    filtered_embeddings = embeddings[:num_nodes][filter_indices]
    
    # 如果有生成的异常节点，添加到过滤后的嵌入中
    if outlier_emb is not None and len(outlier_emb) > 0:
        filtered_embeddings = np.concatenate([filtered_embeddings, outlier_emb], axis=0)

    # 创建tsne可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='random', learning_rate=200.0)
    
    # 对过滤后的原始特征进行tsne
    features_2d = tsne.fit_transform(filtered_features)
    
    # 对过滤后的嵌入进行tsne
    embeddings_2d = tsne.fit_transform(filtered_embeddings)
    print("tsne visualization done! Creating wandb tables...")
    # 创建wandb表格
    # 原始特征空间的tsne
    feature_table_data = []
    for i in range(len(features_2d)):
        feature_table_data.append([
            float(features_2d[i, 0]),
            float(features_2d[i, 1]),
            filtered_node_types[i]
        ])
    
    feature_table = wandb.Table(
        columns=["TSNE_X", "TSNE_Y", "Node_Type"],
        data=feature_table_data
    )
    
    # 嵌入空间的tsne
    embedding_table_data = []
    for i in range(len(embeddings_2d)):
        # 对于生成的异常节点，使用默认标签和类型
        if i < len(filtered_labels):
            label = int(filtered_labels[i])
            node_type = filtered_node_types[i]
        else:
            label = 1  # 生成的异常节点标签为1
            node_type = "generated_anomaly"
        
        embedding_table_data.append([
            float(embeddings_2d[i, 0]),
            float(embeddings_2d[i, 1]),
            node_type
        ])
    
    embedding_table = wandb.Table(
        columns=["TSNE_X", "TSNE_Y", "Node_Type"],
        data=embedding_table_data
    )
    print("wandb tables created! Uploading to wandb...")
    # 记录到wandb
    wandb.log({
        f"tsne_features_epoch_{epoch}": feature_table,
        f"tsne_embeddings_epoch_{epoch}": embedding_table
    })
    print("wandb tables uploaded!")

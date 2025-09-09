import torch
import torch.nn as nn
import torch.nn.functional as F
from masking import masking, compute_loss
from utils import node_neighborhood_feature

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = 1

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias


        # 这里的 x 就是经过 softmax 归一化的注意力权重
        attention_weights = torch.softmax(x, dim=3)

        x = self.att_dropout(attention_weights) # Dropout 应用于注意力权重

        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x, attention_weights


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)

        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)

        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, attn_bias=None):


        y = self.self_attention_norm(x)
        y, attention_weights = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y
        ## 实现的是transformer 和 FFN的LayerNorm 以及相关操作
        
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x, attention_weights


class RAGFormer(nn.Module):
    def __init__(self, n_in, n_h, activation, args):
        super(RAGFormer, self).__init__()

        # 设置设备
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')

        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        self.fc4 = nn.Linear(n_h, n_h, bias=False)
        self.act = nn.ReLU()

        self.n_in = n_in

        # Graph Transformer
        encoders = [EncoderLayer(args.embedding_dim, args.GT_ffn_dim, args.GT_dropout, args.GT_attention_dropout, args.GT_num_heads)
                    for _ in range(args.GT_num_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(args.embedding_dim)

        # Feature projection layers
        self.token_projection = nn.Linear(2 * n_in, args.embedding_dim)

        # Unified token decoder for reconstruction
        self.token_decoder = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.ReLU(),
            nn.Linear(args.embedding_dim, 2 * n_in)
        )

        # 重构损失函数
        self.recon_loss_fn = nn.MSELoss()
        self.emb_normal_mean = torch.zeros(args.embedding_dim, device=self.device)
        
        # 将模型移动到指定设备
        self.to(self.device)

    def forward(self, input_tokens, adj, normal_for_generation_idx, normal_for_train_idx, train_flag, epoch, args, sparse=False):

        concated_input_features = torch.concat((input_tokens.to(args.device), node_neighborhood_feature(adj.squeeze(0), input_tokens.squeeze(0), args.pp_k, args.progregate_alpha).to(args.device).unsqueeze(0)), dim=2)

        # 直接在concatenated tokens上进行masking
        if train_flag:
            mask_ratio = getattr(args, 'mask_ratio', 0.1)
            
            # 创建mask
            num_nodes = concated_input_features.shape[1]
            token_dim = concated_input_features.shape[2]
            
            # 随机选择要mask的节点
            num_mask = int(num_nodes * mask_ratio)
            mask_indices = torch.randperm(num_nodes, device=self.device)[:num_mask]
            
            # 创建masked tokens
            masked_tokens = concated_input_features.clone()
            mask_token = torch.zeros(1, 1, token_dim, device=self.device)  # 使用零作为mask token
            
            # 应用mask
            masked_tokens[0, mask_indices] = mask_token

        else:
            masked_tokens = concated_input_features
            mask_indices = torch.tensor([], device=self.device)



        attention_weights = None # 初始化注意力权重
        agg_attention_weights = None 
        emb = self.token_projection(masked_tokens)
        
        # Graph Transformer编码
        for i, l in enumerate(self.layers):
            emb, current_attention_weights = self.layers[i](emb)
            if i == len(self.layers) - 1: # 拿到最后一层的注意力
                attention_weights = current_attention_weights
                # 聚合多头注意力
                agg_attention_weights = torch.mean(attention_weights, dim=1)
        emb = self.final_ln(emb)
        # emb: [1, num_nodes, hidden_dim]

        # 解码器重构原始输入tokens
        reconstructed_tokens = self.token_decoder(emb)  # [1, num_nodes, input_dim]
        
        # 计算重构损失
        reconstruction_loss = self.recon_loss_fn(reconstructed_tokens, concated_input_features)

        # 计算重构损失或异常分数
        if train_flag:
            # 训练模式：直接使用token重构损失
            reconstruction_loss = self.recon_loss_fn(reconstructed_tokens, concated_input_features)
            anomaly_scores = torch.tensor(0.0, device=self.device)
            m = 0.9

            if epoch != 0:
                self.emb_normal_mean = m * self.emb_normal_mean + (1 - m) * torch.mean(emb[0, normal_for_train_idx], dim=0).detach()
            else:
                self.emb_normal_mean = torch.mean(emb[0, normal_for_train_idx], dim=0).detach()

            # 计算正常节点聚类损失：鼓励正常节点靠近全局中心
            normal_embeddings = emb[0, normal_for_train_idx]  # [num_normal_nodes, emb_dim]
            # 计算每个正常节点到全局中心的欧氏距离平方
            center_distances = torch.sum((emb[0, :, :] - self.emb_normal_mean) ** 2, dim=1)
            # 使用均方误差作为聚类损失
            compact_loss = torch.mean(torch.max(torch.zeros_like(center_distances), center_distances - 1))
            
        else:
            # 推理模式：计算每个节点的异常分数
            reconstruction_loss = torch.tensor(0.0, device=self.device)
            compact_loss = torch.tensor(0.0, device=self.device)
            
            num_nodes = masked_tokens.shape[1]
            anomaly_scores = torch.zeros(num_nodes, device=self.device)
            
            # 并行计算所有节点的异常分数（基于token重构误差）
            reconstruction_errors = torch.nn.functional.mse_loss(
                reconstructed_tokens[0], concated_input_features[0], reduction='none'
            )  # [num_nodes, token_dim]
            
            # 对每个节点的所有维度取平均，得到每个节点的异常分数
            anomaly_scores = reconstruction_errors.mean(dim=1)  # [num_nodes]
            
            # 归一化异常分数到[0,1]区间
            if len(anomaly_scores) > 0:
                min_score = anomaly_scores.min()
                max_score = anomaly_scores.max()
                
                if max_score > min_score:
                    anomaly_scores = (anomaly_scores - min_score) / (max_score - min_score)
                else:
                    # 如果所有分数相同，设置为0.5
                    anomaly_scores = torch.full_like(anomaly_scores, 0.5)

        return emb, anomaly_scores, reconstruction_loss, compact_loss
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

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

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits



class GGADFormer(nn.Module):
    def __init__(self, n_in, n_h, activation, args):
        super(GGADFormer, self).__init__()

        # 设置设备
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
        
        self.gcn1 = GCN(args.embedding_dim, args.embedding_dim, activation)
        self.gcn2 = GCN(args.embedding_dim, args.embedding_dim, activation)

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

        proj_dim = args.proj_dim
        self.token_projection = nn.Linear(2 * proj_dim, args.embedding_dim)

        self.proj_raw = nn.Linear(self.n_in, proj_dim)
        self.proj_prop = nn.Linear(self.n_in, proj_dim)
        self.shared_mlp = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.contrastive_proj_raw = nn.Sequential(nn.Linear(proj_dim, proj_dim // 2), nn.ReLU(), nn.Linear(proj_dim // 2, proj_dim))
        self.contrastive_proj_prop = nn.Sequential(nn.Linear(proj_dim, proj_dim // 2), nn.ReLU(), nn.Linear(proj_dim // 2, proj_dim))

        self.token_decoder = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.ReLU(),
            nn.Linear(args.embedding_dim, 2 * n_in)
        )

        # 重构损失函数
        self.recon_loss_fn = nn.MSELoss()

        # 将模型移动到指定设备
        self.to(self.device)

    def forward(self, input_tokens, adj, normal_for_generation_idx, normal_for_train_idx, train_flag, args, sparse=False):
        # 拆分输入特征
        raw_features = input_tokens[:, :, :self.n_in]
        prop_features = input_tokens[:, :, self.n_in:2*self.n_in]

        # 通过投影和共享MLP得到嵌入
        h_raw = self.shared_mlp(self.proj_raw(raw_features))
        h_prop = self.shared_mlp(self.proj_prop(prop_features))

        # 将不同方面的嵌入拼接后送入transformer
        combined_features = torch.cat([h_raw, h_prop], dim=-1)

        attention_weights = None # 初始化注意力权重
        agg_attention_weights = None 
        emb = self.token_projection(combined_features)
        # emb = self.gcn1(emb, adj, sparse)
        # emb = self.gcn2(emb, adj, sparse)              
        for i, l in enumerate(self.layers):
            emb, current_attention_weights = self.layers[i](emb)
            if i == len(self.layers) - 1: # 拿到最后一层的注意力
                attention_weights = current_attention_weights
                # 聚合多头注意力
                agg_attention_weights = torch.mean(attention_weights, dim=1)
                # agg_attention_weights: [1, num_nodes, num_nodes]
        emb = self.final_ln(emb)
        # emb: [1, num_nodes, hidden_dim]

        # 生成全局中心点
        h_mean = torch.mean(emb, dim=1, keepdim=True)

        outlier_emb = None
        emb_combine = None
        normal_for_generation_emb = emb[:, normal_for_generation_idx, :]
        noise = torch.randn(normal_for_generation_emb.size(), device=self.device) * args.var + args.mean
        noised_normal_for_generation_emb = normal_for_generation_emb + noise
        
        # Add noise into the attribute of sampled abnormal nodes
        # degree = torch.sum(raw_adj[0, :, :], 0)[sample_abnormal_idx]
        # neigh_adj = raw_adj[0, sample_abnormal_idx, :] / torch.unsqueeze(degree, 1)

        # neigh_adj = adj[0, normal_for_generation_idx, :]
        # emb[0, sample_abnormal_idx, :] =self.act(torch.mm(neigh_adj, emb[0, :, :]))
        # emb[0, sample_abnormal_idx, :] = self.fc4(emb[0, sample_abnormal_idx, :])

        # outlier_emb = torch.mm(neigh_adj, emb[0, :, :])
        # outlier_emb = self.act(self.fc4(outlier_emb))

        outlier_emb = self.generate_outliers(emb[:, normal_for_generation_idx, :], emb, args) 
        # outlier_emb: [num_nodes, hidden_dim]
        # emb_con = self.act(self.fc6(emb_con))

        emb_combine = torch.cat((emb[:, normal_for_train_idx, :], torch.unsqueeze(outlier_emb, 0)), 1)

        # TODO ablation study add noise on the selected nodes

        # std = 0.01
        # mean = 0.02
        # noise = torch.randn(emb[:, sample_abnormal_idx, :].size()) * std + mean
        # emb_combine = torch.cat((emb[:, normal_idx, :], emb[:, sample_abnormal_idx, :] + noise), 1)

        # TODO ablation study generate outlier from random noise
        # std = 0.01
        # mean = 0.02
        # emb_con = torch.mm(neigh_adj, emb[0, :, :])
        # noise = torch.randn(emb_con.size()) * std + mean
        # emb_con = self.act(self.fc4(noise))
        # emb_combine = torch.cat((emb[:, normal_idx, :], torch.unsqueeze(emb_con, 0)), 1)
        gna_loss = torch.tensor(0.0, device=emb.device)
        if train_flag:
            # 推拉损失
            h_raw_normal = h_raw[:, normal_for_train_idx, :].squeeze(0)
            h_prop_normal = h_prop[:, normal_for_train_idx, :].squeeze(0)
            z_raw = self.contrastive_proj_raw(h_raw_normal)
            z_prop = self.contrastive_proj_prop(h_prop_normal)
            # L2归一化
            z_raw_norm = F.normalize(z_raw, p=2, dim=1)
            z_prop_norm = F.normalize(z_prop, p=2, dim=1)

            # 计算两个视图间的相似度矩阵 (N_normal x N_normal)
            sim_raw_prop = torch.mm(z_raw_norm, z_prop_norm.T) / args.GNA_temp
            # 计算视图内部的相似度矩阵
            sim_raw_raw = torch.mm(z_raw_norm, z_raw_norm.T) / args.GNA_temp
            sim_prop_prop = torch.mm(z_prop_norm, z_prop_norm.T) / args.GNA_temp

            # 正样本分数是 sim_raw_prop_scaled 的对角线
            positive_pairs = torch.diag(sim_raw_prop)
            # 负样本分数中， raw 对于其他的负样本分数
            exp_sim_raw_prop = torch.exp(sim_raw_prop)
            exp_sim_raw_raw = torch.exp(sim_raw_raw)
            exp_sim_prop_prop = torch.exp(sim_prop_prop)
            
            neg_raw_prop_sum = torch.sum(exp_sim_raw_prop, dim=1) - torch.diag(exp_sim_raw_prop)
            neg_raw_raw_sum = torch.sum(exp_sim_raw_raw, dim=1) - torch.diag(exp_sim_raw_raw)
            denominator_raw_anchor = neg_raw_prop_sum + neg_raw_raw_sum
            loss_raw_anchor = -torch.log(torch.exp(positive_pairs) / denominator_raw_anchor).mean()

            # 负样本分数中， prop 对于其他的负样本分数
            neg_prop_raw_sum = torch.sum(exp_sim_raw_prop, dim=1) - torch.diag(exp_sim_raw_prop)
            neg_prop_prop_sum = torch.sum(exp_sim_prop_prop, dim=1) - torch.diag(exp_sim_prop_prop)
            denominator_prop_anchor = neg_prop_raw_sum + neg_prop_prop_sum
            loss_prop_anchor = -torch.log(torch.exp(positive_pairs) / denominator_prop_anchor).mean()

            gna_loss = (loss_raw_anchor + loss_prop_anchor) * 0.5

            # 对比学习
            # 第一部分，鼓励离群点靠近全局中心点
            # 计算离群点嵌入与全局中心的距离
            outlier_to_center_dist = torch.norm(outlier_emb - h_mean.squeeze(0), p=2, dim=1)
            # 只有超过 confidence_margin 的距离才会产生损失
            margin_excess = outlier_to_center_dist - args.confidence_margin
            con_loss = torch.mean(torch.relu(margin_excess))

            # 重构损失
            # 解码器重构原始输入tokens
            reconstructed_tokens = self.token_decoder(emb)  # [1, num_nodes, input_dim]
        
            # 计算重构损失
            reconstruction_loss = self.recon_loss_fn(reconstructed_tokens, input_tokens)

            f_1 = self.fc1(emb_combine)
        else:
            con_loss = torch.tensor(0.0, device=emb.device)
            reconstruction_loss = torch.tensor(0.0, device=emb.device)
            f_1 = self.fc1(emb)
        f_1 = self.act(f_1)
        f_2 = self.fc2(f_1)
        f_2 = self.act(f_2)
        logits = self.fc3(f_2)
        emb = emb.clone()
        emb[:, normal_for_generation_idx, :] = outlier_emb

        # gna_loss = torch.tensor(0.0, device=emb.device)
        return emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, agg_attention_weights, con_loss, gna_loss, reconstruction_loss    
    def calculate_local_perturbation(self, emb_sampled, full_embeddings, agg_attention_weights, sample_normal_idx, adj, args):
        """
        根据节点的注意力，计算邻居节点可以带来的局部扰动

        Args:
            emb_sampled (Tensor): 采样正常节点的嵌入，形状 [1, num_sampled_nodes, hidden_dim]。
            full_embeddings (Tensor): 所有节点的完整嵌入，形状 [1, num_nodes, hidden_dim]。
            agg_attention_weights (Tensor): 聚合注意力权重矩阵，形状 [1, num_nodes, num_nodes]。
            sample_normal_idx (list or Tensor): 包含要生成离群点的原始正常节点索引的 Python 列表或 Tensor。
            adj (Tensor): 邻接矩阵，形状 [1, num_nodes, num_nodes]，确保是布尔型或0/1。
            args (Namespace): 包含 topk_neighbors_attention 和 hidden_dim。

        Returns:
            Tensor: 聚合的扰动，形状 [1, num_sampled_nodes, hidden_dim]。
        """
        # 将 list 转换为 Tensor
        if isinstance(sample_normal_idx, list):
            sample_normal_idx = torch.tensor(sample_normal_idx, dtype=torch.long, device=full_embeddings.device)

        num_sampled_nodes = sample_normal_idx.numel()
        num_total_nodes = full_embeddings.shape[1]
        hidden_dim = args.hidden_dim
        device = full_embeddings.device

        if num_sampled_nodes == 0:
            return torch.zeros(1, 0, hidden_dim, device=device)

        # adj_matrix 形状: [num_total_nodes, num_total_nodes]
        adj_matrix = adj.squeeze(0).bool() 

        # selected_att_weights 形状: [num_sampled_nodes, num_total_nodes]
        selected_att_weights = agg_attention_weights[0, sample_normal_idx, :]

        # masked_adj_rows 形状: [num_sampled_nodes, num_total_nodes]
        masked_adj_rows = adj_matrix[sample_normal_idx]

        # masked_att_weights 形状: [num_sampled_nodes, num_total_nodes]
        masked_att_weights = selected_att_weights.clone()
        masked_att_weights[~masked_adj_rows] = float('-inf') 

        # 批处理 Top-K 选择
        k = args.topk_neighbors_attention
        k = min(k, num_total_nodes) 

        topk_attention_values, topk_actual_neighbor_indices = torch.topk(masked_att_weights, k=k, dim=1)

        # 过滤无效 Top-K 结果
        is_valid_topk_mask = (topk_attention_values != float('-inf'))

        # 获取 Top-K 邻居的嵌入
        # full_embeddings.squeeze(0) 的形状是 [num_nodes, hidden_dim]
        embeddings_to_gather = full_embeddings.squeeze(0)

        # topk_actual_neighbor_indices 形状: [num_sampled_nodes, k]
        # topk_neighbor_embeddings 形状: [num_sampled_nodes, k, hidden_dim]
        topk_neighbor_embeddings = embeddings_to_gather[topk_actual_neighbor_indices]


        # 批处理聚合：使用加权平均
        topk_attention_values_masked = topk_attention_values.clone()
        topk_attention_values_masked[~is_valid_topk_mask] = 0.0

        sum_topk_attention_values = topk_attention_values_masked.sum(dim=1, keepdim=True)

        normalized_weights = torch.where(
            sum_topk_attention_values > 0,
            topk_attention_values_masked / sum_topk_attention_values,
            torch.zeros_like(topk_attention_values_masked)
        ).unsqueeze(-1)

        aggregated_perturbations = torch.sum(topk_neighbor_embeddings * normalized_weights, dim=1)

        return aggregated_perturbations.unsqueeze(0)
    def generate_outliers(self, normal_embeddings, all_embeddings, args):
        """
        Generate outlier embeddings using the formula: outlier_emb = λ * h_i + (1 - λ) * h_j
        
        Args:
            normal_embeddings: Tensor of shape [1, num_normal_nodes, hidden_dim] 
            all_embeddings: Tensor of shape [1, num_all_nodes, hidden_dim]
            args: Arguments containing outlier generation parameters
            
        Returns:
            outlier_emb: Tensor of shape [num_outliers, hidden_dim]
        """
        num_normal_nodes = normal_embeddings.size(1)
        num_all_nodes = all_embeddings.size(1)
        
        # Number of outliers to generate (same as number of normal nodes for generation)
        num_outliers = num_normal_nodes
        
        # Generate random indices for pairs: i from normal nodes, j from all nodes
        indices_i = torch.arange(num_outliers)
        indices_j = torch.randint(0, num_all_nodes, (num_outliers,))
        
        # Get embeddings h_i from normal nodes and h_j from all nodes
        h_i = normal_embeddings[0, indices_i, :]  # [num_outliers, hidden_dim]
        h_j = all_embeddings[0, indices_j, :]     # [num_outliers, hidden_dim]
        
        # Sample λ from Beta distribution
        # Use α value from args or default to 0.3 if not provided
        alpha = getattr(args, 'outlier_alpha', 0.3)
        beta_dist = Beta(torch.tensor([alpha]), torch.tensor([alpha]))
        lambda_values = beta_dist.sample((num_outliers,)).squeeze(-1).to(normal_embeddings.device)
        
        # Generate outlier embeddings: outlier_emb = λ * h_i + (1 - λ) * h_j
        outlier_emb = lambda_values.unsqueeze(-1) * h_i + (1 - lambda_values.unsqueeze(-1)) * h_j
        
        return outlier_emb
    

class CommunityAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(CommunityAutoencoder, self).__init__()
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU()) # Or nn.Sigmoid() as per ComGA
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space (Community Embedding H)
        self.latent_layer = nn.Linear(in_dim, output_dim)

        # Decoder
        decoder_layers = []
        in_dim = output_dim
        for h_dim in reversed(hidden_dims): # Reverse hidden dims for decoder
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU()) # Or nn.Sigmoid()
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim)) # Output should match input_dim (B)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        h = self.latent_layer(self.encoder(x))
        x_reconstructed = self.decoder(h)
        return h, x_reconstructed
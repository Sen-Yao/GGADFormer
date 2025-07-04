import torch
import torch.nn as nn
import torch.nn.functional as F

from model_GT import SGT, EncoderLayer, FeedForwardNetwork, MultiHeadAttention
from utils import node_neighborhood_feature

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


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


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


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, args):
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn1 = GCN(n_in, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        # self.gcn3 = GCN(n_h, n_h, activation)
        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        self.fc4 = nn.Linear(n_h, n_h, bias=False)
        self.fc6 = nn.Linear(n_h, n_h, bias=False)
        self.fc5 = nn.Linear(n_h, n_in, bias=False)
        self.act = nn.ReLU()
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)
        self.SGT = SGT(n_layers=args.n_layers,
            input_dim=n_in,
            hidden_dim=args.hidden_dim,
            n_class=2,
            num_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            dropout_rate=args.dropout,
            attention_dropout_rate=args.attention_dropout,
            args=args)
        

        # GT only
        self.GT_pre_MLP = nn.Linear(2 * 745, args.hidden_dim)
        encoders = [EncoderLayer(args.hidden_dim, args.ffn_dim, args.dropout, args.attention_dropout, args.n_heads)
                    for _ in range(args.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(args.hidden_dim)

        self.to(args.device)

    def forward(self, seq1, processed_seq1, adj, sample_abnormal_idx, normal_idx, train_flag, args, sparse=False):
        seq1 = seq1.to(args.device)
        adj = adj.to(args.device)
        # 用 GCN 或 SGT

        # GCN：
        #emb, con_loss = self.gcn2(self.gcn1(seq1, adj, sparse), adj, sparse), torch.tensor([0]).to(args.device)

        # GT only
        if True:
            emb = self.GT_pre_MLP(processed_seq1)
            for i, l in enumerate(self.layers):
                emb = self.layers[i](emb)
            # out = torch.split(emb, emb.shape[2] // 2, dim=2)
            # att_emb, hop_emb = out[0], out[1]
            # emb = att_emb * args.alpha + hop_emb * (1 - args.alpha)
            emb = self.final_ln(emb)
            con_loss = torch.tensor([0]).to(args.device)

        # SGT：
        # emb, con_loss = self.SGT(processed_feat)
        # emb = emb.unsqueeze(0).to(args.device)
        
        # print("shape of emb: ", emb.shape)

        emb_con = None
        emb_combine = None
        emb_abnormal = emb[:, sample_abnormal_idx, :]
        
        noise = torch.randn(emb_abnormal.size()) * args.var + args.mean
        emb_abnormal = emb_abnormal + noise.to(args.device)
        # emb_abnormal = emb_abnormal + noise.cuda()
        if train_flag:
            # Add noise into the attribute of sampled abnormal nodes
            # degree = torch.sum(raw_adj[0, :, :], 0)[sample_abnormal_idx]
            # neigh_adj = raw_adj[0, sample_abnormal_idx, :] / torch.unsqueeze(degree, 1)

            neigh_adj = adj[0, sample_abnormal_idx, :]
            # emb[0, sample_abnormal_idx, :] =self.act(torch.mm(neigh_adj, emb[0, :, :]))
            # emb[0, sample_abnormal_idx, :] = self.fc4(emb[0, sample_abnormal_idx, :])

            emb_con = torch.mm(neigh_adj, emb[0, :, :])
            emb_con = self.act(self.fc4(emb_con))
            # emb_con = self.act(self.fc6(emb_con))
            emb_combine = torch.cat((emb[:, normal_idx, :], torch.unsqueeze(emb_con, 0)), 1)

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

            f_1 = self.fc1(emb_combine)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_3 = self.fc3(f_2)
            # f_3 = torch.sigmoid(f_3)
            emb[:, sample_abnormal_idx, :] = emb_con
        else:
            f_1 = self.fc1(emb)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_3 = self.fc3(f_2)
            # f_3 = torch.sigmoid(f_3)

        return emb, emb_combine, f_3, emb_con, emb_abnormal, con_loss

class GGADFormer(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, args):
        super(GGADFormer, self).__init__()
        self.read_mode = readout
        self.gcn1 = GCN(n_in, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        # self.gcn3 = GCN(n_h, n_h, activation)
        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        self.fc4 = nn.Linear(n_h, n_h, bias=False)
        self.fc6 = nn.Linear(n_h, n_h, bias=False)
        self.fc5 = nn.Linear(n_h, n_in, bias=False)
        self.act = nn.ReLU()
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)
        self.SGT = SGT(n_layers=args.n_layers,
            input_dim=n_in,
            hidden_dim=args.hidden_dim,
            n_class=2,
            num_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            dropout_rate=args.dropout,
            attention_dropout_rate=args.attention_dropout,
            args=args)
        

        # GT only
        self.GT_pre_MLP = nn.Linear(2 * 745, args.hidden_dim)
        encoders = [EncoderLayer(args.hidden_dim, args.ffn_dim, args.dropout, args.attention_dropout, args.n_heads)
                    for _ in range(args.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(args.hidden_dim)
        
        # To generate outlier nodes
        self.generate_net = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim), # 输入是 h_p 和 h_attn_weighted_mean 的拼接
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim)
            # nn.Tanh()
        )

        self.to(args.device)

    def forward(self, seq1, processed_seq1, adj, sample_normal_idx, all_normal_idx, train_flag, args, sparse=False):
        seq1 = seq1.to(args.device)
        adj = adj.to(args.device)

        attention_weights = None # 初始化注意力权重
        emb = self.GT_pre_MLP(processed_seq1)
        for i, l in enumerate(self.layers):
            emb, current_attention_weights = self.layers[i](emb)
            if i == len(self.layers) - 1: # 拿到最后一层的注意力
                attention_weights = current_attention_weights
        emb = self.final_ln(emb)
        # emb: [1, num_nodes, hidden_dim]
        # attention_weights: [1, num_heads, num_nodes, num_nodes]

        # 聚合不同注意力头的注意力
        if attention_weights is not None:
            agg_attention_weights = torch.mean(attention_weights, dim=1)
        else:
            agg_attention_weights = None

        emb_con = None
        emb_combine = None
        emb_abnormal = emb[:, sample_normal_idx, :]
        
        noise = torch.randn(emb_abnormal.size()) * args.var + args.mean
        emb_abnormal = emb_abnormal + noise.to(args.device)
        con_loss = torch.tensor([0]).to(args.device)
        if train_flag:
            # 采样出的用于生成离群点的正常节点的上下文表示 h_p，[1, len(sample_normal_idx), hidden_dim]
            h_p = emb[:, sample_normal_idx, :]
            # 传统的均值，直接取平均，即 h_mean = torch.mean(emb, dim=1, keepdim=True)
            # 改进：基于注意力加权的均值。先提取采样节点对应的权重
            selected_query_attention = agg_attention_weights[:, sample_normal_idx, :] # selected_query_attention 形状: [1, len(sample_normal_idx), num_nodes]
            
            # 进行矩阵乘法 (1, len(sample_normal_idx), num_nodes) @ (1, num_nodes, hidden_dim) -> (1, len(sample_normal_idx), hidden_dim)
            # 确保 emb 和 selected_query_attention 都在正确的设备上，并且 batch_size 匹配
            if emb.shape[0] != selected_query_attention.shape[0]:
                h_attn_weighted_mean = torch.bmm(selected_query_attention, emb)
            else:
                h_attn_weighted_mean = torch.bmm(selected_query_attention, emb)

            # 拼接 h_p 和 h_attn_weighted_mean 作为 GenerateNet 的输入，[1, len(sample_normal_idx), 2 * hidden_dim]
            generate_net_input = torch.cat((h_p, h_attn_weighted_mean), dim=2)
            perturbation = self.generate_net(generate_net_input)

            # Enable or disable attention-based local perturbation
            # agg_attention_weights = None # 如果不需要注意力局部扰动，可以将其设置为 None

            # 计算基于注意力局部扰动
            if agg_attention_weights is not None:
                agg_perturbations = self.calculate_local_perturbation(
                    h_p=h_p,
                    full_embeddings=emb, # 传递完整的 emb 以便获取所有节点的特征
                    agg_attention_weights=agg_attention_weights,
                    sample_normal_idx=sample_normal_idx,
                    adj=adj,
                    args=args # 传递 args 以获取 topk_neighbors_attention
                )
            else:
                agg_perturbations = torch.zeros_like(h_p) # 如果没有注意力，局部扰动为0

            
            alpha = args.alpha_outlier_generation
            neigh_adj = adj[0, sample_normal_idx, :]
            # emb_con: [1, len(sample_normal_idx), hidden_dim]
            # 以下为原本的直接通过邻居节点通过 MLP 生成，现已改为通过 perturbation 生成离群点表示
            emb_con_neighbor = self.act(self.fc4(torch.mm(neigh_adj, emb[0, :, :])))
            # 严格按照原文，直接使用 emb_con_neighbor：
            # emb_con = torch.unsqueeze(emb_con_neighbor, 0)
            # 如果 emb_con 不使用注意力，而是类似原文的生成方式：
            # emb_con = h_p + alpha * perturbation - 0 * emb_con_neighbor # emb_con 现在是生成的离群点表示
            # 如果使用注意力：
            emb_con = h_p + 2 * perturbation - 1e-2 * agg_perturbations # emb_con 现在是生成的离群点表示

            # 构建 emb_combine, [1, num_normal_nodes + len(sample_normal_idx), hidden_dim]
            emb_combine = torch.cat((emb[:, all_normal_idx, :], emb_con), 1)


            f_1 = self.fc1(emb_combine)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_3 = self.fc3(f_2)
            # f_3 = torch.sigmoid(f_3)
            # 替换采样节点的嵌入为新生成的离群点嵌入
            # 创建 emb 的一个副本，并在副本上进行修改
            # 这样做可以确保原始的 emb 不被修改，从而允许 PyTorch 正常计算梯度
            emb = emb.clone()
            emb[:, sample_normal_idx, :] = emb_con

            # For loss_contrastive calculation

            # 定义用于对比学习的全局中心：对 h_attn_weighted_mean 求平均
            # new_global_center_for_loss: [1, 1, hidden_dim]
            new_global_center_for_loss = torch.mean(h_attn_weighted_mean, dim=1, keepdim=True)
            global_center_norm = torch.nn.functional.normalize(new_global_center_for_loss, p=2, dim=-1)
            
            emb_norm = torch.nn.functional.normalize(emb, p=2, dim=-1)
            emb_con_norm = torch.nn.functional.normalize(emb_con, p=2, dim=-1)

            # 正常节点与 h_CLS 的相似度
            normal_nodes_emb_norm = emb_norm[:, all_normal_idx, :]
            sim_normal_to_cls = torch.sum(normal_nodes_emb_norm * global_center_norm.expand_as(normal_nodes_emb_norm), dim=-1) / args.temp
            
            mean_pos_sim = torch.mean(sim_normal_to_cls).item()
            std_pos_sim = torch.std(sim_normal_to_cls).item()


            # 正常节点与所有生成的离群点的相似度
            sim_normal_to_outliers = torch.bmm(normal_nodes_emb_norm, emb_con_norm.transpose(1, 2)) / args.temp
            sum_exp_neg_normal = torch.sum(torch.exp(sim_normal_to_outliers), dim=-1)

            mean_neg_sim = torch.mean(sim_normal_to_outliers).item()
            std_neg_sim = torch.std(sim_normal_to_outliers).item()

            # print(f"  Normal-CLS Sim (Positive): Mean={mean_pos_sim:.4f}, Std={std_pos_sim:.4f}")
            # print(f"  Normal-Outlier Sim (Negative): Mean={mean_neg_sim:.4f}, Std={std_neg_sim:.4f}")

            sim_gap_mean = mean_pos_sim - mean_neg_sim
            # print(f"  Mean Sim Gap (Pos - Neg): {sim_gap_mean:.4f}")

            logits_normal_alignment = torch.cat([sim_normal_to_cls.unsqueeze(-1), sim_normal_to_outliers], dim=-1) # [1, len(normal_idx), 1 + len(sample_normal_idx)]
            labels_normal_alignment = torch.zeros(logits_normal_alignment.shape[1], dtype=torch.long).to(args.device).unsqueeze(0) # [1, len(normal_idx)]
            loss_normal_alignment_per_node = -torch.log_softmax(logits_normal_alignment, dim=-1)[:, :, 0]
            L_normal_alignment = torch.mean(loss_normal_alignment_per_node)

            # 构建离群点的负样本集合
            # sim_outlier_to_cls_single 形状 [1, len(sample_normal_idx), 1]
            sim_outlier_to_cls_single = torch.sum(emb_con_norm * global_center_norm.expand_as(emb_con_norm), dim=-1, keepdim=True) / args.temp
            # sim_outlier_to_normals 形状 [1, len(sample_normal_idx), len(normal_idx)]
            sim_outlier_to_normals = torch.bmm(emb_con_norm, normal_nodes_emb_norm.transpose(1, 2)) / args.temp

            # 将 h_CLS_norm 和 normal_nodes_emb_norm 作为离群点的负样本集合
            # logits_outlier_separation 形状 [1, len(sample_normal_idx), 1 + len(normal_idx)]
            logits_outlier_separation = torch.cat([sim_outlier_to_cls_single, sim_outlier_to_normals], dim=-1)

            L_outlier_separation = torch.mean(torch.logsumexp(logits_outlier_separation, dim=-1))
            # 总的对比损失
            con_loss = 1e-3 * L_normal_alignment + 2e-3 * L_outlier_separation
            # 设置 con_loss 为零以 debug
            # con_loss = torch.zeros_like(L_normal_alignment).to(args.device)

            # For loss calculation in main function
            # emb_con [1, len(sample_normal_idx), hidden_dim] -> [len(sample_abnormal_idx), hidden_dim]
            emb_con = emb_con.squeeze(0)
            # con_loss = torch.tensor([0]).to(args.device)
            
        else:
            f_1 = self.fc1(emb)
            f_1 = self.act(f_1)
            f_2 = self.fc2(f_1)
            f_2 = self.act(f_2)
            f_3 = self.fc3(f_2)
            # f_3 = torch.sigmoid(f_3)

        return emb, emb_combine, f_3, emb_con, emb_abnormal, con_loss
    
    def calculate_local_perturbation(self, h_p, full_embeddings, agg_attention_weights, sample_normal_idx, adj, args):
        """
        根据节点的注意力，计算邻居节点可以带来的局部扰动

        Args:
            h_p (Tensor): 采样正常节点的嵌入，形状 [1, num_sampled_nodes, hidden_dim]。
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
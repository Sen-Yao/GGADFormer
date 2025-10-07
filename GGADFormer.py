import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time

from check_gpu_memory import print_gpu_memory_usage, print_tensor_memory, clear_gpu_memory

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

        # 设置批次大小
        self.batchsize = getattr(args, 'batchsize', None)
        
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
        self.read_out = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.token_projection = nn.Linear(self.n_in, args.embedding_dim)

        self.token_decoder = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.ReLU(),
            nn.Linear(args.embedding_dim, (args.pp_k+1) * self.n_in)
        )

        # 重构损失函数
        self.recon_loss_fn = nn.MSELoss()

        # 投影层：将重构误差从2*n_in维度投影到embedding_dim维度
        self.reconstruction_proj = nn.Sequential(
            nn.Linear((args.pp_k+1) * n_in, args.embedding_dim),
            nn.ReLU(),
            nn.Linear(args.embedding_dim, args.embedding_dim)
        )

        # 将模型移动到指定设备
        self.to(self.device)

    def forward(self, input_tokens, adj, _, normal_for_train_idx, train_flag, args, sparse=False):

        # input_tokens: (N, args.pp_k+1, d)
        emb = self.token_projection(input_tokens)
        for i, l in enumerate(self.layers):
            emb, current_attention_weights = self.layers[i](emb)
            if i == len(self.layers) - 1: # 拿到最后一层的注意力
                attention_weights = current_attention_weights
                # 聚合多头注意力
                agg_attention_weights = torch.mean(attention_weights, dim=1)
                # agg_attention_weights: [N, args.pp_k+1, args.pp_k+1]
        emb = self.final_ln(emb)
        # emb: [N, args.pp_k+1, embedding_dim]

        # attention_scores: [N, args.pp_k+1], 表示每个节点对每个hop的注意力分数
        attention_scores = agg_attention_weights[:, 0, :]
        # emb: [1, N, embedding_dim]
        emb = torch.bmm(attention_scores.unsqueeze(1), emb).squeeze(1).unsqueeze(0)

        # 生成全局中心点
        h_mean = torch.mean(emb, dim=1, keepdim=True)

        outlier_emb = None
        emb_combine = None
        noised_normal_for_generation_emb = None
        
        # Add noise into the attribute of sampled abnormal nodes
        # degree = torch.sum(raw_adj[0, :, :], 0)[sample_abnormal_idx]
        # neigh_adj = raw_adj[0, sample_abnormal_idx, :] / torch.unsqueeze(degree, 1)

        # neigh_adj = adj[0, normal_for_generation_idx, :]
        # emb[0, sample_abnormal_idx, :] =self.act(torch.mm(neigh_adj, emb[0, :, :]))
        # emb[0, sample_abnormal_idx, :] = self.fc4(emb[0, sample_abnormal_idx, :])

        # 使用批次大小处理outlier_emb计算
        # outlier_emb: [num_nodes, hidden_dim]
        # emb_con = self.act(self.fc6(emb_con))




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
        proj_loss = torch.tensor(0.0, device=emb.device)
        if train_flag:
            # start_time = time.time()
            # 高效重排
            perm = torch.randperm(normal_for_train_idx.size(0), device=normal_for_train_idx.device)
            normal_for_train_idx = normal_for_train_idx[perm]
            # print(f"time for shuffle:{time.time() - start_time}")
            normal_for_generation_idx = normal_for_train_idx[: int(len(normal_for_train_idx) * args.sample_rate)]            
            normal_for_generation_emb = emb[:, normal_for_generation_idx, :]
            # print(f"time for normal_for_generation_emb:{time.time() - start_time}")
            # Noise
            noise = torch.randn(normal_for_generation_emb.size(), device=self.device) * args.var + args.mean
            noised_normal_for_generation_emb = normal_for_generation_emb + noise
            # print(f"time for noise:{time.time() - start_time}")

            # 重构学习
            reconstructed_tokens = self.token_decoder(emb).squeeze(0)  # [num_nodes, (args.pp_k+1)*n_in]
            reconstruction_error = reconstructed_tokens - input_tokens.view(-1, (args.pp_k+1) * self.n_in)
            # Project reconstruction error to embedding dimension
            reconstruction_error_proj = self.reconstruction_proj(reconstruction_error[normal_for_generation_idx, :])

            outlier_emb = normal_for_generation_emb + args.outlier_alpha * reconstruction_error_proj
            outlier_emb = outlier_emb.squeeze(0)
            # 计算重构损失
            reconstruction_loss = self.recon_loss_fn(reconstructed_tokens, input_tokens.view(-1, (args.pp_k+1) * self.n_in))
            # print(f"time for reconstruction_loss:{time.time() - start_time}")

            # 对比学习
            # 第一部分，鼓励离群点靠近全局中心点
            # 计算离群点嵌入与全局中心的距离
            outlier_to_center_dist = torch.norm(outlier_emb - h_mean.squeeze(0), p=2, dim=1)
            # 只有超过 confidence_margin 的距离才会产生损失
            margin_excess = outlier_to_center_dist - args.confidence_margin
            con_loss = torch.mean(torch.relu(margin_excess))
            # print(f"time for con_loss:{time.time() - start_time}")
            # relative_dist = torch.norm(outlier_emb - normal_for_generation_emb, p=2, dim=1)

            # 约束这个相对距离不能超过一个预设的边距 (margin R)
            # 我们复用 confidence_margin 这个超参数，但它的含义已经改变
            # 变成了伪异常点可以在其“父辈”周围探索的最大半径
            # margin_excess = relative_dist - args.confidence_margin
            # con_loss = torch.mean(torch.relu(margin_excess))

            # 再编码策略
            # 将重构后的 tokens 再编码为 embedding
            reconstructed_tokens = torch.reshape(reconstructed_tokens, (-1, args.pp_k+1, self.n_in))
            reencoded_emb = self.token_projection(reconstructed_tokens)
            for i, l in enumerate(self.layers):
                reencoded_emb, current_attention_weights = self.layers[i](reencoded_emb)
                if i == len(self.layers) - 1: # 拿到最后一层的注意力
                    attention_weights = current_attention_weights
                    # 聚合多头注意力
                    agg_attention_weights = torch.mean(attention_weights, dim=1)
                    # agg_attention_weights: [N, args.pp_k+1, args.pp_k+1]
            reencoded_emb = self.final_ln(reencoded_emb)

            attention_scores = agg_attention_weights[:, 0, :]
            # emb: [1, N, embedding_dim]
            reencoded_emb = torch.bmm(attention_scores.unsqueeze(1), reencoded_emb).squeeze(1)[normal_for_generation_idx, :].detach()
            # 重编码后的 emb 和投影干扰后的 emb 之间的距离需要利用损失函数进行约束
            relative_dist = torch.norm(reencoded_emb - outlier_emb, dim=-1)
            proj_loss = F.relu(relative_dist - args.proj_R_max) + F.relu(args.proj_R_min - relative_dist)
            proj_loss = torch.mean(proj_loss)

            emb_combine = torch.cat((emb[:, normal_for_train_idx, :], torch.unsqueeze(outlier_emb, 0)), 1)

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

        # gna_loss = torch.tensor(0.0, device=emb.device)
        return emb, emb_combine, logits, outlier_emb, noised_normal_for_generation_emb, agg_attention_weights, con_loss, proj_loss, reconstruction_loss    
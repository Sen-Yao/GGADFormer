"""VecGAD model and learning objectives."""

import torch
import torch.nn as nn

from controls import apply_control


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, inputs):
        return self.layer2(self.activation(self.layer1(inputs)))


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout, num_heads):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.scale = 1.0
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        query = self.linear_q(inputs).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(inputs).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(inputs).view(batch_size, -1, self.num_heads, self.head_size)
        query = query.transpose(1, 2) * self.scale
        key = key.transpose(1, 2).transpose(2, 3)
        value = value.transpose(1, 2)
        attention = torch.softmax(torch.matmul(query, key), dim=3)
        output = torch.matmul(self.attention_dropout(attention), value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_size)
        return self.output(output), attention


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout, attention_dropout, num_heads):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, attention_dropout, num_heads)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        update, attention = self.attention(self.attention_norm(inputs))
        inputs = inputs + self.attention_dropout(update)
        inputs = inputs + self.ffn_dropout(self.ffn(self.ffn_norm(inputs)))
        return inputs, attention


class GCN(nn.Module):
    """Initialization-compatible layer retained from the formal model definition."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        self.activation = nn.PReLU()
        self.bias = nn.Parameter(torch.zeros(output_size))
        nn.init.xavier_uniform_(self.fc.weight)


class VecGAD(nn.Module):
    def __init__(self, feature_dim, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.feature_dim = feature_dim

        # Preserve the formal source's module initialization order. These layers
        # are part of the frozen model definition even though this path does not
        # call the graph-convolution branches.
        self.gcn1 = GCN(args.embedding_dim, args.embedding_dim)
        self.gcn2 = GCN(args.embedding_dim, args.embedding_dim)
        self.fc1 = nn.Linear(args.embedding_dim, args.embedding_dim // 2, bias=False)
        self.fc2 = nn.Linear(args.embedding_dim // 2, args.embedding_dim // 4, bias=False)
        self.fc3 = nn.Linear(args.embedding_dim // 4, 1, bias=False)
        self.fc4 = nn.Linear(args.embedding_dim, args.embedding_dim, bias=False)
        self.activation = nn.ReLU()

        self.layers = nn.ModuleList(
            EncoderLayer(
                args.embedding_dim,
                args.ffn_dim,
                args.dropout,
                args.attention_dropout,
                args.num_heads,
            )
            for _ in range(args.num_layers)
        )
        self.final_norm = nn.LayerNorm(args.embedding_dim)
        self.read_out = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.token_projection = nn.Linear(feature_dim, args.embedding_dim)
        flattened_dim = (args.pp_k + 1) * feature_dim
        self.token_decoder = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.ReLU(),
            nn.Linear(args.embedding_dim, flattened_dim),
        )
        self.reconstruction_projection = nn.Sequential(
            nn.Linear(flattened_dim, args.embedding_dim),
            nn.ReLU(),
            nn.Linear(args.embedding_dim, args.embedding_dim),
        )
        self.reconstruction_loss = nn.MSELoss()

        direction_seed = args.seed * 1000003 + 1729
        magnitude_seed = args.seed * 1000003 + 7919
        self.direction_generator = torch.Generator(device=device)
        self.direction_generator.manual_seed(direction_seed)
        self.magnitude_generator = torch.Generator(device=device)
        self.magnitude_generator.manual_seed(magnitude_seed)
        self.to(device)

    def encode(self, tokens):
        encoded = self.token_projection(tokens)
        final_attention = None
        for layer in self.layers:
            encoded, final_attention = layer(encoded)
        encoded = self.final_norm(encoded)
        attention = final_attention.mean(dim=1)
        zero_hop_query = attention[:, 0, :]
        readout = torch.bmm(zero_hop_query.unsqueeze(1), encoded).squeeze(1)
        return readout.unsqueeze(0)

    def score(self, tokens):
        return self.classify(self.encode(tokens)).squeeze(0).squeeze(-1)

    def classify(self, embeddings):
        hidden = self.activation(self.fc1(embeddings))
        hidden = self.activation(self.fc2(hidden))
        return self.fc3(hidden)

    def training_objectives(self, tokens, normal_indices):
        embeddings = self.encode(tokens)
        center = embeddings.mean(dim=1, keepdim=True)

        order = torch.randperm(normal_indices.numel(), device=normal_indices.device)
        normal_indices = normal_indices[order]
        source_count = int(normal_indices.numel() * self.args.sample_rate)
        if source_count == 0:
            raise ValueError("the sampled batch contains too few labeled normal nodes")
        source_indices = normal_indices[:source_count]
        source_embeddings = embeddings[:, source_indices, :]

        # This draw is retained because it is part of the formal training RNG trace.
        noise = torch.randn(source_embeddings.shape, device=self.device)
        _ = source_embeddings + noise * self.args.noise_std + self.args.noise_mean

        reconstructed = self.token_decoder(embeddings).squeeze(0)
        flattened = tokens.reshape(tokens.shape[0], -1)
        discrepancy = reconstructed - flattened
        projected = self.reconstruction_projection(discrepancy[source_indices])
        projected = apply_control(
            projected,
            self.args.control,
            self.direction_generator,
            self.magnitude_generator,
        )
        outliers = (source_embeddings + self.args.outlier_beta * projected).squeeze(0)

        distances = torch.linalg.vector_norm(outliers - center.squeeze(0), dim=1)
        hsc_loss = (
            torch.relu(self.args.ring_R_min - distances)
            + torch.relu(distances - self.args.ring_R_max)
        ).mean()

        reconstructed_tokens = reconstructed.reshape(-1, self.args.pp_k + 1, self.feature_dim)
        reencoded = self.encode(reconstructed_tokens)[:, source_indices, :].detach().squeeze(0)
        token_loss = self.reconstruction_loss(reconstructed, flattened)
        embedding_loss = torch.linalg.vector_norm(
            source_embeddings.squeeze(0) - reencoded, dim=1
        ).mean()
        reconstruction_loss = (
            self.args.lambda_rec_tok * token_loss
            + self.args.lambda_rec_emb * embedding_loss
        )

        combined = torch.cat((embeddings[:, normal_indices, :], outliers.unsqueeze(0)), dim=1)
        logits = self.classify(combined)
        return logits, reconstruction_loss, hsc_loss, source_count

import torch
from torch.nn import Embedding, MultiheadAttention, LayerNorm, Sequential, ModuleList, ReLU, Dropout
from torch.nn import functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout):
        super().__init__()
        self.embeddings = Embedding(num_embeddings, dim)
        self.position_embeddings = Embedding(num_max_positions, dim)
        self.dropout = Dropout(dropout)

        self.attentions = ModuleList()
        self.layer_norms_1 = ModuleList()
        self.feed_forwards = ModuleList()
        self.layer_norms_2 = ModuleList()
        for _ in range(num_layers):
            self.attentions.append(MultiheadAttention(dim, num_heads, dropout=dropout))
            self.layer_norms_1.append(LayerNorm(dim, eps=1e-12))
            self.feed_forwards.append(Sequential(Linear(dim, hidden_dim), ReLU(), Linear(hidden_dim, dim)))
            self.layer_norms_2.append(LayerNorm(dim, eps=1e-12))

    def forward(self, x):
        positions = torch.arange(x.size(-1), device=x.device).unsqueeze(0)
        embeddings_output = self.embeddings(x) + self.position_embeddings(positions).expand_as(x)
        embeddings_output = self.dropout(embeddings_output)

        for attention, layer_norm_1, feed_forward, layer_norm2 in zip(self.attentions, self.layer_norms_1, self.feed_forwards, self.layer_norms_2):
            attention_output = attention(embeddings_output, attn_mask, need_weights=False)
            attention_output = self.dropout(attention_output)
            middle_output = layer_norm_1(embeddings_output + attention_output)
            feed_forward_output = feed_forward(middle_output)
            output = layer_norm2(middle_output + feed_forward_output)
        return output

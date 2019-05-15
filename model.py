import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, num_embeddings, hidden_dim, num_heads, layers, attention_dropout):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings, hidden_dim)
        attentions = [nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout) for _ in range(layers)]
        self.attentions = nn.ModuleList(attentions)

    def forward(self, xb):
        return xb @ self.weights + self.bias

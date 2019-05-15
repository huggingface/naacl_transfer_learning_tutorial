import torch
from torch.nn import Embedding, MultiheadAttention, LayerNorm, Linear, Sequential, ModuleList, ReLU, Dropout
from torch.nn import functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings, dim)
        self.position_embeddings = Embedding(num_max_positions, dim)
        self.dropout = Dropout(dropout)

        self.attentions, self.ln_1, self.feed_forwards, self.ln_2 = ModuleList(), ModuleList(), ModuleList(), ModuleList()
        for _ in range(num_layers):
            self.attentions.append(MultiheadAttention(dim, num_heads, dropout=dropout))
            self.ln_1.append(LayerNorm(dim, eps=1e-12))
            self.feed_forwards.append(Sequential(Linear(dim, hidden_dim), ReLU(), Linear(hidden_dim, dim)))
            self.ln_2.append(LayerNorm(dim, eps=1e-12))

    def forward(self, x):
        seq_length = x.size(-1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        embeddings_output = self.token_embeddings(x) + self.position_embeddings(positions).expand_as(x)
        embeddings_output = self.dropout(embeddings_output)

        causal_attention_mask = torch.triu(torch.full((seq_length, seq_length), -float('Inf'))).view(1, 1, seq_length, seq_length)
        for attention, layer_norm_1, feed_forward, layer_norm2 in zip(self.attentions, self.ln_1, self.feed_forwards, self.ln_2):
            attention_output = attention(embeddings_output, causal_attention_mask, need_weights=False)
            attention_output = self.dropout(attention_output)
            middle_output = layer_norm_1(embeddings_output + attention_output)

            feed_forward_output = feed_forward(middle_output)
            output = layer_norm2(middle_output + feed_forward_output)

        return output

class SimpleTransformerWithLMHead(nn.Module):
    def __init__(self, dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout):
        super().__init__()
        self.transformer = SimpleTransformer(dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout)
        self.lm_head = Linear(dim, num_embeddings, bias=False)
        self.lm_head.weight = self.transformer.token_embeddings.weight

    def forward(self, x, labels=None):
        hidden_states = self.transformer(x)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            return logits, loss

        return logits

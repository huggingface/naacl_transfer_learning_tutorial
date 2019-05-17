import torch
from torch.nn import Module, Embedding, MultiheadAttention, LayerNorm, Linear, Sequential, ModuleList, ReLU, Dropout
from torch.nn import functional as F


class SimpleTransformer(Module):
    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings, embed_dim)
        self.position_embeddings = Embedding(num_max_positions, embed_dim)
        self.dropout = Dropout(dropout)

        self.ln_1, self.attentions, self.ln_2, self.feed_forwards = ModuleList(), ModuleList(), ModuleList(), ModuleList()
        for _ in range(num_layers):
            self.ln_1.append(LayerNorm(embed_dim, eps=1e-12))
            self.attentions.append(MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.ln_2.append(LayerNorm(embed_dim, eps=1e-12))
            self.feed_forwards.append(Sequential(Linear(embed_dim, hidden_dim), ReLU(), Linear(hidden_dim, embed_dim)))

    def forward(self, x):
        seq_length = x.size(-1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        token_embeddings = self.token_embeddings(x)
        h = token_embeddings + self.position_embeddings(positions).expand_as(token_embeddings)
        h = self.dropout(h)

        causal_attention_mask = torch.triu(torch.full((seq_length, seq_length), -float('Inf'))).view(1, 1, seq_length, seq_length)
        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.ln_1, self.attentions, self.ln_2, self.feed_forwards):
            h = layer_norm_1(h)
            x = attention(h, causal_attention_mask, need_weights=False)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h

        return h

class SimpleTransformerWithLMHead(Module):
    def __init__(self, args):
        super().__init__()
        self.transformer = SimpleTransformer(args.embed_dim,
                                             args.hidden_dim,
                                             args.num_embeddings,
                                             args.num_max_positions,
                                             args.num_heads,
                                             args.num_layers,
                                             args.dropout)
        self.lm_head = Linear(args.embed_dim, args.num_embeddings, bias=False)
        self.lm_head.weight = self.transformer.token_embeddings.weight

    def forward(self, x, labels=None):
        hidden_states = self.transformer(x)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return logits, loss

        return logits

import torch
from torch.nn import Module, Embedding, MultiheadAttention, LayerNorm, Linear, Sequential, ModuleList, ReLU, Dropout, CrossEntropyLoss
from torch.nn import functional as F


class Transformer(Module):
    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings, embed_dim)
        self.position_embeddings = Embedding(num_max_positions, embed_dim)
        self.dropout = Dropout(dropout)
        self.attentions, self.feed_forwards, self.ln_1, self.ln_2 = ModuleList(), ModuleList(), ModuleList(), ModuleList()
        for _ in range(num_layers):
            self.attentions.append(MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(Sequential(Linear(embed_dim, hidden_dim), ReLU(), Linear(hidden_dim, embed_dim)))
            self.ln_1.append(LayerNorm(embed_dim, eps=1e-12))
            self.ln_2.append(LayerNorm(embed_dim, eps=1e-12))

    def forward(self, x):
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.token_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        causal_attn_mask = torch.triu(torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype), diagonal=1)
        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.ln_1, self.attentions, self.ln_2, self.feed_forwards):
            assert torch.isnan(h).sum().item() == 0
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=causal_attn_mask, need_weights=False)
            x = self.dropout(x)
            h = x + h
            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h


class TransformerWithLMHead(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config.embed_dim,
                                       config.hidden_dim,
                                       config.num_embeddings,
                                       config.num_max_positions,
                                       config.num_heads,
                                       config.num_layers,
                                       config.dropout)
        self.lm_head = Linear(config.embed_dim, config.num_embeddings, bias=False)
        self.lm_head.weight = self.transformer.token_embeddings.weight  # Tie weights
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (Linear, Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, MultiheadAttention):
            module.out_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.in_proj_weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.out_proj.bias.data.zero_()
            module.in_proj_bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, labels=None):
        hidden_states = self.transformer(x)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = logits[:-1]
            shift_labels = labels[1:]
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return logits, loss

        return logits

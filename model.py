import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

def create_sinusoidal_embeddings(position_embeddings):
    """ From https://github.com/facebookresearch/XLM/blob/master/src/model/transformer.py """
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / position_embeddings.embedding_dim) for j in range(position_embeddings.embedding_dim)]
        for pos in range(position_embeddings.num_embeddings)
    ])
    position_embeddings.weight[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    position_embeddings.weight[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    position_embeddings.weight.detach_()
    position_embeddings.weight.requires_grad = False

class Transformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_embed, num_pos, num_heads, num_layers, dropout, sinusoidal_embeddings):
        """ Transformer (GPT-2 architecture) """
        super().__init__()
        self.tokens_embeddings = nn.Embedding(num_embed, embed_dim)
        self.position_embeddings = nn.Embedding(num_pos, embed_dim)
        if sinusoidal_embeddings:
            create_sinusoidal_embeddings(self.position_embeddings)
        self.dropout = nn.Dropout(dropout)

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                 	nn.ReLU(),
                                                 	nn.Linear(hidden_dim, embed_dim)))
            self.layer_norms_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    def forward(self, x):
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms_1, self.attentions,
                                                                       self.layer_norms_2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h


class TransformerWithLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                       config.num_max_positions, config.num_heads, config.num_layers,
                                       config.dropout, config.sinusoidal_embeddings)
        self.lm_head = nn.Linear(config.embed_dim, config.num_embeddings, bias=False)
        self.lm_head.weight = self.transformer.tokens_embeddings.weight  # Tie weights
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.MultiheadAttention):
            module.out_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.in_proj_weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.out_proj.bias.data.zero_()
            module.in_proj_bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, labels=None):
        hidden_states = self.transformer(x)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = logits[:-1]
            shift_labels = labels[1:]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return logits, loss

        return logits

    def add_new_tokens(self, num_new_tokens):
        " Update token embeddings to add new tokens "
        # Build new embeddings and initialize all new embeddings (in particular the new tokens)
        old = self.transformer.tokens_embeddings
        self.transformer.tokens_embeddings = nn.Embedding(old.num_embeddings + num_new_tokens, old.embedding_dim)
        self.transformer.tokens_embeddings.to(old.weight.device)
        self.init_weights(self.transformer.tokens_embeddings)
        # Copy word embeddings from the previous weights
        self.transformer.tokens_embeddings.weight.data[:old.num_embeddings, :] = old.weight.data[:old.num_embeddings, :]
        # Tie weights again
        self.lm_head.weight = self.transformer.tokens_embeddings.weight

import importlib
import torch
import torch.nn as nn

from pretraining_model import Transformer

class TransformerWithAdapters(Transformer):
    def __init__(self, adapted_dim, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout):
        """ Transformer with adapters (small bottleneck layers) """
        super().__init__(embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout)
        self.adapters_1 = nn.ModuleList()
        self.adapters_2 = nn.ModuleList()
        for _ in range(num_layers):
            self.adapters_1.append(nn.Sequential(nn.Linear(embed_dim, adapted_dim),
                                               nn.ReLU(),
                                               nn.Linear(adapted_dim, embed_dim)))
            self.adapters_2.append(nn.Sequential(nn.Linear(embed_dim, adapted_dim),
                                               nn.ReLU(),
                                               nn.Linear(adapted_dim, embed_dim)))

    def forward(self, x):
        """ Input has shape [seq length, batch] """
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        for l in range(len(self.layer_norm_1)):
            h = self.layer_norms_1[l](h)
            x, _ = self.attentions[l](h, h, h, attn_mask=attn_mask, need_weights=False)
            x = self.dropout(x)
            x = self.adapters_1[l](x)  # Add an adapter after attention
            h = x + h

            h = self.layer_norms_2[l](h)
            x = self.feed_forwards[l](h)
            x = self.dropout(x)
            x = self.adapters_2[l](x)  # Add an adapter after feed-forward
            h = x + h
        return h


class TransformerWithClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        """ Transformer with a classification head on top """
        self.config = config
        self.transformer = Transformer(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                       config.num_max_positions, config.num_heads, config.num_layers,
                                       config.dropout)
        self.classification_head = nn.Linear(config.embed_dim, config.num_classes)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.MultiheadAttention):
            module.out_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.in_proj_weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.out_proj.bias.data.zero_()
            module.in_proj_bias.data.zero_()
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, labels=None):
        """ Input has shape [seq length, batch] """
        hidden_states = self.transformer(x)
        logits = self.classification_head(hidden_states[-1])

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss

        return logits


class TransformerWithClassificationAndLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        """ Transformer with a classification and a language modeling head on top """
        self.config = config
        self.transformer = Transformer(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                       config.num_max_positions, config.num_heads, config.num_layers,
                                       config.dropout)
        self.classification_head = nn.Linear(config.embed_dim, config.num_classes)
        self.lm_head = nn.Linear(config.embed_dim, config.num_embeddings, bias=False)
        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.transformer.tokens_embeddings.weight

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.MultiheadAttention):
            module.out_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.in_proj_weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.out_proj.bias.data.zero_()
            module.in_proj_bias.data.zero_()
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, lm_labels=None, clf_labels=None):
        """ Input has shape [seq length, batch] """
        hidden_states = self.transformer(x)
        lm_logits = self.classification_head(hidden_states)
        clf_logits = self.classification_head(hidden_states[-1])

        loss = []
        if lm_labels is not None:
            shift_logits = lm_logits[:-1]
            shift_labels = lm_labels[1:]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss.append(loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)))

        if clf_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss.append(loss_fct(logits.view(-1, logits.size(-1)), clf_labels.view(-1)))

        if len(loss):
            return (lm_logits, clf_logits), loss

        return lm_logits, clf_logits

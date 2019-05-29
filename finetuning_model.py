import importlib
import torch
import torch.nn as nn

from pretraining_model import Transformer, TransformerWithLMHead

class TransformerWithAdapters(Transformer):
    def __init__(self, adapters_dim, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout):
        """ Transformer with adapters (small bottleneck layers) """
        super().__init__(embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout)
        self.adapters_1 = nn.ModuleList()
        self.adapters_2 = nn.ModuleList()
        for _ in range(num_layers):
            self.adapters_1.append(nn.Sequential(nn.Linear(embed_dim, adapters_dim),
                                                 nn.ReLU(),
                                                 nn.Linear(adapters_dim, embed_dim)))
            self.adapters_2.append(nn.Sequential(nn.Linear(embed_dim, adapters_dim),
                                                 nn.ReLU(),
                                                 nn.Linear(adapters_dim, embed_dim)))

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


class TransformerWithClfHead(TransformerWithLMHead):
    def __init__(self, config, fine_tuning_config):
        """ Transformer with a classification head and a language modeling head on top and optionally adapters. """
        super().__init__(config)
        if fine_tuning_config.adapters_dim > 0:
            self.transformer = TransformerWithAdapters(fine_tuning_config.adapters_dim, config.embed_dim, config.hidden_dim,
                                                       config.num_embeddings, config.num_max_positions, config.num_heads,
                                                       config.num_layers, config.dropout)
        self.classification_head = nn.Linear(config.embed_dim, fine_tuning_config.num_classes)
        self.apply(self.init_weights)

    def forward(self, x, lm_labels=None, clf_labels=None):
        """ Input has shape [seq length, batch] """
        hidden_states = self.transformer(x)
        lm_logits = self.lm_head(hidden_states)
        clf_logits = self.classification_head(hidden_states[-1])

        loss = []
        if clf_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss.append(loss_fct(clf_logits.view(-1, clf_logits.size(-1)), clf_labels.view(-1)))

        if lm_labels is not None:
            shift_logits = lm_logits[:-1]
            shift_labels = lm_labels[1:]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss.append(loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)))

        if len(loss):
            return (lm_logits, clf_logits), loss

        return lm_logits, clf_logits

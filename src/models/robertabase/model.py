import torch
from torch import nn
from torchcrf import CRF
from transformers import AutoModel
from transformers.utils import logging as hf_logging

from .config import DROPOUT, ENCODER_NAME, FREEZE_ENCODER, TASKS

hf_logging.set_verbosity_error()

# for testing mean-pooling
class MeanPool(nn.Module):
    """Simplest possible sentence pooling: average all word vectors,
    ignoring padding. No learned parameters — this is the ablation
    baseline CNNPool is compared against."""
    def forward(self, words, word_mask):
        mask = word_mask.unsqueeze(-1).float()
        summed = (words * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)  # avoid divide-by-zero on empty rows
        return summed / counts

# final best choice for sentence pooling is SelfAttentionPool, which is in model.py
class SelfAttentionPool(nn.Module):
    """Lin et al. 2017 — single-hop simplification for classification use.
    Learns a per-word importance score, then takes a weighted average.
    Far fewer parameters than CNNPool (~1K vs ~885K), directly targeting
    the overfitting-on-minority-classes problem observed in the CNN ablation.
    """
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)  # one score per word

    def forward(self, words, word_mask):
        scores = self.attn(words).squeeze(-1)                      # [batch, seq_len]
        scores = scores.masked_fill(~word_mask.bool(), float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)        # [batch, seq_len, 1]
        return (words * weights).sum(dim=1)                          # [batch, hidden_dim]

    def get_attention_weights(self, words, word_mask):
        """For the interpretability visualization in your report."""
        scores = self.attn(words).squeeze(-1)
        scores = scores.masked_fill(~word_mask.bool(), float('-inf'))
        return torch.softmax(scores, dim=1)

# for testing bi-GRU 
class BiGRUEncoder(nn.Module):
    """Enriches word vectors with sequential, bidirectional context.
    Outputs a sequence of contextualized vectors matching the original shape.
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # Project 2 * hidden_size back down to hidden_size to match existing head dimensions
        self.projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, words, word_mask):
        # Pack/pad sequences if sentences vary wildly, but given your word_mask 
        # and existing setup, passing directly with mask handling is highly effective.
        gru_out, _ = self.gru(words) # Shape: (batch, seq_len, 2 * hidden_size)
        gru_out = self.dropout(gru_out)
        
        # Scale back to (batch, seq_len, hidden_size)
        enriched_words = self.projection(gru_out)
        
        # Apply the mask to zero out padding representations caused by the GRU calculations
        mask = word_mask.unsqueeze(-1).float()
        return enriched_words * mask
    
class RobertaBase(nn.Module):
    """No-CNN ablation baseline: RoBERTa -> mean-pool -> heads, and
    RoBERTa word vectors -> CRF directly for NER (no enrichment step).
    """
    def __init__(self, n_classes: dict[str, int], encoder_name: str = ENCODER_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if FREEZE_ENCODER:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(DROPOUT)
        self.pools = nn.ModuleDict({
            task: SelfAttentionPool(hidden_dim=hidden) for task in TASKS
        })
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, n_classes[task]) for task in TASKS})

        self.ner_head = nn.Linear(hidden, n_classes["ner"])  # straight from RoBERTa, no CNN in between
        self.crf = CRF(n_classes["ner"], batch_first=True)

    @classmethod
    def from_labels(cls, labels: dict[str, list[str]], encoder_name: str = ENCODER_NAME):
        return cls({task: len(names) for task, names in labels.items()}, encoder_name)

    def forward(self, input_ids, attention_mask, word_index, word_mask):
        subwords = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        index = word_index.unsqueeze(-1).expand(-1, -1, subwords.size(-1))
        words = self.dropout(subwords.gather(1, index))\

        predictions = {}
        for task in TASKS:
            task_pooled = self.pools[task](words, word_mask)
            predictions[task] = self.heads[task](self.dropout(task_pooled))

        predictions["ner"] = self.ner_head(words)  # no CNNEnrich step — direct comparison point
        return predictions
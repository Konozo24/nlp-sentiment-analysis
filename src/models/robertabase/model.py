import torch
from torch import nn
from torchcrf import CRF
from transformers import AutoModel
from transformers.utils import logging as hf_logging

from .config import DROPOUT, ENCODER_NAME, FREEZE_ENCODER, TASKS

hf_logging.set_verbosity_error()

class MeanPool(nn.Module):
    """Reference pooling layer for testing a non-learned mean baseline."""

    def forward(self, words: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
        mask = word_mask.unsqueeze(-1).float()
        summed = (words * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts


class SelfAttentionPool(nn.Module):
    """Learn a per-word importance score and return a weighted average."""

    def __init__(self, hidden_dim: int = 768) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, words: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(words).squeeze(-1)                      # [batch, seq_len]
        scores = scores.masked_fill(~word_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)        # [batch, seq_len, 1]
        return (words * weights).sum(dim=1)                          # [batch, hidden_dim]

    def get_attention_weights(self, words: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
        """Return normalized word importance scores for interpretation."""
        scores = self.attn(words).squeeze(-1)
        scores = scores.masked_fill(~word_mask.bool(), float("-inf"))
        return torch.softmax(scores, dim=1)


class BiGRUEncoder(nn.Module):
    """Reference word-level encoder for testing a BiGRU alternative."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, words: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(words)
        gru_out = self.dropout(gru_out)
        enriched_words = self.projection(gru_out)
        mask = word_mask.unsqueeze(-1).float()
        return enriched_words * mask


class RobertaBase(nn.Module):
    """RoBERTa with attention pooling for classification and CRF-based NER."""

    def __init__(self, n_classes: dict[str, int], encoder_name: str = ENCODER_NAME) -> None:
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

        self.ner_head = nn.Linear(hidden, n_classes["ner"])
        self.crf = CRF(n_classes["ner"], batch_first=True)

    @classmethod
    def from_labels(
        cls, labels: dict[str, list[str]], encoder_name: str = ENCODER_NAME
    ) -> "RobertaBase":
        return cls({task: len(names) for task, names in labels.items()}, encoder_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_index: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        subwords = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        index = word_index.unsqueeze(-1).expand(-1, -1, subwords.size(-1))
        words = self.dropout(subwords.gather(1, index))

        predictions = {}
        for task in TASKS:
            task_pooled = self.pools[task](words, word_mask)
            predictions[task] = self.heads[task](self.dropout(task_pooled))

        predictions["ner"] = self.ner_head(words)
        return predictions
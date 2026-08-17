import torch
from torch import nn
from torchcrf import CRF
from transformers import AutoModel
from transformers.utils import logging as hf_logging

from .config import DROPOUT, ENCODER_NAME, FREEZE_ENCODER, KERNEL_SIZES, NER_FILTERS, NER_KERNEL_SIZES, NUM_FILTERS, TASKS

hf_logging.set_verbosity_error()


class CNNPool(nn.Module):
    """Multi-kernel 1D CNN + max-pool: collapses a word sequence into one vector.
    Used for the sentence-level tasks (sentiment/topic/emotion)."""
    def __init__(self, in_dim, num_filters, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_dim, num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.out_dim = num_filters * len(kernel_sizes)

    def forward(self, words, word_mask):
        x = (words * word_mask.unsqueeze(-1)).transpose(1, 2)  # zero padded words first
        pooled = [torch.max(torch.relu(conv(x)), dim=2)[0] for conv in self.convs]
        return torch.cat(pooled, dim=1)


class CNNEnrich(nn.Module):
    """Same-padding 1D CNN that keeps one vector PER WORD (no pooling).
    Used for NER, which needs per-word scores to feed the CRF."""
    def __init__(self, in_dim, num_filters, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_dim, num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.proj = nn.Linear(num_filters * len(kernel_sizes), in_dim)

    def forward(self, words, word_mask):
        x = (words * word_mask.unsqueeze(-1)).transpose(1, 2)
        outs = [torch.relu(conv(x)) for conv in self.convs]
        x = torch.cat(outs, dim=1).transpose(1, 2)
        return self.proj(x)


class RobertaCNN(nn.Module):
    def __init__(self, n_classes: dict[str, int], encoder_name: str = ENCODER_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if FREEZE_ENCODER:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(DROPOUT)

        self.cnn_pool = CNNPool(hidden, NUM_FILTERS, KERNEL_SIZES)
        self.heads = nn.ModuleDict({task: nn.Linear(self.cnn_pool.out_dim, n_classes[task]) for task in TASKS})

        self.ner_cnn = CNNEnrich(hidden, NER_FILTERS, NER_KERNEL_SIZES)
        self.ner_head = nn.Linear(hidden, n_classes["ner"])
        self.crf = CRF(n_classes["ner"], batch_first=True)

    @classmethod
    def from_labels(cls, labels: dict[str, list[str]], encoder_name: str = ENCODER_NAME):
        return cls({task: len(names) for task, names in labels.items()}, encoder_name)

    def forward(self, input_ids, attention_mask, word_index, word_mask):
        subwords = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        index = word_index.unsqueeze(-1).expand(-1, -1, subwords.size(-1))
        words = self.dropout(subwords.gather(1, index))

        pooled = self.cnn_pool(words, word_mask)
        predictions = {task: head(self.dropout(pooled)) for task, head in self.heads.items()}

        enriched = self.ner_cnn(words, word_mask)
        predictions["ner"] = self.ner_head(self.dropout(enriched))
        return predictions
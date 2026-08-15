import torch
from torch import nn
from torchcrf import CRF
from transformers import AutoModel
from transformers.utils import logging as hf_logging

from .config import DROPOUT, ENCODER_NAME, FREEZE_ENCODER, HIDDEN_SIZE, TASKS

# silence the load-time "UNEXPECTED keys" report: expected every run
hf_logging.set_verbosity_error()


class TRABSA(nn.Module):
    def __init__(self, n_classes: dict[str, int], encoder_name: str = ENCODER_NAME):
        """n_classes maps every task name (TASKS + 'ner') to its number of classes."""
        super().__init__()

        # core encooder setup
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if FREEZE_ENCODER:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.encoder.config.hidden_size,  # 768
            hidden_size=HIDDEN_SIZE,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(DROPOUT)
        lstm_out = HIDDEN_SIZE * 2

        self.attention = nn.Linear(lstm_out, 1)
        self.heads = nn.ModuleDict({task: nn.Linear(lstm_out, n_classes[task]) for task in TASKS})
        self.ner_head = nn.Linear(lstm_out, n_classes["ner"])
        self.crf = CRF(n_classes["ner"], batch_first=True)

    @classmethod
    def from_labels(cls, labels: dict[str, list[str]], encoder_name: str = ENCODER_NAME):
        """Build a model sized for the class lists that train.py saved."""
        return cls({task: len(names) for task, names in labels.items()}, encoder_name)

    def forward(self, input_ids, attention_mask, word_index, word_mask):
        subwords = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Keep the first subword of each word -> one vector per word.
        index = word_index.unsqueeze(-1).expand(-1, -1, subwords.size(-1))
        words = subwords.gather(1, index)  # (batch, words, 768)

        outputs, _ = self.lstm(self.dropout(words))  # (batch, words, 512)
        outputs = self.dropout(outputs)

        # Attention pooling: score each word, ignore padding, weighted sum.
        scores = self.attention(outputs).masked_fill(~word_mask.unsqueeze(-1), float("-inf"))
        pooled = (outputs * torch.softmax(scores, dim=1)).sum(dim=1)  # (batch, 512)

        predictions = {task: head(pooled) for task, head in self.heads.items()}
        predictions["ner"] = self.ner_head(outputs)  # per-word scores, fed to the CRF
        return predictions

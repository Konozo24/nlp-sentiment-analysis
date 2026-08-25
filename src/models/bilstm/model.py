"""The BiLSTM architecture.

    word ids
      -> frozen embedding table   (fastText: pretrained 300d + in-domain 100d)
      -> Linear + ReLU + Dropout  (trainable projection into the LSTM's space)
      -> BiLSTM                   (one contextual vector per word)
      -> attention pooling        (weighted sum -> one vector per tweet)
      -> sentiment / emotion / topic heads   (from the pooled vector)
      -> NER head + CRF                      (from the per-word vectors)

One forward pass, four outputs.
"""

import torch
from torch import nn
from torchcrf import CRF

from .config import DROPOUT, FREEZE_EMBEDDINGS, HIDDEN_SIZE, PROJ_DIM, TASKS


class BiLSTM(nn.Module):
    def __init__(self, n_classes: dict[str, int], embeddings: torch.Tensor) -> None:
        super().__init__()
        
        self.embedding = nn.Embedding.from_pretrained(
            embeddings.float(), freeze=FREEZE_EMBEDDINGS, padding_idx=0
        )

        self.projection = nn.Sequential(
            nn.Linear(embeddings.size(1), PROJ_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        self.lstm = nn.LSTM(
            input_size=PROJ_DIM,
            hidden_size=HIDDEN_SIZE,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(DROPOUT)
        lstm_out = HIDDEN_SIZE * 2  # bidirectional output, so need x2

        self.attention = nn.Linear(lstm_out, 1)
        self.heads = nn.ModuleDict({task: nn.Linear(lstm_out, n_classes[task]) for task in TASKS})  # create 3 separate heads (sentiment,emotion,topic)
        self.ner_head = nn.Linear(lstm_out, n_classes["ner"])
        self.crf = CRF(n_classes["ner"], batch_first=True)


    @classmethod
    def from_labels(cls, labels: dict[str, list[str]], embeddings: torch.Tensor) -> "BiLSTM":
        """Build a model sized for the class lists that train.py saved."""
        return cls({task: len(names) for task, names in labels.items()}, embeddings)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
        embeddings_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        input_ids           : (batch, words)      long, 0 = <pad>
        mask                : (batch, words)      bool, True = real word
        embeddings_override : (batch, words, embed_dim) or None

        Returns one logit tensor per task: (batch, n_classes) for the sentence
        tasks, (batch, words, n_ner) for NER.
        """
        # -> (batch, words, embed_dim)
        words = self.embedding(input_ids) if embeddings_override is None else embeddings_override
        words = self.projection(words)  # -> (batch, words, PROJ_DIM)

        # Pack so the backward LSTM never reads padding: <pad> embeddings are zero,
        # but the projection's bias makes them non-zero by the time they reach here.
        lengths = mask.sum(dim=1).cpu()  # pack_padded_sequence wants CPU lengths
        packed = nn.utils.rnn.pack_padded_sequence(
            words, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=mask.size(1)
        )  # -> (batch, words, 2*HIDDEN_SIZE)
        outputs = self.dropout(outputs)

        # attention pooling: score each word, ignore padding, weighted sum.
        scores = self.attention(outputs)  # -> (batch, words, 1)
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)  # -> (batch, words, 1)
        pooled = (outputs * weights).sum(dim=1)  # -> (batch, 2*HIDDEN_SIZE)

        predictions = {task: head(pooled) for task, head in self.heads.items()}
        predictions["ner"] = self.ner_head(outputs)  # -> (batch, words, n_ner)
        return predictions

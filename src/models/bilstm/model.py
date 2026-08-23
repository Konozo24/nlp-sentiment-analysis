"""The BiLSTM architecture.

    word ids
      -> frozen embedding table   (fastText: pretrained 300d + in-domain 100d)
      -> Linear + ReLU + Dropout  (trainable projection into the LSTM's space)
      -> BiLSTM                   (one contextual vector per word)
      -> attention pooling        (weighted sum -> one vector per tweet)
      -> sentiment / emotion / topic heads   (from the pooled vector)
      -> NER head + CRF                      (from the per-word vectors)

One forward pass, four outputs.

Why the embeddings are frozen: the table is ~26.5k x 400. The training split is
~19k tweets. Unfreezing lets the model memorise the training vocabulary rather
than learn the task, and it destroys the very property we built the table for -
that a slang word absent from training still lands near its neighbours. The
trainable projection is what adapts the space instead.
"""

import torch
from torch import nn
from torchcrf import CRF

from .config import DROPOUT, FREEZE_EMBEDDINGS, HIDDEN_SIZE, PROJ_DIM, TASKS


class BiLSTM(nn.Module):
    def __init__(self, n_classes: dict[str, int], embeddings: torch.Tensor) -> None:
        """n_classes maps every task name (TASKS + 'ner') to its number of classes.
        embeddings is the (vocab_size, embed_dim) matrix from build_embeddings.py."""
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
        lstm_out = HIDDEN_SIZE * 2

        self.attention = nn.Linear(lstm_out, 1)
        self.heads = nn.ModuleDict({task: nn.Linear(lstm_out, n_classes[task]) for task in TASKS})
        self.ner_head = nn.Linear(lstm_out, n_classes["ner"])
        self.crf = CRF(n_classes["ner"], batch_first=True)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Explicit initialisation instead of relying on PyTorch's defaults.

        The projection feeds a ReLU, so Kaiming is the matching scheme. The LSTM
        gets orthogonal recurrent weights - the standard choice for RNNs, since
        it keeps repeated multiplication from exploding or vanishing along the
        sequence. nn.Embedding is skipped: its weights are the fastText vectors
        and must not be touched.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, parameter in module.named_parameters():
                if name.startswith("weight_ih"):
                    nn.init.xavier_uniform_(parameter)
                elif name.startswith("weight_hh"):
                    nn.init.orthogonal_(parameter)
                elif name.startswith("bias"):
                    nn.init.zeros_(parameter)

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

        predict.py passes embeddings_override so a word missing from the
        vocabulary can be given a vector composed on the fly from character
        n-grams, rather than collapsing to <unk>.

        Returns one logit tensor per task: (batch, n_classes) for the sentence
        tasks, (batch, words, n_ner) for NER.
        """
        # -> (batch, words, embed_dim)
        words = self.embedding(input_ids) if embeddings_override is None else embeddings_override
        words = self.projection(words)  # -> (batch, words, PROJ_DIM)

        outputs, _ = self.lstm(words)  # -> (batch, words, 2*HIDDEN_SIZE)
        outputs = self.dropout(outputs)

        # attention pooling: score each word, ignore padding, weighted sum.
        scores = self.attention(outputs)  # -> (batch, words, 1)
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)  # -> (batch, words, 1)
        pooled = (outputs * weights).sum(dim=1)  # -> (batch, 2*HIDDEN_SIZE)

        predictions = {task: head(pooled) for task, head in self.heads.items()}
        predictions["ner"] = self.ner_head(outputs)  # -> (batch, words, n_ner)
        return predictions

# Model comparison

All three models scored on the same 4,130 test tweets (shared row set and group-aware split; see src/data_cleaning/base_cleaning.py).

## Sentiment

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| SVM (TF-IDF) | 0.7140 | 0.6884 | 0.6895 | 0.6889 | 0.7142 |
| BiLSTM | 0.7390 | 0.7187 | 0.7653 | 0.7297 | 0.7414 |
| RoBERTa-CNN | 0.7455 | 0.7207 | 0.7628 | 0.7343 | 0.7484 |

## Emotion

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| SVM (TF-IDF) | 0.6666 | 0.4675 | 0.4724 | 0.4662 | 0.6707 |
| BiLSTM | 0.6688 | 0.4869 | 0.6111 | 0.5217 | 0.6897 |
| RoBERTa-CNN | 0.6804 | 0.4853 | 0.5938 | 0.5208 | 0.6964 |

## Topic

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| SVM (TF-IDF) | 0.7339 | 0.7136 | 0.7070 | 0.7097 | 0.7343 |
| BiLSTM | 0.7596 | 0.7258 | 0.7747 | 0.7439 | 0.7611 |
| RoBERTa-CNN | 0.7215 | 0.6929 | 0.7515 | 0.7088 | 0.7236 |

## NER (entity-type presence)

| Model | Micro F1 | Macro F1 | Exact match |
|---|---|---|---|
| SVM (TF-IDF) | 0.9553 | 0.9236 | 0.8462 |
| BiLSTM | 0.9677 | 0.9379 | 0.8896 |
| RoBERTa-CNN | 0.9531 | 0.9087 | 0.8424 |

_Entity-type presence is the only NER framing the three models share. The BiLSTM's and RoBERTa-CNN's token-level BIO scores are in their own metrics.txt and are not comparable with the SVM's._

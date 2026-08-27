# Model comparison

All three models scored on the same 4,130 test tweets (shared row set and group-aware split; see src/data_cleaning/base_cleaning.py).

## Sentiment

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| SVM (TF-IDF) | 0.7114 | 0.6863 | 0.6875 | 0.6868 | 0.7115 |
| BiLSTM | 0.7383 | 0.7119 | 0.7516 | 0.7244 | 0.7419 |
| RoBERTa-base | 0.7588 | 0.7346 | 0.7819 | 0.7484 | 0.7616 |

## Emotion

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| SVM (TF-IDF) | 0.6608 | 0.4554 | 0.4738 | 0.4619 | 0.6655 |
| BiLSTM | 0.6366 | 0.4725 | 0.5718 | 0.4779 | 0.6658 |
| RoBERTa-base | 0.7022 | 0.5107 | 0.6353 | 0.5545 | 0.7147 |

## Topic

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| SVM (TF-IDF) | 0.7354 | 0.7160 | 0.7113 | 0.7131 | 0.7359 |
| BiLSTM | 0.7588 | 0.7382 | 0.7476 | 0.7415 | 0.7597 |
| RoBERTa-base | 0.7801 | 0.7477 | 0.7915 | 0.7647 | 0.7818 |

## NER (entity-type presence)

| Model | Micro F1 | Macro F1 | Exact match |
|---|---|---|---|
| SVM (TF-IDF) | 0.9630 | 0.9340 | 0.8729 |
| BiLSTM | 0.9573 | 0.9161 | 0.8550 |
| RoBERTa-base | 0.9558 | 0.9155 | 0.8470 |

_Entity-type presence is the only NER framing the three models share. Each model's own token-level BIO score is in its own metrics.txt - they use different features and architectures, so those aren't directly comparable with each other._

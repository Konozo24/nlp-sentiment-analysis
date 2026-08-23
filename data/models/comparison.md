# Model comparison

All three models scored on the same 4,130 test tweets (shared row set and group-aware split; see src/data_cleaning/base_cleaning.py).

## Sentiment

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| SVM (TF-IDF) | 0.7114 | 0.6863 | 0.6875 | 0.6868 | 0.7115 |
| BiLSTM | 0.7337 | 0.7106 | 0.7301 | 0.7177 | 0.7353 |
| RoBERTa-CNN | 0.7414 | 0.7178 | 0.7606 | 0.7308 | 0.7437 |

## Emotion

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| SVM (TF-IDF) | 0.6608 | 0.4554 | 0.4738 | 0.4619 | 0.6655 |
| BiLSTM | 0.6731 | 0.4709 | 0.5353 | 0.4918 | 0.6854 |
| RoBERTa-CNN | 0.6828 | 0.4837 | 0.6003 | 0.5242 | 0.6967 |

## Topic

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| SVM (TF-IDF) | 0.7354 | 0.7160 | 0.7113 | 0.7131 | 0.7359 |
| BiLSTM | 0.6775 | 0.6499 | 0.7061 | 0.6606 | 0.6835 |
| RoBERTa-CNN | 0.7426 | 0.7122 | 0.7609 | 0.7272 | 0.7460 |

## NER (entity-type presence)

| Model | Micro F1 | Macro F1 | Exact match |
|---|---|---|---|
| SVM (TF-IDF) | 0.9630 | 0.9340 | 0.8729 |
| BiLSTM | 0.9108 | 0.8521 | 0.6833 |
| RoBERTa-CNN | 0.9515 | 0.9045 | 0.8412 |

_Entity-type presence is the only NER framing the three models share. Each model's own token-level BIO score is in its own metrics.txt - they use different features and architectures, so those aren't directly comparable with each other._

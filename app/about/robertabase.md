### Architecture

```
tweet
    |
cardiffnlp/twitter-roberta-base   (fine-tuned, subword tokens)
    |
gather first-subword-per-word --> one vector per word
    |                                       |
self-attention pooling             NER head --> CRF (tag per word)
    |
sentiment / emotion / topic
```

### A fine-tuned transformer encoder

This model fine-tunes `cardiffnlp/twitter-roberta-base`, a RoBERTa encoder
pretrained on Twitter text, on this project's World Cup tweet data. RoBERTa's
self-attention gives each token context from the rest of the tweet, while the
first subword representation keeps the model aligned with the word-level NER
tags used by this project.

### Attention pooling

For sentiment, emotion, and topic, each task has a small learned attention
pooling layer. It assigns an importance score to each word and combines the
word representations into one task-specific summary. This avoids relying on a
single `[CLS]` token and uses the same contextual word representations that
feed NER.

### The no-CNN baseline

Unlike RoBERTa-CNN, this model sends the contextual word vectors directly to
the NER head and CRF, and uses attention pooling rather than convolution for
the sentence-level tasks. It is the controlled baseline for measuring what
CNN-based local feature extraction contributes.

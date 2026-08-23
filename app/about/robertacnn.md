### Architecture

```
tweet
    |
cardiffnlp/twitter-roberta-base   (fine-tuned, subword tokens)
    |
gather first-subword-per-word --> one vector per word
    |                                       |
multi-kernel CNN + max-pool     same-padding CNN (no pooling)
    |                                       |
sentiment / emotion / topic          NER head --> CRF (tag per word)
```

### A fine-tuned transformer encoder

This is a **transformer-era** model: `cardiffnlp/twitter-roberta-base`, a
RoBERTa encoder pretrained on Twitter text, fine-tuned end-to-end on this
project's data. Every word's representation depends on the whole tweet
around it - self-attention lets each token look at every other token in the
sequence, so the same word gets a different vector depending on what it's
next to. Because the encoder is fine-tuned rather than frozen, its notion of
"context" is adapted to this exact task and this exact dataset, not just to
general English.

RoBERTa tokenizes into subwords, not whole words ('messi' becomes
'mess'+'i'), but this project's NER tags and evaluation are word-level. Each
word's representation is taken from its **first subword's** hidden state,
which keeps the model's output aligned one-to-one with the words a human
reads.

### Why CNN pooling, not the encoder's own [CLS] token

Rather than reading off RoBERTa's pooled `[CLS]` representation, this model
takes the encoder's per-word hidden states and runs them through a small
multi-kernel 1D CNN + max-pool for the sentence-level tasks
(sentiment/emotion/topic), and a separate same-padding CNN that keeps one
vector per word (no pooling) to feed the NER CRF. The CNN layer lets nearby
words' hidden states combine into local patterns before pooling, rather than
relying on a single summary token to capture everything the sentence needs.

### The cost of fine-tuning on ~19k tweets

Full fine-tuning of a 110M-parameter encoder on a training set this size
overfits quickly - this model's own training run converges within the first
few epochs and gains little from training longer. That's a real, expected
trade-off of this approach: more representational power per parameter, but
also more parameters than a training set this size can fully constrain.

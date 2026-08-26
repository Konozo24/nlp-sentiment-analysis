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

This is a **transformer-era** model: `cardiffnlp/twitter-roberta-base`, a
RoBERTa encoder pretrained on Twitter text, fine-tuned end-to-end on this
project's World Cup tweet data. Every word's representation depends on the
whole tweet around it - self-attention lets each token look at every other
token in the sequence, so the same word gets a different vector depending on
what it is next to. Because the encoder is fine-tuned rather than frozen, its
notion of context is adapted to this task and this dataset, not only to
general English.

RoBERTa tokenizes into subwords rather than whole words ('messi' becomes
'mess'+'i'), but this project's NER tags and evaluation are word-level. Each
word's representation is taken from its **first subword's** hidden state,
which keeps the model's output aligned one-to-one with the words a human
reads.

### Why attention pooling instead of `[CLS]`

The `[CLS]` token provides a convenient whole-tweet summary, but it compresses
the sequence into one representation before the task-specific heads make
their decisions. For sentiment, emotion, and topic, a separate learned
attention layer assigns an importance score to each word and combines the
contextual representations into a task-specific summary. Each head can
therefore focus on the evidence most relevant to its task: sentiment can
emphasise opinion-bearing words, while emotion and topic can focus on
different patterns in the same tweet.

### Word-level NER and CRF decoding

NER keeps the full sequence of word representations rather than reducing the
tweet to one summary. A linear head produces scores for each BIO tag at every
word position, and a CRF decodes the sequence as a whole. This allows the
model to use the surrounding context when identifying entity spans while also
preferring structurally valid tag transitions.

### The cost of fine-tuning on this dataset

Fine-tuning the full transformer gives the model substantial representational
capacity, but it also means that many parameters must be learned from a
comparatively modest collection of labelled tweets. This model therefore
requires more computation than the SVM and BiLSTM approaches, with the
trade-off of richer contextual representations for informal, ambiguous
language.

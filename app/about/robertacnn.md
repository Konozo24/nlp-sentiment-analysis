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

### The transformer-era model

This is the **transformer era** model in the group's three-way comparison -
SVM + TF-IDF (classical), BiLSTM + fastText (deep learning, no transformer),
RoBERTa-CNN (this page). Unlike the BiLSTM's frozen, context-independent word
vectors, every word's representation here depends on the whole tweet around
it, because the encoder itself is fine-tuned end-to-end on this dataset.

### Why CNN pooling, not the encoder's own [CLS] token

Rather than reading off RoBERTa's pooled `[CLS]` representation, this model
takes the encoder's per-word hidden states and runs them through a small
multi-kernel 1D CNN + max-pool for sentence-level tasks (sentiment/emotion/
topic), and a separate same-padding CNN that keeps one vector per word (no
pooling) to feed the NER CRF. Hyperparameters intentionally match the
BiLSTM's, so any performance gap traces to the architecture - a fine-tuned
transformer encoder vs. frozen fastText - not to a different training recipe.

### The cost of fine-tuning on ~19k tweets

Full fine-tuning of a 110M-parameter encoder on a training set this size
overfits quickly - this model's own training run converges in the first few
epochs, well before the BiLSTM does, and gains little from training longer.
That's a real, expected trade-off of the transformer era, not a bug: more
representational power, less data to justify it than a larger corpus would.

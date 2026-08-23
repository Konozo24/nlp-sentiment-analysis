### Architecture

```
word ids
    |
frozen embedding table  (fastText: pretrained 300d + in-domain 100d)
    |
Linear -> ReLU -> Dropout      (trainable projection, {PROJ_DIM}d)
    |
BiLSTM ({HIDDEN_SIZE} units/direction, dropout {DROPOUT})  --> one vector per word
    |                                   |
attention-weighted sum                  --> NER head --> CRF (tag per word)
    |
sentiment / emotion / topic heads
```

### Why static embeddings, and why fastText

This is the **deep-learning era** model in the group's three-way comparison:
SVM + TF-IDF (classical), this BiLSTM (deep learning), RoBERTa-CNN
(transformer). There is deliberately no transformer anywhere in it - the
whole point is to show what the era before transformers could and could not do.

Every word here gets **one fixed vector regardless of context**. 'Fire' in
'he's on fire' and 'the manager got fired up' start from the same point; only
the BiLSTM's surrounding context can separate them. That limitation is exactly
what the transformer era removed, which makes this a meaningful baseline
rather than a weaker copy of Jason's model.

The embedding table itself combines two sources:

- **Pretrained fastText** (`cc.en.300`, Common Crawl) - general English meaning.
- **In-domain fastText**, trained here on our own ~58k World Cup tweets -
  including ~10k from 2026. Current slang means what it means *in this corpus*;
  no published embedding can contain it, because the usage postdates them all.

### Handling words nobody has seen

Both halves are fastText, which represents a word as the sum of its character
n-grams. A word missing from the vocabulary is therefore **composed** rather
than discarded: `bonkersss` is reached through `bonk`, `onke`, `kers`. GloVe
and Word2Vec cannot do this at all - an unseen token gets nothing.

Try it in the Live Demo tab: invent a word and watch it still get a vector.

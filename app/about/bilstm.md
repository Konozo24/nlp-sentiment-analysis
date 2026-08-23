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

This is a **deep-learning, pre-transformer** model: a bidirectional LSTM
reading a sequence of fixed word vectors, with no self-attention over the
input and no fine-tuned encoder anywhere in it.

Every word here gets **one fixed vector regardless of context** — 'fire' in
'he's on fire' and 'the manager got fired up' start from the exact same
point. Only the BiLSTM layer's own left-to-right and right-to-left passes
can pull those two apart from there, by combining the fixed vector with
whatever surrounds it in the sentence. An attention layer then learns which
words in that sequence matter most for each task, and pools them into one
vector per tweet.

The embedding table itself combines two sources:

- **Pretrained fastText** (`cc.en.300`, Common Crawl) — general English meaning.
- **In-domain fastText**, trained here on our own ~58k World Cup tweets —
  including ~10k from 2026. Current slang means what it means *in this corpus*;
  no published embedding can contain it, because the usage postdates them all.

### Handling words nobody has seen

Both halves are fastText, which represents a word as the sum of its character
n-grams. A word missing from the vocabulary is therefore **composed** rather
than discarded: `bonkersss` is reached through `bonk`, `onke`, `kers`. GloVe
and Word2Vec cannot do this at all — an unseen token gets nothing.

Try it in the Live Demo tab: invent a word and watch it still get a vector.

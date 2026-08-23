### Architecture

```
tweet
    |
    +-- TF-IDF (1-2 grams, sublinear tf) --> LinearSVC, one per task   (sentiment / emotion / topic)
    |
    +-- per-word features (word, prefix/suffix, neighbours, position)
                        --> LinearSVC, one BIO tag per word            (NER)
```

### Why TF-IDF + a linear SVM

TF-IDF is a **classical, pre-neural** representation: no embeddings, no
neural network anywhere in this pipeline. Each word (or word pair) becomes
one sparse feature, weighted by how distinctive it is across the corpus -
common everywhere ("the", "a") counts for little, rare-but-consistent words
count for a lot. A separate `LinearSVC` is then trained per task on that
fixed representation, finding the hyperplane that best separates the
classes in that sparse feature space.

That representation has no notion of word order or context: "not good" and
"good" share the token "good" with equal weight regardless of the negation
next to it, and two words are only ever related through their raw co-occurrence
statistics, never their meaning. Its strength is the opposite side of that
same coin - with no embedding table to learn and only a linear decision
boundary per task, it needs far less data to avoid overfitting than a neural
model does, which is exactly where it holds its own.

### Confidence, without native probabilities

`LinearSVC` produces a margin (`decision_function`), not a probability. The
confidence shown here comes from turning that margin into one via a sigmoid
(binary tasks) or a softmax over the one-vs-rest margins (multi-class) - an
approximation, not a calibrated probability, which is why this page has no
full class-probability breakdown.

### NER: a second SVM, trained on different features

TF-IDF collapses a whole tweet into one bag-of-words vector - exactly what
loses the sentence/emotion/topic heads' notion of context, and it also means
that representation has no idea *where* a word sits, so it can't drive a
per-word tagger. NER here is a **separate** `LinearSVC`, trained on a
different, per-word feature set instead: each word's own text, its
prefix/suffix, its immediate neighbours, and its position in the tweet. That
local window is enough to place `B-PER`/`I-PER`/`O`-style BIO tags one word
at a time - the standard pre-neural approach to sequence tagging, reframing
"tag this sequence" as "classify this word, given what's around it."

It's still a linear model with no notion of the *sentence itself*, only a
sliding local window, which is the honest limit of this method - unlike
BiLSTM's and RoBERTa-CNN's CRF taggers, which see one contextual vector per
word informed by the whole tweet. Scored two ways: its own token-level BIO
F1 (Performance tab), and the entity-type-presence framing all three models
share, by collapsing its predicted tags the same way the other two do.

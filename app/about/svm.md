### Architecture

```
tweet
    |
TF-IDF (1-2 grams, sublinear tf)
    |
LinearSVC, one-vs-rest per task   (+ spaCy en_core_web_trf for NER)
```

### Why TF-IDF + a linear SVM

This is the **classical era** model in the group's three-way comparison -
no embeddings, no neural network anywhere in it. Each word (or word pair)
becomes one sparse feature weighted by how distinctive it is across the
corpus, and a separate LinearSVC is trained per task on that fixed
representation.

That representation has no notion of word order or context: "not good" and
"good" share the token "good" with equal weight regardless of the negation
next to it. It's a meaningful lower bound for the other two models to beat,
and - per the group's comparison table - it still holds its own, especially
where the training set is small relative to the vocabulary.

### Confidence, without native probabilities

`LinearSVC` produces a margin (`decision_function`), not a probability. The
confidence shown here comes from turning that margin into one via a sigmoid
(binary tasks) or a softmax over the one-vs-rest margins (multi-class) - an
approximation, not a calibrated probability, which is why this page has no
full class-probability breakdown the way the two neural models do.

### NER is a different model entirely

Named-entity recognition here doesn't come from the TF-IDF/SVM pipeline at
all - it runs spaCy's `en_core_web_trf` (a transformer NER pipeline) and maps
its entity types onto this project's PER/ORG/LOC/EVENT scheme. The first
prediction in a session is noticeably slower while that pipeline loads from
disk; every prediction after is fast.

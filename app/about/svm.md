### Architecture

```
tweet
    |
TF-IDF (1-2 grams, sublinear tf)
    |
LinearSVC, one-vs-rest per task   (+ spaCy en_core_web_trf for NER)
```

### Why TF-IDF + a linear SVM

TF-IDF is a **classical, pre-neural** representation: no embeddings, no
neural network anywhere in this pipeline. Each word (or word pair) becomes
one sparse feature, weighted by how distinctive it is across the corpus —
common everywhere ("the", "a") counts for little, rare-but-consistent words
count for a lot. A separate `LinearSVC` is then trained per task on that
fixed representation, finding the hyperplane that best separates the
classes in that sparse feature space.

That representation has no notion of word order or context: "not good" and
"good" share the token "good" with equal weight regardless of the negation
next to it, and two words are only ever related through their raw co-occurrence
statistics, never their meaning. Its strength is the opposite side of that
same coin — with no embedding table to learn and only a linear decision
boundary per task, it needs far less data to avoid overfitting than a neural
model does, which is exactly where it holds its own.

### Confidence, without native probabilities

`LinearSVC` produces a margin (`decision_function`), not a probability. The
confidence shown here comes from turning that margin into one via a sigmoid
(binary tasks) or a softmax over the one-vs-rest margins (multi-class) — an
approximation, not a calibrated probability, which is why this page has no
full class-probability breakdown.

### NER is a different model entirely

Named-entity recognition here doesn't come from the TF-IDF/SVM pipeline at
all - it runs spaCy's `en_core_web_trf` (a transformer NER pipeline) and maps
its entity types onto this project's PER/ORG/LOC/EVENT scheme. The first
prediction in a session is noticeably slower while that pipeline loads from
disk; every prediction after is fast.

"""SVM backend selection and model construction."""

import numpy as np
from .config import TASKS

try:
    import cupy as cp
    from cuml.feature_extraction.text import TfidfVectorizer
    from cuml.multiclass import OneVsRestClassifier
    from cuml.svm import LinearSVC
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    GPU_AVAILABLE = False

if not GPU_AVAILABLE:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC

DEVICE = "cuda" if GPU_AVAILABLE else "cpu"


def make_vectorizer(kwargs):
    return TfidfVectorizer(**kwargs)


def to_numpy(values):
    return cp.asnumpy(values) if GPU_AVAILABLE else np.asarray(values)


WEIGHT_CAP = 10.0  # same cap the BiLSTM and RoBERTa-CNN losses apply


def class_weights(target) -> dict[int, float]:
    """Inverse-frequency weight per class: min(n / (k * count), WEIGHT_CAP).

    n is the number of samples and k the number of classes present, so a class
    appearing less often than average weighs more than 1. The cap keeps a very
    rare class from dominating the objective, and matches the weighting the
    BiLSTM and RoBERTa-CNN build into their CrossEntropyLoss.
    """
    counts = np.bincount(np.asarray(target))
    n, k = len(target), (counts > 0).sum()
    return {
        i: min(n / (k * count), WEIGHT_CAP) for i, count in enumerate(counts) if count > 0
    }


def train_models(features, single_targets, ner_targets):
    """Fit one multiclass SVM per task, plus a one-vs-rest SVM for NER types.

    The three task models are class-weighted with class_weights(). cuML's
    LinearSVC takes no class_weight argument, so on GPU the same weights are
    applied per row through sample_weight instead; both routes optimise the
    same objective.

    The NER model treats each entity type as its own binary problem, so each
    one is balanced independently rather than by a single task's labels.
    """
    models = {}
    for task in TASKS:
        target = single_targets[task]
        weights = class_weights(target)
        sample_weight = np.array([weights[int(value)] for value in target], dtype=np.float32)
        if GPU_AVAILABLE:
            models[task] = LinearSVC().fit(
                features, cp.asarray(target), sample_weight=cp.asarray(sample_weight)
            )
        else:
            models[task] = LinearSVC(class_weight=weights).fit(features, target)

    ovr_kwargs = {} if GPU_AVAILABLE else {"class_weight": "balanced"}
    target = cp.asarray(ner_targets) if GPU_AVAILABLE else ner_targets
    models["ner"] = OneVsRestClassifier(LinearSVC(**ovr_kwargs)).fit(features, target)
    return models

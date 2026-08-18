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


def train_models(features, single_targets, ner_targets):
    """Fit three multiclass SVMs and one one-vs-rest NER-type SVM."""
    kwargs = {} if GPU_AVAILABLE else {"class_weight": "balanced"}
    models = {}
    for task in TASKS:
        target = cp.asarray(single_targets[task]) if GPU_AVAILABLE else single_targets[task]
        models[task] = LinearSVC(**kwargs).fit(features, target)
    target = cp.asarray(ner_targets) if GPU_AVAILABLE else ner_targets
    models["ner"] = OneVsRestClassifier(LinearSVC(**kwargs)).fit(features, target)
    return models

from __future__ import annotations

import numpy as np


def classification_metrics(
    prediction: np.ndarray,
    targets: np.ndarray,
    classes: int,
) -> dict[str, object]:
    prediction = np.asarray(prediction, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (targets, prediction), 1)
    support = confusion.sum(axis=1)
    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(classes, dtype=np.float64),
        where=support > 0,
    )
    total = int(confusion.sum())
    oa = float(np.trace(confusion) / max(total, 1))
    aa = float(recall[support > 0].mean())
    expected = float(confusion.sum(0) @ confusion.sum(1) / max(total * total, 1))
    kappa = float((oa - expected) / max(1.0 - expected, 1e-12))
    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "per_class_recall": recall.tolist(),
        "per_class_support": support.astype(int).tolist(),
        "confusion_matrix": confusion.tolist(),
    }

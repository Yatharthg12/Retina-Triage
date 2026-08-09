from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score,
)

def expected_calibration_error(y_true, probabilities, bins=10):
    y_true = np.asarray(y_true); probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1); predicted = probabilities.argmax(axis=1)
    total = len(y_true); score = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            score += mask.sum() / total * abs((predicted[mask] == y_true[mask]).mean() - confidence[mask].mean())
    return float(score)

def binary_metrics(y_true, score, threshold=.5):
    y_true = np.asarray(y_true).astype(int); predicted = np.asarray(score) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "roc_auc": roc_auc_score(y_true, score) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

def calculate_metrics(y_true, probabilities):
    y_true = np.asarray(y_true).astype(int); probabilities = np.asarray(probabilities)
    y_pred = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(5), zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(5))
    per_class = {}
    for grade in range(5):
        tp = matrix[grade, grade]; fn = matrix[grade].sum() - tp
        fp = matrix[:, grade].sum() - tp; tn = matrix.sum() - tp - fn - fp
        per_class[str(grade)] = {
            "precision": float(precision[grade]), "recall": float(recall[grade]),
            "f1": float(f1[grade]), "support": int(support[grade]),
            "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
        }
    referable_true, high_true = y_true >= 2, y_true >= 3
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mean_absolute_grade_error": float(np.abs(y_true - y_pred).mean()),
        "adjacent_grade_error_rate": float((np.abs(y_true - y_pred) == 1).mean()),
        "severe_underclassification_count": int(((y_true >= 3) & (y_pred <= 1)).sum()),
        "correct_count": int((y_true == y_pred).sum()),
        "dangerous_error_count": int(((y_true >= 3) & (y_pred <= 1)).sum()),
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "multiclass_brier_score": float(np.mean(np.sum((np.eye(5)[y_true] - probabilities) ** 2, axis=1))),
        "per_class": per_class, "classification_report": classification_report(
            y_true, y_pred, labels=np.arange(5), output_dict=True, zero_division=0
        ),
        "confusion_matrix": matrix.tolist(),
        "referable": binary_metrics(referable_true, probabilities[:, 2:].sum(axis=1)),
        "high_risk": binary_metrics(high_true, probabilities[:, 3:].sum(axis=1)),
    }


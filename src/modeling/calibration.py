from __future__ import annotations

import numpy as np

def temperature_scale(probabilities, temperature: float = 1.0):
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, 1e-8, 1.0)
    logits = np.log(probs) / max(float(temperature), 1e-4)
    logits -= np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)

def uncertainty_summary(probabilities) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=np.float64)
    order = np.sort(p)[::-1]
    return {
        "calibrated_confidence": float(order[0]),
        "entropy": float(-np.sum(p * np.log(np.clip(p, 1e-8, 1.0)))),
        "top_two_margin": float(order[0] - order[1]),
        "expected_grade": float(np.dot(p, np.arange(5))),
        "referable_probability": float(p[2:].sum()),
        "high_risk_probability": float(p[3:].sum()),
    }

def fit_temperature(probabilities, labels, candidates=None) -> dict[str, float]:
    """Select temperature by validation negative log likelihood."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    candidates = np.asarray(candidates if candidates is not None else np.linspace(.5, 3.0, 101))
    best_temperature, best_nll = 1.0, float("inf")
    for temperature in candidates:
        scaled = temperature_scale(probabilities, float(temperature))
        nll = float(-np.mean(np.log(np.clip(scaled[np.arange(len(labels)), labels], 1e-8, 1.0))))
        if nll < best_nll:
            best_temperature, best_nll = float(temperature), nll
    return {"temperature": best_temperature, "validation_nll": best_nll}

def threshold_for_target_sensitivity(labels, scores, target: float = .90) -> dict[str, float | None]:
    """Choose the highest validation threshold meeting target sensitivity."""
    labels = np.asarray(labels, dtype=bool); scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if positives == 0:
        return {"threshold": None, "sensitivity": None, "specificity": None, "positives": 0}
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    feasible = []
    for threshold in candidates:
        predicted = scores >= threshold
        sensitivity = float((predicted & labels).sum() / positives)
        specificity = float((~predicted & ~labels).sum() / max((~labels).sum(), 1))
        if sensitivity >= target:
            feasible.append((float(threshold), sensitivity, specificity))
    threshold, sensitivity, specificity = max(feasible, key=lambda item: (item[0], item[2]))
    return {"threshold": threshold, "sensitivity": sensitivity, "specificity": specificity, "positives": positives}

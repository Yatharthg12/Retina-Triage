from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.config import load_config
from src.modeling.metrics import calculate_metrics
from src.training.train import make_dataset

def evaluate(config):
    import tensorflow as tf
    model_path, test_path = Path(config["paths"]["model"]), Path(config["paths"]["splits"]) / "test.csv"
    if not model_path.exists():
        raise FileNotFoundError("Trained model artifact is missing. Run training before evaluation.")
    if not test_path.exists():
        raise FileNotFoundError("Untouched test manifest is missing. Run dataset preparation first.")
    frame = pd.read_csv(test_path)
    ds = make_dataset(frame, config["image_size"], config["stage_one"]["batch_size"])
    model = tf.keras.models.load_model(model_path, compile=False)
    probabilities, latencies = [], []
    for images, _ in ds:
        started = time.perf_counter(); probabilities.append(model.predict(images, verbose=0))
        latencies.extend([(time.perf_counter()-started)*1000/len(images)]*len(images))
    probabilities = np.concatenate(probabilities)
    metrics = calculate_metrics(frame["grade"].values, probabilities)
    metrics["latency_ms"] = {key: float(value) for key, value in {
        "mean": np.mean(latencies), "median": np.median(latencies),
        "p95": np.percentile(latencies, 95), "p99": np.percentile(latencies, 99),
    }.items()}
    out = Path(config["paths"]["evaluation"]); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions = frame.copy()
    predictions["predicted_grade"] = probabilities.argmax(axis=1)
    for grade in range(5): predictions[f"probability_{grade}"] = probabilities[:, grade]
    predictions.to_csv(out / "predictions.csv", index=False)
    for normalized, filename in [(None, "confusion_matrix.png"), ("true", "confusion_matrix_normalized.png")]:
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(frame["grade"], probabilities.argmax(1), labels=range(5),
                                                 normalize=normalized, ax=ax, cmap="Blues")
        fig.tight_layout(); fig.savefig(out / filename, dpi=160); plt.close(fig)
    one_hot = np.eye(5)[frame["grade"].values]
    for kind, display, filename in [("ROC", RocCurveDisplay, "roc_curves.png"),
                                     ("PR", PrecisionRecallDisplay, "pr_curves.png")]:
        fig, ax = plt.subplots(figsize=(7, 6))
        for grade in range(5):
            display.from_predictions(one_hot[:, grade], probabilities[:, grade], name=f"Grade {grade}", ax=ax)
        ax.set_title(f"One-vs-rest {kind} curves"); fig.tight_layout(); fig.savefig(out / filename, dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = frame["grade"].value_counts().reindex(range(5), fill_value=0)
    ax.bar(range(5), counts.values, color="#0a9f96")
    ax.set(xlabel="True DR grade", ylabel="Test images", title="Untouched test-set class distribution")
    fig.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=160); plt.close(fig)
    confidence, correctness = probabilities.max(axis=1), probabilities.argmax(axis=1) == frame["grade"].values
    edges = np.linspace(0, 1, 11); centres, accuracy, observed_confidence = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            centres.append((low + high) / 2); accuracy.append(correctness[mask].mean()); observed_confidence.append(confidence[mask].mean())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#748a94", label="Ideal")
    if centres: ax.plot(observed_confidence, accuracy, "o-", color="#0a9f96", label="Model")
    ax.set(xlabel="Mean confidence", ylabel="Observed accuracy", title="Reliability diagram", xlim=(0, 1), ylim=(0, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=160); plt.close(fig)
    training_dir = out.parent / "training"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plotted = False
    for stage in ("stage_one", "stage_two"):
        log = training_dir / f"{stage}.csv"
        if log.exists():
            history = pd.read_csv(log); plotted = True
            if "loss" in history: axes[0].plot(history["epoch"], history["loss"], label=f"{stage} train")
            if "val_loss" in history: axes[0].plot(history["epoch"], history["val_loss"], "--", label=f"{stage} val")
            if "accuracy" in history: axes[1].plot(history["epoch"], history["accuracy"], label=f"{stage} train")
            if "val_accuracy" in history: axes[1].plot(history["epoch"], history["val_accuracy"], "--", label=f"{stage} val")
    for ax, title in zip(axes, ("Training loss", "Training accuracy")):
        ax.set_title(title); ax.set_xlabel("Epoch")
        if plotted: ax.legend(fontsize=7)
        else: ax.text(.5, .5, "Training logs unavailable", ha="center", transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=160); plt.close(fig)
    html = "<html><body><h1>RetinaTriage AI evaluation</h1><p>Generated from the untouched test manifest.</p><pre>" + json.dumps(metrics, indent=2) + "</pre></body></html>"
    (out / "evaluation_report.html").write_text(html, encoding="utf-8")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    print(json.dumps(evaluate(load_config(args.config)), indent=2))

if __name__ == "__main__":
    main()

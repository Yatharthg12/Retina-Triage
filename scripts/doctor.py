from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config

def present(path): return "available" if Path(path).exists() else "missing"

def main():
    config = load_config()
    print("RetinaTriage AI environment doctor")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Operating system: {platform.platform()}")
    for name in ["Flask", "numpy", "pandas", "Pillow", "opencv-python", "scikit-learn", "tensorflow", "keras", "reportlab", "waitress"]:
        try: version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: version = "NOT INSTALLED"
        print(f"{name}: {version}")
    try:
        import tensorflow as tf
        devices = [d.name for d in tf.config.list_physical_devices()]
        print(f"TensorFlow devices: {', '.join(devices) or 'none'}")
    except Exception as exc:
        print(f"TensorFlow devices: unavailable ({type(exc).__name__})")
    checks = {
        "Dataset": config["paths"]["dataset"],
        "Train split": Path(config["paths"]["splits"]) / "train.csv",
        "Model": config["paths"]["model"],
        "Model metadata": config["paths"]["model_metadata"],
        "Calibration": config["paths"]["calibration"],
        "Evaluation": Path(config["paths"]["evaluation"]) / "metrics.json",
        "Demo samples": Path(config["_root"]) / "demo" / "samples" / "manifest.json",
    }
    for label, path in checks.items(): print(f"{label}: {present(path)} ({path})")
    for label, path in [("Database directory", Path(config["paths"]["database"]).parent),
                        ("Reports directory", Path(config["paths"]["reports"]))]:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path, delete=True): pass
            print(f"{label} write access: yes")
        except OSError:
            print(f"{label} write access: NO")
    try:
        from src.inference.predictor import Predictor
        status = Predictor(config).status()
        print(f"Inference readiness: {status['status']} ({status.get('error') or status['version']})")
        print(f"Confidence status: {status['calibration']['status']}")
    except Exception as exc:
        print(f"Inference readiness: unavailable ({type(exc).__name__}: {exc})")

if __name__ == "__main__":
    main()

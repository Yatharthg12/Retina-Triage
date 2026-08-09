from __future__ import annotations

import hashlib
import base64
import io
import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from src.constants import CLASS_NAMES
from src.data.preprocessing import preprocess_image
from src.data.quality import assess_quality
from src.inference.triage import apply_triage
from src.inference.gradcam import GradCAMError, make_gradcam_heatmap, overlay_heatmap
from src.modeling.calibration import temperature_scale, uncertainty_summary

class ModelUnavailableError(RuntimeError):
    pass

class Predictor:
    def __init__(self, config: dict, model=None):
        self.config = config
        self.model = model
        self.lock = threading.Lock()
        self.temperature = float(config["calibration"].get("temperature", 1.0))
        self.version = config["model"].get("version", "untrained")
        self.metadata = {}
        self.calibration = {
            "temperature": self.temperature,
            "status": "not_available",
            "method": "none",
        }
        self.load_error = None
        metadata_path = Path(config["paths"]["model_metadata"])
        calibration_path = Path(config["paths"]["calibration"])
        if calibration_path.exists():
            self.calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            self.temperature = float(self.calibration["temperature"])
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            mapping = {int(k): v for k, v in self.metadata.get("class_names", {}).items()}
            if mapping and mapping != CLASS_NAMES:
                self.load_error = "Model class mapping is incompatible."
            self.version = self.metadata.get("model_version", self.version)
        model_path = Path(config["paths"]["model"])
        if self.model is None and model_path.exists() and not metadata_path.exists():
            self.load_error = "Model metadata is missing."
        if self.model is None and model_path.exists() and not calibration_path.exists():
            self.load_error = "Calibration parameters are missing."
        expected_hash = self.metadata.get("artifact_sha256")
        if self.model is None and model_path.exists() and expected_hash and not self.load_error:
            actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if actual_hash.lower() != str(expected_hash).lower():
                self.load_error = "Model artifact checksum does not match its metadata."
        if self.model is None and model_path.exists() and not self.load_error:
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path, compile=False)
                expected = (config["image_size"], config["image_size"], 3)
                if tuple(self.model.input_shape[1:]) != expected or self.model.output_shape[-1] != 5:
                    self.model = None
                    self.load_error = "Model dimensions or class count are incompatible."
            except Exception as exc:
                self.load_error = f"Model could not be loaded: {type(exc).__name__}"

    @property
    def available(self) -> bool:
        return self.model is not None and not self.load_error

    def status(self):
        source = self.metadata.get("source", {})
        return {
            "available": self.available,
            "name": self.metadata.get("display_name", "EfficientNetB0 DR severity classifier"),
            "version": self.version, "status": "READY" if self.available else "MODEL NOT INSTALLED",
            "error": self.load_error,
            "class_names": CLASS_NAMES,
            "thresholds": self.config["triage"],
            "quality_gate": {
                "minimum_score": float(self.config["quality"].get("minimum_score", 0.75)),
                "rule": "Scores below the threshold require retake or manual review.",
                "clinically_validated": False,
            },
            "calibration": {
                "status": self.calibration.get("status", "not_available"),
                "method": self.calibration.get("method", "none"),
                "temperature": self.temperature,
                "warning": self.calibration.get("warning"),
            },
            "source": source,
            "published_evaluation": self.metadata.get("published_evaluation"),
            "intended_use": self.metadata.get("intended_use"),
        }

    def predict_bytes(self, raw: bytes, filename: str, case_id: str | None = None,
                      include_gradcam: bool = False, class_index: int | None = None) -> dict:
        started = time.perf_counter()
        processed = preprocess_image(
            raw, self.config["image_size"], self.config["preprocessing"]["clahe"],
            self.config["preprocessing"]["min_dimension"], self.config["preprocessing"]["black_threshold"]
        )
        quality = assess_quality(processed.display, self.config["quality"])
        screening_id = uuid.uuid4().hex
        base = {
            "screening_id": screening_id, "status": "completed",
            "model_available": self.available, "simulated": False,
            "case_id": case_id, "filename": filename, "file_hash": hashlib.sha256(raw).hexdigest(),
            "quality": quality, "preprocessing": processed.metadata,
        }
        if not self.available:
            base.update({
                "status": "model_unavailable", "prediction": None,
                "triage": {
                    "priority": "RETAKE / MANUAL REVIEW" if not quality["gradable"] else "MANUAL REVIEW",
                    "manual_review": True, "urgent": False,
                    "reasons": (
                        [quality["decision_reason"]]
                        if not quality["gradable"]
                        else ["A trained model artifact is not installed."]
                    ),
                },
                "explanation": {}, "model": self.status(),
                "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
            })
            return base
        with self.lock:
            raw_probabilities = np.asarray(self.model.predict(processed.model_input, verbose=0))[0]
        probabilities = temperature_scale(raw_probabilities, self.temperature)
        grade = int(np.argmax(probabilities))
        summary = uncertainty_summary(probabilities)
        prediction = {
            "grade": grade, "label": CLASS_NAMES[grade],
            "probabilities": [round(float(x), 7) for x in probabilities],
            "confidence": round(float(summary["calibrated_confidence"]), 7),
            "confidence_kind": (
                "temperature_calibrated"
                if self.calibration.get("status") == "validated"
                else "raw_softmax"
            ),
            "calibration_status": self.calibration.get("status", "not_available"),
            **{k: round(v, 7) for k, v in summary.items()},
        }
        triage = apply_triage(
            grade, quality["gradable"], summary["calibrated_confidence"], summary["entropy"],
            summary["top_two_margin"], summary["high_risk_probability"], self.config["triage"],
            self.config["quality"].get("minimum_score", 0.75),
        ).to_dict()
        explanation = {}
        if include_gradcam and quality["gradable"]:
            try:
                heatmap = make_gradcam_heatmap(self.model, processed.model_input, class_index)
                rgb = np.asarray(processed.processed)
                heat_rgb = np.uint8(np.clip(heatmap, 0, 1) * 255)
                overlay = overlay_heatmap(rgb, heatmap)
                def data_url(array, mode="RGB"):
                    buffer = io.BytesIO()
                    from PIL import Image
                    Image.fromarray(array, mode=mode).save(buffer, "PNG")
                    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                explanation = {
                    "heatmap": data_url(heat_rgb, "L"),
                    "overlay": data_url(overlay),
                    "processed": data_url(rgb),
                    "class_index": int(class_index) if class_index is not None else grade,
                    "disclaimer": "Grad-CAM highlights image regions that influenced the model output. It is not a lesion segmentation, clinical annotation or diagnosis.",
                }
            except GradCAMError as exc:
                explanation = {"error": str(exc)}
        public_prediction = prediction if quality["gradable"] else None
        base.update({
            "prediction": public_prediction, "triage": triage, "explanation": explanation,
            "advanced_model_output": None,
            "model": {"name": "EfficientNetB0", "version": self.version},
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        return base

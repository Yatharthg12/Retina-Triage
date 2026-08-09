from __future__ import annotations

from src.database.repository import ScreeningRepository

def persist_result(repository: ScreeningRepository, result: dict) -> dict:
    prediction = result.get("prediction") or {}
    quality = result["quality"]
    metrics = quality.get("metrics", {})
    triage = result["triage"]
    return repository.insert({
        "screening_id": result["screening_id"], "case_id": result.get("case_id"),
        "original_filename": result.get("filename", "unnamed"), "file_hash": result.get("file_hash", ""),
        "image_width": metrics.get("width"), "image_height": metrics.get("height"),
        "quality_score": quality.get("quality_score"), "quality_issues": quality.get("issues", []),
        "predicted_grade": prediction.get("grade"), "predicted_label": prediction.get("label"),
        "confidence": prediction.get("confidence", prediction.get("calibrated_confidence")),
        "referable_probability": prediction.get("referable_probability"),
        "high_risk_probability": prediction.get("high_risk_probability"),
        "priority": triage["priority"], "manual_review": triage["manual_review"],
        "review_reasons": triage["reasons"], "model_version": result["model"].get("version", "unavailable"),
        "processing_time_ms": result["processing_time_ms"], "simulated": result.get("simulated", False),
        "probabilities": prediction.get("probabilities", []),
    })

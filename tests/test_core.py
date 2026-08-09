import io
import json

import numpy as np
import pytest
from PIL import Image

from src.config import ConfigError, load_config
from src.constants import CLASS_NAMES, is_high_risk, is_referable
from src.data.preprocessing import ImageValidationError, preprocess_image
from src.data.quality import assess_quality
from src.data.splitting import stratified_split
from src.inference.triage import apply_triage
from src.modeling.calibration import (
    fit_temperature, temperature_scale, threshold_for_target_sensitivity, uncertainty_summary,
)

def test_class_mapping_and_binary_outcomes():
    assert len(CLASS_NAMES) == 5
    assert not is_referable(1) and is_referable(2)
    assert not is_high_risk(2) and is_high_risk(3)

def test_priority_rules():
    thresholds = {"low_confidence": .6, "high_entropy": 1.2, "low_margin": .15, "high_risk_probability": .35}
    assert apply_triage(4, True, .9, .2, .7, .8, thresholds).priority.startswith("URGENT")
    assert apply_triage(3, True, .9, .2, .7, .7, thresholds).priority == "HIGH PRIORITY"
    assert apply_triage(2, True, .9, .2, .7, .1, thresholds).priority == "SPECIALIST REVIEW"
    poor = apply_triage(4, False, .99, .1, .8, .9, thresholds)
    assert poor.priority == "RETAKE / MANUAL REVIEW" and poor.manual_review

def test_uncertainty_escalation():
    thresholds = {"low_confidence": .6, "high_entropy": 1.2, "low_margin": .15, "high_risk_probability": .35}
    result = apply_triage(1, True, .40, 1.4, .04, .40, thresholds)
    assert result.priority == "HIGH PRIORITY"
    assert result.manual_review and len(result.reasons) == 4

def test_temperature_scaling_is_probability():
    scaled = temperature_scale([.1, .1, .2, .2, .4], 1.5)
    assert scaled.sum() == pytest.approx(1)
    summary = uncertainty_summary(scaled)
    assert 0 <= summary["high_risk_probability"] <= 1
    calibration = fit_temperature(np.array([[.8,.05,.05,.05,.05],[.05,.7,.1,.1,.05]]), [0,1])
    assert .5 <= calibration["temperature"] <= 3
    selected = threshold_for_target_sensitivity([1,1,0,0], [.9,.7,.4,.2], .9)
    assert selected["sensitivity"] >= .9

def test_preprocessing_valid_and_corrupt(retinal_bytes):
    result = preprocess_image(retinal_bytes, 224)
    assert result.model_input.shape == (1, 224, 224, 3)
    assert result.processed.size == (224, 224)
    with pytest.raises(ImageValidationError):
        preprocess_image(b"not an image")

def test_quality_calculations(retinal_bytes):
    image = Image.open(io.BytesIO(retinal_bytes))
    thresholds = load_config()["quality"]
    result = assess_quality(image, thresholds)
    assert 0 <= result["quality_score"] <= 1
    assert result["minimum_score"] == pytest.approx(.75)
    assert result["gradable"] == (result["quality_score"] >= .75)
    assert "blur_score" in result["metrics"]
    assert "not clinically validated" in result["method"]
    assert "below 75/100" in result["disclaimer"]

def test_quality_threshold_is_inclusive_and_routes_below_to_review(retinal_bytes):
    image = Image.open(io.BytesIO(retinal_bytes))
    thresholds = load_config()["quality"]
    baseline = assess_quality(image, thresholds)
    at_threshold = {**thresholds, "minimum_score": baseline["quality_score"]}
    assert assess_quality(image, at_threshold)["gradable"]
    above_score = {**thresholds, "minimum_score": min(1.0, baseline["quality_score"] + .01)}
    failed = assess_quality(image, above_score)
    assert not failed["gradable"]
    assert failed["decision"] == "manual_review"

def test_config_missing_section(tmp_path):
    path = tmp_path / "bad.json"; path.write_text('{"paths": {}}')
    with pytest.raises(ConfigError): load_config(path)

def test_duplicate_hashes_cannot_cross_splits():
    import pandas as pd
    rows = []
    for grade in range(5):
        for index in range(8):
            rows.append({"grade": grade, "sha256": f"{grade}-{index}", "image_path": f"{grade}-{index}.png"})
    rows.append({"grade": 0, "sha256": "0-0", "image_path": "duplicate.png"})
    splits = stratified_split(pd.DataFrame(rows), seed=11)
    hashes = [set(frame["sha256"]) for frame in splits.values()]
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])

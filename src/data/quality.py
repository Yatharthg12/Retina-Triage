from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

def assess_quality(image: Image.Image, thresholds: dict) -> dict:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    under = float(np.mean(gray < 15))
    over = float(np.mean(gray > 245))
    field_mask = gray > 10
    coverage = float(field_mask.mean())
    contours, _ = cv2.findContours(field_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circle_detected = False
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
        circle_detected = bool(area > w*h*0.2 and circularity > 0.35)
    observations = []
    checks = [
        (w < thresholds["min_width"] or h < thresholds["min_height"], "Resolution is below the configured minimum."),
        (blur < thresholds["min_blur"], "Image appears blurred."),
        (brightness < thresholds["min_brightness"], "Image is excessively dark."),
        (brightness > thresholds["max_brightness"], "Image is excessively bright."),
        (under > thresholds["max_underexposed_ratio"], "Too much of the image is underexposed."),
        (over > thresholds["max_overexposed_ratio"], "Too much of the image is overexposed."),
        (contrast < thresholds["min_contrast"], "Image contrast is too low."),
        (coverage < thresholds["min_field_coverage"], "Retinal field coverage is insufficient."),
        (not circle_detected, "A plausible retinal field was not detected."),
    ]
    observations.extend(message for failed, message in checks if failed)
    resolution_score = min(1.0, w / max(thresholds["min_width"], 1), h / max(thresholds["min_height"], 1))
    if brightness < thresholds["min_brightness"]:
        brightness_score = brightness / max(thresholds["min_brightness"], 1)
    elif brightness > thresholds["max_brightness"]:
        brightness_score = (255.0 - brightness) / max(255.0 - thresholds["max_brightness"], 1)
    else:
        brightness_score = 1.0
    individual = [
        max(0.0, resolution_score),
        min(1.0, blur / max(thresholds["min_blur"] * 2, 1)),
        max(0.0, min(1.0, brightness_score)),
        min(1.0, contrast / max(thresholds["min_contrast"] * 2, 1)),
        min(1.0, coverage / max(thresholds["min_field_coverage"] * 1.5, 0.01)),
        1.0 - min(1.0, under),
        1.0 - min(1.0, over),
        1.0 if circle_detected else 0.25,
    ]
    quality_score = round(float(np.mean(individual)), 4)
    minimum_score = float(thresholds.get("minimum_score", 0.75))
    gradable = quality_score >= minimum_score
    threshold_points = round(minimum_score * 100)
    decision_reason = (
        f"Composite quality score meets the required {threshold_points}/100 threshold."
        if gradable
        else f"Composite quality score is below the required {threshold_points}/100 threshold."
    )
    issues = list(observations)
    if not gradable:
        issues.insert(0, decision_reason)
    return {
        "gradable": gradable,
        "quality_score": quality_score,
        "minimum_score": minimum_score,
        "decision": "pass" if gradable else "manual_review",
        "decision_reason": decision_reason,
        "issues": issues,
        "observations": observations,
        "metrics": {
            "width": w, "height": h, "aspect_ratio": round(w / h, 4),
            "blur_score": round(blur, 3), "mean_luminance": round(brightness, 3),
            "luminance_std": round(contrast, 3), "underexposed_ratio": round(under, 4),
            "overexposed_ratio": round(over, 4), "contrast": round(contrast, 3),
            "field_coverage": round(coverage, 4), "circular_field_detected": circle_detected,
        },
        "thresholds": thresholds,
        "method": (
            f"Composite heuristic quality gate with a {threshold_points}/100 acceptance threshold; "
            "not clinically validated."
        ),
        "disclaimer": (
            f"Scores below {threshold_points}/100 require retake or manual review. "
            f"A score of {threshold_points}/100 or above only passes this software heuristic; "
            "it does not guarantee clinical gradability or diagnostic reliability."
        ),
    }

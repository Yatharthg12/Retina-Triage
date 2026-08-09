from __future__ import annotations

from dataclasses import asdict, dataclass

PRIORITY_RANK = {
    "URGENT – HIGH PRIORITY": 0,
    "HIGH PRIORITY": 1,
    "RETAKE / MANUAL REVIEW": 2,
    "SPECIALIST REVIEW": 3,
    "FOLLOW-UP": 4,
    "ROUTINE": 5,
}

@dataclass(frozen=True)
class TriageResult:
    priority: str
    manual_review: bool
    urgent: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

def apply_triage(
    grade: int,
    gradable: bool,
    confidence: float,
    entropy: float,
    top_two_margin: float,
    high_risk_probability: float,
    thresholds: dict,
    quality_threshold: float = 0.75,
) -> TriageResult:
    reasons: list[str] = []
    if not gradable:
        threshold_points = round(float(quality_threshold) * 100)
        return TriageResult(
            "RETAKE / MANUAL REVIEW", True, False,
            [
                f"Image quality score is below the required {threshold_points}/100 threshold; "
                "manual review is required."
            ],
        )
    priorities = {
        0: "ROUTINE", 1: "FOLLOW-UP", 2: "SPECIALIST REVIEW",
        3: "HIGH PRIORITY", 4: "URGENT – HIGH PRIORITY",
    }
    priority = priorities[int(grade)]
    if confidence < float(thresholds["low_confidence"]):
        reasons.append("Model confidence is below the review threshold.")
    if entropy > float(thresholds["high_entropy"]):
        reasons.append("Predictive entropy is above the review threshold.")
    if top_two_margin < float(thresholds["low_margin"]):
        reasons.append("The leading severity probabilities are too close.")
    if high_risk_probability >= float(thresholds["high_risk_probability"]):
        if PRIORITY_RANK[priority] > PRIORITY_RANK["HIGH PRIORITY"]:
            priority = "HIGH PRIORITY"
        reasons.append("High-risk disease probability crossed the escalation threshold.")
    if grade >= 3:
        reasons.append("High-risk grade requires ophthalmologist review.")
    return TriageResult(priority, bool(reasons), grade == 4, reasons)

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.constants import CLASS_NAMES
from src.config import load_config

def generate_screening_report(record: dict, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    quality_threshold = float(load_config()["quality"].get("minimum_score", 0.75))
    threshold_points = round(quality_threshold * 100)
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm)
    story = [
        Paragraph("RetinaTriage AI", styles["Title"]),
        Paragraph("Explainable AI-Assisted DR Screening & Clinical Prioritization", styles["Heading2"]),
        Spacer(1, 8),
        Paragraph("<b>RESEARCH USE ONLY — NOT A DIAGNOSIS OR MEDICAL DEVICE</b>", styles["Normal"]),
        Spacer(1, 12),
    ]
    rows = [
        ["Screening ID", record["screening_id"]],
        ["Anonymous case ID", record.get("case_id") or "Not supplied"],
        ["Timestamp", record["created_at"]],
        ["Model version", record["model_version"]],
        ["Image quality score", (
            f"{record['quality_score'] * 100:.0f} / 100"
            if record.get("quality_score") is not None else "Not available"
        )],
        ["Quality acceptance threshold", f"{threshold_points} / 100"],
        ["Quality issues", "; ".join(record.get("quality_issues", [])) or "None detected"],
        ["Predicted severity", (
            f"Grade {record['predicted_grade']} — {record['predicted_label']}"
            if record.get("predicted_grade") is not None else "No disease result"
        )],
        ["Model confidence", (
            f"{record['confidence']:.1%}" if record.get("confidence") is not None else "Not available"
        )],
        ["Referable probability", (
            f"{record['referable_probability']:.1%}" if record.get("referable_probability") is not None else "Not available"
        )],
        ["High-risk probability", (
            f"{record['high_risk_probability']:.1%}" if record.get("high_risk_probability") is not None else "Not available"
        )],
        ["Priority", record["priority"]],
        ["Manual review reasons", "; ".join(record.get("review_reasons", [])) or "None"],
    ]
    table = Table(rows, colWidths=[48*mm, 112*mm], repeatRows=0)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f0f3")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#122d3d")),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#b8c9d1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([table, Spacer(1, 14)])
    if record.get("probabilities"):
        probability_rows = [["Grade", "Class", "Probability"]]
        for grade, probability in enumerate(record["probabilities"]):
            probability_rows.append([str(grade), CLASS_NAMES[grade], f"{probability:.1%}"])
        ptable = Table(probability_rows, colWidths=[18*mm, 112*mm, 30*mm])
        ptable.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12354a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#b8c9d1")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([Paragraph("Five-class output", styles["Heading3"]), ptable, Spacer(1, 14)])
    story.append(Paragraph(
        f"This prototype supports screening research and queue prioritization only. Scores below "
        f"{threshold_points}/100 require retake or manual review; a score of {threshold_points}/100 or above passes "
        "only an unvalidated software heuristic and does not guarantee clinical gradability. "
        "Grad-CAM, when present, indicates regions influencing model output and is not a lesion boundary or clinical "
        "annotation. An ophthalmologist must review the image and make every final clinical decision.",
        styles["BodyText"],
    ))
    doc.build(story)
    return path

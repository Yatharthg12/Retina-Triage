from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

from src.database.connection import connect

JSON_FIELDS = {"quality_issues", "review_reasons", "probabilities"}

class ScreeningRepository:
    def __init__(self, path: str):
        self.path = path

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        data = dict(row)
        for field in JSON_FIELDS:
            data[field] = json.loads(data.get(field) or "[]")
        data["manual_review"] = bool(data["manual_review"])
        data["simulated"] = bool(data["simulated"])
        return data

    def insert(self, record: dict) -> dict:
        values = {
            "screening_id": record["screening_id"],
            "case_id": record.get("case_id"),
            "created_at": record.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "original_filename": record.get("original_filename", "unnamed"),
            "file_hash": record.get("file_hash", ""),
            "image_width": record.get("image_width"),
            "image_height": record.get("image_height"),
            "quality_score": record.get("quality_score"),
            "quality_issues": json.dumps(record.get("quality_issues", [])),
            "predicted_grade": record.get("predicted_grade"),
            "predicted_label": record.get("predicted_label"),
            "confidence": record.get("confidence"),
            "referable_probability": record.get("referable_probability"),
            "high_risk_probability": record.get("high_risk_probability"),
            "priority": record["priority"],
            "manual_review": int(record.get("manual_review", False)),
            "review_reasons": json.dumps(record.get("review_reasons", [])),
            "model_version": record.get("model_version", "unavailable"),
            "processing_time_ms": record.get("processing_time_ms", 0),
            "report_path": record.get("report_path"),
            "simulated": int(record.get("simulated", False)),
            "probabilities": json.dumps(record.get("probabilities", [])),
        }
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        with connect(self.path) as db:
            db.execute(f"INSERT INTO screenings ({columns}) VALUES ({placeholders})", tuple(values.values()))
        return self.get(values["screening_id"])

    def get(self, screening_id: str):
        with connect(self.path) as db:
            return self._decode(db.execute("SELECT * FROM screenings WHERE screening_id = ?", (screening_id,)).fetchone())

    def list(self, limit=100, search=None, priority=None, grade=None, manual_review=None):
        clauses, params = [], []
        if search:
            clauses.append("(case_id LIKE ? OR original_filename LIKE ? OR screening_id LIKE ?)")
            params.extend([f"%{search}%"] * 3)
        if priority:
            clauses.append("priority = ?"); params.append(priority)
        if grade is not None:
            clauses.append("predicted_grade = ?"); params.append(int(grade))
        if manual_review is not None:
            clauses.append("manual_review = ?"); params.append(int(manual_review))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with connect(self.path) as db:
            rows = db.execute(f"SELECT * FROM screenings{where} ORDER BY created_at DESC LIMIT ?", (*params, min(int(limit), 500))).fetchall()
        return [self._decode(row) for row in rows]

    def delete(self, screening_id: str) -> bool:
        with connect(self.path) as db:
            cursor = db.execute("DELETE FROM screenings WHERE screening_id = ?", (screening_id,))
        return cursor.rowcount > 0

    def summary(self) -> dict:
        with connect(self.path) as db:
            total = db.execute("SELECT COUNT(*) FROM screenings").fetchone()[0]
            row = db.execute("""
                SELECT
                  SUM(CASE WHEN priority IN ('URGENT – HIGH PRIORITY','HIGH PRIORITY') THEN 1 ELSE 0 END),
                  SUM(manual_review),
                  SUM(CASE WHEN priority = 'RETAKE / MANUAL REVIEW' THEN 1 ELSE 0 END),
                  AVG(processing_time_ms)
                FROM screenings
            """).fetchone()
            severities = db.execute("SELECT predicted_grade, COUNT(*) count FROM screenings WHERE predicted_grade IS NOT NULL GROUP BY predicted_grade").fetchall()
            priorities = db.execute("SELECT priority, COUNT(*) count FROM screenings GROUP BY priority").fetchall()
        return {
            "total_screenings": total, "high_priority": row[0] or 0,
            "manual_review": row[1] or 0, "poor_quality": row[2] or 0,
            "average_processing_time_ms": round(row[3] or 0, 1),
            "severity_distribution": {str(r[0]): r[1] for r in severities},
            "priority_distribution": {r[0]: r[1] for r in priorities},
        }

    def export_csv(self) -> str:
        rows = self.list(limit=500)
        output = io.StringIO()
        fields = ["screening_id", "case_id", "created_at", "original_filename", "predicted_grade",
                  "predicted_label", "confidence", "priority", "manual_review", "model_version"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
        return output.getvalue()


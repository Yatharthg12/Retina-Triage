from __future__ import annotations

import io
import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from src.data.preprocessing import ImageValidationError
from src.services.reports import generate_screening_report
from src.services.screening import persist_result

api = Blueprint("api", __name__)

def ok(data, status=200):
    return jsonify({"success": True, "data": data, "error": None}), status

def fail(code, message, status=400, details=None, data=None):
    return jsonify({"success": False, "data": data, "error": {
        "code": code, "message": message, "details": details or {}
    }}), status

def _validate_upload(file):
    if not file or not file.filename:
        raise ValueError("Select a JPEG or PNG image.")
    name = Path(file.filename).name
    extension = Path(name).suffix.lower().lstrip(".")
    allowed = current_app.config["SETTINGS"]["uploads"]["allowed_extensions"]
    if extension not in allowed:
        raise ValueError("Only JPG, JPEG and PNG files are accepted.")
    return name

def _run_file(file, case_id=None, include_gradcam=False):
    name = _validate_upload(file)
    raw = file.read()
    if not raw:
        raise ImageValidationError("The uploaded file is empty.")
    predictor = current_app.extensions["predictor"]
    result = predictor.predict_bytes(raw, name, case_id, include_gradcam=include_gradcam)
    if result["model_available"]:
        persist_result(current_app.extensions["repository"], result)
    return result

@api.get("/health")
def health():
    return ok({"status": "healthy", "service": "RetinaTriage AI"})

@api.get("/model/status")
def model_status():
    return ok(current_app.extensions["predictor"].status())

@api.get("/dashboard/summary")
def dashboard_summary():
    data = current_app.extensions["repository"].summary()
    data["model"] = current_app.extensions["predictor"].status()
    data["recent"] = current_app.extensions["repository"].list(limit=8)
    return ok(data)

@api.post("/predict")
def predict():
    try:
        result = _run_file(
            request.files.get("image"), request.form.get("case_id") or None,
            request.form.get("include_gradcam", "true").lower() == "true"
        )
    except ImageValidationError as exc:
        return fail("INVALID_IMAGE", str(exc), 415)
    except ValueError as exc:
        return fail("INVALID_UPLOAD", str(exc), 400)
    except RequestEntityTooLarge:
        raise
    except Exception:
        current_app.logger.exception("Prediction failed")
        return fail("PREDICTION_FAILED", "The screening could not be completed.", 500)
    if not result["model_available"]:
        return fail("MODEL_UNAVAILABLE", "A trained model is not installed. No disease prediction was generated.",
                    503, data=result)
    return ok(result, 201)

@api.post("/predict/batch")
def predict_batch():
    files = request.files.getlist("images")
    maximum = current_app.config["SETTINGS"]["uploads"]["max_batch_files"]
    if not files:
        return fail("EMPTY_BATCH", "Select at least one image.", 400)
    if len(files) > maximum:
        return fail("BATCH_TOO_LARGE", f"A batch may contain at most {maximum} images.", 400)
    results = []
    for file in files:
        try:
            result = _run_file(file)
            results.append({"filename": Path(file.filename).name, "success": result["model_available"], "data": result,
                            "error": None if result["model_available"] else "MODEL_UNAVAILABLE"})
        except (ValueError, ImageValidationError) as exc:
            results.append({"filename": Path(file.filename or "unnamed").name, "success": False, "data": None, "error": str(exc)})
    if not current_app.extensions["predictor"].available:
        return fail("MODEL_UNAVAILABLE", "A trained model is not installed.", 503, data={"results": results})
    return ok({"results": results, "total": len(results), "succeeded": sum(x["success"] for x in results)}, 207)

@api.get("/predictions")
def predictions():
    repo = current_app.extensions["repository"]
    if request.args.get("format") == "csv":
        return send_file(io.BytesIO(repo.export_csv().encode()), mimetype="text/csv",
                         as_attachment=True, download_name="retina-triage-history.csv")
    rows = repo.list(
        request.args.get("limit", 100), request.args.get("search"), request.args.get("priority"),
        request.args.get("grade"), request.args.get("manual_review", type=lambda x: x.lower() == "true")
    )
    return ok(rows)

@api.get("/predictions/<screening_id>")
def prediction_detail(screening_id):
    record = current_app.extensions["repository"].get(screening_id)
    return ok(record) if record else fail("NOT_FOUND", "Screening record not found.", 404)

@api.delete("/predictions/<screening_id>")
def prediction_delete(screening_id):
    deleted = current_app.extensions["repository"].delete(screening_id)
    return ok({"deleted": True}) if deleted else fail("NOT_FOUND", "Screening record not found.", 404)

@api.get("/predictions/<screening_id>/report")
def report(screening_id):
    repo = current_app.extensions["repository"]
    record = repo.get(screening_id)
    if not record:
        return fail("NOT_FOUND", "Screening record not found.", 404)
    path = Path(current_app.config["SETTINGS"]["paths"]["reports"]) / f"{screening_id}.pdf"
    generate_screening_report(record, path)
    return send_file(path, as_attachment=True, download_name=f"retina-triage-{screening_id[:8]}.pdf")

@api.get("/evaluation")
def evaluation():
    directory = Path(current_app.config["SETTINGS"]["paths"]["evaluation"])
    metrics = directory / "metrics.json"
    if not metrics.exists():
        return ok({"status": "not_evaluated", "metrics": None,
                   "published_baseline": current_app.extensions["predictor"].status().get("published_evaluation"),
                   "command": "python -m src.training.evaluate --config configs/default.json"})
    return ok({"status": "evaluated", "metrics": json.loads(metrics.read_text(encoding="utf-8"))})

def _demo_manifest():
    root = Path(current_app.config["SETTINGS"]["_root"]) / "demo" / "samples"
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return root, {"notice": "Demo samples are not installed.", "samples": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = []
    for item in manifest.get("samples", []):
        filename = Path(str(item.get("filename", ""))).name
        if filename == item.get("filename") and (root / filename).is_file():
            available.append({**item, "url": f"/api/demo/samples/{filename}"})
    return root, {**manifest, "samples": available}

@api.get("/demo/samples")
def demo_samples():
    _, manifest = _demo_manifest()
    return ok(manifest)

@api.get("/demo/samples/<filename>")
def demo_sample(filename):
    root, manifest = _demo_manifest()
    allowed = {item["filename"] for item in manifest["samples"]}
    if filename not in allowed:
        return fail("NOT_FOUND", "Demo sample not found.", 404)
    path = root / filename
    return send_file(path, conditional=True, download_name=filename)

@api.get("/model-card")
def model_card():
    path = Path(current_app.config["SETTINGS"]["_root"]) / "docs/MODEL_CARD.md"
    return ok({"content": path.read_text(encoding="utf-8") if path.exists() else "Model card unavailable."})

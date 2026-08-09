from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference.predictor import Predictor
from src.web import create_app


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    print(f"PASS {message}")


def main() -> None:
    settings = load_config()
    predictor = Predictor(settings)
    status = predictor.status()
    check(status["available"], f"model ready ({status.get('error') or status['version']})")
    check(status["calibration"]["status"] == "not_calibrated", "confidence is honestly marked uncalibrated")
    check(status["quality_gate"]["minimum_score"] == 0.75, "quality acceptance threshold is 75/100")

    sample_root = ROOT / "demo" / "samples"
    manifest = json.loads((sample_root / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["samples"]:
        path = sample_root / item["filename"]
        check(path.is_file(), f"sample present: {item['filename']}")
        check(digest(path) == item["sha256"], f"sample checksum: {item['filename']}")

    primary = sample_root / "diabetic_retinopathy_cdc_pd.jpg"
    direct = predictor.predict_bytes(primary.read_bytes(), primary.name, "VERIFY-DIRECT", include_gradcam=True)
    check(direct["quality"]["gradable"], "primary sample meets the 75/100 quality threshold")
    check(direct["prediction"] is not None, "direct model inference returns five-grade output")
    check(direct["prediction"]["confidence_kind"] == "raw_softmax", "prediction confidence label")
    check(direct["explanation"].get("overlay", "").startswith("data:image/png;base64,"), "Grad-CAM overlay")
    check(not direct["simulated"], "prediction is not simulated")

    instance = ROOT / "instance"
    instance.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="presentation-verify-", dir=instance) as temporary:
        temp_root = Path(temporary)
        test_settings = json.loads(json.dumps(settings))
        test_settings.pop("_root", None)
        test_settings["paths"]["database"] = str(temp_root / "verification.sqlite3")
        test_settings["paths"]["reports"] = str(temp_root / "reports")
        config_path = temp_root / "config.json"
        config_path.write_text(json.dumps(test_settings), encoding="utf-8")
        app = create_app(
            {"TESTING": True, "CONFIG_PATH": str(config_path)},
            predictor=Predictor(load_config(config_path)),
        )
        client = app.test_client()
        for route in (
            "/", "/analyze", "/batch", "/model", "/history", "/about",
            "/api/health", "/api/model/status", "/api/evaluation", "/api/demo/samples",
        ):
            check(client.get(route).status_code == 200, f"route {route}")
        samples = client.get("/api/demo/samples").get_json()["data"]["samples"]
        check(len(samples) == len(manifest["samples"]), "sample manifest API")

        response = client.post(
            "/api/predict",
            data={
                "image": (io.BytesIO(primary.read_bytes()), primary.name),
                "case_id": "VERIFY-E2E",
                "include_gradcam": "true",
            },
            content_type="multipart/form-data",
        )
        check(response.status_code == 201, "HTTP upload and inference")
        result = response.get_json()["data"]
        check(result["prediction"] is not None, "HTTP prediction payload")
        screening_id = result["screening_id"]
        report = client.get(f"/api/predictions/{screening_id}/report")
        check(report.status_code == 200 and report.mimetype == "application/pdf", "PDF report generation")
        report.close()
        history = client.get(f"/api/predictions/{screening_id}")
        check(history.status_code == 200, "SQLite audit persistence")
        check(client.delete(f"/api/predictions/{screening_id}").status_code == 200, "audit deletion workflow")

    print("\nPRESENTATION READY")
    print("Launch with: python scripts/run_presentation.py")
    print("Open: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()

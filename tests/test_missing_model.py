import io
import json

from src.config import load_config
from src.inference.predictor import Predictor
from src.web import create_app

def test_missing_model_clean_response(tmp_path, retinal_bytes):
    config = load_config()
    config["paths"]["database"] = str(tmp_path / "missing.sqlite3")
    config["paths"]["model"] = str(tmp_path / "does-not-exist.keras")
    config.pop("_root", None)
    path = tmp_path / "config.json"; path.write_text(json.dumps(config))
    loaded = load_config(path)
    app = create_app({"TESTING": True, "CONFIG_PATH": str(path)}, Predictor(loaded))
    response = app.test_client().post("/api/predict", data={
        "image": (io.BytesIO(retinal_bytes), "eye.png")
    }, content_type="multipart/form-data")
    body = response.get_json()
    assert response.status_code == 503
    assert body["error"]["code"] == "MODEL_UNAVAILABLE"
    assert body["data"]["prediction"] is None

def test_model_checksum_mismatch_is_rejected(tmp_path):
    config = load_config()
    model = tmp_path / "model.keras"
    model.write_bytes(b"not-the-declared-model")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({
        "model_version": "checksum-test",
        "artifact_sha256": "0" * 64,
    }))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "temperature": 1.0,
        "status": "not_calibrated",
        "method": "identity",
    }))
    config["paths"]["model"] = str(model)
    config["paths"]["model_metadata"] = str(metadata)
    config["paths"]["calibration"] = str(calibration)
    predictor = Predictor(config)
    assert not predictor.available
    assert "checksum" in predictor.status()["error"].lower()

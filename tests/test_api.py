import io

def test_pages_and_health(client):
    for route in ["/", "/analyze", "/batch", "/model", "/history", "/about"]:
        assert client.get(route).status_code == 200
    payload = client.get("/api/health").get_json()
    assert payload["success"] and payload["data"]["status"] == "healthy"

def test_model_status(client):
    data = client.get("/api/model/status").get_json()["data"]
    assert data["available"]
    assert data["calibration"]["status"] == "not_calibrated"

def test_valid_upload(client, retinal_bytes):
    response = client.post("/api/predict", data={
        "image": (io.BytesIO(retinal_bytes), "retina.png"), "case_id": "CASE-1"
    }, content_type="multipart/form-data")
    assert response.status_code == 201
    data = response.get_json()["data"]
    assert data["prediction"]["grade"] == 4
    assert data["prediction"]["confidence_kind"] == "raw_softmax"
    assert data["quality"]["minimum_score"] == .75
    assert data["quality"]["gradable"] == (data["quality"]["quality_score"] >= .75)
    assert data["triage"]["urgent"]

def test_quality_below_75_withholds_prediction_and_requires_review(client):
    from PIL import Image
    import numpy as np

    buffer = io.BytesIO()
    Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8)).save(buffer, "PNG")
    response = client.post("/api/predict", data={
        "image": (io.BytesIO(buffer.getvalue()), "dark-retina.png")
    }, content_type="multipart/form-data")
    assert response.status_code == 201
    data = response.get_json()["data"]
    assert data["quality"]["quality_score"] < .75
    assert not data["quality"]["gradable"]
    assert data["prediction"] is None
    assert data["advanced_model_output"] is None
    assert data["triage"]["manual_review"]
    assert "75/100" in data["triage"]["reasons"][0]

def test_demo_sample_pack(client):
    response = client.get("/api/demo/samples")
    assert response.status_code == 200
    samples = response.get_json()["data"]["samples"]
    assert len(samples) >= 3
    sample = client.get(samples[0]["url"])
    assert sample.status_code == 200
    assert sample.mimetype in {"image/jpeg", "image/png"}
    assert client.get("/api/demo/samples/not-installed.jpg").status_code == 404

def test_invalid_extension_and_payload(client):
    response = client.post("/api/predict", data={"image": (io.BytesIO(b"x"), "../evil.exe")},
                           content_type="multipart/form-data")
    assert response.status_code == 400
    response = client.post("/api/predict", data={"image": (io.BytesIO(b"not-image"), "../../retina.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 415

def test_batch_partial_failure(client, retinal_bytes):
    response = client.post("/api/predict/batch", data={"images": [
        (io.BytesIO(retinal_bytes), "good.png"), (io.BytesIO(b"broken"), "../bad.png")
    ]}, content_type="multipart/form-data")
    assert response.status_code == 207
    data = response.get_json()["data"]
    assert data["succeeded"] == 1 and len(data["results"]) == 2

def test_oversized_upload_rejected(client):
    response = client.post("/api/predict", data={
        "image": (io.BytesIO(b"x" * (13 * 1024 * 1024)), "large.png")
    }, content_type="multipart/form-data")
    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "UPLOAD_TOO_LARGE"

def test_history_delete_and_report(client, retinal_bytes):
    created = client.post("/api/predict", data={"image": (io.BytesIO(retinal_bytes), "eye.png")},
                          content_type="multipart/form-data").get_json()["data"]
    screening_id = created["screening_id"]
    assert client.get(f"/api/predictions/{screening_id}/report").status_code == 200
    assert client.delete(f"/api/predictions/{screening_id}").status_code == 200
    assert client.get(f"/api/predictions/{screening_id}").status_code == 404

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.config import load_config
from src.inference.predictor import Predictor
from src.web import create_app

class MockModel:
    input_shape = (None, 224, 224, 3)
    output_shape = (None, 5)
    def predict(self, data, verbose=0):
        return np.array([[.04, .06, .16, .23, .51]], dtype=np.float32)

@pytest.fixture
def retinal_bytes():
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:320, :320]
    mask = (xx-160)**2 + (yy-160)**2 <= 145**2
    image[mask] = [155, 72, 42]
    image[130:150, 140:160] = [240, 200, 110]
    buffer = io.BytesIO(); Image.fromarray(image).save(buffer, "PNG")
    return buffer.getvalue()

@pytest.fixture
def app(tmp_path):
    base = load_config()
    config = json.loads(json.dumps(base))
    config["paths"]["database"] = str(tmp_path / "test.sqlite3")
    config["paths"]["reports"] = str(tmp_path / "reports")
    config_path = tmp_path / "config.json"
    config.pop("_root", None)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    predictor = Predictor(load_config(config_path), model=MockModel())
    return create_app({"TESTING": True, "CONFIG_PATH": str(config_path)}, predictor=predictor)

@pytest.fixture
def client(app):
    return app.test_client()


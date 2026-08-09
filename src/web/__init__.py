from __future__ import annotations

import os

from flask import Flask, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from src.config import load_config
from src.database.connection import initialize
from src.database.repository import ScreeningRepository
from src.inference.predictor import Predictor
from src.logging_config import configure_logging
from src.web.api import api
from src.web.routes import pages

def create_app(test_config: dict | None = None, predictor=None) -> Flask:
    settings = load_config((test_config or {}).get("CONFIG_PATH"))
    configure_logging(settings["logging"]["level"])
    app = Flask(__name__, template_folder="../../templates", static_folder="../../static")
    app.config.update(
        SECRET_KEY=os.getenv("RETINATRIAGE_SECRET_KEY", "development-only-change-me"),
        MAX_CONTENT_LENGTH=settings["uploads"]["max_file_mb"] * 1024 * 1024,
        SETTINGS=settings,
        TESTING=bool((test_config or {}).get("TESTING", False)),
    )
    if test_config:
        app.config.update(test_config)
    initialize(settings["paths"]["database"])
    app.extensions["repository"] = ScreeningRepository(settings["paths"]["database"])
    app.extensions["predictor"] = predictor or Predictor(settings)
    app.register_blueprint(pages)
    app.register_blueprint(api, url_prefix="/api")

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        if request.path.startswith("/api/"):
            return {"success": False, "data": None, "error": {
                "code": "UPLOAD_TOO_LARGE", "message": "The upload exceeds the configured size limit.", "details": {}
            }}, 413
        return render_template("error.html", code=413, message="Upload too large"), 413

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", code=404, message="Page not found"), 404
    return app

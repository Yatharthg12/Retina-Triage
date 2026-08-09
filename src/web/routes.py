from __future__ import annotations

from flask import Blueprint, current_app, render_template

pages = Blueprint("pages", __name__)

@pages.app_context_processor
def inject_globals():
    return {"model_status": current_app.extensions["predictor"].status()}

@pages.get("/")
def dashboard():
    return render_template("dashboard.html", page="dashboard")

@pages.get("/analyze")
def analyze():
    return render_template("analyze.html", page="analyze")

@pages.get("/batch")
def batch():
    return render_template("batch.html", page="batch")

@pages.get("/model")
def model():
    return render_template("model.html", page="model")

@pages.get("/history")
def history():
    return render_template("history.html", page="history")

@pages.get("/about")
def about():
    return render_template("about.html", page="about")


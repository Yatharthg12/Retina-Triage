from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.web import create_app


def main() -> None:
    try:
        from waitress import serve
    except ImportError as exc:
        raise SystemExit(
            "Waitress is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc
    app = create_app()
    status = app.extensions["predictor"].status()
    if not status["available"]:
        raise SystemExit(
            f"Presentation model is not ready: {status.get('error') or status['status']}. "
            "Run: python scripts/setup_presentation.py"
        )
    host = os.getenv("RETINATRIAGE_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    print("RetinaTriage presentation server")
    print(f"Model: {status['version']}")
    print(f"Confidence: {status['calibration']['status']}")
    print(f"Open: http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    serve(app, host=host, port=port, threads=4, expose_tracebacks=False)


if __name__ == "__main__":
    main()

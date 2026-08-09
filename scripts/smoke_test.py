from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.web import create_app

def main():
    app = create_app({"TESTING": True})
    client = app.test_client()
    failures = []
    for route in ["/", "/analyze", "/batch", "/model", "/history", "/about", "/api/health", "/api/model/status", "/api/evaluation", "/api/demo/samples"]:
        response = client.get(route)
        label = "PASS" if response.status_code == 200 else "FAIL"
        print(f"{label} {route} -> {response.status_code}")
        if response.status_code != 200: failures.append(route)
    if failures:
        raise SystemExit(f"Smoke test failed: {failures}")
    print("Smoke test passed: 10 routes.")

if __name__ == "__main__":
    main()

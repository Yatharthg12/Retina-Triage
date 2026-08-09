from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from importlib import metadata as package_metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = {
    "url": (
        "https://huggingface.co/Aldahmashi/DR-EfficientNetB0/resolve/"
        "fb8d14c59bd56aa17fe0dfdea04a83ecd2f2eeac/final_model.keras?download=true"
    ),
    "path": ROOT / "artifacts" / "models" / "best_model.keras",
    "sha256": "e7aa6b69911a2a913a03a6a5669bb7aaeb6bf8f8c81a2a3d92aa2420e5d297d8",
}
REQUIRED_KERAS = "3.13.2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, expected_hash: str, force: bool = False) -> None:
    if target.exists() and sha256(target) == expected_hash:
        print(f"READY {target.relative_to(ROOT)}")
        return
    if target.exists() and not force:
        raise SystemExit(
            f"Checksum mismatch: {target}. Re-run with --force only if replacing this file is intended."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "RetinaTriage/1.0 educational-demo"})
    for attempt in range(1, 4):
        try:
            print(f"DOWNLOAD {target.relative_to(ROOT)} (attempt {attempt}/3)")
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            break
        except (OSError, urllib.error.URLError) as exc:
            if partial.exists():
                partial.unlink()
            if attempt == 3:
                raise SystemExit(f"Download failed for {target.name}: {exc}") from exc
            time.sleep(attempt)
    actual_hash = sha256(partial)
    if actual_hash != expected_hash:
        partial.unlink()
        raise SystemExit(
            f"Downloaded checksum mismatch for {target.name}: expected {expected_hash}, got {actual_hash}"
        )
    partial.replace(target)
    print(f"VERIFIED {target.relative_to(ROOT)}")


def check_runtime() -> None:
    try:
        installed = package_metadata.version("keras")
    except package_metadata.PackageNotFoundError as exc:
        raise SystemExit("Keras is not installed. Run: python -m pip install -r requirements.txt") from exc
    if installed != REQUIRED_KERAS:
        raise SystemExit(
            f"Keras {REQUIRED_KERAS} is required by the presentation model; found {installed}. "
            "Run: python -m pip install -r requirements.txt"
        )
    print(f"READY Keras {installed}")
    try:
        waitress_version = package_metadata.version("waitress")
    except package_metadata.PackageNotFoundError as exc:
        raise SystemExit("Waitress is not installed. Run: python -m pip install -r requirements.txt") from exc
    print(f"READY Waitress {waitress_version}")


def setup_samples(force: bool) -> None:
    sample_root = ROOT / "demo" / "samples"
    manifest = json.loads((sample_root / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["samples"]:
        target = sample_root / item["filename"]
        download(item["download_url"], target, item["sha256"], force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install and verify RetinaTriage presentation assets.")
    parser.add_argument("--force", action="store_true", help="Replace an existing asset only after checksum review.")
    parser.add_argument("--skip-samples", action="store_true", help="Install only the model artifact.")
    args = parser.parse_args()
    check_runtime()
    download(MODEL["url"], MODEL["path"], MODEL["sha256"], force=args.force)
    if not args.skip_samples:
        setup_samples(force=args.force)
    from src.config import load_config
    from src.database.connection import initialize

    initialize(load_config()["paths"]["database"])
    print("READY database")
    print("Presentation assets are installed. Next: python scripts/verify_presentation.py")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()

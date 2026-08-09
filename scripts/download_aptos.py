from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

def safe_extract(archive: Path, destination: Path):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(destination)

def main():
    parser = argparse.ArgumentParser(description="Download APTOS 2019 using the official Kaggle CLI.")
    parser.add_argument("--output", default="data/raw/aptos")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit("Destination is not empty. Use --force only after reviewing its contents.")
    if importlib.util.find_spec("kaggle") is None:
        raise SystemExit("Kaggle is not installed. Run: python -m pip install kaggle")
    credential = Path(os.getenv("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not (credential.exists() or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))):
        raise SystemExit("Kaggle credentials were not found. Configure kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY.")
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "kaggle", "competitions", "download",
                    "-c", "aptos2019-blindness-detection", "-p", str(output)], check=True)
    for archive in output.glob("*.zip"):
        safe_extract(archive, output)
    if not (output / "train.csv").exists() or not (output / "train_images").exists():
        raise SystemExit("Download completed but expected train.csv/train_images were not found.")
    print(f"APTOS dataset ready: {output}")

if __name__ == "__main__":
    main()


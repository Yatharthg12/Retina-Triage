from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.config import load_config
from src.data.adapters import get_adapter
from src.data.splitting import stratified_split

def _inspect(row):
    path = Path(row.image_path)
    if not path.exists():
        return None, "missing image"
    try:
        raw = path.read_bytes()
        with Image.open(path) as image:
            image.verify()
        return {
            **row._asdict(), "sha256": hashlib.sha256(raw).hexdigest(),
            "file_size": len(raw)
        }, None
    except (OSError, UnidentifiedImageError):
        return None, "unreadable image"

def prepare(dataset: str, input_path: Path, output: Path, config: dict) -> dict:
    frame = get_adapter(dataset).load(input_path)
    accepted, rejected = [], []
    for row in frame.itertuples(index=False):
        item, reason = _inspect(row)
        (accepted if item else rejected).append(item or {**row._asdict(), "reason": reason})
    valid = pd.DataFrame(accepted)
    if valid.empty:
        raise ValueError("No valid images were found.")
    if not valid["grade"].isin(range(5)).all():
        raise ValueError("All grades must be integers from 0 to 4.")
    output.mkdir(parents=True, exist_ok=True)
    splits = stratified_split(valid, config["random_seed"], config["dataset"]["train_fraction"],
                              config["dataset"]["validation_fraction"])
    for name, split in splits.items():
        split.to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(rejected).to_csv(output / "rejected.csv", index=False)
    report = {
        "dataset": dataset, "accepted": len(valid), "rejected": len(rejected),
        "duplicates_removed": int(valid["sha256"].duplicated().sum()),
        "class_distribution": {str(k): int(v) for k, v in valid["grade"].value_counts().sort_index().items()},
        "split_counts": {k: len(v) for k, v in splits.items()},
        "random_seed": config["random_seed"],
    }
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def main():
    parser = argparse.ArgumentParser(description="Validate and split a retinal dataset.")
    parser.add_argument("--dataset", default="aptos")
    parser.add_argument("--input", default="data/raw/aptos")
    parser.add_argument("--output", default="data/splits")
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(prepare(args.dataset, Path(args.input), Path(args.output), config), indent=2))

if __name__ == "__main__":
    main()


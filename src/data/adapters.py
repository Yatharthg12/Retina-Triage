from __future__ import annotations

from pathlib import Path
import pandas as pd

class DatasetAdapter:
    def load(self, root: Path) -> pd.DataFrame:
        raise NotImplementedError

class AptosAdapter(DatasetAdapter):
    def load(self, root: Path) -> pd.DataFrame:
        csv_path = root / "train.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"APTOS metadata not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        required = {"id_code", "diagnosis"}
        if not required.issubset(frame.columns):
            raise ValueError(f"APTOS metadata requires columns: {sorted(required)}")
        frame = frame.rename(columns={"id_code": "image_id", "diagnosis": "grade"})
        frame["image_path"] = frame["image_id"].map(lambda x: str(root / "train_images" / f"{x}.png"))
        frame["source"] = "aptos2019"
        return frame

class GenericCSVAdapter(DatasetAdapter):
    def __init__(self, csv_name="metadata.csv", image_column="image", label_column="grade"):
        self.csv_name, self.image_column, self.label_column = csv_name, image_column, label_column
    def load(self, root: Path) -> pd.DataFrame:
        frame = pd.read_csv(root / self.csv_name)
        frame = frame.rename(columns={self.image_column: "image_id", self.label_column: "grade"})
        frame["image_path"] = frame["image_id"].map(lambda x: str(root / str(x)))
        frame["source"] = "generic"
        return frame

def get_adapter(name: str) -> DatasetAdapter:
    if name.lower() == "aptos":
        return AptosAdapter()
    if name.lower() == "generic":
        return GenericCSVAdapter()
    raise ValueError(f"Unsupported dataset adapter: {name}")


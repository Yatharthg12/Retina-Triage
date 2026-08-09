from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split(frame: pd.DataFrame, seed: int = 42,
                     train_fraction: float = .70, validation_fraction: float = .15):
    if frame["sha256"].duplicated().any():
        frame = frame.drop_duplicates("sha256", keep="first").copy()
    temp_fraction = 1.0 - train_fraction
    stratify = frame["grade"] if frame["grade"].value_counts().min() >= 2 else None
    train, temp = train_test_split(frame, test_size=temp_fraction, random_state=seed, stratify=stratify)
    val_share = validation_fraction / temp_fraction
    temp_stratify = temp["grade"] if temp["grade"].value_counts().min() >= 2 else None
    validation, test = train_test_split(temp, train_size=val_share, random_state=seed, stratify=temp_stratify)
    split_frames = {"train": train, "validation": validation, "test": test}
    hashes = [set(f["sha256"]) for f in split_frames.values()]
    if any(hashes[i] & hashes[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Duplicate image hashes crossed dataset splits.")
    return split_frames


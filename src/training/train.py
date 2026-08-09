from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.constants import CLASS_NAMES
from src.modeling.architecture import build_model
from src.modeling.calibration import fit_temperature, temperature_scale, threshold_for_target_sensitivity

def set_deterministic(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)

def make_dataset(frame, image_size, batch_size, training=False, seed=42):
    import tensorflow as tf
    paths = frame["image_path"].astype(str).values
    labels = tf.keras.utils.to_categorical(frame["grade"].values, 5)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def load(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize_with_pad(image, image_size, image_size)
        return image, label
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        augmenter = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(.06),
            tf.keras.layers.RandomZoom(.08),
            tf.keras.layers.RandomTranslation(.06, .06),
            tf.keras.layers.RandomContrast(.10),
        ])
        ds = ds.shuffle(len(frame), seed=seed).map(lambda x, y: (augmenter(x, training=True), y),
                                                   num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train(config):
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    seed = config["random_seed"]; set_deterministic(seed)
    split_dir = Path(config["paths"]["splits"])
    train_csv, val_csv = split_dir / "train.csv", split_dir / "validation.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Prepared train and validation manifests are required. Run the data preparation command first.")
    train_frame, val_frame = pd.read_csv(train_csv), pd.read_csv(val_csv)
    batch = config["stage_one"]["batch_size"]
    train_ds = make_dataset(train_frame, config["image_size"], batch, True, seed)
    val_ds = make_dataset(val_frame, config["image_size"], batch)
    model = build_model(config["image_size"])
    weights = compute_class_weight("balanced", classes=np.arange(5), y=train_frame["grade"])
    class_weight = {i: float(v) for i, v in enumerate(weights)}
    artifact = Path(config["paths"]["model"]); artifact.parent.mkdir(parents=True, exist_ok=True)
    training_dir = Path(config["paths"]["evaluation"]).parent / "training"; training_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(artifact, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["stage_one"]["patience"], restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=.3),
        tf.keras.callbacks.CSVLogger(training_dir / "stage_one.csv"),
        tf.keras.callbacks.TensorBoard(training_dir / "tensorboard"),
    ]
    model.compile(tf.keras.optimizers.Adam(config["stage_one"]["learning_rate"]),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=config["loss"]["label_smoothing"]),
                  metrics=["accuracy"])
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=config["stage_one"]["epochs"],
                         class_weight=class_weight, callbacks=callbacks)
    backbone = next(layer for layer in model.layers if layer.name.startswith("efficientnet"))
    backbone.trainable = True
    for layer in backbone.layers[:-config["stage_two"]["unfreeze_layers"]]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(tf.keras.optimizers.Adam(config["stage_two"]["learning_rate"]),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=config["loss"]["label_smoothing"]),
                  metrics=["accuracy"])
    callbacks[0] = tf.keras.callbacks.ModelCheckpoint(artifact, monitor="val_loss", save_best_only=True)
    callbacks[1] = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["stage_two"]["patience"], restore_best_weights=True)
    callbacks[3] = tf.keras.callbacks.CSVLogger(training_dir / "stage_two.csv")
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=config["stage_two"]["epochs"],
                         class_weight=class_weight, callbacks=callbacks)
    best_model = tf.keras.models.load_model(artifact, compile=False)
    validation_probabilities = best_model.predict(val_ds, verbose=0)
    calibration = fit_temperature(validation_probabilities, val_frame["grade"].values)
    calibrated = temperature_scale(validation_probabilities, calibration["temperature"])
    calibration["referable"] = threshold_for_target_sensitivity(
        val_frame["grade"].values >= 2, calibrated[:, 2:].sum(axis=1),
        target=config["calibration"]["target_referable_sensitivity"]
    )
    calibration["high_risk"] = threshold_for_target_sensitivity(
        val_frame["grade"].values >= 3, calibrated[:, 3:].sum(axis=1),
        target=config["calibration"]["target_high_risk_sensitivity"]
    )
    calibration["status"] = "validated"
    calibration["method"] = "temperature_scaling"
    calibration["selected_on"] = "validation"
    Path(config["paths"]["calibration"]).write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    version = datetime.now(timezone.utc).strftime("aptos-%Y%m%d-%H%M%S")
    metadata = {
        "model_version": version, "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "class_names": CLASS_NAMES, "image_size": config["image_size"], "random_seed": seed,
        "preprocessing": config["preprocessing"], "triage_thresholds": config["triage"],
        "train_records": len(train_frame), "validation_records": len(val_frame),
        "software": {"tensorflow": tf.__version__, "numpy": np.__version__},
    }
    Path(config["paths"]["model_metadata"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (training_dir / "history.json").write_text(json.dumps({
        "stage_one": history1.history, "stage_two": history2.history
    }, default=float, indent=2), encoding="utf-8")
    return metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    try:
        print(json.dumps(train(load_config(args.config)), indent=2))
    except tf_errors() as exc:
        if "OOM" in type(exc).__name__.upper() or "RESOURCEEXHAUSTED" in type(exc).__name__.upper():
            raise SystemExit("TensorFlow ran out of memory. Reduce stage_one.batch_size and retry.") from exc
        raise

def tf_errors():
    try:
        import tensorflow as tf
        return (tf.errors.ResourceExhaustedError,)
    except ImportError:
        return (RuntimeError,)

if __name__ == "__main__":
    main()

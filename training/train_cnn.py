"""
training/train_cnn.py

Train the ResNet50V2 facial emotion classifier for MoodSyncAI.
Mirrors the full training pipeline from 1_Face-emotion-detection.ipynb.

Usage
-----
    python -m training.train_cnn                         # default config
    python -m training.train_cnn --config training/configs/cnn_config.yaml
    python -m training.train_cnn --architecture custom_cnn --epochs 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ── Lazy TF import so the module can be inspected without a GPU ───────────────
def _import_tf():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import ResNet50V2
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.layers import (
        BatchNormalization, Conv2D, Dense, Dropout,
        Flatten, GlobalAveragePooling2D, MaxPooling2D,
    )
    from tensorflow.keras.models import Sequential, Model
    return tf, layers, models, ResNet50V2, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Sequential, Model, BatchNormalization, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPooling2D


# ── Config loading ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "cnn_config.yaml"

def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model builders ─────────────────────────────────────────────────────────────

def build_resnet50v2(cfg: dict):
    """
    Build the ResNet50V2 transfer-learning model used in the notebook.

    Architecture (matches notebook Section: ResNet50V2):
        ResNet50V2 (ImageNet, frozen) → Dropout(0.25) → BN →
        Flatten → Dense(64) → BN → Dropout(0.25) → Dense(7, softmax)

    Final test accuracy achieved in notebook: 66.88 %
    """
    tf, layers, models, ResNet50V2, *_ = _import_tf()

    img_shape   = cfg["model"]["img_shape"]
    num_classes = cfg["model"]["num_classes"]
    dropout     = cfg["model"]["dropout_rate"]
    dense_units = cfg["model"]["dense_units"]

    base = ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_shape, img_shape, 3),
    )
    base.trainable = False

    x = base.output
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    for units in dense_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=base.input, outputs=output)


def build_custom_cnn(cfg: dict):
    """
    Build the custom 3-block CNN used in the notebook (Section: Building CNN Model).

    Architecture:
        3× [Conv2D(32→64), Conv2D(64→128), MaxPool2D, Dropout(0.25)]
        Flatten → Dense(1024→512→256→128→64→32) with BN+Dropout → Dense(7)
    """
    tf, layers, models, _, _, _, _, Sequential, _, BatchNormalization, Conv2D, Dense, Dropout, Flatten, _, MaxPooling2D = _import_tf()

    img_shape   = cfg["model"]["img_shape"]
    num_classes = cfg["model"]["num_classes"]
    dropout     = cfg["model"]["dropout_rate"]

    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_shape, img_shape, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(dropout))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(dropout))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(dropout))

    # Classifier head
    model.add(Flatten())
    for units in [1024, 512, 256, 128, 64, 32]:
        model.add(Dense(units, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation="softmax"))

    return model


def build_model(cfg: dict):
    arch = cfg["model"]["architecture"]
    if arch == "resnet50v2":
        return build_resnet50v2(cfg)
    elif arch == "custom_cnn":
        return build_custom_cnn(cfg)
    else:
        raise ValueError(f"Unknown architecture: {arch!r}. Choose 'resnet50v2' or 'custom_cnn'.")


# ── Callbacks ─────────────────────────────────────────────────────────────────

def build_callbacks(cfg: dict):
    """Return [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau] matching notebook."""
    tf, _, _, _, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, *_ = _import_tf()
    import tensorflow as tf

    cb_cfg = cfg["callbacks"]
    checkpoint_path = cb_cfg["checkpoint"]["path"]
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor=cb_cfg["checkpoint"]["monitor"],
        save_best_only=cb_cfg["checkpoint"]["save_best_only"],
        verbose=cb_cfg["checkpoint"]["verbose"],
    )
    early_stopping = EarlyStopping(
        monitor=cb_cfg["early_stopping"]["monitor"],
        patience=cb_cfg["early_stopping"]["patience"],
        restore_best_weights=cb_cfg["early_stopping"]["restore_best_weights"],
        verbose=cb_cfg["early_stopping"]["verbose"],
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=cb_cfg["reduce_lr_on_plateau"]["monitor"],
        factor=cb_cfg["reduce_lr_on_plateau"]["factor"],
        patience=cb_cfg["reduce_lr_on_plateau"]["patience"],
        verbose=cb_cfg["reduce_lr_on_plateau"]["verbose"],
    )
    return [checkpoint, early_stopping, reduce_lr]


# ── Training entry point ───────────────────────────────────────────────────────

def train(cfg: dict):
    """
    Full training loop. Mirrors the notebook training cells exactly.

    Steps
    -----
    1. Build augmented train generator and test generator
    2. Build model (ResNet50V2 or custom CNN)
    3. Compile with Adam + categorical_crossentropy
    4. Fit with 3 callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    5. Evaluate on test set and print results
    6. Save best model to output path
    """
    from data.preprocessing.image_preprocessing import get_train_generator, get_test_generator

    tf, *_ = _import_tf()

    # ── Generators ────────────────────────────────────────────────────────────
    train_gen = get_train_generator(
        train_dir=cfg["data"]["train_dir"],
        img_shape=cfg["model"]["img_shape"],
        batch_size=cfg["data"]["batch_size"],
    )
    test_gen = get_test_generator(
        test_dir=cfg["data"]["test_dir"],
        img_shape=cfg["model"]["img_shape"],
        batch_size=cfg["data"]["batch_size"],
    )

    steps_per_epoch  = train_gen.n // train_gen.batch_size
    validation_steps = test_gen.n  // test_gen.batch_size

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg)
    model.summary()

    model.compile(
        optimizer=cfg["training"]["optimizer"],
        loss=cfg["training"]["loss"],
        metrics=cfg["training"]["metrics"],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["data"]["batch_size"],
        callbacks=build_callbacks(cfg),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    scores = model.evaluate(test_gen)
    print(f"Test Loss:     {scores[0]:.5f}")
    print(f"Test Accuracy: {scores[1] * 100:.2f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = cfg["output"]["save_path"]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"Model saved → {save_path}")

    return model, history


# ── Plotting utility ──────────────────────────────────────────────────────────

def plot_training_curves(history, save_path: str | Path | None = None):
    """
    Plot accuracy and loss curves — mirrors the notebook plot_curves() helper.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Curves saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train MoodSyncAI CNN emotion classifier")
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="Path to YAML config (default: training/configs/cnn_config.yaml)")
    p.add_argument("--architecture", choices=["resnet50v2", "custom_cnn"],
                   help="Override config architecture")
    p.add_argument("--epochs", type=int,
                   help="Override config epochs")
    p.add_argument("--batch_size", type=int,
                   help="Override config batch_size")
    p.add_argument("--train_dir", help="Override data.train_dir")
    p.add_argument("--test_dir",  help="Override data.test_dir")
    p.add_argument("--save_path", help="Override output.save_path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = load_config(args.config)

    # Apply CLI overrides
    if args.architecture: cfg["model"]["architecture"]  = args.architecture
    if args.epochs:        cfg["training"]["epochs"]     = args.epochs
    if args.batch_size:    cfg["data"]["batch_size"]     = args.batch_size
    if args.train_dir:     cfg["data"]["train_dir"]      = args.train_dir
    if args.test_dir:      cfg["data"]["test_dir"]       = args.test_dir
    if args.save_path:     cfg["output"]["save_path"]    = args.save_path

    model, history = train(cfg)
    plot_training_curves(history, save_path="saved_models/cnn_training_curves.png")

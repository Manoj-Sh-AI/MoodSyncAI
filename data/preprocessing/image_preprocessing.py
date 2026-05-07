"""
data/preprocessing/image_preprocessing.py

FER2013 image preprocessing pipeline for MoodSyncAI.
Mirrors the exact ImageDataGenerator configuration used in
1_Face-emotion-detection.ipynb (Section: Data Preprocessing).

Dataset layout expected:
    data/raw/
        train/
            angry/   disgust/   fear/   happy/   neutral/   sad/   surprise/
        test/
            angry/   disgust/   fear/   happy/   neutral/   sad/   surprise/

Usage
-----
    from data.preprocessing.image_preprocessing import (
        get_train_generator,
        get_test_generator,
        get_class_distribution,
    )
    train_gen = get_train_generator()
    test_gen  = get_test_generator()
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Constants (match notebook) ───────────────────────────────────────────────
IMG_SHAPE  : int = 224          # ResNet50V2 input size
BATCH_SIZE : int = 64
NUM_CLASSES: int = 7
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

RAW_DIR   = Path("data/raw")
TRAIN_DIR = RAW_DIR / "train"
TEST_DIR  = RAW_DIR / "test"


# ── Generators ────────────────────────────────────────────────────────────────
def get_train_generator(
    train_dir: str | Path = TRAIN_DIR,
    img_shape: int = IMG_SHAPE,
    batch_size: int = BATCH_SIZE,
) -> "ImageDataGenerator.iterator":
    """
    Build the augmented training data generator.

    Augmentations applied (matching notebook):
        - Random rotation ±10°
        - Zoom ±20%
        - Width / height shift ±10%
        - Horizontal flip
        - Nearest-neighbour fill mode

    Returns
    -------
    Keras DirectoryIterator (shuffled, categorical labels, RGB, 224×224)
    """
    preprocessor = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    return preprocessor.flow_from_directory(
        str(train_dir),
        class_mode="categorical",
        target_size=(img_shape, img_shape),
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size,
    )


def get_test_generator(
    test_dir: str | Path = TEST_DIR,
    img_shape: int = IMG_SHAPE,
    batch_size: int = BATCH_SIZE,
) -> "ImageDataGenerator.iterator":
    """
    Build the test / validation data generator (no augmentation).

    Returns
    -------
    Keras DirectoryIterator (not shuffled, categorical labels, RGB, 224×224)
    """
    preprocessor = ImageDataGenerator(rescale=1.0 / 255.0)
    return preprocessor.flow_from_directory(
        str(test_dir),
        class_mode="categorical",
        target_size=(img_shape, img_shape),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )


# ── Utilities ─────────────────────────────────────────────────────────────────
def get_class_distribution(
    train_dir: str | Path = TRAIN_DIR,
    test_dir:  str | Path = TEST_DIR,
) -> pd.DataFrame:
    """
    Count images per emotion class in both splits.

    Returns a DataFrame with columns [Train, Test] and class names as index.
    Mirrors the countperclass() analysis in the notebook.
    """
    def _count(directory: Path) -> dict[str, int]:
        return {
            cls: len(os.listdir(directory / cls))
            for cls in sorted(os.listdir(directory))
            if (directory / cls).is_dir()
        }

    train_counts = pd.DataFrame.from_dict(_count(Path(train_dir)), orient="index", columns=["Train"])
    test_counts  = pd.DataFrame.from_dict(_count(Path(test_dir)),  orient="index", columns=["Test"])
    return pd.concat([train_counts, test_counts], axis=1)


def preprocess_single_image(
    filepath: str | Path,
    img_shape: int = IMG_SHAPE,
) -> np.ndarray:
    """
    Load a single image file and prepare it for model inference.

    Steps
    -----
    1. Read and convert to RGB
    2. Resize to (img_shape, img_shape)
    3. Normalise pixel values to [0, 1]

    Does NOT perform face detection — use models/visual/face_detector.py for that.

    Returns
    -------
    np.ndarray  float32  shape (img_shape, img_shape, 3)
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    img   = load_img(str(filepath), target_size=(img_shape, img_shape), color_mode="rgb")
    arr   = img_to_array(img) / 255.0
    return arr.astype(np.float32)


def batch_preprocess_images(
    filepaths: list[str | Path],
    img_shape: int = IMG_SHAPE,
) -> np.ndarray:
    """
    Preprocess a list of image files into a batched numpy array.

    Returns
    -------
    np.ndarray  float32  shape (N, img_shape, img_shape, 3)
    """
    return np.stack([preprocess_single_image(p, img_shape) for p in filepaths], axis=0)

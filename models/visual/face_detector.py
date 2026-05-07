"""
models/visual/face_detector.py
Haar-cascade face detection + preprocessing pipeline.
Mirrors the loadandprepimage() function from 1_Face-emotion-detection.ipynb.
"""
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

CASCADE_PATH = Path("haarcascade_frontalface_default.xml")
CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "data/haarcascades/haarcascade_frontalface_default.xml"
)
IMG_SHAPE = 224


def _ensure_cascade():
    """Download Haar cascade if not present."""
    if not CASCADE_PATH.exists():
        print(f"Downloading Haar cascade to {CASCADE_PATH} ...")
        urllib.request.urlretrieve(CASCADE_URL, CASCADE_PATH)
        print("Downloaded.")


_ensure_cascade()
_face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))


def preprocess_face_image(filename: str, img_shape: int = IMG_SHAPE) -> np.ndarray:
    """
    Load an image file, detect the first face, crop & resize to
    (img_shape, img_shape, 3) and normalise to [0, 1].

    Parameters
    ----------
    filename  : path to a JPEG/PNG face image
    img_shape : target spatial size (default 224)

    Returns
    -------
    np.ndarray  float32  shape (img_shape, img_shape, 3)  values in [0, 1]

    Raises
    ------
    FileNotFoundError  if the image cannot be read
    ValueError         if the cropped region is empty
    """
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f"Image not found: {filename}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        img = img[y : y + h, x : x + w]

    if img.size == 0:
        raise ValueError(f"Empty image after face crop: {filename}")

    img = cv2.resize(img, (img_shape, img_shape))
    img = img / 255.0
    return img.astype(np.float32)

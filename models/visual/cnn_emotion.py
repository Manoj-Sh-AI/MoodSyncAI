"""
models/visual/cnn_emotion.py
ResNet50V2-based facial emotion classifier for MoodSyncAI.
Architecture mirrors 1_Face-emotion-detection.ipynb exactly.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, BatchNormalization, Flatten, Dense
from pathlib import Path

EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
IMG_SHAPE = 224
MODEL_PATH = Path("saved_models/cnn_emotion.pt")          # .h5 also accepted
CHECKPOINT_PATH = Path("ResNet50V2ModelCheckpoint.keras")


def build_resnet50v2(num_classes: int = 7, img_shape: int = IMG_SHAPE) -> tf.keras.Model:
    """Rebuild the exact ResNet50V2 architecture used in training."""
    base = tf.keras.applications.ResNet50V2(
        input_shape=(img_shape, img_shape, 3),
        include_top=False,
        weights=None,
    )
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    model = Sequential([
        base,
        Dropout(0.25),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class CNNEmotionModel:
    """
    Wraps ResNet50V2 for inference.

    Usage
    -----
    model = CNNEmotionModel()
    result = model.predict(image_path="path/to/face.jpg")
    # result = {"emotion": "Happy", "emotion_id": 3, "confidence": 0.99,
    #           "polarity": "positive", "probs": {...}}
    """

    VISUAL_TO_POLARITY = {
        "Happy":   "positive",
        "Surprise": "positive",
        "Neutral": "neutral",
        "Angry":   "negative",
        "Disgust": "negative",
        "Fear":    "negative",
        "Sad":     "negative",
    }

    def __init__(self, model_path: str = str(MODEL_PATH)):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}. "
                "Run 1_Face-emotion-detection.ipynb first and save the model."
            )
        try:
            self._model = tf.keras.models.load_model(str(path), compile=False)
        except OSError:
            # Fall back to weights-only checkpoint
            ckpt = str(CHECKPOINT_PATH)
            if not Path(ckpt).exists():
                raise
            self._model = build_resnet50v2()
            self._model.load_weights(ckpt)

    def predict(self, image: np.ndarray) -> dict:
        """
        Parameters
        ----------
        image : np.ndarray  shape (224, 224, 3) float32 in [0, 1]
                Use face_detector.preprocess_face_image() to prepare it.

        Returns
        -------
        dict with keys: emotion, emotion_id, confidence, polarity, probs
        """
        img_batch = np.expand_dims(image, axis=0)           # (1, 224, 224, 3)
        preds = self._model.predict(img_batch, verbose=0)   # (1, 7)
        scores = preds[0]                                   # (7,)
        top_id = int(np.argmax(scores))
        emotion = EMOTION_CLASSES[top_id]

        return {
            "emotion":    emotion,
            "emotion_id": top_id,
            "confidence": round(float(scores[top_id]), 4),
            "polarity":   self.VISUAL_TO_POLARITY.get(emotion, "neutral"),
            "probs":      {EMOTION_CLASSES[i]: round(float(scores[i]), 4)
                           for i in range(len(EMOTION_CLASSES))},
        }

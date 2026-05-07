"""
models/audio/audio_model.py
============================
AudioEmotionNet — 4-layer 1-D CNN + 2-layer BiLSTM speech emotion classifier.

Architecture (identical to notebook 5 — 5_audio_emotion_recognition.ipynb):

    Input      (B, FEATURE_DIM, T)   e.g. (B, 198, 94)
        │
    Conv1d ×4  with BatchNorm, ReLU, MaxPool(×2 twice), Dropout
        │
        (B, 512, T//4)
        │
    BiLSTM ×2  hidden=256 per direction  →  last time-step
        │
        (B, 512)
        │
    Linear 512→256 → ReLU → Dropout(0.4) → Linear 256→NUM_CLASSES
        │
    Output     (B, NUM_CLASSES)  raw logits  →  softmax for probs

Trained on RAVDESS + CREMA-D (7 emotion classes).
Best validation accuracy: 83.49 %  (saved at epoch 20).

Quick start
-----------
>>> from moodsyncai.models.audio.audio_model import predict_audio
>>> import soundfile as sf
>>> audio, sr = sf.read("clip.wav", dtype="float32")
>>> result = predict_audio(audio, sr)
>>> print(result["emotion"], f"{result['confidence']:.1%}")
happy 87.3%
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from moodsyncai.config import (
    AUDIO_MODEL_PATH,
    LABEL_ENC_PATH,
    DEVICE,
    FEATURE_DIM,
    NUM_CLASSES,
    AUDIO_TO_POLARITY,
)
from moodsyncai.models.audio.audio_features import extract_features


# ── Model definition ──────────────────────────────────────────────────────────

class AudioEmotionNet(nn.Module):
    """1-D CNN → BiLSTM classifier for 7-class speech emotion recognition.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.  Default: ``FEATURE_DIM`` (198).
    num_classes : int
        Number of output emotion classes.  Default: ``NUM_CLASSES`` (7).
    """

    def __init__(
        self,
        in_channels: int = FEATURE_DIM,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            # ── block 1 ───────────────────────────────────────────────────────
            nn.Conv1d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            # ── block 2  (MaxPool: T → T//2) ──────────────────────────────────
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),

            # ── block 3 ───────────────────────────────────────────────────────
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),

            # ── block 4  (MaxPool: T//2 → T//4) ──────────────────────────────
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,        # 256 forward + 256 backward = 512 total
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, in_channels, T)`` — e.g. ``(B, 198, 94)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_classes)`` — raw logits.
        """
        x = self.cnn(x)           # (B, 512, T//4)
        x = x.permute(0, 2, 1)   # (B, T//4, 512)  — LSTM expects (B, seq, feat)
        out, _ = self.lstm(x)     # (B, T//4, 512)
        return self.classifier(out[:, -1])   # last time-step → (B, num_classes)


# ── Loading helpers ───────────────────────────────────────────────────────────

def load_model(
    model_path: Path = AUDIO_MODEL_PATH,
    device: str = DEVICE,
) -> AudioEmotionNet:
    """Instantiate ``AudioEmotionNet`` and load ``best_audio_model.pt`` weights.

    Parameters
    ----------
    model_path : Path
        Path to the ``.pt`` checkpoint.  Default: ``AUDIO_MODEL_PATH``
        from ``config.py``.
    device : str
        Target device, e.g. ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    AudioEmotionNet
        Model set to ``eval()`` mode on *device*.
    """
    net = AudioEmotionNet(
        in_channels=FEATURE_DIM,
        num_classes=NUM_CLASSES,
    ).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net


def load_label_encoder(path: Path = LABEL_ENC_PATH):
    """Load the sklearn ``LabelEncoder`` saved during training.

    Returns
    -------
    sklearn.preprocessing.LabelEncoder
        Maps integer class indices ↔ emotion strings in the exact order
        the model was trained on.  Never hard-code the class list — always
        use this encoder.
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_audio(
    audio: np.ndarray,
    sr: int,
    model: AudioEmotionNet | None = None,
    label_encoder=None,
    device: str = DEVICE,
) -> dict:
    """Run 7-class emotion inference on an in-memory audio array.

    Lazy-loads model weights and label encoder from disk on the first call
    if they are not passed in.  Pass pre-loaded objects in production to
    avoid repeated disk I/O.

    Parameters
    ----------
    audio : np.ndarray
        Raw audio samples (mono or stereo, float32 / int16 / etc.).
    sr : int
        Sample rate of *audio*.
    model : AudioEmotionNet, optional
        Pre-loaded model.  Loaded from ``AUDIO_MODEL_PATH`` if ``None``.
    label_encoder : sklearn LabelEncoder, optional
        Pre-loaded encoder.  Loaded from ``LABEL_ENC_PATH`` if ``None``.
    device : str
        Inference device.

    Returns
    -------
    dict
        +--------------+-------+--------------------------------------------+
        | Key          | Type  | Description                                |
        +==============+=======+============================================+
        | emotion      | str   | Top predicted emotion label                |
        | emotion_id   | int   | Class index                                |
        | confidence   | float | Softmax probability of the top class       |
        | polarity     | str   | ``"positive"`` / ``"neutral"`` /           |
        |              |       | ``"negative"``                             |
        | probs        | dict  | ``{emotion: prob}`` for all 7 classes      |
        +--------------+-------+--------------------------------------------+
    """
    if model is None:
        model = load_model(device=device)
    if label_encoder is None:
        label_encoder = load_label_encoder()

    feat = extract_features(audio, sr)                   # (198, T)
    inp  = torch.tensor(feat).unsqueeze(0).to(device)   # (1, 198, T)

    model.eval()
    with torch.no_grad():
        probs = (
            torch.softmax(model(inp), dim=1)
            .squeeze()
            .cpu()
            .numpy()
        )

    top_id  = int(probs.argmax())
    emotion = label_encoder.classes_[top_id]

    return {
        "emotion":    emotion,
        "emotion_id": top_id,
        "confidence": round(float(probs[top_id]), 4),
        "polarity":   AUDIO_TO_POLARITY[emotion],
        "probs": {
            label_encoder.classes_[i]: round(float(probs[i]), 4)
            for i in range(NUM_CLASSES)
        },
    }

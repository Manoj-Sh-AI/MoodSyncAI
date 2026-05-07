"""
models/visual/video_predictor.py
High-level inference API for video-based emotion recognition.
Mirrors the predict_video / predict_video_from_frames functions in
notebooks/7_video_emotion_recognition_ravdess.ipynb and
notebooks/8_video_text_fusion.ipynb.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from models.visual.video_features import (
    build_frame_transform,
    decode_uniform_frames,
    frames_to_tensor,
)
from models.visual.video_model import VideoEmotionModel

# Polarity bridge: maps 7-class emotion labels → coarse sentiment polarity.
# This mirrors the VIDEO_TO_POLARITY constant used in notebooks 7 & 8.
VIDEO_TO_POLARITY: Dict[str, str] = {
    "happy":    "positive",
    "surprise": "positive",
    "neutral":  "neutral",
    "angry":    "negative",
    "disgust":  "negative",
    "fear":     "negative",
    "sad":      "negative",
}


class VideoEmotionPredictor:
    """Wraps a trained VideoEmotionModel for convenient inference.

    Example usage::

        predictor = VideoEmotionPredictor.from_saved(
            checkpoint_path="saved_models/video_emotion/best_video_model.pt",
            meta_path="saved_models/video_emotion/video_model_meta.json",
            label_encoder_path="saved_models/video_emotion/video_label_encoder.pkl",
        )
        result = predictor.predict_from_path("path/to/clip.mp4")
        print(result["emotion"], result["confidence"])
    """

    def __init__(
        self,
        model: VideoEmotionModel,
        label_encoder,
        meta: dict,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device).eval()
        self.le = label_encoder
        self.meta = meta

        self.num_frames: int = meta.get("num_frames", 16)
        self.frame_size: int = meta.get("frame_size", 112)
        self.emotion_classes: List[str] = list(label_encoder.classes_)
        self.num_classes: int = len(self.emotion_classes)
        self.transform = build_frame_transform(self.frame_size)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_saved(
        cls,
        checkpoint_path: str,
        meta_path: str,
        label_encoder_path: str,
        device: Optional[str] = None,
    ) -> "VideoEmotionPredictor":
        """Instantiate from saved model artefacts (used at serve time)."""
        from models.visual.video_model import load_video_model

        model, le, meta = load_video_model(
            checkpoint_path=checkpoint_path,
            meta_path=meta_path,
            label_encoder_path=label_encoder_path,
            device=device,
        )
        return cls(model=model, label_encoder=le, meta=meta, device=device)

    # ------------------------------------------------------------------
    # Core inference helpers
    # ------------------------------------------------------------------

    def _run_inference(self, clip: torch.Tensor) -> dict:
        """Run forward pass and return a standardised result dict.

        Args:
            clip: Tensor of shape (1, 3, T, H, W) already on self.device.

        Returns:
            dict with keys: emotion, emotion_id, confidence, polarity, probs.
        """
        with torch.no_grad():
            probs = (
                torch.softmax(self.model(clip), dim=1)
                .squeeze()
                .cpu()
                .numpy()
            )
        top_id = int(probs.argmax())
        emotion = self.le.classes_[top_id]
        return {
            "emotion":    emotion,
            "emotion_id": top_id,
            "confidence": round(float(probs[top_id]), 4),
            "polarity":   VIDEO_TO_POLARITY.get(emotion, "neutral"),
            "probs":      {
                self.le.classes_[i]: round(float(probs[i]), 4)
                for i in range(self.num_classes)
            },
        }

    # ------------------------------------------------------------------
    # Public API — matching notebook function signatures
    # ------------------------------------------------------------------

    def predict_from_path(self, video_path: str) -> dict:
        """Run 7-class emotion inference on a video file (.mp4, .flv, …).

        Args:
            video_path: Path to the video file on disk.

        Returns:
            Standardised result dict (see _run_inference).
        """
        clip = (
            frames_to_tensor(
                decode_uniform_frames(video_path, self.num_frames),
                transform=self.transform,
                num_frames=self.num_frames,
                frame_size=self.frame_size,
            )
            .to(self.device)
        )
        return self._run_inference(clip)

    def predict_from_frames(self, frames_rgb: List[np.ndarray]) -> dict:
        """Run inference on a pre-decoded list of RGB frames.

        Useful for in-memory / streaming scenarios where repeated disk I/O
        should be avoided.

        Args:
            frames_rgb: List of (H, W, 3) uint8 NumPy arrays.

        Returns:
            Standardised result dict (see _run_inference).
        """
        clip = (
            frames_to_tensor(
                frames_rgb,
                transform=self.transform,
                num_frames=self.num_frames,
                frame_size=self.frame_size,
            )
            .to(self.device)
        )
        return self._run_inference(clip)

"""
models/fusion/video_text_fusion.py
Video–text fusion layer and mismatch detection for MoodSyncAI.
Mirrors the logic in notebooks/8_video_text_fusion.ipynb.

The module provides:
    • VideoTextFusion      – stateless helper that combines video + text results
    • detect_mismatch      – standalone polarity-based mismatch detector
    • run_fusion           – end-to-end pipeline (video path + text → fusion result)
    • run_fusion_from_frames – same pipeline but accepts pre-decoded frames

Design decisions (from the notebook):
    Severity  | Condition
    ----------|-------------------------------------------------------
    MATCH     | video polarity == text polarity
    SOFT_MISMATCH | polarities differ, but at least one confidence < threshold
    HARD_MISMATCH | polarities differ AND both confidences >= threshold
"""

import time
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONF_THRESHOLD: float = 0.50

SEVERITY_COLOURS: Dict[str, str] = {
    "MATCH":          "#4CAF50",
    "SOFT_MISMATCH":  "#FFA726",
    "HARD_MISMATCH":  "#EF5350",
}

SEVERITY_EMOJI: Dict[str, str] = {
    "MATCH":          "✅",
    "SOFT_MISMATCH":  "⚠️",
    "HARD_MISMATCH":  "❌",
}


# ---------------------------------------------------------------------------
# Core mismatch detector
# ---------------------------------------------------------------------------

def detect_mismatch(
    video_result: dict,
    text_result: dict,
    conf_threshold: float = CONF_THRESHOLD,
) -> dict:
    """Classify the agreement between video emotion and text sentiment.

    Args:
        video_result: Output of VideoEmotionPredictor.predict_from_path /
                      predict_from_frames.  Must contain 'polarity', 'emotion',
                      and 'confidence' keys.
        text_result:  Output of the text sentiment model.  Must contain
                      'label' and 'confidence' keys.
        conf_threshold: Minimum confidence to be considered "confident".

    Returns:
        dict with severity, explanation, badge_colour, emoji, and input
        metadata fields.
    """
    video_pol = video_result["polarity"]
    text_pol  = text_result["label"]      # "negative" | "neutral" | "positive"

    match      = video_pol == text_pol
    both_conf  = (
        video_result["confidence"] >= conf_threshold
        and text_result["confidence"] >= conf_threshold
    )

    if match:
        severity = "MATCH"
        expl = (
            f"Both agree: video shows {video_result['emotion']} "
            f"({video_pol}) and text is {text_pol}."
        )
    elif both_conf:
        severity = "HARD_MISMATCH"
        expl = (
            f"Conflict: video={video_result['emotion']} ({video_pol}, "
            f"{video_result['confidence']:.0%}) vs "
            f"text={text_pol} ({text_result['confidence']:.0%})."
        )
    else:
        severity = "SOFT_MISMATCH"
        expl = (
            f"Uncertain: video={video_result['emotion']} ({video_pol}, "
            f"{video_result['confidence']:.0%}) vs "
            f"text={text_pol} ({text_result['confidence']:.0%}). "
            f"Low confidence in at least one modality."
        )

    return {
        "severity":      severity,
        "video_emotion": video_result["emotion"],
        "video_polarity": video_pol,
        "text_polarity":  text_pol,
        "polarity_match": match,
        "both_confident": both_conf,
        "badge_colour":  SEVERITY_COLOURS[severity],
        "emoji":         SEVERITY_EMOJI[severity],
        "explanation":   expl,
    }


# ---------------------------------------------------------------------------
# High-level fusion pipelines
# ---------------------------------------------------------------------------

class VideoTextFusion:
    """Combines VideoEmotionPredictor and a text sentiment model into one API.

    Args:
        video_predictor: Instance of VideoEmotionPredictor.
        text_model:      Callable that accepts a string and returns a dict
                         with 'label' and 'confidence' keys (e.g. the
                         sentiment_model.SentimentPredictor.predict method or
                         any compatible function).
        conf_threshold:  Mismatch detection confidence threshold.

    Example::

        from models.visual.video_predictor import VideoEmotionPredictor
        from models.text.sentiment_model import SentimentPredictor

        vp = VideoEmotionPredictor.from_saved(...)
        tp = SentimentPredictor.from_saved(...)

        fusion = VideoTextFusion(
            video_predictor=vp,
            text_model=tp.predict,
        )
        result = fusion.run("path/to/video.mp4", "I feel great today!")
        print(result["fusion"]["severity"])
    """

    def __init__(
        self,
        video_predictor,
        text_model,
        conf_threshold: float = CONF_THRESHOLD,
    ):
        self.video_predictor = video_predictor
        self.text_model      = text_model
        self.conf_threshold  = conf_threshold

    def run(self, video_path: str, text: str) -> dict:
        """Full fusion pipeline: video file + transcript text.

        Args:
            video_path: Path to the .mp4 / .flv video clip.
            text:       Transcript or caption associated with the clip.

        Returns:
            dict with 'video', 'text', 'fusion', and 'latency_ms' keys.
        """
        t0 = time.perf_counter()
        video_result = self.video_predictor.predict_from_path(video_path)
        text_result  = self.text_model(text)
        fusion_result = detect_mismatch(
            video_result, text_result, self.conf_threshold
        )
        return {
            "video":      video_result,
            "text":       text_result,
            "fusion":     fusion_result,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def run_from_frames(
        self,
        frames_rgb: List[np.ndarray],
        text: str,
    ) -> dict:
        """Fusion pipeline using pre-decoded frames (avoids repeated disk I/O).

        Args:
            frames_rgb: List of (H, W, 3) uint8 NumPy arrays.
            text:       Transcript or caption text.

        Returns:
            Same structure as :meth:`run`.
        """
        t0 = time.perf_counter()
        video_result  = self.video_predictor.predict_from_frames(frames_rgb)
        text_result   = self.text_model(text)
        fusion_result = detect_mismatch(
            video_result, text_result, self.conf_threshold
        )
        return {
            "video":      video_result,
            "text":       text_result,
            "fusion":     fusion_result,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_fusion_result(result: dict) -> None:
    """Print a human-readable summary of a fusion result dict."""
    v = result["video"]
    t = result["text"]
    f = result["fusion"]

    print("=" * 60)
    print(f"{f['emoji']}  Fusion Result: {f['severity']}")
    print("=" * 60)
    print(f"  Video : {v['emotion']:<10}  conf={v['confidence']:.0%}  polarity={v['polarity']}")
    print(f"  Text  : {t['label']:<10}  conf={t['confidence']:.0%}")
    print(f"  Status: {f['explanation']}")
    print(f"  Latency: {result['latency_ms']} ms")
    print()

    print("  Video 7-class probabilities:")
    for emo, p in sorted(v["probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 20)
        marker = " ← top" if emo == v["emotion"] else ""
        print(f"    {emo:<10} {p * 100:5.1f}%  {bar}{marker}")

    print()
    print("  Text probabilities:")
    for lbl, p in sorted(t["probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 20)
        marker = " ← top" if lbl == t["label"] else ""
        print(f"    {lbl:<10} {p * 100:5.1f}%  {bar}{marker}")

# models/fusion/fusion_layer.py
# ================================
# Weighted-average fusion of modality confidence scores for MoodSyncAI.
# Produces a single fused polarity label + confidence.

# Two fusion pairs are supported:

#     1.  text ↔ image  (ResNet50V2 face emotion + RoBERTa text sentiment)
#         → fuse_modalities()           mirrors Section 8 of 3_fusion_analysis.ipynb

#     2.  text ↔ audio  (AudioEmotionNet speech emotion + RoBERTa text sentiment)
#         → fuse_audio_text_modalities() mirrors Section 8 of 6_fusion_audio_text.ipynb

# Both functions follow identical weighted-average logic — the only difference
# is the polarity map used to collapse the 7-class emotion distribution.

from __future__ import annotations

# ── Polarity maps ─────────────────────────────────────────────────────────────

VISUAL_TO_POLARITY: dict[str, str] = {
    "Happy": "positive",
    "Surprise": "positive",
    "Neutral": "neutral",
    "Angry": "negative",
    "Disgust": "negative",
    "Fear": "negative",
    "Sad": "negative",
}

AUDIO_TO_POLARITY: dict[str, str] = {
    "happy": "positive",
    "surprise": "positive",
    "neutral": "neutral",
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sad": "negative",
}

POLARITY_LABELS = ["negative", "neutral", "positive"]

# ── Default modality weights ──────────────────────────────────────────────────

DEFAULT_VISUAL_WEIGHT = 0.50
DEFAULT_TEXT_WEIGHT = 0.50
DEFAULT_AUDIO_WEIGHT = 0.50


# ── Helpers ───────────────────────────────────────────────────────────────────


def get_visual_polarity(emotion_label: str) -> str:
    """Map a ResNet50V2 emotion class to 3-class polarity."""
    return VISUAL_TO_POLARITY.get(emotion_label, "neutral")


def get_audio_polarity(emotion_label: str) -> str:
    """Map an AudioEmotionNet emotion class to 3-class polarity."""
    return AUDIO_TO_POLARITY.get(emotion_label.lower(), "neutral")


# ── Text ↔ Image fusion (original) ───────────────────────────────────────────


def fuse_modalities(
    visual_result: dict,
    text_result: dict,
    visual_weight: float = DEFAULT_VISUAL_WEIGHT,
    text_weight: float = DEFAULT_TEXT_WEIGHT,
) -> dict:
    """Weighted-average fusion of visual (face) and text polarity vectors.

    Aggregates the 7-class face-emotion probability distribution into
    3-class polarity scores, then blends with the text model\'s 3-class
    output using the supplied weights.

    Parameters
    ----------
    visual_result : dict
        Output of ``CNNEmotionModel.predict()``.
        Expected keys: ``emotion``, ``confidence``, ``polarity``, ``probs``
        (dict of 7 emotion → probability).
    text_result : dict
        Output of ``TextSentimentModel.predict()``.
        Expected keys: ``label``, ``confidence``, ``probs``
        (dict of 3 polarity → probability).
    visual_weight : float
        Weight applied to the visual confidence vector (default 0.50).
    text_weight : float
        Weight applied to the text confidence vector (default 0.50).

    Returns
    -------
    dict
        fused_label      – str  winning polarity label
        fused_confidence – float confidence of the winning polarity
        fused_probs      – dict per-polarity fused score
    """

    def _visual_polarity_probs(v_result: dict) -> dict[str, float]:
        """Aggregate 7-class emotion probs into 3-class polarity probs."""
        pol = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        for emotion, prob in v_result["probs"].items():
            pol[VISUAL_TO_POLARITY.get(emotion, "neutral")] += prob
        return pol

    v_probs = _visual_polarity_probs(visual_result)
    t_probs = text_result["probs"]  # already 3-class

    fused: dict[str, float] = {
        label: visual_weight * v_probs.get(label, 0.0)
        + text_weight * t_probs.get(label, 0.0)
        for label in POLARITY_LABELS
    }
    top_label = max(fused, key=lambda k: fused[k])
    return {
        "fused_label": top_label,
        "fused_confidence": round(fused[top_label], 4),
        "fused_probs": {k: round(v, 4) for k, v in fused.items()},
    }


# ── Text ↔ Audio fusion (new) ─────────────────────────────────────────────────


def fuse_audio_text_modalities(
    audio_result: dict,
    text_result: dict,
    audio_weight: float = DEFAULT_AUDIO_WEIGHT,
    text_weight: float = DEFAULT_TEXT_WEIGHT,
) -> dict:
    """Weighted-average fusion of audio (speech) and text polarity vectors.

    Aggregates the 7-class speech-emotion probability distribution into
    3-class polarity scores, then blends with the text model\'s 3-class
    output using the supplied weights.

    Parameters
    ----------
    audio_result : dict
        Output of ``predict_audio()`` from ``models/audio/audio_model.py``.
        Expected keys: ``emotion``, ``confidence``, ``polarity``, ``probs``
        (dict of 7 emotion → probability).
    text_result : dict
        Output of ``TextSentimentModel.predict()``.
        Expected keys: ``label``, ``confidence``, ``probs``
        (dict of 3 polarity → probability).
    audio_weight : float
        Weight applied to the audio confidence vector (default 0.50).
    text_weight : float
        Weight applied to the text confidence vector (default 0.50).

    Returns
    -------
    dict
        fused_label      – str  winning polarity label
        fused_confidence – float confidence of the winning polarity
        fused_probs      – dict per-polarity fused score
    """

    def _audio_polarity_probs(a_result: dict) -> dict[str, float]:
        """Aggregate 7-class speech-emotion probs into 3-class polarity probs."""
        pol = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        for emotion, prob in a_result["probs"].items():
            pol[AUDIO_TO_POLARITY.get(emotion.lower(), "neutral")] += prob
        return pol

    a_probs = _audio_polarity_probs(audio_result)
    t_probs = text_result["probs"]  # already 3-class

    fused: dict[str, float] = {
        label: audio_weight * a_probs.get(label, 0.0)
        + text_weight * t_probs.get(label, 0.0)
        for label in POLARITY_LABELS
    }
    top_label = max(fused, key=lambda k: fused[k])
    return {
        "fused_label": top_label,
        "fused_confidence": round(fused[top_label], 4),
        "fused_probs": {k: round(v, 4) for k, v in fused.items()},
    }



# models/fusion/mismatch_detector.py
# ====================================
# Mismatch detection for MoodSyncAI — covers two fusion pairs:

#     1.  text ↔ image  (ResNet50V2 face emotion vs RoBERTa text sentiment)
#         → detect_mismatch()        mirrors notebook 3 / 3_fusion_text-image_analysis.ipynb
#         → run_fusion()             convenience wrapper for text-image pair

#     2.  text ↔ audio  (AudioEmotionNet speech emotion vs RoBERTa text sentiment)
#         → detect_audio_text_mismatch()   mirrors notebook 6 / 6_fusion_audio_text.ipynb
#         → run_audio_text_fusion()        convenience wrapper for audio-text pair

# Both pairs share the same three severity levels and the same badge colours /
# emojis — only the polarity mapping and explanation wording differ.

from __future__ import annotations

CONF_THRESHOLD = 0.50  # minimum confidence for HARD_MISMATCH

# ── Polarity maps ─────────────────────────────────────────────────────────────

VISUAL_TO_POLARITY: dict[str, str] = {
    "Happy":    "positive",
    "Surprise": "positive",
    "Neutral":  "neutral",
    "Angry":    "negative",
    "Disgust":  "negative",
    "Fear":     "negative",
    "Sad":      "negative",
}

AUDIO_TO_POLARITY: dict[str, str] = {
    "happy":    "positive",
    "surprise": "positive",
    "neutral":  "neutral",
    "angry":    "negative",
    "disgust":  "negative",
    "fear":     "negative",
    "sad":      "negative",
}

# ── Severity styling ──────────────────────────────────────────────────────────

SEVERITY_COLOURS: dict[str, str] = {
    "MATCH":         "#4CAF50",   # green
    "SOFT_MISMATCH": "#FFA726",   # amber
    "HARD_MISMATCH": "#EF5350",   # red
}

SEVERITY_EMOJI: dict[str, str] = {
    "MATCH":         "✅",
    "SOFT_MISMATCH": "⚠️",
    "HARD_MISMATCH": "❌",
}

# ── Helper ────────────────────────────────────────────────────────────────────

def get_visual_polarity(emotion_label: str) -> str:
    """Map a ResNet50V2 emotion class to 3-class polarity."""
    return VISUAL_TO_POLARITY.get(emotion_label, "neutral")


def get_audio_polarity(emotion_label: str) -> str:
    """Map an AudioEmotionNet emotion class to 3-class polarity."""
    return AUDIO_TO_POLARITY.get(emotion_label.lower(), "neutral")


# ── Text ↔ Image mismatch (original) ─────────────────────────────────────────

def detect_mismatch(
    visual_result: dict,
    text_result:   dict,
    conf_threshold: float = CONF_THRESHOLD,
) -> dict:
    """Compare visual (face) and text polarities and classify the conflict level.

    Severity rules
    --------------
    MATCH         – both polarities agree
    SOFT_MISMATCH – polarities differ but at least one confidence < conf_threshold
    HARD_MISMATCH – polarities differ AND both confidences ≥ conf_threshold

    Parameters
    ----------
    visual_result : dict
        Output of ``CNNEmotionModel.predict()`` — must contain
        ``"emotion"``, ``"confidence"``, and optionally ``"polarity"``.
    text_result : dict
        Output of ``TextSentimentModel.predict()`` — must contain
        ``"label"`` and ``"confidence"``.
    conf_threshold : float
        Minimum confidence required for HARD_MISMATCH (default 0.50).

    Returns
    -------
    dict
        severity, visual_emotion, visual_polarity, text_polarity,
        polarity_match, both_confident, badge_colour, emoji, explanation.
    """
    visual_polarity = visual_result.get("polarity") or get_visual_polarity(
        visual_result["emotion"]
    )
    text_polarity  = text_result["label"]
    polarity_match = visual_polarity == text_polarity
    both_confident = (
        visual_result["confidence"] >= conf_threshold
        and text_result["confidence"] >= conf_threshold
    )

    if polarity_match:
        severity = "MATCH"
        explanation = (
            f"Both modalities agree: face shows {visual_result[\'emotion\']} "
            f"({visual_polarity}) and text is {text_polarity}."
        )
    elif both_confident:
        severity = "HARD_MISMATCH"
        explanation = (
            f"Mismatch detected: face shows {visual_result[\'emotion\']} "
            f"({visual_polarity}, conf={visual_result[\'confidence\']:.0%}) "
            f"but text is {text_polarity} "
            f"(conf={text_result[\'confidence\']:.0%})."
        )
    else:
        severity = "SOFT_MISMATCH"
        explanation = (
            f"Uncertain: face shows {visual_result[\'emotion\']} "
            f"({visual_polarity}, conf={visual_result[\'confidence\']:.0%}) "
            f"vs text {text_polarity} "
            f"(conf={text_result[\'confidence\']:.0%}). "
            "One or both modalities lack sufficient confidence."
        )

    return {
        "severity":        severity,
        "visual_emotion":  visual_result["emotion"],
        "visual_polarity": visual_polarity,
        "text_polarity":   text_polarity,
        "polarity_match":  polarity_match,
        "both_confident":  both_confident,
        "badge_colour":    SEVERITY_COLOURS[severity],
        "emoji":           SEVERITY_EMOJI[severity],
        "explanation":     explanation,
    }


def run_fusion(
    visual_result: dict,
    text_result:   dict,
    conf_threshold: float = CONF_THRESHOLD,
) -> dict:
    """Text-image fusion pipeline.

    Parameters
    ----------
    visual_result : dict   Output of ``CNNEmotionModel.predict()``.
    text_result   : dict   Output of ``TextSentimentModel.predict()``.
    conf_threshold: float  Threshold for HARD_MISMATCH classification.

    Returns
    -------
    dict  with keys: visual, text, fusion.
    """
    mismatch = detect_mismatch(visual_result, text_result, conf_threshold)
    return {
        "visual": visual_result,
        "text":   text_result,
        "fusion": mismatch,
    }


# ── Text ↔ Audio mismatch (new) ───────────────────────────────────────────────

def detect_audio_text_mismatch(
    audio_result:  dict,
    text_result:   dict,
    conf_threshold: float = CONF_THRESHOLD,
) -> dict:
    """Compare audio (speech) and text polarities and classify the conflict.

    Identical severity logic to ``detect_mismatch()`` — only the polarity
    source and explanation wording differ.

    Parameters
    ----------
    audio_result : dict
        Output of ``predict_audio()`` from ``models/audio/audio_model.py``.
        Must contain ``"emotion"``, ``"confidence"``, and ``"polarity"``.
    text_result : dict
        Output of ``TextSentimentModel.predict()`` — must contain
        ``"label"`` and ``"confidence"``.
    conf_threshold : float
        Minimum confidence required for HARD_MISMATCH (default 0.50).

    Returns
    -------
    dict
        severity, audio_emotion, audio_polarity, text_polarity,
        polarity_match, both_confident, badge_colour, emoji, explanation.
    """
    audio_polarity = audio_result.get("polarity") or get_audio_polarity(
        audio_result["emotion"]
    )
    text_polarity  = text_result["label"]
    polarity_match = audio_polarity == text_polarity
    both_confident = (
        audio_result["confidence"] >= conf_threshold
        and text_result["confidence"] >= conf_threshold
    )

    if polarity_match:
        severity = "MATCH"
        explanation = (
            f"Both modalities agree: audio shows {audio_result[\'emotion\']} "
            f"({audio_polarity}) and text is {text_polarity}."
        )
    elif both_confident:
        severity = "HARD_MISMATCH"
        explanation = (
            f"Mismatch detected: audio shows {audio_result[\'emotion\']} "
            f"({audio_polarity}, conf={audio_result[\'confidence\']:.0%}) "
            f"but text is {text_polarity} "
            f"(conf={text_result[\'confidence\']:.0%})."
        )
    else:
        severity = "SOFT_MISMATCH"
        explanation = (
            f"Uncertain: audio shows {audio_result[\'emotion\']} "
            f"({audio_polarity}, conf={audio_result[\'confidence\']:.0%}) "
            f"vs text {text_polarity} "
            f"(conf={text_result[\'confidence\']:.0%}). "
            "One or both modalities lack sufficient confidence."
        )

    return {
        "severity":       severity,
        "audio_emotion":  audio_result["emotion"],
        "audio_polarity": audio_polarity,
        "text_polarity":  text_polarity,
        "polarity_match": polarity_match,
        "both_confident": both_confident,
        "badge_colour":   SEVERITY_COLOURS[severity],
        "emoji":          SEVERITY_EMOJI[severity],
        "explanation":    explanation,
    }


def run_audio_text_fusion(
    audio_result:  dict,
    text_result:   dict,
    conf_threshold: float = CONF_THRESHOLD,
) -> dict:
    """Text-audio fusion pipeline.

    Parameters
    ----------
    audio_result  : dict   Output of ``predict_audio()``.
    text_result   : dict   Output of ``TextSentimentModel.predict()``.
    conf_threshold: float  Threshold for HARD_MISMATCH classification.

    Returns
    -------
    dict  with keys: audio, text, fusion.
    """
    mismatch = detect_audio_text_mismatch(audio_result, text_result, conf_threshold)
    return {
        "audio": audio_result,
        "text":  text_result,
        "fusion": mismatch,
    }
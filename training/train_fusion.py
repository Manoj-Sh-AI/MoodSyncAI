"""
training/train_fusion.py

Fusion layer training / configuration for MoodSyncAI.

Two strategies are supported:

1.  weighted_average (default, no training required)
    visual × 0.6  +  text × 0.4  → fused 3-class probability

2.  learned_mlp  (optional — set fusion.strategy: learned_mlp in config)
    A small MLP trained on paired (visual_probs, text_probs, true_label) data.
    Requires a labelled fusion dataset at data/processed/fusion_train.csv.

Also exposes the mismatch detector logic used in the fusion pipeline.

Usage
-----
    python -m training.train_fusion                        # default config
    python -m training.train_fusion --strategy learned_mlp
    python -m training.train_fusion --visual_weight 0.7 --text_weight 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "fusion_config.yaml"
LABEL_MAP      = {0: "negative", 1: "neutral", 2: "positive"}
LABELS         = ["negative", "neutral", "positive"]


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Weighted-average fusion ───────────────────────────────────────────────────

def weighted_average_fusion(
    visual_probs: np.ndarray | list[float],
    text_probs:   np.ndarray | list[float],
    visual_weight: float = 0.6,
    text_weight:   float = 0.4,
) -> dict:
    """
    Combine visual (3-class mapped) and text probability vectors.

    Parameters
    ----------
    visual_probs   : shape (3,)  — [negative, neutral, positive]
                     produced by mapping ResNet50V2's 7-class output via
                     models.fusion.mismatch_detector.map_visual_to_3class()
    text_probs     : shape (3,)  — direct RoBERTa output
    visual_weight  : weight for the visual branch (default 0.6)
    text_weight    : weight for the text branch (default 0.4)

    Returns
    -------
    dict with keys: label, label_id, confidence, probs
    """
    v = np.array(visual_probs, dtype=float)
    t = np.array(text_probs,   dtype=float)

    assert abs(visual_weight + text_weight - 1.0) < 1e-6, "Weights must sum to 1.0"
    assert len(v) == len(t) == 3, "Both probability vectors must have 3 elements"

    fused   = visual_weight * v + text_weight * t
    top_id  = int(np.argmax(fused))

    return {
        "label":      LABEL_MAP[top_id],
        "label_id":   top_id,
        "confidence": round(float(fused[top_id]), 4),
        "probs":      {LABEL_MAP[i]: round(float(fused[i]), 4) for i in range(3)},
    }


# ── Mismatch detection ────────────────────────────────────────────────────────

SEVERITY_MATCH        = "MATCH"
SEVERITY_SOFT_MISMATCH = "SOFT_MISMATCH"
SEVERITY_HARD_MISMATCH = "HARD_MISMATCH"


def detect_sentiment_mismatch(
    visual_label:      str,
    visual_confidence: float,
    text_label:        str,
    text_confidence:   float,
    soft_threshold:    float = 0.25,
    hard_threshold:    float = 0.50,
) -> dict:
    """
    Determine whether the visual and text sentiment signals align.

    Rules
    -----
    1. MATCH         — same label, OR confidence gap ≤ soft_threshold
    2. SOFT_MISMATCH — different label, confidence gap in (soft, hard]
    3. HARD_MISMATCH — opposite polarity (positive ↔ negative), OR
                       different label with confidence gap > hard_threshold

    Visual→text polarity mapping:
        happy / surprise  → positive
        neutral           → neutral
        angry / disgust / fear / sad → negative

    Mirrors Section 11 of 2_text_model_experiments.ipynb.

    Returns
    -------
    dict with keys: severity, visual_label, text_label,
                    visual_confidence, text_confidence, confidence_gap
    """
    POSITIVE_EMOTIONS = {"positive", "happy", "surprise"}
    NEGATIVE_EMOTIONS = {"negative", "angry", "disgust", "fear", "sad"}

    def polarity(label: str) -> str:
        if label in POSITIVE_EMOTIONS: return "positive"
        if label in NEGATIVE_EMOTIONS: return "negative"
        return "neutral"

    vis_polarity  = polarity(visual_label)
    text_polarity = polarity(text_label)
    conf_gap      = abs(visual_confidence - text_confidence)

    # Polarity inversion always → HARD
    if vis_polarity != text_polarity and {vis_polarity, text_polarity} == {"positive", "negative"}:
        severity = SEVERITY_HARD_MISMATCH

    elif visual_label == text_label or (vis_polarity == text_polarity and conf_gap <= soft_threshold):
        severity = SEVERITY_MATCH

    elif conf_gap > hard_threshold:
        severity = SEVERITY_HARD_MISMATCH

    elif conf_gap > soft_threshold:
        severity = SEVERITY_SOFT_MISMATCH

    else:
        severity = SEVERITY_MATCH

    return {
        "severity":          severity,
        "visual_label":      visual_label,
        "text_label":        text_label,
        "visual_confidence": visual_confidence,
        "text_confidence":   text_confidence,
        "confidence_gap":    round(conf_gap, 4),
    }


# ── Learned MLP fusion (optional) ─────────────────────────────────────────────

def build_fusion_mlp(cfg: dict):
    """
    Build the optional learned neural fusion layer.
    Input: concatenated [visual_probs(3) + text_probs(3)] = 6-dim vector
    Output: softmax over 3 sentiment classes
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    mlp_cfg    = cfg["fusion"]["mlp"]
    hidden     = mlp_cfg["hidden_units"]
    dropout    = mlp_cfg["dropout_rate"]
    activation = mlp_cfg["activation"]

    inputs = layers.Input(shape=(6,), name="fused_input")
    x = inputs
    for units in hidden:
        x = layers.Dense(units, activation=activation)(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(3, activation=mlp_cfg["output_activation"])(x)

    return models.Model(inputs=inputs, outputs=outputs, name="fusion_mlp")


def train_fusion_mlp(cfg: dict):
    """
    Train the learned MLP fusion layer on a labelled CSV dataset.

    The CSV must have columns:
        vis_neg, vis_neu, vis_pos,   (visual 3-class probs)
        txt_neg, txt_neu, txt_pos,   (text 3-class probs)
        label                        (int: 0=neg, 1=neu, 2=pos)
    """
    import pandas as pd
    import tensorflow as tf

    train_cfg = cfg["training"]
    data_path = train_cfg["dataset"]

    print(f"Loading fusion dataset: {data_path}")
    df = pd.read_csv(data_path)

    X = df[["vis_neg", "vis_neu", "vis_pos",
            "txt_neg", "txt_neu", "txt_pos"]].values
    y = tf.keras.utils.to_categorical(df["label"].values, num_classes=3)

    model = build_fusion_mlp(cfg)
    model.summary()
    model.compile(
        optimizer=train_cfg["optimizer"],
        loss=train_cfg["loss"],
        metrics=train_cfg["metrics"],
    )

    history = model.fit(
        X, y,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        validation_split=0.15,
    )

    save_path = cfg["output"]["save_path"]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path.replace(".pt", ".keras"))
    print(f"Fusion MLP saved → {save_path}")
    return model, history


# ── Demo / validation ─────────────────────────────────────────────────────────

def demo_weighted_fusion(cfg: dict):
    """
    Quick sanity check of the weighted-average fusion and mismatch detector.
    Mirrors the assignment brief scenario from the notebook.
    """
    vw = cfg["fusion"]["visual_weight"]
    tw = cfg["fusion"]["text_weight"]
    st = cfg["mismatch"]["soft_threshold"]
    ht = cfg["mismatch"]["hard_threshold"]

    scenarios = [
        # (visual_label, visual_probs, text_label, text_probs)
        ("happy",   [0.05, 0.05, 0.90], "positive", [0.05, 0.05, 0.90]),   # MATCH
        ("sad",     [0.80, 0.10, 0.10], "positive", [0.05, 0.05, 0.90]),   # HARD (polarity inversion)
        ("neutral", [0.15, 0.70, 0.15], "neutral",  [0.20, 0.60, 0.20]),   # MATCH
        ("sad",     [0.60, 0.25, 0.15], "neutral",  [0.25, 0.55, 0.20]),   # SOFT
    ]

    print("\nWeighted-Average Fusion Demo")
    print("-" * 60)
    for vis_label, vis_probs, txt_label, txt_probs in scenarios:
        fused    = weighted_average_fusion(vis_probs, txt_probs, vw, tw)
        mismatch = detect_sentiment_mismatch(
            vis_label, max(vis_probs),
            txt_label, max(txt_probs),
            soft_threshold=st, hard_threshold=ht,
        )
        print(f"Visual: {vis_label:8s} | Text: {txt_label:8s} | "
              f"Fused: {fused['label']:8s} ({fused['confidence'] * 100:.1f}%) | "
              f"{mismatch['severity']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict, strategy_override: str | None = None,
         visual_weight: float | None = None,
         text_weight:   float | None = None):

    if strategy_override:
        cfg["fusion"]["strategy"] = strategy_override
    if visual_weight is not None:
        cfg["fusion"]["visual_weight"] = visual_weight
        cfg["fusion"]["text_weight"]   = 1.0 - visual_weight
    if text_weight is not None:
        cfg["fusion"]["text_weight"]   = text_weight
        cfg["fusion"]["visual_weight"] = 1.0 - text_weight

    strategy = cfg["fusion"]["strategy"]
    print(f"Fusion strategy: {strategy}")

    if strategy == "weighted_average":
        demo_weighted_fusion(cfg)
    elif strategy == "learned_mlp":
        train_fusion_mlp(cfg)
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy!r}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train / configure MoodSyncAI fusion layer")
    p.add_argument("--config",         default=str(DEFAULT_CONFIG))
    p.add_argument("--strategy",       choices=["weighted_average", "learned_mlp"])
    p.add_argument("--visual_weight",  type=float)
    p.add_argument("--text_weight",    type=float)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = load_config(args.config)
    main(cfg,
         strategy_override=args.strategy,
         visual_weight=args.visual_weight,
         text_weight=args.text_weight)

"""
models/text/sentiment_model.py
cardiffnlp/twitter-roberta-base-sentiment-latest wrapper for MoodSyncAI.
Mirrors the production snippet from 2_text_model_experiments.ipynb (Section 14).
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

SAVE_DIR = Path("saved_models/roberta_sentiment")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def preprocess_text(text: str) -> str:
    """
    Replace Twitter @usernames with 'user' and URLs with 'http'.
    Matches the preprocessing applied during training on 124M tweets.
    """
    tokens = []
    for t in text.split():
        if t.startswith("@") and len(t) > 1:
            tokens.append("@user")
        elif t.startswith("http"):
            tokens.append("http")
        else:
            tokens.append(t)
    return " ".join(tokens)


class TextSentimentModel:
    """
    Wraps cardiffnlp/twitter-roberta-base-sentiment-latest for inference.

    Loads from saved_models/roberta_sentiment/ (run 2_text_model_experiments.ipynb first).
    Falls back to the HuggingFace Hub if the local directory is absent.

    Usage
    -----
    model = TextSentimentModel()
    result = model.predict("No, I think the project is going really well.")
    # {"label": "positive", "label_id": 2, "confidence": 0.9673,
    #  "probs": {"negative": 0.0056, "neutral": 0.027, "positive": 0.9673}}
    """

    VISUAL_TO_POLARITY = {
        "Happy":    "positive",
        "Surprise": "positive",
        "Neutral":  "neutral",
        "Angry":    "negative",
        "Disgust":  "negative",
        "Fear":     "negative",
        "Sad":      "negative",
    }

    CONF_THRESHOLD = 0.50
    MISMATCH_LEVELS = ("MATCH", "SOFT_MISMATCH", "HARD_MISMATCH")

    def __init__(self, model_dir: str | None = None):
        src = model_dir or (str(SAVE_DIR) if SAVE_DIR.exists() else MODEL_ID)
        self.config    = AutoConfig.from_pretrained(src)
        self.tokenizer = AutoTokenizer.from_pretrained(src)
        self.model     = AutoModelForSequenceClassification.from_pretrained(src)
        self.model     = self.model.to(DEVICE)
        self.model.eval()

    # ------------------------------------------------------------------
    def predict(self, text: str, max_len: int = 128) -> dict:
        """
        Run inference on a single text string.

        Returns
        -------
        dict with keys: label, label_id, confidence, probs
        """
        clean = preprocess_text(text)
        encoded = self.tokenizer(
            clean,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_len,
        ).to(DEVICE)

        with torch.no_grad():
            logits = self.model(**encoded).logits           # (1, 3)
            scores = torch.softmax(logits, dim=-1)
            scores = scores.squeeze().cpu().numpy()         # (3,)

        top_id = int(np.argmax(scores))
        num_classes = len(self.config.id2label)
        return {
            "label":      self.config.id2label[top_id],
            "label_id":   top_id,
            "confidence": round(float(scores[top_id]), 4),
            "probs":      {
                self.config.id2label[i]: round(float(scores[i]), 4)
                for i in range(num_classes)
            },
        }

    # ------------------------------------------------------------------
    def predict_batch(self, texts: list, batch_size: int = 32, max_len: int = 128) -> list:
        """Run inference on a list of text strings."""
        results = []
        for i in range(0, len(texts), batch_size):
            raw_batch  = texts[i : i + batch_size]
            clean_batch = [preprocess_text(t) for t in raw_batch]
            encoded = self.tokenizer(
                clean_batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_len,
            ).to(DEVICE)
            with torch.no_grad():
                logits = self.model(**encoded).logits
                scores = torch.softmax(logits, dim=-1).cpu().numpy()

            for j, sc in enumerate(scores):
                top_id = int(np.argmax(sc))
                num_classes = len(self.config.id2label)
                results.append({
                    "text":       raw_batch[j],
                    "label":      self.config.id2label[top_id],
                    "label_id":   top_id,
                    "confidence": round(float(sc[top_id]), 4),
                    "probs":      {
                        self.config.id2label[k]: round(float(sc[k]), 4)
                        for k in range(num_classes)
                    },
                })
        return results

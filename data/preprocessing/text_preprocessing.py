"""
data/preprocessing/text_preprocessing.py

Text cleaning, tokenisation, and dataset loading helpers for MoodSyncAI.
Mirrors the preprocessing pipeline used in 2_text_model_experiments.ipynb.

The core preprocessing function (preprocess_text) is shared with
models/text/sentiment_model.py — any changes here must be reflected there.

Usage
-----
    from data.preprocessing.text_preprocessing import (
        preprocess_text,
        load_tweeteval_splits,
        tokenize_batch,
    )
    clean = preprocess_text("Check @john https://example.com great stuff!")
    # "Check @user http great stuff!"
"""

from __future__ import annotations

import re
from pathlib import Path

# ── Twitter-aware preprocessing ───────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Replicate the tweet preprocessing applied during
    cardiffnlp/twitter-roberta-base-sentiment-latest training:
        - @username  → @user
        - http(s)://… → http

    This MUST be applied before tokenising for the RoBERTa model.
    Source: cardiffnlp HuggingFace model card.

    Parameters
    ----------
    text : raw user input string

    Returns
    -------
    cleaned string with usernames and URLs replaced
    """
    tokens = []
    for token in text.split():
        if token.startswith("@") and len(token) > 1:
            tokens.append("@user")
        elif token.startswith("http"):
            tokens.append("http")
        else:
            tokens.append(token)
    return " ".join(tokens)


def clean_text(text: str) -> str:
    """
    Light general-purpose text cleaning:
      - Strip leading / trailing whitespace
      - Collapse multiple spaces
      - Apply Twitter preprocessing (@user, http)

    Parameters
    ----------
    text : raw string

    Returns
    -------
    cleaned string
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return preprocess_text(text)


# ── TweetEval dataset helpers ─────────────────────────────────────────────────

def load_tweeteval_splits(split: str = "test") -> "datasets.Dataset":
    """
    Load a TweetEval sentiment split from the HuggingFace Hub.

    TweetEval is the benchmark used to fine-tune the RoBERTa sentiment model.
    Label mapping: 0 = negative, 1 = neutral, 2 = positive.

    Parameters
    ----------
    split : "train" | "validation" | "test"

    Returns
    -------
    datasets.Dataset with columns: text (str), label (int)
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install the `datasets` library: pip install datasets") from e

    return load_dataset("tweeteval", "sentiment", split=split)


def load_tweeteval_as_dataframe(split: str = "test"):
    """
    Return TweetEval split as a pandas DataFrame with columns [text, label, label_name].
    """
    import pandas as pd

    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
    ds = load_tweeteval_splits(split)
    df = ds.to_pandas()
    df["label_name"] = df["label"].map(LABEL_MAP)
    return df


# ── HuggingFace tokenisation helpers ─────────────────────────────────────────

def tokenize_batch(
    texts: list[str],
    tokenizer,
    max_len: int = 128,
    apply_preprocessing: bool = True,
) -> dict:
    """
    Tokenise a batch of texts using any HuggingFace tokenizer.

    Parameters
    ----------
    texts               : list of raw text strings
    tokenizer           : loaded HuggingFace tokenizer
    max_len             : maximum token length (default 128 — sufficient for
                          spoken sentences; RoBERTa supports up to 512)
    apply_preprocessing : if True, apply Twitter @user / http replacement first

    Returns
    -------
    dict with keys: input_ids, attention_mask (pytorch tensors, batch-first)
    """
    if apply_preprocessing:
        texts = [preprocess_text(t) for t in texts]

    return tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len,
    )


# ── Processed data helpers ────────────────────────────────────────────────────

def save_processed_texts(
    texts: list[str],
    labels: list[int],
    output_path: str | Path,
    split_name: str = "processed",
) -> None:
    """
    Save preprocessed texts and labels to a CSV file in data/processed/.

    Parameters
    ----------
    texts       : list of cleaned text strings
    labels      : integer label list (0=negative, 1=neutral, 2=positive)
    output_path : destination file path
    split_name  : informational label used in the saved CSV column
    """
    import pandas as pd

    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "text":       texts,
        "label":      labels,
        "label_name": [LABEL_MAP.get(l, "unknown") for l in labels],
        "split":      split_name,
    })
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} samples → {output_path}")


def load_processed_texts(filepath: str | Path):
    """
    Load a preprocessed CSV produced by save_processed_texts().

    Returns
    -------
    pd.DataFrame  with columns: text, label, label_name, split
    """
    import pandas as pd
    return pd.read_csv(filepath)

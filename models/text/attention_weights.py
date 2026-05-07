"""
models/text/attention_weights.py
Extracts last-layer average token attention from the RoBERTa sentiment model.
Mirrors Section 10 of 2_text_model_experiments.ipynb.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from models.text.sentiment_model import DEVICE, preprocess_text


def get_attention_weights(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    config: AutoConfig,
    max_len: int = 128,
) -> dict:
    """
    Return the last-layer CLS-row attention for each token.

    Parameters
    ----------
    text      : raw user input (preprocessing applied internally)
    model     : loaded RoBERTa model (output_attentions must be enabled)
    tokenizer : matching tokenizer
    config    : model config (for label mapping)
    max_len   : max token length

    Returns
    -------
    dict with keys:
        tokens    – list[str]   token strings
        attention – np.ndarray  normalised attention weights, shape (seq_len,)
        label     – str         predicted sentiment label
        confidence– float       confidence of predicted label
    """
    clean = preprocess_text(text)
    encoded = tokenizer(
        clean,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len,
    ).to(DEVICE)

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    # Last layer attention: (1, num_heads, seq_len, seq_len)
    last_attn = outputs.attentions[-1]               # (1, H, L, L)
    avg_attn  = last_attn.squeeze(0).mean(0)         # (L, L)   avg over heads
    cls_attn  = avg_attn[0].cpu().numpy()            # (L,)     CLS row

    # Normalise
    cls_attn = cls_attn / (cls_attn.sum() + 1e-8)

    # Sentiment from logits
    scores   = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
    top_id   = int(np.argmax(scores))

    return {
        "tokens":     tokens,
        "attention":  cls_attn,
        "label":      config.id2label[top_id],
        "confidence": round(float(scores[top_id]), 4),
    }

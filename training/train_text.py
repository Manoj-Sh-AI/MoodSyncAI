"""
training/train_text.py

Load, evaluate, and optionally fine-tune the RoBERTa sentiment model
for MoodSyncAI.

The model (cardiffnlp/twitter-roberta-base-sentiment-latest) is already
fine-tuned on TweetEval — this script:
  1. Downloads and verifies the model from HuggingFace Hub
  2. Runs the full TweetEval benchmark evaluation (72.22 % accuracy)
  3. Saves the model + tokenizer to saved_models/roberta_sentiment/
  4. Optionally continues fine-tuning (fine_tuning.enabled = true in config)

Usage
-----
    python -m training.train_text                          # default config
    python -m training.train_text --config training/configs/text_config.yaml
    python -m training.train_text --eval_only              # skip saving
    python -m training.train_text --fine_tune              # enable fine-tuning
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings("ignore")

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "text_config.yaml"


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    """
    Load cardiffnlp/twitter-roberta-base-sentiment-latest from HuggingFace Hub.
    Uses AutoConfig to pull the canonical id2label mapping — no hard-coded labels.
    Mirrors Section 1 of 2_text_model_experiments.ipynb.
    """
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    model_id = cfg["model"]["model_id"]
    device_str = cfg["model"]["device"]
    if device_str == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str

    print(f"Loading tokenizer and model from HuggingFace Hub: {model_id}")
    t0 = time.time()
    config    = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSequenceClassification.from_pretrained(model_id)
    model     = model.to(device)
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s  |  Device: {device}")

    labels = [config.id2label[i] for i in range(len(config.id2label))]
    print(f"Label mapping: {dict(enumerate(labels))}")
    return model, tokenizer, config, device


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Official Twitter preprocessing for this model:
      @username → @user  |  http(s)://… → http
    Mirrors Section 2 of 2_text_model_experiments.ipynb.
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


# ── Inference ─────────────────────────────────────────────────────────────────

def get_text_sentiment(
    text: str,
    model,
    tokenizer,
    config,
    device: str,
    max_len: int = 128,
) -> dict:
    """
    Single-text sentiment inference.
    Returns: {label, label_id, confidence, probs: {label: float}}
    Mirrors Section 3 of 2_text_model_experiments.ipynb.
    """
    import torch

    num_classes = len(config.id2label)
    clean = preprocess_text(text)
    encoded = tokenizer(
        clean,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len,
    ).to(device)

    with torch.no_grad():
        logits = model(**encoded).logits          # (1, 3)
        scores = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()  # (3,)

    top_id = int(np.argmax(scores))
    return {
        "label":      config.id2label[top_id],
        "label_id":   top_id,
        "confidence": round(float(scores[top_id]), 4),
        "probs":      {config.id2label[i]: round(float(scores[i]), 4)
                       for i in range(num_classes)},
    }


def get_text_sentiment_batch(
    texts: list[str],
    model,
    tokenizer,
    config,
    device: str,
    batch_size: int = 64,
    max_len: int = 128,
) -> list[dict]:
    """
    Batch inference over a list of texts.
    Mirrors Section 4 of 2_text_model_experiments.ipynb.
    """
    import torch

    num_classes = len(config.id2label)
    results = []
    for i in range(0, len(texts), batch_size):
        raw   = texts[i: i + batch_size]
        clean = [preprocess_text(t) for t in raw]
        encoded = tokenizer(
            clean,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_len,
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits
            scores = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, 3)

        for j, sc in enumerate(scores):
            top_id = int(np.argmax(sc))
            results.append({
                "text":       raw[j],
                "label":      config.id2label[top_id],
                "label_id":   top_id,
                "confidence": round(float(sc[top_id]), 4),
                "probs":      {config.id2label[k]: round(float(sc[k]), 4)
                               for k in range(num_classes)},
            })
    return results


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_tweeteval(model, tokenizer, config, device: str, cfg: dict) -> dict:
    """
    Run the full TweetEval sentiment benchmark.
    Expected results (matching notebook):
        Accuracy: 72.22%  |  F1 macro: 0.7244

    Mirrors Section 5 of 2_text_model_experiments.ipynb.
    """
    from datasets import load_dataset
    from sklearn.metrics import classification_report, confusion_matrix

    print("Loading TweetEval sentiment test split...")
    ds      = load_dataset("tweeteval", "sentiment", split="test")
    texts   = ds["text"]
    true_labels = ds["label"]

    batch_size = cfg["inference"]["batch_size"]
    print(f"Running inference on {len(ds):,} test samples...")
    t0    = time.time()
    preds = get_text_sentiment_batch(texts, model, tokenizer, config, device,
                                     batch_size=batch_size)
    elapsed = time.time() - t0

    pred_labels = [p["label_id"] for p in preds]
    correct     = sum(p == t for p, t in zip(pred_labels, true_labels))
    acc         = correct / len(true_labels)

    print(f"\nEvaluation complete in {elapsed:.1f}s "
          f"({len(ds) / elapsed:.0f} samples/sec)")
    print(f"Overall Accuracy: {acc * 100:.2f}%  ({correct}/{len(true_labels)})")
    print("\nPer-class metrics:")
    print(classification_report(
        true_labels, pred_labels,
        target_names=["negative", "neutral", "positive"],
        digits=4,
    ))

    return {
        "accuracy": acc,
        "pred_labels":  pred_labels,
        "true_labels":  true_labels,
        "report": classification_report(
            true_labels, pred_labels,
            target_names=["negative", "neutral", "positive"],
            output_dict=True,
        ),
    }


# ── Save ──────────────────────────────────────────────────────────────────────

def save_model(model, tokenizer, config, cfg: dict):
    """
    Save the model, tokenizer, and label config to saved_models/roberta_sentiment/.
    Mirrors Section 10 of 2_text_model_experiments.ipynb.
    """
    import json

    save_dir = Path(cfg["save"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    if cfg["save"]["tokenizer"]:
        tokenizer.save_pretrained(str(save_dir))
        print(f"Tokenizer saved → {save_dir}")
    if cfg["save"]["model"]:
        model.save_pretrained(str(save_dir))
        print(f"Model saved     → {save_dir}")
    if cfg["save"]["config"]:
        with open(save_dir / "label_config.json", "w") as f:
            json.dump({
                "id2label": config.id2label,
                "label2id": config.label2id,
                "model_id": cfg["model"]["model_id"],
            }, f, indent=2)
        print(f"Label config    → {save_dir / 'label_config.json'}")


# ── Optional fine-tuning ──────────────────────────────────────────────────────

def fine_tune(model, tokenizer, config, device: str, cfg: dict):
    """
    Optional continued fine-tuning on TweetEval using HuggingFace Trainer.
    Only runs when fine_tuning.enabled = true in config.
    """
    from datasets import load_dataset
    from transformers import Trainer, TrainingArguments

    ft_cfg = cfg["fine_tuning"]
    print("\nStarting fine-tuning...")

    dataset = load_dataset("tweeteval", "sentiment")

    def tokenize(batch):
        return tokenizer(
            [preprocess_text(t) for t in batch["text"]],
            truncation=True,
            padding="max_length",
            max_length=cfg["model"]["max_len"],
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=str(Path(cfg["save"]["save_dir"]) / "ft_checkpoints"),
        num_train_epochs=ft_cfg["epochs"],
        per_device_train_batch_size=cfg["inference"]["batch_size"],
        learning_rate=ft_cfg["learning_rate"],
        weight_decay=ft_cfg["weight_decay"],
        warmup_steps=ft_cfg["warmup_steps"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    trainer.train()
    print("Fine-tuning complete.")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict, eval_only: bool = False, fine_tune_flag: bool = False):
    model, tokenizer, config, device = load_model_and_tokenizer(cfg)

    # Smoke test (assignment brief scenario)
    demo = "No, I think the project is going really well."
    result = get_text_sentiment(demo, model, tokenizer, config, device)
    print(f"\nSmoke test: '{demo}'")
    print(f"  → {result['label'].upper()}  {result['confidence'] * 100:.1f}%")

    # Evaluation
    eval_results = evaluate_tweeteval(model, tokenizer, config, device, cfg)

    # Fine-tuning (optional)
    if fine_tune_flag or cfg["fine_tuning"]["enabled"]:
        model = fine_tune(model, tokenizer, config, device, cfg)

    # Save
    if not eval_only:
        save_model(model, tokenizer, config, cfg)

    return model, tokenizer, config, eval_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Load/evaluate/fine-tune RoBERTa sentiment model")
    p.add_argument("--config",     default=str(DEFAULT_CONFIG))
    p.add_argument("--eval_only",  action="store_true",
                   help="Skip saving — only evaluate")
    p.add_argument("--fine_tune",  action="store_true",
                   help="Enable fine-tuning regardless of config flag")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = load_config(args.config)
    main(cfg, eval_only=args.eval_only, fine_tune_flag=args.fine_tune)

"""
models/visual/video_model.py
R3D-18 + Temporal Attention Pool backbone for 7-class emotion recognition.
Mirrors the architecture defined in notebooks/7_video_emotion_recognition_ravdess.ipynb.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class TemporalAttentionPool(nn.Module):
    """Soft attention over the temporal dimension after R3D spatial features.

    Accepts a 5-D tensor (B, C, T, H, W) and performs spatial mean-pooling
    to (B, C, T), then learns a per-frame attention weight.
    """

    def __init__(self, in_dim: int = 512):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise RuntimeError(
                f"TemporalAttentionPool expects a 5-D tensor (B,C,T,H,W), got {x.dim()}D"
            )
        # Spatial mean  →  (B, C, T)
        x = x.mean(dim=(3, 4))
        # Transpose      →  (B, T, C)
        x = x.permute(0, 2, 1)
        # Attention over time  →  (B, T, 1)
        w = torch.softmax(self.attn(x), dim=1)
        # Weighted sum   →  (B, C)
        return (w * x).sum(dim=1)


class VideoEmotionModel(nn.Module):
    """R3D-18 backbone + Temporal Attention + FC head for 7-class emotion.

    Architecture:
        R3D-18 (Kinetics-400 pretrained, last avg-pool & fc removed)
        → TemporalAttentionPool(512)
        → LayerNorm(512)
        → Linear(512, 256) → ReLU → Dropout(0.4)
        → Linear(256, num_classes)
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        backbone_r3d = r3d_18(weights=None)
        # Drop the final average-pool and fully-connected layers
        self.backbone = nn.Sequential(*list(backbone_r3d.children())[:-2])
        self.temp_attn = TemporalAttentionPool(512)
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)          # (B, C, T, H, W)
        pooled = self.temp_attn(feats)    # (B, C)
        return self.head(pooled)          # (B, num_classes)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _extract_state_dict(obj) -> Optional[dict]:
    """Handle checkpoints saved as {'state_dict': ...} or directly as a dict."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "net", "weights"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj if isinstance(obj, dict) else None


def _remap_keys(sd: dict) -> dict:
    """Normalise legacy / DataParallel key prefixes."""
    remapped = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        nk = nk.replace("features.", "backbone.")
        nk = nk.replace("pool.", "temp_attn.")
        nk = nk.replace("classifier.", "head.")
        remapped[nk] = v
    return remapped


def _filter_shape_compatible(sd: dict, model: nn.Module) -> dict:
    """Keep only keys whose tensor shapes match the model's state-dict."""
    msd = model.state_dict()
    return {
        k: v
        for k, v in sd.items()
        if k in msd and getattr(v, "shape", None) == msd[k].shape
    }


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_video_model(
    checkpoint_path: str,
    meta_path: str,
    label_encoder_path: str,
    device: Optional[str] = None,
) -> tuple:
    """Load a trained VideoEmotionModel from disk.

    Args:
        checkpoint_path: Path to ``best_video_model.pt``.
        meta_path: Path to ``video_model_meta.json``.
        label_encoder_path: Path to ``video_label_encoder.pkl``.
        device: Target device string.  Defaults to CUDA if available.

    Returns:
        (model, label_encoder, meta)  where *model* is in eval mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(meta_path) as f:
        meta = json.load(f)

    with open(label_encoder_path, "rb") as f:
        le = pickle.load(f)

    num_classes = meta.get("num_classes", len(le.classes_))
    model = VideoEmotionModel(num_classes=num_classes).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    raw_sd = _extract_state_dict(ckpt)
    if raw_sd is None:
        raise RuntimeError("Checkpoint does not contain a state_dict.")

    sd_all = _remap_keys(raw_sd)
    sd_fit = _filter_shape_compatible(sd_all, model)
    load_result = model.load_state_dict(sd_fit, strict=False)
    model.eval()

    missing = getattr(load_result, "missing_keys", [])
    unexpected = getattr(load_result, "unexpected_keys", [])
    loaded = len(sd_fit) - len(unexpected)
    print(f"Video model loaded from {checkpoint_path}")
    print(f"  Architecture : R3D-18 + TemporalAttentionPool")
    print(f"  Loaded weights: {loaded} tensors  |  missing: {len(missing)}  |  unexpected: {len(unexpected)}")
    if missing:
        print("  Missing keys (truncated):", missing[:6], "..." if len(missing) > 6 else "")

    return model, le, meta

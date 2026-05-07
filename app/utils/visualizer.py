from __future__ import annotations

"""
visualizer.py
─────────────
Attention & explainability visualisations for MoodSyncAI.

  - Grad-CAM overlay  → highlights facial regions that drove the CNN prediction
  - Transformer token attention heatmap → highlights text tokens most influential
    in the sentiment/emotion prediction
"""

import io
import numpy as np
import streamlit as st
from PIL import Image

# ── Grad-CAM ───────────────────────────────────────────────────────────────────


def compute_gradcam(
    model,  # PyTorch CNN / ViT model (eval mode)
    image: Image.Image,  # Original PIL image (RGB)
    target_layer_name: str | None = None,
    target_class_idx: int | None = None,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a PyTorch CNN model.

    Args:
        model             – PyTorch model in eval mode with accessible named modules.
        image             – PIL RGB image (will be resized to 224×224 internally).
        target_layer_name – Name of the convolutional layer to hook (e.g. 'layer4').
                            Defaults to the last Conv2d layer found.
        target_class_idx  – Class index to explain. Defaults to the predicted class.

    Returns:
        heatmap – float32 ndarray of shape (H, W) normalised to [0, 1].
    """
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(image).unsqueeze(0)  # (1, C, H, W)
    input_tensor.requires_grad_(True)

    # ── Find target layer ─────────────────────────────────────────────────────
    target_layer = _find_target_layer(model, target_layer_name)

    # ── Register hooks ────────────────────────────────────────────────────────
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0].detach())

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # ── Forward + backward pass ───────────────────────────────────────────────
    model.eval()
    output = model(input_tensor)
    if target_class_idx is None:
        target_class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, target_class_idx].backward()

    fwd_handle.remove()
    bwd_handle.remove()

    # ── Compute CAM ───────────────────────────────────────────────────────────
    act = activations[0].squeeze(0)  # (C, h, w)
    grad = gradients[0].squeeze(0)  # (C, h, w)

    weights = grad.mean(dim=(1, 2))  # Global average pooling → (C,)
    cam = torch.einsum("c,chw->hw", weights, act)
    cam = F.relu(cam)

    # Resize to original image size
    orig_w, orig_h = image.size
    cam = (
        F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )

    # Normalise to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)

    return cam.astype(np.float32)


def overlay_gradcam(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """
    Blend a Grad-CAM heatmap onto the original image.

    Args:
        image    – Original PIL RGB image.
        heatmap  – Float32 array of shape (H, W) in [0, 1].
        alpha    – Opacity of the heatmap overlay (0 = invisible, 1 = full).
        colormap – Matplotlib colormap name (default 'jet').

    Returns:
        PIL Image with the CAM overlay composited on top.
    """
    import matplotlib.pyplot as plt

    # Apply colourmap to heatmap
    cmap = plt.get_cmap(colormap)
    hm_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)  # (H, W, 3)
    hm_pil = Image.fromarray(hm_rgb).resize(image.size, Image.BILINEAR)

    # Composite
    overlay = Image.blend(image.convert("RGB"), hm_pil, alpha=alpha)
    return overlay


def render_gradcam_section(
    image: Image.Image,
    heatmap: np.ndarray | None,
    label: str,
    score: float,
) -> None:
    """
    Streamlit component: renders the original image and its Grad-CAM overlay
    side by side, with a caption showing the predicted label and confidence.

    If heatmap is None (e.g. model does not support Grad-CAM), shows a
    user-friendly fallback message instead.
    """
    st.markdown(
        '<div class="section-label">🔍 Grad-CAM — Visual Attention</div>',
        unsafe_allow_html=True,
    )

    if heatmap is None:
        st.info(
            "Grad-CAM is not available for the current visual model "
            "(ViT pipeline models do not expose intermediate layers directly). "
            "Use a custom CNN loaded via `torch.load()` to enable this feature.",
            icon="ℹ️",
        )
        return

    overlay = overlay_gradcam(image, heatmap)

    col_orig, col_cam = st.columns(2, gap="small")
    with col_orig:
        st.image(image, caption="Original", use_column_width=True)
    with col_cam:
        st.image(
            overlay, caption=f"Grad-CAM → {label} ({score}%)", use_column_width=True
        )

    st.caption(
        "🟥 Red/warm areas = regions most influential for the predicted emotion. "
        "🟦 Blue/cool areas = low influence."
    )


# ── Transformer token attention ────────────────────────────────────────────────


def extract_token_attention(
    text: str,
    model_name: str = "j-hartmann/emotion-english-distilroberta-base",
    layer_idx: int = -1,
    head_idx: int | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Extract averaged self-attention weights from a HuggingFace Transformer model.

    Args:
        text       – Raw input text.
        model_name – HuggingFace model identifier.
        layer_idx  – Transformer layer to extract from (-1 = last layer).
        head_idx   – Attention head index to use. None = average all heads.

    Returns:
        tokens      – List of decoded token strings (subwords).
        attn_matrix – Float32 ndarray of shape (n_tokens, n_tokens).
    """
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
    attn = outputs.attentions[layer_idx].squeeze(0)  # (n_heads, seq, seq)

    if head_idx is not None:
        attn_matrix = attn[head_idx].numpy()
    else:
        attn_matrix = attn.mean(dim=0).numpy()  # average over heads

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    return tokens, attn_matrix.astype(np.float32)


def render_token_attention(
    tokens: list[str],
    attn_matrix: np.ndarray,
    top_k: int = 8,
) -> None:
    """
    Streamlit component: renders a token-level attention bar chart showing
    the most influential tokens in the text prediction.

    Displays the CLS-row attention (row 0), which represents how much each
    token contributed to the global [CLS] representation used for classification.

    Args:
        tokens      – List of subword token strings.
        attn_matrix – (n_tokens, n_tokens) attention weight matrix.
        top_k       – Maximum number of tokens to display.
    """
    st.markdown(
        '<div class="section-label">🔤 Transformer Token Attention</div>',
        unsafe_allow_html=True,
    )

    # CLS token attends to every other token — use that row as token importance
    cls_attn = attn_matrix[0]  # (n_tokens,)
    token_scores = list(zip(tokens, cls_attn))

    # Filter out special tokens
    token_scores = [
        (t, s)
        for t, s in token_scores
        if t not in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>")
    ]

    # Sort and take top-k
    token_scores.sort(key=lambda x: x[1], reverse=True)
    token_scores = token_scores[:top_k]

    # Normalise to percentage
    total = sum(s for _, s in token_scores) or 1.0
    token_scores = [(t, round(s / total * 100, 1)) for t, s in token_scores]

    for token, score in token_scores:
        clean_token = token.replace("▁", "").replace("Ġ", " ").strip() or token
        st.markdown(
            f"""
            <div style="margin-bottom:6px;">
                <div style="display:flex; justify-content:space-between;
                            font-size:13px; color:#aaa; margin-bottom:2px;">
                    <span style="font-family:monospace;">{clean_token}</span>
                    <span>{score}%</span>
                </div>
                <div style="background:#2a2a2a; border-radius:4px; height:8px;">
                    <div style="background:#4f98a3; width:{score}%; height:8px;
                                border-radius:4px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Bars show relative CLS-row attention weight — tokens the model focused on most."
    )


# ── Internal helpers ───────────────────────────────────────────────────────────


def _find_target_layer(model, layer_name: str | None):
    """Return the named module or fall back to the last Conv2d in the model."""
    import torch.nn as nn

    if layer_name is not None:
        return dict(model.named_modules())[layer_name]

    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError(
            "No Conv2d layer found in model. Specify target_layer_name manually."
        )
    return last_conv

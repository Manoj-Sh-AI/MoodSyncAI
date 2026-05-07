from __future__ import annotations

import streamlit as st

# Emotion → hex color mapping used across all bar charts
EMOTION_COLORS: dict[str, str] = {
    "happy":    "#4CAF50",
    "joy":      "#4CAF50",
    "sad":      "#5C6BC0",
    "sadness":  "#5C6BC0",
    "angry":    "#EF5350",
    "anger":    "#EF5350",
    "fear":     "#AB47BC",
    "disgust":  "#FF7043",
    "surprise": "#FFA726",
    "neutral":  "#78909C",
}


def render_results_panel(
    visual_result: dict,
    text_result: dict,
    fusion_result: dict,
) -> None:
    """
    Renders the three-column results panel:
      Left   – Visual emotion confidence bar chart
      Centre – Text sentiment confidence bar chart
      Right  – Fusion badge + polarity chips + fused confidence
    """
    st.markdown("### 📊 Results")

    col_vis, col_txt, col_fuse = st.columns(3, gap="medium")

    with col_vis:
        _render_modality_card(
            title="Visual Emotion (CNN / ViT)",
            top_label=visual_result["top_label"],
            top_score=visual_result["top_score"],
            all_scores=visual_result["all_scores"],
            icon="👁",
        )

    with col_txt:
        _render_modality_card(
            title="Textual Emotion (Transformer)",
            top_label=text_result["top_label"],
            top_score=text_result["top_score"],
            all_scores=text_result["all_scores"],
            icon="💬",
        )

    with col_fuse:
        _render_fusion_card(visual_result, text_result, fusion_result)


# ── Private helpers ────────────────────────────────────────────────────────────

def _render_modality_card(
    title: str,
    top_label: str,
    top_score: float,
    all_scores: dict[str, float],
    icon: str,
) -> None:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)

    color = EMOTION_COLORS.get(top_label.lower(), "#90A4AE")
    st.markdown(
        f"<h3 style='margin:4px 0; color:{color};'>{icon} {top_label.capitalize()}</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span style='color:#888; font-size:13px;'>Confidence: {top_score}%</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    _render_confidence_bars(all_scores)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_fusion_card(
    visual_result: dict,
    text_result: dict,
    fusion_result: dict,
) -> None:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Fusion Result</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Mismatch / aligned badge
    _render_mismatch_badge(fusion_result["mismatch"])
    st.markdown("<br>", unsafe_allow_html=True)

    # Per-modality polarity chips
    st.markdown(
        f'<span class="metric-chip">👁 {visual_result["top_label"].capitalize()} '
        f'— <em>{fusion_result["visual_polarity"]}</em></span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="metric-chip">💬 {text_result["top_label"].capitalize()} '
        f'— <em>{fusion_result["text_polarity"]}</em></span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="metric-chip">🔗 Fused confidence: '
        f'{fusion_result["fused_confidence"]}%</span>',
        unsafe_allow_html=True,
    )

    # Dominant emotion
    st.markdown(
        f'<span class="metric-chip">🎯 Dominant: '
        f'{fusion_result["dominant_emotion"].capitalize()}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_confidence_bars(scores: dict[str, float]) -> None:
    """Render animated horizontal bars for each emotion score."""
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for label, score in sorted_scores:
        color = EMOTION_COLORS.get(label.lower(), "#90A4AE")
        st.markdown(
            f"""
            <div style="margin-bottom:7px;">
                <div style="display:flex; justify-content:space-between;
                            font-size:12px; color:#aaa; margin-bottom:2px;">
                    <span>{label.capitalize()}</span>
                    <span>{score}%</span>
                </div>
                <div style="background:#2a2a2a; border-radius:4px; height:9px; width:100%;">
                    <div style="background:{color}; width:{score}%; height:9px;
                                border-radius:4px; transition:width 0.5s ease;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_mismatch_badge(mismatch: bool) -> None:
    if mismatch:
        st.markdown(
            """
            <div style="background:#FF8F00; color:#fff; padding:8px 18px;
                        border-radius:20px; display:inline-block; font-weight:700;
                        font-size:14px; letter-spacing:0.5px;">
                ⚠️ MISMATCH DETECTED
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background:#2E7D32; color:#fff; padding:8px 18px;
                        border-radius:20px; display:inline-block; font-weight:700;
                        font-size:14px; letter-spacing:0.5px;">
                ✅ SIGNALS ALIGNED
            </div>
            """,
            unsafe_allow_html=True,
        )

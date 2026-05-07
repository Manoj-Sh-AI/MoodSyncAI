from __future__ import annotations

"""
chart_builder.py
────────────────
Plotly-based chart helpers for MoodSyncAI.

  - Emotion confidence bar chart  (single modality)
  - Modality comparison bar chart (visual vs. text side-by-side)
  - Fusion donut chart            (polarity breakdown)
  - Emotion timeline              (optional: webcam / video mode)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Shared colour palette ──────────────────────────────────────────────────────
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

_PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cccccc", family="Inter, sans-serif", size=12),
    margin=dict(l=0, r=0, t=30, b=0),
)


# ── Single-modality confidence bar chart ──────────────────────────────────────

def build_confidence_bar_chart(
    scores: dict[str, float],
    title: str = "Emotion Confidence",
    orientation: str = "h",
) -> go.Figure:
    """
    Build a Plotly horizontal (or vertical) bar chart for emotion confidence scores.

    Args:
        scores      – {emotion_label: confidence_percentage} dict.
        title       – Chart title string.
        orientation – 'h' (horizontal, default) or 'v' (vertical).

    Returns:
        Plotly Figure object — pass to st.plotly_chart().
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    labels = [k.capitalize() for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = [EMOTION_COLORS.get(k.lower(), "#90A4AE") for k, _ in sorted_items]

    if orientation == "h":
        fig = go.Figure(go.Bar(
            x=values, y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v}%" for v in values],
            textposition="outside",
            cliponaxis=False,
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=13)),
            xaxis=dict(range=[0, 110], showgrid=False, showticklabels=False,
                       zeroline=False),
            yaxis=dict(showgrid=False),
            height=220,
            **_PLOTLY_LAYOUT_DEFAULTS,
        )
    else:
        fig = go.Figure(go.Bar(
            x=labels, y=values,
            orientation="v",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v}%" for v in values],
            textposition="outside",
            cliponaxis=False,
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=13)),
            yaxis=dict(range=[0, 115], showgrid=False, showticklabels=False,
                       zeroline=False),
            xaxis=dict(showgrid=False),
            height=260,
            **_PLOTLY_LAYOUT_DEFAULTS,
        )

    return fig


# ── Side-by-side modality comparison ──────────────────────────────────────────

def build_modality_comparison_chart(
    visual_scores: dict[str, float],
    text_scores:   dict[str, float],
) -> go.Figure:
    """
    Build a grouped bar chart comparing visual vs. text emotion scores
    for the union of emotion labels present in both modalities.

    Args:
        visual_scores – {emotion_label: confidence_%} from visual model.
        text_scores   – {emotion_label: confidence_%} from text model.

    Returns:
        Plotly Figure.
    """
    # Align on the union of labels
    all_labels = sorted(set(visual_scores) | set(text_scores))
    vis_vals   = [visual_scores.get(l, 0.0) for l in all_labels]
    txt_vals   = [text_scores.get(l, 0.0)   for l in all_labels]
    cap_labels = [l.capitalize() for l in all_labels]

    fig = go.Figure([
        go.Bar(
            name="👁 Visual",
            x=cap_labels,
            y=vis_vals,
            marker_color="#4f98a3",
            text=[f"{v}%" for v in vis_vals],
            textposition="outside",
        ),
        go.Bar(
            name="💬 Text",
            x=cap_labels,
            y=txt_vals,
            marker_color="#FFA726",
            text=[f"{v}%" for v in txt_vals],
            textposition="outside",
        ),
    ])

    fig.update_layout(
        barmode="group",
        title=dict(text="Visual vs. Text Emotion Scores", font=dict(size=13)),
        yaxis=dict(range=[0, 120], showgrid=False, showticklabels=False,
                   zeroline=False),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        height=300,
        **_PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


# ── Fusion polarity donut ──────────────────────────────────────────────────────

def build_fusion_donut(fusion_result: dict) -> go.Figure:
    """
    Build a donut chart showing the weighted polarity breakdown from fusion.

    The two slices represent the visual and text contribution to the
    fused confidence score.

    Args:
        fusion_result – dict returned by models/fusion/fusion_layer.fuse().

    Returns:
        Plotly Figure.
    """
    labels = ["👁 Visual contribution", "💬 Text contribution"]

    # Derive contributions from the stored confidence values
    vis_conf  = fusion_result.get("visual_confidence",  50.0)
    txt_conf  = fusion_result.get("text_confidence",    50.0)
    values    = [vis_conf, txt_conf]

    mismatch  = fusion_result.get("mismatch", False)
    ring_color = "#FF8F00" if mismatch else "#4f98a3"

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.60,
        marker=dict(
            colors=["#4f98a3", "#FFA726"],
            line=dict(color="#111", width=2),
        ),
        textinfo="label+percent",
        textfont=dict(size=11),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))

    status_text = "MISMATCH" if mismatch else "ALIGNED"
    fig.add_annotation(
        text=f"<b>{status_text}</b>",
        x=0.5, y=0.5,
        font=dict(size=14, color=ring_color),
        showarrow=False,
    )

    fig.update_layout(
        title=dict(text="Fusion Polarity Breakdown", font=dict(size=13)),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1, x=0.1, bgcolor="rgba(0,0,0,0)"),
        height=280,
        **_PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


# ── Emotion timeline (optional — webcam / video mode) ─────────────────────────

def build_emotion_timeline(
    timestamps: list[float],
    emotion_series: dict[str, list[float]],
    title: str = "Emotion Timeline",
) -> go.Figure:
    """
    Build a multi-line area chart showing how emotion confidence evolves
    over time (used in webcam / video optional feature).

    Args:
        timestamps     – List of time values in seconds.
        emotion_series – {emotion_label: [confidence_at_t0, t1, ...]} dict.
        title          – Chart title string.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    for emotion, values in emotion_series.items():
        if len(values) != len(timestamps):
            continue
        color = EMOTION_COLORS.get(emotion.lower(), "#90A4AE")
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode="lines",
            name=emotion.capitalize(),
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=color.replace(")", ", 0.08)").replace("rgb(", "rgba("),
            hovertemplate=f"{emotion.capitalize()}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="Time (s)", showgrid=False),
        yaxis=dict(title="Confidence %", range=[0, 105], showgrid=False,
                   zeroline=False),
        legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        height=300,
        **_PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


# ── Streamlit render wrappers ─────────────────────────────────────────────────

def render_confidence_chart(scores: dict[str, float], title: str = "") -> None:
    """Render a single-modality confidence bar chart directly in Streamlit."""
    fig = build_confidence_bar_chart(scores, title=title)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_comparison_chart(
    visual_scores: dict[str, float],
    text_scores:   dict[str, float],
) -> None:
    """Render the side-by-side modality comparison chart in Streamlit."""
    fig = build_modality_comparison_chart(visual_scores, text_scores)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_fusion_donut(fusion_result: dict) -> None:
    """Render the fusion polarity donut chart in Streamlit."""
    fig = build_fusion_donut(fusion_result)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_timeline(
    timestamps: list[float],
    emotion_series: dict[str, list[float]],
) -> None:
    """Render the emotion timeline chart in Streamlit (video/webcam mode)."""
    fig = build_emotion_timeline(timestamps, emotion_series)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

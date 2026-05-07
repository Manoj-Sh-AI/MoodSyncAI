from __future__ import annotations

import streamlit as st


def render_generative_summary(
    visual_result: dict,
    text_result: dict,
    fusion_result: dict,
    summary_text: str,
) -> None:
    """
    Renders the generative language model output section.

    Displays the summary text in a styled callout box, with a collapsible
    debug panel showing raw model inputs for transparency.

    Args:
        visual_result  – dict from run_visual_inference()
        text_result    – dict from run_text_inference()
        fusion_result  – dict from fuse()
        summary_text   – natural language string from generate_summary()
    """
    st.markdown("---")
    st.markdown(
        '<div class="section-label" style="margin-bottom:8px;">'
        "🤖 Generative Summary (Language Model)"
        "</div>",
        unsafe_allow_html=True,
    )

    # Determine callout accent colour based on mismatch state
    accent = "#FF8F00" if fusion_result["mismatch"] else "#4f98a3"
    border_label = "⚠️ Incongruence detected" if fusion_result["mismatch"] else "✅ Consistent emotional state"

    st.markdown(
        f"""
        <div style="
            background:#1a1a2e;
            border-left: 4px solid {accent};
            border-radius: 0 10px 10px 0;
            padding: 16px 20px 16px 20px;
            margin-top: 4px;
        ">
            <div style="font-size:11px; font-weight:700; letter-spacing:1px;
                        text-transform:uppercase; color:{accent}; margin-bottom:8px;">
                {border_label}
            </div>
            <div style="font-size:15px; line-height:1.75; color:#ccc;">
                {summary_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Confidence snapshot beneath the summary ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    snap_cols = st.columns(4)
    _metric(snap_cols[0], "👁 Visual",   visual_result["top_label"].capitalize(),  f"{visual_result['top_score']}%")
    _metric(snap_cols[1], "💬 Text",     text_result["top_label"].capitalize(),    f"{text_result['top_score']}%")
    _metric(snap_cols[2], "🔗 Fused",   fusion_result["dominant_emotion"].capitalize(), f"{fusion_result['fused_confidence']}%")
    _metric(snap_cols[3], "📶 Status",  "Mismatch" if fusion_result["mismatch"] else "Aligned", "")

    # ── Raw debug expander ────────────────────────────────────────────────────
    with st.expander("🔬 Raw model output (debug / demo)"):
        st.json(
            {
                "visual_result":  visual_result,
                "text_result":    text_result,
                "fusion_result":  fusion_result,
            }
        )


def _metric(col, label: str, value: str, sub: str) -> None:
    """Renders a compact metric chip inside a given Streamlit column."""
    with col:
        st.markdown(
            f"""
            <div style="
                background:#1c1c1c;
                border:1px solid #2a2a2a;
                border-radius:10px;
                padding:12px 14px;
                text-align:center;
            ">
                <div style="font-size:11px; color:#666; margin-bottom:4px;">{label}</div>
                <div style="font-size:15px; font-weight:700; color:#e0e0e0;">{value}</div>
                <div style="font-size:12px; color:#888;">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

from __future__ import annotations

import streamlit as st


# Example prompts shown as quick-fill buttons
_EXAMPLE_PROMPTS: list[str] = [
    "No, I think the project is going really well.",
    "Everything is fine, don't worry about me.",
    "I am so excited about this opportunity!",
    "I'm not sure I can handle this right now.",
]


def render_text_input() -> tuple[str, float]:
    """
    Renders the text input widget and fusion weight slider.

    Returns:
        text_input   – the user's typed or pasted text (may be empty string).
        visual_weight – float in [0.0, 1.0] representing the visual modality weight.
    """
    st.markdown(
        '<div class="section-label">Text Input — What They Said</div>',
        unsafe_allow_html=True,
    )

    # Quick-fill example buttons
    st.markdown(
        '<div style="font-size:12px; color:#666; margin-bottom:6px;">'
        "Try an example →</div>",
        unsafe_allow_html=True,
    )
    example_cols = st.columns(len(_EXAMPLE_PROMPTS))
    selected_example = ""
    for col, prompt in zip(example_cols, _EXAMPLE_PROMPTS):
        with col:
            short = prompt[:22] + "…" if len(prompt) > 22 else prompt
            if st.button(short, key=f"example_{prompt[:10]}", use_container_width=True):
                selected_example = prompt

    # Main text area — pre-fill with selected example if clicked
    default_text = selected_example if selected_example else st.session_state.get("text_value", "")
    text_input = st.text_area(
        label="Enter speech or typed text",
        value=default_text,
        placeholder="e.g. No, I think the project is going really well.",
        height=115,
        label_visibility="collapsed",
        key="text_input_area",
    )

    # Live character / word count
    word_count = len(text_input.split()) if text_input.strip() else 0
    char_count = len(text_input)
    st.markdown(
        f'<div style="font-size:11px; color:#555; text-align:right; margin-top:2px;">'
        f"{word_count} words · {char_count} chars</div>",
        unsafe_allow_html=True,
    )

    # ── Fusion weight slider ──────────────────────────────────────────────────
    st.markdown(
        '<div class="section-label" style="margin-top:18px;">Fusion Weight</div>',
        unsafe_allow_html=True,
    )
    visual_weight: float = st.slider(
        label="Visual modality weight",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
        label_visibility="collapsed",
        help=(
            "Proportion of the final fused score contributed by the facial signal. "
            "The text signal receives the remainder (1 − visual weight)."
        ),
    )
    st.markdown(
        f'<span class="metric-chip">👁 Visual: {int(visual_weight * 100)}%</span>'
        f'<span class="metric-chip">💬 Text: {int((1 - visual_weight) * 100)}%</span>',
        unsafe_allow_html=True,
    )

    return text_input, visual_weight

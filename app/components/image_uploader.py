from __future__ import annotations

import streamlit as st
from PIL import Image


def render_image_uploader() -> Image.Image | None:
    """
    Renders the face image upload widget.

    Returns the uploaded PIL Image (RGB) or None if no file has been uploaded.
    """
    st.markdown(
        '<div class="section-label">Visual Input — Face Image</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        label="Upload a face photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="face_image_uploader",
    )

    if uploaded_file is None:
        _render_empty_state()
        return None

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True, caption="Uploaded face")

    # Show basic image metadata below the preview
    w, h = image.size
    st.markdown(
        f'<span class="metric-chip">📐 {w} × {h} px</span>'
        f'<span class="metric-chip">🖼 {uploaded_file.type}</span>',
        unsafe_allow_html=True,
    )

    return image


def _render_empty_state() -> None:
    """Placeholder shown before any image is uploaded."""
    st.markdown(
        """
        <div style="
            border: 2px dashed #2a2a2a;
            border-radius: 12px;
            padding: 40px 20px;
            text-align: center;
            color: #555;
            font-size: 14px;
        ">
            <div style="font-size: 36px; margin-bottom: 10px;">📷</div>
            <div>Upload a face image to begin</div>
            <div style="font-size: 12px; margin-top: 6px;">JPG · PNG · WEBP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

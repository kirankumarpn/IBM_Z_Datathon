# app.py - Streamlit pitch deck builder
"""
A 100% Python web application using Streamlit that lets the user
* navigate through slides,
* edit slide titles and bullet‑point content,
* preview the slide with a dark cyber‑security theme, and
* export the whole deck as PPTX or PDF.
The heavy lifting (PPTX / PDF generation) is delegated to the backend
modules we created earlier.
"""

import streamlit as st
from typing import List, Dict

# Backend imports
from backend.default_data import get_default_presentation
from backend.pptx_generator import generate_pitch_deck_pptx
from backend.pdf_generator import generate_pitch_deck_pdf

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _init_presentation() -> Dict:
    """Load the default presentation on first run and store it in session state."""
    if "presentation" not in st.session_state:
        st.session_state.presentation = get_default_presentation()
        st.session_state.current_idx = 0
    return st.session_state.presentation


def _render_slide(slide: Dict) -> None:
    """Render the currently selected slide using HTML + CSS for a premium look."""
    # Custom CSS – dark background, cyan headers, white body, subtle shadows
    st.markdown(
        """
        <style>
        .slide-container {
            background: #0a143c; /* NAVY */
            color: #ffffff;    /* WHITE */
            border-radius: 12px;
            padding: 2rem;
            margin-top: 1rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
            backdrop-filter: blur(6px);
        }
        .slide-title {
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            color: #00c8ff; /* CYAN */
            margin-bottom: 1rem;
        }
        .slide-bullet {
            font-family: 'Inter', sans-serif;
            font-size: 1.4rem;
            margin-left: 1.5rem;
            position: relative;
        }
        .slide-bullet::before {
            content: "\2022"; /* bullet */
            position: absolute;
            left: -1.5rem;
            color: #00c8ff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Build HTML for slide content
    title = slide.get("title", "")
    content = slide.get("content", [])
    html = f"<div class='slide-container'>"
    html += f"<div class='slide-title'>{title}</div>"
    for line in content:
        html += f"<div class='slide-bullet'>{line}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _save_current_slide(title: str, content_text: str) -> None:
    """Persist edits made in the UI back to the session‑state presentation."""
    idx = st.session_state.current_idx
    # Split multiline text area into a list of strings (ignore empty lines)
    content_list = [ln.strip() for ln in content_text.split("\n") if ln.strip()]
    st.session_state.presentation["slides"][idx]["title"] = title
    st.session_state.presentation["slides"][idx]["content"] = content_list

# ---------------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Botnet Detection Pitch Deck Builder", layout="wide")
presentation = _init_presentation()
slides = presentation["slides"]

# Sidebar – slide navigation & management
with st.sidebar:
    st.header("Slides")
    slide_titles = [s["title"] for s in slides]
    selected = st.radio("Select slide", options=range(len(slides)), format_func=lambda i: slide_titles[i])
    st.session_state.current_idx = selected

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add slide"):
            new_slide = {"title": "New Slide", "content": []}
            slides.append(new_slide)
            st.session_state.current_idx = len(slides) - 1
    with col2:
        if st.button("Delete slide") and len(slides) > 1:
            del slides[st.session_state.current_idx]
            st.session_state.current_idx = max(0, st.session_state.current_idx - 1)

    st.markdown("---")
    if st.button("Export PPTX"):
        pptx_bytes = generate_pitch_deck_pptx(presentation)
        st.download_button(
            label="Download PPTX",
            data=pptx_bytes,
            file_name="pitch_deck.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    if st.button("Export PDF"):
        pdf_bytes = generate_pitch_deck_pdf(presentation)
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="pitch_deck.pdf",
            mime="application/pdf",
        )

# Main area – edit & preview the selected slide
current_slide = slides[st.session_state.current_idx]
st.subheader("Edit Slide")
new_title = st.text_input("Slide title", value=current_slide.get("title", ""), key="title_input")
content_text = "\n".join(current_slide.get("content", []))
new_content = st.text_area("Bullet points (one per line)", value=content_text, height=200, key="content_input")
if st.button("Save changes"):
    _save_current_slide(new_title, new_content)
    st.success("Slide saved!")

st.subheader("Preview")
_render_slide(slides[st.session_state.current_idx])

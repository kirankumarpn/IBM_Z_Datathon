# pdf_generator.py - PDF generator for the pitch deck
"""
Generates a 16:9 landscape PDF file from the presentation dict defined in backend/default_data.py.
Implements the same dark cyber theme as the PowerPoint generator.
"""

import io
from typing import Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, inch
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Frame

# Theme colours (converted to RGB fractions for reportlab)
NAVY = colors.Color(10/255, 20/255, 60/255)   # background
CYAN = colors.Color(0/255, 200/255, 255/255)  # title accent
WHITE = colors.white

PAGE_WIDTH = 13.333 * inch  # 16:9 width
PAGE_HEIGHT = 7.5 * inch    # 16:9 height

from reportlab.lib.styles import ParagraphStyle
styles = getSampleStyleSheet()
# Define a custom title style based on the existing Title style
TitleStyle = ParagraphStyle(
    name='TitleStyle',
    parent=styles['Title'],
    fontSize=36,
    textColor=CYAN,
)
styles.add(TitleStyle)
# Adjust the normal style for slide content
styles['Normal'].fontSize = 24
styles['Normal'].textColor = WHITE

def _draw_background(canv):
    canv.saveState()
    canv.setFillColor(NAVY)
    canv.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
    canv.restoreState()

def _add_title_page(canv, title: str, subtitle: str):
    _draw_background(canv)
    canv.setFillColor(CYAN)
    canv.setFont("Helvetica-Bold", 48)
    canv.drawCentredString(PAGE_WIDTH/2, PAGE_HEIGHT*0.65, title)
    canv.setFillColor(WHITE)
    canv.setFont("Helvetica", 28)
    canv.drawCentredString(PAGE_WIDTH/2, PAGE_HEIGHT*0.5, subtitle)

def _add_content_page(canv, title: str, content: List[str]):
    _draw_background(canv)
    # Title
    canv.setFillColor(CYAN)
    canv.setFont("Helvetica-Bold", 36)
    canv.drawString(inch, PAGE_HEIGHT - inch*1.2, title)
    # Content bullets
    canv.setFillColor(WHITE)
    canv.setFont("Helvetica", 24)
    y = PAGE_HEIGHT - inch*2.0
    bullet = u"\u2022"
    for line in content:
        canv.drawString(inch*1.2, y, f"{bullet} {line}")
        y -= inch*0.4

def generate_pitch_deck_pdf(presentation_data: Dict) -> bytes:
    """Generate a PDF deck from the presentation dict and return bytes."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape((PAGE_WIDTH, PAGE_HEIGHT)))

    meta = presentation_data.get("metadata", {})
    # Title slide
    _add_title_page(c, meta.get("project_name", "Presentation"), meta.get("subtitle", ""))
    c.showPage()

    for slide in presentation_data.get("slides", []):
        _add_content_page(c, slide.get("title", ""), slide.get("content", []))
        c.showPage()

    c.save()
    buf.seek(0)
    return buf.getvalue()

if __name__ == "__main__":
    from backend.default_data import get_default_presentation
    data = get_default_presentation()
    pdf_bytes = generate_pitch_deck_pdf(data)
    with open("test_deck.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("Test PDF generated: test_deck.pdf")

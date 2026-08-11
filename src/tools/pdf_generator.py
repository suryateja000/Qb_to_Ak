"""
Answer key PDF generation tool.
"""
import tempfile
from fpdf import FPDF
from src.utils.helpers import sanitize_for_pdf


def create_answer_key_pdf(qna_pairs: list[dict]) -> str:
    """Generate a formatted PDF answer key from question-answer pairs.
    
    Returns the path to the generated PDF file.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Answer Key", 0, 1, 'C')
    pdf.ln(5)

    for i, pair in enumerate(qna_pairs):
        pdf.set_font("Arial", 'B', 12)
        pdf.multi_cell(0, 10, sanitize_for_pdf(f"Question {i+1}: {pair['question']}"))
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, sanitize_for_pdf(f"Answer: {pair['answer']}"))
        pdf.ln(5)
        if i < len(qna_pairs) - 1:
            pdf.line(pdf.get_x(), pdf.get_y(),
                     pdf.get_x() + pdf.w - 2 * pdf.l_margin, pdf.get_y())
            pdf.ln(5)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf_output_path = tmp_pdf.name
    pdf.output(pdf_output_path, "F")
    return pdf_output_path

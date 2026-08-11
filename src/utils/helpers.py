"""
Helper / utility functions.
"""


def sanitize_for_pdf(text: str) -> str:
    """Replace unicode characters that FPDF/Arial cannot encode."""
    replacements = {
        '\u2018': "'", '\u2019': "'",   # curly single quotes
        '\u201c': '"', '\u201d': '"',   # curly double quotes
        '\u2013': '-', '\u2014': '-',   # en-dash, em-dash
        '\u2026': '...', '\u2022': '*', # ellipsis, bullet
        '\u00b7': '*',                  # middle dot
        '\u2192': '->', '\u2190': '<-', # arrows
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Fallback: encode to latin-1, replacing anything still unsupported
    return text.encode('latin-1', errors='replace').decode('latin-1')

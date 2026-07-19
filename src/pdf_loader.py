"""
PDF Document Loader

Extracts text from PDF files using PyMuPDF (fitz).
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict


def load_pdf(file_path: str) -> Dict:
    """
    Load a PDF file and extract all text.

    Args:
        file_path: Path to the PDF file

    Returns:
        Dictionary with title, text, and metadata
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if not file_path.suffix.lower() == '.pdf':
        raise ValueError(f"File is not a PDF: {file_path}")

    try:
        doc = fitz.open(file_path)
        text_content = []
        page_count = len(doc)

        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                text_content.append(f"--- Page {page_num} ---\n{text}")

        full_text = "\n".join(text_content)
        doc.close()

        return {
            "title": file_path.name,
            "text": full_text,
            "metadata": {
                "source": str(file_path),
                "type": "pdf",
                "filename": file_path.name,
                "pages": page_count,
                "empty_text": not full_text.strip(),
            }
        }

    except Exception as e:
        raise RuntimeError(f"Error reading PDF {file_path}: {str(e)}")

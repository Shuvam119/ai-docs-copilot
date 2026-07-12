"""
DOCX Document Loader

Extracts text from Word documents (.docx) using python-docx.
"""

from docx import Document
from pathlib import Path
from typing import Dict


def load_docx(file_path: str) -> Dict:
    """
    Load a DOCX file and extract all text.

    Args:
        file_path: Path to the DOCX file

    Returns:
        Dictionary with title, text, and metadata
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    if not file_path.suffix.lower() == '.docx':
        raise ValueError(f"File is not a DOCX: {file_path}")

    try:
        doc = Document(file_path)
        paragraphs = []

        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)

        full_text = "\n".join(paragraphs)

        return {
            "title": file_path.name,
            "text": full_text,
            "metadata": {
                "source": str(file_path),
                "type": "docx",
                "filename": file_path.name,
                "paragraphs": len(paragraphs)
            }
        }

    except Exception as e:
        raise RuntimeError(f"Error reading DOCX {file_path}: {str(e)}")

"""
Document Loader Dispatcher

Routes files to the appropriate loader based on file extension.
"""

from pathlib import Path
from typing import Dict
from src.pdf_loader import load_pdf
from src.docx_loader import load_docx


def load_document(file_path: str) -> Dict:
    """
    Load any supported document type.

    Automatically detects file type and routes to appropriate loader.

    Args:
        file_path: Path to the document

    Returns:
        Dictionary with title, text, and metadata

    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        return load_pdf(str(file_path))
    elif suffix == '.docx':
        return load_docx(str(file_path))
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported types: .pdf, .docx"
        )


def load_documents_from_directory(directory_path: str) -> list:
    """
    Load all supported documents from a directory.

    Args:
        directory_path: Path to directory containing documents

    Returns:
        List of loaded documents
    """
    directory = Path(directory_path)

    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    supported_extensions = {'.pdf', '.docx'}
    documents = []

    for file_path in directory.iterdir():
        if file_path.suffix.lower() in supported_extensions:
            try:
                doc = load_document(str(file_path))
                documents.append(doc)
            except Exception as e:
                print(f"⚠️  Error loading {file_path.name}: {str(e)}")
                continue

    return documents

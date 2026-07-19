"""
Document Loader Dispatcher

Routes files to the appropriate loader based on file extension.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from src.docx_loader import load_docx
from src.pdf_loader import load_pdf
from src.config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Result of loading documents from a directory."""

    documents: List[Dict] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    empty_files: List[str] = field(default_factory=list)


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
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path_obj}")

    suffix = file_path_obj.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(str(file_path_obj))
    if suffix == ".docx":
        return load_docx(str(file_path_obj))

    raise ValueError(
        f"Unsupported file type: {suffix}. Supported types: .pdf, .docx"
    )


def load_documents_from_directory(directory_path: str) -> LoadResult:
    """
    Load all supported documents from a directory.

    Args:
        directory_path: Path to directory containing documents

    Returns:
        LoadResult with loaded documents, failures, and empty-file warnings
    """
    directory = Path(directory_path)

    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    result = LoadResult()

    for file_path in sorted(directory.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            doc = load_document(str(file_path))
            if doc["metadata"].get("empty_text"):
                result.empty_files.append(file_path.name)
            result.documents.append(doc)
            logger.info("Loaded document: %s", file_path.name)
        except Exception as exc:
            message = f"{file_path.name}: {exc}"
            result.failed_files.append(message)
            logger.error("Error loading %s: %s", file_path.name, exc)

    return result

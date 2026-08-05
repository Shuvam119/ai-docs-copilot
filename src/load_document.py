"""
Document Loader Dispatcher

Routes files to the appropriate loader based on file extension.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.docx_loader import load_docx
from src.pdf_loader import load_pdf
from src.config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

LOADER_MAPPING = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    # future: add ".md": load_markdown
}

# --- Parsed-text cache ------------------------------------------------------
# Extracting PDF/DOCX text is the most expensive part of indexing. Documents
# are cached by (resolved path, size, mtime) so unchanged files are never
# re-parsed across incremental builds. The cache is keyed on the file's
# stat signature, so a changed file automatically misses and re-parses.
_TEXT_CACHE: Dict[tuple, Dict] = {}
_TEXT_CACHE_MAX = 64


def _cache_key(file_path: Path) -> Optional[tuple]:
    try:
        stat = file_path.stat()
    except OSError:
        return None
    return (str(file_path.resolve()), stat.st_size, stat.st_mtime_ns)


def invalidate_document_cache(filename: Optional[str] = None) -> None:
    """Drop cached parsed text after upload, delete, or reindex.

    Args:
        filename: When given, only that document is evicted; otherwise the
            whole cache is cleared.
    """
    if filename is None:
        _TEXT_CACHE.clear()
        return
    for key in [key for key in _TEXT_CACHE
                if Path(key[0]).name == filename]:
        _TEXT_CACHE.pop(key, None)


def _copy_document(document: Dict) -> Dict:
    """Return a shallow copy so callers may mutate metadata safely."""
    return {
        **document,
        "metadata": dict(document.get("metadata") or {}),
    }


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

    loader = LOADER_MAPPING.get(suffix)
    if loader:
        key = _cache_key(file_path_obj)
        if key is not None:
            cached = _TEXT_CACHE.get(key)
            if cached is not None:
                return _copy_document(cached)
        document = loader(str(file_path_obj))
        if key is not None:
            _TEXT_CACHE[key] = document
            if len(_TEXT_CACHE) > _TEXT_CACHE_MAX:
                for stale in list(_TEXT_CACHE)[:_TEXT_CACHE_MAX // 2]:
                    _TEXT_CACHE.pop(stale, None)
        return _copy_document(document)

    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(
        f"Unsupported file type: {suffix}. Supported types: {supported}"
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

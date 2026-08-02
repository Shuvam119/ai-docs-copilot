"""Enterprise document metadata extraction and normalization."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DOCUMENT_TYPES = (
    "User Guide", "Administrator Guide", "SOP", "Job Aid", "FAQ",
    "Release Notes", "Known Issues", "API Documentation", "Training Manual",
    "Troubleshooting Guide",
)


def _title_from_filename(filename: str) -> str:
    """Create a readable title without a detected version suffix."""
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    return re.sub(r"\b(v(?:ersion)?\s*\d+(?:\.\d+)*)\b", "", stem, flags=re.I).strip().title()


def _first_match(pattern: str, text: str, default: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else default


def _keywords(text: str, limit: int = 10) -> List[str]:
    """Extract useful non-trivial terms without an additional model dependency."""
    stop_words = {"this", "that", "with", "from", "your", "will", "have", "into", "using",
                  "when", "where", "document", "guide", "version", "the", "and", "for", "are", "you"}
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text.lower())
    unique: List[str] = []
    for word in words:
        if word not in stop_words and word not in unique:
            unique.append(word)
        if len(unique) == limit:
            break
    return unique


LIFECYCLE_STATUSES = (
    "Fresh", "Aging", "Stale", "Archived", "Needs Review",
)


def _parse_iso_date(value: str) -> Optional[date]:
    try:
        return date.fromisoformat(value)
    except Exception:
        return None


def determine_lifecycle_status(metadata: Dict[str, Any]) -> str:
    """Assign a lifecycle classification using metadata signals."""
    explicit = metadata.get("lifecycle_status")
    if explicit in LIFECYCLE_STATUSES:
        return explicit

    version = str(metadata.get("version", "Unspecified")).strip().lower()
    if version == "unspecified":
        return "Fresh"

    title = str(metadata.get("title", "")).lower()
    document_type = str(metadata.get("document_type", "")).lower()
    if "archive" in title or "archive" in document_type or "legacy" in title:
        return "Archived"

    last_updated = _parse_iso_date(metadata.get("last_updated", ""))
    publication_date = _parse_iso_date(metadata.get("publication_date", ""))
    reference_date = last_updated or publication_date or date.today()
    days_old = (date.today() - reference_date).days

    if days_old <= 60:
        return "Fresh"
    if days_old <= 180:
        return "Aging"
    if days_old <= 365:
        return "Needs Review"
    return "Stale"


def parse_version_tuple(version: str) -> tuple[int, ...] | None:
    """Parse version strings like 'v1.2' or '2.0' into a numeric tuple."""
    match = re.search(r"(\d+(?:\.\d+)*)", str(version))
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split('.'))


def extract_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a consistent enterprise metadata record from file name and content."""
    source = document["metadata"]
    text = document.get("text", "")
    filename = source["filename"]
    haystack = f"{filename}\n{text[:5000]}"
    lowered = haystack.lower()

    document_type = next(
        (kind for kind in DOCUMENT_TYPES if kind.lower() in lowered), "User Guide")
    if "troubleshoot" in lowered:
        document_type = "Troubleshooting Guide"
    elif "release note" in lowered:
        document_type = "Release Notes"
    elif "known issue" in lowered:
        document_type = "Known Issues"
    elif "api" in lowered:
        document_type = "API Documentation"

    audience = "End User"
    if "administrator" in lowered or "admin guide" in lowered:
        audience = "Administrator"
    elif "support engineer" in lowered or "troubleshooting" in lowered:
        audience = "Support Engineer"
    elif "technical writer" in lowered:
        audience = "Technical Writer"
    elif "product manager" in lowered:
        audience = "Product Manager"

    version = _first_match(
        r"\b(?:version|ver|v)\s*(\d+(?:\.\d+)*)\b", haystack, "Unspecified")
    product = _first_match(
        r"(?:product|application|platform)\s*[:\-]\s*([^\n]{2,60})", text, "General")
    department = _first_match(
        r"(?:department|owner|team)\s*[:\-]\s*([^\n]{2,60})", text, "Documentation")
    author = _first_match(
        r"(?:author|written by|created by)\s*[:\-]\s*([^\n]{2,80})", text, "Unknown")
    publication_date = _first_match(
        r"(?:published|publication|released)(?:\s+on)?\s*[:\-]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
        text,
        "",
    )
    summary_source = " ".join(text.split())
    summary = summary_source[:360].rsplit(" ", 1)[0] if len(
        summary_source) > 360 else summary_source

    metadata = {
        "title": _title_from_filename(filename), "product": product,
        "version": version, "document_type": document_type, "audience": audience,
        "department": department, "author": author, "last_updated": str(date.today()),
        "publication_date": publication_date or str(date.today()),
        "keywords": _keywords(haystack), "summary": summary or "No extractable summary.",
    }
    metadata["lifecycle_status"] = determine_lifecycle_status(metadata)
    return metadata


def metadata_match_score(metadata: Dict[str, Any], filters: Dict[str, str]) -> float:
    """Calculate agreement with the active metadata filters."""
    active = [(key, value)
              for key, value in filters.items() if value and value != "All"]
    if not active:
        return 1.0
    matches = sum(str(metadata.get(key, "")).lower() == value.lower()
                  for key, value in active)
    return matches / len(active)


def shared_topics(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
    """Return keywords shared by two metadata records."""
    return sorted(set(str(left.get("keywords", "")).split(",")) & set(str(right.get("keywords", "")).split(",")))

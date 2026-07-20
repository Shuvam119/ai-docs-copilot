"""Enterprise document metadata extraction and normalization."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List


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
    stop_words = {"this", "that", "with", "from", "your", "will", "have", "into", "using", "when", "where", "document", "guide", "version", "the", "and", "for", "are", "you"}
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text.lower())
    unique: List[str] = []
    for word in words:
        if word not in stop_words and word not in unique:
            unique.append(word)
        if len(unique) == limit:
            break
    return unique


def extract_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a consistent enterprise metadata record from file name and content."""
    source = document["metadata"]
    text = document.get("text", "")
    filename = source["filename"]
    haystack = f"{filename}\n{text[:5000]}"
    lowered = haystack.lower()

    document_type = next((kind for kind in DOCUMENT_TYPES if kind.lower() in lowered), "User Guide")
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

    version = _first_match(r"\b(?:version|ver|v)\s*(\d+(?:\.\d+)*)\b", haystack, "Unspecified")
    product = _first_match(r"(?:product|application|platform)\s*[:\-]\s*([^\n]{2,60})", text, "General")
    department = _first_match(r"(?:department|owner|team)\s*[:\-]\s*([^\n]{2,60})", text, "Documentation")
    summary_source = " ".join(text.split())
    summary = summary_source[:360].rsplit(" ", 1)[0] if len(summary_source) > 360 else summary_source

    return {
        "title": _title_from_filename(filename), "product": product,
        "version": version, "document_type": document_type, "audience": audience,
        "department": department, "last_updated": str(date.today()),
        "keywords": _keywords(haystack), "summary": summary or "No extractable summary.",
    }


def metadata_match_score(metadata: Dict[str, Any], filters: Dict[str, str]) -> float:
    """Calculate agreement with the active metadata filters."""
    active = [(key, value) for key, value in filters.items() if value and value != "All"]
    if not active:
        return 1.0
    matches = sum(str(metadata.get(key, "")).lower() == value.lower() for key, value in active)
    return matches / len(active)


def shared_topics(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
    """Return keywords shared by two metadata records."""
    return sorted(set(str(left.get("keywords", "")).split(",")) & set(str(right.get("keywords", "")).split(",")))

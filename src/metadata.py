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

DOCUMENT_TYPE_ALIASES = {
    "standard operating procedure": "SOP",
    "sop": "SOP",
    "user guide": "User Guide",
    "users guide": "User Guide",
    "user's guide": "User Guide",
    "administrator guide": "Administrator Guide",
    "administrators guide": "Administrator Guide",
    "admin guide": "Administrator Guide",
    "job aid": "Job Aid",
    "quick reference": "Job Aid",
    "quick reference guide": "Job Aid",
    "faq": "FAQ",
    "release notes": "Release Notes",
    "release note": "Release Notes",
    "known issues": "Known Issues",
    "known issue": "Known Issues",
    "api documentation": "API Documentation",
    "api guide": "API Documentation",
    "api reference": "API Documentation",
    "api reference guide": "API Documentation",
    "training manual": "Training Manual",
    "training guide": "Training Manual",
    "troubleshooting guide": "Troubleshooting Guide",
    "troubleshooting": "Troubleshooting Guide",
}

_FILENAME_TYPE_SIGNALS = (
    ("api", "API Documentation"),
    ("integration guide", "API Documentation"),
    ("job aid", "Job Aid"),
    ("quick reference", "Job Aid"),
    ("sop", "SOP"),
    ("faq", "FAQ"),
    ("release notes", "Release Notes"),
    ("known issues", "Known Issues"),
    ("user guide", "User Guide"),
    ("users guide", "User Guide"),
    ("admin guide", "Administrator Guide"),
    ("administrator", "Administrator Guide"),
    ("training", "Training Manual"),
    ("troubleshoot", "Troubleshooting Guide"),
)


def _title_from_filename(filename: str) -> str:
    """Create a readable title without a detected version suffix."""
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    return re.sub(r"\b(v(?:ersion)?\s*\d+(?:\.\d+)*)\b", "", stem, flags=re.I).strip().title()


def _first_match(pattern: str, text: str, default: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else default


_HEADER_BODY_MARKERS = (
    r"summary",
    r"table\s+of\s+contents",
    r"\d+\.\s+(?:introduction|overview|purpose\s+and\s+scope)",
)


def _header_section(text: str, max_chars: int = 3000) -> str:
    """Return the header/front-matter region of a document.

    Product identity is only trustworthy in the metadata/header block, so the
    scan region is truncated at the first body-content marker to avoid
    misreading product names mentioned in the body prose.
    """
    marker = re.search(
        r"(?im)^\s*(?:%s)\b" % "|".join(_HEADER_BODY_MARKERS), text)
    if marker:
        text = text[:marker.start()]
    return text[:max_chars]


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
    "Fresh", "Need Update", "Needs Deprecation", "Aging", "Stale",
    "Archived", "Needs Review",
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


def _normalized_version(version: tuple[int, ...], length: int = 3) -> tuple[int, ...]:
    """Normalize a version tuple to a fixed length so '1' == '1.0' == '1.0.0'."""
    return version + (0,) * max(0, length - len(version))


def apply_version_lifecycle(documents: Iterable[Dict[str, Any]]) -> None:
    """
    Classify lifecycle status by comparing versions within each title group.

    Rules:
      - The latest version of a title is marked "Fresh".
      - Older versions up to one major behind the latest are marked
        "Need Update" (a newer version exists but the document may still
        be relevant).
      - Older versions two or more majors behind the latest are marked
        "Needs Deprecation" (stale or obsolete, should be retired).
      - Documents without a parseable version are left unchanged so their
        date-based classification is preserved.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for document in documents:
        title = document.get("metadata", {}).get("title", "")
        groups.setdefault(title, []).append(document)

    for group in groups.values():
        versioned = [
            (document, parse_version_tuple(document["metadata"].get("version", "")))
            for document in group
        ]
        available = [version for _, version in versioned if version is not None]
        if not available:
            continue

        latest = max(available, key=_normalized_version)
        latest_normalized = _normalized_version(latest)

        for document, version in versioned:
            if version is None:
                continue
            normalized = _normalized_version(version)
            if normalized == latest_normalized:
                document["metadata"]["lifecycle_status"] = "Fresh"
            elif latest_normalized[0] - normalized[0] >= 2:
                document["metadata"]["lifecycle_status"] = "Needs Deprecation"
            else:
                document["metadata"]["lifecycle_status"] = "Need Update"


def _normalize_document_type(value: str) -> str:
    """Clean a raw Document Type value and map it to a canonical label."""
    cleaned = re.sub(r"\s*\|\s*.*$", "", value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return DOCUMENT_TYPE_ALIASES.get(cleaned.lower(), cleaned)


def _declared_document_type(text: str) -> str:
    """Read the Document Type field declared in the document's metadata block."""
    match = re.search(
        r"(?im)^\s*document\s+type\b\s*(?:[:|\-])?\s*([^\n|]{1,80})", text)
    return _normalize_document_type(match.group(1)) if match else ""


def _infer_document_type(haystack: str) -> str:
    """Best-effort type detection used when the document declares none."""
    filename = haystack.split("\n", 1)[0].lower()
    for signal, kind in _FILENAME_TYPE_SIGNALS:
        if signal in filename:
            return kind
    lowered = haystack.lower()
    return next((kind for kind in DOCUMENT_TYPES if kind.lower() in lowered), "User Guide")


def _document_type_from_document(text: str, filename: str) -> str:
    """Prefer the type declared by the document itself, else infer one."""
    declared = _declared_document_type(text)
    if declared:
        return declared
    return _infer_document_type(f"{filename}\n{text[:5000]}")


def extract_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a consistent enterprise metadata record from file name and content."""
    source = document["metadata"]
    text = document.get("text", "")
    filename = source["filename"]
    haystack = f"{filename}\n{text[:5000]}"
    lowered = haystack.lower()

    document_type = _document_type_from_document(text, filename)

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
        r"(?:product|application|platform)\s*[:\-]\s*([^\n]{2,60})",
        _header_section(text),
        "General",
    )
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

"""
Index Builder

Orchestrates document loading, chunking, embedding, and vector store indexing.
"""

import hashlib
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from src.chunker import DocumentChunker
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DUPLICATE_SEMANTIC_FLOOR,
    DUPLICATE_THRESHOLD,
    EMBEDDING_MODEL,
    RAW_DATA_DIR,
    VECTORSTORE_DIR,
)
from src.embedder import EmbeddingsGenerator
from src.load_document import load_documents_from_directory
from src.perf import log_duration
from src.vector_db import VectorStore
from src.metadata import (
    apply_version_lifecycle,
    extract_metadata,
    parse_version_tuple,
)

logger = logging.getLogger(__name__)


class _DetectionContext:
    """Shared caches for one duplicate-detection pass.

    Duplicate detection compares every document pair, normalizes text, hashes
    it, and sometimes embeds it. All of those steps are deterministic and
    symmetric, so caching them per pass turns the previous O(n^2) re-work
    (every document re-scanned inside every other document's detection) into
    one compute-per-result pass without changing any decision or score.
    """

    __slots__ = (
        "normalized", "text_hash", "family_key", "copy_marker",
        "version_tuple", "pair_similarity", "semantic_embedding",
        "indexed_text",
    )

    def __init__(self) -> None:
        self.normalized = {}
        self.text_hash = {}
        self.family_key = {}
        self.copy_marker = {}
        self.version_tuple = {}
        self.pair_similarity = {}
        self.semantic_embedding = {}
        self.indexed_text = {}


@dataclass
class IndexStats:
    """Statistics from an index build operation."""

    document_count: int
    chunk_count: int
    filenames: List[str]
    failed_files: List[str] = field(default_factory=list)
    empty_files: List[str] = field(default_factory=list)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class IndexBuilder:
    """Builds and rebuilds the document vector index."""

    def __init__(
        self,
        raw_data_dir: str | None = None,
        vectorstore_path: str | None = None,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        embedder: Optional[EmbeddingsGenerator] = None,
        semantic_floor: Optional[float] = None,
    ) -> None:
        """
        Initialize the index builder.

        Args:
            raw_data_dir: Directory containing source documents
            vectorstore_path: Path to ChromaDB persistent storage
            collection_name: ChromaDB collection name
            embedding_model: Sentence-transformers model identifier
            embedder: Optional pre-built EmbeddingsGenerator to reuse so the
                SentenceTransformer model is not reloaded
            semantic_floor: Minimum normalized-text similarity that triggers a
                semantic embedding confirmation for an otherwise-inconclusive
                pair. Pairs below this floor are never duplicates.
        """
        self.raw_data_dir = raw_data_dir or str(RAW_DATA_DIR)
        self.vectorstore_path = vectorstore_path or str(VECTORSTORE_DIR)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.semantic_floor = (
            semantic_floor if semantic_floor is not None
            else DUPLICATE_SEMANTIC_FLOOR)

        self.chunker = DocumentChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self._embedder: EmbeddingsGenerator | None = embedder
        self._vector_store: VectorStore | None = None

    @property
    def embedder(self) -> EmbeddingsGenerator:
        """Lazy-load the embedding model."""
        if self._embedder is None:
            self._embedder = EmbeddingsGenerator(
                model_name=self.embedding_model)
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        """Lazy-load the vector store."""
        if self._vector_store is None:
            self._vector_store = VectorStore(
                vectorstore_path=self.vectorstore_path,
                collection_name=self.collection_name,
            )
        return self._vector_store

    def build(
        self,
        rebuild: bool = False,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        reindex_files: Optional[List[str]] = None,
    ) -> IndexStats:
        """
        Build the vector index from documents in the raw data directory.

        Args:
            rebuild: If True, clear the existing collection before indexing
            progress_callback: Receives a percentage and status while indexing
            reindex_files: Filenames already in the index that should be
                re-indexed (e.g. their source files were overwritten)

        Returns:
            IndexStats with document and chunk counts

        Raises:
            ValueError: If no documents are found
        """
        def report_progress(percentage: int, status: str) -> None:
            if progress_callback:
                progress_callback(percentage, status)

        report_progress(5, "Reading source documents")
        logger.info("Loading documents from %s", self.raw_data_dir)
        _load_start = time.perf_counter()
        load_result = load_documents_from_directory(self.raw_data_dir)
        documents = load_result.documents
        log_duration(logger, "Document loading", _load_start)

        if not documents:
            detail = ""
            if load_result.failed_files:
                detail = f" Failed files: {'; '.join(load_result.failed_files)}"
            raise ValueError(
                "No PDF or DOCX documents found in data/raw/. "
                "Upload documents before building the index."
                + detail
            )

        report_progress(20, "Extracting enterprise metadata")
        _metadata_start = time.perf_counter()
        for document in documents:
            document["metadata"].update(extract_metadata(document))

        # Apply version-aware lifecycle rules: latest version is Fresh,
        # older versions become "Need Update" or "Needs Deprecation".
        apply_version_lifecycle(documents)
        log_duration(logger, "Metadata extraction", _metadata_start)

        logger.info("Loaded %d document(s)", len(documents))
        all_documents = documents

        if rebuild:
            self.vector_store.clear_collection()
            existing_filenames: set[str] = set()
        else:
            existing_stats = self.vector_store.get_stats()
            existing_filenames = set(existing_stats["filenames"])
            required_fields = {"title", "product", "version", "document_type",
                               "audience", "department", "keywords", "summary"}
            legacy_documents = {
                item["filename"] for item in existing_stats.get("documents", [])
                if not required_fields.issubset(item)
            }
            for filename in legacy_documents:
                self.vector_store.delete_document(filename)
            reindex = set(reindex_files or [])
            for filename in reindex & existing_filenames:
                self.vector_store.delete_document(filename)
            documents = [
                doc for doc in documents
                if doc["metadata"]["filename"] not in existing_filenames
                or doc["metadata"]["filename"] in legacy_documents
                or doc["metadata"]["filename"] in reindex
            ]

        if not documents:
            report_progress(100, "All documents are already indexed")
            return IndexStats(0, 0, [], load_result.failed_files, load_result.empty_files)

        # Detect duplicates for the documents that will actually be indexed.
        # Every uploaded document is compared against all other documents in
        # the batch (same-upload duplicates) as well as already-indexed
        # documents (repository duplicates) before any chunk is embedded or
        # stored, so the duplicate status lands in the stored metadata.
        # Comparison is content-based and format-independent, so PDF and DOCX
        # copies of the same document are detected as duplicates. Version-aware:
        # same-family documents with clearly different versions are classified
        # as Related Version, never as Duplicate.
        _detection_start = time.perf_counter()
        detection_context = _DetectionContext()
        for document in documents:
            filename = document["metadata"]["filename"]
            matches = self.find_duplicates(
                filename, DUPLICATE_THRESHOLD, document_list=all_documents,
                detection_context=detection_context)
            duplicates = [
                match for match in matches
                if match.get("match_type", "duplicate") == "duplicate"]
            related = [
                match for match in matches
                if match.get("match_type") == "related_version"]
            if duplicates:
                best = max(duplicates, key=lambda item: item["similarity"])
                document["metadata"]["duplicate"] = True
                document["metadata"]["duplicate_of"] = best["metadata"]["filename"]
                document["metadata"]["duplicate_score"] = best["similarity"]
            else:
                document["metadata"]["duplicate"] = False
                document["metadata"]["duplicate_of"] = ""
                document["metadata"]["duplicate_score"] = 0.0
            if related:
                best_related = max(
                    related, key=lambda item: item["similarity"])
                document["metadata"]["related_version"] = True
                document["metadata"]["related_version_of"] = (
                    best_related["metadata"]["filename"])
                document["metadata"]["related_version_score"] = (
                    best_related["similarity"])
            else:
                document["metadata"]["related_version"] = False
                document["metadata"]["related_version_of"] = ""
                document["metadata"]["related_version_score"] = 0.0
        log_duration(logger, "Duplicate detection", _detection_start)

        filenames = [doc["metadata"]["filename"] for doc in documents]

        _chunk_start = time.perf_counter()
        chunks = self.chunker.chunk_documents(documents)
        log_duration(logger, "Chunk creation", _chunk_start)
        logger.info("Created %d chunk(s)", len(chunks))

        if not chunks:
            report_progress(100, "No searchable text could be extracted")
            return IndexStats(0, 0, filenames, load_result.failed_files, load_result.empty_files)

        report_progress(40, "Creating semantic embeddings for new documents")
        _embed_start = time.perf_counter()
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)
        log_duration(logger, "Embedding generation", _embed_start)

        report_progress(85, "Saving searchable knowledge")
        _insert_start = time.perf_counter()
        added = self.vector_store.add_chunks(chunks_with_embeddings)
        log_duration(logger, "Chroma insert", _insert_start)
        logger.info("Indexed %d chunk(s) into vector store", added)
        report_progress(100, "Knowledge index is ready")

        return IndexStats(
            document_count=len(filenames),
            chunk_count=added,
            filenames=filenames,
            failed_files=load_result.failed_files,
            empty_files=load_result.empty_files,
        )

    def get_stats(self) -> Dict:
        """Return current vector store statistics."""
        return self.vector_store.get_stats()

    def remove_orphaned_documents(self) -> list[str]:
        """Purge indexed documents whose source files no longer exist."""
        raw_files = {
            source.name for source in Path(self.raw_data_dir).iterdir()
            if source.suffix.lower() in {".pdf", ".docx"}
        }
        stats = self.vector_store.get_stats()
        orphaned = [filename for filename in stats.get(
            "filenames", []) if filename not in raw_files]
        for filename in orphaned:
            self.vector_store.delete_document(filename)
        return orphaned

    def delete_document(self, filename: str, delete_source: bool = True) -> int:
        """Remove an indexed document and, optionally, its uploaded source file."""
        removed = self.vector_store.delete_document(filename)
        if delete_source:
            source = Path(self.raw_data_dir) / filename
            if source.exists():
                source.unlink()
        return removed

    def clear_repository(self, delete_sources: bool = False) -> None:
        """Clear the index and optionally remove uploaded source files."""
        self.vector_store.clear_collection()
        if delete_sources:
            for source in Path(self.raw_data_dir).iterdir():
                if source.suffix.lower() in {".pdf", ".docx"}:
                    source.unlink()

    _BULLET_GLYPHS = (
        "\u2022"  # bullet
        "\u25cf"  # black circle
        "\u25aa"  # black small square
        "\u25a0"  # black square
        "\u25e6"  # white bullet
        "\u2023"  # triangular bullet
        "\u2043"  # hyphen bullet
        "\u2219"  # bullet operator
        "\u00b7"  # middle dot
        "\u2757"  # heavy exclamation
        "\u26a0"  # warning sign
    )

    def _document_hash(self, text: str) -> str:
        normalized_text = self._normalize_text(text).encode("utf-8")
        return hashlib.sha256(normalized_text).hexdigest()

    def _hash_of(
        self, context: Optional[_DetectionContext], filename: str,
        text: str,
    ) -> str:
        """SHA-256 of the normalized text, cached per document per pass."""
        if context is not None and filename:
            digest = context.text_hash.get(filename)
            if digest is not None:
                return digest
        normalized = self._normalized_of(context, filename, text)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if context is not None and filename:
            context.text_hash[filename] = digest
        return digest

    def _normalized_of(
        self, context: Optional[_DetectionContext], filename: str,
        text: str,
    ) -> str:
        """Normalized document text, computed at most once per document."""
        if context is not None and filename:
            cached = context.normalized.get(filename)
            if cached is not None:
                return cached
        normalized = self._normalize_text(text)
        if context is not None and filename:
            context.normalized[filename] = normalized
        return normalized

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize extracted text so PDF and DOCX copies compare alike.

        Applies, in order: Unicode normalization, removal of PDF page-break
        markers, replacement of bullet glyphs with a plain space, collapse of
        repeated whitespace/line breaks, and lowercasing. Meaningful content is
        preserved; only file-type extraction artifacts are removed so the same
        document rendered as PDF and DOCX produces comparable text.
        """
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", str(text))
        normalized = normalized.replace("\f", " ")
        normalized = re.sub(
            r"(?m)^\s*---\s*Page\s+\d+\s*---\s*$", " ", normalized)
        normalized = re.sub(
            r"[" + IndexBuilder._BULLET_GLYPHS + r"]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip().lower()

    def _family_key(
        self, metadata: Dict, context: Optional[_DetectionContext] = None,
    ) -> str:
        """Stable identity of a document family, independent of its version.

        Versions and copy markers are stripped so 'UserGuide_v3' and
        'UserGuide_Copy' belong to the same family and 'SOP_Final (1)'
        collapses to 'SOP_Final'. Whether a copy marker was present is tracked
        separately by _has_copy_marker so it can still drive duplicate
        classification.
        """
        filename = str(metadata.get("filename", ""))
        if context is not None and filename:
            cached = context.family_key.get(filename)
            if cached is not None:
                return cached
        stem = Path(filename).stem.lower().replace("_", " ")
        name = re.sub(r"\bv?(?:ersion)?\s*\d+(?:\.\d+)*\b", "", stem)
        name = re.sub(r"[\(\[]\s*\d+\s*[\)\]]", "", name)
        name = re.sub(r"\b(copy|duplicate|dup|backup)\b", "", name)
        name = re.sub(r"[\W_]+", " ", name)
        key = re.sub(r"\s+", " ", name).strip()
        if context is not None and filename:
            context.family_key[filename] = key
        return key

    @staticmethod
    def _has_copy_marker(
        filename: str, context: Optional[_DetectionContext] = None,
    ) -> bool:
        """True when a filename looks like an OS-created copy.

        Catches 'UserGuide_Copy', 'FAQ_duplicate' and browser-style suffixes
        such as 'SOP_Final (1)'. Bare numbers and 'final'/'draft' are not
        treated as copy markers so legitimately versioned files are not
        misclassified.
        """
        if context is not None and filename:
            cached = context.copy_marker.get(filename)
            if cached is not None:
                return cached
        stem = Path(str(filename or "")).stem.lower()
        has_marker = bool(
            re.search(r"\b(copy|duplicate|dup|backup)\b", stem)
            or re.search(r"[\(\[]\s*\d+\s*[\)\]]", stem))
        if context is not None and filename:
            context.copy_marker[filename] = has_marker
        return has_marker

    def _version_tuple(
        self, metadata: Dict, context: Optional[_DetectionContext] = None,
    ):
        """Numeric version tuple for a document, cached per document."""
        filename = str(metadata.get("filename", ""))
        if context is not None and filename:
            cached = context.version_tuple.get(filename)
            if cached is not None:
                return cached
        version = parse_version_tuple(str(metadata.get("version", "")))
        if context is not None and filename:
            context.version_tuple[filename] = version
        return version

    def _versions_differ(
        self, left_meta: Dict, right_meta: Dict,
        context: Optional[_DetectionContext] = None,
    ) -> bool:
        left = self._version_tuple(left_meta, context)
        right = self._version_tuple(right_meta, context)
        return left is not None and right is not None and left != right

    def _pair_intent(
        self, left_meta: Dict, right_meta: Dict,
        context: Optional[_DetectionContext] = None,
    ) -> str:
        """Metadata-only relationship between two documents.

        Returns 'duplicate', 'related_version', or 'none'.

        'duplicate' means the pair shares a document family and is NOT a
        distinct version (same version, a copy marker in a filename, or no
        version info at all) — actual content overlap is still confirmed by
        the caller before a duplicate is reported. 'related_version' is
        returned only when both documents carry clearly different versions of
        one family; such pairs are never duplicates, no matter how similar
        their content is.
        """
        if self._family_key(left_meta, context) != self._family_key(
                right_meta, context):
            return "none"
        if self._versions_differ(left_meta, right_meta, context):
            return "related_version"
        if (self._has_copy_marker(left_meta.get("filename", ""), context)
                or self._has_copy_marker(right_meta.get("filename", ""), context)):
            return "duplicate"
        return "duplicate"

    def _assess_pair(
        self,
        left: Dict,
        right: Dict,
        content_sim: float,
        threshold: float,
        context: Optional[_DetectionContext] = None,
    ) -> tuple[str, float]:
        """Classify a document pair as 'duplicate', 'related_version', or 'none'.

        Returns (kind, similarity). Exact content identity (on normalized
        text) is always a duplicate; same-family documents with clearly
        different versions are always 'related_version' and never duplicates.

        Decision order:
          1. Metadata intent.
          2. Normalized-text similarity: at/above the threshold it is a
             duplicate without further work (existing behaviour).
          3. For likely matches (text similarity at/above the semantic floor)
             whose metadata is inconclusive or whose text is just under the
             threshold, a semantic embedding comparison confirms the match
             before it is classified. Pairs whose text clearly differs are
             never embedded.
        """
        left_meta = left["metadata"]
        right_meta = right["metadata"]
        left_filename = str(left_meta.get("filename", ""))
        right_filename = str(right_meta.get("filename", ""))
        left_text = str(left.get("text") or "")
        right_text = str(right.get("text") or "")

        if left_text and right_text and self._hash_of(
                context, left_filename, left_text) == self._hash_of(
                context, right_filename, right_text):
            return "duplicate", 1.0

        intent = self._pair_intent(left_meta, right_meta, context)
        if intent == "related_version":
            return "related_version", content_sim
        if content_sim >= threshold:
            return "duplicate", content_sim
        if content_sim >= self.semantic_floor and left_text and right_text:
            semantic_sim = self._semantic_similarity(
                left_filename, left_text, right_filename, right_text,
                context)
            if semantic_sim >= threshold:
                return "duplicate", semantic_sim
        return "none", content_sim

    def _semantic_similarity(
        self,
        left_filename: str,
        left_text: str,
        right_filename: str,
        right_text: str,
        context: Optional[_DetectionContext] = None,
    ) -> float:
        """Cosine similarity of the two documents' normalized text embeddings.

        Only called for pairs that already look alike on normalized text, so
        embeddings are computed just for likely matches. A shared embedding
        cache avoids embedding the same text twice within one detection pass.
        """
        if context is None:
            context = _DetectionContext()

        def embed(text: str) -> np.ndarray:
            vector = context.semantic_embedding.get(text)
            if vector is None:
                vector = self.embedder.embed_text(
                    text[:4000], is_query=False)
                context.semantic_embedding[text] = vector
            return vector

        left_vec = embed(self._normalized_of(
            context, left_filename, left_text))
        right_vec = embed(self._normalized_of(
            context, right_filename, right_text))
        left_norm = float(np.linalg.norm(left_vec))
        right_norm = float(np.linalg.norm(right_vec))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))

    def _chunk_similarity(
        self, left_filename: str, left_text: str, right_filename: str,
        right_text: str, context: Optional[_DetectionContext] = None,
        max_chars: int = 20000, min_ratio: float = 0.5,
    ) -> float:
        """Estimate near-duplicate similarity on normalized text.

        Text is normalized (Unicode, page breaks, bullets, whitespace, case)
        before comparing so PDF/DOCX extraction differences do not hide an
        identical copy. The reported score is the best of a character-level
        match (preserves existing same-format behaviour) and a word-level
        match (robust to cross-format reflow and list-marker differences).
        Comparisons are bounded and length-filtered so duplicate detection
        stays cheap even for large repositories. Results are symmetric and
        memoized per pass so each pair is scored only once.
        """
        if context is not None:
            left_norm = self._normalized_of(
                context, left_filename, left_text)[:max_chars]
            right_norm = self._normalized_of(
                context, right_filename, right_text)[:max_chars]
            pair_key = (left_norm, right_norm)
            cached = context.pair_similarity.get(pair_key)
            if cached is not None:
                return cached
        else:
            left_norm = self._normalize_text(left_text)[:max_chars]
            right_norm = self._normalize_text(right_text)[:max_chars]
        if not left_norm or not right_norm:
            return 0.0
        shorter, longer = sorted((left_norm, right_norm), key=len)
        if len(shorter) / len(longer) < min_ratio:
            return 0.0
        char_sim = SequenceMatcher(None, left_norm, right_norm).ratio()
        word_sim = SequenceMatcher(
            None, left_norm.split(), right_norm.split(),
            autojunk=False).ratio()
        similarity = max(char_sim, word_sim)
        if context is not None:
            context.pair_similarity[pair_key] = similarity
        return similarity

    def compare_documents(self, left: str, right: str) -> Dict:
        """Compare indexed document content using chunk-level sequence matching."""
        left_chunks, right_chunks = self.vector_store.document_chunks(
            left), self.vector_store.document_chunks(right)
        left_text, right_text = "\n".join(c["text"] for c in left_chunks), "\n".join(
            c["text"] for c in right_chunks)
        matcher = SequenceMatcher(
            None, left_text.splitlines(), right_text.splitlines())
        additions, removals, changes = [], [], []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "insert":
                additions.extend(right_text.splitlines()[j1:j2][:5])
            elif tag == "delete":
                removals.extend(left_text.splitlines()[i1:i2][:5])
            elif tag == "replace":
                changes.append(" ".join(right_text.splitlines()[j1:j2])[:300])
        return {"similarity": round(matcher.ratio() * 100), "additions": additions[:10], "removed_sections": removals[:10], "changed_procedures": changes[:10]}

    def find_duplicates(
        self,
        filename: str,
        threshold: float,
        document_list: Optional[List[Dict]] = None,
        detection_context: Optional[_DetectionContext] = None,
    ) -> List[Dict]:
        """Find duplicates and version relations for a document before indexing.

        Detection order:
          1. Content comparison against every other document in the upload
             batch (same-upload duplicates, including cross-format copies).
          2. Metadata-first comparison against already-indexed documents.
          3. Semantic (embedding) search against the indexed collection to
             discover accidental copies whose metadata does not relate them.

        Each pair is decided by metadata intent first, then normalized-text
        similarity (format-independent: PDF and DOCX copies of the same
        document compare alike). Pairs whose text is a likely match but not
        conclusively identical are confirmed with a semantic embedding
        comparison before being classified as duplicates; pairs whose text
        clearly differs are never embedded.

        Version-aware: two documents that share a document family but carry
        clearly different versions are classified as 'related_version', never
        'duplicate', no matter how similar their content is.

        Each returned dict includes 'source_filename', 'match_filename',
        'metadata', 'similarity', 'text', 'reason' and 'match_type'
        ('duplicate' or 'related_version').

        When ``detection_context`` is shared across calls (as during a single
        index build), normalized text, hashes, family keys, and pair
        similarities are computed once per document instead of once per
        document-pair, eliminating the repeated work of per-document scans.
        """
        source = Path(self.raw_data_dir) / filename
        if not source.exists():
            return []
        document = document_list or load_documents_from_directory(
            self.raw_data_dir).documents
        candidate = next(
            (item for item in document if item["metadata"]["filename"] == filename), None)
        if not candidate:
            return []
        context = detection_context or _DetectionContext()

        matches: List[Dict] = []
        seen_filenames: set[str] = set()

        def record(kind: str, other_filename: str, metadata: Dict,
                   similarity: float, text: str, reason: str) -> None:
            matches.append({
                "source_filename": filename,
                "match_filename": other_filename,
                "metadata": metadata,
                "similarity": similarity,
                "text": text,
                "reason": reason,
                "match_type": kind,
            })

        # 1. Same-batch comparison. Every other document that shares the raw
        # directory is compared directly because its full text is available.
        candidate_text = candidate.get("text", "")
        for other in document:
            other_filename = other["metadata"]["filename"]
            if other_filename == filename or other_filename in seen_filenames:
                continue
            content_sim = self._chunk_similarity(
                filename, candidate_text, other_filename,
                other.get("text", ""), context)
            kind, similarity = self._assess_pair(
                candidate, other, content_sim, threshold, context)
            if kind != "none":
                record(kind, other_filename, other["metadata"], similarity,
                       other.get("text", ""), "text")
                seen_filenames.add(other_filename)

        indexed = self.vector_store.get_stats()
        if indexed["total_chunks"]:
            if not context.indexed_text:
                context.indexed_text = self.vector_store.document_text_map()
            indexed_text = context.indexed_text
            candidate_meta = candidate["metadata"]
            # 2. Repository comparison: metadata first, content confirmation
            # only when the pair is a plausible duplicate candidate.
            for existing in indexed.get("documents", []):
                other_filename = existing.get("filename", "")
                if other_filename == filename or other_filename in seen_filenames:
                    continue
                intent = self._pair_intent(candidate_meta, existing, context)
                if intent == "related_version":
                    continue
                if intent != "duplicate":
                    continue
                other_text = indexed_text.get(other_filename, "")
                kind, similarity = self._assess_pair(
                    candidate, {"metadata": existing, "text": other_text},
                    self._chunk_similarity(
                        filename, candidate_text, other_filename,
                        other_text, context), threshold, context)
                if kind == "duplicate":
                    record(kind, other_filename, existing, similarity,
                           other_text[:180], "metadata_indexed")
                seen_filenames.add(other_filename)

            # 3. Semantic fallback against the indexed collection. The embedding
            # search only DISCOVERS candidate matches; the embedding cosine is
            # not a reliable duplicate signal on its own (boilerplate-heavy
            # corpora score 0.95+ for unrelated documents), so every candidate
            # is confirmed against its full extracted text instead.
            #
            # When every indexed document is part of the current batch (a full
            # rebuild clears the collection first, so the index can only
            # contain documents the batch already compared in step 1), the
            # search can discover nothing new and the embedding is skipped.
            indexed_filenames = set(indexed.get("filenames", []))
            batch_filenames = {
                item["metadata"]["filename"] for item in document}
            if candidate_text and not (
                    indexed_filenames and indexed_filenames <= batch_filenames):
                vector = self.embedder.embed_text(
                    candidate_text[:4000], is_query=False).tolist()
                for match in self.vector_store.search(vector, top_k=10):
                    other_filename = match["metadata"]["filename"]
                    if other_filename == filename or other_filename in seen_filenames:
                        continue
                    other_text = indexed_text.get(other_filename, "")
                    kind, similarity = self._assess_pair(
                        candidate, {"metadata": match["metadata"],
                                    "text": other_text},
                        self._chunk_similarity(
                            filename, candidate_text, other_filename,
                            other_text, context), threshold, context)
                    if kind != "none":
                        record(kind, other_filename, match["metadata"],
                               similarity, other_text[:180], "embedding")
                        seen_filenames.add(other_filename)

        return matches

    def has_index(self) -> bool:
        """Return True if the vector store contains indexed chunks."""
        return self.get_stats()["total_chunks"] > 0

    def create_rag_pipeline(self):
        """
        Create a RAG pipeline wired to the current index.

        Returns:
            RAGPipeline ready for question answering

        Raises:
            ValueError: If no indexed chunks are available
        """
        if not self.has_index():
            raise ValueError(
                "No indexed documents found. Upload documents and rebuild the index."
            )

        from src.llm import LLMClient
        from src.rag import RAGPipeline
        from src.retriever import Retriever

        retriever = Retriever(self.embedder, self.vector_store)
        llm = LLMClient()
        return RAGPipeline(self.embedder, retriever, llm)

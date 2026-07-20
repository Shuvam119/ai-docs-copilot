"""
Index Builder

Orchestrates document loading, chunking, embedding, and vector store indexing.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.chunker import DocumentChunker
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RAW_DATA_DIR,
    VECTORSTORE_DIR,
)
from src.embedder import EmbeddingsGenerator
from src.load_document import load_documents_from_directory
from src.vector_db import VectorStore
from src.metadata import extract_metadata

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Statistics from an index build operation."""

    document_count: int
    chunk_count: int
    filenames: List[str]
    failed_files: List[str] = field(default_factory=list)
    empty_files: List[str] = field(default_factory=list)


class IndexBuilder:
    """Builds and rebuilds the document vector index."""

    def __init__(
        self,
        raw_data_dir: str | None = None,
        vectorstore_path: str | None = None,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        """
        Initialize the index builder.

        Args:
            raw_data_dir: Directory containing source documents
            vectorstore_path: Path to ChromaDB persistent storage
            collection_name: ChromaDB collection name
            embedding_model: Sentence-transformers model identifier
        """
        self.raw_data_dir = raw_data_dir or str(RAW_DATA_DIR)
        self.vectorstore_path = vectorstore_path or str(VECTORSTORE_DIR)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self.chunker = DocumentChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self._embedder: EmbeddingsGenerator | None = None
        self._vector_store: VectorStore | None = None

    @property
    def embedder(self) -> EmbeddingsGenerator:
        """Lazy-load the embedding model."""
        if self._embedder is None:
            self._embedder = EmbeddingsGenerator(model_name=self.embedding_model)
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
    ) -> IndexStats:
        """
        Build the vector index from documents in the raw data directory.

        Args:
            rebuild: If True, clear the existing collection before indexing
            progress_callback: Receives a percentage and status while indexing

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
        load_result = load_documents_from_directory(self.raw_data_dir)
        documents = load_result.documents

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
        for document in documents:
            document["metadata"].update(extract_metadata(document))
        filenames = [doc["metadata"]["filename"] for doc in documents]
        logger.info("Loaded %d document(s)", len(documents))

        if rebuild:
            self.vector_store.clear_collection()
            existing_filenames: set[str] = set()
        else:
            existing_stats = self.vector_store.get_stats()
            existing_filenames = set(existing_stats["filenames"])
            required_fields = {"title", "product", "version", "document_type", "audience", "department", "keywords", "summary"}
            legacy_documents = {
                item["filename"] for item in existing_stats.get("documents", [])
                if not required_fields.issubset(item)
            }
            for filename in legacy_documents:
                self.vector_store.delete_document(filename)
            documents = [
                doc for doc in documents
                if doc["metadata"]["filename"] not in existing_filenames
                or doc["metadata"]["filename"] in legacy_documents
            ]
        if not documents:
            report_progress(100, "All documents are already indexed")
            stats = self.vector_store.get_stats()
            return IndexStats(len(filenames), 0, filenames, load_result.failed_files, load_result.empty_files)
        chunks = self.chunker.chunk_documents(documents)
        logger.info("Created %d chunk(s)", len(chunks))

        report_progress(40, "Creating semantic embeddings for new documents")
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)

        report_progress(85, "Saving searchable knowledge")
        added = self.vector_store.add_chunks(chunks_with_embeddings)
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

    def compare_documents(self, left: str, right: str) -> Dict:
        """Compare indexed document content using chunk-level sequence matching."""
        from difflib import SequenceMatcher
        left_chunks, right_chunks = self.vector_store.document_chunks(left), self.vector_store.document_chunks(right)
        left_text, right_text = "\n".join(c["text"] for c in left_chunks), "\n".join(c["text"] for c in right_chunks)
        matcher = SequenceMatcher(None, left_text.splitlines(), right_text.splitlines())
        additions, removals, changes = [], [], []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "insert": additions.extend(right_text.splitlines()[j1:j2][:5])
            elif tag == "delete": removals.extend(left_text.splitlines()[i1:i2][:5])
            elif tag == "replace": changes.append(" ".join(right_text.splitlines()[j1:j2])[:300])
        return {"similarity": round(matcher.ratio() * 100), "additions": additions[:10], "removed_sections": removals[:10], "changed_procedures": changes[:10]}

    def find_duplicates(self, filename: str, threshold: float) -> List[Dict]:
        """Find semantic near-duplicates for a newly indexed/uploaded document."""
        source = Path(self.raw_data_dir) / filename
        if not source.exists() or not self.vector_store.get_stats()["total_chunks"]:
            return []
        document = load_documents_from_directory(self.raw_data_dir).documents
        candidate = next((item for item in document if item["metadata"]["filename"] == filename), None)
        if not candidate:
            return []
        vector = self.embedder.embed_text(candidate["text"][:4000], is_query=True).tolist()
        matches = self.vector_store.search(vector, top_k=5)
        return [match for match in matches if match["metadata"]["filename"] != filename and match["similarity"] >= threshold]

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

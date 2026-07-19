"""
Index Builder

Orchestrates document loading, chunking, embedding, and vector store indexing.
"""

import logging
from dataclasses import dataclass, field
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
        rebuild: bool = True,
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

        report_progress(20, "Preparing document content")
        filenames = [doc["metadata"]["filename"] for doc in documents]
        logger.info("Loaded %d document(s)", len(documents))

        chunks = self.chunker.chunk_documents(documents)
        logger.info("Created %d chunk(s)", len(chunks))

        report_progress(40, "Creating semantic embeddings")
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)

        if rebuild:
            report_progress(75, "Refreshing the knowledge index")
            self.vector_store.clear_collection()

        report_progress(85, "Saving searchable knowledge")
        added = self.vector_store.add_chunks(chunks_with_embeddings)
        logger.info("Indexed %d chunk(s) into vector store", added)
        report_progress(100, "Knowledge index is ready")

        return IndexStats(
            document_count=len(documents),
            chunk_count=added,
            filenames=filenames,
            failed_files=load_result.failed_files,
            empty_files=load_result.empty_files,
        )

    def get_stats(self) -> Dict:
        """Return current vector store statistics."""
        return self.vector_store.get_stats()

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

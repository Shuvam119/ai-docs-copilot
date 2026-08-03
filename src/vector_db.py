"""
Vector Database Management

Handles ChromaDB initialization, chunk storage, and vector retrieval.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

from src.config import COLLECTION_NAME, VECTORSTORE_DIR

logger = logging.getLogger(__name__)

# --- Process-wide ChromaDB singletons -------------------------------------
# Creating a PersistentClient is expensive and multiple clients against the
# same on-disk path contend for the underlying SQLite database. The client,
# collection, and cached stats below are shared at module level so every
# VectorStore/IndexBuilder in the process uses exactly ONE client and ONE
# collection per (path, collection) — regardless of how many objects (Streamlit
# sessions, RAG pipelines) are constructed.
_client_registry: Dict[str, chromadb.ClientAPI] = {}
_collection_registry: Dict[str, chromadb.Collection] = {}
_stats_registry: Dict[str, Dict] = {}
_registry_lock = threading.Lock()


def _registry_key(vectorstore_path: str, collection_name: str) -> str:
    return f"{Path(vectorstore_path).resolve()}::{collection_name}"


def _get_client(vectorstore_path: str) -> chromadb.ClientAPI:
    """Return the shared PersistentClient for a path, creating it once."""
    path = str(Path(vectorstore_path).resolve())
    with _registry_lock:
        client = _client_registry.get(path)
        if client is None:
            client = chromadb.PersistentClient(path=path)
            _client_registry[path] = client
            logger.info("Created ChromaDB PersistentClient at %s", path)
    return client


def _get_collection(vectorstore_path: str, collection_name: str) -> chromadb.Collection:
    """Return the shared collection for a path/name, creating it once."""
    key = _registry_key(vectorstore_path, collection_name)
    path = str(Path(vectorstore_path).resolve())
    with _registry_lock:
        collection = _collection_registry.get(key)
        if collection is None:
            client = _client_registry.get(path)
            if client is None:
                client = chromadb.PersistentClient(path=path)
                _client_registry[path] = client
                logger.info("Created ChromaDB PersistentClient at %s", path)
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            _collection_registry[key] = collection
            logger.info("Created ChromaDB collection %r at %s", collection_name, key)
    return collection


def _drop_cached_collection(vectorstore_path: str, collection_name: str) -> None:
    """Drop cached collection/stats so they are recreated on next use."""
    key = _registry_key(vectorstore_path, collection_name)
    with _registry_lock:
        _collection_registry.pop(key, None)
        _stats_registry.pop(key, None)


class VectorStore:
    """Manages vector database with ChromaDB."""

    def __init__(
        self,
        vectorstore_path: str | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """
        Initialize ChromaDB vector store.

        Args:
            vectorstore_path: Path to persistent storage directory
            collection_name: Name of the collection
        """
        self.vectorstore_path = Path(vectorstore_path or VECTORSTORE_DIR)
        self.collection_name = collection_name

        self.vectorstore_path.mkdir(parents=True, exist_ok=True)

        # Warm the shared client/collection so initialization happens exactly
        # once per process.
        _get_collection(str(self.vectorstore_path), collection_name)

        logger.info(
            "Vector store ready at %s (collection: %s)",
            self.vectorstore_path,
            collection_name,
        )

    @property
    def client(self) -> chromadb.ClientAPI:
        """The process-wide PersistentClient for this store's path."""
        return _get_client(str(self.vectorstore_path))

    @property
    def collection(self) -> chromadb.Collection:
        """The process-wide collection for this store's path/name."""
        return _get_collection(str(self.vectorstore_path), self.collection_name)

    def _stats_key(self) -> str:
        return _registry_key(str(self.vectorstore_path), self.collection_name)

    def _invalidate_stats(self) -> None:
        with _registry_lock:
            _stats_registry.pop(self._stats_key(), None)

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Add chunks with embeddings to the vector store.

        Args:
            chunks: List of chunks with 'embedding', 'text', and 'metadata'

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict] = []

        for chunk in chunks:
            metadata = chunk["metadata"]
            chunk_id = metadata["chunk_id"]

            ids.append(chunk_id)
            embeddings.append(chunk["embedding"].tolist())
            documents.append(chunk["text"])
            metadatas.append({
                "filename": metadata["filename"], "chunk_id": chunk_id,
                "document_type": metadata["document_type"], "chunk_index": metadata["chunk_index"],
                "total_chunks": metadata["total_chunks"], "source": metadata.get("source", ""),
                "title": metadata.get("title", metadata["filename"]),
                "product": metadata.get("product", "General"), "version": metadata.get("version", "Unspecified"),
                "audience": metadata.get("audience", "End User"), "department": metadata.get("department", "Documentation"),
                "author": metadata.get("author", "Unknown"),
                "last_updated": metadata.get("last_updated", ""),
                "publication_date": metadata.get("publication_date", ""),
                "lifecycle_status": metadata.get("lifecycle_status", "Fresh"),
                "keywords": ", ".join(metadata.get("keywords", [])),
                "summary": metadata.get("summary", ""),
            })

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        self._invalidate_stats()
        logger.info("Added %d chunk(s) to vector store", len(ids))
        return len(ids)

    def search(self, query_embedding: List[float], top_k: int = 5, where: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return

        Returns:
            List of retrieved chunks with metadata
        """
        count = self.collection.count()
        if not count:
            return []
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=min(top_k, count), where=where)

        retrieved: List[Dict] = []
        if results and results["ids"] and results["ids"][0]:
            for doc_id, document, distance, metadata in zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            ):
                retrieved.append(
                    {
                        "id": doc_id,
                        "text": document,
                        "distance": distance,
                        "similarity": 1 - distance,
                        "metadata": metadata,
                    }
                )

        return retrieved

    def get_stats(self) -> Dict:
        """Get vector store statistics.

        Results are cached process-wide and invalidated automatically whenever
        chunks are added, deleted, or the collection is cleared, so the same
        full metadata scan is not repeated on every question or library open.
        """
        key = self._stats_key()
        cached = _stats_registry.get(key)
        if cached is not None:
            return dict(cached)

        count = self.collection.count()
        document_count = 0
        filenames: set[str] = set()
        document_metadata: Dict[str, Dict] = {}

        if count > 0:
            all_metadata = self.collection.get(include=["metadatas"])
            filenames = {
                meta["filename"]
                for meta in all_metadata["metadatas"]
                if meta and "filename" in meta
            }
            document_count = len(filenames)
            for meta in all_metadata["metadatas"]:
                if meta and meta.get("filename") not in document_metadata:
                    document_metadata[meta["filename"]] = meta

        result = {
            "collection": self.collection_name,
            "total_chunks": count,
            "document_count": document_count,
            "filenames": sorted(filenames),
            "vectorstore_path": str(self.vectorstore_path),
            "documents": [document_metadata[name] for name in sorted(document_metadata)],
            "products": sorted({meta.get("product", "General") for meta in document_metadata.values()}),
            "versions": sorted({meta.get("version", "Unspecified") for meta in document_metadata.values()}),
            "document_types": sorted({meta.get("document_type", "Unknown") for meta in document_metadata.values()}),
            "departments": sorted({meta.get("department", "Documentation") for meta in document_metadata.values()}),
            "audiences": sorted({meta.get("audience", "End User") for meta in document_metadata.values()}),
            "lifecycle_statuses": sorted({meta.get("lifecycle_status", "Fresh") for meta in document_metadata.values()}),
        }
        _stats_registry[key] = result
        return dict(result)

    def document_chunks(self, filename: str) -> List[Dict]:
        """Return all chunks belonging to one indexed document in document order."""
        data = self.collection.get(where={"filename": filename}, include=[
                                   "documents", "metadatas"])
        chunks = [{"id": item_id, "text": text, "metadata": meta} for item_id,
                  text, meta in zip(data["ids"], data["documents"], data["metadatas"])]
        return sorted(chunks, key=lambda item: item["metadata"].get("chunk_index", 0))

    def delete_document(self, filename: str) -> int:
        """Delete indexed chunks for a document and return the number removed."""
        existing = self.collection.get(where={"filename": filename})
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
            self._invalidate_stats()
        return len(ids)

    def clear_collection(self) -> None:
        """Delete and recreate the collection for a clean rebuild."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            logger.debug(
                "Collection %s did not exist or could not be deleted",
                self.collection_name,
            )

        _drop_cached_collection(str(self.vectorstore_path), self.collection_name)
        _get_collection(str(self.vectorstore_path), self.collection_name)
        logger.info("Cleared collection: %s", self.collection_name)

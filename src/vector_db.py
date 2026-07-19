"""
Vector Database Management

Handles ChromaDB initialization, chunk storage, and vector retrieval.
"""

import logging
from pathlib import Path
from typing import Dict, List

import chromadb

from src.config import COLLECTION_NAME, VECTORSTORE_DIR

logger = logging.getLogger(__name__)


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

        self.client = chromadb.PersistentClient(path=str(self.vectorstore_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "Vector store initialized at %s (collection: %s)",
            self.vectorstore_path,
            collection_name,
        )

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
            metadatas.append(
                {
                    "filename": metadata["filename"],
                    "chunk_id": chunk_id,
                    "document_type": metadata["document_type"],
                    "chunk_index": metadata["chunk_index"],
                    "total_chunks": metadata["total_chunks"],
                    "source": metadata.get("source", ""),
                }
            )

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Added %d chunk(s) to vector store", len(ids))
        return len(ids)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return

        Returns:
            List of retrieved chunks with metadata
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

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
        """Get vector store statistics."""
        count = self.collection.count()
        document_count = 0
        filenames: set[str] = set()

        if count > 0:
            all_metadata = self.collection.get(include=["metadatas"])
            filenames = {
                meta["filename"]
                for meta in all_metadata["metadatas"]
                if meta and "filename" in meta
            }
            document_count = len(filenames)

        return {
            "collection": self.collection_name,
            "total_chunks": count,
            "document_count": document_count,
            "filenames": sorted(filenames),
            "vectorstore_path": str(self.vectorstore_path),
        }

    def clear_collection(self) -> None:
        """Delete and recreate the collection for a clean rebuild."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            logger.debug(
                "Collection %s did not exist or could not be deleted",
                self.collection_name,
            )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared collection: %s", self.collection_name)

"""
Vector Database Management

Handles ChromaDB initialization, chunk storage, and vector retrieval.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict
import json


class VectorStore:
    """Manages vector database with ChromaDB."""

    def __init__(self, vectorstore_path: str = "vectorstore", collection_name: str = "documents"):
        """
        Initialize ChromaDB vector store.

        Args:
            vectorstore_path: Path to persistent storage directory
            collection_name: Name of the collection
        """
        self.vectorstore_path = Path(vectorstore_path)
        self.collection_name = collection_name

        # Create directory if it doesn't exist
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.vectorstore_path))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"✅ Vector Store initialized")
        print(f"   Path: {self.vectorstore_path}")
        print(f"   Collection: {collection_name}")

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

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            # Generate unique ID
            chunk_id = f"{chunk['metadata']['filename']}_{chunk['metadata']['chunk_index']}"

            ids.append(chunk_id)
            # Convert numpy to list
            embeddings.append(chunk["embedding"].tolist())
            documents.append(chunk["text"])

            # Store metadata as JSON string (ChromaDB limitation)
            metadatas.append({
                "filename": chunk["metadata"]["filename"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "total_chunks": chunk["metadata"]["total_chunks"],
                "source": chunk["metadata"].get("source", ""),
                "type": chunk["metadata"].get("type", "")
            })

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(f"✅ Added {len(ids)} chunks to vector store")
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
            n_results=top_k
        )

        # Parse results
        retrieved = []
        if results and len(results["ids"]) > 0:
            for i, (doc_id, document, distance, metadata) in enumerate(
                zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0]
                )
            ):
                retrieved.append({
                    "id": doc_id,
                    "text": document,
                    "distance": distance,
                    "similarity": 1 - distance,  # Convert distance to similarity
                    "metadata": metadata
                })

        return retrieved

    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        count = self.collection.count()
        return {
            "collection": self.collection_name,
            "total_chunks": count,
            "vectorstore_path": str(self.vectorstore_path)
        }

    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(name=self.collection_name)
        print(f"✅ Deleted collection: {self.collection_name}")

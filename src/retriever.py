"""
Retriever Module

Retrieves relevant document chunks for a given query using semantic search.
"""

from typing import Dict, List, Optional

from src.config import SIMILARITY_THRESHOLD, TOP_K
from src.metadata import metadata_match_score


class Retriever:
    """Retrieves relevant chunks from the vector database."""

    def __init__(self, embedder, vector_store):
        """
        Initialize retriever.

        Args:
            embedder: EmbeddingsGenerator instance
            vector_store: VectorStore instance
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = TOP_K, filters: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The user's question or query
            top_k: Number of top results to return

        Returns:
            List of relevant chunks with metadata and similarity scores
        """
        query_embedding = self.embedder.embed_text(query, is_query=True)
        query_embedding_list = query_embedding.tolist()

        # Search vector database
        results = self.vector_store.search(query_embedding_list, top_k=max(top_k * 8, 30))
        filters = filters or {}
        latest_versions = self._latest_versions() if filters.get("version") == "Latest Version" else {}
        ranked = []
        for result in results:
            metadata = result["metadata"]
            if latest_versions and metadata.get("version") != latest_versions.get(metadata.get("title")):
                continue
            scoring_filters = {key: value for key, value in filters.items() if key != "version" or value != "Latest Version"}
            agreement = metadata_match_score(metadata, scoring_filters)
            if any(value not in ("", "All") for value in filters.values()) and not agreement:
                continue
            result["metadata_agreement"] = agreement
            result["ranking_score"] = (result["similarity"] * 0.8) + (agreement * 0.2)
            ranked.append(result)
        return sorted(ranked, key=lambda item: item["ranking_score"], reverse=True)[:top_k]

    def _latest_versions(self) -> Dict[str, str]:
        """Identify the newest numeric version for each document title."""
        latest: Dict[str, str] = {}
        for document in self.vector_store.get_stats().get("documents", []):
            title, version = document.get("title", ""), document.get("version", "Unspecified")
            if not title or version == "Unspecified":
                continue
            def version_key(value: str) -> tuple:
                return tuple(int(part) for part in value.split(".") if part.isdigit())
            if title not in latest or version_key(version) > version_key(latest[title]):
                latest[title] = version
        return latest

    def retrieve_with_context(self, query: str, top_k: int = TOP_K, filters: Optional[Dict[str, str]] = None) -> Dict:
        """
        Retrieve chunks and prepare context for LLM.

        Args:
            query: The user's question
            top_k: Number of top results

        Returns:
            Dictionary with query and formatted context
        """
        results = self.retrieve(query, top_k=top_k, filters=filters)

        # Format context from retrieved chunks
        sources = set()

        for result in results:
            filename = result["metadata"]["filename"]
            sources.add(filename)
        best_similarity = max((r["similarity"] for r in results), default=0.0)
        low_confidence = not results or best_similarity < SIMILARITY_THRESHOLD
        related_documents = self._related_documents(results)

        return {
            "query": query,
            "sources": list(sources),
            "retrieved_chunks": results,
            "num_chunks": len(results),
            "best_similarity": best_similarity,
            "low_confidence": low_confidence,
            "related_documents": related_documents,
        }

    @staticmethod
    def _related_documents(results: List[Dict]) -> List[str]:
        """Recommend unique source documents based on retrieved evidence."""
        documents = []
        for result in results:
            title = result["metadata"].get("title", result["metadata"].get("filename", ""))
            if title and title not in documents:
                documents.append(title)
        return documents[:4]

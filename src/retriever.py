"""
Retriever Module

Retrieves relevant document chunks for a given query using semantic search.
"""

from typing import Dict, List

from src.config import SIMILARITY_THRESHOLD, TOP_K


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

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
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
        results = self.vector_store.search(query_embedding_list, top_k=top_k)

        return results

    def retrieve_with_context(self, query: str, top_k: int = TOP_K) -> Dict:
        """
        Retrieve chunks and prepare context for LLM.

        Args:
            query: The user's question
            top_k: Number of top results

        Returns:
            Dictionary with query and formatted context
        """
        results = self.retrieve(query, top_k=top_k)

        # Format context from retrieved chunks
        context_parts = []
        sources = set()

        for result in results:
            filename = result["metadata"]["filename"]
            context_parts.append(f"[Source: {filename}]\n{result['text']}")
            sources.add(filename)

        context = "\n\n---\n\n".join(context_parts)
        best_similarity = max((r["similarity"] for r in results), default=0.0)
        low_confidence = not results or best_similarity < SIMILARITY_THRESHOLD

        return {
            "query": query,
            "context": context,
            "sources": list(sources),
            "retrieved_chunks": results,
            "num_chunks": len(results),
            "best_similarity": best_similarity,
            "low_confidence": low_confidence,
        }

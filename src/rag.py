"""
RAG Pipeline

Orchestrates the complete Retrieval-Augmented Generation pipeline.
"""

from typing import Dict, Optional

from src.config import NAVIGATOR_PROMPT, TOP_K


class RAGPipeline:
    """Complete RAG pipeline: Query → Embed → Retrieve → Generate."""

    def __init__(self, embedder, retriever, llm) -> None:
        """
        Initialize RAG pipeline.

        Args:
            embedder: EmbeddingsGenerator instance
            retriever: Retriever instance
            llm: LLMClient instance
        """
        self.embedder = embedder
        self.retriever = retriever
        self.llm = llm

    def answer(
        self,
        query: str,
        top_k: int = TOP_K,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Answer a question using the RAG pipeline.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            system_prompt: Optional custom system prompt

        Returns:
            Dictionary with answer, sources, and retrieved chunks
        """
        retrieval = self.retriever.retrieve_with_context(query, top_k=top_k)

        llm_result = self.llm.generate_navigation_response(
            query,
            retrieval["context"],
            retrieval["sources"],
            system_prompt=system_prompt or NAVIGATOR_PROMPT,
        )

        return {
            "query": query,
            "answer": llm_result["answer"],
            "sources": llm_result["sources"],
            "related_articles": llm_result["related_articles"],
            "suggested_next_steps": llm_result["suggested_next_steps"],
            "confidence": round(retrieval["best_similarity"] * 100),
            "retrieved_chunks": retrieval["retrieved_chunks"],
            "num_chunks": retrieval["num_chunks"],
            "low_confidence": retrieval["low_confidence"],
            "best_similarity": retrieval["best_similarity"],
        }

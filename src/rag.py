"""
RAG Pipeline

Orchestrates the complete Retrieval-Augmented Generation pipeline.
"""

from typing import Dict, Optional

from src.config import SYSTEM_PROMPT, TOP_K


class RAGPipeline:
    """Complete RAG pipeline: Query → Embed → Retrieve → Generate."""

    def __init__(self, embedder, retriever, llm) -> None:
        """
        Initialize RAG pipeline.

        Args:
            embedder: EmbeddingsGenerator instance
            retriever: Retriever instance
            llm: GeminiLLM instance
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

        llm_result = self.llm.generate_answer_with_sources(
            query,
            retrieval["context"],
            retrieval["sources"],
            system_prompt=system_prompt or SYSTEM_PROMPT,
        )

        return {
            "query": query,
            "answer": llm_result["answer"],
            "sources": llm_result["sources"],
            "retrieved_chunks": retrieval["retrieved_chunks"],
            "num_chunks": retrieval["num_chunks"],
        }

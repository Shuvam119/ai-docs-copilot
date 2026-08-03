"""
RAG Pipeline

Orchestrates the complete Retrieval-Augmented Generation pipeline.
"""

import logging
import time
from typing import Callable, Dict, List, Optional

from src.config import TOP_K
from src.context_builder import ContextBuilder

logger = logging.getLogger(__name__)


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
        self.context_builder = ContextBuilder()

    def answer(
        self,
        query: str,
        top_k: int = TOP_K,
        system_prompt: Optional[str] = None,
        audience: str = "End User",
        filters: Optional[Dict[str, str]] = None,
        conversation_history: Optional[List[Dict]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        """
        Answer a question using the RAG pipeline.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            system_prompt: Optional custom system prompt
            progress_callback: Optional callback invoked with a phase name
                ("searching", "generating", "complete") so callers can show
                progress feedback. Purely informational; retrieval and answer
                generation are unaffected when it is None.

        Returns:
            Dictionary with answer, sources, and retrieved chunks
        """
        overall_start = time.perf_counter()

        if progress_callback:
            progress_callback("searching")
        retrieval = self.retriever.retrieve_with_context(query, top_k=top_k, filters=filters)

        context_start = time.perf_counter()
        prompt = self.context_builder.build(query, retrieval, audience, conversation_history or [])
        logger.info(
            "Context building completed in %.3f seconds",
            time.perf_counter() - context_start,
        )

        llm_start = time.perf_counter()
        if progress_callback:
            progress_callback("generating")
        llm_result = self.llm.generate_navigation_response(
            query, prompt["user_prompt"],
            retrieval["sources"],
            system_prompt=system_prompt or prompt["system_prompt"],
        )
        logger.info(
            "LLM response completed in %.3f seconds",
            time.perf_counter() - llm_start,
        )

        confidence = self._confidence(retrieval)

        if progress_callback:
            progress_callback("complete")

        logger.info(
            "Total answer generation completed in %.3f seconds",
            time.perf_counter() - overall_start,
        )

        return {
            "query": query,
            "answer": llm_result["answer"],
            "sources": llm_result["sources"],
            "related_articles": llm_result["related_articles"],
            "suggested_next_steps": llm_result["suggested_next_steps"],
            "confidence": confidence,
            "confidence_label": self._confidence_label(confidence),
            "retrieved_chunks": retrieval["retrieved_chunks"],
            "num_chunks": retrieval["num_chunks"],
            "low_confidence": retrieval["low_confidence"],
            "best_similarity": retrieval["best_similarity"],
            "related_documents": retrieval["related_documents"],
        }

    @staticmethod
    def _confidence(retrieval: Dict) -> int:
        """Combine similarity, corroboration, and filter agreement into confidence."""
        similarity = retrieval["best_similarity"] * 0.65
        support = min(retrieval["num_chunks"] / 5, 1.0) * 0.20
        agreement = (sum(chunk.get("metadata_agreement", 1.0) for chunk in retrieval["retrieved_chunks"]) / max(retrieval["num_chunks"], 1)) * 0.15
        return round((similarity + support + agreement) * 100)

    @staticmethod
    def _confidence_label(score: int) -> str:
        return "High" if score >= 75 else "Medium" if score >= 50 else "Low"

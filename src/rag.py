"""
RAG Pipeline

Orchestrates the complete Retrieval-Augmented Generation pipeline.
"""

from typing import Dict, List, Optional


class RAGPipeline:
    """Complete RAG pipeline: Query → Embed → Retrieve → Generate."""

    def __init__(self, embedder, retriever, llm):
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
        top_k: int = 5,
        system_prompt: Optional[str] = None
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
        # Step 1: Retrieve relevant context
        retrieval = self.retriever.retrieve_with_context(query, top_k=top_k)

        # Step 2: Generate answer with Gemini
        llm_result = self.llm.generate_answer_with_sources(
            query,
            retrieval['context'],
            retrieval['sources']
        )

        # Step 3: Combine results
        return {
            "query": query,
            "answer": llm_result['answer'],
            "sources": llm_result['sources'],
            "retrieved_chunks": retrieval['retrieved_chunks'],
            "num_chunks": retrieval['num_chunks']
        }

    def answer_streaming(
        self,
        query: str,
        top_k: int = 5
    ):
        """
        Answer with streaming support (for future UI integration).

        Args:
            query: User's question
            top_k: Number of chunks to retrieve

        Yields:
            Streaming response tokens
        """
        # For now, return full response
        # TODO: Implement streaming in Gemini API
        result = self.answer(query, top_k=top_k)
        yield result

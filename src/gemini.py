"""
Gemini LLM Integration

Handles communication with Google Gemini API for generating grounded answers.
"""

import logging
import os
from typing import Dict, Optional

from dotenv import load_dotenv
from google import genai

from src.config import GEMINI_MODEL, SYSTEM_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)


class GeminiLLM:
    """Generates answers using Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL) -> None:
        """
        Initialize Gemini client.

        Args:
            api_key: Google Gemini API key (or set GEMINI_API_KEY env var)
            model: Gemini model identifier
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Set it in .env file or pass api_key parameter."
            )

        self.client = genai.Client(api_key=self.api_key)
        logger.info("Gemini LLM initialized (model: %s)", self.model)

    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate an answer based on query and context.

        Args:
            query: User's question
            context: Retrieved document context
            system_prompt: Optional custom system prompt

        Returns:
            Generated answer
        """
        prompt = system_prompt or SYSTEM_PROMPT
        full_prompt = (
            f"{prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
        )

        return response.text

    def generate_answer_with_sources(
        self,
        query: str,
        context: str,
        sources: list,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Generate answer and include source information.

        Args:
            query: User's question
            context: Retrieved document context
            sources: List of source file names
            system_prompt: Optional custom system prompt

        Returns:
            Dictionary with answer and sources
        """
        answer = self.generate_answer(
            query, context, system_prompt=system_prompt)

        return {
            "answer": answer,
            "sources": sources,
            "query": query,
        }

"""
Groq LLM Integration

Provides a provider-neutral interface backed by the Groq API.
"""

import json
import logging
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.config import LLM_MODEL, NAVIGATOR_PROMPT, SYSTEM_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """Generates answers using the Groq API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = LLM_MODEL,
    ) -> None:
        """
        Initialize Groq client.

        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            model: Groq model identifier
        """

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Set it in .env file or pass api_key parameter."
            )

        # Groq API keys start with "gsk_"
        if not self.api_key.startswith("gsk_"):
            raise ValueError(
                "GROQ_API_KEY does not appear to be valid.\n"
                "Create one at https://console.groq.com/keys"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        logger.info("Groq LLM initialized (model: %s)", self.model)

    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate an answer based on retrieved context.

        Args:
            query: User's question
            context: Retrieved document context
            system_prompt: Optional custom system prompt

        Returns:
            Generated answer
        """

        prompt = system_prompt or SYSTEM_PROMPT

        full_prompt = f"""
{prompt}

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{query}
""",
                },
            ],
        )

        return response.choices[0].message.content.strip()

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
            query=query,
            context=context,
            system_prompt=system_prompt,
        )

        return {
            "answer": answer,
            "sources": sources,
            "query": query,
        }

    def generate_navigation_response(
        self,
        query: str,
        context: str,
        sources: List[str],
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Generate a structured knowledge navigation response.

        Returns answer text plus related articles and suggested next steps.
        """
        prompt = system_prompt or NAVIGATOR_PROMPT
        source_list = ", ".join(sources) if sources else "none"

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question:\n{query}\n\n"
                        f"Available source documents: {source_list}"
                    ),
                },
            ],
        )

        raw = response.choices[0].message.content.strip()
        answer = raw
        related_articles: List[str] = []
        suggested_next_steps: List[str] = []

        try:
            parsed = json.loads(raw)
            answer = str(parsed.get("answer", "")).strip() or answer
            related_articles = self._normalize_string_list(
                parsed.get("related_articles", [])
            )
            suggested_next_steps = self._normalize_string_list(
                parsed.get("suggested_next_steps", [])
            )
        except json.JSONDecodeError:
            logger.warning("Navigation response was not valid JSON; using raw text")

        if not answer:
            answer = "The information could not be found."

        return {
            "answer": answer,
            "related_articles": related_articles,
            "suggested_next_steps": suggested_next_steps,
            "sources": sources,
            "query": query,
        }

    @staticmethod
    def _normalize_string_list(values: object, limit: int = 4) -> List[str]:
        """Return a cleaned list of non-empty strings."""
        if not isinstance(values, list):
            return []

        cleaned = [str(value).strip() for value in values if str(value).strip()]
        return cleaned[:limit]

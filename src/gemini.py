"""
Gemini LLM Integration

Handles communication with Google Gemini API for generating grounded answers.
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GeminiLLM:
    """Generates answers using Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client.

        Args:
            api_key: Google Gemini API key (or set GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Set it in .env file or pass api_key parameter"
            )

        # Import here to avoid dependency if not using Gemini
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel("gemini-pro")

        print("✅ Gemini LLM initialized")

    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
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
        if system_prompt is None:
            system_prompt = """You are a helpful documentation assistant.
            
Use ONLY the provided context to answer the question.

If the answer is not in the context, say:
"I don't have information about this in the available documents."

Always be accurate and cite relevant sections when possible."""

        # Build full prompt
        prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""

        # Call Gemini
        response = self.client.generate_content(prompt)

        return response.text

    def generate_answer_with_sources(
        self,
        query: str,
        context: str,
        sources: list
    ) -> Dict:
        """
        Generate answer and include source information.

        Args:
            query: User's question
            context: Retrieved document context
            sources: List of source file names

        Returns:
            Dictionary with answer and sources
        """
        answer = self.generate_answer(query, context)

        return {
            "answer": answer,
            "sources": sources,
            "query": query
        }

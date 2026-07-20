"""Structured context engineering prompt assembly."""

from __future__ import annotations

from typing import Dict, List

from src.config import NAVIGATOR_PROMPT


class ContextBuilder:
    """Builds a traceable, structured prompt rather than sending raw chunks."""

    def build(self, question: str, retrieval: Dict, audience: str, history: List[Dict]) -> Dict[str, str]:
        """Return system prompt and user content for the LLM."""
        evidence = []
        for item in retrieval["retrieved_chunks"]:
            meta = item["metadata"]
            evidence.append(
                "[DOCUMENT]\n"
                f"Title: {meta.get('title', meta.get('filename'))}\nProduct: {meta.get('product')} | "
                f"Version: {meta.get('version')} | Type: {meta.get('document_type')}\n"
                f"Audience: {meta.get('audience')} | Department: {meta.get('department')}\n"
                f"Content:\n{item['text']}"
            )
        audience_guidance = {
            "End User": "Use simple language and clear steps.",
            "Support Engineer": "Prioritize diagnostics, causes, and safe troubleshooting steps.",
            "Technical Writer": "Identify gaps, duplication, ambiguity, and concrete documentation improvements.",
            "Administrator": "Focus on configuration, access, operations, and risk.",
            "Product Manager": "Focus on product impact, versions, and decisions.",
        }.get(audience, "Adapt to the selected audience.")
        recent_history = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-6:]) or "None"
        return {
            "system_prompt": f"{NAVIGATOR_PROMPT}\n\nSelected audience: {audience}. {audience_guidance}",
            "user_prompt": (
                "ENTERPRISE CONTEXT\n"
                f"Audience: {audience}\nConversation history:\n{recent_history}\n\n"
                f"Retrieved evidence:\n{'\n\n---\n\n'.join(evidence)}\n\n"
                f"Related documents: {', '.join(retrieval.get('related_documents', [])) or 'None'}\n"
                f"Question: {question}"
            ),
        }

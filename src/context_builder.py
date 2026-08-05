"""Structured context engineering prompt assembly."""

from __future__ import annotations

from typing import Dict, List

from src.config import NAVIGATOR_PROMPT

_AUDIENCE_GUIDANCE = {
    "End User": (
        "Answer in plain, beginner-friendly language. Use numbered steps for any procedure, "
        "explain technical terms in simple words on first use, avoid jargon and internal "
        "abbreviations, and do not assume prior product knowledge."
    ),
    "Support Engineer": (
        "Answer as a support engineer running diagnostics. Structure the answer with the "
        "headings Possible Cause, Resolution, and Verification whenever the evidence supports "
        "them. Emphasize root causes, troubleshooting steps, relevant logs, known issues, and "
        "safe fixes rather than writing an end-user tutorial."
    ),
    "Technical Writer": (
        "Answer as a technical writer reviewing documentation quality. Evaluate the retrieved "
        "documents and call out documentation gaps, missing prerequisites, ambiguous wording, "
        "content duplication, consistency issues, and outdated versions. Suggest concrete "
        "improvements and note whether any content should be moved to another guide."
    ),
    "Administrator": (
        "Answer as a system administrator responsible for operations. Focus on configuration, "
        "permissions, deployment, security, operations, and maintenance. Provide operational "
        "guidance, note any setup or compliance considerations, and call out steps a deployer "
        "must complete."
    ),
    "Product Manager": (
        "Answer as a product manager. Focus on feature impact, version differences, release "
        "implications, business context, and roadmap considerations. Highlight what changes "
        "between versions and the customer value or trade-offs involved."
    ),
}
_DEFAULT_AUDIENCE_GUIDANCE = "Adapt to the selected audience."

# The system prompt depends only on the selected audience (the navigator
# prompt is static), so it is assembled once per audience and reused for
# every subsequent question instead of being re-formatted each call.
_system_prompt_cache: Dict[str, str] = {}


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
        system_prompt = _system_prompt_cache.get(audience)
        if system_prompt is None:
            audience_guidance = _AUDIENCE_GUIDANCE.get(
                audience, _DEFAULT_AUDIENCE_GUIDANCE)
            audience_instructions = (
                f"Audience: {audience}\n"
                f"Instructions: {audience_guidance}\n"
                "Apply these instructions to every answer. Do not require the user to mention the "
                "audience role; the selected audience always shapes the wording, tone, detail level, "
                "structure, and recommendations."
            )
            system_prompt = f"{NAVIGATOR_PROMPT}\n\n{audience_instructions}"
            _system_prompt_cache[audience] = system_prompt
        recent_history = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history[-6:]) or "None"
        retrieved_evidence = "\n\n---\n\n".join(evidence)
        return {
            "system_prompt": system_prompt,
            "user_prompt": (
                "ENTERPRISE CONTEXT\n"
                f"Conversation history:\n{recent_history}\n\n"
                f"Retrieved evidence:\n{retrieved_evidence}\n\n"
                f"Related documents: {', '.join(retrieval.get('related_documents', [])) or 'None'}\n"
                f"Question: {question}"
            ),
        }

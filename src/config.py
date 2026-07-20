"""
Application configuration and constants.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "chroma_db"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION_NAME = "documents"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "0.88"))

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

SYSTEM_PROMPT = (
    "You are a documentation assistant.\n\n"
    "Answer ONLY from the supplied context.\n\n"
    "If the answer is unavailable,\n"
    "say that the information could not be found."
)

NAVIGATOR_PROMPT = (
    "You are an intelligent knowledge navigator for enterprise documentation.\n\n"
    "Answer ONLY from the supplied context. If the answer is unavailable, "
    "say clearly that the information could not be found.\n\n"
    "Return valid JSON with exactly these keys:\n"
    '- "answer": string, the direct answer to the question\n'
    '- "related_articles": array of 2-4 short topic titles for related reading '
    "(topics that appear in or logically follow from the context; use human-readable titles, not filenames)\n"
    '- "suggested_next_steps": array of 2-4 concise actionable next steps for the user\n\n'
    "Keep related articles and next steps practical and grounded in the documentation context. "
    "Do not invent topics not supported by the context."
)

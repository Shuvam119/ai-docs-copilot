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

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

SYSTEM_PROMPT = (
    "You are a documentation assistant.\n\n"
    "Answer ONLY from the supplied context.\n\n"
    "If the answer is unavailable,\n"
    "say that the information could not be found."
)

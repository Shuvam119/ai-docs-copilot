"""
Application configuration and constants.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "chroma_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "documents"
GEMINI_MODEL = "gemini-2.0-flash"

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

SYSTEM_PROMPT = (
    "You are a documentation assistant.\n\n"
    "Answer ONLY from the supplied context.\n\n"
    "If the answer is unavailable,\n"
    "say that the information could not be found."
)

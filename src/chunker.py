"""
Text Chunking

Breaks large documents into semantic chunks suitable for embeddings.
Uses LangChain's RecursiveCharacterTextSplitter.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict


class DocumentChunker:
    """Chunks documents into overlapping passages."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Initialize the chunker.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk a single document.

        Args:
            document: Document dict from load_document with 'text' and 'metadata'

        Returns:
            List of chunk dicts with chunk text, metadata, and chunk index
        """
        text = document["text"]
        metadata = document["metadata"]

        # Split the text
        chunk_texts = self.splitter.split_text(text)

        # Create chunk objects
        chunks = []
        for chunk_idx, chunk_text in enumerate(chunk_texts, 1):
            chunk = {
                "text": chunk_text,
                "metadata": {
                    **metadata,  # Include original metadata
                    "chunk_index": chunk_idx,
                    "chunk_size": len(chunk_text),
                    "total_chunks": len(chunk_texts)
                }
            }
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dicts

        Returns:
            List of chunk dicts from all documents
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        return all_chunks

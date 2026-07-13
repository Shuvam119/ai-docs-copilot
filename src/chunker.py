"""
Text Chunking

Breaks large documents into semantic chunks suitable for embeddings.
Uses LangChain's RecursiveCharacterTextSplitter.
"""

from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


class DocumentChunker:
    """Chunks documents into overlapping passages."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
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
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk a single document.

        Args:
            document: Document dict from load_document with 'text' and 'metadata'

        Returns:
            List of chunk dicts with chunk text and metadata
        """
        text = document["text"]
        metadata = document["metadata"]
        filename = metadata["filename"]
        document_type = metadata.get("type", "unknown")

        chunk_texts = self.splitter.split_text(text)
        total_chunks = len(chunk_texts)

        chunks: List[Dict] = []
        for chunk_idx, chunk_text in enumerate(chunk_texts, 1):
            chunk_id = f"{filename}_{chunk_idx}"
            chunk = {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "document_type": document_type,
                    "chunk_index": chunk_idx,
                    "chunk_size": len(chunk_text),
                    "total_chunks": total_chunks,
                },
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
        all_chunks: List[Dict] = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        return all_chunks

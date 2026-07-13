"""
Embeddings Generator

Converts text chunks into vector embeddings using sentence-transformers.
Uses the BAAI/bge-small-en-v1.5 model.
"""

import logging
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    """Generates embeddings for text chunks."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        """
        Initialize the embeddings generator.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)

        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(
            "Embedding model loaded (dimension: %d)",
            self.embedding_dim,
        )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embedding vectors
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of chunk dicts with 'text' key

        Returns:
            List of chunks with 'embedding' key added
        """
        chunk_texts = [chunk["text"] for chunk in chunks]

        logger.info("Generating embeddings for %d chunk(s)", len(chunks))
        embeddings = self.embed_texts(chunk_texts)

        chunks_with_embeddings: List[Dict] = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy["embedding"] = embedding
            chunks_with_embeddings.append(chunk_copy)

        logger.info("Generated %d embedding(s)", len(chunks_with_embeddings))
        return chunks_with_embeddings

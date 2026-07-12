"""
Embeddings Generator

Converts text chunks into vector embeddings using sentence-transformers.
Uses the BAAI/bge-small-en-v1.5 model.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np


class EmbeddingsGenerator:
    """Generates embeddings for text chunks."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the embeddings generator.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        print(f"📥 Loading embedding model: {model_name}")
        print("   (First run will download ~130-150 MB)")

        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"✅ Model loaded successfully!")
        print(f"   Embedding dimension: {self.embedding_dim}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of chunk dicts with 'text' key

        Returns:
            List of chunks with 'embedding' key added
        """
        chunk_texts = [chunk["text"] for chunk in chunks]

        print(f"\n🔄 Generating embeddings for {len(chunks)} chunk(s)...")
        embeddings = self.embed_texts(chunk_texts)

        # Add embedding to each chunk
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy["embedding"] = embedding
            chunks_with_embeddings.append(chunk_copy)

        print(f"✅ Generated {len(chunks_with_embeddings)} embeddings")

        return chunks_with_embeddings

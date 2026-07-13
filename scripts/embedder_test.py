"""Test embedding generation."""

import numpy as np

from test_utils import setup_project


def main() -> None:
    setup_project()

    from src.chunker import DocumentChunker
    from src.config import RAW_DATA_DIR
    from src.embedder import EmbeddingsGenerator
    from src.load_document import load_documents_from_directory

    print("=" * 60)
    print("Embeddings Generator Test")
    print("=" * 60)

    documents = load_documents_from_directory(str(RAW_DATA_DIR))
    if not documents:
        print("No documents loaded. Run: python scripts/create_samples.py")
        return

    print(f"Loaded {len(documents)} document(s)")

    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunk(s)")

    embedder = EmbeddingsGenerator()
    chunks_with_embeddings = embedder.embed_chunks(chunks)

    first_chunk = chunks_with_embeddings[0]
    emb = first_chunk["embedding"]

    print("\nEmbedding Statistics:")
    print(f"  Shape: {emb.shape}")
    print(f"  Type: {emb.dtype}")
    print(f"  Sample: {emb[:5]}")

    all_embs = np.array([chunk["embedding"] for chunk in chunks_with_embeddings])
    print(f"\n  Total: {len(chunks_with_embeddings)}")
    print(f"  Dimension: {all_embs.shape[1]}")
    print(f"  Min: {all_embs.min():.4f}")
    print(f"  Max: {all_embs.max():.4f}")

    emb1 = chunks_with_embeddings[0]["embedding"]
    emb2 = (
        chunks_with_embeddings[-1]["embedding"]
        if len(chunks_with_embeddings) > 1
        else emb1
    )
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"  Similarity (first vs last): {similarity:.4f}")

    print("\n" + "=" * 60)
    print("Embeddings Generator Test: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

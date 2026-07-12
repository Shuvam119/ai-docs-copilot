"""Test Embeddings - Phase 3"""
import sys
from pathlib import Path


def main():
    import os
    import numpy as np

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    from src.load_document import load_documents_from_directory
    from src.chunker import DocumentChunker
    from src.embedder import EmbeddingsGenerator

    print("=" * 60)
    print("Embeddings Generator Test")
    print("=" * 60)

    print("\nPhase 3: Embeddings")
    raw_data_dir = project_root / "data" / "raw"
    documents = load_documents_from_directory(str(raw_data_dir))

    if not documents:
        print("No documents loaded")
        return

    print(f"Loaded {len(documents)} documents")

    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("\n" + "-" * 60)
    embedder = EmbeddingsGenerator()

    print("\n" + "-" * 60)
    chunks_with_embeddings = embedder.embed_chunks(chunks)

    print("\n" + "-" * 60)
    print("\nEmbedding Statistics:")

    first_chunk = chunks_with_embeddings[0]
    emb = first_chunk["embedding"]

    print(f"  Shape: {emb.shape}")
    print(f"  Type: {emb.dtype}")
    print(f"  Sample: {emb[:5]}")

    all_embs = np.array([c["embedding"] for c in chunks_with_embeddings])
    print(f"\n  Total: {len(chunks_with_embeddings)}")
    print(f"  Dimension: {all_embs.shape[1]}")
    print(f"  Min: {all_embs.min():.4f}")
    print(f"  Max: {all_embs.max():.4f}")

    emb1 = chunks_with_embeddings[0]["embedding"]
    emb2 = chunks_with_embeddings[-1]["embedding"] if len(
        chunks_with_embeddings) > 1 else emb1
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"  Similarity: {sim:.4f}")

    print("\n" + "=" * 60)
    print("Phase 3 Test: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

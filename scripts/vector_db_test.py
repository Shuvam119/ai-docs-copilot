"""Test Vector Database - Phase 4"""
import sys
from pathlib import Path


def main():
    import os

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    from src.load_document import load_documents_from_directory
    from src.chunker import DocumentChunker
    from src.embedder import EmbeddingsGenerator
    from src.vector_db import VectorStore

    print("=" * 60)
    print("Vector Database Test")
    print("=" * 60)

    # Load, chunk, embed
    print("\nPhase 4: Vector Database")
    raw_data_dir = project_root / "data" / "raw"
    documents = load_documents_from_directory(str(raw_data_dir))

    print(f"Loaded {len(documents)} documents")

    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("\nEmbedding chunks...")
    embedder = EmbeddingsGenerator()
    chunks_with_embeddings = embedder.embed_chunks(chunks)

    # Initialize vector store
    print("\n" + "-" * 60)
    vectorstore_path = project_root / "vectorstore"
    vector_store = VectorStore(
        str(vectorstore_path), collection_name="documents")

    # Add chunks to vector store
    print("\nAdding chunks to vector store...")
    added = vector_store.add_chunks(chunks_with_embeddings)
    print(f"Added {added} chunks")

    # Get stats
    print("\n" + "-" * 60)
    stats = vector_store.get_stats()
    print(f"\nVector Store Statistics:")
    print(f"  Collection: {stats['collection']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Path: {stats['vectorstore_path']}")

    # Test retrieval with first chunk's embedding
    print("\n" + "-" * 60)
    print("\nTesting Retrieval:")
    query_embedding = chunks_with_embeddings[0]["embedding"].tolist()

    results = vector_store.search(query_embedding, top_k=3)
    print(f"Top 3 similar chunks for query:")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. {result['metadata']['filename']}")
        print(f"     Similarity: {result['similarity']:.4f}")
        print(f"     Text: {result['text'][:100]}...")

    print("\n" + "=" * 60)
    print("Phase 4 Test: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Test Retriever - Phase 5"""
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
    from src.retriever import Retriever

    print("=" * 60)
    print("Retriever Test")
    print("=" * 60)

    # Load, chunk, embed
    print("\nPhase 5: Retrieval")
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

    # Add chunks
    print("\nAdding chunks to vector store...")
    added = vector_store.add_chunks(chunks_with_embeddings)
    print(f"Added {added} chunks")

    # Initialize retriever
    print("\n" + "-" * 60)
    print("\nInitializing Retriever...")
    retriever = Retriever(embedder, vector_store)
    print("✅ Retriever ready")

    # Test queries
    print("\n" + "-" * 60)
    print("\nTesting Retrieval with Sample Queries:")

    test_queries = [
        "How do I onboard a vendor?",
        "What is the API guide?",
        "Tell me about authentication"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("-" * 60)

        # Retrieve with context
        retrieval_result = retriever.retrieve_with_context(query, top_k=3)

        print(f"Found {retrieval_result['num_chunks']} relevant chunks")
        print(f"Sources: {', '.join(retrieval_result['sources'])}")

        print(f"\nRetrieved Chunks:")
        for i, chunk in enumerate(retrieval_result['retrieved_chunks'], 1):
            sim = chunk['similarity']
            text = chunk['text'][:80]
            filename = chunk['metadata']['filename']
            print(f"\n  {i}. [{filename}] (Similarity: {sim:.4f})")
            print(f"     {text}...")

    print("\n" + "=" * 60)
    print("Phase 5 Test: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

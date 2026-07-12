"""Test Gemini Integration - Phase 6"""
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
    from src.gemini import GeminiLLM

    print("=" * 60)
    print("Gemini Integration Test")
    print("=" * 60)

    # Setup pipeline
    print("\nPhase 6: Gemini Integration")

    raw_data_dir = project_root / "data" / "raw"
    documents = load_documents_from_directory(str(raw_data_dir))
    print(f"Loaded {len(documents)} documents")

    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("\nEmbedding chunks...")
    embedder = EmbeddingsGenerator()
    chunks_with_embeddings = embedder.embed_chunks(chunks)

    vectorstore_path = project_root / "vectorstore"
    vector_store = VectorStore(
        str(vectorstore_path), collection_name="documents")
    print("Adding chunks to vector store...")
    vector_store.add_chunks(chunks_with_embeddings)

    retriever = Retriever(embedder, vector_store)
    print("Retriever initialized")

    # Initialize Gemini
    print("\n" + "-" * 60)
    print("Initializing Gemini LLM...")
    try:
        llm = GeminiLLM()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nSkipping Gemini test (API key not configured)")
        print("To test, add GEMINI_API_KEY to .env file")
        return

    # Test queries
    print("\n" + "-" * 60)
    print("Testing RAG with Gemini:\n")

    test_queries = [
        "How do I onboard a vendor?",
        "What does the API guide cover?"
    ]

    for query in test_queries:
        print(f"{'='*60}")
        print(f"Query: {query}\n")

        # Retrieve context
        retrieval = retriever.retrieve_with_context(query, top_k=3)

        print(
            f"Retrieved {retrieval['num_chunks']} chunks from: {', '.join(retrieval['sources'])}\n")

        # Generate answer
        print("Generating answer with Gemini...\n")
        result = llm.generate_answer_with_sources(
            query,
            retrieval['context'],
            retrieval['sources']
        )

        print(f"Answer:")
        print(result['answer'])
        print(f"\nSources: {', '.join(result['sources'])}")

    print("\n" + "=" * 60)
    print("Phase 6 Test: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

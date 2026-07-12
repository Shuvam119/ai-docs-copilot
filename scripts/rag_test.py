"""Test RAG Pipeline - Phase 7"""
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
    from src.rag import RAGPipeline

    print("=" * 60)
    print("RAG Pipeline Test")
    print("=" * 60)

    # Setup components
    print("\nPhase 7: RAG Pipeline")

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

    # Initialize LLM
    print("\n" + "-" * 60)
    print("Initializing LLM...")
    try:
        llm = GeminiLLM()
    except ValueError as e:
        print(f"LLM not available: {e}")
        return

    # Initialize RAG Pipeline
    print("\nInitializing RAG Pipeline...")
    rag = RAGPipeline(embedder, retriever, llm)
    print("✅ RAG Pipeline ready!\n")

    # Test queries
    print("=" * 60)
    print("Testing RAG Pipeline:\n")

    test_queries = [
        "How do I onboard a vendor?",
        "What is the API guide about?"
    ]

    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 60)

        result = rag.answer(query, top_k=3)

        print(f"\nAnswer:")
        print(result['answer'])
        print(f"\nSources: {', '.join(result['sources'])}")
        print(f"Retrieved chunks: {result['num_chunks']}")
        print()

    print("=" * 60)
    print("Phase 7 Test: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

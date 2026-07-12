"""
Build Vector Index

Orchestrates the complete pipeline to build the vector index from documents.
Usage: python scripts/build_index.py
"""

import sys
from pathlib import Path


def main():
    import os
    from datetime import datetime

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    from src.load_document import load_documents_from_directory
    from src.chunker import DocumentChunker
    from src.embedder import EmbeddingsGenerator
    from src.vector_db import VectorStore

    print("=" * 70)
    print(" " * 20 + "BUILD VECTOR INDEX")
    print("=" * 70)

    start_time = datetime.now()

    # Step 1: Load documents
    print("\n[1/4] Loading documents...")
    raw_data_dir = project_root / "data" / "raw"

    if not raw_data_dir.exists():
        print(f"ERROR: Data directory not found: {raw_data_dir}")
        return

    documents = load_documents_from_directory(str(raw_data_dir))

    if not documents:
        print("ERROR: No documents found in data/raw/")
        print("Please add PDF or DOCX files to data/raw/")
        return

    print(f"✅ Loaded {len(documents)} document(s)")
    for doc in documents:
        print(f"   - {doc['title']} ({len(doc['text'])} chars)")

    # Step 2: Chunk documents
    print("\n[2/4] Chunking documents...")
    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)

    print(f"✅ Created {len(chunks)} chunk(s)")

    # Step 3: Generate embeddings
    print("\n[3/4] Generating embeddings...")
    print("      (First run downloads model ~150MB)")

    embedder = EmbeddingsGenerator()
    chunks_with_embeddings = embedder.embed_chunks(chunks)

    # Step 4: Build vector store
    print("\n[4/4] Building vector store...")
    vectorstore_path = project_root / "vectorstore"
    vector_store = VectorStore(
        str(vectorstore_path), collection_name="documents")

    added = vector_store.add_chunks(chunks_with_embeddings)

    # Final summary
    print("\n" + "=" * 70)
    print("BUILD COMPLETE!")
    print("=" * 70)

    stats = vector_store.get_stats()
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\nSummary:")
    print(f"  Documents: {len(documents)}")
    print(f"  Chunks: {stats['total_chunks']}")
    print(f"  Embeddings: {added}")
    print(f"  Vector Store: {stats['vectorstore_path']}")
    print(f"  Time: {elapsed:.1f} seconds")

    print("\n✅ Your vector index is ready!")
    print("   Run: streamlit run ui/app.py")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

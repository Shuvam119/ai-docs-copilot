"""Test text chunking."""

from test_utils import setup_project


def main() -> None:
    setup_project()

    from src.chunker import DocumentChunker
    from src.config import RAW_DATA_DIR
    from src.load_document import load_documents_from_directory

    print("=" * 60)
    print("Document Chunking Test")
    print("=" * 60)

    documents = load_documents_from_directory(str(RAW_DATA_DIR))
    if not documents:
        print("No documents loaded. Run: python scripts/create_samples.py")
        return

    print(f"Loaded {len(documents)} document(s)")

    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunk(s)\n")

    for doc in documents:
        doc_title = doc["title"]
        doc_chunks = [
            chunk
            for chunk in chunks
            if chunk["metadata"]["filename"] == doc_title
        ]
        print(f"- {doc_title}: {len(doc_chunks)} chunk(s)")
        if doc_chunks:
            print(f"  First chunk: {doc_chunks[0]['text'][:120]}...")

    total_chunk_size = sum(len(chunk["text"]) for chunk in chunks)
    avg_chunk_size = total_chunk_size / len(chunks) if chunks else 0

    print(f"\nStatistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Total characters: {total_chunk_size:,}")
    print(f"  Average chunk size: {avg_chunk_size:.0f} characters")

    print("\n" + "=" * 60)
    print("Document Chunking Test: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

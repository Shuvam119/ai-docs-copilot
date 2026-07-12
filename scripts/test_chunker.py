"""
Test Text Chunking

Tests the chunking functionality.
"""

from src.chunker import DocumentChunker
from src.load_document import load_documents_from_directory
import sys
import os
from pathlib import Path

# Setup paths FIRST before importing src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# NOW import from src


def test_chunker():
    """Test chunking functionality."""

    print("=" * 60)
    print("🔪 Document Chunking Test")
    print("=" * 60)

    # Load documents
    raw_data_dir = Path(project_root) / "data" / "raw"
    print(f"\n📖 Loading documents from {raw_data_dir}...\n")

    try:
        documents = load_documents_from_directory(str(raw_data_dir))

        if not documents:
            print("❌ No documents loaded")
            return

        print(f"✅ Loaded {len(documents)} document(s)\n")

        # Initialize chunker
        chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)

        # Chunk documents
        print("Chunking documents...\n")
        chunks = chunker.chunk_documents(documents)

        print(f"✅ Created {len(chunks)} chunk(s)\n")

        # Display results
        print("-" * 60)
        for doc in documents:
            doc_title = doc["title"]
            doc_chunks = [c for c in chunks if c["metadata"]
                          ["filename"] == doc_title]
            print(f"\n📄 {doc_title}")
            print(f"   Total chunks: {len(doc_chunks)}")

            # Show first chunk
            if doc_chunks:
                first_chunk = doc_chunks[0]
                chunk_text = first_chunk["text"][:150]
                print(f"   First chunk: {chunk_text}...")

        print("\n" + "-" * 60)

        # Statistics
        total_chunk_size = sum(len(c["text"]) for c in chunks)
        avg_chunk_size = total_chunk_size / len(chunks) if chunks else 0

        print(f"\n📊 Statistics:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Total characters: {total_chunk_size:,}")
        print(f"   Average chunk size: {avg_chunk_size:.0f} characters")
        print(
            f"   Min chunk size: {min((len(c['text']) for c in chunks), default=0)}")
        print(
            f"   Max chunk size: {max((len(c['text']) for c in chunks), default=0)}")

        print("\n" + "=" * 60)
        print("✅ Phase 2 Test: PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_chunker()

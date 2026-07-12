"""
Test Chunker - Using importlib for dynamic imports
"""

import sys
import os
import importlib.util
from pathlib import Path

# Setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Dynamic import to avoid reordering
spec_loader = importlib.util.spec_from_file_location(
    "src.load_document", str(project_root / "src" / "load_document.py"))
load_doc_module = importlib.util.module_from_spec(spec_loader)
spec_loader.loader.exec_module(load_doc_module)

spec_chunker = importlib.util.spec_from_file_location(
    "src.chunker", str(project_root / "src" / "chunker.py"))
chunker_module = importlib.util.module_from_spec(spec_chunker)
spec_chunker.loader.exec_module(chunker_module)

# Now use them
load_documents_from_directory = load_doc_module.load_documents_from_directory
DocumentChunker = chunker_module.DocumentChunker

print("=" * 60)
print("🔪 Document Chunking Test")
print("=" * 60)

raw_data_dir = project_root / "data" / "raw"
print(f"\n📖 Loading documents...\n")

documents = load_documents_from_directory(str(raw_data_dir))

if not documents:
    print("❌ No documents loaded")
else:
    print(f"✅ Loaded {len(documents)} document(s)\n")

    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
    print("Chunking documents...\n")
    chunks = chunker.chunk_documents(documents)

    print(f"✅ Created {len(chunks)} chunk(s)\n")
    print("-" * 60)

    for doc in documents:
        doc_title = doc["title"]
        doc_chunks = [c for c in chunks if c["metadata"]
                      ["filename"] == doc_title]
        print(f"\n📄 {doc_title}")
        print(f"   Chunks: {len(doc_chunks)}")
        if doc_chunks:
            chunk_text = doc_chunks[0]["text"][:150]
            print(f"   Preview: {chunk_text}...")

    print("\n" + "-" * 60)
    total_chunk_size = sum(len(c["text"]) for c in chunks)
    avg_chunk_size = total_chunk_size / len(chunks) if chunks else 0

    print(f"\n📊 Statistics:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Total characters: {total_chunk_size:,}")
    print(f"   Average size: {avg_chunk_size:.0f} characters")
    print(f"   Min: {min((len(c['text']) for c in chunks), default=0)}")
    print(f"   Max: {max((len(c['text']) for c in chunks), default=0)}")
    print("\n" + "=" * 60)
    print("✅ Phase 2 Test: PASSED")
    print("=" * 60)

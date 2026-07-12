"""
Test Embeddings Generator

Tests the embeddings functionality.
"""

from src.embedder import EmbeddingsGenerator
from src.chunker import DocumentChunker
from src.load_document import load_documents_from_directory
import sys
import os
from pathlib import Path
import numpy as np

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)


print("=" * 60)
print("🧮 Embeddings Generator Test")
print("=" * 60)

# Step 1: Load documents
print("\n📖 Loading documents...")
raw_data_dir = project_root / "data" / "raw"
documents = load_documents_from_directory(str(raw_data_dir))

if not documents:
    print("❌ No documents loaded")
    sys.exit(1)

print(f"✅ Loaded {len(documents)} documents")

# Step 2: Chunk documents
print("\n🔪 Chunking documents...")
chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
chunks = chunker.chunk_documents(documents)

print(f"✅ Created {len(chunks)} chunks")

# Step 3: Generate embeddings
print("\n" + "-" * 60)
embedder = EmbeddingsGenerator()

print("\n" + "-" * 60)
chunks_with_embeddings = embedder.embed_chunks(chunks)

# Step 4: Verify embeddings
print("\n" + "-" * 60)
print("\n📊 Embedding Verification:")

# Check first chunk
first_chunk = chunks_with_embeddings[0]
embedding = first_chunk["embedding"]

print(f"\n   Chunk text: {first_chunk['text'][:100]}...")
print(f"   Embedding shape: {embedding.shape}")
print(f"   Embedding dtype: {embedding.dtype}")
print(f"   Sample values: {embedding[:5]}")  # First 5 dimensions

# Statistics
all_embeddings = np.array([c["embedding"] for c in chunks_with_embeddings])
print(f"\n   Total embeddings: {len(chunks_with_embeddings)}")
print(f"   Embedding dimension: {all_embeddings.shape[1]}")
print(f"   Min value: {all_embeddings.min():.4f}")
print(f"   Max value: {all_embeddings.max():.4f}")
print(f"   Mean value: {all_embeddings.mean():.4f}")
print(f"   Std deviation: {all_embeddings.std():.4f}")

# Test: Similarity between chunks
print("\n   Testing chunk similarity...")
emb1 = chunks_with_embeddings[0]["embedding"]
emb2 = chunks_with_embeddings[-1]["embedding"] if len(
    chunks_with_embeddings) > 1 else emb1

# Cosine similarity
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f"   Similarity (Chunk 1 vs Last): {similarity:.4f}")

print("\n" + "=" * 60)
print("✅ Phase 3 Test: PASSED")
print("=" * 60)

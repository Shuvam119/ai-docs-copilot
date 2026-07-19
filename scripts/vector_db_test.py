"""Test ChromaDB vector store operations."""

from test_utils import build_test_index, setup_project


def main() -> None:
    setup_project()

    print("=" * 60)
    print("Vector Database Test")
    print("=" * 60)

    try:
        builder, stats = build_test_index(rebuild=True)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print("Run: python scripts/create_samples.py")
        return

    store_stats = builder.get_stats()
    print(f"\nVector Store Statistics:")
    print(f"  Collection: {store_stats['collection']}")
    print(f"  Documents: {store_stats['document_count']}")
    print(f"  Total chunks: {store_stats['total_chunks']}")
    print(f"  Path: {store_stats['vectorstore_path']}")

    query = "API authentication"
    query_embedding = builder.embedder.embed_text(query, is_query=True).tolist()
    results = builder.vector_store.search(query_embedding, top_k=3)

    print(f"\nTop 3 similar chunks for query: '{query}'")
    for idx, result in enumerate(results, 1):
        print(f"\n  {idx}. {result['metadata']['filename']}")
        print(f"     Similarity: {result['similarity']:.4f}")
        print(f"     Text: {result['text'][:100]}...")

    print("\n" + "=" * 60)
    print("Vector Database Test: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

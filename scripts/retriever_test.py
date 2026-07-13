"""Test semantic retrieval from the vector store."""

from test_utils import build_test_index, setup_project


def main() -> None:
    setup_project()

    from src.retriever import Retriever

    print("=" * 60)
    print("Retriever Test")
    print("=" * 60)

    try:
        builder, stats = build_test_index(rebuild=True)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print("Run: python scripts/create_samples.py")
        return

    print(f"Indexed {stats.document_count} document(s), {stats.chunk_count} chunk(s)")

    retriever = Retriever(builder.embedder, builder.vector_store)

    test_queries = [
        "How do I onboard a vendor?",
        "What is the API guide?",
        "Tell me about authentication",
    ]

    print("\nTesting retrieval:\n")
    for query in test_queries:
        print("=" * 60)
        print(f"Query: {query}")
        print("-" * 60)

        retrieval_result = retriever.retrieve_with_context(query, top_k=3)

        print(f"Found {retrieval_result['num_chunks']} relevant chunk(s)")
        print(f"Sources: {', '.join(retrieval_result['sources'])}")

        for idx, chunk in enumerate(retrieval_result["retrieved_chunks"], 1):
            filename = chunk["metadata"]["filename"]
            similarity = chunk["similarity"]
            preview = chunk["text"][:80]
            print(f"\n  {idx}. [{filename}] (Similarity: {similarity:.4f})")
            print(f"     {preview}...")

    print("\n" + "=" * 60)
    print("Retriever Test: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

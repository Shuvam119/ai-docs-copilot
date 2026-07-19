"""Test RAG Pipeline end-to-end."""

from test_utils import build_test_index, setup_project


def main() -> None:
    setup_project()

    print("=" * 60)
    print("RAG Pipeline Test")
    print("=" * 60)

    try:
        builder, stats = build_test_index(rebuild=True)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print("Run: python scripts/create_samples.py")
        return

    print(f"Indexed {stats.document_count} document(s), {stats.chunk_count} chunk(s)")

    try:
        rag = builder.create_rag_pipeline()
    except ValueError as exc:
        print(f"LLM not available: {exc}")
        print("Add GROQ_API_KEY to .env to run this test.")
        return

    test_queries = [
        "How do I onboard a vendor?",
        "What is the API guide about?",
    ]

    print("\nTesting RAG Pipeline:\n")
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 60)

        result = rag.answer(query, top_k=3)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {', '.join(result['sources'])}")
        print(f"Retrieved chunks: {result['num_chunks']}\n")

    print("=" * 60)
    print("RAG Pipeline Test: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

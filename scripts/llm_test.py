"""Test LLM integration with retrieved context."""

from test_utils import build_test_index, setup_project


def main() -> None:
    setup_project()

    from src.llm import LLMClient

    print("=" * 60)
    print("LLM Integration Test")
    print("=" * 60)

    try:
        builder, stats = build_test_index(rebuild=True)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print("Run: python scripts/create_samples.py")
        return

    print(f"Indexed {stats.document_count} document(s), {stats.chunk_count} chunk(s)")

    try:
        llm = LLMClient()
    except ValueError as exc:
        print(f"Error: {exc}")
        print("Add GROQ_API_KEY to .env to run this test.")
        return

    from src.retriever import Retriever

    retriever = Retriever(builder.embedder, builder.vector_store)

    test_queries = [
        "How do I onboard a vendor?",
        "What does the API guide cover?",
    ]

    print("\nTesting LLM with retrieved context:\n")
    for query in test_queries:
        print("=" * 60)
        print(f"Query: {query}\n")

        retrieval = retriever.retrieve_with_context(query, top_k=3)
        print(
            f"Retrieved {retrieval['num_chunks']} chunk(s) from: "
            f"{', '.join(retrieval['sources'])}\n"
        )

        result = llm.generate_answer_with_sources(
            query,
            retrieval["context"],
            retrieval["sources"],
        )

        print(f"Answer:\n{result['answer']}")
        print(f"\nSources: {', '.join(result['sources'])}\n")

    print("=" * 60)
    print("LLM Integration Test: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

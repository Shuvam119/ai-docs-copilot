"""Test document loaders for PDF and DOCX files."""

from test_utils import setup_project


def main() -> None:
    setup_project()

    from src.config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS
    from src.load_document import load_documents_from_directory

    print("=" * 60)
    print("Document Loader Test")
    print("=" * 60)
    print(f"\nScanning: {RAW_DATA_DIR}\n")

    if not RAW_DATA_DIR.exists():
        print(f"Directory not found: {RAW_DATA_DIR}")
        return

    all_files = list(RAW_DATA_DIR.iterdir())
    if not all_files:
        print("No files found in data/raw/")
        print("Run: python scripts/create_samples.py")
        return

    load_result = load_documents_from_directory(str(RAW_DATA_DIR))
    documents = load_result.documents
    if not documents:
        print("No supported documents found (.pdf or .docx)")
        print(f"Found files: {[path.name for path in all_files]}")
        return

    if load_result.failed_files:
        print(f"Failed files: {load_result.failed_files}")
    if load_result.empty_files:
        print(f"Empty files: {load_result.empty_files}")

    print(f"Successfully loaded {len(documents)} document(s)\n")

    total_chars = 0
    for idx, doc in enumerate(documents, 1):
        title = doc["title"]
        doc_type = doc["metadata"].get("type", "unknown")
        char_count = len(doc["text"])
        total_chars += char_count

        print(f"{idx}. {title} ({doc_type.upper()})")
        print(f"   Characters: {char_count:,}")
        print(f"   Source: {doc['metadata']['source']}")
        if doc_type == "pdf":
            print(f"   Pages: {doc['metadata'].get('pages', 'n/a')}")
        if doc_type == "docx":
            print(f"   Paragraphs: {doc['metadata'].get('paragraphs', 'n/a')}")
        print()

    print("-" * 60)
    print(f"Total documents: {len(documents)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    print("=" * 60)
    print("Document Loader Test: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

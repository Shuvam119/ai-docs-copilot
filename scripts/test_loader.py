"""
Test Document Loaders

Tests the document loading functionality for PDF and DOCX files.
"""

from src.load_document import load_documents_from_directory
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)


def test_loaders():
    """Test loading all documents from data/raw directory."""

    raw_data_dir = Path(__file__).parent.parent / "data" / "raw"

    print("=" * 60)
    print("📄 Document Loader Test")
    print("=" * 60)
    print(f"\nScanning: {raw_data_dir}\n")

    if not raw_data_dir.exists():
        print(f"❌ Directory not found: {raw_data_dir}")
        return

    # Check for any files
    all_files = list(raw_data_dir.iterdir())
    if not all_files:
        print("⚠️  No files found in data/raw/")
        print("\n💡 Tip: Add PDF and DOCX files to data/raw/ to test the loaders")
        return

    # Load documents
    try:
        documents = load_documents_from_directory(str(raw_data_dir))

        if not documents:
            print("❌ No supported documents found (.pdf or .docx)")
            print(f"Found files: {[f.name for f in all_files]}")
            return

        print(f"✅ Successfully loaded {len(documents)} document(s)\n")

        # Display info for each document
        total_chars = 0
        for idx, doc in enumerate(documents, 1):
            title = doc["title"]
            text = doc["text"]
            char_count = len(text)
            doc_type = doc["metadata"]["type"]

            total_chars += char_count

            print(f"{idx}. {title} ({doc_type.upper()})")
            print(f"   Characters: {char_count:,}")
            print(f"   Source: {doc['metadata']['source']}")
            if 'pages' in doc['metadata']:
                print(f"   Pages: {doc['metadata']['pages']}")
            if 'paragraphs' in doc['metadata']:
                print(f"   Paragraphs: {doc['metadata']['paragraphs']}")
            print()

        print("-" * 60)
        print(f"Total documents: {len(documents)}")
        print(f"Total characters: {total_chars:,}")
        print("=" * 60)
        print("✅ Phase 1 Test: PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_loaders()

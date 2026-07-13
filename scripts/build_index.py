"""
Build Vector Index

Orchestrates the complete pipeline to build the vector index from documents.
Usage: python scripts/build_index.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    """Build the vector index from documents in data/raw/."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from src.config import RAW_DATA_DIR
    from src.index_builder import IndexBuilder

    print("=" * 70)
    print(" " * 20 + "BUILD VECTOR INDEX")
    print("=" * 70)

    start_time = datetime.now()

    if not RAW_DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {RAW_DATA_DIR}")
        return

    try:
        builder = IndexBuilder()
        stats = builder.build(rebuild=True)
        store_stats = builder.get_stats()
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return

    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("BUILD COMPLETE!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Documents: {stats.document_count}")
    print(f"  Chunks: {stats.chunk_count}")
    print(f"  Vector Store: {store_stats['vectorstore_path']}")
    print(f"  Time: {elapsed:.1f} seconds")
    print("\nYour vector index is ready.")
    print("Run: streamlit run ui/app.py")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

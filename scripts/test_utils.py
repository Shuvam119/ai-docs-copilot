"""
Shared helpers for CLI test scripts.
"""

import os
import sys
from pathlib import Path


def setup_project() -> Path:
    """Add project root to sys.path and set working directory."""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    return project_root


def build_test_index(rebuild: bool = True):
    """
    Build a test index from documents in data/raw/.

    Args:
        rebuild: Whether to clear the existing collection first

    Returns:
        Tuple of (IndexBuilder, IndexStats)

    Raises:
        ValueError: If no documents are available
    """
    from src.index_builder import IndexBuilder

    builder = IndexBuilder()
    stats = builder.build(rebuild=rebuild)
    return builder, stats

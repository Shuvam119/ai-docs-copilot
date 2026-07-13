# -*- coding: utf-8 -*-
"""
Streamlit Web App - AI Documentation Copilot
"""

import logging
import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS, TOP_K
from src.index_builder import IndexBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Docs Copilot",
    page_icon="📚",
    layout="wide",
)

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_index_builder() -> IndexBuilder:
    """Return a cached IndexBuilder instance."""
    return IndexBuilder()


def save_uploaded_files(uploaded_files) -> list[str]:
    """
    Save uploaded files to data/raw/.

    Args:
        uploaded_files: Streamlit UploadedFile objects

    Returns:
        List of saved filenames
    """
    saved: list[str] = []
    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        destination = RAW_DATA_DIR / uploaded_file.name
        destination.write_bytes(uploaded_file.getbuffer())
        saved.append(uploaded_file.name)
        logger.info("Saved uploaded file: %s", uploaded_file.name)

    return saved


def rebuild_index(builder: IndexBuilder) -> bool:
    """
    Rebuild the vector index and refresh the RAG pipeline.

    Args:
        builder: IndexBuilder instance

    Returns:
        True if rebuild succeeded
    """
    try:
        with st.spinner("Rebuilding index..."):
            builder.build(rebuild=True)
            st.session_state.rag_pipeline = builder.create_rag_pipeline()
            st.session_state.index_stats = builder.get_stats()
            st.session_state.index_ready = True
        return True
    except ValueError as exc:
        st.error(str(exc))
        st.session_state.rag_pipeline = None
        st.session_state.index_ready = False
        st.session_state.index_stats = builder.get_stats()
        return False
    except Exception as exc:
        logger.exception("Index rebuild failed")
        st.error(f"Failed to rebuild index: {exc}")
        return False


def load_existing_pipeline(builder: IndexBuilder) -> None:
    """Load an existing index into the RAG pipeline if available."""
    stats = builder.get_stats()
    st.session_state.index_stats = stats

    if stats["total_chunks"] == 0:
        st.session_state.rag_pipeline = None
        st.session_state.index_ready = False
        return

    try:
        st.session_state.rag_pipeline = builder.create_rag_pipeline()
        st.session_state.index_ready = True
    except ValueError as exc:
        st.session_state.rag_pipeline = None
        st.session_state.index_ready = False
        logger.warning("Could not load pipeline: %s", exc)


def render_retrieved_chunks(chunks: list[dict]) -> None:
    """Render expandable retrieved chunk details."""
    with st.expander("Retrieved Chunks", expanded=False):
        if not chunks:
            st.write("No chunks retrieved.")
            return

        for idx, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "unknown")
            chunk_id = metadata.get("chunk_id", chunk.get("id", f"chunk_{idx}"))
            similarity = chunk.get("similarity", 0.0)

            st.markdown(f"**Chunk {idx}** — `{filename}` (`{chunk_id}`)")
            st.caption(f"Similarity: {similarity:.3f}")
            st.text(chunk.get("text", ""))
            if idx < len(chunks):
                st.divider()


# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "index_stats" not in st.session_state:
    st.session_state.index_stats = {
        "document_count": 0,
        "total_chunks": 0,
    }
if "pipeline_loaded" not in st.session_state:
    builder = get_index_builder()
    load_existing_pipeline(builder)
    st.session_state.pipeline_loaded = True


builder = get_index_builder()
stats = st.session_state.index_stats

# Sidebar
with st.sidebar:
    st.header("Index")

    st.metric("Indexed Documents", stats.get("document_count", 0))
    st.metric("Indexed Chunks", stats.get("total_chunks", 0))

    if st.button("Rebuild Index", use_container_width=True):
        if rebuild_index(builder):
            st.success("Index rebuilt successfully.")
            st.rerun()

    st.divider()

    top_k = st.slider("Retrieved Chunks (Top K)", 1, 10, TOP_K)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main page
st.title("AI Documentation Copilot")
st.markdown(
    "Upload PDF or DOCX documents, then ask questions grounded in your documentation."
)

st.subheader("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF or DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Save and Index Documents", use_container_width=True):
    saved = save_uploaded_files(uploaded_files)
    if saved:
        if rebuild_index(builder):
            st.success(
                f"Successfully indexed {len(saved)} file(s): "
                f"{', '.join(saved)}"
            )
            st.rerun()
    else:
        st.warning("No supported files were uploaded. Only PDF and DOCX are accepted.")

st.divider()

if not st.session_state.index_ready:
    st.info(
        "Upload PDF or DOCX files above to build the index, "
        "or click **Rebuild Index** in the sidebar if documents already exist in `data/raw/`."
    )
else:
    st.subheader("Ask a Question")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                if message.get("sources"):
                    st.markdown("**Sources:** " + ", ".join(message["sources"]))
                if message.get("retrieved_chunks"):
                    render_retrieved_chunks(message["retrieved_chunks"])

    query = st.chat_input("Ask about your documents...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    result = st.session_state.rag_pipeline.answer(query, top_k=top_k)
                    answer = result["answer"]
                    sources = result["sources"]
                    retrieved_chunks = result["retrieved_chunks"]

                    st.write(answer)

                    if sources:
                        st.markdown("**Sources:** " + ", ".join(sources))

                    render_retrieved_chunks(retrieved_chunks)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "retrieved_chunks": retrieved_chunks,
                        }
                    )
                except Exception as exc:
                    logger.exception("Question answering failed")
                    st.error(f"Error: {exc}")
                    st.session_state.messages.pop()

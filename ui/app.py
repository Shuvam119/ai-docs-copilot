"""Streamlit application for the AI Documentation Copilot."""

import html
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
    page_title="Knowledge Copilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

st.markdown(
    """
    <style>
        .stApp { background: #f7f8fc; }
        [data-testid="stSidebar"] { background: #111a33; }
        [data-testid="stSidebar"] * { color: #f6f8ff; }
        .hero {
            background: radial-gradient(circle at 85% 20%, #5f7cff 0, transparent 30%),
                        linear-gradient(120deg, #121c38 0%, #253b82 100%);
            border-radius: 22px;
            color: white;
            padding: 2.25rem 2.5rem;
            margin: 0.25rem 0 1.75rem;
            box-shadow: 0 16px 35px rgba(26, 47, 107, 0.18);
        }
        .eyebrow { color: #b8c6ff; font-size: .78rem; font-weight: 700; letter-spacing: .12em; }
        .hero h1 { font-size: 2.25rem; margin: .4rem 0 .55rem; }
        .hero p { color: #e4eaff; font-size: 1.04rem; margin: 0; max-width: 42rem; }
        .metric-card {
            background: white; border: 1px solid #e5e9f2; border-radius: 14px;
            padding: 1rem 1.1rem; margin-bottom: .65rem;
        }
        .metric-label { color: #667085; font-size: .77rem; font-weight: 700; text-transform: uppercase; }
        .metric-value { color: #172554; font-size: 1.55rem; font-weight: 750; margin-top: .2rem; }
        .source-card {
            border-left: 3px solid #6079ee; background: #f6f8ff; border-radius: 0 10px 10px 0;
            padding: .7rem .85rem; margin: .5rem 0;
        }
        .source-name { color: #253b82; font-size: .82rem; font-weight: 700; }
        .source-text { color: #475467; font-size: .9rem; margin-top: .25rem; }
        .welcome-card { background: white; border: 1px dashed #c7d2fe; border-radius: 16px; padding: 2rem; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_index_builder() -> IndexBuilder:
    """Return a cached IndexBuilder instance."""
    return IndexBuilder()


def save_uploaded_files(uploaded_files) -> list[str]:
    """Save supported uploaded files and return their names."""
    saved = []
    for uploaded_file in uploaded_files:
        if Path(uploaded_file.name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        (RAW_DATA_DIR / uploaded_file.name).write_bytes(uploaded_file.getbuffer())
        saved.append(uploaded_file.name)
    return saved


def clear_session() -> None:
    """Clear conversation and in-memory state without deleting the knowledge index."""
    st.session_state.clear()
    st.rerun()


def has_api_key() -> bool:
    """Return True when a Groq API key is configured."""
    key = os.getenv("GROQ_API_KEY", "")
    return bool(key and key.startswith("gsk_"))


def show_index_warnings(stats) -> None:
    """Surface partial indexing failures and empty-document warnings."""
    if stats.failed_files:
        st.warning(
            "Some files could not be indexed:\n\n"
            + "\n".join(f"- {name}" for name in stats.failed_files)
        )
    if stats.empty_files:
        st.warning(
            "These files had no extractable text (they may be scanned/image-only):\n\n"
            + "\n".join(f"- {name}" for name in stats.empty_files)
        )


def rebuild_index(builder: IndexBuilder) -> bool:
    """Rebuild the index while presenting meaningful progress to the user."""
    progress_bar = st.progress(0, text="Preparing your knowledge workspace")
    status = st.empty()

    def update_progress(percentage: int, message: str) -> None:
        progress_bar.progress(percentage, text=message)
        status.caption(f"{percentage}% · {message}")

    try:
        stats = builder.build(rebuild=True, progress_callback=update_progress)
        st.session_state.index_stats = builder.get_stats()
        st.session_state.index_ready = True
        if has_api_key():
            st.session_state.rag_pipeline = builder.create_rag_pipeline()
        else:
            st.session_state.rag_pipeline = None
        progress_bar.progress(100, text="Knowledge index ready")
        status.success(
            f"Indexed {stats.document_count} document(s) into {stats.chunk_count} searchable chunks."
        )
        show_index_warnings(stats)
        return True
    except ValueError as exc:
        st.session_state.rag_pipeline = None
        st.session_state.index_ready = False
        st.error(str(exc))
        return False
    except Exception as exc:
        logger.exception("Index rebuild failed")
        st.error(f"Indexing could not be completed: {exc}")
        return False


def load_existing_pipeline(builder: IndexBuilder) -> None:
    """Load the persistent index into a usable RAG pipeline when available."""
    stats = builder.get_stats()
    st.session_state.index_stats = stats
    if not stats["total_chunks"]:
        st.session_state.rag_pipeline = None
        st.session_state.index_ready = False
        return

    st.session_state.index_ready = True
    if not has_api_key():
        st.session_state.rag_pipeline = None
        return

    try:
        st.session_state.rag_pipeline = builder.create_rag_pipeline()
    except ValueError as exc:
        logger.warning("Could not load pipeline: %s", exc)
        st.session_state.rag_pipeline = None


def snippet(text: str, limit: int = 320) -> str:
    """Return a compact, single-line source preview."""
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[:limit].rstrip()}…"


def render_source_snippets(chunks: list[dict]) -> None:
    """Display the retrieved evidence directly beneath an answer."""
    if not chunks:
        return
    st.caption("Grounded in the following source excerpts")
    for chunk in chunks[:3]:
        metadata = chunk.get("metadata", {})
        filename = html.escape(str(metadata.get("filename", "Unknown source")))
        chunk_id = html.escape(str(metadata.get("chunk_id", "")))
        similarity = chunk.get("similarity")
        match_label = (
            f" · {similarity * 100:.0f}% match" if similarity is not None else ""
        )
        excerpt = html.escape(snippet(chunk.get("text", "")))
        st.markdown(
            f'<div class="source-card"><div class="source-name">{filename} · {chunk_id}{match_label}</div>'
            f'<div class="source-text">{excerpt}</div></div>',
            unsafe_allow_html=True,
        )
    if len(chunks) > 3:
        with st.expander(f"View {len(chunks) - 3} more source excerpt(s)"):
            for chunk in chunks[3:]:
                metadata = chunk.get("metadata", {})
                similarity = chunk.get("similarity")
                match_label = (
                    f" ({similarity * 100:.0f}% match)" if similarity is not None else ""
                )
                st.markdown(
                    f"**{metadata.get('filename', 'Unknown source')}**{match_label}"
                )
                st.write(snippet(chunk.get("text", ""), 600))


for key, value in {
    "messages": [],
    "rag_pipeline": None,
    "index_ready": False,
    "index_stats": {"document_count": 0, "total_chunks": 0, "filenames": []},
}.items():
    st.session_state.setdefault(key, value)

builder = get_index_builder()
if "pipeline_loaded" not in st.session_state:
    load_existing_pipeline(builder)
    st.session_state.pipeline_loaded = True

stats = st.session_state.index_stats

with st.sidebar:
    st.markdown("## ✦ Knowledge Copilot")
    st.caption("Enterprise document intelligence")
    if not has_api_key():
        st.error("GROQ_API_KEY is missing. Add it to `.env` to enable answers.")
    st.divider()
    st.markdown('<div class="metric-card"><div class="metric-label">Documents indexed</div><div class="metric-value">{}</div></div>'.format(stats.get("document_count", 0)), unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><div class="metric-label">Searchable chunks</div><div class="metric-value">{}</div></div>'.format(stats.get("total_chunks", 0)), unsafe_allow_html=True)
    indexed_files = stats.get("filenames", [])
    if indexed_files:
        with st.expander(f"Indexed files ({len(indexed_files)})"):
            for name in indexed_files:
                st.caption(f"📄 {name}")
    st.divider()
    top_k = st.slider("Sources per answer", 1, 10, TOP_K, help="Controls how many document passages are considered for each answer.")
    if st.button("↻ Rebuild knowledge index", use_container_width=True):
        rebuild_index(builder)
    if st.button("Clear session", use_container_width=True, type="secondary"):
        clear_session()
    st.caption("Clearing a session keeps your indexed documents intact.")

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">KNOWLEDGE, MADE CONVERSATIONAL</div>
      <h1>Your documentation, ready to answer.</h1>
      <p>Ask precise questions across your trusted internal documents. Every response is grounded in retrieved source material.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

upload_col, summary_col = st.columns([2.2, 1])
with upload_col:
    uploaded_files = st.file_uploader(
        "Add knowledge sources",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload PDF or DOCX files to add them to the knowledge index.",
    )
with summary_col:
    state_label = "Ready to answer" if st.session_state.index_ready else "Index needed"
    st.info(f"**{state_label}**\n\nPDF and DOCX sources are supported.")

if uploaded_files and st.button("Index uploaded documents", type="primary"):
    saved = save_uploaded_files(uploaded_files)
    if saved and rebuild_index(builder):
        st.success(f"Added {len(saved)} document(s): {', '.join(saved)}")
        st.rerun()
    if not saved:
        st.warning("No supported files were selected.")

st.divider()

if not st.session_state.index_ready:
    st.markdown(
        """
        <div class="welcome-card">
          <h3>Build your knowledge workspace</h3>
          <p>Upload a document above or select <b>Rebuild knowledge index</b> to begin asking grounded questions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("### Ask your knowledge base")
    st.caption("Answers cite the passages used to generate them.")
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])
            if message["role"] == "assistant":
                if message.get("low_confidence"):
                    st.warning(
                        "No strong match was found in your documents. "
                        "The answer may be unreliable."
                    )
                render_source_snippets(message.get("retrieved_chunks", []))

    query = st.chat_input("Ask a question about your documentation")
    if query:
        if not has_api_key():
            st.error("Add GROQ_API_KEY to `.env` before asking questions.")
        elif st.session_state.rag_pipeline is None:
            st.session_state.rag_pipeline = builder.create_rag_pipeline()

        if has_api_key() and st.session_state.rag_pipeline:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user", avatar="👤"):
                st.write(query)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Finding trusted evidence and drafting an answer…"):
                    try:
                        result = st.session_state.rag_pipeline.answer(query, top_k=top_k)
                        if result.get("low_confidence"):
                            st.warning(
                                "No strong match was found in your documents. "
                                "The answer may be unreliable."
                            )
                        st.write(result["answer"])
                        render_source_snippets(result["retrieved_chunks"])
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": result["answer"],
                                "retrieved_chunks": result["retrieved_chunks"],
                                "low_confidence": result.get("low_confidence", False),
                            }
                        )
                    except Exception as exc:
                        logger.exception("Question answering failed")
                        st.error(f"Unable to answer this question: {exc}")
                        st.session_state.messages.pop()

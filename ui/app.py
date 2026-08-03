from __future__ import annotations

import streamlit as st
import html
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


try:
    from src.config import DUPLICATE_THRESHOLD, RAW_DATA_DIR, SUPPORTED_EXTENSIONS, TOP_K
except ModuleNotFoundError:
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import DUPLICATE_THRESHOLD, RAW_DATA_DIR, SUPPORTED_EXTENSIONS, TOP_K


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title='AI Knowledge Navigator',
    page_icon='🧭',
    layout='wide',
    initial_sidebar_state='expanded',
)

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

st.markdown(
    '''
    <style>
        .stApp { background: #f7f8fc; }
        [data-testid="stSidebar"] { background: #111a33; }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaption { color: #e8edff !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div {
            background: #ffffff !important; border: 1px solid #7f93e8 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] [data-baseweb="input"] input {
            color: #172554 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #172554 !important; }
        [data-testid="stSidebar"] .stButton button {
            background: #4159c7; border-color: #7187ee; color: #ffffff;
        }
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
        .navigator-card {
            background: white; border: 1px solid #e5e9f2; border-radius: 18px;
            padding: 1.35rem 1.5rem; margin: .35rem 0 1rem;
            box-shadow: 0 8px 24px rgba(26, 47, 107, 0.06);
        }
        .navigator-label {
            color: #667085; font-size: .72rem; font-weight: 700;
            letter-spacing: .12em; text-transform: uppercase; margin-bottom: .55rem;
        }
        .navigator-answer { color: #101828; font-size: 1rem; line-height: 1.65; margin: 0; }
        .navigator-divider { border: 0; border-top: 1px solid #eaecf0; margin: 1.1rem 0; }
        .confidence-pill {
            display: inline-block; border-radius: 999px; padding: .28rem .75rem;
            font-size: .78rem; font-weight: 700; margin-bottom: .85rem;
        }
        .confidence-high { background: #ecfdf3; color: #027a48; }
        .confidence-medium { background: #fffaeb; color: #b54708; }
        .confidence-low { background: #fef3f2; color: #b42318; }
        .navigator-list { margin: .35rem 0 0; padding-left: 0; list-style: none; }
        .navigator-list li {
            color: #344054; font-size: .92rem; line-height: 1.55;
            padding: .28rem 0 .28rem 1.35rem; position: relative;
        }
        .navigator-list li::before {
            content: "•"; color: #6079ee; font-weight: 700;
            position: absolute; left: 0;
        }
        .navigator-sources li::before { content: "✓"; color: #12b76a; }
        .lifecycle-badge {
            display: inline-flex; align-items: center; gap: .35rem;
            border-radius: 999px; padding: .2rem .65rem; font-size: .78rem;
            font-weight: 700; margin-left: .5rem;
        }
        .lifecycle-fresh { background: #ecfdf5; color: #064e3b; }
        .lifecycle-need-update { background: #ffedd5; color: #9a3412; }
        .lifecycle-needs-deprecation { background: #fee2e2; color: #991b1b; }
        .lifecycle-aging { background: #fef3c7; color: #78350f; }
        .lifecycle-stale { background: #fee2e2; color: #991b1b; }
        .lifecycle-archived { background: #f8fafc; color: #475569; }
        .lifecycle-needs-review { background: #e0e7ff; color: #3730a3; }
        .source-library-meta { color: #475467; font-size: .92rem; margin-top: .3rem; }
        [data-testid="stDialog"] > div {
            border-radius: 22px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: 0 28px 70px rgba(15, 23, 42, 0.25), 0 2px 10px rgba(15, 23, 42, 0.08);
        }
        [data-testid="stDialog"] [slot="title"] {
            border-bottom: 1px solid #eef1f6;
            color: #0f172a;
            font-size: 1.2rem;
            padding: 1.4rem 1.5rem 1rem;
        }
        [data-testid="stDialog"] [slot="title"] + div {
            background: linear-gradient(180deg, #fbfcff 0%, #ffffff 100%);
            max-height: min(64vh, 680px);
            overflow-y: auto;
            padding: 1.25rem 1.5rem 1.5rem;
        }
        .library-summary { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1.1rem; }
        .library-summary-item {
            display: inline-flex; align-items: center; gap: .4rem;
            background: #eef2ff; color: #475467; border: 1px solid #e0e7ff;
            border-radius: 999px; padding: .38rem .85rem;
            font-size: .8rem; font-weight: 600;
        }
        .library-summary-item strong { color: #253b82; font-size: .9rem; }
        .library-grid-count { color: #98a2b3; font-size: .78rem; font-weight: 600; margin: .25rem 0 .6rem; }
        .library-card {
            background: #ffffff; border: 1px solid #e6e9f2; border-radius: 16px;
            padding: 1rem 1.1rem 1.05rem; margin-bottom: .9rem;
            display: flex; flex-direction: column; gap: .7rem;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }
        .library-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.09);
            border-color: #c7d2fe;
        }
        .library-card-top { display: flex; align-items: flex-start; justify-content: space-between; gap: .6rem; }
        .library-card-title { color: #0f172a; font-size: .95rem; font-weight: 800; line-height: 1.4; min-width: 0; }
        .library-tags { display: flex; flex-wrap: wrap; gap: .4rem; }
        .library-tag {
            display: inline-flex; align-items: center;
            background: #f1f4ff; color: #4054c7; border: 1px solid #dbe3ff;
            border-radius: 999px; font-size: .72rem; font-weight: 700;
            padding: .22rem .6rem;
        }
        .library-tag-keyword {
            background: #f8fafc; color: #475569; border: 1px solid #e2e8f0;
        }
        .library-tag-legend { color: #98a2b3; font-size: .72rem; margin: .4rem 0 .2rem; }
        .library-meta { display: grid; grid-template-columns: 1fr 1fr; gap: .35rem .8rem; }
        .library-meta div { color: #64748b; font-size: .78rem; }
        .library-meta div strong { color: #334155; font-weight: 700; }
        .library-empty {
            background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 16px;
            padding: 2.4rem 1.5rem; text-align: center;
        }
        .library-empty-title { color: #334155; font-size: .95rem; font-weight: 800; margin-bottom: .3rem; }
        .library-empty-text { color: #64748b; font-size: .85rem; }
        .navigator-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        @media (max-width: 768px) { .navigator-grid { grid-template-columns: 1fr; } }
    </style>
    ''',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_embedder():
    from src.config import EMBEDDING_MODEL
    from src.embedder import EmbeddingsGenerator
    return EmbeddingsGenerator(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_builder():
    from src.index_builder import IndexBuilder
    return IndexBuilder(embedder=get_embedder())


def save_uploaded_files(uploaded_files) -> list[str]:
    saved = []
    for uploaded_file in uploaded_files:
        if Path(uploaded_file.name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        (RAW_DATA_DIR / uploaded_file.name).write_bytes(uploaded_file.getbuffer())
        saved.append(uploaded_file.name)
    return saved


def clear_session() -> None:
    st.session_state.clear()
    st.rerun()


def has_api_key() -> bool:
    key = os.getenv('GROQ_API_KEY', '')
    return bool(key and key.startswith('gsk_'))


def show_index_warnings(stats) -> None:
    failed_files = None
    empty_files = None
    if hasattr(stats, 'get'):
        failed_files = stats.get('failed_files')
        empty_files = stats.get('empty_files')
    else:
        failed_files = getattr(stats, 'failed_files', None)
        empty_files = getattr(stats, 'empty_files', None)

    if failed_files:
        st.warning(
            'Some files could not be indexed:\n\n'
            + '\n'.join(f'- {name}' for name in failed_files)
        )
    if empty_files:
        st.warning(
            'These files had no extractable text (they may be scanned/image-only):\n\n'
            + '\n'.join(f'- {name}' for name in empty_files)
        )


def rebuild_index(builder: IndexBuilder, rebuild: bool = False, reindex_files: list[str] | None = None) -> bool:
    progress_bar = st.progress(0, text='Preparing your knowledge workspace')
    status = st.empty()

    def update_progress(percentage: int, message: str) -> None:
        progress_bar.progress(percentage, text=message)
        status.caption(f'{percentage}% · {message}')

    try:
        stats = builder.build(
            rebuild=rebuild, progress_callback=update_progress, reindex_files=reindex_files)
        st.session_state.index_stats = builder.get_stats()
        st.session_state.index_ready = True
        if has_api_key():
            st.session_state.rag_pipeline = builder.create_rag_pipeline()
        else:
            st.session_state.rag_pipeline = None
        progress_bar.progress(100, text='Knowledge index ready')
        status.success(
            f'Indexed {stats.document_count} document(s) into {stats.chunk_count} searchable chunks.'
        )
        show_index_warnings(stats)
        return True
    except ValueError as exc:
        st.session_state.rag_pipeline = None
        st.session_state.index_ready = False
        st.error(str(exc))
        return False
    except Exception as exc:
        logger.exception('Index rebuild failed')
        st.error(f'Indexing could not be completed: {exc}')
        return False


def ensure_index_loaded() -> None:
    """Lazily load the existing index and stats on first interactive use."""
    if st.session_state.get('pipeline_loaded'):
        return
    with st.spinner('Loading your knowledge assistant…'):
        load_existing_pipeline(get_builder())
    st.session_state.pipeline_loaded = True


def load_existing_pipeline(builder: IndexBuilder) -> None:
    if 'pipeline_loaded' in st.session_state:
        return

    with st.spinner('Loading knowledge index…'):
        orphaned = builder.remove_orphaned_documents()
        if orphaned:
            logger.info('Removed orphaned documents: %s', ', '.join(orphaned))

        stats = builder.get_stats()
    st.session_state.index_stats = stats
    st.session_state.index_ready = bool(stats.get('total_chunks'))

    if not st.session_state.index_ready or not has_api_key():
        st.session_state.rag_pipeline = None
        return

    ensure_rag_pipeline(builder)


def ensure_rag_pipeline(builder: IndexBuilder) -> None:
    if st.session_state.get('rag_pipeline') is not None:
        return
    if not st.session_state.index_ready or not has_api_key():
        st.session_state.rag_pipeline = None
        return
    try:
        with st.spinner('Loading AI model...'):
            st.session_state.rag_pipeline = builder.create_rag_pipeline()
    except Exception as exc:
        logger.warning('Could not initialize RAG pipeline: %s', exc)
        st.session_state.rag_pipeline = None


def snippet(text: str, limit: int = 320) -> str:
    compact = ' '.join(text.split())
    return compact if len(compact) <= limit else f'{compact[:limit].rstrip()}…'


def confidence_class(confidence: int) -> str:
    if confidence >= 75:
        return 'confidence-high'
    if confidence >= 50:
        return 'confidence-medium'
    return 'confidence-low'


def lifecycle_badge(metadata: dict) -> str:
    status = str(metadata.get('lifecycle_status', 'Fresh'))
    tone = status.lower().replace(' ', '-')
    icon = {
        'Fresh': '🟢',
        'Need Update': '🟠',
        'Needs Deprecation': '🔴',
        'Aging': '🟡',
        'Stale': '🔴',
        'Archived': '⚪',
        'Needs Review': '🔵',
    }.get(status, '⚪')
    return (
        f'<span class="lifecycle-badge lifecycle-{tone}" title="Lifecycle: {html.escape(status)}">{icon} '
        f'{html.escape(status)}</span>'
    )


def build_assistant_message(result: dict) -> dict:
    return {
        'role': 'assistant',
        'content': result.get('answer', ''),
        'answer': result.get('answer', ''),
        'confidence': result.get('confidence', 0),
        'sources': result.get('sources', []),
        'related_articles': result.get('related_articles', []),
        'related_documents': result.get('related_documents', []),
        'suggested_next_steps': result.get('suggested_next_steps', []),
        'retrieved_chunks': result.get('retrieved_chunks', []),
        'low_confidence': result.get('low_confidence', False),
    }


def render_knowledge_navigator(message: dict, message_index: int) -> None:
    answer = message.get('answer') or message.get('content', '')
    confidence = int(message.get('confidence', 0))
    related_articles = message.get('related_articles', [])
    related_documents = message.get('related_documents', [])
    next_steps = message.get('suggested_next_steps', [])

    confidence_label = confidence_class(confidence)
    if message.get('low_confidence'):
        confidence_label = 'confidence-low'

    source_items = {}
    for chunk in message.get('retrieved_chunks', []):
        metadata = chunk.get('metadata', {})
        filename = metadata.get('filename', 'Unknown source')
        if filename not in source_items:
            title = html.escape(metadata.get('title', filename))
            version = html.escape(str(metadata.get('version', 'Unspecified')))
            type_label = html.escape(metadata.get('document_type', ''))
            audience_label = html.escape(metadata.get('audience', ''))
            updated = html.escape(metadata.get('last_updated', ''))
            badge = lifecycle_badge(metadata)
            source_items[filename] = (
                f'<li><strong>{title}</strong> · v{version} {badge}<br />'
                f'<span class="source-library-meta">{type_label} · {audience_label} · Updated {updated}</span></li>'
            )

    source_html = ''.join(source_items.values()) or '<li>No sources found</li>'

    st.markdown(
        f'''
        <div class="navigator-card">
            <div class="navigator-label">Answer</div>
            <div class="confidence-pill {confidence_label}">Confidence: {confidence}%</div>
            <p class="navigator-answer">{html.escape(answer)}</p>
            <hr class="navigator-divider" />
            <div class="navigator-label">Source References</div>
            <ul class="navigator-list navigator-sources">
                {source_html}
            </ul>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    if related_articles or related_documents or next_steps:
        left, right = st.columns(2)
        with left:
            if related_articles:
                st.markdown('**Search more**')
                for idx, topic in enumerate(related_articles):
                    if st.button(topic, key=f'related_{message_index}_{idx}_{topic}', use_container_width=True):
                        st.session_state.pending_follow_up = topic
                        st.rerun()
            if related_documents:
                st.markdown('**Related Documents**')
                for document in related_documents:
                    st.caption(f'Document: {document}')
        with right:
            if next_steps:
                st.markdown('**Suggested Next Steps**')
                for step in next_steps:
                    st.markdown(f'- {step}')

    render_source_snippets(message.get('retrieved_chunks', []))


def answer_question(query: str, top_k: int, audience: str, filters: dict) -> None:
    st.session_state.messages.append({'role': 'user', 'content': query})
    with st.chat_message('user', avatar='👤'):
        st.write(query)

    with st.chat_message('assistant', avatar='🤖'):
        status_area = st.empty()
        status = status_area.status('Searching knowledge base...', expanded=False)
        try:
            def update_status(phase: str) -> None:
                label = {
                    'searching': 'Searching knowledge base...',
                    'generating': 'Generating answer...',
                }.get(phase)
                if label:
                    status.update(label=label, state='running')

            result = st.session_state.rag_pipeline.answer(
                query, top_k=top_k, audience=audience, filters=filters,
                conversation_history=st.session_state.messages,
                progress_callback=update_status,
            )
            status_area.empty()

            if result.get('low_confidence'):
                st.warning(
                    'No strong match was found in your documents. '
                    'Review the sources carefully before acting on this answer.'
                )
            assistant_message = build_assistant_message(result)
            render_knowledge_navigator(
                assistant_message,
                message_index=len(st.session_state.messages),
            )
            st.session_state.messages.append(assistant_message)
        except Exception as exc:
            status_area.empty()
            logger.exception('Question answering failed')
            st.error(f'Unable to answer this question: {exc}')
            st.session_state.messages.pop()


def render_source_snippets(chunks: list[dict]) -> None:
    if not chunks:
        return
    st.caption('Grounded in the following source excerpts')
    for chunk in chunks[:3]:
        metadata = chunk.get('metadata', {})
        filename = html.escape(str(metadata.get('filename', 'Unknown source')))
        chunk_id = html.escape(str(metadata.get('chunk_id', '')))
        similarity = chunk.get('similarity')
        match_label = f' · {similarity * 100:.0f}% match' if similarity is not None else ''
        badge = lifecycle_badge(metadata)
        excerpt = html.escape(snippet(chunk.get('text', '')))
        st.markdown(
            f"<div class=\"source-card\"><div class=\"source-name\">{filename} {badge} · v{html.escape(str(metadata.get('version', 'Unspecified')))}{match_label}</div>"
            f"<div class=\"source-text\">{excerpt}</div></div>",
            unsafe_allow_html=True,
        )
        with st.expander(f'Metadata for {filename}'):
            st.json({
                'Title': metadata.get('title'), 'Product': metadata.get('product'),
                'Version': metadata.get('version'), 'Document Type': metadata.get('document_type'),
                'Audience': metadata.get('audience'), 'Department': metadata.get('department'),
                'Author': metadata.get('author'), 'Last Updated': metadata.get('last_updated'),
                'Keywords': metadata.get('keywords'), 'Summary': metadata.get('summary'),
            })
    if len(chunks) > 3:
        with st.expander(f'View {len(chunks) - 3} more source excerpt(s)'):
            for chunk in chunks[3:]:
                metadata = chunk.get('metadata', {})
                similarity = chunk.get('similarity')
                match_label = f' ({similarity * 100:.0f}% match)' if similarity is not None else ''
                st.markdown(
                    f"**{metadata.get('filename', 'Unknown source')}**{match_label}")
                st.write(snippet(chunk.get('text', ''), 600))


def version_key(item: dict) -> tuple[int, ...]:
    """Numeric sort key for a document's version field."""
    return tuple(
        int(part) for part in str(item.get('version', '')).split('.') if part.isdigit()
    )


def sort_documents_by_version(documents: list[dict]) -> list[dict]:
    return sorted(
        documents,
        key=lambda item: (version_key(item), str(item.get('title', '')).lower()),
        reverse=True,
    )


def sort_documents(documents: list[dict], sort_by: str) -> list[dict]:
    if sort_by == 'Title A–Z':
        return sorted(
            documents, key=lambda item: str(item.get('title', '')).lower())
    if sort_by == 'Recently updated':
        return sorted(
            documents,
            key=lambda item: str(item.get('last_updated', '')),
            reverse=True,
        )
    if sort_by == 'Version: oldest first':
        return sorted(
            documents,
            key=lambda item: (version_key(item), str(item.get('title', '')).lower()),
        )
    return sort_documents_by_version(documents)


def is_latest_version(document: dict, documents: list[dict]) -> bool:
    title = str(document.get('title', ''))
    siblings = [doc for doc in documents if str(doc.get('title', '')) == title]
    if not siblings:
        return True
    latest = max(siblings, key=lambda item: version_key(item))
    return latest.get('filename') == document.get('filename')


def pick_keywords(keywords: str, limit: int = 2) -> list[str]:
    """Choose a few readable keywords, skipping filename- and filler-derived terms."""
    ignored = {
        'docx', 'pdf', 'page', 'corporation', 'confidential', 'summary',
        'revision', 'version', 'document', 'documents', 'purpose', 'policy',
    }
    picked: list[str] = []
    for keyword in str(keywords or '').split(','):
        token = keyword.strip()
        if len(token) < 3 or '_' in token or ' ' in token:
            continue
        if token.lower() in ignored or re.search(r'v?\d', token, flags=re.I):
            continue
        if token not in picked:
            picked.append(token)
        if len(picked) == limit:
            break
    return picked


def build_library_card(document: dict) -> str:
    title = html.escape(document.get(
        'title', document.get('filename', 'Unknown')))
    filename = html.escape(str(document.get('filename', '')))
    version = html.escape(str(document.get('version', 'Unspecified')))
    author = html.escape(str(document.get('author', 'Unknown')))
    last_updated = html.escape(str(document.get('last_updated', '')))
    document_type = html.escape(str(document.get('document_type', '')))
    chunks = document.get('total_chunks', 0)
    badge_html = lifecycle_badge(document)

    meta_rows = [('Version', version), ('Chunks', str(chunks))]
    if last_updated:
        meta_rows.append(('Updated', last_updated))
    if author and author != 'Unknown':
        meta_rows.append(('Author', author))
    meta_html = ''.join(
        f'<div><strong>{html.escape(label)}</strong>: {value}</div>'
        for label, value in meta_rows
    )

    tags = []
    if document_type and document_type.lower() not in {'unknown', 'nan', 'none'}:
        tags.append(
            f'<span class="library-tag" title="Document type">{document_type}</span>')
    for keyword in pick_keywords(document.get('keywords', '')):
        tags.append(
            f'<span class="library-tag library-tag-keyword" title="Keyword">#{keyword}</span>')

    return (
        '<div class="library-card">'
        f'<div class="library-card-top"><div class="library-card-title" title="{filename}">{title}</div>{badge_html}</div>'
        f'<div class="library-tags">{"".join(tags)}</div>'
        f'<div class="library-meta">{meta_html}</div>'
        '</div>'
    )


@st.dialog('Source Library', width='large', icon=':material/auto_stories:')
def render_source_library_panel(documents: list[dict]) -> None:
    documents = sort_documents_by_version(documents)
    total_chunks = sum(int(doc.get('total_chunks', 0) or 0) for doc in documents)
    document_types = sorted(
        {doc.get('document_type', 'Unknown') for doc in documents})
    lifecycle_statuses = sorted(
        {doc.get('lifecycle_status', 'Fresh') for doc in documents})

    st.markdown(
        f'''
        <div class="library-summary">
            <span class="library-summary-item"><strong>{len(documents)}</strong> document(s)</span>
            <span class="library-summary-item"><strong>{total_chunks}</strong> searchable chunk(s)</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    if not documents:
        st.markdown(
            '<div class="library-empty">'
            '<div class="library-empty-title">No indexed documents yet</div>'
            '<div class="library-empty-text">Upload PDF or DOCX files and rebuild the index to populate the source library.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button('Close source library', key='close_library', use_container_width=True):
            st.session_state.show_library = False
            st.rerun()
        return

    if st.session_state.pop('library_reset_requested', False):
        for key in ['library_search', 'library_filter_type',
                    'library_filter_status', 'library_sort', 'library_latest_only']:
            st.session_state.pop(key, None)

    search = st.text_input(
        'Search',
        placeholder='Filter by title or file name',
        key='library_search',
        label_visibility='collapsed',
    ).strip().lower()

    type_col, status_col = st.columns(2)
    with type_col:
        type_filter = st.selectbox(
            'Document type', ['All'] + document_types, key='library_filter_type')
    with status_col:
        status_filter = st.selectbox(
            'Lifecycle', ['All'] + lifecycle_statuses, key='library_filter_status')

    sort_col, latest_col, reset_col = st.columns([1.6, 1.2, 1])
    with sort_col:
        sort_by = st.selectbox(
            'Sort by',
            ['Version: newest first', 'Version: oldest first',
             'Title A–Z', 'Recently updated'],
            key='library_sort',
        )
    with latest_col:
        latest_only = st.checkbox(
            'Only latest versions', key='library_latest_only')
    with reset_col:
        st.markdown('')
        if st.button('Reset filters', key='library_reset', use_container_width=True):
            st.session_state.library_reset_requested = True

    filtered = [
        doc for doc in documents
        if (not search
            or search in str(doc.get('title', '')).lower()
            or search in str(doc.get('filename', '')).lower())
        and (type_filter == 'All'
             or doc.get('document_type', 'Unknown') == type_filter)
        and (status_filter == 'All'
             or doc.get('lifecycle_status', 'Fresh') == status_filter)
        and (not latest_only or is_latest_version(doc, documents))
    ]
    filtered = sort_documents(filtered, sort_by)

    if filtered:
        st.markdown(
            f'<div class="library-grid-count">{len(filtered)} of {len(documents)} shown</div>',
            unsafe_allow_html=True,
        )
        with st.container(height=520, border=False):
            columns = st.columns(2)
            for index, document in enumerate(filtered):
                with columns[index % 2]:
                    st.markdown(build_library_card(document), unsafe_allow_html=True)
        st.markdown(
            '<div class="library-tag-legend">Blue tags: document type · '
            'grey tags: keywords · badges: lifecycle status</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="library-empty">'
            '<div class="library-empty-title">No documents match your filter</div>'
            '<div class="library-empty-text">Try a different keyword, or clear the search box above.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if st.button('Close source library', key='close_library', use_container_width=True):
        st.session_state.show_library = False
        st.rerun()


for key, value in {
    'messages': [],
    'rag_pipeline': None,
    'index_ready': False,
    'index_stats': {'document_count': 0, 'total_chunks': 0, 'filenames': []},
    'show_library': False,
    'audience': 'End User',
}.items():
    st.session_state.setdefault(key, value)

stats = st.session_state.index_stats

with st.sidebar:
    st.markdown('## ✦ Knowledge Copilot')
    st.caption('Enterprise document intelligence')
    if not has_api_key():
        st.error('GROQ_API_KEY is missing. Add it to `.env` to enable answers.')
    st.divider()
    st.markdown(
        f"<div class=\"metric-card\"><div class=\"metric-label\">Documents indexed</div><div class=\"metric-value\">{stats.get('document_count', 0)}</div></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class=\"metric-card\"><div class=\"metric-label\">Searchable chunks</div><div class=\"metric-value\">{stats.get('total_chunks', 0)}</div></div>", unsafe_allow_html=True)
    indexed_files = stats.get('filenames', [])
    if st.button('Open source library', key='open_library', use_container_width=True):
        st.session_state.show_library = True
        st.rerun()
    st.markdown('### Search settings')
    audience = st.selectbox(
        'Answer for', [
            'End User', 'Support Engineer', 'Technical Writer',
            'Administrator', 'Product Manager'
        ], index=['End User', 'Support Engineer', 'Technical Writer', 'Administrator', 'Product Manager'].index(st.session_state.audience), key='audience'
    )
    top_k = st.slider(
        'Sources per answer', 1, 10, TOP_K,
        key='top_k', help='Controls how many document passages are considered for each answer.'
    )
    filters = {}
    if indexed_files:
        with st.expander('Filter by metadata', expanded=False):
            filter_product = st.selectbox(
                'Product', ['All'] + stats.get('products', []), key='filter_product')
            filter_version = st.selectbox(
                'Version', ['All', 'Latest Version'] + stats.get('versions', []), key='filter_version')
            filter_department = st.selectbox(
                'Department', ['All'] + stats.get('departments', []), key='filter_department')
            filter_doc_type = st.selectbox(
                'Document type', ['All'] + stats.get('document_types', []), key='filter_doc_type')
        if filter_product != 'All':
            filters['product'] = filter_product
        if filter_version != 'All':
            filters['version'] = filter_version
        if filter_department != 'All':
            filters['department'] = filter_department
        if filter_doc_type != 'All':
            filters['document_type'] = filter_doc_type
    st.caption('Answer style: ' + st.session_state.audience)
    if indexed_files:
        with st.expander(f'Indexed files ({len(indexed_files)})'):
            for name in indexed_files:
                st.caption(f'📄 {name}')
    st.divider()
    if st.button('↻ Rebuild knowledge index', use_container_width=True):
        rebuild_index(get_builder(), rebuild=True)
    if indexed_files:
        with st.expander('Repository management'):
            delete_filename = st.selectbox(
                'Delete indexed document', indexed_files)
            if st.button('Delete document', use_container_width=True):
                get_builder().delete_document(delete_filename)
                st.session_state.index_stats = get_builder().get_stats()
                st.rerun()
            if st.button('Clear index (keep uploads)', use_container_width=True):
                get_builder().clear_repository(delete_sources=False)
                load_existing_pipeline(get_builder())
                st.rerun()
            if st.button('Delete repository and uploads', type='secondary', use_container_width=True):
                get_builder().clear_repository(delete_sources=True)
                load_existing_pipeline(get_builder())
                st.rerun()
    if st.button('Clear session', use_container_width=True, type='secondary'):
        clear_session()
    st.caption('Clearing a session keeps your indexed documents intact.')

st.markdown(
    '''
    <section class="hero">
      <div class="eyebrow">AI KNOWLEDGE NAVIGATOR</div>
      <h1>Your documentation, intelligently guided.</h1>
      <p>Ask a question and get a grounded answer with sources and suggested next steps — not just a chat reply.</p>
    </section>
    ''',
    unsafe_allow_html=True,
)

upload_col, summary_col = st.columns([2.2, 1])
with upload_col:
    uploaded_files = st.file_uploader(
        'Add knowledge sources',
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help='Upload PDF or DOCX files to add them to the knowledge index.',
    )
with summary_col:
    state_label = 'Ready to answer' if st.session_state.index_ready else 'Index needed'
    st.info(f'**{state_label}**\n\nPDF and DOCX sources are supported.')

if uploaded_files and st.button('Index uploaded documents', type='primary'):
    saved = save_uploaded_files(uploaded_files)
    duplicates = []
    reindex_files = []
    if saved:
        existing = set(st.session_state.index_stats.get('filenames', []))
        reindex_files = [name for name in saved if name in existing]
        from src.load_document import load_documents_from_directory
        document_collection = load_documents_from_directory(
            RAW_DATA_DIR).documents
        for filename in saved:
            duplicates.extend(get_builder().find_duplicates(
                filename, DUPLICATE_THRESHOLD, document_list=document_collection))
    if duplicates:
        best = max(duplicates, key=lambda item: item['similarity'])
        st.warning(
            f"Duplicate Document Detected — {best['similarity']:.0%} similar to {best['metadata']['filename']}. Duplicated section: {snippet(best['text'], 180)}"
        )
    if saved and rebuild_index(get_builder(), rebuild=False, reindex_files=reindex_files):
        st.success(f"Added {len(saved)} document(s): {', '.join(saved)}")
        st.rerun()
    if not saved:
        st.warning('No supported files were selected.')

st.divider()

# The source library opens as a floating modal window from the sidebar.

if not st.session_state.index_ready:
    st.markdown(
        '''
        <div class="welcome-card">
          <h3>Build your knowledge workspace</h3>
          <p>Upload a document above or select <b>Rebuild knowledge index</b> to begin asking grounded questions.</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

if st.session_state.show_library:
    st.session_state.show_library = False
    try:
        with st.spinner('Loading your source library…'):
            fresh_stats = get_builder().get_stats()
        st.session_state.index_stats = fresh_stats
    except Exception as exc:
        logger.exception('Failed to refresh source library statistics')
        fresh_stats = st.session_state.get('index_stats', {})
        st.warning(f'The source library could not be refreshed: {exc}')
    render_source_library_panel(fresh_stats.get('documents', []))

st.markdown('### AI Knowledge Navigator')
st.caption(
    'Structured answers with sources, related articles, and suggested next steps.')
for message_index, message in enumerate(st.session_state.messages):
    avatar = '🤖' if message['role'] == 'assistant' else '👤'
    with st.chat_message(message['role'], avatar=avatar):
        if message['role'] == 'assistant':
            if message.get('low_confidence'):
                st.warning(
                    'No strong match was found in your documents. '
                    'Review the sources carefully before acting on this answer.'
                )
            render_knowledge_navigator(
                message, message_index=message_index)
        else:
            st.write(message['content'])

pending_follow_up = st.session_state.pop('pending_follow_up', None)
if pending_follow_up and has_api_key():
    ensure_rag_pipeline(get_builder())
    if st.session_state.rag_pipeline:
        answer_question(
            pending_follow_up,
            top_k=st.session_state.top_k,
            audience=st.session_state.audience,
            filters=filters,
        )

query = st.chat_input('Ask a question about your documentation')
if query:
    if not has_api_key():
        st.error('Add GROQ_API_KEY to `.env` before asking questions.')
    else:
        ensure_index_loaded()
        if not st.session_state.index_ready:
            st.warning(
                'The knowledge index is not loaded yet. '
                'Upload documents and rebuild the index to start asking questions.'
            )
        else:
            ensure_rag_pipeline(get_builder())
            if st.session_state.rag_pipeline:
                answer_question(
                    query,
                    top_k=st.session_state.top_k,
                    audience=st.session_state.audience,
                    filters=filters,
                )
            else:
                st.error(
                    'The AI model could not be loaded. '
                    'Check the configuration and try again.'
                )

import streamlit as st
import html
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


try:
    from src.config import DUPLICATE_THRESHOLD, RAW_DATA_DIR, SUPPORTED_EXTENSIONS, TOP_K
    from src.index_builder import IndexBuilder
except ModuleNotFoundError:
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import DUPLICATE_THRESHOLD, RAW_DATA_DIR, SUPPORTED_EXTENSIONS, TOP_K
    from src.index_builder import IndexBuilder


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
        .lifecycle-aging { background: #fef3c7; color: #78350f; }
        .lifecycle-stale { background: #fee2e2; color: #991b1b; }
        .lifecycle-archived { background: #f8fafc; color: #475569; }
        .lifecycle-needs-review { background: #e0e7ff; color: #3730a3; }
        .source-library-card {
            background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 18px;
            padding: 1rem 1.1rem; margin-bottom: 1rem;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .source-library-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 35px rgba(15, 23, 42, 0.08);
        }
        .source-library-heading { color: #0f172a; font-size: 1rem; font-weight: 800; margin-bottom: .35rem; }
        .source-library-meta { color: #475467; font-size: .92rem; margin-top: .3rem; }
        .source-library-title { font-size: 1.15rem; font-weight: 900; letter-spacing: -0.02em; margin: 0; }
        .source-library-subtitle { color: #64748b; font-size: .95rem; margin-top: .25rem; }
        .modal-backdrop {
            position: fixed; inset: 0; z-index: 998;
            background: rgba(15, 23, 42, 0.42);
            backdrop-filter: blur(6px);
        }
        .floating-library {
            margin: 0 auto;
            width: min(760px, calc(100vw - 48px));
            max-height: calc(88vh);
            overflow-y: auto;
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 26px;
            box-shadow: 0 40px 90px rgba(15, 23, 42, 0.18);
            padding: 1.6rem 1.6rem 1.8rem;
            position: relative;
            z-index: 999;
        }
        .floating-library-header-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 1.4rem;
        }
        .floating-library-close {
            color: #ffffff; background: #111827; border-radius: 999px;
            width: 2.9rem; height: 2.9rem; display: inline-flex;
            align-items: center; justify-content: center;
            border: 1px solid rgba(255,255,255,0.17);
            cursor: pointer;
            transition: transform 0.15s ease, background 0.15s ease;
            font-size: 1.05rem;
            line-height: 1;
        }
        .floating-library-close:hover {
            background: #0f172a;
            transform: translateY(-1px);
        }
        .floating-library-title { margin: 0; }
        .navigator-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        @media (max-width: 768px) { .navigator-grid { grid-template-columns: 1fr; } }
    </style>
    ''',
    unsafe_allow_html=True,
)


@st.cache_resource
def get_index_builder() -> IndexBuilder:
    return IndexBuilder()


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


def rebuild_index(builder: IndexBuilder, rebuild: bool = False) -> bool:
    progress_bar = st.progress(0, text='Preparing your knowledge workspace')
    status = st.empty()

    def update_progress(percentage: int, message: str) -> None:
        progress_bar.progress(percentage, text=message)
        status.caption(f'{percentage}% · {message}')

    try:
        stats = builder.build(
            rebuild=rebuild, progress_callback=update_progress)
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


def load_existing_pipeline(builder: IndexBuilder) -> None:
    orphaned = builder.remove_orphaned_documents()
    if orphaned:
        logger.info('Removed orphaned documents: %s', ', '.join(orphaned))
    stats = builder.get_stats()
    st.session_state.index_stats = stats
    if not stats.get('total_chunks'):
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
        logger.warning('Could not load pipeline: %s', exc)
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
        'Aging': '🟡',
        'Stale': '🔴',
        'Archived': '⚪',
        'Needs Review': '🔵',
    }.get(status, '⚪')
    return (
        f'<span class="lifecycle-badge lifecycle-{tone}">{icon} '
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
        with st.spinner('Navigating your knowledge base…'):
            try:
                result = st.session_state.rag_pipeline.answer(
                    query, top_k=top_k, audience=audience, filters=filters,
                    conversation_history=st.session_state.messages,
                )
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


def render_source_library_panel(documents: list[dict]) -> None:
    if not documents:
        return

    cards = []
    for document in documents:
        title = html.escape(document.get(
            'title', document.get('filename', 'Unknown')))
        version = html.escape(str(document.get('version', 'Unspecified')))
        author = html.escape(document.get('author', 'Unknown'))
        last_updated = html.escape(str(document.get('last_updated', '')))
        badge_html = lifecycle_badge(document)
        cards.append(
            f'<div class="source-library-card">'
            f'<div class="source-library-heading">{title} {badge_html}</div>'
            f'<div class="source-library-meta">Version {version} · Author: {author}</div>'
            f'<div class="source-library-meta">Last published: {last_updated}</div>'
            f'</div>'
        )

    with st.container():
        st.markdown(
            '<div class="floating-library">'
            '<div class="floating-library-header-row">'
            '<div>'
            '<div class="source-library-title">Source Library</div>'
            '<div class="source-library-subtitle">Browse recent documents, versions, and lifecycle status in one place.</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button('Close source library', key='close_library', use_container_width=True, help='Close source library'):
            st.session_state.show_library = False
            st.rerun()

        if cards:
            for card in cards:
                st.markdown(card, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="source-library-card">'
                '<div class="source-library-heading">No indexed documents found</div>'
                '<div class="source-library-meta">Upload documents or rebuild the index to populate the source library.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


for key, value in {
    'messages': [],
    'rag_pipeline': None,
    'index_ready': False,
    'index_stats': {'document_count': 0, 'total_chunks': 0, 'filenames': []},
    'show_library': False,
}.items():
    st.session_state.setdefault(key, value)

builder = get_index_builder()
if 'pipeline_loaded' not in st.session_state:
    load_existing_pipeline(builder)
    st.session_state.pipeline_loaded = True

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
    audience = st.selectbox('Answer for', [
        'End User', 'Support Engineer', 'Technical Writer', 'Administrator', 'Product Manager'
    ])
    filters = {}
    st.caption('Answer style: ' + audience)
    if indexed_files:
        with st.expander(f'Indexed files ({len(indexed_files)})'):
            for name in indexed_files:
                st.caption(f'📄 {name}')
    st.divider()
    top_k = st.slider('Sources per answer', 1, 10, TOP_K,
                      help='Controls how many document passages are considered for each answer.')
    if st.button('↻ Rebuild knowledge index', use_container_width=True):
        rebuild_index(builder, rebuild=True)
    if indexed_files:
        with st.expander('Repository management'):
            delete_filename = st.selectbox(
                'Delete indexed document', indexed_files)
            if st.button('Delete document', use_container_width=True):
                builder.delete_document(delete_filename)
                st.session_state.index_stats = builder.get_stats()
                st.rerun()
            if st.button('Clear index (keep uploads)', use_container_width=True):
                builder.clear_repository(delete_sources=False)
                load_existing_pipeline(builder)
                st.rerun()
            if st.button('Delete repository and uploads', type='secondary', use_container_width=True):
                builder.clear_repository(delete_sources=True)
                load_existing_pipeline(builder)
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
    for filename in saved:
        duplicates.extend(builder.find_duplicates(
            filename, DUPLICATE_THRESHOLD))
    if duplicates:
        best = max(duplicates, key=lambda item: item['similarity'])
        st.warning(
            f"Duplicate Document Detected — {best['similarity']:.0%} similar to {best['metadata']['filename']}. Duplicated section: {snippet(best['text'], 180)}"
        )
    if saved and rebuild_index(builder, rebuild=False):
        st.success(f"Added {len(saved)} document(s): {', '.join(saved)}")
        st.rerun()
    if not saved:
        st.warning('No supported files were selected.')

st.divider()

# The source library is now available through the floating panel in the sidebar.

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
    documents = stats.get('documents', [])
    sorted_documents = sorted(
        documents,
        key=lambda item: tuple(int(part) for part in str(
            item.get('version', '')).split('.') if part.isdigit()),
        reverse=True,
    )
    render_source_library_panel(sorted_documents)

else:
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
        if st.session_state.rag_pipeline is None:
            st.session_state.rag_pipeline = builder.create_rag_pipeline()
        if st.session_state.rag_pipeline:
            answer_question(pending_follow_up, top_k=top_k,
                            audience=audience, filters=filters)

    query = st.chat_input('Ask a question about your documentation')
    if query:
        if not has_api_key():
            st.error('Add GROQ_API_KEY to `.env` before asking questions.')
        elif st.session_state.rag_pipeline is None:
            st.session_state.rag_pipeline = builder.create_rag_pipeline()

        if has_api_key() and st.session_state.rag_pipeline:
            answer_question(query, top_k=top_k,
                            audience=audience, filters=filters)

# AI Docs Copilot

A production-quality RAG (Retrieval-Augmented Generation) proof of concept for technical writing teams. Upload PDF and DOCX documents, then ask questions grounded in your documentation with source citations.

## Features

- PDF and DOCX document ingestion
- Local embeddings with `BAAI/bge-small-en-v1.5`
- Persistent vector search with ChromaDB
- Grounded answers via Google Gemini
- Streamlit UI with upload, index rebuild, and retrieved chunk inspection

## Tech Stack

- Python 3.12
- Streamlit
- ChromaDB
- Sentence Transformers
- LangChain Text Splitters
- PyMuPDF
- python-docx
- Google Gemini API (`google-genai`)

## Setup

### 1. Activate virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure API key

Copy the example env file and add your Gemini API key:

```powershell
copy .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=your_key_here
```

Get a key at: https://ai.google.dev

### 4. (Optional) Create sample documents

```powershell
python scripts/create_samples.py
```

This creates sample PDF and DOCX files in `data/raw/`.

## Run the application

```powershell
streamlit run ui/app.py
```

### Using the UI

1. Upload PDF or DOCX files in the main panel
2. Click **Save and Index Documents** — files are saved to `data/raw/` and indexed automatically
3. Ask questions in the chat interface
4. Review answers, source filenames, and expandable retrieved chunks
5. Use **Rebuild Index** in the sidebar to re-index all files in `data/raw/`

## CLI index build

You can also build the index from the command line:

```powershell
python scripts/build_index.py
```

The persistent Chroma database is stored at `vectorstore/chroma_db/`.

## Project structure

```
ai-docs-copilot/
├── data/
│   └── raw/              # Source PDF and DOCX files
├── src/
│   ├── config.py         # Paths and constants
│   ├── load_document.py  # Document loader dispatcher
│   ├── pdf_loader.py     # PDF text extraction
│   ├── docx_loader.py    # DOCX text extraction
│   ├── chunker.py        # RecursiveCharacterTextSplitter
│   ├── embedder.py       # Sentence-transformers embeddings
│   ├── vector_db.py      # ChromaDB storage and search
│   ├── retriever.py      # Semantic retrieval
│   ├── gemini.py         # Gemini LLM integration
│   ├── index_builder.py  # Index build orchestration
│   └── rag.py            # RAG pipeline
├── ui/
│   └── app.py            # Streamlit application
├── scripts/              # CLI utilities and tests
├── vectorstore/
│   └── chroma_db/        # Persistent ChromaDB index
├── .env.example
└── requirements.txt
```

## Test scripts

Run these from the project root after adding documents to `data/raw/`:

| Script | Purpose |
|--------|---------|
| `python scripts/test_loader.py` | Document loading |
| `python scripts/test_chunker.py` | Text chunking |
| `python scripts/embedder_test.py` | Embedding generation |
| `python scripts/vector_db_test.py` | ChromaDB storage and search |
| `python scripts/retriever_test.py` | Semantic retrieval |
| `python scripts/gemini_test.py` | Gemini answers (requires API key) |
| `python scripts/rag_test.py` | Full RAG pipeline (requires API key) |

## RAG pipeline

```
Question → Embedding → Chroma Search → Prompt → Gemini → Answer
```

- Chunk size: 800 characters, overlap: 100
- Top-K retrieval: 5
- Answers are constrained to retrieved context only

## Requirements

- Python 3.12
- See `requirements.txt` for package versions

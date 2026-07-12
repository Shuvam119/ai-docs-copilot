# AI Docs Copilot

A RAG-powered document assistant that lets you ask questions about PDF and Word documents with citations.

## Features

- 📄 Support for PDF and DOCX documents
- 🔍 Vector search with ChromaDB
- 🤖 AI-powered answers with Gemini
- 💬 Web-based chat interface with Streamlit
- 📌 Document citations in responses

## Setup

### 1. Activate Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure API Key

Add your Gemini API key to `.env`:

```
GEMINI_API_KEY=your_key_here
```

Get a key at: https://ai.google.dev

### 4. Add Documents

Place PDF and DOCX files in `data/raw/`

### 5. Build Vector Index

```powershell
python scripts/build_index.py
```

### 6. Run the UI

```powershell
streamlit run ui/app.py
```

## Project Structure

```
ai-docs-copilot/
├── config/          # Configuration files
├── data/
│   ├── raw/         # Original documents
│   ├── processed/   # Processed documents
│   └── chunks/      # Text chunks
├── src/             # Application modules
├── ui/              # Streamlit interface
├── vectorstore/     # ChromaDB index
├── scripts/         # Utility scripts
└── tests/           # Tests
```

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

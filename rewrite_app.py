

#!/usr/bin/env python
# -*- coding: utf-8 -*-

content = '''# -*-coding: utf-8 -*-
"""
Streamlit Web App - AI Documentation Copilot
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Setup paths FIRST before any src imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# NOW import src modules after path setup
from src.rag import RAGPipeline
from src.llm import LLMClient
from src.retriever import Retriever
from src.vector_db import VectorStore
from src.embedder import EmbeddingsGenerator
from src.chunker import DocumentChunker
from src.load_document import load_documents_from_directory

# Page config
st.set_page_config(
    page_title="AI Docs Copilot",
    page_icon="book",
    layout="wide"
)

st.title("AI Documentation Copilot")
st.markdown("Ask questions about your documents with AI assistance.")

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
    st.session_state.initialized = False
    st.session_state.messages = []


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    with st.spinner("Loading pipeline..."):
        try:
            raw_data_dir = PROJECT_ROOT / "data" / "raw"
            documents = load_documents_from_directory(str(raw_data_dir))
            
            if not documents:
                st.error("No documents in data/raw/")
                return False
            
            chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
            chunks = chunker.chunk_documents(documents)
            
            embedder = EmbeddingsGenerator()
            chunks_with_embeddings = embedder.embed_chunks(chunks)
            
            vectorstore_path = PROJECT_ROOT / "vectorstore"
            vector_store = VectorStore(str(vectorstore_path), collection_name="documents")
            vector_store.add_chunks(chunks_with_embeddings)
            
            retriever = Retriever(embedder, vector_store)
            llm = LLMClient()
            
            st.session_state.rag_pipeline = RAGPipeline(embedder, retriever, llm)
            st.session_state.initialized = True
            
            return True
            
        except ValueError as e:
            st.error(str(e))
            return False
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return False


# Sidebar
with st.sidebar:
    st.header("Settings")
    
    if st.button("Initialize Pipeline", use_container_width=True):
        if initialize_pipeline():
            st.success("Ready!")
    
    if st.session_state.initialized:
        st.info("System initialized")
    else:
        st.warning("Click Initialize Pipeline")
    
    st.divider()
    
    top_k = st.slider("Retrieved chunks", 1, 10, 5)
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main content
if not st.session_state.initialized:
    st.info("1. Click Initialize Pipeline\\n2. Ask a question")
else:
    st.subheader("Ask a Question")
    
    # Chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if "sources" in message:
                    with st.expander("Sources"):
                        for s in message["sources"]:
                            st.write(f"- {s}")
    
    # Input
    query = st.chat_input("Ask about documents...")
    
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_pipeline.answer(query, top_k=top_k)
                    answer = result["answer"]
                    st.write(answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": result["sources"]
                    })
                    
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.write(f"- {s}")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.messages.pop()
'''

with open('ui/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("File rewritten successfully!")

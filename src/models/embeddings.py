"""
Embeddings and vector store creation.
"""
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


@st.cache_resource(show_spinner="Processing material PDFs into vector store...")
def create_vector_store(_material_files_contents: tuple, api_key: str):
    """Build a FAISS vector store from uploaded material PDF files."""
    if not _material_files_contents:
        return None

    all_docs = []
    temp_files_paths = []
    try:
        for file_content in _material_files_contents:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                temp_files_paths.append(tmp_file.name)
                loader = PyPDFLoader(tmp_file.name)
                all_docs.extend(loader.load())

        if not all_docs:
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        split_docs = text_splitter.split_documents(all_docs)

        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )
        vector_store = FAISS.from_documents(split_docs, embeddings)
        return vector_store
    finally:
        for path in temp_files_paths:
            if os.path.exists(path):
                os.remove(path)

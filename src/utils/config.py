"""
Configuration & environment variable loading.
"""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Try environment variables first (local .env)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Fallback to Streamlit secrets (Streamlit Cloud)
if not GOOGLE_API_KEY:
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        elif "env" in st.secrets and "GOOGLE_API_KEY" in st.secrets["env"]:
            GOOGLE_API_KEY = st.secrets["env"]["GOOGLE_API_KEY"]
    except Exception:
        pass

# Model configuration
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001"

# Text splitter settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retriever settings
RETRIEVER_K = 3

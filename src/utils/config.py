"""
Configuration & environment variable loading.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model configuration
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001"

# Text splitter settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retriever settings
RETRIEVER_K = 3

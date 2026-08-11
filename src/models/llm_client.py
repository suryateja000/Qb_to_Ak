"""
LLM client initialization and configuration.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.config import LLM_MODEL


def get_llm(api_key: str, temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Create and return a configured Gemini LLM instance."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=temperature,
    )

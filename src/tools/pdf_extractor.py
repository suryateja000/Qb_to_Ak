"""
PDF question extraction tool.
"""
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.models.llm_client import get_llm
from src.prompts.system_prompts import QUESTION_EXTRACTION_PROMPT


@st.cache_data(show_spinner="Extracting questions from Question Bank...")
def extract_questions_from_pdf(pdf_content: bytes, api_key: str) -> list[str]:
    """Extract individual questions from a PDF using LLM analysis."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_content)
        pdf_path = tmp_file.name

    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        full_pdf_text = "\n\n".join([page.page_content for page in pages])

        if not full_pdf_text.strip():
            return []

        llm = get_llm(api_key, temperature=0.1)
        prompt = PromptTemplate.from_template(QUESTION_EXTRACTION_PROMPT)
        extraction_chain = {"pdf_content": RunnablePassthrough()} | prompt | llm | StrOutputParser()
        extracted_questions_string = extraction_chain.invoke(full_pdf_text)
        list_of_questions = [q.strip() for q in extracted_questions_string.split('\n') if q.strip()]
        return list_of_questions
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

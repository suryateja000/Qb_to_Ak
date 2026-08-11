"""
Answer generation tool — uses RAG or general knowledge.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from src.models.llm_client import get_llm
from src.utils.config import RETRIEVER_K
from src.prompts.system_prompts import QA_SYSTEM_PROMPT, GENERAL_QA_PROMPT


def generate_answer(question: str, vector_store, api_key: str) -> str:
    """Generate an answer using RAG (if materials provided) or general knowledge."""
    llm = get_llm(api_key, temperature=0.3)

    if vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
        qa_prompt = PromptTemplate.from_template(QA_SYSTEM_PROMPT)
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response_dict = retrieval_chain.invoke({"input": question})
        return response_dict['answer']
    else:
        prompt = PromptTemplate.from_template(GENERAL_QA_PROMPT)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question})

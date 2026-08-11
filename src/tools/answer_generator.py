"""
Answer generation tool — uses RAG or general knowledge.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.models.llm_client import get_llm
from src.utils.config import RETRIEVER_K
from src.prompts.system_prompts import QA_SYSTEM_PROMPT, GENERAL_QA_PROMPT


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate_answer(question: str, vector_store, api_key: str) -> str:
    """Generate an answer using RAG (if materials provided) or general knowledge."""
    llm = get_llm(api_key, temperature=0.3)

    if vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
        qa_prompt = PromptTemplate.from_template(QA_SYSTEM_PROMPT)
        
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain.invoke(question)
    else:
        prompt = PromptTemplate.from_template(GENERAL_QA_PROMPT)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question})

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain 
import tempfile
import os
from fpdf import FPDF


google_api_key = "AIzaSyAY40SFfpOXvzghfrT0PDKnTC1rpCqFZ8I" 

if not google_api_key:
    st.error("Error: GOOGLE_API_KEY not found. Please configure it.")
    st.stop()


@st.cache_data(show_spinner="Extracting questions from Question Bank...")
def extract_questions_from_pdf_cached(pdf_content, api_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_content)
        pdf_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        full_pdf_text = "\n\n".join([page.page_content for page in pages])
        
        if not full_pdf_text.strip():
            return []
            
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", 
                                     google_api_key=api_key,
                                     temperature=0.1) 

        question_extraction_prompt_template = """
You are an expert AI assistant. Your task is to carefully read the following text extracted from a PDF document and identify all the individual questions.
For each question you identify, you must also assess if it's a "small question" (likely requiring a brief answer) or a "long answer" type (likely requiring a more detailed explanation).
Assign 2 marks for "small questions" and 5 marks for "long answers".
Please list each distinct question followed by its assigned marks in square brackets. For example:
What is the capital of France? [2m]
Explain the process of photosynthesis in detail. [5m]
Each question with its marks should be on a new line.
If the PDF text itself contains marks for a question (e.g., "(5 marks)"), prioritize those marks from the PDF and format them as [Xm]. If no marks are present in the PDF for a question, use your judgment to assign [2m] or [5m].
Text from PDF:
---
{pdf_content}
---
Extracted Questions with Marks (list each on a new line, e.g., Question text [Xm]):
"""
        prompt = PromptTemplate.from_template(question_extraction_prompt_template)
        extraction_chain = {"pdf_content": RunnablePassthrough()} | prompt | llm | StrOutputParser()
        extracted_questions_string = extraction_chain.invoke(full_pdf_text)
        list_of_questions = [q.strip() for q in extracted_questions_string.split('\n') if q.strip()]
        return list_of_questions
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

@st.cache_resource(show_spinner="Processing material PDFs into vector store...")
def create_vector_store_from_materials(_material_files_contents, api_key):
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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_docs)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        return vector_store
    finally:
        for path in temp_files_paths:
            if os.path.exists(path):
                os.remove(path)

def generate_answer(question: str, vector_store, api_key: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", 
                                 google_api_key=api_key,
                                 temperature=0.3)
    
    if vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 
        
        qa_system_prompt = """You are an expert assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer or the context is not relevant, state that you cannot answer based on the provided documents, and then try to answer based on your knowledge. \
Keep the answer concise and accurate. Highlight key information.

Context:
{context}

Question: {input}

Answer:""" 
        
        qa_prompt = PromptTemplate.from_template(qa_system_prompt)
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        retrieval_augmented_qa_chain = create_retrieval_chain(retriever, document_chain)
        response_dict = retrieval_augmented_qa_chain.invoke({"input": question})
        return response_dict['answer']
        
    else:
        general_prompt_template_text = """Answer the following question based on your general knowledge.
Question: {question}
Answer:"""
        prompt = PromptTemplate.from_template(general_prompt_template_text)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question})


def create_answer_key_pdf(qna_pairs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Answer Key", 0, 1, 'C')
    pdf.ln(5)

    for i, pair in enumerate(qna_pairs):
        pdf.set_font("Arial", 'B', 12)
        pdf.multi_cell(0, 10, f"Question {i+1}: {pair['question']}") 
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, f"Answer: {pair['answer']}") 
        pdf.ln(5) 
        if i < len(qna_pairs) - 1:
             pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + pdf.w - 2 * pdf.l_margin, pdf.get_y()) 
             pdf.ln(5)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf_output_path = tmp_pdf.name
    pdf.output(pdf_output_path, "F") 
    return pdf_output_path


st.set_page_config(layout="wide")
st.title("📚 AI-Powered Answer Key Generator")

if "extracted_questions" not in st.session_state:
    st.session_state.extracted_questions = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qna_pairs" not in st.session_state:
    st.session_state.qna_pairs = []
if "selected_question_for_answer" not in st.session_state:
    st.session_state.selected_question_for_answer = None
if "current_answer" not in st.session_state: 
    st.session_state.current_answer = ""
if "material_uploader_key" not in st.session_state: 
    st.session_state.material_uploader_key = 0 


with st.sidebar:
    st.header("Upload Files")
    question_bank_uploaded_file = st.file_uploader("1. Upload Question Bank PDF", type="pdf", key="qb_uploader")
    
    if question_bank_uploaded_file:
        qb_content = question_bank_uploaded_file.getvalue()
        if not st.session_state.extracted_questions or st.session_state.get('last_qb_name') != question_bank_uploaded_file.name:
            st.session_state.extracted_questions = extract_questions_from_pdf_cached(qb_content, google_api_key)
            st.session_state.last_qb_name = question_bank_uploaded_file.name
            st.session_state.qna_pairs = [] 
            st.session_state.current_answer = ""
            st.session_state.selected_question_for_answer = None
            if st.session_state.extracted_questions:
                st.success(f"Extracted {len(st.session_state.extracted_questions)} questions.")
            else:
                st.warning("No questions extracted.")
    
    material_uploaded_files = st.file_uploader(
        "2. Upload Material PDFs (Optional)", 
        accept_multiple_files=True, 
        type="pdf",
        key=f"material_uploader_{st.session_state.material_uploader_key}" 
    )

    if material_uploaded_files:
        material_contents = tuple(file.getvalue() for file in material_uploaded_files) 
        st.session_state.vector_store = create_vector_store_from_materials(material_contents, google_api_key)
        if st.session_state.vector_store:
            st.success(f"Processed {len(material_uploaded_files)} material file(s).")
    elif not material_uploaded_files and st.session_state.vector_store is not None:
        st.session_state.vector_store = None
        st.info("Material files cleared. Vector store reset.")
        st.session_state.material_uploader_key += 1 
        st.rerun()


st.header("Answer Generation")

if st.session_state.extracted_questions:
    selectbox_key = f"question_selector_{len(st.session_state.extracted_questions)}"
    
    current_selection_index = None
    if st.session_state.selected_question_for_answer and st.session_state.selected_question_for_answer in st.session_state.extracted_questions:
        current_selection_index = st.session_state.extracted_questions.index(st.session_state.selected_question_for_answer)

    new_selected_question = st.selectbox(
        "Select a question to answer:",
        st.session_state.extracted_questions,
        index=current_selection_index, 
        placeholder="Choose a question...",
        key=selectbox_key
    )

    if new_selected_question != st.session_state.selected_question_for_answer:
        st.session_state.selected_question_for_answer = new_selected_question
        st.session_state.current_answer = "" 
        existing_pair = next((item for item in st.session_state.qna_pairs if item["question"] == new_selected_question), None)
        if existing_pair:
            st.session_state.current_answer = existing_pair["answer"]


    if st.session_state.selected_question_for_answer:
        st.markdown(f"**Selected Question:** {st.session_state.selected_question_for_answer}")
        
        if st.button("🤖 Generate Answer", key="gen_answer_button"):
            with st.spinner("Generating answer... This may take a moment..."):
                answer_text = generate_answer(
                    st.session_state.selected_question_for_answer,
                    st.session_state.vector_store,
                    google_api_key
                )
                st.session_state.current_answer = answer_text
            
            existing_q_indices = [i for i, pair in enumerate(st.session_state.qna_pairs) if pair["question"] == st.session_state.selected_question_for_answer]
            if existing_q_indices:
                st.session_state.qna_pairs[existing_q_indices[0]]["answer"] = st.session_state.current_answer
            else:
                st.session_state.qna_pairs.append({
                    "question": st.session_state.selected_question_for_answer,
                    "answer": st.session_state.current_answer
                })
            st.rerun() 
        if st.session_state.current_answer:
            st.subheader("Generated Answer:")
            st.markdown(st.session_state.current_answer)
else:
    st.info("Please upload a Question Bank PDF in the sidebar to extract questions.")


if st.session_state.qna_pairs:
    st.divider()
    st.header("📋 Compiled Answer Key")
    for i, pair in enumerate(st.session_state.qna_pairs):
        with st.expander(f"Question {i+1}: {pair['question']}", expanded=False):
            st.markdown(f"**Answer:** {pair['answer']}")
else:
    st.info("No answers generated yet to display in the answer key.")


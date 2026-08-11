"""
📚 AI-Powered Answer Key Generator
Main Streamlit application — clean, professional SaaS layout.
"""
import streamlit as st
from src.utils.config import GOOGLE_API_KEY
from src.tools.pdf_extractor import extract_questions_from_pdf
from src.tools.answer_generator import generate_answer
from src.tools.pdf_generator import create_answer_key_pdf
from src.models.embeddings import create_vector_store

# ── App Configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Answer Key Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clean SaaS Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global Reset ── */
    :root {
        --primary: #0F172A;
        --secondary: #64748B;
        --accent: #4F46E5; /* Indigo */
        --accent-hover: #4338CA;
        --bg-main: #F8FAFC;
        --surface: #FFFFFF;
        --border: #E2E8F0;
        --radius: 8px;
    }

    .stApp {
        background-color: var(--bg-main) !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: var(--primary) !important;
    }
    
    /* ── Typography ── */
    h1, h2, h3 {
        color: var(--primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    .main-header {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: var(--secondary);
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] {
        border: 1px dashed #CBD5E1 !important;
        border-radius: var(--radius) !important;
        background-color: var(--surface) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: var(--radius) !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"] {
        background-color: var(--accent) !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--accent-hover) !important;
    }

    /* ── Expanders & Selectboxes ── */
    [data-testid="stExpander"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stSelectbox"] > div > div {
        background-color: var(--surface) !important;
        border-color: var(--border) !important;
    }

    /* ── Answer Bubble ── */
    .answer-content {
        background-color: var(--surface);
        border-left: 4px solid var(--accent);
        border-radius: 4px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        line-height: 1.6;
    }
    
    /* Hide default elements */
    #MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Guard: API key ────────────────────────────────────────────────────
if not GOOGLE_API_KEY:
    st.error("Error: `GOOGLE_API_KEY` not found. Please add it to your `.env` file.")
    st.stop()


# ── Session State ─────────────────────────────────────────────────────
defaults = {
    "extracted_questions": [],
    "vector_store": None,
    "qna_pairs": [],
    "selected_question_for_answer": None,
    "current_answer": "",
    "material_uploader_key": 0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ── Sidebar Controls (Inputs) ─────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2>📁 Data Sources</h2>", unsafe_allow_html=True)
    
    st.markdown("**1. Question Bank (Required)**")
    question_bank_uploaded_file = st.file_uploader(
        "Upload Question PDF",
        type="pdf",
        key="qb_uploader",
        label_visibility="collapsed"
    )

    if question_bank_uploaded_file:
        qb_content = question_bank_uploaded_file.getvalue()
        if (
            not st.session_state.extracted_questions
            or st.session_state.get("last_qb_name") != question_bank_uploaded_file.name
        ):
            with st.spinner("Extracting questions..."):
                st.session_state.extracted_questions = extract_questions_from_pdf(
                    qb_content, GOOGLE_API_KEY
                )
                st.session_state.last_qb_name = question_bank_uploaded_file.name
                st.session_state.qna_pairs = []
                st.session_state.current_answer = ""
                st.session_state.selected_question_for_answer = None

            if st.session_state.extracted_questions:
                st.success(f"Loaded {len(st.session_state.extracted_questions)} questions.")
            else:
                st.error("No questions found.")
    
    st.divider()
    
    st.markdown("**2. Reference Materials (Optional)**")
    material_uploaded_files = st.file_uploader(
        "Upload Context PDFs",
        accept_multiple_files=True,
        type="pdf",
        key=f"material_uploader_{st.session_state.material_uploader_key}",
        label_visibility="collapsed"
    )

    if material_uploaded_files:
        material_contents = tuple(f.getvalue() for f in material_uploaded_files)
        with st.spinner("Indexing references..."):
            st.session_state.vector_store = create_vector_store(
                material_contents, GOOGLE_API_KEY
            )
        if st.session_state.vector_store:
            st.success(f"Indexed {len(material_uploaded_files)} file(s).")
    elif not material_uploaded_files and st.session_state.vector_store is not None:
        st.session_state.vector_store = None
        st.session_state.material_uploader_key += 1
        st.rerun()


# ── Main Content Area ─────────────────────────────────────────────────
st.markdown('<div class="main-header">Answer Key Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Select a question from your uploaded bank to generate an AI-powered answer.</div>', unsafe_allow_html=True)

if not st.session_state.extracted_questions:
    st.info("👈 Please upload a Question Bank PDF in the sidebar to get started.")
else:
    # ── Answer Generation ──
    st.markdown("### Generate Answers")
    
    selectbox_key = f"question_selector_{len(st.session_state.extracted_questions)}"
    current_selection_index = None
    if (
        st.session_state.selected_question_for_answer
        and st.session_state.selected_question_for_answer
        in st.session_state.extracted_questions
    ):
        current_selection_index = st.session_state.extracted_questions.index(
            st.session_state.selected_question_for_answer
        )

    new_selected_question = st.selectbox(
        "Select a question to answer:",
        st.session_state.extracted_questions,
        index=current_selection_index,
        key=selectbox_key,
    )

    if new_selected_question != st.session_state.selected_question_for_answer:
        st.session_state.selected_question_for_answer = new_selected_question
        st.session_state.current_answer = ""
        existing_pair = next(
            (
                item
                for item in st.session_state.qna_pairs
                if item["question"] == new_selected_question
            ),
            None,
        )
        if existing_pair:
            st.session_state.current_answer = existing_pair["answer"]

    if st.button("Generate Answer", type="primary"):
        with st.spinner("Processing..."):
            answer_text = generate_answer(
                st.session_state.selected_question_for_answer,
                st.session_state.vector_store,
                GOOGLE_API_KEY,
            )
            st.session_state.current_answer = answer_text

        existing_q_indices = [
            i
            for i, pair in enumerate(st.session_state.qna_pairs)
            if pair["question"] == st.session_state.selected_question_for_answer
        ]
        if existing_q_indices:
            st.session_state.qna_pairs[existing_q_indices[0]]["answer"] = (
                st.session_state.current_answer
            )
        else:
            st.session_state.qna_pairs.append(
                {
                    "question": st.session_state.selected_question_for_answer,
                    "answer": st.session_state.current_answer,
                }
            )
        st.rerun()

    # Display Answer
    if st.session_state.current_answer:
        st.markdown(
            f'<div class="answer-content"><strong>Answer:</strong><br><br>{st.session_state.current_answer}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Review & Export ──
    if st.session_state.qna_pairs:
        st.markdown("### Compiled Answer Key")

        for i, pair in enumerate(st.session_state.qna_pairs):
            with st.expander(f"Q{i+1}: {pair['question'][:80]}...", expanded=False):
                st.markdown(f"**Question:**\n{pair['question']}")
                st.markdown("---")
                st.markdown(f"**Answer:**\n{pair['answer']}")

        st.markdown("<br>", unsafe_allow_html=True)
        
        pdf_path = create_answer_key_pdf(st.session_state.qna_pairs)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="Download PDF",
                data=pdf_file,
                file_name="answer_key.pdf",
                mime="application/pdf",
                type="secondary"
            )

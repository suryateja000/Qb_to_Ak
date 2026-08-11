# 📚 AI-Powered Answer Key Generator

An intelligent web application that automatically generates comprehensive answer keys from question bank PDFs using advanced AI technology. Built with Streamlit and powered by Google's Gemini AI, this tool revolutionizes how educators and students create and access answer keys.

### Live Demo : https://question-bank-to-answer-key.streamlit.app/

---

## 🌟 Features

- **Intelligent Question Extraction**: Automatically identifies and extracts questions from PDF documents.
- **AI-Powered Answer Generation**: Generates accurate, contextual answers using Google's latest Gemini 2.5 Flash model.
- **RAG (Retrieval-Augmented Generation)**: Utilizes uploaded reference materials (syllabus, notes) to ground the AI answers in specific context.
- **Professional UI/UX**: Clean, responsive SaaS-style layout built directly in Streamlit.
- **PDF Answer Key Export**: Generates professional PDF answer keys with proper formatting and Unicode character support.
- **Modular Architecture**: Clean `src/` based directory structure for maintainability and scalability.

## 📁 Project Structure

The project has been architected for modularity and ease of maintenance:

```text
Qb_to_Ak/
├── 📄 main.py                  # Main Streamlit application entry point
├── 📄 .env.example             # Environment variables template
├── 📄 requirements.txt         # Project dependencies
├── 📁 data/                    # Placeholder for local data/knowledge bases
├── 📁 logs/                    # Local execution logs
└── 📁 src/                     # Core logic modules
    ├── 📁 models/              # AI clients and vector store generation
    │   ├── llm_client.py
    │   └── embeddings.py
    ├── 📁 prompts/             # System prompts for LangChain
    │   └── system_prompts.py
    ├── 📁 tools/               # PDF extraction and answer generation tools
    │   ├── answer_generator.py
    │   ├── pdf_extractor.py
    │   └── pdf_generator.py
    └── 📁 utils/               # Configuration and helpers
        ├── config.py
        └── helpers.py
```

## 🛠️ Tech Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **LangChain**: AI application development framework
- **Google Gemini AI**: Large language model for answer generation
- **FAISS**: Vector database for document similarity search
- **PyPDFLoader / FPDF**: PDF document processing and generation

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Qb_to_Ak.git
cd Qb_to_Ak
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a `.env` file in the root directory and add your Google Gemini API key:
```bash
cp .env.example .env
```
Add your key inside `.env`:
```text
GOOGLE_API_KEY=your_api_key_here
```

### 4. Run the application
```bash
streamlit run main.py
```
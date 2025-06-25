# 📚 AI-Powered Answer Key Generator

An intelligent web application that automatically generates comprehensive answer keys from question bank PDFs using advanced AI technology. Built with Streamlit and powered by Google's Gemini AI, this tool revolutionizes how educators and students create and access answer keys.
### Live : https://question-bank-to-answer-key.streamlit.app/
## 🌟 Features

- **Intelligent Question Extraction**: Automatically identifies and extracts questions from PDF documents with smart mark assignment
- **AI-Powered Answer Generation**: Generates accurate, contextual answers using Google's Gemini AI model
- **RAG (Retrieval-Augmented Generation)**: Utilizes uploaded material PDFs to provide more accurate, context-aware answers
- **Interactive Question Selection**: Easy-to-use interface for selecting and answering questions individually
- **PDF Answer Key Export**: Generates professional PDF answer keys with proper formatting
- **Real-time Processing**: Streamlined workflow with caching for improved performance
- **Mark Assignment**: Automatically assigns appropriate marks (2m for short answers, 5m for long answers)

## 🛠️ Tech Stack

### Backend
- **Python**: Core programming language
- **Streamlit**: Web application framework
- **LangChain**: AI application development framework
- **Google Gemini AI**: Large language model for answer generation
- **FAISS**: Vector database for document similarity search
- **PyPDFLoader**: PDF document processing
- **FPDF**: PDF generation for answer keys

### AI & ML Components
- **GoogleGenerativeAI**: Text generation and embeddings
- **RecursiveCharacterTextSplitter**: Document chunking for optimal processing
- **Vector Store**: Semantic search capabilities for relevant context retrieval

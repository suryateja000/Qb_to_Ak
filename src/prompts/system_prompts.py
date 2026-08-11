"""
All prompt templates used by the application.
"""

QUESTION_EXTRACTION_PROMPT = """
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

QA_SYSTEM_PROMPT = """You are an expert assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer or the context is not relevant, state that you cannot answer based on the provided documents, and then try to answer based on your knowledge. \
Keep the answer concise and accurate. Highlight key information.

Context:
{context}

Question: {input}

Answer:"""

GENERAL_QA_PROMPT = """Answer the following question based on your general knowledge.
Question: {question}
Answer:"""

# Building a Privacy-First AI Document Analyzer Using Local RAG & LLMs

This project demonstrates a **privacy-first AI document analyzer** built using
Retrieval-Augmented Generation (RAG) and a **fully local LLM stack**.

## Key Features
- Upload PDF/DOCX documents
- Ask factual questions with source grounding
- Generate document-level summaries
- Hallucination-safe responses ("Not found" when applicable)
- No external AI APIs (no OpenAI, no cloud inference)

## Tech Stack
- Streamlit
- FAISS (local vector store)
- Sentence-Transformers (local embeddings)
- Ollama (local LLM inference)
- LangChain (chunking & orchestration)

## Privacy Guarantee
All document processing and AI inference occur locally.
No document content is sent to external services.

## How to Run Locally
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py

import streamlit as st
from utils.ingest import ingest_documents
from utils.qa import answer_question

st.set_page_config(page_title="AI Document Analyzer")

st.title("📄 AI Document Analyzer")

uploaded_files = st.file_uploader(
    "Upload documents (PDF or DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Indexing documents..."):
        vectorstore = ingest_documents(uploaded_files)
    st.success("Documents indexed successfully")

    question = st.text_input("Ask a question based on the documents")

    mode = st.radio("Response mode", ["Plain", "Comprehensive"])

    if question:
        with st.spinner("Generating answer..."):
            response, sources = answer_question(question, vectorstore, mode)
        st.subheader("Answer")
        st.write(response)

        st.subheader("Sources")
        for s in sources:
            st.write(f"- {s}")
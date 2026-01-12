from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def ingest_documents(files):
    texts = []
    metadatas = []

    for file in files:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    texts.append(text)
                    metadatas.append({
                        "source": file.name,
                        "page": i + 1
                    })

        elif file.name.endswith(".docx"):
            doc = Document(file)
            for i, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    texts.append(para.text)
                    metadatas.append({
                        "source": file.name,
                        "section": i + 1
                    })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = splitter.create_documents(texts, metadatas=metadatas)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

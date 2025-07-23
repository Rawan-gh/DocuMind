import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from tempfile import NamedTemporaryFile

# Arabic-compatible embedding model (you can choose others too)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_pdf_and_create_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(pages, embeddings)
    return db

def answer_question(db, question):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    return qa.run(question)

def main():
    st.set_page_config(page_title="DocuMind", layout="centered")
    st.title("🧠 DocuMind")
    st.write("Ask your PDF a question – now supports Arabic and English!")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.success("PDF uploaded successfully!")

        question = st.text_input("💬 Ask a question about the PDF (Arabic or English):")

        if st.button("Get Answer") and question:
            with st.spinner("Analyzing the document..."):
                db = load_pdf_and_create_vectorstore(tmp_path)
                answer = answer_question(db, question)
                st.markdown("### ✅ Answer:")
                st.write(answer)

if __name__ == "__main__":
    main()

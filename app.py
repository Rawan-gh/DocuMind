import streamlit as st
from helper import load_pdf_and_create_qa
from tempfile import NamedTemporaryFile

# Page setup
st.set_page_config(page_title="DocuMind - PDF Assistant", layout="wide")

# Custom CSS for professional styling
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-title {
        font-size: 36px;
        font-weight: 600;
        color: #1a237e;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 40px;
    }
    .stTextInput > div > div > input {
        border: 1px solid #ced4da;
        border-radius: 6px;
        padding: 10px;
    }
    .stTextInput > label {
        font-weight: 500;
        color: #343a40;
    }
    .stMarkdown h3 {
        color: #0d47a1;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">DocuMind</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload any PDF and ask questions based on its content</div>', unsafe_allow_html=True)

# PDF Upload
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("PDF uploaded successfully. You may now ask your questions.")

    with st.spinner("Processing the document... Please wait."):
        qa = load_pdf_and_create_qa(tmp_path)

    question = st.text_input("Enter your question:")

    if question:
        with st.spinner("Generating answer..."):
            answer = qa.run(question)

        st.markdown("### Answer:")
        st.write(answer)

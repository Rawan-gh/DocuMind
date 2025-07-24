# app.py
import streamlit as st
from helper import load_pdf_and_create_qa
from tempfile import NamedTemporaryFile
import time
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="DocuMind - PDF Assistant", layout="wide")
st.title("DocuMind - PDF Q&A and Analysis")

# ---- Session State ----
if "history" not in st.session_state:
    st.session_state.history = []
if "pdf_info" not in st.session_state:
    st.session_state.pdf_info = {}
if "qa" not in st.session_state:
    st.session_state.qa = None

tab1, tab2 = st.tabs(["Chat with PDF", "Analysis"])

with tab1:
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

    if uploaded_file is not None:
        # re-build only if new file
        if st.session_state.qa is None or st.session_state.pdf_info.get("file_name") != uploaded_file.name:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            st.session_state.pdf_info["file_name"] = uploaded_file.name
            st.success(f"PDF `{uploaded_file.name}` uploaded successfully.")

            with st.spinner("Processing the document..."):
                st.session_state.qa = load_pdf_and_create_qa(tmp_path)

        question = st.text_input("Enter your question:")
        if question:
            start = time.time()
            with st.spinner("Generating answer..."):
                answer = st.session_state.qa.run(question)
            rt = round(time.time() - start, 2)

            st.markdown("### Answer:")
            st.write(answer)

            st.session_state.history.append({
                "q": question,
                "a": answer,
                "rt": rt,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

with tab2:
    st.header("PDF Analysis & Usage Statistics")

    if st.session_state.pdf_info.get("file_name"):
        st.write(f"**File Name:** {st.session_state.pdf_info['file_name']}")
        st.write(f"**Questions Asked:** {len(st.session_state.history)}")

        if st.session_state.history:
            st.subheader("Q&A History")
            for i, item in enumerate(st.session_state.history, 1):
                st.markdown(f"**{i}. Question:** {item['q']}")
                st.markdown(f"**Answer:** {item['a']}")
                st.markdown(f"*Time:* {item['ts']} • *Response time:* {item['rt']} sec")
                st.markdown("---")

            st.subheader("Table View")
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df)
        else:
            st.info("No questions asked yet.")
    else:
        st.info("Please upload a PDF and ask questions to see statistics here.")

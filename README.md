# DocuMind – Chat with your PDF (No OpenAI)

DocuMind is a Streamlit app that lets you upload any PDF and ask questions about it.  
It uses **Hugging Face models** (no OpenAI, no API keys required):

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** `google/flan-t5-small`

## Features
- Upload a PDF and query it
- Retrieval-Augmented Generation (RAG) using FAISS
- Analysis tab: questions history, response times, etc.

## Setup

```bash
# (Optional) conda
conda create -n documind python=3.10 -y
conda activate documind

# Install
pip install -r requirements.txt

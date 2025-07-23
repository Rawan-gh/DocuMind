Ask Your PDF
A simple Streamlit-based app that lets you upload a PDF and ask questions about its content in any language, including Arabic and English.
🌟 Features

Upload a PDF file
Ask questions about the PDF content
Get answers using an open-source AI model (distilgpt2)
Supports multilingual text with a robust embedding model

🛠️ Installation

Clone the repository or download the files.
Install the required dependencies:pip install -r requirements.txt


Run the app:streamlit run app.py



🛠️ Usage

Upload a PDF file using the file uploader.
Enter a question about the PDF content in the text input.
View the AI-generated answer based on the PDF.

📝 Notes

Uses distilgpt2 for question answering, a lightweight open-source model.
Supports multilingual PDFs (e.g., Arabic, English) via a multilingual embedding model.
No OpenAI API key is required, making it free to use with local computation.
For better performance, consider using a larger model like gpt2-medium (update model_id in app.py).

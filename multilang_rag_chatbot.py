import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import tempfile
from datetime import datetime

# Setup logging folder and file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "user_logs.txt")

def log_interaction(user, email, filename, query, response):
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | User: {user} | Email: {email} | File: {filename}\n")
        f.write(f"Query: {query}\nResponse: {response}\n{'-'*40}\n")

st.title("📄 Multilingual RAG Chatbot")

user_name = st.text_input("Your Name")
user_email = st.text_input("Your Email")

if not user_name or not user_email:
    st.warning("Please enter your name and email.")
    st.stop()

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Using Groq LLM with your API key from secrets
    llm = ChatGroq(groq_api_key=st.secrets["groq"]["GROQ_API_KEY"], model_name="llama-3.3-70b-versatile", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    query = st.text_input("Ask your question:")

    if query:
        response = qa.run(query)
        st.markdown(f"**Answer:** {response}")

        log_interaction(user_name, user_email, uploaded_file.name, query, response)

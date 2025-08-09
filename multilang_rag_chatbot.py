import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import tempfile
import csv
from datetime import datetime

# Set Groq API key
os.environ["GROQ_API_KEY"] = st.secrets["groq"]["GROQ_API_KEY"]

st.set_page_config(page_title="📄 Multilingual RAG Chatbot (Groq)")
st.title("📄 Multilingual RAG Chatbot with Groq")

# Sidebar user info
st.sidebar.header("🔐 User Login")
user_name = st.sidebar.text_input("Your Name")
user_email = st.sidebar.text_input("Email")

if not user_name or not user_email:
    st.warning("Please enter your name and email.")
    st.stop()

# File upload
uploaded_file = st.file_uploader("📎 Upload PDF", type=["pdf"])

if uploaded_file:
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_name = uploaded_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load and split
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Embeddings (free & local from HuggingFace)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = FAISS.from_documents(chunks, embeddings)

    # LLM (Groq)
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    st.success(f"📄 {file_name} loaded. Ask your question!")

    query = st.text_input("💬 Ask your question (any language):")
    if query:
        response = qa.run(query)
        st.markdown(f"**Answer:** {response}")

        # Save log
        log_file = "usage_log.csv"
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Time", "User Name", "Email", "File Name", "Query", "Response"])
            writer.writerow([upload_time, user_name, user_email, file_name, query, response])



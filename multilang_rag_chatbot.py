import streamlit as st
import os
import tempfile
from langchain_community.chat_models import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from datetime import datetime

# Set page title
st.set_page_config(page_title="Multi-Language RAG Chatbot", page_icon="🌍")

# User inputs for logging
user_name = st.text_input("Enter your name")
user_email = st.text_input("Enter your email")

# Upload file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_input("Ask a question about the document")

# Function to log data to a text file
def log_user_interaction(name, email, file_name, query_text):
    log_file = "user_interactions.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | Name: {name} | Email: {email} | File: {file_name} | Query: {query_text}\n")

# Ensure API key is loaded
groq_api_key = st.secrets["groq"]["GROQ_API_KEY"]

if uploaded_file and query and user_name and user_email:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = ChatGroq(model="llama3-8b-8192", temperature=0, groq_api_key=groq_api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    response = qa.run(query)

    # Log user input and file info
    log_user_interaction(user_name, user_email, uploaded_file.name, query)

    st.write("### Response:")
    st.write(response)

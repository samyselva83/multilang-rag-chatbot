import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os
import datetime

# ------------------ SETTINGS ------------------ #
GROQ_API_KEY = st.secrets["groq"]["GROQ_API_KEY"]

# ------------------ STREAMLIT UI ------------------ #
st.set_page_config(page_title="Multilang RAG Chatbot", layout="wide")
st.title("🌍 Multilang RAG Chatbot")

# User info inputs
user_name = st.text_input("Enter your Name")
user_email = st.text_input("Enter your Email")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Query input
query = st.text_area("Ask your question")

if st.button("Submit Query"):
    if not uploaded_file:
        st.error("Please upload a PDF file first.")
    elif not query.strip():
        st.error("Please enter a query.")
    else:
        # ------------------ PDF PROCESSING ------------------ #
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyMuPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # Use HuggingFace embeddings (no API key needed)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Groq LLM for answering
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        response = qa.run(query)

        # ------------------ LOGGING ------------------ #
        log_entry = (
            f"{datetime.datetime.now()} | Name: {user_name} | Email: {user_email} | "
            f"File: {uploaded_file.name} | Query: {query} | Response: {response}\n"
        )
        with open("user_logs.txt", "a", encoding="utf-8") as log_file:
            log_file.write(log_entry)

        # Show result
        st.success(response)



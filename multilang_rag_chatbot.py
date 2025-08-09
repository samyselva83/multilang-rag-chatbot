import streamlit as st
from langchain_community.chat_models import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from datetime import datetime
import os

# -------------------
# API Key (from Streamlit Secrets)
# -------------------
groq_api_key = st.secrets["groq"]["GROQ_API_KEY"]

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Multilingual RAG Chatbot", layout="wide")
st.title("🌐 Multilingual RAG Chatbot (GROQ)")

st.sidebar.header("Upload and Settings")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
language = st.sidebar.text_input("Language (e.g., en, fr, de, hi)", value="en")
query = st.text_area("Ask a question based on the document:")

# -------------------
# Logging setup
# -------------------
LOG_FILE = "usage_log.txt"

def log_usage(user_question, bot_answer):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] Q: {user_question}\nA: {bot_answer}\n{'-'*50}\n")

# -------------------
# Document Processing
# -------------------
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyMuPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # -------------------
    # LLM with GROQ
    # -------------------
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    if st.button("Get Answer"):
        if query.strip():
            result = qa.invoke({"query": query})
            answer = result["result"]
            st.write("**Answer:**", answer)

            # Show source documents (optional)
            with st.expander("Source Chunks"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {i+1}:** {doc.page_content[:200]}...")

            # Log usage
            log_usage(query, answer)
        else:
            st.warning("Please enter a question.")
else:
    st.info("Upload a PDF to start.")

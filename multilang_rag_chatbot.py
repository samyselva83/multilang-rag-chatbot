import os
import streamlit as st
import uuid
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

st.set_page_config(page_title="📄 Multi Language RAG Chatbot")
st.title("📄 AskDocs AI")
st.header(" Multi-Language RAG AI Chatbot")
st.markdown(" by Selvakumar")

# Sidebar user info
st.sidebar.header("🔐 User Login")
user_name = st.sidebar.text_input("Your Name", key="user_name")
user_email = st.sidebar.text_input("Email", key="user_email")

if not user_name or not user_email:
    st.warning("Please enter your name and email.")
    st.stop()

uploaded_file = st.file_uploader("📎 Upload PDF", type=["pdf"], key="file_uploader")

# Initialize session state for vectorstore and QA chain
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "upload_time" not in st.session_state:
    st.session_state.upload_time = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

if uploaded_file:
    # Generate unique temp file name per upload to avoid collisions
    unique_id = str(uuid.uuid4())
    temp_pdf_path = f"temp_{unique_id}.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load and split only if vectorstore is not created or new file uploaded
    if st.session_state.file_name != uploaded_file.name:
        loader = PyMuPDFLoader(temp_pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

        # Save to session state
        st.session_state.vectorstore = vectorstore
        st.session_state.qa_chain = qa
        st.session_state.upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.file_name = uploaded_file.name

        st.success(f"📄 {uploaded_file.name} loaded and processed successfully with {len(chunks)} chunks.")
    else:
        st.success(f"📄 {uploaded_file.name} already loaded.")

    # Remove the temp file after loading (optional)
    os.remove(temp_pdf_path)

    query = st.text_input("💬 Ask your question (any language):", key="query_input")

    if st.button("Submit Query") and query.strip() != "":
        if st.session_state.qa_chain:
            response = st.session_state.qa_chain.run(query)
            st.markdown(f"**Answer:** {response}")

            # Log user interaction
            log_file = "usage_log.csv"
            file_exists = os.path.isfile(log_file)
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Time", "User Name", "Email", "File Name", "Query", "Response"])
                writer.writerow([st.session_state.upload_time, user_name, user_email, st.session_state.file_name, query, response])
        else:
            st.error("Please upload and process a PDF first.")










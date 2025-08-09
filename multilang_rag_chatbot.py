import os
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory

# ==========================
# Setup Logging
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "user_logs.txt")

def log_interaction(user_input, bot_response):
    """Append conversation logs with timestamps."""
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(f"{datetime.now()} - USER: {user_input}\n")
        log_file.write(f"{datetime.now()} - BOT: {bot_response}\n\n")

# ==========================
# Load Documents
# ==========================
def load_documents(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    return loader.load()

# ==========================
# Split Documents
# ==========================
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# ==========================
# Build Vector Store
# ==========================
def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

# ==========================
# Create Conversational Chain
# ==========================
def create_chatbot(vector_store):
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.3, "max_length": 512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm, vector_store.as_retriever(), memory=memory)

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    pdf_file = "sample.pdf"  # Change to your PDF path

    print("Loading documents...")
    docs = load_documents(pdf_file)

    print("Splitting documents...")
    chunks = split_documents(docs)

    print("Building vector store...")
    vector_store = build_vector_store(chunks)

    print("Creating chatbot...")
    chatbot = create_chatbot(vector_store)

    print("\nChatbot is ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = chatbot({"question": user_input})
        bot_reply = response["answer"]

        print(f"Bot: {bot_reply}")
        log_interaction(user_input, bot_reply)

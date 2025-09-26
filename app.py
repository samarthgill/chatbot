import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

# -------------------------------
# CONFIGURATION
# -------------------------------
PDF_PATH = "docs/medical.pdf"        # Your PDF file
FAISS_INDEX_PATH = "faiss_index"     # Path to save/load FAISS
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama2:latest"

# -------------------------------
# LOAD / CREATE VECTORSTORE
# -------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(FAISS_INDEX_PATH):
        # Allow loading trusted pickle
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        docs = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)

    return vectorstore

# -------------------------------
# CREATE QA CHAIN
# -------------------------------
@st.cache_resource
def init_qa_chain():
    vectorstore = load_vectorstore()
    llm = Ollama(model=LLM_MODEL, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,
    )
    return qa_chain

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Medical QA Bot", page_icon="🩺")
st.title("💊 Medical Chat Bot")

qa_chain = init_qa_chain()

query = st.text_input("Ask me anything about medicine:")

if query:
    with st.spinner("Searching answer..."):
        response = qa_chain.invoke({"query": query})
    st.markdown("**Answer:**")
    st.write(response['result'])

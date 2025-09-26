from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader

# Paths
PDF_FOLDER = Path("data")
FAISS_DIR = Path("faiss_index")

# Load PDFs and split
def load_docs():
    docs = []
    for pdf_file in PDF_FOLDER.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())
    return docs

# Initialize embeddings
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

# Initialize or load FAISS
def init_vectorstore(embeddings):
    if FAISS_DIR.exists():
        vectorstore = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    else:
        docs = load_docs()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(str(FAISS_DIR))
    return vectorstore

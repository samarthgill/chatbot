# 🏥 Medical Chat Bot

A **Medical Chat Bot** built with **LangChain, FAISS, and Ollama** that allows users to ask medical-related queries.  
The bot retrieves answers from medical documents (PDFs, books) using **embeddings** and provides fast, accurate responses.  

---

## 🚀 Features
- 📚 Upload & process medical books/documents  
- 🔍 Semantic search with FAISS  
- 💬 Interactive chat UI (built with Streamlit + HTML template)  
- ⚡ Fast responses using local embeddings and LLM  

---
📂 Project Structure
medical-chat-bot/
│── app.py              # Main Streamlit app
│── requirements.txt    # Dependencies
│── README.md           # Project description
│── .gitignore          # Ignore unnecessary files
│
├── data/               # Store medical PDFs
├── faiss_index/        # Auto-generated FAISS DB
├── templates/          # Chat UI (HTML)
└── static/             # CSS, images

🚀 Installation

1️⃣ Clone this repository

git clone https://github.com/samarthgill/chatbot.git

cd chatbot

2️⃣ Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate     # Windows  
source venv/bin/activate  # Linux/Mac  

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Run the Streamlit app

streamlit run app.py

🧰 Tech Stack

Python 🐍

Streamlit → Frontend UI

LangChain → Orchestration

FAISS → Vector Search

Ollama → Local LLMs (Llama2 / Llama3)

⚡ Performance Optimization

Embeddings stored in FAISS index → Faster queries

Uses local Ollama models → No API cost, runs offline

Cache enabled to avoid recomputation

⚠️ Disclaimer

This chatbot is for educational and informational purposes only.
It is NOT a substitute for professional medical advice.
Always consult a qualified doctor for health concerns.

🤝 Contributing

Pull requests are welcome! If you’d like to add features, improve UI, or optimize performance, fork the repo and submit a PR.

⭐ Support

If you like this project, please give it a ⭐ on GitHub to support further development!

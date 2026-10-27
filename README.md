# DocuMind 🧠

A lightweight, privacy-first Retrieval-Augmented Generation (RAG) web application to upload documents and query them locally using FAISS and HuggingFace LLMs.

---

## 🚀 Features

- **Document Ingestion:** Upload and index `.pdf` and `.txt` files seamlessly.
- **Fast Semantic Search:** Vector embeddings powered by `sentence-transformers/all-MiniLM-L6-v2` and indexed via **FAISS**.
- **Local Generation:** Context-grounded response generation using Hugging Face causal language models (e.g., `facebook/opt-125m`).
- **Interactive UI:** Clean, dark-mode web interface built with Flask for uploading documents and running real-time queries.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Vector Search & Indexing:** FAISS (`faiss-cpu`), NumPy
- **Embeddings & LLM:** Hugging Face `transformers`, `sentence-transformers`, PyTorch
- **Document Parsing:** PyPDF

---

## 📦 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/kavin120/DocuMind.git
cd DocuMind
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 📄 License
MIT License

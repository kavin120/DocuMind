import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pypdf  # Added for PDF support

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / "documents"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max

# Ensure directories exist
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)

# Configuration
INDEX_PATH = Path(__file__).parent / "faiss_index.bin"
EMBEDDINGS_PATH = Path(__file__).parent / "embeddings.npy"
CHUNKS_PATH = Path(__file__).parent / "chunks.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "google/flan-t5-base" 
TOP_K = 4

# Global models (lazy loaded to speed up startup)
embedder = None
llm_tokenizer = None
llm_model = None

def get_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer(EMBEDDING_MODEL)
    return embedder

def get_llm():
    global llm_tokenizer, llm_model
    if llm_model is None:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    return llm_tokenizer, llm_model

def load_and_chunk(docs_dir: Path, chunk_size: int = 400, chunk_overlap: int = 80) -> list:
    chunks = []
    for filepath in docs_dir.glob("*"):
        if filepath.suffix.lower() not in ['.txt', '.pdf']:
            continue
            
        text = ""
        if filepath.suffix.lower() == '.txt':
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        elif filepath.suffix.lower() == '.pdf':
            try:
                reader = pypdf.PdfReader(filepath)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            except Exception as e:
                print(f"Error reading PDF {filepath.name}: {e}")
                continue

        # Clean multiple newlines and whitespace
        clean_text = " ".join(text.split())
        if not clean_text:
            continue

        # Chunk with sliding window overlap
        start = 0
        while start < len(clean_text):
            end = start + chunk_size
            chunk = clean_text[start:end].strip()
            if chunk:
                chunks.append({"text": chunk, "source": filepath.name})
            start += chunk_size - chunk_overlap
    return chunks

def build_index():
    chunks = load_and_chunk(app.config['UPLOAD_FOLDER'])
    if not chunks:
        return False
    
    emb = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = emb.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return True

def load_index():
    if not INDEX_PATH.exists():
        if not build_index():
            return None, None
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    allowed_exts = ['.txt', '.pdf']
    if file and any(file.filename.lower().endswith(ext) for ext in allowed_exts):
        # Clear previous documents so only the new file is indexed
        for old_file in app.config['UPLOAD_FOLDER'].glob("*"):
            if old_file.is_file():
                try:
                    old_file.unlink()
                except Exception as e:
                    print(f"Error removing old file {old_file.name}: {e}")

        # Remove old index artifacts
        for artifact in [INDEX_PATH, EMBEDDINGS_PATH, CHUNKS_PATH]:
            if artifact.exists():
                try:
                    artifact.unlink()
                except Exception as e:
                    print(f"Error removing {artifact.name}: {e}")

        filename = secure_filename(file.filename)
        file.save(app.config['UPLOAD_FOLDER'] / filename)
        # Rebuild index on new upload
        build_index()
        return jsonify({'message': f'Previous documents cleared. {filename} uploaded and indexed successfully.'})
    return jsonify({'error': 'Only .txt and .pdf files are allowed'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get('question')
    if not query:
        return jsonify({'error': 'Question is required'}), 400

    idx, chunks = load_index()
    if idx is None:
        return jsonify({'error': 'No documents indexed. Please upload a file first.'}), 400

    # Retrieve
    emb = get_embedder()
    q_emb = emb.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")
    distances, ids = idx.search(q_emb, TOP_K)
    
    retrieved = []
    for i, score in zip(ids[0], distances[0]):
        if i >= 0:
            retrieved.append(chunks[i])
            
    if not retrieved:
        return jsonify({'answer': 'I could not find any relevant information.'})

    # Generate Answer
    context = "\n\n".join([c["text"] for c in retrieved])
    prompt = f"Please read the following context and answer the question accurately.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    tokenizer, model = get_llm()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    sources = ", ".join(list(set(c["source"] for c in retrieved)))

    return jsonify({
        'answer': answer, 
        'sources': sources,
        'context_snippets': [c["text"] for c in retrieved[:2]]
    })

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)

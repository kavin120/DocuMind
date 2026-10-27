import os
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Simple RAG pipeline
# 1. Load documents (plain text files) from a folder
# 2. Compute embeddings using SentenceTransformer
# 3. Store embeddings in a FAISS index
# 4. Retrieve top-k relevant chunks for a query
# 5. Generate answer using an LLM conditioned on retrieved context

# Configuration
DOCS_DIR = Path(__file__).parent / "documents"
INDEX_PATH = Path(__file__).parent / "faiss_index.bin"
EMBEDDINGS_PATH = Path(__file__).parent / "embeddings.npy"
CHUNKS_PATH = Path(__file__).parent / "chunks.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # replace with any HF causal LM you have access to
TOP_K = 5

# Helper to load and chunk documents
def load_and_chunk(docs_dir: Path, chunk_size: int = 200) -> list:
    """Load .txt files and split them into chunks of roughly `chunk_size` characters.
    Returns a list of dicts with keys: 'text', 'source' (filename)."""
    chunks = []
    for txt_file in docs_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        # naive split by paragraph, then further split if needed
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            if len(para) <= chunk_size:
                chunks.append({"text": para, "source": txt_file.name})
            else:
                # split long paragraph into fixed-size windows (overlap optional)
                for i in range(0, len(para), chunk_size):
                    chunk_text = para[i : i + chunk_size]
                    chunks.append({"text": chunk_text, "source": txt_file.name})
    return chunks

# Build or load FAISS index
def build_index():
    print("🔨 Building FAISS index...")
    chunks = load_and_chunk(DOCS_DIR)
    if not chunks:
        raise RuntimeError(f"No .txt documents found in {DOCS_DIR}")

    # Compute embeddings
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product for normalized vectors
    index.add(embeddings)

    # Save index and auxiliary data
    faiss.write_index(index, str(INDEX_PATH))
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ Index built with {len(chunks)} chunks.")
    return index, chunks


def load_index():
    if not INDEX_PATH.exists():
        return build_index()
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

# Retrieve top‑k relevant chunks for a query
def retrieve(query: str, index, chunks, k: int = TOP_K):
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")
    distances, ids = index.search(q_emb, k)
    results = []
    for idx, score in zip(ids[0], distances[0]):
        if idx < 0:
            continue
        results.append({"text": chunks[idx]["text"], "source": chunks[idx]["source"], "score": float(score)})
    return results

# Generate answer using the LLM
def generate_answer(query: str, retrieved_chunks):
    # Concatenate retrieved contexts
    context = "\n\n".join([c["text"] for c in retrieved_chunks])
    prompt = (
        f"You are an assistant answering questions using the provided context.\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:" 
    )
    # Load LLM pipeline (text generation)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype="auto", device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, do_sample=True, temperature=0.7)
    answer = generator(prompt, return_full_text=False)[0]["generated_text"].strip()
    return answer

def answer_question(query: str):
    index, chunks = load_index()
    retrieved = retrieve(query, index, chunks)
    if not retrieved:
        return "I couldn't find any relevant information."
    answer = generate_answer(query, retrieved)
    sources = ", ".join({c["source"] for c in retrieved})
    return f"{answer}\n\nSources: {sources}"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple RAG QA system")
    parser.add_argument("question", nargs="?", help="Question to answer. If omitted, an interactive prompt is shown.")
    args = parser.parse_args()

    if args.question:
        print(answer_question(args.question))
    else:
        while True:
            try:
                q = input("\nEnter a question (or 'exit' to quit): ")
                if q.lower() in {"exit", "quit"}:
                    break
                print(answer_question(q))
            except KeyboardInterrupt:
                break

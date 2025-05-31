import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np

# Load real model once globally
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    return np.array(embeddings, dtype="float32")
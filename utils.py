import fitz  # PyMuPDF
import numpy as np
from openai import OpenAI # Using OpenAI for embeddings

def load_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=1000, overlap=100, max_chunks=500):
    chunks = []
    start = 0
    while start < len(text) and len(chunks) < max_chunks:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks, client: OpenAI, model_name: str):
    # Ensure chunks is a list of strings
    if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
        raise ValueError("Input 'chunks' must be a list of strings.")
    if not chunks:
        return np.array([]) # Return empty array if no chunks

    try:
        response = client.embeddings.create(
            input=chunks,
            model=model_name
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    except Exception as e:
        print(f"[ERROR] Failed to get embeddings from OpenAI: {e}")
        # Depending on how you want to handle errors, you might re-raise or return None/empty
        raise
    return np.array(embeddings, dtype="float32")
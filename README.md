# RAG Chatbot for Clinical Guidelines

This is a Retrieval-Augmented Generation (RAG) chatbot that uses:
- Clinical guidelines in PDF format
- FAISS for vector search
- Sentence Transformers for embeddings
- Hugging Face LLM (`google/flan-t5-base`)
- Gradio UI for interaction

## Project Structure

```
rag-chatbot/
├── data/
│   └── clinical_guidelines.pdf
├── rag_pipeline.py
├── utils.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup & Run

```bash
pip install -r requirements.txt
python app.py
```

## Deploy to Hugging Face Spaces
- Upload all files
- Use `app.py` as the entry point

from utils import load_pdf_text, chunk_text, embed_chunks
from config_loader import load_config
from transformers import pipeline, AutoTokenizer
import faiss
import numpy as np

class RAGPipeline:
    def __init__(self, pdf_path, config_path="config.yaml"):
        print(f"[INFO] Loading config from: {config_path}")
        self.config = load_config(config_path)

        print(f"[INFO] Reading and processing PDF: {pdf_path}")
        self.text = load_pdf_text(pdf_path)
        self.chunks = chunk_text(self.text)
        print(f"[INFO] Total chunks created: {len(self.chunks)}")

        print("[INFO] Embedding chunks...")
        self.embeddings = embed_chunks(self.chunks)

        print("[INFO] Building FAISS vector index...")
        self.index = self.build_faiss_index(self.embeddings)
        print("[INFO] Vector index built and populated.")

        model_name = self.config["model"]["name"]
        print(f"[INFO] Loading model: {model_name}")
        self.qa_pipeline = pipeline(
            task=self.config["model"]["task"],
            model=model_name,
            trust_remote_code=self.config["model"].get("trust_remote_code", False)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config["model"].get("trust_remote_code", False)
        )
        self.max_input_tokens = min(getattr(self.tokenizer, 'model_max_length', 512), 2048)

    def build_faiss_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def query(self, question):
        print(f"[QUERY] Received question: {question}")
        question_embedding = embed_chunks([question])
        top_k = self.config["retrieval"]["top_k"]
        _, I = self.index.search(np.array(question_embedding), top_k)

        print(f"[INFO] Retrieved top {top_k} chunks for context:")
        context = "\n".join([self.chunks[i] for i in I[0]])
        for idx in I[0]:
            print(f" - Chunk {idx}: {self.chunks[idx][:100]}...")

        prompt = self.config["prompt"]["template"].format(context=context, question=question)
        tokenized = self.tokenizer(prompt, truncation=True, max_length=self.max_input_tokens, return_tensors="pt")
        truncated_prompt = self.tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)

        print("[INFO] Final prompt sent to the model:")
        print(truncated_prompt)

        result = self.qa_pipeline(truncated_prompt, max_new_tokens=self.config["model"]["max_new_tokens"])
        response = result[0]["generated_text"]

        print(f"[RESPONSE] {response}")
        return response

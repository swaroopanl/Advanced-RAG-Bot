import os
import psutil
import faiss
import numpy as np
import os
from langchain_openai import ChatOpenAI
from openai import OpenAI # For OpenAI embeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import load_pdf_text, chunk_text, embed_chunks
from config_loader import load_config

def log_system_usage(label="[INFO]"):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # in MB
    cpu = process.cpu_percent(interval=0.2)
    print(f"{label} | Memory: {mem:.2f} MB | CPU: {cpu:.2f}%")

class RAGPipeline:
    def __init__(self, pdf_path, config_path="config.yaml"):
        print(f"[INFO] Loading config from: {config_path}")
        self.config = load_config(config_path)

        print(f"[INFO] Reading and processing PDF: {pdf_path}")
        log_system_usage("[MEM/CU] Before loading PDF")
        self.text = load_pdf_text(pdf_path)
        log_system_usage("[MEM/CU] After loading PDF")

        chunk_size = self.config["chunking"]["size"]
        overlap = self.config["chunking"]["overlap"]
        # Read chunking parameters correctly from config
        # chunk_size is already correctly read as self.config["chunking"]["size"]
        current_overlap = self.config["chunking"]["overlap"] 
        current_max_chunks = self.config["chunking"]["max_chunks"]

        self.chunks = chunk_text(self.text, chunk_size, current_overlap, current_max_chunks)
        print(f"[INFO] Total chunks created: {len(self.chunks)}")
        log_system_usage("[MEM/CU] After chunking")

        # Load embedding configuration
        embedding_config = self.config.get("embedding")
        if embedding_config is None:
            print("[ERROR] Missing 'embedding' section in config.yaml.")
            raise ValueError("Configuration Error: 'embedding' section is missing in config.yaml.")

        embedding_provider = embedding_config.get("provider")
        self.embedding_model_name = embedding_config.get("model_name") # Store for use in query method
        embedding_model_name = self.embedding_model_name # Use local variable for immediate use

        if embedding_provider != "openai" or not self.embedding_model_name:
            print(f"[ERROR] Invalid or missing embedding provider/model in config.yaml. Expected provider 'openai' and a 'model_name'. Found: {embedding_provider}, {embedding_model_name}")
            raise ValueError("Configuration Error: Invalid embedding configuration in config.yaml.")

        print(f"[INFO] Generating embeddings using OpenAI model: {self.embedding_model_name}...")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] OPENAI_API_KEY environment variable not found.")
            raise ValueError("OPENAI_API_KEY environment variable not found. Please set it before running the application.")

        # OPENAI_API_KEY environment variable must be set for OpenAI client to work
        try:
            self.openai_embed_client = OpenAI(api_key=api_key)
            self.embeddings = embed_chunks(self.chunks, client=self.openai_embed_client, model_name=self.embedding_model_name)
            print("[INFO] Embeddings generated successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize OpenAI client or generate embeddings: {e}")
            # The specific error from OpenAI client (like authentication) will be more informative now
            raise
        log_system_usage("[MEM/CU] After embeddings")

        print("[INFO] Building FAISS vector index...")
        log_system_usage("[MEM/CU] Before FAISS index")
        self.index = self.build_faiss_index(self.embeddings)
        print("[INFO] Vector index built and populated.")
        log_system_usage("[MEM/CU] After FAISS index")

        # Initialize OpenAI LLM
        openai_config = self.config.get("openai")
        if openai_config is None:
            print("[ERROR] Missing 'openai' section in config.yaml. This section is required for LLM initialization.")
            raise ValueError("Configuration Error: 'openai' section is missing in config.yaml.")

        model_name = openai_config.get("model_name")
        if model_name is None:
            print("[ERROR] Missing 'model_name' under 'openai' section in config.yaml.")
            raise ValueError("Configuration Error: 'openai.model_name' is missing in config.yaml.")

        temperature = openai_config.get("temperature") # Can be 0.0, so check for None explicitly
        if temperature is None:
            print("[ERROR] Missing 'temperature' under 'openai' section in config.yaml.")
            raise ValueError("Configuration Error: 'openai.temperature' is missing in config.yaml.")
        
        max_output_tokens = openai_config.get("max_output_tokens")
        if max_output_tokens is None:
            print("[ERROR] Missing 'max_output_tokens' under 'openai' section in config.yaml.")
            raise ValueError("Configuration Error: 'openai.max_output_tokens' is missing in config.yaml.")

        print(f"[INFO] Initializing OpenAI model: {model_name}")
        
        # api_key should have been loaded by the embedding client part, but good to ensure it's available
        # or re-fetch if this part could be initialized independently in a different context (though not currently the case)
        if not api_key: # api_key was fetched before embedding client initialization
            api_key = os.getenv("OPENAI_API_KEY") # Attempt to fetch again if somehow not set
            if not api_key:
                print("[ERROR] OPENAI_API_KEY environment variable not found when initializing ChatOpenAI.")
                raise ValueError("OPENAI_API_KEY environment variable not found. Please set it before running the application.")

        # OPENAI_API_KEY environment variable must be set for ChatOpenAI to work
        try:
            self.llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_output_tokens
            )
            print("[INFO] ChatOpenAI model initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize ChatOpenAI: {e}")
            # The specific error from ChatOpenAI (like authentication) will be more informative now
            raise

    def build_faiss_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def query(self, question: str):
        print(f"[INFO] Received query: {question}")
        log_system_usage("[MEM/CU] Before query processing")

        # 1. Embed the question
        print(f"[INFO] Embedding question using {self.embedding_model_name}...")
        if not hasattr(self, 'openai_embed_client') or not hasattr(self, 'embedding_model_name'):
            print("[ERROR] OpenAI embedding client or model name not initialized. This should not happen.")
            # Or raise an error, as this indicates a problem in __init__
            raise RuntimeError("Embedding client/model not initialized properly.")
        question_embedding = embed_chunks([question], client=self.openai_embed_client, model_name=self.embedding_model_name)
        log_system_usage("[MEM/CU] After question embedding")

        top_k = self.config["retrieval"]["top_k"]
        _, I = self.index.search(np.array(question_embedding), top_k)

        print(f"[INFO] Retrieved top {top_k} chunks for context:")
        context = "\n".join([self.chunks[i] for i in I[0]])
        for idx in I[0]:
            print(f" - Chunk {idx}: {self.chunks[idx][:100]}...")

        prompt_template_str = self.config["prompt"]["template"]
        # Ensure the template uses 'context' and 'question' as input variables
        try:
            prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
        except Exception as e:
            print(f"[ERROR] Invalid prompt template in config.yaml: {e}")
            print("[ERROR] Ensure your template string is valid and uses {{context}} and {{question}} placeholders.")
            return "Error: Invalid prompt template configuration."

        output_parser = StrOutputParser()
        
        # Create the LangChain LCEL chain
        chain = prompt_template | self.llm | output_parser

        print("[INFO] Sending request to OpenAI model...")
        log_system_usage("[MEM/CU] Before OpenAI model inference")
        
        try:
            # The input to invoke should match the variables in the prompt_template
            response = chain.invoke({"context": context, "question": question})
        except Exception as e:
            print(f"[ERROR] Error during OpenAI API call: {e}")
            # Consider more specific error handling if needed (e.g., API key issues, rate limits)
            return "Error: Could not get a response from the language model due to an API or configuration issue."
            
        print(f"[RESPONSE] {response}")

        log_system_usage("[MEM/CU] After model inference")
        return response
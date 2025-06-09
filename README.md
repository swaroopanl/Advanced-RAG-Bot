---
title: Advanced Clinical Intelligent RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.20.0"
python_version: "3.9"
app_file: app.py
pinned: false
secrets:
  - OPENAI_API_KEY
---

# Advanced Clinical Intelligent RAG Chatbot

This project implements a Retrieval Augmented Generation (RAG) chatbot designed to answer questions based on the content of uploaded PDF documents, with a focus on clinical or technical information. It uses OpenAI's language models for both generating embeddings and answering questions, and Gradio for the user interface.

## Features

-   Upload PDF documents for information retrieval.
-   Ask questions about the content of the uploaded PDF.
-   Utilizes OpenAI's `gpt-3.5-turbo` (configurable) for question answering.
-   Uses OpenAI's `text-embedding-3-small` (configurable) for generating text embeddings.
-   Built with Python, LangChain, FAISS (for vector search), and Gradio.
-   Configuration managed via `config.yaml`.

## Project Structure

```
Advanced-Rag-chatbot/
├── app.py                # Main Gradio application
├── rag_pipeline.py       # Core RAG logic (PDF processing, embedding, QA)
├── utils.py              # Utility functions (PDF loading, chunking, embedding calls)
├── config.yaml           # Configuration for models, chunking, prompts, UI
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup and Local Execution

1.  **Clone the Repository (if applicable)**:
    ```bash
    # git clone <your-repo-url>
    # cd rag-chatbot
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set OpenAI API Key**:
    You need an OpenAI API key for this application to work. Set it as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
    Replace `"your_openai_api_key_here"` with your actual key.

5.  **Run the Application**:
    ```bash
    python app.py
    ```
    The application will typically be available at `http://127.0.0.1:7860` (Gradio will provide the exact URL).

## Configuration (`config.yaml`)

The `config.yaml` file allows you to customize various aspects of the application:

-   **`openai`**: LLM model name, temperature, max output tokens.
-   **`retrieval`**: Top-k documents to retrieve, temperature for retrieval (if applicable).
-   **`chunking`**: Chunk size, overlap, max chunks for processing PDFs.
-   **`embedding`**: Embedding provider (currently 'openai') and model name.
-   **`app`**: UI title for the Gradio interface.
-   **`prompt`**: The system prompt template used to instruct the LLM.

## Deploying to Hugging Face Spaces

This application is designed to be easily deployed as a Hugging Face Space.

1.  **Ensure your code is in a Git repository** (e.g., on GitHub or Hugging Face Hub).
    Make sure `app.py`, `rag_pipeline.py`, `utils.py`, `config.yaml`, `requirements.txt`, and this `README.md` are committed.

2.  **Create a New Space on Hugging Face**:
    -   Go to [huggingface.co/new-space](https://huggingface.co/new-space).
    -   Choose a Space name.
    -   Select "Gradio" as the Space SDK.
    -   Choose your hardware (free tier is available).
    -   Link it to your Git repository.

3.  **Add Your OpenAI API Key as a Secret**:
    -   In your Space's settings, navigate to the "Secrets" section.
    -   Click on "New secret".
    -   Enter `OPENAI_API_KEY` as the **Name** of the secret.
    -   Paste your actual OpenAI API key into the **Value** field.
    -   **Important**: The application code (`rag_pipeline.py`) is already set up to read this environment variable. **Do not hardcode your API key in any file.**

4.  **Deploy**: The Space should automatically build from your repository and deploy the Gradio application. Any pushes to your main branch will trigger a rebuild.

## Known Limitations

-   **PDF Complexity**: Very complex PDFs with many images or unusual formatting might not parse perfectly.
-   **Resource Usage**: Processing very large PDF files can be memory and time-intensive, especially on free Hugging Face Spaces tiers. This might lead to slower responses or potential crashes if resource limits are exceeded.
-   **Embedding Costs**: Using OpenAI's embedding and language models incurs costs based on your usage. Monitor your OpenAI account.

## Future Enhancements (Ideas)

-   Support for other document types (e.g., .txt, .docx).
-   Option to choose different embedding models or LLMs via the UI.
-   Caching of embeddings for previously processed PDFs.
-   More advanced error handling and user feedback.

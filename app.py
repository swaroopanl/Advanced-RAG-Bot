import gradio as gr
import yaml
from rag_pipeline import RAGPipeline

# Function to load configuration from YAML file
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config()
app_config = config.get('app', {})

# Use a global variable to avoid reloading the pipeline
pipeline = None

def process_pdf_and_question(pdf_file, question):
    global pipeline

    # Only initialize pipeline once per session or if PDF changes
    if pipeline is None or pipeline.source_path != pdf_file.name:
        pipeline = RAGPipeline(pdf_path=pdf_file.name)
        pipeline.source_path = pdf_file.name  # Save to check next time

    answer = pipeline.query(question)
    return answer

iface = gr.Interface(
    fn=process_pdf_and_question,
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Ask a question")],
    outputs="text",
    title=app_config.get('title', 'RAG Chatbot')
)

iface.launch()
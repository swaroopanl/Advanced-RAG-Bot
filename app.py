import gradio as gr
from rag_pipeline import RAGPipeline

pipeline = None

def process_pdf_and_question(pdf_file, question):
    global pipeline
    if pdf_file is not None:
        pipeline = RAGPipeline(pdf_file.name)
    if pipeline:
        return pipeline.query(question)
    return "Please upload a PDF first."

iface = gr.Interface(
    fn=process_pdf_and_question,
    inputs=[
        gr.File(label="Upload Clinical Documentation (preferably PDF Docs)", file_types=[".pdf"]),
        gr.Textbox(label="Ask your medical question")
    ],
    outputs="text",
    title="Clinical RAG Assistant",
    description="Upload clinical guidelines and ask questions based on them."
)

iface.launch()

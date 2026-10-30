"""Entry point for Hugging Face Spaces (Gradio SDK). Configure with LLM_PROVIDER / LLM_MODEL
and the matching API key as Space secrets; without them the demo shows retrieved excerpts only."""

from drug_rag.pipeline import load_pipeline
from drug_rag.ui import build_app

demo = build_app(load_pipeline())

if __name__ == "__main__":
    demo.launch()

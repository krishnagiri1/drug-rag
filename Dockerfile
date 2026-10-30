FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    DRUG_RAG_BACKEND=fastembed \
    DRUG_RAG_CHUNK_SIZE=1000 \
    DRUG_RAG_MODE=hybrid_rerank

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY drug_rag ./drug_rag

# Bake the embedding and reranker models into the image, so containers start fast
# and never need to download anything at runtime.
RUN python -c "from drug_rag.embeddings import FastEmbedder, FastReranker; FastEmbedder(); FastReranker()"

# The prebuilt 1,000-character index (run `python -m drug_rag index` before `docker build`).
COPY data/index/cs1000 ./data/index/cs1000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "drug_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]

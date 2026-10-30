# Running the API in Docker (free, local)

The image contains the code, both ONNX models, and the prebuilt 1,000-character index, so it needs no
downloads at runtime. For answers it calls the Ollama model already running on your computer.

## 1. Install Docker Desktop

Download Docker Desktop for Windows from docker.com (free for personal use) and run the installer with
the default options (it uses WSL 2). Restart if asked, then start Docker Desktop and wait until it says
"Engine running".

## 2. Build the image

From the project folder (with `data/index/cs1000` already built by `python -m drug_rag index`):

```powershell
docker build -t drug-rag .
```

The first build takes a few minutes (Python packages and the two models). The image is about 1–1.5 GB.

## 3. Run it

```powershell
docker run --rm -p 8000:8000 `
  -e LLM_PROVIDER=ollama -e LLM_MODEL=qwen2.5:14b `
  -e LLM_BASE_URL=http://host.docker.internal:11434 `
  drug-rag
```

`host.docker.internal` lets the container reach Ollama on your computer. Wait for
`Application startup complete` (building the keyword index takes up to a minute).

## 4. Try it

Open http://localhost:8000/docs for the interactive API page, or from a second PowerShell window:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType "application/json" `
  -Body '{"question": "Can duloxetine be taken with MAO inhibitors?"}'
```

Stop the container with `Ctrl+C`.

## Optional: share the image

Create a free Docker Hub account, then:

```powershell
docker login
docker tag drug-rag <your-dockerhub-username>/drug-rag:latest
docker push <your-dockerhub-username>/drug-rag:latest
```

The same image runs on any cloud VM with Docker (for example AWS EC2 or Google Cloud). On a server without a GPU,
point it at a hosted LLM API instead of Ollama with `-e LLM_PROVIDER=openai -e LLM_MODEL=<model> -e OPENAI_API_KEY=...`.

# DOXIE: Local Document RAG and Resume Analyzer

DOXIE is a local-only web application for document analysis and resume evaluation using Retrieval-Augmented Generation (RAG) with Ollama (local LLM) and ChromaDB. It allows you to upload PDFs/text, index them, summarize content, and chat with AI grounded in your documents—all without sending data to the cloud.

# Features
- **Document Upload & Extraction**: Upload PDFs or paste text; extract and refine content.
- **RAG Chat**: Ask questions based on indexed documents with source references.
- **Summarization**: Get concise summaries of documents.
- **Resume Analyzer**: Analyze resumes against job roles using local AI.
- **Session Management**: In-memory sessions for multi-user demo (single-machine).

# Prerequisites
- **Python 3.8+**: Download from [python.org](https://www.python.org/).
- **Ollama**: Install from [ollama.com](https://ollama.com). Required for local AI inference.
  - Pull a model: `ollama pull llama3.2` (or your preferred model; defaults to `llama3.2`).
  - Ensure Ollama is running locally (default: `http://127.0.0.1:11434`).

# Installation
1. **Clone the Repository**:
   
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   

2. **Install Dependencies**:
   
   pip install -r requirements.txt
   

## Usage
1. **Start Ollama** (in a separate terminal):
   
   ollama serve
   

2. **Run the App**:
   
   python r-ai.py
   
   - Opens at `http://127.0.0.1:8000` (customize with `ATLAS_HOST` and `ATLAS_PORT` env vars).

3. **Use the Web Interface**:
   - **Home Page**: Upload/index documents, chat with RAG.
   - **Resume Page**: Paste resume text and specify a role for analysis.
   - Sessions are managed via cookies; reload for a new session.

## Project Structure
- main.py: FastAPI app with routes.
- r-ai.py: Launcher script.
- rag_engine.py: RAG logic with ChromaDB and embeddings.
- local_llm.py: Ollama integration for AI tasks.
- `templates/`: HTML templates (Jinja2).
- `static/`: CSS/JS assets.
- `chroma_data/`: Local ChromaDB storage (auto-generated, ignored in Git).

## Troubleshooting
- **Ollama Errors**: Ensure Ollama is installed, running, and the model is pulled.
- **Port Issues**: Change `ATLAS_PORT` if 8000 is in use.
- **Dependencies**: Use a virtual environment (`python -m venv venv; venv\Scripts\activate` on Windows).
- **Data Reset**: Use the "Refurbish" button in the app to clear indexed data.


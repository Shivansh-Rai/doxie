"""
Launch the DOXIE web app (FastAPI + local Ollama).

Usage:
    pip install -r requirements.txt
    python r-ai.py

Then open http://127.0.0.1:8000

Requires Ollama running locally: https://ollama.com
Example: ollama pull llama3.2

"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.getenv("ATLAS_PORT", "8000"))
    host = os.getenv("ATLAS_HOST", "127.0.0.1")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()

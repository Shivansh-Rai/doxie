"""
RAG: chunk → local embeddings → ChromaDB → retrieve → Ollama (local only).
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from pypdf import PdfReader

from local_llm import LocalLLMError, ollama_chat

load_dotenv()

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_data"
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")

SYSTEM_PROMPT = """You are a precise assistant. Answer ONLY using the CONTEXT excerpts below.
If the answer is not contained or clearly implied in the context, say that the document does not contain enough information — do not invent facts or use outside knowledge.
Be concise. Reference [Source: …] when useful."""


def collection_name_for_session(session_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id.strip())[:72]
    return f"u_{safe}"


def _client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _embedding_fn():
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def _get_collection(session_id: str):
    client = _client()
    name = collection_name_for_session(session_id)
    return client.get_or_create_collection(
        name=name,
        embedding_function=_embedding_fn(),
        metadata={"session": session_id},
    )


def delete_session_collection(session_id: str) -> None:
    client = _client()
    name = collection_name_for_session(session_id)
    try:
        client.delete_collection(name)
    except Exception:
        pass


def reset_all_index_data() -> None:
    """Remove all persisted Chroma data across sessions."""
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 140) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _extract_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except Exception:
        return ""
    parts: list[str] = []
    try:
        for page in reader.pages:
            try:
                t = page.extract_text()
            except Exception:
                continue
            if t:
                parts.append(t)
    except Exception:
        return "\n".join(parts) if parts else ""
    return "\n".join(parts)


def _extract_txt(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_text(filename: str, data: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf(data)
    return _extract_txt(data)


def ingest_text(session_id: str, filename: str, text: str) -> dict[str, Any]:
    raw = text.strip()
    parts = chunk_text(raw)
    if not parts:
        return {"chunks": 0, "filename": filename, "warn": "No extractable text."}

    delete_session_collection(session_id)
    coll = _get_collection(session_id)
    base = hashlib.sha256(raw.encode()).hexdigest()[:16]
    ids = [f"{base}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(parts))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(parts))]
    coll.add(ids=ids, documents=parts, metadatas=metadatas)
    return {"chunks": len(parts), "filename": filename, "warn": None}


def collection_count(session_id: str) -> int:
    try:
        return _get_collection(session_id).count()
    except Exception:
        return 0


def retrieve(session_id: str, question: str, k: int = 6) -> tuple[list[str], list[str]]:
    coll = _get_collection(session_id)
    res = coll.query(query_texts=[question], n_results=k)
    docs = (res.get("documents") or [[]])[0] or []
    metas = (res.get("metadatas") or [[]])[0] or []
    sources = []
    for m in metas:
        if isinstance(m, dict) and m.get("source"):
            sources.append(str(m["source"]))
        else:
            sources.append("document")
    return docs, sources


def _build_user_message(question: str, contexts: list[str], sources: list[str]) -> str:
    blocks = []
    for src, ctx in zip(sources, contexts):
        blocks.append(f"[Source: {src}]\n{ctx}")
    context_block = "\n\n---\n\n".join(blocks)
    return f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"


def generate_local_answer(question: str, contexts: list[str], sources: list[str]) -> str:
    user_msg = _build_user_message(question, contexts, sources)
    return ollama_chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )


def _extractive_fallback(contexts: list[str], sources: list[str]) -> str:
    lines = [
        "**Ollama unavailable.** Showing retrieved excerpts from your document:",
        "",
    ]
    for i, (src, ctx) in enumerate(zip(sources, contexts), start=1):
        preview = ctx[:1500] + ("…" if len(ctx) > 1500 else "")
        lines.append(f"{i}. **{src}** — {preview}\n")
    return "\n".join(lines)


def rag_answer(session_id: str, question: str, k: int = 6) -> dict[str, Any]:
    q = question.strip()
    if not q:
        return {"answer": "", "sources": [], "backend": "", "error": "Empty question."}

    if collection_count(session_id) == 0:
        return {
            "answer": "Index your document first: edit the text if needed, then click **Sync knowledge base**.",
            "sources": [],
            "backend": "",
            "error": None,
        }

    contexts, sources = retrieve(session_id, q, k=k)
    if not contexts:
        return {
            "answer": "No strongly matching passages found. Try rephrasing.",
            "sources": [],
            "backend": "local",
            "error": None,
        }

    try:
        answer = generate_local_answer(q, contexts, sources)
        return {"answer": answer, "sources": sources, "backend": "Ollama", "error": None}
    except LocalLLMError as e:
        return {
            "answer": _extractive_fallback(contexts, sources),
            "sources": sources,
            "backend": f"fallback ({e})",
            "error": None,
        }

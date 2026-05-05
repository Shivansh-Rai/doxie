"""
DOXIE — local-only document RAG + resume analyzer.

Run:
  pip install -r requirements.txt
  python r-ai.py

Open http://127.0.0.1:8000 (override with ATLAS_HOST / ATLAS_PORT env vars).

Optional: uvicorn main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.templating import Jinja2Templates

from local_llm import LocalLLMError, analyze_resume, summarize_document
from rag_engine import collection_count, extract_text, ingest_text, rag_answer, reset_all_index_data

ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))

app = FastAPI(title="DOXIE Local AI")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

# In-memory session state (single-machine demo)
CHATS: dict[str, list[dict[str, str]]] = {}
DOC_META: dict[str, dict] = {}


def _valid_sid(s: str | None) -> bool:
    if not s:
        return False
    try:
        uuid.UUID(s)
        return True
    except ValueError:
        return False


def session_from_request(request: Request) -> str:
    sid = request.cookies.get("atlas_sid")
    if _valid_sid(sid):
        return sid  # type: ignore[return-value]
    raise HTTPException(status_code=400, detail="Missing session. Reload the page.")


def ensure_session_cookie(response: HTMLResponse, request: Request) -> str:
    sid = request.cookies.get("atlas_sid")
    if not _valid_sid(sid):
        sid = str(uuid.uuid4())
        response.set_cookie(
            key="atlas_sid",
            value=sid,
            max_age=90 * 24 * 3600,
            httponly=True,
            samesite="lax",
            path="/",
        )
    CHATS.setdefault(sid, [])
    DOC_META.setdefault(sid, {"filename": "", "chunks": 0})
    return sid


@app.get("/", response_class=HTMLResponse)
async def page_home(request: Request):
    response = templates.TemplateResponse(
        request,
        "index.html",
        {"title": "DOXIE Doc Analyzer"},
    )
    ensure_session_cookie(response, request)
    return response


@app.get("/resume", response_class=HTMLResponse)
async def page_resume(request: Request):
    response = templates.TemplateResponse(
        request,
        "resume.html",
        {"title": "Resume analyzer"},
    )
    ensure_session_cookie(response, request)
    return response


class IndexBody(BaseModel):
    text: str
    filename: str = Field(default="document.txt")


class ChatBody(BaseModel):
    message: str


class SummarizeBody(BaseModel):
    text: str


class ResumeBody(BaseModel):
    resume_text: str
    role: str


@app.get("/api/session")
async def api_session(request: Request):
    sid = session_from_request(request)
    return {
        "chat": CHATS.get(sid, []),
        "doc": DOC_META.get(sid, {}),
        "indexed_chunks": collection_count(sid),
    }


@app.post("/api/upload")
async def api_upload(request: Request, file: UploadFile = File(...)):
    sid = session_from_request(request)
    data = await file.read()
    name = file.filename or "upload"
    text = extract_text(name, data)
    DOC_META[sid] = {"filename": name, "chunks": collection_count(sid)}
    return {"filename": name, "text": text, "warning": None if text.strip() else "No text extracted."}


@app.post("/api/index")
async def api_index(request: Request, body: IndexBody):
    sid = session_from_request(request)
    fn = body.filename.strip() or "document.txt"
    info = ingest_text(sid, fn, body.text)
    if info.get("warn"):
        DOC_META[sid] = {"filename": fn, "chunks": 0}
        raise HTTPException(status_code=400, detail=info["warn"])
    DOC_META[sid] = {"filename": fn, "chunks": info["chunks"]}
    return {"ok": True, "chunks": info["chunks"], "filename": fn}


@app.post("/api/summarize")
async def api_summarize(request: Request, body: SummarizeBody):
    session_from_request(request)
    try:
        summary = summarize_document(body.text)
    except LocalLLMError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"summary": summary}


@app.post("/api/chat")
async def api_chat(request: Request, body: ChatBody):
    sid = session_from_request(request)
    msg = body.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message.")
    CHATS.setdefault(sid, []).append({"role": "user", "content": msg})
    out = rag_answer(sid, msg, k=6)
    answer = out.get("answer") or ""
    if out.get("error"):
        answer = out["error"]
    meta = ", ".join(dict.fromkeys(out.get("sources") or []))
    CHATS[sid].append({"role": "assistant", "content": answer})
    return {"reply": answer, "sources": out.get("sources") or [], "meta": meta}


@app.post("/api/chat/clear")
async def api_chat_clear(request: Request):
    sid = session_from_request(request)
    CHATS[sid] = []
    return {"ok": True}


@app.post("/api/system/refurbish")
async def api_system_refurbish(request: Request):
    # Keep the current browser session but clear all local app state and indexed data.
    sid = session_from_request(request)
    reset_all_index_data()
    CHATS.clear()
    DOC_META.clear()
    CHATS[sid] = []
    DOC_META[sid] = {"filename": "", "chunks": 0}
    return {"ok": True}


@app.post("/api/resume/analyze")
async def api_resume_analyze(request: Request, body: ResumeBody):
    session_from_request(request)
    try:
        result = analyze_resume(body.resume_text, body.role)
    except LocalLLMError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"role": body.role.strip(), **result}
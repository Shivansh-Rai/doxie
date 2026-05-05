"""Local inference via Ollama only — no cloud APIs."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


class LocalLLMError(RuntimeError):
    pass


def ollama_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    body = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        r = requests.post(url, json=body, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        raise LocalLLMError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Install from https://ollama.com "
            f"and run: ollama pull {OLLAMA_MODEL}"
        ) from e
    try:
        data = r.json()
    except ValueError as e:
        raise LocalLLMError(f"Invalid response from Ollama ({r.status_code}).") from e
    err = data.get("error")
    if err:
        raise LocalLLMError(str(err))
    msg = data.get("message") or {}
    text = (msg.get("content") or "").strip()
    if not text:
        raise LocalLLMError("Ollama returned an empty response.")
    return text


def summarize_document(text: str, max_chars: int = 14000) -> str:
    slim = text.strip()
    if not slim:
        raise LocalLLMError("No text to summarize.")
    if len(slim) > max_chars:
        slim = slim[:max_chars] + "\n\n[…truncated for local model context…]"
    messages = [
        {
            "role": "system",
            "content": (
                "You summarize documents clearly using headings, bullets, and tight prose. "
                "Use ONLY the provided text—do not invent facts."
            ),
        },
        {"role": "user", "content": f"Summarize the following document:\n\n{slim}"},
    ]
    return ollama_chat(messages, temperature=0.15)


def analyze_resume(resume_text: str, role: str, max_chars: int = 12000) -> dict[str, Any]:
    role = role.strip()
    if not role:
        raise LocalLLMError("Please enter the target role.")
    slim = resume_text.strip()
    if not slim:
        raise LocalLLMError("No resume text to analyze.")
    if len(slim) > max_chars:
        slim = slim[:max_chars] + "\n\n[…truncated…]"

    schema_hint = """Respond with ONLY valid JSON (no markdown code fences), exactly:
{"key_skills": "<concise string>", "project_details": "<concise string>", "suitability_score": <integer 0-10>, "suitability_rationale": "<short string>"}

suitability_score: how well the candidate fits the stated role (0-10)."""

    messages = [
        {
            "role": "system",
            "content": "You evaluate resumes against a target role. Output strict JSON only.",
        },
        {
            "role": "user",
            "content": f"Target role: {role}\n\nResume:\n---\n{slim}\n---\n\n{schema_hint}",
        },
    ]
    raw = ollama_chat(messages, temperature=0.1)
    return _parse_resume_json(raw)


def _parse_resume_json(raw: str) -> dict[str, Any]:
    s = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if fence:
        s = fence.group(1).strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return {
            "key_skills": s[:2000],
            "project_details": "",
            "suitability_score": None,
            "suitability_rationale": "Model returned non-JSON; showing raw text above.",
            "raw": raw,
        }
    score = data.get("suitability_score")
    if score is not None:
        try:
            score = max(0, min(10, int(score)))
        except (TypeError, ValueError):
            score = None
    return {
        "key_skills": str(data.get("key_skills") or ""),
        "project_details": str(data.get("project_details") or ""),
        "suitability_score": score,
        "suitability_rationale": str(data.get("suitability_rationale") or ""),
    }

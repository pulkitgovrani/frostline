"""Local LLM via Ollama. Optional: the plan works without it."""
from __future__ import annotations

import json
import os
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("FROSTLINE_MODEL", "gemma4:e4b")

SYSTEM = (
    "You are a friendly, practical allotment gardener writing a short weekend 'field note'. "
    "You will get a JSON plan computed by code. Use ONLY facts from it: never invent dates, crops or numbers. "
    "Write 90-140 words, warm and concrete, in plain text (no markdown headers). "
    "Order the jobs as a short outdoor trip, mention any frost warning first, and end with one line "
    "encouraging the reader to put the phone down and go outside."
)


def field_note(plan: dict, timeout: int = 180) -> str | None:
    body = json.dumps({
        "model": MODEL, "stream": False, "think": False,
        "options": {"temperature": 0.4, "num_predict": 400},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(plan)}],
    }).encode()
    req = urllib.request.Request(f"{HOST}/api/chat", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (localhost)
            return json.load(r)["message"]["content"].strip() or None
    except Exception:
        return None


def fallback_note(plan: dict) -> str:
    jobs = "; ".join(f"{t['action']} {t['crop']}" for t in plan["tasks"][:5]) or "nothing urgent, so just walk the beds and enjoy it"
    return f"{plan['frost_watch']} This week outside: {jobs}. Put the phone down and go get some soil under your nails."

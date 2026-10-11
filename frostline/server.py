from __future__ import annotations

import json
import os
import sys
import urllib.parse
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .core import build_plan

WEB = Path(__file__).resolve().parent.parent / "web"


class H(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        if u.path == "/":
            return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/plan":
            q = urllib.parse.parse_qs(u.query)
            try:
                place = q["place"][0]
                d = date.fromisoformat(q["date"][0]) if "date" in q else None
                plan = build_plan(place, d, use_llm=q.get("llm", ["1"])[0] != "0")
                return self._send(200, json.dumps(plan).encode(), "application/json")
            except Exception as e:  # shown to the user in the UI
                return self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
        self._send(404, b"not found", "text/plain")


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"FrostLine on http://{host}:{port}")
    ThreadingHTTPServer((host, port), H).serve_forever()

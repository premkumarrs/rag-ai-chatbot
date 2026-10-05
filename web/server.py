"""Serve the chat page on its own local port and keep the models loaded."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    LLM_KEEP_ALIVE,
    LLM_NUM_CTX,
    LLM_WARMUP,
    OLLAMA_BASE_URL,
    OLLAMA_HOST,
)

HOST = "127.0.0.1"
PORT = 5173
WEB_DIR = Path(__file__).resolve().parent


def _post(path: str, payload: dict, host: str) -> None:
    request = urllib.request.Request(
        host.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        response.read()


def warm_models() -> None:
    """Load chat and embedding models so the first question is not a cold start."""
    if not LLM_WARMUP:
        return
    try:
        _post(
            "/api/embed",
            {"model": EMBEDDING_MODEL, "input": "warmup", "keep_alive": LLM_KEEP_ALIVE},
            OLLAMA_HOST,
        )
        _post(
            "/api/generate",
            {
                "model": CHAT_MODEL,
                "prompt": "Reply with OK.",
                "stream": False,
                "think": False,
                "keep_alive": LLM_KEEP_ALIVE,
                "options": {
                    "num_predict": 8,
                    "num_ctx": LLM_NUM_CTX,
                    "temperature": 0,
                },
            },
            OLLAMA_BASE_URL,
        )
        print(f"Models are loaded and will stay in memory for {LLM_KEEP_ALIVE}.")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Model warmup skipped: {exc}")


class ChatHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)


def main() -> None:
    threading.Thread(target=warm_models, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), ChatHandler)
    print(f"Chat page: http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()

"""Serve the chat page on its own local port and keep the models loaded."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = 5173
OLLAMA = "http://127.0.0.1:11434"
WEB_DIR = Path(__file__).resolve().parent


def _post(path: str, payload: dict) -> None:
    request = urllib.request.Request(
        OLLAMA + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        response.read()


def warm_models() -> None:
    """Load chat and embedding models so the first question is not a cold start."""
    try:
        _post(
            "/api/embed",
            {"model": "nomic-embed-text", "input": "warmup", "keep_alive": "30m"},
        )
        _post(
            "/api/generate",
            {
                "model": "qwen3.5:4b",
                "prompt": "Reply with OK.",
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {"num_predict": 8, "num_ctx": 4096, "temperature": 0},
            },
        )
        print("Models are loaded and will stay in memory for 30 minutes.")
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

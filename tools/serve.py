"""Startet die Website lokal (wie auf Cloudflare Pages, inkl. Sicherheits-Header).

    python3 tools/serve.py            ->  http://127.0.0.1:8000
"""
from __future__ import annotations

import fnmatch
import http.server
import socketserver
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


def load_rules() -> list[tuple[str, list[tuple[str, str]]]]:
    rules, cur = [], None
    for line in (WEB / "_headers").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            cur = (line.strip(), [])
            rules.append(cur)
        elif cur is not None and ":" in line:
            k, v = line.strip().split(":", 1)
            cur[1].append((k.strip(), v.strip()))
    return rules


RULES = load_rules()


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".mjs": "text/javascript", ".js": "text/javascript"}

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def end_headers(self):
        path = self.path.split("?")[0]
        for pattern, headers in RULES:
            if fnmatch.fnmatch(path, pattern):
                for k, v in headers:
                    self.send_header(k, v)
        super().end_headers()

    def log_message(self, *a):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as srv:
        srv.daemon_threads = True
        print(f"PlanDigitalizer läuft auf http://127.0.0.1:{port}  (Beenden mit Ctrl+C)")
        srv.serve_forever()


if __name__ == "__main__":
    main()

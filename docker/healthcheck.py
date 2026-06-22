"""Minimal HTTP server that satisfies Fly.io's healthcheck on port 8080.

The AccuBet worker has no public web UI — this just keeps the machine alive
and returns a JSON status blob so the platform knows the container is healthy.
Run CLI commands via: fly ssh console -a accubet
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"status": "ok", "app": "accubet"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass  # suppress access log noise


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"AccuBet health server listening on port {port}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

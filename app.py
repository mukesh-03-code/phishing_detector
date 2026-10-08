#!/usr/bin/env python3
"""
Standalone Web & Chrome Extension API Server for Project 3: PhishGuard AI
Run: python3 app.py (listens on port 8000 for Chrome Extension + Web UI)
"""

import http.server
import json
import os
import socketserver
from ml_model import PhishingMLDetector

detector = PhishingMLDetector()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>PhishGuard AI — URL & Email Detector</title>
<style>
  body { font-family: system-ui, sans-serif; background: #090d16; color: #f1f5f9; margin: 0; padding: 24px; }
  .card { background: #111827; border: 1px solid #26334d; border-radius: 12px; padding: 18px; margin-bottom: 18px; max-width: 900px; margin-left: auto; margin-right: auto; }
  input, textarea { width: 100%; background: #050811; border: 1px solid #334155; color: #fff; padding: 10px; border-radius: 8px; margin-bottom: 10px; box-sizing: border-box; }
  button { background: #0284c7; color: #fff; border: none; padding: 10px 16px; border-radius: 8px; font-weight: 600; cursor: pointer; }
  pre { background: #050811; padding: 14px; border-radius: 8px; overflow-x: auto; color: #38bdf8; }
</style></head>
<body>
  <div class="card">
    <h2>PhishGuard AI — URL & Email Phishing Detector</h2>
    <input id="url" value="http://paypa1-account-security-verify.xyz/login?session=8821" />
    <button onclick="scanUrl()">Analyze URL</button>
    <pre id="out"></pre>
  </div>
<script>
async function scanUrl() {
  const url = document.getElementById('url').value;
  const r = await fetch('/api/phishing/url', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({url})});
  document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
}
scanUrl();
</script>
</body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode() or "{}")
        if self.path == "/api/phishing/email":
            res = detector.predict_email(body.get("subject", ""), body.get("sender", ""), body.get("body", ""))
        else:
            res = detector.predict_url(body.get("url", "https://www.google.com"))
        raw = json.dumps(res).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[+] PhishGuard AI Server running on http://0.0.0.0:{port}")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
        httpd.serve_forever()

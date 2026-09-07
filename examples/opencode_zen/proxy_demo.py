#!/usr/bin/env python3
"""Transport proxy execution path demonstration: HTTP requests are routed through the proxy.

Spawns two local server threads: one HTTP proxy, one upstream LLM mock endpoint.
Then creates an HttpLLMAgent(proxy=...) that calls upstream.
Verifies:
  1. Proxy observes the absolute-form request line
  2. Upstream receives the LLM payload
  3. Response is successfully routed back through the proxy

Run: python examples/opencode_zen/proxy_demo.py
"""

import asyncio
import http.client
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from framework import HttpLLMAgent


upstream_hits: list[str] = []
proxy_hits: list[str] = []


class Upstream(BaseHTTPRequestHandler):
    """Simulates an upstream LLM endpoint returning OpenAI-compatible responses."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        upstream_hits.append(self.rfile.read(length).decode())
        body = json.dumps({
            "choices": [{"message": {"content": "[Via Proxy] Hello, response received through proxy!"}}]
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, fmt, *args):
        pass


class Proxy(BaseHTTPRequestHandler):
    """Minimal HTTP forward proxy: forwards absolute-form requests upstream."""

    def do_POST(self):
        proxy_hits.append(self.path)  # Request line seen by proxy (absolute URL)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        parsed = urlparse(self.path)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80)
        conn.request(self.command, parsed.path or "/", body=body, headers=dict(self.headers))
        resp = conn.getresponse()
        data = resp.read()
        self.send_response(resp.status)
        for key, val in resp.getheaders():
            self.send_header(key, val)
        self.end_headers()
        self.wfile.write(data)
        conn.close()

    def log_message(self, fmt, *args):
        pass


async def main():
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    threading.Thread(target=proxy.serve_forever, daemon=True).start()

    try:
        agent = HttpLLMAgent(
            base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1/chat/completions",
            proxy=f"http://127.0.0.1:{proxy.server_address[1]}",
        )
        print(f"LLM Endpoint: {agent.base_url}")
        print(f"Proxy Address: {agent.proxy}")

        result = await agent.process("Hello", "Short reply", "demo-model")
        print("\nResult:", result)

        print("\nProxy received (absolute-form request line):", proxy_hits[0])
        payload = json.loads(upstream_hits[0])
        print("Upstream received model:", payload["model"],
              "| user message:", payload["messages"][-1]["content"])

        print("\nConclusion: Requests are forwarded through proxy and returned through proxy.")
        await agent.close()
    finally:
        proxy.shutdown()
        upstream.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

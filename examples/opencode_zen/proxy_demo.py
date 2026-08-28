#!/usr/bin/env python3
"""传输代理真实路径演示：HTTP 请求确实“经代理出去”。

起两个本地进程：一个做 HTTP 代理，一个做“上游 LLM 端点”(OpenAI 形状)。
然后在本地起 HttpLLMAgent(proxy=...)，让它去调用上游。如果一切正常：

  1. 代理侧能看到 agent 发来的绝对形式请求行 (absolute-form URL)
  2. 上游真的收到了 LLM payload
  3. 返回结果经代理转回来

这就直观证明了 config 里的 "proxy": "http://..." 是传输层代理，
HTTP 请求是真正经它出去的——不是仅仅存了个属性。

运行:  cd 项目根目录 && python examples/opencode_zen/proxy_demo.py
"""

import asyncio
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from framework import HttpLLMAgent


upstream_hits: list[str] = []
proxy_hits: list[str] = []


class Upstream(BaseHTTPRequestHandler):
    """扮演真实 LLM 端点（本地版），返回 OpenAI 形状的响应。"""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        upstream_hits.append(self.rfile.read(length).decode())
        body = json.dumps({
            "choices": [{"message": {"content": "【经过代理返回】你好，我是代理转发回来的回复！"}}]
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, fmt, *args):
        pass


class Proxy(BaseHTTPRequestHandler):
    """最小 HTTP 转发代理：绝对形式请求 → 转发到上游。"""

    def do_POST(self):
        proxy_hits.append(self.path)  # 代理看到的请求行（绝对 URL）
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
            base_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",  # 真实 LLM 端点
            proxy=f"http://127.0.0.1:{proxy.server_address[1]}",          # HTTP 请求经它出去
        )
        print(f"LLM 端点: {agent.base_url}")
        print(f"代理地址: {agent.proxy}")

        result = await agent.process("你好", "简短回复", "demo-model")
        print("\n返回结果:", result)

        print("\n代理侧实际收到(绝对形式请求行):", proxy_hits[0])
        payload = json.loads(upstream_hits[0])
        print("上游实际收到 模型名:", payload["model"],
              "| 用户消息:", payload["messages"][-1]["content"])

        print("\n结论: 请求确实经代理转发到上游，再经代理转回 —— HTTP 请求通过代理。")
        await agent.close()
    finally:
        proxy.shutdown()
        upstream.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
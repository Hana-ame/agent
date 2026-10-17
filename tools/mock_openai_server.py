#!/usr/bin/env python
"""本地假 OpenAI 端点：只为验证一次 shot 的**完整 HTTP 链路**是否通。

它不会"生成"任何东西——收到请求就把标准答案（默认 SPEC 的渲染结果）当作
模型回复吐回去。所以它证明的是：DAG → 边的 prompt → HTTP → 解析 → assemble →
自校验这条链没断；**不证明**模型能力。

真正的判定必须跑 `tools/oneshot_chain.py`（不接 --offline），用 laptop-6.8
或 sensenova 的真端点。本脚本只是把"除了模型以外的一切"先验掉。

    python tools/mock_openai_server.py --port 8000
    # 另开一个终端：
    python tools/oneshot_chain.py --generations 3 --model-key laptop-6.8
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def answer_for(prompt: str) -> str:
    """按 prompt 意图给答案：要 JSON 给 JSON，要代码给对应文件的原码。"""
    from vea_v6.parts.bootstrap import (
        FRAMEWORK_SPEC,
        FRAME_FILES,
        framework_sources,
        render_framework_code,
        signature,
        spec_to_text,
    )

    head = prompt[:400]
    src = framework_sources()

    # 单文件重写：按 "文件：<path>" 那行定位（必须先于任何"重写/源码"泛匹配）
    for line in prompt.splitlines():
        if line.startswith("文件："):
            path = line.split("：", 1)[1].strip()
            if path in src:
                return src[path]
    if "签名" in head:
        return json.dumps(
            {"files": [{"path": p, "sig": signature(p, src[p])} for p in FRAME_FILES]},
            ensure_ascii=False,
        )
    if "SPEC" in head or "可执行规格" in head:
        return spec_to_text(FRAMEWORK_SPEC)
    return render_framework_code()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # 静音默认访问日志
        pass

    def do_POST(self) -> None:  # noqa: N802 - http.server 的约定
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n).decode("utf-8") if n else "{}"
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            req = {}
        msgs = req.get("messages") or []
        prompt = ""
        for m in msgs:
            c = m.get("content")
            if isinstance(c, str):
                prompt += c + "\n"
            elif isinstance(c, list):
                prompt += "".join(p.get("text", "") for p in c if isinstance(p, dict)) + "\n"

        text = answer_for(prompt)
        want_stream = bool(req.get("stream"))

        if want_stream:
            # 客户端默认 stream=True，按 SSE 逐块吐；否则它解析不到内容会判"空响应"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            chunk = 2048
            for i in range(0, len(text), chunk):
                ev = {
                    "id": "mock",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": text[i : i + chunk]}, "finish_reason": None}],
                }
                self.wfile.write(b"data: " + json.dumps(ev, ensure_ascii=False).encode("utf-8") + b"\n\n")
            done = {
                "id": "mock",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(b"data: " + json.dumps(done).encode("utf-8") + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        body = {
            "id": "mock",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(text)},
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        data = json.dumps({"object": "list", "data": [{"id": "mock", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock OpenAI 端点在 http://127.0.0.1:{args.port}/v1  (Ctrl-C 结束)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())

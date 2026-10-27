"""用本地 mock HTTP server 验证 GraphNode 的 429 重试链路。"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import dag8  # noqa: E402
ROOT = V8.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from v7.llm import OpenAICompatibleClient  # noqa: E402


class _Flaky(BaseHTTPRequestHandler):
    attempts = 0
    fail_first = 2

    def do_POST(self):  # noqa: N802
        n = type(self).attempts
        type(self).attempts += 1
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)

        if n < type(self).fail_first:
            body = json.dumps({
                "error": {"message": "rate_limit exceeded", "type": "rate_limit_error"}
            }).encode()
            self.send_response(429)
        else:
            body = json.dumps({
                "id": "mock",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "done"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)

        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: ANN002
        pass


def _serve():
    _Flaky.attempts = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Flaky)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    return server, thread, base


def test_graph_retries_429_over_http(tmp_path, monkeypatch):
    import dag8.nodes as nodes_mod

    monkeypatch.setattr(nodes_mod, "_RETRY_SLEEPS", (0.0, 0.0))
    server, thread, base = _serve()
    try:
        v7 = {
            "name": "mock_v7",
            "models": {"m": {"api": base, "key": "not-needed", "model": "mock"}},
            "contents": [{"tags": ["seed"], "contents": "S"}],
            "edges": [{"id": "e", "input_tags": ["seed"], "output_tags": ["out"],
                       "processor": {"model": "m", "prompt": "P {contents}"}}],
        }
        (tmp_path / "v7.json").write_text(json.dumps(v7), encoding="utf-8")
        parent = {
            "name": "mock_parent", "seed_tag": "seed", "seed_content": "S",
            "result_tag": "out",
            "edges": [{"id": "g", "kind": "graph",
                       "input_tags": ["seed"], "output_tags": ["out"],
                       "graph": {"path": "v7.json", "seed_tag": "seed",
                                 "result_tag": "out", "timeout": 30}}],
        }
        cfg = tmp_path / "dag.json"
        cfg.write_text(json.dumps(parent), encoding="utf-8")

        res = dag8.run(
            cfg, db_path=tmp_path / "r.sqlite3", max_seconds=30,
            graph_llm_client=OpenAICompatibleClient(timeout=30, max_retries=3),
        )
        assert res.status == "ok", res.summary()
        assert res.result == "done"
        assert _Flaky.attempts >= 3
    finally:
        server.shutdown()
        thread.join(2)

"""自适应并发的 HTTP 集成测试（mock server 控制 429）。"""
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


class _Limiter(BaseHTTPRequestHandler):
    """prompt 里含 FAIL 的请求一律 429，其余 200 ok。"""

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8", "replace")
        if "FAIL" in raw:
            body = json.dumps({
                "error": {"message": "429 rate_limit", "type": "rate_limit_error"}
            }).encode()
            self.send_response(429)
        else:
            body = json.dumps({
                "id": "mock",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
            }).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: ANN002
        pass


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_adaptive_reduces_workers_on_429(tmp_path, monkeypatch):
    import dag8.nodes as nodes_mod

    monkeypatch.setattr(nodes_mod, "_RETRY_SLEEPS", (0.0, 0.0))

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Limiter)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        for name, prompt in (("a", "OK"), ("b", "FAIL"), ("c", "FAIL"),
                             ("d", "OK"), ("e", "OK")):
            out_tag = "out" if name == "e" else name
            _write(tmp_path / "graphs" / f"{name}.json", {
                "name": f"sub_{name}",
                "models": {"m": {"api": base, "key": "k", "model": "mock"}},
                "seed_tag": "task", "contents": [],
                "edges": [{"id": name, "input_tags": ["task"],
                           "output_tags": [out_tag],
                           "processor": {"model": "m",
                                         "prompt": f"{prompt} {{contents}}"}}],
            })

        edges = [
            {"id": f"g{n}", "kind": "graph", "layer": 0,
             "input_tags": ["task"], "output_tags": [n],
             "graph": {"path": f"graphs/{n}.json", "seed_tag": "task",
                       "result_tag": n, "timeout": 30}}
            for n in ("a", "b", "c", "d")
        ]
        edges.append({
            "id": "e", "kind": "graph", "layer": 1,
            "input_tags": ["a"], "output_tags": ["out"],
            "graph": {"path": "graphs/e.json", "seed_tag": "task",
                      "result_tag": "out", "timeout": 30},
        })
        _write(tmp_path / "dag.json", {
            "name": "adaptive_http", "seed_tag": "task", "seed_content": "x",
            "result_tag": "out", "edges": edges,
        })

        res = dag8.run(tmp_path / "dag.json", db_path=tmp_path / "r.sqlite3",
                       max_workers=4, keep_going=True, adaptive=True,
                       graph_llm_client=OpenAICompatibleClient(timeout=30, max_retries=1))
        assert res.workers_used == [4, 2], (res.workers_used, res.summary())
        assert res.status == "partial", [(e.edge_id, e.status, e.error) for e in res.edges]
        assert res.result == "ok"
    finally:
        server.shutdown()
        thread.join(2)

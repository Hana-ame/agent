"""http 节点的本地集成测试。"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import dag8  # noqa: E402
from dag8 import config as cfg_mod  # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/ok":
            self._send(200, "hello")
        elif self.path == "/err":
            self._send(500, "boom")
        else:
            self._send(404, "nope")

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8", "replace")
        self._send(200, "echo:" + body)

    def _send(self, code: int, text: str) -> None:
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: ANN002
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    t.join(2)


def _write(tmp_path: Path, url: str, **extra) -> Path:
    edge = {"id": "h", "kind": "http", "input_tags": ["task"],
            "output_tags": ["out"], "url": url, **extra}
    data = {"name": "http_demo", "seed_tag": "task", "seed_content": "S",
            "result_tag": "out", "edges": [edge]}
    p = tmp_path / "dag.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_http_get_ok(tmp_path, server):
    res = dag8.run(_write(tmp_path, server + "/ok"), db_path=tmp_path / "r.sqlite3")
    assert res.status == "ok"
    assert res.result == "hello"


def test_http_post_renders_body(tmp_path, server):
    res = dag8.run(_write(tmp_path, server + "/echo", method="POST",
                          body="payload:{tag:task}"),
                   db_path=tmp_path / "r.sqlite3")
    assert res.status == "ok"
    assert res.result == "echo:payload:S"


def test_http_error_status(tmp_path, server):
    res = dag8.run(_write(tmp_path, server + "/err"), db_path=tmp_path / "r.sqlite3")
    assert res.status == "error"
    assert "HTTP 500" in (res.edges[0].error or "")


def test_http_rejects_non_http(tmp_path):
    res = dag8.run(_write(tmp_path, "file:///etc/passwd"),
                   db_path=tmp_path / "r.sqlite3")
    assert res.status == "error"
    assert "http(s)" in (res.edges[0].error or "")


def test_http_config_requires_url(tmp_path):
    data = {"name": "t", "seed_tag": "a", "seed_content": "x", "result_tag": "out",
            "edges": [{"id": "h", "kind": "http", "input_tags": ["a"],
                       "output_tags": ["out"]}]}
    p = tmp_path / "dag.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="url"):
        cfg_mod.load(p)

#!/usr/bin/env python3
"""抖动退避基准：限速 mock server 下比较 fixed vs jitter 重试。

```bash
python scripts/jitter_benchmark.py --rounds 3
```
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
for p in (str(V8), str(V8.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dag8  # noqa: E402
import dag8.nodes as nodes_mod  # noqa: E402


class _Throttle(BaseHTTPRequestHandler):
    interval = 0.15
    last = 0.0
    lock = threading.Lock()
    codes = {"200": 0, "429": 0}

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        with type(self).lock:
            now = time.time()
            if now - type(self).last < type(self).interval:
                code = 429
            else:
                type(self).last = now
                code = 200
            type(self).codes[str(code)] += 1
        if code == 429:
            body = b'{"error": {"message": "429 rate_limit"}}'
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


def _build(tmp: Path, base: str) -> Path:
    for name in ("a", "b", "c", "d"):
        (tmp / "graphs").mkdir(parents=True, exist_ok=True)
        (tmp / "graphs" / f"{name}.json").write_text(json.dumps({
            "name": f"sub_{name}",
            "models": {"m": {"api": base, "key": "k", "model": "mock"}},
            "seed_tag": "task", "contents": [],
            "edges": [{"id": name, "input_tags": ["task"], "output_tags": [name],
                       "processor": {"model": "m", "prompt": "确"}}],
        }), encoding="utf-8")
    edges = [{"id": f"g{n}", "kind": "graph", "layer": 0,
              "input_tags": ["task"], "output_tags": [n],
              "graph": {"path": f"graphs/{n}.json", "seed_tag": "task",
                        "result_tag": n, "timeout": 30}}
             for n in ("a", "b", "c", "d")]
    edges.append({"id": "merge", "kind": "graph", "layer": 1,
                  "input_tags": ["a", "b", "c", "d"], "output_tags": ["out"],
                  "graph": {"path": "graphs/a.json", "seed_tag": "task",
                            "result_tag": "a", "timeout": 30}})
    cfg = tmp / "dag.json"
    cfg.write_text(json.dumps({
        "name": "jitter_bench", "seed_tag": "task", "seed_content": "x",
        "result_tag": "out", "edges": edges,
    }), encoding="utf-8")
    return cfg


def run_once(cfg: Path, jitter: bool) -> tuple[float, dict]:
    original = nodes_mod.jittered
    if not jitter:
        nodes_mod.jittered = lambda base, attempt, **kw: base * (kw.get("factor", 2.0) ** attempt)
    _Throttle.codes = {"200": 0, "429": 0}
    _Throttle.last = 0.0
    t0 = time.perf_counter()
    res = dag8.run(cfg, db_path=cfg.parent / "r.sqlite3", max_workers=4)
    wall = time.perf_counter() - t0
    nodes_mod.jittered = original
    return wall, {"status": res.status, **dict(_Throttle.codes)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    nodes_mod._RETRY_SLEEPS = (0.3, 0.6, 1.2)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Throttle)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        cfg = _build(V8 / "data" / "jitter_bench", base)
        rows = []
        for i in range(args.rounds):
            for jitter in (False, True):
                wall, info = run_once(cfg, jitter)
                rows.append({"round": i + 1, "jitter": jitter, "wall": round(wall, 2), **info})
                print(f"round {i+1} jitter={jitter} wall={wall:.2f}s {info}", flush=True)
        fixed = [r["wall"] for r in rows if not r["jitter"] and r["status"] == "ok"]
        jit = [r["wall"] for r in rows if r["jitter"] and r["status"] == "ok"]
        print(f"\nfixed p50 {statistics.median(fixed) if fixed else 0:.2f}s / "
              f"jitter p50 {statistics.median(jit) if jit else 0:.2f}s")
        out = V8 / "data" / "jitter_bench.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写入 {out}")
        return 0
    finally:
        server.shutdown()
        thread.join(2)


if __name__ == "__main__":
    sys.exit(main())

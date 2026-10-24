"""缓存 TTL 的离线测试。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

import dag8  # noqa: E402
from dag8.__main__ import main as dag8_main  # noqa: E402
from dag8.cache import NodeCache  # noqa: E402

BASE = {
    "name": "cachetest", "seed_tag": "task", "seed_content": "seed",
    "result_tag": "out",
    "pi": {"provider": "laptop", "model": "deepseek-flash",
           "timeout": 10, "retries": 1},
    "edges": [{"id": "e", "input_tags": ["task"], "output_tags": ["out"],
               "prompt": "P {contents}"}],
}


class _Agent:
    def __init__(self):
        self.calls = 0

    def run(self, prompt, **kw):
        self.calls += 1
        return "fixed"


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "dag.json"
    p.write_text(json.dumps(BASE), encoding="utf-8")
    return p


def test_cache_roundtrip(tmp_path):
    c = NodeCache(tmp_path)
    c.put("k", "v")
    assert c.get("k") == "v"


def test_cache_ttl_none_keeps_hit(tmp_path):
    c = NodeCache(tmp_path, ttl=None)
    c.put("k", "v")
    assert c.get("k") == "v"


def test_cache_ttl_expiry(tmp_path):
    NodeCache(tmp_path, ttl=0).put("k", "v")
    assert NodeCache(tmp_path, ttl=0).get("k") is None
    assert NodeCache(tmp_path, ttl=100).get("k") == "v"


def test_cache_ttl_old_entry_without_timestamp(tmp_path):
    key = "abc"
    p = tmp_path / key[:2] / f"{key}.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"output": "old"}), encoding="utf-8")
    assert NodeCache(tmp_path, ttl=10).get(key) is None


def test_scheduler_cache_ttl(tmp_path):
    cfg = _write(tmp_path)
    cache_dir = tmp_path / "cache"

    fresh = _Agent()
    dag8.run(cfg, db_path=tmp_path / "a.sqlite3", agent=fresh,
             use_cache=True, cache_dir=cache_dir, cache_ttl=100)
    dag8.run(cfg, db_path=tmp_path / "b.sqlite3", agent=fresh,
             use_cache=True, cache_dir=cache_dir, cache_ttl=100)
    assert fresh.calls == 1, "TTL 未过期时应命中缓存"

    stale = _Agent()
    dag8.run(cfg, db_path=tmp_path / "c.sqlite3", agent=stale,
             use_cache=True, cache_dir=cache_dir, cache_ttl=0)
    dag8.run(cfg, db_path=tmp_path / "d.sqlite3", agent=stale,
             use_cache=True, cache_dir=cache_dir, cache_ttl=0)
    assert stale.calls == 2, "TTL=0 时每次都该 miss"


def test_cli_accepts_cache_ttl(tmp_path):
    cfg = _write(tmp_path)
    assert dag8_main(["run", str(cfg), "--cache", "--cache-ttl", "60",
                      "--dry-run"]) == 0


def test_cache_info_evict_clear(tmp_path):
    c = NodeCache(tmp_path)
    c.put("a", "x")
    c.put("b", "y")
    info = c.info()
    assert info["entries"] == 2
    assert info["bytes"] > 0

    p = c._path("a")
    data = json.loads(p.read_text(encoding="utf-8"))
    data["created_ts"] = time.time() - 1000
    p.write_text(json.dumps(data), encoding="utf-8")

    assert c.evict(100) == 1
    assert c.info()["entries"] == 1
    assert c.clear() == 1
    assert c.info()["entries"] == 0


def test_cli_cache_info_and_clear(tmp_path, capsys):
    c = NodeCache(tmp_path)
    c.put("a", "x")

    assert dag8_main(["cache", "--dir", str(tmp_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["entries"] == 1

    assert dag8_main(["cache", "--dir", str(tmp_path), "--clear"]) == 0
    assert c.info()["entries"] == 0

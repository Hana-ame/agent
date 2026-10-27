#!/usr/bin/env python3
"""扫描所有 DSH session，抽取最终结论与元数据。"""
from __future__ import annotations
import io, json, os, re, time, zstandard
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
DSH_HOME = Path(os.environ.get("DSH_HOME", str(Path.home() / ".dsh")))
STATE = DSH_HOME / "session-manager" / "state.json"
SESSIONS = DSH_HOME / "sessions"
OUT = ROOT / "v8" / "data" / "session_elo" / "sessions.json"
LIMIT = int(os.environ.get("DSH_ELO_LIMIT", "0") or 0)

def text_of(msg: Dict[str, Any]) -> str:
    return "\n".join(str(b["text"]) for b in (msg.get("content") or [])
                     if b.get("type") == "text" and b.get("text")).strip()

def redact(t: str) -> str:
    t = re.sub(r"sk-[A-Za-z0-9_\-]{16,}", "[REDACTED]", t)
    t = re.sub(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[\"']?[^\s\"']{8,}",
               r"\1=[REDACTED]", t)
    return t

def enc(cwd: str) -> str:
    return "--" + cwd.strip("/").replace("/", "-") + "--"

def one(sid: str, meta: Dict[str, Any]) -> Dict[str, Any] | None:
    cwd = str(meta.get("cwd") or "")
    path = SESSIONS / enc(cwd) / sid / "session.v4.jsonl.zstd"
    if not path.exists():
        return None
    try:
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(path.read_bytes())) as r:
            lines = r.read().decode("utf-8", "replace").splitlines()
    except Exception:
        return None
    req = concl = status = ""
    turns = tools = 0
    for line in lines:
        try: obj = json.loads(line)
        except ValueError: continue
        t, d = obj.get("type"), obj.get("data") or {}
        if t == "user/message":
            v = text_of(d)
            if v: req = v
        elif t == "assistant/message":
            v = text_of(d.get("message") or {})
            if v: concl = v
        elif t == "turn/end":
            turns += 1; status = str((d.get("reason") or {}).get("kind") or status)
        elif t == "tool/call": tools += 1
    ts = meta.get("updatedAt") or meta.get("firstSeenAt") or 0
    return {
        "session_id": sid, "title": meta.get("title") or "", "kind": meta.get("kind") or "",
        "cwd": cwd, "updated_iso": time.strftime("%Y-%m-%d %H:%M", time.localtime(ts / 1000)) if ts else "",
        "turns": turns, "tools": tools, "status": status,
        "request": redact(req), "conclusion": redact(concl),
        "evaluable": bool(concl.strip()),
    }

def main() -> int:
    by = json.loads(STATE.read_text(encoding="utf-8"))["byId"]
    rows = sorted(by.items(), key=lambda kv: kv[1].get("updatedAt") or 0, reverse=True)
    if LIMIT > 0:
        rows = rows[:LIMIT]
    out: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for item in pool.map(lambda kv: one(kv[0], kv[1]), rows):
            if item is not None:
                out.append(item)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "count": len(out), "items": out}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"path": str(OUT), "count": len(out),
                      "evaluable": sum(1 for x in out if x["evaluable"]),
                      "by_kind": {k: sum(1 for x in out if x["kind"] == k)
                                  for k in {x["kind"] for x in out}}}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

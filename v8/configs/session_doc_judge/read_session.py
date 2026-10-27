#!/usr/bin/env python3
"""读取指定 DSH session，输出原始结论与元数据。"""
from __future__ import annotations
import io, json, os, re, time, zstandard
from pathlib import Path
from typing import Any, Dict, List

DSH_HOME = Path(os.environ.get("DSH_HOME", str(Path.home() / ".dsh")))
STATE = DSH_HOME / "session-manager" / "state.json"
SESSIONS = DSH_HOME / "sessions"

def text_of(msg: Dict[str, Any]) -> str:
    return "\n".join(str(b["text"]) for b in (msg.get("content") or [])
                     if b.get("type") == "text" and b.get("text")).strip()

def redact(t: str) -> str:
    t = re.sub(r"sk-[A-Za-z0-9_\-]{16,}", "[REDACTED]", t)
    t = re.sub(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[\"']?[^\s\"']{8,}",
               r"\1=[REDACTED]", t)
    return t

def main() -> int:
    data = json.loads(os.environ.get("DAG8_INPUTS") or "{}")
    raw = str(data.get("session_id") or "").strip()
    by = json.loads(STATE.read_text(encoding="utf-8"))["byId"]
    if raw not in by:
        m = [s for s in by if s.startswith(raw)]
        if len(m) != 1:
            raise SystemExit(f"session 无法唯一解析: {raw}")
        raw = m[0]
    meta = by[raw]
    cwd = str(meta.get("cwd") or "")
    path = SESSIONS / ("--" + cwd.strip("/").replace("/", "-") + "--") / raw / "session.v4.jsonl.zstd"
    if not path.exists():
        raise SystemExit(f"transcript 不存在: {path}")
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(path.read_bytes())) as r:
        lines = r.read().decode("utf-8", "replace").splitlines()
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
    print(json.dumps({
        "session_id": raw, "title": meta.get("title") or "", "cwd": cwd,
        "updated_iso": time.strftime("%Y-%m-%d %H:%M", time.localtime(ts / 1000)) if ts else "未知",
        "turns": turns, "tools": tools, "status": status,
        "request": redact(req), "conclusion": redact(concl),
    }, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

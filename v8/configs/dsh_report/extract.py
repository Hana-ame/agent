#!/usr/bin/env python3
"""从 DSH 历史 session 中抽取结论，输出 JSON 供 v7/v8/v9 生成报告。

- 数据源：~/.dsh/session-manager/state.json + ~/.dsh/sessions/<cwd>/<id>/session.v4.jsonl.zstd
- 只取当前工作区的 top-level session，排除当前会话
- 结论 = 最后一个 assistant text block（跳过 reasoning）
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
import zstandard
from pathlib import Path
from typing import Any, Dict, List

DSH_HOME = Path(os.environ.get("DSH_HOME", str(Path.home() / ".dsh")))
STATE = DSH_HOME / "session-manager" / "state.json"
SESSIONS = DSH_HOME / "sessions"
CWD = "/mnt/c/Users/lumin/Downloads/agent"

REDACT = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "[REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[\"']?[^\s\"']{8,}"),
     r"\1=[REDACTED]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{12,}"), "Bearer [REDACTED]"),
]


def redact(text: str) -> str:
    for pat, rep in REDACT:
        text = pat.sub(rep, text)
    return text


def read_lines(path: Path) -> List[str]:
    raw = path.read_bytes()
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)) as reader:
        return reader.read().decode("utf-8", "replace").splitlines()


def text_of(message: Dict[str, Any]) -> str:
    out = []
    for block in message.get("content") or []:
        if block.get("type") == "text" and block.get("text"):
            out.append(str(block["text"]))
    return "\n".join(out).strip()


def extract_session(session_id: str, meta: Dict[str, Any]) -> Dict[str, Any] | None:
    encoded = "--" + CWD.strip("/").replace("/", "-") + "--"
    path = SESSIONS / encoded / session_id / "session.v4.jsonl.zstd"
    if not path.exists():
        return None
    last_user = ""
    last_assistant = ""
    turns = 0
    tools = 0
    status = ""
    for line in read_lines(path):
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        kind = obj.get("type")
        data = obj.get("data") or {}
        if kind == "user/message":
            value = text_of(data)
            if value:
                last_user = value
        elif kind == "assistant/message":
            value = text_of(data.get("message") or {})
            if value:
                last_assistant = value
        elif kind == "turn/end":
            turns += 1
            status = str((data.get("reason") or {}).get("kind") or status)
        elif kind == "tool/call":
            tools += 1
    if not last_assistant:
        return None
    return {
        "session_id": session_id,
        "title": meta.get("title") or "",
        "description": meta.get("description") or "",
        "cwd": meta.get("cwd") or "",
        "updated_at": meta.get("updatedAt") or 0,
        "updated_iso": time.strftime(
            "%Y-%m-%d %H:%M",
            time.localtime((meta.get("updatedAt") or 0) / 1000),
        ),
        "turns": turns,
        "tools": tools,
        "status": status,
        "request": redact(last_user),
        "conclusion": redact(last_assistant),
    }


def collect(limit: int = 6) -> List[Dict[str, Any]]:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    current = os.environ.get("DSH_SESSION_ID", "")
    rows = []
    for session_id, meta in state["byId"].items():
        if meta.get("cwd") != CWD:
            continue
        if meta.get("kind") != "top-level":
            continue
        if session_id == current:
            continue
        item = extract_session(session_id, meta)
        if item:
            rows.append(item)
    rows.sort(key=lambda x: (len(x["conclusion"]), x["updated_at"]), reverse=True)
    return rows[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--out", default="v8/data/dsh_report/input.json")
    args = ap.parse_args()
    items = collect(args.limit)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sessions": items}, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "count": len(items),
        "sessions": [
            {"id": x["session_id"][:20], "title": x["title"],
             "chars": len(x["conclusion"]), "turns": x["turns"]}
            for x in items
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
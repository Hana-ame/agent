#!/usr/bin/env python3
"""v8 script 节点：抽取指定 DSH session 的结论。

输入：DAG8_INPUTS["session_id"]（完整 id 或前缀）
输出：JSON，含 title / request / conclusion / turns / tools / status
"""
from __future__ import annotations

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
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(path.read_bytes())) as reader:
        return reader.read().decode("utf-8", "replace").splitlines()


def text_of(message: Dict[str, Any]) -> str:
    return "\n".join(
        str(block["text"]) for block in (message.get("content") or [])
        if block.get("type") == "text" and block.get("text")
    ).strip()


def encoded_cwd(cwd: str) -> str:
    return "--" + cwd.strip("/").replace("/", "-") + "--"


def resolve_session(raw: str, by_id: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    raw = raw.strip()
    if raw in by_id:
        return raw, by_id[raw]
    matches = [sid for sid in by_id if sid.startswith(raw)]
    if len(matches) == 1:
        return matches[0], by_id[matches[0]]
    if not matches:
        raise SystemExit(f"session 不存在: {raw}")
    raise SystemExit(f"session 前缀不唯一: {raw} -> {matches[:5]}")


def extract(session_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    path = (SESSIONS / encoded_cwd(str(meta.get("cwd") or ""))
            / session_id / "session.v4.jsonl.zstd")
    if not path.exists():
        raise SystemExit(f"transcript 不存在: {path}")
    last_user = ""
    last_assistant = ""
    turns = tools = 0
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
        raise SystemExit("该 session 没有 assistant 结论")
    ts = meta.get("updatedAt") or meta.get("firstSeenAt") or 0
    updated_iso = (time.strftime("%Y-%m-%d %H:%M", time.localtime(ts / 1000))
                   if ts else "未知")
    return {
        "session_id": session_id,
        "title": meta.get("title") or "",
        "description": meta.get("description") or "",
        "cwd": meta.get("cwd") or "",
        "updated_iso": updated_iso,
        "turns": turns,
        "tools": tools,
        "status": status,
        "request": redact(last_user),
        "conclusion": redact(last_assistant),
    }


def main() -> int:
    inputs = json.loads(os.environ.get("DAG8_INPUTS") or "{}")
    raw = str(inputs.get("session_id") or os.environ.get("DSH_SESSION", "")).strip()
    if not raw:
        raise SystemExit("缺少 session_id")
    state = json.loads(STATE.read_text(encoding="utf-8"))
    session_id, meta = resolve_session(raw, state["byId"])
    print(json.dumps(extract(session_id, meta), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
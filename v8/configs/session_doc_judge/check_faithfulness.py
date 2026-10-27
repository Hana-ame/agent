#!/usr/bin/env python3
"""忠实/可追溯检查：文档内容与 session 原文的重合度、session id、source。"""
from __future__ import annotations
import json, os, re

def shingles(text, n=4):
    text = re.sub(r"\s+", "", text)
    return {text[i:i+n] for i in range(max(0, len(text)-n+1))}

def main() -> int:
    data = json.loads(os.environ["DAG8_INPUTS"])
    session = json.loads(data["session"]); doc = json.loads(data["doc"])["text"]
    src = (session.get("request") or "") + "\n" + (session.get("conclusion") or "")
    ds, ss = shingles(doc), shingles(src)
    supported = len(ds & ss) / max(1, len(ds))
    sid = session.get("session_id") or ""
    has_sid = sid in doc
    has_source = ("dsh-session://" in doc) or ("source" in doc.lower() and sid in doc)
    score = 0.7 * supported + (0.2 if has_sid else 0.0) + (0.1 if has_source else 0.0)
    print(json.dumps({
        "supported_ratio": round(supported, 3),
        "session_id_present": has_sid,
        "source_present": has_source,
        "score": round(min(1.0, score), 3),
    }, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

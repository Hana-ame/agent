"""把一段对话渲染成“会话状态”文本，供每条预设检查。

模式：
- full      全文 + 元信息
- last      只取最后 N 条消息（默认 1 条，即“最近一条”）
- mainline  主轴：优先用对话自带 mainline；否则去掉工具/系统噪音只留 user/assistant
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

META_KEYS = ("time", "model", "provider", "title", "url", "agent")

_RENDER_ROLE = {
    "user": "用户", "assistant": "助手", "system": "系统",
    "tool": "工具", "function": "工具",
}


def load_conv(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt_meta(conv: Dict[str, Any]) -> List[str]:
    meta = conv.get("meta") or {}
    lines = []
    for k in META_KEYS:
        v = meta.get(k)
        if v:
            lines.append(f"- {k}: {v}")
    return lines


def _fmt_msgs(messages: Sequence[Dict[str, Any]]) -> List[str]:
    out = []
    for i, m in enumerate(messages, 1):
        role = str(m.get("role", "?")).lower()
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        label = _RENDER_ROLE.get(role, role)
        out.append(f"{i}. [{label}] {content}")
    return out


def render_state(conv: Dict[str, Any], mode: str = "mainline",
                 last_n: int = 1) -> str:
    messages = list(conv.get("messages") or [])
    parts = []
    meta = _fmt_meta(conv)
    if meta:
        parts.append("## 会话元信息\n" + "\n".join(meta))

    if mode == "full":
        body = _fmt_msgs(messages)
    elif mode == "last":
        body = _fmt_msgs(messages[-max(1, int(last_n)):])
    elif mode == "mainline":
        spine = conv.get("mainline")
        if not spine:
            spine = [m for m in messages
                     if str(m.get("role", "")).lower() in ("user", "assistant")]
        body = _fmt_msgs(spine)
    else:
        raise ValueError(f"未知状态模式：{mode}（可选 full/last/mainline）")

    if body:
        parts.append("## 会话状态\n" + "\n".join(body))
    return "\n\n".join(parts)

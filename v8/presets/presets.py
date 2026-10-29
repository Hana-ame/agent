"""预设：一段完整 system prompt + 元信息。检查时由专属 LLM 依据它判定会话状态。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Preset:
    id: str
    name: str
    system_prompt: str
    on_hit: str = ""   # 命中后动作（通知/记录/路由…），目前只是元信息

    @property
    def tag(self) -> str:
        return "check_" + re.sub(r"[^A-Za-z0-9_]", "_", self.id)


def load_presets(directory: str | Path) -> List[Preset]:
    d = Path(directory)
    out: List[Preset] = []
    for p in sorted(d.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        out.append(Preset(
            id=str(data.get("id") or p.stem),
            name=str(data.get("name") or p.stem),
            system_prompt=str(data["system_prompt"]),
            on_hit=str(data.get("on_hit") or ""),
        ))
    ids = [x.id for x in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"预设 id 重复：{ids}")
    return out

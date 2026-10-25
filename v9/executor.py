"""v9 执行适配器：不自己造调度，直接用 v7 图 + v8 组织。

- generate(prompt)：v8 跑一个 graph 节点，里面是 v7 单边 LLM 图
- fetch(url)：v8 跑一个 http 节点
- 两者都复用 v8 的 sqlite 轨迹、缓存与超时
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
V8 = ROOT / "v8"
for p in (str(ROOT), str(V8)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dag8  # noqa: E402

CFG_GEN = Path(__file__).resolve().parent / "configs" / "gen.json"
CFG_GET = Path(__file__).resolve().parent / "configs" / "get.json"
DB = Path(__file__).resolve().parent / "data" / "executor.sqlite3"


class V8Executor:
    def __init__(self, db_path: str | Path = DB) -> None:
        self.db = Path(db_path)
        self.db.parent.mkdir(parents=True, exist_ok=True)

    def _run(self, cfg: Path, seed_tag: str, seed: str) -> str:
        res = dag8.run(str(cfg), seed_content=seed,
                       db_path=str(self.db), max_workers=1, rpm=3,
                       use_cache=False, max_seconds=300)
        if res.status != "ok" or not res.result:
            raise RuntimeError(f"v8 {cfg.name} 失败：{res.status} "
                               f"{res.summary()}")
        return res.result

    def generate(self, prompt: str) -> str:
        return self._run(CFG_GEN, "task", prompt)

    def fetch(self, url: str) -> Dict[str, str]:
        return {"url": url, "text": self._run(CFG_GET, "url", url)}

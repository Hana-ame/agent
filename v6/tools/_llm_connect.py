"""框架自举用的连接层（不是框架的公共 API）。

两个函数：

- `connect_llm(model_key)`：按 `models.json` 里的注册项造一个真实可用的模型客户端。
  它返回的对象可被 **旧版或新版** 的任何一代 `OpenAICompatibleLLM` 直接使用——
  只要把同名方法挂上去即可（见 `_ensure_interface`）。
- `connect_model_config(model_key)`：取同一条注册项的 `{model, api, key, params}`。

为什么单独一层：不同代框架的 `llm.py` 里的类名/构造签名可能不一样
（有的要求 `base_url` 必填、有的参数叫 `api`），连接层统一兜住，
生成的代码就不必关心"这一代怎么传参"。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
MODELS_FILE = ROOT / "models.json"


def connect_model_config(model_key: str = "laptop-6.8") -> Dict[str, Any]:
    """从 `models.json` 取一条注册项；缺文件时退回内置默认（端点固定，不换 api）。"""
    try:
        data = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    cfg = dict(data.get(model_key) or {})
    cfg.setdefault("api", os.environ.get("VEA_BASE_URL", "https://laptop-8000.moonchan.xyz/v1"))
    cfg.setdefault("key", os.environ.get("VEA_API_KEY", "not-needed"))
    cfg.setdefault("model", "deepseek-v4-flash")
    cfg.setdefault("params", {"temperature": 0.2, "max_tokens": 32768})
    return cfg


def _ensure_interface(obj: Any) -> Any:
    """让客户端同时具备 `ask` / `_post_sync` / `complete` 三种入口。

    `_post_sync` 是同步实现，键名在各代之间稳定（实测三代都有），
    所以只要它有，就一定能补出另外两个入口。
    """
    post = getattr(obj, "_post_sync", None)
    if not callable(post):
        return obj

    def complete(prompt: str, **kw: Any) -> str:
        return post(prompt, **kw)

    async def ask(prompt: str, **kw: Any) -> str:
        import asyncio

        return await asyncio.to_thread(complete, prompt, **kw)

    if not callable(getattr(obj, "ask", None)):
        obj.ask = ask  # type: ignore[attr-defined]
    if not callable(getattr(obj, "complete", None)):
        obj.complete = complete  # type: ignore[attr-defined]
    return obj


def connect_llm(model_key: str = "laptop-6.8", cls: Any = None) -> Any:
    """造一个真实的模型客户端（默认用当前代框架里的 `OpenAICompatibleLLM`）。"""
    cfg = connect_model_config(model_key)
    if cls is None:
        from vea_v6.parts.llm import OpenAICompatibleLLM as cls  # type: ignore[no-redef]
    base = cfg["api"]
    candidates = (
        dict(model=cfg["model"], api_key=cfg["key"], base_url=base),
        dict(model=cfg["model"], api=base, key=cfg["key"]),
        dict(model=cfg["model"]),
        dict(),
    )
    last: Exception | None = None
    for kw in candidates:
        try:
            obj = cls(**kw)
        except TypeError as exc:  # 签名不接受这些键
            last = exc
            continue
        obj.model = cfg["model"]
        if getattr(obj, "base_url", None) in (None, "", "https://api.openai.com/v1"):
            obj.base_url = base
        obj.api_key = cfg["key"]
        obj.params = dict(cfg.get("params") or {})
        return _ensure_interface(obj)
    raise RuntimeError(f"无法实例化模型客户端 {cls!r}: {last}")

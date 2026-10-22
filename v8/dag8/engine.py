"""v8 引擎：读 JSON 配置 → 按 layer 跑 edge → 每条 edge 交给 pi → 结果进 sqlite3。

刻意做小：没有相似度匹配、没有预设边、没有默认处理器。
匹配就是 tag 精确相等，缺失就跳过并记 `skipped`，行为可预测。

一次运行的完整轨迹都能在库里查到：
    run_id → runs 一行 / contents 若干 / edges 若干
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dag8 import config as cfg_mod
from dag8.pi import PiAgent, PiError
from dag8.store import Store

__all__ = ["run", "RunResult", "EdgeRun"]


@dataclass
class EdgeRun:
    edge_id: str
    layer: int
    status: str            # ok | skipped | error
    elapsed: float = 0.0
    content_id: Optional[int] = None
    error: Optional[str] = None
    prompt_chars: int = 0
    output_chars: int = 0


@dataclass
class RunResult:
    run_id: str
    dag_name: str
    result: Optional[str]
    result_tag: str
    edges: List[EdgeRun] = field(default_factory=list)
    db_path: str = ""
    status: str = "ok"

    def summary(self) -> str:
        lines = [
            f"run {self.run_id} · {self.dag_name} · {self.status}",
            f"  result: {self.result_tag} = "
            f"{'(空)' if not self.result else f'{len(self.result)} 字符'}",
        ]
        for e in self.edges:
            mark = {"ok": "✓", "skipped": "·", "error": "✗"}.get(e.status, "?")
            extra = f"  {e.elapsed:.1f}s  {e.output_chars} 字符" if e.status == "ok" else ""
            err = f"  {e.error[:80]}" if e.error else ""
            lines.append(f"  {mark} [{e.layer}] {e.edge_id}{extra}{err}")
        return "\n".join(lines)


def run(
    config_path: str | Path,
    *,
    seed_content: Optional[str] = None,
    db_path: Optional[str | Path] = None,
    agent: Optional[PiAgent] = None,
    dry_run: bool = False,
) -> RunResult:
    """跑一张配置好的图。

    `dry_run=True` 时不调 pi，把渲染后的 prompt 原样当作输出写库 —— 用来验证
    配置与占位符渲染，不烧额度。
    """
    cfg = cfg_mod.load(config_path)
    db = Path(db_path or cfg.db_path or (Path(config_path).resolve().parent.parent / "data" / "dag8.sqlite3"))

    seed = seed_content if seed_content is not None else (cfg.seed_content or "")
    if not cfg.seed_tag:
        raise ValueError(f"{config_path} 没有 seed_tag，无法注入种子")

    with Store(db) as store:
        run_id = store.new_run(cfg.name, str(Path(config_path).resolve()), cfg.seed_tag, seed)
        store.add_content(run_id, [cfg.seed_tag], seed, gen=0, source_edge=None)

        result = RunResult(run_id=run_id, dag_name=cfg.name, result=None,
                           result_tag=cfg.result_tag, db_path=str(db))

        for edge in cfg.sorted_edges():
            # 收集该边需要的输入
            per_tag: Dict[str, List[str]] = {}
            missing: List[str] = []
            for tag in edge.input_tags:
                items = store.contents_for_tag(run_id, tag)
                if not items:
                    missing.append(tag)
                    continue
                per_tag[tag] = [it["content"] for it in items]

            if missing:
                rec = EdgeRun(edge.id, edge.layer, "skipped", error=f"缺输入 tag: {missing}")
                result.edges.append(rec)
                store.log_edge(run_id, edge.id, edge.layer, edge.input_tags, edge.output_tags,
                               status="skipped", error=rec.error)
                continue

            prompt = cfg_mod.render_prompt(edge.prompt, per_tag)
            provider, model = cfg.pi.provider, cfg.pi.model
            if edge.model:
                provider, _, model = edge.model.partition("/")
                if not model:
                    provider, model = cfg.pi.provider, edge.model

            t0 = time.perf_counter()
            try:
                if dry_run:
                    out = prompt
                else:
                    runner = agent or PiAgent(cfg.pi)
                    out = runner.run(
                        prompt,
                        provider=provider,
                        model=model,
                        tools=cfg.pi.tools if edge.tools is None else edge.tools,
                        timeout=edge.timeout or cfg.pi.timeout,
                    )
            except PiError as exc:
                elapsed = time.perf_counter() - t0
                rec = EdgeRun(edge.id, edge.layer, "error", elapsed=elapsed,
                              error=str(exc), prompt_chars=len(prompt))
                result.edges.append(rec)
                store.log_edge(run_id, edge.id, edge.layer, edge.input_tags, edge.output_tags,
                               model=f"{provider}/{model}", prompt_chars=len(prompt),
                               status="error", error=str(exc), elapsed=elapsed)
                result.status = "error"
                store.finish_run(run_id, "error", error=f"{edge.id}: {exc}")
                return result

            elapsed = time.perf_counter() - t0
            gen = max((it["gen"] for it in store.all_contents(run_id)), default=0) + 1
            cid = store.add_content(run_id, edge.output_tags, out, gen=gen, source_edge=edge.id)
            rec = EdgeRun(edge.id, edge.layer, "ok", elapsed=elapsed, content_id=cid,
                          prompt_chars=len(prompt), output_chars=len(out))
            result.edges.append(rec)
            store.log_edge(run_id, edge.id, edge.layer, edge.input_tags, edge.output_tags,
                           model=f"{provider}/{model}", prompt_chars=len(prompt),
                           status="ok", elapsed=elapsed, content_id=cid)

        result.result = store.result(run_id, cfg.result_tag)
        if result.result is None:
            result.status = "incomplete"
            store.finish_run(run_id, "incomplete", result_tag=cfg.result_tag)
        else:
            store.finish_run(run_id, "ok", result_tag=cfg.result_tag,
                             result_content_id=store.contents_for_tag(run_id, cfg.result_tag)[-1]["id"])
        return result

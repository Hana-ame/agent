"""库式 DAG 会话：脚本自己驱动 step()，卡住时在外部处理再 set 回去。

```python
from dag8 import Dag

d = Dag("configs/demo.json", seed_content="任务")
while True:
    r = d.step()
    if r.status == "blocked":
        print("缺输入：", r.blocked)
        print("当前内容：", d.contents)
        d.set("external_result", "脚本在外面算好的结果")
    elif r.status in ("done", "error", "incomplete"):
        break
print(d.result)
```

和 `dag8.run()` 的区别：run() 一次跑完整张图；Dag 把控制权交给调用方，
每次 step 只执行一条已就绪的边，便于脚本在“卡住”时插入自己的处理。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dag8 import config as cfg_mod
from dag8.contracts import canonical_output, validate_output
from dag8.nodes import build_node
from dag8.scheduler import _run_judge, approval_block_reason
from dag8.store import Store

__all__ = ["Dag", "StepResult"]


@dataclass
class StepResult:
    status: str                      # ran | blocked | done | error | incomplete
    ran: List[str] = field(default_factory=list)
    blocked: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class Dag:
    def __init__(self, config_path: str | Path, *,
                 db_path: Optional[str | Path] = None,
                 seed_content: Optional[str] = None,
                 seed_tag: Optional[str] = None,
                 agent: Any = None) -> None:
        self.cfg = cfg_mod.load(config_path)
        self.config_path = Path(config_path).resolve()
        db = Path(db_path or self.cfg.db_path or
                  (self.config_path.parent.parent / "data" / "dag8.sqlite3"))
        self.store = Store(db)
        self.agent = agent
        tag = seed_tag or self.cfg.seed_tag or "seed"
        content = self.cfg.seed_content if seed_content is None else seed_content
        self.run_id = self.store.new_run(self.cfg.name, str(self.config_path),
                                         tag, content or "")
        if content:
            self.store.add_content(self.run_id, [tag], content)
        self._finished = False

    # -- 只读视图 -----------------------------------------------------------
    @property
    def contents(self) -> Dict[str, List[str]]:
        """当前内容池：tag -> 所有内容（按写入顺序）。"""
        out: Dict[str, List[str]] = {}
        for item in self.store.all_contents(self.run_id):
            for tag in item["tags"]:
                out.setdefault(tag, []).append(item["content"])
        return out

    @property
    def result(self) -> Optional[str]:
        return self.store.result(self.run_id, self.cfg.result_tag)

    @property
    def status(self) -> str:
        return self.store.run_info(self.run_id)["status"]

    @property
    def blocked(self) -> List[Dict[str, Any]]:
        pending, _ = self._scan()
        return self._blocked_entries(pending)

    # -- 写回外部分结果 ------------------------------------------------------
    def set(self, tag: str, content: str) -> int:
        if not tag:
            raise ValueError("tag 不能为空")
        return self.store.add_content(self.run_id, [tag], str(content),
                                      source_edge="external")

    # -- 执行 ---------------------------------------------------------------
    def step(self) -> StepResult:
        if self._finished:
            return StepResult("done")
        if self.result is not None:
            self._finish_ok()
            return StepResult("done")

        while True:
            pending, done = self._scan()
            skipped = self._skip_when(pending)
            if skipped:
                # when 不满足的边记为 skipped 后继续看下一批
                continue

            ready = [e for e in pending
                     if not self._missing(e) and not self._approval_reason(e)]
            if ready:
                res = self._execute(ready[0])
                if res.status == "ran" and self.result is not None:
                    self._finish_ok()
                    return StepResult("done", ran=res.ran)
                return res

            blocked = self._blocked_entries(pending)
            if blocked:
                return StepResult("blocked", blocked=blocked)
            if not pending:
                if self.result is not None:
                    self._finish_ok()
                    return StepResult("done")
                return StepResult("incomplete")
            # 只剩不可达的边（缺输入且没有标记 blocked 的情况）
            return StepResult("blocked", blocked=[
                {"edge": e.id, "missing": self._missing(e)} for e in pending
            ])

    def run_until_blocked(self, max_steps: int = 1000) -> StepResult:
        last = StepResult("incomplete")
        for _ in range(max_steps):
            last = self.step()
            if last.status in ("done", "blocked", "error", "incomplete"):
                return last
        self._finish("incomplete", error=f"超过 {max_steps} 步")
        return StepResult("incomplete", error=f"超过 {max_steps} 步")

    # -- 内部 ---------------------------------------------------------------
    def _scan(self):
        logged = {e["edge_id"] for e in self.store.edge_records(self.run_id)}
        pending = [e for e in self.cfg.sorted_edges() if e.id not in logged]
        return pending, logged

    def _missing(self, edge) -> List[str]:
        if edge.join == "any":
            if any(self.store.contents_for_tag(self.run_id, t)
                   for t in edge.input_tags):
                return []
            return list(edge.input_tags)
        return [t for t in edge.input_tags
                if not self.store.contents_for_tag(self.run_id, t)
                and t not in edge.optional_inputs]

    def _approval_reason(self, edge) -> Optional[str]:
        if not edge.approval:
            return None
        return approval_block_reason(edge, self.cfg)

    def _blocked_entries(self, pending) -> List[Dict[str, Any]]:
        out = []
        for e in pending:
            missing = self._missing(e)
            if missing:
                out.append({"edge": e.id, "missing": missing})
                continue
            reason = self._approval_reason(e)
            if reason:
                out.append({"edge": e.id, "reason": reason})
        return out

    def _skip_when(self, pending) -> List[str]:
        skipped = []
        from dag8.scheduler import when_matches
        for edge in pending:
            if self._missing(edge) or not edge.when:
                continue
            if when_matches(edge.when, self.store, self.run_id):
                continue
            self.store.log_edge(self.run_id, edge.id, edge.layer,
                                edge.input_tags, edge.output_tags,
                                status="skipped", error="when 条件不满足")
            skipped.append(edge.id)
        return skipped

    def _execute(self, edge) -> StepResult:
        inputs = {
            t: [c["content"] for c in self.store.contents_for_tag(self.run_id, t)]
            for t in edge.input_tags
        }
        node = build_node(edge, self.cfg, agent=self.agent)
        preview = node.preview(inputs)
        t0 = time.perf_counter()
        try:
            reason = ""
            out = ""
            for attempt in range(int(edge.repair) + 1):
                out = node.run(inputs, repair=reason)
                ok, reason = validate_output(out, edge.expect)
                if ok:
                    break
                if not node.supports_repair or attempt >= int(edge.repair):
                    raise RuntimeError(f"输出契约失败：{reason}")
            out = canonical_output(out, edge.expect)
            if edge.judge:
                out = _run_judge(edge, out, self.cfg, self.agent, None)
            cid = self.store.add_content(self.run_id, edge.output_tags, out,
                                         source_edge=edge.id)
            self.store.log_edge(self.run_id, edge.id, edge.layer,
                                edge.input_tags, edge.output_tags,
                                model=node.label, prompt_chars=len(preview),
                                status="ok", elapsed=time.perf_counter() - t0,
                                content_id=cid)
            return StepResult("ran", ran=[edge.id])
        except Exception as exc:  # noqa: BLE001
            self.store.log_edge(self.run_id, edge.id, edge.layer,
                                edge.input_tags, edge.output_tags,
                                model=node.label, prompt_chars=len(preview),
                                status="error", error=str(exc),
                                elapsed=time.perf_counter() - t0)
            return StepResult("error", ran=[edge.id], error=str(exc))

    def _finish_ok(self) -> None:
        if self._finished:
            return
        items = self.store.contents_for_tag(self.run_id, self.cfg.result_tag)
        self.store.finish_run(self.run_id, "ok", result_tag=self.cfg.result_tag,
                              result_content_id=items[-1]["id"] if items else None)
        self._finished = True

    def _finish(self, status: str, *, error: Optional[str] = None) -> None:
        if self._finished:
            return
        self.store.finish_run(self.run_id, status, error=error)
        self._finished = True

    def close(self) -> None:
        if not self._finished:
            self._finish("incomplete", error="脚本未跑完就关闭")
        self.store.close()

    def __enter__(self) -> "Dag":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

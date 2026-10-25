"""v8 调度器：只做四件事 —— 按依赖取输入、构造节点、调用、落库。

刻意做小：没有相似度匹配、没有默认处理器、没有 fn 节点。
匹配就是 tag 精确相等，缺失就跳过并记 `skipped`，行为可预测。

调度形式：**波次（wave）**。每一轮把所有「输入 tag 已就绪」的节点挑出来，
按 `max_workers` 并发执行；跑完把结果写回 sqlite，再进入下一轮。所以：

- 默认 `max_workers=1` = 旧的严格串行（按 layer 顺序）
- `max_workers>1` = 同一波次并行；互不依赖的分支可以真正同时跑
- 节点内部依赖仍然由 tag 决定，波次不会破坏正确性

节点只有两种（见 `dag8/nodes.py`）：
- `pi`    ：一次 pi agent 调用（默认，也是唯一原生节点）
- `graph` ：一张 v7 图（实验特性）

缓存（可选，`use_cache=True`）：按「节点指纹 + 输入 + 模型」命中后直接复用输出，
不调 pi / 不跑 v7。默认关闭；同一张图反复调、或换模型对比时打开。

一次运行的完整轨迹都能在库里查到：
    run_id → runs 一行 / contents 若干 / edges 若干
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dag8 import config as cfg_mod
from dag8.cache import NodeCache
from dag8.contracts import canonical_output, validate_output
from dag8.nodes import Node, NodeError, build_node
from dag8.pi import PiAgent, PiError
from dag8.ratelimit import shared_limiter
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
    cached: bool = False


@dataclass
class RunResult:
    run_id: str
    dag_name: str
    result: Optional[str]
    result_tag: str
    edges: List[EdgeRun] = field(default_factory=list)
    db_path: str = ""
    status: str = "ok"
    workers_used: List[int] = field(default_factory=list)
    cost_estimate: float = 0.0

    def summary(self) -> str:
        lines = [
            f"run {self.run_id} · {self.dag_name} · {self.status}",
            f"  result: {self.result_tag} = "
            f"{'(空)' if not self.result else f'{len(self.result)} 字符'}",
        ]
        for e in self.edges:
            mark = {"ok": "✓", "skipped": "·", "error": "✗"}.get(e.status, "?")
            extra = f"  {e.elapsed:.1f}s  {e.output_chars} 字符" if e.status == "ok" else ""
            if e.cached:
                extra += "  (cache)"
            err = f"  {e.error[:80]}" if e.error else ""
            lines.append(f"  {mark} [{e.layer}] {e.edge_id}{extra}{err}")
        return "\n".join(lines)


def is_rate_limit(message: str) -> bool:
    """429 / 限流 / 批量超时都算过载信号（AIMD 降并发）。"""
    low = message.lower()
    return ("429" in message or "rate" in low or "too many" in low
            or "限流" in message or "quota" in low
            or "timeout" in low or "timed out" in low or "超时" in message)


@dataclass
class AdaptiveConcurrency:
    """AIMD：429 时减半，成功时 +1，夹在 [min,max]。"""

    max_workers: int
    min_workers: int = 1
    current: int = field(init=False)

    def __post_init__(self) -> None:
        self.current = max(self.min_workers, int(self.max_workers))

    def observe(self, errors: Sequence[str], successes: int) -> int:
        if any(is_rate_limit(e) for e in errors):
            self.current = max(self.min_workers, self.current // 2)
        elif successes > 0:
            self.current = min(self.max_workers, self.current + 1)
        return self.current


_JUDGE_PROMPT = (
    "你是严格的评审。请给下面内容的完成质量打分（0-10），"
    "只输出 JSON：{\"score\": 0, \"reason\": \"...\"}\n\n内容：\n{output}"
)


def _parse_judge(text: str) -> Tuple[float, str]:
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(raw[start:end + 1])
        except ValueError:
            data = None
        if isinstance(data, dict) and "score" in data:
            try:
                return float(data["score"]), str(data.get("reason") or "")
            except (TypeError, ValueError):
                pass
    m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
    if m:
        return float(m.group(1)), ""
    raise NodeError(f"judge 输出不是合法评分 JSON：{raw[:200]}")


def _run_judge(edge: Any, output: str, cfg: Any, agent: Any,
               limiter: Optional[object]) -> str:
    spec = edge.judge or {}
    template = str(spec.get("prompt") or _JUDGE_PROMPT)
    prompt = (template.replace("{output}", output)
              if "{output}" in template else f"{template}\n\n{output}")
    provider, model = cfg.pi.provider, cfg.pi.model
    if spec.get("model"):
        raw_model = str(spec["model"])
        prov, sep, name = raw_model.partition("/")
        if sep:
            provider, model = prov, name
        else:
            model = raw_model
    if limiter is not None:
        limiter.acquire()
    runner = agent or PiAgent(cfg.pi)
    try:
        raw = runner.run(prompt, provider=provider, model=model,
                         timeout=cfg.pi.timeout)
    except PiError as exc:
        raise NodeError(f"judge 调用失败：{exc}") from exc
    score, reason = _parse_judge(raw)
    if not 0 <= score <= 10:
        raise NodeError(f"judge 分数越界：{score}")
    if score < float(spec["min_score"]):
        raise NodeError(f"judge 分数 {score:g} < {spec['min_score']:g}：{reason}")
    return output


def _edge_cost(model: str, prompt_chars: int, output_chars: int,
               prices: Dict[Any, Any]) -> float:
    prov, _, name = str(model or "").partition("/")
    if not name or (prov, name) not in prices:
        return 0.0
    pin, pout = prices[(prov, name)]
    return ((prompt_chars or 0) / 2.0 * pin
            + (output_chars or 0) / 2.0 * pout) / 1e6


def approval_block_reason(edge: Any, cfg: Any) -> Optional[str]:
    """人工闸门：None = 放行；否则返回阻止原因。"""
    spec = edge.approval
    if not spec:
        return None
    base = Path(cfg.path).resolve().parent if cfg.path else Path.cwd()
    marker = Path(str(spec["marker"]))
    if not marker.is_absolute():
        marker = base / marker
    if not marker.is_file():
        return f"等待人工批准：创建 {marker} 后 --resume 续跑"
    text = marker.read_text(encoding="utf-8", errors="replace").strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            data = {}
        if isinstance(data, dict) and data.get("approved") is False:
            return f"人工拒绝：{data.get('reason') or marker}"
    return None


def when_matches(when: Dict[str, Any], store: Store, run_id: str) -> bool:
    """条件边：tag 内容是否满足 when。"""
    text = "\n".join(it["content"] for it in store.contents_for_tag(run_id, str(when["tag"])))
    present = bool(text.strip())
    if when.get("present") is True and not present:
        return False
    if when.get("present") is False and present:
        return False
    if when.get("absent") is True and present:
        return False
    if when.get("absent") is False and not present:
        return False
    if "pattern" in when and not re.search(str(when["pattern"]), text, re.S):
        return False
    if "equals" in when and text.strip() != str(when["equals"]):
        return False
    return True


Job = Tuple[Any, Node, Dict[str, List[str]], str, Optional[str]]  # edge,node,inputs,preview,key


def _run_jobs(jobs: Sequence[Job], *, dry_run: bool, max_workers: int,
              limiter: Optional[RateLimiter] = None, cfg: Any = None,
              agent: Any = None) -> List[Tuple]:
    """执行一批已就绪的节点；返回 (edge, label, preview_len, elapsed, 输出或异常, cached, key)。

    线程里不碰 sqlite、也不碰缓存 —— 输入/缓存键在进线程前就备好，输出由主线程统一落库。
    """

    def work(job: Job) -> Tuple:
        edge, node, per_tag, preview, key = job
        t0 = time.perf_counter()
        try:
            if (limiter is not None and not dry_run
                    and getattr(node, "uses_api", False)
                    and not getattr(node, "self_limits", False)):
                limiter.acquire()
            out = preview if dry_run else node.run(per_tag)
            if not dry_run and edge.expect:
                ok, reason = validate_output(out, edge.expect)
                retries = max(0, int(edge.repair))
                while not ok and retries > 0 and getattr(node, "supports_repair", False):
                    retries -= 1
                    out = node.run(per_tag, repair=(
                        f"上一次输出未通过格式校验：{reason}。"
                        "请重新输出，严格满足要求，不要解释。"
                    ))
                    ok, reason = validate_output(out, edge.expect)
                if not ok:
                    raise NodeError(f"输出契约失败：{reason}")
            out = canonical_output(out, edge.expect)
            if not dry_run and edge.judge:
                out = _run_judge(edge, out, cfg, agent, limiter)
            return (edge, node.label, len(preview), time.perf_counter() - t0, out, False, key)
        except (PiError, NodeError) as exc:
            return (edge, node.label, len(preview), time.perf_counter() - t0, exc, False, key)
        except Exception as exc:  # noqa: BLE001 - 节点内部 bug 不应拖垮整个 run
            wrapped = NodeError(f"节点内部错误：{type(exc).__name__}: {exc}")
            return (edge, node.label, len(preview), time.perf_counter() - t0,
                    wrapped, False, key)

    if dry_run or max_workers <= 1 or len(jobs) <= 1:
        return [work(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
        return list(pool.map(work, jobs))


def run(
    config_path: str | Path,
    *,
    seed_content: Optional[str] = None,
    extra_seeds: Optional[Dict[str, str]] = None,
    db_path: Optional[str | Path] = None,
    agent: Optional[PiAgent] = None,
    graph_llm_client: Optional[Any] = None,
    dry_run: bool = False,
    max_workers: int = 1,
    rpm: Optional[int] = None,
    rate_window: float = 60.0,
    max_seconds: Optional[float] = None,
    limiter: Optional[object] = None,
    use_cache: bool = False,
    cache_dir: Optional[str | Path] = None,
    refresh_cache: bool = False,
    cache_ttl: Optional[float] = None,
    resume: Optional[str] = None,
    resume_last: bool = False,
    keep_going: bool = False,
    adaptive: bool = False,
    max_cost: Optional[float] = None,
    auto_model: bool = False,
) -> RunResult:
    """按依赖顺序调用节点。

    `dry_run=True` 时不执行节点，把每个节点的 preview 文本当作输出写库 ——
    用来验证配置、占位符与图路径，不烧额度。

    `agent` / `graph_llm_client` 是测试注入点：前者替身 pi，后者替身 v7 图的 LLM 客户端。
    `max_workers` 控制同一波次的并发数（默认 1 = 串行）。

    `use_cache=True` 打开节点结果缓存；`refresh_cache=True` 忽略已有条目但仍写入。
    """
    cfg = cfg_mod.load(config_path)
    db = Path(db_path or cfg.db_path or (Path(config_path).resolve().parent.parent / "data" / "dag8.sqlite3"))
    if auto_model:
        from dag8.router import choose_model
        with Store(db) as _route_store:
            best = choose_model(_route_store, min_calls=3)
        if best:
            cfg.pi.provider = best["provider"]
            cfg.pi.model = best["model"]
    workers = max(1, int(max_workers))
    if limiter is None and not dry_run and rpm:
        limiter = shared_limiter(rpm, window=rate_window)
    deadline = (None if max_seconds is None
                else time.monotonic() + float(max_seconds))
    controller = AdaptiveConcurrency(workers) if adaptive else None
    prices = {}
    if max_cost is not None:
        from dag8.cost import load_prices
        prices = load_prices()
    spent = 0.0
    cache: Optional[NodeCache] = (None if (dry_run or not use_cache)
                                  else NodeCache(cache_dir, ttl=cache_ttl))

    seed = seed_content if seed_content is not None else (cfg.seed_content or "")
    if not cfg.seed_tag:
        raise ValueError(f"{config_path} 没有 seed_tag，无法注入种子")

    with Store(db) as store:
        if resume_last and not resume:
            row = store.conn.execute(
                "SELECT run_id FROM runs WHERE config_path=? AND status!='ok' "
                "ORDER BY started_at DESC LIMIT 1",
                (str(Path(config_path).resolve()),),
            ).fetchone()
            if row:
                resume = row["run_id"]
        if resume:
            try:
                store.run_info(resume)
            except KeyError as exc:
                raise ValueError(f"要续跑的 run 不存在：{resume}") from exc
            run_id = resume
            store.resume_run(run_id)
            if not store.contents_for_tag(run_id, cfg.seed_tag):
                store.add_content(run_id, [cfg.seed_tag], seed, gen=0, source_edge=None)
            for tag, text in (extra_seeds or {}).items():
                if not store.contents_for_tag(run_id, tag):
                    store.add_content(run_id, [tag], text, gen=0, source_edge=None)
            latest = store.edge_latest(run_id)
            pending = [e for e in cfg.sorted_edges()
                       if latest.get(e.id, {}).get("status") != "ok"]
            # 清掉未完成边在旧尝试里的记录，避免失败历史混进新一轮
            store.delete_edges(run_id, [e.id for e in pending])
        else:
            run_id = store.new_run(cfg.name, str(Path(config_path).resolve()), cfg.seed_tag, seed)
            store.add_content(run_id, [cfg.seed_tag], seed, gen=0, source_edge=None)
            for tag, text in (extra_seeds or {}).items():
                store.add_content(run_id, [tag], text, gen=0, source_edge=None)
            pending = cfg.sorted_edges()

        result = RunResult(run_id=run_id, dag_name=cfg.name, result=None,
                           result_tag=cfg.result_tag, db_path=str(db))
        had_error = False
        first_error: Optional[str] = None

        if not pending:
            out = store.result(run_id, cfg.result_tag)
            if out is None:
                result.status = "incomplete"
                store.finish_run(run_id, "incomplete", result_tag=cfg.result_tag)
            else:
                result.result = out
                store.finish_run(run_id, "ok", result_tag=cfg.result_tag,
                                 result_content_id=store.contents_for_tag(run_id, cfg.result_tag)[-1]["id"])
            return result
        while pending:
            # 0a) 成本预算；到额就停止，剩余边标 skipped
            if max_cost is not None and spent >= max_cost:
                for edge in pending:
                    rec = EdgeRun(edge.id, edge.layer, "skipped",
                                  error=f"超出成本预算 {max_cost}")
                    result.edges.append(rec)
                    store.log_edge(run_id, edge.id, edge.layer, edge.input_tags,
                                   edge.output_tags, status="skipped", error=rec.error)
                result.status = "over_budget"
                result.cost_estimate = round(spent, 6)
                store.finish_run(run_id, "over_budget",
                                 error=f"超出成本预算 {max_cost}")
                return result

            # 0) 整次运行的墙钟预算；到点就不再调度，剩余边标 skipped
            if deadline is not None and time.monotonic() >= deadline:
                for edge in pending:
                    rec = EdgeRun(edge.id, edge.layer, "skipped",
                                  error=f"超出运行预算 {max_seconds}s")
                    result.edges.append(rec)
                    store.log_edge(run_id, edge.id, edge.layer, edge.input_tags,
                                   edge.output_tags, status="skipped", error=rec.error)
                result.status = "timeout"
                store.finish_run(run_id, "timeout",
                                 error=f"超出运行预算 {max_seconds}s")
                return result

            # 1) 挑出输入已就绪的波次；其余的等下一轮
            ready, blocked, when_skipped, approval_blocked = [], [], [], []
            for edge in pending:
                if edge.join == "any":
                    satisfied = any(store.contents_for_tag(run_id, t)
                                    for t in edge.input_tags)
                else:
                    satisfied = all(
                        store.contents_for_tag(run_id, t) or t in edge.optional_inputs
                        for t in edge.input_tags
                    )
                if not satisfied:
                    blocked.append(edge)
                elif edge.when and not when_matches(edge.when, store, run_id):
                    when_skipped.append(edge)
                elif not dry_run and edge.approval and (
                        reason := approval_block_reason(edge, cfg)):
                    approval_blocked.append((edge, reason))
                else:
                    ready.append(edge)

            for edge in when_skipped:
                rec = EdgeRun(edge.id, edge.layer, "skipped",
                              error="when 条件不满足")
                result.edges.append(rec)
                store.log_edge(run_id, edge.id, edge.layer, edge.input_tags,
                               edge.output_tags, status="skipped", error=rec.error)

            for edge, reason in approval_blocked:
                had_error = True
                first_error = first_error or f"{edge.id}: {reason}"
                rec = EdgeRun(edge.id, edge.layer, "error", error=reason)
                result.edges.append(rec)
                store.log_edge(run_id, edge.id, edge.layer, edge.input_tags,
                               edge.output_tags, status="error", error=reason)

            if not ready:
                if when_skipped or approval_blocked:
                    pending = blocked
                    continue
                # 再也等不到输入 → 剩下的全标 skipped，结束
                for edge in blocked:
                    missing = [t for t in edge.input_tags
                               if not store.contents_for_tag(run_id, t)
                               and t not in edge.optional_inputs]
                    rec = EdgeRun(edge.id, edge.layer, "skipped",
                                  error=f"缺输入 tag: {missing}")
                    result.edges.append(rec)
                    store.log_edge(run_id, edge.id, edge.layer, edge.input_tags,
                                   edge.output_tags, status="skipped", error=rec.error)
                break

            # 2) 主线程：取输入、建节点、查缓存；命中直接算结果
            jobs: List[Job] = []
            outcomes: List[Tuple] = []
            for edge in ready:
                # 可选 tag 缺失时给空列表：占位符渲染成 ""，不阻塞节点
                per_tag = {t: [it["content"] for it in store.contents_for_tag(run_id, t)]
                           for t in edge.input_tags}
                try:
                    node = build_node(edge, cfg, agent=agent,
                                      graph_llm_client=graph_llm_client,
                                      limiter=limiter)
                    preview = node.preview(per_tag)
                    key = (node.cache_key(per_tag)
                           if (cache is not None and edge.cache is not False) else None)
                    hit = None if (cache is None or key is None or refresh_cache) else cache.get(key)
                    if hit is not None:
                        outcomes.append((edge, node.label, len(preview), 0.0, hit, True, key))
                        continue
                    jobs.append((edge, node, per_tag, preview, key))
                except (PiError, NodeError, OSError) as exc:
                    outcomes.append((edge, "", 0, 0.0, exc, False, None))
                except Exception as exc:  # noqa: BLE001
                    outcomes.append((edge, "", 0, 0.0,
                                     NodeError(f"节点构建失败：{type(exc).__name__}: {exc}"),
                                     False, None))

            # 3) 并行执行未命中的节点（线程里只做调用）
            wave_workers = controller.current if controller else workers
            result.workers_used.append(wave_workers)
            outcomes += _run_jobs(jobs, dry_run=dry_run, max_workers=wave_workers,
                                  limiter=limiter, cfg=cfg, agent=agent)
            by_edge = {o[0].id: o for o in outcomes}

            # 4) 主线程统一落库；失败累计（keep_going 时继续跑其它分支）
            for edge in ready:
                _, label, preview_len, elapsed, payload, cached, key = by_edge[edge.id]
                if isinstance(payload, (PiError, NodeError)):
                    had_error = True
                    first_error = first_error or f"{edge.id}: {payload}"
                    rec = EdgeRun(edge.id, edge.layer, "error", elapsed=elapsed,
                                  error=str(payload), prompt_chars=preview_len)
                    result.edges.append(rec)
                    store.log_edge(run_id, edge.id, edge.layer, edge.input_tags,
                                   edge.output_tags, model=label,
                                   prompt_chars=preview_len, status="error",
                                   error=str(payload), elapsed=elapsed)
                    continue

                gen = max((it["gen"] for it in store.all_contents(run_id)), default=0) + 1
                cid = store.add_content(run_id, edge.output_tags, payload, gen=gen,
                                        source_edge=edge.id)
                rec = EdgeRun(edge.id, edge.layer, "ok", elapsed=elapsed,
                              content_id=cid, prompt_chars=preview_len,
                              output_chars=len(payload), cached=cached)
                result.edges.append(rec)
                store.log_edge(run_id, edge.id, edge.layer, edge.input_tags,
                               edge.output_tags, model=label,
                               prompt_chars=preview_len, status="ok",
                               elapsed=elapsed, content_id=cid, cached=cached)
                if (cache is not None and not cached and key is not None
                        and not getattr(node, "used_fallback", False)):
                    cache.put(key, payload, meta={"edge": edge.id, "label": label})

            if max_cost is not None:
                for (_, label, plen, _, payload, cached, _k) in outcomes:
                    if not isinstance(payload, (PiError, NodeError)) and not cached:
                        spent += _edge_cost(label, plen, len(payload), prices)
                result.cost_estimate = round(spent, 6)

            if controller is not None:
                errs = [str(o[4]) for o in outcomes
                        if isinstance(o[4], (PiError, NodeError))]
                succ = sum(1 for o in outcomes
                           if not isinstance(o[4], (PiError, NodeError)))
                controller.observe(errs, succ)

            if had_error and not keep_going:
                result.status = "error"
                store.finish_run(run_id, "error", error=first_error)
                return result

            pending = blocked

        result.result = store.result(run_id, cfg.result_tag)
        if result.result is None:
            status = "error" if had_error else "incomplete"
            store.finish_run(run_id, status, result_tag=cfg.result_tag,
                             error=first_error if had_error else None)
        else:
            status = "partial" if had_error else "ok"
            store.finish_run(run_id, status, result_tag=cfg.result_tag,
                             error=first_error if had_error else None,
                             result_content_id=store.contents_for_tag(run_id, cfg.result_tag)[-1]["id"])
        result.status = status
        return result

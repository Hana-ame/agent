"""v8 节点：调度器只认三种执行体。

| kind      | 执行体 | 说明 |
|---|---|---|
| `pi`      | `PiNode`      | 一次 pi agent 调用（默认；v8 的原生 LLM 节点） |
| `graph`   | `GraphNode`   | 一张 v7 图（实验）。把 v7 DAG 整个当节点跑，取它的 result_tag 输出 |
| `command` | `CommandNode` | 一段确定性 shell 命令。取 URL / 抓取 / 落盘这类不该交给 LLM 的活 |

调度器（`dag8.scheduler`）不关心 prompt/HTTP，只负责按顺序调用节点的
`run(inputs)`。所以新增节点类型 = 新增一个带 `run()` 的类，调度器不用改。

`inputs` 的形态是 `{tag: [内容, ...]}`，和配置里的 `input_tags` 一一对应。
"""
from __future__ import annotations

import functools
import os
import json
import sqlite3
import subprocess
import sys
import time

import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from dag8.backoff import jittered
from dag8.cache import fingerprint
from dag8.config import DAGConfig, EdgeConfig, GraphConfig, PiConfig, render_prompt
from dag8.pi import PiAgent, PiError, PiLLMClient

__all__ = ["NodeError", "Node", "PiNode", "GraphNode", "CommandNode", "build_node"]


class NodeError(RuntimeError):
    """节点执行失败。调度器据此把边标成 error 并停下。"""


class Node(Protocol):
    """节点契约：吃 `{tag: [内容]}`，吐一段文本。"""

    label: str

    def preview(self, inputs: Dict[str, List[str]]) -> str:
        """dry-run 用：不执行，只返回会被送进节点的文本。"""

    def run(self, inputs: Dict[str, List[str]], repair: str = "") -> str:
        ...

    def cache_key(self, inputs: Dict[str, List[str]]) -> str:
        """内容寻址的缓存键：同样的节点 + 同样的输入 + 同样的模型才命中。"""


def _join(inputs: Dict[str, List[str]]) -> str:
    return "\n\n---\n\n".join(c for contents in inputs.values() for c in contents)


class PiNode:
    """一次 pi agent 调用：渲染 prompt → `pi -p` → 文本。"""

    def __init__(
        self,
        prompt: str,
        *,
        pi: PiConfig,
        provider: str,
        model: str,
        timeout: Optional[float] = None,
        tools: Optional[bool] = None,
        agent: Optional[PiAgent] = None,
    ) -> None:
        self.prompt = prompt
        self.pi = pi
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.tools = tools
        self.agent = agent
        self.label = f"{provider}/{model}"
        self.supports_repair = True
        self.uses_api = True
        self.self_limits = False
        self.used_fallback = False

    def preview(self, inputs: Dict[str, List[str]]) -> str:
        return render_prompt(self.prompt, inputs)

    def cache_key(self, inputs: Dict[str, List[str]]) -> str:
        return fingerprint({
            "kind": "pi",
            "provider": self.provider,
            "model": self.model,
            "tools": self.pi.tools if self.tools is None else self.tools,
            "thinking": self.pi.thinking,
            "prompt": self.preview(inputs),
        })

    def run(self, inputs: Dict[str, List[str]], repair: str = "") -> str:
        prompt = self.preview(inputs)
        if repair:
            prompt = f"{prompt}\n\n{repair}"
        runner = self.agent or PiAgent(self.pi)

        specs = [(self.provider, self.model)]
        for raw in self.pi.fallbacks:
            prov, _, mdl = str(raw).partition("/")
            if not mdl:
                prov, mdl = self.provider, raw
            specs.append((prov, mdl))

        last: Optional[PiError] = None
        for i, (provider, model) in enumerate(specs):
            try:
                out = runner.run(
                    prompt,
                    provider=provider,
                    model=model,
                    tools=self.pi.tools if self.tools is None else self.tools,
                    timeout=self.timeout or self.pi.timeout,
                )
                self.label = f"{provider}/{model}"
                self.used_fallback = i > 0
                return out
            except PiError as exc:
                last = exc
                if not exc.retryable or i == len(specs) - 1:
                    raise
        raise last  # pragma: no cover


class V8GraphNode:
    """一张 v8 图作为节点（graph.engine="v8"）：子图内部可波次并行。

    - 输入内容拼成子图种子；子图自己的 seed_tag / result_tag 生效
    - 共享父级限速器（若父级有）；子图用独立临时 db
    - `graph.workers` 控制子图并发，`graph.timeout` 是总预算
    """

    def __init__(self, graph: GraphConfig, *, base_dir: Path,
                 limiter: Optional[Any] = None) -> None:
        self.graph = graph
        self.base_dir = Path(base_dir)
        self.limiter = limiter
        self.label = f"v8graph:{Path(graph.path).name}"
        self.supports_repair = False
        self.uses_api = True
        self.self_limits = True

    def preview(self, inputs: Dict[str, List[str]]) -> str:
        path = self._path(self.graph.path)
        if not path.is_file():
            raise NodeError(f"v8 子图不存在：{path.resolve()}")
        return _join(inputs)

    def _path(self, raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else (self.base_dir / p)

    def cache_key(self, inputs: Dict[str, List[str]]) -> str:
        p = self._path(self.graph.path)
        text = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
        return fingerprint({
            "kind": "v8graph",
            "graph": text,
            "workers": self.graph.workers,
            "seed": self.preview(inputs),
        })

    def run(self, inputs: Dict[str, List[str]], repair: str = "") -> str:
        import shutil
        import tempfile

        from dag8.scheduler import run as schedule_run

        path = self._path(self.graph.path).resolve()
        if not path.is_file():
            raise NodeError(f"v8 子图不存在：{path}")
        seed = self.preview(inputs)
        deadline = time.monotonic() + self.graph.timeout
        tmp = Path(tempfile.mkdtemp(prefix="dag8_nested_"))
        try:
            res = schedule_run(
                path,
                seed_content=seed,
                db_path=tmp / "nested.sqlite3",
                max_workers=max(1, int(self.graph.workers)),
                max_seconds=max(0.1, deadline - time.monotonic()),
                limiter=self.limiter,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        if res.status != "ok":
            raise NodeError(f"v8 子图未成功（{path}）：{res.summary()}")
        if res.result is None:
            raise NodeError(f"v8 子图没有产出 result_tag={res.result_tag!r}")
        return res.result


class DbNode:
    """只读 SQLite 查询节点：SQL + 参数绑定，输出 JSON（columns/rows）。"""

    def __init__(self, db_path: str, query: str, *,
                 params: Optional[List[str]] = None,
                 base_dir: Optional[Path] = None,
                 max_rows: int = 1000, read_only: bool = True,
                 timeout: float = 60.0) -> None:
        self.db = Path(db_path) if Path(db_path).is_absolute() else Path(base_dir or Path.cwd()) / db_path
        self.query = query
        self.params = list(params or [])
        self.max_rows = int(max_rows)
        self.read_only = bool(read_only)
        self.timeout = float(timeout)
        self.label = "db"
        self.supports_repair = False
        self.uses_api = False

    def preview(self, inputs: Dict[str, List[str]]) -> str:
        params = [render_prompt(p, inputs) for p in self.params]
        return f"{self.query}\nparams={params}"

    def cache_key(self, inputs: Dict[str, List[str]]) -> str:
        return fingerprint({
            "kind": "db",
            "path": str(self.db),
            "query": self.query,
            "params": [render_prompt(p, inputs) for p in self.params],
            "max_rows": self.max_rows,
            "read_only": self.read_only,
        })

    def run(self, inputs: Dict[str, List[str]], repair: str = "") -> str:
        if not self.db.is_file():
            raise NodeError(f"db 文件不存在：{self.db}")
        params = [render_prompt(p, inputs) for p in self.params]
        uri = f"file:{self.db}?mode=ro" if self.read_only else str(self.db)
        try:
            conn = sqlite3.connect(uri, uri=self.read_only, timeout=self.timeout)
            cur = conn.execute(self.query, params)
            cols = [d[0] for d in (cur.description or [])]
            rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.Error as exc:
            raise NodeError(f"SQL 执行失败：{exc}") from exc
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        truncated = len(rows) > self.max_rows
        rows = rows[:self.max_rows]
        return json.dumps({"columns": cols,
                           "rows": [list(r) for r in rows],
                           "truncated": truncated},
                          ensure_ascii=False, default=str)


class HttpNode:
    """确定性 HTTP 请求节点：取 API / 网页，不经过模型。"""

    def __init__(self, url: str, *, method: str = "GET",
                 headers: Optional[Dict[str, str]] = None,
                 body: Optional[str] = None,
                 timeout: float = 60.0, max_chars: int = 20000) -> None:
        self.url = url
        self.method = method.upper()
        self.headers = dict(headers or {})
        self.body = body
        self.timeout = float(timeout)
        self.max_chars = int(max_chars)
        self.label = f"http:{self.method}"
        self.supports_repair = False
        self.uses_api = False

    def preview(self, inputs: Dict[str, List[str]]) -> str:
        url = render_prompt(self.url, inputs)
        body = render_prompt(self.body, inputs) if self.body else ""
        return f"{self.method} {url}\n{body}".strip()

    def cache_key(self, inputs: Dict[str, List[str]]) -> str:
        return fingerprint({
            "kind": "http",
            "method": self.method,
            "url": render_prompt(self.url, inputs),
            "headers": {k: render_prompt(str(v), inputs) for k, v in self.headers.items()},
            "body": render_prompt(self.body, inputs) if self.body else "",
            "max_chars": self.max_chars,
        })

    def run(self, inputs: Dict[str, List[str]], repair: str = "") -> str:
        url = render_prompt(self.url, inputs)
        if not url.startswith(("http://", "https://")):
            raise NodeError(f"http 只支持 http(s) URL：{url[:120]}")
        headers = {k: render_prompt(str(v), inputs) for k, v in self.headers.items()}
        body = render_prompt(self.body, inputs) if self.body else None
        try:
            resp = requests.request(self.method, url, headers=headers,
                                    data=body, timeout=self.timeout)
        except requests.RequestException as exc:
            raise NodeError(f"HTTP 请求失败：{type(exc).__name__}: {exc}") from exc
        if resp.status_code >= 400:
            raise NodeError(f"HTTP {resp.status_code}：{resp.text[:500]}")
        return resp.text[:self.max_chars]


class ScriptNode:
    """脚本边：执行一个脚本文件，输入是 JSON（env DAG8_INPUTS + stdin），输出 stdout。"""

    def __init__(self, script: Optional[str] = None, command: Optional[str] = None, *,
                 cwd: Optional[Path] = None, timeout: float = 600.0) -> None:
        self.script = script
        self.command = command
        self.cwd = Path(cwd) if cwd else None
        self.timeout = float(timeout)
        self.label = f"script:{script or (command or '')[:40]}"
        self.supports_repair = False
        self.uses_api = False

    def preview(self, inputs: Dict[str, List[str]]) -> str:
        return self.label

    def cache_key(self, inputs: Dict[str, List[str]]) -> str:
        mtime = 0.0
        if self.script:
            p = Path(self.script) if Path(self.script).is_absolute() else Path(self.cwd or Path.cwd()) / self.script
            if p.is_file():
                mtime = p.stat().st_mtime
        return fingerprint({"kind": "script", "script": self.script,
                            "command": self.command, "mtime": mtime,
                            "inputs": inputs})

    def run(self, inputs: Dict[str, List[str]], repair: str = "") -> str:
        flat = {tag: "\n\n---\n\n".join(vals) for tag, vals in inputs.items()}
        payload = json.dumps(flat, ensure_ascii=False)
        env = dict(os.environ)
        env["DAG8_INPUTS"] = payload
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            if self.script:
                p = Path(self.script)
                if not p.is_absolute():
                    p = Path(self.cwd or Path.cwd()) / p
                if not p.is_file():
                    raise NodeError(f"script 不存在：{p}")
                args = [sys.executable, str(p)] if p.suffix == ".py" else [str(p)]
                proc = subprocess.run(args, input=payload, capture_output=True,
                                      text=True, encoding="utf-8",
                                      errors="replace", timeout=self.timeout,
                                      cwd=str(self.cwd) if self.cwd else None, env=env)
            else:
                proc = subprocess.run(self.command or "", shell=True, input=payload,
                                      capture_output=True, text=True, encoding="utf-8",
                                      errors="replace", timeout=self.timeout,
                                      cwd=str(self.cwd) if self.cwd else None, env=env)
        except subprocess.TimeoutExpired as exc:
            raise NodeError(f"script 超时（{self.timeout:.0f}s）：{self.label}") from exc
        except OSError as exc:
            raise NodeError(f"script 启动失败：{type(exc).__name__}: {exc}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise NodeError(f"script 退出码 {proc.returncode}：{detail[:1000]}")
        return proc.stdout.rstrip("\n")


class CommandNode:
    """一段确定性 shell 命令：把「取 URL / 抓字幕 / 落盘」这类活交给代码，不交给 LLM。

    - `command` 里可以用 prompt 占位符（`{url}` / `{tag:source}` / `{contents}` 等）
    - `stdin=True` 时，把 `{contents}` 渲染后的文本从 stdin 喂给命令（适合长 Markdown，
      避免把整篇正文塞进命令行）
    - 工作目录默认是配置文件所在目录，所以命令里写相对路径（如 `fetch_bili.py`）即可
    """

    def __init__(
        self,
        command: str,
        *,
        cwd: Optional[Path] = None,
        timeout: float = 600.0,
        stdin: bool = False,
    ) -> None:
        self.command = command
        self.cwd = Path(cwd) if cwd else None
        self.timeout = float(timeout)
        self.stdin = bool(stdin)
        self.label = "command"
        self.supports_repair = False
        self.uses_api = False

    def preview(self, inputs: Dict[str, List[str]]) -> str:
        return render_prompt(self.command, inputs)

    def cache_key(self, inputs: Dict[str, List[str]]) -> str:
        return fingerprint({
            "kind": "command",
            "command": self.preview(inputs),
            "stdin": render_prompt("{contents}", inputs) if self.stdin else "",
            "cwd": str(self.cwd or ""),
            "timeout": self.timeout,
        })

    def run(self, inputs: Dict[str, List[str]], repair: str = "") -> str:
        cmd = self.preview(inputs)
        stdin_text = render_prompt("{contents}", inputs) if self.stdin else None
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                input=stdin_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                cwd=str(self.cwd) if self.cwd else None,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise NodeError(f"command 超时（{self.timeout:.0f}s）：{cmd[:200]}") from exc
        except OSError as exc:
            raise NodeError(f"command 启动失败：{type(exc).__name__}: {exc}") from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise NodeError(f"command 退出码 {proc.returncode}：{detail[:1000]}")
        out = (proc.stdout or "").strip()
        if not out:
            detail = (proc.stderr or "").strip()
            raise NodeError(f"command 返回空内容：{detail[:500]}")
        return out


@functools.lru_cache(maxsize=1)
def _v7() -> Dict[str, Any]:
    """按需加载 v7 的 dag 包（只有 graph 节点才需要）。"""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from v7 import DAG
        from v7.llm import OpenAICompatibleClient
        from v7.schema import ModelRegistry
    except ImportError:  # 兼容旧目录名 dag/
        try:
            from dag import DAG
            from dag.llm import OpenAICompatibleClient
            from dag.schema import ModelRegistry
        except ImportError as exc:  # pragma: no cover
            raise NodeError(
                "graph 节点需要 v7/ 或 dag/；请在仓库根目录下运行，"
                "或把仓库根加入 PYTHONPATH"
            ) from exc
    return {"DAG": DAG, "OpenAICompatibleClient": OpenAICompatibleClient,
            "ModelRegistry": ModelRegistry}


class _LimitedClient:
    """把 v7 客户端的每次 complete 都过一遍限速器。"""

    def __init__(self, inner: Any, limiter: Any) -> None:
        self._inner = inner
        self._limiter = limiter

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        self._limiter.acquire()
        return self._inner.complete(*args, **kwargs)

    def embed(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.embed(*args, **kwargs)


_RETRY_SLEEPS: tuple = (10.0, 30.0)
_RETRYABLE = ("429", "Too Many Requests", "rate limit", "timed out",
              "Timeout", "Temporary failure", "Connection")


class GraphNode:
    """一张 v7 图作为一个节点。

    - 节点的输入（按 `input_tags` 收集）拼成种子，注入 v7 图的 `seed_tag`
    - 跑 v7 DAG 直到 `result_tag`，把该内容当作节点输出
    - v7 图里的模型配置：优先图 JSON 自带的 `models`，否则用 `graph.models` 指向的
      models.json；两者都没有时，v7 会落到默认 openai 端点并失败 —— 所以至少给一个
    """

    def __init__(
        self,
        graph: GraphConfig,
        *,
        base_dir: Path,
        llm_client: Optional[Any] = None,
        limiter: Optional[Any] = None,
    ) -> None:
        self.graph = graph
        self.base_dir = Path(base_dir)
        self.llm_client = llm_client
        self.limiter = limiter
        self.label = f"graph:{Path(graph.path).name}"
        self.supports_repair = False
        self.uses_api = True
        self.self_limits = True

    def preview(self, inputs: Dict[str, List[str]]) -> str:
        # dry-run 也验一下图路径，别让"配置看起来没问题"骗过预检
        path = self._path(self.graph.path)
        if not path.is_file():
            raise NodeError(f"v7 图不存在：{path.resolve()}")
        return _join(inputs)

    def _path(self, raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else (self.base_dir / p)

    def cache_key(self, inputs: Dict[str, List[str]]) -> str:
        graph_path = self._path(self.graph.path)
        graph_text = graph_path.read_text(encoding="utf-8", errors="replace") if graph_path.is_file() else ""
        models_text = ""
        if self.graph.models:
            mp = self._path(self.graph.models)
            models_text = mp.read_text(encoding="utf-8", errors="replace") if mp.is_file() else ""
        return fingerprint({
            "kind": "graph",
            "graph": graph_text,
            "models": models_text,
            "seed_tag": self.graph.seed_tag,
            "result_tag": self.graph.result_tag,
            "max_steps": self.graph.max_steps,
            "seed": self.preview(inputs),
        })

    def run(self, inputs: Dict[str, List[str]], repair: str = "") -> str:
        graph_path = self._path(self.graph.path).resolve()
        if not graph_path.is_file():
            raise NodeError(f"v7 图不存在：{graph_path}")

        models_path = self._path(self.graph.models).resolve() if self.graph.models else None
        seed = self.preview(inputs)

        # v7 客户端自身只做 3 次短退避，网关 RPM 紧时仍会 429。
        # 这里在节点层重建子图再试，退避 10s/30s（可被测试 monkeypatch）。
        deadline = time.monotonic() + self.graph.timeout
        last: Optional[Exception] = None
        for attempt in range(len(_RETRY_SLEEPS) + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NodeError(
                    f"graph 节点超时（总预算 {self.graph.timeout:.0f}s，"
                    f"已尝试 {attempt} 次）"
                )
            try:
                out = self._execute(graph_path, models_path, seed, remaining)
            except NodeError:
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
                msg = f"{type(exc).__name__}: {exc}"
                retryable = any(k.lower() in msg.lower() for k in _RETRYABLE)
                if attempt >= len(_RETRY_SLEEPS) or not retryable:
                    raise NodeError(
                        f"v7 图执行失败（{graph_path}）：{msg}"
                    ) from exc
                remaining_after = deadline - time.monotonic()
                if remaining_after <= 0:
                    raise NodeError(
                        f"graph 节点超时（总预算 {self.graph.timeout:.0f}s）"
                    ) from exc
                time.sleep(min(jittered(_RETRY_SLEEPS[attempt], 0, factor=1.0),
                               remaining_after))
                continue
            if time.monotonic() > deadline:
                raise NodeError(
                    f"graph 节点超时（总预算 {self.graph.timeout:.0f}s，"
                    f"单次执行已超时）"
                )
            return out
        raise NodeError(f"v7 图执行失败（{graph_path}）：{last}")  # pragma: no cover

    def _execute(self, graph_path: Path, models_path: Optional[Path], seed: str,
                 remaining: float) -> str:
        mods = _v7()
        per_call = max(30.0, min(remaining, 300.0))
        base_client = (self.llm_client
                       or PiLLMClient(timeout=per_call))
        client = (_LimitedClient(base_client, self.limiter)
                  if self.limiter is not None else base_client)
        d = mods["DAG"](
            name=f"v8:{graph_path.stem}",
            models=str(models_path) if models_path else None,
            llm_client=client,
        )
        d.load(str(graph_path))
        if models_path:  # 显式 graph.models 优先于图 JSON 内联的 models
            d.models = mods["ModelRegistry"](path=str(models_path))

        # v7 的 LLM 边必须有模型配置；空 registry 会落到默认 openai 端点，
        # 报错又远又难懂。这里提前拦掉。
        needs_llm = any(getattr(e.processor, "fn", None) is None for e in d.edges)
        if needs_llm and not d.models:
            raise NodeError(
                f"v7 图 {graph_path} 有 LLM 边但没有模型配置："
                "请在图 JSON 里写 models，或用 graph.models 指向 models.json"
            )

        d.add_content(contents=seed, tags=[self.graph.seed_tag], gen=0)
        d.run(until_tag=self.graph.result_tag, max_steps=self.graph.max_steps)

        out = d.get_result(self.graph.result_tag)
        if out is None:
            raise NodeError(
                f"v7 图 {graph_path} 没有产出 {self.graph.result_tag!r}"
                f"（max_steps={self.graph.max_steps}）"
            )
        return out


def build_node(
    edge: EdgeConfig,
    cfg: DAGConfig,
    *,
    agent: Optional[PiAgent] = None,
    graph_llm_client: Optional[Any] = None,
    limiter: Optional[Any] = None,
) -> Node:
    """把一条 edge 配置变成可执行节点。调度器只调这个工厂，不认识具体类型。"""
    if edge.kind == "pi":
        provider, model = cfg.pi.provider, cfg.pi.model
        if edge.model:
            provider, _, model = edge.model.partition("/")
            if not model:
                provider, model = cfg.pi.provider, edge.model
        return PiNode(
            edge.prompt,
            pi=cfg.pi,
            provider=provider,
            model=model,
            timeout=edge.timeout,
            tools=edge.tools,
            agent=agent,
        )
    if edge.kind == "graph":
        base = Path(cfg.path).resolve().parent if cfg.path else Path.cwd()
        if edge.graph.engine == "v8":
            return V8GraphNode(edge.graph, base_dir=base, limiter=limiter)
        return GraphNode(edge.graph, base_dir=base, llm_client=graph_llm_client,
                         limiter=limiter)
    if edge.kind == "db":
        base = Path(cfg.path).resolve().parent if cfg.path else Path.cwd()
        return DbNode(edge.db_path or "", edge.query or "",
                      params=edge.params, base_dir=base,
                      max_rows=edge.max_rows, read_only=edge.read_only,
                      timeout=edge.timeout or 60.0)
    if edge.kind == "http":
        return HttpNode(edge.url or "", method=edge.method, headers=edge.headers,
                        body=edge.body, timeout=edge.timeout or 60.0,
                        max_chars=edge.max_chars)
    if edge.kind == "script":
        base = Path(cfg.path).resolve().parent if cfg.path else Path.cwd()
        return ScriptNode(edge.script, edge.command, cwd=base,
                          timeout=edge.timeout or 600.0)
    if edge.kind == "command":
        base = Path(cfg.path).resolve().parent if cfg.path else Path.cwd()
        return CommandNode(
            edge.command or "",
            cwd=base,
            timeout=edge.timeout or 600.0,
            stdin=edge.stdin,
        )
    raise NodeError(f"未知节点类型：{edge.kind!r}")

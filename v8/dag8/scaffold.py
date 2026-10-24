"""从项目结构生成 v8 图：每个模块一个节点，按目录树分解。

思路：单一 agent 每次都要「重新调查」整个项目；把每个模块变成一个**可独立重跑**
的节点，调查结果互不影响，图的结构直接跟随代码结构：

    repo (seed)
      ├─ inspect_dag8    → mod_dag8    (command 节点，并行)
      ├─ inspect_scripts → mod_scripts
      ├─ inspect_tests   → mod_tests
      └─ report          → report      (pi 节点，汇总)

- 模块节点是 `command` 节点：跑 `scripts/inspect_module.py`，输出 JSON 指纹
- `cache: false`：扫描当前文件是易变的，不能吃缓存
- 模块节点带 `expect`：输出必须是含 module/loc/hash 的 JSON
- 汇总节点是 pi 节点：把各模块指纹交给模型写架构报告

这样改一个模块只需要重跑它自己；并行度 = 模块数；每个模块的结果都能单独查。
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

V8 = Path(__file__).resolve().parent.parent
DEFAULT_INSPECTOR = V8 / "scripts" / "inspect_module.py"

IGNORE = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    "data", "out", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}

REVIEW_PROMPT = (
    "你是严格的代码评审。下面是各模块的 diff（可能已截断）：\n\n{contents}\n\n"
    "请输出中文评审：\n"
    "1) 每个模块最关键的问题（正确性/边界/性能），每条一句话\n"
    "2) 跨模块影响\n"
    "3) 建议优先修的 1~3 项\n"
    "只输出评审正文，不要重复 diff。"
)

REPORT_PROMPT = (
    "你是架构分析助手。下面是各模块的结构指纹（JSON，含 LOC、文件数、定义数、导入）：\n\n"
    "{contents}\n\n"
    "请输出一份中文架构报告：\n"
    "1) 模块清单与体量（LOC / 文件数）\n"
    "2) 依赖方向与耦合点（根据 imports）\n"
    "3) 风险与建议\n"
    "只输出报告正文，不要开场白。"
)


def scan_modules(root: str | Path, *, depth: int = 1, mode: str = "package") -> List[Path]:
    """扫描模块目录。

    - `mode="package"`：只收含 `__init__.py` 的目录；一个都没有时退回 `dir`
    - `mode="dir"`：收顶层目录（跳过隐藏/.git/data 等）
    - `depth`：相对 root 的最大深度（1 = 顶层）
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"不是目录：{root}")

    if mode not in ("package", "dir"):
        raise ValueError(f"mode 只支持 package/dir，收到 {mode!r}")

    found: List[Path] = []
    if mode == "package":
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in IGNORE or child.name.startswith("."):
                continue
            _collect_packages(child, depth - 1, found)
        if not found:
            mode = "dir"

    if mode == "dir":
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name not in IGNORE and not child.name.startswith("."):
                found.append(child)

    return [p for i, p in enumerate(found) if p not in found[:i]]


def changed_modules(root: str | Path) -> List[Path]:
    """只返回本轮 git 变更（含未跟踪）涉及的顶层模块。

    适合「改哪块跑哪块」：不必每次全量调查。按路径第一段映射到顶层目录。
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"不是目录：{root}")

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(root), *args],
                              capture_output=True, text=True)

    if git("rev-parse", "--git-dir").returncode != 0:
        raise ValueError(f"不是 git 仓库：{root}")

    paths: set[str] = set()
    diff = git("diff", "--name-only", "HEAD")
    if diff.returncode == 0:
        paths.update(x for x in diff.stdout.splitlines() if x.strip())
    untracked = git("ls-files", "--others", "--exclude-standard")
    if untracked.returncode == 0:
        paths.update(x for x in untracked.stdout.splitlines() if x.strip())

    modules: List[Path] = []
    for raw in paths:
        first = Path(raw).parts[0]
        d = root / first
        if d.is_dir() and d.name not in IGNORE and not d.name.startswith("."):
            if d not in modules:
                modules.append(d)
    return sorted(modules)


def _collect_packages(d: Path, budget: int, out: List[Path]) -> None:
    if budget < 0:
        return
    if (d / "__init__.py").exists():
        out.append(d)
    if budget == 0:
        return
    for child in sorted(d.iterdir()):
        if child.is_dir() and child.name not in IGNORE and not child.name.startswith("."):
            _collect_packages(child, budget - 1, out)


def _slug(root: Path, module: Path) -> str:
    rel = module.relative_to(root).as_posix().replace("/", "_")
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", rel)[:40].strip("_")
    return slug or "root"


def build_config(
    root: str | Path,
    modules: List[Path],
    *,
    inspector: str | Path | None = None,
    provider: str = "laptop",
    model: str = "deepseek-v4-flash",
    max_modules: int = 12,
) -> Dict[str, Any]:
    root = Path(root).resolve()
    inspector = Path(inspector or DEFAULT_INSPECTOR).resolve()
    modules = list(modules)[:max_modules]
    if not modules:
        raise ValueError("没有扫描到任何模块")

    edges: List[Dict[str, Any]] = []
    tags: List[str] = []
    for module in modules:
        slug = _slug(root, module)
        tag = f"mod_{slug}"
        tags.append(tag)
        edges.append({
            "id": f"inspect_{slug}",
            "kind": "command",
            "layer": 0,
            "input_tags": ["repo"],
            "output_tags": [tag],
            "command": f'python3 "{inspector}" "{module}"',
            "expect": {"type": "json", "required_keys": ["module", "loc", "hash"]},
            "cache": False,   # 扫描当前文件，易变，禁用缓存
        })

    edges.append({
        "id": "report",
        "layer": 1,
        "input_tags": tags,
        "output_tags": ["report"],
        "prompt": REPORT_PROMPT,
    })

    return {
        "name": f"{root.name}_arch",
        "_comment": "由 dag8 scaffold 生成：每个模块一个 command 节点（并行），最后一个 pi 节点汇总。",
        "seed_tag": "repo",
        "seed_content": str(root),
        "result_tag": "report",
        "pi": {
            "provider": provider,
            "model": model,
            "tools": False,
            "timeout": 240,
            "retries": 3,
        },
        "edges": edges,
    }


def build_review_config(
    root: str | Path,
    modules: List[Path],
    *,
    provider: str = "laptop",
    model: str = "deepseek-v4-flash",
    max_modules: int = 8,
    max_diff_chars: int = 20000,
) -> Dict[str, Any]:
    """每个模块一个 git diff command 节点，最后一个 pi 节点汇总评审。"""
    root = Path(root).resolve()
    modules = list(modules)[:max_modules]
    if not modules:
        raise ValueError("没有扫描到任何模块")

    edges: List[Dict[str, Any]] = []
    tags: List[str] = []
    for module in modules:
        rel = module.relative_to(root).as_posix()
        slug = _slug(root, module)
        tag = f"diff_{slug}"
        tags.append(tag)
        edges.append({
            "id": f"diff_{slug}",
            "kind": "command",
            "layer": 0,
            "input_tags": ["repo"],
            "output_tags": [tag],
            "command": (f'git -C "{{tag:repo}}" diff --unified=3 -- "{rel}" '
                        f'| head -c {int(max_diff_chars)}'),
            "cache": False,
        })

    edges.append({
        "id": "review",
        "layer": 1,
        "input_tags": tags,
        "output_tags": ["review"],
        "prompt": REVIEW_PROMPT,
        "expect": {"min_chars": 20},
    })

    return {
        "name": f"{root.name}_review",
        "_comment": "由 dag8 scaffold --review 生成：每模块一个 git diff 节点，汇总成评审。",
        "seed_tag": "repo",
        "seed_content": str(root),
        "result_tag": "review",
        "pi": {
            "provider": provider,
            "model": model,
            "tools": False,
            "timeout": 240,
            "retries": 3,
        },
        "edges": edges,
    }


def build_composed_config(
    root: str | Path,
    modules: List[Path],
    *,
    provider: str = "laptop",
    model: str = "deepseek-v4-flash",
    subdir: str = "graphs",
    max_modules: int = 8,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """每模块一张 v8 子图，父图用 graph(engine="v8") 并行组合。

    - 子图 = build_config(单模块)：inspect command + pi 报告
    - 父图：repo → 每个模块子图（并行）→ 一个 pi 汇总报告
    - 子图文件写到父配置同级的 `subdir/`
    """
    root = Path(root).resolve()
    modules = list(modules)[:max_modules]
    if not modules:
        raise ValueError("没有扫描到任何模块")

    subgraphs: Dict[str, Dict[str, Any]] = {}
    graph_edges: List[Dict[str, Any]] = []
    tags: List[str] = []
    for module in modules:
        slug = _slug(root, module)
        name = f"{slug}.json"
        sub = build_config(root, [module], provider=provider, model=model,
                           max_modules=1)
        sub["name"] = f"{slug}_module"
        subgraphs[name] = sub

        tag = f"mod_{slug}"
        tags.append(tag)
        graph_edges.append({
            "id": f"g_{slug}",
            "kind": "graph",
            "layer": 0,
            "input_tags": ["repo"],
            "output_tags": [tag],
            "graph": {
                "path": f"{subdir}/{name}",
                "engine": "v8",
                "workers": 1,
                "timeout": 180,
            },
        })

    graph_edges.append({
        "id": "report",
        "layer": 1,
        "input_tags": tags,
        "output_tags": ["report"],
        "prompt": REPORT_PROMPT,
        "expect": {"min_chars": 20},
    })

    parent = {
        "name": f"{root.name}_composed",
        "_comment": "由 dag8 scaffold --compose 生成：每模块一张 v8 子图，父图并行组合。",
        "seed_tag": "repo",
        "seed_content": str(root),
        "result_tag": "report",
        "pi": {
            "provider": provider,
            "model": model,
            "tools": False,
            "timeout": 300,
            "retries": 3,
        },
        "edges": graph_edges,
    }
    return parent, subgraphs


def build_composed_review_config(
    root: str | Path,
    modules: List[Path],
    *,
    provider: str = "laptop",
    model: str = "deepseek-v4-flash",
    subdir: str = "graphs",
    max_modules: int = 4,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """每模块一张「diff + 评审」v8 子图，父图并行组合后汇总。"""
    root = Path(root).resolve()
    modules = list(modules)[:max_modules]
    if not modules:
        raise ValueError("没有扫描到任何模块")

    subgraphs: Dict[str, Dict[str, Any]] = {}
    graph_edges: List[Dict[str, Any]] = []
    tags: List[str] = []
    for module in modules:
        slug = _slug(root, module)
        name = f"{slug}.json"
        sub = build_review_config(root, [module], provider=provider, model=model,
                                  max_modules=1)
        sub["name"] = f"{slug}_review"
        subgraphs[name] = sub

        tag = f"review_{slug}"
        tags.append(tag)
        graph_edges.append({
            "id": f"g_{slug}",
            "kind": "graph",
            "layer": 0,
            "input_tags": ["repo"],
            "output_tags": [tag],
            "graph": {
                "path": f"{subdir}/{name}",
                "engine": "v8",
                "workers": 1,
                "timeout": 240,
            },
        })

    graph_edges.append({
        "id": "summary",
        "layer": 1,
        "input_tags": tags,
        "output_tags": ["review"],
        "prompt": (
            "下面是各模块的评审。请汇总成一份整体评审：\n"
            "1) 跨模块问题与重复\n2) 优先级排序\n3) 建议先修的 1~3 项\n"
            "只输出汇总正文。\n\n{contents}"
        ),
        "expect": {"min_chars": 20},
    })

    parent = {
        "name": f"{root.name}_review_composed",
        "_comment": "由 dag8 scaffold --compose --review 生成：每模块评审子图并行，父图汇总。",
        "seed_tag": "repo",
        "seed_content": str(root),
        "result_tag": "review",
        "pi": {
            "provider": provider,
            "model": model,
            "tools": False,
            "timeout": 300,
            "retries": 3,
        },
        "edges": graph_edges,
    }
    return parent, subgraphs


def write_config(config: Dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p

"""Genome：一次候选的完整可复现描述。

一个 Genome = 一份 v8 `dag.json` + 运行参数（workers）+ 变异谱系。
候选只改 DAG 配置，不改 `evolve/` 与 `dag8/` 代码。
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

IGNORE = shutil.ignore_patterns(
    "cache", "out", "data", "__pycache__", ".git", "*.sqlite3", "*.sqlite3-*"
)

STYLE_SUFFIXES = [
    "\n\n【候选风格】定稿更克制：少用形容词，多给因果链；每个判断后面跟一句限制条件。",
    "\n\n【候选风格】定稿更像现场笔记：保留少量口语引号，但每个口语判断后补一句书面解释。",
    "\n\n【候选风格】定稿更偏管理者视角：每节末尾加一句“这对管理者意味着什么”。",
    "\n\n【候选风格】定稿更短：优先删抽象议论，保留案例、数字、勘误表；不要写结语。",
]

REQUIREMENT_SUFFIXES = [
    "\n\n【候选要求】第二章第 1 条必须出现贾谊《过秦论》与三苏/苏洵，不能省略。",
    "\n\n【候选要求】3.2 必须恰好五阶段，且与第二章内容不重复。",
    "\n\n【候选要求】3.4 必须是三列八行，原文识别词列逐条覆盖指定词。",
]


@dataclass
class Genome:
    id: str
    name: str
    config: Dict[str, Any]
    config_path: str
    workers: int = 1
    lineage: Dict[str, Any] = field(default_factory=dict)

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def from_config(cls, path: str | Path, *, name: str = "base",
                    workers: int = 1) -> "Genome":
        p = Path(path).resolve()
        config = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            id=_fingerprint(config, workers),
            name=name,
            config=config,
            config_path=str(p),
            workers=int(workers),
            lineage={"parent": None, "op": "seed", "detail": ""},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "config": self.config,
            "config_path": self.config_path,
            "workers": self.workers,
            "lineage": self.lineage,
        }

    # -- 运行参数 / 摘要 ----------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        edges = self.config.get("edges", [])
        kinds: Dict[str, int] = {}
        for e in edges:
            kinds[e.get("kind", "pi")] = kinds.get(e.get("kind", "pi"), 0) + 1
        prompt_chars = sum(len(e.get("prompt", "")) for e in edges)
        models = sorted({e.get("model", self.config.get("pi", {}).get("model", "?"))
                         for e in edges if e.get("kind", "pi") == "pi"})
        return {
            "name": self.name,
            "workers": self.workers,
            "edges": len(edges),
            "kinds": kinds,
            "prompt_chars": prompt_chars,
            "models": models,
            "lineage": self.lineage,
        }

    # -- 变异 ---------------------------------------------------------------
    def mutate(self, rng: random.Random, allowed: Optional[List[str]] = None) -> "Genome":
        ops = allowed or ["prompt_style", "prompt_requirement", "thinking",
                          "contract", "critique_node", "workers"]
        op = rng.choice(ops)
        child = Genome(
            id=self.id,
            name=self.name,
            config=copy.deepcopy(self.config),
            config_path=self.config_path,
            workers=self.workers,
            lineage={"parent": self.id, "op": op, "detail": ""},
        )
        detail = ""
        if op == "prompt_style":
            detail = _append_to_final(child.config, rng.choice(STYLE_SUFFIXES))
        elif op == "prompt_requirement":
            detail = _append_to_final(child.config, rng.choice(REQUIREMENT_SUFFIXES))
        elif op == "thinking":
            pi = child.config.setdefault("pi", {})
            pi["thinking"] = None if pi.get("thinking") == "off" else "off"
            detail = f"thinking={pi.get('thinking')}"
        elif op == "contract":
            detail = _tighten_contract(child.config)
        elif op == "critique_node":
            detail = _add_critique_node(child.config)
        elif op == "workers":
            child.workers = 2 if child.workers == 1 else 1
            detail = f"workers={child.workers}"

        if not detail:
            # 变异必须至少改变一点东西；退化为 workers / thinking。
            child.workers = 2 if child.workers == 1 else 1
            detail = f"fallback workers={child.workers}"
        child.lineage["detail"] = detail
        child.name = f"{self.name}-{op}"
        child.id = _fingerprint(child.config, child.workers)
        return child

    def replicate(self, index: int, total: int) -> "Genome":
        """同一个配置的第 index 次重复运行（用于测随机波动，不改变 genome）。"""
        return Genome(
            id=f"{self.id}-r{index}",
            name=f"{self.name}#r{index}",
            config=copy.deepcopy(self.config),
            config_path=self.config_path,
            workers=self.workers,
            lineage={**self.lineage, "op": "repeat",
                     "detail": f"repeat {index}/{total} of {self.id}"},
        )

    # -- 落盘 ---------------------------------------------------------------
    def materialize(self, dest_dir: str | Path) -> Path:
        """把候选配置及其同目录脚本复制到独立目录，返回 dag.json 路径。

        复制整个配置目录是为了让 `command` 节点的相对路径（如
        `python3 trim_md.py`）在候选目录里仍然成立。
        """
        dest = Path(dest_dir)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        src_dir = Path(self.config_path).resolve().parent
        cfg_dir = dest / "config"
        # 先复制到系统临时目录，再 move 进 dest：避免 dest 恰好在 src_dir 内部时
        # copytree 递归复制自己（测试里很常见）。
        with tempfile.TemporaryDirectory(prefix="evolve-genome-") as td:
            tmp_cfg = Path(td) / "config"
            shutil.copytree(src_dir, tmp_cfg, ignore=IGNORE)
            shutil.move(str(tmp_cfg), str(cfg_dir))
        out = cfg_dir / "dag.json"
        out.write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8")
        return out


def _fingerprint(config: Dict[str, Any], workers: int) -> str:
    raw = json.dumps({"config": config, "workers": workers},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _pi_edges(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [e for e in config.get("edges", [])
            if e.get("kind", "pi") == "pi" and "prompt" in e]


def _append_to_final(config: Dict[str, Any], suffix: str) -> str:
    edges = _pi_edges(config)
    if not edges:
        return ""
    target = next((e for e in edges if "final" in e.get("output_tags", [])), edges[-1])
    target["prompt"] = str(target.get("prompt", "")) + suffix
    return f"append to {target.get('id')} (+{len(suffix)} chars)"


def _tighten_contract(config: Dict[str, Any]) -> str:
    edges = _pi_edges(config)
    if not edges:
        return ""
    target = next((e for e in edges if "final" in e.get("output_tags", [])), edges[-1])
    expect = dict(target.get("expect") or {})
    expect.setdefault("min_chars", 4000)
    expect.setdefault("max_chars", 12000)
    expect.setdefault("min_sections", 9)
    expect.setdefault("pattern", "```mermaid")
    target["expect"] = expect
    target["repair"] = max(1, int(target.get("repair", 0)))
    return f"expect on {target.get('id')}"


def _add_critique_node(config: Dict[str, Any]) -> str:
    """在 final 前加一条 self_critique 边，并把它接进 final 的输入。"""
    edges = config.get("edges", [])
    final = next((e for e in edges if "final" in e.get("output_tags", [])), None)
    if final is None or "draft" not in final.get("input_tags", []):
        return ""
    if any("critique" in e.get("output_tags", []) for e in edges):
        return ""
    layer = int(final.get("layer", 0))
    crit = {
        "id": "self_critique",
        "kind": "pi",
        "layer": layer,
        "input_tags": ["draft"],
        "output_tags": ["critique"],
        "model": final.get("model") or config.get("pi", {}).get("model"),
        "prompt": (
            "审校下面的初稿。只列 5~8 条最影响质量的问题（事实/结构/覆盖），"
            "每条一句话，不要改写初稿。\n\n{tag:draft}"
        ),
        "expect": {"min_chars": 30, "max_chars": 3000},
        "repair": 1,
    }
    final.setdefault("input_tags", []).append("critique")
    final["prompt"] = str(final.get("prompt", "")) + "\n\n【自审意见】\n{tag:critique}"
    final["layer"] = layer + 1
    edges.append(crit)
    return "add self_critique → final"

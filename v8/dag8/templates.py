"""v8 任务模板：把反复出现的任务做成可复用配置。

单一 agent 每次都要重新想“怎么拆、prompt 怎么写、输出什么格式”；
模板把已验证的结构固化下来，`dag8 new <template>` 直接生成一张图：

| 模板 | 输入 | 输出 | 特点 |
|---|---|---|---|
| `summarize` | 长文 | 3~5 条要点 | 契约：最少 50 字 |
| `extract_json` | 文本 | JSON | 契约：JSON + 必需键；失败回灌重试 |
| `classify` | 文本 | JSON(label/confidence) | 同上，适合分流 |
| `translate` | 文本 | 译文 | 只输出译文 |
| `review_diff` | 仓库路径 | 代码评审 | command 节点先跑 `git diff`，再交给 pi |

模板只是起点：生成的 JSON 可以继续改 prompt、加边、换成 graph 节点。
"""
from __future__ import annotations

from typing import Any, Dict

__all__ = ["TEMPLATES", "build", "list_templates"]

PI = {
    "provider": "laptop",
    "model": "deepseek-v4-flash",
    "tools": False,
    "timeout": 240,
    "retries": 3,
}


def _pi(provider: str, model: str) -> Dict[str, Any]:
    return {**PI, "provider": provider, "model": model}


def _summarize(provider: str, model: str) -> Dict[str, Any]:
    return {
        "name": "summarize_template",
        "_template": "summarize：长文 → 3~5 条要点",
        "seed_tag": "doc",
        "seed_content": "（把你的长文放在这里）",
        "result_tag": "summary",
        "pi": _pi(provider, model),
        "edges": [{
            "id": "summarize",
            "layer": 0,
            "input_tags": ["doc"],
            "output_tags": ["summary"],
            "prompt": "把下面的内容压缩成 3~5 条要点，每条一行，只基于原文：\n\n{contents}",
            "expect": {"min_chars": 50},
            "repair": 1,
        }],
    }


def _extract_json(provider: str, model: str) -> Dict[str, Any]:
    return {
        "name": "extract_json_template",
        "_template": "extract_json：文本 → JSON（entities/count）",
        "seed_tag": "text",
        "seed_content": "（把待抽取文本放在这里）",
        "result_tag": "data",
        "pi": _pi(provider, model),
        "edges": [{
            "id": "extract",
            "layer": 0,
            "input_tags": ["text"],
            "output_tags": ["data"],
            "prompt": (
                "从下面的文本抽取实体，只输出 JSON：\n"
                '{"entities": ["..."], "count": 0}\n\n'
                "文本：\n{contents}"
            ),
            "expect": {"type": "json", "required_keys": ["entities", "count"]},
            "repair": 1,
        }],
    }


def _classify(provider: str, model: str) -> Dict[str, Any]:
    return {
        "name": "classify_template",
        "_template": "classify：文本 → JSON(label/confidence)",
        "seed_tag": "text",
        "seed_content": "（把待分类文本放在这里）",
        "result_tag": "label",
        "pi": _pi(provider, model),
        "edges": [{
            "id": "classify",
            "layer": 0,
            "input_tags": ["text"],
            "output_tags": ["label"],
            "prompt": (
                "把下面的文本分到一类（bug / feature / question / other）。"
                '只输出 JSON：{"label": "...", "confidence": 0.0}\n\n'
                "文本：\n{contents}"
            ),
            "expect": {"type": "json", "required_keys": ["label", "confidence"]},
            "repair": 1,
        }],
    }


def _translate(provider: str, model: str) -> Dict[str, Any]:
    return {
        "name": "translate_template",
        "_template": "translate：文本 → 英文译文",
        "seed_tag": "text",
        "seed_content": "（把待翻译文本放在这里）",
        "result_tag": "translation",
        "pi": _pi(provider, model),
        "edges": [{
            "id": "translate",
            "layer": 0,
            "input_tags": ["text"],
            "output_tags": ["translation"],
            "prompt": "把下面的内容翻译成英文，只输出译文，不要解释：\n\n{contents}",
            "expect": {"min_chars": 10},
        }],
    }


def _review_diff(provider: str, model: str) -> Dict[str, Any]:
    return {
        "name": "review_diff_template",
        "_template": "review_diff：仓库路径 → git diff → 代码评审",
        "seed_tag": "repo",
        "seed_content": ".",
        "result_tag": "review",
        "pi": _pi(provider, model),
        "edges": [
            {
                "id": "diff",
                "kind": "command",
                "layer": 0,
                "input_tags": ["repo"],
                "output_tags": ["diff"],
                "command": "git -C {tag:repo} diff --unified=3",
                "expect": {"min_chars": 10},
                "cache": False,
            },
            {
                "id": "review",
                "layer": 1,
                "input_tags": ["diff"],
                "output_tags": ["review"],
                "prompt": (
                    "你是严格的代码评审。根据下面的 diff 只指出最关键的 1~3 个问题"
                    "（正确性、边界、性能），每条一句话，不要改写代码：\n\n{contents}"
                ),
                "expect": {"min_chars": 20},
            },
        ],
    }


def _bug_triage(provider: str, model: str) -> Dict[str, Any]:
    return {
        "name": "bug_triage_template",
        "_template": "bug_triage：缺陷描述 → JSON(severity/component/summary/next_step)",
        "seed_tag": "report",
        "seed_content": "（把缺陷/报错/复现步骤放在这里）",
        "result_tag": "triage",
        "pi": _pi(provider, model),
        "edges": [{
            "id": "triage",
            "layer": 0,
            "input_tags": ["report"],
            "output_tags": ["triage"],
            "prompt": (
                "对下面的缺陷做分诊，只输出 JSON：\n"
                '{"severity": "low|medium|high", "component": "…", '
                '"summary": "…", "next_step": "…"}\n\n'
                "缺陷：\n{contents}"
            ),
            "expect": {
                "schema": {
                    "type": "object",
                    "required": ["severity", "component", "summary", "next_step"],
                    "properties": {
                        "severity": {"enum": ["low", "medium", "high"]},
                        "component": {"type": "string"},
                        "summary": {"type": "string"},
                        "next_step": {"type": "string"},
                    },
                    "additionalProperties": False,
                }
            },
            "repair": 1,
        }],
    }


def _test_plan(provider: str, model: str) -> Dict[str, Any]:
    return {
        "name": "test_plan_template",
        "_template": "test_plan：需求/代码 → 测试计划（Markdown）",
        "seed_tag": "spec",
        "seed_content": "（把需求或代码放在这里）",
        "result_tag": "test_plan",
        "pi": _pi(provider, model),
        "edges": [{
            "id": "plan",
            "layer": 0,
            "input_tags": ["spec"],
            "output_tags": ["test_plan"],
            "prompt": (
                "为下面的需求写一份测试计划，用 Markdown，至少包含三个二级标题："
                "范围、关键用例、风险与边界。只输出计划正文：\n\n{contents}"
            ),
            "expect": {"min_sections": 3, "min_chars": 80},
            "repair": 1,
        }],
    }


def _refactor_plan(provider: str, model: str) -> Dict[str, Any]:
    return {
        "name": "refactor_plan_template",
        "_template": "refactor_plan：代码 → 重构计划（Markdown）",
        "seed_tag": "code",
        "seed_content": "（把待重构代码放在这里）",
        "result_tag": "refactor_plan",
        "pi": _pi(provider, model),
        "edges": [{
            "id": "plan",
            "layer": 0,
            "input_tags": ["code"],
            "output_tags": ["refactor_plan"],
            "prompt": (
                "为下面的代码写重构计划，用 Markdown，至少包含：问题、目标结构、"
                "迁移步骤、风险。不要直接改写代码，只给计划：\n\n{contents}"
            ),
            "expect": {"min_sections": 3, "min_chars": 100},
            "repair": 1,
        }],
    }


TEMPLATES = {
    "summarize": _summarize,
    "bug_triage": _bug_triage,
    "test_plan": _test_plan,
    "refactor_plan": _refactor_plan,
    "extract_json": _extract_json,
    "classify": _classify,
    "translate": _translate,
    "review_diff": _review_diff,
}


def list_templates() -> Dict[str, str]:
    return {name: fn("laptop", "deepseek-v4-flash")["_template"] for name, fn in TEMPLATES.items()}


def build(name: str, *, provider: str = "laptop", model: str = "deepseek-v4-flash",
          seed: str | None = None, seed_tag: str | None = None) -> Dict[str, Any]:
    if name not in TEMPLATES:
        raise ValueError(f"未知模板：{name!r}；可选 {sorted(TEMPLATES)}")
    cfg = TEMPLATES[name](provider, model)
    if seed is not None:
        cfg["seed_content"] = seed
    if seed_tag is not None:
        cfg["seed_tag"] = seed_tag
    return cfg

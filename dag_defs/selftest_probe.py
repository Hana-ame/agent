"""评测器的标定桩：输出完全可控，用来验证 dag-eval 的指标本身算得对。

两个桩都不联网、不调模型，CI 里也能跑：

| DAG | 行为 | 期望判定 |
|---|---|---|
| `selftest-stable` | 每次输出逐字节相同 | **PASS**（稳定性 1.000、正确率 100%） |
| `selftest-noisy`  | 每次多出几行不同内容 | **WARN**（Jaccard / 长度 CV 掉下来） |

它们不是业务 DAG，只用来回答一个问题：**当输入完美时，评测器会不会给满分**。
如果 `selftest-stable` 都判不出 PASS，说明评测器坏了，别拿它去评真 DAG。

用法：

```bash
python -m dags run dag-eval --seed '{"target": "selftest-stable", "runs": 3}'
python -m dags run dag-eval --seed '{"target": "selftest-noisy",  "runs": 3}'
```
"""
from __future__ import annotations

import json
from typing import Any

from dag_agent import DAG
from dags.registry import register, timed

__all__ = ["build_selftest_stable", "build_selftest_noisy", "verify_report"]

_STABLE_BODY = (
    "## 摘要\n"
    "本次巡检覆盖 5 个主题，全部通过结构校验，无异常项。\n"
    "响应时间稳定在预期区间，未发现抖动或截断。\n"
    "\n"
    "## 主题索引\n"
    "- [主题 A](http://example.invalid/a)\n"
    "- [主题 B](http://example.invalid/b)\n"
    "- [主题 C](http://example.invalid/c)\n"
)

# 噪声桩的调用计数：用来让每次输出长度不同（模拟真实模型输出的抖动）
_NOISY_RUN = {"n": 0}


def verify_report(text: str) -> tuple[bool, str]:
    """评测器会调用这个判语义正确率。桩的判据就是「结构三段齐全且不截断」。"""
    text = text or ""
    if "## 摘要" not in text:
        return False, "缺摘要段"
    if "## 主题索引" not in text:
        return False, "缺索引段"
    if not text.rstrip().endswith(("。", ")", "）")):
        return False, "结尾不完整"
    items = text.count("\n- [")
    if items < 3:
        return False, f"索引只有 {items} 条"
    return True, f"结构完整，索引 {items} 条"


@timed
def _stable_generate(payload: Any) -> str:
    return _STABLE_BODY


@timed
def _noisy_generate(payload: Any) -> str:
    _NOISY_RUN["n"] += 1
    n = _NOISY_RUN["n"]
    lines = [
        "## 摘要",
        f"第 {n} 次巡检：覆盖 {n * 4} 个主题，发现 {n} 处需要关注的变化。",
        "其余指标在预期区间内。",
        "",
        "## 主题索引",
    ]
    lines += [f"- [主题 {n}-{i}](http://example.invalid/{n}/{i})" for i in range(1, 3 + n)]
    return "\n".join(lines) + "\n"


@register(
    "selftest-stable",
    "评测标定桩（确定性）：每次输出完全相同，dag-eval 应判 PASS",
    seed_tag="probe_seed",
    seed_content="stable",
    final_tag="probe_report",
    default_output="dag_defs/out/selftest_stable.md",
)
def build_selftest_stable() -> DAG:
    d = DAG(name="selftest_stable")
    d.add_edge(
        input_tags=["probe_seed"],
        processor=_stable_generate,
        output_tags=["probe_report"],
        layer=0,
    )
    return d


@register(
    "selftest-noisy",
    "评测标定桩（抖动）：每次输出长度递增，dag-eval 应判 WARN",
    seed_tag="probe_seed",
    seed_content="noisy",
    final_tag="probe_report",
    default_output="dag_defs/out/selftest_noisy.md",
)
def build_selftest_noisy() -> DAG:
    d = DAG(name="selftest_noisy")
    d.add_edge(
        input_tags=["probe_seed"],
        processor=_noisy_generate,
        output_tags=["probe_report"],
        layer=0,
    )
    return d

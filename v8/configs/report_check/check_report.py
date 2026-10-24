#!/usr/bin/env python3
"""确定性汇报体检：从 stdin 读 Markdown，输出 JSON 指标。

只做可机器判断的事（章节、数字、长句、术语、模糊词），
语义质量交给 DAG 里的三个评审节点。
"""
from __future__ import annotations

import json
import re
import sys

CONCLUSION = ("结论", "摘要", "总结", "先说结论", "核心结论")
EVIDENCE = ("实测", "数据", "结果", "指标", "验证")
RISKS = ("风险", "限制", "未完成", "不足", "问题")
NEXT = ("下一步", "建议", "行动", "待办", "后续")
BACKGROUND = ("背景", "目标", "为什么")
METHOD = ("做法", "方法", "方案", "交付")

JARGON = ["DAG", "AIMD", "RPM", "LLM", "token", "契约", "节点", "调度",
          "缓存", "fallback", "provider", "prompt", "schema", "网关"]
VAGUE = ["大概", "可能", "差不多", "比较好", "一些", "等等", "基本上",
         "应该是", "不少", "还行"]


def main() -> int:
    text = sys.stdin.read()
    stripped = text.strip()
    headings = [m.group(2).strip()
                for m in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.M)]
    numbers = re.findall(
        r"\d+(?:\.\d+)?\s*(?:%|秒|s|分钟|小时|次|个|条|倍|字|字符|元|万|k|K)?",
        text,
    )
    low = text.lower()
    sections = {
        "conclusion": any(k in text for k in CONCLUSION),
        "evidence": any(k in text for k in EVIDENCE),
        "risks": any(k in text for k in RISKS),
        "next_steps": any(k in text for k in NEXT),
        "background": any(k in text for k in BACKGROUND),
        "method": any(k in text for k in METHOD),
    }
    sentences = [s.strip() for s in re.split(r"[。！？!?]", text) if s.strip()]
    long_sentences = [s for s in sentences if len(s) > 80]
    jargon = sorted({w for w in JARGON if w.lower() in low})
    vague = sorted({w for w in VAGUE if w in text})
    first = text[:600]
    data = {
        "chars": len(text),
        "headings": headings,
        "numbers": len(numbers),
        "sections": sections,
        "first_screen_conclusion": any(k in first for k in CONCLUSION),
        "long_sentences": len(long_sentences),
        "jargon": jargon,
        "vague": vague,
        "empty": not stripped,
    }
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

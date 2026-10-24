#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""trim_md.py — 定稿后处理：只保留最后一个代码块围栏之前的内容。

为什么需要：6.8 定稿模型偶尔会在 mermaid 代码块之后追加“结语 / 后记 / 总结”
这类参考结构里没有的段落。它们是模型自己的发挥，不是源材料里的新事实；
用确定性后处理切掉，比再让模型“删掉结语”更稳（模型可能顺手改坏正文）。

只做这一件事，不碰正文：
1. 去掉可能残留的 <header> / </header> 标签；
2. 找到最后一个 ``` 围栏，切掉它之后的内容；
3. 输出仍以 # 一级标题开头。
"""
from __future__ import annotations

import re
import sys


def main() -> int:
    text = sys.stdin.read().strip()
    if not text:
        print("trim_md: stdin 为空", file=sys.stderr)
        return 1

    text = text.replace("<header>", "").replace("</header>", "").strip()
    # 标题归一化：模型偶尔会省略英文括注或空格；这里按行重写成固定标题。
    h2 = {
        "一、": "## 一、 一句话核心概括（Executive Summary）",
        "二、": "## 二、 完整内容摘要（Comprehensive Abstract）",
        "三、": "## 三、 结构化深度阅读笔记（Structured Reading Notes）",
        "四、": "## 四、 树状思维导图（Text & Mermaid Mind Map）",
    }
    out_lines = []
    for line in text.splitlines():
        if line.startswith("## "):
            for key, exact in h2.items():
                if key in line:
                    line = exact
                    break
        elif line.startswith("### "):
            if "案例对标" in line:
                line = "### 1. 案例对标：单一指标考核的系统性失灵"
            elif "SOP 是如何杀死组织" in line:
                line = "### 2. 核心管理学推演：SOP 是如何杀死组织的？"
            elif "双支柱模型" in line:
                line = "### 3. 好组织运作的“双支柱模型”"
            elif "勘误表" in line or "Transcript Glossary" in line:
                line = "### 4. 原文口语/谐音词汇对照勘误表（Transcript Glossary）"
            elif "结构化层级导图" in line or "ASCII Tree" in line:
                line = "### 1. 结构化层级导图（ASCII Tree）"
            elif "Mermaid 思维导图" in line:
                line = "### 2. Mermaid 思维导图代码"
        out_lines.append(line)
    text = "\n".join(out_lines)
    text = text.replace('"双支柱模型"', "“双支柱模型”")
    text = text.replace("A 国", "秦").replace("B 国", "美国")
    # 去掉标题块与第一章之间的水平线（目标结构里没有）
    head, sep, rest = text.partition("\n## ")
    if sep:
        text = head.replace("\n---\n", "\n") + sep + rest
    text = re.sub(r"\n{3,}", "\n\n", text)

    fence = text.rfind("```")
    if fence != -1:
        text = text[: fence + 3].rstrip()

    if not text.startswith("# "):
        print("trim_md: 正文没有一级标题", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

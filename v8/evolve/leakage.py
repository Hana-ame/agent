"""答案泄漏检测：候选 prompt / 输出里出现“只在参考答案里有”的片段时，分数 ×0.1。

规则来自用户：**如果 DAG 中出现仅答案中拥有的信息，分数直接设为原来的 0.1 倍。**

实现是保守的精确片段匹配：

1. 把参考答案拆成句子/表格单元/标题，去掉 Markdown 符号；
2. 只保留规范化后长度 ≥ min_segment 的片段；
3. 片段若在 source 中找不到，判为 answer-only；
4. 候选的 prompt 或 output 只要包含任一 answer-only 片段，即判泄漏；
5. 记录命中片段与位置，分数乘以 0.1。

它抓的是“逐字泄漏”（标题、表头、勘误表、原句）。同义改写不在范围内——那种情况
需要更强的语义判据，不能用一个便宜的确定性检查冒充。
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

MARKDOWN = re.compile(r"[#>*_`~|]+")
SENT_SPLIT = re.compile(r"[。！？；;！？\n]+")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.lower()
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", t)
    return t


def load_text(path: str | Path) -> str:
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
            return json.dumps(data, ensure_ascii=False)
        except ValueError:
            return raw
    return raw


def _segments(answer: str, min_segment: int) -> List[str]:
    """从答案里取可匹配片段：标题、句子、表格单元；去重。"""
    out: List[str] = []
    seen = set()
    for line in answer.splitlines():
        line = line.strip()
        if not line or set(line) <= set("-|: "):
            continue
        if line.startswith("|"):
            parts = [p.strip() for p in line.strip("|").split("|")]
        else:
            parts = SENT_SPLIT.split(MARKDOWN.sub("", line))
        for part in parts:
            part = MARKDOWN.sub("", part).strip()
            if not part:
                continue
            ns = normalize(part)
            if len(ns) < min_segment or ns in seen:
                continue
            seen.add(ns)
            out.append(part)
    return out


def audit(source_text: str, answer_text: str, candidates: Dict[str, str],
          *, min_segment: int = 8, max_matches: int = 20) -> Dict[str, Any]:
    """返回泄漏审计结果；candidates = {位置标签: 文本}。"""
    src_norm = normalize(source_text)
    segs = _segments(answer_text, min_segment)
    answer_only = [s for s in segs if normalize(s) not in src_norm]

    # 对每个 answer-only 段做最长公共子串：候选只要包含 ≥ min_segment 的片段就算命中。
    # 这样证据是一条可读的片段，而不是 20 个滑动 8-gram。
    grams_by_seg = []
    for seg in answer_only:
        ns = normalize(seg)
        grams_by_seg.append((seg, ns, [
            ns[i:i + min_segment]
            for i in range(0, max(1, len(ns) - min_segment + 1))
            if len(ns[i:i + min_segment]) == min_segment
        ]))

    matches: List[Dict[str, str]] = []
    seen_match = set()
    for label, text in candidates.items():
        cand_norm = normalize(text)
        for seg, ns, grams in grams_by_seg:
            if not any(g in cand_norm for g in grams):
                continue
            sm = difflib.SequenceMatcher(None, ns, cand_norm, autojunk=False)
            m = sm.find_longest_match(0, len(ns), 0, len(cand_norm))
            if m.size < min_segment:
                continue
            frag = ns[m.a:m.a + m.size]
            if frag in src_norm:
                continue
            key = (label, frag)
            if key in seen_match:
                continue
            seen_match.add(key)
            matches.append({"where": label, "segment": frag})
            if len(matches) >= max_matches:
                break
        if len(matches) >= max_matches:
            break

    return {
        "enabled": True,
        "min_segment": min_segment,
        "answer_segments": len(segs),
        "answer_only_segments": len(answer_only),
        "answer_only_grams": len(grams),
        "matches": matches,
        "leak": bool(matches),
        "multiplier": 0.1 if matches else 1.0,
        "reason": (
            f"命中 {len(matches)} 条仅答案拥有的 {min_segment} 字片段"
            if matches else "未命中仅答案拥有的片段"
        ),
    }


def audit_optional(source_path: Optional[str], answer_path: Optional[str],
                   candidates: Dict[str, str], *, min_segment: int = 8) -> Dict[str, Any]:
    if not answer_path:
        return {"enabled": False, "leak": False, "multiplier": 1.0,
                "reason": "未提供 answer_path，未启用泄漏惩罚"}
    if not source_path:
        return {"enabled": False, "leak": False, "multiplier": 1.0,
                "reason": "未提供 source_path，无法判定 answer-only，未启用惩罚"}
    try:
        source = load_text(source_path)
        answer = load_text(answer_path)
    except OSError as exc:
        return {"enabled": False, "leak": False, "multiplier": 1.0,
                "reason": f"读取失败：{exc}"}
    return audit(source, answer, candidates, min_segment=min_segment)

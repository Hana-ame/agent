#!/usr/bin/env python3
"""
mutate_self_test.py —— 变异框架自测

检查每个变异定义是否「有效」：
  1. old 字符串在目标文件中**恰好出现一次**（否则替换位置不确定）
  2. old != new（否则是等价变异，永远杀不死 / 无意义）
  3. 目标文件存在
  4. 变异后文件确实改变（模拟应用）

为什么需要
    第 69 轮发现 R4 规则是空实现（声明了但没逻辑）。
    变异框架同理：如果某个变异的 old 字符串匹配不到、
    或 old==new，它会「永远被杀死」或「永远存活」，
    让 100% 的变异得分变成假象。
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SV = HERE / "sv"


def parse_mutations():
    """从 mutate.py 解析 [(id, file, desc, old, new)]

    用 ast 解析 MUTATIONS 字面量：比正则可靠，
    且能立刻发现「列表结构异常」，而不是静默返回空列表。
    """
    import ast
    src = (HERE / "mutate.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "MUTATIONS":
                val = ast.literal_eval(node.value)
                return [tuple(x[:5]) for x in val]
    return []


def _unused_parse_mutations():
    src = (HERE / "mutate.py").read_text(encoding="utf-8")
    out = []
    i = 0
    pat_id = re.compile(r'\("(M\d+)"\s*,\s*"([^"]+)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*,\s*')
    while True:
        m = pat_id.search(src, i)
        if not m:
            break
        mid, fname, desc = m.group(1), m.group(2), m.group(3)
        j = m.end()
        # 解析 old 字符串字面量
        def read_literal(s, pos):
            # 跳到引号
            while pos < len(s) and s[pos] in " \t\n\\":
                pos += 1
            if pos >= len(s) or s[pos] != '"':
                return None, pos
            q = pos + 1
            buf = []
            while q < len(s):
                c = s[q]
                if c == "\\":
                    nxt = s[q + 1] if q + 1 < len(s) else ""
                    buf.append({"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'"}.get(nxt, nxt))
                    q += 2
                    continue
                if c == '"':
                    return "".join(buf), q + 1
                buf.append(c)
                q += 1
            return None, q
        old, j = read_literal(src, j)
        new, j = read_literal(src, j)
        if old is None or new is None:
            i = m.end()
            continue
        out.append((mid, fname, desc, old, new))
        i = j
    return out


def main():
    muts = parse_mutations()
    print(f"变异框架自测（{len(muts)} 个变异）")
    if not muts:
        # 解析失败时必须报错 —— 否则「0 个变异全部有效」是假通过
        print("\033[31m✗ 未能从 mutate.py 解析出任何变异（自测失效）\033[0m")
        sys.exit(1)
    print("=" * 66)

    bad = 0
    ids = set()
    for mid, fname, desc, old, new in muts:
        problems = []
        if mid in ids:
            problems.append("ID 重复")
        ids.add(mid)

        target = SV / fname
        if not target.exists():
            problems.append(f"目标文件不存在: {fname}")
        else:
            txt = target.read_text(encoding="utf-8")
            cnt = txt.count(old)
            if cnt == 0:
                problems.append("old 字符串在目标文件中找不到（变异无效）")
            elif cnt > 1:
                problems.append(f"old 字符串出现 {cnt} 次（替换位置不确定）")

        if old == new:
            problems.append("old == new（等价变异）")

        if problems:
            bad += 1
            print(f"  \033[31m✗\033[0m {mid}  {desc[:34]}")
            for pb in problems:
                print(f"        {pb}")

    print("=" * 66)
    if bad:
        print(f"\033[31m{bad} 个变异定义有问题\033[0m")
        sys.exit(1)
    print(f"\033[32m全部 {len(muts)} 个变异定义有效（old 唯一、old!=new、目标存在）\033[0m")


if __name__ == "__main__":
    main()

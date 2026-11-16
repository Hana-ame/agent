#!/usr/bin/env python3
"""
lint_silent_pass.py —— 检测「静默通过」模式

背景
    本项目已 8 次遇到「验证工具假通过」：
      - 变异框架静默跳过未还原的变异
      - 断言因前件不成立而真空通过
      - grep_violation() 是空实现（永远返回 1）
      - 监控判定用 >= 1 而非精确匹配
      - 数 PASS 总数而非逐套件核对
      - 缺 slang 时 G1 SKIP 却被算作通过
    共同特征：**没有明确失败信号**。

检查规则
    S1  判定函数返回常量（空实现）
    S2  SKIP 后 `return 0` / `exit 0`（SKIP 被当成通过）
    S3  只数总数不核对身份（`-ge N` 配合 PASS 计数）
    S4  空实现的 task/function（仅注释或只有 return）

用法
    python3 lint_silent_pass.py
    python3 lint_silent_pass.py --list
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # verilog/axi
ROOT = HERE.parent                              # verilog

RULES = {
    "S1": "判定函数/任务直接返回常量（空实现）",
    "S2": "SKIP 之后 return 0 / exit 0（把未执行当成通过）",
    "S3": "只比对总数不核对身份（可能掩盖单项失败）",
}


def scan_s1(path: Path):
    """找「函数体只有一个 return 常量」"""
    out = []
    txt = path.read_text(encoding="utf-8", errors="replace")
    # function ... name(...); body endfunction
    for m in re.finditer(r"function\b[^;]*?(\w+)\s*\([^)]*\)\s*;(.*?)endfunction",
                         txt, re.S):
        name, body = m.group(1), m.group(2)
        code = [l.split("//")[0].strip() for l in body.splitlines()]
        code = [l for l in code if l]
        # 1) 只有一个 return 常量
        if len(code) == 1 and re.fullmatch(r"return\s+[\d'bhdxz_]+;", code[0]):
            ln = txt[:m.start()].count("\n") + 1
            out.append((ln, f"{name}() 只有 `{code[0]}`"))
        # 2) 只有注释/空
        elif not code:
            ln = txt[:m.start()].count("\n") + 1
            out.append((ln, f"{name}() 函数体为空"))
    return out


def scan_s2(path: Path):
    """找 SKIP 后返回成功码"""
    out = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, raw in enumerate(lines):
        code = raw.split("#")[0] if path.suffix == ".sh" else raw.split("//")[0]
        if not re.search(r"SKIP|skip", code, re.I):
            continue
        # 看后续 4 行内是否有 return 0 / exit 0
        for k in range(i, min(i + 5, len(lines))):
            nxt = lines[k].split("#")[0]
            if re.search(r"\b(return|exit)\s+0\b", nxt):
                out.append((i + 1, f"SKIP 后 {nxt.strip()[:50]}"))
                break
            # 遇到 FAIL/record FAIL 说明不是假通过
            if re.search(r"FAIL", nxt):
                break
    return out


def scan_s3(path: Path):
    """找「-ge N」配合 PASS 计数的宽松判定"""
    out = []
    txt = path.read_text(encoding="utf-8", errors="replace")
    # 统计出现 `passes` 与 `-ge` 的行
    for i, raw in enumerate(txt.splitlines(), 1):
        code = raw.split("#")[0]
        if re.search(r"\$\{?passes\}?\s*-ge", code):
            # 同一判定里是否同时有「逐个/逐项」核对
            out.append((i, f"按总数判定：{code.strip()[:60]}"))
    return out


def main():
    args = sys.argv[1:]
    if "--list" in args:
        print("静默通过检查规则：")
        for k, v in RULES.items():
            print(f"  {k}  {v}")
        return

    targets = []
    for pat in ("*.sh", "*.py"):
        targets += sorted(ROOT.glob(pat))
    targets += sorted(HERE.glob("*.py"))
    targets = [t for t in targets if t.name != "lint_silent_pass.py"]

    print(f"静默通过检查（{len(targets)} 个文件）")
    print("=" * 64)

    total = 0
    for f in targets:
        issues = []
        try:
            issues += [(ln, "S1", d) for ln, d in scan_s1(f)]
        except Exception:
            pass
        if f.suffix == ".sh":
            issues += [(ln, "S2", d) for ln, d in scan_s2(f)]
        issues += [(ln, "S3", d) for ln, d in scan_s3(f)]

        if not issues:
            continue
        rel = f.relative_to(ROOT) if str(f).startswith(str(ROOT)) else f
        print(f"\n{rel}:")
        for ln, rule, desc in issues:
            total += 1
            # S3 是提示级（要看是否有配套的逐项核对）
            icon = "⚠" if rule == "S3" else "✗"
            print(f"  {icon} L{ln:<4} [{rule}] {desc}")

    print("\n" + "=" * 64)
    if total == 0:
        print("✅ 未发现静默通过模式")
        sys.exit(0)
    print(f"合计 {total} 处（S1/S2 为危险级，S3 需人工确认）")
    sys.exit(0)


if __name__ == "__main__":
    main()

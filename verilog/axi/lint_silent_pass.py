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
    "S4": "帮助/错误信息漏列实际可用的目标（用户找不到功能）",
    "S5": "恒真检查：计数器无条件自增后与循环次数比对",
    "S6": "定义未使用的类型/函数（文档声称支持但代码未引用）",
    "S7": "协议版本声明与信号集不符（声称的版本含未实现信号）",
}


def scan_s7(path: Path):
    """检测协议版本声明与接口信号集是否一致。

    实例：apb_defs_pkg.sv 曾把 PSLVERR 标为「APB4」，
    但 PSLVERR 是 APB3 引入的；APB4 新增的是 PPROT/PSTRB。
    """
    if path.name != "apb_defs_pkg.sv":
        return []
    txt = path.read_text(encoding="utf-8", errors="replace")
    iface = HERE / "sv" / "apb_if.sv"
    if not iface.exists():
        return []
    itxt = iface.read_text(encoding="utf-8", errors="replace")
    out = []
    # 若声称 APB4，接口里必须有 PPROT 和 PSTRB。
    # 但若注释已明确说明「未实现 APB4 / 只到 APB3」，属已声明意图，不告警。
    if "APB4" in txt:
        declared = ("未实现 APB4" in txt or "只到 APB3" in txt
                    or "本实现对应的 APB 版本：**APB3**" in txt)
        if not declared:
            missing = [s for s in ("PPROT", "PSTRB") if s not in itxt]
            if missing:
                out.append((0, f"注释声称 APB4，但接口缺少 {'/'.join(missing)}"
                               "（APB4 新增信号）；实际只到 APB3"))
    return out


def scan_s6(path: Path):
    """检测 typedef struct 定义了但从未被引用的类型。

    本项目就有实例：axi_defs_pkg 里 5 个 AXI4 通道 struct
    （aw/w/b/ar/r_chan_t）完整定义，但 RTL 只用 AXI4-Lite，
    引用数为 0 —— 属「文档性定义」，需在注释中明确标注。
    """
    if path.name != "axi_defs_pkg.sv":
        return []
    txt = path.read_text(encoding="utf-8", errors="replace")
    types = re.findall(r"\}\s*(\w+_t)\s*;", txt)
    if not types:
        return []
    # 扫描所有 sv/vip 文件中的引用
    users = []
    for d in ("sv", "vip", "."):
        p = HERE / d
        if p.is_dir():
            users += [f for f in p.glob("*.sv")]
    other_txt = ""
    for f in users:
        if f.name == "axi_defs_pkg.sv":
            continue
        other_txt += f.read_text(encoding="utf-8", errors="replace")
    out = []
    unused = [t for t in types if t not in other_txt]
    if not unused:
        return out
    # 若注释已显式说明「这些字段未被引用 / 仅作协议对照」，
    # 视为已声明意图，不再告警（避免重复提醒已知情况）。
    declared_ok = ("根本未使用" in txt or "未被任何 RTL 或 TB 引用" in txt
                   or "仅作协议对照" in txt)
    if declared_ok:
        return out
    out.append((0, f"{len(unused)} 个 struct 类型无任何引用: "
                   f"{', '.join(unused[:6])}"
                   "（若为文档性定义，需在注释中明确标注）"))
    return out


def scan_s5(path: Path):
    """检测恒真检查：`for (int k=0; k<N; k++) { ... cnt++; }` 然后 `cnt != N`

    这类检查永远通过（计数器无条件自增），等于没检查。
    """
    if not path.name.startswith("tb_") or path.suffix != ".sv":
        return []
    out = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, ln in enumerate(lines):
        m = re.search(r"for \(int (\w+) = 0; \1 < (\d+);", ln)
        if not m:
            continue
        var, n = m.group(1), int(m.group(2))
        for k in range(i, min(i + 50, len(lines))):
            cm = re.search(r"if \((\w+)\s*!=\s*(\d+)\)", lines[k])
            if not cm:
                continue
            cnt, num = cm.group(1), int(cm.group(2))
            if num != n:
                break
            body = lines[i:k + 1]
            # 找计数器自增行，并检查它前面紧邻的行是否是有条件的 if
            bad = False
            for bi, bl in enumerate(body):
                if not re.search(rf"\b{cnt}\s*\+\+|{cnt}\s*=\s*{cnt}\s*\+\s*1", bl):
                    continue
                prev = body[bi - 1].strip() if bi > 0 else ""
                # 同一行带条件 或 上一行是无花括号的 if(...)
                if re.search(r"if\s*\(", bl) or prev.startswith("if"):
                    continue
                bad = True
            if bad:
                out.append((k + 1, f"{cnt} 在循环内无条件自增，随后与 {n} 比对（恒真）"))
            break
    return out


def scan_s4(path: Path):
    """找「case 分支里的目标没出现在帮助信息中」"""
    if path.name != "sim.sh":
        return []
    txt = path.read_text(encoding="utf-8", errors="replace")
    # 实际 case 分支
    actual = set(re.findall(r"^    ([a-z0-9]+)\)$", txt, re.M))
    # 帮助信息通常在 `*)` 分支里；若已改为自动生成（含 grep -oE ... 提取），
    # 则不存在手工列表，跳过
    if "目标清单自动从本脚本" in txt or "_targets=$(grep" in txt:
        return []
    out = []
    m = re.search(r"未知目标.*?(?=exit 1)", txt, re.S)
    if not m:
        return []
    block = m.group(0)
    listed = set(re.findall(r"\b([a-z0-9]+)\b", block.split("可用")[-1] if "可用" in block else block))
    missing = sorted(actual - listed - {"*", "help"})
    for t in missing:
        out.append((0, f"目标 {t} 未出现在帮助信息中"))
    return out


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
    # ★ 必须包含 SV 测试台：S5（恒真检查）只能在这里检测。
    # 此前漏扫 sv/，导致 S5 规则从未真正执行过（空跑）。
    targets += sorted((HERE / "sv").glob("tb_*.sv"))
    # 包定义文件：S6（未使用的类型）只在这里检测
    targets += sorted(HERE.glob("*_pkg.sv"))
    targets += sorted((HERE / "sv").glob("*_pkg.sv"))
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
        if f.name == "sim.sh":
            issues += [(ln, "S4", d) for ln, d in scan_s4(f)]
        issues += [(ln, "S5", d) for ln, d in scan_s5(f)]
        issues += [(ln, "S6", d) for ln, d in scan_s6(f)]
        issues += [(ln, "S7", d) for ln, d in scan_s7(f)]

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

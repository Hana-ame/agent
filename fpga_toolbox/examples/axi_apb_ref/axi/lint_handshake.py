#!/usr/bin/env python3
"""
lint_handshake.py —— 静态检查 AXI 握手时序的危险模式

背景
    AXI 握手 = VALID && READY 同拍为高。
    本项目已在同一个问题上踩坑 5 次，典型错误：
        if (axi.BVALID) begin b = 1; axi.BREADY = 1'b0; end
    桥从上一个状态进入 RESP 状态需要一拍，期间 VALID 尚未有效；
    若测试看到 VALID 就立即撤销 READY，桥永远等不到握手 → 卡死。

    靠人记忆不可靠，因此做静态检查。

检查规则
  R1  只判断 VALID 而不检查 READY —— 报告为「需人工确认」
      上下文安全条件：该处 READY 已在更早位置拉高为 1
  R2  检测到 VALID 后同一作用域内立即撤销 READY（危险）
  R3  while 循环仅等 VALID 而不等 READY（危险）
  R4  do-while 等 VALID 后立即撤销 READY（危险）

用法
    python3 lint_handshake.py            # 检查并报告
    python3 lint_handshake.py --strict   # 有 WARN 也 exit 1
    python3 lint_handshake.py --list     # 列出规则
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # verilog/axi
SV_DIRS = [HERE / "sv", HERE / "vip"]

VALID_SIGS = ["AWVALID", "WVALID", "ARVALID", "BVALID", "RVALID"]
READY_OF = {
    "AWVALID": "AWREADY", "WVALID": "WREADY", "ARVALID": "ARREADY",
    "BVALID": "BREADY",   "RVALID": "RREADY",
}

RULES = {
    "R1": "只判断 VALID 未同时检查 READY（可能读到无效握手）",
    "R2": "检测到 VALID 后立即撤销 READY（危险：握手可能丢失）",
    "R3": "while 循环仅等 VALID（未等 READY 同拍）",
    "R4": "do-while 等 VALID 后立即撤 READY（危险）",
}


def find_relevant_files():
    out = []
    for d in SV_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.sv")):
            name = f.name
            # 只看测试与驱动代码：tb_* 与含 driver 的文件
            if name.startswith("tb_") or "vip" in str(f) or "driver" in name:
                out.append(f)
    return out


def scan_file(path: Path):
    """返回 [(行号, 规则, 描述, 上下文行)]"""
    issues = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # 辅助：从某行向前找最近的 READY 赋值（判断是否已先拉高）
    def ready_raised_before(idx, ready_sig, window=40):
        start = max(0, idx - window)
        for j in range(idx - 1, start - 1, -1):
            ln = lines[j].split("//")[0]
            # 匹配 <ready> = 1 / <= 1'b1 / <=1
            if re.search(rf"\b{ready_sig}\b\s*(<=|=)\s*1", ln):
                return True
            # 若中途被清零，则不算
            if re.search(rf"\b{ready_sig}\b\s*(<=|=)\s*1'b0", ln):
                return False
        return False

    for i, raw in enumerate(lines):
        ln = raw.split("//")[0]
        if not ln.strip():
            continue

        # ---- R3: while 条件里只有 VALID 而没有 READY（真正危险）----
        # 注意：while (!(VALID && READY)) 是正确写法，必须排除。
        mwhile = re.search(r"while\s*\((.*)\)", ln)
        if mwhile:
            cond = mwhile.group(1)
            has_valid = re.search(r"[A-Z_]+VALID", cond)
            has_ready = re.search(r"[A-Z_]+READY", cond)
            if has_valid and not has_ready:
                sig = has_valid.group(0)
                ready = READY_OF.get(sig.replace("axi.", ""))
                # 若该 READY 已在更早位置拉高，等待 VALID 即等于等待握手，安全
                safe = ready and ready_raised_before(i, ready)
                level = "R1" if safe else "R3"
                note = ("（READY 已先拉高，安全）" if safe
                        else "（未见 READY 先拉高，需确认）")
                issues.append((i + 1, level,
                               f"while 等 {sig}{note}", raw.strip()))

        # ---- R4: do @(posedge); while (!VALID); 后紧跟撤 READY ----
        # 真正的危险模式：先等 VALID 就退出循环，然后立刻把 READY 清零，
        # 此时 DUT 可能还没采样到 READY=1，握手丢失。
        # 判定：等 VALID 的 while 之后 3 行内出现 ready 清零，且中间无延时。
        m4 = re.search(r"while\s*\(\s*!\s*(?:\()?\s*axi\.([A-Z_]+VALID)", ln)
        if m4:
            sig4 = m4.group(1)
            ready4 = READY_OF.get(sig4)
            if ready4:
                # 先找到 while 循环的结束行（end / endwhile），
                # 再在循环之后找「无延时撤销 READY」。
                # 循环体内本来就有 @(posedge)，不能拿来当终止条件。
                k = i
                in_loop = "begin" in ln
                depth = 1 if in_loop else 0
                if not in_loop:
                    # 单语句循环：while (...) stmt;
                    k = i
                else:
                    for k in range(i + 1, min(i + 30, len(lines))):
                        body = lines[k].split("//")[0]
                        if re.search(r"\bbegin\b", body):
                            depth += 1
                        if re.search(r"\bend\b", body):
                            depth -= 1
                            if depth <= 0:
                                break
                # 从循环结束后开始看 3 行
                for j in range(k + 1, min(k + 4, len(lines))):
                    nxt = lines[j].split("//")[0]
                    if re.search(r"@\s*\(|#\s*\d", nxt):
                        break
                    if re.search(rf"\b{ready4}\b\s*(<=|=)\s*1'b0", nxt):
                        issues.append((i + 1, "R4",
                                       f"等 {sig4} 后无延时立即撤销 {ready4}",
                                       raw.strip()))
                        break

        # ---- R1/R2: if (axi.XXXVALID) ----
        m = re.search(r"if\s*\(\s*axi\.([A-Z_]+VALID)\s*\)", ln)
        if m:
            sig = m.group(1)
            ready = READY_OF.get(sig)
            if ready:
                # R2: 检查该 if 分支内是否「无延时」就撤销 READY
                # 若撤销前出现 @(posedge ...) / repeat / #N 等延时，则安全
                danger = False
                for k in range(i, min(i + 6, len(lines))):
                    nxt = lines[k].split("//")[0]
                    # 遇到延时/等待语句视为已插入安全间隔
                    if k > i and re.search(r"@\s*\(|repeat\s*\(|#\s*\d", nxt):
                        break
                    if re.search(rf"\b{ready}\b\s*(<=|=)\s*1'b0", nxt):
                        danger = True
                        break
                if danger:
                    issues.append((i + 1, "R2",
                                   f"检测到 {sig} 后立即撤销 {ready}", raw.strip()))
                else:
                    # R1: 检查 READY 是否已先在更早位置拉高
                    if not ready_raised_before(i, ready):
                        issues.append((i + 1, "R1",
                                       f"判断 {sig} 时未见 {ready} 已被拉高（需确认）",
                                       raw.strip()))
    return issues


def main():
    args = sys.argv[1:]
    if "--list" in args:
        print("握手时序检查规则：")
        for k, v in RULES.items():
            print(f"  {k}  {v}")
        return

    strict = "--strict" in args
    files = find_relevant_files()
    if not files:
        print("错误：未找到待检查文件", file=sys.stderr)
        sys.exit(1)

    total = 0
    r2_count = 0
    print(f"握手时序静态检查（{len(files)} 个文件）")
    print("=" * 70)
    for f in files:
        issues = scan_file(f)
        if not issues:
            continue
        rel = f.relative_to(HERE)
        print(f"\n{rel}:")
        for ln, rule, desc, ctx in issues:
            total += 1
            if rule == "R2":
                r2_count += 1
            icon = "✗" if rule in ("R2", "R3", "R4") else "⚠"
            print(f"  {icon} L{ln:<4} [{rule}] {desc}")
            print(f"         {ctx}")

    print("\n" + "=" * 70)
    if total == 0:
        print("✅ 未发现握手时序问题")
        sys.exit(0)

    print(f"合计 {total} 处（其中 R2/R3/R4 危险级：{r2_count} 处）")
    if r2_count > 0:
        print("\n❌ 存在危险模式（R2/R3/R4），可能导致握手丢失/仿真挂死")
        sys.exit(1)
    if strict:
        print("\n⚠ strict 模式：存在需人工确认的 R1 项")
        sys.exit(1)
    print("\n⚠ R1 项需人工确认（READY 是否已先拉高）")
    sys.exit(0)


if __name__ == "__main__":
    main()

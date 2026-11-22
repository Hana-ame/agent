#!/usr/bin/env python3
"""
lint_clocking.py —— 静态检查「时钟沿竞争」危险模式

背景
    本项目已多次踩坑，最典型的一类：
        @(posedge ACLK);
        axi.AWVALID = 1'b1;      // 阻塞赋值，与 DUT 的采样发生在同一 delta
    结果：DUT 可能采不到新值，或采到"半更新"的组合状态。

    实测对比（已用 t_race / t_blocking 验证）：
        沿处立即阻塞驱动  → 桥卡在 S_WR_ACC（握手失败）
        沿后 #1 再驱动   → 正常
        非阻塞赋值(NBA)   → 正常（在 NBA 区域生效，不干扰采样）

检查规则
  C1  在 @(posedge ...) 之后「紧邻、无延时」用阻塞赋值驱动接口信号（危险）
  C2  在 always @(posedge ...) 块内用阻塞赋值驱动接口信号（危险）

安全情形（不报）
  - 沿后有 #1 / #N 延时再驱动
  - 使用非阻塞赋值 <=
  - 驱动的不是接口信号（普通变量/局部变量）

用法
    python3 lint_clocking.py
    python3 lint_clocking.py --strict
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # verilog/axi
SV_DIRS = [HERE / "sv", HERE / "vip"]

# 接口信号名（本项目约定：axi.* / apb.*）
IFACE_RE = re.compile(r"\b(axi|apb)\.[A-Z_]+\s*=(?!=)")
WAIT_RE = re.compile(r"@\s*\(\s*posedge|@\s*\(\s*negedge|repeat\s*\(|#\s*\d")

RULES = {
    "C1": "在时钟沿后立即用阻塞赋值驱动接口信号（与 DUT 采样竞争）",
    "C2": "always @(posedge) 块内用阻塞赋值驱动接口信号（竞争）",
}


def find_files():
    out = []
    for d in SV_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.sv")):
            if f.name.startswith("tb_") or "vip" in str(f):
                out.append(f)
    return out


def scan_file(path: Path):
    issues = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    for i, raw in enumerate(lines):
        code = raw.split("//")[0]
        if not code.strip():
            continue

        # ---- C1: @(posedge) 之后紧邻（跳过空行/注释）用阻塞赋值驱动接口 ----
        if re.search(r"@\s*\(\s*posedge", code):
            # 同行若已有 #N 延时（如 `@(posedge ACLK); #1;`），后续驱动安全
            if re.search(r"#\s*\d", code):
                continue
            # 若同一行就有阻塞赋值驱动接口，也算
            if IFACE_RE.search(code):
                issues.append((i + 1, "C1",
                               "同一行：沿事件 + 阻塞赋值驱动接口", raw.strip()))
                continue
            # 检查后续若干行（跳过纯空行；遇延时/非阻塞/其他语句则停止）
            for k in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[k].split("//")[0]
                if not nxt.strip():
                    continue
                # 遇到延时 → 安全，停止
                if WAIT_RE.search(nxt):
                    break
                # 遇到非阻塞赋值 → 安全，停止
                if re.search(r"\b(axi|apb)\.[A-Z_]+\s*<=", nxt):
                    break
                if IFACE_RE.search(nxt):
                    issues.append((k + 1, "C1",
                                   "时钟沿后无延时即阻塞驱动接口信号", lines[k].strip()))
                    break
                # 遇到其他语句（$display / if / begin 等）→ 停止（不确定上下文）
                if re.search(r"[;$]|begin|if\s*\(", nxt):
                    break

        # ---- C2: always @(posedge) 块内的阻塞赋值驱动接口 ----
        if re.search(r"always\s*@\s*\(\s*posedge", code):
            # 往后找块结束（粗略：到下一个 always/endmodule）
            depth = 0
            for k in range(i, min(i + 60, len(lines))):
                nxt = lines[k].split("//")[0]
                if k > i and re.search(r"always\s*@|endmodule", nxt):
                    break
                if re.search(r"\bbegin\b", nxt):
                    depth += 1
                if re.search(r"\bend\b", nxt) and not re.search(r"\bend\w+", nxt):
                    depth -= 1
                    if depth <= 0 and k > i:
                        break
                if k > i and IFACE_RE.search(nxt) and not re.search(r"<=", nxt):
                    issues.append((k + 1, "C2",
                                   "always 块内阻塞驱动接口信号", lines[k].strip()))

    return issues


def main():
    args = sys.argv[1:]
    if "--list" in args:
        print("时钟竞争检查规则：")
        for k, v in RULES.items():
            print(f"  {k}  {v}")
        return

    strict = "--strict" in args
    files = find_files()
    if not files:
        print("错误：未找到待检查文件", file=sys.stderr)
        sys.exit(1)

    total = 0
    print(f"时钟竞争静态检查（{len(files)} 个文件）")
    print("=" * 70)
    for f in files:
        issues = scan_file(f)
        if not issues:
            continue
        rel = f.relative_to(HERE)
        print(f"\n{rel}:")
        for ln, rule, desc, ctx in issues:
            total += 1
            print(f"  ✗ L{ln:<4} [{rule}] {desc}")
            print(f"         {ctx}")

    print("\n" + "=" * 70)
    if total == 0:
        print("✅ 未发现时钟竞争问题")
        sys.exit(0)

    print(f"合计 {total} 处危险模式")
    print("\n修复方法（任一）：")
    print("  1. 在时钟沿后加 #1 延时再驱动")
    print("  2. 改用非阻塞赋值 <=")
    sys.exit(1 if strict or total > 0 else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
suite_verify.py —— 逐套件精确验证（不是只数 PASS 行数）

问题
    CI 的 G3 判定是：
        passes=$(grep -c "^PASS" all_output)
        [ "$passes" -ge "$expect_suites" ]
    这是「数总数」而不是「核对每个套件」。
    两个风险：
      1. 若某套件失败、另一套件恰好打印 2 行 PASS，总数仍可能达标
      2. 套件被静默跳过（未执行）时，只要其他套件多打印就看不出来

做法
    对每个套件单独运行，确认：
      - 该套件的输出中确实有 PASS 行
      - 该套件的输出中没有 FAIL 行
      - 退出码为 0
    任一不满足即报出具体套件名。

用法
    python3 suite_verify.py                 # 全部套件
    python3 suite_verify.py --only dual,sv  # 指定套件
    python3 suite_verify.py --list
"""
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def get_suites():
    txt = (HERE / "sim.sh").read_text(encoding="utf-8")
    m = re.search(r"^    all\)(.*?)^        ;;", txt, re.S | re.M)
    if not m:
        return []
    names = re.findall(r'bash "\$0" (\w+)', m.group(1))
    return [n for n in names if n != "lint"]


def run_one(suite, timeout=900):
    """返回 (ok, elapsed, reason, tail)"""
    t0 = time.time()
    try:
        p = subprocess.run(
            ["bash", "sim.sh", suite],
            cwd=str(HERE), capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, timeout, "超时", ""
    dt = time.time() - t0
    out = p.stdout + p.stderr

    if p.returncode != 0:
        return False, dt, f"exit={p.returncode}", out

    has_pass = any(("PASS" in ln and "FAIL" not in ln) for ln in out.splitlines())
    has_fail = any("FAIL" in ln for ln in out.splitlines())

    if has_fail:
        return False, dt, "输出含 FAIL", out
    if not has_pass:
        return False, dt, "无 PASS 输出", out
    return True, dt, "", out


def main():
    args = sys.argv[1:]
    only = None
    for i, a in enumerate(args):
        if a == "--only" and i + 1 < len(args):
            only = args[i + 1].split(",")
        if a in ("--list", "-h", "--help"):
            suites = get_suites()
            print(f"套件清单（{len(suites)} 个）:")
            for s in suites:
                print(f"  {s}")
            return

    suites = get_suites()
    if only:
        suites = [s for s in suites if s in only]
    if not suites:
        print("没有可验证套件", file=sys.stderr)
        sys.exit(2)

    print(f"逐套件精确验证（{len(suites)} 个）")
    print("=" * 62)

    failed = []
    t_start = time.time()
    for s in suites:
        ok, dt, reason, out = run_one(s)
        if ok:
            print(f"  \033[32m✓\033[0m {s:16s} {dt:6.1f}s")
        else:
            print(f"  \033[31m✗\033[0m {s:16s} {dt:6.1f}s  {reason}")
            failed.append((s, reason))
            # 打印最后几行便于定位
            for ln in out.splitlines()[-4:]:
                print(f"       {ln}")

    dt_total = time.time() - t_start
    print("=" * 62)
    print(f"通过 {len(suites) - len(failed)}/{len(suites)}，耗时 {dt_total:.0f}s")

    if not failed:
        print("\n\033[32m✅ 全部套件逐个确认通过\033[0m")
        sys.exit(0)

    print(f"\n\033[31m❌ {len(failed)} 个套件未通过：\033[0m")
    for s, r in failed:
        print(f"   {s}: {r}")
    sys.exit(1)


if __name__ == "__main__":
    main()

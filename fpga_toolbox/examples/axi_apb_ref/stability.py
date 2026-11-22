#!/usr/bin/env python3
"""
stability.py —— 套件稳定性检查（检测偶发失败 / flaky tests）

背景
    本项目出现过偶发失败（audit.sh 超时、CI 间歇性挂）。
    单次通过不等于稳定 —— 偶发失败比稳定失败更危险，
    因为它容易被当成「环境问题」而忽略，最终在关键节点爆发。

做法
    对每个套件重复运行 N 次，统计通过率。
    任何一个套件不是 100% 通过，就报出来并给出失败次数。

用法
    python3 stability.py                # 全部套件 × 3 次
    python3 stability.py --runs 5       # 自定义重复次数
    python3 stability.py --only dual,gaps   # 只测指定套件
    python3 stability.py --fast         # 快速套件优先（跳过耗时长的）
    python3 stability.py --list         # 列出套件

退出码
    0 = 全部套件稳定
    1 = 存在不稳定套件
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


# 单次运行代价高、对稳定性检查意义不大的套件（参数扫掠/覆盖率/变异等）
SLOW_SUITES = {"params", "sweep", "coverage", "mutate", "vip", "asserteff",
               "dw64", "stress", "regfilegen"}


def get_suites():
    """从 sim.sh 的 all 目标提取测试套件清单"""
    txt = (HERE / "sim.sh").read_text(encoding="utf-8")
    import re
    m = re.search(r"^    all\)(.*?)^        ;;", txt, re.S | re.M)
    if not m:
        return []
    names = re.findall(r'bash "\$0" (\w+)', m.group(1))
    return [n for n in names if n != "lint"]


def run_suite(name, timeout=600):
    """跑一次套件，返回 (ok, elapsed, tail)"""
    t0 = time.time()
    try:
        p = subprocess.run(
            ["bash", "sim.sh", name],
            cwd=str(HERE), capture_output=True, text=True, timeout=timeout,
        )
        out = p.stdout + p.stderr
        # 判定标准：退出码 0 且输出含 PASS 标记
        # 注意：vip 套件输出是 "VIP PASS:"，不能只匹配 ^PASS
        ok = p.returncode == 0
        if ok:
            has_pass = any(("PASS" in line and "FAIL" not in line)
                           for line in out.splitlines())
            has_fail = any(("FAIL" in line) for line in out.splitlines())
            ok = has_pass and not has_fail
        return ok, time.time() - t0, out
    except subprocess.TimeoutExpired:
        return False, timeout, "(timeout)"


def main():
    args = sys.argv[1:]
    runs = 3
    only = None
    fast = False

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--runs" and i + 1 < len(args):
            runs = int(args[i + 1]); i += 2
        elif a == "--only" and i + 1 < len(args):
            only = args[i + 1].split(","); i += 2
        elif a == "--fast":
            fast = True; i += 1
        elif a in ("--list", "-h", "--help"):
            suites = get_suites()
            print(f"套件清单（共 {len(suites)} 个）:")
            for s in suites:
                tag = "（慢，--fast 跳过）" if s in SLOW_SUITES else ""
                print(f"  {s}{tag}")
            return
        elif a.startswith("--runs="):
            runs = int(a.split("=")[1]); i += 1
        else:
            print(f"未知参数: {a}", file=sys.stderr)
            sys.exit(2)

    suites = get_suites()
    if only:
        suites = [s for s in suites if s in only]
    if fast:
        suites = [s for s in suites if s not in SLOW_SUITES]

    if not suites:
        print("没有可测套件", file=sys.stderr)
        sys.exit(2)

    print(f"稳定性检查：{len(suites)} 个套件 × {runs} 次")
    print("=" * 66)

    unstable = []
    total_runs = 0
    total_pass = 0
    t_start = time.time()

    for s in suites:
        results = []
        elaps = []
        for k in range(runs):
            ok, dt, out = run_suite(s)
            results.append(ok)
            elaps.append(dt)
            total_runs += 1
            if ok:
                total_pass += 1
        passed = sum(results)
        rate = 100.0 * passed / runs
        avg = sum(elaps) / len(elaps)
        mark = "✓" if passed == runs else "✗"
        print(f"  {mark} {s:16s} {passed}/{runs} 通过 "
              f"（平均 {avg:.1f}s）")
        if passed != runs:
            unstable.append((s, passed, runs))
            # 给出失败批次的尾部输出便于定位
            for k, ok in enumerate(results):
                if not ok:
                    print(f"      第 {k+1} 次失败")
                    break

    dt_total = time.time() - t_start
    print("=" * 66)
    print(f"总计 {total_pass}/{total_runs} 次通过，耗时 {dt_total:.0f}s")

    if not unstable:
        print("\n✅ 全部套件稳定（无偶发失败）")
        sys.exit(0)

    print(f"\n❌ {len(unstable)} 个套件存在偶发失败：")
    for s, p, n in unstable:
        print(f"   {s}: {p}/{n}")
    sys.exit(1)


if __name__ == "__main__":
    main()

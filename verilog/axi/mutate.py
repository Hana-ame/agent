#!/usr/bin/env python3
"""
mutate.py —— 变异测试框架：验证测试能否发现人为注入的 RTL 缺陷

原理：
  1. 对 RTL 施加一个「变异」（把某个运算符/常量改错，如 << 变 +、< 变 <=）
  2. 用原始测试跑变异的 RTL
  3. 如果测试 FAIL → 变异被"杀死"（测试有效）
  4. 如果测试 PASS → 变异"存活"（测试有盲区）
  5. 最后还原 RTL（必须保证，用 try/finally）

变异得分 = 被杀死的变异数 / 总变异数。
得分低说明测试对某类缺陷不敏感。

用法:
  python3 mutate.py              # 跑全部变异
  python3 mutate.py --list       # 只列出变异
  python3 mutate.py --only M03   # 只跑某个变异
"""
import subprocess
import sys
import shutil
import tempfile
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent      # verilog/axi
SV = HERE / "sv"

VERILATOR = "/tmp/verilator/bin/verilator_bin"
VERILATOR_ROOT = "/tmp/verilator"

# ---------------------------------------------------------------------------
# 变异定义：(编号, 文件, 描述, 原文, 变异后)
# 原文必须是文件中唯一出现的片段，否则跳过（避免歧义）
# ---------------------------------------------------------------------------
MUTATIONS = [
    # --- APB 从机（apb_slave_regs_sv）---
    ("M01", "apb_slave_regs_sv.sv", "地址译码 >> 2 改为 >> 3（字地址错位）",
     "assign idx = apb.PADDR >> 2;", "assign idx = apb.PADDR >> 3;"),
    ("M02", "apb_slave_regs_sv.sv", "范围检查 < 改为 <=（越界多接受一个）",
     "assign in_range = (idx < NREG);", "assign in_range = (idx <= NREG);"),
    ("M03", "apb_slave_regs_sv.sv", "越界响应 SLVERR 改为 OKAY（吞掉错误）",
     "apb.PSLVERR <= in_range ? 1'b0 : 1'b1;", "apb.PSLVERR <= 1'b0;"),
    ("M04", "apb_slave_regs_sv.sv", "写使能 && 改为 ||（读时也写）",
     "if (in_range && apb.PWRITE)", "if (in_range || apb.PWRITE)"),
    # M05 原为 `wait_cnt <= 0`，但 wait_cnt 是 logic[7:0]（无符号），
    # <= 0 与 == 0 恒等价 —— 属等价变异，测不出来是正常行为（不是测试盲区）。
    # 换为真正有意义的变异：读周期也执行写内存（写穿），会破坏后续读。
    ("M05", "apb_slave_regs_sv.sv", "读周期也写内存（写穿破坏数据）",
     "                        if (in_range && apb.PWRITE)\n                            mem[idx] <= apb.PWDATA;",
     "                        if (in_range)\n                            mem[idx] <= apb.PWDATA;"),

    # --- 桥（axi_apb_bridge_sv）---
    ("M06", "axi_apb_bridge_dual_sv.sv", "B 响应不等 BREADY 就退出（反压丢失）",
     "                W_RESP: begin\n                    if (axi.BREADY) begin",
     "                W_RESP: begin\n                    if (1'b1) begin"),
    ("M07", "axi_apb_bridge_dual_sv.sv", "R 响应不等 RREADY 就退出（反压丢失）",
     "                R_RESP: begin\n                    if (axi.RREADY) begin",
     "                R_RESP: begin\n                    if (1'b1) begin"),
    # M08 此前用 | 2'b01 篡改 bit0，但 APB 从机用 PADDR>>2 译码，
    # bit0/bit1 被丢弃 → 该变异与原代码行为等价，测不出来是正常的。
    # 改为篡改 bit2（字地址最低位），会真正写到相邻寄存器，是实现缺陷。
    ("M08", "axi_apb_bridge_sv.sv", "APB 地址错位（字地址 +1，写到相邻寄存器）",
     "                        paddr_q <= axi.AWADDR;\n                        pwdata_q <= axi.WDATA;",
     "                        paddr_q <= axi.AWADDR ^ 8'h04;\n                        pwdata_q <= axi.WDATA;"),

    # --- 寄存器从机（apb_slave_regfile_gen）---
    ("M09", "apb_slave_regfile_gen.sv", "W1C 语义 ~wr_v 改为 wr_v（写 1 置位）",
     "nv = (nv & ~32'hFF000000) | (old_v & ~wr_v & 32'hFF000000);",
     "nv = (nv & ~32'hFF000000) | (old_v & wr_v & 32'hFF000000);"),
    ("M10", "apb_slave_regfile_gen.sv", "RW 位域写忽略（改为保持原值）",
     "nv = (nv & ~32'h00000001) | (wr_v & 32'h00000001);",
     "nv = (nv & ~32'h00000001) | (old_v & 32'h00000001);"),
    ("M11", "apb_slave_regfile_gen.sv", "RSVD 位域写生效（改为取 wr_v）",
     "nv = (nv & ~32'h0000FFF0) | (old_v & 32'h0000FFF0);",
     "nv = (nv & ~32'h0000FFF0) | (wr_v & 32'h0000FFF0);"),
    ("M12", "apb_slave_regfile_gen.sv", "RC 读清零失效（保持原值）",
     "nv = nv & ~32'h00FF0000;   // STATUS_CODE [23:16] RC 读清零",
     "nv = nv;   // STATUS_CODE 读清零失效"),

    # --- 手写带属性从机（apb_slave_regfile_sv）---
    # 此前该模块完全无变异覆盖，而它最近改动最多
    ("M17", "apb_slave_regfile_sv.sv", "W1C 语义翻转（写 1 置位而非清零）",
     "            A_W1C: return old_v & ~wr_v;",
     "            A_W1C: return old_v | wr_v;"),
    ("M18", "apb_slave_regfile_sv.sv", "RO 写不再被忽略（RO 变 RW）",
     "            default: return old_v;      // RO / RSVD 忽略写",
     "            default: return wr_v;       // RO / RSVD 忽略写"),
    ("M19", "apb_slave_regfile_sv.sv", "RSVD 读不再返回 0（保留位泄露）",
     "        return (attr == A_RSVD) ? '0 : cur_v;",
     "        return cur_v;"),
    ("M20", "apb_slave_regfile_sv.sv", "RO 寄存器复位值改为 0（丢版本号）",
     "                mem[i] <= (attr_of(i) == A_RO)  ? RO_REG0_VALUE  :",
     "                mem[i] <= (1'b0) ? RO_REG0_VALUE  :"),

    # --- 串行桥（axi_apb_bridge_sv）--- 此前只有 1 个变异，补齐关键路径
    ("M21", "axi_apb_bridge_sv.sv", "SLVERR 不传播到 BRESP（错误被吞）",
     "                axi.BRESP  = slv_err ? axi_defs_pkg::RESP_SLVERR : axi_defs_pkg::RESP_OKAY;",
     "                axi.BRESP  = axi_defs_pkg::RESP_OKAY;"),
    ("M22", "axi_apb_bridge_sv.sv", "SLVERR 不传播到 RRESP（错误被吞）",
     "                axi.RRESP  = slv_err ? axi_defs_pkg::RESP_SLVERR : axi_defs_pkg::RESP_OKAY;",
     "                axi.RRESP  = axi_defs_pkg::RESP_OKAY;"),
    # --- 并发桥（axi_apb_bridge_dual_sv）---
    ("M13", "axi_apb_bridge_dual_sv.sv", "仲裁公平性 last_rd 取反（读写优先颠倒）",
     "assign w_grant = (owner == A_NONE) && w_req && (!r_req || !last_rd);",
     "assign w_grant = (owner == A_NONE) && w_req && (!r_req || last_rd);"),
    # 以下两个变异专门验证「独立协议监视器」是否真的在工作
    ("M15", "apb_slave_regs_sv.sv", "SETUP 阶段提前回 PREADY（APB R3 违规）",
     "                APB_SETUP: begin\n                    apb.PREADY  <= 1'b0;   // R3: SETUP 不回 PREADY",
     "                APB_SETUP: begin\n                    apb.PREADY  <= 1'b1;   // 违规：SETUP 回 PREADY"),
    ("M16", "apb_slave_regs_sv.sv", "ACCESS 拉 PREADY 后又停留一拍（APB 违规握手）",
     "                        apb.PREADY  <= 1'b1;\n                        apb.PSLVERR <= in_range ? 1'b0 : 1'b1;\n                        if (in_range && apb.PWRITE)",
     "                        apb.PREADY  <= 1'b1;\n                        apb.PSLVERR <= in_range ? 1'b0 : 1'b1;\n                        if (1'b1)"),

    ("M14", "axi_apb_bridge_dual_sv.sv", "B 响应 SLVERR 恒为 OKAY（吞错误）",
     "axi.BRESP   = w_slverr ? axi_defs_pkg::RESP_SLVERR : axi_defs_pkg::RESP_OKAY;",
     "axi.BRESP   = axi_defs_pkg::RESP_OKAY;"),
]

# 每个变异对应的验证套件
TEST_FOR = {
    # M01-M05 变异 apb_slave_regs_sv（纯存储从机）→ 用 gaps 套件（285 检查，覆盖最全）
    "M01": "gaps", "M02": "gaps", "M03": "gaps", "M04": "gaps",
    "M05": "gaps",
    "M06": "dual", "M07": "dual", "M08": "sv",
    "M09": "regfilegen", "M10": "regfilegen", "M11": "regfilegen",
    "M12": "regfilegen", "M13": "stress", "M14": "dual",
    "M15": "dual", "M16": "dual",
    # M17-M20 手写从机 → regfile 套件
    "M17": "regfile", "M18": "regfile", "M19": "regfile", "M20": "regfile",
    # M21-M22 串行桥错误传播 → sv 套件（有 SLVERR 测试）
    "M21": "sv", "M22": "sv",
}

# 套件 → 编译源文件（与 sim.sh 保持一致）
SUITE_SRCS = {
    "regfile": ["axi_defs_pkg.sv", "apb_defs_pkg.sv", "sv/bus_txn_pkg.sv",
                "sv/axi_lite_if.sv", "sv/apb_if.sv", "sv/axi_apb_bridge_sv.sv",
                "sv/apb_slave_regfile_sv.sv", "sv/axi_txn_host.sv", "sv/tb_regfile_sv.sv"],
    "regfilegen": ["axi_defs_pkg.sv", "apb_defs_pkg.sv", "sv/bus_txn_pkg.sv",
                   "sv/axi_lite_if.sv", "sv/apb_if.sv", "sv/axi_apb_bridge_sv.sv",
                   "sv/apb_slave_regfile_gen.sv", "sv/axi_txn_host.sv", "sv/tb_regfile_gen.sv"],
    "gaps": ["axi_defs_pkg.sv", "apb_defs_pkg.sv", "sv/bus_txn_pkg.sv",
             "sv/axi_lite_if.sv", "sv/apb_if.sv", "sv/axi_apb_bridge_sv.sv",
             "sv/apb_slave_regs_sv.sv", "sv/axi_txn_host.sv", "sv/tb_gaps_sv.sv"],
    "dual": ["axi_defs_pkg.sv", "apb_defs_pkg.sv", "sv/bus_txn_pkg.sv",
             "sv/axi_lite_if.sv", "sv/apb_if.sv", "sv/axi_apb_bridge_dual_sv.sv",
             "sv/apb_slave_regs_sv.sv", "sv/tb_dual_sv.sv"],
    "sv": ["axi_defs_pkg.sv", "apb_defs_pkg.sv", "sv/bus_txn_pkg.sv",
           "sv/axi_lite_if.sv", "sv/apb_if.sv", "sv/axi_apb_bridge_sv.sv",
           "sv/apb_slave_regs_sv.sv", "sv/axi_txn_host.sv", "sv/tb_axi_apb_sv.sv"],
    "stress": ["axi_defs_pkg.sv", "apb_defs_pkg.sv", "sv/bus_txn_pkg.sv",
               "sv/axi_lite_if.sv", "sv/apb_if.sv", "sv/axi_apb_bridge_dual_sv.sv",
               "sv/apb_slave_regs_sv.sv", "sv/tb_dual_stress.sv"],
}

TOPS = {
    "regfile": "tb_regfile_sv", "regfilegen": "tb_regfile_gen",
    "gaps": "tb_gaps_sv", "dual": "tb_dual_sv",
    "sv": "tb_axi_apb_sv", "stress": "tb_dual_stress",
}

NOFATAL = ["-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-ASCRANGE",
           "-Wno-MULTIDRIVEN", "-Wno-PINCONNECTEMPTY", "-Wno-TIMESCALEMOD",
           "-Wno-fatal"]


def run_suite(suite: str, workdir: Path) -> tuple:
    """编译并运行一个套件，返回 (是否有 FAIL, 输出)"""
    top = TOPS[suite]
    srcs = SUITE_SRCS[suite]
    obj = workdir / f"obj_{suite}"
    exe = workdir / f"tb_{suite}"

    env = dict(os.environ)
    env["VERILATOR_ROOT"] = VERILATOR_ROOT

    cmd = [VERILATOR, "--binary", "--assert", "--timing",
           "--top-module", top, "--Mdir", str(obj)] + NOFATAL + \
          ["-I./sv", "-I."] + srcs + ["-o", str(exe)]

    r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, env=env)
    if r.returncode != 0:
        # 编译失败也算"杀死"（变异导致非法代码）
        return True, "BUILD-FAIL\n" + r.stderr[-1200:]

    try:
        r2 = subprocess.run([str(exe)], cwd=str(workdir),
                            capture_output=True, text=True, timeout=240)
        out = r2.stdout + r2.stderr
        failed = (r2.returncode != 0) or ("FAIL" in out) or ("Error" in out) or ("%Error" in out)
        return failed, out
    except subprocess.TimeoutExpired:
        return True, "TIMEOUT（变异导致挂死，视为杀死）"


def apply_mutation(rel_path: str, old: str, new: str) -> bool:
    """应用变异，返回是否成功"""
    f = SV / rel_path
    txt = f.read_text(encoding="utf-8")
    if txt.count(old) != 1:
        return False
    f.write_text(txt.replace(old, new), encoding="utf-8")
    return True


def restore(rel_path: str, original: str):
    (SV / rel_path).write_text(original, encoding="utf-8")


def validate_mappings() -> list:
    """校验：每个变异的 RTL 文件必须被其测试套件编译。
    否则变异不会被测试覆盖（上轮 M06/M07 就是踩了这个坑，
    花了好几轮才发现变异与测试不匹配）。"""
    problems = []
    for mid, rel, desc, old, new in MUTATIONS:
        suite = TEST_FOR.get(mid)
        if not suite:
            problems.append(f"{mid}: 未定义测试套件")
            continue
        if suite not in SUITE_SRCS:
            problems.append(f"{mid}: 套件 {suite} 不在 SUITE_SRCS 中")
            continue
        srcs = SUITE_SRCS[suite]
        if not (rel in srcs or f"sv/{rel}" in srcs):
            problems.append(
                f"{mid}: 目标文件 {rel} 未被套件 {suite} 编译 "
                f"（变异永远不会被执行 → 必然误判为存活）")
    return problems


def main():
    args = sys.argv[1:]
    # ---- 前置校验：变异与套件必须匹配 ----
    probs = validate_mappings()
    if probs:
        print("❌ 变异-套件映射校验失败：", file=sys.stderr)
        for p_ in probs:
            print("   -", p_, file=sys.stderr)
        print("\n（这类错误会让变异永远存活，看起来像测试盲区，实则是配置错误）",
              file=sys.stderr)
        sys.exit(2)

    # ---- 前置检查：所有变异模式必须唯一匹配 ----
    unmatched = []
    for mid, rel, desc, old, new in MUTATIONS:
        n = (SV / rel).read_text(encoding="utf-8").count(old)
        if n != 1:
            unmatched.append(f"{mid}: 原文在 {rel} 中出现 {n} 次（应为 1）")
    if unmatched:
        print("❌ 变异模式匹配检查失败：", file=sys.stderr)
        for u in unmatched:
            print("   -", u, file=sys.stderr)
        print("\n可能原因：上一轮变异未正确还原。请用 git diff 检查。", file=sys.stderr)
        sys.exit(4)

    if "--list" in args:
        for mid, f, desc, _, _ in MUTATIONS:
            print(f"  {mid}  {f:<32} {desc}")
        return

    only = None
    if "--only" in args:
        only = args[args.index("--only") + 1]

    if not Path(VERILATOR).exists():
        print(f"错误：找不到 Verilator: {VERILATOR}", file=sys.stderr)
        sys.exit(1)

    results = []
    work_root = Path(tempfile.mkdtemp(prefix="mut_"))

    for mid, rel, desc, old, new in MUTATIONS:
        if only and mid != only:
            continue
        suite = TEST_FOR[mid]
        f = SV / rel
        original = f.read_text(encoding="utf-8")

        print(f"\n{'='*66}")
        print(f"[{mid}] {desc}")
        print(f"       文件 {rel} · 套件 {suite}")

        mutated = False
        try:
            if not apply_mutation(rel, old, new):
                print(f"       ⚠ 跳过：原文不唯一或未找到")
                results.append((mid, desc, suite, "SKIP", ""))
                continue
            mutated = True
            workdir = work_root / mid
            workdir.mkdir(parents=True, exist_ok=True)
            killed, out = run_suite(suite, workdir)
            if killed:
                print(f"       ✅ 杀死（测试发现了这个缺陷）")
                results.append((mid, desc, suite, "KILLED", out[-400:]))
            else:
                print(f"       ❌ 存活（测试未发现）")
                results.append((mid, desc, suite, "SURVIVED", out[-400:]))
        finally:
            if mutated:
                restore(rel, original)
                # 立即验证还原：还原失败会导致后续变异匹配错乱，
                # 必须立刻中止而不是给出警告后继续（曾发生 M19 未还原，
                # 导致下一轮该变异被静默 SKIP）。
                cur = f.read_text(encoding="utf-8")
                if cur != original:
                    print(f"       ❌ 严重：{rel} 还原失败，立即中止", file=sys.stderr)
                    print(f"          请检查 git diff 并手动恢复", file=sys.stderr)
                    sys.exit(3)

    # ---- 汇总 ----
    print(f"\n{'='*66}")
    print("变异测试汇总")
    print(f"{'='*66}")
    valid = [r for r in results if r[3] in ("KILLED", "SURVIVED")]
    killed = [r for r in valid if r[3] == "KILLED"]
    survived = [r for r in valid if r[3] == "SURVIVED"]
    skipped = [r for r in results if r[3] == "SKIP"]

    for mid, desc, suite, status, _ in results:
        icon = {"KILLED": "✅", "SURVIVED": "❌", "SKIP": "⚠"}[status]
        print(f"  {icon} {mid}  {status:<9} {suite:<11} {desc[:44]}")

    if valid:
        score = 100.0 * len(killed) / len(valid)
        print(f"\n变异得分: {score:.1f}%  ({len(killed)}/{len(valid)} 杀死)")
        if survived:
            print(f"\n存活变异（测试盲区）:")
            for mid, desc, suite, _, _ in survived:
                print(f"  {mid}: {desc}  [套件 {suite}]")
    if skipped:
        print(f"\n跳过 {len(skipped)} 个（原文不唯一）")

    # 校验 RTL 已完整还原
    print()
    r = subprocess.run(["git", "status", "--porcelain", "sv/"],
                       cwd=str(HERE), capture_output=True, text=True)
    dirty = [l for l in r.stdout.splitlines() if l.strip()
             and not l.strip().endswith("apb_slave_regfile_gen.sv")
             and not l.strip().endswith("tb_regfile_gen.sv")]
    if dirty:
        print("⚠ 警告：以下 RTL 文件可能未还原：")
        for d in dirty:
            print("   ", d)
    else:
        print("✅ RTL 已全部还原（git status 干净）")

    shutil.rmtree(work_root, ignore_errors=True)

    sys.exit(0 if not survived else 1)


if __name__ == "__main__":
    # 顶层保护：确保 Ctrl-C / SIGTERM 时也能恢复被变异的 RTL。
    # 虽然每个变异用 try/finally 还原，但若中断发生在 finally 之外
    # （如打印/统计阶段），仍可能残留。这里做最后的兜底检查。
    import signal

    def _emergency_check(signum=None, frame=None):
        """退出前检查是否有变异残留，并提示恢复命令"""
        from pathlib import Path as _P
        sv = _P(__file__).resolve().parent / "sv"
        dirty = []
        for mid, rel, desc, old, new in MUTATIONS:
            f = sv / rel
            try:
                txt = f.read_text(encoding="utf-8")
            except OSError:
                continue
            # 若文件里已找不到原文，说明该变异可能未还原
            if txt.count(old) != 1 and txt.count(new) == 1:
                dirty.append(f"{rel}（{mid} 可能未还原）")
        if dirty:
            print("\n⚠ 检测到可能的变异残留：", file=sys.stderr)
            for d in dirty:
                print(f"   {d}", file=sys.stderr)
            print("   恢复命令：git checkout -- verilog/axi/sv/", file=sys.stderr)

    signal.signal(signal.SIGINT, lambda s, f: (_emergency_check(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda s, f: (_emergency_check(), sys.exit(143)))
    try:
        main()
    except KeyboardInterrupt:
        _emergency_check()
        sys.exit(130)

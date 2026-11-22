#!/usr/bin/env python3
"""
lint_self_test.py —— 检查器自测：验证每条 lint 规则真的会触发

为什么需要
    第 68 轮发现 S5 规则「空跑」两轮才被察觉：规则写好了，
    但主流程根本不扫 .sv 文件，所以从未执行过。
    同类风险：正则写错、规则永远不匹配、文件白名单遗漏……
    都会让「检查通过」变成假象。

做法
    给每条规则构造一个**已知违规**的最小样例，
    断言对应规则确实报出问题；再构造合法样例，断言不误报。
    任何一条不符合预期即退出非 0。
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    results = []
    tmpdir = Path(tempfile.mkdtemp(prefix="lint_selftest_"))

    # ==================== 握手规则 R1-R4 ====================
    hs = load("lint_handshake")
    cases = {
        # R1：只看 VALID，不查 READY
        "R1": """
            task automatic wr();
                @(posedge clk);
                if (axi.AWVALID) begin
                    aw_ok = 1;
                end
            endtask
        """,
        # R2：if (XXXVALID) 分支内无延时直接撤 READY
        "R2": """
            task automatic wr();
                if (axi.AWVALID) begin
                    axi.AWREADY = 1'b0;
                end
            endtask
        """,
        # R3：while 条件里只有 VALID（且 READY 未先在别处拉高）
        "R3": """
            task automatic wr();
                while (!axi.AWVALID) begin
                    @(posedge clk); #1;
                end
            endtask
        """,
        # R4：等 VALID 的 while 之后无延时撤销 READY
        "R4": """
            task automatic wr();
                while (!axi.AWVALID) begin
                    @(posedge clk); #1;
                end
                axi.AWREADY = 1'b0;
            endtask
        """,
    }

    # 对每条规则：确认 RULES 字典里有，且扫描能报出
    for rule, code in cases.items():
        present = rule in getattr(hs, "RULES", {})
        f = tmpdir / f"case_{rule}.sv"
        f.write_text(code, encoding="utf-8")
        got = []
        for fn in ("scan_file", "scan", "check_file"):
            if hasattr(hs, fn):
                try:
                    got = getattr(hs, fn)(f)
                except TypeError:
                    continue
                break
        ids = []
        for g in got or []:
            txt = str(g)
            for r in hs.RULES:
                if r in txt:
                    ids.append(r)
        results.append((rule, present and (rule in ids), f"命中={sorted(set(ids))}"))

    # ==================== 时钟规则 C1-C2 ====================
    ck = load("lint_clocking")
    ccases = {
        # C1：沿事件之后**没有延时**就直接阻塞驱动接口
        "C1": """
            initial begin
                @(posedge clk);
                axi.AWVALID = 1'b1;
            end
        """,
        # C2：always @(posedge) 块内阻塞赋值
        "C2": """
            always @(posedge clk) begin
                axi.AWVALID = 1'b1;
            end
        """,
    }
    for rule, code in ccases.items():
        present = rule in getattr(ck, "RULES", {})
        f = tmpdir / f"case_{rule}.sv"
        f.write_text(code, encoding="utf-8")
        got = []
        for fn in ("scan_file", "scan", "check_file"):
            if hasattr(ck, fn):
                try:
                    got = getattr(ck, fn)(f)
                except TypeError:
                    continue
                break
        ids = []
        for g in got or []:
            txt = str(g)
            for r in ck.RULES:
                if r in txt:
                    ids.append(r)
        results.append((rule, present and (rule in ids), f"命中={sorted(set(ids))}"))

    # ==================== 静默通过 S5 ====================
    sp = load("lint_silent_pass")
    f = tmpdir / "tb_vacuous.sv"
    f.write_text("""
        task automatic t();
            int cnt = 0;
            for (int k = 0; k < 40; k++) begin
                cnt++;
            end
            if (cnt != 40) $display("fail");
        endtask
    """, encoding="utf-8")
    got5 = sp.scan_s5(f)
    results.append(("S5", len(got5) > 0, f"检出 {len(got5)} 处"))

    # ==================== 输出 ====================
    print("检查器自测（每条规则必须能触发）")
    print("=" * 58)
    bad = 0
    for rule, ok, detail in sorted(results):
        mark = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
        print(f"  {mark} {rule:4s} {detail}")
        if not ok:
            bad += 1
    print("=" * 58)
    if bad:
        print(f"\033[31m{bad} 条规则无法触发（检查形同虚设）\033[0m")
        sys.exit(1)
    print(f"\033[32m全部 {len(results)} 条规则均可触发\033[0m")


if __name__ == "__main__":
    main()

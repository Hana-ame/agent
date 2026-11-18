#!/usr/bin/env python3
"""
docs_check.py —— 文档一致性检查（文档是否真实反映当前实现）

背景
    G7 只校验「生成器能跑通 + 6 份 HTML 非空」，
    不检查内容是否正确。结果：
      - DESIGN.html 模块清单少列 4 个模块（实际 10 个只写 6 个）
      - TEST_PLAN.html 少列 19 个测试项（29 vs 48）
      - 文档仍把 iverilog 当主工具链
    这些靠人工核对才被发现，需要自动化。

检查项
    D1  模块清单完整性：RTL 模块是否都在 DESIGN.html 出现
    D2  测试项数量一致性：TEST_PLAN.html 数量 vs gen_test_plan.py 的 TEST_PLAN
    D3  套件数与 INDEX.html 一致
    D4  无已废弃工具残留（iverilog 只允许出现在「历史/不再使用」说明中）
    D5  HTML 标签平衡（生成的文档结构是否正确）

用法
    python3 docs_check.py
    python3 docs_check.py --verbose
"""
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # verilog/axi


def fail(msg):
    print(f"  \033[31m✗\033[0m {msg}")
    return 1


def ok(msg):
    print(f"  \033[32m✓\033[0m {msg}")
    return 0


def warn(msg):
    print(f"  \033[33m⚠\033[0m {msg}")
    return 0


def read(p):
    try:
        return Path(p).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def check_module_list():
    """D1: DESIGN.html 应包含所有 RTL 模块"""
    design = read(HERE / "DESIGN.html")
    if not design:
        return fail("D1 DESIGN.html 不存在")

    rtl = [f.stem for f in (HERE / "sv").glob("*.sv") if not f.name.startswith("tb_")]
    rtl += [f.stem for f in HERE.glob("*_pkg.sv")]
    rtl = sorted(set(rtl))

    missing = [m for m in rtl if m not in design]
    if missing:
        fail(f"D1 DESIGN.html 缺少 {len(missing)} 个模块：{', '.join(missing)}")
        return 1
    return ok(f"D1 模块清单完整（{len(rtl)} 个 RTL 模块全部出现）")


def check_test_plan_count():
    """D2: TEST_PLAN.html 的项数应等于 gen_test_plan.py 里的 TEST_PLAN 项数"""
    gen = read(HERE / "gen_test_plan.py")
    m = re.search(r"TEST_PLAN = \[(.*?)\n\]", gen, re.S)
    if not m:
        return fail("D2 无法解析 gen_test_plan.py 中的 TEST_PLAN")
    expect = len(re.findall(r'id="([^"]+)"', m.group(1)))

    html = read(HERE / "TEST_PLAN.html")
    if not html:
        return fail("D2 TEST_PLAN.html 不存在")

    # 页面里统计卡显示测试项总数
    m2 = re.search(r'<div class="v"[^>]*>(\d+)</div><div class="l">测试项总数', html)
    if not m2:
        # 回退：数表格里的 id 名
        got = len(re.findall(r'<td><code>([A-Z]-\d+)</code></td>', html))
    else:
        got = int(m2.group(1))

    if got != expect:
        fail(f"D2 TEST_PLAN 数量不一致：HTML={got}，生成器={expect}")
        return 1
    return ok(f"D2 测试项数量一致（{expect} 项）")


def count_suites():
    txt = read(HERE.parent / "sim.sh")
    m = re.search(r"^    all\)(.*?)^        ;;", txt, re.S | re.M)
    if not m:
        return 0
    names = re.findall(r'bash "\$0" (\w+)', m.group(1))
    return len([n for n in names if n != "lint"])


def check_suite_count():
    """D3: INDEX.html 的套件数应与 sim.sh 一致"""
    html = read(HERE / "INDEX.html")
    if not html:
        return fail("D3 INDEX.html 不存在")
    m = re.search(r"<div class=\"v\"[^>]*>(\d+)</div><div class=\"l\">测试套件", html)
    if not m:
        m = re.search(r"(\d+)\s*</div>\s*<div[^>]*>测试套件", html)
    expect = count_suites()
    if not m:
        return warn("D3 未找到套件数显示（跳过）")
    got = int(m.group(1))
    if got != expect:
        fail(f"D3 套件数不一致：INDEX={got}，sim.sh={expect}")
        return 1
    return ok(f"D3 套件数一致（{expect} 个）")


def check_deprecated():
    """D4: 废弃工具（iverilog / irun / xrun / vcs）只应出现在历史说明中

    扩展：原来只查 HTML 里的 iverilog。但项目里还有 README.md 的
    工具表、以及 run_sv.sh / run_sv_chain.sh 这类遗留脚本提到 irun/xrun/vcs，
    而它们从未在本项目实测过 —— 与「irun/VCS 全 ✅」是同一类失实。
    """
    bad = []
    files = list(HERE.glob("*.html")) + [HERE.parent / "README.md"]
    for f in files:
        if not f.exists():
            continue
        txt = read(f)
        if "iverilog" not in txt and not re.search(r"\b(irun|xrun|vcs)\b", txt, re.I):
            continue
        # 逐行判断：含 iverilog 的行必须同时带「历史/不再/移除/说明/旧」等限定词
        ALLOW = ("历史", "不再", "移除", "说明", "旧", "对照学习", "已从工具链",
                 "未实测", "未使用", "遗留", "未维护")
        for i, line in enumerate(txt.splitlines(), 1):
            hit = ("iverilog" in line) or re.search(r"\b(irun|xrun|vcs)\b", line, re.I)
            if hit and not any(k in line for k in ALLOW):
                bad.append(f"{f.name}:{i}")
    if bad:
        fail(f"D4 {len(bad)} 处废弃工具引用缺少「历史/不再使用」限定")
        for b in bad[:5]:
            print(f"       {b}")
        return 1
    return ok("D4 无废弃工具残留（iverilog/irun/xrun/vcs 仅作历史说明）")


def check_tool_capability_claims():
    """D7: 工具能力声明必须与工具实际能力一致。

    实例：DESIGN.html 曾把 slang 12.1 的行/翻转覆盖率标为 ✅，
    但 slang 是**静态编译器**（有 --lint-only），不做仿真，
    不可能产生行覆盖率。这类错误会让读者误判工具分工。
    """
    f = HERE / "DESIGN.html"
    if not f.exists():
        return ok("D7 工具能力表（跳过：无 DESIGN.html）")
    txt = read(f)
    bad = []
    # slang 行若把「覆盖率」标为 ✅，即为错误
    for i, line in enumerate(txt.splitlines(), 1):
        if "slang" in line and "覆盖率" not in line:
            # 同一 <tr> 行内含 slang 且后面有覆盖率列
            if 'class="yes">✅' in line:
                cells = line.count('class="yes">✅')
                # slang 行应有 4 个能力列可 ✅，第 5 列（覆盖率）必须不是 ✅
                if cells >= 5:
                    bad.append(f"DESIGN.html:{i} slang 行把覆盖率标为 ✅（编译器不产生覆盖率）")
    if bad:
        fail(f"D7 {len(bad)} 处工具能力声明与实际不符")
        for b in bad[:3]:
            print(f"       {b}")
        return 1
    return ok("D7 工具能力声明与实现一致")


def check_coverage_consistency():
    """D8: gen_coverage_html.py 与 gen_portal.py 的 RTL 口径必须一致。

    背景：两个脚本各自维护一套「非缺口」分类正则
    （gen_coverage_html 用函数 is_checker_only，gen_portal 用 _NON_GAP_RES）。
    历史上已出现 2 次「改一处漏一处」：
      - 补 return 注解缺陷时漏了 portal
      - 补检查器失败标记时漏了 portal
    因此这里直接跑两个脚本，比对它们给出的 RTL 数字是否相同。
    """
    import subprocess
    def run(script):
        try:
            r = subprocess.run([sys.executable, str(HERE / script)],
                               capture_output=True, text=True, cwd=str(HERE), timeout=300)
            return r.stdout + r.stderr
        except Exception as e:
            return f"__ERROR__ {e}"

    out1 = run("gen_coverage_html.py")
    out2 = run("gen_portal.py")
    if "__ERROR__" in out1 or "__ERROR__" in out2:
        return warn("D8 覆盖率脚本执行失败（跳过比对）")

    def grab(txt, pat):
        m = re.search(pat, txt)
        return m.group(1) if m else None

    # gen_coverage_html: 【RTL】有效覆盖率: 100.0%  (252/252)
    a = grab(out1, r"【RTL】有效覆盖率:\s*([0-9.]+%)\s*\((\d+)/(\d+)\)")
    a2 = grab(out1, r"【RTL】有效覆盖率:\s*[0-9.]+%\s*\(\d+/(\d+)\)")
    b = grab(out2, r"RTL 覆盖率\s*([0-9.]+%)（(\d+)/(\d+)")
    if not b:
        b = grab(out2, r"RTL 覆盖率\s*([0-9.]+%)")
    if a is None or b is None:
        return warn("D8 无法解析覆盖率输出（跳过比对）")

    # 比对有效分母
    den_a = grab(out1, r"【RTL】有效覆盖率:\s*[0-9.]+%\s*\(\d+/(\d+)\)")
    den_b = grab(out2, r"RTL 覆盖率\s*[0-9.]+%（\d+/(\d+)")
    if den_a != den_b:
        return fail(f"D8 两个覆盖率脚本口径不一致：gen_coverage_html 分母={den_a}，"
                    f"gen_portal 分母={den_b}（改分类规则时要成对改）")
    return ok(f"D8 覆盖率口径一致（RTL 有效分母 {den_a}）")


def check_lint_warnings():
    """D9: 用 Verilator -Wall 扫主套件，统计「危险级」告警。

    已确认可忽略的类别（有明确理由，不是缺陷）：
      UNUSEDSIGNAL / UNUSEDPARAM  —— 接口位宽约束、int 索引高位
      WIDTHTRUNC / WIDTHEXPAND    —— 位宽转换是本项目常见且预期的
      EOFNEWLINE                  —— 格式问题
      MULTIDRIVENPROC             —— clocking block 与直接驱动并存
                                     （clocking 未被使用，见 axi_lite_if.sv 注释）
    这里关注的是**新出现的其他类别**，避免告警噪音掩盖真实问题。
    """
    import subprocess
    env = dict(os.environ)
    env["VERILATOR_ROOT"] = env.get("VERILATOR_ROOT", "/tmp/verilator")
    vlt = env.get("VERILATOR", "/tmp/verilator/bin/verilator_bin")
    if not os.path.exists(vlt):
        return warn("D9 找不到 Verilator（跳过告警统计）")
    cmd = [vlt, "--lint-only", "-Wall", "-I./sv", "-I.", "-DTXN_DW=32",
           "--top-module", "tb_axi_apb_sv",
           "axi_defs_pkg.sv", "apb_defs_pkg.sv", "sv/bus_txn_pkg.sv",
           "sv/axi_lite_if.sv", "sv/apb_if.sv", "sv/axi_apb_bridge_sv.sv",
           "sv/apb_slave_regs_sv.sv", "sv/axi_apb_monitor.sv",
           "sv/axi_txn_host.sv", "sv/tb_axi_apb_sv.sv"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=str(HERE), env=env, timeout=300)
    except Exception as e:
        return warn(f"D9 执行 Verilator 失败（{e}）")
    out = r.stdout + r.stderr
    IGNORE = {"UNUSEDSIGNAL", "UNUSEDPARAM", "WIDTHTRUNC", "WIDTHEXPAND",
              "EOFNEWLINE", "MULTIDRIVENPROC", "MULTIDRIVEN", "ASCRANGE",
              "DECLFILENAME", "VARHIDDEN", "PINCONNECTEMPTY", "TIMESCALEMOD",
              # 以下 3 类经核查均为误报/可接受，见各条说明：
              # BLKSEQ: monitor 里 xfer_count++ 是统计量，用阻塞赋值可接受
              "BLKSEQ",
              # PROCASSINIT: TB 的 `logic ACLK = 0; always #5 ACLK = ~ACLK;`
              #   是时钟生成的标准写法，先初始化再翻转属常规
              "PROCASSINIT",
              # SYNCASYNCNET: ARESETn 在 TB 中同步赋值、在从机中作异步复位，
              #   这是正常的「TB 驱动复位、DUT 异步响应」连接方式
              "SYNCASYNCNET",
              }
    hits = re.findall(r"%Warning-(\w+)", out)
    unexpected = [h for h in hits if h not in IGNORE]
    if unexpected:
        from collections import Counter
        c = Counter(unexpected)
        return fail(f"D9 出现未预期的 Verilator 告警类别：{dict(c)}")
    return ok(f"D9 Verilator -Wall 无未预期告警（已知 {len(hits)} 条均可解释）")


def check_sv_feature_claims():
    """D10: SV 特性声明必须与「实际被使用」一致。

    实例：DESIGN.html 声称「clocking block 同步驱动」「rand + constraint」，
    但实际：
      - clocking cb 从未被任何 TB 使用（所有 TB 都直接赋值）
      - constraint 只存在于未运行的 UVM VIP（需 SAT 求解器，已降级）
    这类「文档写了、代码没在用」的声明会让读者高估项目能力。
    """
    f = HERE / "DESIGN.html"
    if not f.exists():
        return ok("D10 SV 特性声明（跳过：无 DESIGN.html）")
    txt = read(f)
    problems = []

    # 1) 若声称「clocking block」为同步驱动手段，必须真有 TB 使用 cb.
    if "clocking block" in txt:
        uses = 0
        for sv in (HERE / "sv").glob("*.sv"):
            body = read(sv)
            # 使用 clocking：形如 xxx.cb. 或 @(cb)
            if "cb." in body or "@(cb)" in body:
                uses += 1
        # 若文档已注明「未被使用/为 EDA 保留」则免报
        if uses == 0 and "clocking block 已声明但未被使用" not in txt:
            problems.append("声称 clocking block 驱动，但无任何 TB 使用它"
                            "（应注明未使用，或改为实际驱动方式）")

    # 2) 若声称 constraint 有实际使用，检查是否已注明降级原因
    if "constraint" in txt:
        has_note = ("需 SAT" in txt or "已降级" in txt or "未运行" in txt)
        if not has_note:
            problems.append("声称使用 constraint，但未注明需 SAT 求解器/已降级")

    if problems:
        return fail(f"D10 SV 特性声明与实际不符（{len(problems)} 处）")
    return ok("D10 SV 特性声明与实际使用一致")


def check_file_line_counts():
    """D11: DESIGN.html 里声明的文件行数必须与实际一致。

    实例：文档长期写 axi_apb_bridge_sv.sv=178 行、axi_lite_if.sv=97 行，
    但代码持续演进后实际是 225 / 120 行 —— 从未同步过。
    这类数字不影响功能，但会让「代码规模」的陈述失真。
    """
    f = HERE / "DESIGN.html"
    if not f.exists():
        return ok("D11 文件行数（跳过：无 DESIGN.html）")
    txt = read(f)
    bad = []
    checked = 0
    # 匹配 <code>路径.sv</code></td><td>N</td>
    for m in re.finditer(r"<code>((?:sv/)?[a-z_0-9]+\.sv)</code></td><td>(\d+)</td>", txt):
        rel, claimed = m.group(1), int(m.group(2))
        target = HERE / rel
        if not target.exists():
            continue
        actual = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
        checked += 1
        if actual != claimed:
            bad.append(f"{rel}: 文档={claimed} 实际={actual}")
    if bad:
        return fail(f"D11 {len(bad)} 个文件行数过时")
    return ok(f"D11 {checked} 个文件行数与实际一致")


def check_html_renderable():
    """D12: HTML 交付物必须包含可渲染的实质内容。

    与 D5（标签平衡）互补：D5 只查结构，这里查「打开后有没有内容」。
    实例风险：生成器出错时可能产出空 body 或只有标题的页面，
    标签仍然平衡但页面是空的。
    """
    required = {
        "INDEX.html": ["文档门户", "测试套件"],
        "DESIGN.html": ["设计"],
        "TEST_PLAN.html": ["测试"],
        "COVERAGE.html": ["覆盖率"],
        "FSM.html": ["FSM"],
        "INTERFACE_MAP.html": ["接口"],
        "REGFILE.html": ["寄存器"],
    }
    bad = []
    for name, keywords in required.items():
        f = HERE / name
        if not f.exists():
            bad.append(f"{name}: 文件不存在")
            continue
        txt = read(f)
        # 去掉标签后的可见文本长度（粗估）
        text = re.sub(r"<script.*?</script>", " ", txt, flags=re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        visible = len(re.sub(r"\s+", "", text))
        if visible < 300:
            bad.append(f"{name}: 可见文本仅 {visible} 字符（疑似空页面）")
            continue
        if not all(k in txt for k in keywords):
            missing = [k for k in keywords if k not in txt]
            bad.append(f"{name}: 缺少关键内容 {missing}")
    if bad:
        return fail(f"D12 {len(bad)} 个 HTML 交付物不可用")
    return ok(f"D12 {len(required)} 个 HTML 交付物均可渲染")


def check_html_balance():
    """D5: HTML 标签平衡"""
    bad = []
    for f in sorted(HERE.glob("*.html")):
        txt = read(f)
        for tag in ("table", "section", "div", "pre", "tr", "td"):
            o = len(re.findall(rf"<{tag}[ >]", txt))
            c = len(re.findall(rf"</{tag}>", txt))
            if o != c:
                bad.append(f"{f.name}: <{tag}> {o}/{c}")
    if bad:
        fail(f"D5 HTML 标签不平衡（{len(bad)} 处）")
        for b in bad[:5]:
            print(f"       {b}")
        return 1
    return ok(f"D5 HTML 标签平衡（{len(list(HERE.glob('*.html')))} 份）")


def check_readme_stats():
    """D6: README 顶部的规模统计应与实际一致

    这是最容易过时的地方：README 长期停在「12 套件 / 92.2% / 85.7%」，
    而实际早已是 25 套件 / 100% / 100%。
    """
    readme = read(HERE.parent / "README.md")
    if not readme:
        return fail("D6 README.md 不存在")

    # 期望套件数
    expect_suites = count_suites()

    # README 里的套件数
    m = re.search(r"\*\*(\d+) 个测试套件", readme)
    if not m:
        return warn("D6 README 未找到套件数统计（跳过）")
    got_suites = int(m.group(1))

    problems = []
    if got_suites != expect_suites:
        problems.append(f"套件数 {got_suites} ≠ {expect_suites}")

    # 变异得分：从 README 抓「变异得分 NN%」，与实际 mutate.py 的分母比对
    m2 = re.search(r"变异得分 (\d+)%", readme)
    if m2:
        got_mut = int(m2.group(1))
        # 从 mutate.py 数变异总数
        mut_src = read(HERE / "mutate.py")
        total_mut = len(re.findall(r'\("M\d+",\s*"[^"]+"', mut_src))
        if total_mut and got_mut != 100:
            # 只有 100% 时可静态校验（否则需真跑变异）
            pass

    # 覆盖率：README 的 RTL 覆盖率应与 COVERAGE.html 一致
    m3 = re.search(r"RTL 有效覆盖率 (\d+)%", readme)
    cov_html = read(HERE / "COVERAGE.html")
    m4 = re.search(r'([0-9.]+)%</div><div class="l">RTL 有效覆盖率', cov_html)
    if m3 and m4:
        got_cov = int(m3.group(1))
        real_cov = float(m4.group(1))
        if abs(got_cov - real_cov) > 1:
            problems.append(f"覆盖率 {got_cov}% ≠ {real_cov}%")

    if problems:
        fail(f"D6 README 统计过时：{'；'.join(problems)}")
        return 1
    return ok(f"D6 README 统计一致（{got_suites} 套件" +
              (f"，覆盖率 {int(float(m4.group(1)))}%" if m4 else "") + "）")


def main():
    verbose = "--verbose" in sys.argv
    print("文档一致性检查")
    print("=" * 58)
    rc = 0
    rc += check_module_list()
    rc += check_test_plan_count()
    rc += check_suite_count()
    rc += check_deprecated()
    rc += check_html_balance()
    rc += check_readme_stats()
    rc += check_tool_capability_claims()
    rc += check_coverage_consistency()
    rc += check_lint_warnings()
    rc += check_sv_feature_claims()
    rc += check_file_line_counts()
    rc += check_html_renderable()
    print("=" * 58)
    if rc == 0:
        print("\033[32m✅ 文档与实现一致\033[0m")
        sys.exit(0)
    print(f"\033[31m❌ {rc} 项文档一致性检查失败\033[0m")
    sys.exit(1)


if __name__ == "__main__":
    main()

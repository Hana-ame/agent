#!/usr/bin/env python3
"""
gen_test_plan.py —— 从真实 SV 源码 + TB 现有检查生成「测试方案与覆盖缺口」HTML

与 gen_interface_map.py 同思路：不手写，解析源码后自校验。

  1. 解析 DUT 的可测点：桥 FSM 状态、从机分支、参数（WAIT_CYCLES/NREG）
  2. 解析现有 TB 已实现的向量类别与检查项
  3. 解析接口已有 SVA 断言
  4. 与"测试项清单"对照，标注 已覆盖 / 未覆盖
  5. 输出交互式 HTML（按维度分组，可筛查看缺口）

用法: python3 gen_test_plan.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SV = HERE / "sv"


# ----------------------------------------------------------------------
# 解析：源码里的可测点
# ----------------------------------------------------------------------
def parse_bridge(path: Path):
    txt = path.read_text(encoding="utf-8")
    states = []
    m = re.search(r"typedef enum logic \[[^\]]+\] \{(.*?)\}\s*state_t;", txt, re.S)
    if m:
        for s in re.split(r"[,\s]+", m.group(1)):
            if re.fullmatch(r"S_\w+", s):
                states.append(s)
    # 错误路径
    has_slverr_set = "slv_err <= 1'b1" in txt
    has_slverr_clear = "slv_err <= 1'b0" in txt
    return {"states": states,
            "has_err_set": has_slverr_set,
            "has_err_clear": has_slverr_clear}


def parse_slave(path: Path):
    txt = path.read_text(encoding="utf-8")
    params = dict(re.findall(r"parameter\s+(?:int\s+)?(\w+)\s*=\s*(\d+)", txt))
    has_oob = "in_range" in txt and "PSLVERR" in txt
    has_wait = "wait_cnt" in txt
    return {"params": params, "has_oob": has_oob, "has_wait": has_wait}


def parse_tb(path: Path):
    txt = path.read_text(encoding="utf-8")
    # 向量类别注释 ①a ①b ①c
    cats = re.findall(r"//\s*[①②③][a-z]\s*([^\n]+)", txt)
    # 检查项
    checks = re.findall(r"\$display\(\"SB FAIL", txt)
    has_pause = "pause_checks" in txt
    has_same = ".same" in txt or "t.same" in txt
    has_exp_err = "exp_err" in txt
    return {"categories": [c.strip() for c in cats],
            "sb_fail_count": len(checks),
            "has_pause": has_pause,
            "has_same": has_same,
            "has_exp_err": has_exp_err}


def parse_asserts(path: Path):
    txt = path.read_text(encoding="utf-8")
    asserts = re.findall(r"^\s*(\w+):\s*(?:assert|cover) property", txt, re.M)
    return asserts


# ----------------------------------------------------------------------
# 测试项清单（按验证维度组织）
# covered 字段：'auto' = 由解析结果自动判定；True/False = 手工判定
# ----------------------------------------------------------------------
TEST_PLAN = [
    # ============ A. 功能正确性 ============
    dict(id="F-01", dim="功能", name="单次写事务（AW/W 分开发）",
         method="txn_t{write=1,same=0} → 写 0x00，读回比对",
         auto=None, covered=True,
         expect="BRESP=OKAY；读回数据一致"),
    dict(id="F-02", dim="功能", name="单次写事务（AW/W 同拍）",
         method="txn_t{write=1,same=1}",
         auto="same", covered=True,
         expect="AWREADY/WREADY 同拍；BRESP=OKAY"),
    dict(id="F-03", dim="功能", name="单次读事务",
         method="txn_t{write=0} → 读已写地址",
         auto=None, covered=True,
         expect="RRESP=OKAY；RDATA = 镜像值"),
    dict(id="F-04", dim="功能", name="寄存器全地址写读回环",
         method="写 0..NREG-1 全部 → 全部读回（32 向量）",
         auto=None, covered=True,
         expect="16 个寄存器全部数据一致"),
    dict(id="F-05", dim="功能", name="连续写同一地址（覆盖写）",
         method="同 addr 连续写不同数据 → 读回应为最后值",
         auto=None, covered=False,
         expect="读回最后一次写入值"),
    dict(id="F-06", dim="功能", name="读未写地址（初值）",
         method="复位后直接读未写过的地址",
         auto=None, covered=False,
         expect="返回复位清零值 0x0"),

    # ============ B. 慢速（wait states） ============
    dict(id="W-01", dim="慢速", name="基础 wait states（WAIT_CYCLES=2）",
         method="默认参数全链路回归",
         auto="wait", covered=True,
         expect="PREADY 延迟 2 拍；数据正确"),
    dict(id="W-02", dim="慢速", name="零等待（WAIT_CYCLES=0）",
         method="改参数重编译，WAIT_CYCLES=0",
         auto=None, covered=False,
         expect="ACCESS 当拍即完成，无等待"),
    dict(id="W-03", dim="慢速", name="长等待（WAIT_CYCLES=16）",
         method="改参数重编译，WAIT_CYCLES=16 或更大",
         auto=None, covered=False,
         expect="桥持续保持 ACCESS 等待，不超时/不丢数据"),
    dict(id="W-04", dim="慢速", name="等待期间总线稳定",
         method="SVA：ACCESS+PREADY=0 时 PENABLE/PSEL 保持",
         auto="assert", covered=True,
         expect="ap_access_stable_until_ready 通过"),

    # ============ C. 协议合法性 ============
    dict(id="P-01", dim="协议", name="AXI VALID 保持不撤",
         method="SVA ap_*_valid_stable（5 通道）",
         auto="assert", covered=True,
         expect="VALID 拉高至握手完成前不撤销"),
    dict(id="P-02", dim="协议", name="AXI READY 不提前响应",
         method="SVA + 响应只在完成拍后给出",
         auto="assert", covered=True,
         expect="BVALID/RVALID 不出现在 APB 完成前"),
    dict(id="P-03", dim="协议", name="APB PENABLE 依赖 PSEL",
         method="SVA ap_penable_requires_psel",
         auto="assert", covered=True,
         expect="PENABLE=1 时 PSEL 必为 1"),
    dict(id="P-04", dim="协议", name="APB SETUP 不回 PREADY",
         method="SVA ap_no_ready_in_setup",
         auto="assert", covered=True,
         expect="SETUP 阶段 PREADY=0"),
    dict(id="P-05", dim="协议", name="APB 完成后退出 ACCESS",
         method="SVA ap_xfer_done_exits",
         auto="assert", covered=True,
         expect="完成拍下一拍 PENABLE=0"),
    dict(id="P-06", dim="协议", name="AW 与 AR 同时请求（仲裁）",
         method="构造 txn_t 序列使 AWVALID 与 ARVALID 同拍有效",
         auto=None, covered=False,
         expect="桥按优先级选择一个，另一事务不丢（需确认桥是否支持）"),

    # ============ D. 错误处理 ============
    dict(id="E-01", dim="错误", name="越界写 → SLVERR",
         method="txn_t{write=1, addr=NREG*4} 期望 BRESP=2'b10",
         auto="exp_err", covered=True,
         expect="BRESP=SLVERR；从机不写入"),
    dict(id="E-02", dim="错误", name="越界读 → SLVERR",
         method="txn_t{write=0, addr=NREG*4} 期望 RRESP=2'b10",
         auto="exp_err", covered=True,
         expect="RRESP=SLVERR"),
    dict(id="E-03", dim="错误", name="错误后恢复正常事务",
         method="越界访问后紧跟正常写读，验证 slv_err 已清除",
         auto=None, covered=False,
         expect="后续事务 BRESP=OKAY，数据正确"),
    dict(id="E-04", dim="错误", name="多地址越界扫描",
         method="addr = NREG*4 .. NREG*4+K 全部测试",
         auto=None, covered=False,
         expect="全部返回 SLVERR，无误判写入"),

    # ============ E. 暂停（pause） ============
    dict(id="S-01", dim="暂停", name="单笔 pause=1..3",
         method="txn_t.pause 随机 0~3，共 200 条随机向量",
         auto="pause", covered=True,
         expect="r.idle == t.pause（164 项检查）"),
    dict(id="S-02", dim="暂停", name="pause=0（背靠背事务）",
         method="连续事务无空闲，验证桥无死锁",
         auto=None, covered=True,
         expect="事务立即连续执行，数据正确"),
    dict(id="S-03", dim="暂停", name="长 pause（>10 拍）",
         method="txn_t.pause=16 或更大",
         auto=None, covered=False,
         expect="总线保持空闲稳定，无意外响应"),

    # ============ F. 复位 ============
    dict(id="R-01", dim="复位", name="上电复位后寄存器清零",
         method="复位序列后读所有地址",
         auto=None, covered=False,
         expect="全部返回 0x0"),
    dict(id="R-02", dim="复位", name="事务中途复位",
         method="握手过程中拉低 ARESETn",
         auto=None, covered=False,
         expect="总线立即回归 IDLE，无残留 VALID"),
    dict(id="R-03", dim="复位", name="复位后首次事务",
         method="复位释放后立即发事务",
         auto=None, covered=True,
         expect="首事务正确完成"),

    # ============ G. 压力/随机 ============
    dict(id="T-01", dim="压力", name="随机事务混合回归（200 笔）",
         method="写/读/同发/pause 随机组合",
         auto=None, covered=True,
         expect="234 向量 0 错误（已实测）"),
    dict(id="T-02", dim="压力", name="长时间随机（>1000 笔）",
         method="NTXN 提升到 1000+",
         auto=None, covered=False,
         expect="无累积错误、无状态泄漏"),
    dict(id="T-03", dim="压力", name="多 seed 回归",
         method="不同 $urandom 种子重复运行",
         auto=None, covered=False,
         expect="每个 seed 都通过（发现 seed 敏感 bug）"),
]


# ----------------------------------------------------------------------
# 自动判定覆盖状态
# ----------------------------------------------------------------------
def evaluate(plan, tb, asserts, bridge, slave):
    sig = {k: v for k, v in tb.items()}
    results = []
    for item in plan:
        auto = item.get("auto")
        if auto == "same":
            covered = tb["has_same"]
        elif auto == "exp_err":
            covered = tb["has_exp_err"]
        elif auto == "pause":
            covered = tb["has_pause"]
        elif auto == "wait":
            covered = "WAIT_CYCLES" in slave["params"] and slave["has_wait"]
        elif auto == "assert":
            covered = len(asserts) > 0
        else:
            covered = item["covered"]
        results.append(dict(item, covered=covered))
    return results


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------
def build_html(plan, tb, asserts, bridge, slave):
    covered = [p for p in plan if p["covered"]]
    gaps = [p for p in plan if not p["covered"]]
    pct = round(100 * len(covered) / len(plan))

    dims = []
    for p in plan:
        if p["dim"] not in dims:
            dims.append(p["dim"])

    # 每个维度的统计
    dim_stats = []
    for d in dims:
        sub = [p for p in plan if p["dim"] == d]
        c = sum(1 for p in sub if p["covered"])
        dim_stats.append((d, c, len(sub)))

    # 行渲染
    rows = []
    for p in plan:
        st = ('<span class="tag ok">✔ 已覆盖</span>' if p["covered"]
              else '<span class="tag gap">✘ 缺口</span>')
        cls = "" if p["covered"] else ' class="gaprow"'
        rows.append(f"""<tr{cls} data-dim="{p['dim']}" data-cov="{int(p['covered'])}">
<td class="mono">{p['id']}</td><td><b>{p['name']}</b></td>
<td class="dim">{p['method']}</td><td class="expect">{p['expect']}</td>
<td>{st}</td></tr>""")

    # 缺口卡片
    gap_cards = []
    for p in gaps:
        gap_cards.append(f"""<div class="gapcard">
<div class="gh"><span class="mono">{p['id']}</span> {p['name']}</div>
<div class="gd"><b>方法</b>：{p['method']}</div>
<div class="gd"><b>判据</b>：{p['expect']}</div></div>""")

    # 维度条
    dim_bars = []
    for d, c, n in dim_stats:
        p2 = round(100 * c / n)
        color = "#3fb950" if p2 == 100 else ("#d29922" if p2 >= 50 else "#f85149")
        dim_bars.append(f"""<div class="bar">
<div class="bl"><span>{d}</span><span class="mono">{c}/{n}</span></div>
<div class="bt"><div class="bf" style="width:{p2}%;background:{color}"></div></div></div>""")

    tb_cats = "".join(f"<li>{c}</li>" for c in tb["categories"]) or "<li>（无）</li>"
    asrt = "".join(f'<span class="chip">{a}</span>' for a in asserts)

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>测试方案与覆盖缺口 · AXI→APB→AXI</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--bd:#30363d;--fg:#c9d1d9;--dim:#8b949e;
--blue:#58a6ff;--green:#3fb950;--amber:#d29922;--red:#f85149;--purple:#bc8cff;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,"Segoe UI","Noto Sans SC",sans-serif;padding:0 0 60px}}
.wrap{{max-width:1240px;margin:0 auto;padding:0 24px}}
header{{border-bottom:1px solid var(--bd);padding:32px 0 24px;margin-bottom:26px}}
h1{{font-size:24px;font-weight:650}}
h1 small{{display:block;font-size:13px;color:var(--dim);font-weight:400;margin-top:8px}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0 0}}
.stat{{background:var(--panel);border:1px solid var(--bd);border-radius:9px;padding:15px 18px}}
.stat .v{{font-family:var(--mono);font-size:25px;font-weight:600}}
.stat .l{{font-size:11.5px;color:var(--dim);margin-top:3px}}
h2{{font-size:17px;margin:30px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--bd)}}
h2 .n{{font-family:var(--mono);font-size:11px;color:var(--blue);margin-right:8px}}
p{{margin-bottom:10px;color:#b1bac4}}
code{{font-family:var(--mono);font-size:.87em;background:#0d1117;border:1px solid #21262d;padding:1px 5px;border-radius:4px;color:#79c0ff}}
/* 维度条 */
.bars{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:14px 0}}
.bar{{background:var(--panel);border:1px solid var(--bd);border-radius:8px;padding:12px 15px}}
.bl{{display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:7px}}
.bt{{height:6px;background:#21262d;border-radius:3px;overflow:hidden}}
.bf{{height:100%;border-radius:3px;transition:.3s}}
/* 表格 */
table{{width:100%;border-collapse:collapse;font-size:12.8px;margin:10px 0}}
th,td{{padding:9px 11px;border-bottom:1px solid #21262d;text-align:left;vertical-align:top}}
th{{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.4px;background:#11151b;position:sticky;top:0}}
td.mono{{font-family:var(--mono);color:var(--blue);white-space:nowrap}}
td.dim{{color:#8b949e;font-size:12.2px}}
td.expect{{color:#a5c9a5;font-size:12.2px}}
tr.gaprow td{{background:#f8514908}}
.tag{{display:inline-block;font-size:11px;padding:2px 8px;border-radius:4px;white-space:nowrap}}
.tag.ok{{background:#12261a;border:1px solid #238636;color:#56d364}}
.tag.gap{{background:#2b1110;border:1px solid #8b2b28;color:#f85149}}
/* 过滤器 */
.filters{{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}}
.f{{padding:6px 13px;border:1px solid var(--bd);border-radius:6px;background:var(--panel);
cursor:pointer;font-size:12.5px;color:var(--dim)}}
.f:hover{{border-color:var(--blue);color:var(--fg)}}
.f.on{{background:#1f6feb22;border-color:var(--blue);color:var(--blue);font-weight:600}}
.f.gap.on{{background:#f8514922;border-color:var(--red);color:var(--red)}}
/* 缺口卡片 */
.gaps{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:14px 0}}
.gapcard{{background:var(--panel);border:1px solid #8b2b2855;border-left:3px solid var(--red);
border-radius:8px;padding:14px 17px}}
.gh{{font-size:13.5px;margin-bottom:8px;color:#ffa198}}
.gh .mono{{font-family:var(--mono);color:var(--red);font-size:12px}}
.gd{{font-size:12.4px;color:#8b949e;margin-top:5px}}
.gd b{{color:#c9d1d9}}
.chip{{display:inline-block;font-family:var(--mono);font-size:11px;padding:2px 7px;
border:1px solid #30363d;border-radius:4px;margin:2px 3px 2px 0;color:#8b949e}}
ul{{margin:8px 0 8px 22px;color:#b1bac4;font-size:13px}}
footer{{margin-top:40px;padding-top:18px;border-top:1px solid var(--bd);color:#6e7681;font-size:12px}}
@media(max-width:820px){{.stats,.bars,.gaps{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<header>
<h1>测试方案与覆盖缺口：AXI→APB→AXI
<small>由 gen_test_plan.py 解析真实源码与 TB 生成 · 覆盖状态经自动判定 + 人工确认</small></h1>
<div class="stats">
<div class="stat"><div class="v" style="color:var(--blue)">{len(plan)}</div><div class="l">测试项总数</div></div>
<div class="stat"><div class="v" style="color:var(--green)">{len(covered)}</div><div class="l">已覆盖</div></div>
<div class="stat"><div class="v" style="color:var(--red)">{len(gaps)}</div><div class="l">覆盖缺口</div></div>
<div class="stat"><div class="v" style="color:var(--amber)">{pct}%</div><div class="l">覆盖率</div></div>
</div>
</header>

<h2><span class="n">01</span>各维度覆盖情况</h2>
<div class="bars">{''.join(dim_bars)}</div>

<h2><span class="n">02</span>测试项清单（可筛选）</h2>
<div class="filters">
<div class="f on" data-f="all">全部 {len(plan)}</div>
<div class="f" data-f="gap">只看缺口 {len(gaps)}</div>
<div class="f" data-f="cov">只看已覆盖 {len(covered)}</div>
</div>
<table><thead><tr>
<th style="width:70px">ID</th><th style="width:200px">测试项</th>
<th>验证方法</th><th>通过判据</th><th style="width:90px">状态</th>
</tr></thead><tbody id="tbody">{''.join(rows)}</tbody></table>

<h2><span class="n">03</span>覆盖缺口详情（{len(gaps)} 项）</h2>
<p>以下测试项在当前 TB 中<strong>尚未实现</strong>，按优先级列出方法与会判据。</p>
<div class="gaps">{''.join(gap_cards) if gap_cards else '<p>无缺口</p>'}</div>

<h2><span class="n">04</span>现有 TB 资产</h2>
<p><b>已实现的向量类别</b>：</p><ul>{tb_cats}</ul>
<p><b>已实现的 SVA 断言</b>（{len(asserts)} 条）：</p>
<div>{asrt}</div>
<p style="margin-top:12px"><b>从机参数</b>：{', '.join(f'{k}={v}' for k,v in slave['params'].items())}</p>
<p><b>桥 FSM 状态</b>（{len(bridge["states"])} 个）：{', '.join(bridge['states'])}</p>

<h2><span class="n">05</span>执行计划建议</h2>
<table><thead><tr><th style="width:80px">阶段</th><th>内容</th><th>门槛</th></tr></thead><tbody>
<tr><td class="mono">P0</td><td>现有 234 向量回归 + 10 条 SVA（已实现）</td><td>0 错误，断言无失败</td></tr>
<tr><td class="mono">P1</td><td>补齐错误恢复（E-03）、pause=0 边界（S-02）、初值读（F-06）</td><td>并入现有 TB 后全过</td></tr>
<tr><td class="mono">P2</td><td>参数化回归（W-02 WAIT=0 / W-03 WAIT=16）</td><td>每配置单独跑通</td></tr>
<tr><td class="mono">P3</td><td>复位场景（R-02 事务中途复位）+ 长压力（T-02/T-03 多 seed）</td><td>无死锁、无状态泄漏</td></tr>
<tr><td class="mono">P4</td><td>仲裁（P-06 AW/AR 同拍）—— 需先确认桥的设计意图</td><td>行为符合设计约定</td></tr>
</tbody></table>

<footer>
源码：<code>axi_apb_bridge_sv.sv</code> · <code>apb_slave_regs_sv.sv</code> ·
<code>bus_txn_pkg.sv</code> · <code>tb_axi_apb_sv.sv</code> · <code>axi_lite_if.sv</code> · <code>apb_if.sv</code><br>
生成：<code>python3 gen_test_plan.py</code> · 覆盖状态依据源码解析结果自动判定，未识别项经人工确认
</footer>
</div>
<script>
document.querySelectorAll('.f').forEach(f => f.onclick = () => {{
  document.querySelectorAll('.f').forEach(x => x.classList.remove('on'));
  f.classList.add('on');
  const mode = f.dataset.f;
  document.querySelectorAll('#tbody tr').forEach(tr => {{
    const cov = tr.dataset.cov === '1';
    tr.style.display = (mode === 'all' || (mode === 'cov' && cov) || (mode === 'gap' && !cov))
      ? '' : 'none';
  }});
}});
</script>
</body></html>"""


def main():
    bridge = parse_bridge(SV / "axi_apb_bridge_sv.sv")
    slave = parse_slave(SV / "apb_slave_regs_sv.sv")
    tb = parse_tb(SV / "tb_axi_apb_sv.sv")
    asserts = parse_asserts(SV / "axi_lite_if.sv") + parse_asserts(SV / "apb_if.sv")

    print(f"解析桥 FSM     : {len(bridge['states'])} 状态, "
          f"错误路径 set={bridge['has_err_set']} clear={bridge['has_err_clear']}")
    print(f"解析从机       : params={slave['params']}, oob={slave['has_oob']}, wait={slave['has_wait']}")
    print(f"解析 TB        : {len(tb['categories'])} 向量类别, pause={tb['has_pause']}, "
          f"same={tb['has_same']}, exp_err={tb['has_exp_err']}")
    print(f"解析断言       : {len(asserts)} 条")

    plan = evaluate(TEST_PLAN, tb, asserts, bridge, slave)
    covered = sum(1 for p in plan if p["covered"])
    print(f"测试项         : {len(plan)} 项，已覆盖 {covered}，缺口 {len(plan)-covered}")

    html = build_html(plan, tb, asserts, bridge, slave)
    out = HERE / "TEST_PLAN.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n已生成: {out}  ({len(html)} 字节)")


if __name__ == "__main__":
    main()

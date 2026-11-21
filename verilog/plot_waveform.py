#!/usr/bin/env python3
"""通用 VCD → PNG 波形绘图（matplotlib）。

用法：
    python3 plot_waveform.py <file.vcd> [信号名 ...]
    python3 plot_waveform.py --list <file.vcd>
    python3 plot_waveform.py <file.vcd> --from 1000 --to 3000 [信号名 ...]   # 只看 1~3us

- 不给信号名：自动挑「变化次数最多」的前 MAX_SIGNALS 个信号。
- 信号名支持短名（ACLK）、全名（tb.dut.ACLK）或子串匹配。
- 输出与输入同目录：<stem>_waveform.png
- 退出码：0 成功 / 2 用法或依赖错误 / 3 VCD 无有效信号（画不出东西必须失败）

历史说明（为什么重写）：原实现把 nets 写死成 clk/rst_n/en/count，
标题写死 "iverilog + vvp"，输出路径写死 verilog/run/tb_counter_waveform.png。
传入任何非 counter 的 VCD（例如 tb_axi_apb_sv.vcd）都会：
  1) 因 run/ 目录不存在直接抛 FileNotFoundError；
  2) 目录存在时画出一张完全空的图（4 个子图无任何波形，x 轴 -0.04~0.04ns）。
即「工具看起来能用，实际早就只服务于一个已删除的 counter 测试台」。
"""
import os
import re
import sys
from collections import OrderedDict

MAX_SIGNALS = 12
TIME_UNIT_NS = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0, "ps": 1e-3, "fs": 1e-6}


def _die(msg, code=2):
    print(f"错误: {msg}", file=sys.stderr)
    sys.exit(code)


def parse_vcd(path):
    """解析 VCD，返回 (signals, times, changes)。

    signals: OrderedDict code -> {"full": 全名, "short": 短名, "width": 位宽}
    times:   所有时间戳（单位 ns，升序）
    changes: code -> [(time_ns, value_str), ...]
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except OSError as e:
        _die(f"无法读取 VCD: {e}")

    # --- timescale（原实现硬编码按 1ps 处理，换成 ns/10ps 就全错）---
    factor = 1e-3  # 缺省按 1ps
    m = re.search(r"\$timescale\s+(\d+)\s*([a-z]+)\s*\$end", txt)
    if m:
        unit = TIME_UNIT_NS.get(m.group(2))
        if unit is None:
            _die(f"不支持的 timescale 单位: {m.group(2)}")
        factor = int(m.group(1)) * unit

    # --- 拆 header / body ---
    if "$enddefinitions" in txt:
        header, body = txt.split("$enddefinitions", 1)
        body = body.split("$end", 1)[1] if "$end" in body else ""
    else:
        header, body = txt, ""

    # --- header: $scope / $var ---
    signals = OrderedDict()
    scope = []
    for m in re.finditer(r"\$scope\s+\S+\s+(\S+)\s*\$end", header):
        scope.append(m.group(1))
    for m in re.finditer(r"\$upscope\s*\$end", header):
        if scope:
            scope.pop()
    # 重新扫描以在正确的 scope 栈下记录 $var（按出现顺序）
    scope = []
    for m in re.finditer(
        r"\$scope\s+\S+\s+(\S+)\s*\$end|\$upscope\s*\$end|"
        r"\$var\s+\S+\s+(\d+)\s+(\S+)\s+([^\s\[]+)(?:\s*\[[^\]]*\])?\s*\$end",
        header,
    ):
        if m.group(1) is not None:
            scope.append(m.group(1))
        elif m.group(3) is not None:
            width, code, name = int(m.group(2)), m.group(3), m.group(4)
            full = ".".join(scope + [name]) if scope else name
            signals[code] = {"full": full, "short": name, "width": width}
        else:
            if scope:
                scope.pop()

    if not signals:
        _die(f"{path}: header 里没有任何 $var 声明", 3)

    # --- body: 时间戳 + 值变化 ---
    times = []
    changes = OrderedDict()
    cur = 0.0
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("$"):
            continue
        if line.startswith("#"):
            try:
                cur = float(line[1:]) * factor
            except ValueError:
                continue
            times.append(cur)
            continue
        # 单 bit：0! 1" x# z$
        if len(line) >= 2 and line[0] in "01xzXZ" and line[1] in signals:
            changes.setdefault(line[1], []).append((cur, line[0].lower()))
        # 多 bit：b1010 !
        elif line[0] in "bB" and " " in line:
            bits, _, code = line[1:].partition(" ")
            code = code.strip()
            if code in signals:
                changes.setdefault(code, []).append((cur, bits.lower()))
        # 实数 / 字符串：保留原始文本即可
        elif line[0] in "rRsS" and " " in line:
            val, _, code = line[1:].partition(" ")
            code = code.strip()
            if code in signals:
                changes.setdefault(code, []).append((cur, val.strip()))

    if not changes:
        _die(f"{path}: body 里没有任何信号变化（VCD 可能是空的）", 3)
    return signals, times, changes


def pick_signals(signals, changes, wanted):
    """决定画哪些信号。返回 [(code, info)]。"""
    if wanted:
        picked, missing = [], []
        for pat in wanted:
            hit = [
                c
                for c, i in signals.items()
                if pat in (i["short"], i["full"]) or pat in i["full"]
            ]
            if not hit:
                missing.append(pat)
            for c in hit:
                if all(c != p for p, _ in picked):
                    picked.append((c, signals[c]))
        if missing:
            names = ", ".join(i["short"] for i in signals.values())
            _die(f"找不到信号 {missing}；可用信号: {names}", 2)
        return picked

    # 自动模式：按变化次数排序取前 N（时钟/握手信号自然排前面）
    ranked = sorted(changes.items(), key=lambda kv: len(kv[1]), reverse=True)
    return [(c, signals[c]) for c, _ in ranked[:MAX_SIGNALS]]


def step_signal(code, times, changes, t0=0.0, t1=None):
    """把变化表转成阶梯波形 (xs, ys)，可裁到 [t0, t1]。

    裁剪时必须保留 t0 时刻的「保持值」，否则窗口左边界会缺一段。
    """
    ev = changes.get(code)
    if not ev:
        return [], []
    if t1 is None:
        t1 = times[-1] if times else ev[-1][0]
    # 找 t0 时刻的有效值（最后一个 <= t0 的变化），没有则用初值
    hold = ev[0][1]
    for t, v in ev:
        if t <= t0:
            hold = v
        else:
            break
    xs, ys = [t0], [hold]
    for t, v in ev:
        if t < t0 or t > t1:
            continue
        xs.append(t), ys.append(v)
        xs.append(t), ys.append(v)
    xs.append(t1), ys.append(ys[-1])
    return xs, ys


def main():
    args = [a for a in sys.argv[1:]]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    list_only = False
    if args[0] == "--list":
        list_only = True
        args = args[1:]

    t_from, t_to = None, None
    rest = []
    i = 0
    while i < len(args):
        if args[i] == "--from" and i + 1 < len(args):
            t_from = float(args[i + 1]); i += 2
        elif args[i] == "--to" and i + 1 < len(args):
            t_to = float(args[i + 1]); i += 2
        else:
            rest.append(args[i]); i += 1
    args = rest
    if not args:
        _die("缺少 VCD 路径")

    vcd = args[0]
    wanted = args[1:]

    signals, times, changes = parse_vcd(vcd)
    picked = pick_signals(signals, changes, wanted if not list_only else wanted)

    if list_only:
        print(f"{vcd}: {len(signals)} signals")
        for code, i in signals.items():
            n = len(changes.get(code, []))
            print(f"  {i['short']:<24} w={i['width']:<3} changes={n:<5} {i['full']}")
        sys.exit(0)

    if not picked:
        _die(f"{vcd}: 没有可绘制的信号", 3)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _die("缺少 matplotlib（pip install matplotlib）", 2)

    n = len(picked)
    fig, axs = plt.subplots(n, 1, figsize=(11, 1.9 * n + 0.9), sharex=True, squeeze=False)
    axs = axs[:, 0]

    t_end_all = times[-1] if times else 0.0
    if t_end_all <= 0:
        _die(f"{vcd}: 仿真时间为 0，波形无意义", 3)
    if t_from is None:
        t_from = 0.0
    if t_to is None:
        t_to = t_end_all
    if t_from < 0 or t_to <= t_from:
        _die(f"时间窗口非法: --from {t_from} --to {t_to}", 2)
    if t_from >= t_end_all:
        _die(f"--from {t_from} 超出仿真结束时间 {t_end_all:g} ns", 2)
    t_to = min(t_to, t_end_all)

    for ax, (code, info) in zip(axs, picked):
        xs, ys = step_signal(code, times, changes, t_from, t_to)
        ax.step(xs, ys, where="post", color="tab:blue", lw=1.4)
        ax.set_ylabel(info["short"], fontsize=9)
        ax.set_yticks([0, 1] if info["width"] == 1 else [])
        ax.set_ylim(-0.25, 1.35)
        ax.grid(True, axis="x", alpha=0.3)
        # 向量值标注：变化太多就别标了（会糊成一团）
        if info["width"] > 1 and 1 <= len(changes.get(code, [])) <= 30:
            prev = None
            for x, v in changes[code]:
                if v != prev:
                    ax.text(x, 1.05, v, fontsize=7, ha="left", va="bottom", clip_on=True)
                    prev = v

    tmax = t_end_all
    axs[-1].set_xlim(t_from, t_to)
    axs[-1].set_xlabel("time (ns)")

    title = os.path.basename(vcd)
    fig.suptitle(f"{title}   ({n} signals, {t_from:g} ~ {t_to:g} ns)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    stem = os.path.splitext(os.path.basename(vcd))[0]
    out_dir = os.path.dirname(os.path.abspath(vcd))
    out = os.path.join(out_dir, f"{stem}_waveform.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)

    if not os.path.exists(out) or os.path.getsize(out) < 1000:
        _die(f"PNG 生成失败或异常小: {out}", 3)

    total = sum(len(changes.get(c, [])) for c, _ in picked)
    print(f"saved: {out}")
    print(f"  signals={n} edge_events={total} window={t_from:g}~{t_to:g}ns")
    for code, info in picked:
        print(f"  - {info['short']:<20} w={info['width']:<3} changes={len(changes.get(code, []))}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""解析 tb_counter.vcd 并画出波形图（matplotlib，简洁版）。"""
import re, sys
from collections import OrderedDict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_vcd(path):
    """返回 {net: ([(time, value)...])}，time 单位换算成 ns。"""
    with open(path) as f:
        txt = f.read()

    # 1) 收集信号引用名 -> 网名（向量名可能带 " [3:0]" 后缀）
    ref2net = {}
    net_bits = {}
    for m in re.finditer(
        r"\$var\s+\S+\s+(\d+)\s+(\S+)\s+(\S+)(?:\s*\[\d+:\d+\])?\s*\$end", txt
    ):
        width, code, name = int(m.group(1)), m.group(2), m.group(3)
        ref2net[code] = name
        net_bits[name] = width
    # 去掉层级前缀，取短名
    def short(name):
        return name.rsplit(".", 1)[-1] if "." in name else name
    ref2short = {c: short(n) for c, n in ref2net.items()}

    # 2) 时间戳与信号变化
    times = [0.0]
    cur_time = 0.0
    changes = OrderedDict()
    body = txt.split("$end", 1)[1] if "$end" in txt else txt
    # timescale 1ps -> ns
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("$") or line.startswith("#0"):
            continue
        if line.startswith("#"):
            cur_time = float(line[1:]) / 1000.0  # ps -> ns
            times.append(cur_time)
        else:
            # 单 bit: VCD 写成 "0<code>" 或 "1<code>"（值紧跟码，无空格）
            if len(line) >= 2 and line[0] in "01xzXZ" and not line.startswith("b"):
                val, code = line[0], line[1:]
                net = ref2short.get(code)
                if net:
                    changes.setdefault(net, []).append((cur_time, val))
            elif line.startswith("b") and " " in line:
                # 多 bit: b<val> <code>
                bits, code = line[1:].split(None, 1)
                net = ref2short.get(code)
                if net:
                    changes.setdefault(net, []).append((cur_time, bits))
    return ref2short, times, changes

def step_signal(net_name, times, changes):
    """把带时间戳的变化表转成阶梯波形 (t, v) 列表。"""
    xs, ys = [], []
    if net_name not in changes:
        return xs, ys
    # 从 0 到第一个变化点之间视为初值
    first = changes[net_name][0][1]
    xs.append(0.0); ys.append(first)
    for t, v in changes[net_name]:
        xs.append(t); ys.append(v)
        xs.append(t); ys.append(v)
    last_t = max(times) if times else xs[-1]
    xs.append(last_t); ys.append(ys[-1])
    return xs, ys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "verilog/tb_counter.vcd"
    _, times, changes = parse_vcd(path)

    nets = ["clk", "rst_n", "en", "count"]
    fig, axs = plt.subplots(len(nets), 1, figsize=(10, 2.6 * len(nets)), sharex=True)
    fig.suptitle("tb_counter simulation (iverilog + vvp)", fontsize=12)

    labels = {"clk": "clk", "rst_n": "rst_n", "en": "en", "count": "count[3:0]"}
    for ax, net in zip(axs, nets):
        xs, ys = step_signal(net, times, changes)
        ax.step(xs, ys, where="post", color="tab:blue", lw=1.5)
        ax.set_ylabel(labels[net])
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3)
        # count 用步进数字标注
        if net == "count" and ys:
            prev = None
            for x, y in zip(xs, ys):
                if x == 0 or y != prev:
                    ax.text(x + 0.5, 1.08, str(int(y, 2) if isinstance(y, str) and set(y) <= {"0","1"} else y),
                            fontsize=8, ha="left", va="bottom")
                    prev = y
    axs[-1].set_xlabel("time (ns)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = "verilog/run/tb_counter_waveform.png"
    fig.savefig(out, dpi=130)
    print("saved:", out)

if __name__ == "__main__":
    main()
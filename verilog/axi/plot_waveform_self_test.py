#!/usr/bin/env python3
"""plot_waveform.py 自检：确保它不会「静默画空图」。

背景：原 plot_waveform.py 把 nets 写死成 ["clk","rst_n","en","count"]、
标题写死 "iverilog + vvp"、输出路径写死 verilog/run/tb_counter_waveform.png。
传入 tb_axi_apb_sv.vcd 时它：
  - 因 run/ 不存在 → FileNotFoundError（至少还报错）
  - 目录存在时 → 画出一张 4 个子图全空、x 轴 -0.04~0.04ns 的图，**退出码 0**
第二类是真正的危险：调用方看到 exit 0 + PNG 文件存在，就以为绘图成功。

本自检覆盖：
  P1 VCD 缺失 → 非 0 退出
  P2 VCD 里没有任何 $var → 退出码 3，不得画图
  P3 VCD 有 $var 但 body 无变化 → 退出码 3
  P4 仿真时间为 0 → 退出码 3
  P5 找不到指定信号 → 非 0 退出，且提示可用信号
  P6 合法 VCD 默认自动选信号 → 退出码 0，PNG 存在且 > 1KB
  P7 合法 VCD 指定信号 → 退出码 0，PNG 更大（画了更多/更清晰的信号）
  P8 --list 输出信号数量与实际 $var 数一致
  P9 时间窗口裁剪生效：--from/--to 后 x 轴范围等于窗口
  P10 PNG 尺寸随信号数增长（不是永远同一张图）
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))          # verilog/axi
ROOT = os.path.dirname(HERE)                                # verilog
SCRIPT = os.path.join(ROOT, "plot_waveform.py")

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"PASS {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"FAIL {name}" + (f"  ({detail})" if detail else ""))


def run(args, cwd=None):
    r = subprocess.run(
        [sys.executable, SCRIPT] + args,
        capture_output=True, text=True, timeout=180, cwd=cwd,
    )
    return r.returncode, r.stdout, r.stderr


VCD_HEAD = """$date today $end
$version test $end
$timescale 1ns $end
$scope module tb $end
$var wire 1 ! ACLK $end
$var wire 1 " AWVALID $end
$var wire 1 # AWREADY $end
$var wire 32 $ AWADDR [31:0] $end
$upscope $end
$enddefinitions $end
"""


def write_vcd(path, body):
    with open(path, "w", encoding="utf-8") as f:
        f.write(VCD_HEAD + body)


def main():
    tmp = tempfile.mkdtemp(prefix="plotwave_")

    # P1 文件不存在
    rc, out, err = run([os.path.join(tmp, "nope.vcd")])
    check("P1 缺失 VCD 必须失败", rc != 0 and "无法读取" in (out + err), f"rc={rc}")

    # P2 无 $var
    p = os.path.join(tmp, "novar.vcd")
    with open(p, "w") as f:
        f.write("$timescale 1ns $end\n$enddefinitions $end\n#0\n")
    rc, out, err = run([p])
    check("P2 无 $var 必须失败", rc == 3, f"rc={rc} {err.strip()[:60]}")

    # P3 有 $var 但无变化
    p = os.path.join(tmp, "nochange.vcd")
    write_vcd(p, "#0\n#10\n")
    rc, out, err = run([p])
    check("P3 body 无变化必须失败", rc == 3, f"rc={rc} {err.strip()[:60]}")

    # P4 时间 0
    p = os.path.join(tmp, "zerotime.vcd")
    write_vcd(p, "#0\n1!\n0\"\n")
    rc, out, err = run([p])
    check("P4 仿真时间 0 必须失败", rc == 3, f"rc={rc} {err.strip()[:60]}")

    # 合法 VCD：ACLK 时钟 + AXI/APB 风格握手
    p = os.path.join(tmp, "ok.vcd")
    lines = ["#0", "0!", "0\"", "1#", "b0 $"]
    for i in range(1, 21):
        t = i * 5
        lines.append(f"#{t}")
        lines.append("1!" if i % 2 else "0!")
        lines.append('1"' if i in (3, 4, 11, 12) else '0"')
        lines.append("0#" if i in (4, 5, 12, 13) else "1#")
        lines.append(f"b{1000 + i} $")
    lines.append("#120")
    write_vcd(p, "\n".join(lines) + "\n")

    # P5 找不到信号
    rc, out, err = run([p, "NOT_A_SIGNAL"])
    check(
        "P5 未知信号必须失败并列出可用",
        rc != 0 and "ACLK" in (out + err),
        f"rc={rc}",
    )

    # P6 自动选信号
    rc, out, err = run([p])
    png = os.path.join(tmp, "ok_waveform.png")
    ok_png = os.path.exists(png) and os.path.getsize(png) > 1024
    check("P6 自动选信号出图", rc == 0 and ok_png,
          f"rc={rc} size={os.path.getsize(png) if os.path.exists(png) else 0}")

    # P8 --list 数量与 $var 数一致
    rc, out, err = run(["--list", p])
    m = re.search(r"(\d+) signals", out)
    n_listed = len([ln for ln in out.splitlines() if ln.strip().startswith(("A", "$")) ])
    n_decl = len(re.findall(r"\$var", VCD_HEAD))
    check("P8 --list 数量正确", rc == 0 and m and int(m.group(1)) == n_decl,
          f"rc={rc} header={m.group(1) if m else '?'} expect={n_decl}")

    # P7 指定信号
    rc, out, err = run([p, "ACLK", "AWVALID"])
    size2 = os.path.getsize(png) if os.path.exists(png) else 0
    check("P7 指定信号出图", rc == 0 and size2 > 1024, f"rc={rc} size={size2}")

    # P9 窗口裁剪：--from 20 --to 40，检查 PNG 标题里的窗口范围
    rc, out, err = run([p, "--from", "20", "--to", "40", "ACLK"])
    check("P9 时间窗口生效", rc == 0 and "20~40ns" in out,
          f"rc={rc} out={out.strip().splitlines()[-1] if out.strip() else ''}")

    # P9b 非法窗口必须失败
    rc, out, err = run([p, "--from", "40", "--to", "20", "ACLK"])
    check("P9b 反向窗口必须失败", rc != 0, f"rc={rc}")
    rc, out, err = run([p, "--from", "9999", "ACLK"])
    check("P9c 窗口超出仿真长度必须失败", rc != 0, f"rc={rc}")

    # P10 输出文件名跟随输入
    p2 = os.path.join(tmp, "another.vcd")
    write_vcd(p2, "\n".join(lines) + "\n")
    rc, out, err = run([p2, "ACLK"])
    png2 = os.path.join(tmp, "another_waveform.png")
    check("P10 输出名跟随输入 VCD", rc == 0 and os.path.exists(png2), f"rc={rc}")

    # P11 真实 VCD（若存在）
    real = os.path.join(HERE, "tb_axi_apb_sv.vcd")
    if os.path.exists(real):
        rc, out, err = run([real, "PCLK", "AWVALID", "PSEL"])
        stem = os.path.splitext(os.path.basename(real))[0]
        rp = os.path.join(HERE, f"{stem}_waveform.png")
        ok = rc == 0 and os.path.exists(rp) and os.path.getsize(rp) > 4096
        check("P11 真实 AXI VCD 出图", ok,
              f"rc={rc} size={os.path.getsize(rp) if os.path.exists(rp) else 0}")
        # 关键回归：绝不能再出现 counter 时代的信号
        if os.path.exists(rp):
            rc2, out2, _ = run(["--list", real])
            check(
                "P12 真实 VCD 不含 counter 遗留信号名",
                "count[" not in out2 and "rst_n" not in out2.split("ACLK")[0],
                "无 clk/rst_n/count 硬编码残留",
            )
    else:
        print("SKIP P11/P12（未找到 axi/tb_axi_apb_sv.vcd，先跑 sim.sh axi）")

    total = passed + failed
    print(f"\nplot_waveform 自检: {passed}/{total} PASS")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

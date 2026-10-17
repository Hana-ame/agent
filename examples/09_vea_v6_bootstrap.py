"""示例 9：自举 —— 用 DAG 从零生成一个 DAG 框架，并证明它能生成自己.

这是本次任务的核心演示，分四步：

1. **生成**：``create_bootstrap_dag()`` 是一个标准 V6 DAG。喂入一段描述（或留空用内置 SPEC），
   它产出 ``tag="result"`` 的**完整框架源码**（minivea，约 70KB 单文件、零依赖）。
   拓扑：``input → framework_spec → framework_code → fixed_point → result``，
   最后一条边需要两个输入 tag，**必然**由预设边把两路拼成一条 —— 这是预设边在真实任务里的用法。

2. **自验**：生成的框架内嵌自身 SPEC（base64）与自身源码（base64），
   调它的 ``generate_framework()`` 能**逐字节**重建自己（quine 固定点）。

3. **再生成一个新框架**：给它一份**新的 SPEC**（同一段提示词、不同描述），
   它照样产出另一份完整、可运行、自稳定的框架 —— 这是「框架生成框架」的真正证明，
   而不是「一次性的代码模板」。

4. **落地**：把产出写到 ``build/`` 下并用子进程真正 import 一次，确保是可用代码。

运行方式：
    python3 examples/09_vea_v6_bootstrap.py
"""

import hashlib
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vea_v6 import (
    FRAMEWORK_SPEC,
    bootstrap,
    create_bootstrap_dag,
    render_framework_code,
    verify_fixed_point,
    verify_self_reproduction,
)


def show(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    # =========================================================================
    # 1. 只用 DAG 跑一遍（不碰 LLM、不碰网络），看拓扑怎么流转
    # =========================================================================
    show("1. 自举 DAG 的拓扑与流转")
    dag = create_bootstrap_dag()
    dag.send({"tags": ["input"], "content": "", "gen": 0})  # 空内容 = 用内置 FRAMEWORK_SPEC
    print("  边:")
    for e in dag.edges:
        print(f"    layer={e.layer} {e.input_tags} -> {e.output_tags}"
              f"  processor={'<callable>' if callable(e.processor) else e.processor!r}")

    step = 0
    while not dag.has_result() and step < 8:
        step += 1
        new = dag.run()
        print(f"  第 {step} 次 run() 新增 {len(new)} 条: "
              f"{[(i['tags'], len(i['content'])) for i in new]}")

    code1 = dag.get_result()
    print(f"\n  tag=result 产出 {len(code1)} 字节，sha256[:16]={digest(code1)}")
    print(f"  context 全量 tag 序列: {[i['tags'] for i in dag.context]}")
    preset_items = [i for i in dag.context if i.get("_preset_synthesized")]
    print(f"  预设边合成条目数: {len(preset_items)}（顶层汇聚那一条边必然触发）")

    # =========================================================================
    # 2. 自验：固定点 + 生成物自我复现
    # =========================================================================
    show("2. 固定点校验")
    ok, reason = verify_fixed_point(FRAMEWORK_SPEC)
    print(f"  同一 SPEC 两次渲染逐字节相同: {'✓' if ok else '✗ ' + reason}")

    ok2, reason2 = verify_self_reproduction(code1)
    print(f"  把生成物当模块加载、调它自己的 generate_framework(): "
          f"{'✓ 逐字节复现自身' if ok2 else '✗ ' + reason2}")

    # =========================================================================
    # 3. 同一段提示词、不同描述 → 生成另一个新框架
    # =========================================================================
    show("3. 再生成一个新框架（换个描述 / SPEC）")
    desc = "一个面向多模态检索的 tag 驱动 DAG 框架，要求边可按 layer 并行、结果合并写回 contents"
    res = bootstrap(desc)
    code2 = res.framework_code
    print(f"  描述: {desc}")
    print(f"  产出 {len(code2)} 字节，sha256[:16]={digest(code2)}")
    print(f"  与第一份相同: {code2 == code1}（应为 False：SPEC 变了）")
    print(f"  新框架自身稳定: {'✓' if res.stable else '✗ ' + res.reason}")
    ok3, reason3 = verify_self_reproduction(code2)
    print(f"  新框架也能生成自己: {'✓' if ok3 else '✗ ' + reason3}")

    # 反复生成同一描述，必须逐字节一致（稳定性）
    again = bootstrap(desc).framework_code
    print(f"  同一描述连跑两次: {'✓ 逐字节一致' if again == code2 else '✗ 不一致'}")

    # =========================================================================
    # 4. 落地：写成真实文件并用子进程 import，证明是可运行代码
    # =========================================================================
    show("4. 落地成文件并真实 import")
    out_dir = ROOT / "build"
    out_dir.mkdir(exist_ok=True)
    target = out_dir / "minivea_generated.py"
    target.write_text(code1, encoding="utf-8")
    print(f"  已写出: {target}")

    probe_path = out_dir / "_smoke_probe.py"
    probe_path.write_text(
        "import sys\n"
        f"sys.path.insert(0, r'{out_dir}')\n"
        "import minivea_generated as m\n"
        "dag = m.DAG('smoke')\n"
        "dag.add_edge(['input'], lambda s: s[::-1], ['result'], layer=1)\n"
        "dag.send({'tags': ['input'], 'content': 'hello', 'gen': 0})\n"
        "dag.run()\n"
        "print('  smoke result:', dag.get_result())\n"
        "print('  SPEC version:', m.FRAMEWORK_SPEC['version'])\n"
        "src = open(r'"
        + str(target)
        + "', encoding='utf-8').read()\n"
        "print('  self-repro:', m.generate_framework() == src)\n",
        encoding="utf-8",
    )
    print("  子进程 import 结果:", flush=True)
    try:
        subprocess.run([sys.executable, str(probe_path)], check=True)
    finally:
        probe_path.unlink(missing_ok=True)

    show("完成")
    print(f"  产物: {target}")
    print("  生成物可用: `python3 build/minivea_generated.py` 无副作用（纯库模块）")


if __name__ == "__main__":
    main()

"""Master Parallel Runner for All DAG Examples.

Discovers all standalone DAG examples in `examples/ex*.py`, executes them concurrently
in parallel using multi-processing / sub-processes, validates all assertions, and outputs
a unified benchmark execution report.
"""

from __future__ import annotations

import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path


def run_single_example(script_path: Path) -> dict:
    """Run a single example script in an isolated subprocess."""
    start_time = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=40,
        )
        duration = time.time() - start_time
        return {
            "name": script_path.name,
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "duration": duration,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {
            "name": script_path.name,
            "success": False,
            "returncode": -1,
            "duration": time.time() - start_time,
            "stdout": "",
            "stderr": "Execution timed out (> 40s)",
        }
    except Exception as ex:
        return {
            "name": script_path.name,
            "success": False,
            "returncode": -2,
            "duration": time.time() - start_time,
            "stdout": "",
            "stderr": str(ex),
        }


def run_all_parallel(max_workers: int = 8) -> bool:
    examples_dir = Path(__file__).resolve().parent
    scripts = sorted(list(examples_dir.glob("ex*.py")))

    print("=" * 80)
    print(f"🚀 开始并发并行运行 {len(scripts)} 个 DAG 示例脚本 (Workers: {max_workers})...")
    print("=" * 80)

    results = []
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_script = {executor.submit(run_single_example, s): s for s in scripts}
        for future in concurrent.futures.as_completed(future_to_script):
            res = future.result()
            results.append(res)
            status_icon = "✅ PASS" if res["success"] else "❌ FAIL"
            print(f"  [{status_icon}] {res['name']:<45} ({res['duration']:.2f}s)")
            if not res["success"]:
                print(f"       ERROR: {res['stderr'] or res['stdout']}")

    total_time = time.time() - t_start
    all_passed = all(r["success"] for r in results)

    print("\n" + "=" * 80)
    print(f"📊 并行测试统计报告:")
    print(f"  总示例数: {len(results)}")
    print(f"  通过数量: {sum(1 for r in results if r['success'])}")
    print(f"  失败数量: {sum(1 for r in results if not r['success'])}")
    print(f"  总并发耗时: {total_time:.2f} 秒 (平均单测耗时: {sum(r['duration'] for r in results)/len(results):.2f}s)")
    print("=" * 80)

    if all_passed:
        print("🎉 完美！全部 20+ 个 DAG 架构示例脚本在并行并发执行中 100% 通过验证！")
    else:
        print("⚠️ 存在未通过的示例脚本，请检查上方日志。")

    return all_passed


if __name__ == "__main__":
    success = run_all_parallel()
    sys.exit(0 if success else 1)

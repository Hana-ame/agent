"""集成测试 — 真实调用 opencode 运行模型，记录 success/fail 并查询计数"""
import model_tracker as fm
import opencode


def test_real_call_qwen3_success():
    """真实调用 Qwen3-8B 运行 1+1=?，记录 success/fail 并查询计数"""
    model = "siliconflow-cn/Qwen/Qwen3-8B"
    print(f"\n测试: 真实调用 {model} 运行 '1+1=?'")

    # 1. 拉取 models 列表，确认模型存在
    models = opencode.models(filter_free=True)
    print(f"  models 列表共 {len(models)} 个")
    assert model in models, f"{model} 不在 models 列表中"

    # 2. 运行提问
    print(f"  执行 opencode.run('1+1=?', model={model})")
    result = opencode.run("1+1=?", model=model, timeout=60)
    print(f"  返回: {result}")

    # 3. 根据返回记录 success 或 fail
    if result.get("success"):
        fm.record_call(model, success=True)
        print("  记录: success=True")
    else:
        fm.record_call(model, success=False)
        print("  记录: success=False")

    # 4. 查询计数
    stats = fm.get_stats(model)
    print(f"  统计: {stats}")
    assert stats is not None
    assert stats["calls"] >= 1
    assert stats["successes"] + stats["failures"] == stats["calls"]

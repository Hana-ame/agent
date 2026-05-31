"""model_tracker 模块测试 — 用 opencode.models() 真实模型列表测试"""
import pytest
import db
import model_tracker as fm
import opencode

# 从 opencode models 获取真实模型列表
ALL_MODELS = opencode.models(filter_free=False)
# 选一个 opencode 模型、一个 nvidia 模型做测试
M_OPEN = next(m for m in ALL_MODELS if m.startswith("opencode/"))
M_NVIDIA = next(m for m in ALL_MODELS if m.startswith("nvidia/"))


def _clean():
    """清理测试数据"""
    conn = db.get_conn()
    for m in [M_OPEN, M_NVIDIA]:
        conn.execute("DELETE FROM usage WHERE model = ?", (m,))
        conn.execute("DELETE FROM models WHERE model = ?", (m,))
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def setup_teardown():
    _clean()
    yield
    _clean()


def test_list_free_models_adds_usage_record():
    """list_free_models() 应为每个没有 usage 记录的模型自动添加 (0,0,0,0,0)"""
    print("\n测试: list_free_models 自动补全 usage 记录")
    print(f"  测试模型: {M_OPEN}")

    conn = fm._get_conn()
    conn.execute("INSERT INTO models (model, provider) VALUES (?, ?)", (M_OPEN, "opencode"))
    conn.commit()
    conn.close()

    conn = db.get_conn()
    before = conn.execute("SELECT * FROM usage WHERE model = ?", (M_OPEN,)).fetchone()
    conn.close()
    print(f"  调用前 usage: {before}")
    assert before is None

    result = fm.list_free_models()
    print(f"  list_free_models 返回: {len(result)} 个模型")
    assert M_OPEN in result

    conn = db.get_conn()
    after = conn.execute("SELECT * FROM usage WHERE model = ?", (M_OPEN,)).fetchone()
    conn.close()
    print(f"  调用后 usage: {after}")
    assert after is not None
    # (model, calls=0, successes=0, failures=0, good=0, bad=0)
    assert after[1] == 0
    assert after[2] == 0
    assert after[3] == 0
    assert after[4] == 0
    assert after[5] == 0


def test_list_free_models_does_not_overwrite_existing():
    """已有 usage 记录时 list_free_models 不覆盖"""
    print("\n测试: 已有 usage 时不覆盖")
    print(f"  测试模型: {M_NVIDIA}")

    conn = fm._get_conn()
    conn.execute("INSERT INTO models (model, provider) VALUES (?, ?)", (M_NVIDIA, "nvidia"))
    conn.execute("INSERT INTO usage (model, calls, successes, failures, good, bad) VALUES (?, 5, 3, 2, 4, 1)",
                 (M_NVIDIA,))
    conn.commit()
    conn.close()

    fm.list_free_models()

    stats = fm.get_stats(M_NVIDIA)
    print(f"  调用后 stats: {stats}")
    assert stats is not None
    assert stats["calls"] == 5
    assert stats["successes"] == 3
    assert stats["failures"] == 2
    assert stats["good"] == 4
    assert stats["bad"] == 1


def test_record_call_updates_usage():
    """record_call 应正确累加调用统计"""
    print("\n测试: record_call 累加统计")
    print(f"  测试模型: {M_OPEN}")

    conn = fm._get_conn()
    conn.execute("INSERT INTO models (model, provider) VALUES (?, ?)", (M_OPEN, "opencode"))
    conn.commit()
    conn.close()

    fm.record_call(M_OPEN, success=True, good=1, bad=0)
    fm.record_call(M_OPEN, success=True, good=1, bad=0)
    fm.record_call(M_OPEN, success=False, good=0, bad=1)

    stats = fm.get_stats(M_OPEN)
    print(f"  统计结果: {stats}")
    assert stats is not None
    assert stats == {"model": M_OPEN, "calls": 3, "successes": 2, "failures": 1, "good": 2, "bad": 1}


def test_record_call_good_bad():
    """record_call 的 good/bad 参数独立于 success"""
    print("\n测试: record_call good/bad 计数")
    print(f"  测试模型: {M_NVIDIA}")

    conn = fm._get_conn()
    conn.execute("INSERT INTO models (model, provider) VALUES (?, ?)", (M_NVIDIA, "nvidia"))
    conn.commit()
    conn.close()

    # 成功调用但回答质量差
    fm.record_call(M_NVIDIA, success=True, good=0, bad=1)
    # 成功调用且回答质量好
    fm.record_call(M_NVIDIA, success=True, good=1, bad=0)
    # 失败调用
    fm.record_call(M_NVIDIA, success=False, good=0, bad=0)

    stats = fm.get_stats(M_NVIDIA)
    print(f"  统计结果: {stats}")
    assert stats is not None
    assert stats["calls"] == 3
    assert stats["successes"] == 2
    assert stats["failures"] == 1
    assert stats["good"] == 1
    assert stats["bad"] == 1


def test_get_stats_specific_model():
    """查询指定模型的统计"""
    print(f"\n测试: 查询指定模型 {M_NVIDIA}")

    conn = fm._get_conn()
    conn.execute("INSERT INTO models (model, provider) VALUES (?, ?)", (M_NVIDIA, "nvidia"))
    conn.execute("INSERT INTO usage (model, calls, successes, failures, good, bad) VALUES (?, 10, 8, 2, 7, 1)",
                 (M_NVIDIA,))
    conn.commit()
    conn.close()

    stats = fm.get_stats(M_NVIDIA)
    print(f"  结果: {stats}")
    assert stats is not None
    assert stats == {"model": M_NVIDIA, "calls": 10, "successes": 8, "failures": 2, "good": 7, "bad": 1}


def test_get_stats_nonexistent():
    """查询不存在的模型 → None"""
    print("\n测试: 查询不存在的模型")
    result = fm.get_stats("xyz_999_nonexistent")
    print(f"  结果: {result}")
    assert result is None


def test_get_stats_all():
    """get_stats() 返回全部模型列表，插入的模型应在其中"""
    print("\n测试: 查询全部模型统计")

    conn = fm._get_conn()
    conn.execute("INSERT INTO models (model, provider) VALUES (?, ?)", (M_OPEN, "opencode"))
    conn.execute("INSERT INTO models (model, provider) VALUES (?, ?)", (M_NVIDIA, "nvidia"))
    conn.execute("INSERT INTO usage (model, calls, successes, failures, good, bad) VALUES (?, ?, ?, ?, ?, ?)",
                 (M_OPEN, 7, 6, 1, 5, 1))
    conn.commit()
    conn.close()

    result = fm.get_stats()
    print(f"  共 {len(result)} 个模型")

    open_stats = next(s for s in result if s["model"] == M_OPEN)
    nvidia_stats = next(s for s in result if s["model"] == M_NVIDIA)
    print(f"  {M_OPEN}: {open_stats}")
    print(f"  {M_NVIDIA}: {nvidia_stats}")

    assert open_stats == {"model": M_OPEN, "calls": 7, "successes": 6, "failures": 1, "good": 5, "bad": 1}
    assert nvidia_stats == {"model": M_NVIDIA, "calls": 0, "successes": 0, "failures": 0, "good": 0, "bad": 0}


def test_record_call_auto_insert():
    """record_call 对不存在的模型自动插入到 models 表"""
    print("\n测试: record_call 自动插入新模型")
    print(f"  测试模型: {M_OPEN}")
    fm.record_call(M_OPEN, success=True, good=1, bad=0)
    stats = fm.get_stats(M_OPEN)
    print(f"  结果: {stats}")
    assert stats is not None
    assert stats == {"model": M_OPEN, "calls": 1, "successes": 1, "failures": 0, "good": 1, "bad": 0}

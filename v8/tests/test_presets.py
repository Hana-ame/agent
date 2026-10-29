"""预设检查 DAG：加载、状态渲染、构图、聚合脚本。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from presets import build_dag, load_presets, render_state  # noqa: E402
from presets.state import load_conv  # noqa: E402
from presets import __main__ as cli  # noqa: E402

EX = V8 / "presets" / "examples"
PRESET_DIR = EX / "presets"
CONV = EX / "conv_example.json"


def test_load_presets():
    presets = load_presets(PRESET_DIR)
    assert len(presets) == 3
    ids = [p.id for p in presets]
    assert len(ids) == len(set(ids))
    assert all(p.tag.startswith("check_") for p in presets)
    assert all(p.system_prompt for p in presets)


def test_render_state_modes():
    conv = load_conv(CONV)
    full = render_state(conv, mode="full")
    last = render_state(conv, mode="last")
    mainline = render_state(conv, mode="mainline")
    assert "会话元信息" in full
    assert "[工具]" in full          # full 含全文
    assert "最近一轮" not in last
    assert "[用户]" in last and "[助手]" not in last   # last=最后一条
    assert "[工具]" not in mainline  # mainline 过滤工具
    assert len(full) > len(mainline)
    # last 模式只含最后一条（用户）
    assert "[助手]" not in last


def test_build_dag():
    presets = load_presets(PRESET_DIR)
    cfg = build_dag(presets)
    assert cfg["seed_tag"] == "state"
    assert cfg["result_tag"] == "result"
    kinds = [e["kind"] for e in cfg["edges"]]
    assert kinds.count("pi") == 3
    assert kinds[-1] == "script"
    agg = cfg["edges"][-1]
    assert agg["output_tags"] == ["result"]
    assert agg["input_tags"] == [p.tag for p in presets]
    assert cfg["pi"]["model"] == "sensenova-6.8-flash-lite"
    for e, p in zip(cfg["edges"][:3], presets):
        assert "{tag:state}" in e["prompt"]
        assert p.system_prompt.strip()[:20] in e["prompt"]


def test_aggregate_script():
    env = dict(os.environ)
    env["DAG8_INPUTS"] = json.dumps({
        "check_a": json.dumps({"satisfied": True, "score": 4, "reason": "命中"}),
        "check_b": json.dumps({"satisfied": False, "score": 1, "reason": "没有"}),
        "check_c": "不是 JSON",
    }, ensure_ascii=False)
    out = subprocess.run([sys.executable, str(V8 / "presets" / "scripts" / "aggregate.py")],
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["checked"] == 3
    assert data["satisfied"] == 1
    assert data["hits"][0]["preset"] == "a"
    bad = [r for r in data["all"] if r["preset"] == "c"]
    assert bad and bad[0]["satisfied"] is False


def test_cli_dry(capsys):
    rc = cli.main(["--presets", str(PRESET_DIR), "--conv", str(CONV),
                   "--state", "mainline", "--dry"])
    assert rc == 0
    out = capsys.readouterr().out
    cfg = json.loads(out)
    assert cfg["name"].startswith("preset_check_")

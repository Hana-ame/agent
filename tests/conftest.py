"""pytest 全局配置：路径注入、标记注册、外网门控。

设计原则（2026-10-21）：
1. 默认 `pytest tests/` **必须离线跑通** —— CI 稳定，不依赖 stage1st.com / LLM 端点。
2. 需要真实外网的用例一律打 `@pytest.mark.network`，只有显式 `--run-network` 才执行。
   早先 test_stage1st_dag 直接抓真站，没网的环境里是「挂住」而不是「失败」。
3. 所有 DAG 产物写入 tmp_path，不污染工作树。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="执行 @pytest.mark.network（真实外网）用例",
    )


def pytest_configure(config: pytest.Config) -> None:
    for line in (
        "network: 需要真实外网（stage1st.com 等），默认跳过",
        "llm: 需要真实 LLM 端点，默认跳过",
        "slow: 单次执行超过 10s",
    ):
        config.addinivalue_line("markers", line)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="需要真实外网/端点：加 --run-network 才跑")
    for item in items:
        if "network" in item.keywords or "llm" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def fake_models() -> dict:
    """不落盘、不带真 key 的模型配置（指向必然连不上的本地端口）。"""
    return {"mock": {"api": "http://127.0.0.1:9/v1", "key": "not-needed", "model": "mock"}}


@pytest.fixture
def clean_specs():
    """测试结束把 DAG_SPECS 恢复原样，避免注册表被测试污染。"""
    from dags.registry import DAG_SPECS

    snapshot = dict(DAG_SPECS)
    yield DAG_SPECS
    DAG_SPECS.clear()
    DAG_SPECS.update(snapshot)

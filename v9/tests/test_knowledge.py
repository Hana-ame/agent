from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v9.knowledge import Knowledge  # noqa: E402


def test_claim_requires_source_or_derivation():
    k = Knowledge()
    k.add_entity("a")
    k.add_entity("b")
    with pytest.raises(ValueError):
        k.add_claim("a", "rel", "b", "无来源")
    cid = k.add_claim("a", "rel", "b", "有来源", source_url="https://x")
    assert cid > 0
    k.close()


def test_unexplored_pairs_skip_linked_and_explored():
    k = Knowledge()
    for name in ("a", "b", "c"):
        k.add_entity(name)
    k.add_claim("a", "rel", "b", "a-b", source_url="https://x")
    k.mark_explored("a", "c", "none")
    assert k.unexplored_pairs(10) == [("b", "c")]
    k.close()

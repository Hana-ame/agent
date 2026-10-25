from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v9.expand import expand  # noqa: E402
from v9.knowledge import Knowledge  # noqa: E402


def _fetch(url):
    return {"url": url, "text": json.dumps({
        "query": {"pages": {"1": {"links": [
            {"title": "New A"}, {"title": "Existing"}, {"title": "New B"},
        ]}}}
    })}


def test_expand_adds_new_entities_and_skips_existing():
    k = Knowledge()
    k.add_entity("Existing")
    added = expand(k, _fetch, "Seed", limit=2)
    assert added == ["New A", "New B"]
    names = {e["name"] for e in k.entities()}
    assert names == {"Existing", "New A", "New B"}
    k.close()


def test_expand_respects_limit():
    k = Knowledge()
    added = expand(k, _fetch, "Seed", limit=1)
    assert added == ["New A"]
    k.close()

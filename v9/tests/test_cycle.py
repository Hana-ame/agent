from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V8 = ROOT / "v8"
for p in (str(ROOT), str(V8)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dag8.__main__ import main as dag8_main  # noqa: E402
from v9.knowledge import Knowledge  # noqa: E402

CFG = ROOT / "v9" / "configs" / "cycle.json"
CFG_DERIVE = ROOT / "v9" / "configs" / "derive.json"
COMMIT = ROOT / "v9" / "scripts" / "commit.py"
COMMIT_DERIVED = ROOT / "v9" / "scripts" / "commit_derived.py"


def _run_commit(tmp: Path, claim: dict) -> dict:
    env = dict(os.environ)
    env["V9_DB"] = str(tmp / "k.sqlite3")
    env["DAG8_INPUTS"] = json.dumps({
        "claim_json": json.dumps(claim, ensure_ascii=False),
        "a": "A", "b": "B",
        "url_a": "https://en.wikipedia.org/api/rest_v1/page/summary/A",
        "url_b": "https://en.wikipedia.org/api/rest_v1/page/summary/B",
        "ev_a": "A text", "ev_b": "B text",
    }, ensure_ascii=False)
    out = subprocess.run([sys.executable, str(COMMIT)], cwd=str(ROOT),
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_cycle_config_lints():
    assert dag8_main(["lint", str(CFG)]) == 0


def test_commit_script_writes_claim(tmp_path):
    res = _run_commit(tmp_path, {
        "relation": "rel", "statement": "A 与 B 有关系",
        "confidence": 0.8,
        "evidence_url": "https://en.wikipedia.org/wiki/A"})
    assert res["status"] == "claim"
    k = Knowledge(tmp_path / "k.sqlite3")
    assert len(k.claims()) == 1
    assert k.claims()[0]["source_url"].endswith("/summary/A")
    k.close()


def test_commit_script_rejects_bad_source(tmp_path):
    res = _run_commit(tmp_path, {
        "relation": "rel", "statement": "编的", "confidence": 0.9,
        "evidence_url": "https://evil"})
    assert res["status"] == "rejected"
    k = Knowledge(tmp_path / "k.sqlite3")
    assert k.claims() == []
    k.close()


def test_derive_config_lints():
    assert dag8_main(["lint", str(CFG_DERIVE)]) == 0


def test_commit_derived_script(tmp_path):
    env = dict(os.environ)
    env["V9_DB"] = str(tmp_path / "k.sqlite3")
    env["DAG8_INPUTS"] = json.dumps({
        "claim_json": json.dumps({
            "relation": "uses", "statement": "A 使用 C", "confidence": 0.7}),
        "c1_id": "1", "c2_id": "2", "a": "A", "c": "C",
    })
    out = subprocess.run([sys.executable, str(COMMIT_DERIVED)], cwd=str(ROOT),
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["status"] == "claim"
    k = Knowledge(tmp_path / "k.sqlite3")
    c = k.claims()[0]
    assert c["source_url"] is None
    assert json.loads(c["derived_from"]) == [1, 2]
    k.close()


def test_find_chain():
    from v9.cycle import find_chain

    k = Knowledge()
    k.add_entity("A")
    k.add_entity("B")
    k.add_entity("C")
    k.add_claim("A", "uses", "B", "A 使用 B", source_url="https://x/1")
    k.add_claim("B", "part_of", "C", "B 属于 C", source_url="https://x/2")
    chain = find_chain(k)
    assert chain is not None
    assert chain[:3] == ("A", "B", "C")
    k.close()

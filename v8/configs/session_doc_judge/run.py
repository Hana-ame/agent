#!/usr/bin/env python3
"""入口：评判 session + 结论文档。

python3 v8/configs/session_doc_judge/run.py \
  --session <id> --doc reports/session_xxx_report.md
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (str(ROOT), str(ROOT / "v8")):
    if p not in sys.path:
        sys.path.insert(0, p)
import dag8  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = HERE / "dag.json"
DEFAULT_SESSION = "session-adf80908-d20f-4ffa-b00f-952b9b4bf15d"
DEFAULT_DOC = ROOT / "reports" / "session_adf80908-d20_report.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=DEFAULT_SESSION)
    ap.add_argument("--doc", default=str(DEFAULT_DOC))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rpm", type=int, default=0)
    args = ap.parse_args()
    doc = Path(args.doc)
    if not doc.is_absolute():
        doc = (ROOT / doc).resolve()
    res = dag8.run(
        str(CFG), seed_content=args.session,
        extra_seeds={"doc_path": str(doc)},
        db_path=str(ROOT / "v8" / "data" / "session_doc_judge.sqlite3"),
        max_workers=args.workers, rpm=None, use_cache=False, max_seconds=300,
    )
    print(res.summary())
    if res.result:
        try:
            out = json.loads(res.result)
            print(json.dumps({k: out.get(k) for k in ("verdict", "score", "report", "findings")},
                             ensure_ascii=False, indent=2))
        except ValueError:
            print(res.result)
    return 0 if res.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

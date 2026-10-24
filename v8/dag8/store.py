"""v8 存储层：所有内容与执行记录都落在 sqlite3。

三张表：

| 表 | 存什么 |
|---|---|
| `runs` | 一次 DAG 运行的元信息（配置路径、种子、起止时间、结果指针） |
| `contents` | 每个 edge 的输入/输出内容（tag 列表 + 正文 + 代次） |
| `edges` | 每次 edge 执行的记录（模型、耗时、状态、错误、产出的 content id） |

设计取向：**一次运行 = 一个 run_id**，所有内容都挂在它下面。
同一个 DAG 跑多次会在同一个库里留下多组 run，可以直接对比。
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "dag8.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    dag_name      TEXT NOT NULL,
    config_path   TEXT NOT NULL,
    seed_tag      TEXT,
    seed_content  TEXT,
    started_at    REAL NOT NULL,
    finished_at   REAL,
    status        TEXT NOT NULL DEFAULT 'running',
    error         TEXT,
    result_tag    TEXT,
    result_content_id INTEGER
);

CREATE TABLE IF NOT EXISTS contents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    tags        TEXT NOT NULL,          -- JSON array
    content     TEXT NOT NULL,
    gen         INTEGER NOT NULL DEFAULT 0,
    source_edge TEXT,
    created_at  REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_contents_run ON contents(run_id);
CREATE INDEX IF NOT EXISTS idx_contents_edge ON contents(run_id, source_edge);

CREATE TABLE IF NOT EXISTS edges (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    edge_id      TEXT NOT NULL,
    layer        INTEGER NOT NULL,
    input_tags   TEXT NOT NULL,         -- JSON array
    output_tags  TEXT NOT NULL,         -- JSON array
    model        TEXT,
    prompt_chars INTEGER,
    status       TEXT NOT NULL,         -- ok | skipped | error
    error        TEXT,
    elapsed      REAL,
    content_id   INTEGER,
    cached       INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_edges_run ON edges(run_id);
"""


class Store:
    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """给旧库补列（CREATE TABLE IF NOT EXISTS 不会改已存在的表）。"""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(edges)")}
        if "cached" not in cols:
            self.conn.execute("ALTER TABLE edges ADD COLUMN cached INTEGER NOT NULL DEFAULT 0")

    # -- runs ---------------------------------------------------------------
    def new_run(
        self,
        dag_name: str,
        config_path: str,
        seed_tag: str,
        seed_content: str,
        run_id: Optional[str] = None,
    ) -> str:
        run_id = run_id or uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO runs (run_id, dag_name, config_path, seed_tag, seed_content, started_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, dag_name, config_path, seed_tag, seed_content, time.time()),
        )
        self.conn.commit()
        return run_id

    def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        result_tag: Optional[str] = None,
        result_content_id: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at=?, status=?, error=?, result_tag=?, result_content_id=? "
            "WHERE run_id=?",
            (time.time(), status, error, result_tag, result_content_id, run_id),
        )
        self.conn.commit()

    # -- contents -----------------------------------------------------------
    def add_content(
        self,
        run_id: str,
        tags: List[str],
        content: str,
        *,
        gen: int = 0,
        source_edge: Optional[str] = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO contents (run_id, tags, content, gen, source_edge, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, json.dumps(list(tags), ensure_ascii=False), content, gen, source_edge, time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def contents_for_tag(self, run_id: str, tag: str) -> List[Dict[str, Any]]:
        """取带某 tag 的全部内容（按 gen、id 升序）。"""
        try:
            rows = self.conn.execute(
                "SELECT * FROM contents WHERE run_id=? AND EXISTS ("
                "SELECT 1 FROM json_each(contents.tags) WHERE value=?) "
                "ORDER BY gen, id", (run_id, tag)
            ).fetchall()
        except sqlite3.OperationalError:  # 旧 SQLite 没有 JSON1，退回全表扫描
            rows = self.conn.execute(
                "SELECT * FROM contents WHERE run_id=? ORDER BY gen, id", (run_id,)
            ).fetchall()
            rows = [r for r in rows if tag in json.loads(r["tags"])]
        out = []
        for r in rows:
            tags = json.loads(r["tags"])
            if tag in tags:
                out.append(
                    {
                        "id": r["id"],
                        "tags": tags,
                        "content": r["content"],
                        "gen": r["gen"],
                        "source_edge": r["source_edge"],
                    }
                )
        return out

    def all_contents(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM contents WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
        return [
            {
                "id": r["id"],
                "tags": json.loads(r["tags"]),
                "content": r["content"],
                "gen": r["gen"],
                "source_edge": r["source_edge"],
            }
            for r in rows
        ]

    # -- edges --------------------------------------------------------------
    def log_edge(
        self,
        run_id: str,
        edge_id: str,
        layer: int,
        input_tags: List[str],
        output_tags: List[str],
        *,
        model: Optional[str] = None,
        prompt_chars: int = 0,
        status: str = "ok",
        error: Optional[str] = None,
        elapsed: Optional[float] = None,
        content_id: Optional[int] = None,
        cached: bool = False,
    ) -> None:
        self.conn.execute(
            "INSERT INTO edges (run_id, edge_id, layer, input_tags, output_tags, model, "
            "prompt_chars, status, error, elapsed, content_id, cached, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                edge_id,
                layer,
                json.dumps(list(input_tags), ensure_ascii=False),
                json.dumps(list(output_tags), ensure_ascii=False),
                model,
                prompt_chars,
                status,
                error,
                elapsed,
                content_id,
                1 if cached else 0,
                time.time(),
            ),
        )
        self.conn.commit()

    # -- 查询 ---------------------------------------------------------------
    def edge_latest(self, run_id: str) -> Dict[str, Dict[str, Any]]:
        """每条边的最新一次记录（同一 run 内可能有多条尝试）。"""
        rows = self.conn.execute(
            "SELECT * FROM edges WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
        latest: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            latest[r["edge_id"]] = dict(r)
        return latest

    def delete_edges(self, run_id: str, edge_ids: List[str]) -> None:
        ids = [str(x) for x in edge_ids]
        if not ids:
            return
        q = ",".join("?" * len(ids))
        self.conn.execute(
            f"DELETE FROM edges WHERE run_id=? AND edge_id IN ({q})",
            (run_id, *ids),
        )
        self.conn.commit()

    def resume_run(self, run_id: str) -> None:
        self.conn.execute(
            "UPDATE runs SET status='running', finished_at=NULL, error=NULL WHERE run_id=?",
            (run_id,),
        )
        self.conn.commit()

    def run_info(self, run_id: str) -> Dict[str, Any]:
        r = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not r:
            raise KeyError(f"未知 run: {run_id}")
        return dict(r)

    def edge_records(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM edges WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def abort_run(self, run_id: str) -> bool:
        """把卡在 running 的 run 标为 interrupted（供 --resume-last 接管）。"""
        cur = self.conn.execute(
            "UPDATE runs SET status='interrupted', finished_at=?, error=? "
            "WHERE run_id=? AND status='running'",
            (time.time(), "手动标记中断", run_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def prune(self, keep: int = 20, *, dry_run: bool = True) -> int:
        """删除除最近 keep 个之外的所有 run（含 contents/edges）。"""
        keep = max(0, int(keep))
        rows = self.conn.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC"
        ).fetchall()
        old = [r["run_id"] for r in rows[keep:]]
        if dry_run or not old:
            return len(old)
        for rid in old:
            self.conn.execute("DELETE FROM contents WHERE run_id=?", (rid,))
            self.conn.execute("DELETE FROM edges WHERE run_id=?", (rid,))
            self.conn.execute("DELETE FROM runs WHERE run_id=?", (rid,))
        self.conn.commit()
        return len(old)

    def result(self, run_id: str, tag: str) -> Optional[str]:
        items = self.contents_for_tag(run_id, tag)
        return items[-1]["content"] if items else None

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

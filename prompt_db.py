"""
Prompt 执行数据库。

表结构 prompts:
  id        INTEGER PK  自增
  context   TEXT         JSON 数组，包含其他行的 id（可为空）
  agent     TEXT         agent 名称
  model     TEXT         模型名
  prompt    TEXT         输入 prompt 文本
  response  TEXT         运行结果
  log       TEXT         JSON {input_tokens, output_tokens, elapsed_ms, timestamp}
  status    TEXT         pending / done / failed
  score     REAL         质量评分 0.0~1.0（由 judge 给出）
  elo       REAL         ELO 分数（初始 1500）
"""

import json
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "prompts.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    context   TEXT    NOT NULL DEFAULT '[]',
    agent     TEXT    NOT NULL DEFAULT '',
    model     TEXT    NOT NULL DEFAULT '',
    prompt    TEXT    NOT NULL DEFAULT '',
    response  TEXT    NOT NULL DEFAULT '',
    log       TEXT    NOT NULL DEFAULT '{}',
    status    TEXT    NOT NULL DEFAULT 'pending',
    score     REAL    NOT NULL DEFAULT 0.0,
    elo       REAL    NOT NULL DEFAULT 1500.0
)
"""


class PromptDB:
    def __init__(self, db_path=DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(SCHEMA)
            conn.commit()

    def add(self, prompt, agent="", model="", context=None):
        """插入一条 pending 记录，返回 id。"""
        ctx = json.dumps(context or [])
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute(
                    "INSERT INTO prompts (context, agent, model, prompt, response, log, status) "
                    "VALUES (?, ?, ?, ?, '', '{}', 'pending')",
                    (ctx, agent, model, prompt),
                )
                conn.commit()
                return cur.lastrowid

    def done(self, pid, response, log=None):
        """标记为 done，写入 response 和 log。"""
        log_json = json.dumps(log or {})
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE prompts SET response=?, log=?, status='done' WHERE id=?",
                    (response, log_json, pid),
                )
                conn.commit()

    def failed(self, pid, error=""):
        """标记为 failed。"""
        log = json.dumps({"error": error})
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE prompts SET response=?, log=?, status='failed' WHERE id=?",
                    (error, log, pid),
                )
                conn.commit()

    def get(self, pid):
        """获取单条记录，返回 dict 或 None。"""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM prompts WHERE id=?", (pid,)).fetchone()
        return dict(row) if row else None

    def list_all(self):
        """返回所有记录。"""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM prompts ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def list_by_status(self, status):
        """按 status 过滤。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM prompts WHERE status=? ORDER BY id", (status,)
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, pid):
        with self._lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM prompts WHERE id=?", (pid,))
                conn.commit()

    def update_score(self, pid, score):
        """更新质量评分 0.0~1.0。"""
        with self._lock:
            with self._conn() as conn:
                conn.execute("UPDATE prompts SET score=? WHERE id=?", (score, pid))
                conn.commit()

    def update_elo(self, pid, elo):
        """更新 ELO 分数。"""
        with self._lock:
            with self._conn() as conn:
                conn.execute("UPDATE prompts SET elo=? WHERE id=?", (elo, pid))
                conn.commit()

    def elo_match(self, winner_id, loser_id, k=32):
        """ELO 对战：winner 击败 loser，更新双方分数。"""
        w = self.get(winner_id)
        l = self.get(loser_id)
        if not w or not l:
            return
        ra, rb = w["elo"], l["elo"]
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 - ea
        new_ra = ra + k * (1 - ea)
        new_rb = rb + k * (0 - eb)
        self.update_elo(winner_id, round(new_ra, 2))
        self.update_elo(loser_id, round(new_rb, 2))
        return {"winner_elo": round(new_ra, 2), "loser_elo": round(new_rb, 2)}


def parse_log(log_str):
    """解析 log JSON 字符串为 dict。"""
    if isinstance(log_str, dict):
        return log_str
    try:
        return json.loads(log_str)
    except (json.JSONDecodeError, TypeError):
        return {}


if __name__ == "__main__":
    db = PromptDB()
    print(f"数据库: {DB_PATH}")

    # 清理测试数据
    with db._conn() as conn:
        conn.execute("DELETE FROM prompts")
        conn.commit()

    # 插入几条测试记录
    pid1 = db.add("1+1=?", agent="Null", model="mimo-v2.5-free")
    db.done(pid1, "2", {"input_tokens": 10, "output_tokens": 5, "elapsed_ms": 1200})
    print(f"  插入 #{pid1} 并标记 done")

    pid2 = db.add("背一首古诗", agent="Null", model="mimo-v2.5-free")
    db.done(pid2, "床前明月光...", {"input_tokens": 12, "output_tokens": 20, "elapsed_ms": 3000})
    print(f"  插入 #{pid2} 并标记 done")

    pid3 = db.add("总结上面两首诗", agent="Null", model="mimo-v2.5-free", context=[pid1, pid2])
    print(f"  插入 #{pid3}（引用 #{pid1}, #{pid2}），status=pending")

    pid4 = db.add("这个 prompt 会失败", agent="Null", model="mimo-v2.5-free")
    db.failed(pid4, "timeout")
    print(f"  插入 #{pid4} 并标记 failed")

    # 查询
    print("\n所有记录:")
    for r in db.list_all():
        log = parse_log(r["log"])
        print(f"  #{r['id']} | {r['status']:<8} | ctx={r['context'][:30]} | {r['prompt'][:30]} | log={log}")

    print(f"\npending: {len(db.list_by_status('pending'))}")
    print(f"done:    {len(db.list_by_status('done'))}")
    print(f"failed:  {len(db.list_by_status('failed'))}")

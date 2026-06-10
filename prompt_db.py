"""
Prompt 执行数据库。

表结构 prompts:
  id        INTEGER PK  自增
  context   TEXT         输入（文本 / JSON 数组 / 混合）
  agent     TEXT         agent 名称
  model     TEXT         模型名
  response  TEXT         运行结果
  log       TEXT         JSON {input_tokens, output_tokens, elapsed_ms, timestamp}
  status    TEXT         pending / done / failed
  score     REAL         质量评分 0.0~1.0
  elo       REAL         ELO 分数（初始 1500）
"""

import json
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).parent / "prompts.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    context   TEXT    NOT NULL DEFAULT '',
    agent     TEXT    NOT NULL DEFAULT '',
    model     TEXT    NOT NULL DEFAULT '',
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
            # 迁移：确保 score 和 elo 列存在
            for col, typ, default in [
                ("score", "REAL", "0.0"),
                ("elo", "REAL", "1500.0"),
            ]:
                try:
                    conn.execute(f"SELECT {col} FROM prompts LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute(
                        f"ALTER TABLE prompts ADD COLUMN {col} {typ} NOT NULL DEFAULT {default}"
                    )
            # 迁移：如果 prompt 列存在，迁移到 context 并删除
            try:
                conn.execute("SELECT prompt FROM prompts LIMIT 1")
                # prompt 列存在，将 prompt 内容复制到 context（如果 context 为空）
                conn.execute(
                    "UPDATE prompts SET context = prompt WHERE context = '[]' OR context = ''"
                )
                conn.execute("ALTER TABLE prompts DROP COLUMN prompt")
            except sqlite3.OperationalError:
                pass  # prompt 列不存在，无需迁移
            conn.commit()

    def add(self, context, agent="", model=""):
        """
        插入一条 pending 记录，返回 id。

        context 可以是:
          - str: 直接作为 prompt 文本
          - list: JSON 数组，包含 id 引用或混合文本
        """
        if isinstance(context, list):
            ctx = json.dumps(context, ensure_ascii=False)
        else:
            ctx = str(context)
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute(
                    "INSERT INTO prompts (context, agent, model, response, log, status) "
                    "VALUES (?, ?, ?, '', '{}', 'pending')",
                    (ctx, agent, model),
                )
                conn.commit()
                return cur.lastrowid

    def done(self, pid, response, log=None):
        """标记为 done，写入 response 和 log。"""
        log_json = json.dumps(log or {}, ensure_ascii=False)
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE prompts SET response=?, log=?, status='done' WHERE id=?",
                    (response, log_json, pid),
                )
                conn.commit()

    def failed(self, pid, error=""):
        """标记为 failed。"""
        log = json.dumps({"error": error}, ensure_ascii=False)
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

    def update_agent(self, pid, agent):
        """更新 agent。"""
        with self._lock:
            with self._conn() as conn:
                conn.execute("UPDATE prompts SET agent=? WHERE id=?", (agent, pid))
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




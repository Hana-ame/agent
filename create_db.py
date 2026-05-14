"""
数据库初始化脚本

独立于 FastAPI 运行，可直接执行初始化数据库。
用法：
    python create_db.py              # 创建/重置 state.db
    python create_db.py --seed        # 创建并写入默认节点
    python create_db.py --path /tmp/test.db  # 指定路径
"""
import os
import sys
import sqlite3

# ── 配置 ───────────────────────────────────────────
DB_PATH = "state.db"
SEED_DEFAULTS = False
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--seed":
        SEED_DEFAULTS = True
    elif arg == "--path" and i + 2 < len(sys.argv):
        DB_PATH = sys.argv[i + 2]


def get_conn():
    return sqlite3.connect(DB_PATH, timeout=10)


def init_db():
    """创建所有表结构"""
    conn = get_conn()
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")

        # —— 节点表 ——
        conn.execute("""CREATE TABLE IF NOT EXISTS nodes (
            id          TEXT PRIMARY KEY,
            model       TEXT NOT NULL DEFAULT '',
            prompt      TEXT NOT NULL DEFAULT '',
            input_tags  TEXT NOT NULL DEFAULT '',
            output_tags TEXT NOT NULL DEFAULT '',
            active      INTEGER NOT NULL DEFAULT 1,
            context_mode INTEGER NOT NULL DEFAULT 0,
            interval    INTEGER NOT NULL DEFAULT 0
        )""")

        # —— 消息/KV 表 ——
        conn.execute("""CREATE TABLE IF NOT EXISTS kv (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tag         TEXT NOT NULL,
            value       TEXT NOT NULL DEFAULT '',
            parent_id   INTEGER,
            trace_id    TEXT,
            processed   INTEGER NOT NULL DEFAULT 0,
            next_hop    TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        # —— 追踪节点记录 ——
        conn.execute("""CREATE TABLE IF NOT EXISTS trace_node (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id    TEXT NOT NULL,
            node_id     TEXT NOT NULL,
            status      TEXT NOT NULL,
            input_tags  TEXT,
            output_tags TEXT,
            elapsed     REAL,
            total_tokens INTEGER DEFAULT 0,
            timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        # —— 执行日志 ——
        conn.execute("""CREATE TABLE IF NOT EXISTS node_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id     TEXT NOT NULL,
            model       TEXT,
            status      TEXT NOT NULL,
            input_tags  TEXT,
            output_tags TEXT,
            elapsed     REAL,
            input_tokens  INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            total_tokens  INTEGER DEFAULT 0,
            cost        REAL DEFAULT 0,
            timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        # —— 游标表（旧系统遗留）——
        conn.execute("""CREATE TABLE IF NOT EXISTS node_cursor (
            node_id     TEXT NOT NULL,
            tag         TEXT NOT NULL,
            last_kv_id  INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (node_id, tag)
        )""")

        # —— Trace 评价表 ——
        conn.execute("""CREATE TABLE IF NOT EXISTS trace_eval (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id    TEXT NOT NULL,
            node_id     TEXT NOT NULL,
            score       REAL NOT NULL DEFAULT 0,
            comment     TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        # —— Prompt 表 ——
        conn.execute("""CREATE TABLE IF NOT EXISTS prompt (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            prev_id     INTEGER,
            tag         TEXT NOT NULL DEFAULT '',
            prompt      TEXT NOT NULL DEFAULT '',
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        conn.commit()
        print(f"[create_db] 数据库已初始化: {os.path.abspath(DB_PATH)}")
    finally:
        conn.close()


def seed_defaults():
    """写入默认节点"""
    conn = get_conn()
    try:
        existing = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        if existing > 0:
            print("[create_db] nodes 表已有数据，跳过默认节点")
            return

        defaults = [
            ("analyzer", "opencode/deepseek-v4-flash-free",
             "分析以下用户输入的核心要点和需求：\n\n{user_input}\n\n请用中文回复。",
             "user_input", "analysis", 1),
            ("responder", "opencode/minimax-m2.5-free",
             "基于以下分析结果，生成一段友好、完整的回复：\n\n{analysis}",
             "analysis", "reply", 1),
        ]
        for n in defaults:
            conn.execute(
                "INSERT INTO nodes (id, model, prompt, input_tags, output_tags, active) "
                "VALUES (?, ?, ?, ?, ?, ?)", n
            )
        conn.commit()
        print(f"[create_db] 已写入 {len(defaults)} 个默认节点")
    finally:
        conn.close()


# ── 主入口 ─────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    if SEED_DEFAULTS:
        seed_defaults()
    print("[create_db] 完成")

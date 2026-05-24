"""
数据库初始化脚本

独立于 FastAPI 运行。
用法：
    python create_db.py              # 初始化 state.db
    python create_db.py --seed        # 初始化并写入默认节点（7 个免费模型）
    python create_db.py --path /tmp/test.db  # 指定路径
"""
import os
import sys
import sqlite3
import json

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
    """创建当前系统的表结构"""
    conn = get_conn()
    try:
        conn.execute("PRAGMA journal_mode=WAL")

        conn.execute("""CREATE TABLE IF NOT EXISTS prompt (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            prev_id     INTEGER,
            tag         TEXT NOT NULL DEFAULT '',
            prompt      TEXT NOT NULL DEFAULT '',
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            processed   INTEGER NOT NULL DEFAULT 0
        )""")

        conn.execute("""CREATE TABLE IF NOT EXISTS node (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            accept_tags TEXT NOT NULL DEFAULT '',
            output_tag  TEXT NOT NULL DEFAULT '',
            model       TEXT NOT NULL DEFAULT '',
            prompt      TEXT NOT NULL DEFAULT '',
            interval    INTEGER NOT NULL DEFAULT 5,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        conn.execute("""CREATE TABLE IF NOT EXISTS node_exec (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            input_ids       TEXT NOT NULL DEFAULT '',
            output_id       INTEGER,
            node_name       TEXT NOT NULL,
            model           TEXT NOT NULL DEFAULT '',
            status          TEXT NOT NULL DEFAULT 'success',
            error           TEXT NOT NULL DEFAULT '',
            elapsed         REAL NOT NULL DEFAULT 0,
            input_tokens    INTEGER NOT NULL DEFAULT 0,
            output_tokens   INTEGER NOT NULL DEFAULT 0,
            total_tokens    INTEGER NOT NULL DEFAULT 0,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        conn.execute("""CREATE TABLE IF NOT EXISTS pipeline (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            definition  TEXT NOT NULL DEFAULT '{}',
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        conn.execute("""CREATE TABLE IF NOT EXISTS pipeline_exec (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_id   INTEGER NOT NULL,
            input_data    TEXT NOT NULL DEFAULT '{}',
            output_data   TEXT NOT NULL DEFAULT '{}',
            status        TEXT NOT NULL DEFAULT 'running',
            error         TEXT NOT NULL DEFAULT '',
            steps_log     TEXT NOT NULL DEFAULT '[]',
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pipeline_id) REFERENCES pipeline(id)
        )""")

        # 兼容旧表添加列
        for col, tbl in [("model", "node"), ("processed", "prompt")]:
            try:
                conn.execute(
                    f"ALTER TABLE {tbl} ADD COLUMN {col} "
                    f"{'INTEGER NOT NULL DEFAULT 0' if col == 'processed' else 'TEXT NOT NULL DEFAULT \"\"' }"
                )
            except Exception:
                pass

        conn.commit()
        print(f"[create_db] 数据库已初始化: {os.path.abspath(DB_PATH)}")
    finally:
        conn.close()


def seed_defaults():
    """写入默认节点（7 个免费模型）和示例流水线"""
    conn = get_conn()
    try:
        existing = conn.execute("SELECT COUNT(*) FROM node").fetchone()[0]
        if existing == 0:
            defaults = [
                ("analyzer",   "user_input",   "analysis",       "opencode/deepseek-v4-flash-free",
                 "分析以下用户输入的核心要点和需求：\n\n{user_input}\n\n请用中文回复。", 5),
                ("responder",  "analysis",     "reply",          "opencode/minimax-m2.5-free",
                 "基于以下分析结果，生成一段友好、完整的回复：\n\n{analysis}", 5),
                ("translator", "text,lang",    "translated",     "opencode/qwen3.6-plus-free",
                 "翻译。目标语言：{lang}\n\n原文：\n{text}", 10),
                ("summarizer", "long_text",    "summary",        "google/gemma-4-31b-it",
                 "用简洁的语言总结以下内容，提取关键信息：\n\n{long_text}", 5),
                ("coder",      "coding_task",  "code",           "opencode/big-pickle",
                 "你是编程助手。请完成以下任务，输出代码：\n\n{coding_task}", 10),
                ("creative",   "creative_prompt", "creative_output", "siliconflow-cn/Qwen/Qwen3-8B",
                 "你是一位创意作家。请根据以下提示进行创作：\n\n{creative_prompt}", 10),
                ("general",    "general_input","general_output", "opencode/nemotron-3-super-free",
                 "{general_input}", 5),
            ]
            for name, accept, out, model, prompt, interval in defaults:
                conn.execute(
                    "INSERT INTO node (name, accept_tags, output_tag, model, prompt, interval) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (name, accept, out, model, prompt, interval),
                )
            print(f"[create_db] 已写入 {len(defaults)} 个默认节点")
        else:
            print("[create_db] node 表已有数据，跳过默认节点")

        # 示例流水线
        existing_pl = conn.execute("SELECT COUNT(*) FROM pipeline").fetchone()[0]
        if existing_pl == 0:
            sample_pipelines = [
                (
                    "分析-响应链",
                    json.dumps({
                        "entry": "analyzer",
                        "steps": [
                            {
                                "id": "analyzer",
                                "name": "分析器",
                                "model": "opencode/deepseek-v4-flash-free",
                                "prompt": "分析以下用户输入的核心要点和需求：\n\n{user_input}\n\n请用中文回复。",
                                "output_key": "analysis",
                                "next": "responder"
                            },
                            {
                                "id": "responder",
                                "name": "响应器",
                                "model": "opencode/minimax-m2.5-free",
                                "prompt": "基于以下分析结果，生成一段友好、完整的回复：\n\n{analysis}",
                                "output_key": "response",
                                "next": None
                            }
                        ]
                    }, ensure_ascii=False),
                ),
                (
                    "翻译-审查链",
                    json.dumps({
                        "entry": "translator",
                        "steps": [
                            {
                                "id": "translator",
                                "name": "翻译器",
                                "model": "opencode/qwen3.6-plus-free",
                                "prompt": "将以下文本翻译成{target_lang}：\n\n{source_text}",
                                "output_key": "translated",
                                "next": "reviewer"
                            },
                            {
                                "id": "reviewer",
                                "name": "审查员",
                                "model": "google/gemma-4-31b-it",
                                "prompt": "审查以下翻译是否准确自然，如有问题请修正：\n\n{translated}",
                                "output_key": "final_output",
                                "next": None
                            }
                        ]
                    }, ensure_ascii=False),
                ),
            ]
            for name, definition in sample_pipelines:
                conn.execute(
                    "INSERT INTO pipeline (name, definition) VALUES (?, ?)",
                    (name, definition),
                )
            print(f"[create_db] 已写入 {len(sample_pipelines)} 个示例流水线")
        else:
            print("[create_db] pipeline 表已有数据，跳过示例流水线")

        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    if SEED_DEFAULTS:
        seed_defaults()
    print("[create_db] 完成")

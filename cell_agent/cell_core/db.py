import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "cell.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS organelles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    capability TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT 'local',
    api_base TEXT DEFAULT '',
    cost_per_token REAL DEFAULT 0.0,
    quality_score REAL DEFAULT 0.5,
    speed_score REAL DEFAULT 0.5,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mrna (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organelle_id INTEGER NOT NULL,
    template TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    quality_score REAL DEFAULT 0.5,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organelle_id) REFERENCES organelles(id)
);

CREATE TABLE IF NOT EXISTS dna (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    steps_json TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dna_id INTEGER,
    input_json TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    quality_score REAL,
    total_cost REAL DEFAULT 0.0,
    total_time_ms INTEGER DEFAULT 0,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dna_id) REFERENCES dna(id)
);

CREATE TABLE IF NOT EXISTS step_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    step_index INTEGER NOT NULL,
    organelle_id INTEGER,
    input_json TEXT,
    output_json TEXT,
    protein_used TEXT,
    cost REAL DEFAULT 0.0,
    time_ms INTEGER DEFAULT 0,
    quality_score REAL,
    status TEXT DEFAULT 'pending',
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(id),
    FOREIGN KEY (organelle_id) REFERENCES organelles(id)
);

CREATE TABLE IF NOT EXISTS evolution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organelle_id INTEGER,
    mrna_id INTEGER,
    change_type TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    reason TEXT,
    impact_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organelle_id) REFERENCES organelles(id),
    FOREIGN KEY (mrna_id) REFERENCES mrna(id)
);

CREATE TABLE IF NOT EXISTS task_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    action TEXT NOT NULL DEFAULT 'process',
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);
"""


def get_connection(db_path=None):
    path = db_path or str(DB_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path=None):
    conn = get_connection(db_path)
    for stmt in SCHEMA_SQL.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    conn.close()


def seed_default_data(db_path=None):
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM organelles")
    if cur.fetchone()[0] == 0:
        organelles = [
            ("coder", "code_generation", "local", "", 0.0),
            ("translator", "translation", "local", "", 0.0),
            ("summarizer", "summarization", "local", "", 0.0),
            ("reviewer", "code_review", "local", "", 0.0),
            ("optimizer", "optimization", "local", "", 0.0),
        ]
        cur.executemany(
            "INSERT INTO organelles (name, capability, model, api_base, cost_per_token) VALUES (?, ?, ?, ?, ?)",
            organelles,
        )

    cur.execute("SELECT COUNT(*) FROM mrna")
    if cur.fetchone()[0] == 0:
        templates = [
            (1, "You are a code generator. Generate {language} code for: {task}. Output only the code block."),
            (2, "Translate the following {source_lang} text to {target_lang}. Text: {text}. Output only the translation."),
            (3, "Summarize the following text concisely: {text}. Output a bullet-point summary."),
            (4, "Review this {language} code for bugs, style issues, and improvements:\n```\n{code}\n```\nOutput a structured review."),
            (5, "Optimize the following {language} code for performance and readability:\n```\n{code}\n```\nOutput the optimized code."),
        ]
        cur.executemany(
            "INSERT INTO mrna (organelle_id, template) VALUES (?, ?)",
            templates,
        )

    cur.execute("SELECT COUNT(*) FROM dna")
    if cur.fetchone()[0] == 0:
        pipelines = [
            (
                "code_review_pipeline",
                "Generate code, then review it",
                json.dumps([
                    {"organelle": "coder", "step_name": "generate"},
                    {"organelle": "reviewer", "step_name": "review"},
                ]),
            ),
            (
                "translate_summarize",
                "Translate text then summarize",
                json.dumps([
                    {"organelle": "translator", "step_name": "translate"},
                    {"organelle": "summarizer", "step_name": "summarize"},
                ]),
            ),
            (
                "code_optimize_pipeline",
                "Generate, review, then optimize code",
                json.dumps([
                    {"organelle": "coder", "step_name": "generate"},
                    {"organelle": "reviewer", "step_name": "review"},
                    {"organelle": "optimizer", "step_name": "optimize"},
                ]),
            ),
        ]
        cur.executemany(
            "INSERT INTO dna (name, description, steps_json) VALUES (?, ?, ?)",
            pipelines,
        )

    conn.commit()
    conn.close()

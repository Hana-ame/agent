import sqlite3
import time
import threading
from typing import List, Optional

DB_PATH = "messagebox.db"

class MessageBoxStorage:
    def __init__(self):
        self._write_lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_id ON messages(id)')

    def _get_conn(self):
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _generate_id(self) -> int:
        base_ts = int(time.time() * 1000)
        msg_id = base_ts
        while True:
            with self._get_conn() as conn:
                cur = conn.execute("SELECT 1 FROM messages WHERE id = ?", (msg_id,))
                if cur.fetchone() is None:
                    return msg_id
                msg_id += 1

    def create(self, content: str) -> dict:
        msg_id = self._generate_id()
        timestamp = int(time.time())
        with self._write_lock:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO messages (id, content, timestamp) VALUES (?, ?, ?)",
                    (msg_id, content, timestamp)
                )
                conn.commit()
        return {"id": msg_id, "content": content, "timestamp": timestamp}

    def poll_latest(self) -> Optional[dict]:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT id, content, timestamp FROM messages ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def poll_next(self, after_id: int) -> Optional[dict]:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT id, content, timestamp FROM messages WHERE id > ? ORDER BY id ASC LIMIT 1",
                (after_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def list_all(self) -> List[dict]:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT id, content, timestamp FROM messages ORDER BY id ASC"
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]

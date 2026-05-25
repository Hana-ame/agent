from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sqlite3
import time
import threading
from typing import List, Optional
from pathlib import Path

DB_PATH = "messagebox_channels.db"

class MessageBoxStorage:
    def __init__(self):
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_channel_id ON messages(channel, id)')

    def _get_conn(self):
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, channel: str, content: str) -> dict:
        timestamp = int(time.time() * 1000)
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "INSERT INTO messages (channel, content, timestamp) VALUES (?, ?, ?)",
                    (channel, content, timestamp)
                )
                msg_id = cur.lastrowid
                conn.commit()
        return {"id": msg_id, "channel": channel, "content": content, "timestamp": timestamp}

    def poll_latest(self, channel: str) -> Optional[dict]:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT id, content, timestamp FROM messages WHERE channel = ? ORDER BY id DESC LIMIT 1",
                (channel,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def poll_next(self, channel: str, after_id: int) -> Optional[dict]:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT id, content, timestamp FROM messages WHERE channel = ? AND id > ? ORDER BY id ASC LIMIT 1",
                (channel, after_id)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def list_all(self, channel: str) -> List[dict]:
        with self._get_conn() as conn:
            cur = conn.execute(
                "SELECT id, content, timestamp FROM messages WHERE channel = ? ORDER BY id ASC",
                (channel,)
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]

storage = MessageBoxStorage()
app = FastAPI()

class MessageRequest(BaseModel):
    content: str

@app.post("/api/{channel}/message")
async def create_msg(channel: str, req: MessageRequest):
    return storage.create(channel, req.content)

@app.get("/api/{channel}/latest")
async def get_latest(channel: str):
    msg = storage.poll_latest(channel)
    if not msg:
        raise HTTPException(status_code=404, detail="No messages in channel")
    return msg

@app.get("/api/{channel}/next")
async def get_next(channel: str, after_id: int):
    msg = storage.poll_next(channel, after_id)
    if not msg:
        raise HTTPException(status_code=404, detail="No next message in channel")
    return msg

@app.get("/api/{channel}/list")
async def get_list(channel: str):
    return storage.list_all(channel)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

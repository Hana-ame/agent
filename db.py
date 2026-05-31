import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "simpleai.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

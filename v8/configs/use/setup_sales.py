#!/usr/bin/env python3
"""建一个演示用 SQLite：v8/data/use/sales.sqlite3。"""
import sqlite3
from pathlib import Path

db = Path(__file__).resolve().parents[2] / "data" / "use" / "sales.sqlite3"
db.parent.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(db)
con.executescript("""
DROP TABLE IF EXISTS sales;
CREATE TABLE sales (region TEXT, amount INTEGER);
INSERT INTO sales VALUES ('华东', 128), ('华南', 96), ('华北', 71), ('西南', 43);
""")
con.commit(); con.close()
print(db)

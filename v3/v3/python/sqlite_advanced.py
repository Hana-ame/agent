import sqlite3
import tempfile
import os

with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
    db_path = tmp.name
    # 启用 Row 工厂
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)')
    
    # 事务插入
    try:
        with conn:
            cur.execute('INSERT INTO users (name) VALUES (?)', ('Alice',))
            cur.execute('INSERT INTO users (name) VALUES (?)', ('Bob',))
            # 故意触发错误回滚
            raise ValueError("Simulated error")
    except ValueError:
        pass  # 事务已回滚
    
    # 插入成功的事务
    with conn:
        cur.execute('INSERT INTO users (name) VALUES (?)', ('Charlie',))
    
    cur.execute('SELECT * FROM users')
    rows = cur.fetchall()
    for row in rows:
        print(f"User: {row['name']} (ID: {row['id']})")
    
    conn.close()


# 测试并修改，until通过测试。

import sqlite3


class DataBase:
    def __init__(self, db_path="simpleai.db"):
        # 在这里初始化一个sqlite3 数据库，开启wal。
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.cursor = self.conn.cursor()

        self.prompts = Table(self, "Prompts", prompts_schema, indexes=prompts_indexes)
        self.requests = Table(self, "Requests", requests_schema, indexes=requests_indexes)

    def close(self):
        """Close the database connection."""
        self.conn.close()


class Table():
    # 传入 DataBase 实例，不使用继承
    def __init__(self, database, table_name, columns, indexes=None):
        """
        Initialize a generic table.

        Args:
            database: A DataBase instance.
            table_name: Name of the table.
            columns: Dict mapping column names to their SQL type strings,
                     e.g. {"name": "TEXT NOT NULL", "age": "INTEGER"}.
            indexes: Optional list of column names to create indexes on.
        """
        self.db = database
        self.table_name = table_name
        self.columns = columns

        col_defs = ", ".join(f"{name} {dtype}" for name, dtype in columns.items())
        self.db.cursor.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})"
        )
        self.db.conn.commit()

        # 在 schema 中指定的列上自动创建索引
        if indexes:
            for col in indexes:
                index_name = f"idx_{table_name}_{col}"
                self.db.cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({col})"
                )
            self.db.conn.commit()

    def Insert(self, values_dict):
        """
        Insert a row into the table.

        Args:
            values_dict: Dict mapping column names to values,
                          e.g. {"name": "Alice", "age": 30}.
        """
        columns = ", ".join(values_dict.keys())
        placeholders = ", ".join(["?" for _ in values_dict])
        values = tuple(values_dict.values())
        self.db.cursor.execute(
            f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})",
            values
        )
        self.db.conn.commit()
        return self.db.cursor.lastrowid

    def Read(self, columns="*", condition=None, order_by=None):
        """
        Read rows from the table.

        Args:
            columns: Columns to select (default: "*").
            condition: Optional WHERE clause (e.g. "age > 25").
            order_by: Optional ORDER BY clause (e.g. "age DESC").

        Returns:
            List of rows (tuples).
        """
        query = f"SELECT {columns} FROM {self.table_name}"
        if condition:
            query += f" WHERE {condition}"
        if order_by:
            query += f" ORDER BY {order_by}"
        self.db.cursor.execute(query)
        return self.db.cursor.fetchall()

    def Update(self, values_dict, condition):
        """
        Update rows in the table.

        Args:
            values_dict: Dict mapping column names to new values.
            condition: WHERE clause (e.g. "id=1").

        Returns:
            Number of rows affected.
        """
        set_clause = ", ".join(f"{col}=?" for col in values_dict.keys())
        values = tuple(values_dict.values())
        self.db.cursor.execute(
            f"UPDATE {self.table_name} SET {set_clause} WHERE {condition}",
            values,
        )
        self.db.conn.commit()
        return self.db.cursor.rowcount

prompts_schema = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "previous_id": "INTEGER REFERENCES Prompts(id)",
    "prompt": "TEXT NOT NULL",
    "agent": "TEXT",
    "model": "TEXT",
    "response": "TEXT",
    "abstract": "TEXT",
    "should_end": "INTEGER DEFAULT 0",  # flag: 1=应结束对话, 0=继续对话
}

prompts_indexes = ["previous_id"]

requests_schema = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "prompt_id": "INTEGER REFERENCES Prompts(id)",
    "agent_name": "TEXT",
    "start_time": "TEXT",
    "end_time": "TEXT",
    "input_tokens": "INTEGER",
    "output_tokens": "INTEGER",
    "success": "INTEGER",
    "include_history": "INTEGER DEFAULT 0",  # flag: 1=包括之前的对话, 0=单独这段对话
}

requests_indexes = ["prompt_id"]

    

if __name__ == "__main__":
    db = DataBase("simpleai.db")
    print(f"Prompts table created. Columns: {list(db.prompts.columns.keys())}")
    print(f"Requests table created. Columns: {list(db.requests.columns.keys())}")

    # 插入测试数据 - Prompts (should_end=0: 继续对话)
    row_id = db.prompts.Insert({
        "prompt": "Hello, who are you?",
        "agent": "assistant",
        "model": "gpt-4",
        "response": "I am an AI assistant.",
        "should_end": 0,
    })
    print(f"Inserted prompt row id: {row_id}")

    # 读取测试数据
    rows = db.prompts.Read()
    print(f"All prompts: {rows}")

    # 插入测试数据 - Prompts (should_end=1: 应结束对话)
    row_id2 = db.prompts.Insert({
        "prompt": "Goodbye!",
        "agent": "assistant",
        "model": "gpt-4",
        "response": "Goodbye, have a nice day!",
        "should_end": 1,
    })
    print(f"Inserted prompt (should_end=1) row id: {row_id2}")

    # 按条件读取
    rows2 = db.prompts.Read(condition="agent='assistant'")
    print(f"Prompts where agent='assistant': {rows2}")

    # 按 should_end 筛选
    rows_should_end = db.prompts.Read(condition="should_end=1")
    print(f"Prompts with should_end=1: {rows_should_end}")
    rows_continue = db.prompts.Read(condition="should_end=0")
    print(f"Prompts with should_end=0: {rows_continue}")

    # 插入测试数据 - Requests (include_history=0: 单独这段对话)
    req_id = db.requests.Insert({
        "prompt_id": row_id,
        "agent_name": "assistant",
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-01-01T00:00:01",
        "input_tokens": 50,
        "output_tokens": 150,
        "success": 1,
        "include_history": 0,
    })
    print(f"Inserted request (include_history=0) row id: {req_id}")

    # 插入测试数据 - Requests (include_history=1: 包括之前的对话)
    req_id2 = db.requests.Insert({
        "prompt_id": row_id,
        "agent_name": "assistant",
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-01-01T00:00:02",
        "input_tokens": 80,
        "output_tokens": 200,
        "success": 1,
        "include_history": 1,
    })
    print(f"Inserted request (include_history=1) row id: {req_id2}")

    # 读取请求数据
    req_rows = db.requests.Read()
    print(f"All requests: {req_rows}")

    # 按条件读取请求
    req_rows2 = db.requests.Read(condition="agent_name='assistant'")
    print(f"Requests where agent_name='assistant': {req_rows2}")

    # 按 include_history 筛选
    req_history = db.requests.Read(condition="include_history=1")
    print(f"Requests with include_history=1: {req_history}")
    req_no_history = db.requests.Read(condition="include_history=0")
    print(f"Requests with include_history=0: {req_no_history}")

    db.close()
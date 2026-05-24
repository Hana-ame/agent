"""
Tests for database.py — covers DataBase init, Table CRUD, and edge cases.
"""
import os
import sqlite3
import pytest

from database import DataBase, Table

TEST_DB = "test_simpleai.db"


@pytest.fixture(autouse=True)
def cleanup():
    """Remove the test database before and after each test."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


# ───────────────────────── DataBase tests ─────────────────────────


class TestDataBaseInit:
    def test_creates_db_file(self):
        db = DataBase(TEST_DB)
        assert os.path.exists(TEST_DB)
        db.close()

    def test_wal_mode_enabled(self):
        db = DataBase(TEST_DB)
        cursor = db.conn.execute("PRAGMA journal_mode;")
        row = cursor.fetchone()
        assert row[0].lower() == "wal", f"Expected WAL, got {row[0]}"
        db.close()

    def test_foreign_keys_enabled(self):
        db = DataBase(TEST_DB)
        cursor = db.conn.execute("PRAGMA foreign_keys;")
        row = cursor.fetchone()
        assert row[0] == 1, f"Expected 1, got {row[0]}"
        db.close()

    def test_prompts_table_exists(self):
        db = DataBase(TEST_DB)
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='Prompts'"
        ).fetchall()
        assert len(tables) == 1
        assert tables[0][0] == "Prompts"
        db.close()

    def test_prompts_table_schema(self):
        db = DataBase(TEST_DB)
        col_info = db.conn.execute("PRAGMA table_info(Prompts)").fetchall()
        col_names = {row[1] for row in col_info}
        expected = {"id", "previous_id", "prompt", "agent", "model", "response", "abstract", "should_end"}
        assert col_names == expected, f"Expected {expected}, got {col_names}"

        # id is primary key and autoincrement
        id_col = next(row for row in col_info if row[1] == "id")
        assert id_col[5] == 1  # pk flag
        db.close()

    def test_prompts_table_accessible_via_attribute(self):
        db = DataBase(TEST_DB)
        assert hasattr(db, "prompts")
        assert isinstance(db.prompts, Table)
        assert db.prompts.table_name == "Prompts"
        db.close()

    def test_close(self):
        db = DataBase(TEST_DB)
        db.close()
        # After close, any operation should raise sqlite3.ProgrammingError
        with pytest.raises(sqlite3.ProgrammingError):
            db.conn.execute("SELECT 1")


# ───────────────────────── Table CRUD tests ─────────────────────────


class TestTableCRUD:
    def setup_db(self):
        db = DataBase(TEST_DB)
        # Create a simple table for testing
        schema = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "name": "TEXT NOT NULL", "age": "INTEGER"}
        t = Table(db, "people", schema)
        return db, t

    def test_create_table(self):
        db, t = self.setup_db()
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='people'"
        ).fetchall()
        assert len(tables) == 1
        db.close()

    def test_insert(self):
        db, t = self.setup_db()
        row_id = t.Insert({"name": "Alice", "age": 30})
        assert row_id == 1, f"Expected 1, got {row_id}"
        db.close()

    def test_insert_multiple(self):
        db, t = self.setup_db()
        id1 = t.Insert({"name": "Alice", "age": 30})
        id2 = t.Insert({"name": "Bob", "age": 25})
        assert id1 == 1
        assert id2 == 2
        db.close()

    def test_insert_without_optional_column(self):
        db, t = self.setup_db()
        row_id = t.Insert({"name": "Charlie"})  # age is optional (nullable)
        assert row_id == 1
        db.close()

    def test_read_all(self):
        db, t = self.setup_db()
        t.Insert({"name": "Alice", "age": 30})
        t.Insert({"name": "Bob", "age": 25})
        rows = t.Read()
        assert len(rows) == 2
        assert rows[0][1] == "Alice"
        assert rows[1][1] == "Bob"
        db.close()

    def test_read_with_condition(self):
        db, t = self.setup_db()
        t.Insert({"name": "Alice", "age": 30})
        t.Insert({"name": "Bob", "age": 25})
        t.Insert({"name": "Charlie", "age": 30})
        rows = t.Read(condition="age = 30")
        assert len(rows) == 2
        for row in rows:
            assert row[2] == 30  # age column
        db.close()

    def test_read_with_order_by(self):
        db, t = self.setup_db()
        t.Insert({"name": "Charlie", "age": 35})
        t.Insert({"name": "Alice", "age": 30})
        t.Insert({"name": "Bob", "age": 25})
        rows = t.Read(order_by="age ASC")
        ages = [row[2] for row in rows]
        assert ages == [25, 30, 35]
        db.close()

    def test_read_specific_columns(self):
        db, t = self.setup_db()
        t.Insert({"name": "Alice", "age": 30})
        rows = t.Read(columns="name")
        assert len(rows) == 1
        assert rows[0] == ("Alice",)
        db.close()

    def test_read_empty_table(self):
        db, t = self.setup_db()
        rows = t.Read()
        assert rows == []
        db.close()


# ───────────────────────── Table indexes tests ─────────────────────────


class TestTableIndexes:
    def test_index_creation(self):
        """Table with indexes parameter creates SQL indexes."""
        db = DataBase(TEST_DB)
        schema = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "email": "TEXT", "name": "TEXT"}
        t = Table(db, "users", schema, indexes=["email"])

        indexes = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='users'"
        ).fetchall()
        index_names = [row[0] for row in indexes]
        assert "idx_users_email" in index_names
        db.close()

    def test_no_indexes_by_default(self):
        """Table without indexes parameter creates no extra indexes.

        Note: INTEGER PRIMARY KEY does NOT create an auto-index in SQLite
        because it's an alias for the rowid.
        """
        db = DataBase(TEST_DB)
        schema = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "val": "TEXT"}
        t = Table(db, "plain", schema)  # no indexes=

        indexes = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='plain'"
        ).fetchall()
        # No user-defined indexes should exist; only possible auto-index
        # from a non-INTEGER primary key, but INTEGER PRIMARY KEY is not indexed.
        user_indexes = [r[0] for r in indexes if not r[0].startswith("sqlite_autoindex")]
        assert len(user_indexes) == 0, f"Expected no user indexes, got {user_indexes}"
        db.close()

    def test_multiple_indexes(self):
        """Multiple columns can be indexed at once."""
        db = DataBase(TEST_DB)
        schema = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "a": "TEXT", "b": "TEXT", "c": "TEXT"}
        t = Table(db, "multi_idx", schema, indexes=["a", "c"])

        indexes = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='multi_idx'"
        ).fetchall()
        index_names = [row[0] for row in indexes]
        assert "idx_multi_idx_a" in index_names
        assert "idx_multi_idx_c" in index_names
        assert "idx_multi_idx_b" not in index_names
        db.close()


# ───────────────────────── Prompts table specific tests ─────────────────────────


class TestPromptsTable:
    def test_insert_prompt(self):
        db = DataBase(TEST_DB)
        row_id = db.prompts.Insert({
            "prompt": "Hello",
            "agent": "TestAgent",
            "model": "gpt-4",
            "response": "Hi there!",
            "abstract": "A greeting",
        })
        assert row_id == 1
        db.close()

    def test_insert_chain(self):
        db = DataBase(TEST_DB)
        id1 = db.prompts.Insert({"prompt": "First", "agent": "A"})
        id2 = db.prompts.Insert({"prompt": "Second", "agent": "A", "previous_id": id1})
        assert id2 == 2

        # Verify the chain
        rows = db.prompts.Read()
        assert len(rows) == 2
        # Column order: id(0), previous_id(1), prompt(2), agent(3), model(4), response(5), abstract(6)
        assert rows[0][0] == 1          # id
        assert rows[0][1] is None       # previous_id (not set)
        assert rows[0][2] == "First"    # prompt
        assert rows[1][0] == 2          # id
        assert rows[1][1] == 1          # previous_id points to first row
        assert rows[1][2] == "Second"   # prompt
        db.close()

    def test_foreign_key_enforced(self):
        db = DataBase(TEST_DB)
        # Inserting a prompt with a non-existent previous_id should fail
        with pytest.raises(sqlite3.IntegrityError):
            db.prompts.Insert({"prompt": "Orphan", "previous_id": 999})
        db.close()

    def test_previous_id_index_exists(self):
        """Verify that an index on previous_id is automatically created."""
        db = DataBase(TEST_DB)
        indexes = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='Prompts'"
        ).fetchall()
        index_names = [row[0] for row in indexes]
        assert "idx_Prompts_previous_id" in index_names, (
            f"Expected idx_Prompts_previous_id in {index_names}"
        )
        db.close()


# ───────────────────────── Requests table specific tests ─────────────────────────


class TestRequestsTable:
    def test_requests_table_exists(self):
        db = DataBase(TEST_DB)
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='Requests'"
        ).fetchall()
        assert len(tables) == 1
        assert tables[0][0] == "Requests"
        db.close()

    def test_requests_table_schema(self):
        db = DataBase(TEST_DB)
        col_info = db.conn.execute("PRAGMA table_info(Requests)").fetchall()
        col_names = {row[1] for row in col_info}
        expected = {"id", "prompt_id", "agent_name", "start_time", "end_time",
                    "input_tokens", "output_tokens", "success", "include_history"}
        assert col_names == expected, f"Expected {expected}, got {col_names}"

        # id is primary key and autoincrement
        id_col = next(row for row in col_info if row[1] == "id")
        assert id_col[5] == 1  # pk flag
        db.close()

    def test_requests_table_accessible_via_attribute(self):
        db = DataBase(TEST_DB)
        assert hasattr(db, "requests")
        assert isinstance(db.requests, Table)
        assert db.requests.table_name == "Requests"
        db.close()

    def test_insert_request(self):
        db = DataBase(TEST_DB)
        # First insert a prompt to satisfy foreign key
        prompt_id = db.prompts.Insert({
            "prompt": "Test prompt",
            "agent": "TestAgent",
            "model": "gpt-4",
        })
        req_id = db.requests.Insert({
            "prompt_id": prompt_id,
            "agent_name": "TestAgent",
            "start_time": "2024-06-01T10:00:00",
            "end_time": "2024-06-01T10:00:05",
            "input_tokens": 100,
            "output_tokens": 200,
            "success": 1,
        })
        assert req_id == 1
        db.close()

    def test_insert_multiple_requests(self):
        db = DataBase(TEST_DB)
        prompt_id = db.prompts.Insert({"prompt": "P1", "agent": "A"})
        id1 = db.requests.Insert({
            "prompt_id": prompt_id, "agent_name": "A",
            "start_time": "t1", "end_time": "t2",
            "input_tokens": 10, "output_tokens": 20, "success": 1,
        })
        id2 = db.requests.Insert({
            "prompt_id": prompt_id, "agent_name": "B",
            "start_time": "t3", "end_time": "t4",
            "input_tokens": 30, "output_tokens": 40, "success": 0,
        })
        assert id1 == 1
        assert id2 == 2
        rows = db.requests.Read()
        assert len(rows) == 2
        db.close()

    def test_request_foreign_key_enforced(self):
        db = DataBase(TEST_DB)
        # Inserting a request with a non-existent prompt_id should fail
        with pytest.raises(sqlite3.IntegrityError):
            db.requests.Insert({
                "prompt_id": 999, "agent_name": "Ghost",
                "start_time": "t1", "end_time": "t2",
                "input_tokens": 0, "output_tokens": 0, "success": 1,
            })
        db.close()

    def test_request_read_with_condition(self):
        db = DataBase(TEST_DB)
        prompt_id = db.prompts.Insert({"prompt": "P1", "agent": "A"})
        db.requests.Insert({
            "prompt_id": prompt_id, "agent_name": "AgentX",
            "start_time": "t1", "end_time": "t2",
            "input_tokens": 50, "output_tokens": 100, "success": 1,
        })
        db.requests.Insert({
            "prompt_id": prompt_id, "agent_name": "AgentY",
            "start_time": "t3", "end_time": "t4",
            "input_tokens": 60, "output_tokens": 120, "success": 0,
        })
        # Filter by success flag
        success_rows = db.requests.Read(condition="success = 1")
        assert len(success_rows) == 1
        assert success_rows[0][2] == "AgentX"  # agent_name column
        db.close()

    def test_request_prompt_id_index_exists(self):
        """Verify that an index on prompt_id is automatically created."""
        db = DataBase(TEST_DB)
        indexes = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='Requests'"
        ).fetchall()
        index_names = [row[0] for row in indexes]
        assert "idx_Requests_prompt_id" in index_names, (
            f"Expected idx_Requests_prompt_id in {index_names}"
        )
        db.close()


# ───────────────────────── Edge cases ─────────────────────────


class TestEdgeCases:
    def test_reinitialize_same_db(self):
        """Opening the same database again should re-use existing tables."""
        db1 = DataBase(TEST_DB)
        db1.prompts.Insert({"prompt": "Hello"})
        db1.close()

        db2 = DataBase(TEST_DB)
        rows = db2.prompts.Read()
        assert len(rows) == 1
        db2.close()

    def test_multiple_tables(self):
        db = DataBase(TEST_DB)
        schema1 = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "val": "TEXT"}
        schema2 = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "val": "TEXT"}
        t1 = Table(db, "table_a", schema1)
        t2 = Table(db, "table_b", schema2)

        t1.Insert({"val": "from_a"})
        t2.Insert({"val": "from_b"})

        assert len(t1.Read()) == 1
        assert len(t2.Read()) == 1
        db.close()

    def test_sql_injection_resistance(self):
        """Column names and condition strings could be dangerous — test basic safety."""
        db = DataBase(TEST_DB)
        schema = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", "data": "TEXT"}
        t = Table(db, "safe_test", schema)

        # Even with suspicious values, parameterized insertion is safe
        t.Insert({"data": "'; DROP TABLE safe_test; --"})
        rows = t.Read()
        assert len(rows) == 1
        assert rows[0][1] == "'; DROP TABLE safe_test; --"
        db.close()

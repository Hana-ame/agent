#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
memory - 长期记忆存储工具（SQLite版）

用于存储教训、经验、知识片段等，支持标签分类和检索。

用法：
    py utils.py memory add <内容> [--tags tag1,tag2,...]
        添加一条记忆，可选标签（逗号分隔）

    py utils.py memory list [--tag TAG] [--all]
        列出记忆。可用 --tag 按标签筛选，不加则列出所有。
        加上 --all 显示完整内容（默认只显示前50字符）

    py utils.py memory search <关键词>
        模糊搜索内容或标签中包含关键词的记忆

    py utils.py memory delete <id>
        删除指定 ID 的记忆

    py utils.py memory update <id> [--content 新内容] [--tags 新标签]
        更新指定 ID 的记忆内容和/或标签（至少提供一个选项）

输出格式：
    - 成功时：输出相关提示信息。
    - 失败时：输出统一格式错误块：
      === memory <子命令> ===
      错误：具体错误信息
      === end of memory <子命令> ===
"""

import sys
import os
import sqlite3
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils import core

DB_FILE = "memory.db"
DUMP_FILE = "memory_dump.txt"

def get_db_path(root_path: Path) -> Path:
    return root_path / DB_FILE

def get_dump_path(root_path: Path) -> Path:
    return root_path / DUMP_FILE

def init_db(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def get_connection(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = dict_factory
    return conn

def generate_id() -> str:
    return f"{int(time.time()*1000)}"

def _dump_memories(root_path: Path, memories: List[Dict]) -> str:
    """将记忆列表写入 dump 文件，并返回提示信息"""
    dump_path = get_dump_path(root_path)
    with open(dump_path, 'w', encoding='utf-8') as f:
        if not memories:
            f.write("暂无记忆。\n")
        else:
            for m in memories:
                f.write(f"ID: {m['id']}  [{m['created']}] 标签: {m['tags']}\n")
                f.write(f"  内容: {m['content']}\n\n")
    return f"详情已保存到 {DUMP_FILE}，可用 'py utils.py cat {DUMP_FILE}' 查看。"

def _handle_error(subcmd: str, msg: str) -> str:
    return f"=== memory {subcmd} ===\n错误：{msg}\n=== end of memory {subcmd} ==="

def run(ctx: core.Context, args: List[str]) -> str:
    if not args:
        return __doc__.strip()

    subcmd = args[0].lower()
    root = Path(ctx.root_path)
    output_lines = []

    try:
        if subcmd == "add":
            parser = argparse.ArgumentParser(prog="py utils.py memory add", add_help=False)
            parser.add_argument("content", nargs="?", help="记忆内容")
            parser.add_argument("--tags", "-t", help="逗号分隔的标签")
            try:
                parsed = parser.parse_args(args[1:])
            except SystemExit:
                return _handle_error(subcmd, "参数解析失败")
            if not parsed.content:
                return _handle_error(subcmd, "必须提供内容。")
            tags = parsed.tags.split(",") if parsed.tags else []

            db_path = get_db_path(root)
            init_db(db_path)
            conn = get_connection(db_path)
            c = conn.cursor()
            mem_id = generate_id()
            tags_str = ",".join(tags) if tags else ""
            c.execute(
                "INSERT INTO memories (id, content, tags, created, updated) VALUES (?, ?, ?, datetime('now'), datetime('now'))",
                (mem_id, content, tags_str)
            )
            conn.commit()
            c.execute("SELECT id, content, tags, created, updated FROM memories ORDER BY created DESC")
            memories = c.fetchall()
            conn.close()
            dump_msg = _dump_memories(root, memories)
            output_lines.append(f"记忆已添加，ID: {mem_id}")
            output_lines.append(dump_msg)

        elif subcmd == "list":
            parser = argparse.ArgumentParser(prog="py utils.py memory list", add_help=False)
            parser.add_argument("--tag", help="按标签筛选")
            parser.add_argument("--all", action="store_true", help="显示完整内容")
            parsed = parser.parse_args(args[1:])

            db_path = get_db_path(root)
            init_db(db_path)
            conn = get_connection(db_path)
            c = conn.cursor()
            if parsed.tag:
                c.execute(
                    "SELECT id, content, tags, created, updated FROM memories WHERE tags LIKE ? ORDER BY created DESC",
                    (f"%{parsed.tag}%",)
                )
            else:
                c.execute("SELECT id, content, tags, created, updated FROM memories ORDER BY created DESC")
            rows = c.fetchall()
            conn.close()

            dump_path = get_dump_path(root)
            with open(dump_path, 'w', encoding='utf-8') as f:
                if not rows:
                    f.write("暂无记忆。\n")
                else:
                    f.write(f"共 {len(rows)} 条记忆：\n")
                    for row in rows:
                        content = row['content']
                        if not parsed.all and len(content) > 50:
                            content = content[:47] + "..."
                        tags_str = row['tags'] if row['tags'] else ""
                        f.write(f"ID: {row['id']}  [{row['created']}] 标签: {tags_str}\n")
                        f.write(f"  内容: {content}\n\n")
            output_lines.append(f"记忆列表已保存到 {DUMP_FILE}，可用 'py utils.py cat {DUMP_FILE}' 查看。")

        elif subcmd == "search":
            parser = argparse.ArgumentParser(prog="py utils.py memory search", add_help=False)
            parser.add_argument("keyword", nargs="?", help="搜索关键词")
            parsed = parser.parse_args(args[1:])
            if not parsed.keyword:
                return _handle_error(subcmd, "必须提供关键词。")

            db_path = get_db_path(root)
            init_db(db_path)
            conn = get_connection(db_path)
            c = conn.cursor()
            pattern = f"%{parsed.keyword}%"
            c.execute(
                "SELECT id, content, tags, created, updated FROM memories WHERE content LIKE ? OR tags LIKE ? ORDER BY created DESC",
                (pattern, pattern)
            )
            rows = c.fetchall()
            conn.close()

            dump_path = get_dump_path(root)
            with open(dump_path, 'w', encoding='utf-8') as f:
                if not rows:
                    f.write(f"未找到包含 '{parsed.keyword}' 的记忆。\n")
                else:
                    f.write(f"找到 {len(rows)} 条相关记忆：\n")
                    for row in rows:
                        tags_str = row['tags'] if row['tags'] else ""
                        f.write(f"ID: {row['id']}  [{row['created']}] 标签: {tags_str}\n")
                        f.write(f"  内容: {row['content']}\n\n")
            output_lines.append(f"搜索结果已保存到 {DUMP_FILE}，可用 'py utils.py cat {DUMP_FILE}' 查看。")

        elif subcmd == "delete":
            parser = argparse.ArgumentParser(prog="py utils.py memory delete", add_help=False)
            parser.add_argument("id", help="要删除的记忆 ID")
            parsed = parser.parse_args(args[1:])

            db_path = get_db_path(root)
            init_db(db_path)
            conn = get_connection(db_path)
            c = conn.cursor()
            c.execute("DELETE FROM memories WHERE id = ?", (parsed.id,))
            deleted = c.rowcount > 0
            conn.commit()
            if not deleted:
                conn.close()
                return _handle_error(subcmd, f"未找到 ID 为 '{parsed.id}' 的记忆。")
            c.execute("SELECT id, content, tags, created, updated FROM memories ORDER BY created DESC")
            memories = c.fetchall()
            conn.close()
            dump_msg = _dump_memories(root, memories)
            output_lines.append(f"记忆 {parsed.id} 已删除。")
            output_lines.append(dump_msg)

        elif subcmd == "update":
            parser = argparse.ArgumentParser(prog="py utils.py memory update", add_help=False)
            parser.add_argument("id", help="要更新的记忆 ID")
            parser.add_argument("--content", help="新内容")
            parser.add_argument("--tags", "-t", help="逗号分隔的新标签")
            parsed = parser.parse_args(args[1:])
            if not parsed.content and parsed.tags is None:
                return _handle_error(subcmd, "至少提供新内容或新标签之一。")

            db_path = get_db_path(root)
            init_db(db_path)
            conn = get_connection(db_path)
            c = conn.cursor()

            # 检查是否存在
            c.execute("SELECT id FROM memories WHERE id = ?", (parsed.id,))
            if not c.fetchone():
                conn.close()
                return _handle_error(subcmd, f"未找到 ID 为 '{parsed.id}' 的记忆。")

            updates = []
            params = []
            if parsed.content is not None:
                updates.append("content = ?")
                params.append(parsed.content)
            if parsed.tags is not None:
                updates.append("tags = ?")
                params.append(",".join(parsed.tags.split(",")))
            updates.append("updated = datetime('now')")
            params.append(parsed.id)

            sql = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"
            c.execute(sql, params)
            conn.commit()

            c.execute("SELECT id, content, tags, created, updated FROM memories ORDER BY created DESC")
            memories = c.fetchall()
            conn.close()
            dump_msg = _dump_memories(root, memories)
            output_lines.append(f"记忆 {parsed.id} 已更新。")
            output_lines.append(dump_msg)

        else:
            return _handle_error(subcmd, f"未知的子命令: {subcmd}")

    except Exception as e:
        return _handle_error(subcmd, str(e))

    return "\n".join(output_lines)
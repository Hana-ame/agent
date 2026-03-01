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
"""

import sys
import os
import sqlite3
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils import core

DB_FILE = "memory.db"  # SQLite数据库文件，位于根目录
DUMP_FILE = "memory_dump.txt"  # 用于输出列表的临时文件，便于查看

def get_db_path(root_path: Path) -> Path:
    return root_path / DB_FILE

def get_dump_path(root_path: Path) -> Path:
    return root_path / DUMP_FILE

def init_db(db_path: Path):
    """初始化数据库表"""
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
    """将查询结果转换为字典"""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def get_connection(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = dict_factory
    return conn

def generate_id() -> str:
    """生成唯一ID（基于时间戳）"""
    return f"{int(time.time()*1000)}"

def dump_memories_to_file(root_path: Path, memories: List[Dict]):
    """将记忆列表写入临时文件，方便查看"""
    dump_path = get_dump_path(root_path)
    with open(dump_path, 'w', encoding='utf-8') as f:
        if not memories:
            f.write("暂无记忆。\n")
            return
        for m in memories:
            f.write(f"ID: {m['id']}  [{m['created']}] 标签: {m['tags']}\n")
            f.write(f"  内容: {m['content']}\n\n")

def add_memory(root_path: Path, content: str, tags: Optional[List[str]] = None):
    """添加一条记忆"""
    db_path = get_db_path(root_path)
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
    # 获取最新列表并写入dump文件
    c.execute("SELECT id, content, tags, created, updated FROM memories ORDER BY created DESC")
    memories = c.fetchall()
    dump_memories_to_file(root_path, memories)
    conn.close()
    print(f"记忆已添加，ID: {mem_id}")
    print(f"当前记忆列表已保存到 {DUMP_FILE}，可用 'py utils.py read {DUMP_FILE}' 查看。")

def list_memory(root_path: Path, tag: Optional[str] = None, show_all: bool = False):
    """列出记忆，可按标签筛选，并将结果写入dump文件"""
    db_path = get_db_path(root_path)
    init_db(db_path)
    conn = get_connection(db_path)
    c = conn.cursor()
    if tag:
        c.execute(
            "SELECT id, content, tags, created, updated FROM memories WHERE tags LIKE ? ORDER BY created DESC",
            (f"%{tag}%",)
        )
    else:
        c.execute("SELECT id, content, tags, created, updated FROM memories ORDER BY created DESC")
    rows = c.fetchall()
    conn.close()

    # 写入dump文件
    dump_path = get_dump_path(root_path)
    with open(dump_path, 'w', encoding='utf-8') as f:
        if not rows:
            f.write("暂无记忆。\n")
        else:
            f.write(f"共 {len(rows)} 条记忆：\n")
            for row in rows:
                content = row['content']
                if not show_all and len(content) > 50:
                    content = content[:47] + "..."
                tags_str = row['tags'] if row['tags'] else ""
                f.write(f"ID: {row['id']}  [{row['created']}] 标签: {tags_str}\n")
                f.write(f"  内容: {content}\n\n")

    print(f"记忆列表已保存到 {DUMP_FILE}，可用 'py utils.py read {DUMP_FILE}' 查看。")

def search_memory(root_path: Path, keyword: str):
    """模糊搜索内容或标签中的关键词，并将结果写入dump文件"""
    db_path = get_db_path(root_path)
    init_db(db_path)
    conn = get_connection(db_path)
    c = conn.cursor()
    pattern = f"%{keyword}%"
    c.execute(
        "SELECT id, content, tags, created, updated FROM memories WHERE content LIKE ? OR tags LIKE ? ORDER BY created DESC",
        (pattern, pattern)
    )
    rows = c.fetchall()
    conn.close()

    dump_path = get_dump_path(root_path)
    with open(dump_path, 'w', encoding='utf-8') as f:
        if not rows:
            f.write(f"未找到包含 '{keyword}' 的记忆。\n")
        else:
            f.write(f"找到 {len(rows)} 条相关记忆：\n")
            for row in rows:
                tags_str = row['tags'] if row['tags'] else ""
                f.write(f"ID: {row['id']}  [{row['created']}] 标签: {tags_str}\n")
                f.write(f"  内容: {row['content']}\n\n")

    print(f"搜索结果已保存到 {DUMP_FILE}，可用 'py utils.py read {DUMP_FILE}' 查看。")

def delete_memory(root_path: Path, mem_id: str):
    """删除指定ID的记忆"""
    db_path = get_db_path(root_path)
    init_db(db_path)
    conn = get_connection(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
    deleted = c.rowcount > 0
    conn.commit()
    if deleted:
        # 获取最新列表并写入dump文件
        c.execute("SELECT id, content, tags, created, updated FROM memories ORDER BY created DESC")
        memories = c.fetchall()
        dump_memories_to_file(root_path, memories)
        conn.close()
        print(f"记忆 {mem_id} 已删除。")
        print(f"当前记忆列表已保存到 {DUMP_FILE}，可用 'py utils.py read {DUMP_FILE}' 查看。")
    else:
        conn.close()
        print(f"未找到 ID 为 '{mem_id}' 的记忆。")

def update_memory(root_path: Path, mem_id: str, content: Optional[str] = None, tags: Optional[List[str]] = None):
    """更新记忆的内容和/或标签"""
    db_path = get_db_path(root_path)
    init_db(db_path)
    conn = get_connection(db_path)
    c = conn.cursor()

    # 先检查是否存在
    c.execute("SELECT id FROM memories WHERE id = ?", (mem_id,))
    if not c.fetchone():
        print(f"未找到 ID 为 '{mem_id}' 的记忆。")
        conn.close()
        return

    updates = []
    params = []
    if content is not None:
        updates.append("content = ?")
        params.append(content)
    if tags is not None:
        updates.append("tags = ?")
        params.append(",".join(tags))

    if not updates:
        print("错误：至少提供新内容或新标签之一。")
        conn.close()
        return

    updates.append("updated = datetime('now')")
    params.append(mem_id)

    sql = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"
    c.execute(sql, params)
    conn.commit()

    # 获取最新列表并写入dump文件
    c.execute("SELECT id, content, tags, created, updated FROM memories ORDER BY created DESC")
    memories = c.fetchall()
    dump_memories_to_file(root_path, memories)
    conn.close()
    print(f"记忆 {mem_id} 已更新。")
    print(f"当前记忆列表已保存到 {DUMP_FILE}，可用 'py utils.py read {DUMP_FILE}' 查看。")

def run(ctx: core.Context, args: List[str]):
    """命令行入口"""
    if not args:
        print(__doc__.strip())
        return 1

    subcmd = args[0].lower()
    # 将 root_path 转换为 Path 对象，避免字符串除法错误
    root = Path(ctx.root_path)

    if subcmd == "add":
        parser = argparse.ArgumentParser(prog="py utils.py memory add", add_help=False)
        parser.add_argument("content", nargs="?", help="记忆内容")
        parser.add_argument("--tags", "-t", help="逗号分隔的标签")
        try:
            parsed = parser.parse_args(args[1:])
        except SystemExit:
            return 1
        if not parsed.content:
            print("错误：必须提供内容。")
            return 1
        tags = parsed.tags.split(",") if parsed.tags else []
        add_memory(root, parsed.content, tags)

    elif subcmd == "list":
        parser = argparse.ArgumentParser(prog="py utils.py memory list", add_help=False)
        parser.add_argument("--tag", help="按标签筛选")
        parser.add_argument("--all", action="store_true", help="显示完整内容")
        parsed = parser.parse_args(args[1:])
        list_memory(root, parsed.tag, parsed.all)

    elif subcmd == "search":
        parser = argparse.ArgumentParser(prog="py utils.py memory search", add_help=False)
        parser.add_argument("keyword", nargs="?", help="搜索关键词")
        parsed = parser.parse_args(args[1:])
        if not parsed.keyword:
            print("错误：必须提供关键词。")
            return 1
        search_memory(root, parsed.keyword)

    elif subcmd == "delete":
        parser = argparse.ArgumentParser(prog="py utils.py memory delete", add_help=False)
        parser.add_argument("id", help="要删除的记忆 ID")
        parsed = parser.parse_args(args[1:])
        delete_memory(root, parsed.id)

    elif subcmd == "update":
        parser = argparse.ArgumentParser(prog="py utils.py memory update", add_help=False)
        parser.add_argument("id", help="要更新的记忆 ID")
        parser.add_argument("--content", help="新内容")
        parser.add_argument("--tags", "-t", help="逗号分隔的新标签")
        parsed = parser.parse_args(args[1:])
        if not parsed.content and parsed.tags is None:
            print("错误：至少提供新内容或新标签之一。")
            return 1
        tags = parsed.tags.split(",") if parsed.tags else None
        update_memory(root, parsed.id, parsed.content, tags)

    else:
        print(f"未知的子命令: {subcmd}")
        print(__doc__.strip())
        return 1

    return 0
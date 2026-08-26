import json
import random
import sqlite3
import threading
from itertools import chain, combinations
from pathlib import Path

from opencode import run as opencode_run

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "prompt_lab.db"


# ── DB ──────────────────────────────────────────────────────────────────

class PromptDB:
    """SQLite 封装。"""

    def __init__(self, db_path=DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent       TEXT    NOT NULL DEFAULT '',
                    model       TEXT    NOT NULL DEFAULT '',
                    prompt_ids  TEXT    NOT NULL DEFAULT '[]',
                    prompt_text TEXT    NOT NULL DEFAULT '',
                    full_prompt TEXT    NOT NULL DEFAULT '',
                    response    TEXT    NOT NULL DEFAULT '',
                    status      TEXT    NOT NULL DEFAULT 'pending',
                    reply_rate  REAL    NOT NULL DEFAULT 0.0
                )
            """)
            conn.commit()

    def add_prompt(self, agent, model, prompt_ids, prompt_text, full_prompt):
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "INSERT INTO prompts (agent, model, prompt_ids, prompt_text, full_prompt, response, status, reply_rate) "
                    "VALUES (?, ?, ?, ?, ?, '', 'pending', 0.0)",
                    (agent, model, json.dumps(prompt_ids), prompt_text, full_prompt),
                )
                conn.commit()
                return cur.lastrowid

    def update_result(self, pid, response, status):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE prompts SET response=?, status=? WHERE id=?",
                    (response, status, pid),
                )
                conn.commit()

    def update_reply_rate(self, pid, reply_rate):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE prompts SET reply_rate=? WHERE id=?",
                    (reply_rate, pid),
                )
                conn.commit()

    def get_succeed_prompts(self):
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM prompts WHERE status='succeed' ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_prompt_by_id(self, pid):
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM prompts WHERE id=?", (pid,)).fetchone()
        return dict(row) if row else None

    def get_tried_combinations(self):
        """返回已尝试过的 prompt_id 组合集合（frozenset）。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT prompt_ids FROM prompts WHERE prompt_ids != '[]'"
            ).fetchall()
        tried = set()
        for r in rows:
            ids = json.loads(r["prompt_ids"])
            if ids:
                tried.add(frozenset(ids))
        return tried

    def get_all(self):
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM prompts ORDER BY id").fetchall()
        return [dict(r) for r in rows]


# ── 组合工具 ─────────────────────────────────────────────────────────────

def powerset_nonempty(iterable):
    """返回非空幂集（排除空集）。"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def find_untried_combinations(succeed_ids, tried):
    """枚举 succeed_ids 的所有非空组合，返回未尝试过的 list of frozenset。"""
    untried = []
    for combo in powerset_nonempty(succeed_ids):
        fs = frozenset(combo)
        if fs not in tried:
            untried.append(fs)
    return untried


# ── Prompt 构建 ──────────────────────────────────────────────────────────

def build_full_prompt(prompt_text, ref_responses):
    """拼接原始文本 + 引用 response。"""
    parts = [prompt_text]
    if ref_responses:
        parts.append("\n--- 以下为历史参考回复 ---")
        for pid, resp in ref_responses.items():
            parts.append(f"\n[参考 #{pid}]\n{resp}")
    return "\n".join(parts)


# ── Agent 调用 ───────────────────────────────────────────────────────────

def call_agent(agent, model, full_prompt, timeout=600):
    """调用 opencode，返回 (response_text, success)。"""
    try:
        result = opencode_run(full_prompt, agent=agent, model=model, timeout=timeout)
        out = result["output"]
        if isinstance(out, dict):
            text = json.dumps(out, indent=2, ensure_ascii=False)
        else:
            text = str(out)
        return text, result["success"]
    except Exception as e:
        return f"Error: {e}", False


# ── 评判 Agent ───────────────────────────────────────────────────────────

JUDGE_PROMPT = """你是一个回复质量评判专家。请对以下 AI 回复进行评分。

评分标准（0.0 ~ 1.0）：
- 1.0: 完美，完全解决问题，信息准确完整
- 0.7-0.9: 良好，基本解决问题，有些小瑕疵
- 0.4-0.6: 一般，部分解决问题，有明显不足
- 0.1-0.3: 差，几乎没有解决问题
- 0.0: 完全无用

原始问题：
{original}

AI 回复：
{response}

请只输出一个 0.0~1.0 之间的浮点数，不要任何解释。
评分："""


def judge_response(original_text, response_text, judge_model="siliconflow-cn/Qwen/Qwen3-8B"):
    """评判回复质量，返回 0.0~1.0 的分数。"""
    prompt = JUDGE_PROMPT.format(original=original_text, response=response_text)
    try:
        result = opencode_run(prompt, agent="Null", model=judge_model, timeout=60)
        out = result["output"]
        if isinstance(out, dict):
            out = str(out)
        # 提取浮点数
        text = str(out).strip()
        import re
        m = re.search(r"[\d.]+(?:[eE][+-]?\d+)?", text)
        if m:
            rate = float(m.group())
            return max(0.0, min(1.0, rate))
        return 0.0
    except Exception as e:
        print(f"  [评判失败] {e}")
        return 0.0


# ── 显示 ─────────────────────────────────────────────────────────────────

def print_table(db):
    """打印表格。"""
    rows = db.get_all()
    if not rows:
        print("（表为空）")
        return
    print(f"\n{'id':<5} {'agent':<20} {'model':<30} {'prompt_ids':<15} {'status':<10} {'rate':<6} {'response 前80'}")
    print("-" * 120)
    for r in rows:
        resp_preview = (r["response"] or "")[:80].replace("\n", " ")
        print(f"{r['id']:<5} {r['agent']:<20} {r['model']:<30} {str(r['prompt_ids']):<15} {r['status']:<10} {r['reply_rate']:<6.2f} {resp_preview}")


# ── 主循环 ───────────────────────────────────────────────────────────────

DEFAULT_AGENT = "Auto666"
DEFAULT_MODEL = "deepseek-v4-flash-free"
JUDGE_MODEL = "siliconflow-cn/Qwen/Qwen3-8B"


def main():
    db = PromptDB()

    print("=" * 60)
    print("  Prompt 组合探索系统")
    print("=" * 60)
    print()
    print("命令:")
    print("  a <prompt>  — 添加新 prompt（根 prompt，不引用其他）")
    print("  r           — 自动随机选未尝试组合并执行")
    print("  l           — 列出所有记录")
    print("  s <id>      — 查看某条记录的详细 response")
    print("  j <id>      — 重新评判某条记录的 reply_rate")
    print("  q           — 退出")
    print()

    while True:
        try:
            cmd = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not cmd:
            continue

        # ── quit ──
        if cmd == "q":
            print("退出。")
            break

        # ── list ──
        if cmd == "l":
            print_table(db)
            continue

        # ── show detail ──
        if cmd.startswith("s "):
            try:
                pid = int(cmd.split()[1])
            except (IndexError, ValueError):
                print("用法: s <id>")
                continue
            r = db.get_prompt_by_id(pid)
            if r:
                print(f"\n--- #{pid} response ---")
                print(r["response"])
            else:
                print(f"id={pid} 不存在")
            continue

        # ── re-judge ──
        if cmd.startswith("j "):
            try:
                pid = int(cmd.split()[1])
            except (IndexError, ValueError):
                print("用法: j <id>")
                continue
            r = db.get_prompt_by_id(pid)
            if not r:
                print(f"id={pid} 不存在")
                continue
            if not r["response"]:
                print(f"id={pid} 没有 response")
                continue
            print(f"评判 #{pid} ...")
            rate = judge_response(r["prompt_text"], r["response"], JUDGE_MODEL)
            db.update_reply_rate(pid, rate)
            print(f"  reply_rate = {rate:.4f}")
            continue

        # ── add root prompt ──
        if cmd.startswith("a "):
            prompt_text = cmd[2:].strip()
            pid = db.add_prompt(DEFAULT_AGENT, DEFAULT_MODEL, [], prompt_text, prompt_text)
            print(f"添加根 prompt #{pid}: {prompt_text[:60]}...")
            print(f"执行 #{pid} ...")
            response, ok = call_agent(DEFAULT_AGENT, DEFAULT_MODEL, prompt_text)
            status = "succeed" if ok else "failed"
            db.update_result(pid, response, status)
            if ok:
                print("  ✅ succeed")
                rate = judge_response(prompt_text, response, JUDGE_MODEL)
                db.update_reply_rate(pid, rate)
                print(f"  reply_rate = {rate:.4f}")
            else:
                print("  ❌ failed")
            continue

        # ── run random untried ──
        if cmd == "r":
            succeed = db.get_succeed_prompts()
            if len(succeed) < 1:
                print("至少需要 1 条 succeed 记录才能组合。先用 a 添加根 prompt。")
                continue

            succeed_ids = [p["id"] for p in succeed]
            tried = db.get_tried_combinations()
            untried = find_untried_combinations(succeed_ids, tried)

            print(f"succeed 记录: {len(succeed)} 条")
            print(f"已尝试组合: {len(tried)} 种")
            print(f"未尝试组合: {len(untried)} 种")

            if not untried:
                print("所有组合均已尝试！")
                continue

            # 随机选一个未尝试组合
            chosen = random.choice(untried)
            chosen_list = sorted(chosen)
            print(f"选中组合: {chosen_list}")

            # 取最后一条 succeed 记录的 prompt_text 作为基础
            base_text = succeed[-1]["prompt_text"]

            # 构建引用 response
            ref_responses = {}
            for pid in chosen_list:
                pr = db.get_prompt_by_id(pid)
                if pr:
                    ref_responses[pid] = pr["response"]

            full_prompt = build_full_prompt(base_text, ref_responses)
            pid = db.add_prompt(DEFAULT_AGENT, DEFAULT_MODEL, chosen_list, base_text, full_prompt)
            print(f"添加 #{pid}: prompt_ids={chosen_list}")
            print(f"执行 #{pid} ...")

            response, ok = call_agent(DEFAULT_AGENT, DEFAULT_MODEL, full_prompt)
            status = "succeed" if ok else "failed"
            db.update_result(pid, response, status)

            if ok:
                print("  ✅ succeed")
                print(f"评判 #{pid} ...")
                rate = judge_response(base_text, response, JUDGE_MODEL)
                db.update_reply_rate(pid, rate)
                print(f"  reply_rate = {rate:.4f}")
            else:
                print("  ❌ failed")
            continue

        print(f"未知命令: {cmd}")


if __name__ == "__main__":
    main()

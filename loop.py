import json
import subprocess
import sys
import time
import logging
from pathlib import Path
from opencode import Opencode

BASE_DIR = Path(__file__).parent
oc = Opencode()
RETRY_FILE = BASE_DIR / ".task_retries"

# --- 日志配置 ---
_handlers = [logging.FileHandler(BASE_DIR / "loop.log", encoding="utf-8")]
if sys.stderr.isatty():
    _handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=_handlers,
)
logger = logging.getLogger("loop")


def get_actionable():
    """返回 (code/instruction) 类型的未处理帖子列表，无任务时返回 []。"""
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "check_pending_prompts.py")],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode not in (0, 1):
            logger.warning("check_pending_prompts exited %d: %s",
                           result.returncode, result.stderr[:200])
            return []
        data = json.loads(result.stdout)
        actionable = [
            p for p in data.get("pending", [])
            if p.get("type") in ("code", "instruction")
        ]
        logger.info("预检查: total=%s pending=%s actionable=%s",
                    data.get("total_posts", "?"),
                    len(data.get("pending", [])),
                    len(actionable))
        for p in actionable:
            logger.info("  · no=%s [%s] %s", p["no"], p["type"], p["txt"][:80])
        return actionable
    except Exception as e:
        logger.exception("预检查失败: %s", e)
        return []


MAX_RETRIES = 3


def _load_retries():
    """加载重试记录，返回 {no: count} 字典。兼容旧格式（list → set）。"""
    if RETRY_FILE.is_file():
        try:
            data = json.loads(RETRY_FILE.read_text())
            if isinstance(data, list):
                # 兼容旧格式：list → {no: 1}
                return {str(no): 1 for no in data}
            return data
        except Exception:
            return {}
    return {}


def _save_retries(retries):
    RETRY_FILE.write_text(json.dumps(retries))


def run_once(task):
    """处理单个任务。task 是 check_pending_prompts 输出的 dict。"""
    no = task["no"]
    logger.info("处理任务 no=%s: %s", no, task["txt"][:80])
    (BASE_DIR / ".last_update").touch()

    prompt = (
        f"你的任务：处理 Board 666 帖子 no={no}。\n\n"
        f"帖子内容：\n\"\"\"\n{task['txt']}\n\"\"\"\n\n"
        f"线程ID: {task['thread_id']}\n"
        f"类型: {task['type']}\n\n"
        "重要规则：\n"
        "1. 只处理这一个帖子，其他帖子一律忽略\n"
        "2. 不需要运行 check_pending_prompts.py——任务已明确指定\n"
        "3. 严格遵循工作流：读需求 → Checklist → 执行 → 验证Checklist → 生成报告 → 上传 → commit/push → 回复摘要链接\n"
        "4. 每条Checklist通过后立即标记 [x]，不得留 [ ]\n"
        "5. 必须回复帖子才算任务完成，否则视为失败\n"
        "6. 遇到权限不足或无法完成的情况，回复帖子说明原因后标记为完成\n"
        "7. 回复使用 moonchan.py reply 命令，昵称用 Auto666\n"
        "8. 执行完成后，生成详细 Markdown 报告上传至 upload.moonchan.xyz，再回复摘要+链接"
    )
    logger.debug(f"发送 Prompt: {prompt}")

    try:
        start_time = time.time()
        result = oc.run_prompt(prompt, agent="Auto666")
        duration = time.time() - start_time

        logger.info(f"Auto666 响应成功 (耗时 {duration:.2f}s)")
        logger.debug(f"完整响应结果:\n{result}")
        return result
    except Exception as e:
        logger.exception(f"执行出错: {e}")
        raise


def main():
    logger.info("启动 Board 666 监听服务...")
    while True:
        try:
            tasks = get_actionable()
            retries = _load_retries()

            # 清理已经不存在的任务记录
            current_nos = {str(t["no"]) for t in tasks}
            stale = [no for no in retries if no not in current_nos]
            for no in stale:
                del retries[no]
                logger.info("任务 no=%s 已不在列表中，清除重试记录", no)

            # 每个任务最多重试 MAX_RETRIES 次
            fresh = [t for t in tasks if retries.get(str(t["no"]), 0) < MAX_RETRIES]
            abandoned = [t for t in tasks if retries.get(str(t["no"]), 0) >= MAX_RETRIES]

            if abandoned:
                logger.warning("放弃 %d 个任务（超过%d次重试）: %s",
                               len(abandoned), MAX_RETRIES,
                               [t["no"] for t in abandoned])

            if not fresh:
                if not tasks:
                    if retries:
                        _save_retries({})
                    logger.info("无未处理任务")
                else:
                    logger.info("所有 %d 个任务均已超过重试上限，跳过", len(tasks))
            else:
                task = fresh[0]
                no_str = str(task["no"])
                try:
                    run_once(task)
                except Exception:
                    pass
                retries[no_str] = retries.get(no_str, 0) + 1
                logger.info("任务 no=%s 第 %d/%d 次尝试",
                            no_str, retries[no_str], MAX_RETRIES)
                _save_retries(retries)

        except Exception:
            pass
        time.sleep(60)


if __name__ == "__main__":
    main()

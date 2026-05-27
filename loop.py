import json
import subprocess
import sys
import time
import logging
from pathlib import Path
from opencode import Opencode

BASE_DIR = Path(__file__).parent
oc = Opencode()

# --- 日志配置 ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / "loop.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
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


def run_once(task):
    """处理单个任务。task 是 check_pending_prompts 输出的 dict。"""
    logger.info("处理任务 no=%s: %s", task["no"], task["txt"][:80])
    (BASE_DIR / ".last_update").touch()

    prompt = (
        f"你的任务：处理 Board 666 帖子 no={task['no']}。\n\n"
        f"帖子内容：\n\"\"\"\n{task['txt']}\n\"\"\"\n\n"
        f"线程ID: {task['thread_id']}\n"
        f"类型: {task['type']}\n\n"
        "重要规则：\n"
        "1. 只处理这一个帖子，其他帖子一律忽略\n"
        "2. 不需要运行 check_pending_prompts.py——任务已明确指定\n"
        "3. 严格遵循工作流：读需求 → Checklist → 执行 → 验证Checklist → commit → push → 回复帖子\n"
        "4. 每条Checklist通过后立即标记 [x]，不得留 [ ]\n"
        "5. 回复帖子使用 moonchan.py reply 命令"
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
            if tasks:
                # 每轮只处理第一个（最新的）actionable 任务
                run_once(tasks[0])
            else:
                logger.info("无未处理任务，跳过 opencode 调用")
        except Exception:
            pass
        time.sleep(60)


if __name__ == "__main__":
    main()

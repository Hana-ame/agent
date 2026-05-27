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


def has_pending_tasks():
    """运行 check_pending_prompts.py，只关注 code / instruction 类型的未处理帖子。"""
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "check_pending_prompts.py")],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode not in (0, 1):
            logger.warning("check_pending_prompts exited with %d: %s",
                           result.returncode, result.stderr[:200])
            return False
        data = json.loads(result.stdout)
        actionable = [
            p for p in data.get("pending", [])
            if p.get("type") in ("code", "instruction")
        ]
        has = len(actionable) > 0
        logger.info("预检查: total=%s pending=%s actionable=%s → trigger=%s",
                    data.get("total_posts", "?"),
                    len(data.get("pending", [])),
                    len(actionable), has)
        if has:
            for p in actionable:
                logger.info("  · no=%s [%s] %s", p["no"], p["type"], p["txt"][:80])
        return has
    except Exception as e:
        logger.exception("预检查失败: %s", e)
        return False


def run_once():
    logger.info("触发 Auto666 检查 Board 666...")
    (BASE_DIR / ".last_update").touch()

    prompt = "检查 Board 666 的最新帖子，找出未处理的需求，按照帖子要求处理，回复结果。"
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
            if has_pending_tasks():
                run_once()
            else:
                logger.info("无未处理任务，跳过 opencode 调用")
        except Exception:
            pass  # 异常已在 run_once 中记录
        time.sleep(60)


if __name__ == "__main__":
    main()

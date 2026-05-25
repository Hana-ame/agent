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

def run_once():
    logger.info("触发 Auto666 检查 Board 666...")
    (BASE_DIR / ".last_update").touch()
    
    prompt = "检查 Board 666 的最新帖子，找出未处理的需求，将代码需求写成 Python 脚本并执行，回复结果。"
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
            run_once()
        except Exception:
            pass # 异常已在 run_once 中记录
        time.sleep(60)

if __name__ == "__main__":
    main()

#!/bin/bash
# restart_loop666.sh — Kill 旧 loop666 进程，启动新 loop666
# Usage: bash restart_loop666.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "[restart_loop666] 查找并终止旧 loop666 进程..."

# 杀掉所有 loop666.py 进程
pids=$(pgrep -f "loop666.py" 2>/dev/null)
if [ -n "$pids" ]; then
    echo "[restart_loop666] 发现进程 PID: $pids，发送 SIGTERM..."
    kill $pids 2>/dev/null
    sleep 2
    # 强制 kill 还在跑的
    pids=$(pgrep -f "loop666.py" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "[restart_loop666] 强制 kill..."
        kill -9 $pids 2>/dev/null
    fi
    echo "[restart_loop666] 旧进程已终止。"
else
    echo "[restart_loop666] 未发现运行中的 loop666 进程。"
fi

echo "[restart_loop666] 启动新 loop666 (nohup)..."

nohup python3 loop666.py >> loop666.log 2>&1 &

echo "[restart_loop666] 新 loop666 已启动，PID: $!"
echo "[restart_loop666] 日志输出: loop666.log"

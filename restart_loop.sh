#!/bin/bash
# restart_loop.sh — 检查 .last_update，超1小时则 kill 并重启 loop.py
# Usage: bash restart_loop.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

LAST_UPDATE_FILE=".last_update"

if [ -f "$LAST_UPDATE_FILE" ]; then
    last_mtime=$(stat -c %Y "$LAST_UPDATE_FILE")
    now=$(date +%s)
    age=$((now - last_mtime))

    if [ "$age" -lt 1800 ]; then
        echo "[restart_loop] .last_update 距今 ${age}s，不足半小时，无需重启"
        exit 0
    fi
    echo "[restart_loop] .last_update 距今 ${age}s，超过1小时，准备重启"
else
    echo "[restart_loop] .last_update 不存在，准备重启"
fi

echo "[restart_loop] 查找并终止 loop.py 进程..."
pids=$(pgrep -f "loop\.py" 2>/dev/null)
if [ -n "$pids" ]; then
    echo "[restart_loop] 发现 PID: $pids，发送 SIGTERM..."
    kill $pids 2>/dev/null
    sleep 2
    pids=$(pgrep -f "loop\.py" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "[restart_loop] 强制 kill..."
        kill -9 $pids 2>/dev/null
    fi
    echo "[restart_loop] 旧进程已终止"
else
    echo "[restart_loop] 未发现运行中的 loop.py 进程"
fi

echo "[restart_loop] 启动新 loop.py (nohup)..."
nohup python3 loop.py > loop.log 2>&1 &

echo "[restart_loop] 新 loop.py 已启动，PID: $!"
echo "[restart_loop] 日志输出: loop.log"

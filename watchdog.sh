#!/bin/bash
# watchdog.sh — 确保 gemini-proxy 运行，然后每隔1分钟跑一次 restart_loop.sh

PROXY_DIR="$(dirname "$0")/gemini-proxy"

# 检查并启动 gemini-proxy
if ! curl -sf --noproxy "localhost,127.0.0.1,::1" http://localhost:8317/health >/dev/null 2>&1; then
    echo "[watchdog] proxy 未运行，启动中..."
    cd "$PROXY_DIR"
    python server.py &
    sleep 2
    if curl -sf --noproxy "localhost,127.0.0.1,::1" http://localhost:8317/health >/dev/null 2>&1; then
        echo "[watchdog] proxy 启动成功"
    else
        echo "[watchdog] proxy 启动失败!"
    fi
    cd - >/dev/null
fi

while true; do
    bash "$(dirname "$0")/restart_loop.sh"
    sleep 60
done

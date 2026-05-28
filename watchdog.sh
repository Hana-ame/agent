#!/bin/bash
# watchdog.sh — 守护 gemini-proxy、loop666.py、loop.py 三个进程
# loop666.py 内部会检查并拉起 loop.py

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROXY_DIR="$SCRIPT_DIR/gemini-proxy"

# ── 检查并启动 gemini-proxy ──
if ! curl -sf --noproxy "localhost,127.0.0.1,::1" http://localhost:8317/health >/dev/null 2>&1; then
    echo "[watchdog] proxy 未运行，启动中..."
    cd "$PROXY_DIR"
    python3 server.py &
    sleep 2
    if curl -sf --noproxy "localhost,127.0.0.1,::1" http://localhost:8317/health >/dev/null 2>&1; then
        echo "[watchdog] proxy 启动成功"
    else
        echo "[watchdog] proxy 启动失败!"
    fi
    cd "$SCRIPT_DIR"
fi

# ── 检查并启动 loop666.py ──
if ! pgrep -f "loop666.py" >/dev/null 2>&1; then
    echo "[watchdog] loop666.py 未运行，启动中..."
    cd "$SCRIPT_DIR"
    nohup python3 loop666.py >> loop666.log 2>&1 &
    echo "[watchdog] loop666.py 已启动，PID: $!"
fi

# ── 主循环：定期检查进程存活 ──
while true; do
    sleep 60

    # 检查 loop666.py 是否存活
    if ! pgrep -f "loop666.py" >/dev/null 2>&1; then
        echo "[watchdog] loop666.py 已退出，重新启动..."
        cd "$SCRIPT_DIR"
        nohup python3 loop666.py >> loop666.log 2>&1 &
        echo "[watchdog] loop666.py 已启动，PID: $!"
    fi

    # 检查 loop.py 是否存活（loop666 也会检查，这里做双重保险）
    if ! pgrep -f "loop\.py" >/dev/null 2>&1; then
        echo "[watchdog] loop.py 未运行，通过 restart_loop.sh 启动..."
        bash "$SCRIPT_DIR/restart_loop.sh"
    fi
done

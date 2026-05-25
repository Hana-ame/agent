#!/bin/bash
# watchdog.sh — 每隔1分钟跑一次 restart_loop.sh

while true; do
    bash "$(dirname "$0")/restart_loop.sh"
    sleep 60
done

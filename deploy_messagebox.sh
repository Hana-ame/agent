#!/bin/bash
# deploy_messagebox.sh

SERVER="root@bwh.moonchan.xyz"
PORT="26275"
FILE="messagebox_service.py"

echo "[Deploy] Copying $FILE to $SERVER..."
scp -P $PORT $FILE $SERVER:/root/messagebox_service.py

echo "[Deploy] Installing dependencies on server..."
ssh -p $PORT $SERVER "python3 -m venv /root/mb_venv && /root/mb_venv/bin/pip install fastapi uvicorn sse-starlette"

echo "[Deploy] Starting service in background..."
ssh -p $PORT $SERVER "nohup /root/mb_venv/bin/python3 /root/messagebox_service.py > /root/messagebox.log 2>&1 &"

echo "[Deploy] Verifying service..."
sleep 5
ssh -p $PORT $SERVER "curl -x \"\" http://127.0.0.1:8000/api/testchan/message -H 'Content-Type: application/json' -d '{\"content\":\"deploy test\"}'"

echo "[Deploy] Done."

@echo off
rem TaskPad launcher (detached via Task Scheduler).
rem Code page stays default: this file must remain pure ASCII.
cd /d "C:\Users\lumin\Downloads\agent\taskpad"
"C:\Users\lumin\.workbuddy\binaries\node\versions\22.22.2-6\node.exe" "C:\Users\lumin\Downloads\agent\taskpad\server.js" --port 8335 --log "C:\Users\lumin\Downloads\agent\taskpad\data\server-8335.log"

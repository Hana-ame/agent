#!/bin/bash
cd "$(dirname "$0")"
pip install -q -r requirements.txt
exec python server.py

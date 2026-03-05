#!/bin/bash
cd /Users/panda/Projects/japan-momentum

# FastAPI起動
/usr/bin/python3 -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 &
FASTAPI_PID=$!
sleep 2

# Named Tunnel起動（固定URL: jpmomentum.com）
cloudflared tunnel --config /Users/panda/Projects/japan-momentum/infra/tunnel.yml run

kill $FASTAPI_PID

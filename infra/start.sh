#!/bin/bash
cd /Users/panda/Projects/japan-momentum

# FastAPI起動（バックグラウンド）
/usr/bin/python3 dashboard/app.py &
FASTAPI_PID=$!
echo "FastAPI PID: $FASTAPI_PID"

# 少し待ってから起動確認
sleep 2
curl -s http://localhost:8080/api/today > /dev/null && echo "FastAPI: OK" || echo "FastAPI: 起動失敗"

# Cloudflare Tunnel起動
cloudflared tunnel run --config /Users/panda/Projects/japan-momentum/infra/tunnel.yml

# Tunnel終了したらFastAPIも止める
kill $FASTAPI_PID

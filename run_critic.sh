#!/bin/bash
# run_critic.sh - System Critic 実行 + Discord通知
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs improvements

echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
PYTHONUNBUFFERED=1 python3 system_critic.py >> logs/critic.log 2>&1

# 通知ファイルがあればOpenClaw経由で送信
NOTIFY_FILE="improvements/.pending_notify.json"
if [ -f "$NOTIFY_FILE" ]; then
    echo "pending_critic_notify" >> logs/critic.log
fi
echo "完了"

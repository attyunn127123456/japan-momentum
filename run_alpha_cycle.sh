#!/bin/bash
# 1サイクル: 仮説生成 → HF評価（ウォッチドッグ付き）
LOCKFILE="/tmp/alpha_cycle.lock"
LOG="logs/alpha_cycle.log"
cd "$(dirname "$0")"
mkdir -p logs

# 重複起動防止
if [ -f "$LOCKFILE" ]; then
    PID=$(cat "$LOCKFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "$(date '+%H:%M:%S') サイクル実行中 (PID: $PID) スキップ" | tee -a $LOG
        exit 0
    fi
fi
echo $$ > "$LOCKFILE"

echo "=== Alpha Cycle 開始 $(date '+%Y-%m-%d %H:%M') ===" | tee -a $LOG

echo "[1/2] 仮説生成..." | tee -a $LOG
PYTHONUNBUFFERED=1 python3 deep_alpha_engine.py >> $LOG 2>&1
if [ $? -ne 0 ]; then
    echo "仮説生成失敗" | tee -a $LOG
    rm -f "$LOCKFILE"; exit 1
fi

echo "[2/2] HF評価（ウォッチドッグ付き）..." | tee -a $LOG
bash run_evaluate_watchdog.sh
EVAL_CODE=$?

rm -f "$LOCKFILE"

if [ $EVAL_CODE -eq 0 ]; then
    echo "=== 完了 $(date '+%Y-%m-%d %H:%M') ===" | tee -a $LOG
    openclaw system event --text "Alpha Cycle完了: 新仮説生成+HF評価終了。jpmomentum.com を確認" --mode now
else
    echo "=== 評価失敗 $(date '+%Y-%m-%d %H:%M') ===" | tee -a $LOG
fi

#!/bin/bash
# ウォッチドッグ付き評価エンジン実行
# 最大3回リトライ、重複起動防止

LOCKFILE="/tmp/evaluate_hypothesis.lock"
LOG="/Users/panda/Projects/japan-momentum/logs/evaluate.log"
cd /Users/panda/Projects/japan-momentum
mkdir -p logs

# 重複起動防止
if [ -f "$LOCKFILE" ]; then
    EXISTING_PID=$(cat "$LOCKFILE")
    if kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "$(date '+%H:%M:%S') 既に実行中 (PID: $EXISTING_PID)" | tee -a "$LOG"
        exit 0
    fi
fi

MAX_RETRY=3
for i in $(seq 1 $MAX_RETRY); do
    echo "$(date '+%H:%M:%S') 評価エンジン起動 (試行 $i/$MAX_RETRY)" | tee -a "$LOG"
    
    PYTHONUNBUFFERED=1 python3 evaluate_hypothesis.py >> "$LOG" 2>&1 &
    PID=$!
    echo $PID > "$LOCKFILE"
    
    wait $PID
    EXIT_CODE=$?
    rm -f "$LOCKFILE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date '+%H:%M:%S') ✅ 評価完了 (exit 0)" | tee -a "$LOG"
        exit 0
    else
        echo "$(date '+%H:%M:%S') ⚠️ 異常終了 (exit $EXIT_CODE) → リトライ待機10秒" | tee -a "$LOG"
        sleep 10
    fi
done

echo "$(date '+%H:%M:%S') ❌ 3回失敗。ループ停止" | tee -a "$LOG"
exit 1

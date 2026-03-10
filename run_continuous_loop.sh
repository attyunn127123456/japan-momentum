#!/bin/bash
# 自律ループ: サイクル完了→即次のサイクルを起動し続ける
# heartbeatで死活監視される。二重起動防止のlockfile付き。

LOOP_LOCK="/tmp/alpha_continuous_loop.lock"
LOG="logs/alpha_cycle.log"
cd "$(dirname "$0")"
mkdir -p logs

# 重複起動防止
if [ -f "$LOOP_LOCK" ]; then
    PID=$(cat "$LOOP_LOCK")
    if kill -0 "$PID" 2>/dev/null; then
        echo "$(date '+%H:%M:%S') ループ既に稼働中 (PID: $PID)" | tee -a $LOG
        exit 0
    fi
fi
echo $$ > "$LOOP_LOCK"

echo "$(date '+%Y-%m-%d %H:%M:%S') 🔁 自律ループ開始" | tee -a $LOG

cleanup() {
    rm -f "$LOOP_LOCK"
    echo "$(date '+%H:%M:%S') ループ停止" | tee -a $LOG
    exit 0
}
trap cleanup SIGTERM SIGINT

CYCLE=0
while true; do
    CYCLE=$((CYCLE + 1))
    echo "$(date '+%H:%M:%S') === サイクル #$CYCLE 開始 ===" | tee -a $LOG

    # lockfileクリア
    rm -f /tmp/evaluate_hypothesis.lock /tmp/alpha_cycle.lock

    # 1サイクル実行（仮説生成→評価）
    bash run_alpha_cycle.sh
    EXIT=$?

    if [ $EXIT -eq 0 ]; then
        echo "$(date '+%H:%M:%S') ✅ サイクル #$CYCLE 完了 → 即次へ" | tee -a $LOG
    else
        echo "$(date '+%H:%M:%S') ⚠️ サイクル #$CYCLE 失敗(exit $EXIT) → 30秒待機" | tee -a $LOG
        sleep 30
    fi
done

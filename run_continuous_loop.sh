#!/bin/bash
# 自律ループ v2 — シンプル・確実版
# lockfileを使わず、pgrep で自分自身の重複を防ぐ

LOG="logs/alpha_cycle.log"
cd "$(dirname "$0")"
mkdir -p logs

# 重複起動防止（自分以外に同じスクリプトが動いていたら終了）
RUNNING=$(pgrep -f "run_continuous_loop" | grep -v $$ | wc -l | tr -d ' ')
if [ "$RUNNING" -gt "0" ]; then
    echo "$(date '+%H:%M:%S') 重複起動検知 → 終了" >> $LOG
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') 🔁 自律ループ開始 (PID: $$)" | tee -a $LOG

CYCLE=0
while true; do
    CYCLE=$((CYCLE + 1))
    echo "$(date '+%H:%M:%S') === サイクル #$CYCLE ===" | tee -a $LOG

    # 仮説生成
    echo "$(date '+%H:%M:%S') [1/2] 仮説生成..." | tee -a $LOG
    PYTHONUNBUFFERED=1 python3 deep_alpha_engine.py >> $LOG 2>&1
    if [ $? -ne 0 ]; then
        echo "$(date '+%H:%M:%S') ⚠️ 仮説生成失敗 → 30秒待機" | tee -a $LOG
        sleep 30; continue
    fi

    # 評価（lockfile不使用・pgrep で重複防止）
    if pgrep -f "evaluate_hypothesis.py" > /dev/null; then
        echo "$(date '+%H:%M:%S') 評価中プロセスあり → スキップ" | tee -a $LOG
    else
        echo "$(date '+%H:%M:%S') [2/2] HF評価..." | tee -a $LOG
        PYTHONUNBUFFERED=1 python3 evaluate_hypothesis.py >> $LOG 2>&1
        if [ $? -ne 0 ]; then
            echo "$(date '+%H:%M:%S') ⚠️ 評価失敗 → 継続" | tee -a $LOG
        fi
    fi

    echo "$(date '+%H:%M:%S') ✅ サイクル #$CYCLE 完了 → 即次へ" | tee -a $LOG
done

#!/bin/bash
# run_daily.sh - 日次シグナル生成 + ペーパートレード実行
# OpenClaw cronから月〜金 15:35 JST に呼ばれる
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
echo "Step 1: シグナル生成"
python3 daily_signal_output.py
echo ""

echo "Step 2: ペーパートレード実行"
python3 -c "
from paper_trading_engine import run_paper_trade_from_signals
result = run_paper_trade_from_signals()
if result:
    print('ペーパートレード完了')
else:
    print('ペーパートレード: スキップまたはエラー')
"
echo ""
echo "===== 完了: $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
